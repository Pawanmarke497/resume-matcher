"""LangGraph pipeline.

    START -> parse_resume -> extract_skills -> match_jobs -> critique -> [cover_letter] -> END
                  |                                |                         ^
                  +-- (bad input) -> END           +-- (no jobs) -> END      +-- only if requested

Each node returns a partial state update. `errors` and `warnings` accumulate across
nodes through the `operator.add` reducer.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.job_index import JobIndex
from app.llm import LLMService
from app.matching import rank_jobs
from app.schemas import Critique, JobMatch, ResumeProfile
from app.skills import dedupe_skills, extract_skills

logger = logging.getLogger(__name__)

MIN_RESUME_CHARS = 50
OFFLINE_WARNING = (
    "Running in offline mode: resume parsing is heuristic and feedback is templated. "
    "Set OPENAI_API_KEY or ANTHROPIC_API_KEY for LLM-quality output."
)


class GraphState(TypedDict, total=False):
    # inputs
    resume_text: str
    top_k: int
    generate_cover_letter: bool
    target_job_id: str | None
    # produced by nodes
    profile: ResumeProfile
    matches: list[JobMatch]
    critique: Critique
    cover_letter: str
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]


def build_graph(llm: LLMService, index: JobIndex, settings: Settings):
    # ------------------------------------------------------------------ nodes
    def parse_resume(state: GraphState) -> dict:
        text = state["resume_text"].strip()
        if len(text) < MIN_RESUME_CHARS:
            return {
                "errors": [
                    "Resume text is empty or too short. If this is a scanned PDF, "
                    "it needs OCR first."
                ]
            }
        try:
            profile = llm.parse_resume(text)
        except Exception:
            logger.exception("Resume parsing failed")
            return {"errors": ["Resume parsing failed. Check server logs."]}
        update: dict = {"profile": profile}
        if llm.is_offline:
            update["warnings"] = [OFFLINE_WARNING]
        return update

    def extract_skills_node(state: GraphState) -> dict:
        """Normalise LLM skills and add any taxonomy skills the LLM missed."""
        profile = state["profile"]
        merged = dedupe_skills([*profile.skills, *extract_skills(state["resume_text"])])
        return {"profile": profile.model_copy(update={"skills": merged})}

    def match_jobs(state: GraphState) -> dict:
        if len(index) == 0:
            return {"matches": [], "warnings": ["The job catalog is empty."]}
        target = state.get("target_job_id")
        matches = rank_jobs(
            state["profile"],
            state["resume_text"],
            index,
            top_k=state.get("top_k", 5),
            semantic_weight=settings.semantic_weight,
            include_ids=(target,) if target else (),
        )
        return {"matches": matches}

    def critique(state: GraphState) -> dict:
        matches = state["matches"]
        target_id = state.get("target_job_id") or matches[0].job_id
        match = next(m for m in matches if m.job_id == target_id)
        job = index.get(target_id)
        try:
            result = llm.critique(state["profile"], job, match)
        except Exception:
            logger.exception("Critique generation failed")
            return {
                "target_job_id": target_id,
                "warnings": ["Feedback generation failed; matches are still valid."],
            }
        return {"target_job_id": target_id, "critique": result}

    def cover_letter(state: GraphState) -> dict:
        target_id = state["target_job_id"]
        match = next(m for m in state["matches"] if m.job_id == target_id)
        try:
            letter = llm.cover_letter(state["profile"], index.get(target_id), match)
        except Exception:
            logger.exception("Cover letter generation failed")
            return {"warnings": ["Cover letter generation failed."]}
        return {"cover_letter": letter}

    # ---------------------------------------------------------------- routing
    def after_parse(state: GraphState) -> str:
        return "extract_skills" if state.get("profile") and not state.get("errors") else "end"

    def after_match(state: GraphState) -> str:
        return "critique" if state.get("matches") else "end"

    def after_critique(state: GraphState) -> str:
        wants_letter = state.get("generate_cover_letter") and state.get("target_job_id")
        return "cover_letter" if wants_letter else "end"

    # ------------------------------------------------------------------ graph
    g = StateGraph(GraphState)
    g.add_node("parse_resume", parse_resume)
    g.add_node("extract_skills", extract_skills_node)
    g.add_node("match_jobs", match_jobs)
    g.add_node("critique", critique)
    g.add_node("cover_letter", cover_letter)

    g.add_edge(START, "parse_resume")
    g.add_conditional_edges(
        "parse_resume", after_parse, {"extract_skills": "extract_skills", "end": END}
    )
    g.add_edge("extract_skills", "match_jobs")
    g.add_conditional_edges("match_jobs", after_match, {"critique": "critique", "end": END})
    g.add_conditional_edges(
        "critique", after_critique, {"cover_letter": "cover_letter", "end": END}
    )
    g.add_edge("cover_letter", END)
    return g.compile()
