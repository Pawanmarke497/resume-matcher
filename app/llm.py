"""LLM service: structured resume parsing, critique, and cover letters.

Two implementations share one interface:
  * LangChainLLMService: real LLM via LangChain (OpenAI or Anthropic), using Pydantic
    structured output.
  * OfflineLLMService: deterministic heuristics / templates, so the whole pipeline
    runs (and is testable) without any API key.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.schemas import Critique, Job, JobMatch, ResumeProfile
from app.skills import extract_skills


class LLMService(Protocol):
    is_offline: bool

    def parse_resume(self, resume_text: str) -> ResumeProfile: ...

    def critique(self, profile: ResumeProfile, job: Job, match: JobMatch) -> Critique: ...

    def cover_letter(self, profile: ResumeProfile, job: Job, match: JobMatch) -> str: ...


# ------------------------------------------------------------------------ prompts
PARSE_SYSTEM = (
    "You extract structured data from resumes. Use only information that is explicitly "
    "present in the resume text. Never invent employers, dates, degrees or skills. "
    "Use null for anything missing. For years_experience, sum the professional roles; "
    "treat 'present' as the current year."
)

CRITIQUE_SYSTEM = (
    "You are a candid, constructive career coach reviewing a resume against one job. "
    "Be specific and honest. Only cite strengths that are evidenced in the resume. "
    "Suggestions must be truthful edits (reordering, quantifying, clarifying), never "
    "advice to claim skills or experience the candidate does not have."
)
CRITIQUE_HUMAN = """Candidate profile (JSON):
{profile}

Target job: {title} at {company}
Job description:
{description}

Computed match: {score}/100
Skills matched: {matched}
Skills missing: {missing}"""

COVER_SYSTEM = (
    "You write concise, professional cover letters (about 250 words, 3-4 short "
    "paragraphs, plain text, no placeholders except the greeting if the hiring manager's "
    "name is unknown). Use ONLY facts from the candidate profile. Do not claim any skill "
    "or experience that is not in the profile. If the candidate lacks a required skill, "
    "do not mention it; emphasise genuine strengths and adjacent experience instead."
)
COVER_HUMAN = """Candidate profile (JSON):
{profile}

Target job: {title} at {company}
Job description:
{description}

Skills the candidate has that the job wants: {matched}"""


# ------------------------------------------------------------------ real LLM service
class LangChainLLMService:
    is_offline = False

    def __init__(self, chat_model, provider: str) -> None:
        # OpenAI's strict json_schema mode rejects optional fields with defaults, so
        # force function-calling there. Anthropic uses tool calling by default.
        kwargs = {"method": "function_calling"} if provider == "openai" else {}
        parse_prompt = ChatPromptTemplate.from_messages(
            [("system", PARSE_SYSTEM), ("human", "Resume:\n\n{resume_text}")]
        )
        critique_prompt = ChatPromptTemplate.from_messages(
            [("system", CRITIQUE_SYSTEM), ("human", CRITIQUE_HUMAN)]
        )
        cover_prompt = ChatPromptTemplate.from_messages(
            [("system", COVER_SYSTEM), ("human", COVER_HUMAN)]
        )
        self._parse = parse_prompt | chat_model.with_structured_output(ResumeProfile, **kwargs)
        self._critique = critique_prompt | chat_model.with_structured_output(Critique, **kwargs)
        self._cover = cover_prompt | chat_model | StrOutputParser()

    def parse_resume(self, resume_text: str) -> ResumeProfile:
        return self._parse.invoke({"resume_text": resume_text[:20000]})

    def critique(self, profile: ResumeProfile, job: Job, match: JobMatch) -> Critique:
        return self._critique.invoke(self._job_vars(profile, job, match))

    def cover_letter(self, profile: ResumeProfile, job: Job, match: JobMatch) -> str:
        return self._cover.invoke(self._job_vars(profile, job, match)).strip()

    @staticmethod
    def _job_vars(profile: ResumeProfile, job: Job, match: JobMatch) -> dict:
        return {
            "profile": profile.model_dump_json(indent=1),
            "title": job.title,
            "company": job.company,
            "description": job.description,
            "score": match.score,
            "matched": ", ".join(match.matched_skills) or "none",
            "missing": ", ".join(match.missing_skills) or "none",
        }


# ------------------------------------------------------------------ offline service
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
_YEARS = re.compile(r"(\d{1,2})\+?\s*(?:years|yrs)", re.IGNORECASE)
_RANGE = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)\b",
    re.IGNORECASE,
)


class OfflineLLMService:
    is_offline = True

    def parse_resume(self, resume_text: str) -> ResumeProfile:
        lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
        name = None
        if lines and len(lines[0].split()) <= 5 and not _EMAIL.search(lines[0]):
            name = lines[0]
        email = _EMAIL.search(resume_text)
        phone = _PHONE.search(resume_text)

        this_year = date.today().year
        span = 0
        for start, end in _RANGE.findall(resume_text):
            end_year = this_year if end.lower() in {"present", "current", "now"} else int(end)
            span += max(0, end_year - int(start))
        stated = [int(m) for m in _YEARS.findall(resume_text)]
        years = float(max([span, *stated])) if (span or stated) else None

        headline = lines[1] if len(lines) > 1 and len(lines[1]) < 90 else None
        return ResumeProfile(
            full_name=name,
            email=email.group(0) if email else None,
            phone=phone.group(0).strip() if phone else None,
            headline=headline,
            summary=None,
            skills=extract_skills(resume_text),
            years_experience=years,
        )

    def critique(self, profile: ResumeProfile, job: Job, match: JobMatch) -> Critique:
        total = len(match.matched_skills) + len(match.missing_skills)
        strengths = []
        if match.matched_skills:
            strengths.append(
                f"Covers {len(match.matched_skills)} of {total} required skills: "
                f"{', '.join(match.matched_skills)}."
            )
        if profile.years_experience:
            strengths.append(f"About {profile.years_experience:g} years of experience listed.")
        gaps = [f"No evidence of {s} in the resume." for s in match.missing_skills]
        suggestions = [
            f"If you have real experience with {s}, add it with a concrete project or "
            "result; if not, a small project would help close the gap."
            for s in match.missing_skills[:3]
        ]
        suggestions.append(
            "Lead each role with measurable outcomes (latency, cost, users, accuracy)."
        )
        suggestions.append(
            f"Mirror the job's wording where it is accurate, e.g. the title '{job.title}'."
        )
        level = "strong" if match.score >= 70 else "moderate" if match.score >= 45 else "limited"
        return Critique(
            fit_summary=(
                f"{level.capitalize()} fit for {job.title} at {job.company} "
                f"(match score {match.score:g}/100)."
            ),
            strengths=strengths or ["No required skills were detected in the resume."],
            gaps=gaps,
            suggestions=suggestions,
        )

    def cover_letter(self, profile: ResumeProfile, job: Job, match: JobMatch) -> str:
        name = profile.full_name or "Candidate"
        skills = ", ".join(match.matched_skills[:5]) or "my technical background"
        years = (
            f" With about {profile.years_experience:g} years of experience,"
            if profile.years_experience
            else ""
        )
        return (
            f"Dear Hiring Manager,\n\n"
            f"I am writing to apply for the {job.title} position at {job.company}. "
            f"The role's focus lines up closely with my hands-on work in {skills}.\n\n"
            f"{years.strip() + ' ' if years else ''}I have built and shipped work that "
            f"relies on these skills, and I am keen to apply them to {job.company}'s "
            f"goals.\n\n"
            f"I would welcome the chance to discuss how I can contribute. Thank you for "
            f"your time and consideration.\n\n"
            f"Sincerely,\n{name}"
        )


# ----------------------------------------------------------------------- factory
def get_llm_service(settings: Settings) -> LLMService:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return LangChainLLMService(ChatOpenAI(model=settings.llm_model, temperature=0), "openai")
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return LangChainLLMService(
            ChatAnthropic(model=settings.llm_model, temperature=0), "anthropic"
        )
    return OfflineLLMService()
