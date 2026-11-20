"""Rank jobs for a resume: blend of embedding similarity and required-skill coverage."""

from __future__ import annotations

from app.job_index import JobIndex
from app.schemas import Job, JobMatch, ResumeProfile
from app.skills import canonicalize

# Cosine similarity between a resume and a relevant job description rarely exceeds
# ~0.75, so rescale so that "very similar" lands near 1.0 instead of ~0.6.
SEMANTIC_CEILING = 0.75
MAX_QUERY_CHARS = 6000


def build_query(profile: ResumeProfile, resume_text: str) -> str:
    skills = ", ".join(profile.skills)
    head = " ".join(filter(None, [profile.headline, profile.summary]))
    return f"{head}\nSkills: {skills}\n{resume_text}"[:MAX_QUERY_CHARS]


def score_job(
    profile: ResumeProfile, job: Job, cosine: float, semantic_weight: float
) -> JobMatch:
    resume_skills = {canonicalize(s).lower() for s in profile.skills}
    required = [canonicalize(s) for s in job.required_skills]
    matched = [s for s in required if s.lower() in resume_skills]
    missing = [s for s in required if s.lower() not in resume_skills]

    semantic = max(0.0, min(1.0, cosine / SEMANTIC_CEILING))
    skill = len(matched) / len(required) if required else semantic
    overall = 100 * (semantic_weight * semantic + (1 - semantic_weight) * skill)

    return JobMatch(
        job_id=job.id,
        title=job.title,
        company=job.company,
        location=job.location,
        score=round(overall, 1),
        semantic_score=round(semantic, 3),
        skill_score=round(skill, 3),
        matched_skills=matched,
        missing_skills=missing,
    )


def rank_jobs(
    profile: ResumeProfile,
    resume_text: str,
    index: JobIndex,
    top_k: int,
    semantic_weight: float = 0.5,
    include_ids: tuple[str, ...] = (),
) -> list[JobMatch]:
    """Top-k matches, best first. Jobs in `include_ids` are always returned."""
    # Score the whole catalog when it is small (or a specific job is required);
    # otherwise re-rank a generous candidate pool from the vector search.
    pool = len(index) if include_ids else max(top_k * 5, 25)
    candidates = index.search(build_query(profile, resume_text), k=pool)

    matches = [score_job(profile, job, cos, semantic_weight) for job, cos in candidates]
    matches.sort(key=lambda m: m.score, reverse=True)

    top = matches[:top_k]
    shown = {m.job_id for m in top}
    extras = [m for m in matches[top_k:] if m.job_id in include_ids and m.job_id not in shown]
    return top + extras
