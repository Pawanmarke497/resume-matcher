"""Pydantic models: used for LLM structured output and for the API contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- resume
class Education(BaseModel):
    institution: str
    degree: str | None = None
    year: str | None = None


class Experience(BaseModel):
    title: str
    company: str | None = None
    duration: str | None = Field(None, description="e.g. '2021 - 2024'")
    highlights: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """Structured representation of a resume (target of LLM structured output)."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    headline: str | None = Field(None, description="One-line professional title")
    summary: str | None = Field(None, description="2-3 sentence professional summary")
    skills: list[str] = Field(
        default_factory=list,
        description="Technical and professional skills explicitly present in the resume",
    )
    years_experience: float | None = Field(
        None, description="Total years of professional experience, if determinable"
    )
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)


# ----------------------------------------------------------------------------- jobs
class JobIn(BaseModel):
    title: str
    company: str
    location: str | None = None
    description: str = Field(min_length=30)
    required_skills: list[str] = Field(
        default_factory=list,
        description="Leave empty to auto-extract from the description",
    )


class Job(JobIn):
    id: str


# --------------------------------------------------------------------------- output
class JobMatch(BaseModel):
    job_id: str
    title: str
    company: str
    location: str | None = None
    score: float = Field(description="Overall match, 0-100")
    semantic_score: float = Field(description="Embedding similarity, 0-1")
    skill_score: float = Field(description="Share of required skills found, 0-1")
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """Resume feedback for one target job (target of LLM structured output)."""

    fit_summary: str = Field(description="2-3 sentence honest assessment of fit")
    strengths: list[str] = Field(description="Strengths relevant to this role")
    gaps: list[str] = Field(description="Missing or weak areas for this role")
    suggestions: list[str] = Field(
        description="Concrete, truthful resume improvements for this role"
    )


class AnalysisResult(BaseModel):
    profile: ResumeProfile
    matches: list[JobMatch]
    target_job_id: str | None = None
    critique: Critique | None = None
    cover_letter: str | None = None
    warnings: list[str] = Field(default_factory=list)
