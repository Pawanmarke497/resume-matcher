"""Runtime settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "offline": "offline",
}


@dataclass(frozen=True)
class Settings:
    # "openai" | "anthropic" | "offline"
    llm_provider: str = "offline"
    llm_model: str = "offline"
    # "openai" | "offline"
    embedding_provider: str = "offline"
    embedding_model: str = "text-embedding-3-small"
    jobs_file: Path = ROOT / "data" / "jobs.json"
    max_upload_mb: int = 5
    # Final score = semantic_weight * semantic + (1 - semantic_weight) * skill coverage
    semantic_weight: float = 0.5

    @classmethod
    def from_env(cls) -> "Settings":
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))

        llm = os.getenv("LLM_PROVIDER", "auto").lower()
        if llm == "auto":
            llm = "openai" if has_openai else "anthropic" if has_anthropic else "offline"

        emb = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
        if emb == "auto":
            emb = "openai" if has_openai else "offline"

        return cls(
            llm_provider=llm,
            llm_model=os.getenv("LLM_MODEL", DEFAULT_MODELS.get(llm, "")),
            embedding_provider=emb,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            jobs_file=Path(os.getenv("JOBS_FILE", str(ROOT / "data" / "jobs.json"))),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "5")),
            semantic_weight=float(os.getenv("SEMANTIC_WEIGHT", "0.5")),
        )
