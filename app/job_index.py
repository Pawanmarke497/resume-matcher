"""Job catalog + vector index (LangChain InMemoryVectorStore).

The catalog is persisted to a JSON file and re-embedded on startup. To scale up, swap
`InMemoryVectorStore` for Chroma / Qdrant / pgvector. Only `add_documents` and
`similarity_search_with_score` are used, so the swap is a one-line change.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.schemas import Job, JobIn
from app.skills import dedupe_skills, extract_skills


def job_text(job: Job) -> str:
    skills = ", ".join(job.required_skills)
    return f"{job.title} at {job.company}. {job.description} Required skills: {skills}"


class JobIndex:
    def __init__(self, embeddings: Embeddings, jobs_file: Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._store = InMemoryVectorStore(embeddings)
        self._jobs_file = jobs_file
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- loading
    def load(self) -> None:
        if not self._jobs_file or not self._jobs_file.exists():
            return
        for raw in json.loads(self._jobs_file.read_text(encoding="utf-8")):
            self._add(Job(**raw))

    def _persist(self) -> None:
        if self._jobs_file:
            self._jobs_file.parent.mkdir(parents=True, exist_ok=True)
            payload = [j.model_dump() for j in self._jobs.values()]
            self._jobs_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _add(self, job: Job) -> Job:
        if not job.required_skills:
            job = job.model_copy(
                update={"required_skills": extract_skills(f"{job.title}. {job.description}")}
            )
        else:
            job = job.model_copy(update={"required_skills": dedupe_skills(job.required_skills)})
        self._jobs[job.id] = job
        self._store.add_documents(
            [Document(page_content=job_text(job), metadata={"job_id": job.id})],
            ids=[job.id],
        )
        return job

    # ------------------------------------------------------------------- API
    def add(self, job_in: JobIn) -> Job:
        with self._lock:
            job = self._add(Job(id=uuid.uuid4().hex[:8], **job_in.model_dump()))
            self._persist()
            return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def __len__(self) -> int:
        return len(self._jobs)

    def search(self, query: str, k: int) -> list[tuple[Job, float]]:
        """Return (job, cosine_similarity) pairs, best first."""
        if not self._jobs:
            return []
        hits = self._store.similarity_search_with_score(query, k=min(k, len(self._jobs)))
        return [
            (self._jobs[doc.metadata["job_id"]], float(score))
            for doc, score in hits
            if doc.metadata.get("job_id") in self._jobs
        ]
