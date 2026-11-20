"""FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from app.config import ROOT, Settings
from app.embeddings import get_embeddings
from app.graph import build_graph
from app.job_index import JobIndex
from app.llm import get_llm_service
from app.parsing import UnreadableFile, UnsupportedFileType, extract_text
from app.schemas import AnalysisResult, Job, JobIn

logging.basicConfig(level=logging.INFO)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        index = JobIndex(get_embeddings(settings), settings.jobs_file)
        index.load()
        llm = get_llm_service(settings)
        app.state.settings = settings
        app.state.index = index
        app.state.graph = build_graph(llm, index, settings)
        yield

    app = FastAPI(
        title="Resume & Job Matcher",
        description="Parses a resume, ranks jobs, and generates feedback and cover letters.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/health")
    def health(request: Request) -> dict:
        s: Settings = request.app.state.settings
        return {
            "status": "ok",
            "llm_provider": s.llm_provider,
            "embedding_provider": s.embedding_provider,
            "jobs_indexed": len(request.app.state.index),
        }

    # ------------------------------------------------------------------ jobs
    @app.get("/jobs", response_model=list[Job])
    def list_jobs(request: Request) -> list[Job]:
        return request.app.state.index.all()

    @app.post("/jobs", response_model=Job, status_code=201)
    def add_job(job: JobIn, request: Request) -> Job:
        return request.app.state.index.add(job)

    @app.get("/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str, request: Request) -> Job:
        job = request.app.state.index.get(job_id)
        if job is None:
            raise HTTPException(404, f"Job '{job_id}' not found")
        return job

    # --------------------------------------------------------------- analyze
    @app.post("/analyze", response_model=AnalysisResult)
    async def analyze(
        request: Request,
        file: UploadFile = File(..., description="Resume as PDF, DOCX or TXT"),
        top_k: int = Form(5, ge=1, le=20),
        generate_cover_letter: bool = Form(False),
        target_job_id: str | None = Form(None, description="Defaults to the best match"),
    ) -> AnalysisResult:
        s: Settings = request.app.state.settings
        index: JobIndex = request.app.state.index

        target = (target_job_id or "").strip() or None
        if target and index.get(target) is None:
            raise HTTPException(404, f"Job '{target}' not found")

        limit = s.max_upload_mb * 1024 * 1024
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(413, f"File exceeds {s.max_upload_mb} MB")

        try:
            text = await run_in_threadpool(extract_text, file.filename or "", content)
        except UnsupportedFileType as exc:
            raise HTTPException(415, str(exc)) from exc
        except UnreadableFile as exc:
            raise HTTPException(422, str(exc)) from exc

        state = await request.app.state.graph.ainvoke(
            {
                "resume_text": text,
                "top_k": top_k,
                "generate_cover_letter": generate_cover_letter,
                "target_job_id": target,
                "errors": [],
                "warnings": [],
            }
        )
        if state.get("errors"):
            raise HTTPException(422, " ".join(state["errors"]))

        return AnalysisResult(
            profile=state["profile"],
            matches=state.get("matches", []),
            target_job_id=state.get("target_job_id"),
            critique=state.get("critique"),
            cover_letter=state.get("cover_letter"),
            warnings=state.get("warnings", []),
        )

    return app


app = create_app()
