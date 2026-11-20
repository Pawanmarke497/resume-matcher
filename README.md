# Resume & Job Matcher

Upload a resume (PDF, DOCX or TXT) and get:

1. A **structured profile** (Pydantic) with contact info, skills and experience
2. **Ranked job matches** from a vector-indexed catalog, with matched and missing skills
3. **Tailored feedback** for a target job (strengths, gaps, truthful suggestions)
4. An optional **cover letter**

**Stack:** FastAPI · LangGraph · LangChain · RAG-style retrieval over a job index · Pydantic structured output. Works with OpenAI or Anthropic, **and fully offline with no API key**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add OPENAI_API_KEY or ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the web UI, or http://localhost:8000/docs for the API.

```bash
curl -F "file=@resume.pdf" -F top_k=5 -F generate_cover_letter=true \
     http://localhost:8000/analyze
```

Run the tests: `pytest`. Run in Docker: `docker build -t resume-matcher . && docker run -p 8000:8000 --env-file .env resume-matcher`

## Offline mode vs LLM mode

| | Offline (no key) | With an API key |
|---|---|---|
| Resume parsing | regex and skill taxonomy | LLM with Pydantic structured output |
| Embeddings | hashed bag-of-words (keyword overlap) | OpenAI embeddings (semantic) |
| Feedback / letter | templates | LLM, constrained to facts in the resume |

Offline mode exists so you can develop and test without spending money, and results include a warning saying so. For real quality, set a key.
Anthropic has no embeddings API, so with `ANTHROPIC_API_KEY` alone the LLM is Claude and embeddings stay offline. Add `OPENAI_API_KEY` (or swap in another embedding provider in `app/embeddings.py`) for semantic retrieval.

## Architecture

```
POST /analyze (file, top_k, generate_cover_letter, target_job_id)
        │
   extract text (pypdf / python-docx)
        │
        ▼
   ┌──────────────── LangGraph ────────────────┐
   │ parse_resume ──► extract_skills           │
   │   │ (bad input → END)     │               │
   │                           ▼               │
   │                      match_jobs ──(no jobs → END)
   │                           │               │
   │                           ▼               │
   │                       critique            │
   │                           │ (if requested)│
   │                           ▼               │
   │                     cover_letter ──► END  │
   └───────────────────────────────────────────┘
```

| File | Role |
|---|---|
| `app/main.py` | FastAPI app, upload validation, endpoints |
| `app/graph.py` | LangGraph state, nodes, conditional edges |
| `app/llm.py` | Prompts + structured output (LangChain) and the offline fallback |
| `app/job_index.py` | Job catalog + `InMemoryVectorStore`, persisted to `data/jobs.json` |
| `app/matching.py` | Scoring: blends embedding similarity and required-skill coverage |
| `app/skills.py` | Skill taxonomy, normalisation, extraction |
| `app/parsing.py` | PDF / DOCX / TXT to text |
| `app/schemas.py` | All Pydantic models |

### How the score works

`score = 100 × (w · semantic + (1 − w) · skill_coverage)`, with `w = SEMANTIC_WEIGHT` (default 0.5).
*Semantic* is the cosine similarity of resume vs job text, rescaled. *Skill coverage* is the share of the job's required skills found in the resume. Skills are normalised through the taxonomy first, so "Postgres" and "PostgreSQL" count as the same.

### Design decisions worth knowing

- **Skills are merged from two sources.** The LLM's list is normalised and unioned with deterministic taxonomy hits, so a missed skill doesn't silently lower a match.
- **Feedback and letters can't invent facts.** Prompts only allow what's in the parsed profile, and suggestions must be truthful edits.
- **A requested `target_job_id` is always included** in the returned matches, even if it isn't in the top-k.
- **Failures degrade gracefully.** If feedback or letter generation fails, you still get the matches and a warning.

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/analyze` | multipart: `file`, `top_k` (1-20), `generate_cover_letter`, `target_job_id` |
| `GET` | `/jobs` · `/jobs/{id}` | list or fetch jobs |
| `POST` | `/jobs` | add a job (`title`, `company`, `description`; skills auto-extracted if omitted) |
| `GET` | `/health` | provider info and job count |

Errors: `404` unknown job, `413` file too large, `415` unsupported type, `422` unreadable or empty resume.

## Extending it

- **Real vector DB:** replace `InMemoryVectorStore` in `app/job_index.py` with Chroma, Qdrant or pgvector.
- **Better ranking:** add a cross-encoder or Cohere reranker over the top candidates in `matching.py`.
- **Scanned PDFs:** add OCR (e.g. `pytesseract`) when extracted text is empty.
- **Persistence and users:** store analyses in Postgres and add a LangGraph checkpointer for resumable runs.
- **Streaming:** use `graph.astream(...)` with SSE so the UI shows each step as it finishes.
- **Evaluation:** build a small labelled set of resume/job pairs and track top-1 / top-3 accuracy as you change weights or prompts.
- **Privacy:** resumes are personal data. This app doesn't store uploads, but if you add persistence, add retention limits and consent.

## Privacy note

In LLM mode, resume text is sent to your chosen provider. In offline mode nothing leaves your machine.
