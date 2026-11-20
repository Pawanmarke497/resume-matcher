import pytest

from app.embeddings import HashingEmbeddings
from app.job_index import JobIndex
from app.llm import OfflineLLMService
from app.matching import rank_jobs
from app.parsing import UnsupportedFileType, extract_text
from app.schemas import JobIn
from app.skills import canonicalize, dedupe_skills, extract_skills


def test_extract_skills_whole_token_matching():
    found = extract_skills("Wrote JavaScript and C++ services. Used fast api.")
    assert "JavaScript" in found
    assert "C++" in found
    assert "FastAPI" in found
    assert "Java" not in found  # "Java" must not match inside "JavaScript"
    assert "Java" in extract_skills("Five years of Java, Spring and SQL.")


def test_canonicalize_and_dedupe():
    assert canonicalize("postgres") == "PostgreSQL"
    assert canonicalize("Fast API") == "FastAPI"
    assert canonicalize("Obscure Tool") == "Obscure Tool"
    assert dedupe_skills(["python", "Python", "PYTHON", "k8s", "Kubernetes"]) == [
        "Python",
        "Kubernetes",
    ]


def test_extract_text_txt_and_unsupported():
    assert extract_text("cv.txt", b"hello\r\n\r\n\r\n\r\nworld") == "hello\n\nworld"
    with pytest.raises(UnsupportedFileType):
        extract_text("cv.exe", b"x")


def test_hashing_embeddings_are_deterministic_and_normalised():
    emb = HashingEmbeddings(dim=256)
    a, b = emb.embed_query("python fastapi docker"), emb.embed_query("python fastapi docker")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_offline_parser_extracts_contact_and_experience(ai_resume_text):
    profile = OfflineLLMService().parse_resume(ai_resume_text)
    assert profile.full_name == "Priya Sharma"
    assert profile.email == "priya.sharma@example.com"
    assert profile.years_experience and profile.years_experience >= 4
    assert "LangGraph" in profile.skills


def test_ranking_prefers_relevant_job(ai_resume_text):
    index = JobIndex(HashingEmbeddings())
    index.add(JobIn(
        title="AI Engineer", company="A",
        description="Build RAG and LLM apps with LangChain, LangGraph and FastAPI in Python.",
    ))
    index.add(JobIn(
        title="Frontend Developer", company="B",
        description="Build React and TypeScript interfaces with HTML, CSS and GraphQL.",
    ))
    profile = OfflineLLMService().parse_resume(ai_resume_text)
    matches = rank_jobs(profile, ai_resume_text, index, top_k=2)
    assert [m.title for m in matches] == ["AI Engineer", "Frontend Developer"]
    assert matches[0].score > matches[1].score
    assert "LangGraph" in matches[0].matched_skills
    assert "React" in matches[1].missing_skills


def test_include_ids_forces_job_into_results(ai_resume_text):
    index = JobIndex(HashingEmbeddings())
    ids = [
        index.add(JobIn(title=f"Role {i}", company="X", description="Python backend " * 5)).id
        for i in range(4)
    ]
    profile = OfflineLLMService().parse_resume(ai_resume_text)
    matches = rank_jobs(profile, ai_resume_text, index, top_k=1, include_ids=(ids[-1],))
    assert ids[-1] in {m.job_id for m in matches}
