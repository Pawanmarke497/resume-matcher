def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "offline"
    assert body["jobs_indexed"] == 9


def test_analyze_txt_returns_ranked_matches_and_feedback(client, ai_resume_text):
    r = client.post(
        "/analyze",
        files={"file": ("resume.txt", ai_resume_text.encode(), "text/plain")},
        data={"top_k": "3"},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["profile"]["email"] == "priya.sharma@example.com"
    assert "FastAPI" in body["profile"]["skills"]

    matches = body["matches"]
    assert len(matches) == 3
    assert matches[0]["job_id"] == "ai-engineer"
    assert matches == sorted(matches, key=lambda m: m["score"], reverse=True)

    assert body["target_job_id"] == "ai-engineer"
    assert body["critique"]["fit_summary"]
    assert body["cover_letter"] is None
    assert any("offline" in w.lower() for w in body["warnings"])


def test_analyze_docx_with_cover_letter_and_explicit_target(client, docx_bytes):
    r = client.post(
        "/analyze",
        files={"file": ("resume.docx", docx_bytes, "application/octet-stream")},
        data={"top_k": "2", "generate_cover_letter": "true", "target_job_id": "devops-engineer"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_job_id"] == "devops-engineer"
    assert "devops-engineer" in {m["job_id"] for m in body["matches"]}  # forced into results
    assert "DevOps Engineer" in body["cover_letter"]
    assert "Priya Sharma" in body["cover_letter"]


def test_analyze_rejects_bad_input(client, ai_resume_text):
    unsupported = client.post("/analyze", files={"file": ("cv.exe", b"MZ", "application/x-msdownload")})
    assert unsupported.status_code == 415

    too_short = client.post("/analyze", files={"file": ("cv.txt", b"hi", "text/plain")})
    assert too_short.status_code == 422

    unknown_job = client.post(
        "/analyze",
        files={"file": ("cv.txt", ai_resume_text.encode(), "text/plain")},
        data={"target_job_id": "nope"},
    )
    assert unknown_job.status_code == 404

    bad_pdf = client.post("/analyze", files={"file": ("cv.pdf", b"not a pdf", "application/pdf")})
    assert bad_pdf.status_code == 422


def test_add_job_is_searchable_and_listed(client, ai_resume_text):
    created = client.post(
        "/jobs",
        json={
            "title": "LangGraph Specialist",
            "company": "Agentic Co",
            "description": "Design LangGraph agents with LangChain, RAG, FastAPI and Python for production.",
        },
    )
    assert created.status_code == 201
    job = created.json()
    assert "LangGraph" in job["required_skills"]  # auto-extracted

    assert client.get(f"/jobs/{job['id']}").status_code == 200
    assert len(client.get("/jobs").json()) == 10

    r = client.post(
        "/analyze",
        files={"file": ("resume.txt", ai_resume_text.encode(), "text/plain")},
        data={"top_k": "3"},
    )
    assert job["id"] in {m["job_id"] for m in r.json()["matches"]}


def test_home_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Resume" in r.text
