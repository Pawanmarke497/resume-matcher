"""A small skill taxonomy used to normalise and extract skills deterministically.

LLM-extracted skills are free-form ("fast api", "Postgres"). Mapping everything to a
canonical name makes resume-vs-job skill overlap reliable. Extend TAXONOMY for your domain.
"""

from __future__ import annotations

import re

# canonical name -> aliases (matched case-insensitively as whole tokens)
TAXONOMY: dict[str, list[str]] = {
    # languages
    "Python": ["python"],
    "JavaScript": ["javascript"],
    "TypeScript": ["typescript"],
    "Java": ["java"],
    "Go": ["golang"],
    "C++": ["c++"],
    "SQL": ["sql"],
    # web / backend
    "FastAPI": ["fastapi", "fast api"],
    "Flask": ["flask"],
    "Django": ["django"],
    "Node.js": ["node.js", "nodejs"],
    "React": ["react", "reactjs", "react.js"],
    "HTML/CSS": ["html", "css", "html/css", "html5", "css3"],
    "REST APIs": ["rest api", "rest apis", "restful", "restful apis"],
    "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservice"],
    "Pydantic": ["pydantic"],
    # AI / ML
    "LangChain": ["langchain"],
    "LangGraph": ["langgraph"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "LLMs": ["llm", "llms", "large language model", "large language models"],
    "Prompt Engineering": ["prompt engineering"],
    "Embeddings": ["embedding", "embeddings"],
    "Vector Databases": [
        "vector database", "vector databases", "vector db", "vector store",
        "chroma", "pinecone", "qdrant", "weaviate", "pgvector", "faiss",
    ],
    "OpenAI API": ["openai"],
    "Hugging Face": ["hugging face", "huggingface"],
    "NLP": ["nlp", "natural language processing"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "MLOps": ["mlops"],
    # data
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb"],
    "Redis": ["redis"],
    "Spark": ["spark", "pyspark"],
    "Airflow": ["airflow"],
    "Kafka": ["kafka"],
    "ETL": ["etl", "elt"],
    # infra
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "GCP": ["gcp", "google cloud"],
    "Azure": ["azure"],
    "Terraform": ["terraform"],
    "Linux": ["linux"],
    "CI/CD": ["ci/cd", "cicd", "github actions", "jenkins", "gitlab ci"],
    "Git": ["git"],
    # practices
    "Testing": ["pytest", "unit testing", "unit tests"],
    "Agile": ["agile", "scrum"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canon, _aliases in TAXONOMY.items():
    _ALIAS_TO_CANONICAL[_canon.lower()] = _canon
    for _a in _aliases:
        _ALIAS_TO_CANONICAL[_a.lower()] = _canon


def _build_pattern(aliases: list[str]) -> re.Pattern[str]:
    body = "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
    # Not preceded/followed by a word char, '+' or '#', so "Java" != "JavaScript"
    # and "C++" is matched correctly.
    return re.compile(rf"(?<![\w+#])(?:{body})(?![\w+#])", re.IGNORECASE)


_PATTERNS: dict[str, re.Pattern[str]] = {
    canon: _build_pattern([canon, *aliases]) for canon, aliases in TAXONOMY.items()
}


def canonicalize(skill: str) -> str:
    """Map a free-form skill string to its canonical name (or tidy it if unknown)."""
    cleaned = skill.strip()
    return _ALIAS_TO_CANONICAL.get(cleaned.lower(), cleaned)


def extract_skills(text: str) -> list[str]:
    """Return canonical skills found in free text, in taxonomy order."""
    return [canon for canon, pattern in _PATTERNS.items() if pattern.search(text)]


def dedupe_skills(skills: list[str]) -> list[str]:
    """Canonicalise and de-duplicate (case-insensitive), preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in skills:
        canon = canonicalize(s)
        if canon and canon.lower() not in seen:
            seen.add(canon.lower())
            out.append(canon)
    return out
