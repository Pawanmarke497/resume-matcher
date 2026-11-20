"""Verifies the LangChain wiring (prompt -> structured output / string parser)
without calling a real model, using a fake chat model."""

from itertools import cycle

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.llm import LangChainLLMService
from app.schemas import Critique, Job, JobMatch, ResumeProfile


class FakeChat(GenericFakeChatModel):
    """Returns canned Pydantic objects for structured output, canned text otherwise."""

    def with_structured_output(self, schema, **kwargs):
        self.__dict__.setdefault("seen_kwargs", []).append(kwargs)
        if schema is ResumeProfile:
            obj = ResumeProfile(full_name="Test User", skills=["python", "fast api"])
        else:
            obj = Critique(
                fit_summary="Good fit.", strengths=["a"], gaps=["b"], suggestions=["c"]
            )
        return RunnableLambda(lambda _: obj)


def _service(provider: str = "openai"):
    chat = FakeChat(messages=cycle([AIMessage(content="  Dear team, hello.  ")]))
    return LangChainLLMService(chat, provider), chat


JOB = Job(id="j1", title="AI Engineer", company="Acme", description="x" * 40)
MATCH = JobMatch(
    job_id="j1", title="AI Engineer", company="Acme", score=80,
    semantic_score=0.8, skill_score=0.8, matched_skills=["Python"], missing_skills=[],
)


def test_structured_parse_and_critique_and_letter():
    svc, _ = _service()
    profile = svc.parse_resume("some resume text " * 10)
    assert profile.full_name == "Test User"
    assert svc.critique(profile, JOB, MATCH).fit_summary == "Good fit."
    assert svc.cover_letter(profile, JOB, MATCH) == "Dear team, hello."  # stripped


def test_openai_uses_function_calling_but_anthropic_does_not():
    _, openai_chat = _service("openai")
    assert all(k == {"method": "function_calling"} for k in openai_chat.__dict__["seen_kwargs"])
    _, anthropic_chat = _service("anthropic")
    assert all(k == {} for k in anthropic_chat.__dict__["seen_kwargs"])
