"""Embedding providers behind LangChain's `Embeddings` interface."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np
from langchain_core.embeddings import Embeddings

from app.config import Settings

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the to with "
    "will you your we our this their they them i my me he she his her was were been "
    "can must should would could using use used etc".split()
)


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (feature hashing + sublinear TF).

    Needs no API key or model download, so the project runs and tests offline.
    It captures keyword overlap, not meaning, so use OpenAI embeddings (or any
    sentence-transformer) in production.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        tokens = [
            t for t in re.findall(r"[a-z0-9+#]+", text.lower())
            if t not in _STOPWORDS and len(t) > 1
        ]
        vec = np.zeros(self.dim, dtype=np.float64)
        for token, count in Counter(tokens).items():
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(count))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)
    return HashingEmbeddings()
