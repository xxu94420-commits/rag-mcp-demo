"""Embedding providers.

Two interchangeable providers:

- ``TfidfEmbedder``: pure-local, numpy-based TF-IDF vectors. No API key,
  no model download. Intended for demo / CI, not production quality.
- ``OpenAIEmbedder``: OpenAI-compatible ``/embeddings`` endpoint, configured
  via environment variables.
"""

from __future__ import annotations

import math
import os
import re
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Split into lowercase word tokens (CJK kept as single chars)."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


class Embedder(Protocol):
    #: True for corpus-dependent embedders (e.g. TF-IDF) that must be
    #: re-fitted and re-embed the whole index after every ingest.
    stateful: bool

    def fit(self, documents: list[str]) -> None: ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class TfidfEmbedder:
    """TF-IDF vectors with L2 normalization.

    ``fit`` builds the vocabulary and IDF weights from the corpus;
    queries are projected onto the same vocabulary (OOV tokens ignored).
    """

    stateful = True

    def __init__(self) -> None:
        self._vocabulary: dict[str, int] = {}
        self._idf: np.ndarray | None = None

    def fit(self, documents: list[str]) -> None:
        docs_tokens = [set(tokenize(d)) for d in documents]
        vocab: dict[str, int] = {}
        df: dict[str, int] = {}
        for tokens in docs_tokens:
            for tok in tokens:
                df[tok] = df.get(tok, 0) + 1
        # keep tokens that appear in at least 1 document and at most 95%
        n_docs = max(len(documents), 1)
        for tok, count in sorted(df.items()):
            if count <= 0.95 * n_docs:
                vocab[tok] = len(vocab)
        self._vocabulary = vocab
        self._idf = np.array(
            [math.log((1 + n_docs) / (1 + df[t])) + 1.0 for t in vocab],
            dtype=np.float64,
        )

    def _vectorize(self, text: str) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("embedder not fitted")
        vec = np.zeros(len(self._vocabulary), dtype=np.float64)
        for tok in tokenize(text):
            idx = self._vocabulary.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        vec = vec * self._idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vectorize(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(text)


class OpenAIEmbedder:
    """OpenAI-compatible embeddings endpoint."""

    stateful = False

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.timeout = timeout
        if not self.api_key or not self.base_url:
            raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")

    def fit(self, documents: list[str]) -> None:  # stateless provider
        return None

    def _embed(self, texts: list[str]) -> np.ndarray:
        import httpx  # lazy import: only needed when this provider is used

        resp = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda item: item["index"])
        return np.vstack([np.asarray(item["embedding"]) for item in data])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


def build_embedder() -> Embedder:
    """Pick an embedder from the environment (OpenAI if configured, else TF-IDF)."""
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        return OpenAIEmbedder()
    return TfidfEmbedder()
