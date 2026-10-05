"""Vector store: chunked documents + matrix, cosine-similarity search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    index: int  # position within its source document


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)
    matrix: np.ndarray | None = None  # (n_chunks, dim), L2-normalized rows

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        if self.matrix is None:
            self.matrix = vectors
        else:
            if vectors.shape[1] != self.matrix.shape[1]:
                raise ValueError("vector dimension mismatch")
            self.matrix = np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[SearchHit]:
        if self.matrix is None or not self.chunks:
            return []
        q = query_vector / max(np.linalg.norm(query_vector), 1e-12)
        scores = self.matrix @ q
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchHit(chunk=self.chunks[i], score=float(scores[i])) for i in top]

    # ---- persistence -------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = [
            {"id": c.id, "text": c.text, "source": c.source, "index": c.index}
            for c in self.chunks
        ]
        (directory / "chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        np.save(directory / "vectors.npy", self.matrix)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in meta]
        matrix = np.load(directory / "vectors.npy")
        return cls(chunks=chunks, matrix=matrix)
