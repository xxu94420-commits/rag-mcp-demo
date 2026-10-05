"""RAG pipeline: ingest documents, index chunks, answer with citations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .chunking import recursive_split
from .embeddings import Embedder
from .llm import ChatModel
from .store import Chunk, SearchHit, VectorStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)  # [{source, score, snippet}]


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        chat_model: ChatModel,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
    ) -> None:
        self.embedder = embedder
        self.chat_model = chat_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = VectorStore()

    # ---- ingest ------------------------------------------------------
    def _reindex(self) -> None:
        """Re-fit a stateful embedder on the full corpus and re-embed all
        chunks, so IDF statistics stay consistent after incremental ingests."""
        if not getattr(self.embedder, "stateful", False):
            return
        texts = [c.text for c in self.store.chunks]
        if not texts:
            return
        self.embedder.fit(texts)
        vectors = self.embedder.embed_documents(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.store.matrix = vectors / norms

    def ingest_text(self, text: str, source: str) -> int:
        """Chunk + index one document; returns number of chunks created."""
        pieces = recursive_split(text, self.chunk_size, self.chunk_overlap)
        if not pieces:
            return 0
        doc_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
        chunks = [
            Chunk(id=f"{doc_id}-{i}", text=p, source=source, index=i)
            for i, p in enumerate(pieces)
        ]
        if getattr(self.embedder, "stateful", False):
            # corpus-dependent embedder: append first, then refit + re-embed
            self.store.chunks.extend(chunks)
            self._reindex()
        else:
            vectors = self.embedder.embed_documents(pieces)
            self.store.add(chunks, vectors)
        return len(chunks)

    def ingest_path(self, path: Path) -> int:
        path = Path(path)
        if path.is_dir():
            total = 0
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in SUPPORTED_SUFFIXES and child.is_file():
                    total += self.ingest_path(child)
            return total
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported file type: {path.suffix} ({path})")
        text = path.read_text(encoding="utf-8")
        return self.ingest_text(text, source=str(path))

    # ---- query -------------------------------------------------------
    def retrieve(self, question: str, k: int = 4) -> list[SearchHit]:
        q_vec = self.embedder.embed_query(question)
        return self.store.search(q_vec, k=k)

    def ask(self, question: str, k: int = 4) -> Answer:
        hits = self.retrieve(question, k=k)
        text = self.chat_model.answer(question, hits)
        sources = [
            {
                "source": hit.chunk.source,
                "score": round(hit.score, 4),
                "snippet": hit.chunk.text[:120],
            }
            for hit in hits
        ]
        return Answer(text=text, sources=sources)

    # ---- persistence -------------------------------------------------
    def save(self, directory: Path) -> None:
        self.store.save(directory)

    @classmethod
    def load(
        cls, directory: Path, embedder: Embedder, chat_model: ChatModel
    ) -> "RAGPipeline":
        pipeline = cls(embedder=embedder, chat_model=chat_model)
        pipeline.store = VectorStore.load(directory)
        pipeline._reindex()
        return pipeline
