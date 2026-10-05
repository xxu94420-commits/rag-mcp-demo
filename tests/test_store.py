"""Vector store + TF-IDF embedder tests."""

import numpy as np
import pytest

from rag_mcp_demo.embeddings import TfidfEmbedder
from rag_mcp_demo.store import Chunk, VectorStore


def _fitted_embedder():
    docs = [
        "玫瑰是红色的花，常在春天开放。",
        "Python 是一种广泛使用的编程语言。",
        "Java 也是编程语言，常用于后端服务。",
    ]
    emb = TfidfEmbedder()
    emb.fit(docs)
    return emb, docs


def test_tfidf_retrieves_relevant_chunk():
    emb, docs = _fitted_embedder()
    store = VectorStore()
    chunks = [Chunk(id=f"c{i}", text=d, source=f"doc{i}.md", index=0) for i, d in enumerate(docs)]
    store.add(chunks, emb.embed_documents(docs))

    hits = store.search(emb.embed_query("什么编程语言适合写后端"), k=2)
    assert len(hits) == 2
    assert "编程语言" in hits[0].chunk.text
    assert hits[0].score >= hits[-1].score


def test_empty_store_returns_empty():
    store = VectorStore()
    assert store.search(np.zeros(5), k=3) == []


def test_dimension_mismatch_rejected():
    emb, docs = _fitted_embedder()
    store = VectorStore()
    chunks = [Chunk(id="c0", text=docs[0], source="a.md", index=0)]
    store.add(chunks, emb.embed_documents([docs[0]]))
    with pytest.raises(ValueError):
        store.add(chunks, np.zeros((1, 3)))


def test_save_and_load_roundtrip(tmp_path):
    emb, docs = _fitted_embedder()
    store = VectorStore()
    chunks = [Chunk(id=f"c{i}", text=d, source=f"doc{i}.md", index=i) for i, d in enumerate(docs)]
    store.add(chunks, emb.embed_documents(docs))
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded.chunks) == 3
    hits = loaded.search(emb.embed_query("编程语言"), k=1)
    assert "编程语言" in hits[0].chunk.text
