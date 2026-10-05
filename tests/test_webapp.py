"""Tests for the web demo (FastAPI TestClient)."""

from fastapi.testclient import TestClient

from rag_mcp_demo.webapp import app, get_pipeline


def _client() -> TestClient:
    get_pipeline()  # warm the in-memory index before serving
    return TestClient(app)


def test_index_page_lists_indexed_docs():
    resp = _client().get("/")
    assert resp.status_code == 200
    assert "rag-mcp-demo" in resp.text
    assert "已索引文档" in resp.text


def test_ask_returns_answer_and_sources():
    resp = _client().post("/ask", json={"question": "MCP 服务器暴露哪三类原语？"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]
    assert len(data["sources"]) >= 1
    assert {"source", "score", "snippet"} <= set(data["sources"][0])


def test_ask_rejects_empty_question():
    resp = _client().post("/ask", json={"question": ""})
    assert resp.status_code == 422
