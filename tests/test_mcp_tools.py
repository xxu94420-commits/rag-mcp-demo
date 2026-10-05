"""MCP tool layer tests (no transport needed)."""

import pytest

from rag_mcp_demo.llm import ExtractiveFallback
from rag_mcp_demo.embeddings import TfidfEmbedder
from rag_mcp_demo.mcp_tools import RagTools
from rag_mcp_demo.pipeline import RAGPipeline


@pytest.fixture()
def tools() -> RagTools:
    p = RAGPipeline(embedder=TfidfEmbedder(), chat_model=ExtractiveFallback())
    p.ingest_text("MCP 服务器通过标准化接口暴露工具。支持 stdio 传输。", source="mcp.md")
    p.ingest_text("RAG 通过检索增强生成，把外部资料注入提示词。", source="rag.md")
    return RagTools(p)


def test_list_documents(tools):
    docs = tools.list_documents()
    assert {d["source"] for d in docs} == {"mcp.md", "rag.md"}
    assert all(d["chunks"] >= 1 for d in docs)


def test_search_documents_ranking(tools):
    hits = tools.search_documents("stdio 传输方式", k=2)
    assert hits[0]["source"] == "mcp.md"
    assert hits[0]["score"] >= hits[-1]["score"]


def test_ask_document(tools):
    result = tools.ask_document("MCP 如何暴露工具？")
    assert result["answer"]
    assert result["sources"][0]["source"] == "mcp.md"


def test_k_is_capped(tools):
    assert len(tools.search_documents("MCP", k=100)) <= tools.max_k


def test_unknown_tool_rejected(tools):
    with pytest.raises(ValueError):
        tools.call("delete_everything", {})


def test_schema_covers_all_tools(tools):
    names = {t["name"] for t in tools.schema()}
    assert names == {"list_documents", "search_documents", "ask_document"}
