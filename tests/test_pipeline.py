"""End-to-end pipeline tests with the deterministic fallback model."""

from rag_mcp_demo.llm import ExtractiveFallback
from rag_mcp_demo.embeddings import TfidfEmbedder
from rag_mcp_demo.pipeline import RAGPipeline

CORPUS = {
    "agent.md": "AI Agent 是一种能够自主规划任务并调用工具完成目标的大模型应用形态。"
    "Agent 通常包含任务规划、工具调用、记忆和反思四个核心模块。"
    "ReAct 模式通过交替进行推理和行动，提升了 Agent 完成复杂任务的成功率。",
    "mcp.md": "MCP（Model Context Protocol）是一个开放协议，用于让大模型应用连接外部数据源和工具。"
    "MCP 采用客户端-服务器架构，服务器通过标准化接口暴露工具、资源和提示。"
    "MCP 传输层支持 stdio 和 HTTP 两种方式。",
}


def _pipeline() -> RAGPipeline:
    p = RAGPipeline(embedder=TfidfEmbedder(), chat_model=ExtractiveFallback())
    for source, text in CORPUS.items():
        p.ingest_text(text, source=source)
    return p


def test_ingest_counts_chunks():
    p = RAGPipeline(
        embedder=TfidfEmbedder(), chat_model=ExtractiveFallback(),
        chunk_size=50, chunk_overlap=10,
    )
    n = p.ingest_text(CORPUS["agent.md"], source="agent.md")
    assert n >= 2  # 约 130 字文本按 50 字块长应分为多块


def test_ask_returns_answer_with_sources():
    p = _pipeline()
    answer = p.ask("Agent 的核心模块有哪些？")
    assert answer.text
    assert len(answer.sources) >= 1
    assert any("agent.md" in s["source"] for s in answer.sources)
    assert all(0 <= s["score"] <= 1 for s in answer.sources)


def test_ask_unrelated_question_graceful():
    p = _pipeline()
    answer = p.ask("量子引力理论的最新进展是什么？")
    # 降级模式不应编造，应明确说明资料不足
    assert "资料" in answer.text


def test_index_persistence_roundtrip(tmp_path):
    p = _pipeline()
    p.save(tmp_path)
    loaded = RAGPipeline.load(tmp_path, embedder=TfidfEmbedder(), chat_model=ExtractiveFallback())
    assert len(loaded.store.chunks) == len(p.store.chunks)


def test_ingest_unsupported_suffix(tmp_path):
    p = _pipeline()
    bad = tmp_path / "doc.pdf"
    bad.write_bytes(b"%PDF-1.4")
    try:
        p.ingest_path(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised
