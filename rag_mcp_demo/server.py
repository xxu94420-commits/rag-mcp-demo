"""MCP server (stdio transport) exposing the RAG pipeline as tools.

Run:  python -m rag_mcp_demo.server

Client config example (Claude Desktop / Cursor):
    {
      "mcpServers": {
        "rag-demo": {
          "command": "python",
          "args": ["-m", "rag_mcp_demo.server"],
          "cwd": "/path/to/rag-mcp-demo",
          "env": {"RAG_INDEX_DIR": ".ragindex"}
        }
      }
    }
"""

from __future__ import annotations

import os
from pathlib import Path

from .embeddings import build_embedder
from .llm import build_chat_model
from .mcp_tools import RagTools
from .pipeline import RAGPipeline

DEFAULT_INDEX_DIR = ".ragindex"


def build_tools(index_dir: str | None = None) -> RagTools:
    """Load (or start empty) the pipeline and wrap it as MCP tools."""
    directory = Path(index_dir or os.environ.get("RAG_INDEX_DIR", DEFAULT_INDEX_DIR))
    embedder = build_embedder()
    chat_model = build_chat_model()
    if directory.exists():
        pipeline = RAGPipeline.load(directory, embedder=embedder, chat_model=chat_model)
    else:
        pipeline = RAGPipeline(embedder=embedder, chat_model=chat_model)
    return RagTools(pipeline)


def main() -> None:
    from mcp.server.fastmcp import FastMCP

    tools = build_tools()
    mcp = FastMCP("rag-mcp-demo")

    @mcp.tool(description="列出已索引的文档及其分块数量")
    def list_documents() -> list[dict]:
        return tools.list_documents()

    @mcp.tool(description="在已索引文档中检索与查询最相关的分块，返回打分排序的结果")
    def search_documents(query: str, k: int = 4) -> list[dict]:
        return tools.search_documents(query, k=k)

    @mcp.tool(description="基于已索引文档回答问题，返回答案与引用来源")
    def ask_document(question: str, k: int = 4) -> dict:
        return tools.ask_document(question, k=k)

    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
