"""MCP tool layer: thin wrappers over RAGPipeline, shared by the MCP server.

Kept separate from server.py so the tools can be unit-tested without any
MCP transport.
"""

from __future__ import annotations

from .pipeline import RAGPipeline


class RagTools:
    def __init__(self, pipeline: RAGPipeline, max_k: int = 8) -> None:
        self.pipeline = pipeline
        self.max_k = max_k

    def list_documents(self) -> list[dict]:
        """List indexed source documents and their chunk counts."""
        counts: dict[str, int] = {}
        for chunk in self.pipeline.store.chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return [
            {"source": source, "chunks": n}
            for source, n in sorted(counts.items())
        ]

    def search_documents(self, query: str, k: int = 4) -> list[dict]:
        """Semantic-ish search over indexed chunks; returns ranked hits with scores."""
        k = max(1, min(k, self.max_k))
        hits = self.pipeline.retrieve(query, k=k)
        return [
            {
                "source": hit.chunk.source,
                "chunk_index": hit.chunk.index,
                "score": round(hit.score, 4),
                "text": hit.chunk.text,
            }
            for hit in hits
        ]

    def ask_document(self, question: str, k: int = 4) -> dict:
        """Answer a question from indexed documents, with cited sources."""
        k = max(1, min(k, self.max_k))
        answer = self.pipeline.ask(question, k=k)
        return {"answer": answer.text, "sources": answer.sources}

    def schema(self) -> list[dict]:
        """Tool schemas in MCP format (used by server.py and tests)."""
        return [
            {
                "name": "list_documents",
                "description": "列出已索引的文档及其分块数量",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "search_documents",
                "description": "在已索引文档中检索与查询最相关的分块，返回打分排序的结果",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索关键词或问题"},
                        "k": {"type": "integer", "description": "返回条数", "default": 4},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ask_document",
                "description": "基于已索引文档回答问题，返回答案与引用来源",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "要回答的问题"},
                        "k": {"type": "integer", "description": "检索分块数", "default": 4},
                    },
                    "required": ["question"],
                },
            },
        ]

    def call(self, name: str, arguments: dict) -> dict:
        if name == "list_documents":
            return {"documents": self.list_documents()}
        if name == "search_documents":
            return {"hits": self.search_documents(**arguments)}
        if name == "ask_document":
            return self.ask_document(**arguments)
        raise ValueError(f"unknown tool: {name}")
