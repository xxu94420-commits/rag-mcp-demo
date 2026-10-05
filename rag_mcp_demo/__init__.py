"""rag-mcp-demo: a minimal, verifiable RAG pipeline with an MCP server.

- Local-first: runs with TF-IDF embeddings and extractive fallback answers,
  no API key required.
- Optional OpenAI-compatible endpoints for embeddings and chat completion.
- MCP (Model Context Protocol) server exposing the pipeline as tools.
"""

__version__ = "0.1.0"
