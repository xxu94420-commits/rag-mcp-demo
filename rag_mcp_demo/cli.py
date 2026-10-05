"""Command line interface: ingest documents, ask questions, serve MCP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import build_embedder
from .llm import build_chat_model
from .pipeline import RAGPipeline
from .server import DEFAULT_INDEX_DIR


def build_pipeline(index_dir: Path) -> RAGPipeline:
    embedder = build_embedder()
    chat_model = build_chat_model()
    if index_dir.exists():
        return RAGPipeline.load(index_dir, embedder=embedder, chat_model=chat_model)
    return RAGPipeline(embedder=embedder, chat_model=chat_model)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-mcp-demo", description=__doc__)
    parser.add_argument("--index", default=DEFAULT_INDEX_DIR, help="索引目录")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="导入文件或目录（.txt/.md），构建索引")
    p_ingest.add_argument("path", help="文件或目录路径")

    p_ask = sub.add_parser("ask", help="基于索引回答问题")
    p_ask.add_argument("question")
    p_ask.add_argument("--k", type=int, default=4)

    sub.add_parser("serve", help="启动 MCP server（stdio）")

    args = parser.parse_args()
    index_dir = Path(args.index)

    if args.command == "serve":
        from .server import main as serve_main

        serve_main()
        return

    pipeline = build_pipeline(index_dir)

    if args.command == "ingest":
        n = pipeline.ingest_path(Path(args.path))
        pipeline.save(index_dir)
        print(f"已导入 {n} 个分块，索引保存在 {index_dir}/")
    elif args.command == "ask":
        answer = pipeline.ask(args.question, k=args.k)
        print(answer.text)
        print("\n来源:")
        print(json.dumps(answer.sources, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
