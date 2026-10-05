"""Minimal web demo for the RAG pipeline.

Serves a single-page UI plus a JSON /ask endpoint. On startup it indexes the
bundled ``sample_docs`` in memory, so the demo runs fully offline — the
extractive fallback answers without any external LLM API key.

Run locally:
    uvicorn rag_mcp_demo.webapp:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .embeddings import build_embedder
from .llm import build_chat_model
from .pipeline import RAGPipeline

SAMPLE_DOCS = Path(__file__).resolve().parent.parent / "sample_docs"

app = FastAPI(title="rag-mcp-demo", description="RAG 检索问答 + MCP 工具服务演示")

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Build the pipeline once and index the bundled sample documents."""
    global _pipeline
    if _pipeline is None:
        pipeline = RAGPipeline(embedder=build_embedder(), chat_model=build_chat_model())
        if SAMPLE_DOCS.is_dir():
            pipeline.ingest_path(SAMPLE_DOCS)
        _pipeline = pipeline
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.on_event("startup")
def _warmup() -> None:
    get_pipeline()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    pipeline = get_pipeline()
    docs = sorted({c.source for c in pipeline.store.chunks})
    doc_items = "".join(
        f"<li><code>{Path(d).name}</code></li>" for d in docs
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rag-mcp-demo · RAG 检索问答演示</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         max-width: 760px; margin: 0 auto; padding: 32px 20px; color: #1f2328; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: #59636e; font-size: 14px; margin-bottom: 24px; }}
  .card {{ border: 1px solid #d1d9e0; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  input[type=text] {{ width: 100%; box-sizing: border-box; padding: 10px 12px;
                     border: 1px solid #d1d9e0; border-radius: 6px; font-size: 15px; }}
  button {{ margin-top: 10px; padding: 8px 20px; border: 0; border-radius: 6px;
           background: #1f6feb; color: #fff; font-size: 15px; cursor: pointer; }}
  button:disabled {{ background: #8c959f; cursor: default; }}
  #answer {{ white-space: pre-wrap; line-height: 1.7; }}
  .src {{ font-size: 13px; color: #59636e; border-top: 1px dashed #d1d9e0;
         padding-top: 8px; margin-top: 8px; }}
  .score {{ color: #1a7f37; }}
  ul {{ padding-left: 20px; margin: 6px 0; }}
  code {{ background: #f0f2f5; padding: 1px 5px; border-radius: 4px; }}
  a {{ color: #1f6feb; }}
</style>
</head>
<body>
<h1>rag-mcp-demo</h1>
<div class="sub">RAG 检索问答 + MCP 工具服务 · 抽取式回答，无需外部大模型 API ·
<a href="https://github.com/xxu94420-commits/rag-mcp-demo">GitHub</a></div>

<div class="card">
  <b>已索引文档</b>（{sum(len(c.text) for c in pipeline.store.chunks)} 字符 /
  {len(pipeline.store.chunks)} 个分块）
  <ul>{doc_items}</ul>
</div>

<div class="card">
  <input id="q" type="text" placeholder="对上面的文档提问，例如：MCP 服务器暴露哪三类原语？"
         onkeydown="if(event.key==='Enter')ask()">
  <button id="btn" onclick="ask()">提问</button>
</div>

<div class="card" id="result" style="display:none">
  <b>回答</b>
  <div id="answer"></div>
  <div id="sources"></div>
</div>

<script>
async function ask() {{
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = '检索中…';
  try {{
    const resp = await fetch('/ask', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{question: q}})
    }});
    const data = await resp.json();
    document.getElementById('result').style.display = 'block';
    document.getElementById('answer').textContent = data.answer;
    document.getElementById('sources').innerHTML = data.sources.map(s =>
      `<div class="src">📄 <code>${{s.source}}</code> ·
       相似度 <span class="score">${{s.score}}</span><br>${{s.snippet}}…</div>`
    ).join('');
  }} catch (e) {{
    document.getElementById('answer').textContent = '请求失败：' + e;
    document.getElementById('result').style.display = 'block';
  }} finally {{
    btn.disabled = false; btn.textContent = '提问';
  }}
}}
</script>
</body>
</html>"""


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    answer = get_pipeline().ask(req.question, k=req.k)
    return AskResponse(answer=answer.text, sources=answer.sources)
