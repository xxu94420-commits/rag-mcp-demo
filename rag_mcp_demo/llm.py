"""LLM providers for answer generation.

- ``OpenAIChat``: OpenAI-compatible ``/chat/completions``.
- ``ExtractiveFallback``: no model at all; picks the most relevant sentence
  from retrieved chunks so the pipeline stays verifiable without API keys.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from .store import SearchHit

SENTENCE_RE = re.compile(r"[^。！？.!?\n]+[。！？.!?\n]*")

# 疑问与功能词：对 CJK 单字匹配噪声极大，抽取评分时直接剔除
STOPWORDS = set("的 了 在 和 与 或 是 什么 哪些 怎么 如何 吗 呢 吧 啊 有 哪些 为什么".split())


class ChatModel(Protocol):
    def answer(self, question: str, hits: list[SearchHit]) -> str: ...


PROMPT_TEMPLATE = """你是严谨的问答助手。仅依据给定资料回答问题；资料不足时明确说"资料中未提及"，不要编造。回答末尾用 [1][2] 形式标注引用来源编号。

资料:
{context}

问题: {question}
"""


def format_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(
        f"[{i}] （来源: {hit.chunk.source}）\n{hit.chunk.text}"
        for i, hit in enumerate(hits, start=1)
    )


class OpenAIChat:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("CHAT_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")
        self.timeout = timeout

    def answer(self, question: str, hits: list[SearchHit]) -> str:
        import httpx

        if not self.api_key or not self.base_url:
            raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是严谨的问答助手，只依据给定资料回答。"},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(
                        context=format_context(hits), question=question)},
                ],
                "temperature": 0.2,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class ExtractiveFallback:
    """Deterministic fallback: extract the sentence whose token overlap with
    the question is largest. Clearly marked as fallback, never pretends to be
    a generative model."""

    def answer(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return "资料库中没有相关内容。"
        from .embeddings import tokenize

        q_tokens = {t for t in tokenize(question) if t not in STOPWORDS}
        best_sentence, best_score, best_ref = "", 0.0, 1
        for ref, hit in enumerate(hits, start=1):
            for m in SENTENCE_RE.finditer(hit.chunk.text):
                sentence = m.group(0).strip()
                if len(sentence) < 4:
                    continue
                s_tokens = set(tokenize(sentence))
                # 多字词（英文词、数字）权重 1.0，CJK 单字权重 0.2，
                # 避免"进行/推理"这类单字巧合造成误匹配
                score = sum(
                    1.0 if len(t) > 1 else 0.2
                    for t in q_tokens & s_tokens
                ) / max(len(q_tokens), 1)
                if score > best_score:
                    best_sentence, best_score, best_ref = sentence, score, ref
        if not best_sentence or best_score < 0.1:
            return "资料中没有找到与问题相关的内容（降级抽取模式）。"
        return f"{best_sentence} [{best_ref}]（注: 当前为无 LLM 的降级抽取回答）"


def build_chat_model() -> ChatModel:
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        return OpenAIChat()
    return ExtractiveFallback()
