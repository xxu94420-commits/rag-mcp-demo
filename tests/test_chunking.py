"""Recursive splitter tests."""

import pytest

from rag_mcp_demo.chunking import recursive_split


def test_short_text_single_chunk():
    assert recursive_split("短文本。") == ["短文本。"]


def test_chunks_within_size():
    text = "第一段内容。" * 50  # 300 chars
    chunks = recursive_split(text, chunk_size=80, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)


def test_paragraph_boundary_respected():
    text = "甲段落。\n\n乙段落。\n\n丙段落。"
    chunks = recursive_split(text, chunk_size=200, chunk_overlap=20)
    # 总长短于块长时合并为一个块，但段落不得被从中间切断
    assert len(chunks) == 1
    for paragraph in ("甲段落。", "乙段落。", "丙段落。"):
        assert paragraph in chunks[0]


def test_overlap_carries_context():
    text = ("前置上下文信息。" * 6) + "关键句在这里出现。" + ("后续内容。" * 6)
    chunks = recursive_split(text, chunk_size=60, chunk_overlap=15)
    assert len(chunks) >= 2
    joined = "".join(chunks)
    assert "关键句在这里出现。" in joined


def test_invalid_params():
    with pytest.raises(ValueError):
        recursive_split("x", chunk_size=0)
    with pytest.raises(ValueError):
        recursive_split("x", chunk_size=10, chunk_overlap=10)
