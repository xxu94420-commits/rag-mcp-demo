"""Text chunking utilities.

Two-phase design:
1. ``_leaf_pieces`` splits text into pieces no longer than ``chunk_size``,
   preferring natural boundaries (paragraphs → sentences → words), with hard
   character cuts as the last resort.
2. ``_merge_with_overlap`` greedily merges consecutive pieces back into
   chunks, prefixing each chunk (after the first) with a tail overlap from the
   previous one. The merge budget accounts for the overlap, so no final chunk
   exceeds ``chunk_size``.
"""

from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " "]

MAX_CHUNK_CHARS_HARD_CAP = 10_000


def _split_once(text: str, separators: list[str]) -> list[str] | None:
    """Split text on the first separator that occurs; None if none occurs."""
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            # re-attach the separator to the end of each piece so Chinese
            # sentence punctuation survives inside chunks
            return [p + sep for p in parts[:-1]] + [parts[-1]]
    return None


def _leaf_pieces(
    text: str, chunk_size: int, separators: list[str]
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = _split_once(text, separators)
    if pieces is None:
        # no separator available: hard cuts, sized so that even after the
        # merge phase adds an overlap the chunk size invariant holds
        step = max(1, chunk_size // 2)
        return [
            text[start : start + step]
            for start in range(0, len(text), step)
        ]

    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= chunk_size:
            out.append(piece)
            continue
        # piece too large: recurse with the remaining (finer) separators
        idx = next(
            (i for i, sep in enumerate(separators) if sep in piece), -1
        )
        finer = separators[idx + 1 :] if idx >= 0 else []
        if not finer:
            step = max(1, chunk_size // 2)
            out.extend(
                piece[start : start + step]
                for start in range(0, len(piece), step)
            )
        else:
            out.extend(_leaf_pieces(piece, chunk_size, list(finer)))
    return out


def _merge_with_overlap(
    pieces: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    if not pieces:
        return []

    bodies: list[str] = []
    for i, piece in enumerate(pieces):
        budget = chunk_size if i == 0 else chunk_size - chunk_overlap
        if budget <= 0:
            raise ValueError("chunk_size must be larger than chunk_overlap")
        if bodies and len(bodies[-1]) + len(piece) <= budget:
            bodies[-1] += piece
        else:
            bodies.append(piece)

    chunks = [bodies[0]]
    for prev, body in zip(bodies, bodies[1:]):
        chunks.append(prev[-chunk_overlap:] + body)
    return [c for c in chunks if c.strip()]


def recursive_split(
    text: str,
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into overlapping chunks of at most ``chunk_size`` chars."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    if len(text) > MAX_CHUNK_CHARS_HARD_CAP:
        raise ValueError(f"single document too large ({len(text)} chars)")

    separators = separators or DEFAULT_SEPARATORS
    pieces = _leaf_pieces(text, chunk_size, list(separators))
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)
