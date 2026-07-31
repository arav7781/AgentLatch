"""Response sampling and compression for large tool outputs.

When a tool returns a massive JSON blob (e.g., 1 000 database rows), blindly
passing it to the LLM wastes tokens, increases latency, and risks
hallucination.  This module provides a ``sample_response`` function that
intelligently compresses responses before they re-enter the context window.
"""

from __future__ import annotations

import json
from typing import Any

# Keys that commonly hold list-typed data in tool responses.
_LIST_KEYS: tuple[str, ...] = ("rows", "results", "data", "items", "records", "entries")

# Rough chars-per-token ratio (conservative — 1 token ≈ 4 chars on average).
_CHARS_PER_TOKEN = 4


def _try_parse_json(raw: str) -> tuple[Any, bool]:
    """Attempt to parse *raw* as JSON. Returns ``(parsed, True)`` on success."""
    try:
        return json.loads(raw), True
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw, False


def _sample_list(data: list, limit: int) -> list:
    """Slice a list and append AgentLatch sampling metadata."""
    if len(data) <= limit:
        return data
    sampled = data[:limit]
    sampled.append(
        {
            "_agentlatch_sampled": True,
            "shown": limit,
            "total": len(data),
        }
    )
    return sampled


def _walk_and_sample(obj: Any, sample_rows: int) -> Any:
    """Walk a JSON-like structure and sample list values under known keys."""
    if isinstance(obj, list):
        return _sample_list(obj, sample_rows)

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if key in _LIST_KEYS and isinstance(value, list):
                result[key] = _sample_list(value, sample_rows)
            else:
                result[key] = value
        return result

    return obj


def _serialize(obj: Any) -> str:
    """Best-effort JSON serialization used only for measuring size."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _deep_sample(obj: Any, limit: int, depth: int = 6) -> Any:
    """Recursively slice every list, not just those under known keys.

    :func:`_walk_and_sample` only looks one level down and only under
    ``_LIST_KEYS``.  That is the right default — it is predictable and leaves
    unfamiliar shapes alone.  This is the fallback used once a payload is
    known to be over budget, where thoroughness matters more than restraint.
    """
    if depth <= 0:
        return obj
    if isinstance(obj, list):
        sliced = _sample_list(obj, limit)
        return [_deep_sample(item, limit, depth - 1) for item in sliced]
    if isinstance(obj, dict):
        return {k: _deep_sample(v, limit, depth - 1) for k, v in obj.items()}
    return obj


def _trim_leaves(obj: Any, max_chars: int, depth: int = 6) -> Any:
    """Truncate oversized string leaves while preserving the structure.

    Structure is the last thing to go: an LLM can work with a dict whose
    values are clipped, but not with a dict that has become a broken string.
    """
    if depth <= 0:
        return obj
    if isinstance(obj, str):
        return _truncate_string(obj, max_chars)
    if isinstance(obj, list):
        return [_trim_leaves(item, max_chars, depth - 1) for item in obj]
    if isinstance(obj, dict):
        return {k: _trim_leaves(v, max_chars, depth - 1) for k, v in obj.items()}
    return obj


def _fit_structured(obj: Any, max_tokens: int, sample_rows: int | None) -> Any:
    """Shrink a dict/list until its serialized form fits *max_tokens*.

    Applied only when the payload is already over budget.  Escalates in three
    stages, cheapest and least destructive first:

    1. Halve the row limit repeatedly, sampling every nested list.
    2. Trim long string leaves.
    3. Give up gracefully and return a marker object describing what was
       dropped — never an exception, and never a silently oversized payload.
    """
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(_serialize(obj)) <= max_chars:
        return obj

    rows = sample_rows if sample_rows and sample_rows > 0 else 50
    candidate = obj
    while rows >= 1:
        candidate = _deep_sample(obj, rows)
        if len(_serialize(candidate)) <= max_chars:
            return candidate
        rows //= 2

    # Structure survived but the leaves are too fat — clip them.
    leaf_budget = max(32, max_chars // 20)
    trimmed = _trim_leaves(candidate, leaf_budget)
    if len(_serialize(trimmed)) <= max_chars:
        return trimmed

    original = len(_serialize(obj))
    return {
        "_agentlatch_truncated": True,
        "reason": "Payload exceeded the token budget even after sampling.",
        "original_chars": original,
        "budget_chars": max_chars,
        "preview": _serialize(trimmed)[: max(0, max_chars - 200)],
    }


def _truncate_string(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars* with a descriptive marker."""
    if len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    marker = f"...[truncated — {remaining:,} chars remaining]"
    return text[:max_chars] + marker


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_response(
    raw: Any,
    *,
    max_tokens: int | None = None,
    sample_rows: int | None = None,
) -> Any:
    """Compress a tool response for LLM consumption.

    Args:
        raw:          The raw return value from the tool function.
        max_tokens:   Approximate token ceiling.  Strings exceeding
                      ``max_tokens * 4`` characters are truncated with a
                      trailing marker.  Dicts and lists are held to the same
                      budget but stay structured — they are sampled and
                      leaf-trimmed rather than stringified.
        sample_rows:  If the response contains a JSON list (at the root or
                      under common keys like ``rows``, ``results``, ``data``),
                      keep only the first *N* elements and append sampling
                      metadata.

    Returns:
        The processed response — same type as *raw* when possible.
    """
    if max_tokens is None and sample_rows is None:
        return raw  # Nothing to do.

    # --- Phase 1: Parse if string -------------------------------------------
    is_string = isinstance(raw, str)
    if is_string:
        parsed, was_json = _try_parse_json(raw)
    else:
        parsed = raw
        was_json = False

    # --- Phase 2: Row sampling ----------------------------------------------
    if sample_rows is not None:
        parsed = _walk_and_sample(parsed, sample_rows)

    # --- Phase 3: Re-serialize and truncate ---------------------------------
    if is_string:
        if was_json:
            serialized = json.dumps(parsed, ensure_ascii=False)
        else:
            serialized = parsed  # Plain text — stays as-is.

        if max_tokens is not None:
            max_chars = max_tokens * _CHARS_PER_TOKEN
            serialized = _truncate_string(serialized, max_chars)

        return serialized

    # Non-string path: keep the payload structured, but still hold it to the
    # token budget.  A dict of 1 000 rows overflows a context window exactly
    # as fast as the string form does.
    if max_tokens is not None:
        parsed = _fit_structured(parsed, max_tokens, sample_rows)
    return parsed
