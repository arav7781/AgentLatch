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
        max_tokens:   Approximate token ceiling.  If the serialized response
                      exceeds ``max_tokens * 4`` characters, it is truncated
                      with a trailing marker.
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

    # Non-string path: return the (potentially sampled) object directly.
    # Truncation only applies to serialized strings.
    return parsed
