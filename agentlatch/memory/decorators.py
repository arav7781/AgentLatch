"""Memory-aware decorators — ``@context_aware`` and ``@intent``.

These decorators layer on top of (or alongside) ``@safe_tool`` to give
tools structured, evolving memory in multi-step agent workflows.

Decorator stacking order (outermost → innermost)::

    @intent("external_api_query")
    @context_aware(delta=True)
    @safe_tool(timeout=5.0)
    async def search_api(query: str) -> str:
        ...

``@intent`` sets the intent tag in context for the duration of the call.
``@context_aware`` creates a memory snapshot after each successful execution.
``@safe_tool`` handles error interception and timing (unchanged).
"""

from __future__ import annotations

import functools
import inspect
import json
import time
from collections.abc import Callable
from typing import Any, TypeVar, overload

from agentlatch._types import EventStatus, MemorySnapshot
from agentlatch.memory.context import (
    get_agent_id,
    get_intent,
    get_memory,
    get_node_context,
    get_session_id,
    reset_intent,
    set_intent,
)
from agentlatch.tracker import end_child, get_trace, start_child

F = TypeVar("F", bound=Callable[..., Any])


# =====================================================================
# @intent
# =====================================================================


@overload
def intent(tag: str) -> Callable[[F], F]: ...


@overload
def intent(tag: Callable[..., Any]) -> Callable[..., Any]: ...


def intent(tag: str | Callable[..., Any]) -> Callable[[F], F] | Callable[..., Any]:
    """Decorator that tags a function with an intent label.

    The intent is stored in a ``ContextVar`` and is automatically picked
    up by ``@context_aware`` when creating memory snapshots.

    Usage::

        @intent("database_query")
        @safe_tool
        def query_db(sql: str) -> str:
            ...

    Can also be used imperatively inside a LangGraph node::

        from agentlatch.memory import set_intent
        set_intent("research_phase")
    """

    def decorator(fn: F) -> F:
        intent_tag = tag if isinstance(tag, str) else fn.__name__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = set_intent(intent_tag)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    reset_intent(token)

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = set_intent(intent_tag)
                try:
                    return fn(*args, **kwargs)
                finally:
                    reset_intent(token)

            return sync_wrapper  # type: ignore[return-value]

    # Support @intent (bare — uses function name) and @intent("tag")
    if callable(tag):
        fn = tag
        tag_str = fn.__name__
        return decorator(fn)  # type: ignore[arg-type]

    return decorator  # type: ignore[return-value]


# =====================================================================
# @context_aware
# =====================================================================


@overload
def context_aware(func: F) -> F: ...


@overload
def context_aware(
    *,
    delta: bool = ...,
    progressive: bool = ...,
    summary_fn: Callable[[Any], Any] | None = ...,
) -> Callable[[F], F]: ...


def context_aware(
    func: F | None = None,
    *,
    delta: bool = False,
    progressive: bool = False,
    summary_fn: Callable[[Any], Any] | None = None,
) -> F | Callable[[F], F]:
    """Decorator that adds structured memory to a tool function.

    After each successful call, a ``MemorySnapshot`` is stored in the
    active memory backend.  Downstream nodes and sub-agents can then
    query relevant history without re-executing expensive operations.

    Can be used bare (``@context_aware``) or with options
    (``@context_aware(delta=True, progressive=True)``).

    Args:
        delta:        If ``True``, store only the difference from the
                      last output (same tool + intent).  Reduces storage
                      and token cost for repeated queries.
        progressive:  If ``True``, store the full output in memory but
                      return only a lightweight summary/reference to the
                      caller.  Useful for large payloads.
        summary_fn:   Custom function to produce the summary when
                      *progressive* is enabled.  Defaults to a built-in
                      that returns type + size information.
    """

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = await fn(*args, **kwargs)
                _record_snapshot(
                    fn.__name__, args, kwargs, result,
                    delta=delta, progressive=progressive,
                    summary_fn=summary_fn,
                )
                if progressive:
                    return _make_progressive_summary(
                        fn.__name__, result, summary_fn
                    )
                return result

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)
                _record_snapshot(
                    fn.__name__, args, kwargs, result,
                    delta=delta, progressive=progressive,
                    summary_fn=summary_fn,
                )
                if progressive:
                    return _make_progressive_summary(
                        fn.__name__, result, summary_fn
                    )
                return result

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator  # type: ignore[return-value]


# =====================================================================
# Internal Helpers
# =====================================================================


def _summarize_input(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Create a lightweight summary of function arguments."""
    summary: dict[str, Any] = {}

    if args:
        summarized_args = []
        for a in args:
            summarized_args.append(_summarize_value(a))
        summary["args"] = summarized_args

    if kwargs:
        summary["kwargs"] = {k: _summarize_value(v) for k, v in kwargs.items()}

    return summary


def _summarize_value(value: Any) -> Any:
    """Summarize a single value for storage."""
    if isinstance(value, str):
        if len(value) > 200:
            return f"{value[:200]}... [{len(value)} chars]"
        return value
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, dict):
        return f"<dict with {len(value)} keys>"
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} with {len(value)} items>"
    return f"<{type(value).__name__}>"


def _summarize_output(result: Any) -> Any:
    """Create a storable summary of the function output."""
    if result is None:
        return None
    if isinstance(result, str):
        # Try to parse as JSON for structured storage.
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return {k: _summarize_value(v) for k, v in parsed.items()}
            if isinstance(parsed, list):
                return f"<JSON list with {len(parsed)} items>"
            return parsed
        except (json.JSONDecodeError, TypeError):
            if len(result) > 500:
                return f"{result[:500]}... [{len(result)} chars]"
            return result
    if isinstance(result, dict):
        return {k: _summarize_value(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return f"<{type(result).__name__} with {len(result)} items>"
    if isinstance(result, (int, float, bool)):
        return result
    return f"<{type(result).__name__}>"


def _record_snapshot(
    tool_name: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    *,
    delta: bool,
    progressive: bool,
    summary_fn: Callable[[Any], Any] | None,
) -> None:
    """Store a MemorySnapshot in the active backend (if any)."""
    memory = get_memory()
    if memory is None:
        return

    # Record a MEMORY_OP trace event if tracing is active.
    trace_active = get_trace() is not None
    event = start_child(f"memory:{tool_name}") if trace_active else None

    try:
        output_summary = _summarize_output(result)

        # Delta computation.
        delta_data = None
        if delta:
            previous = memory.get_last_snapshot(
                tool_name, intent=get_intent()
            )
            if previous is not None:
                delta_data = memory.compute_delta(
                    previous.get("output_summary"), output_summary
                )
                if delta_data is not None:
                    # Store delta instead of full output.
                    output_summary = delta_data

        snapshot = MemorySnapshot(
            tool_name=tool_name,
            intent=get_intent(),
            input_summary=_summarize_input(args, kwargs),
            output_summary=output_summary,
            timestamp=time.time(),
            node_context=get_node_context(),
            status="success",
            delta=delta_data,
            agent_id=get_agent_id(),
            session_id=get_session_id(),
        )

        memory.store(snapshot)

        if event:
            end_child(event, EventStatus.MEMORY_OP)

    except Exception:
        # Memory operations must never break the tool call.
        if event:
            end_child(event, EventStatus.ERROR)


def _make_progressive_summary(
    tool_name: str,
    result: Any,
    summary_fn: Callable[[Any], Any] | None,
) -> Any:
    """Return a lightweight reference instead of the full result."""
    if summary_fn is not None:
        return summary_fn(result)

    # Default progressive summary.
    if isinstance(result, str):
        return json.dumps({
            "_agentlatch_ref": True,
            "tool": tool_name,
            "type": "string",
            "length": len(result),
            "preview": result[:200] if len(result) > 200 else result,
            "hint": "Full result stored in AgentLatch memory. "
                    "Query with intent or tool_name to retrieve.",
        })

    if isinstance(result, dict):
        return {
            "_agentlatch_ref": True,
            "tool": tool_name,
            "type": "dict",
            "keys": list(result.keys()),
            "hint": "Full result stored in AgentLatch memory.",
        }

    if isinstance(result, (list, tuple)):
        return {
            "_agentlatch_ref": True,
            "tool": tool_name,
            "type": type(result).__name__,
            "length": len(result),
            "hint": "Full result stored in AgentLatch memory.",
        }

    # Fallback — return as-is.
    return result
