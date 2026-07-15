"""Decorators that form AgentLatch's public API.

* ``@safe_tool``     — wraps tool functions with error interception & timing.
* ``@profile_agent`` — wraps the outer agent loop with tracing & visualization.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import json
from collections.abc import Callable
from typing import Any, TypeVar, overload

from agentlatch._types import ErrorPayload, EventStatus
from agentlatch.sampler import sample_response
from agentlatch.tracker import (
    end_child,
    finalize_trace,
    get_trace,
    init_trace,
    start_child,
)

F = TypeVar("F", bound=Callable[..., Any])

# =====================================================================
# @safe_tool
# =====================================================================


def _build_error_payload(exc: Exception) -> ErrorPayload:
    """Translate a raw Python exception into an LLM-friendly JSON dict."""
    return {
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
        "instruction": (
            "The tool execution failed. Review your parameters and "
            "retry with corrected inputs."
        ),
    }


def _build_timeout_payload(name: str, timeout: float) -> ErrorPayload:
    """Payload returned when a tool exceeds its allowed time budget."""
    return {
        "status": "error",
        "error_type": "TimeoutError",
        "message": f"Tool '{name}' exceeded the {timeout}s timeout.",
        "instruction": (
            "The tool timed out. Consider simplifying the request or "
            "breaking it into smaller steps."
        ),
    }


@overload
def safe_tool(func: F) -> F: ...


@overload
def safe_tool(
    *,
    timeout: float | None = ...,
    on_fail: str = ...,
    max_response_tokens: int | None = ...,
    sample_rows: int | None = ...,
) -> Callable[[F], F]: ...


def safe_tool(
    func: F | None = None,
    *,
    timeout: float | None = None,
    on_fail: str = "instruct_llm",
    max_response_tokens: int | None = None,
    sample_rows: int | None = None,
) -> F | Callable[[F], F]:
    """Decorator that makes a tool function resilient and observable.

    Can be used bare (``@safe_tool``) or with arguments
    (``@safe_tool(timeout=5.0, sample_rows=10)``).

    On exception the decorated function returns a JSON error string to the
    caller (typically the LLM) instead of raising, preventing silent crashes.

    Args:
        timeout:              Optional wall-clock budget in seconds.
        on_fail:              Error strategy (currently ``"instruct_llm"``).
        max_response_tokens:  Approximate token ceiling for responses.
        sample_rows:          If the response contains a list, keep only
                              the first N elements.
    """

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # If no active trace, still protect — just skip timing.
                trace_active = get_trace() is not None
                event = start_child(fn.__name__) if trace_active else None

                try:
                    if timeout is not None:
                        result = await asyncio.wait_for(
                            fn(*args, **kwargs), timeout=timeout
                        )
                    else:
                        result = await fn(*args, **kwargs)
                except asyncio.TimeoutError:
                    payload = _build_timeout_payload(fn.__name__, timeout or 0.0)
                    if event:
                        end_child(event, EventStatus.TIMEOUT, payload)
                    return json.dumps(payload)
                except Exception as exc:
                    payload = _build_error_payload(exc)
                    if event:
                        end_child(event, EventStatus.ERROR, payload)
                    return json.dumps(payload)
                else:
                    result = sample_response(
                        result,
                        max_tokens=max_response_tokens,
                        sample_rows=sample_rows,
                    )
                    if event:
                        end_child(event, EventStatus.SUCCESS)
                    return result

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace_active = get_trace() is not None
                event = start_child(fn.__name__) if trace_active else None

                try:
                    if timeout is not None:
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as pool:
                            future = pool.submit(fn, *args, **kwargs)
                            result = future.result(timeout=timeout)
                    else:
                        result = fn(*args, **kwargs)
                except concurrent.futures.TimeoutError:
                    payload = _build_timeout_payload(fn.__name__, timeout or 0.0)
                    if event:
                        end_child(event, EventStatus.TIMEOUT, payload)
                    return json.dumps(payload)
                except Exception as exc:
                    payload = _build_error_payload(exc)
                    if event:
                        end_child(event, EventStatus.ERROR, payload)
                    return json.dumps(payload)
                else:
                    result = sample_response(
                        result,
                        max_tokens=max_response_tokens,
                        sample_rows=sample_rows,
                    )
                    if event:
                        end_child(event, EventStatus.SUCCESS)
                    return result

            return sync_wrapper  # type: ignore[return-value]

    # Handle bare @safe_tool (no parentheses) vs @safe_tool(timeout=5)
    if func is not None:
        return decorator(func)
    return decorator  # type: ignore[return-value]


# =====================================================================
# @profile_agent
# =====================================================================


def profile_agent(
    func: F | None = None,
    *,
    name: str | None = None,
) -> F | Callable[[F], F]:
    """Decorator for the main agent loop.

    Initializes the trace context, runs the agent, then renders the
    execution flamegraph to the terminal.

    Can be used bare (``@profile_agent``) or with a custom name
    (``@profile_agent(name="MyAgent")``).
    """

    def decorator(fn: F) -> F:
        label = name or fn.__name__

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Import here to avoid circular imports at module level.
                from agentlatch.banner import initialize_latch
                from agentlatch.config import is_dev_mode
                from agentlatch.renderer import render_flamegraph

                if is_dev_mode():
                    initialize_latch()
                init_trace(label)

                try:
                    result = await fn(*args, **kwargs)
                finally:
                    trace = finalize_trace()
                    if is_dev_mode():
                        render_flamegraph(trace)

                return result

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                from agentlatch.banner import initialize_latch
                from agentlatch.config import is_dev_mode
                from agentlatch.renderer import render_flamegraph

                if is_dev_mode():
                    initialize_latch()
                init_trace(label)

                try:
                    result = fn(*args, **kwargs)
                finally:
                    trace = finalize_trace()
                    if is_dev_mode():
                        render_flamegraph(trace)

                return result

            return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator  # type: ignore[return-value]
