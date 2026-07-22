"""Context-variable-based execution trace engine.

Builds a tree of ``TraceEvent`` nodes using Python ``contextvars`` so that
developers never need to thread a ``trace_id`` through their function calls.
"""

from __future__ import annotations

import contextvars
import time
import datetime
from dataclasses import dataclass, field
from typing import Any

from agentlatch._types import ErrorPayload, EventStatus

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


def _current_timestamp() -> float:
    return time.time()


def _format_iso(ts: float | None) -> str:
    if ts is None:
        return ""
    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class TraceEvent:
    """A single node in the execution timeline tree.

    Attributes:
        name:            Human-readable label (usually the function name).
        start_time:      Monotonic timestamp when execution began.
        end_time:        Monotonic timestamp when execution finished (``None``
                         while the event is still in progress).
        start_timestamp: High-precision epoch timestamp (seconds since Unix epoch).
        end_timestamp:   High-precision epoch timestamp when event finished.
        status:          Final outcome — success, error, timeout, or state_node.
        error_payload:   Structured error dict returned to the LLM on failure.
        metadata:        Arbitrary key-value metadata (state deltas, node context, etc.).
        children:        Nested tool calls or state node executions *inside* this event.
        depth:           Nesting depth (0 = root).
        parent:          Back-reference to the parent event (``None`` for root).
    """

    name: str
    start_time: float
    end_time: float | None = None
    start_timestamp: float = field(default_factory=_current_timestamp)
    end_timestamp: float | None = None
    status: EventStatus = EventStatus.SUCCESS
    error_payload: ErrorPayload | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[TraceEvent] = field(default_factory=list)
    depth: int = 0
    parent: TraceEvent | None = field(default=None, repr=False)

    # -- Convenience ---------------------------------------------------------

    @property
    def duration(self) -> float:
        """Wall-clock seconds elapsed. Returns 0.0 while in-progress."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000.0

    @property
    def duration_us(self) -> float:
        """Duration in microseconds."""
        return self.duration * 1000000.0

    @property
    def start_time_iso(self) -> str:
        """ISO-8601 formatted start timestamp with millisecond precision."""
        return _format_iso(self.start_timestamp)

    @property
    def end_time_iso(self) -> str:
        """ISO-8601 formatted end timestamp with millisecond precision."""
        return _format_iso(self.end_timestamp)


# ---------------------------------------------------------------------------
# Context Variables
# ---------------------------------------------------------------------------

_active_trace: contextvars.ContextVar[TraceEvent | None] = contextvars.ContextVar(
    "_active_trace", default=None
)
"""Root ``TraceEvent`` for the current agent execution."""

_current_parent: contextvars.ContextVar[TraceEvent | None] = contextvars.ContextVar(
    "_current_parent", default=None
)
"""The deepest open event — new children are appended here."""


# ---------------------------------------------------------------------------
# Public Helpers
# ---------------------------------------------------------------------------


def init_trace(name: str, metadata: dict[str, Any] | None = None) -> TraceEvent:
    """Create the root trace event and bind it to the current context.

    This should be called **once** at the start of the ``@profile_agent``
    decorated function.
    """
    root = TraceEvent(
        name=name,
        start_time=time.monotonic(),
        start_timestamp=time.time(),
        depth=0,
        metadata=metadata or {},
    )
    _active_trace.set(root)
    _current_parent.set(root)
    return root


def start_child(
    name: str,
    status: EventStatus = EventStatus.SUCCESS,
    metadata: dict[str, Any] | None = None,
) -> TraceEvent:
    """Open a new child event under the current parent.

    Returns the newly-created child so the caller can later pass it to
    :func:`end_child`.

    Raises:
        RuntimeError: If called outside of an active ``@profile_agent`` scope.
    """
    parent = _current_parent.get()
    if parent is None:
        raise RuntimeError(
            "start_child() called outside of an active @profile_agent scope. "
            "Ensure your agent loop is decorated with @profile_agent."
        )

    child = TraceEvent(
        name=name,
        start_time=time.monotonic(),
        start_timestamp=time.time(),
        status=status,
        metadata=metadata or {},
        depth=parent.depth + 1,
        parent=parent,
    )
    parent.children.append(child)
    _current_parent.set(child)
    return child


def end_child(
    event: TraceEvent,
    status: EventStatus | None = None,
    error_payload: ErrorPayload | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Finalize a child event and pop back to its parent.

    Args:
        event:         The ``TraceEvent`` returned by :func:`start_child`.
        status:        The outcome of the operation.
        error_payload: Optional structured error dict for failed tools.
        metadata:      Optional key-value pairs to update on the event.
    """
    event.end_time = time.monotonic()
    event.end_timestamp = time.time()
    if status is not None:
        event.status = status
    if error_payload is not None:
        event.error_payload = error_payload
    if metadata:
        event.metadata.update(metadata)
    # Pop back to the parent so the next start_child appends correctly.
    _current_parent.set(event.parent)


def finalize_trace() -> TraceEvent:
    """Stamp the root event's end time and return the complete tree.

    Raises:
        RuntimeError: If no active trace exists.
    """
    root = _active_trace.get()
    if root is None:
        raise RuntimeError("finalize_trace() called but no active trace exists.")
    root.end_time = time.monotonic()
    # Clean up context vars.
    _active_trace.set(None)
    _current_parent.set(None)
    return root


def get_trace() -> TraceEvent | None:
    """Return the current root trace, or ``None`` if not tracing."""
    return _active_trace.get()


def reset_context() -> None:
    """Reset all context variables.  Useful for test isolation."""
    _active_trace.set(None)
    _current_parent.set(None)
