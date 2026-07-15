"""Context-variable-based execution trace engine.

Builds a tree of ``TraceEvent`` nodes using Python ``contextvars`` so that
developers never need to thread a ``trace_id`` through their function calls.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field

from agentlatch._types import ErrorPayload, EventStatus

# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class TraceEvent:
    """A single node in the execution timeline tree.

    Attributes:
        name:          Human-readable label (usually the function name).
        start_time:    Monotonic timestamp when execution began.
        end_time:      Monotonic timestamp when execution finished (``None``
                       while the event is still in progress).
        status:        Final outcome — success, error, or timeout.
        error_payload: Structured error dict returned to the LLM on failure.
        children:      Nested tool calls executed *inside* this event.
        depth:         Nesting depth (0 = root).
        parent:        Back-reference to the parent event (``None`` for root).
    """

    name: str
    start_time: float
    end_time: float | None = None
    status: EventStatus = EventStatus.SUCCESS
    error_payload: ErrorPayload | None = None
    children: list[TraceEvent] = field(default_factory=list)
    depth: int = 0
    parent: TraceEvent | None = field(default=None, repr=False)

    # -- Convenience ---------------------------------------------------------

    @property
    def duration(self) -> float:
        """Wall-clock seconds elapsed.  Returns 0.0 while in-progress."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


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


def init_trace(name: str) -> TraceEvent:
    """Create the root trace event and bind it to the current context.

    This should be called **once** at the start of the ``@profile_agent``
    decorated function.
    """
    root = TraceEvent(name=name, start_time=time.monotonic(), depth=0)
    _active_trace.set(root)
    _current_parent.set(root)
    return root


def start_child(name: str) -> TraceEvent:
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
        depth=parent.depth + 1,
        parent=parent,
    )
    parent.children.append(child)
    _current_parent.set(child)
    return child


def end_child(
    event: TraceEvent,
    status: EventStatus = EventStatus.SUCCESS,
    error_payload: ErrorPayload | None = None,
) -> None:
    """Finalize a child event and pop back to its parent.

    Args:
        event:         The ``TraceEvent`` returned by :func:`start_child`.
        status:        The outcome of the operation.
        error_payload: Optional structured error dict for failed tools.
    """
    event.end_time = time.monotonic()
    event.status = status
    event.error_payload = error_payload
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
