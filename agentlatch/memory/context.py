"""Context-variable-based memory state management.

Mirrors the pattern used in ``agentlatch.tracker`` — all memory state is
propagated via ``contextvars`` so callers never need to thread a memory
handle through their function signatures.

Context variables:
    _active_memory   — the current ``MemoryBackend`` instance (or None).
    _active_intent   — the current intent tag (set by ``@intent``).
    _active_node     — the current DAG node label (for LangGraph integration).
    _active_agent_id — the current agent identifier (for multi-agent systems).
    _session_id      — persistent session ID across the entire pipeline run.
"""

from __future__ import annotations

import contextvars
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentlatch.memory.backend import MemoryBackend

# ---------------------------------------------------------------------------
# Context Variables
# ---------------------------------------------------------------------------

_active_memory: contextvars.ContextVar[MemoryBackend | None] = contextvars.ContextVar(
    "_active_memory", default=None
)

_active_intent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_intent", default=None
)

_active_node: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_node", default=None
)

_active_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_active_agent_id", default=None
)

_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_session_id", default=None
)


# ---------------------------------------------------------------------------
# Public Helpers — Memory Backend
# ---------------------------------------------------------------------------


def init_memory(backend: MemoryBackend | None = None) -> MemoryBackend:
    """Initialize the memory backend for the current context.

    If *backend* is ``None``, a default ``SQLiteBackend`` is created with
    an in-memory database (suitable for single-run agents).

    Returns the active backend instance.
    """
    if backend is None:
        from agentlatch.memory.sqlite_backend import SQLiteBackend

        backend = SQLiteBackend()

    _active_memory.set(backend)

    # Assign a session ID if not already set.
    if _session_id.get() is None:
        _session_id.set(str(uuid.uuid4()))

    return backend


def get_memory() -> MemoryBackend | None:
    """Return the active memory backend, or ``None`` if not initialized."""
    return _active_memory.get()


# ---------------------------------------------------------------------------
# Public Helpers — Intent
# ---------------------------------------------------------------------------


def set_intent(intent: str) -> contextvars.Token[str | None]:
    """Set the active intent tag and return a reset token."""
    return _active_intent.set(intent)


def get_intent() -> str | None:
    """Return the current intent tag, or ``None``."""
    return _active_intent.get()


def reset_intent(token: contextvars.Token[str | None]) -> None:
    """Restore the intent to its previous value using *token*."""
    _active_intent.reset(token)


# ---------------------------------------------------------------------------
# Public Helpers — Node Context (for DAG / LangGraph)
# ---------------------------------------------------------------------------


def set_node_context(node: str) -> contextvars.Token[str | None]:
    """Set the active DAG node label and return a reset token."""
    return _active_node.set(node)


def get_node_context() -> str | None:
    """Return the current DAG node label, or ``None``."""
    return _active_node.get()


def reset_node_context(token: contextvars.Token[str | None]) -> None:
    """Restore the node context to its previous value."""
    _active_node.reset(token)


# ---------------------------------------------------------------------------
# Public Helpers — Agent ID (for multi-agent leader/sub-agent systems)
# ---------------------------------------------------------------------------


def set_agent_id(agent_id: str) -> contextvars.Token[str | None]:
    """Set the active agent identifier and return a reset token."""
    return _active_agent_id.set(agent_id)


def get_agent_id() -> str | None:
    """Return the current agent identifier, or ``None``."""
    return _active_agent_id.get()


def reset_agent_id(token: contextvars.Token[str | None]) -> None:
    """Restore the agent ID to its previous value."""
    _active_agent_id.reset(token)


# ---------------------------------------------------------------------------
# Public Helpers — Session
# ---------------------------------------------------------------------------


def get_session_id() -> str | None:
    """Return the current session ID, or ``None``."""
    return _session_id.get()


# ---------------------------------------------------------------------------
# Reset (for tests)
# ---------------------------------------------------------------------------


def reset_memory_context() -> None:
    """Clear all memory-related context variables.  Useful for test isolation."""
    _active_memory.set(None)
    _active_intent.set(None)
    _active_node.set(None)
    _active_agent_id.set(None)
    _session_id.set(None)
