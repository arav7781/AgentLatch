"""Shared types and enumerations for AgentLatch."""

from __future__ import annotations

import enum
from typing import Any, TypeAlias, TypedDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventStatus(enum.Enum):
    """Status of a traced execution event."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RETRY = "retry"
    MEMORY_OP = "memory_op"
    LEARNING = "learning"


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

ErrorPayload: TypeAlias = dict[str, Any]
"""Structured error information returned to the LLM instead of raising."""


# ---------------------------------------------------------------------------
# Memory Types
# ---------------------------------------------------------------------------


class MemorySnapshot(TypedDict, total=False):
    """Structured memory record created after a tool call.

    Captures the full context of a tool invocation so downstream nodes,
    sub-agents, or future calls to the same tool can retrieve relevant
    history without re-executing.
    """

    id: str
    tool_name: str
    intent: str | None
    input_summary: dict[str, Any]
    output_summary: Any
    timestamp: float
    node_context: str | None
    status: str
    delta: dict[str, Any] | None
    agent_id: str | None
    session_id: str | None


class ToolLearning(TypedDict, total=False):
    """Record of learned improvements after repeated tool failures.

    Stored in the memory backend so future invocations can benefit from
    past failure analysis.
    """

    id: str
    tool_name: str
    failure_count: int
    failure_patterns: list[dict[str, Any]]
    suggested_docstring: str | None
    suggested_params: dict[str, Any] | None
    correction_hints: list[str]
    timestamp: float
