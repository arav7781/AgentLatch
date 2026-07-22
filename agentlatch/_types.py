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
    STATE_NODE = "state_node"


# ---------------------------------------------------------------------------
# LangGraph & State Execution Types
# ---------------------------------------------------------------------------


class PerStateMetric(TypedDict):
    """Execution timing and invocation metrics for a specific LangGraph node/state."""

    count: int
    total_duration_sec: float
    avg_duration_sec: float
    min_duration_sec: float
    max_duration_sec: float
    percentage_of_graph: float
    state_keys_modified: list[str]
    errors_count: int
    error_details: list[str]


class StateTransition(TypedDict):
    """A recorded transition from one state/node to another in LangGraph."""

    from_state: str
    to_state: str
    duration_sec: float
    timestamp_iso: str


class StateNodeLog(TypedDict):
    """Structured high-precision log entry for a single state node execution."""

    node_name: str
    start_time_iso: str
    end_time_iso: str
    duration_ms: float
    duration_us: float
    status: str
    state_input_keys: list[str]
    state_output_keys: list[str]
    delta_keys: list[str]
    errors_count: int
    error_details: list[str]


class StateExecutionMetrics(TypedDict):
    """Calculated breakdown of all state node executions in a LangGraph graph trace."""

    graph_name: str
    total_graph_duration_sec: float
    total_state_nodes_executed: int
    per_state_metrics: dict[str, PerStateMetric]
    transitions: list[StateTransition]
    state_logs: list[StateNodeLog]



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
