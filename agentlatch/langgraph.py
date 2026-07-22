"""LangGraph State Execution Profiling & Precision Logging for AgentLatch.

Provides seamless wrapping of LangGraph StateGraph / CompiledStateGraph instances,
intercepting each state node execution with microsecond timing precision, state key
delta tracking, transition analysis, and performance metrics calculation.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, TypeVar, overload

from agentlatch._types import (
    EventStatus,
    PerStateMetric,
    StateExecutionMetrics,
    StateNodeLog,
    StateTransition,
)
from agentlatch.memory.context import set_node_context
from agentlatch.tracker import (
    TraceEvent,
    end_child,
    get_trace,
    init_trace,
    start_child,
)

logger = logging.getLogger("agentlatch.langgraph")

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------


def _extract_state_keys(state: Any) -> list[str]:
    """Extract dictionary keys from a LangGraph state object or return [] if not dict-like."""
    if isinstance(state, dict):
        return sorted(list(state.keys()))
    if hasattr(state, "__dict__"):
        return sorted(list(getattr(state, "__dict__").keys()))
    return []


def _compute_state_delta(input_state: Any, output_state: Any) -> list[str]:
    """Identify keys added or modified between input and output state snapshots."""
    if not isinstance(input_state, dict) or not isinstance(output_state, dict):
        if isinstance(output_state, dict):
            return sorted(list(output_state.keys()))
        return []

    modified: list[str] = []
    for k, v in output_state.items():
        if k not in input_state or input_state[k] != v:
            modified.append(k)
    return sorted(modified)


# ---------------------------------------------------------------------------
# Node Wrapper Function
# ---------------------------------------------------------------------------


def wrap_state_node(node_name: str, node_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a single LangGraph node function with high-precision AgentLatch execution profiling."""
    if inspect.iscoroutinefunction(node_fn):

        @functools.wraps(node_fn)
        async def async_node_wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            token = set_node_context(node_name)
            input_keys = _extract_state_keys(state)
            trace_active = get_trace() is not None

            event = (
                start_child(
                    f"node:{node_name}",
                    status=EventStatus.STATE_NODE,
                    metadata={"node_name": node_name, "is_state_node": True},
                )
                if trace_active
                else None
            )

            try:
                result = await node_fn(state, *args, **kwargs)
                output_state = result if isinstance(result, dict) else state
                delta_keys = _compute_state_delta(state, output_state)
                output_keys = (
                    _extract_state_keys(output_state)
                    if isinstance(output_state, dict)
                    else input_keys
                )

                if event:
                    end_child(
                        event,
                        status=EventStatus.STATE_NODE,
                        metadata={
                            "input_keys": input_keys,
                            "output_keys": output_keys,
                            "delta_keys": delta_keys,
                        },
                    )

                logger.debug(
                    f"[AgentLatch LangGraph] State Node '{node_name}' completed "
                    f"in {event.duration_ms:.2f}ms | Deltas: {delta_keys}"
                    if event
                    else f"[AgentLatch LangGraph] State Node '{node_name}' completed."
                )
                return result
            except Exception as exc:
                if event:
                    end_child(
                        event,
                        status=EventStatus.ERROR,
                        error_payload={
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                raise
            finally:
                set_node_context("")

        return async_node_wrapper
    else:

        @functools.wraps(node_fn)
        def sync_node_wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            token = set_node_context(node_name)
            input_keys = _extract_state_keys(state)
            trace_active = get_trace() is not None

            event = (
                start_child(
                    f"node:{node_name}",
                    status=EventStatus.STATE_NODE,
                    metadata={"node_name": node_name, "is_state_node": True},
                )
                if trace_active
                else None
            )

            try:
                result = node_fn(state, *args, **kwargs)
                output_state = result if isinstance(result, dict) else state
                delta_keys = _compute_state_delta(state, output_state)
                output_keys = (
                    _extract_state_keys(output_state)
                    if isinstance(output_state, dict)
                    else input_keys
                )

                if event:
                    end_child(
                        event,
                        status=EventStatus.STATE_NODE,
                        metadata={
                            "input_keys": input_keys,
                            "output_keys": output_keys,
                            "delta_keys": delta_keys,
                        },
                    )

                logger.debug(
                    f"[AgentLatch LangGraph] State Node '{node_name}' completed "
                    f"in {event.duration_ms:.2f}ms | Deltas: {delta_keys}"
                    if event
                    else f"[AgentLatch LangGraph] State Node '{node_name}' completed."
                )
                return result
            except Exception as exc:
                if event:
                    end_child(
                        event,
                        status=EventStatus.ERROR,
                        error_payload={
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                raise
            finally:
                set_node_context("")

        return sync_node_wrapper


# ---------------------------------------------------------------------------
# Graph & State Profiler Wrapper
# ---------------------------------------------------------------------------


def wrap_langgraph(graph: Any) -> Any:
    """Wrap a LangGraph StateGraph, CompiledStateGraph, or custom graph object.

    Automatically instruments each node in the graph with AgentLatch high-precision
    state execution timing, state key delta calculation, and transition tracing.

    Args:
        graph: LangGraph ``StateGraph`` or compiled graph instance.

    Returns:
        The instrumented graph object.
    """
    # Case 1: Uncompiled StateGraph — wrap add_node and compile
    if hasattr(graph, "add_node") and hasattr(graph, "nodes"):
        original_add_node = graph.add_node

        def wrapped_add_node(node_name: str, action: Any) -> Any:
            wrapped_action = wrap_state_node(node_name, action)
            return original_add_node(node_name, wrapped_action)

        graph.add_node = wrapped_add_node

        # Wrap existing nodes if already added
        if isinstance(graph.nodes, dict):
            for name, fn in list(graph.nodes.items()):
                if not getattr(fn, "_agentlatch_wrapped", False):
                    wrapped = wrap_state_node(name, fn)
                    setattr(wrapped, "_agentlatch_wrapped", True)
                    graph.nodes[name] = wrapped

        original_compile = getattr(graph, "compile", None)
        if original_compile:

            def wrapped_compile(*args: Any, **kwargs: Any) -> Any:
                compiled = original_compile(*args, **kwargs)
                return wrap_langgraph(compiled)

            graph.compile = wrapped_compile

    # Case 2: Compiled StateGraph / CompiledGraph / PregelNode
    if hasattr(graph, "nodes") and isinstance(graph.nodes, dict):
        for name, node in list(graph.nodes.items()):
            # 1) Official langgraph PregelNode with bound.func / bound.afunc
            if hasattr(node, "bound"):
                bound = getattr(node, "bound")
                target_attr = "func" if hasattr(bound, "func") else ("afunc" if hasattr(bound, "afunc") else None)
                if target_attr and callable(getattr(bound, target_attr)):
                    orig_fn = getattr(bound, target_attr)
                    if not getattr(orig_fn, "_agentlatch_wrapped", False):
                        wrapped_fn = wrap_state_node(name, orig_fn)
                        setattr(wrapped_fn, "_agentlatch_wrapped", True)
                        setattr(bound, target_attr, wrapped_fn)
            # 2) Objects with runnable attribute
            elif hasattr(node, "runnable") and callable(getattr(node, "runnable")):
                orig_fn = getattr(node, "runnable")
                if not getattr(orig_fn, "_agentlatch_wrapped", False):
                    wrapped_fn = wrap_state_node(name, orig_fn)
                    setattr(wrapped_fn, "_agentlatch_wrapped", True)
                    node.runnable = wrapped_fn
            # 3) Direct callables (e.g. MockStateGraph)
            elif callable(node) and not getattr(node, "_agentlatch_wrapped", False):
                wrapped_fn = wrap_state_node(name, node)
                setattr(wrapped_fn, "_agentlatch_wrapped", True)
                graph.nodes[name] = wrapped_fn

    return graph


# ---------------------------------------------------------------------------
# Calculation Engine
# ---------------------------------------------------------------------------


def calculate_state_execution(trace: TraceEvent | None = None) -> StateExecutionMetrics:
    """Calculate high-precision state execution metrics for all LangGraph nodes in a trace.

    Args:
        trace: The finalized root ``TraceEvent`` (defaults to active trace or empty).

    Returns:
        Structured ``StateExecutionMetrics`` dictionary.
    """
    root = trace or get_trace()
    graph_name = root.name if root else "LangGraphWorkflow"
    if root:
        if root.end_time is not None:
            total_graph_dur = root.duration
        else:
            total_graph_dur = max(0.000001, time.monotonic() - root.start_time)
    else:
        total_graph_dur = 0.0

    state_events: list[TraceEvent] = []

    def _collect_state_nodes(event: TraceEvent) -> None:
        if event.status == EventStatus.STATE_NODE or event.metadata.get("is_state_node"):
            state_events.append(event)
        for child in event.children:
            _collect_state_nodes(child)

    if root:
        _collect_state_nodes(root)

    # Sort state events by start timestamp
    state_events.sort(key=lambda e: e.start_timestamp)

    per_state_raw: dict[str, list[TraceEvent]] = {}
    state_logs: list[StateNodeLog] = []
    transitions: list[StateTransition] = []

    prev_node = "START"

    for ev in state_events:
        node_name = ev.metadata.get("node_name", ev.name.replace("node:", ""))
        if node_name not in per_state_raw:
            per_state_raw[node_name] = []
        per_state_raw[node_name].append(ev)

        # Transition record
        transitions.append(
            {
                "from_state": prev_node,
                "to_state": node_name,
                "duration_sec": ev.duration,
                "timestamp_iso": ev.start_time_iso,
            }
        )
        prev_node = node_name

        # State log entry
        state_logs.append(
            {
                "node_name": node_name,
                "start_time_iso": ev.start_time_iso,
                "end_time_iso": ev.end_time_iso,
                "duration_ms": round(ev.duration_ms, 3),
                "duration_us": round(ev.duration_us, 1),
                "status": ev.status.value,
                "state_input_keys": ev.metadata.get("input_keys", []),
                "state_output_keys": ev.metadata.get("output_keys", []),
                "delta_keys": ev.metadata.get("delta_keys", []),
            }
        )

    if prev_node != "START":
        transitions.append(
            {
                "from_state": prev_node,
                "to_state": "END",
                "duration_sec": 0.0,
                "timestamp_iso": root.end_time_iso if root else "",
            }
        )

    # Build per-state calculated breakdown
    per_state_metrics: dict[str, PerStateMetric] = {}
    for node_name, events in per_state_raw.items():
        durations = [e.duration for e in events]
        total_dur = sum(durations)
        count = len(events)
        avg_dur = total_dur / count if count > 0 else 0.0
        min_dur = min(durations) if durations else 0.0
        max_dur = max(durations) if durations else 0.0
        percentage = (total_dur / total_graph_dur * 100.0) if total_graph_dur > 0 else 0.0

        all_deltas: set[str] = set()
        for e in events:
            all_deltas.update(e.metadata.get("delta_keys", []))

        per_state_metrics[node_name] = {
            "count": count,
            "total_duration_sec": round(total_dur, 6),
            "avg_duration_sec": round(avg_dur, 6),
            "min_duration_sec": round(min_dur, 6),
            "max_duration_sec": round(max_dur, 6),
            "percentage_of_graph": round(percentage, 2),
            "state_keys_modified": sorted(list(all_deltas)),
        }

    return {
        "graph_name": graph_name,
        "total_graph_duration_sec": round(total_graph_dur, 6),
        "total_state_nodes_executed": len(state_events),
        "per_state_metrics": per_state_metrics,
        "transitions": transitions,
        "state_logs": state_logs,
    }


# ---------------------------------------------------------------------------
# High-Precision State Logger Output
# ---------------------------------------------------------------------------


def log_state_execution(
    metrics: StateExecutionMetrics,
    *,
    logger_instance: logging.Logger | None = None,
    print_console: bool = True,
) -> None:
    """Emit structured high-precision log output for LangGraph state executions.

    Args:
        metrics: The ``StateExecutionMetrics`` dict returned by ``calculate_state_execution``.
        logger_instance: Optional Logger instance to log to (defaults to module logger).
        print_console: If True, prints formatted state summary to stderr/stdout.
    """
    log = logger_instance or logger

    log.info(
        f"⚡ AgentLatch LangGraph Execution Profile [{metrics['graph_name']}] — "
        f"Total Duration: {metrics['total_graph_duration_sec'] * 1000:.2f}ms | "
        f"Nodes Executed: {metrics['total_state_nodes_executed']}"
    )

    for state_log in metrics["state_logs"]:
        log.info(
            f"  • Node [{state_log['node_name']}] | "
            f"Start: {state_log['start_time_iso']} | "
            f"Duration: {state_log['duration_ms']:.2f}ms ({state_log['duration_us']:.0f}µs) | "
            f"State Deltas: {state_log['delta_keys']}"
        )

    if print_console:
        print(f"\n⚡ AgentLatch LangGraph State Breakdown [{metrics['graph_name']}]")
        print("─" * 70)
        print(f"{'State Node':<20} {'Count':<8} {'Total (ms)':<12} {'Avg (ms)':<12} {'Graph %':<10}")
        print("─" * 70)
        for node_name, stat in metrics["per_state_metrics"].items():
            pct_str = f"{stat['percentage_of_graph']:.1f}%"
            print(
                f"{node_name:<20} "
                f"{stat['count']:<8} "
                f"{stat['total_duration_sec'] * 1000:<12.2f} "
                f"{stat['avg_duration_sec'] * 1000:<12.2f} "
                f"{pct_str:<10}"
            )
        print("─" * 70)
        if metrics["transitions"]:
            seq = " ➔ ".join(t["from_state"] for t in metrics["transitions"]) + f" ➔ {metrics['transitions'][-1]['to_state']}"
            print(f"🔄 State Trajectory: {seq}\n")
