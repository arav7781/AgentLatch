"""Tests for AgentLatch LangGraph state execution tracking and precision calculation."""

from __future__ import annotations

import asyncio
import time
from typing import TypedDict

from agentlatch import (
    calculate_state_execution,
    get_trace,
    log_state_execution,
    profile_agent,
    wrap_langgraph,
    wrap_state_node,
)
from agentlatch._types import EventStatus
from agentlatch.tracker import reset_context


class StateSchema(TypedDict, total=False):
    query: str
    documents: list[str]
    analysis: str
    final_output: str


def setup_function():
    reset_context()


def test_wrap_state_node_sync():
    """Test wrapping a sync state node function."""

    @profile_agent(name="TestAgent")
    def run():
        def retrieve_node(state: StateSchema) -> dict:
            time.sleep(0.01)
            return {"documents": ["doc1", "doc2"]}

        wrapped = wrap_state_node("retrieve", retrieve_node)
        initial_state: StateSchema = {"query": "test query"}
        result = wrapped(initial_state)
        assert result == {"documents": ["doc1", "doc2"]}

        trace = get_trace()
        assert trace is not None
        assert len(trace.children) == 1
        node_event = trace.children[0]
        assert node_event.name == "node:retrieve"
        assert node_event.status == EventStatus.STATE_NODE
        assert node_event.duration > 0
        assert node_event.duration_ms > 0
        assert node_event.duration_us > 0
        assert node_event.metadata["input_keys"] == ["query"]
        assert node_event.metadata["output_keys"] == ["documents"]

    run()


def test_wrap_state_node_async():
    """Test wrapping an async state node function."""

    @profile_agent(name="AsyncAgent")
    async def run():
        async def async_node(state: StateSchema) -> dict:
            await asyncio.sleep(0.01)
            return {"analysis": "analyzed"}

        wrapped = wrap_state_node("async_analyze", async_node)
        result = await wrapped({"query": "q"})
        assert result == {"analysis": "analyzed"}

    asyncio.run(run())


def test_wrap_langgraph_mock_graph():
    """Test wrap_langgraph with a mock StateGraph structure."""

    class MockStateGraph:
        def __init__(self):
            self.nodes = {}

        def add_node(self, name, action):
            self.nodes[name] = action

        def compile(self):
            return self

        def invoke(self, input_state):
            state = dict(input_state)
            for _name, action in self.nodes.items():
                res = action(state)
                if isinstance(res, dict):
                    state.update(res)
            return state

    graph = MockStateGraph()
    graph.add_node("step1", lambda s: {"documents": ["a"]})
    graph.add_node("step2", lambda s: {"analysis": "done"})

    wrapped_graph = wrap_langgraph(graph)

    @profile_agent(name="GraphAgent")
    def run_pipeline():
        res = wrapped_graph.invoke({"query": "hello"})
        assert res["documents"] == ["a"]
        assert res["analysis"] == "done"

        metrics = calculate_state_execution()
        assert metrics["total_state_nodes_executed"] == 2
        assert "step1" in metrics["per_state_metrics"]
        assert "step2" in metrics["per_state_metrics"]
        assert metrics["per_state_metrics"]["step1"]["count"] == 1
        assert metrics["per_state_metrics"]["step2"]["count"] == 1
        assert metrics["per_state_metrics"]["step1"]["state_keys_modified"] == [
            "documents"
        ]
        assert metrics["per_state_metrics"]["step2"]["state_keys_modified"] == [
            "analysis"
        ]
        assert len(metrics["transitions"]) == 3  # START -> step1 -> step2 -> END

    run_pipeline()


def test_log_state_execution(capsys):
    """Test that log_state_execution formats state metrics cleanly."""
    metrics = {
        "graph_name": "TestGraph",
        "total_graph_duration_sec": 0.05,
        "total_state_nodes_executed": 2,
        "per_state_metrics": {
            "node_a": {
                "count": 1,
                "total_duration_sec": 0.02,
                "avg_duration_sec": 0.02,
                "min_duration_sec": 0.02,
                "max_duration_sec": 0.02,
                "percentage_of_graph": 40.0,
                "state_keys_modified": ["docs"],
            },
            "node_b": {
                "count": 1,
                "total_duration_sec": 0.03,
                "avg_duration_sec": 0.03,
                "min_duration_sec": 0.03,
                "max_duration_sec": 0.03,
                "percentage_of_graph": 60.0,
                "state_keys_modified": ["output"],
            },
        },
        "transitions": [
            {
                "from_state": "START",
                "to_state": "node_a",
                "duration_sec": 0.02,
                "timestamp_iso": "2026-07-22T00:00:00.000Z",
            },
            {
                "from_state": "node_a",
                "to_state": "node_b",
                "duration_sec": 0.03,
                "timestamp_iso": "2026-07-22T00:00:00.020Z",
            },
            {
                "from_state": "node_b",
                "to_state": "END",
                "duration_sec": 0.0,
                "timestamp_iso": "2026-07-22T00:00:00.050Z",
            },
        ],
        "state_logs": [
            {
                "node_name": "node_a",
                "start_time_iso": "2026-07-22T00:00:00.000Z",
                "end_time_iso": "2026-07-22T00:00:00.020Z",
                "duration_ms": 20.0,
                "duration_us": 20000.0,
                "status": "state_node",
                "state_input_keys": ["query"],
                "state_output_keys": ["query", "docs"],
                "delta_keys": ["docs"],
            }
        ],
    }

    log_state_execution(metrics, print_console=True)
    captured = capsys.readouterr()
    assert "AgentLatch LangGraph State Breakdown" in captured.out
    assert "node_a" in captured.out
    assert "node_b" in captured.out
    assert "State Trajectory: START ➔ node_a ➔ node_b ➔ END" in captured.out


def test_unparsed_llm_function_error_detection():
    """Test that LLM unparsed raw function call strings (<function=...) are detected as errors."""

    @profile_agent(name="HallucinationAgent")
    def run():
        def buggy_llm_node(state: StateSchema) -> dict:
            return {
                "messages": ['<function=search_flights>{"airport": "BLR"}</function>']
            }

        wrapped = wrap_state_node("flight_specialist", buggy_llm_node)
        wrapped({"query": "book flight"})

        metrics = calculate_state_execution()
        assert metrics["per_state_metrics"]["flight_specialist"]["errors_count"] == 1
        assert (
            "LLMUnparsedToolCallError"
            in metrics["per_state_metrics"]["flight_specialist"]["error_details"][0]
        )

    run()
