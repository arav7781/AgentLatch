"""Tests for agentlatch.tracker — context engine and TraceEvent tree."""

from __future__ import annotations

import threading
import time

import pytest

from agentlatch._types import EventStatus
from agentlatch.tracker import (
    TraceEvent,
    end_child,
    finalize_trace,
    get_trace,
    init_trace,
    reset_context,
    start_child,
)


@pytest.fixture(autouse=True)
def _clean_context():
    """Ensure every test starts with a blank context."""
    reset_context()
    yield
    reset_context()


# -------------------------------------------------------------------
# TraceEvent basics
# -------------------------------------------------------------------

class TestTraceEvent:
    def test_duration_while_in_progress(self):
        ev = TraceEvent(name="wip", start_time=time.monotonic())
        assert ev.duration == 0.0

    def test_duration_after_completion(self):
        start = time.monotonic()
        ev = TraceEvent(name="done", start_time=start, end_time=start + 1.5)
        assert ev.duration == pytest.approx(1.5)


# -------------------------------------------------------------------
# init_trace
# -------------------------------------------------------------------

class TestInitTrace:
    def test_creates_root_event(self):
        root = init_trace("my_agent")
        assert root.name == "my_agent"
        assert root.depth == 0
        assert root.parent is None
        assert root.children == []
        assert root.end_time is None

    def test_sets_context_var(self):
        root = init_trace("agent")
        assert get_trace() is root


# -------------------------------------------------------------------
# start_child / end_child
# -------------------------------------------------------------------

class TestChildEvents:
    def test_appends_to_parent(self):
        root = init_trace("agent")
        child = start_child("tool_a")

        assert len(root.children) == 1
        assert root.children[0] is child
        assert child.parent is root
        assert child.depth == 1

    def test_nested_children(self):
        root = init_trace("agent")
        child = start_child("tool_a")
        grandchild = start_child("sub_call")

        assert len(child.children) == 1
        assert child.children[0] is grandchild
        assert grandchild.depth == 2
        assert grandchild.parent is child

    def test_end_child_pops_parent(self):
        root = init_trace("agent")
        child = start_child("tool_a")
        end_child(child, EventStatus.SUCCESS)

        # After ending child, next start_child should attach to root.
        child2 = start_child("tool_b")
        assert child2.parent is root
        assert len(root.children) == 2

    def test_end_child_records_status_and_payload(self):
        init_trace("agent")
        child = start_child("bad_tool")
        payload = {"status": "error", "error_type": "ValueError"}
        end_child(child, EventStatus.ERROR, payload)

        assert child.status == EventStatus.ERROR
        assert child.error_payload == payload
        assert child.end_time is not None

    def test_start_child_outside_scope_raises(self):
        # No init_trace called — should raise.
        with pytest.raises(RuntimeError, match="outside of an active"):
            start_child("orphan")


# -------------------------------------------------------------------
# finalize_trace
# -------------------------------------------------------------------

class TestFinalizeTrace:
    def test_stamps_end_time(self):
        root = init_trace("agent")
        time.sleep(0.01)  # tiny sleep so duration > 0
        result = finalize_trace()

        assert result is root
        assert result.end_time is not None
        assert result.duration > 0

    def test_clears_context(self):
        init_trace("agent")
        finalize_trace()
        assert get_trace() is None

    def test_finalize_without_trace_raises(self):
        with pytest.raises(RuntimeError, match="no active trace"):
            finalize_trace()


# -------------------------------------------------------------------
# Context isolation across threads
# -------------------------------------------------------------------

class TestContextIsolation:
    def test_threads_do_not_share_context(self):
        """Two threads each run their own trace — they must not interfere."""
        results: dict[str, TraceEvent] = {}
        barrier = threading.Barrier(2)

        def worker(name: str):
            root = init_trace(name)
            start_child(f"{name}_tool")
            barrier.wait()  # force interleaving
            time.sleep(0.01)
            end_child(root.children[0], EventStatus.SUCCESS)
            results[name] = finalize_trace()

        t1 = threading.Thread(target=worker, args=("agent_1",))
        t2 = threading.Thread(target=worker, args=("agent_2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["agent_1"].name == "agent_1"
        assert results["agent_2"].name == "agent_2"
        assert len(results["agent_1"].children) == 1
        assert results["agent_1"].children[0].name == "agent_1_tool"
        assert len(results["agent_2"].children) == 1
        assert results["agent_2"].children[0].name == "agent_2_tool"
