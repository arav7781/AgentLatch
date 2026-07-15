"""Tests for agentlatch.renderer — flamegraph output."""

from __future__ import annotations

import time
from io import StringIO

from rich.console import Console

from agentlatch._types import EventStatus
from agentlatch.renderer import render_flamegraph
from agentlatch.tracker import TraceEvent


def _make_trace(
    name: str = "test_agent",
    duration: float = 2.0,
    children: list[TraceEvent] | None = None,
) -> TraceEvent:
    """Factory for test traces with controlled timing."""
    start = time.monotonic()
    root = TraceEvent(
        name=name,
        start_time=start,
        end_time=start + duration,
        children=children or [],
    )
    return root


def _make_child(
    name: str,
    parent_start: float,
    offset: float,
    duration: float,
    status: EventStatus = EventStatus.SUCCESS,
    error_payload: dict | None = None,
) -> TraceEvent:
    """Create a child event at a specific offset from parent start."""
    return TraceEvent(
        name=name,
        start_time=parent_start + offset,
        end_time=parent_start + offset + duration,
        depth=1,
        status=status,
        error_payload=error_payload,
    )


def _capture_render(trace: TraceEvent) -> str:
    """Render to a string buffer and return the output."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=80)
    render_flamegraph(trace, console=console)
    return buf.getvalue()


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestRenderFlamegraph:
    def test_no_crash_with_children(self):
        """Renderer must complete without exception for a normal trace."""
        start = time.monotonic()
        children = [
            _make_child("tool_a", start, 0.2, 0.5),
            _make_child("tool_b", start, 1.0, 0.3),
        ]
        trace = _make_trace(children=children)
        trace.start_time = start
        trace.end_time = start + 2.0

        output = _capture_render(trace)
        assert "AGENTLATCH EXECUTION PROFILE" in output
        assert "tool_a" in output
        assert "tool_b" in output

    def test_render_with_errors(self):
        """Error events must appear with red indicators."""
        start = time.monotonic()
        children = [
            _make_child(
                "bad_tool",
                start,
                0.1,
                0.3,
                status=EventStatus.ERROR,
                error_payload={
                    "status": "error",
                    "error_type": "ValueError",
                    "message": "bad input",
                },
            ),
        ]
        trace = _make_trace(children=children)
        trace.start_time = start
        trace.end_time = start + 1.0

        output = _capture_render(trace)
        assert "bad_tool" in output
        assert "ERROR" in output

    def test_render_empty_trace(self):
        """A trace with no children should render gracefully."""
        trace = _make_trace(children=[])
        output = _capture_render(trace)
        assert "No tool calls recorded" in output

    def test_render_timeout_event(self):
        """Timeout events must appear with timeout indicators."""
        start = time.monotonic()
        children = [
            _make_child(
                "slow_tool",
                start,
                0.0,
                5.0,
                status=EventStatus.TIMEOUT,
                error_payload={
                    "status": "error",
                    "error_type": "TimeoutError",
                    "message": "exceeded 5s",
                },
            ),
        ]
        trace = _make_trace(duration=5.0, children=children)
        trace.start_time = start
        trace.end_time = start + 5.0

        output = _capture_render(trace)
        assert "TIMEOUT" in output
        assert "slow_tool" in output

    def test_legend_present(self):
        """Output must contain a legend explaining the color scheme."""
        start = time.monotonic()
        children = [_make_child("x", start, 0.0, 0.5)]
        trace = _make_trace(children=children)
        trace.start_time = start
        trace.end_time = start + 1.0

        output = _capture_render(trace)
        assert "Legend" in output
        assert "LLM Reasoning" in output
