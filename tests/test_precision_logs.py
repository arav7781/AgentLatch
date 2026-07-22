"""Tests for high-precision logging and microsecond timing in AgentLatch."""

from __future__ import annotations

import time

from agentlatch.renderer import _format_duration
from agentlatch.tracker import TraceEvent, init_trace, reset_context


def setup_function():
    reset_context()


def test_trace_event_precision_timestamps():
    """Test high-precision epoch, ISO, ms, and us properties of TraceEvent."""
    event = TraceEvent(name="test_event", start_time=time.monotonic(), start_timestamp=time.time())
    time.sleep(0.012)
    event.end_time = time.monotonic()
    event.end_timestamp = time.time()

    assert event.duration >= 0.010
    assert event.duration_ms >= 10.0
    assert event.duration_us >= 10000.0
    assert "Z" in event.start_time_iso
    assert "Z" in event.end_time_iso
    assert "T" in event.start_time_iso


def test_format_duration_precision():
    """Test microsecond and millisecond duration string formatting."""
    # Sub-millisecond (microseconds)
    us_str = _format_duration(0.000450, high_precision=True)
    assert us_str == "450.0µs"

    # Milliseconds
    ms_str = _format_duration(0.01234, high_precision=True)
    assert ms_str == "12.34ms"

    # Seconds
    s_str = _format_duration(1.23456, high_precision=True)
    assert s_str == "1.235s"


def test_init_trace_metadata():
    """Test init_trace with metadata dictionary."""
    root = init_trace("RootAgent", metadata={"env": "test"})
    assert root.name == "RootAgent"
    assert root.metadata == {"env": "test"}
    assert root.start_timestamp > 0
