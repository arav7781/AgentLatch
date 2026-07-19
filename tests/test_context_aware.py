"""Tests for @context_aware decorator."""

from __future__ import annotations

import asyncio
import json

import pytest

from agentlatch.banner import reset_banner
from agentlatch.decorators import context_aware, profile_agent, safe_tool
from agentlatch.memory.context import (
    init_memory,
    reset_memory_context,
)
from agentlatch.memory.sqlite_backend import SQLiteBackend
from agentlatch.tracker import reset_context


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset all state before/after each test."""
    reset_context()
    reset_memory_context()
    reset_banner()
    yield
    reset_context()
    reset_memory_context()
    reset_banner()


# ===================================================================
# @context_aware — sync
# ===================================================================


class TestContextAwareSync:
    def test_snapshot_created_on_success(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware
        def my_tool(x: int) -> int:
            return x * 2

        result = my_tool(5)
        assert result == 10

        snaps = backend.query(tool_name="my_tool")
        assert len(snaps) == 1
        assert snaps[0]["status"] == "success"

    def test_input_output_summarized(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        greet("Alice")

        snaps = backend.query(tool_name="greet")
        assert len(snaps) == 1
        assert snaps[0]["input_summary"] is not None
        assert snaps[0]["output_summary"] is not None

    def test_no_memory_backend_is_harmless(self):
        """context_aware should not crash when no memory is initialized."""

        @context_aware
        def no_memory_tool() -> str:
            return "fine"

        assert no_memory_tool() == "fine"

    def test_works_with_safe_tool(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware
        @safe_tool
        def protected_tool(x: int) -> int:
            return x + 1

        @profile_agent(enable_memory=False)
        def agent():
            return protected_tool(10)

        result = agent()
        assert result == 11

        snaps = backend.query(tool_name="protected_tool")
        assert len(snaps) == 1

    def test_delta_mode_stores_diff(self):
        backend = SQLiteBackend()
        init_memory(backend)

        call_count = 0

        @context_aware(delta=True)
        def changing_tool() -> dict:
            nonlocal call_count
            call_count += 1
            return {"count": call_count, "fixed": "value"}

        changing_tool()  # First call — full snapshot.
        changing_tool()  # Second call — should store delta.

        snaps = backend.query(tool_name="changing_tool", limit=2)
        assert len(snaps) == 2
        # Most recent snapshot should have a delta.
        latest = snaps[0]
        assert latest["delta"] is not None

    def test_progressive_mode_returns_reference(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware(progressive=True)
        def big_data_tool() -> str:
            return "A" * 1000

        result = big_data_tool()

        # Result should be a JSON reference.
        parsed = json.loads(result)
        assert parsed["_agentlatch_ref"] is True
        assert parsed["tool"] == "big_data_tool"
        assert "hint" in parsed

        # Full data should be in memory.
        snaps = backend.query(tool_name="big_data_tool")
        assert len(snaps) == 1

    def test_progressive_with_custom_summary_fn(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware(progressive=True, summary_fn=lambda r: f"length={len(r)}")
        def tool_with_custom_summary() -> str:
            return "hello world"

        result = tool_with_custom_summary()
        assert result == "length=11"

    def test_multiple_calls_create_multiple_snapshots(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware
        def multi_call_tool(x: int) -> int:
            return x

        for i in range(5):
            multi_call_tool(i)

        snaps = backend.query(tool_name="multi_call_tool", limit=10)
        assert len(snaps) == 5


# ===================================================================
# @context_aware — async
# ===================================================================


class TestContextAwareAsync:
    def test_async_snapshot_created(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware
        async def async_tool(x: int) -> int:
            return x * 3

        result = asyncio.run(async_tool(4))
        assert result == 12

        snaps = backend.query(tool_name="async_tool")
        assert len(snaps) == 1

    def test_async_progressive(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware(progressive=True)
        async def async_big_tool() -> dict:
            return {"data": list(range(100))}

        result = asyncio.run(async_big_tool())
        assert isinstance(result, dict)
        assert result["_agentlatch_ref"] is True

    def test_async_with_safe_tool(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @context_aware
        @safe_tool
        async def async_protected(x: int) -> int:
            return x + 10

        @profile_agent(enable_memory=False)
        async def agent():
            return await async_protected(5)

        result = asyncio.run(agent())
        assert result == 15

        snaps = backend.query(tool_name="async_protected")
        assert len(snaps) == 1


# ===================================================================
# @context_aware with @profile_agent integration
# ===================================================================


class TestContextAwareProfileIntegration:
    def test_profile_agent_auto_inits_memory(self):
        @context_aware
        @safe_tool
        def tool_in_profile() -> str:
            return "ok"

        @profile_agent
        def full_agent():
            return tool_in_profile()

        result = full_agent()
        assert result == "ok"
