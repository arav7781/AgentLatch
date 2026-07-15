"""Tests for agentlatch.decorators — @safe_tool and @profile_agent."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from agentlatch.banner import reset_banner
from agentlatch.decorators import profile_agent, safe_tool
from agentlatch.tracker import get_trace, reset_context


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset context + banner before each test."""
    reset_context()
    reset_banner()
    yield
    reset_context()
    reset_banner()


# ===================================================================
# @safe_tool — sync
# ===================================================================


class TestSafeToolSync:
    def test_success_returns_normally(self):
        @safe_tool
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_catches_exception_returns_json(self):
        @safe_tool
        def explode():
            raise ValueError("bad input")

        result = explode()
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error_type"] == "ValueError"
        assert "bad input" in data["message"]
        assert "instruction" in data

    def test_error_payload_has_all_keys(self):
        @safe_tool
        def fail():
            raise RuntimeError("oops")

        data = json.loads(fail())
        required_keys = {"status", "error_type", "message", "instruction"}
        assert required_keys.issubset(data.keys())

    def test_preserves_functools_wraps(self):
        @safe_tool
        def my_tool():
            """Tool docstring."""
            return 42

        assert my_tool.__name__ == "my_tool"
        assert my_tool.__doc__ == "Tool docstring."

    def test_timeout_returns_timeout_error(self):
        @safe_tool(timeout=0.1)
        def slow_tool():
            time.sleep(5)
            return "done"

        result = slow_tool()
        data = json.loads(result)
        assert data["error_type"] == "TimeoutError"
        assert data["status"] == "error"

    def test_bare_decorator_and_parameterized_both_work(self):
        @safe_tool
        def bare():
            return "bare"

        @safe_tool(timeout=10.0)
        def parameterized():
            return "param"

        assert bare() == "bare"
        assert parameterized() == "param"


# ===================================================================
# @safe_tool — async
# ===================================================================


class TestSafeToolAsync:
    @pytest.mark.asyncio
    async def test_async_success(self):
        @safe_tool
        async def async_add(a: int, b: int) -> int:
            return a + b

        assert await async_add(1, 2) == 3

    @pytest.mark.asyncio
    async def test_async_catches_exception(self):
        @safe_tool
        async def async_fail():
            raise KeyError("missing")

        result = await async_fail()
        data = json.loads(result)
        assert data["error_type"] == "KeyError"

    @pytest.mark.asyncio
    async def test_async_timeout(self):
        @safe_tool(timeout=0.1)
        async def async_slow():
            await asyncio.sleep(10)

        result = await async_slow()
        data = json.loads(result)
        assert data["error_type"] == "TimeoutError"


# ===================================================================
# @profile_agent
# ===================================================================


class TestProfileAgent:
    def test_records_trace(self):
        @profile_agent
        def my_agent():
            return "result"

        result = my_agent()
        assert result == "result"
        # After profile_agent finishes, the trace is finalized and cleared.
        assert get_trace() is None

    def test_named_profile(self):
        @profile_agent(name="CustomAgent")
        def agent():
            return 42

        assert agent() == 42

    def test_nested_tools_produce_correct_tree(self, capsys):
        @safe_tool
        def tool_a():
            time.sleep(0.05)
            return "a"

        @safe_tool
        def tool_b():
            time.sleep(0.02)
            return "b"

        @profile_agent
        def agent():
            r1 = tool_a()
            r2 = tool_b()
            return f"{r1}+{r2}"

        result = agent()
        assert result == "a+b"

    def test_tool_failure_inside_profile(self):
        @safe_tool
        def bad_tool():
            raise RuntimeError("boom")

        @profile_agent
        def agent():
            err = bad_tool()
            return err  # returns JSON string, not crash

        result = agent()
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_async_profile_agent(self):
        @safe_tool
        async def async_tool():
            return "async_result"

        @profile_agent
        async def async_agent():
            return await async_tool()

        result = await async_agent()
        assert result == "async_result"
