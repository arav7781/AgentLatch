"""Tests for @intent decorator."""

from __future__ import annotations

import asyncio

import pytest

from agentlatch.banner import reset_banner
from agentlatch.decorators import context_aware, intent, safe_tool
from agentlatch.memory.context import (
    get_intent,
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
# @intent — sync
# ===================================================================


class TestIntentSync:
    def test_sets_intent_during_call(self):
        captured = []

        @intent("database_query")
        def tool_with_intent():
            captured.append(get_intent())
            return "done"

        tool_with_intent()
        assert captured[0] == "database_query"

    def test_clears_intent_after_call(self):
        @intent("temp_intent")
        def temp_tool():
            return "ok"

        temp_tool()
        assert get_intent() is None

    def test_restores_previous_intent(self):
        from agentlatch.memory.context import reset_intent, set_intent

        outer_token = set_intent("outer")

        @intent("inner")
        def inner_tool():
            assert get_intent() == "inner"
            return "inner_result"

        inner_tool()
        # After inner_tool completes, outer intent should be restored.
        assert get_intent() == "outer"
        reset_intent(outer_token)

    def test_intent_flows_to_memory_snapshot(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @intent("web_search")
        @context_aware
        def search_tool(query: str) -> str:
            return f"results for {query}"

        search_tool("python decorators")

        snaps = backend.query(tool_name="search_tool")
        assert len(snaps) == 1
        assert snaps[0]["intent"] == "web_search"

    def test_memory_query_filters_by_intent(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @intent("api_call")
        @context_aware
        def api_tool() -> str:
            return "api_result"

        @intent("db_call")
        @context_aware
        def db_tool() -> str:
            return "db_result"

        api_tool()
        db_tool()

        api_snaps = backend.query(intent="api_call")
        assert len(api_snaps) == 1
        assert api_snaps[0]["tool_name"] == "api_tool"

        db_snaps = backend.query(intent="db_call")
        assert len(db_snaps) == 1
        assert db_snaps[0]["tool_name"] == "db_tool"

    def test_preserves_function_metadata(self):
        @intent("test_tag")
        def documented_tool():
            """My docstring."""
            return 42

        assert documented_tool.__name__ == "documented_tool"
        assert documented_tool.__doc__ == "My docstring."

    def test_stacked_with_safe_tool(self):
        @intent("stack_test")
        @safe_tool
        def stacked_tool(x: int) -> int:
            return x * 2

        result = stacked_tool(5)
        assert result == 10


# ===================================================================
# @intent — async
# ===================================================================


class TestIntentAsync:
    def test_async_intent_sets_and_clears(self):
        captured = []

        @intent("async_intent")
        async def async_tool():
            captured.append(get_intent())
            return "async_done"

        asyncio.run(async_tool())
        assert captured[0] == "async_intent"
        assert get_intent() is None

    def test_async_intent_with_context_aware(self):
        backend = SQLiteBackend()
        init_memory(backend)

        @intent("async_search")
        @context_aware
        async def async_search(q: str) -> str:
            return f"found: {q}"

        asyncio.run(async_search("test"))

        snaps = backend.query(intent="async_search")
        assert len(snaps) == 1
        assert snaps[0]["tool_name"] == "async_search"


# ===================================================================
# Nested Intents
# ===================================================================


class TestNestedIntents:
    def test_nested_intents_restore_correctly(self):
        captured = []

        @intent("outer_intent")
        def outer():
            captured.append(("outer_before", get_intent()))
            inner()
            captured.append(("outer_after", get_intent()))

        @intent("inner_intent")
        def inner():
            captured.append(("inner", get_intent()))

        outer()

        assert captured[0] == ("outer_before", "outer_intent")
        assert captured[1] == ("inner", "inner_intent")
        assert captured[2] == ("outer_after", "outer_intent")
