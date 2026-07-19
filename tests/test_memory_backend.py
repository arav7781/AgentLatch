"""Tests for agentlatch.memory — SQLiteBackend and memory context."""

from __future__ import annotations

import time

import pytest

from agentlatch._types import MemorySnapshot, ToolLearning
from agentlatch.memory.context import (
    get_intent,
    get_memory,
    get_node_context,
    get_session_id,
    init_memory,
    reset_memory_context,
    set_intent,
    set_node_context,
)
from agentlatch.memory.sqlite_backend import SQLiteBackend


@pytest.fixture(autouse=True)
def _clean_memory():
    """Reset memory context before/after each test."""
    reset_memory_context()
    yield
    reset_memory_context()


# ===================================================================
# SQLiteBackend — CRUD
# ===================================================================


class TestSQLiteBackendCRUD:
    def test_store_and_query(self):
        backend = SQLiteBackend()
        snap = MemorySnapshot(
            tool_name="query_db",
            intent="database_query",
            input_summary={"args": ["SELECT * FROM users"]},
            output_summary={"rows": "<list with 5 items>"},
            timestamp=time.time(),
            status="success",
        )
        snap_id = backend.store(snap)
        assert snap_id

        results = backend.query(tool_name="query_db")
        assert len(results) == 1
        assert results[0]["tool_name"] == "query_db"
        assert results[0]["intent"] == "database_query"

    def test_query_by_intent(self):
        backend = SQLiteBackend()
        backend.store(MemorySnapshot(
            tool_name="search", intent="web_search",
            timestamp=time.time(), status="success",
        ))
        backend.store(MemorySnapshot(
            tool_name="search", intent="db_search",
            timestamp=time.time(), status="success",
        ))

        web = backend.query(intent="web_search")
        assert len(web) == 1
        assert web[0]["intent"] == "web_search"

        db = backend.query(intent="db_search")
        assert len(db) == 1

    def test_query_by_node_context(self):
        backend = SQLiteBackend()
        backend.store(MemorySnapshot(
            tool_name="tool_a", node_context="retriever_node",
            timestamp=time.time(), status="success",
        ))
        backend.store(MemorySnapshot(
            tool_name="tool_b", node_context="generator_node",
            timestamp=time.time(), status="success",
        ))

        results = backend.query(node_context="retriever_node")
        assert len(results) == 1
        assert results[0]["tool_name"] == "tool_a"

    def test_query_by_agent_id(self):
        backend = SQLiteBackend()
        backend.store(MemorySnapshot(
            tool_name="tool_x", agent_id="leader",
            timestamp=time.time(), status="success",
        ))
        backend.store(MemorySnapshot(
            tool_name="tool_y", agent_id="sub_agent_1",
            timestamp=time.time(), status="success",
        ))

        results = backend.query(agent_id="leader")
        assert len(results) == 1
        assert results[0]["tool_name"] == "tool_x"

    def test_query_limit(self):
        backend = SQLiteBackend()
        for i in range(20):
            backend.store(MemorySnapshot(
                tool_name="tool", timestamp=time.time() + i,
                status="success",
            ))

        results = backend.query(limit=5)
        assert len(results) == 5

    def test_query_order_is_newest_first(self):
        backend = SQLiteBackend()
        backend.store(MemorySnapshot(
            tool_name="old", timestamp=100.0, status="success",
        ))
        backend.store(MemorySnapshot(
            tool_name="new", timestamp=200.0, status="success",
        ))

        results = backend.query(limit=2)
        assert results[0]["tool_name"] == "new"
        assert results[1]["tool_name"] == "old"

    def test_get_last_snapshot(self):
        backend = SQLiteBackend()
        backend.store(MemorySnapshot(
            tool_name="search", intent="web",
            timestamp=100.0, status="success",
        ))
        backend.store(MemorySnapshot(
            tool_name="search", intent="web",
            timestamp=200.0, status="success",
        ))

        last = backend.get_last_snapshot("search", intent="web")
        assert last is not None
        assert last["timestamp"] == 200.0

    def test_get_last_snapshot_returns_none_when_empty(self):
        backend = SQLiteBackend()
        assert backend.get_last_snapshot("nonexistent") is None


# ===================================================================
# SQLiteBackend — Tool Learning
# ===================================================================


class TestSQLiteBackendLearning:
    def test_store_and_retrieve_learning(self):
        backend = SQLiteBackend()
        learning = ToolLearning(
            tool_name="query_db",
            failure_count=3,
            failure_patterns=[
                {"error_type": "ProgrammingError", "message": "column not found"}
            ],
            correction_hints=["Check column names against the schema"],
            timestamp=time.time(),
        )
        backend.store_learning("query_db", learning)

        results = backend.get_learnings("query_db")
        assert len(results) == 1
        assert results[0]["failure_count"] == 3
        assert len(results[0]["correction_hints"]) == 1

    def test_multiple_learnings(self):
        backend = SQLiteBackend()
        for i in range(3):
            backend.store_learning("tool", ToolLearning(
                tool_name="tool",
                failure_count=i + 1,
                timestamp=time.time() + i,
            ))

        results = backend.get_learnings("tool")
        assert len(results) == 3
        # Newest first.
        assert results[0]["failure_count"] == 3


# ===================================================================
# SQLiteBackend — Delta Computation
# ===================================================================


class TestSQLiteBackendDelta:
    def test_compute_delta_identical(self):
        backend = SQLiteBackend()
        assert backend.compute_delta({"a": 1}, {"a": 1}) is None

    def test_compute_delta_dict_change(self):
        backend = SQLiteBackend()
        delta = backend.compute_delta(
            {"a": 1, "b": 2},
            {"a": 1, "b": 3, "c": 4},
        )
        assert delta is not None
        assert "b" in delta
        assert "c" in delta
        assert "a" not in delta

    def test_compute_delta_non_dict(self):
        backend = SQLiteBackend()
        delta = backend.compute_delta("old_value", "new_value")
        assert delta == {"old": "old_value", "new": "new_value"}


# ===================================================================
# SQLiteBackend — Stats & Lifecycle
# ===================================================================


class TestSQLiteBackendStats:
    def test_stats(self):
        backend = SQLiteBackend()
        backend.store(MemorySnapshot(
            tool_name="t", timestamp=time.time(), status="success",
        ))
        stats = backend.stats()
        assert stats["backend"] == "sqlite"
        assert stats["snapshot_count"] == 1
        assert stats["learning_count"] == 0

    def test_close_is_safe(self):
        backend = SQLiteBackend()
        backend.close()
        backend.close()  # Double close should not raise.


# ===================================================================
# Memory Context
# ===================================================================


class TestMemoryContext:
    def test_init_memory_creates_sqlite_default(self):
        backend = init_memory()
        assert isinstance(backend, SQLiteBackend)
        assert get_memory() is backend

    def test_init_memory_with_custom_backend(self):
        custom = SQLiteBackend(":memory:")
        backend = init_memory(custom)
        assert backend is custom

    def test_session_id_auto_assigned(self):
        init_memory()
        assert get_session_id() is not None

    def test_intent_set_and_get(self):
        token = set_intent("database_query")
        assert get_intent() == "database_query"
        from agentlatch.memory.context import reset_intent
        reset_intent(token)
        assert get_intent() is None

    def test_node_context_set_and_get(self):
        token = set_node_context("retriever_node")
        assert get_node_context() == "retriever_node"
        from agentlatch.memory.context import reset_node_context
        reset_node_context(token)
        assert get_node_context() is None

    def test_reset_clears_everything(self):
        init_memory()
        set_intent("test")
        set_node_context("node")
        reset_memory_context()
        assert get_memory() is None
        assert get_intent() is None
        assert get_node_context() is None
        assert get_session_id() is None


# ===================================================================
# Backend Isolation
# ===================================================================


class TestBackendIsolation:
    def test_separate_backends_dont_share_data(self):
        backend_a = SQLiteBackend()
        backend_b = SQLiteBackend()

        backend_a.store(MemorySnapshot(
            tool_name="tool_a", timestamp=time.time(), status="success",
        ))
        backend_b.store(MemorySnapshot(
            tool_name="tool_b", timestamp=time.time(), status="success",
        ))

        assert len(backend_a.query(tool_name="tool_a")) == 1
        assert len(backend_a.query(tool_name="tool_b")) == 0
        assert len(backend_b.query(tool_name="tool_b")) == 1
        assert len(backend_b.query(tool_name="tool_a")) == 0
