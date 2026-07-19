# Phase 10 — Context-Aware Memory & Multi-Agent Pipeline

> Feature: **Memory & Recovery Pipeline** (Multi-Agent DAG & LangGraph Resilience).
> Status: **✅ Phase A Done (Core Memory + Decorators + SQLite) · ⏳ Phase B & C Planned**.
> Depends on: **Phase 2** (contextvars engine) and **Phase 3** (decorators).

---

## 1. Goal
Provide a complete, pluggable memory and recovery pipeline for complex agentic DAG workflows and multi-agent systems (especially LangGraph leader/sub-agent topologies).

The pipeline solves context rot in multi-step agent runs by capturing structured memory snapshots after tool execution, allowing sub-agents and downstream DAG nodes to query upstream memory by intent, tool name, node context, or agent ID.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P10-1** | `contextvars` for Memory State | Propagate `_active_memory`, `_active_intent`, `_active_node`, `_active_agent_id`, and `_session_id` seamlessly without parameter threading. |
| **D-P10-2** | Zero-Dependency Default (`SQLiteBackend`) | Built on standard library `sqlite3` with WAL mode, JSON columns, and indexed queries. Works out-of-the-box in-memory or persisted. |
| **D-P10-3** | Layered Decorator Composition | `@intent("tag")` sets context → `@context_aware` handles memory snapshots → `@safe_tool` handles resilience/timing. Each decorator remains focused. |
| **D-P10-4** | Non-blocking Memory Safety | Memory snapshot creation or query failures are wrapped in try-except so they never crash the tool execution itself. |
| **D-P10-5** | Pluggable Sub-Package Backends | External vector (pgvector, Qdrant) and graph (Neo4j) backends live as sub-packages gated by optional extras (`agentlatch[vector]`, etc.). |

---

## 3. Implementation Phases

### Phase A — Memory Core + `@context_aware` + `@intent` (Done)
- **`MemoryBackend` (Abstract Base Class):** Defines contract for `store`, `query`, `get_last_snapshot`, `compute_delta`, `store_learning`, `get_learnings`, and `close`.
- **`SQLiteBackend`:** Standard library SQLite implementation with automatic schema generation, indexing, and WAL mode.
- **`@intent("tag")`:** ContextVar-based tagger that marks the active intent for subsequent tool calls and memory snapshots.
- **`@context_aware`:**
  - `delta=True`: Computes diff between current output and last snapshot (same tool + intent), storing only changes.
  - `progressive=True`: Stores full output in memory and returns a lightweight JSON reference/summary to the caller.
- **`@profile_agent` Integration:** Automatically initializes the default or custom `MemoryBackend` on start and closes it on finalization.

### Phase B — Advanced Self-Healing & Recovery (Planned)
- **`RetryPolicy`:** Exponential backoff retry loop inside `@safe_tool`.
- **`@recovery_hook`:** Allows registering custom fallback functions when a tool fails repeatedly.
- **Tool Learning Mode:** Captures failure patterns after $N$ consecutive errors and stores suggested docstring/schema fixes in memory.

### Phase C — Pluggable Backends & Vector Extras (Planned)
- **`PostgresBackend`:** `psycopg` + `pgvector` for vector similarity memory queries.
- **`QdrantBackend`:** `qdrant-client` integration.
- **`Neo4jBackend`:** Graph database representation of agent tool interactions.

---

## 4. Data Flow (Multi-Agent DAG Memory)

```
 [ Leader Agent (@profile_agent) ] -> Initializes Trace + Memory Context
              │
              ├─── Node 1: Research (set_node_context("research_node"), set_agent_id("researcher"))
              │      └─> @intent("research") @context_aware @safe_tool search_docs()
              │            └─> Stores MemorySnapshot (tool="search_docs", intent="research")
              │
              ├─── Node 2: Analysis (set_node_context("analysis_node"), set_agent_id("analyst"))
              │      └─> Analyst queries MemoryBackend: query(intent="research")
              │      └─> @intent("analyze") @context_aware(delta=True) analyze_data()
              │            └─> Stores Delta MemorySnapshot (changes vs previous run)
              │
              └─── Node 3: Writer (set_node_context("writing_node"), set_agent_id("writer"))
                     └─> Writer queries MemoryBackend: query(limit=20)
                     └─> @intent("write") @context_aware(progressive=True) generate_report()
                           └─> Stores full report in Memory, returns lightweight reference
```

---

## 5. Safety, Isolation, & Correctness

- **Context Isolation:** Memory state uses `contextvars`, isolating concurrent async tasks and threads.
- **Graceful Fallback:** If `@context_aware` runs without an active memory backend (e.g. outside `@profile_agent`), it executes the underlying function safely without throwing errors.
- **Metadata Integrity:** All decorators use `@functools.wraps` to preserve function signatures and docstrings for LLM tool extraction.

---

## 6. Verification & Tests

Implemented in [`tests/test_memory_backend.py`](../../tests/test_memory_backend.py), [`tests/test_context_aware.py`](../../tests/test_context_aware.py), and [`tests/test_intent.py`](../../tests/test_intent.py):
- SQLite CRUD, indexing, and `get_last_snapshot`.
- ContextVar isolation across thread boundaries.
- Delta calculation and progressive summary rendering.
- Sync and async decorator execution paths.
- End-to-end multi-agent execution in [`examples/memory_langgraph_agent.py`](../../examples/memory_langgraph_agent.py).

---

## 7. Files Touched

| File | Change |
|---|---|
| [`agentlatch/_types.py`](../_types.py) | **[MODIFY]** Added `RETRY`, `MEMORY_OP`, `LEARNING` event statuses + `MemorySnapshot` / `ToolLearning` TypedDicts. |
| [`agentlatch/memory/__init__.py`](../memory/__init__.py) | **[NEW]** Package re-exports for memory subsystem. |
| [`agentlatch/memory/backend.py`](../memory/backend.py) | **[NEW]** Abstract `MemoryBackend` contract. |
| [`agentlatch/memory/context.py`](../memory/context.py) | **[NEW]** ContextVar state management for memory, intent, node, agent ID, session. |
| [`agentlatch/memory/sqlite_backend.py`](../memory/sqlite_backend.py) | **[NEW]** Zero-dependency `SQLiteBackend`. |
| [`agentlatch/memory/decorators.py`](../memory/decorators.py) | **[NEW]** `@context_aware` and `@intent` decorators. |
| [`agentlatch/decorators.py`](../decorators.py) | **[MODIFY]** Integrated memory initialization into `@profile_agent`. |
| [`agentlatch/__init__.py`](../__init__.py) | **[MODIFY]** Public API exports for memory symbols; version set to `0.2.0`. |
| [`agentlatch/renderer.py`](../renderer.py) | **[MODIFY]** Enhanced flamegraph with memory ops, retries, and legend. |
| [`pyproject.toml`](../../pyproject.toml) | **[MODIFY]** Version 0.2.0 + optional dependency extras for vector/graph backends. |

---

## 8. Acceptance Criteria

- Full suite of 106 unit tests passing.
- `@context_aware` captures inputs, outputs, timestamps, and intent without modifying return types (unless `progressive=True`).
- `delta=True` stores only diffs on subsequent calls.
- Flamegraph displays `🧠 MEMORY` spans and memory counts.
