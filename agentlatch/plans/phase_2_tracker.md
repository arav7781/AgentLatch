# Phase 2 — Context & Timing Engine

> Feature: **Execution Tracking** (see [`INSTRUCTIONS.md`](../../INSTRUCTIONS.md) — Phase 2).
> Status: **✅ Done** · Depends on: **Phase 1** (core types).
> Written to document the ContextVar-based trace execution mechanism.

---

## 1. Goal
Implement a thread-safe, coroutine-safe execution tracing engine using Python's standard `contextvars`. This allows recording the duration and structure of nested tool calls inside an agent run without requiring developers to pass a trace ID or reference object through their entire code path.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P2-1** | Use standard library `contextvars` | Ensure out-of-the-box thread-safety and compatibility with async/await frameworks (e.g. asyncio, Starlette, FastAPI) without manual lock handling. |
| **D-P2-2** | Monotonic Clock (`time.monotonic`) | Guard against system clock jumps/skew during runs, guaranteeing accurate elapsed millisecond measurements. |
| **D-P2-3** | Stack-based push/pop on context update | As new child spans open, re-bind `_current_parent` to point to the active child. On finalization, pop back to the parent trace. |

## 3. Implementation
- **`TraceEvent` class:** Represents a node in the execution tree. Contains attributes for tracking names, monotonic timestamps, children lists, nesting depth, and parent back-references. It exposes a `.duration` property.
- **Context Variables:**
  - `_active_trace`: Stores the overall root `TraceEvent` for the agent thread/coroutine execution.
  - `_current_parent`: Tracks the deepest active node where new nested tool/llm events are attached.
- **Functions:**
  - `init_trace(name)`: Instantiates and registers the root trace event in the context.
  - `start_child(name)`: Creates a nested child trace under the current parent and pushes it as the active parent.
  - `end_child(event, status, error_payload)`: Closes timestamps, attaches status/errors, and pops context parent back to `event.parent`.
  - `finalize_trace()`: Closes the root timer, clears context variables, and returns the final tree.

## 4. Data Flow
```
[init_trace("AgentRun")] -> root (active=root, parent=root)
     │
     ├─> [start_child("tool_a")] -> child_a (parent=root, active=child_a)
     │        │
     │        └─> [end_child(child_a)] -> pops parent back to root
     │
     ├─> [start_child("tool_b")] -> child_b (parent=root, active=child_b)
     │        │
     │        └─> [end_child(child_b)] -> pops parent back to root
     │
[finalize_trace()] -> stamps end_time, clears context, returns tree
```

## 5. Safety, Isolation, & Correctness
- **Thread/Task Isolation:** `contextvars` are natively isolated per thread and async Task. An execution in thread A or request queue A cannot overwrite or access details from thread/request B.
- **Context Integrity:** Every `start_child` must pair with an `end_child` or the tracker context stack remains offset. This is handled dynamically via decorators in Phase 3.

## 6. Tests
Implemented in [`tests/test_tracker.py`](../../tests/test_tracker.py):
- Validation of trace tree hierarchy.
- Thread-level isolation verification (running concurrent traces on separate threads to ensure they do not contaminate each other).
- Error mapping inside trace tree nodes.

## 7. Files Touched
| File | Change |
|---|---|
| [`agentlatch/tracker.py`](../../agentlatch/tracker.py) | **[NEW]** Trace tracking implementation (`TraceEvent` and state modifiers). |

## 8. Acceptance Criteria
- Nested tool calls correctly display in the final trace structure with exact relative hierarchy and parent relationships.
- Consecutive calls in asynchronous event loops or concurrent threads are fully isolated from each other.
