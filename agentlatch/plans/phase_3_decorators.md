# Phase 3 — Resilience & Profiling Decorators

> Feature: **Decorators** (see [`INSTRUCTIONS.md`](../../INSTRUCTIONS.md) — Phase 3).
> Status: **✅ Done** · Depends on: **Phase 2** (timing engine) and **Phase 7** (response sampling).
> Written to document execution decorators for tools and agent loops.

---

## 1. Goal
Provide a clean public API via two core decorators:
1. `@safe_tool`: Protects individual functions (tools) from crashing the agent loop on error/timeout, returning structured JSON errors directly to the calling LLM instead.
2. `@profile_agent`: Starts/stops tracing on the root agent execution function and triggers visual flamegraph reporting.

Support both synchronous and asynchronous operations transparently.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P3-1** | Dual bare/args decorator syntax | Allow developers to write both `@safe_tool` and `@safe_tool(timeout=5.0)` naturally. |
| **D-P3-2** | Async detection via standard library | Use `inspect.iscoroutinefunction()` to dynamically wrap sync and async functions with the correct wrapper types. |
| **D-P3-3** | Thread-based timeouts for sync functions | Execute sync tool calls inside a single-worker `ThreadPoolExecutor` and fetch results with a timeout parameter. Avoid Unix-specific signals (like `signal.alarm`) to support Windows and multi-threaded runtimes. |
| **D-P3-4** | JSON-formatted error responses | Standardize tool failures into structured keys (e.g. `status`, `error_type`, `message`, `instruction`) to guide the LLM's self-correction loops. |
| **D-P3-5** | Late initialization of visuals | Lazy load `initialize_latch` (banner) and `render_flamegraph` inside the wrapper to prevent circular import issues. |

## 3. Implementation
- **`@safe_tool`:**
  - **Exception interception:** Captures python exceptions, translates them into `ErrorPayload` dictionaries, and returns them as a serialized JSON string.
  - **Timeout handling:**
    - For sync functions: runs target in `ThreadPoolExecutor.submit()` and calls `future.result(timeout=timeout)`.
    - For async functions: wraps with `asyncio.wait_for(..., timeout=timeout)`.
  - **Response compression:** Chains the returned result through `sample_response` to apply token and list item constraints.
- **`@profile_agent`:**
  - Shows startup banner on first execution if dev mode is active.
  - Registers the root trace in context.
  - Executes the wrapped agent function, and in a `finally` block, stops the trace timer and triggers `render_flamegraph()`.

## 4. Execution Flow (Sync / Async Exception Path)
```
  [ Agent Code executes @safe_tool ]
                  │
         ┌────────┴────────┐
         ▼                 ▼
   [ Normal Path ]    [ Exception / Timeout ]
         │                 │
         │           Translate exception to ErrorPayload
         │           e.g. {"status": "error", "error_type": "...", ...}
         │                 │
         ▼                 ▼
   Return raw result   Return JSON string
```

## 5. Safety, Isolation, & Correctness
- **Context Preservation:** By wrapping decorators in `functools.wraps`, all docstrings, parameter names, and function details are preserved, ensuring compatibility with LLM tool-calling extraction systems (e.g., Pydantic parsing).
- **Graceful degradation:** If `@safe_tool` is executed outside of an active `@profile_agent` context, it skips tracking events but still acts as a crash guard, trapping errors and enforcing timeouts safely.

## 6. Tests
Implemented in [`tests/test_decorators.py`](../../tests/test_decorators.py):
- Synchronous and asynchronous tool success/failure checks.
- Synchronous and asynchronous timeouts.
- Proper behavior when running nested tools inside an agent loop.
- Preservation of original function metadata via `functools.wraps`.

## 7. Files Touched
| File | Change |
|---|---|
| [`agentlatch/decorators.py`](../../agentlatch/decorators.py) | **[NEW]** Implement `@safe_tool` and `@profile_agent` wrappers. |

## 8. Acceptance Criteria
- Functions decorated with `@safe_tool` return a JSON error string on failure, and do not raise exceptions.
- Functions decorated with `@profile_agent` generate a complete execution trace graph printed to the terminal.
