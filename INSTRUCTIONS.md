# AgentLatch Development Instructions

Complete development roadmap and technical specifications.

---

## Phase 1: Project Scaffolding & Core Types
- `pyproject.toml` with `rich>=13.0` as the only runtime dependency
- `_types.py` defines `EventStatus` enum (`SUCCESS`, `ERROR`, `TIMEOUT`) and `ErrorPayload` type alias
- Python 3.10+ required

## Phase 2: Context & Timing Engine (`tracker.py`)
- `TraceEvent` dataclass: `name`, `start_time`, `end_time`, `status`, `error_payload`, `children`, `depth`, `parent`
- `_active_trace` ContextVar: holds the root trace for the current execution
- `_current_parent` ContextVar: holds the current parent node for nesting
- Helper functions: `init_trace()`, `start_child()`, `end_child()`, `finalize_trace()`, `get_trace()`, `reset_context()`
- Parent/child stack managed via ContextVar push/pop pattern

## Phase 3: Decorators (`decorators.py`)

### `@safe_tool(timeout=None, on_fail="instruct_llm")`
- Supports both sync and async functions via `inspect.iscoroutinefunction()`
- On exception: captures error, returns JSON string to caller (NOT raising)
- Timeout mechanism: `concurrent.futures.ThreadPoolExecutor` for sync, `asyncio.wait_for` for async
- **No `signal.alarm`** — thread-based for cross-platform safety

### `@profile_agent(name=None)`
- Fires `initialize_latch()` banner on first invocation
- Initializes trace via `init_trace()`
- On completion: finalizes trace and renders flamegraph
- Supports sync and async

## Phase 4: Terminal Flamegraph Renderer (`renderer.py`)
- Uses `rich.console.Console`, `rich.panel.Panel`, `rich.table.Table`, `rich.text.Text`
- Color scheme: Blue (LLM reasoning), Green (tool OK), Red (tool ERROR), Yellow (tool TIMEOUT)
- Bar width proportional to duration: `width_ratio = child_duration / total_duration`
- Summary table with per-tool breakdown
- Legend row explaining colors

## Phase 4.5: Startup Banner Animation (`banner.py`)
- **Progressive Character-Filling Effect:**
  - Target text: `"⚡ AGENTLATCH v0.1.0"`
  - Character pool: `['█', '▓', '▒', '░', '#', '@', '%', '&']`
  - ~18 frames over ~324ms (`time.sleep(0.018)` per frame)
  - Columns resolve left-to-right with 1.5-frame stagger
  - Uses `rich.live.Live` with `refresh_per_second=60` for flicker-free rendering
- **TTY Guard:** If `sys.stdout.isatty()` is False or `CI=true` / `TERM=dumb` env vars detected → fallback to single clean line: `[AgentLatch] Session initialized.`
- **Once-per-process:** Module-level `_banner_shown` flag prevents repeated animations
- **Compact layout:** Single-line banner, NOT multi-line ASCII art

## Phase 5: Package API (`__init__.py`)
```python
from agentlatch import profile_agent, safe_tool, render_flamegraph, TraceEvent, get_trace
```

## Phase 6: Testing
All tests in `tests/` — run with `pytest tests/ -v`

### `test_tracker.py` (13 tests)
- TraceEvent duration calculations
- init_trace, start_child, end_child, finalize_trace
- Nested children, error payloads
- Thread isolation (two traces in separate threads)

### `test_decorators.py` (14 tests)
- @safe_tool: sync success, exception → JSON, timeout, functools.wraps
- @safe_tool: async success, exception, timeout
- @profile_agent: trace recording, named profile, nested tools, tool failure, async

### `test_renderer.py` (5 tests)
- Normal trace rendering, error events, empty trace, timeout events, legend

### `test_banner.py` (3 tests)
- No-crash, fires-once guard, subtitle content

## Architectural Safeguards & Portability
- **Thread-Based Timeouts:** `@safe_tool` timeouts enforced via `concurrent.futures.ThreadPoolExecutor` using `future.result(timeout=X)`. Signal-based alarms are strictly banned.
- **TTY Enforcement:** Startup banner actively sniffs execution environment. If `sys.stdout.isatty()` is False, or `CI=true` is detected, all ANSI animation loops are bypassed.
- **Compact Visual Presence:** Single-line title block with high-impact decryption animation. No massive multi-line ASCII art.
