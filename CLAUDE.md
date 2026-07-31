# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AgentLatch is a PyPI library (`agentlatch`, currently 0.2.2) providing terminal-native resilience and observability middleware for AI agents. `rich` is the only runtime dependency; everything else (Starlette, vector backends) is an optional extra and must be imported lazily or behind a `try/except ImportError` that tells the user which extra to install (see `agentlatch/middleware.py`).

Python 3.10+ (CI matrix: 3.10–3.13).

## Commands

```bash
uv pip install -e ".[dev,server]"   # dev setup (pytest + pytest-asyncio + starlette)

pytest -v                            # full suite (testpaths=tests, asyncio_mode=auto)
pytest tests/harness -v              # harness subsystem only
pytest tests/test_sampler.py -v      # single file
pytest tests/test_decorators.py::test_safe_tool_timeout -v   # single test

ruff check .                         # lint (CI fails on this)
ruff format --check .                # format check (CI fails on this)
ruff check --fix . && ruff format .  # apply fixes

uv build                             # build wheel + sdist
pre-commit run --all-files           # ruff + bandit + detect-secrets + hygiene hooks
```

`asyncio_mode = "auto"` means async tests need no `@pytest.mark.asyncio`.

## Architecture

Everything hangs off two independent `contextvars` layers. Nothing threads IDs through user function signatures — this is the central design constraint, and any new feature should follow it.

**Trace layer (`tracker.py`)** — `_active_trace` holds the root `TraceEvent`; `_current_parent` is the deepest open node. `start_child()` pushes, `end_child()` pops back to `event.parent`. The result is a tree, rendered by `renderer.py` and serialized by `middleware.py`.

**Memory layer (`memory/context.py`)** — mirrors the same pattern with `_active_memory`, `_active_intent`, `_active_node`, `_active_agent_id`, `_session_id`. `memory/backend.py` defines the `MemoryBackend` ABC; `memory/sqlite_backend.py` is the only implementation currently in-tree (the `vector`/`qdrant`/`graph` extras are declared in `pyproject.toml` but their backend modules do not exist yet).

### Execution flow

`@profile_agent` (in `decorators.py`) is the entry point: it fires the banner, calls `init_trace()`, calls `init_memory()` unless one is already active, runs the function, then in a `finally` calls `finalize_trace()`, renders the flamegraph, and closes any memory backend it created. It only creates memory / renders visuals when `config.is_dev_mode()` is true.

`@safe_tool` wraps individual tools. Key behaviors that tests depend on:
- Exceptions never propagate — they become a JSON **string** return value built by `_build_error_payload`, so the LLM can self-correct.
- `_sanitize_error_message` redacts file paths, credential-looking assignments, and traceback fragments, then truncates at 200 chars. Disable with `safe_mode=False`.
- Timeouts are `concurrent.futures.ThreadPoolExecutor` for sync and `asyncio.wait_for` for async. **`signal.alarm` is banned** — it is not cross-platform and does not work off the main thread.
- If no trace is active (`get_trace()` is None) the tool still runs and is still protected; only timing is skipped. Preserve this — tools must be usable outside `@profile_agent`.
- On success the result passes through `sampler.sample_response()`.

Every decorator branches on `inspect.iscoroutinefunction()` and ships a sync and an async wrapper; both must be implemented and tested for any new decorator. All support both bare (`@safe_tool`) and called (`@safe_tool(timeout=5)`) forms via the `func is not None` sentinel pattern, with `@overload` stubs for typing.

### Decorator stacking

Order matters, outermost to innermost: `@intent(...)` → `@context_aware(...)` → `@safe_tool(...)`. `@intent` sets the intent ContextVar for the call duration; `@context_aware` writes a `MemorySnapshot` after successful execution; `@safe_tool` does interception and timing.

`decorators.py` re-exports `context_aware` and `intent` from `memory/decorators.py` so all four decorators can be imported from one place. `profile_agent` imports `banner`, `config`, `memory.context`, and `renderer` *inside* the wrapper to avoid circular imports — keep those local.

### LangGraph integration (`langgraph.py`)

`wrap_langgraph(graph)` / `wrap_state_node(name, fn)` intercept each node execution to record microsecond timings, state-key deltas, and transitions. `calculate_state_execution()` aggregates a `TraceEvent` tree into `StateExecutionMetrics`; `log_state_execution()` prints it. `_inspect_errors` also flags raw `<function=...>` strings in LLM output as `LLMUnparsedToolCallError` — a hallucinated tool call that never executed.

### Harness (`agentlatch/harness/`)

A second, larger subsystem layered on the same foundations. Where the decorators make *your own* tools resilient, the harness governs tools an agent framework owns. One pipeline, regardless of source:

```
ToolCall -> PermissionGate -> [Sandbox] -> Compactor -> ToolResult
```

`Harness.wrap(adapter)` returns the target with the same calling shape (a compiled graph stays `.invoke()`-able). `harness/_types.py` is the contract every stage shares — change it and all four subsystems are affected.

- **`adapters/`** — `FrameworkAdapter` ABC plus `CallableAdapter` (generic dict of callables, also the reference implementation), `LangGraphAdapter`, `CrewAIAdapter`. Adapters **duck-type** framework internals rather than isinstance-checking, which is what survives version drift. LangGraph and CrewAI hold tools in mutable owned lists, so those two patch in place and implement `restore()`.
- **`permissions.py`** — `Rule` / `PermissionPolicy` / `PermissionGate`. Three tiers, and **most-restrictive-wins** when several rules match: a tool that is both auto-approved by glob and matches a block regex is blocked. Block patterns are written with word boundaries so "check performance" doesn't trip the `rm` rule — there are explicit false-positive tests, keep them passing.
- **`sandbox/`** — `ThreadSandbox` (timeout containment only) and `DockerSandbox` (ephemeral, `network_mode="none"`, `cap_drop=ALL`, read-only root, no host mounts, `request.env` only). `import docker` is lazy; the module imports fine without the SDK.
- **`compaction.py`** — `Compactor` (delegates to `sampler.sample_response`, does not reimplement it) and `ToolRegistry` for progressive disclosure.

Invariants specific to the harness:

- **Fail closed, twice over.** `PermissionPolicy` defaults unmatched tools to `HUMAN`, not `AUTO`; and a `HUMAN` tier with no approval callback, a callback that raises, or a callback that times out all **deny**. Never "helpfully" default any of these to allow.
- **No sandbox configured means code execution is refused**, never run on the host. An unconfigured harness must not be an escape hatch.
- **Nothing raises outward.** Blocked, denied, crashed, and timed-out all return structured payloads, same rationale as `@safe_tool`.
- **`SandboxResult.terminated`** distinguishes killed (Docker, `True`) from abandoned-still-running (thread, `False`). `leaked_work` keys off it. A Python thread cannot be killed — don't let a docstring or test imply otherwise.
- **`ToolCall.command_text()` flattens nested args to depth 5** so a dangerous string inside a kwarg or list still reaches the block rules. Bounded, not unlimited, so a cyclic argument can't stall the gate.

Verified against a live colima daemon: socket discovery (`DOCKER_HOST` → `from_env()` → colima/Docker Desktop/Rancher socket paths, each proved with `ping()`), read-only root, network isolation, host env not inherited, timeout force-kill, and no leaked containers.

### Dev-mode guard

`config.is_dev_mode()` gates all ASCII output: programmatic `set_dev_mode()` override wins, then `AGENTLATCH_ENV=production` → off, else on. `banner.py` additionally checks `sys.stdout.isatty()`, `CI=true`, and `TERM=dumb`, and uses a module-level `_banner_shown` flag so the animation fires once per process. Never emit ASCII visuals without going through these guards.

## Conventions

- Shared types live in `_types.py`, including `__version__` — bump it there **and** in `pyproject.toml` together.
- Tests that touch context must reset it: an `autouse` fixture calling `tracker.reset_context()` and `reset_memory_context()` before and after (see `tests/test_context_aware.py`).
- `tests/conftest.py` intentionally prints the banner at session start.
- `agentlatch/plans/` ships phase-by-phase design docs as package data; `INSTRUCTIONS.md` is the roadmap those derive from.
- Runnable examples in `examples/` (vanilla, LangGraph, multi-agent DAG, FastAPI, Groq); the Groq/Tavily ones need `GROQ_API_KEY` / `TAVILY_API_KEY`.
