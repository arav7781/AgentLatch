# Phase 1 — Project Scaffolding & Core Types

> Feature: **Project Scaffolding** (see [`INSTRUCTIONS.md`](../INSTRUCTIONS.md) — Phase 1).
> Status: **✅ Done** · Depends on: None.
> Written to document the initial baseline.

---

## 1. Goal
Establish the Python project structure, package scaffolding, and central type definitions for the AgentLatch library. Ensure zero heavy runtime dependencies while declaring necessary structures for exception payloads and status codes.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P1-1** | Minimum Python Version = `3.10` | Leverage modern typing features (e.g., `TypeAlias`, union operator `|`). |
| **D-P1-2** | Run-time Dependencies = Only `rich` | Keep the package lightweight, highly portable, and fast to load. |
| **D-P1-3** | Core Event Statuses = `SUCCESS`, `ERROR`, `TIMEOUT` | Standardize tracking status across all synchronous and asynchronous operations. |
| **D-P1-4** | Error Payload format = `dict[str, Any]` | Allow extensible, JSON-serializable keys describing tool failures for LLM feedback. |

## 3. Implementation

- **`pyproject.toml`:** Declares dependencies, build backend (`setuptools`), formatters/linters (`ruff`), and testing utilities (`pytest`, `pytest-asyncio`).
- **`agentlatch/_types.py`:** Defines the `EventStatus` enum and `ErrorPayload` TypeAlias:
  ```python
  class EventStatus(enum.Enum):
      SUCCESS = "success"
      ERROR = "error"
      TIMEOUT = "timeout"

  ErrorPayload: TypeAlias = dict[str, Any]
  ```

## 4. Architecture
```
[ pyproject.toml ] ────────> Configures Build System (setuptools) & Dev Tools (ruff/pytest)
         │
         ▼
[ agentlatch/_types.py ] ──> Shared types used by tracking, decorating, & rendering engines
```

## 5. Safety, Isolation, & Correctness
- Declaring types strictly using native standard libraries ensures no dependency bloat or binary incompatibilities.
- Using Python `enum.Enum` enforces type safety on execution state mapping.

## 6. Tests
- Covered via imports and type checks in the subsequent phases. No dedicated tests are required for basic types alone.

## 7. Files Touched
| File | Change |
|---|---|
| [`pyproject.toml`](../pyproject.toml) | **[NEW]** Setup project dependencies, meta details, dev options. |
| [`agentlatch/_types.py`](../agentlatch/_types.py) | **[NEW]** Define `EventStatus` and `ErrorPayload`. |

## 8. Acceptance Criteria
- Python package can be successfully installed in a virtual environment.
- Type checker resolves `EventStatus` and `ErrorPayload` correctly.
