# Phase 9 — Comprehensive Testing

> Feature: **Testing Suite** (see [`INSTRUCTIONS.md`](../../INSTRUCTIONS.md) — Phase 6).
> Status: **✅ Done** · Depends on: All previous phases.
> Written to document the testing suite architecture.

---

## 1. Goal
Ensure the long-term reliability and robustness of all AgentLatch modules (tracking, decorating, sampling, middleware, rendering, and banners) under various runtime conditions. Support thread safety, async loops, error structures, and boundary checks.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P9-1** | High Coverage Target | Write tests covering all components to ensure no regressions during refactors. |
| **D-P9-2** | Parallel Execution Safety | Verify that test configurations reset global module configurations (`_dev_mode_override`, `_banner_shown`) cleanly between execution loops. |
| **D-P9-3** | Test Isolation | Avoid dependency cross-contamination by checking each system independently and isolating variables. |

## 3. Implementation
Tests are organized by module under the `tests/` directory:
- **`test_tracker.py`:** Focuses on hierarchical timing structures, monotonic duration arithmetic, parent-child transitions, and concurrency checks.
- **`test_decorators.py`:** Validates sync/async wrappers, exceptions serialization, and ThreadPool timeouts.
- **`test_sampler.py`:** Validates key checking, list slicing, and string token truncations.
- **`test_middleware.py`:** Runs integration tests using Starlette test utilities to check response headers and json bodies.
- **`test_renderer.py`:** Validates styling and panel drawings.
- **`test_banner.py`:** Verifies once-per-process banners and fallback paths.

## 4. Test Layout
```
tests/
  ├── test_tracker.py     (ContextVar thread and task isolation check)
  ├── test_decorators.py  (Sync/Async decorators, timeouts & recovery check)
  ├── test_sampler.py     (Data slicing and token limits check)
  ├── test_middleware.py  (Starlette integration, response injection check)
  ├── test_renderer.py    (Flamegraph console visualization check)
  └── test_banner.py      (Decryption banner once-per-process checks)
```

## 5. Safety, Isolation, & Correctness
- Using `pytest-asyncio` ensures async loops are clean, isolated, and properly closed.
- Test suites run standard mock classes to capture stdout parameters without flooding physical consoles.

## 6. Execution Command
To run all tests with verbose output:
```bash
pytest tests/ -v
```

## 7. Files Touched
| File | Change |
|---|---|
| [`tests/test_tracker.py`](../../tests/test_tracker.py) | **[NEW]** Tests for context tracking. |
| [`tests/test_decorators.py`](../../tests/test_decorators.py) | **[NEW]** Tests for decorators. |
| [`tests/test_sampler.py`](../../tests/test_sampler.py) | **[NEW]** Tests for row/token sampling. |
| [`tests/test_middleware.py`](../../tests/test_middleware.py) | **[NEW]** Tests for HTTP middleware. |
| [`tests/test_renderer.py`](../../tests/test_renderer.py) | **[NEW]** Tests for Rich flamegraphs. |
| [`tests/test_banner.py`](../../tests/test_banner.py) | **[NEW]** Tests for decrypting initialization banner. |

## 8. Acceptance Criteria
- All tests pass successfully.
- No memory leaks or unhandled thread failures occur during test suites execution.
