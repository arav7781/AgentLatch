# Phase 6 — Global Configuration Management

> Feature: **Runtime Configuration** (see [`INSTRUCTIONS.md`](../../INSTRUCTIONS.md) — Architectural Safeguards & Portability).
> Status: **✅ Done** · Depends on: None.
> Written to document environment detection and console options.

---

## 1. Goal
Provide environment-level and programmatic controls to toggle developer-friendly terminal features (ASCII animations, timing graphs). Ensure that in production, headless, or server environments (where HTTP endpoints are serving JSON), stdout visuals are suppressed automatically.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P6-1** | Priority-based Environment Check | Check variables in priority order:<br>1. Programmatic override (`set_dev_mode`) / tests.<br>2. Environment variable `AGENTLATCH_ENV` (if `"production"`, silence visual rendering).<br>3. Default to True (development mode). |
| **D-P6-2** | Visual-only toggle | Suppress only terminal outputs; core middleware traces, timing logic, and sampling should remain active and fully functional. |
| **D-P6-3** | Global State isolation | Expose clean helper methods (`is_dev_mode`, `set_dev_mode`, `reset_dev_mode`) rather than direct access to config structures. |

## 3. Implementation
- **`agentlatch/config.py`:**
  - Manages global state `_dev_mode_override`.
  - Exposes `is_dev_mode()` to check `AGENTLATCH_ENV` and returns status boolean.
  - Exposes `set_dev_mode(bool)` to programmatically override configuration.
  - Exposes `reset_dev_mode()` to reset state back to checking environment.
- **Integration:** The decorators wrapper imports and queries `is_dev_mode()` before calling `initialize_latch()` or `render_flamegraph(trace)`.

## 4. Configuration Check Flow
```
        [ Check if override is set ]
               /            \
          (Yes)              (No)
          /                    \
Return override state      Read AGENTLATCH_ENV
                           (Default: "development")
                                 │
                           Return True if != "production"
```

## 5. Safety, Isolation, & Correctness
- Programmatic overrides take immediate effect on subsequent calls, even if runtime threads are active.
- Global config modifications are guarded inside a central module structure, avoiding mutable state fragmentation across modules.

## 6. Tests
Implemented and verified as part of the integration tests:
- Programmatic overrides correctly control the console render trigger.
- Verification that setting `AGENTLATCH_ENV=production` suppresses flamegraph visualization without breaking trace data mapping.

## 7. Files Touched
| File | Change |
|---|---|
| [`agentlatch/config.py`](../../agentlatch/config.py) | **[NEW]** Environment detection and programmatic configuration toggle. |

## 8. Acceptance Criteria
- Setting `AGENTLATCH_ENV=production` silences all flamegraph panels and startup animations from standard output.
- Custom execution scripts can bypass environment configs using `set_dev_mode(True/False)` at runtime.
