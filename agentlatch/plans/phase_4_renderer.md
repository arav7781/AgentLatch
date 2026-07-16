# Phase 4 — Terminal Flamegraph Renderer

> Feature: **Terminal Visualization** (see [`INSTRUCTIONS.md`](../../INSTRUCTIONS.md) — Phase 4).
> Status: **✅ Done** · Depends on: **Phase 2** (timing engine).
> Written to document the CLI flamegraph drawing mechanism.

---

## 1. Goal
Provide immediate visibility into execution timelines directly in the developer's terminal. Generate an interactive-looking color-coded ASCII flamegraph and breakdown summary table of tools called vs. LLM reasoning duration, without external API keys or remote logging platforms.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P4-1** | Color palette for events | - Bright Blue: LLM reasoning (gaps between tool starts/ends)<br>- Bright Green: Success<br>- Bright Red: Error<br>- Bright Yellow: Timeout |
| **D-P4-2** | Proportionate bar widths | Ensure segments represent relative execution times correctly (e.g. `(duration / total_duration) * total_columns`). |
| **D-P4-3** | Detailed breakdown table | Display a table mapping tool names, exact timings, outcome status, and full exception details. |

## 3. Implementation
- **Visual Palette (`_COLORS`):** Associates statuses and sections to console styles.
- **Bar Chars (`_BAR_CHARS`):** Defines ASCII blocks representing execution states (e.g. solid block `█` for tools, shaded `░` for LLM gap blocks).
- **Helper functions:**
  - `_format_duration(seconds)`: Converts seconds to human-friendly strings (`µs`, `ms`, or `s`).
  - `_build_timeline_bar(trace)`: Iterates through sorted children, calculating gaps before, between, and after tool runs to map LLM thinking time and tool timelines.
  - `_build_summary_table(trace)`: Generates a table showing tool call statistics.
  - `render_flamegraph(trace)`: Entrypoint method putting headers, graphs, tables, and legends together.

## 4. Architecture
```
   [ TraceEvent Tree ]
            │
            ├─> [ _build_timeline_bar() ] ───> Rich Text Timeline (Color Bars)
            ├─> [ _build_summary_table() ] ──> Rich Table (Status/Error/Durations)
            │
            ▼
   [ Printed to Terminal via rich.Console ]
```

## 5. Safety, Isolation, & Correctness
- Visual computations handle edge cases gracefully (e.g. zero-duration traces, traces with no children, and huge execution gaps).
- Output is rendered using a standard `rich.console.Console` wrapper, allowing customization or output capture if testing.

## 6. Tests
Implemented in [`tests/test_renderer.py`](../../tests/test_renderer.py):
- Test rendering logic for error-free traces.
- Test rendering with tool timeouts.
- Test rendering of exception message mapping.
- Verify fallback behavior when rendering empty traces.

## 7. Files Touched
| File | Change |
|---|---|
| [`agentlatch/renderer.py`](../../agentlatch/renderer.py) | **[NEW]** Terminal renderer logic implementation. |

## 8. Acceptance Criteria
- Flamegraph displays correct proportions for LLM vs. tool execution times.
- Legend displays matching colors and symbols for all execution states.
