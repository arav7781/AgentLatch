# Phase 5 — Startup Banner Animation

> Feature: **Startup Banner** (see [`INSTRUCTIONS.md`](../../INSTRUCTIONS.md) — Phase 4.5).
> Status: **✅ Done** · Depends on: None.
> Written to document the interactive startup visualization.

---

## 1. Goal
Provide a premium, terminal-native interactive welcome experience. Show a "decrypting" ASCII art animation spelling `AGENT LATCH` surrounded by cosmic clouds and stars. The decryption sweeping diagonally (top-left to bottom-right) should resolve from noise characters into the final clean block text, followed by typing welcome text.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P5-1** | Once-per-process guard | Utilize a global flag `_banner_shown` to ensure the animation fires only once per execution process, avoiding visual pollution on consecutive runs. |
| **D-P5-2** | Decryption effect with noise chars | Fill the unresolved characters with noise/encryption glyphs (e.g. `▓`, `▒`, `░`, `╬`) and swap them to target chars dynamically. |
| **D-P5-3** | Non-TTY / CI Fallback | Automatically detect non-interactive environments (CI, piped outputs, dumb terminals) to skip frame-by-frame sleep delays, printing the static colored logo immediately. |
| **D-P5-4** | Use `rich.live.Live` | Render the animation using a live display container to avoid terminal screen flickering during redrawing loops. |

## 3. Implementation
- **Block Letters (`_LETTER_DATA`):** Renders word symbols dynamically in 5 lines of monospace blocks.
- **Atmosphere Scenes (`_ATMOS_TOP`, `_ATMOS_BOTTOM`):** Defines background stars, boundaries, and clouds.
- **Diagonal Sweep Map (`_build_resolve_map`):** Computes when each character should decrypt based on row/column ratios combined with a random jitter value.
- **Interactive Loops (`_play_animation`):** Updates the console screen over 32 frames with a delay of 15ms per frame.
- **Typing Effect (`_type_text`):** Emits welcome strings char-by-char.

## 4. Animation Pipeline
```
[Noise Screen] ──(Diagonal Resolve Map)──> [Sweep Phase] ──> [Final ASCII Logo]
                                                                     │
                                                                     ▼
                                                             [Typing Welcome]
```

## 5. Safety, Isolation, & Correctness
- Interactive environment detection checks standard output `sys.stdout.isatty()` along with `CI` and `TERM` environment flags to disable timing waits in logs.
- Safe console print logic handles different screen resolutions gracefully.

## 6. Tests
Implemented in [`tests/test_banner.py`](../../tests/test_banner.py):
- Verification that the banner initialization executes without raising exceptions.
- Verification that repeated triggers are correctly blocked by the process-level guard flag.
- Assertion of tagline metadata strings.

## 7. Files Touched
| File | Change |
|---|---|
| [`agentlatch/banner.py`](../../agentlatch/banner.py) | **[NEW]** Decryption algorithm and welcome screen logic. |

## 8. Acceptance Criteria
- Banner performs a smooth decryption sweep from noise to full layout on interactive terminals.
- Running inside test frameworks, CI setups, or pipeline redirections redirects directly to the fallback static output immediately.
