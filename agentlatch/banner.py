"""Startup banner animation — Claude Code-style cosmic reveal.

Displays a large ASCII art banner with atmospheric cloud formations and
scattered stars. The art progressively "decrypts" from block/hash noise
into the final scene via a diagonal sweep, followed by a typing effect
for the welcome message.

Gracefully degrades in non-TTY environments (CI, Docker, piped output).
"""

from __future__ import annotations

import os
import random
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text

# ---------------------------------------------------------------------------
# Banner Art — Claude Code inspired cosmic atmosphere
# ---------------------------------------------------------------------------

_ART_LINES: list[str] = [
    "...............................................................",
    "",
    "     *                                       █████▓▓░          ",
    "                                 *         ███▓░     ░░        ",
    "            ░░░░░░                        ███▓░                ",
    "    ░░░   ░░░░░░░░░░                      ███▓░                ",
    "   ░░░░░░░░░░░░░░░░░░░    *                ██▓░░      ▓       ",
    "                                             ░▓▓███▓▓░        ",
    " *                                 ░░░░                       ",
    "                                 ░░░░░░░░                     ",
    "                               ░░░░░░░░░░░░░░░░               ",
    "                                                      *       ",
    "      ⚡ A G E N T L A T C H                *                 ",
    "                      *                                       ",
    "...............................................................",
]

_WELCOME_LINE = "  Terminal-native agent resilience middleware  v0.1.0"
_READY_LINE = "  Let's get started."

# ---------------------------------------------------------------------------
# Noise / Animation Config
# ---------------------------------------------------------------------------

_NOISE_CHARS: list[str] = [
    "█", "▓", "▒", "░", "#", "@", "%", "&", "╬", "╠", "╣", "╋", "┃", "┫",
]

_TOTAL_FRAMES = 30        # total decryption frames
_FRAME_DELAY = 0.016      # ~16ms per frame (~480ms total)
_TYPING_DELAY = 0.028     # per character for welcome text

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_banner_shown: bool = False

# ---------------------------------------------------------------------------
# Style Mapping
# ---------------------------------------------------------------------------


def _char_style(char: str, line: str) -> str:
    """Determine the Rich style for a resolved character."""
    if char == "⚡":
        return "bold bright_yellow"
    if char == "*":
        return "bright_yellow"
    if char == ".":
        return "dim white"
    if char in "█▓▒░":
        return "bright_cyan"
    # Letters in the AGENTLATCH branding line
    if "⚡" in line and char.isalpha():
        return "bold bright_cyan"
    return "white"


# ---------------------------------------------------------------------------
# Resolve Map — pre-compute when each character "decrypts"
# ---------------------------------------------------------------------------


def _build_resolve_map(
    lines: list[str],
    total_frames: int,
) -> list[list[int]]:
    """Assign a resolve-frame to every non-space character.

    - Dot borders (top/bottom) resolve first (frames 0–3).
    - Interior art decrypts in a diagonal sweep (top-left → bottom-right)
      with slight random jitter for an organic feel.
    """
    max_row = len(lines)
    max_col = max(len(ln) for ln in lines) if lines else 1

    rmap: list[list[int]] = []
    for row, line in enumerate(lines):
        row_map: list[int] = []
        for col, ch in enumerate(line):
            if ch == " ":
                # Spaces are always transparent — resolve instantly.
                row_map.append(-1)
            elif ch == ".":
                # Dot borders flicker in very early.
                row_map.append(random.randint(0, 3))
            else:
                # Diagonal sweep: mix of row + col progress.
                row_pct = row / max(max_row - 1, 1)
                col_pct = col / max(max_col - 1, 1)
                progress = row_pct * 0.55 + col_pct * 0.45
                base = int(progress * (total_frames - 6)) + 3
                jitter = random.randint(-2, 2)
                frame = max(2, min(total_frames - 1, base + jitter))
                row_map.append(frame)
        rmap.append(row_map)
    return rmap


# ---------------------------------------------------------------------------
# Frame Renderer
# ---------------------------------------------------------------------------


def _render_frame(
    frame: int,
    lines: list[str],
    resolve_map: list[list[int]],
) -> Text:
    """Produce one animation frame as a Rich Text object."""
    output = Text()

    for row, line in enumerate(lines):
        for col, ch in enumerate(line):
            if ch == " " or ch == "":
                output.append(" ")
                continue

            resolve_at = resolve_map[row][col]

            if frame >= resolve_at:
                # Resolved — show real character with final color.
                style = _char_style(ch, line)
                output.append(ch, style=style)
            else:
                # Still encrypted — show cycling noise.
                noise = random.choice(_NOISE_CHARS)
                output.append(noise, style="dim bright_white")

        output.append("\n")

    return output


# ---------------------------------------------------------------------------
# Typing Effect
# ---------------------------------------------------------------------------


def _type_text(console: Console, text: str, style: str, delay: float) -> None:
    """Print text character-by-character with a typing effect."""
    for ch in text:
        console.print(ch, end="", style=style, highlight=False)
        time.sleep(delay)
    console.print()  # newline


# ---------------------------------------------------------------------------
# Interactive Animation
# ---------------------------------------------------------------------------


def _play_animation(console: Console) -> None:
    """Run the full cosmic reveal sequence."""
    resolve_map = _build_resolve_map(_ART_LINES, _TOTAL_FRAMES)

    # Phase 1: Decryption sweep of the ASCII art.
    with Live(
        _render_frame(0, _ART_LINES, resolve_map),
        console=console,
        refresh_per_second=62,
        transient=True,  # we'll print the final art ourselves
    ) as live:
        for frame in range(_TOTAL_FRAMES):
            live.update(_render_frame(frame, _ART_LINES, resolve_map))
            time.sleep(_FRAME_DELAY)

    # Print the final, fully-resolved art (stays on screen).
    final_art = _render_frame(_TOTAL_FRAMES, _ART_LINES, resolve_map)
    console.print(final_art, end="")

    # Phase 2: Welcome text types in.
    console.print()
    _type_text(console, _WELCOME_LINE, "bright_white", _TYPING_DELAY)
    time.sleep(0.15)
    _type_text(console, _READY_LINE, "dim bright_green", _TYPING_DELAY * 0.8)
    console.print()


# ---------------------------------------------------------------------------
# Non-TTY Fallback
# ---------------------------------------------------------------------------


def _print_fallback(console: Console) -> None:
    """Clean single-line output for CI / Docker / piped environments."""
    console.print("[AgentLatch] ⚡ AGENTLATCH v0.1.0")
    console.print("  Terminal-native agent resilience middleware")
    console.print()


# ---------------------------------------------------------------------------
# Environment Detection
# ---------------------------------------------------------------------------


def _is_interactive() -> bool:
    """Return True only if stdout is a real interactive terminal."""
    if not sys.stdout.isatty():
        return False
    if os.environ.get("CI", "").lower() in ("true", "1"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_latch(console: Console | None = None) -> None:
    """Display the AgentLatch startup banner (once per process).

    - **Interactive terminal**: plays the full cosmic decryption animation
      followed by a typing effect for the welcome message.
    - **Non-TTY / CI / Docker**: prints a single clean line.

    Calling this more than once per process is a no-op.
    """
    global _banner_shown
    if _banner_shown:
        return
    _banner_shown = True

    con = console or Console()

    if _is_interactive():
        _play_animation(con)
    else:
        _print_fallback(con)


def reset_banner() -> None:
    """Reset the banner flag.  Useful for tests."""
    global _banner_shown
    _banner_shown = False
