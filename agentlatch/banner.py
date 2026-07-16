"""Startup banner animation — Claude Code-style cosmic reveal.

Displays a large ASCII art banner spelling AGENT / LATCH in bold block
letters, surrounded by cosmic cloud formations and scattered stars.
The entire scene progressively "decrypts" from noise into the final art
via a diagonal sweep, followed by a typing effect for the welcome message.

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
# Block Letter Definitions  (each char is exactly 6 columns wide)
# ---------------------------------------------------------------------------

_LETTER_DATA: dict[str, list[str]] = {
    "A": [" ████ ", "██  ██", "██████", "██  ██", "██  ██"],
    "G": [" █████", "██    ", "██ ███", "██  ██", " █████"],
    "E": ["██████", "██    ", "████  ", "██    ", "██████"],
    "N": ["██  ██", "███ ██", "██████", "██ ███", "██  ██"],
    "T": ["██████", "  ██  ", "  ██  ", "  ██  ", "  ██  "],
    "L": ["██    ", "██    ", "██    ", "██    ", "██████"],
    "C": [" █████", "██    ", "██    ", "██    ", " █████"],
    "H": ["██  ██", "██  ██", "██████", "██  ██", "██  ██"],
}


def _build_word_lines(word: str, indent: int = 4, gap: int = 2) -> list[str]:
    """Render a word as 5 lines of block-letter ASCII art."""
    pad = " " * indent
    sep = " " * gap
    return [pad + sep.join(_LETTER_DATA[ch][row] for ch in word) for row in range(5)]


# ---------------------------------------------------------------------------
# Build the Full Art Scene
# ---------------------------------------------------------------------------

_ATMOS_TOP: list[str] = [
    "...............................................................",
    "",
    "  *        ░░░░░░                              █████▓▓░       ",
    "         ░░░░░░░░░░              *           ███▓░   ░░       ",
    "       ░░░░░░░░░░░░░░░░                      ███▓░            ",
    "    *                        *                 ██▓░░    ▓     ",
    "                                                ░▓▓██▓▓░     ",
    "",
]

_AGENT_LINES: list[str] = _build_word_lines("AGENT")
_DIVIDER: list[str] = ["                       ⚡"]
_LATCH_LINES: list[str] = _build_word_lines("LATCH")

_ATMOS_BOTTOM: list[str] = [
    "",
    " *                       ░░░░                          *      ",
    "                       ░░░░░░░░                               ",
    "                     ░░░░░░░░░░░░░░                           ",
    "                                                *             ",
    "...............................................................",
]

_ART_LINES: list[str] = (
    _ATMOS_TOP + _AGENT_LINES + _DIVIDER + _LATCH_LINES + _ATMOS_BOTTOM
)

# Pre-compute which rows are ASCII-art text (not atmosphere).
_TEXT_ROW_START = len(_ATMOS_TOP)
_TEXT_ROW_END = _TEXT_ROW_START + len(_AGENT_LINES) + len(_DIVIDER) + len(_LATCH_LINES)
_TEXT_ROWS: set[int] = set(range(_TEXT_ROW_START, _TEXT_ROW_END))

# ---------------------------------------------------------------------------
# Welcome / Tagline
# ---------------------------------------------------------------------------

_WELCOME_LINE = "  Terminal-native agent resilience middleware  v0.1.0"
_READY_LINE = "  Let's get started."

# ---------------------------------------------------------------------------
# Noise / Animation Config
# ---------------------------------------------------------------------------

_NOISE_CHARS: list[str] = [
    "█",
    "▓",
    "▒",
    "░",
    "#",
    "@",
    "%",
    "&",
    "╬",
    "╠",
    "╣",
    "╋",
    "┃",
    "┫",
]

_TOTAL_FRAMES = 32  # total decryption frames
_FRAME_DELAY = 0.015  # ~15ms per frame → ~480ms total
_TYPING_DELAY = 0.025  # per character for welcome text

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_banner_shown: bool = False

# ---------------------------------------------------------------------------
# Style Mapping
# ---------------------------------------------------------------------------


def _char_style(char: str, row: int) -> str:
    """Determine the Rich style for a resolved character."""
    if char == "⚡":
        return "bold bright_yellow"
    if char == "*":
        return "bright_yellow"
    if char == ".":
        return "dim white"

    # ASCII-art text rows → bold block letters
    if row in _TEXT_ROWS:
        return "bold bright_cyan"

    # Atmosphere gradient (clouds with depth)
    if char == "█":
        return "bright_cyan"
    if char == "▓":
        return "cyan"
    if char == "▒":
        return "dim cyan"
    if char == "░":
        return "dim bright_white"

    return "white"


# ---------------------------------------------------------------------------
# Resolve Map — when each character "decrypts"
# ---------------------------------------------------------------------------


def _build_resolve_map(
    lines: list[str],
    total_frames: int,
) -> list[list[int]]:
    """Assign a resolve-frame to every non-space character.

    Dot borders flicker in first (frames 0–3).
    The rest decrypts in a diagonal sweep (top-left → bottom-right)
    with random jitter for an organic feel.
    """
    max_row = len(lines)
    max_col = max((len(ln) for ln in lines), default=1)

    rmap: list[list[int]] = []
    for row, line in enumerate(lines):
        row_map: list[int] = []
        for col, ch in enumerate(line):
            if ch == " ":
                row_map.append(-1)  # always transparent
            elif ch == ".":
                row_map.append(random.randint(0, 3))  # dots early
            else:
                row_pct = row / max(max_row - 1, 1)
                col_pct = col / max(max_col - 1, 1)
                progress = row_pct * 0.55 + col_pct * 0.45
                base = int(progress * (total_frames - 6)) + 3
                jitter = random.randint(-3, 3)
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
            if ch == " ":
                output.append(" ")
                continue

            resolve_at = resolve_map[row][col]

            if frame >= resolve_at:
                style = _char_style(ch, row)
                output.append(ch, style=style)
            else:
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

    # Use transient=False so the final frame remains in place, avoiding double-printing.
    with Live(
        _render_frame(0, _ART_LINES, resolve_map),
        console=console,
        refresh_per_second=62,
        transient=False,
    ) as live:
        for frame in range(_TOTAL_FRAMES + 1):
            live.update(_render_frame(frame, _ART_LINES, resolve_map))
            time.sleep(_FRAME_DELAY)


    # Phase 2: Welcome text types in.
    console.print()
    _type_text(console, _WELCOME_LINE, "bright_white", _TYPING_DELAY)
    time.sleep(0.12)
    _type_text(console, _READY_LINE, "dim bright_green", _TYPING_DELAY * 0.7)
    console.print()


# ---------------------------------------------------------------------------
# Non-TTY Fallback
# ---------------------------------------------------------------------------


def _print_fallback(console: Console) -> None:
    """Clean output for CI / Docker / piped environments."""
    # Still show the ASCII art, just without animation.
    for row, line in enumerate(_ART_LINES):
        styled = Text()
        for ch in line:
            if ch == " ":
                styled.append(" ")
            else:
                styled.append(ch, style=_char_style(ch, row))
        console.print(styled)

    console.print()
    console.print(_WELCOME_LINE, style="bright_white")
    console.print(_READY_LINE, style="dim bright_green")
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
      with block-letter ASCII art reveal and typing effect.
    - **Non-TTY / CI / Docker**: prints the static colored art without animation.

    Calling this more than once per process is a no-op.
    """
    global _banner_shown
    if _banner_shown:
        return
    _banner_shown = True

    if os.environ.get("AGENTLATCH_ENV", "").lower().strip() == "production":
        return

    con = console or Console()

    if _is_interactive():
        _play_animation(con)
    else:
        _print_fallback(con)


def reset_banner() -> None:
    """Reset the banner flag.  Useful for tests."""
    global _banner_shown
    _banner_shown = False
