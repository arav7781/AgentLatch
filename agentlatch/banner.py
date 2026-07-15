"""Startup banner animation — Claude Code-style "decryption" reveal.

Displays a progressive character-filling effect where block/hash noise
resolves column-by-column into the actual banner text over ~300ms.

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
# Configuration
# ---------------------------------------------------------------------------

_BANNER_TEXT = "⚡ AGENTLATCH v0.1.0"
_SUBTITLE = "Terminal-native agent resilience middleware"

_NOISE_CHARS = ["█", "▓", "▒", "░", "#", "@", "%", "&", "╬", "╠", "╣"]

_TOTAL_FRAMES = 18
_FRAME_DELAY = 0.018  # ~18ms per frame → ~324ms total
_STAGGER = 1.5  # frames of stagger between adjacent columns resolving

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_banner_shown: bool = False


# ---------------------------------------------------------------------------
# Core Animation
# ---------------------------------------------------------------------------


def _is_interactive() -> bool:
    """Return True only if stdout is a real interactive terminal."""
    if not sys.stdout.isatty():
        return False
    # Respect common CI environment signals.
    if os.environ.get("CI", "").lower() in ("true", "1"):
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    return True


def _render_frame(frame: int, target: str) -> Text:
    """Produce a single animation frame.

    Characters resolve left-to-right: each column *c* resolves at frame
    ``int(c * _STAGGER)``.  Before that frame, a random noise character
    is shown; after, the real character appears.
    """
    line = Text()

    for col, char in enumerate(target):
        resolve_at = int(col * _STAGGER)

        if frame >= resolve_at:
            # Resolved — show the real character.
            line.append(char, style="bold bright_cyan")
        else:
            # Still scrambled — show random noise.
            noise = random.choice(_NOISE_CHARS)
            line.append(noise, style="dim bright_white")

    return line


def _play_animation(console: Console) -> None:
    """Run the full decryption reveal using ``rich.live.Live``."""
    with Live(
        _render_frame(0, _BANNER_TEXT),
        console=console,
        refresh_per_second=60,
        transient=True,  # replace the Live area with final output
    ) as live:
        for frame in range(_TOTAL_FRAMES):
            live.update(_render_frame(frame, _BANNER_TEXT))
            time.sleep(_FRAME_DELAY)

    # Print the final resolved banner (stays on screen after Live ends).
    final = Text()
    final.append(_BANNER_TEXT, style="bold bright_cyan")
    console.print(final)

    # Subtitle.
    console.print(f"  [dim]{_SUBTITLE}[/dim]")
    console.print()


def _print_fallback(console: Console) -> None:
    """Non-animated fallback for non-TTY environments."""
    console.print(f"[AgentLatch] {_BANNER_TEXT}")
    console.print(f"  {_SUBTITLE}")
    console.print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_latch(console: Console | None = None) -> None:
    """Display the AgentLatch startup banner (once per process).

    - **Interactive terminal**: plays the decryption animation.
    - **Non-TTY / CI / Docker**: prints a single clean line.

    Calling this more than once is a no-op.
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
