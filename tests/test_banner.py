"""Tests for agentlatch.banner — startup animation with block-letter art."""

from __future__ import annotations

import re
from io import StringIO

from rich.console import Console

from agentlatch.banner import initialize_latch, reset_banner

# Regex to strip ANSI escape codes from Rich output.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _capture_banner() -> str:
    """Render the banner into a string buffer and return plain text."""
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=80)
    initialize_latch(console=console)
    return _strip_ansi(buf.getvalue())


class TestBanner:
    def setup_method(self):
        reset_banner()

    def teardown_method(self):
        reset_banner()

    def test_runs_without_crash(self):
        """initialize_latch() must complete without exception."""
        output = _capture_banner()
        assert len(output) > 0

    def test_only_fires_once(self):
        """Calling initialize_latch twice should produce output only once."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=80)
        initialize_latch(console=console)
        first_len = len(buf.getvalue())

        buf.truncate(0)
        buf.seek(0)
        initialize_latch(console=console)
        second_len = len(buf.getvalue())

        assert first_len > 0
        assert second_len == 0

    def test_contains_block_art(self):
        """Output should contain block-letter ASCII art (██ blocks)."""
        output = _capture_banner()
        assert "██" in output

    def test_contains_welcome_text(self):
        """Output should include the welcome tagline."""
        output = _capture_banner()
        assert "resilience middleware" in output

    def test_contains_version(self):
        """Output should include the version number."""
        from agentlatch import __version__

        output = _capture_banner()
        assert f"v{__version__}" in output

    def test_contains_lightning_bolt(self):
        """The ⚡ divider between AGENT and LATCH must appear."""
        output = _capture_banner()
        assert "⚡" in output

    def test_contains_dot_borders(self):
        """Output must include the dotted border lines."""
        output = _capture_banner()
        assert "....." in output

    def test_contains_stars(self):
        """Output must include atmospheric star characters."""
        output = _capture_banner()
        assert "*" in output

    def test_contains_ready_message(self):
        """Output must include the 'Let's get started' message."""
        output = _capture_banner()
        assert "get started" in output
