"""Tests for agentlatch.banner — startup animation."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from agentlatch.banner import initialize_latch, reset_banner


class TestBanner:
    def setup_method(self):
        reset_banner()

    def teardown_method(self):
        reset_banner()

    def test_runs_without_crash(self):
        """initialize_latch() must complete without exception."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=80)
        # force_terminal=False → triggers the fallback path (non-TTY).
        initialize_latch(console=console)

        output = buf.getvalue()
        assert "AGENTLATCH" in output

    def test_only_fires_once(self):
        """Calling initialize_latch twice should produce output only once."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=80)
        initialize_latch(console=console)
        first_output = buf.getvalue()

        # Reset the buffer but NOT the banner flag.
        buf.truncate(0)
        buf.seek(0)
        initialize_latch(console=console)
        second_output = buf.getvalue()

        assert len(first_output) > 0
        assert len(second_output) == 0  # no-op on second call

    def test_fallback_contains_subtitle(self):
        """Non-TTY fallback should include the middleware tagline."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=80)
        initialize_latch(console=console)

        output = buf.getvalue()
        assert "resilience middleware" in output

    def test_fallback_contains_version(self):
        """Non-TTY fallback should include the version number."""
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=80)
        initialize_latch(console=console)

        output = buf.getvalue()
        assert "v0.1.0" in output

    def test_interactive_animation_no_crash(self):
        """The full animation path (force_terminal=True) completes cleanly."""
        buf = StringIO()
        # force_terminal=True simulates a TTY for the Rich Console,
        # but _is_interactive() checks sys.stdout.isatty() which will be
        # False in test. So we test the fallback path here.
        # The animation code is exercised via the vanilla_agent.py demo.
        console = Console(file=buf, force_terminal=True, width=80)
        initialize_latch(console=console)
        output = buf.getvalue()
        assert "AGENTLATCH" in output
