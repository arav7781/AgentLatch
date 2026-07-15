"""Environment detection and runtime configuration.

Controls whether AgentLatch renders ASCII visuals (banner, flamegraph) to
the terminal.  In production / server environments, visuals are suppressed
automatically — only the structured data (headers, JSON profile) is emitted.

**Rules:**

*  ``AGENTLATCH_ENV=production``  → visuals OFF
*  ``AGENTLATCH_ENV=development`` → visuals ON  (default when unset)
*  Programmatic override via ``set_dev_mode(True/False)``
"""

from __future__ import annotations

import os

_dev_mode_override: bool | None = None


def is_dev_mode() -> bool:
    """Return ``True`` if ASCII visuals should be rendered.

    Checks (in priority order):
    1. Programmatic override via :func:`set_dev_mode`.
    2. ``AGENTLATCH_ENV`` environment variable (``production`` → False).
    3. Default: ``True`` (development mode).
    """
    if _dev_mode_override is not None:
        return _dev_mode_override

    env = os.environ.get("AGENTLATCH_ENV", "development").lower().strip()
    return env != "production"


def set_dev_mode(enabled: bool) -> None:
    """Programmatically force dev mode on or off."""
    global _dev_mode_override
    _dev_mode_override = enabled


def reset_dev_mode() -> None:
    """Clear the programmatic override.  Useful for tests."""
    global _dev_mode_override
    _dev_mode_override = None
