"""Shared types and enumerations for AgentLatch."""

from __future__ import annotations

import enum
from typing import Any, TypeAlias

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventStatus(enum.Enum):
    """Status of a traced execution event."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

ErrorPayload: TypeAlias = dict[str, Any]
"""Structured error information returned to the LLM instead of raising."""
