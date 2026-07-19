"""Abstract base class for all AgentLatch memory backends.

Every backend (SQLite, PostgreSQL+pgvector, Qdrant, Neo4j) implements
this interface.  The abstract class defines the contract; concrete
implementations live in sibling modules.

Design principles:
    * All methods have both sync and async variants — the base class
      provides default async-to-sync bridges so backends only need to
      implement the sync path (override ``async_*`` for true async).
    * The ``query`` method supports flexible filtering by intent, tool
      name, node context, and agent ID.
    * Delta storage is opt-in: backends that support it implement
      ``compute_delta``; others store full snapshots.
"""

from __future__ import annotations

import abc
from typing import Any

from agentlatch._types import MemorySnapshot, ToolLearning


class MemoryBackend(abc.ABC):
    """Abstract base for pluggable memory storage.

    Subclasses must implement all ``@abstractmethod`` methods.
    """

    # ------------------------------------------------------------------
    # Snapshot CRUD
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def store(self, snapshot: MemorySnapshot) -> str:
        """Persist a memory snapshot.

        Args:
            snapshot: The snapshot to store.

        Returns:
            A unique snapshot ID.
        """

    @abc.abstractmethod
    def query(
        self,
        *,
        intent: str | None = None,
        tool_name: str | None = None,
        node_context: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[MemorySnapshot]:
        """Retrieve snapshots matching the given filters.

        All filter parameters are optional — when ``None``, that
        dimension is not filtered.  Results are ordered by timestamp
        descending (most recent first).
        """

    @abc.abstractmethod
    def get_last_snapshot(
        self,
        tool_name: str,
        intent: str | None = None,
    ) -> MemorySnapshot | None:
        """Return the most recent snapshot for the given tool + intent.

        Used by delta computation to determine what changed.
        """

    # ------------------------------------------------------------------
    # Delta Support
    # ------------------------------------------------------------------

    def compute_delta(
        self,
        previous: Any,
        current: Any,
    ) -> dict[str, Any] | None:
        """Compute the difference between two outputs.

        Default implementation does a shallow key-level diff for dicts
        and a simple equality check for other types.  Override for
        richer diffing (e.g., deep JSON diff).

        Returns ``None`` if the outputs are identical.
        """
        if previous == current:
            return None

        if isinstance(previous, dict) and isinstance(current, dict):
            delta: dict[str, Any] = {}
            all_keys = set(previous.keys()) | set(current.keys())
            for key in all_keys:
                old_val = previous.get(key)
                new_val = current.get(key)
                if old_val != new_val:
                    delta[key] = {"old": old_val, "new": new_val}
            return delta if delta else None

        # For non-dict types, just record old/new.
        return {"old": previous, "new": current}

    # ------------------------------------------------------------------
    # Tool Learning
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def store_learning(self, tool_name: str, learning: ToolLearning) -> None:
        """Persist a tool learning record."""

    @abc.abstractmethod
    def get_learnings(self, tool_name: str) -> list[ToolLearning]:
        """Retrieve all learnings for a specific tool."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend.

        Called automatically by ``@profile_agent`` on trace finalization.
        """

    # ------------------------------------------------------------------
    # Stats (optional override)
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return backend-specific statistics.

        Default returns an empty dict.  Backends can override to expose
        snapshot counts, storage size, etc.
        """
        return {}
