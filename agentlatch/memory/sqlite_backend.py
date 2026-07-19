"""SQLite-backed memory storage — the zero-dependency default.

Uses Python's built-in ``sqlite3`` module so no additional packages are
needed.  Supports both file-based persistence (``SQLiteBackend("agent.db")``)
and in-memory databases (``SQLiteBackend()`` or ``SQLiteBackend(":memory:")``).

Schema:
    ``snapshots`` — one row per tool invocation memory record.
    ``learnings`` — accumulated tool failure analysis records.

JSON columns are stored as TEXT and deserialized on read.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from agentlatch._types import MemorySnapshot, ToolLearning
from agentlatch.memory.backend import MemoryBackend

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_SNAPSHOTS = """\
CREATE TABLE IF NOT EXISTS snapshots (
    id            TEXT PRIMARY KEY,
    tool_name     TEXT NOT NULL,
    intent        TEXT,
    input_summary TEXT,          -- JSON
    output_summary TEXT,         -- JSON
    timestamp     REAL NOT NULL,
    node_context  TEXT,
    status        TEXT NOT NULL DEFAULT 'success',
    delta         TEXT,          -- JSON  (NULL if full snapshot)
    agent_id      TEXT,
    session_id    TEXT
);
"""

_CREATE_LEARNINGS = """\
CREATE TABLE IF NOT EXISTS learnings (
    id               TEXT PRIMARY KEY,
    tool_name        TEXT NOT NULL,
    failure_count    INTEGER NOT NULL DEFAULT 0,
    failure_patterns TEXT,       -- JSON array
    suggested_docstring TEXT,
    suggested_params TEXT,       -- JSON
    correction_hints TEXT,       -- JSON array
    timestamp        REAL NOT NULL
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_snap_tool    ON snapshots(tool_name);",
    "CREATE INDEX IF NOT EXISTS idx_snap_intent  ON snapshots(intent);",
    "CREATE INDEX IF NOT EXISTS idx_snap_node    ON snapshots(node_context);",
    "CREATE INDEX IF NOT EXISTS idx_snap_agent   ON snapshots(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_snap_ts      ON snapshots(timestamp DESC);",
    "CREATE INDEX IF NOT EXISTS idx_learn_tool   ON learnings(tool_name);",
]


# ---------------------------------------------------------------------------
# SQLite Backend
# ---------------------------------------------------------------------------


class SQLiteBackend(MemoryBackend):
    """SQLite-based memory backend — lightweight default for AgentLatch.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
                 ``":memory:"`` for ephemeral in-process storage.
                 Use a file path (e.g. ``".agentlatch.db"``) for
                 persistence across runs.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        cur = self._conn.cursor()
        cur.execute(_CREATE_SNAPSHOTS)
        cur.execute(_CREATE_LEARNINGS)
        for idx_sql in _CREATE_INDEXES:
            cur.execute(idx_sql)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Snapshot CRUD
    # ------------------------------------------------------------------

    def store(self, snapshot: MemorySnapshot) -> str:
        snap_id = snapshot.get("id") or str(uuid.uuid4())

        self._conn.execute(
            """\
            INSERT INTO snapshots
                (id, tool_name, intent, input_summary, output_summary,
                 timestamp, node_context, status, delta, agent_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snap_id,
                snapshot.get("tool_name", ""),
                snapshot.get("intent"),
                _to_json(snapshot.get("input_summary")),
                _to_json(snapshot.get("output_summary")),
                snapshot.get("timestamp", time.time()),
                snapshot.get("node_context"),
                snapshot.get("status", "success"),
                _to_json(snapshot.get("delta")),
                snapshot.get("agent_id"),
                snapshot.get("session_id"),
            ),
        )
        self._conn.commit()
        return snap_id

    def query(
        self,
        *,
        intent: str | None = None,
        tool_name: str | None = None,
        node_context: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[MemorySnapshot]:
        clauses: list[str] = []
        params: list[Any] = []

        if intent is not None:
            clauses.append("intent = ?")
            params.append(intent)
        if tool_name is not None:
            clauses.append("tool_name = ?")
            params.append(tool_name)
        if node_context is not None:
            clauses.append("node_context = ?")
            params.append(node_context)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM snapshots{where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_snapshot(row) for row in rows]

    def get_last_snapshot(
        self,
        tool_name: str,
        intent: str | None = None,
    ) -> MemorySnapshot | None:
        if intent is not None:
            row = self._conn.execute(
                "SELECT * FROM snapshots WHERE tool_name = ? AND intent = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (tool_name, intent),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM snapshots WHERE tool_name = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (tool_name,),
            ).fetchone()

        return _row_to_snapshot(row) if row else None

    # ------------------------------------------------------------------
    # Tool Learning
    # ------------------------------------------------------------------

    def store_learning(self, tool_name: str, learning: ToolLearning) -> None:
        learn_id = learning.get("id") or str(uuid.uuid4())

        self._conn.execute(
            """\
            INSERT INTO learnings
                (id, tool_name, failure_count, failure_patterns,
                 suggested_docstring, suggested_params, correction_hints,
                 timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                learn_id,
                tool_name,
                learning.get("failure_count", 0),
                _to_json(learning.get("failure_patterns")),
                learning.get("suggested_docstring"),
                _to_json(learning.get("suggested_params")),
                _to_json(learning.get("correction_hints")),
                learning.get("timestamp", time.time()),
            ),
        )
        self._conn.commit()

    def get_learnings(self, tool_name: str) -> list[ToolLearning]:
        rows = self._conn.execute(
            "SELECT * FROM learnings WHERE tool_name = ? ORDER BY timestamp DESC",
            (tool_name,),
        ).fetchall()
        return [_row_to_learning(row) for row in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass  # Already closed — harmless.

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        snap_count = self._conn.execute(
            "SELECT COUNT(*) FROM snapshots"
        ).fetchone()[0]
        learn_count = self._conn.execute(
            "SELECT COUNT(*) FROM learnings"
        ).fetchone()[0]
        return {
            "backend": "sqlite",
            "db_path": self._db_path,
            "snapshot_count": snap_count,
            "learning_count": learn_count,
        }


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _to_json(obj: Any) -> str | None:
    """Serialize to JSON string, or ``None`` if obj is ``None``."""
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=str)


def _from_json(raw: str | None) -> Any:
    """Deserialize from JSON string, or ``None`` if raw is ``None``."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _row_to_snapshot(row: sqlite3.Row) -> MemorySnapshot:
    """Convert a database row into a ``MemorySnapshot`` dict."""
    return MemorySnapshot(
        id=row["id"],
        tool_name=row["tool_name"],
        intent=row["intent"],
        input_summary=_from_json(row["input_summary"]),
        output_summary=_from_json(row["output_summary"]),
        timestamp=row["timestamp"],
        node_context=row["node_context"],
        status=row["status"],
        delta=_from_json(row["delta"]),
        agent_id=row["agent_id"],
        session_id=row["session_id"],
    )


def _row_to_learning(row: sqlite3.Row) -> ToolLearning:
    """Convert a database row into a ``ToolLearning`` dict."""
    return ToolLearning(
        id=row["id"],
        tool_name=row["tool_name"],
        failure_count=row["failure_count"],
        failure_patterns=_from_json(row["failure_patterns"]) or [],
        suggested_docstring=row["suggested_docstring"],
        suggested_params=_from_json(row["suggested_params"]),
        correction_hints=_from_json(row["correction_hints"]) or [],
        timestamp=row["timestamp"],
    )
