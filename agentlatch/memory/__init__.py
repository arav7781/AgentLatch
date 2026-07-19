"""AgentLatch Memory — pluggable memory backends for agent workflows.

Provides structured, evolving memory for multi-agent DAG pipelines.
The default backend is SQLite (zero additional dependencies).

Quick start::

    from agentlatch.memory import init_memory, SQLiteBackend

    # In-memory (ephemeral, default)
    init_memory()

    # Persistent (file-based)
    init_memory(SQLiteBackend(".agentlatch.db"))

For vector backends, install extras::

    pip install "agentlatch[vector]"   # PostgreSQL + pgvector
    pip install "agentlatch[qdrant]"   # Qdrant
    pip install "agentlatch[graph]"    # Neo4j
"""

from agentlatch.memory.backend import MemoryBackend
from agentlatch.memory.context import (
    get_agent_id,
    get_intent,
    get_memory,
    get_node_context,
    get_session_id,
    init_memory,
    reset_memory_context,
    set_agent_id,
    set_intent,
    set_node_context,
)
from agentlatch.memory.sqlite_backend import SQLiteBackend

__all__ = [
    "MemoryBackend",
    "SQLiteBackend",
    "init_memory",
    "get_memory",
    "get_intent",
    "set_intent",
    "get_node_context",
    "set_node_context",
    "get_agent_id",
    "set_agent_id",
    "get_session_id",
    "reset_memory_context",
]
