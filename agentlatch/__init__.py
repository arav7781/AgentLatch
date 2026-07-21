"""AgentLatch — Terminal-native agent resilience middleware.

Two decorators are all you need::

    from agentlatch import profile_agent, safe_tool

    @safe_tool
    def query_db(sql: str) -> str:
        ...

    @profile_agent
    def run_agent():
        result = query_db("SELECT ...")
        ...

For memory-aware workflows::

    from agentlatch import context_aware, intent, safe_tool, profile_agent

    @intent("database_query")
    @context_aware(delta=True)
    @safe_tool
    def query_db(sql: str) -> str:
        ...

    @profile_agent(name="MyAgent")
    def run_agent():
        result = query_db("SELECT ...")
        ...
"""

from agentlatch.config import is_dev_mode, set_dev_mode
from agentlatch.decorators import context_aware, intent, profile_agent, safe_tool
from agentlatch.memory import (
    MemoryBackend,
    SQLiteBackend,
    get_memory,
    init_memory,
)
from agentlatch.renderer import render_flamegraph
from agentlatch.sampler import sample_response
from agentlatch.tracker import TraceEvent, get_trace

__version__ = "0.2.1"
__all__ = [
    # Decorators
    "profile_agent",
    "safe_tool",
    "context_aware",
    "intent",
    # Memory
    "MemoryBackend",
    "SQLiteBackend",
    "init_memory",
    "get_memory",
    # Rendering & Sampling
    "render_flamegraph",
    "sample_response",
    # Config
    "is_dev_mode",
    "set_dev_mode",
    # Tracing
    "TraceEvent",
    "get_trace",
    # Meta
    "__version__",
]
