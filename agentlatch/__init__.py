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
"""

from agentlatch.decorators import profile_agent, safe_tool
from agentlatch.renderer import render_flamegraph
from agentlatch.tracker import TraceEvent, get_trace

__version__ = "0.1.0"
__all__ = [
    "profile_agent",
    "safe_tool",
    "render_flamegraph",
    "TraceEvent",
    "get_trace",
    "__version__",
]
