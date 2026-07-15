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

from agentlatch.config import is_dev_mode, set_dev_mode
from agentlatch.decorators import profile_agent, safe_tool
from agentlatch.renderer import render_flamegraph
from agentlatch.sampler import sample_response
from agentlatch.tracker import TraceEvent, get_trace

__version__ = "0.1.0"
__all__ = [
    "profile_agent",
    "safe_tool",
    "render_flamegraph",
    "sample_response",
    "is_dev_mode",
    "set_dev_mode",
    "TraceEvent",
    "get_trace",
    "__version__",
]
