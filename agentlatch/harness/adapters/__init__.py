"""Framework adapters — how the harness attaches to an agent framework.

Importing this package never imports LangGraph or CrewAI; each adapter
resolves its framework lazily inside its methods.

    from agentlatch.harness.adapters import LangGraphAdapter

    secured = harness.wrap(LangGraphAdapter(my_graph))
"""

from agentlatch.harness.adapters.base import FrameworkAdapter
from agentlatch.harness.adapters.crewai import CrewAIAdapter
from agentlatch.harness.adapters.generic import CallableAdapter
from agentlatch.harness.adapters.langgraph import LangGraphAdapter

__all__ = [
    "FrameworkAdapter",
    "CallableAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
]
