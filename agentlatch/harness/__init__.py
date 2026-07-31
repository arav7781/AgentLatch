"""AgentLatch Harness — a universal execution layer for AI agents.

Orchestration frameworks (LangGraph, CrewAI, AutoGen) decide *what* an agent
does next.  None of them decide what it is *allowed* to do, where its code
runs, or what happens when a tool returns 40 000 tokens.  The harness is that
missing layer, and it attaches to any of them in one line::

    from agentlatch.harness import Harness
    from agentlatch.harness.adapters import LangGraphAdapter
    from agentlatch.harness.sandbox import DockerSandbox

    harness = Harness(sandbox=DockerSandbox(), on_approval=cli_approval_callback)
    secured_agent = harness.wrap(LangGraphAdapter(my_graph))

Every tool call then flows through one pipeline::

    ToolCall -> PermissionGate -> [Sandbox] -> Compactor -> ToolResult

The four layers, and why each exists:

* **Adapters** — the harness knows nothing about any framework.  It knows the
  :class:`FrameworkAdapter` interface, which is what makes it portable.
* **Permissions** — three tiers.  Read-only tools run; state-changing tools ask
  a human; dangerous commands are refused outright and cannot be approved.
  Unrecognised tools default to *asking*, not running.
* **Sandbox** — LLM-authored code runs in an ephemeral, network-isolated
  container, never on the host.  With no sandbox configured, code execution is
  refused rather than silently run locally.
* **Compaction** — oversized tool output is sampled or summarized before it
  re-enters the context window, and tool schemas are disclosed progressively
  instead of all at once.

Only ``rich`` and the standard library are required.  ``docker`` is needed just
for :class:`DockerSandbox` and is imported lazily.
"""

from agentlatch.harness._types import (
    ApprovalCallback,
    Decision,
    ExecutionRequest,
    ExecutionStatus,
    Language,
    PermissionTier,
    SandboxResult,
    ToolCall,
    ToolDescriptor,
    ToolResult,
)
from agentlatch.harness.adapters import (
    CallableAdapter,
    CrewAIAdapter,
    FrameworkAdapter,
    LangGraphAdapter,
)
from agentlatch.harness.compaction import Compactor, ToolRegistry
from agentlatch.harness.core import Harness, secure
from agentlatch.harness.permissions import (
    PermissionGate,
    PermissionPolicy,
    Rule,
    cli_approval_callback,
)
from agentlatch.harness.sandbox import BaseSandbox, DockerSandbox, ThreadSandbox

__all__ = [
    # Core
    "Harness",
    "secure",
    # Adapters
    "FrameworkAdapter",
    "CallableAdapter",
    "LangGraphAdapter",
    "CrewAIAdapter",
    # Permissions
    "PermissionPolicy",
    "PermissionGate",
    "Rule",
    "cli_approval_callback",
    # Sandboxes
    "BaseSandbox",
    "ThreadSandbox",
    "DockerSandbox",
    # Context
    "Compactor",
    "ToolRegistry",
    # Types
    "ToolCall",
    "ToolResult",
    "ToolDescriptor",
    "Decision",
    "PermissionTier",
    "ExecutionRequest",
    "ExecutionStatus",
    "SandboxResult",
    "Language",
    "ApprovalCallback",
]
