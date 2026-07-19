"""Multi-agent DAG workflow example — AgentLatch memory with LangGraph-style nodes.

Run:
    python examples/memory_langgraph_agent.py

Demonstrates a **leader/sub-agent** architecture:
    * A Leader Agent orchestrates three specialist sub-agents.
    * Each sub-agent has its own tools decorated with @intent, @context_aware,
      and @safe_tool.
    * Memory flows across DAG nodes — downstream agents can query upstream
      results without re-executing.
    * Delta mode avoids redundant storage when a tool is called repeatedly.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    Leader Agent                      │
    │  Orchestrates sub-agents, queries cross-node memory  │
    └──────┬──────────────┬──────────────┬────────────────┘
           │              │              │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────────┐
    │  Researcher  │ │  Analyst   │ │  Writer       │
    │  Sub-Agent   │ │  Sub-Agent │ │  Sub-Agent    │
    │  @intent     │ │  @intent   │ │  @intent      │
    │  "research"  │ │  "analyze" │ │  "write"      │
    └─────────────┘ └────────────┘ └───────────────┘
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Bootstrap local package path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentlatch import (
    SQLiteBackend,
    context_aware,
    init_memory,
    intent,
    profile_agent,
    safe_tool,
)
from agentlatch.memory.context import (
    get_memory,
    set_agent_id,
    set_node_context,
)


# ---------------------------------------------------------------------------
# Shared State (LangGraph-style)
# ---------------------------------------------------------------------------


@dataclass
class WorkflowState:
    """State passed between DAG nodes."""

    messages: list[str] = field(default_factory=list)
    research_data: dict = field(default_factory=dict)
    analysis: str = ""
    final_report: str = ""
    error_count: int = 0


# ---------------------------------------------------------------------------
# Sub-Agent 1: Researcher
# ---------------------------------------------------------------------------


@intent("research")
@context_aware
@safe_tool
def search_documents(query: str) -> str:
    """Researcher sub-agent: search a document corpus."""
    time.sleep(0.2)  # Simulate vector DB lookup.
    return (
        '{"results": ['
        '{"doc": "AgentLatch uses contextvars for tracing", "score": 0.95},'
        '{"doc": "Memory snapshots enable cross-node data sharing", "score": 0.88},'
        '{"doc": "Delta updates reduce token consumption by 60%", "score": 0.82}'
        "]}"
    )


@intent("research")
@context_aware
@safe_tool
def fetch_external_data(source: str) -> str:
    """Researcher sub-agent: fetch data from external API."""
    time.sleep(0.15)  # Simulate API call.
    return (
        '{"source": "' + source + '", '
        '"data": {"users": 1500, "active_today": 342, "growth": "12%"}}'
    )


# ---------------------------------------------------------------------------
# Sub-Agent 2: Analyst
# ---------------------------------------------------------------------------


@intent("analyze")
@context_aware(delta=True)
@safe_tool
def analyze_data(data_summary: str) -> str:
    """Analyst sub-agent: analyze collected research data."""
    time.sleep(0.3)  # Simulate LLM analysis.
    return (
        '{"insights": ['
        '"User growth is 12% — above industry average",'
        '"Document relevance scores are high (>0.8)",'
        '"Memory system reduces context window usage significantly"'
        "], "
        '"confidence": 0.91}'
    )


@intent("analyze")
@context_aware(delta=True)
@safe_tool
def validate_findings(findings: str) -> str:
    """Analyst sub-agent: cross-validate findings."""
    time.sleep(0.1)  # Simulate validation.
    return '{"validated": true, "issues": [], "quality_score": 0.94}'


# ---------------------------------------------------------------------------
# Sub-Agent 3: Writer
# ---------------------------------------------------------------------------


@intent("write")
@context_aware(progressive=True)
@safe_tool
def generate_report(context: str) -> str:
    """Writer sub-agent: generate the final report."""
    time.sleep(0.25)  # Simulate text generation.
    return (
        "# Research Report\n\n"
        "## Key Findings\n"
        "- User growth is 12%, above industry average.\n"
        "- Document relevance scores are consistently high (>0.8).\n"
        "- The AgentLatch memory system reduces context window usage "
        "significantly through delta updates.\n\n"
        "## Recommendations\n"
        "1. Continue scaling the user base with current growth trajectory.\n"
        "2. Implement delta-aware caching for repeated tool calls.\n"
        "3. Enable cross-agent memory sharing for complex DAG workflows.\n\n"
        "## Confidence: 91%\n"
    )


# ---------------------------------------------------------------------------
# Leader Agent — Orchestrates the DAG
# ---------------------------------------------------------------------------


@profile_agent(
    name="LeaderAgent",
    memory_backend=SQLiteBackend(".agentlatch_example.db"),
)
def run_leader_agent() -> WorkflowState:
    """Leader agent: orchestrate the multi-agent workflow.

    The leader runs three phases (research → analyze → write),
    with memory flowing between nodes.
    """
    state = WorkflowState()
    memory = get_memory()

    # === Phase 1: Research ===
    print("\n🔍 Phase 1: Research")

    # Set the node context so memory snapshots are tagged.
    node_token = set_node_context("research_node")
    agent_token = set_agent_id("researcher")
    time.sleep(0.1)  # Leader reasoning.

    # Researcher sub-agent executes its tools.
    docs = search_documents("agent memory systems")
    state.messages.append(f"[Researcher] Documents: {docs[:80]}...")

    ext_data = fetch_external_data("analytics_api")
    state.messages.append(f"[Researcher] External: {ext_data[:80]}...")

    state.research_data = {"docs": docs, "external": ext_data}

    # === Phase 2: Analysis ===
    print("📊 Phase 2: Analysis")

    from agentlatch.memory.context import reset_node_context, reset_agent_id
    reset_node_context(node_token)
    reset_agent_id(agent_token)

    node_token = set_node_context("analysis_node")
    agent_token = set_agent_id("analyst")
    time.sleep(0.05)  # Leader reasoning.

    # Analyst queries upstream memory (research phase results).
    if memory:
        research_memories = memory.query(intent="research", limit=5)
        print(f"   📦 Analyst found {len(research_memories)} research memories")

    analysis = analyze_data(docs[:100])
    state.messages.append(f"[Analyst] Analysis: {analysis[:80]}...")

    validation = validate_findings(analysis[:100])
    state.messages.append(f"[Analyst] Validation: {validation[:60]}...")

    state.analysis = analysis

    # === Phase 3: Writing ===
    print("✍️  Phase 3: Writing")

    reset_node_context(node_token)
    reset_agent_id(agent_token)

    node_token = set_node_context("writing_node")
    agent_token = set_agent_id("writer")
    time.sleep(0.05)  # Leader reasoning.

    # Writer queries all upstream memory.
    if memory:
        all_memories = memory.query(limit=20)
        print(f"   📦 Writer has access to {len(all_memories)} total memories")

    report = generate_report(state.analysis[:200])
    state.final_report = report
    state.messages.append(f"[Writer] Report generated (progressive ref)")

    reset_node_context(node_token)
    reset_agent_id(agent_token)

    # === Summary ===
    print("\n" + "=" * 60)
    print("📋 Workflow Summary")
    print("=" * 60)
    for msg in state.messages:
        print(f"  → {msg}")

    if memory:
        stats = memory.stats()
        print(f"\n  💾 Memory Stats: {stats}")
        print(f"  📊 Total Snapshots: {stats.get('snapshot_count', 0)}")

    print()
    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_leader_agent()
