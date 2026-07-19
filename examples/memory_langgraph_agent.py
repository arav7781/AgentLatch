"""Multi-Agent Leader/Sub-Agent DAG Example — ChatGroq LLM + AgentLatch Memory.

Run:
    export GROQ_API_KEY="your-groq-api-key"
    python examples/memory_langgraph_agent.py

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │              Leader Agent (ChatGroq)                 │
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

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Bootstrap local package path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentlatch import (
    SQLiteBackend,
    context_aware,
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
# ChatGroq Setup
# ---------------------------------------------------------------------------

GROQ_KEY = os.environ.get("GROQ_API_KEY")

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    HAS_GROQ = bool(GROQ_KEY)
except ImportError:
    HAS_GROQ = False

llm = (
    ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_KEY, temperature=0.2)
    if HAS_GROQ
    else None
)


# ---------------------------------------------------------------------------
# Shared State
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
# Sub-Agent Tools
# ---------------------------------------------------------------------------


@intent("research")
@context_aware
@safe_tool
def search_documents(query: str) -> str:
    """Researcher sub-agent tool: search document corpus."""
    time.sleep(0.18)  # Simulate DB lookup
    return json.dumps(
        {
            "results": [
                {
                    "doc": "AgentLatch uses contextvars for trace and memory propagation",
                    "score": 0.96,
                },
                {
                    "doc": "MemorySnapshots enable cross-node data sharing across sub-agents",
                    "score": 0.91,
                },
                {
                    "doc": "Delta updates eliminate redundant storage in iterative LLM runs",
                    "score": 0.85,
                },
            ]
        }
    )


@intent("research")
@context_aware
@safe_tool
def fetch_external_data(source: str) -> str:
    """Researcher sub-agent tool: fetch metric data."""
    time.sleep(0.15)  # Simulate API lookup
    return json.dumps(
        {
            "source": source,
            "metrics": {
                "total_users": 24500,
                "daily_active": 4120,
                "growth_rate_pct": 14.5,
            },
        }
    )


@intent("analyze")
@context_aware(delta=True)
@safe_tool
def analyze_data(data_summary: str) -> str:
    """Analyst sub-agent tool: process research metrics."""
    time.sleep(0.22)  # Simulate analytical tool execution
    return json.dumps(
        {
            "status": "analyzed",
            "processed_input_length": len(data_summary),
            "insights_extracted": 3,
        }
    )


@intent("write")
@context_aware(progressive=True)
@safe_tool
def generate_report(context: str) -> str:
    """Writer sub-agent tool: compile executive report."""
    time.sleep(0.25)  # Simulate formatting tool
    return json.dumps(
        {
            "report_type": "Executive Summary",
            "context_length": len(context),
            "status": "ready",
        }
    )


# ---------------------------------------------------------------------------
# Leader Agent & Sub-Agent Execution Nodes
# ---------------------------------------------------------------------------


@profile_agent(
    name="LeaderAgent",
    memory_backend=SQLiteBackend(".agentlatch_example.db"),
)
def run_leader_agent(
    user_query: str = "Analyze system performance and agent memory efficiency",
) -> WorkflowState:
    """Leader agent orchestrates Researcher, Analyst, and Writer sub-agents using ChatGroq LLM."""
    state = WorkflowState()
    memory = get_memory()

    llm_status = (
        "LIVE ChatGroq (llama-3.3-70b-versatile)"
        if HAS_GROQ
        else "Simulated LLM (set GROQ_API_KEY for live ChatGroq)"
    )
    print(f"\n⚡ Executing Multi-Agent Leader/Sub-Agent DAG [{llm_status}]\n")

    # === Phase 1: Researcher Sub-Agent ===
    print("🔍 Phase 1: Researcher Sub-Agent")
    node_token = set_node_context("research_node")
    agent_token = set_agent_id("researcher_agent")

    raw_docs = search_documents(user_query)
    ext_data = fetch_external_data("analytics_service")

    if llm:
        print("  🤖 [Researcher Sub-Agent] Calling ChatGroq LLM...")
        resp = llm.invoke(
            [
                SystemMessage(content="You are a senior technical researcher."),
                HumanMessage(
                    content=f"Synthesize research for '{user_query}':\nDocs: {raw_docs}\nMetrics: {ext_data}"
                ),
            ]
        )
        research_notes = str(resp.content)
    else:
        print("  💡 [Researcher Sub-Agent] Synthesizing research outputs...")
        research_notes = f"Research synthesis for '{user_query}': Growth 14.5%, active users 4,120. Memory propagation verified."

    state.messages.append(f"[Researcher] Notes: {research_notes[:80]}...")
    state.research_data = {"notes": research_notes, "raw": raw_docs}

    # === Phase 2: Analyst Sub-Agent ===
    print("\n📊 Phase 2: Analyst Sub-Agent")
    from agentlatch.memory.context import reset_agent_id, reset_node_context

    reset_node_context(node_token)
    reset_agent_id(agent_token)

    node_token = set_node_context("analysis_node")
    agent_token = set_agent_id("analyst_agent")

    # Query upstream research memory
    if memory:
        upstream_memories = memory.query(intent="research", limit=5)
        print(
            f"   📦 Analyst Sub-Agent queried {len(upstream_memories)} upstream research memories from ChatGroq session."
        )

    analyze_data(research_notes[:120])

    if llm:
        print("  🤖 [Analyst Sub-Agent] Calling ChatGroq LLM...")
        resp = llm.invoke(
            [
                SystemMessage(content="You are a data analyst."),
                HumanMessage(
                    content=f"Analyze these research findings and provide strategic implications:\n{research_notes}"
                ),
            ]
        )
        analysis_out = str(resp.content)
    else:
        print("  💡 [Analyst Sub-Agent] Formulating data analysis...")
        analysis_out = "Analysis: Growth rate of 14.5% is strong. AgentLatch memory reduces token consumption by 60%."

    state.analysis = analysis_out
    state.messages.append(f"[Analyst] Analysis: {analysis_out[:80]}...")

    # === Phase 3: Writer Sub-Agent ===
    print("\n✍️  Phase 3: Writer Sub-Agent")
    reset_node_context(node_token)
    reset_agent_id(agent_token)

    node_token = set_node_context("writing_node")
    agent_token = set_agent_id("writer_agent")

    if memory:
        all_memories = memory.query(limit=20)
        print(
            f"   📦 Writer Sub-Agent accessed {len(all_memories)} total memories in pipeline."
        )

    generate_report(state.analysis[:150])

    if llm:
        print("  🤖 [Writer Sub-Agent] Calling ChatGroq LLM...")
        resp = llm.invoke(
            [
                SystemMessage(content="You are a professional technical writer."),
                HumanMessage(
                    content=f"Compile a final executive summary report based on:\nAnalysis:\n{state.analysis}"
                ),
            ]
        )
        final_report = str(resp.content)
    else:
        print("  💡 [Writer Sub-Agent] Compiling final executive report...")
        final_report = (
            f"# Executive Summary Report\n\n"
            f"## Key Takeaways\n"
            f"- User Growth: 14.5%\n"
            f"- Analysis: {state.analysis[:120]}...\n\n"
            f"## Conclusion\n"
            f"Multi-Agent DAG workflow executed successfully with ChatGroq + AgentLatch memory."
        )

    state.final_report = final_report
    state.messages.append("[Writer] Report generated (progressive reference stored)")

    reset_node_context(node_token)
    reset_agent_id(agent_token)

    # === Summary ===
    print("\n" + "=" * 60)
    print("📋 Final Executive Briefing (ChatGroq):")
    print("=" * 60)
    print(f"{state.final_report}\n")

    if memory:
        stats = memory.stats()
        print(f"💾 Memory Stats: {stats}")
        print(f"📊 Total Snapshots Recorded: {stats.get('snapshot_count', 0)}\n")

    return state


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_leader_agent()
