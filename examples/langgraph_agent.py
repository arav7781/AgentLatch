"""LangGraph-style agent example — demonstrates AgentLatch with a state-machine mock.

Run:
    python examples/langgraph_agent.py

Simulates a LangGraph-like state machine where each node is a function
decorated with @safe_tool, and the overall graph execution is traced
by @profile_agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentlatch import profile_agent, safe_tool


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Minimal state object passed between graph nodes."""

    messages: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    error_count: int = 0


# ---------------------------------------------------------------------------
# Graph Nodes (tools)
# ---------------------------------------------------------------------------


@safe_tool
def retrieve_documents(state: AgentState) -> AgentState:
    """Node 1: Retrieve relevant documents."""
    time.sleep(0.25)  # simulate vector DB lookup
    state.data["documents"] = [
        "AgentLatch uses contextvars for tracing.",
        "Rich library renders terminal flamegraphs.",
    ]
    state.messages.append("[retrieve] Found 2 documents.")
    return state


@safe_tool
def analyze_with_llm(state: AgentState) -> AgentState:
    """Node 2: LLM analyzes the retrieved documents."""
    time.sleep(0.4)  # simulate LLM inference
    docs = state.data.get("documents", [])
    state.data["analysis"] = f"Analyzed {len(docs)} documents successfully."
    state.messages.append("[analyze] LLM analysis complete.")
    return state


@safe_tool
def generate_response(state: AgentState) -> AgentState:
    """Node 3: Generate the final response."""
    time.sleep(0.15)  # simulate response generation
    analysis = state.data.get("analysis", "No analysis available.")
    state.data["response"] = f"Final answer based on: {analysis}"
    state.messages.append("[generate] Response generated.")
    return state


# ---------------------------------------------------------------------------
# Graph Execution
# ---------------------------------------------------------------------------


@profile_agent(name="LangGraphAgent")
def run_graph() -> AgentState:
    """Execute the state machine graph."""
    state = AgentState()

    # Simulate LLM routing decision.
    time.sleep(0.1)

    # Execute nodes in sequence (like a LangGraph linear chain).
    state = retrieve_documents(state)
    time.sleep(0.05)  # router / LLM reasoning between nodes

    state = analyze_with_llm(state)
    time.sleep(0.05)

    state = generate_response(state)

    print("\n[Graph] Execution trace:")
    for msg in state.messages:
        print(f"  → {msg}")
    print(f"\n[Graph] Final: {state.data['response']}\n")

    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_graph()
