"""LangGraph integration example — demonstrates AgentLatch with LangGraph's StateGraph API.

Run:
    python examples/langgraph_agent.py

Features demonstrated:
1. Decorating LangGraph node functions with @safe_tool, @context_aware, and @intent.
2. Cross-node memory access inside graph nodes using get_memory().
3. Tracing full LangGraph StateGraph execution with @profile_agent.
4. Auto-fallback MockStateGraph if the `langgraph` package is not installed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, TypedDict

# Bootstrap local package path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentlatch import (
    SQLiteBackend,
    context_aware,
    get_memory,
    intent,
    profile_agent,
    safe_tool,
)
from agentlatch.memory.context import set_agent_id, set_node_context

# ---------------------------------------------------------------------------
# LangGraph Import with Fallback Mock
# ---------------------------------------------------------------------------

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

    # Minimal MockStateGraph for zero-dependency execution
    START = "__start__"
    END = "__end__"

    class StateGraph:  # type: ignore[no-redef]
        """Mock StateGraph mirroring the official LangGraph API."""

        def __init__(self, state_schema: type) -> None:
            self.state_schema = state_schema
            self.nodes: dict[str, Any] = {}
            self.edges: list[tuple[str, str]] = []

        def add_node(self, name: str, action: Any) -> None:
            self.nodes[name] = action

        def add_edge(self, start_key: str, end_key: str) -> None:
            self.edges.append((start_key, end_key))

        def compile(self) -> CompiledGraph:
            return CompiledGraph(self.nodes, self.edges)

    class CompiledGraph:
        def __init__(self, nodes: dict[str, Any], edges: list[tuple[str, str]]) -> None:
            self.nodes = nodes
            self.edges = edges

        def invoke(self, input_state: dict[str, Any]) -> dict[str, Any]:
            state = dict(input_state)
            current = "retrieve"
            while current != END and current in self.nodes:
                # Set node context for memory snapshot tagging
                set_node_context(current)
                node_fn = self.nodes[current]
                update = node_fn(state)
                if isinstance(update, dict):
                    state.update(update)

                # Determine next node from edges
                next_node = END
                for src, dst in self.edges:
                    if src == current:
                        next_node = dst
                        break
                current = next_node
            return state


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """LangGraph State schema passed between nodes."""

    query: str
    documents: list[str]
    analysis: str
    final_answer: str


# ---------------------------------------------------------------------------
# 2. Graph Nodes (Decorated with AgentLatch)
# ---------------------------------------------------------------------------


@intent("retrieval")
@context_aware
@safe_tool
def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Node 1: Fetch documents from vector database."""
    set_agent_id("retriever_agent")
    time.sleep(0.2)  # Simulate DB lookup latency

    docs = [
        "AgentLatch integrates seamlessly with LangGraph StateGraph.",
        "Contextvars engine propagates trace & memory across graph nodes.",
        "MemorySnapshots enable cross-node state recovery.",
    ]
    print(f"  [Node: retrieve] Found {len(docs)} documents.")
    return {"documents": docs}


@intent("analysis")
@context_aware(delta=True)
@safe_tool
def analyze_node(state: AgentState) -> dict[str, Any]:
    """Node 2: LLM analyzes retrieved documents + queries memory."""
    set_agent_id("analyst_agent")
    time.sleep(0.25)  # Simulate LLM reasoning latency

    docs = state.get("documents", [])

    # Query upstream memory for past retrieval snapshots
    memory = get_memory()
    past_retrievals = memory.query(intent="retrieval", limit=5) if memory else []

    analysis_text = (
        f"Analyzed {len(docs)} documents. "
        f"Cross-node memory hits for 'retrieval' intent: {len(past_retrievals)}."
    )
    print(f"  [Node: analyze] {analysis_text}")
    return {"analysis": analysis_text}


@intent("generation")
@context_aware(progressive=True)
@safe_tool
def generate_node(state: AgentState) -> dict[str, Any]:
    """Node 3: Formulate final response."""
    set_agent_id("writer_agent")
    time.sleep(0.15)  # Simulate response generation latency

    analysis = state.get("analysis", "No analysis")
    final_answer = (
        f"Final Answer: LangGraph + AgentLatch workflow executed successfully! "
        f"({analysis})"
    )
    print("  [Node: generate] Response generated.")
    return {"final_answer": final_answer}


# ---------------------------------------------------------------------------
# 3. Build & Compile LangGraph StateGraph
# ---------------------------------------------------------------------------


from agentlatch import (
    SQLiteBackend,
    calculate_state_execution,
    context_aware,
    get_memory,
    intent,
    log_state_execution,
    profile_agent,
    safe_tool,
    wrap_langgraph,
)
from agentlatch.memory.context import set_agent_id, set_node_context


def create_langgraph_pipeline() -> Any:
    """Construct and compile the LangGraph StateGraph wrapped with AgentLatch profiling."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("generate", generate_node)

    # Add linear edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", END)

    # Wrap StateGraph for automated high-precision state execution tracking
    return wrap_langgraph(workflow.compile())


# ---------------------------------------------------------------------------
# 4. Traced Agent Loop
# ---------------------------------------------------------------------------


@profile_agent(
    name="LangGraphAgent",
    memory_backend=SQLiteBackend(".langgraph_memory.db"),
)
def run_langgraph_agent(
    query: str = "Explain AgentLatch with LangGraph",
) -> dict[str, Any]:
    """Execute the compiled LangGraph pipeline wrapped in an AgentLatch profile."""
    mode = (
        "Official langgraph package"
        if HAS_LANGGRAPH
        else "AgentLatch LangGraph mock engine"
    )
    print(f"\n🚀 Running LangGraph Pipeline ({mode})...\n")

    pipeline = create_langgraph_pipeline()
    initial_state: AgentState = {
        "query": query,
        "documents": [],
        "analysis": "",
        "final_answer": "",
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    print(f"📋 Output: {final_state.get('final_answer')}")
    print("=" * 60)

    # Calculate and output precision state metrics
    metrics = calculate_state_execution()
    log_state_execution(metrics, print_console=True)

    memory = get_memory()
    if memory:
        stats = memory.stats()
        print(f"💾 Memory Snapshots Recorded: {stats.get('snapshot_count', 0)}\n")

    return final_state


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_langgraph_agent()
