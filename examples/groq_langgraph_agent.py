"""Real LangGraph + ChatGroq (Groq LLM) Agent Example — AgentLatch Memory & Resilience.

Run:
    export GROQ_API_KEY="your-groq-api-key"
    python examples/groq_langgraph_agent.py

Features:
1. Uses LangChain's `ChatGroq` as the core reasoning engine.
2. Combines `@safe_tool`, `@context_aware`, and `@intent` on real tools.
3. LangGraph node loops execute real ChatGroq LLM prompts, tool invocations, and memory propagation.
4. Traced with `@profile_agent` for full CLI flamegraph visualization.
"""

from __future__ import annotations

import json
import os
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
# ChatGroq & LangGraph Imports with Graceful Fallback
# ---------------------------------------------------------------------------

GROQ_KEY = os.environ.get("GROQ_API_KEY")

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    HAS_GROQ = bool(GROQ_KEY)
except ImportError:
    HAS_GROQ = False

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore[import-not-found]

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

    START = "__start__"
    END = "__end__"

    class StateGraph:  # type: ignore[no-redef]
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
            current = "researcher"
            while current != END and current in self.nodes:
                set_node_context(current)
                node_fn = self.nodes[current]
                update = node_fn(state)
                if isinstance(update, dict):
                    state.update(update)

                next_node = END
                for src, dst in self.edges:
                    if src == current:
                        next_node = dst
                        break
                current = next_node
            return state


# Initialize ChatGroq LLM if API Key and package are available
if HAS_GROQ:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_KEY,
        temperature=0.2,
    )
else:
    llm = None


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------


class GroqAgentState(TypedDict):
    """Shared state for the ChatGroq LangGraph pipeline."""

    user_query: str
    research_notes: str
    code_solution: str
    final_response: str


# ---------------------------------------------------------------------------
# 2. Resilient & Memory-Aware Tools
# ---------------------------------------------------------------------------


@intent("doc_search")
@context_aware
@safe_tool(timeout=5.0)
def search_vector_db(topic: str) -> str:
    """Search internal database for documentation and references."""
    time.sleep(0.15)  # Simulate DB latency
    return json.dumps({
        "topic": topic,
        "results": [
            "AgentLatch uses contextvars for trace and memory context propagation.",
            "Decorators @safe_tool, @context_aware, and @intent compose together.",
            "ChatGroq provides ultra-fast LLM inference for agentic loops.",
        ],
    })


@intent("code_exec")
@context_aware(delta=True)
@safe_tool(timeout=5.0)
def execute_python_code(code: str) -> str:
    """Safely execute Python code snippet."""
    time.sleep(0.2)  # Simulate sandbox execution
    if "syntax_error" in code:
        raise SyntaxError("Invalid syntax on line 3: unexpected token")

    return json.dumps({
        "status": "success",
        "output": "Result: [100, 200, 300]\nExecution time: 0.04s",
    })


# ---------------------------------------------------------------------------
# 3. LangGraph Nodes with ChatGroq LLM Agents
# ---------------------------------------------------------------------------


def researcher_agent_node(state: GroqAgentState) -> dict[str, Any]:
    """Node 1: Researcher LLM Agent (uses ChatGroq)."""
    set_agent_id("researcher_agent")
    query = state["user_query"]

    # Call tool with memory tagging
    raw_docs = search_vector_db(query)

    if llm:
        print("  🤖 [Researcher Node] Calling ChatGroq LLM...")
        messages = [
            SystemMessage(content="You are an expert technical researcher."),
            HumanMessage(
                content=f"Summarize these research docs for the user query '{query}':\n{raw_docs}"
            ),
        ]
        response = llm.invoke(messages)
        notes = str(response.content)
    else:
        print("  💡 [Researcher Node] (Simulated LLM response - set GROQ_API_KEY for live ChatGroq)")
        notes = f"Research summary based on docs for '{query}': AgentLatch + ChatGroq integration validated."

    print(f"  [Researcher] Notes: {notes[:90]}...")
    return {"research_notes": notes}


def coder_agent_node(state: GroqAgentState) -> dict[str, Any]:
    """Node 2: Coder LLM Agent (uses ChatGroq & queries upstream memory)."""
    set_agent_id("coder_agent")

    memory = get_memory()
    # Query upstream research memory created in Node 1
    past_research = memory.query(intent="doc_search", limit=5) if memory else []
    print(f"  📦 [Coder Node] Found {len(past_research)} upstream research memories.")

    # Run code execution tool with delta memory tracking
    code_res = execute_python_code("def compute(): return [100, 200, 300]")

    if llm:
        print("  🤖 [Coder Node] Calling ChatGroq LLM...")
        messages = [
            SystemMessage(content="You are a senior Python software engineer."),
            HumanMessage(
                content=f"Based on research notes:\n{state['research_notes']}\n"
                f"And execution output:\n{code_res}\nWrite a clean Python solution."
            ),
        ]
        response = llm.invoke(messages)
        solution = str(response.content)
    else:
        print("  💡 [Coder Node] (Simulated LLM response - set GROQ_API_KEY for live ChatGroq)")
        solution = "def agentlatch_groq_demo():\n    return 'ChatGroq + LangGraph AgentLatch pipeline success'"

    print(f"  [Coder] Solution generated.")
    return {"code_solution": solution}


def synthesizer_agent_node(state: GroqAgentState) -> dict[str, Any]:
    """Node 3: Synthesizer LLM Agent (Final output)."""
    set_agent_id("synthesizer_agent")

    if llm:
        print("  🤖 [Synthesizer Node] Calling ChatGroq LLM...")
        messages = [
            SystemMessage(content="You are an executive assistant preparing a final briefing."),
            HumanMessage(
                content=f"Combine research:\n{state['research_notes']}\n"
                f"And solution:\n{state['code_solution']}\nInto a short final response."
            ),
        ]
        response = llm.invoke(messages)
        final_answer = str(response.content)
    else:
        print("  💡 [Synthesizer Node] (Simulated LLM response - set GROQ_API_KEY for live ChatGroq)")
        final_answer = (
            f"Final Answer: ChatGroq LangGraph pipeline complete!\n"
            f"Research: {state['research_notes'][:60]}...\n"
            f"Code: {state['code_solution'][:60]}..."
        )

    return {"final_response": final_answer}


# ---------------------------------------------------------------------------
# 4. Build LangGraph Pipeline
# ---------------------------------------------------------------------------


def build_groq_langgraph_pipeline() -> Any:
    """Build and compile the ChatGroq LangGraph workflow."""
    workflow = StateGraph(GroqAgentState)

    workflow.add_node("researcher", researcher_agent_node)
    workflow.add_node("coder", coder_agent_node)
    workflow.add_node("synthesizer", synthesizer_agent_node)

    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 5. Profile Traced Loop
# ---------------------------------------------------------------------------


@profile_agent(
    name="GroqLangGraphAgent",
    memory_backend=SQLiteBackend(".groq_memory.db"),
)
def run_groq_agent(query: str = "Build a memory-aware agent using ChatGroq and LangGraph") -> GroqAgentState:
    """Execute ChatGroq LangGraph pipeline with AgentLatch tracing."""
    llm_status = f"LIVE ChatGroq (model: llama-3.3-70b-versatile)" if HAS_GROQ else "Simulated LLM (set GROQ_API_KEY for live ChatGroq)"
    print(f"\n⚡ Running Groq + LangGraph Pipeline [{llm_status}]\n")

    pipeline = build_groq_langgraph_pipeline()

    initial_state: GroqAgentState = {
        "user_query": query,
        "research_notes": "",
        "code_solution": "",
        "final_response": "",
    }

    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    print("📋 Final Response:")
    print("=" * 60)
    print(f"{final_state['final_response']}\n")

    memory = get_memory()
    if memory:
        stats = memory.stats()
        print(f"💾 Memory Snapshots Recorded: {stats.get('snapshot_count', 0)}\n")

    return final_state


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_groq_agent()
