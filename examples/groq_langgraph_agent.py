"""Real LangGraph + ChatGroq (Groq LLM) Agent Example — Cyclic Feedback Loop with Memory.

Run:
    export GROQ_API_KEY="your-groq-api-key"
    python examples/groq_langgraph_agent.py

Features:
1. **Cyclic DAG Loop**: Coder Agent -> Critic Reviewer -> Conditional Feedback Router -> Loop back to Coder Agent.
2. **Delta Memory Tracking**: Each iteration of the code revision loop stores incremental delta diffs (`@context_aware(delta=True)`).
3. **Live ChatGroq LLM**: Runs real ChatGroq (`llama-3.3-70b-versatile`) reasoning loops across all nodes.
4. **Resilient Tool Protocol**: Uses `@safe_tool` and `@intent` tagging for context-aware memory propagation.
5. **Traced Profiling**: Full CLI flamegraph visualization via `@profile_agent`.
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
            self.conditional_edges: dict[str, Any] = {}

        def add_node(self, name: str, action: Any) -> None:
            self.nodes[name] = action

        def add_edge(self, start_key: str, end_key: str) -> None:
            self.edges.append((start_key, end_key))

        def add_conditional_edges(
            self, source: str, router_fn: Any, path_map: dict[str, str]
        ) -> None:
            self.conditional_edges[source] = (router_fn, path_map)

        def compile(self) -> CompiledGraph:
            return CompiledGraph(self.nodes, self.edges, self.conditional_edges)

    class CompiledGraph:
        def __init__(
            self,
            nodes: dict[str, Any],
            edges: list[tuple[str, str]],
            conditional_edges: dict[str, Any],
        ) -> None:
            self.nodes = nodes
            self.edges = edges
            self.conditional_edges = conditional_edges

        def invoke(self, input_state: dict[str, Any]) -> dict[str, Any]:
            state = dict(input_state)
            current = "researcher"
            step_count = 0
            max_steps = 15

            while current != END and current in self.nodes and step_count < max_steps:
                step_count += 1
                set_node_context(current)
                node_fn = self.nodes[current]
                update = node_fn(state)
                if isinstance(update, dict):
                    state.update(update)

                # Check conditional edge first
                if current in self.conditional_edges:
                    router_fn, path_map = self.conditional_edges[current]
                    route_key = router_fn(state)
                    current = path_map.get(route_key, END)
                    continue

                # Standard edge lookup
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
# 1. Cyclic State Definition
# ---------------------------------------------------------------------------


class GroqCyclicState(TypedDict):
    """Shared state for the ChatGroq LangGraph cyclic pipeline."""

    user_query: str
    research_notes: str
    code_solution: str
    critic_feedback: str
    quality_score: float
    revision_count: int
    max_revisions: int
    final_response: str


# ---------------------------------------------------------------------------
# 2. Resilient & Memory-Aware Tools
# ---------------------------------------------------------------------------


@intent("doc_search")
@context_aware
@safe_tool(timeout=5.0)
def search_vector_db(topic: str) -> str:
    """Search internal knowledge base for architectural specifications."""
    time.sleep(0.15)  # Simulate DB latency
    return json.dumps(
        {
            "topic": topic,
            "results": [
                "AgentLatch uses contextvars for trace and memory context propagation.",
                "Cyclic DAG loops track incremental delta diffs with @context_aware(delta=True).",
                "ChatGroq provides ultra-fast LLM inference for agentic loops.",
            ],
        }
    )


@intent("code_validation")
@context_aware(delta=True)
@safe_tool(timeout=5.0)
def validate_code_execution(code: str, iteration: int) -> str:
    """Validate Python code solution and check execution metrics."""
    time.sleep(0.18)  # Simulate test suite runner
    passed = iteration > 1 or "type_hints" in code.lower()

    return json.dumps(
        {
            "iteration": iteration,
            "tests_passed": passed,
            "issues": []
            if passed
            else ["Missing explicit type annotations and docstring documentation."],
            "execution_time_ms": 12,
        }
    )


# ---------------------------------------------------------------------------
# 3. LangGraph Nodes with ChatGroq LLM Agents
# ---------------------------------------------------------------------------


def researcher_agent_node(state: GroqCyclicState) -> dict[str, Any]:
    """Node 1: Researcher Agent (ChatGroq)."""
    set_agent_id("researcher_agent")
    query = state["user_query"]

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
        print(
            "  💡 [Researcher Node] (Simulated LLM response - set GROQ_API_KEY for live ChatGroq)"
        )
        notes = f"Research summary for '{query}': AgentLatch + ChatGroq cyclic feedback pipeline."

    print(f"  [Researcher] Notes: {notes[:80]}...")
    return {"research_notes": notes}


def coder_agent_node(state: GroqCyclicState) -> dict[str, Any]:
    """Node 2: Coder Agent (ChatGroq) — Refines code across loop iterations."""
    set_agent_id("coder_agent")

    revision = state.get("revision_count", 0) + 1
    feedback = state.get("critic_feedback", "")

    memory = get_memory()
    past_research = memory.query(intent="doc_search", limit=5) if memory else []
    print(
        f"\n  💻 [Coder Node - Iteration #{revision}] Found {len(past_research)} research memories."
    )

    if feedback:
        print(f"  📋 [Coder Node] Addressing Critic Feedback: {feedback[:80]}...")

    if llm:
        print("  🤖 [Coder Node] Calling ChatGroq LLM...")
        prompt = (
            f"Write a Python production function for: '{state['user_query']}'.\n"
            f"Research Notes:\n{state['research_notes']}\n"
        )
        if feedback:
            prompt += f"\nPrevious Critic Feedback to fix:\n{feedback}\nAdd type_hints and clean docstring."

        messages = [
            SystemMessage(
                content="You are a senior Python engineer. Write clean, robust code."
            ),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        solution = str(response.content)
    else:
        print(f"  💡 [Coder Node] (Simulated LLM response - Revision #{revision})")
        if revision == 1:
            solution = "def calculate_totals(items):\n    return sum(items)"
        else:
            solution = (
                "def calculate_totals(items: list[float]) -> float:\n"
                '    """Calculate total sum with type_hints."""\n'
                "    return float(sum(items))"
            )

    # Validate execution with delta memory snapshot
    val_result = validate_code_execution(solution, revision)
    val_data = (
        json.loads(val_result)
        if isinstance(val_result, str) and val_result.startswith("{")
        else {}
    )

    print(
        f"  [Coder] Generated solution (Iteration #{revision}). Passed: {val_data.get('tests_passed')}"
    )
    return {
        "code_solution": solution,
        "revision_count": revision,
    }


def critic_reviewer_node(state: GroqCyclicState) -> dict[str, Any]:
    """Node 3: Critic Reviewer Agent (ChatGroq) — Evaluates code quality."""
    set_agent_id("critic_agent")
    solution = state.get("code_solution", "")
    revision = state.get("revision_count", 1)

    print(f"  🧐 [Critic Node] Evaluating solution quality (Iteration #{revision})...")

    if llm:
        print("  🤖 [Critic Node] Calling ChatGroq LLM for Code Review...")
        messages = [
            SystemMessage(
                content="You are a strict code auditor. Evaluate the Python solution. "
                "If type hints or docstrings are missing, request revision. "
                "Return JSON with keys: 'approved': bool, 'score': float (0.0-1.0), 'feedback': str."
            ),
            HumanMessage(content=f"Code Solution to review:\n{solution}"),
        ]
        response = llm.invoke(messages)
        content = str(response.content)

        # Parse JSON output from LLM
        try:
            # Clean markdown codeblocks if present
            clean = content.replace("```json", "").replace("```", "").strip()
            eval_data = json.loads(clean)
            approved = eval_data.get("approved", False)
            score = float(eval_data.get("score", 0.7))
            feedback = eval_data.get("feedback", "Needs improvements.")
        except Exception:
            # Fallback heuristic if LLM outputs freeform text
            approved = "type_hints" in solution.lower() or revision >= 2
            score = 0.95 if approved else 0.65
            feedback = (
                "Approved" if approved else "Add type_hints to function signature."
            )
    else:
        print(f"  💡 [Critic Node] (Simulated Evaluation - Iteration #{revision})")
        approved = "type_hints" in solution.lower() or revision >= 2
        score = 0.95 if approved else 0.65
        feedback = (
            "Approved: Code includes type_hints and docstrings."
            if approved
            else "Revision Required: Add explicit type_hints and docstring."
        )

    print(f"  [Critic] Evaluation Result: Approved={approved}, Score={score:.2f}")

    return {
        "quality_score": score,
        "critic_feedback": feedback if not approved else "",
    }


def synthesizer_agent_node(state: GroqCyclicState) -> dict[str, Any]:
    """Node 4: Synthesizer Node — Summarizes full iterative workflow."""
    set_agent_id("synthesizer_agent")

    memory = get_memory()
    total_memories = len(memory.query(limit=50)) if memory else 0
    delta_snapshots = memory.query(intent="code_validation", limit=10) if memory else []

    if llm:
        print(
            "  🤖 [Synthesizer Node] Calling ChatGroq LLM for Final Executive Briefing..."
        )
        messages = [
            SystemMessage(content="You are an executive report compiler."),
            HumanMessage(
                content=f"Summarize the final solution after {state['revision_count']} iterations:\n"
                f"Solution:\n{state['code_solution']}\n"
                f"Quality Score: {state['quality_score']}\n"
                f"DAG Memories Captured: {total_memories}"
            ),
        ]
        response = llm.invoke(messages)
        final_answer = str(response.content)
    else:
        print("  💡 [Synthesizer Node] (Simulated LLM response)")
        final_answer = (
            f"Final Briefing: Code successfully refined after {state['revision_count']} iterations.\n"
            f"Quality Score: {state['quality_score']:.2f}\n"
            f"Total Memory Snapshots: {total_memories} (Delta snapshots: {len(delta_snapshots)})\n"
            f"Final Solution:\n{state['code_solution']}"
        )

    return {"final_response": final_answer}


# ---------------------------------------------------------------------------
# 4. Conditional Router for Cyclic Loop
# ---------------------------------------------------------------------------


def loop_router(state: GroqCyclicState) -> str:
    """Dynamic router: decides whether to loop back to Coder or proceed to Synthesizer."""
    score = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    max_revs = state.get("max_revisions", 2)

    if score < 0.85 and revisions < max_revs:
        print(
            f"  🔄 [Loop Router] Score {score:.2f} < 0.85 (Revision #{revisions}/{max_revs}) -> LOOP BACK TO CODER NODE"
        )
        return "needs_revision"

    print(
        f"  ✅ [Loop Router] Score {score:.2f} >= 0.85 (or max revisions reached) -> PROCEED TO SYNTHESIZER NODE"
    )
    return "approved"


# ---------------------------------------------------------------------------
# 5. Build Cyclic LangGraph Pipeline
# ---------------------------------------------------------------------------


def build_cyclic_langgraph_pipeline() -> Any:
    """Build and compile the cyclic LangGraph feedback workflow."""
    workflow = StateGraph(GroqCyclicState)

    workflow.add_node("researcher", researcher_agent_node)
    workflow.add_node("coder", coder_agent_node)
    workflow.add_node("critic", critic_reviewer_node)
    workflow.add_node("synthesizer", synthesizer_agent_node)

    # Entry edge
    workflow.add_edge(START, "researcher")
    workflow.add_edge("researcher", "coder")
    workflow.add_edge("coder", "critic")

    # Cyclic Conditional Edge (critic -> coder OR critic -> synthesizer)
    workflow.add_conditional_edges(
        "critic",
        loop_router,
        {
            "needs_revision": "coder",  # Loop back edge!
            "approved": "synthesizer",  # Exit edge!
        },
    )

    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# 6. Profile Traced Loop
# ---------------------------------------------------------------------------


@profile_agent(
    name="GroqCyclicLangGraphAgent",
    memory_backend=SQLiteBackend(".groq_cyclic_memory.db"),
)
def run_groq_cyclic_agent(
    query: str = "Build a memory-aware agent with ChatGroq and LangGraph",
) -> GroqCyclicState:
    """Execute ChatGroq cyclic LangGraph pipeline with AgentLatch tracing."""
    llm_status = (
        "LIVE ChatGroq (model: llama-3.3-70b-versatile)"
        if HAS_GROQ
        else "Simulated LLM (set GROQ_API_KEY for live ChatGroq)"
    )
    print(f"\n⚡ Running Groq + LangGraph Cyclic Feedback Loop [{llm_status}]\n")

    pipeline = build_cyclic_langgraph_pipeline()

    initial_state: GroqCyclicState = {
        "user_query": query,
        "research_notes": "",
        "code_solution": "",
        "critic_feedback": "",
        "quality_score": 0.0,
        "revision_count": 0,
        "max_revisions": 3,
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
    run_groq_cyclic_agent()
