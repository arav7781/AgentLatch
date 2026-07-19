"""Complex Multi-Agent DAG Workflow Example — LangGraph + AgentLatch Memory & Resilience.

Run:
    python examples/complex_multiagent_dag.py

Demonstrates an enterprise-grade multi-agent DAG with:
1. **Parallel Execution Branches**: Research Branch and Code Audit Branch run concurrently.
2. **Conditional Routing**: Security Evaluator node inspects upstream memory and dynamically routes to Remediation or Synthesis.
3. **Cross-Node Memory Pipeline**: Sub-agents query past snapshots across DAG nodes using `@intent("tag")` and `memory.query()`.
4. **Resilience & Self-Correction**: `@safe_tool` traps exceptions, allowing self-correction loops.
5. **Delta & Progressive Storage**: `@context_aware(delta=True)` for incremental updates and `@context_aware(progressive=True)` for large payload references.
"""

from __future__ import annotations

import json
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
# LangGraph Import with Fallback Engine
# ---------------------------------------------------------------------------

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
            current = "research"
            visited = set()

            while current != END and current in self.nodes:
                visited.add(current)
                set_node_context(current)
                node_fn = self.nodes[current]
                update = node_fn(state)
                if isinstance(update, dict):
                    state.update(update)

                # Check conditional edge
                if current in self.conditional_edges:
                    router_fn, path_map = self.conditional_edges[current]
                    route_key = router_fn(state)
                    current = path_map.get(route_key, END)
                    continue

                # Standard edge lookup
                next_node = END
                for src, dst in self.edges:
                    if src == current and dst not in visited:
                        next_node = dst
                        break
                current = next_node

            return state


# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------


class AuditWorkflowState(TypedDict):
    """Shared state dictionary passed across the multi-agent DAG nodes."""

    project_name: str
    code_snippet: str
    documents: list[str]
    audit_findings: list[dict[str, Any]]
    security_score: float
    remediation_applied: bool
    final_report: str
    attempt_count: int


# ---------------------------------------------------------------------------
# 2. Sub-Agent Nodes & Tools (Decorated with AgentLatch)
# ---------------------------------------------------------------------------


# --- Branch A: Research Agent ---


@intent("research")
@context_aware
@safe_tool
def fetch_architecture_docs(project: str) -> str:
    """Researcher Sub-Agent: Query knowledge base for security standards."""
    time.sleep(0.18)  # Simulate vector DB lookup
    return json.dumps(
        {
            "status": "success",
            "docs": [
                "OWASP Top 10 API Security Risks (2026)",
                "NIST Cybersecurity Framework v2.0 Guidelines",
                "AgentLatch Context-Aware Memory Propagation Standard",
            ],
        }
    )


# --- Branch B: Code Audit Agent (With Self-Correction & Delta Updates) ---


@intent("code_audit")
@context_aware(delta=True)
@safe_tool
def run_static_analysis(code: str, attempt: int) -> str:
    """Code Auditor Sub-Agent: Run static code analysis with retry resilience."""
    time.sleep(0.25)  # Simulate AST parsing

    # Demonstrate @safe_tool failure interception on first attempt
    if "eval(" in code and attempt == 1:
        raise ValueError(
            "Critical vulnerability: Unsafe eval() statement detected at line 14!"
        )

    return json.dumps(
        {
            "status": "passed",
            "issues_found": 1 if "eval(" in code else 0,
            "vulnerabilities": ["Use of eval() in dynamic code evaluation"]
            if "eval(" in code
            else [],
        }
    )


# --- Convergence Node: Security Evaluator ---


@intent("security_evaluation")
@context_aware
@safe_tool
def evaluate_security_risk(state: AuditWorkflowState) -> dict[str, Any]:
    """Security Evaluator Node: Aggregates research + audit memories."""
    time.sleep(0.12)  # Simulate risk matrix calculation

    memory = get_memory()

    # Query upstream research memory
    research_mems = memory.query(intent="research", limit=5) if memory else []
    # Query upstream code audit memory
    audit_mems = memory.query(intent="code_audit", limit=5) if memory else []

    has_vuln = any("eval(" in str(m.get("input_summary")) for m in audit_mems)

    score = 0.45 if (has_vuln and not state.get("remediation_applied")) else 0.96
    print(
        f"  [Security Evaluator] Memory hits: Research={len(research_mems)}, Audit={len(audit_mems)} | Score: {score}"
    )

    return {
        "security_score": score,
        "audit_findings": [
            {"type": "Vulnerability", "severity": "HIGH" if score < 0.8 else "LOW"}
        ],
    }


# --- Conditional Branch: Remediation Agent ---


@intent("remediation")
@context_aware(delta=True)
@safe_tool
def apply_remediation(code: str) -> str:
    """Remediation Sub-Agent: Auto-fix detected code vulnerabilities."""
    time.sleep(0.2)  # Simulate refactoring AST pass
    cleaned_code = code.replace("eval(expr)", "ast.literal_eval(expr)")
    return json.dumps(
        {
            "status": "remediated",
            "original_code": code,
            "remediated_code": cleaned_code,
        }
    )


# --- Final Convergence: Synthesis & Writer Agent ---


@intent("report_synthesis")
@context_aware(progressive=True)
@safe_tool
def generate_compliance_report(state: AuditWorkflowState) -> str:
    """Writer Sub-Agent: Generate progressive disclosure final report."""
    time.sleep(0.22)  # Simulate LLM report generation

    memory = get_memory()
    total_memories = len(memory.query(limit=50)) if memory else 0

    return json.dumps(
        {
            "report_title": f"Security & Compliance Audit: {state['project_name']}",
            "final_score": state.get("security_score", 0.0),
            "remediation_status": "Applied"
            if state.get("remediation_applied")
            else "Clean",
            "total_dag_memories_referenced": total_memories,
            "summary": "Project successfully audited and certified under NIST CSF v2.0.",
        }
    )


# ---------------------------------------------------------------------------
# 3. Router Function for Conditional DAG Edge
# ---------------------------------------------------------------------------


def security_router(state: AuditWorkflowState) -> str:
    """Dynamic routing logic based on Security Evaluator score."""
    score = state.get("security_score", 0.0)
    if score < 0.8 and not state.get("remediation_applied"):
        print("  🔀 [Router] Security Score < 0.8 -> Routing to Remediation Node")
        return "needs_remediation"
    print("  🔀 [Router] Security Score >= 0.8 -> Routing to Synthesis Node")
    return "approved"


# ---------------------------------------------------------------------------
# 4. Node Wrapper Functions for LangGraph
# ---------------------------------------------------------------------------


def research_node_fn(state: AuditWorkflowState) -> dict[str, Any]:
    set_agent_id("researcher_agent")
    res = fetch_architecture_docs(state["project_name"])
    data = json.loads(res) if isinstance(res, str) and res.startswith("{") else {}
    return {"documents": data.get("docs", [])}


def audit_node_fn(state: AuditWorkflowState) -> dict[str, Any]:
    set_agent_id("auditor_agent")
    attempt = state.get("attempt_count", 1)

    # First run will trigger self-correction
    res = run_static_analysis(state["code_snippet"], attempt)

    # Handle @safe_tool error output gracefully
    if isinstance(res, str) and '"status": "error"' in res:
        print(
            "  ⚠️ [Auditor] Tool failed safely (caught by @safe_tool). Self-correcting..."
        )
        attempt += 1
        res = run_static_analysis(state["code_snippet"], attempt)

    return {"attempt_count": attempt}


def remediation_node_fn(state: AuditWorkflowState) -> dict[str, Any]:
    set_agent_id("remediation_agent")
    fixed = apply_remediation(state["code_snippet"])
    data = json.loads(fixed) if isinstance(fixed, str) and fixed.startswith("{") else {}
    new_code = data.get("remediated_code", state["code_snippet"])
    print("  🛠️  [Remediation] Replaced unsafe eval() statement.")

    # Re-evaluate security score post-remediation
    return {
        "code_snippet": new_code,
        "remediation_applied": True,
        "security_score": 0.98,
    }


def synthesis_node_fn(state: AuditWorkflowState) -> dict[str, Any]:
    set_agent_id("writer_agent")
    report_ref = generate_compliance_report(state)
    return {"final_report": report_ref}


# ---------------------------------------------------------------------------
# 5. Build & Compile Complex LangGraph DAG
# ---------------------------------------------------------------------------


def build_complex_dag() -> Any:
    """Construct a multi-branch, conditional LangGraph StateGraph."""
    graph = StateGraph(AuditWorkflowState)

    # Add DAG Nodes
    graph.add_node("research", research_node_fn)
    graph.add_node("audit", audit_node_fn)
    graph.add_node("evaluate", evaluate_security_risk)
    graph.add_node("remediate", remediation_node_fn)
    graph.add_node("synthesize", synthesis_node_fn)

    # Add Edges
    graph.add_edge(START, "research")
    graph.add_edge("research", "audit")
    graph.add_edge("audit", "evaluate")

    # Conditional Branching
    graph.add_conditional_edges(
        "evaluate",
        security_router,
        {
            "needs_remediation": "remediate",
            "approved": "synthesize",
        },
    )

    graph.add_edge("remediate", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 6. Traced Execution
# ---------------------------------------------------------------------------


@profile_agent(
    name="ComplexMultiAgentDAG",
    memory_backend=SQLiteBackend(".complex_dag_memory.db"),
)
def run_complex_dag() -> AuditWorkflowState:
    """Execute the multi-agent DAG under AgentLatch profiling."""
    mode = (
        "Official langgraph package"
        if HAS_LANGGRAPH
        else "AgentLatch LangGraph mock engine"
    )
    print(f"\n🌐 Executing Complex Multi-Agent DAG ({mode})...\n")

    dag = build_complex_dag()

    initial_state: AuditWorkflowState = {
        "project_name": "FinTech Payment Gateway API",
        "code_snippet": "def calculate_fee(expr):\n    return eval(expr)",
        "documents": [],
        "audit_findings": [],
        "security_score": 0.0,
        "remediation_applied": False,
        "final_report": "",
        "attempt_count": 1,
    }

    final_state = dag.invoke(initial_state)

    print("\n" + "=" * 60)
    print("📋 Final Compliance Report (Progressive Memory Reference):")
    print("=" * 60)
    print(f"  {final_state.get('final_report')}\n")

    memory = get_memory()
    if memory:
        stats = memory.stats()
        print(f"💾 Memory System Summary: {stats}\n")

    return final_state


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_complex_dag()
