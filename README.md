![AgentLatch Banner](banner.png)

**Terminal-native agent resilience middleware for Python.**

AgentLatch is a zero-dependency framework that makes AI agents resilient, observable, and memory-aware. It solves three massive pain points in agent development:

1. **Silent Tool Failures** — When an LLM executes a tool that crashes, AgentLatch intercepts the Python exception, prevents a runtime crash, and feeds a structured JSON error back to the LLM so it can self-correct.

2. **Blind Latency** — It tracks millisecond execution time of LLM vs. tools and prints a color-coded ASCII flamegraph directly in the terminal using the `rich` library. No API keys, no dashboards, no cloud.

3. **Context Rot in Multi-Agent Workflows** — In long-running DAG pipelines, LLMs forget key information and repeat mistakes. `@context_aware` creates structured memory snapshots with delta updates, progressive disclosure, and intent tagging so sub-agents can query upstream results without re-executing.

## Quick Install

To install the core package:
```bash
pip install agentlatch
```

To install with **FastAPI/Starlette HTTP Middleware** support:
```bash
pip install "agentlatch[server]"
```

To install with **vector memory backends**:
```bash
pip install "agentlatch[vector]"    # PostgreSQL + pgvector
pip install "agentlatch[qdrant]"     # Qdrant
pip install "agentlatch[graph]"      # Neo4j
pip install "agentlatch[all-memory]" # All backends
```

## Setup Virtual Environment

Before installing the package, it is recommended to create and activate a virtual environment to isolate your dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate
```

## Core Use Cases

AgentLatch is built to address critical requirements of production-ready AI agents:

* **Exception Interception & Self-Correction**: Raw tool crashes throw exceptions that break agent runs. `@safe_tool` translates these exceptions into structured JSON error prompts. The LLM parses the error feedback and corrects its parameters or query dynamically without failing.
* **Context Window Budgeting (Sampling)**: Large list returns or massive token blocks can overflow context windows. `@safe_tool(max_response_tokens=N, sample_rows=N)` automatically truncates response strings and lists, injecting metadata so the LLM is aware of the sampling.
* **Execution Timeline and Flamegraphs**: Track down slow operations (e.g. database lookups, external APIs). `@profile_agent` creates a visual breakdown of your tool durations vs. LLM reasoning directly in the CLI.
* **HTTP Endpoint Observability**: Debug agent execution flows during integration testing. `AgentLatchMiddleware` injects detailed trace logs directly into your Starlette/FastAPI headers and JSON response bodies for Postman or cURL debugging.
* **Structured Memory for Multi-Agent DAGs**: `@context_aware` creates memory snapshots after tool calls with intent tagging, delta updates, and progressive disclosure. Sub-agents in a DAG can query upstream results without re-executing expensive operations.

## Usage


### 1. Resilient Decorators
```python
from agentlatch import profile_agent, safe_tool

@safe_tool
def query_database(sql: str) -> str:
    """This tool is now protected — exceptions become JSON errors."""
    import sqlite3
    conn = sqlite3.connect("my.db")
    return str(conn.execute(sql).fetchall())

@safe_tool(timeout=5.0)
def call_api(url: str) -> str:
    """This tool has a 5-second cross-platform timeout."""
    import requests
    return requests.get(url).text

@profile_agent
def run_agent():
    """The agent loop — traced and visualized automatically."""
    result = query_database("SELECT * FROM users")
    weather = call_api("https://api.weather.com/sf")
    return f"Got {result} and {weather}"

run_agent()
```

### 2. Smart Response Sampling
Prevent large tool outputs from blowing up your LLM context window:
```python
# Limit response to ~2048 tokens and keep only first 10 list items/rows
@safe_tool(max_response_tokens=2048, sample_rows=10)
def fetch_large_dataset():
    # Returns 1,000 DB records. AgentLatch will slice to 10
    # and append sampling metadata: {"_agentlatch_sampled": true, "shown": 10, "total": 1000}
    ...
```

### 3. FastAPI / Starlette HTTP Middleware (Postman Visibility)
Get instant visibility into your agent execution flow directly in your API responses when testing via Postman or curl:
```python
from fastapi import FastAPI
from agentlatch.middleware import AgentLatchMiddleware

app = FastAPI()

# Adds timing headers and appends trace data to JSON responses
app.add_middleware(
    AgentLatchMiddleware,
    inject_profile=True,  # Appends "_agentlatch" to JSON response body
    trace_name="MyChatAgent"
)
```

## What Happens

1. **Execution**: Every `@safe_tool` call is timed, protected, and sampled.
2. **On Error**: Instead of crashing, the tool returns a JSON error string:
   ```json
   {
     "status": "error",
     "error_type": "ProgrammingError",
     "message": "column 'age' does not exist",
     "instruction": "The tool execution failed. Review your parameters and retry with corrected inputs."
   }
   ```
3. **On Completion (CLI)**: A rich flamegraph is printed to the terminal in development mode.
4. **On Completion (HTTP / Postman)**:
   * **Headers tab**:
     ```
     X-AgentLatch-Version: 0.1.0
     X-AgentLatch-Duration-Ms: 1234
     X-AgentLatch-Tools-Ms: 850
     X-AgentLatch-Errors: 1
     ```
   * **Response Body**:
     ```json
     {
       "response": "Based on the database...",
       "_agentlatch": {
         "version": "0.1.0",
         "trace_id": "abc-123",
         "total_ms": 1234,
         "tool_ms": 850,
         "llm_reasoning_ms": 384,
         "tools": [
           {"name": "query_database", "duration_ms": 305, "status": "success"}
         ],
         "errors_count": 0
       }
     }
     ```

## Features

| Feature | Description |
|---------|-------------|
| `@safe_tool` | Wraps any function — catches exceptions, returns JSON errors |
| `@safe_tool(timeout=N)` | Adds a thread-based timeout (cross-platform) |
| `@safe_tool(sample_rows=N)` | Automatically slices massive JSON list outputs to first N items |
| `@safe_tool(max_response_tokens=N)` | Truncates tool string responses if they exceed approximate token budget |
| `@context_aware` | Creates structured memory snapshots after each successful tool call |
| `@context_aware(delta=True)` | Stores only the diff from the last output — reduces storage and token cost |
| `@context_aware(progressive=True)` | Returns a lightweight reference; full data stored in memory |
| `@intent("tag")` | Tags tool calls with intent labels for cross-node memory retrieval |
| `@profile_agent` | Traces the full agent loop, initializes memory, renders the flamegraph |
| `AgentLatchMiddleware` | Starlette/FastAPI middleware for Postman & curl trace observability |
| Pluggable Memory Backends | SQLite (default), PostgreSQL+pgvector, Qdrant, Neo4j |
| Async support | All decorators work with `async def` functions |
| Dev Mode Guard | Automatically suppresses ASCII visuals in production (`AGENTLATCH_ENV=production`) |
| Framework agnostic | Works with LangGraph, AutoGen, CrewAI, or vanilla scripts |

## Memory System

### Basic Usage
```python
from agentlatch import context_aware, intent, safe_tool, profile_agent

@intent("database_query")
@context_aware(delta=True)
@safe_tool
def query_db(sql: str) -> str:
    """Memory-aware tool with delta tracking."""
    import sqlite3
    conn = sqlite3.connect("my.db")
    return str(conn.execute(sql).fetchall())

@profile_agent  # Auto-initializes SQLite memory
def run_agent():
    # First call: stores full snapshot in memory.
    result = query_db("SELECT * FROM users")
    # Second call: stores only the delta (changed rows).
    result = query_db("SELECT * FROM users WHERE active=1")
    return result
```

### Official LangGraph StateGraph Integration
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agentlatch import context_aware, intent, safe_tool, profile_agent, get_memory

# 1. Define State
class AgentState(TypedDict):
    query: str
    documents: list[str]
    analysis: str

# 2. Define Traced & Memory-Aware Nodes
@intent("retrieval")
@context_aware
@safe_tool
def retrieve_node(state: AgentState) -> dict:
    return {"documents": ["Doc 1", "Doc 2"]}

@intent("analysis")
@context_aware(delta=True)
@safe_tool
def analyze_node(state: AgentState) -> dict:
    memory = get_memory()
    # Query upstream memory recorded during "retrieval" node
    upstream_docs = memory.query(intent="retrieval") if memory else []
    return {"analysis": f"Analyzed {len(state['documents'])} docs (upstream hits: {len(upstream_docs)})"}

# 3. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("analyze", analyze_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "analyze")
workflow.add_edge("analyze", END)
pipeline = workflow.compile()

# 4. Traced Execution
@profile_agent(name="LangGraphAgent")
def run_langgraph():
    return pipeline.invoke({"query": "LangGraph + AgentLatch", "documents": [], "analysis": ""})
```

### Multi-Agent DAG (Leader / Sub-Agent)
```python
from agentlatch import context_aware, intent, safe_tool, profile_agent, SQLiteBackend
from agentlatch.memory.context import get_memory, set_agent_id, set_node_context

# --- Sub-Agent Tools ---
@intent("research")
@context_aware
@safe_tool
def search_docs(query: str) -> str:
    return '{"results": ["doc1", "doc2"]}'

@intent("analyze")
@context_aware(delta=True)
@safe_tool
def analyze(data: str) -> str:
    return '{"insight": "growth is 12%"}'

# --- Leader Agent ---
@profile_agent(name="LeaderAgent", memory_backend=SQLiteBackend(".agent.db"))
def run_pipeline():
    # Phase 1: Research node
    set_node_context("research_node")
    set_agent_id("researcher")
    docs = search_docs("AI agents")

    # Phase 2: Analysis node (can query upstream memory)
    set_node_context("analysis_node")
    set_agent_id("analyst")
    memory = get_memory()
    research = memory.query(intent="research", limit=5)  # Cross-node query!
    result = analyze(str(research))
    return result
```

### Custom Memory Backend
```python
from agentlatch import profile_agent, SQLiteBackend

# Persistent file-based memory
@profile_agent(memory_backend=SQLiteBackend(".agentlatch.db"))
def persistent_agent():
    ...

# Disable memory entirely
@profile_agent(enable_memory=False)
def no_memory_agent():
    ...
```

## Running Examples

```bash
# Vanilla agent with a forced failure + self-correction
python examples/vanilla_agent.py

# LangGraph StateGraph pipeline
python examples/langgraph_agent.py

# Multi-agent DAG with memory (leader + sub-agents)
python examples/memory_langgraph_agent.py

# Enterprise Complex Multi-Agent DAG (Parallel branches, conditional router, self-correction)
python examples/complex_multiagent_dag.py

# Real ChatGroq + LangGraph Agent (requires GROQ_API_KEY)
export GROQ_API_KEY="your-groq-key"
python examples/groq_langgraph_agent.py

# FastAPI + LangGraph + Groq Agent (requires GROQ_API_KEY)
export GROQ_API_KEY="your-groq-key"
uvicorn examples.fastapi_agent:app --reload
```

## Running Tests

```bash
uv pip install -e ".[server]"
pytest tests/ -v
```

## Development Plans

All detailed design documents and implementation plans for the development phases are included directly in the package under the `agentlatch.plans` subpackage (located inside the [agentlatch/plans/](file:///Users/aravsaxena/Downloads/dao/AgentLatch/agentlatch/plans) directory).

## Architecture

```mermaid
graph TD
    subgraph Decorators
        PA["@profile_agent"] -->|init_trace + init_memory| B(contextvars State)
        CA["@context_aware"] -->|snapshot| MS[MemorySnapshot]
        IN["@intent"] -->|tag| CV[ContextVar: intent]
    end

    subgraph Execution Loop
        B -->|Runs Agent| C[Agent LLM Reasoning]
        C -->|Calls Tool| D["@safe_tool"]
        D -->|start_child| B
        D -->|Executes| E[Wrapped Tool Function]
        E -->|Succeeds| F[Response Content]
        E -->|Throws| G[Structured JSON Error]
        F -->|end_child| B
        G -->|end_child| B
    end

    subgraph Memory Layer
        MS --> MB[MemoryBackend]
        MB --> SQ[SQLiteBackend]
        MB --> PG[PostgresBackend]
        MB --> QD[QdrantBackend]
        MB --> N4[Neo4jBackend]
    end

    subgraph Output
        B -->|finalize_trace| H[render_flamegraph]
        H -->|Prints| I[Terminal CLI Console]
    end
```

- **`contextvars`** — Thread-safe trace and memory propagation without manual IDs
- **`concurrent.futures`** — Cross-platform timeouts (no `signal.alarm`)
- **`sqlite3`** — Zero-dependency default memory backend
- **`rich`** — Premium terminal rendering
- **`starlette`** — Lightweight core HTTP middleware support

## License

MIT
