![AgentLatch Banner](banner.png)

**Terminal-native agent resilience middleware for Python.**

AgentLatch is a zero-dependency framework that makes AI agents resilient and observable. It solves two massive pain points in agent development:

1. **Silent Tool Failures** — When an LLM executes a tool that crashes, AgentLatch intercepts the Python exception, prevents a runtime crash, and feeds a structured JSON error back to the LLM so it can self-correct.

2. **Blind Latency** — It tracks millisecond execution time of LLM vs. tools and prints a color-coded ASCII flamegraph directly in the terminal using the `rich` library. No API keys, no dashboards, no cloud.

## Quick Install

```bash
uv pip install -e ".[dev]"
```

## Usage

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
    """This tool has a 5-second timeout."""
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

## What Happens

1. **Startup**: A Claude Code-style "decryption" banner animation plays.
2. **Execution**: Every `@safe_tool` call is timed and protected.
3. **On Error**: Instead of crashing, the tool returns a JSON error string:
   ```json
   {
     "status": "error",
     "error_type": "ProgrammingError",
     "message": "column 'age' does not exist",
     "instruction": "The tool execution failed. Review your parameters and retry with corrected inputs."
   }
   ```
4. **On Completion**: A rich flamegraph is printed to the terminal:
   ```
   ┌─────────────────────────────────────────────────────────┐
   │  ⚡ AGENTLATCH EXECUTION PROFILE                        │
   │  Total: 1.23s │ Tools: 0.85s │ LLM Reasoning: 0.38s    │
   ├─────────────────────────────────────────────────────────┤
   │  ████████████████████████████████████████████ 1.23s      │
   │  ░░░░████████████░░░████░░░░░░░░░░░░░░░░░░░░            │
   │      query_db 0.5s  call_api 0.2s                       │
   │      ▲ ERROR                                             │
   │  Legend: █ LLM  █ Tool (OK)  █ Tool (ERROR)              │
   └─────────────────────────────────────────────────────────┘
   ```

## Features

| Feature | Description |
|---------|-------------|
| `@safe_tool` | Wraps any function — catches exceptions, returns JSON errors |
| `@safe_tool(timeout=N)` | Adds a thread-based timeout (cross-platform) |
| `@profile_agent` | Traces the full agent loop and renders the flamegraph |
| Async support | Both decorators work with `async def` functions |
| Framework agnostic | Works with LangGraph, AutoGen, CrewAI, or vanilla scripts |
| CI-safe | Banner auto-disables in non-TTY environments |

## Running Examples

```bash
# Vanilla agent with a forced failure + self-correction
python examples/vanilla_agent.py

# LangGraph-style state machine
python examples/langgraph_agent.py
```

## Running Tests

```bash
uv pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture

- **`contextvars`** — Thread-safe trace propagation without manual trace IDs
- **`concurrent.futures`** — Cross-platform timeouts (no `signal.alarm`)
- **`rich`** — Premium terminal rendering (the only external dependency)

## License

MIT
