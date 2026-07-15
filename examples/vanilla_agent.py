"""Vanilla agent example — demonstrates AgentLatch with a mock agent loop.

Run:
    python examples/vanilla_agent.py

This simulates an AI agent that:
1. Calls a database query tool (which fails on the first attempt).
2. Receives the structured JSON error instead of crashing.
3. "Self-corrects" by retrying with a fixed query.
4. Calls a weather API tool (succeeds).
5. Prints the AgentLatch execution flamegraph.
"""

from __future__ import annotations

import time

from agentlatch import profile_agent, safe_tool


# ---------------------------------------------------------------------------
# Tools — decorated with @safe_tool for resilience + timing
# ---------------------------------------------------------------------------


@safe_tool
def query_database(sql: str) -> str:
    """Execute a SQL query against the database."""
    time.sleep(0.3)  # simulate DB latency

    # Simulate a failure for bad SQL.
    if "nonexistent_column" in sql:
        raise Exception(
            'column "nonexistent_column" does not exist\n'
            "LINE 1: SELECT nonexistent_column FROM users\n"
            "               ^"
        )

    return '{"rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'


@safe_tool(timeout=5.0)
def call_weather_api(city: str) -> str:
    """Fetch weather data for a given city."""
    time.sleep(0.15)  # simulate HTTP latency
    return f'{{"city": "{city}", "temp_c": 22, "condition": "Sunny"}}'


# ---------------------------------------------------------------------------
# Agent loop — decorated with @profile_agent for tracing + flamegraph
# ---------------------------------------------------------------------------


@profile_agent(name="VanillaAgent")
def run_agent() -> str:
    """Simulate an LLM agent reasoning loop."""

    # Step 1: LLM "thinks" for a moment.
    time.sleep(0.2)

    # Step 2: LLM tries a bad query (will fail gracefully).
    result = query_database("SELECT nonexistent_column FROM users")
    print(f"\n[Agent] Tool response (attempt 1): {result}")

    # Step 3: LLM "reads" the error JSON and retries.
    time.sleep(0.1)  # LLM reasoning
    result = query_database("SELECT id, name FROM users")
    print(f"[Agent] Tool response (attempt 2): {result}")

    # Step 4: LLM calls another tool.
    time.sleep(0.05)  # LLM reasoning
    weather = call_weather_api("San Francisco")
    print(f"[Agent] Weather data: {weather}")

    # Step 5: LLM produces final answer.
    time.sleep(0.1)  # LLM reasoning
    final = (
        "Based on the database, we have 2 users: Alice and Bob. "
        "The weather in San Francisco is 22°C and Sunny."
    )
    print(f"\n[Agent] Final answer: {final}\n")
    return final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_agent()
