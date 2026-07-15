"""FastAPI + LangGraph + Groq agent — demonstrates AgentLatch middleware.

Run::

    export GROQ_API_KEY="your-groq-api-key"
    uv pip install fastapi uvicorn langchain-groq langgraph
    uvicorn examples.fastapi_agent:app --reload

Then open Postman and POST to ``http://localhost:8000/chat`` with body::

    {"message": "How many users are in the database and what's the weather?"}

You'll see the full AgentLatch execution profile in the response body
under the ``_agentlatch`` key, plus ``X-AgentLatch-*`` headers.
"""

from __future__ import annotations

import json
import os
import time

from fastapi import FastAPI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langchain_groq import ChatGroq
from pydantic import BaseModel

from agentlatch import safe_tool, set_dev_mode
from agentlatch.middleware import AgentLatchMiddleware

# ---------------------------------------------------------------------------
# Config: suppress ASCII visuals in server mode, keep structured data
# ---------------------------------------------------------------------------
set_dev_mode(False)

# ---------------------------------------------------------------------------
# LLM Setup — Groq with Llama 4 Scout
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
)

# ---------------------------------------------------------------------------
# Tools — wrapped with @safe_tool for resilience + timing + sampling
# ---------------------------------------------------------------------------


@safe_tool(timeout=10.0, sample_rows=5)
def query_database(sql: str) -> str:
    """Execute a SQL query against the users database.

    Args:
        sql: A valid SQL query string.

    Returns:
        JSON string with query results.
    """
    time.sleep(0.1)  # Simulate DB latency

    if "nonexistent" in sql.lower():
        raise Exception(
            'column "nonexistent_column" does not exist\n'
            "LINE 1: SELECT nonexistent_column FROM users\n"
            "               ^"
        )

    # Mock database result
    rows = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
    ]
    return json.dumps({"rows": rows, "count": len(rows)})


@safe_tool(timeout=5.0)
def get_weather(city: str) -> str:
    """Fetch current weather for a city.

    Args:
        city: The city name.

    Returns:
        JSON string with weather data.
    """
    time.sleep(0.08)  # Simulate API latency
    return json.dumps(
        {"city": city, "temp_c": 22, "condition": "Sunny", "humidity": 65}
    )


# ---------------------------------------------------------------------------
# Register tools with LangChain for the LLM to call
# ---------------------------------------------------------------------------


@langchain_tool
def query_database_tool(sql: str) -> str:
    """Execute a SQL query against the users database. Use this to look up user information."""
    return query_database(sql)


@langchain_tool
def get_weather_tool(city: str) -> str:
    """Fetch current weather data for a city. Use this when asked about weather conditions."""
    return get_weather(city)


TOOLS = [query_database_tool, get_weather_tool]
llm_with_tools = llm.bind_tools(TOOLS)

# Map tool names to callables for dispatch
TOOL_MAP = {
    "query_database_tool": query_database_tool,
    "get_weather_tool": get_weather_tool,
}

# ---------------------------------------------------------------------------
# Simple ReAct Agent Loop
# ---------------------------------------------------------------------------


def run_agent(user_message: str) -> str:
    """Run a simple tool-calling agent loop."""
    messages = [HumanMessage(content=user_message)]

    # Allow up to 5 iterations of tool calling
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls, the agent is done
        if not response.tool_calls:
            return response.content

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            tool_fn = TOOL_MAP.get(tool_name)
            if tool_fn is None:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            else:
                result = tool_fn.invoke(tool_args)

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    return messages[-1].content if messages else "No response generated."


# ---------------------------------------------------------------------------
# FastAPI App with AgentLatch Middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgentLatch Demo API",
    description="LangGraph + Groq agent with AgentLatch middleware",
)

app.add_middleware(
    AgentLatchMiddleware,
    inject_profile=True,
    trace_name="GroqLlamaAgent",
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat endpoint — send a message, get an agent response.

    The AgentLatch middleware automatically injects execution profiling
    into the response headers and body.
    """
    answer = run_agent(request.message)
    return ChatResponse(response=answer)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "agentlatch": "active"}
