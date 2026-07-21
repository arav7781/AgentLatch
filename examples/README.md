# AgentLatch Examples & Workflows

This directory contains functional examples of how to integrate AgentLatch into different Python AI agent architectures. It covers raw/vanilla agents, state-machine graphs (e.g. LangGraph), and HTTP middleware endpoints (e.g. FastAPI/Starlette) with LangChain tools.

---

## 1. Vanilla Python Agent Flow

* **File**: [`vanilla_agent.py`](vanilla_agent.py)
* **Description**: Simulates a standard iterative agent reasoning loop. The agent calls a database query tool that deliberately fails on its first attempt, allowing the agent to self-correct and execute a successful retry.
* **Key Features Demonstrated**:
  * `@profile_agent` tracking the main loop and printing the terminal flamegraph.
  * `@safe_tool` catching database Exceptions and converting them to structured JSON.
  * `@safe_tool(timeout=5.0)` wrapping a simulated weather API query.

```bash
# Execute the vanilla agent example
python examples/vanilla_agent.py
```

---

## 2. State-Machine / Graph Workflow (LangGraph Style)

* **File**: [`langgraph_agent.py`](langgraph_agent.py)
* **Description**: Simulates a graph execution loop similar to LangGraph, where each node in the graph represents a distinct operation (retrieval, analysis, generation) passed along an `AgentState` context.
* **Key Features Demonstrated**:
  * Wrapping node methods directly with `@safe_tool`.
  * Wrapping the overall graph execution entry node with `@profile_agent` to trace the entire multi-node traversal timeline.

```bash
# Execute the LangGraph-style agent example
python examples/langgraph_agent.py
```

```mermaid
graph LR
    Start([Start]) --> Retrieve[retrieve_documents Node]
    Retrieve --> Analyze[analyze_with_llm Node]
    Analyze --> Generate[generate_response Node]
    Generate --> End([End])

    classDef nodeStyle fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:#ecf0f1;
    class Retrieve,Analyze,Generate nodeStyle;
```

---

## 3. Production FastAPI & LangChain Integration

* **File**: [`fastapi_agent.py`](fastapi_agent.py)
* **Description**: A full REST API endpoint using FastAPI and LangGraph's ReAct agent pattern. The agent uses LangChain tools under the hood and makes live LLM calls using the Groq API client.
* **Key Features Demonstrated**:
  * `AgentLatchMiddleware` capturing request traces and injecting them into headers and JSON response bodies.
  * LangChain tool decorators wrapping underlying `@safe_tool` routines:
    ```python
    @safe_tool(timeout=10.0, sample_rows=5)
    def query_database(sql: str) -> str:
        ...

    @langchain_tool
    def query_database_tool(sql: str) -> str:
        """LangChain wrapper tool."""
        return query_database(sql)
    ```
  * `set_dev_mode(False)` to suppress CLI terminal visual outputs while retaining structured timing header telemetry.

```bash
# 1. Set your API Key
export GROQ_API_KEY="your-groq-api-key"

# 2. Run the FastAPI dev server
uvicorn examples.fastapi_agent:app --reload
```

To test, send a POST request to `http://localhost:8000/chat`:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "How many users are in the database?"}'
```
You will receive the structured trace response directly in the response headers and inside the JSON payload under the `_agentlatch` key.

---

## 4. Complex Multi-Agent Customer Support DAG (ChatGroq + Tavily)

* **File**: [`groq_customer_support_bot.py`](groq_customer_support_bot.py)
* **Description**: Enterprise-grade multi-agent customer support state-machine using `ChatGroq` (`llama-3.1-8b-instant`) as the reasoning engine and `TavilySearchResults` for live web search queries. Zero OpenAI or Anthropic dependencies required.
* **Architecture**:

```mermaid
graph TD
    Start([START]) --> Router[Primary Dispatcher Router Node]
    
    Router -- flight_inquiry --> FlightAgent[Flight Specialist Agent]
    Router -- booking_inquiry --> BookingAgent[Booking Specialist Agent]
    Router -- policy_inquiry --> PolicyAgent[Policy Compliance Auditor]
    Router -- web_search --> WebAgent[Web Researcher Agent]

    FlightAgent --> FlightTools[Flight SQLite Tools]
    FlightTools --> FlightAgent
    
    BookingAgent -. queries memory .-> Memory[(AgentLatch Memory)]
    Memory -. upstream details .-> BookingAgent
    BookingAgent --> BookingTools[Hotels & Car Rental Tools]
    BookingTools --> BookingAgent
    
    PolicyAgent --> PolicyRetriever[TF-IDF Policy Search]
    
    FlightAgent --> End([END])
    BookingAgent --> End
    PolicyAgent --> End
    WebAgent --> End

    classDef nodeStyle fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    class Router,FlightAgent,BookingAgent,PolicyAgent,WebAgent nodeStyle;
```

* **Key Features Demonstrated**:
  * **Multi-Agent Specialist Routing**: Central router node classifies user intent and dispatches execution to domain-specific sub-agents (`flight_specialist`, `booking_specialist`, `policy_auditor`, `web_researcher`).
  * **Cross-Node Memory Intelligence**: The `booking_specialist` node queries upstream AgentLatch memory (`get_memory().query(intent="flight_fetch")`) to discover destination airport and schedule automatically!
  * **Complete Decorator Stack**:
    * `@profile_agent(name="GroqComplexSupportBot")`: Traces multi-agent execution timeline, tool latencies, and memory snapshot deltas.
    * `@safe_tool(timeout=5.0)`: Wraps all domain tools with cross-platform thread timeouts and error conversion.
    * `@intent(...)`: Indexes tool executions into intent streams for cross-agent querying.
    * `@context_aware`: Records memory snapshots with `delta=True` (incremental diffs) and `progressive=True` (payload reference summaries).
  * **Local TF-IDF Policy Search**: Zero-dependency cosine similarity search over Swiss Airlines policy FAQ documents.

```bash
# 1. Export API Keys
export GROQ_API_KEY="your-groq-api-key"
export TAVILY_API_KEY="your-tavily-api-key"

# 2. Run Interactive CLI Mode
python examples/groq_customer_support_bot.py

# 3. Or Run Single Query Mode
python examples/groq_customer_support_bot.py "Can you check my current flight details?"
```

#### Stopping the Session & Viewing the Flamegraph Report

When running in interactive mode (`👤 User: ` prompt):
* **To stop & view report**: Type `exit` (or `quit`, `q`, `done`) and press **Enter**.
* **Shortcut**: Press **Enter** on an empty prompt, or press `Ctrl + C` / `Ctrl + D`.

Upon exit, `@profile_agent` automatically finalizes the execution trace and prints the color-coded **ASCII Flamegraph & Memory Summary Table** directly in your terminal screen.



