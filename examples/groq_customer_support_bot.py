"""Complex Multi-Agent Airline Customer Support DAG — ChatGroq + Tavily + AgentLatch.

Features:
1. **Multi-Agent Specialist DAG**:
   - Primary Router Node -> Sub-Agents (Flight Specialist, Booking Specialist, Policy Auditor, Web Researcher) -> Synthesizer Node.
   - Multi-node state machine using LangGraph `StateGraph`.
2. **Cross-Node Memory Intelligence**:
   - Booking Specialist automatically inspects upstream AgentLatch memory (`get_memory().query(intent="flight_fetch")`)
     to discover the passenger's destination airport and schedule without requiring duplicate user prompts!
3. **Complete AgentLatch Decorator Suite**:
   - `@profile_agent`: Traces full multi-agent graph hierarchy, tool timing, and memory snapshots.
   - `@safe_tool`: Enforces cross-platform time budgets and converts exceptions into LLM retry instructions.
   - `@intent`: Indexes tool calls into searchable intent streams across sub-agents.
   - `@context_aware`: Records memory snapshots with `delta=True` (diff storage) and `progressive=True` (payload summaries).
4. **Zero OpenAI/Anthropic Dependencies**:
   - Uses `ChatGroq` (`llama-3.1-8b-instant`), Tavily Web Search, and a local TF-IDF policy retriever.
5. **Interactive Terminal Session**:
   - Interactive user prompt loop that renders AgentLatch's visual flamegraph and memory breakdown on exit.

Usage:
    export GROQ_API_KEY="your-groq-api-key"
    export TAVILY_API_KEY="your-tavily-api-key"
    python examples/groq_customer_support_bot.py
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

import pandas as pd
import requests

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
# API Key Verification & Optional Imports
# ---------------------------------------------------------------------------

GROQ_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY")

try:
    from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_groq import ChatGroq
    HAS_GROQ = bool(GROQ_KEY)
except ImportError:
    HAS_GROQ = False

try:
    from langchain_community.tools import TavilySearchResults
    HAS_TAVILY = bool(TAVILY_KEY)
except ImportError:
    HAS_TAVILY = False

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

# ---------------------------------------------------------------------------
# 1. Database Initialization & Local Policy Retriever
# ---------------------------------------------------------------------------

DB_FILE = "travel2.sqlite"
BACKUP_DB = "travel2.backup.sqlite"

def prepare_database(db_path: str = DB_FILE, backup_path: str = BACKUP_DB) -> str:
    """Download SQLite travel DB if missing and update timestamps to relative present."""
    db_url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
    if not os.path.exists(db_path):
        print("⏬ Downloading SQLite travel database...")
        res = requests.get(db_url, timeout=30)
        res.raise_for_status()
        with open(db_path, "wb") as f:
            f.write(res.content)
        shutil.copy(db_path, backup_path)

    # Shift timestamps relative to present
    shutil.copy(backup_path, db_path)
    conn = sqlite3.connect(db_path)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn).name.tolist()
    tdf = {t: pd.read_sql(f"SELECT * from {t}", conn) for t in tables}

    if "flights" in tdf and not tdf["flights"].empty:
        example_time = pd.to_datetime(tdf["flights"]["actual_departure"].replace(r"\N", pd.NaT)).max()
        if pd.notnull(example_time):
            current_time = pd.to_datetime("now").tz_localize(example_time.tz)
            time_diff = current_time - example_time

            if "bookings" in tdf:
                tdf["bookings"]["book_date"] = (
                    pd.to_datetime(tdf["bookings"]["book_date"].replace(r"\N", pd.NaT), utc=True) + time_diff
                )

            datetime_columns = ["scheduled_departure", "scheduled_arrival", "actual_departure", "actual_arrival"]
            for col in datetime_columns:
                if col in tdf["flights"]:
                    tdf["flights"][col] = pd.to_datetime(tdf["flights"][col].replace(r"\N", pd.NaT)) + time_diff

            for table_name, df in tdf.items():
                df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.commit()

    conn.close()
    return db_path


class LocalTFIDFRetriever:
    """Pure Python TF-IDF Cosine Similarity Policy Retriever (No OpenAI Key Required)."""

    def __init__(self, docs: list[dict[str, str]]) -> None:
        self.docs = docs
        self.vocabulary: dict[str, int] = {}
        self.doc_vectors: list[dict[int, float]] = []

        tokenized_docs = [self._tokenize(doc["page_content"]) for doc in docs]
        all_words = set(w for tokens in tokenized_docs for w in tokens)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}

        num_docs = len(docs)
        df_counts = Counter(w for tokens in tokenized_docs for w in set(tokens))
        self.idf = {
            w: math.log((1 + num_docs) / (1 + df_counts[w])) + 1.0
            for w in all_words
        }

        for tokens in tokenized_docs:
            tf = Counter(tokens)
            vec = {}
            for word, count in tf.items():
                idx = self.vocabulary[word]
                vec[idx] = (count / max(1, len(tokens))) * self.idf[word]
            self.doc_vectors.append(vec)

    @classmethod
    def from_url(cls, url: str) -> LocalTFIDFRetriever:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        faq_text = res.text
        sections = [
            {"page_content": txt.strip()}
            for txt in re.split(r"(?=\n##)", faq_text)
            if txt.strip()
        ]
        return cls(sections)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _cosine_similarity(self, vec1: dict[int, float], vec2: dict[int, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        sum1 = sum(val**2 for val in vec1.values())
        sum2 = sum(val**2 for val in vec2.values())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        return numerator / denominator if denominator else 0.0

    def query(self, query_str: str, k: int = 2) -> list[str]:
        tokens = self._tokenize(query_str)
        tf = Counter(tokens)
        query_vec = {}
        for word, count in tf.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                query_vec[idx] = (count / max(1, len(tokens))) * self.idf.get(word, 1.0)

        scores = [
            (idx, self._cosine_similarity(query_vec, doc_vec))
            for idx, doc_vec in enumerate(self.doc_vectors)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.docs[idx]["page_content"] for idx, score in scores[:k]]


POLICY_RETRIEVER: Optional[LocalTFIDFRetriever] = None

def get_policy_retriever() -> LocalTFIDFRetriever:
    global POLICY_RETRIEVER
    if POLICY_RETRIEVER is None:
        url = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
        POLICY_RETRIEVER = LocalTFIDFRetriever.from_url(url)
    return POLICY_RETRIEVER


# ---------------------------------------------------------------------------
# 2. AgentLatch Instrumented Domain Tools
# ---------------------------------------------------------------------------

@intent("policy_lookup")
@context_aware(progressive=True)
@safe_tool(timeout=5.0)
def lookup_policy(query: str) -> str:
    """Consult Swiss Airlines company policy FAQ before performing changes or bookings."""
    retriever = get_policy_retriever()
    results = retriever.query(query, k=2)
    combined = "\n\n".join(results) if results else "No specific policy document found for query."
    if len(combined) > 1500:
        combined = combined[:1500] + "\n... (policy snippet truncated for brevity)"
    return combined


@intent("flight_fetch")
@context_aware
@safe_tool(timeout=5.0)
def fetch_user_flight_information(passenger_id: str = "3442 587242") -> list[dict[str, Any]]:
    """Fetch user's booked tickets, flight details, and seat assignments by passenger ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = """
    SELECT t.ticket_no, t.book_ref, f.flight_id, f.flight_no, f.departure_airport,
           f.arrival_airport, f.scheduled_departure, f.scheduled_arrival,
           bp.seat_no, tf.fare_conditions
    FROM tickets t
    JOIN ticket_flights tf ON t.ticket_no = tf.ticket_no
    JOIN flights f ON tf.flight_id = f.flight_id
    LEFT JOIN boarding_passes bp ON bp.ticket_no = t.ticket_no AND bp.flight_id = f.flight_id
    WHERE t.passenger_id = ?
    """
    cursor.execute(query, (passenger_id,))
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


@intent("flight_search")
@context_aware
@safe_tool(timeout=5.0)
def search_flights(
    departure_airport: Optional[str] = None,
    arrival_airport: Optional[str] = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search available flights by departure and arrival airports."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT * FROM flights WHERE 1=1"
    params = []
    if departure_airport:
        query += " AND departure_airport = ?"
        params.append(departure_airport)
    if arrival_airport:
        query += " AND arrival_airport = ?"
        params.append(arrival_airport)
    query += " LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


@intent("ticket_update")
@context_aware(delta=True)
@safe_tool(timeout=5.0)
def update_ticket_to_new_flight(ticket_no: str, new_flight_id: int) -> str:
    """Update user's ticket to a new valid flight ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT flight_id FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    if not cursor.fetchone():
        conn.close()
        return f"No ticket found for ticket_no: {ticket_no}"
    cursor.execute("UPDATE ticket_flights SET flight_id = ? WHERE ticket_no = ?", (new_flight_id, ticket_no))
    conn.commit()
    conn.close()
    return f"Ticket {ticket_no} successfully updated to new flight {new_flight_id}."


@intent("ticket_cancel")
@context_aware(delta=True)
@safe_tool(timeout=5.0)
def cancel_ticket(ticket_no: str) -> str:
    """Cancel user's ticket and remove flight reservation."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ticket_flights WHERE ticket_no = ?", (ticket_no,))
    conn.commit()
    conn.close()
    return f"Ticket {ticket_no} successfully cancelled."


@intent("hotel_search")
@context_aware
@safe_tool(timeout=5.0)
def search_hotels(location: Optional[str] = None, name: Optional[str] = None) -> list[dict[str, Any]]:
    """Search hotel options by location or hotel name."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT * FROM hotels WHERE 1=1"
    params = []
    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


@intent("car_search")
@context_aware
@safe_tool(timeout=5.0)
def search_car_rentals(location: Optional[str] = None, name: Optional[str] = None) -> list[dict[str, Any]]:
    """Search car rental options by location or rental company name."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT * FROM car_rentals WHERE 1=1"
    params = []
    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


@intent("excursion_search")
@context_aware
@safe_tool(timeout=5.0)
def search_trip_recommendations(location: Optional[str] = None) -> list[dict[str, Any]]:
    """Search local trip recommendations and excursions."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []
    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    cursor.execute(query, params)
    rows = cursor.fetchall()
    cols = [col[0] for col in cursor.description]
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


@intent("web_search")
@context_aware(progressive=True)
@safe_tool(timeout=10.0)
def web_search(query: str) -> str:
    """Execute live web search using Tavily Search API."""
    if HAS_TAVILY:
        tavily_tool = TavilySearchResults(max_results=2)
        res = tavily_tool.invoke({"query": query})
        return str(res)
    return f"[Mock Web Search Result for '{query}']: Swiss Airlines operates flight routes worldwide."


# Tool Registry
FLIGHT_TOOLS = [fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket]
BOOKING_TOOLS = [search_hotels, search_car_rentals, search_trip_recommendations]
POLICY_TOOLS = [lookup_policy]
WEB_TOOLS = [web_search]

# ---------------------------------------------------------------------------
# 3. LangGraph Complex Multi-Agent State & Nodes
# ---------------------------------------------------------------------------

class CustomerSupportState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent_target: str
    flight_data: Optional[list[dict[str, Any]]]
    policy_notes: Optional[str]
    booking_options: Optional[list[dict[str, Any]]]
    final_output: str


def get_groq_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_retries=3,
        groq_api_key=GROQ_KEY,
    )


# Node 1: Primary Dispatcher Router
def router_node(state: CustomerSupportState) -> dict[str, Any]:
    """Inspect user input and route to specialized sub-agent."""
    set_node_context("router")
    set_agent_id("primary_dispatcher")
    
    last_msg = state["messages"][-1].content if state["messages"] else ""
    text_lower = str(last_msg).lower()

    if any(k in text_lower for k in ["policy", "baggage", "luggage", "rules", "refund", "cancel policy"]):
        target = "policy_auditor"
    elif any(k in text_lower for k in ["hotel", "car", "rental", "excursion", "tour", "activity", "booking"]):
        target = "booking_specialist"
    elif any(k in text_lower for k in ["flight", "ticket", "seat", "departure", "reschedule"]):
        target = "flight_specialist"
    elif any(k in text_lower for k in ["news", "search", "web", "weather", "status"]):
        target = "web_researcher"
    else:
        target = "flight_specialist"

    print(f"  🔀 [Router Node] Classified query intent -> Target: '{target}'")
    return {"intent_target": target}


# Router Edge Evaluator
def route_next_agent(state: CustomerSupportState) -> str:
    return state.get("intent_target", "flight_specialist")


# Node 2: Flight Specialist Node
def flight_specialist_node(state: CustomerSupportState, config: RunnableConfig) -> dict[str, Any]:
    """Handles flight retrieval, ticket updates, and scheduling."""
    set_node_context("flight_specialist")
    set_agent_id("flight_agent")
    
    llm = get_groq_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Flight Specialist Agent for Swiss Airlines.\n"
            "Use flight tools to answer queries, check reservations, or search flights.\n"
            "Passenger ID: {user_info}"
        ),
        ("placeholder", "{messages}"),
    ]).partial(user_info=config.get("configurable", {}).get("passenger_id", "3442 587242"))

    runnable = prompt | llm.bind_tools(FLIGHT_TOOLS)
    res = runnable.invoke(state)

    # If tool calls returned, execute them
    if getattr(res, "tool_calls", None):
        print("  ✈️ [Flight Specialist] Executing flight tools...")
        tool_results = []
        for tc in res.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            if tool_name == "fetch_user_flight_information":
                pid = config.get("configurable", {}).get("passenger_id", "3442 587242")
                out = fetch_user_flight_information(passenger_id=pid)
            elif tool_name == "search_flights":
                out = search_flights(**tool_args)
            elif tool_name == "update_ticket_to_new_flight":
                out = update_ticket_to_new_flight(**tool_args)
            elif tool_name == "cancel_ticket":
                out = cancel_ticket(**tool_args)
            else:
                out = "Tool not found"
            tool_results.append(out)

        # Synthesize final response
        summary_prompt = f"Summarize flight details clearly for the user: {tool_results}"
        final_ai = llm.invoke([HumanMessage(content=summary_prompt)])
        return {"flight_data": tool_results, "messages": [final_ai]}

    return {"messages": [res]}


# Node 3: Booking Specialist Node (With Cross-Node AgentLatch Memory Querying!)
def booking_specialist_node(state: CustomerSupportState, config: RunnableConfig) -> dict[str, Any]:
    """Handles accommodation and transport reservations, querying upstream flight memory."""
    set_node_context("booking_specialist")
    set_agent_id("booking_agent")
    
    llm = get_groq_llm()
    
    # Query upstream AgentLatch memory for past flight details
    memory = get_memory()
    past_flight_snapshots = memory.query(intent="flight_fetch", limit=3) if memory else []
    
    location_hint = "Zurich"  # Default fallback
    if past_flight_snapshots:
        print(f"  🧠 [Booking Specialist] Retreived {len(past_flight_snapshots)} upstream flight snapshots from AgentLatch Memory!")
        snap = past_flight_snapshots[0]
        snapshot_payload = snap.get("output_payload") if isinstance(snap, dict) else getattr(snap, "output_payload", None)
        if isinstance(snapshot_payload, list) and snapshot_payload:
            arrival_ap = snapshot_payload[0].get("arrival_airport")
            if arrival_ap:
                location_hint = arrival_ap

    print(f"  🏨 [Booking Specialist] Searching hotels & car rentals at location '{location_hint}'...")
    hotels = search_hotels(location=location_hint)
    cars = search_car_rentals(location=location_hint)

    summary_prompt = (
        f"Available booking options at {location_hint}:\n"
        f"Hotels: {hotels[:2]}\nCar Rentals: {cars[:2]}\n"
        f"Provide a helpful travel package recommendation."
    )
    res = llm.invoke([HumanMessage(content=summary_prompt)])
    return {"booking_options": hotels + cars, "messages": [res]}


# Node 4: Policy Compliance Auditor Node
def policy_auditor_node(state: CustomerSupportState) -> dict[str, Any]:
    """Audits Swiss Airlines FAQ policies."""
    set_node_context("policy_auditor")
    set_agent_id("policy_agent")

    last_user_msg = state["messages"][-1].content if state["messages"] else "baggage allowance"
    print(f"  📜 [Policy Auditor] Checking FAQ policies for: '{last_user_msg}'...")

    policy_text = lookup_policy(query=str(last_user_msg))

    llm = get_groq_llm()
    audit_prompt = (
        f"Swiss Airlines Policy Document:\n{policy_text}\n\n"
        f"User Query: {last_user_msg}\n"
        f"Summarize exact rules, limits, or requirements clearly."
    )
    res = llm.invoke([HumanMessage(content=audit_prompt)])
    return {"policy_notes": policy_text, "messages": [res]}


# Node 5: Web Research Node
def web_researcher_node(state: CustomerSupportState) -> dict[str, Any]:
    """Executes external web queries using Tavily API."""
    set_node_context("web_researcher")
    set_agent_id("web_agent")

    last_user_msg = state["messages"][-1].content if state["messages"] else "Swiss Airlines info"
    print(f"  🌐 [Web Researcher] Executing Tavily web search for: '{last_user_msg}'...")

    search_res = web_search(query=str(last_user_msg))

    llm = get_groq_llm()
    res = llm.invoke([
        SystemMessage(content="Summarize the web search results accurately."),
        HumanMessage(content=f"Search Query: {last_user_msg}\nResults: {search_res}")
    ])
    return {"messages": [res]}


# ---------------------------------------------------------------------------
# 4. Build & Compile Complex LangGraph Workflow
# ---------------------------------------------------------------------------

def create_complex_support_graph() -> Any:
    """Construct multi-agent DAG workflow."""
    if not HAS_GROQ or not HAS_LANGGRAPH:
        print("⚠️ Missing langchain-groq or langgraph dependencies.")
        return None

    builder = StateGraph(CustomerSupportState)

    # Add Specialist Sub-Agent Nodes
    builder.add_node("router", router_node)
    builder.add_node("flight_specialist", flight_specialist_node)
    builder.add_node("booking_specialist", booking_specialist_node)
    builder.add_node("policy_auditor", policy_auditor_node)
    builder.add_node("web_researcher", web_researcher_node)

    # Entry edge
    builder.add_edge(START, "router")

    # Conditional Routing Edges from Router -> Sub-Agents
    builder.add_conditional_edges(
        "router",
        route_next_agent,
        {
            "flight_specialist": "flight_specialist",
            "booking_specialist": "booking_specialist",
            "policy_auditor": "policy_auditor",
            "web_researcher": "web_researcher",
        },
    )

    # Sub-agents terminal edge -> END
    builder.add_edge("flight_specialist", END)
    builder.add_edge("booking_specialist", END)
    builder.add_edge("policy_auditor", END)
    builder.add_edge("web_researcher", END)

    memory_saver = MemorySaver()
    return builder.compile(checkpointer=memory_saver)


# ---------------------------------------------------------------------------
# 5. Traced Interactive Execution Entry Point
# ---------------------------------------------------------------------------

@profile_agent(
    name="GroqComplexSupportBot",
    memory_backend=SQLiteBackend(".complex_support_memory.db"),
)
def run_groq_complex_support_bot(
    passenger_id: str = "3442 587242",
) -> None:
    """Execute complex multi-agent customer support workflow traced by AgentLatch."""
    print("\n✈️ Initializing Swiss Airlines Complex Multi-Agent Customer Support System...")
    prepare_database()

    graph = create_complex_support_graph()
    if graph is None:
        return

    config = {
        "configurable": {
            "passenger_id": passenger_id,
            "thread_id": "complex_session_001",
        }
    }

    print("\n💬 Swiss Airlines Multi-Agent Interactive Terminal")
    print("   Ask any flight, hotel, policy, or web search query (or type 'exit' to view AgentLatch flamegraph):\n")

    # Non-interactive CLI argument execution
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        print(f"👤 User: {query_text}\n" + "-" * 50)
        _process_multi_agent_query(graph, query_text, config)
        return

    # Non-interactive pipe execution
    if not sys.stdin.isatty():
        query_text = sys.stdin.read().strip()
        if query_text:
            print(f"👤 User: {query_text}\n" + "-" * 50)
            _process_multi_agent_query(graph, query_text, config)
        return

    # Interactive turn-by-turn CLI session
    while True:
        try:
            user_input = input("👤 User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user.")
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q", "done"):
            print("\nEnding session. Generating AgentLatch multi-agent flamegraph profile...")
            break

        print("-" * 50)
        _process_multi_agent_query(graph, user_input, config)
        print("-" * 50 + "\n")

    memory = get_memory()
    if memory:
        stats = memory.stats()
        print(f"\n💾 AgentLatch Multi-Agent Snapshots Recorded: {stats.get('snapshot_count', 0)}\n")


def _process_multi_agent_query(graph: Any, query: str, config: dict[str, Any]) -> None:
    """Stream response across multi-agent nodes."""
    try:
        events = graph.stream(
            {"messages": [HumanMessage(content=query)]},
            config,
            stream_mode="values",
        )
        last_printed_id = None
        for event in events:
            msgs = event.get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                msg_id = getattr(last_msg, "id", None) or str(id(last_msg))
                msg_type = getattr(last_msg, "type", "")
                content = getattr(last_msg, "content", "")

                if msg_type == "ai" and content and msg_id != last_printed_id:
                    if isinstance(content, str):
                        print(f"\n🤖 Agent Response:\n{content}\n")
                        last_printed_id = msg_id
                    elif isinstance(content, list) and content:
                        text_bits = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                        if text_bits:
                            print(f"\n🤖 Agent Response:\n{' '.join(text_bits)}\n")
                            last_printed_id = msg_id
    except Exception as err:
        print(f"\n⚠️ Error processing multi-agent query: {err}\n")


if __name__ == "__main__":
    if not GROQ_KEY:
        print("❌ Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it via: export GROQ_API_KEY='your_api_key'")
        sys.exit(1)

    run_groq_complex_support_bot()
