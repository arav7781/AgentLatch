"""Tests for the harness framework adapters.

Neither LangGraph nor CrewAI is installed in this environment — and that is the
point.  Every adapter duck-types against a small set of attributes, so the
fakes below reproduce exactly those attributes and nothing else.  If an adapter
ever starts requiring a real framework class, these tests fail.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from agentlatch.harness._types import ToolCall
from agentlatch.harness.adapters.crewai import CrewAIAdapter
from agentlatch.harness.adapters.generic import CallableAdapter
from agentlatch.harness.adapters.langgraph import LangGraphAdapter
from agentlatch.memory.context import get_node_context

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class Recorder:
    """Interceptor that records every call and then lets it through."""

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def __call__(self, call: ToolCall, invoke: Any) -> Any:
        self.calls.append(call)
        return invoke()

    @property
    def last(self) -> ToolCall:
        assert self.calls, "interceptor was never invoked"
        return self.calls[-1]


class Blocker:
    """Interceptor that denies everything without ever calling ``invoke``."""

    def __init__(self, verdict: Any = "BLOCKED") -> None:
        self.verdict = verdict
        self.calls: list[ToolCall] = []

    def __call__(self, call: ToolCall, invoke: Any) -> Any:
        self.calls.append(call)
        return self.verdict


# ---------------------------------------------------------------------------
# CallableAdapter
# ---------------------------------------------------------------------------


def search(query: str, limit: int = 10) -> list[str]:
    """Search the index.

    Extra prose that must not leak into the summary.
    """
    return [query] * limit


def collect(*items, **options) -> int:
    """Collect things."""
    return len(items) + len(options)


def untyped(a, b=3):
    return a


async def fetch(url: str) -> str:
    """Fetch a URL."""
    return f"body:{url}"


def test_discover_builds_schema_from_signature():
    descriptors = {d.name: d for d in CallableAdapter({"search": search}).discover()}
    schema = descriptors["search"].schema

    assert descriptors["search"].summary == "Search the index."
    assert schema["properties"]["query"] == {"type": "str"}
    assert schema["properties"]["limit"] == {"type": "int", "default": 10}
    assert schema["required"] == ["query"]
    assert schema["returns"] == "list[str]"


def test_discover_handles_varargs_and_missing_annotations():
    adapter = CallableAdapter([collect, untyped])
    descriptors = {d.name: d for d in adapter.discover()}

    collect_props = descriptors["collect"].schema["properties"]
    assert collect_props["items"]["variadic"] == "positional"
    assert collect_props["options"]["variadic"] == "keyword"
    assert descriptors["collect"].schema["required"] == []

    untyped_schema = descriptors["untyped"].schema
    assert untyped_schema["properties"]["a"] == {"type": "Any"}
    assert untyped_schema["properties"]["b"] == {"type": "Any", "default": 3}
    assert untyped_schema["required"] == ["a"]
    assert "returns" not in untyped_schema
    assert descriptors["untyped"].summary == "Callable tool 'untyped'."


def test_discover_survives_uninspectable_builtin():
    adapter = CallableAdapter({"len": len})
    (descriptor,) = adapter.discover()
    assert descriptor.name == "len"
    assert isinstance(descriptor.schema, dict)


def test_bind_routes_through_interceptor_and_populates_call():
    recorder = Recorder()
    original = {"search": search}
    bound = CallableAdapter(original).bind(recorder)

    assert bound["search"]("agents", limit=2) == ["agents", "agents"]

    call = recorder.last
    assert call.name == "search"
    assert call.args == ("agents",)
    assert call.kwargs == {"limit": 2}
    assert call.framework == "generic"
    assert call.id


def test_bind_does_not_mutate_the_input_mapping():
    original = {"search": search}
    adapter = CallableAdapter(original)
    bound = adapter.bind(Recorder())

    assert original["search"] is search
    assert adapter.tools["search"] is search
    assert bound is not original
    assert bound["search"] is not search


def test_bind_preserves_functools_wraps():
    bound = CallableAdapter({"search": search}).bind(Recorder())
    wrapper = bound["search"]

    assert wrapper.__name__ == "search"
    assert wrapper.__doc__ == search.__doc__
    assert wrapper.__wrapped__ is search


async def test_bind_supports_async_tools():
    recorder = Recorder()
    bound = CallableAdapter({"fetch": fetch}).bind(recorder)

    assert inspect.iscoroutinefunction(bound["fetch"])
    assert await bound["fetch"]("http://x") == "body:http://x"
    assert recorder.last.name == "fetch"
    assert recorder.last.args == ("http://x",)


async def test_blocking_interceptor_prevents_execution():
    executed: list[str] = []

    def writer(path: str) -> str:
        executed.append(path)
        return "written"

    async def async_writer(path: str) -> str:
        executed.append(path)
        return "written"

    blocker = Blocker({"status": "blocked"})
    bound = CallableAdapter({"writer": writer, "async_writer": async_writer}).bind(
        blocker
    )

    assert bound["writer"]("/etc/passwd") == {"status": "blocked"}
    assert await bound["async_writer"]("/etc/passwd") == {"status": "blocked"}
    assert executed == []
    assert [c.name for c in blocker.calls] == ["writer", "async_writer"]


def test_invoke_is_zero_argument_safe_and_overridable():
    seen: list[tuple[Any, ...]] = []

    def echo(value: str) -> str:
        seen.append((value,))
        return value

    def rewriter(call: ToolCall, invoke: Any) -> Any:
        first = invoke()  # replays the original arguments
        second = invoke("substituted")  # forwards explicit arguments
        return first, second

    bound = CallableAdapter({"echo": echo}).bind(rewriter)
    assert bound["echo"]("original") == ("original", "substituted")
    assert seen == [("original",), ("substituted",)]


def test_callable_adapter_rejects_non_callables():
    with pytest.raises(TypeError):
        CallableAdapter([object()])
    with pytest.raises(TypeError):
        CallableAdapter(42)


# ---------------------------------------------------------------------------
# LangGraph fakes
# ---------------------------------------------------------------------------


class FakeBound:
    """Stands in for a ``PregelNode.bound`` runnable holding ``.func``."""

    def __init__(self, func: Any, tools: list[Any] | None = None) -> None:
        self.func = func
        if tools is not None:
            self.tools = tools


class FakePregelNode:
    """Stands in for ``CompiledStateGraph.nodes[name]``."""

    def __init__(self, func: Any, tools: list[Any] | None = None) -> None:
        self.bound = FakeBound(func, tools)


class FakeRunnableNode:
    """A node that is a Runnable — it exposes ``.invoke``, not ``.func``."""

    def __init__(self, func: Any) -> None:
        self.invoke = func


class FakeCompiledGraph:
    """Minimal compiled-graph surface: ``.nodes`` plus ``.invoke``."""

    def __init__(self, nodes: dict[str, Any]) -> None:
        self.nodes = nodes

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        for name, node in self.nodes.items():
            fn = getattr(getattr(node, "bound", None), "func", None)
            if fn is None:
                fn = getattr(node, "invoke", None)
            if fn is None and callable(node):
                fn = node
            if fn is not None:
                state = fn(state) or state
            del name
        return state


class FakeStateGraph:
    """Uncompiled graph: has ``.compile``, has no ``.invoke``."""

    def __init__(self, nodes: dict[str, Any]) -> None:
        self.nodes = nodes
        self.compiled = False

    def compile(self, **_kwargs: Any) -> FakeCompiledGraph:
        self.compiled = True
        return FakeCompiledGraph(dict(self.nodes))


class FakeLangGraphTool:
    def __init__(self, name: str, description: str, func: Any) -> None:
        self.name = name
        self.description = description
        self.func = func


def planner(state: dict[str, Any]) -> dict[str, Any]:
    """Plan the work."""
    return {**state, "planned": True}


def executor(state: dict[str, Any]) -> dict[str, Any]:
    """Execute the plan."""
    return {**state, "executed": True}


# ---------------------------------------------------------------------------
# LangGraphAdapter
# ---------------------------------------------------------------------------


def _graph() -> FakeCompiledGraph:
    lookup = FakeLangGraphTool("lookup", "Look a fact up.\nMore detail.", untyped)
    return FakeCompiledGraph(
        {
            "__start__": object(),
            "planner": FakePregelNode(planner, tools=[lookup]),
            "executor": FakePregelNode(executor),
        }
    )


def test_langgraph_discover_lists_nodes_and_bound_tools():
    descriptors = LangGraphAdapter(_graph()).discover()
    by_name = {d.name: d for d in descriptors}

    assert "__start__" not in by_name
    assert by_name["planner"].summary == "Plan the work."
    assert by_name["planner"].tags == ["langgraph", "node"]
    assert by_name["executor"].summary == "Execute the plan."

    lookup = by_name["lookup"]
    assert lookup.summary == "Look a fact up."
    assert lookup.tags == ["langgraph", "tool", "node:planner"]
    assert lookup.schema["properties"]["a"] == {"type": "Any"}


def test_langgraph_discover_returns_empty_for_unrecognised_structure():
    class Alien:
        def invoke(self, state: Any) -> Any:
            return state

    assert LangGraphAdapter(Alien()).discover() == []


def test_langgraph_bind_keeps_graph_invocable_and_stamps_node_metadata():
    recorder = Recorder()
    graph = _graph()
    adapter = LangGraphAdapter(graph)
    bound = adapter.bind(recorder)

    assert bound is graph
    result = bound.invoke({"seed": 1})

    assert result == {"seed": 1, "planned": True, "executed": True}
    assert [c.name for c in recorder.calls] == ["planner", "executor"]
    for call in recorder.calls:
        assert call.framework == "langgraph"
        assert call.metadata["node"] == call.name
        assert call.metadata["kind"] == "node"
        assert call.args == ({"seed": 1},) or call.args[0]["seed"] == 1


def test_langgraph_bind_sets_and_clears_node_context():
    seen: list[str | None] = []

    def peek(state: dict[str, Any]) -> dict[str, Any]:
        seen.append(get_node_context())
        return state

    graph = FakeCompiledGraph({"peek": FakePregelNode(peek)})
    LangGraphAdapter(graph).bind(Recorder()).invoke({})

    assert seen == ["peek"]
    assert get_node_context() in (None, "")


def test_langgraph_bind_wraps_runnable_and_plain_function_nodes():
    recorder = Recorder()
    graph = FakeCompiledGraph(
        {"runnable": FakeRunnableNode(planner), "plain": executor}
    )
    adapter = LangGraphAdapter(graph)
    adapter.bind(recorder)

    assert graph.nodes["runnable"].invoke is not planner
    assert graph.nodes["plain"] is not executor
    assert graph.nodes["plain"].__name__ == "executor"

    graph.invoke({})
    assert sorted(c.name for c in recorder.calls) == ["plain", "runnable"]

    adapter.restore()
    assert graph.nodes["runnable"].invoke is planner
    assert graph.nodes["plain"] is executor


def test_langgraph_bind_is_idempotent():
    graph = _graph()
    adapter = LangGraphAdapter(graph)
    adapter.bind(Recorder())
    first = graph.nodes["planner"].bound.func
    adapter.bind(Recorder())

    assert graph.nodes["planner"].bound.func is first


def test_langgraph_compiles_an_uncompiled_state_graph():
    state_graph = FakeStateGraph({"planner": FakePregelNode(planner)})
    adapter = LangGraphAdapter(state_graph)

    assert state_graph.compiled is True
    assert adapter.source is state_graph
    assert isinstance(adapter.target, FakeCompiledGraph)
    assert [d.name for d in adapter.discover()] == ["planner"]


def test_langgraph_rejects_a_non_graph():
    with pytest.raises((TypeError, ImportError)):
        LangGraphAdapter(object())


async def test_langgraph_bind_supports_async_nodes():
    recorder = Recorder()

    async def async_node(state: dict[str, Any]) -> dict[str, Any]:
        """Async node."""
        return {**state, "async": True}

    graph = FakeCompiledGraph({"async_node": FakePregelNode(async_node)})
    LangGraphAdapter(graph).bind(recorder)

    wrapped = graph.nodes["async_node"].bound.func
    assert inspect.iscoroutinefunction(wrapped)
    assert await wrapped({"seed": 1}) == {"seed": 1, "async": True}
    assert recorder.last.name == "async_node"


# ---------------------------------------------------------------------------
# CrewAI fakes
# ---------------------------------------------------------------------------


class FakeFuncTool:
    """A ``@tool``-style CrewAI tool: implementation lives on ``.func``."""

    def __init__(self, name: str, description: str, func: Any) -> None:
        self.name = name
        self.description = description
        self.func = func


class FakeBaseTool:
    """A ``BaseTool`` subclass: implementation lives on ``._run``."""

    name = "file_write"
    description = "Write a file to disk."

    def __init__(self) -> None:
        self.written: list[str] = []

    def _run(self, path: str, contents: str = "") -> str:
        self.written.append(path)
        return f"wrote {path}"


class FakeAgent:
    def __init__(self, role: str, tools: list[Any]) -> None:
        self.role = role
        self.tools = tools


class FakeCrew:
    def __init__(self, agents: list[FakeAgent]) -> None:
        self.agents = agents

    def kickoff(self, inputs: dict[str, Any] | None = None) -> str:
        del inputs
        return "done"


def _crew() -> tuple[FakeCrew, FakeFuncTool, FakeBaseTool]:
    web_search = FakeFuncTool("web_search", "Search the web.\nDetails.", search)
    file_write = FakeBaseTool()
    crew = FakeCrew(
        [
            FakeAgent("researcher", [web_search]),
            FakeAgent("writer", [file_write]),
        ]
    )
    return crew, web_search, file_write


# ---------------------------------------------------------------------------
# CrewAIAdapter
# ---------------------------------------------------------------------------


def test_crewai_discover_covers_every_agent_and_tags_roles():
    crew, _, _ = _crew()
    descriptors = CrewAIAdapter(crew).discover()
    by_name = {d.name: d for d in descriptors}

    assert set(by_name) == {"web_search", "file_write"}
    assert by_name["web_search"].summary == "Search the web."
    assert by_name["web_search"].tags == ["crewai", "researcher"]
    assert by_name["web_search"].schema["required"] == ["query"]
    assert by_name["file_write"].tags == ["crewai", "writer"]
    assert by_name["file_write"].schema["properties"]["contents"]["default"] == ""


def test_crewai_discover_uses_declared_args_schema():
    class ArgsSchema:
        @staticmethod
        def model_json_schema() -> dict[str, Any]:
            return {"type": "object", "properties": {"q": {"type": "string"}}}

    tool = FakeFuncTool("declared", "Declared schema.", search)
    tool.args_schema = ArgsSchema
    crew = FakeCrew([FakeAgent("analyst", [tool])])

    (descriptor,) = CrewAIAdapter(crew).discover()
    assert descriptor.schema == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }


def test_crewai_bind_patches_tools_and_stamps_agent_role():
    recorder = Recorder()
    crew, web_search, file_write = _crew()
    adapter = CrewAIAdapter(crew)
    bound = adapter.bind(recorder)

    assert bound is crew
    assert bound.kickoff() == "done"
    assert web_search.func is not search

    assert web_search.func("agents", limit=1) == ["agents"]
    assert file_write._run("/tmp/out.txt") == "wrote /tmp/out.txt"

    first, second = recorder.calls
    assert first.name == "web_search"
    assert first.framework == "crewai"
    assert first.args == ("agents",)
    assert first.kwargs == {"limit": 1}
    assert first.metadata == {"agent_role": "researcher", "tool": "web_search"}

    assert second.name == "file_write"
    assert second.metadata["agent_role"] == "writer"
    assert file_write.written == ["/tmp/out.txt"]


def test_crewai_restore_puts_originals_back_exactly():
    crew, web_search, file_write = _crew()
    adapter = CrewAIAdapter(crew)

    original_run = file_write._run
    adapter.bind(Recorder())
    assert web_search.func is not search
    assert "_run" in file_write.__dict__

    adapter.restore()

    assert web_search.func is search
    assert "_run" not in file_write.__dict__
    assert file_write._run.__func__ is original_run.__func__
    assert file_write._run("/tmp/x") == "wrote /tmp/x"


def test_crewai_blocking_interceptor_stops_the_tool_body():
    blocker = Blocker("denied")
    crew, _, file_write = _crew()
    CrewAIAdapter(crew).bind(blocker)

    assert file_write._run("/etc/passwd") == "denied"
    assert file_write.written == []
    assert blocker.calls[0].metadata["agent_role"] == "writer"


def test_crewai_bind_is_idempotent_and_tolerates_toolless_agents():
    crew, web_search, _ = _crew()
    crew.agents.append(FakeAgent("idle", []))
    adapter = CrewAIAdapter(crew)
    adapter.bind(Recorder())
    wrapped = web_search.func
    adapter.bind(Recorder())

    assert web_search.func is wrapped


def test_crewai_accepts_a_bare_agent():
    web_search = FakeFuncTool("web_search", "Search the web.", search)
    agent = FakeAgent("solo", [web_search])
    adapter = CrewAIAdapter(agent)

    (descriptor,) = adapter.discover()
    assert descriptor.tags == ["crewai", "solo"]

    recorder = Recorder()
    adapter.bind(recorder)
    web_search.func("q")
    assert recorder.last.metadata["agent_role"] == "solo"


def test_crewai_rejects_a_non_crew():
    with pytest.raises((TypeError, ImportError)):
        CrewAIAdapter(object())


# ---------------------------------------------------------------------------
# Cross-adapter contract
# ---------------------------------------------------------------------------


def test_every_adapter_reports_its_framework_on_the_tool_call():
    generic = Recorder()
    CallableAdapter({"search": search}).bind(generic)["search"]("q")

    graph_recorder = Recorder()
    LangGraphAdapter(_graph()).bind(graph_recorder).invoke({})

    crew_recorder = Recorder()
    crew, web_search, _ = _crew()
    CrewAIAdapter(crew).bind(crew_recorder)
    web_search.func("q")

    assert generic.last.framework == "generic"
    assert graph_recorder.last.framework == "langgraph"
    assert crew_recorder.last.framework == "crewai"
    for recorder in (generic, graph_recorder, crew_recorder):
        for call in recorder.calls:
            assert isinstance(call, ToolCall)
            assert isinstance(call.name, str) and call.name
            assert isinstance(call.args, tuple)
            assert isinstance(call.kwargs, dict)
            assert isinstance(call.metadata, dict)
