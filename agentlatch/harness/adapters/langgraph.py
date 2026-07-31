"""LangGraph adapter for the AgentLatch Harness.

Where :mod:`agentlatch.langgraph` *observes* a graph (timings, state deltas),
this adapter *governs* one: every node execution becomes a
:class:`~agentlatch.harness._types.ToolCall` that the harness may allow,
pause on, or block outright.

The two layers compose.  This adapter sets the same ``_active_node`` context
variable via :func:`agentlatch.memory.context.set_node_context`, so a graph can
be wrapped by ``wrap_langgraph`` for observability *and* bound to a harness for
control, in either order.

**This adapter patches in place.**  LangGraph holds node callables inside
``Pregel`` node objects that the runtime resolves at execution time, so there
is no outer object to wrap that would still be reached by ``.invoke()``.
:meth:`LangGraphAdapter.bind` therefore mutates the compiled graph's nodes and
returns the same object — still fully ``.invoke()``-able — and
:meth:`LangGraphAdapter.restore` puts every original callable back.

LangGraph is never imported at module import time; it is not a hard dependency.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

from agentlatch.harness._types import ToolDescriptor
from agentlatch.harness.adapters.base import FrameworkAdapter
from agentlatch.harness.adapters.generic import (
    Interceptor,
    build_schema,
    route_callable,
    summarize,
)
from agentlatch.memory.context import set_node_context

logger = logging.getLogger("agentlatch.harness.adapters.langgraph")

RESERVED_NODES = frozenset({"__start__", "__end__", "__interrupt__", "START", "END"})

_MARKER = "_agentlatch_harness_bound"


def _require_langgraph() -> Any:
    """Import LangGraph, or raise an error naming the extra to install.

    Returns:
        The imported ``langgraph`` module.

    Raises:
        ImportError: If LangGraph is not installed.
    """
    try:
        import langgraph  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "LangGraphAdapter requires LangGraph. Install it with "
            "`pip install 'agentlatch[langgraph]'` (or `pip install langgraph`)."
        ) from exc
    return langgraph


class LangGraphAdapter(FrameworkAdapter):
    """Routes every LangGraph node execution through a harness interceptor.

    Accepts a **compiled** graph (anything exposing ``.invoke``).  An
    uncompiled ``StateGraph`` — one that still exposes ``.compile()`` and no
    ``.invoke()`` — is compiled with no arguments on construction, and the
    compiled result becomes :attr:`target`; the original is kept on
    :attr:`source`.  Pass a graph you compiled yourself if you need
    checkpointers or interrupts.

    Args:
        target: A compiled LangGraph graph, or a ``StateGraph`` to compile.

    Raises:
        TypeError: If *target* looks like neither a graph nor a compiled graph.
        RuntimeError: If an uncompiled ``StateGraph`` cannot be compiled.
    """

    framework = "langgraph"

    def __init__(self, target: Any) -> None:
        self.source: Any = target
        super().__init__(self._compile_if_needed(target))
        self._patches: list[tuple[Any, str, Any]] = []

    # -- Construction --------------------------------------------------------

    @staticmethod
    def _compile_if_needed(target: Any) -> Any:
        """Return a compiled graph, compiling an uncompiled ``StateGraph``."""
        if callable(getattr(target, "invoke", None)):
            return target

        compile_fn = getattr(target, "compile", None)
        if callable(compile_fn):
            try:
                return compile_fn()
            except Exception as exc:
                raise RuntimeError(
                    "LangGraphAdapter received an uncompiled StateGraph and "
                    "could not compile it automatically. Call `graph.compile("
                    "...)` yourself and pass the compiled graph instead. "
                    f"Underlying error: {exc}"
                ) from exc

        if isinstance(getattr(target, "nodes", None), Mapping):
            # Graph-like enough to introspect and patch, just not invocable.
            return target

        _require_langgraph()
        raise TypeError(
            "LangGraphAdapter expects a compiled LangGraph graph (with "
            "`.invoke`) or a StateGraph (with `.compile`); got "
            f"{type(target).__name__!r}."
        )

    # -- Structure probing ---------------------------------------------------

    def _node_map(self) -> dict[str, Any]:
        """Best-effort ``{node_name: node}`` mapping for the target graph.

        LangGraph's internals move between versions, so this tries the
        documented surfaces in order and gives up quietly rather than raising.
        """
        for probe in (self._nodes_attr, self._nodes_via_get_graph):
            try:
                nodes = probe()
            except Exception:  # noqa: BLE001 - version drift must not raise
                logger.debug("LangGraph node probe failed", exc_info=True)
                continue
            if nodes:
                return nodes
        return {}

    def _nodes_attr(self) -> dict[str, Any]:
        nodes = getattr(self.target, "nodes", None)
        if isinstance(nodes, Mapping):
            return {
                str(name): node
                for name, node in nodes.items()
                if str(name) not in RESERVED_NODES
            }
        return {}

    def _nodes_via_get_graph(self) -> dict[str, Any]:
        get_graph = getattr(self.target, "get_graph", None)
        if not callable(get_graph):
            return {}
        drawable = get_graph()
        nodes = getattr(drawable, "nodes", None)
        if isinstance(nodes, Mapping):
            return {
                str(name): node
                for name, node in nodes.items()
                if str(name) not in RESERVED_NODES
            }
        if isinstance(nodes, (list, tuple, set)):
            return {
                str(getattr(n, "id", n)): n
                for n in nodes
                if str(getattr(n, "id", n)) not in RESERVED_NODES
            }
        return {}

    @staticmethod
    def _node_callable(node: Any) -> tuple[Any, str] | None:
        """Locate the callable a node actually executes.

        Returns:
            An ``(owner, attribute)`` pair to patch, or ``None`` when the node
            is a bare callable (which the caller replaces in the node map).
        """
        bound = getattr(node, "bound", None)
        if bound is not None:
            for attr in ("func", "afunc"):
                if callable(getattr(bound, attr, None)):
                    return bound, attr

        if callable(getattr(node, "runnable", None)):
            return node, "runnable"

        # A Runnable exposes `.invoke`; a plain function does not.
        if not inspect.isroutine(node) and callable(getattr(node, "invoke", None)):
            return node, "invoke"

        return None

    @staticmethod
    def _node_tools(node: Any) -> list[Any]:
        """Collect any tool objects reachable from *node*."""
        for owner in (node, getattr(node, "bound", None)):
            if owner is None:
                continue
            tools = getattr(owner, "tools", None)
            if isinstance(tools, (list, tuple)):
                return list(tools)
        return []

    # -- Discovery -----------------------------------------------------------

    def discover(self) -> list[ToolDescriptor]:
        """Describe the graph's nodes and any tools bound to them.

        Returns:
            A descriptor per node (tagged ``"node"``) plus one per reachable
            tool (tagged ``"tool"`` and ``"node:<name>"``).  An empty list is
            returned when the graph's structure is not recognised — LangGraph
            internals change between versions and discovery is advisory.
        """
        nodes = self._node_map()
        if not nodes:
            return []

        descriptors: list[ToolDescriptor] = []
        seen_tools: set[str] = set()

        for name, node in nodes.items():
            target = self._node_callable(node)
            doc_source = getattr(target[0], target[1]) if target else node
            descriptors.append(
                ToolDescriptor(
                    name=name,
                    summary=summarize(doc_source, f"LangGraph node {name!r}."),
                    schema={"type": "object", "kind": "node", "node": name},
                    tags=["langgraph", "node"],
                )
            )

            for tool in self._node_tools(node):
                descriptor = self._describe_tool(tool, name)
                if descriptor is None or descriptor.name in seen_tools:
                    continue
                seen_tools.add(descriptor.name)
                descriptors.append(descriptor)

        return descriptors

    @staticmethod
    def _describe_tool(tool: Any, node_name: str) -> ToolDescriptor | None:
        """Build a descriptor for a tool bound to a node, or ``None``."""
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if not isinstance(name, str) or not name:
            return None
        fn = getattr(tool, "func", None) or getattr(tool, "_run", None) or tool
        summary = getattr(tool, "description", None) or summarize(
            fn, f"LangGraph tool {name!r}."
        )
        return ToolDescriptor(
            name=name,
            summary=str(summary).strip().splitlines()[0] if summary else name,
            schema=build_schema(fn),
            tags=["langgraph", "tool", f"node:{node_name}"],
        )

    # -- Binding -------------------------------------------------------------

    def bind(self, interceptor: Interceptor) -> Any:
        """Route every node execution through *interceptor*.

        Each node's callable is replaced with a wrapper that builds a
        :class:`ToolCall` named after the node (``metadata["node"]`` carries
        the same name), sets the AgentLatch node context, and hands control to
        the interceptor.  Nodes already bound by this adapter are skipped, so
        binding is idempotent.

        This mutates the compiled graph in place; call :meth:`restore` to undo
        it.

        Args:
            interceptor: ``(call, invoke) -> Any`` gate supplied by the
                harness.

        Returns:
            The same graph object, still ``.invoke()``-able.
        """
        nodes = self._node_map()
        raw_nodes = getattr(self.target, "nodes", None)

        for name, node in nodes.items():
            slot = self._node_callable(node)
            if slot is not None:
                owner, attr = slot
                original = getattr(owner, attr)
                if getattr(original, _MARKER, False):
                    continue
                wrapped = self._wrap_node(name, original, interceptor)
                try:
                    setattr(owner, attr, wrapped)
                except Exception:  # noqa: BLE001 - frozen/pydantic node objects
                    logger.debug("Could not patch node %r", name, exc_info=True)
                    continue
                self._patches.append((owner, attr, original))
            elif callable(node) and isinstance(raw_nodes, dict):
                if getattr(node, _MARKER, False):
                    continue
                wrapped = self._wrap_node(name, node, interceptor)
                raw_nodes[name] = wrapped
                self._patches.append((raw_nodes, name, node))

        return self.target

    def _wrap_node(
        self,
        name: str,
        fn: Callable[..., Any],
        interceptor: Interceptor,
    ) -> Callable[..., Any]:
        """Wrap one node callable with node context plus interception."""
        routed = route_callable(
            self,
            name,
            fn,
            interceptor,
            metadata={"node": name, "kind": "node"},
        )

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_node(*args: Any, **kwargs: Any) -> Any:
                set_node_context(name)
                try:
                    return await routed(*args, **kwargs)
                finally:
                    set_node_context("")

            setattr(async_node, _MARKER, True)
            return async_node

        @functools.wraps(fn)
        def sync_node(*args: Any, **kwargs: Any) -> Any:
            set_node_context(name)
            try:
                return routed(*args, **kwargs)
            finally:
                set_node_context("")

        setattr(sync_node, _MARKER, True)
        return sync_node

    # -- Undo ----------------------------------------------------------------

    def restore(self) -> Any:
        """Put every patched node callable back, newest patch first.

        Returns:
            The graph, unbound.
        """
        for owner, key, original in reversed(self._patches):
            try:
                if isinstance(owner, dict):
                    owner[key] = original
                else:
                    setattr(owner, key, original)
            except Exception:  # noqa: BLE001 - best effort teardown
                logger.debug("Could not restore node slot %r", key, exc_info=True)
        self._patches.clear()
        return self.target
