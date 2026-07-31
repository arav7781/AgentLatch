"""CrewAI adapter for the AgentLatch Harness.

A CrewAI ``Crew`` owns ``Agent`` objects, and each agent owns a list of tool
objects.  There is no registry to swap and no dispatcher to intercept: the
agent executor reaches into its own ``tools`` list and calls the tool's
``.func`` / ``._run`` directly.

**This adapter therefore patches in place.**  :meth:`CrewAIAdapter.bind`
replaces the callable attribute on each tool object and returns the same crew,
still ``.kickoff()``-able.  :meth:`CrewAIAdapter.restore` puts every original
callable back exactly, so the mutation is reversible.

CrewAI is never imported at module import time; it is not a hard dependency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agentlatch.harness._types import ToolDescriptor
from agentlatch.harness.adapters.base import FrameworkAdapter
from agentlatch.harness.adapters.generic import (
    Interceptor,
    build_schema,
    route_callable,
    summarize,
)

logger = logging.getLogger("agentlatch.harness.adapters.crewai")

CALLABLE_ATTRS: tuple[str, ...] = ("func", "_run", "run")
"""Attributes a CrewAI tool may hold its implementation under, most specific
first.  ``func`` is what ``@tool`` produces; ``_run`` is the ``BaseTool``
subclass hook; ``run`` is the legacy surface."""

_MARKER = "_agentlatch_harness_bound"


def _require_crewai() -> Any:
    """Import CrewAI, or raise an error naming the extra to install.

    Returns:
        The imported ``crewai`` module.

    Raises:
        ImportError: If CrewAI is not installed.
    """
    try:
        import crewai  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "CrewAIAdapter requires CrewAI. Install it with "
            "`pip install 'agentlatch[crewai]'` (or `pip install crewai`)."
        ) from exc
    return crewai


class CrewAIAdapter(FrameworkAdapter):
    """Routes every CrewAI tool call through a harness interceptor.

    Args:
        target: A CrewAI ``Crew`` (anything exposing an ``agents`` list).  A
            bare ``Agent`` is accepted too and treated as a crew of one.

    Raises:
        TypeError: If *target* exposes neither ``agents`` nor ``tools``.
    """

    framework = "crewai"

    def __init__(self, target: Any) -> None:
        super().__init__(target)
        if not hasattr(target, "agents") and not hasattr(target, "tools"):
            _require_crewai()
            raise TypeError(
                "CrewAIAdapter expects a Crew (with `.agents`) or an Agent "
                f"(with `.tools`); got {type(target).__name__!r}."
            )
        self._patches: list[tuple[Any, str, Any, bool]] = []

    # -- Structure probing ---------------------------------------------------

    def _agents(self) -> list[Any]:
        """Return the crew's agents, or the target itself if it is an agent."""
        agents = getattr(self.target, "agents", None)
        if isinstance(agents, (list, tuple)):
            return list(agents)
        if hasattr(self.target, "tools"):
            return [self.target]
        return []

    @staticmethod
    def _agent_role(agent: Any, index: int) -> str:
        """Human-readable role for *agent*, falling back to its position."""
        for attr in ("role", "name", "id"):
            value = getattr(agent, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"agent[{index}]"

    @staticmethod
    def _agent_tools(agent: Any) -> list[Any]:
        """Return the agent's tool objects, tolerating a missing list."""
        tools = getattr(agent, "tools", None)
        if isinstance(tools, (list, tuple)):
            return list(tools)
        return []

    @staticmethod
    def _tool_name(tool: Any, index: int) -> str:
        """Name a tool the way CrewAI's prompt would refer to it."""
        for attr in ("name", "__name__"):
            value = getattr(tool, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for attr in CALLABLE_ATTRS:
            fn = getattr(tool, attr, None)
            inner = getattr(fn, "__name__", None)
            if isinstance(inner, str) and inner:
                return inner
        return f"{type(tool).__name__}[{index}]"

    @staticmethod
    def _tool_slot(tool: Any) -> tuple[str, Callable[..., Any]] | None:
        """Find which attribute holds the tool's implementation.

        Returns:
            An ``(attribute, callable)`` pair, or ``None`` when the tool
            exposes no callable this adapter recognises.
        """
        for attr in CALLABLE_ATTRS:
            fn = getattr(tool, attr, None)
            if callable(fn):
                return attr, fn
        if callable(tool):
            return "__call__", tool
        return None

    @staticmethod
    def _tool_schema(tool: Any, fn: Any) -> dict[str, Any]:
        """Prefer the tool's declared args schema, else read the signature."""
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            for method in ("model_json_schema", "schema"):
                builder = getattr(args_schema, method, None)
                if callable(builder):
                    try:
                        result = builder()
                    except Exception:  # noqa: BLE001 - pydantic version drift
                        logger.debug("args_schema render failed", exc_info=True)
                        continue
                    if isinstance(result, dict):
                        return result
        return build_schema(fn)

    # -- Discovery -----------------------------------------------------------

    def discover(self) -> list[ToolDescriptor]:
        """Describe every tool held by every agent in the crew.

        The owning agent's role is stamped into
        :attr:`ToolDescriptor.tags` so a multi-agent crew stays traceable when
        two agents share a tool name.

        Returns:
            One descriptor per agent-owned tool; empty when the crew exposes
            no recognisable tools.
        """
        descriptors: list[ToolDescriptor] = []

        for agent_index, agent in enumerate(self._agents()):
            role = self._agent_role(agent, agent_index)
            for tool_index, tool in enumerate(self._agent_tools(agent)):
                name = self._tool_name(tool, tool_index)
                slot = self._tool_slot(tool)
                fn = slot[1] if slot else tool
                description = getattr(tool, "description", None)
                summary = (
                    str(description).strip().splitlines()[0]
                    if isinstance(description, str) and description.strip()
                    else summarize(fn, f"CrewAI tool {name!r}.")
                )
                descriptors.append(
                    ToolDescriptor(
                        name=name,
                        summary=summary,
                        schema=self._tool_schema(tool, fn),
                        tags=["crewai", role],
                    )
                )

        return descriptors

    # -- Binding -------------------------------------------------------------

    def bind(self, interceptor: Interceptor) -> Any:
        """Route every agent-owned tool call through *interceptor*.

        Patches each tool's implementation attribute in place — CrewAI holds
        tools in agent-owned lists that the executor reads directly, so there
        is nothing to wrap from the outside.  Already-bound tools are skipped,
        making this idempotent; :meth:`restore` reverses it.

        Args:
            interceptor: ``(call, invoke) -> Any`` gate supplied by the
                harness.

        Returns:
            The same crew object, still ``.kickoff()``-able.
        """
        for agent_index, agent in enumerate(self._agents()):
            role = self._agent_role(agent, agent_index)
            for tool_index, tool in enumerate(self._agent_tools(agent)):
                slot = self._tool_slot(tool)
                if slot is None:
                    continue
                attr, original = slot
                if attr == "__call__" or getattr(original, _MARKER, False):
                    continue

                name = self._tool_name(tool, tool_index)
                wrapped = route_callable(
                    self,
                    name,
                    original,
                    interceptor,
                    metadata={"agent_role": role, "tool": name},
                )
                setattr(wrapped, _MARKER, True)

                # ``_run`` is usually a class-level method: remember whether
                # the instance owned the attribute so restore leaves no trace.
                owned = attr in getattr(tool, "__dict__", {})
                if not self._set_attr(tool, attr, wrapped):
                    continue
                self._patches.append((tool, attr, original, owned))

        return self.target

    @staticmethod
    def _set_attr(tool: Any, attr: str, value: Any) -> bool:
        """Assign *value* onto *tool*, working around frozen pydantic models."""
        try:
            setattr(tool, attr, value)
            return True
        except Exception:  # noqa: BLE001 - pydantic may reject assignment
            try:
                object.__setattr__(tool, attr, value)
                return True
            except Exception:  # noqa: BLE001 - genuinely immutable tool
                logger.debug("Could not patch tool attribute %r", attr, exc_info=True)
                return False

    # -- Undo ----------------------------------------------------------------

    def restore(self) -> Any:
        """Put every patched tool callable back, newest patch first.

        Returns:
            The crew, unbound.
        """
        for tool, attr, original, owned in reversed(self._patches):
            if not owned:
                try:
                    delattr(tool, attr)
                    continue
                except Exception:  # noqa: BLE001 - fall back to reassignment
                    logger.debug(
                        "Could not delete patched attr %r", attr, exc_info=True
                    )
            self._set_attr(tool, attr, original)
        self._patches.clear()
        return self.target
