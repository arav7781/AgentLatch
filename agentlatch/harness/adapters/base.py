"""Framework adapter interface.

An adapter teaches the harness how to find and re-bind the tools of one agent
framework.  The harness itself knows nothing about LangGraph or CrewAI — it
only knows this interface, which is what makes it framework-agnostic.

Contract for implementers:

* :meth:`discover` reports the tools the harness may govern.
* :meth:`bind` returns an object of the *same shape* as the wrapped one — a
  compiled graph stays invocable as a compiled graph — with every tool call
  routed through the supplied interceptor.
* Binding must not mutate the caller's object where that can be avoided.
  Prefer wrapping over in-place patching; when patching is the only option
  (frameworks that hold tools in private lists), say so in the docstring.
* Importing this module must never import the target framework.  Do framework
  imports lazily inside methods so ``agentlatch[harness]`` stays installable
  without LangGraph or CrewAI present.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

from agentlatch.harness._types import ToolCall, ToolDescriptor


class FrameworkAdapter(abc.ABC):
    """Adapts one agent framework to the harness."""

    framework: str = "generic"

    def __init__(self, target: Any) -> None:
        self.target = target

    # -- Discovery -----------------------------------------------------------

    @abc.abstractmethod
    def discover(self) -> list[ToolDescriptor]:
        """Return descriptors for every tool the target exposes.

        Used for progressive disclosure and for pre-assigning permission
        tiers.  Return an empty list when the framework hides its tools.
        """

    # -- Binding -------------------------------------------------------------

    @abc.abstractmethod
    def bind(self, interceptor: Callable[[ToolCall, Callable[..., Any]], Any]) -> Any:
        """Return the target with all tool calls routed through *interceptor*.

        The interceptor receives the :class:`ToolCall` and an ``invoke``
        callable, and returns whatever should be handed back to the framework
        in place of the tool's return value.

        ``invoke`` supports **both** calling conventions, intentionally:

        * ``invoke()`` — replay the call exactly as the agent made it.  This is
          the common case, and it means an interceptor never has to unpack
          ``call.args`` / ``call.kwargs`` just to pass them straight through.
        * ``invoke(*args, **kwargs)`` — override the arguments.  This is what
          lets a harness sanitize or rewrite a call before letting it run,
          rather than being limited to allow/deny.

        Adapters must implement both; see ``adapters.generic.make_invoke`` for
        the reference implementation.
        """

    # -- Unbinding -----------------------------------------------------------

    def restore(self) -> Any:
        """Undo any in-place patching this adapter performed.

        Adapters that wrap rather than mutate need not override this — the
        default is a no-op returning the target.  Adapters that patch (CrewAI
        holds tools in agent-owned lists, LangGraph in a node map) must
        override it so a harness can be detached cleanly, which matters in
        tests and in long-lived processes that re-wrap the same object.
        """
        return self.target

    # -- Helpers for subclasses ---------------------------------------------

    def make_call(
        self,
        name: str,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> ToolCall:
        """Build a :class:`ToolCall` stamped with this adapter's framework.

        Args:
            metadata: Metadata mapping.  Prefer this over *extra* — passing
                keys as keyword arguments cannot express a metadata key named
                ``name``, ``args``, ``kwargs``, or ``metadata``.
            **extra: Convenience shorthand, merged into *metadata*.
        """
        merged = {**(metadata or {}), **extra}
        return ToolCall(
            name=name,
            args=args,
            kwargs=kwargs or {},
            framework=self.framework,
            metadata=merged,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} framework={self.framework!r}>"
