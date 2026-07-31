"""The zero-framework adapter — plain Python callables.

:class:`CallableAdapter` is the reference implementation of
:class:`~agentlatch.harness.adapters.base.FrameworkAdapter`.  It governs a bare
``dict`` (or list) of functions, which is what an agent looks like before a
framework is involved, and what every other adapter degrades to internally.

This module also hosts the two helpers the LangGraph and CrewAI adapters reuse
so that schema shape and interceptor semantics stay identical everywhere:

* :func:`build_schema` — render an ``inspect.Signature`` as a JSON-ish schema.
* :func:`route_callable` — wrap a callable so its invocation flows through an
  interceptor, preserving sync/async behaviour and ``functools.wraps``.

Nothing here imports a third-party package.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from agentlatch.harness._types import ToolCall, ToolDescriptor
from agentlatch.harness.adapters.base import FrameworkAdapter

Interceptor = Callable[[ToolCall, Callable[..., Any]], Any]

_SELF_PARAMS = frozenset({"self", "cls"})


# ---------------------------------------------------------------------------
# Shared helpers (reused by the LangGraph and CrewAI adapters)
# ---------------------------------------------------------------------------


def render_annotation(annotation: Any) -> str:
    """Render a type annotation as a stable, readable string.

    Args:
        annotation: Whatever ``inspect`` reported — a class, a typing
            construct, a string (PEP 563 / ``from __future__ import
            annotations``), or the ``empty`` sentinel.

    Returns:
        ``"Any"`` when the parameter is unannotated, otherwise a short name
        such as ``"int"`` or ``"list[str]"``.
    """
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return str(annotation).replace("typing.", "")


def build_schema(fn: Any) -> dict[str, Any]:
    """Derive a parameter schema from a callable's signature.

    Unannotated parameters, ``*args``/``**kwargs``, builtins with no
    introspectable signature, and bound-method ``self`` are all handled
    without raising — discovery must never be the thing that breaks a run.

    Args:
        fn: The callable to introspect.

    Returns:
        A dict with ``type``, ``properties`` and ``required`` keys, plus
        ``returns`` when the callable annotates its return type.  Variadic
        parameters carry a ``variadic`` marker and are never required.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}, "required": []}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for index, (pname, param) in enumerate(sig.parameters.items()):
        if index == 0 and pname in _SELF_PARAMS:
            continue
        entry: dict[str, Any] = {"type": render_annotation(param.annotation)}
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            entry["variadic"] = "positional"
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            entry["variadic"] = "keyword"
        elif param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        else:
            required.append(pname)
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            entry["keyword_only"] = True
        properties[pname] = entry

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    if sig.return_annotation is not inspect.Signature.empty:
        schema["returns"] = render_annotation(sig.return_annotation)
    return schema


def summarize(fn: Any, fallback: str) -> str:
    """Return the first docstring line of *fn*, or *fallback* if it has none.

    Args:
        fn: The callable (or tool object) whose docstring to read.
        fallback: Summary to use when no usable docstring exists.
    """
    doc = inspect.getdoc(fn) or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return fallback


def make_invoke(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Callable[..., Any]:
    """Build the ``invoke`` callable handed to an interceptor.

    The result is both *zero-argument safe* (calling ``invoke()`` replays the
    original arguments, which is what the base contract promises) and
    *transparent* (calling ``invoke(*args, **kwargs)`` forwards whatever the
    interceptor decided to substitute).

    Args:
        fn: The underlying tool callable.
        args: Positional arguments the agent supplied.
        kwargs: Keyword arguments the agent supplied.
    """

    @functools.wraps(fn)
    def invoke(*override_args: Any, **override_kwargs: Any) -> Any:
        if not override_args and not override_kwargs:
            return fn(*args, **kwargs)
        return fn(*override_args, **override_kwargs)

    return invoke


def route_callable(
    adapter: FrameworkAdapter,
    name: str,
    fn: Callable[..., Any],
    interceptor: Interceptor,
    metadata: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    """Wrap *fn* so every invocation is routed through *interceptor*.

    A coroutine function yields a coroutine function, so awaiting behaviour is
    unchanged for the framework calling it.  ``functools.wraps`` is applied so
    ``__name__``, ``__doc__`` and ``__wrapped__`` survive — frameworks read
    those to build tool schemas.

    Args:
        adapter: Adapter whose ``framework`` stamps the :class:`ToolCall`.
        name: Tool name as the agent framework knows it.
        fn: The original callable.
        interceptor: ``(call, invoke) -> Any`` gate supplied by the harness.
        metadata: Extra fields merged into ``ToolCall.metadata``.

    Returns:
        A wrapper with the same calling shape as *fn*.
    """
    extras: dict[str, Any] = dict(metadata or {})

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call = adapter.make_call(name, args, kwargs, **extras)
            result = interceptor(call, make_invoke(fn, args, kwargs))
            if inspect.isawaitable(result):
                return await result
            return result

        return async_wrapper

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        call = adapter.make_call(name, args, kwargs, **extras)
        return interceptor(call, make_invoke(fn, args, kwargs))

    return sync_wrapper


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CallableAdapter(FrameworkAdapter):
    """Governs a plain collection of Python callables.

    Use this when there is no agent framework in the loop — a hand-rolled
    ReAct loop, a tool registry passed to an LLM client, or a test harness::

        adapter = CallableAdapter({"search": search, "write": write_file})
        tools = adapter.bind(harness.interceptor)
        tools["search"]("langgraph")   # routed through the permission gate

    Binding never mutates the caller's mapping: :meth:`bind` returns a brand
    new dict of wrappers.

    Args:
        target: Either a mapping of tool name to callable, or an iterable of
            callables whose names are taken from ``__name__``.

    Raises:
        TypeError: If *target* is neither a mapping nor an iterable of
            callables, or a listed entry is not callable.
    """

    framework = "generic"

    def __init__(
        self, target: Mapping[str, Callable[..., Any]] | Iterable[Any]
    ) -> None:
        super().__init__(target)
        self.tools: dict[str, Callable[..., Any]] = self._normalize(target)

    # -- Normalisation -------------------------------------------------------

    @staticmethod
    def _normalize(target: Any) -> dict[str, Callable[..., Any]]:
        """Coerce the accepted input shapes into a ``{name: callable}`` dict."""
        if isinstance(target, Mapping):
            items = list(target.items())
        elif isinstance(target, Iterable) and not isinstance(target, (str, bytes)):
            items = []
            for fn in target:
                if not callable(fn):
                    raise TypeError(
                        f"CallableAdapter expects callables, got {type(fn).__name__!r}."
                    )
                items.append((_callable_name(fn), fn))
        elif callable(target):
            items = [(_callable_name(target), target)]
        else:
            raise TypeError(
                "CallableAdapter expects a mapping of name -> callable, an "
                f"iterable of callables, or a single callable; got "
                f"{type(target).__name__!r}."
            )

        tools: dict[str, Callable[..., Any]] = {}
        for name, fn in items:
            if not callable(fn):
                raise TypeError(f"Tool {name!r} is not callable.")
            tools[str(name)] = fn
        return tools

    # -- Discovery -----------------------------------------------------------

    def discover(self) -> list[ToolDescriptor]:
        """Describe every registered callable.

        Returns:
            One :class:`ToolDescriptor` per tool, with the summary taken from
            the first docstring line and the schema derived from the
            signature.
        """
        descriptors: list[ToolDescriptor] = []
        for name, fn in self.tools.items():
            descriptors.append(
                ToolDescriptor(
                    name=name,
                    summary=summarize(fn, f"Callable tool {name!r}."),
                    schema=build_schema(fn),
                    tags=["generic"],
                )
            )
        return descriptors

    # -- Binding -------------------------------------------------------------

    def bind(self, interceptor: Interceptor) -> dict[str, Callable[..., Any]]:
        """Return a new dict of wrapped callables.

        The adapter's own ``tools`` mapping and the caller's original mapping
        are both left untouched.

        Args:
            interceptor: ``(call, invoke) -> Any`` gate supplied by the
                harness.

        Returns:
            A fresh ``{name: wrapped_callable}`` dict.
        """
        return {
            name: route_callable(self, name, fn, interceptor)
            for name, fn in self.tools.items()
        }


def _callable_name(fn: Any) -> str:
    """Best-effort name for a callable that may be a partial or an object."""
    for attr in ("__name__", "name"):
        value = getattr(fn, attr, None)
        if isinstance(value, str) and value:
            return value
    inner = getattr(fn, "func", None)
    if inner is not None and inner is not fn:
        return _callable_name(inner)
    return type(fn).__name__
