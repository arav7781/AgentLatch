"""Shared types for the AgentLatch Harness.

The harness sits between an agent framework (LangGraph, CrewAI, AutoGen, or a
plain function) and the operating system.  Every tool call the agent attempts
flows through the same pipeline::

    ToolCall  ->  PermissionGate  ->  Sandbox  ->  Compactor  ->  ToolResult

This module defines the vocabulary every stage shares.  Nothing here imports a
third-party package: the harness core is standard library only.
"""

from __future__ import annotations

import enum
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PermissionTier(enum.Enum):
    """How a tool call is allowed to proceed.

    Tiers are ordered by increasing restriction.  When several rules match a
    call, the most restrictive tier wins — a tool that is both "read-only" and
    "matches a blocked pattern" is blocked.
    """

    AUTO = "auto"
    """Tier 1 — safe and read-only.  Executes immediately."""

    HUMAN = "human"
    """Tier 2 — mutates state.  Pauses for approval before executing."""

    BLOCKED = "blocked"
    """Tier 3 — dangerous.  Never executes, under any approval."""

    @property
    def restrictiveness(self) -> int:
        """Sort key used to resolve conflicting rule matches (higher wins)."""
        return {"auto": 0, "human": 1, "blocked": 2}[self.value]


class ExecutionStatus(enum.Enum):
    """Terminal state of a sandboxed execution or an intercepted tool call."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    DENIED = "denied"
    """Tier 2 call that a human explicitly rejected."""


class Language(enum.Enum):
    """What a sandbox is being asked to execute."""

    PYTHON = "python"
    SHELL = "shell"


# ---------------------------------------------------------------------------
# Tool calls and results
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return str(uuid.uuid4())


_FLATTEN_DEPTH = 5
"""Default nesting depth searched by :meth:`ToolCall.command_text`."""


@dataclass
class ToolCall:
    """A single tool invocation intercepted before the ReAct "Act" phase.

    Attributes:
        name:      Tool name as the agent framework knows it.
        args:      Positional arguments the agent supplied.
        kwargs:    Keyword arguments the agent supplied.
        id:        Unique identifier, used to correlate approvals and results.
        framework: Which adapter produced this call ("langgraph", "crewai", ...).
        metadata:  Adapter-specific extras (node name, agent role, raw payload).
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    framework: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """Render a short, human-readable form for approval prompts and logs.

        Long argument values are elided — an approval prompt must stay
        readable, and the full payload is available on the call itself.
        """
        parts = [repr(_elide(a)) for a in self.args]
        parts += [f"{k}={_elide(v)!r}" for k, v in sorted(self.kwargs.items())]
        return f"{self.name}({', '.join(parts)})"

    def command_text(self, depth: int = _FLATTEN_DEPTH) -> str:
        """Flatten every string argument into one blob for rule matching.

        Block rules match against this, so a dangerous command is caught
        whether the agent passed it positionally, by keyword, or buried in a
        nested structure.

        Args:
            depth: How many container levels to descend.  The default covers
                realistic tool payloads (a dict of lists of dicts).  It is
                bounded rather than unlimited so a deeply recursive or cyclic
                argument cannot stall the gate — the fail-closed default tier
                is the backstop for anything deeper.
        """
        chunks: list[str] = [self.name]
        for value in (*self.args, *self.kwargs.values()):
            chunks.extend(_flatten_strings(value, depth))
        return "\n".join(chunks)


def _elide(value: Any, limit: int = 80) -> Any:
    """Shorten oversized string values for display."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...[+{len(value) - limit} chars]"
    return value


def _flatten_strings(value: Any, depth: int = _FLATTEN_DEPTH) -> list[str]:
    """Collect string leaves from a nested structure, bounded by *depth*."""
    if isinstance(value, str):
        return [value]
    if depth <= 0:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            if isinstance(k, str):
                out.append(k)
            out.extend(_flatten_strings(v, depth - 1))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_strings(item, depth - 1))
        return out
    return []


@dataclass
class ToolResult:
    """Outcome of an intercepted tool call, after gating and compaction.

    A harness never raises on tool failure.  Like ``@safe_tool``, it returns a
    structured result the LLM can read and self-correct from.
    """

    status: ExecutionStatus
    content: Any = None
    error: dict[str, Any] | None = None
    duration_ms: float = 0.0
    tier: PermissionTier | None = None
    compacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ExecutionStatus.OK

    def to_payload(self) -> Any:
        """Return the value handed back to the agent.

        Successful calls yield their content untouched so existing tool
        contracts keep working.  Failures yield the structured error dict,
        matching the shape ``agentlatch.decorators`` already returns.
        """
        if self.ok:
            return self.content
        return self.error or {
            "status": "error",
            "error_type": self.status.value,
            "message": "Tool call did not complete.",
            "instruction": (
                "The tool did not run. Review your parameters and retry, or "
                "choose a different tool."
            ),
        }


# ---------------------------------------------------------------------------
# Permission decisions
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    """The permission gate's verdict on a single tool call."""

    tier: PermissionTier
    reason: str = ""
    rule_name: str | None = None
    approved: bool | None = None
    """Tier 2 only: ``True``/``False`` once a human answered, ``None`` before."""

    @property
    def may_execute(self) -> bool:
        """Whether the call is cleared to run right now."""
        if self.tier is PermissionTier.BLOCKED:
            return False
        if self.tier is PermissionTier.HUMAN:
            return self.approved is True
        return True


ApprovalCallback = Callable[[ToolCall, Decision], bool]
"""Tier 2 hook. Receives the call and the pending decision, returns approval.

Implementations may block (CLI prompt), call out to a webhook, or consult a
queue.  Returning ``False`` — or raising — denies the call.
"""


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------


@dataclass
class ExecutionRequest:
    """A request to run agent-authored code away from the host process."""

    code: str
    language: Language = Language.PYTHON
    timeout: float = 30.0
    env: dict[str, str] = field(default_factory=dict)
    """Explicit environment for the sandbox. The host environment is never
    inherited — an agent must not be able to read the host's API keys."""
    workdir: str = "/sandbox"
    memory_limit_mb: int | None = 256
    network: bool = False
    """Networking is off by default. Enabling it removes a containment layer."""


@dataclass
class SandboxResult:
    """Captured output of a sandboxed execution.

    Attributes:
        terminated: Whether the runtime actually stopped the work.  Only
            meaningful on a ``TIMEOUT``, where the distinction matters
            operationally: :class:`DockerSandbox` force-kills the container
            (``True``), while :class:`ThreadSandbox` cannot kill a Python
            thread and abandons it still running (``False``).  An abandoned
            thread keeps consuming CPU and may still write to stdout after
            the deadline, so a caller that retries on timeout needs to know
            which happened.  ``None`` when the question does not apply.
    """

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0.0
    runtime: str = ""
    terminated: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def leaked_work(self) -> bool:
        """``True`` when the run timed out and was *not* actually stopped."""
        return self.status is ExecutionStatus.TIMEOUT and self.terminated is False

    @property
    def ok(self) -> bool:
        return self.status is ExecutionStatus.OK

    def to_payload(self) -> dict[str, Any]:
        """Structured form handed back to the LLM."""
        payload: dict[str, Any] = {
            "status": "success" if self.ok else "error",
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 2),
            "runtime": self.runtime,
        }
        if not self.ok:
            payload["error_type"] = self.status.value
            payload["instruction"] = (
                "The sandboxed execution failed. Read stderr, correct the "
                "code, and retry."
            )
        return payload


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class SandboxRuntime(Protocol):
    """Pluggable execution backend.

    Implementations must guarantee that :meth:`run` returns rather than raises
    for ordinary failure — a crashed or timed-out execution is a
    ``SandboxResult``, not an exception.  Only genuine misconfiguration
    (missing Docker daemon, unusable image) may raise.
    """

    name: str

    def run(self, request: ExecutionRequest) -> SandboxResult:
        """Execute *request* in isolation and capture its output."""
        ...

    def close(self) -> None:
        """Release any held resources (clients, pools, stray containers)."""
        ...


@runtime_checkable
class Interceptor(Protocol):
    """Callable the adapters route every tool invocation through.

    The harness supplies this; adapters do not implement it.

    The return type is deliberately widened to include awaitables.  An
    interceptor wrapping an async tool returns a coroutine, and the adapter
    awaits it — so a single sync-looking protocol covers both paths without
    forcing adapters to branch on which kind of interceptor they were given.
    """

    def __call__(
        self, call: ToolCall, invoke: Callable[..., Any]
    ) -> Any | Awaitable[Any]:
        """Gate *call*, then run *invoke* if permitted."""
        ...


# ---------------------------------------------------------------------------
# Tool descriptors (progressive disclosure)
# ---------------------------------------------------------------------------


@dataclass
class ToolDescriptor:
    """Metadata the harness holds about a tool it can expose to the agent.

    Progressive disclosure means the initial system prompt carries only
    :attr:`summary` for each tool.  The full :attr:`schema` is fetched on
    demand when the agent asks to use one.
    """

    name: str
    summary: str
    schema: dict[str, Any] | None = None
    tier: PermissionTier | None = None
    framework: str | None = None
    """Which adapter surfaced this tool — a harness may govern several."""
    tags: list[str] = field(default_factory=list)

    def brief(self) -> dict[str, str]:
        """The lightweight form loaded up front."""
        return {"name": self.name, "summary": self.summary}


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


class Stopwatch:
    """Monotonic elapsed-time helper shared by the harness stages."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0
