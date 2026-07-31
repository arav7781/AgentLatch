"""The Harness — one gate every agent tool call passes through.

The harness owns the execution layer that orchestration frameworks leave to
you.  It wraps any framework via an adapter and enforces the same pipeline
regardless of what produced the call::

    ToolCall -> PermissionGate -> [Sandbox] -> Compactor -> ToolResult

Design notes:

* **Nothing raises outward.**  A blocked call, a denied approval, a crashed
  tool, and a timed-out container all come back as structured data the LLM can
  read and correct from.  This mirrors ``@safe_tool``: an agent that gets an
  exception is an agent that stops; an agent that gets an error payload is an
  agent that retries.
* **It composes with the existing tracer.**  When a run happens inside
  ``@profile_agent``, every intercepted call becomes a ``TraceEvent`` child, so
  harness activity shows up in the flamegraph for free.
* **Fail closed.**  The default policy sends unrecognised tools to a human, and
  a missing approval callback denies rather than allows.

Usage::

    from agentlatch.harness import Harness
    from agentlatch.harness.adapters import LangGraphAdapter

    harness = Harness(on_approval=cli_approval_callback)
    secured_agent = harness.wrap(LangGraphAdapter(my_graph))
    secured_agent.invoke({"messages": [...]})
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable
from typing import Any

from agentlatch._types import EventStatus
from agentlatch.harness._types import (
    ApprovalCallback,
    Decision,
    ExecutionRequest,
    ExecutionStatus,
    Language,
    PermissionTier,
    SandboxResult,
    Stopwatch,
    ToolCall,
    ToolDescriptor,
    ToolResult,
)
from agentlatch.harness.adapters.base import FrameworkAdapter
from agentlatch.harness.compaction import Compactor, ToolRegistry
from agentlatch.harness.permissions import PermissionGate, PermissionPolicy
from agentlatch.harness.sandbox.base import BaseSandbox
from agentlatch.tracker import end_child, get_trace, start_child

# Tool names that mean "run this code", routed to the sandbox instead of
# being called in-process.  Extend via Harness(code_tools=[...]).
_DEFAULT_CODE_TOOLS: tuple[str, ...] = (
    "python",
    "python_repl",
    "execute_python",
    "run_python",
    "code_interpreter",
    "shell",
    "bash",
    "terminal",
    "execute_shell",
    "run_command",
)


class Harness:
    """Security and context boundary around any agent framework.

    Args:
        policy:       Permission rules.  Defaults to
                      :meth:`PermissionPolicy.default`, which fails closed.
        sandbox:      Runtime for agent-authored code.  When ``None``, code
                      tools are refused rather than run on the host — an
                      unconfigured harness must not become an escape hatch.
        compactor:    Output compaction.  ``None`` disables it.
        registry:     Tool catalog for progressive disclosure.
        on_approval:  Tier 2 human-in-the-loop hook.
        code_tools:   Extra tool names to route to the sandbox.
        audit_log:    Mutable list that receives one dict per decision.
    """

    def __init__(
        self,
        *,
        policy: PermissionPolicy | None = None,
        sandbox: BaseSandbox | None = None,
        compactor: Compactor | None = None,
        registry: ToolRegistry | None = None,
        on_approval: ApprovalCallback | None = None,
        approval_timeout: float | None = None,
        code_tools: Iterable[str] | None = None,
        audit_log: list[dict[str, Any]] | None = None,
    ) -> None:
        self.policy = policy or PermissionPolicy.default()
        self.audit_log: list[dict[str, Any]] = (
            audit_log if audit_log is not None else []
        )
        self.gate = PermissionGate(
            policy=self.policy,
            on_approval=on_approval,
            approval_timeout=approval_timeout,
            audit_log=self.audit_log,
        )
        self.sandbox = sandbox
        self.compactor = compactor if compactor is not None else Compactor()
        self.registry = registry or ToolRegistry()
        self.code_tools = set(_DEFAULT_CODE_TOOLS) | set(code_tools or ())
        self._adapters: list[FrameworkAdapter] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wrap(self, adapter: FrameworkAdapter) -> Any:
        """Secure an agent framework and return it, same shape as before.

        The returned object is a drop-in replacement for the one the adapter
        wrapped — a compiled graph stays ``.invoke()``-able.
        """
        self._adapters.append(adapter)
        for descriptor in adapter.discover():
            self.registry.register(self._tier_descriptor(descriptor))
        return adapter.bind(self.intercept)

    def restore(self) -> None:
        """Undo every in-place patch made by adapters that needed one."""
        for adapter in self._adapters:
            restore = getattr(adapter, "restore", None)
            if callable(restore):
                restore()
        self._adapters.clear()

    def intercept(self, call: ToolCall, invoke: Callable[..., Any]) -> Any:
        """Run one tool call through the full pipeline.

        This is the interceptor handed to adapters.  It returns the payload the
        framework receives in place of the tool's return value — never raising,
        so a refused call reads as feedback rather than a crash.
        """
        if inspect.iscoroutinefunction(invoke):
            return self._intercept_async(call, invoke)
        return self._intercept_sync(call, invoke)

    def execute_code(
        self,
        code: str,
        language: Language | str = Language.PYTHON,
        *,
        timeout: float = 30.0,
        network: bool = False,
    ) -> SandboxResult:
        """Run agent-authored code in the configured sandbox.

        Returns an ERROR result rather than raising when no sandbox is
        configured.  Running unsandboxed on the host is never the fallback.
        """
        if isinstance(language, str):
            try:
                language = Language(language.lower())
            except ValueError:
                return SandboxResult(
                    status=ExecutionStatus.ERROR,
                    stderr=f"Unsupported language {language!r}.",
                    runtime="none",
                )
        if self.sandbox is None:
            return SandboxResult(
                status=ExecutionStatus.BLOCKED,
                stderr=(
                    "No sandbox configured. Code execution is refused rather "
                    "than run on the host. Construct the harness with "
                    "Harness(sandbox=DockerSandbox()) to enable it."
                ),
                runtime="none",
            )
        request = ExecutionRequest(
            code=code, language=language, timeout=timeout, network=network
        )
        return self.sandbox.run(request)

    def system_prompt_block(self) -> str:
        """Brief tool catalog for the initial system prompt."""
        return self.registry.system_prompt_block()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _intercept_sync(self, call: ToolCall, invoke: Callable[..., Any]) -> Any:
        watch = Stopwatch()
        event = start_child(call.name) if get_trace() is not None else None

        decision = self.gate.check(call)
        if not decision.may_execute:
            result = self._refusal(call, decision, watch)
            self._close_event(event, result)
            return result.to_payload()

        try:
            raw = self._run_tool(call, invoke)
        except Exception as exc:  # noqa: BLE001 - boundary: feed back, don't crash
            result = self._failure(exc, decision, watch)
            self._close_event(event, result)
            return result.to_payload()

        result = self._succeed(raw, decision, watch)
        self._close_event(event, result)
        return result.to_payload()

    async def _intercept_async(self, call: ToolCall, invoke: Callable[..., Any]) -> Any:
        watch = Stopwatch()
        event = start_child(call.name) if get_trace() is not None else None

        decision = self.gate.check(call)
        if not decision.may_execute:
            result = self._refusal(call, decision, watch)
            self._close_event(event, result)
            return result.to_payload()

        try:
            if self._is_code_tool(call):
                raw = self._run_code_tool(call)
            else:
                raw = await invoke(*call.args, **call.kwargs)
        except Exception as exc:  # noqa: BLE001 - boundary: feed back, don't crash
            result = self._failure(exc, decision, watch)
            self._close_event(event, result)
            return result.to_payload()

        result = self._succeed(raw, decision, watch)
        self._close_event(event, result)
        return result.to_payload()

    def _run_tool(self, call: ToolCall, invoke: Callable[..., Any]) -> Any:
        """Dispatch to the sandbox for code tools, otherwise call directly."""
        if self._is_code_tool(call):
            return self._run_code_tool(call)
        return invoke(*call.args, **call.kwargs)

    def _is_code_tool(self, call: ToolCall) -> bool:
        return call.name.lower() in self.code_tools

    def _run_code_tool(self, call: ToolCall) -> Any:
        """Extract the code payload from a code tool call and sandbox it."""
        code = self._extract_code(call)
        if code is None:
            return {
                "status": "error",
                "error_type": "MissingCode",
                "message": (
                    f"Tool {call.name!r} is registered as a code tool but no "
                    "code argument was found."
                ),
                "instruction": "Pass the code as the first argument or as code=.",
            }
        language = (
            Language.SHELL
            if call.name.lower()
            in {"shell", "bash", "terminal", "execute_shell", "run_command"}
            else Language.PYTHON
        )
        return self.execute_code(code, language).to_payload()

    @staticmethod
    def _extract_code(call: ToolCall) -> str | None:
        for key in ("code", "command", "query", "script", "input"):
            value = call.kwargs.get(key)
            if isinstance(value, str):
                return value
        if call.args and isinstance(call.args[0], str):
            return call.args[0]
        return None

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _refusal(
        self, call: ToolCall, decision: Decision, watch: Stopwatch
    ) -> ToolResult:
        blocked = decision.tier is PermissionTier.BLOCKED
        status = ExecutionStatus.BLOCKED if blocked else ExecutionStatus.DENIED
        if blocked:
            instruction = (
                "This action is permanently blocked by policy and will never "
                "be approved. Do not retry it. Choose a different approach."
            )
        else:
            instruction = (
                "A human reviewer declined this action. Do not retry it "
                "unchanged — explain what you need, or try a safer step."
            )
        return ToolResult(
            status=status,
            error={
                "status": "error",
                "error_type": status.value,
                "message": (
                    f"Tool call {call.describe()} was refused: "
                    f"{decision.reason or decision.tier.value}"
                ),
                "instruction": instruction,
            },
            duration_ms=watch.elapsed_ms,
            tier=decision.tier,
            metadata={"rule": decision.rule_name},
        )

    def _failure(
        self, exc: Exception, decision: Decision, watch: Stopwatch
    ) -> ToolResult:
        from agentlatch.decorators import _build_error_payload

        return ToolResult(
            status=ExecutionStatus.ERROR,
            error=_build_error_payload(exc),
            duration_ms=watch.elapsed_ms,
            tier=decision.tier,
        )

    def _succeed(self, raw: Any, decision: Decision, watch: Stopwatch) -> ToolResult:
        compacted = False
        content = raw
        if self.compactor is not None:
            outcome = self.compactor.compact(raw)
            content = outcome.content
            compacted = outcome.compacted
        return ToolResult(
            status=ExecutionStatus.OK,
            content=content,
            duration_ms=watch.elapsed_ms,
            tier=decision.tier,
            compacted=compacted,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tier_descriptor(self, descriptor: ToolDescriptor) -> ToolDescriptor:
        """Annotate a descriptor with the tier its name resolves to."""
        if descriptor.tier is None:
            probe = ToolCall(name=descriptor.name)
            descriptor.tier = self.policy.evaluate(probe).tier
        return descriptor

    @staticmethod
    def _close_event(event: Any, result: ToolResult) -> None:
        """Mirror the harness outcome onto the trace tree, when tracing."""
        if event is None:
            return
        mapping = {
            ExecutionStatus.OK: EventStatus.SUCCESS,
            ExecutionStatus.ERROR: EventStatus.ERROR,
            ExecutionStatus.TIMEOUT: EventStatus.TIMEOUT,
            ExecutionStatus.BLOCKED: EventStatus.ERROR,
            ExecutionStatus.DENIED: EventStatus.ERROR,
        }
        end_child(
            event,
            mapping.get(result.status, EventStatus.ERROR),
            result.error,
            {"tier": result.tier.value if result.tier else None},
        )

    def __repr__(self) -> str:
        box = self.sandbox.name if self.sandbox else "none"
        return (
            f"<Harness sandbox={box} rules={len(self.policy.rules)} "
            f"tools={len(self.registry)}>"
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def secure(
    target: Any,
    *,
    framework: str = "auto",
    **kwargs: Any,
) -> tuple[Any, Harness]:
    """One-liner: wrap *target* with a default harness.

    Args:
        target:    The graph, crew, or dict of callables to secure.
        framework: ``"auto"`` sniffs the target's shape; otherwise one of
                   ``"langgraph"``, ``"crewai"``, ``"generic"``.
        **kwargs:  Forwarded to :class:`Harness`.

    Returns:
        ``(secured_target, harness)`` — keep the harness to read the audit log.
    """
    from agentlatch.harness.adapters import (
        CallableAdapter,
        CrewAIAdapter,
        LangGraphAdapter,
    )

    if framework == "auto":
        if isinstance(target, dict) or isinstance(target, list):
            framework = "generic"
        elif hasattr(target, "kickoff") or hasattr(target, "agents"):
            framework = "crewai"
        elif hasattr(target, "invoke") or hasattr(target, "nodes"):
            framework = "langgraph"
        else:
            framework = "generic"

    adapters = {
        "langgraph": LangGraphAdapter,
        "crewai": CrewAIAdapter,
        "generic": CallableAdapter,
    }
    if framework not in adapters:
        raise ValueError(
            f"Unknown framework {framework!r}. Expected one of {sorted(adapters)}."
        )

    harness = Harness(**kwargs)
    return harness.wrap(adapters[framework](target)), harness


def json_error(payload: dict[str, Any]) -> str:
    """Serialize an error payload the way ``@safe_tool`` does."""
    return json.dumps(payload)
