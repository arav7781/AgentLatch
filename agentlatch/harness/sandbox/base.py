"""Base class and shared helpers for sandbox runtimes.

A sandbox runs agent-authored code somewhere that is not the host process.
Runtimes differ in how strong that boundary is:

* :class:`~agentlatch.harness.sandbox.thread.ThreadSandbox` — same process,
  separate thread, hard wall-clock timeout.  Contains *accidents*, not
  *adversaries*: a determined payload can still touch the filesystem and the
  network.  Use it for trusted code where you only need timeout safety.
* :class:`~agentlatch.harness.sandbox.docker.DockerSandbox` — ephemeral
  container, no network, dropped capabilities, read-only root.  Use it for
  anything an LLM wrote.

Both return a :class:`~agentlatch.harness._types.SandboxResult` instead of
raising, so a failed execution feeds back to the LLM as data.
"""

from __future__ import annotations

import abc

from agentlatch.harness._types import (
    ExecutionRequest,
    ExecutionStatus,
    SandboxResult,
)


class BaseSandbox(abc.ABC):
    """Abstract sandbox runtime.

    Subclasses implement :meth:`_execute`.  The public :meth:`run` wrapper
    enforces the "never raise on ordinary failure" contract, so subclasses may
    let exceptions escape ``_execute`` and still satisfy the protocol.
    """

    name: str = "base"

    # -- Subclass hook -------------------------------------------------------

    @abc.abstractmethod
    def _execute(self, request: ExecutionRequest) -> SandboxResult:
        """Run *request* and capture its output."""

    # -- Public API ----------------------------------------------------------

    def run(self, request: ExecutionRequest) -> SandboxResult:
        """Execute *request*, converting unexpected errors into a result."""
        if not request.code or not request.code.strip():
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                stderr="Empty code payload — nothing to execute.",
                runtime=self.name,
            )
        try:
            return self._execute(request)
        except Exception as exc:  # noqa: BLE001 - boundary: must not propagate
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                stderr=f"{type(exc).__name__}: {exc}",
                runtime=self.name,
            )

    def close(self) -> None:  # noqa: B027 - optional hook, not every runtime holds resources
        """Release resources.

        Deliberately concrete rather than abstract: a runtime that holds
        nothing (ThreadSandbox) should not be forced to write an empty
        override just to satisfy the interface.
        """

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> BaseSandbox:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
