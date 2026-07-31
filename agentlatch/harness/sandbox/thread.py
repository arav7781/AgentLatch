"""In-process thread sandbox with a hard wall-clock timeout.

This is the *weak* runtime.  It exists so a harness can bound the runtime of
trusted code without requiring a container engine.  It is not a security
boundary — see :class:`ThreadSandbox` for the full set of caveats and use
:class:`~agentlatch.harness.sandbox.docker.DockerSandbox` for anything an LLM
wrote.
"""

from __future__ import annotations

import builtins as _builtins
import concurrent.futures
import io
import sys
import threading
import traceback
from collections.abc import Sequence
from typing import Any

from agentlatch.harness._types import (
    ExecutionRequest,
    ExecutionStatus,
    Language,
    SandboxResult,
    Stopwatch,
)
from agentlatch.harness.sandbox.base import BaseSandbox

# ``sys.stdout`` / ``sys.stderr`` are process-global.  Two sandboxed runs that
# swapped them at the same time would capture each other's output, so every run
# serializes on this lock for the duration of the swap.
_CAPTURE_LOCK = threading.Lock()


class ThreadSandbox(BaseSandbox):
    """Run Python source in a worker thread with a wall-clock budget.

    The code is compiled and ``exec``'d in a fresh namespace on a
    :class:`concurrent.futures.ThreadPoolExecutor` worker, and the calling
    thread waits with ``future.result(timeout=...)``.  ``signal.alarm`` is
    deliberately not used: it is POSIX-only and only works on the main thread,
    which is the same reason ``agentlatch.decorators`` uses a thread pool.

    Known limitations — read these before choosing this runtime:

    * **A timeout does not kill anything.**  Python offers no supported way to
      terminate a running thread.  When :meth:`run` reports
      :attr:`~agentlatch.harness._types.ExecutionStatus.TIMEOUT`, the worker
      thread is *still running* in the background and will keep burning CPU
      until it finishes on its own.  The result is honest about the deadline;
      it is not evidence that the work stopped.  For real termination use
      :class:`~agentlatch.harness.sandbox.docker.DockerSandbox`, which kills
      the container.
    * **``allowed_builtins`` is a guardrail, not a jail.**  Restricting
      ``__builtins__`` stops an honest mistake (a stray ``open()``), nothing
      more.  In-process ``exec`` sandboxing is escapable — object graphs reach
      back to the real interpreter through ordinary attribute traversal — so a
      hostile payload owns the host process.  Anything LLM-authored belongs in
      Docker.
    * **Output capture is process-wide.**  While a run holds the capture lock,
      ``print`` calls from unrelated threads land in that run's captured
      stdout.  Output produced by a timed-out thread *after* the deadline
      escapes to the real stdout, because the originals are restored as soon
      as the deadline passes.
    * **Shell is unsupported.**  ``Language.SHELL`` returns an error result
      rather than quietly running something else.

    Args:
        allowed_builtins: Optional whitelist of builtin names exposed to the
            executed code.  ``None`` (the default) exposes the full ``builtins``
            module.  A sequence — even an empty one — swaps in a reduced
            mapping containing only the names listed that actually exist.
    """

    name = "thread"

    def __init__(self, *, allowed_builtins: Sequence[str] | None = None) -> None:
        self.allowed_builtins = (
            None if allowed_builtins is None else tuple(allowed_builtins)
        )

    # -- BaseSandbox hook ----------------------------------------------------

    def _execute(self, request: ExecutionRequest) -> SandboxResult:
        """Compile and run *request* on a worker thread.

        Args:
            request: The execution request.  Only
                :attr:`~agentlatch.harness._types.Language.PYTHON` is
                supported; ``code`` and ``timeout`` are honoured and the
                container-oriented fields (``env``, ``workdir``,
                ``memory_limit_mb``, ``network``) are ignored because this
                runtime has no boundary to apply them to.
        """
        if request.language is not Language.PYTHON:
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                stderr=(
                    f"ThreadSandbox executes Python only; got "
                    f"{request.language.value!r}. Use DockerSandbox for shell "
                    f"commands — it runs them in an isolated container."
                ),
                runtime=self.name,
                metadata={"language": request.language.value},
            )

        watch = Stopwatch()
        try:
            compiled = compile(request.code, "<agentlatch-sandbox>", "exec")
        except SyntaxError:
            return SandboxResult(
                status=ExecutionStatus.ERROR,
                stderr=traceback.format_exc(),
                exit_code=1,
                duration_ms=watch.elapsed_ms,
                runtime=self.name,
            )

        namespace = self._build_namespace()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="agentlatch-sandbox"
        )
        try:
            with _CAPTURE_LOCK:
                real_stdout, real_stderr = sys.stdout, sys.stderr
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                try:
                    future = pool.submit(exec, compiled, namespace)
                    try:
                        future.result(timeout=request.timeout)
                    except concurrent.futures.TimeoutError:
                        # The worker cannot be killed; it runs on. Say so.
                        future.cancel()
                        return SandboxResult(
                            status=ExecutionStatus.TIMEOUT,
                            stdout=stdout_buf.getvalue(),
                            stderr=(
                                f"Execution exceeded the {request.timeout}s "
                                f"timeout. The thread was abandoned, not "
                                f"killed — it keeps running in the background. "
                                f"Use DockerSandbox for enforced termination."
                            ),
                            duration_ms=watch.elapsed_ms,
                            runtime=self.name,
                            terminated=False,
                            metadata={"abandoned_thread": True},
                        )
                    except SystemExit as exc:
                        code = exc.code if isinstance(exc.code, int) else 0
                        return SandboxResult(
                            status=(
                                ExecutionStatus.OK
                                if code == 0
                                else ExecutionStatus.ERROR
                            ),
                            stdout=stdout_buf.getvalue(),
                            stderr=stderr_buf.getvalue(),
                            exit_code=code,
                            duration_ms=watch.elapsed_ms,
                            runtime=self.name,
                        )
                    except Exception:
                        return SandboxResult(
                            status=ExecutionStatus.ERROR,
                            stdout=stdout_buf.getvalue(),
                            stderr=stderr_buf.getvalue() + traceback.format_exc(),
                            exit_code=1,
                            duration_ms=watch.elapsed_ms,
                            runtime=self.name,
                        )
                finally:
                    sys.stdout, sys.stderr = real_stdout, real_stderr
        finally:
            # Never wait: a timed-out worker would block shutdown forever.
            pool.shutdown(wait=False)

        return SandboxResult(
            status=ExecutionStatus.OK,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            exit_code=0,
            duration_ms=watch.elapsed_ms,
            runtime=self.name,
        )

    # -- Internals -----------------------------------------------------------

    def _build_namespace(self) -> dict[str, Any]:
        """Create the fresh globals mapping the code is executed in."""
        namespace: dict[str, Any] = {"__name__": "__agentlatch_sandbox__"}
        if self.allowed_builtins is None:
            namespace["__builtins__"] = _builtins
        else:
            namespace["__builtins__"] = {
                name: getattr(_builtins, name)
                for name in self.allowed_builtins
                if hasattr(_builtins, name)
            }
        return namespace
