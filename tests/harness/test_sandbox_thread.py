"""Tests for ThreadSandbox — real execution, no mocks."""

from __future__ import annotations

import sys
import threading

from agentlatch.harness._types import (
    ExecutionRequest,
    ExecutionStatus,
    Language,
)
from agentlatch.harness.sandbox.thread import ThreadSandbox


def test_success_captures_stdout():
    sandbox = ThreadSandbox()
    result = sandbox.run(ExecutionRequest(code="print('hello harness')"))

    assert result.status is ExecutionStatus.OK
    assert result.ok
    assert result.stdout.strip() == "hello harness"
    assert result.stderr == ""
    assert result.exit_code == 0
    assert result.runtime == "thread"
    assert result.duration_ms >= 0.0


def test_stderr_is_captured_separately():
    sandbox = ThreadSandbox()
    code = "import sys; print('out'); print('err', file=sys.stderr)"
    result = sandbox.run(ExecutionRequest(code=code))

    assert result.status is ExecutionStatus.OK
    assert result.stdout.strip() == "out"
    assert result.stderr.strip() == "err"


def test_host_stdout_is_restored():
    original = sys.stdout
    ThreadSandbox().run(ExecutionRequest(code="print('x')"))
    assert sys.stdout is original


def test_exception_inside_code_yields_error_with_traceback():
    sandbox = ThreadSandbox()
    code = "print('before')\nraise ValueError('boom')"
    result = sandbox.run(ExecutionRequest(code=code))

    assert result.status is ExecutionStatus.ERROR
    assert not result.ok
    assert result.exit_code == 1
    assert result.stdout.strip() == "before"
    assert "Traceback" in result.stderr
    assert "ValueError: boom" in result.stderr


def test_syntax_error_yields_error():
    result = ThreadSandbox().run(ExecutionRequest(code="def oops(:"))

    assert result.status is ExecutionStatus.ERROR
    assert "SyntaxError" in result.stderr


def test_timeout_returns_timeout_status():
    sandbox = ThreadSandbox()
    code = "import time\nprint('starting', flush=True)\ntime.sleep(5)"
    result = sandbox.run(ExecutionRequest(code=code, timeout=0.3))

    assert result.status is ExecutionStatus.TIMEOUT
    assert not result.ok
    assert "timeout" in result.stderr.lower()
    # The runtime is honest that nothing was actually killed.
    assert result.metadata.get("abandoned_thread") is True
    assert result.duration_ms < 4000


def test_shell_language_is_rejected_not_silently_run():
    result = ThreadSandbox().run(
        ExecutionRequest(code="echo hi", language=Language.SHELL)
    )

    assert result.status is ExecutionStatus.ERROR
    assert "Python only" in result.stderr
    assert "DockerSandbox" in result.stderr
    assert result.stdout == ""


def test_empty_code_handled_by_base():
    for code in ("", "   \n\t "):
        result = ThreadSandbox().run(ExecutionRequest(code=code))
        assert result.status is ExecutionStatus.ERROR
        assert "Empty code" in result.stderr
        assert result.runtime == "thread"


def test_restricted_builtins_block_unlisted_names():
    sandbox = ThreadSandbox(allowed_builtins=["print", "len"])

    allowed = sandbox.run(ExecutionRequest(code="print(len('abcd'))"))
    assert allowed.status is ExecutionStatus.OK
    assert allowed.stdout.strip() == "4"

    denied = sandbox.run(ExecutionRequest(code="open('/etc/passwd')"))
    assert denied.status is ExecutionStatus.ERROR
    assert "NameError" in denied.stderr


def test_unrestricted_builtins_are_the_default():
    result = ThreadSandbox().run(ExecutionRequest(code="print(sorted([2, 1]))"))
    assert result.status is ExecutionStatus.OK
    assert result.stdout.strip() == "[1, 2]"


def test_namespace_is_fresh_between_runs():
    sandbox = ThreadSandbox()
    first = sandbox.run(ExecutionRequest(code="leaked = 42"))
    second = sandbox.run(ExecutionRequest(code="print(leaked)"))

    assert first.status is ExecutionStatus.OK
    assert second.status is ExecutionStatus.ERROR
    assert "NameError" in second.stderr


def test_system_exit_zero_is_ok():
    result = ThreadSandbox().run(
        ExecutionRequest(code="import sys; print('bye'); sys.exit(0)")
    )
    assert result.status is ExecutionStatus.OK
    assert result.stdout.strip() == "bye"
    assert result.exit_code == 0


def test_system_exit_nonzero_is_error():
    result = ThreadSandbox().run(ExecutionRequest(code="import sys; sys.exit(3)"))
    assert result.status is ExecutionStatus.ERROR
    assert result.exit_code == 3


def test_concurrent_runs_do_not_interleave_captured_output():
    """Two simultaneous runs must not steal each other's stdout."""
    sandbox = ThreadSandbox()
    codes = {
        "alpha": "\n".join(
            ["import time"]
            + [f"print('alpha-{i}'); time.sleep(0.01)" for i in range(8)]
        ),
        "beta": "\n".join(
            ["import time"] + [f"print('beta-{i}'); time.sleep(0.01)" for i in range(8)]
        ),
    }
    results: dict[str, object] = {}
    barrier = threading.Barrier(len(codes))

    def worker(tag: str) -> None:
        barrier.wait()
        results[tag] = sandbox.run(ExecutionRequest(code=codes[tag], timeout=10))

    threads = [threading.Thread(target=worker, args=(tag,)) for tag in codes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert set(results) == set(codes)
    for tag in codes:
        result = results[tag]
        assert result.status is ExecutionStatus.OK, result.stderr
        lines = result.stdout.split()
        assert lines == [f"{tag}-{i}" for i in range(8)]
        other = "beta" if tag == "alpha" else "alpha"
        assert other not in result.stdout


def test_close_is_safe_and_context_manager_works():
    with ThreadSandbox() as sandbox:
        assert sandbox.run(ExecutionRequest(code="print(1)")).ok
    ThreadSandbox().close()
