"""Tests for DockerSandbox.

These run with the docker SDK absent and no daemon: the SDK is faked by
patching ``sys.modules["docker"]``.  One real integration test is included but
skipped unless ``AGENTLATCH_DOCKER_TESTS=1``.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from agentlatch.harness._types import (
    ExecutionRequest,
    ExecutionStatus,
    Language,
)
from agentlatch.harness.sandbox import docker as docker_mod
from agentlatch.harness.sandbox.docker import DockerSandbox

COLIMA = os.path.expanduser("~/.colima/default/docker.sock")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeContainer:
    """Stand-in for a docker-py container object."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        wait_result=None,
        wait_exc: BaseException | None = None,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._wait_result = {"StatusCode": 0} if wait_result is None else wait_result
        self._wait_exc = wait_exc
        self.wait_calls: list[dict] = []
        self.killed = 0
        self.removed: list[bool] = []

    def wait(self, **kwargs):
        self.wait_calls.append(kwargs)
        if self._wait_exc is not None:
            raise self._wait_exc
        return self._wait_result

    def logs(self, stdout=True, stderr=True):
        if stdout and not stderr:
            return self._stdout
        if stderr and not stdout:
            return self._stderr
        return self._stdout + self._stderr

    def kill(self):
        self.killed += 1

    def remove(self, force=False):
        self.removed.append(force)


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self._container = container
        self.run_kwargs: dict = {}

    def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self._container


class FakeClient:
    def __init__(
        self, container: FakeContainer | None = None, base_url="fake://"
    ) -> None:
        self.containers = FakeContainers(container or FakeContainer())
        self.base_url = base_url
        self.closed = 0
        self.pings = 0

    def ping(self):
        self.pings += 1
        return True

    def close(self):
        self.closed += 1


def install_fake_docker(monkeypatch, *, from_env=None, docker_client=None):
    """Put a fake ``docker`` module into ``sys.modules`` for one test."""
    module = types.ModuleType("docker")

    def _unavailable(*args, **kwargs):
        raise ConnectionError("no daemon here")

    module.from_env = from_env or _unavailable
    module.DockerClient = docker_client or _unavailable
    monkeypatch.setitem(sys.modules, "docker", module)
    return module


def make_sandbox(
    container: FakeContainer, **kwargs
) -> tuple[DockerSandbox, FakeClient]:
    client = FakeClient(container)
    return DockerSandbox(client=client, **kwargs), client


# ---------------------------------------------------------------------------
# Import laziness
# ---------------------------------------------------------------------------


def test_module_imports_without_docker_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "docker", None)
    # Constructing must not touch the SDK at all.
    sandbox = DockerSandbox()
    assert sandbox.name == "docker"
    assert sandbox.image == "python:3.11-slim"


def test_missing_sdk_raises_import_error_naming_the_extra(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "docker":
            raise ImportError("No module named 'docker'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "docker", raising=False)
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        DockerSandbox()._import_docker()
    assert "pip install agentlatch[sandbox]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Socket discovery
# ---------------------------------------------------------------------------


def test_discovery_picks_colima_socket_when_only_it_exists(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    built: list[str] = []

    def docker_client(base_url):
        built.append(base_url)
        return FakeClient(base_url=base_url)

    install_fake_docker(monkeypatch, docker_client=docker_client)
    monkeypatch.setattr(os.path, "exists", lambda p: p == COLIMA)

    client = DockerSandbox()._make_client()

    assert built == [f"unix://{COLIMA}"]
    assert client.base_url == f"unix://{COLIMA}"
    assert client.pings == 1


def test_docker_host_wins_over_everything(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2375")
    from_env_calls = []

    def from_env():
        from_env_calls.append(1)
        raise ConnectionError("should never be reached")

    install_fake_docker(
        monkeypatch,
        from_env=from_env,
        docker_client=lambda base_url: FakeClient(base_url=base_url),
    )
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    client = DockerSandbox()._make_client()

    assert client.base_url == "tcp://127.0.0.1:2375"
    assert from_env_calls == []


def test_from_env_is_tried_before_sockets(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    install_fake_docker(monkeypatch, from_env=lambda: FakeClient(base_url="from_env"))
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    assert DockerSandbox()._make_client().base_url == "from_env"


def test_runtime_error_names_every_path_tried(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://nope:2375")
    install_fake_docker(monkeypatch)  # every factory raises
    monkeypatch.setattr(os.path, "exists", lambda p: p == COLIMA)

    with pytest.raises(RuntimeError) as excinfo:
        DockerSandbox()._make_client()

    message = str(excinfo.value)
    assert "DOCKER_HOST=tcp://nope:2375" in message
    assert "docker.from_env()" in message
    for candidate in docker_mod.SOCKET_CANDIDATES:
        assert os.path.expanduser(candidate) in message
    assert "no such socket" in message  # non-existent ones are labelled
    assert f"unix://{COLIMA}" in message  # the existing one was truly attempted


def test_ping_failure_rejects_the_client(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)

    class DeadClient(FakeClient):
        def ping(self):
            raise ConnectionError("daemon not running")

    install_fake_docker(monkeypatch, from_env=lambda: DeadClient())
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    with pytest.raises(RuntimeError) as excinfo:
        DockerSandbox()._make_client()
    assert "daemon not running" in str(excinfo.value)


def test_client_is_created_once_and_reused(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    calls: list[int] = []

    def from_env():
        calls.append(1)
        return FakeClient()

    install_fake_docker(monkeypatch, from_env=from_env)
    sandbox = DockerSandbox()
    assert sandbox.client is sandbox.client
    assert calls == [1]


# ---------------------------------------------------------------------------
# Container hardening
# ---------------------------------------------------------------------------


def test_container_kwargs_are_hardened():
    container = FakeContainer(stdout=b"hi\n")
    sandbox, client = make_sandbox(container)
    sandbox.run(ExecutionRequest(code="print('hi')", memory_limit_mb=128))

    kwargs = client.containers.run_kwargs
    assert kwargs["network_mode"] == "none"
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges"]
    assert kwargs["read_only"] is True
    assert kwargs["detach"] is True
    assert kwargs["user"] == "nobody"
    assert kwargs["mem_limit"] == "128m"
    assert kwargs["pids_limit"] == 128
    assert kwargs["working_dir"] == "/sandbox"
    assert "/sandbox" in kwargs["tmpfs"]
    assert kwargs["image"] == "python:3.11-slim"


def test_code_travels_as_command_not_a_mount():
    container = FakeContainer()
    sandbox, client = make_sandbox(container)
    sandbox.run(ExecutionRequest(code="print(1)"))

    kwargs = client.containers.run_kwargs
    assert kwargs["command"] == ["python", "-c", "print(1)"]
    assert "volumes" not in kwargs
    assert "mounts" not in kwargs


def test_shell_language_uses_sh():
    container = FakeContainer()
    sandbox, client = make_sandbox(container)
    sandbox.run(ExecutionRequest(code="echo hi", language=Language.SHELL))

    assert client.containers.run_kwargs["command"] == ["/bin/sh", "-c", "echo hi"]


def test_only_request_env_is_passed(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_API_KEY", "sk-do-not-leak")
    container = FakeContainer()
    sandbox, client = make_sandbox(container)
    sandbox.run(ExecutionRequest(code="print(1)", env={"MODE": "test"}))

    assert client.containers.run_kwargs["environment"] == {"MODE": "test"}


def test_network_true_opts_into_an_interface():
    container = FakeContainer()
    sandbox, client = make_sandbox(container)
    sandbox.run(ExecutionRequest(code="print(1)", network=True))

    assert client.containers.run_kwargs["network_mode"] != "none"


def test_user_none_omits_the_user_kwarg():
    container = FakeContainer()
    sandbox, client = make_sandbox(container, user=None)
    sandbox.run(ExecutionRequest(code="print(1)"))

    assert "user" not in client.containers.run_kwargs


# ---------------------------------------------------------------------------
# Output, exit codes, cleanup
# ---------------------------------------------------------------------------


def test_stdout_and_stderr_are_captured_separately():
    container = FakeContainer(stdout=b"out-here\n", stderr=b"err-here\n")
    sandbox, _ = make_sandbox(container)
    result = sandbox.run(ExecutionRequest(code="print(1)"))

    assert result.status is ExecutionStatus.OK
    assert result.stdout == "out-here\n"
    assert result.stderr == "err-here\n"
    assert result.exit_code == 0
    assert result.runtime == "docker"


def test_undecodable_bytes_are_replaced_not_raised():
    container = FakeContainer(stdout=b"\xff\xfe bad")
    sandbox, _ = make_sandbox(container)
    result = sandbox.run(ExecutionRequest(code="print(1)"))

    assert result.status is ExecutionStatus.OK
    assert "bad" in result.stdout


def test_nonzero_exit_code_is_an_error():
    container = FakeContainer(stderr=b"boom\n", wait_result={"StatusCode": 2})
    sandbox, _ = make_sandbox(container)
    result = sandbox.run(ExecutionRequest(code="raise SystemExit(2)"))

    assert result.status is ExecutionStatus.ERROR
    assert result.exit_code == 2
    assert result.stderr == "boom\n"


def test_container_is_always_cleaned_up_on_success():
    container = FakeContainer()
    sandbox, _ = make_sandbox(container)
    sandbox.run(ExecutionRequest(code="print(1)"))

    assert container.killed == 1
    assert container.removed == [True]


def test_cleanup_runs_even_when_wait_raises():
    container = FakeContainer(wait_exc=RuntimeError("daemon exploded"))
    sandbox, _ = make_sandbox(container)
    result = sandbox.run(ExecutionRequest(code="print(1)"))

    # BaseSandbox converts the escape into an ERROR result...
    assert result.status is ExecutionStatus.ERROR
    assert "daemon exploded" in result.stderr
    # ...and the container is still gone.
    assert container.killed == 1
    assert container.removed == [True]


def test_cleanup_survives_a_container_that_is_already_dead():
    class Zombie(FakeContainer):
        def kill(self):
            raise RuntimeError("container is not running")

        def remove(self, force=False):
            raise RuntimeError("no such container")

    container = Zombie()
    sandbox, _ = make_sandbox(container)
    result = sandbox.run(ExecutionRequest(code="print(1)"))

    assert result.status is ExecutionStatus.OK


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class ReadTimeout(Exception):
    """Mimics requests' read-timeout surfaced by container.wait."""


def test_timeout_force_kills_and_reports_timeout():
    container = FakeContainer(
        stdout=b"partial\n",
        wait_exc=ReadTimeout("HTTPConnectionPool: Read timed out. (read timeout=2)"),
    )
    sandbox, _ = make_sandbox(container)
    result = sandbox.run(
        ExecutionRequest(code="import time; time.sleep(99)", timeout=2)
    )

    assert result.status is ExecutionStatus.TIMEOUT
    assert result.stdout == "partial\n"
    assert "timeout" in result.stderr.lower()
    assert result.metadata.get("killed") is True
    # Killed on the timeout path, then again by the unconditional cleanup.
    assert container.killed >= 1
    assert container.removed == [True]


def test_timeout_is_passed_through_to_wait():
    container = FakeContainer()
    sandbox, _ = make_sandbox(container)
    sandbox.run(ExecutionRequest(code="print(1)", timeout=7.5))

    assert container.wait_calls == [{"timeout": 7.5}]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_closes_the_client():
    client = FakeClient()
    sandbox = DockerSandbox(client=client)
    sandbox.close()

    assert client.closed == 1
    sandbox.close()  # idempotent
    assert client.closed == 1


def test_empty_code_never_starts_a_container():
    container = FakeContainer()
    sandbox, client = make_sandbox(container)
    result = sandbox.run(ExecutionRequest(code="   "))

    assert result.status is ExecutionStatus.ERROR
    assert client.containers.run_kwargs == {}


# ---------------------------------------------------------------------------
# Optional real-daemon integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AGENTLATCH_DOCKER_TESTS") != "1",
    reason="set AGENTLATCH_DOCKER_TESTS=1 to run against a real Docker daemon",
)
def test_real_container_roundtrip():
    with DockerSandbox() as sandbox:
        result = sandbox.run(
            ExecutionRequest(code="print('from the container')", timeout=120)
        )
        assert result.status is ExecutionStatus.OK, result.stderr
        assert "from the container" in result.stdout
        assert result.exit_code == 0

        blocked = sandbox.run(
            ExecutionRequest(
                code=("import socket; socket.create_connection(('1.1.1.1', 80), 5)"),
                timeout=120,
            )
        )
        assert blocked.status is ExecutionStatus.ERROR  # no network
