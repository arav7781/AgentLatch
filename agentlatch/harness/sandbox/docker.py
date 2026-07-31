"""Container sandbox: one ephemeral, network-isolated container per execution.

This is the runtime to use for LLM-authored code.  Every execution gets a fresh
container with no network, no capabilities, a read-only root filesystem, and
only the environment variables the caller passed explicitly.  The container is
killed and removed unconditionally when the run finishes, times out, or errors.

``docker`` is an optional extra.  It is imported lazily so that importing this
module — which ``agentlatch.harness.sandbox`` does eagerly — never fails on a
machine without the SDK installed.
"""

from __future__ import annotations

import os
from typing import Any

from agentlatch.harness._types import (
    ExecutionRequest,
    ExecutionStatus,
    Language,
    SandboxResult,
    Stopwatch,
)
from agentlatch.harness.sandbox.base import BaseSandbox

DEFAULT_IMAGE = "python:3.11-slim"

#: Unix sockets probed, in order, when ``DOCKER_HOST`` is unset and
#: ``docker.from_env()`` cannot reach a daemon.  Colima comes first: on macOS it
#: is the common case and it does *not* create ``/var/run/docker.sock``.
SOCKET_CANDIDATES: tuple[str, ...] = (
    "~/.colima/default/docker.sock",
    "~/.colima/docker.sock",
    "/var/run/docker.sock",
    "~/.docker/run/docker.sock",  # Docker Desktop
    "~/.rd/docker.sock",  # Rancher Desktop
)

_MISSING_SDK = (
    "DockerSandbox requires the docker SDK, which is an optional extra. "
    "Install it with `pip install agentlatch[sandbox]`."
)


def _looks_like_timeout(exc: BaseException) -> bool:
    """Whether *exc* is docker-py's way of reporting a ``wait`` deadline.

    The SDK surfaces a ``wait(timeout=...)`` expiry as a ``requests``
    ``ConnectionError`` wrapping a ``ReadTimeoutError``, so the type alone is
    not conclusive and the message has to be inspected too.

    Args:
        exc: Exception raised by ``container.wait``.
    """
    blob = f"{type(exc).__name__} {exc}".lower()
    return "timeout" in blob or "timed out" in blob


class DockerSandbox(BaseSandbox):
    """Execute a request inside a locked-down, single-use container.

    Hardening applied to every container: ``network_mode="none"`` unless the
    request opts in, all capabilities dropped, ``no-new-privileges``, a
    read-only root filesystem with a small writable tmpfs mounted at the
    request's workdir, a pid ceiling, a memory ceiling, and an unprivileged
    user.  No host path is ever mounted — the code travels as the container
    command (``python -c`` or ``sh -c``) — and only ``request.env`` is passed,
    so the agent cannot read the host's credentials out of ``os.environ``.

    Args:
        image: Image the container is created from.  Must contain a ``python``
            interpreter for :attr:`~agentlatch.harness._types.Language.PYTHON`
            requests and ``/bin/sh`` for shell requests.
        user: User the container process runs as.  ``"nobody"`` by default;
            pass ``None`` to accept the image's default user, which some
            images require in order to import their site-packages.
        pids_limit: Maximum number of processes inside the container, a cheap
            fork-bomb guard.
        tmpfs_size_mb: Size of the writable tmpfs mounted at the workdir.
        socket_candidates: Override the Unix sockets probed during client
            discovery.  Mostly a testing seam.
        client: Pre-built docker client.  When given, discovery is skipped
            entirely and :meth:`close` still closes it.
    """

    name = "docker"

    def __init__(
        self,
        *,
        image: str = DEFAULT_IMAGE,
        user: str | None = "nobody",
        pids_limit: int = 128,
        tmpfs_size_mb: int = 64,
        socket_candidates: tuple[str, ...] | None = None,
        client: Any | None = None,
    ) -> None:
        self.image = image
        self.user = user
        self.pids_limit = pids_limit
        self.tmpfs_size_mb = tmpfs_size_mb
        self.socket_candidates = (
            SOCKET_CANDIDATES if socket_candidates is None else tuple(socket_candidates)
        )
        self._client = client

    # -- Client discovery ----------------------------------------------------

    @staticmethod
    def _import_docker() -> Any:
        """Import the optional ``docker`` SDK, or explain how to get it."""
        try:
            import docker  # noqa: PLC0415 - deliberate lazy optional import
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_MISSING_SDK) from exc
        return docker

    def _make_client(self) -> Any:
        """Find a reachable Docker daemon and return a verified client.

        Tries ``DOCKER_HOST``, then ``docker.from_env()``, then each socket in
        :attr:`socket_candidates` that exists on disk.  Every candidate is
        proved with ``client.ping()`` before being accepted — a client object
        constructs happily against a dead socket, so construction alone means
        nothing.

        Raises:
            RuntimeError: No candidate answered.  The message lists every
                endpoint tried and why it was rejected, because that list is
                the only useful thing to hand someone debugging a colima or
                Rancher setup.
        """
        docker = self._import_docker()
        tried: list[str] = []

        host = os.environ.get("DOCKER_HOST")
        if host:
            client = self._try_client(
                lambda: docker.DockerClient(base_url=host),
                f"DOCKER_HOST={host}",
                tried,
            )
            if client is not None:
                return client

        client = self._try_client(docker.from_env, "docker.from_env()", tried)
        if client is not None:
            return client

        for candidate in self.socket_candidates:
            path = os.path.expanduser(candidate)
            if not os.path.exists(path):
                tried.append(f"{path} (no such socket)")
                continue
            base_url = f"unix://{path}"
            client = self._try_client(
                lambda url=base_url: docker.DockerClient(base_url=url),
                base_url,
                tried,
            )
            if client is not None:
                return client

        raise RuntimeError(
            "Could not reach a Docker daemon. Tried, in order:\n  "
            + "\n  ".join(tried)
            + "\nStart your engine (e.g. `colima start`) or set DOCKER_HOST."
        )

    @staticmethod
    def _try_client(factory: Any, label: str, tried: list[str]) -> Any | None:
        """Build a client via *factory* and keep it only if ``ping`` succeeds.

        Args:
            factory: Zero-argument callable returning a docker client.
            label:   Human-readable endpoint name recorded in *tried*.
            tried:   Accumulator of attempted endpoints, mutated in place so
                     the final error message can name all of them.
        """
        try:
            client = factory()
            client.ping()
        except Exception as exc:  # noqa: BLE001 - any failure means "try next"
            tried.append(f"{label} ({type(exc).__name__}: {exc})")
            return None
        return client

    @property
    def client(self) -> Any:
        """The shared docker client, discovered on first use."""
        if self._client is None:
            self._client = self._make_client()
        return self._client

    # -- BaseSandbox hook ----------------------------------------------------

    def _execute(self, request: ExecutionRequest) -> SandboxResult:
        """Run *request* in a fresh container and collect its output.

        Args:
            request: What to execute.  ``timeout`` bounds ``container.wait``,
                ``memory_limit_mb`` becomes the cgroup memory ceiling,
                ``network`` gates whether the container gets an interface, and
                ``env`` is the *complete* environment — nothing is inherited.
        """
        watch = Stopwatch()
        container = None
        try:
            container = self.client.containers.run(**self._container_kwargs(request))
            try:
                wait_result = container.wait(timeout=request.timeout)
            except Exception as exc:
                if not _looks_like_timeout(exc):
                    raise
                self._force_kill(container)
                return SandboxResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout=self._logs(container, stdout=True),
                    stderr=(
                        f"Execution exceeded the {request.timeout}s timeout; "
                        f"the container was killed."
                    ),
                    duration_ms=watch.elapsed_ms,
                    runtime=self.name,
                    terminated=True,
                    metadata={"image": self.image, "killed": True},
                )

            exit_code = _status_code(wait_result)
            return SandboxResult(
                status=(
                    ExecutionStatus.OK if exit_code == 0 else ExecutionStatus.ERROR
                ),
                stdout=self._logs(container, stdout=True),
                stderr=self._logs(container, stdout=False),
                exit_code=exit_code,
                duration_ms=watch.elapsed_ms,
                runtime=self.name,
                metadata={"image": self.image},
            )
        finally:
            # Unconditional: a leaked container is a real bug, and every exit
            # path from this method — success, timeout, raise — lands here.
            if container is not None:
                self._force_kill(container)
                self._force_remove(container)

    # -- Internals -----------------------------------------------------------

    def _container_kwargs(self, request: ExecutionRequest) -> dict[str, Any]:
        """Assemble the hardened ``containers.run`` keyword arguments.

        Args:
            request: Request whose language selects the command form and whose
                limits become container constraints.
        """
        if request.language is Language.SHELL:
            command = ["/bin/sh", "-c", request.code]
        else:
            command = ["python", "-c", request.code]

        kwargs: dict[str, Any] = {
            "image": self.image,
            "command": command,
            "detach": True,
            "network_mode": "bridge" if request.network else "none",
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "read_only": True,
            "pids_limit": self.pids_limit,
            "working_dir": request.workdir,
            "tmpfs": {request.workdir: f"rw,noexec,nosuid,size={self.tmpfs_size_mb}m"},
            # Only what the caller passed. os.environ is never forwarded.
            "environment": dict(request.env),
            "stdout": True,
            "stderr": True,
        }
        if request.memory_limit_mb:
            kwargs["mem_limit"] = f"{request.memory_limit_mb}m"
        if self.user is not None:
            kwargs["user"] = self.user
        return kwargs

    @staticmethod
    def _logs(container: Any, *, stdout: bool) -> str:
        """Read one stream off *container*, never raising.

        Args:
            container: Container to read from.
            stdout:    ``True`` for stdout, ``False`` for stderr.  The two are
                       fetched separately so they stay unmerged in the result.
        """
        try:
            raw = container.logs(stdout=stdout, stderr=not stdout)
        except Exception:  # noqa: BLE001 - a dead container has no logs
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw or "")

    @staticmethod
    def _force_kill(container: Any) -> None:
        """Kill *container*, ignoring the "already dead" case."""
        try:
            container.kill()
        except Exception:  # noqa: BLE001 - already exited is the common case
            pass

    @staticmethod
    def _force_remove(container: Any) -> None:
        """Remove *container*, ignoring the "already gone" case."""
        try:
            container.remove(force=True)
        except Exception:  # noqa: BLE001 - removal is best-effort by design
            pass

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the shared docker client, if one was ever created."""
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - closing must not raise
                pass


def _status_code(wait_result: Any) -> int | None:
    """Pull the exit code out of a ``container.wait`` payload.

    Args:
        wait_result: Whatever the SDK returned — normally
            ``{"StatusCode": int, "Error": ...}``, but older shapes return a
            bare integer.
    """
    if isinstance(wait_result, dict):
        code = wait_result.get("StatusCode")
        return int(code) if isinstance(code, (int, float)) else None
    if isinstance(wait_result, int):
        return wait_result
    return None
