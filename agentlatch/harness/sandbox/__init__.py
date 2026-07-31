"""Pluggable ephemeral execution runtimes.

Importing this package never imports ``docker`` — :class:`DockerSandbox`
resolves its client lazily, so the base install stays dependency-free.

    from agentlatch.harness.sandbox import ThreadSandbox, DockerSandbox

    with DockerSandbox() as box:
        result = box.run(ExecutionRequest(code="print(2 + 2)"))
"""

from agentlatch.harness.sandbox.base import BaseSandbox
from agentlatch.harness.sandbox.docker import DockerSandbox
from agentlatch.harness.sandbox.thread import ThreadSandbox

__all__ = ["BaseSandbox", "ThreadSandbox", "DockerSandbox"]
