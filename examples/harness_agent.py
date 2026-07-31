"""AgentLatch Harness — end-to-end demo with no agent framework required.

Shows all four layers on a plain dict of tools, so it runs with nothing
installed beyond the base package::

    python examples/harness_agent.py

Add ``--docker`` to route code execution through a real ephemeral container
(requires ``pip install agentlatch[sandbox]`` and a running daemon — colima,
Docker Desktop, or Rancher all work).
"""

from __future__ import annotations

import sys

from agentlatch import profile_agent
from agentlatch.harness import (
    Compactor,
    Harness,
    PermissionPolicy,
    ThreadSandbox,
    ToolCall,
)
from agentlatch.harness.adapters import CallableAdapter

# ---------------------------------------------------------------------------
# A handful of tools, one per permission tier
# ---------------------------------------------------------------------------


def read_config(path: str) -> str:
    """Tier 1 — read-only, auto-approved."""
    return f"contents of {path}: debug=true"


def list_users() -> list[dict[str, object]]:
    """Tier 1 — returns far more rows than any context window wants."""
    return [{"id": i, "email": f"user{i}@example.com"} for i in range(500)]


def delete_user(user_id: int) -> str:
    """Tier 2 — mutates state, so a human is asked first."""
    return f"deleted user {user_id}"


def run_command(command: str) -> str:
    """Tier 3 material — the harness routes this to the sandbox, and blocks
    outright anything matching a destructive pattern."""
    return f"(never reached for blocked input) {command}"


TOOLS = {
    "read_config": read_config,
    "list_users": list_users,
    "delete_user": delete_user,
    "run_command": run_command,
}


# ---------------------------------------------------------------------------
# Approval hook — Tier 2 human-in-the-loop
# ---------------------------------------------------------------------------


def auto_approve_reads(call: ToolCall, decision) -> bool:
    """Stand-in for a webhook or CLI prompt.

    Approves deleting user 1, rejects everything else — enough to show both
    branches without needing interactive input.
    """
    approved = call.kwargs.get("user_id") == 1 or call.args[:1] == (1,)
    verdict = "APPROVED" if approved else "REJECTED"
    print(f"   [human] {call.describe()} -> {verdict}")
    return approved


def main() -> None:
    use_docker = "--docker" in sys.argv

    sandbox = None
    if use_docker:
        from agentlatch.harness import DockerSandbox

        sandbox = DockerSandbox()
        print("Using DockerSandbox (ephemeral, network-isolated).\n")
    else:
        sandbox = ThreadSandbox()
        print("Using ThreadSandbox. Pass --docker for real container isolation.\n")

    harness = Harness(
        policy=PermissionPolicy.default(),
        sandbox=sandbox,
        compactor=Compactor(max_tokens=256, sample_rows=5),
        on_approval=auto_approve_reads,
    )

    tools = harness.wrap(CallableAdapter(TOOLS))

    print("--- Progressive disclosure: what the system prompt actually gets ---")
    print(harness.system_prompt_block())
    print()

    @profile_agent(name="HarnessDemo")
    def run_agent() -> None:
        print("1. Tier 1 read-only tool (runs immediately)")
        print("   ->", tools["read_config"]("/etc/app.conf"))

        print("\n2. Tier 1 tool returning 500 rows (compacted before the LLM sees it)")
        rows = tools["list_users"]()
        shown = len(rows) if isinstance(rows, list) else "?"
        print(f"   -> {shown} items after compaction (was 500)")

        print("\n3. Tier 2 state change, approved")
        print("   ->", tools["delete_user"](user_id=1))

        print("\n4. Tier 2 state change, rejected")
        print("   ->", tools["delete_user"](user_id=99))

        print("\n5. Tier 3 destructive command (blocked by policy, never executes)")
        print("   ->", tools["run_command"]("rm -rf / --no-preserve-root"))

        print("\n6. Safe code, executed in the sandbox rather than on the host")
        result = harness.execute_code("print(sum(range(10)))")
        print("   ->", result.stdout.strip() or result.stderr.strip())

    run_agent()

    print("\n--- Audit log ---")
    for entry in harness.audit_log:
        print(
            f"  {entry.get('tool'):<14} tier={entry.get('tier'):<8} "
            f"allowed={entry.get('allowed')}  {entry.get('reason', '')[:48]}"
        )

    if sandbox is not None:
        sandbox.close()


if __name__ == "__main__":
    main()
