"""Tiered permission system for the AgentLatch Harness.

Every tool call the harness intercepts is classified into one of three tiers
before anything executes:

* :attr:`~agentlatch.harness._types.PermissionTier.AUTO` — read-only, runs
  immediately.
* :attr:`~agentlatch.harness._types.PermissionTier.HUMAN` — mutates state,
  pauses for an approval callback.
* :attr:`~agentlatch.harness._types.PermissionTier.BLOCKED` — destructive,
  never executes and never reaches the approval callback.

The module is standard library only, like the rest of the harness core.  Two
invariants drive every design choice here:

1. **Fail closed.**  An unrecognised tool is Tier 2, not Tier 1.  A missing,
   broken, or slow approval callback denies rather than allows.
2. **Most restrictive wins.**  Rules are not first-match.  All matching rules
   are collected and the highest
   :attr:`~agentlatch.harness._types.PermissionTier.restrictiveness` decides,
   so an auto-approved tool name cannot smuggle a blocked command through.
"""

from __future__ import annotations

import concurrent.futures
import fnmatch
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from agentlatch.harness._types import (
    ApprovalCallback,
    Decision,
    PermissionTier,
    ToolCall,
)

__all__ = [
    "Rule",
    "PermissionPolicy",
    "PermissionGate",
    "cli_approval_callback",
    "READ_ONLY_TOOL_GLOBS",
    "STATE_CHANGING_TOOL_GLOBS",
    "BLOCK_PATTERNS",
]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    """One matching rule: a tier plus the calls it applies to.

    A rule matches a call when *any* of its configured matchers hit — the tool
    name glob list or the command regex.  Matchers are deliberately OR'd:
    a block rule should fire whether the danger is visible in the tool name or
    only in the arguments.

    Attributes:
        name:    Short identifier surfaced on the resulting
                 :class:`~agentlatch.harness._types.Decision`.
        tier:    Tier this rule assigns when it matches.
        tools:   Glob patterns (``fnmatch``) tested against ``ToolCall.name``,
                 e.g. ``["read_*", "get_*"]``.  ``None`` disables name matching.
        pattern: Regex tested against ``ToolCall.command_text()``.  Accepts a
                 string or a pre-compiled pattern.  ``None`` disables it.
        reason:  Human-readable justification shown in approval prompts, audit
                 entries, and the denial message handed back to the LLM.

    Raises:
        ValueError: If neither ``tools`` nor ``pattern`` is supplied.  A rule
            with no matchers would match every call, which silently rewrites
            the policy's tier for the whole tool surface.
    """

    name: str
    tier: PermissionTier
    tools: list[str] | None = None
    pattern: str | re.Pattern[str] | None = None
    reason: str = ""

    _regex: re.Pattern[str] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the matchers and compile the regex exactly once.

        Compiling here rather than in :meth:`matches` keeps the hot path free
        of regex construction — the gate runs on every single tool call.
        """
        if not self.tools and self.pattern is None:
            raise ValueError(
                f"Rule {self.name!r} defines neither 'tools' nor 'pattern'. "
                "A rule with no matchers would match every tool call; specify "
                "at least one matcher."
            )
        if self.pattern is None:
            self._regex = None
        elif isinstance(self.pattern, re.Pattern):
            self._regex = self.pattern
        else:
            self._regex = re.compile(self.pattern)

    @property
    def regex(self) -> re.Pattern[str] | None:
        """The compiled pattern, or ``None`` when the rule matches by name."""
        return self._regex

    def matches(self, call: ToolCall) -> bool:
        """Report whether this rule applies to *call*.

        Args:
            call: The intercepted tool call.

        Returns:
            ``True`` if the tool name matches any glob, or the compiled regex
            finds a hit anywhere in ``call.command_text()``.
        """
        if self.tools:
            for glob in self.tools:
                if fnmatch.fnmatch(call.name, glob):
                    return True
        if self._regex is not None and self._regex.search(call.command_text()):
            return True
        return False


# ---------------------------------------------------------------------------
# Default rule sets
# ---------------------------------------------------------------------------

READ_ONLY_TOOL_GLOBS: list[str] = [
    "read_*",
    "get_*",
    "list_*",
    "search_*",
    "fetch_*",
    "query_*",
    "describe_*",
    "view_*",
]
"""Tool-name globs treated as Tier 1 by :meth:`PermissionPolicy.default`."""

STATE_CHANGING_TOOL_GLOBS: list[str] = [
    "write_*",
    "delete_*",
    "create_*",
    "update_*",
    "send_*",
    "deploy_*",
    "execute_*",
    "run_*",
    "publish_*",
    "drop_*",
]
"""Tool-name globs treated as Tier 2 by :meth:`PermissionPolicy.default`."""


# Every pattern below is anchored on word boundaries or command position so it
# fires on a real command and not on prose that happens to contain the letters.
# "check performance metrics" must not trip the rm rules; "format the output"
# must not trip mkfs; "run the addendum" must not trip the dd rule.
BLOCK_PATTERNS: list[tuple[str, str, str]] = [
    (
        "block_rm_recursive_force",
        r"(?i)\brm\b(?:\s+-{1,2}[\w-]+)*\s+-[a-z]*(?:rf|fr)[a-z]*\b",
        "Recursive forced delete (rm -rf / rm -fr) is irreversible.",
    ),
    (
        "block_rm_root_target",
        r"(?i)\brm\b(?:\s+-{1,2}[\w-]+)+\s+(?:/|~|\*)(?:\s|/|$)",
        "Recursive delete aimed at /, ~ or * destroys the filesystem.",
    ),
    (
        "block_mkfs",
        r"(?i)\bmkfs(?:\.[a-z0-9]+)?\b",
        "Formatting a filesystem erases the target device.",
    ),
    (
        "block_dd_to_device",
        r"(?i)\bdd\b[^\n]*\bof=\s*/dev/\w+",
        "Writing raw blocks to a device overwrites the disk.",
    ),
    (
        "block_fork_bomb",
        r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        "Fork bomb — exhausts every process slot on the host.",
    ),
    (
        "block_power_state",
        r"(?im)(?:^|[;&|]\s*|\bsudo\s+)(?:/s?bin/)?"
        r"(?:shutdown|reboot|halt|poweroff)\b",
        "Powering off or rebooting the host kills the agent and its operator.",
    ),
    (
        "block_chmod_777_root",
        r"(?i)\bchmod\b(?:\s+-{1,2}[\w-]+)*\s+0?777\s+(?:/|~)(?:\s|$)",
        "chmod 777 on / or ~ makes the whole tree world-writable.",
    ),
    (
        "block_curl_pipe_shell",
        r"(?i)\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|k|da|c)?sh\b",
        "Piping a download straight into a shell executes unreviewed code.",
    ),
    (
        "block_history_clear",
        r"(?i)\bhistory\s+-c\b",
        "Clearing shell history destroys the audit trail.",
    ),
    (
        "block_credential_file_write",
        r"(?i)(?:>>?|\btee\b|\bsed\s+-i\b|\brm\b|\bmv\b|\bcp\b|\bchown\b|"
        r"\bchmod\b|\bpasswd\b)[^\n]*/etc/(?:passwd|shadow)\b",
        "Modifying /etc/passwd or /etc/shadow tampers with system accounts.",
    ),
    (
        "block_force_push_main",
        r"(?i)\bgit\s+push\b(?=[^\n]*(?:--force\b|--force-with-lease\b|\s-f\b))"
        r"(?=[^\n]*\b(?:main|master)\b)",
        "Force-pushing main/master can destroy shared history.",
    ),
    (
        "block_sql_drop",
        r"(?i)\bdrop\s+(?:database|schema|table)\b",
        "DROP removes a database object and everything in it.",
    ),
    (
        "block_sql_truncate",
        # The bare form is matched case-sensitively so English prose such as
        # "truncate the output" cannot trip it; SQL keywords are upper-cased.
        r"(?i:\btruncate\s+table\b)|\bTRUNCATE\s+[\"`\[]?\w+",
        "TRUNCATE empties a table with no undo.",
    ),
    (
        "block_sudo",
        r"(?i)\bsudo\s+\S",
        "Privilege escalation is outside the agent's blast radius.",
    ),
]
"""``(name, regex, reason)`` triples used to build the Tier 3 rules.

Exported so applications can inspect, subset, or extend the shipped defaults
instead of re-deriving them.
"""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class PermissionPolicy:
    """An ordered collection of :class:`Rule` objects plus a fallback tier.

    Evaluation is *not* first-match.  Every matching rule is collected and the
    most restrictive tier wins, so adding a block rule can never be defeated by
    an earlier, broader allow rule.

    The default fallback is
    :attr:`~agentlatch.harness._types.PermissionTier.HUMAN`, never ``AUTO``.
    A policy is a list of the calls someone thought about; a tool nobody
    thought about is by definition unreviewed.  Failing closed means the worst
    case for an unknown tool is an approval prompt, whereas failing open means
    the worst case is arbitrary unreviewed side effects.  Callers who genuinely
    want an allow-by-default sandbox must opt in explicitly.

    Args:
        rules:        Rules to evaluate, in declaration order.  Order affects
                      only tie-breaking between rules of equal tier (first
                      match of the winning tier is reported).
        default_tier: Tier applied when no rule matches.  Defaults to ``HUMAN``.
    """

    def __init__(
        self,
        rules: list[Rule],
        default_tier: PermissionTier = PermissionTier.HUMAN,
    ) -> None:
        self.rules: list[Rule] = list(rules)
        self.default_tier = default_tier

    def __repr__(self) -> str:
        return (
            f"PermissionPolicy(rules={len(self.rules)}, "
            f"default_tier={self.default_tier.value!r})"
        )

    def add_rule(self, rule: Rule) -> PermissionPolicy:
        """Append *rule* to the policy.

        Args:
            rule: The rule to add.  Appending a stricter rule is always safe:
                because evaluation takes the most restrictive match, a later
                rule can tighten an earlier verdict but never loosen it.

        Returns:
            ``self``, so rules can be chained onto a freshly built policy.
        """
        self.rules.append(rule)
        return self

    def matching_rules(self, call: ToolCall) -> list[Rule]:
        """Return every rule that applies to *call*, in declaration order.

        Args:
            call: The intercepted tool call.
        """
        return [rule for rule in self.rules if rule.matches(call)]

    def evaluate(self, call: ToolCall) -> Decision:
        """Classify *call* into a tier.

        Args:
            call: The intercepted tool call.

        Returns:
            A :class:`~agentlatch.harness._types.Decision` carrying the most
            restrictive matching tier, with ``rule_name`` and ``reason`` taken
            from the rule that won.  With no matches, the policy's
            ``default_tier`` is used and the reason says so.
        """
        matches = self.matching_rules(call)
        if not matches:
            return Decision(
                tier=self.default_tier,
                reason=(
                    f"No rule matched {call.name!r}; applied the policy default "
                    f"tier {self.default_tier.value!r} (fail closed)."
                ),
                rule_name=None,
            )

        winner = max(matches, key=lambda rule: rule.tier.restrictiveness)
        reason = winner.reason or f"Matched rule {winner.name!r}."
        if len(matches) > 1:
            others = ", ".join(r.name for r in matches if r is not winner)
            reason = f"{reason} (most restrictive of: {winner.name}, {others})"
        return Decision(tier=winner.tier, reason=reason, rule_name=winner.name)

    @classmethod
    def default(cls) -> PermissionPolicy:
        """Build the shipped starter policy.

        Tier 1 covers read-only tool-name globs, Tier 2 covers state-changing
        globs, and Tier 3 covers the destructive shell/SQL patterns in
        :data:`BLOCK_PATTERNS`.  Anything unmatched falls through to the
        ``HUMAN`` default.

        Returns:
            A new :class:`PermissionPolicy`.  Each call returns a fresh object,
            so mutating one policy never affects another.
        """
        rules: list[Rule] = [
            Rule(
                name="auto_read_only_tools",
                tier=PermissionTier.AUTO,
                tools=list(READ_ONLY_TOOL_GLOBS),
                reason="Read-only tool name; no state is mutated.",
            ),
            Rule(
                name="human_state_changing_tools",
                tier=PermissionTier.HUMAN,
                tools=list(STATE_CHANGING_TOOL_GLOBS),
                reason="Tool mutates state; a human must approve it.",
            ),
        ]
        rules += [
            Rule(
                name=name,
                tier=PermissionTier.BLOCKED,
                pattern=pattern,
                reason=reason,
            )
            for name, pattern, reason in BLOCK_PATTERNS
        ]
        return cls(rules, default_tier=PermissionTier.HUMAN)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class PermissionGate:
    """Applies a :class:`PermissionPolicy` and drives the Tier 2 human loop.

    The gate is the only object the rest of the harness talks to.  It answers
    one question — may this call run right now? — and records the answer.

    Args:
        policy:           Policy used to classify calls.
        on_approval:      Tier 2 hook.  Receives ``(call, decision)`` and
            returns ``True`` to approve.  ``None`` means no human is reachable,
            which denies every Tier 2 call rather than allowing it.
        approval_timeout: Seconds to wait for ``on_approval``.  ``None`` waits
            forever (appropriate for an interactive CLI, not for a service).
            The wait uses a ``ThreadPoolExecutor``; ``signal.alarm`` is banned
            in this repo because it is not cross-platform and does not work off
            the main thread.
        audit_log:        List to append audit entries to.  A fresh list is
            created when omitted.  Passing a shared list lets several gates
            write one trail.
    """

    def __init__(
        self,
        policy: PermissionPolicy,
        on_approval: ApprovalCallback | None = None,
        approval_timeout: float | None = None,
        audit_log: list[dict[str, Any]] | None = None,
    ) -> None:
        self.policy = policy
        self.on_approval = on_approval
        self.approval_timeout = approval_timeout
        self._audit_log: list[dict[str, Any]] = (
            audit_log if audit_log is not None else []
        )

    @property
    def audit_entries(self) -> list[dict[str, Any]]:
        """Read accessor for the audit trail (the live list, not a copy)."""
        return self._audit_log

    def check(self, call: ToolCall) -> Decision:
        """Decide whether *call* may execute, prompting a human if required.

        Tier 1 returns immediately.  Tier 3 returns immediately and never
        reaches ``on_approval`` — a blocked call is not negotiable.  Tier 2
        invokes the approval callback and records its verdict on
        ``decision.approved``.

        Every outcome, including denials and blocks, is appended to the audit
        log before returning.

        Args:
            call: The intercepted tool call.

        Returns:
            The :class:`~agentlatch.harness._types.Decision`.  Consult
            ``decision.may_execute`` rather than reading the tier directly.
        """
        decision = self.policy.evaluate(call)

        if decision.tier is PermissionTier.HUMAN:
            decision.approved = self._request_approval(call, decision)

        self._record(call, decision)
        return decision

    def _request_approval(self, call: ToolCall, decision: Decision) -> bool:
        """Run the Tier 2 approval hook, denying on every failure mode.

        Args:
            call:     The call awaiting approval.
            decision: The pending decision; its ``reason`` is extended with the
                failure cause when approval cannot be obtained.

        Returns:
            ``True`` only when the callback returned a truthy verdict in time.
        """
        if self.on_approval is None:
            decision.reason = (
                f"{decision.reason} Denied: tier 'human' requires approval but "
                "no on_approval callback is configured."
            )
            return False

        try:
            if self.approval_timeout is None:
                approved = self.on_approval(call, decision)
            else:
                pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="agentlatch-approval"
                )
                future = pool.submit(self.on_approval, call, decision)
                try:
                    approved = future.result(timeout=self.approval_timeout)
                except concurrent.futures.TimeoutError:
                    decision.reason = (
                        f"{decision.reason} Denied: no approval within "
                        f"{self.approval_timeout}s."
                    )
                    return False
                finally:
                    # Deliberately never joined: a hook that never returns must
                    # not hold the agent hostage past its own timeout.
                    pool.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:  # noqa: BLE001 - a broken hook must deny.
            decision.reason = (
                f"{decision.reason} Denied: approval callback raised "
                f"{type(exc).__name__}: {exc}"
            )
            return False

        if not approved:
            decision.reason = f"{decision.reason} Denied by approver."
            return False
        return True

    def _record(self, call: ToolCall, decision: Decision) -> None:
        """Append one structured entry describing *decision* to the audit log.

        Args:
            call:     The call that was checked.
            decision: The verdict returned to the caller.
        """
        self._audit_log.append(
            {
                "timestamp": time.time(),
                "tool": call.name,
                "call_id": call.id,
                "tier": decision.tier.value,
                "allowed": decision.may_execute,
                "approved": decision.approved,
                "rule_name": decision.rule_name,
                "reason": decision.reason,
            }
        )


# ---------------------------------------------------------------------------
# Approval callbacks
# ---------------------------------------------------------------------------


def cli_approval_callback(call: ToolCall, decision: Decision) -> bool:
    """Prompt the operator on stdin for a Tier 2 approval.

    Args:
        call:     The call awaiting approval; rendered with
            :meth:`~agentlatch.harness._types.ToolCall.describe`.
        decision: The pending decision, whose reason explains why approval is
            required.

    Returns:
        ``True`` only if the operator typed ``y``/``yes``.  Anything else — a
        bare Enter, EOF, or a non-TTY stdin — denies.  Auto-denying off-TTY is
        deliberate: in CI there is nobody to ask, so the choice is between
        hanging the pipeline and silently granting write access, and neither is
        acceptable.
    """
    if not (sys.stdin is not None and sys.stdin.isatty()):
        print(
            f"[agentlatch] Auto-denied {call.name}: approval required but "
            "stdin is not interactive.",
            file=sys.stderr,
        )
        return False

    print(f"\n[agentlatch] Approval required: {call.describe()}")
    if decision.reason:
        print(f"  reason: {decision.reason}")
    try:
        answer = input("  Allow this call? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("  -> denied")
        return False
    return answer in {"y", "yes"}
