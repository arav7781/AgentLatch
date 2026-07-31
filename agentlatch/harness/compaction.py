"""Context compaction and progressive tool disclosure for the harness.

Two things blow up an agent's context window, and neither is the agent's
fault:

1. **Tool output.**  A single ``SELECT *`` or a verbose stack trace can be
   larger than everything else in the conversation combined.  :class:`Compactor`
   caps every tool result at a token budget *before* it re-enters the context.
2. **Tool schemas.**  A catalog of thirty tools with full JSON-Schema parameter
   blocks costs thousands of tokens in the system prompt on turn one, most of
   them for tools the agent will never call.  :class:`ToolRegistry` loads only
   ``name`` + ``summary`` up front and hands out full schemas on demand.

Neither class raises for ordinary failure.  A compaction bug must never be the
reason an agent run dies, and an LLM guessing a tool name is expected
behaviour, not an exception.
"""

from __future__ import annotations

import difflib
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from agentlatch.harness._types import ToolDescriptor
from agentlatch.sampler import _CHARS_PER_TOKEN, sample_response

# Marker key added to compacted dicts / lists, mirroring the
# ``_agentlatch_sampled`` convention in ``agentlatch.sampler``.
COMPACTION_KEY = "_agentlatch_compacted"

# Text form of the same marker, appended to compacted strings.
_STRING_MARKER_PREFIX = f"[{COMPACTION_KEY}:"

# Headroom reserved out of the budget for the annotation itself, so that an
# annotated result still lands *under* ``max_tokens`` rather than one marker
# over it.
_ANNOTATION_TOKEN_RESERVE = 32

# Share of the head_tail budget spent on the head.  The remainder is the tail.
_HEAD_SHARE = 0.6

_VALID_STRATEGIES = ("sample", "head_tail", "summarize")

# Name of the tool an agent calls to pull a full schema on demand.
DISCLOSURE_TOOL_NAME = "get_tool_schema"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _serialize(payload: Any) -> str:
    """Render *payload* as the text an LLM would actually receive."""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(payload)


def estimate_tokens(payload: Any) -> int:
    """Estimate the token cost of *payload*.

    This is a **rough heuristic, not a tokenizer**.  It reuses the repo-wide
    ~4-characters-per-token ratio (``agentlatch.sampler._CHARS_PER_TOKEN``) and
    will be wrong — often by 20% or more — for code, non-Latin scripts, or
    dense JSON punctuation.  It is deliberately cheap: compaction runs on every
    tool call, and paying for a real tokenizer there would cost more than the
    tokens it saves.  Treat the result as a budget guardrail, never as a
    billing figure.

    Args:
        payload: Any value.  Strings are measured directly; everything else is
            JSON-serialized first (falling back to ``str`` for exotic objects).

    Returns:
        Estimated token count, rounded up.  Never negative.
    """
    return math.ceil(len(_serialize(payload)) / _CHARS_PER_TOKEN)


def _char_len(payload: Any) -> int:
    """Serialized character length of *payload*."""
    return len(_serialize(payload))


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CompactionResult:
    """Outcome of one compaction pass.

    Attributes:
        content:         The payload to hand back to the agent.  Same Python
                         type as the input whenever the input was structured.
        original_tokens: Estimated tokens before compaction.
        final_tokens:    Estimated tokens after compaction, annotation
                         included.
        compacted:       Whether anything was removed.  ``False`` means
                         *content is byte-identical to the input*.
        strategy_used:   ``"sample"``, ``"head_tail"``, ``"summarize"``, or
                         ``"none"`` when the payload fit as-is.
        notes:           Human-readable trail of fallbacks and clamps.  Empty
                         on the happy path.
    """

    content: Any
    original_tokens: int = 0
    final_tokens: int = 0
    compacted: bool = False
    strategy_used: str = "none"
    notes: list[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        """Estimated tokens kept out of the context window."""
        return max(0, self.original_tokens - self.final_tokens)


# ---------------------------------------------------------------------------
# Structure helpers
# ---------------------------------------------------------------------------


def _is_annotated(payload: Any) -> bool:
    """Whether *payload* already carries an AgentLatch compaction marker."""
    if isinstance(payload, str):
        return _STRING_MARKER_PREFIX in payload
    if isinstance(payload, dict):
        return COMPACTION_KEY in payload
    if isinstance(payload, list):
        return any(
            isinstance(item, dict) and COMPACTION_KEY in item for item in payload
        )
    return False


def _map_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    """Rebuild *obj*, applying *fn* to every string leaf."""
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_strings(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_map_strings(v, fn) for v in obj)
    return obj


def _truncate_text(text: str, max_chars: int) -> str:
    """Keep the head of *text* only, with a count of what was dropped."""
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    return text[:max_chars] + f"...[truncated — {dropped:,} chars remaining]"


def _head_tail_text(text: str, max_chars: int) -> str:
    """Keep both ends of *text*, eliding the middle.

    Args:
        text:      The text to shrink.
        max_chars: Total character budget, marker included.

    Returns:
        ``head + elision marker + tail``, or *text* unchanged if it already
        fits.
    """
    if len(text) <= max_chars:
        return text

    # Size the marker against the worst case so the result cannot exceed
    # ``max_chars`` once the real (smaller) figure is substituted in.
    marker_budget = len(_elision_marker(len(text)))
    body = max(20, max_chars - marker_budget)
    head_chars = max(1, int(body * _HEAD_SHARE))
    tail_chars = max(1, body - head_chars)

    dropped = len(text) - head_chars - tail_chars
    if dropped <= 0:
        return text
    return text[:head_chars] + _elision_marker(dropped) + text[-tail_chars:]


def _elision_marker(dropped: int) -> str:
    return f"\n...[elided — {dropped:,} chars omitted from the middle]...\n"


def _head_tail_list(data: list, limit: int) -> list:
    """Keep the first ~60% and last ~40% of *limit* elements of *data*."""
    if len(data) <= limit:
        return data
    head_n = max(1, int(limit * _HEAD_SHARE))
    tail_n = max(1, limit - head_n)
    return [
        *data[:head_n],
        {
            "_agentlatch_sampled": True,
            "shown": head_n + tail_n,
            "total": len(data),
            "note": f"head {head_n} + tail {tail_n}; middle elided",
        },
        *data[-tail_n:],
    ]


def _walk_head_tail(obj: Any, limit: int) -> Any:
    """Head/tail-sample lists in *obj*, mirroring ``sampler._walk_and_sample``."""
    if isinstance(obj, list):
        return _head_tail_list(obj, limit)
    if isinstance(obj, dict):
        return {
            k: (_head_tail_list(v, limit) if isinstance(v, list) else v)
            for k, v in obj.items()
        }
    return obj


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


class Compactor:
    """Caps tool output so a single call cannot overflow the context window.

    The compactor is the last stage of the harness pipeline
    (``PermissionGate -> Sandbox -> Compactor -> ToolResult``).  Everything it
    emits is annotated, because an LLM that cannot tell a truncated result from
    a complete one will confidently report the truncated view as the whole
    truth.

    Args:
        max_tokens:  Approximate token ceiling for a single tool result.
        sample_rows: Row cap handed to :func:`agentlatch.sampler.sample_response`.
            ``None`` disables row sampling and leaves only text-level shrinking.
        strategy:    ``"sample"`` (default), ``"head_tail"``, or ``"summarize"``.
            An unrecognised value falls back to ``"sample"`` and is recorded in
            :attr:`CompactionResult.notes`.
        summarizer:  Required by the ``"summarize"`` strategy.  Called as
            ``summarizer(text, max_tokens)`` and expected to return text.  If it
            is missing or raises, compaction falls back to ``"sample"`` rather
            than failing the tool call.
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        sample_rows: int | None = 50,
        strategy: str = "sample",
        summarizer: Callable[[str, int], str] | None = None,
    ) -> None:
        self.max_tokens = max(1, int(max_tokens))
        self.sample_rows = sample_rows
        self.strategy = strategy
        self.summarizer = summarizer

    # -- public ------------------------------------------------------------

    def estimate_tokens(self, payload: Any) -> int:
        """Rough token estimate for *payload* (see :func:`estimate_tokens`)."""
        return estimate_tokens(payload)

    def compact(self, payload: Any) -> CompactionResult:
        """Shrink *payload* to fit the configured token budget.

        Args:
            payload: Raw tool output — a string, a dict, a list, or any
                JSON-serializable structure.  Structured input stays
                structured: a dict is never stringified just to be truncated,
                its inner lists and long string leaves are shrunk instead.

        Returns:
            A :class:`CompactionResult`.  When the payload already fits,
            ``content`` is the untouched input and ``compacted`` is ``False``.
        """
        original_tokens = estimate_tokens(payload)
        notes: list[str] = []

        # Already carries a marker — re-annotating would stack markers and
        # mislead the model about how much was dropped.
        if _is_annotated(payload):
            return CompactionResult(
                content=payload,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                compacted=True,
                strategy_used="none",
                notes=["payload already compacted; left unchanged"],
            )

        if original_tokens <= self.max_tokens:
            return CompactionResult(
                content=payload,
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                compacted=False,
                strategy_used="none",
            )

        strategy = self.strategy
        if strategy not in _VALID_STRATEGIES:
            notes.append(f"unknown strategy {strategy!r}; fell back to 'sample'")
            strategy = "sample"

        if strategy == "summarize":
            content, strategy = self._run_summarize(payload, notes)
        else:
            content = None

        if content is None:
            content = self._shrink(payload, strategy)

        content = self._annotate(content, original_tokens, strategy)
        final_tokens = estimate_tokens(content)
        if final_tokens > self.max_tokens:
            notes.append(
                f"result still ~{final_tokens} tokens after compaction "
                f"(budget {self.max_tokens}); payload resisted shrinking"
            )

        return CompactionResult(
            content=content,
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            compacted=True,
            strategy_used=strategy,
            notes=notes,
        )

    # -- strategies --------------------------------------------------------

    @property
    def _budget_chars(self) -> int:
        """Character budget, minus headroom for the annotation."""
        usable = max(16, self.max_tokens - _ANNOTATION_TOKEN_RESERVE)
        return usable * _CHARS_PER_TOKEN

    def _run_summarize(self, payload: Any, notes: list[str]) -> tuple[Any | None, str]:
        """Try the injected summarizer; report a fallback instead of raising."""
        if self.summarizer is None:
            notes.append(
                "strategy='summarize' requested but no summarizer was "
                "provided; fell back to 'sample'"
            )
            return None, "sample"

        budget = max(1, self.max_tokens - _ANNOTATION_TOKEN_RESERVE)
        try:
            summary = self.summarizer(_serialize(payload), budget)
        except Exception as exc:  # noqa: BLE001 — compaction must never crash a run.
            notes.append(
                f"summarizer raised {type(exc).__name__}: {exc}; fell back to 'sample'"
            )
            return None, "sample"

        if not isinstance(summary, str):
            summary = _serialize(summary)
        # A summarizer can overshoot its budget; the budget is the contract.
        return _truncate_text(summary, self._budget_chars), "summarize"

    def _shrink(self, payload: Any, strategy: str) -> Any:
        """Apply ``sample`` / ``head_tail`` shrinking to *payload*."""
        budget = self._budget_chars

        if isinstance(payload, str):
            if strategy == "head_tail":
                return _head_tail_text(payload, budget)
            # ``sample_response`` already parses JSON strings, samples the
            # lists inside them, re-serializes, and truncates — reuse it whole.
            return sample_response(
                payload,
                max_tokens=max(1, self.max_tokens - _ANNOTATION_TOKEN_RESERVE),
                sample_rows=self.sample_rows,
            )

        if not isinstance(payload, (dict, list)):
            return _truncate_text(_serialize(payload), budget)

        return self._fit_structured(payload, budget, strategy)

    def _fit_structured(self, payload: Any, budget: int, strategy: str) -> Any:
        """Shrink a dict/list *in place as a structure*.

        ``sample_response`` samples rows but does not enforce a token ceiling
        on non-string input, so the row cap is tightened until the serialized
        form fits.  Only if that is not enough are long string leaves trimmed —
        the shape of the data is the last thing to go, because the shape is
        what lets the agent write correct follow-up code.
        """
        rows = self.sample_rows
        candidate = self._sample_rows(payload, rows, strategy)

        while rows is not None and rows > 1 and _char_len(candidate) > budget:
            rows = max(1, rows // 2)
            candidate = self._sample_rows(payload, rows, strategy)

        if _char_len(candidate) > budget:
            candidate = self._trim_leaves(candidate, budget, strategy)
        return candidate

    def _sample_rows(self, payload: Any, rows: int | None, strategy: str) -> Any:
        """One row-sampling pass over structured *payload*."""
        if rows is None:
            return payload
        if strategy == "head_tail":
            return _walk_head_tail(payload, rows)
        return sample_response(payload, sample_rows=rows)

    def _trim_leaves(self, candidate: Any, budget: int, strategy: str) -> Any:
        """Halve the per-string cap until the structure fits the budget."""
        trim = _head_tail_text if strategy == "head_tail" else _truncate_text
        cap = budget
        result = candidate
        while cap > 16:
            cap //= 2
            result = _map_strings(candidate, lambda s, c=cap: trim(s, c))
            if _char_len(result) <= budget:
                break
        return result

    # -- annotation --------------------------------------------------------

    def _annotate(self, content: Any, original_tokens: int, strategy: str) -> Any:
        """Stamp *content* so the model knows it is reading a subset.

        Mirrors the ``_agentlatch_sampled`` shape from
        :mod:`agentlatch.sampler`: a metadata dict for structured payloads, a
        trailing text marker for strings.
        """
        meta = {
            "original_tokens": original_tokens,
            "final_tokens": estimate_tokens(content),
            "strategy": strategy,
        }

        if isinstance(content, dict):
            return {**content, COMPACTION_KEY: meta}
        if isinstance(content, list):
            return [*content, {COMPACTION_KEY: meta}]

        text = content if isinstance(content, str) else _serialize(content)
        return (
            f"{text}\n{_STRING_MARKER_PREFIX} strategy={strategy}, "
            f"original_tokens={original_tokens}, "
            f"final_tokens={meta['final_tokens']}]"
        )


# ---------------------------------------------------------------------------
# Progressive tool disclosure
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Holds tool descriptors and reveals full schemas only on demand.

    The system prompt gets :meth:`system_prompt_block` — one line per tool,
    name and summary.  When the model decides it wants a tool it calls the
    disclosure tool (:meth:`as_disclosure_tool`) and gets the full parameter
    schema back for that one tool.  For a catalog of any size this is the
    difference between paying for every schema on every turn and paying for the
    two or three the agent actually uses.

    Args:
        descriptors: Optional initial descriptors to register.
    """

    def __init__(self, descriptors: Iterable[ToolDescriptor] | None = None) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        self.disclosed: set[str] = set()
        if descriptors:
            self.register_many(descriptors)

    # -- registration ------------------------------------------------------

    def register(self, descriptor: ToolDescriptor) -> ToolDescriptor:
        """Add (or replace) one descriptor.

        Args:
            descriptor: The tool metadata to hold.

        Returns:
            The registered descriptor, for chaining.
        """
        self._tools[descriptor.name] = descriptor
        return descriptor

    def register_many(self, descriptors: Iterable[ToolDescriptor]) -> None:
        """Register every descriptor in *descriptors*."""
        for descriptor in descriptors:
            self.register(descriptor)

    @property
    def names(self) -> list[str]:
        """Registered tool names, in registration order."""
        return list(self._tools)

    def get(self, name: str) -> ToolDescriptor | None:
        """Return the descriptor for *name*, or ``None``."""
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # -- brief layer -------------------------------------------------------

    def brief_catalog(self) -> list[dict[str, str]]:
        """Name + summary for every tool — and nothing else.

        Returns:
            One ``{"name": ..., "summary": ...}`` dict per tool.  Schemas are
            deliberately absent; this is the payload that ships in the initial
            system prompt.
        """
        return [descriptor.brief() for descriptor in self._tools.values()]

    def system_prompt_block(self) -> str:
        """Render the brief catalog as a drop-in system-prompt section.

        Returns:
            A compact text block listing every tool as ``- name: summary``,
            followed by one line telling the agent how to fetch a full schema.
        """
        if not self._tools:
            return "No tools are available."

        lines = ["Available tools (summaries only):"]
        lines += [
            f"- {d.name}: {d.summary}"
            + (f" [tier: {d.tier.value}]" if d.tier is not None else "")
            for d in self._tools.values()
        ]
        lines.append(
            f"Call `{DISCLOSURE_TOOL_NAME}(name=...)` to get the full parameter "
            "schema for a tool before you use it."
        )
        return "\n".join(lines)

    # -- disclosure --------------------------------------------------------

    def disclose(self, name: str) -> dict[str, Any]:
        """Return the full schema for one tool.

        Never raises.  A model guessing at a tool name is ordinary behaviour,
        so an unknown name comes back as a structured error the model can read
        and correct from — the same contract ``@safe_tool`` uses.

        Args:
            name: The tool name the agent asked for.

        Returns:
            On success: ``{"status": "success", "name", "summary", "schema",
            "tier", "tags"}``.  On failure: ``{"status": "error",
            "error_type": "unknown_tool", "message", "suggestions",
            "available", "instruction"}``.
        """
        descriptor = self._tools.get(name)
        if descriptor is None:
            return self._unknown_tool(name)

        self.disclosed.add(name)
        return {
            "status": "success",
            "name": descriptor.name,
            "summary": descriptor.summary,
            "schema": descriptor.schema or {},
            "tier": descriptor.tier.value if descriptor.tier else None,
            "tags": list(descriptor.tags),
        }

    def disclose_many(self, names: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Disclose several tools at once.

        Args:
            names: Tool names to reveal.

        Returns:
            A mapping of requested name to its :meth:`disclose` payload —
            including error payloads for names that do not exist.
        """
        return {name: self.disclose(name) for name in names}

    def _unknown_tool(self, name: str) -> dict[str, Any]:
        """Structured "no such tool" payload with spelling suggestions."""
        suggestions = difflib.get_close_matches(name, self.names, n=3, cutoff=0.5)
        message = f"No tool named {name!r} is registered."
        if suggestions:
            message += f" Did you mean: {', '.join(suggestions)}?"
        return {
            "status": "error",
            "error_type": "unknown_tool",
            "message": message,
            "requested": name,
            "suggestions": suggestions,
            "available": self.names,
            "instruction": (
                "Pick a tool name from 'available' (or 'suggestions') and call "
                "the disclosure tool again."
            ),
        }

    # -- accounting --------------------------------------------------------

    def token_savings(self) -> int:
        """Estimated tokens saved versus loading every full schema up front.

        Computed from the actual descriptors: the cost of the full catalog,
        minus what progressive disclosure really spent — the brief catalog, the
        disclosure tool's own schema, and the full schemas of the tools the
        agent has disclosed so far.

        Returns:
            Estimated tokens saved.  Clamped at zero: with a tiny catalog, or
            once nearly everything has been disclosed, the disclosure machinery
            can cost more than it saves, and reporting a negative saving as a
            win would be dishonest.
        """
        if not self._tools:
            return 0

        upfront = estimate_tokens([self._full_form(d) for d in self._tools.values()])
        spent = estimate_tokens(self.brief_catalog())
        spent += estimate_tokens(self.as_disclosure_tool())
        spent += sum(
            estimate_tokens(self._full_form(self._tools[n]))
            for n in self.disclosed
            if n in self._tools
        )
        return max(0, upfront - spent)

    @staticmethod
    def _full_form(descriptor: ToolDescriptor) -> dict[str, Any]:
        """The payload a non-progressive framework would load up front."""
        return {
            "name": descriptor.name,
            "description": descriptor.summary,
            "parameters": descriptor.schema or {},
        }

    # -- the disclosure tool itself ----------------------------------------

    def as_disclosure_tool(self) -> dict[str, Any]:
        """Return a plain tool schema the agent framework can register.

        This is what closes the loop: the model sees summaries, sees this tool,
        and pulls the schemas it needs itself.  The returned dict is
        framework-agnostic (``name`` / ``description`` / ``parameters`` with a
        JSON-Schema object), so an adapter can reshape it for OpenAI, Anthropic,
        or LangChain without the registry knowing which.

        Returns:
            A tool-schema dict.  The ``name`` parameter carries an ``enum`` of
            the registered tool names, which keeps the model from inventing
            one.
        """
        return {
            "name": DISCLOSURE_TOOL_NAME,
            "description": (
                "Get the full parameter schema for one tool. The system prompt "
                "lists only tool names and summaries; call this before using a "
                "tool so you know its exact arguments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the tool to describe.",
                        "enum": self.names,
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        }


__all__ = [
    "COMPACTION_KEY",
    "DISCLOSURE_TOOL_NAME",
    "CompactionResult",
    "Compactor",
    "ToolRegistry",
    "estimate_tokens",
]
