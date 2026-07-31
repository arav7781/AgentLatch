"""Tests for agentlatch.harness.compaction — budgeting and disclosure.

No mocks: every test drives the real ``Compactor`` / ``ToolRegistry`` against
real payloads, and the "summarize" tests inject a plain Python function.
"""

from __future__ import annotations

import json

import pytest

from agentlatch.harness._types import PermissionTier, ToolDescriptor
from agentlatch.harness.compaction import (
    COMPACTION_KEY,
    DISCLOSURE_TOOL_NAME,
    CompactionResult,
    Compactor,
    ToolRegistry,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _big_rows(n: int = 500) -> list[dict]:
    return [
        {"id": i, "name": f"User {i}", "email": f"u{i}@example.com"} for i in range(n)
    ]


def _catalog() -> list[ToolDescriptor]:
    """A realistic dozen-tool catalog with non-trivial schemas."""
    tools = []
    for i in range(12):
        tools.append(
            ToolDescriptor(
                name=f"tool_{i}",
                summary=f"Does useful thing number {i}.",
                schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search query to run against the backing "
                                "store. Supports boolean operators and quoted "
                                "phrases."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum rows to return.",
                            "default": 25,
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "Include per-row provenance data.",
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
                tier=PermissionTier.AUTO,
                tags=["search", "read-only"],
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_monotonic_in_input_size(self):
        """Bigger input never estimates fewer tokens."""
        sizes = [0, 10, 100, 1_000, 10_000, 100_000]
        estimates = [estimate_tokens("x" * n) for n in sizes]
        assert estimates == sorted(estimates)
        assert estimates[-1] > estimates[0]

    def test_monotonic_for_structures(self):
        """Growing a list grows the estimate."""
        previous = 0
        for n in (1, 5, 50, 500):
            current = estimate_tokens(_big_rows(n))
            assert current > previous
            previous = current

    def test_uses_four_chars_per_token_heuristic(self):
        """Estimation follows the repo-wide ~4-chars-per-token ratio."""
        assert estimate_tokens("x" * 400) == 100

    def test_handles_non_serializable_objects(self):
        """Exotic objects fall back to str() rather than raising."""
        assert estimate_tokens(object()) > 0


# ---------------------------------------------------------------------------
# Passthrough
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_small_payload_untouched(self):
        """A payload under budget comes back byte-identical, compacted False."""
        payload = {"status": "ok", "rows": [1, 2, 3]}
        result = Compactor(max_tokens=2048).compact(payload)

        assert isinstance(result, CompactionResult)
        assert result.content == payload
        assert result.content is payload
        assert result.compacted is False
        assert result.strategy_used == "none"
        assert result.notes == []
        assert result.final_tokens == result.original_tokens

    def test_small_string_untouched(self):
        text = "All good."
        result = Compactor(max_tokens=100).compact(text)
        assert result.content == text
        assert result.compacted is False
        assert COMPACTION_KEY not in result.content


# ---------------------------------------------------------------------------
# Sample strategy
# ---------------------------------------------------------------------------


class TestSampleStrategy:
    def test_oversized_string_lands_under_budget(self):
        """A huge text blob is truncated below the token ceiling."""
        text = "y" * 80_000
        compactor = Compactor(max_tokens=100)
        result = compactor.compact(text)

        assert result.compacted is True
        assert result.strategy_used == "sample"
        assert result.final_tokens <= 100
        assert result.original_tokens > 100
        assert len(result.content) < len(text)
        assert result.tokens_saved > 0

    def test_string_is_annotated(self):
        result = Compactor(max_tokens=64).compact("z" * 40_000)
        assert COMPACTION_KEY in result.content
        assert "strategy=sample" in result.content

    def test_list_of_dicts_sampled_to_sample_rows(self):
        """A big row list is cut to sample_rows with metadata appended."""
        result = Compactor(max_tokens=1024, sample_rows=50).compact(_big_rows(500))

        content = result.content
        assert isinstance(content, list)
        assert result.compacted is True

        # 50 real rows, the sampler's metadata row, then the compaction marker.
        assert content[0] == {
            "id": 0,
            "name": "User 0",
            "email": "u0@example.com",
        }
        sampled_meta = content[50]
        assert sampled_meta["_agentlatch_sampled"] is True
        assert sampled_meta["shown"] == 50
        assert sampled_meta["total"] == 500

        assert COMPACTION_KEY in content[-1]
        marker = content[-1][COMPACTION_KEY]
        assert marker["strategy"] == "sample"
        assert marker["original_tokens"] > marker["final_tokens"]

    def test_rows_tightened_when_sample_rows_still_too_big(self):
        """The row cap shrinks further when 50 rows still blow the budget."""
        result = Compactor(max_tokens=64, sample_rows=50).compact(_big_rows(500))
        rows = [r for r in result.content if "id" in r]
        assert 0 < len(rows) < 50

    def test_nested_dict_stays_a_dict(self):
        """Structured payloads are never stringified just to be truncated."""
        payload = {
            "query": "SELECT * FROM users",
            "meta": {"engine": "postgres", "ms": 12.5},
            "rows": _big_rows(400),
        }
        result = Compactor(max_tokens=512).compact(payload)

        content = result.content
        assert isinstance(content, dict)
        assert content["query"] == "SELECT * FROM users"
        assert isinstance(content["meta"], dict)
        assert content["meta"]["engine"] == "postgres"
        assert isinstance(content["rows"], list)
        assert content[COMPACTION_KEY]["strategy"] == "sample"

    def test_json_string_payload_is_sampled(self):
        """A JSON *string* goes through the sampler's parse/sample/reserialize."""
        raw = json.dumps({"rows": _big_rows(400)})
        result = Compactor(max_tokens=256).compact(raw)

        assert isinstance(result.content, str)
        assert result.compacted is True
        assert result.final_tokens <= 256


# ---------------------------------------------------------------------------
# head_tail strategy
# ---------------------------------------------------------------------------


class TestHeadTailStrategy:
    def test_keeps_both_ends_of_text(self):
        """The point of head_tail: the error at the end of a log survives."""
        text = "STARTOFLOG " + ("filler line\n" * 20_000) + " FATALERRORHERE"
        result = Compactor(max_tokens=200, strategy="head_tail").compact(text)

        assert result.compacted is True
        assert result.strategy_used == "head_tail"
        assert "STARTOFLOG" in result.content
        assert "FATALERRORHERE" in result.content
        assert "elided" in result.content
        assert result.final_tokens <= 200

    def test_head_only_truncation_would_lose_the_tail(self):
        """Contrast case: the default sample strategy drops the tail."""
        text = "STARTOFLOG " + ("filler line\n" * 20_000) + " FATALERRORHERE"
        sampled = Compactor(max_tokens=200).compact(text)
        assert "STARTOFLOG" in sampled.content
        assert "FATALERRORHERE" not in sampled.content

    def test_keeps_both_ends_of_a_list(self):
        rows = _big_rows(500)
        result = Compactor(max_tokens=512, strategy="head_tail").compact(rows)

        ids = [r["id"] for r in result.content if "id" in r]
        assert 0 in ids
        assert 499 in ids

    def test_short_text_is_not_elided(self):
        result = Compactor(max_tokens=5000, strategy="head_tail").compact("hi")
        assert result.content == "hi"
        assert result.compacted is False


# ---------------------------------------------------------------------------
# summarize strategy
# ---------------------------------------------------------------------------


class TestSummarizeStrategy:
    def test_uses_injected_summarizer(self):
        calls: list[int] = []

        def summarizer(text: str, max_tokens: int) -> str:
            calls.append(max_tokens)
            return f"SUMMARY of {len(text)} chars: the job failed."

        result = Compactor(
            max_tokens=128, strategy="summarize", summarizer=summarizer
        ).compact("q" * 60_000)

        assert result.strategy_used == "summarize"
        assert result.compacted is True
        assert "SUMMARY of 60000 chars" in result.content
        assert result.final_tokens <= 128
        assert calls and calls[0] > 0

    def test_missing_summarizer_falls_back_to_sample(self):
        """No summarizer must degrade, not raise — a run never dies here."""
        result = Compactor(max_tokens=100, strategy="summarize").compact("q" * 40_000)

        assert result.strategy_used == "sample"
        assert result.compacted is True
        assert any("summarizer" in note for note in result.notes)
        assert any("sample" in note for note in result.notes)

    def test_raising_summarizer_falls_back_to_sample(self):
        def boom(text: str, max_tokens: int) -> str:
            raise RuntimeError("model unavailable")

        result = Compactor(
            max_tokens=100, strategy="summarize", summarizer=boom
        ).compact("q" * 40_000)

        assert result.strategy_used == "sample"
        assert any("RuntimeError" in note for note in result.notes)

    def test_unknown_strategy_falls_back_to_sample(self):
        result = Compactor(max_tokens=100, strategy="teleport").compact("q" * 40_000)
        assert result.strategy_used == "sample"
        assert any("teleport" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


class TestIdempotence:
    def test_string_markers_do_not_stack(self):
        compactor = Compactor(max_tokens=100)
        once = compactor.compact("w" * 40_000)
        twice = compactor.compact(once.content)

        assert twice.content == once.content
        assert twice.content.count(COMPACTION_KEY) == 1
        assert twice.strategy_used == "none"

    def test_dict_markers_do_not_stack(self):
        compactor = Compactor(max_tokens=256)
        once = compactor.compact({"rows": _big_rows(500)})
        twice = compactor.compact(once.content)

        assert twice.content is once.content
        assert list(twice.content).count(COMPACTION_KEY) == 1

    def test_list_markers_do_not_stack(self):
        compactor = Compactor(max_tokens=256)
        once = compactor.compact(_big_rows(500))
        twice = compactor.compact(once.content)

        markers = [
            item
            for item in twice.content
            if isinstance(item, dict) and COMPACTION_KEY in item
        ]
        assert len(markers) == 1

    def test_repeated_compaction_is_stable(self):
        compactor = Compactor(max_tokens=100)
        content = "e" * 40_000
        for _ in range(4):
            content = compactor.compact(content).content
        assert content.count(COMPACTION_KEY) == 1


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_brief_catalog_excludes_schemas(self):
        registry = ToolRegistry(_catalog())
        catalog = registry.brief_catalog()

        assert len(catalog) == 12
        for entry in catalog:
            assert set(entry) == {"name", "summary"}
        assert "properties" not in json.dumps(catalog)

    def test_register_and_register_many(self):
        registry = ToolRegistry()
        assert len(registry) == 0

        registry.register(ToolDescriptor(name="solo", summary="One tool."))
        assert "solo" in registry
        registry.register_many(_catalog())
        assert len(registry) == 13
        assert registry.get("tool_3").summary == "Does useful thing number 3."
        assert registry.get("nope") is None

    def test_system_prompt_block(self):
        registry = ToolRegistry(_catalog())
        block = registry.system_prompt_block()

        assert "tool_0: Does useful thing number 0." in block
        assert DISCLOSURE_TOOL_NAME in block
        assert "properties" not in block
        assert block.count("\n") >= 12

    def test_system_prompt_block_when_empty(self):
        assert "No tools" in ToolRegistry().system_prompt_block()

    def test_disclose_returns_full_schema(self):
        registry = ToolRegistry(_catalog())
        revealed = registry.disclose("tool_4")

        assert revealed["status"] == "success"
        assert revealed["name"] == "tool_4"
        assert revealed["schema"]["properties"]["query"]["type"] == "string"
        assert revealed["tier"] == "auto"
        assert revealed["tags"] == ["search", "read-only"]
        assert registry.disclosed == {"tool_4"}

    def test_disclose_unknown_returns_error_with_suggestions(self):
        registry = ToolRegistry(_catalog())
        error = registry.disclose("tool_")

        assert error["status"] == "error"
        assert error["error_type"] == "unknown_tool"
        assert error["suggestions"]
        assert all(s.startswith("tool_") for s in error["suggestions"])
        assert "Did you mean" in error["message"]
        assert error["available"] == registry.names
        assert "tool_" not in registry.disclosed

    def test_disclose_unknown_never_raises(self):
        registry = ToolRegistry(_catalog())
        for bad in ("", "!!!", "a" * 300, "TOOL_1"):
            payload = registry.disclose(bad)
            assert payload["status"] == "error"
            assert isinstance(payload["suggestions"], list)

    def test_disclose_many(self):
        registry = ToolRegistry(_catalog())
        payloads = registry.disclose_many(["tool_1", "ghost", "tool_2"])

        assert set(payloads) == {"tool_1", "ghost", "tool_2"}
        assert payloads["tool_1"]["status"] == "success"
        assert payloads["ghost"]["status"] == "error"
        assert registry.disclosed == {"tool_1", "tool_2"}

    def test_token_savings_positive_for_realistic_catalog(self):
        registry = ToolRegistry(_catalog())
        assert registry.token_savings() > 0

    def test_token_savings_shrinks_as_tools_are_disclosed(self):
        registry = ToolRegistry(_catalog())
        before = registry.token_savings()
        registry.disclose_many(["tool_0", "tool_1", "tool_2"])
        after = registry.token_savings()

        assert after < before
        assert after >= 0

    def test_token_savings_never_negative(self):
        registry = ToolRegistry([ToolDescriptor(name="t", summary="s")])
        registry.disclose("t")
        assert registry.token_savings() >= 0
        assert ToolRegistry().token_savings() == 0

    def test_as_disclosure_tool_shape(self):
        registry = ToolRegistry(_catalog())
        tool = registry.as_disclosure_tool()

        assert tool["name"] == DISCLOSURE_TOOL_NAME
        assert isinstance(tool["description"], str) and tool["description"]

        params = tool["parameters"]
        assert params["type"] == "object"
        assert params["required"] == ["name"]
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["name"]["enum"] == registry.names
        # Must be plain JSON so any framework adapter can reshape it.
        assert json.loads(json.dumps(tool)) == tool

    def test_disclosure_tool_round_trip(self):
        """End to end: pick a name off the enum, disclose it, get a schema."""
        registry = ToolRegistry(_catalog())
        chosen = registry.as_disclosure_tool()["parameters"]["properties"]["name"][
            "enum"
        ][2]
        assert registry.disclose(chosen)["status"] == "success"


# ---------------------------------------------------------------------------
# Compactor + registry interplay
# ---------------------------------------------------------------------------


class TestCompactedDisclosure:
    def test_error_payloads_survive_compaction(self):
        """A disclosure error is small enough to reach the model intact."""
        registry = ToolRegistry(_catalog())
        error = registry.disclose("nope")
        result = Compactor(max_tokens=2048).compact(error)

        assert result.compacted is False
        assert result.content["error_type"] == "unknown_tool"


@pytest.mark.parametrize("strategy", ["sample", "head_tail", "summarize"])
def test_every_strategy_annotates_and_bounds(strategy):
    """Whatever the strategy, output is annotated and near the budget."""
    compactor = Compactor(
        max_tokens=120,
        strategy=strategy,
        summarizer=lambda text, mt: "short summary",
    )
    result = compactor.compact("p" * 50_000)

    assert result.compacted is True
    assert COMPACTION_KEY in result.content
    assert result.final_tokens <= 120
