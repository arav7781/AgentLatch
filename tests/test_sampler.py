"""Tests for agentlatch.sampler — response compression engine."""

from __future__ import annotations

import json

from agentlatch.sampler import sample_response


class TestPassthrough:
    def test_none_params_passthrough(self):
        """If no limits are set, the response passes through unchanged."""
        data = '{"rows": [1, 2, 3]}'
        assert sample_response(data) == data

    def test_small_string_passthrough(self):
        """Small responses below all limits pass through unchanged."""
        data = '{"name": "Alice"}'
        result = sample_response(data, max_tokens=1000, sample_rows=50)
        assert result == data

    def test_non_json_string_passthrough(self):
        """Plain text strings that aren't JSON pass through unchanged."""
        data = "Hello, this is a plain text tool response."
        result = sample_response(data, sample_rows=5)
        assert result == data

    def test_non_string_passthrough(self):
        """Dict/list objects pass through when under limits."""
        data = {"name": "Alice", "age": 30}
        result = sample_response(data, sample_rows=50)
        assert result == data


class TestListSampling:
    def test_root_list_sampling(self):
        """A root-level JSON list gets sliced with metadata appended."""
        items = list(range(100))
        raw = json.dumps(items)
        result = json.loads(sample_response(raw, sample_rows=5))

        assert len(result) == 6  # 5 items + 1 metadata
        assert result[:5] == [0, 1, 2, 3, 4]
        assert result[5]["_agentlatch_sampled"] is True
        assert result[5]["shown"] == 5
        assert result[5]["total"] == 100

    def test_dict_rows_key_sampling(self):
        """A dict with 'rows' key gets its list sampled."""
        raw = json.dumps({"rows": list(range(50)), "count": 50})
        result = json.loads(sample_response(raw, sample_rows=3))

        assert len(result["rows"]) == 4  # 3 items + metadata
        assert result["rows"][3]["_agentlatch_sampled"] is True
        assert result["rows"][3]["total"] == 50
        assert result["count"] == 50  # other keys untouched

    def test_dict_results_key_sampling(self):
        """Works with 'results' key too."""
        data = {"results": [{"id": i} for i in range(20)]}
        raw = json.dumps(data)
        result = json.loads(sample_response(raw, sample_rows=2))

        assert len(result["results"]) == 3  # 2 items + metadata
        assert result["results"][0]["id"] == 0
        assert result["results"][1]["id"] == 1

    def test_dict_data_key_sampling(self):
        """Works with 'data' key too."""
        data = {"data": list(range(10))}
        raw = json.dumps(data)
        result = json.loads(sample_response(raw, sample_rows=2))

        assert len(result["data"]) == 3  # 2 + metadata

    def test_list_under_limit_unchanged(self):
        """Lists smaller than sample_rows pass through unchanged."""
        raw = json.dumps([1, 2, 3])
        result = sample_response(raw, sample_rows=10)
        assert json.loads(result) == [1, 2, 3]

    def test_non_string_list_sampling(self):
        """Non-string list objects get sampled directly."""
        data = list(range(100))
        result = sample_response(data, sample_rows=5)

        assert len(result) == 6
        assert result[5]["_agentlatch_sampled"] is True


class TestTokenTruncation:
    def test_long_string_truncation(self):
        """Strings exceeding max_tokens get truncated with marker."""
        # 100 tokens * 4 chars = 400 char limit
        long_text = "A" * 1000
        result = sample_response(long_text, max_tokens=100)

        assert len(result) < 1000
        assert result.startswith("A" * 400)
        assert "truncated" in result
        assert "600" in result  # 1000 - 400 = 600 remaining

    def test_json_truncation(self):
        """JSON responses exceeding max_tokens get truncated after re-serialization."""
        rows = [{"id": i, "name": f"User {i}"} for i in range(1000)]
        raw = json.dumps({"rows": rows})
        result = sample_response(raw, max_tokens=50)

        assert len(result) <= 250  # 50 * 4 + marker length
        assert "truncated" in result

    def test_under_limit_no_truncation(self):
        """Strings under the limit don't get truncated."""
        data = "Short response"
        result = sample_response(data, max_tokens=1000)
        assert result == data


class TestCombined:
    def test_sampling_then_truncation(self):
        """Both sampling and truncation applied together."""
        rows = [{"id": i, "data": "x" * 100} for i in range(1000)]
        raw = json.dumps({"rows": rows})

        result = sample_response(raw, max_tokens=100, sample_rows=3)

        assert "truncated" in result or len(result) <= 400
        # The rows should have been sampled first
        # Then the serialized result truncated if still too long
