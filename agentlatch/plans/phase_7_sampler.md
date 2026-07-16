# Phase 7 — Response Sampling & Truncation

> Feature: **Response Sampling** (see [`README.md`](../../README.md) — Smart Response Sampling).
> Status: **✅ Done** · Depends on: None.
> Written to document the LLM context compression engine.

---

## 1. Goal
Intelligently compress large tool returns (like database responses or API payloads containing thousands of lines) before passing them back to the LLM. This prevents token window bloat, reduces inference latencies, and lowers hallucination risks by pruning repetitive structures while preserving context.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P7-1** | Target lists under common keys | Automatically inspect list values under standard data containers: `rows`, `results`, `data`, `items`, `records`, `entries`. |
| **D-P7-2** | Inplace metadata injection | Append a clear trailing tracking payload block indicating the truncation size: `{"_agentlatch_sampled": True, "shown": limit, "total": total}`. |
| **D-P7-3** | Approximate token sizing | Convert token requirements to rough character ceilings using a conservative character-to-token ratio of `4` (`_CHARS_PER_TOKEN = 4`). |
| **D-P7-4** | Format preservation | If the input is a JSON string, deserialize it, perform list pruning, and re-serialize it. If it is plain text or an object structure, process and return the matching type. |

## 3. Implementation
- **`agentlatch/sampler.py`:**
  - `_try_parse_json(raw)`: Parses text inputs to see if they contain valid JSON.
  - `_sample_list(data, limit)`: Slices standard Python list structures and appends sampling metadata if they exceed constraints.
  - `_walk_and_sample(obj, sample_rows)`: Recursively traverses JSON structures to identify list keys.
  - `_truncate_string(text, max_chars)`: Truncates long text blocks and appends a descriptive marker indicating the remaining length.
  - `sample_response(raw, max_tokens, sample_rows)`: Main entrypoint mapping row constraints and length limits.

## 4. Architecture
```
    [ Raw Tool Response ]
              │
      ┌───────┴───────┐
      ▼               ▼
[ JSON String ]  [ Python Object ]
      │               │
  Deserialize       Walk and slice lists
      │               │
      ├───────────────┘
      ▼
[ Truncate to Max Chars ] ──> [ Sampled Output Response ]
```

## 5. Safety, Isolation, & Correctness
- Safely recovers from malformed JSON and JSON parsing errors, degrading gracefully to raw string truncation.
- Preserves object structures and ensures that metadata injected in JSON lists does not conflict with original list typing.

## 6. Tests
Implemented in [`tests/test_sampler.py`](../../tests/test_sampler.py):
- Validation of list slicing under target keys (e.g. `items`, `rows`).
- Serialization check for plain text and raw dictionaries.
- Verification of token truncation markers.
- Verification that payloads within standard sizing criteria remain completely untouched.

## 7. Files Touched
| File | Change |
|---|---|
| [`agentlatch/sampler.py`](../../agentlatch/sampler.py) | **[NEW]** Sampling algorithms, key lists, and truncation helper functions. |

## 8. Acceptance Criteria
- Databases output lists are pruned correctly to the limit, appending a valid `_agentlatch_sampled` tracking dictionary.
- Extremely long response strings are truncated at the approximate token budget, containing clear remaining characters count details.
