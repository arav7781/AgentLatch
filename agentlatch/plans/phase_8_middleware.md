# Phase 8 — HTTP Server Middleware

> Feature: **HTTP Observability Middleware** (see [`README.md`](../README.md) — FastAPI / Starlette HTTP Middleware).
> Status: **✅ Done** · Depends on: **Phase 2** (timing engine) and **Phase 1** (core types).
> Written to document Starlette / FastAPI middleware integration.

---

## 1. Goal
Provide immediate visibility into agent execution states directly in REST response bodies and headers during local testing (e.g. inside Postman, Insomnia, or Curl). Create a lightweight Starlette-based ASGI middleware that wraps request lifecycles, collects timing data of downstream tasks, and injects structured traces.

## 2. Locked Decisions

| # | Decision | Rationale |
|---|---|---|
| **D-P8-1** | Header Injection | Always inject summary metadata using `X-AgentLatch-` headers, including execution duration, tool time, and exception statuses. |
| **D-P8-2** | JSON Body Injection | Dynamically inspect response `Content-Type`. If `application/json` is detected, append a detailed `_agentlatch` dictionary at the root of the JSON response block. |
| **D-P8-3** | Non-blocking Error Fallback | If JSON decoding or payload parsing fails, do not crash the request lifecycle; simply return the response body intact, ensuring HTTP headers are still set. |
| **D-P8-4** | Optional dependencies setup | Ensure that Starlette imports are lazy or wrapped with explicit errors to avoid forcing non-web developers to install web servers. |

## 3. Implementation
- **`agentlatch/middleware.py`:**
  - `_trace_to_dict(trace)`: Converts `TraceEvent` structures into formatted JSON objects outlining tool breakdowns, timing parameters, and error counts.
  - `AgentLatchMiddleware`: Implements Starlette `BaseHTTPMiddleware`.
    - Initializes the tracing context before passing the request to the endpoint handler via `call_next`.
    - Extracts trace information from context, sets HTTP response headers, and updates the response body.
- **Dependency Guard:** Raises descriptive `ImportError` if `starlette` is not present, prompting users to run `pip install agentlatch[server]`.

## 4. HTTP Request Pipeline
```
[ Incoming HTTP Request ]
           │
           ▼
[ init_trace("AgentRequest") ]
           │
           ▼
   [ call_next(request) ]  ──────> Runs Endpoint logic & Tool Calls
           │
           ▼
[ finalize_trace() & get_trace() ]
           │
           ▼
[ Inject X-AgentLatch Headers ]
           │
           ▼
[ Inject _agentlatch into JSON ]
           │
           ▼
[ Return Response to Client ]
```

## 5. Safety, Isolation, & Correctness
- Resolves content length headers dynamically by removing the original `Content-Length` header on payload modification, letting the server handle dynamic chunk encoding correctly.
- Prevents memory leaks by ensuring the trace lifecycle is finalized and context variable references are released.

## 6. Tests
Implemented in [`tests/test_middleware.py`](../tests/test_middleware.py):
- Validation of header injection on standard JSON endpoints.
- Integration tests using Starlette `TestClient`.
- Verification of body injection on success and tool failure states.
- Assertion that non-JSON responses are left completely unaltered.

## 7. Files Touched
| File | Change |
|---|---|
| [`agentlatch/middleware.py`](../agentlatch/middleware.py) | **[NEW]** Starlette BaseHTTPMiddleware implementation. |

## 8. Acceptance Criteria
- Response headers include:
  - `X-AgentLatch-Version`
  - `X-AgentLatch-Trace-Id`
  - `X-AgentLatch-Duration-Ms`
  - `X-AgentLatch-Tools-Ms`
  - `X-AgentLatch-Errors`
- JSON responses contain a root `_agentlatch` tracing context block.
