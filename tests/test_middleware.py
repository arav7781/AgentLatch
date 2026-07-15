"""Tests for agentlatch.middleware — HTTP middleware for Postman visibility."""

from __future__ import annotations

import time

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from agentlatch.config import reset_dev_mode, set_dev_mode
from agentlatch.decorators import safe_tool
from agentlatch.middleware import AgentLatchMiddleware

# ---------------------------------------------------------------------------
# Test fixtures: build a tiny Starlette app with the middleware
# ---------------------------------------------------------------------------


@safe_tool
def mock_tool(query: str) -> str:
    """A tool that simulates a DB call."""
    time.sleep(0.01)
    if "bad" in query:
        raise ValueError("column not found")
    return '{"rows": [{"id": 1}]}'


def chat_endpoint(request: Request) -> JSONResponse:
    """Simulate an agent processing a /chat request."""
    # The middleware has already initialized the trace.
    # Simulate tool calls inside the request.
    result1 = mock_tool("bad query")
    result2 = mock_tool("good query")
    return JSONResponse({"response": "Done", "tool_results": [result1, result2]})


def plain_endpoint(request: Request) -> PlainTextResponse:
    """A non-JSON endpoint."""
    mock_tool("good query")
    return PlainTextResponse("OK")


def _build_app(*, inject_profile: bool = True) -> Starlette:
    app = Starlette(
        routes=[
            Route("/chat", chat_endpoint),
            Route("/plain", plain_endpoint),
        ],
    )
    app.add_middleware(
        AgentLatchMiddleware,
        inject_profile=inject_profile,
        trace_name="TestAgent",
    )
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMiddlewareHeaders:
    def setup_method(self):
        set_dev_mode(False)  # Suppress ASCII visuals in tests.

    def teardown_method(self):
        reset_dev_mode()

    def test_headers_present(self):
        """Response must include X-AgentLatch-* headers."""
        client = TestClient(_build_app())
        resp = client.get("/chat")

        assert resp.status_code == 200
        assert "X-AgentLatch-Version" in resp.headers
        assert "X-AgentLatch-Trace-Id" in resp.headers
        assert "X-AgentLatch-Duration-Ms" in resp.headers
        assert "X-AgentLatch-Tools-Ms" in resp.headers
        assert "X-AgentLatch-Errors" in resp.headers

    def test_header_values_correct(self):
        """Header values must reflect actual execution."""
        client = TestClient(_build_app())
        resp = client.get("/chat")

        assert resp.headers["X-AgentLatch-Version"] == "0.1.0"
        assert int(resp.headers["X-AgentLatch-Duration-Ms"]) > 0
        assert int(resp.headers["X-AgentLatch-Errors"]) >= 1  # bad query fails

    def test_plain_response_headers_only(self):
        """Non-JSON responses get headers but no body injection."""
        client = TestClient(_build_app())
        resp = client.get("/plain")

        assert resp.status_code == 200
        assert "X-AgentLatch-Version" in resp.headers
        assert resp.text == "OK"  # Body unchanged


class TestMiddlewareBody:
    def setup_method(self):
        set_dev_mode(False)

    def teardown_method(self):
        reset_dev_mode()

    def test_profile_injected_in_body(self):
        """JSON responses must include _agentlatch key when inject_profile=True."""
        client = TestClient(_build_app(inject_profile=True))
        resp = client.get("/chat")
        body = resp.json()

        assert "_agentlatch" in body
        assert "response" in body  # original body preserved

    def test_profile_contains_tool_data(self):
        """The injected profile must contain tool names, durations, statuses."""
        client = TestClient(_build_app(inject_profile=True))
        resp = client.get("/chat")
        profile = resp.json()["_agentlatch"]

        assert "tools" in profile
        assert len(profile["tools"]) >= 2
        tool_names = [t["name"] for t in profile["tools"]]
        assert "mock_tool" in tool_names

        # At least one error from the "bad query"
        assert profile["errors_count"] >= 1
        assert profile["total_ms"] > 0
        assert profile["tool_ms"] > 0

    def test_profile_has_version_and_trace_id(self):
        """Profile must include version and trace_id."""
        client = TestClient(_build_app(inject_profile=True))
        resp = client.get("/chat")
        profile = resp.json()["_agentlatch"]

        assert profile["version"] == "0.1.0"
        assert len(profile["trace_id"]) > 0

    def test_disabled_injection(self):
        """inject_profile=False suppresses body injection but keeps headers."""
        client = TestClient(_build_app(inject_profile=False))
        resp = client.get("/chat")
        body = resp.json()

        assert "_agentlatch" not in body
        assert "X-AgentLatch-Version" in resp.headers
