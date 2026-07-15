"""Starlette / FastAPI HTTP middleware for AgentLatch.

Wraps every request with an AgentLatch trace so that tool calls made during
the request are automatically profiled.  The execution profile is injected
into response **headers** (always) and optionally into the JSON **body**.

This is the *word-of-mouth engine*: when a developer tests their ``/chat``
endpoint in Postman, they see the full AgentLatch profile — tool timings,
error counts, and the execution timeline — right in the response.

Usage::

    from fastapi import FastAPI
    from agentlatch.middleware import AgentLatchMiddleware

    app = FastAPI()
    app.add_middleware(AgentLatchMiddleware)

Requires ``starlette``.  Install via::

    pip install agentlatch[server]
"""

from __future__ import annotations

import json
import uuid
from typing import Any

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response
except ImportError as exc:
    raise ImportError(
        "AgentLatchMiddleware requires Starlette (or FastAPI).  "
        "Install it with:  pip install agentlatch[server]"
    ) from exc

from agentlatch import __version__
from agentlatch._types import EventStatus
from agentlatch.tracker import TraceEvent, finalize_trace, get_trace, init_trace


def _trace_to_dict(trace: TraceEvent) -> dict[str, Any]:
    """Serialize a TraceEvent tree into a JSON-safe dict."""
    total_ms = round(trace.duration * 1000)
    tool_ms = round(sum(c.duration for c in trace.children) * 1000)
    llm_ms = max(0, total_ms - tool_ms)
    errors = sum(1 for c in trace.children if c.status == EventStatus.ERROR)

    tools: list[dict[str, Any]] = []
    for child in trace.children:
        entry: dict[str, Any] = {
            "name": child.name,
            "duration_ms": round(child.duration * 1000),
            "status": child.status.value,
        }
        if child.error_payload:
            entry["error"] = child.error_payload.get("message", "")
        tools.append(entry)

    return {
        "version": __version__,
        "trace_id": str(uuid.uuid4()),
        "total_ms": total_ms,
        "tool_ms": tool_ms,
        "llm_reasoning_ms": llm_ms,
        "tools": tools,
        "errors_count": errors,
    }


class AgentLatchMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that injects AgentLatch execution profiles.

    Args:
        app:             The ASGI application.
        inject_profile:  If ``True``, append an ``_agentlatch`` key to JSON
                         response bodies.  Headers are always injected.
        header_prefix:   Prefix for response headers (default ``X-AgentLatch``).
        trace_name:      Label for the root trace event.
    """

    def __init__(
        self,
        app: Any,
        *,
        inject_profile: bool = True,
        header_prefix: str = "X-AgentLatch",
        trace_name: str = "AgentRequest",
    ) -> None:
        super().__init__(app)
        self.inject_profile = inject_profile
        self.header_prefix = header_prefix
        self.trace_name = trace_name

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Wrap the request with an AgentLatch trace."""
        # Initialize trace for this request.
        init_trace(self.trace_name)

        # Process the request.
        response: Response = await call_next(request)

        # Finalize and extract profile.
        trace = get_trace()
        if trace is None:
            return response

        trace = finalize_trace()
        profile = _trace_to_dict(trace)

        # --- Inject headers (always) ---
        pfx = self.header_prefix
        response.headers[f"{pfx}-Version"] = __version__
        response.headers[f"{pfx}-Trace-Id"] = profile["trace_id"]
        response.headers[f"{pfx}-Duration-Ms"] = str(profile["total_ms"])
        response.headers[f"{pfx}-Tools-Ms"] = str(profile["tool_ms"])
        response.headers[f"{pfx}-Errors"] = str(profile["errors_count"])

        # --- Inject body (opt-in) ---
        if self.inject_profile:
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                # Read the existing body.
                body_bytes = b""
                async for chunk in response.body_iterator:
                    if isinstance(chunk, str):
                        body_bytes += chunk.encode("utf-8")
                    else:
                        body_bytes += chunk

                try:
                    body = json.loads(body_bytes)
                    if isinstance(body, dict):
                        body["_agentlatch"] = profile
                        new_body = json.dumps(body, ensure_ascii=False)
                        return Response(
                            content=new_body,
                            status_code=response.status_code,
                            headers={
                                k: v
                                for k, v in response.headers.items()
                                if k.lower() != "content-length"
                            },
                            media_type="application/json",
                        )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Not valid JSON — fall through, headers still set.

        return response
