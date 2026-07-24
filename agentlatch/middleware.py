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
from agentlatch.config import is_dev_mode
from agentlatch.tracker import TraceEvent, finalize_trace, get_trace, init_trace

# Default body size limit (10 MiB) to prevent memory exhaustion when
# buffering responses for JSON profile injection.
_DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024


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
        expose_headers:  If ``None`` (default), headers are injected only
                         in dev mode.  Set to ``True`` / ``False`` to
                         force the behaviour regardless of environment.
        max_body_size:   Maximum response body size (in bytes) to buffer
                         when injecting the profile into JSON bodies.
                         Responses larger than this are left untouched
                         (headers are still injected if enabled).
                         Defaults to 10 MiB.
    """

    def __init__(
        self,
        app: Any,
        *,
        inject_profile: bool = True,
        header_prefix: str = "X-AgentLatch",
        trace_name: str = "AgentRequest",
        expose_headers: bool | None = None,
        max_body_size: int = _DEFAULT_MAX_BODY_BYTES,
    ) -> None:
        super().__init__(app)
        self.inject_profile = inject_profile
        self.header_prefix = header_prefix
        self.trace_name = trace_name
        self.expose_headers = expose_headers
        self.max_body_size = max_body_size

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

        # Decide whether to expose headers.  When expose_headers is None
        # (the default), fall back to dev-mode detection so production
        # deployments don't accidentally leak version/timing info.
        should_expose = (
            self.expose_headers
            if self.expose_headers is not None
            else is_dev_mode()
        )

        # --- Inject headers (when enabled) ---
        if should_expose:
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
                # Read the existing body, respecting the size limit.
                body_bytes = b""
                exceeded_limit = False
                async for chunk in response.body_iterator:
                    if isinstance(chunk, str):
                        body_bytes += chunk.encode("utf-8")
                    else:
                        body_bytes += chunk
                    if len(body_bytes) > self.max_body_size:
                        exceeded_limit = True
                        break

                if not exceeded_limit:
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
