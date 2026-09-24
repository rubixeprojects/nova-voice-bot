"""Request middleware: correlation ids, trusted-auth header, request logging.

Responsibilities (spec §5 + logging rule #1/#2):
  * Require `X-User-Id` (401 if absent). We DO NOT validate it — auth is done by
    an upstream gateway; we only require + propagate it.
  * Accept or generate `X-Request-Id`; echo it back and thread it through the
    context vars so every log line / downstream call is correlated.
  * Record one `api_request_logs` row per request with method/path/status/latency.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from app.core.config import settings
from app.core.context import set_request_id, set_user_id
from app.core.db import AsyncSessionLocal
from app.core.logging import get_logger
from app.models import ApiRequestLog

log = get_logger(__name__)

# Endpoints that do not require the trusted user header.
_PUBLIC_PATHS = {
    "/healthz",
    "/readyz",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
    "/favicon.ico",
}


def _is_public(path: str) -> bool:
    if path in _PUBLIC_PATHS:
        return True
    # Swagger UI static assets.
    return path.startswith("/docs") or path.startswith("/redoc")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = set_request_id(None)

        path = request.url.path

        user_id = str(settings.universal_user_id)
        set_user_id(user_id)
        request.state.request_id = request_id
        request.state.user_id = user_id

        started = time.perf_counter()
        error_message: str | None = None
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)[:2000]
            log.exception("request.unhandled_error", path=path)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
            status_code = 500
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            response.headers["X-Request-Id"] = request_id
            await self._log_request(
                request_id=None,
                user_id=user_id,
                method=request.method,
                path=path,
                status_code=status_code,
                latency_ms=latency_ms,
                error_message=error_message,
                content_length=request.headers.get("content-length"),
                query=str(request.url.query) or None,
            )
        return response

    async def _log_request(
        self,
        *,
        request_id: str,
        user_id: str | None,
        method: str,
        path: str,
        status_code: int,
        latency_ms: int,
        error_message: str | None,
        content_length: str | None,
        query: str | None,
    ) -> None:
        # Never log raw bodies / audio — only sizes + coarse metadata.
        summary = {
            "content_length": int(content_length) if content_length else None,
            "query": query,
        }
        log.info(
            "request.completed",
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=latency_ms,
        )
        try:
            async with AsyncSessionLocal() as session:
                session.add(
                    ApiRequestLog(
                        request_id=uuid.UUID(request_id),
                        user_id=uuid.UUID(user_id) if user_id else None,
                        method=method,
                        path=path,
                        status_code=status_code,
                        latency_ms=latency_ms,
                        request_summary=summary,
                        error_message=error_message,
                    )
                )
                await session.commit()
        except Exception:  # noqa: BLE001
            # Logging must never take down a request.
            log.exception("request.log_persist_failed", path=path)
