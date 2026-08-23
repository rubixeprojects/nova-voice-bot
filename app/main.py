"""FastAPI application entrypoint.

Wires routers, middleware, structured logging, and best-effort bootstrap of the
Qdrant collection / OpenSearch index / S3 bucket on startup. Swagger UI is
served at /docs, ReDoc at /redoc, OpenAPI JSON at /openapi.json.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse

from app.api import chat, conversations, documents, health
from app.core.clients import ensure_opensearch_index, ensure_qdrant_collection
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("app.startup", env=settings.app_env)
    # Best-effort: don't crash the API if a backing store is briefly down.
    for name, fn in (
        ("qdrant", ensure_qdrant_collection),
        ("opensearch", ensure_opensearch_index),
    ):
        try:
            fn()
        except Exception:  # noqa: BLE001
            log.warning("app.bootstrap_failed", component=name)
    yield
    log.info("app.shutdown")


app = FastAPI(
    title="Assamese RAG API",
    version="0.1.0",
    description=(
        "Retrieval-Augmented Generation over Assamese documents with text & "
        "voice chat.\n\n"
        "**Authentication:** All `/api/v1/*` endpoints require the trusted "
        "`X-User-Id` header (UUID) set by the upstream gateway. Ops/docs "
        "endpoints are public.\n\n"
        "**Correlation:** Optional `X-Request-Id` header (UUID). Generated if "
        "absent and echoed on every response.\n\n"
        "**Swagger:** Click **Authorize**, enter a test user UUID, then try "
        "endpoints from this page."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "persistAuthorization": True,
        "displayRequestDuration": True,
        "filter": True,
        "tryItOutEnabled": True,
        "docExpansion": "list",
    },
    openapi_tags=[
        {"name": "documents", "description": "Upload, list, rename, delete PDFs"},
        {"name": "chat", "description": "Text/voice chat over indexed documents"},
        {"name": "conversations", "description": "Conversation history management"},
        {"name": "ops", "description": "Liveness and readiness probes"},
    ],
)

app.add_middleware(RequestContextMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(conversations.router)


_PUBLIC_OPENAPI_PATHS = {
    "/",
    "/healthz",
    "/readyz",
    "/openapi.json",
}


def custom_openapi():
    """Document auth/correlation headers in Swagger; exempt public routes."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["UserIdHeader"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-User-Id",
        "description": (
            "Trusted user UUID from the upstream auth gateway. "
            "Example: 550e8400-e29b-41d4-a716-446655440000"
        ),
    }
    security_schemes["RequestIdHeader"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Request-Id",
        "description": "Optional request correlation UUID (generated if omitted).",
    }
    schema["security"] = [{"UserIdHeader": []}, {"RequestIdHeader": []}]

    for path, path_item in schema.get("paths", {}).items():
        if path in _PUBLIC_OPENAPI_PATHS:
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation["security"] = []

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "assamese-rag", "docs": "/docs", "openapi": "/openapi.json"}
@app.get("/stt-test")
async def stt_test():
    return FileResponse("app/static/stt_test.html")