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
from fastapi.middleware.cors import CORSMiddleware
configure_logging()
log = get_logger(__name__)


from app.ingestion.embedding import _embedding_backend, _load_model
from app.retrieval.rerank import (
    _use_hf_api as _reranker_uses_hf,
    _is_endpoint as _reranker_is_endpoint,
    _load as _load_reranker,
)

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

    # Pre-load local models ONLY when local is the primary backend. With HF
    # inference (HF_TOKEN) or a custom HTTP endpoint configured, skip the
    # ~4.5 GB local warm standby — it lazy-loads on demand if a remote call
    # ever fails. Keeps RAM sane on small hosts.
    if _embedding_backend() == "local":
        try:
            _load_model()
        except Exception:  # noqa: BLE001
            log.warning("app.bootstrap_failed", component="bge_m3_local")

    if not _reranker_uses_hf() and not _reranker_is_endpoint():
        try:
            _load_reranker()
        except Exception:  # noqa: BLE001
            log.warning("app.bootstrap_failed", component="bge_reranker_local")

    yield
    log.info("app.shutdown")


app = FastAPI(
    title="Assamese RAG API",
    version="0.1.0",
    description=(
        "Retrieval-Augmented Generation over Assamese documents with text & "
        "voice chat.\n\n"
        "**User Scope:** All requests use the universal university user ID "
        "configured in `UNIVERSAL_USER_ID`.\n\n"
        "**Conversation:** Each conversation is separated by its "
        "`conversation_id`.\n\n"
        "**Swagger:** No `X-User-Id` or `X-Request-Id` headers are required."
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
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


@app.get("/voice", include_in_schema=False)
async def voice_client():
    """Combined chat + voice test client (talks to this API + ws://<host>:8766)."""
    return FileResponse("app/static/voice_client.html")