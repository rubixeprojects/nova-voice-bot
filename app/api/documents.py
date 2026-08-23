"""Document management endpoints (spec §5 Documents)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db
from app.core.deps import get_current_user_id, get_request_id_dep
from app.core.logging import get_logger
from app.models import Document
from app.retrieval import bm25, dense
from app.schemas.documents import (
    DeleteResponse,
    DocumentListResponse,
    DocumentOut,
    DocumentRenameRequest,
    DocumentUploadResponse,
)
from app.workers.celery_app import celery_app

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def _get_owned_doc(
    db: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> Document:
    doc = (
        await db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document (async ingestion)",
)
async def upload_document(
    file: UploadFile = File(...),
    user_id: uuid.UUID = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id_dep),
    db: AsyncSession = Depends(get_db),
):
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.max_upload_mb} MB",
        )
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    document_id = uuid.uuid4()

    doc = Document(
        id=document_id,
        user_id=user_id,
        original_filename=file.filename or "document.pdf",
        display_name=file.filename or "document.pdf",
        raw_pdf=data,
        mime_type=file.content_type or "application/pdf",
        file_size_bytes=len(data),
        status="uploaded",
    )
    db.add(doc)
    await db.flush()

    # Enqueue async ingestion — propagate request_id for end-to-end tracing.
    celery_app.send_task(
        "ingest_document", args=[str(document_id), request_id]
    )
    log.info("document.uploaded", document_id=str(document_id), size=len(data))

    return DocumentUploadResponse(document_id=document_id, status="uploaded")


@router.get("", response_model=DocumentListResponse, summary="List documents")
async def list_documents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    base = select(Document).where(
        Document.user_id == user_id, Document.deleted_at.is_(None)
    )
    total = (
        await db.execute(
            select(func.count()).select_from(base.subquery())
        )
    ).scalar_one()
    rows = (
        await db.execute(
            base.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return DocumentListResponse(
        items=[DocumentOut.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentOut, summary="Get document detail")
async def get_document(
    document_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, document_id, user_id)
    return DocumentOut.model_validate(doc)


@router.patch(
    "/{document_id}", response_model=DocumentOut, summary="Rename a document"
)
async def rename_document(
    document_id: uuid.UUID,
    body: DocumentRenameRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, document_id, user_id)
    doc.display_name = body.name
    await db.flush()
    log.info("document.renamed", document_id=str(document_id))
    return DocumentOut.model_validate(doc)


@router.delete(
    "/{document_id}", response_model=DeleteResponse, summary="Delete a document"
)
async def delete_document(
    document_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    doc = await _get_owned_doc(db, document_id, user_id)

    # Purge from search backends (best-effort; log but don't fail the delete).
    try:
        dense.delete_by_document(document_id)
    except Exception:  # noqa: BLE001
        log.exception("document.delete.qdrant_failed", document_id=str(document_id))
    try:
        bm25.delete_by_document(document_id)
    except Exception:  # noqa: BLE001
        log.exception("document.delete.opensearch_failed", document_id=str(document_id))

    await db.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(deleted_at=func.now())
    )
    log.info("document.deleted", document_id=str(document_id))
    return DeleteResponse(id=document_id, deleted=True)
