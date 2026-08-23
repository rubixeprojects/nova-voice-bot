"""Ingestion pipeline Celery task — delegates to app.pipeline.ingest."""
from __future__ import annotations

import uuid

from sqlalchemy import select

from app.core.clients import ensure_opensearch_index, ensure_qdrant_collection
from app.core.config import settings
from app.core.context import new_job_request_id, set_request_id, set_user_id
from app.core.db import get_sync_db
from app.core.logging import get_logger
from app.core.stage_logger import stage
from app.ingestion.embedding import embed_texts, release_tokenizer
from app.models import Chunk, Document
from app.pipeline.ingest import run_ingestion_pipeline
from app.retrieval import bm25, dense
from app.workers.celery_app import celery_app

log = get_logger(__name__)


def _set_status(db, doc: Document, status: str, error: str | None = None) -> None:
    doc.status = status
    if error is not None:
        doc.error_message = error
    db.add(doc)
    db.flush()
    log.info("ingest.status", document_id=str(doc.id), status=status)


def _chunk_payload(doc: Document, doc_uuid: uuid.UUID, draft, cid, prev_id, next_id, source_file: str) -> dict:
    return {
        "chunk_id": str(cid),
        "document_id": str(doc_uuid),
        "user_id": str(doc.user_id),
        "chunk_index": draft.chunk_index,
        "section_id": str(draft.section_id) if draft.section_id else None,
        "section_title": draft.section_title,
        "heading_path": draft.heading_path,
        "page_number": draft.page_start,
        "page_start": draft.page_start,
        "page_end": draft.page_end,
        "block_type": draft.block_type,
        "document_type": doc.document_type,
        "quality_score": draft.quality_score,
        "prev_chunk_id": str(prev_id) if prev_id else None,
        "next_chunk_id": str(next_id) if next_id else None,
        "token_count": draft.token_count,
        "source_language": draft.language,
        "ocr_confidence": draft.ocr_confidence,
        "source_file": source_file,
        "text": draft.text,
    }


@celery_app.task(name="ingest_document", bind=True, max_retries=0)
def ingest_document(
    self, document_id: str, request_id: str | None = None
) -> dict:
    job_request_id = request_id or new_job_request_id()
    set_request_id(job_request_id)
    doc_uuid = uuid.UUID(document_id)

    ensure_qdrant_collection()
    ensure_opensearch_index()

    with get_sync_db() as db:
        doc = db.execute(
            select(Document).where(Document.id == doc_uuid)
        ).scalar_one_or_none()
        if doc is None:
            log.error("ingest.doc_not_found", document_id=document_id)
            return {"status": "not_found"}
        set_user_id(str(doc.user_id))

        try:
            raw = doc.raw_pdf

            _set_status(db, doc, "ocr_in_progress")
            with stage(db, "parse", component="document_router", document_id=doc_uuid,
                       request_id=job_request_id) as st:
                st.input({"bytes": len(raw)})
                artifacts = run_ingestion_pipeline(raw)
                st.output({
                    "document_type": artifacts.profile.document_type,
                    "pages": len(artifacts.pages),
                    "sections": len(artifacts.sections),
                    "chunks": len(artifacts.chunks),
                })

            doc.page_count = artifacts.profile.page_count or len(artifacts.pages)
            doc.document_type = artifacts.profile.document_type
            doc.language = artifacts.detected_language
            _set_status(db, doc, "structuring")

            drafts = artifacts.chunks
            if not drafts:
                _set_status(db, doc, "failed", error="No embeddable text extracted")
                return {"status": "failed", "reason": "empty"}

            chunk_ids = [uuid.uuid4() for _ in drafts]

            _set_status(db, doc, "embedding")
            release_tokenizer()
            embed_component = "hf_inference" if settings.hf_token else "bge_m3"
            with stage(db, "embedding", component=embed_component, document_id=doc_uuid,
                       request_id=job_request_id) as st:
                st.input({
                    "chunks": len(drafts),
                    "batch_size": settings.embedding_batch_size,
                    "parallel": settings.embedding_parallel_workers,
                })
                vectors = embed_texts([d.text for d in drafts])
                st.output({"vectors": len(vectors), "dim": len(vectors[0]) if vectors else 0})

            qdrant_points: list[dict] = []
            os_docs: list[dict] = []
            source_file = doc.original_filename

            for i, (draft, cid, vec) in enumerate(zip(drafts, chunk_ids, vectors)):
                prev_id = chunk_ids[i - 1] if i > 0 else None
                next_id = chunk_ids[i + 1] if i < len(chunk_ids) - 1 else None
                payload = _chunk_payload(doc, doc_uuid, draft, cid, prev_id, next_id, source_file)

                db.add(Chunk(
                    id=cid,
                    document_id=doc_uuid,
                    user_id=doc.user_id,
                    chunk_index=draft.chunk_index,
                    section_id=draft.section_id,
                    section_title=draft.section_title,
                    page_number=draft.page_start,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    heading_path=draft.heading_path,
                    block_type=draft.block_type,
                    quality_score=draft.quality_score,
                    language=draft.language,
                    text=draft.text,
                    token_count=draft.token_count,
                    prev_chunk_id=prev_id,
                    next_chunk_id=next_id,
                    ocr_confidence=draft.ocr_confidence,
                    qdrant_point_id=cid,
                    opensearch_doc_id=str(cid),
                ))

                qdrant_points.append({"id": str(cid), "vector": vec, "payload": payload})
                os_docs.append({"_id": str(cid), **{k: v for k, v in payload.items() if k != "chunk_id"}, "chunk_id": str(cid)})

            db.flush()

            with stage(db, "index_write", component="qdrant", document_id=doc_uuid,
                       request_id=job_request_id) as st:
                dense.upsert_chunks(qdrant_points)
                st.output({"points": len(qdrant_points)})

            with stage(db, "index_write", component="opensearch",
                       document_id=doc_uuid, request_id=job_request_id) as st:
                bm25.index_chunks(os_docs)
                st.output({"docs": len(os_docs)})

            _set_status(db, doc, "ready")
            log.info("ingest.done", document_id=document_id, chunks=len(drafts))
            return {"status": "ready", "chunks": len(drafts)}

        except Exception as exc:  # noqa: BLE001
            log.exception("ingest.failed", document_id=document_id)
            try:
                _set_status(db, doc, "failed", error=str(exc)[:2000])
            except Exception:  # noqa: BLE001
                log.exception("ingest.status_update_failed", document_id=document_id)
            return {"status": "failed", "error": str(exc)}
