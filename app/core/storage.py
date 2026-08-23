"""Stub — raw PDF bytes are stored directly in PostgreSQL (documents.raw_pdf)."""
from __future__ import annotations


def delete_object(key: str) -> None:
    s3 = _client()
    s3.delete_object(Bucket=settings.s3_bucket, Key=key)
    log.info("s3.delete", key=key)
