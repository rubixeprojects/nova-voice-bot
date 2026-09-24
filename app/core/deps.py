"""FastAPI dependencies for extracting the trusted identity/correlation ids.

The middleware already enforces presence; these just surface typed values to
handlers and keep OpenAPI/Swagger documenting the required headers.
"""
from __future__ import annotations
import uuid
from app.core.config import settings
from fastapi import HTTPException


async def get_current_user_id() -> uuid.UUID:
    try:
        return uuid.UUID(settings.universal_user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail="UNIVERSAL_USER_ID must be a valid UUID",
        ) from exc


