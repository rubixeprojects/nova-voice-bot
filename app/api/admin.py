"""Admin endpoints: view/set the admin-curated list of allowed languages."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db
from app.core.logging import get_logger
from app.core.system_settings import (
    ALL_LANGUAGES,
    get_allowed_languages,
    set_allowed_languages,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class SetLanguagesRequest(BaseModel):
    languages: list[str]


def _check_admin_password(x_admin_password: str | None) -> None:
    if not settings.admin_password:
        raise HTTPException(
            status_code=503, detail="Admin password not configured on server"
        )
    if x_admin_password != settings.admin_password:
        raise HTTPException(status_code=401, detail="Invalid admin password")


@router.get("/languages", summary="Get the admin-allowed language list")
async def get_languages(db: AsyncSession = Depends(get_db)):
    return {"allowed_languages": await get_allowed_languages(db)}


@router.post("/languages", summary="Set the admin-allowed language list (admin only)")
async def post_languages(
    body: SetLanguagesRequest,
    x_admin_password: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    _check_admin_password(x_admin_password)

    invalid = [c for c in body.languages if c not in ALL_LANGUAGES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown language code(s): {invalid}. Valid codes: {sorted(ALL_LANGUAGES)}",
        )

    try:
        saved = await set_allowed_languages(db, body.languages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info("admin.allowed_languages_changed", languages=saved)
    return {"allowed_languages": saved}
