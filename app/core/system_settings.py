"""Shared helper: read the admin-configured list of allowed languages, and
resolve a user's requested language against that list.

Used by chat.py (FastAPI, async session) and voice_ws_server.py (via the
admin HTTP API) so the bot answers strictly in one of the admin-allowed
languages, even when the user types/speaks in a different one.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import SystemSetting

# Master list of language codes the app knows how to handle at all.
ALL_LANGUAGES = {"as", "en", "hi", "kn", "bn"}

SHORT_TO_HEADER = {
    "as": "as-IN",
    "en": "en-IN",
    "hi": "hi-IN",
    "kn": "kn-IN",
    "bn": "bn-IN",
}
HEADER_TO_SHORT = {v: k for k, v in SHORT_TO_HEADER.items()}

_ALLOWED_KEY = "allowed_languages"
_DEFAULT_ALLOWED = ["as"]


async def get_allowed_languages(db: AsyncSession) -> list[str]:
    """Returns the admin-curated list of short codes (e.g. ['as', 'hi']).

    Falls back to Assamese-only if the admin hasn't set anything yet.
    """
    row = (
        await db.execute(
            select(SystemSetting).where(SystemSetting.key == _ALLOWED_KEY)
        )
    ).scalar_one_or_none()
    if not row or not row.value:
        return list(_DEFAULT_ALLOWED)
    codes = [c.strip() for c in row.value.split(",") if c.strip()]
    return codes or list(_DEFAULT_ALLOWED)


async def set_allowed_languages(db: AsyncSession, codes: list[str]) -> list[str]:
    """Validates and saves the admin-curated list. Returns the saved list."""
    codes = [c for c in codes if c in ALL_LANGUAGES]
    if not codes:
        raise ValueError("At least one valid language must be selected")

    row = (
        await db.execute(
            select(SystemSetting).where(SystemSetting.key == _ALLOWED_KEY)
        )
    ).scalar_one_or_none()
    value = ",".join(codes)
    if row is None:
        db.add(SystemSetting(key=_ALLOWED_KEY, value=value))
    else:
        row.value = value
    return codes


async def resolve_response_language_code(
    db: AsyncSession, requested_header_code: str | None
) -> str:
    """Given what the user's dropdown sent (e.g. 'hi-IN' or None), returns
    the header-format code (en-IN/hi-IN/as-IN/kn-IN/bn-IN) the response must
    actually use — the user's pick if it's in the admin-allowed list,
    otherwise the first allowed language.
    """
    allowed = await get_allowed_languages(db)
    requested_short = (
        HEADER_TO_SHORT.get(requested_header_code) if requested_header_code else None
    )
    chosen = requested_short if requested_short in allowed else allowed[0]
    return SHORT_TO_HEADER.get(chosen, "as-IN")
