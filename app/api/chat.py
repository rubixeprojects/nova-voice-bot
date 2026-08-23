"""Chat endpoints: text/voice, streaming/non-streaming (spec Â§4, Â§5 Chat)."""
from __future__ import annotations

import json
import time
import uuid
import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.db import AsyncSessionLocal, get_db
from app.core.deps import get_current_user_id, get_request_id_dep
from app.core.logging import get_logger
from app.llm import sarvam_client
from app.cleaner.unicode import detect_query_language
from app.llm.prompts import build_messages
from app.models import Conversation, Document, Message
from app.retrieval.pipeline import retrieve
from app.retrieval.types import RetrievedChunk
from app.schemas.chat import (
    ChatTextRequest,
    ChatTextResponse,
    ChatVoiceResponse,
    SourceRef,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
async def _get_or_create_conversation(
    db: AsyncSession,
    conversation_id: uuid.UUID | None,
    user_id: uuid.UUID,
    seed_title: str,
) -> Conversation:
    if conversation_id is not None:
        conv = (
            await db.execute(
                select(Conversation).where(
                    Conversation.id == conversation_id,
                    Conversation.user_id == user_id,
                    Conversation.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv
    conv = Conversation(user_id=user_id, title=seed_title[:80] or "New conversation")
    db.add(conv)
    await db.flush()
    return conv


async def _load_history(
    db: AsyncSession, conversation_id: uuid.UUID, turns: int
) -> list[dict]:
    rows = (
        await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(turns * 2)
        )
    ).scalars().all()
    rows = list(reversed(rows))
    return [{"role": m.role, "content": m.content_text} for m in rows]


async def _doc_name_map(
    db: AsyncSession, chunks: list[RetrievedChunk]
) -> dict[str, str]:
    doc_ids = {c.document_id for c in chunks if c.document_id}
    if not doc_ids:
        return {}
    rows = (
        await db.execute(
            select(Document.id, Document.display_name).where(
                Document.id.in_(doc_ids)
            )
        )
    ).all()
    return {str(r[0]): r[1] for r in rows}


def _build_sources(
    chunks: list[RetrievedChunk], name_map: dict[str, str]
) -> list[SourceRef]:
    sources: list[SourceRef] = []
    for c in chunks:
        sources.append(
            SourceRef(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name=name_map.get(str(c.document_id)),
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                section_title=c.section_title,
                score=round(c.score, 4),
                preview=(c.text or "")[:180],
            )
        )
    return sources


async def _persist_turn(
    db: AsyncSession,
    conv: Conversation,
    user_id: uuid.UUID,
    *,
    user_text: str,
    assistant_text: str,
    input_type: str,
    chunks: list[RetrievedChunk],
    latency_ms: int,
    audio_key: str | None = None,
) -> None:
    db.add(
        Message(
            conversation_id=conv.id,
            user_id=user_id,
            role="user",
            input_type=input_type,
            content_text=user_text,
            raw_audio_s3_key=audio_key,
        )
    )
    db.add(
        Message(
            conversation_id=conv.id,
            user_id=user_id,
            role="assistant",
            input_type="text",
            content_text=assistant_text,
            retrieved_chunk_ids=[str(c.chunk_id) for c in chunks],
            model_used=settings.sarvam_llm_model,
            latency_ms=latency_ms,
        )
    )
    conv.last_message_at = func.now()
    await db.flush()


# --------------------------------------------------------------------------- #
# Text â€” non-streaming
# --------------------------------------------------------------------------- #
@router.post(
    "/text", response_model=ChatTextResponse, summary="Chat over documents (text)"
)
async def chat_text(
    body: ChatTextRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id_dep),
    db: AsyncSession = Depends(get_db),
):
    started = time.perf_counter()
    conv = await _get_or_create_conversation(
        db, body.conversation_id, user_id, body.message
    )
    _cleaned, chunks = await retrieve(
        db, body.message, user_id,
        document_ids=body.document_ids,
        conversation_id=conv.id, request_id=request_id,
    )
    history = await _load_history(db, conv.id, settings.conversation_memory_turns)
    messages = build_messages(
        body.message, chunks, history,
        query_language=detect_query_language(body.message),
    )

    answer = await sarvam_client.chat_completion(
        messages, db=db, conversation_id=conv.id, request_id=request_id
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    name_map = await _doc_name_map(db, chunks)
    await _persist_turn(
        db, conv, user_id, user_text=body.message, assistant_text=answer,
        input_type="text", chunks=chunks, latency_ms=latency_ms,
    )
    return ChatTextResponse(
        conversation_id=conv.id,
        answer=answer,
        sources=_build_sources(chunks, name_map),
    )


# --------------------------------------------------------------------------- #
# Text â€” streaming (SSE)
# --------------------------------------------------------------------------- #
@router.post("/text/stream", summary="Chat over documents (text, SSE stream)")
async def chat_text_stream(
    body: ChatTextRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id_dep),
):
    async def event_gen():
        started = time.perf_counter()
        # Own the session for the whole stream lifetime.
        async with AsyncSessionLocal() as db:
            try:
                conv = await _get_or_create_conversation(
                    db, body.conversation_id, user_id, body.message
                )
                yield {"event": "conversation", "data": json.dumps(
                    {"conversation_id": str(conv.id)})}

                _cleaned, chunks = await retrieve(
                    db, body.message, user_id,
                    document_ids=body.document_ids,
                    conversation_id=conv.id, request_id=request_id,
                )
                history = await _load_history(
                    db, conv.id, settings.conversation_memory_turns
                )
                messages = build_messages(
                    body.message, chunks, history,
                    query_language=detect_query_language(body.message),
                )

                answer_parts: list[str] = []
                async for delta in sarvam_client.chat_completion_stream(
                    messages, db=db, conversation_id=conv.id, request_id=request_id
                ):
                    answer_parts.append(delta)
                    yield {"event": "token", "data": json.dumps({"delta": delta})}

                answer = "".join(answer_parts)
                latency_ms = int((time.perf_counter() - started) * 1000)
                name_map = await _doc_name_map(db, chunks)
                await _persist_turn(
                    db, conv, user_id, user_text=body.message,
                    assistant_text=answer, input_type="text",
                    chunks=chunks, latency_ms=latency_ms,
                )
                await db.commit()

                yield {"event": "sources", "data": json.dumps({
                    "conversation_id": str(conv.id),
                    "sources": [s.model_dump(mode="json")
                                for s in _build_sources(chunks, name_map)],
                })}
                yield {"event": "done", "data": json.dumps({"ok": True})}
            except Exception as exc:  # noqa: BLE001
                await db.rollback()
                log.exception("chat.stream_failed")
                yield {"event": "error", "data": json.dumps({"detail": str(exc)})}

    return EventSourceResponse(event_gen())


# --------------------------------------------------------------------------- #
# Voice â€” non-streaming
# --------------------------------------------------------------------------- #
@router.post(
    "/voice", response_model=ChatVoiceResponse,
    summary="Chat over documents (voice input, text answer)",
)
async def chat_voice(
    file: UploadFile = File(...),
    conversation_id: uuid.UUID | None = Form(default=None),
    document_ids: str | None = Form(
        default=None, description="JSON array of document UUIDs"
    ),
    user_id: uuid.UUID = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id_dep),
    db: AsyncSession = Depends(get_db),
):
    started = time.perf_counter()
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Empty audio file")

    doc_ids = _parse_doc_ids(document_ids)

    # Raw audio is not persisted (MinIO removed).
    audio_key = None

    conv = await _get_or_create_conversation(db, conversation_id, user_id, "Voice chat")

    stt = await sarvam_client.transcribe(
        audio,
        filename=file.filename or "audio.wav",
        content_type=file.content_type or "audio/wav",
        db=db,
        conversation_id=conv.id,
        request_id=request_id,
    )

    transcript = stt["transcript"]

    if not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not transcribe audio",
        )

    # Voice control commands â€” handle before RAG/LLM/TTS.
    # Supports English, Hindi, Kannada, and Assamese.
    stop_commands = {
        # English
        "stop",
        "stop listening",
        "cancel",
        "exit",
        "quit",

        # Hindi
        "à¤°à¥à¤•à¥‹",
        "à¤°à¥à¤• à¤œà¤¾à¤“",
        "à¤¬à¤‚à¤¦ à¤•à¤°à¥‹",
        "à¤¬à¤‚à¤¦ à¤•à¤° à¤¦à¥‹",
        "à¤°à¥à¤•à¤¿à¤",

        # Kannada
        "à²¨à²¿à²²à³à²²à²¿à²¸à³",
        "à²¨à²¿à²²à³à²²à²¿à²¸à²¿",
        "à²¨à²¿à²²à³à²²à²¿à²¸à³ à²•à³‡à²³à³à²µà³à²¦à²¨à³à²¨à³",

        # Assamese
        "à§°'à¦¬à¦¾",
        "à§°à¦¬à¦¾",
        "à§°à§ˆ à¦¯à§‹à§±à¦¾",
        "à§°à§ˆ à¦¯à¦¾à¦“à¦•",
        "à¦¬à¦¨à§à¦§ à¦•à§°à¦¾",
        "à¦¬à¦¨à§à¦§ à¦•à§°à¦•",
        "à¦¬à¦¨à§à¦§ à¦•à§°",
    }

    normalized_transcript = " ".join(
        transcript.lower().strip().split()
    )

    if normalized_transcript in stop_commands:
        return ChatVoiceResponse(
            conversation_id=conv.id,
            transcript=transcript,
            answer="Okay, stopping.",
            sources=[],
            audio_base64="",
        )

    language_code = (
        stt.get("language_code") or "en-IN"
    )

    if language_code not in {"en-IN", "kn-IN"}:
        language_code = "en-IN"
    intent = await sarvam_client.route_query(
        transcript,
        db=db,
        request_id=request_id,
    )



    history = await _load_history(
        db,
        conv.id,
        settings.conversation_memory_turns,
    )

    if intent == "rag":
        _cleaned, chunks = await retrieve(
            db,
            transcript,
            user_id,
            document_ids=doc_ids,
            conversation_id=conv.id,
            request_id=request_id,
        )
    else:
        chunks = []

    messages = build_messages(
        transcript,
        chunks,
        history,
        query_language=language_code,
    )
    answer = await sarvam_client.chat_completion(
        messages, db=db, conversation_id=conv.id, request_id=request_id
    )
    audio_answer = await sarvam_client.text_to_speech(
        text=answer,
        language_code=language_code,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    name_map = await _doc_name_map(db, chunks)
    await _persist_turn(
        db, conv, user_id, user_text=transcript, assistant_text=answer,
        input_type="voice", chunks=chunks, latency_ms=latency_ms,
        audio_key=audio_key,
    )
    return ChatVoiceResponse(
        conversation_id=conv.id,
        transcript=transcript,
        answer=answer,
        sources=_build_sources(chunks, name_map),
        audio_base64=base64.b64encode(audio_answer).decode("utf-8"),
    )


# --------------------------------------------------------------------------- #
# Voice â€” streaming (SSE): partial transcript, then answer tokens
# --------------------------------------------------------------------------- #
@router.post("/voice/stream", summary="Chat over documents (voice, SSE stream)")
async def chat_voice_stream(
    file: UploadFile = File(...),
    conversation_id: uuid.UUID | None = Form(default=None),
    document_ids: str | None = Form(default=None),
    user_id: uuid.UUID = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id_dep),
):
    audio = await file.read()

    if not audio:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file",
        )

    filename = file.filename or "audio.wav"
    content_type = file.content_type or "audio/wav"
    doc_ids = _parse_doc_ids(document_ids)

    async def event_gen():
        started = time.perf_counter()

        async with AsyncSessionLocal() as db:
            try:
                audio_key = None

                conv = await _get_or_create_conversation(
                    db,
                    conversation_id,
                    user_id,
                    "Voice chat",
                )

                yield {
                    "event": "conversation",
                    "data": json.dumps(
                        {
                            "conversation_id": str(conv.id)
                        }
                    ),
                }

                # ---------------------------------------------------------
                # STT
                # ---------------------------------------------------------

                stt = await sarvam_client.transcribe(
                    audio,
                    filename=filename,
                    content_type=content_type,
                    db=db,
                    conversation_id=conv.id,
                    request_id=request_id,
                )

                transcript = stt["transcript"]

                language_code = stt.get("language_code") or "en-IN"

                if language_code not in {"en-IN", "kn-IN"}:
                    language_code = "en-IN"

                log.info(
                    "voice.language_detected",
                    language_code=language_code,
                    transcript_chars=len(transcript),
                )

                yield {
                    "event": "transcript",
                    "data": json.dumps(
                        {
                            "transcript": transcript,
                            "language_code": language_code,
                        }
                    ),
                }

                if not transcript.strip():
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {
                                "detail": "Could not transcribe audio"
                            }
                        ),
                    }
                    return

                # ---------------------------------------------------------
                # ROUTER
                # ---------------------------------------------------------

                intent = await sarvam_client.route_query(
                    transcript,
                    db=db,
                    request_id=request_id,
                )

                log.info(
                    "voice.query_routed",
                    intent=intent,
                )

                # ---------------------------------------------------------
                # Conversation history
                # ---------------------------------------------------------

                history = await _load_history(
                    db,
                    conv.id,
                    settings.conversation_memory_turns,
                )

                # ---------------------------------------------------------
                # RAG only when required
                # ---------------------------------------------------------

                if intent == "rag":
                    _cleaned, chunks = await retrieve(
                        db,
                        transcript,
                        user_id,
                        document_ids=doc_ids,
                        conversation_id=conv.id,
                        request_id=request_id,
                    )
                else:
                    chunks = []

                # ---------------------------------------------------------
                # Build LLM messages
                # ---------------------------------------------------------

                messages = build_messages(
                    transcript,
                    chunks,
                    history,
                    query_language=language_code,
                )

                # ---------------------------------------------------------
                # Stream LLM response
                # ---------------------------------------------------------

                answer_parts: list[str] = []

                async for delta in sarvam_client.chat_completion_stream(
                    messages,
                    db=db,
                    conversation_id=conv.id,
                    request_id=request_id,
                ):
                    answer_parts.append(delta)

                    yield {
                        "event": "token",
                        "data": json.dumps(
                            {
                                "delta": delta
                            }
                        ),
                    }

                answer = "".join(answer_parts)

                latency_ms = int(
                    (time.perf_counter() - started) * 1000
                )

                # ---------------------------------------------------------
                # Persist conversation
                # ---------------------------------------------------------

                name_map = await _doc_name_map(
                    db,
                    chunks,
                )

                await _persist_turn(
                    db,
                    conv,
                    user_id,
                    user_text=transcript,
                    assistant_text=answer,
                    input_type="voice",
                    chunks=chunks,
                    latency_ms=latency_ms,
                    audio_key=audio_key,
                )

                await db.commit()

                # ---------------------------------------------------------
                # Sources
                # ---------------------------------------------------------

                yield {
                    "event": "sources",
                    "data": json.dumps(
                        {
                            "conversation_id": str(conv.id),
                            "sources": [
                                s.model_dump(mode="json")
                                for s in _build_sources(
                                    chunks,
                                    name_map,
                                )
                            ],
                        }
                    ),
                }

                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "ok": True
                        }
                    ),
                }

            except Exception as exc:  # noqa: BLE001
                await db.rollback()

                log.exception(
                    "chat.voice_stream_failed"
                )

                yield {
                    "event": "error",
                    "data": json.dumps(
                        {
                            "detail": str(exc)
                        }
                    ),
                }

    return EventSourceResponse(event_gen())
# --------------------------------------------------------------------------- #
# Voice â€” STT only
# --------------------------------------------------------------------------- #
@router.post("/voice/transcribe", summary="Transcribe microphone audio")
async def transcribe_voice(
    file: UploadFile = File(...),
    user_id: uuid.UUID = Depends(get_current_user_id),
    request_id: str = Depends(get_request_id_dep),
    db: AsyncSession = Depends(get_db),
):
    audio = await file.read()

    if not audio:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file",
        )

    result = await sarvam_client.transcribe(
        audio,
        filename=file.filename or "recording.webm",
        content_type=file.content_type or "audio/webm",
        db=db,
        request_id=request_id,
    )

    transcript = result.get("transcript", "")

    if not transcript.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not transcribe audio",
        )

    return {
        "transcript": transcript,
        "language_code": result.get("language_code"),
        "confidence": result.get("confidence"),
    }

def _parse_doc_ids(raw: str | None) -> list[uuid.UUID] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return [uuid.UUID(str(x)) for x in parsed]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail="document_ids must be a JSON array of UUIDs"
        ) from exc
