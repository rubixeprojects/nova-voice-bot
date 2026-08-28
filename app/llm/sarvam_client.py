"""Sarvam AI client wrappers: Saaras v3 STT, sarvam-30b chat, Vision OCR.

Every external call is:
  * time-bounded and retried with exponential backoff (3 attempts),
  * logged per-attempt into pipeline_stage_logs (logging rule #5) when a db
    session + correlation ids are supplied,
  * summarized safely â€” we log lengths/hashes/previews, never full audio or
    document text (logging rule #2).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.core.stage_logger import record_stage_async, record_stage_sync

log = get_logger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5  # seconds


def _headers() -> dict:
    return {
        "api-subscription-key": settings.sarvam_api_key,
        "Authorization": f"Bearer {settings.sarvam_api_key}",
    }


def _err_detail(exc: Exception) -> str:
    """Build a log-safe error string, including the API response body when the
    failure is an HTTP status error (Sarvam returns the real reason in the body,
    which ``str(exc)`` omits)."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            body = exc.response.text
        except Exception:  # noqa: BLE001
            body = ""
        return f"{exc} | body={body[:500]}" if body else str(exc)
    return str(exc)


def _preview(text: str | None, n: int = 120) -> str:
    if not text:
        return ""
    return text[:n]


def _summary(text: str | None) -> dict:
    if not text:
        return {"len": 0, "sha1": None, "preview": ""}
    return {
        "len": len(text),
        "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
        "preview": _preview(text),
    }


def _extract_message_text(message: dict | None) -> str | None:
    """Pull assistant text from a Sarvam/OpenAI-style message object."""
    if not message:
        return None
    for key in ("content", "reasoning_content", "text"):
        val = message.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _response_debug_summary(data: dict) -> dict:
    """Safe metadata for logs when parsing fails (never log full body)."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    return {
        "model": data.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "message_keys": list(message.keys()),
        "content_is_null": content is None,
        "content_len": len(content) if isinstance(content, str) else 0,
        "reasoning_len": len(reasoning) if isinstance(reasoning, str) else 0,
    }


def _extract_chat_answer(data: dict) -> str:
    """Parse non-streaming Sarvam chat completion into answer text."""
    choices = data.get("choices")
    if not choices:
        raise ValueError(
            "Sarvam response missing choices: "
            f"{json.dumps(_response_debug_summary(data))}"
        )
    choice = choices[0] or {}
    message = choice.get("message") or {}
    text = _extract_message_text(message)
    if text:
        return text

    # Some providers expose a top-level text field.
    for key in ("output", "text", "response"):
        val = data.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    raise ValueError(
        "Sarvam returned empty assistant message: "
        f"{json.dumps(_response_debug_summary(data))}"
    )


def _extract_stream_delta(chunk: dict) -> str | None:
    """Extract a text delta from one SSE chunk."""
    try:
        choices = chunk.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        for key in ("content", "reasoning_content", "text"):
            val = delta.get(key)
            if val:
                return str(val)
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
# Vision OCR fallback (sync â€” used by ingestion worker)
# --------------------------------------------------------------------------- #
def sarvam_vision_ocr(
    image_png: bytes,
    *,
    page_number: int,
    db=None,
    document_id: uuid.UUID | None = None,
) -> tuple[str, float]:
    """OCR a single page image via Sarvam's document-intelligence endpoint.

    Returns (text, confidence). Confidence from Sarvam Vision is treated as high
    (0.9) when the call succeeds, since the API does not always return a numeric
    score â€” TODO: replace with a real per-page score if/when exposed.
    """
    # TODO(endpoint): Confirm the exact Sarvam Vision/doc-intelligence route.
    url = settings.sarvam_base_url.rstrip("/") + "/v1/vision/ocr"
    last_err: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    url,
                    headers=_headers(),
                    files={"file": (f"page_{page_number}.png", image_png, "image/png")},
                    data={"language": "as-IN"},
                )
                resp.raise_for_status()
                payload = resp.json()
            text = payload.get("text") or payload.get("transcript") or ""
            conf = float(payload.get("confidence", 0.9))
            dur = int((time.perf_counter() - started) * 1000)
            if db is not None:
                record_stage_sync(
                    db, "ocr", status="success", component="sarvam_vision",
                    duration_ms=dur, document_id=document_id, request_id=None,
                    input_summary={"page": page_number, "attempt": attempt},
                    output_summary=_summary(text),
                )
            return text, conf
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            detail = _err_detail(exc)
            dur = int((time.perf_counter() - started) * 1000)
            if db is not None:
                record_stage_sync(
                    db, "ocr", status="failed", component="sarvam_vision",
                    duration_ms=dur, document_id=document_id, request_id=None,
                    input_summary={"page": page_number, "attempt": attempt},
                    error_message=detail,
                )
            log.warning("sarvam.vision.attempt_failed", attempt=attempt, error=detail)
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"Sarvam Vision OCR failed after {_MAX_ATTEMPTS} attempts: {_err_detail(last_err)}"
    )


# --------------------------------------------------------------------------- #
# STT: Saaras v3 (async â€” used by voice chat)
# --------------------------------------------------------------------------- #
async def transcribe(
    audio_bytes: bytes,
    *,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    db=None,
    conversation_id: uuid.UUID | None = None,
    language_code: str = "unknown",
) -> dict:
    """Transcribe multilingual audio using Sarvam Saaras v3.

    Pass a specific language_code (e.g. "hi-IN") when the caller already
    knows the language, for better accuracy than auto-detection. Defaults
    to "unknown" (auto-detect) for backward compatibility.
    """
    url = settings.sarvam_base_url.rstrip("/") + "/speech-to-text"
    last_err: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        started = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    url,
                    headers=_headers(),
                    files={
                        "file": (
                            filename,
                            audio_bytes,
                            content_type,
                        )
                    },
                    data={
                        "model": settings.sarvam_stt_model,
                        "mode": "transcribe",
                        "language_code": language_code,
                    },
                )

                resp.raise_for_status()
                payload = resp.json()

            transcript = (
                payload.get("transcript")
                or payload.get("text")
                or ""
            )

            out = {
                "transcript": transcript,
                "language_code": payload.get("language_code"),
                "confidence": payload.get("language_probability"),
            }

            dur = int((time.perf_counter() - started) * 1000)

            if db is not None:
                await record_stage_async(
                    db,
                    "stt",
                    status="success",
                    component="saaras_v3",
                    duration_ms=dur,
                    conversation_id=conversation_id,
                    request_id=None,
                    input_summary={
                        "audio_bytes": len(audio_bytes),
                        "attempt": attempt,
                    },
                    output_summary={
                        **_summary(transcript),
                        "language_code": out["language_code"],
                        "language_confidence": out["confidence"],
                    },
                )

            log.info(
                "sarvam.stt.ok",
                duration_ms=dur,
                chars=len(transcript),
                language_code=out["language_code"],
                language_confidence=out["confidence"],
            )

            return out

        except Exception as exc:
            last_err = exc
            detail = _err_detail(exc)
            dur = int((time.perf_counter() - started) * 1000)

            if db is not None:
                await record_stage_async(
                    db,
                    "stt",
                    status="failed",
                    component="saaras_v3",
                    duration_ms=dur,
                    conversation_id=conversation_id,
                    request_id=None,
                    input_summary={
                        "audio_bytes": len(audio_bytes),
                        "attempt": attempt,
                    },
                    error_message=detail,
                )

            log.warning(
                "sarvam.stt.attempt_failed",
                attempt=attempt,
                error=detail,
            )

            if attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(
                    _BACKOFF_BASE * (2 ** (attempt - 1))
                )

    raise RuntimeError(
        f"Sarvam STT failed after {_MAX_ATTEMPTS} attempts: "
        f"{_err_detail(last_err)}"
    )
# --------------------------------------------------------------------------- #
# TTS: Sarvam Bulbul v3
# --------------------------------------------------------------------------- #
async def text_to_speech(
    text: str,
    language_code: str,
) -> bytes:
    """
    Convert assistant text to speech using Sarvam Bulbul v3.

    language_code comes from STT:
        en-IN
        kn-IN

    """

    supported_languages = {
        "en-IN",
        "hi-IN",
        "kn-IN",
        "as-IN",
    }

    if language_code not in supported_languages:
        language_code = "en-IN"

    url = settings.sarvam_base_url.rstrip("/") + "/text-to-speech"

    payload = {
        "text": text,
        "target_language_code": language_code,
        "speaker": "shubh",
        "model": settings.sarvam_tts_model,
        "output_audio_codec": "wav",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            url,
            headers={
                "api-subscription-key": settings.sarvam_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
        )

        response.raise_for_status()

        data = response.json()

    audio_base64 = data["audios"][0]

    return base64.b64decode(audio_base64)

# --------------------------------------------------------------------------- #
# LLM: sarvam-30b chat completion (async)
# --------------------------------------------------------------------------- #
def _chat_url() -> str:
    return settings.sarvam_base_url.rstrip("/") + "/v1/chat/completions"
async def route_query(
    query: str,
    *,
    db=None,
) -> str:
    """
    Decide whether the user's query needs document retrieval.

    Returns:
        "rag"          -> use document retrieval
        "conversation" -> answer directly using general knowledge
    """

    router_messages = [
        {
            "role": "system",
            "content": (
                "You are a query router for a voice assistant.\n"
                "Return ONLY one word: rag or conversation.\n\n"
                "Return 'rag' when the user asks about information that "
                "could come from uploaded documents, projects, policies, "
                "fees, attendance, rules, reports, MIS, Copilot, or any "
                "specific information stored in the documents.\n"
                "Return 'conversation' for greetings, small talk, general "
                "knowledge, casual questions, opinions, jokes, or questions "
                "that do not require the uploaded documents."
            ),
                    },
        {
            "role": "user",
            "content": query,
        },
    ]

    result = await chat_completion(
        router_messages,
        db=db,
        
    )

    result = result.strip().lower()

    if result == "rag":
        return "rag"

    if result == "conversation":
        return "conversation"

    # Handle accidental extra text from the router model.
    first_word = result.split()[0] if result.split() else ""

    if first_word == "rag":
        return "rag"

    return "conversation"

async def chat_completion(
    messages: list[dict],
    *,
    db=None,
    conversation_id: uuid.UUID | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Non-streaming chat completion. Returns the full answer text."""
    payload = {
        "model": settings.sarvam_llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    last_err: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        data: dict | None = None
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                resp = await client.post(
                    _chat_url(), headers=_headers(), json=payload
                )
                resp.raise_for_status()
                data = resp.json()
            answer = _extract_chat_answer(data)
            dur = int((time.perf_counter() - started) * 1000)
            if db is not None:
                await record_stage_async(
                    db, "llm_generation", status="success", component="sarvam_30b",
                    duration_ms=dur, conversation_id=conversation_id,
                    request_id=None,
                    input_summary={"messages": len(messages), "attempt": attempt},
                    output_summary=_summary(answer),
                )
            return answer
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            detail = _err_detail(exc)
            dur = int((time.perf_counter() - started) * 1000)
            if data is not None:
                log.warning(
                    "sarvam.llm.parse_failed",
                    attempt=attempt,
                    error=detail,
                    response=_response_debug_summary(data),
                )
            if db is not None:
                await record_stage_async(
                    db, "llm_generation", status="failed", component="sarvam_30b",
                    duration_ms=dur, conversation_id=conversation_id,
                    request_id=None,
                    input_summary={"messages": len(messages), "attempt": attempt},
                    error_message=detail,
                )
            log.warning("sarvam.llm.attempt_failed", attempt=attempt, error=detail)
            if attempt < _MAX_ATTEMPTS:
                import asyncio

                await asyncio.sleep(_BACKOFF_BASE * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"Sarvam LLM failed after {_MAX_ATTEMPTS} attempts: {_err_detail(last_err)}"
    )


async def chat_completion_stream(
    messages: list[dict],
    *,
    db=None,
    conversation_id: uuid.UUID | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """Streaming chat completion. Yields answer token deltas.

    Note: streaming is not retried mid-stream (once tokens start flowing a retry
    would duplicate output); connection-setup failures before the first token
    are retried by re-entering. A single success/failed stage row is logged.
    """
    payload = {
        "model": settings.sarvam_llm_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    started = time.perf_counter()
    produced = 0
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", _chat_url(), headers=_headers(), json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = _extract_stream_delta(obj)
                    except Exception:  # noqa: BLE001
                        continue
                    if delta:
                        produced += len(delta)
                        yield delta
        if produced == 0:
            raise ValueError("Sarvam stream returned no text deltas")
        dur = int((time.perf_counter() - started) * 1000)
        if db is not None:
            await record_stage_async(
                db, "llm_generation", status="success", component="sarvam_30b",
                duration_ms=dur, conversation_id=conversation_id,
                request_id=None,
                input_summary={"messages": len(messages), "stream": True},
                output_summary={"chars": produced},
            )
    except Exception as exc:  # noqa: BLE001
        dur = int((time.perf_counter() - started) * 1000)
        if db is not None:
            await record_stage_async(
                db, "llm_generation", status="failed", component="sarvam_30b",
                duration_ms=dur, conversation_id=conversation_id,
                request_id=None,
                input_summary={"messages": len(messages), "stream": True},
                error_message=_err_detail(exc),
            )
        raise
