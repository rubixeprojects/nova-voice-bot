"""Bilingual prompt construction for sarvam-105b."""
from __future__ import annotations

from pathlib import Path

import yaml

from app.cleaner.unicode import detect_query_language
from app.retrieval.types import RetrievedChunk

def _load_persona() -> tuple[str, str]:
    """Return (name, prompt) from persona.yml; fall back gracefully."""
    persona_file = Path(__file__).resolve().parents[2] / "persona.yml"
    try:
        data = yaml.safe_load(persona_file.read_text()) or {}
        name = data.get("name", "Nova").strip()
        prompt = data.get("prompt", "").strip()
        if prompt:
            return name, prompt
    except Exception:
        pass
    return "Nova", "You are Nova, your assistant."

_PERSONA_NAME, _PERSONA_PROMPT = _load_persona()

SYSTEM_PROMPT = (
    _PERSONA_PROMPT + "\n"
    f"When users open with a greeting (hi, hello, hey, good morning, etc.), respond only with: 'Hi! I'm {_PERSONA_NAME}, your assistant. How can I help you?'\n"
    "When users ask how you are or make small talk, respond warmly and briefly — do not repeat your introduction.\n"
    "Never reveal your reasoning, planning, or thinking process — output only your final answer.\n"
    "Never say 'based on the context', 'according to the provided context', or similar — just answer directly.\n"
    "Never redirect users to external websites, FAQs, or official resources — give the answer or say you don't have it.\n"
    "Keep answers concise; give detail only when the user asks for it.\n"
    "For document-based answers cite sources with [S#] markers; skip citations for casual conversation.\n"
    "If a question is completely outside your domain, say so briefly and offer to help with something relevant."
)

def build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        text = c.payload.get("expanded_text") or c.text
        title = c.section_title or ""
        page = f"p.{c.page_number}" if c.page_number is not None else ""
        header = f"[S{i}]" + (f" {title}" if title else "") + (f" ({page})" if page else "")
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def _build_user_turn(
    query: str,
    context: str,
    language_name: str,
) -> str:
    return (
        f"Context:\n{context}\n\n"
        "----\n"
        f"User question: {query}\n\n"
        f"IMPORTANT: Your response MUST be written entirely in {language_name}. "
        "Do not use any other language regardless of the language of the context above. "
        "If relevant context is provided, use it to answer accurately and cite sources with [S#] markers."
    )


def build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
    *,
    query_language: str | None = None,
) -> list[dict]:
    """Assemble chat messages using the STT-detected language."""
    lang = query_language or detect_query_language(query)

    # Accept both short codes (from UI/detect) and Sarvam full codes (from STT)
    language_name = {
        "en": "English",   "en-IN": "English",
        "hi": "Hindi",     "hi-IN": "Hindi",
        "kn": "Kannada",   "kn-IN": "Kannada",
        "bn": "Bengali",   "bn-IN": "Bengali",
        "ta": "Tamil",     "ta-IN": "Tamil",
        "te": "Telugu",    "te-IN": "Telugu",
        "ml": "Malayalam", "ml-IN": "Malayalam",
        "mr": "Marathi",   "mr-IN": "Marathi",
        "gu": "Gujarati",  "gu-IN": "Gujarati",
        "pa": "Punjabi",   "pa-IN": "Punjabi",
        "od": "Odia",      "od-IN": "Odia",
        "as": "Assamese",  "as-IN": "Assamese",
        "indic": "the same Indic language as the user's question",
    }.get(lang, "English")

    system_prompt = (
        SYSTEM_PROMPT
        + f"\n\nCRITICAL LANGUAGE RULE: You MUST respond ONLY in {language_name}. "
        f"Regardless of the language of the retrieved context, your answer must be in {language_name}. "
        "Do NOT switch languages mid-response."
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt}
    ]

    if history:
        messages.extend(history)

    context = build_context_block(chunks)

    messages.append(
        {
            "role": "user",
            "content": _build_user_turn(query, context, language_name),
        }
    )

    return messages
