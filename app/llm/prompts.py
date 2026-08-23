"""Bilingual prompt construction for sarvam-105b."""
from __future__ import annotations

from app.cleaner.unicode import detect_query_language
from app.retrieval.types import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a helpful multilingual assistant.\n"
    "1. Answer naturally, conversationally, and concisely.\n"
    "2. For casual conversation, greetings, and general questions "
    "that do not require documents, answer directly using your general knowledge.\n"
    "3. Keep answers short and conversational — normally 1-3 sentences.\n"
    "4. Give concise explanations with short sentences unless the user asks for detail.\n"
    "5. Do not provide analysis or reasoning unless requested.\n"
    "6. For document-based answers, cite relevant sources with [S#] markers. "
    "Do not use [S#] citations for casual conversation."
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
