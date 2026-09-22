"""Production ingestion orchestrator (P0–P4)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.chunker.semantic import chunk_sections
from app.cleaner.headers import strip_headers_footers
from app.cleaner.unicode import (
    clean_text,
    detect_language,
    indic_script_ratio,
    latin_script_ratio,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.ocr import run_ocr
from app.layout.blocks import classify_blocks
from app.layout.sections import detect_sections
from app.parser.router import detect_profile, pages_from_ocr, parse_document
from app.parser.other_formats import pages_from_docx, pages_from_plaintext
from app.pipeline.types import ChunkDraft, DocumentProfile, ParsedPage

log = get_logger(__name__)


@dataclass
class IngestArtifacts:
    profile: DocumentProfile
    pages: list[ParsedPage]
    sections: list
    chunks: list[ChunkDraft]
    page_confidence: dict[int, float]
    detected_language: str = "as"


def _merge_parsed_with_ocr(
    parsed: list[ParsedPage], ocr_pages: list[tuple[int, str, float]]
) -> list[ParsedPage]:
    ocr_by_page = {pn: (txt, conf) for pn, txt, conf in ocr_pages}
    out: list[ParsedPage] = []
    for page in parsed:
        pn = page.page_number
        ocr_txt, ocr_conf = ocr_by_page.get(pn, ("", 0.0))
        use_ocr = (
            (
                page.indic_ratio < 0.15
                and latin_script_ratio(page.text) < 0.15
            )
            or len(page.text.strip()) < 20
        ) and ocr_txt.strip()
        if use_ocr:
            cleaned, q = clean_text(ocr_txt, preserve_line_breaks=True)
            out.extend(
                pages_from_ocr([(pn, cleaned, ocr_conf)], engine="ocr")
            )
        else:
            cleaned, _ = clean_text(page.text, preserve_line_breaks=True)
            page.text = cleaned
            page.indic_ratio = indic_script_ratio(cleaned)
            out.append(page)
    # Pages only in OCR
    parsed_nums = {p.page_number for p in parsed}
    for pn, txt, conf in ocr_pages:
        if pn not in parsed_nums and txt.strip():
            cleaned, _ = clean_text(txt, preserve_line_breaks=True)
            out.extend(pages_from_ocr([(pn, cleaned, conf)], engine="ocr"))
    out.sort(key=lambda p: p.page_number)
    return out


def run_ingestion_pipeline(file_bytes: bytes, filename: str = "") -> IngestArtifacts:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "pdf"
    page_conf: dict[int, float] = {}
    if ext == "docx":
        pages = pages_from_docx(file_bytes)
        profile = DocumentProfile(
            document_type="docx",
            has_bookmarks=False,
            has_tables=any(b.block_type == "table" for p in pages for b in (p.blocks or [])),
            avg_indic_ratio=pages[0].indic_ratio if pages else 0.0,
            page_count=1,
        )
    elif ext in ("txt", "md", "markdown"):
        pages = pages_from_plaintext(file_bytes, is_markdown=(ext != "txt"))
        profile = DocumentProfile(
            document_type="markdown" if ext != "txt" else "text",
            has_bookmarks=False,
            has_tables=False,
            avg_indic_ratio=pages[0].indic_ratio if pages else 0.0,
            page_count=1,
        )
    else:
        # --- PDF path: completely unchanged from before ---
        pdf_bytes = file_bytes
        profile, parsed = parse_document(pdf_bytes)
        ocr_result = run_ocr(pdf_bytes)
        ocr_pages = [(p.page_number, p.text, p.confidence) for p in ocr_result.pages]
        page_conf = {p.page_number: p.confidence for p in ocr_result.pages}

        if profile.document_type in ("scanned_pdf", "mixed") or profile.avg_indic_ratio < 0.12:
            pages = pages_from_ocr(ocr_pages, engine="ocr_merged")
            if parsed:
                pages = _merge_parsed_with_ocr(parsed, ocr_pages)
        else:
            pages = _merge_parsed_with_ocr(parsed, ocr_pages) if parsed else pages_from_ocr(ocr_pages)
    pages = strip_headers_footers(pages)
    pages = classify_blocks(pages)
    sections = detect_sections(pages)
    combined_text = "\n".join(p.text for p in pages if p.text.strip())
    detected_language = detect_language(
        combined_text, default=settings.default_document_language
    )
    chunks = chunk_sections(
        sections,
        document_type=profile.document_type,
        source_language=detected_language,
        page_confidence=page_conf,
    )
    log.info(
        "pipeline.ingest",
        document_type=profile.document_type,
        detected_language=detected_language,
        pages=len(pages),
        sections=len(sections),
        chunks=len(chunks),
    )
    return IngestArtifacts(
        profile=profile,
        pages=pages,
        sections=sections,
        chunks=chunks,
        page_confidence=page_conf,
        detected_language=detected_language,
    )
