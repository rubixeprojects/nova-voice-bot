"""Plain-text extraction for non-PDF formats (.docx, .txt, .md).

Produces the same ParsedPage/TextBlock shapes the PDF pipeline produces,
so everything downstream (classify_blocks, detect_sections, chunk_sections)
works unmodified.
"""
from __future__ import annotations

from app.cleaner.unicode import indic_script_ratio
from app.pipeline.types import ParsedPage, TextBlock


def pages_from_docx(file_bytes: bytes) -> list[ParsedPage]:
    import io
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(file_bytes))
    blocks: list[TextBlock] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "") if para.style else ""
        heading_level = None
        block_type = "paragraph"
        if style_name.lower().startswith("heading"):
            block_type = "heading"
            digits = "".join(ch for ch in style_name if ch.isdigit())
            heading_level = int(digits) if digits else 1
        blocks.append(
            TextBlock(
                block_type=block_type,
                text=text,
                page_start=1,
                page_end=1,
                confidence=1.0,
                heading_level=heading_level,
            )
        )

    for table in doc.tables:
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            rows_text.append(" | ".join(cells))
        table_text = "\n".join(rows_text).strip()
        if table_text:
            blocks.append(
                TextBlock(
                    block_type="table",
                    text=table_text,
                    page_start=1,
                    page_end=1,
                    confidence=1.0,
                )
            )

    full_text = "\n".join(b.text for b in blocks)
    return [
        ParsedPage(
            page_number=1,
            text=full_text,
            confidence=1.0,
            engine="docx",
            blocks=blocks,
            indic_ratio=indic_script_ratio(full_text),
        )
    ]


def pages_from_plaintext(file_bytes: bytes, is_markdown: bool = False) -> list[ParsedPage]:
    text = file_bytes.decode("utf-8", errors="replace")
    raw_lines = text.split("\n")

    # Group lines into paragraphs by blank-line breaks, the way a human reader
    # would — a lone line surrounded by blank lines is a heading/title; several
    # consecutive non-blank lines are one paragraph or list, kept together as
    # a single block instead of being split line-by-line.
    paragraphs: list[list[str]] = []
    current: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        paragraphs.append(current)

    blocks: list[TextBlock] = []
    for para_lines in paragraphs:
        block_type = "paragraph"
        heading_level = None

        if is_markdown and para_lines[0].startswith("#"):
            block_type = "heading"
            heading_level = len(para_lines[0]) - len(para_lines[0].lstrip("#"))
            para_lines = [para_lines[0].lstrip("#").strip()] + para_lines[1:]
        elif len(para_lines) == 1:
            # A single short standalone line (no blank-line-separated siblings)
            # reads as a heading/title, same convention PDFs use.
            block_type = "heading"

        combined_text = "\n".join(para_lines)
        blocks.append(
            TextBlock(
                block_type=block_type,
                text=combined_text,
                page_start=1,
                page_end=1,
                confidence=1.0,
                heading_level=heading_level,
            )
        )

    full_text = "\n\n".join(b.text for b in blocks)
    return [
        ParsedPage(
            page_number=1,
            text=full_text,
            confidence=1.0,
            engine="markdown" if is_markdown else "plaintext",
            blocks=blocks,
            indic_ratio=indic_script_ratio(full_text),
        )
    ]