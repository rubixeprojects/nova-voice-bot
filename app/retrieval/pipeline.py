"""End-to-end retrieval: embed → dense ‖ bm25 → RRF → MMR → rerank → context."""
from __future__ import annotations

import asyncio
import time
import uuid
import re
from app.core.clients import get_qdrant
from app.core.config import settings
from app.core.logging import get_logger
from app.core.stage_logger import astage, record_stage_async
from app.cleaner.unicode import clean_text
from app.ingestion.embedding import embed_query
from app.prompt_builder.context import build_context
from app.retrieval import bm25, dense
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import rerank
from app.retrieval.types import RetrievedChunk
from app.retriever.debug import log_retrieval_trace
from app.retriever.mmr import maximal_marginal_relevance

log = get_logger(__name__)
def _expand_split_lists(reranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """If a chunk states a count like '29 MOOCs' or '12 courses', follow its
    next_chunk_id chain in Qdrant to pull in any sibling chunks that continue
    the same list but scored too low to be reranked in on their own. Generic —
    no hardcoded numbers, phrases, or document names."""
    count_pattern = re.compile(r"\b(\d+)\b.{0,60}?\b([A-Za-z]+s)\b")

    existing_ids = {c.chunk_id for c in reranked}
    to_add: list[RetrievedChunk] = []
    client = get_qdrant()

    for chunk in reranked:
        match = count_pattern.search(chunk.text)
        if not match:
            continue
        stated_count = int(match.group(1))

        # Walk forward through the chunk chain, collecting siblings.
        next_id = chunk.next_chunk_id
        collected_items = 0
        hops = 0
        while next_id and hops < 10:  # safety cap, not a real limit in practice
            hops += 1
            if next_id in existing_ids:
                break  # already in reranked set, chain likely already covered
            points = client.retrieve(
                collection_name=settings.qdrant_collection,
                ids=[str(next_id)],
                with_payload=True,
            )
            if not points:
                break
            p = points[0].payload or {}
            text = p.get("text", "")

            # Rough line-count as a stand-in for "items in this chunk".
            # Try common separators in order: bullets/newlines first, then
            # double-space (seen in this corpus), falling back to single lines.
            lines = [l.strip() for l in re.split(r"\n|•|▪|\u2022|\uf0b7", text) if l.strip()]
            if len(lines) <= 1:
                lines = [l.strip() for l in text.split("  ") if l.strip()]
            if len(lines) <= 1:
                lines = [text.strip()] if text.strip() else []
            collected_items += max(len(lines), 1)

            new_chunk = RetrievedChunk(
                chunk_id=uuid.UUID(str(p.get("chunk_id") or next_id)),
                document_id=uuid.UUID(str(p.get("document_id"))) if p.get("document_id") else None,
                text=text,
                score=chunk.score,  # inherit parent's relevance, it's part of the same list
                chunk_index=p.get("chunk_index"),
                section_id=uuid.UUID(str(p["section_id"])) if p.get("section_id") else None,
                section_title=p.get("section_title"),
                page_number=p.get("page_start") or p.get("page_number"),
                prev_chunk_id=uuid.UUID(str(p["prev_chunk_id"])) if p.get("prev_chunk_id") else None,
                next_chunk_id=uuid.UUID(str(p["next_chunk_id"])) if p.get("next_chunk_id") else None,
                source="list_chain_expansion",
                payload=p,
            )
            to_add.append(new_chunk)
            existing_ids.add(new_chunk.chunk_id)

            if collected_items >= stated_count:
                break
            next_id = new_chunk.next_chunk_id

    return reranked + to_add

async def retrieve(
    db,
    query: str,
    user_id: uuid.UUID,
    *,
    document_ids: list[uuid.UUID] | None = None,
    conversation_id: uuid.UUID | None = None,
    request_id: str | uuid.UUID | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    cleaned, _ = clean_text(query)
    embed_component = "hf_inference" if settings.hf_token else "bge_m3"

    async with astage(db, "query_embedding", component=embed_component,
                      conversation_id=conversation_id, request_id=request_id) as st:
        qvec = await asyncio.to_thread(embed_query, cleaned)
        st.output({"dim": len(qvec), "backend": embed_component})

    async def _dense_search():
        t0 = time.perf_counter()
        res = await asyncio.to_thread(
            dense.dense_search, qvec, user_id, document_ids, settings.dense_top_k
        )
        return res, int((time.perf_counter() - t0) * 1000)

    async def _bm25_search():
        t0 = time.perf_counter()
        res = await asyncio.to_thread(
            bm25.bm25_search, cleaned, user_id, document_ids, settings.bm25_top_k
        )
        return res, int((time.perf_counter() - t0) * 1000)

    (dense_hits, dense_ms), (bm25_hits, bm25_ms) = await asyncio.gather(
        _dense_search(), _bm25_search()
    )

    await record_stage_async(
        db, "dense_retrieval", status="success", component="qdrant",
        duration_ms=dense_ms, conversation_id=conversation_id, request_id=request_id,
        output_summary={"hits": len(dense_hits)},
    )
    await record_stage_async(
        db, "bm25_retrieval", status="success", component="opensearch",
        duration_ms=bm25_ms, conversation_id=conversation_id, request_id=request_id,
        output_summary={"hits": len(bm25_hits)},
    )

    async with astage(db, "fusion", component="rrf",
                      conversation_id=conversation_id, request_id=request_id) as st:
        fused = reciprocal_rank_fusion([dense_hits, bm25_hits], top_k=settings.rrf_top_k)
        st.output({"candidates": len(fused)})

    if not fused:
        log_retrieval_trace(cleaned, embed_backend=embed_component,
                            dense_hits=dense_hits, bm25_hits=bm25_hits,
                            fused=[], mmr=[], reranked=[], final=[])
        return cleaned, []

    async with astage(db, "mmr", component="lexical_mmr",
                      conversation_id=conversation_id, request_id=request_id) as st:
        mmr_hits = await asyncio.to_thread(
            maximal_marginal_relevance, fused, top_k=settings.mmr_top_k
        )
        st.output({"candidates": len(mmr_hits)})

    rerank_input = mmr_hits[: settings.rerank_input_top_k]
    async with astage(db, "rerank", component="bge_reranker_v2_m3",
                      conversation_id=conversation_id, request_id=request_id) as st:
        reranked = await asyncio.to_thread(
            rerank, cleaned, rerank_input, settings.rerank_top_k
        )
        st.output({"kept": len(reranked)})

    reranked = _expand_split_lists(reranked)
    async with astage(db, "context_build", component="prompt_builder",
                      conversation_id=conversation_id, request_id=request_id) as st:
        final = await asyncio.to_thread(build_context, reranked)
        st.output({"chunks": len(final)})

    log_retrieval_trace(
        cleaned,
        embed_backend=embed_component,
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        fused=fused,
        mmr=mmr_hits,
        reranked=reranked,
        final=final,
    )
    return cleaned, final
