# Assamese RAG — System Architecture

Production-grade retrieval-augmented generation (RAG) for **Assamese and mixed Assamese–English documents**. The system handles government circulars, legal PDFs, books, OCR scans, Q&A datasets, and table-heavy documents through a unified ingestion and retrieval pipeline.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Infrastructure & Services](#infrastructure--services)
3. [Ingestion Pipeline (P0–P4)](#ingestion-pipeline-p0p4)
4. [Retrieval Pipeline (P3)](#retrieval-pipeline-p3)
5. [Chat & LLM Layer](#chat--llm-layer)
6. [Voice Layer](#voice-layer)
7. [Data Model & Storage](#data-model--storage)
8. [Configuration Reference](#configuration-reference)
9. [Observability & Evaluation](#observability--evaluation)
10. [Deployment](#deployment)
11. [Module Map](#module-map)

---

## High-Level Overview

```
┌─────────────┐     upload PDF      ┌──────────┐     Celery task      ┌─────────────────────────────┐
│   Client    │ ──────────────────► │ FastAPI  │ ──────────────────► │ Worker: Ingestion Pipeline  │
│  (Swagger)  │                     │   API    │                     │  P0→P1→P2→P4→Embed→Index    │
└─────────────┘                     └──────────┘                     └──────────────┬──────────────┘
       │                                  │                                         │
       │ chat query                       │                                         ▼
       │                                  │                          ┌──────────────────────────┐
       └──────────────────────────────────┼─────────────────────────►│ Postgres │ Qdrant │ OS  │
                                          │                          └──────────────────────────┘
                                          ▼
                               ┌─────────────────────────────┐
                               │ Retrieval: Dense ‖ BM25     │
                               │ → RRF → MMR → Rerank        │
                               │ → Context Builder → Sarvam  │
                               └─────────────────────────────┘
```

**Design goals**

| Goal | Approach |
|------|----------|
| Any document type | Type detection + routed parsers (PyMuPDF, OCR, pdfplumber tables) |
| Assamese script quality | NFC normalization, OCR corruption repair, quality gating |
| Semantic structure | Layout blocks → sections → adaptive chunks |
| Hybrid retrieval | BGE-M3 dense + Indic BM25, fused with RRF |
| Answer quality | MMR diversity, cross-encoder rerank, ordered context assembly |
| Production ops | Async Celery ingestion, stage logging, HF Inference API for ML |
| Real-time voice | WS server with TEN VAD + adaptive noise gating, barge-in cancellation, sentence-level TTS streaming — reuses the same RAG pipeline via the text SSE endpoint |

---

## Infrastructure & Services

| Service | Role | Port (local) |
|---------|------|--------------|
| **FastAPI (`api`)** | REST + SSE chat, document upload | 8000 |
| **Celery (`worker`)** | Background ingestion (parse → embed → index) | — |
| **Voice WS Server** (`voice_ws_server.py`) | Real-time voice session: VAD, barge-in, STT→RAG→TTS orchestration | 8766 (`VOICE_WS_PORT`) |
| **PostgreSQL** | Documents, chunks, conversations, pipeline logs | 5432 |
| **Qdrant** | Dense vectors (1024-dim, cosine) | 6333 |
| **OpenSearch** | BM25 with ICU analyzer for Indic scripts | 9200 |
| **Redis** | Celery broker/backend | 6379 |
| **MinIO** | S3-compatible object storage for PDFs | 9000 |
| **Hugging Face Inference** | BGE-M3 embeddings + BGE reranker (when `HF_TOKEN` set) | external |
| **Sarvam AI** | LLM (`sarvam-30b`), STT (`saaras:v3`), TTS, Vision OCR fallback — all via `app/llm/sarvam_client.py` | external |

Orchestration: `docker compose up` runs migrate → api + worker + data stores. The voice WS server runs as a separate lightweight process (`python voice_ws_server.py`) alongside the API — it is a client of the FastAPI `/chat/text/stream` endpoint, not a Celery/Compose service itself in the current setup.

---

## Ingestion Pipeline (P0–P4)

Entry point: `app.pipeline.ingest.run_ingestion_pipeline(pdf_bytes)`  
Worker orchestration: `app.workers.tasks.ingest_document`

### Flow

```
PDF bytes
    │
    ▼
┌───────────────────┐
│ P1: Type Detection│  digital_pdf | scanned_pdf | mixed | table_heavy
│  parser/router.py │
└─────────┬─────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
PyMuPDF      PaddleOCR / Sarvam Vision
(pymupdf)    (ocr.py) — merged per-page
    │           │
    └─────┬─────┘
          ▼
┌───────────────────┐
│ P0: Unicode Clean │  NFC, garbage strip, OCR repairs, quality_score
│  cleaner/unicode  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ P1: Header Strip  │  repeating headers/footers/page numbers
│  cleaner/headers  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ P4: Table Extract │  pdfplumber → markdown table blocks
│  parser/tables    │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ P2: Layout Blocks │  paragraph | heading | list | qa | table
│  layout/blocks    │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ P2: Sections      │  heading tree, page spans
│  layout/sections  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ P0/P2: Chunking   │  adaptive 350–450 tok target, max 512, overlap 50
│  chunker/semantic │  Q&A pair preservation, dedup, noise filter
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ P4: BGE-M3 Embed  │  HF API (parallel) or local fallback
│  ingestion/embed  │
└─────────┬─────────┘
          ▼
    Postgres + Qdrant + OpenSearch
```

### P0 — Unicode & Chunk Quality

**Module:** `app/cleaner/unicode.py`, `app/chunker/dedup.py`

- **NFC normalization** for Bengali–Assamese conjuncts
- Strip control/invisible chars and known OCR garbage (e.g. `Ł`, combining marks)
- **`quality_score`** (0–1): indic ratio, garbage penalty, mixed-script tolerance
- **`is_embeddable`**: gate chunks below `UNICODE_QUALITY_THRESHOLD` (default 0.45)
- **Dedup**: SHA-256 text hash
- **Merge tiny chunks** (< `CHUNK_MIN_TOKENS`) into same-section neighbor
- **Noise filter**: drop empty, low-quality, or oversized chunks

### P1 — Parsing & Routing

**Modules:** `app/parser/router.py`, `pymupdf_parser.py`, `app/ingestion/ocr.py`

| Document type | Parser strategy |
|---------------|-----------------|
| `digital_pdf` | PyMuPDF text layer + font-size headings |
| `scanned_pdf` | OCR primary (PaddleOCR `lang=bn`, pdf text layer first) |
| `mixed` | Per-page merge: PyMuPDF where indic ratio ≥ 0.15, else OCR |
| `table_heavy` | pdfplumber tables merged as markdown blocks |

**OCR fallback chain:** PDF text layer → PaddleOCR → Sarvam Vision (configurable)

### P2 — Layout & Semantic Sections

**Modules:** `app/layout/blocks.py`, `app/layout/sections.py`

- Classify blocks: `paragraph`, `heading`, `list`, `qa`, `table`, `caption`
- Build **section tree** from headings with `heading_path` breadcrumbs
- Chunker respects section boundaries — never merges across sections

### P0/P2 — Adaptive Chunking

**Module:** `app/chunker/semantic.py`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CHUNK_TARGET_TOKENS` | 400 | Greedy pack target |
| `CHUNK_MAX_TOKENS` | 512 | Hard ceiling |
| `CHUNK_OVERLAP_TOKENS` | 50 | Context continuity |
| `CHUNK_MIN_TOKENS` | 50 | Merge threshold |

- Token counting via BGE-M3 tokenizer (`embedding.count_tokens`)
- Q&A documents: question+answer kept as atomic units
- Lists and tables chunked as distinct block types

### P4 — Performance & Tables

- **Parallel HF embedding**: `ThreadPoolExecutor` with `EMBEDDING_PARALLEL_WORKERS` (default 4)
- **Batched API calls**: `EMBEDDING_BATCH_SIZE` chunks per batch
- **Table extraction**: `pdfplumber` → markdown tables appended to page blocks
- **Worker memory**: `release_tokenizer()` before embedding to reduce RAM

### Ingestion Worker Stages

`app/workers/tasks.py` records each stage in `pipeline_stage_logs`:

| Stage | Component | Output |
|-------|-----------|--------|
| `parse` | `document_router` | pages, sections, chunk drafts |
| `embedding` | `hf_inference` / `bge_m3` | 1024-dim vectors |
| `index_write` | `qdrant`, `opensearch` | indexed points/docs |

---

## Retrieval Pipeline (P3)

Entry point: `app.retrieval.pipeline.retrieve`

### Flow

```
User query
    │
    ▼
Unicode clean (same P0 cleaner)
    │
    ▼
BGE-M3 query embedding (HF API or local)
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  │
Qdrant dense       OpenSearch BM25        │  parallel
(top 20)           (top 20, ICU analyzer)│
    │                  │                  │
    └────────┬─────────┘                  │
             ▼                            │
      RRF fusion (top 20)                 │
             ▼                            │
      MMR lexical diversity (top 20)      │  Jaccard on tokens, λ=0.7
             ▼                            │
      Cross-encoder rerank (top 5)        │  BGE-reranker-v2-m3
             ▼                            │
      Context builder (≤3000 tokens)      │  sort by page, merge adjacent
             ▼                            │
      Sarvam LLM prompt                   │
```

### Stage Details

| Stage | Module | Notes |
|-------|--------|-------|
| Dense search | `app/retrieval/dense.py` | Cosine similarity, user_id + document_id filters |
| BM25 search | `app/retrieval/bm25.py` | `assamese_icu` analyzer |
| RRF fusion | `app/retrieval/fusion.py` | `RRF_K=60`, merges dense + BM25 ranks |
| MMR | `app/retriever/mmr.py` | Reduces near-duplicate passages |
| Rerank | `app/retrieval/rerank.py` | HF API or local CrossEncoder |
| Context | `app/prompt_builder/context.py` | Reading order, adjacent merge, token cap |
| Debug trace | `app/retriever/debug.py` | Full rank list when `DEBUG_RETRIEVAL=true` |

### Embedding & Reranker Backends

Both follow the same priority:

1. Custom HTTP endpoint (if model path starts with `http`)
2. **Hugging Face Inference API** (if `HF_TOKEN` set)
3. Local `sentence-transformers` (dev fallback)

This avoids loading ~2 GB BGE-M3 and reranker weights in Docker containers.

---

## Chat & LLM Layer

**Endpoints:** `app/api/chat.py`

| Route | Mode |
|-------|------|
| `POST /api/v1/chat/text` | Synchronous text chat |
| `POST /api/v1/chat/text/stream` | SSE token streaming |
| `POST /api/v1/chat/voice` | STT → retrieve → answer |
| `POST /api/v1/chat/voice/stream` | SSE with transcript + tokens |

**LLM:** Sarvam `sarvam-30b` via `app/llm/sarvam_client.py`  
**Prompts:** `app/llm/prompts.py` — Assamese-only answers, no chain-of-thought, `[S#]` citations

**Conversation memory:** last `CONVERSATION_MEMORY_TURNS` turns (default 6)

> **Note:** `app/llm/sarvam_client.py` also exposes `transcribe()` (STT) and `text_to_speech()` (TTS), used by the Voice Layer below. The real-time voice path does **not** go through `POST /api/v1/chat/voice` — it runs as a separate WebSocket service that calls `POST /api/v1/chat/text/stream` directly (transcript in, SSE tokens out), so it reuses the exact same retrieval + LLM pipeline as text chat.

---

## Voice Layer

Real-time, barge-in-capable voice conversation. Runs as a **standalone WebSocket service** (`voice_ws_server.py`), separate from the FastAPI app — it orchestrates VAD, Sarvam STT/TTS, and calls the existing RAG text-streaming endpoint rather than duplicating retrieval logic.

**Entry point:** `voice_ws_server.py` (`ws_handler` → one `VoiceSession` per connection)
**Client:** `voice_component.py` — a Streamlit-embedded HTML/JS widget (mic capture + audio playback)
**Config:** `voice_config.py` (`VoiceConfig`)

### Topology

```
Browser (voice_component.py)
    │ getUserMedia (16kHz, mono, echo cancel + noise suppress)
    │ ScriptProcessorNode → Float32 → PCM16
    │ raw binary frames, continuous, over WS
    ▼
┌───────────────────────────────────────────────────┐
│ Voice WS Server  ws://VOICE_WS_HOST:VOICE_WS_PORT  │
│                                                     │
│  VoiceSession.state:                               │
│   IDLE → LISTENING → PROCESSING → SPEAKING ─┐       │
│           ▲                                 │       │
│           └─────────── barge-in ────────────┘       │
│                                                     │
│  ┌────────────┐      ┌───────────────────────┐    │
│  │ TEN VAD    │      │ AmplitudeGate          │    │
│  │ (speech    │◄────►│ (adaptive noise floor, │    │
│  │  segments) │      │  close-talk gate)      │    │
│  └─────┬──────┘      └───────────────────────┘    │
│        ▼ confirmed utterance                       │
│  InterruptController.new_turn()                    │
│  (cancels any in-flight turn's asyncio task)        │
│        ▼                                            │
│  1. Sarvam STT  transcribe(wav, language_code)      │
│  2. POST /api/v1/chat/text/stream (SSE, reused)     │
│  3. Buffer deltas → split into complete sentences   │
│  4. Sarvam TTS per completed sentence                │
│  5. base64 WAV → client (streamed, one sentence      │
│     at a time — not a wait for the full answer)      │
└───────────────────────────────────────────────────┘
    │ JSON: state / transcript / answer_delta / answer
    │ JSON: audio (base64 WAV, is_final)
    ▼
Browser: queued sequential playback (Web Audio API).
`stop_playback` cmd from server clears the queue instantly on barge-in.
```

### Session State Machine

| State | Meaning |
|-------|---------|
| `IDLE` | First ~1.5s of audio after connect — used only to seed the ambient noise floor (`AmplitudeGate.force_seed_floor`), not yet listening for utterances |
| `LISTENING` | Accepting mic audio; VAD assembles speech segments; noise floor keeps updating while nobody is speaking |
| `PROCESSING` | An utterance was confirmed; STT is running (brief) |
| `SPEAKING` | Answer is streaming from the RAG endpoint and being spoken sentence-by-sentence; VAD stays active in this state to detect barge-in |

### Voice Activity Detection & Barge-in

- **VAD engine:** [TEN VAD](https://github.com/TEN-framework/ten-vad) (`ten_vad` / `TenVad`), run on 16ms hops over 16kHz PCM16.
- **Utterance confirmation:** speech starts once `VAD_PREFIX_PADDING_MS` (240ms) of consecutive hops are all above `VAD_THRESHOLD` (0.7); ends once `VAD_SILENCE_DURATION_MS` (1000ms) of hops all fall below threshold. A rolling `recent_audio` buffer captures the pre-roll so the confirmed prefix isn't clipped.
- **Safety valve:** a speech buffer is force-flushed at 8s to avoid unbounded buffering in a noisy room that never produces a clean silence gap.
- **`AmplitudeGate`:** tracks an exponentially-averaged ambient noise floor (dB RMS) and requires a segment to be a configurable margin above it before treating it as real speech — filters keyboard clicks, background chatter, or media playing nearby.
  - Initial utterance: requires **12dB** above the floor to be treated as close-talking.
  - **Barge-in specifically requires 20dB** above the floor (stricter), plus a 400ms grace period after the bot starts speaking, before it's accepted as a genuine interrupt — this avoids the bot's own audio bleed or ambient noise falsely cutting off playback.
- **Cancellation:** `InterruptController` assigns each confirmed utterance a `turn_id` and cancels the previous turn's asyncio task outright. Every step of the pipeline (`speak()`, the SSE read loop, the final playback wait) re-checks `turn_id` before acting, so a barge-in cleanly aborts in-flight STT/LLM/TTS work rather than racing it.
- On barge-in, the server immediately sends a `{"type":"cmd","name":"stop_playback"}` message so the client clears its audio queue and stops the current source — this is what makes the cutoff feel instant on the client side, not just server-side.

### Turn Pipeline (per confirmed utterance)

1. **STT:** the buffered utterance is packaged as a WAV (`pcm_to_wav`) and sent whole to Sarvam (`transcribe`, batch — not streaming STT) with the session's pinned `language_code`.
2. Transcript is sent to the client immediately (`{"type":"transcript"}`) so the UI can show it before the answer arrives.
3. The transcript is POSTed to the **existing** `POST /api/v1/chat/text/stream` SSE endpoint (`X-User-Id`, `X-Language` headers) — same retrieval/rerank/LLM pipeline as text chat, no separate voice RAG path.
4. As SSE `delta` tokens arrive, they're appended to a sentence buffer and split on sentence boundaries (regex covers `.!?` and Assamese/Bengali `।`). Each completed sentence — held back by one, so the true last sentence can be flagged `is_final` — is sent to TTS as soon as it's ready, **not** after the full answer completes. This keeps time-to-first-audio low.
5. **TTS:** each sentence goes through Sarvam (`text_to_speech(text, lang_code)`) after stripping `[S#]` citation markers (`strip_source_markers` / `clean_answer` — citations are useful in text but not spoken). Each result is base64-encoded and pushed to the client as its own `{"type":"audio"}` message.
6. Once the full answer is assembled, a final cleaned `{"type":"answer"}` message is sent, and the server waits (bounded by total spoken-audio duration + 5s) for a `playback_finished` ack from the client before returning to `LISTENING`.

### Language Handling

The client sends `{"type":"cmd","name":"set_language","language":"as-IN"}` (one of `en-IN` / `hi-IN` / `as-IN` / `kn-IN`); the server **pins** `VoiceSession.selected_language` for the rest of the connection and uses it for both STT and TTS calls. Default is `en_IN`. Unlike the text-chat language toggle, this is per-connection session state on the WS server, not a per-request header from the client each turn.

### Config (`voice_config.py`)

| Setting | Source | Default | Notes |
|---------|--------|---------|-------|
| `WS_HOST` / `WS_PORT` | env: `VOICE_WS_HOST` / `VOICE_WS_PORT` | `0.0.0.0` / `8766` | |
| `RAG_API_URL` | env: `RAG_API_URL` | `http://localhost:8000/api/v1/chat/text` | `/text` suffix swapped for `/text/stream` at call time |
| `SAMPLE_RATE` / `CHANNELS` / `SAMPLE_WIDTH` | hardcoded | 16000 / 1 / 2 (16-bit) | must match the frontend capture settings exactly |
| `VAD_HOP_SIZE_MS` / `VAD_PREFIX_PADDING_MS` / `VAD_SILENCE_DURATION_MS` / `VAD_THRESHOLD` | hardcoded | 16 / 240 / 1000 / 0.7 | mirrors `ten_vad_python/config.py` |
| `UNIVERSAL_USER_ID` | env, **required** | — | read directly via `os.environ[...]`; server raises if unset |

---

## Data Model & Storage

### PostgreSQL (`app/models/tables.py`)

**documents**
- `status`: `uploaded` → `ocr_in_progress` → `structuring` → `embedding` → `ready` | `failed`
- `document_type`: from ingestion profile
- `page_count`, S3 location, user ownership

**chunks** (migration `0002_chunk_metadata_v2`)
- Core: `chunk_index`, `section_id`, `section_title`, `token_count`
- v2 metadata: `page_start`, `page_end`, `heading_path` (JSONB), `block_type`, `quality_score`, `language`, `text`
- Graph: `prev_chunk_id`, `next_chunk_id`
- Index refs: `qdrant_point_id`, `opensearch_doc_id`

### Qdrant Payload

Each point stores the full retrieval payload:

```json
{
  "chunk_id", "document_id", "user_id", "chunk_index",
  "section_id", "section_title", "heading_path",
  "page_start", "page_end", "page_number",
  "block_type", "document_type", "quality_score",
  "prev_chunk_id", "next_chunk_id", "token_count",
  "source_language", "ocr_confidence", "source_file", "text"
}
```

Payload indexes: `user_id`, `document_id`, `block_type`, `document_type`

### OpenSearch Mapping

Same fields as Qdrant payload; `text` uses `assamese_icu` analyzer (ICU plugin).

---

## Configuration Reference

See `.env.example` for the full list. Key groups:

| Group | Variables |
|-------|-----------|
| HF Inference | `HF_TOKEN`, `BGE_M3_*`, `BGE_RERANKER_*`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_PARALLEL_WORKERS` |
| Chunking | `CHUNK_TARGET_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_MIN_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `UNICODE_QUALITY_THRESHOLD` |
| Retrieval | `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_TOP_K`, `MMR_TOP_K`, `MMR_LAMBDA`, `RERANK_INPUT_TOP_K`, `RERANK_TOP_K`, `CONTEXT_MAX_TOKENS` |
| OCR | `OCR_PRIMARY_ENGINE`, `OCR_FALLBACK_ENGINE`, `OCR_CONFIDENCE_THRESHOLD` |
| Voice | `VOICE_WS_HOST`, `VOICE_WS_PORT`, `RAG_API_URL`, `UNIVERSAL_USER_ID` (VAD thresholds are currently hardcoded in `voice_config.py`, not env-driven) |
| Debug | `DEBUG_RETRIEVAL`, `LOG_LEVEL` |

---

## Observability & Evaluation

### Pipeline Stage Logging

Every ingestion and retrieval stage writes to `pipeline_stage_logs` with:
- `request_id`, `document_id` / `conversation_id`
- `stage_name`, `component`, `status`, `duration_ms`
- `input_summary`, `output_summary` (JSONB)

### Retrieval Debug

Set `DEBUG_RETRIEVAL=true` to log ranked previews at each stage (dense → bm25 → fused → mmr → reranked → final).

### Evaluation Harness

**Module:** `app/evaluation/runner.py`

Metrics: Recall@5, Recall@10, MRR, hit rate, latency  
Usage: provide JSON eval cases with `query` + `expected_chunk_ids`, pass a `retrieve_fn`.

---

## Deployment

### Local Development

```bash
cp .env.example .env   # set HF_TOKEN, SARVAM_API_KEY
docker compose up --build
# Swagger: http://localhost:8000/docs
# Qdrant UI: http://localhost:6333/dashboard
```

### Voice WS Server

Runs separately from Compose — a standalone `websockets` process that talks to the API over HTTP:

```bash
export UNIVERSAL_USER_ID=<user-id>          # required, no default
export RAG_API_URL=http://localhost:8000/api/v1/chat/text
python voice_ws_server.py
# listens on ws://0.0.0.0:8766 by default (VOICE_WS_PORT)
```

Requires the FastAPI `api` service already running (it calls `/api/v1/chat/text/stream`). Additional Python deps beyond the core app: `websockets`, `aiohttp`, `ten_vad`, `numpy`. The Streamlit client (`voice_component.py`) connects to `ws://localhost:{VOICE_WS_PORT}` and needs browser mic permission.

### Migrations

```bash
docker compose run migrate   # alembic upgrade head
```

Migration `0002_chunk_metadata_v2` adds chunk metadata columns and `documents.document_type`.

### Re-ingestion

After pipeline upgrades, re-upload documents or re-trigger ingestion so chunks carry v2 metadata in Qdrant/OpenSearch.

### Memory Notes

| Component | With HF_TOKEN | Without HF_TOKEN |
|-----------|---------------|------------------|
| Worker | ~2–4 GB (OCR + batched HF calls) | ~6–8 GB (local BGE-M3) |
| API | ~512 MB–1 GB (HF rerank) | ~4 GB (local reranker cold load) |

---

## Module Map

```
app/
├── api/                    # FastAPI routes (chat, documents, health)
├── cleaner/                # P0: unicode.py, headers.py
├── parser/                 # P1/P4: router, pymupdf_parser, tables
├── layout/                 # P2: blocks, sections
├── chunker/                # P0/P2: semantic, dedup
├── pipeline/               # Orchestrator: ingest.py, types.py
├── ingestion/              # Legacy + embedding, ocr (still used by pipeline)
├── workers/                # Celery tasks
├── retrieval/              # dense, bm25, fusion, rerank, pipeline
├── retriever/              # P3: mmr, debug
├── prompt_builder/         # P3: context assembly
├── evaluation/             # P3: benchmark runner
├── llm/                    # Sarvam client, prompts
├── core/                   # config, db, clients, storage, logging
├── models/                 # SQLAlchemy tables
└── db/migrations/          # Alembic (0001 initial, 0002 chunk metadata v2)

# Voice service — separate process, sits alongside app/ (not inside it)
voice_ws_server.py           # WS server: VoiceSession, TEN VAD, AmplitudeGate,
                              #   InterruptController, STT→RAG-stream→TTS pipeline
voice_config.py               # VoiceConfig: WS host/port, audio format, VAD constants
voice_component.py            # Streamlit-embedded HTML/JS client: mic capture,
                              #   PCM16 streaming, queued WAV playback, barge-in UI
```

### Priority Implementation Summary (P0–P4)

| Priority | Scope | Status |
|----------|-------|--------|
| **P0** | Unicode clean, quality gating, dedup, adaptive chunking | ✅ |
| **P1** | Document routing, PyMuPDF parser, header strip, OCR merge | ✅ |
| **P2** | Layout blocks, semantic sections, section-aware chunking | ✅ |
| **P3** | MMR, rerank HF, context builder, debug trace, eval harness | ✅ |
| **P4** | Parallel HF embedding, pdfplumber tables, schema migration | ✅ |
| **Voice** | Real-time WS voice: TEN VAD, adaptive noise gating, barge-in cancellation, sentence-streamed Sarvam TTS, reuses text SSE pipeline | ✅ |

---

## Request Lifecycle Example

1. **Upload** `buhi-air-xadhu.pdf` → stored in MinIO, Postgres row `status=uploaded`
2. **Worker** runs `ingest_document` → profile `digital_pdf`, 138 chunks
3. **Embed** via HF Inference API (~9 min for 138 chunks with batching)
4. **Index** 138 Qdrant points + 138 OpenSearch docs → `status=ready`
5. **Chat** "এই গ্ৰন্থৰ বিষয়ে কি?" → retrieve → 5 reranked chunks → Sarvam answer with `[S1]` citations

---

*Last updated: implementation of enterprise redesign P0–P4, plus the real-time Voice Layer (WS server, VAD/barge-in, Sarvam STT+TTS).*
