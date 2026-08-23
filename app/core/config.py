"""Central application settings, loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Sarvam ---
    sarvam_api_key: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai"
    sarvam_stt_model: str = "saaras:v3"
    sarvam_tts_model: str = "bulbul:v3"
    sarvam_llm_model: str = "sarvam-105b"

    # --- Postgres ---
    postgres_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/assamese_rag"
    )

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "assamese_doc_chunks"

    # --- OpenSearch ---
    opensearch_url: str = "http://localhost:9200"
    opensearch_index: str = "assamese_doc_chunks_bm25"
    opensearch_user: str = ""
    opensearch_password: str = ""

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Embeddings / reranker ---
    hf_token: str = ""
    bge_m3_model_path_or_endpoint: str = "BAAI/bge-m3"
    bge_m3_dim: int = 1024
    bge_reranker_model_path_or_endpoint: str = "BAAI/bge-reranker-v2-m3"
    embedding_batch_size: int = 8
    embedding_parallel_workers: int = 4

    # --- OCR ---
    ocr_primary_engine: str = "paddleocr"
    ocr_fallback_engine: str = "sarvam_vision"
    ocr_confidence_threshold: float = 0.60
    ocr_paddle_lang_indic: str = "devanagari"
    ocr_paddle_lang_latin: str = "en"

    # --- Chunking (P0/P2) ---
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 512
    chunk_min_tokens: int = 50
    chunk_overlap_tokens: int = 50
    unicode_quality_threshold: float = 0.45
    default_document_language: str = "as"

    # --- Retrieval (P3) ---
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_top_k: int = 20
    rrf_k: int = 60
    mmr_top_k: int = 20
    mmr_lambda: float = 0.7
    rerank_input_top_k: int = 20
    rerank_top_k: int = 5
    context_max_tokens: int = 3000
    conversation_memory_turns: int = 6
    debug_retrieval: bool = False

    # --- App ---
    app_env: str = "local"
    log_level: str = "INFO"
    max_upload_mb: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
