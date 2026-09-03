"""BGE-M3 embeddings + tokenizer for chunk sizing.

Backends (first match wins):
  1. HF_TOKEN set          -> Hugging Face Inference API (no local model weights)
  2. BGE_M3_* is http URL  -> custom /embed endpoint
  3. otherwise             -> local sentence-transformers (dev fallback only)

Token counting uses the cached BGE-M3 tokenizer when present; otherwise a
script-aware heuristic (no Hugging Face Hub access — avoids Docker DNS issues).
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

import httpx

from app.cleaner.unicode import indic_script_ratio, latin_script_ratio
from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_model = None
_tokenizer = None
_hf_client = None


class _ApproxTokenizer:
    """Offline fallback when BGE-M3 tokenizer files are not cached."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        indic = indic_script_ratio(text)
        latin = latin_script_ratio(text)
        # Indic scripts tend to tokenize denser than Latin in XLM-R.
        chars_per_token = 2.5 if indic >= latin else 4.0
        return list(range(max(1, int(len(text) / chars_per_token))))


def _hf_hub_cache_root() -> Path:
    for key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw)
            return p / "hub" if p.name != "hub" and (p / "hub").is_dir() else p
    return Path.home() / ".cache" / "huggingface" / "hub"


def _cached_tokenizer_available(name: str) -> bool:
    """True when tokenizer files are already on disk (no Hub round-trip)."""
    model_dir = _hf_hub_cache_root() / f"models--{name.replace('/', '--')}"
    if not model_dir.is_dir():
        return False
    return any(model_dir.rglob("tokenizer_config.json"))


def _load_cached_tokenizer(name: str):
    """Load tokenizer from local cache only — never contacts Hugging Face Hub."""
    from transformers import AutoTokenizer

    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline


def _model_id() -> str:
    return settings.bge_m3_model_path_or_endpoint


def _use_hf_api() -> bool:
    return bool(settings.hf_token)


def _is_custom_endpoint() -> bool:
    return settings.bge_m3_model_path_or_endpoint.startswith("http")


def _embedding_backend() -> str:
    if _is_custom_endpoint():
        return "custom_http"
    if _use_hf_api():
        return "hf_inference"
    return "local"


_HF_TIMEOUT_SECONDS = float(os.environ.get("HF_EMBED_TIMEOUT_SECONDS", "18"))


def _get_hf_client():
    global _hf_client
    if _hf_client is not None:
        return _hf_client
    with _lock:
        if _hf_client is None:
            from huggingface_hub import InferenceClient

            log.info(
                "bge_m3.hf_client",
                model=_model_id(),
                provider="hf-inference",
                timeout=_HF_TIMEOUT_SECONDS,
            )
            _hf_client = InferenceClient(
                provider="hf-inference",
                api_key=settings.hf_token,
                timeout=_HF_TIMEOUT_SECONDS,
            )
    return _hf_client


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            path = _model_id()
            log.info("bge_m3.load", path=path, backend="sentence_transformers")
            _model = SentenceTransformer(path)
    return _model


def get_tokenizer():
    """BGE-M3 tokenizer (vocab only) for chunk token counting."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _lock:
        if _tokenizer is None:
            name = _model_id() if not _is_custom_endpoint() else "BAAI/bge-m3"
            # HF Inference mode: never download tokenizer — heuristic is enough
            # for chunk sizing and avoids flaky Docker DNS to huggingface.co.
            if _use_hf_api() or not _cached_tokenizer_available(name):
                if not _use_hf_api() and not _cached_tokenizer_available(name):
                    log.warning(
                        "bge_m3.tokenizer_offline_fallback",
                        model=name,
                        reason="cache_miss",
                    )
                else:
                    log.info(
                        "bge_m3.tokenizer_offline_fallback",
                        model=name,
                        reason="hf_inference_api",
                    )
                _tokenizer = _ApproxTokenizer()
            else:
                _tokenizer = _load_cached_tokenizer(name)
                log.info("bge_m3.tokenizer", source="cache", model=name)
    return _tokenizer


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


def release_tokenizer() -> None:
    """Drop cached tokenizer to free RAM before a long embedding run."""
    global _tokenizer
    _tokenizer = None


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _to_vector(raw: Any) -> list[float]:
    """Convert HF feature_extraction output to a single L2-normalized vector."""
    import numpy as np

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr.mean(axis=0)
    else:
        raise ValueError(f"Unexpected embedding shape: {arr.shape}")
    return _l2_normalize(vec.tolist())


def _embed_one_hf(text: str) -> list[float]:
    from huggingface_hub.errors import HfHubHTTPError

    client = _get_hf_client()
    try:
        raw = client.feature_extraction(text, model=_model_id())
    except HfHubHTTPError as exc:
        log.error(
            "bge_m3.hf_error",
            status=exc.response.status_code if exc.response else None,
            detail=str(exc)[:500],
        )
        raise
    return _to_vector(raw)


def _embed_batch_hf_parallel(texts: list[str]) -> list[list[float]]:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub.errors import HfHubHTTPError

    workers = max(1, settings.embedding_parallel_workers)
    results: list[list[float] | None] = [None] * len(texts)

    def _one(idx: int, text: str) -> tuple[int, list[float]]:
        for attempt in range(10):
            try:
                return idx, _embed_one_hf(text)
            except HfHubHTTPError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    wait = min(30.0, 2.0 ** attempt)
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("HF inference unavailable after retries (503)")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i, t) for i, t in enumerate(texts)]
        for fut in as_completed(futures):
            idx, vec = fut.result()
            results[idx] = vec
    if any(v is None for v in results):
        raise RuntimeError("HF parallel embedding returned incomplete results")
    return results  # type: ignore[return-value]


def _embed_batch_hf(texts: list[str]) -> list[list[float]]:
    if settings.embedding_parallel_workers > 1 and len(texts) > 1:
        return _embed_batch_hf_parallel(texts)
    import time

    from huggingface_hub.errors import HfHubHTTPError

    out: list[list[float]] = []
    for text in texts:
        for attempt in range(10):
            try:
                out.append(_embed_one_hf(text))
                break
            except HfHubHTTPError as exc:
                if exc.response is not None and exc.response.status_code == 503:
                    wait = min(30.0, 2.0 ** attempt)
                    time.sleep(wait)
                    continue
                raise
        else:
            raise RuntimeError("HF inference unavailable after retries (503)")
    return out


def _embed_batch_remote(texts: list[str]) -> list[list[float]]:
    url = settings.bge_m3_model_path_or_endpoint.rstrip("/") + "/embed"
    resp = httpx.post(url, json={"texts": texts}, timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _embed_batch_local(texts: list[str]) -> list[list[float]]:
    model = _load_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


def _embed_batch(texts: list[str], *, skip_hf: bool = False, circuit: dict | None = None) -> list[list[float]]:
    backend = _embedding_backend()
    if backend == "custom_http":
        return _embed_batch_remote(texts)
    if backend == "hf_inference" and not skip_hf:
        try:
            return _embed_batch_hf(texts)
        except Exception as exc:
            log.error(
                "bge_m3.hf_fallback_local",
                error=str(exc)[:500],
                error_type=type(exc).__name__,
            )
            if circuit is not None:
                circuit["open"] = True
            return _embed_batch_local(texts)
    return _embed_batch_local(texts)


def embed_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    size = batch_size or settings.embedding_batch_size
    size = max(1, size)
    total = len(texts)
    out: list[list[float]] = []
    circuit: dict = {"open": False}

    for start in range(0, total, size):
        batch = texts[start : start + size]
        batch_vecs = _embed_batch(batch, skip_hf=circuit["open"], circuit=circuit)
        out.extend(batch_vecs)
        if total > size:
            log.info(
                "bge_m3.embed_batch",
                backend=_embedding_backend(),
                batch_start=start,
                batch_size=len(batch),
                total=total,
                circuit_open=circuit["open"],
            )

    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text], batch_size=1)[0]
