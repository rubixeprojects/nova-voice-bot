"""Cross-encoder reranking with BGE-reranker-v2-m3.

Backends (first match wins):
  1. BGE_RERANKER_* is http URL  -> custom /rerank endpoint
  2. HF_TOKEN set                -> Hugging Face Inference router (httpx)
  3. otherwise                   -> local CrossEncoder (sentence-transformers)
"""
from __future__ import annotations

import threading
from typing import Any

import httpx
import os
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk

log = get_logger(__name__)

_lock = threading.Lock()
_reranker = None

_HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference"
_HF_RERANK_TIMEOUT_SECONDS = float(os.environ.get("HF_RERANK_TIMEOUT_SECONDS", "18"))

def _is_endpoint() -> bool:
    return settings.bge_reranker_model_path_or_endpoint.startswith("http")


def _use_hf_api() -> bool:
    return bool(settings.hf_token) and not _is_endpoint()


def _model_id() -> str:
    return settings.bge_reranker_model_path_or_endpoint


def _load():
    global _reranker
    if _reranker is not None:
        return _reranker
    with _lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            path = _model_id()
            log.info("reranker.load", path=path, backend="sentence_transformers")
            _reranker = CrossEncoder(path)
    return _reranker


def _parse_hf_scores(raw: Any, n: int) -> list[float]:
    """Parse HF sequence-classification / reranker responses."""
    # Batch response: [[{"label": "...", "score": 0.9}, ...]]
    if (
        isinstance(raw, list)
        and raw
        and isinstance(raw[0], list)
        and raw[0]
        and isinstance(raw[0][0], dict)
        and "score" in raw[0][0]
    ):
        return [float(item["score"]) for item in raw[0][:n]]

    if isinstance(raw, dict):
        if "scores" in raw:
            raw = raw["scores"]
        elif "logits" in raw:
            raw = raw["logits"]

    if not isinstance(raw, list):
        raise ValueError(f"Unexpected rerank response type: {type(raw)}")
    if not raw:
        return [0.0] * n

    if isinstance(raw[0], dict) and "score" in raw[0]:
        return [float(item["score"]) for item in raw[:n]]
    if isinstance(raw[0], (int, float)):
        return [float(x) for x in raw[:n]]
    if isinstance(raw[0], list):
        out: list[float] = []
        for item in raw[:n]:
            if isinstance(item, list) and item:
                if isinstance(item[0], dict):
                    out.append(float(item[0]["score"]))
                else:
                    out.append(float(item[0]))
            else:
                out.append(float(item))
        return out
    return [float(x) for x in raw[:n]]


def _scores_hf(query: str, texts: list[str]) -> list[float]:
    """Score query–passage pairs via HF Inference router."""
    import time

    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

    model = _model_id()
    url = f"{_HF_INFERENCE_BASE}/models/{model}"
    headers = build_hf_headers(token=settings.hf_token)
    payload = {
        "inputs": [{"text": query, "text_pair": text} for text in texts],
    }
    session = get_session()

    for attempt in range(10):
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=_HF_RERANK_TIMEOUT_SECONDS)
            hf_raise_for_status(resp)
            return _parse_hf_scores(resp.json(), len(texts))
        except HfHubHTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 503:
                wait = min(30.0, 2.0 ** attempt)
                log.warning("reranker.hf_loading", attempt=attempt, wait_s=round(wait, 1))
                time.sleep(wait)
                continue
            log.warning("reranker.hf_failed", status=status, error=str(exc)[:300])
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("reranker.hf_failed", error=str(exc)[:300])
            break

    log.info("reranker.hf_fallback_local")
    return _scores_local(query, texts)


def _scores_local(query: str, texts: list[str]) -> list[float]:
    import time
    t0 = time.perf_counter()
    model = _load()
    t1 = time.perf_counter()
    log.info("reranker.local_load_time", seconds=round(t1 - t0, 2))
    pairs = [[query, t] for t in texts]
    raw = model.predict(pairs, show_progress_bar=False)
    t2 = time.perf_counter()
    log.info("reranker.local_predict_time", seconds=round(t2 - t1, 2), n_pairs=len(pairs))
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, float):
        raw = [raw]
    return [float(x) for x in raw]


def _scores(query: str, texts: list[str]) -> list[float]:
    if _is_endpoint():
        url = settings.bge_reranker_model_path_or_endpoint.rstrip("/") + "/rerank"
        resp = httpx.post(url, json={"query": query, "texts": texts}, timeout=120)
        resp.raise_for_status()
        return resp.json()["scores"]
    if _use_hf_api():
        return _scores_hf(query, texts)
    return _scores_local(query, texts)


def rerank(
    query: str, candidates: list[RetrievedChunk], top_k: int | None = None
) -> list[RetrievedChunk]:
    top_k = top_k or settings.rerank_top_k
    if not candidates:
        return []
    backend = "custom_http" if _is_endpoint() else ("hf_inference" if _use_hf_api() else "local")
    scores = _scores(query, [c.text for c in candidates])
    for c, s in zip(candidates, scores):
        c.score = float(s)
        c.source = "reranked"
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    log.info(
        "rerank.done",
        backend=backend,
        candidates=len(candidates),
        kept=min(top_k, len(ranked)),
    )
    return ranked[:top_k]
