"""
ULI — Celery Worker
Handles embedding generation and reranking jobs asynchronously.
"""
from __future__ import annotations

import os
import logging

from celery import Celery

logger = logging.getLogger("uli.worker")

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1")
RESULT_URL = os.environ.get("CELERY_RESULT_URL", "redis://localhost:6379/2")

celery_app = Celery(
    "uli_worker",
    broker  = BROKER_URL,
    backend = RESULT_URL,
)

celery_app.conf.update(
    task_serializer          = "json",
    result_serializer        = "json",
    accept_content           = ["json"],
    timezone                 = "Asia/Kolkata",
    enable_utc               = True,
    worker_prefetch_multiplier = 1,     # One task at a time per worker (ML models)
    task_acks_late           = True,    # Ack after completion, not receipt
    task_routes = {
        "uli.worker.tasks.embed_section":   {"queue": "embedding"},
        "uli.worker.tasks.rerank_batch":    {"queue": "reranking"},
    },
)


@celery_app.task(name="uli.worker.tasks.embed_section", bind=True, max_retries=3)
def embed_section(self, section_id: str, text: str, act_id: str) -> dict:
    """
    Generate 1536-dim embedding for a section and upsert to Pinecone.
    Runs in worker to avoid blocking the API event loop.
    """
    import asyncio
    import openai

    try:
        client  = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        resp    = client.embeddings.create(
            input      = text[:8000],   # Truncate to model limit
            model      = "text-embedding-3-small",
            dimensions = 1536,
        )
        embedding = resp.data[0].embedding
        logger.info("Embedded section %s (%d dims)", section_id, len(embedding))
        return {"section_id": section_id, "embedding_dim": len(embedding), "status": "ok"}
    except Exception as exc:
        logger.error("Embedding failed for %s: %s", section_id, exc)
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@celery_app.task(name="uli.worker.tasks.rerank_batch", bind=True, max_retries=2)
def rerank_batch(self, query: str, candidates: list) -> list:
    """
    Rerank a batch of candidates using BGE-reranker-large.
    Returns sorted list with scores.
    """
    try:
        from sentence_transformers import CrossEncoder
        model  = CrossEncoder("BAAI/bge-reranker-large", max_length=512)
        pairs  = [(query, c.get("text", "")) for c in candidates]
        scores = model.predict(pairs, batch_size=len(pairs))
        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1], reverse=True
        )
        return [{"candidate": c, "score": float(s)} for c, s in ranked]
    except Exception as exc:
        logger.error("Rerank failed: %s", exc)
        raise self.retry(exc=exc, countdown=5)
