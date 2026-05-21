"""
ULI — FastAPI Application
Endpoints: /query, /health, /ready, /metrics
Wires together all agents, DB clients, and budget manager.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from uli.models import FinalOutput, OutputMode
from uli.utils.metrics import (
    metrics_output,
    record_human_review,
    record_loop_back,
    record_quarantine,
    record_request_duration,
    record_token_spend,
    update_average_confidence,
)
from uli.utils.token_budget import TokenBudgetManager

logger = logging.getLogger("uli.api")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION STATE (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────

_engine = None          # InductiveReasoningEngine
_ready  = False         # flipped True when all deps are healthy


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _ready
    logger.info("ULI API starting — initialising dependencies…")
    try:
        _engine = await _build_engine()
        _ready  = True
        logger.info("ULI API ready.")
    except Exception as e:
        logger.error("Startup failed: %s", e)
        _ready = False
    yield
    logger.info("ULI API shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

async def _build_engine():
    """Wire all components. Falls back to mock mode if env vars absent."""
    import openai

    from uli.agents.dna_parser import BranchRouterAgent, LegalDNAParser
    from uli.agents.pipeline import (
        AnalystAgent, InductiveReasoningEngine,
        ResearcherAgent, ScribeAgent, ValidatorAgent,
    )
    from uli.db.njdg_client import AuditLogger, LiveVeracityDB, NJDGClient

    use_mock = os.environ.get("USE_MOCK_NJDG", "true").lower() == "true"

    llm_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    # NJDG
    if use_mock:
        from unittest.mock import AsyncMock, MagicMock
        from uli.models import CaseStatus
        from uuid import uuid4
        njdg = MagicMock()
        njdg.get_case_status   = AsyncMock(return_value=CaseStatus(
            case_id=uuid4(), citation_key="MOCK", status="decided", is_recent_sc=True
        ))
        njdg.get_act_amendments = AsyncMock(return_value=[])
        njdg.update_veracity    = AsyncMock(return_value=True)
    else:
        njdg = NJDGClient()

    # DB / Redis — use mocks when no real connection configured
    from unittest.mock import AsyncMock, MagicMock
    db_pool = MagicMock()
    db_pool.execute  = AsyncMock(return_value=None)
    db_pool.fetchrow = AsyncMock(return_value=None)

    redis_client = MagicMock()
    redis_client.get   = AsyncMock(return_value=None)
    redis_client.setex = AsyncMock(return_value=True)

    audit   = AuditLogger(db_pool=db_pool)
    budget  = TokenBudgetManager(db_pool=db_pool)

    # KnowledgePrism — mock ES / Pinecone / Neo4j when USE_MOCK_NJDG=true
    if use_mock:
        from uli.models import RetrievalResult, VerifiedCitation, Court
        prism = MagicMock()
        prism.retrieve = AsyncMock(return_value=RetrievalResult(
            verified_citations=[
                VerifiedCitation(
                    section_id="mock-s1",
                    citation_key="(2023) 1 SCC 1",
                    score=0.99,
                    text="Mock legal provision for development mode.",
                    act_id="mock-act-001",
                    landmark_flag=False,
                    court=Court.SUPREME_COURT,
                    year=2023,
                )
            ],
            quarantined=[],
            trigger_loop_back=False,
        ))
    else:
        from elasticsearch import AsyncElasticsearch
        from neo4j import AsyncGraphDatabase
        from uli.db.pinecone_config import get_pinecone_index
        from uli.retrieval.knowledge_prism import CrossEncoderReranker, KnowledgePrism
        es      = AsyncElasticsearch(os.environ.get("ES_URL", "http://elasticsearch:9200"))
        pc      = get_pinecone_index()
        neo4j   = AsyncGraphDatabase.driver(os.environ.get("NEO4J_URL", "bolt://localhost:7687"))
        reranker = CrossEncoderReranker()
        prism   = KnowledgePrism(es, pc, neo4j, reranker)

    return InductiveReasoningEngine(
        parser     = LegalDNAParser(openai_client=llm_client),
        router     = BranchRouterAgent(),
        researcher = ResearcherAgent(prism=prism),
        analyst    = AnalystAgent(openai_client=llm_client),
        validator  = ValidatorAgent(njdg_client=njdg, audit_logger=audit),
        scribe     = ScribeAgent(openai_client=llm_client),
        budget_mgr = budget,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Universal Legal Intelligence (ULI)",
    description = "Supreme Court of India — AI-powered legal reasoning engine",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:       str
    output_mode: OutputMode = OutputMode.JUDGMENT


class QueryResponse(BaseModel):
    content:                str
    mode:                   str
    verified_citations:     int
    quarantined_citations:  int
    loops_taken:            int
    total_tokens_used:      int
    human_review_required:  bool
    citation_needed:        list
    conflict_log:           list
    error:                  Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Main legal reasoning endpoint. Runs the 4-agent pipeline."""
    if not _ready or _engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    t0 = time.perf_counter()
    try:
        output: FinalOutput = await _engine.reason(req.query, req.output_mode)
    except Exception as e:
        logger.exception("Engine error for query: %s", req.query[:80])
        raise HTTPException(status_code=500, detail=str(e))

    duration = time.perf_counter() - t0
    meta     = output.metadata

    # ── Emit metrics ──────────────────────────────────────────────────────────
    record_request_duration(req.output_mode.value, duration)

    if meta:
        for cn in output.citation_needed:
            record_quarantine(cn.reason or "unknown")
        if meta.loops_taken > 0:
            record_loop_back()
        if meta.human_review_required:
            record_human_review()
        for phase, tokens in (_engine.budget.phase_report.items()
                              if hasattr(_engine, "budget") else {}.items()):
            record_token_spend(phase, tokens)
        if meta.verified_citations > 0:
            update_average_confidence(
                sum(vc.score for vc in output.citation_needed) / max(1, meta.verified_citations)
                if output.citation_needed else 0.99
            )

    return QueryResponse(
        content               = output.content,
        mode                  = output.mode.value,
        verified_citations    = meta.verified_citations if meta else 0,
        quarantined_citations = meta.quarantined        if meta else 0,
        loops_taken           = meta.loops_taken        if meta else 0,
        total_tokens_used     = meta.total_tokens_used  if meta else 0,
        human_review_required = meta.human_review_required if meta else True,
        citation_needed       = [cn.model_dump() for cn in output.citation_needed],
        conflict_log          = output.conflict_log,
        error                 = output.error,
    )


@app.get("/health")
async def health():
    """Liveness probe — always returns 200 if process is alive."""
    return {"status": "ok", "service": "uli-api"}


@app.get("/ready")
async def ready():
    """
    Readiness probe — gated on NJDG connectivity + engine initialisation.
    Returns 503 until all dependencies are healthy.
    """
    if not _ready:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return {"status": "ready", "engine": "loaded"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    body, content_type = metrics_output()
    return Response(content=body, media_type=content_type)


@app.get("/api/v1/budget/limits")
async def budget_limits():
    """Return current token budget configuration."""
    return {
        "phase_limits":  TokenBudgetManager.PHASE_LIMITS,
        "total_limit":   TokenBudgetManager.TOTAL_LIMIT,
        "description": {
            "dna_parse":     "GPT-4o-mini sub-call for legal taxonomy classification",
            "retrieval":     "Zero LLM — vector + BM25 retrieval only",
            "analyst_irac":  "GPT-4o chain-of-thought IRAC construction",
            "validation":    "Zero LLM — NJDG API verification only",
            "scribe_output": "GPT-4o formatted judgment / summary / brief output",
        },
    }
