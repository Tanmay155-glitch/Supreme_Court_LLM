"""
ULI — Phase 4 unit tests
Tests for KnowledgePrism: repealed statute quarantine, landmark override,
constitutional boost, and loop-back trigger.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uli.agents.dna_parser import RecencyWeightConfig, RouterResult, SearchMode
from uli.models import (
    ActStatus, Court, DenseHit, KGHit, RankedHit,
    SparseHit, VerifiedCitation,
)
from uli.retrieval.knowledge_prism import (
    CrossEncoderReranker, KnowledgePrism,
    apply_recency_weight, compute_citation_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _make_router(namespaces=None, boost=1.0, mode=SearchMode.SINGLE_VECTOR) -> RouterResult:
    return RouterResult(
        namespaces           = namespaces or ["ns_constitutional"],
        search_mode          = mode,
        recency_config       = RecencyWeightConfig(),
        router_confidence    = 0.90,
        constitutional_boost = boost,
    )


def _make_sparse_hit(section_id: str, score: float, text: str = "text") -> SparseHit:
    return SparseHit(section_id=section_id, score=score, text=text, act_id="act-001")


def _make_dense_hit(section_id: str, score: float, metadata: dict = None) -> DenseHit:
    return DenseHit(
        section_id = section_id,
        score      = score,
        metadata   = metadata or {
            "act_id": "act-001",
            "status": "in_force",
            "landmark_flag": False,
            "court": "Supreme Court",
            "year_enacted": 2020,
        },
    )


def _make_ranked_hit(section_id: str, score: float, metadata: dict = None) -> RankedHit:
    return RankedHit(
        section_id     = section_id,
        reranker_score = score,
        metadata       = metadata or {
            "act_id": "act-001",
            "status": "in_force",
            "landmark_flag": False,
            "court": "Supreme Court",
            "year_enacted": 2020,
            "citation_key": f"(2020) 1 SCC {section_id[-3:]}",
            "text": "Sample legal provision.",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Repealed statute scores 0.0 → quarantined
# ─────────────────────────────────────────────────────────────────────────────

def test_repealed_statute_scores_zero():
    """STATUS_MULTIPLIER[REPEALED] = 0.0 → score always 0.0 regardless of reranker."""
    hit      = _make_ranked_hit("sec-66a", score=0.99)
    config   = RecencyWeightConfig()
    recency  = apply_recency_weight(hit, config)
    cs       = compute_citation_score(hit, recency, ActStatus.REPEALED)

    assert cs.score == 0.0, f"Repealed statute must score 0.0, got {cs.score}"
    assert not cs.passed,   "Repealed statute must not pass confidence threshold"
    assert cs.reason == "Repealed statute"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Landmark 1973 judgment scores >= 0.96 despite age
# ─────────────────────────────────────────────────────────────────────────────

def test_landmark_override_ignores_age():
    """landmark_flag=True → recency weight always == landmark_override (0.96)."""
    hit = _make_ranked_hit("sec-kesavananda", score=1.0, metadata={
        "act_id":       "act-kesavananda",
        "status":       "in_force",
        "landmark_flag": True,
        "court":         "Supreme Court",
        "year_enacted":  1973,             # 53 years ago — would normally be penalised
        "citation_key":  "AIR 1973 SC 1461",
        "text":          "Basic structure doctrine.",
    })
    config  = RecencyWeightConfig()
    recency = apply_recency_weight(hit, config)

    assert recency == config.landmark_override, (
        f"Landmark override should be {config.landmark_override}, got {recency}"
    )

    cs = compute_citation_score(hit, recency, ActStatus.IN_FORCE)
    assert cs.score >= 0.96, f"Landmark citation score must be >= 0.96, got {cs.score}"
    assert cs.passed,        "Landmark citation must pass confidence threshold"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Cross-namespace fusion applies 1.3 constitutional boost
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cross_namespace_constitutional_boost():
    """
    When constitutional namespace is in router result, dense hits from
    ns_constitutional must be boosted by 1.3.
    """
    from uli.retrieval.knowledge_prism import _reciprocal_rank_fusion

    # No sparse hits — we're testing only the dense boost
    sparse = []
    dense  = [
        _make_dense_hit("sec-con-001", 0.85, metadata={
            "act_id": "a1", "status": "in_force", "landmark_flag": False,
            "court": "Supreme Court", "year_enacted": 2015,
            "_namespace": "ns_constitutional",   # Constitutional lane → boosted ×1.3
        }),
        _make_dense_hit("sec-tax-001", 0.85, metadata={          # same score, no boost
            "act_id": "a2", "status": "in_force", "landmark_flag": False,
            "court": "Supreme Court", "year_enacted": 2018,
            "_namespace": "ns_tax",
        }),
    ]
    kg = []

    fused = _reciprocal_rank_fusion(
        sparse, dense, kg,
        constitutional_boost = 1.3,
        top_k = 5,
    )

    ids = [f[0] for f in fused]
    assert "sec-con-001" in ids, "Constitutional section must appear in fused results"
    assert "sec-tax-001" in ids, "Tax section must appear in fused results"

    score_con = next(f[1] for f in fused if f[0] == "sec-con-001")
    score_tax = next(f[1] for f in fused if f[0] == "sec-tax-001")
    assert score_con > score_tax, (
        f"Constitutional boost 1.3 should make sec-con-001 ({score_con:.4f}) "
        f"rank above sec-tax-001 ({score_tax:.4f}) when base scores are equal"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — trigger_loop_back=True when >= 3 citations quarantined
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trigger_loop_back_on_three_quarantined():
    """
    When the reranker returns hits that all score below threshold,
    trigger_loop_back must be True.
    """
    # Build mock clients that return low-scoring hits
    mock_es = MagicMock()
    mock_es.search = AsyncMock(return_value={
        "hits": {"hits": [
            {"_id": f"sec-{i:03d}", "_score": 0.4,
             "_source": {"text": f"text {i}", "act_id": "act-001",
                         "sub_branch": "Constitutional"}}
            for i in range(5)
        ]}
    })

    mock_pinecone = MagicMock()
    mock_pinecone.query = MagicMock(return_value=MagicMock(matches=[
        MagicMock(id=f"sec-{i:03d}", score=0.4, metadata={
            "act_id": "act-001", "status": "in_force",
            "landmark_flag": False, "court": "Supreme Court",
            "year_enacted": 2010,
        })
        for i in range(5)
    ]))

    # Proper async iterator mock for neo4j result
    class _AsyncIter:
        def __aiter__(self): return self
        async def __anext__(self): raise StopAsyncIteration

    _mock_result = MagicMock()
    _mock_result.__aiter__ = lambda self: _AsyncIter()

    _mock_session = MagicMock()
    _mock_session.run = AsyncMock(return_value=_mock_result)
    _mock_session.__aenter__ = AsyncMock(return_value=_mock_session)
    _mock_session.__aexit__  = AsyncMock(return_value=None)

    mock_neo4j = MagicMock()
    mock_neo4j.session = MagicMock(return_value=_mock_session)

    # Reranker that returns scores too low to pass threshold (0.98)
    mock_reranker = MagicMock()
    mock_reranker.rerank = MagicMock(return_value=[
        _make_ranked_hit(f"sec-{i:03d}", score=0.55, metadata={
            "act_id": "act-001", "status": "in_force",
            "landmark_flag": False, "court": "Supreme Court",
            "year_enacted": 2010,
            "citation_key": f"(2010) 1 SCC {i+1}",
            "text": f"Provision {i+1}.",
        })
        for i in range(5)
    ])

    # Inject dummy embedder — bypasses OpenAI entirely in tests
    async def _dummy_embedder(q: str): return [0.1] * 1536

    prism  = KnowledgePrism(mock_es, mock_pinecone, mock_neo4j, mock_reranker,
                             embedder=_dummy_embedder)
    router = _make_router()
    result = await prism.retrieve("test query", router, RecencyWeightConfig())

    assert result.trigger_loop_back, (
        "trigger_loop_back must be True when >= 3 citations are quarantined"
    )
    assert len(result.quarantined) >= 3, (
        f"Expected >= 3 quarantined, got {len(result.quarantined)}"
    )
    assert result.tighter_query is not None, "tighter_query must be set on loop-back trigger"
    assert len(result.verified_citations) == 0, (
        "No citations should pass when all scores are low"
    )
