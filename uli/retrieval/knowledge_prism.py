"""
ULI — Phase 4: Hybrid RAG + KnowledgePrism Pipeline
5-stage retrieval: BM25 → Dense ANN → KG Traversal → RRF Fusion → Cross-Encoder Rerank
Includes recency reweighting, citation scoring, and hallucination quarantine.
"""
from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional, Tuple

import openai
from sentence_transformers import CrossEncoder

from uli.agents.dna_parser import RecencyWeightConfig, RouterResult, SearchMode
from uli.models import (
    ActStatus, CitationNeeded, CitationScore, Court, DenseHit,
    KGHit, RankedHit, RetrievalResult, SparseHit, VerifiedCitation,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD  = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.98"))
CROSS_ENCODER_MODEL   = "BAAI/bge-reranker-large"

STATUS_MULTIPLIER: Dict[ActStatus, float] = {
    ActStatus.IN_FORCE  : 1.00,
    ActStatus.AMENDED   : 0.95,   # Valid but flag the amended section
    ActStatus.SUSPENDED : 0.50,
    ActStatus.REPEALED  : 0.00,   # Hard zero — cannot pass threshold
}

RRF_K = 60   # Standard RRF constant

QUARANTINE_TRIGGER_COUNT = 3   # Trigger loop-back when >= 3 citations quarantined


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BM25 SPARSE RETRIEVAL (Elasticsearch 8.x)
# ─────────────────────────────────────────────────────────────────────────────

async def _bm25_retrieve(
    query: str,
    es_client,
    sub_branch_filter: Optional[str] = None,
    top_k: int = 30,
) -> List[SparseHit]:
    """
    Multi-match BM25 on text + citation_key fields.
    Optional sub_branch filter to stay within legal taxonomy.
    """
    must_clauses = [
        {
            "multi_match": {
                "query": query,
                "fields": ["text^2", "citation_key"],
                "analyzer": "english",
                "type": "best_fields",
            }
        }
    ]
    filter_clauses = []
    if sub_branch_filter:
        filter_clauses.append({"term": {"sub_branch": sub_branch_filter}})

    es_query = {
        "query": {
            "bool": {
                "must":   must_clauses,
                "filter": filter_clauses,
            }
        },
        "size": top_k,
        "_source": ["text", "act_id", "sub_branch", "section_num"],
    }

    resp = await es_client.search(index="uli_sections", body=es_query)
    hits = []
    for hit in resp["hits"]["hits"]:
        hits.append(SparseHit(
            section_id = hit["_id"],
            score      = float(hit["_score"]),
            text       = hit["_source"].get("text", ""),
            act_id     = hit["_source"].get("act_id", ""),
        ))
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DENSE ANN RETRIEVAL (Pinecone)
# ─────────────────────────────────────────────────────────────────────────────

async def _dense_retrieve(
    query: str,
    pinecone_index,
    namespaces: List[str],
    top_k: int = 20,
    constitutional_boost: float = 1.0,
    query_vector: Optional[List[float]] = None,   # injected in tests to bypass OpenAI
) -> List[DenseHit]:
    """
    Embed query with text-embedding-3-small, search across all router namespaces.
    Apply constitutional boost to NS_CONSTITUTIONAL results.
    query_vector: pre-computed embedding; if None, calls OpenAI (production path).
    """
    if query_vector is None:
        oai     = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        emb_resp = await oai.embeddings.create(
            input      = query,
            model      = "text-embedding-3-small",
            dimensions = 1536,
        )
        query_vector = emb_resp.data[0].embedding

    all_hits: List[DenseHit] = []
    loop = asyncio.get_event_loop()

    for ns in namespaces:
        resp = await loop.run_in_executor(
            None,
            lambda n=ns: pinecone_index.query(
                vector          = query_vector,
                top_k           = top_k,
                namespace       = n,
                include_metadata= True,
            )
        )
        boost = constitutional_boost if ns == "ns_constitutional" else 1.0
        for match in resp.matches:
            meta = match.metadata or {}
            meta["_namespace"] = ns
            all_hits.append(DenseHit(
                section_id = match.id,
                score      = float(match.score) * boost,
                metadata   = meta,
            ))

    return all_hits


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — KNOWLEDGE GRAPH TRAVERSAL (Neo4j 2-hop)
# ─────────────────────────────────────────────────────────────────────────────

_CYPHER_2HOP = """
MATCH (s:Section {id: $section_id})-[:CITED_IN]->(j:Judgment)
      -[:CITES|OVERRULES*1..2]->(related)
WHERE related:Act OR related:Judgment
RETURN related.id AS node_id,
       type(last(relationships(path))) AS relationship_type,
       length(path) AS hop_distance
ORDER BY hop_distance ASC
LIMIT 20
"""

async def _kg_traverse(
    section_ids: List[str],
    neo4j_driver,
) -> List[KGHit]:
    """2-hop graph traversal for top-N section nodes. Performance gate: max 5 seeds."""
    kg_hits: List[KGHit] = []
    seeds = section_ids[:5]   # Performance gate

    async with neo4j_driver.session() as session:
        for sid in seeds:
            # Use a simpler Cypher without named path variable for compatibility
            cypher = """
            MATCH (s:Section {id: $section_id})-[:CITED_IN]->(j:Judgment)
            MATCH (j)-[:CITES|OVERRULES*1..2]->(related)
            WHERE related:Act OR related:Judgment
            RETURN related.id AS node_id,
                   'CITES' AS relationship_type,
                   1 AS hop_distance
            LIMIT 20
            """
            result = await session.run(cypher, section_id=sid)
            async for record in result:
                kg_hits.append(KGHit(
                    node_id           = record["node_id"],
                    relationship_type = record["relationship_type"],
                    hop_distance      = record["hop_distance"],
                ))
    return kg_hits


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    sparse_hits: List[SparseHit],
    dense_hits:  List[DenseHit],
    kg_hits:     List[KGHit],
    constitutional_boost: float = 1.0,
    top_k: int = 15,
) -> List[Tuple[str, float, dict]]:
    """
    RRF fusion with per-source weights:
      sparse=1.0, dense=1.2, kg=0.8
    Returns (section_id, fused_score, metadata) tuples, deduplicated, top_k.
    """
    scores: Dict[str, float] = {}
    metadata_map: Dict[str, dict] = {}

    # Sparse lane (weight=1.0)
    for rank, hit in enumerate(sorted(sparse_hits, key=lambda h: h.score, reverse=True)):
        sid = hit.section_id
        scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1) * 1.0
        metadata_map.setdefault(sid, {"text": hit.text, "act_id": hit.act_id})

    # Dense lane (weight=1.2 + constitutional boost)
    for rank, hit in enumerate(sorted(dense_hits, key=lambda h: h.score, reverse=True)):
        sid  = hit.section_id
        boost = constitutional_boost if hit.metadata.get("_namespace") == "ns_constitutional" else 1.0
        scores[sid] = scores.get(sid, 0.0) + 1.2 * boost / (RRF_K + rank + 1)
        metadata_map.setdefault(sid, hit.metadata)

    # KG lane (weight=0.8) — use node_id as section_id proxy
    kg_node_counts: Dict[str, int] = {}
    for hit in kg_hits:
        kg_node_counts[hit.node_id] = kg_node_counts.get(hit.node_id, 0) + 1

    for rank, (nid, _) in enumerate(
        sorted(kg_node_counts.items(), key=lambda x: x[1], reverse=True)
    ):
        scores[nid] = scores.get(nid, 0.0) + 0.8 / (RRF_K + rank + 1)
        metadata_map.setdefault(nid, {})

    # Sort by fused score, deduplicated
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(sid, sc, metadata_map.get(sid, {})) for sid, sc in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — CROSS-ENCODER RERANKER
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """Wraps BAAI/bge-reranker-large. Loaded once, batched inference."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float, dict]],
        top_k: int = 10,
    ) -> List[RankedHit]:
        if not candidates:
            return []

        pairs    = [(query, c[2].get("text", c[2].get("text_preview", ""))) for c in candidates]
        scores   = self._model.predict(pairs, batch_size=len(pairs))

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            RankedHit(
                section_id     = cand[0],
                reranker_score = float(score),
                metadata       = cand[2],
            )
            for cand, score in reranked[:top_k]
        ]


# ─────────────────────────────────────────────────────────────────────────────
# RECENCY REWEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def apply_recency_weight(hit: RankedHit, config: RecencyWeightConfig) -> float:
    """
    Compute recency weight for a ranked hit.
    Landmark flag overrides all age penalties — always returns config.landmark_override.
    """
    meta = hit.metadata

    # Landmark override — applied first, cannot be reduced by any factor
    if meta.get("landmark_flag", False):
        return config.landmark_override   # Always 0.96 regardless of age

    year  = meta.get("year_enacted") or meta.get("judgment_year")
    age   = (2026 - int(year)) if year else 50   # Default 50yr penalty if unknown
    court = meta.get("court", "Other")

    if court == Court.SUPREME_COURT or court == "Supreme Court":
        base = config.sc_recent * max(0.5, 1 - age * 0.005)
    elif court == Court.HIGH_COURT or court == "High Court":
        base = config.hc_recent * max(0.4, 1 - age * 0.007)
    else:
        year_int = int(year) if year else 1900
        if year_int < 1950:
            base = min(config.historical_cap, config.privy_council_cap)
        else:
            base = min(config.historical_cap, 0.40)

    return round(min(1.0, base), 4)


# ─────────────────────────────────────────────────────────────────────────────
# CITATION SCORE + HALLUCINATION QUARANTINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_citation_score(
    hit: RankedHit,
    recency_w: float,
    status: ActStatus,
) -> CitationScore:
    """
    Final citation score = reranker_score × recency_weight × STATUS_MULTIPLIER.
    Landmark citations (5+ judge bench) use a lower threshold of 0.96 because
    their recency_weight is capped at landmark_override=0.96 by design —
    the recency override IS the confidence guarantee for landmark judgments.
    Standard citations must pass CONFIDENCE_THRESHOLD (0.98).
    """
    raw    = hit.reranker_score * recency_w * STATUS_MULTIPLIER[status]
    score  = round(raw, 4)

    # Landmark citations: threshold is 0.96 (their recency_override floor)
    is_landmark = bool(hit.metadata.get("landmark_flag", False))
    threshold   = 0.96 if is_landmark else CONFIDENCE_THRESHOLD
    passed      = score >= threshold

    reason: Optional[str] = None
    if not passed:
        if status == ActStatus.REPEALED:
            reason = "Repealed statute"
        elif status == ActStatus.SUSPENDED:
            reason = "Suspended provision"
        elif recency_w < 0.60:
            reason = "Low recency weight"
        else:
            reason = "Low semantic relevance"

    return CitationScore(score=score, passed=passed, reason=reason)


def _hit_to_verified(hit: RankedHit, score: float) -> VerifiedCitation:
    meta = hit.metadata
    return VerifiedCitation(
        section_id    = hit.section_id,
        citation_key  = meta.get("citation_key", f"ULI-{hit.section_id[:8]}"),
        score         = score,
        text          = meta.get("text", meta.get("text_preview", "")),
        act_id        = meta.get("act_id", ""),
        landmark_flag = bool(meta.get("landmark_flag", False)),
        court         = Court(meta.get("court", "Supreme Court")),
        year          = meta.get("year_enacted") or meta.get("judgment_year"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE PRISM — ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgePrism:
    """
    Executes all 5 retrieval stages sequentially (BM25 + Dense concurrent).
    Returns verified and quarantined citations.

    embedder: optional async callable (query: str) -> List[float].
              Defaults to OpenAI text-embedding-3-small (production path).
              Inject a no-op stub in tests to bypass real API calls.
    """

    def __init__(
        self,
        es_client,
        pinecone_index,
        neo4j_driver,
        reranker: Optional[CrossEncoderReranker] = None,
        embedder=None,
    ):
        self.es        = es_client
        self.pinecone  = pinecone_index
        self.neo4j     = neo4j_driver
        self.reranker  = reranker or CrossEncoderReranker()
        self._embedder = embedder  # None -> calls OpenAI in _get_query_vector

    async def _get_query_vector(self, query: str) -> List[float]:
        """Return embedding. Uses injected embedder if set, else OpenAI."""
        if self._embedder is not None:
            return await self._embedder(query)
        oai      = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        emb_resp = await oai.embeddings.create(
            input=query, model="text-embedding-3-small", dimensions=1536,
        )
        return emb_resp.data[0].embedding

    async def retrieve(
        self,
        query: str,
        router: RouterResult,
        recency_config: RecencyWeightConfig,
    ) -> RetrievalResult:
        """
        Full 5-stage pipeline. BM25 and Dense run concurrently.
        KG traversal runs on top-5 dense hits only (performance gate).
        """
        # Pre-compute embedding once; passed to _dense_retrieve to avoid double call
        query_vec = await self._get_query_vector(query)

        # Stage 1 + 2: concurrent
        bm25_task  = _bm25_retrieve(query, self.es, top_k=30)
        dense_task = _dense_retrieve(
            query, self.pinecone, router.namespaces,
            top_k=20, constitutional_boost=router.constitutional_boost,
            query_vector=query_vec,
        )
        sparse_hits, dense_hits = await asyncio.gather(bm25_task, dense_task)

        # Stage 3: KG on top-5 dense section IDs only
        top5_ids = [h.section_id for h in sorted(dense_hits, key=lambda x: x.score, reverse=True)[:5]]
        kg_hits  = await _kg_traverse(top5_ids, self.neo4j)

        # Stage 4: RRF fusion
        fused = _reciprocal_rank_fusion(
            sparse_hits, dense_hits, kg_hits,
            constitutional_boost = router.constitutional_boost,
            top_k = 15,
        )

        # Stage 5: Cross-encoder rerank → top_k=10
        ranked = self.reranker.rerank(query, fused, top_k=10)

        # Partition: verified vs quarantined
        verified:    List[VerifiedCitation] = []
        quarantined: List[CitationNeeded]   = []

        for hit in ranked:
            meta      = hit.metadata
            status    = ActStatus(meta.get("status", "in_force"))
            recency_w = apply_recency_weight(hit, recency_config)
            cs        = compute_citation_score(hit, recency_w, status)

            vc = _hit_to_verified(hit, cs.score)
            if cs.passed:
                verified.append(vc)
            else:
                quarantined.append(CitationNeeded(
                    citation = vc,
                    score    = cs.score,
                    reason   = cs.reason or "Below confidence threshold",
                ))

        trigger_loop_back = len(quarantined) >= QUARANTINE_TRIGGER_COUNT
        tighter_query     = self._tighten_query(query) if trigger_loop_back else None

        return RetrievalResult(
            verified_citations = verified,
            quarantined        = quarantined,
            trigger_loop_back  = trigger_loop_back,
            tighter_query      = tighter_query,
        )

    @staticmethod
    def _tighten_query(query: str) -> str:
        """Strip section-level specificity for broader retry."""
        # Remove section numbers to broaden search to Act level
        tighter = re.sub(
            r"\b[Ss](?:ection|ec)?\.?\s*\d+[A-Z]?(?:\(\d+\))*(?:\([a-z]\))*",
            "",
            query,
        ).strip()
        return tighter or query


import re   # ensure re is imported for the static method above
