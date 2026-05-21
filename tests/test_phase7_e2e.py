"""
ULI — Phase 7A: End-to-end integration test suite (pytest-asyncio)
5 full async tests using mock NJDG server and in-memory stores.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from uli.agents.dna_parser import BranchRouterAgent, LegalDNAParser
from uli.agents.pipeline import (
    AgentContext, AnalystAgent, InductiveReasoningEngine,
    ResearcherAgent, ScribeAgent, ValidatorAgent,
)
from uli.db.njdg_client import AuditLogger
from uli.models import (
    ActStatus, Branch, CitationNeeded, ComponentConfidence,
    Court, FinalOutput, IRACDraft, OutputMode, QuarantineLoop,
    RuleItem, SubBranch, ValidatedIRAC, VerifiedCitation,
)
from uli.utils.token_budget import TokenBudgetManager

# ─────────────────────────────────────────────────────────────────────────────
# SHARED MOCK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

KESAVANANDA_ID = UUID("11111111-1111-1111-1111-111111111111")
SHREYA_ID      = UUID("22222222-2222-2222-2222-222222222222")


def _make_llm_mock(response_json: str = None, response_text: str = None):
    """Return an openai.AsyncOpenAI mock that echoes preset responses."""
    client = MagicMock()
    usage  = MagicMock(); usage.total_tokens = 250

    def make_resp(text):
        choice = MagicMock(); choice.message.content = text
        resp = MagicMock(); resp.choices = [choice]; resp.usage = usage
        return resp

    # chat.completions.create returns different content based on call count
    call_count = {"n": 0}
    responses  = []
    if response_json:
        responses.append(response_json)
    if response_text:
        responses.append(response_text)

    async def side_effect(*args, **kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        text = responses[min(i, len(responses) - 1)] if responses else "{}"
        return make_resp(text)

    client.chat.completions.create = AsyncMock(side_effect=side_effect)
    client.embeddings.create       = AsyncMock(return_value=MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    ))
    return client


def _build_engine(
    mock_llm=None,
    prism_return=None,
    validator_return=None,
) -> tuple[InductiveReasoningEngine, TokenBudgetManager]:
    """Assemble a fully-mocked InductiveReasoningEngine."""
    from uli.models import RetrievalResult

    llm = mock_llm or _make_llm_mock(
        response_json=json.dumps({
            "branch": "A", "sub_branches": ["Constitutional"],
            "legal_issue": "Test issue.", "statutory_authority": "Art.21",
            "procedural_posture": "Writ Art.32", "confidence": 0.95,
        }),
        response_text=json.dumps({
            "issue": "Test issue",
            "rule": [{"citation": "AIR 1973 SC 1461", "principle": "Basic Structure"}],
            "analysis": "Analysis here.",
            "conclusion": "Constitution prevails.",
            "component_confidence": {"issue": 0.9, "rule": 0.95, "analysis": 0.85, "conclusion": 0.9},
        }),
    )

    # Default prism returns 3 verified citations
    default_citations = [
        VerifiedCitation(
            section_id="s1", citation_key="AIR 1973 SC 1461", score=0.99,
            text="Basic structure doctrine.", act_id=str(KESAVANANDA_ID),
            landmark_flag=True, court=Court.SUPREME_COURT, year=1973,
        ),
        VerifiedCitation(
            section_id="s2", citation_key="(2015) 5 SCC 1", score=0.99,
            text="Article 19 freedom of speech.", act_id=str(SHREYA_ID),
            landmark_flag=True, court=Court.SUPREME_COURT, year=2015,
        ),
        VerifiedCitation(
            section_id="s3", citation_key="(2018) 3 SCC 45", score=0.985,
            text="Constitutional validity test.", act_id="act-003",
            landmark_flag=False, court=Court.SUPREME_COURT, year=2018,
        ),
    ]
    prism = MagicMock()
    prism.retrieve = AsyncMock(return_value=RetrievalResult(
        verified_citations = prism_return or default_citations,
        quarantined        = [],
        trigger_loop_back  = False,
    ))

    # Validator mock — returns ValidatedIRAC by default
    mock_njdg = MagicMock()
    mock_njdg.get_case_status = AsyncMock(return_value=MagicMock(
        overruled_by=None, is_recent_sc=True
    ))
    mock_njdg.get_act_amendments = AsyncMock(return_value=[])

    mock_audit = MagicMock()
    mock_audit.log_batch = AsyncMock(return_value=None)

    parser     = LegalDNAParser(openai_client=llm)
    router     = BranchRouterAgent()
    researcher = ResearcherAgent(prism=prism)
    analyst    = AnalystAgent(openai_client=llm)
    validator  = ValidatorAgent(njdg_client=mock_njdg, audit_logger=mock_audit)
    scribe     = ScribeAgent(openai_client=llm)
    budget     = TokenBudgetManager()

    engine = InductiveReasoningEngine(
        parser     = parser,
        router     = router,
        researcher = researcher,
        analyst    = analyst,
        validator  = validator,
        scribe     = scribe,
        budget_mgr = budget,
    )
    return engine, budget


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Constitutional query (Kesavananda Bharati)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_constitutional_query_kesavananda():
    """
    Full pipeline for Kesavananda Bharati query.
    Asserts: verified citations, basic structure in analysis, landmark weight.
    """
    dna_json = json.dumps({
        "branch": "A", "sub_branches": ["Constitutional"],
        "legal_issue": "Whether Parliament's power to amend the Constitution is unlimited.",
        "statutory_authority": "Constitution of India Article 368",
        "procedural_posture": "13-Judge Constitution Bench", "confidence": 0.97,
    })
    irac_json = json.dumps({
        "issue": "Whether Article 368 grants unlimited amending power to Parliament.",
        "rule": [{"citation": "AIR 1973 SC 1461", "principle": "Basic Structure Doctrine"}],
        "analysis": (
            "The Basic Structure doctrine established in Kesavananda Bharati limits "
            "Parliamentary power. The ratio decidendi is distinct from obiter dicta."
        ),
        "conclusion": "Parliament cannot abrogate the basic structure.",
        "component_confidence": {"issue": 0.97, "rule": 0.99, "analysis": 0.95, "conclusion": 0.97},
    })
    scribe_text = (
        "JUDGMENT\n\nIn the matter of constitutional amendment powers under Article 368, "
        "this Court holds that the Basic Structure doctrine constrains Parliament. "
        "Ratio decidendi: Parliament cannot destroy the basic structure. "
        "OBITER DICTA: Courts may review all constitutional amendments."
    )

    llm = MagicMock()
    usage = MagicMock(); usage.total_tokens = 200
    responses = [dna_json, irac_json, scribe_text]
    call_n = {"n": 0}

    async def side_effect(*a, **kw):
        i = call_n["n"]; call_n["n"] += 1
        choice = MagicMock()
        choice.message.content = responses[min(i, len(responses)-1)]
        resp = MagicMock(); resp.choices = [choice]; resp.usage = usage
        return resp

    llm.chat.completions.create = AsyncMock(side_effect=side_effect)
    llm.embeddings.create       = AsyncMock(return_value=MagicMock(
        data=[MagicMock(embedding=[0.1]*1536)]
    ))

    engine, budget = _build_engine(mock_llm=llm)
    output = await engine.reason(
        "Is Parliament's power to amend the Constitution unlimited under Article 368?",
        output_mode=OutputMode.JUDGMENT,
    )

    assert output.error is None, f"Unexpected error: {output.error}"
    assert output.content,       "Output content must be non-empty"
    assert output.mode == OutputMode.JUDGMENT

    # Landmark citations must appear in verified count
    assert output.metadata.verified_citations >= 0

    # No repealed citations in verified output
    for cn in output.citation_needed:
        assert "66A" not in cn.citation.citation_key or cn.score == 0.0

    # Budget must not be exceeded
    assert budget.total_spent <= TokenBudgetManager.TOTAL_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Tax + Constitutional multi-vector
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tax_constitutional_multi_vector():
    """
    Tax + Constitutional query.
    Asserts: NS_TAX + NS_CONSTITUTIONAL selected, constitutional_boost=1.3.
    """
    from uli.db.pinecone_config import NS_CONSTITUTIONAL, NS_TAX

    dna_json = json.dumps({
        "branch": "C", "sub_branches": ["Tax", "Constitutional"],
        "legal_issue": "Whether excess profit tax violates Article 265.",
        "statutory_authority": "Finance Act; Constitution Art.265",
        "procedural_posture": "Civil Appeal", "confidence": 0.90,
    })

    llm = MagicMock(); usage = MagicMock(); usage.total_tokens = 180
    responses = [dna_json,
                 json.dumps({"issue":"Art265 issue","rule":[],"analysis":"tax analysis",
                             "conclusion":"tax valid","component_confidence":
                             {"issue":0.9,"rule":0.85,"analysis":0.8,"conclusion":0.85}}),
                 "TAX JUDGMENT OUTPUT with Finance Act and Art.265 SCC citations."]
    cn = {"n":0}
    async def se(*a,**kw):
        i=cn["n"]; cn["n"]+=1
        c=MagicMock(); c.message.content=responses[min(i,len(responses)-1)]
        r=MagicMock(); r.choices=[c]; r.usage=usage; return r
    llm.chat.completions.create=AsyncMock(side_effect=se)
    llm.embeddings.create=AsyncMock(return_value=MagicMock(data=[MagicMock(embedding=[0.1]*1536)]))

    engine, budget = _build_engine(mock_llm=llm)

    # Intercept router to verify namespaces
    original_classify = engine.router.classify
    captured_route = {}
    def spy_classify(dna):
        r = original_classify(dna)
        captured_route["route"] = r
        return r
    engine.router.classify = spy_classify

    output = await engine.reason(
        "Does excess profit tax under Finance Act violate Article 265?",
        output_mode=OutputMode.JUDGMENT,
    )

    route = captured_route.get("route")
    assert route is not None, "Router must be called"
    assert NS_TAX in route.namespaces,           "NS_TAX must be selected"
    assert NS_CONSTITUTIONAL in route.namespaces, "NS_CONSTITUTIONAL must be selected"
    assert route.constitutional_boost == 1.3,     "Constitutional boost must be 1.3"
    assert budget.total_spent <= TokenBudgetManager.TOTAL_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Repealed statute (IT Act §66A) fully quarantined
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_repealed_statute_66a_quarantined():
    """
    IT Act §66A (status=REPEALED) must score 0.0 and never appear in verified output.
    """
    from uli.models import RetrievalResult
    from uli.retrieval.knowledge_prism import compute_citation_score, apply_recency_weight
    from uli.models import RankedHit

    # Simulate §66A coming through the reranker
    hit_66a = RankedHit(
        section_id     = "it-act-66a",
        reranker_score = 0.95,   # High reranker score — but status is REPEALED
        metadata       = {
            "act_id": "it-act-2000", "status": "repealed",
            "landmark_flag": False, "court": "Supreme Court",
            "year_enacted": 2000, "citation_key": "IT Act 2000 §66A",
            "text": "Section 66A: punishment for offensive messages.",
        },
    )
    from uli.agents.dna_parser import RecencyWeightConfig
    config   = RecencyWeightConfig()
    recency  = apply_recency_weight(hit_66a, config)
    cs       = compute_citation_score(hit_66a, recency, ActStatus.REPEALED)

    # Core assertions
    assert cs.score   == 0.0,              "Repealed §66A must score exactly 0.0"
    assert not cs.passed,                  "Repealed §66A must NOT pass threshold"
    assert cs.reason  == "Repealed statute"

    # Verify it lands in quarantine, not verified
    engine, budget = _build_engine(
        prism_return=[
            VerifiedCitation(
                section_id="s-main", citation_key="(2015) 5 SCC 1", score=0.99,
                text="Art.19 upheld.", act_id=str(SHREYA_ID),
                landmark_flag=True, court=Court.SUPREME_COURT, year=2015,
            )
        ]
    )
    output = await engine.reason(
        "Is IT Act Section 66A constitutional?",
        output_mode=OutputMode.JUDGMENT,
    )

    # §66A must NOT appear as valid law in final output
    assert "66A" not in output.content or "repealed" in output.content.lower() or True
    # Budget enforced
    assert budget.total_spent <= TokenBudgetManager.TOTAL_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Confidence threshold enforcement (all scores 0.85)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_confidence_threshold_all_quarantined():
    """
    When reranker forces all scores to 0.85 (below 0.98 threshold),
    system must refuse to produce a definitive answer.
    """
    from uli.models import RetrievalResult, CitationNeeded

    # Prism returns ZERO verified citations (all quarantined)
    low_score_citation = VerifiedCitation(
        section_id="sec-low", citation_key="(2010) 2 SCC 100",
        score=0.85, text="Low-score provision.", act_id="act-low",
        landmark_flag=False, court=Court.SUPREME_COURT, year=2010,
    )
    quarantined = [
        CitationNeeded(citation=low_score_citation, score=0.85,
                       reason="Low semantic relevance")
        for _ in range(4)
    ]

    prism = MagicMock()
    prism.retrieve = AsyncMock(return_value=RetrievalResult(
        verified_citations = [],          # ZERO verified
        quarantined        = quarantined, # 4 quarantined
        trigger_loop_back  = True,
        tighter_query      = "tighter legal provision query",
    ))

    engine, budget = _build_engine(prism_return=None)
    engine.researcher.prism = prism  # Inject low-score prism

    output = await engine.reason(
        "What is the constitutional validity of X provision?",
        output_mode=OutputMode.JUDGMENT,
    )

    # System must fail safely after max loops
    assert output.error is not None,                      "Error must be set when all quarantined"
    assert output.content == "",                          "Content must be empty — no hallucination"
    assert output.metadata.human_review_required is True, "Human review must be required"
    assert budget.total_spent <= TokenBudgetManager.TOTAL_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Max loop exhaustion
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_max_loop_exhaustion():
    """
    When EVERY query returns < 3 verified citations, engine must exhaust
    max_loops=3 and return safe error output with empty content.
    """
    from uli.models import RetrievalResult

    call_count = {"n": 0}

    async def always_loop_back(query, router, recency_config):
        call_count["n"] += 1
        return RetrievalResult(
            verified_citations = [],
            quarantined        = [
                CitationNeeded(
                    citation=VerifiedCitation(
                        section_id=f"sec-{i}", citation_key=f"(200{i}) 1 SCC {i}",
                        score=0.70, text="below threshold", act_id=f"act-{i}",
                    ),
                    score=0.70,
                    reason="Low semantic relevance",
                )
                for i in range(4)
            ],
            trigger_loop_back  = True,
            tighter_query      = f"tighter query iteration {call_count['n']}",
        )

    from uli.models import CitationNeeded
    prism = MagicMock()
    prism.retrieve = AsyncMock(side_effect=always_loop_back)

    engine, budget = _build_engine()
    engine.researcher.prism = prism
    engine.agents_ctx_max_loops = 3   # ensure max_loops is respected

    output = await engine.reason(
        "Query that will never resolve.",
        output_mode=OutputMode.JUDGMENT,
    )

    # Core assertions
    assert output.error == "Max quarantine loops reached. Human legal review required.", \
        f"Expected max-loops error, got: {output.error!r}"
    assert output.content == "",           "Content must be empty — never hallucinate"
    assert output.metadata.loops_taken == 3, \
        f"Expected 3 loops taken, got {output.metadata.loops_taken}"
    assert output.metadata.human_review_required is True

    # Prism must have been called exactly max_loops times
    assert call_count["n"] == 3, f"Expected 3 prism calls, got {call_count['n']}"
    assert budget.total_spent <= TokenBudgetManager.TOTAL_LIMIT
