#!/usr/bin/env python3
"""
ULI — Post-Build Verification Script
Runs all checks from the verification checklist without Docker.
"""
import sys
import subprocess


CHECKS = []


def check(name):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("Import check — InductiveReasoningEngine")
def check_imports():
    from uli.agents.pipeline import InductiveReasoningEngine
    from uli.agents.dna_parser import LegalDNAParser, BranchRouterAgent
    from uli.retrieval.knowledge_prism import KnowledgePrism
    from uli.db.njdg_client import NJDGClient, LiveVeracityDB
    from uli.utils.token_budget import TokenBudgetManager
    from uli.utils.metrics import metrics_output
    return True


@check("Token budget — phase limits correct")
def check_token_limits():
    from uli.utils.token_budget import TokenBudgetManager
    bm = TokenBudgetManager()
    assert bm.PHASE_LIMITS["dna_parse"]     == 300
    assert bm.PHASE_LIMITS["retrieval"]     == 0
    assert bm.PHASE_LIMITS["analyst_irac"]  == 800
    assert bm.PHASE_LIMITS["validation"]    == 0
    assert bm.PHASE_LIMITS["scribe_output"] == 1200
    assert bm.TOTAL_LIMIT                   == 2300
    return True


@check("Repealed statute → score 0.0 (zero-hallucination proof)")
def check_repealed_zero():
    from uli.models import ActStatus
    from uli.retrieval.knowledge_prism import (
        STATUS_MULTIPLIER, apply_recency_weight, compute_citation_score,
    )
    from uli.models import RankedHit
    from uli.agents.dna_parser import RecencyWeightConfig
    hit = RankedHit(section_id="s1", reranker_score=0.99, metadata={
        "act_id": "a1", "status": "in_force", "landmark_flag": False,
        "court": "Supreme Court", "year_enacted": 2000,
    })
    config   = RecencyWeightConfig()
    recency  = apply_recency_weight(hit, config)
    cs       = compute_citation_score(hit, recency, ActStatus.REPEALED)
    assert cs.score == 0.0, f"Expected 0.0, got {cs.score}"
    assert not cs.passed
    return True


@check("Landmark 1973 judgment → weight >= 0.96 (recency override proof)")
def check_landmark_weight():
    from uli.models import ActStatus, RankedHit
    from uli.retrieval.knowledge_prism import apply_recency_weight, compute_citation_score
    from uli.agents.dna_parser import RecencyWeightConfig
    hit = RankedHit(section_id="kesavananda", reranker_score=1.0, metadata={
        "act_id": "a1", "status": "in_force", "landmark_flag": True,
        "court": "Supreme Court", "year_enacted": 1973,
    })
    config  = RecencyWeightConfig()
    recency = apply_recency_weight(hit, config)
    assert recency >= 0.96, f"Landmark weight {recency} must be >= 0.96"
    cs = compute_citation_score(hit, recency, ActStatus.IN_FORCE)
    assert cs.score >= 0.96
    return True


@check("Prometheus metrics — all 6 metrics registered")
def check_metrics():
    from uli.utils.metrics import (
        uli_citation_quarantine_total,
        uli_average_confidence_score,
        uli_loop_back_total,
        uli_tokens_per_phase,
        uli_request_duration_seconds,
        uli_human_review_required_total,
        metrics_output,
    )
    body, ctype = metrics_output()
    assert b"uli_citation_quarantine_total" in body
    assert b"uli_average_confidence_score"  in body
    assert b"uli_loop_back_total"           in body
    assert b"uli_tokens_per_phase"          in body
    assert b"uli_request_duration_seconds"  in body
    assert b"uli_human_review_required_total" in body
    return True


@check("TokenBudgetManager — fallback triggered on budget exhaustion")
def check_budget_fallback():
    from uli.utils.token_budget import TokenBudgetManager
    bm = TokenBudgetManager()
    with bm.phase("analyst_irac", max_tokens=800) as ctx:
        ctx.record(800)
    with bm.phase("scribe_output", max_tokens=1200) as ctx:
        ctx.record(1200)
    with bm.phase("dna_parse", max_tokens=300):
        pass
    assert bm.fallback_triggered
    return True


@check("Models — ActMetadata repealed sets recency_weight=0.0")
def check_model_validators():
    import hashlib
    from uli.models import ActMetadata, ActStatus, Branch, SubBranch
    act = ActMetadata(
        name="Test Repealed Act",
        short_title="TRA",
        year_enacted=1990,
        status=ActStatus.REPEALED,
        branch=Branch.A,
        sub_branch=SubBranch.CONSTITUTIONAL,
        citation_key="AIR 1990 SC 100",
        recency_weight=0.80,
    )
    assert act.recency_weight == 0.0, f"Repealed act must have recency_weight=0.0, got {act.recency_weight}"
    return True


@check("Models — landmark_flag adds +0.15 boost, capped at 1.0")
def check_landmark_boost():
    from uli.models import ActMetadata, ActStatus, Branch, SubBranch
    act = ActMetadata(
        name="Basic Structure Judgment",
        short_title="BSJ",
        year_enacted=1973,
        status=ActStatus.IN_FORCE,
        branch=Branch.A,
        sub_branch=SubBranch.CONSTITUTIONAL,
        landmark_flag=True,
        citation_key="AIR 1973 SC 1461",
        recency_weight=0.80,
    )
    assert act.recency_weight == min(1.0, 0.80 + 0.15), (
        f"Expected {min(1.0, 0.80+0.15)}, got {act.recency_weight}"
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 68)
    print("  ULI POST-BUILD VERIFICATION")
    print("═" * 68)

    passed = 0
    failed = 0

    for name, fn in CHECKS:
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗  {name}")
            print(f"     ERROR: {e}")
            failed += 1

    print("─" * 68)
    print(f"  RESULT: {passed} passed, {failed} failed")
    print("═" * 68 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
