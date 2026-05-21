"""
ULI — Token Budget Manager tests
Verifies phase limits, total ceiling, fallback trigger, and spend recording.
"""
from __future__ import annotations

import pytest

from uli.utils.token_budget import TokenBudgetExceeded, TokenBudgetManager


def test_phase_limits_defined():
    """All required phases must be present with correct limits."""
    bm = TokenBudgetManager()
    assert bm.PHASE_LIMITS["dna_parse"]     == 300
    assert bm.PHASE_LIMITS["retrieval"]     == 0
    assert bm.PHASE_LIMITS["analyst_irac"]  == 800
    assert bm.PHASE_LIMITS["validation"]    == 0
    assert bm.PHASE_LIMITS["scribe_output"] == 1200
    assert bm.TOTAL_LIMIT                   == 2300


def test_record_spend_within_limit():
    """Recording spend within phase limit should succeed."""
    bm = TokenBudgetManager()
    with bm.phase("dna_parse", max_tokens=300) as ctx:
        ctx.record(200)
    assert bm.total_spent == 200
    assert bm.phase_report["dna_parse"] == 200


def test_record_spend_exceeds_phase_limit():
    """Recording spend above phase limit must raise TokenBudgetExceeded."""
    bm = TokenBudgetManager()
    with pytest.raises(TokenBudgetExceeded) as exc_info:
        with bm.phase("dna_parse", max_tokens=300) as ctx:
            ctx.record(400)   # 400 > 300
    assert exc_info.value.phase == "dna_parse"
    assert exc_info.value.used  == 400
    assert exc_info.value.limit == 300


def test_total_limit_triggers_fallback():
    """When projected spend exceeds TOTAL_LIMIT - buffer, fallback is triggered."""
    bm = TokenBudgetManager()
    # Exhaust most of the budget manually
    with bm.phase("analyst_irac", max_tokens=800) as ctx:
        ctx.record(800)
    with bm.phase("scribe_output", max_tokens=1200) as ctx:
        ctx.record(1200)
    # Now dna_parse would push past ceiling — fallback should trigger
    with bm.phase("dna_parse", max_tokens=300):
        pass
    assert bm.fallback_triggered, "Fallback must be triggered when total budget is exhausted"


def test_zero_token_phases():
    """Retrieval and validation phases must have zero LLM token limit."""
    bm = TokenBudgetManager()
    assert bm.PHASE_LIMITS["retrieval"]  == 0
    assert bm.PHASE_LIMITS["validation"] == 0


def test_budget_reset():
    """After reset(), spend counters must be zeroed."""
    bm = TokenBudgetManager()
    with bm.phase("dna_parse", max_tokens=300) as ctx:
        ctx.record(150)
    bm.reset()
    assert bm.total_spent == 0
    assert bm.phase_report == {}
    assert not bm.fallback_triggered


def test_remaining_decreases():
    """Remaining budget must decrease by the amount recorded."""
    bm = TokenBudgetManager()
    initial_remaining = bm.remaining
    with bm.phase("dna_parse", max_tokens=300) as ctx:
        ctx.record(200)
    assert bm.remaining == initial_remaining - 200


def test_repr_shows_spend():
    """__repr__ must include request_id, spent, and total."""
    bm = TokenBudgetManager()
    r  = repr(bm)
    assert "spent=0/2300" in r
