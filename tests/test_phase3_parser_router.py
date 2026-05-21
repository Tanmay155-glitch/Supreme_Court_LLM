"""
ULI — Phase 3 unit tests
Tests for LegalDNAParser and BranchRouterAgent.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from uli.agents.dna_parser import BranchRouterAgent, LegalDNAParser
from uli.db.pinecone_config import (
    NS_CONSTITUTIONAL, NS_CORPORATE, NS_CRIMINAL,
    NS_ENVIRONMENTAL, NS_TAX,
)
from uli.models import Branch, SubBranch


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _mock_parser_gpt(json_response: str):
    """Patch GPT-4o-mini to return fixed JSON."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = json_response
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIZED TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

PARSE_TEST_CASES = [
    pytest.param(
        # 1. Kesavananda Bharati v State of Kerala (1973) — 13-judge bench
        (
            "Kesavananda Bharati v State of Kerala (1973) AIR 1973 SC 1461 — "
            "13-Judge Constitution Bench. Is the power of Parliament to amend "
            "the Constitution under Article 368 unlimited?"
        ),
        {
            "branch": "A",
            "sub_branches": ["Constitutional"],
            "legal_issue": "Whether Parliament's amending power under Article 368 is unlimited.",
            "statutory_authority": "Constitution of India Article 368",
            "procedural_posture": "13-Judge Constitution Bench",
            "confidence": 0.97,
        },
        Branch.A,
        SubBranch.CONSTITUTIONAL,
        13,
        [NS_CONSTITUTIONAL],
        id="kesavananda_13_judge",
    ),
    pytest.param(
        # 2. Commissioner of Income Tax v Bombay — Tax + Constitutional
        (
            "Commissioner of Income Tax v Bombay — Does the excess profit tax "
            "levied under the Finance Act violate Article 265 of the Constitution?"
        ),
        {
            "branch": "C",
            "sub_branches": ["Tax", "Constitutional"],
            "legal_issue": "Whether excess profit tax violates Art.265.",
            "statutory_authority": "Finance Act; Constitution Article 265",
            "procedural_posture": "Special Leave Petition",
            "confidence": 0.90,
        },
        Branch.C,
        SubBranch.TAX,
        None,
        [NS_TAX, NS_CONSTITUTIONAL],
        id="cit_bombay_tax_constitutional",
    ),
    pytest.param(
        # 3. Shreya Singhal v Union of India (2015) — §66A repealed
        (
            "Shreya Singhal v Union of India (2015) 5 SCC 1 — "
            "Is Section 66A of the Information Technology Act constitutional?"
        ),
        {
            "branch": "A",
            "sub_branches": ["Constitutional"],
            "legal_issue": "Constitutionality of IT Act Section 66A under Article 19.",
            "statutory_authority": "Information Technology Act §66A; Constitution Art.19",
            "procedural_posture": "Writ Petition Art.32",
            "confidence": 0.96,
        },
        Branch.A,
        SubBranch.CONSTITUTIONAL,
        None,
        [NS_CONSTITUTIONAL],
        id="shreya_singhal_66a",
    ),
    pytest.param(
        # 4. M/s Transmission Corpn of AP v CIT — Tax, single_vector
        (
            "M/s Transmission Corporation of Andhra Pradesh Ltd v Commissioner "
            "of Income Tax — Tax deduction at source under Income Tax Act."
        ),
        {
            "branch": "C",
            "sub_branches": ["Tax"],
            "legal_issue": "Whether TDS obligations apply to transmission charges.",
            "statutory_authority": "Income Tax Act §194",
            "procedural_posture": "Civil Appeal",
            "confidence": 0.88,
        },
        Branch.C,
        SubBranch.TAX,
        None,
        [NS_TAX, NS_CONSTITUTIONAL],
        id="transmission_corp_ap_tax",
    ),
    pytest.param(
        # 5. Sterlite Industries v UoI — Environmental + Corporate, multi_vector
        (
            "Sterlite Industries (India) Ltd v Union of India — "
            "Environmental clearance for copper smelter plant under "
            "Environment Protection Act and Companies Act."
        ),
        {
            "branch": "C",
            "sub_branches": ["Environmental", "Insolvency"],
            "legal_issue": "Validity of environmental clearance and corporate liability.",
            "statutory_authority": "Environment Protection Act; Companies Act",
            "procedural_posture": "Writ Petition",
            "confidence": 0.85,
        },
        Branch.C,
        SubBranch.ENVIRONMENTAL,
        None,
        [NS_ENVIRONMENTAL, NS_CORPORATE],
        id="sterlite_env_corporate",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,gpt_json,expected_branch,expected_primary_sub,expected_bench,expected_ns",
    PARSE_TEST_CASES,
)
async def test_legal_dna_parser_and_router(
    query, gpt_json, expected_branch, expected_primary_sub,
    expected_bench, expected_ns
):
    """Verify parser produces correct LegalDNA and router selects correct namespaces."""
    import json as _json

    client = _mock_parser_gpt(_json.dumps(gpt_json))
    parser = LegalDNAParser(openai_client=client)
    router = BranchRouterAgent()

    dna    = await parser.parse(query)
    result = router.classify(dna)

    # Branch check
    assert dna.branch == expected_branch, (
        f"Expected branch {expected_branch}, got {dna.branch}"
    )

    # Primary sub-branch
    assert dna.sub_branches[0] == expected_primary_sub, (
        f"Expected primary sub_branch {expected_primary_sub}, got {dna.sub_branches[0]}"
    )

    # Bench strength (if detectable from query)
    if expected_bench is not None:
        assert dna.bench_strength == expected_bench, (
            f"Expected bench_strength={expected_bench}, got {dna.bench_strength}"
        )

    # Namespace check: all expected namespaces must appear in router result
    for ns in expected_ns:
        assert ns in result.namespaces, (
            f"Expected namespace '{ns}' not in router result: {result.namespaces}"
        )

    # Constitutional boost if constitutional namespace selected
    if NS_CONSTITUTIONAL in result.namespaces:
        assert result.constitutional_boost == 1.3, (
            f"Expected constitutional_boost=1.3, got {result.constitutional_boost}"
        )

    # Confidence passthrough
    assert 0.0 <= result.router_confidence <= 1.0
