"""
ULI — pytest conftest.py
Shared fixtures for NJDG client, veracity DB, and sample citations.
Uses mock NJDG server + in-memory Redis + SQLite async.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from typing import AsyncGenerator, List
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
import pytest_asyncio

from uli.models import (
    ActStatus, CitationNeeded, CaseStatus, Court,
    VeracityResult, VerifiedCitation,
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION-SCOPED EVENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# MOCK NJDG CLIENT
# ─────────────────────────────────────────────────────────────────────────────

KESAVANANDA_ID = UUID("11111111-1111-1111-1111-111111111111")
SHREYA_ID      = UUID("22222222-2222-2222-2222-222222222222")
OVERRULED_ID   = UUID("33333333-3333-3333-3333-333333333333")
SUSPENDED_ID   = UUID("44444444-4444-4444-4444-444444444444")


def _make_case_status(case_id: UUID) -> CaseStatus:
    data_map = {
        KESAVANANDA_ID: CaseStatus(
            case_id=KESAVANANDA_ID, citation_key="AIR 1973 SC 1461",
            status="decided", overruled_by=None, is_recent_sc=False, decided_year=1973,
        ),
        SHREYA_ID: CaseStatus(
            case_id=SHREYA_ID, citation_key="(2015) 5 SCC 1",
            status="decided", overruled_by=None, is_recent_sc=True, decided_year=2015,
        ),
        OVERRULED_ID: CaseStatus(
            case_id=OVERRULED_ID, citation_key="(1980) 3 SCC 625",
            status="decided", overruled_by="(2015) 5 SCC 1",
            is_recent_sc=False, decided_year=1980,
        ),
        SUSPENDED_ID: CaseStatus(
            case_id=SUSPENDED_ID, citation_key="(2020) 7 SCC 200",
            status="suspended", overruled_by=None, is_recent_sc=True, decided_year=2020,
        ),
    }
    return data_map.get(
        case_id,
        CaseStatus(case_id=case_id, citation_key="UNKNOWN", status="decided",
                   is_recent_sc=True, decided_year=2022),
    )


@pytest_asyncio.fixture
async def njdg_client():
    """Mock NJDG client that returns seeded test data."""
    client = MagicMock()

    async def mock_get_case_status(case_id):
        return _make_case_status(case_id)

    async def mock_get_amendments(act_id):
        return []

    client.get_case_status   = mock_get_case_status
    client.get_act_amendments = mock_get_amendments
    client.update_veracity   = AsyncMock(return_value=True)
    client.get_judgment_ratio = AsyncMock(return_value=MagicMock(
        ratio="Stub ratio", is_landmark=False
    ))

    yield client


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY REDIS MOCK
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryRedis:
    """Minimal async Redis mock for testing."""

    def __init__(self):
        self._store: dict = {}
        self._ttls:  dict = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def setex(self, key: str, ttl: int, value: str):
        self._store[key] = value
        self._ttls[key]  = ttl
        return True

    async def delete(self, key: str):
        self._store.pop(key, None)
        self._ttls.pop(key, None)

    def flushall(self):
        self._store.clear()
        self._ttls.clear()


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY DATABASE (SQLite via aiosqlite)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryDB:
    """Minimal async DB mock using an in-memory dict store."""

    def __init__(self):
        self._citations: dict = {}
        self._audit:     list = []

    async def fetchrow(self, query: str, *args):
        if args:
            return self._citations.get(args[0])
        return None

    async def execute(self, query: str, *args):
        if "INSERT INTO citations" in query and args:
            self._citations[args[0]] = {
                "citation_hash": args[0],
                "score":         args[1] if len(args) > 1 else 0.0,
                "status":        args[2] if len(args) > 2 else "in_force",
                "last_verified": datetime.utcnow(),
                "overruled_by":  None,
                "ttl_seconds":   3600,
            }
        elif "INSERT INTO audit_log" in query:
            self._audit.append({"timestamp": datetime.utcnow(), "args": args})

    @property
    def audit_entries(self):
        return list(self._audit)


@pytest_asyncio.fixture
async def veracity_db(njdg_client):
    """LiveVeracityDB with in-memory Redis mock and SQLite-backed DB mock."""
    from uli.db.njdg_client import LiveVeracityDB
    redis = InMemoryRedis()
    db    = InMemoryDB()
    yield LiveVeracityDB(redis_client=redis, db_pool=db, njdg_client=njdg_client)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE CITATIONS — 4 covering all status types
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_citations() -> List[VerifiedCitation]:
    """
    Returns 4 VerifiedCitation instances covering:
      1. IN_FORCE  — Kesavananda Bharati (landmark, 1973)
      2. IN_FORCE  — Shreya Singhal (recent SC, §66A repealed)
      3. REPEALED  — IT Act §66A (should score 0.0)
      4. SUSPENDED — Synthetic suspended provision
    """
    return [
        VerifiedCitation(
            section_id    = str(KESAVANANDA_ID),
            citation_key  = "AIR 1973 SC 1461",
            score         = 0.99,
            text          = (
                "The power of Parliament to amend the Constitution under Article 368 "
                "does not include the power to destroy or damage the basic structure "
                "or framework of the Constitution."
            ),
            act_id        = str(KESAVANANDA_ID),
            landmark_flag = True,
            court         = Court.SUPREME_COURT,
            year          = 1973,
        ),
        VerifiedCitation(
            section_id    = str(SHREYA_ID),
            citation_key  = "(2015) 5 SCC 1",
            score         = 0.99,
            text          = (
                "Section 66A of the Information Technology Act is struck down in its "
                "entirety being violative of Article 19(1)(a) and not saved by "
                "Article 19(2)."
            ),
            act_id        = str(SHREYA_ID),
            landmark_flag = True,
            court         = Court.SUPREME_COURT,
            year          = 2015,
        ),
        VerifiedCitation(
            section_id    = "it-act-66a-section",
            citation_key  = "IT Act 2000 §66A",
            score         = 0.0,   # Repealed — must score 0.0
            text          = "Section 66A — Punishment for sending offensive messages...",
            act_id        = str(SHREYA_ID),
            landmark_flag = False,
            court         = Court.SUPREME_COURT,
            year          = 2000,
        ),
        VerifiedCitation(
            section_id    = str(SUSPENDED_ID),
            citation_key  = "(2020) 7 SCC 200",
            score         = 0.50,   # Suspended — 0.50 multiplier
            text          = "Provision suspended pending review by larger bench.",
            act_id        = str(SUSPENDED_ID),
            landmark_flag = False,
            court         = Court.SUPREME_COURT,
            year          = 2020,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MOCK KNOWLEDGE PRISM (for agent tests)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_prism(sample_citations):
    """KnowledgePrism that returns pre-baked verified citations."""
    from uli.models import RetrievalResult
    prism = MagicMock()

    async def mock_retrieve(query, router, recency_config):
        # By default: return first 2 sample citations as verified
        return RetrievalResult(
            verified_citations = sample_citations[:2],
            quarantined        = [],
            trigger_loop_back  = False,
        )

    prism.retrieve = mock_retrieve
    return prism
