"""
ULI — Phase 6: NJDG API Client + Live Veracity Database
RS256 JWT auth, mutual TLS, Redis + PostgreSQL two-layer cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID, uuid4

import httpx
from jose import jwt
from tenacity import retry, stop_after_attempt, wait_exponential

from uli.models import (
    ActMetadata, Amendment, CaseStatus, IngestReceipt,
    RatioResult, SearchResult, VeracityResult,
)

logger = logging.getLogger("uli.njdg")


# ─────────────────────────────────────────────────────────────────────────────
# NJDG CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class NJDGClient:
    """
    Async NJDG API client.
    Auth: RS256 JWT (PEM key from env NJDG_PRIVATE_KEY).
    TLS:  mutual TLS on POST/PUT (client cert from env NJDG_CLIENT_CERT / NJDG_CLIENT_KEY).
    """

    BASE_URL = os.environ.get("NJDG_BASE_URL", "https://api.njdg.gov.in/uli/v2")

    def __init__(self):
        self._private_key = os.environ.get("NJDG_PRIVATE_KEY", "")
        self._key_id      = os.environ.get("NJDG_KEY_ID", "uli-key-1")
        client_cert       = os.environ.get("NJDG_CLIENT_CERT", "")
        client_key        = os.environ.get("NJDG_CLIENT_KEY", "")

        cert = (client_cert, client_key) if (client_cert and client_key) else None

        self._http = httpx.AsyncClient(
            base_url = self.BASE_URL,
            cert     = cert,
            timeout  = httpx.Timeout(connect=2.0, read=5.0),
            headers  = {"Content-Type": "application/json"},
        )

    # ── JWT builder ───────────────────────────────────────────────────────────

    def _build_jwt(self) -> str:
        """RS256 JWT — exp=now+300s, jti=uuid4."""
        now = datetime.now(tz=timezone.utc)
        claims = {
            "iss": "uli-system",
            "aud": "njdg-api",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=300)).timestamp()),
            "jti": str(uuid4()),
        }
        headers = {"kid": self._key_id}
        return jwt.encode(claims, self._private_key, algorithm="RS256", headers=headers)

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._build_jwt()}"}

    # ── API methods ───────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8),
           reraise=True)
    async def get_case_status(self, case_id: Optional[UUID]) -> CaseStatus:
        if case_id is None:
            return CaseStatus(case_id=uuid4(), citation_key="UNKNOWN",
                              status="unknown", is_recent_sc=False)
        resp = await self._http.get(
            f"/cases/{case_id}/status",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return CaseStatus(
            case_id      = case_id,
            citation_key = data.get("citation_key", ""),
            status       = data.get("status", "unknown"),
            overruled_by = data.get("overruled_by"),
            is_recent_sc = data.get("is_recent_sc", False),
            decided_year = data.get("decided_year"),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8),
           reraise=True)
    async def get_act_amendments(self, act_id: Optional[UUID]) -> List[Amendment]:
        if act_id is None:
            return []
        resp = await self._http.get(
            f"/acts/{act_id}/amendments",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        items = resp.json().get("amendments", [])
        return [Amendment(**item) for item in items]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8),
           reraise=True)
    async def multi_vector_search(
        self,
        query:          str,
        namespaces:     List[str],
        top_k:          int,
        min_confidence: float,
    ) -> SearchResult:
        payload = {
            "query":          query,
            "namespaces":     namespaces,
            "top_k":          top_k,
            "min_confidence": min_confidence,
        }
        resp = await self._http.post(
            "/search/multi-vector",
            json    = payload,
            headers = self._auth_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return SearchResult(
            hits    = data.get("hits", []),
            total   = data.get("total", 0),
            took_ms = data.get("took_ms", 0.0),
        )

    async def ingest_act(self, act: ActMetadata) -> IngestReceipt:
        payload = act.model_dump(mode="json")
        resp = await self._http.post(
            "/ingest/act",
            json    = payload,
            headers = self._auth_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return IngestReceipt(
            act_id        = act.act_id,
            section_count = data.get("section_count", 0),
        )

    async def get_judgment_ratio(self, citation: str) -> RatioResult:
        resp = await self._http.get(
            f"/judgments/ratio",
            params  = {"citation": citation},
            headers = self._auth_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        return RatioResult(
            citation_key = citation,
            ratio        = data.get("ratio", ""),
            is_landmark  = data.get("is_landmark", False),
        )

    async def update_veracity(self, citation_hash: str, new_score: float) -> bool:
        resp = await self._http.put(
            f"/citations/{citation_hash}/veracity",
            json    = {"score": new_score},
            headers = self._auth_headers(),
        )
        return resp.status_code == 200

    async def close(self) -> None:
        await self._http.aclose()


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogger:
    """Append-only audit logger. Writes to PostgreSQL audit_log table."""

    def __init__(self, db_pool):
        self._db = db_pool

    async def log_batch(self, citations, ctx) -> None:
        for vc in citations:
            try:
                await self._db.execute(
                    """
                    INSERT INTO audit_log
                        (citation_hash, score, agent_id, timestamp, phase_token_spend)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    """,
                    hashlib.sha256(vc.citation_key.encode()).hexdigest(),
                    float(vc.score),
                    "validator-agent",
                    datetime.utcnow(),
                    json.dumps(ctx.token_spend),
                )
            except Exception as e:
                logger.error("Audit log failed for %s: %s", vc.citation_key, e)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE VERACITY DATABASE — Two-layer cache (Redis hot + PostgreSQL persistent)
# ─────────────────────────────────────────────────────────────────────────────

class LiveVeracityDB:
    """
    Three-layer verification:
      Layer 1 — Redis  (hot cache, TTL-based)
      Layer 2 — PostgreSQL (persistent store, TTL checked in Python)
      Layer 3 — Live NJDG API call (source of truth)
    Write-through on Layer 3 miss.
    """

    def __init__(self, redis_client, db_pool, njdg_client: NJDGClient):
        self.redis = redis_client
        self.db    = db_pool
        self.njdg  = njdg_client

    async def verify(self, citation_key: str, case_id: Optional[UUID] = None) -> VeracityResult:
        hash_key = hashlib.sha256(citation_key.encode()).hexdigest()

        # ── Layer 1: Redis ─────────────────────────────────────────────────
        try:
            cached = await self.redis.get(f"ver:{hash_key}")
            if cached:
                v = VeracityResult(**json.loads(cached))
                if not self._is_expired(v):
                    return v
        except Exception:
            pass   # Redis miss / error — fall through

        # ── Layer 2: PostgreSQL ────────────────────────────────────────────
        try:
            row = await self.db.fetchrow(
                "SELECT * FROM citations WHERE citation_hash = $1", hash_key
            )
            if row and not self._is_expired_row(row):
                return self._row_to_result(row)
        except Exception:
            pass   # DB miss / error — fall through

        # ── Layer 3: Live NJDG API ─────────────────────────────────────────
        try:
            case_status = await self.njdg.get_case_status(case_id)
            score = 0.0 if case_status.overruled_by else self._compute_score(case_status)
        except Exception as e:
            logger.error("NJDG API call failed for %s: %s", citation_key, e)
            score = 0.50   # Conservative degraded score

        result = VeracityResult(
            citation_hash = hash_key,
            score         = score,
            status        = getattr(case_status, "status", "unknown"),
            last_verified = datetime.utcnow(),
            overruled_by  = getattr(case_status, "overruled_by", None),
            ttl_seconds   = 3600 if getattr(case_status, "is_recent_sc", False) else 86400,
        )

        # Write-through: Redis + PostgreSQL
        await self._write_redis(hash_key, result)
        await self._write_postgres(hash_key, result)

        # Audit log — append only
        await self._write_audit(hash_key, result.score)

        return result

    # ── TTL helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_expired(v: VeracityResult) -> bool:
        elapsed = (datetime.utcnow() - v.last_verified).total_seconds()
        return elapsed > v.ttl_seconds

    @staticmethod
    def _is_expired_row(row) -> bool:
        last_verified = row["last_verified"]
        ttl_seconds   = row["ttl_seconds"]
        if not last_verified:
            return True
        elapsed = (datetime.utcnow() - last_verified.replace(tzinfo=None)).total_seconds()
        return elapsed > ttl_seconds

    @staticmethod
    def _row_to_result(row) -> VeracityResult:
        return VeracityResult(
            citation_hash = row["citation_hash"],
            score         = float(row["score"]),
            status        = row["status"],
            last_verified = row["last_verified"].replace(tzinfo=None),
            overruled_by  = row.get("overruled_by"),
            ttl_seconds   = row["ttl_seconds"],
        )

    @staticmethod
    def _compute_score(case_status: CaseStatus) -> float:
        """Base score from case status metadata."""
        if case_status.status in ("decided", "in_force"):
            return 0.99 if case_status.is_recent_sc else 0.85
        return 0.50

    async def _write_redis(self, hash_key: str, result: VeracityResult) -> None:
        try:
            await self.redis.setex(
                f"ver:{hash_key}",
                result.ttl_seconds,
                result.model_dump_json(),
            )
        except Exception as e:
            logger.warning("Redis write failed: %s", e)

    async def _write_postgres(self, hash_key: str, result: VeracityResult) -> None:
        try:
            await self.db.execute(
                """
                INSERT INTO citations
                    (citation_hash, citation_key, score, status,
                     last_verified, overruled_by, ttl_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (citation_hash) DO UPDATE SET
                    score         = EXCLUDED.score,
                    status        = EXCLUDED.status,
                    last_verified = EXCLUDED.last_verified,
                    overruled_by  = EXCLUDED.overruled_by,
                    ttl_seconds   = EXCLUDED.ttl_seconds
                """,
                *result.db_tuple(),
                hash_key,   # citation_key (duplicate key for ON CONFLICT)
            )
        except Exception as e:
            logger.warning("PostgreSQL write failed: %s", e)

    async def _write_audit(self, hash_key: str, score: float) -> None:
        try:
            await self.db.execute(
                """
                INSERT INTO audit_log
                    (citation_hash, score, agent_id, timestamp, phase_token_spend)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                hash_key,
                score,
                "veracity-db",
                datetime.utcnow(),
                json.dumps({}),
            )
        except Exception as e:
            logger.warning("Audit log write failed: %s", e)
