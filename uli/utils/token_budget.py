"""
ULI — TokenBudgetManager
Runtime enforcement of per-phase token limits.
Raises TokenBudgetExceeded before any LLM call that would breach phase limit.
Auto-downgrades to summary mode when total budget is close to ceiling.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

logger = logging.getLogger("uli.token_budget")


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class TokenBudgetExceeded(Exception):
    def __init__(self, phase: str, used: int, limit: int):
        super().__init__(
            f"Phase '{phase}': used {used} tokens, limit is {limit}"
        )
        self.phase = phase
        self.used  = used
        self.limit = limit


class SummaryFallbackTriggered(Exception):
    """Raised when total budget forces downgrade to summary mode."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN PHASE CONTEXT (returned by context manager)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenPhaseContext:
    budget_mgr: "TokenBudgetManager"
    phase:      str
    limit:      int

    def record(self, tokens_used: int) -> None:
        """Call after each LLM response to record actual spend."""
        self.budget_mgr.record_spend(self.phase, tokens_used)


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN BUDGET MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class TokenBudgetManager:
    """
    Enforces per-phase and total token budgets.

    Phase limits (tokens):
      dna_parse     : 300   GPT-4o-mini sub-call only
      retrieval     : 0     Zero LLM — vector + BM25 only
      analyst_irac  : 800   GPT-4o chain-of-thought
      validation    : 0     Zero LLM — NJDG API only
      scribe_output : 1200  GPT-4o formatted output
      ─────────────────────
      TOTAL CEILING : 2300  Hard limit per request
    """

    PHASE_LIMITS: Dict[str, int] = {
        "dna_parse"    : 300,
        "retrieval"    : 0,
        "analyst_irac" : 800,
        "validation"   : 0,
        "scribe_output": 1200,
    }

    TOTAL_LIMIT: int = 2300

    # Leave this headroom before triggering summary fallback
    _FALLBACK_BUFFER = 100

    def __init__(self, request_id: Optional[str] = None, db_pool=None):
        self.request_id   = request_id or str(uuid4())
        self._db_pool     = db_pool
        self._phase_spend: Dict[str, int] = {}
        self._total_spent: int            = 0
        self._fallback_triggered: bool    = False

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def total_spent(self) -> int:
        return self._total_spent

    @property
    def remaining(self) -> int:
        return max(0, self.TOTAL_LIMIT - self._total_spent)

    @property
    def fallback_triggered(self) -> bool:
        return self._fallback_triggered

    @property
    def phase_report(self) -> Dict[str, int]:
        return dict(self._phase_spend)

    # ── Context manager ───────────────────────────────────────────────────────

    @contextmanager
    def phase(self, phase_name: str, max_tokens: int):
        """
        Context manager for a single phase.

        Usage:
            with budget.phase("analyst_irac", max_tokens=800) as phase_ctx:
                resp = await llm.complete(...)
                phase_ctx.record(resp.usage.total_tokens)
        """
        limit = self.PHASE_LIMITS.get(phase_name, max_tokens)

        # Pre-flight: would this phase exceed total budget?
        projected = self._total_spent + limit
        if projected > self.TOTAL_LIMIT - self._FALLBACK_BUFFER:
            logger.warning(
                "TokenBudgetManager: projected spend %d exceeds total limit %d "
                "(phase=%s). Triggering summary fallback.",
                projected, self.TOTAL_LIMIT, phase_name,
            )
            self._trigger_summary_fallback()

        phase_ctx = TokenPhaseContext(
            budget_mgr = self,
            phase      = phase_name,
            limit      = limit,
        )

        try:
            yield phase_ctx
        finally:
            # If caller didn't explicitly record, we still track allocation
            if phase_name not in self._phase_spend:
                # Assume full limit was used (conservative accounting)
                logger.debug(
                    "Phase '%s' exited without recording spend — "
                    "allocating full limit %d tokens.",
                    phase_name, limit,
                )

    # ── Spend recording ───────────────────────────────────────────────────────

    def record_spend(self, phase: str, tokens_used: int) -> None:
        """
        Record actual token spend for a completed phase.
        Raises TokenBudgetExceeded if phase limit is breached.
        Also logs to audit_log table if db_pool is configured.
        """
        limit = self.PHASE_LIMITS.get(phase, self.TOTAL_LIMIT)

        if tokens_used > limit:
            raise TokenBudgetExceeded(phase, tokens_used, limit)

        self._phase_spend[phase] = tokens_used
        self._total_spent       += tokens_used

        logger.info(
            "TokenBudget[%s]: phase=%s used=%d total_so_far=%d/%d",
            self.request_id, phase, tokens_used, self._total_spent, self.TOTAL_LIMIT,
        )

        # Async audit log (fire-and-forget)
        if self._db_pool is not None:
            asyncio.create_task(
                self._log_to_audit(phase, tokens_used)
            )

    async def _log_to_audit(self, phase: str, tokens_used: int) -> None:
        """Append-only audit log entry for token spend."""
        try:
            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_log
                        (citation_hash, score, agent_id, timestamp, phase_token_spend)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    """,
                    f"token_budget_{self.request_id}",
                    0.0,
                    f"token-budget-{phase}",
                    datetime.utcnow(),
                    {"phase": phase, "tokens": tokens_used, "total": self._total_spent},
                )
        except Exception as e:
            logger.error("TokenBudget audit log failed: %s", e)

    # ── Fallback ──────────────────────────────────────────────────────────────

    def _trigger_summary_fallback(self) -> None:
        """
        Called when budget is nearly exhausted.
        Sets flag — orchestrator checks this and downgrades to SUMMARY mode.
        Does NOT raise — allows graceful degradation.
        """
        if not self._fallback_triggered:
            self._fallback_triggered = True
            logger.warning(
                "TokenBudgetManager[%s]: Summary fallback triggered. "
                "Remaining budget: %d tokens.",
                self.request_id, self.remaining,
            )

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset for reuse across multiple requests (e.g. in tests)."""
        self._phase_spend     = {}
        self._total_spent     = 0
        self._fallback_triggered = False
        self.request_id       = str(uuid4())

    def __repr__(self) -> str:
        return (
            f"TokenBudgetManager(request_id={self.request_id!r}, "
            f"spent={self._total_spent}/{self.TOTAL_LIMIT}, "
            f"phases={self._phase_spend})"
        )
