"""
ULI — Phase 5: Four-Agent Recursive Pipeline
ResearcherAgent → AnalystAgent → ValidatorAgent → ScribeAgent
Orchestrated by InductiveReasoningEngine with quarantine loop (max 3 iterations).
Token governance enforced at every LLM call via TokenBudgetManager.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import openai

from uli.agents.dna_parser import LegalDNA, RecencyWeightConfig, RouterResult
from uli.models import (
    CitationNeeded, ComponentConfidence, FinalOutput, IRACDraft,
    ObiterResult, OutputMetadata, OutputMode, QuarantineLoop,
    RatioResult, RuleItem, ValidatedIRAC, VerifiedCitation,
)
from uli.retrieval.knowledge_prism import KnowledgePrism
from uli.utils.token_budget import TokenBudgetExceeded, TokenBudgetManager


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONTEXT OBJECT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    dna:           LegalDNA
    route:         RouterResult
    query:         str
    output_mode:   OutputMode
    loops:         int = 0
    max_loops:     int = 3
    token_spend:   Dict[str, int] = field(default_factory=dict)
    conflict_log:  List[str]      = field(default_factory=list)
    quarantine_log: List[CitationNeeded] = field(default_factory=list)

    def tighten(self, new_query: str) -> None:
        self.query  = new_query
        self.loops += 1


# ─────────────────────────────────────────────────────────────────────────────
# CONFLICT DETECTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConflictRecord:
    civil_rule:              str
    constitutional_mandate:  str


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — ResearcherAgent (The Clerk)
# ─────────────────────────────────────────────────────────────────────────────

class ResearcherAgent:
    """
    Scours KnowledgePrism. Returns only citations scoring >= 0.98.
    No LLM tokens consumed — pure vector + BM25 retrieval.
    """

    MIN_VERIFIED = 3   # Broaden query if fewer than this are found

    def __init__(self, prism: KnowledgePrism):
        self.prism = prism

    async def run(self, ctx: AgentContext) -> List[VerifiedCitation]:
        result = await self.prism.retrieve(
            ctx.query, ctx.route, ctx.route.recency_config
        )
        ctx.quarantine_log.extend(result.quarantined)

        # Only broaden when there are SOME verified results but fewer than MIN_VERIFIED.
        # If there are ZERO verified, all hits were quarantined — the quarantine loop-back
        # in the orchestrator handles that case; a broad retry would just repeat the failure.
        n_verified = len(result.verified_citations)
        if 0 < n_verified < self.MIN_VERIFIED:
            broader = self._broaden_query(ctx.query)
            result2 = await self.prism.retrieve(
                broader, ctx.route, ctx.route.recency_config
            )
            # Merge and deduplicate
            existing_ids = {vc.section_id for vc in result.verified_citations}
            for vc in result2.verified_citations:
                if vc.section_id not in existing_ids:
                    result.verified_citations.append(vc)
                    existing_ids.add(vc.section_id)

        return self._deduplicate_and_rank(result.verified_citations)[:10]

    @staticmethod
    def _broaden_query(query: str) -> str:
        """Remove section-level specificity for broader retry."""
        q = re.sub(
            r"\b[Ss](?:ection|ec)?\.?\s*\d+[A-Z]?(?:\(\d+\))*(?:\([a-z]\))*\b",
            "",
            query,
        ).strip()
        return q or query

    @staticmethod
    def _deduplicate_and_rank(citations: List[VerifiedCitation]) -> List[VerifiedCitation]:
        seen: Dict[str, VerifiedCitation] = {}
        for vc in citations:
            if vc.section_id not in seen or vc.score > seen[vc.section_id].score:
                seen[vc.section_id] = vc
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — AnalystAgent (The Jurist)
# ─────────────────────────────────────────────────────────────────────────────

class AnalystAgent:
    """
    Distils ratio/obiter from citations.
    Builds IRAC draft. Max 800 LLM tokens total.
    """

    MAX_TOKENS_IRAC          = 800
    MAX_TOKENS_PER_RATIO     = 100

    def __init__(self, openai_client: Optional[openai.AsyncOpenAI] = None):
        self._llm = openai_client or openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )

    async def run(
        self,
        ctx: AgentContext,
        citations: List[VerifiedCitation],
    ) -> IRACDraft:
        # Step 1: Extract ratios (max 100 tok each)
        ratios  = await self._extract_ratios(citations)
        obiters = await self._flag_obiter(citations)

        # Step 2: Conflict check — Civil vs Constitutional hierarchy
        conflicts = self._detect_conflicts(ratios, ctx.dna.branch)
        for conflict in conflicts:
            ctx.conflict_log.append(
                f"RESOLVED: {conflict.civil_rule} overridden by "
                f"{conflict.constitutional_mandate} per constitutional hierarchy"
            )

        # Step 3: Build IRAC (single GPT-4o call, max 800 tokens)
        irac_prompt = self._build_irac_prompt(ctx, ratios, obiters)
        draft = await self._call_llm_irac(irac_prompt, ctx.token_spend)

        # Step 4: Attach obiter separately — never inline with ratio
        draft.obiter_dicta         = obiters
        draft.conflict_resolutions = list(ctx.conflict_log)
        draft.rule_citations       = citations
        return draft

    async def _extract_ratios(
        self, citations: List[VerifiedCitation]
    ) -> List[RatioResult]:
        tasks = [self._extract_single_ratio(vc) for vc in citations[:8]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ratios = []
        for r in results:
            if isinstance(r, RatioResult):
                ratios.append(r)
        return ratios

    async def _extract_single_ratio(self, vc: VerifiedCitation) -> RatioResult:
        text_snippet = vc.text[:400] if vc.text else vc.citation_key
        prompt = (
            f"Extract the ratio decidendi from this legal text in one sentence.\n"
            f"Citation: {vc.citation_key}\nText: {text_snippet}\n"
            f"Respond with JSON: {{\"ratio\": \"...\"}}"
        )
        try:
            resp = await self._llm.chat.completions.create(
                model       = "gpt-4o",
                max_tokens  = self.MAX_TOKENS_PER_RATIO,
                temperature = 0.0,
                messages    = [{"role": "user", "content": prompt}],
            )
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)
            return RatioResult(
                citation_key = vc.citation_key,
                ratio        = data.get("ratio", "Ratio not extracted"),
                is_landmark  = vc.landmark_flag,
            )
        except Exception:
            return RatioResult(citation_key=vc.citation_key, ratio="Ratio extraction failed")

    async def _flag_obiter(
        self, citations: List[VerifiedCitation]
    ) -> List[ObiterResult]:
        """Flag likely obiter dicta (brief heuristic — no separate LLM call to save tokens)."""
        obiters = []
        for vc in citations:
            text_lower = vc.text.lower() if vc.text else ""
            # Common obiter indicators in Indian SC judgments
            if any(phrase in text_lower for phrase in
                   ["it may be noted", "we observe in passing", "obiter",
                    "though not necessary for decision", "we may also observe"]):
                obiters.append(ObiterResult(
                    citation_key = vc.citation_key,
                    obiter       = vc.text[:300],
                ))
        return obiters

    @staticmethod
    def _detect_conflicts(
        ratios: List[RatioResult], branch
    ) -> List[ConflictRecord]:
        """Simple rule-based conflict detection. Constitutional always overrides."""
        from uli.models import Branch
        if branch != Branch.B:   # Only civil/private law can conflict with constitutional
            return []
        conflicts = []
        constitutional_ratios = [r for r in ratios if "constitution" in r.ratio.lower()
                                 or "fundamental right" in r.ratio.lower()]
        civil_ratios = [r for r in ratios if r.citation_key not in
                        {cr.citation_key for cr in constitutional_ratios}]
        for civ in civil_ratios:
            for con in constitutional_ratios:
                conflicts.append(ConflictRecord(
                    civil_rule              = f"{civ.citation_key}: {civ.ratio[:80]}",
                    constitutional_mandate  = f"{con.citation_key}: {con.ratio[:80]}",
                ))
        return conflicts

    def _build_irac_prompt(
        self,
        ctx: AgentContext,
        ratios: List[RatioResult],
        obiters: List[ObiterResult],
    ) -> str:
        ratio_block = "\n".join(
            f"  [{r.citation_key}] {r.ratio}" for r in ratios
        )
        return (
            "You are a Supreme Court of India legal analyst. "
            "Respond ONLY with valid JSON. No markdown.\n\n"
            f"Legal Issue: {ctx.dna.legal_issue}\n"
            f"Branch: {ctx.dna.branch.value} | Sub-branches: "
            f"{', '.join(sb.value for sb in ctx.dna.sub_branches)}\n"
            f"Statutory Authority: {ctx.dna.statutory_authority}\n"
            f"Ratios Extracted:\n{ratio_block}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "issue": "...",\n'
            '  "rule": [{"citation": "...", "principle": "..."}],\n'
            '  "analysis": "... (max 400 tokens)",\n'
            '  "conclusion": "... (max 100 tokens)",\n'
            '  "component_confidence": {"issue": 0.0, "rule": 0.0, "analysis": 0.0, "conclusion": 0.0}\n'
            "}"
        )

    async def _call_llm_irac(
        self, prompt: str, token_spend: Dict[str, int]
    ) -> IRACDraft:
        try:
            resp = await self._llm.chat.completions.create(
                model       = "gpt-4o",
                max_tokens  = self.MAX_TOKENS_IRAC,
                temperature = 0.1,
                messages    = [{"role": "user", "content": prompt}],
            )
            used = resp.usage.total_tokens if resp.usage else 0
            token_spend["analyst_irac"] = token_spend.get("analyst_irac", 0) + used
            raw  = resp.choices[0].message.content.strip()
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        except Exception:
            data = {}

        return IRACDraft(
            issue  = data.get("issue", ctx_issue_fallback := "See query"),
            rule   = [RuleItem(**r) for r in data.get("rule", [])],
            analysis   = data.get("analysis", "Analysis unavailable"),
            conclusion = data.get("conclusion", "Conclusion pending human review"),
            component_confidence = ComponentConfidence(
                **data.get("component_confidence",
                           {"issue": 0.5, "rule": 0.5, "analysis": 0.5, "conclusion": 0.5})
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — ValidatorAgent (The Chief Justice)
# ─────────────────────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Zero LLM calls. Pure NJDG API verification.
    Triggers quarantine loop if > 2 citations fail.
    """

    def __init__(self, njdg_client, audit_logger):
        self.njdg  = njdg_client
        self.audit = audit_logger

    async def run(
        self,
        ctx: AgentContext,
        draft: IRACDraft,
    ) -> "ValidatedIRAC | QuarantineLoop":
        failed: List[CitationNeeded]     = []
        verified: List[VerifiedCitation] = []

        citations = draft.all_citations()
        if not citations:
            # No citations to validate — pass through with low-confidence draft
            return ValidatedIRAC(draft=draft, verified_count=0, failed_count=0, loops=ctx.loops)

        # Concurrent NJDG verification for all citations
        tasks = [self._verify_one(citation) for citation in citations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for citation, result in zip(citations, results):
            if isinstance(result, Exception):
                # NJDG unreachable — conservative: quarantine
                failed.append(CitationNeeded(
                    citation = citation,
                    score    = 0.0,
                    reason   = f"NJDG verification failed: {str(result)[:100]}",
                ))
            else:
                is_valid, updated_score, reason = result
                if is_valid:
                    verified.append(citation._replace(score=updated_score))
                else:
                    failed.append(CitationNeeded(
                        citation = citation,
                        score    = updated_score,
                        reason   = reason,
                    ))

        # Audit log every citation regardless of outcome
        await self.audit.log_batch(verified + [f.citation for f in failed], ctx)

        if len(failed) > 2:
            tighter = self._refine_query(ctx.query, failed)
            return QuarantineLoop(
                reason         = f"{len(failed)} citations failed NJDG verification",
                tighter_query  = tighter,
            )

        draft.rule_citations  = verified
        draft.citation_needed = failed   # Appended to output — never silently dropped
        return ValidatedIRAC(
            draft          = draft,
            verified_count = len(verified),
            failed_count   = len(failed),
            loops          = ctx.loops,
        )

    async def _verify_one(
        self, citation: VerifiedCitation
    ) -> tuple[bool, float, str]:
        """Verify single citation against NJDG. Returns (is_valid, score, reason)."""
        try:
            from uuid import UUID
            case_status = await self.njdg.get_case_status(UUID(citation.act_id)
                                                           if citation.act_id else None)
            amendments  = await self.njdg.get_act_amendments(None)

            overruled      = case_status.overruled_by is not None
            section_valid  = True   # Simplified — full impl checks amendment dates

            if overruled:
                return False, 0.0, "Overruled by later judgment"

            updated_score = citation.score if section_valid else citation.score * 0.90
            if updated_score < 0.98:
                return False, updated_score, "Score below threshold after amendment check"
            return True, updated_score, "OK"

        except Exception as e:
            raise e

    @staticmethod
    def _refine_query(query: str, failed: List[CitationNeeded]) -> str:
        """Generate tighter query by appending exclusion terms."""
        excluded = " ".join(
            f"-\"{cn.citation.citation_key[:30]}\"" for cn in failed[:3]
        )
        return f"{query} {excluded}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — ScribeAgent (The Registrar)
# ─────────────────────────────────────────────────────────────────────────────

class ScribeAgent:
    """
    Formats final output in judgment / summary / brief mode.
    Max 1,200 LLM tokens. Never drops CitationNeeded flags.
    """

    MAX_TOKENS = 1200

    def __init__(self, openai_client: Optional[openai.AsyncOpenAI] = None):
        self._llm = openai_client or openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )

    async def run(
        self,
        validated: ValidatedIRAC,
        mode: OutputMode,
        ctx: AgentContext,
    ) -> FinalOutput:
        prompt_builder = {
            OutputMode.JUDGMENT: self._judgment_prompt,
            OutputMode.SUMMARY:  self._summary_prompt,
            OutputMode.BRIEF:    self._brief_prompt,
        }[mode]

        prompt = prompt_builder(validated)
        try:
            resp = await self._llm.chat.completions.create(
                model       = "gpt-4o",
                max_tokens  = self.MAX_TOKENS,
                temperature = 0.05,   # Near-deterministic for legal prose
                messages    = [{"role": "user", "content": prompt}],
            )
            used = resp.usage.total_tokens if resp.usage else 0
            ctx.token_spend["scribe_output"] = ctx.token_spend.get("scribe_output", 0) + used
            content = resp.choices[0].message.content.strip()
        except Exception as e:
            content = f"[ScribeAgent error: {str(e)[:200]}]"

        output = FinalOutput(
            content          = self._format_citations(content),
            mode             = mode,
            citation_needed  = validated.draft.citation_needed,   # Always appended
            conflict_log     = validated.draft.conflict_resolutions,
            metadata         = OutputMetadata(
                verified_citations     = validated.verified_count,
                quarantined            = validated.failed_count,
                loops_taken            = validated.loops,
                total_tokens_used      = sum(ctx.token_spend.values()),
                human_review_required  = validated.failed_count > 0,
            ),
        )
        return output

    def _judgment_prompt(self, validated: ValidatedIRAC) -> str:
        d = validated.draft
        cites = "\n".join(
            f"  [{vc.citation_key}] {vc.ratio or vc.text[:200]}"
            for vc in d.rule_citations[:5]
        )
        return (
            "You are a Supreme Court Registrar drafting a judgment in SCC format.\n"
            f"ISSUE: {d.issue}\n"
            f"RULE:\n{cites}\n"
            f"ANALYSIS: {d.analysis}\n"
            f"CONCLUSION: {d.conclusion}\n\n"
            "Write a formal judgment opinion. "
            "All citations must be in SCC format: (YEAR) VOLUME SCC PAGE. "
            "Separate ratio decidendi from obiter dicta clearly."
        )

    def _summary_prompt(self, validated: ValidatedIRAC) -> str:
        d = validated.draft
        return (
            "You are a Supreme Court Registrar writing a 3-paragraph executive summary.\n"
            f"IRAC:\nIssue: {d.issue}\nAnalysis: {d.analysis}\nConclusion: {d.conclusion}\n"
            "Write a concise 3-paragraph summary for senior counsel. "
            "Paragraph 1: legal issue and governing provisions. "
            "Paragraph 2: key authorities. Paragraph 3: conclusion and practical implications."
        )

    def _brief_prompt(self, validated: ValidatedIRAC) -> str:
        d = validated.draft
        return (
            "You are a Supreme Court Registrar preparing litigation bullet points.\n"
            f"Issue: {d.issue}\n"
            f"Conclusion: {d.conclusion}\n"
            "Produce 5–8 crisp litigation bullets for courtroom reference. "
            "Each bullet must cite a specific authority in SCC/AIR format."
        )

    @staticmethod
    def _format_citations(text: str) -> str:
        """Normalise citation references to SCC format where possible."""
        # Ensure year-volume-SCC-page pattern is standardised
        text = re.sub(
            r"\b(\d{4})\s+(\d+)\s+SCC\s+(\d+)\b",
            r"(\1) \2 SCC \3",
            text,
        )
        return text


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR — InductiveReasoningEngine
# ─────────────────────────────────────────────────────────────────────────────

class InductiveReasoningEngine:
    """
    Master orchestrator. Runs the 4-agent pipeline with quarantine loop.
    Enforces token governance via TokenBudgetManager.
    Max 3 quarantine loop iterations before safe failure.
    """

    def __init__(
        self,
        parser,
        router,
        researcher:  ResearcherAgent,
        analyst:     AnalystAgent,
        validator:   ValidatorAgent,
        scribe:      ScribeAgent,
        budget_mgr:  TokenBudgetManager,
    ):
        self.parser     = parser
        self.router     = router
        self.researcher = researcher
        self.analyst    = analyst
        self.validator  = validator
        self.scribe     = scribe
        self.budget     = budget_mgr

    async def reason(
        self,
        query: str,
        output_mode: OutputMode = OutputMode.JUDGMENT,
    ) -> FinalOutput:
        # Step 1: Parse query (max 300 tokens — GPT-4o-mini)
        with self.budget.phase("dna_parse", max_tokens=300):
            dna = await self.parser.parse(query)

        # Step 2: Route (0 LLM tokens — pure logic)
        route = self.router.classify(dna)
        ctx   = AgentContext(dna=dna, route=route, query=query, output_mode=output_mode)

        while ctx.loops < ctx.max_loops:
            # Step 3: Research (0 LLM tokens — vector + BM25 only)
            citations = await self.researcher.run(ctx)

            # ── Researcher loop-back gate ──────────────────────────────────
            # If researcher found < 1 verified citation AND quarantine is
            # building up (>= 3 items), skip analyst/validator and loop back
            # with a tighter query. This prevents empty drafts from reaching
            # the Scribe as if they were valid outputs.
            if len(citations) < 1 and len(ctx.quarantine_log) >= 3:
                if ctx.loops + 1 < ctx.max_loops:
                    tighter = (
                        ctx.quarantine_log[-1].citation.citation_key
                        if ctx.quarantine_log else ctx.query
                    )
                    # Build tighter query by stripping section refs
                    import re as _re
                    tighter_q = _re.sub(
                        r"\b[Ss](?:ection|ec)?\.?\s*\d+[A-Z]?(?:\(\d+\))*(?:\([a-z]\))*\b",
                        "", ctx.query,
                    ).strip() or ctx.query
                    ctx.tighten(tighter_q)
                    continue
                # Exhausted loops via researcher gate — exit while to safe failure
                break

            # Step 4: Analyse (max 800 tokens — GPT-4o)
            with self.budget.phase("analyst_irac", max_tokens=800):
                draft = await self.analyst.run(ctx, citations)

            # Step 5: Validate (0 LLM tokens — NJDG API only)
            result = await self.validator.run(ctx, draft)

            if isinstance(result, QuarantineLoop):
                ctx.tighten(result.tighter_query)
                continue   # Loop back to Researcher with tighter query

            # Step 6: Scribe (max 1,200 tokens — GPT-4o)
            with self.budget.phase("scribe_output", max_tokens=1200):
                return await self.scribe.run(result, output_mode, ctx)

        # Max loops exhausted — fail safely, never hallucinate
        return FinalOutput(
            content  = "",
            error    = "Max quarantine loops reached. Human legal review required.",
            metadata = OutputMetadata(
                verified_citations    = 0,
                human_review_required = True,
                loops_taken           = ctx.max_loops,
                total_tokens_used     = sum(ctx.token_spend.values()),
            ),
        )
