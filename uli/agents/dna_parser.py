"""
ULI — Phase 3: LegalDNAParser + BranchRouterAgent
Two-pass parser: regex+spaCy NER → GPT-4o-mini classification.
Branch router maps LegalDNA to Pinecone namespaces and search strategy.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import openai
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

from uli.db.pinecone_config import (
    NAMESPACE_BRANCH_MAP, NS_CONSTITUTIONAL, NS_CORPORATE,
    NS_CRIMINAL, NS_ENVIRONMENTAL, NS_IP, NS_PRIVATE, NS_TAX,
)
from uli.models import Branch, SubBranch


# ─────────────────────────────────────────────────────────────────────────────
# LEGAL DNA DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LegalDNA:
    statutory_authority: str               # "Constitution Art.21; Passports Act 1967 §10(3)(c)"
    procedural_posture:  str               # "Writ Petition Art.32, 7-Judge Constitution Bench"
    legal_issue:         str               # Core legal question
    branch:              Branch
    sub_branches:        List[SubBranch]
    bench_strength:      Optional[int]  = None
    year_of_case:        Optional[int]  = None
    confidence:          float          = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

_RE_ARTICLE  = re.compile(r"Art(?:icle)?\.?\s*(\d+[A-Z]?(?:\(\d+\))*)", re.IGNORECASE)
_RE_SECTION  = re.compile(
    r"[Ss](?:ection|ec)?\.?\s*(\d+[A-Z]?(?:\(\d+\))*(?:\([a-z]\))*)", re.IGNORECASE
)
_RE_BENCH    = re.compile(r"(\d+)[-\s](?:Judge|Member|Bench)", re.IGNORECASE)
_RE_YEAR     = re.compile(r"\((\d{4})\)")
_RE_AIR_YEAR = re.compile(r"\bAIR\s+(\d{4})\b")


# ─────────────────────────────────────────────────────────────────────────────
# 50 CANONICAL INDIAN ACTS FOR spaCy EntityRuler
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_ACTS: List[Dict[str, str]] = [
    {"label": "ACT", "pattern": "Indian Penal Code"},
    {"label": "ACT", "pattern": "IPC"},
    {"label": "ACT", "pattern": "Code of Civil Procedure"},
    {"label": "ACT", "pattern": "CPC"},
    {"label": "ACT", "pattern": "Code of Criminal Procedure"},
    {"label": "ACT", "pattern": "CrPC"},
    {"label": "ACT", "pattern": "Constitution of India"},
    {"label": "ACT", "pattern": "Income Tax Act"},
    {"label": "ACT", "pattern": "Income-tax Act"},
    {"label": "ACT", "pattern": "Information Technology Act"},
    {"label": "ACT", "pattern": "IT Act"},
    {"label": "ACT", "pattern": "Companies Act"},
    {"label": "ACT", "pattern": "Insolvency and Bankruptcy Code"},
    {"label": "ACT", "pattern": "IBC"},
    {"label": "ACT", "pattern": "Environment Protection Act"},
    {"label": "ACT", "pattern": "Copyright Act"},
    {"label": "ACT", "pattern": "Patents Act"},
    {"label": "ACT", "pattern": "Trade Marks Act"},
    {"label": "ACT", "pattern": "Hindu Marriage Act"},
    {"label": "ACT", "pattern": "Transfer of Property Act"},
    {"label": "ACT", "pattern": "Indian Contract Act"},
    {"label": "ACT", "pattern": "Specific Relief Act"},
    {"label": "ACT", "pattern": "Arbitration and Conciliation Act"},
    {"label": "ACT", "pattern": "Prevention of Corruption Act"},
    {"label": "ACT", "pattern": "PMLA"},
    {"label": "ACT", "pattern": "Prevention of Money-Laundering Act"},
    {"label": "ACT", "pattern": "NDPS Act"},
    {"label": "ACT", "pattern": "Narcotic Drugs and Psychotropic Substances Act"},
    {"label": "ACT", "pattern": "Representation of the People Act"},
    {"label": "ACT", "pattern": "Right to Information Act"},
    {"label": "ACT", "pattern": "RTI Act"},
    {"label": "ACT", "pattern": "Consumer Protection Act"},
    {"label": "ACT", "pattern": "Competition Act"},
    {"label": "ACT", "pattern": "Foreign Exchange Management Act"},
    {"label": "ACT", "pattern": "FEMA"},
    {"label": "ACT", "pattern": "Securities and Exchange Board of India Act"},
    {"label": "ACT", "pattern": "SEBI Act"},
    {"label": "ACT", "pattern": "Banking Regulation Act"},
    {"label": "ACT", "pattern": "Negotiable Instruments Act"},
    {"label": "ACT", "pattern": "Motor Vehicles Act"},
    {"label": "ACT", "pattern": "Land Acquisition Act"},
    {"label": "ACT", "pattern": "Passports Act"},
    {"label": "ACT", "pattern": "Finance Act"},
    {"label": "ACT", "pattern": "Customs Act"},
    {"label": "ACT", "pattern": "Goods and Services Tax"},
    {"label": "ACT", "pattern": "GST"},
    {"label": "ACT", "pattern": "Air Prevention and Control of Pollution Act"},
    {"label": "ACT", "pattern": "Water Act"},
    {"label": "ACT", "pattern": "Forest Conservation Act"},
    {"label": "ACT", "pattern": "Wildlife Protection Act"},
]


def _load_spacy_model() -> Language:
    """Load spaCy model with custom entity ruler for Indian Acts."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback: blank English model
        nlp = spacy.blank("en")

    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
        ruler.add_patterns(CANONICAL_ACTS)

    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# LEGAL DNA PARSER
# ─────────────────────────────────────────────────────────────────────────────

class LegalDNAParser:
    """
    Two-pass parser:
      Pass 1 — regex + spaCy NER  (zero LLM tokens)
      Pass 2 — GPT-4o-mini sub-call capped at 300 tokens for taxonomy
    """

    def __init__(self, openai_client: Optional[openai.AsyncOpenAI] = None):
        self._nlp    = _load_spacy_model()
        self._client = openai_client or openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )

    # ── Public entry point ───────────────────────────────────────────────────

    async def parse(self, query: str) -> LegalDNA:
        pass1 = self._pass1_regex_spacy(query)
        dna   = await self._pass2_gpt_mini(query, pass1)
        return dna

    # ── Pass 1: Regex + spaCy ────────────────────────────────────────────────

    def _pass1_regex_spacy(self, text: str) -> Dict[str, Any]:
        doc = self._nlp(text)

        articles   = _RE_ARTICLE.findall(text)
        sections   = _RE_SECTION.findall(text)
        bench_m    = _RE_BENCH.search(text)
        year_m     = _RE_YEAR.findall(text) or _RE_AIR_YEAR.findall(text)

        acts = [ent.text for ent in doc.ents if ent.label_ == "ACT"]

        bench_strength = int(bench_m.group(1)) if bench_m else None
        year_of_case   = int(year_m[0]) if year_m else None

        return {
            "articles":      articles,
            "sections":      sections,
            "acts":          acts,
            "bench_strength": bench_strength,
            "year_of_case":   year_of_case,
        }

    # ── Pass 2: GPT-4o-mini (max 300 tokens) ─────────────────────────────────

    async def _pass2_gpt_mini(
        self, query: str, entities: Dict[str, Any]
    ) -> LegalDNA:
        system_prompt = (
            "You are a legal taxonomy classifier for the Supreme Court of India. "
            "Respond ONLY with valid JSON — no markdown, no explanation."
        )
        user_prompt = (
            f"Query: {query}\n"
            f"Extracted entities: {json.dumps(entities)}\n\n"
            "Classify and return JSON with these exact keys:\n"
            "  branch: 'A' | 'B' | 'C'\n"
            "  sub_branches: array of strings from "
            "[Constitutional,Administrative,Criminal,Contract,Tort,Property,"
            "Family,Tax,IP,Environmental,Insolvency]\n"
            "  legal_issue: one sentence\n"
            "  statutory_authority: precise statute + section refs\n"
            "  procedural_posture: writ type, bench info\n"
            "  confidence: float 0.0-1.0\n"
        )

        try:
            response = await self._client.chat.completions.create(
                model       = "gpt-4o-mini",
                max_tokens  = 300,           # Hard cap — never exceed
                temperature = 0.0,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw  = response.choices[0].message.content.strip()
            data = json.loads(raw)
        except Exception:
            # Fallback: use regex-only entities with low confidence
            data = {
                "branch":              "A",
                "sub_branches":        ["Constitutional"],
                "legal_issue":         query[:200],
                "statutory_authority": ", ".join(entities.get("acts", [])),
                "procedural_posture":  "Unknown",
                "confidence":          0.30,
            }

        # Map string sub_branches to enum values
        sub_branches = []
        for sb in data.get("sub_branches", ["Constitutional"]):
            try:
                sub_branches.append(SubBranch(sb))
            except ValueError:
                pass
        if not sub_branches:
            sub_branches = [SubBranch.CONSTITUTIONAL]

        return LegalDNA(
            statutory_authority = data.get("statutory_authority", ""),
            procedural_posture  = data.get("procedural_posture", ""),
            legal_issue         = data.get("legal_issue", query[:200]),
            branch              = Branch(data.get("branch", "A")),
            sub_branches        = sub_branches,
            bench_strength      = entities.get("bench_strength"),
            year_of_case        = entities.get("year_of_case"),
            confidence          = float(data.get("confidence", 0.5)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER ENUMS AND DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

class SearchMode(str, Enum):
    SINGLE_VECTOR         = "single_vector"
    MULTI_VECTOR          = "multi_vector"
    CROSS_NAMESPACE_FUSION = "cross_namespace_fusion"


@dataclass
class RecencyWeightConfig:
    sc_recent:         float = 0.90    # Recent Supreme Court decisions
    hc_recent:         float = 0.65    # Recent High Court decisions
    landmark_override: float = 0.96    # Cannot be reduced by age
    historical_cap:    float = 0.40    # Pre-1950 non-landmark ceiling
    privy_council_cap: float = 0.30    # Pre-independence Privy Council


@dataclass
class RouterResult:
    namespaces:           List[str]
    search_mode:          SearchMode
    recency_config:       RecencyWeightConfig
    router_confidence:    float
    constitutional_boost: float = 1.0   # 1.3 if constitutional namespace involved


# ─────────────────────────────────────────────────────────────────────────────
# BRANCH ROUTER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class BranchRouterAgent:
    """
    Routes LegalDNA to correct Pinecone namespaces and sets search strategy.
    Zero LLM calls — pure deterministic logic.
    """

    DEFAULT_RECENCY = RecencyWeightConfig()

    def classify(self, dna: LegalDNA) -> RouterResult:
        primary = dna.sub_branches[0] if dna.sub_branches else SubBranch.CONSTITUTIONAL
        namespaces, mode = self._route(dna.sub_branches, primary)

        constitutional_boost = (
            1.3 if NS_CONSTITUTIONAL in namespaces else 1.0
        )

        return RouterResult(
            namespaces            = namespaces,
            search_mode           = mode,
            recency_config        = self.DEFAULT_RECENCY,
            router_confidence     = dna.confidence,
            constitutional_boost  = constitutional_boost,
        )

    # ── Routing table (match/case on primary sub_branch) ─────────────────────

    def _route(
        self, sub_branches: List[SubBranch], primary: SubBranch
    ) -> tuple[List[str], SearchMode]:

        # Multi-branch: fuse all relevant namespaces
        if len(sub_branches) > 1:
            seen: List[str] = []
            for sb in sub_branches:
                for ns in NAMESPACE_BRANCH_MAP.get(sb, []):
                    if ns not in seen:
                        seen.append(ns)
            return seen, SearchMode.CROSS_NAMESPACE_FUSION

        match primary:
            case SubBranch.TAX:
                return [NS_TAX, NS_CONSTITUTIONAL], SearchMode.MULTI_VECTOR
            case SubBranch.CONSTITUTIONAL:
                return [NS_CONSTITUTIONAL], SearchMode.SINGLE_VECTOR
            case SubBranch.ADMINISTRATIVE:
                return [NS_CONSTITUTIONAL], SearchMode.SINGLE_VECTOR
            case SubBranch.CRIMINAL:
                return [NS_CRIMINAL], SearchMode.SINGLE_VECTOR
            case SubBranch.IP:
                return [NS_IP], SearchMode.SINGLE_VECTOR
            case SubBranch.ENVIRONMENTAL:
                return [NS_ENVIRONMENTAL, NS_CORPORATE], SearchMode.MULTI_VECTOR
            case SubBranch.INSOLVENCY:
                return [NS_CORPORATE], SearchMode.SINGLE_VECTOR
            case SubBranch.CONTRACT | SubBranch.TORT | SubBranch.PROPERTY | SubBranch.FAMILY:
                return [NS_PRIVATE], SearchMode.SINGLE_VECTOR
            case _:
                return [NS_CONSTITUTIONAL], SearchMode.SINGLE_VECTOR
