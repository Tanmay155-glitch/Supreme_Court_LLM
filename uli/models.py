"""
ULI — Universal Legal Intelligence
Phase 2A: Unified metadata schema for all Indian legislation 1860–2026.
Pydantic v2 with full validators, computed fields, and type safety.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class Branch(str, Enum):
    A = "A"   # Public Law
    B = "B"   # Private Law
    C = "C"   # Specialized


class SubBranch(str, Enum):
    # Branch A — Public Law
    CONSTITUTIONAL = "Constitutional"
    ADMINISTRATIVE = "Administrative"
    CRIMINAL       = "Criminal"
    # Branch B — Private Law
    CONTRACT       = "Contract"
    TORT           = "Tort"
    PROPERTY       = "Property"
    FAMILY         = "Family"
    # Branch C — Specialized
    TAX            = "Tax"
    IP             = "IP"
    ENVIRONMENTAL  = "Environmental"
    INSOLVENCY     = "Insolvency"


class ActStatus(str, Enum):
    IN_FORCE  = "in_force"
    REPEALED  = "repealed"
    AMENDED   = "amended"
    SUSPENDED = "suspended"


class OutputMode(str, Enum):
    JUDGMENT = "judgment"   # Full SCC-format prose
    SUMMARY  = "summary"    # 3-paragraph executive
    BRIEF    = "brief"      # Litigation bullet points


class Court(str, Enum):
    SUPREME_COURT = "Supreme Court"
    HIGH_COURT    = "High Court"
    TRIBUNAL      = "Tribunal"
    PRIVY_COUNCIL = "Privy Council"
    OTHER         = "Other"


# ─────────────────────────────────────────────────────────────────────────────
# BLUEBOOK CITATION REGEX
# ─────────────────────────────────────────────────────────────────────────────

# Indian SCC format: (YEAR) VOLUME SCC PAGE  or  AIR YEAR COURT PAGE
CITATION_KEY_RE = re.compile(
    r"^(?:"
    r"\(\d{4}\)\s+\d+\s+SCC\s+\d+"             # (2015) 5 SCC 1
    r"|AIR\s+\d{4}\s+SC\s+\d+"                  # AIR 1973 SC 1461
    r"|\d{4}\s+\(\d+\)\s+SCC\s+\d+"             # 1985 (3) SCC 545
    r"|ILR\s+\d{4}\s+\d+\s+[A-Z]+\s+\d+"        # ILR 1908 35 Cal 60
    r")$"
)


# ─────────────────────────────────────────────────────────────────────────────
# CORE DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Amendment(BaseModel):
    amendment_id:      UUID     = Field(default_factory=uuid4)
    act_id:            UUID
    amending_act:      str
    section_affected:  str
    effective_date:    datetime
    amendment_text:    str
    gazette_reference: Optional[str] = None

    model_config = {"frozen": True}


class Section(BaseModel):
    section_id:   UUID      = Field(default_factory=uuid4)
    act_id:       UUID
    section_num:  str                        # e.g. "10", "10A", "10(3)(c)"
    title:        Optional[str] = None
    text:         str
    embedding:    Optional[List[float]] = None  # ada-002 = 1536 dims, BGE-M3 = 768
    sub_branch:   SubBranch
    is_repealed:  bool = False

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dim(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) not in (1536, 768):
            raise ValueError(
                f"Embedding must be 1536 (ada-002) or 768 (BGE-M3), got {len(v)}"
            )
        return v

    model_config = {"frozen": False}


class ActMetadata(BaseModel):
    act_id:           UUID        = Field(default_factory=uuid4)
    name:             str         = Field(min_length=3, max_length=512)
    short_title:      str
    year_enacted:     int
    status:           ActStatus   = ActStatus.IN_FORCE
    branch:           Branch
    sub_branch:       SubBranch
    gazette_number:   Optional[str]      = None
    landmark_flag:    bool                = False   # 5+ judge bench
    overruled_by:     Optional[str]      = None    # Citation key of overruling judgment
    sections:         List[Section]      = Field(default_factory=list)
    amendment_history: List[Amendment]   = Field(default_factory=list)
    citation_key:     str                          # SCC / AIR / ILR format
    recency_weight:   float              = 0.75

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("year_enacted")
    @classmethod
    def validate_year(cls, v: int) -> int:
        current_year = datetime.utcnow().year
        if not (1860 <= v <= current_year):
            raise ValueError(f"year_enacted must be between 1860 and {current_year}, got {v}")
        return v

    @field_validator("citation_key")
    @classmethod
    def validate_citation_key(cls, v: str) -> str:
        if not CITATION_KEY_RE.match(v.strip()):
            raise ValueError(
                f"citation_key must be in Bluebook/SCC/AIR format, got: '{v}'"
            )
        return v.strip()

    @model_validator(mode="after")
    def apply_business_rules(self) -> "ActMetadata":
        # Repealed acts → zero weight
        if self.status == ActStatus.REPEALED:
            object.__setattr__(self, "recency_weight", 0.0)

        # Overruled acts → zero weight
        if self.overruled_by is not None:
            object.__setattr__(self, "recency_weight", 0.0)

        # Landmark boost: +0.15, cap at 1.0
        if self.landmark_flag and self.recency_weight > 0:
            boosted = min(1.0, self.recency_weight + 0.15)
            object.__setattr__(self, "recency_weight", round(boosted, 4))

        # Clamp to 0.0–1.0
        clamped = max(0.0, min(1.0, self.recency_weight))
        object.__setattr__(self, "recency_weight", round(clamped, 4))
        return self

    # ── Computed fields ───────────────────────────────────────────────────────

    @computed_field  # type: ignore[misc]
    @property
    def citation_hash(self) -> str:
        return hashlib.sha256(self.citation_key.encode()).hexdigest()

    model_config = {"frozen": False}


class CitationRecord(BaseModel):
    citation_hash:  str
    citation_key:   str
    score:          float
    status:         ActStatus
    last_verified:  datetime     = Field(default_factory=datetime.utcnow)
    overruled_by:   Optional[str] = None
    ttl_seconds:    int           = 3600

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    model_config = {"frozen": False}


class AuditEntry(BaseModel):
    id:                UUID     = Field(default_factory=uuid4)
    citation_hash:     str
    score:             float
    agent_id:          str
    timestamp:         datetime = Field(default_factory=datetime.utcnow)
    phase_token_spend: dict     = Field(default_factory=dict)  # JSONB in PG

    model_config = {"frozen": True}   # Immutable — append-only


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL / PIPELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class SparseHit(BaseModel):
    section_id: str
    score:      float
    text:       str
    act_id:     str


class DenseHit(BaseModel):
    section_id: str
    score:      float
    metadata:   dict


class KGHit(BaseModel):
    node_id:           str
    relationship_type: str
    hop_distance:      int


class RankedHit(BaseModel):
    section_id:     str
    reranker_score: float
    metadata:       dict


class CitationScore(BaseModel):
    score:  float
    passed: bool
    reason: Optional[str] = None


class VerifiedCitation(BaseModel):
    section_id:    str
    citation_key:  str
    score:         float
    text:          str
    act_id:        str
    landmark_flag: bool   = False
    court:         Court  = Court.SUPREME_COURT
    year:          Optional[int] = None
    ratio:         Optional[str] = None

    def _replace(self, **kwargs) -> "VerifiedCitation":
        data = self.model_dump()
        data.update(kwargs)
        return VerifiedCitation(**data)


class CitationNeeded(BaseModel):
    citation:    VerifiedCitation
    score:       float
    reason:      str


class RetrievalResult(BaseModel):
    verified_citations: List[VerifiedCitation]  = Field(default_factory=list)
    quarantined:        List[CitationNeeded]     = Field(default_factory=list)
    trigger_loop_back:  bool                     = False
    tighter_query:      Optional[str]            = None


# ─────────────────────────────────────────────────────────────────────────────
# IRAC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class RuleItem(BaseModel):
    citation:  str
    principle: str


class ComponentConfidence(BaseModel):
    issue:      float
    rule:       float
    analysis:   float
    conclusion: float


class RatioResult(BaseModel):
    citation_key: str
    ratio:        str
    is_landmark:  bool = False


class ObiterResult(BaseModel):
    citation_key: str
    obiter:       str


class IRACDraft(BaseModel):
    issue:                str
    rule:                 List[RuleItem]            = Field(default_factory=list)
    analysis:             str
    conclusion:           str
    component_confidence: ComponentConfidence
    obiter_dicta:         List[ObiterResult]        = Field(default_factory=list)
    conflict_resolutions: List[str]                 = Field(default_factory=list)
    rule_citations:       List[VerifiedCitation]    = Field(default_factory=list)
    citation_needed:      List[CitationNeeded]      = Field(default_factory=list)

    def all_citations(self) -> List[VerifiedCitation]:
        return list(self.rule_citations)


class ValidatedIRAC(BaseModel):
    draft:          IRACDraft
    verified_count: int
    failed_count:   int
    loops:          int = 0


class QuarantineLoop(BaseModel):
    reason:        str
    tighter_query: str


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class OutputMetadata(BaseModel):
    verified_citations:    int
    quarantined:           int   = 0
    loops_taken:           int   = 0
    total_tokens_used:     int   = 0
    human_review_required: bool  = False


class FinalOutput(BaseModel):
    content:         str
    mode:            OutputMode              = OutputMode.JUDGMENT
    citation_needed: List[CitationNeeded]   = Field(default_factory=list)
    conflict_log:    List[str]              = Field(default_factory=list)
    metadata:        Optional[OutputMetadata] = None
    error:           Optional[str]           = None


# ─────────────────────────────────────────────────────────────────────────────
# NJDG / VERACITY MODELS
# ─────────────────────────────────────────────────────────────────────────────

class CaseStatus(BaseModel):
    case_id:       UUID
    citation_key:  str
    status:        str
    overruled_by:  Optional[str] = None
    is_recent_sc:  bool          = False
    decided_year:  Optional[int] = None


class IngestReceipt(BaseModel):
    act_id:       UUID
    ingested_at:  datetime = Field(default_factory=datetime.utcnow)
    section_count: int


class SearchResult(BaseModel):
    hits:   List[dict]
    total:  int
    took_ms: float


class VeracityResult(BaseModel):
    citation_hash: str
    score:         float
    status:        str
    last_verified: datetime
    overruled_by:  Optional[str] = None
    ttl_seconds:   int = 3600

    def db_tuple(self) -> tuple:
        return (
            self.citation_hash,
            self.score,
            self.status,
            self.last_verified.isoformat(),
            self.overruled_by,
            self.ttl_seconds,
        )
