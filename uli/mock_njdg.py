"""
ULI — Mock NJDG Server (FastAPI)
Local dev + CI stub server with seeded test data.
Run with: uvicorn uli.mock_njdg:app --port 8001
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="ULI Mock NJDG Server", version="1.0.0")


# ─────────────────────────────────────────────────────────────────────────────
# SEED DATA
# ─────────────────────────────────────────────────────────────────────────────

_KESAVANANDA_ID  = UUID("11111111-1111-1111-1111-111111111111")
_SHREYA_ID       = UUID("22222222-2222-2222-2222-222222222222")
_OVERRULED_ID    = UUID("33333333-3333-3333-3333-333333333333")
_SUSPENDED_ACT   = UUID("44444444-4444-4444-4444-444444444444")

_CASE_DB = {
    _KESAVANANDA_ID: {
        "citation_key": "AIR 1973 SC 1461",
        "status":       "decided",
        "overruled_by": None,
        "is_recent_sc": False,
        "decided_year": 1973,
        "is_landmark":  True,
        "bench_size":   13,
    },
    _SHREYA_ID: {
        "citation_key": "(2015) 5 SCC 1",
        "status":       "decided",
        "overruled_by": None,
        "is_recent_sc": True,
        "decided_year": 2015,
        "is_landmark":  True,
        "bench_size":   2,
        "act_sections_repealed": ["IT_ACT_66A"],
    },
    _OVERRULED_ID: {
        "citation_key": "(1980) 3 SCC 625",
        "status":       "decided",
        "overruled_by": "(2015) 5 SCC 1",   # Overruled by Shreya Singhal
        "is_recent_sc": False,
        "decided_year": 1980,
    },
    _SUSPENDED_ACT: {
        "citation_key": "(2020) 7 SCC 200",
        "status":       "suspended",          # Suspended provision
        "overruled_by": None,
        "is_recent_sc": True,
        "decided_year": 2020,
    },
}

_ACT_DB = {
    _KESAVANANDA_ID: {
        "name": "Constitution of India",
        "amendments": [],
    },
    _SHREYA_ID: {
        "name": "Information Technology Act 2000",
        "amendments": [
            {
                "amendment_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "act_id": str(_SHREYA_ID),
                "amending_act": "Information Technology (Amendment) Act 2008",
                "section_affected": "66A",
                "effective_date": "2009-02-05T00:00:00",
                "amendment_text": "Section 66A declared unconstitutional by SC",
                "gazette_reference": "IT Act Amendment Gazette 2009",
            }
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────────────────

class CaseStatusResponse(BaseModel):
    case_id:       str
    citation_key:  str
    status:        str
    overruled_by:  Optional[str] = None
    is_recent_sc:  bool          = False
    decided_year:  Optional[int] = None


class AmendmentItem(BaseModel):
    amendment_id:      str
    act_id:            str
    amending_act:      str
    section_affected:  str
    effective_date:    str
    amendment_text:    str
    gazette_reference: Optional[str] = None


class AmendmentsResponse(BaseModel):
    act_id:     str
    amendments: List[AmendmentItem]


class SearchHit(BaseModel):
    section_id:   str
    score:        float
    text_preview: str
    namespace:    str


class SearchResponse(BaseModel):
    hits:    List[SearchHit]
    total:   int
    took_ms: float


class IngestRequest(BaseModel):
    act_id:      str
    name:        str
    section_count: int = 0


class IngestResponse(BaseModel):
    act_id:        str
    ingested_at:   str
    section_count: int


class RatioResponse(BaseModel):
    citation_key: str
    ratio:        str
    is_landmark:  bool


class VeracityUpdateRequest(BaseModel):
    score: float


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "mock-njdg"}


@app.get("/cases/{case_id}/status", response_model=CaseStatusResponse)
async def get_case_status(case_id: UUID):
    record = _CASE_DB.get(case_id)
    if not record:
        # Return a generic decided status for unknown IDs (permissive for tests)
        return CaseStatusResponse(
            case_id      = str(case_id),
            citation_key = f"ULI-{str(case_id)[:8]}",
            status       = "decided",
            is_recent_sc = True,
            decided_year = 2020,
        )
    return CaseStatusResponse(case_id=str(case_id), **{
        k: v for k, v in record.items()
        if k in CaseStatusResponse.model_fields
    })


@app.get("/acts/{act_id}/amendments", response_model=AmendmentsResponse)
async def get_act_amendments(act_id: UUID):
    record = _ACT_DB.get(act_id, {"name": "Unknown Act", "amendments": []})
    amendments = [
        AmendmentItem(**{k: str(v) if not isinstance(v, str) else v
                        for k, v in a.items()})
        for a in record.get("amendments", [])
    ]
    return AmendmentsResponse(act_id=str(act_id), amendments=amendments)


@app.post("/search/multi-vector", response_model=SearchResponse)
async def multi_vector_search(payload: dict):
    query      = payload.get("query", "")
    namespaces = payload.get("namespaces", [])
    top_k      = payload.get("top_k", 10)

    # Stub: return seeded hits based on namespace
    hits = []
    for i, ns in enumerate(namespaces[:3]):
        hits.append(SearchHit(
            section_id   = f"sec-{ns}-{i+1:04d}",
            score        = 0.95 - i * 0.05,
            text_preview = f"[STUB] Legal provision from {ns} matching: {query[:50]}",
            namespace    = ns,
        ))

    return SearchResponse(hits=hits[:top_k], total=len(hits), took_ms=12.5)


@app.post("/ingest/act", response_model=IngestResponse)
async def ingest_act(payload: IngestRequest):
    return IngestResponse(
        act_id        = payload.act_id,
        ingested_at   = datetime.utcnow().isoformat(),
        section_count = payload.section_count,
    )


@app.get("/judgments/ratio", response_model=RatioResponse)
async def get_judgment_ratio(citation: str):
    # Return known ratios for seeded citations
    ratios = {
        "AIR 1973 SC 1461": (
            "Parliament cannot abrogate or damage the basic structure of the Constitution "
            "even while exercising its constituent power under Article 368.",
            True,
        ),
        "(2015) 5 SCC 1": (
            "Section 66A of the IT Act is unconstitutional as it imposes unreasonable "
            "restrictions on free speech under Article 19(1)(a).",
            True,
        ),
    }
    ratio_text, is_landmark = ratios.get(citation, (f"Ratio for {citation}", False))
    return RatioResponse(
        citation_key = citation,
        ratio        = ratio_text,
        is_landmark  = is_landmark,
    )


@app.put("/citations/{citation_hash}/veracity")
async def update_veracity(citation_hash: str, payload: VeracityUpdateRequest):
    if not (0.0 <= payload.score <= 1.0):
        raise HTTPException(status_code=422, detail="Score must be between 0.0 and 1.0")
    return {"citation_hash": citation_hash, "updated_score": payload.score, "success": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
