# ULI — Universal Legal Intelligence
### Supreme Court of India · Production-Grade LLM Framework

[![Tests](https://img.shields.io/badge/tests-pytest--asyncio-green)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Proprietary-red)]()

---

## Overview

ULI is a 7-phase, multi-agent AI framework for the Supreme Court of India that delivers:

| Guarantee | Mechanism |
|-----------|-----------|
| **Zero citation hallucination** | `confidence >= 0.98` hard threshold; repealed statutes score `0.0` |
| **IRAC-structured reasoning** | Every response: Issue → Rule → Analysis → Conclusion |
| **Multi-branch legal taxonomy** | Branch A (Public) · B (Private) · C (Specialized) |
| **Live citation verification** | NJDG API with Redis + PostgreSQL two-layer cache |
| **Token governance** | `TokenBudgetManager` enforces 2,300 token ceiling per request |

---

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  InductiveReasoningEngine (Orchestrator)                        │
│                                                                 │
│  Phase 1: LegalDNAParser (GPT-4o-mini, max 300 tokens)         │
│           Regex+spaCy Pass 1 → GPT-4o-mini Pass 2              │
│                                                                 │
│  Phase 2: BranchRouterAgent (0 LLM tokens)                     │
│           LegalDNA → Pinecone namespaces + search strategy      │
│                                                                 │
│  Phase 3: ResearcherAgent (0 LLM tokens)                       │
│    ┌─────────────────────────────────┐                          │
│    │ KnowledgePrism (5-stage RAG)    │                          │
│    │  1. BM25 sparse (ES 8.x)        │ ──┐                      │
│    │  2. Dense ANN (Pinecone)        │   ├─ asyncio.gather()    │
│    │  3. KG traversal (Neo4j 2-hop)  │ ──┘                      │
│    │  4. Reciprocal Rank Fusion      │                          │
│    │  5. Cross-Encoder reranker      │                          │
│    └─────────────────────────────────┘                          │
│           ↓ citations.score >= 0.98 only                        │
│                                                                 │
│  Phase 4: AnalystAgent (GPT-4o, max 800 tokens)                │
│           ratio/obiter extraction → IRAC draft                  │
│                                                                 │
│  Phase 5: ValidatorAgent (0 LLM tokens — NJDG API only)        │
│           Live case status + amendment verification             │
│           ↓ if > 2 fail → QuarantineLoop (max 3 iterations)    │
│                                                                 │
│  Phase 6: ScribeAgent (GPT-4o, max 1,200 tokens)               │
│           judgment / summary / brief formatting                 │
│                                                                 │
│  TokenBudgetManager — enforces 2,300 token hard ceiling         │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
FinalOutput (IRAC + SCC citations + CitationNeeded flags + metadata)
```

### Token Budget (per request)

| Phase | Agent | Max Tokens | Model |
|-------|-------|------------|-------|
| `dna_parse` | LegalDNAParser | **300** | GPT-4o-mini |
| `retrieval` | KnowledgePrism | **0** | — (vector only) |
| `analyst_irac` | AnalystAgent | **800** | GPT-4o |
| `validation` | ValidatorAgent | **0** | — (NJDG API only) |
| `scribe_output` | ScribeAgent | **1,200** | GPT-4o |
| **TOTAL CEILING** | | **2,300** | |

---

## Quick Start

### 1. Clone & configure
```bash
git clone <repo-url>
cd uli
cp .env.example .env
# Edit .env — add OPENAI_API_KEY at minimum
```

### 2. Run with Docker Compose (development)
```bash
docker compose up --build
# All 8 services: uli-api, uli-worker, postgres, redis,
#                 elasticsearch, mock-njdg, prometheus, grafana
```

### 3. Verify
```bash
# Health check
curl http://localhost:8000/health
# {"status":"ok","service":"uli-api"}

# Readiness
curl http://localhost:8000/ready
# {"status":"ready","engine":"loaded"}

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Is Parliament'\''s power to amend the Constitution unlimited?", "output_mode": "judgment"}'

# Token budget limits
curl http://localhost:8000/api/v1/budget/limits

# Metrics
curl http://localhost:8000/metrics
```

### 4. Run tests
```bash
pip install -e ".[dev]"
pytest tests/ -v --asyncio-mode=auto
```

---

## Project Structure

```
uli/
├── uli/
│   ├── models.py                # Phase 2A: All Pydantic v2 schemas
│   ├── agents/
│   │   ├── dna_parser.py        # Phase 3: LegalDNAParser + BranchRouterAgent
│   │   └── pipeline.py          # Phase 5: 4 agents + InductiveReasoningEngine
│   ├── retrieval/
│   │   └── knowledge_prism.py   # Phase 4: 5-stage hybrid RAG pipeline
│   ├── db/
│   │   ├── schema.sql            # Phase 2B: PostgreSQL DDL
│   │   ├── pinecone_config.py    # Phase 2D: Pinecone namespace config
│   │   └── njdg_client.py        # Phase 6: NJDG client + LiveVeracityDB
│   ├── api/
│   │   └── main.py               # Phase 7: FastAPI app
│   ├── utils/
│   │   ├── token_budget.py       # Phase 7D: TokenBudgetManager
│   │   └── metrics.py            # Phase 7D: Prometheus metrics
│   ├── mock_njdg.py              # Phase 6: Mock NJDG FastAPI server
│   └── worker/
│       └── __init__.py           # Celery worker (embedding + reranking)
├── alembic/
│   ├── env.py                    # Phase 2C: Async Alembic setup
│   └── versions/
│       └── 0001_initial_schema.py
├── tests/
│   ├── conftest.py               # Phase 6: Shared fixtures
│   ├── test_phase3_parser_router.py
│   ├── test_phase4_retrieval.py
│   ├── test_phase7_e2e.py        # Phase 7A: 5 E2E tests
│   └── test_token_budget.py
├── k8s/                          # Phase 7C: Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── deployment-api.yaml
│   └── workloads.yaml            # Worker, Service, Ingress, HPA
├── docker/                       # Phase 7B: Docker assets
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── Dockerfile.mock-njdg
│   ├── prometheus.yml
│   └── grafana/
├── docker-compose.yml
├── alembic.ini
├── pyproject.toml
├── requirements.txt
└── .env.example
```

---

## Non-Negotiable System Constraints

1. **Confidence Threshold**: Every citation must score `>= 0.98` or be tagged `CitationNeeded`
2. **Hierarchy of Laws**: Constitution > Central Statutes > State Laws > Subordinate Legislation
3. **Recency Rule**: Landmark judgments (`landmark_flag=True`, 5+ judge bench) override all recency penalties
4. **Repealed Statutes**: `STATUS_MULTIPLIER[REPEALED] = 0.0` — mathematically impossible to pass threshold
5. **Output Format**: Every response uses IRAC: Issue → Rule → Analysis → Conclusion
6. **Agent Order**: Researcher → Analyst → Validator → Scribe (Validator may trigger quarantine loop, max 3)
7. **Token Governance**: `TokenBudgetManager` enforces per-phase limits; total ceiling 2,300 tokens/request

---

## Post-Build Verification Checklist

```bash
# 1. All tests pass
pytest tests/ -v --asyncio-mode=auto

# 2. All 8 Docker services healthy
docker compose up --build

# 3. Readiness probe
curl localhost:8000/ready
# → {"status":"ready"}

# 4. Import check
python -c "from uli.agents.pipeline import InductiveReasoningEngine; print('imports OK')"

# 5. Repealed statute → score 0.0
pytest tests/test_phase4_retrieval.py::test_repealed_statute_scores_zero -v

# 6. Landmark override → weight >= 0.96
pytest tests/test_phase4_retrieval.py::test_landmark_override_ignores_age -v

# 7. Token budget enforcement
pytest tests/test_token_budget.py -v

# 8. Prometheus metrics
curl localhost:8000/metrics | grep uli_
```

---

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `uli_citation_quarantine_total` | Counter | Citations quarantined (by reason) |
| `uli_average_confidence_score` | Gauge | Rolling avg confidence across verified citations |
| `uli_loop_back_total` | Counter | Quarantine loop-backs triggered |
| `uli_tokens_per_phase` | Histogram | Token spend distribution per agent phase |
| `uli_request_duration_seconds` | Histogram | End-to-end latency per output mode |
| `uli_human_review_required_total` | Counter | Requests escalated for human review |

Grafana: http://localhost:3000 (admin/admin)  
Prometheus: http://localhost:9090
