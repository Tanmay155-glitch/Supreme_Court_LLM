"""
ULI — Prometheus Metrics
All 6 required metrics exposed on GET /metrics via prometheus_client.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# ── Registry ──────────────────────────────────────────────────────────────────
REGISTRY = CollectorRegistry(auto_describe=True)

# ── Metric 1: Citations quarantined (labelled by reason) ─────────────────────
uli_citation_quarantine_total = Counter(
    "uli_citation_quarantine_total",
    "Total citations quarantined below confidence threshold",
    labelnames=["reason"],
    registry=REGISTRY,
)

# ── Metric 2: Rolling average confidence score across verified citations ──────
uli_average_confidence_score = Gauge(
    "uli_average_confidence_score",
    "Rolling average confidence score across all verified citations",
    registry=REGISTRY,
)

# ── Metric 3: Quarantine loop-backs triggered ─────────────────────────────────
uli_loop_back_total = Counter(
    "uli_loop_back_total",
    "Total quarantine loop-backs triggered",
    registry=REGISTRY,
)

# ── Metric 4: Token spend distribution per phase ─────────────────────────────
uli_tokens_per_phase = Histogram(
    "uli_tokens_per_phase",
    "Token spend distribution per agent phase",
    labelnames=["phase"],
    buckets=[50, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000, 2300],
    registry=REGISTRY,
)

# ── Metric 5: End-to-end request latency per output mode ─────────────────────
uli_request_duration_seconds = Histogram(
    "uli_request_duration_seconds",
    "End-to-end request latency in seconds",
    labelnames=["output_mode"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
    registry=REGISTRY,
)

# ── Metric 6: Requests requiring human escalation ────────────────────────────
uli_human_review_required_total = Counter(
    "uli_human_review_required_total",
    "Total requests that required human legal review escalation",
    registry=REGISTRY,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (called from pipeline/API layer)
# ─────────────────────────────────────────────────────────────────────────────

def record_quarantine(reason: str, count: int = 1) -> None:
    safe_reason = reason.replace(" ", "_").lower()[:64]
    uli_citation_quarantine_total.labels(reason=safe_reason).inc(count)


def record_loop_back() -> None:
    uli_loop_back_total.inc()


def record_token_spend(phase: str, tokens: int) -> None:
    uli_tokens_per_phase.labels(phase=phase).observe(tokens)


def record_request_duration(output_mode: str, duration_seconds: float) -> None:
    uli_request_duration_seconds.labels(output_mode=output_mode).observe(duration_seconds)


def record_human_review() -> None:
    uli_human_review_required_total.inc()


def update_average_confidence(score: float) -> None:
    uli_average_confidence_score.set(score)


def metrics_output() -> tuple[bytes, str]:
    """Return (body_bytes, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
