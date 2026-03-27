"""Prometheus metrics for the Self-Healing Decision Engine."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, generate_latest

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
recovery_actions_total = Counter(
    "decision_engine_recovery_actions_total",
    "Total number of recovery actions executed",
    ["action_type", "status"],
)

policy_evaluations_total = Counter(
    "decision_engine_policy_evaluations_total",
    "Total number of policy evaluations",
    ["policy_name", "result"],
)

alerts_received_total = Counter(
    "decision_engine_alerts_received_total",
    "Total number of anomaly alerts received",
    ["anomaly_type"],
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
recovery_duration_seconds = Histogram(
    "decision_engine_recovery_duration_seconds",
    "Duration of recovery actions in seconds",
    ["action_type"],
    buckets=(5, 10, 30, 60, 120, 300, 600),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
active_recoveries = Gauge(
    "decision_engine_active_recoveries",
    "Number of currently active recovery actions",
)

websocket_connections = Gauge(
    "decision_engine_websocket_connections",
    "Number of active WebSocket connections",
)


def get_metrics() -> bytes:
    """Return Prometheus metrics in exposition format."""
    return generate_latest()
