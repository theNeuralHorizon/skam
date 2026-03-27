"""Prometheus metrics exposed by the anomaly detection service."""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Anomaly scores -- one gauge per (service, detector-stage)
# ---------------------------------------------------------------------------
anomaly_score = Gauge(
    "anomaly_score",
    "Current anomaly score for a service from a given detector stage",
    ["service", "detector"],
)

# ---------------------------------------------------------------------------
# Anomaly event counter
# ---------------------------------------------------------------------------
anomaly_detected_total = Counter(
    "anomaly_detected_total",
    "Total anomalies detected, partitioned by service and anomaly type",
    ["service", "anomaly_type"],
)

# ---------------------------------------------------------------------------
# Detection cycle latency
# ---------------------------------------------------------------------------
detection_cycle_duration_seconds = Histogram(
    "detection_cycle_duration_seconds",
    "Time taken for a full detection cycle across all services",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# Training data gauge
# ---------------------------------------------------------------------------
model_training_samples = Gauge(
    "model_training_samples",
    "Number of samples available for training a detector stage",
    ["detector"],
)
