"""Pydantic models for the anomaly detection service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ServiceMetrics(BaseModel):
    """Raw telemetry collected from Prometheus for a single service."""

    service: str
    timestamp: datetime
    request_rate: float = Field(default=0.0, ge=0.0, description="Requests per second")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="5xx ratio")
    p99_latency: float = Field(default=0.0, ge=0.0, description="99th-percentile latency in seconds")
    cpu_usage: float = Field(default=0.0, ge=0.0, description="CPU core-seconds per second")
    memory_usage: float = Field(default=0.0, ge=0.0, description="Memory usage in bytes")
    memory_limit: float = Field(default=0.0, ge=0.0, description="Memory limit in bytes (0 if unknown)")

    def feature_vector(self) -> list[float]:
        """Return the 5-element feature vector used by detectors."""
        return [
            self.request_rate,
            self.error_rate,
            self.p99_latency,
            self.cpu_usage,
            self.memory_usage,
        ]


class AnomalyResult(BaseModel):
    """Result of anomaly detection for a single service at one point in time."""

    service: str
    timestamp: datetime
    isolation_forest_score: float = Field(ge=0.0, le=1.0)
    lstm_score: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    is_anomaly: bool = False
    anomaly_type: Optional[str] = Field(
        default=None,
        description='Category: "latency", "error_rate", "resource", or "availability"',
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DetectorStatus(BaseModel):
    """Operational status of the anomaly detection pipeline."""

    is_trained: bool = False
    last_detection_at: Optional[datetime] = None
    services_monitored: list[str] = Field(default_factory=list)
    anomalies_detected: int = 0
    model_version: str = "1.0.0"


class AnomalyHistoryQuery(BaseModel):
    """Query parameters for the anomaly history endpoint."""

    service: Optional[str] = None
    minutes: int = Field(default=60, ge=1, le=1440)
    anomalies_only: bool = False


# ── Dashboard-compatible score models (for /anomaly/api/scores) ─────


class PerEnsembleScores(BaseModel):
    """Per-ensemble anomaly scores for dashboard comparison."""

    isolation_forest: float = Field(default=0.0, ge=0.0, le=1.0)
    lstm_autoencoder: float = Field(default=0.0, ge=0.0, le=1.0)
    if_lstm_combined: float = Field(default=0.0, ge=0.0, le=1.0)
    xgboost_lstm: float = Field(default=0.0, ge=0.0, le=1.0)
    xgboost_attention: float = Field(default=0.0, ge=0.0, le=1.0)
    xgboost_meta: float = Field(default=0.0, ge=0.0, le=1.0)


class ServiceScore(BaseModel):
    """Full score object for a single service (dashboard format)."""

    service: str
    isoforest_score: float = 0.0
    lstm_score: float = 0.0
    ensemble_score: float = 0.0
    is_anomaly: bool = False
    features: dict[str, float] = Field(default_factory=dict)
    severity_label: str = "normal"
    severity_level: int = 0
    consecutive_windows: int = 0
    score_velocity: float = 0.0
    prediction: Optional[dict[str, Any]] = None
    per_ensemble: PerEnsembleScores = Field(default_factory=PerEnsembleScores)


class ScoresResponse(BaseModel):
    """Response format for /anomaly/api/scores."""

    scores: list[ServiceScore] = Field(default_factory=list)
    threshold: float = 0.7


# ── Prediction models ─────────────────────────────────────────────────


class PredictionResult(BaseModel):
    """Output of a prediction algorithm for a single service."""

    service: str
    prediction_type: str  # "score_trajectory", "capacity_exhaustion", "repeat_failure"
    predicted_event: str  # "threshold_breach", "oom_kill", "recurring_anomaly"
    time_to_event_seconds: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    current_value: float
    predicted_value: float
    threshold: float
    recommended_action: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class PredictionAlert(BaseModel):
    """Alert sent to the decision engine for preemptive recovery."""

    service: str
    prediction_type: str
    predicted_event: str
    time_to_event_seconds: float
    confidence: float
    current_score: float
    recommended_action: Optional[str] = None
    timestamp: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class PredictionsResponse(BaseModel):
    """Response format for /anomaly/api/predictions."""

    predictions: list[PredictionResult] = Field(default_factory=list)
    generated_at: datetime
