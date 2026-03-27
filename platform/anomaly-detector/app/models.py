"""Pydantic models for the anomaly detection service."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

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
