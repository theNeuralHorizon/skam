"""Pydantic models for the Self-Healing Decision Engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class FeatureContributionInfo(BaseModel):
    """Serialised feature contribution from the severity classifier."""

    feature: str
    z_score: float = 0.0
    contribution_pct: float = 0.0
    direction: str = "high"


class AnomalyAlert(BaseModel):
    """Incoming anomaly alert from the anomaly-detector service."""

    service: str
    anomaly_type: str  # "latency", "error_rate", "resource", "availability"
    severity: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    metrics: dict[str, Any] = Field(default_factory=dict)

    # Severity classification fields (populated when severity classifier is active)
    severity_label: str = Field(
        default="normal",
        description='Severity level: "normal", "low", "medium", "high", "critical"',
    )
    severity_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Numeric severity: 0=normal, 1=low, 2=medium, 3=high, 4=critical",
    )
    consecutive_anomaly_windows: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive detection windows with anomalous scores",
    )
    score_velocity: float = Field(
        default=0.0,
        description="Rate of score change per second (positive = worsening)",
    )
    top_contributors: list[FeatureContributionInfo] = Field(
        default_factory=list,
        description="Features driving the anomaly, ordered by contribution",
    )


class RecoveryAction(BaseModel):
    """A recovery action created in response to an anomaly alert."""

    id: str
    alert: AnomalyAlert
    action_type: str  # "restart_pod", "scale_up", "rolling_restart", "remove_network_policy", "increase_resources", "restart_redis"
    target_service: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # "pending", "executing", "validating", "success", "failed", "rolled_back"
    started_at: datetime
    completed_at: Optional[datetime] = None
    validation_result: Optional[dict[str, Any]] = None


class HealingPolicy(BaseModel):
    """Policy that maps anomaly conditions to recovery actions."""

    name: str
    anomaly_type: str
    severity_threshold: float = Field(ge=0.0, le=1.0)
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    cooldown_seconds: int = 120
    max_retries: int = 2

    # Severity-aware policy fields
    min_severity_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Minimum severity_level required to trigger (0=any, 4=critical only)",
    )
    critical_cooldown_seconds: int = Field(
        default=0,
        ge=0,
        description="Override cooldown for CRITICAL severity (0 = use normal cooldown)",
    )
    escalation_after_cycles: int = Field(
        default=3,
        ge=1,
        description="Escalate action after this many HIGH+ severity cycles without recovery",
    )


class SystemStatus(BaseModel):
    """Overall system status snapshot."""

    healthy_services: list[str]
    unhealthy_services: list[str]
    active_recoveries: int
    recent_actions: list[RecoveryAction]
    policies: list[HealingPolicy]
