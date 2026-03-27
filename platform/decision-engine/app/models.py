"""Pydantic models for the Self-Healing Decision Engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnomalyAlert(BaseModel):
    """Incoming anomaly alert from the anomaly-detector service."""

    service: str
    anomaly_type: str  # "latency", "error_rate", "resource", "availability"
    severity: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    metrics: dict[str, Any] = Field(default_factory=dict)


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


class SystemStatus(BaseModel):
    """Overall system status snapshot."""

    healthy_services: list[str]
    unhealthy_services: list[str]
    active_recoveries: int
    recent_actions: list[RecoveryAction]
    policies: list[HealingPolicy]
