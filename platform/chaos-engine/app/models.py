"""Pydantic models for the SKAM Chaos Engine."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExperimentTarget(BaseModel):
    """Defines which Kubernetes resources to target."""

    namespace: str = "default"
    label_selector: str = Field(
        ...,
        description="Kubernetes label selector (e.g. 'app=payment-service')",
        examples=["app=payment-service", "tier=backend,env=staging"],
    )


class ExperimentConfig(BaseModel):
    """Configuration for a single chaos experiment."""

    name: str = Field(..., min_length=1, max_length=128)
    target: ExperimentTarget
    fault_type: Literal[
        "pod_kill",
        "pod_crash_loop",
        "cpu_stress",
        "memory_pressure",
        "network_partition",
        "latency_injection",
    ]
    parameters: dict = Field(default_factory=dict)
    duration_seconds: int = Field(default=60, ge=1, le=3600)


class ExperimentStatus(BaseModel):
    """Current status of a running or completed experiment."""

    id: str
    name: str
    fault_type: str
    status: Literal["running", "completed", "failed", "rolled_back"]
    started_at: datetime
    ended_at: Optional[datetime] = None
    target: ExperimentTarget
    recovery_info: Optional[dict] = None
    error_message: Optional[str] = None


class ScenarioConfig(BaseModel):
    """A multi-step chaos scenario that runs experiments sequentially."""

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="")
    steps: list[ExperimentConfig] = Field(..., min_length=1)
    delay_between_steps: int = Field(default=10, ge=0, le=300)
