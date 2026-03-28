"""Severity classification for detected anomalies.

Maps raw anomaly scores to actionable severity levels with
feature-level attribution for root cause hints.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
from typing import Optional
import time
import numpy as np


class Severity(IntEnum):
    NORMAL = 0
    LOW = 1      # Score 0.5-0.65, minor drift
    MEDIUM = 2   # Score 0.65-0.8, sustained anomaly
    HIGH = 3     # Score 0.8-0.9, significant deviation
    CRITICAL = 4 # Score > 0.9, service likely down

    @property
    def label(self) -> str:
        return self.name.lower()

    @property
    def color(self) -> str:
        return {0: "green", 1: "blue", 2: "yellow", 3: "orange", 4: "red"}[self.value]

    @property
    def max_response_time_s(self) -> float:
        """Target time to begin recovery action."""
        return {0: float("inf"), 1: 120.0, 2: 60.0, 3: 30.0, 4: 15.0}[self.value]


@dataclass
class FeatureContribution:
    """Which features are driving the anomaly."""
    feature_name: str
    z_score: float
    contribution_pct: float  # 0-100, how much this feature contributes
    direction: str  # "high" or "low"


@dataclass
class SeverityAssessment:
    """Complete severity assessment for a service at a point in time."""
    service: str
    severity: Severity
    ensemble_score: float
    if_score: float
    lstm_score: float
    consecutive_anomaly_windows: int
    score_velocity: float  # rate of change per second
    top_contributors: list[FeatureContribution] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    recommended_response_time_s: float = 0.0

    def __post_init__(self):
        self.recommended_response_time_s = self.severity.max_response_time_s

    def to_dict(self) -> dict:
        return {
            "service": self.service,
            "severity": self.severity.label,
            "severity_level": self.severity.value,
            "ensemble_score": round(self.ensemble_score, 4),
            "if_score": round(self.if_score, 4),
            "lstm_score": round(self.lstm_score, 4),
            "consecutive_windows": self.consecutive_anomaly_windows,
            "score_velocity": round(self.score_velocity, 4),
            "max_response_time_s": self.recommended_response_time_s,
            "top_contributors": [
                {
                    "feature": c.feature_name,
                    "z_score": round(c.z_score, 2),
                    "contribution_pct": round(c.contribution_pct, 1),
                    "direction": c.direction,
                }
                for c in self.top_contributors
            ],
            "timestamp": self.timestamp,
        }


class SeverityClassifier:
    """Classifies anomaly severity per service using score history and feature analysis."""

    FEATURE_NAMES = ["request_rate", "error_rate", "latency", "cpu_usage", "memory_usage"]

    def __init__(
        self,
        history_size: int = 40,
        anomaly_threshold: float = 0.5,
    ):
        self._history_size = history_size
        self._anomaly_threshold = anomaly_threshold
        # Per-service tracking
        self._score_history: dict[str, deque[tuple[float, float]]] = {}  # (timestamp, score)
        self._consecutive_anomaly: dict[str, int] = {}
        self._baseline_stats: dict[str, dict] = {}  # {feature: {mean, std}}

    def _ensure_service(self, service: str) -> None:
        if service not in self._score_history:
            self._score_history[service] = deque(maxlen=self._history_size)
            self._consecutive_anomaly[service] = 0

    def update_baseline(self, service: str, features: np.ndarray) -> None:
        """Update running baseline statistics for feature-level z-scores."""
        if service not in self._baseline_stats:
            self._baseline_stats[service] = {
                "mean": features.copy(),
                "var": np.ones_like(features),
                "count": 1,
            }
        else:
            stats = self._baseline_stats[service]
            stats["count"] += 1
            alpha = min(0.05, 2.0 / (stats["count"] + 1))
            stats["mean"] = (1 - alpha) * stats["mean"] + alpha * features
            diff = features - stats["mean"]
            stats["var"] = (1 - alpha) * stats["var"] + alpha * diff ** 2
            stats["var"] = np.maximum(stats["var"], 1e-8)

    def _compute_feature_contributions(
        self, service: str, features: np.ndarray
    ) -> list[FeatureContribution]:
        """Compute which features are most anomalous using z-scores."""
        if service not in self._baseline_stats:
            return []

        stats = self._baseline_stats[service]
        std = np.sqrt(stats["var"])
        z_scores = np.abs((features - stats["mean"]) / std)
        total_z = z_scores.sum()

        if total_z < 1e-8:
            return []

        contributions = []
        for i, name in enumerate(self.FEATURE_NAMES[: len(features)]):
            if z_scores[i] > 1.5:  # Only include notable deviations
                direction = "high" if features[i] > stats["mean"][i] else "low"
                contributions.append(
                    FeatureContribution(
                        feature_name=name,
                        z_score=float(z_scores[i]),
                        contribution_pct=float(z_scores[i] / total_z * 100),
                        direction=direction,
                    )
                )

        contributions.sort(key=lambda c: c.z_score, reverse=True)
        return contributions[:5]  # Top 5 contributors

    def _compute_velocity(self, service: str) -> float:
        """Rate of score change per second (positive = worsening)."""
        history = self._score_history.get(service)
        if not history or len(history) < 3:
            return 0.0

        recent = list(history)[-5:]
        if len(recent) < 2:
            return 0.0

        dt = recent[-1][0] - recent[0][0]
        if dt < 1.0:
            return 0.0

        dscore = recent[-1][1] - recent[0][1]
        return dscore / dt

    def classify(
        self,
        service: str,
        ensemble_score: float,
        if_score: float = 0.0,
        lstm_score: float = 0.0,
        features: Optional[np.ndarray] = None,
    ) -> SeverityAssessment:
        """Classify the severity of an anomaly for a service."""
        self._ensure_service(service)
        now = time.time()

        # Update history
        self._score_history[service].append((now, ensemble_score))

        # Update consecutive anomaly count
        if ensemble_score >= self._anomaly_threshold:
            self._consecutive_anomaly[service] += 1
        else:
            self._consecutive_anomaly[service] = 0

        # Update baseline (only on normal data)
        if features is not None and ensemble_score < self._anomaly_threshold:
            self.update_baseline(service, features)

        # Compute velocity
        velocity = self._compute_velocity(service)
        consecutive = self._consecutive_anomaly[service]

        # Feature contributions
        contributors = []
        if features is not None:
            contributors = self._compute_feature_contributions(service, features)

        # ── Severity classification logic ──
        severity = Severity.NORMAL

        if ensemble_score >= 0.9:
            severity = Severity.CRITICAL
        elif ensemble_score >= 0.8:
            severity = Severity.HIGH
        elif ensemble_score >= 0.65:
            severity = Severity.MEDIUM
        elif ensemble_score >= 0.5:
            severity = Severity.LOW

        # Escalation rules: upgrade severity based on context
        if severity >= Severity.LOW:
            # Sustained anomaly escalation: 5+ consecutive windows upgrades by 1 level
            if consecutive >= 5 and severity < Severity.CRITICAL:
                severity = Severity(min(severity.value + 1, Severity.CRITICAL))

            # Rapid worsening escalation: score rising fast
            if velocity > 0.02 and severity < Severity.CRITICAL:  # >0.02/s is rapid
                severity = Severity(min(severity.value + 1, Severity.CRITICAL))

            # Both models agree it's bad: boost confidence
            if if_score > 0.7 and lstm_score > 0.7 and severity < Severity.HIGH:
                severity = Severity(min(severity.value + 1, Severity.CRITICAL))

        return SeverityAssessment(
            service=service,
            severity=severity,
            ensemble_score=ensemble_score,
            if_score=if_score,
            lstm_score=lstm_score,
            consecutive_anomaly_windows=consecutive,
            score_velocity=velocity,
            top_contributors=contributors,
            timestamp=now,
        )
