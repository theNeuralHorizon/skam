"""Predictive outage detection -- forecasts threshold breaches and resource exhaustion."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Optional

import numpy as np
import structlog

from app.models import PredictionResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SCORE_HISTORY_LEN = 20       # 5 min at 15s intervals
_MEMORY_HISTORY_LEN = 40      # 10 min at 15s intervals
_EMA_ALPHA = 0.3              # EMA smoothing factor
_SCORE_HORIZON_SECONDS = 300  # 5-minute prediction horizon
_MEMORY_HORIZON_SECONDS = 600 # 10-minute prediction horizon
_MEMORY_USAGE_WARN = 0.7      # warn when usage > 70% of limit
_MEMORY_USAGE_CRITICAL = 0.9  # critical when usage > 90% of limit
_REPEAT_FAILURE_MIN = 2        # minimum recoveries to flag repeat failure


class OutagePredictor:
    """Lightweight prediction layer that runs after each detection cycle.

    Three algorithms:
      A) Score Trajectory -- EMA velocity extrapolation to threshold breach
      B) Capacity Exhaustion -- linear regression on memory usage / limit
      C) Repeat Failure -- recovery count escalation
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self._threshold = threshold

        # Per-service state
        self._score_history: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=_SCORE_HISTORY_LEN)
        )
        self._memory_history: dict[str, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=_MEMORY_HISTORY_LEN)
        )
        self._ema_velocity: dict[str, float] = defaultdict(float)

    def update_and_predict(
        self,
        service: str,
        timestamp: datetime,
        ensemble_score: float,
        memory_usage: float,
        memory_limit: Optional[float],
        recent_recovery_count: int = 0,
    ) -> tuple[float, list[PredictionResult]]:
        """Run all prediction algorithms and return (score_velocity, predictions).

        Parameters
        ----------
        service : str
            The service name.
        timestamp : datetime
            Current detection timestamp.
        ensemble_score : float
            The combined IF+LSTM ensemble score (0-1).
        memory_usage : float
            Current memory usage in bytes.
        memory_limit : float | None
            Memory limit in bytes (None or 0 if unknown).
        recent_recovery_count : int
            Number of recovery actions for this service in the last 30 minutes.

        Returns
        -------
        tuple[float, list[PredictionResult]]
            (score_velocity_ema, list of predictions)
        """
        ts = timestamp.timestamp()
        predictions: list[PredictionResult] = []

        # --- Algorithm A: Score Trajectory ---
        velocity = self._update_score_velocity(service, ts, ensemble_score)
        pred_a = self._predict_score_trajectory(service, ensemble_score, velocity)
        if pred_a:
            predictions.append(pred_a)

        # --- Algorithm B: Capacity Exhaustion ---
        self._memory_history[service].append((ts, memory_usage))
        pred_b = self._predict_capacity_exhaustion(service, memory_usage, memory_limit)
        if pred_b:
            predictions.append(pred_b)

        # --- Algorithm C: Repeat Failure ---
        pred_c = self._predict_repeat_failure(service, recent_recovery_count)
        if pred_c:
            predictions.append(pred_c)

        return velocity, predictions

    # -- Algorithm A: Score Trajectory ----------------------------------------

    def _update_score_velocity(self, service: str, ts: float, score: float) -> float:
        """Compute EMA-smoothed score velocity (score-units per second)."""
        history = self._score_history[service]
        history.append((ts, score))

        if len(history) < 2:
            return 0.0

        prev_ts, prev_score = history[-2]
        dt = ts - prev_ts
        if dt <= 0:
            return self._ema_velocity[service]

        raw_velocity = (score - prev_score) / dt
        prev_ema = self._ema_velocity[service]
        ema = _EMA_ALPHA * raw_velocity + (1 - _EMA_ALPHA) * prev_ema
        self._ema_velocity[service] = ema
        return ema

    def _predict_score_trajectory(
        self, service: str, current_score: float, velocity: float
    ) -> Optional[PredictionResult]:
        """Predict threshold breach via linear extrapolation of EMA velocity."""
        if current_score >= self._threshold:
            return None  # already breached
        if velocity <= 0:
            return None  # score is stable or falling

        time_to_breach = (self._threshold - current_score) / velocity

        if time_to_breach > _SCORE_HORIZON_SECONDS:
            return None  # too far out to be reliable

        # Confidence: higher when velocity is strong and score is already elevated
        base_confidence = min(1.0, velocity / 0.01)  # 0.01/s is a fast rise
        score_factor = current_score / self._threshold  # closer to threshold = higher
        confidence = min(1.0, base_confidence * 0.6 + score_factor * 0.4)

        return PredictionResult(
            service=service,
            prediction_type="score_trajectory",
            predicted_event="threshold_breach",
            time_to_event_seconds=round(time_to_breach, 1),
            confidence=round(confidence, 3),
            current_value=round(current_score, 4),
            predicted_value=round(self._threshold, 4),
            threshold=self._threshold,
            recommended_action="scale_up",
            details={
                "velocity_per_second": round(velocity, 6),
                "horizon_seconds": _SCORE_HORIZON_SECONDS,
            },
        )

    # -- Algorithm B: Capacity Exhaustion ------------------------------------

    def _predict_capacity_exhaustion(
        self,
        service: str,
        memory_usage: float,
        memory_limit: Optional[float],
    ) -> Optional[PredictionResult]:
        """Predict OOMKill via linear regression on memory history."""
        if not memory_limit or memory_limit <= 0:
            return None

        usage_ratio = memory_usage / memory_limit

        # Immediate danger check
        if usage_ratio > _MEMORY_USAGE_CRITICAL:
            return PredictionResult(
                service=service,
                prediction_type="capacity_exhaustion",
                predicted_event="oom_kill",
                time_to_event_seconds=60.0,
                confidence=0.9,
                current_value=round(memory_usage, 0),
                predicted_value=round(memory_limit, 0),
                threshold=memory_limit,
                recommended_action="increase_resources",
                details={
                    "usage_ratio": round(usage_ratio, 3),
                    "memory_usage_mb": round(memory_usage / (1024 * 1024), 1),
                    "memory_limit_mb": round(memory_limit / (1024 * 1024), 1),
                },
            )

        history = self._memory_history[service]
        if len(history) < 4:
            return None

        # Linear regression on memory history
        times = np.array([t for t, _ in history])
        values = np.array([v for _, v in history])

        # Normalize times to avoid numerical issues
        t0 = times[0]
        times_norm = times - t0

        try:
            slope, intercept = np.polyfit(times_norm, values, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None

        if slope <= 1e-10:
            return None  # memory is stable or shrinking (guard near-zero)

        time_to_limit = (memory_limit - memory_usage) / slope

        if time_to_limit > _MEMORY_HORIZON_SECONDS or time_to_limit <= 0:
            return None

        if usage_ratio < _MEMORY_USAGE_WARN:
            return None  # not close enough to worry yet

        # Confidence from R-squared
        y_pred = slope * times_norm + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        confidence = max(0.0, min(1.0, r_squared * 0.7 + usage_ratio * 0.3))

        return PredictionResult(
            service=service,
            prediction_type="capacity_exhaustion",
            predicted_event="oom_kill",
            time_to_event_seconds=round(time_to_limit, 1),
            confidence=round(confidence, 3),
            current_value=round(memory_usage, 0),
            predicted_value=round(memory_limit, 0),
            threshold=memory_limit,
            recommended_action="increase_resources",
            details={
                "usage_ratio": round(usage_ratio, 3),
                "slope_bytes_per_sec": round(slope, 1),
                "r_squared": round(r_squared, 3),
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 1),
                "memory_limit_mb": round(memory_limit / (1024 * 1024), 1),
            },
        )

    # -- Algorithm C: Repeat Failure -----------------------------------------

    @staticmethod
    def _predict_repeat_failure(
        service: str,
        recent_recovery_count: int,
    ) -> Optional[PredictionResult]:
        """Flag services with repeated recovery actions as likely to fail again."""
        if recent_recovery_count < _REPEAT_FAILURE_MIN:
            return None

        confidence = min(1.0, recent_recovery_count / 4.0)

        return PredictionResult(
            service=service,
            prediction_type="repeat_failure",
            predicted_event="recurring_anomaly",
            time_to_event_seconds=0.0,
            confidence=round(confidence, 3),
            current_value=float(recent_recovery_count),
            predicted_value=float(recent_recovery_count + 1),
            threshold=float(_REPEAT_FAILURE_MIN),
            recommended_action="increase_resources",
            details={
                "recovery_count_30m": recent_recovery_count,
                "escalation": "resource increase recommended over restart",
            },
        )
