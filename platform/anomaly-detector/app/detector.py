"""Ensemble anomaly detection pipeline -- combines Isolation Forest + LSTM."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import structlog

from app.isolation_forest import IsolationForestDetector
from app.lstm_detector import LSTMDetector
from app.metrics import anomaly_detected_total, anomaly_score
from app.models import AnomalyResult, ServiceMetrics

logger = structlog.get_logger(__name__)

_COLD_START_SAMPLES = 200
_ANOMALY_THRESHOLD = 0.7
_CONSECUTIVE_WINDOWS = 2
_IF_WEIGHT = 0.4
_LSTM_WEIGHT = 0.6

# Feature index -> anomaly type mapping
_FEATURE_ANOMALY_MAP: dict[int, str] = {
    0: "availability",   # request_rate drop
    1: "error_rate",     # error_rate spike
    2: "latency",        # p99_latency spike
    3: "resource",       # cpu_usage spike
    4: "resource",       # memory_usage spike
}


class AnomalyDetector:
    """Two-stage anomaly detection ensemble.

    Stage 1: Isolation Forest for point anomalies (per-sample).
    Stage 2: LSTM Autoencoder for temporal anomalies (sliding window).

    The first ``_COLD_START_SAMPLES`` per service are assumed to represent
    normal behaviour and are used exclusively for training.
    """

    def __init__(self) -> None:
        self._if_detector = IsolationForestDetector()
        self._lstm_detector = LSTMDetector()

        # Per-service sample counts for cold-start handling
        self._sample_counts: dict[str, int] = defaultdict(int)

        # Consecutive anomaly window tracker: service -> deque of bools
        self._anomaly_windows: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=_CONSECUTIVE_WINDOWS)
        )

        # History ring-buffer (last ~60 min at 15 s intervals ~ 240 entries)
        self._history: deque[AnomalyResult] = deque(maxlen=2400)
        self._anomaly_count: int = 0
        self._last_detection: datetime | None = None
        self._model_version: str = "1.0.0"

    # -- public API ----------------------------------------------------------

    @property
    def anomaly_count(self) -> int:
        return self._anomaly_count

    @property
    def last_detection_at(self) -> datetime | None:
        return self._last_detection

    @property
    def model_version(self) -> str:
        return self._model_version

    def is_trained(self) -> bool:
        return self._if_detector.is_trained

    def history(
        self,
        service: str | None = None,
        since: datetime | None = None,
        anomalies_only: bool = False,
    ) -> list[AnomalyResult]:
        """Return anomaly results matching the given filters."""
        out: list[AnomalyResult] = []
        for r in self._history:
            if service and r.service != service:
                continue
            if since and r.timestamp < since:
                continue
            if anomalies_only and not r.is_anomaly:
                continue
            out.append(r)
        return out

    def current_anomalies(self) -> list[AnomalyResult]:
        """Return the most recent result per service where is_anomaly=True."""
        latest: dict[str, AnomalyResult] = {}
        for r in reversed(self._history):
            if r.service in latest:
                continue
            if r.is_anomaly:
                latest[r.service] = r
        return list(latest.values())

    async def detect(self, metrics: ServiceMetrics) -> AnomalyResult:
        """Run the full detection pipeline for a single service snapshot."""
        features = np.array(metrics.feature_vector(), dtype=np.float64)
        service = metrics.service

        self._sample_counts[service] += 1
        count = self._sample_counts[service]

        # Always feed data into both detectors
        self._if_detector.add_sample(features)
        self._lstm_detector.push(service, features.astype(np.float32))

        # --- cold start: train only, no scoring ---
        if count <= _COLD_START_SAMPLES:
            if count == _COLD_START_SAMPLES:
                logger.info("detector.cold_start_complete", service=service, samples=count)
                self._if_detector.train()
                self._lstm_detector.train()
            return self._make_result(metrics, 0.0, 0.0, 0.0)

        # --- Stage 1: Isolation Forest ---
        if_score = self._if_detector.predict(features)
        anomaly_score.labels(service=service, detector="isolation_forest").set(if_score)

        # --- Stage 2: LSTM Autoencoder ---
        lstm_score = 0.0
        if self._lstm_detector.window_ready(service) and self._lstm_detector.is_trained:
            lstm_score = self._lstm_detector.predict_for_service(service)
            anomaly_score.labels(service=service, detector="lstm").set(lstm_score)

        # --- Combine scores ---
        if self._lstm_detector.is_trained and self._lstm_detector.window_ready(service):
            combined = _IF_WEIGHT * if_score + _LSTM_WEIGHT * lstm_score
        else:
            combined = if_score

        anomaly_score.labels(service=service, detector="combined").set(combined)

        # --- Anomaly decision (require consecutive windows above threshold) ---
        is_above = combined > _ANOMALY_THRESHOLD
        self._anomaly_windows[service].append(is_above)
        is_anomaly = all(self._anomaly_windows[service]) and len(self._anomaly_windows[service]) == _CONSECUTIVE_WINDOWS

        # --- Classify anomaly type ---
        anomaly_type = None
        confidence = combined
        if is_anomaly:
            anomaly_type = self._classify_anomaly(features, service)
            self._anomaly_count += 1
            anomaly_detected_total.labels(service=service, anomaly_type=anomaly_type or "unknown").inc()
            logger.warning(
                "detector.anomaly_detected",
                service=service,
                combined_score=round(combined, 4),
                anomaly_type=anomaly_type,
                if_score=round(if_score, 4),
                lstm_score=round(lstm_score, 4),
            )

        result = self._make_result(
            metrics, if_score, lstm_score, combined,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            confidence=confidence,
        )
        self._history.append(result)
        self._last_detection = result.timestamp

        # --- Periodic retraining ---
        if count % 500 == 0:
            self._if_detector.train()
            self._lstm_detector.train()

        return result

    # -- internals -----------------------------------------------------------

    def _classify_anomaly(self, features: np.ndarray, service: str) -> str:
        """Determine which feature contributes most to the anomaly."""
        if self._if_detector._count < 2:
            return "unknown"

        mean = self._if_detector._mean
        std = self._if_detector._running_std()
        z_scores = np.abs((features - mean) / std)

        most_deviant_idx = int(np.argmax(z_scores))
        return _FEATURE_ANOMALY_MAP.get(most_deviant_idx, "unknown")

    @staticmethod
    def _make_result(
        metrics: ServiceMetrics,
        if_score: float,
        lstm_score: float,
        combined: float,
        *,
        is_anomaly: bool = False,
        anomaly_type: str | None = None,
        confidence: float = 0.0,
    ) -> AnomalyResult:
        return AnomalyResult(
            service=metrics.service,
            timestamp=metrics.timestamp,
            isolation_forest_score=float(np.clip(if_score, 0.0, 1.0)),
            lstm_score=float(np.clip(lstm_score, 0.0, 1.0)),
            combined_score=float(np.clip(combined, 0.0, 1.0)),
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )
