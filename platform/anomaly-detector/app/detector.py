"""Ensemble anomaly detection pipeline -- six-model scoring."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import structlog

from app.attention_detector import AttentionDetector
from app.isolation_forest import IsolationForestDetector
from app.lstm_detector import LSTMDetector
from app.metrics import anomaly_detected_total, anomaly_score
from app.models import (
    AnomalyResult,
    PerEnsembleScores,
    ServiceMetrics,
    ServiceScore,
)
from app.ocsvm_detector import OCSVMDetector
from app.xgboost_detector import XGBoostDetector

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

    When pre-trained model paths are provided, the detector starts
    immediately with zero cold-start delay.  Otherwise it falls back
    to the original behaviour of collecting ``_COLD_START_SAMPLES``
    per service before scoring.
    """

    def __init__(
        self,
        if_model_path: str | None = None,
        lstm_model_path: str | None = None,
    ) -> None:
        # Resolve paths: explicit args > env vars > None
        if_path = if_model_path or os.getenv("IF_MODEL_PATH")
        lstm_path = lstm_model_path or os.getenv("LSTM_MODEL_PATH")

        self._if_detector = IsolationForestDetector(pretrained_path=if_path)
        self._lstm_detector = LSTMDetector(pretrained_path=lstm_path)
        self._xgb_detector = XGBoostDetector()
        self._attention_detector = AttentionDetector()
        self._ocsvm_detector = OCSVMDetector()
        self._pretrained = self._if_detector.is_trained

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
        self._model_version: str = "2.0.0"

        # Latest per-service scores for the /api/scores endpoint
        self._latest_scores: dict[str, ServiceScore] = {}

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

        # Always feed data into all detectors
        self._if_detector.add_sample(features)
        self._lstm_detector.push(service, features.astype(np.float32))
        self._attention_detector.push(service, features.astype(np.float32))

        # --- cold start: train only, no scoring ---
        if not self._pretrained and count <= _COLD_START_SAMPLES:
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

        # --- Stage 3: XGBoost (point-wise) ---
        xgb_score = self._xgb_detector.predict(features) if self._xgb_detector.is_trained else 0.0

        # --- Stage 4: Attention (temporal) ---
        attn_score = 0.0
        if self._attention_detector.is_trained and self._attention_detector.window_ready(service):
            attn_score = self._attention_detector.predict_for_service(service)

        # --- Stage 5: One-Class SVM (point-wise) ---
        ocsvm_score = self._ocsvm_detector.predict(features) if self._ocsvm_detector.is_trained else 0.0

        # --- Combine IF+LSTM (production decision) ---
        if self._lstm_detector.is_trained and self._lstm_detector.window_ready(service):
            combined = _IF_WEIGHT * if_score + _LSTM_WEIGHT * lstm_score
        else:
            combined = if_score

        anomaly_score.labels(service=service, detector="combined").set(combined)

        # --- Composite ensemble scores ---
        xgb_lstm_score = 0.5 * xgb_score + 0.5 * lstm_score
        # attn_score already combines XGB + attention internally

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

        # --- Store latest per-ensemble scores for the API ---
        severity_label, severity_level = self._classify_severity(combined)
        self._latest_scores[service] = ServiceScore(
            service=service,
            isoforest_score=round(if_score, 4),
            lstm_score=round(lstm_score, 4),
            ensemble_score=round(combined, 4),
            is_anomaly=is_anomaly,
            features={
                "request_rate": metrics.request_rate,
                "error_ratio": metrics.error_rate,
                "latency_p50": metrics.p99_latency * 0.6,  # approximate
                "latency_p99": metrics.p99_latency,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage_mb": metrics.memory_usage / (1024 * 1024),
                "restart_count": 0,
            },
            severity_label=severity_label,
            severity_level=severity_level,
            consecutive_windows=sum(self._anomaly_windows[service]),
            per_ensemble=PerEnsembleScores(
                isolation_forest=round(if_score, 4),
                lstm_autoencoder=round(lstm_score, 4),
                if_lstm_combined=round(combined, 4),
                xgboost_lstm=round(xgb_lstm_score, 4),
                xgboost_attention=round(attn_score, 4),
                ocsvm=round(ocsvm_score, 4),
            ),
        )

        # --- Periodic retraining ---
        if count % 500 == 0:
            self._if_detector.train()
            self._lstm_detector.train()

        return result

    def get_latest_scores(self) -> list[ServiceScore]:
        """Return the most recent per-ensemble scores for all monitored services."""
        return list(self._latest_scores.values())

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _classify_severity(score: float) -> tuple[str, int]:
        """Map combined score to severity label and level."""
        if score >= 0.8:
            return "critical", 4
        if score >= 0.65:
            return "high", 3
        if score >= 0.5:
            return "medium", 2
        if score >= 0.3:
            return "low", 1
        return "normal", 0

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
