"""Ensemble anomaly detection pipeline -- six-model weighted scoring.

All trained models contribute to the production anomaly decision via a
weighted ensemble.  When auxiliary models (XGBoost, Attention, OCSVM) are
not yet trained, the system gracefully falls back to IF+LSTM only.
"""

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
from app.metrics import anomaly_detected_total, anomaly_score, anomaly_score_velocity, prediction_generated_total
from app.models import (
    AnomalyResult,
    PerEnsembleScores,
    PredictionResult,
    ServiceMetrics,
    ServiceScore,
)
from app.predictor import OutagePredictor
from app.xgboost_detector import XGBoostDetector
from app.xgboost_meta_detector import XGBoostMetaDetector

logger = structlog.get_logger(__name__)

_COLD_START_SAMPLES = 200
_ANOMALY_THRESHOLD = 0.7
_CONSECUTIVE_WINDOWS = 2

# Ensemble weights -- used when all models are available.
# When a model is not trained/ready, its weight is redistributed
# proportionally among the active models.
_ENSEMBLE_WEIGHTS = {
    "isolation_forest": 0.20,
    "lstm":             0.25,
    "xgboost_lstm":     0.25,  # XGBoost+LSTM combined
    "xgboost_attention": 0.20,  # XGBoost+Attention combined
    "xgboost_meta":     0.10,
}
# Fallback weights when only IF+LSTM are available (cold start / no aux models)
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
        self._meta_detector = XGBoostMetaDetector()
        self._pretrained = self._if_detector.is_trained

        # Share the LSTM model with the meta-detector for embedding extraction
        if self._meta_detector.is_trained and self._lstm_detector.is_trained:
            self._meta_detector.set_lstm_model(self._lstm_detector._model)

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

        # Prediction engine
        self._predictor = OutagePredictor(threshold=_ANOMALY_THRESHOLD)
        self._latest_predictions: dict[str, list[PredictionResult]] = {}
        self._recovery_counts: dict[str, int] = {}

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

        # --- Stage 5: XGBoost Meta-Learner (end-to-end ensemble) ---
        self._meta_detector.push(service, features.astype(np.float32))
        meta_score = 0.0
        if self._meta_detector.is_trained and self._meta_detector.window_ready(service):
            meta_score = self._meta_detector.predict_for_service(service)

        # --- Composite ensemble scores ---
        xgb_lstm_score = 0.5 * xgb_score + 0.5 * lstm_score
        # attn_score already combines XGB + attention internally

        # --- Weighted ensemble (all available models) ---
        combined = self._compute_ensemble_score(
            if_score=if_score,
            lstm_score=lstm_score,
            lstm_ready=self._lstm_detector.is_trained and self._lstm_detector.window_ready(service),
            xgb_lstm_score=xgb_lstm_score,
            xgb_trained=self._xgb_detector.is_trained,
            attn_score=attn_score,
            attn_ready=self._attention_detector.is_trained and self._attention_detector.window_ready(service),
            meta_score=meta_score,
            meta_trained=self._meta_detector.is_trained and self._meta_detector.window_ready(service),
        )

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

        # --- Prediction layer ---
        score_velocity, predictions = self._predictor.update_and_predict(
            service=service,
            timestamp=metrics.timestamp,
            ensemble_score=combined,
            memory_usage=metrics.memory_usage,
            memory_limit=metrics.memory_limit if metrics.memory_limit > 0 else None,
            recent_recovery_count=self._recovery_counts.get(service, 0),
        )
        self._latest_predictions[service] = predictions
        anomaly_score_velocity.labels(service=service).set(score_velocity)
        for pred in predictions:
            prediction_generated_total.labels(
                service=service, prediction_type=pred.prediction_type
            ).inc()

        best_prediction = max(predictions, key=lambda p: p.confidence) if predictions else None

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
            score_velocity=round(score_velocity, 6),
            prediction=best_prediction.model_dump() if best_prediction else None,
            per_ensemble=PerEnsembleScores(
                isolation_forest=round(if_score, 4),
                lstm_autoencoder=round(lstm_score, 4),
                if_lstm_combined=round(combined, 4),
                xgboost_lstm=round(xgb_lstm_score, 4),
                xgboost_attention=round(attn_score, 4),
                xgboost_meta=round(meta_score, 4),
            ),
        )

        # --- Periodic retraining ---
        if count % 500 == 0:
            self._if_detector.train()
            self._lstm_detector.train()

        return result

    def get_latest_scores(self) -> list[ServiceScore]:
        """Return the most recent per-ensemble scores for all monitored services."""
        # Snapshot to avoid RuntimeError if dict changes during iteration
        return list(dict(self._latest_scores).values())

    def get_latest_predictions(self) -> list[PredictionResult]:
        """Return all current predictions across services, sorted by confidence."""
        # Snapshot to avoid RuntimeError if dict changes during iteration
        snapshot = dict(self._latest_predictions)
        all_preds: list[PredictionResult] = []
        for preds in snapshot.values():
            all_preds.extend(preds)
        all_preds.sort(key=lambda p: p.confidence, reverse=True)
        return all_preds

    def set_recovery_counts(self, counts: dict[str, int]) -> None:
        """Update per-service recovery counts (from decision engine)."""
        self._recovery_counts = dict(counts)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _compute_ensemble_score(
        *,
        if_score: float,
        lstm_score: float,
        lstm_ready: bool,
        xgb_lstm_score: float,
        xgb_trained: bool,
        attn_score: float,
        attn_ready: bool,
        meta_score: float,
        meta_trained: bool,
    ) -> float:
        """Compute weighted ensemble score from all available models.

        When auxiliary models are not ready, their weights are redistributed
        proportionally among the active models.  Falls back to IF+LSTM (or
        IF-only) when no auxiliary models are available.
        """
        # Build {model_key: score} for models that are active this cycle
        active: dict[str, float] = {"isolation_forest": if_score}

        if lstm_ready:
            active["lstm"] = lstm_score

        if xgb_trained and lstm_ready:
            active["xgboost_lstm"] = xgb_lstm_score

        if attn_ready:
            active["xgboost_attention"] = attn_score

        if meta_trained:
            active["xgboost_meta"] = meta_score

        # If only IF is available, return it directly
        if len(active) == 1:
            return if_score

        # If only IF+LSTM available, use legacy weights for backward compat
        if set(active.keys()) == {"isolation_forest", "lstm"}:
            return _IF_WEIGHT * if_score + _LSTM_WEIGHT * lstm_score

        # Redistribute weights of inactive models proportionally
        total_active_weight = sum(_ENSEMBLE_WEIGHTS[k] for k in active)
        combined = sum(
            (_ENSEMBLE_WEIGHTS[k] / total_active_weight) * score
            for k, score in active.items()
        )
        return float(np.clip(combined, 0.0, 1.0))

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
