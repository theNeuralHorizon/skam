"""One-Class SVM point-wise anomaly detector wrapper."""

from __future__ import annotations

import os

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OCSVMDetector:
    """Loads a pre-trained One-Class SVM and returns anomaly score."""

    def __init__(self, pretrained_path: str | None = None) -> None:
        path = pretrained_path or os.getenv("OCSVM_MODEL_PATH")
        self._model = None
        self._scaler = None
        self._is_trained = False

        if path and os.path.isfile(path):
            try:
                import joblib

                artefact = joblib.load(path)
                self._model = artefact["model"]
                self._scaler = artefact["scaler"]
                self._is_trained = True
                logger.info(
                    "ocsvm_detector.loaded",
                    path=path,
                    samples=artefact.get("training_samples", "?"),
                )
            except Exception:
                logger.warning("ocsvm_detector.load_failed", path=path, exc_info=True)
        else:
            logger.info("ocsvm_detector.no_model", path=path)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict(self, features: np.ndarray) -> float:
        """Return anomaly score in [0, 1] via sigmoid-mapped decision_function."""
        if not self._is_trained:
            return 0.0
        try:
            x = self._scaler.transform(features.reshape(1, -1))
            raw = self._model.decision_function(x)[0]
            # Negative raw = more anomalous; sigmoid maps to [0, 1]
            score = 1.0 / (1.0 + np.exp(2.0 * raw))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            logger.debug("ocsvm_detector.predict_failed", exc_info=True)
            return 0.0
