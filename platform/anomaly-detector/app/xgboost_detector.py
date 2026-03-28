"""XGBoost point-wise anomaly detector wrapper."""

from __future__ import annotations

import os

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class XGBoostDetector:
    """Loads a pre-trained XGBoost classifier and returns anomaly probability."""

    def __init__(self, pretrained_path: str | None = None) -> None:
        path = pretrained_path or os.getenv("XGB_MODEL_PATH")
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
                    "xgboost_detector.loaded",
                    path=path,
                    samples=artefact.get("training_samples", "?"),
                )
            except Exception:
                logger.warning("xgboost_detector.load_failed", path=path, exc_info=True)
        else:
            logger.info("xgboost_detector.no_model", path=path)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def predict(self, features: np.ndarray) -> float:
        """Return anomaly probability in [0, 1] for a single feature vector."""
        if not self._is_trained:
            return 0.0
        try:
            x = self._scaler.transform(features.reshape(1, -1))
            proba = float(self._model.predict_proba(x)[0][1])
            return float(np.clip(proba, 0.0, 1.0))
        except Exception:
            logger.debug("xgboost_detector.predict_failed", exc_info=True)
            return 0.0
