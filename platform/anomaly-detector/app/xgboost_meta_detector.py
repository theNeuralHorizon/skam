"""XGBoost Meta-Learner ensemble detector.

Combines XGBoost probability + PCA-compressed LSTM embedding + LSTM
reconstruction error + raw features through a trained XGBoost meta-learner.

Joint features (11-dim):
    [xgb_proba(1), pca_embedding(4), recon_error(1), raw_features(5)]
"""

from __future__ import annotations

import os
from collections import defaultdict, deque

import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)

_WINDOW_SIZE = 20


class XGBoostMetaDetector:
    """End-to-end XGBoost+LSTM ensemble via trained meta-learner."""

    def __init__(
        self,
        ensemble_path: str | None = None,
        lstm_model: object | None = None,
    ) -> None:
        path = ensemble_path or os.getenv("XGB_META_MODEL_PATH")
        self._meta_model = None
        self._pca = None
        self._xgb_model = None
        self._xgb_scaler = None
        self._lstm_model = lstm_model  # shared LSTM reference
        self._feat_min: np.ndarray | None = None
        self._feat_max: np.ndarray | None = None
        self._feat_range: np.ndarray | None = None
        self._optimal_threshold = 0.45
        self._is_trained = False

        # Per-service sliding windows (for LSTM encoding)
        self._windows: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=_WINDOW_SIZE)
        )

        if path and os.path.isfile(path):
            try:
                import joblib

                artefact = joblib.load(path)
                self._meta_model = artefact["meta_model"]
                self._pca = artefact["pca"]
                self._xgb_model = artefact["xgb_model"]
                self._xgb_scaler = artefact["xgb_scaler"]
                self._optimal_threshold = artefact.get("optimal_threshold", 0.45)

                feat_min = np.array(artefact["lstm_feat_min"], dtype=np.float32)
                feat_max = np.array(artefact["lstm_feat_max"], dtype=np.float32)
                self._feat_min = feat_min
                self._feat_max = feat_max
                self._feat_range = feat_max - feat_min
                self._feat_range[self._feat_range == 0] = 1.0

                self._is_trained = True
                logger.info(
                    "xgboost_meta_detector.loaded",
                    path=path,
                    auc=artefact.get("val_auc_roc", "?"),
                    f1=artefact.get("val_f1", "?"),
                    threshold=self._optimal_threshold,
                )
            except Exception:
                logger.warning("xgboost_meta_detector.load_failed", path=path, exc_info=True)
        else:
            logger.info("xgboost_meta_detector.no_model", path=path)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def push(self, service: str, features: np.ndarray) -> None:
        """Add a feature vector to the per-service sliding window."""
        self._windows[service].append(features.copy())

    def window_ready(self, service: str) -> bool:
        return len(self._windows[service]) >= _WINDOW_SIZE

    def set_lstm_model(self, lstm_model: object) -> None:
        """Set the shared LSTM model reference (called after detector init)."""
        self._lstm_model = lstm_model

    def predict_for_service(self, service: str) -> float:
        """Return meta-learner anomaly score in [0, 1]."""
        if not self._is_trained:
            return 0.0
        if not self.window_ready(service):
            return 0.0

        try:
            latest = self._windows[service][-1]
            window = np.array(list(self._windows[service]), dtype=np.float32)

            # 1. XGBoost probability
            xgb_scaled = self._xgb_scaler.transform(latest.reshape(1, -1))
            xgb_proba = float(self._xgb_model.predict_proba(xgb_scaled)[0][1])

            # 2. LSTM embedding + reconstruction error
            if self._lstm_model is not None and hasattr(self._lstm_model, "encode"):
                window_norm = (window - self._feat_min) / self._feat_range
                wt = torch.FloatTensor(window_norm).unsqueeze(0)
                with torch.no_grad():
                    lstm_emb = self._lstm_model.encode(wt).squeeze(0).numpy()
                    recon = self._lstm_model(wt)
                    recon_err = float(torch.mean((wt - recon) ** 2).item())
            else:
                lstm_emb = np.zeros(32, dtype=np.float32)
                recon_err = 0.0

            # 3. PCA compress embedding
            emb_pca = self._pca.transform(lstm_emb.reshape(1, -1))[0]

            # 4. Build compact feature vector
            compact = np.concatenate([
                [xgb_proba],
                emb_pca,
                [recon_err],
                latest,
            ]).reshape(1, -1)

            # 5. Meta-learner prediction
            score = float(self._meta_model.predict_proba(compact)[0][1])
            return float(np.clip(score, 0.0, 1.0))

        except Exception:
            logger.debug("xgboost_meta_detector.predict_failed", exc_info=True)
            return 0.0
