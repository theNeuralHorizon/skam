"""XGBoost + Self-Attention temporal anomaly detector wrapper."""

from __future__ import annotations

import os
from collections import defaultdict, deque

import numpy as np
import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)

_WINDOW_SIZE = 20


# ── SelfAttentionScorer (duplicated from ml/training/attention_model.py
#    because the Docker image does not ship the ml/ package) ──────────

class SelfAttentionScorer(nn.Module):
    """Temporal anomaly scorer using multi-head self-attention."""

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        attn_out, _ = self.attention(h, h, h)
        h = self.layer_norm(h + attn_out)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


class AttentionDetector:
    """Combines XGBoost (point-wise) + Self-Attention (temporal) scoring."""

    def __init__(
        self,
        attention_path: str | None = None,
        xgb_path: str | None = None,
    ) -> None:
        attn_path = attention_path or os.getenv("ATTENTION_MODEL_PATH")
        xgb_path = xgb_path or os.getenv("XGB_ATTENTION_MODEL_PATH")

        self._attn_model: SelfAttentionScorer | None = None
        self._xgb_model = None
        self._xgb_scaler = None
        self._feat_min: np.ndarray | None = None
        self._feat_max: np.ndarray | None = None
        self._is_trained = False
        self._xgb_weight = 0.5
        self._attn_weight = 0.5

        # Per-service sliding windows
        self._windows: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=_WINDOW_SIZE)
        )

        # Load attention model
        if attn_path and os.path.isfile(attn_path):
            try:
                checkpoint = torch.load(attn_path, map_location="cpu", weights_only=False)
                self._attn_model = SelfAttentionScorer(
                    input_dim=checkpoint.get("input_dim", 5),
                    d_model=checkpoint.get("d_model", 32),
                    num_heads=checkpoint.get("num_heads", 4),
                )
                self._attn_model.load_state_dict(checkpoint["model_state_dict"])
                self._attn_model.eval()
                self._feat_min = np.array(checkpoint["feat_min"], dtype=np.float32)
                self._feat_max = np.array(checkpoint["feat_max"], dtype=np.float32)
                logger.info("attention_detector.attention_loaded", path=attn_path)
            except Exception:
                logger.warning("attention_detector.attention_load_failed", path=attn_path, exc_info=True)

        # Load XGBoost component
        if xgb_path and os.path.isfile(xgb_path):
            try:
                import joblib

                artefact = joblib.load(xgb_path)
                self._xgb_model = artefact["model"]
                self._xgb_scaler = artefact["scaler"]
                self._xgb_weight = artefact.get("xgb_weight", 0.5)
                self._attn_weight = artefact.get("attention_weight", 0.5)
                logger.info("attention_detector.xgb_loaded", path=xgb_path)
            except Exception:
                logger.warning("attention_detector.xgb_load_failed", path=xgb_path, exc_info=True)

        self._is_trained = self._attn_model is not None or self._xgb_model is not None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def push(self, service: str, features: np.ndarray) -> None:
        """Add a feature vector to the per-service sliding window."""
        self._windows[service].append(features.copy())

    def window_ready(self, service: str) -> bool:
        return len(self._windows[service]) >= _WINDOW_SIZE

    def predict_for_service(self, service: str) -> float:
        """Return combined XGBoost + Attention anomaly score in [0, 1]."""
        if not self._is_trained:
            return 0.0

        scores: list[float] = []
        weights: list[float] = []

        # XGBoost component (point-wise, latest sample)
        if self._xgb_model is not None and self._xgb_scaler is not None:
            try:
                latest = self._windows[service][-1]
                x = self._xgb_scaler.transform(latest.reshape(1, -1))
                xgb_score = float(self._xgb_model.predict_proba(x)[0][1])
                scores.append(xgb_score)
                weights.append(self._xgb_weight)
            except Exception:
                logger.debug("attention_detector.xgb_predict_failed", exc_info=True)

        # Attention component (temporal, needs full window)
        if self._attn_model is not None and self.window_ready(service):
            try:
                window = np.array(list(self._windows[service]), dtype=np.float32)
                # Min-max normalise
                feat_range = self._feat_max - self._feat_min
                feat_range[feat_range == 0] = 1.0
                window_norm = (window - self._feat_min) / feat_range
                tensor = torch.FloatTensor(window_norm).unsqueeze(0)  # (1, T, 5)
                with torch.no_grad():
                    attn_score = float(self._attn_model(tensor).item())
                scores.append(attn_score)
                weights.append(self._attn_weight)
            except Exception:
                logger.debug("attention_detector.attn_predict_failed", exc_info=True)

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        combined = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return float(np.clip(combined, 0.0, 1.0))
