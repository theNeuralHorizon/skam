"""Extended ML evaluation metrics for anomaly detection benchmarking.

Beyond standard AUC-ROC and F1, these metrics capture calibration quality,
ranking stability, detection speed, and operational characteristics that
matter for production anomaly detection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    log_loss,
)


@dataclass
class ExtendedMetrics:
    """Comprehensive evaluation metrics for anomaly detection."""

    # Standard
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    f1_best: float = 0.0
    best_threshold: float = 0.0
    precision_at_07: float = 0.0
    recall_at_07: float = 0.0
    fpr_at_07: float = 0.0

    # Correlation and agreement
    mcc: float = 0.0  # Matthews Correlation Coefficient (-1 to 1, 0 = random)
    cohens_kappa: float = 0.0  # Inter-rater agreement adjusted for chance

    # Calibration (how well do scores reflect true probabilities)
    brier_score: float = 0.0  # Lower is better (0 = perfect calibration)
    log_loss_val: float = 0.0  # Cross-entropy loss

    # Score quality
    score_separation: float = 0.0  # Mean anomaly score - mean normal score
    score_overlap_pct: float = 0.0  # % of anomaly scores within normal score range
    normal_score_std: float = 0.0  # Stability of normal scores (lower = more stable)
    anomaly_score_std: float = 0.0

    # Operational
    detection_latency_samples: float = 0.0  # Avg samples until first detection after anomaly starts
    false_alarm_rate: float = 0.0  # % of normal samples incorrectly flagged
    miss_rate: float = 0.0  # % of anomalous samples missed

    # Robustness — AUC-ROC broken down by fault type
    auc_roc_per_fault: dict = field(default_factory=dict)

    # Throughput
    cold_start_samples: int = 0
    throughput_scores_per_sec: float = 0.0
    training_time_s: float = 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary with rounded floats."""
        d: dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, str, bool)):
                d[k] = round(v, 4) if isinstance(v, float) else v
            elif isinstance(v, dict):
                d[k] = {
                    kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()
                }
            elif v is None:
                d[k] = None
        return d


def compute_extended_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.7,
) -> ExtendedMetrics:
    """Compute all extended metrics from ground truth and anomaly scores.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth labels (0 = normal, 1 = anomaly).
    scores : np.ndarray
        Anomaly scores in [0, 1] from an ensemble.
    threshold : float
        Fixed decision threshold for precision/recall/FPR metrics.

    Returns
    -------
    ExtendedMetrics
        Populated dataclass with all computed metrics.
    """
    m = ExtendedMetrics()

    # ── Standard metrics ──────────────────────────────────────────────────
    try:
        m.auc_roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        m.auc_roc = 0.0

    try:
        m.auc_pr = float(average_precision_score(y_true, scores))
    except ValueError:
        m.auc_pr = 0.0

    # Best F1 across all thresholds
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_true, scores)
    f1s = 2 * prec_arr * rec_arr / np.maximum(prec_arr + rec_arr, 1e-8)
    best_idx = np.argmax(f1s)
    m.f1_best = float(f1s[best_idx])
    m.best_threshold = float(thresh_arr[best_idx]) if best_idx < len(thresh_arr) else 0.5

    # ── Metrics at fixed threshold ────────────────────────────────────────
    preds = (scores >= threshold).astype(int)
    m.precision_at_07 = float(precision_score(y_true, preds, zero_division=0))
    m.recall_at_07 = float(recall_score(y_true, preds, zero_division=0))
    fp = int(((preds == 1) & (y_true == 0)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    m.fpr_at_07 = float(fp / max(fp + tn, 1))

    # ── MCC — best single metric for imbalanced binary classification ────
    m.mcc = float(matthews_corrcoef(y_true, preds))

    # ── Cohen's Kappa — agreement corrected for chance ────────────────────
    m.cohens_kappa = float(cohen_kappa_score(y_true, preds))

    # ── Calibration metrics ───────────────────────────────────────────────
    scores_clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    m.brier_score = float(brier_score_loss(y_true, scores_clipped))
    try:
        m.log_loss_val = float(log_loss(y_true, scores_clipped))
    except ValueError:
        m.log_loss_val = float("inf")

    # ── Score quality ─────────────────────────────────────────────────────
    normal_mask = y_true == 0
    anomaly_mask = y_true == 1
    normal_scores = scores[normal_mask]
    anomaly_scores = scores[anomaly_mask]

    if len(normal_scores) > 0 and len(anomaly_scores) > 0:
        m.score_separation = float(anomaly_scores.mean() - normal_scores.mean())
        m.normal_score_std = float(normal_scores.std())
        m.anomaly_score_std = float(anomaly_scores.std())

        n_min, n_max = float(normal_scores.min()), float(normal_scores.max())
        overlap = ((anomaly_scores >= n_min) & (anomaly_scores <= n_max)).sum()
        m.score_overlap_pct = float(overlap / len(anomaly_scores) * 100)

    # ── Operational metrics ───────────────────────────────────────────────
    m.false_alarm_rate = float(fp / max(len(normal_scores), 1) * 100)
    m.miss_rate = float(fn / max(len(anomaly_scores), 1) * 100)

    return m
