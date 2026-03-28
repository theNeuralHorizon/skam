#!/usr/bin/env python3
"""Benchmarking framework for comparing anomaly detection ensembles.

Evaluates detection quality on the RCAEval dataset using standard
metrics: precision, recall, F1, AUC-ROC, AUC-PR, detection latency,
and false positive rate.

Usage::

    python -m ml.training.benchmark
    python -m ml.training.benchmark --fault-types cpu mem delay
    python -m ml.training.benchmark --output-dir ml/benchmark_results

Each "ensemble" is a scoring function that takes a feature vector
(or sequence) and returns an anomaly score in [0, 1].  New ensembles
are registered via the ``@register_ensemble`` decorator.

Metrics computed per ensemble
-----------------------------

+-----------------+------------------------------------------------------+
| Metric          | What it measures                                     |
+=================+======================================================+
| AUC-ROC         | Overall discrimination (threshold-agnostic)          |
+-----------------+------------------------------------------------------+
| AUC-PR          | Precision-recall trade-off (better for imbalanced)   |
+-----------------+------------------------------------------------------+
| F1 @ best       | F1 at the optimal threshold                          |
+-----------------+------------------------------------------------------+
| Precision @ 0.7 | Precision when threshold = 0.7 (our production val)  |
+-----------------+------------------------------------------------------+
| Recall @ 0.7    | Recall when threshold = 0.7                          |
+-----------------+------------------------------------------------------+
| FPR @ 0.7       | False positive rate at threshold 0.7                 |
+-----------------+------------------------------------------------------+
| Detection lag   | Seconds between injection and first detection        |
+-----------------+------------------------------------------------------+
| Cold start      | Samples needed before first valid score              |
+-----------------+------------------------------------------------------+
| Throughput      | Scores per second                                    |
+-----------------+------------------------------------------------------+
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.training.data_loader import (
    FEATURE_ORDER,
    build_training_arrays,
    iter_scenarios,
    extract_service_features,
    downsample_to_interval,
)
from ml.training.metrics import ExtendedMetrics, compute_extended_metrics


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════


class Ensemble(Protocol):
    """Protocol for ensemble scoring functions."""

    name: str
    cold_start_samples: int

    def fit(self, X_normal: np.ndarray) -> None:
        """Train on normal-only data."""
        ...

    def score(self, features: np.ndarray) -> float:
        """Return anomaly score in [0, 1] for a single sample."""
        ...


_REGISTRY: dict[str, type] = {}


def register_ensemble(cls: type) -> type:
    _REGISTRY[cls.name] = cls
    return cls


# ═══════════════════════════════════════════════════════════════════════
# Ensemble implementations
# ═══════════════════════════════════════════════════════════════════════


@register_ensemble
class IsolationForestEnsemble:
    name = "isolation_forest"
    cold_start_samples = 50

    def __init__(self) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()
        self._model = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1,
        )
        self._train_scores: np.ndarray | None = None

    def fit(self, X_normal: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_normal)
        self._model.fit(X_scaled)
        # Store training scores for percentile-rank mapping
        self._train_scores = self._model.decision_function(X_scaled)

    def score(self, features: np.ndarray) -> float:
        x = self._scaler.transform(features.reshape(1, -1))
        raw = self._model.decision_function(x)[0]
        # Percentile rank against training distribution:
        # lower raw score = more anomalous, so invert the percentile
        if self._train_scores is not None:
            pct = (self._train_scores < raw).mean()
            return float(np.clip(1.0 - pct, 0, 1))
        return float(np.clip(1.0 / (1.0 + np.exp(5.0 * raw)), 0, 1))


@register_ensemble
class ZScoreEnsemble:
    """Statistical baseline — no training needed, immediate detection."""

    name = "zscore"
    cold_start_samples = 0

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X_normal: np.ndarray) -> None:
        self._mean = X_normal.mean(axis=0)
        self._std = X_normal.std(axis=0)
        self._std[self._std < 1e-8] = 1.0

    def score(self, features: np.ndarray) -> float:
        if self._mean is None:
            return 0.0
        z = np.abs((features - self._mean) / self._std)
        max_z = float(z.max())
        # Map z-score to [0,1]: z=3 → 0.5, z=6 → ~0.95
        return float(np.clip(1.0 / (1.0 + np.exp(-1.0 * (max_z - 3.0))), 0, 1))


@register_ensemble
class IQREnsemble:
    """Interquartile range — robust to outliers, zero-training baseline."""

    name = "iqr"
    cold_start_samples = 0

    def __init__(self) -> None:
        self._q1: np.ndarray | None = None
        self._q3: np.ndarray | None = None
        self._iqr: np.ndarray | None = None

    def fit(self, X_normal: np.ndarray) -> None:
        self._q1 = np.percentile(X_normal, 25, axis=0)
        self._q3 = np.percentile(X_normal, 75, axis=0)
        self._iqr = self._q3 - self._q1
        self._iqr[self._iqr < 1e-8] = 1.0

    def score(self, features: np.ndarray) -> float:
        if self._q1 is None:
            return 0.0
        lower = self._q1 - 1.5 * self._iqr
        upper = self._q3 + 1.5 * self._iqr
        # Count how many features are outside bounds
        below = np.maximum(0, lower - features) / self._iqr
        above = np.maximum(0, features - upper) / self._iqr
        deviation = np.maximum(below, above)
        max_dev = float(deviation.max())
        return float(np.clip(max_dev / 3.0, 0, 1))  # normalize: 3 IQR units → 1.0


@register_ensemble
class EWMAEnsemble:
    """Exponentially Weighted Moving Average — online, minimal cold start."""

    name = "ewma"
    cold_start_samples = 10

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha = alpha
        self._ewma: np.ndarray | None = None
        self._ewma_var: np.ndarray | None = None

    def fit(self, X_normal: np.ndarray) -> None:
        self._ewma = X_normal.mean(axis=0)
        self._ewma_var = X_normal.var(axis=0)
        self._ewma_var[self._ewma_var < 1e-8] = 1.0

    def score(self, features: np.ndarray) -> float:
        if self._ewma is None:
            return 0.0
        deviation = np.abs(features - self._ewma) / np.sqrt(self._ewma_var)
        # Update EWMA
        self._ewma = self._alpha * features + (1 - self._alpha) * self._ewma
        diff = features - self._ewma
        self._ewma_var = self._alpha * diff ** 2 + (1 - self._alpha) * self._ewma_var
        self._ewma_var[self._ewma_var < 1e-8] = 1.0

        max_dev = float(deviation.max())
        return float(np.clip(1.0 / (1.0 + np.exp(-1.0 * (max_dev - 3.0))), 0, 1))


@register_ensemble
class OneClassSVMEnsemble:
    """One-Class SVM — better for small datasets than IF."""

    name = "ocsvm"
    cold_start_samples = 30

    def __init__(self) -> None:
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()
        self._model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)

    def fit(self, X_normal: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_normal)
        # Subsample if too large (OCSVM is O(n^2))
        if len(X_scaled) > 2000:
            idx = np.random.default_rng(42).choice(len(X_scaled), 2000, replace=False)
            X_scaled = X_scaled[idx]
        self._model.fit(X_scaled)

    def score(self, features: np.ndarray) -> float:
        x = self._scaler.transform(features.reshape(1, -1))
        raw = self._model.decision_function(x)[0]
        return float(np.clip(1.0 / (1.0 + np.exp(2.0 * raw)), 0, 1))


@register_ensemble
class CombinedIFLSTMEnsemble:
    """Our production ensemble: IF (0.4) + LSTM reconstruction error (0.6).

    For the benchmark, LSTM contributes via per-point reconstruction error
    against a sliding window, approximated by training-set error statistics.
    """

    name = "if_lstm_combined"
    cold_start_samples = 200

    def __init__(self) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()
        self._if_model = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1,
        )
        self._train_scores: np.ndarray | None = None
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

    def fit(self, X_normal: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_normal)
        self._if_model.fit(X_scaled)
        self._train_scores = self._if_model.decision_function(X_scaled)
        self._feat_mean = X_normal.mean(axis=0)
        self._feat_std = X_normal.std(axis=0)
        self._feat_std[self._feat_std < 1e-8] = 1.0

    def score(self, features: np.ndarray) -> float:
        # IF score via percentile rank
        x = self._scaler.transform(features.reshape(1, -1))
        raw = self._if_model.decision_function(x)[0]
        if_score = 0.0
        if self._train_scores is not None:
            pct = (self._train_scores < raw).mean()
            if_score = float(np.clip(1.0 - pct, 0, 1))

        # Simple Mahalanobis-like proxy for temporal component
        z = np.abs((features - self._feat_mean) / self._feat_std)
        mahal_score = float(np.clip(z.mean() / 5.0, 0, 1))

        # Weighted combination (0.4 IF + 0.6 temporal proxy)
        combined = 0.4 * if_score + 0.6 * mahal_score
        return float(np.clip(combined, 0, 1))


@register_ensemble
class XGBoostLSTMEnsemble:
    """XGBoost for feature-based anomalies + LSTM for temporal patterns.

    XGBoost is trained as a cross-prediction regressor on normal data:
    each feature is predicted from the remaining four.  At inference the
    normalised mean absolute prediction error serves as an anomaly signal,
    combined 50/50 with a Mahalanobis-like LSTM proxy (same approach as
    CombinedIFLSTMEnsemble).
    """

    name = "xgboost_lstm"
    cold_start_samples = 100

    def __init__(self) -> None:
        import xgboost as xgb
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()
        self._xgb_models: list = []  # one regressor per feature
        self._n_features: int = 0
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._train_errors: np.ndarray | None = None
        self._xgb = xgb

    def fit(self, X_normal: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_normal)
        self._n_features = X_scaled.shape[1]
        self._xgb_models = []

        # Cross-prediction: predict feature i from all other features
        for i in range(self._n_features):
            mask = [j for j in range(self._n_features) if j != i]
            model = self._xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_scaled[:, mask], X_scaled[:, i])
            self._xgb_models.append(model)

        # Store training error distribution for normalisation
        errors = self._cross_prediction_error(X_scaled)
        self._train_errors = errors
        self._feat_mean = X_normal.mean(axis=0)
        self._feat_std = X_normal.std(axis=0)
        self._feat_std[self._feat_std < 1e-8] = 1.0

    def _cross_prediction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Mean absolute cross-prediction error per sample."""
        total_err = np.zeros(len(X_scaled))
        for i, model in enumerate(self._xgb_models):
            mask = [j for j in range(self._n_features) if j != i]
            pred = model.predict(X_scaled[:, mask])
            total_err += np.abs(X_scaled[:, i] - pred)
        return total_err / self._n_features

    def score(self, features: np.ndarray) -> float:
        x = self._scaler.transform(features.reshape(1, -1))
        error = float(self._cross_prediction_error(x)[0])

        # Percentile-rank against training errors
        if self._train_errors is not None:
            pct = (self._train_errors < error).mean()
            xgb_score = float(np.clip(pct, 0, 1))
        else:
            xgb_score = float(np.clip(error / 5.0, 0, 1))

        # LSTM temporal proxy (Mahalanobis-like, same as CombinedIFLSTMEnsemble)
        z = np.abs((features - self._feat_mean) / self._feat_std)
        lstm_proxy = float(np.clip(z.mean() / 5.0, 0, 1))

        return float(np.clip(0.5 * xgb_score + 0.5 * lstm_proxy, 0, 1))


@register_ensemble
class XGBoostAttentionEnsemble:
    """XGBoost with learned feature attention weights.

    Instead of DistilBERT (which requires text input), we use a
    single-head self-attention mechanism on the 5-feature vectors to
    learn which features matter most for anomaly detection.  The
    attention weights are derived from the training data covariance
    structure, then applied to weight the XGBoost cross-prediction
    errors per feature.
    """

    name = "xgboost_attention"
    cold_start_samples = 50

    def __init__(self) -> None:
        import xgboost as xgb
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()
        self._xgb_models: list = []
        self._n_features: int = 0
        self._attention_weights: np.ndarray | None = None
        self._train_errors: np.ndarray | None = None
        self._xgb = xgb

    def fit(self, X_normal: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_normal)
        self._n_features = X_scaled.shape[1]
        self._xgb_models = []

        # Cross-prediction regressors (same as XGBoostLSTMEnsemble)
        for i in range(self._n_features):
            mask = [j for j in range(self._n_features) if j != i]
            model = self._xgb.XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_scaled[:, mask], X_scaled[:, i])
            self._xgb_models.append(model)

        # Learn attention weights via single-head self-attention on
        # the covariance matrix of the training features.
        # Q = K = V = X_scaled — attention = softmax(Q K^T / sqrt(d))
        # We reduce to per-feature importance by averaging attention
        # rows and then taking the column-wise mean.
        d = self._n_features
        cov = np.cov(X_scaled, rowvar=False)  # (d, d)
        # Scaled dot-product attention analogue
        attn_logits = cov / np.sqrt(d)
        # Softmax per row
        exp_logits = np.exp(attn_logits - attn_logits.max(axis=1, keepdims=True))
        attn = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        # Per-feature importance = column mean of attention matrix
        self._attention_weights = attn.mean(axis=0)
        # Normalise so weights sum to 1
        self._attention_weights /= self._attention_weights.sum()

        # Store training errors for percentile mapping
        self._train_errors = self._weighted_error(X_scaled)

    def _weighted_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Attention-weighted mean absolute cross-prediction error."""
        per_feat = np.zeros((len(X_scaled), self._n_features))
        for i, model in enumerate(self._xgb_models):
            mask = [j for j in range(self._n_features) if j != i]
            pred = model.predict(X_scaled[:, mask])
            per_feat[:, i] = np.abs(X_scaled[:, i] - pred)

        # Weight by attention
        return (per_feat * self._attention_weights).sum(axis=1)

    def score(self, features: np.ndarray) -> float:
        x = self._scaler.transform(features.reshape(1, -1))
        error = float(self._weighted_error(x)[0])

        # Percentile-rank against training errors
        if self._train_errors is not None:
            pct = (self._train_errors < error).mean()
            return float(np.clip(pct, 0, 1))
        return float(np.clip(error / 5.0, 0, 1))


# ═══════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkResult:
    """Full benchmark result for one ensemble, including extended metrics."""

    ensemble_name: str
    metrics: ExtendedMetrics = field(default_factory=ExtendedMetrics)

    # Convenience accessors for the most common metrics
    @property
    def auc_roc(self) -> float:
        return self.metrics.auc_roc

    @property
    def auc_pr(self) -> float:
        return self.metrics.auc_pr

    @property
    def f1_best(self) -> float:
        return self.metrics.f1_best

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dict."""
        d = self.metrics.to_dict()
        d["ensemble_name"] = self.ensemble_name
        return d


def run_benchmark(
    X_normal: np.ndarray,
    X_anomalous: np.ndarray,
    ensemble_names: list[str] | None = None,
    per_fault_data: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[BenchmarkResult]:
    """Run all registered ensembles on the provided data.

    Training: 70% of normal data (only normal -- unsupervised)
    Testing: balanced 50/50 split of held-out normal and anomalous data

    Using a balanced test set avoids inflating AUC-PR from prevalence
    bias (e.g. 77% anomalous would give a random baseline of 0.77).

    Parameters
    ----------
    X_normal, X_anomalous : np.ndarray
        Feature matrices for normal and anomalous data.
    ensemble_names : list[str] | None
        Subset of ensembles to evaluate (default: all registered).
    per_fault_data : dict[str, tuple[np.ndarray, np.ndarray]] | None
        Mapping from fault type name to (X_normal, X_anomalous) arrays.
        Used to compute per-fault AUC-ROC breakdown.
    """
    results: list[BenchmarkResult] = []
    targets = ensemble_names or list(_REGISTRY.keys())

    rng = np.random.default_rng(42)

    # Split normal: 70% train, 30% test
    idx_n = rng.permutation(len(X_normal))
    split_n = int(0.7 * len(X_normal))
    X_train = X_normal[idx_n[:split_n]]
    X_test_normal = X_normal[idx_n[split_n:]]

    # Split anomalous: 70% held back, 30% test
    idx_a = rng.permutation(len(X_anomalous))
    split_a = int(0.7 * len(X_anomalous))
    X_test_anomalous = X_anomalous[idx_a[split_a:]]

    # Balance test set: equal normal and anomalous (50/50 prevalence)
    test_size = min(len(X_test_normal), len(X_test_anomalous))
    X_test_normal = X_test_normal[:test_size]
    X_test_anomalous = X_test_anomalous[:test_size]

    X_test = np.vstack([X_test_normal, X_test_anomalous])
    y_true = np.concatenate([
        np.zeros(test_size, dtype=np.int32),
        np.ones(test_size, dtype=np.int32),
    ])

    print(f"\n{'='*70}")
    print(f"  Benchmark: {len(X_train):,} train | {test_size:,} test-normal | {test_size:,} test-anomaly (50/50)")
    print(f"  Random baseline: AUC-ROC=0.500  AUC-PR=0.500")
    print(f"{'='*70}")

    for name in targets:
        if name not in _REGISTRY:
            print(f"  [SKIP] {name} not registered")
            continue

        cls = _REGISTRY[name]
        ensemble = cls()
        print(f"\n  Evaluating: {name} (cold_start={ensemble.cold_start_samples})")

        # Train
        t0 = time.time()
        ensemble.fit(X_train)
        train_time = time.time() - t0

        # Score all test points
        t0 = time.time()
        scores = np.array([ensemble.score(x) for x in X_test])
        score_time = time.time() - t0
        throughput = len(X_test) / max(score_time, 1e-6)

        # Compute extended metrics
        m = compute_extended_metrics(y_true, scores, threshold=0.7)
        m.cold_start_samples = ensemble.cold_start_samples
        m.throughput_scores_per_sec = round(throughput, 0)
        m.training_time_s = round(train_time, 3)

        # Per-fault-type AUC-ROC breakdown
        if per_fault_data is not None:
            fault_aucs: dict[str, float] = {}
            for ft, (xn_ft, xa_ft) in per_fault_data.items():
                # Score a small balanced sample per fault type
                ft_rng = np.random.default_rng(42)
                ft_size = min(200, len(xn_ft), len(xa_ft))
                if ft_size < 10:
                    continue
                xn_sub = xn_ft[ft_rng.choice(len(xn_ft), ft_size, replace=False)]
                xa_sub = xa_ft[ft_rng.choice(len(xa_ft), ft_size, replace=False)]
                ft_X = np.vstack([xn_sub, xa_sub])
                ft_y = np.concatenate([np.zeros(ft_size), np.ones(ft_size)])
                ft_scores = np.array([ensemble.score(x) for x in ft_X])
                try:
                    fault_aucs[ft] = round(float(roc_auc_score(ft_y, ft_scores)), 4)
                except ValueError:
                    fault_aucs[ft] = 0.0
            m.auc_roc_per_fault = fault_aucs

        result = BenchmarkResult(ensemble_name=name, metrics=m)
        results.append(result)

        print(f"    AUC-ROC={m.auc_roc:.4f}  AUC-PR={m.auc_pr:.4f}  "
              f"F1={m.f1_best:.4f}@{m.best_threshold:.3f}  "
              f"P@0.7={m.precision_at_07:.3f}  R@0.7={m.recall_at_07:.3f}  "
              f"FPR@0.7={m.fpr_at_07:.3f}  "
              f"MCC={m.mcc:.3f}  Brier={m.brier_score:.4f}  "
              f"throughput={m.throughput_scores_per_sec:.0f}/s")
        if m.auc_roc_per_fault:
            parts = "  ".join(f"{ft}={v:.3f}" for ft, v in m.auc_roc_per_fault.items())
            print(f"    Per-fault AUC-ROC: {parts}")

    return results


def print_leaderboard(results: list[BenchmarkResult]) -> None:
    """Print a sorted comparison table with extended metrics."""
    ranked = sorted(results, key=lambda r: r.auc_roc, reverse=True)

    print(f"\n{'='*130}")
    print(f"  LEADERBOARD (ranked by AUC-ROC)")
    print(f"{'='*130}")
    header = (
        f"  {'Rank':<5} {'Ensemble':<22} {'AUC-ROC':>8} {'AUC-PR':>8} "
        f"{'F1@best':>8} {'MCC':>6} {'Kappa':>6} {'Brier':>7} "
        f"{'P@0.7':>7} {'R@0.7':>7} {'FPR@0.7':>8} "
        f"{'Cold':>6} {'Speed':>10} {'Train':>7}"
    )
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for i, r in enumerate(ranked, 1):
        m = r.metrics
        print(
            f"  {i:<5} {r.ensemble_name:<22} {m.auc_roc:>8.4f} {m.auc_pr:>8.4f} "
            f"{m.f1_best:>8.4f} {m.mcc:>6.3f} {m.cohens_kappa:>6.3f} {m.brier_score:>7.4f} "
            f"{m.precision_at_07:>7.3f} {m.recall_at_07:>7.3f} "
            f"{m.fpr_at_07:>8.3f} {m.cold_start_samples:>6} "
            f"{m.throughput_scores_per_sec:>8.0f}/s {m.training_time_s:>6.2f}s"
        )

    # Score quality sub-table
    print(f"\n  {'--- Score Quality ---':^60}")
    print(f"  {'Ensemble':<22} {'Separation':>11} {'Overlap%':>9} {'NormStd':>8} {'AnoStd':>8} {'FAR%':>7} {'Miss%':>7}")
    for r in ranked:
        m = r.metrics
        print(
            f"  {r.ensemble_name:<22} {m.score_separation:>11.4f} {m.score_overlap_pct:>8.1f}% "
            f"{m.normal_score_std:>8.4f} {m.anomaly_score_std:>8.4f} "
            f"{m.false_alarm_rate:>6.1f}% {m.miss_rate:>6.1f}%"
        )

    print(f"{'='*130}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark anomaly detection ensembles")
    parser.add_argument(
        "--data-dir", type=str,
        default=str(REPO_ROOT / "ml" / "data" / "rcaeval" / "online-boutique"),
    )
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "ml" / "benchmark_results"))
    parser.add_argument("--fault-types", nargs="+", default=None)
    parser.add_argument("--ensembles", nargs="+", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: Dataset not found at {data_dir}")
        print("Run: python ml/training/train_models.py to download the dataset first")
        sys.exit(1)

    print("Loading RCAEval data (balanced across fault types)...")

    # Load per-fault-type and balance to avoid bias from trivially-separable
    # faults (cpu/mem) dominating the evaluation.
    available_faults = args.fault_types or ["cpu", "mem", "delay", "disk", "loss"]
    per_fault_normal: list[np.ndarray] = []
    per_fault_anomalous: list[np.ndarray] = []
    loaded_faults: list[str] = []
    rng = np.random.default_rng(42)

    min_samples = float("inf")
    for ft in available_faults:
        try:
            xn, xa, _ = build_training_arrays(data_dir, fault_types=[ft], interval_seconds=15)
            min_samples = min(min_samples, len(xn), len(xa))
        except ValueError:
            continue

    min_samples = int(min_samples)
    print(f"  Balancing to {min_samples} samples per fault type")

    for ft in available_faults:
        try:
            xn, xa, _ = build_training_arrays(data_dir, fault_types=[ft], interval_seconds=15)
        except ValueError:
            print(f"  [SKIP] No data for fault type: {ft}")
            continue
        idx_n = rng.choice(len(xn), min(min_samples, len(xn)), replace=False)
        idx_a = rng.choice(len(xa), min(min_samples, len(xa)), replace=False)
        per_fault_normal.append(xn[idx_n])
        per_fault_anomalous.append(xa[idx_a])
        loaded_faults.append(ft)
        print(f"  {ft}: {len(idx_n)} normal + {len(idx_a)} anomalous")

    X_normal = np.vstack(per_fault_normal)
    X_anomalous = np.vstack(per_fault_anomalous)
    print(f"  Total: {len(X_normal):,} normal, {len(X_anomalous):,} anomalous")

    # Build per-fault data dict for per-fault AUC-ROC breakdown
    per_fault_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ft, xn, xa in zip(loaded_faults, per_fault_normal, per_fault_anomalous):
        per_fault_dict[ft] = (xn, xa)

    results = run_benchmark(
        X_normal, X_anomalous,
        ensemble_names=args.ensembles,
        per_fault_data=per_fault_dict,
    )
    print_leaderboard(results)

    # Save extended results
    output_path = output_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
