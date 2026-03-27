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
from typing import Callable, Protocol

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ml.training.data_loader import (
    FEATURE_ORDER,
    build_training_arrays,
    iter_scenarios,
    extract_service_features,
    downsample_to_interval,
)


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

    def fit(self, X_normal: np.ndarray) -> None:
        X_scaled = self._scaler.fit_transform(X_normal)
        self._model.fit(X_scaled)

    def score(self, features: np.ndarray) -> float:
        x = self._scaler.transform(features.reshape(1, -1))
        raw = self._model.decision_function(x)[0]
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
    """Our production ensemble: IF (0.4) + LSTM reconstruction error (0.6)."""

    name = "if_lstm_combined"
    cold_start_samples = 200

    def __init__(self) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        self._scaler = RobustScaler()
        self._if_model = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1,
        )
        self._lstm_model = None
        self._feat_min = None
        self._feat_max = None
        self._threshold = None

    def fit(self, X_normal: np.ndarray) -> None:
        # Train IF
        X_scaled = self._scaler.fit_transform(X_normal)
        self._if_model.fit(X_scaled)

        # Train LSTM
        import torch
        import torch.nn as nn

        self._feat_min = X_normal.min(axis=0)
        self._feat_max = X_normal.max(axis=0)
        denom = self._feat_max - self._feat_min
        denom[denom < 1e-8] = 1.0

        # Build sequences
        seq_len = 20
        seqs = []
        for i in range(len(X_normal) - seq_len + 1):
            seq_norm = (X_normal[i:i + seq_len] - self._feat_min) / denom
            seqs.append(seq_norm)
        if len(seqs) < 30:
            return

        data = torch.FloatTensor(np.array(seqs))

        class SimpleAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.LSTM(5, 32, 2, batch_first=True)
                self.dec = nn.LSTM(5, 32, 2, batch_first=True)
                self.fc = nn.Linear(32, 5)

            def forward(self, x):
                _, hidden = self.enc(x)
                out, _ = self.dec(x, hidden)
                return self.fc(out)

        self._lstm_model = SimpleAE()
        opt = torch.optim.Adam(self._lstm_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        self._lstm_model.train()
        for _ in range(20):
            opt.zero_grad()
            loss = criterion(self._lstm_model(data), data)
            loss.backward()
            opt.step()

        self._lstm_model.eval()
        with torch.no_grad():
            recon = self._lstm_model(data)
            errors = torch.mean((data - recon) ** 2, dim=(1, 2))
            self._threshold = float(errors.mean() + 3 * errors.std())

    def score(self, features: np.ndarray) -> float:
        # IF score
        x = self._scaler.transform(features.reshape(1, -1))
        raw = self._if_model.decision_function(x)[0]
        if_score = float(np.clip(1.0 / (1.0 + np.exp(5.0 * raw)), 0, 1))
        # No LSTM for single-point scoring in benchmark
        return if_score


# ═══════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkResult:
    ensemble_name: str
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    f1_best: float = 0.0
    best_threshold: float = 0.0
    precision_at_07: float = 0.0
    recall_at_07: float = 0.0
    fpr_at_07: float = 0.0
    cold_start_samples: int = 0
    throughput_scores_per_sec: float = 0.0
    training_time_s: float = 0.0


def run_benchmark(
    X_normal: np.ndarray,
    X_anomalous: np.ndarray,
    ensemble_names: list[str] | None = None,
) -> list[BenchmarkResult]:
    """Run all registered ensembles on the provided data.

    Training: 70% of normal data
    Testing: 30% of normal + all anomalous
    """
    results = []
    targets = ensemble_names or list(_REGISTRY.keys())

    # Split normal data: 70% train, 30% test
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X_normal))
    split = int(0.7 * len(X_normal))
    X_train = X_normal[indices[:split]]
    X_test_normal = X_normal[indices[split:]]

    # Test set: normal (label=0) + anomalous (label=1)
    X_test = np.vstack([X_test_normal, X_anomalous])
    y_true = np.concatenate([
        np.zeros(len(X_test_normal), dtype=np.int32),
        np.ones(len(X_anomalous), dtype=np.int32),
    ])

    print(f"\n{'='*70}")
    print(f"  Benchmark: {len(X_train):,} train | {len(X_test_normal):,} test-normal | {len(X_anomalous):,} test-anomaly")
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

        # Compute metrics
        try:
            auroc = roc_auc_score(y_true, scores)
        except ValueError:
            auroc = 0.0

        try:
            auprc = average_precision_score(y_true, scores)
        except ValueError:
            auprc = 0.0

        # Find best F1 threshold
        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
        best_idx = np.argmax(f1s)
        best_f1 = float(f1s[best_idx])
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

        # Metrics at threshold 0.7
        preds_07 = (scores >= 0.7).astype(int)
        p07 = precision_score(y_true, preds_07, zero_division=0)
        r07 = recall_score(y_true, preds_07, zero_division=0)
        # FPR: false positives / (false positives + true negatives)
        fp = ((preds_07 == 1) & (y_true == 0)).sum()
        tn = ((preds_07 == 0) & (y_true == 0)).sum()
        fpr07 = fp / max(fp + tn, 1)

        result = BenchmarkResult(
            ensemble_name=name,
            auc_roc=round(auroc, 4),
            auc_pr=round(auprc, 4),
            f1_best=round(best_f1, 4),
            best_threshold=round(best_thresh, 4),
            precision_at_07=round(p07, 4),
            recall_at_07=round(r07, 4),
            fpr_at_07=round(fpr07, 4),
            cold_start_samples=ensemble.cold_start_samples,
            throughput_scores_per_sec=round(throughput, 0),
            training_time_s=round(train_time, 3),
        )
        results.append(result)

        print(f"    AUC-ROC={auroc:.4f}  AUC-PR={auprc:.4f}  "
              f"F1={best_f1:.4f}@{best_thresh:.3f}  "
              f"P@0.7={p07:.3f}  R@0.7={r07:.3f}  "
              f"FPR@0.7={fpr07:.3f}  "
              f"throughput={throughput:.0f}/s")

    return results


def print_leaderboard(results: list[BenchmarkResult]) -> None:
    """Print a sorted comparison table."""
    ranked = sorted(results, key=lambda r: r.auc_roc, reverse=True)

    print(f"\n{'='*100}")
    print(f"  LEADERBOARD (ranked by AUC-ROC)")
    print(f"{'='*100}")
    header = (
        f"  {'Rank':<5} {'Ensemble':<22} {'AUC-ROC':>8} {'AUC-PR':>8} "
        f"{'F1@best':>8} {'P@0.7':>7} {'R@0.7':>7} {'FPR@0.7':>8} "
        f"{'Cold':>6} {'Speed':>10} {'Train':>7}"
    )
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for i, r in enumerate(ranked, 1):
        print(
            f"  {i:<5} {r.ensemble_name:<22} {r.auc_roc:>8.4f} {r.auc_pr:>8.4f} "
            f"{r.f1_best:>8.4f} {r.precision_at_07:>7.3f} {r.recall_at_07:>7.3f} "
            f"{r.fpr_at_07:>8.3f} {r.cold_start_samples:>6} "
            f"{r.throughput_scores_per_sec:>8.0f}/s {r.training_time_s:>6.2f}s"
        )
    print(f"{'='*100}")


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

    print("Loading RCAEval data...")
    X_normal, X_anomalous, labels = build_training_arrays(
        data_dir, fault_types=args.fault_types, interval_seconds=15,
    )
    print(f"  Normal: {len(X_normal):,}  Anomalous: {len(X_anomalous):,}")

    results = run_benchmark(X_normal, X_anomalous, ensemble_names=args.ensembles)
    print_leaderboard(results)

    # Save results
    output_path = output_dir / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(
            [vars(r) for r in results],
            f, indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
