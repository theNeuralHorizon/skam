"""Shared fixtures for the SKAM test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_MODELS_DIR = REPO_ROOT / "ml" / "models"
ML_DATA_DIR = REPO_ROOT / "ml" / "data"
RCAEVAL_DIR = ML_DATA_DIR / "rcaeval" / "online-boutique"


@pytest.fixture()
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture()
def models_dir() -> Path:
    return ML_MODELS_DIR


@pytest.fixture()
def rcaeval_dir() -> Path:
    return RCAEVAL_DIR


@pytest.fixture()
def normal_features() -> np.ndarray:
    """Synthetic normal feature vectors matching RCAEval distributions.

    Based on training_stats.json median values:
        request_rate ~71, error_rate ~0, latency ~0.06,
        cpu_usage ~5.4, memory_usage ~39M
    """
    rng = np.random.default_rng(42)
    n = 500
    return np.column_stack([
        rng.normal(loc=71.0, scale=20.0, size=n).clip(0),      # request_rate
        rng.exponential(scale=0.001, size=n).clip(0, 0.05),     # error_rate
        rng.lognormal(mean=-2.8, sigma=0.5, size=n).clip(0),    # latency
        rng.normal(loc=5.4, scale=1.5, size=n).clip(0),         # cpu_usage
        rng.normal(loc=39e6, scale=10e6, size=n).clip(0),       # memory_usage
    ]).astype(np.float32)


@pytest.fixture()
def anomalous_features() -> np.ndarray:
    """Synthetic anomalous feature vectors — clear deviations from normal."""
    rng = np.random.default_rng(99)
    n = 200
    return np.column_stack([
        rng.normal(loc=5.0, scale=3.0, size=n).clip(0),         # request_rate DROP
        rng.normal(loc=0.3, scale=0.1, size=n).clip(0, 1.0),    # error_rate SPIKE
        rng.normal(loc=2.0, scale=0.5, size=n).clip(0),          # latency SPIKE
        rng.normal(loc=95.0, scale=5.0, size=n).clip(0),         # cpu_usage SPIKE
        rng.normal(loc=450e6, scale=50e6, size=n).clip(0),       # memory_usage SPIKE
    ]).astype(np.float32)


@pytest.fixture()
def normal_sequences(normal_features: np.ndarray) -> np.ndarray:
    """Sliding-window sequences from normal features for LSTM testing."""
    seq_len = 20
    seqs = []
    for i in range(len(normal_features) - seq_len + 1):
        seqs.append(normal_features[i : i + seq_len])
    return np.array(seqs[:100])  # keep 100 sequences


@pytest.fixture()
def anomalous_sequences(anomalous_features: np.ndarray) -> np.ndarray:
    """Sliding-window sequences from anomalous features."""
    seq_len = 20
    seqs = []
    for i in range(len(anomalous_features) - seq_len + 1):
        seqs.append(anomalous_features[i : i + seq_len])
    return np.array(seqs[:50])
