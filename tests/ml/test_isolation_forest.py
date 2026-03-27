"""Validation tests for the Isolation Forest detector.

Covers: training, scoring, pre-trained loading, normalisation,
score distribution, edge cases, and discrimination quality.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler

# Ensure the platform module is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "platform" / "anomaly-detector"))

from app.isolation_forest import IsolationForestDetector


class TestIsolationForestTraining:
    """Validate IF trains correctly from buffered samples."""

    def test_starts_untrained(self) -> None:
        det = IsolationForestDetector()
        assert not det.is_trained
        assert det.sample_count == 0

    def test_trains_after_sufficient_samples(self, normal_features: np.ndarray) -> None:
        det = IsolationForestDetector()
        for row in normal_features[:60]:
            det.add_sample(row)
        det.train()
        assert det.is_trained

    def test_refuses_train_with_too_few_samples(self) -> None:
        det = IsolationForestDetector()
        for i in range(10):
            det.add_sample(np.random.rand(5))
        det.train()  # should be a no-op
        assert not det.is_trained

    def test_auto_retrains_at_threshold(self) -> None:
        """Adding 1000+ samples should trigger automatic retraining."""
        det = IsolationForestDetector()
        rng = np.random.default_rng(42)
        for _ in range(1001):
            det.add_sample(rng.normal(size=5))
        assert det.is_trained

    def test_sample_count_increments(self) -> None:
        det = IsolationForestDetector()
        for i in range(10):
            det.add_sample(np.random.rand(5))
        assert det.sample_count == 10


class TestIsolationForestScoring:
    """Validate score output and discrimination."""

    @pytest.fixture()
    def trained_detector(self, normal_features: np.ndarray) -> IsolationForestDetector:
        det = IsolationForestDetector(contamination=0.05, n_estimators=100)
        for row in normal_features:
            det.add_sample(row)
        det.train()
        return det

    def test_score_range(self, trained_detector: IsolationForestDetector) -> None:
        """Scores must be in [0, 1]."""
        rng = np.random.default_rng(123)
        for _ in range(100):
            features = rng.normal(size=5)
            score = trained_detector.predict(features)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_normal_data_scores_low(
        self,
        trained_detector: IsolationForestDetector,
        normal_features: np.ndarray,
    ) -> None:
        """Normal data should mostly score below 0.5."""
        scores = [trained_detector.predict(row) for row in normal_features[:100]]
        median_score = np.median(scores)
        assert median_score < 0.5, f"Normal median score {median_score:.3f} too high"

    def test_anomalous_data_scores_high(
        self,
        trained_detector: IsolationForestDetector,
        anomalous_features: np.ndarray,
    ) -> None:
        """Anomalous data should score significantly higher than normal."""
        scores = [trained_detector.predict(row) for row in anomalous_features[:100]]
        median_score = np.median(scores)
        assert median_score > 0.5, f"Anomaly median score {median_score:.3f} too low"

    def test_score_separation(
        self,
        trained_detector: IsolationForestDetector,
        normal_features: np.ndarray,
        anomalous_features: np.ndarray,
    ) -> None:
        """Anomaly scores should be statistically higher than normal scores."""
        normal_scores = [trained_detector.predict(r) for r in normal_features[:100]]
        anomaly_scores = [trained_detector.predict(r) for r in anomalous_features[:100]]
        assert np.mean(anomaly_scores) > np.mean(normal_scores), (
            f"Anomaly mean ({np.mean(anomaly_scores):.3f}) should exceed "
            f"normal mean ({np.mean(normal_scores):.3f})"
        )

    def test_untrained_returns_zero(self) -> None:
        det = IsolationForestDetector()
        score = det.predict(np.random.rand(5))
        assert score == 0.0

    def test_rejects_wrong_feature_count(self) -> None:
        det = IsolationForestDetector()
        with pytest.raises(AssertionError):
            det.add_sample(np.array([1.0, 2.0, 3.0]))  # only 3 features


class TestIsolationForestPretrainedLoading:
    """Validate loading of pre-trained joblib artefact."""

    def test_loads_pretrained_from_path(self, models_dir: Path) -> None:
        model_path = models_dir / "isolation_forest.pkl"
        if not model_path.exists():
            pytest.skip("Pre-trained IF model not found")

        det = IsolationForestDetector(pretrained_path=str(model_path))
        assert det.is_trained
        # Should produce valid scores immediately
        score = det.predict(np.array([71.0, 0.0, 0.06, 5.4, 39e6]))
        assert 0.0 <= score <= 1.0

    def test_loads_from_env_var(self, models_dir: Path) -> None:
        model_path = models_dir / "isolation_forest.pkl"
        if not model_path.exists():
            pytest.skip("Pre-trained IF model not found")

        os.environ["IF_MODEL_PATH"] = str(model_path)
        try:
            det = IsolationForestDetector()
            assert det.is_trained
        finally:
            del os.environ["IF_MODEL_PATH"]

    def test_graceful_fallback_on_missing_path(self) -> None:
        """Should not crash when model path doesn't exist."""
        det = IsolationForestDetector(pretrained_path="/nonexistent/model.pkl")
        assert not det.is_trained

    def test_uses_robust_scaler_from_artefact(self, models_dir: Path) -> None:
        model_path = models_dir / "isolation_forest.pkl"
        if not model_path.exists():
            pytest.skip("Pre-trained IF model not found")

        det = IsolationForestDetector(pretrained_path=str(model_path))
        assert det._scaler is not None
        assert isinstance(det._scaler, RobustScaler)

    def test_pretrained_normal_scores_low(self, models_dir: Path) -> None:
        """Pre-trained model should give low scores to normal-looking data."""
        model_path = models_dir / "isolation_forest.pkl"
        if not model_path.exists():
            pytest.skip("Pre-trained IF model not found")

        det = IsolationForestDetector(pretrained_path=str(model_path))
        # Values near training distribution medians
        normal = np.array([71.0, 0.0, 0.06, 5.4, 39e6])
        score = det.predict(normal)
        assert score < 0.6, f"Normal data scored {score:.3f} on pretrained model"

    def test_pretrained_anomaly_scores_high(self, models_dir: Path) -> None:
        """Pre-trained model should give high scores to extreme data."""
        model_path = models_dir / "isolation_forest.pkl"
        if not model_path.exists():
            pytest.skip("Pre-trained IF model not found")

        det = IsolationForestDetector(pretrained_path=str(model_path))
        # Extreme values far outside training distribution
        anomaly = np.array([0.0, 0.5, 10.0, 99.0, 900e6])
        score = det.predict(anomaly)
        assert score > 0.4, f"Anomaly data scored {score:.3f} on pretrained model"


class TestIsolationForestNormalisation:
    """Validate that normalisation doesn't introduce NaN/Inf."""

    def test_no_nan_in_normalised_output(self, normal_features: np.ndarray) -> None:
        det = IsolationForestDetector()
        for row in normal_features[:60]:
            det.add_sample(row)
        normalised = det._normalise(normal_features[:10])
        assert not np.any(np.isnan(normalised))
        assert not np.any(np.isinf(normalised))

    def test_zero_variance_feature_handled(self) -> None:
        """If one feature is constant, normalisation should not produce NaN."""
        det = IsolationForestDetector()
        for _ in range(60):
            det.add_sample(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        normalised = det._normalise(np.array([[1.0, 0.0, 0.0, 0.0, 0.0]]))
        assert not np.any(np.isnan(normalised))
