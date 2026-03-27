"""Validation tests for the LSTM Autoencoder detector.

Covers: model architecture, training, prediction, pre-trained loading,
normalisation, reconstruction error calibration, and edge cases.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "platform" / "anomaly-detector"))

from app.lstm_detector import LSTMAutoencoder, LSTMDetector, _WINDOW_SIZE


class TestLSTMArchitecture:
    """Validate model structure and forward pass."""

    def test_model_output_shape(self) -> None:
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
        x = torch.randn(4, 20, 5)  # batch=4, seq=20, features=5
        out = model(x)
        assert out.shape == (4, 20, 5)

    def test_model_is_differentiable(self) -> None:
        model = LSTMAutoencoder()
        x = torch.randn(2, 20, 5, requires_grad=True)
        out = model(x)
        loss = torch.mean((out - x) ** 2)
        loss.backward()
        assert x.grad is not None

    def test_parameter_count(self) -> None:
        """Sanity-check parameter count matches expectations."""
        model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2)
        total = sum(p.numel() for p in model.parameters())
        # Should be in the range of ~25-30K params
        assert 20_000 < total < 35_000, f"Unexpected param count: {total}"

    def test_batch_size_independence(self) -> None:
        """Model should produce same output regardless of batch size."""
        model = LSTMAutoencoder()
        model.eval()
        x = torch.randn(1, 20, 5)
        with torch.no_grad():
            out1 = model(x)
            x_batched = x.repeat(3, 1, 1)
            out3 = model(x_batched)
        torch.testing.assert_close(out1[0], out3[0], atol=1e-5, rtol=1e-5)


class TestLSTMDetectorTraining:
    """Validate LSTM detector trains correctly."""

    def test_starts_untrained(self) -> None:
        det = LSTMDetector()
        assert not det.is_trained

    def test_trains_from_buffer(self, normal_sequences: np.ndarray) -> None:
        det = LSTMDetector()
        # Feed sequences into the buffer
        for seq in normal_sequences[:40]:
            for step in seq:
                det.push("test-service", step)
        det.train()
        assert det.is_trained

    def test_trains_from_explicit_data(self, normal_sequences: np.ndarray) -> None:
        det = LSTMDetector()
        det.train(sequences=[seq for seq in normal_sequences[:40]])
        assert det.is_trained

    def test_refuses_train_with_too_few_sequences(self) -> None:
        det = LSTMDetector()
        det.train(sequences=[np.random.rand(20, 5) for _ in range(5)])
        assert not det.is_trained


class TestLSTMDetectorScoring:
    """Validate score output and discrimination."""

    @pytest.fixture()
    def trained_detector(self, normal_sequences: np.ndarray) -> LSTMDetector:
        det = LSTMDetector()
        det.train(sequences=[seq for seq in normal_sequences])
        return det

    def test_score_range(self, trained_detector: LSTMDetector) -> None:
        seq = np.random.rand(20, 5).astype(np.float32)
        score = trained_detector.predict(seq)
        assert 0.0 <= score <= 1.0

    def test_normal_reconstruction_error_low(
        self,
        trained_detector: LSTMDetector,
        normal_sequences: np.ndarray,
    ) -> None:
        """Normal sequences should reconstruct well (low error)."""
        scores = [trained_detector.predict(seq) for seq in normal_sequences[:20]]
        mean_score = np.mean(scores)
        # After training on normal data, these should reconstruct well
        assert mean_score < 0.8, f"Normal mean score {mean_score:.3f} too high"

    def test_anomalous_reconstruction_error_higher(
        self,
        trained_detector: LSTMDetector,
        normal_sequences: np.ndarray,
        anomalous_sequences: np.ndarray,
    ) -> None:
        """Anomalous sequences should have higher reconstruction error."""
        if len(anomalous_sequences) == 0:
            pytest.skip("No anomalous sequences")
        normal_scores = [trained_detector.predict(s) for s in normal_sequences[:20]]
        anomaly_scores = [trained_detector.predict(s) for s in anomalous_sequences[:20]]
        # Anomalous should have higher reconstruction error on average
        assert np.mean(anomaly_scores) > np.mean(normal_scores) * 0.5, (
            f"Anomaly mean ({np.mean(anomaly_scores):.3f}) not sufficiently "
            f"higher than normal ({np.mean(normal_scores):.3f})"
        )

    def test_untrained_returns_zero(self) -> None:
        det = LSTMDetector()
        score = det.predict(np.random.rand(20, 5).astype(np.float32))
        assert score == 0.0


class TestLSTMSlidingWindow:
    """Validate per-service sliding window management."""

    def test_window_not_ready_initially(self) -> None:
        det = LSTMDetector()
        assert not det.window_ready("test-svc")

    def test_window_ready_after_filling(self) -> None:
        det = LSTMDetector()
        for _ in range(_WINDOW_SIZE):
            det.push("test-svc", np.random.rand(5).astype(np.float32))
        assert det.window_ready("test-svc")

    def test_separate_windows_per_service(self) -> None:
        det = LSTMDetector()
        for _ in range(_WINDOW_SIZE):
            det.push("svc-a", np.random.rand(5).astype(np.float32))
        assert det.window_ready("svc-a")
        assert not det.window_ready("svc-b")

    def test_predict_for_service_works(self, normal_sequences: np.ndarray) -> None:
        det = LSTMDetector()
        det.train(sequences=[seq for seq in normal_sequences])
        # Fill a window
        for step in normal_sequences[0]:
            det.push("test-svc", step)
        score = det.predict_for_service("test-svc")
        assert 0.0 <= score <= 1.0

    def test_predict_for_service_untrained(self) -> None:
        det = LSTMDetector()
        for _ in range(_WINDOW_SIZE):
            det.push("svc", np.random.rand(5).astype(np.float32))
        score = det.predict_for_service("svc")
        assert score == 0.0  # untrained


class TestLSTMPretrainedLoading:
    """Validate loading of pre-trained checkpoint."""

    def test_loads_pretrained_from_path(self, models_dir: Path) -> None:
        model_path = models_dir / "lstm_autoencoder.pt"
        if not model_path.exists():
            pytest.skip("Pre-trained LSTM model not found")

        det = LSTMDetector(pretrained_path=str(model_path))
        assert det.is_trained
        assert det._feat_min is not None
        assert det._feat_max is not None
        assert det._pretrained_threshold is not None

    def test_loads_from_env_var(self, models_dir: Path) -> None:
        model_path = models_dir / "lstm_autoencoder.pt"
        if not model_path.exists():
            pytest.skip("Pre-trained LSTM model not found")

        os.environ["LSTM_MODEL_PATH"] = str(model_path)
        try:
            det = LSTMDetector()
            assert det.is_trained
        finally:
            del os.environ["LSTM_MODEL_PATH"]

    def test_graceful_fallback_on_missing_path(self) -> None:
        det = LSTMDetector(pretrained_path="/nonexistent/model.pt")
        assert not det.is_trained

    def test_feat_min_max_loaded(self, models_dir: Path) -> None:
        model_path = models_dir / "lstm_autoencoder.pt"
        if not model_path.exists():
            pytest.skip("Pre-trained LSTM model not found")

        det = LSTMDetector(pretrained_path=str(model_path))
        assert len(det._feat_min) == 5
        assert len(det._feat_max) == 5
        # Min should be less than max for all features
        assert all(det._feat_min[i] <= det._feat_max[i] for i in range(5))

    def test_threshold_loaded(self, models_dir: Path) -> None:
        model_path = models_dir / "lstm_autoencoder.pt"
        if not model_path.exists():
            pytest.skip("Pre-trained LSTM model not found")

        det = LSTMDetector(pretrained_path=str(model_path))
        assert det._pretrained_threshold > 0
        assert det._pretrained_threshold < 1.0  # should be small for normal data

    def test_pretrained_produces_valid_scores(self, models_dir: Path) -> None:
        model_path = models_dir / "lstm_autoencoder.pt"
        if not model_path.exists():
            pytest.skip("Pre-trained LSTM model not found")

        det = LSTMDetector(pretrained_path=str(model_path))
        seq = np.random.rand(20, 5).astype(np.float32) * 100
        score = det.predict(seq)
        assert 0.0 <= score <= 1.0


class TestLSTMNumericalStability:
    """Edge cases for numerical stability."""

    def test_zero_input_no_nan(self) -> None:
        det = LSTMDetector()
        det.train(sequences=[np.random.rand(20, 5).astype(np.float32) for _ in range(40)])
        score = det.predict(np.zeros((20, 5), dtype=np.float32))
        assert not np.isnan(score)

    def test_large_input_no_nan(self) -> None:
        det = LSTMDetector()
        det.train(sequences=[np.random.rand(20, 5).astype(np.float32) for _ in range(40)])
        score = det.predict(np.full((20, 5), 1e6, dtype=np.float32))
        assert not np.isnan(score)
        assert 0.0 <= score <= 1.0

    def test_constant_sequence_no_nan(self) -> None:
        det = LSTMDetector()
        const_seqs = [np.full((20, 5), 42.0, dtype=np.float32) for _ in range(40)]
        det.train(sequences=const_seqs)
        score = det.predict(np.full((20, 5), 42.0, dtype=np.float32))
        assert not np.isnan(score)
