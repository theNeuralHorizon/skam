"""Validation tests for the RCAEval data loader.

Covers: parsing, feature extraction, downsampling, label splitting,
sequence windowing, and edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.training.data_loader import (
    FEATURE_ORDER,
    SERVICES,
    Scenario,
    _to_sequences,
    build_sequence_arrays,
    build_training_arrays,
    downsample_to_interval,
    extract_all_services,
    extract_service_features,
    iter_scenarios,
    load_scenario,
)


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════


class TestExtractServiceFeatures:
    """Validate that raw RCAEval columns map to the canonical 5-feature order."""

    def test_extracts_correct_columns(self) -> None:
        df = pd.DataFrame({
            "time": [1, 2, 3],
            "frontend_cpu": [1.0, 2.0, 3.0],
            "frontend_mem": [100.0, 200.0, 300.0],
            "frontend_load": [10.0, 20.0, 30.0],
            "frontend_latency": [0.01, 0.02, 0.03],
            "frontend_error": [0.0, 0.0, 0.0],
        })
        result = extract_service_features(df, "frontend")
        assert result is not None
        assert list(result.columns) == FEATURE_ORDER
        assert result.shape == (3, 5)

    def test_column_order_matches_feature_order(self) -> None:
        """The order must be: request_rate, error_rate, latency, cpu_usage, memory_usage."""
        df = pd.DataFrame({
            "time": [1],
            "svc_cpu": [10.0],
            "svc_mem": [500.0],
            "svc_load": [42.0],
            "svc_latency": [0.05],
            "svc_error": [0.01],
        })
        result = extract_service_features(df, "svc")
        assert result is not None
        row = result.iloc[0].to_dict()
        assert row["request_rate"] == 42.0     # _load
        assert row["error_rate"] == 0.01       # _error
        assert row["latency"] == 0.05          # _latency
        assert row["cpu_usage"] == 10.0        # _cpu
        assert row["memory_usage"] == 500.0    # _mem

    def test_missing_error_column_fills_with_zero(self) -> None:
        """Many services in the dataset lack error columns."""
        df = pd.DataFrame({
            "time": [1, 2],
            "svc_cpu": [1.0, 2.0],
            "svc_mem": [100.0, 200.0],
            "svc_load": [10.0, 20.0],
            "svc_latency": [0.01, 0.02],
            # No svc_error column
        })
        result = extract_service_features(df, "svc")
        assert result is not None
        assert (result["error_rate"] == 0.0).all()

    def test_returns_none_for_missing_required_column(self) -> None:
        """Must return None if cpu, mem, load, or latency columns are missing."""
        df = pd.DataFrame({
            "time": [1],
            "svc_cpu": [1.0],
            # Missing svc_mem, svc_load, svc_latency
        })
        result = extract_service_features(df, "svc")
        assert result is None

    def test_returns_none_for_nonexistent_service(self) -> None:
        df = pd.DataFrame({"time": [1], "other_cpu": [1.0]})
        result = extract_service_features(df, "nonexistent")
        assert result is None


class TestExtractAllServices:
    """Validate multi-service extraction."""

    def test_extracts_multiple_services(self) -> None:
        df = pd.DataFrame({
            "time": [1, 2],
            "frontend_cpu": [1.0, 2.0],
            "frontend_mem": [100.0, 200.0],
            "frontend_load": [10.0, 20.0],
            "frontend_latency": [0.01, 0.02],
            "frontend_error": [0.0, 0.0],
            "redis_cpu": [0.5, 0.6],
            "redis_mem": [50.0, 60.0],
            "redis_load": [5.0, 6.0],
            "redis_latency": [0.001, 0.002],
        })
        result = extract_all_services(df, services=["frontend", "redis"])
        assert "frontend" in result
        assert "redis" in result
        assert result["frontend"].shape == (2, 5)

    def test_skips_services_without_data(self) -> None:
        df = pd.DataFrame({
            "time": [1],
            "frontend_cpu": [1.0],
            "frontend_mem": [100.0],
            "frontend_load": [10.0],
            "frontend_latency": [0.01],
        })
        result = extract_all_services(df, services=["frontend", "nonexistent"])
        assert "frontend" in result
        assert "nonexistent" not in result


# ═══════════════════════════════════════════════════════════════════════
# Downsampling
# ═══════════════════════════════════════════════════════════════════════


class TestDownsample:
    """Validate downsampling from 1s to 15s resolution."""

    def test_reduces_row_count(self) -> None:
        n = 150  # 150 seconds → 10 bins at 15s interval
        df = pd.DataFrame(np.random.rand(n, 5), columns=FEATURE_ORDER)
        timestamps = pd.Series(range(1000, 1000 + n))
        result = downsample_to_interval(df, timestamps, interval_seconds=15)
        assert len(result) == 10

    def test_uses_mean_aggregation(self) -> None:
        """Values within each bin should be averaged."""
        df = pd.DataFrame({
            "request_rate": [10.0, 20.0, 30.0],
            "error_rate": [0.0, 0.0, 0.0],
            "latency": [0.01, 0.02, 0.03],
            "cpu_usage": [1.0, 2.0, 3.0],
            "memory_usage": [100.0, 200.0, 300.0],
        })
        timestamps = pd.Series([1000, 1001, 1002])
        # All 3 points fall in same 15s bin
        result = downsample_to_interval(df, timestamps, interval_seconds=15)
        assert len(result) == 1
        assert result["request_rate"].iloc[0] == pytest.approx(20.0)
        assert result["cpu_usage"].iloc[0] == pytest.approx(2.0)

    def test_preserves_feature_count(self) -> None:
        df = pd.DataFrame(np.random.rand(100, 5), columns=FEATURE_ORDER)
        timestamps = pd.Series(range(100))
        result = downsample_to_interval(df, timestamps, 15)
        assert result.shape[1] == 5


# ═══════════════════════════════════════════════════════════════════════
# Sequence windowing
# ═══════════════════════════════════════════════════════════════════════


class TestToSequences:
    """Validate sliding window generation."""

    def test_correct_shape(self) -> None:
        arr = np.random.rand(50, 5).astype(np.float32)
        seqs = _to_sequences(arr, seq_length=20)
        assert seqs.shape == (31, 20, 5)  # 50 - 20 + 1 = 31

    def test_too_short_returns_empty(self) -> None:
        arr = np.random.rand(10, 5).astype(np.float32)
        seqs = _to_sequences(arr, seq_length=20)
        assert seqs.shape[0] == 0

    def test_exact_length_returns_one(self) -> None:
        arr = np.random.rand(20, 5).astype(np.float32)
        seqs = _to_sequences(arr, seq_length=20)
        assert seqs.shape == (1, 20, 5)

    def test_sequences_are_contiguous(self) -> None:
        """Each sequence should start one step after the previous."""
        arr = np.arange(50 * 5, dtype=np.float32).reshape(50, 5)
        seqs = _to_sequences(arr, seq_length=3)
        # First seq starts at row 0, second at row 1
        np.testing.assert_array_equal(seqs[0, 0], arr[0])
        np.testing.assert_array_equal(seqs[1, 0], arr[1])
        np.testing.assert_array_equal(seqs[0, 2], arr[2])


# ═══════════════════════════════════════════════════════════════════════
# Scenario loading (requires downloaded dataset)
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture()
def _has_dataset(rcaeval_dir: Path) -> None:
    if not rcaeval_dir.exists():
        pytest.skip("RCAEval dataset not downloaded")


@pytest.mark.usefixtures("_has_dataset")
class TestScenarioLoading:
    """Tests that require the actual RCAEval dataset on disk."""

    def test_load_single_scenario(self, rcaeval_dir: Path) -> None:
        scenario_dir = rcaeval_dir / "adservice_cpu" / "1"
        if not scenario_dir.exists():
            pytest.skip("adservice_cpu/1 not found")
        s = load_scenario(scenario_dir)
        assert isinstance(s, Scenario)
        assert s.service == "adservice"
        assert s.fault_type == "cpu"
        assert s.run == 1
        assert s.inject_time > 0
        assert len(s.data) > 0

    def test_normal_anomaly_split(self, rcaeval_dir: Path) -> None:
        scenario_dir = rcaeval_dir / "adservice_cpu" / "1"
        if not scenario_dir.exists():
            pytest.skip("adservice_cpu/1 not found")
        s = load_scenario(scenario_dir)
        n_normal = s.normal_mask.sum()
        n_anomaly = s.anomaly_mask.sum()
        assert n_normal > 0, "Should have normal data before injection"
        assert n_anomaly > 0, "Should have anomalous data after injection"
        assert n_normal + n_anomaly == len(s.data)

    def test_iter_scenarios_yields_expected_count(self, rcaeval_dir: Path) -> None:
        """Should yield 5 runs for a single service+fault combo."""
        scenarios = list(iter_scenarios(
            rcaeval_dir,
            services=["adservice"],
            fault_types=["cpu"],
        ))
        assert len(scenarios) == 5

    def test_build_training_arrays_shape(self, rcaeval_dir: Path) -> None:
        X_n, X_a, labels = build_training_arrays(
            rcaeval_dir,
            fault_types=["cpu"],
            interval_seconds=15,
        )
        assert X_n.ndim == 2
        assert X_n.shape[1] == 5
        assert X_a.ndim == 2
        assert X_a.shape[1] == 5
        assert len(labels) == len(X_n) + len(X_a)
        assert (labels[:len(X_n)] == 0).all()
        assert (labels[len(X_n):] == 1).all()

    def test_build_sequence_arrays_shape(self, rcaeval_dir: Path) -> None:
        X_n, X_a, labels = build_sequence_arrays(
            rcaeval_dir,
            seq_length=20,
            interval_seconds=15,
            fault_types=["cpu"],
        )
        assert X_n.ndim == 3
        assert X_n.shape[1] == 20
        assert X_n.shape[2] == 5

    def test_no_nan_or_inf_in_training_data(self, rcaeval_dir: Path) -> None:
        X_n, X_a, _ = build_training_arrays(
            rcaeval_dir,
            fault_types=["cpu"],
        )
        assert not np.any(np.isnan(X_n)), "Normal data contains NaN"
        assert not np.any(np.isinf(X_n)), "Normal data contains Inf"
        assert not np.any(np.isnan(X_a)), "Anomalous data contains NaN"
        assert not np.any(np.isinf(X_a)), "Anomalous data contains Inf"

    def test_all_fault_types_present(self, rcaeval_dir: Path) -> None:
        """The dataset should have cpu, mem, delay, disk, loss fault types."""
        fault_types_seen = set()
        for s in iter_scenarios(rcaeval_dir, services=["adservice"]):
            fault_types_seen.add(s.fault_type)
        expected = {"cpu", "mem", "delay", "disk", "loss"}
        assert fault_types_seen == expected, f"Missing faults: {expected - fault_types_seen}"
