"""RCAEval data loader — transforms Online Boutique telemetry into the 5-feature
vectors expected by our anomaly detection models.

Dataset layout
--------------
online-boutique/{service}_{fault}/{run}/data.csv
online-boutique/{service}_{fault}/{run}/inject_time.txt

Each CSV has columns like ``{service}_cpu``, ``{service}_mem``,
``{service}_load``, ``{service}_latency``, ``{service}_error`` for every
service in the mesh.  ``inject_time.txt`` contains a Unix timestamp marking
the moment the fault was injected (roughly the midpoint of the 70-minute
trace).

Our anomaly detector expects a **5-feature vector per service** every 15 s:

    [request_rate, error_rate, p99_latency, cpu_usage, memory_usage]

This module handles the mapping from raw RCAEval columns to that format.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


# ── Feature mapping ──────────────────────────────────────────────────────────
# RCAEval columns use varying suffixes across dataset versions:
#   RE1 (per-scenario CSVs): _workload, _latency (p90), _latency-50, _latency-90
#   Multi-source sample:     _workload, _latency-50, _latency-90
#
# We try multiple suffix variants in priority order.
FEATURE_MAP: dict[str, list[str]] = {
    "cpu_usage":     ["_cpu"],
    "memory_usage":  ["_mem"],
    "request_rate":  ["_load", "_workload"],
    "latency":       ["_latency", "_latency-90", "_latency-50"],
    "error_rate":    ["_error"],
}

# Canonical feature ordering (must match anomaly detector)
FEATURE_ORDER = ["request_rate", "error_rate", "latency", "cpu_usage", "memory_usage"]

# Services present in Online Boutique that map well to our platform
SERVICES = [
    "adservice",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "frontend",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
]


@dataclass
class Scenario:
    """One fault-injection scenario (single CSV + inject_time)."""

    service: str
    fault_type: str
    run: int
    data: pd.DataFrame
    inject_time: int
    normal_mask: pd.Series = field(repr=False, default=None)  # type: ignore[assignment]
    anomaly_mask: pd.Series = field(repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.normal_mask = self.data["time"] < self.inject_time
        self.anomaly_mask = self.data["time"] >= self.inject_time


def load_scenario(scenario_dir: str | Path) -> Scenario:
    """Load a single scenario directory.

    Parameters
    ----------
    scenario_dir:
        Path like ``online-boutique/adservice_cpu/1/``
    """
    scenario_dir = Path(scenario_dir)
    csv_path = scenario_dir / "data.csv"
    inject_path = scenario_dir / "inject_time.txt"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data.csv in {scenario_dir}")

    df = pd.read_csv(csv_path)

    # Clean up: drop duplicate time column, fill NaN
    if "time.1" in df.columns:
        df = df.drop(columns=["time.1"])
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    # Parse scenario metadata from path
    parent = scenario_dir.parent.name  # e.g. "adservice_cpu"
    parts = parent.rsplit("_", 1)
    service = parts[0]
    fault_type = parts[1] if len(parts) == 2 else "unknown"
    run = int(scenario_dir.name)

    inject_time = 0
    if inject_path.exists():
        inject_time = int(inject_path.read_text().strip())

    return Scenario(
        service=service,
        fault_type=fault_type,
        run=run,
        data=df,
        inject_time=inject_time,
    )


def iter_scenarios(
    dataset_dir: str | Path,
    fault_types: list[str] | None = None,
    services: list[str] | None = None,
) -> Iterator[Scenario]:
    """Iterate over all scenarios in the dataset.

    Parameters
    ----------
    dataset_dir:
        Path to ``online-boutique/`` directory.
    fault_types:
        Optional filter — e.g. ``["cpu", "mem", "delay"]``.
    services:
        Optional filter — e.g. ``["frontend", "cartservice"]``.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for scenario_group in sorted(dataset_dir.iterdir()):
        if not scenario_group.is_dir():
            continue

        parts = scenario_group.name.rsplit("_", 1)
        svc = parts[0]
        fault = parts[1] if len(parts) == 2 else "unknown"

        if services and svc not in services:
            continue
        if fault_types and fault not in fault_types:
            continue

        for run_dir in sorted(scenario_group.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                yield load_scenario(run_dir)
            except Exception as exc:
                print(f"  [WARN] Skipping {run_dir}: {exc}")


def extract_service_features(
    df: pd.DataFrame,
    service: str,
) -> pd.DataFrame | None:
    """Extract the 5-feature vector for *service* from a raw RCAEval DataFrame.

    Tries multiple column suffix variants (e.g. ``_load`` then ``_workload``)
    for each canonical feature.  Returns a DataFrame with columns matching
    ``FEATURE_ORDER``, or ``None`` if required columns are not present.
    """
    features: dict[str, pd.Series] = {}
    missing: list[str] = []

    for canonical, suffixes in FEATURE_MAP.items():
        found = False
        for suffix in suffixes:
            col = f"{service}{suffix}"
            if col in df.columns:
                features[canonical] = df[col]
                found = True
                break

        if not found:
            # error_rate is optional — many services lack it
            if canonical == "error_rate":
                features[canonical] = pd.Series(0.0, index=df.index)
            else:
                missing.append(f"{service}{suffixes[0]}")

    if missing:
        return None

    result = pd.DataFrame(features, index=df.index)

    # Ensure consistent column order
    return result[FEATURE_ORDER]


def extract_all_services(
    df: pd.DataFrame,
    services: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Extract 5-feature DataFrames for every service found in *df*.

    Returns ``{service_name: DataFrame}`` where each DataFrame has columns
    matching ``FEATURE_ORDER``.
    """
    targets = services or SERVICES
    result: dict[str, pd.DataFrame] = {}

    for svc in targets:
        feat = extract_service_features(df, svc)
        if feat is not None:
            result[svc] = feat

    return result


def downsample_to_interval(
    df: pd.DataFrame,
    timestamps: pd.Series,
    interval_seconds: int = 15,
) -> pd.DataFrame:
    """Downsample from 1-second to *interval_seconds* resolution.

    Uses mean aggregation over non-overlapping windows to match what
    Prometheus scrapes would look like at a 15 s interval.
    """
    # Create time bins
    t0 = timestamps.iloc[0]
    bins = (timestamps - t0) // interval_seconds
    grouped = df.groupby(bins).mean()
    return grouped.reset_index(drop=True)


def build_training_arrays(
    dataset_dir: str | Path,
    target_service: str | None = None,
    interval_seconds: int = 15,
    fault_types: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X_normal, X_anomalous, y) arrays from the full dataset.

    Parameters
    ----------
    dataset_dir:
        Path to ``online-boutique/``.
    target_service:
        If set, extract features only for this service.  Otherwise extract
        the *fault target* service from each scenario.
    interval_seconds:
        Downsample interval (default 15 s to match Prometheus scrape).
    fault_types:
        Filter to these fault types.

    Returns
    -------
    X_normal : ndarray, shape (N, 5)
        Feature vectors from pre-injection (normal) windows.
    X_anomalous : ndarray, shape (M, 5)
        Feature vectors from post-injection (anomalous) windows.
    labels : ndarray, shape (N+M,)
        0 = normal, 1 = anomalous.
    """
    normals: list[np.ndarray] = []
    anomalies: list[np.ndarray] = []

    for scenario in iter_scenarios(dataset_dir, fault_types=fault_types):
        svc = target_service or scenario.service
        feat = extract_service_features(scenario.data, svc)
        if feat is None:
            continue

        timestamps = scenario.data["time"]

        # Split into normal / anomalous
        normal_feat = feat.loc[scenario.normal_mask]
        anomaly_feat = feat.loc[scenario.anomaly_mask]
        normal_ts = timestamps.loc[scenario.normal_mask]
        anomaly_ts = timestamps.loc[scenario.anomaly_mask]

        # Downsample
        if len(normal_feat) > 0:
            ds_normal = downsample_to_interval(normal_feat, normal_ts, interval_seconds)
            normals.append(ds_normal.values)

        if len(anomaly_feat) > 0:
            ds_anomaly = downsample_to_interval(anomaly_feat, anomaly_ts, interval_seconds)
            anomalies.append(ds_anomaly.values)

    if not normals or not anomalies:
        raise ValueError("No data found — check dataset_dir and filters")

    X_normal = np.vstack(normals).astype(np.float32)
    X_anomalous = np.vstack(anomalies).astype(np.float32)

    labels = np.concatenate([
        np.zeros(len(X_normal), dtype=np.int32),
        np.ones(len(X_anomalous), dtype=np.int32),
    ])

    return X_normal, X_anomalous, labels


def build_sequence_arrays(
    dataset_dir: str | Path,
    seq_length: int = 20,
    interval_seconds: int = 15,
    fault_types: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windowed sequences for LSTM training.

    Returns
    -------
    X_normal_seq : ndarray, shape (N, seq_length, 5)
    X_anomalous_seq : ndarray, shape (M, seq_length, 5)
    labels : ndarray, shape (N+M,)
    """
    normal_seqs: list[np.ndarray] = []
    anomaly_seqs: list[np.ndarray] = []

    for scenario in iter_scenarios(dataset_dir, fault_types=fault_types):
        feat = extract_service_features(scenario.data, scenario.service)
        if feat is None:
            continue

        timestamps = scenario.data["time"]

        # Process normal portion
        normal_feat = feat.loc[scenario.normal_mask]
        normal_ts = timestamps.loc[scenario.normal_mask]
        if len(normal_feat) > seq_length * interval_seconds:
            ds = downsample_to_interval(normal_feat, normal_ts, interval_seconds)
            seqs = _to_sequences(ds.values, seq_length)
            if len(seqs) > 0:
                normal_seqs.append(seqs)

        # Process anomalous portion
        anomaly_feat = feat.loc[scenario.anomaly_mask]
        anomaly_ts = timestamps.loc[scenario.anomaly_mask]
        if len(anomaly_feat) > seq_length * interval_seconds:
            ds = downsample_to_interval(anomaly_feat, anomaly_ts, interval_seconds)
            seqs = _to_sequences(ds.values, seq_length)
            if len(seqs) > 0:
                anomaly_seqs.append(seqs)

    if not normal_seqs:
        raise ValueError("No normal sequences found")

    X_normal = np.vstack(normal_seqs).astype(np.float32)
    X_anomalous = np.vstack(anomaly_seqs).astype(np.float32) if anomaly_seqs else np.empty(
        (0, seq_length, 5), dtype=np.float32,
    )

    labels = np.concatenate([
        np.zeros(len(X_normal), dtype=np.int32),
        np.ones(len(X_anomalous), dtype=np.int32),
    ])

    return X_normal, X_anomalous, labels


def _to_sequences(arr: np.ndarray, seq_length: int) -> np.ndarray:
    """Sliding window over a 2D array → 3D array of overlapping sequences."""
    if len(arr) < seq_length:
        return np.empty((0, seq_length, arr.shape[1]), dtype=arr.dtype)

    sequences = []
    for i in range(len(arr) - seq_length + 1):
        sequences.append(arr[i : i + seq_length])
    return np.array(sequences)
