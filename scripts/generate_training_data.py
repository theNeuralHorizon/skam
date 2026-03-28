#!/usr/bin/env python3
"""
Generate synthetic training data from real TrainTicket Prometheus KPI data.

Pipeline:
  1. Load real CSVs from data/trainticket/
  2. Map raw KPI columns -> SKAM 16-feature format (matching FeatureEngineer output)
  3. Fit multivariate Gaussian per-service from real distributions
  4. Sample N synthetic points preserving learned correlations
  5. Inject anomaly samples using the labeled anomaly window
  6. Save per-service .npz files for model training

Usage:
  python scripts/generate_training_data.py
  python scripts/generate_training_data.py --samples 500 --anomaly-ratio 0.1
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import deque

import numpy as np

# --- Configuration ------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "trainticket"
EXPERIMENT = "ts-auth-mongo_MongoDB_4.4.15_2022-07-27"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "platform" / "anomaly-detector" / "training_data"

SERVICE_MAP = {
    "ts-auth-service":         "api-gateway",
    "ts-user-service":         "user-service",
    "ts-order-service":        "product-service",
    "ts-travel-service":       "order-service",
    "ts-payment-service":      "payment-service",
    "ts-food-service":         "cart-service",
    "ts-notification-service": "notification-service",
}

# Rows where the labeled anomaly occurs in this experiment
ANOMALY_ROWS = set(range(21, 26))  # rows 21-25 (0-indexed from CSV data rows)

# The 16 features that FeatureEngineer.extract() produces
FEATURE_NAMES = [
    "request_rate", "error_rate", "latency_p50", "latency_p99",
    "cpu_usage", "memory_usage_mb", "restart_count", "error_ratio",
    "latency_spread", "request_rate_zscore", "error_rate_zscore",
    "latency_p99_zscore", "cpu_zscore", "request_rate_delta",
    "error_rate_delta", "latency_delta",
]


def load_raw_csv(service_name: str) -> list[dict]:
    """Load a single service's MicroRCA CSV."""
    kpi_dir = DATA_DIR / "anomalies_microservice_trainticket_version_configurations" / EXPERIMENT / "MicroRCA"
    csv_path = kpi_dir / f"{service_name}_microRCA.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found")
        return []

    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "ctn_cpu": float(row["ctn_cpu"]),
                "ctn_network": float(row["ctn_network"]),
                "ctn_memory": float(row["ctn_memory"]),
                "node_cpu": float(row["node_cpu"]),
                "node_network": float(row["node_network"]),
                "node_memory": float(row["node_memory"]),
            })
    return rows


def raw_to_features(rows: list[dict], is_anomaly_service: bool) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw KPI rows into SKAM's 16-feature vectors + labels.

    Mirrors the logic of FeatureEngineer.extract() to produce the same
    feature space that the models will see at inference time.
    """
    features_list = []
    labels = []

    # Rolling windows for z-score and delta computation
    window_size = 20
    history = {k: deque(maxlen=window_size) for k in [
        "request_rate", "error_rate", "latency_p99", "cpu_usage"
    ]}

    for i, row in enumerate(rows):
        is_anomaly = is_anomaly_service and i in ANOMALY_ROWS

        # Map raw metrics -> SKAM raw features
        cpu = row["ctn_cpu"]
        net = row["ctn_network"]
        mem = row["ctn_memory"]
        node_cpu = row["node_cpu"]

        # Derive request_rate from node_cpu (higher CPU ~ more requests)
        request_rate = node_cpu * 200 + np.random.uniform(-2, 2)

        # Error rate
        if is_anomaly:
            error_rate = request_rate * (0.08 + cpu * 0.5 + np.random.uniform(0, 0.05))
        else:
            error_rate = request_rate * (0.005 + cpu * 0.02 + np.random.uniform(0, 0.003))

        # Latency
        latency_p50 = 0.01 + net * 1e5
        latency_p99 = latency_p50 * 3 + (1.5 if is_anomaly else 0)

        # Memory in MB
        memory_mb = max(20, mem / 1000 + 30)

        # Restart count
        restart_count = float(np.random.randint(2, 6) if is_anomaly else np.random.randint(0, 2))

        # Store in history for rolling stats
        history["request_rate"].append(request_rate)
        history["error_rate"].append(error_rate)
        history["latency_p99"].append(latency_p99)
        history["cpu_usage"].append(cpu)

        # Derived features
        error_ratio = error_rate / request_rate if request_rate > 0 else 0.0
        latency_spread = latency_p99 - latency_p50

        # Z-scores (same logic as FeatureEngineer._zscore)
        def zscore(window):
            if len(window) < 3:
                return 0.0
            arr = np.array(list(window))
            return float((arr[-1] - arr.mean()) / max(arr.std(), 1e-10))

        # Rate of change (same logic as FeatureEngineer._rate_of_change)
        def delta(window):
            if len(window) < 2:
                return 0.0
            prev, curr = window[-2], window[-1]
            return float((curr - prev) / max(abs(prev), 1e-10))

        feature_vec = [
            request_rate,
            error_rate,
            latency_p50,
            latency_p99,
            cpu,
            memory_mb,
            restart_count,
            error_ratio,
            latency_spread,
            zscore(history["request_rate"]),
            zscore(history["error_rate"]),
            zscore(history["latency_p99"]),
            zscore(history["cpu_usage"]),
            delta(history["request_rate"]),
            delta(history["error_rate"]),
            delta(history["latency_p99"]),
        ]

        features_list.append(feature_vec)
        labels.append(1 if is_anomaly else 0)

    return np.array(features_list), np.array(labels)


def generate_synthetic(real_features: np.ndarray, real_labels: np.ndarray,
                       n_normal: int, n_anomaly: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data by fitting per-class Gaussian distributions
    to real feature vectors and sampling from them.

    This preserves the learned covariance structure so the synthetic data
    follows the same statistical patterns as the real TrainTicket data.
    """
    normal_mask = real_labels == 0
    anomaly_mask = real_labels == 1

    normal_data = real_features[normal_mask]
    anomaly_data = real_features[anomaly_mask]

    # Fit multivariate Gaussian to normal data
    normal_mean = normal_data.mean(axis=0)
    normal_cov = np.cov(normal_data, rowvar=False)
    # Regularize covariance to ensure positive semi-definite
    normal_cov += np.eye(normal_cov.shape[0]) * 1e-6

    # Sample normal synthetic points
    syn_normal = np.random.multivariate_normal(normal_mean, normal_cov, size=n_normal)

    # For anomaly data -- if we have enough real anomaly samples
    if len(anomaly_data) >= 3:
        anomaly_mean = anomaly_data.mean(axis=0)
        anomaly_cov = np.cov(anomaly_data, rowvar=False)
        anomaly_cov += np.eye(anomaly_cov.shape[0]) * 1e-6
        syn_anomaly = np.random.multivariate_normal(anomaly_mean, anomaly_cov, size=n_anomaly)
    else:
        # Perturb normal distribution to create synthetic anomalies
        # Increase error-related features, decrease latency-sensitive ones
        anomaly_mean = normal_mean.copy()
        anomaly_mean[1] *= 5     # error_rate 5x
        anomaly_mean[3] *= 3     # latency_p99 3x
        anomaly_mean[4] *= 4     # cpu_usage 4x
        anomaly_mean[6] += 3     # restart_count +3
        anomaly_mean[7] *= 5     # error_ratio 5x
        anomaly_mean[8] *= 3     # latency_spread 3x
        syn_anomaly = np.random.multivariate_normal(anomaly_mean, normal_cov * 2, size=n_anomaly)

    # Clamp non-negative features
    for col in [0, 1, 2, 3, 4, 5, 6, 7, 8]:  # rates, latencies, counts
        syn_normal[:, col] = np.maximum(0, syn_normal[:, col])
        syn_anomaly[:, col] = np.maximum(0, syn_anomaly[:, col])

    # Combine
    X = np.vstack([syn_normal, syn_anomaly])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def main():
    parser = argparse.ArgumentParser(description="Generate training data from real KPIs")
    parser.add_argument("--samples", type=int, default=500, help="Total normal samples per service")
    parser.add_argument("--anomaly-ratio", type=float, default=0.1, help="Fraction of anomaly samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    n_normal = args.samples
    n_anomaly = int(args.samples * args.anomaly_ratio)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[datagen] Generating {n_normal} normal + {n_anomaly} anomaly samples per service")
    print(f"[datagen] Source: {DATA_DIR / 'anomalies_microservice_trainticket_version_configurations' / EXPERIMENT}")
    print(f"[datagen] Output: {OUTPUT_DIR}\n")

    # Anomaly services in this experiment
    anomaly_src_services = {"ts-auth-service", "ts-order-service"}

    for src_name, skam_name in SERVICE_MAP.items():
        print(f"  {skam_name} <- {src_name}")
        raw = load_raw_csv(src_name)
        if not raw:
            continue

        is_anomaly_svc = src_name in anomaly_src_services
        real_features, real_labels = raw_to_features(raw, is_anomaly_svc)
        print(f"    real: {len(real_features)} rows, {real_labels.sum():.0f} anomalies, {real_features.shape[1]} features")

        syn_X, syn_y = generate_synthetic(real_features, real_labels, n_normal, n_anomaly)
        print(f"    synthetic: {len(syn_X)} samples ({(syn_y==0).sum():.0f} normal, {(syn_y==1).sum():.0f} anomaly)")

        # Save
        out_path = OUTPUT_DIR / f"{skam_name}.npz"
        np.savez(out_path,
                 features=syn_X,
                 labels=syn_y,
                 feature_names=FEATURE_NAMES,
                 source_experiment=EXPERIMENT,
                 source_service=src_name)
        print(f"    saved: {out_path} ({syn_X.nbytes / 1024:.1f} KB)")

    # Also save the real data for reference
    print("\n  saving raw real features for reference...")
    for src_name, skam_name in SERVICE_MAP.items():
        raw = load_raw_csv(src_name)
        if not raw:
            continue
        is_anomaly_svc = src_name in anomaly_src_services
        real_X, real_y = raw_to_features(raw, is_anomaly_svc)
        out_path = OUTPUT_DIR / f"{skam_name}_real.npz"
        np.savez(out_path, features=real_X, labels=real_y, feature_names=FEATURE_NAMES)

    print(f"\n[datagen] Done! Training data saved to {OUTPUT_DIR}")
    print(f"[datagen] Run the anomaly detector to auto-load these files.")


if __name__ == "__main__":
    main()
