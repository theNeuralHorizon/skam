#!/usr/bin/env python3
"""Export Prometheus metrics as CSV training data for ML model retraining.

Queries the Prometheus HTTP API for the 5 feature metrics across all
monitored services and writes a labelled CSV suitable for training
XGBoost, Isolation Forest, LSTM, and other anomaly detection models.

Usage::

    # Export last 2 hours of data (requires kubectl port-forward first)
    kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 &
    python scripts/export_training_data.py \\
        --prometheus-url http://localhost:9090 \\
        --hours 2 \\
        --output ml/data/own-logs/training_data.csv

    # With anomaly injection timestamps for labelling
    python scripts/export_training_data.py \\
        --prometheus-url http://localhost:9090 \\
        --hours 4 \\
        --inject-file scripts/inject_times.json \\
        --output ml/data/own-logs/training_data.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

SERVICES = [
    "api-gateway",
    "user-service",
    "product-service",
    "order-service",
    "payment-service",
    "notification-service",
]

# Same PromQL queries used by the anomaly detector collector
QUERIES = {
    "request_rate": 'rate(http_requests_total{{app="{service}"}}[1m])',
    "error_rate": 'rate(http_requests_total{{app="{service}",status=~"5.."}}[1m])',
    "p99_latency": 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{app="{service}"}}[1m]))',
    "cpu_usage": 'rate(container_cpu_usage_seconds_total{{pod=~"{service}.*"}}[1m])',
    "memory_usage": 'container_memory_usage_bytes{{pod=~"{service}.*"}}',
}


def query_range(
    client: httpx.Client,
    query: str,
    start: datetime,
    end: datetime,
    step: str = "15s",
) -> list[tuple[float, float]]:
    """Execute a Prometheus range query and return [(timestamp, value), ...]."""
    resp = client.get(
        "/api/v1/query_range",
        params={
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "success":
        return []

    results = data["data"]["result"]
    if not results:
        return []

    # Take first result series
    values = results[0].get("values", [])
    out = []
    for ts, val in values:
        try:
            v = float(val)
            if v != v:  # NaN
                v = 0.0
            out.append((float(ts), v))
        except (ValueError, TypeError):
            out.append((float(ts), 0.0))
    return out


def export_service(
    client: httpx.Client,
    service: str,
    start: datetime,
    end: datetime,
    step: str,
) -> pd.DataFrame:
    """Export all 5 features for a single service as a DataFrame."""
    series: dict[str, dict[float, float]] = {}

    for metric_name, query_template in QUERIES.items():
        query = query_template.format(service=service)
        data = query_range(client, query, start, end, step)
        series[metric_name] = {ts: val for ts, val in data}

    if not any(series.values()):
        return pd.DataFrame()

    # Get all timestamps
    all_ts = sorted(set().union(*(s.keys() for s in series.values())))
    if not all_ts:
        return pd.DataFrame()

    rows = []
    for ts in all_ts:
        row = {
            "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "service": service,
        }
        for metric_name in QUERIES:
            row[metric_name] = series[metric_name].get(ts, 0.0)

        # Compute error ratio
        req_rate = row.get("request_rate", 0.0)
        err_rate = row.get("error_rate", 0.0)
        if req_rate > 0:
            row["error_rate"] = min(err_rate / req_rate, 1.0)
        else:
            row["error_rate"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def label_data(
    df: pd.DataFrame,
    inject_times: dict[str, list[list[str]]] | None,
) -> pd.DataFrame:
    """Add a 'label' column: 0 = normal, 1 = anomaly."""
    df["label"] = 0

    if inject_times is None:
        return df

    for service, intervals in inject_times.items():
        for interval in intervals:
            inject_start = pd.Timestamp(interval[0])
            inject_end = pd.Timestamp(interval[1]) if len(interval) > 1 else inject_start + pd.Timedelta(minutes=5)

            mask = (
                (df["service"] == service)
                & (pd.to_datetime(df["timestamp"]) >= inject_start)
                & (pd.to_datetime(df["timestamp"]) <= inject_end)
            )
            df.loc[mask, "label"] = 1

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Prometheus metrics as CSV training data",
    )
    parser.add_argument(
        "--prometheus-url", type=str, default="http://localhost:9090",
        help="Prometheus HTTP API URL",
    )
    parser.add_argument("--hours", type=float, default=2.0, help="Hours of data to export")
    parser.add_argument("--step", type=str, default="15s", help="Query step interval")
    parser.add_argument(
        "--output", type=str, default="ml/data/own-logs/training_data.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--inject-file", type=str, default=None,
        help="JSON file with anomaly injection timestamps per service",
    )
    parser.add_argument(
        "--services", nargs="+", default=None,
        help="Services to export (default: all)",
    )
    args = parser.parse_args()

    services = args.services or SERVICES
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)

    print(f"Exporting Prometheus metrics")
    print(f"  Source    : {args.prometheus_url}")
    print(f"  Range     : {start.isoformat()} → {end.isoformat()}")
    print(f"  Step      : {args.step}")
    print(f"  Services  : {', '.join(services)}")
    print(f"  Output    : {args.output}")

    inject_times = None
    if args.inject_file:
        with open(args.inject_file) as f:
            inject_times = json.load(f)
        print(f"  Inject    : {args.inject_file}")

    client = httpx.Client(
        base_url=args.prometheus_url,
        timeout=httpx.Timeout(30.0, connect=10.0),
    )

    all_dfs: list[pd.DataFrame] = []
    for service in services:
        print(f"  Exporting {service}...", end=" ")
        try:
            df = export_service(client, service, start, end, args.step)
            if df.empty:
                print("no data")
            else:
                print(f"{len(df)} samples")
                all_dfs.append(df)
        except Exception as e:
            print(f"ERROR: {e}")

    client.close()

    if not all_dfs:
        print("\nNo data exported. Is Prometheus scraping metrics?")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = label_data(combined, inject_times)

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_path, index=False)
    print(f"\nExported {len(combined):,} rows to {output_path}")
    print(f"  Normal : {(combined['label'] == 0).sum():,}")
    print(f"  Anomaly: {(combined['label'] == 1).sum():,}")


if __name__ == "__main__":
    main()
