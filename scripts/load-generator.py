#!/usr/bin/env python3
"""
Synthetic load generator for SKAM microservices.

Generates continuous HTTP traffic against the API gateway to produce
baseline telemetry for the anomaly detection ML models. Supports
configurable patterns: steady, ramp, and burst.

Usage:
    python scripts/load-generator.py
    python scripts/load-generator.py --rps 50 --pattern burst
    python scripts/load-generator.py --gateway http://localhost:30080
"""

import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass

import httpx


@dataclass
class LoadConfig:
    gateway_url: str
    target_rps: int
    pattern: str  # "steady", "ramp", "burst"
    duration_seconds: int
    ramp_step: int
    burst_multiplier: int
    burst_interval: int


# ── Request Definitions ──────────────────────────────────────

REQUESTS = [
    # User endpoints
    {"method": "GET", "path": "/api/users", "weight": 10},
    {"method": "GET", "path": "/api/users/1", "weight": 8},
    {"method": "GET", "path": "/api/users/2", "weight": 5},
    {"method": "POST", "path": "/api/users", "weight": 2, "body": {
        "username": "loadtest_{rand}",
        "email": "loadtest_{rand}@example.com",
        "password": "testpass123",
    }},
    # Product endpoints
    {"method": "GET", "path": "/api/products", "weight": 15},
    {"method": "GET", "path": "/api/products/1", "weight": 10},
    {"method": "GET", "path": "/api/products/2", "weight": 8},
    {"method": "GET", "path": "/api/products?category=electronics", "weight": 5},
    # Order endpoints
    {"method": "GET", "path": "/api/orders/user/1", "weight": 5},
    {"method": "POST", "path": "/api/orders", "weight": 3, "body": {
        "user_id": 1,
        "product_id": 1,
        "quantity": 1,
    }},
    {"method": "POST", "path": "/api/orders", "weight": 2, "body": {
        "user_id": 2,
        "product_id": 3,
        "quantity": 2,
    }},
    # Payment and notification lookups
    {"method": "GET", "path": "/api/payments/health", "weight": 3},
    {"method": "GET", "path": "/api/notifications/health", "weight": 3},
]


def weighted_choice(requests: list[dict]) -> dict:
    """Pick a random request weighted by its frequency weight."""
    total = sum(r["weight"] for r in requests)
    rand = random.uniform(0, total)
    cumulative = 0
    for req in requests:
        cumulative += req["weight"]
        if rand <= cumulative:
            return req
    return requests[-1]


# ── Load Patterns ────────────────────────────────────────────

async def steady_load(client: httpx.AsyncClient, config: LoadConfig, stats: dict):
    """Constant RPS for the entire duration."""
    interval = 1.0 / config.target_rps
    end_time = time.time() + config.duration_seconds

    while time.time() < end_time:
        asyncio.create_task(send_request(client, config, stats))
        await asyncio.sleep(interval)


async def ramp_load(client: httpx.AsyncClient, config: LoadConfig, stats: dict):
    """Gradually increase RPS from 1 to target_rps over the duration."""
    step_duration = config.duration_seconds / max(config.target_rps, 1)
    current_rps = 1
    end_time = time.time() + config.duration_seconds

    while time.time() < end_time and current_rps <= config.target_rps:
        interval = 1.0 / current_rps
        step_end = time.time() + step_duration
        while time.time() < step_end and time.time() < end_time:
            asyncio.create_task(send_request(client, config, stats))
            await asyncio.sleep(interval)
        current_rps += config.ramp_step
        print(f"  [ramp] RPS: {current_rps}")


async def burst_load(client: httpx.AsyncClient, config: LoadConfig, stats: dict):
    """Steady RPS with periodic bursts at higher multiplier."""
    interval = 1.0 / config.target_rps
    burst_interval_s = config.burst_interval
    burst_duration_s = 5
    end_time = time.time() + config.duration_seconds
    last_burst = time.time()

    while time.time() < end_time:
        now = time.time()
        if now - last_burst >= burst_interval_s:
            # Burst phase
            print(f"  [burst] {config.target_rps * config.burst_multiplier} RPS for 5s")
            burst_end = now + burst_duration_s
            burst_interval = 1.0 / (config.target_rps * config.burst_multiplier)
            while time.time() < burst_end:
                asyncio.create_task(send_request(client, config, stats))
                await asyncio.sleep(burst_interval)
            last_burst = time.time()
        else:
            asyncio.create_task(send_request(client, config, stats))
            await asyncio.sleep(interval)


# ── Request Execution ────────────────────────────────────────

async def send_request(client: httpx.AsyncClient, config: LoadConfig, stats: dict):
    """Send a single HTTP request to the gateway."""
    req = weighted_choice(REQUESTS)
    url = f"{config.gateway_url}{req['path']}"
    method = req["method"]

    body = None
    if "body" in req:
        body = dict(req["body"])
        # Replace {rand} placeholders with random values
        for key, val in body.items():
            if isinstance(val, str) and "{rand}" in val:
                body[key] = val.replace("{rand}", str(random.randint(10000, 99999)))

    try:
        start = time.perf_counter()
        if method == "GET":
            resp = await client.get(url, timeout=10.0)
        else:
            resp = await client.post(url, json=body, timeout=10.0)
        duration = time.perf_counter() - start

        stats["total"] += 1
        stats["latencies"].append(duration)

        if resp.status_code >= 500:
            stats["errors_5xx"] += 1
        elif resp.status_code >= 400:
            stats["errors_4xx"] += 1
        else:
            stats["success"] += 1

    except (httpx.TimeoutException, httpx.ConnectError) as e:
        stats["total"] += 1
        stats["connection_errors"] += 1


# ── Stats Reporter ───────────────────────────────────────────

async def report_stats(stats: dict, interval: int = 10):
    """Print stats every N seconds."""
    while True:
        await asyncio.sleep(interval)
        total = stats["total"]
        if total == 0:
            continue

        latencies = stats["latencies"][-1000:]  # Last 1000 for percentiles
        p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
        p99 = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0

        print(
            f"  requests={total} "
            f"success={stats['success']} "
            f"4xx={stats['errors_4xx']} "
            f"5xx={stats['errors_5xx']} "
            f"conn_err={stats['connection_errors']} "
            f"p50={p50:.3f}s p99={p99:.3f}s"
        )


# ── Main ─────────────────────────────────────────────────────

async def run(config: LoadConfig):
    stats = {
        "total": 0,
        "success": 0,
        "errors_4xx": 0,
        "errors_5xx": 0,
        "connection_errors": 0,
        "latencies": [],
    }

    print(f"SKAM Load Generator")
    print(f"  target:   {config.gateway_url}")
    print(f"  rps:      {config.target_rps}")
    print(f"  pattern:  {config.pattern}")
    print(f"  duration: {config.duration_seconds}s")
    print()

    async with httpx.AsyncClient() as client:
        reporter = asyncio.create_task(report_stats(stats))

        patterns = {
            "steady": steady_load,
            "ramp": ramp_load,
            "burst": burst_load,
        }
        load_fn = patterns[config.pattern]

        try:
            await load_fn(client, config, stats)
        except KeyboardInterrupt:
            pass
        finally:
            reporter.cancel()

    # Final summary
    total = stats["total"]
    print(f"\n{'='*50}")
    print(f"Load test complete")
    print(f"  Total requests:     {total}")
    print(f"  Successful:         {stats['success']}")
    print(f"  Client errors (4xx):{stats['errors_4xx']}")
    print(f"  Server errors (5xx):{stats['errors_5xx']}")
    print(f"  Connection errors:  {stats['connection_errors']}")
    if stats["latencies"]:
        lats = sorted(stats["latencies"])
        print(f"  p50 latency:        {lats[len(lats)//2]:.3f}s")
        print(f"  p99 latency:        {lats[int(len(lats)*0.99)]:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="SKAM Load Generator")
    parser.add_argument("--gateway", default="http://localhost:30080", help="API gateway URL")
    parser.add_argument("--rps", type=int, default=20, help="Target requests per second")
    parser.add_argument("--pattern", choices=["steady", "ramp", "burst"], default="steady")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--ramp-step", type=int, default=5, help="RPS increase per step (ramp mode)")
    parser.add_argument("--burst-multiplier", type=int, default=3, help="Burst RPS multiplier")
    parser.add_argument("--burst-interval", type=int, default=30, help="Seconds between bursts")
    args = parser.parse_args()

    config = LoadConfig(
        gateway_url=args.gateway,
        target_rps=args.rps,
        pattern=args.pattern,
        duration_seconds=args.duration,
        ramp_step=args.ramp_step,
        burst_multiplier=args.burst_multiplier,
        burst_interval=args.burst_interval,
    )

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
