#!/usr/bin/env python3
"""
Load generator for SKAM microservices.
Sends realistic HTTP traffic to the API gateway so the anomaly detector
has non-zero metrics to work with.

Usage:
  python load_generator.py                          # defaults
  python load_generator.py --rps 50 --duration 300  # 50 req/s for 5 min
  python load_generator.py --burst                  # spike pattern
"""

import asyncio
import argparse
import random
import time
from dataclasses import dataclass

import aiohttp

GATEWAY = "http://localhost:8080"

ENDPOINTS = [
    ("GET",  "/api/users/usr-001"),
    ("GET",  "/api/products"),
    ("GET",  "/api/products/prod-001"),
    ("POST", "/api/orders"),
    ("GET",  "/api/orders/ord-001"),
    ("POST", "/api/cart/items"),
    ("GET",  "/api/cart/usr-001"),
    ("POST", "/api/payments/process"),
]

ORDER_BODY = {
    "user_id": "usr-001",
    "items": [{"product_id": "prod-001", "quantity": 1}],
}

CART_BODY = {"user_id": "usr-001", "product_id": "prod-001", "quantity": 1}
PAYMENT_BODY = {"order_id": "ord-001", "amount": 29.99, "method": "card"}


@dataclass
class Stats:
    total: int = 0
    ok: int = 0
    err: int = 0
    lat_sum: float = 0.0

    @property
    def avg_lat(self):
        return (self.lat_sum / self.total * 1000) if self.total else 0

    def line(self):
        return f"  reqs={self.total}  ok={self.ok}  err={self.err}  avg={self.avg_lat:.0f}ms"


async def send_request(session, method, path, stats):
    url = f"{GATEWAY}{path}"
    body = None
    if "orders" in path and method == "POST":
        body = ORDER_BODY
    elif "cart" in path and method == "POST":
        body = CART_BODY
    elif "payments" in path and method == "POST":
        body = PAYMENT_BODY

    t0 = time.monotonic()
    try:
        kwargs = {"json": body} if body else {}
        async with session.request(method, url, **kwargs, timeout=aiohttp.ClientTimeout(total=5)) as r:
            stats.total += 1
            stats.lat_sum += time.monotonic() - t0
            if r.status < 500:
                stats.ok += 1
            else:
                stats.err += 1
    except Exception:
        stats.total += 1
        stats.err += 1
        stats.lat_sum += time.monotonic() - t0


async def steady_load(rps, duration, gateway):
    global GATEWAY
    GATEWAY = gateway
    stats = Stats()
    delay = 1.0 / rps
    end_at = time.monotonic() + duration

    print(f"[load] steady {rps} req/s for {duration}s -> {gateway}")

    async with aiohttp.ClientSession() as session:
        while time.monotonic() < end_at:
            method, path = random.choice(ENDPOINTS)
            asyncio.create_task(send_request(session, method, path, stats))
            await asyncio.sleep(delay + random.uniform(-delay * 0.2, delay * 0.2))

            elapsed = time.monotonic() - (end_at - duration)
            if stats.total % (rps * 5) == 0 and stats.total > 0:
                print(f"  [{elapsed:.0f}s]{stats.line()}")

        await asyncio.sleep(1)  # drain in-flight

    print(f"[load] done.{stats.line()}")
    return stats


async def burst_load(gateway, cycles=5, burst_rps=100, calm_rps=10, burst_dur=15, calm_dur=30):
    global GATEWAY
    GATEWAY = gateway
    stats = Stats()
    print(f"[load] burst pattern: {cycles} cycles of {burst_rps}rps/{burst_dur}s + {calm_rps}rps/{calm_dur}s")

    async with aiohttp.ClientSession() as session:
        for c in range(cycles):
            # Burst phase
            print(f"  cycle {c+1}/{cycles}: BURST {burst_rps} rps")
            end_at = time.monotonic() + burst_dur
            delay = 1.0 / burst_rps
            while time.monotonic() < end_at:
                method, path = random.choice(ENDPOINTS)
                asyncio.create_task(send_request(session, method, path, stats))
                await asyncio.sleep(delay)

            # Calm phase
            print(f"  cycle {c+1}/{cycles}: calm {calm_rps} rps")
            end_at = time.monotonic() + calm_dur
            delay = 1.0 / calm_rps
            while time.monotonic() < end_at:
                method, path = random.choice(ENDPOINTS)
                asyncio.create_task(send_request(session, method, path, stats))
                await asyncio.sleep(delay)

        await asyncio.sleep(1)

    print(f"[load] done.{stats.line()}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="SKAM load generator")
    parser.add_argument("--gateway", default=GATEWAY, help="API gateway URL")
    parser.add_argument("--rps", type=int, default=20, help="Requests per second")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--burst", action="store_true", help="Use burst pattern instead of steady")
    args = parser.parse_args()

    if args.burst:
        asyncio.run(burst_load(args.gateway))
    else:
        asyncio.run(steady_load(args.rps, args.duration, args.gateway))


if __name__ == "__main__":
    main()
