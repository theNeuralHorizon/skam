#!/usr/bin/env python3
"""Aggressive chaos experiments with longer durations for better ML training data.

Runs multiple rounds of fault injection with:
- Longer fault durations (3-5 minutes instead of 1-2)
- Higher intensity (more latency, multiple pods killed)
- Multiple services targeted per round
- 2-minute baseline gaps between faults
"""

import asyncio
import json
import sys
import time

import httpx

CHAOS_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30090"
HEALING_URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:30092"

SCENARIOS = [
    # Round 1: Sustained pod kills across multiple services
    {
        "name": "sustained-pod-kill-order",
        "fault_type": "pod_kill",
        "target": {"namespace": "default", "label_selector": "app=order-service"},
        "duration_seconds": 180,
        "parameters": {},
        "description": "Kill order-service pod (3 min recovery window)",
    },
    {
        "name": "sustained-pod-kill-user",
        "fault_type": "pod_kill",
        "target": {"namespace": "default", "label_selector": "app=user-service"},
        "duration_seconds": 180,
        "parameters": {},
        "description": "Kill user-service pod (3 min recovery window)",
    },
    # Round 2: Heavy latency injection
    {
        "name": "heavy-latency-product",
        "fault_type": "latency_injection",
        "target": {"namespace": "default", "label_selector": "app=product-service"},
        "duration_seconds": 240,
        "parameters": {"delay_ms": 1000, "jitter_ms": 200},
        "description": "Inject 1000ms latency into product-service (4 min)",
    },
    {
        "name": "heavy-latency-order",
        "fault_type": "latency_injection",
        "target": {"namespace": "default", "label_selector": "app=order-service"},
        "duration_seconds": 240,
        "parameters": {"delay_ms": 800, "jitter_ms": 150},
        "description": "Inject 800ms latency into order-service (4 min)",
    },
    # Round 3: Network partitions on critical services
    {
        "name": "netpart-payment-long",
        "fault_type": "network_partition",
        "target": {"namespace": "default", "label_selector": "app=payment-service"},
        "duration_seconds": 180,
        "parameters": {},
        "description": "Block payment-service traffic (3 min)",
    },
    {
        "name": "netpart-notification-long",
        "fault_type": "network_partition",
        "target": {"namespace": "default", "label_selector": "app=notification-service"},
        "duration_seconds": 180,
        "parameters": {},
        "description": "Block notification-service traffic (3 min)",
    },
    # Round 4: Memory pressure
    {
        "name": "memory-pressure-product",
        "fault_type": "memory_pressure",
        "target": {"namespace": "default", "label_selector": "app=product-service"},
        "duration_seconds": 240,
        "parameters": {"memory_limit": "48Mi"},
        "description": "Extreme memory limit on product-service (4 min, 48Mi)",
    },
    # Round 5: Cache failure + simultaneous pod kill
    {
        "name": "cache-kill-extended",
        "fault_type": "pod_kill",
        "target": {"namespace": "default", "label_selector": "app=redis"},
        "duration_seconds": 180,
        "parameters": {},
        "description": "Kill Redis pod (3 min, affects order + notification)",
    },
    # Round 6: API gateway stress
    {
        "name": "latency-gateway",
        "fault_type": "latency_injection",
        "target": {"namespace": "default", "label_selector": "app=api-gateway"},
        "duration_seconds": 300,
        "parameters": {"delay_ms": 500, "jitter_ms": 100},
        "description": "Inject 500ms latency into api-gateway (5 min)",
    },
    # Round 7: Repeat pod kills on different services
    {
        "name": "pod-kill-payment",
        "fault_type": "pod_kill",
        "target": {"namespace": "default", "label_selector": "app=payment-service"},
        "duration_seconds": 180,
        "parameters": {},
        "description": "Kill payment-service pod (3 min)",
    },
    {
        "name": "pod-kill-notification",
        "fault_type": "pod_kill",
        "target": {"namespace": "default", "label_selector": "app=notification-service"},
        "duration_seconds": 180,
        "parameters": {},
        "description": "Kill notification-service pod (3 min)",
    },
]

BASELINE_GAP = 120  # 2 minutes between faults


async def inject_fault(client, scenario):
    """Inject a single fault and wait for its duration."""
    body = {
        "name": scenario["name"],
        "fault_type": scenario["fault_type"],
        "target": scenario["target"],
        "duration_seconds": scenario["duration_seconds"],
        "parameters": scenario.get("parameters", {}),
    }
    resp = await client.post(f"{CHAOS_URL}/experiments", json=body)
    resp.raise_for_status()
    data = resp.json()
    return data.get("id", "unknown")


async def main():
    total_start = time.time()

    print("=" * 60)
    print("  AGGRESSIVE CHAOS - EXTENDED FAULT INJECTION")
    print("=" * 60)
    print(f"  Chaos Engine : {CHAOS_URL}")
    print(f"  Scenarios    : {len(SCENARIOS)}")
    print(f"  Baseline gap : {BASELINE_GAP}s between faults")

    estimated = sum(s["duration_seconds"] for s in SCENARIOS) + BASELINE_GAP * (len(SCENARIOS) - 1)
    print(f"  Est. duration: {estimated // 60}m {estimated % 60}s")
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, scenario in enumerate(SCENARIOS, 1):
            print(f"  [{i}/{len(SCENARIOS)}] {scenario['description']}")

            try:
                exp_id = await inject_fault(client, scenario)
                print(f"    Injected: {exp_id}")
            except Exception as e:
                print(f"    FAILED: {e}")
                continue

            # Wait for fault duration
            duration = scenario["duration_seconds"]
            print(f"    Waiting {duration}s for fault to run...", end="", flush=True)
            await asyncio.sleep(duration)
            print(" done")

            # Baseline gap (except after last)
            if i < len(SCENARIOS):
                print(f"    Baseline gap: {BASELINE_GAP}s...", end="", flush=True)
                await asyncio.sleep(BASELINE_GAP)
                print(" done")

            elapsed = time.time() - total_start
            print(f"    Elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s")
            print()

    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"  COMPLETE: {len(SCENARIOS)} faults in {int(total_elapsed // 60)}m {int(total_elapsed % 60)}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
