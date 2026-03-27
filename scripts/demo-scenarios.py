#!/usr/bin/env python3
"""
SKAM Demo Scenario Runner

Runs all 5 chaos scenarios sequentially, demonstrating the full
inject → detect → decide → recover closed loop.

Usage:
    python scripts/demo-scenarios.py
    python scripts/demo-scenarios.py --scenario 1
    python scripts/demo-scenarios.py --chaos-url http://localhost:30090
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass

import httpx


@dataclass
class Scenario:
    number: int
    name: str
    description: str
    experiment: dict
    expected_detection: str
    expected_recovery: str
    wait_for_detection_s: int
    wait_for_recovery_s: int


SCENARIOS = [
    Scenario(
        number=1,
        name="Pod Kill Recovery",
        description="Kill an order-service pod and watch autonomous recovery",
        experiment={
            "name": "scenario-1-pod-kill",
            "target": {"namespace": "default", "label_selector": "app=order-service"},
            "fault_type": "pod_kill",
            "parameters": {"count": 1},
            "duration_seconds": 60,
        },
        expected_detection="Error rate spike + availability drop",
        expected_recovery="Pod auto-restart via Kubernetes API",
        wait_for_detection_s=30,
        wait_for_recovery_s=60,
    ),
    Scenario(
        number=2,
        name="Memory Pressure",
        description="Set extremely low memory limits on order-service causing OOMKills",
        experiment={
            "name": "scenario-2-memory-pressure",
            "target": {"namespace": "default", "label_selector": "app=order-service"},
            "fault_type": "memory_pressure",
            "parameters": {"limit_mb": 64},
            "duration_seconds": 120,
        },
        expected_detection="Resource anomaly + pod restart events",
        expected_recovery="Increase memory limits and restart deployment",
        wait_for_detection_s=45,
        wait_for_recovery_s=90,
    ),
    Scenario(
        number=3,
        name="Network Partition",
        description="Block all traffic to payment-service via NetworkPolicy",
        experiment={
            "name": "scenario-3-network-partition",
            "target": {"namespace": "default", "label_selector": "app=payment-service"},
            "fault_type": "network_partition",
            "parameters": {},
            "duration_seconds": 90,
        },
        expected_detection="Payment timeouts → order failures → error rate spike",
        expected_recovery="Remove blocking NetworkPolicy, traffic resumes",
        wait_for_detection_s=30,
        wait_for_recovery_s=60,
    ),
    Scenario(
        number=4,
        name="Cascading Failure via Latency",
        description="Inject 500ms latency into product-service, causing upstream cascade",
        experiment={
            "name": "scenario-4-latency-cascade",
            "target": {"namespace": "default", "label_selector": "app=product-service"},
            "fault_type": "latency_injection",
            "parameters": {"delay_ms": 500, "jitter_ms": 100},
            "duration_seconds": 120,
        },
        expected_detection="Latency anomaly propagating through API gateway",
        expected_recovery="HPA scale-up on product-service (2→4 replicas)",
        wait_for_detection_s=45,
        wait_for_recovery_s=120,
    ),
    Scenario(
        number=5,
        name="Cache Failure",
        description="Kill the Redis pod, disrupting order and notification services",
        experiment={
            "name": "scenario-5-cache-failure",
            "target": {"namespace": "default", "label_selector": "app=redis"},
            "fault_type": "pod_kill",
            "parameters": {"count": 1},
            "duration_seconds": 90,
        },
        expected_detection="Connection errors in order-service and notification-service",
        expected_recovery="Restart Redis pod and trigger cache warm-up",
        wait_for_detection_s=20,
        wait_for_recovery_s=60,
    ),
]


def print_header(text: str):
    width = 60
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}")


def print_phase(phase: str, description: str):
    print(f"\n  [{phase}] {description}")


async def check_system_health(healing_url: str) -> dict:
    """Check the decision engine's view of system health."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{healing_url}/status", timeout=5.0)
            return resp.json()
        except Exception:
            return {"error": "decision engine unreachable"}


async def inject_fault(chaos_url: str, experiment: dict) -> dict:
    """Send an experiment to the chaos engine."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{chaos_url}/experiments",
            json=experiment,
            timeout=10.0,
        )
        return resp.json()


async def get_experiment_status(chaos_url: str, experiment_id: str) -> dict:
    """Get the status of a running experiment."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{chaos_url}/experiments/{experiment_id}",
            timeout=5.0,
        )
        return resp.json()


async def get_recovery_actions(healing_url: str) -> list:
    """Get recent recovery actions from the decision engine."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{healing_url}/actions", timeout=5.0)
            return resp.json()
        except Exception:
            return []


async def wait_with_progress(seconds: int, label: str):
    """Wait with a progress indicator."""
    for i in range(seconds):
        remaining = seconds - i
        print(f"\r  Waiting... {remaining}s remaining  ", end="", flush=True)
        await asyncio.sleep(1)
    print(f"\r  {label} wait complete.           ")


async def run_scenario(scenario: Scenario, chaos_url: str, healing_url: str):
    """Execute a single chaos scenario."""
    print_header(f"Scenario {scenario.number}: {scenario.name}")
    print(f"  {scenario.description}")

    # Phase 1: Pre-check
    print_phase("PRE-CHECK", "Verifying system health before injection")
    health = await check_system_health(healing_url)
    print(f"  System status: {json.dumps(health, indent=2, default=str)[:200]}")

    # Phase 2: Inject
    print_phase("INJECT", f"Injecting fault: {scenario.experiment['fault_type']}")
    start_time = time.time()
    try:
        result = await inject_fault(chaos_url, scenario.experiment)
        experiment_id = result.get("id", "unknown")
        print(f"  Experiment started: {experiment_id}")
        print(f"  Target: {scenario.experiment['target']['label_selector']}")
    except Exception as e:
        print(f"  ERROR: Failed to inject fault: {e}")
        print(f"  (Is the chaos engine running at {chaos_url}?)")
        return False

    # Phase 3: Wait for detection
    print_phase("DETECT", f"Expected: {scenario.expected_detection}")
    await wait_with_progress(scenario.wait_for_detection_s, "Detection")

    # Phase 4: Wait for recovery
    print_phase("RECOVER", f"Expected: {scenario.expected_recovery}")
    await wait_with_progress(scenario.wait_for_recovery_s, "Recovery")

    # Phase 5: Verify
    print_phase("VERIFY", "Checking recovery outcome")
    elapsed = time.time() - start_time
    actions = await get_recovery_actions(healing_url)
    recent = [a for a in actions if a.get("target_service") in scenario.experiment["target"]["label_selector"]]

    if recent:
        print(f"  Recovery actions taken: {len(recent)}")
        for action in recent[-3:]:
            print(f"    - {action.get('action_type', '?')}: {action.get('status', '?')}")
    else:
        print(f"  No recovery actions recorded (may not be connected)")

    health = await check_system_health(healing_url)
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"  Final health: {json.dumps(health, indent=2, default=str)[:200]}")

    return True


async def run_all_scenarios(chaos_url: str, healing_url: str, specific: int | None = None):
    """Run all scenarios sequentially."""
    print_header("SKAM Chaos Engineering Demo")
    print(f"  Chaos Engine:    {chaos_url}")
    print(f"  Decision Engine: {healing_url}")
    print(f"  Scenarios:       {'All 5' if specific is None else f'#{specific}'}")

    scenarios = SCENARIOS
    if specific is not None:
        scenarios = [s for s in SCENARIOS if s.number == specific]
        if not scenarios:
            print(f"  ERROR: Scenario #{specific} not found")
            return

    for i, scenario in enumerate(scenarios):
        success = await run_scenario(scenario, chaos_url, healing_url)

        if i < len(scenarios) - 1:
            print(f"\n  Pausing 30s before next scenario...")
            await asyncio.sleep(30)

    print_header("Demo Complete")
    print("  All scenarios executed. Check the dashboard for full timeline.")


def main():
    parser = argparse.ArgumentParser(description="SKAM Demo Scenario Runner")
    parser.add_argument("--chaos-url", default="http://localhost:30090")
    parser.add_argument("--healing-url", default="http://localhost:30092")
    parser.add_argument("--scenario", type=int, default=None, help="Run specific scenario (1-5)")
    args = parser.parse_args()

    asyncio.run(run_all_scenarios(args.chaos_url, args.healing_url, args.scenario))


if __name__ == "__main__":
    main()
