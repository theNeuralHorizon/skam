"""FastAPI router for chaos experiment management."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, status

from app.faults import FaultInjector
from app.metrics import (
    chaos_active_experiments,
    chaos_experiment_duration_seconds,
    chaos_experiments_total,
    chaos_faults_injected_total,
)
from app.models import ExperimentConfig, ExperimentStatus, ScenarioConfig

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])
scenario_router = APIRouter(prefix="/scenarios", tags=["scenarios"])

# ---------------------------------------------------------------------------
# In-memory experiment store and manager reference
# ---------------------------------------------------------------------------

_experiments: dict[str, ExperimentStatus] = {}
_tasks: dict[str, asyncio.Task] = {}
_injector: FaultInjector | None = None


def set_injector(injector: FaultInjector) -> None:
    """Called once at application startup to wire the K8s fault injector."""
    global _injector  # noqa: PLW0603
    _injector = injector


def _get_injector() -> FaultInjector:
    if _injector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FaultInjector not initialised -- Kubernetes client may have failed to load",
        )
    return _injector


# ---------------------------------------------------------------------------
# Background experiment runner
# ---------------------------------------------------------------------------


async def _run_experiment(experiment_id: str, config: ExperimentConfig) -> None:
    """Execute a single experiment lifecycle: inject -> wait -> rollback."""
    exp = _experiments[experiment_id]
    injector = _get_injector()
    fault_method = getattr(injector, config.fault_type, None)

    if fault_method is None:
        exp.status = "failed"
        exp.error_message = f"Unknown fault type: {config.fault_type}"
        exp.ended_at = datetime.now(timezone.utc)
        chaos_active_experiments.dec()
        chaos_experiments_total.labels(fault_type=config.fault_type, status="failed").inc()
        return

    rollback_info: dict[str, Any] | None = None
    start_ts = time.monotonic()

    try:
        logger.info(
            "experiment.inject.start",
            experiment_id=experiment_id,
            fault_type=config.fault_type,
            target=config.target.model_dump(),
        )

        rollback_info = await fault_method(config.target, config.parameters)
        exp.recovery_info = rollback_info
        chaos_faults_injected_total.labels(fault_type=config.fault_type).inc()

        logger.info(
            "experiment.inject.complete",
            experiment_id=experiment_id,
            rollback_info=rollback_info,
        )

        # Wait for the requested duration (or until cancelled).
        await asyncio.sleep(config.duration_seconds)

        # Auto-rollback after duration elapses.
        await injector.rollback(config.fault_type, rollback_info)

        elapsed = time.monotonic() - start_ts
        exp.status = "completed"
        exp.ended_at = datetime.now(timezone.utc)
        chaos_experiments_total.labels(fault_type=config.fault_type, status="completed").inc()
        chaos_experiment_duration_seconds.labels(fault_type=config.fault_type).observe(elapsed)

        logger.info("experiment.completed", experiment_id=experiment_id, elapsed_s=round(elapsed, 2))

    except asyncio.CancelledError:
        # Explicit stop -- attempt rollback.
        logger.info("experiment.cancelled", experiment_id=experiment_id)
        if rollback_info:
            try:
                await injector.rollback(config.fault_type, rollback_info)
                exp.status = "rolled_back"
            except Exception:
                logger.exception("experiment.rollback_on_cancel_failed", experiment_id=experiment_id)
                exp.status = "failed"
                exp.error_message = "Rollback after cancellation failed"
        else:
            exp.status = "rolled_back"
        elapsed = time.monotonic() - start_ts
        exp.ended_at = datetime.now(timezone.utc)
        chaos_experiments_total.labels(fault_type=config.fault_type, status=exp.status).inc()
        chaos_experiment_duration_seconds.labels(fault_type=config.fault_type).observe(elapsed)

    except Exception as exc:
        logger.exception("experiment.failed", experiment_id=experiment_id)
        # Best-effort rollback.
        if rollback_info:
            try:
                await injector.rollback(config.fault_type, rollback_info)
            except Exception:
                logger.exception("experiment.rollback_on_failure_failed", experiment_id=experiment_id)
        elapsed = time.monotonic() - start_ts
        exp.status = "failed"
        exp.error_message = str(exc)
        exp.ended_at = datetime.now(timezone.utc)
        chaos_experiments_total.labels(fault_type=config.fault_type, status="failed").inc()
        chaos_experiment_duration_seconds.labels(fault_type=config.fault_type).observe(elapsed)

    finally:
        chaos_active_experiments.dec()
        _tasks.pop(experiment_id, None)


# ---------------------------------------------------------------------------
# Experiment endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=ExperimentStatus, status_code=status.HTTP_201_CREATED)
async def create_experiment(config: ExperimentConfig) -> ExperimentStatus:
    """Create and immediately start a new chaos experiment."""
    experiment_id = str(uuid.uuid4())

    exp = ExperimentStatus(
        id=experiment_id,
        name=config.name,
        fault_type=config.fault_type,
        status="running",
        started_at=datetime.now(timezone.utc),
        target=config.target,
    )
    _experiments[experiment_id] = exp
    chaos_active_experiments.inc()

    task = asyncio.create_task(_run_experiment(experiment_id, config))
    _tasks[experiment_id] = task

    logger.info("experiment.created", experiment_id=experiment_id, name=config.name)
    return exp


@router.get("", response_model=list[ExperimentStatus])
async def list_experiments() -> list[ExperimentStatus]:
    """Return all experiments (active and historical)."""
    return list(_experiments.values())


@router.get("/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment(experiment_id: str) -> ExperimentStatus:
    """Return the status of a single experiment."""
    exp = _experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    return exp


@router.post("/{experiment_id}/stop", response_model=ExperimentStatus)
async def stop_experiment(experiment_id: str) -> ExperimentStatus:
    """Stop a running experiment and trigger rollback."""
    exp = _experiments.get(experiment_id)
    if exp is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    task = _tasks.get(experiment_id)
    if task is None or task.done():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment {experiment_id} is not currently running",
        )

    logger.info("experiment.stop_requested", experiment_id=experiment_id)
    task.cancel()

    # Give the task a moment to process the cancellation.
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass

    return _experiments[experiment_id]


# ---------------------------------------------------------------------------
# Scenario endpoints
# ---------------------------------------------------------------------------


@scenario_router.post("", response_model=list[ExperimentStatus], status_code=status.HTTP_201_CREATED)
async def run_scenario(scenario: ScenarioConfig) -> list[ExperimentStatus]:
    """Run a multi-step chaos scenario.

    Each step is executed sequentially with a configurable delay between steps.
    If any step fails the remaining steps are skipped.
    """
    logger.info("scenario.start", name=scenario.name, steps=len(scenario.steps))
    results: list[ExperimentStatus] = []

    for idx, step in enumerate(scenario.steps):
        experiment_id = str(uuid.uuid4())
        exp = ExperimentStatus(
            id=experiment_id,
            name=f"{scenario.name}/step-{idx}:{step.name}",
            fault_type=step.fault_type,
            status="running",
            started_at=datetime.now(timezone.utc),
            target=step.target,
        )
        _experiments[experiment_id] = exp
        chaos_active_experiments.inc()

        task = asyncio.create_task(_run_experiment(experiment_id, step))
        _tasks[experiment_id] = task

        # Wait for this step to finish before moving to the next.
        try:
            await task
        except asyncio.CancelledError:
            pass

        results.append(_experiments[experiment_id])

        if _experiments[experiment_id].status == "failed":
            logger.warning(
                "scenario.step_failed",
                scenario=scenario.name,
                step=idx,
                experiment_id=experiment_id,
            )
            break

        if idx < len(scenario.steps) - 1 and scenario.delay_between_steps > 0:
            logger.info("scenario.delay", seconds=scenario.delay_between_steps)
            await asyncio.sleep(scenario.delay_between_steps)

    logger.info("scenario.complete", name=scenario.name, executed=len(results))
    return results
