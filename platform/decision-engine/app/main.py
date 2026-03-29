"""Self-Healing Decision Engine — FastAPI application entry point."""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, JSONResponse

from app.events import EventManager
from app.executor import RecoveryExecutor
from app.metrics import (
    active_recoveries,
    alerts_received_total,
    get_metrics,
    recovery_actions_total,
    recovery_duration_seconds,
)
from app.models import AnomalyAlert, PredictionAlert, RecoveryAction, SystemStatus
from app.policies import PolicyEngine
from app.validator import RecoveryValidator

# ---------------------------------------------------------------------------
# Structured logging setup
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("LOG_FORMAT") == "console"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_config().get("min_level", 0)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("decision-engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server.monitoring:9090")
ANOMALY_DETECTOR_URL = os.getenv("ANOMALY_DETECTOR_URL", "http://anomaly-detector:8091")
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
NAMESPACE = os.getenv("NAMESPACE", "skam-platform")

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------
policy_engine: PolicyEngine | None = None
executor: RecoveryExecutor | None = None
validator: RecoveryValidator | None = None
event_manager: EventManager = EventManager()

# Bounded history of recent actions (latest 200)
recent_actions: deque[RecoveryAction] = deque(maxlen=200)
# Bounded history of recent predictions (latest 100)
recent_predictions: deque[dict] = deque(maxlen=100)
# Set of services currently being healed (prevents duplicate concurrent healing)
active_action_ids: set[str] = set()
_state_lock = asyncio.Lock()
_recovery_tasks: set[asyncio.Task] = set()

_health_task: asyncio.Task | None = None


# ---------------------------------------------------------------------------
# Background: periodic cluster health check
# ---------------------------------------------------------------------------
async def _health_monitor_loop() -> None:
    """Periodically query Prometheus for cluster-wide anomalies."""
    while True:
        try:
            await _check_cluster_health()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("health_monitor_error")
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


async def _check_cluster_health() -> None:
    """Independently check cluster health via Prometheus and synthesise alerts."""
    queries = {
        "pod_restarts": (
            "sum by (pod) (increase(kube_pod_container_status_restarts_total"
            f'{{namespace="{NAMESPACE}"}}[5m])) > 3'
        ),
        "high_cpu": (
            "sum by (pod) (rate(container_cpu_usage_seconds_total"
            f'{{namespace="{NAMESPACE}"}}[5m])) > 0.9'
        ),
    }

    async with httpx.AsyncClient(timeout=10) as http:
        for check_name, query in queries.items():
            try:
                resp = await http.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query},
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                results = data.get("data", {}).get("result", [])
                if results:
                    logger.warning(
                        "health_check_anomaly",
                        check=check_name,
                        matches=len(results),
                    )
            except httpx.RequestError:
                logger.debug("prometheus_unreachable_during_health_check")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    global policy_engine, executor, validator, _health_task

    logger.info("startup", prometheus=PROMETHEUS_URL, namespace=NAMESPACE)

    policy_engine = PolicyEngine()
    executor = RecoveryExecutor()
    validator = RecoveryValidator()

    _health_task = asyncio.create_task(_health_monitor_loop())
    logger.info("health_monitor_started", interval=HEALTH_CHECK_INTERVAL)

    yield

    # Shutdown
    if _health_task:
        _health_task.cancel()
        try:
            await _health_task
        except asyncio.CancelledError:
            pass

    # Wait for in-flight recovery actions to complete (up to 10s)
    if _recovery_tasks:
        logger.info("shutdown.waiting_for_recoveries", count=len(_recovery_tasks))
        done, pending = await asyncio.wait(_recovery_tasks, timeout=10.0)
        if pending:
            logger.warning("shutdown.cancelling_stale_recoveries", count=len(pending))
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    logger.info("shutdown_complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Self-Healing Decision Engine",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/ready")
async def ready():
    if policy_engine is None or executor is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "components not initialised"},
        )
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        content=get_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/status")
async def status() -> SystemStatus:
    policies = policy_engine.get_policies() if policy_engine else []
    active = [a for a in recent_actions if a.status in ("pending", "executing", "validating")]
    unhealthy = list({a.target_service for a in active})
    healthy_set = {a.target_service for a in recent_actions if a.status == "success"} - set(unhealthy)

    return SystemStatus(
        healthy_services=sorted(healthy_set),
        unhealthy_services=sorted(unhealthy),
        active_recoveries=len(active_action_ids),
        recent_actions=list(recent_actions)[-20:],
        policies=policies,
    )


@app.get("/actions")
async def list_actions(limit: int = 50):
    return list(recent_actions)[-limit:]


@app.get("/policies")
async def list_policies():
    if policy_engine is None:
        return []
    return policy_engine.get_policies()


# ---------------------------------------------------------------------------
# Alert ingestion
# ---------------------------------------------------------------------------

@app.post("/alerts")
async def receive_alert(alert: AnomalyAlert):
    """Receive an anomaly alert, evaluate policies, and execute recovery."""
    logger.info(
        "alert_received",
        service=alert.service,
        anomaly_type=alert.anomaly_type,
        severity=alert.severity,
        combined_score=alert.combined_score,
    )
    alerts_received_total.labels(anomaly_type=alert.anomaly_type).inc()

    # Broadcast to WebSocket clients
    await event_manager.emit_anomaly_detected(alert.model_dump(mode="json"))

    if policy_engine is None:
        return {"status": "rejected", "reason": "engine not initialised"}

    action = policy_engine.evaluate(alert)
    if action is None:
        return {"status": "no_action", "reason": "no matching policy or cooldown active"}

    # Track the recovery task for graceful shutdown
    task = asyncio.create_task(_execute_recovery(action))
    _recovery_tasks.add(task)
    task.add_done_callback(_recovery_tasks.discard)

    return {
        "status": "accepted",
        "action_id": action.id,
        "action_type": action.action_type,
        "target_service": action.target_service,
    }


# ---------------------------------------------------------------------------
# Prediction alert ingestion
# ---------------------------------------------------------------------------

@app.post("/prediction-alerts")
async def receive_prediction_alert(alert: PredictionAlert):
    """Receive a prediction alert, evaluate preemptive policies, and optionally act."""
    logger.info(
        "prediction_alert_received",
        service=alert.service,
        prediction_type=alert.prediction_type,
        confidence=alert.confidence,
        time_to_event=alert.time_to_event_seconds,
    )

    # Store and broadcast
    recent_predictions.append(alert.model_dump(mode="json"))
    await event_manager.emit_prediction_raised(alert.model_dump(mode="json"))

    if policy_engine is None:
        return {"status": "rejected", "reason": "engine not initialised"}

    action = policy_engine.evaluate_prediction(alert)
    if action is None:
        return {
            "status": "prediction_noted",
            "reason": "no preemptive action triggered (cooldown or below confidence threshold)",
        }

    # Fire-and-forget the preemptive recovery pipeline
    asyncio.create_task(_execute_recovery(action))

    return {
        "status": "preemptive_action",
        "action_id": action.id,
        "action_type": action.action_type,
        "target_service": action.target_service,
        "prediction_type": alert.prediction_type,
    }


@app.get("/predictions")
async def list_predictions(limit: int = 50):
    """Return recent prediction alerts."""
    return list(recent_predictions)[-limit:]


@app.get("/decision/api/predictions")
async def list_predictions_dashboard(limit: int = 50):
    """Return predictions in dashboard-compatible format."""
    return {"predictions": list(recent_predictions)[-limit:]}


# ---------------------------------------------------------------------------
# Recovery pipeline
# ---------------------------------------------------------------------------

async def _execute_recovery(action: RecoveryAction) -> None:
    """Full recovery lifecycle: execute → validate → broadcast result."""
    log = logger.bind(
        action_id=action.id,
        action_type=action.action_type,
        service=action.target_service,
    )

    action.status = "executing"
    async with _state_lock:
        active_action_ids.add(action.id)
        recent_actions.append(action)
    active_recoveries.inc()
    start = time.monotonic()

    await event_manager.emit_recovery_started(action.model_dump(mode="json"))
    log.info("recovery_executing")

    try:
        result = await executor.execute(
            action_type=action.action_type,
            service=action.target_service,
            parameters=dict(action.parameters),
        )

        if not result.get("success"):
            action.status = "failed"
            action.completed_at = datetime.now(timezone.utc)
            action.validation_result = result
            log.error("recovery_execution_failed", result=result)
            recovery_actions_total.labels(
                action_type=action.action_type, status="failed"
            ).inc()
            await event_manager.emit_recovery_failed(action.model_dump(mode="json"))
            return

        # Validate
        action.status = "validating"
        log.info("recovery_validating")

        validation = await validator.validate(
            action_type=action.action_type,
            target_service=action.target_service,
            parameters=dict(action.parameters),
        )

        action.validation_result = validation
        action.completed_at = datetime.now(timezone.utc)

        if validation.get("success"):
            action.status = "success"
            log.info("recovery_success", validation=validation)
            recovery_actions_total.labels(
                action_type=action.action_type, status="success"
            ).inc()
            await event_manager.emit_recovery_completed(action.model_dump(mode="json"))
        else:
            action.status = "failed"
            log.warning("recovery_validation_failed", validation=validation)
            recovery_actions_total.labels(
                action_type=action.action_type, status="failed"
            ).inc()
            await event_manager.emit_recovery_failed(action.model_dump(mode="json"))

    except Exception:
        action.status = "failed"
        action.completed_at = datetime.now(timezone.utc)
        log.exception("recovery_unexpected_error")
        recovery_actions_total.labels(
            action_type=action.action_type, status="failed"
        ).inc()
        await event_manager.emit_recovery_failed(action.model_dump(mode="json"))

    finally:
        duration = time.monotonic() - start
        recovery_duration_seconds.labels(action_type=action.action_type).observe(duration)
        async with _state_lock:
            active_action_ids.discard(action.id)
        active_recoveries.dec()
        log.info("recovery_completed", duration_seconds=round(duration, 2), status=action.status)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await event_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await event_manager.disconnect(websocket)
