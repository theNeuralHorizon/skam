"""FastAPI application for the ML Anomaly Detection service."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

import httpx

from app.collector import MetricsCollector
from app.detector import AnomalyDetector
from app.metrics import detection_cycle_duration_seconds
from app.models import (
    AnomalyHistoryQuery,
    AnomalyResult,
    DetectorStatus,
    PredictionAlert,
    PredictionsResponse,
    ScoresResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if os.getenv("LOG_FORMAT") == "console" else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.getenv("LOG_LEVEL", "20"))  # default INFO
    ),
)

logger = structlog.get_logger("anomaly_detector")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

collector: MetricsCollector | None = None
detector: AnomalyDetector | None = None
_bg_task: asyncio.Task | None = None
_background_tasks: set[asyncio.Task] = set()
_ready: bool = False

DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL_SECONDS", "15"))
DECISION_ENGINE_URL = os.getenv("DECISION_ENGINE_URL", "http://decision-engine:8090")
_PREDICTION_CONFIDENCE_THRESHOLD = 0.6


def _track_task(coro, *, name: str = "background") -> asyncio.Task:
    """Create a tracked background task that logs failures and self-cleans."""
    task = asyncio.create_task(coro, name=name)

    def _on_done(t: asyncio.Task) -> None:
        _background_tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            logger.warning("background_task.failed", task_name=name, error=str(exc))

    task.add_done_callback(_on_done)
    _background_tasks.add(task)
    return task


# ---------------------------------------------------------------------------
# Background detection loop
# ---------------------------------------------------------------------------

async def _detection_loop() -> None:
    """Periodically collect metrics and run the detection pipeline."""
    global _ready

    assert collector is not None
    assert detector is not None

    logger.info("detection_loop.started", interval_seconds=DETECTION_INTERVAL)
    _ready = True

    while True:
        try:
            start = time.monotonic()
            all_metrics = await collector.collect_all_services()

            results: list[AnomalyResult] = []
            for svc_metrics in all_metrics:
                result = await detector.detect(svc_metrics)
                results.append(result)

            elapsed = time.monotonic() - start
            detection_cycle_duration_seconds.observe(elapsed)

            anomalies = [r for r in results if r.is_anomaly]

            # --- Prediction alerts ---
            predictions = detector.get_latest_predictions()
            high_confidence = [p for p in predictions if p.confidence >= _PREDICTION_CONFIDENCE_THRESHOLD]
            for pred in high_confidence:
                _track_task(_send_prediction_alert(pred), name=f"prediction_alert:{pred.service}")

            # --- Fetch recovery counts for repeat-failure algorithm ---
            _track_task(_fetch_recovery_counts(), name="fetch_recovery_counts")

            logger.info(
                "detection_loop.cycle_complete",
                services=len(all_metrics),
                anomalies=len(anomalies),
                predictions=len(high_confidence),
                duration_s=round(elapsed, 3),
            )

        except asyncio.CancelledError:
            logger.info("detection_loop.cancelled")
            raise
        except Exception:
            logger.error("detection_loop.error", exc_info=True)

        await asyncio.sleep(DETECTION_INTERVAL)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global collector, detector, _bg_task

    logger.info("startup.begin")

    collector = MetricsCollector()
    await collector.start()

    detector = AnomalyDetector()

    _bg_task = asyncio.create_task(_detection_loop())
    logger.info("startup.complete")

    yield

    logger.info("shutdown.begin")
    if _bg_task:
        _bg_task.cancel()
        try:
            await _bg_task
        except asyncio.CancelledError:
            pass

    # Cancel any remaining tracked background tasks
    for task in list(_background_tasks):
        task.cancel()
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)
    _background_tasks.clear()

    if collector:
        await collector.stop()

    logger.info("shutdown.complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SKAM Anomaly Detector",
    description="Two-stage ML anomaly detection (Isolation Forest + LSTM Autoencoder) for platform telemetry",
    version="1.0.0",
    lifespan=lifespan,
)


# -- health / readiness -----------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if not _ready:
        return PlainTextResponse("not ready", status_code=503)
    return {"status": "ready"}


# -- Prometheus metrics ------------------------------------------------------

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        generate_latest().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# -- anomaly endpoints -------------------------------------------------------

@app.get("/anomalies", response_model=list[AnomalyResult])
async def get_anomalies():
    """Return current (most recent) anomalies per service."""
    if detector is None:
        return []
    return detector.current_anomalies()


@app.get("/anomalies/history", response_model=list[AnomalyResult])
async def get_anomaly_history(
    service: str | None = Query(default=None, description="Filter by service name"),
    minutes: int = Query(default=60, ge=1, le=1440, description="Look-back window in minutes"),
    anomalies_only: bool = Query(default=False, description="Only return entries flagged as anomalies"),
):
    """Return historical anomaly results within the given time window."""
    if detector is None:
        return []
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return detector.history(service=service, since=since, anomalies_only=anomalies_only)


@app.get("/status", response_model=DetectorStatus)
async def get_status():
    """Return operational status of the anomaly detector."""
    if detector is None:
        return DetectorStatus()
    from app.collector import MONITORED_SERVICES

    return DetectorStatus(
        is_trained=detector.is_trained(),
        last_detection_at=detector.last_detection_at,
        services_monitored=MONITORED_SERVICES,
        anomalies_detected=detector.anomaly_count,
        model_version=detector.model_version,
    )


# -- dashboard-compatible endpoints ----------------------------------------

@app.get("/anomaly/api/scores")
async def get_scores():
    """Return per-ensemble scores for all services (dashboard format)."""
    if detector is None:
        return {"scores": [], "threshold": 0.7}
    scores = detector.get_latest_scores()
    return {"scores": [s.model_dump() for s in scores], "threshold": 0.7}


@app.get("/anomaly/api/status")
async def get_anomaly_api_status():
    """Return detector status in dashboard-compatible format."""
    if detector is None:
        return {"running": False, "services_monitored": 0, "total_anomalies": 0}
    from app.collector import MONITORED_SERVICES

    return {
        "running": True,
        "services_monitored": len(MONITORED_SERVICES),
        "total_anomalies": detector.anomaly_count,
    }


# -- prediction endpoints ---------------------------------------------------

@app.get("/predictions")
async def get_predictions():
    """Return current predictions from all algorithms."""
    if detector is None:
        return PredictionsResponse(predictions=[], generated_at=datetime.now(timezone.utc))
    preds = detector.get_latest_predictions()
    return PredictionsResponse(predictions=preds, generated_at=datetime.now(timezone.utc))


@app.get("/anomaly/api/predictions")
async def get_predictions_dashboard():
    """Return predictions in dashboard-compatible format."""
    if detector is None:
        return {"predictions": [], "generated_at": datetime.now(timezone.utc).isoformat()}
    preds = detector.get_latest_predictions()
    return {
        "predictions": [p.model_dump() for p in preds],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# -- prediction helpers ------------------------------------------------------

async def _send_prediction_alert(pred) -> None:
    """POST a high-confidence prediction to the decision engine (fire-and-forget)."""
    try:
        alert = PredictionAlert(
            service=pred.service,
            prediction_type=pred.prediction_type,
            predicted_event=pred.predicted_event,
            time_to_event_seconds=pred.time_to_event_seconds,
            confidence=pred.confidence,
            current_score=pred.current_value,
            recommended_action=pred.recommended_action,
            timestamp=datetime.now(timezone.utc),
            details=pred.details,
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{DECISION_ENGINE_URL}/prediction-alerts",
                json=alert.model_dump(mode="json"),
            )
            if resp.status_code != 200:
                logger.warning("prediction_alert.send_failed", status=resp.status_code)
    except Exception:
        logger.warning("prediction_alert.send_error", exc_info=True)


async def _fetch_recovery_counts() -> None:
    """Query the decision engine for recent recovery actions to feed repeat-failure detection."""
    if detector is None:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{DECISION_ENGINE_URL}/actions", params={"limit": 50})
            if resp.status_code != 200:
                return
            actions = resp.json()
            # Count per-service actions in the last 30 minutes
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
            counts: dict[str, int] = {}
            for action in actions:
                started = action.get("started_at", "")
                try:
                    action_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    if action_time >= cutoff:
                        svc = action.get("target_service", "")
                        counts[svc] = counts.get(svc, 0) + 1
                except (ValueError, AttributeError):
                    pass
            detector.set_recovery_counts(counts)
    except Exception:
        logger.warning("recovery_counts.fetch_error", exc_info=True)
