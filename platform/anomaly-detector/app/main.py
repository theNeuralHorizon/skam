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

from app.collector import MetricsCollector
from app.detector import AnomalyDetector
from app.metrics import detection_cycle_duration_seconds
from app.models import AnomalyHistoryQuery, AnomalyResult, DetectorStatus

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
_ready: bool = False

DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL_SECONDS", "15"))


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
            logger.info(
                "detection_loop.cycle_complete",
                services=len(all_metrics),
                anomalies=len(anomalies),
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
