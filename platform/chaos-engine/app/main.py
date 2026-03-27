"""SKAM Chaos Engine -- FastAPI application entry point."""

from __future__ import annotations

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kubernetes import client, config as k8s_config
from prometheus_client import make_asgi_app

from app.experiments import router as experiments_router, scenario_router, set_injector
from app.faults import FaultInjector

# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("chaos_engine")

# ---------------------------------------------------------------------------
# Kubernetes client initialisation
# ---------------------------------------------------------------------------

_k8s_ready = False


def _init_kubernetes() -> FaultInjector | None:
    """Try in-cluster config first, then fall back to local kubeconfig."""
    global _k8s_ready  # noqa: PLW0603

    try:
        k8s_config.load_incluster_config()
        logger.info("k8s.config.in_cluster")
    except k8s_config.ConfigException:
        try:
            k8s_config.load_kube_config()
            logger.info("k8s.config.kubeconfig")
        except k8s_config.ConfigException:
            logger.error("k8s.config.failed", detail="Could not load in-cluster or kubeconfig")
            return None

    core_v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    networking_v1 = client.NetworkingV1Api()
    _k8s_ready = True
    return FaultInjector(core_v1, apps_v1, networking_v1)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app.startup")
    injector = _init_kubernetes()
    if injector is not None:
        set_injector(injector)
        logger.info("app.fault_injector.ready")
    else:
        logger.warning("app.fault_injector.unavailable", detail="Kubernetes operations will fail")
    yield
    logger.info("app.shutdown")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SKAM Chaos Engine",
    description="Programmatic Kubernetes-native chaos engineering service",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics as a sub-application.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers.
app.include_router(experiments_router)
app.include_router(scenario_router)

# ---------------------------------------------------------------------------
# Health / readiness endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health():
    """Liveness probe -- always returns 200 if the process is running."""
    return {"status": "healthy"}


@app.get("/ready", tags=["health"])
async def readiness():
    """Readiness probe -- returns 200 only when the K8s client is initialised."""
    if not _k8s_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Kubernetes client not initialised"},
        )
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
