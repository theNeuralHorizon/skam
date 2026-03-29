"""Prometheus metric collector -- fetches telemetry for each monitored service."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from app.models import ServiceMetrics

logger = structlog.get_logger(__name__)

PROMETHEUS_URL: str = os.getenv(
    "PROMETHEUS_URL",
    "http://prometheus-kube-prometheus-prometheus.monitoring:9090",
)

MONITORED_SERVICES: list[str] = [
    "api-gateway",
    "user-service",
    "product-service",
    "order-service",
    "payment-service",
    "notification-service",
]


class MetricsCollector:
    """Queries Prometheus HTTP API and returns structured ``ServiceMetrics``."""

    def __init__(self, prometheus_url: str | None = None) -> None:
        self._base_url = (prometheus_url or PROMETHEUS_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("metrics_collector.started", prometheus_url=self._base_url)

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- public API ----------------------------------------------------------

    async def collect_service_metrics(self, service: str) -> ServiceMetrics:
        """Collect all five feature metrics for *service* from Prometheus."""
        now = datetime.now(timezone.utc)

        request_rate = await self._instant_query(
            f'rate(http_requests_total{{app="{service}"}}[1m])'
        )
        total_rate = request_rate  # reuse for denominator

        error_rate_raw = await self._instant_query(
            f'rate(http_requests_total{{app="{service}",status=~"5.."}}[1m])'
        )
        error_rate = (error_rate_raw / total_rate) if total_rate > 0 else 0.0

        p99_latency = await self._instant_query(
            f'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{{app="{service}"}}[1m]))'
        )

        cpu_usage = await self._instant_query(
            f'rate(container_cpu_usage_seconds_total{{pod=~"{service}.*"}}[1m])'
        )

        memory_usage = await self._instant_query(
            f'container_memory_usage_bytes{{pod=~"{service}.*"}}'
        )

        memory_limit = await self._instant_query(
            f'container_spec_memory_limit_bytes{{pod=~"{service}.*"}}'
        )

        return ServiceMetrics(
            service=service,
            timestamp=now,
            request_rate=request_rate,
            error_rate=min(max(error_rate, 0.0), 1.0),
            p99_latency=max(p99_latency, 0.0),
            cpu_usage=max(cpu_usage, 0.0),
            memory_usage=max(memory_usage, 0.0),
            memory_limit=max(memory_limit, 0.0),
        )

    async def collect_all_services(self) -> list[ServiceMetrics]:
        """Collect metrics for every monitored service, skipping failures."""
        results: list[ServiceMetrics] = []
        for svc in MONITORED_SERVICES:
            try:
                metrics = await self.collect_service_metrics(svc)
                results.append(metrics)
            except Exception:
                logger.warning("metrics_collector.service_failed", service=svc, exc_info=True)
        return results

    # -- internals -----------------------------------------------------------

    async def _instant_query(self, promql: str) -> float:
        """Execute a Prometheus instant query and return the scalar value.

        Returns ``0.0`` when the query produces no result or the upstream is
        unreachable -- the detector pipeline treats zeros as a missing-data
        signal rather than crashing.
        """
        if self._client is None:
            return 0.0

        try:
            resp = await self._client.get(
                "/api/v1/query",
                params={"query": promql},
            )
            resp.raise_for_status()
            return self._extract_value(resp.json())
        except Exception:
            logger.warning("metrics_collector.query_failed", query=promql, exc_info=True)
            return 0.0

    @staticmethod
    def _extract_value(payload: dict[str, Any]) -> float:
        """Pull the first numeric value from a Prometheus JSON response."""
        try:
            result = payload["data"]["result"]
            if not result:
                return 0.0
            # Instant query returns [[timestamp, "value"], ...]
            value_str = result[0]["value"][1]
            value = float(value_str)
            # Prometheus may return NaN/Inf for edge cases
            if value != value or value == float("inf") or value == float("-inf"):
                return 0.0
            return value
        except (KeyError, IndexError, TypeError, ValueError):
            return 0.0
