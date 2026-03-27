"""Shared Prometheus metrics utilities for platform services."""

import time
from functools import wraps
from typing import Callable

from prometheus_client import Counter, Histogram, Gauge


def create_http_metrics(service_name: str) -> tuple[Counter, Histogram]:
    """Create standard HTTP metrics for a platform service."""
    requests_total = Counter(
        f"{service_name}_http_requests_total",
        f"Total HTTP requests to {service_name}",
        ["method", "endpoint", "status_code"],
    )
    request_duration = Histogram(
        f"{service_name}_http_request_duration_seconds",
        f"HTTP request duration for {service_name}",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    return requests_total, request_duration


def timed(histogram: Histogram, labels: dict | None = None):
    """Decorator to time a function and record it in a histogram."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                if labels:
                    histogram.labels(**labels).observe(duration)
                else:
                    histogram.observe(duration)

        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if a function is an async coroutine function."""
    import asyncio
    return asyncio.iscoroutinefunction(func)
