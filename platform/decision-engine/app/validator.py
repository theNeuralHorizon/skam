"""Recovery validation — verifies that a recovery action actually fixed the problem."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import structlog
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = structlog.get_logger(__name__)

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server.monitoring:9090")
NAMESPACE = os.getenv("NAMESPACE", "skam-platform")
VALIDATION_TIMEOUT = 120  # seconds
VALIDATION_INTERVAL = 15  # seconds


class RecoveryValidator:
    """Validates whether a recovery action successfully healed the target service."""

    def __init__(self) -> None:
        self._k8s_available = False
        try:
            config.load_incluster_config()
            self._k8s_available = True
        except config.ConfigException:
            try:
                config.load_kube_config()
                self._k8s_available = True
            except config.ConfigException:
                logger.warning("k8s_config_unavailable_for_validator")

        if self._k8s_available:
            self.core_v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
        else:
            self.core_v1 = None
            self.apps_v1 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def validate(
        self,
        action_type: str,
        target_service: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate that the recovery action was effective.

        Polls every ``VALIDATION_INTERVAL`` seconds up to ``VALIDATION_TIMEOUT``
        seconds.  Returns ``{"success": bool, "message": str, "metrics": dict}``.
        """
        namespace = parameters.get("namespace", NAMESPACE)

        validators = {
            "restart_pod": self._validate_restart_pod,
            "scale_up": self._validate_scale_up,
            "rolling_restart": self._validate_rolling_restart,
            "remove_network_policy": self._validate_network_policy_removed,
            "increase_resources": self._validate_restart_pod,  # same check
            "restart_redis": self._validate_restart_pod,
        }

        handler = validators.get(action_type, self._validate_generic)

        elapsed = 0
        last_result: dict[str, Any] = {}

        while elapsed < VALIDATION_TIMEOUT:
            try:
                result = await handler(
                    service=target_service,
                    namespace=namespace,
                    parameters=parameters,
                )
                if result.get("success"):
                    logger.info(
                        "validation_passed",
                        action_type=action_type,
                        service=target_service,
                        elapsed=elapsed,
                    )
                    return result
                last_result = result
            except Exception as exc:
                logger.warning(
                    "validation_check_error",
                    action_type=action_type,
                    service=target_service,
                    error=str(exc),
                )
                last_result = {
                    "success": False,
                    "message": f"Validation error: {exc}",
                    "metrics": {},
                }

            await asyncio.sleep(VALIDATION_INTERVAL)
            elapsed += VALIDATION_INTERVAL

        logger.warning(
            "validation_timeout",
            action_type=action_type,
            service=target_service,
        )
        return last_result or {
            "success": False,
            "message": f"Validation timed out after {VALIDATION_TIMEOUT}s",
            "metrics": {},
        }

    # ------------------------------------------------------------------
    # Specific validators
    # ------------------------------------------------------------------

    async def _validate_restart_pod(
        self, service: str, namespace: str, **_kw: Any
    ) -> dict[str, Any]:
        """Check that at least one pod for *service* is Running and Ready."""
        if not self._k8s_available:
            return {
                "success": True,
                "message": "[dry-run] Assumed pod is Running",
                "metrics": {},
            }

        pods = await asyncio.to_thread(
            self.core_v1.list_namespaced_pod,
            namespace=namespace,
            label_selector=f"app={service}",
        )

        running_ready = 0
        total = len(pods.items)

        for pod in pods.items:
            if pod.status.phase == "Running":
                containers_ready = all(
                    cs.ready
                    for cs in (pod.status.container_statuses or [])
                    if cs is not None
                )
                if containers_ready:
                    running_ready += 1

        success = running_ready > 0
        return {
            "success": success,
            "message": f"{running_ready}/{total} pods Running+Ready"
            if success
            else f"No Ready pods for {service} ({total} total)",
            "metrics": {"running_ready": running_ready, "total_pods": total},
        }

    async def _validate_scale_up(
        self, service: str, namespace: str, parameters: dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        """Verify the deployment has the expected number of ready replicas."""
        if not self._k8s_available:
            return {
                "success": True,
                "message": "[dry-run] Assumed scale-up succeeded",
                "metrics": {},
            }

        try:
            deploy = await asyncio.to_thread(
                self.apps_v1.read_namespaced_deployment,
                name=service,
                namespace=namespace,
            )
        except ApiException as exc:
            return {
                "success": False,
                "message": f"Could not read deployment: {exc.reason}",
                "metrics": {},
            }

        desired = deploy.spec.replicas or 1
        ready = deploy.status.ready_replicas or 0
        success = ready >= desired

        return {
            "success": success,
            "message": f"{ready}/{desired} replicas ready"
            if success
            else f"Only {ready}/{desired} replicas ready",
            "metrics": {"desired_replicas": desired, "ready_replicas": ready},
        }

    async def _validate_rolling_restart(
        self, service: str, namespace: str, **_kw: Any
    ) -> dict[str, Any]:
        """Query Prometheus to check whether the error rate has dropped below threshold."""
        query = (
            f'sum(rate(http_requests_total{{service="{service}",status=~"5.."}}[2m]))'
            f' / sum(rate(http_requests_total{{service="{service}"}}[2m]))'
        )

        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query},
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("data", {}).get("result", [])
            if not results:
                # No data could mean zero errors — treat as success
                return {
                    "success": True,
                    "message": "No error rate data (assuming healthy)",
                    "metrics": {"error_rate": 0.0},
                }

            error_rate = float(results[0]["value"][1])
            success = error_rate < 0.05  # 5 % threshold

            return {
                "success": success,
                "message": f"Error rate: {error_rate:.2%}"
                if success
                else f"Error rate still high: {error_rate:.2%}",
                "metrics": {"error_rate": error_rate},
            }
        except Exception as exc:
            # If Prometheus is unreachable, fall back to pod-readiness check
            logger.warning("prometheus_query_failed", error=str(exc))
            return await self._validate_restart_pod(
                service=service, namespace=namespace
            )

    async def _validate_network_policy_removed(
        self, service: str, namespace: str, parameters: dict[str, Any], **_kw: Any
    ) -> dict[str, Any]:
        """Verify that inter-service calls succeed by querying Prometheus success rates."""
        query = (
            f'sum(rate(http_requests_total{{service="{service}",status=~"2.."}}[1m]))'
        )

        try:
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(
                    f"{PROMETHEUS_URL}/api/v1/query",
                    params={"query": query},
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("data", {}).get("result", [])
            if results:
                success_rate = float(results[0]["value"][1])
                success = success_rate > 0
                return {
                    "success": success,
                    "message": f"Success request rate: {success_rate:.2f}/s",
                    "metrics": {"success_rate": success_rate},
                }

            # No data — fall back to pod check
            return await self._validate_restart_pod(
                service=service, namespace=namespace
            )
        except Exception as exc:
            logger.warning("prometheus_query_failed", error=str(exc))
            return await self._validate_restart_pod(
                service=service, namespace=namespace
            )

    async def _validate_generic(
        self, service: str, namespace: str, **_kw: Any
    ) -> dict[str, Any]:
        """Fallback validator: just check pod readiness."""
        return await self._validate_restart_pod(service=service, namespace=namespace)
