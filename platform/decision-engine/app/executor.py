"""Recovery action executor -- performs Kubernetes operations to heal services.

Supports urgency-based execution (CRITICAL actions skip the queue),
post-action validation with severity-specific timeouts, and healing
speed metrics (time from anomaly detection to recovery confirmation).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import structlog
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = structlog.get_logger(__name__)

NAMESPACE = os.getenv("NAMESPACE", "skam-platform")

# Severity level constants (mirror anomaly-detector Severity IntEnum)
_SEV_CRITICAL = 4
_SEV_HIGH = 3

# Validation timeouts per severity level (seconds)
_VALIDATION_TIMEOUTS: dict[int, float] = {
    0: 120.0,  # NORMAL
    1: 120.0,  # LOW
    2: 90.0,   # MEDIUM
    3: 60.0,   # HIGH
    4: 30.0,   # CRITICAL
}


class HealingSpeedRecord:
    """Tracks the time from anomaly detection to confirmed recovery."""

    __slots__ = ("service", "action_id", "severity_level", "detection_ts",
                 "action_started_ts", "recovery_confirmed_ts", "healing_duration_s")

    def __init__(
        self,
        service: str,
        action_id: str,
        severity_level: int,
        detection_ts: float,
    ) -> None:
        self.service = service
        self.action_id = action_id
        self.severity_level = severity_level
        self.detection_ts = detection_ts
        self.action_started_ts: float = 0.0
        self.recovery_confirmed_ts: float = 0.0
        self.healing_duration_s: float = 0.0

    def mark_started(self) -> None:
        self.action_started_ts = time.time()

    def mark_recovered(self) -> None:
        self.recovery_confirmed_ts = time.time()
        self.healing_duration_s = self.recovery_confirmed_ts - self.detection_ts

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "action_id": self.action_id,
            "severity_level": self.severity_level,
            "detection_ts": self.detection_ts,
            "action_started_ts": self.action_started_ts,
            "recovery_confirmed_ts": self.recovery_confirmed_ts,
            "healing_duration_s": round(self.healing_duration_s, 3),
        }


class RecoveryExecutor:
    """Executes recovery actions against the Kubernetes API.

    Features:
    - Urgency-based execution: CRITICAL actions bypass the normal queue
      and execute immediately via ``execute_urgent``.
    - Post-action validation with severity-specific timeouts.
    - Healing speed metric tracking (detection -> recovery confirmation).
    """

    def __init__(self) -> None:
        self._k8s_available = False
        try:
            config.load_incluster_config()
            self._k8s_available = True
            logger.info("k8s_in_cluster_config_loaded")
        except config.ConfigException:
            try:
                config.load_kube_config()
                self._k8s_available = True
                logger.info("k8s_kube_config_loaded")
            except config.ConfigException:
                logger.warning(
                    "k8s_config_unavailable",
                    msg="Kubernetes client could not be configured; executor will operate in dry-run mode",
                )

        if self._k8s_available:
            self.core_v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.autoscaling_v1 = client.AutoscalingV1Api()
            self.networking_v1 = client.NetworkingV1Api()
        else:
            self.core_v1 = None
            self.apps_v1 = None
            self.autoscaling_v1 = None
            self.networking_v1 = None

        # Healing speed tracking (last 200 records)
        self._healing_records: deque[HealingSpeedRecord] = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_type: str,
        service: str,
        parameters: dict[str, Any],
        severity_level: int = 0,
        detection_ts: float = 0.0,
        action_id: str = "",
    ) -> dict[str, Any]:
        """Dispatch to the appropriate recovery method.

        Returns a dict with at least ``{"success": bool, "message": str}``.

        Args:
            action_type: The recovery action to perform.
            service: Target Kubernetes service name.
            parameters: Action-specific parameters.
            severity_level: Numeric severity (0-4) for validation timeout selection.
            detection_ts: Epoch timestamp when the anomaly was first detected.
            action_id: Unique identifier for the recovery action.
        """
        namespace = parameters.pop("namespace", NAMESPACE)

        # Start healing speed tracking
        record = HealingSpeedRecord(
            service=service,
            action_id=action_id,
            severity_level=severity_level,
            detection_ts=detection_ts or time.time(),
        )
        record.mark_started()

        dispatch = {
            "restart_pod": self.restart_pod,
            "scale_up": self.scale_up,
            "rolling_restart": self.rolling_restart,
            "remove_network_policy": self.remove_network_policy,
            "increase_resources": self.increase_resources,
            "restart_redis": self.restart_redis,
        }

        handler = dispatch.get(action_type)
        if handler is None:
            msg = f"Unknown action type: {action_type}"
            logger.error("unknown_action_type", action_type=action_type)
            return {"success": False, "message": msg}

        try:
            result = await handler(service=service, namespace=namespace, **parameters)
        except Exception as exc:
            logger.exception(
                "executor_error",
                action_type=action_type,
                service=service,
                error=str(exc),
            )
            return {"success": False, "message": f"Executor error: {exc}"}

        # Post-action validation
        if result.get("success"):
            validation_timeout = _VALIDATION_TIMEOUTS.get(severity_level, 120.0)
            validation = await self._validate_recovery(
                service=service,
                namespace=namespace,
                timeout_s=validation_timeout,
                severity_level=severity_level,
            )
            result["validation"] = validation

            if validation.get("recovered"):
                record.mark_recovered()
                self._healing_records.append(record)
                result["healing_duration_s"] = record.healing_duration_s
                logger.info(
                    "healing_speed_recorded",
                    service=service,
                    action_id=action_id,
                    severity_level=severity_level,
                    healing_duration_s=record.healing_duration_s,
                )

        return result

    async def execute_urgent(
        self,
        action_type: str,
        service: str,
        parameters: dict[str, Any],
        detection_ts: float = 0.0,
        action_id: str = "",
    ) -> dict[str, Any]:
        """Execute a CRITICAL-severity action immediately, bypassing any queue.

        This is a fast-path for CRITICAL alerts that should not wait behind
        lower-priority actions in the execution pipeline.
        """
        logger.warning(
            "urgent_execution_started",
            action_type=action_type,
            service=service,
            action_id=action_id,
        )
        return await self.execute(
            action_type=action_type,
            service=service,
            parameters=parameters,
            severity_level=_SEV_CRITICAL,
            detection_ts=detection_ts,
            action_id=action_id,
        )

    # ------------------------------------------------------------------
    # Healing speed metrics
    # ------------------------------------------------------------------

    def get_healing_records(self, service: str | None = None) -> list[dict[str, Any]]:
        """Return recent healing speed records, optionally filtered by service."""
        records = self._healing_records
        if service:
            records = [r for r in records if r.service == service]
        return [r.to_dict() for r in records]

    def average_healing_time(self, service: str | None = None) -> float:
        """Return the average healing duration in seconds across recent records."""
        records = self._healing_records
        if service:
            records = [r for r in records if r.service == service]
        if not records:
            return 0.0
        return sum(r.healing_duration_s for r in records) / len(records)

    # ------------------------------------------------------------------
    # Post-action validation
    # ------------------------------------------------------------------

    async def _validate_recovery(
        self,
        service: str,
        namespace: str,
        timeout_s: float,
        severity_level: int,
    ) -> dict[str, Any]:
        """Validate that the service recovered after the action.

        Checks that all pods are Running and Ready within the severity-specific
        timeout window.  Returns a dict with validation outcome.
        """
        log = logger.bind(
            action="validate_recovery",
            service=service,
            timeout_s=timeout_s,
            severity_level=severity_level,
        )

        if not self._k8s_available:
            log.info("dry_run_validation")
            return {
                "recovered": True,
                "message": f"[dry-run] Assumed recovery for {service}",
                "timeout_s": timeout_s,
            }

        start = time.time()
        poll_interval = min(5.0, timeout_s / 4)
        attempts = 0

        while (time.time() - start) < timeout_s:
            attempts += 1
            try:
                pods = await asyncio.to_thread(
                    self.core_v1.list_namespaced_pod,
                    namespace=namespace,
                    label_selector=f"app={service}",
                )

                if pods.items:
                    all_ready = True
                    for p in pods.items:
                        if p.status.phase != "Running":
                            all_ready = False
                            break
                        for cs in (p.status.container_statuses or []):
                            if cs and not cs.ready:
                                all_ready = False
                                break

                    if all_ready:
                        elapsed = time.time() - start
                        log.info(
                            "recovery_validated",
                            elapsed_s=round(elapsed, 2),
                            attempts=attempts,
                        )
                        return {
                            "recovered": True,
                            "message": f"All pods for {service} are Running and Ready",
                            "elapsed_s": round(elapsed, 2),
                            "attempts": attempts,
                        }
            except ApiException as exc:
                log.warning("validation_api_error", error=str(exc))

            await asyncio.sleep(poll_interval)

        elapsed = time.time() - start
        log.warning("recovery_validation_timeout", elapsed_s=round(elapsed, 2))
        return {
            "recovered": False,
            "message": f"Recovery not confirmed for {service} within {timeout_s}s",
            "elapsed_s": round(elapsed, 2),
            "attempts": attempts,
        }

    # ------------------------------------------------------------------
    # Recovery actions
    # ------------------------------------------------------------------

    async def restart_pod(
        self, service: str, namespace: str, **_kwargs: Any
    ) -> dict[str, Any]:
        """Delete the oldest pod of a deployment so the controller recreates it."""
        log = logger.bind(action="restart_pod", service=service, namespace=namespace)

        if not self._k8s_available:
            log.info("dry_run")
            return {"success": True, "message": f"[dry-run] Would restart oldest pod of {service}"}

        pods = await asyncio.to_thread(
            self.core_v1.list_namespaced_pod,
            namespace=namespace,
            label_selector=f"app={service}",
        )

        if not pods.items:
            log.warning("no_pods_found")
            return {"success": False, "message": f"No pods found for {service}"}

        # Pick the oldest pod by creation timestamp
        oldest = min(pods.items, key=lambda p: p.metadata.creation_timestamp)
        pod_name = oldest.metadata.name

        log.info("deleting_pod", pod=pod_name)
        await asyncio.to_thread(
            self.core_v1.delete_namespaced_pod,
            name=pod_name,
            namespace=namespace,
        )

        log.info("pod_deleted", pod=pod_name)
        return {
            "success": True,
            "message": f"Deleted pod {pod_name}; deployment will recreate it",
            "deleted_pod": pod_name,
        }

    async def scale_up(
        self,
        service: str,
        namespace: str,
        replicas_add: int = 2,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Increase replicas on a deployment (or patch HPA minReplicas)."""
        log = logger.bind(action="scale_up", service=service, replicas_add=replicas_add)

        if not self._k8s_available:
            log.info("dry_run")
            return {"success": True, "message": f"[dry-run] Would scale {service} +{replicas_add}"}

        # Try HPA first
        try:
            hpa = await asyncio.to_thread(
                self.autoscaling_v1.read_namespaced_horizontal_pod_autoscaler,
                name=service,
                namespace=namespace,
            )
            new_min = (hpa.spec.min_replicas or 1) + replicas_add
            hpa.spec.min_replicas = new_min
            await asyncio.to_thread(
                self.autoscaling_v1.patch_namespaced_horizontal_pod_autoscaler,
                name=service,
                namespace=namespace,
                body=hpa,
            )
            log.info("hpa_scaled", new_min_replicas=new_min)
            return {
                "success": True,
                "message": f"HPA {service} minReplicas set to {new_min}",
                "new_min_replicas": new_min,
            }
        except ApiException as exc:
            if exc.status != 404:
                raise
            log.debug("hpa_not_found_falling_back_to_deployment")

        # Fall back to deployment scale
        deploy = await asyncio.to_thread(
            self.apps_v1.read_namespaced_deployment,
            name=service,
            namespace=namespace,
        )
        current = deploy.spec.replicas or 1
        new_replicas = current + replicas_add

        await asyncio.to_thread(
            self.apps_v1.patch_namespaced_deployment_scale,
            name=service,
            namespace=namespace,
            body={"spec": {"replicas": new_replicas}},
        )

        log.info("deployment_scaled", previous=current, new=new_replicas)
        return {
            "success": True,
            "message": f"Scaled {service} from {current} to {new_replicas} replicas",
            "previous_replicas": current,
            "new_replicas": new_replicas,
        }

    async def rolling_restart(
        self, service: str, namespace: str, **_kwargs: Any
    ) -> dict[str, Any]:
        """Trigger a rolling restart by annotating the deployment pod template."""
        log = logger.bind(action="rolling_restart", service=service)

        if not self._k8s_available:
            log.info("dry_run")
            return {"success": True, "message": f"[dry-run] Would rolling-restart {service}"}

        restart_ts = datetime.now(timezone.utc).isoformat()
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": restart_ts,
                        }
                    }
                }
            }
        }

        await asyncio.to_thread(
            self.apps_v1.patch_namespaced_deployment,
            name=service,
            namespace=namespace,
            body=patch_body,
        )

        log.info("rolling_restart_triggered", restart_at=restart_ts)
        return {
            "success": True,
            "message": f"Rolling restart triggered for {service}",
            "restart_annotation": restart_ts,
        }

    async def remove_network_policy(
        self,
        service: str,
        namespace: str,
        policy_prefix: str = "chaos-",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Delete NetworkPolicies whose name starts with *policy_prefix*."""
        log = logger.bind(
            action="remove_network_policy", service=service, prefix=policy_prefix
        )

        if not self._k8s_available:
            log.info("dry_run")
            return {
                "success": True,
                "message": f"[dry-run] Would remove {policy_prefix}* NetworkPolicies in {namespace}",
            }

        policies = await asyncio.to_thread(
            self.networking_v1.list_namespaced_network_policy,
            namespace=namespace,
        )

        deleted: list[str] = []
        for pol in policies.items:
            if pol.metadata.name.startswith(policy_prefix):
                await asyncio.to_thread(
                    self.networking_v1.delete_namespaced_network_policy,
                    name=pol.metadata.name,
                    namespace=namespace,
                )
                deleted.append(pol.metadata.name)
                log.info("network_policy_deleted", policy=pol.metadata.name)

        if not deleted:
            log.info("no_matching_network_policies")
            return {
                "success": True,
                "message": f"No NetworkPolicies with prefix '{policy_prefix}' found",
                "deleted": [],
            }

        return {
            "success": True,
            "message": f"Deleted {len(deleted)} NetworkPolicies: {', '.join(deleted)}",
            "deleted": deleted,
        }

    async def increase_resources(
        self,
        service: str,
        namespace: str,
        cpu: str = "500m",
        memory: str = "512Mi",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Patch the first container's resource requests/limits on a deployment."""
        log = logger.bind(action="increase_resources", service=service, cpu=cpu, memory=memory)

        if not self._k8s_available:
            log.info("dry_run")
            return {
                "success": True,
                "message": f"[dry-run] Would set {service} resources to cpu={cpu}, memory={memory}",
            }

        deploy = await asyncio.to_thread(
            self.apps_v1.read_namespaced_deployment,
            name=service,
            namespace=namespace,
        )

        container = deploy.spec.template.spec.containers[0]
        container.resources = client.V1ResourceRequirements(
            requests={"cpu": cpu, "memory": memory},
            limits={"cpu": cpu, "memory": memory},
        )

        await asyncio.to_thread(
            self.apps_v1.patch_namespaced_deployment,
            name=service,
            namespace=namespace,
            body=deploy,
        )

        log.info("resources_increased", cpu=cpu, memory=memory)
        return {
            "success": True,
            "message": f"Resources for {service} set to cpu={cpu}, memory={memory}",
        }

    async def restart_redis(
        self, namespace: str, service: str = "redis", **_kwargs: Any
    ) -> dict[str, Any]:
        """Delete the redis pod and wait for a replacement to become Ready."""
        log = logger.bind(action="restart_redis", namespace=namespace)

        if not self._k8s_available:
            log.info("dry_run")
            return {"success": True, "message": "[dry-run] Would restart redis pod"}

        pods = await asyncio.to_thread(
            self.core_v1.list_namespaced_pod,
            namespace=namespace,
            label_selector="app=redis",
        )

        if not pods.items:
            log.warning("no_redis_pods_found")
            return {"success": False, "message": "No redis pods found"}

        pod_name = pods.items[0].metadata.name
        log.info("deleting_redis_pod", pod=pod_name)

        await asyncio.to_thread(
            self.core_v1.delete_namespaced_pod,
            name=pod_name,
            namespace=namespace,
        )

        # Wait for a new redis pod to become ready (up to 90 seconds)
        for attempt in range(18):
            await asyncio.sleep(5)
            new_pods = await asyncio.to_thread(
                self.core_v1.list_namespaced_pod,
                namespace=namespace,
                label_selector="app=redis",
            )
            for p in new_pods.items:
                if p.metadata.name != pod_name and p.status.phase == "Running":
                    ready = all(
                        cs.ready
                        for cs in (p.status.container_statuses or [])
                        if cs is not None
                    )
                    if ready:
                        log.info("redis_pod_ready", pod=p.metadata.name)
                        return {
                            "success": True,
                            "message": f"Redis pod {p.metadata.name} is Running and Ready",
                            "new_pod": p.metadata.name,
                        }

        log.warning("redis_pod_not_ready_timeout")
        return {
            "success": False,
            "message": "Redis pod did not become ready within 90 seconds",
        }
