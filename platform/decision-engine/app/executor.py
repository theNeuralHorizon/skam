"""Recovery action executor — performs Kubernetes operations to heal services."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

import structlog
from kubernetes import client, config
from kubernetes.client.rest import ApiException

logger = structlog.get_logger(__name__)

NAMESPACE = os.getenv("NAMESPACE", "skam-platform")


class RecoveryExecutor:
    """Executes recovery actions against the Kubernetes API."""

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

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    async def execute(
        self, action_type: str, service: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch to the appropriate recovery method.

        Returns a dict with at least ``{"success": bool, "message": str}``.
        """
        namespace = parameters.pop("namespace", NAMESPACE)

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
            return await handler(service=service, namespace=namespace, **parameters)
        except Exception as exc:
            logger.exception(
                "executor_error",
                action_type=action_type,
                service=service,
                error=str(exc),
            )
            return {"success": False, "message": f"Executor error: {exc}"}

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
