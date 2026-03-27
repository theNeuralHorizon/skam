"""Core fault injection implementations for the SKAM Chaos Engine.

Each public method accepts an ``ExperimentTarget`` and a parameters dict,
performs the corresponding Kubernetes mutation, and returns *rollback info* --
a dict that contains everything needed to undo the change later.

The ``rollback`` method accepts a fault type string and the rollback info dict
and reverses the mutation.
"""

from __future__ import annotations

import random
from typing import Any

import structlog
from kubernetes import client
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream as k8s_stream

from app.models import ExperimentTarget

logger = structlog.get_logger(__name__)


class FaultInjector:
    """Kubernetes-native fault injector.

    Holds references to the required Kubernetes API clients and exposes one
    method per supported fault type plus a generic ``rollback`` dispatcher.
    """

    # Sentinel image used to force a CrashLoopBackOff.
    _CRASH_IMAGE = "invalid:crash-loop-sentinel"

    # Label applied to resources created by the chaos engine so they can be
    # identified and cleaned up.
    _LABEL = "skam-chaos-engine"

    def __init__(self, core_v1: client.CoreV1Api, apps_v1: client.AppsV1Api, networking_v1: client.NetworkingV1Api) -> None:
        self._core = core_v1
        self._apps = apps_v1
        self._networking = networking_v1

    # ------------------------------------------------------------------
    # Fault: pod_kill
    # ------------------------------------------------------------------

    async def pod_kill(self, target: ExperimentTarget, params: dict[str, Any]) -> dict:
        """Delete one or more pods matching *target*.

        Parameters
        ----------
        params.count : int, default 1
            Number of pods to kill.  If more pods exist than *count*, a random
            subset is chosen.
        """
        count = int(params.get("count", 1))
        pods = self._list_pods(target)
        if not pods:
            raise RuntimeError(
                f"No pods found for selector '{target.label_selector}' "
                f"in namespace '{target.namespace}'"
            )

        victims = random.sample(pods, min(count, len(pods)))
        deleted_names: list[str] = []

        for pod in victims:
            name = pod.metadata.name
            logger.info(
                "chaos.pod_kill",
                pod=name,
                namespace=target.namespace,
            )
            self._core.delete_namespaced_pod(
                name=name,
                namespace=target.namespace,
                grace_period_seconds=0,
            )
            deleted_names.append(name)

        return {"deleted_pods": deleted_names, "namespace": target.namespace}

    # ------------------------------------------------------------------
    # Fault: pod_crash_loop
    # ------------------------------------------------------------------

    async def pod_crash_loop(self, target: ExperimentTarget, params: dict[str, Any]) -> dict:
        """Patch a Deployment to use an invalid image, triggering CrashLoopBackOff.

        Parameters
        ----------
        params.deployment : str
            Name of the deployment to patch.
        """
        deployment_name = params.get("deployment")
        if not deployment_name:
            raise ValueError("'deployment' parameter is required for pod_crash_loop")

        deployment = self._apps.read_namespaced_deployment(
            name=deployment_name,
            namespace=target.namespace,
        )

        original_images: dict[str, str] = {}
        containers = deployment.spec.template.spec.containers
        for container in containers:
            original_images[container.name] = container.image

        logger.info(
            "chaos.pod_crash_loop.inject",
            deployment=deployment_name,
            namespace=target.namespace,
            original_images=original_images,
        )

        # Patch every container to use the invalid image.
        patch_containers = [
            {"name": name, "image": self._CRASH_IMAGE}
            for name in original_images
        ]
        self._apps.patch_namespaced_deployment(
            name=deployment_name,
            namespace=target.namespace,
            body={"spec": {"template": {"spec": {"containers": patch_containers}}}},
        )

        return {
            "deployment": deployment_name,
            "namespace": target.namespace,
            "original_images": original_images,
        }

    # ------------------------------------------------------------------
    # Fault: cpu_stress
    # ------------------------------------------------------------------

    async def cpu_stress(self, target: ExperimentTarget, params: dict[str, Any]) -> dict:
        """Exec a CPU stress workload inside target pod(s).

        Parameters
        ----------
        params.cores : int, default 2
            Number of CPU workers to spawn.
        params.load_percent : int, default 80
            Target CPU load percentage per worker.
        """
        cores = int(params.get("cores", 2))
        load_pct = int(params.get("load_percent", 80))

        pods = self._list_pods(target)
        if not pods:
            raise RuntimeError(
                f"No pods found for selector '{target.label_selector}' "
                f"in namespace '{target.namespace}'"
            )

        stressed_pods: list[str] = []
        for pod in pods:
            name = pod.metadata.name
            # Use stress-ng if available, fall back to a dd/yes pipeline.
            cmd = [
                "sh", "-c",
                (
                    f"if command -v stress-ng >/dev/null 2>&1; then "
                    f"  stress-ng --cpu {cores} --cpu-load {load_pct} --timeout 0 & "
                    f"else "
                    f"  for i in $(seq 1 {cores}); do yes > /dev/null & done; "
                    f"fi; "
                    f"echo '__chaos_stress_started__'"
                ),
            ]
            logger.info(
                "chaos.cpu_stress.inject",
                pod=name,
                namespace=target.namespace,
                cores=cores,
                load_pct=load_pct,
            )
            try:
                k8s_stream(
                    self._core.connect_get_namespaced_pod_exec,
                    name,
                    target.namespace,
                    command=cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )
                stressed_pods.append(name)
            except ApiException as exc:
                logger.warning(
                    "chaos.cpu_stress.exec_failed",
                    pod=name,
                    error=str(exc),
                )

        return {
            "stressed_pods": stressed_pods,
            "namespace": target.namespace,
            "cores": cores,
        }

    # ------------------------------------------------------------------
    # Fault: memory_pressure
    # ------------------------------------------------------------------

    async def memory_pressure(self, target: ExperimentTarget, params: dict[str, Any]) -> dict:
        """Patch Deployment to set very low memory limits on containers.

        Parameters
        ----------
        params.limit_mb : int, default 64
            Memory limit in MiB to set on every container.
        params.deployment : str | None
            Explicit deployment name.  If omitted, the first deployment
            matching *target.label_selector* is used.
        """
        limit_mb = int(params.get("limit_mb", 64))
        deployment_name = params.get("deployment")

        if not deployment_name:
            deployment_name = self._find_deployment(target)

        deployment = self._apps.read_namespaced_deployment(
            name=deployment_name,
            namespace=target.namespace,
        )

        original_limits: dict[str, dict | None] = {}
        containers = deployment.spec.template.spec.containers
        for c in containers:
            limits = None
            if c.resources and c.resources.limits:
                limits = dict(c.resources.limits)
            original_limits[c.name] = limits

        logger.info(
            "chaos.memory_pressure.inject",
            deployment=deployment_name,
            namespace=target.namespace,
            limit_mb=limit_mb,
            original_limits=original_limits,
        )

        patch_containers = [
            {
                "name": name,
                "resources": {"limits": {"memory": f"{limit_mb}Mi"}},
            }
            for name in original_limits
        ]
        self._apps.patch_namespaced_deployment(
            name=deployment_name,
            namespace=target.namespace,
            body={"spec": {"template": {"spec": {"containers": patch_containers}}}},
        )

        return {
            "deployment": deployment_name,
            "namespace": target.namespace,
            "original_limits": original_limits,
        }

    # ------------------------------------------------------------------
    # Fault: network_partition
    # ------------------------------------------------------------------

    async def network_partition(self, target: ExperimentTarget, params: dict[str, Any]) -> dict:
        """Create a NetworkPolicy that blocks all ingress and egress to target pods.

        The policy name is deterministic so we can locate it during rollback.
        """
        policy_name = f"chaos-netpart-{target.label_selector.replace('=', '-').replace(',', '-')}"
        # Truncate to comply with K8s naming rules.
        policy_name = policy_name[:63].rstrip("-")

        match_labels = self._selector_to_labels(target.label_selector)

        policy = client.V1NetworkPolicy(
            metadata=client.V1ObjectMeta(
                name=policy_name,
                namespace=target.namespace,
                labels={"managed-by": self._LABEL},
            ),
            spec=client.V1NetworkPolicySpec(
                pod_selector=client.V1LabelSelector(match_labels=match_labels),
                policy_types=["Ingress", "Egress"],
                ingress=[],   # empty = deny all
                egress=[],    # empty = deny all
            ),
        )

        logger.info(
            "chaos.network_partition.inject",
            policy=policy_name,
            namespace=target.namespace,
            match_labels=match_labels,
        )

        try:
            self._networking.create_namespaced_network_policy(
                namespace=target.namespace,
                body=policy,
            )
        except ApiException as exc:
            if exc.status == 409:
                logger.warning(
                    "chaos.network_partition.already_exists",
                    policy=policy_name,
                )
            else:
                raise

        return {
            "policy_name": policy_name,
            "namespace": target.namespace,
        }

    # ------------------------------------------------------------------
    # Fault: latency_injection
    # ------------------------------------------------------------------

    async def latency_injection(self, target: ExperimentTarget, params: dict[str, Any]) -> dict:
        """Exec ``tc netem`` commands inside target pods to add network latency.

        Parameters
        ----------
        params.delay_ms : int, default 500
            Base added delay in milliseconds.
        params.jitter_ms : int, default 100
            Random jitter in milliseconds.
        """
        delay_ms = int(params.get("delay_ms", 500))
        jitter_ms = int(params.get("jitter_ms", 100))

        pods = self._list_pods(target)
        if not pods:
            raise RuntimeError(
                f"No pods found for selector '{target.label_selector}' "
                f"in namespace '{target.namespace}'"
            )

        affected_pods: list[str] = []
        for pod in pods:
            name = pod.metadata.name
            cmd = [
                "sh", "-c",
                (
                    f"tc qdisc add dev eth0 root netem delay {delay_ms}ms {jitter_ms}ms "
                    f"|| tc qdisc change dev eth0 root netem delay {delay_ms}ms {jitter_ms}ms"
                ),
            ]
            logger.info(
                "chaos.latency_injection.inject",
                pod=name,
                namespace=target.namespace,
                delay_ms=delay_ms,
                jitter_ms=jitter_ms,
            )
            try:
                k8s_stream(
                    self._core.connect_get_namespaced_pod_exec,
                    name,
                    target.namespace,
                    command=cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )
                affected_pods.append(name)
            except ApiException as exc:
                logger.warning(
                    "chaos.latency_injection.exec_failed",
                    pod=name,
                    error=str(exc),
                )

        return {
            "affected_pods": affected_pods,
            "namespace": target.namespace,
            "delay_ms": delay_ms,
            "jitter_ms": jitter_ms,
        }

    # ------------------------------------------------------------------
    # Rollback dispatcher
    # ------------------------------------------------------------------

    async def rollback(self, fault_type: str, rollback_info: dict) -> None:
        """Undo a previously injected fault using its *rollback_info*."""
        handler = {
            "pod_kill": self._rollback_pod_kill,
            "pod_crash_loop": self._rollback_pod_crash_loop,
            "cpu_stress": self._rollback_cpu_stress,
            "memory_pressure": self._rollback_memory_pressure,
            "network_partition": self._rollback_network_partition,
            "latency_injection": self._rollback_latency_injection,
        }.get(fault_type)

        if handler is None:
            logger.error("chaos.rollback.unknown_fault", fault_type=fault_type)
            return

        logger.info("chaos.rollback.start", fault_type=fault_type, info=rollback_info)
        await handler(rollback_info)
        logger.info("chaos.rollback.complete", fault_type=fault_type)

    # ------------------------------------------------------------------
    # Rollback implementations
    # ------------------------------------------------------------------

    async def _rollback_pod_kill(self, info: dict) -> None:
        # Pods are ephemeral; the owning controller (Deployment / ReplicaSet)
        # will recreate them automatically.  Nothing to undo.
        logger.info(
            "chaos.rollback.pod_kill.noop",
            deleted_pods=info.get("deleted_pods"),
        )

    async def _rollback_pod_crash_loop(self, info: dict) -> None:
        deployment = info["deployment"]
        namespace = info["namespace"]
        original_images: dict[str, str] = info["original_images"]

        patch_containers = [
            {"name": name, "image": image}
            for name, image in original_images.items()
        ]
        logger.info(
            "chaos.rollback.pod_crash_loop",
            deployment=deployment,
            namespace=namespace,
            restoring=original_images,
        )
        self._apps.patch_namespaced_deployment(
            name=deployment,
            namespace=namespace,
            body={"spec": {"template": {"spec": {"containers": patch_containers}}}},
        )

    async def _rollback_cpu_stress(self, info: dict) -> None:
        namespace = info["namespace"]
        for pod_name in info.get("stressed_pods", []):
            cmd = ["sh", "-c", "pkill -f stress-ng; pkill -f 'yes'; true"]
            logger.info("chaos.rollback.cpu_stress", pod=pod_name)
            try:
                k8s_stream(
                    self._core.connect_get_namespaced_pod_exec,
                    pod_name,
                    namespace,
                    command=cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )
            except ApiException as exc:
                logger.warning(
                    "chaos.rollback.cpu_stress.exec_failed",
                    pod=pod_name,
                    error=str(exc),
                )

    async def _rollback_memory_pressure(self, info: dict) -> None:
        deployment = info["deployment"]
        namespace = info["namespace"]
        original_limits: dict[str, dict | None] = info["original_limits"]

        patch_containers = []
        for name, limits in original_limits.items():
            if limits is None:
                patch_containers.append({"name": name, "resources": {"limits": None}})
            else:
                patch_containers.append({"name": name, "resources": {"limits": limits}})

        logger.info(
            "chaos.rollback.memory_pressure",
            deployment=deployment,
            namespace=namespace,
        )
        self._apps.patch_namespaced_deployment(
            name=deployment,
            namespace=namespace,
            body={"spec": {"template": {"spec": {"containers": patch_containers}}}},
        )

    async def _rollback_network_partition(self, info: dict) -> None:
        policy_name = info["policy_name"]
        namespace = info["namespace"]
        logger.info(
            "chaos.rollback.network_partition",
            policy=policy_name,
            namespace=namespace,
        )
        try:
            self._networking.delete_namespaced_network_policy(
                name=policy_name,
                namespace=namespace,
            )
        except ApiException as exc:
            if exc.status == 404:
                logger.warning(
                    "chaos.rollback.network_partition.not_found",
                    policy=policy_name,
                )
            else:
                raise

    async def _rollback_latency_injection(self, info: dict) -> None:
        namespace = info["namespace"]
        for pod_name in info.get("affected_pods", []):
            cmd = ["sh", "-c", "tc qdisc del dev eth0 root || true"]
            logger.info("chaos.rollback.latency_injection", pod=pod_name)
            try:
                k8s_stream(
                    self._core.connect_get_namespaced_pod_exec,
                    pod_name,
                    namespace,
                    command=cmd,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,
                )
            except ApiException as exc:
                logger.warning(
                    "chaos.rollback.latency_injection.exec_failed",
                    pod=pod_name,
                    error=str(exc),
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _list_pods(self, target: ExperimentTarget) -> list:
        result = self._core.list_namespaced_pod(
            namespace=target.namespace,
            label_selector=target.label_selector,
            field_selector="status.phase=Running",
        )
        return result.items

    def _find_deployment(self, target: ExperimentTarget) -> str:
        result = self._apps.list_namespaced_deployment(
            namespace=target.namespace,
            label_selector=target.label_selector,
        )
        if not result.items:
            raise RuntimeError(
                f"No deployments found for selector '{target.label_selector}' "
                f"in namespace '{target.namespace}'"
            )
        return result.items[0].metadata.name

    @staticmethod
    def _selector_to_labels(selector: str) -> dict[str, str]:
        """Convert ``'app=foo,tier=backend'`` into ``{'app': 'foo', 'tier': 'backend'}``."""
        labels: dict[str, str] = {}
        for part in selector.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip()
        return labels
