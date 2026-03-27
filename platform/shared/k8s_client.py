"""Shared Kubernetes client initialization for platform services."""

import structlog
from kubernetes import client, config
from kubernetes.client import ApiException

logger = structlog.get_logger()


def create_k8s_clients() -> tuple[client.CoreV1Api, client.AppsV1Api, client.AutoscalingV2Api, client.NetworkingV1Api]:
    """Initialize Kubernetes API clients with in-cluster or local kubeconfig fallback."""
    try:
        config.load_incluster_config()
        logger.info("loaded in-cluster kubernetes config")
    except config.ConfigException:
        try:
            config.load_kube_config()
            logger.info("loaded local kubeconfig")
        except config.ConfigException as e:
            logger.error("failed to load kubernetes config", error=str(e))
            raise

    core_v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()
    autoscaling_v2 = client.AutoscalingV2Api()
    networking_v1 = client.NetworkingV1Api()

    return core_v1, apps_v1, autoscaling_v2, networking_v1


def get_pods_by_label(core_v1: client.CoreV1Api, namespace: str, label_selector: str) -> list:
    """Get pods matching a label selector in a namespace."""
    try:
        pods = core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
        )
        return pods.items
    except ApiException as e:
        logger.error("failed to list pods", namespace=namespace, selector=label_selector, error=str(e))
        return []


def get_deployment(apps_v1: client.AppsV1Api, name: str, namespace: str) -> client.V1Deployment | None:
    """Get a deployment by name."""
    try:
        return apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
    except ApiException as e:
        if e.status == 404:
            logger.warning("deployment not found", name=name, namespace=namespace)
            return None
        raise
