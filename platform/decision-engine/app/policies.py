"""Policy engine that maps anomaly alerts to recovery actions."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from app.metrics import policy_evaluations_total
from app.models import AnomalyAlert, HealingPolicy, RecoveryAction

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default healing policies  (ordered by priority — first match wins)
# ---------------------------------------------------------------------------
DEFAULT_POLICIES: list[HealingPolicy] = [
    HealingPolicy(
        name="network-partition-fix",
        anomaly_type="availability",
        severity_threshold=0.9,
        action_type="remove_network_policy",
        parameters={"policy_prefix": "chaos-"},
        cooldown_seconds=30,
    ),
    HealingPolicy(
        name="pod-crash-recovery",
        anomaly_type="availability",
        severity_threshold=0.8,
        action_type="restart_pod",
        parameters={},
        cooldown_seconds=60,
    ),
    HealingPolicy(
        name="cache-failure-recovery",
        anomaly_type="availability",
        severity_threshold=0.8,
        action_type="restart_redis",
        parameters={},
        cooldown_seconds=120,
    ),
    HealingPolicy(
        name="high-latency-scale",
        anomaly_type="latency",
        severity_threshold=0.7,
        action_type="scale_up",
        parameters={"replicas_add": 2},
        cooldown_seconds=180,
    ),
    HealingPolicy(
        name="error-rate-restart",
        anomaly_type="error_rate",
        severity_threshold=0.7,
        action_type="rolling_restart",
        parameters={},
        cooldown_seconds=120,
    ),
    HealingPolicy(
        name="cpu-saturation-scale",
        anomaly_type="resource",
        severity_threshold=0.8,
        action_type="scale_up",
        parameters={"replicas_add": 2},
        cooldown_seconds=180,
    ),
]


class PolicyEngine:
    """Evaluates anomaly alerts against healing policies and produces recovery actions."""

    def __init__(self, policies: list[HealingPolicy] | None = None) -> None:
        self.policies: list[HealingPolicy] = list(policies or DEFAULT_POLICIES)
        # Cooldown tracker:  key = (service, action_type) → last execution epoch
        self._cooldowns: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, alert: AnomalyAlert) -> Optional[RecoveryAction]:
        """Find the first matching policy for *alert* and return a RecoveryAction.

        Returns ``None`` if no policy matches or the cooldown window has not
        elapsed for the matching action.
        """
        for policy in self.policies:
            if not self._policy_matches(policy, alert):
                continue

            cooldown_key = (alert.service, policy.action_type)

            if self._in_cooldown(cooldown_key, policy.cooldown_seconds):
                logger.info(
                    "policy_cooldown_active",
                    policy=policy.name,
                    service=alert.service,
                    action_type=policy.action_type,
                )
                policy_evaluations_total.labels(
                    policy_name=policy.name, result="cooldown"
                ).inc()
                continue

            # Record the cooldown timestamp *before* returning the action so
            # that duplicate alerts arriving in quick succession are suppressed.
            self._cooldowns[cooldown_key] = time.time()

            action = RecoveryAction(
                id=str(uuid.uuid4()),
                alert=alert,
                action_type=policy.action_type,
                target_service=alert.service,
                parameters=dict(policy.parameters),
                status="pending",
                started_at=datetime.now(timezone.utc),
            )

            logger.info(
                "policy_matched",
                policy=policy.name,
                service=alert.service,
                action_type=policy.action_type,
                severity=alert.severity,
                action_id=action.id,
            )
            policy_evaluations_total.labels(
                policy_name=policy.name, result="matched"
            ).inc()
            return action

        logger.debug(
            "no_policy_matched",
            service=alert.service,
            anomaly_type=alert.anomaly_type,
            severity=alert.severity,
        )
        return None

    def get_policies(self) -> list[HealingPolicy]:
        return list(self.policies)

    def add_policy(self, policy: HealingPolicy) -> None:
        self.policies.insert(0, policy)  # highest priority
        logger.info("policy_added", policy=policy.name)

    def remove_policy(self, name: str) -> bool:
        before = len(self.policies)
        self.policies = [p for p in self.policies if p.name != name]
        removed = len(self.policies) < before
        if removed:
            logger.info("policy_removed", policy=name)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_matches(policy: HealingPolicy, alert: AnomalyAlert) -> bool:
        """Check whether a policy's conditions match the alert."""
        if policy.anomaly_type != alert.anomaly_type:
            return False
        if alert.severity < policy.severity_threshold:
            return False
        # Special case: redis-specific policy only applies to redis services
        if policy.action_type == "restart_redis" and "redis" not in alert.service.lower():
            return False
        return True

    def _in_cooldown(self, key: tuple[str, str], cooldown_seconds: int) -> bool:
        last = self._cooldowns.get(key)
        if last is None:
            return False
        return (time.time() - last) < cooldown_seconds
