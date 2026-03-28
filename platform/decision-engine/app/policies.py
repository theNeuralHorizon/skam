"""Policy engine that maps anomaly alerts to recovery actions.

Severity-aware: policies declare a minimum severity level, CRITICAL alerts
bypass normal cooldowns, and sustained HIGH+ severity triggers escalation.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from app.metrics import policy_evaluations_total
from app.models import AnomalyAlert, HealingPolicy, RecoveryAction

logger = structlog.get_logger(__name__)

# Severity level constants (mirror anomaly-detector Severity IntEnum)
_SEV_NORMAL = 0
_SEV_LOW = 1
_SEV_MEDIUM = 2
_SEV_HIGH = 3
_SEV_CRITICAL = 4

# ---------------------------------------------------------------------------
# Default healing policies  (ordered by priority -- first match wins)
# ---------------------------------------------------------------------------
DEFAULT_POLICIES: list[HealingPolicy] = [
    HealingPolicy(
        name="network-partition-fix",
        anomaly_type="availability",
        severity_threshold=0.9,
        action_type="remove_network_policy",
        parameters={"policy_prefix": "chaos-"},
        cooldown_seconds=30,
        min_severity_level=_SEV_HIGH,
        critical_cooldown_seconds=10,
        escalation_after_cycles=2,
    ),
    HealingPolicy(
        name="pod-crash-recovery",
        anomaly_type="availability",
        severity_threshold=0.8,
        action_type="restart_pod",
        parameters={},
        cooldown_seconds=60,
        min_severity_level=_SEV_MEDIUM,
        critical_cooldown_seconds=15,
        escalation_after_cycles=3,
    ),
    HealingPolicy(
        name="cache-failure-recovery",
        anomaly_type="availability",
        severity_threshold=0.8,
        action_type="restart_redis",
        parameters={},
        cooldown_seconds=120,
        min_severity_level=_SEV_MEDIUM,
        critical_cooldown_seconds=30,
        escalation_after_cycles=3,
    ),
    HealingPolicy(
        name="high-latency-scale",
        anomaly_type="latency",
        severity_threshold=0.7,
        action_type="scale_up",
        parameters={"replicas_add": 2},
        cooldown_seconds=180,
        min_severity_level=_SEV_LOW,
        critical_cooldown_seconds=30,
        escalation_after_cycles=3,
    ),
    HealingPolicy(
        name="error-rate-restart",
        anomaly_type="error_rate",
        severity_threshold=0.7,
        action_type="rolling_restart",
        parameters={},
        cooldown_seconds=120,
        min_severity_level=_SEV_LOW,
        critical_cooldown_seconds=20,
        escalation_after_cycles=3,
    ),
    HealingPolicy(
        name="cpu-saturation-scale",
        anomaly_type="resource",
        severity_threshold=0.8,
        action_type="scale_up",
        parameters={"replicas_add": 2},
        cooldown_seconds=180,
        min_severity_level=_SEV_MEDIUM,
        critical_cooldown_seconds=30,
        escalation_after_cycles=3,
    ),
]

# Maps action_type -> escalated action_type for severity escalation
_ESCALATION_MAP: dict[str, str] = {
    "restart_pod": "rolling_restart",
    "scale_up": "scale_up",          # escalate by increasing replicas_add
    "rolling_restart": "rolling_restart",
    "restart_redis": "restart_redis",
    "remove_network_policy": "remove_network_policy",
}

# Additional parameters applied when escalating an action
_ESCALATION_PARAMS: dict[str, dict] = {
    "scale_up": {"replicas_add": 4},  # double the normal scale-up
}


class PolicyEngine:
    """Evaluates anomaly alerts against healing policies and produces recovery actions.

    Severity-aware features:
    - Policies can declare a ``min_severity_level`` gate.
    - CRITICAL severity bypasses normal cooldowns (uses ``critical_cooldown_seconds``).
    - Sustained HIGH+ severity for ``escalation_after_cycles`` triggers an
      escalated action (e.g. restart_pod -> rolling_restart).
    """

    def __init__(self, policies: list[HealingPolicy] | None = None) -> None:
        self.policies: list[HealingPolicy] = list(policies or DEFAULT_POLICIES)
        # Cooldown tracker:  key = (service, action_type) -> last execution epoch
        self._cooldowns: dict[tuple[str, str], float] = {}
        # Escalation tracker: key = (service, policy_name) -> count of HIGH+ cycles
        self._escalation_counters: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, alert: AnomalyAlert) -> Optional[RecoveryAction]:
        """Find the first matching policy for *alert* and return a RecoveryAction.

        Returns ``None`` if no policy matches or the cooldown window has not
        elapsed for the matching action.
        """
        is_critical = alert.severity_level >= _SEV_CRITICAL

        for policy in self.policies:
            if not self._policy_matches(policy, alert):
                continue

            cooldown_key = (alert.service, policy.action_type)

            # Determine effective cooldown: CRITICAL uses shorter cooldown
            effective_cooldown = policy.cooldown_seconds
            if is_critical and policy.critical_cooldown_seconds > 0:
                effective_cooldown = policy.critical_cooldown_seconds

            if self._in_cooldown(cooldown_key, effective_cooldown):
                logger.info(
                    "policy_cooldown_active",
                    policy=policy.name,
                    service=alert.service,
                    action_type=policy.action_type,
                    effective_cooldown=effective_cooldown,
                    is_critical=is_critical,
                )
                policy_evaluations_total.labels(
                    policy_name=policy.name, result="cooldown"
                ).inc()
                continue

            # Record the cooldown timestamp *before* returning the action so
            # that duplicate alerts arriving in quick succession are suppressed.
            self._cooldowns[cooldown_key] = time.time()

            # --- Escalation check ---
            action_type = policy.action_type
            parameters = dict(policy.parameters)
            escalation_key = (alert.service, policy.name)

            if alert.severity_level >= _SEV_HIGH:
                self._escalation_counters[escalation_key] = (
                    self._escalation_counters.get(escalation_key, 0) + 1
                )
            else:
                self._escalation_counters[escalation_key] = 0

            escalated = False
            if self._escalation_counters.get(escalation_key, 0) >= policy.escalation_after_cycles:
                escalated_type = _ESCALATION_MAP.get(action_type)
                if escalated_type:
                    action_type = escalated_type
                    parameters.update(_ESCALATION_PARAMS.get(action_type, {}))
                    escalated = True
                    logger.warning(
                        "policy_escalated",
                        policy=policy.name,
                        service=alert.service,
                        original_action=policy.action_type,
                        escalated_action=action_type,
                        high_severity_cycles=self._escalation_counters[escalation_key],
                    )
                # Reset counter after escalation
                self._escalation_counters[escalation_key] = 0

            action = RecoveryAction(
                id=str(uuid.uuid4()),
                alert=alert,
                action_type=action_type,
                target_service=alert.service,
                parameters=parameters,
                status="critical_priority" if is_critical else "pending",
                started_at=datetime.now(timezone.utc),
            )

            logger.info(
                "policy_matched",
                policy=policy.name,
                service=alert.service,
                action_type=action_type,
                severity=alert.severity,
                severity_label=alert.severity_label,
                severity_level=alert.severity_level,
                action_id=action.id,
                escalated=escalated,
                is_critical=is_critical,
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
            severity_label=alert.severity_label,
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
        # Gate on minimum severity level (if the classifier populated it)
        if alert.severity_level < policy.min_severity_level:
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
