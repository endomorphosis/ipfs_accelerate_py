"""Shared recovery denials; observations and policies never grant authority.

An untrusted retained-attempt marker may withhold an action. It cannot settle
an UNKNOWN callback, authorize retry, or override a native owner capability.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


INFRASTRUCTURE_PROFILE = "infrastructure-only/v1"
INFRASTRUCTURE_STALLS = frozenset({
    "probe_failed", "probe_stale", "native_catalogue_read_unavailable",
    "service_fleet_report_unavailable", "service_fleet_report_invalid",
    "service_fleet_unhealthy", "stalled_no_progress",
})
NATIVE_CATALOGUE_WAITS = frozenset({
    "native_catalogue_activation_missing", "native_catalogue_expected_blocked",
    "native_catalogue_provider_outcome_unknown", "native_catalogue_outcome_unknown",
    "native_catalogue_attempt_budget_exhausted",
    "native_catalogue_scope_mismatch", "native_catalogue_population_invalid",
    "native_catalogue_probe_invalid",
})



def validated_infrastructure_policy(board: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """An opt-in diagnostic policy grants no native task recovery authority."""
    value = board.get("recovery_policy")
    if _valid_owner_retention_policy(value):
        return None
    if value is None:
        if board.get("probe_profile") == "native_catalogue":
            raise ValueError("native catalogue probes require infrastructure-only recovery")
        return None
    fields = {"profile", "repairable_stalls", "max_actions_per_window",
              "window_seconds", "circuit_breaker_seconds"}
    if (type(value) is not dict or not fields <= set(value)
            or set(value) - fields - {"stop_own_repair_on_wait"}
            or type(value.get("stop_own_repair_on_wait", False)) is not bool
            or value.get("profile") != INFRASTRUCTURE_PROFILE):
        raise ValueError("invalid closed recovery_policy")
    stalls = value["repairable_stalls"]
    if (type(stalls) is not list or not stalls or any(type(x) is not str for x in stalls)
            or len(stalls) != len(set(stalls)) or not set(stalls) <= INFRASTRUCTURE_STALLS):
        raise ValueError("invalid infrastructure repairable stalls")
    if type(value["max_actions_per_window"]) is not int or not 1 <= value["max_actions_per_window"] <= 100:
        raise ValueError("invalid infrastructure action budget")
    for key in ("window_seconds", "circuit_breaker_seconds"):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError("invalid infrastructure time budget")
    if type(board.get("max_repair_attempts")) is not int or board["max_repair_attempts"] < 1:
        raise ValueError("infrastructure recovery requires a cumulative repair cap")
    return value


_WAIT_REASON = "expired_attempt_settlement_unavailable"
_SETTLEMENT_REASON = "production_attempt_settlement_required"
_CUSTODY_SCHEMA = "ipfs_accelerate_py/attempt-custody-observation@1"
_METADATA_CONTAINERS = (
    "details", "state", "summary", "task_state", "last_cycle_result",
    "observation", "retention_observation", "custody", "custody_observation",
)
_METADATA_COLLECTIONS = ("lanes", "lane_states", "lane_summaries", "retained_attempts")
_MAX_OBSERVATION_NODES = 4096


def retained_attempt_wait(observation: Mapping[str, Any]) -> bool:
    """Find a denial in known task/lane metadata, never arbitrary payload text.

    Cycles are harmless. An overlarge selected metadata graph is inconclusive
    and also withholds recovery; this cannot manufacture any positive right.
    """
    if not isinstance(observation, Mapping):
        return False
    pending: list[Any] = [observation]
    visited: set[int] = set()
    while pending:
        value = pending.pop()
        if not isinstance(value, Mapping) or id(value) in visited:
            continue
        visited.add(id(value))
        if len(visited) > _MAX_OBSERVATION_NODES:
            return True
        if (value.get("selection_idle_reason") == _WAIT_REASON
                or value.get("reason") == _SETTLEMENT_REASON
                or value.get("retained_attempt_settlement_required") is True
                or (value.get("schema") == _CUSTODY_SCHEMA
                    and value.get("callback_outcome") == "unknown")):
            return True
        pending.extend(value.get(key) for key in _METADATA_CONTAINERS)
        for key in _METADATA_COLLECTIONS:
            collection = value.get(key)
            if isinstance(collection, Mapping):
                # Both a single summary and a mapping indexed by lane are used.
                pending.append(collection)
                pending.extend(collection.values())
            elif isinstance(collection, (list, tuple)):
                pending.extend(collection)
    return False


def owner_retention_required(config: Mapping[str, Any]) -> bool:
    """Policy presence retains the owner, including malformed policy values."""
    return isinstance(config, Mapping) and "recovery_policy" in config


def _valid_owner_retention_policy(policy: Any) -> bool:
    if (not isinstance(policy, Mapping)
            or set(policy) != {"schema", "mode", "profile_id", "reason"}
            or policy.get("schema") != "agent-supervisor/owner-retention@1"
            or policy.get("mode") != "retain"):
        return False
    return all(
        type(policy.get(key)) is str
        and bool(policy[key].strip())
        and len(policy[key]) <= maximum
        for key, maximum in (("profile_id", 256), ("reason", 1024))
    )


def recovery_denial(config: Mapping[str, Any], observation: Mapping[str, Any],
                    *, allow_infrastructure: bool = False) -> str:
    """Return a denial for the requested boundary, never an execution grant.

    Only the watchdog and coding-repair boundaries opt into infrastructure
    diagnosis. Native heal and service-restart callers retain the strict
    default, including for a valid infrastructure-only policy.
    """
    if type(allow_infrastructure) is not bool:
        raise ValueError("allow_infrastructure must be a Boolean")
    retained_owner = owner_retention_required(config)
    infrastructure = None
    if retained_owner and not _valid_owner_retention_policy(config["recovery_policy"]):
        try:
            infrastructure = validated_infrastructure_policy(config)
        except (ValueError, TypeError, KeyError):
            return "invalid_owner_retention_policy"
        if infrastructure is None:
            return "invalid_owner_retention_policy"
    if retained_attempt_wait(observation):
        return "retained_attempt_settlement_wait"
    if infrastructure is not None:
        return "" if allow_infrastructure else "infrastructure_only_recovery"
    if retained_owner:
        return "native_owner_retention_required"
    return ""
