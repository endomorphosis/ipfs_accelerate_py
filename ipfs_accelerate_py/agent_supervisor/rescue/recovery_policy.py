"""Shared recovery denials; observations and policies never grant authority.

An untrusted retained-attempt marker may withhold an action. It cannot settle
an UNKNOWN callback, authorize retry, or override a native owner capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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


def recovery_denial(config: Mapping[str, Any], observation: Mapping[str, Any]) -> str:
    """Return only a reason to refuse recovery, never an authorization."""
    retained_owner = owner_retention_required(config)
    if retained_owner and not _valid_owner_retention_policy(config["recovery_policy"]):
        return "invalid_owner_retention_policy"
    if retained_attempt_wait(observation):
        return "retained_attempt_settlement_wait"
    if retained_owner:
        return "native_owner_retention_required"
    return ""
