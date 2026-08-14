"""Periodic full-checkpoint policy (IPS-042).

``CheckpointPolicy`` decides whether a seal transition must produce a full
checkpoint.  Triggers include cadence (every N accepted seals), release tags,
circuit/key/lock/trust/schema changes, cache corruption, low reuse ratio, and
maximum delta-chain depth.  Defaults fail closed.  An incremental caller
cannot override a fired trigger.

Interfaces: ``CheckpointPolicy``, ``CheckpointDecision``,
``evaluate_checkpoint_policy``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

EVIDENCE_SUBSET: Final[str] = "ips/checkpoint-policy@1"
POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "checkpoint-policy@1"
)
DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "checkpoint-decision@1"
)
STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "checkpoint-evaluation-state@1"
)

# Closed ordered trigger reason codes (plan §8.1 / §14).
CHECKPOINT_TRIGGERS: Final[tuple[str, ...]] = (
    "first_state",
    "missing_parent",
    "periodic_cadence",
    "release_tag",
    "circuit_or_key_change",
    "dependency_lock_change",
    "trust_policy_change",
    "schema_change",
    "canonicalization_change",
    "environment_change",
    "cache_corruption",
    "uncertain_cache_integrity",
    "low_reuse_ratio",
    "excessive_delta_chain_depth",
    "explicit_force",
    "full_fallback_required",
)

# Defaults match datasets VerificationPolicy sample + plan fail-closed posture.
DEFAULT_FULL_CHECKPOINT_EVERY_N_SEALS: Final[int] = 50
DEFAULT_MAX_DELTA_CHAIN_DEPTH: Final[int] = 32
DEFAULT_MIN_REUSE_RATIO_BASIS_POINTS: Final[int] = 2500


class CheckpointPolicyError(ValueError):
    """Fail-closed checkpoint-policy contract violation."""


class CheckpointMode(str, Enum):
    """Closed outcome mode for a checkpoint decision."""

    FULL_CHECKPOINT = "full_checkpoint"
    INCREMENTAL = "incremental"


class CheckpointTrigger(str, Enum):
    """Closed trigger reasons that force a full checkpoint."""

    FIRST_STATE = "first_state"
    MISSING_PARENT = "missing_parent"
    PERIODIC_CADENCE = "periodic_cadence"
    RELEASE_TAG = "release_tag"
    CIRCUIT_OR_KEY_CHANGE = "circuit_or_key_change"
    DEPENDENCY_LOCK_CHANGE = "dependency_lock_change"
    TRUST_POLICY_CHANGE = "trust_policy_change"
    SCHEMA_CHANGE = "schema_change"
    CANONICALIZATION_CHANGE = "canonicalization_change"
    ENVIRONMENT_CHANGE = "environment_change"
    CACHE_CORRUPTION = "cache_corruption"
    UNCERTAIN_CACHE_INTEGRITY = "uncertain_cache_integrity"
    LOW_REUSE_RATIO = "low_reuse_ratio"
    EXCESSIVE_DELTA_CHAIN_DEPTH = "excessive_delta_chain_depth"
    EXPLICIT_FORCE = "explicit_force"
    FULL_FALLBACK_REQUIRED = "full_fallback_required"


def _cid(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CheckpointPolicyError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointPolicyError(f"{name} must be a positive integer")
    if value < 1:
        raise CheckpointPolicyError(f"{name} must be >= 1")
    return value


def _require_nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointPolicyError(f"{name} must be a non-negative integer")
    if value < 0:
        raise CheckpointPolicyError(f"{name} must be >= 0")
    return value


def _require_bps(value: Any, name: str) -> int:
    bps = _require_nonneg_int(value, name)
    if bps > 10000:
        raise CheckpointPolicyError(f"{name} must be <= 10000")
    return bps


def parse_checkpoint_trigger(value: str | CheckpointTrigger) -> CheckpointTrigger:
    if isinstance(value, CheckpointTrigger):
        return value
    text = str(value).strip()
    try:
        return CheckpointTrigger(text)
    except ValueError as exc:
        raise CheckpointPolicyError(
            f"unknown checkpoint trigger {value!r}; closed set is "
            f"{list(CHECKPOINT_TRIGGERS)}"
        ) from exc


@dataclass(frozen=True, slots=True)
class CheckpointPolicy:
    """Fail-closed periodic and mandated full-checkpoint controls.

    Parameters mirror plan §14 and datasets ``VerificationPolicy`` checkpoint
    fields.  ``allow_incremental_override`` is always false: a caller that
    prefers incremental cannot suppress a fired trigger.
    """

    full_checkpoint_every_n_seals: int = DEFAULT_FULL_CHECKPOINT_EVERY_N_SEALS
    max_delta_chain_depth: int = DEFAULT_MAX_DELTA_CHAIN_DEPTH
    min_reuse_ratio_basis_points: int = DEFAULT_MIN_REUSE_RATIO_BASIS_POINTS
    require_full_on_release_tag: bool = True
    require_full_on_circuit_or_key_change: bool = True
    require_full_on_dependency_lock_change: bool = True
    require_full_on_trust_policy_change: bool = True
    require_full_on_schema_or_canonicalization_change: bool = True
    require_full_on_environment_change: bool = True
    require_full_on_cache_corruption: bool = True
    require_full_on_first_state: bool = True
    require_full_on_missing_parent: bool = True
    allow_incremental_override: bool = False
    policy_id: str = "checkpoint-policy/default"
    schema: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "full_checkpoint_every_n_seals",
            _require_positive_int(
                self.full_checkpoint_every_n_seals,
                "full_checkpoint_every_n_seals",
            ),
        )
        object.__setattr__(
            self,
            "max_delta_chain_depth",
            _require_positive_int(
                self.max_delta_chain_depth, "max_delta_chain_depth"
            ),
        )
        object.__setattr__(
            self,
            "min_reuse_ratio_basis_points",
            _require_bps(
                self.min_reuse_ratio_basis_points,
                "min_reuse_ratio_basis_points",
            ),
        )
        for field in (
            "require_full_on_release_tag",
            "require_full_on_circuit_or_key_change",
            "require_full_on_dependency_lock_change",
            "require_full_on_trust_policy_change",
            "require_full_on_schema_or_canonicalization_change",
            "require_full_on_environment_change",
            "require_full_on_cache_corruption",
            "require_full_on_first_state",
            "require_full_on_missing_parent",
            "allow_incremental_override",
        ):
            object.__setattr__(
                self, field, _require_bool(getattr(self, field), field)
            )
        # Hard fail-closed: incremental callers may never override triggers.
        if self.allow_incremental_override:
            raise CheckpointPolicyError(
                "allow_incremental_override must be false; checkpoint triggers "
                "cannot be overridden by an incremental caller"
            )
        if self.schema != POLICY_SCHEMA:
            raise CheckpointPolicyError(f"schema must be {POLICY_SCHEMA}")
        policy_id = str(self.policy_id).strip()
        if not policy_id:
            raise CheckpointPolicyError("policy_id must be non-empty")
        object.__setattr__(self, "policy_id", policy_id)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": EVIDENCE_SUBSET,
            "policy_id": self.policy_id,
            "full_checkpoint_every_n_seals": self.full_checkpoint_every_n_seals,
            "max_delta_chain_depth": self.max_delta_chain_depth,
            "min_reuse_ratio_basis_points": self.min_reuse_ratio_basis_points,
            "require_full_on_release_tag": self.require_full_on_release_tag,
            "require_full_on_circuit_or_key_change": (
                self.require_full_on_circuit_or_key_change
            ),
            "require_full_on_dependency_lock_change": (
                self.require_full_on_dependency_lock_change
            ),
            "require_full_on_trust_policy_change": (
                self.require_full_on_trust_policy_change
            ),
            "require_full_on_schema_or_canonicalization_change": (
                self.require_full_on_schema_or_canonicalization_change
            ),
            "require_full_on_environment_change": (
                self.require_full_on_environment_change
            ),
            "require_full_on_cache_corruption": (
                self.require_full_on_cache_corruption
            ),
            "require_full_on_first_state": self.require_full_on_first_state,
            "require_full_on_missing_parent": self.require_full_on_missing_parent,
            "allow_incremental_override": self.allow_incremental_override,
            "checkpoint_triggers": list(CHECKPOINT_TRIGGERS),
        }

    def policy_cid(self) -> str:
        return _cid(
            {
                "domain": "ips.checkpoint_policy.v1",
                "payload": self.to_canonical(),
            }
        )

    @classmethod
    def from_canonical(cls, payload: Mapping[str, Any]) -> CheckpointPolicy:
        if not isinstance(payload, Mapping):
            raise CheckpointPolicyError("CheckpointPolicy payload must be a mapping")
        return cls(
            full_checkpoint_every_n_seals=int(
                payload.get(
                    "full_checkpoint_every_n_seals",
                    DEFAULT_FULL_CHECKPOINT_EVERY_N_SEALS,
                )
            ),
            max_delta_chain_depth=int(
                payload.get(
                    "max_delta_chain_depth", DEFAULT_MAX_DELTA_CHAIN_DEPTH
                )
            ),
            min_reuse_ratio_basis_points=int(
                payload.get(
                    "min_reuse_ratio_basis_points",
                    DEFAULT_MIN_REUSE_RATIO_BASIS_POINTS,
                )
            ),
            require_full_on_release_tag=bool(
                payload.get("require_full_on_release_tag", True)
            ),
            require_full_on_circuit_or_key_change=bool(
                payload.get("require_full_on_circuit_or_key_change", True)
            ),
            require_full_on_dependency_lock_change=bool(
                payload.get("require_full_on_dependency_lock_change", True)
            ),
            require_full_on_trust_policy_change=bool(
                payload.get("require_full_on_trust_policy_change", True)
            ),
            require_full_on_schema_or_canonicalization_change=bool(
                payload.get(
                    "require_full_on_schema_or_canonicalization_change", True
                )
            ),
            require_full_on_environment_change=bool(
                payload.get("require_full_on_environment_change", True)
            ),
            require_full_on_cache_corruption=bool(
                payload.get("require_full_on_cache_corruption", True)
            ),
            require_full_on_first_state=bool(
                payload.get("require_full_on_first_state", True)
            ),
            require_full_on_missing_parent=bool(
                payload.get("require_full_on_missing_parent", True)
            ),
            allow_incremental_override=bool(
                payload.get("allow_incremental_override", False)
            ),
            policy_id=str(payload.get("policy_id") or "checkpoint-policy/default"),
            schema=str(payload.get("schema") or POLICY_SCHEMA),
        )

    @classmethod
    def default(cls) -> CheckpointPolicy:
        """Production-oriented fail-closed defaults."""

        return cls()


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationState:
    """Observed chain and change-context facts for one decision."""

    seals_since_last_full_checkpoint: int = 0
    delta_chain_depth: int = 0
    estimated_reuse_ratio_basis_points: int | None = None
    has_accepted_parent: bool = True
    is_first_state: bool = False
    is_release_tag: bool = False
    circuit_or_key_changed: bool = False
    dependency_lock_changed: bool = False
    trust_policy_changed: bool = False
    schema_changed: bool = False
    canonicalization_changed: bool = False
    environment_changed: bool = False
    cache_corruption_detected: bool = False
    uncertain_cache_integrity: bool = False
    force_full_checkpoint: bool = False
    full_fallback_required: bool = False
    # Caller preference is advisory only and never suppresses a trigger.
    prefer_incremental: bool = False
    schema: str = STATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "seals_since_last_full_checkpoint",
            _require_nonneg_int(
                self.seals_since_last_full_checkpoint,
                "seals_since_last_full_checkpoint",
            ),
        )
        object.__setattr__(
            self,
            "delta_chain_depth",
            _require_nonneg_int(self.delta_chain_depth, "delta_chain_depth"),
        )
        if self.estimated_reuse_ratio_basis_points is not None:
            object.__setattr__(
                self,
                "estimated_reuse_ratio_basis_points",
                _require_bps(
                    self.estimated_reuse_ratio_basis_points,
                    "estimated_reuse_ratio_basis_points",
                ),
            )
        for field in (
            "has_accepted_parent",
            "is_first_state",
            "is_release_tag",
            "circuit_or_key_changed",
            "dependency_lock_changed",
            "trust_policy_changed",
            "schema_changed",
            "canonicalization_changed",
            "environment_changed",
            "cache_corruption_detected",
            "uncertain_cache_integrity",
            "force_full_checkpoint",
            "full_fallback_required",
            "prefer_incremental",
        ):
            object.__setattr__(
                self, field, _require_bool(getattr(self, field), field)
            )
        if self.schema != STATE_SCHEMA:
            raise CheckpointPolicyError(f"state schema must be {STATE_SCHEMA}")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seals_since_last_full_checkpoint": (
                self.seals_since_last_full_checkpoint
            ),
            "delta_chain_depth": self.delta_chain_depth,
            "estimated_reuse_ratio_basis_points": (
                self.estimated_reuse_ratio_basis_points
            ),
            "has_accepted_parent": self.has_accepted_parent,
            "is_first_state": self.is_first_state,
            "is_release_tag": self.is_release_tag,
            "circuit_or_key_changed": self.circuit_or_key_changed,
            "dependency_lock_changed": self.dependency_lock_changed,
            "trust_policy_changed": self.trust_policy_changed,
            "schema_changed": self.schema_changed,
            "canonicalization_changed": self.canonicalization_changed,
            "environment_changed": self.environment_changed,
            "cache_corruption_detected": self.cache_corruption_detected,
            "uncertain_cache_integrity": self.uncertain_cache_integrity,
            "force_full_checkpoint": self.force_full_checkpoint,
            "full_fallback_required": self.full_fallback_required,
            "prefer_incremental": self.prefer_incremental,
        }

    @classmethod
    def from_canonical(cls, payload: Mapping[str, Any]) -> CheckpointEvaluationState:
        if not isinstance(payload, Mapping):
            raise CheckpointPolicyError(
                "CheckpointEvaluationState payload must be a mapping"
            )
        reuse = payload.get("estimated_reuse_ratio_basis_points")
        return cls(
            seals_since_last_full_checkpoint=int(
                payload.get("seals_since_last_full_checkpoint", 0)
            ),
            delta_chain_depth=int(payload.get("delta_chain_depth", 0)),
            estimated_reuse_ratio_basis_points=(
                None if reuse is None else int(reuse)
            ),
            has_accepted_parent=bool(payload.get("has_accepted_parent", True)),
            is_first_state=bool(payload.get("is_first_state", False)),
            is_release_tag=bool(payload.get("is_release_tag", False)),
            circuit_or_key_changed=bool(
                payload.get("circuit_or_key_changed", False)
            ),
            dependency_lock_changed=bool(
                payload.get("dependency_lock_changed", False)
            ),
            trust_policy_changed=bool(payload.get("trust_policy_changed", False)),
            schema_changed=bool(payload.get("schema_changed", False)),
            canonicalization_changed=bool(
                payload.get("canonicalization_changed", False)
            ),
            environment_changed=bool(payload.get("environment_changed", False)),
            cache_corruption_detected=bool(
                payload.get("cache_corruption_detected", False)
            ),
            uncertain_cache_integrity=bool(
                payload.get("uncertain_cache_integrity", False)
            ),
            force_full_checkpoint=bool(
                payload.get("force_full_checkpoint", False)
            ),
            full_fallback_required=bool(
                payload.get("full_fallback_required", False)
            ),
            prefer_incremental=bool(payload.get("prefer_incremental", False)),
            schema=str(payload.get("schema") or STATE_SCHEMA),
        )


@dataclass(frozen=True, slots=True)
class CheckpointDecision:
    """Deterministic full-versus-incremental decision under a policy."""

    schema: str
    evidence_subset: str
    mode: CheckpointMode
    require_full_checkpoint: bool
    allow_incremental: bool
    reasons: tuple[str, ...]
    policy_cid: str
    policy_id: str
    incremental_override_attempted: bool
    incremental_override_honored: bool
    seals_since_last_full_checkpoint: int
    delta_chain_depth: int
    estimated_reuse_ratio_basis_points: int | None
    full_checkpoint_every_n_seals: int
    max_delta_chain_depth: int
    min_reuse_ratio_basis_points: int

    def __post_init__(self) -> None:
        if self.schema != DECISION_SCHEMA:
            raise CheckpointPolicyError(f"schema must be {DECISION_SCHEMA}")
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise CheckpointPolicyError(
                f"evidence_subset must be {EVIDENCE_SUBSET}"
            )
        if type(self.require_full_checkpoint) is not bool:
            raise CheckpointPolicyError("require_full_checkpoint must be a boolean")
        if type(self.allow_incremental) is not bool:
            raise CheckpointPolicyError("allow_incremental must be a boolean")
        if self.require_full_checkpoint and self.allow_incremental:
            raise CheckpointPolicyError(
                "full checkpoint requirement cannot allow incremental"
            )
        if self.require_full_checkpoint and self.mode is not CheckpointMode.FULL_CHECKPOINT:
            raise CheckpointPolicyError(
                "require_full_checkpoint requires mode full_checkpoint"
            )
        if not self.require_full_checkpoint and self.mode is not CheckpointMode.INCREMENTAL:
            raise CheckpointPolicyError(
                "allow_incremental requires mode incremental"
            )
        if self.incremental_override_honored:
            raise CheckpointPolicyError(
                "incremental override must never be honored"
            )
        # Stable sorted unique reasons.
        ordered = tuple(sorted(set(self.reasons)))
        for reason in ordered:
            parse_checkpoint_trigger(reason)
        object.__setattr__(self, "reasons", ordered)

    @property
    def full_checkpoint_required(self) -> bool:
        return self.require_full_checkpoint

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "mode": self.mode.value,
            "require_full_checkpoint": self.require_full_checkpoint,
            "allow_incremental": self.allow_incremental,
            "reasons": list(self.reasons),
            "policy_cid": self.policy_cid,
            "policy_id": self.policy_id,
            "incremental_override_attempted": self.incremental_override_attempted,
            "incremental_override_honored": self.incremental_override_honored,
            "seals_since_last_full_checkpoint": (
                self.seals_since_last_full_checkpoint
            ),
            "delta_chain_depth": self.delta_chain_depth,
            "estimated_reuse_ratio_basis_points": (
                self.estimated_reuse_ratio_basis_points
            ),
            "full_checkpoint_every_n_seals": self.full_checkpoint_every_n_seals,
            "max_delta_chain_depth": self.max_delta_chain_depth,
            "min_reuse_ratio_basis_points": self.min_reuse_ratio_basis_points,
            "checkpoint_triggers": list(CHECKPOINT_TRIGGERS),
        }

    def decision_cid(self) -> str:
        return _cid(
            {
                "domain": "ips.checkpoint_decision.v1",
                "payload": self.to_canonical(),
            }
        )


def _coerce_policy(
    policy: CheckpointPolicy | Mapping[str, Any] | None,
) -> CheckpointPolicy:
    if policy is None:
        return CheckpointPolicy.default()
    if isinstance(policy, CheckpointPolicy):
        return policy
    if isinstance(policy, Mapping):
        return CheckpointPolicy.from_canonical(policy)
    raise CheckpointPolicyError(
        "policy must be CheckpointPolicy, mapping, or None"
    )


def _coerce_state(
    state: CheckpointEvaluationState | Mapping[str, Any],
) -> CheckpointEvaluationState:
    if isinstance(state, CheckpointEvaluationState):
        return state
    if isinstance(state, Mapping):
        return CheckpointEvaluationState.from_canonical(state)
    raise CheckpointPolicyError(
        "state must be CheckpointEvaluationState or mapping"
    )


def evaluate_checkpoint_policy(
    policy: CheckpointPolicy | Mapping[str, Any] | None,
    state: CheckpointEvaluationState | Mapping[str, Any],
) -> CheckpointDecision:
    """Evaluate whether a full checkpoint is mandatory.

    Fired triggers always force ``full_checkpoint``.  A caller that sets
    ``prefer_incremental`` cannot suppress any trigger.
    """

    active = _coerce_policy(policy)
    observed = _coerce_state(state)
    reasons: set[str] = set()

    if observed.is_first_state and active.require_full_on_first_state:
        reasons.add(CheckpointTrigger.FIRST_STATE.value)
    if (
        not observed.has_accepted_parent
        and active.require_full_on_missing_parent
    ):
        reasons.add(CheckpointTrigger.MISSING_PARENT.value)
    if (
        observed.seals_since_last_full_checkpoint
        >= active.full_checkpoint_every_n_seals
    ):
        reasons.add(CheckpointTrigger.PERIODIC_CADENCE.value)
    if observed.is_release_tag and active.require_full_on_release_tag:
        reasons.add(CheckpointTrigger.RELEASE_TAG.value)
    if (
        observed.circuit_or_key_changed
        and active.require_full_on_circuit_or_key_change
    ):
        reasons.add(CheckpointTrigger.CIRCUIT_OR_KEY_CHANGE.value)
    if (
        observed.dependency_lock_changed
        and active.require_full_on_dependency_lock_change
    ):
        reasons.add(CheckpointTrigger.DEPENDENCY_LOCK_CHANGE.value)
    if (
        observed.trust_policy_changed
        and active.require_full_on_trust_policy_change
    ):
        reasons.add(CheckpointTrigger.TRUST_POLICY_CHANGE.value)
    if active.require_full_on_schema_or_canonicalization_change:
        if observed.schema_changed:
            reasons.add(CheckpointTrigger.SCHEMA_CHANGE.value)
        if observed.canonicalization_changed:
            reasons.add(CheckpointTrigger.CANONICALIZATION_CHANGE.value)
    if (
        observed.environment_changed
        and active.require_full_on_environment_change
    ):
        reasons.add(CheckpointTrigger.ENVIRONMENT_CHANGE.value)
    if active.require_full_on_cache_corruption:
        if observed.cache_corruption_detected:
            reasons.add(CheckpointTrigger.CACHE_CORRUPTION.value)
        if observed.uncertain_cache_integrity:
            reasons.add(CheckpointTrigger.UNCERTAIN_CACHE_INTEGRITY.value)
    if observed.delta_chain_depth >= active.max_delta_chain_depth:
        reasons.add(CheckpointTrigger.EXCESSIVE_DELTA_CHAIN_DEPTH.value)
    if (
        observed.estimated_reuse_ratio_basis_points is not None
        and observed.estimated_reuse_ratio_basis_points
        < active.min_reuse_ratio_basis_points
    ):
        reasons.add(CheckpointTrigger.LOW_REUSE_RATIO.value)
    if observed.force_full_checkpoint:
        reasons.add(CheckpointTrigger.EXPLICIT_FORCE.value)
    if observed.full_fallback_required:
        reasons.add(CheckpointTrigger.FULL_FALLBACK_REQUIRED.value)

    ordered_reasons = tuple(sorted(reasons))
    require_full = bool(ordered_reasons)
    # prefer_incremental is recorded but never honored against a trigger.
    override_attempted = bool(observed.prefer_incremental and require_full)

    return CheckpointDecision(
        schema=DECISION_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        mode=(
            CheckpointMode.FULL_CHECKPOINT
            if require_full
            else CheckpointMode.INCREMENTAL
        ),
        require_full_checkpoint=require_full,
        allow_incremental=not require_full,
        reasons=ordered_reasons,
        policy_cid=active.policy_cid(),
        policy_id=active.policy_id,
        incremental_override_attempted=override_attempted,
        incremental_override_honored=False,
        seals_since_last_full_checkpoint=(
            observed.seals_since_last_full_checkpoint
        ),
        delta_chain_depth=observed.delta_chain_depth,
        estimated_reuse_ratio_basis_points=(
            observed.estimated_reuse_ratio_basis_points
        ),
        full_checkpoint_every_n_seals=active.full_checkpoint_every_n_seals,
        max_delta_chain_depth=active.max_delta_chain_depth,
        min_reuse_ratio_basis_points=active.min_reuse_ratio_basis_points,
    )


def decide_checkpoint(
    *,
    policy: CheckpointPolicy | Mapping[str, Any] | None = None,
    seals_since_last_full_checkpoint: int = 0,
    delta_chain_depth: int = 0,
    estimated_reuse_ratio_basis_points: int | None = None,
    has_accepted_parent: bool = True,
    is_first_state: bool = False,
    is_release_tag: bool = False,
    circuit_or_key_changed: bool = False,
    dependency_lock_changed: bool = False,
    trust_policy_changed: bool = False,
    schema_changed: bool = False,
    canonicalization_changed: bool = False,
    environment_changed: bool = False,
    cache_corruption_detected: bool = False,
    uncertain_cache_integrity: bool = False,
    force_full_checkpoint: bool = False,
    full_fallback_required: bool = False,
    prefer_incremental: bool = False,
) -> CheckpointDecision:
    """Convenience facade over :func:`evaluate_checkpoint_policy`."""

    return evaluate_checkpoint_policy(
        policy,
        CheckpointEvaluationState(
            seals_since_last_full_checkpoint=seals_since_last_full_checkpoint,
            delta_chain_depth=delta_chain_depth,
            estimated_reuse_ratio_basis_points=estimated_reuse_ratio_basis_points,
            has_accepted_parent=has_accepted_parent,
            is_first_state=is_first_state,
            is_release_tag=is_release_tag,
            circuit_or_key_changed=circuit_or_key_changed,
            dependency_lock_changed=dependency_lock_changed,
            trust_policy_changed=trust_policy_changed,
            schema_changed=schema_changed,
            canonicalization_changed=canonicalization_changed,
            environment_changed=environment_changed,
            cache_corruption_detected=cache_corruption_detected,
            uncertain_cache_integrity=uncertain_cache_integrity,
            force_full_checkpoint=force_full_checkpoint,
            full_fallback_required=full_fallback_required,
            prefer_incremental=prefer_incremental,
        ),
    )


__all__ = (
    "CHECKPOINT_TRIGGERS",
    "DECISION_SCHEMA",
    "DEFAULT_FULL_CHECKPOINT_EVERY_N_SEALS",
    "DEFAULT_MAX_DELTA_CHAIN_DEPTH",
    "DEFAULT_MIN_REUSE_RATIO_BASIS_POINTS",
    "EVIDENCE_SUBSET",
    "POLICY_SCHEMA",
    "STATE_SCHEMA",
    "CheckpointDecision",
    "CheckpointEvaluationState",
    "CheckpointMode",
    "CheckpointPolicy",
    "CheckpointPolicyError",
    "CheckpointTrigger",
    "decide_checkpoint",
    "evaluate_checkpoint_policy",
    "parse_checkpoint_trigger",
)
