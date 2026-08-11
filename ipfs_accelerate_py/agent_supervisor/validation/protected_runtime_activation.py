"""ASE3-026 protected runtime activation authorization and observation.

Two-evidence operator gate:

1. **Pre-effect authorization**
   (``protected-runtime-activation-authorization@1``) binds an inactive exact
   tree, old generation, target old+1 CAS/lease, guardian, bounded flags, and
   expiry. It **must** state ``authorization_effect_observed: false`` and must
   never claim a birth, heartbeat, cursor, refill, reload, or completion.

2. **Post-activation observation**
   (``protected-runtime-post-activation-observation@1``) joins actual same-
   generation lifecycle and monitor births, leases, fences, heartbeats,
   cursors, and refill append→recompile→dispatch/adoption. Authorization alone
   never proves the effect ran and cannot make public facades selectable.

Only a :class:`ReviewedHostNamespaceReconciler` may consume a validated
authorization to enable one exact old+1 generation. Retries adopt that winner.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

ACTIVATION_AUTHORIZATION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-runtime-activation-authorization@1"
)
POST_ACTIVATION_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "protected-runtime-post-activation-observation@1"
)
RUNTIME_GENERATION_CAS_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.runtime-generation-activation-cas@1"
)
GATE_GENERATOR: Final = (
    "ipfs_accelerate_py.agent_supervisor.protected-runtime-activation-gate@1"
)
BOARD_NAMESPACE: Final = "agent-supervisor-prompt-only-self-improvement-v3"
ACTIVATION_TASK_ID: Final = "ASE3-026"
GUARDIAN_TYPE: Final = "ReviewedHostNamespaceReconciler"

REQUIRED_POST_ACTIVATION_OBSERVATIONS: Final = (
    "lifecycle_process_birth",
    "lifecycle_lease_fence_heartbeat_and_cursor",
    "monitor_process_birth",
    "monitor_lease_fence_heartbeat_and_cursor",
    "refill_append_recompile_dispatch_or_adoption",
)

# Forbidden effect-claim keys inside a pre-effect authorization body.
_FORBIDDEN_AUTH_EFFECT_KEYS: Final = frozenset(
    {
        "lifecycle_process_birth",
        "monitor_process_birth",
        "heartbeat_at_ms",
        "event_cursor",
        "refill_dispatch",
        "refill_adoption",
        "completion_claimed",
        "already_active",
        "runtime_effect_started",
        "process_birth",
        "activation_effect_observed",
    }
)

_AUTHORIZATION_TOP_LEVEL: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "task_id",
    "receipt_phase",
    "authorization_id",
    "authorization_effect_observed",
    "inactive_tree",
    "old_generation",
    "target_generation",
    "cas_lease",
    "guardian",
    "bounded_flags",
    "quiescence",
    "expiry_at",
    "operator_review",
    "denials",
)

_OBSERVATION_TOP_LEVEL: Final = (
    "schema",
    "created_at",
    "board_namespace",
    "task_id",
    "observation_id",
    "authorization_binding",
    "target_generation",
    "lifecycle",
    "monitor",
    "refill",
    "joined_running",
    "required_observations",
    "operator_review",
    "denials",
)


class ProtectedRuntimeActivationError(ValueError):
    """Fail-closed activation gate violation."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtectedRuntimeActivationError(f"{label}: expected object")
    return value


def _require_str(value: Any, *, label: str, non_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ProtectedRuntimeActivationError(f"{label}: expected string")
    if non_empty and not value.strip():
        raise ProtectedRuntimeActivationError(f"{label}: non-empty string required")
    return value


def _require_bool(value: Any, *, label: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise ProtectedRuntimeActivationError(
            f"{label}: expected exact JSON boolean"
        )
    if expected is not None and value is not expected:
        raise ProtectedRuntimeActivationError(
            f"{label}: expected {str(expected).lower()}"
        )
    return value


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtectedRuntimeActivationError(f"{label}: expected integer")
    if minimum is not None and value < minimum:
        raise ProtectedRuntimeActivationError(f"{label}: expected >= {minimum}")
    return value


def _utc_ok(value: Any, *, label: str) -> str:
    text = _require_str(value, label=label)
    # Strict Zulu form YYYY-MM-DDTHH:MM:SSZ
    if len(text) != 20 or text[10] != "T" or not text.endswith("Z"):
        raise ProtectedRuntimeActivationError(f"{label}: UTC Zulu timestamp required")
    return text


def activation_authorization_id(payload: Mapping[str, Any]) -> str:
    """Derive authorization identity without a self-selected ID cycle."""

    body = dict(payload)
    body.pop("authorization_id", None)
    review = body.get("operator_review")
    if isinstance(review, Mapping):
        unsigned = dict(review)
        unsigned.pop("signature", None)
        body["operator_review"] = unsigned
    return _canonical_sha256(body)


def post_activation_observation_id(payload: Mapping[str, Any]) -> str:
    """Derive observation identity without a self-selected ID cycle."""

    body = dict(payload)
    body.pop("observation_id", None)
    review = body.get("operator_review")
    if isinstance(review, Mapping):
        unsigned = dict(review)
        unsigned.pop("signature", None)
        body["operator_review"] = unsigned
    return _canonical_sha256(body)


@dataclass(frozen=True)
class RuntimeGenerationActivationCAS:
    """One exact old+1 generation CAS/lease identity."""

    schema: str
    old_generation: int
    target_generation: int
    lease_id: str
    cas_token: str
    tree_id: str
    guardian_identity: str
    host_namespace: str

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_GENERATION_CAS_SCHEMA:
            raise ProtectedRuntimeActivationError("unsupported generation CAS schema")
        if self.old_generation < 0:
            raise ProtectedRuntimeActivationError("old_generation must be >= 0")
        if self.target_generation != self.old_generation + 1:
            raise ProtectedRuntimeActivationError(
                "target_generation must equal old_generation + 1"
            )
        if not self.lease_id or not self.cas_token or not self.tree_id:
            raise ProtectedRuntimeActivationError("CAS identity is incomplete")
        if not self.guardian_identity or not self.host_namespace:
            raise ProtectedRuntimeActivationError("guardian identity is incomplete")

    @property
    def content_id(self) -> str:
        return _canonical_sha256(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessJoinEvidence:
    """One process side of the lifecycle/monitor join."""

    role: str
    process_cid: str
    process_birth_identity: str
    lease_id: str
    fencing_generation: int
    heartbeat_at_ms: int
    event_cursor: str
    generation: int
    healthy: bool

    def __post_init__(self) -> None:
        if self.role not in {"lifecycle", "monitor"}:
            raise ProtectedRuntimeActivationError(f"unknown process role: {self.role}")
        if not all(
            (
                self.process_cid,
                self.process_birth_identity,
                self.lease_id,
                self.event_cursor,
            )
        ):
            raise ProtectedRuntimeActivationError(
                f"{self.role}: incomplete process evidence"
            )
        if self.fencing_generation < 1 or self.generation < 1:
            raise ProtectedRuntimeActivationError(
                f"{self.role}: generation/fence must be positive"
            )
        if self.heartbeat_at_ms <= 0:
            raise ProtectedRuntimeActivationError(
                f"{self.role}: heartbeat_at_ms must be positive"
            )
        if type(self.healthy) is not bool:
            raise ProtectedRuntimeActivationError(
                f"{self.role}: healthy must be exact boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessJoinEvidence":
        return cls(
            role=str(value.get("role") or ""),
            process_cid=str(value.get("process_cid") or ""),
            process_birth_identity=str(value.get("process_birth_identity") or ""),
            lease_id=str(value.get("lease_id") or ""),
            fencing_generation=int(value.get("fencing_generation") or 0),
            heartbeat_at_ms=int(value.get("heartbeat_at_ms") or 0),
            event_cursor=str(value.get("event_cursor") or ""),
            generation=int(value.get("generation") or 0),
            healthy=bool(value.get("healthy")),
        )


@dataclass(frozen=True)
class RefillActivationEvidence:
    """Observed refill saga terminal for the activated generation."""

    plan_root_cid: str
    logical_attempt_id: str
    phase: str
    epoch: int
    disposition: str
    generation: int

    def __post_init__(self) -> None:
        if self.phase not in {"DISPATCHED", "ADOPTED"}:
            raise ProtectedRuntimeActivationError(
                "refill observation requires DISPATCHED or ADOPTED terminal"
            )
        if self.disposition not in {"dispatched", "adopted", "terminal"}:
            raise ProtectedRuntimeActivationError(
                "refill disposition must be dispatched/adopted/terminal"
            )
        if not self.plan_root_cid or not self.logical_attempt_id:
            raise ProtectedRuntimeActivationError("refill identity incomplete")
        if self.epoch < 0 or self.generation < 1:
            raise ProtectedRuntimeActivationError("refill epoch/generation invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RefillActivationEvidence":
        return cls(
            plan_root_cid=str(value.get("plan_root_cid") or ""),
            logical_attempt_id=str(value.get("logical_attempt_id") or ""),
            phase=str(value.get("phase") or ""),
            epoch=int(value.get("epoch") or 0),
            disposition=str(value.get("disposition") or ""),
            generation=int(value.get("generation") or 0),
        )


def validate_activation_authorization(
    payload: Mapping[str, Any],
    *,
    now_ms: int | None = None,
) -> tuple[str, ...]:
    """Fail-closed pre-effect authorization validation.

    Any birth/heartbeat/cursor/refill/completion claim fails closed.
    """

    errors: list[str] = []
    prefix = "protected_runtime_activation.authorization"

    if set(payload) != set(_AUTHORIZATION_TOP_LEVEL):
        errors.append(f"{prefix}: exact top-level fields required")
        # Continue best-effort so callers see structural issues too.

    if payload.get("schema") != ACTIVATION_AUTHORIZATION_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    if payload.get("task_id") != ACTIVATION_TASK_ID:
        errors.append(f"{prefix}.task_id: expected {ACTIVATION_TASK_ID}")
    if payload.get("receipt_phase") != "pre_effect_authorization":
        errors.append(f"{prefix}.receipt_phase: pre_effect_authorization required")

    try:
        _utc_ok(payload.get("created_at"), label=f"{prefix}.created_at")
    except ProtectedRuntimeActivationError as exc:
        errors.append(str(exc))
    try:
        expiry = _utc_ok(payload.get("expiry_at"), label=f"{prefix}.expiry_at")
    except ProtectedRuntimeActivationError as exc:
        errors.append(str(exc))
        expiry = ""

    try:
        effect_observed = _require_bool(
            payload.get("authorization_effect_observed"),
            label=f"{prefix}.authorization_effect_observed",
            expected=False,
        )
    except ProtectedRuntimeActivationError as exc:
        errors.append(str(exc))
        effect_observed = True  # fail closed

    if effect_observed is not False:
        errors.append(
            f"{prefix}.authorization_effect_observed: must be false "
            "(authorization alone never proves the effect ran)"
        )

    # Forbidden effect keys anywhere in the authorization body.
    serialized = json.dumps(payload, sort_keys=True, default=str)
    for key in sorted(_FORBIDDEN_AUTH_EFFECT_KEYS):
        # authorization_effect_observed is allowed as the explicit false claim.
        if key == "activation_effect_observed":
            continue
        if f'"{key}"' in serialized and key != "authorization_effect_observed":
            # Only flag when the key appears as a true-ish claim path outside denials.
            pass
    # Stronger structural scan of nested maps for effect claims.
    for path, value in _walk(payload):
        leaf = path.rsplit(".", 1)[-1]
        if leaf in _FORBIDDEN_AUTH_EFFECT_KEYS and path != (
            "authorization_effect_observed"
        ):
            if value not in (False, None, "", 0):
                errors.append(
                    f"{prefix}: forbidden effect claim {path}={value!r}"
                )

    tree = payload.get("inactive_tree")
    if not isinstance(tree, Mapping):
        errors.append(f"{prefix}.inactive_tree: expected object")
    else:
        for field in ("head", "tree", "branch"):
            if not isinstance(tree.get(field), str) or not str(tree.get(field)).strip():
                errors.append(f"{prefix}.inactive_tree.{field}: required")
        if tree.get("active_owned_effects") not in (0, False):
            if tree.get("active_owned_effects") != 0:
                errors.append(
                    f"{prefix}.inactive_tree.active_owned_effects: must be zero"
                )
        if tree.get("quiescent") is not True:
            errors.append(f"{prefix}.inactive_tree.quiescent: must be true")

    try:
        old_gen = _require_int(
            payload.get("old_generation"),
            label=f"{prefix}.old_generation",
            minimum=0,
        )
        target_gen = _require_int(
            payload.get("target_generation"),
            label=f"{prefix}.target_generation",
            minimum=1,
        )
        if target_gen != old_gen + 1:
            errors.append(
                f"{prefix}.target_generation: must equal old_generation + 1"
            )
    except ProtectedRuntimeActivationError as exc:
        errors.append(str(exc))

    cas = payload.get("cas_lease")
    if not isinstance(cas, Mapping):
        errors.append(f"{prefix}.cas_lease: expected object")
    else:
        for field in ("lease_id", "cas_token", "tree_id"):
            if not isinstance(cas.get(field), str) or not str(cas.get(field)).strip():
                errors.append(f"{prefix}.cas_lease.{field}: required")
        if cas.get("one_generation_winner_required") is not True:
            errors.append(
                f"{prefix}.cas_lease.one_generation_winner_required: must be true"
            )

    guardian = payload.get("guardian")
    if not isinstance(guardian, Mapping):
        errors.append(f"{prefix}.guardian: expected object")
    else:
        if guardian.get("type") != GUARDIAN_TYPE:
            errors.append(f"{prefix}.guardian.type: {GUARDIAN_TYPE} required")
        for field in ("guardian_identity", "host_namespace", "review_cid"):
            if not isinstance(guardian.get(field), str) or not str(
                guardian.get(field)
            ).strip():
                errors.append(f"{prefix}.guardian.{field}: required")
        if guardian.get("review_required") is not True:
            errors.append(f"{prefix}.guardian.review_required: must be true")

    flags = payload.get("bounded_flags")
    if not isinstance(flags, Mapping):
        errors.append(f"{prefix}.bounded_flags: expected object")
    else:
        # Pre-effect authorization describes the *intended* post-activation
        # flag transition; it must not claim they are already active.
        if flags.get("already_active") is True:
            errors.append(f"{prefix}.bounded_flags.already_active: must not be true")
        if flags.get("codebase_refill_enabled") is not False:
            errors.append(
                f"{prefix}.bounded_flags.codebase_refill_enabled: must stay false"
            )
        if flags.get("prompt_program_refill_enabled_after_activation") is not True:
            errors.append(
                f"{prefix}.bounded_flags.prompt_program_refill_enabled_after_activation: "
                "must be true"
            )
        if flags.get("objective_refill_enabled_after_activation") is not True:
            errors.append(
                f"{prefix}.bounded_flags.objective_refill_enabled_after_activation: "
                "must be true"
            )
        if flags.get("monitor_enabled_after_activation") is not True:
            errors.append(
                f"{prefix}.bounded_flags.monitor_enabled_after_activation: must be true"
            )
        if flags.get("legacy_hash_sharding_for_active_slices") is not False:
            errors.append(
                f"{prefix}.bounded_flags.legacy_hash_sharding_for_active_slices: "
                "must stay false"
            )

    quiescence = payload.get("quiescence")
    if not isinstance(quiescence, Mapping):
        errors.append(f"{prefix}.quiescence: expected object")
    else:
        if quiescence.get("zero_old_generation_descendants") is not True:
            errors.append(
                f"{prefix}.quiescence.zero_old_generation_descendants: required"
            )
        if quiescence.get("zero_owned_worker_provider_merge_effects") is not True:
            errors.append(
                f"{prefix}.quiescence.zero_owned_worker_provider_merge_effects: required"
            )
        if quiescence.get("refill_and_monitor_dormant") is not True:
            errors.append(
                f"{prefix}.quiescence.refill_and_monitor_dormant: required"
            )

    denials = payload.get("denials")
    if not isinstance(denials, Mapping):
        errors.append(f"{prefix}.denials: expected object")
    else:
        expected_denials = {
            "authorization_may_claim_activation_effect": False,
            "authorization_alone_proves_effect_ran": False,
            "public_facade_selectable_from_authorization": False,
            "broad_legacy_codebase_refill_allowed": False,
            "self_attested_guardian_allowed": False,
            "client_owned_monitor_allowed": False,
        }
        for key, expected in expected_denials.items():
            if denials.get(key) is not expected:
                errors.append(f"{prefix}.denials.{key}: expected {expected!r}")

    review = payload.get("operator_review")
    if not isinstance(review, Mapping):
        errors.append(f"{prefix}.operator_review: expected object")
    else:
        if review.get("required") is not True:
            errors.append(f"{prefix}.operator_review.required: must be true")
        if not isinstance(review.get("reviewer_identity"), str) or not str(
            review.get("reviewer_identity")
        ).strip():
            errors.append(f"{prefix}.operator_review.reviewer_identity: required")
        if review.get("self_review_allowed") is not False:
            errors.append(
                f"{prefix}.operator_review.self_review_allowed: must be false"
            )

    observed_id = payload.get("authorization_id")
    if not isinstance(observed_id, str) or not observed_id.startswith("sha256:"):
        errors.append(f"{prefix}.authorization_id: sha256 digest required")
    else:
        try:
            expected_id = activation_authorization_id(payload)
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}.authorization_id: {exc}")
        else:
            if observed_id != expected_id:
                errors.append(
                    f"{prefix}.authorization_id: canonical identity mismatch"
                )

    if expiry and now_ms is not None:
        try:
            # Parse Zulu without importing datetime if possible
            from datetime import datetime, timezone

            expiry_ms = int(
                datetime.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                * 1000
            )
            if now_ms > expiry_ms:
                errors.append(f"{prefix}.expiry_at: authorization expired")
        except ValueError:
            errors.append(f"{prefix}.expiry_at: unparsable")

    return tuple(errors)


def validate_post_activation_observation(
    payload: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
    authorization_sha256: str | None = None,
) -> tuple[str, ...]:
    """Validate joined post-activation observation evidence."""

    errors: list[str] = []
    prefix = "protected_runtime_activation.observation"

    if set(payload) != set(_OBSERVATION_TOP_LEVEL):
        errors.append(f"{prefix}: exact top-level fields required")

    if payload.get("schema") != POST_ACTIVATION_OBSERVATION_SCHEMA:
        errors.append(f"{prefix}.schema: unsupported schema")
    if payload.get("board_namespace") != BOARD_NAMESPACE:
        errors.append(f"{prefix}.board_namespace: mismatch")
    if payload.get("task_id") != ACTIVATION_TASK_ID:
        errors.append(f"{prefix}.task_id: expected {ACTIVATION_TASK_ID}")

    try:
        _utc_ok(payload.get("created_at"), label=f"{prefix}.created_at")
    except ProtectedRuntimeActivationError as exc:
        errors.append(str(exc))

    binding = payload.get("authorization_binding")
    if not isinstance(binding, Mapping):
        errors.append(f"{prefix}.authorization_binding: expected object")
    else:
        if not isinstance(binding.get("authorization_id"), str):
            errors.append(f"{prefix}.authorization_binding.authorization_id: required")
        if not isinstance(binding.get("authorization_sha256"), str):
            errors.append(
                f"{prefix}.authorization_binding.authorization_sha256: required"
            )
        if binding.get("authorization_alone_proves_effect") is not False:
            errors.append(
                f"{prefix}.authorization_binding.authorization_alone_proves_effect: "
                "must be false"
            )
        if authorization is not None:
            expected_id = authorization.get("authorization_id")
            if binding.get("authorization_id") != expected_id:
                errors.append(
                    f"{prefix}.authorization_binding.authorization_id: mismatch"
                )
            if authorization.get("authorization_effect_observed") is not False:
                errors.append(
                    f"{prefix}.authorization_binding: authorization must remain "
                    "pre-effect (authorization_effect_observed:false)"
                )
        if (
            authorization_sha256 is not None
            and binding.get("authorization_sha256") != authorization_sha256
        ):
            errors.append(
                f"{prefix}.authorization_binding.authorization_sha256: mismatch"
            )

    try:
        target_gen = _require_int(
            payload.get("target_generation"),
            label=f"{prefix}.target_generation",
            minimum=1,
        )
    except ProtectedRuntimeActivationError as exc:
        errors.append(str(exc))
        target_gen = -1

    if authorization is not None:
        if payload.get("target_generation") != authorization.get("target_generation"):
            errors.append(
                f"{prefix}.target_generation: must match authorization target"
            )

    lifecycle_raw = payload.get("lifecycle")
    monitor_raw = payload.get("monitor")
    lifecycle: ProcessJoinEvidence | None = None
    monitor: ProcessJoinEvidence | None = None
    try:
        lifecycle = ProcessJoinEvidence.from_dict(
            _require_mapping(lifecycle_raw, label=f"{prefix}.lifecycle")
        )
        if lifecycle.role != "lifecycle":
            errors.append(f"{prefix}.lifecycle.role: must be 'lifecycle'")
        if target_gen > 0 and lifecycle.generation != target_gen:
            errors.append(
                f"{prefix}.lifecycle.generation: must match target_generation"
            )
        if lifecycle.healthy is not True:
            errors.append(f"{prefix}.lifecycle.healthy: must be true")
    except (ProtectedRuntimeActivationError, TypeError, ValueError) as exc:
        errors.append(f"{prefix}.lifecycle: {exc}")

    try:
        monitor = ProcessJoinEvidence.from_dict(
            _require_mapping(monitor_raw, label=f"{prefix}.monitor")
        )
        if monitor.role != "monitor":
            errors.append(f"{prefix}.monitor.role: must be 'monitor'")
        if target_gen > 0 and monitor.generation != target_gen:
            errors.append(
                f"{prefix}.monitor.generation: must match target_generation"
            )
        if monitor.healthy is not True:
            errors.append(f"{prefix}.monitor.healthy: must be true")
    except (ProtectedRuntimeActivationError, TypeError, ValueError) as exc:
        errors.append(f"{prefix}.monitor: {exc}")

    if (
        lifecycle is not None
        and monitor is not None
        and lifecycle.process_birth_identity == monitor.process_birth_identity
    ):
        errors.append(
            f"{prefix}: lifecycle and monitor births must be distinct identities"
        )

    try:
        refill = RefillActivationEvidence.from_dict(
            _require_mapping(payload.get("refill"), label=f"{prefix}.refill")
        )
        if target_gen > 0 and refill.generation != target_gen:
            errors.append(f"{prefix}.refill.generation: must match target_generation")
    except (ProtectedRuntimeActivationError, TypeError, ValueError) as exc:
        errors.append(f"{prefix}.refill: {exc}")

    joined = payload.get("joined_running")
    if not isinstance(joined, Mapping):
        errors.append(f"{prefix}.joined_running: expected object")
    else:
        if joined.get("joined") is not True:
            errors.append(f"{prefix}.joined_running.joined: must be true")
        if joined.get("same_generation") is not True:
            errors.append(f"{prefix}.joined_running.same_generation: must be true")
        required_fields = joined.get("join_fields")
        expected_join = [
            "lifecycle_process_birth",
            "lifecycle_lease",
            "lifecycle_fence",
            "lifecycle_heartbeat",
            "lifecycle_event_cursor",
            "monitor_process_birth",
            "monitor_lease",
            "monitor_fence",
            "monitor_heartbeat",
            "monitor_event_cursor",
        ]
        if required_fields != expected_join:
            errors.append(f"{prefix}.joined_running.join_fields: exact join required")

    required = payload.get("required_observations")
    if required != list(REQUIRED_POST_ACTIVATION_OBSERVATIONS):
        errors.append(
            f"{prefix}.required_observations: exact population required"
        )

    denials = payload.get("denials")
    if not isinstance(denials, Mapping):
        errors.append(f"{prefix}.denials: expected object")
    else:
        if denials.get("observation_retroactively_authorizes") is not False:
            errors.append(
                f"{prefix}.denials.observation_retroactively_authorizes: "
                "must be false"
            )
        if denials.get("authorization_alone_sufficient") is not False:
            errors.append(
                f"{prefix}.denials.authorization_alone_sufficient: must be false"
            )

    review = payload.get("operator_review")
    if not isinstance(review, Mapping):
        errors.append(f"{prefix}.operator_review: expected object")
    else:
        if review.get("required") is not True:
            errors.append(f"{prefix}.operator_review.required: must be true")
        if not isinstance(review.get("reviewer_identity"), str) or not str(
            review.get("reviewer_identity")
        ).strip():
            errors.append(f"{prefix}.operator_review.reviewer_identity: required")

    observed_id = payload.get("observation_id")
    if not isinstance(observed_id, str) or not observed_id.startswith("sha256:"):
        errors.append(f"{prefix}.observation_id: sha256 digest required")
    else:
        try:
            expected_id = post_activation_observation_id(payload)
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}.observation_id: {exc}")
        else:
            if observed_id != expected_id:
                errors.append(
                    f"{prefix}.observation_id: canonical identity mismatch"
                )

    return tuple(errors)


def _walk(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            items.append((path, child))
            items.extend(_walk(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            items.append((path, child))
            items.extend(_walk(child, path))
    return items


def build_activation_authorization(
    *,
    inactive_head: str,
    inactive_tree: str,
    branch: str,
    old_generation: int,
    lease_id: str,
    cas_token: str,
    tree_id: str,
    guardian_identity: str,
    host_namespace: str,
    review_cid: str,
    reviewer_identity: str,
    created_at: str | None = None,
    expiry_at: str | None = None,
) -> dict[str, Any]:
    """Build a pre-effect authorization body with derived authorization_id."""

    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0)
    created = created_at or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    expiry = expiry_at or (now + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    target = old_generation + 1
    payload: dict[str, Any] = {
        "schema": ACTIVATION_AUTHORIZATION_SCHEMA,
        "created_at": created,
        "board_namespace": BOARD_NAMESPACE,
        "task_id": ACTIVATION_TASK_ID,
        "receipt_phase": "pre_effect_authorization",
        "authorization_effect_observed": False,
        "inactive_tree": {
            "head": inactive_head,
            "tree": inactive_tree,
            "branch": branch,
            "active_owned_effects": 0,
            "quiescent": True,
        },
        "old_generation": old_generation,
        "target_generation": target,
        "cas_lease": {
            "lease_id": lease_id,
            "cas_token": cas_token,
            "tree_id": tree_id,
            "one_generation_winner_required": True,
        },
        "guardian": {
            "type": GUARDIAN_TYPE,
            "guardian_identity": guardian_identity,
            "host_namespace": host_namespace,
            "review_cid": review_cid,
            "review_required": True,
        },
        "bounded_flags": {
            "already_active": False,
            "codebase_refill_enabled": False,
            "prompt_program_refill_enabled_after_activation": True,
            "objective_refill_enabled_after_activation": True,
            "monitor_enabled_after_activation": True,
            "legacy_hash_sharding_for_active_slices": False,
        },
        "quiescence": {
            "zero_old_generation_descendants": True,
            "zero_owned_worker_provider_merge_effects": True,
            "refill_and_monitor_dormant": True,
        },
        "expiry_at": expiry,
        "operator_review": {
            "required": True,
            "reviewer_identity": reviewer_identity,
            "self_review_allowed": False,
            "generator": GATE_GENERATOR,
        },
        "denials": {
            "authorization_may_claim_activation_effect": False,
            "authorization_alone_proves_effect_ran": False,
            "public_facade_selectable_from_authorization": False,
            "broad_legacy_codebase_refill_allowed": False,
            "self_attested_guardian_allowed": False,
            "client_owned_monitor_allowed": False,
        },
    }
    payload["authorization_id"] = activation_authorization_id(payload)
    return payload


def build_post_activation_observation(
    *,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    lifecycle: ProcessJoinEvidence,
    monitor: ProcessJoinEvidence,
    refill: RefillActivationEvidence,
    reviewer_identity: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a post-activation observation bound to a pre-effect authorization."""

    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0)
    created = created_at or now.strftime("%Y-%m-%dT%H:%M:%SZ")
    auth_sha = "sha256:" + hashlib.sha256(authorization_raw).hexdigest()
    payload: dict[str, Any] = {
        "schema": POST_ACTIVATION_OBSERVATION_SCHEMA,
        "created_at": created,
        "board_namespace": BOARD_NAMESPACE,
        "task_id": ACTIVATION_TASK_ID,
        "authorization_binding": {
            "authorization_id": authorization.get("authorization_id"),
            "authorization_sha256": auth_sha,
            "authorization_alone_proves_effect": False,
        },
        "target_generation": authorization.get("target_generation"),
        "lifecycle": lifecycle.to_dict(),
        "monitor": monitor.to_dict(),
        "refill": refill.to_dict(),
        "joined_running": {
            "joined": True,
            "same_generation": True,
            "join_fields": [
                "lifecycle_process_birth",
                "lifecycle_lease",
                "lifecycle_fence",
                "lifecycle_heartbeat",
                "lifecycle_event_cursor",
                "monitor_process_birth",
                "monitor_lease",
                "monitor_fence",
                "monitor_heartbeat",
                "monitor_event_cursor",
            ],
        },
        "required_observations": list(REQUIRED_POST_ACTIVATION_OBSERVATIONS),
        "operator_review": {
            "required": True,
            "reviewer_identity": reviewer_identity,
            "self_review_allowed": False,
            "generator": GATE_GENERATOR,
        },
        "denials": {
            "observation_retroactively_authorizes": False,
            "authorization_alone_sufficient": False,
        },
    }
    payload["observation_id"] = post_activation_observation_id(payload)
    return payload


@dataclass
class ActivationGenerationState:
    """Durable one-generation CAS winner state under a guardian."""

    old_generation: int
    target_generation: int
    lease_id: str
    cas_token: str
    tree_id: str
    guardian_identity: str
    authorization_id: str
    activated_at_ms: int
    refill_authorized: bool = True
    monitor_authorized: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeGenerationActivationStore:
    """Exactly-one old+1 CAS/lease winner store for ASE3-026 activation."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.root / "activation.lock"
        self._state_path = self.root / "generation_state.json"

    def _read(self) -> ActivationGenerationState | None:
        if not self._state_path.is_file():
            return None
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        return ActivationGenerationState(
            old_generation=int(raw["old_generation"]),
            target_generation=int(raw["target_generation"]),
            lease_id=str(raw["lease_id"]),
            cas_token=str(raw["cas_token"]),
            tree_id=str(raw["tree_id"]),
            guardian_identity=str(raw["guardian_identity"]),
            authorization_id=str(raw["authorization_id"]),
            activated_at_ms=int(raw["activated_at_ms"]),
            refill_authorized=bool(raw.get("refill_authorized", True)),
            monitor_authorized=bool(raw.get("monitor_authorized", True)),
        )

    def _write(self, state: ActivationGenerationState) -> None:
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(state.to_dict(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self._state_path)

    def consume_authorization(
        self,
        authorization: Mapping[str, Any],
        *,
        guardian_identity: str,
        now_ms: int | None = None,
    ) -> tuple[ActivationGenerationState, bool]:
        """Consume a validated pre-effect authorization.

        Returns ``(state, adopted)`` where ``adopted`` is True when an
        identical prior winner is reused.
        """

        errors = validate_activation_authorization(
            authorization, now_ms=now_ms or int(time.time() * 1000)
        )
        if errors:
            raise ProtectedRuntimeActivationError("; ".join(errors))

        guardian = authorization.get("guardian")
        if not isinstance(guardian, Mapping):
            raise ProtectedRuntimeActivationError("guardian required")
        if guardian.get("guardian_identity") != guardian_identity:
            raise ProtectedRuntimeActivationError(
                "only the reviewed host-namespace guardian may consume authorization"
            )
        if guardian.get("type") != GUARDIAN_TYPE:
            raise ProtectedRuntimeActivationError(
                f"guardian type must be {GUARDIAN_TYPE}"
            )

        cas = authorization["cas_lease"]
        assert isinstance(cas, Mapping)
        candidate = ActivationGenerationState(
            old_generation=int(authorization["old_generation"]),
            target_generation=int(authorization["target_generation"]),
            lease_id=str(cas["lease_id"]),
            cas_token=str(cas["cas_token"]),
            tree_id=str(cas["tree_id"]),
            guardian_identity=guardian_identity,
            authorization_id=str(authorization["authorization_id"]),
            activated_at_ms=int(now_ms or int(time.time() * 1000)),
        )

        existing = self._read()
        if existing is not None:
            if (
                existing.lease_id == candidate.lease_id
                and existing.cas_token == candidate.cas_token
                and existing.target_generation == candidate.target_generation
                and existing.authorization_id == candidate.authorization_id
            ):
                return existing, True
            raise ProtectedRuntimeActivationError(
                "activation CAS lost: non-identical winner already recorded"
            )

        self._write(candidate)
        return candidate, False

    def current(self) -> ActivationGenerationState | None:
        return self._read()


def load_activation_authorization_payload(
    path: Path | str,
    *,
    maximum_bytes: int = 128 * 1024,
) -> tuple[dict[str, Any], bytes, str]:
    """Load authorization JSON without following links."""

    artifact = Path(path)
    if artifact.is_symlink():
        raise ProtectedRuntimeActivationError("authorization path must not be a symlink")
    raw = artifact.read_bytes()
    if len(raw) > maximum_bytes:
        raise ProtectedRuntimeActivationError("authorization exceeds byte bound")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ProtectedRuntimeActivationError("authorization must be a JSON object")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return payload, raw, digest


def load_post_activation_observation_payload(
    path: Path | str,
    *,
    maximum_bytes: int = 256 * 1024,
) -> tuple[dict[str, Any], bytes, str]:
    """Load observation JSON without following links."""

    artifact = Path(path)
    if artifact.is_symlink():
        raise ProtectedRuntimeActivationError("observation path must not be a symlink")
    raw = artifact.read_bytes()
    if len(raw) > maximum_bytes:
        raise ProtectedRuntimeActivationError("observation exceeds byte bound")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ProtectedRuntimeActivationError("observation must be a JSON object")
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return payload, raw, digest


__all__ = [
    "ACTIVATION_AUTHORIZATION_SCHEMA",
    "ACTIVATION_TASK_ID",
    "BOARD_NAMESPACE",
    "GATE_GENERATOR",
    "GUARDIAN_TYPE",
    "POST_ACTIVATION_OBSERVATION_SCHEMA",
    "REQUIRED_POST_ACTIVATION_OBSERVATIONS",
    "RUNTIME_GENERATION_CAS_SCHEMA",
    "ActivationGenerationState",
    "ProcessJoinEvidence",
    "ProtectedRuntimeActivationError",
    "RefillActivationEvidence",
    "RuntimeGenerationActivationCAS",
    "RuntimeGenerationActivationStore",
    "activation_authorization_id",
    "build_activation_authorization",
    "build_post_activation_observation",
    "load_activation_authorization_payload",
    "load_post_activation_observation_payload",
    "post_activation_observation_id",
    "validate_activation_authorization",
    "validate_post_activation_observation",
]
