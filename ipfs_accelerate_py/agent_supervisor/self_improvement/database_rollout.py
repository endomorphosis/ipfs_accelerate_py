"""Staged canary, default cutover, and rollback for DuckDB/Quack (DQP-038).

Interfaces: ``DatabaseRolloutPolicy@1``, ``DatabaseCutoverReceipt@1``

Advances database authority through the closed stage ladder::

    off -> observe -> shadow -> assist -> canary -> default

Each stage binds an exact authority mode, store generation, schema revision,
Quack profile, evidence roots, expiry, kill switch, and operator action.
New local programs default to Quack **only** under a valid release gate.
Rollback switches the authority/read route without deleting history and
without accepting legacy dual writes as a permanent two-authority design.

Dual write is temporary evidence collection (shadow/assist), never a lasting
architecture. Kill switch and exact-rollback override every score.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.quack_capabilities import DEFAULT_QUACK_BETA_LIMITATIONS
from ..task_sources.task_source import StateAuthorityMode


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_ROLLOUT_POLICY_INTERFACE: Final[str] = "DatabaseRolloutPolicy@1"
DATABASE_CUTOVER_RECEIPT_INTERFACE: Final[str] = "DatabaseCutoverReceipt@1"
DATABASE_ROLLOUT_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-038"
GOAL_ID: Final[str] = "DQP-G080"
EVIDENCE: Final[str] = "dqp/database-rollout@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
ROLLOUT_POLICY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-rollout-policy@1"
CUTOVER_RECEIPT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-cutover-receipt@1"
ROLLOUT_BINDING_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-rollout-binding@1"
ROLLOUT_EVIDENCE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-rollout-evidence@1"
RELEASE_GATE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-release-gate@1"
ROLLBACK_RECEIPT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-rollback-receipt@1"

DEFAULT_MAX_BACKUP_AGE_SECONDS: Final[int] = 30 * 24 * 3600
DEFAULT_EVIDENCE_MAX_AGE_SECONDS: Final[int] = 7 * 24 * 3600
DEFAULT_SCHEMA_REVISION: Final[str] = "control-plane-schema@1"
DEFAULT_QUACK_PROFILE: Final[str] = "duckdb-1.5.x-quack-pinned"
DEFAULT_STORE_ID: Final[str] = "control.duckdb"
MAX_TEXT_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 256
MAX_HISTORY: Final[int] = 10_000

# Evidence kinds required for promotion past shadow (closed set).
REQUIRED_EVIDENCE_KINDS: Final[tuple[str, ...]] = (
    "chaos_security",
    "canary_e2e",
    "churn_quality",
    "shadow_parity",
    "backup_restore",
)

# Closed reason codes covering the DQP-038 evidence subset.
REASON_PROMOTION_DENIED: Final[str] = "promotion_denied"
REASON_STALE_EVIDENCE: Final[str] = "stale_evidence"
REASON_PARTIAL_ROLLOUT: Final[str] = "partial_rollout"
REASON_SERVER_UNAVAILABLE: Final[str] = "server_unavailable"
REASON_BACKUP_AGE: Final[str] = "backup_age"
REASON_ROLLBACK: Final[str] = "rollback"
REASON_LEGACY_EXPORT: Final[str] = "legacy_export"
REASON_BETA_WAIVER: Final[str] = "beta_waiver"
REASON_REMOTE_PROHIBITION: Final[str] = "remote_prohibition"
REASON_KILL_SWITCH: Final[str] = "kill_switch"
REASON_MISSING_EVIDENCE: Final[str] = "missing_evidence"
REASON_INVALID_STAGE: Final[str] = "invalid_stage"
REASON_HISTORY_PRESERVED: Final[str] = "history_preserved"
REASON_NO_LEGACY_DUAL_WRITE: Final[str] = "no_legacy_dual_write"
REASON_DEFAULT_REQUIRES_CANARY: Final[str] = "default_requires_canary"
REASON_RELEASE_GATE_INVALID: Final[str] = "release_gate_invalid"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class DatabaseRolloutStage(str, Enum):
    """Closed, ordered authority stage ladder for DuckDB/Quack cutover."""

    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    CANARY = "canary"
    DEFAULT = "default"

    @property
    def rank(self) -> int:
        return _STAGE_ORDER.index(self)


_STAGE_ORDER: Final[tuple[DatabaseRolloutStage, ...]] = tuple(DatabaseRolloutStage)


class RolloutDisposition(str, Enum):
    """Closed promotion / hold / rollback disposition."""

    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"
    DENY = "deny"


class EvidenceStatus(str, Enum):
    """Admission status for one required evidence cell."""

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"
    SYNTHETIC = "synthetic"
    SKIPPED = "skipped"
    STALE = "stale"


class OperatorAction(str, Enum):
    """Closed operator actions that may mutate rollout state."""

    PROMOTE = "promote"
    ROLLBACK = "rollback"
    KILL_SWITCH = "kill_switch"
    KILL_SWITCH_CLEAR = "kill_switch_clear"
    STATUS = "status"
    DEFAULT_PROGRAM = "default_program"


class DatabaseRolloutError(ValueError):
    """Fail-closed rejection for unsafe rollout inputs or transitions."""


# Stage → authority mode binding (exact; not a free-form feature flag).
_STAGE_AUTHORITY: Final[Mapping[DatabaseRolloutStage, StateAuthorityMode]] = (
    MappingProxyType(
        {
            DatabaseRolloutStage.OFF: StateAuthorityMode.EMBEDDED_MAINTENANCE,
            DatabaseRolloutStage.OBSERVE: StateAuthorityMode.EMBEDDED_MAINTENANCE,
            DatabaseRolloutStage.SHADOW: StateAuthorityMode.QUACK_SHADOW,
            DatabaseRolloutStage.ASSIST: StateAuthorityMode.QUACK_SHADOW,
            DatabaseRolloutStage.CANARY: StateAuthorityMode.QUACK_AUTHORITATIVE,
            DatabaseRolloutStage.DEFAULT: StateAuthorityMode.QUACK_AUTHORITATIVE,
        }
    )
)

# Stages that may dual-observe legacy projections (temporary evidence only).
_DUAL_OBSERVATION_STAGES: Final[frozenset[DatabaseRolloutStage]] = frozenset(
    {
        DatabaseRolloutStage.SHADOW,
        DatabaseRolloutStage.ASSIST,
    }
)

# Stages that grant Quack as scheduling authority for bound programs.
_QUACK_AUTHORITY_STAGES: Final[frozenset[DatabaseRolloutStage]] = frozenset(
    {
        DatabaseRolloutStage.SHADOW,
        DatabaseRolloutStage.ASSIST,
        DatabaseRolloutStage.CANARY,
        DatabaseRolloutStage.DEFAULT,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DatabaseRolloutError("rollout data must be canonical JSON") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def content_identity(payload: Any) -> str:
    raw = _canonical_json(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise DatabaseRolloutError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise DatabaseRolloutError(f"{name} must not be empty")
    if "\x00" in result:
        raise DatabaseRolloutError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise DatabaseRolloutError(f"{name} exceeds its {maximum}-byte bound")
    return result


def _optional_text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if value is None or value == "":
        return ""
    return _text(value, name, maximum=maximum)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DatabaseRolloutError(f"{name} must be a boolean")
    return value


def _nonnegative_int(value: Any, name: str, *, maximum: int = 10**18) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatabaseRolloutError(f"{name} must be a non-negative integer")
    if value < 0 or value > maximum:
        raise DatabaseRolloutError(f"{name} out of bounds")
    return value


def _stage(value: Any, name: str = "stage") -> DatabaseRolloutStage:
    if isinstance(value, DatabaseRolloutStage):
        return value
    try:
        return DatabaseRolloutStage(str(getattr(value, "value", value)).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in DatabaseRolloutStage)
        raise DatabaseRolloutError(f"{name} must be one of: {allowed}") from exc


def _evidence_status(value: Any, name: str) -> EvidenceStatus:
    if isinstance(value, EvidenceStatus):
        return value
    try:
        return EvidenceStatus(str(getattr(value, "value", value)).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in EvidenceStatus)
        raise DatabaseRolloutError(f"{name} must be one of: {allowed}") from exc


def authority_mode_for_stage(stage: DatabaseRolloutStage | str) -> StateAuthorityMode:
    """Return the exact StateAuthorityMode bound to ``stage``."""

    selected = _stage(stage, "stage")
    return _STAGE_AUTHORITY[selected]


def dual_observation_allowed(stage: DatabaseRolloutStage | str) -> bool:
    """Return whether temporary dual observation is allowed at ``stage``."""

    return _stage(stage, "stage") in _DUAL_OBSERVATION_STAGES


def closed_rollout_stages() -> tuple[str, ...]:
    """Return the closed stage vocabulary in stable order."""

    return tuple(item.value for item in DatabaseRolloutStage)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatabaseRolloutBinding:
    """Exact coordinates bound into every promotion and rollback receipt."""

    SCHEMA: ClassVar[str] = ROLLOUT_BINDING_SCHEMA

    repository_id: str
    tree_id: str
    store_id: str = DEFAULT_STORE_ID
    store_generation: int = 1
    schema_revision: str = DEFAULT_SCHEMA_REVISION
    quack_profile: str = DEFAULT_QUACK_PROFILE
    program_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", _text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(self, "store_id", _text(self.store_id, "store_id"))
        object.__setattr__(
            self,
            "store_generation",
            _nonnegative_int(self.store_generation, "store_generation"),
        )
        object.__setattr__(
            self, "schema_revision", _text(self.schema_revision, "schema_revision")
        )
        object.__setattr__(
            self, "quack_profile", _text(self.quack_profile, "quack_profile")
        )
        if self.program_id:
            object.__setattr__(
                self, "program_id", _text(self.program_id, "program_id")
            )

    @property
    def binding_id(self) -> str:
        return content_identity(self.to_dict(include_binding_id=False))

    def to_dict(self, *, include_binding_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "store_id": self.store_id,
            "store_generation": self.store_generation,
            "schema_revision": self.schema_revision,
            "quack_profile": self.quack_profile,
            "program_id": self.program_id,
        }
        if include_binding_id:
            payload["binding_id"] = self.binding_id
        return payload


@dataclass(frozen=True)
class RolloutEvidenceCell:
    """One required evidence cell admitted into a release gate."""

    SCHEMA: ClassVar[str] = ROLLOUT_EVIDENCE_SCHEMA

    kind: str
    status: EvidenceStatus
    evidence_id: str = ""
    tree_id: str = ""
    age_seconds: int = 0
    passed: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _text(self.kind, "kind", maximum=64))
        object.__setattr__(
            self, "status", _evidence_status(self.status, "status")
        )
        if self.evidence_id:
            object.__setattr__(
                self, "evidence_id", _text(self.evidence_id, "evidence_id")
            )
        if self.tree_id:
            object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "age_seconds", _nonnegative_int(self.age_seconds, "age_seconds")
        )
        object.__setattr__(self, "passed", _boolean(self.passed, "passed"))
        if self.detail:
            object.__setattr__(
                self, "detail", _text(self.detail, "detail", maximum=256)
            )

    @property
    def admitted(self) -> bool:
        return (
            self.status is EvidenceStatus.MEASURED
            and self.passed
            and bool(self.evidence_id)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind,
            "status": self.status.value,
            "evidence_id": self.evidence_id,
            "tree_id": self.tree_id,
            "age_seconds": self.age_seconds,
            "passed": self.passed,
            "admitted": self.admitted,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReleaseGateEvaluation:
    """Joined release-gate decision for promotion / default cutover."""

    SCHEMA: ClassVar[str] = RELEASE_GATE_SCHEMA

    valid: bool
    reason_codes: tuple[str, ...]
    evidence: tuple[RolloutEvidenceCell, ...]
    server_available: bool
    backup_age_seconds: int
    backup_age_ok: bool
    remote_bind_prohibited: bool
    beta_limitations: tuple[str, ...]
    beta_waiver_recorded: bool
    dual_write_accepted: bool
    history_deletion_requested: bool
    partial_rollout: bool
    kill_switch_engaged: bool
    binding: DatabaseRolloutBinding
    evaluated_at: str = field(default_factory=_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "valid": self.valid,
            "reason_codes": list(self.reason_codes),
            "evidence": [item.to_dict() for item in self.evidence],
            "server_available": self.server_available,
            "backup_age_seconds": self.backup_age_seconds,
            "backup_age_ok": self.backup_age_ok,
            "remote_bind_prohibited": self.remote_bind_prohibited,
            "beta_limitations": list(self.beta_limitations),
            "beta_waiver_recorded": self.beta_waiver_recorded,
            "dual_write_accepted": self.dual_write_accepted,
            "history_deletion_requested": self.history_deletion_requested,
            "partial_rollout": self.partial_rollout,
            "kill_switch_engaged": self.kill_switch_engaged,
            "binding": self.binding.to_dict(),
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True)
class DatabaseRolloutPolicy:
    """``DatabaseRolloutPolicy@1`` — closed stage ladder and hard overrides.

    Defaults stay off. Kill switch and exact-rollback requirements override
    every score. ``default`` is absent from the default allowed set until an
    operator policy explicitly includes it **and** canary evidence is current.
    """

    SCHEMA: ClassVar[str] = ROLLOUT_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = DATABASE_ROLLOUT_POLICY_INTERFACE

    policy_id: str = "policy:database-rollout@1"
    policy_revision: str = "1"
    allowed_stages: tuple[DatabaseRolloutStage, ...] = (
        DatabaseRolloutStage.OFF,
        DatabaseRolloutStage.OBSERVE,
        DatabaseRolloutStage.SHADOW,
        DatabaseRolloutStage.ASSIST,
        DatabaseRolloutStage.CANARY,
    )
    kill_switch_engaged: bool = False
    require_exact_rollback: bool = True
    require_canary_before_default: bool = True
    max_backup_age_seconds: int = DEFAULT_MAX_BACKUP_AGE_SECONDS
    max_evidence_age_seconds: int = DEFAULT_EVIDENCE_MAX_AGE_SECONDS
    allow_remote_bind: bool = False
    accept_legacy_dual_writes: bool = False
    allow_history_deletion: bool = False
    beta_waiver_required: bool = True
    new_program_default_stage: DatabaseRolloutStage = DatabaseRolloutStage.OFF

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )
        raw = self.allowed_stages
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise DatabaseRolloutError("allowed_stages must be a sequence")
        normalized = tuple(
            item
            for item in DatabaseRolloutStage
            if item in {_stage(entry, "allowed_stages") for entry in raw}
        )
        if DatabaseRolloutStage.OFF not in normalized:
            raise DatabaseRolloutError("allowed_stages must include off")
        object.__setattr__(self, "allowed_stages", normalized)
        object.__setattr__(
            self, "kill_switch_engaged", _boolean(self.kill_switch_engaged, "kill_switch_engaged")
        )
        object.__setattr__(
            self,
            "require_exact_rollback",
            _boolean(self.require_exact_rollback, "require_exact_rollback"),
        )
        object.__setattr__(
            self,
            "require_canary_before_default",
            _boolean(
                self.require_canary_before_default, "require_canary_before_default"
            ),
        )
        object.__setattr__(
            self,
            "max_backup_age_seconds",
            _nonnegative_int(self.max_backup_age_seconds, "max_backup_age_seconds"),
        )
        object.__setattr__(
            self,
            "max_evidence_age_seconds",
            _nonnegative_int(
                self.max_evidence_age_seconds, "max_evidence_age_seconds"
            ),
        )
        object.__setattr__(
            self, "allow_remote_bind", _boolean(self.allow_remote_bind, "allow_remote_bind")
        )
        # Dual-write permanence and history deletion are always refused.
        object.__setattr__(self, "accept_legacy_dual_writes", False)
        object.__setattr__(self, "allow_history_deletion", False)
        object.__setattr__(
            self,
            "beta_waiver_required",
            _boolean(self.beta_waiver_required, "beta_waiver_required"),
        )
        object.__setattr__(
            self,
            "new_program_default_stage",
            _stage(self.new_program_default_stage, "new_program_default_stage"),
        )

    def allows(self, stage: DatabaseRolloutStage | str) -> bool:
        return _stage(stage, "stage") in self.allowed_stages

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": DATABASE_ROLLOUT_CONTRACT_VERSION,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "allowed_stages": [item.value for item in self.allowed_stages],
            "kill_switch_engaged": self.kill_switch_engaged,
            "require_exact_rollback": self.require_exact_rollback,
            "require_canary_before_default": self.require_canary_before_default,
            "max_backup_age_seconds": self.max_backup_age_seconds,
            "max_evidence_age_seconds": self.max_evidence_age_seconds,
            "allow_remote_bind": self.allow_remote_bind,
            "accept_legacy_dual_writes": False,
            "allow_history_deletion": False,
            "beta_waiver_required": self.beta_waiver_required,
            "new_program_default_stage": self.new_program_default_stage.value,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
        }

    @classmethod
    def default(cls) -> "DatabaseRolloutPolicy":
        return cls()

    @classmethod
    def with_default_cutover(cls) -> "DatabaseRolloutPolicy":
        """Operator policy that admits the full ladder including default."""

        return cls(
            allowed_stages=tuple(DatabaseRolloutStage),
            new_program_default_stage=DatabaseRolloutStage.DEFAULT,
        )


@dataclass(frozen=True)
class DatabaseCutoverReceipt:
    """``DatabaseCutoverReceipt@1`` — durable promotion / rollback receipt.

    A serialized receipt is never authority by itself: verification replays
    source evidence and recomputes the release gate.
    """

    SCHEMA: ClassVar[str] = CUTOVER_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = DATABASE_CUTOVER_RECEIPT_INTERFACE

    disposition: RolloutDisposition
    from_stage: DatabaseRolloutStage
    to_stage: DatabaseRolloutStage
    authority_mode: str
    binding: DatabaseRolloutBinding
    release_gate_valid: bool
    operator_action: OperatorAction
    reason_codes: tuple[str, ...] = ()
    history_preserved: bool = True
    legacy_dual_write_accepted: bool = False
    kill_switch_engaged: bool = False
    new_program_defaults_to_quack: bool = False
    dual_observation: bool = False
    detail: str = ""
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    created_at: str = field(default_factory=_utc_iso)
    history_length: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disposition",
            self.disposition
            if isinstance(self.disposition, RolloutDisposition)
            else RolloutDisposition(str(self.disposition)),
        )
        object.__setattr__(self, "from_stage", _stage(self.from_stage, "from_stage"))
        object.__setattr__(self, "to_stage", _stage(self.to_stage, "to_stage"))
        object.__setattr__(
            self, "authority_mode", _text(self.authority_mode, "authority_mode", maximum=64)
        )
        if not isinstance(self.operator_action, OperatorAction):
            object.__setattr__(
                self, "operator_action", OperatorAction(str(self.operator_action))
            )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _text(item, "reason_codes.item", maximum=128)
                for item in self.reason_codes[:MAX_REASON_CODES]
            ),
        )
        # Safety floors: never claim dual-write permanence or history loss.
        object.__setattr__(self, "legacy_dual_write_accepted", False)
        object.__setattr__(self, "history_preserved", True)

    @property
    def accepted(self) -> bool:
        return self.disposition is RolloutDisposition.PROMOTE

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": DATABASE_ROLLOUT_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "disposition": self.disposition.value,
            "accepted": self.accepted,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "authority_mode": self.authority_mode,
            "binding": self.binding.to_dict(),
            "release_gate_valid": self.release_gate_valid,
            "operator_action": self.operator_action.value,
            "reason_codes": list(self.reason_codes),
            "history_preserved": True,
            "legacy_dual_write_accepted": False,
            "kill_switch_engaged": self.kill_switch_engaged,
            "new_program_defaults_to_quack": self.new_program_defaults_to_quack,
            "dual_observation": self.dual_observation,
            "detail": self.detail,
            "created_at": self.created_at,
            "history_length": self.history_length,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


@dataclass(frozen=True)
class RollbackReceipt:
    """Route-only rollback receipt; history is never deleted."""

    SCHEMA: ClassVar[str] = ROLLBACK_RECEIPT_SCHEMA

    from_stage: DatabaseRolloutStage
    to_stage: DatabaseRolloutStage
    authority_mode: str
    history_preserved: bool
    legacy_dual_write_accepted: bool
    binding: DatabaseRolloutBinding
    reason_codes: tuple[str, ...]
    created_at: str = field(default_factory=_utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "authority_mode": self.authority_mode,
            "history_preserved": self.history_preserved,
            "legacy_dual_write_accepted": self.legacy_dual_write_accepted,
            "binding": self.binding.to_dict(),
            "reason_codes": list(self.reason_codes),
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Rollout engine
# ---------------------------------------------------------------------------


class DatabaseRollout:
    """Hermetic staged canary / default cutover / rollback controller.

    Interface projection for ``DatabaseRolloutPolicy@1``.
    """

    INTERFACE: ClassVar[str] = DATABASE_ROLLOUT_POLICY_INTERFACE

    def __init__(
        self,
        policy: DatabaseRolloutPolicy | None = None,
        *,
        binding: DatabaseRolloutBinding | None = None,
    ) -> None:
        self.policy = policy or DatabaseRolloutPolicy.default()
        self.binding = binding or DatabaseRolloutBinding(
            repository_id="repository:hermetic",
            tree_id="tree:hermetic",
        )
        self.stage = DatabaseRolloutStage.OFF
        self.authority_mode = authority_mode_for_stage(self.stage)
        self.kill_switch_engaged = bool(self.policy.kill_switch_engaged)
        self._history: list[dict[str, Any]] = []
        self._canary_completed = False
        self._last_receipt: DatabaseCutoverReceipt | None = None
        self._last_gate: ReleaseGateEvaluation | None = None
        self._append(
            {
                "event": "init",
                "stage": self.stage.value,
                "authority_mode": self.authority_mode.value,
            }
        )

    # -- history ------------------------------------------------------------

    def _append(self, event: Mapping[str, Any]) -> None:
        if len(self._history) >= MAX_HISTORY:
            raise DatabaseRolloutError("rollout history bound exceeded")
        payload = dict(event)
        payload.setdefault("at", _utc_iso())
        self._history.append(payload)

    @property
    def history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(dict(item)) for item in self._history)

    @property
    def last_receipt(self) -> DatabaseCutoverReceipt | None:
        return self._last_receipt

    @property
    def last_gate(self) -> ReleaseGateEvaluation | None:
        return self._last_gate

    # -- release gate -------------------------------------------------------

    def evaluate_release_gate(
        self,
        evidence: Sequence[RolloutEvidenceCell] | Mapping[str, Any] | None = None,
        *,
        server_available: bool = True,
        backup_age_seconds: int = 0,
        remote_bind_requested: bool = False,
        beta_waiver_recorded: bool = True,
        dual_write_accepted: bool = False,
        history_deletion_requested: bool = False,
        partial_rollout: bool = False,
        target_stage: DatabaseRolloutStage | str | None = None,
    ) -> ReleaseGateEvaluation:
        """Join evidence and hard floors into a release-gate decision."""

        cells = self._normalize_evidence(evidence)
        reasons: list[str] = []
        target = (
            _stage(target_stage, "target_stage")
            if target_stage is not None
            else self.stage
        )

        if self.kill_switch_engaged or self.policy.kill_switch_engaged:
            reasons.append(REASON_KILL_SWITCH)

        # Permanent dual-write acceptance and history deletion are always refused.
        if dual_write_accepted or self.policy.accept_legacy_dual_writes:
            reasons.append(REASON_LEGACY_EXPORT)
            dual_write_accepted = False

        if history_deletion_requested or self.policy.allow_history_deletion:
            reasons.append(REASON_HISTORY_PRESERVED)
            history_deletion_requested = False

        remote_prohibited = not self.policy.allow_remote_bind
        if remote_bind_requested and remote_prohibited:
            reasons.append(REASON_REMOTE_PROHIBITION)

        backup_age_seconds = _nonnegative_int(backup_age_seconds, "backup_age_seconds")
        backup_age_ok = backup_age_seconds <= self.policy.max_backup_age_seconds

        # Shadow+ requires a live state-owner; canary+ joins full release evidence.
        if target.rank >= DatabaseRolloutStage.SHADOW.rank:
            if not server_available:
                reasons.append(REASON_SERVER_UNAVAILABLE)

        if target.rank >= DatabaseRolloutStage.CANARY.rank:
            if not backup_age_ok:
                reasons.append(REASON_BACKUP_AGE)
            if partial_rollout:
                reasons.append(REASON_PARTIAL_ROLLOUT)
            if self.policy.beta_waiver_required and not beta_waiver_recorded:
                reasons.append(REASON_BETA_WAIVER)

        # Evidence admission for stages that grant production-affecting authority.
        if target.rank >= DatabaseRolloutStage.CANARY.rank:
            by_kind = {cell.kind: cell for cell in cells}
            for kind in REQUIRED_EVIDENCE_KINDS:
                cell = by_kind.get(kind)
                if cell is None:
                    reasons.append(REASON_MISSING_EVIDENCE)
                    continue
                if cell.status is EvidenceStatus.STALE or (
                    cell.age_seconds > self.policy.max_evidence_age_seconds
                ):
                    reasons.append(REASON_STALE_EVIDENCE)
                elif cell.status in {
                    EvidenceStatus.SYNTHETIC,
                    EvidenceStatus.SKIPPED,
                    EvidenceStatus.UNAVAILABLE,
                }:
                    reasons.append(REASON_MISSING_EVIDENCE)
                elif not cell.admitted:
                    reasons.append(REASON_PROMOTION_DENIED)
                elif cell.tree_id and cell.tree_id != self.binding.tree_id:
                    reasons.append(REASON_STALE_EVIDENCE)

            if (
                self.policy.require_canary_before_default
                and target is DatabaseRolloutStage.DEFAULT
                and not self._canary_completed
            ):
                reasons.append(REASON_DEFAULT_REQUIRES_CANARY)

        # Deduplicate while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for code in reasons:
            if code not in seen:
                seen.add(code)
                ordered.append(code)

        valid = not ordered
        gate = ReleaseGateEvaluation(
            valid=valid,
            reason_codes=tuple(ordered),
            evidence=tuple(cells),
            server_available=bool(server_available),
            backup_age_seconds=backup_age_seconds,
            backup_age_ok=backup_age_ok,
            remote_bind_prohibited=remote_prohibited,
            beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
            beta_waiver_recorded=bool(beta_waiver_recorded),
            dual_write_accepted=False,
            history_deletion_requested=False,
            partial_rollout=bool(partial_rollout),
            kill_switch_engaged=bool(
                self.kill_switch_engaged or self.policy.kill_switch_engaged
            ),
            binding=self.binding,
        )
        self._last_gate = gate
        self._append(
            {
                "event": "release_gate",
                "valid": gate.valid,
                "target_stage": target.value,
                "reason_codes": list(gate.reason_codes),
            }
        )
        return gate

    def _normalize_evidence(
        self,
        evidence: Sequence[RolloutEvidenceCell] | Mapping[str, Any] | None,
    ) -> list[RolloutEvidenceCell]:
        if evidence is None:
            return []
        if isinstance(evidence, Mapping):
            cells: list[RolloutEvidenceCell] = []
            for kind, payload in evidence.items():
                if isinstance(payload, RolloutEvidenceCell):
                    cells.append(payload)
                    continue
                if not isinstance(payload, Mapping):
                    raise DatabaseRolloutError(
                        f"evidence cell for {kind!r} must be a mapping"
                    )
                cells.append(
                    RolloutEvidenceCell(
                        kind=str(payload.get("kind", kind)),
                        status=payload.get("status", EvidenceStatus.UNAVAILABLE),
                        evidence_id=str(payload.get("evidence_id", "") or ""),
                        tree_id=str(payload.get("tree_id", "") or ""),
                        age_seconds=int(payload.get("age_seconds", 0) or 0),
                        passed=bool(payload.get("passed", False)),
                        detail=str(payload.get("detail", "") or ""),
                    )
                )
            return cells
        return [item if isinstance(item, RolloutEvidenceCell) else RolloutEvidenceCell(
            kind=str(getattr(item, "kind", "unknown")),
            status=getattr(item, "status", EvidenceStatus.UNAVAILABLE),
            evidence_id=str(getattr(item, "evidence_id", "") or ""),
            tree_id=str(getattr(item, "tree_id", "") or ""),
            age_seconds=int(getattr(item, "age_seconds", 0) or 0),
            passed=bool(getattr(item, "passed", False)),
            detail=str(getattr(item, "detail", "") or ""),
        ) for item in evidence]

    # -- promote / rollback -------------------------------------------------

    def promote(
        self,
        target: DatabaseRolloutStage | str,
        *,
        evidence: Sequence[RolloutEvidenceCell] | Mapping[str, Any] | None = None,
        server_available: bool = True,
        backup_age_seconds: int = 0,
        remote_bind_requested: bool = False,
        beta_waiver_recorded: bool = True,
        dual_write_accepted: bool = False,
        history_deletion_requested: bool = False,
        partial_rollout: bool = False,
        operator_action: OperatorAction = OperatorAction.PROMOTE,
    ) -> DatabaseCutoverReceipt:
        """Advance one stage at a time under a valid release gate."""

        target_stage = _stage(target, "target")
        from_stage = self.stage

        if self.kill_switch_engaged or self.policy.kill_switch_engaged:
            return self._deny(
                from_stage=from_stage,
                to_stage=from_stage,
                reasons=(REASON_KILL_SWITCH, REASON_PROMOTION_DENIED),
                detail="kill switch engaged; promotion refused",
                operator_action=operator_action,
                release_gate_valid=False,
            )

        if not self.policy.allows(target_stage):
            return self._deny(
                from_stage=from_stage,
                to_stage=from_stage,
                reasons=(REASON_INVALID_STAGE, REASON_PROMOTION_DENIED),
                detail=f"stage {target_stage.value} is not in allowed_stages",
                operator_action=operator_action,
                release_gate_valid=False,
            )

        # One stage at a time; equal stage is a no-op hold.
        if target_stage.rank < from_stage.rank:
            return self._deny(
                from_stage=from_stage,
                to_stage=from_stage,
                reasons=(REASON_INVALID_STAGE, REASON_PROMOTION_DENIED),
                detail="promotion cannot move backward; use rollback",
                operator_action=operator_action,
                release_gate_valid=False,
            )
        if target_stage.rank > from_stage.rank + 1:
            return self._deny(
                from_stage=from_stage,
                to_stage=from_stage,
                reasons=(REASON_PARTIAL_ROLLOUT, REASON_PROMOTION_DENIED),
                detail="promotion advances one stage at a time",
                operator_action=operator_action,
                release_gate_valid=False,
                partial_rollout=True,
            )
        if target_stage is from_stage:
            receipt = DatabaseCutoverReceipt(
                disposition=RolloutDisposition.HOLD,
                from_stage=from_stage,
                to_stage=from_stage,
                authority_mode=self.authority_mode.value,
                binding=self.binding,
                release_gate_valid=True,
                operator_action=operator_action,
                reason_codes=("already_at_stage",),
                history_preserved=True,
                kill_switch_engaged=self.kill_switch_engaged,
                new_program_defaults_to_quack=self.new_programs_default_to_quack(),
                dual_observation=dual_observation_allowed(from_stage),
                detail="already at requested stage",
                history_length=len(self._history),
            )
            self._last_receipt = receipt
            self._append({"event": "hold", "stage": from_stage.value})
            return receipt

        gate = self.evaluate_release_gate(
            evidence,
            server_available=server_available,
            backup_age_seconds=backup_age_seconds,
            remote_bind_requested=remote_bind_requested,
            beta_waiver_recorded=beta_waiver_recorded,
            dual_write_accepted=dual_write_accepted,
            history_deletion_requested=history_deletion_requested,
            partial_rollout=partial_rollout,
            target_stage=target_stage,
        )
        if not gate.valid:
            return self._deny(
                from_stage=from_stage,
                to_stage=from_stage,
                reasons=gate.reason_codes + (REASON_PROMOTION_DENIED,),
                detail="release gate refused promotion",
                operator_action=operator_action,
                release_gate_valid=False,
            )

        # Apply promotion.
        self.stage = target_stage
        self.authority_mode = authority_mode_for_stage(target_stage)
        if target_stage is DatabaseRolloutStage.CANARY:
            self._canary_completed = True
        defaults_to_quack = self.new_programs_default_to_quack()
        receipt = DatabaseCutoverReceipt(
            disposition=RolloutDisposition.PROMOTE,
            from_stage=from_stage,
            to_stage=target_stage,
            authority_mode=self.authority_mode.value,
            binding=self.binding,
            release_gate_valid=True,
            operator_action=operator_action,
            reason_codes=(),
            history_preserved=True,
            kill_switch_engaged=self.kill_switch_engaged,
            new_program_defaults_to_quack=defaults_to_quack,
            dual_observation=dual_observation_allowed(target_stage),
            detail=f"promoted {from_stage.value} -> {target_stage.value}",
            history_length=len(self._history) + 1,
        )
        self._last_receipt = receipt
        self._append(
            {
                "event": "promote",
                "from_stage": from_stage.value,
                "to_stage": target_stage.value,
                "authority_mode": self.authority_mode.value,
                "receipt_id": receipt.identity_id,
                "history_preserved": True,
                "legacy_dual_write_accepted": False,
            }
        )
        return receipt

    def rollback(
        self,
        target: DatabaseRolloutStage | str | None = None,
        *,
        delete_history: bool = False,
        accept_legacy_dual_writes: bool = False,
    ) -> DatabaseCutoverReceipt:
        """Switch authority route only; preserve history; refuse dual writes.

        When ``target`` is omitted, roll back one stage (or to ``off`` under
        kill switch / exact-rollback policy).
        """

        from_stage = self.stage
        if target is None:
            if from_stage is DatabaseRolloutStage.OFF:
                target_stage = DatabaseRolloutStage.OFF
            else:
                target_stage = _STAGE_ORDER[from_stage.rank - 1]
        else:
            target_stage = _stage(target, "target")

        reasons: list[str] = [REASON_ROLLBACK]

        if delete_history:
            # Refuse; record the attempt without mutating history.
            reasons.append(REASON_HISTORY_PRESERVED)
            delete_history = False

        if accept_legacy_dual_writes:
            reasons.append(REASON_NO_LEGACY_DUAL_WRITE)
            accept_legacy_dual_writes = False

        if target_stage.rank > from_stage.rank:
            return self._deny(
                from_stage=from_stage,
                to_stage=from_stage,
                reasons=(REASON_INVALID_STAGE, REASON_ROLLBACK),
                detail="rollback cannot advance stage",
                operator_action=OperatorAction.ROLLBACK,
                release_gate_valid=False,
            )

        # Exact rollback: route change only.
        self.stage = target_stage
        self.authority_mode = authority_mode_for_stage(target_stage)
        # Leaving default/canary clears the new-program default; canary flag
        # remains as historical evidence of a completed canary run.
        dual = dual_observation_allowed(target_stage)
        # Never re-enable dual write as permanent architecture on rollback.
        if accept_legacy_dual_writes:
            dual = False

        receipt = DatabaseCutoverReceipt(
            disposition=RolloutDisposition.ROLLBACK,
            from_stage=from_stage,
            to_stage=target_stage,
            authority_mode=self.authority_mode.value,
            binding=self.binding,
            release_gate_valid=True,
            operator_action=OperatorAction.ROLLBACK,
            reason_codes=tuple(reasons),
            history_preserved=True,
            kill_switch_engaged=self.kill_switch_engaged,
            new_program_defaults_to_quack=self.new_programs_default_to_quack(),
            dual_observation=dual,
            detail=f"rollback {from_stage.value} -> {target_stage.value}",
            history_length=len(self._history) + 1,
        )
        self._last_receipt = receipt
        self._append(
            {
                "event": "rollback",
                "from_stage": from_stage.value,
                "to_stage": target_stage.value,
                "authority_mode": self.authority_mode.value,
                "history_preserved": True,
                "history_length": len(self._history),
                "legacy_dual_write_accepted": False,
                "receipt_id": receipt.identity_id,
            }
        )
        return receipt

    def engage_kill_switch(self) -> DatabaseCutoverReceipt:
        """Force report-only / off route; cancel future promotion."""

        self.kill_switch_engaged = True
        from_stage = self.stage
        # Kill switch forces off without deleting history.
        self.rollback(
            DatabaseRolloutStage.OFF,
            delete_history=False,
            accept_legacy_dual_writes=False,
        )
        # Re-tag as kill-switch action.
        tagged = DatabaseCutoverReceipt(
            disposition=RolloutDisposition.ROLLBACK,
            from_stage=from_stage,
            to_stage=DatabaseRolloutStage.OFF,
            authority_mode=self.authority_mode.value,
            binding=self.binding,
            release_gate_valid=True,
            operator_action=OperatorAction.KILL_SWITCH,
            reason_codes=(REASON_KILL_SWITCH, REASON_ROLLBACK, REASON_HISTORY_PRESERVED),
            history_preserved=True,
            kill_switch_engaged=True,
            new_program_defaults_to_quack=False,
            dual_observation=False,
            detail="kill switch engaged; route forced to off",
            history_length=len(self._history),
        )
        self._last_receipt = tagged
        self._append(
            {
                "event": "kill_switch",
                "from_stage": from_stage.value,
                "to_stage": DatabaseRolloutStage.OFF.value,
            }
        )
        return tagged

    def clear_kill_switch(self) -> None:
        """Clear the kill switch; does not auto-promote."""

        self.kill_switch_engaged = False
        self._append({"event": "kill_switch_clear", "stage": self.stage.value})

    # -- defaults -----------------------------------------------------------

    def new_programs_default_to_quack(self) -> bool:
        """True only when stage is default, gate-valid path, and policy admits it."""

        if self.kill_switch_engaged or self.policy.kill_switch_engaged:
            return False
        if self.stage is not DatabaseRolloutStage.DEFAULT:
            return False
        if not self.policy.allows(DatabaseRolloutStage.DEFAULT):
            return False
        if (
            self.policy.require_canary_before_default
            and not self._canary_completed
        ):
            return False
        if self.policy.new_program_default_stage is not DatabaseRolloutStage.DEFAULT:
            return False
        return True

    def default_stage_for_new_program(self) -> DatabaseRolloutStage:
        """Stage applied to newly registered local programs."""

        if self.new_programs_default_to_quack():
            return DatabaseRolloutStage.DEFAULT
        # Fail closed: new programs stay off until release gate + default cutover.
        return DatabaseRolloutStage.OFF

    def register_new_program(self, program_id: str) -> dict[str, Any]:
        """Bind a new local program to the current default stage policy."""

        program_id = _text(program_id, "program_id")
        stage = self.default_stage_for_new_program()
        mode = authority_mode_for_stage(stage)
        record = {
            "event": "default_program",
            "program_id": program_id,
            "stage": stage.value,
            "authority_mode": mode.value,
            "defaults_to_quack": stage is DatabaseRolloutStage.DEFAULT,
            "release_gate_required": stage is DatabaseRolloutStage.DEFAULT,
        }
        self._append(record)
        return dict(record)

    # -- status -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Machine-readable rollout status snapshot."""

        return {
            "interface": self.INTERFACE,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "stage": self.stage.value,
            "authority_mode": self.authority_mode.value,
            "kill_switch_engaged": self.kill_switch_engaged,
            "canary_completed": self._canary_completed,
            "new_program_defaults_to_quack": self.new_programs_default_to_quack(),
            "default_stage_for_new_program": self.default_stage_for_new_program().value,
            "dual_observation": dual_observation_allowed(self.stage),
            "history_length": len(self._history),
            "history_preserved": True,
            "accept_legacy_dual_writes": False,
            "binding": self.binding.to_dict(),
            "policy": self.policy.to_dict(),
            "beta_limitations": list(DEFAULT_QUACK_BETA_LIMITATIONS),
            "last_receipt": (
                self._last_receipt.to_dict() if self._last_receipt is not None else None
            ),
            "last_gate": (
                self._last_gate.to_dict() if self._last_gate is not None else None
            ),
        }

    def _deny(
        self,
        *,
        from_stage: DatabaseRolloutStage,
        to_stage: DatabaseRolloutStage,
        reasons: Sequence[str],
        detail: str,
        operator_action: OperatorAction,
        release_gate_valid: bool,
        partial_rollout: bool = False,
    ) -> DatabaseCutoverReceipt:
        codes = tuple(reasons)
        if partial_rollout and REASON_PARTIAL_ROLLOUT not in codes:
            codes = codes + (REASON_PARTIAL_ROLLOUT,)
        receipt = DatabaseCutoverReceipt(
            disposition=RolloutDisposition.DENY,
            from_stage=from_stage,
            to_stage=to_stage,
            authority_mode=self.authority_mode.value,
            binding=self.binding,
            release_gate_valid=release_gate_valid,
            operator_action=operator_action,
            reason_codes=codes,
            history_preserved=True,
            kill_switch_engaged=self.kill_switch_engaged,
            new_program_defaults_to_quack=self.new_programs_default_to_quack(),
            dual_observation=dual_observation_allowed(from_stage),
            detail=detail,
            history_length=len(self._history) + 1,
        )
        self._last_receipt = receipt
        self._append(
            {
                "event": "deny",
                "from_stage": from_stage.value,
                "to_stage": to_stage.value,
                "reason_codes": list(codes),
                "detail": detail,
            }
        )
        return receipt


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def default_passing_evidence(
    *,
    tree_id: str = "tree:hermetic",
    age_seconds: int = 60,
) -> dict[str, dict[str, Any]]:
    """Hermetic evidence map that satisfies the full release gate."""

    return {
        kind: {
            "kind": kind,
            "status": EvidenceStatus.MEASURED.value,
            "evidence_id": f"evidence:{kind}:hermetic",
            "tree_id": tree_id,
            "age_seconds": age_seconds,
            "passed": True,
            "detail": f"hermetic {kind} pass",
        }
        for kind in REQUIRED_EVIDENCE_KINDS
    }


def run_full_cutover(
    *,
    policy: DatabaseRolloutPolicy | None = None,
    binding: DatabaseRolloutBinding | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> tuple[DatabaseRollout, DatabaseCutoverReceipt]:
    """Promote hermetically through the full ladder to default cutover."""

    selected_policy = policy or DatabaseRolloutPolicy.with_default_cutover()
    selected_binding = binding or DatabaseRolloutBinding(
        repository_id="repository:hermetic",
        tree_id="tree:hermetic",
    )
    rollout = DatabaseRollout(selected_policy, binding=selected_binding)
    cells = evidence or default_passing_evidence(tree_id=selected_binding.tree_id)
    # off -> observe -> shadow -> assist need no canary evidence.
    for stage in (
        DatabaseRolloutStage.OBSERVE,
        DatabaseRolloutStage.SHADOW,
        DatabaseRolloutStage.ASSIST,
    ):
        receipt = rollout.promote(stage, evidence=cells)
        if receipt.disposition is not RolloutDisposition.PROMOTE:
            return rollout, receipt
    # canary + default require the full gate.
    for stage in (DatabaseRolloutStage.CANARY, DatabaseRolloutStage.DEFAULT):
        receipt = rollout.promote(
            stage,
            evidence=cells,
            server_available=True,
            backup_age_seconds=0,
            beta_waiver_recorded=True,
        )
        if receipt.disposition is not RolloutDisposition.PROMOTE:
            return rollout, receipt
    return rollout, receipt


__all__ = (
    "DATABASE_CUTOVER_RECEIPT_INTERFACE",
    "DATABASE_ROLLOUT_POLICY_INTERFACE",
    "DEFAULT_MAX_BACKUP_AGE_SECONDS",
    "DEFAULT_QUACK_BETA_LIMITATIONS",
    "DEFAULT_QUACK_PROFILE",
    "EVIDENCE",
    "GOAL_ID",
    "REASON_BACKUP_AGE",
    "REASON_BETA_WAIVER",
    "REASON_DEFAULT_REQUIRES_CANARY",
    "REASON_HISTORY_PRESERVED",
    "REASON_KILL_SWITCH",
    "REASON_LEGACY_EXPORT",
    "REASON_MISSING_EVIDENCE",
    "REASON_NO_LEGACY_DUAL_WRITE",
    "REASON_PARTIAL_ROLLOUT",
    "REASON_PROMOTION_DENIED",
    "REASON_REMOTE_PROHIBITION",
    "REASON_ROLLBACK",
    "REASON_SERVER_UNAVAILABLE",
    "REASON_STALE_EVIDENCE",
    "REQUIRED_EVIDENCE_KINDS",
    "TASK_ID",
    "DatabaseCutoverReceipt",
    "DatabaseRollout",
    "DatabaseRolloutBinding",
    "DatabaseRolloutError",
    "DatabaseRolloutPolicy",
    "DatabaseRolloutStage",
    "EvidenceStatus",
    "OperatorAction",
    "ReleaseGateEvaluation",
    "RollbackReceipt",
    "RolloutDisposition",
    "RolloutEvidenceCell",
    "authority_mode_for_stage",
    "closed_rollout_stages",
    "content_identity",
    "default_passing_evidence",
    "dual_observation_allowed",
    "run_full_cutover",
)
