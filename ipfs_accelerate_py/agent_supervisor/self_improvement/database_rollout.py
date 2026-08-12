"""Staged DuckDB/Quack canary cutover, rollback, and operator policy (DQP-038).

Interfaces: ``DatabaseRolloutPolicy@1``, ``DatabaseCutoverReceipt@1``

Stages:

```text
off -> observe -> shadow -> assist -> canary -> default
                                          \\-> rollback
```

Promotion is evidence-gated and fail-closed. Default cutover is serialized and
requires current chaos, canary, churn/quality, and shadow receipts for the
exact tree/schema/profile. Rollback changes the authority route without
deleting history or accepting legacy dual writes.
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

from ..task_sources.task_source import StateAuthorityMode
from ..task_sources.quack_capabilities import DEFAULT_QUACK_BETA_LIMITATIONS


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_ROLLOUT_POLICY_INTERFACE: Final[str] = "DatabaseRolloutPolicy@1"
DATABASE_CUTOVER_RECEIPT_INTERFACE: Final[str] = "DatabaseCutoverReceipt@1"
ROLLOUT_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-038"
GOAL_ID: Final[str] = "DQP-G080"
EVIDENCE: Final[str] = "dqp/database-rollout@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
POLICY_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-rollout-policy@1"
CUTOVER_RECEIPT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-cutover-receipt@1"
STAGE_BINDING_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-rollout-stage-binding@1"
EVIDENCE_BUNDLE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/database-rollout-evidence@1"

MAX_TEXT_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 256
DEFAULT_EVIDENCE_MAX_AGE_SECONDS: Final[int] = 86_400
DEFAULT_BACKUP_MAX_AGE_SECONDS: Final[int] = 86_400

# Required evidence roots for canary → default promotion.
REQUIRED_EVIDENCE_ROOTS: Final[tuple[str, ...]] = (
    "chaos",
    "canary",
    "churn_quality",
    "shadow",
    "backup",
    "schema",
    "quack_profile",
)

# Stage → authority mode mapping.
STAGE_AUTHORITY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "off": StateAuthorityMode.LEGACY_IMPORT.value,
        "observe": StateAuthorityMode.EMBEDDED_MAINTENANCE.value,
        "shadow": StateAuthorityMode.QUACK_SHADOW.value,
        "assist": StateAuthorityMode.QUACK_SHADOW.value,
        "canary": StateAuthorityMode.QUACK_AUTHORITATIVE.value,
        "default": StateAuthorityMode.QUACK_AUTHORITATIVE.value,
        "rollback": StateAuthorityMode.EMBEDDED_MAINTENANCE.value,
    }
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RolloutStage(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    SHADOW = "shadow"
    ASSIST = "assist"
    CANARY = "canary"
    DEFAULT = "default"
    ROLLBACK = "rollback"


class CutoverVerdict(str, Enum):
    PROMOTED = "promoted"
    HELD = "held"
    DENIED = "denied"
    ROLLED_BACK = "rolled_back"


class DenialReason(str, Enum):
    STALE_EVIDENCE = "stale_evidence"
    MISSING_EVIDENCE = "missing_evidence"
    PARTIAL_ROLLOUT = "partial_rollout"
    SERVER_UNAVAILABLE = "server_unavailable"
    BACKUP_AGE = "backup_age"
    KILL_SWITCH = "kill_switch"
    BETA_WAIVER_REQUIRED = "beta_waiver_required"
    REMOTE_PROHIBITED = "remote_prohibited"
    ILLEGAL_TRANSITION = "illegal_transition"
    DUAL_WRITE_FORBIDDEN = "dual_write_forbidden"
    SYNTHETIC_EVIDENCE = "synthetic_evidence"
    SKIPPED_EVIDENCE = "skipped_evidence"


# Allowed forward transitions (kill-switch graph).
_FORWARD: Final[Mapping[RolloutStage, frozenset[RolloutStage]]] = MappingProxyType(
    {
        RolloutStage.OFF: frozenset({RolloutStage.OBSERVE, RolloutStage.ROLLBACK}),
        RolloutStage.OBSERVE: frozenset(
            {RolloutStage.SHADOW, RolloutStage.OFF, RolloutStage.ROLLBACK}
        ),
        RolloutStage.SHADOW: frozenset(
            {RolloutStage.ASSIST, RolloutStage.OBSERVE, RolloutStage.ROLLBACK}
        ),
        RolloutStage.ASSIST: frozenset(
            {RolloutStage.CANARY, RolloutStage.SHADOW, RolloutStage.ROLLBACK}
        ),
        RolloutStage.CANARY: frozenset(
            {RolloutStage.DEFAULT, RolloutStage.ASSIST, RolloutStage.ROLLBACK}
        ),
        RolloutStage.DEFAULT: frozenset({RolloutStage.ROLLBACK, RolloutStage.CANARY}),
        RolloutStage.ROLLBACK: frozenset(
            {
                RolloutStage.OFF,
                RolloutStage.OBSERVE,
                RolloutStage.SHADOW,
                RolloutStage.ASSIST,
                RolloutStage.CANARY,
            }
        ),
    }
)


class DatabaseRolloutError(ValueError):
    """Fail-closed rejection for illegal rollout transitions or evidence."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _nonnegative_int(value: Any, name: str, *, maximum: int = 10**18) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatabaseRolloutError(f"{name} must be a non-negative integer")
    if value < 0 or value > maximum:
        raise DatabaseRolloutError(f"{name} out of bounds")
    return value


def content_identity(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_stage(value: Any) -> RolloutStage:
    if isinstance(value, RolloutStage):
        return value
    text = _text(value, "stage", maximum=32)
    try:
        return RolloutStage(text)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RolloutStage)
        raise DatabaseRolloutError(
            f"stage must be one of {{{allowed}}}; got {text!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Evidence + policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceItem:
    """One named evidence root bound to tree/schema/profile."""

    root: str
    identity: str
    age_seconds: int
    passed: bool
    synthetic: bool = False
    skipped: bool = False
    tree_id: str = ""
    schema_checksum: str = ""
    profile_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _text(self.root, "root", maximum=64))
        object.__setattr__(
            self, "identity", _text(self.identity, "identity", maximum=128)
        )
        object.__setattr__(
            self, "age_seconds", _nonnegative_int(self.age_seconds, "age_seconds")
        )
        if self.tree_id:
            object.__setattr__(
                self, "tree_id", _text(self.tree_id, "tree_id", maximum=256)
            )
        if self.schema_checksum:
            object.__setattr__(
                self,
                "schema_checksum",
                _text(self.schema_checksum, "schema_checksum", maximum=128),
            )
        if self.profile_id:
            object.__setattr__(
                self, "profile_id", _text(self.profile_id, "profile_id", maximum=128)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "identity": self.identity,
            "age_seconds": self.age_seconds,
            "passed": self.passed,
            "synthetic": self.synthetic,
            "skipped": self.skipped,
            "tree_id": self.tree_id,
            "schema_checksum": self.schema_checksum,
            "profile_id": self.profile_id,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    SCHEMA: ClassVar[str] = EVIDENCE_BUNDLE_SCHEMA

    items: tuple[EvidenceItem, ...]
    tree_id: str
    schema_checksum: str
    store_generation: int
    quack_profile: str
    server_available: bool = True
    remote_endpoint: bool = False
    beta_waiver: bool = False
    backup_age_seconds: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self, "schema_checksum", _text(self.schema_checksum, "schema_checksum")
        )
        object.__setattr__(
            self,
            "store_generation",
            _nonnegative_int(self.store_generation, "store_generation"),
        )
        object.__setattr__(
            self, "quack_profile", _text(self.quack_profile, "quack_profile")
        )
        object.__setattr__(
            self,
            "backup_age_seconds",
            _nonnegative_int(self.backup_age_seconds, "backup_age_seconds"),
        )

    def by_root(self) -> Mapping[str, EvidenceItem]:
        return {item.root: item for item in self.items}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "items": [item.to_dict() for item in self.items],
            "tree_id": self.tree_id,
            "schema_checksum": self.schema_checksum,
            "store_generation": self.store_generation,
            "quack_profile": self.quack_profile,
            "server_available": self.server_available,
            "remote_endpoint": self.remote_endpoint,
            "beta_waiver": self.beta_waiver,
            "backup_age_seconds": self.backup_age_seconds,
        }


@dataclass(frozen=True)
class DatabaseRolloutPolicy:
    """``DatabaseRolloutPolicy@1`` sealed promotion policy."""

    SCHEMA: ClassVar[str] = POLICY_SCHEMA
    INTERFACE: ClassVar[str] = DATABASE_ROLLOUT_POLICY_INTERFACE

    evidence_max_age_seconds: int = DEFAULT_EVIDENCE_MAX_AGE_SECONDS
    backup_max_age_seconds: int = DEFAULT_BACKUP_MAX_AGE_SECONDS
    require_all_evidence_roots: bool = True
    allow_remote: bool = False
    require_beta_waiver_for_default: bool = True
    kill_switch_engaged: bool = False
    allow_legacy_dual_write: bool = False
    required_roots: tuple[str, ...] = REQUIRED_EVIDENCE_ROOTS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_max_age_seconds",
            _nonnegative_int(
                self.evidence_max_age_seconds, "evidence_max_age_seconds"
            ),
        )
        object.__setattr__(
            self,
            "backup_max_age_seconds",
            _nonnegative_int(self.backup_max_age_seconds, "backup_max_age_seconds"),
        )
        object.__setattr__(
            self,
            "required_roots",
            tuple(
                _text(item, "required_roots.item", maximum=64)
                for item in self.required_roots
            ),
        )
        if self.allow_legacy_dual_write:
            raise DatabaseRolloutError(
                "policy must not allow legacy dual writes after cutover"
            )

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": ROLLOUT_CONTRACT_VERSION,
            "evidence_max_age_seconds": self.evidence_max_age_seconds,
            "backup_max_age_seconds": self.backup_max_age_seconds,
            "require_all_evidence_roots": self.require_all_evidence_roots,
            "allow_remote": self.allow_remote,
            "require_beta_waiver_for_default": self.require_beta_waiver_for_default,
            "kill_switch_engaged": self.kill_switch_engaged,
            "allow_legacy_dual_write": False,
            "required_roots": list(self.required_roots),
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


@dataclass(frozen=True)
class StageBinding:
    SCHEMA: ClassVar[str] = STAGE_BINDING_SCHEMA

    stage: RolloutStage
    authority_mode: str
    roots: tuple[str, ...]
    store_generation: int
    schema_checksum: str
    quack_profile: str
    kill_switch: bool
    operator_action: str
    expiry_seconds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stage": self.stage.value if isinstance(self.stage, Enum) else self.stage,
            "authority_mode": self.authority_mode,
            "roots": list(self.roots),
            "store_generation": self.store_generation,
            "schema_checksum": self.schema_checksum,
            "quack_profile": self.quack_profile,
            "kill_switch": self.kill_switch,
            "operator_action": self.operator_action,
            "expiry_seconds": self.expiry_seconds,
        }


@dataclass(frozen=True)
class DatabaseCutoverReceipt:
    """``DatabaseCutoverReceipt@1`` promotion / hold / rollback receipt."""

    SCHEMA: ClassVar[str] = CUTOVER_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = DATABASE_CUTOVER_RECEIPT_INTERFACE

    verdict: CutoverVerdict
    from_stage: RolloutStage
    to_stage: RolloutStage
    binding: StageBinding
    policy_identity: str
    evidence_identity: str
    history_preserved: bool
    dual_write_accepted: bool
    denial_reasons: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    beta_limitations: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utc_iso)
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verdict",
            self.verdict
            if isinstance(self.verdict, CutoverVerdict)
            else CutoverVerdict(str(self.verdict)),
        )
        object.__setattr__(self, "from_stage", parse_stage(self.from_stage))
        object.__setattr__(self, "to_stage", parse_stage(self.to_stage))
        # Dual writes never accepted on cutover path.
        object.__setattr__(self, "dual_write_accepted", False)
        object.__setattr__(
            self,
            "denial_reasons",
            tuple(
                _text(item, "denial_reasons.item", maximum=96)
                for item in self.denial_reasons[:MAX_REASON_CODES]
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _text(item, "reason_codes.item", maximum=96)
                for item in self.reason_codes[:MAX_REASON_CODES]
            ),
        )

    @property
    def promoted(self) -> bool:
        return self.verdict is CutoverVerdict.PROMOTED

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": ROLLOUT_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "verdict": self.verdict.value
            if isinstance(self.verdict, Enum)
            else self.verdict,
            "promoted": self.promoted,
            "from_stage": self.from_stage.value,
            "to_stage": self.to_stage.value,
            "binding": self.binding.to_dict(),
            "policy_identity": self.policy_identity,
            "evidence_identity": self.evidence_identity,
            "history_preserved": self.history_preserved,
            "dual_write_accepted": False,
            "denial_reasons": list(self.denial_reasons),
            "reason_codes": list(self.reason_codes),
            "beta_limitations": list(self.beta_limitations),
            "created_at": self.created_at,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DatabaseRollout:
    """Serialized staged cutover controller."""

    INTERFACE: ClassVar[str] = DATABASE_ROLLOUT_POLICY_INTERFACE

    def __init__(
        self,
        policy: DatabaseRolloutPolicy | None = None,
        *,
        initial_stage: RolloutStage = RolloutStage.OFF,
    ) -> None:
        self.policy = policy or DatabaseRolloutPolicy()
        self.stage = parse_stage(initial_stage)
        self._history: list[dict[str, Any]] = [
            {"event": "init", "stage": self.stage.value}
        ]
        self._legacy_export_only = False

    @property
    def authority_mode(self) -> str:
        return STAGE_AUTHORITY[self.stage.value]

    def allowed_transitions(self) -> frozenset[RolloutStage]:
        return _FORWARD[self.stage]

    def evaluate_promotion_gates(
        self,
        target: RolloutStage,
        evidence: EvidenceBundle,
    ) -> tuple[bool, tuple[str, ...]]:
        """Return (allowed, denial_reasons) for a prospective transition."""

        target = parse_stage(target)
        denials: list[str] = []

        if target not in self.allowed_transitions():
            denials.append(DenialReason.ILLEGAL_TRANSITION.value)

        if self.policy.kill_switch_engaged and target is not RolloutStage.ROLLBACK:
            denials.append(DenialReason.KILL_SWITCH.value)

        if not evidence.server_available and target in {
            RolloutStage.CANARY,
            RolloutStage.DEFAULT,
            RolloutStage.ASSIST,
        }:
            denials.append(DenialReason.SERVER_UNAVAILABLE.value)

        if evidence.remote_endpoint and not self.policy.allow_remote:
            denials.append(DenialReason.REMOTE_PROHIBITED.value)

        if target is RolloutStage.DEFAULT:
            if (
                self.policy.require_beta_waiver_for_default
                and not evidence.beta_waiver
            ):
                denials.append(DenialReason.BETA_WAIVER_REQUIRED.value)
            if evidence.backup_age_seconds > self.policy.backup_max_age_seconds:
                denials.append(DenialReason.BACKUP_AGE.value)

            by_root = evidence.by_root()
            for root in self.policy.required_roots:
                item = by_root.get(root)
                if item is None:
                    denials.append(f"{DenialReason.MISSING_EVIDENCE.value}:{root}")
                    continue
                if item.skipped:
                    denials.append(f"{DenialReason.SKIPPED_EVIDENCE.value}:{root}")
                if item.synthetic:
                    denials.append(f"{DenialReason.SYNTHETIC_EVIDENCE.value}:{root}")
                if not item.passed:
                    denials.append(f"{DenialReason.PARTIAL_ROLLOUT.value}:{root}")
                if item.age_seconds > self.policy.evidence_max_age_seconds:
                    denials.append(f"{DenialReason.STALE_EVIDENCE.value}:{root}")
                if item.tree_id and item.tree_id != evidence.tree_id:
                    denials.append(f"tree_mismatch:{root}")
                if (
                    item.schema_checksum
                    and item.schema_checksum != evidence.schema_checksum
                ):
                    denials.append(f"schema_mismatch:{root}")
                if item.profile_id and item.profile_id != evidence.quack_profile:
                    denials.append(f"profile_mismatch:{root}")

        if self.policy.allow_legacy_dual_write:
            denials.append(DenialReason.DUAL_WRITE_FORBIDDEN.value)

        return (not denials, tuple(denials))

    def transition(
        self,
        target: RolloutStage | str,
        evidence: EvidenceBundle,
        *,
        operator_action: str = "promote",
        force_rollback: bool = False,
    ) -> DatabaseCutoverReceipt:
        """Attempt a staged transition under policy and evidence gates."""

        target_stage = parse_stage(target)
        from_stage = self.stage

        if force_rollback or target_stage is RolloutStage.ROLLBACK:
            # Rollback always allowed as kill switch; never deletes history.
            binding = StageBinding(
                stage=RolloutStage.ROLLBACK,
                authority_mode=STAGE_AUTHORITY[RolloutStage.ROLLBACK.value],
                roots=tuple(self.policy.required_roots),
                store_generation=evidence.store_generation,
                schema_checksum=evidence.schema_checksum,
                quack_profile=evidence.quack_profile,
                kill_switch=True,
                operator_action="rollback",
            )
            self.stage = RolloutStage.ROLLBACK
            self._legacy_export_only = True
            self._history.append(
                {
                    "event": "rollback",
                    "from": from_stage.value,
                    "to": RolloutStage.ROLLBACK.value,
                    "history_length": len(self._history),
                }
            )
            return DatabaseCutoverReceipt(
                verdict=CutoverVerdict.ROLLED_BACK,
                from_stage=from_stage,
                to_stage=RolloutStage.ROLLBACK,
                binding=binding,
                policy_identity=self.policy.identity_id,
                evidence_identity=content_identity(evidence.to_dict()),
                history_preserved=True,
                dual_write_accepted=False,
                reason_codes=("rollback",),
                beta_limitations=tuple(DEFAULT_QUACK_BETA_LIMITATIONS),
            )

        allowed, denials = self.evaluate_promotion_gates(target_stage, evidence)
        binding = StageBinding(
            stage=target_stage if allowed else from_stage,
            authority_mode=STAGE_AUTHORITY[
                (target_stage if allowed else from_stage).value
            ],
            roots=tuple(self.policy.required_roots),
            store_generation=evidence.store_generation,
            schema_checksum=evidence.schema_checksum,
            quack_profile=evidence.quack_profile,
            kill_switch=self.policy.kill_switch_engaged,
            operator_action=_text(operator_action, "operator_action", maximum=64),
        )

        if not allowed:
            verdict = (
                CutoverVerdict.HELD
                if DenialReason.STALE_EVIDENCE.value
                in " ".join(denials)
                or any(d.startswith("stale_evidence") for d in denials)
                else CutoverVerdict.DENIED
            )
            self._history.append(
                {
                    "event": "denied",
                    "from": from_stage.value,
                    "to": target_stage.value,
                    "denials": list(denials),
                }
            )
            return DatabaseCutoverReceipt(
                verdict=verdict,
                from_stage=from_stage,
                to_stage=from_stage,
                binding=binding,
                policy_identity=self.policy.identity_id,
                evidence_identity=content_identity(evidence.to_dict()),
                history_preserved=True,
                dual_write_accepted=False,
                denial_reasons=denials,
                reason_codes=denials,
                beta_limitations=tuple(DEFAULT_QUACK_BETA_LIMITATIONS),
            )

        self.stage = target_stage
        if target_stage is RolloutStage.DEFAULT:
            self._legacy_export_only = True
        self._history.append(
            {
                "event": "promoted",
                "from": from_stage.value,
                "to": target_stage.value,
            }
        )
        return DatabaseCutoverReceipt(
            verdict=CutoverVerdict.PROMOTED,
            from_stage=from_stage,
            to_stage=target_stage,
            binding=binding,
            policy_identity=self.policy.identity_id,
            evidence_identity=content_identity(evidence.to_dict()),
            history_preserved=True,
            dual_write_accepted=False,
            reason_codes=("promoted",),
            beta_limitations=tuple(DEFAULT_QUACK_BETA_LIMITATIONS),
        )


def hermetic_passing_evidence(
    *,
    tree_id: str = "tree:sha256:dqp038",
    schema_checksum: str = "sha256:" + ("aa" * 32),
    store_generation: int = 1,
    quack_profile: str = "profile:quack-1.5.2-loopback",
    beta_waiver: bool = True,
    backup_age_seconds: int = 600,
    age_seconds: int = 300,
) -> EvidenceBundle:
    """Build a full current evidence bundle that satisfies default promotion."""

    items = [
        EvidenceItem(
            root=root,
            identity=f"evidence:{root}:pass",
            age_seconds=age_seconds,
            passed=True,
            tree_id=tree_id,
            schema_checksum=schema_checksum,
            profile_id=quack_profile,
        )
        for root in REQUIRED_EVIDENCE_ROOTS
    ]
    return EvidenceBundle(
        items=tuple(items),
        tree_id=tree_id,
        schema_checksum=schema_checksum,
        store_generation=store_generation,
        quack_profile=quack_profile,
        server_available=True,
        remote_endpoint=False,
        beta_waiver=beta_waiver,
        backup_age_seconds=backup_age_seconds,
    )


def run_staged_cutover_to_default(
    *,
    policy: DatabaseRolloutPolicy | None = None,
    evidence: EvidenceBundle | None = None,
) -> tuple[DatabaseRollout, list[DatabaseCutoverReceipt]]:
    """Drive off→…→default under hermetic passing evidence."""

    controller = DatabaseRollout(policy=policy or DatabaseRolloutPolicy())
    bundle = evidence or hermetic_passing_evidence()
    path = (
        RolloutStage.OBSERVE,
        RolloutStage.SHADOW,
        RolloutStage.ASSIST,
        RolloutStage.CANARY,
        RolloutStage.DEFAULT,
    )
    receipts: list[DatabaseCutoverReceipt] = []
    for stage in path:
        receipts.append(controller.transition(stage, bundle))
        if not receipts[-1].promoted:
            break
    return controller, receipts


__all__ = (
    "DATABASE_CUTOVER_RECEIPT_INTERFACE",
    "DATABASE_ROLLOUT_POLICY_INTERFACE",
    "DEFAULT_BACKUP_MAX_AGE_SECONDS",
    "DEFAULT_EVIDENCE_MAX_AGE_SECONDS",
    "EVIDENCE",
    "GOAL_ID",
    "REQUIRED_EVIDENCE_ROOTS",
    "STAGE_AUTHORITY",
    "TASK_ID",
    "CutoverVerdict",
    "DatabaseCutoverReceipt",
    "DatabaseRollout",
    "DatabaseRolloutError",
    "DatabaseRolloutPolicy",
    "DenialReason",
    "EvidenceBundle",
    "EvidenceItem",
    "RolloutStage",
    "StageBinding",
    "content_identity",
    "hermetic_passing_evidence",
    "parse_stage",
    "run_staged_cutover_to_default",
)
