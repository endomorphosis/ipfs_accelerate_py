"""SPAR-040 shadow_plan rollout gate.

This module extends current supervisor operational authority with the
``ShadowPlanGate@1`` rollout gate.  It nominates activation of ``shadow_plan``
from sealed ``bootstrap``, then runs complete analysis and planning without
source mutation, merge, root promotion, or routing influence.

False and unsafe candidates are compared and retained as negative evidence.
Soft scores never override a hard constraint.  Workers cannot change rollout
mode.  Later modes (``shadow_apply``, ``guarded``, ``required``) remain sealed
behind their own gate tasks.

The gate is nomination-only: it cannot authorize a transition, completion,
merge, or competing authority.  Observational metadata is excluded from
identity.  Dry-run is deterministic and never mutates.  Network is denied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)


TASK_ID: Final[str] = "SPAR-040"
GOAL_ID: Final[str] = "SPAR-G073"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "operational refactoring authority"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.rollout@1"
)
PREDECESSOR_TASK_IDS: Final[tuple[str, ...]] = ("SPAR-038", "SPAR-039")

SHADOW_PLAN_GATE_INTERFACE: Final[str] = "ShadowPlanGate@1"
ROLLOUT_MODE_INTERFACE: Final[str] = "RolloutMode@1"
MODE_CONTRACT_INTERFACE: Final[str] = "RolloutModeContract@1"
SHADOW_PLAN_CANDIDATE_INTERFACE: Final[str] = "ShadowPlanCandidate@1"
CANDIDATE_COMPARISON_INTERFACE: Final[str] = "ShadowPlanCandidateComparison@1"
SHADOW_PLAN_RECEIPT_INTERFACE: Final[str] = "ShadowPlanReceipt@1"
TYPED_TERMINAL_INTERFACE: Final[str] = "TypedTerminal@1"

SHADOW_PLAN_GATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-plan-gate@1"
)
ROLLOUT_MODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rollout-mode@1"
)
MODE_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/rollout-mode-contract@1"
)
SHADOW_PLAN_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-plan-candidate@1"
)
CANDIDATE_COMPARISON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-plan-candidate-comparison@1"
)
SHADOW_PLAN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-plan-receipt@1"
)
TYPED_TERMINAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/shadow-plan-typed-terminal@1"
)
ROLLOUT_BASELINE_SCHEMA: Final[str] = "spar/rollout-baseline@1"

ROLLOUT_CONTRACT_VERSION: Final[str] = "1"

GATE_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
GATE_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
GATE_CAN_CREATE_AUTHORITY: Final[bool] = False
GATE_CAN_CHANGE_MODE: Final[bool] = False
GATE_WRITES_REPOSITORY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
WORKER_MAY_CHANGE_MODE: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
GATE_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
NETWORK_DENIED: Final[bool] = True
NETWORK_DENY: Final[str] = "deny"
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
NEGATIVE_EVIDENCE_RETAINED: Final[bool] = True
SHADOW_PLAN_SOURCE_MUTATION: Final[bool] = False
SHADOW_PLAN_MERGE: Final[bool] = False
SHADOW_PLAN_INFLUENCES_ROUTING: Final[bool] = False
SHADOW_PLAN_PROMOTES_ROOT: Final[bool] = False
COMPLETE_ANALYSIS_REQUIRED: Final[bool] = True
COMPLETE_PLANNING_REQUIRED: Final[bool] = True
FALSE_UNSAFE_CANDIDATES_RETAINED: Final[bool] = True
BOOTSTRAP_UNTIL_SPAR_040: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_CANDIDATES: Final[int] = 16_384
MAX_VIOLATIONS: Final[int] = 4_096
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_ROOT_GENERATION: Final[int] = 1_000_000_000

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

ROLLOUT_MODES: Final[tuple[str, ...]] = (
    "bootstrap",
    "shadow_plan",
    "shadow_apply",
    "guarded",
    "required",
)
DECLARED_ROLLOUT_MODES: Final[frozenset[str]] = frozenset(ROLLOUT_MODES)
SHADOW_PLAN_MODE: Final[str] = "shadow_plan"
BOOTSTRAP_MODE: Final[str] = "bootstrap"
ALLOWED_CURRENT_MODES: Final[frozenset[str]] = frozenset(
    {BOOTSTRAP_MODE, SHADOW_PLAN_MODE}
)
PROGRESSIVE_MODES: Final[tuple[str, ...]] = (
    "shadow_apply",
    "guarded",
    "required",
)

MODE_CONTRACTS: Final[Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        "bootstrap": MappingProxyType(
            {
                "source_mutation": False,
                "merge": False,
                "gate_task": "SPAR-000",
            }
        ),
        "shadow_plan": MappingProxyType(
            {
                "source_mutation": False,
                "merge": False,
                "gate_task": "SPAR-040",
            }
        ),
        "shadow_apply": MappingProxyType(
            {
                "source_mutation": "disposable_worktree_only",
                "merge": False,
                "gate_task": "SPAR-041",
            }
        ),
        "guarded": MappingProxyType(
            {
                "source_mutation": "Tier A and qualified Tier B",
                "merge": "current authority gates",
                "gate_task": "SPAR-042",
            }
        ),
        "required": MappingProxyType(
            {
                "source_mutation": "receipt-bound by autonomy tier",
                "merge": "current authority gates",
                "gate_task": "SPAR-043",
            }
        ),
    }
)

RECEIPT_FLOOR: Final[tuple[str, ...]] = (
    "pre_world_root_cid",
    "program_graph_snapshot_cid",
    "partition_candidate_cid",
    "boundary_contract_set_cid",
    "transformation_packet_cid",
    "context_receipt_cid",
    "route_decision_cid",
    "validation_receipt_cids",
    "refactor_transition_cid",
    "post_world_root_cid",
    "expected_root_generation",
    "resulting_root_generation",
    "rollout_mode",
)
DECLARED_RECEIPT_FLOOR: Final[frozenset[str]] = frozenset(RECEIPT_FLOOR)

HARD_CONSTRAINTS: Final[tuple[str, ...]] = (
    "scc",
    "state",
    "consumer",
    "compatibility",
    "ordering",
    "resource",
    "frontier",
    "proof",
    "transaction",
)
DECLARED_HARD_CONSTRAINTS: Final[frozenset[str]] = frozenset(HARD_CONSTRAINTS)

CANDIDATE_DISPOSITIONS: Final[tuple[str, ...]] = ("admitted", "false", "unsafe")
DECLARED_CANDIDATE_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    CANDIDATE_DISPOSITIONS
)

EXISTING_ADAPTER_AUTHORITIES: Final[tuple[str, ...]] = (
    "datasets_semantic",
    "kit_storage",
    "kit_vfs",
    "accelerator_supervisor",
    "accelerator_runtime",
    "spar_narrow",
)

_FORBIDDEN_CAPSULE_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FunctionSemanticCapsule",
        "MethodSemanticCapsule",
        "ClassSemanticCapsule",
        "TopLevelBlockCapsule",
        "ModuleSemanticCapsule",
        "PackageSemanticCapsule",
        "CallsiteSemanticCapsule",
        "StateOwnerCapsule",
        "RegistrationCapsule",
        "ResourceLifecycleCapsule",
        "RolloutStore",
    }
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "can_change_mode",
    "projection_is_authority",
    "writes_repository",
    "worker_self_approval",
    "worker_may_change_mode",
    "influences_routing",
    "source_mutation",
    "merge",
)

FORBIDDEN_ROLLOUT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "RolloutStore",
        "authorize_mode_change",
        "worker_set_mode",
        "mutate_source",
        "influence_routing",
        "admit_by_score",
        "self_approve_rollout",
        "promote_root",
        "skip_sealed_gate",
    }
)


class RolloutError(ValueError):
    """Fail-closed violation of a SPAR-040 shadow_plan rollout contract."""


class RolloutMode(str, Enum):
    BOOTSTRAP = "bootstrap"
    SHADOW_PLAN = "shadow_plan"
    SHADOW_APPLY = "shadow_apply"
    GUARDED = "guarded"
    REQUIRED = "required"


class GateStatus(str, Enum):
    NOMINATED_SHADOW_PLAN = "nominated_shadow_plan"
    TYPED_TERMINAL = "typed_terminal"


class TerminalKind(str, Enum):
    UNSUPPORTED = "unsupported"
    HUMAN_REVIEW = "human_review"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


class CandidateDisposition(str, Enum):
    ADMITTED = "admitted"
    FALSE = "false"
    UNSAFE = "unsafe"


DECLARED_GATE_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in GateStatus
)
DECLARED_TERMINAL_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in TerminalKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise RolloutError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise RolloutError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise RolloutError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise RolloutError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise RolloutError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise RolloutError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RolloutError(f"{name} must be a boolean")
    return value


def _int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is bool or type(value) is not int:
        raise RolloutError(f"{name} must be an integer")
    if value < minimum:
        raise RolloutError(f"{name} is out of range")
    if maximum is not None and value > maximum:
        raise RolloutError(f"{name} is out of range")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise RolloutError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise RolloutError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise RolloutError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise RolloutError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise RolloutError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise RolloutError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise RolloutError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise RolloutError(f"{name} does not verify")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise RolloutError(f"unknown {name}: {text}") from exc


def _project(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _project(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_project(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _project(to_dict())
    raise RolloutError(f"unsupported projected type {type(value).__name__}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    projected = _project(value)
    if not isinstance(projected, dict):
        raise RolloutError(f"{name} must be an object")
    _reject_excluded(projected, name)
    return projected


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, (), []):
        ordered: tuple[str, ...] = ()
    elif not isinstance(values, (list, tuple)):
        raise RolloutError(f"{name} must be a list")
    else:
        ordered = tuple(sorted(_cid(item, name) for item in values))
    if len(ordered) != len(set(ordered)):
        raise RolloutError(f"{name} must not contain duplicates")
    if required and not ordered:
        raise RolloutError(f"{name} must not be empty")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise RolloutError(f"{name} exceeds maximum length")
    return ordered


def _ordered_cids(values: Any, name: str) -> tuple[str, ...]:
    if values in (None, (), []):
        return ()
    if not isinstance(values, (list, tuple)):
        raise RolloutError(f"{name} must be a list")
    ordered = tuple(_cid(item, name) for item in values)
    if len(ordered) != len(set(ordered)):
        raise RolloutError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise RolloutError(f"{name} exceeds maximum length")
    return ordered


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        claimed = payload.pop(flag)
        if flag in {"source_mutation", "merge", "influences_routing"}:
            if claimed is not False:
                raise RolloutError(f"{name} cannot claim {flag}")
            continue
        if claimed is not False:
            raise RolloutError(f"{name} cannot claim {flag}")


def _reject_non_admitting(payload: Mapping[str, Any], name: str) -> None:
    present = _NON_ADMITTING_EVIDENCE & set(payload)
    if present:
        raise RolloutError(
            f"{name} cannot admit a shadow plan from {sorted(present)}"
        )


def _mode(value: Any, name: str = "rollout_mode") -> str:
    return _enum(value, RolloutMode, name)


def rollout_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def sealed_rollout_baseline() -> dict[str, Any]:
    """In-code copy of the sealed SPAR rollout baseline. Not write authority."""

    return {
        "schema": ROLLOUT_BASELINE_SCHEMA,
        "current_mode": BOOTSTRAP_MODE,
        "worker_may_change_mode": False,
        "modes": {
            mode: {
                "source_mutation": dict(MODE_CONTRACTS[mode])["source_mutation"],
                "merge": dict(MODE_CONTRACTS[mode])["merge"],
                "gate_task": dict(MODE_CONTRACTS[mode])["gate_task"],
            }
            for mode in ROLLOUT_MODES
        },
        "receipt_floor": list(RECEIPT_FLOOR),
    }


def rollout_gate_descriptor() -> dict[str, Any]:
    return {
        "schema": SHADOW_PLAN_GATE_SCHEMA,
        "interface": SHADOW_PLAN_GATE_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "analyzer_id": ANALYZER_ID,
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "authority_owner": AUTHORITY_OWNER,
        "nomination_only": True,
        "writes_repository": False,
        "worker_may_change_mode": False,
        "source_mutation": False,
        "merge": False,
        "influences_routing": False,
        "nominated_mode": SHADOW_PLAN_MODE,
        "gate_task": TASK_ID,
        "network": NETWORK_DENY,
        "false_unsafe_candidates_retained": True,
    }


def _parse_terminal(value: Any) -> TypedTerminal | None:
    if value in (None, "", {}):
        return None
    if isinstance(value, TypedTerminal):
        return value
    return TypedTerminal.from_mapping(value)


@dataclass(frozen=True, slots=True)
class TypedTerminal:
    """Unsupported required behavior is a typed terminal, never success."""

    kind: str
    reason: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "kind",
            "reason",
            "terminal_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(self.kind, TerminalKind, "kind"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TYPED_TERMINAL_SCHEMA,
            "interface": TYPED_TERMINAL_INTERFACE,
            "kind": self.kind,
            "reason": self.reason,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def terminal_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["terminal_cid"] = self.terminal_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypedTerminal":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("terminal_cid")
        if payload.pop("schema") != TYPED_TERMINAL_SCHEMA:
            raise RolloutError("unsupported TypedTerminal schema")
        if payload.pop("interface") != TYPED_TERMINAL_INTERFACE:
            raise RolloutError("unsupported TypedTerminal interface")
        result = cls(**payload)
        _verify_cid(claimed, result.terminal_cid, "TypedTerminal terminal_cid")
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | "TypedTerminal") -> "TypedTerminal":
        if isinstance(data, TypedTerminal):
            return data
        if "terminal_cid" in data:
            return cls.from_dict(data)
        payload = _mapping(data, "terminal")
        return cls(kind=payload.get("kind"), reason=payload.get("reason"))


@dataclass(frozen=True, slots=True)
class ModeContract:
    """Sealed mutation/merge/gate contract for one rollout mode."""

    mode: str
    source_mutation: bool | str
    merge: bool | str
    gate_task: str

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "mode",
            "source_mutation",
            "merge",
            "gate_task",
            "contract_cid",
            "worker_may_change_mode",
        }
    )

    def __post_init__(self) -> None:
        mode = _mode(self.mode, "mode")
        object.__setattr__(self, "mode", mode)
        expected = MODE_CONTRACTS[mode]
        if self.source_mutation != expected["source_mutation"]:
            raise RolloutError("source_mutation does not match sealed mode contract")
        if self.merge != expected["merge"]:
            raise RolloutError("merge does not match sealed mode contract")
        object.__setattr__(self, "gate_task", _text(self.gate_task, "gate_task"))
        if self.gate_task != expected["gate_task"]:
            raise RolloutError("gate_task does not match sealed mode contract")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MODE_CONTRACT_SCHEMA,
            "interface": MODE_CONTRACT_INTERFACE,
            "mode": self.mode,
            "source_mutation": self.source_mutation,
            "merge": self.merge,
            "gate_task": self.gate_task,
            "worker_may_change_mode": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def contract_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["contract_cid"] = self.contract_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModeContract":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("contract_cid")
        if payload.pop("schema") != MODE_CONTRACT_SCHEMA:
            raise RolloutError("unsupported ModeContract schema")
        if payload.pop("interface") != MODE_CONTRACT_INTERFACE:
            raise RolloutError("unsupported ModeContract interface")
        if payload.pop("worker_may_change_mode") is not False:
            raise RolloutError("workers cannot change rollout mode")
        result = cls(**payload)
        _verify_cid(claimed, result.contract_cid, "ModeContract contract_cid")
        return result

    @classmethod
    def for_mode(cls, mode: str) -> "ModeContract":
        normalized = _mode(mode, "mode")
        expected = MODE_CONTRACTS[normalized]
        return cls(
            mode=normalized,
            source_mutation=expected["source_mutation"],
            merge=expected["merge"],
            gate_task=expected["gate_task"],
        )


@dataclass(frozen=True, slots=True)
class ShadowPlanCandidate:
    """One planned candidate. Soft scores cannot override hard constraints."""

    candidate_cid: str
    disposition: str
    hard_constraint_violations: Sequence[str] = ()
    score: int = 0

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "candidate_cid",
            "disposition",
            "hard_constraint_violations",
            "score",
            "comparison_cid",
            "score_is_advisory",
            "soft_signals_cannot_override_hard_constraints",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid"))
        object.__setattr__(
            self,
            "disposition",
            _enum(self.disposition, CandidateDisposition, "disposition"),
        )
        object.__setattr__(self, "score", _int(self.score, "score", minimum=0))
        violations = self.hard_constraint_violations
        if violations in (None, (), []):
            ordered: tuple[str, ...] = ()
        elif not isinstance(violations, (list, tuple)):
            raise RolloutError("hard_constraint_violations must be a list")
        else:
            seen: set[str] = set()
            collected: list[str] = []
            for item in violations:
                name = _text(item, "hard_constraint_violations")
                if name not in DECLARED_HARD_CONSTRAINTS:
                    raise RolloutError(f"unknown hard constraint: {name}")
                if name in seen:
                    raise RolloutError("hard_constraint_violations must not contain duplicates")
                seen.add(name)
                collected.append(name)
            ordered = tuple(collected)
        if len(ordered) > MAX_VIOLATIONS:
            raise RolloutError("hard_constraint_violations exceeds maximum length")
        object.__setattr__(self, "hard_constraint_violations", ordered)
        if self.disposition == CandidateDisposition.ADMITTED.value and ordered:
            raise RolloutError("admitted candidates cannot carry hard constraint violations")
        if self.disposition == CandidateDisposition.UNSAFE.value and not ordered:
            raise RolloutError("unsafe candidates require hard constraint violations")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SHADOW_PLAN_CANDIDATE_SCHEMA,
            "interface": SHADOW_PLAN_CANDIDATE_INTERFACE,
            "candidate_cid": self.candidate_cid,
            "disposition": self.disposition,
            "hard_constraint_violations": list(self.hard_constraint_violations),
            "score": self.score,
            "score_is_advisory": True,
            "soft_signals_cannot_override_hard_constraints": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def comparison_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["comparison_cid"] = self.comparison_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowPlanCandidate":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("comparison_cid")
        if payload.pop("schema") != SHADOW_PLAN_CANDIDATE_SCHEMA:
            raise RolloutError("unsupported ShadowPlanCandidate schema")
        if payload.pop("interface") != SHADOW_PLAN_CANDIDATE_INTERFACE:
            raise RolloutError("unsupported ShadowPlanCandidate interface")
        if payload.pop("score_is_advisory") is not True:
            raise RolloutError("candidate scores remain advisory")
        if payload.pop("soft_signals_cannot_override_hard_constraints") is not True:
            raise RolloutError("soft signals cannot override hard constraints")
        result = cls(**payload)
        _verify_cid(claimed, result.comparison_cid, "ShadowPlanCandidate comparison_cid")
        return result

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | "ShadowPlanCandidate"
    ) -> "ShadowPlanCandidate":
        if isinstance(data, ShadowPlanCandidate):
            return data
        if "comparison_cid" in data:
            return cls.from_dict(data)
        payload = _mapping(data, "candidate")
        return cls(
            candidate_cid=payload.get("candidate_cid"),
            disposition=payload.get("disposition"),
            hard_constraint_violations=payload.get("hard_constraint_violations") or (),
            score=payload.get("score", 0),
        )


def _parse_candidates(values: Any) -> tuple[ShadowPlanCandidate, ...]:
    if values in (None, (), []):
        raise RolloutError("shadow_plan requires planned candidates")
    if not isinstance(values, (list, tuple)):
        raise RolloutError("candidates must be a list")
    if len(values) > MAX_CANDIDATES:
        raise RolloutError("candidates exceeds maximum length")
    parsed = tuple(ShadowPlanCandidate.from_mapping(item) for item in values)
    cids = [item.candidate_cid for item in parsed]
    if len(cids) != len(set(cids)):
        raise RolloutError("candidates must not contain duplicate identities")
    return parsed


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    """Deterministic comparison that retains false and unsafe candidates."""

    tree_id: str
    ranked_admitted_cids: Sequence[str]
    false_candidate_cids: Sequence[str]
    unsafe_candidate_cids: Sequence[str]
    retained_negative_cids: Sequence[str]

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "ranked_admitted_cids",
            "false_candidate_cids",
            "unsafe_candidate_cids",
            "retained_negative_cids",
            "comparison_cid",
            "false_unsafe_candidates_retained",
            "soft_signals_cannot_override_hard_constraints",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "ranked_admitted_cids",
            _ordered_cids(list(self.ranked_admitted_cids), "ranked_admitted_cids"),
        )
        object.__setattr__(
            self,
            "false_candidate_cids",
            _cids(list(self.false_candidate_cids), "false_candidate_cids"),
        )
        object.__setattr__(
            self,
            "unsafe_candidate_cids",
            _cids(list(self.unsafe_candidate_cids), "unsafe_candidate_cids"),
        )
        object.__setattr__(
            self,
            "retained_negative_cids",
            _cids(list(self.retained_negative_cids), "retained_negative_cids"),
        )
        admitted = set(self.ranked_admitted_cids)
        false_ids = set(self.false_candidate_cids)
        unsafe_ids = set(self.unsafe_candidate_cids)
        if admitted & false_ids or admitted & unsafe_ids or false_ids & unsafe_ids:
            raise RolloutError("candidate dispositions must be disjoint")
        expected_negative = tuple(sorted(false_ids | unsafe_ids))
        if tuple(self.retained_negative_cids) != expected_negative:
            raise RolloutError("false and unsafe candidates must be retained")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": CANDIDATE_COMPARISON_SCHEMA,
            "interface": CANDIDATE_COMPARISON_INTERFACE,
            "tree_id": self.tree_id,
            "ranked_admitted_cids": list(self.ranked_admitted_cids),
            "false_candidate_cids": list(self.false_candidate_cids),
            "unsafe_candidate_cids": list(self.unsafe_candidate_cids),
            "retained_negative_cids": list(self.retained_negative_cids),
            "false_unsafe_candidates_retained": True,
            "soft_signals_cannot_override_hard_constraints": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def comparison_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["comparison_cid"] = self.comparison_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateComparison":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("comparison_cid")
        if payload.pop("schema") != CANDIDATE_COMPARISON_SCHEMA:
            raise RolloutError("unsupported CandidateComparison schema")
        if payload.pop("interface") != CANDIDATE_COMPARISON_INTERFACE:
            raise RolloutError("unsupported CandidateComparison interface")
        if payload.pop("false_unsafe_candidates_retained") is not True:
            raise RolloutError("false and unsafe candidates must be retained")
        if payload.pop("soft_signals_cannot_override_hard_constraints") is not True:
            raise RolloutError("soft signals cannot override hard constraints")
        result = cls(**payload)
        _verify_cid(claimed, result.comparison_cid, "CandidateComparison comparison_cid")
        return result


def compare_and_retain_candidates(
    candidates: Sequence[ShadowPlanCandidate] | Sequence[Mapping[str, Any]],
    *,
    tree_id: str,
) -> CandidateComparison:
    """Rank admitted candidates; retain false and unsafe as negative evidence."""

    parsed = tuple(
        item
        if isinstance(item, ShadowPlanCandidate)
        else ShadowPlanCandidate.from_mapping(item)
        for item in candidates
    )
    if not parsed:
        raise RolloutError("shadow_plan requires planned candidates")
    admitted = [item for item in parsed if item.disposition == "admitted"]
    false_ids = tuple(
        sorted(item.candidate_cid for item in parsed if item.disposition == "false")
    )
    unsafe_ids = tuple(
        sorted(item.candidate_cid for item in parsed if item.disposition == "unsafe")
    )
    ranked = tuple(
        item.candidate_cid
        for item in sorted(
            admitted,
            key=lambda item: (-item.score, item.candidate_cid),
        )
    )
    return CandidateComparison(
        tree_id=tree_id,
        ranked_admitted_cids=ranked,
        false_candidate_cids=false_ids,
        unsafe_candidate_cids=unsafe_ids,
        retained_negative_cids=tuple(sorted(set(false_ids) | set(unsafe_ids))),
    )


@dataclass(frozen=True, slots=True)
class ShadowPlanReceipt:
    """Nomination-only SPAR-040 shadow_plan rollout receipt."""

    tree_id: str
    status: str
    current_mode: str
    nominated_mode: str
    mode_contract: ModeContract
    comparison: CandidateComparison
    pre_world_root_cid: str
    post_world_root_cid: str
    program_graph_snapshot_cid: str
    partition_candidate_cid: str
    boundary_contract_set_cid: str
    transformation_packet_cid: str
    context_receipt_cid: str
    route_decision_cid: str
    validation_receipt_cids: Sequence[str] = ()
    refactor_transition_cid: str = ""
    expected_root_generation: int = 0
    resulting_root_generation: int = 0
    analyzer_id: str = ANALYZER_ID
    worktree_id: str = ""
    lease_id: str = ""
    fence_id: str = ""
    negative_evidence_cids: Sequence[str] = ()
    terminal: TypedTerminal | None = None

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "status",
            "current_mode",
            "nominated_mode",
            "mode_contract",
            "comparison",
            "pre_world_root_cid",
            "post_world_root_cid",
            "program_graph_snapshot_cid",
            "partition_candidate_cid",
            "boundary_contract_set_cid",
            "transformation_packet_cid",
            "context_receipt_cid",
            "route_decision_cid",
            "validation_receipt_cids",
            "refactor_transition_cid",
            "expected_root_generation",
            "resulting_root_generation",
            "rollout_mode",
            "analyzer_id",
            "worktree_id",
            "lease_id",
            "fence_id",
            "negative_evidence_cids",
            "terminal",
            "receipt_cid",
            "nominated",
            "accepted",
            "gate_is_nomination_only",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_change_mode",
            "projection_is_authority",
            "writes_repository",
            "worker_self_approval",
            "worker_may_change_mode",
            "influences_routing",
            "source_mutation",
            "merge",
            "root_promoted",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "status", _enum(self.status, GateStatus, "status"))
        object.__setattr__(self, "current_mode", _mode(self.current_mode, "current_mode"))
        object.__setattr__(
            self, "nominated_mode", _mode(self.nominated_mode, "nominated_mode")
        )
        if self.nominated_mode != SHADOW_PLAN_MODE:
            raise RolloutError("SPAR-040 can only nominate shadow_plan")
        if self.current_mode not in ALLOWED_CURRENT_MODES:
            raise RolloutError("SPAR-040 cannot skip or regress sealed rollout gates")
        contract = self.mode_contract
        if not isinstance(contract, ModeContract):
            contract = ModeContract.from_dict(contract)
        if contract.mode != SHADOW_PLAN_MODE:
            raise RolloutError("shadow_plan gate must bind the shadow_plan contract")
        object.__setattr__(self, "mode_contract", contract)
        comparison = self.comparison
        if not isinstance(comparison, CandidateComparison):
            comparison = CandidateComparison.from_dict(comparison)
        if comparison.tree_id != self.tree_id:
            raise RolloutError("receipt tree_id does not match comparison")
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(
            self, "pre_world_root_cid", _cid(self.pre_world_root_cid, "pre_world_root_cid")
        )
        object.__setattr__(
            self,
            "post_world_root_cid",
            _cid(self.post_world_root_cid, "post_world_root_cid"),
        )
        if self.post_world_root_cid != self.pre_world_root_cid:
            raise RolloutError("shadow_plan cannot mutate or promote the world root")
        object.__setattr__(
            self,
            "program_graph_snapshot_cid",
            _cid(self.program_graph_snapshot_cid, "program_graph_snapshot_cid"),
        )
        object.__setattr__(
            self,
            "partition_candidate_cid",
            _cid(self.partition_candidate_cid, "partition_candidate_cid"),
        )
        object.__setattr__(
            self,
            "boundary_contract_set_cid",
            _cid(self.boundary_contract_set_cid, "boundary_contract_set_cid"),
        )
        object.__setattr__(
            self,
            "transformation_packet_cid",
            _cid(self.transformation_packet_cid, "transformation_packet_cid"),
        )
        object.__setattr__(
            self, "context_receipt_cid", _cid(self.context_receipt_cid, "context_receipt_cid")
        )
        object.__setattr__(
            self, "route_decision_cid", _cid(self.route_decision_cid, "route_decision_cid")
        )
        object.__setattr__(
            self,
            "validation_receipt_cids",
            _cids(list(self.validation_receipt_cids), "validation_receipt_cids"),
        )
        object.__setattr__(
            self,
            "refactor_transition_cid",
            _optional_cid(self.refactor_transition_cid, "refactor_transition_cid"),
        )
        if self.refactor_transition_cid:
            raise RolloutError("shadow_plan cannot publish a refactor transition")
        object.__setattr__(
            self,
            "expected_root_generation",
            _int(
                self.expected_root_generation,
                "expected_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        object.__setattr__(
            self,
            "resulting_root_generation",
            _int(
                self.resulting_root_generation,
                "resulting_root_generation",
                maximum=MAX_ROOT_GENERATION,
            ),
        )
        if self.resulting_root_generation != self.expected_root_generation:
            raise RolloutError("shadow_plan cannot promote root generation")
        object.__setattr__(self, "analyzer_id", _text(self.analyzer_id, "analyzer_id"))
        if self.analyzer_id != ANALYZER_ID:
            raise RolloutError("analyzer_id must remain SPAR-040")
        object.__setattr__(
            self, "worktree_id", _text(self.worktree_id, "worktree_id", empty=True)
        )
        object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id", empty=True))
        object.__setattr__(self, "fence_id", _text(self.fence_id, "fence_id", empty=True))
        object.__setattr__(
            self,
            "negative_evidence_cids",
            _cids(list(self.negative_evidence_cids), "negative_evidence_cids"),
        )
        expected_negative = tuple(comparison.retained_negative_cids)
        if tuple(self.negative_evidence_cids) != expected_negative:
            raise RolloutError("receipt must retain compared false and unsafe candidates")
        terminal = self.terminal
        if terminal not in (None,):
            if not isinstance(terminal, TypedTerminal):
                terminal = TypedTerminal.from_mapping(terminal)
        else:
            terminal = None
        object.__setattr__(self, "terminal", terminal)
        self._assert_status_invariants()

    def _assert_status_invariants(self) -> None:
        if self.status == GateStatus.NOMINATED_SHADOW_PLAN.value:
            if self.terminal is not None:
                raise RolloutError("nominated shadow_plan cannot carry a typed terminal")
        if self.status == GateStatus.TYPED_TERMINAL.value:
            if self.terminal is None:
                raise RolloutError("typed terminal status requires a typed terminal")
        if (
            self.status != GateStatus.TYPED_TERMINAL.value
            and self.terminal is not None
        ):
            raise RolloutError("non-terminal status cannot carry a typed terminal")

    @property
    def nominated(self) -> bool:
        return self.status == GateStatus.NOMINATED_SHADOW_PLAN.value

    @property
    def accepted(self) -> bool:
        return False

    @property
    def rollout_mode(self) -> str:
        return self.nominated_mode

    @property
    def root_promoted(self) -> bool:
        return False

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SHADOW_PLAN_RECEIPT_SCHEMA,
            "interface": SHADOW_PLAN_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "status": self.status,
            "current_mode": self.current_mode,
            "nominated_mode": self.nominated_mode,
            "rollout_mode": self.nominated_mode,
            "mode_contract": self.mode_contract.to_dict(),
            "comparison": self.comparison.to_dict(),
            "pre_world_root_cid": self.pre_world_root_cid,
            "post_world_root_cid": self.post_world_root_cid,
            "program_graph_snapshot_cid": self.program_graph_snapshot_cid,
            "partition_candidate_cid": self.partition_candidate_cid,
            "boundary_contract_set_cid": self.boundary_contract_set_cid,
            "transformation_packet_cid": self.transformation_packet_cid,
            "context_receipt_cid": self.context_receipt_cid,
            "route_decision_cid": self.route_decision_cid,
            "validation_receipt_cids": list(self.validation_receipt_cids),
            "refactor_transition_cid": self.refactor_transition_cid,
            "expected_root_generation": self.expected_root_generation,
            "resulting_root_generation": self.resulting_root_generation,
            "analyzer_id": ANALYZER_ID,
            "worktree_id": self.worktree_id,
            "lease_id": self.lease_id,
            "fence_id": self.fence_id,
            "negative_evidence_cids": list(self.negative_evidence_cids),
            "terminal": None if self.terminal is None else self.terminal.to_dict(),
            "nominated": self.nominated,
            "accepted": False,
            "gate_is_nomination_only": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_change_mode": False,
            "projection_is_authority": False,
            "writes_repository": False,
            "worker_self_approval": False,
            "worker_may_change_mode": False,
            "influences_routing": False,
            "source_mutation": False,
            "merge": False,
            "root_promoted": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ShadowPlanReceipt":
        _reject_excluded(data, cls.__name__)
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != SHADOW_PLAN_RECEIPT_SCHEMA:
            raise RolloutError("unsupported ShadowPlanReceipt schema")
        if payload.pop("interface") != SHADOW_PLAN_RECEIPT_INTERFACE:
            raise RolloutError("unsupported ShadowPlanReceipt interface")
        if payload.pop("accepted") is not False:
            raise RolloutError("workers cannot self-approve a shadow_plan")
        if payload.pop("gate_is_nomination_only") is not True:
            raise RolloutError("gate must remain nomination_only")
        if payload.pop("rollout_mode") != SHADOW_PLAN_MODE:
            raise RolloutError("receipt rollout_mode must remain shadow_plan")
        if payload.pop("root_promoted") is not False:
            raise RolloutError("shadow_plan cannot promote a root")
        nominated = payload.pop("nominated")
        _pop_authority_flags(payload, "ShadowPlanReceipt")
        payload["mode_contract"] = ModeContract.from_dict(payload["mode_contract"])
        payload["comparison"] = CandidateComparison.from_dict(payload["comparison"])
        if payload["terminal"] is not None:
            payload["terminal"] = TypedTerminal.from_dict(payload["terminal"])
        result = cls(**payload)
        if nominated is not result.nominated:
            raise RolloutError("nominated flag does not match status")
        _verify_cid(claimed, result.receipt_cid, "ShadowPlanReceipt receipt_cid")
        return result


def _require_complete_analysis_and_planning(payload: Mapping[str, Any]) -> None:
    if _bool(payload.get("analysis_complete", False), "analysis_complete") is not True:
        raise RolloutError("shadow_plan requires complete analysis")
    if _bool(payload.get("planning_complete", False), "planning_complete") is not True:
        raise RolloutError("shadow_plan requires complete planning")


def _reject_worker_mode_change(payload: Mapping[str, Any]) -> str:
    if payload.get("worker_may_change_mode", False) is not False:
        raise RolloutError("workers cannot change rollout mode")
    requested = payload.get("requested_mode", "")
    if requested not in (None, "", SHADOW_PLAN_MODE):
        raise RolloutError("workers cannot change rollout mode")
    current = _mode(payload.get("current_mode", BOOTSTRAP_MODE), "current_mode")
    if current not in ALLOWED_CURRENT_MODES:
        raise RolloutError("SPAR-040 cannot skip or regress sealed rollout gates")
    return current


def _reject_mutation_merge_routing(payload: Mapping[str, Any]) -> None:
    if payload.get("mutate", False) is not False:
        raise RolloutError("shadow_plan cannot mutate")
    if payload.get("source_mutation", False) is not False:
        raise RolloutError("shadow_plan cannot mutate source")
    if payload.get("merge", False) is not False:
        raise RolloutError("shadow_plan cannot merge")
    if payload.get("routing_influence", False) is not False:
        raise RolloutError("shadow_plan cannot influence routing")
    if payload.get("influences_routing", False) is not False:
        raise RolloutError("shadow_plan cannot influence routing")
    proposed_route = payload.get("proposed_route_decision_cid", "")
    route = payload.get("route_decision_cid", "")
    if proposed_route not in (None, "", route):
        raise RolloutError("shadow_plan cannot influence routing")
    post = payload.get("post_world_root_cid", "")
    pre = payload.get("pre_world_root_cid", "")
    if post not in (None, "", pre):
        raise RolloutError("shadow_plan cannot mutate or promote the world root")
    resulting = payload.get("resulting_root_generation", payload.get("expected_root_generation"))
    expected = payload.get("expected_root_generation")
    if resulting not in (None, expected):
        raise RolloutError("shadow_plan cannot promote root generation")
    if payload.get("refactor_transition_cid") not in (None, ""):
        raise RolloutError("shadow_plan cannot publish a refactor transition")


def _receipt_from_payload(
    payload: Mapping[str, Any],
    *,
    status: str,
    current_mode: str,
    comparison: CandidateComparison,
    terminal: TypedTerminal | None = None,
) -> ShadowPlanReceipt:
    pre = _cid(payload.get("pre_world_root_cid"), "pre_world_root_cid")
    expected = _int(
        payload.get("expected_root_generation", 0),
        "expected_root_generation",
        maximum=MAX_ROOT_GENERATION,
    )
    return ShadowPlanReceipt(
        tree_id=payload.get("tree_id"),
        status=status,
        current_mode=current_mode,
        nominated_mode=SHADOW_PLAN_MODE,
        mode_contract=ModeContract.for_mode(SHADOW_PLAN_MODE),
        comparison=comparison,
        pre_world_root_cid=pre,
        post_world_root_cid=pre,
        program_graph_snapshot_cid=payload.get("program_graph_snapshot_cid"),
        partition_candidate_cid=payload.get("partition_candidate_cid"),
        boundary_contract_set_cid=payload.get("boundary_contract_set_cid"),
        transformation_packet_cid=payload.get("transformation_packet_cid"),
        context_receipt_cid=payload.get("context_receipt_cid"),
        route_decision_cid=payload.get("route_decision_cid"),
        validation_receipt_cids=payload.get("validation_receipt_cids") or (),
        refactor_transition_cid="",
        expected_root_generation=expected,
        resulting_root_generation=expected,
        worktree_id=payload.get("worktree_id", ""),
        lease_id=payload.get("lease_id", ""),
        fence_id=payload.get("fence_id", ""),
        negative_evidence_cids=comparison.retained_negative_cids,
        terminal=terminal,
    )


def run_shadow_plan(
    evidence: Mapping[str, Any],
    *,
    mutate: bool = False,
) -> ShadowPlanReceipt:
    """Run complete analysis and planning without mutation or routing influence."""

    if mutate is not False:
        raise RolloutError("shadow_plan cannot mutate")
    payload = _mapping(evidence, "shadow-plan evidence")
    _reject_non_admitting(payload, "shadow-plan evidence")
    if payload.get("network", NETWORK_DENY) != NETWORK_DENY:
        raise RolloutError("network is denied")
    if payload.get("mutate", False) is not False:
        raise RolloutError("shadow_plan cannot mutate")
    current_mode = _reject_worker_mode_change(payload)
    _reject_mutation_merge_routing(payload)
    terminal = _parse_terminal(payload.get("terminal"))
    candidates = _parse_candidates(payload.get("candidates"))
    tree_id = _tree_id(payload.get("tree_id"))
    comparison = compare_and_retain_candidates(candidates, tree_id=tree_id)
    if payload.get("analysis_available", True) is not True:
        terminal = TypedTerminal(
            kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
            reason="required analysis capability is unavailable",
        )
    if payload.get("planning_available", True) is not True:
        terminal = TypedTerminal(
            kind=TerminalKind.CAPABILITY_UNAVAILABLE.value,
            reason="required planning capability is unavailable",
        )
    if terminal is not None:
        return _receipt_from_payload(
            payload,
            status=GateStatus.TYPED_TERMINAL.value,
            current_mode=current_mode,
            comparison=comparison,
            terminal=terminal,
        )
    _require_complete_analysis_and_planning(payload)
    return _receipt_from_payload(
        payload,
        status=GateStatus.NOMINATED_SHADOW_PLAN.value,
        current_mode=current_mode,
        comparison=comparison,
    )


def activate_shadow_plan(evidence: Mapping[str, Any]) -> ShadowPlanReceipt:
    """Nominate SPAR-040 activation of shadow_plan from sealed bootstrap."""

    return run_shadow_plan(evidence)


def dry_run_shadow_plan(evidence: Mapping[str, Any]) -> ShadowPlanReceipt:
    """Deterministic dry-run. Never mutates and never accepts a mode change."""

    return run_shadow_plan(evidence, mutate=False)


class ShadowPlanGate:
    """SPAR-040 shadow_plan rollout gate. Nomination-only; workers cannot change mode."""

    interface: ClassVar[str] = SHADOW_PLAN_GATE_INTERFACE
    schema: ClassVar[str] = SHADOW_PLAN_GATE_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def run(
        self,
        evidence: Mapping[str, Any],
        *,
        mutate: bool = False,
    ) -> ShadowPlanReceipt:
        return run_shadow_plan(evidence, mutate=mutate)

    def activate(self, evidence: Mapping[str, Any]) -> ShadowPlanReceipt:
        return activate_shadow_plan(evidence)

    def compare(
        self,
        candidates: Sequence[ShadowPlanCandidate] | Sequence[Mapping[str, Any]],
        *,
        tree_id: str,
    ) -> CandidateComparison:
        return compare_and_retain_candidates(candidates, tree_id=tree_id)

    def dry_run(self, evidence: Mapping[str, Any]) -> ShadowPlanReceipt:
        return dry_run_shadow_plan(evidence)


def encode_canonical_receipt(receipt: ShadowPlanReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> ShadowPlanReceipt:
    return ShadowPlanReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise RolloutError(
            f"shadow_plan gate must not define competing types: {sorted(overlap)}"
        )
    if "RolloutStore" in names:
        raise RolloutError("RolloutStore is not a SPAR rollout contract")


__all__ = [
    "ALLOWED_CURRENT_MODES",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "BOOTSTRAP_MODE",
    "BOOTSTRAP_UNTIL_SPAR_040",
    "CANDIDATE_COMPARISON_INTERFACE",
    "CANDIDATE_DISPOSITIONS",
    "COMPLETE_ANALYSIS_REQUIRED",
    "COMPLETE_PLANNING_REQUIRED",
    "CandidateComparison",
    "CandidateDisposition",
    "DECLARED_CANDIDATE_DISPOSITIONS",
    "DECLARED_GATE_STATUSES",
    "DECLARED_HARD_CONSTRAINTS",
    "DECLARED_RECEIPT_FLOOR",
    "DECLARED_ROLLOUT_MODES",
    "DECLARED_TERMINAL_KINDS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXISTING_ADAPTER_AUTHORITIES",
    "FALSE_UNSAFE_CANDIDATES_RETAINED",
    "FORBIDDEN_ROLLOUT_NAMES",
    "GATE_CAN_AUTHORIZE_COMPLETION",
    "GATE_CAN_AUTHORIZE_TRANSITION",
    "GATE_CAN_CHANGE_MODE",
    "GATE_CAN_CREATE_AUTHORITY",
    "GATE_IS_NOMINATION_ONLY",
    "GATE_WRITES_REPOSITORY",
    "GOAL_ID",
    "HARD_CONSTRAINTS",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODE_CONTRACT_INTERFACE",
    "MODE_CONTRACTS",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "ModeContract",
    "NEGATIVE_EVIDENCE_RETAINED",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "PREDECESSOR_TASK_IDS",
    "PROGRAM",
    "PROGRESSIVE_MODES",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "RECEIPT_FLOOR",
    "ROLLOUT_BASELINE_SCHEMA",
    "ROLLOUT_CONTRACT_VERSION",
    "ROLLOUT_MODES",
    "ROLLOUT_MODE_INTERFACE",
    "RolloutError",
    "RolloutMode",
    "SHADOW_PLAN_CANDIDATE_INTERFACE",
    "SHADOW_PLAN_GATE_INTERFACE",
    "SHADOW_PLAN_INFLUENCES_ROUTING",
    "SHADOW_PLAN_MERGE",
    "SHADOW_PLAN_MODE",
    "SHADOW_PLAN_PROMOTES_ROOT",
    "SHADOW_PLAN_RECEIPT_INTERFACE",
    "SHADOW_PLAN_SOURCE_MUTATION",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "ShadowPlanCandidate",
    "ShadowPlanGate",
    "ShadowPlanReceipt",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TYPED_TERMINAL_INTERFACE",
    "TerminalKind",
    "TypedTerminal",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_MAY_CHANGE_MODE",
    "WORKER_SELF_APPROVAL",
    "activate_shadow_plan",
    "assert_not_competing_capsule_family",
    "compare_and_retain_candidates",
    "decode_canonical_receipt",
    "dry_run_shadow_plan",
    "encode_canonical_receipt",
    "provider_free_exports",
    "rollout_cid_profile",
    "rollout_gate_descriptor",
    "run_shadow_plan",
    "sealed_rollout_baseline",
]
