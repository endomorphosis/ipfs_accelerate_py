"""SPAR-034 proof-carrying procedure adapter for accepted refactor waves.

This module extends the current procedure authority with
``ProofCarryingProcedureRefactorAdapter@1``.  It consumes SPAR-025 extraction
waves, SPAR-031 normalization receipts, SPAR-032 unpromoted synthesis
candidates, and SPAR-033 accepted transitions, then:

* normalizes accepted trajectories;
* anti-unifies compatible plans;
* infers preconditions, effects, and rollback;
* preserves allowed typed holes;
* qualifies held-out and adversarial cases; and
* nominates promotion through the existing procedure authority.

``ProofCarryingProcedureCompiler`` remains the procedure authority.  This
adapter does not mint a competing procedure, certificate, registry, or
completion authority.  A procedure cannot self-certify.  Promotion requires
an independent held-out or adversarial qualification route.  Vector, model,
and heuristic evidence cannot admit a procedure.  Paths and credentials are
never turned into parameters.  Validation is never dropped.  Forbidden hole
types fail closed.  Unsupported required behavior is a typed terminal, never
success.

The adapter is nomination-only.  Observational metadata is excluded from
identity.  Dry-run is deterministic and never mutates.  Network is denied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Callable, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)


TASK_ID: Final[str] = "SPAR-034"
GOAL_ID: Final[str] = "SPAR-G062"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "procedure compilation"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.procedure_adapter@1"
)
PROCEDURE_AUTHORITY: Final[str] = "ProofCarryingProcedureCompiler"
PROCEDURE_AUTHORITY_INTERFACE: Final[str] = "ProofCarryingProcedureCompiler@1"
PROCEDURE_AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"

PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_INTERFACE: Final[str] = (
    "ProofCarryingProcedureRefactorAdapter@1"
)
NORMALIZED_REFACTOR_TRAJECTORY_INTERFACE: Final[str] = "NormalizedRefactorTrajectory@1"
ANTI_UNIFIED_REFACTOR_PLAN_INTERFACE: Final[str] = "AntiUnifiedRefactorPlan@1"
INFERRED_PROCEDURE_CONTRACT_INTERFACE: Final[str] = "InferredProcedureContract@1"
PRESERVED_HOLE_INTERFACE: Final[str] = "PreservedHole@1"
HELD_OUT_QUALIFICATION_INTERFACE: Final[str] = "HeldOutQualification@1"
ADVERSARIAL_QUALIFICATION_INTERFACE: Final[str] = "AdversarialQualification@1"
PROCEDURE_PROMOTION_NOMINATION_INTERFACE: Final[str] = (
    "ProcedurePromotionNomination@1"
)
REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE: Final[str] = (
    "RefactorProcedureCompilationReceipt@1"
)

PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-procedure-refactor-adapter@1"
)
NORMALIZED_REFACTOR_TRAJECTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/normalized-refactor-trajectory@1"
)
ANTI_UNIFIED_REFACTOR_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/anti-unified-refactor-plan@1"
)
INFERRED_PROCEDURE_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/inferred-procedure-contract@1"
)
PRESERVED_HOLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/preserved-refactor-hole@1"
)
HELD_OUT_QUALIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/held-out-procedure-qualification@1"
)
ADVERSARIAL_QUALIFICATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-procedure-qualification@1"
)
PROCEDURE_PROMOTION_NOMINATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-promotion-nomination@1"
)
REFACTOR_PROCEDURE_COMPILATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-procedure-compilation-receipt@1"
)

PROCEDURE_ADAPTER_CONTRACT_VERSION: Final[str] = "1"

PROCEDURE_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
PROCEDURE_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
PROCEDURE_CAN_CREATE_AUTHORITY: Final[bool] = False
PROCEDURE_CAN_CREATE_PROCEDURE_AUTHORITY: Final[bool] = False
PROCEDURE_CANNOT_SELF_CERTIFY: Final[bool] = True
PROCEDURE_CANNOT_SELF_PROMOTE: Final[bool] = True
HELD_OUT_QUALIFICATION_REQUIRED_FOR_PROMOTION: Final[bool] = True
ADVERSARIAL_QUALIFICATION_ROUTE_EXISTS: Final[bool] = True
CANDIDATES_REMAIN_UNPROMOTED: Final[bool] = True
HOLES_PRESERVED: Final[bool] = True
VALIDATION_NEVER_DROPPED: Final[bool] = True
PATHS_NEVER_BECOME_PARAMETERS: Final[bool] = True
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
TEST_PASS_IS_NOT_PROOF: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
ADAPTER_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
NETWORK_DENIED: Final[bool] = True
NETWORK_DENY: Final[str] = "deny"
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
UNKNOWN_REMAINS_UNKNOWN: Final[bool] = True
INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS: Final[bool] = True
GENERAL_PYTHON_EQUIVALENCE_CLAIMED: Final[bool] = False
IMPLICIT_INSTALL_FORBIDDEN: Final[bool] = True
IMPLICIT_NETWORK_FORBIDDEN: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_TRAJECTORIES: Final[int] = 256
MAX_HOLES: Final[int] = 128
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_PRECONDITIONS: Final[int] = 64
MAX_EFFECTS: Final[int] = 64
MAX_LOST_DETAILS: Final[int] = 256
MAX_COUNTEREXAMPLES: Final[int] = 256

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

DEFAULT_PRECONDITIONS: Final[tuple[str, ...]] = (
    "accepted_predecessor",
    "exact_preimage",
    "raw_source",
)
DEFAULT_EFFECTS: Final[tuple[str, ...]] = ("bounded_internal_implementation",)
DEFAULT_ROLLBACK: Final[str] = "restore_exact_preimages"

ALLOWED_HOLE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "SELECT_ONE_OF_ALLOWED_SYMBOLS",
        "GENERATE_DOCSTRING",
        "PROPOSE_BOUNDED_PATCH",
        "CLASSIFY_FAILURE",
        "CHOOSE_APPROVED_REPAIR_TEMPLATE",
        "SUGGEST_MISSING_TEST_CASE",
        "SUGGEST_LEMMA",
    }
)
FORBIDDEN_HOLE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "AUTHORITY_DECISION",
        "POLICY_DECISION",
        "CONFIRMATION",
        "TRUSTED_KEY_SELECTION",
        "TEST_OMISSION",
        "PROOF_ACCEPTANCE",
        "RELEASE_PROMOTION",
        "TASK_COMPLETION",
        "UNBOUNDED_SHELL_COMMAND",
    }
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
    "can_create_procedure_authority",
    "projection_is_authority",
    "self_certified",
    "self_promoted",
)

_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "body",
        "code",
        "contents",
        "content",
        "file_content",
        "file_contents",
        "file_text",
        "full_ast",
        "full_graph",
        "full_proof",
        "full_source",
        "full_trace",
        "lean_source",
        "proof_body",
        "proof_text",
        "proof_transcript",
        "prover_output",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "transcript",
        "witness",
        "snippet",
        "solver_trace",
        "raw_output",
        "hidden_witness",
        "private_witness",
    }
)

FORBIDDEN_PROCEDURE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "self_certify",
        "self_promote",
        "authorize_completion",
        "authorize_transition",
        "create_procedure_authority",
        "drop_holes",
        "omit_validation",
        "admit_by_similarity",
        "suppress_raw_source",
        "open_network",
        "claim_general_equivalence",
        "promote_without_qualification",
        "path_parameter",
    }
)

_SELF_PRODUCERS: Final[frozenset[str]] = frozenset(
    {
        ANALYZER_ID,
        "self",
        "self-promoted",
        "self-certified",
        "procedure_adapter",
        "ProofCarryingProcedureRefactorAdapter",
        TASK_ID,
    }
)

ADVERSARIAL_SOURCE_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "specification",
        "replayed_counterexample",
        "independently_observed",
        "runtime_observation",
    }
)

PromoteFn = Callable[..., Mapping[str, Any]]


class ProcedureAdapterError(ValueError):
    """Fail-closed violation of a SPAR-034 procedure-adapter contract."""


class TrajectoryOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class CompilationStatus(str, Enum):
    NOMINATED = "nominated"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    REFUSED = "refused"


class QualificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    MISSING = "missing"


class QualificationRoute(str, Enum):
    HELD_OUT = "held_out"
    ADVERSARIAL = "adversarial"


class AdversarialPolarity(str, Enum):
    MUST_SURVIVE = "must_survive"
    MUST_FAIL_CLOSED = "must_fail_closed"


DECLARED_COMPILATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in CompilationStatus
)
DECLARED_QUALIFICATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in QualificationStatus
)
DECLARED_QUALIFICATION_ROUTES: Final[frozenset[str]] = frozenset(
    item.value for item in QualificationRoute
)
DECLARED_OUTCOMES: Final[frozenset[str]] = frozenset(
    item.value for item in TrajectoryOutcome
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise ProcedureAdapterError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ProcedureAdapterError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise ProcedureAdapterError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise ProcedureAdapterError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise ProcedureAdapterError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise ProcedureAdapterError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProcedureAdapterError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ProcedureAdapterError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise ProcedureAdapterError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise ProcedureAdapterError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise ProcedureAdapterError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise ProcedureAdapterError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise ProcedureAdapterError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _reject_forbidden_bodies(payload: Any, name: str) -> None:
    if isinstance(payload, Mapping) and not isinstance(payload, (str, bytes, bytearray)):
        present = _FORBIDDEN_BODY_KEYS & set(payload)
        if present:
            raise ProcedureAdapterError(
                f"{name} must remain body-free; forbidden keys: {sorted(present)}"
            )
        for key, item in payload.items():
            _reject_forbidden_bodies(item, f"{name}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, item in enumerate(payload):
            _reject_forbidden_bodies(item, f"{name}[{index}]")


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise ProcedureAdapterError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> str:
    cid = _cid(claimed, name)
    if cid != computed:
        raise ProcedureAdapterError(f"{name} does not verify")
    return cid


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise ProcedureAdapterError(f"{name} cannot claim {flag}")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ProcedureAdapterError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise ProcedureAdapterError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise ProcedureAdapterError(f"{name} must not contain duplicates")
    return ordered


def _unique_ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ProcedureAdapterError(f"{name} must be a list")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if len(ordered) > limit:
        raise ProcedureAdapterError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _cids(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProcedureAdapterError(f"{name} must be a list of CIDs")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise ProcedureAdapterError(f"{name} must not be empty")
    if len(ordered) != len(set(ordered)):
        raise ProcedureAdapterError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise ProcedureAdapterError(f"{name} exceeds maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise ProcedureAdapterError(f"{name} exceeds path bound")
    normalized = raw.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or normalized in {".", ""}
        or normalized.startswith("./")
        or any(char in normalized for char in "*?[]{}")
        or "//" in normalized
        or normalized.endswith("/")
    ):
        raise ProcedureAdapterError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise ProcedureAdapterError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProcedureAdapterError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ProcedureAdapterError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise ProcedureAdapterError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProcedureAdapterError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise ProcedureAdapterError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise ProcedureAdapterError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise ProcedureAdapterError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise ProcedureAdapterError(f"missing {name}")
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        _reject_excluded(value, name)
        _reject_forbidden_bodies(value, name)
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping) and not isinstance(
            payload, (str, bytes, bytearray)
        ):
            _reject_excluded(payload, name)
            _reject_forbidden_bodies(payload, name)
            return dict(payload)
    raise ProcedureAdapterError(f"{name} must be a mapping")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProcedureAdapterError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        items.append(_as_mapping(item, name))
    if len(items) > MAX_MEMBERS:
        raise ProcedureAdapterError(f"{name} exceeds maximum length")
    return tuple(items)


def _outcome_value(value: Any, name: str = "outcome") -> str:
    if isinstance(value, TrajectoryOutcome):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_OUTCOMES:
        raise ProcedureAdapterError(f"unsupported {name} {text!r}")
    return text


def _status_value(value: Any, name: str = "status") -> str:
    if isinstance(value, CompilationStatus):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_COMPILATION_STATUSES:
        raise ProcedureAdapterError(f"unsupported {name} {text!r}")
    return text


def _qualification_status(value: Any, name: str = "status") -> str:
    if isinstance(value, QualificationStatus):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_QUALIFICATION_STATUSES:
        raise ProcedureAdapterError(f"unsupported {name} {text!r}")
    return text


def _network_value(value: Any) -> str:
    text = _text(value, "network")
    if text != NETWORK_DENY:
        raise ProcedureAdapterError("network is denied")
    return NETWORK_DENY


def _evidence_class(value: Any, name: str = "evidence_class") -> str:
    return _text(value, name)


def _hole_type(value: Any, name: str = "hole_type") -> str:
    text = _text(value, name)
    if text in FORBIDDEN_HOLE_TYPES:
        raise ProcedureAdapterError(f"forbidden hole type {text}")
    if text not in ALLOWED_HOLE_TYPES:
        raise ProcedureAdapterError(f"unknown hole type {text}")
    return text


def _reject_non_admitting(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("suppress_raw_source") is True:
        raise ProcedureAdapterError("vectors cannot suppress raw-source fallback")
    evidence = payload.get("evidence_class")
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise ProcedureAdapterError(
            f"vector, model, or heuristic evidence cannot admit {name}"
        )
    for flag in ("admit_procedure", "admit_proof", "admit_equivalence", "admit_promotion"):
        if payload.get(flag) is True:
            raise ProcedureAdapterError(
                f"vector, model, or heuristic evidence cannot admit {name}"
            )


def procedure_adapter_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def procedure_adapter_descriptor() -> dict[str, Any]:
    return {
        "interface": PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "procedure_authority": PROCEDURE_AUTHORITY,
        "procedure_authority_interface": PROCEDURE_AUTHORITY_INTERFACE,
        "predicted_symbols": ("ProofCarryingProcedure refactor adapter",),
        "raw_source_required": True,
        "nomination_only": True,
        "network": NETWORK_DENY,
        "procedure_cannot_self_certify": True,
        "procedure_cannot_self_promote": True,
        "held_out_or_adversarial_qualification_required": True,
        "holes_preserved": True,
        "validation_never_dropped": True,
        "paths_never_become_parameters": True,
        "candidates_remain_unpromoted": True,
        "claims_general_equivalence": False,
        "unknown_remains_unknown": True,
        "forbids": tuple(sorted(FORBIDDEN_PROCEDURE_NAMES)),
    }


@dataclass(frozen=True, slots=True)
class NormalizedRefactorTrajectory:
    """Body-free accepted or rejected trajectory bound to one SPAR-025 wave."""

    tree_id: str
    wave_receipt_cid: str
    packet_cid: str
    transition_cid: str
    write_paths: Sequence[str]
    validation_commands: Sequence[str]
    raw_source_cids: Sequence[str]
    preconditions: Sequence[str]
    effects: Sequence[str]
    rollback: str
    outcome: str
    evidence_class: str = "transition"
    holes: Sequence[str] = ()

    interface: ClassVar[str] = NORMALIZED_REFACTOR_TRAJECTORY_INTERFACE
    schema: ClassVar[str] = NORMALIZED_REFACTOR_TRAJECTORY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "wave_receipt_cid",
            "packet_cid",
            "transition_cid",
            "write_paths",
            "validation_commands",
            "raw_source_cids",
            "preconditions",
            "effects",
            "rollback",
            "outcome",
            "evidence_class",
            "holes",
            "adapter_is_nomination_only",
            "raw_source_required",
            "network",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "trajectory_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self, "wave_receipt_cid", _cid(self.wave_receipt_cid, "wave_receipt_cid")
        )
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(
            self, "transition_cid", _cid(self.transition_cid, "transition_cid")
        )
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self, "validation_commands", _commands(self.validation_commands)
        )
        object.__setattr__(
            self,
            "raw_source_cids",
            _cids(self.raw_source_cids, "raw_source_cids", required=True),
        )
        object.__setattr__(
            self,
            "preconditions",
            _unique_ordered_text(
                self.preconditions, "preconditions", limit=MAX_PRECONDITIONS
            ),
        )
        object.__setattr__(
            self,
            "effects",
            _unique_ordered_text(self.effects, "effects", limit=MAX_EFFECTS),
        )
        object.__setattr__(self, "rollback", _text(self.rollback, "rollback"))
        outcome = _outcome_value(self.outcome)
        evidence = _evidence_class(self.evidence_class)
        if evidence in _NON_ADMITTING_EVIDENCE and outcome == TrajectoryOutcome.ACCEPTED.value:
            raise ProcedureAdapterError(
                "vector, model, or heuristic evidence cannot admit a trajectory"
            )
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "evidence_class", evidence)
        holes = _unique_ordered_text(self.holes, "holes", limit=MAX_HOLES)
        for hole in holes:
            _hole_type(hole, "holes")
        object.__setattr__(self, "holes", holes)

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
            "schema": NORMALIZED_REFACTOR_TRAJECTORY_SCHEMA,
            "interface": NORMALIZED_REFACTOR_TRAJECTORY_INTERFACE,
            "tree_id": self.tree_id,
            "wave_receipt_cid": self.wave_receipt_cid,
            "packet_cid": self.packet_cid,
            "transition_cid": self.transition_cid,
            "write_paths": list(self.write_paths),
            "validation_commands": list(self.validation_commands),
            "raw_source_cids": list(self.raw_source_cids),
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "rollback": self.rollback,
            "outcome": self.outcome,
            "evidence_class": self.evidence_class,
            "holes": list(self.holes),
            "adapter_is_nomination_only": True,
            "raw_source_required": True,
            "network": NETWORK_DENY,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def trajectory_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["trajectory_cid"] = self.trajectory_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizedRefactorTrajectory":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("trajectory_cid")
        if payload.pop("schema") != NORMALIZED_REFACTOR_TRAJECTORY_SCHEMA:
            raise ProcedureAdapterError("unsupported NormalizedRefactorTrajectory schema")
        if payload.pop("interface") != NORMALIZED_REFACTOR_TRAJECTORY_INTERFACE:
            raise ProcedureAdapterError(
                "unsupported NormalizedRefactorTrajectory interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ProcedureAdapterError("adapter must remain nomination_only")
        if payload.pop("raw_source_required") is not True:
            raise ProcedureAdapterError("raw_source_required cannot be disabled")
        if payload.pop("network") != NETWORK_DENY:
            raise ProcedureAdapterError("network must remain deny")
        result = cls(**payload)
        _verify_cid(claimed, result.trajectory_cid, "trajectory_cid")
        return result


def compile_normalized_trajectory(**fields: Any) -> NormalizedRefactorTrajectory:
    extra = set(fields) - {
        "tree_id",
        "wave_receipt_cid",
        "packet_cid",
        "transition_cid",
        "write_paths",
        "validation_commands",
        "raw_source_cids",
        "preconditions",
        "effects",
        "rollback",
        "outcome",
        "evidence_class",
        "holes",
    }
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise ProcedureAdapterError(
            f"trajectory identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise ProcedureAdapterError(f"unknown trajectory field: {sorted(extra)}")
    return NormalizedRefactorTrajectory(**fields)


@dataclass(frozen=True, slots=True)
class PreservedHole:
    """Allowed typed residual retained from anti-unification."""

    hole_id: str
    hole_type: str
    origin: str
    required: bool = True

    interface: ClassVar[str] = PRESERVED_HOLE_INTERFACE
    schema: ClassVar[str] = PRESERVED_HOLE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "hole_id",
            "hole_type",
            "origin",
            "required",
            "hole_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "hole_id", _text(self.hole_id, "hole_id"))
        object.__setattr__(self, "hole_type", _hole_type(self.hole_type))
        object.__setattr__(self, "origin", _text(self.origin, "origin"))
        object.__setattr__(self, "required", _bool(self.required, "required"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PRESERVED_HOLE_SCHEMA,
            "interface": PRESERVED_HOLE_INTERFACE,
            "hole_id": self.hole_id,
            "hole_type": self.hole_type,
            "origin": self.origin,
            "required": self.required,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def hole_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["hole_cid"] = self.hole_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PreservedHole":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("hole_cid")
        if payload.pop("schema") != PRESERVED_HOLE_SCHEMA:
            raise ProcedureAdapterError("unsupported PreservedHole schema")
        if payload.pop("interface") != PRESERVED_HOLE_INTERFACE:
            raise ProcedureAdapterError("unsupported PreservedHole interface")
        result = cls(**payload)
        _verify_cid(claimed, result.hole_cid, "hole_cid")
        return result


def compile_preserved_hole(**fields: Any) -> PreservedHole:
    return PreservedHole(**fields)


@dataclass(frozen=True, slots=True)
class AntiUnifiedRefactorPlan:
    """Structural anti-unification of accepted trajectories. Nomination-only."""

    tree_id: str
    trajectory_cids: Sequence[str]
    shared_write_paths: Sequence[str]
    shared_validation_commands: Sequence[str]
    shared_preconditions: Sequence[str]
    shared_effects: Sequence[str]
    shared_rollback: str
    holes: Sequence[PreservedHole]
    lost_details: Sequence[str] = ()
    counterexample_cids: Sequence[str] = ()
    anti_unified: bool = True

    interface: ClassVar[str] = ANTI_UNIFIED_REFACTOR_PLAN_INTERFACE
    schema: ClassVar[str] = ANTI_UNIFIED_REFACTOR_PLAN_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "trajectory_cids",
            "shared_write_paths",
            "shared_validation_commands",
            "shared_preconditions",
            "shared_effects",
            "shared_rollback",
            "holes",
            "lost_details",
            "counterexample_cids",
            "anti_unified",
            "paths_never_become_parameters",
            "validation_never_dropped",
            "adapter_is_nomination_only",
            "can_authorize_completion",
            "plan_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "trajectory_cids",
            _cids(self.trajectory_cids, "trajectory_cids", required=True),
        )
        object.__setattr__(
            self,
            "shared_write_paths",
            _exact_paths(self.shared_write_paths, "shared_write_paths", required=True),
        )
        object.__setattr__(
            self,
            "shared_validation_commands",
            _commands(self.shared_validation_commands, "shared_validation_commands"),
        )
        object.__setattr__(
            self,
            "shared_preconditions",
            _unique_ordered_text(
                self.shared_preconditions,
                "shared_preconditions",
                limit=MAX_PRECONDITIONS,
            ),
        )
        object.__setattr__(
            self,
            "shared_effects",
            _unique_ordered_text(
                self.shared_effects, "shared_effects", limit=MAX_EFFECTS
            ),
        )
        object.__setattr__(
            self, "shared_rollback", _text(self.shared_rollback, "shared_rollback")
        )
        holes = tuple(
            item if isinstance(item, PreservedHole) else PreservedHole.from_dict(item)
            for item in self.holes
        )
        if len(holes) > MAX_HOLES:
            raise ProcedureAdapterError("holes exceed maximum length")
        object.__setattr__(self, "holes", holes)
        object.__setattr__(
            self,
            "lost_details",
            _unique_ordered_text(
                self.lost_details, "lost_details", limit=MAX_LOST_DETAILS
            ),
        )
        object.__setattr__(
            self,
            "counterexample_cids",
            _cids(self.counterexample_cids, "counterexample_cids", required=False),
        )
        object.__setattr__(self, "anti_unified", _bool(self.anti_unified, "anti_unified"))
        if self.anti_unified and self.counterexample_cids:
            raise ProcedureAdapterError(
                "anti-unified plans cannot retain immutable counterexamples"
            )
        if not self.anti_unified and not self.counterexample_cids:
            raise ProcedureAdapterError(
                "failed anti-unification requires immutable counterexamples"
            )

    @property
    def can_authorize_completion(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ANTI_UNIFIED_REFACTOR_PLAN_SCHEMA,
            "interface": ANTI_UNIFIED_REFACTOR_PLAN_INTERFACE,
            "tree_id": self.tree_id,
            "trajectory_cids": list(self.trajectory_cids),
            "shared_write_paths": list(self.shared_write_paths),
            "shared_validation_commands": list(self.shared_validation_commands),
            "shared_preconditions": list(self.shared_preconditions),
            "shared_effects": list(self.shared_effects),
            "shared_rollback": self.shared_rollback,
            "holes": [hole.to_dict() for hole in self.holes],
            "lost_details": list(self.lost_details),
            "counterexample_cids": list(self.counterexample_cids),
            "anti_unified": self.anti_unified,
            "paths_never_become_parameters": True,
            "validation_never_dropped": True,
            "adapter_is_nomination_only": True,
            "can_authorize_completion": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def plan_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["plan_cid"] = self.plan_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AntiUnifiedRefactorPlan":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("plan_cid")
        if payload.pop("schema") != ANTI_UNIFIED_REFACTOR_PLAN_SCHEMA:
            raise ProcedureAdapterError("unsupported AntiUnifiedRefactorPlan schema")
        if payload.pop("interface") != ANTI_UNIFIED_REFACTOR_PLAN_INTERFACE:
            raise ProcedureAdapterError("unsupported AntiUnifiedRefactorPlan interface")
        if payload.pop("paths_never_become_parameters") is not True:
            raise ProcedureAdapterError("paths must never become parameters")
        if payload.pop("validation_never_dropped") is not True:
            raise ProcedureAdapterError("validation must never be dropped")
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ProcedureAdapterError("adapter must remain nomination_only")
        _pop_authority_flags(payload, cls.__name__)
        result = cls(**payload)
        _verify_cid(claimed, result.plan_cid, "plan_cid")
        return result


@dataclass(frozen=True, slots=True)
class InferredProcedureContract:
    """Preconditions, effects, and rollback inferred from an anti-unified plan."""

    plan_cid: str
    preconditions: Sequence[str]
    effects: Sequence[str]
    rollback: str
    postconditions: Sequence[str] = ()
    invariants: Sequence[str] = ()

    interface: ClassVar[str] = INFERRED_PROCEDURE_CONTRACT_INTERFACE
    schema: ClassVar[str] = INFERRED_PROCEDURE_CONTRACT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "plan_cid",
            "preconditions",
            "effects",
            "rollback",
            "postconditions",
            "invariants",
            "adapter_is_nomination_only",
            "can_authorize_completion",
            "contract_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self,
            "preconditions",
            _unique_ordered_text(
                self.preconditions, "preconditions", limit=MAX_PRECONDITIONS
            ),
        )
        if not self.preconditions:
            raise ProcedureAdapterError("preconditions must not be empty")
        object.__setattr__(
            self,
            "effects",
            _unique_ordered_text(self.effects, "effects", limit=MAX_EFFECTS),
        )
        if not self.effects:
            raise ProcedureAdapterError("effects must not be empty")
        object.__setattr__(self, "rollback", _text(self.rollback, "rollback"))
        object.__setattr__(
            self,
            "postconditions",
            _unique_ordered_text(
                self.postconditions, "postconditions", limit=MAX_PRECONDITIONS
            ),
        )
        object.__setattr__(
            self,
            "invariants",
            _unique_ordered_text(self.invariants, "invariants", limit=MAX_PRECONDITIONS),
        )

    @property
    def can_authorize_completion(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": INFERRED_PROCEDURE_CONTRACT_SCHEMA,
            "interface": INFERRED_PROCEDURE_CONTRACT_INTERFACE,
            "plan_cid": self.plan_cid,
            "preconditions": list(self.preconditions),
            "effects": list(self.effects),
            "rollback": self.rollback,
            "postconditions": list(self.postconditions),
            "invariants": list(self.invariants),
            "adapter_is_nomination_only": True,
            "can_authorize_completion": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "InferredProcedureContract":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("contract_cid")
        if payload.pop("schema") != INFERRED_PROCEDURE_CONTRACT_SCHEMA:
            raise ProcedureAdapterError("unsupported InferredProcedureContract schema")
        if payload.pop("interface") != INFERRED_PROCEDURE_CONTRACT_INTERFACE:
            raise ProcedureAdapterError(
                "unsupported InferredProcedureContract interface"
            )
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ProcedureAdapterError("adapter must remain nomination_only")
        _pop_authority_flags(payload, cls.__name__)
        result = cls(**payload)
        _verify_cid(claimed, result.contract_cid, "contract_cid")
        return result


@dataclass(frozen=True, slots=True)
class HeldOutQualification:
    """Disjoint held-out qualification. Passing does not self-certify."""

    status: str
    held_out_trajectory_cids: Sequence[str] = ()
    reasons: Sequence[str] = ()

    interface: ClassVar[str] = HELD_OUT_QUALIFICATION_INTERFACE
    schema: ClassVar[str] = HELD_OUT_QUALIFICATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "route",
            "status",
            "held_out_trajectory_cids",
            "reasons",
            "self_certified",
            "can_authorize_completion",
            "qualification_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _qualification_status(self.status))
        object.__setattr__(
            self,
            "held_out_trajectory_cids",
            _cids(
                self.held_out_trajectory_cids,
                "held_out_trajectory_cids",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            _unique_ordered_text(self.reasons, "reasons", limit=MAX_COUNTEREXAMPLES),
        )
        if self.status == QualificationStatus.MISSING.value and self.held_out_trajectory_cids:
            raise ProcedureAdapterError("missing held-out qualification has no trajectories")
        if (
            self.status == QualificationStatus.PASSED.value
            and not self.held_out_trajectory_cids
        ):
            raise ProcedureAdapterError("passed held-out qualification requires trajectories")

    @property
    def route(self) -> str:
        return QualificationRoute.HELD_OUT.value

    @property
    def self_certified(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": HELD_OUT_QUALIFICATION_SCHEMA,
            "interface": HELD_OUT_QUALIFICATION_INTERFACE,
            "route": self.route,
            "status": self.status,
            "held_out_trajectory_cids": list(self.held_out_trajectory_cids),
            "reasons": list(self.reasons),
            "self_certified": False,
            "can_authorize_completion": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def qualification_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["qualification_cid"] = self.qualification_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HeldOutQualification":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("qualification_cid")
        if payload.pop("schema") != HELD_OUT_QUALIFICATION_SCHEMA:
            raise ProcedureAdapterError("unsupported HeldOutQualification schema")
        if payload.pop("interface") != HELD_OUT_QUALIFICATION_INTERFACE:
            raise ProcedureAdapterError("unsupported HeldOutQualification interface")
        if payload.pop("route") != QualificationRoute.HELD_OUT.value:
            raise ProcedureAdapterError("held-out route must remain held_out")
        if payload.pop("self_certified") is not False:
            raise ProcedureAdapterError("procedure cannot self-certify")
        _pop_authority_flags(payload, cls.__name__)
        result = cls(**payload)
        _verify_cid(claimed, result.qualification_cid, "qualification_cid")
        return result


@dataclass(frozen=True, slots=True)
class AdversarialQualification:
    """Adversarial qualification route. Passing does not self-certify."""

    status: str
    case_ids: Sequence[str] = ()
    reasons: Sequence[str] = ()

    interface: ClassVar[str] = ADVERSARIAL_QUALIFICATION_INTERFACE
    schema: ClassVar[str] = ADVERSARIAL_QUALIFICATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "route",
            "status",
            "case_ids",
            "reasons",
            "self_certified",
            "can_authorize_completion",
            "qualification_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _qualification_status(self.status))
        object.__setattr__(
            self,
            "case_ids",
            _unique_ordered_text(self.case_ids, "case_ids", limit=MAX_COUNTEREXAMPLES),
        )
        object.__setattr__(
            self,
            "reasons",
            _unique_ordered_text(self.reasons, "reasons", limit=MAX_COUNTEREXAMPLES),
        )
        if self.status == QualificationStatus.MISSING.value and self.case_ids:
            raise ProcedureAdapterError("missing adversarial qualification has no cases")
        if self.status == QualificationStatus.PASSED.value and not self.case_ids:
            raise ProcedureAdapterError("passed adversarial qualification requires cases")

    @property
    def route(self) -> str:
        return QualificationRoute.ADVERSARIAL.value

    @property
    def self_certified(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ADVERSARIAL_QUALIFICATION_SCHEMA,
            "interface": ADVERSARIAL_QUALIFICATION_INTERFACE,
            "route": self.route,
            "status": self.status,
            "case_ids": list(self.case_ids),
            "reasons": list(self.reasons),
            "self_certified": False,
            "can_authorize_completion": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def qualification_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["qualification_cid"] = self.qualification_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdversarialQualification":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("qualification_cid")
        if payload.pop("schema") != ADVERSARIAL_QUALIFICATION_SCHEMA:
            raise ProcedureAdapterError("unsupported AdversarialQualification schema")
        if payload.pop("interface") != ADVERSARIAL_QUALIFICATION_INTERFACE:
            raise ProcedureAdapterError(
                "unsupported AdversarialQualification interface"
            )
        if payload.pop("route") != QualificationRoute.ADVERSARIAL.value:
            raise ProcedureAdapterError("adversarial route must remain adversarial")
        if payload.pop("self_certified") is not False:
            raise ProcedureAdapterError("procedure cannot self-certify")
        _pop_authority_flags(payload, cls.__name__)
        result = cls(**payload)
        _verify_cid(claimed, result.qualification_cid, "qualification_cid")
        return result


@dataclass(frozen=True, slots=True)
class ProcedurePromotionNomination:
    """Nomination toward the existing procedure authority. Never a promotion."""

    plan_cid: str
    contract_cid: str
    qualification_routes: Sequence[str]
    nominated: bool
    procedure_authority: str = PROCEDURE_AUTHORITY
    authority_response_cid: str = ""

    interface: ClassVar[str] = PROCEDURE_PROMOTION_NOMINATION_INTERFACE
    schema: ClassVar[str] = PROCEDURE_PROMOTION_NOMINATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "plan_cid",
            "contract_cid",
            "qualification_routes",
            "nominated",
            "promoted",
            "self_certified",
            "self_promoted",
            "procedure_authority",
            "procedure_authority_interface",
            "authority_response_cid",
            "adapter_is_nomination_only",
            "can_authorize_completion",
            "can_create_procedure_authority",
            "nomination_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_cid", _cid(self.plan_cid, "plan_cid"))
        object.__setattr__(self, "contract_cid", _cid(self.contract_cid, "contract_cid"))
        routes = _unique_ordered_text(
            self.qualification_routes, "qualification_routes", limit=8
        )
        unknown = set(routes) - DECLARED_QUALIFICATION_ROUTES
        if unknown:
            raise ProcedureAdapterError(
                f"unknown qualification routes: {sorted(unknown)}"
            )
        object.__setattr__(self, "qualification_routes", routes)
        object.__setattr__(self, "nominated", _bool(self.nominated, "nominated"))
        if self.nominated and not routes:
            raise ProcedureAdapterError(
                "promotion nomination requires held-out or adversarial qualification"
            )
        authority = _text(self.procedure_authority, "procedure_authority")
        if authority != PROCEDURE_AUTHORITY:
            raise ProcedureAdapterError(
                "procedure authority must remain ProofCarryingProcedureCompiler"
            )
        object.__setattr__(self, "procedure_authority", PROCEDURE_AUTHORITY)
        object.__setattr__(
            self,
            "authority_response_cid",
            _optional_cid(self.authority_response_cid, "authority_response_cid"),
        )

    @property
    def promoted(self) -> bool:
        return False

    @property
    def self_certified(self) -> bool:
        return False

    @property
    def self_promoted(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_procedure_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PROCEDURE_PROMOTION_NOMINATION_SCHEMA,
            "interface": PROCEDURE_PROMOTION_NOMINATION_INTERFACE,
            "plan_cid": self.plan_cid,
            "contract_cid": self.contract_cid,
            "qualification_routes": list(self.qualification_routes),
            "nominated": self.nominated,
            "promoted": False,
            "self_certified": False,
            "self_promoted": False,
            "procedure_authority": PROCEDURE_AUTHORITY,
            "procedure_authority_interface": PROCEDURE_AUTHORITY_INTERFACE,
            "authority_response_cid": self.authority_response_cid,
            "adapter_is_nomination_only": True,
            "can_authorize_completion": False,
            "can_create_procedure_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def nomination_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["nomination_cid"] = self.nomination_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProcedurePromotionNomination":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("nomination_cid")
        if payload.pop("schema") != PROCEDURE_PROMOTION_NOMINATION_SCHEMA:
            raise ProcedureAdapterError(
                "unsupported ProcedurePromotionNomination schema"
            )
        if payload.pop("interface") != PROCEDURE_PROMOTION_NOMINATION_INTERFACE:
            raise ProcedureAdapterError(
                "unsupported ProcedurePromotionNomination interface"
            )
        if payload.pop("promoted") is not False:
            raise ProcedureAdapterError("adapter cannot promote")
        if payload.pop("self_certified") is not False:
            raise ProcedureAdapterError("procedure cannot self-certify")
        if payload.pop("self_promoted") is not False:
            raise ProcedureAdapterError("procedure cannot self-promote")
        if payload.pop("procedure_authority_interface") != PROCEDURE_AUTHORITY_INTERFACE:
            raise ProcedureAdapterError("procedure authority interface drifted")
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ProcedureAdapterError("adapter must remain nomination_only")
        _pop_authority_flags(payload, cls.__name__)
        result = cls(**payload)
        _verify_cid(claimed, result.nomination_cid, "nomination_cid")
        return result


@dataclass(frozen=True, slots=True)
class RefactorProcedureCompilationReceipt:
    """Content-addressed SPAR-034 compilation receipt. Nomination-only."""

    tree_id: str
    status: str
    trajectory_cids: Sequence[str]
    plan_cid: str
    contract_cid: str
    hole_cids: Sequence[str]
    held_out_status: str
    adversarial_status: str
    qualification_routes: Sequence[str]
    nomination_cid: str
    negative_transition_cids: Sequence[str] = ()
    typed_terminal: bool = False
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE
    schema: ClassVar[str] = REFACTOR_PROCEDURE_COMPILATION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "status",
            "trajectory_cids",
            "plan_cid",
            "contract_cid",
            "hole_cids",
            "held_out_status",
            "adversarial_status",
            "qualification_routes",
            "nomination_cid",
            "negative_transition_cids",
            "typed_terminal",
            "analyzer_id",
            "adapter_is_nomination_only",
            "procedure_authority",
            "promoted",
            "self_certified",
            "candidates_remain_unpromoted",
            "holes_preserved",
            "mutated",
            "deterministic",
            "network",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_create_procedure_authority",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "status", _status_value(self.status))
        object.__setattr__(
            self,
            "trajectory_cids",
            _cids(self.trajectory_cids, "trajectory_cids", required=False),
        )
        object.__setattr__(self, "plan_cid", _optional_cid(self.plan_cid, "plan_cid"))
        object.__setattr__(
            self, "contract_cid", _optional_cid(self.contract_cid, "contract_cid")
        )
        object.__setattr__(
            self, "hole_cids", _cids(self.hole_cids, "hole_cids", required=False)
        )
        object.__setattr__(
            self, "held_out_status", _qualification_status(self.held_out_status)
        )
        object.__setattr__(
            self,
            "adversarial_status",
            _qualification_status(self.adversarial_status),
        )
        routes = _unique_ordered_text(
            self.qualification_routes, "qualification_routes", limit=8
        )
        unknown = set(routes) - DECLARED_QUALIFICATION_ROUTES
        if unknown:
            raise ProcedureAdapterError(
                f"unknown qualification routes: {sorted(unknown)}"
            )
        object.__setattr__(self, "qualification_routes", routes)
        object.__setattr__(
            self, "nomination_cid", _optional_cid(self.nomination_cid, "nomination_cid")
        )
        object.__setattr__(
            self,
            "negative_transition_cids",
            _cids(
                self.negative_transition_cids,
                "negative_transition_cids",
                required=False,
            ),
        )
        object.__setattr__(
            self, "typed_terminal", _bool(self.typed_terminal, "typed_terminal")
        )
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise ProcedureAdapterError("receipt analyzer_id must remain SPAR-034")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if (
            self.status == CompilationStatus.NOMINATED.value
            and not self.qualification_routes
        ):
            raise ProcedureAdapterError(
                "nominated procedures require held-out or adversarial qualification"
            )

    @property
    def mutated(self) -> bool:
        return False

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def promoted(self) -> bool:
        return False

    @property
    def self_certified(self) -> bool:
        return False

    @property
    def candidates_remain_unpromoted(self) -> bool:
        return True

    @property
    def holes_preserved(self) -> bool:
        return True

    @property
    def adapter_is_nomination_only(self) -> bool:
        return True

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def can_create_procedure_authority(self) -> bool:
        return False

    @property
    def network(self) -> str:
        return NETWORK_DENY

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_PROCEDURE_COMPILATION_RECEIPT_SCHEMA,
            "interface": REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "status": self.status,
            "trajectory_cids": list(self.trajectory_cids),
            "plan_cid": self.plan_cid,
            "contract_cid": self.contract_cid,
            "hole_cids": list(self.hole_cids),
            "held_out_status": self.held_out_status,
            "adversarial_status": self.adversarial_status,
            "qualification_routes": list(self.qualification_routes),
            "nomination_cid": self.nomination_cid,
            "negative_transition_cids": list(self.negative_transition_cids),
            "typed_terminal": self.typed_terminal,
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "procedure_authority": PROCEDURE_AUTHORITY,
            "promoted": False,
            "self_certified": False,
            "candidates_remain_unpromoted": True,
            "holes_preserved": True,
            "mutated": False,
            "deterministic": True,
            "network": NETWORK_DENY,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_create_procedure_authority": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorProcedureCompilationReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != REFACTOR_PROCEDURE_COMPILATION_RECEIPT_SCHEMA:
            raise ProcedureAdapterError(
                "unsupported RefactorProcedureCompilationReceipt schema"
            )
        if payload.pop("interface") != REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE:
            raise ProcedureAdapterError(
                "unsupported RefactorProcedureCompilationReceipt interface"
            )
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ProcedureAdapterError("adapter must remain nomination_only")
        if payload.pop("procedure_authority") != PROCEDURE_AUTHORITY:
            raise ProcedureAdapterError(
                "procedure authority must remain ProofCarryingProcedureCompiler"
            )
        if payload.pop("promoted") is not False:
            raise ProcedureAdapterError("adapter cannot promote")
        if payload.pop("self_certified") is not False:
            raise ProcedureAdapterError("procedure cannot self-certify")
        if payload.pop("candidates_remain_unpromoted") is not True:
            raise ProcedureAdapterError("candidates must remain unpromoted")
        if payload.pop("holes_preserved") is not True:
            raise ProcedureAdapterError("holes must be preserved")
        if payload.pop("mutated") is not False:
            raise ProcedureAdapterError("receipt must remain non-mutating")
        if payload.pop("deterministic") is not True:
            raise ProcedureAdapterError("receipt must remain deterministic")
        if payload.pop("network") != NETWORK_DENY:
            raise ProcedureAdapterError("network must remain deny")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise ProcedureAdapterError("receipt analyzer_id must remain SPAR-034")
        _pop_authority_flags(payload, cls.__name__)
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


def _compile_wave(wave: Mapping[str, Any] | Any) -> dict[str, Any]:
    if wave is None:
        raise ProcedureAdapterError("SPAR-025 wave is required")
    payload = _as_mapping(wave, "SPAR-025 wave")
    _reject_non_admitting(payload, "SPAR-025 wave")
    if payload.get("writes_repository") is True:
        raise ProcedureAdapterError("SPAR-025 wave cannot write the repository")
    if payload.get("executor_is_nomination_only") is False:
        raise ProcedureAdapterError("SPAR-025 executor must remain nomination_only")
    receipt_cid = payload.get("receipt_cid") or payload.get("wave_cid")
    if receipt_cid in (None, ""):
        raise ProcedureAdapterError("SPAR-025 receipt_cid is required")
    packet_cids = payload.get("packet_cids")
    if packet_cids in (None, ()):
        raise ProcedureAdapterError("SPAR-025 packet_cids are required")
    write_paths = payload.get("write_paths")
    if write_paths in (None, ()):
        raise ProcedureAdapterError("SPAR-025 write_paths are required")
    return {
        "tree_id": _tree_id(payload.get("tree_id")),
        "receipt_cid": _cid(receipt_cid, "SPAR-025 receipt_cid"),
        "packet_cids": _cids(packet_cids, "SPAR-025 packet_cids", required=True),
        "write_paths": _exact_paths(write_paths, "SPAR-025 write_paths"),
        "status": _text(payload.get("status", "applied"), "SPAR-025 status"),
    }


def _transition_tree_id(payload: Mapping[str, Any]) -> str:
    reuse_key = payload.get("reuse_key")
    if isinstance(reuse_key, Mapping):
        tree = reuse_key.get("tree_id")
        if tree not in (None, ""):
            return _tree_id(tree)
    tree = payload.get("tree_id")
    if tree in (None, ""):
        raise ProcedureAdapterError("SPAR-033 tree_id is required")
    return _tree_id(tree)


def _compile_transition(transition: Mapping[str, Any] | Any) -> dict[str, Any]:
    if transition is None:
        raise ProcedureAdapterError("SPAR-033 transition is required")
    payload = _as_mapping(transition, "SPAR-033 transition")
    _reject_non_admitting(payload, "SPAR-033 transition")
    outcome = _outcome_value(payload.get("outcome", TrajectoryOutcome.ACCEPTED.value))
    evidence = _evidence_class(payload.get("evidence_class", "transition"))
    if evidence in _NON_ADMITTING_EVIDENCE and outcome == TrajectoryOutcome.ACCEPTED.value:
        raise ProcedureAdapterError(
            "vector, model, or heuristic evidence cannot admit a transition"
        )
    raw_source_cids = payload.get("raw_source_cids")
    if raw_source_cids in (None, (), []):
        raise ProcedureAdapterError("raw source is required")
    write_paths = payload.get("write_paths")
    if write_paths in (None, ()):
        raise ProcedureAdapterError(
            "SPAR-033 write_paths must not be empty; unrestricted scope is rejected"
        )
    commands = payload.get("validation_commands")
    if commands in (None, ()):
        raise ProcedureAdapterError("SPAR-033 validation_commands are required")
    wave_receipt_cid = payload.get("wave_receipt_cid")
    if wave_receipt_cid in (None, ""):
        raise ProcedureAdapterError("SPAR-033 wave_receipt_cid is required")
    packet_cid = payload.get("packet_cid")
    if packet_cid in (None, ""):
        raise ProcedureAdapterError("SPAR-033 packet_cid is required")
    preconditions = payload.get("preconditions") or list(DEFAULT_PRECONDITIONS)
    effects = payload.get("effects") or list(DEFAULT_EFFECTS)
    rollback = payload.get("rollback") or DEFAULT_ROLLBACK
    holes = payload.get("holes") or []
    transition_cid = payload.get("transition_cid")
    if transition_cid in (None, ""):
        transition_cid = cid_for_dag_json(
            {
                "wave_receipt_cid": wave_receipt_cid,
                "packet_cid": packet_cid,
                "write_paths": list(write_paths),
                "outcome": outcome,
            }
        )
    return {
        "tree_id": _transition_tree_id(payload),
        "outcome": outcome,
        "evidence_class": evidence,
        "wave_receipt_cid": _cid(wave_receipt_cid, "SPAR-033 wave_receipt_cid"),
        "packet_cid": _cid(packet_cid, "SPAR-033 packet_cid"),
        "raw_source_cids": _cids(raw_source_cids, "raw_source_cids", required=True),
        "write_paths": _exact_paths(write_paths, "SPAR-033 write_paths"),
        "validation_commands": _commands(commands, "SPAR-033 validation_commands"),
        "preconditions": _unique_ordered_text(
            preconditions, "preconditions", limit=MAX_PRECONDITIONS
        ),
        "effects": _unique_ordered_text(effects, "effects", limit=MAX_EFFECTS),
        "rollback": _text(rollback, "rollback"),
        "holes": _unique_ordered_text(holes, "holes", limit=MAX_HOLES),
        "transition_cid": _cid(transition_cid, "SPAR-033 transition_cid"),
    }


def _bind_transition_to_wave(
    transition: Mapping[str, Any],
    waves: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    wave = waves.get(transition["wave_receipt_cid"])
    if wave is None:
        raise ProcedureAdapterError("SPAR-033 wave_receipt_cid must match SPAR-025")
    if transition["tree_id"] != wave["tree_id"]:
        raise ProcedureAdapterError("SPAR-033 tree_id must match SPAR-025")
    if transition["packet_cid"] not in wave["packet_cids"]:
        raise ProcedureAdapterError(
            "SPAR-025 wave packet_cids must include SPAR-033 packet_cid"
        )
    if tuple(transition["write_paths"]) != tuple(wave["write_paths"]):
        raise ProcedureAdapterError("SPAR-025/033 write_paths must match")
    return wave


def _bind_optional_receipt(
    receipt: Mapping[str, Any] | Any | None,
    *,
    name: str,
    tree_id: str,
) -> None:
    if receipt is None:
        return
    payload = _as_mapping(receipt, name)
    _reject_non_admitting(payload, name)
    receipt_tree = payload.get("tree_id")
    if receipt_tree not in (None, "") and _tree_id(receipt_tree) != tree_id:
        raise ProcedureAdapterError(f"{name} tree_id must match SPAR-025")
    if payload.get("claims_general_equivalence") is True:
        raise ProcedureAdapterError("general Python equivalence is not claimed")
    if payload.get("candidates_remain_unpromoted") is False:
        raise ProcedureAdapterError("SPAR-032 candidates must remain unpromoted")
    if payload.get("can_authorize_completion") is True:
        raise ProcedureAdapterError(f"{name} cannot authorize completion")
    if payload.get("can_create_proof_authority") is True:
        raise ProcedureAdapterError(f"{name} cannot create proof authority")


def _normalize_transition(
    transition: Mapping[str, Any],
) -> NormalizedRefactorTrajectory:
    holes = list(transition["holes"])
    for hole in holes:
        _hole_type(hole, "holes")
    return compile_normalized_trajectory(
        tree_id=transition["tree_id"],
        wave_receipt_cid=transition["wave_receipt_cid"],
        packet_cid=transition["packet_cid"],
        transition_cid=transition["transition_cid"],
        write_paths=list(transition["write_paths"]),
        validation_commands=list(transition["validation_commands"]),
        raw_source_cids=list(transition["raw_source_cids"]),
        preconditions=list(transition["preconditions"]),
        effects=list(transition["effects"]),
        rollback=transition["rollback"],
        outcome=transition["outcome"],
        evidence_class=transition["evidence_class"],
        holes=holes,
    )


def anti_unify_refactor_plans(
    trajectories: Sequence[NormalizedRefactorTrajectory],
) -> AntiUnifiedRefactorPlan:
    """Anti-unify accepted trajectories. Paths never become parameters."""

    accepted = [
        item
        for item in trajectories
        if item.outcome == TrajectoryOutcome.ACCEPTED.value
    ]
    if len(accepted) < 2:
        raise ProcedureAdapterError("anti-unification requires at least two trajectories")
    tree_id = accepted[0].tree_id
    for item in accepted:
        if item.tree_id != tree_id:
            raise ProcedureAdapterError("anti-unification tree_id mismatch")
    path_sets = [set(item.write_paths) for item in accepted]
    shared_paths = set.intersection(*path_sets)
    if not shared_paths:
        counterexample = accepted[1].trajectory_cid
        return AntiUnifiedRefactorPlan(
            tree_id=tree_id,
            trajectory_cids=[item.trajectory_cid for item in accepted],
            shared_write_paths=list(accepted[0].write_paths),
            shared_validation_commands=list(accepted[0].validation_commands),
            shared_preconditions=list(accepted[0].preconditions),
            shared_effects=list(accepted[0].effects),
            shared_rollback=accepted[0].rollback,
            holes=(),
            lost_details=(),
            counterexample_cids=[counterexample],
            anti_unified=False,
        )
    ordered_shared_paths = tuple(
        path for path in accepted[0].write_paths if path in shared_paths
    )
    lost_details: list[str] = []
    for item in accepted:
        for path in item.write_paths:
            if path not in shared_paths:
                lost_details.append(f"write_path:{path}")
        lost_details.append(f"packet_cid:{item.packet_cid}")
    command_union: list[str] = []
    seen_commands: set[str] = set()
    for item in accepted:
        for command in item.validation_commands:
            if command not in seen_commands:
                seen_commands.add(command)
                command_union.append(command)
    precondition_sets = [set(item.preconditions) for item in accepted]
    shared_preconditions = tuple(
        item
        for item in accepted[0].preconditions
        if item in set.intersection(*precondition_sets)
    )
    if not shared_preconditions:
        raise ProcedureAdapterError("anti-unification dropped all preconditions")
    effect_union: list[str] = []
    seen_effects: set[str] = set()
    for item in accepted:
        for effect in item.effects:
            if effect not in seen_effects:
                seen_effects.add(effect)
                effect_union.append(effect)
    rollbacks = {item.rollback for item in accepted}
    holes: list[PreservedHole] = []
    if len(rollbacks) != 1:
        return AntiUnifiedRefactorPlan(
            tree_id=tree_id,
            trajectory_cids=[item.trajectory_cid for item in accepted],
            shared_write_paths=list(ordered_shared_paths),
            shared_validation_commands=command_union,
            shared_preconditions=list(shared_preconditions),
            shared_effects=effect_union,
            shared_rollback=accepted[0].rollback,
            holes=(),
            lost_details=lost_details,
            counterexample_cids=[accepted[1].trajectory_cid],
            anti_unified=False,
        )
    packet_cids = {item.packet_cid for item in accepted}
    if len(packet_cids) > 1:
        holes.append(
            compile_preserved_hole(
                hole_id="hole:packet",
                hole_type="SELECT_ONE_OF_ALLOWED_SYMBOLS",
                origin="packet_cid",
                required=True,
            )
        )
    declared_holes = []
    for item in accepted:
        declared_holes.extend(item.holes)
    for index, hole_type in enumerate(dict.fromkeys(declared_holes)):
        holes.append(
            compile_preserved_hole(
                hole_id=f"hole:declared:{index}",
                hole_type=hole_type,
                origin="trajectory",
                required=True,
            )
        )
    return AntiUnifiedRefactorPlan(
        tree_id=tree_id,
        trajectory_cids=[item.trajectory_cid for item in accepted],
        shared_write_paths=list(ordered_shared_paths),
        shared_validation_commands=command_union,
        shared_preconditions=list(shared_preconditions),
        shared_effects=effect_union,
        shared_rollback=next(iter(rollbacks)),
        holes=tuple(holes),
        lost_details=lost_details,
        counterexample_cids=(),
        anti_unified=True,
    )


def infer_procedure_contract(
    plan: AntiUnifiedRefactorPlan,
) -> InferredProcedureContract:
    if not plan.anti_unified:
        raise ProcedureAdapterError("cannot infer a contract from a failed anti-unification")
    postconditions = tuple(plan.shared_validation_commands)
    return InferredProcedureContract(
        plan_cid=plan.plan_cid,
        preconditions=list(plan.shared_preconditions),
        effects=list(plan.shared_effects),
        rollback=plan.shared_rollback,
        postconditions=list(postconditions),
        invariants=("holes_preserved", "validation_never_dropped"),
    )


def preserve_holes(plan: AntiUnifiedRefactorPlan) -> tuple[PreservedHole, ...]:
    return tuple(plan.holes)


def _held_out_trajectories(
    held_out: Sequence[Mapping[str, Any] | Any],
    waves: Mapping[str, Mapping[str, Any]],
    tree_id: str,
) -> tuple[NormalizedRefactorTrajectory, ...]:
    compiled: list[NormalizedRefactorTrajectory] = []
    for item in _mapping_sequence(held_out, "held_out"):
        transition = _compile_transition(item)
        if transition["tree_id"] != tree_id:
            raise ProcedureAdapterError("held-out tree_id must match SPAR-025")
        _bind_transition_to_wave(transition, waves)
        if transition["outcome"] != TrajectoryOutcome.ACCEPTED.value:
            raise ProcedureAdapterError("held-out trajectories must be accepted")
        compiled.append(_normalize_transition(transition))
    return tuple(compiled)


def qualify_held_out(
    plan: AntiUnifiedRefactorPlan,
    training: Sequence[NormalizedRefactorTrajectory],
    held_out: Sequence[NormalizedRefactorTrajectory],
) -> HeldOutQualification:
    if not held_out:
        return HeldOutQualification(status=QualificationStatus.MISSING.value)
    training_ids = {item.trajectory_cid for item in training}
    training_waves = {item.wave_receipt_cid for item in training}
    training_transitions = {item.transition_cid for item in training}
    reasons: list[str] = []
    qualified: list[str] = []
    shared_paths = set(plan.shared_write_paths)
    allowed_commands = set(plan.shared_validation_commands)
    for item in held_out:
        if item.trajectory_cid in training_ids:
            reasons.append("held_out_not_disjoint")
            continue
        if item.wave_receipt_cid in training_waves:
            reasons.append("held_out_wave_not_disjoint")
            continue
        if item.transition_cid in training_transitions:
            reasons.append("held_out_transition_not_disjoint")
            continue
        if item.tree_id != plan.tree_id:
            reasons.append("held_out_tree_mismatch")
            continue
        if set(item.write_paths).isdisjoint(shared_paths):
            reasons.append("held_out_write_paths_disjoint")
            continue
        if not set(item.validation_commands).issubset(allowed_commands):
            reasons.append("held_out_dropped_validation")
            continue
        if item.rollback != plan.shared_rollback:
            reasons.append("held_out_rollback_mismatch")
            continue
        qualified.append(item.trajectory_cid)
    if not qualified:
        return HeldOutQualification(
            status=QualificationStatus.FAILED.value,
            held_out_trajectory_cids=[item.trajectory_cid for item in held_out],
            reasons=reasons or ["held_out_failed"],
        )
    return HeldOutQualification(
        status=QualificationStatus.PASSED.value,
        held_out_trajectory_cids=qualified,
        reasons=(),
    )


def qualify_adversarial(
    plan: AntiUnifiedRefactorPlan,
    adversarial: Sequence[Mapping[str, Any] | Any],
) -> AdversarialQualification:
    cases = _mapping_sequence(adversarial, "adversarial")
    if not cases:
        return AdversarialQualification(status=QualificationStatus.MISSING.value)
    reasons: list[str] = []
    passed_ids: list[str] = []
    for case in cases:
        _reject_non_admitting(case, "adversarial")
        case_id = _text(case.get("case_id", ""), "case_id")
        polarity = case.get("polarity", AdversarialPolarity.MUST_SURVIVE.value)
        if isinstance(polarity, AdversarialPolarity):
            polarity = polarity.value
        polarity = _text(polarity, "polarity")
        source = _text(
            case.get("source_authority", "specification"), "source_authority"
        )
        if source not in ADVERSARIAL_SOURCE_AUTHORITIES:
            reasons.append("adversarial_source_cannot_admit")
            continue
        if source == "replayed_counterexample" and case.get("replayed") is False:
            reasons.append("raw countermodels cannot refute until replay")
            continue
        tree = case.get("tree_id")
        if tree not in (None, "") and _tree_id(tree) != plan.tree_id:
            reasons.append("adversarial_tree_mismatch")
            continue
        if polarity == AdversarialPolarity.MUST_FAIL_CLOSED.value:
            claim = _text(case.get("claim", "self_certify"), "claim")
            if claim not in FORBIDDEN_PROCEDURE_NAMES:
                reasons.append("adversarial_unknown_fail_closed_claim")
                continue
            passed_ids.append(case_id)
            continue
        if polarity != AdversarialPolarity.MUST_SURVIVE.value:
            reasons.append("unknown adversarial polarity")
            continue
        conflict = case.get("conflicts_with", "")
        if conflict == "rollback" and case.get("rollback") not in (None, "", plan.shared_rollback):
            reasons.append("adversarial_rollback_conflict")
            continue
        if conflict in FORBIDDEN_HOLE_TYPES:
            reasons.append("adversarial_forbidden_hole")
            continue
        passed_ids.append(case_id)
    if len(passed_ids) != len(cases):
        return AdversarialQualification(
            status=QualificationStatus.FAILED.value,
            case_ids=[_text(case.get("case_id", ""), "case_id") for case in cases],
            reasons=reasons or ["adversarial_failed"],
        )
    return AdversarialQualification(
        status=QualificationStatus.PASSED.value,
        case_ids=passed_ids,
        reasons=(),
    )


def nominate_procedure_promotion(
    plan: AntiUnifiedRefactorPlan,
    contract: InferredProcedureContract,
    routes: Sequence[str],
    *,
    promote: PromoteFn | None = None,
) -> ProcedurePromotionNomination:
    if not plan.anti_unified:
        raise ProcedureAdapterError("cannot nominate a failed anti-unification")
    if not routes:
        raise ProcedureAdapterError(
            "promotion nomination requires held-out or adversarial qualification"
        )
    response_cid = ""
    if promote is not None:
        qualname = getattr(promote, "__qualname__", "")
        producer = getattr(promote, "__self__", None)
        producer_name = type(producer).__name__ if producer is not None else ""
        if (
            "ProofCarryingProcedureRefactorAdapter" in qualname
            or producer_name == "ProofCarryingProcedureRefactorAdapter"
        ):
            raise ProcedureAdapterError("procedure cannot self-certify")
        response = promote(
            nomination={
                "plan_cid": plan.plan_cid,
                "contract_cid": contract.contract_cid,
                "qualification_routes": list(routes),
                "procedure_authority": PROCEDURE_AUTHORITY,
            },
            network=NETWORK_DENY,
            self_certified=False,
            can_authorize_completion=False,
        )
        if not isinstance(response, Mapping):
            raise ProcedureAdapterError("procedure authority response must be a mapping")
        _reject_forbidden_bodies(response, "procedure authority response")
        if response.get("self_certified") is True or response.get("self_promoted") is True:
            raise ProcedureAdapterError("procedure cannot self-certify")
        if response.get("can_authorize_completion") is True:
            raise ProcedureAdapterError("procedure authority cannot grant completion")
        producer_id = str(response.get("producer") or response.get("producer_id") or "")
        if producer_id in _SELF_PRODUCERS:
            raise ProcedureAdapterError("procedure cannot self-certify")
        response_cid = _optional_cid(
            response.get("authority_response_cid") or response.get("receipt_cid") or "",
            "authority_response_cid",
        )
    return ProcedurePromotionNomination(
        plan_cid=plan.plan_cid,
        contract_cid=contract.contract_cid,
        qualification_routes=list(routes),
        nominated=True,
        authority_response_cid=response_cid,
    )


def _empty_receipt(
    *,
    tree_id: str,
    status: str,
    trajectories: Sequence[NormalizedRefactorTrajectory] = (),
    negatives: Sequence[str] = (),
    held_out: HeldOutQualification | None = None,
    adversarial: AdversarialQualification | None = None,
    plan: AntiUnifiedRefactorPlan | None = None,
    contract: InferredProcedureContract | None = None,
    nomination: ProcedurePromotionNomination | None = None,
    typed_terminal: bool = True,
) -> RefactorProcedureCompilationReceipt:
    held = held_out or HeldOutQualification(status=QualificationStatus.MISSING.value)
    adv = adversarial or AdversarialQualification(
        status=QualificationStatus.MISSING.value
    )
    routes: list[str] = []
    if held.status != QualificationStatus.MISSING.value:
        routes.append(QualificationRoute.HELD_OUT.value)
    if adv.status != QualificationStatus.MISSING.value:
        routes.append(QualificationRoute.ADVERSARIAL.value)
    return RefactorProcedureCompilationReceipt(
        tree_id=tree_id,
        status=status,
        trajectory_cids=[item.trajectory_cid for item in trajectories],
        plan_cid=plan.plan_cid if plan is not None else "",
        contract_cid=contract.contract_cid if contract is not None else "",
        hole_cids=[hole.hole_cid for hole in (plan.holes if plan is not None else ())],
        held_out_status=held.status,
        adversarial_status=adv.status,
        qualification_routes=routes if status == CompilationStatus.NOMINATED.value else (),
        nomination_cid=nomination.nomination_cid if nomination is not None else "",
        negative_transition_cids=list(negatives),
        typed_terminal=typed_terminal,
    )


def compile_refactor_procedure(
    *,
    waves: Sequence[Mapping[str, Any] | Any],
    transitions: Sequence[Mapping[str, Any] | Any],
    held_out: Sequence[Mapping[str, Any] | Any] = (),
    adversarial: Sequence[Mapping[str, Any] | Any] = (),
    synthesis_receipt: Mapping[str, Any] | Any | None = None,
    normalization_receipt: Mapping[str, Any] | Any | None = None,
    promote: PromoteFn | None = None,
    mutate: bool = False,
    network: str = NETWORK_DENY,
    vector_evidence: Any = None,
) -> RefactorProcedureCompilationReceipt:
    """Compile accepted refactor waves into a procedure nomination."""

    if mutate is True:
        raise ProcedureAdapterError("cannot mutate")
    _network_value(network)
    if vector_evidence not in (None, (), {}, []):
        payload = (
            dict(vector_evidence)
            if isinstance(vector_evidence, Mapping)
            else {"evidence_class": "vector_candidate"}
        )
        _reject_non_admitting(payload, "procedure")
        raise ProcedureAdapterError(
            "vector, model, or heuristic evidence cannot admit a procedure"
        )
    wave_items = _mapping_sequence(waves, "SPAR-025 waves")
    if not wave_items:
        raise ProcedureAdapterError("SPAR-025 wave is required")
    transition_items = _mapping_sequence(transitions, "SPAR-033 transitions")
    if not transition_items:
        raise ProcedureAdapterError("SPAR-033 transition is required")
    compiled_waves = [_compile_wave(item) for item in wave_items]
    tree_id = compiled_waves[0]["tree_id"]
    for wave in compiled_waves:
        if wave["tree_id"] != tree_id:
            raise ProcedureAdapterError("SPAR-025 tree_id mismatch")
    waves_by_cid = {wave["receipt_cid"]: wave for wave in compiled_waves}
    compiled_transitions = [_compile_transition(item) for item in transition_items]
    if len(compiled_transitions) > MAX_TRAJECTORIES:
        raise ProcedureAdapterError("trajectories exceed maximum length")
    accepted: list[NormalizedRefactorTrajectory] = []
    negatives: list[str] = []
    for transition in compiled_transitions:
        _bind_transition_to_wave(transition, waves_by_cid)
        normalized = _normalize_transition(transition)
        if normalized.outcome == TrajectoryOutcome.REJECTED.value:
            negatives.append(normalized.transition_cid)
            continue
        accepted.append(normalized)
    _bind_optional_receipt(synthesis_receipt, name="SPAR-032", tree_id=tree_id)
    _bind_optional_receipt(normalization_receipt, name="SPAR-031", tree_id=tree_id)
    if len(accepted) < 2:
        return _empty_receipt(
            tree_id=tree_id,
            status=CompilationStatus.UNKNOWN.value,
            trajectories=accepted,
            negatives=negatives,
            typed_terminal=True,
        )
    plan = anti_unify_refactor_plans(accepted)
    if not plan.anti_unified:
        return _empty_receipt(
            tree_id=tree_id,
            status=CompilationStatus.REJECTED.value,
            trajectories=accepted,
            negatives=negatives,
            plan=plan,
            typed_terminal=True,
        )
    contract = infer_procedure_contract(plan)
    held_out_normalized = _held_out_trajectories(held_out, waves_by_cid, tree_id)
    held = qualify_held_out(plan, accepted, held_out_normalized)
    adv = qualify_adversarial(plan, adversarial)
    routes: list[str] = []
    if held.status != QualificationStatus.MISSING.value:
        routes.append(QualificationRoute.HELD_OUT.value)
    if adv.status != QualificationStatus.MISSING.value:
        routes.append(QualificationRoute.ADVERSARIAL.value)
    if not routes:
        return _empty_receipt(
            tree_id=tree_id,
            status=CompilationStatus.INCOMPLETE.value,
            trajectories=accepted,
            negatives=negatives,
            plan=plan,
            contract=contract,
            held_out=held,
            adversarial=adv,
            typed_terminal=True,
        )
    if (
        held.status == QualificationStatus.FAILED.value
        or adv.status == QualificationStatus.FAILED.value
    ):
        return _empty_receipt(
            tree_id=tree_id,
            status=CompilationStatus.REJECTED.value,
            trajectories=accepted,
            negatives=negatives,
            plan=plan,
            contract=contract,
            held_out=held,
            adversarial=adv,
            typed_terminal=True,
        )
    nomination = nominate_procedure_promotion(
        plan, contract, routes, promote=promote
    )
    return RefactorProcedureCompilationReceipt(
        tree_id=tree_id,
        status=CompilationStatus.NOMINATED.value,
        trajectory_cids=[item.trajectory_cid for item in accepted],
        plan_cid=plan.plan_cid,
        contract_cid=contract.contract_cid,
        hole_cids=[hole.hole_cid for hole in plan.holes],
        held_out_status=held.status,
        adversarial_status=adv.status,
        qualification_routes=routes,
        nomination_cid=nomination.nomination_cid,
        negative_transition_cids=negatives,
        typed_terminal=False,
    )


def dry_run_refactor_procedure(
    *,
    waves: Sequence[Mapping[str, Any] | Any],
    transitions: Sequence[Mapping[str, Any] | Any],
    held_out: Sequence[Mapping[str, Any] | Any] = (),
    adversarial: Sequence[Mapping[str, Any] | Any] = (),
    synthesis_receipt: Mapping[str, Any] | Any | None = None,
    normalization_receipt: Mapping[str, Any] | Any | None = None,
    promote: PromoteFn | None = None,
    network: str = NETWORK_DENY,
    vector_evidence: Any = None,
) -> RefactorProcedureCompilationReceipt:
    """Deterministic dry-run. Never mutates."""

    result = compile_refactor_procedure(
        waves=waves,
        transitions=transitions,
        held_out=held_out,
        adversarial=adversarial,
        synthesis_receipt=synthesis_receipt,
        normalization_receipt=normalization_receipt,
        promote=promote,
        mutate=False,
        network=network,
        vector_evidence=vector_evidence,
    )
    if result.mutated is not False or result.deterministic is not True:
        raise ProcedureAdapterError("dry-run must remain deterministic and non-mutating")
    return result


class ProofCarryingProcedureRefactorAdapter:
    """SPAR-034 ProofCarryingProcedure refactor adapter. Nomination-only."""

    interface: ClassVar[str] = PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_INTERFACE
    schema: ClassVar[str] = PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID
    receipt_interface: ClassVar[str] = REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE
    procedure_authority: ClassVar[str] = PROCEDURE_AUTHORITY

    def compile(
        self,
        *,
        waves: Sequence[Mapping[str, Any] | Any],
        transitions: Sequence[Mapping[str, Any] | Any],
        held_out: Sequence[Mapping[str, Any] | Any] = (),
        adversarial: Sequence[Mapping[str, Any] | Any] = (),
        synthesis_receipt: Mapping[str, Any] | Any | None = None,
        normalization_receipt: Mapping[str, Any] | Any | None = None,
        promote: PromoteFn | None = None,
        vector_evidence: Any = None,
    ) -> RefactorProcedureCompilationReceipt:
        return compile_refactor_procedure(
            waves=waves,
            transitions=transitions,
            held_out=held_out,
            adversarial=adversarial,
            synthesis_receipt=synthesis_receipt,
            normalization_receipt=normalization_receipt,
            promote=promote,
            vector_evidence=vector_evidence,
        )

    def dry_run(
        self,
        *,
        waves: Sequence[Mapping[str, Any] | Any],
        transitions: Sequence[Mapping[str, Any] | Any],
        held_out: Sequence[Mapping[str, Any] | Any] = (),
        adversarial: Sequence[Mapping[str, Any] | Any] = (),
        synthesis_receipt: Mapping[str, Any] | Any | None = None,
        normalization_receipt: Mapping[str, Any] | Any | None = None,
        promote: PromoteFn | None = None,
        vector_evidence: Any = None,
    ) -> RefactorProcedureCompilationReceipt:
        return dry_run_refactor_procedure(
            waves=waves,
            transitions=transitions,
            held_out=held_out,
            adversarial=adversarial,
            synthesis_receipt=synthesis_receipt,
            normalization_receipt=normalization_receipt,
            promote=promote,
            vector_evidence=vector_evidence,
        )


def encode_canonical_trajectory(
    trajectory: NormalizedRefactorTrajectory,
) -> dict[str, Any]:
    return trajectory.to_dict()


def decode_canonical_trajectory(
    payload: Mapping[str, Any],
) -> NormalizedRefactorTrajectory:
    return NormalizedRefactorTrajectory.from_dict(payload)


def encode_canonical_plan(plan: AntiUnifiedRefactorPlan) -> dict[str, Any]:
    return plan.to_dict()


def decode_canonical_plan(payload: Mapping[str, Any]) -> AntiUnifiedRefactorPlan:
    return AntiUnifiedRefactorPlan.from_dict(payload)


def encode_canonical_contract(contract: InferredProcedureContract) -> dict[str, Any]:
    return contract.to_dict()


def decode_canonical_contract(payload: Mapping[str, Any]) -> InferredProcedureContract:
    return InferredProcedureContract.from_dict(payload)


def encode_canonical_nomination(
    nomination: ProcedurePromotionNomination,
) -> dict[str, Any]:
    return nomination.to_dict()


def decode_canonical_nomination(
    payload: Mapping[str, Any],
) -> ProcedurePromotionNomination:
    return ProcedurePromotionNomination.from_dict(payload)


def encode_canonical_receipt(
    receipt: RefactorProcedureCompilationReceipt,
) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
) -> RefactorProcedureCompilationReceipt:
    return RefactorProcedureCompilationReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise ProcedureAdapterError(
            f"procedure adapter must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_IS_NOMINATION_ONLY",
    "ADVERSARIAL_QUALIFICATION_INTERFACE",
    "ADVERSARIAL_QUALIFICATION_ROUTE_EXISTS",
    "ALLOWED_HOLE_TYPES",
    "ANALYZER_ID",
    "ANTI_UNIFIED_REFACTOR_PLAN_INTERFACE",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "CANDIDATES_REMAIN_UNPROMOTED",
    "DECLARED_COMPILATION_STATUSES",
    "DECLARED_OUTCOMES",
    "DECLARED_QUALIFICATION_ROUTES",
    "DECLARED_QUALIFICATION_STATUSES",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "FORBIDDEN_HOLE_TYPES",
    "FORBIDDEN_PROCEDURE_NAMES",
    "GENERAL_PYTHON_EQUIVALENCE_CLAIMED",
    "GOAL_ID",
    "HELD_OUT_QUALIFICATION_INTERFACE",
    "HELD_OUT_QUALIFICATION_REQUIRED_FOR_PROMOTION",
    "HOLES_PRESERVED",
    "IDENTITY_EXCLUDED_FIELDS",
    "IMPLICIT_INSTALL_FORBIDDEN",
    "IMPLICIT_NETWORK_FORBIDDEN",
    "INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS",
    "INFERRED_PROCEDURE_CONTRACT_INTERFACE",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "NORMALIZED_REFACTOR_TRAJECTORY_INTERFACE",
    "PATHS_NEVER_BECOME_PARAMETERS",
    "PRESERVED_HOLE_INTERFACE",
    "PROCEDURE_ADAPTER_CONTRACT_VERSION",
    "PROCEDURE_AUTHORITY",
    "PROCEDURE_AUTHORITY_INTERFACE",
    "PROCEDURE_AUTHORITY_OWNER",
    "PROCEDURE_CANNOT_SELF_CERTIFY",
    "PROCEDURE_CANNOT_SELF_PROMOTE",
    "PROCEDURE_CAN_AUTHORIZE_COMPLETION",
    "PROCEDURE_CAN_AUTHORIZE_TRANSITION",
    "PROCEDURE_CAN_CREATE_AUTHORITY",
    "PROCEDURE_CAN_CREATE_PROCEDURE_AUTHORITY",
    "PROCEDURE_PROMOTION_NOMINATION_INTERFACE",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "PROOF_CARRYING_PROCEDURE_REFACTOR_ADAPTER_INTERFACE",
    "RAW_SOURCE_REQUIRED",
    "REFACTOR_PROCEDURE_COMPILATION_RECEIPT_INTERFACE",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TEST_PASS_IS_NOT_PROOF",
    "UNKNOWN_REMAINS_UNKNOWN",
    "VALIDATION_NEVER_DROPPED",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "AdversarialPolarity",
    "AdversarialQualification",
    "AntiUnifiedRefactorPlan",
    "CompilationStatus",
    "HeldOutQualification",
    "InferredProcedureContract",
    "NormalizedRefactorTrajectory",
    "PreservedHole",
    "ProcedureAdapterError",
    "ProcedurePromotionNomination",
    "ProofCarryingProcedureRefactorAdapter",
    "QualificationRoute",
    "QualificationStatus",
    "RefactorProcedureCompilationReceipt",
    "TrajectoryOutcome",
    "anti_unify_refactor_plans",
    "assert_not_competing_capsule_family",
    "compile_normalized_trajectory",
    "compile_preserved_hole",
    "compile_refactor_procedure",
    "decode_canonical_contract",
    "decode_canonical_nomination",
    "decode_canonical_plan",
    "decode_canonical_receipt",
    "decode_canonical_trajectory",
    "dry_run_refactor_procedure",
    "encode_canonical_contract",
    "encode_canonical_nomination",
    "encode_canonical_plan",
    "encode_canonical_receipt",
    "encode_canonical_trajectory",
    "infer_procedure_contract",
    "nominate_procedure_promotion",
    "preserve_holes",
    "procedure_adapter_cid_profile",
    "procedure_adapter_descriptor",
    "provider_free_exports",
    "qualify_adversarial",
    "qualify_held_out",
]
