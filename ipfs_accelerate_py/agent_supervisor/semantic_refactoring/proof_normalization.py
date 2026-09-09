"""SPAR-031 proof-backed normalization, interpolation, and refinement adapter.

This module extends current supervisor proof authorities with
``RefactorProofNormalizationAdapter@1``.  It consumes SPAR-027 translation
validation, SPAR-028 differential mismatches, and SPAR-030 reconstructed
proofs / replayed counterexamples, then:

* saturates a bounded typed e-graph under reviewed equality rewrites;
* extracts interpolants for smaller boundary summaries when independently
  justified;
* refines coarse abstractions only from independently replayed
  counterexamples; and
* re-enters SPAR-027 translation validation.

Production Tactician, Hammer, and translation validation remain the proof
and validation authorities.  This adapter does not mint a competing proof,
kernel, or completion authority.  Equality saturation cannot claim general
Python equivalence.  Unvalidated interpolants fail closed.  Raw
countermodels cannot refine abstractions until independent replay.  Unknown
remains unknown.  Unsupported required behavior is a typed terminal, never
success.

The adapter is nomination-only.  Vector, model, and heuristic evidence
cannot admit a normal form or suppress raw-source fallback.  Observational
metadata is excluded from identity.  Dry-run is deterministic and never
mutates.  Network is denied.
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


TASK_ID: Final[str] = "SPAR-031"
GOAL_ID: Final[str] = "SPAR-G053"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "proof normalization"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.proof_normalization@1"
)
TRANSLATION_VALIDATION_OWNER: Final[str] = "ipfs_accelerate_py"
DIFFERENTIAL_OWNER: Final[str] = "ipfs_accelerate_py"
PROOF_ADAPTER_OWNER: Final[str] = "ipfs_accelerate_py"

REFACTOR_PROOF_NORMALIZATION_ADAPTER_INTERFACE: Final[str] = (
    "RefactorProofNormalizationAdapter@1"
)
SUITABLE_EXPRESSION_INTERFACE: Final[str] = "SuitableExpression@1"
EQUALITY_REWRITE_INTERFACE: Final[str] = "EqualityRewrite@1"
EGRAPH_SATURATION_INTERFACE: Final[str] = "EGraphSaturation@1"
INTERPOLANT_INTERFACE: Final[str] = "Interpolant@1"
ABSTRACTION_REFINEMENT_INTERFACE: Final[str] = "AbstractionRefinement@1"
BOUNDARY_SUMMARY_INTERFACE: Final[str] = "BoundarySummary@1"
NORMALIZATION_FORM_INTERFACE: Final[str] = "NormalizationForm@1"
NORMALIZATION_STEP_INTERFACE: Final[str] = "NormalizationStep@1"
PROOF_NORMALIZATION_RECEIPT_INTERFACE: Final[str] = "ProofNormalizationReceipt@1"
VALIDATION_REENTRY_INTERFACE: Final[str] = "ValidationReentry@1"

REFACTOR_PROOF_NORMALIZATION_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-proof-normalization-adapter@1"
)
SUITABLE_EXPRESSION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/suitable-expression@1"
)
EQUALITY_REWRITE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/equality-rewrite@1"
)
EGRAPH_SATURATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/egraph-saturation@1"
)
INTERPOLANT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/interpolant@1"
)
ABSTRACTION_REFINEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/abstraction-refinement@1"
)
BOUNDARY_SUMMARY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/boundary-summary@1"
)
NORMALIZATION_FORM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/normalization-form@1"
)
NORMALIZATION_STEP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/normalization-step@1"
)
PROOF_NORMALIZATION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-normalization-receipt@1"
)
VALIDATION_REENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/normalization-validation-reentry@1"
)
EQUALITY_SATURATION_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-normalization-capability@1"
)

PROOF_NORMALIZATION_CONTRACT_VERSION: Final[str] = "1"

PROOF_NORMALIZATION_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
PROOF_NORMALIZATION_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
PROOF_NORMALIZATION_CAN_CREATE_AUTHORITY: Final[bool] = False
PROOF_NORMALIZATION_CAN_CREATE_PROOF_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
TEST_PASS_IS_NOT_PROOF: Final[bool] = True
NORMAL_FORM_CANNOT_ADMIT_PROOFS: Final[bool] = True
RAW_COUNTERMODEL_CANNOT_REFUTE: Final[bool] = True
UNVALIDATED_INTERPOLANTS_FAIL_CLOSED: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
ADAPTER_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
NETWORK_DENIED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
UNKNOWN_REMAINS_UNKNOWN: Final[bool] = True
GUESSED_AXIOMS_REJECTED: Final[bool] = True
INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS: Final[bool] = True
GENERAL_PYTHON_EQUIVALENCE_CLAIMED: Final[bool] = False
IMPLICIT_INSTALL_FORBIDDEN: Final[bool] = True
IMPLICIT_NETWORK_FORBIDDEN: Final[bool] = True
NORMALIZED_FORMS_REENTER_VALIDATION: Final[bool] = True
FORMS_REMAIN_UNPROMOTED: Final[bool] = True

NETWORK_DENY: Final[str] = "deny"
REWRITE_SELECTOR_DETERMINISTIC: Final[str] = "deterministic"
DEFAULT_MAX_STEPS: Final[int] = 8
ABSOLUTE_MAX_STEPS: Final[int] = 64
DEFAULT_MAX_EXPRESSIONS: Final[int] = 64
VALIDATOR_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.translation_validation@1"
)

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_EXPRESSIONS: Final[int] = 256
MAX_REWRITES: Final[int] = 256
MAX_INTERPOLANTS: Final[int] = 128
MAX_REFINEMENTS: Final[int] = 128
MAX_SUMMARIES: Final[int] = 128
MAX_FORMS: Final[int] = 64
MAX_STEPS: Final[int] = ABSOLUTE_MAX_STEPS
MAX_ARGS: Final[int] = 32
MAX_COST: Final[int] = 1_024

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

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
    "can_create_proof_authority",
    "projection_is_authority",
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

FORBIDDEN_NORMALIZATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "claim_general_equivalence",
        "admit_vector_proof",
        "promote_candidate_to_proof",
        "guess_axiom",
        "suppress_raw_source",
        "open_network",
        "implicit_install",
        "collapse_evidence",
    }
)

EXPRESSION_SOURCE_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "specification",
        "exact_static_fact",
        "admitted_contract",
        "reconstructed_proof",
    }
)

FORBIDDEN_EXPRESSION_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
        "guessed",
        "retrieved_text",
        "natural_language",
        "proof_candidate",
    }
)

REWRITE_JUSTIFICATIONS: Final[frozenset[str]] = frozenset(
    {
        "reconstructed_proof",
        "admitted_contract",
        "specification",
    }
)

INTERPOLANT_JUSTIFICATIONS: Final[frozenset[str]] = frozenset(
    {
        "reconstructed_proof",
        "replayed_counterexample",
    }
)

UNSUPPORTED_SEMANTICS: Final[frozenset[str]] = frozenset(
    {
        "higher_order",
        "dependent",
        "dynamic",
        "native",
        "concurrency",
        "lifetime",
        "unbounded",
    }
)

_AVAILABLE_FEATURES: Final[tuple[str, ...]] = (
    "typed_egraph",
    "equality_saturation",
    "congruence_rebuild",
    "reviewed_rewrite_replay",
    "finite_interpolation",
    "abstraction_refinement_from_replayed_ce",
    "boundary_summary_extraction",
)

_UNAVAILABLE_FEATURES: Final[tuple[tuple[str, str], ...]] = (
    ("external_egg_runtime", "no competing egg authority"),
    ("solver_semantic_side_conditions", "unsupported required semantics"),
    ("kernel_equivalence", "hammer remains production proof authority"),
    ("higher_order_interpolation", "unsupported required semantics"),
    ("unbounded_egraph", "steps remain bounded"),
    ("general_python_equivalence", "general Python equivalence is not claimed"),
)


class ProofNormalizationError(ValueError):
    """Fail-closed violation of a SPAR-031 proof-normalization contract."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "malformed",
        negative_evidence_cids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.negative_evidence_cids = tuple(negative_evidence_cids)


class ExpressionKind(str, Enum):
    EXPRESSION = "expression"
    ADAPTER = "adapter"
    IMPORT_REWRITE = "import_rewrite"
    STATE_PROJECTION = "state_projection"
    BOUNDARY_SUMMARY = "boundary_summary"


class FeatureStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class NormalizationStatus(str, Enum):
    NORMALIZED = "normalized"
    INTERPOLATED = "interpolated"
    REFINED = "refined"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    STALE = "stale"


class StepKind(str, Enum):
    EQUALITY_SATURATION = "equality_saturation"
    INTERPOLATION = "interpolation"
    ABSTRACTION_REFINEMENT = "abstraction_refinement"
    VALIDATION_REENTRY = "validation_reentry"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    STALE = "stale"
    ERROR = "error"


class RefinementGrain(str, Enum):
    COARSE = "coarse"
    REFINED = "refined"


class AbstractionKind(str, Enum):
    TYPE = "type"
    EFFECT = "effect"
    STATE = "state"
    INIT_ORDER = "init_order"
    PROTOCOL = "protocol"
    IMPORT = "import"
    BOUNDARY = "boundary"


class ValidationReentryStatus(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"
    MISSING = "missing"


DECLARED_EXPRESSION_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in ExpressionKind
)
DECLARED_NORMALIZATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in NormalizationStatus
)
DECLARED_STEP_KINDS: Final[frozenset[str]] = frozenset(item.value for item in StepKind)
DECLARED_REFINEMENT_GRAINS: Final[frozenset[str]] = frozenset(
    item.value for item in RefinementGrain
)
DECLARED_ABSTRACTION_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in AbstractionKind
)
DECLARED_VALIDATION_REENTRY_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in ValidationReentryStatus
)
DECLARED_FEATURE_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in FeatureStatus
)

ValidateFn = Callable[..., Mapping[str, Any]]


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise ProofNormalizationError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ProofNormalizationError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise ProofNormalizationError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise ProofNormalizationError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise ProofNormalizationError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise ProofNormalizationError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProofNormalizationError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int) -> int:
    if type(value) is bool or type(value) is not int:
        raise ProofNormalizationError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ProofNormalizationError(f"{name} is out of bounds")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ProofNormalizationError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise ProofNormalizationError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise ProofNormalizationError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise ProofNormalizationError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise ProofNormalizationError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise ProofNormalizationError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _reject_forbidden_bodies(payload: Any, name: str) -> None:
    if isinstance(payload, Mapping) and not isinstance(payload, (str, bytes, bytearray)):
        present = _FORBIDDEN_BODY_KEYS & set(payload)
        if present:
            raise ProofNormalizationError(
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
        raise ProofNormalizationError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise ProofNormalizationError(f"{name} does not verify")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise ProofNormalizationError(f"unknown {name}: {text}") from exc


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise ProofNormalizationError(f"{name} cannot claim {flag}")


def _ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ProofNormalizationError(f"{name} must be a list")
    ordered = tuple(_text(item, name) for item in values)
    if len(ordered) > limit:
        raise ProofNormalizationError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise ProofNormalizationError(f"{name} must not contain duplicates")
    return ordered


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, ()):
        if required:
            raise ProofNormalizationError(f"{name} are required")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProofNormalizationError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise ProofNormalizationError(f"{name} are required")
    if len(ordered) != len(set(ordered)):
        raise ProofNormalizationError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise ProofNormalizationError(f"{name} exceed maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise ProofNormalizationError(f"{name} exceeds path bound")
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
        raise ProofNormalizationError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise ProofNormalizationError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProofNormalizationError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ProofNormalizationError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise ProofNormalizationError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProofNormalizationError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise ProofNormalizationError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise ProofNormalizationError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise ProofNormalizationError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise ProofNormalizationError(f"missing {name}")
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
    raise ProofNormalizationError(f"{name} must be a mapping")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProofNormalizationError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping) and not isinstance(item, (str, bytes, bytearray)):
            _reject_forbidden_bodies(item, name)
            items.append(dict(item))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                _reject_forbidden_bodies(payload, name)
                items.append(dict(payload))
                continue
        raise ProofNormalizationError(f"{name} items must be objects")
    if len(items) > MAX_MEMBERS:
        raise ProofNormalizationError(f"{name} exceeds maximum length")
    return tuple(items)


def _require_tree(value: Any, expected: str, label: str) -> None:
    if value in (None, ""):
        raise ProofNormalizationError(f"{label} tree_id is required")
    actual = _tree_id(value)
    if actual != expected:
        raise ProofNormalizationError(f"{label} tree_id does not match")


def _reject_non_admitting_payload(payload: Mapping[str, Any], name: str) -> None:
    evidence = payload.get("evidence_class")
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise ProofNormalizationError(f"{name} {evidence} cannot admit")
    if payload.get("admit_proof") is True or payload.get("admits_proof") is True:
        raise ProofNormalizationError(f"{name} cannot admit proofs")
    if payload.get("admit_equivalence") is True:
        raise ProofNormalizationError(f"{name} cannot admit")
    if payload.get("guessed_axiom") is True:
        raise ProofNormalizationError("guessed axioms are rejected")


def _reject_vector_admission(vector_evidence: Any) -> None:
    if vector_evidence in (None, (), {}):
        return
    if isinstance(vector_evidence, Mapping):
        if vector_evidence.get("suppress_raw_source") is True:
            raise ProofNormalizationError("vectors cannot suppress raw-source fallback")
        _reject_non_admitting_payload(vector_evidence, "vector_evidence")
        evidence = vector_evidence.get("evidence_class")
        if evidence in _NON_ADMITTING_EVIDENCE:
            raise ProofNormalizationError("vector/model/heuristic evidence cannot admit")
        return
    raise ProofNormalizationError("vector_evidence must be an object")


def _network_value(value: Any) -> str:
    if value in (None, ""):
        return NETWORK_DENY
    text = _text(value, "network")
    if text != NETWORK_DENY:
        raise ProofNormalizationError("network is denied")
    return NETWORK_DENY


def proof_normalization_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def proof_normalization_descriptor() -> dict[str, Any]:
    return {
        "interface": REFACTOR_PROOF_NORMALIZATION_ADAPTER_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "raw_source_required": True,
        "nomination_only": True,
        "network": NETWORK_DENY,
        "claims_general_equivalence": False,
        "normal_form_cannot_admit_proofs": True,
        "raw_countermodel_cannot_refute": True,
        "unvalidated_interpolants_fail_closed": True,
        "unknown_remains_unknown": True,
        "guessed_axioms_rejected": True,
        "normalized_forms_reenter_validation": True,
        "forms_remain_unpromoted": True,
        "rewrite_selector": REWRITE_SELECTOR_DETERMINISTIC,
        "forbids": tuple(sorted(FORBIDDEN_NORMALIZATION_NAMES)),
    }


def _wave_cid(wave: Mapping[str, Any]) -> str:
    value = wave.get("receipt_cid") or wave.get("wave_cid")
    if value in (None, ""):
        raise ProofNormalizationError("SPAR-025 receipt_cid is required")
    return _cid(value, "SPAR-025 receipt_cid")


def _wave_packet_cids(wave: Mapping[str, Any]) -> tuple[str, ...]:
    values = wave.get("packet_cids")
    if values in (None, ()):
        raise ProofNormalizationError("SPAR-025 packet_cids are required")
    return _cids(values, "SPAR-025 packet_cids", required=True)


def _wave_write_paths(wave: Mapping[str, Any]) -> tuple[str, ...]:
    values = wave.get("write_paths")
    if values in (None, ()):
        raise ProofNormalizationError("SPAR-025 write_paths are required")
    return _exact_paths(values, "SPAR-025 write_paths")


def _selection_cid(selection: Mapping[str, Any]) -> str:
    value = selection.get("validation_selection_cid") or selection.get("selection_cid")
    if value in (None, ""):
        raise ProofNormalizationError("SPAR-026 validation_selection_cid is required")
    return _cid(value, "SPAR-026 validation_selection_cid")


def _selection_packet_cid(selection: Mapping[str, Any]) -> str:
    value = selection.get("packet_cid")
    if value in (None, ""):
        raise ProofNormalizationError("SPAR-026 packet_cid is required")
    return _cid(value, "SPAR-026 packet_cid")


def _selection_write_paths(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("write_paths")
    if values in (None, ()):
        raise ProofNormalizationError("SPAR-026 write_paths are required")
    return _exact_paths(values, "SPAR-026 write_paths")


def _selection_sources(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("raw_source_cids")
    if values in (None, (), []):
        raise ProofNormalizationError("raw source required")
    sources = _cids(values, "raw_source_cids", required=True)
    if not sources:
        raise ProofNormalizationError("raw source required")
    return sources


def _selection_commands(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("validation_commands")
    if values in (None, ()):
        raise ProofNormalizationError("validation_commands must not be empty")
    return _commands(values, "validation_commands")


def _bind_predecessors(
    *,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
) -> tuple[str, str, str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    wave_payload = _as_mapping(wave, "SPAR-025 wave")
    selection_payload = _as_mapping(selection, "SPAR-026 selection")
    tree_id = _tree_id(wave_payload.get("tree_id"))
    _require_tree(selection_payload.get("tree_id"), tree_id, "SPAR-026")
    _reject_non_admitting_payload(wave_payload, "SPAR-025 wave")
    _reject_non_admitting_payload(selection_payload, "SPAR-026 selection")

    status = _text(wave_payload.get("status") or "applied", "SPAR-025 status")
    if status != "applied":
        raise ProofNormalizationError("SPAR-025 wave must be applied")
    if wave_payload.get("writes_repository") is True:
        raise ProofNormalizationError("SPAR-025 wave cannot write the repository")
    if wave_payload.get("executor_is_nomination_only") is False:
        raise ProofNormalizationError("SPAR-025 executor must remain nomination_only")
    if selection_payload.get("adapter_is_nomination_only") is False:
        raise ProofNormalizationError("SPAR-026 adapter must remain nomination_only")
    if selection_payload.get("raw_source_required") is False:
        raise ProofNormalizationError("raw source required")
    if selection_payload.get("datasets_owns_selection") is False:
        raise ProofNormalizationError("datasets remains the selection authority")

    packet_cid = _selection_packet_cid(selection_payload)
    packet_cids = _wave_packet_cids(wave_payload)
    if packet_cid not in packet_cids:
        raise ProofNormalizationError("SPAR-026 packet_cid is not in SPAR-025 packet_cids")

    write_paths = _selection_write_paths(selection_payload)
    wave_paths = _wave_write_paths(wave_payload)
    if write_paths != wave_paths:
        raise ProofNormalizationError("SPAR-025/SPAR-026 write_paths do not match")

    return (
        tree_id,
        _wave_cid(wave_payload),
        _selection_cid(selection_payload),
        packet_cid,
        write_paths,
        _selection_sources(selection_payload),
        _selection_commands(selection_payload),
    )


def _strip_envelope(payload: Mapping[str, Any], *cid_fields: str) -> dict[str, Any]:
    data = dict(payload)
    data.pop("schema", None)
    data.pop("interface", None)
    for field in cid_fields:
        data.pop(field, None)
    for flag in _AUTHORITY_FLAG_NAMES:
        data.pop(flag, None)
    data.pop("nomination_only", None)
    data.pop("adapter_is_nomination_only", None)
    return data


@dataclass(frozen=True, slots=True)
class SuitableExpression:
    """One finite suitable expression, adapter, import, or projection."""

    expression_id: str
    term_id: str
    kind: str
    tree_id: str
    source_cid: str
    source_authority: str
    operator: str = ""
    arg_term_ids: Sequence[str] = ()
    finite: bool = True
    required: bool = True
    cost: int = 1
    semantics: str = ""

    interface: ClassVar[str] = SUITABLE_EXPRESSION_INTERFACE
    schema: ClassVar[str] = SUITABLE_EXPRESSION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "expression_id",
            "term_id",
            "kind",
            "tree_id",
            "source_cid",
            "source_authority",
            "operator",
            "arg_term_ids",
            "finite",
            "required",
            "cost",
            "semantics",
            "expression_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expression_id", _text(self.expression_id, "expression_id")
        )
        object.__setattr__(self, "term_id", _cid(self.term_id, "term_id"))
        object.__setattr__(self, "kind", _enum(self.kind, ExpressionKind, "kind"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "source_cid", _cid(self.source_cid, "source_cid"))
        authority = _text(self.source_authority, "source_authority")
        if authority in FORBIDDEN_EXPRESSION_AUTHORITIES:
            raise ProofNormalizationError(
                "guessed/vector/model expressions cannot become axioms"
            )
        if authority not in EXPRESSION_SOURCE_AUTHORITIES:
            raise ProofNormalizationError(
                f"unsupported expression source_authority {authority!r}"
            )
        object.__setattr__(self, "source_authority", authority)
        object.__setattr__(
            self, "operator", _text(self.operator, "operator", empty=True)
        )
        args = (
            _ordered_text(list(self.arg_term_ids), "arg_term_ids", limit=MAX_ARGS)
            if self.arg_term_ids
            else ()
        )
        for item in args:
            _cid(item, "arg_term_ids")
        object.__setattr__(self, "arg_term_ids", args)
        object.__setattr__(self, "finite", _bool(self.finite, "finite"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(
            self, "cost", _int(self.cost, "cost", minimum=0, maximum=MAX_COST)
        )
        semantics = _text(self.semantics, "semantics", empty=True)
        if semantics in UNSUPPORTED_SEMANTICS and self.required:
            raise ProofNormalizationError("unsupported required expression semantics")
        object.__setattr__(self, "semantics", semantics)
        if self.required and self.finite is False:
            raise ProofNormalizationError("unsupported required non-finite expression")
        if self.operator and not args:
            raise ProofNormalizationError("operator expressions require arg_term_ids")
        if args and not self.operator:
            raise ProofNormalizationError("arg_term_ids require an operator")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SUITABLE_EXPRESSION_SCHEMA,
            "interface": SUITABLE_EXPRESSION_INTERFACE,
            "expression_id": self.expression_id,
            "term_id": self.term_id,
            "kind": self.kind,
            "tree_id": self.tree_id,
            "source_cid": self.source_cid,
            "source_authority": self.source_authority,
            "operator": self.operator,
            "arg_term_ids": list(self.arg_term_ids),
            "finite": self.finite,
            "required": self.required,
            "cost": self.cost,
            "semantics": self.semantics,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def expression_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["expression_cid"] = self.expression_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SuitableExpression":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("expression_cid")
        if payload.pop("schema") != SUITABLE_EXPRESSION_SCHEMA:
            raise ProofNormalizationError("unsupported SuitableExpression schema")
        if payload.pop("interface") != SUITABLE_EXPRESSION_INTERFACE:
            raise ProofNormalizationError("unsupported SuitableExpression interface")
        result = cls(**payload)
        _verify_cid(claimed, result.expression_cid, "expression_cid")
        return result


@dataclass(frozen=True, slots=True)
class EqualityRewrite:
    """One reviewed, oriented equality rewrite. Nomination-only."""

    rule_id: str
    lhs_term_id: str
    rhs_term_id: str
    review_ref: str
    justification: str
    tree_id: str
    reconstruction_cid: str = ""
    oriented: bool = True
    cost: int = 1
    theory_id: str = ""

    interface: ClassVar[str] = EQUALITY_REWRITE_INTERFACE
    schema: ClassVar[str] = EQUALITY_REWRITE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "rule_id",
            "lhs_term_id",
            "rhs_term_id",
            "review_ref",
            "justification",
            "tree_id",
            "reconstruction_cid",
            "oriented",
            "cost",
            "theory_id",
            "rewrite_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, "rule_id"))
        object.__setattr__(self, "lhs_term_id", _cid(self.lhs_term_id, "lhs_term_id"))
        object.__setattr__(self, "rhs_term_id", _cid(self.rhs_term_id, "rhs_term_id"))
        object.__setattr__(self, "review_ref", _text(self.review_ref, "review_ref"))
        justification = _text(self.justification, "justification")
        if justification not in REWRITE_JUSTIFICATIONS:
            raise ProofNormalizationError(
                f"unsupported rewrite justification {justification!r}"
            )
        object.__setattr__(self, "justification", justification)
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        reconstruction = _optional_cid(self.reconstruction_cid, "reconstruction_cid")
        if justification == "reconstructed_proof" and not reconstruction:
            raise ProofNormalizationError(
                "reconstructed_proof rewrite requires reconstruction_cid"
            )
        object.__setattr__(self, "reconstruction_cid", reconstruction)
        if _bool(self.oriented, "oriented") is not True:
            raise ProofNormalizationError("equality rewrites must remain oriented")
        object.__setattr__(self, "oriented", True)
        object.__setattr__(
            self, "cost", _int(self.cost, "cost", minimum=0, maximum=MAX_COST)
        )
        object.__setattr__(
            self, "theory_id", _text(self.theory_id, "theory_id", empty=True)
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EQUALITY_REWRITE_SCHEMA,
            "interface": EQUALITY_REWRITE_INTERFACE,
            "rule_id": self.rule_id,
            "lhs_term_id": self.lhs_term_id,
            "rhs_term_id": self.rhs_term_id,
            "review_ref": self.review_ref,
            "justification": self.justification,
            "tree_id": self.tree_id,
            "reconstruction_cid": self.reconstruction_cid,
            "oriented": True,
            "cost": self.cost,
            "theory_id": self.theory_id,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def rewrite_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["rewrite_cid"] = self.rewrite_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EqualityRewrite":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("rewrite_cid")
        if payload.pop("schema") != EQUALITY_REWRITE_SCHEMA:
            raise ProofNormalizationError("unsupported EqualityRewrite schema")
        if payload.pop("interface") != EQUALITY_REWRITE_INTERFACE:
            raise ProofNormalizationError("unsupported EqualityRewrite interface")
        result = cls(**payload)
        _verify_cid(claimed, result.rewrite_cid, "rewrite_cid")
        return result


@dataclass(frozen=True, slots=True)
class EGraphClass:
    """One e-class after bounded equality saturation."""

    class_id: str
    member_term_ids: Sequence[str]
    representative_term_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_id", _text(self.class_id, "class_id"))
        members = _cids(self.member_term_ids, "member_term_ids", required=True)
        object.__setattr__(self, "member_term_ids", members)
        representative = _cid(self.representative_term_id, "representative_term_id")
        if representative not in members:
            raise ProofNormalizationError("representative_term_id must be a class member")
        object.__setattr__(self, "representative_term_id", representative)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "member_term_ids": list(self.member_term_ids),
            "representative_term_id": self.representative_term_id,
        }


@dataclass(frozen=True, slots=True)
class EGraphSaturation:
    """Bounded equality-saturation snapshot. Nomination-only."""

    tree_id: str
    theory_id: str
    classes: Sequence[EGraphClass | Mapping[str, Any]] = ()
    rewrite_cids: Sequence[str] = ()
    steps_used: int = 0
    timed_out: bool = False

    interface: ClassVar[str] = EGRAPH_SATURATION_INTERFACE
    schema: ClassVar[str] = EGRAPH_SATURATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "theory_id",
            "classes",
            "rewrite_cids",
            "steps_used",
            "timed_out",
            "saturation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "theory_id", _text(self.theory_id, "theory_id", empty=True))
        classes: list[EGraphClass] = []
        for item in self.classes:
            if isinstance(item, EGraphClass):
                classes.append(item)
            elif isinstance(item, Mapping):
                classes.append(
                    EGraphClass(
                        class_id=item["class_id"],
                        member_term_ids=item["member_term_ids"],
                        representative_term_id=item["representative_term_id"],
                    )
                )
            else:
                raise ProofNormalizationError("classes items must be objects")
        object.__setattr__(self, "classes", tuple(classes))
        object.__setattr__(self, "rewrite_cids", _cids(self.rewrite_cids, "rewrite_cids"))
        object.__setattr__(
            self,
            "steps_used",
            _int(self.steps_used, "steps_used", minimum=0, maximum=ABSOLUTE_MAX_STEPS),
        )
        object.__setattr__(self, "timed_out", _bool(self.timed_out, "timed_out"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EGRAPH_SATURATION_SCHEMA,
            "interface": EGRAPH_SATURATION_INTERFACE,
            "tree_id": self.tree_id,
            "theory_id": self.theory_id,
            "classes": [item.identity_payload() for item in self.classes],
            "rewrite_cids": list(self.rewrite_cids),
            "steps_used": self.steps_used,
            "timed_out": self.timed_out,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def saturation_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["saturation_cid"] = self.saturation_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EGraphSaturation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("saturation_cid")
        if payload.pop("schema") != EGRAPH_SATURATION_SCHEMA:
            raise ProofNormalizationError("unsupported EGraphSaturation schema")
        if payload.pop("interface") != EGRAPH_SATURATION_INTERFACE:
            raise ProofNormalizationError("unsupported EGraphSaturation interface")
        result = cls(**payload)
        _verify_cid(claimed, result.saturation_cid, "saturation_cid")
        return result


@dataclass(frozen=True, slots=True)
class Interpolant:
    """One independently justified interpolant. Unvalidated interpolants fail closed."""

    interpolant_id: str
    predicate_id: str
    tree_id: str
    a_term_ids: Sequence[str]
    b_term_ids: Sequence[str]
    justification: str
    justification_cid: str
    validated: bool = True

    interface: ClassVar[str] = INTERPOLANT_INTERFACE
    schema: ClassVar[str] = INTERPOLANT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "interpolant_id",
            "predicate_id",
            "tree_id",
            "a_term_ids",
            "b_term_ids",
            "justification",
            "justification_cid",
            "validated",
            "interpolant_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "interpolant_id", _text(self.interpolant_id, "interpolant_id")
        )
        object.__setattr__(self, "predicate_id", _text(self.predicate_id, "predicate_id"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        a_terms = _cids(self.a_term_ids, "a_term_ids", required=True)
        b_terms = _cids(self.b_term_ids, "b_term_ids", required=True)
        if set(a_terms) & set(b_terms):
            raise ProofNormalizationError("interpolant partitions must be disjoint")
        object.__setattr__(self, "a_term_ids", a_terms)
        object.__setattr__(self, "b_term_ids", b_terms)
        justification = _text(self.justification, "justification")
        if justification not in INTERPOLANT_JUSTIFICATIONS:
            raise ProofNormalizationError(
                f"unsupported interpolant justification {justification!r}"
            )
        object.__setattr__(self, "justification", justification)
        object.__setattr__(
            self, "justification_cid", _cid(self.justification_cid, "justification_cid")
        )
        if _bool(self.validated, "validated") is not True:
            raise ProofNormalizationError("unvalidated interpolants fail closed")
        object.__setattr__(self, "validated", True)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": INTERPOLANT_SCHEMA,
            "interface": INTERPOLANT_INTERFACE,
            "interpolant_id": self.interpolant_id,
            "predicate_id": self.predicate_id,
            "tree_id": self.tree_id,
            "a_term_ids": list(self.a_term_ids),
            "b_term_ids": list(self.b_term_ids),
            "justification": self.justification,
            "justification_cid": self.justification_cid,
            "validated": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def interpolant_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["interpolant_cid"] = self.interpolant_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Interpolant":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("interpolant_cid")
        if payload.pop("schema") != INTERPOLANT_SCHEMA:
            raise ProofNormalizationError("unsupported Interpolant schema")
        if payload.pop("interface") != INTERPOLANT_INTERFACE:
            raise ProofNormalizationError("unsupported Interpolant interface")
        result = cls(**payload)
        _verify_cid(claimed, result.interpolant_cid, "interpolant_cid")
        return result


@dataclass(frozen=True, slots=True)
class AbstractionRefinement:
    """One CEGAR-style refinement justified by a replayed counterexample."""

    predicate_id: str
    kind: str
    tree_id: str
    source_cid: str
    grain: str = RefinementGrain.REFINED.value
    distinguishing: bool = True
    replayed: bool = True

    interface: ClassVar[str] = ABSTRACTION_REFINEMENT_INTERFACE
    schema: ClassVar[str] = ABSTRACTION_REFINEMENT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "predicate_id",
            "kind",
            "tree_id",
            "source_cid",
            "grain",
            "distinguishing",
            "replayed",
            "refinement_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate_id", _text(self.predicate_id, "predicate_id"))
        object.__setattr__(self, "kind", _enum(self.kind, AbstractionKind, "kind"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "source_cid", _cid(self.source_cid, "source_cid"))
        object.__setattr__(self, "grain", _enum(self.grain, RefinementGrain, "grain"))
        object.__setattr__(
            self, "distinguishing", _bool(self.distinguishing, "distinguishing")
        )
        replayed = _bool(self.replayed, "replayed")
        if replayed is not True:
            raise ProofNormalizationError(
                "raw countermodels cannot refine abstractions until replay"
            )
        object.__setattr__(self, "replayed", True)
        if self.distinguishing and self.grain != RefinementGrain.REFINED.value:
            raise ProofNormalizationError("distinguishing predicates must be refined")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ABSTRACTION_REFINEMENT_SCHEMA,
            "interface": ABSTRACTION_REFINEMENT_INTERFACE,
            "predicate_id": self.predicate_id,
            "kind": self.kind,
            "tree_id": self.tree_id,
            "source_cid": self.source_cid,
            "grain": self.grain,
            "distinguishing": self.distinguishing,
            "replayed": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def refinement_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["refinement_cid"] = self.refinement_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AbstractionRefinement":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("refinement_cid")
        if payload.pop("schema") != ABSTRACTION_REFINEMENT_SCHEMA:
            raise ProofNormalizationError("unsupported AbstractionRefinement schema")
        if payload.pop("interface") != ABSTRACTION_REFINEMENT_INTERFACE:
            raise ProofNormalizationError("unsupported AbstractionRefinement interface")
        result = cls(**payload)
        _verify_cid(claimed, result.refinement_cid, "refinement_cid")
        return result


@dataclass(frozen=True, slots=True)
class BoundarySummary:
    """Smaller boundary summary extracted from interpolants/refinements."""

    summary_id: str
    expression_id: str
    tree_id: str
    interpolant_cids: Sequence[str] = ()
    refinement_cids: Sequence[str] = ()
    smaller: bool = True

    interface: ClassVar[str] = BOUNDARY_SUMMARY_INTERFACE
    schema: ClassVar[str] = BOUNDARY_SUMMARY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "summary_id",
            "expression_id",
            "tree_id",
            "interpolant_cids",
            "refinement_cids",
            "smaller",
            "summary_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_id", _text(self.summary_id, "summary_id"))
        object.__setattr__(
            self, "expression_id", _text(self.expression_id, "expression_id")
        )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        interpolants = _cids(self.interpolant_cids, "interpolant_cids")
        refinements = _cids(self.refinement_cids, "refinement_cids")
        if not interpolants and not refinements:
            raise ProofNormalizationError(
                "boundary summary requires interpolant_cids or refinement_cids"
            )
        object.__setattr__(self, "interpolant_cids", interpolants)
        object.__setattr__(self, "refinement_cids", refinements)
        if _bool(self.smaller, "smaller") is not True:
            raise ProofNormalizationError("boundary summaries must remain smaller")
        object.__setattr__(self, "smaller", True)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BOUNDARY_SUMMARY_SCHEMA,
            "interface": BOUNDARY_SUMMARY_INTERFACE,
            "summary_id": self.summary_id,
            "expression_id": self.expression_id,
            "tree_id": self.tree_id,
            "interpolant_cids": list(self.interpolant_cids),
            "refinement_cids": list(self.refinement_cids),
            "smaller": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def summary_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["summary_cid"] = self.summary_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundarySummary":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("summary_cid")
        if payload.pop("schema") != BOUNDARY_SUMMARY_SCHEMA:
            raise ProofNormalizationError("unsupported BoundarySummary schema")
        if payload.pop("interface") != BOUNDARY_SUMMARY_INTERFACE:
            raise ProofNormalizationError("unsupported BoundarySummary interface")
        result = cls(**payload)
        _verify_cid(claimed, result.summary_cid, "summary_cid")
        return result


@dataclass(frozen=True, slots=True)
class NormalizationForm:
    """Extracted normal form. Unpromoted and nomination-only."""

    form_id: str
    tree_id: str
    representative_term_id: str
    member_term_ids: Sequence[str]
    write_paths: Sequence[str]
    source_cids: Sequence[str]
    unpromoted: bool = True

    interface: ClassVar[str] = NORMALIZATION_FORM_INTERFACE
    schema: ClassVar[str] = NORMALIZATION_FORM_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "form_id",
            "tree_id",
            "representative_term_id",
            "member_term_ids",
            "write_paths",
            "source_cids",
            "unpromoted",
            "form_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "form_id", _text(self.form_id, "form_id"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        members = _cids(self.member_term_ids, "member_term_ids", required=True)
        representative = _cid(self.representative_term_id, "representative_term_id")
        if representative not in members:
            raise ProofNormalizationError("representative_term_id must be a class member")
        object.__setattr__(self, "member_term_ids", members)
        object.__setattr__(self, "representative_term_id", representative)
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self, "source_cids", _cids(self.source_cids, "source_cids", required=True)
        )
        if _bool(self.unpromoted, "unpromoted") is not True:
            raise ProofNormalizationError("normalized forms remain unpromoted")
        object.__setattr__(self, "unpromoted", True)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": NORMALIZATION_FORM_SCHEMA,
            "interface": NORMALIZATION_FORM_INTERFACE,
            "form_id": self.form_id,
            "tree_id": self.tree_id,
            "representative_term_id": self.representative_term_id,
            "member_term_ids": list(self.member_term_ids),
            "write_paths": list(self.write_paths),
            "source_cids": list(self.source_cids),
            "unpromoted": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def form_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["form_cid"] = self.form_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizationForm":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("form_cid")
        if payload.pop("schema") != NORMALIZATION_FORM_SCHEMA:
            raise ProofNormalizationError("unsupported NormalizationForm schema")
        if payload.pop("interface") != NORMALIZATION_FORM_INTERFACE:
            raise ProofNormalizationError("unsupported NormalizationForm interface")
        if payload.get("unpromoted") is not True:
            raise ProofNormalizationError("normalized forms remain unpromoted")
        result = cls(**payload)
        _verify_cid(claimed, result.form_cid, "form_cid")
        return result


@dataclass(frozen=True, slots=True)
class NormalizationStep:
    """One bounded normalization, interpolation, or refinement step."""

    step_index: int
    kind: str
    outcome: str
    artifact_cid: str = ""
    reason_code: str = ""

    interface: ClassVar[str] = NORMALIZATION_STEP_INTERFACE
    schema: ClassVar[str] = NORMALIZATION_STEP_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "step_index",
            "kind",
            "outcome",
            "artifact_cid",
            "reason_code",
            "step_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_index",
            _int(self.step_index, "step_index", minimum=1, maximum=ABSOLUTE_MAX_STEPS),
        )
        object.__setattr__(self, "kind", _enum(self.kind, StepKind, "kind"))
        object.__setattr__(
            self, "outcome", _enum(self.outcome, NormalizationStatus, "outcome")
        )
        object.__setattr__(
            self, "artifact_cid", _optional_cid(self.artifact_cid, "artifact_cid")
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", empty=True)
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": NORMALIZATION_STEP_SCHEMA,
            "interface": NORMALIZATION_STEP_INTERFACE,
            "step_index": self.step_index,
            "kind": self.kind,
            "outcome": self.outcome,
            "artifact_cid": self.artifact_cid,
            "reason_code": self.reason_code,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def step_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["step_cid"] = self.step_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizationStep":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("step_cid")
        if payload.pop("schema") != NORMALIZATION_STEP_SCHEMA:
            raise ProofNormalizationError("unsupported NormalizationStep schema")
        if payload.pop("interface") != NORMALIZATION_STEP_INTERFACE:
            raise ProofNormalizationError("unsupported NormalizationStep interface")
        result = cls(**payload)
        _verify_cid(claimed, result.step_cid, "step_cid")
        return result


@dataclass(frozen=True, slots=True)
class ValidationReentry:
    """SPAR-027 re-entry record. A pass remains nomination-only."""

    form_cid: str
    tree_id: str
    status: str
    result_cid: str = ""
    validator_id: str = VALIDATOR_ID
    reason_code: str = ""

    interface: ClassVar[str] = VALIDATION_REENTRY_INTERFACE
    schema: ClassVar[str] = VALIDATION_REENTRY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "form_cid",
            "tree_id",
            "status",
            "result_cid",
            "validator_id",
            "reason_code",
            "reentry_cid",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "nomination_only",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "form_cid", _cid(self.form_cid, "form_cid"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self, "status", _enum(self.status, ValidationReentryStatus, "status")
        )
        object.__setattr__(self, "result_cid", _optional_cid(self.result_cid, "result_cid"))
        validator = _text(self.validator_id, "validator_id")
        if validator != VALIDATOR_ID:
            raise ProofNormalizationError("validator_id must remain SPAR-027")
        object.__setattr__(self, "validator_id", VALIDATOR_ID)
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", empty=True)
        )
        if self.status == ValidationReentryStatus.VALIDATED.value and self.reason_code:
            raise ProofNormalizationError("validated re-entry cannot carry a terminal reason")
        if (
            self.status != ValidationReentryStatus.VALIDATED.value
            and not self.reason_code
        ):
            raise ProofNormalizationError("typed terminal re-entry requires a reason")

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
    def nomination_only(self) -> bool:
        return True

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": VALIDATION_REENTRY_SCHEMA,
            "interface": VALIDATION_REENTRY_INTERFACE,
            "form_cid": self.form_cid,
            "tree_id": self.tree_id,
            "status": self.status,
            "result_cid": self.result_cid,
            "validator_id": VALIDATOR_ID,
            "reason_code": self.reason_code,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "nomination_only": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def reentry_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["reentry_cid"] = self.reentry_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationReentry":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("reentry_cid")
        if payload.pop("schema") != VALIDATION_REENTRY_SCHEMA:
            raise ProofNormalizationError("unsupported ValidationReentry schema")
        if payload.pop("interface") != VALIDATION_REENTRY_INTERFACE:
            raise ProofNormalizationError("unsupported ValidationReentry interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("nomination_only") is not True:
            raise ProofNormalizationError("validation re-entry must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.reentry_cid, "reentry_cid")
        return result


@dataclass(frozen=True, slots=True)
class EqualitySaturationCapability:
    """Availability record for one proof-normalization feature."""

    feature: str
    status: str
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature", _text(self.feature, "feature"))
        object.__setattr__(self, "status", _enum(self.status, FeatureStatus, "status"))
        object.__setattr__(self, "note", _text(self.note, "note", empty=True))

    @property
    def available(self) -> bool:
        return self.status == FeatureStatus.AVAILABLE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EQUALITY_SATURATION_CAPABILITY_SCHEMA,
            "feature": self.feature,
            "status": self.status,
            "note": self.note,
        }


def proof_normalization_capabilities() -> tuple[EqualitySaturationCapability, ...]:
    """Return the closed inventory of available and unavailable features."""

    available = tuple(
        EqualitySaturationCapability(
            feature=feature,
            status=FeatureStatus.AVAILABLE.value,
            note="implemented_in_refactor_proof_normalization_adapter",
        )
        for feature in _AVAILABLE_FEATURES
    )
    unavailable = tuple(
        EqualitySaturationCapability(
            feature=feature,
            status=FeatureStatus.UNAVAILABLE.value,
            note=note,
        )
        for feature, note in _UNAVAILABLE_FEATURES
    )
    return (*available, *unavailable)


def compile_suitable_expression(**fields: Any) -> SuitableExpression:
    return SuitableExpression(**fields)


def compile_equality_rewrite(**fields: Any) -> EqualityRewrite:
    return EqualityRewrite(**fields)


def compile_interpolant(**fields: Any) -> Interpolant:
    return Interpolant(**fields)


def compile_abstraction_refinement(**fields: Any) -> AbstractionRefinement:
    return AbstractionRefinement(**fields)


def compile_boundary_summary(**fields: Any) -> BoundarySummary:
    return BoundarySummary(**fields)


def _coerce_expression(
    item: SuitableExpression | Mapping[str, Any],
    *,
    tree_id: str,
) -> SuitableExpression:
    if isinstance(item, SuitableExpression):
        expression = item
    elif isinstance(item, Mapping) and "expression_cid" in item and "schema" in item:
        _reject_forbidden_bodies(item, "expression")
        expression = SuitableExpression.from_dict(item)
    elif isinstance(item, Mapping):
        _reject_forbidden_bodies(item, "expression")
        expression = SuitableExpression(**_strip_envelope(item, "expression_cid"))
    else:
        raise ProofNormalizationError("expressions items must be objects")
    if expression.tree_id != tree_id:
        raise ProofNormalizationError("expression tree_id does not match")
    return expression


def _coerce_rewrite(
    item: EqualityRewrite | Mapping[str, Any],
    *,
    tree_id: str,
) -> EqualityRewrite:
    if isinstance(item, EqualityRewrite):
        rewrite = item
    elif isinstance(item, Mapping) and "rewrite_cid" in item and "schema" in item:
        _reject_forbidden_bodies(item, "rewrite")
        rewrite = EqualityRewrite.from_dict(item)
    elif isinstance(item, Mapping):
        _reject_forbidden_bodies(item, "rewrite")
        rewrite = EqualityRewrite(**_strip_envelope(item, "rewrite_cid"))
    else:
        raise ProofNormalizationError("rewrites items must be objects")
    if rewrite.tree_id != tree_id:
        raise ProofNormalizationError("rewrite tree_id does not match")
    return rewrite


def _coerce_interpolant(
    item: Interpolant | Mapping[str, Any],
    *,
    tree_id: str,
) -> Interpolant:
    if isinstance(item, Interpolant):
        interpolant = item
    elif isinstance(item, Mapping) and "interpolant_cid" in item and "schema" in item:
        _reject_forbidden_bodies(item, "interpolant")
        interpolant = Interpolant.from_dict(item)
    elif isinstance(item, Mapping):
        _reject_forbidden_bodies(item, "interpolant")
        interpolant = Interpolant(**_strip_envelope(item, "interpolant_cid"))
    else:
        raise ProofNormalizationError("interpolants items must be objects")
    if interpolant.tree_id != tree_id:
        raise ProofNormalizationError("interpolant tree_id does not match")
    return interpolant


def _coerce_refinement(
    item: AbstractionRefinement | Mapping[str, Any],
    *,
    tree_id: str,
) -> AbstractionRefinement:
    if isinstance(item, AbstractionRefinement):
        refinement = item
    elif isinstance(item, Mapping) and "refinement_cid" in item and "schema" in item:
        _reject_forbidden_bodies(item, "refinement")
        refinement = AbstractionRefinement.from_dict(item)
    elif isinstance(item, Mapping):
        _reject_forbidden_bodies(item, "refinement")
        refinement = AbstractionRefinement(**_strip_envelope(item, "refinement_cid"))
    else:
        raise ProofNormalizationError("refinements items must be objects")
    if refinement.tree_id != tree_id:
        raise ProofNormalizationError("refinement tree_id does not match")
    return refinement


def _ingest_proof_reconstructions(
    proof_receipt: Mapping[str, Any] | Any | None,
    *,
    tree_id: str,
) -> tuple[frozenset[str], bool]:
    if proof_receipt in (None, (), {}):
        return frozenset(), False
    payload = _as_mapping(proof_receipt, "SPAR-030 proof receipt")
    _reject_non_admitting_payload(payload, "SPAR-030 proof receipt")
    receipt_tree = payload.get("tree_id")
    if receipt_tree not in (None, ""):
        _require_tree(receipt_tree, tree_id, "SPAR-030")
    reconstruction_cids: set[str] = set()
    stale = False
    for item in _mapping_sequence(payload.get("reconstructions"), "SPAR-030 reconstructions"):
        if item.get("kernel_checked") is not True:
            continue
        cid = item.get("reconstruction_cid") or item.get("artifact_cid")
        if cid in (None, ""):
            raise ProofNormalizationError("SPAR-030 reconstruction_cid is required")
        reconstruction_cids.add(_cid(cid, "SPAR-030 reconstruction_cid"))
    for item in _mapping_sequence(payload.get("steps"), "SPAR-030 steps"):
        kind = item.get("kind")
        if kind == "stale":
            stale = True
        if kind == "reconstructed_proof" and item.get("kernel_checked") is True:
            cid = item.get("reconstruction_cid") or item.get("artifact_cid")
            if cid not in (None, ""):
                reconstruction_cids.add(_cid(cid, "SPAR-030 reconstruction_cid"))
    return frozenset(reconstruction_cids), stale


def _ingest_proof_replays(
    proof_receipt: Mapping[str, Any] | Any | None,
    *,
    tree_id: str,
) -> frozenset[str]:
    if proof_receipt in (None, (), {}):
        return frozenset()
    payload = _as_mapping(proof_receipt, "SPAR-030 proof receipt")
    replay_cids: set[str] = set()
    for item in _mapping_sequence(payload.get("replays"), "SPAR-030 replays"):
        if item.get("replayed") is not True:
            continue
        cid = item.get("replay_cid") or item.get("artifact_cid") or item.get("source_cid")
        if cid in (None, ""):
            raise ProofNormalizationError("SPAR-030 replay source_cid is required")
        replay_cids.add(_cid(cid, "SPAR-030 replay source_cid"))
    return frozenset(replay_cids)


def _ingest_differential_refinements(
    differential_receipt: Mapping[str, Any] | Any | None,
    *,
    tree_id: str,
) -> tuple[AbstractionRefinement, ...]:
    if differential_receipt in (None, (), {}):
        return ()
    payload = _as_mapping(differential_receipt, "SPAR-028 differential receipt")
    _reject_non_admitting_payload(payload, "SPAR-028 differential receipt")
    receipt_tree = payload.get("tree_id")
    if receipt_tree not in (None, ""):
        _require_tree(receipt_tree, tree_id, "SPAR-028")
    produced: list[AbstractionRefinement] = []
    comparisons = payload.get("comparisons") or payload.get("mismatches") or ()
    for item in _mapping_sequence(comparisons, "SPAR-028 comparisons"):
        independently_observed = item.get("independently_observed") is True
        replayed = item.get("replayed") is True
        if item.get("abstraction_gap") is True and not (independently_observed or replayed):
            raise ProofNormalizationError(
                "raw countermodels cannot refine abstractions until replay"
            )
        if item.get("mismatch") is True and not (independently_observed or replayed):
            raise ProofNormalizationError(
                "SPAR-028 mismatches cannot refute until independently observed"
            )
        if item.get("abstraction_gap") is not True:
            continue
        if not (independently_observed or replayed):
            raise ProofNormalizationError(
                "raw countermodels cannot refine abstractions until replay"
            )
        source_cid = item.get("source_cid") or item.get("replay_cid")
        if source_cid in (None, ""):
            raise ProofNormalizationError("SPAR-028 comparison source_cid is required")
        predicate_id = _text(
            item.get("predicate_id") or item.get("mismatch_id") or "pred:diff",
            "predicate_id",
        )
        kind = item.get("kind") or AbstractionKind.BOUNDARY.value
        produced.append(
            AbstractionRefinement(
                predicate_id=predicate_id,
                kind=kind,
                tree_id=tree_id,
                source_cid=_cid(source_cid, "SPAR-028 source_cid"),
                grain=RefinementGrain.REFINED.value,
                distinguishing=True,
                replayed=True,
            )
        )
    return tuple(produced)


def _representative(
    members: Sequence[str],
    *,
    costs: Mapping[str, int],
) -> str:
    return min(members, key=lambda term: (int(costs.get(term, 1)), term))


def saturate_egraph(
    *,
    tree_id: str,
    expressions: Sequence[SuitableExpression],
    rewrites: Sequence[EqualityRewrite],
    max_steps: int = DEFAULT_MAX_STEPS,
    theory_id: str = "",
) -> EGraphSaturation:
    """Saturate a bounded typed e-graph under reviewed oriented rewrites."""

    bound = _int(max_steps, "max_steps", minimum=1, maximum=ABSOLUTE_MAX_STEPS)
    parent: dict[str, str] = {}
    costs: dict[str, int] = {}

    def _add(term: str, cost: int = 1) -> None:
        if term not in parent:
            parent[term] = term
            costs[term] = cost
        else:
            costs[term] = min(costs[term], cost)

    for expression in expressions:
        _add(expression.term_id, expression.cost)
        for arg in expression.arg_term_ids:
            _add(arg)

    for rewrite in rewrites:
        _add(rewrite.lhs_term_id, rewrite.cost)
        _add(rewrite.rhs_term_id, rewrite.cost)

    def find(term: str) -> str:
        while parent[term] != term:
            parent[term] = parent[parent[term]]
            term = parent[term]
        return term

    def union(left: str, right: str) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right
        return True

    steps_used = 0
    timed_out = False
    applied: list[str] = []
    ordered_rewrites = tuple(sorted(rewrites, key=lambda item: item.rule_id))
    for rewrite in ordered_rewrites:
        if steps_used >= bound:
            timed_out = True
            break
        if union(rewrite.lhs_term_id, rewrite.rhs_term_id):
            applied.append(rewrite.rewrite_cid)
            steps_used += 1

    operators = tuple(
        item for item in expressions if item.operator and item.arg_term_ids
    )
    changed = True
    while changed and not timed_out:
        changed = False
        for index, left in enumerate(operators):
            for right in operators[index + 1 :]:
                if steps_used >= bound:
                    timed_out = True
                    break
                if left.operator != right.operator:
                    continue
                if len(left.arg_term_ids) != len(right.arg_term_ids):
                    continue
                if all(
                    find(left_arg) == find(right_arg)
                    for left_arg, right_arg in zip(left.arg_term_ids, right.arg_term_ids)
                ):
                    if union(left.term_id, right.term_id):
                        changed = True
                        steps_used += 1
            if timed_out:
                break

    grouped: dict[str, list[str]] = {}
    for term in parent:
        grouped.setdefault(find(term), []).append(term)
    classes = tuple(
        EGraphClass(
            class_id=f"eclass:{root}",
            member_term_ids=tuple(sorted(members)),
            representative_term_id=_representative(members, costs=costs),
        )
        for root, members in sorted(grouped.items())
    )
    return EGraphSaturation(
        tree_id=tree_id,
        theory_id=theory_id,
        classes=classes,
        rewrite_cids=tuple(sorted(set(applied))),
        steps_used=steps_used,
        timed_out=timed_out,
    )


def interpolate_boundary(
    interpolant: Interpolant | Mapping[str, Any],
    *,
    tree_id: str,
    known_term_ids: Sequence[str],
    reconstruction_cids: Sequence[str] = (),
    replay_cids: Sequence[str] = (),
) -> Interpolant:
    """Admit one independently justified interpolant or fail closed."""

    typed = _coerce_interpolant(interpolant, tree_id=tree_id)
    known = set(known_term_ids)
    missing = [term for term in (*typed.a_term_ids, *typed.b_term_ids) if term not in known]
    if missing:
        raise ProofNormalizationError("interpolant terms must be declared expressions")
    if typed.justification == "reconstructed_proof":
        allowed = set(reconstruction_cids)
        if allowed and typed.justification_cid not in allowed:
            raise ProofNormalizationError("SPAR-030 reconstruction_cid is required")
    elif typed.justification == "replayed_counterexample":
        allowed = set(replay_cids)
        if allowed and typed.justification_cid not in allowed:
            raise ProofNormalizationError("SPAR-030 replay source_cid is required")
    return typed


def refine_abstraction(
    refinement: AbstractionRefinement | Mapping[str, Any],
    *,
    tree_id: str,
) -> AbstractionRefinement:
    """Admit one replayed-counterexample refinement or fail closed."""

    return _coerce_refinement(refinement, tree_id=tree_id)


def _validation_status(value: Any) -> str:
    if value in (None, ""):
        return ValidationReentryStatus.MISSING.value
    text = _text(value, "status")
    if text not in DECLARED_VALIDATION_REENTRY_STATUSES:
        raise ProofNormalizationError(f"unknown SPAR-027 status: {text}")
    return text


def reenter_translation_validation(
    form: NormalizationForm,
    *,
    validate: ValidateFn | None,
    tree_id: str,
    write_paths: Sequence[str],
    validation_commands: Sequence[str],
    packet_cid: str,
    wave_cid: str,
    selection_cid: str,
) -> ValidationReentry:
    """Re-enter SPAR-027. A pass cannot complete or promote the form."""

    if validate is None:
        return ValidationReentry(
            form_cid=form.form_cid,
            tree_id=tree_id,
            status=ValidationReentryStatus.MISSING.value,
            reason_code="validation_reentry_required",
        )
    payload = dict(
        validate(
            form=form.to_dict(),
            tree_id=tree_id,
            write_paths=list(write_paths),
            validation_commands=list(validation_commands),
            packet_cid=packet_cid,
            wave_cid=wave_cid,
            selection_cid=selection_cid,
            network=NETWORK_DENY,
        )
    )
    _reject_forbidden_bodies(payload, "SPAR-027 validation result")
    _reject_non_admitting_payload(payload, "SPAR-027 validation result")
    status = _validation_status(payload.get("status"))
    result_tree = payload.get("tree_id")
    if result_tree not in (None, ""):
        _require_tree(result_tree, tree_id, "SPAR-027")
    reason = _text(
        payload.get("reason_code") or payload.get("terminal_reason") or "",
        "reason_code",
        empty=True,
    )
    if status == ValidationReentryStatus.VALIDATED.value:
        reason = ""
    elif not reason:
        reason = status
    return ValidationReentry(
        form_cid=form.form_cid,
        tree_id=tree_id,
        status=status,
        result_cid=_optional_cid(payload.get("result_cid"), "result_cid"),
        reason_code=reason,
    )


def _status_from_work(
    *,
    forms: Sequence[NormalizationForm],
    interpolants: Sequence[Interpolant],
    refinements: Sequence[AbstractionRefinement],
    timed_out: bool,
    stale: bool,
    reentry: ValidationReentry | None,
    expressions: Sequence[SuitableExpression],
) -> str:
    if stale:
        return NormalizationStatus.STALE.value
    if timed_out:
        return NormalizationStatus.TIMEOUT.value
    if reentry is not None and reentry.status == ValidationReentryStatus.REJECTED.value:
        return NormalizationStatus.REJECTED.value
    if reentry is not None and reentry.status == ValidationReentryStatus.UNSUPPORTED.value:
        return NormalizationStatus.UNSUPPORTED.value
    if reentry is not None and reentry.status == ValidationReentryStatus.INCOMPLETE.value:
        return NormalizationStatus.INCOMPLETE.value
    if forms:
        return NormalizationStatus.NORMALIZED.value
    if interpolants:
        return NormalizationStatus.INTERPOLATED.value
    if refinements:
        return NormalizationStatus.REFINED.value
    if not expressions:
        return NormalizationStatus.UNKNOWN.value
    return NormalizationStatus.UNKNOWN.value


@dataclass(frozen=True, slots=True)
class ProofNormalizationReceipt:
    """Content-addressed SPAR-031 proof-normalization receipt. Nomination-only."""

    tree_id: str
    packet_cid: str
    wave_cid: str
    selection_cid: str
    expressions: Sequence[SuitableExpression | Mapping[str, Any]] = ()
    rewrites: Sequence[EqualityRewrite | Mapping[str, Any]] = ()
    interpolants: Sequence[Interpolant | Mapping[str, Any]] = ()
    refinements: Sequence[AbstractionRefinement | Mapping[str, Any]] = ()
    summaries: Sequence[BoundarySummary | Mapping[str, Any]] = ()
    forms: Sequence[NormalizationForm | Mapping[str, Any]] = ()
    steps: Sequence[NormalizationStep | Mapping[str, Any]] = ()
    saturation: EGraphSaturation | Mapping[str, Any] | None = None
    reentry: ValidationReentry | Mapping[str, Any] | None = None
    write_paths: Sequence[str] = ()
    raw_source_cids: Sequence[str] = ()
    validation_commands: Sequence[str] = ()
    status: NormalizationStatus | str = NormalizationStatus.UNKNOWN
    max_steps: int = DEFAULT_MAX_STEPS
    steps_used: int = 0
    selector_mode: str = REWRITE_SELECTOR_DETERMINISTIC
    network: str = NETWORK_DENY
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    unknown_remains_unknown: bool = True
    claims_general_equivalence: bool = False
    normal_form_cannot_admit_proofs: bool = True
    raw_countermodel_cannot_refute: bool = True
    unvalidated_interpolants_fail_closed: bool = True
    normalized_forms_reenter_validation: bool = True
    forms_remain_unpromoted: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = PROOF_NORMALIZATION_RECEIPT_INTERFACE
    schema: ClassVar[str] = PROOF_NORMALIZATION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "wave_cid",
            "selection_cid",
            "expressions",
            "rewrites",
            "interpolants",
            "refinements",
            "summaries",
            "forms",
            "steps",
            "saturation",
            "reentry",
            "write_paths",
            "raw_source_cids",
            "validation_commands",
            "status",
            "max_steps",
            "steps_used",
            "selector_mode",
            "network",
            "analyzer_id",
            "adapter_is_nomination_only",
            "unknown_remains_unknown",
            "claims_general_equivalence",
            "normal_form_cannot_admit_proofs",
            "raw_countermodel_cannot_refute",
            "unvalidated_interpolants_fail_closed",
            "normalized_forms_reenter_validation",
            "forms_remain_unpromoted",
            "mutated",
            "deterministic",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_create_proof_authority",
            "projection_is_authority",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        tree_id = _tree_id(self.tree_id)
        expressions = tuple(
            _coerce_expression(item, tree_id=tree_id) for item in self.expressions
        )
        rewrites = tuple(_coerce_rewrite(item, tree_id=tree_id) for item in self.rewrites)
        interpolants = tuple(
            _coerce_interpolant(item, tree_id=tree_id) for item in self.interpolants
        )
        refinements = tuple(
            _coerce_refinement(item, tree_id=tree_id) for item in self.refinements
        )
        summaries: list[BoundarySummary] = []
        for item in self.summaries:
            if isinstance(item, BoundarySummary):
                summaries.append(item)
            elif isinstance(item, Mapping) and "summary_cid" in item and "schema" in item:
                summaries.append(BoundarySummary.from_dict(item))
            elif isinstance(item, Mapping):
                summaries.append(BoundarySummary(**_strip_envelope(item, "summary_cid")))
            else:
                raise ProofNormalizationError("summaries items must be objects")
        forms: list[NormalizationForm] = []
        for item in self.forms:
            if isinstance(item, NormalizationForm):
                forms.append(item)
            elif isinstance(item, Mapping) and "form_cid" in item and "schema" in item:
                forms.append(NormalizationForm.from_dict(item))
            elif isinstance(item, Mapping):
                forms.append(NormalizationForm(**_strip_envelope(item, "form_cid")))
            else:
                raise ProofNormalizationError("forms items must be objects")
        steps: list[NormalizationStep] = []
        for item in self.steps:
            if isinstance(item, NormalizationStep):
                steps.append(item)
            elif isinstance(item, Mapping) and "step_cid" in item and "schema" in item:
                steps.append(NormalizationStep.from_dict(item))
            elif isinstance(item, Mapping):
                steps.append(NormalizationStep(**_strip_envelope(item, "step_cid")))
            else:
                raise ProofNormalizationError("steps items must be objects")
        if len(expressions) > MAX_EXPRESSIONS:
            raise ProofNormalizationError("expressions exceed maximum length")
        if len(rewrites) > MAX_REWRITES:
            raise ProofNormalizationError("rewrites exceed maximum length")
        if len(interpolants) > MAX_INTERPOLANTS:
            raise ProofNormalizationError("interpolants exceed maximum length")
        if len(refinements) > MAX_REFINEMENTS:
            raise ProofNormalizationError("refinements exceed maximum length")
        if len(summaries) > MAX_SUMMARIES:
            raise ProofNormalizationError("summaries exceed maximum length")
        if len(forms) > MAX_FORMS:
            raise ProofNormalizationError("forms exceed maximum length")
        if len(steps) > MAX_STEPS:
            raise ProofNormalizationError("steps exceed maximum length")
        saturation: EGraphSaturation | None
        if self.saturation is None:
            saturation = None
        elif isinstance(self.saturation, EGraphSaturation):
            saturation = self.saturation
        elif isinstance(self.saturation, Mapping) and "saturation_cid" in self.saturation:
            saturation = EGraphSaturation.from_dict(self.saturation)
        elif isinstance(self.saturation, Mapping):
            saturation = EGraphSaturation(**_strip_envelope(self.saturation, "saturation_cid"))
        else:
            raise ProofNormalizationError("saturation must be an object")
        reentry: ValidationReentry | None
        if self.reentry is None:
            reentry = None
        elif isinstance(self.reentry, ValidationReentry):
            reentry = self.reentry
        elif isinstance(self.reentry, Mapping) and "reentry_cid" in self.reentry:
            reentry = ValidationReentry.from_dict(self.reentry)
        elif isinstance(self.reentry, Mapping):
            payload = _strip_envelope(self.reentry, "reentry_cid")
            reentry = ValidationReentry(**payload)
        else:
            raise ProofNormalizationError("reentry must be an object")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(self, "selection_cid", _cid(self.selection_cid, "selection_cid"))
        object.__setattr__(self, "expressions", expressions)
        object.__setattr__(self, "rewrites", rewrites)
        object.__setattr__(self, "interpolants", interpolants)
        object.__setattr__(self, "refinements", refinements)
        object.__setattr__(self, "summaries", tuple(summaries))
        object.__setattr__(self, "forms", tuple(forms))
        object.__setattr__(self, "steps", tuple(steps))
        object.__setattr__(self, "saturation", saturation)
        object.__setattr__(self, "reentry", reentry)
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self,
            "raw_source_cids",
            _cids(self.raw_source_cids, "raw_source_cids", required=True),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _commands(self.validation_commands, "validation_commands"),
        )
        object.__setattr__(
            self, "status", _enum(self.status, NormalizationStatus, "status")
        )
        object.__setattr__(
            self,
            "max_steps",
            _int(self.max_steps, "max_steps", minimum=1, maximum=ABSOLUTE_MAX_STEPS),
        )
        object.__setattr__(
            self,
            "steps_used",
            _int(self.steps_used, "steps_used", minimum=0, maximum=ABSOLUTE_MAX_STEPS),
        )
        if self.selector_mode != REWRITE_SELECTOR_DETERMINISTIC:
            raise ProofNormalizationError("rewrite selection must remain deterministic")
        object.__setattr__(self, "selector_mode", REWRITE_SELECTOR_DETERMINISTIC)
        object.__setattr__(self, "network", _network_value(self.network))
        if self.analyzer_id != ANALYZER_ID:
            raise ProofNormalizationError("receipt analyzer_id must remain SPAR-031")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if self.adapter_is_nomination_only is not True:
            raise ProofNormalizationError("adapter must remain nomination_only")
        if self.unknown_remains_unknown is not True:
            raise ProofNormalizationError("unknown must remain unknown")
        if self.claims_general_equivalence is not False:
            raise ProofNormalizationError("general Python equivalence is not claimed")
        if self.normal_form_cannot_admit_proofs is not True:
            raise ProofNormalizationError("normal forms cannot admit proofs")
        if self.raw_countermodel_cannot_refute is not True:
            raise ProofNormalizationError("raw countermodels cannot refute until replay")
        if self.unvalidated_interpolants_fail_closed is not True:
            raise ProofNormalizationError("unvalidated interpolants fail closed")
        if self.normalized_forms_reenter_validation is not True:
            raise ProofNormalizationError("normalized forms must re-enter validation")
        if self.forms_remain_unpromoted is not True:
            raise ProofNormalizationError("normalized forms remain unpromoted")
        if self.mutated is not False:
            raise ProofNormalizationError("adapter cannot mutate")
        if self.deterministic is not True:
            raise ProofNormalizationError("normalization must remain deterministic")
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "unknown_remains_unknown", True)
        object.__setattr__(self, "claims_general_equivalence", False)
        object.__setattr__(self, "normal_form_cannot_admit_proofs", True)
        object.__setattr__(self, "raw_countermodel_cannot_refute", True)
        object.__setattr__(self, "unvalidated_interpolants_fail_closed", True)
        object.__setattr__(self, "normalized_forms_reenter_validation", True)
        object.__setattr__(self, "forms_remain_unpromoted", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)
        if any(item.unpromoted is not True for item in forms):
            raise ProofNormalizationError("normalized forms remain unpromoted")

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
    def can_create_proof_authority(self) -> bool:
        return False

    @property
    def projection_is_authority(self) -> bool:
        return False

    @property
    def typed_terminal(self) -> bool:
        return self.status not in {
            NormalizationStatus.NORMALIZED.value,
            NormalizationStatus.INTERPOLATED.value,
            NormalizationStatus.REFINED.value,
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PROOF_NORMALIZATION_RECEIPT_SCHEMA,
            "interface": PROOF_NORMALIZATION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "wave_cid": self.wave_cid,
            "selection_cid": self.selection_cid,
            "expressions": [item.identity_payload() for item in self.expressions],
            "rewrites": [item.identity_payload() for item in self.rewrites],
            "interpolants": [item.identity_payload() for item in self.interpolants],
            "refinements": [item.identity_payload() for item in self.refinements],
            "summaries": [item.identity_payload() for item in self.summaries],
            "forms": [item.identity_payload() for item in self.forms],
            "steps": [item.identity_payload() for item in self.steps],
            "saturation": None
            if self.saturation is None
            else self.saturation.identity_payload(),
            "reentry": None if self.reentry is None else self.reentry.identity_payload(),
            "write_paths": list(self.write_paths),
            "raw_source_cids": list(self.raw_source_cids),
            "validation_commands": list(self.validation_commands),
            "status": self.status,
            "max_steps": self.max_steps,
            "steps_used": self.steps_used,
            "selector_mode": REWRITE_SELECTOR_DETERMINISTIC,
            "network": NETWORK_DENY,
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "unknown_remains_unknown": True,
            "claims_general_equivalence": False,
            "normal_form_cannot_admit_proofs": True,
            "raw_countermodel_cannot_refute": True,
            "unvalidated_interpolants_fail_closed": True,
            "normalized_forms_reenter_validation": True,
            "forms_remain_unpromoted": True,
            "mutated": False,
            "deterministic": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_create_proof_authority": False,
            "projection_is_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["expressions"] = [item.to_dict() for item in self.expressions]
        payload["rewrites"] = [item.to_dict() for item in self.rewrites]
        payload["interpolants"] = [item.to_dict() for item in self.interpolants]
        payload["refinements"] = [item.to_dict() for item in self.refinements]
        payload["summaries"] = [item.to_dict() for item in self.summaries]
        payload["forms"] = [item.to_dict() for item in self.forms]
        payload["steps"] = [item.to_dict() for item in self.steps]
        payload["saturation"] = (
            None if self.saturation is None else self.saturation.to_dict()
        )
        payload["reentry"] = None if self.reentry is None else self.reentry.to_dict()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProofNormalizationReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != PROOF_NORMALIZATION_RECEIPT_SCHEMA:
            raise ProofNormalizationError("unsupported ProofNormalizationReceipt schema")
        if payload.pop("interface") != PROOF_NORMALIZATION_RECEIPT_INTERFACE:
            raise ProofNormalizationError(
                "unsupported ProofNormalizationReceipt interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ProofNormalizationError("adapter must remain nomination_only")
        if payload.pop("unknown_remains_unknown") is not True:
            raise ProofNormalizationError("unknown must remain unknown")
        if payload.pop("claims_general_equivalence") is not False:
            raise ProofNormalizationError("general Python equivalence is not claimed")
        if payload.pop("normal_form_cannot_admit_proofs") is not True:
            raise ProofNormalizationError("normal forms cannot admit proofs")
        if payload.pop("raw_countermodel_cannot_refute") is not True:
            raise ProofNormalizationError("raw countermodels cannot refute until replay")
        if payload.pop("unvalidated_interpolants_fail_closed") is not True:
            raise ProofNormalizationError("unvalidated interpolants fail closed")
        if payload.pop("normalized_forms_reenter_validation") is not True:
            raise ProofNormalizationError("normalized forms must re-enter validation")
        if payload.pop("forms_remain_unpromoted") is not True:
            raise ProofNormalizationError("normalized forms remain unpromoted")
        if payload.pop("mutated") is not False:
            raise ProofNormalizationError("adapter cannot mutate")
        if payload.pop("deterministic") is not True:
            raise ProofNormalizationError("normalization must remain deterministic")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise ProofNormalizationError("receipt analyzer_id must remain SPAR-031")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


def run_proof_normalization(
    *,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    expressions: Sequence[SuitableExpression | Mapping[str, Any]] = (),
    rewrites: Sequence[EqualityRewrite | Mapping[str, Any]] = (),
    interpolants: Sequence[Interpolant | Mapping[str, Any]] = (),
    refinements: Sequence[AbstractionRefinement | Mapping[str, Any]] = (),
    proof_receipt: Mapping[str, Any] | Any | None = None,
    differential_receipt: Mapping[str, Any] | Any | None = None,
    validate: ValidateFn | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    mutate: bool = False,
    vector_evidence: Any = None,
    network: str = NETWORK_DENY,
    theory_id: str = "",
) -> ProofNormalizationReceipt:
    """Run bounded e-graph normalization, interpolation, and refinement."""

    if mutate is not False:
        raise ProofNormalizationError("adapter cannot mutate")
    _reject_vector_admission(vector_evidence)
    _network_value(network)
    bound = _int(max_steps, "max_steps", minimum=1, maximum=ABSOLUTE_MAX_STEPS)
    (
        tree_id,
        wave_cid,
        selection_cid,
        packet_cid,
        write_paths,
        raw_source_cids,
        validation_commands,
    ) = _bind_predecessors(wave=wave, selection=selection)

    typed_expressions = tuple(
        _coerce_expression(item, tree_id=tree_id) for item in expressions
    )
    if len(typed_expressions) > DEFAULT_MAX_EXPRESSIONS:
        raise ProofNormalizationError("expressions exceed search bound")
    seen_expressions: set[str] = set()
    for item in typed_expressions:
        if item.expression_id in seen_expressions:
            raise ProofNormalizationError("expression_id must be unique")
        seen_expressions.add(item.expression_id)
        if item.source_cid not in raw_source_cids:
            raise ProofNormalizationError(
                "expression source_cid must be a declared raw source"
            )

    typed_rewrites = tuple(_coerce_rewrite(item, tree_id=tree_id) for item in rewrites)
    if len(typed_rewrites) > MAX_REWRITES:
        raise ProofNormalizationError("rewrites exceed maximum length")
    seen_rewrites: set[str] = set()
    for item in typed_rewrites:
        if item.rule_id in seen_rewrites:
            raise ProofNormalizationError("rule_id must be unique")
        seen_rewrites.add(item.rule_id)

    reconstruction_cids, stale = _ingest_proof_reconstructions(
        proof_receipt, tree_id=tree_id
    )
    replay_cids = _ingest_proof_replays(proof_receipt, tree_id=tree_id)
    if reconstruction_cids:
        for item in typed_rewrites:
            if (
                item.justification == "reconstructed_proof"
                and item.reconstruction_cid not in reconstruction_cids
            ):
                raise ProofNormalizationError("SPAR-030 reconstruction_cid is required")

    known_term_ids = tuple(
        dict.fromkeys(
            (
                *(expression.term_id for expression in typed_expressions),
                *(arg for expression in typed_expressions for arg in expression.arg_term_ids),
                *(rewrite.lhs_term_id for rewrite in typed_rewrites),
                *(rewrite.rhs_term_id for rewrite in typed_rewrites),
            )
        )
    )

    typed_interpolants = tuple(
        interpolate_boundary(
            item,
            tree_id=tree_id,
            known_term_ids=known_term_ids,
            reconstruction_cids=tuple(reconstruction_cids),
            replay_cids=tuple(replay_cids),
        )
        for item in interpolants
    )
    if len(typed_interpolants) > MAX_INTERPOLANTS:
        raise ProofNormalizationError("interpolants exceed maximum length")

    typed_refinements = [
        refine_abstraction(item, tree_id=tree_id) for item in refinements
    ]
    typed_refinements.extend(
        _ingest_differential_refinements(differential_receipt, tree_id=tree_id)
    )
    unique_refinements: dict[str, AbstractionRefinement] = {}
    for item in typed_refinements:
        unique_refinements[item.refinement_cid] = item
    typed_refinements = list(unique_refinements.values())
    if len(typed_refinements) > MAX_REFINEMENTS:
        raise ProofNormalizationError("refinements exceed maximum length")

    saturation = saturate_egraph(
        tree_id=tree_id,
        expressions=typed_expressions,
        rewrites=typed_rewrites,
        max_steps=bound,
        theory_id=theory_id,
    )

    forms: list[NormalizationForm] = []
    for eclass in saturation.classes:
        if len(eclass.member_term_ids) < 2:
            continue
        forms.append(
            NormalizationForm(
                form_id=eclass.class_id,
                tree_id=tree_id,
                representative_term_id=eclass.representative_term_id,
                member_term_ids=eclass.member_term_ids,
                write_paths=write_paths,
                source_cids=raw_source_cids,
                unpromoted=True,
            )
        )

    summaries: list[BoundarySummary] = []
    interpolant_cids = tuple(item.interpolant_cid for item in typed_interpolants)
    refinement_cids = tuple(item.refinement_cid for item in typed_refinements)
    for expression in typed_expressions:
        if expression.kind != ExpressionKind.BOUNDARY_SUMMARY.value:
            continue
        if not interpolant_cids and not refinement_cids:
            continue
        summaries.append(
            BoundarySummary(
                summary_id=f"summary:{expression.expression_id}",
                expression_id=expression.expression_id,
                tree_id=tree_id,
                interpolant_cids=interpolant_cids,
                refinement_cids=refinement_cids,
                smaller=True,
            )
        )

    steps: list[NormalizationStep] = []
    step_index = 1
    if saturation.rewrite_cids or saturation.timed_out:
        steps.append(
            NormalizationStep(
                step_index=step_index,
                kind=StepKind.TIMEOUT.value
                if saturation.timed_out
                else StepKind.EQUALITY_SATURATION.value,
                outcome=(
                    NormalizationStatus.TIMEOUT.value
                    if saturation.timed_out
                    else NormalizationStatus.NORMALIZED.value
                ),
                artifact_cid=saturation.saturation_cid,
                reason_code="timeout" if saturation.timed_out else "",
            )
        )
        step_index += 1
    for interpolant in typed_interpolants:
        steps.append(
            NormalizationStep(
                step_index=step_index,
                kind=StepKind.INTERPOLATION.value,
                outcome=NormalizationStatus.INTERPOLATED.value,
                artifact_cid=interpolant.interpolant_cid,
            )
        )
        step_index += 1
    for refinement in typed_refinements:
        steps.append(
            NormalizationStep(
                step_index=step_index,
                kind=StepKind.ABSTRACTION_REFINEMENT.value,
                outcome=NormalizationStatus.REFINED.value,
                artifact_cid=refinement.refinement_cid,
            )
        )
        step_index += 1

    reentry: ValidationReentry | None = None
    if forms:
        reentry = reenter_translation_validation(
            forms[0],
            validate=validate,
            tree_id=tree_id,
            write_paths=write_paths,
            validation_commands=validation_commands,
            packet_cid=packet_cid,
            wave_cid=wave_cid,
            selection_cid=selection_cid,
        )
        steps.append(
            NormalizationStep(
                step_index=step_index,
                kind=StepKind.VALIDATION_REENTRY.value,
                outcome=(
                    NormalizationStatus.NORMALIZED.value
                    if reentry.status == ValidationReentryStatus.VALIDATED.value
                    else NormalizationStatus.REJECTED.value
                    if reentry.status == ValidationReentryStatus.REJECTED.value
                    else NormalizationStatus.UNKNOWN.value
                ),
                artifact_cid=reentry.reentry_cid,
                reason_code=reentry.reason_code,
            )
        )
        step_index += 1
    elif not typed_expressions and not typed_rewrites:
        steps.append(
            NormalizationStep(
                step_index=step_index,
                kind=StepKind.UNKNOWN.value,
                outcome=NormalizationStatus.UNKNOWN.value,
                reason_code="unknown_remains_unknown",
            )
        )

    if stale:
        steps.append(
            NormalizationStep(
                step_index=step_index,
                kind=StepKind.STALE.value,
                outcome=NormalizationStatus.STALE.value,
                reason_code="stale_proof_receipt",
            )
        )

    status = _status_from_work(
        forms=forms,
        interpolants=typed_interpolants,
        refinements=typed_refinements,
        timed_out=saturation.timed_out,
        stale=stale,
        reentry=reentry,
        expressions=typed_expressions,
    )
    return ProofNormalizationReceipt(
        tree_id=tree_id,
        packet_cid=packet_cid,
        wave_cid=wave_cid,
        selection_cid=selection_cid,
        expressions=typed_expressions,
        rewrites=typed_rewrites,
        interpolants=typed_interpolants,
        refinements=tuple(typed_refinements),
        summaries=tuple(summaries),
        forms=tuple(forms),
        steps=tuple(steps),
        saturation=saturation,
        reentry=reentry,
        write_paths=write_paths,
        raw_source_cids=raw_source_cids,
        validation_commands=validation_commands,
        status=status,
        max_steps=bound,
        steps_used=saturation.steps_used,
    )


def dry_run_proof_normalization(
    *,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    expressions: Sequence[SuitableExpression | Mapping[str, Any]] = (),
    rewrites: Sequence[EqualityRewrite | Mapping[str, Any]] = (),
    interpolants: Sequence[Interpolant | Mapping[str, Any]] = (),
    refinements: Sequence[AbstractionRefinement | Mapping[str, Any]] = (),
    proof_receipt: Mapping[str, Any] | Any | None = None,
    differential_receipt: Mapping[str, Any] | Any | None = None,
    validate: ValidateFn | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    vector_evidence: Any = None,
    network: str = NETWORK_DENY,
    theory_id: str = "",
) -> ProofNormalizationReceipt:
    """Deterministic non-mutating proof-backed normalization."""

    result = run_proof_normalization(
        wave=wave,
        selection=selection,
        expressions=expressions,
        rewrites=rewrites,
        interpolants=interpolants,
        refinements=refinements,
        proof_receipt=proof_receipt,
        differential_receipt=differential_receipt,
        validate=validate,
        max_steps=max_steps,
        mutate=False,
        vector_evidence=vector_evidence,
        network=network,
        theory_id=theory_id,
    )
    if result.mutated is not False or result.deterministic is not True:
        raise ProofNormalizationError("dry-run must remain deterministic and non-mutating")
    return result


class RefactorProofNormalizationAdapter:
    """SPAR-031 e-graph / interpolation / refinement adapter. Nomination-only."""

    interface: ClassVar[str] = REFACTOR_PROOF_NORMALIZATION_ADAPTER_INTERFACE
    schema: ClassVar[str] = REFACTOR_PROOF_NORMALIZATION_ADAPTER_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID
    receipt_interface: ClassVar[str] = PROOF_NORMALIZATION_RECEIPT_INTERFACE

    def compile_expression(self, **fields: Any) -> SuitableExpression:
        return compile_suitable_expression(**fields)

    def compile_rewrite(self, **fields: Any) -> EqualityRewrite:
        return compile_equality_rewrite(**fields)

    def saturate(
        self,
        *,
        tree_id: str,
        expressions: Sequence[SuitableExpression],
        rewrites: Sequence[EqualityRewrite],
        max_steps: int = DEFAULT_MAX_STEPS,
        theory_id: str = "",
    ) -> EGraphSaturation:
        return saturate_egraph(
            tree_id=tree_id,
            expressions=expressions,
            rewrites=rewrites,
            max_steps=max_steps,
            theory_id=theory_id,
        )

    def interpolate(
        self,
        interpolant: Interpolant | Mapping[str, Any],
        *,
        tree_id: str,
        known_term_ids: Sequence[str],
        reconstruction_cids: Sequence[str] = (),
        replay_cids: Sequence[str] = (),
    ) -> Interpolant:
        return interpolate_boundary(
            interpolant,
            tree_id=tree_id,
            known_term_ids=known_term_ids,
            reconstruction_cids=reconstruction_cids,
            replay_cids=replay_cids,
        )

    def refine(
        self,
        refinement: AbstractionRefinement | Mapping[str, Any],
        *,
        tree_id: str,
    ) -> AbstractionRefinement:
        return refine_abstraction(refinement, tree_id=tree_id)

    def normalize(
        self,
        *,
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        expressions: Sequence[SuitableExpression | Mapping[str, Any]] = (),
        rewrites: Sequence[EqualityRewrite | Mapping[str, Any]] = (),
        interpolants: Sequence[Interpolant | Mapping[str, Any]] = (),
        refinements: Sequence[AbstractionRefinement | Mapping[str, Any]] = (),
        proof_receipt: Mapping[str, Any] | Any | None = None,
        differential_receipt: Mapping[str, Any] | Any | None = None,
        validate: ValidateFn | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        vector_evidence: Any = None,
        theory_id: str = "",
    ) -> ProofNormalizationReceipt:
        return run_proof_normalization(
            wave=wave,
            selection=selection,
            expressions=expressions,
            rewrites=rewrites,
            interpolants=interpolants,
            refinements=refinements,
            proof_receipt=proof_receipt,
            differential_receipt=differential_receipt,
            validate=validate,
            max_steps=max_steps,
            vector_evidence=vector_evidence,
            theory_id=theory_id,
        )

    def dry_run(
        self,
        *,
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        expressions: Sequence[SuitableExpression | Mapping[str, Any]] = (),
        rewrites: Sequence[EqualityRewrite | Mapping[str, Any]] = (),
        interpolants: Sequence[Interpolant | Mapping[str, Any]] = (),
        refinements: Sequence[AbstractionRefinement | Mapping[str, Any]] = (),
        proof_receipt: Mapping[str, Any] | Any | None = None,
        differential_receipt: Mapping[str, Any] | Any | None = None,
        validate: ValidateFn | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        vector_evidence: Any = None,
        theory_id: str = "",
    ) -> ProofNormalizationReceipt:
        return dry_run_proof_normalization(
            wave=wave,
            selection=selection,
            expressions=expressions,
            rewrites=rewrites,
            interpolants=interpolants,
            refinements=refinements,
            proof_receipt=proof_receipt,
            differential_receipt=differential_receipt,
            validate=validate,
            max_steps=max_steps,
            vector_evidence=vector_evidence,
            theory_id=theory_id,
        )


def encode_canonical_receipt(receipt: ProofNormalizationReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> ProofNormalizationReceipt:
    return ProofNormalizationReceipt.from_dict(payload)


def encode_canonical_expression(expression: SuitableExpression) -> dict[str, Any]:
    return expression.to_dict()


def decode_canonical_expression(payload: Mapping[str, Any]) -> SuitableExpression:
    return SuitableExpression.from_dict(payload)


def encode_canonical_rewrite(rewrite: EqualityRewrite) -> dict[str, Any]:
    return rewrite.to_dict()


def decode_canonical_rewrite(payload: Mapping[str, Any]) -> EqualityRewrite:
    return EqualityRewrite.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise ProofNormalizationError(
            f"proof normalization must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_IS_NOMINATION_ONLY",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "ABSTRACTION_REFINEMENT_INTERFACE",
    "BOUNDARY_SUMMARY_INTERFACE",
    "DECLARED_ABSTRACTION_KINDS",
    "DECLARED_EXPRESSION_KINDS",
    "DECLARED_FEATURE_STATUSES",
    "DECLARED_NORMALIZATION_STATUSES",
    "DECLARED_REFINEMENT_GRAINS",
    "DECLARED_STEP_KINDS",
    "DECLARED_VALIDATION_REENTRY_STATUSES",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EGRAPH_SATURATION_INTERFACE",
    "EQUALITY_REWRITE_INTERFACE",
    "FORBIDDEN_NORMALIZATION_NAMES",
    "FORMS_REMAIN_UNPROMOTED",
    "GENERAL_PYTHON_EQUIVALENCE_CLAIMED",
    "GOAL_ID",
    "GUESSED_AXIOMS_REJECTED",
    "IDENTITY_EXCLUDED_FIELDS",
    "IMPLICIT_INSTALL_FORBIDDEN",
    "IMPLICIT_NETWORK_FORBIDDEN",
    "INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS",
    "INTERPOLANT_INTERFACE",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "NORMALIZATION_FORM_INTERFACE",
    "NORMALIZATION_STEP_INTERFACE",
    "NORMALIZED_FORMS_REENTER_VALIDATION",
    "NORMAL_FORM_CANNOT_ADMIT_PROOFS",
    "PROGRAM",
    "PROOF_NORMALIZATION_CAN_AUTHORIZE_COMPLETION",
    "PROOF_NORMALIZATION_CAN_AUTHORIZE_TRANSITION",
    "PROOF_NORMALIZATION_CAN_CREATE_AUTHORITY",
    "PROOF_NORMALIZATION_CAN_CREATE_PROOF_AUTHORITY",
    "PROOF_NORMALIZATION_CONTRACT_VERSION",
    "PROOF_NORMALIZATION_RECEIPT_INTERFACE",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_COUNTERMODEL_CANNOT_REFUTE",
    "RAW_SOURCE_REQUIRED",
    "REFACTOR_PROOF_NORMALIZATION_ADAPTER_INTERFACE",
    "REWRITE_SELECTOR_DETERMINISTIC",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "SUITABLE_EXPRESSION_INTERFACE",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TEST_PASS_IS_NOT_PROOF",
    "UNKNOWN_REMAINS_UNKNOWN",
    "UNVALIDATED_INTERPOLANTS_FAIL_CLOSED",
    "VALIDATION_REENTRY_INTERFACE",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "AbstractionKind",
    "AbstractionRefinement",
    "BoundarySummary",
    "EGraphClass",
    "EGraphSaturation",
    "EqualityRewrite",
    "EqualitySaturationCapability",
    "ExpressionKind",
    "FeatureStatus",
    "Interpolant",
    "NormalizationForm",
    "NormalizationStatus",
    "NormalizationStep",
    "ProofNormalizationError",
    "ProofNormalizationReceipt",
    "RefactorProofNormalizationAdapter",
    "RefinementGrain",
    "StepKind",
    "SuitableExpression",
    "ValidationReentry",
    "ValidationReentryStatus",
    "assert_not_competing_capsule_family",
    "compile_abstraction_refinement",
    "compile_boundary_summary",
    "compile_equality_rewrite",
    "compile_interpolant",
    "compile_suitable_expression",
    "decode_canonical_expression",
    "decode_canonical_receipt",
    "decode_canonical_rewrite",
    "dry_run_proof_normalization",
    "encode_canonical_expression",
    "encode_canonical_receipt",
    "encode_canonical_rewrite",
    "interpolate_boundary",
    "proof_normalization_capabilities",
    "proof_normalization_cid_profile",
    "proof_normalization_descriptor",
    "provider_free_exports",
    "reenter_translation_validation",
    "refine_abstraction",
    "run_proof_normalization",
    "saturate_egraph",
]
