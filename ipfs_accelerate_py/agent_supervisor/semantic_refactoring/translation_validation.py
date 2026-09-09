"""SPAR-027 translation-validation contracts and orchestrator.

This module extends current supervisor validation orchestration with
``TranslationValidationRequest@1``, ``TranslationValidationResult@1``, and
``RefactorEquivalenceClaim@1``.  It consumes SPAR-025 extraction-wave
receipts and SPAR-026 selection mappings, then validates one concrete
original/candidate extraction ``P -> P'`` across syntax, imports, APIs,
types/effects, contracts, tests, proofs, traces, state, compatibility,
and resources.

Evidence classes remain separate.  A test proves only tested executions;
a reconstructed proof proves only its encoded model; a trace proves only
observations.  Vector, model, and heuristic evidence cannot admit
equivalence.  General Python equivalence is not claimed.  Procedure-compiler
tool translation validation is not this extraction receipt.

The orchestrator is nomination-only.  It cannot authorize a transition,
completion, or competing authority.  Unsupported required behavior is a
typed terminal, never success.  Observational metadata is excluded from
identity.  Dry-run is deterministic and never mutates.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)
from .selection_adapter import FALLBACK_NONE


TASK_ID: Final[str] = "SPAR-027"
GOAL_ID: Final[str] = "SPAR-G051"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "validation orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.translation_validation@1"
)

TRANSLATION_VALIDATION_REQUEST_INTERFACE: Final[str] = (
    "TranslationValidationRequest@1"
)
TRANSLATION_VALIDATION_RESULT_INTERFACE: Final[str] = (
    "TranslationValidationResult@1"
)
REFACTOR_EQUIVALENCE_CLAIM_INTERFACE: Final[str] = "RefactorEquivalenceClaim@1"
TRANSLATION_VALIDATION_ORCHESTRATOR_INTERFACE: Final[str] = (
    "TranslationValidationOrchestrator@1"
)
OBSERVATION_PROFILE_INTERFACE: Final[str] = "ObservationProfile@1"
DIMENSION_VERDICT_INTERFACE: Final[str] = "DimensionVerdict@1"

TRANSLATION_VALIDATION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/translation-validation-request@1"
)
TRANSLATION_VALIDATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/translation-validation-result@1"
)
REFACTOR_EQUIVALENCE_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-equivalence-claim@1"
)
TRANSLATION_VALIDATION_ORCHESTRATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/translation-validation-orchestrator@1"
)
OBSERVATION_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observation-profile@1"
)
DIMENSION_VERDICT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dimension-verdict@1"
)

VALIDATION_CONTRACT_VERSION: Final[str] = "1"

VALIDATION_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
VALIDATION_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
VALIDATION_CAN_CREATE_AUTHORITY: Final[bool] = False
VALIDATION_CLAIMS_GENERAL_PYTHON_EQUIVALENCE: Final[bool] = False
VALIDATION_COLLAPSES_EVIDENCE_CLASSES: Final[bool] = False
PROCEDURE_COMPILER_IS_NOT_EXTRACTION_RECEIPT: Final[bool] = True
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
TEST_PASS_IS_NOT_PROOF: Final[bool] = True
RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT: Final[bool] = True
PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
ORCHESTRATOR_IS_NOMINATION_ONLY: Final[bool] = True
CLAIM_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_DIMENSIONS: Final[int] = 64

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
    "projection_is_authority",
)

FORBIDDEN_EQUIVALENCE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "prove_python_equivalence",
        "collapse_evidence",
        "admit_from_vectors",
        "promote_test_to_proof",
    }
)


class TranslationValidationError(ValueError):
    """Fail-closed violation of a SPAR-027 translation-validation contract."""


class EvidenceClass(str, Enum):
    EXACT_STATIC_FACT = "exact_static_fact"
    MAY_FACT = "may_fact"
    RUNTIME_OBSERVATION = "runtime_observation"
    SPECIFICATION = "specification"
    TEST = "test"
    PROOF_CANDIDATE = "proof_candidate"
    RECONSTRUCTED_PROOF = "reconstructed_proof"
    COUNTERMODEL = "countermodel"
    REPLAYED_COUNTEREXAMPLE = "replayed_counterexample"
    VECTOR_CANDIDATE = "vector_candidate"
    MODEL_HYPOTHESIS = "model_hypothesis"
    DECISION = "decision"
    ACCEPTED_TRANSITION = "accepted_transition"


class ValidationDimension(str, Enum):
    SYNTAX = "syntax"
    IMPORTS = "imports"
    APIS = "apis"
    TYPES_EFFECTS = "types_effects"
    CONTRACTS = "contracts"
    TESTS = "tests"
    PROOFS = "proofs"
    TRACES = "traces"
    STATE = "state"
    COMPATIBILITY = "compatibility"
    RESOURCES = "resources"


class DimensionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    NOT_IN_PROFILE = "not_in_profile"


class ValidationStatus(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"


DECLARED_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    item.value for item in EvidenceClass
)
DECLARED_DIMENSIONS: Final[frozenset[str]] = frozenset(
    item.value for item in ValidationDimension
)
DECLARED_DIMENSION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in DimensionStatus
)
DECLARED_VALIDATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in ValidationStatus
)

DEFAULT_REQUIRED_DIMENSIONS: Final[tuple[str, ...]] = tuple(
    item.value for item in ValidationDimension
)

DIMENSION_ADMITTED_EVIDENCE: Final[Mapping[str, frozenset[str]]] = {
    ValidationDimension.SYNTAX.value: frozenset(
        {EvidenceClass.EXACT_STATIC_FACT.value}
    ),
    ValidationDimension.IMPORTS.value: frozenset(
        {EvidenceClass.EXACT_STATIC_FACT.value}
    ),
    ValidationDimension.APIS.value: frozenset(
        {EvidenceClass.EXACT_STATIC_FACT.value, EvidenceClass.SPECIFICATION.value}
    ),
    ValidationDimension.TYPES_EFFECTS.value: frozenset(
        {EvidenceClass.EXACT_STATIC_FACT.value, EvidenceClass.MAY_FACT.value}
    ),
    ValidationDimension.CONTRACTS.value: frozenset(
        {EvidenceClass.SPECIFICATION.value, EvidenceClass.EXACT_STATIC_FACT.value}
    ),
    ValidationDimension.TESTS.value: frozenset({EvidenceClass.TEST.value}),
    ValidationDimension.PROOFS.value: frozenset(
        {EvidenceClass.RECONSTRUCTED_PROOF.value}
    ),
    ValidationDimension.TRACES.value: frozenset(
        {EvidenceClass.RUNTIME_OBSERVATION.value}
    ),
    ValidationDimension.STATE.value: frozenset(
        {EvidenceClass.EXACT_STATIC_FACT.value, EvidenceClass.RUNTIME_OBSERVATION.value}
    ),
    ValidationDimension.COMPATIBILITY.value: frozenset(
        {
            EvidenceClass.SPECIFICATION.value,
            EvidenceClass.TEST.value,
            EvidenceClass.EXACT_STATIC_FACT.value,
        }
    ),
    ValidationDimension.RESOURCES.value: frozenset(
        {EvidenceClass.EXACT_STATIC_FACT.value, EvidenceClass.RUNTIME_OBSERVATION.value}
    ),
}

NEGATIVE_PROOF_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        EvidenceClass.COUNTERMODEL.value,
        EvidenceClass.REPLAYED_COUNTEREXAMPLE.value,
    }
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise TranslationValidationError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise TranslationValidationError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise TranslationValidationError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise TranslationValidationError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise TranslationValidationError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise TranslationValidationError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TranslationValidationError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise TranslationValidationError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise TranslationValidationError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise TranslationValidationError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise TranslationValidationError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise TranslationValidationError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise TranslationValidationError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise TranslationValidationError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise TranslationValidationError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TranslationValidationError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise TranslationValidationError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise TranslationValidationError(f"{name} must not contain duplicates")
    return ordered


def _ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TranslationValidationError(f"{name} must be a list")
    ordered = tuple(_text(item, name) for item in values)
    if len(ordered) > limit:
        raise TranslationValidationError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise TranslationValidationError(f"{name} must not contain duplicates")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise TranslationValidationError(f"{name} exceeds path bound")
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
        raise TranslationValidationError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise TranslationValidationError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TranslationValidationError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise TranslationValidationError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise TranslationValidationError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TranslationValidationError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise TranslationValidationError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise TranslationValidationError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise TranslationValidationError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise TranslationValidationError(f"missing {name}")
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        _reject_excluded(value, name)
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping) and not isinstance(
            payload, (str, bytes, bytearray)
        ):
            _reject_excluded(payload, name)
            return dict(payload)
    raise TranslationValidationError(f"{name} must be a mapping")


def _nested_mapping(value: Any, name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TranslationValidationError(f"{name} must be an object")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TranslationValidationError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping) and not isinstance(item, (str, bytes, bytearray)):
            items.append(dict(item))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                items.append(dict(payload))
                continue
        raise TranslationValidationError(f"{name} items must be objects")
    if len(items) > MAX_MEMBERS:
        raise TranslationValidationError(f"{name} exceeds maximum length")
    return tuple(items)


def _attr_or_key(payload: Mapping[str, Any] | Any, *names: str) -> Any:
    if isinstance(payload, Mapping):
        for name in names:
            if name in payload:
                return payload[name]
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
    return None


def translation_validation_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def translation_validation_descriptor() -> dict[str, Any]:
    return {
        "interface": TRANSLATION_VALIDATION_ORCHESTRATOR_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "raw_source_required": True,
        "nomination_only": True,
        "claims_general_python_equivalence": False,
        "collapses_evidence_classes": False,
        "procedure_compiler_is_not_extraction_receipt": True,
        "forbids": tuple(sorted(FORBIDDEN_EQUIVALENCE_NAMES)),
    }


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise TranslationValidationError(f"{name} cannot claim {flag}")


def _dimension_value(value: Any, name: str = "dimension") -> str:
    if isinstance(value, ValidationDimension):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_DIMENSIONS:
        raise TranslationValidationError(f"unsupported {name} {text!r}")
    return text


def _evidence_class_value(value: Any, name: str = "evidence_class") -> str:
    if isinstance(value, EvidenceClass):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_EVIDENCE_CLASSES:
        raise TranslationValidationError(f"unsupported {name} {text!r}")
    return text


def _status_value(value: Any, name: str = "status") -> str:
    if isinstance(value, DimensionStatus):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_DIMENSION_STATUSES:
        raise TranslationValidationError(f"unsupported {name} {text!r}")
    return text


def _validation_status_value(value: Any, name: str = "status") -> str:
    if isinstance(value, ValidationStatus):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_VALIDATION_STATUSES:
        raise TranslationValidationError(f"unsupported {name} {text!r}")
    return text


def _dimensions(values: Any, name: str) -> tuple[str, ...]:
    ordered = _ordered_text(values, name, limit=MAX_DIMENSIONS)
    for item in ordered:
        if item not in DECLARED_DIMENSIONS:
            raise TranslationValidationError(f"unsupported {name} {item!r}")
    return ordered


def _cids(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TranslationValidationError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise TranslationValidationError(f"{name} must not be empty")
    if len(ordered) != len(set(ordered)):
        raise TranslationValidationError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise TranslationValidationError(f"{name} exceeds maximum length")
    return ordered


def _require_tree(actual: Any, expected: str, name: str) -> str:
    tree = _tree_id(actual)
    if tree != expected:
        raise TranslationValidationError(f"{name} tree_id does not match packet tree_id")
    return tree


def admitted_evidence_classes(dimension: str) -> frozenset[str]:
    key = _dimension_value(dimension)
    return DIMENSION_ADMITTED_EVIDENCE[key]


def assert_evidence_class_admitted(dimension: str, evidence_class: str) -> None:
    dim = _dimension_value(dimension)
    evidence = _evidence_class_value(evidence_class)
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise TranslationValidationError(
            "vector, model, or heuristic evidence cannot admit translation validation"
        )
    if evidence == EvidenceClass.PROOF_CANDIDATE.value:
        raise TranslationValidationError("proof_candidate cannot admit proofs")
    if evidence == EvidenceClass.ACCEPTED_TRANSITION.value:
        raise TranslationValidationError(
            "accepted_transition cannot admit a translation-validation dimension"
        )
    if dim == ValidationDimension.PROOFS.value and evidence in NEGATIVE_PROOF_EVIDENCE:
        return
    if evidence not in admitted_evidence_classes(dim):
        raise TranslationValidationError(
            f"{evidence} cannot admit {dim}; evidence classes must not collapse"
        )


def assert_evidence_classes_not_collapsed(
    verdicts: Sequence["DimensionVerdict"] | Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed when one evidence class or CID covers disjoint dimensions."""

    passed: list[tuple[str, str, str]] = []
    for item in verdicts:
        if isinstance(item, DimensionVerdict):
            dimension = item.dimension
            evidence = item.evidence_class
            cid = item.evidence_cid
            status = item.status
        else:
            dimension = _dimension_value(item.get("dimension"))
            evidence = _evidence_class_value(item.get("evidence_class"))
            cid = _optional_cid(item.get("evidence_cid"), "evidence_cid")
            status = _status_value(item.get("status"))
        if status != DimensionStatus.PASS.value:
            continue
        assert_evidence_class_admitted(dimension, evidence)
        passed.append((dimension, evidence, cid))

    by_cid: dict[str, list[tuple[str, str]]] = {}
    for dimension, evidence, cid in passed:
        if not cid:
            continue
        by_cid.setdefault(cid, []).append((dimension, evidence))
    for cid, items in by_cid.items():
        dimensions = {dimension for dimension, _evidence in items}
        if len(dimensions) <= 1:
            continue
        admitted = [admitted_evidence_classes(dimension) for dimension, _evidence in items]
        shared = admitted[0]
        for item in admitted[1:]:
            shared = shared & item
        if not shared:
            raise TranslationValidationError(
                "evidence classes must not collapse across disjoint dimensions"
            )


def _packet_cid(packet: Mapping[str, Any]) -> str:
    claimed = packet.get("packet_cid")
    if claimed in (None, ""):
        raise TranslationValidationError("SPAR-019 packet_cid is required")
    return _cid(claimed, "packet_cid")


def _packet_write_paths(packet: Mapping[str, Any]) -> tuple[str, ...]:
    if "write_paths" in packet:
        return _exact_paths(packet["write_paths"], "write_paths")
    scope = _nested_mapping(packet.get("effect_scope"), "effect_scope")
    if "write_paths" in scope:
        return _exact_paths(scope["write_paths"], "write_paths")
    raise TranslationValidationError("SPAR-019 packet write_paths are required")


def _packet_source_cids(packet: Mapping[str, Any]) -> tuple[str, ...]:
    preimage = _nested_mapping(packet.get("preimage"), "preimage")
    sources = preimage.get("source_cids", packet.get("source_cids"))
    if sources in (None, (), []) or not sources:
        raise TranslationValidationError(
            "raw source required: SPAR-019 preimage source_cids"
        )
    ordered = _cids(list(sources), "source_cids")
    if not ordered:
        raise TranslationValidationError(
            "raw source required: SPAR-019 preimage source_cids"
        )
    return ordered


def _packet_validation_commands(packet: Mapping[str, Any]) -> tuple[str, ...]:
    commands = packet.get("validation_commands")
    if commands in (None, ()):
        raise TranslationValidationError("SPAR-019 validation_commands are required")
    return _commands(commands)


def _wave_receipt_cid(wave: Mapping[str, Any]) -> str:
    claimed = wave.get("receipt_cid") or wave.get("wave_receipt_cid")
    if claimed in (None, ""):
        raise TranslationValidationError("SPAR-025 receipt_cid is required")
    return _cid(claimed, "wave_receipt_cid")


def _wave_packet_cids(wave: Mapping[str, Any]) -> tuple[str, ...]:
    packets = wave.get("packet_cids")
    if packets in (None, ()):
        raise TranslationValidationError("SPAR-025 packet_cids are required")
    return _cids(list(packets), "packet_cids")


def _selection_cid(selection: Mapping[str, Any]) -> str:
    claimed = (
        selection.get("validation_selection_cid")
        or selection.get("producer_selection_cid")
        or selection.get("selection_cid")
    )
    if claimed in (None, ""):
        raise TranslationValidationError("SPAR-026 selection_cid is required")
    return _cid(claimed, "selection_cid")


def _full_suite_required(selection: Mapping[str, Any]) -> bool:
    claimed = selection.get("full_suite_required")
    if type(claimed) is bool:
        return claimed
    fallback = selection.get("effective_fallback")
    if fallback in (None, "", FALLBACK_NONE):
        return False
    return True


def _reject_vector_admission(vector_evidence: Any) -> None:
    if vector_evidence in (None, (), {}):
        return
    payload = (
        _as_mapping(vector_evidence, "vector_evidence")
        if not isinstance(vector_evidence, Sequence)
        else None
    )
    items: tuple[Mapping[str, Any], ...]
    if payload is not None:
        items = (payload,)
    else:
        items = _mapping_sequence(vector_evidence, "vector_evidence")
    for item in items:
        evidence = str(item.get("evidence_class") or item.get("kind") or "")
        if item.get("suppress_raw_source") is True or item.get("skip_raw_source") is True:
            raise TranslationValidationError(
                "vectors/projections cannot suppress raw-source fallback"
            )
        if (
            item.get("admit_equivalence") is True
            or item.get("collapse_evidence_classes") is True
            or item.get("claims_general_python_equivalence") is True
        ):
            raise TranslationValidationError(
                "vector, model, or heuristic evidence cannot admit equivalence"
            )
        if evidence in _NON_ADMITTING_EVIDENCE and (
            item.get("admits_validation") is True or item.get("admits_equivalence") is True
        ):
            raise TranslationValidationError(
                "vector, model, or heuristic evidence cannot admit translation validation"
            )


def _reject_non_admitting_payload(payload: Mapping[str, Any], name: str) -> None:
    evidence = str(payload.get("evidence_class") or "")
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise TranslationValidationError(
            f"vector, model, or heuristic evidence cannot admit {name}"
        )
    if payload.get("claims_general_python_equivalence") is True:
        raise TranslationValidationError("general Python equivalence is not claimed")
    if payload.get("collapse_evidence_classes") is True:
        raise TranslationValidationError("evidence classes must not collapse")


def _parse_sources(sources: Mapping[str, str], name: str) -> None:
    if not isinstance(sources, Mapping) or isinstance(sources, (str, bytes, bytearray)):
        raise TranslationValidationError(f"{name} must be a mapping of path to source")
    if not sources:
        raise TranslationValidationError(f"{name} must not be empty")
    for path, text in sources.items():
        _exact_path(path, name)
        if type(text) is not str:
            raise TranslationValidationError(f"{name} values must be strings")
        try:
            ast.parse(text)
        except SyntaxError as exc:
            raise TranslationValidationError(f"{name} failed to parse") from exc


def _derive_syntax_evidence(
    *,
    original_source_cids: Sequence[str],
    candidate_source_cids: Sequence[str],
    original_sources: Mapping[str, str] | None,
    candidate_sources: Mapping[str, str] | None,
) -> dict[str, Any] | None:
    if original_sources is None and candidate_sources is None:
        return None
    if original_sources is None or candidate_sources is None:
        raise TranslationValidationError(
            "syntax derivation requires original and candidate raw sources"
        )
    try:
        _parse_sources(original_sources, "original_sources")
        _parse_sources(candidate_sources, "candidate_sources")
        status = DimensionStatus.PASS.value
    except TranslationValidationError as exc:
        if "failed to parse" not in str(exc):
            raise
        status = DimensionStatus.FAIL.value
    evidence_cid = cid_for_dag_json(
        {
            "dimension": ValidationDimension.SYNTAX.value,
            "evidence_class": EvidenceClass.EXACT_STATIC_FACT.value,
            "original_source_cids": list(original_source_cids),
            "candidate_source_cids": list(candidate_source_cids),
            "parsed": status == DimensionStatus.PASS.value,
        }
    )
    return {
        "dimension": ValidationDimension.SYNTAX.value,
        "evidence_class": EvidenceClass.EXACT_STATIC_FACT.value,
        "evidence_cid": evidence_cid,
        "status": status,
        "full_suite_executed": False,
    }


@dataclass(frozen=True, slots=True)
class ObservationProfile:
    """Declared observation bound for one P -> P' validation."""

    required_dimensions: Sequence[str] = DEFAULT_REQUIRED_DIMENSIONS
    optional_dimensions: Sequence[str] = ()
    observation_bound_cid: str = ""
    claims_general_python_equivalence: bool = False

    interface: ClassVar[str] = OBSERVATION_PROFILE_INTERFACE
    schema: ClassVar[str] = OBSERVATION_PROFILE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "required_dimensions",
            "optional_dimensions",
            "observation_bound_cid",
            "claims_general_python_equivalence",
            "profile_cid",
        }
    )

    def __post_init__(self) -> None:
        required = _dimensions(list(self.required_dimensions), "required_dimensions")
        optional = _dimensions(list(self.optional_dimensions), "optional_dimensions")
        overlap = set(required) & set(optional)
        if overlap:
            raise TranslationValidationError(
                f"observation profile dimensions cannot be both required and optional: "
                f"{sorted(overlap)}"
            )
        object.__setattr__(self, "required_dimensions", required)
        object.__setattr__(self, "optional_dimensions", optional)
        object.__setattr__(
            self,
            "observation_bound_cid",
            _optional_cid(self.observation_bound_cid, "observation_bound_cid"),
        )
        if (
            _bool(
                self.claims_general_python_equivalence,
                "claims_general_python_equivalence",
            )
            is not False
        ):
            raise TranslationValidationError("general Python equivalence is not claimed")
        object.__setattr__(self, "claims_general_python_equivalence", False)
        if not required and not optional:
            raise TranslationValidationError("observation profile must declare dimensions")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": OBSERVATION_PROFILE_SCHEMA,
            "interface": OBSERVATION_PROFILE_INTERFACE,
            "required_dimensions": list(self.required_dimensions),
            "optional_dimensions": list(self.optional_dimensions),
            "observation_bound_cid": self.observation_bound_cid,
            "claims_general_python_equivalence": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def profile_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["profile_cid"] = self.profile_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationProfile":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("profile_cid")
        if payload.pop("schema") != OBSERVATION_PROFILE_SCHEMA:
            raise TranslationValidationError("unsupported ObservationProfile schema")
        if payload.pop("interface") != OBSERVATION_PROFILE_INTERFACE:
            raise TranslationValidationError("unsupported ObservationProfile interface")
        if payload.pop("claims_general_python_equivalence") is not False:
            raise TranslationValidationError("general Python equivalence is not claimed")
        result = cls(**payload)
        _verify_cid(claimed, result.profile_cid, "profile_cid")
        return result

    def covers(self, dimension: str) -> bool:
        dim = _dimension_value(dimension)
        return dim in self.required_dimensions or dim in self.optional_dimensions

    def is_required(self, dimension: str) -> bool:
        return _dimension_value(dimension) in self.required_dimensions


@dataclass(frozen=True, slots=True)
class DimensionVerdict:
    """One dimension's evidence-class-preserving verdict."""

    dimension: str
    evidence_class: str
    status: str
    evidence_cid: str = ""
    required: bool = True
    full_suite_executed: bool = False

    interface: ClassVar[str] = DIMENSION_VERDICT_INTERFACE
    schema: ClassVar[str] = DIMENSION_VERDICT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "dimension",
            "evidence_class",
            "status",
            "evidence_cid",
            "required",
            "full_suite_executed",
            "verdict_cid",
        }
    )

    def __post_init__(self) -> None:
        dimension = _dimension_value(self.dimension)
        evidence = _evidence_class_value(self.evidence_class)
        status = _status_value(self.status)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        executed = _bool(self.full_suite_executed, "full_suite_executed")
        if executed and dimension != ValidationDimension.TESTS.value:
            raise TranslationValidationError(
                "full_suite_executed applies only to the tests dimension"
            )
        object.__setattr__(self, "full_suite_executed", executed)
        if status == DimensionStatus.PASS.value:
            if not self.evidence_cid:
                raise TranslationValidationError("passing verdict requires evidence_cid")
            if evidence in NEGATIVE_PROOF_EVIDENCE:
                raise TranslationValidationError(
                    "countermodel evidence cannot pass the proofs dimension"
                )
            assert_evidence_class_admitted(dimension, evidence)
        if status == DimensionStatus.FAIL.value and evidence in NEGATIVE_PROOF_EVIDENCE:
            if dimension != ValidationDimension.PROOFS.value:
                raise TranslationValidationError(
                    "countermodel evidence is confined to the proofs dimension"
                )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": DIMENSION_VERDICT_SCHEMA,
            "interface": DIMENSION_VERDICT_INTERFACE,
            "dimension": self.dimension,
            "evidence_class": self.evidence_class,
            "status": self.status,
            "evidence_cid": self.evidence_cid,
            "required": self.required,
            "full_suite_executed": self.full_suite_executed,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def verdict_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["verdict_cid"] = self.verdict_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DimensionVerdict":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("verdict_cid")
        if payload.pop("schema") != DIMENSION_VERDICT_SCHEMA:
            raise TranslationValidationError("unsupported DimensionVerdict schema")
        if payload.pop("interface") != DIMENSION_VERDICT_INTERFACE:
            raise TranslationValidationError("unsupported DimensionVerdict interface")
        result = cls(**payload)
        _verify_cid(claimed, result.verdict_cid, "verdict_cid")
        return result


def _verdict_from_mapping(
    item: Mapping[str, Any] | DimensionVerdict,
    *,
    required: bool,
) -> DimensionVerdict:
    if isinstance(item, DimensionVerdict):
        if item.required != required:
            return DimensionVerdict(
                dimension=item.dimension,
                evidence_class=item.evidence_class,
                status=item.status,
                evidence_cid=item.evidence_cid,
                required=required,
                full_suite_executed=item.full_suite_executed,
            )
        return item
    payload = dict(item)
    covers = payload.pop("covers", None) or payload.pop("covers_all_dimensions", None)
    if covers not in (None, False, (), []):
        raise TranslationValidationError("evidence classes must not collapse")
    return DimensionVerdict(
        dimension=payload.get("dimension"),
        evidence_class=payload.get("evidence_class"),
        status=payload.get("status"),
        evidence_cid=payload.get("evidence_cid", ""),
        required=required if "required" not in payload else payload.get("required"),
        full_suite_executed=payload.get("full_suite_executed", False),
    )


def _missing_verdict(dimension: str, *, required: bool) -> DimensionVerdict:
    return DimensionVerdict(
        dimension=dimension,
        evidence_class=(
            EvidenceClass.RECONSTRUCTED_PROOF.value
            if dimension == ValidationDimension.PROOFS.value
            else next(iter(admitted_evidence_classes(dimension)))
        ),
        status=(
            DimensionStatus.MISSING.value
            if required
            else DimensionStatus.NOT_IN_PROFILE.value
        ),
        evidence_cid="",
        required=required,
    )


@dataclass(frozen=True, slots=True)
class TranslationValidationRequest:
    """Bound original/candidate extraction plus declared observation profile."""

    tree_id: str
    packet_cid: str
    wave_receipt_cid: str
    selection_cid: str
    original_source_cids: Sequence[str]
    candidate_source_cids: Sequence[str]
    write_paths: Sequence[str]
    validation_commands: Sequence[str]
    observation_profile: ObservationProfile
    dimension_evidence: Sequence[DimensionVerdict]
    full_suite_required: bool = False
    raw_source_required: bool = True
    orchestrator_is_nomination_only: bool = True
    claims_general_python_equivalence: bool = False
    collapse_evidence_classes: bool = False

    interface: ClassVar[str] = TRANSLATION_VALIDATION_REQUEST_INTERFACE
    schema: ClassVar[str] = TRANSLATION_VALIDATION_REQUEST_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "wave_receipt_cid",
            "selection_cid",
            "original_source_cids",
            "candidate_source_cids",
            "write_paths",
            "validation_commands",
            "observation_profile",
            "dimension_evidence",
            "full_suite_required",
            "raw_source_required",
            "orchestrator_is_nomination_only",
            "claims_general_python_equivalence",
            "collapse_evidence_classes",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "request_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(
            self, "wave_receipt_cid", _cid(self.wave_receipt_cid, "wave_receipt_cid")
        )
        object.__setattr__(
            self, "selection_cid", _cid(self.selection_cid, "selection_cid")
        )
        originals = _cids(list(self.original_source_cids), "original_source_cids")
        candidates = _cids(list(self.candidate_source_cids), "candidate_source_cids")
        if not originals:
            raise TranslationValidationError("raw source required")
        if not candidates:
            raise TranslationValidationError("candidate source_cids are required")
        object.__setattr__(self, "original_source_cids", originals)
        object.__setattr__(self, "candidate_source_cids", candidates)
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self, "validation_commands", _commands(self.validation_commands)
        )
        profile = self.observation_profile
        if isinstance(profile, Mapping):
            profile = ObservationProfile.from_dict(profile)
        if not isinstance(profile, ObservationProfile):
            raise TranslationValidationError(
                "observation_profile must be an ObservationProfile"
            )
        object.__setattr__(self, "observation_profile", profile)
        verdicts = tuple(
            _verdict_from_mapping(
                item,
                required=profile.is_required(
                    item.dimension
                    if isinstance(item, DimensionVerdict)
                    else item.get("dimension")
                ),
            )
            for item in self.dimension_evidence
        )
        seen: set[str] = set()
        for verdict in verdicts:
            if verdict.dimension in seen:
                raise TranslationValidationError(
                    f"duplicate dimension evidence: {verdict.dimension}"
                )
            seen.add(verdict.dimension)
            if not profile.covers(verdict.dimension):
                raise TranslationValidationError(
                    f"{verdict.dimension} is outside the observation profile"
                )
        object.__setattr__(self, "dimension_evidence", verdicts)
        assert_evidence_classes_not_collapsed(verdicts)
        object.__setattr__(
            self, "full_suite_required", _bool(self.full_suite_required, "full_suite_required")
        )
        if _bool(self.raw_source_required, "raw_source_required") is not True:
            raise TranslationValidationError("raw_source_required cannot be disabled")
        if (
            _bool(
                self.orchestrator_is_nomination_only,
                "orchestrator_is_nomination_only",
            )
            is not True
        ):
            raise TranslationValidationError("orchestrator must remain nomination_only")
        if (
            _bool(
                self.claims_general_python_equivalence,
                "claims_general_python_equivalence",
            )
            is not False
        ):
            raise TranslationValidationError("general Python equivalence is not claimed")
        if (
            _bool(self.collapse_evidence_classes, "collapse_evidence_classes")
            is not False
        ):
            raise TranslationValidationError("evidence classes must not collapse")
        object.__setattr__(self, "raw_source_required", True)
        object.__setattr__(self, "orchestrator_is_nomination_only", True)
        object.__setattr__(self, "claims_general_python_equivalence", False)
        object.__setattr__(self, "collapse_evidence_classes", False)

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
            "schema": TRANSLATION_VALIDATION_REQUEST_SCHEMA,
            "interface": TRANSLATION_VALIDATION_REQUEST_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "wave_receipt_cid": self.wave_receipt_cid,
            "selection_cid": self.selection_cid,
            "original_source_cids": list(self.original_source_cids),
            "candidate_source_cids": list(self.candidate_source_cids),
            "write_paths": list(self.write_paths),
            "validation_commands": list(self.validation_commands),
            "observation_profile": self.observation_profile.to_dict(),
            "dimension_evidence": [item.to_dict() for item in self.dimension_evidence],
            "full_suite_required": self.full_suite_required,
            "raw_source_required": True,
            "orchestrator_is_nomination_only": True,
            "claims_general_python_equivalence": False,
            "collapse_evidence_classes": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def request_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["request_cid"] = self.request_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TranslationValidationRequest":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("request_cid")
        if payload.pop("schema") != TRANSLATION_VALIDATION_REQUEST_SCHEMA:
            raise TranslationValidationError(
                "unsupported TranslationValidationRequest schema"
            )
        if payload.pop("interface") != TRANSLATION_VALIDATION_REQUEST_INTERFACE:
            raise TranslationValidationError(
                "unsupported TranslationValidationRequest interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("raw_source_required") is not True:
            raise TranslationValidationError("raw_source_required cannot be disabled")
        if payload.pop("orchestrator_is_nomination_only") is not True:
            raise TranslationValidationError("orchestrator must remain nomination_only")
        if payload.pop("claims_general_python_equivalence") is not False:
            raise TranslationValidationError("general Python equivalence is not claimed")
        if payload.pop("collapse_evidence_classes") is not False:
            raise TranslationValidationError("evidence classes must not collapse")
        profile = ObservationProfile.from_dict(payload.pop("observation_profile"))
        evidence = tuple(
            DimensionVerdict.from_dict(item) for item in payload.pop("dimension_evidence")
        )
        result = cls(
            observation_profile=profile,
            dimension_evidence=evidence,
            **payload,
        )
        _verify_cid(claimed, result.request_cid, "request_cid")
        return result


def _evaluate_verdicts(
    request: TranslationValidationRequest,
) -> tuple[tuple[DimensionVerdict, ...], str, str]:
    profile = request.observation_profile
    provided = {item.dimension: item for item in request.dimension_evidence}
    verdicts: list[DimensionVerdict] = []
    for dimension in profile.required_dimensions:
        item = provided.get(dimension)
        if item is None:
            verdicts.append(_missing_verdict(dimension, required=True))
            continue
        if (
            request.full_suite_required
            and dimension == ValidationDimension.TESTS.value
            and item.status == DimensionStatus.PASS.value
            and item.full_suite_executed is not True
        ):
            verdicts.append(
                DimensionVerdict(
                    dimension=dimension,
                    evidence_class=item.evidence_class,
                    status=DimensionStatus.MISSING.value,
                    evidence_cid=item.evidence_cid,
                    required=True,
                    full_suite_executed=False,
                )
            )
            continue
        if item.evidence_class in NEGATIVE_PROOF_EVIDENCE:
            verdicts.append(
                DimensionVerdict(
                    dimension=dimension,
                    evidence_class=item.evidence_class,
                    status=DimensionStatus.FAIL.value,
                    evidence_cid=item.evidence_cid,
                    required=True,
                )
            )
            continue
        verdicts.append(
            DimensionVerdict(
                dimension=item.dimension,
                evidence_class=item.evidence_class,
                status=item.status,
                evidence_cid=item.evidence_cid,
                required=True,
                full_suite_executed=item.full_suite_executed,
            )
        )
    for dimension in profile.optional_dimensions:
        item = provided.get(dimension)
        if item is None:
            verdicts.append(_missing_verdict(dimension, required=False))
            continue
        verdicts.append(
            DimensionVerdict(
                dimension=item.dimension,
                evidence_class=item.evidence_class,
                status=item.status,
                evidence_cid=item.evidence_cid,
                required=False,
                full_suite_executed=item.full_suite_executed,
            )
        )
    assert_evidence_classes_not_collapsed(verdicts)
    required = [item for item in verdicts if item.required]
    if any(item.status == DimensionStatus.UNSUPPORTED.value for item in required):
        reasons = tuple(
            item.dimension
            for item in required
            if item.status == DimensionStatus.UNSUPPORTED.value
        )
        return (
            tuple(verdicts),
            ValidationStatus.UNSUPPORTED.value,
            "unsupported required dimension: " + ",".join(reasons),
        )
    if any(item.status == DimensionStatus.MISSING.value for item in required):
        reasons = tuple(
            item.dimension
            for item in required
            if item.status == DimensionStatus.MISSING.value
        )
        return (
            tuple(verdicts),
            ValidationStatus.INCOMPLETE.value,
            "missing required dimension: " + ",".join(reasons),
        )
    if any(item.status == DimensionStatus.FAIL.value for item in required):
        reasons = tuple(
            item.dimension for item in required if item.status == DimensionStatus.FAIL.value
        )
        return (
            tuple(verdicts),
            ValidationStatus.REJECTED.value,
            "failed required dimension: " + ",".join(reasons),
        )
    if any(item.status != DimensionStatus.PASS.value for item in required):
        return (
            tuple(verdicts),
            ValidationStatus.INCOMPLETE.value,
            "required dimensions did not all pass",
        )
    return (tuple(verdicts), ValidationStatus.VALIDATED.value, "")


@dataclass(frozen=True, slots=True)
class TranslationValidationResult:
    """Per-dimension extraction receipt. Nomination-only; not completion."""

    tree_id: str
    request_cid: str
    packet_cid: str
    wave_receipt_cid: str
    selection_cid: str
    verdicts: Sequence[DimensionVerdict]
    status: str
    terminal_reason: str = ""
    mutated: bool = False
    deterministic: bool = True
    raw_source_required: bool = True
    orchestrator_is_nomination_only: bool = True
    claims_general_python_equivalence: bool = False
    collapse_evidence_classes: bool = False
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = TRANSLATION_VALIDATION_RESULT_INTERFACE
    schema: ClassVar[str] = TRANSLATION_VALIDATION_RESULT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "request_cid",
            "packet_cid",
            "wave_receipt_cid",
            "selection_cid",
            "verdicts",
            "status",
            "terminal_reason",
            "conjunction_passed",
            "mutated",
            "deterministic",
            "raw_source_required",
            "orchestrator_is_nomination_only",
            "claims_general_python_equivalence",
            "collapse_evidence_classes",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "analyzer_id",
            "result_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "request_cid", _cid(self.request_cid, "request_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(
            self, "wave_receipt_cid", _cid(self.wave_receipt_cid, "wave_receipt_cid")
        )
        object.__setattr__(
            self, "selection_cid", _cid(self.selection_cid, "selection_cid")
        )
        verdicts = tuple(
            item if isinstance(item, DimensionVerdict) else DimensionVerdict.from_dict(item)
            for item in self.verdicts
        )
        if not verdicts:
            raise TranslationValidationError("verdicts must not be empty")
        object.__setattr__(self, "verdicts", verdicts)
        assert_evidence_classes_not_collapsed(verdicts)
        status = _validation_status_value(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "terminal_reason",
            _text(self.terminal_reason, "terminal_reason", empty=True),
        )
        if status == ValidationStatus.VALIDATED.value and self.terminal_reason:
            raise TranslationValidationError("validated result cannot carry a terminal reason")
        if status != ValidationStatus.VALIDATED.value and not self.terminal_reason:
            raise TranslationValidationError("typed terminal requires a reason")
        if _bool(self.mutated, "mutated") is not False:
            raise TranslationValidationError("orchestrator cannot mutate")
        if _bool(self.deterministic, "deterministic") is not True:
            raise TranslationValidationError("orchestrator must remain deterministic")
        if _bool(self.raw_source_required, "raw_source_required") is not True:
            raise TranslationValidationError("raw_source_required cannot be disabled")
        if (
            _bool(
                self.orchestrator_is_nomination_only,
                "orchestrator_is_nomination_only",
            )
            is not True
        ):
            raise TranslationValidationError("orchestrator must remain nomination_only")
        if (
            _bool(
                self.claims_general_python_equivalence,
                "claims_general_python_equivalence",
            )
            is not False
        ):
            raise TranslationValidationError("general Python equivalence is not claimed")
        if (
            _bool(self.collapse_evidence_classes, "collapse_evidence_classes")
            is not False
        ):
            raise TranslationValidationError("evidence classes must not collapse")
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise TranslationValidationError("result analyzer_id must remain SPAR-027")
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)
        object.__setattr__(self, "raw_source_required", True)
        object.__setattr__(self, "orchestrator_is_nomination_only", True)
        object.__setattr__(self, "claims_general_python_equivalence", False)
        object.__setattr__(self, "collapse_evidence_classes", False)
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)

    @property
    def conjunction_passed(self) -> bool:
        return self.status == ValidationStatus.VALIDATED.value

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
    def typed_terminal(self) -> bool:
        return self.status in {
            ValidationStatus.UNSUPPORTED.value,
            ValidationStatus.INCOMPLETE.value,
            ValidationStatus.REJECTED.value,
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": TRANSLATION_VALIDATION_RESULT_SCHEMA,
            "interface": TRANSLATION_VALIDATION_RESULT_INTERFACE,
            "tree_id": self.tree_id,
            "request_cid": self.request_cid,
            "packet_cid": self.packet_cid,
            "wave_receipt_cid": self.wave_receipt_cid,
            "selection_cid": self.selection_cid,
            "verdicts": [item.to_dict() for item in self.verdicts],
            "status": self.status,
            "terminal_reason": self.terminal_reason,
            "conjunction_passed": self.conjunction_passed,
            "mutated": False,
            "deterministic": True,
            "raw_source_required": True,
            "orchestrator_is_nomination_only": True,
            "claims_general_python_equivalence": False,
            "collapse_evidence_classes": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "analyzer_id": ANALYZER_ID,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def result_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["result_cid"] = self.result_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TranslationValidationResult":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("result_cid")
        if payload.pop("schema") != TRANSLATION_VALIDATION_RESULT_SCHEMA:
            raise TranslationValidationError(
                "unsupported TranslationValidationResult schema"
            )
        if payload.pop("interface") != TRANSLATION_VALIDATION_RESULT_INTERFACE:
            raise TranslationValidationError(
                "unsupported TranslationValidationResult interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        conjunction = payload.pop("conjunction_passed")
        if payload.pop("raw_source_required") is not True:
            raise TranslationValidationError("raw_source_required cannot be disabled")
        if payload.pop("orchestrator_is_nomination_only") is not True:
            raise TranslationValidationError("orchestrator must remain nomination_only")
        if payload.pop("claims_general_python_equivalence") is not False:
            raise TranslationValidationError("general Python equivalence is not claimed")
        if payload.pop("collapse_evidence_classes") is not False:
            raise TranslationValidationError("evidence classes must not collapse")
        if payload.pop("mutated") is not False:
            raise TranslationValidationError("orchestrator cannot mutate")
        if payload.pop("deterministic") is not True:
            raise TranslationValidationError("orchestrator must remain deterministic")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise TranslationValidationError("result analyzer_id must remain SPAR-027")
        verdicts = tuple(
            DimensionVerdict.from_dict(item) for item in payload.pop("verdicts")
        )
        result = cls(verdicts=verdicts, **payload)
        if conjunction is not result.conjunction_passed:
            raise TranslationValidationError("conjunction_passed does not match status")
        _verify_cid(claimed, result.result_cid, "result_cid")
        return result


@dataclass(frozen=True, slots=True)
class RefactorEquivalenceClaim:
    """Bounded equivalence under a declared profile. Not general Python equivalence."""

    tree_id: str
    result_cid: str
    request_cid: str
    equivalent_under_profile: bool
    per_dimension_evidence: Sequence[Mapping[str, str]]
    general_python_equivalence: bool = False
    claim_is_nomination_only: bool = True
    collapse_evidence_classes: bool = False
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = REFACTOR_EQUIVALENCE_CLAIM_INTERFACE
    schema: ClassVar[str] = REFACTOR_EQUIVALENCE_CLAIM_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "result_cid",
            "request_cid",
            "equivalent_under_profile",
            "per_dimension_evidence",
            "general_python_equivalence",
            "claim_is_nomination_only",
            "collapse_evidence_classes",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "analyzer_id",
            "claim_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "result_cid", _cid(self.result_cid, "result_cid"))
        object.__setattr__(self, "request_cid", _cid(self.request_cid, "request_cid"))
        equivalent = _bool(self.equivalent_under_profile, "equivalent_under_profile")
        object.__setattr__(self, "equivalent_under_profile", equivalent)
        records = _mapping_sequence(
            self.per_dimension_evidence, "per_dimension_evidence"
        )
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in records:
            dimension = _dimension_value(item.get("dimension"))
            evidence = _evidence_class_value(item.get("evidence_class"))
            if dimension in seen:
                raise TranslationValidationError(
                    f"duplicate claim dimension: {dimension}"
                )
            seen.add(dimension)
            extra = set(item) - {"dimension", "evidence_class"}
            if extra:
                raise TranslationValidationError(
                    f"unknown per_dimension_evidence field: {sorted(extra)}"
                )
            if equivalent:
                assert_evidence_class_admitted(dimension, evidence)
            normalized.append({"dimension": dimension, "evidence_class": evidence})
        object.__setattr__(
            self,
            "per_dimension_evidence",
            tuple(sorted(normalized, key=lambda item: item["dimension"])),
        )
        if equivalent:
            for item in self.per_dimension_evidence:
                assert_evidence_class_admitted(item["dimension"], item["evidence_class"])
            classes = {item["evidence_class"] for item in self.per_dimension_evidence}
            dimensions = {item["dimension"] for item in self.per_dimension_evidence}
            if len(dimensions) > 1 and len(classes) == 1:
                shared = next(iter(classes))
                if not all(shared in admitted_evidence_classes(dim) for dim in dimensions):
                    raise TranslationValidationError(
                        "evidence classes must not collapse across disjoint dimensions"
                    )
        if _bool(self.general_python_equivalence, "general_python_equivalence") is not False:
            raise TranslationValidationError("general Python equivalence is not claimed")
        if _bool(self.claim_is_nomination_only, "claim_is_nomination_only") is not True:
            raise TranslationValidationError("claim must remain nomination_only")
        if (
            _bool(self.collapse_evidence_classes, "collapse_evidence_classes")
            is not False
        ):
            raise TranslationValidationError("evidence classes must not collapse")
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise TranslationValidationError("claim analyzer_id must remain SPAR-027")
        object.__setattr__(self, "general_python_equivalence", False)
        object.__setattr__(self, "claim_is_nomination_only", True)
        object.__setattr__(self, "collapse_evidence_classes", False)
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)

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
            "schema": REFACTOR_EQUIVALENCE_CLAIM_SCHEMA,
            "interface": REFACTOR_EQUIVALENCE_CLAIM_INTERFACE,
            "tree_id": self.tree_id,
            "result_cid": self.result_cid,
            "request_cid": self.request_cid,
            "equivalent_under_profile": self.equivalent_under_profile,
            "per_dimension_evidence": [dict(item) for item in self.per_dimension_evidence],
            "general_python_equivalence": False,
            "claim_is_nomination_only": True,
            "collapse_evidence_classes": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "analyzer_id": ANALYZER_ID,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def claim_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["claim_cid"] = self.claim_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorEquivalenceClaim":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("claim_cid")
        if payload.pop("schema") != REFACTOR_EQUIVALENCE_CLAIM_SCHEMA:
            raise TranslationValidationError(
                "unsupported RefactorEquivalenceClaim schema"
            )
        if payload.pop("interface") != REFACTOR_EQUIVALENCE_CLAIM_INTERFACE:
            raise TranslationValidationError(
                "unsupported RefactorEquivalenceClaim interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("general_python_equivalence") is not False:
            raise TranslationValidationError("general Python equivalence is not claimed")
        if payload.pop("claim_is_nomination_only") is not True:
            raise TranslationValidationError("claim must remain nomination_only")
        if payload.pop("collapse_evidence_classes") is not False:
            raise TranslationValidationError("evidence classes must not collapse")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise TranslationValidationError("claim analyzer_id must remain SPAR-027")
        result = cls(**payload)
        _verify_cid(claimed, result.claim_cid, "claim_cid")
        return result


class TranslationValidationOrchestrator:
    """Bind SPAR-025/026 inputs and emit an extraction translation receipt."""

    interface: ClassVar[str] = TRANSLATION_VALIDATION_ORCHESTRATOR_INTERFACE
    schema: ClassVar[str] = TRANSLATION_VALIDATION_ORCHESTRATOR_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def compile_request(
        self,
        *,
        packet: Mapping[str, Any] | Any,
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        dimension_evidence: Sequence[Mapping[str, Any] | DimensionVerdict],
        observation_profile: ObservationProfile | Mapping[str, Any] | None = None,
        original_source_cids: Sequence[str] | None = None,
        candidate_source_cids: Sequence[str] | None = None,
        original_sources: Mapping[str, str] | None = None,
        candidate_sources: Mapping[str, str] | None = None,
        vector_evidence: Any = None,
    ) -> TranslationValidationRequest:
        return compile_translation_validation_request(
            packet=packet,
            wave=wave,
            selection=selection,
            dimension_evidence=dimension_evidence,
            observation_profile=observation_profile,
            original_source_cids=original_source_cids,
            candidate_source_cids=candidate_source_cids,
            original_sources=original_sources,
            candidate_sources=candidate_sources,
            vector_evidence=vector_evidence,
        )

    def validate(
        self,
        request: TranslationValidationRequest | Mapping[str, Any],
        *,
        mutate: bool = False,
    ) -> TranslationValidationResult:
        return validate_translation(request, mutate=mutate)

    def claim(
        self,
        result: TranslationValidationResult | Mapping[str, Any],
    ) -> RefactorEquivalenceClaim:
        return compile_equivalence_claim(result)

    def dry_run(
        self,
        request: TranslationValidationRequest | Mapping[str, Any],
    ) -> TranslationValidationResult:
        return dry_run_translation_validation(request)


def compile_translation_validation_request(
    *,
    packet: Mapping[str, Any] | Any,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    dimension_evidence: Sequence[Mapping[str, Any] | DimensionVerdict],
    observation_profile: ObservationProfile | Mapping[str, Any] | None = None,
    original_source_cids: Sequence[str] | None = None,
    candidate_source_cids: Sequence[str] | None = None,
    original_sources: Mapping[str, str] | None = None,
    candidate_sources: Mapping[str, str] | None = None,
    vector_evidence: Any = None,
) -> TranslationValidationRequest:
    """Bind SPAR-019/025/026 mappings into one extraction validation request."""

    packet_map = _as_mapping(packet, "SPAR-019 packet")
    wave_map = _as_mapping(wave, "SPAR-025 wave")
    selection_map = _as_mapping(selection, "SPAR-026 selection")
    _reject_non_admitting_payload(packet_map, "SPAR-019 packet")
    _reject_non_admitting_payload(wave_map, "SPAR-025 wave")
    _reject_non_admitting_payload(selection_map, "SPAR-026 selection")
    _reject_vector_admission(vector_evidence)

    tree_id = _tree_id(packet_map.get("tree_id"))
    _require_tree(wave_map.get("tree_id"), tree_id, "SPAR-025")
    selection_tree = selection_map.get("tree_id")
    if selection_tree not in (None, ""):
        _require_tree(selection_tree, tree_id, "SPAR-026")

    packet_cid = _packet_cid(packet_map)
    wave_packets = _wave_packet_cids(wave_map)
    if packet_cid not in wave_packets:
        raise TranslationValidationError(
            "SPAR-025 wave packet_cids must include SPAR-019 packet_cid"
        )
    selection_packet = selection_map.get("packet_cid")
    if selection_packet not in (None, "") and _cid(selection_packet, "packet_cid") != packet_cid:
        raise TranslationValidationError("SPAR-026 packet_cid does not match SPAR-019")

    originals = (
        _cids(list(original_source_cids), "original_source_cids")
        if original_source_cids is not None
        else _packet_source_cids(packet_map)
    )
    candidates = candidate_source_cids
    if candidates is None:
        after = wave_map.get("after_source_cids") or wave_map.get("candidate_source_cids")
        if after in (None, (), []):
            raise TranslationValidationError("candidate source_cids are required")
        candidates = after
    candidate_cids = _cids(list(candidates), "candidate_source_cids")

    if isinstance(observation_profile, ObservationProfile):
        profile = observation_profile
    elif observation_profile in (None, ""):
        profile = ObservationProfile()
    else:
        profile = ObservationProfile.from_dict(_as_mapping(observation_profile, "observation_profile"))

    evidence_items = [dict(item) if isinstance(item, Mapping) else item.to_dict() for item in dimension_evidence]
    derived = _derive_syntax_evidence(
        original_source_cids=originals,
        candidate_source_cids=candidate_cids,
        original_sources=original_sources,
        candidate_sources=candidate_sources,
    )
    if derived is not None and not any(
        _dimension_value(
            item.get("dimension") if isinstance(item, Mapping) else item["dimension"]
        )
        == ValidationDimension.SYNTAX.value
        for item in evidence_items
    ):
        evidence_items.append(derived)

    write_paths = _packet_write_paths(packet_map)
    wave_paths = wave_map.get("write_paths")
    if wave_paths not in (None, ()):
        resolved_wave_paths = _exact_paths(wave_paths, "write_paths")
        if tuple(sorted(resolved_wave_paths)) != tuple(sorted(write_paths)):
            raise TranslationValidationError("SPAR-025 write_paths must match SPAR-019")

    return TranslationValidationRequest(
        tree_id=tree_id,
        packet_cid=packet_cid,
        wave_receipt_cid=_wave_receipt_cid(wave_map),
        selection_cid=_selection_cid(selection_map),
        original_source_cids=originals,
        candidate_source_cids=candidate_cids,
        write_paths=write_paths,
        validation_commands=_packet_validation_commands(packet_map),
        observation_profile=profile,
        dimension_evidence=tuple(evidence_items),
        full_suite_required=_full_suite_required(selection_map),
    )


def validate_translation(
    request: TranslationValidationRequest | Mapping[str, Any],
    *,
    mutate: bool = False,
) -> TranslationValidationResult:
    """Evaluate the request conjunction without collapsing evidence classes."""

    if mutate is not False:
        raise TranslationValidationError("orchestrator cannot mutate")
    resolved = (
        request
        if isinstance(request, TranslationValidationRequest)
        else TranslationValidationRequest.from_dict(request)
    )
    verdicts, status, reason = _evaluate_verdicts(resolved)
    return TranslationValidationResult(
        tree_id=resolved.tree_id,
        request_cid=resolved.request_cid,
        packet_cid=resolved.packet_cid,
        wave_receipt_cid=resolved.wave_receipt_cid,
        selection_cid=resolved.selection_cid,
        verdicts=verdicts,
        status=status,
        terminal_reason=reason,
    )


def dry_run_translation_validation(
    request: TranslationValidationRequest | Mapping[str, Any],
) -> TranslationValidationResult:
    """Deterministic non-mutating evaluation of one translation request."""

    result = validate_translation(request, mutate=False)
    if result.mutated is not False or result.deterministic is not True:
        raise TranslationValidationError("dry-run must remain deterministic and non-mutating")
    return result


def compile_equivalence_claim(
    result: TranslationValidationResult | Mapping[str, Any],
) -> RefactorEquivalenceClaim:
    """Nominate a bounded equivalence claim from a validation result."""

    resolved = (
        result
        if isinstance(result, TranslationValidationResult)
        else TranslationValidationResult.from_dict(result)
    )
    per_dimension = tuple(
        {"dimension": item.dimension, "evidence_class": item.evidence_class}
        for item in resolved.verdicts
        if item.required and item.status == DimensionStatus.PASS.value
    )
    return RefactorEquivalenceClaim(
        tree_id=resolved.tree_id,
        result_cid=resolved.result_cid,
        request_cid=resolved.request_cid,
        equivalent_under_profile=resolved.conjunction_passed,
        per_dimension_evidence=per_dimension,
    )


def encode_canonical_request(request: TranslationValidationRequest) -> dict[str, Any]:
    return request.to_dict()


def decode_canonical_request(payload: Mapping[str, Any]) -> TranslationValidationRequest:
    return TranslationValidationRequest.from_dict(payload)


def encode_canonical_result(result: TranslationValidationResult) -> dict[str, Any]:
    return result.to_dict()


def decode_canonical_result(payload: Mapping[str, Any]) -> TranslationValidationResult:
    return TranslationValidationResult.from_dict(payload)


def encode_canonical_claim(claim: RefactorEquivalenceClaim) -> dict[str, Any]:
    return claim.to_dict()


def decode_canonical_claim(payload: Mapping[str, Any]) -> RefactorEquivalenceClaim:
    return RefactorEquivalenceClaim.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise TranslationValidationError(
            f"translation validation must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "CLAIM_IS_NOMINATION_ONLY",
    "DECLARED_DIMENSIONS",
    "DECLARED_DIMENSION_STATUSES",
    "DECLARED_EVIDENCE_CLASSES",
    "DECLARED_VALIDATION_STATUSES",
    "DEFAULT_REQUIRED_DIMENSIONS",
    "DIMENSION_ADMITTED_EVIDENCE",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "FORBIDDEN_EQUIVALENCE_NAMES",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "ORCHESTRATOR_IS_NOMINATION_ONLY",
    "PROCEDURE_COMPILER_IS_NOT_EXTRACTION_RECEIPT",
    "PROGRAM",
    "PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REFACTOR_EQUIVALENCE_CLAIM_INTERFACE",
    "RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TEST_PASS_IS_NOT_PROOF",
    "TRANSLATION_VALIDATION_ORCHESTRATOR_INTERFACE",
    "TRANSLATION_VALIDATION_REQUEST_INTERFACE",
    "TRANSLATION_VALIDATION_RESULT_INTERFACE",
    "VALIDATION_CAN_AUTHORIZE_COMPLETION",
    "VALIDATION_CAN_AUTHORIZE_TRANSITION",
    "VALIDATION_CAN_CREATE_AUTHORITY",
    "VALIDATION_CLAIMS_GENERAL_PYTHON_EQUIVALENCE",
    "VALIDATION_COLLAPSES_EVIDENCE_CLASSES",
    "VALIDATION_CONTRACT_VERSION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "DimensionStatus",
    "DimensionVerdict",
    "EvidenceClass",
    "ObservationProfile",
    "RefactorEquivalenceClaim",
    "TranslationValidationError",
    "TranslationValidationOrchestrator",
    "TranslationValidationRequest",
    "TranslationValidationResult",
    "ValidationDimension",
    "ValidationStatus",
    "admitted_evidence_classes",
    "assert_evidence_class_admitted",
    "assert_evidence_classes_not_collapsed",
    "assert_not_competing_capsule_family",
    "compile_equivalence_claim",
    "compile_translation_validation_request",
    "decode_canonical_claim",
    "decode_canonical_request",
    "decode_canonical_result",
    "dry_run_translation_validation",
    "encode_canonical_claim",
    "encode_canonical_request",
    "encode_canonical_result",
    "provider_free_exports",
    "translation_validation_cid_profile",
    "translation_validation_descriptor",
    "validate_translation",
]
