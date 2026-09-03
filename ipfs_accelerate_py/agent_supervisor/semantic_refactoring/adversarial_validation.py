"""SPAR-029 property, metamorphic, mutation, and adversarial validation.

This module extends current supervisor validation orchestration with
``MutationCampaignRequest@1``, ``MutationCampaignReceipt@1``, and
``RefactorMutationAndAdversarialValidator@1``.  It consumes SPAR-025
extraction-wave receipts and SPAR-026 selection mappings, then reuses or
generates bounded property/metamorphic relations, mutants, and adversarial
cases for moved boundaries and compatibility façades.

Evidence classes remain separate.  A test proves only tested executions; a
killed mutant proves only that oracle; an adversarial observation proves
only that case.  Vector, model, and heuristic evidence cannot admit a
campaign.  General Python equivalence is not claimed.  Critical survivors
and unknown required dynamics block acceptance.  Unsupported required
behavior is a typed terminal, never success.

The validator is nomination-only.  It cannot authorize a transition,
completion, source mutation, or competing authority.  Observational
metadata is excluded from identity.  Dry-run is deterministic and never
mutates.
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


TASK_ID: Final[str] = "SPAR-029"
GOAL_ID: Final[str] = "SPAR-G052"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "validation orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.adversarial_validation@1"
)

MUTATION_CAMPAIGN_REQUEST_INTERFACE: Final[str] = "MutationCampaignRequest@1"
MUTATION_CAMPAIGN_RECEIPT_INTERFACE: Final[str] = "MutationCampaignReceipt@1"
REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_INTERFACE: Final[str] = (
    "RefactorMutationAndAdversarialValidator@1"
)
PROPERTY_RELATION_INTERFACE: Final[str] = "PropertyRelation@1"
METAMORPHIC_RELATION_INTERFACE: Final[str] = "MetamorphicRelation@1"
MUTANT_SPEC_INTERFACE: Final[str] = "MutantSpec@1"
ADVERSARIAL_CASE_INTERFACE: Final[str] = "AdversarialCase@1"
RELATION_VERDICT_INTERFACE: Final[str] = "RelationVerdict@1"
MUTANT_VERDICT_INTERFACE: Final[str] = "MutantVerdict@1"
ADVERSARIAL_VERDICT_INTERFACE: Final[str] = "AdversarialVerdict@1"
CAMPAIGN_PROFILE_INTERFACE: Final[str] = "CampaignProfile@1"

MUTATION_CAMPAIGN_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-campaign-request@1"
)
MUTATION_CAMPAIGN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutation-campaign-receipt@1"
)
REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-mutation-and-adversarial-validator@1"
)
PROPERTY_RELATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/property-relation@1"
)
METAMORPHIC_RELATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/metamorphic-relation@1"
)
MUTANT_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutant-spec@1"
)
ADVERSARIAL_CASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-case@1"
)
RELATION_VERDICT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/relation-verdict@1"
)
MUTANT_VERDICT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mutant-verdict@1"
)
ADVERSARIAL_VERDICT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-verdict@1"
)
CAMPAIGN_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/campaign-profile@1"
)

VALIDATION_CONTRACT_VERSION: Final[str] = "1"

VALIDATION_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
VALIDATION_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
VALIDATION_CAN_CREATE_AUTHORITY: Final[bool] = False
VALIDATION_CLAIMS_GENERAL_PYTHON_EQUIVALENCE: Final[bool] = False
VALIDATION_COLLAPSES_EVIDENCE_CLASSES: Final[bool] = False
CRITICAL_SURVIVOR_BLOCKS_ACCEPTANCE: Final[bool] = True
UNKNOWN_REQUIRED_DYNAMICS_BLOCK_ACCEPTANCE: Final[bool] = True
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
TEST_PASS_IS_NOT_PROOF: Final[bool] = True
MUTANT_KILL_IS_NOT_COMPLETION: Final[bool] = True
RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT: Final[bool] = True
PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
VALIDATOR_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
VALIDATOR_CAN_MUTATE_SOURCE: Final[bool] = False

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_RELATIONS: Final[int] = 64
MAX_MUTANTS: Final[int] = 64
MAX_ADVERSARIAL: Final[int] = 64
MAX_FAMILIES: Final[int] = 8

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
        "kill_all_mutants_without_evidence",
    }
)

_DYNAMIC_IMPORT_KINDS: Final[frozenset[str]] = frozenset(
    {
        "dynamic_import",
        "unknown_dynamic_import",
        "importlib",
        "__import__",
        "plugin_entry_point",
    }
)


class AdversarialValidationError(ValueError):
    """Fail-closed violation of a SPAR-029 mutation/adversarial contract."""


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


class CampaignFamily(str, Enum):
    PROPERTY = "property"
    METAMORPHIC = "metamorphic"
    MUTATION = "mutation"
    ADVERSARIAL = "adversarial"


class PropertyKind(str, Enum):
    FACADE_REEXPORT_IDENTITY = "facade_reexport_identity"
    BOUNDARY_ASSUME_GUARANTEE = "boundary_assume_guarantee"
    STAR_EXPORT_NAMES = "star_export_names"
    WRAPPER_EXCEPTION_IDENTITY = "wrapper_exception_identity"


class MetamorphicKind(str, Enum):
    FACADE_VS_EXTRACTED_CALL = "facade_vs_extracted_call"
    IMPORT_PATH_ALIAS = "import_path_alias"
    REGISTRATION_ORDER = "registration_order"
    SERIALIZATION_ROUNDTRIP = "serialization_roundtrip"


class MutantKind(str, Enum):
    DROP_REEXPORT = "drop_reexport"
    RENAME_FACADE_SYMBOL = "rename_facade_symbol"
    INVERT_BOUNDARY_GUARD = "invert_boundary_guard"
    SKIP_REGISTRATION = "skip_registration"
    WEAKEN_EXCEPTION = "weaken_exception"
    SWAP_IMPORT_PATH = "swap_import_path"
    DROP_WRAPPER = "drop_wrapper"
    ALTER_DEFAULT = "alter_default"
    SUPPRESS_DEPRECATION = "suppress_deprecation"


class AdversarialKind(str, Enum):
    UNKNOWN_DYNAMIC_IMPORT = "unknown_dynamic_import"
    REQUIRED_UNRESOLVED_DYNAMIC = "required_unresolved_dynamic"
    PLUGIN_ENTRY_POINT_DRIFT = "plugin_entry_point_drift"
    REGISTRY_ALIAS_COLLISION = "registry_alias_collision"
    PICKLE_QUALNAME_BREAK = "pickle_qualname_break"
    PATCH_TARGET_MISS = "patch_target_miss"
    STAR_EXPORT_LEAK = "star_export_leak"


class RelationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    NOT_IN_PROFILE = "not_in_profile"


class MutantStatus(str, Enum):
    KILLED = "killed"
    SURVIVED = "survived"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    NOT_IN_PROFILE = "not_in_profile"


class CampaignStatus(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"


DECLARED_EVIDENCE_CLASSES: Final[frozenset[str]] = frozenset(
    item.value for item in EvidenceClass
)
DECLARED_FAMILIES: Final[frozenset[str]] = frozenset(
    item.value for item in CampaignFamily
)
DECLARED_PROPERTY_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in PropertyKind
)
DECLARED_METAMORPHIC_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in MetamorphicKind
)
DECLARED_MUTANT_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in MutantKind
)
DECLARED_ADVERSARIAL_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in AdversarialKind
)
DECLARED_RELATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in RelationStatus
)
DECLARED_MUTANT_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in MutantStatus
)
DECLARED_CAMPAIGN_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in CampaignStatus
)

DEFAULT_REQUIRED_FAMILIES: Final[tuple[str, ...]] = tuple(
    item.value for item in CampaignFamily
)

DEFAULT_CRITICAL_MUTANT_KINDS: Final[tuple[str, ...]] = (
    MutantKind.DROP_REEXPORT.value,
    MutantKind.RENAME_FACADE_SYMBOL.value,
    MutantKind.INVERT_BOUNDARY_GUARD.value,
)

FAMILY_ADMITTED_EVIDENCE: Final[Mapping[str, frozenset[str]]] = {
    CampaignFamily.PROPERTY.value: frozenset(
        {
            EvidenceClass.EXACT_STATIC_FACT.value,
            EvidenceClass.SPECIFICATION.value,
            EvidenceClass.TEST.value,
        }
    ),
    CampaignFamily.METAMORPHIC.value: frozenset(
        {
            EvidenceClass.TEST.value,
            EvidenceClass.RUNTIME_OBSERVATION.value,
            EvidenceClass.SPECIFICATION.value,
        }
    ),
    CampaignFamily.MUTATION.value: frozenset(
        {
            EvidenceClass.TEST.value,
            EvidenceClass.RUNTIME_OBSERVATION.value,
        }
    ),
    CampaignFamily.ADVERSARIAL.value: frozenset(
        {
            EvidenceClass.TEST.value,
            EvidenceClass.RUNTIME_OBSERVATION.value,
            EvidenceClass.COUNTERMODEL.value,
            EvidenceClass.REPLAYED_COUNTEREXAMPLE.value,
            EvidenceClass.MAY_FACT.value,
            EvidenceClass.EXACT_STATIC_FACT.value,
        }
    ),
}


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise AdversarialValidationError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise AdversarialValidationError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise AdversarialValidationError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise AdversarialValidationError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise AdversarialValidationError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise AdversarialValidationError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise AdversarialValidationError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise AdversarialValidationError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise AdversarialValidationError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise AdversarialValidationError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise AdversarialValidationError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise AdversarialValidationError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise AdversarialValidationError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise AdversarialValidationError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise AdversarialValidationError(f"{name} does not verify")


def _ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise AdversarialValidationError(f"{name} must be a list")
    ordered = tuple(_text(item, name) for item in values)
    if len(ordered) > limit:
        raise AdversarialValidationError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise AdversarialValidationError(f"{name} must not contain duplicates")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise AdversarialValidationError(f"{name} exceeds path bound")
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
        raise AdversarialValidationError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise AdversarialValidationError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise AdversarialValidationError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise AdversarialValidationError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise AdversarialValidationError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise AdversarialValidationError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise AdversarialValidationError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise AdversarialValidationError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise AdversarialValidationError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise AdversarialValidationError(f"missing {name}")
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
    raise AdversarialValidationError(f"{name} must be a mapping")


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
    raise AdversarialValidationError(f"{name} must be an object")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AdversarialValidationError(f"{name} must be a list")
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
        raise AdversarialValidationError(f"{name} items must be objects")
    if len(items) > MAX_MEMBERS:
        raise AdversarialValidationError(f"{name} exceeds maximum length")
    return tuple(items)


def adversarial_validation_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def adversarial_validation_descriptor() -> dict[str, Any]:
    return {
        "interface": REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "raw_source_required": True,
        "nomination_only": True,
        "claims_general_python_equivalence": False,
        "collapses_evidence_classes": False,
        "critical_survivor_blocks_acceptance": True,
        "unknown_required_dynamics_block_acceptance": True,
        "can_mutate_source": False,
        "forbids": tuple(sorted(FORBIDDEN_EQUIVALENCE_NAMES)),
    }


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise AdversarialValidationError(f"{name} cannot claim {flag}")


def _family_value(value: Any, name: str = "family") -> str:
    if isinstance(value, CampaignFamily):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_FAMILIES:
        raise AdversarialValidationError(f"unsupported {name} {text!r}")
    return text


def _evidence_class_value(value: Any, name: str = "evidence_class") -> str:
    if isinstance(value, EvidenceClass):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_EVIDENCE_CLASSES:
        raise AdversarialValidationError(f"unsupported {name} {text!r}")
    return text


def _relation_status_value(value: Any, name: str = "status") -> str:
    if isinstance(value, RelationStatus):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_RELATION_STATUSES:
        raise AdversarialValidationError(f"unsupported {name} {text!r}")
    return text


def _mutant_status_value(value: Any, name: str = "status") -> str:
    if isinstance(value, MutantStatus):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_MUTANT_STATUSES:
        raise AdversarialValidationError(f"unsupported {name} {text!r}")
    return text


def _campaign_status_value(value: Any, name: str = "status") -> str:
    if isinstance(value, CampaignStatus):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_CAMPAIGN_STATUSES:
        raise AdversarialValidationError(f"unsupported {name} {text!r}")
    return text


def _families(values: Any, name: str) -> tuple[str, ...]:
    ordered = _ordered_text(values, name, limit=MAX_FAMILIES)
    for item in ordered:
        if item not in DECLARED_FAMILIES:
            raise AdversarialValidationError(f"unsupported {name} {item!r}")
    return ordered


def _cids(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise AdversarialValidationError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise AdversarialValidationError(f"{name} must not be empty")
    if len(ordered) != len(set(ordered)):
        raise AdversarialValidationError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise AdversarialValidationError(f"{name} exceeds maximum length")
    return ordered


def _require_tree(actual: Any, expected: str, name: str) -> str:
    tree = _tree_id(actual)
    if tree != expected:
        raise AdversarialValidationError(f"{name} tree_id does not match packet tree_id")
    return tree


def _kind_for_family(family: str, kind: Any) -> str:
    text = _text(getattr(kind, "value", kind), "kind")
    allowed = {
        CampaignFamily.PROPERTY.value: DECLARED_PROPERTY_KINDS,
        CampaignFamily.METAMORPHIC.value: DECLARED_METAMORPHIC_KINDS,
        CampaignFamily.MUTATION.value: DECLARED_MUTANT_KINDS,
        CampaignFamily.ADVERSARIAL.value: DECLARED_ADVERSARIAL_KINDS,
    }[family]
    if text not in allowed:
        raise AdversarialValidationError(f"unsupported {family} kind {text!r}")
    return text


def admitted_evidence_classes(family: str) -> frozenset[str]:
    key = _family_value(family)
    return FAMILY_ADMITTED_EVIDENCE[key]


def assert_evidence_class_admitted(family: str, evidence_class: str) -> None:
    fam = _family_value(family)
    evidence = _evidence_class_value(evidence_class)
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise AdversarialValidationError(
            "vector, model, or heuristic evidence cannot admit mutation validation"
        )
    if evidence == EvidenceClass.PROOF_CANDIDATE.value:
        raise AdversarialValidationError("proof_candidate cannot admit proofs")
    if evidence == EvidenceClass.ACCEPTED_TRANSITION.value:
        raise AdversarialValidationError(
            "accepted_transition cannot admit a mutation-validation family"
        )
    if evidence not in admitted_evidence_classes(fam):
        raise AdversarialValidationError(
            f"{evidence} cannot admit {fam}; evidence classes must not collapse"
        )


def assert_evidence_classes_not_collapsed(
    items: Sequence[Mapping[str, Any]] | Sequence[Any],
) -> None:
    passed: list[tuple[str, str, str]] = []
    for item in items:
        if isinstance(item, Mapping):
            family = _family_value(item.get("family"))
            evidence = _evidence_class_value(item.get("evidence_class"))
            cid = _optional_cid(item.get("evidence_cid"), "evidence_cid")
            status = str(item.get("status") or "")
        else:
            family = _family_value(getattr(item, "family"))
            evidence = _evidence_class_value(getattr(item, "evidence_class"))
            cid = _optional_cid(getattr(item, "evidence_cid"), "evidence_cid")
            status = str(getattr(item, "status"))
        if status not in {
            RelationStatus.PASS.value,
            MutantStatus.KILLED.value,
        }:
            continue
        assert_evidence_class_admitted(family, evidence)
        passed.append((family, evidence, cid))
    by_cid: dict[str, list[tuple[str, str]]] = {}
    for family, evidence, cid in passed:
        if not cid:
            continue
        by_cid.setdefault(cid, []).append((family, evidence))
    for cid, grouped in by_cid.items():
        families = {family for family, _evidence in grouped}
        classes = {evidence for _family, evidence in grouped}
        if len(classes) > 1:
            raise AdversarialValidationError(
                "evidence classes must not collapse across disjoint families"
            )
        if len(families) <= 1:
            continue
        admitted = [admitted_evidence_classes(family) for family, _evidence in grouped]
        shared = admitted[0]
        for item in admitted[1:]:
            shared = shared & item
        if not shared:
            raise AdversarialValidationError(
                "evidence classes must not collapse across disjoint families"
            )


def _packet_cid(packet: Mapping[str, Any]) -> str:
    claimed = packet.get("packet_cid")
    if claimed in (None, ""):
        raise AdversarialValidationError("SPAR-019 packet_cid is required")
    return _cid(claimed, "packet_cid")


def _packet_write_paths(packet: Mapping[str, Any]) -> tuple[str, ...]:
    if "write_paths" in packet:
        return _exact_paths(packet["write_paths"], "write_paths")
    scope = _nested_mapping(packet.get("effect_scope"), "effect_scope")
    if "write_paths" in scope:
        return _exact_paths(scope["write_paths"], "write_paths")
    raise AdversarialValidationError("SPAR-019 packet write_paths are required")


def _packet_source_cids(packet: Mapping[str, Any]) -> tuple[str, ...]:
    preimage = _nested_mapping(packet.get("preimage"), "preimage")
    sources = preimage.get("source_cids", packet.get("source_cids"))
    if sources in (None, (), []) or not sources:
        raise AdversarialValidationError(
            "raw source required: SPAR-019 preimage source_cids"
        )
    ordered = _cids(list(sources), "source_cids")
    if not ordered:
        raise AdversarialValidationError(
            "raw source required: SPAR-019 preimage source_cids"
        )
    return ordered


def _packet_validation_commands(packet: Mapping[str, Any]) -> tuple[str, ...]:
    commands = packet.get("validation_commands")
    if commands in (None, ()):
        raise AdversarialValidationError("SPAR-019 validation_commands are required")
    return _commands(commands)


def _wave_receipt_cid(wave: Mapping[str, Any]) -> str:
    claimed = wave.get("receipt_cid") or wave.get("wave_receipt_cid")
    if claimed in (None, ""):
        raise AdversarialValidationError("SPAR-025 receipt_cid is required")
    return _cid(claimed, "wave_receipt_cid")


def _wave_packet_cids(wave: Mapping[str, Any]) -> tuple[str, ...]:
    packets = wave.get("packet_cids")
    if packets in (None, ()):
        raise AdversarialValidationError("SPAR-025 packet_cids are required")
    return _cids(list(packets), "packet_cids")


def _selection_cid(selection: Mapping[str, Any]) -> str:
    claimed = (
        selection.get("validation_selection_cid")
        or selection.get("producer_selection_cid")
        or selection.get("selection_cid")
    )
    if claimed in (None, ""):
        raise AdversarialValidationError("SPAR-026 selection_cid is required")
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
            raise AdversarialValidationError(
                "vectors/projections cannot suppress raw-source fallback"
            )
        if (
            item.get("admit_equivalence") is True
            or item.get("collapse_evidence_classes") is True
            or item.get("claims_general_python_equivalence") is True
            or item.get("admit_campaign") is True
        ):
            raise AdversarialValidationError(
                "vector, model, or heuristic evidence cannot admit mutation validation"
            )
        if evidence in _NON_ADMITTING_EVIDENCE and (
            item.get("admits_validation") is True or item.get("admits_equivalence") is True
        ):
            raise AdversarialValidationError(
                "vector, model, or heuristic evidence cannot admit mutation validation"
            )


def _reject_non_admitting_payload(payload: Mapping[str, Any], name: str) -> None:
    evidence = str(payload.get("evidence_class") or "")
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise AdversarialValidationError(
            f"vector, model, or heuristic evidence cannot admit {name}"
        )
    if payload.get("claims_general_python_equivalence") is True:
        raise AdversarialValidationError("general Python equivalence is not claimed")
    if payload.get("collapse_evidence_classes") is True:
        raise AdversarialValidationError("evidence classes must not collapse")


def _parse_sources(sources: Mapping[str, str], name: str) -> dict[str, ast.AST]:
    if not isinstance(sources, Mapping) or isinstance(sources, (str, bytes, bytearray)):
        raise AdversarialValidationError(f"{name} must be a mapping of path to source")
    if not sources:
        raise AdversarialValidationError(f"{name} must not be empty")
    parsed: dict[str, ast.AST] = {}
    for path, text in sources.items():
        _exact_path(path, name)
        if type(text) is not str:
            raise AdversarialValidationError(f"{name} values must be strings")
        try:
            parsed[path] = ast.parse(text)
        except SyntaxError as exc:
            raise AdversarialValidationError(f"{name} failed to parse") from exc
    return parsed


def _public_names(tree: ast.AST) -> frozenset[str]:
    declared: set[str] = set()
    all_names: list[str] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
                        extracted: list[str] = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and type(elt.value) is str:
                                extracted.append(elt.value)
                        all_names = extracted
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    declared.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                declared.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                declared.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name != "*" and not name.startswith("_"):
                    declared.add(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", 1)[0]
                if not name.startswith("_"):
                    declared.add(name)
    if all_names is not None:
        return frozenset(all_names)
    return frozenset(declared)


def _subject_id(write_paths: Sequence[str]) -> str:
    return write_paths[0] if write_paths else "bound-subject"


def _facade_kinds(
    wave: Mapping[str, Any],
    facade_plan: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    kinds: list[str] = []
    for key in ("adapter_kinds", "rewrite_kinds", "edit_kinds", "migration_kinds"):
        values = wave.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for item in values:
                if type(item) is str and item not in kinds:
                    kinds.append(item)
    if facade_plan:
        for key in ("migration_kinds", "kinds"):
            values = facade_plan.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes, bytearray)
            ):
                for item in values:
                    if type(item) is str and item not in kinds:
                        kinds.append(item)
        plans = _mapping_sequence(facade_plan.get("plans") or facade_plan.get("migrations"), "plans")
        for plan in plans:
            kind = plan.get("kind") or plan.get("migration_kind")
            if type(kind) is str and kind not in kinds:
                kinds.append(kind)
    return tuple(kinds)


def _truthy_unresolved(value: Any) -> bool:
    if value is True:
        return True
    if type(value) is int and not isinstance(value, bool) and value > 0:
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
        return len(value) > 0
    return False


def unresolved_required_dynamics(
    graph: Mapping[str, Any] | None,
    frontier: Mapping[str, Any] | None,
) -> bool:
    payloads = [item for item in (graph, frontier) if item]
    for payload in payloads:
        for key in (
            "unresolved_required_dynamics",
            "required_unresolved_dynamics",
            "unknown_required_dynamics",
            "required_unresolved",
        ):
            if _truthy_unresolved(payload.get(key)):
                return True
        if _truthy_unresolved(payload.get("unresolved_count")):
            return True
        unknown = payload.get("unknown_kinds")
        if isinstance(unknown, Sequence) and not isinstance(unknown, (str, bytes, bytearray)):
            if len(unknown) > 0:
                return True
        frontier_payload = payload.get("unresolved_frontier")
        if isinstance(frontier_payload, Sequence) and not isinstance(
            frontier_payload, (str, bytes, bytearray, Mapping)
        ):
            if len(frontier_payload) > 0:
                return True
        nested = _nested_mapping(frontier_payload, "unresolved_frontier") if frontier_payload else {}
        if _truthy_unresolved(nested.get("unresolved_count")) or _truthy_unresolved(
            nested.get("items")
        ):
            return True
        findings = _mapping_sequence(payload.get("findings"), "findings")
        for finding in findings:
            required = finding.get("required")
            unresolved = finding.get("unresolved") is True or str(
                finding.get("presence") or ""
            ) in {"unknown", "unresolved", "required_unknown"}
            if required is not False and unresolved:
                return True
    return False


def unknown_dynamic_import_present(
    graph: Mapping[str, Any] | None,
    frontier: Mapping[str, Any] | None,
) -> bool:
    payloads = [item for item in (graph, frontier) if item]
    for payload in payloads:
        unknown = payload.get("unknown_kinds")
        if isinstance(unknown, Sequence) and not isinstance(unknown, (str, bytes, bytearray)):
            if any(str(item) in _DYNAMIC_IMPORT_KINDS for item in unknown):
                return True
        findings = _mapping_sequence(payload.get("findings"), "findings")
        nested = payload.get("unresolved_frontier")
        if isinstance(nested, Mapping):
            findings = findings + _mapping_sequence(nested.get("items"), "unresolved_frontier.items")
        for finding in findings:
            kind = str(
                finding.get("kind")
                or finding.get("inventory_label")
                or finding.get("dynamic_kind")
                or ""
            )
            if kind in _DYNAMIC_IMPORT_KINDS or "dynamic_import" in kind:
                if finding.get("unresolved") is not False:
                    return True
    return False


@dataclass(frozen=True, slots=True)
class CampaignProfile:
    """Declared families bound for one mutation/adversarial campaign."""

    required_families: Sequence[str] = DEFAULT_REQUIRED_FAMILIES
    optional_families: Sequence[str] = ()
    claims_general_python_equivalence: bool = False

    interface: ClassVar[str] = CAMPAIGN_PROFILE_INTERFACE
    schema: ClassVar[str] = CAMPAIGN_PROFILE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "required_families",
            "optional_families",
            "claims_general_python_equivalence",
            "profile_cid",
        }
    )

    def __post_init__(self) -> None:
        required = _families(list(self.required_families), "required_families")
        optional = _families(list(self.optional_families), "optional_families")
        overlap = set(required) & set(optional)
        if overlap:
            raise AdversarialValidationError(
                f"campaign profile families cannot be both required and optional: "
                f"{sorted(overlap)}"
            )
        object.__setattr__(self, "required_families", required)
        object.__setattr__(self, "optional_families", optional)
        if (
            _bool(
                self.claims_general_python_equivalence,
                "claims_general_python_equivalence",
            )
            is not False
        ):
            raise AdversarialValidationError("general Python equivalence is not claimed")
        object.__setattr__(self, "claims_general_python_equivalence", False)
        if not required and not optional:
            raise AdversarialValidationError("campaign profile must declare families")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": CAMPAIGN_PROFILE_SCHEMA,
            "interface": CAMPAIGN_PROFILE_INTERFACE,
            "required_families": list(self.required_families),
            "optional_families": list(self.optional_families),
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
    def from_dict(cls, data: Mapping[str, Any]) -> "CampaignProfile":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("profile_cid")
        if payload.pop("schema") != CAMPAIGN_PROFILE_SCHEMA:
            raise AdversarialValidationError("unsupported CampaignProfile schema")
        if payload.pop("interface") != CAMPAIGN_PROFILE_INTERFACE:
            raise AdversarialValidationError("unsupported CampaignProfile interface")
        if payload.pop("claims_general_python_equivalence") is not False:
            raise AdversarialValidationError("general Python equivalence is not claimed")
        result = cls(**payload)
        _verify_cid(claimed, result.profile_cid, "profile_cid")
        return result

    def covers(self, family: str) -> bool:
        fam = _family_value(family)
        return fam in self.required_families or fam in self.optional_families

    def is_required(self, family: str) -> bool:
        return _family_value(family) in self.required_families


def _coerce_profile(value: CampaignProfile | Mapping[str, Any] | None) -> CampaignProfile:
    if isinstance(value, CampaignProfile):
        return value
    if value in (None, ""):
        return CampaignProfile()
    return CampaignProfile.from_dict(_as_mapping(value, "campaign_profile"))


@dataclass(frozen=True, slots=True)
class PropertyRelation:
    """Bounded property that must hold on original and candidate surfaces."""

    relation_id: str
    kind: str
    subject_id: str
    required: bool = True
    critical: bool = True
    reused: bool = False

    interface: ClassVar[str] = PROPERTY_RELATION_INTERFACE
    schema: ClassVar[str] = PROPERTY_RELATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "relation_id",
            "kind",
            "subject_id",
            "required",
            "critical",
            "reused",
            "relation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_id", _text(self.relation_id, "relation_id"))
        object.__setattr__(
            self, "kind", _kind_for_family(CampaignFamily.PROPERTY.value, self.kind)
        )
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        object.__setattr__(self, "reused", _bool(self.reused, "reused"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PROPERTY_RELATION_SCHEMA,
            "interface": PROPERTY_RELATION_INTERFACE,
            "relation_id": self.relation_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "required": self.required,
            "critical": self.critical,
            "reused": self.reused,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def relation_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["relation_cid"] = self.relation_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PropertyRelation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("relation_cid")
        if payload.pop("schema") != PROPERTY_RELATION_SCHEMA:
            raise AdversarialValidationError("unsupported PropertyRelation schema")
        if payload.pop("interface") != PROPERTY_RELATION_INTERFACE:
            raise AdversarialValidationError("unsupported PropertyRelation interface")
        result = cls(**payload)
        _verify_cid(claimed, result.relation_cid, "relation_cid")
        return result


@dataclass(frozen=True, slots=True)
class MetamorphicRelation:
    """Bounded relation between façade/extracted or alias/original observations."""

    relation_id: str
    kind: str
    subject_id: str
    required: bool = True
    critical: bool = True
    reused: bool = False

    interface: ClassVar[str] = METAMORPHIC_RELATION_INTERFACE
    schema: ClassVar[str] = METAMORPHIC_RELATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "relation_id",
            "kind",
            "subject_id",
            "required",
            "critical",
            "reused",
            "relation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_id", _text(self.relation_id, "relation_id"))
        object.__setattr__(
            self,
            "kind",
            _kind_for_family(CampaignFamily.METAMORPHIC.value, self.kind),
        )
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        object.__setattr__(self, "reused", _bool(self.reused, "reused"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": METAMORPHIC_RELATION_SCHEMA,
            "interface": METAMORPHIC_RELATION_INTERFACE,
            "relation_id": self.relation_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "required": self.required,
            "critical": self.critical,
            "reused": self.reused,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def relation_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["relation_cid"] = self.relation_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetamorphicRelation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("relation_cid")
        if payload.pop("schema") != METAMORPHIC_RELATION_SCHEMA:
            raise AdversarialValidationError("unsupported MetamorphicRelation schema")
        if payload.pop("interface") != METAMORPHIC_RELATION_INTERFACE:
            raise AdversarialValidationError("unsupported MetamorphicRelation interface")
        result = cls(**payload)
        _verify_cid(claimed, result.relation_cid, "relation_cid")
        return result


@dataclass(frozen=True, slots=True)
class MutantSpec:
    """Bounded, unapplied mutant of a moved boundary or façade."""

    mutant_id: str
    kind: str
    subject_id: str
    required: bool = True
    critical: bool = True
    reused: bool = False

    interface: ClassVar[str] = MUTANT_SPEC_INTERFACE
    schema: ClassVar[str] = MUTANT_SPEC_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "mutant_id",
            "kind",
            "subject_id",
            "required",
            "critical",
            "reused",
            "mutant_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "mutant_id", _text(self.mutant_id, "mutant_id"))
        object.__setattr__(
            self, "kind", _kind_for_family(CampaignFamily.MUTATION.value, self.kind)
        )
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        object.__setattr__(self, "reused", _bool(self.reused, "reused"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MUTANT_SPEC_SCHEMA,
            "interface": MUTANT_SPEC_INTERFACE,
            "mutant_id": self.mutant_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "required": self.required,
            "critical": self.critical,
            "reused": self.reused,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def mutant_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["mutant_cid"] = self.mutant_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutantSpec":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("mutant_cid")
        if payload.pop("schema") != MUTANT_SPEC_SCHEMA:
            raise AdversarialValidationError("unsupported MutantSpec schema")
        if payload.pop("interface") != MUTANT_SPEC_INTERFACE:
            raise AdversarialValidationError("unsupported MutantSpec interface")
        result = cls(**payload)
        _verify_cid(claimed, result.mutant_cid, "mutant_cid")
        return result


@dataclass(frozen=True, slots=True)
class AdversarialCase:
    """Bounded adversarial case for moved boundaries, plugins, or dynamics."""

    case_id: str
    kind: str
    subject_id: str
    required: bool = True
    critical: bool = True
    reused: bool = False

    interface: ClassVar[str] = ADVERSARIAL_CASE_INTERFACE
    schema: ClassVar[str] = ADVERSARIAL_CASE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "case_id",
            "kind",
            "subject_id",
            "required",
            "critical",
            "reused",
            "case_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "kind",
            _kind_for_family(CampaignFamily.ADVERSARIAL.value, self.kind),
        )
        object.__setattr__(self, "subject_id", _text(self.subject_id, "subject_id"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        object.__setattr__(self, "reused", _bool(self.reused, "reused"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ADVERSARIAL_CASE_SCHEMA,
            "interface": ADVERSARIAL_CASE_INTERFACE,
            "case_id": self.case_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "required": self.required,
            "critical": self.critical,
            "reused": self.reused,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def case_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["case_cid"] = self.case_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdversarialCase":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("case_cid")
        if payload.pop("schema") != ADVERSARIAL_CASE_SCHEMA:
            raise AdversarialValidationError("unsupported AdversarialCase schema")
        if payload.pop("interface") != ADVERSARIAL_CASE_INTERFACE:
            raise AdversarialValidationError("unsupported AdversarialCase interface")
        result = cls(**payload)
        _verify_cid(claimed, result.case_cid, "case_cid")
        return result


@dataclass(frozen=True, slots=True)
class RelationVerdict:
    """One property or metamorphic verdict. Evidence class is preserved."""

    family: str
    item_id: str
    kind: str
    evidence_class: str
    status: str
    evidence_cid: str = ""
    required: bool = True
    critical: bool = True
    full_suite_executed: bool = False

    interface: ClassVar[str] = RELATION_VERDICT_INTERFACE
    schema: ClassVar[str] = RELATION_VERDICT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "family",
            "item_id",
            "kind",
            "evidence_class",
            "status",
            "evidence_cid",
            "required",
            "critical",
            "full_suite_executed",
            "verdict_cid",
        }
    )

    def __post_init__(self) -> None:
        family = _family_value(self.family)
        if family not in {
            CampaignFamily.PROPERTY.value,
            CampaignFamily.METAMORPHIC.value,
        }:
            raise AdversarialValidationError("relation verdict family must be property or metamorphic")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "item_id", _text(self.item_id, "item_id"))
        object.__setattr__(self, "kind", _kind_for_family(family, self.kind))
        evidence = _evidence_class_value(self.evidence_class)
        status = _relation_status_value(self.status)
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        executed = _bool(self.full_suite_executed, "full_suite_executed")
        if executed and evidence != EvidenceClass.TEST.value:
            raise AdversarialValidationError(
                "full_suite_executed applies only to test evidence"
            )
        object.__setattr__(self, "full_suite_executed", executed)
        if status == RelationStatus.PASS.value:
            if not self.evidence_cid:
                raise AdversarialValidationError("passing verdict requires evidence_cid")
            assert_evidence_class_admitted(family, evidence)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": RELATION_VERDICT_SCHEMA,
            "interface": RELATION_VERDICT_INTERFACE,
            "family": self.family,
            "item_id": self.item_id,
            "kind": self.kind,
            "evidence_class": self.evidence_class,
            "status": self.status,
            "evidence_cid": self.evidence_cid,
            "required": self.required,
            "critical": self.critical,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationVerdict":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("verdict_cid")
        if payload.pop("schema") != RELATION_VERDICT_SCHEMA:
            raise AdversarialValidationError("unsupported RelationVerdict schema")
        if payload.pop("interface") != RELATION_VERDICT_INTERFACE:
            raise AdversarialValidationError("unsupported RelationVerdict interface")
        result = cls(**payload)
        _verify_cid(claimed, result.verdict_cid, "verdict_cid")
        return result


@dataclass(frozen=True, slots=True)
class MutantVerdict:
    """One mutant kill/survive/unknown verdict. Not completion authority."""

    mutant_id: str
    kind: str
    evidence_class: str
    status: str
    evidence_cid: str = ""
    required: bool = True
    critical: bool = True
    full_suite_executed: bool = False
    family: str = CampaignFamily.MUTATION.value

    interface: ClassVar[str] = MUTANT_VERDICT_INTERFACE
    schema: ClassVar[str] = MUTANT_VERDICT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "family",
            "mutant_id",
            "kind",
            "evidence_class",
            "status",
            "evidence_cid",
            "required",
            "critical",
            "full_suite_executed",
            "verdict_cid",
        }
    )

    def __post_init__(self) -> None:
        family = _family_value(self.family)
        if family != CampaignFamily.MUTATION.value:
            raise AdversarialValidationError("mutant verdict family must remain mutation")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "mutant_id", _text(self.mutant_id, "mutant_id"))
        object.__setattr__(self, "kind", _kind_for_family(family, self.kind))
        evidence = _evidence_class_value(self.evidence_class)
        status = _mutant_status_value(self.status)
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        executed = _bool(self.full_suite_executed, "full_suite_executed")
        if executed and evidence != EvidenceClass.TEST.value:
            raise AdversarialValidationError(
                "full_suite_executed applies only to test evidence"
            )
        object.__setattr__(self, "full_suite_executed", executed)
        if status == MutantStatus.KILLED.value:
            if not self.evidence_cid:
                raise AdversarialValidationError("killed mutant requires evidence_cid")
            assert_evidence_class_admitted(family, evidence)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MUTANT_VERDICT_SCHEMA,
            "interface": MUTANT_VERDICT_INTERFACE,
            "family": self.family,
            "mutant_id": self.mutant_id,
            "kind": self.kind,
            "evidence_class": self.evidence_class,
            "status": self.status,
            "evidence_cid": self.evidence_cid,
            "required": self.required,
            "critical": self.critical,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "MutantVerdict":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("verdict_cid")
        if payload.pop("schema") != MUTANT_VERDICT_SCHEMA:
            raise AdversarialValidationError("unsupported MutantVerdict schema")
        if payload.pop("interface") != MUTANT_VERDICT_INTERFACE:
            raise AdversarialValidationError("unsupported MutantVerdict interface")
        result = cls(**payload)
        _verify_cid(claimed, result.verdict_cid, "verdict_cid")
        return result


@dataclass(frozen=True, slots=True)
class AdversarialVerdict:
    """One adversarial-case verdict. Unknown required dynamics stay blockers."""

    case_id: str
    kind: str
    evidence_class: str
    status: str
    evidence_cid: str = ""
    required: bool = True
    critical: bool = True
    family: str = CampaignFamily.ADVERSARIAL.value

    interface: ClassVar[str] = ADVERSARIAL_VERDICT_INTERFACE
    schema: ClassVar[str] = ADVERSARIAL_VERDICT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "family",
            "case_id",
            "kind",
            "evidence_class",
            "status",
            "evidence_cid",
            "required",
            "critical",
            "verdict_cid",
        }
    )

    def __post_init__(self) -> None:
        family = _family_value(self.family)
        if family != CampaignFamily.ADVERSARIAL.value:
            raise AdversarialValidationError(
                "adversarial verdict family must remain adversarial"
            )
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "kind", _kind_for_family(family, self.kind))
        evidence = _evidence_class_value(self.evidence_class)
        status = _relation_status_value(self.status)
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "evidence_cid", _optional_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "critical", _bool(self.critical, "critical"))
        if status == RelationStatus.PASS.value:
            if not self.evidence_cid:
                raise AdversarialValidationError("passing verdict requires evidence_cid")
            assert_evidence_class_admitted(family, evidence)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ADVERSARIAL_VERDICT_SCHEMA,
            "interface": ADVERSARIAL_VERDICT_INTERFACE,
            "family": self.family,
            "case_id": self.case_id,
            "kind": self.kind,
            "evidence_class": self.evidence_class,
            "status": self.status,
            "evidence_cid": self.evidence_cid,
            "required": self.required,
            "critical": self.critical,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "AdversarialVerdict":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("verdict_cid")
        if payload.pop("schema") != ADVERSARIAL_VERDICT_SCHEMA:
            raise AdversarialValidationError("unsupported AdversarialVerdict schema")
        if payload.pop("interface") != ADVERSARIAL_VERDICT_INTERFACE:
            raise AdversarialValidationError("unsupported AdversarialVerdict interface")
        result = cls(**payload)
        _verify_cid(claimed, result.verdict_cid, "verdict_cid")
        return result


def _relation_from_mapping(
    item: Mapping[str, Any] | PropertyRelation | MetamorphicRelation,
    family: str,
) -> PropertyRelation | MetamorphicRelation:
    if isinstance(item, (PropertyRelation, MetamorphicRelation)):
        return item
    payload = dict(item)
    if family == CampaignFamily.PROPERTY.value:
        if "schema" in payload:
            return PropertyRelation.from_dict(payload)
        return PropertyRelation(
            relation_id=payload.get("relation_id") or payload.get("item_id"),
            kind=payload.get("kind"),
            subject_id=payload.get("subject_id"),
            required=payload.get("required", True),
            critical=payload.get("critical", True),
            reused=payload.get("reused", False),
        )
    if "schema" in payload:
        return MetamorphicRelation.from_dict(payload)
    return MetamorphicRelation(
        relation_id=payload.get("relation_id") or payload.get("item_id"),
        kind=payload.get("kind"),
        subject_id=payload.get("subject_id"),
        required=payload.get("required", True),
        critical=payload.get("critical", True),
        reused=payload.get("reused", False),
    )


def _mutant_from_mapping(item: Mapping[str, Any] | MutantSpec) -> MutantSpec:
    if isinstance(item, MutantSpec):
        return item
    payload = dict(item)
    if "schema" in payload:
        return MutantSpec.from_dict(payload)
    return MutantSpec(
        mutant_id=payload.get("mutant_id") or payload.get("item_id"),
        kind=payload.get("kind"),
        subject_id=payload.get("subject_id"),
        required=payload.get("required", True),
        critical=payload.get("critical", payload.get("kind") in DEFAULT_CRITICAL_MUTANT_KINDS),
        reused=payload.get("reused", False),
    )


def _case_from_mapping(item: Mapping[str, Any] | AdversarialCase) -> AdversarialCase:
    if isinstance(item, AdversarialCase):
        return item
    payload = dict(item)
    if "schema" in payload:
        return AdversarialCase.from_dict(payload)
    return AdversarialCase(
        case_id=payload.get("case_id") or payload.get("item_id"),
        kind=payload.get("kind"),
        subject_id=payload.get("subject_id"),
        required=payload.get("required", True),
        critical=payload.get("critical", True),
        reused=payload.get("reused", False),
    )


def _relation_verdict_from_mapping(
    item: Mapping[str, Any] | RelationVerdict,
    *,
    family: str | None = None,
    required: bool | None = None,
    critical: bool | None = None,
) -> RelationVerdict:
    if isinstance(item, RelationVerdict):
        if required is None and critical is None:
            return item
        return RelationVerdict(
            family=item.family,
            item_id=item.item_id,
            kind=item.kind,
            evidence_class=item.evidence_class,
            status=item.status,
            evidence_cid=item.evidence_cid,
            required=item.required if required is None else required,
            critical=item.critical if critical is None else critical,
            full_suite_executed=item.full_suite_executed,
        )
    payload = dict(item)
    covers = payload.pop("covers", None) or payload.pop("covers_all_families", None)
    if covers not in (None, False, (), []):
        raise AdversarialValidationError("evidence classes must not collapse")
    resolved_family = family or payload.get("family")
    return RelationVerdict(
        family=resolved_family,
        item_id=payload.get("item_id") or payload.get("relation_id"),
        kind=payload.get("kind"),
        evidence_class=payload.get("evidence_class"),
        status=payload.get("status"),
        evidence_cid=payload.get("evidence_cid", ""),
        required=payload.get("required", True) if required is None else required,
        critical=payload.get("critical", True) if critical is None else critical,
        full_suite_executed=payload.get("full_suite_executed", False),
    )


def _mutant_verdict_from_mapping(
    item: Mapping[str, Any] | MutantVerdict,
    *,
    required: bool | None = None,
    critical: bool | None = None,
) -> MutantVerdict:
    if isinstance(item, MutantVerdict):
        if required is None and critical is None:
            return item
        return MutantVerdict(
            mutant_id=item.mutant_id,
            kind=item.kind,
            evidence_class=item.evidence_class,
            status=item.status,
            evidence_cid=item.evidence_cid,
            required=item.required if required is None else required,
            critical=item.critical if critical is None else critical,
            full_suite_executed=item.full_suite_executed,
        )
    payload = dict(item)
    covers = payload.pop("covers", None) or payload.pop("covers_all_families", None)
    if covers not in (None, False, (), []):
        raise AdversarialValidationError("evidence classes must not collapse")
    return MutantVerdict(
        mutant_id=payload.get("mutant_id") or payload.get("item_id"),
        kind=payload.get("kind"),
        evidence_class=payload.get("evidence_class"),
        status=payload.get("status"),
        evidence_cid=payload.get("evidence_cid", ""),
        required=payload.get("required", True) if required is None else required,
        critical=payload.get("critical", True) if critical is None else critical,
        full_suite_executed=payload.get("full_suite_executed", False),
        family=payload.get("family", CampaignFamily.MUTATION.value),
    )


def _adversarial_verdict_from_mapping(
    item: Mapping[str, Any] | AdversarialVerdict,
    *,
    required: bool | None = None,
    critical: bool | None = None,
) -> AdversarialVerdict:
    if isinstance(item, AdversarialVerdict):
        if required is None and critical is None:
            return item
        return AdversarialVerdict(
            case_id=item.case_id,
            kind=item.kind,
            evidence_class=item.evidence_class,
            status=item.status,
            evidence_cid=item.evidence_cid,
            required=item.required if required is None else required,
            critical=item.critical if critical is None else critical,
        )
    payload = dict(item)
    covers = payload.pop("covers", None) or payload.pop("covers_all_families", None)
    if covers not in (None, False, (), []):
        raise AdversarialValidationError("evidence classes must not collapse")
    return AdversarialVerdict(
        case_id=payload.get("case_id") or payload.get("item_id"),
        kind=payload.get("kind"),
        evidence_class=payload.get("evidence_class"),
        status=payload.get("status"),
        evidence_cid=payload.get("evidence_cid", ""),
        required=payload.get("required", True) if required is None else required,
        critical=payload.get("critical", True) if critical is None else critical,
        family=payload.get("family", CampaignFamily.ADVERSARIAL.value),
    )


def generate_bounded_relations(
    *,
    write_paths: Sequence[str],
    facade_kinds: Sequence[str] = (),
    campaign_profile: CampaignProfile | None = None,
) -> tuple[tuple[PropertyRelation, ...], tuple[MetamorphicRelation, ...]]:
    """Nominate bounded property and metamorphic relations for façade/boundary moves."""

    profile = campaign_profile or CampaignProfile()
    subject = _subject_id(write_paths)
    properties: list[PropertyRelation] = []
    metamorphics: list[MetamorphicRelation] = []
    if profile.covers(CampaignFamily.PROPERTY.value):
        properties.append(
            PropertyRelation(
                relation_id="property:facade_reexport_identity",
                kind=PropertyKind.FACADE_REEXPORT_IDENTITY.value,
                subject_id=subject,
                required=profile.is_required(CampaignFamily.PROPERTY.value),
            )
        )
        properties.append(
            PropertyRelation(
                relation_id="property:boundary_assume_guarantee",
                kind=PropertyKind.BOUNDARY_ASSUME_GUARANTEE.value,
                subject_id=subject,
                required=profile.is_required(CampaignFamily.PROPERTY.value),
            )
        )
        if "star_export" in facade_kinds:
            properties.append(
                PropertyRelation(
                    relation_id="property:star_export_names",
                    kind=PropertyKind.STAR_EXPORT_NAMES.value,
                    subject_id=subject,
                    required=profile.is_required(CampaignFamily.PROPERTY.value),
                )
            )
    if profile.covers(CampaignFamily.METAMORPHIC.value):
        metamorphics.append(
            MetamorphicRelation(
                relation_id="metamorphic:facade_vs_extracted_call",
                kind=MetamorphicKind.FACADE_VS_EXTRACTED_CALL.value,
                subject_id=subject,
                required=profile.is_required(CampaignFamily.METAMORPHIC.value),
            )
        )
        metamorphics.append(
            MetamorphicRelation(
                relation_id="metamorphic:import_path_alias",
                kind=MetamorphicKind.IMPORT_PATH_ALIAS.value,
                subject_id=subject,
                required=profile.is_required(CampaignFamily.METAMORPHIC.value),
            )
        )
        if "registry" in facade_kinds:
            metamorphics.append(
                MetamorphicRelation(
                    relation_id="metamorphic:registration_order",
                    kind=MetamorphicKind.REGISTRATION_ORDER.value,
                    subject_id=subject,
                    required=profile.is_required(CampaignFamily.METAMORPHIC.value),
                )
            )
        if "serialization" in facade_kinds or "pickle" in facade_kinds:
            metamorphics.append(
                MetamorphicRelation(
                    relation_id="metamorphic:serialization_roundtrip",
                    kind=MetamorphicKind.SERIALIZATION_ROUNDTRIP.value,
                    subject_id=subject,
                    required=profile.is_required(CampaignFamily.METAMORPHIC.value),
                )
            )
    if len(properties) > MAX_RELATIONS or len(metamorphics) > MAX_RELATIONS:
        raise AdversarialValidationError("generated relations exceed bound")
    return tuple(properties), tuple(metamorphics)


def generate_bounded_mutants(
    *,
    write_paths: Sequence[str],
    facade_kinds: Sequence[str] = (),
    campaign_profile: CampaignProfile | None = None,
) -> tuple[MutantSpec, ...]:
    """Nominate bounded unapplied mutants for moved boundaries and façades."""

    profile = campaign_profile or CampaignProfile()
    if not profile.covers(CampaignFamily.MUTATION.value):
        return ()
    subject = _subject_id(write_paths)
    required = profile.is_required(CampaignFamily.MUTATION.value)
    mutants = [
        MutantSpec(
            mutant_id=f"mutant:{kind}",
            kind=kind,
            subject_id=subject,
            required=required,
            critical=True,
        )
        for kind in DEFAULT_CRITICAL_MUTANT_KINDS
    ]
    extra: list[str] = []
    if "registry" in facade_kinds:
        extra.append(MutantKind.SKIP_REGISTRATION.value)
    if "wrapper" in facade_kinds:
        extra.append(MutantKind.DROP_WRAPPER.value)
    if "deprecation" in facade_kinds:
        extra.append(MutantKind.SUPPRESS_DEPRECATION.value)
    for kind in extra:
        mutants.append(
            MutantSpec(
                mutant_id=f"mutant:{kind}",
                kind=kind,
                subject_id=subject,
                required=required,
                critical=kind != MutantKind.SUPPRESS_DEPRECATION.value,
            )
        )
    if len(mutants) > MAX_MUTANTS:
        raise AdversarialValidationError("generated mutants exceed bound")
    return tuple(mutants)


def generate_bounded_adversarial_cases(
    *,
    write_paths: Sequence[str],
    facade_kinds: Sequence[str] = (),
    campaign_profile: CampaignProfile | None = None,
) -> tuple[AdversarialCase, ...]:
    """Nominate bounded adversarial cases, including unknown required dynamics."""

    profile = campaign_profile or CampaignProfile()
    if not profile.covers(CampaignFamily.ADVERSARIAL.value):
        return ()
    subject = _subject_id(write_paths)
    required = profile.is_required(CampaignFamily.ADVERSARIAL.value)
    cases = [
        AdversarialCase(
            case_id="adversarial:required_unresolved_dynamic",
            kind=AdversarialKind.REQUIRED_UNRESOLVED_DYNAMIC.value,
            subject_id=subject,
            required=required,
            critical=True,
        ),
        AdversarialCase(
            case_id="adversarial:unknown_dynamic_import",
            kind=AdversarialKind.UNKNOWN_DYNAMIC_IMPORT.value,
            subject_id=subject,
            required=required,
            critical=True,
        ),
    ]
    extras = {
        "plugin": AdversarialKind.PLUGIN_ENTRY_POINT_DRIFT.value,
        "registry": AdversarialKind.REGISTRY_ALIAS_COLLISION.value,
        "pickle": AdversarialKind.PICKLE_QUALNAME_BREAK.value,
        "serialization": AdversarialKind.PICKLE_QUALNAME_BREAK.value,
        "patch_target": AdversarialKind.PATCH_TARGET_MISS.value,
        "star_export": AdversarialKind.STAR_EXPORT_LEAK.value,
    }
    seen = {item.kind for item in cases}
    for facade_kind, adversarial_kind in extras.items():
        if facade_kind in facade_kinds and adversarial_kind not in seen:
            cases.append(
                AdversarialCase(
                    case_id=f"adversarial:{adversarial_kind}",
                    kind=adversarial_kind,
                    subject_id=subject,
                    required=required,
                    critical=True,
                )
            )
            seen.add(adversarial_kind)
    if len(cases) > MAX_ADVERSARIAL:
        raise AdversarialValidationError("generated adversarial cases exceed bound")
    return tuple(cases)


def _merge_relations(
    declared: Sequence[Any],
    generated: Sequence[PropertyRelation] | Sequence[MetamorphicRelation],
    family: str,
) -> tuple[Any, ...]:
    merged: list[Any] = []
    seen: set[str] = set()
    for item in declared:
        relation = _relation_from_mapping(item, family)
        object.__setattr__(relation, "reused", True)
        if relation.relation_id in seen:
            raise AdversarialValidationError(
                f"duplicate {family} relation: {relation.relation_id}"
            )
        seen.add(relation.relation_id)
        merged.append(relation)
    for item in generated:
        if item.relation_id in seen or item.kind in {
            relation.kind for relation in merged
        }:
            continue
        merged.append(item)
    limit = MAX_RELATIONS
    if len(merged) > limit:
        raise AdversarialValidationError(f"{family} relations exceed bound")
    return tuple(merged)


def _merge_mutants(
    declared: Sequence[Any],
    generated: Sequence[MutantSpec],
) -> tuple[MutantSpec, ...]:
    merged: list[MutantSpec] = []
    seen: set[str] = set()
    for item in declared:
        mutant = _mutant_from_mapping(item)
        object.__setattr__(mutant, "reused", True)
        if mutant.mutant_id in seen:
            raise AdversarialValidationError(f"duplicate mutant: {mutant.mutant_id}")
        seen.add(mutant.mutant_id)
        merged.append(mutant)
    for item in generated:
        if item.mutant_id in seen or item.kind in {mutant.kind for mutant in merged}:
            continue
        merged.append(item)
    if len(merged) > MAX_MUTANTS:
        raise AdversarialValidationError("mutants exceed bound")
    return tuple(merged)


def _merge_cases(
    declared: Sequence[Any],
    generated: Sequence[AdversarialCase],
) -> tuple[AdversarialCase, ...]:
    merged: list[AdversarialCase] = []
    seen: set[str] = set()
    for item in declared:
        case = _case_from_mapping(item)
        object.__setattr__(case, "reused", True)
        if case.case_id in seen:
            raise AdversarialValidationError(f"duplicate adversarial case: {case.case_id}")
        seen.add(case.case_id)
        merged.append(case)
    for item in generated:
        if item.case_id in seen or item.kind in {case.kind for case in merged}:
            continue
        merged.append(item)
    if len(merged) > MAX_ADVERSARIAL:
        raise AdversarialValidationError("adversarial cases exceed bound")
    return tuple(merged)


def _derive_property_verdicts(
    properties: Sequence[PropertyRelation],
    original_sources: Mapping[str, str] | None,
    candidate_sources: Mapping[str, str] | None,
    existing_ids: set[str],
) -> list[RelationVerdict]:
    if original_sources is None and candidate_sources is None:
        return []
    if original_sources is None or candidate_sources is None:
        raise AdversarialValidationError(
            "property derivation requires original and candidate raw sources"
        )
    try:
        original_trees = _parse_sources(original_sources, "original_sources")
        candidate_trees = _parse_sources(candidate_sources, "candidate_sources")
        parse_failed = False
    except AdversarialValidationError as exc:
        if "failed to parse" not in str(exc):
            raise
        parse_failed = True
        original_trees = {}
        candidate_trees = {}
    original_names: set[str] = set()
    candidate_names: set[str] = set()
    if not parse_failed:
        for tree in original_trees.values():
            original_names.update(_public_names(tree))
        for tree in candidate_trees.values():
            candidate_names.update(_public_names(tree))
    derived: list[RelationVerdict] = []
    for relation in properties:
        if relation.relation_id in existing_ids:
            continue
        if relation.kind != PropertyKind.FACADE_REEXPORT_IDENTITY.value:
            continue
        if parse_failed:
            status = RelationStatus.FAIL.value
        elif original_names <= candidate_names:
            status = RelationStatus.PASS.value
        else:
            status = RelationStatus.FAIL.value
        evidence_cid = cid_for_dag_json(
            {
                "family": CampaignFamily.PROPERTY.value,
                "kind": relation.kind,
                "status": status,
                "original_names": sorted(original_names),
                "candidate_names": sorted(candidate_names),
                "parsed": not parse_failed,
            }
        )
        derived.append(
            RelationVerdict(
                family=CampaignFamily.PROPERTY.value,
                item_id=relation.relation_id,
                kind=relation.kind,
                evidence_class=EvidenceClass.EXACT_STATIC_FACT.value,
                status=status,
                evidence_cid=evidence_cid,
                required=relation.required,
                critical=relation.critical,
            )
        )
    return derived


def _derive_adversarial_verdicts(
    cases: Sequence[AdversarialCase],
    graph: Mapping[str, Any] | None,
    frontier: Mapping[str, Any] | None,
    existing_ids: set[str],
) -> list[AdversarialVerdict]:
    if graph is None and frontier is None:
        return []
    derived: list[AdversarialVerdict] = []
    unresolved = unresolved_required_dynamics(graph, frontier)
    dynamic_import = unknown_dynamic_import_present(graph, frontier)
    for case in cases:
        if case.case_id in existing_ids:
            continue
        if case.kind == AdversarialKind.REQUIRED_UNRESOLVED_DYNAMIC.value:
            status = (
                RelationStatus.UNSUPPORTED.value
                if unresolved
                else RelationStatus.PASS.value
            )
        elif case.kind == AdversarialKind.UNKNOWN_DYNAMIC_IMPORT.value:
            status = (
                RelationStatus.UNSUPPORTED.value
                if dynamic_import
                else RelationStatus.PASS.value
            )
        else:
            continue
        evidence_cid = ""
        if status == RelationStatus.PASS.value:
            evidence_cid = cid_for_dag_json(
                {
                    "family": CampaignFamily.ADVERSARIAL.value,
                    "kind": case.kind,
                    "unresolved_required_dynamics": unresolved,
                    "unknown_dynamic_import": dynamic_import,
                }
            )
        derived.append(
            AdversarialVerdict(
                case_id=case.case_id,
                kind=case.kind,
                evidence_class=EvidenceClass.EXACT_STATIC_FACT.value,
                status=status,
                evidence_cid=evidence_cid,
                required=case.required,
                critical=case.critical,
            )
        )
    return derived


def _index_unique(items: Sequence[Any], attr: str, name: str) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for item in items:
        key = getattr(item, attr)
        if key in indexed:
            raise AdversarialValidationError(f"duplicate {name}: {key}")
        indexed[key] = item
    return indexed


@dataclass(frozen=True, slots=True)
class MutationCampaignRequest:
    """Bound SPAR-025/026 campaign plus reused/generated relations and mutants."""

    tree_id: str
    packet_cid: str
    wave_receipt_cid: str
    selection_cid: str
    original_source_cids: Sequence[str]
    candidate_source_cids: Sequence[str]
    write_paths: Sequence[str]
    validation_commands: Sequence[str]
    campaign_profile: CampaignProfile
    properties: Sequence[PropertyRelation]
    metamorphics: Sequence[MetamorphicRelation]
    mutants: Sequence[MutantSpec]
    adversarial_cases: Sequence[AdversarialCase]
    relation_verdicts: Sequence[RelationVerdict]
    mutant_verdicts: Sequence[MutantVerdict]
    adversarial_verdicts: Sequence[AdversarialVerdict]
    full_suite_required: bool = False
    raw_source_required: bool = True
    validator_is_nomination_only: bool = True
    claims_general_python_equivalence: bool = False
    collapse_evidence_classes: bool = False

    interface: ClassVar[str] = MUTATION_CAMPAIGN_REQUEST_INTERFACE
    schema: ClassVar[str] = MUTATION_CAMPAIGN_REQUEST_SCHEMA
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
            "campaign_profile",
            "properties",
            "metamorphics",
            "mutants",
            "adversarial_cases",
            "relation_verdicts",
            "mutant_verdicts",
            "adversarial_verdicts",
            "full_suite_required",
            "raw_source_required",
            "validator_is_nomination_only",
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
            raise AdversarialValidationError("raw source required")
        if not candidates:
            raise AdversarialValidationError("candidate source_cids are required")
        object.__setattr__(self, "original_source_cids", originals)
        object.__setattr__(self, "candidate_source_cids", candidates)
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self, "validation_commands", _commands(self.validation_commands)
        )
        profile = _coerce_profile(self.campaign_profile)
        object.__setattr__(self, "campaign_profile", profile)
        properties = tuple(
            _relation_from_mapping(item, CampaignFamily.PROPERTY.value)
            for item in self.properties
        )
        metamorphics = tuple(
            _relation_from_mapping(item, CampaignFamily.METAMORPHIC.value)
            for item in self.metamorphics
        )
        mutants = tuple(_mutant_from_mapping(item) for item in self.mutants)
        cases = tuple(_case_from_mapping(item) for item in self.adversarial_cases)
        _index_unique(properties, "relation_id", "property relation")
        _index_unique(metamorphics, "relation_id", "metamorphic relation")
        _index_unique(mutants, "mutant_id", "mutant")
        _index_unique(cases, "case_id", "adversarial case")
        for collection, family in (
            (properties, CampaignFamily.PROPERTY.value),
            (metamorphics, CampaignFamily.METAMORPHIC.value),
            (mutants, CampaignFamily.MUTATION.value),
            (cases, CampaignFamily.ADVERSARIAL.value),
        ):
            if collection and not profile.covers(family):
                raise AdversarialValidationError(
                    f"{family} is outside the campaign profile"
                )
        object.__setattr__(self, "properties", properties)
        object.__setattr__(self, "metamorphics", metamorphics)
        object.__setattr__(self, "mutants", mutants)
        object.__setattr__(self, "adversarial_cases", cases)
        relation_verdicts = tuple(
            _relation_verdict_from_mapping(item) for item in self.relation_verdicts
        )
        mutant_verdicts = tuple(
            _mutant_verdict_from_mapping(item) for item in self.mutant_verdicts
        )
        adversarial_verdicts = tuple(
            _adversarial_verdict_from_mapping(item) for item in self.adversarial_verdicts
        )
        _index_unique(relation_verdicts, "item_id", "relation verdict")
        _index_unique(mutant_verdicts, "mutant_id", "mutant verdict")
        _index_unique(adversarial_verdicts, "case_id", "adversarial verdict")
        object.__setattr__(self, "relation_verdicts", relation_verdicts)
        object.__setattr__(self, "mutant_verdicts", mutant_verdicts)
        object.__setattr__(self, "adversarial_verdicts", adversarial_verdicts)
        collapse_items = [
            {
                "family": item.family,
                "evidence_class": item.evidence_class,
                "evidence_cid": item.evidence_cid,
                "status": item.status,
            }
            for item in (*relation_verdicts, *mutant_verdicts, *adversarial_verdicts)
        ]
        assert_evidence_classes_not_collapsed(collapse_items)
        object.__setattr__(
            self, "full_suite_required", _bool(self.full_suite_required, "full_suite_required")
        )
        if _bool(self.raw_source_required, "raw_source_required") is not True:
            raise AdversarialValidationError("raw_source_required cannot be disabled")
        if (
            _bool(self.validator_is_nomination_only, "validator_is_nomination_only")
            is not True
        ):
            raise AdversarialValidationError("validator must remain nomination_only")
        if (
            _bool(
                self.claims_general_python_equivalence,
                "claims_general_python_equivalence",
            )
            is not False
        ):
            raise AdversarialValidationError("general Python equivalence is not claimed")
        if (
            _bool(self.collapse_evidence_classes, "collapse_evidence_classes")
            is not False
        ):
            raise AdversarialValidationError("evidence classes must not collapse")
        object.__setattr__(self, "raw_source_required", True)
        object.__setattr__(self, "validator_is_nomination_only", True)
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
            "schema": MUTATION_CAMPAIGN_REQUEST_SCHEMA,
            "interface": MUTATION_CAMPAIGN_REQUEST_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "wave_receipt_cid": self.wave_receipt_cid,
            "selection_cid": self.selection_cid,
            "original_source_cids": list(self.original_source_cids),
            "candidate_source_cids": list(self.candidate_source_cids),
            "write_paths": list(self.write_paths),
            "validation_commands": list(self.validation_commands),
            "campaign_profile": self.campaign_profile.to_dict(),
            "properties": [item.to_dict() for item in self.properties],
            "metamorphics": [item.to_dict() for item in self.metamorphics],
            "mutants": [item.to_dict() for item in self.mutants],
            "adversarial_cases": [item.to_dict() for item in self.adversarial_cases],
            "relation_verdicts": [item.to_dict() for item in self.relation_verdicts],
            "mutant_verdicts": [item.to_dict() for item in self.mutant_verdicts],
            "adversarial_verdicts": [item.to_dict() for item in self.adversarial_verdicts],
            "full_suite_required": self.full_suite_required,
            "raw_source_required": True,
            "validator_is_nomination_only": True,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "MutationCampaignRequest":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("request_cid")
        if payload.pop("schema") != MUTATION_CAMPAIGN_REQUEST_SCHEMA:
            raise AdversarialValidationError(
                "unsupported MutationCampaignRequest schema"
            )
        if payload.pop("interface") != MUTATION_CAMPAIGN_REQUEST_INTERFACE:
            raise AdversarialValidationError(
                "unsupported MutationCampaignRequest interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("raw_source_required") is not True:
            raise AdversarialValidationError("raw_source_required cannot be disabled")
        if payload.pop("validator_is_nomination_only") is not True:
            raise AdversarialValidationError("validator must remain nomination_only")
        if payload.pop("claims_general_python_equivalence") is not False:
            raise AdversarialValidationError("general Python equivalence is not claimed")
        if payload.pop("collapse_evidence_classes") is not False:
            raise AdversarialValidationError("evidence classes must not collapse")
        profile = CampaignProfile.from_dict(payload.pop("campaign_profile"))
        properties = tuple(
            PropertyRelation.from_dict(item) for item in payload.pop("properties")
        )
        metamorphics = tuple(
            MetamorphicRelation.from_dict(item) for item in payload.pop("metamorphics")
        )
        mutants = tuple(MutantSpec.from_dict(item) for item in payload.pop("mutants"))
        cases = tuple(
            AdversarialCase.from_dict(item) for item in payload.pop("adversarial_cases")
        )
        relation_verdicts = tuple(
            RelationVerdict.from_dict(item) for item in payload.pop("relation_verdicts")
        )
        mutant_verdicts = tuple(
            MutantVerdict.from_dict(item) for item in payload.pop("mutant_verdicts")
        )
        adversarial_verdicts = tuple(
            AdversarialVerdict.from_dict(item)
            for item in payload.pop("adversarial_verdicts")
        )
        result = cls(
            campaign_profile=profile,
            properties=properties,
            metamorphics=metamorphics,
            mutants=mutants,
            adversarial_cases=cases,
            relation_verdicts=relation_verdicts,
            mutant_verdicts=mutant_verdicts,
            adversarial_verdicts=adversarial_verdicts,
            **payload,
        )
        _verify_cid(claimed, result.request_cid, "request_cid")
        return result


def _missing_relation_verdict(item: PropertyRelation | MetamorphicRelation, family: str) -> RelationVerdict:
    return RelationVerdict(
        family=family,
        item_id=item.relation_id,
        kind=item.kind,
        evidence_class=next(iter(admitted_evidence_classes(family))),
        status=(
            RelationStatus.MISSING.value
            if item.required
            else RelationStatus.NOT_IN_PROFILE.value
        ),
        evidence_cid="",
        required=item.required,
        critical=item.critical,
    )


def _missing_mutant_verdict(item: MutantSpec) -> MutantVerdict:
    return MutantVerdict(
        mutant_id=item.mutant_id,
        kind=item.kind,
        evidence_class=EvidenceClass.TEST.value,
        status=(
            MutantStatus.MISSING.value
            if item.required
            else MutantStatus.NOT_IN_PROFILE.value
        ),
        evidence_cid="",
        required=item.required,
        critical=item.critical,
    )


def _missing_adversarial_verdict(item: AdversarialCase) -> AdversarialVerdict:
    return AdversarialVerdict(
        case_id=item.case_id,
        kind=item.kind,
        evidence_class=EvidenceClass.EXACT_STATIC_FACT.value,
        status=(
            RelationStatus.MISSING.value
            if item.required
            else RelationStatus.NOT_IN_PROFILE.value
        ),
        evidence_cid="",
        required=item.required,
        critical=item.critical,
    )


def _evaluate_campaign(
    request: MutationCampaignRequest,
) -> tuple[
    tuple[RelationVerdict, ...],
    tuple[MutantVerdict, ...],
    tuple[AdversarialVerdict, ...],
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
]:
    relation_index = _index_unique(request.relation_verdicts, "item_id", "relation verdict")
    mutant_index = _index_unique(request.mutant_verdicts, "mutant_id", "mutant verdict")
    case_index = _index_unique(request.adversarial_verdicts, "case_id", "adversarial verdict")
    relations: list[RelationVerdict] = []
    for family, items in (
        (CampaignFamily.PROPERTY.value, request.properties),
        (CampaignFamily.METAMORPHIC.value, request.metamorphics),
    ):
        for item in items:
            found = relation_index.get(item.relation_id)
            if found is None:
                relations.append(_missing_relation_verdict(item, family))
                continue
            status = found.status
            if (
                request.full_suite_required
                and found.evidence_class == EvidenceClass.TEST.value
                and status == RelationStatus.PASS.value
                and found.full_suite_executed is not True
            ):
                relations.append(
                    RelationVerdict(
                        family=family,
                        item_id=item.relation_id,
                        kind=item.kind,
                        evidence_class=found.evidence_class,
                        status=RelationStatus.MISSING.value,
                        evidence_cid=found.evidence_cid,
                        required=item.required,
                        critical=item.critical,
                        full_suite_executed=False,
                    )
                )
                continue
            relations.append(
                RelationVerdict(
                    family=family,
                    item_id=item.relation_id,
                    kind=item.kind,
                    evidence_class=found.evidence_class,
                    status=status,
                    evidence_cid=found.evidence_cid,
                    required=item.required,
                    critical=item.critical,
                    full_suite_executed=found.full_suite_executed,
                )
            )
    mutants: list[MutantVerdict] = []
    for item in request.mutants:
        found = mutant_index.get(item.mutant_id)
        if found is None:
            mutants.append(_missing_mutant_verdict(item))
            continue
        status = found.status
        if (
            request.full_suite_required
            and found.evidence_class == EvidenceClass.TEST.value
            and status == MutantStatus.KILLED.value
            and found.full_suite_executed is not True
        ):
            mutants.append(
                MutantVerdict(
                    mutant_id=item.mutant_id,
                    kind=item.kind,
                    evidence_class=found.evidence_class,
                    status=MutantStatus.MISSING.value,
                    evidence_cid=found.evidence_cid,
                    required=item.required,
                    critical=item.critical,
                    full_suite_executed=False,
                )
            )
            continue
        mutants.append(
            MutantVerdict(
                mutant_id=item.mutant_id,
                kind=item.kind,
                evidence_class=found.evidence_class,
                status=status,
                evidence_cid=found.evidence_cid,
                required=item.required,
                critical=item.critical,
                full_suite_executed=found.full_suite_executed,
            )
        )
    cases: list[AdversarialVerdict] = []
    for item in request.adversarial_cases:
        found = case_index.get(item.case_id)
        if found is None:
            cases.append(_missing_adversarial_verdict(item))
            continue
        cases.append(
            AdversarialVerdict(
                case_id=item.case_id,
                kind=item.kind,
                evidence_class=found.evidence_class,
                status=found.status,
                evidence_cid=found.evidence_cid,
                required=item.required,
                critical=item.critical,
            )
        )

    collapse_items = [
        {
            "family": item.family,
            "evidence_class": item.evidence_class,
            "evidence_cid": item.evidence_cid,
            "status": item.status,
        }
        for item in (*relations, *mutants, *cases)
    ]
    assert_evidence_classes_not_collapsed(collapse_items)

    required_relations = [item for item in relations if item.required]
    required_mutants = [item for item in mutants if item.required]
    required_cases = [item for item in cases if item.required]
    critical_survivors = tuple(
        item.mutant_id
        for item in mutants
        if item.required and item.critical and item.status == MutantStatus.SURVIVED.value
    )
    unknown_required = tuple(
        [
            *[
                item.item_id
                for item in required_relations
                if item.critical and item.status == RelationStatus.UNSUPPORTED.value
            ],
            *[
                item.mutant_id
                for item in required_mutants
                if item.critical
                and item.status in {MutantStatus.UNKNOWN.value, MutantStatus.UNSUPPORTED.value}
            ],
            *[
                item.case_id
                for item in required_cases
                if item.critical and item.status == RelationStatus.UNSUPPORTED.value
            ],
        ]
    )

    if unknown_required:
        return (
            tuple(relations),
            tuple(mutants),
            tuple(cases),
            CampaignStatus.UNSUPPORTED.value,
            "unknown required dynamics: " + ",".join(unknown_required),
            critical_survivors,
            unknown_required,
        )
    missing = tuple(
        [
            *[
                item.item_id
                for item in required_relations
                if item.status == RelationStatus.MISSING.value
            ],
            *[
                item.mutant_id
                for item in required_mutants
                if item.status == MutantStatus.MISSING.value
            ],
            *[
                item.case_id
                for item in required_cases
                if item.status == RelationStatus.MISSING.value
            ],
        ]
    )
    if missing:
        return (
            tuple(relations),
            tuple(mutants),
            tuple(cases),
            CampaignStatus.INCOMPLETE.value,
            "missing required campaign item: " + ",".join(missing),
            critical_survivors,
            unknown_required,
        )
    if critical_survivors:
        return (
            tuple(relations),
            tuple(mutants),
            tuple(cases),
            CampaignStatus.REJECTED.value,
            "critical survivor: " + ",".join(critical_survivors),
            critical_survivors,
            unknown_required,
        )
    failed = tuple(
        [
            *[
                item.item_id
                for item in required_relations
                if item.status == RelationStatus.FAIL.value
            ],
            *[
                item.case_id
                for item in required_cases
                if item.status == RelationStatus.FAIL.value
            ],
        ]
    )
    if failed:
        return (
            tuple(relations),
            tuple(mutants),
            tuple(cases),
            CampaignStatus.REJECTED.value,
            "failed required campaign item: " + ",".join(failed),
            critical_survivors,
            unknown_required,
        )
    if any(
        item.status != RelationStatus.PASS.value
        and item.status != RelationStatus.NOT_IN_PROFILE.value
        for item in required_relations
    ) or any(
        item.status != MutantStatus.KILLED.value
        and item.status != MutantStatus.NOT_IN_PROFILE.value
        and not (item.status == MutantStatus.SURVIVED.value and item.critical is False)
        for item in required_mutants
    ) or any(
        item.status != RelationStatus.PASS.value
        and item.status != RelationStatus.NOT_IN_PROFILE.value
        for item in required_cases
    ):
        return (
            tuple(relations),
            tuple(mutants),
            tuple(cases),
            CampaignStatus.INCOMPLETE.value,
            "required campaign items did not all pass",
            critical_survivors,
            unknown_required,
        )
    return (
        tuple(relations),
        tuple(mutants),
        tuple(cases),
        CampaignStatus.VALIDATED.value,
        "",
        critical_survivors,
        unknown_required,
    )


@dataclass(frozen=True, slots=True)
class MutationCampaignReceipt:
    """Campaign receipt. Nomination-only; not completion or write authority."""

    tree_id: str
    request_cid: str
    packet_cid: str
    wave_receipt_cid: str
    selection_cid: str
    relation_verdicts: Sequence[RelationVerdict]
    mutant_verdicts: Sequence[MutantVerdict]
    adversarial_verdicts: Sequence[AdversarialVerdict]
    status: str
    terminal_reason: str = ""
    critical_survivors: Sequence[str] = ()
    unknown_required_dynamics: Sequence[str] = ()
    mutated: bool = False
    deterministic: bool = True
    raw_source_required: bool = True
    validator_is_nomination_only: bool = True
    claims_general_python_equivalence: bool = False
    collapse_evidence_classes: bool = False
    analyzer_id: str = ANALYZER_ID

    interface: ClassVar[str] = MUTATION_CAMPAIGN_RECEIPT_INTERFACE
    schema: ClassVar[str] = MUTATION_CAMPAIGN_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "request_cid",
            "packet_cid",
            "wave_receipt_cid",
            "selection_cid",
            "relation_verdicts",
            "mutant_verdicts",
            "adversarial_verdicts",
            "status",
            "terminal_reason",
            "conjunction_passed",
            "critical_survivors",
            "unknown_required_dynamics",
            "mutated",
            "deterministic",
            "raw_source_required",
            "validator_is_nomination_only",
            "claims_general_python_equivalence",
            "collapse_evidence_classes",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "analyzer_id",
            "receipt_cid",
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
        relations = tuple(
            item if isinstance(item, RelationVerdict) else RelationVerdict.from_dict(item)
            for item in self.relation_verdicts
        )
        mutants = tuple(
            item if isinstance(item, MutantVerdict) else MutantVerdict.from_dict(item)
            for item in self.mutant_verdicts
        )
        cases = tuple(
            item
            if isinstance(item, AdversarialVerdict)
            else AdversarialVerdict.from_dict(item)
            for item in self.adversarial_verdicts
        )
        object.__setattr__(self, "relation_verdicts", relations)
        object.__setattr__(self, "mutant_verdicts", mutants)
        object.__setattr__(self, "adversarial_verdicts", cases)
        collapse_items = [
            {
                "family": item.family,
                "evidence_class": item.evidence_class,
                "evidence_cid": item.evidence_cid,
                "status": item.status,
            }
            for item in (*relations, *mutants, *cases)
        ]
        assert_evidence_classes_not_collapsed(collapse_items)
        status = _campaign_status_value(self.status)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "terminal_reason",
            _text(self.terminal_reason, "terminal_reason", empty=True),
        )
        if status == CampaignStatus.VALIDATED.value and self.terminal_reason:
            raise AdversarialValidationError("validated result cannot carry a terminal reason")
        if status != CampaignStatus.VALIDATED.value and not self.terminal_reason:
            raise AdversarialValidationError("typed terminal requires a reason")
        object.__setattr__(
            self,
            "critical_survivors",
            _ordered_text(list(self.critical_survivors), "critical_survivors", limit=MAX_MUTANTS)
            if not isinstance(self.critical_survivors, tuple)
            or any(type(item) is not str for item in self.critical_survivors)
            else tuple(self.critical_survivors),
        )
        object.__setattr__(
            self,
            "unknown_required_dynamics",
            _ordered_text(
                list(self.unknown_required_dynamics),
                "unknown_required_dynamics",
                limit=MAX_ADVERSARIAL + MAX_MUTANTS + MAX_RELATIONS,
            )
            if not isinstance(self.unknown_required_dynamics, tuple)
            or any(type(item) is not str for item in self.unknown_required_dynamics)
            else tuple(self.unknown_required_dynamics),
        )
        if status == CampaignStatus.VALIDATED.value and (
            self.critical_survivors or self.unknown_required_dynamics
        ):
            raise AdversarialValidationError(
                "validated campaign cannot retain critical survivors or unknown required dynamics"
            )
        if _bool(self.mutated, "mutated") is not False:
            raise AdversarialValidationError("validator cannot mutate")
        if _bool(self.deterministic, "deterministic") is not True:
            raise AdversarialValidationError("validator must remain deterministic")
        if _bool(self.raw_source_required, "raw_source_required") is not True:
            raise AdversarialValidationError("raw_source_required cannot be disabled")
        if (
            _bool(self.validator_is_nomination_only, "validator_is_nomination_only")
            is not True
        ):
            raise AdversarialValidationError("validator must remain nomination_only")
        if (
            _bool(
                self.claims_general_python_equivalence,
                "claims_general_python_equivalence",
            )
            is not False
        ):
            raise AdversarialValidationError("general Python equivalence is not claimed")
        if (
            _bool(self.collapse_evidence_classes, "collapse_evidence_classes")
            is not False
        ):
            raise AdversarialValidationError("evidence classes must not collapse")
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise AdversarialValidationError("receipt analyzer_id must remain SPAR-029")
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)
        object.__setattr__(self, "raw_source_required", True)
        object.__setattr__(self, "validator_is_nomination_only", True)
        object.__setattr__(self, "claims_general_python_equivalence", False)
        object.__setattr__(self, "collapse_evidence_classes", False)
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)

    @property
    def conjunction_passed(self) -> bool:
        return self.status == CampaignStatus.VALIDATED.value

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
            CampaignStatus.UNSUPPORTED.value,
            CampaignStatus.INCOMPLETE.value,
            CampaignStatus.REJECTED.value,
        }

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": MUTATION_CAMPAIGN_RECEIPT_SCHEMA,
            "interface": MUTATION_CAMPAIGN_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "request_cid": self.request_cid,
            "packet_cid": self.packet_cid,
            "wave_receipt_cid": self.wave_receipt_cid,
            "selection_cid": self.selection_cid,
            "relation_verdicts": [item.to_dict() for item in self.relation_verdicts],
            "mutant_verdicts": [item.to_dict() for item in self.mutant_verdicts],
            "adversarial_verdicts": [item.to_dict() for item in self.adversarial_verdicts],
            "status": self.status,
            "terminal_reason": self.terminal_reason,
            "conjunction_passed": self.conjunction_passed,
            "critical_survivors": list(self.critical_survivors),
            "unknown_required_dynamics": list(self.unknown_required_dynamics),
            "mutated": False,
            "deterministic": True,
            "raw_source_required": True,
            "validator_is_nomination_only": True,
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
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MutationCampaignReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != MUTATION_CAMPAIGN_RECEIPT_SCHEMA:
            raise AdversarialValidationError(
                "unsupported MutationCampaignReceipt schema"
            )
        if payload.pop("interface") != MUTATION_CAMPAIGN_RECEIPT_INTERFACE:
            raise AdversarialValidationError(
                "unsupported MutationCampaignReceipt interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        conjunction = payload.pop("conjunction_passed")
        if payload.pop("raw_source_required") is not True:
            raise AdversarialValidationError("raw_source_required cannot be disabled")
        if payload.pop("validator_is_nomination_only") is not True:
            raise AdversarialValidationError("validator must remain nomination_only")
        if payload.pop("claims_general_python_equivalence") is not False:
            raise AdversarialValidationError("general Python equivalence is not claimed")
        if payload.pop("collapse_evidence_classes") is not False:
            raise AdversarialValidationError("evidence classes must not collapse")
        if payload.pop("mutated") is not False:
            raise AdversarialValidationError("validator cannot mutate")
        if payload.pop("deterministic") is not True:
            raise AdversarialValidationError("validator must remain deterministic")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise AdversarialValidationError("receipt analyzer_id must remain SPAR-029")
        relations = tuple(
            RelationVerdict.from_dict(item) for item in payload.pop("relation_verdicts")
        )
        mutants = tuple(
            MutantVerdict.from_dict(item) for item in payload.pop("mutant_verdicts")
        )
        cases = tuple(
            AdversarialVerdict.from_dict(item)
            for item in payload.pop("adversarial_verdicts")
        )
        result = cls(
            relation_verdicts=relations,
            mutant_verdicts=mutants,
            adversarial_verdicts=cases,
            **payload,
        )
        if conjunction is not result.conjunction_passed:
            raise AdversarialValidationError("conjunction_passed does not match status")
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


class RefactorMutationAndAdversarialValidator:
    """Reuse/generate bounded relations and mutants; block critical survivors."""

    interface: ClassVar[str] = REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_INTERFACE
    schema: ClassVar[str] = REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def compile_request(
        self,
        *,
        packet: Mapping[str, Any] | Any,
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        campaign_profile: CampaignProfile | Mapping[str, Any] | None = None,
        properties: Sequence[Mapping[str, Any] | PropertyRelation] = (),
        metamorphics: Sequence[Mapping[str, Any] | MetamorphicRelation] = (),
        mutants: Sequence[Mapping[str, Any] | MutantSpec] = (),
        adversarial_cases: Sequence[Mapping[str, Any] | AdversarialCase] = (),
        relation_verdicts: Sequence[Mapping[str, Any] | RelationVerdict] = (),
        mutant_verdicts: Sequence[Mapping[str, Any] | MutantVerdict] = (),
        adversarial_verdicts: Sequence[Mapping[str, Any] | AdversarialVerdict] = (),
        original_source_cids: Sequence[str] | None = None,
        candidate_source_cids: Sequence[str] | None = None,
        original_sources: Mapping[str, str] | None = None,
        candidate_sources: Mapping[str, str] | None = None,
        facade_plan: Mapping[str, Any] | Any | None = None,
        graph: Mapping[str, Any] | Any | None = None,
        frontier: Mapping[str, Any] | Any | None = None,
        vector_evidence: Any = None,
        generate: bool = True,
    ) -> MutationCampaignRequest:
        return compile_mutation_campaign_request(
            packet=packet,
            wave=wave,
            selection=selection,
            campaign_profile=campaign_profile,
            properties=properties,
            metamorphics=metamorphics,
            mutants=mutants,
            adversarial_cases=adversarial_cases,
            relation_verdicts=relation_verdicts,
            mutant_verdicts=mutant_verdicts,
            adversarial_verdicts=adversarial_verdicts,
            original_source_cids=original_source_cids,
            candidate_source_cids=candidate_source_cids,
            original_sources=original_sources,
            candidate_sources=candidate_sources,
            facade_plan=facade_plan,
            graph=graph,
            frontier=frontier,
            vector_evidence=vector_evidence,
            generate=generate,
        )

    def validate(
        self,
        request: MutationCampaignRequest | Mapping[str, Any],
        *,
        mutate: bool = False,
    ) -> MutationCampaignReceipt:
        return validate_mutation_campaign(request, mutate=mutate)

    def dry_run(
        self,
        request: MutationCampaignRequest | Mapping[str, Any],
    ) -> MutationCampaignReceipt:
        return dry_run_mutation_campaign(request)


def compile_mutation_campaign_request(
    *,
    packet: Mapping[str, Any] | Any,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    campaign_profile: CampaignProfile | Mapping[str, Any] | None = None,
    properties: Sequence[Mapping[str, Any] | PropertyRelation] = (),
    metamorphics: Sequence[Mapping[str, Any] | MetamorphicRelation] = (),
    mutants: Sequence[Mapping[str, Any] | MutantSpec] = (),
    adversarial_cases: Sequence[Mapping[str, Any] | AdversarialCase] = (),
    relation_verdicts: Sequence[Mapping[str, Any] | RelationVerdict] = (),
    mutant_verdicts: Sequence[Mapping[str, Any] | MutantVerdict] = (),
    adversarial_verdicts: Sequence[Mapping[str, Any] | AdversarialVerdict] = (),
    original_source_cids: Sequence[str] | None = None,
    candidate_source_cids: Sequence[str] | None = None,
    original_sources: Mapping[str, str] | None = None,
    candidate_sources: Mapping[str, str] | None = None,
    facade_plan: Mapping[str, Any] | Any | None = None,
    graph: Mapping[str, Any] | Any | None = None,
    frontier: Mapping[str, Any] | Any | None = None,
    vector_evidence: Any = None,
    generate: bool = True,
) -> MutationCampaignRequest:
    """Bind SPAR-019/025/026 mappings and reuse/generate a bounded campaign."""

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
        raise AdversarialValidationError(
            "SPAR-025 wave packet_cids must include SPAR-019 packet_cid"
        )
    selection_packet = selection_map.get("packet_cid")
    if selection_packet not in (None, "") and _cid(selection_packet, "packet_cid") != packet_cid:
        raise AdversarialValidationError("SPAR-026 packet_cid does not match SPAR-019")

    originals = (
        _cids(list(original_source_cids), "original_source_cids")
        if original_source_cids is not None
        else _packet_source_cids(packet_map)
    )
    candidates = candidate_source_cids
    if candidates is None:
        after = wave_map.get("after_source_cids") or wave_map.get("candidate_source_cids")
        if after in (None, (), []):
            raise AdversarialValidationError("candidate source_cids are required")
        candidates = after
    candidate_cids = _cids(list(candidates), "candidate_source_cids")

    write_paths = _packet_write_paths(packet_map)
    wave_paths = wave_map.get("write_paths")
    if wave_paths not in (None, ()):
        resolved_wave_paths = _exact_paths(wave_paths, "write_paths")
        if tuple(sorted(resolved_wave_paths)) != tuple(sorted(write_paths)):
            raise AdversarialValidationError("SPAR-025 write_paths must match SPAR-019")

    facade_map = None if facade_plan in (None, "") else _as_mapping(facade_plan, "SPAR-018 facade_plan")
    graph_map = None if graph in (None, "") else _as_mapping(graph, "SPAR-007 graph")
    frontier_map = None if frontier in (None, "") else _as_mapping(frontier, "SPAR-008 frontier")
    if facade_map is not None:
        _reject_non_admitting_payload(facade_map, "SPAR-018 facade_plan")
    if graph_map is not None:
        graph_tree = graph_map.get("tree_id") or _nested_mapping(
            graph_map.get("binding"), "SPAR-007 binding"
        ).get("tree_id")
        if graph_tree not in (None, ""):
            _require_tree(graph_tree, tree_id, "SPAR-007")
    if frontier_map is not None:
        frontier_tree = frontier_map.get("tree_id")
        if frontier_tree not in (None, ""):
            _require_tree(frontier_tree, tree_id, "SPAR-008")

    profile = _coerce_profile(campaign_profile)
    facade_kinds = _facade_kinds(wave_map, facade_map)
    generated_properties, generated_metamorphics = (
        generate_bounded_relations(
            write_paths=write_paths,
            facade_kinds=facade_kinds,
            campaign_profile=profile,
        )
        if generate
        else ((), ())
    )
    generated_mutants = (
        generate_bounded_mutants(
            write_paths=write_paths,
            facade_kinds=facade_kinds,
            campaign_profile=profile,
        )
        if generate
        else ()
    )
    generated_cases = (
        generate_bounded_adversarial_cases(
            write_paths=write_paths,
            facade_kinds=facade_kinds,
            campaign_profile=profile,
        )
        if generate
        else ()
    )
    resolved_properties = _merge_relations(
        properties, generated_properties, CampaignFamily.PROPERTY.value
    )
    resolved_metamorphics = _merge_relations(
        metamorphics, generated_metamorphics, CampaignFamily.METAMORPHIC.value
    )
    resolved_mutants = _merge_mutants(mutants, generated_mutants)
    resolved_cases = _merge_cases(adversarial_cases, generated_cases)

    resolved_relation_verdicts = [
        _relation_verdict_from_mapping(item) for item in relation_verdicts
    ]
    resolved_mutant_verdicts = [
        _mutant_verdict_from_mapping(item) for item in mutant_verdicts
    ]
    resolved_adversarial_verdicts = [
        _adversarial_verdict_from_mapping(item) for item in adversarial_verdicts
    ]
    existing_relation_ids = {item.item_id for item in resolved_relation_verdicts}
    resolved_relation_verdicts.extend(
        _derive_property_verdicts(
            resolved_properties,
            original_sources,
            candidate_sources,
            existing_relation_ids,
        )
    )
    derived_cases = _derive_adversarial_verdicts(
        resolved_cases,
        graph_map,
        frontier_map,
        existing_ids=set(),
    )
    if derived_cases:
        derived_ids = {item.case_id for item in derived_cases}
        resolved_adversarial_verdicts = [
            item
            for item in resolved_adversarial_verdicts
            if item.case_id not in derived_ids
        ] + derived_cases

    return MutationCampaignRequest(
        tree_id=tree_id,
        packet_cid=packet_cid,
        wave_receipt_cid=_wave_receipt_cid(wave_map),
        selection_cid=_selection_cid(selection_map),
        original_source_cids=originals,
        candidate_source_cids=candidate_cids,
        write_paths=write_paths,
        validation_commands=_packet_validation_commands(packet_map),
        campaign_profile=profile,
        properties=resolved_properties,
        metamorphics=resolved_metamorphics,
        mutants=resolved_mutants,
        adversarial_cases=resolved_cases,
        relation_verdicts=tuple(resolved_relation_verdicts),
        mutant_verdicts=tuple(resolved_mutant_verdicts),
        adversarial_verdicts=tuple(resolved_adversarial_verdicts),
        full_suite_required=_full_suite_required(selection_map),
    )


def validate_mutation_campaign(
    request: MutationCampaignRequest | Mapping[str, Any],
    *,
    mutate: bool = False,
) -> MutationCampaignReceipt:
    """Evaluate the campaign without mutating sources or collapsing evidence."""

    if mutate is not False:
        raise AdversarialValidationError("validator cannot mutate")
    resolved = (
        request
        if isinstance(request, MutationCampaignRequest)
        else MutationCampaignRequest.from_dict(request)
    )
    (
        relations,
        mutants,
        cases,
        status,
        reason,
        survivors,
        unknown,
    ) = _evaluate_campaign(resolved)
    return MutationCampaignReceipt(
        tree_id=resolved.tree_id,
        request_cid=resolved.request_cid,
        packet_cid=resolved.packet_cid,
        wave_receipt_cid=resolved.wave_receipt_cid,
        selection_cid=resolved.selection_cid,
        relation_verdicts=relations,
        mutant_verdicts=mutants,
        adversarial_verdicts=cases,
        status=status,
        terminal_reason=reason,
        critical_survivors=survivors,
        unknown_required_dynamics=unknown,
    )


def dry_run_mutation_campaign(
    request: MutationCampaignRequest | Mapping[str, Any],
) -> MutationCampaignReceipt:
    """Deterministic non-mutating evaluation of one mutation campaign."""

    result = validate_mutation_campaign(request, mutate=False)
    if result.mutated is not False or result.deterministic is not True:
        raise AdversarialValidationError(
            "dry-run must remain deterministic and non-mutating"
        )
    return result


def encode_canonical_request(request: MutationCampaignRequest) -> dict[str, Any]:
    return request.to_dict()


def decode_canonical_request(payload: Mapping[str, Any]) -> MutationCampaignRequest:
    return MutationCampaignRequest.from_dict(payload)


def encode_canonical_receipt(receipt: MutationCampaignReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> MutationCampaignReceipt:
    return MutationCampaignReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise AdversarialValidationError(
            f"adversarial validation must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADVERSARIAL_CASE_INTERFACE",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "CAMPAIGN_PROFILE_INTERFACE",
    "CRITICAL_SURVIVOR_BLOCKS_ACCEPTANCE",
    "DECLARED_ADVERSARIAL_KINDS",
    "DECLARED_CAMPAIGN_STATUSES",
    "DECLARED_EVIDENCE_CLASSES",
    "DECLARED_FAMILIES",
    "DECLARED_METAMORPHIC_KINDS",
    "DECLARED_MUTANT_KINDS",
    "DECLARED_MUTANT_STATUSES",
    "DECLARED_PROPERTY_KINDS",
    "DECLARED_RELATION_STATUSES",
    "DEFAULT_CRITICAL_MUTANT_KINDS",
    "DEFAULT_REQUIRED_FAMILIES",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "FAMILY_ADMITTED_EVIDENCE",
    "FORBIDDEN_EQUIVALENCE_NAMES",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "METAMORPHIC_RELATION_INTERFACE",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "MUTANT_KILL_IS_NOT_COMPLETION",
    "MUTANT_SPEC_INTERFACE",
    "MUTATION_CAMPAIGN_RECEIPT_INTERFACE",
    "MUTATION_CAMPAIGN_REQUEST_INTERFACE",
    "PROGRAM",
    "PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS",
    "PROPERTY_RELATION_INTERFACE",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REFACTOR_MUTATION_AND_ADVERSARIAL_VALIDATOR_INTERFACE",
    "RUNTIME_OBSERVATION_IS_NOT_STATIC_FACT",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TEST_PASS_IS_NOT_PROOF",
    "UNKNOWN_REQUIRED_DYNAMICS_BLOCK_ACCEPTANCE",
    "VALIDATION_CAN_AUTHORIZE_COMPLETION",
    "VALIDATION_CAN_AUTHORIZE_TRANSITION",
    "VALIDATION_CAN_CREATE_AUTHORITY",
    "VALIDATION_CLAIMS_GENERAL_PYTHON_EQUIVALENCE",
    "VALIDATION_COLLAPSES_EVIDENCE_CLASSES",
    "VALIDATION_CONTRACT_VERSION",
    "VALIDATOR_CAN_MUTATE_SOURCE",
    "VALIDATOR_IS_NOMINATION_ONLY",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "AdversarialCase",
    "AdversarialKind",
    "AdversarialValidationError",
    "AdversarialVerdict",
    "CampaignFamily",
    "CampaignProfile",
    "CampaignStatus",
    "EvidenceClass",
    "MetamorphicKind",
    "MetamorphicRelation",
    "MutantKind",
    "MutantSpec",
    "MutantStatus",
    "MutantVerdict",
    "MutationCampaignReceipt",
    "MutationCampaignRequest",
    "PropertyKind",
    "PropertyRelation",
    "RefactorMutationAndAdversarialValidator",
    "RelationStatus",
    "RelationVerdict",
    "admitted_evidence_classes",
    "adversarial_validation_cid_profile",
    "adversarial_validation_descriptor",
    "assert_evidence_class_admitted",
    "assert_evidence_classes_not_collapsed",
    "assert_not_competing_capsule_family",
    "compile_mutation_campaign_request",
    "decode_canonical_receipt",
    "decode_canonical_request",
    "dry_run_mutation_campaign",
    "encode_canonical_receipt",
    "encode_canonical_request",
    "generate_bounded_adversarial_cases",
    "generate_bounded_mutants",
    "generate_bounded_relations",
    "provider_free_exports",
    "unknown_dynamic_import_present",
    "unresolved_required_dynamics",
    "validate_mutation_campaign",
]
