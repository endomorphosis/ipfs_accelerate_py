"""SPAR-032 bounded CEGIS/CEGAR boundary and adapter synthesis.

This module extends current supervisor proof/validation authorities with
``BoundaryAdapterSynthesizer@1``.  It consumes SPAR-027 translation-validation
hooks, SPAR-028 differential mismatches, and SPAR-030 obligation/counterexample
mappings, then:

* compiles explicit examples, replayed counterexamples, and finite obligations;
* enumerates a finite adapter/guard/protocol/state/initialization grammar;
* refines coarse abstractions from spurious counterexamples (CEGAR);
* rejects templates against independently replayed counterexamples (CEGIS);
* re-enters SPAR-027 translation validation; and
* emits a nomination-only synthesis receipt.

Synthesized candidates cannot admit proofs, authorize a transition, or
complete a task.  Raw countermodels cannot refute until replay.  Vector,
model, and heuristic evidence cannot admit a candidate.  Unknown remains
unknown.  Unsupported required behavior is a typed terminal, never success.
Observational metadata is excluded from identity.  Dry-run is deterministic
and never mutates.  Network is denied.
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


TASK_ID: Final[str] = "SPAR-032"
GOAL_ID: Final[str] = "SPAR-G053"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "bounded synthesis"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.bounded_synthesis@1"
)
TRANSLATION_VALIDATION_OWNER: Final[str] = "ipfs_accelerate_py"
DIFFERENTIAL_OWNER: Final[str] = "ipfs_accelerate_py"
PROOF_ADAPTER_OWNER: Final[str] = "ipfs_accelerate_py"

BOUNDARY_ADAPTER_SYNTHESIZER_INTERFACE: Final[str] = "BoundaryAdapterSynthesizer@1"
SYNTHESIS_EXAMPLE_INTERFACE: Final[str] = "SynthesisExample@1"
SYNTHESIS_COUNTEREXAMPLE_INTERFACE: Final[str] = "SynthesisCounterexample@1"
SYNTHESIS_OBLIGATION_INTERFACE: Final[str] = "SynthesisObligation@1"
ABSTRACTION_PREDICATE_INTERFACE: Final[str] = "AbstractionPredicate@1"
ADAPTER_CANDIDATE_INTERFACE: Final[str] = "AdapterCandidate@1"
SYNTHESIS_ROUND_INTERFACE: Final[str] = "SynthesisRound@1"
VALIDATION_REENTRY_INTERFACE: Final[str] = "ValidationReentry@1"
BOUNDED_SYNTHESIS_RECEIPT_INTERFACE: Final[str] = "BoundedSynthesisReceipt@1"

BOUNDARY_ADAPTER_SYNTHESIZER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/boundary-adapter-synthesizer@1"
)
SYNTHESIS_EXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/synthesis-example@1"
)
SYNTHESIS_COUNTEREXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/synthesis-counterexample@1"
)
SYNTHESIS_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/synthesis-obligation@1"
)
ABSTRACTION_PREDICATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/abstraction-predicate@1"
)
ADAPTER_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adapter-candidate@1"
)
SYNTHESIS_ROUND_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/synthesis-round@1"
)
VALIDATION_REENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/validation-reentry@1"
)
BOUNDED_SYNTHESIS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/bounded-synthesis-receipt@1"
)

SYNTHESIS_CONTRACT_VERSION: Final[str] = "1"

SYNTHESIS_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
SYNTHESIS_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
SYNTHESIS_CAN_CREATE_AUTHORITY: Final[bool] = False
SYNTHESIS_CAN_CREATE_PROOF_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
TEST_PASS_IS_NOT_PROOF: Final[bool] = True
CANDIDATE_CANNOT_ADMIT_PROOFS: Final[bool] = True
RAW_COUNTERMODEL_CANNOT_REFUTE: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
SYNTHESIZER_IS_NOMINATION_ONLY: Final[bool] = True
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
SYNTHESIZED_CANDIDATES_REENTER_VALIDATION: Final[bool] = True
CANDIDATES_REMAIN_UNPROMOTED: Final[bool] = True

NETWORK_DENY: Final[str] = "deny"
TEMPLATE_SELECTOR_DETERMINISTIC: Final[str] = "deterministic"
DEFAULT_MAX_ROUNDS: Final[int] = 8
ABSOLUTE_MAX_ROUNDS: Final[int] = 64
DEFAULT_MAX_OBLIGATIONS: Final[int] = 64
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
MAX_EXAMPLES: Final[int] = 256
MAX_COUNTEREXAMPLES: Final[int] = 256
MAX_OBLIGATIONS: Final[int] = 256
MAX_ABSTRACTIONS: Final[int] = 128
MAX_CANDIDATES: Final[int] = 64
MAX_ROUNDS: Final[int] = ABSOLUTE_MAX_ROUNDS

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

FORBIDDEN_SYNTHESIS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "claim_general_equivalence",
        "admit_vector_proof",
        "promote_candidate_to_proof",
        "guess_axiom",
        "suppress_raw_source",
        "open_network",
        "implicit_install",
        "collapse_evidence",
        "promote_candidate",
        "authorize_completion",
    }
)

EXAMPLE_SOURCE_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "specification",
        "exact_static_fact",
        "admitted_contract",
        "test",
    }
)

FORBIDDEN_EXAMPLE_AUTHORITIES: Final[frozenset[str]] = frozenset(
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

COUNTEREXAMPLE_SOURCE_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "replayed_counterexample",
        "runtime_observation",
        "specification",
        "countermodel",
        "validation_failure",
        "differential_mismatch",
    }
)

REFUTING_COUNTEREXAMPLE_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "replayed_counterexample",
        "runtime_observation",
        "specification",
        "validation_failure",
        "differential_mismatch",
    }
)


class BoundedSynthesisError(ValueError):
    """Fail-closed violation of a SPAR-032 bounded synthesis contract."""

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


class AdapterArtifactKind(str, Enum):
    BOUNDARY_ADAPTER = "boundary_adapter"
    GUARD = "guard"
    PROTOCOL = "protocol"
    STATE_MAPPING = "state_mapping"
    INITIALIZATION_REPAIR = "initialization_repair"


class TemplateKind(str, Enum):
    IDENTITY = "identity"
    WRAPPER = "wrapper"
    PROJECTION = "projection"
    GUARD = "guard"
    PROTOCOL = "protocol"
    INITIALIZATION = "initialization"


class ExampleKind(str, Enum):
    BOUNDARY = "boundary"
    GUARD = "guard"
    PROTOCOL = "protocol"
    STATE = "state"
    INITIALIZATION = "initialization"
    IDENTITY = "identity"
    WRAPPER = "wrapper"


class ExamplePolarity(str, Enum):
    SATISFY = "satisfy"


class ObligationKind(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    CONDITION = "condition"
    INVARIANT = "invariant"
    ASSUMPTION = "assumption"
    GUARANTEE = "guarantee"
    EXCEPTION = "exception"
    EFFECT = "effect"
    STATE = "state"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"


class ObligationStatus(str, Enum):
    LOWERED = "lowered"
    RESIDUAL = "residual"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"


class AbstractionGrain(str, Enum):
    COARSE = "coarse"
    REFINED = "refined"


class AbstractionKind(str, Enum):
    TYPE = "type"
    EFFECT = "effect"
    STATE = "state"
    INIT_ORDER = "init_order"
    PROTOCOL = "protocol"


class SynthesisStatus(str, Enum):
    NOMINATED = "nominated"
    INCOMPLETE = "incomplete"
    REFUTED = "refuted"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class RoundMode(str, Enum):
    CEGIS = "cegis"
    CEGAR = "cegar"
    VALIDATION = "validation"


class RoundOutcome(str, Enum):
    COVERED = "covered"
    REFUTED = "refuted"
    REFINED = "refined"
    UNCOVERED = "uncovered"
    VALIDATED = "validated"
    VALIDATION_REJECTED = "validation_rejected"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ValidationReentryStatus(str, Enum):
    VALIDATED = "validated"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"
    MISSING = "missing"


DECLARED_ADAPTER_ARTIFACT_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in AdapterArtifactKind
)
DECLARED_TEMPLATE_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in TemplateKind
)
DECLARED_EXAMPLE_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in ExampleKind
)
DECLARED_OBLIGATION_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in ObligationKind
)
DECLARED_OBLIGATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in ObligationStatus
)
DECLARED_SYNTHESIS_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in SynthesisStatus
)
DECLARED_ROUND_MODES: Final[frozenset[str]] = frozenset(item.value for item in RoundMode)
DECLARED_ROUND_OUTCOMES: Final[frozenset[str]] = frozenset(
    item.value for item in RoundOutcome
)
DECLARED_VALIDATION_REENTRY_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in ValidationReentryStatus
)

TEMPLATE_TO_ARTIFACT: Final[Mapping[str, str]] = {
    TemplateKind.IDENTITY.value: AdapterArtifactKind.BOUNDARY_ADAPTER.value,
    TemplateKind.WRAPPER.value: AdapterArtifactKind.BOUNDARY_ADAPTER.value,
    TemplateKind.PROJECTION.value: AdapterArtifactKind.STATE_MAPPING.value,
    TemplateKind.GUARD.value: AdapterArtifactKind.GUARD.value,
    TemplateKind.PROTOCOL.value: AdapterArtifactKind.PROTOCOL.value,
    TemplateKind.INITIALIZATION.value: AdapterArtifactKind.INITIALIZATION_REPAIR.value,
}

TEMPLATE_CATALOG: Final[tuple[Mapping[str, Any], ...]] = (
    {
        "template_id": TemplateKind.IDENTITY.value,
        "kind": AdapterArtifactKind.BOUNDARY_ADAPTER.value,
        "covers_example_kinds": (
            ExampleKind.BOUNDARY.value,
            ExampleKind.IDENTITY.value,
        ),
        "covers_obligation_kinds": (
            ObligationKind.INPUT.value,
            ObligationKind.OUTPUT.value,
        ),
    },
    {
        "template_id": TemplateKind.WRAPPER.value,
        "kind": AdapterArtifactKind.BOUNDARY_ADAPTER.value,
        "covers_example_kinds": (
            ExampleKind.BOUNDARY.value,
            ExampleKind.WRAPPER.value,
        ),
        "covers_obligation_kinds": (
            ObligationKind.INPUT.value,
            ObligationKind.OUTPUT.value,
            ObligationKind.EFFECT.value,
        ),
    },
    {
        "template_id": TemplateKind.PROJECTION.value,
        "kind": AdapterArtifactKind.STATE_MAPPING.value,
        "covers_example_kinds": (ExampleKind.STATE.value,),
        "covers_obligation_kinds": (ObligationKind.STATE.value,),
    },
    {
        "template_id": TemplateKind.GUARD.value,
        "kind": AdapterArtifactKind.GUARD.value,
        "covers_example_kinds": (ExampleKind.GUARD.value,),
        "covers_obligation_kinds": (
            ObligationKind.CONDITION.value,
            ObligationKind.INVARIANT.value,
            ObligationKind.ASSUMPTION.value,
            ObligationKind.GUARANTEE.value,
        ),
    },
    {
        "template_id": TemplateKind.PROTOCOL.value,
        "kind": AdapterArtifactKind.PROTOCOL.value,
        "covers_example_kinds": (ExampleKind.PROTOCOL.value,),
        "covers_obligation_kinds": (
            ObligationKind.EFFECT.value,
            ObligationKind.EXCEPTION.value,
        ),
    },
    {
        "template_id": TemplateKind.INITIALIZATION.value,
        "kind": AdapterArtifactKind.INITIALIZATION_REPAIR.value,
        "covers_example_kinds": (ExampleKind.INITIALIZATION.value,),
        "covers_obligation_kinds": (ObligationKind.RESOURCE.value,),
    },
)

ValidateFn = Callable[..., Mapping[str, Any]]


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise BoundedSynthesisError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise BoundedSynthesisError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise BoundedSynthesisError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise BoundedSynthesisError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise BoundedSynthesisError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise BoundedSynthesisError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise BoundedSynthesisError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int) -> int:
    if type(value) is bool or type(value) is not int:
        raise BoundedSynthesisError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise BoundedSynthesisError(f"{name} is out of bounds")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise BoundedSynthesisError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise BoundedSynthesisError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise BoundedSynthesisError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise BoundedSynthesisError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise BoundedSynthesisError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise BoundedSynthesisError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _reject_forbidden_bodies(payload: Any, name: str) -> None:
    if isinstance(payload, Mapping) and not isinstance(payload, (str, bytes, bytearray)):
        present = _FORBIDDEN_BODY_KEYS & set(payload)
        if present:
            raise BoundedSynthesisError(
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
        raise BoundedSynthesisError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise BoundedSynthesisError(f"{name} does not verify")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise BoundedSynthesisError(f"unknown {name}: {text}") from exc


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise BoundedSynthesisError(f"{name} cannot claim {flag}")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise BoundedSynthesisError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise BoundedSynthesisError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise BoundedSynthesisError(f"{name} must not contain duplicates")
    return ordered


def _ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise BoundedSynthesisError(f"{name} must be a list")
    ordered = tuple(_text(item, name) for item in values)
    if len(ordered) > limit:
        raise BoundedSynthesisError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise BoundedSynthesisError(f"{name} must not contain duplicates")
    return ordered


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, ()):
        if required:
            raise BoundedSynthesisError(f"{name} are required")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise BoundedSynthesisError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise BoundedSynthesisError(f"{name} are required")
    if len(ordered) != len(set(ordered)):
        raise BoundedSynthesisError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise BoundedSynthesisError(f"{name} exceed maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise BoundedSynthesisError(f"{name} exceeds path bound")
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
        raise BoundedSynthesisError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise BoundedSynthesisError(f"{name} must be a normalized repository-relative path")
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise BoundedSynthesisError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise BoundedSynthesisError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise BoundedSynthesisError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise BoundedSynthesisError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise BoundedSynthesisError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise BoundedSynthesisError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise BoundedSynthesisError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise BoundedSynthesisError(f"missing {name}")
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
    raise BoundedSynthesisError(f"{name} must be a mapping")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BoundedSynthesisError(f"{name} must be a list")
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
        raise BoundedSynthesisError(f"{name} items must be objects")
    if len(items) > MAX_MEMBERS:
        raise BoundedSynthesisError(f"{name} exceeds maximum length")
    return tuple(items)


def _require_tree(value: Any, expected: str, label: str) -> None:
    if value in (None, ""):
        raise BoundedSynthesisError(f"{label} tree_id is required")
    actual = _tree_id(value)
    if actual != expected:
        raise BoundedSynthesisError(f"{label} tree_id does not match")


def _reject_non_admitting_payload(payload: Mapping[str, Any], name: str) -> None:
    evidence = payload.get("evidence_class")
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise BoundedSynthesisError(f"{name} {evidence} cannot admit")
    if payload.get("admit_proof") is True or payload.get("admits_proof") is True:
        raise BoundedSynthesisError(f"{name} cannot admit proofs")
    if payload.get("admit_equivalence") is True:
        raise BoundedSynthesisError(f"{name} cannot admit")
    if payload.get("guessed_axiom") is True:
        raise BoundedSynthesisError("guessed axioms are rejected")
    if payload.get("promote_candidate") is True:
        raise BoundedSynthesisError("synthesized candidates remain unpromoted")


def _reject_vector_admission(vector_evidence: Any) -> None:
    if vector_evidence in (None, (), {}):
        return
    if isinstance(vector_evidence, Mapping):
        if vector_evidence.get("suppress_raw_source") is True:
            raise BoundedSynthesisError("vectors cannot suppress raw-source fallback")
        _reject_non_admitting_payload(vector_evidence, "vector_evidence")
        evidence = vector_evidence.get("evidence_class")
        if evidence in _NON_ADMITTING_EVIDENCE:
            raise BoundedSynthesisError("vector/model/heuristic evidence cannot admit")
        return
    raise BoundedSynthesisError("vector_evidence must be an object")


def _network_value(value: Any) -> str:
    if value in (None, ""):
        return NETWORK_DENY
    text = _text(value, "network")
    if text != NETWORK_DENY:
        raise BoundedSynthesisError("network is denied")
    return NETWORK_DENY


def bounded_synthesis_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def bounded_synthesis_descriptor() -> dict[str, Any]:
    return {
        "interface": BOUNDARY_ADAPTER_SYNTHESIZER_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "raw_source_required": True,
        "nomination_only": True,
        "network": NETWORK_DENY,
        "claims_general_equivalence": False,
        "candidate_cannot_admit_proofs": True,
        "raw_countermodel_cannot_refute": True,
        "unknown_remains_unknown": True,
        "guessed_axioms_rejected": True,
        "synthesized_candidates_reenter_validation": True,
        "candidates_remain_unpromoted": True,
        "template_selector": TEMPLATE_SELECTOR_DETERMINISTIC,
        "forbids": tuple(sorted(FORBIDDEN_SYNTHESIS_NAMES)),
    }


def _wave_cid(wave: Mapping[str, Any]) -> str:
    value = wave.get("receipt_cid") or wave.get("wave_cid")
    if value in (None, ""):
        raise BoundedSynthesisError("SPAR-025 receipt_cid is required")
    return _cid(value, "SPAR-025 receipt_cid")


def _wave_packet_cids(wave: Mapping[str, Any]) -> tuple[str, ...]:
    values = wave.get("packet_cids")
    if values in (None, ()):
        raise BoundedSynthesisError("SPAR-025 packet_cids are required")
    return _cids(values, "SPAR-025 packet_cids", required=True)


def _wave_write_paths(wave: Mapping[str, Any]) -> tuple[str, ...]:
    values = wave.get("write_paths")
    if values in (None, ()):
        raise BoundedSynthesisError("SPAR-025 write_paths are required")
    return _exact_paths(values, "SPAR-025 write_paths")


def _selection_cid(selection: Mapping[str, Any]) -> str:
    value = selection.get("validation_selection_cid") or selection.get("selection_cid")
    if value in (None, ""):
        raise BoundedSynthesisError("SPAR-026 validation_selection_cid is required")
    return _cid(value, "SPAR-026 validation_selection_cid")


def _selection_packet_cid(selection: Mapping[str, Any]) -> str:
    value = selection.get("packet_cid")
    if value in (None, ""):
        raise BoundedSynthesisError("SPAR-026 packet_cid is required")
    return _cid(value, "SPAR-026 packet_cid")


def _selection_write_paths(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("write_paths")
    if values in (None, ()):
        raise BoundedSynthesisError("SPAR-026 write_paths are required")
    return _exact_paths(values, "SPAR-026 write_paths")


def _selection_sources(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("raw_source_cids")
    if values in (None, (), []):
        raise BoundedSynthesisError("raw source is required")
    return _cids(values, "raw source", required=True)


def _selection_commands(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("validation_commands")
    if values in (None, ()):
        raise BoundedSynthesisError("SPAR-026 validation_commands are required")
    return _commands(values, "SPAR-026 validation_commands")


def _bind_predecessors(
    *,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
) -> tuple[str, str, str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if wave is None:
        raise BoundedSynthesisError("SPAR-025 wave is required")
    if selection is None:
        raise BoundedSynthesisError("SPAR-026 selection is required")
    wave_map = _as_mapping(wave, "SPAR-025 wave")
    selection_map = _as_mapping(selection, "SPAR-026 selection")
    _reject_non_admitting_payload(wave_map, "SPAR-025 wave")
    _reject_non_admitting_payload(selection_map, "SPAR-026 selection")
    tree_id = _tree_id(wave_map.get("tree_id"))
    selection_tree = selection_map.get("tree_id")
    if selection_tree not in (None, ""):
        _require_tree(selection_tree, tree_id, "SPAR-026")
    write_paths = _wave_write_paths(wave_map)
    selection_paths = _selection_write_paths(selection_map)
    if write_paths != selection_paths:
        raise BoundedSynthesisError("SPAR-025/026 write_paths must match")
    packet_cid = _selection_packet_cid(selection_map)
    wave_packets = _wave_packet_cids(wave_map)
    if packet_cid not in wave_packets:
        raise BoundedSynthesisError(
            "SPAR-025 wave packet_cids must include SPAR-026 packet_cid"
        )
    return (
        tree_id,
        _wave_cid(wave_map),
        _selection_cid(selection_map),
        packet_cid,
        write_paths,
        _selection_sources(selection_map),
        _selection_commands(selection_map),
    )


@dataclass(frozen=True, slots=True)
class SynthesisExample:
    """One explicit positive example. Body-free; never a guessed axiom."""

    example_id: str
    kind: str
    obligation_id: str
    tree_id: str
    source_cid: str
    polarity: str = ExamplePolarity.SATISFY.value
    template_hint: str = ""
    source_authority: str = "specification"

    interface: ClassVar[str] = SYNTHESIS_EXAMPLE_INTERFACE
    schema: ClassVar[str] = SYNTHESIS_EXAMPLE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "example_id",
            "kind",
            "obligation_id",
            "tree_id",
            "source_cid",
            "polarity",
            "template_hint",
            "source_authority",
            "example_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _text(self.example_id, "example_id"))
        object.__setattr__(self, "kind", _enum(self.kind, ExampleKind, "kind"))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "source_cid", _cid(self.source_cid, "source_cid"))
        object.__setattr__(
            self, "polarity", _enum(self.polarity, ExamplePolarity, "polarity")
        )
        hint = _text(self.template_hint, "template_hint", empty=True)
        if hint and hint not in DECLARED_TEMPLATE_KINDS:
            raise BoundedSynthesisError(f"unknown template_hint: {hint}")
        object.__setattr__(self, "template_hint", hint)
        authority = _text(self.source_authority, "source_authority")
        if authority in FORBIDDEN_EXAMPLE_AUTHORITIES:
            raise BoundedSynthesisError("guessed/vector/model examples cannot admit")
        if authority not in EXAMPLE_SOURCE_AUTHORITIES:
            raise BoundedSynthesisError(
                f"unsupported example source_authority {authority!r}"
            )
        object.__setattr__(self, "source_authority", authority)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SYNTHESIS_EXAMPLE_SCHEMA,
            "interface": SYNTHESIS_EXAMPLE_INTERFACE,
            "example_id": self.example_id,
            "kind": self.kind,
            "obligation_id": self.obligation_id,
            "tree_id": self.tree_id,
            "source_cid": self.source_cid,
            "polarity": self.polarity,
            "template_hint": self.template_hint,
            "source_authority": self.source_authority,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def example_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["example_cid"] = self.example_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SynthesisExample":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("example_cid")
        if payload.pop("schema") != SYNTHESIS_EXAMPLE_SCHEMA:
            raise BoundedSynthesisError("unsupported SynthesisExample schema")
        if payload.pop("interface") != SYNTHESIS_EXAMPLE_INTERFACE:
            raise BoundedSynthesisError("unsupported SynthesisExample interface")
        result = cls(**payload)
        _verify_cid(claimed, result.example_cid, "example_cid")
        return result


@dataclass(frozen=True, slots=True)
class SynthesisCounterexample:
    """Independently obtained counterexample. Raw diagnostics cannot refute."""

    counterexample_id: str
    kind: str
    obligation_id: str
    tree_id: str
    source_cid: str
    source_authority: str
    replayed: bool = False
    conflicts_with: str = ""
    abstraction_gap: bool = False
    predicate_id: str = ""

    interface: ClassVar[str] = SYNTHESIS_COUNTEREXAMPLE_INTERFACE
    schema: ClassVar[str] = SYNTHESIS_COUNTEREXAMPLE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "counterexample_id",
            "kind",
            "obligation_id",
            "tree_id",
            "source_cid",
            "source_authority",
            "replayed",
            "conflicts_with",
            "abstraction_gap",
            "predicate_id",
            "counterexample_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "counterexample_id",
            _text(self.counterexample_id, "counterexample_id"),
        )
        object.__setattr__(self, "kind", _enum(self.kind, ExampleKind, "kind"))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "source_cid", _cid(self.source_cid, "source_cid"))
        authority = _text(self.source_authority, "source_authority")
        if authority not in COUNTEREXAMPLE_SOURCE_AUTHORITIES:
            raise BoundedSynthesisError(
                f"unsupported counterexample source_authority {authority!r}"
            )
        object.__setattr__(self, "source_authority", authority)
        replayed = _bool(self.replayed, "replayed")
        if (
            replayed
            and authority not in REFUTING_COUNTEREXAMPLE_AUTHORITIES
        ):
            raise BoundedSynthesisError("raw countermodels cannot refute until replay")
        if authority == "countermodel" and replayed is True:
            raise BoundedSynthesisError("raw countermodels cannot refute until replay")
        object.__setattr__(self, "replayed", replayed)
        conflict = _text(self.conflicts_with, "conflicts_with", empty=True)
        if conflict and conflict not in DECLARED_TEMPLATE_KINDS:
            raise BoundedSynthesisError(f"unknown conflicts_with: {conflict}")
        object.__setattr__(self, "conflicts_with", conflict)
        object.__setattr__(
            self, "abstraction_gap", _bool(self.abstraction_gap, "abstraction_gap")
        )
        object.__setattr__(
            self, "predicate_id", _text(self.predicate_id, "predicate_id", empty=True)
        )
        if self.abstraction_gap and not self.predicate_id:
            raise BoundedSynthesisError("abstraction_gap requires predicate_id")
        if self.abstraction_gap and not replayed:
            raise BoundedSynthesisError("raw countermodels cannot refine abstractions")

    @property
    def can_refute(self) -> bool:
        return self.replayed is True and self.source_authority in (
            REFUTING_COUNTEREXAMPLE_AUTHORITIES
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SYNTHESIS_COUNTEREXAMPLE_SCHEMA,
            "interface": SYNTHESIS_COUNTEREXAMPLE_INTERFACE,
            "counterexample_id": self.counterexample_id,
            "kind": self.kind,
            "obligation_id": self.obligation_id,
            "tree_id": self.tree_id,
            "source_cid": self.source_cid,
            "source_authority": self.source_authority,
            "replayed": self.replayed,
            "conflicts_with": self.conflicts_with,
            "abstraction_gap": self.abstraction_gap,
            "predicate_id": self.predicate_id,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def counterexample_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["counterexample_cid"] = self.counterexample_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SynthesisCounterexample":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("counterexample_cid")
        if payload.pop("schema") != SYNTHESIS_COUNTEREXAMPLE_SCHEMA:
            raise BoundedSynthesisError("unsupported SynthesisCounterexample schema")
        if payload.pop("interface") != SYNTHESIS_COUNTEREXAMPLE_INTERFACE:
            raise BoundedSynthesisError("unsupported SynthesisCounterexample interface")
        result = cls(**payload)
        _verify_cid(claimed, result.counterexample_cid, "counterexample_cid")
        return result


@dataclass(frozen=True, slots=True)
class SynthesisObligation:
    """Finite lowered obligation ingested as a mapping from SPAR-030."""

    obligation_id: str
    kind: str
    tree_id: str
    contract_cid: str
    finite: bool = True
    required: bool = True
    status: str = ObligationStatus.LOWERED.value

    interface: ClassVar[str] = SYNTHESIS_OBLIGATION_INTERFACE
    schema: ClassVar[str] = SYNTHESIS_OBLIGATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "obligation_id",
            "kind",
            "tree_id",
            "contract_cid",
            "finite",
            "required",
            "status",
            "obligation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(self, "kind", _enum(self.kind, ObligationKind, "kind"))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "contract_cid", _cid(self.contract_cid, "contract_cid"))
        object.__setattr__(self, "finite", _bool(self.finite, "finite"))
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(
            self, "status", _enum(self.status, ObligationStatus, "status")
        )
        if self.required and self.status == ObligationStatus.INCOMPLETE.value:
            raise BoundedSynthesisError("incomplete SPAR-030 obligation")
        if self.required and self.status == ObligationStatus.UNSUPPORTED.value:
            raise BoundedSynthesisError("unsupported required SPAR-030 obligation")
        if self.required and self.finite is False:
            raise BoundedSynthesisError("unsupported required non-finite obligation")
        if self.status == ObligationStatus.LOWERED.value and not self.finite:
            raise BoundedSynthesisError("lowered obligations must be finite")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SYNTHESIS_OBLIGATION_SCHEMA,
            "interface": SYNTHESIS_OBLIGATION_INTERFACE,
            "obligation_id": self.obligation_id,
            "kind": self.kind,
            "tree_id": self.tree_id,
            "contract_cid": self.contract_cid,
            "finite": self.finite,
            "required": self.required,
            "status": self.status,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def obligation_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["obligation_cid"] = self.obligation_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SynthesisObligation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("obligation_cid")
        if payload.pop("schema") != SYNTHESIS_OBLIGATION_SCHEMA:
            raise BoundedSynthesisError("unsupported SynthesisObligation schema")
        if payload.pop("interface") != SYNTHESIS_OBLIGATION_INTERFACE:
            raise BoundedSynthesisError("unsupported SynthesisObligation interface")
        result = cls(**payload)
        _verify_cid(claimed, result.obligation_cid, "obligation_cid")
        return result


@dataclass(frozen=True, slots=True)
class AbstractionPredicate:
    """One CEGAR abstraction predicate. Coarse until a spurious CE refines it."""

    predicate_id: str
    kind: str
    grain: str = AbstractionGrain.COARSE.value
    distinguishing: bool = False

    interface: ClassVar[str] = ABSTRACTION_PREDICATE_INTERFACE
    schema: ClassVar[str] = ABSTRACTION_PREDICATE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "predicate_id",
            "kind",
            "grain",
            "distinguishing",
            "predicate_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate_id", _text(self.predicate_id, "predicate_id"))
        object.__setattr__(self, "kind", _enum(self.kind, AbstractionKind, "kind"))
        object.__setattr__(self, "grain", _enum(self.grain, AbstractionGrain, "grain"))
        object.__setattr__(
            self, "distinguishing", _bool(self.distinguishing, "distinguishing")
        )
        if self.distinguishing and self.grain != AbstractionGrain.REFINED.value:
            raise BoundedSynthesisError("distinguishing predicates must be refined")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ABSTRACTION_PREDICATE_SCHEMA,
            "interface": ABSTRACTION_PREDICATE_INTERFACE,
            "predicate_id": self.predicate_id,
            "kind": self.kind,
            "grain": self.grain,
            "distinguishing": self.distinguishing,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def predicate_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["predicate_cid"] = self.predicate_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AbstractionPredicate":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("predicate_cid")
        if payload.pop("schema") != ABSTRACTION_PREDICATE_SCHEMA:
            raise BoundedSynthesisError("unsupported AbstractionPredicate schema")
        if payload.pop("interface") != ABSTRACTION_PREDICATE_INTERFACE:
            raise BoundedSynthesisError("unsupported AbstractionPredicate interface")
        result = cls(**payload)
        _verify_cid(claimed, result.predicate_cid, "predicate_cid")
        return result


@dataclass(frozen=True, slots=True)
class AdapterCandidate:
    """One unpromoted adapter/guard/protocol/state/init candidate."""

    candidate_id: str
    kind: str
    template_id: str
    tree_id: str
    obligation_ids: Sequence[str] = ()
    example_cids: Sequence[str] = ()
    counterexample_cids: Sequence[str] = ()
    abstraction_cids: Sequence[str] = ()
    write_paths: Sequence[str] = ()
    source_cids: Sequence[str] = ()
    unpromoted: bool = True

    interface: ClassVar[str] = ADAPTER_CANDIDATE_INTERFACE
    schema: ClassVar[str] = ADAPTER_CANDIDATE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "candidate_id",
            "kind",
            "template_id",
            "tree_id",
            "obligation_ids",
            "example_cids",
            "counterexample_cids",
            "abstraction_cids",
            "write_paths",
            "source_cids",
            "unpromoted",
            "candidate_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, AdapterArtifactKind, "kind")
        )
        template = _enum(self.template_id, TemplateKind, "template_id")
        if TEMPLATE_TO_ARTIFACT[template] != self.kind:
            raise BoundedSynthesisError("template_id does not match candidate kind")
        object.__setattr__(self, "template_id", template)
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "obligation_ids",
            _ordered_text(list(self.obligation_ids), "obligation_ids", limit=MAX_OBLIGATIONS)
            if self.obligation_ids
            else (),
        )
        object.__setattr__(
            self, "example_cids", _cids(self.example_cids, "example_cids")
        )
        object.__setattr__(
            self,
            "counterexample_cids",
            _cids(self.counterexample_cids, "counterexample_cids"),
        )
        object.__setattr__(
            self, "abstraction_cids", _cids(self.abstraction_cids, "abstraction_cids")
        )
        object.__setattr__(
            self,
            "write_paths",
            _exact_paths(self.write_paths, "write_paths") if self.write_paths else (),
        )
        object.__setattr__(
            self, "source_cids", _cids(self.source_cids, "source_cids", required=True)
        )
        if _bool(self.unpromoted, "unpromoted") is not True:
            raise BoundedSynthesisError("synthesized candidates remain unpromoted")
        object.__setattr__(self, "unpromoted", True)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": ADAPTER_CANDIDATE_SCHEMA,
            "interface": ADAPTER_CANDIDATE_INTERFACE,
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "template_id": self.template_id,
            "tree_id": self.tree_id,
            "obligation_ids": list(self.obligation_ids),
            "example_cids": list(self.example_cids),
            "counterexample_cids": list(self.counterexample_cids),
            "abstraction_cids": list(self.abstraction_cids),
            "write_paths": list(self.write_paths),
            "source_cids": list(self.source_cids),
            "unpromoted": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def candidate_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["candidate_cid"] = self.candidate_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdapterCandidate":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("candidate_cid")
        if payload.pop("schema") != ADAPTER_CANDIDATE_SCHEMA:
            raise BoundedSynthesisError("unsupported AdapterCandidate schema")
        if payload.pop("interface") != ADAPTER_CANDIDATE_INTERFACE:
            raise BoundedSynthesisError("unsupported AdapterCandidate interface")
        if payload.get("unpromoted") is not True:
            raise BoundedSynthesisError("synthesized candidates remain unpromoted")
        result = cls(**payload)
        _verify_cid(claimed, result.candidate_cid, "candidate_cid")
        return result


@dataclass(frozen=True, slots=True)
class SynthesisRound:
    """One bounded CEGIS/CEGAR/validation round. Body-free."""

    round_index: int
    mode: str
    template_id: str
    outcome: str
    candidate_cid: str = ""
    counterexample_cids: Sequence[str] = ()
    abstraction_cids: Sequence[str] = ()
    reason_code: str = ""

    interface: ClassVar[str] = SYNTHESIS_ROUND_INTERFACE
    schema: ClassVar[str] = SYNTHESIS_ROUND_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "round_index",
            "mode",
            "template_id",
            "outcome",
            "candidate_cid",
            "counterexample_cids",
            "abstraction_cids",
            "reason_code",
            "round_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "round_index",
            _int(self.round_index, "round_index", minimum=1, maximum=ABSOLUTE_MAX_ROUNDS),
        )
        object.__setattr__(self, "mode", _enum(self.mode, RoundMode, "mode"))
        object.__setattr__(
            self, "template_id", _enum(self.template_id, TemplateKind, "template_id")
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, RoundOutcome, "outcome")
        )
        object.__setattr__(
            self, "candidate_cid", _optional_cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self,
            "counterexample_cids",
            _cids(self.counterexample_cids, "counterexample_cids"),
        )
        object.__setattr__(
            self, "abstraction_cids", _cids(self.abstraction_cids, "abstraction_cids")
        )
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", empty=True)
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SYNTHESIS_ROUND_SCHEMA,
            "interface": SYNTHESIS_ROUND_INTERFACE,
            "round_index": self.round_index,
            "mode": self.mode,
            "template_id": self.template_id,
            "outcome": self.outcome,
            "candidate_cid": self.candidate_cid,
            "counterexample_cids": list(self.counterexample_cids),
            "abstraction_cids": list(self.abstraction_cids),
            "reason_code": self.reason_code,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def round_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["round_cid"] = self.round_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SynthesisRound":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("round_cid")
        if payload.pop("schema") != SYNTHESIS_ROUND_SCHEMA:
            raise BoundedSynthesisError("unsupported SynthesisRound schema")
        if payload.pop("interface") != SYNTHESIS_ROUND_INTERFACE:
            raise BoundedSynthesisError("unsupported SynthesisRound interface")
        result = cls(**payload)
        _verify_cid(claimed, result.round_cid, "round_cid")
        return result


@dataclass(frozen=True, slots=True)
class ValidationReentry:
    """SPAR-027 re-entry record. A pass remains nomination-only."""

    candidate_cid: str
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
            "candidate_cid",
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
        object.__setattr__(
            self, "candidate_cid", _cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self, "status", _enum(self.status, ValidationReentryStatus, "status")
        )
        object.__setattr__(self, "result_cid", _optional_cid(self.result_cid, "result_cid"))
        validator = _text(self.validator_id, "validator_id")
        if validator != VALIDATOR_ID:
            raise BoundedSynthesisError("validator_id must remain SPAR-027")
        object.__setattr__(self, "validator_id", VALIDATOR_ID)
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", empty=True)
        )
        if self.status == ValidationReentryStatus.VALIDATED.value and self.reason_code:
            raise BoundedSynthesisError("validated re-entry cannot carry a terminal reason")
        if (
            self.status != ValidationReentryStatus.VALIDATED.value
            and not self.reason_code
        ):
            raise BoundedSynthesisError("typed terminal re-entry requires a reason")

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
            "candidate_cid": self.candidate_cid,
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
            raise BoundedSynthesisError("unsupported ValidationReentry schema")
        if payload.pop("interface") != VALIDATION_REENTRY_INTERFACE:
            raise BoundedSynthesisError("unsupported ValidationReentry interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("nomination_only") is not True:
            raise BoundedSynthesisError("validation re-entry must remain nomination_only")
        result = cls(**payload)
        _verify_cid(claimed, result.reentry_cid, "reentry_cid")
        return result


def _strip_envelope(payload: Mapping[str, Any], *cid_fields: str) -> dict[str, Any]:
    data = dict(payload)
    data.pop("schema", None)
    data.pop("interface", None)
    for field in cid_fields:
        data.pop(field, None)
    return data


def _coerce_example(
    item: SynthesisExample | Mapping[str, Any], *, tree_id: str
) -> SynthesisExample:
    if isinstance(item, SynthesisExample):
        example = item
    elif isinstance(item, Mapping) and "example_cid" in item and "schema" in item:
        _reject_forbidden_bodies(item, "synthesis example")
        example = SynthesisExample.from_dict(item)
    elif isinstance(item, Mapping):
        _reject_forbidden_bodies(item, "synthesis example")
        example = SynthesisExample(**_strip_envelope(item, "example_cid"))
    else:
        raise BoundedSynthesisError("examples items must be objects")
    if example.tree_id != tree_id:
        raise BoundedSynthesisError("example tree_id does not match")
    return example


def _coerce_counterexample(
    item: SynthesisCounterexample | Mapping[str, Any], *, tree_id: str
) -> SynthesisCounterexample:
    if isinstance(item, SynthesisCounterexample):
        counterexample = item
    elif isinstance(item, Mapping) and "counterexample_cid" in item and "schema" in item:
        _reject_forbidden_bodies(item, "synthesis counterexample")
        counterexample = SynthesisCounterexample.from_dict(item)
    elif isinstance(item, Mapping):
        _reject_forbidden_bodies(item, "synthesis counterexample")
        counterexample = SynthesisCounterexample(
            **_strip_envelope(item, "counterexample_cid")
        )
    else:
        raise BoundedSynthesisError("counterexamples items must be objects")
    if counterexample.tree_id != tree_id:
        raise BoundedSynthesisError("counterexample tree_id does not match")
    return counterexample


def _coerce_obligation(
    item: SynthesisObligation | Mapping[str, Any], *, tree_id: str
) -> SynthesisObligation:
    if isinstance(item, SynthesisObligation):
        obligation = item
    elif isinstance(item, Mapping) and "obligation_cid" in item and "schema" in item:
        _reject_forbidden_bodies(item, "synthesis obligation")
        obligation = SynthesisObligation.from_dict(item)
    elif isinstance(item, Mapping):
        _reject_forbidden_bodies(item, "synthesis obligation")
        payload = _strip_envelope(item, "obligation_cid")
        payload.pop("clause_id", None)
        payload.pop("polarity", None)
        payload.pop("premise_ids", None)
        payload.pop("residual_reason", None)
        obligation = SynthesisObligation(**payload)
    else:
        raise BoundedSynthesisError("obligations items must be objects")
    if obligation.tree_id != tree_id:
        raise BoundedSynthesisError("SPAR-030 tree_id does not match")
    return obligation


def _coerce_abstraction(
    item: AbstractionPredicate | Mapping[str, Any],
) -> AbstractionPredicate:
    if isinstance(item, AbstractionPredicate):
        return item
    if isinstance(item, Mapping) and "predicate_cid" in item and "schema" in item:
        return AbstractionPredicate.from_dict(item)
    if isinstance(item, Mapping):
        return AbstractionPredicate(**_strip_envelope(item, "predicate_cid"))
    raise BoundedSynthesisError("abstractions items must be objects")


def compile_synthesis_example(**fields: Any) -> SynthesisExample:
    _reject_forbidden_bodies(fields, "synthesis example")
    return SynthesisExample(**fields)


def compile_synthesis_counterexample(**fields: Any) -> SynthesisCounterexample:
    _reject_forbidden_bodies(fields, "synthesis counterexample")
    return SynthesisCounterexample(**fields)


def compile_synthesis_obligation(**fields: Any) -> SynthesisObligation:
    _reject_forbidden_bodies(fields, "synthesis obligation")
    return SynthesisObligation(**fields)


def compile_abstraction_predicate(**fields: Any) -> AbstractionPredicate:
    _reject_forbidden_bodies(fields, "abstraction predicate")
    return AbstractionPredicate(**fields)


def enumerate_templates() -> tuple[Mapping[str, Any], ...]:
    """Return the deterministic finite adapter grammar."""

    return TEMPLATE_CATALOG


def _template_covers_example(template: Mapping[str, Any], example: SynthesisExample) -> bool:
    if example.template_hint:
        return example.template_hint == template["template_id"]
    return example.kind in template["covers_example_kinds"]


def _template_covers_obligation(
    template: Mapping[str, Any], obligation: SynthesisObligation
) -> bool:
    if obligation.status != ObligationStatus.LOWERED.value:
        return False
    return obligation.kind in template["covers_obligation_kinds"]


def _abstraction_distinguishes(
    abstractions: Sequence[AbstractionPredicate],
    counterexample: SynthesisCounterexample,
) -> bool:
    if not counterexample.abstraction_gap:
        return False
    return any(
        item.predicate_id == counterexample.predicate_id and item.distinguishing
        for item in abstractions
    )


def _check_template(
    template: Mapping[str, Any],
    *,
    examples: Sequence[SynthesisExample],
    counterexamples: Sequence[SynthesisCounterexample],
    obligations: Sequence[SynthesisObligation],
    abstractions: Sequence[AbstractionPredicate],
) -> tuple[str, SynthesisCounterexample | None]:
    if examples:
        if any(not _template_covers_example(template, item) for item in examples):
            return "uncovered", None
    else:
        required = [
            item
            for item in obligations
            if item.required and item.status == ObligationStatus.LOWERED.value
        ]
        if required and any(
            not _template_covers_obligation(template, item) for item in required
        ):
            return "uncovered", None
    for counterexample in counterexamples:
        if not counterexample.can_refute:
            continue
        if counterexample.abstraction_gap and not _abstraction_distinguishes(
            abstractions, counterexample
        ):
            return "abstraction_gap", counterexample
        if counterexample.abstraction_gap:
            continue
        if counterexample.conflicts_with == template["template_id"]:
            return "refuted", counterexample
    return "covered", None


def _ingest_proof_counterexamples(
    proof_receipt: Mapping[str, Any] | Any | None,
    *,
    tree_id: str,
) -> tuple[SynthesisCounterexample, ...]:
    if proof_receipt in (None, (), {}):
        return ()
    payload = _as_mapping(proof_receipt, "SPAR-030 proof receipt")
    _reject_non_admitting_payload(payload, "SPAR-030 proof receipt")
    receipt_tree = payload.get("tree_id")
    if receipt_tree not in (None, ""):
        _require_tree(receipt_tree, tree_id, "SPAR-030")
    ingested: list[SynthesisCounterexample] = []
    for item in _mapping_sequence(payload.get("replays"), "SPAR-030 replays"):
        replayed = item.get("replayed") is True
        source_cid = item.get("replay_cid") or item.get("source_cid") or item.get(
            "artifact_cid"
        )
        if source_cid in (None, ""):
            raise BoundedSynthesisError("SPAR-030 replay source_cid is required")
        kind_raw = item.get("kind") or ExampleKind.BOUNDARY.value
        kind = (
            kind_raw
            if kind_raw in DECLARED_EXAMPLE_KINDS
            else ExampleKind.BOUNDARY.value
        )
        ingested.append(
            SynthesisCounterexample(
                counterexample_id=_text(
                    item.get("counterexample_id")
                    or f"replay:{item.get('obligation_id')}",
                    "counterexample_id",
                ),
                kind=kind,
                obligation_id=_text(item.get("obligation_id"), "obligation_id"),
                tree_id=_tree_id(item.get("tree_id") or tree_id),
                source_cid=_cid(source_cid, "source_cid"),
                source_authority=_text(
                    item.get("source_authority") or "replayed_counterexample",
                    "source_authority",
                ),
                replayed=replayed,
                conflicts_with=_text(
                    item.get("conflicts_with") or "", "conflicts_with", empty=True
                ),
                abstraction_gap=item.get("abstraction_gap") is True,
                predicate_id=_text(
                    item.get("predicate_id") or "", "predicate_id", empty=True
                ),
            )
        )
    for item in _mapping_sequence(payload.get("steps"), "SPAR-030 steps"):
        kind = item.get("kind")
        if kind != "countermodel":
            continue
        if item.get("replayed") is True:
            continue
        source_cid = item.get("artifact_cid") or item.get("source_cid")
        if source_cid in (None, ""):
            continue
        ingested.append(
            SynthesisCounterexample(
                counterexample_id=_text(
                    item.get("counterexample_id")
                    or f"raw:{item.get('obligation_id')}",
                    "counterexample_id",
                ),
                kind=ExampleKind.BOUNDARY.value,
                obligation_id=_text(item.get("obligation_id"), "obligation_id"),
                tree_id=_tree_id(item.get("tree_id") or tree_id),
                source_cid=_cid(source_cid, "source_cid"),
                source_authority="countermodel",
                replayed=False,
            )
        )
    return tuple(ingested)


def _ingest_differential_counterexamples(
    differential_receipt: Mapping[str, Any] | Any | None,
    *,
    tree_id: str,
) -> tuple[SynthesisCounterexample, ...]:
    if differential_receipt in (None, (), {}):
        return ()
    payload = _as_mapping(differential_receipt, "SPAR-028 differential receipt")
    _reject_non_admitting_payload(payload, "SPAR-028 differential receipt")
    receipt_tree = payload.get("tree_id")
    if receipt_tree not in (None, ""):
        _require_tree(receipt_tree, tree_id, "SPAR-028")
    ingested: list[SynthesisCounterexample] = []
    comparisons = payload.get("comparisons") or payload.get("mismatches") or ()
    for item in _mapping_sequence(comparisons, "SPAR-028 comparisons"):
        status = item.get("status") or item.get("verdict")
        if status not in {"mismatch", "disagree", "fail"}:
            continue
        if item.get("independently_observed") is not True:
            raise BoundedSynthesisError(
                "SPAR-028 mismatches cannot refute until independently observed"
            )
        source_cid = item.get("comparison_cid") or item.get("source_cid")
        if source_cid in (None, ""):
            raise BoundedSynthesisError("SPAR-028 comparison source_cid is required")
        dimension = _text(item.get("dimension") or ExampleKind.BOUNDARY.value, "kind")
        kind = dimension if dimension in DECLARED_EXAMPLE_KINDS else ExampleKind.BOUNDARY.value
        ingested.append(
            SynthesisCounterexample(
                counterexample_id=_text(
                    item.get("counterexample_id") or f"diff:{dimension}",
                    "counterexample_id",
                ),
                kind=kind,
                obligation_id=_text(
                    item.get("obligation_id") or "obl:differential", "obligation_id"
                ),
                tree_id=_tree_id(item.get("tree_id") or tree_id),
                source_cid=_cid(source_cid, "source_cid"),
                source_authority="differential_mismatch",
                replayed=True,
                conflicts_with=_text(
                    item.get("conflicts_with") or "", "conflicts_with", empty=True
                ),
            )
        )
    return tuple(ingested)


def _unique_counterexamples(
    items: Sequence[SynthesisCounterexample],
) -> tuple[SynthesisCounterexample, ...]:
    ordered: list[SynthesisCounterexample] = []
    seen: set[str] = set()
    for item in items:
        if item.counterexample_id in seen:
            continue
        seen.add(item.counterexample_id)
        ordered.append(item)
    if len(ordered) > MAX_COUNTEREXAMPLES:
        raise BoundedSynthesisError("counterexamples exceed maximum length")
    return tuple(ordered)


def _make_candidate(
    template: Mapping[str, Any],
    *,
    tree_id: str,
    examples: Sequence[SynthesisExample],
    counterexamples: Sequence[SynthesisCounterexample],
    obligations: Sequence[SynthesisObligation],
    abstractions: Sequence[AbstractionPredicate],
    write_paths: Sequence[str],
    source_cids: Sequence[str],
) -> AdapterCandidate:
    obligation_ids = tuple(
        item.obligation_id
        for item in obligations
        if item.status == ObligationStatus.LOWERED.value
    )
    if not obligation_ids and examples:
        obligation_ids = tuple(
            dict.fromkeys(item.obligation_id for item in examples)
        )
    return AdapterCandidate(
        candidate_id=f"{template['template_id']}:{','.join(obligation_ids) or 'none'}",
        kind=template["kind"],
        template_id=template["template_id"],
        tree_id=tree_id,
        obligation_ids=obligation_ids,
        example_cids=tuple(item.example_cid for item in examples),
        counterexample_cids=tuple(
            item.counterexample_cid for item in counterexamples if item.can_refute
        ),
        abstraction_cids=tuple(item.predicate_cid for item in abstractions),
        write_paths=write_paths,
        source_cids=source_cids,
    )


def _validation_status(value: Any) -> str:
    if value in (None, ""):
        return ValidationReentryStatus.INCOMPLETE.value
    text = _text(value, "validation status")
    if text in DECLARED_VALIDATION_REENTRY_STATUSES:
        return text
    if text == "validated":
        return ValidationReentryStatus.VALIDATED.value
    raise BoundedSynthesisError(f"unknown validation status: {text}")


def reenter_translation_validation(
    candidate: AdapterCandidate,
    *,
    validate: ValidateFn | None,
    tree_id: str,
    write_paths: Sequence[str],
    validation_commands: Sequence[str],
    packet_cid: str,
    wave_cid: str,
    selection_cid: str,
) -> ValidationReentry:
    """Re-enter SPAR-027. A pass cannot complete or promote the candidate."""

    if validate is None:
        return ValidationReentry(
            candidate_cid=candidate.candidate_cid,
            tree_id=tree_id,
            status=ValidationReentryStatus.MISSING.value,
            reason_code="validation_reentry_required",
        )
    payload = dict(
        validate(
            candidate=candidate.to_dict(),
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
    reason = _text(payload.get("reason_code") or payload.get("terminal_reason") or "", "reason_code", empty=True)
    if status == ValidationReentryStatus.VALIDATED.value:
        reason = ""
    elif not reason:
        reason = status
    return ValidationReentry(
        candidate_cid=candidate.candidate_cid,
        tree_id=tree_id,
        status=status,
        result_cid=_optional_cid(payload.get("result_cid"), "result_cid"),
        reason_code=reason,
    )


def _status_from_loop(
    *,
    candidate: AdapterCandidate | None,
    reentry: ValidationReentry | None,
    rounds: Sequence[SynthesisRound],
    examples: Sequence[SynthesisExample],
    obligations: Sequence[SynthesisObligation],
    counterexamples: Sequence[SynthesisCounterexample],
    rounds_used: int,
    max_rounds: int,
    timed_out: bool,
) -> str:
    if timed_out:
        return SynthesisStatus.TIMEOUT.value
    if reentry is not None and reentry.status == ValidationReentryStatus.VALIDATED.value:
        return SynthesisStatus.NOMINATED.value
    if reentry is not None and reentry.status == ValidationReentryStatus.UNSUPPORTED.value:
        return SynthesisStatus.BLOCKED.value
    if reentry is not None and reentry.status == ValidationReentryStatus.MISSING.value:
        return SynthesisStatus.INCOMPLETE.value
    if candidate is not None and reentry is None:
        return SynthesisStatus.INCOMPLETE.value
    if any(item.outcome == RoundOutcome.REFUTED.value for item in rounds) and candidate is None:
        if any(item.can_refute and not item.abstraction_gap for item in counterexamples):
            return SynthesisStatus.REFUTED.value
        return SynthesisStatus.INCOMPLETE.value
    if not examples and not obligations and not any(item.can_refute for item in counterexamples):
        return SynthesisStatus.UNKNOWN.value
    if rounds_used >= max_rounds and candidate is None:
        return SynthesisStatus.TIMEOUT.value
    return SynthesisStatus.INCOMPLETE.value


@dataclass(frozen=True, slots=True)
class BoundedSynthesisReceipt:
    """Bounded CEGIS/CEGAR synthesis receipt. Nomination-only."""

    tree_id: str
    packet_cid: str
    wave_cid: str
    selection_cid: str
    examples: Sequence[SynthesisExample | Mapping[str, Any]] = ()
    counterexamples: Sequence[SynthesisCounterexample | Mapping[str, Any]] = ()
    obligations: Sequence[SynthesisObligation | Mapping[str, Any]] = ()
    abstractions: Sequence[AbstractionPredicate | Mapping[str, Any]] = ()
    rounds: Sequence[SynthesisRound | Mapping[str, Any]] = ()
    candidates: Sequence[AdapterCandidate | Mapping[str, Any]] = ()
    reentry: ValidationReentry | Mapping[str, Any] | None = None
    write_paths: Sequence[str] = ()
    raw_source_cids: Sequence[str] = ()
    validation_commands: Sequence[str] = ()
    status: SynthesisStatus | str = SynthesisStatus.UNKNOWN
    max_rounds: int = DEFAULT_MAX_ROUNDS
    rounds_used: int = 0
    selector_mode: str = TEMPLATE_SELECTOR_DETERMINISTIC
    network: str = NETWORK_DENY
    analyzer_id: str = ANALYZER_ID
    synthesizer_is_nomination_only: bool = True
    unknown_remains_unknown: bool = True
    claims_general_equivalence: bool = False
    candidate_cannot_admit_proofs: bool = True
    raw_countermodel_cannot_refute: bool = True
    synthesized_candidates_reenter_validation: bool = True
    candidates_remain_unpromoted: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = BOUNDED_SYNTHESIS_RECEIPT_INTERFACE
    schema: ClassVar[str] = BOUNDED_SYNTHESIS_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "wave_cid",
            "selection_cid",
            "examples",
            "counterexamples",
            "obligations",
            "abstractions",
            "rounds",
            "candidates",
            "reentry",
            "write_paths",
            "raw_source_cids",
            "validation_commands",
            "status",
            "max_rounds",
            "rounds_used",
            "selector_mode",
            "network",
            "analyzer_id",
            "synthesizer_is_nomination_only",
            "unknown_remains_unknown",
            "claims_general_equivalence",
            "candidate_cannot_admit_proofs",
            "raw_countermodel_cannot_refute",
            "synthesized_candidates_reenter_validation",
            "candidates_remain_unpromoted",
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
        examples = tuple(_coerce_example(item, tree_id=tree_id) for item in self.examples)
        counterexamples = tuple(
            _coerce_counterexample(item, tree_id=tree_id) for item in self.counterexamples
        )
        obligations = tuple(
            _coerce_obligation(item, tree_id=tree_id) for item in self.obligations
        )
        abstractions = tuple(_coerce_abstraction(item) for item in self.abstractions)
        rounds: list[SynthesisRound] = []
        for item in self.rounds:
            if isinstance(item, SynthesisRound):
                rounds.append(item)
            elif isinstance(item, Mapping) and "round_cid" in item and "schema" in item:
                rounds.append(SynthesisRound.from_dict(item))
            elif isinstance(item, Mapping):
                rounds.append(SynthesisRound(**_strip_envelope(item, "round_cid")))
            else:
                raise BoundedSynthesisError("rounds items must be objects")
        candidates: list[AdapterCandidate] = []
        for item in self.candidates:
            if isinstance(item, AdapterCandidate):
                candidates.append(item)
            elif isinstance(item, Mapping) and "candidate_cid" in item and "schema" in item:
                candidates.append(AdapterCandidate.from_dict(item))
            elif isinstance(item, Mapping):
                candidates.append(
                    AdapterCandidate(**_strip_envelope(item, "candidate_cid"))
                )
            else:
                raise BoundedSynthesisError("candidates items must be objects")
        if len(examples) > MAX_EXAMPLES:
            raise BoundedSynthesisError("examples exceed maximum length")
        if len(counterexamples) > MAX_COUNTEREXAMPLES:
            raise BoundedSynthesisError("counterexamples exceed maximum length")
        if len(obligations) > MAX_OBLIGATIONS:
            raise BoundedSynthesisError("obligations exceed maximum length")
        if len(abstractions) > MAX_ABSTRACTIONS:
            raise BoundedSynthesisError("abstractions exceed maximum length")
        if len(candidates) > MAX_CANDIDATES:
            raise BoundedSynthesisError("candidates exceed maximum length")
        if len(rounds) > MAX_ROUNDS:
            raise BoundedSynthesisError("rounds exceed maximum length")
        reentry: ValidationReentry | None
        if self.reentry is None:
            reentry = None
        elif isinstance(self.reentry, ValidationReentry):
            reentry = self.reentry
        elif isinstance(self.reentry, Mapping) and "reentry_cid" in self.reentry:
            reentry = ValidationReentry.from_dict(self.reentry)
        elif isinstance(self.reentry, Mapping):
            payload = _strip_envelope(self.reentry, "reentry_cid")
            for flag in _AUTHORITY_FLAG_NAMES:
                payload.pop(flag, None)
            payload.pop("nomination_only", None)
            reentry = ValidationReentry(**payload)
        else:
            raise BoundedSynthesisError("reentry must be an object")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(self, "selection_cid", _cid(self.selection_cid, "selection_cid"))
        object.__setattr__(self, "examples", examples)
        object.__setattr__(self, "counterexamples", counterexamples)
        object.__setattr__(self, "obligations", obligations)
        object.__setattr__(self, "abstractions", abstractions)
        object.__setattr__(self, "rounds", tuple(rounds))
        object.__setattr__(self, "candidates", tuple(candidates))
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
            self, "status", _enum(self.status, SynthesisStatus, "status")
        )
        object.__setattr__(
            self,
            "max_rounds",
            _int(self.max_rounds, "max_rounds", minimum=1, maximum=ABSOLUTE_MAX_ROUNDS),
        )
        object.__setattr__(
            self,
            "rounds_used",
            _int(self.rounds_used, "rounds_used", minimum=0, maximum=ABSOLUTE_MAX_ROUNDS),
        )
        if self.selector_mode != TEMPLATE_SELECTOR_DETERMINISTIC:
            raise BoundedSynthesisError("template selection must remain deterministic")
        object.__setattr__(self, "selector_mode", TEMPLATE_SELECTOR_DETERMINISTIC)
        object.__setattr__(self, "network", _network_value(self.network))
        if self.analyzer_id != ANALYZER_ID:
            raise BoundedSynthesisError("receipt analyzer_id must remain SPAR-032")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if self.synthesizer_is_nomination_only is not True:
            raise BoundedSynthesisError("synthesizer must remain nomination_only")
        if self.unknown_remains_unknown is not True:
            raise BoundedSynthesisError("unknown must remain unknown")
        if self.claims_general_equivalence is not False:
            raise BoundedSynthesisError("general Python equivalence is not claimed")
        if self.candidate_cannot_admit_proofs is not True:
            raise BoundedSynthesisError("candidates cannot admit proofs")
        if self.raw_countermodel_cannot_refute is not True:
            raise BoundedSynthesisError("raw countermodels cannot refute until replay")
        if self.synthesized_candidates_reenter_validation is not True:
            raise BoundedSynthesisError("synthesized candidates must re-enter validation")
        if self.candidates_remain_unpromoted is not True:
            raise BoundedSynthesisError("synthesized candidates remain unpromoted")
        if self.mutated is not False:
            raise BoundedSynthesisError("synthesizer cannot mutate")
        if self.deterministic is not True:
            raise BoundedSynthesisError("synthesis must remain deterministic")
        object.__setattr__(self, "synthesizer_is_nomination_only", True)
        object.__setattr__(self, "unknown_remains_unknown", True)
        object.__setattr__(self, "claims_general_equivalence", False)
        object.__setattr__(self, "candidate_cannot_admit_proofs", True)
        object.__setattr__(self, "raw_countermodel_cannot_refute", True)
        object.__setattr__(self, "synthesized_candidates_reenter_validation", True)
        object.__setattr__(self, "candidates_remain_unpromoted", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)
        if any(item.unpromoted is not True for item in candidates):
            raise BoundedSynthesisError("synthesized candidates remain unpromoted")

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
        return self.status != SynthesisStatus.NOMINATED.value

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BOUNDED_SYNTHESIS_RECEIPT_SCHEMA,
            "interface": BOUNDED_SYNTHESIS_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "wave_cid": self.wave_cid,
            "selection_cid": self.selection_cid,
            "examples": [item.identity_payload() for item in self.examples],
            "counterexamples": [item.identity_payload() for item in self.counterexamples],
            "obligations": [item.identity_payload() for item in self.obligations],
            "abstractions": [item.identity_payload() for item in self.abstractions],
            "rounds": [item.identity_payload() for item in self.rounds],
            "candidates": [item.identity_payload() for item in self.candidates],
            "reentry": None if self.reentry is None else self.reentry.identity_payload(),
            "write_paths": list(self.write_paths),
            "raw_source_cids": list(self.raw_source_cids),
            "validation_commands": list(self.validation_commands),
            "status": self.status,
            "max_rounds": self.max_rounds,
            "rounds_used": self.rounds_used,
            "selector_mode": TEMPLATE_SELECTOR_DETERMINISTIC,
            "network": NETWORK_DENY,
            "analyzer_id": ANALYZER_ID,
            "synthesizer_is_nomination_only": True,
            "unknown_remains_unknown": True,
            "claims_general_equivalence": False,
            "candidate_cannot_admit_proofs": True,
            "raw_countermodel_cannot_refute": True,
            "synthesized_candidates_reenter_validation": True,
            "candidates_remain_unpromoted": True,
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
        payload["examples"] = [item.to_dict() for item in self.examples]
        payload["counterexamples"] = [item.to_dict() for item in self.counterexamples]
        payload["obligations"] = [item.to_dict() for item in self.obligations]
        payload["abstractions"] = [item.to_dict() for item in self.abstractions]
        payload["rounds"] = [item.to_dict() for item in self.rounds]
        payload["candidates"] = [item.to_dict() for item in self.candidates]
        payload["reentry"] = None if self.reentry is None else self.reentry.to_dict()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundedSynthesisReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != BOUNDED_SYNTHESIS_RECEIPT_SCHEMA:
            raise BoundedSynthesisError("unsupported BoundedSynthesisReceipt schema")
        if payload.pop("interface") != BOUNDED_SYNTHESIS_RECEIPT_INTERFACE:
            raise BoundedSynthesisError("unsupported BoundedSynthesisReceipt interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("synthesizer_is_nomination_only") is not True:
            raise BoundedSynthesisError("synthesizer must remain nomination_only")
        if payload.pop("unknown_remains_unknown") is not True:
            raise BoundedSynthesisError("unknown must remain unknown")
        if payload.pop("claims_general_equivalence") is not False:
            raise BoundedSynthesisError("general Python equivalence is not claimed")
        if payload.pop("candidate_cannot_admit_proofs") is not True:
            raise BoundedSynthesisError("candidates cannot admit proofs")
        if payload.pop("raw_countermodel_cannot_refute") is not True:
            raise BoundedSynthesisError("raw countermodels cannot refute until replay")
        if payload.pop("synthesized_candidates_reenter_validation") is not True:
            raise BoundedSynthesisError("synthesized candidates must re-enter validation")
        if payload.pop("candidates_remain_unpromoted") is not True:
            raise BoundedSynthesisError("synthesized candidates remain unpromoted")
        if payload.pop("mutated") is not False:
            raise BoundedSynthesisError("synthesizer cannot mutate")
        if payload.pop("deterministic") is not True:
            raise BoundedSynthesisError("synthesis must remain deterministic")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise BoundedSynthesisError("receipt analyzer_id must remain SPAR-032")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


def run_bounded_synthesis(
    *,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    examples: Sequence[SynthesisExample | Mapping[str, Any]] = (),
    counterexamples: Sequence[SynthesisCounterexample | Mapping[str, Any]] = (),
    obligations: Sequence[SynthesisObligation | Mapping[str, Any]] = (),
    abstractions: Sequence[AbstractionPredicate | Mapping[str, Any]] = (),
    proof_receipt: Mapping[str, Any] | Any | None = None,
    differential_receipt: Mapping[str, Any] | Any | None = None,
    validate: ValidateFn | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    mutate: bool = False,
    vector_evidence: Any = None,
    network: str = NETWORK_DENY,
) -> BoundedSynthesisReceipt:
    """Run bounded CEGIS/CEGAR adapter synthesis and re-enter validation."""

    if mutate is not False:
        raise BoundedSynthesisError("synthesizer cannot mutate")
    _reject_vector_admission(vector_evidence)
    _network_value(network)
    bound = _int(max_rounds, "max_rounds", minimum=1, maximum=ABSOLUTE_MAX_ROUNDS)
    (
        tree_id,
        wave_cid,
        selection_cid,
        packet_cid,
        write_paths,
        raw_source_cids,
        validation_commands,
    ) = _bind_predecessors(wave=wave, selection=selection)

    typed_examples = tuple(_coerce_example(item, tree_id=tree_id) for item in examples)
    if len(typed_examples) > MAX_EXAMPLES:
        raise BoundedSynthesisError("examples exceed maximum length")
    seen_examples: set[str] = set()
    for item in typed_examples:
        if item.example_id in seen_examples:
            raise BoundedSynthesisError("example_id must be unique")
        seen_examples.add(item.example_id)
        if item.source_cid not in raw_source_cids:
            raise BoundedSynthesisError("example source_cid must be a declared raw source")

    typed_obligations = tuple(
        _coerce_obligation(item, tree_id=tree_id) for item in obligations
    )
    if len(typed_obligations) > DEFAULT_MAX_OBLIGATIONS:
        raise BoundedSynthesisError("obligations exceed search bound")
    seen_obligations: set[str] = set()
    for item in typed_obligations:
        if item.obligation_id in seen_obligations:
            raise BoundedSynthesisError("obligation_id must be unique")
        seen_obligations.add(item.obligation_id)

    typed_counterexamples = [
        _coerce_counterexample(item, tree_id=tree_id) for item in counterexamples
    ]
    typed_counterexamples.extend(_ingest_proof_counterexamples(proof_receipt, tree_id=tree_id))
    typed_counterexamples.extend(
        _ingest_differential_counterexamples(differential_receipt, tree_id=tree_id)
    )
    counterexample_set = _unique_counterexamples(typed_counterexamples)

    typed_abstractions = [_coerce_abstraction(item) for item in abstractions]
    if len(typed_abstractions) > MAX_ABSTRACTIONS:
        raise BoundedSynthesisError("abstractions exceed maximum length")

    rounds: list[SynthesisRound] = []
    candidates: list[AdapterCandidate] = []
    reentry: ValidationReentry | None = None
    surviving: AdapterCandidate | None = None
    timed_out = False
    rounds_used = 0
    templates = enumerate_templates()

    if (
        not typed_examples
        and not typed_obligations
        and not any(item.can_refute for item in counterexample_set)
    ):
        return BoundedSynthesisReceipt(
            tree_id=tree_id,
            packet_cid=packet_cid,
            wave_cid=wave_cid,
            selection_cid=selection_cid,
            examples=typed_examples,
            counterexamples=counterexample_set,
            obligations=typed_obligations,
            abstractions=tuple(typed_abstractions),
            rounds=(),
            candidates=(),
            write_paths=write_paths,
            raw_source_cids=raw_source_cids,
            validation_commands=validation_commands,
            status=SynthesisStatus.UNKNOWN.value,
            max_rounds=bound,
            rounds_used=0,
        )

    while rounds_used < bound and surviving is None:
        rounds_used += 1
        refined = False
        covered_template: Mapping[str, Any] | None = None
        covered_ce: tuple[str, ...] = tuple(
            item.counterexample_cid for item in counterexample_set if item.can_refute
        )
        abstraction_cids = tuple(item.predicate_cid for item in typed_abstractions)
        for template in templates:
            outcome, witness = _check_template(
                template,
                examples=typed_examples,
                counterexamples=counterexample_set,
                obligations=typed_obligations,
                abstractions=typed_abstractions,
            )
            if outcome == "uncovered":
                continue
            if outcome == "abstraction_gap" and witness is not None:
                if _abstraction_distinguishes(typed_abstractions, witness):
                    continue
                typed_abstractions.append(
                    AbstractionPredicate(
                        predicate_id=witness.predicate_id,
                        kind=AbstractionKind.TYPE.value
                        if witness.kind != ExampleKind.STATE.value
                        else AbstractionKind.STATE.value,
                        grain=AbstractionGrain.REFINED.value,
                        distinguishing=True,
                    )
                )
                rounds.append(
                    SynthesisRound(
                        round_index=rounds_used,
                        mode=RoundMode.CEGAR.value,
                        template_id=template["template_id"],
                        outcome=RoundOutcome.REFINED.value,
                        counterexample_cids=(witness.counterexample_cid,),
                        abstraction_cids=tuple(
                            item.predicate_cid for item in typed_abstractions
                        ),
                        reason_code="abstraction_gap",
                    )
                )
                refined = True
                break
            if outcome == "refuted" and witness is not None:
                rounds.append(
                    SynthesisRound(
                        round_index=rounds_used,
                        mode=RoundMode.CEGIS.value,
                        template_id=template["template_id"],
                        outcome=RoundOutcome.REFUTED.value,
                        counterexample_cids=(witness.counterexample_cid,),
                        abstraction_cids=abstraction_cids,
                        reason_code="replayed_counterexample",
                    )
                )
                continue
            covered_template = template
            break
        if refined:
            continue
        if covered_template is None:
            if rounds_used >= bound:
                timed_out = True
            break
        candidate = _make_candidate(
            covered_template,
            tree_id=tree_id,
            examples=typed_examples,
            counterexamples=counterexample_set,
            obligations=typed_obligations,
            abstractions=typed_abstractions,
            write_paths=write_paths,
            source_cids=raw_source_cids,
        )
        candidates.append(candidate)
        rounds.append(
            SynthesisRound(
                round_index=rounds_used,
                mode=RoundMode.CEGIS.value,
                template_id=covered_template["template_id"],
                outcome=RoundOutcome.COVERED.value,
                candidate_cid=candidate.candidate_cid,
                counterexample_cids=covered_ce,
                abstraction_cids=tuple(item.predicate_cid for item in typed_abstractions),
                reason_code="covered",
            )
        )
        reentry = reenter_translation_validation(
            candidate,
            validate=validate,
            tree_id=tree_id,
            write_paths=write_paths,
            validation_commands=validation_commands,
            packet_cid=packet_cid,
            wave_cid=wave_cid,
            selection_cid=selection_cid,
        )
        if reentry.status == ValidationReentryStatus.VALIDATED.value:
            rounds.append(
                SynthesisRound(
                    round_index=rounds_used,
                    mode=RoundMode.VALIDATION.value,
                    template_id=covered_template["template_id"],
                    outcome=RoundOutcome.VALIDATED.value,
                    candidate_cid=candidate.candidate_cid,
                    counterexample_cids=covered_ce,
                    abstraction_cids=tuple(
                        item.predicate_cid for item in typed_abstractions
                    ),
                    reason_code="",
                )
            )
            surviving = candidate
            break
        if reentry.status == ValidationReentryStatus.MISSING.value:
            surviving = candidate
            break
        if reentry.status == ValidationReentryStatus.UNSUPPORTED.value:
            break
        rejected_ce = SynthesisCounterexample(
            counterexample_id=f"validation:{covered_template['template_id']}:{rounds_used}",
            kind=ExampleKind.BOUNDARY.value
            if covered_template["template_id"]
            in {TemplateKind.IDENTITY.value, TemplateKind.WRAPPER.value}
            else (
                ExampleKind.STATE.value
                if covered_template["template_id"] == TemplateKind.PROJECTION.value
                else ExampleKind.GUARD.value
                if covered_template["template_id"] == TemplateKind.GUARD.value
                else ExampleKind.PROTOCOL.value
                if covered_template["template_id"] == TemplateKind.PROTOCOL.value
                else ExampleKind.INITIALIZATION.value
            ),
            obligation_id=candidate.obligation_ids[0]
            if candidate.obligation_ids
            else "obl:validation",
            tree_id=tree_id,
            source_cid=reentry.result_cid or raw_source_cids[0],
            source_authority="validation_failure",
            replayed=True,
            conflicts_with=covered_template["template_id"],
        )
        counterexample_set = _unique_counterexamples(
            (*counterexample_set, rejected_ce)
        )
        rounds.append(
            SynthesisRound(
                round_index=rounds_used,
                mode=RoundMode.VALIDATION.value,
                template_id=covered_template["template_id"],
                outcome=RoundOutcome.VALIDATION_REJECTED.value,
                candidate_cid=candidate.candidate_cid,
                counterexample_cids=(rejected_ce.counterexample_cid,),
                abstraction_cids=tuple(item.predicate_cid for item in typed_abstractions),
                reason_code=reentry.reason_code or "validation_rejected",
            )
        )
        reentry = None
        if rounds_used >= bound:
            timed_out = True

    if rounds_used >= bound and surviving is None and not timed_out:
        if any(item.outcome == RoundOutcome.COVERED.value for item in rounds):
            timed_out = False
        elif any(item.outcome == RoundOutcome.REFUTED.value for item in rounds):
            timed_out = False
        else:
            timed_out = True
            rounds.append(
                SynthesisRound(
                    round_index=bound,
                    mode=RoundMode.CEGIS.value,
                    template_id=TemplateKind.IDENTITY.value,
                    outcome=RoundOutcome.TIMEOUT.value,
                    reason_code="max_rounds",
                )
            )

    status = _status_from_loop(
        candidate=surviving,
        reentry=reentry,
        rounds=rounds,
        examples=typed_examples,
        obligations=typed_obligations,
        counterexamples=counterexample_set,
        rounds_used=rounds_used,
        max_rounds=bound,
        timed_out=timed_out,
    )
    return BoundedSynthesisReceipt(
        tree_id=tree_id,
        packet_cid=packet_cid,
        wave_cid=wave_cid,
        selection_cid=selection_cid,
        examples=typed_examples,
        counterexamples=counterexample_set,
        obligations=typed_obligations,
        abstractions=tuple(typed_abstractions),
        rounds=rounds,
        candidates=tuple(candidates),
        reentry=reentry,
        write_paths=write_paths,
        raw_source_cids=raw_source_cids,
        validation_commands=validation_commands,
        status=status,
        max_rounds=bound,
        rounds_used=rounds_used,
    )


def dry_run_bounded_synthesis(
    *,
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    examples: Sequence[SynthesisExample | Mapping[str, Any]] = (),
    counterexamples: Sequence[SynthesisCounterexample | Mapping[str, Any]] = (),
    obligations: Sequence[SynthesisObligation | Mapping[str, Any]] = (),
    abstractions: Sequence[AbstractionPredicate | Mapping[str, Any]] = (),
    proof_receipt: Mapping[str, Any] | Any | None = None,
    differential_receipt: Mapping[str, Any] | Any | None = None,
    validate: ValidateFn | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    vector_evidence: Any = None,
) -> BoundedSynthesisReceipt:
    """Deterministic non-mutating bounded CEGIS/CEGAR synthesis."""

    result = run_bounded_synthesis(
        wave=wave,
        selection=selection,
        examples=examples,
        counterexamples=counterexamples,
        obligations=obligations,
        abstractions=abstractions,
        proof_receipt=proof_receipt,
        differential_receipt=differential_receipt,
        validate=validate,
        max_rounds=max_rounds,
        mutate=False,
        vector_evidence=vector_evidence,
    )
    if result.mutated is not False or result.deterministic is not True:
        raise BoundedSynthesisError("dry-run must remain deterministic and non-mutating")
    return result


class BoundaryAdapterSynthesizer:
    """SPAR-032 bounded CEGIS/CEGAR synthesizer. Nomination-only."""

    interface: ClassVar[str] = BOUNDARY_ADAPTER_SYNTHESIZER_INTERFACE
    schema: ClassVar[str] = BOUNDARY_ADAPTER_SYNTHESIZER_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID
    receipt_interface: ClassVar[str] = BOUNDED_SYNTHESIS_RECEIPT_INTERFACE

    def synthesize(
        self,
        *,
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        examples: Sequence[SynthesisExample | Mapping[str, Any]] = (),
        counterexamples: Sequence[SynthesisCounterexample | Mapping[str, Any]] = (),
        obligations: Sequence[SynthesisObligation | Mapping[str, Any]] = (),
        abstractions: Sequence[AbstractionPredicate | Mapping[str, Any]] = (),
        proof_receipt: Mapping[str, Any] | Any | None = None,
        differential_receipt: Mapping[str, Any] | Any | None = None,
        validate: ValidateFn | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        vector_evidence: Any = None,
    ) -> BoundedSynthesisReceipt:
        return run_bounded_synthesis(
            wave=wave,
            selection=selection,
            examples=examples,
            counterexamples=counterexamples,
            obligations=obligations,
            abstractions=abstractions,
            proof_receipt=proof_receipt,
            differential_receipt=differential_receipt,
            validate=validate,
            max_rounds=max_rounds,
            vector_evidence=vector_evidence,
        )

    def dry_run(
        self,
        *,
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        examples: Sequence[SynthesisExample | Mapping[str, Any]] = (),
        counterexamples: Sequence[SynthesisCounterexample | Mapping[str, Any]] = (),
        obligations: Sequence[SynthesisObligation | Mapping[str, Any]] = (),
        abstractions: Sequence[AbstractionPredicate | Mapping[str, Any]] = (),
        proof_receipt: Mapping[str, Any] | Any | None = None,
        differential_receipt: Mapping[str, Any] | Any | None = None,
        validate: ValidateFn | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        vector_evidence: Any = None,
    ) -> BoundedSynthesisReceipt:
        return dry_run_bounded_synthesis(
            wave=wave,
            selection=selection,
            examples=examples,
            counterexamples=counterexamples,
            obligations=obligations,
            abstractions=abstractions,
            proof_receipt=proof_receipt,
            differential_receipt=differential_receipt,
            validate=validate,
            max_rounds=max_rounds,
            vector_evidence=vector_evidence,
        )

    def reenter(
        self,
        candidate: AdapterCandidate,
        *,
        validate: ValidateFn | None,
        tree_id: str,
        write_paths: Sequence[str],
        validation_commands: Sequence[str],
        packet_cid: str,
        wave_cid: str,
        selection_cid: str,
    ) -> ValidationReentry:
        return reenter_translation_validation(
            candidate,
            validate=validate,
            tree_id=tree_id,
            write_paths=write_paths,
            validation_commands=validation_commands,
            packet_cid=packet_cid,
            wave_cid=wave_cid,
            selection_cid=selection_cid,
        )


def encode_canonical_receipt(receipt: BoundedSynthesisReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> BoundedSynthesisReceipt:
    return BoundedSynthesisReceipt.from_dict(payload)


def encode_canonical_candidate(candidate: AdapterCandidate) -> dict[str, Any]:
    return candidate.to_dict()


def decode_canonical_candidate(payload: Mapping[str, Any]) -> AdapterCandidate:
    return AdapterCandidate.from_dict(payload)


def encode_canonical_reentry(reentry: ValidationReentry) -> dict[str, Any]:
    return reentry.to_dict()


def decode_canonical_reentry(payload: Mapping[str, Any]) -> ValidationReentry:
    return ValidationReentry.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise BoundedSynthesisError(
            f"bounded synthesis must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_CANDIDATE_INTERFACE",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "BOUNDED_SYNTHESIS_RECEIPT_INTERFACE",
    "BOUNDARY_ADAPTER_SYNTHESIZER_INTERFACE",
    "CANDIDATES_REMAIN_UNPROMOTED",
    "CANDIDATE_CANNOT_ADMIT_PROOFS",
    "DECLARED_ADAPTER_ARTIFACT_KINDS",
    "DECLARED_ROUND_MODES",
    "DECLARED_ROUND_OUTCOMES",
    "DECLARED_SYNTHESIS_STATUSES",
    "DECLARED_TEMPLATE_KINDS",
    "DECLARED_VALIDATION_REENTRY_STATUSES",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "FORBIDDEN_SYNTHESIS_NAMES",
    "GENERAL_PYTHON_EQUIVALENCE_CLAIMED",
    "GOAL_ID",
    "GUESSED_AXIOMS_REJECTED",
    "IDENTITY_EXCLUDED_FIELDS",
    "IMPLICIT_INSTALL_FORBIDDEN",
    "IMPLICIT_NETWORK_FORBIDDEN",
    "INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_COUNTERMODEL_CANNOT_REFUTE",
    "RAW_SOURCE_REQUIRED",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "SYNTHESIS_CAN_AUTHORIZE_COMPLETION",
    "SYNTHESIS_CAN_AUTHORIZE_TRANSITION",
    "SYNTHESIS_CAN_CREATE_AUTHORITY",
    "SYNTHESIS_CAN_CREATE_PROOF_AUTHORITY",
    "SYNTHESIS_CONTRACT_VERSION",
    "SYNTHESIS_COUNTEREXAMPLE_INTERFACE",
    "SYNTHESIS_EXAMPLE_INTERFACE",
    "SYNTHESIS_OBLIGATION_INTERFACE",
    "SYNTHESIS_ROUND_INTERFACE",
    "SYNTHESIZED_CANDIDATES_REENTER_VALIDATION",
    "SYNTHESIZER_IS_NOMINATION_ONLY",
    "TASK_ID",
    "TEMPLATE_CATALOG",
    "TEMPLATE_SELECTOR_DETERMINISTIC",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TEST_PASS_IS_NOT_PROOF",
    "UNKNOWN_REMAINS_UNKNOWN",
    "VALIDATION_REENTRY_INTERFACE",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "ABSTRACTION_PREDICATE_INTERFACE",
    "AdapterArtifactKind",
    "AdapterCandidate",
    "AbstractionPredicate",
    "BoundedSynthesisError",
    "BoundedSynthesisReceipt",
    "BoundaryAdapterSynthesizer",
    "RoundMode",
    "RoundOutcome",
    "SynthesisCounterexample",
    "SynthesisExample",
    "SynthesisObligation",
    "SynthesisRound",
    "SynthesisStatus",
    "TemplateKind",
    "ValidationReentry",
    "ValidationReentryStatus",
    "assert_not_competing_capsule_family",
    "bounded_synthesis_cid_profile",
    "bounded_synthesis_descriptor",
    "compile_abstraction_predicate",
    "compile_synthesis_counterexample",
    "compile_synthesis_example",
    "compile_synthesis_obligation",
    "decode_canonical_candidate",
    "decode_canonical_receipt",
    "decode_canonical_reentry",
    "dry_run_bounded_synthesis",
    "encode_canonical_candidate",
    "encode_canonical_receipt",
    "encode_canonical_reentry",
    "enumerate_templates",
    "provider_free_exports",
    "reenter_translation_validation",
    "run_bounded_synthesis",
]
