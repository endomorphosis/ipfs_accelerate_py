"""SPAR-030 Tactician/Hammer remodularization adapter for boundary obligations.

This module extends current supervisor proof authorities with
``TacticianHammerRemodularizationAdapter@1``.  It consumes SPAR-016 boundary
contracts, SPAR-025 extraction-wave receipts, and SPAR-026 selection mappings,
then:

* builds content-addressed, body-free premise corpora;
* lowers and decomposes finite assume/guarantee obligations;
* runs bounded proof/countermodel search;
* reconstructs proofs only through reconstructed-proof evidence; and
* independently replays supported countermodels.

Production Tactician and Hammer remain the proof authorities.  This adapter
does not mint a competing proof, kernel, or completion authority.  Proof
candidates cannot admit proofs.  Raw solver countermodels cannot refute until
independent replay.  Unknown remains unknown.  Unsupported required behavior
is a typed terminal, never success.

The adapter is nomination-only.  Vector, model, and heuristic evidence cannot
admit a proof or suppress raw-source fallback.  Observational metadata is
excluded from identity.  Dry-run is deterministic and never mutates.  Network
is denied.
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


TASK_ID: Final[str] = "SPAR-030"
GOAL_ID: Final[str] = "SPAR-G053"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "proof search"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.proof_adapter@1"
)
TACTICIAN_AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
HAMMER_AUTHORITY_OWNER: Final[str] = "ipfs_datasets_py"

TACTICIAN_HAMMER_ADAPTER_INTERFACE: Final[str] = (
    "TacticianHammerRemodularizationAdapter@1"
)
PREMISE_CORPUS_INTERFACE: Final[str] = "PremiseCorpus@1"
BOUNDARY_OBLIGATION_INTERFACE: Final[str] = "BoundaryObligation@1"
OBLIGATION_DECOMPOSITION_INTERFACE: Final[str] = "ObligationDecomposition@1"
PROOF_RECONSTRUCTION_INTERFACE: Final[str] = "ProofReconstruction@1"
COUNTERMODEL_REPLAY_INTERFACE: Final[str] = "CountermodelReplay@1"
BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE: Final[str] = "BoundedProofSearchReceipt@1"

TACTICIAN_HAMMER_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-remodularization-adapter@1"
)
PREMISE_CORPUS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-premise-corpus@1"
)
BOUNDARY_OBLIGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-boundary-obligation@1"
)
OBLIGATION_DECOMPOSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-obligation-decomposition@1"
)
PROOF_RECONSTRUCTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-proof-reconstruction@1"
)
COUNTERMODEL_REPLAY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-countermodel-replay@1"
)
BOUNDED_PROOF_SEARCH_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/bounded-proof-search-receipt@1"
)

PROOF_ADAPTER_CONTRACT_VERSION: Final[str] = "1"

PROOF_ADAPTER_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
PROOF_ADAPTER_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
PROOF_ADAPTER_CAN_CREATE_AUTHORITY: Final[bool] = False
PROOF_ADAPTER_CAN_CREATE_PROOF_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
TEST_PASS_IS_NOT_PROOF: Final[bool] = True
PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS: Final[bool] = True
RAW_COUNTERMODEL_CANNOT_REFUTE: Final[bool] = True
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
TACTICIAN_OWNS_SEARCH: Final[bool] = True
HAMMER_OWNS_PRODUCTION_PROOF: Final[bool] = True

NETWORK_DENY: Final[str] = "deny"
PREMISE_SELECTOR_DETERMINISTIC: Final[str] = "deterministic"
DEFAULT_MAX_STEPS: Final[int] = 8
ABSOLUTE_MAX_STEPS: Final[int] = 64
DEFAULT_MAX_OBLIGATIONS: Final[int] = 64

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_PREMISES: Final[int] = 256
MAX_OBLIGATIONS: Final[int] = 256
MAX_RESIDUALS: Final[int] = 128
MAX_CLAUSES: Final[int] = 256

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

FORBIDDEN_PROOF_NAMES: Final[frozenset[str]] = frozenset(
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

FINITE_CLAUSE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "input",
        "output",
        "condition",
        "invariant",
        "assumption",
        "guarantee",
        "exception",
        "effect",
        "state",
        "authorization",
        "resource",
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

ADMITTED_CONTRACT_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "admitted",
        "proof",
        "retrieval",
        "review",
        "abstention",
    }
)

INCOMPLETE_CONTRACT_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "incomplete",
        "incomplete_contract",
        "undispositioned",
        "missing",
    }
)

UNSUPPORTED_CONTRACT_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "unsupported",
        "unsupported_required",
        "unknown_required",
    }
)

AXIOM_SOURCE_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "specification",
        "exact_static_fact",
        "reconstructed_proof",
        "admitted_contract",
    }
)

FORBIDDEN_AXIOM_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
        "guessed",
        "retrieved_text",
        "natural_language",
        "proof_candidate",
        "test",
    }
)


class ProofAdapterError(ValueError):
    """Fail-closed violation of a SPAR-030 Tactician/Hammer adapter contract."""

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


class ClauseKind(str, Enum):
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


class ClausePolarity(str, Enum):
    ASSUME = "assume"
    GUARANTEE = "guarantee"


class ObligationStatus(str, Enum):
    LOWERED = "lowered"
    RESIDUAL = "residual"
    UNSUPPORTED = "unsupported"
    INCOMPLETE = "incomplete"


class SearchOutcome(str, Enum):
    VERIFIED = "verified"
    CANDIDATE = "candidate"
    COUNTEREXAMPLE = "counterexample"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    STALE = "stale"
    ERROR = "error"


class SearchStatus(str, Enum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ArtifactKind(str, Enum):
    RECONSTRUCTED_PROOF = "reconstructed_proof"
    PROOF_CANDIDATE = "proof_candidate"
    COUNTERMODEL = "countermodel"
    REPLAYED_COUNTEREXAMPLE = "replayed_counterexample"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    STALE = "stale"
    ERROR = "error"


class Conclusiveness(str, Enum):
    CONCLUSIVE_PROOF = "conclusive_proof"
    CONCLUSIVE_REFUTATION = "conclusive_refutation"
    NON_CONCLUSIVE = "non_conclusive"
    DIAGNOSTIC = "diagnostic"


DECLARED_CLAUSE_KINDS: Final[frozenset[str]] = frozenset(item.value for item in ClauseKind)
DECLARED_CLAUSE_POLARITIES: Final[frozenset[str]] = frozenset(
    item.value for item in ClausePolarity
)
DECLARED_OBLIGATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in ObligationStatus
)
DECLARED_SEARCH_OUTCOMES: Final[frozenset[str]] = frozenset(
    item.value for item in SearchOutcome
)
DECLARED_SEARCH_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in SearchStatus
)
DECLARED_ARTIFACT_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in ArtifactKind
)
DECLARED_CONCLUSIVENESS: Final[frozenset[str]] = frozenset(
    item.value for item in Conclusiveness
)

CONCLUSIVE_PROOF_KINDS: Final[frozenset[str]] = frozenset(
    {ArtifactKind.RECONSTRUCTED_PROOF.value}
)
CONCLUSIVE_REFUTATION_KINDS: Final[frozenset[str]] = frozenset(
    {ArtifactKind.REPLAYED_COUNTEREXAMPLE.value}
)
NON_CONCLUSIVE_KINDS: Final[frozenset[str]] = frozenset(
    {
        ArtifactKind.PROOF_CANDIDATE.value,
        ArtifactKind.COUNTERMODEL.value,
        ArtifactKind.TIMEOUT.value,
        ArtifactKind.UNSUPPORTED.value,
        ArtifactKind.UNAVAILABLE.value,
        ArtifactKind.UNKNOWN.value,
        ArtifactKind.STALE.value,
        ArtifactKind.ERROR.value,
    }
)


HammerSearchFn = Callable[..., Mapping[str, Any]]
TacticianSearchFn = Callable[..., Mapping[str, Any]]
ReplayFn = Callable[..., Mapping[str, Any]]
ReconstructFn = Callable[..., Mapping[str, Any]]


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise ProofAdapterError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ProofAdapterError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise ProofAdapterError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise ProofAdapterError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise ProofAdapterError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise ProofAdapterError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProofAdapterError(f"{name} must be a boolean")
    return value


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int) -> int:
    if type(value) is bool or type(value) is not int:
        raise ProofAdapterError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ProofAdapterError(f"{name} is out of bounds")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ProofAdapterError("tree_id must be a lowercase hex Git tree identity")
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise ProofAdapterError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise ProofAdapterError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise ProofAdapterError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise ProofAdapterError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise ProofAdapterError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _reject_forbidden_bodies(payload: Any, name: str) -> None:
    if isinstance(payload, Mapping) and not isinstance(payload, (str, bytes, bytearray)):
        present = _FORBIDDEN_BODY_KEYS & set(payload)
        if present:
            raise ProofAdapterError(
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
        raise ProofAdapterError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise ProofAdapterError(f"{name} does not verify")


def _enum(value: Any, enum_type: type[Enum], name: str) -> str:
    if isinstance(value, enum_type):
        return value.value
    text = _text(value, name)
    try:
        return enum_type(text).value
    except ValueError as exc:
        raise ProofAdapterError(f"unknown {name}: {text}") from exc


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise ProofAdapterError(f"{name} cannot claim {flag}")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ProofAdapterError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise ProofAdapterError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise ProofAdapterError(f"{name} must not contain duplicates")
    return ordered


def _ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ProofAdapterError(f"{name} must be a list")
    ordered = tuple(_text(item, name) for item in values)
    if len(ordered) > limit:
        raise ProofAdapterError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise ProofAdapterError(f"{name} must not contain duplicates")
    return ordered


def _cids(values: Any, name: str, *, required: bool = False) -> tuple[str, ...]:
    if values in (None, ()):
        if required:
            raise ProofAdapterError(f"{name} are required")
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProofAdapterError(f"{name} must be a list")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise ProofAdapterError(f"{name} are required")
    if len(ordered) != len(set(ordered)):
        raise ProofAdapterError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise ProofAdapterError(f"{name} exceed maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise ProofAdapterError(f"{name} exceeds path bound")
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
        raise ProofAdapterError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise ProofAdapterError(f"{name} must be a normalized repository-relative path")
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProofAdapterError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ProofAdapterError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise ProofAdapterError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ProofAdapterError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise ProofAdapterError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise ProofAdapterError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise ProofAdapterError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise ProofAdapterError(f"missing {name}")
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
    raise ProofAdapterError(f"{name} must be a mapping")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProofAdapterError(f"{name} must be a list")
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
        raise ProofAdapterError(f"{name} items must be objects")
    if len(items) > MAX_MEMBERS:
        raise ProofAdapterError(f"{name} exceeds maximum length")
    return tuple(items)


def _require_tree(value: Any, expected: str, label: str) -> None:
    if value in (None, ""):
        raise ProofAdapterError(f"{label} tree_id is required")
    actual = _tree_id(value)
    if actual != expected:
        raise ProofAdapterError(f"{label} tree_id does not match")


def _reject_non_admitting_payload(payload: Mapping[str, Any], name: str) -> None:
    evidence = payload.get("evidence_class")
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise ProofAdapterError(f"{name} {evidence} cannot admit")
    if payload.get("admit_proof") is True or payload.get("admits_proof") is True:
        raise ProofAdapterError(f"{name} cannot admit proofs")
    if payload.get("admit_equivalence") is True:
        raise ProofAdapterError(f"{name} cannot admit")
    if payload.get("guessed_axiom") is True:
        raise ProofAdapterError("guessed axioms are rejected")


def _reject_vector_admission(vector_evidence: Any) -> None:
    if vector_evidence in (None, (), {}):
        return
    if isinstance(vector_evidence, Mapping):
        if vector_evidence.get("suppress_raw_source") is True:
            raise ProofAdapterError("vectors cannot suppress raw-source fallback")
        _reject_non_admitting_payload(vector_evidence, "vector_evidence")
        evidence = vector_evidence.get("evidence_class")
        if evidence in _NON_ADMITTING_EVIDENCE:
            raise ProofAdapterError("vector/model/heuristic evidence cannot admit")
        return
    raise ProofAdapterError("vector_evidence must be an object")


def _network_value(value: Any) -> str:
    if value in (None, ""):
        return NETWORK_DENY
    text = _text(value, "network")
    if text != NETWORK_DENY:
        raise ProofAdapterError("network is denied")
    return NETWORK_DENY


def proof_adapter_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def proof_adapter_descriptor() -> dict[str, Any]:
    return {
        "interface": TACTICIAN_HAMMER_ADAPTER_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "raw_source_required": True,
        "nomination_only": True,
        "network": NETWORK_DENY,
        "claims_general_equivalence": False,
        "proof_candidate_cannot_admit_proofs": True,
        "raw_countermodel_cannot_refute": True,
        "unknown_remains_unknown": True,
        "guessed_axioms_rejected": True,
        "tactician_owns_search": True,
        "hammer_owns_production_proof": True,
        "forbids": tuple(sorted(FORBIDDEN_PROOF_NAMES)),
    }


def _wave_cid(wave: Mapping[str, Any]) -> str:
    value = wave.get("receipt_cid") or wave.get("wave_cid")
    if value in (None, ""):
        raise ProofAdapterError("SPAR-025 receipt_cid is required")
    return _cid(value, "SPAR-025 receipt_cid")


def _wave_packet_cids(wave: Mapping[str, Any]) -> tuple[str, ...]:
    values = wave.get("packet_cids")
    if values in (None, ()):
        raise ProofAdapterError("SPAR-025 packet_cids are required")
    return _cids(values, "SPAR-025 packet_cids", required=True)


def _wave_write_paths(wave: Mapping[str, Any]) -> tuple[str, ...]:
    values = wave.get("write_paths")
    if values in (None, ()):
        raise ProofAdapterError("SPAR-025 write_paths are required")
    return _exact_paths(values, "SPAR-025 write_paths")


def _selection_cid(selection: Mapping[str, Any]) -> str:
    value = selection.get("validation_selection_cid") or selection.get("selection_cid")
    if value in (None, ""):
        raise ProofAdapterError("SPAR-026 validation_selection_cid is required")
    return _cid(value, "SPAR-026 validation_selection_cid")


def _selection_packet_cid(selection: Mapping[str, Any]) -> str:
    value = selection.get("packet_cid")
    if value in (None, ""):
        raise ProofAdapterError("SPAR-026 packet_cid is required")
    return _cid(value, "SPAR-026 packet_cid")


def _selection_write_paths(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("write_paths")
    if values in (None, ()):
        raise ProofAdapterError("SPAR-026 write_paths are required")
    return _exact_paths(values, "SPAR-026 write_paths")


def _selection_sources(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("raw_source_cids")
    if values in (None, (), []):
        raise ProofAdapterError("raw source required")
    sources = _cids(values, "raw_source_cids", required=True)
    if not sources:
        raise ProofAdapterError("raw source required")
    return sources


def _selection_commands(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("validation_commands")
    if values in (None, ()):
        raise ProofAdapterError("validation_commands must not be empty")
    return _commands(values, "validation_commands")


def _selection_proof_ids(selection: Mapping[str, Any]) -> tuple[str, ...]:
    values = selection.get("selected_proof_ids")
    if values in (None, ()):
        return ()
    return _ordered_text(values, "selected_proof_ids", limit=MAX_OBLIGATIONS)


def _bind_predecessors(
    *,
    contracts: Sequence[Mapping[str, Any]],
    wave: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[str, str, str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not contracts:
        raise ProofAdapterError("SPAR-016 contracts are required")
    tree_id = _tree_id(wave.get("tree_id"))
    _require_tree(selection.get("tree_id"), tree_id, "SPAR-026")
    for index, contract in enumerate(contracts):
        _require_tree(contract.get("tree_id"), tree_id, f"SPAR-016[{index}]")
        _reject_non_admitting_payload(contract, f"SPAR-016[{index}]")

    status = _text(wave.get("status") or "applied", "SPAR-025 status")
    if status != "applied":
        raise ProofAdapterError("SPAR-025 wave must be applied")
    if wave.get("writes_repository") is True:
        raise ProofAdapterError("SPAR-025 wave cannot write the repository")
    if wave.get("executor_is_nomination_only") is False:
        raise ProofAdapterError("SPAR-025 executor must remain nomination_only")
    if selection.get("adapter_is_nomination_only") is False:
        raise ProofAdapterError("SPAR-026 adapter must remain nomination_only")
    if selection.get("raw_source_required") is False:
        raise ProofAdapterError("raw source required")
    if selection.get("datasets_owns_selection") is False:
        raise ProofAdapterError("datasets remains the selection authority")

    packet_cid = _selection_packet_cid(selection)
    packet_cids = _wave_packet_cids(wave)
    if packet_cid not in packet_cids:
        raise ProofAdapterError("SPAR-026 packet_cid is not in SPAR-025 packet_cids")

    write_paths = _selection_write_paths(selection)
    wave_paths = _wave_write_paths(wave)
    if write_paths != wave_paths:
        raise ProofAdapterError("SPAR-025/SPAR-026 write_paths do not match")

    return (
        tree_id,
        _wave_cid(wave),
        _selection_cid(selection),
        packet_cid,
        write_paths,
        _selection_sources(selection),
        _selection_commands(selection),
    )


def _contract_disposition(contract: Mapping[str, Any]) -> str:
    if contract.get("incomplete") is True or contract.get("incomplete_contract") is True:
        raise ProofAdapterError("incomplete SPAR-016 contract is a typed terminal")
    if contract.get("unsupported_required") is True:
        raise ProofAdapterError("unsupported required SPAR-016 contract is a typed terminal")
    if contract.get("guessed_axiom") is True:
        raise ProofAdapterError("guessed axioms are rejected")
    disposition = contract.get("disposition")
    if disposition in (None, ""):
        raise ProofAdapterError("incomplete SPAR-016 contract is a typed terminal")
    text = _text(disposition, "SPAR-016 disposition")
    if text in INCOMPLETE_CONTRACT_MARKERS:
        raise ProofAdapterError("incomplete SPAR-016 contract is a typed terminal")
    if text in UNSUPPORTED_CONTRACT_MARKERS:
        raise ProofAdapterError("unsupported required SPAR-016 contract is a typed terminal")
    if text not in ADMITTED_CONTRACT_DISPOSITIONS:
        raise ProofAdapterError(f"unsupported SPAR-016 disposition {text!r}")
    return text


def _clause_kind(value: Any) -> str:
    text = _enum(value, ClauseKind, "kind") if not isinstance(value, str) else _text(value, "kind")
    if text not in DECLARED_CLAUSE_KINDS:
        raise ProofAdapterError(f"unknown clause kind: {text}")
    return text


def _clause_polarity(value: Any) -> str:
    if value in (None, ""):
        return ClausePolarity.GUARANTEE.value
    return _enum(value, ClausePolarity, "polarity")


def _conclusiveness_for(kind: str, *, kernel_checked: bool, replayed: bool) -> str:
    if kind == ArtifactKind.RECONSTRUCTED_PROOF.value and kernel_checked:
        return Conclusiveness.CONCLUSIVE_PROOF.value
    if kind == ArtifactKind.REPLAYED_COUNTEREXAMPLE.value and replayed:
        return Conclusiveness.CONCLUSIVE_REFUTATION.value
    if kind in {ArtifactKind.COUNTERMODEL.value, ArtifactKind.PROOF_CANDIDATE.value}:
        return Conclusiveness.DIAGNOSTIC.value
    return Conclusiveness.NON_CONCLUSIVE.value


def _outcome_for(kind: str) -> str:
    mapping = {
        ArtifactKind.RECONSTRUCTED_PROOF.value: SearchOutcome.VERIFIED.value,
        ArtifactKind.PROOF_CANDIDATE.value: SearchOutcome.CANDIDATE.value,
        ArtifactKind.REPLAYED_COUNTEREXAMPLE.value: SearchOutcome.COUNTEREXAMPLE.value,
        ArtifactKind.COUNTERMODEL.value: SearchOutcome.COUNTEREXAMPLE.value,
        ArtifactKind.TIMEOUT.value: SearchOutcome.TIMEOUT.value,
        ArtifactKind.UNSUPPORTED.value: SearchOutcome.UNSUPPORTED.value,
        ArtifactKind.UNAVAILABLE.value: SearchOutcome.UNAVAILABLE.value,
        ArtifactKind.UNKNOWN.value: SearchOutcome.UNKNOWN.value,
        ArtifactKind.STALE.value: SearchOutcome.STALE.value,
        ArtifactKind.ERROR.value: SearchOutcome.ERROR.value,
    }
    return mapping[kind]


@dataclass(frozen=True, slots=True)
class PremiseRecord:
    """One content-addressed, body-free premise. Never a guessed axiom."""

    premise_id: str
    premise_cid: str
    source_authority: str
    tree_id: str
    corpus_revision: str
    environment_id: str = ""
    toolchain_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "premise_id", _text(self.premise_id, "premise_id"))
        object.__setattr__(self, "premise_cid", _cid(self.premise_cid, "premise_cid"))
        authority = _text(self.source_authority, "source_authority")
        if authority in FORBIDDEN_AXIOM_AUTHORITIES:
            raise ProofAdapterError("guessed/vector/model premises cannot become axioms")
        if authority not in AXIOM_SOURCE_AUTHORITIES:
            raise ProofAdapterError(f"unsupported premise source_authority {authority!r}")
        object.__setattr__(self, "source_authority", authority)
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self, "corpus_revision", _text(self.corpus_revision, "corpus_revision")
        )
        object.__setattr__(
            self,
            "environment_id",
            _text(self.environment_id, "environment_id", empty=True),
        )
        object.__setattr__(
            self, "toolchain_id", _text(self.toolchain_id, "toolchain_id", empty=True)
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "premise_id": self.premise_id,
            "premise_cid": self.premise_cid,
            "source_authority": self.source_authority,
            "tree_id": self.tree_id,
            "corpus_revision": self.corpus_revision,
            "environment_id": self.environment_id,
            "toolchain_id": self.toolchain_id,
        }


@dataclass(frozen=True, slots=True)
class PremiseCorpus:
    """Content-addressed premise corpus bound to current tree/toolchain roots."""

    tree_id: str
    corpus_revision: str
    premises: Sequence[PremiseRecord | Mapping[str, Any]] = ()
    environment_id: str = ""
    toolchain_id: str = ""
    selector_mode: str = PREMISE_SELECTOR_DETERMINISTIC

    interface: ClassVar[str] = PREMISE_CORPUS_INTERFACE
    schema: ClassVar[str] = PREMISE_CORPUS_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "corpus_revision",
            "premises",
            "environment_id",
            "toolchain_id",
            "selector_mode",
            "corpus_cid",
        }
    )

    def __post_init__(self) -> None:
        tree_id = _tree_id(self.tree_id)
        revision = _text(self.corpus_revision, "corpus_revision")
        environment_id = _text(self.environment_id, "environment_id", empty=True)
        toolchain_id = _text(self.toolchain_id, "toolchain_id", empty=True)
        selector = _text(self.selector_mode, "selector_mode")
        if selector != PREMISE_SELECTOR_DETERMINISTIC:
            raise ProofAdapterError("premise selection must remain deterministic")
        records: list[PremiseRecord] = []
        seen: set[str] = set()
        for item in self.premises:
            record = (
                item
                if isinstance(item, PremiseRecord)
                else PremiseRecord(**dict(item))
            )
            if record.tree_id != tree_id:
                raise ProofAdapterError("premise tree_id does not match corpus tree")
            if record.corpus_revision != revision:
                raise ProofAdapterError("stale premise corpus_revision")
            if environment_id and record.environment_id and record.environment_id != environment_id:
                raise ProofAdapterError("premise environment_id does not match corpus")
            if toolchain_id and record.toolchain_id and record.toolchain_id != toolchain_id:
                raise ProofAdapterError("premise toolchain_id does not match corpus")
            if record.premise_id in seen:
                raise ProofAdapterError("premise_id must be unique")
            seen.add(record.premise_id)
            records.append(record)
        if len(records) > MAX_PREMISES:
            raise ProofAdapterError("premises exceed maximum length")
        object.__setattr__(self, "tree_id", tree_id)
        object.__setattr__(self, "corpus_revision", revision)
        object.__setattr__(self, "premises", tuple(records))
        object.__setattr__(self, "environment_id", environment_id)
        object.__setattr__(self, "toolchain_id", toolchain_id)
        object.__setattr__(self, "selector_mode", PREMISE_SELECTOR_DETERMINISTIC)

    @property
    def premise_ids(self) -> tuple[str, ...]:
        return tuple(item.premise_id for item in self.premises)

    @property
    def premise_cids(self) -> tuple[str, ...]:
        return tuple(sorted(item.premise_cid for item in self.premises))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PREMISE_CORPUS_SCHEMA,
            "interface": PREMISE_CORPUS_INTERFACE,
            "tree_id": self.tree_id,
            "corpus_revision": self.corpus_revision,
            "premises": [item.identity_payload() for item in self.premises],
            "environment_id": self.environment_id,
            "toolchain_id": self.toolchain_id,
            "selector_mode": PREMISE_SELECTOR_DETERMINISTIC,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def corpus_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["corpus_cid"] = self.corpus_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PremiseCorpus":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("corpus_cid")
        if payload.pop("schema") != PREMISE_CORPUS_SCHEMA:
            raise ProofAdapterError("unsupported PremiseCorpus schema")
        if payload.pop("interface") != PREMISE_CORPUS_INTERFACE:
            raise ProofAdapterError("unsupported PremiseCorpus interface")
        result = cls(**payload)
        _verify_cid(claimed, result.corpus_cid, "corpus_cid")
        return result


@dataclass(frozen=True, slots=True)
class BoundaryObligation:
    """One finite lowered boundary obligation. Nomination-only."""

    obligation_id: str
    contract_cid: str
    clause_id: str
    kind: str
    polarity: str
    tree_id: str
    premise_ids: Sequence[str] = ()
    required: bool = True
    finite: bool = True
    status: ObligationStatus | str = ObligationStatus.LOWERED
    residual_reason: str = ""

    interface: ClassVar[str] = BOUNDARY_OBLIGATION_INTERFACE
    schema: ClassVar[str] = BOUNDARY_OBLIGATION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "obligation_id",
            "contract_cid",
            "clause_id",
            "kind",
            "polarity",
            "tree_id",
            "premise_ids",
            "required",
            "finite",
            "status",
            "residual_reason",
            "obligation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "contract_cid", _cid(self.contract_cid, "contract_cid"))
        object.__setattr__(self, "clause_id", _text(self.clause_id, "clause_id"))
        object.__setattr__(self, "kind", _clause_kind(self.kind))
        object.__setattr__(self, "polarity", _clause_polarity(self.polarity))
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(
            self,
            "premise_ids",
            _ordered_text(list(self.premise_ids), "premise_ids", limit=MAX_PREMISES)
            if self.premise_ids
            else (),
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "finite", _bool(self.finite, "finite"))
        object.__setattr__(
            self, "status", _enum(self.status, ObligationStatus, "status")
        )
        object.__setattr__(
            self,
            "residual_reason",
            _text(self.residual_reason, "residual_reason", empty=True),
        )
        if self.finite is False and self.status == ObligationStatus.LOWERED.value:
            raise ProofAdapterError("non-finite clauses cannot lower as obligations")
        if self.status == ObligationStatus.LOWERED.value and not self.finite:
            raise ProofAdapterError("lowered obligations must be finite")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BOUNDARY_OBLIGATION_SCHEMA,
            "interface": BOUNDARY_OBLIGATION_INTERFACE,
            "obligation_id": self.obligation_id,
            "contract_cid": self.contract_cid,
            "clause_id": self.clause_id,
            "kind": self.kind,
            "polarity": self.polarity,
            "tree_id": self.tree_id,
            "premise_ids": list(self.premise_ids),
            "required": self.required,
            "finite": self.finite,
            "status": self.status,
            "residual_reason": self.residual_reason,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundaryObligation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("obligation_cid")
        if payload.pop("schema") != BOUNDARY_OBLIGATION_SCHEMA:
            raise ProofAdapterError("unsupported BoundaryObligation schema")
        if payload.pop("interface") != BOUNDARY_OBLIGATION_INTERFACE:
            raise ProofAdapterError("unsupported BoundaryObligation interface")
        result = cls(**payload)
        _verify_cid(claimed, result.obligation_cid, "obligation_cid")
        return result


@dataclass(frozen=True, slots=True)
class ObligationDecomposition:
    """Finite obligation lowering bound to SPAR-016/025/026 predecessors."""

    tree_id: str
    corpus_cid: str
    wave_cid: str
    selection_cid: str
    packet_cid: str
    contract_cids: Sequence[str]
    obligations: Sequence[BoundaryObligation | Mapping[str, Any]]
    residuals: Sequence[Mapping[str, Any]] = ()
    write_paths: Sequence[str] = ()
    raw_source_cids: Sequence[str] = ()
    validation_commands: Sequence[str] = ()
    selected_proof_ids: Sequence[str] = ()

    interface: ClassVar[str] = OBLIGATION_DECOMPOSITION_INTERFACE
    schema: ClassVar[str] = OBLIGATION_DECOMPOSITION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "corpus_cid",
            "wave_cid",
            "selection_cid",
            "packet_cid",
            "contract_cids",
            "obligations",
            "residuals",
            "write_paths",
            "raw_source_cids",
            "validation_commands",
            "selected_proof_ids",
            "decomposition_cid",
        }
    )

    def __post_init__(self) -> None:
        obligations: list[BoundaryObligation] = []
        for item in self.obligations:
            if isinstance(item, BoundaryObligation):
                obligations.append(item)
            elif isinstance(item, Mapping):
                if "obligation_cid" in item:
                    obligations.append(BoundaryObligation.from_dict(item))
                else:
                    obligations.append(BoundaryObligation(**dict(item)))
            else:
                raise ProofAdapterError("obligations items must be objects")
        if len(obligations) > MAX_OBLIGATIONS:
            raise ProofAdapterError("obligations exceed maximum length")
        seen: set[str] = set()
        for item in obligations:
            if item.obligation_id in seen:
                raise ProofAdapterError("obligation_id must be unique")
            seen.add(item.obligation_id)
            if item.tree_id != _tree_id(self.tree_id):
                raise ProofAdapterError("obligation tree_id does not match decomposition")
        residuals = _mapping_sequence(self.residuals, "residuals")
        if len(residuals) > MAX_RESIDUALS:
            raise ProofAdapterError("residuals exceed maximum length")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "corpus_cid", _cid(self.corpus_cid, "corpus_cid"))
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(self, "selection_cid", _cid(self.selection_cid, "selection_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(
            self, "contract_cids", _cids(self.contract_cids, "contract_cids", required=True)
        )
        object.__setattr__(self, "obligations", tuple(obligations))
        object.__setattr__(self, "residuals", residuals)
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
            self,
            "selected_proof_ids",
            _ordered_text(list(self.selected_proof_ids), "selected_proof_ids", limit=MAX_OBLIGATIONS)
            if self.selected_proof_ids
            else (),
        )

    @property
    def lowered_obligation_ids(self) -> tuple[str, ...]:
        return tuple(
            item.obligation_id
            for item in self.obligations
            if item.status == ObligationStatus.LOWERED.value
        )

    @property
    def required_obligation_ids(self) -> tuple[str, ...]:
        return tuple(
            item.obligation_id
            for item in self.obligations
            if item.required and item.status == ObligationStatus.LOWERED.value
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": OBLIGATION_DECOMPOSITION_SCHEMA,
            "interface": OBLIGATION_DECOMPOSITION_INTERFACE,
            "tree_id": self.tree_id,
            "corpus_cid": self.corpus_cid,
            "wave_cid": self.wave_cid,
            "selection_cid": self.selection_cid,
            "packet_cid": self.packet_cid,
            "contract_cids": list(self.contract_cids),
            "obligations": [item.identity_payload() for item in self.obligations],
            "residuals": [dict(item) for item in self.residuals],
            "write_paths": list(self.write_paths),
            "raw_source_cids": list(self.raw_source_cids),
            "validation_commands": list(self.validation_commands),
            "selected_proof_ids": list(self.selected_proof_ids),
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def decomposition_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["obligations"] = [item.to_dict() for item in self.obligations]
        payload["decomposition_cid"] = self.decomposition_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObligationDecomposition":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("decomposition_cid")
        if payload.pop("schema") != OBLIGATION_DECOMPOSITION_SCHEMA:
            raise ProofAdapterError("unsupported ObligationDecomposition schema")
        if payload.pop("interface") != OBLIGATION_DECOMPOSITION_INTERFACE:
            raise ProofAdapterError("unsupported ObligationDecomposition interface")
        result = cls(**payload)
        _verify_cid(claimed, result.decomposition_cid, "decomposition_cid")
        return result


@dataclass(frozen=True, slots=True)
class ProofReconstruction:
    """Independently reconstructed proof. Candidates cannot inhabit this type."""

    obligation_id: str
    reconstruction_cid: str
    kernel_checked: bool
    tree_id: str
    toolchain_id: str
    environment_id: str
    corpus_revision: str

    interface: ClassVar[str] = PROOF_RECONSTRUCTION_INTERFACE
    schema: ClassVar[str] = PROOF_RECONSTRUCTION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "obligation_id",
            "reconstruction_cid",
            "kernel_checked",
            "tree_id",
            "toolchain_id",
            "environment_id",
            "corpus_revision",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(
            self, "reconstruction_cid", _cid(self.reconstruction_cid, "reconstruction_cid")
        )
        if _bool(self.kernel_checked, "kernel_checked") is not True:
            raise ProofAdapterError("reconstructed proofs require kernel_checked")
        object.__setattr__(self, "kernel_checked", True)
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "toolchain_id", _text(self.toolchain_id, "toolchain_id"))
        object.__setattr__(
            self, "environment_id", _text(self.environment_id, "environment_id")
        )
        object.__setattr__(
            self, "corpus_revision", _text(self.corpus_revision, "corpus_revision")
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": PROOF_RECONSTRUCTION_SCHEMA,
            "interface": PROOF_RECONSTRUCTION_INTERFACE,
            "obligation_id": self.obligation_id,
            "reconstruction_cid": self.reconstruction_cid,
            "kernel_checked": True,
            "tree_id": self.tree_id,
            "toolchain_id": self.toolchain_id,
            "environment_id": self.environment_id,
            "corpus_revision": self.corpus_revision,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "ProofReconstruction":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != PROOF_RECONSTRUCTION_SCHEMA:
            raise ProofAdapterError("unsupported ProofReconstruction schema")
        if payload.pop("interface") != PROOF_RECONSTRUCTION_INTERFACE:
            raise ProofAdapterError("unsupported ProofReconstruction interface")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


@dataclass(frozen=True, slots=True)
class CountermodelReplay:
    """Independently replayed countermodel. Raw diagnostics cannot inhabit this type."""

    obligation_id: str
    replay_cid: str
    replayed: bool
    tree_id: str
    toolchain_id: str
    environment_id: str
    corpus_revision: str

    interface: ClassVar[str] = COUNTERMODEL_REPLAY_INTERFACE
    schema: ClassVar[str] = COUNTERMODEL_REPLAY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "obligation_id",
            "replay_cid",
            "replayed",
            "tree_id",
            "toolchain_id",
            "environment_id",
            "corpus_revision",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "replay_cid", _cid(self.replay_cid, "replay_cid"))
        if _bool(self.replayed, "replayed") is not True:
            raise ProofAdapterError("raw countermodels cannot refute until replay")
        object.__setattr__(self, "replayed", True)
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "toolchain_id", _text(self.toolchain_id, "toolchain_id"))
        object.__setattr__(
            self, "environment_id", _text(self.environment_id, "environment_id")
        )
        object.__setattr__(
            self, "corpus_revision", _text(self.corpus_revision, "corpus_revision")
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": COUNTERMODEL_REPLAY_SCHEMA,
            "interface": COUNTERMODEL_REPLAY_INTERFACE,
            "obligation_id": self.obligation_id,
            "replay_cid": self.replay_cid,
            "replayed": True,
            "tree_id": self.tree_id,
            "toolchain_id": self.toolchain_id,
            "environment_id": self.environment_id,
            "corpus_revision": self.corpus_revision,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "CountermodelReplay":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != COUNTERMODEL_REPLAY_SCHEMA:
            raise ProofAdapterError("unsupported CountermodelReplay schema")
        if payload.pop("interface") != COUNTERMODEL_REPLAY_INTERFACE:
            raise ProofAdapterError("unsupported CountermodelReplay interface")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


@dataclass(frozen=True, slots=True)
class ProofSearchStep:
    """One bounded search step over a finite obligation."""

    obligation_id: str
    kind: str
    outcome: str
    conclusiveness: str
    artifact_cid: str = ""
    kernel_checked: bool = False
    replayed: bool = False
    reason_code: str = ""

    def __post_init__(self) -> None:
        kind = _enum(self.kind, ArtifactKind, "kind")
        object.__setattr__(self, "obligation_id", _text(self.obligation_id, "obligation_id"))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "outcome", _enum(self.outcome, SearchOutcome, "outcome"))
        object.__setattr__(
            self,
            "conclusiveness",
            _enum(self.conclusiveness, Conclusiveness, "conclusiveness"),
        )
        object.__setattr__(
            self, "artifact_cid", _optional_cid(self.artifact_cid, "artifact_cid")
        )
        object.__setattr__(self, "kernel_checked", _bool(self.kernel_checked, "kernel_checked"))
        object.__setattr__(self, "replayed", _bool(self.replayed, "replayed"))
        object.__setattr__(
            self, "reason_code", _text(self.reason_code, "reason_code", empty=True)
        )
        if kind == ArtifactKind.PROOF_CANDIDATE.value and self.conclusiveness != (
            Conclusiveness.DIAGNOSTIC.value
        ):
            raise ProofAdapterError("proof_candidate cannot admit proofs")
        if kind == ArtifactKind.RECONSTRUCTED_PROOF.value and not self.kernel_checked:
            raise ProofAdapterError("reconstructed proofs require kernel_checked")
        if kind == ArtifactKind.REPLAYED_COUNTEREXAMPLE.value and not self.replayed:
            raise ProofAdapterError("replayed counterexamples require replayed")
        if kind == ArtifactKind.COUNTERMODEL.value and self.conclusiveness == (
            Conclusiveness.CONCLUSIVE_REFUTATION.value
        ):
            raise ProofAdapterError("raw countermodels cannot refute until replay")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "kind": self.kind,
            "outcome": self.outcome,
            "conclusiveness": self.conclusiveness,
            "artifact_cid": self.artifact_cid,
            "kernel_checked": self.kernel_checked,
            "replayed": self.replayed,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class BoundedProofSearchReceipt:
    """Bounded Tactician/Hammer search receipt. Nomination-only."""

    tree_id: str
    corpus_cid: str
    decomposition_cid: str
    wave_cid: str
    selection_cid: str
    packet_cid: str
    steps: Sequence[ProofSearchStep | Mapping[str, Any]]
    reconstructions: Sequence[ProofReconstruction | Mapping[str, Any]] = ()
    replays: Sequence[CountermodelReplay | Mapping[str, Any]] = ()
    write_paths: Sequence[str] = ()
    raw_source_cids: Sequence[str] = ()
    validation_commands: Sequence[str] = ()
    status: SearchStatus | str = SearchStatus.UNKNOWN
    max_steps: int = DEFAULT_MAX_STEPS
    steps_used: int = 0
    selector_mode: str = PREMISE_SELECTOR_DETERMINISTIC
    network: str = NETWORK_DENY
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    unknown_remains_unknown: bool = True
    claims_general_equivalence: bool = False
    proof_candidate_cannot_admit_proofs: bool = True
    raw_countermodel_cannot_refute: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE
    schema: ClassVar[str] = BOUNDED_PROOF_SEARCH_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "corpus_cid",
            "decomposition_cid",
            "wave_cid",
            "selection_cid",
            "packet_cid",
            "steps",
            "reconstructions",
            "replays",
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
            "proof_candidate_cannot_admit_proofs",
            "raw_countermodel_cannot_refute",
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
        steps: list[ProofSearchStep] = []
        for item in self.steps:
            if isinstance(item, ProofSearchStep):
                steps.append(item)
            elif isinstance(item, Mapping):
                steps.append(ProofSearchStep(**dict(item)))
            else:
                raise ProofAdapterError("steps items must be objects")
        reconstructions: list[ProofReconstruction] = []
        for item in self.reconstructions:
            if isinstance(item, ProofReconstruction):
                reconstructions.append(item)
            elif isinstance(item, Mapping):
                if "receipt_cid" in item:
                    reconstructions.append(ProofReconstruction.from_dict(item))
                else:
                    reconstructions.append(ProofReconstruction(**dict(item)))
            else:
                raise ProofAdapterError("reconstructions items must be objects")
        replays: list[CountermodelReplay] = []
        for item in self.replays:
            if isinstance(item, CountermodelReplay):
                replays.append(item)
            elif isinstance(item, Mapping):
                if "receipt_cid" in item:
                    replays.append(CountermodelReplay.from_dict(item))
                else:
                    replays.append(CountermodelReplay(**dict(item)))
            else:
                raise ProofAdapterError("replays items must be objects")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "corpus_cid", _cid(self.corpus_cid, "corpus_cid"))
        object.__setattr__(
            self, "decomposition_cid", _cid(self.decomposition_cid, "decomposition_cid")
        )
        object.__setattr__(self, "wave_cid", _cid(self.wave_cid, "wave_cid"))
        object.__setattr__(self, "selection_cid", _cid(self.selection_cid, "selection_cid"))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "steps", tuple(steps))
        object.__setattr__(self, "reconstructions", tuple(reconstructions))
        object.__setattr__(self, "replays", tuple(replays))
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
        object.__setattr__(self, "status", _enum(self.status, SearchStatus, "status"))
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
        if self.selector_mode != PREMISE_SELECTOR_DETERMINISTIC:
            raise ProofAdapterError("premise selection must remain deterministic")
        object.__setattr__(self, "selector_mode", PREMISE_SELECTOR_DETERMINISTIC)
        object.__setattr__(self, "network", _network_value(self.network))
        if self.analyzer_id != ANALYZER_ID:
            raise ProofAdapterError("receipt analyzer_id must remain SPAR-030")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if self.adapter_is_nomination_only is not True:
            raise ProofAdapterError("adapter must remain nomination_only")
        if self.unknown_remains_unknown is not True:
            raise ProofAdapterError("unknown must remain unknown")
        if self.claims_general_equivalence is not False:
            raise ProofAdapterError("general Python equivalence is not claimed")
        if self.proof_candidate_cannot_admit_proofs is not True:
            raise ProofAdapterError("proof_candidate cannot admit proofs")
        if self.raw_countermodel_cannot_refute is not True:
            raise ProofAdapterError("raw countermodels cannot refute until replay")
        if self.mutated is not False:
            raise ProofAdapterError("adapter cannot mutate")
        if self.deterministic is not True:
            raise ProofAdapterError("search must remain deterministic")
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "unknown_remains_unknown", True)
        object.__setattr__(self, "claims_general_equivalence", False)
        object.__setattr__(self, "proof_candidate_cannot_admit_proofs", True)
        object.__setattr__(self, "raw_countermodel_cannot_refute", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)

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
        return self.status != SearchStatus.VERIFIED.value

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": BOUNDED_PROOF_SEARCH_RECEIPT_SCHEMA,
            "interface": BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "corpus_cid": self.corpus_cid,
            "decomposition_cid": self.decomposition_cid,
            "wave_cid": self.wave_cid,
            "selection_cid": self.selection_cid,
            "packet_cid": self.packet_cid,
            "steps": [item.identity_payload() for item in self.steps],
            "reconstructions": [item.identity_payload() for item in self.reconstructions],
            "replays": [item.identity_payload() for item in self.replays],
            "write_paths": list(self.write_paths),
            "raw_source_cids": list(self.raw_source_cids),
            "validation_commands": list(self.validation_commands),
            "status": self.status,
            "max_steps": self.max_steps,
            "steps_used": self.steps_used,
            "selector_mode": PREMISE_SELECTOR_DETERMINISTIC,
            "network": NETWORK_DENY,
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "unknown_remains_unknown": True,
            "claims_general_equivalence": False,
            "proof_candidate_cannot_admit_proofs": True,
            "raw_countermodel_cannot_refute": True,
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
        payload["reconstructions"] = [item.to_dict() for item in self.reconstructions]
        payload["replays"] = [item.to_dict() for item in self.replays]
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundedProofSearchReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != BOUNDED_PROOF_SEARCH_RECEIPT_SCHEMA:
            raise ProofAdapterError("unsupported BoundedProofSearchReceipt schema")
        if payload.pop("interface") != BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE:
            raise ProofAdapterError("unsupported BoundedProofSearchReceipt interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ProofAdapterError("adapter must remain nomination_only")
        if payload.pop("unknown_remains_unknown") is not True:
            raise ProofAdapterError("unknown must remain unknown")
        if payload.pop("claims_general_equivalence") is not False:
            raise ProofAdapterError("general Python equivalence is not claimed")
        if payload.pop("proof_candidate_cannot_admit_proofs") is not True:
            raise ProofAdapterError("proof_candidate cannot admit proofs")
        if payload.pop("raw_countermodel_cannot_refute") is not True:
            raise ProofAdapterError("raw countermodels cannot refute until replay")
        if payload.pop("mutated") is not False:
            raise ProofAdapterError("adapter cannot mutate")
        if payload.pop("deterministic") is not True:
            raise ProofAdapterError("search must remain deterministic")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise ProofAdapterError("receipt analyzer_id must remain SPAR-030")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


def compile_premise_corpus(
    *,
    tree_id: str,
    corpus_revision: str,
    premises: Sequence[Mapping[str, Any] | PremiseRecord],
    environment_id: str = "",
    toolchain_id: str = "",
    selector_mode: str = PREMISE_SELECTOR_DETERMINISTIC,
    vector_evidence: Any = None,
) -> PremiseCorpus:
    """Build one content-addressed, body-free premise corpus."""

    _reject_vector_admission(vector_evidence)
    records: list[PremiseRecord] = []
    for item in premises:
        if isinstance(item, PremiseRecord):
            records.append(item)
            continue
        payload = _as_mapping(item, "premise")
        _reject_non_admitting_payload(payload, "premise")
        records.append(
            PremiseRecord(
                premise_id=payload.get("premise_id"),
                premise_cid=payload.get("premise_cid"),
                source_authority=payload.get("source_authority"),
                tree_id=payload.get("tree_id", tree_id),
                corpus_revision=payload.get("corpus_revision", corpus_revision),
                environment_id=payload.get("environment_id", environment_id),
                toolchain_id=payload.get("toolchain_id", toolchain_id),
            )
        )
    return PremiseCorpus(
        tree_id=tree_id,
        corpus_revision=corpus_revision,
        premises=records,
        environment_id=environment_id,
        toolchain_id=toolchain_id,
        selector_mode=selector_mode,
    )


def _lower_clause(
    *,
    contract: Mapping[str, Any],
    clause: Mapping[str, Any],
    tree_id: str,
    premise_ids: Sequence[str],
) -> tuple[BoundaryObligation | None, dict[str, Any] | None]:
    contract_cid = contract.get("contract_cid")
    if contract_cid in (None, ""):
        raise ProofAdapterError("SPAR-016 contract_cid is required")
    clause_id = clause.get("clause_id") or clause.get("id")
    if clause_id in (None, ""):
        raise ProofAdapterError("SPAR-016 clause_id is required")
    kind = _clause_kind(clause.get("kind"))
    polarity = _clause_polarity(clause.get("polarity"))
    finite = clause.get("finite")
    if finite is None:
        finite = True
    finite = _bool(finite, "finite")
    required = clause.get("required")
    if required is None:
        required = True
    required = _bool(required, "required")
    semantics = clause.get("semantics") or contract.get("semantics") or ""
    if semantics:
        semantics = _text(semantics, "semantics")
    obligation_id = clause.get("obligation_id") or f"{_cid(contract_cid, 'contract_cid')}:{clause_id}"
    residual_reason = ""
    status = ObligationStatus.LOWERED.value
    if not finite or kind not in FINITE_CLAUSE_KINDS:
        status = ObligationStatus.RESIDUAL.value
        residual_reason = "non_finite_clause"
    if semantics in UNSUPPORTED_SEMANTICS:
        status = ObligationStatus.UNSUPPORTED.value
        residual_reason = f"unsupported_semantics:{semantics}"
        if required:
            raise ProofAdapterError(
                "unsupported required SPAR-016 contract is a typed terminal"
            )
    obligation = BoundaryObligation(
        obligation_id=_text(obligation_id, "obligation_id"),
        contract_cid=_cid(contract_cid, "contract_cid"),
        clause_id=_text(clause_id, "clause_id"),
        kind=kind,
        polarity=polarity,
        tree_id=tree_id,
        premise_ids=premise_ids,
        required=required,
        finite=finite and status == ObligationStatus.LOWERED.value,
        status=status,
        residual_reason=residual_reason,
    )
    residual = None
    if status != ObligationStatus.LOWERED.value:
        residual = {
            "clause_id": obligation.clause_id,
            "obligation_id": obligation.obligation_id,
            "reason": residual_reason,
            "required": required,
        }
        if required and status != ObligationStatus.RESIDUAL.value:
            raise ProofAdapterError(
                "unsupported required SPAR-016 contract is a typed terminal"
            )
        if required and not finite:
            raise ProofAdapterError(
                "unsupported required SPAR-016 contract is a typed terminal"
            )
    return obligation, residual


def lower_boundary_obligations(
    *,
    contracts: Sequence[Mapping[str, Any] | Any],
    corpus: PremiseCorpus | Mapping[str, Any],
    wave: Mapping[str, Any] | Any,
    selection: Mapping[str, Any] | Any,
    vector_evidence: Any = None,
) -> ObligationDecomposition:
    """Lower finite SPAR-016 clauses into exact boundary obligations."""

    _reject_vector_admission(vector_evidence)
    wave_map = _as_mapping(wave, "SPAR-025 wave")
    selection_map = _as_mapping(selection, "SPAR-026 selection")
    contract_maps = tuple(
        _as_mapping(item, "SPAR-016 contract") for item in contracts
    )
    (
        tree_id,
        wave_cid,
        selection_cid,
        packet_cid,
        write_paths,
        raw_source_cids,
        validation_commands,
    ) = _bind_predecessors(
        contracts=contract_maps,
        wave=wave_map,
        selection=selection_map,
    )
    typed_corpus = (
        corpus if isinstance(corpus, PremiseCorpus) else PremiseCorpus.from_dict(corpus)
    )
    if typed_corpus.tree_id != tree_id:
        raise ProofAdapterError("premise corpus tree_id does not match SPAR-025")

    selected = _selection_proof_ids(selection_map)
    obligations: list[BoundaryObligation] = []
    residuals: list[dict[str, Any]] = []
    contract_cids: list[str] = []
    for contract in contract_maps:
        _contract_disposition(contract)
        contract_cid = _cid(contract.get("contract_cid"), "SPAR-016 contract_cid")
        contract_cids.append(contract_cid)
        clauses = _mapping_sequence(contract.get("clauses"), "SPAR-016 clauses")
        if not clauses:
            raise ProofAdapterError("incomplete SPAR-016 contract is a typed terminal")
        if len(clauses) > MAX_CLAUSES:
            raise ProofAdapterError("SPAR-016 clauses exceed maximum length")
        for clause in clauses:
            obligation, residual = _lower_clause(
                contract=contract,
                clause=clause,
                tree_id=tree_id,
                premise_ids=typed_corpus.premise_ids,
            )
            if obligation is not None:
                obligations.append(obligation)
            if residual is not None:
                residuals.append(residual)

    if selected:
        selected_set = set(selected)
        known = {item.obligation_id for item in obligations}
        missing = selected_set - known
        if missing:
            raise ProofAdapterError(
                f"SPAR-026 selected_proof_ids missing from lowered obligations: {sorted(missing)}"
            )

    return ObligationDecomposition(
        tree_id=tree_id,
        corpus_cid=typed_corpus.corpus_cid,
        wave_cid=wave_cid,
        selection_cid=selection_cid,
        packet_cid=packet_cid,
        contract_cids=contract_cids,
        obligations=obligations,
        residuals=residuals,
        write_paths=write_paths,
        raw_source_cids=raw_source_cids,
        validation_commands=validation_commands,
        selected_proof_ids=selected,
    )


def reconstruct_proof(
    artifact: Mapping[str, Any] | ProofReconstruction,
    *,
    tree_id: str,
    corpus_revision: str,
    environment_id: str,
    toolchain_id: str,
) -> ProofReconstruction:
    """Admit a reconstructed proof only when kernel-checked and root-bound."""

    if isinstance(artifact, ProofReconstruction):
        reconstruction = artifact
    else:
        payload = _as_mapping(artifact, "proof reconstruction")
        kind = payload.get("kind")
        if kind not in (None, "", ArtifactKind.RECONSTRUCTED_PROOF.value):
            raise ProofAdapterError("proof_candidate cannot admit proofs")
        if payload.get("kernel_checked") is not True:
            raise ProofAdapterError("reconstructed proofs require kernel_checked")
        reconstruction = ProofReconstruction(
            obligation_id=payload.get("obligation_id"),
            reconstruction_cid=payload.get("reconstruction_cid")
            or payload.get("artifact_cid"),
            kernel_checked=True,
            tree_id=payload.get("tree_id", tree_id),
            toolchain_id=payload.get("toolchain_id", toolchain_id),
            environment_id=payload.get("environment_id", environment_id),
            corpus_revision=payload.get("corpus_revision", corpus_revision),
        )
    if reconstruction.tree_id != tree_id:
        raise ProofAdapterError("stale reconstruction tree_id")
    if reconstruction.corpus_revision != corpus_revision:
        raise ProofAdapterError("stale reconstruction corpus_revision")
    if reconstruction.environment_id != environment_id:
        raise ProofAdapterError("stale reconstruction environment_id")
    if reconstruction.toolchain_id != toolchain_id:
        raise ProofAdapterError("stale reconstruction toolchain_id")
    return reconstruction


def replay_countermodel(
    artifact: Mapping[str, Any] | CountermodelReplay,
    *,
    tree_id: str,
    corpus_revision: str,
    environment_id: str,
    toolchain_id: str,
    replay: ReplayFn | None = None,
) -> CountermodelReplay:
    """Replay a supported countermodel; raw diagnostics remain non-authoritative."""

    if isinstance(artifact, CountermodelReplay):
        replayed = artifact
    else:
        payload = _as_mapping(artifact, "countermodel")
        kind = payload.get("kind")
        if kind == ArtifactKind.COUNTERMODEL.value and payload.get("replayed") is not True:
            if replay is None:
                raise ProofAdapterError("raw countermodels cannot refute until replay")
            payload = dict(replay(**payload))
            _reject_forbidden_bodies(payload, "countermodel replay")
        if payload.get("replayed") is not True and payload.get("kind") != (
            ArtifactKind.REPLAYED_COUNTEREXAMPLE.value
        ):
            raise ProofAdapterError("raw countermodels cannot refute until replay")
        replayed = CountermodelReplay(
            obligation_id=payload.get("obligation_id"),
            replay_cid=payload.get("replay_cid") or payload.get("artifact_cid"),
            replayed=True,
            tree_id=payload.get("tree_id", tree_id),
            toolchain_id=payload.get("toolchain_id", toolchain_id),
            environment_id=payload.get("environment_id", environment_id),
            corpus_revision=payload.get("corpus_revision", corpus_revision),
        )
    if replayed.tree_id != tree_id:
        raise ProofAdapterError("stale countermodel tree_id")
    if replayed.corpus_revision != corpus_revision:
        raise ProofAdapterError("stale countermodel corpus_revision")
    if replayed.environment_id != environment_id:
        raise ProofAdapterError("stale countermodel environment_id")
    if replayed.toolchain_id != toolchain_id:
        raise ProofAdapterError("stale countermodel toolchain_id")
    return replayed


def _artifact_for_obligation(
    *,
    obligation: BoundaryObligation,
    artifacts: Mapping[str, Mapping[str, Any]],
    hammer: HammerSearchFn | None,
    tactician: TacticianSearchFn | None,
    corpus: PremiseCorpus,
    max_steps: int,
) -> dict[str, Any]:
    provided = artifacts.get(obligation.obligation_id)
    if provided is not None:
        payload = dict(provided)
        _reject_forbidden_bodies(payload, "proof artifact")
        return payload
    request = {
        "obligation_id": obligation.obligation_id,
        "obligation_cid": obligation.obligation_cid,
        "corpus_cid": corpus.corpus_cid,
        "tree_id": corpus.tree_id,
        "corpus_revision": corpus.corpus_revision,
        "environment_id": corpus.environment_id,
        "toolchain_id": corpus.toolchain_id,
        "max_steps": max_steps,
        "network": NETWORK_DENY,
        "selector_mode": PREMISE_SELECTOR_DETERMINISTIC,
    }
    if hammer is not None:
        payload = dict(hammer(**request))
        _reject_forbidden_bodies(payload, "hammer artifact")
        return payload
    if tactician is not None:
        payload = dict(tactician(**request))
        _reject_forbidden_bodies(payload, "tactician artifact")
        return payload
    return {
        "obligation_id": obligation.obligation_id,
        "kind": ArtifactKind.UNKNOWN.value,
        "artifact_cid": "",
        "kernel_checked": False,
        "replayed": False,
        "reason_code": "unknown",
    }


def _step_from_artifact(
    *,
    obligation: BoundaryObligation,
    artifact: Mapping[str, Any],
    corpus: PremiseCorpus,
    reconstruct: ReconstructFn | None,
    replay: ReplayFn | None,
) -> tuple[ProofSearchStep, ProofReconstruction | None, CountermodelReplay | None]:
    kind_value = artifact.get("kind") or ArtifactKind.UNKNOWN.value
    kind = _enum(kind_value, ArtifactKind, "kind")
    tree_id = artifact.get("tree_id") or corpus.tree_id
    if _tree_id(tree_id) != corpus.tree_id:
        kind = ArtifactKind.STALE.value
    corpus_revision = artifact.get("corpus_revision") or corpus.corpus_revision
    if corpus.corpus_revision and corpus_revision != corpus.corpus_revision:
        kind = ArtifactKind.STALE.value
    environment_id = artifact.get("environment_id") or corpus.environment_id
    if corpus.environment_id and environment_id and environment_id != corpus.environment_id:
        kind = ArtifactKind.STALE.value
    toolchain_id = artifact.get("toolchain_id") or corpus.toolchain_id
    if corpus.toolchain_id and toolchain_id and toolchain_id != corpus.toolchain_id:
        kind = ArtifactKind.STALE.value

    reconstruction: ProofReconstruction | None = None
    replayed: CountermodelReplay | None = None
    kernel_checked = artifact.get("kernel_checked") is True
    was_replayed = artifact.get("replayed") is True or kind == (
        ArtifactKind.REPLAYED_COUNTEREXAMPLE.value
    )
    reason = _text(artifact.get("reason_code") or "", "reason_code", empty=True)
    artifact_cid = _optional_cid(artifact.get("artifact_cid"), "artifact_cid")

    if kind == ArtifactKind.PROOF_CANDIDATE.value:
        step = ProofSearchStep(
            obligation_id=obligation.obligation_id,
            kind=kind,
            outcome=_outcome_for(kind),
            conclusiveness=Conclusiveness.DIAGNOSTIC.value,
            artifact_cid=artifact_cid,
            kernel_checked=False,
            replayed=False,
            reason_code=reason or "proof_candidate",
        )
        return step, None, None

    if kind == ArtifactKind.RECONSTRUCTED_PROOF.value:
        source = dict(artifact)
        if reconstruct is not None and source.get("kernel_checked") is not True:
            source = dict(reconstruct(**source))
        reconstruction = reconstruct_proof(
            source,
            tree_id=corpus.tree_id,
            corpus_revision=corpus.corpus_revision,
            environment_id=environment_id or corpus.environment_id,
            toolchain_id=toolchain_id or corpus.toolchain_id,
        )
        step = ProofSearchStep(
            obligation_id=obligation.obligation_id,
            kind=kind,
            outcome=SearchOutcome.VERIFIED.value,
            conclusiveness=Conclusiveness.CONCLUSIVE_PROOF.value,
            artifact_cid=reconstruction.reconstruction_cid,
            kernel_checked=True,
            replayed=False,
            reason_code=reason,
        )
        return step, reconstruction, None

    if kind == ArtifactKind.REPLAYED_COUNTEREXAMPLE.value or (
        kind == ArtifactKind.COUNTERMODEL.value and was_replayed
    ):
        replayed = replay_countermodel(
            artifact,
            tree_id=corpus.tree_id,
            corpus_revision=corpus.corpus_revision,
            environment_id=environment_id or corpus.environment_id,
            toolchain_id=toolchain_id or corpus.toolchain_id,
            replay=replay,
        )
        step = ProofSearchStep(
            obligation_id=obligation.obligation_id,
            kind=ArtifactKind.REPLAYED_COUNTEREXAMPLE.value,
            outcome=SearchOutcome.COUNTEREXAMPLE.value,
            conclusiveness=Conclusiveness.CONCLUSIVE_REFUTATION.value,
            artifact_cid=replayed.replay_cid,
            kernel_checked=False,
            replayed=True,
            reason_code=reason,
        )
        return step, None, replayed

    if kind == ArtifactKind.COUNTERMODEL.value:
        step = ProofSearchStep(
            obligation_id=obligation.obligation_id,
            kind=kind,
            outcome=SearchOutcome.COUNTEREXAMPLE.value,
            conclusiveness=Conclusiveness.DIAGNOSTIC.value,
            artifact_cid=artifact_cid,
            kernel_checked=False,
            replayed=False,
            reason_code=reason or "raw_countermodel",
        )
        return step, None, None

    step = ProofSearchStep(
        obligation_id=obligation.obligation_id,
        kind=kind,
        outcome=_outcome_for(kind),
        conclusiveness=_conclusiveness_for(kind, kernel_checked=kernel_checked, replayed=False),
        artifact_cid=artifact_cid,
        kernel_checked=False,
        replayed=False,
        reason_code=reason or kind,
    )
    return step, None, None


def _status_from_steps(
    *,
    steps: Sequence[ProofSearchStep],
    required_ids: Sequence[str],
    steps_used: int,
    max_steps: int,
) -> str:
    by_id = {item.obligation_id: item for item in steps}
    if any(item.kind == ArtifactKind.REPLAYED_COUNTEREXAMPLE.value for item in steps):
        return SearchStatus.REFUTED.value
    if any(item.kind == ArtifactKind.STALE.value for item in steps):
        return SearchStatus.REJECTED.value
    if any(item.kind == ArtifactKind.ERROR.value for item in steps):
        return SearchStatus.REJECTED.value
    if steps_used > max_steps or any(
        item.kind == ArtifactKind.TIMEOUT.value for item in steps
    ):
        return SearchStatus.TIMEOUT.value
    required_steps = [by_id[item] for item in required_ids if item in by_id]
    missing = [item for item in required_ids if item not in by_id]
    if missing:
        return SearchStatus.INCOMPLETE.value
    if any(item.kind == ArtifactKind.UNSUPPORTED.value for item in required_steps):
        return SearchStatus.BLOCKED.value
    if any(item.kind == ArtifactKind.UNAVAILABLE.value for item in required_steps):
        return SearchStatus.BLOCKED.value
    if any(item.kind == ArtifactKind.UNKNOWN.value for item in required_steps):
        return SearchStatus.UNKNOWN.value
    if any(item.kind == ArtifactKind.PROOF_CANDIDATE.value for item in required_steps):
        return SearchStatus.INCOMPLETE.value
    if any(item.kind == ArtifactKind.COUNTERMODEL.value for item in required_steps):
        return SearchStatus.INCOMPLETE.value
    if required_steps and all(
        item.kind == ArtifactKind.RECONSTRUCTED_PROOF.value and item.kernel_checked
        for item in required_steps
    ):
        return SearchStatus.VERIFIED.value
    if not required_ids:
        return SearchStatus.VERIFIED.value
    return SearchStatus.INCOMPLETE.value


def run_bounded_proof_search(
    *,
    decomposition: ObligationDecomposition | Mapping[str, Any],
    corpus: PremiseCorpus | Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
    hammer: HammerSearchFn | None = None,
    tactician: TacticianSearchFn | None = None,
    reconstruct: ReconstructFn | None = None,
    replay: ReplayFn | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    mutate: bool = False,
    vector_evidence: Any = None,
    network: str = NETWORK_DENY,
) -> BoundedProofSearchReceipt:
    """Run bounded proof/countermodel search over lowered finite obligations."""

    if mutate is not False:
        raise ProofAdapterError("adapter cannot mutate")
    _reject_vector_admission(vector_evidence)
    _network_value(network)
    typed = (
        decomposition
        if isinstance(decomposition, ObligationDecomposition)
        else ObligationDecomposition.from_dict(decomposition)
    )
    typed_corpus = (
        corpus if isinstance(corpus, PremiseCorpus) else PremiseCorpus.from_dict(corpus)
    )
    if typed.corpus_cid != typed_corpus.corpus_cid:
        raise ProofAdapterError("decomposition corpus_cid does not match premise corpus")
    if typed.tree_id != typed_corpus.tree_id:
        raise ProofAdapterError("decomposition tree_id does not match premise corpus")
    bound = _int(max_steps, "max_steps", minimum=1, maximum=ABSOLUTE_MAX_STEPS)

    artifact_map: dict[str, Mapping[str, Any]] = {}
    if isinstance(artifacts, Mapping):
        for key, value in artifacts.items():
            artifact_map[_text(key, "artifact obligation_id")] = _as_mapping(
                value, "proof artifact"
            )
    elif artifacts:
        for item in _mapping_sequence(artifacts, "proof artifacts"):
            obligation_id = _text(item.get("obligation_id"), "artifact obligation_id")
            artifact_map[obligation_id] = item

    selected = set(typed.selected_proof_ids) if typed.selected_proof_ids else None
    targets = [
        item
        for item in typed.obligations
        if item.status == ObligationStatus.LOWERED.value
        and (selected is None or item.obligation_id in selected)
    ]
    if len(targets) > DEFAULT_MAX_OBLIGATIONS:
        raise ProofAdapterError("obligations exceed search bound")

    steps: list[ProofSearchStep] = []
    reconstructions: list[ProofReconstruction] = []
    replays: list[CountermodelReplay] = []
    steps_used = 0
    for obligation in targets:
        steps_used += 1
        if steps_used > bound:
            steps.append(
                ProofSearchStep(
                    obligation_id=obligation.obligation_id,
                    kind=ArtifactKind.TIMEOUT.value,
                    outcome=SearchOutcome.TIMEOUT.value,
                    conclusiveness=Conclusiveness.NON_CONCLUSIVE.value,
                    reason_code="max_steps",
                )
            )
            continue
        artifact = _artifact_for_obligation(
            obligation=obligation,
            artifacts=artifact_map,
            hammer=hammer,
            tactician=tactician,
            corpus=typed_corpus,
            max_steps=bound,
        )
        step, reconstruction, replayed = _step_from_artifact(
            obligation=obligation,
            artifact=artifact,
            corpus=typed_corpus,
            reconstruct=reconstruct,
            replay=replay,
        )
        steps.append(step)
        if reconstruction is not None:
            reconstructions.append(reconstruction)
        if replayed is not None:
            replays.append(replayed)

    required_ids = typed.required_obligation_ids
    if selected is not None:
        required_ids = tuple(item for item in required_ids if item in selected)
    status = _status_from_steps(
        steps=steps,
        required_ids=required_ids,
        steps_used=steps_used,
        max_steps=bound,
    )
    return BoundedProofSearchReceipt(
        tree_id=typed.tree_id,
        corpus_cid=typed_corpus.corpus_cid,
        decomposition_cid=typed.decomposition_cid,
        wave_cid=typed.wave_cid,
        selection_cid=typed.selection_cid,
        packet_cid=typed.packet_cid,
        steps=steps,
        reconstructions=reconstructions,
        replays=replays,
        write_paths=typed.write_paths,
        raw_source_cids=typed.raw_source_cids,
        validation_commands=typed.validation_commands,
        status=status,
        max_steps=bound,
        steps_used=min(steps_used, bound) if steps_used <= bound else steps_used,
    )


def dry_run_bounded_proof_search(
    *,
    decomposition: ObligationDecomposition | Mapping[str, Any],
    corpus: PremiseCorpus | Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
    hammer: HammerSearchFn | None = None,
    tactician: TacticianSearchFn | None = None,
    reconstruct: ReconstructFn | None = None,
    replay: ReplayFn | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    vector_evidence: Any = None,
) -> BoundedProofSearchReceipt:
    """Deterministic non-mutating bounded proof/countermodel search."""

    result = run_bounded_proof_search(
        decomposition=decomposition,
        corpus=corpus,
        artifacts=artifacts,
        hammer=hammer,
        tactician=tactician,
        reconstruct=reconstruct,
        replay=replay,
        max_steps=max_steps,
        mutate=False,
        vector_evidence=vector_evidence,
    )
    if result.mutated is not False or result.deterministic is not True:
        raise ProofAdapterError("dry-run must remain deterministic and non-mutating")
    return result


class TacticianHammerAdapter:
    """SPAR-030 Tactician/Hammer remodularization adapter. Nomination-only."""

    interface: ClassVar[str] = TACTICIAN_HAMMER_ADAPTER_INTERFACE
    schema: ClassVar[str] = TACTICIAN_HAMMER_ADAPTER_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID
    receipt_interface: ClassVar[str] = BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE

    def compile_corpus(
        self,
        *,
        tree_id: str,
        corpus_revision: str,
        premises: Sequence[Mapping[str, Any] | PremiseRecord],
        environment_id: str = "",
        toolchain_id: str = "",
        vector_evidence: Any = None,
    ) -> PremiseCorpus:
        return compile_premise_corpus(
            tree_id=tree_id,
            corpus_revision=corpus_revision,
            premises=premises,
            environment_id=environment_id,
            toolchain_id=toolchain_id,
            vector_evidence=vector_evidence,
        )

    def lower(
        self,
        *,
        contracts: Sequence[Mapping[str, Any] | Any],
        corpus: PremiseCorpus | Mapping[str, Any],
        wave: Mapping[str, Any] | Any,
        selection: Mapping[str, Any] | Any,
        vector_evidence: Any = None,
    ) -> ObligationDecomposition:
        return lower_boundary_obligations(
            contracts=contracts,
            corpus=corpus,
            wave=wave,
            selection=selection,
            vector_evidence=vector_evidence,
        )

    def search(
        self,
        *,
        decomposition: ObligationDecomposition | Mapping[str, Any],
        corpus: PremiseCorpus | Mapping[str, Any],
        artifacts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
        hammer: HammerSearchFn | None = None,
        tactician: TacticianSearchFn | None = None,
        reconstruct: ReconstructFn | None = None,
        replay: ReplayFn | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        vector_evidence: Any = None,
    ) -> BoundedProofSearchReceipt:
        return run_bounded_proof_search(
            decomposition=decomposition,
            corpus=corpus,
            artifacts=artifacts,
            hammer=hammer,
            tactician=tactician,
            reconstruct=reconstruct,
            replay=replay,
            max_steps=max_steps,
            vector_evidence=vector_evidence,
        )

    def dry_run(
        self,
        *,
        decomposition: ObligationDecomposition | Mapping[str, Any],
        corpus: PremiseCorpus | Mapping[str, Any],
        artifacts: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
        hammer: HammerSearchFn | None = None,
        tactician: TacticianSearchFn | None = None,
        reconstruct: ReconstructFn | None = None,
        replay: ReplayFn | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        vector_evidence: Any = None,
    ) -> BoundedProofSearchReceipt:
        return dry_run_bounded_proof_search(
            decomposition=decomposition,
            corpus=corpus,
            artifacts=artifacts,
            hammer=hammer,
            tactician=tactician,
            reconstruct=reconstruct,
            replay=replay,
            max_steps=max_steps,
            vector_evidence=vector_evidence,
        )

    def reconstruct(
        self,
        artifact: Mapping[str, Any] | ProofReconstruction,
        *,
        tree_id: str,
        corpus_revision: str,
        environment_id: str,
        toolchain_id: str,
    ) -> ProofReconstruction:
        return reconstruct_proof(
            artifact,
            tree_id=tree_id,
            corpus_revision=corpus_revision,
            environment_id=environment_id,
            toolchain_id=toolchain_id,
        )

    def replay(
        self,
        artifact: Mapping[str, Any] | CountermodelReplay,
        *,
        tree_id: str,
        corpus_revision: str,
        environment_id: str,
        toolchain_id: str,
        replay: ReplayFn | None = None,
    ) -> CountermodelReplay:
        return replay_countermodel(
            artifact,
            tree_id=tree_id,
            corpus_revision=corpus_revision,
            environment_id=environment_id,
            toolchain_id=toolchain_id,
            replay=replay,
        )


def encode_canonical_corpus(corpus: PremiseCorpus) -> dict[str, Any]:
    return corpus.to_dict()


def decode_canonical_corpus(payload: Mapping[str, Any]) -> PremiseCorpus:
    return PremiseCorpus.from_dict(payload)


def encode_canonical_decomposition(
    decomposition: ObligationDecomposition,
) -> dict[str, Any]:
    return decomposition.to_dict()


def decode_canonical_decomposition(payload: Mapping[str, Any]) -> ObligationDecomposition:
    return ObligationDecomposition.from_dict(payload)


def encode_canonical_receipt(receipt: BoundedProofSearchReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> BoundedProofSearchReceipt:
    return BoundedProofSearchReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise ProofAdapterError(
            f"proof adapter must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_IS_NOMINATION_ONLY",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "BOUNDED_PROOF_SEARCH_RECEIPT_INTERFACE",
    "BOUNDARY_OBLIGATION_INTERFACE",
    "COUNTERMODEL_REPLAY_INTERFACE",
    "DECLARED_ARTIFACT_KINDS",
    "DECLARED_CLAUSE_KINDS",
    "DECLARED_CLAUSE_POLARITIES",
    "DECLARED_CONCLUSIVENESS",
    "DECLARED_OBLIGATION_STATUSES",
    "DECLARED_SEARCH_OUTCOMES",
    "DECLARED_SEARCH_STATUSES",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "FORBIDDEN_PROOF_NAMES",
    "GENERAL_PYTHON_EQUIVALENCE_CLAIMED",
    "GOAL_ID",
    "GUESSED_AXIOMS_REJECTED",
    "HAMMER_OWNS_PRODUCTION_PROOF",
    "IDENTITY_EXCLUDED_FIELDS",
    "IMPLICIT_INSTALL_FORBIDDEN",
    "IMPLICIT_NETWORK_FORBIDDEN",
    "INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "OBLIGATION_DECOMPOSITION_INTERFACE",
    "PREMISE_CORPUS_INTERFACE",
    "PREMISE_SELECTOR_DETERMINISTIC",
    "PROGRAM",
    "PROOF_ADAPTER_CAN_AUTHORIZE_COMPLETION",
    "PROOF_ADAPTER_CAN_AUTHORIZE_TRANSITION",
    "PROOF_ADAPTER_CAN_CREATE_AUTHORITY",
    "PROOF_ADAPTER_CAN_CREATE_PROOF_AUTHORITY",
    "PROOF_ADAPTER_CONTRACT_VERSION",
    "PROOF_CANDIDATE_CANNOT_ADMIT_PROOFS",
    "PROOF_RECONSTRUCTION_INTERFACE",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_COUNTERMODEL_CANNOT_REFUTE",
    "RAW_SOURCE_REQUIRED",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TACTICIAN_HAMMER_ADAPTER_INTERFACE",
    "TACTICIAN_OWNS_SEARCH",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TEST_PASS_IS_NOT_PROOF",
    "UNKNOWN_REMAINS_UNKNOWN",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "ArtifactKind",
    "BoundaryObligation",
    "BoundedProofSearchReceipt",
    "ClauseKind",
    "ClausePolarity",
    "Conclusiveness",
    "CountermodelReplay",
    "ObligationDecomposition",
    "ObligationStatus",
    "PremiseCorpus",
    "PremiseRecord",
    "ProofAdapterError",
    "ProofReconstruction",
    "ProofSearchStep",
    "SearchOutcome",
    "SearchStatus",
    "TacticianHammerAdapter",
    "assert_not_competing_capsule_family",
    "compile_premise_corpus",
    "decode_canonical_corpus",
    "decode_canonical_decomposition",
    "decode_canonical_receipt",
    "dry_run_bounded_proof_search",
    "encode_canonical_corpus",
    "encode_canonical_decomposition",
    "encode_canonical_receipt",
    "lower_boundary_obligations",
    "proof_adapter_cid_profile",
    "proof_adapter_descriptor",
    "provider_free_exports",
    "reconstruct_proof",
    "replay_countermodel",
    "run_bounded_proof_search",
]
