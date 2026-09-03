"""SPAR-033 exact refactor-state, transition, proof, and procedure reuse.

This module extends current supervisor memory with ``RefactorTransition@1``
and an exact reuse decision.  It builds freshness-bound reuse keys over
trees, state, partitions, policy, environment, toolchain, obligations,
validation, and procedure version, then retains accepted and rejected
episodes.

Exact key identity is the only reuse path.  Similarity yields context
only.  Stale or mismatched evidence revokes reuse.  Negative episodes
remain first-class and block reuse of the same key.  Vector, model, and
heuristic evidence cannot admit reuse or suppress raw-source fallback.

The adapter is nomination-only.  It cannot authorize a transition,
completion, merge, or competing authority.  Observational metadata is
excluded from identity.  Dry-run is deterministic and never mutates.
Network is denied.  Proofs do not transfer across stale toolchain,
profile, or tree bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)


TASK_ID: Final[str] = "SPAR-033"
GOAL_ID: Final[str] = "SPAR-G061"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "exact reuse"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.refactor_memory@1"
)

REFACTOR_REUSE_KEY_INTERFACE: Final[str] = "RefactorReuseKey@1"
REFACTOR_TRANSITION_INTERFACE: Final[str] = "RefactorTransition@1"
REFACTOR_REUSE_DECISION_INTERFACE: Final[str] = "RefactorReuseDecision@1"
REFACTOR_MEMORY_STORE_INTERFACE: Final[str] = "RefactorMemoryStore@1"
REFACTOR_MEMORY_RECEIPT_INTERFACE: Final[str] = "RefactorMemoryReceipt@1"
REFACTOR_MEMORY_INTERFACE: Final[str] = "RefactorMemory@1"

REFACTOR_REUSE_KEY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-reuse-key@1"
)
REFACTOR_TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-transition@1"
)
REFACTOR_REUSE_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-reuse-decision@1"
)
REFACTOR_MEMORY_STORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-memory-store@1"
)
REFACTOR_MEMORY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-memory-receipt@1"
)
REFACTOR_MEMORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-memory@1"
)

MEMORY_CONTRACT_VERSION: Final[str] = "1"

MEMORY_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
MEMORY_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
MEMORY_CAN_CREATE_AUTHORITY: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
SIMILARITY_YIELDS_CONTEXT_ONLY: Final[bool] = True
STALE_EVIDENCE_REVOKES_REUSE: Final[bool] = True
MISMATCHED_EVIDENCE_REVOKES_REUSE: Final[bool] = True
NEGATIVE_EPISODES_RETAINED: Final[bool] = True
EXACT_KEY_REQUIRED_FOR_REUSE: Final[bool] = True
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
PROOF_TRANSFER_ACROSS_STALE_BINDINGS: Final[bool] = False

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_EPISODES: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_REASONS: Final[int] = 64

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
    "similarity_authoritative",
    "vector_authoritative",
)

FORBIDDEN_REUSE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "admit_by_similarity",
        "authorize_completion",
        "authorize_transition",
        "reuse_stale_proof",
        "suppress_raw_source",
        "transfer_stale_binding",
        "vector_reuse",
    }
)

REUSE_KEY_DIMENSIONS: Final[tuple[str, ...]] = (
    "tree_id",
    "state_cid",
    "partition_cid",
    "policy_cid",
    "environment_cid",
    "toolchain_id",
    "obligation_root_cid",
    "validation_cid",
    "procedure_version",
)

DIMENSION_MISMATCH_REASONS: Final[Mapping[str, str]] = {
    "tree_id": "stale_tree",
    "state_cid": "mismatched_state",
    "partition_cid": "mismatched_partition",
    "policy_cid": "mismatched_policy",
    "environment_cid": "mismatched_environment",
    "toolchain_id": "mismatched_toolchain",
    "obligation_root_cid": "mismatched_obligations",
    "validation_cid": "mismatched_validation",
    "procedure_version": "mismatched_procedure_version",
}

DECLARED_OUTCOMES: Final[frozenset[str]] = frozenset({"accepted", "rejected"})
DECLARED_DECISIONS: Final[frozenset[str]] = frozenset(
    {"reuse", "revoke", "context_only"}
)
DECLARED_REVOKE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "stale_tree",
        "mismatched_state",
        "mismatched_partition",
        "mismatched_policy",
        "mismatched_environment",
        "mismatched_toolchain",
        "mismatched_obligations",
        "mismatched_validation",
        "mismatched_procedure_version",
        "negative_episode_blocks_reuse",
        "similarity_not_exact",
        "no_exact_match",
        "missing_raw_source",
        "vector_or_model_evidence",
        "incomplete_key",
    }
)


class RefactorMemoryError(ValueError):
    """Fail-closed violation of a SPAR-033 refactor-memory contract."""


class TransitionOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ReuseDecisionKind(str, Enum):
    REUSE = "reuse"
    REVOKE = "revoke"
    CONTEXT_ONLY = "context_only"


class RevokeReason(str, Enum):
    STALE_TREE = "stale_tree"
    MISMATCHED_STATE = "mismatched_state"
    MISMATCHED_PARTITION = "mismatched_partition"
    MISMATCHED_POLICY = "mismatched_policy"
    MISMATCHED_ENVIRONMENT = "mismatched_environment"
    MISMATCHED_TOOLCHAIN = "mismatched_toolchain"
    MISMATCHED_OBLIGATIONS = "mismatched_obligations"
    MISMATCHED_VALIDATION = "mismatched_validation"
    MISMATCHED_PROCEDURE_VERSION = "mismatched_procedure_version"
    NEGATIVE_EPISODE_BLOCKS_REUSE = "negative_episode_blocks_reuse"
    SIMILARITY_NOT_EXACT = "similarity_not_exact"
    NO_EXACT_MATCH = "no_exact_match"
    MISSING_RAW_SOURCE = "missing_raw_source"
    VECTOR_OR_MODEL_EVIDENCE = "vector_or_model_evidence"
    INCOMPLETE_KEY = "incomplete_key"


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise RefactorMemoryError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise RefactorMemoryError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise RefactorMemoryError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise RefactorMemoryError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise RefactorMemoryError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise RefactorMemoryError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RefactorMemoryError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise RefactorMemoryError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise RefactorMemoryError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise RefactorMemoryError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise RefactorMemoryError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise RefactorMemoryError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise RefactorMemoryError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise RefactorMemoryError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise RefactorMemoryError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise RefactorMemoryError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise RefactorMemoryError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise RefactorMemoryError(f"{name} must not contain duplicates")
    return ordered


def _unique_ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise RefactorMemoryError(f"{name} must be a list")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if len(ordered) > limit:
        raise RefactorMemoryError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _cids(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RefactorMemoryError(f"{name} must be a list of CIDs")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise RefactorMemoryError(f"{name} must not be empty")
    if len(ordered) != len(set(ordered)):
        raise RefactorMemoryError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise RefactorMemoryError(f"{name} exceeds maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise RefactorMemoryError(f"{name} exceeds path bound")
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
        raise RefactorMemoryError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise RefactorMemoryError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RefactorMemoryError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise RefactorMemoryError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise RefactorMemoryError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RefactorMemoryError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise RefactorMemoryError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise RefactorMemoryError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise RefactorMemoryError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise RefactorMemoryError(f"missing {name}")
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
    raise RefactorMemoryError(f"{name} must be a mapping")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RefactorMemoryError(f"{name} must be a list")
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
        raise RefactorMemoryError(f"{name} items must be objects")
    if len(items) > MAX_MEMBERS:
        raise RefactorMemoryError(f"{name} exceeds maximum length")
    return tuple(items)


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise RefactorMemoryError(f"{name} cannot claim {flag}")


def _outcome_value(value: Any, name: str = "outcome") -> str:
    if isinstance(value, TransitionOutcome):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_OUTCOMES:
        raise RefactorMemoryError(f"unsupported {name} {text!r}")
    return text


def _decision_value(value: Any, name: str = "decision") -> str:
    if isinstance(value, ReuseDecisionKind):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_DECISIONS:
        raise RefactorMemoryError(f"unsupported {name} {text!r}")
    return text


def _reason_value(value: Any, name: str = "reason") -> str:
    if isinstance(value, RevokeReason):
        return value.value
    text = _text(getattr(value, "value", value), name)
    if text not in DECLARED_REVOKE_REASONS:
        raise RefactorMemoryError(f"unsupported {name} {text!r}")
    return text


def _reasons(values: Any, name: str = "reasons") -> tuple[str, ...]:
    if values in (None, ()):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RefactorMemoryError(f"{name} must be a list")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        reason = _reason_value(item, name)
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    if len(ordered) > MAX_REASONS:
        raise RefactorMemoryError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _evidence_class(value: Any, name: str = "evidence_class") -> str:
    text = _text(value, name)
    return text


def _network_value(value: Any) -> str:
    text = _text(value, "network")
    if text != NETWORK_DENY:
        raise RefactorMemoryError("network must remain deny")
    return NETWORK_DENY


def refactor_memory_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def refactor_memory_descriptor() -> dict[str, Any]:
    return {
        "interface": REFACTOR_MEMORY_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "predicted_symbols": (
            REFACTOR_TRANSITION_INTERFACE,
            REFACTOR_REUSE_DECISION_INTERFACE,
        ),
        "reuse_key_dimensions": REUSE_KEY_DIMENSIONS,
        "exact_key_required_for_reuse": True,
        "similarity_yields_context_only": True,
        "stale_evidence_revokes_reuse": True,
        "mismatched_evidence_revokes_reuse": True,
        "negative_episodes_retained": True,
        "raw_source_required": True,
        "nomination_only": True,
        "network": NETWORK_DENY,
        "forbids": tuple(sorted(FORBIDDEN_REUSE_NAMES)),
    }


def mismatch_reasons(
    query: "RefactorReuseKey",
    stored: "RefactorReuseKey",
) -> tuple[str, ...]:
    """Return ordered freshness/mismatch reasons for two reuse keys."""

    reasons: list[str] = []
    for dimension in REUSE_KEY_DIMENSIONS:
        if getattr(query, dimension) != getattr(stored, dimension):
            reasons.append(DIMENSION_MISMATCH_REASONS[dimension])
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class RefactorReuseKey:
    """Exact freshness-bound reuse key. Changing any dimension changes identity."""

    tree_id: str
    state_cid: str
    partition_cid: str
    policy_cid: str
    environment_cid: str
    toolchain_id: str
    obligation_root_cid: str
    validation_cid: str
    procedure_version: str

    interface: ClassVar[str] = REFACTOR_REUSE_KEY_INTERFACE
    schema: ClassVar[str] = REFACTOR_REUSE_KEY_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "state_cid",
            "partition_cid",
            "policy_cid",
            "environment_cid",
            "toolchain_id",
            "obligation_root_cid",
            "validation_cid",
            "procedure_version",
            "key_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "state_cid", _cid(self.state_cid, "state_cid"))
        object.__setattr__(
            self, "partition_cid", _cid(self.partition_cid, "partition_cid")
        )
        object.__setattr__(self, "policy_cid", _cid(self.policy_cid, "policy_cid"))
        object.__setattr__(
            self, "environment_cid", _cid(self.environment_cid, "environment_cid")
        )
        object.__setattr__(
            self, "toolchain_id", _text(self.toolchain_id, "toolchain_id")
        )
        object.__setattr__(
            self,
            "obligation_root_cid",
            _cid(self.obligation_root_cid, "obligation_root_cid"),
        )
        object.__setattr__(
            self, "validation_cid", _cid(self.validation_cid, "validation_cid")
        )
        object.__setattr__(
            self,
            "procedure_version",
            _text(self.procedure_version, "procedure_version"),
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_REUSE_KEY_SCHEMA,
            "interface": REFACTOR_REUSE_KEY_INTERFACE,
            "tree_id": self.tree_id,
            "state_cid": self.state_cid,
            "partition_cid": self.partition_cid,
            "policy_cid": self.policy_cid,
            "environment_cid": self.environment_cid,
            "toolchain_id": self.toolchain_id,
            "obligation_root_cid": self.obligation_root_cid,
            "validation_cid": self.validation_cid,
            "procedure_version": self.procedure_version,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def key_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["key_cid"] = self.key_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorReuseKey":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("key_cid")
        if payload.pop("schema") != REFACTOR_REUSE_KEY_SCHEMA:
            raise RefactorMemoryError("unsupported RefactorReuseKey schema")
        if payload.pop("interface") != REFACTOR_REUSE_KEY_INTERFACE:
            raise RefactorMemoryError("unsupported RefactorReuseKey interface")
        result = cls(**payload)
        _verify_cid(claimed, result.key_cid, "key_cid")
        return result


def compile_reuse_key(**fields: Any) -> RefactorReuseKey:
    """Compile one exact freshness-bound reuse key or fail closed."""

    missing = [name for name in REUSE_KEY_DIMENSIONS if name not in fields]
    if missing:
        raise RefactorMemoryError(f"incomplete_key: missing {missing}")
    extra = set(fields) - set(REUSE_KEY_DIMENSIONS)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise RefactorMemoryError(
            f"reuse key identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise RefactorMemoryError(f"unknown reuse key field: {sorted(extra)}")
    return RefactorReuseKey(**fields)


@dataclass(frozen=True, slots=True)
class RefactorTransition:
    """Accepted or rejected refactor episode bound to an exact reuse key."""

    reuse_key: RefactorReuseKey
    outcome: str
    wave_receipt_cid: str
    validation_result_cid: str
    proof_receipt_cid: str
    packet_cid: str
    raw_source_cids: Sequence[str]
    write_paths: Sequence[str]
    validation_commands: Sequence[str]
    evidence_class: str = "transition"
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    raw_source_required: bool = True
    network: str = NETWORK_DENY

    interface: ClassVar[str] = REFACTOR_TRANSITION_INTERFACE
    schema: ClassVar[str] = REFACTOR_TRANSITION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "reuse_key",
            "outcome",
            "negative_episode",
            "wave_receipt_cid",
            "validation_result_cid",
            "proof_receipt_cid",
            "packet_cid",
            "raw_source_cids",
            "write_paths",
            "validation_commands",
            "evidence_class",
            "analyzer_id",
            "adapter_is_nomination_only",
            "raw_source_required",
            "network",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "similarity_authoritative",
            "vector_authoritative",
            "transition_cid",
        }
    )

    def __post_init__(self) -> None:
        key = self.reuse_key
        if not isinstance(key, RefactorReuseKey):
            if isinstance(key, Mapping):
                key = RefactorReuseKey.from_dict(key)
            else:
                raise RefactorMemoryError("reuse_key must be a RefactorReuseKey")
        object.__setattr__(self, "reuse_key", key)
        outcome = _outcome_value(self.outcome)
        object.__setattr__(self, "outcome", outcome)
        evidence = _evidence_class(self.evidence_class)
        if evidence in _NON_ADMITTING_EVIDENCE and outcome == TransitionOutcome.ACCEPTED.value:
            raise RefactorMemoryError(
                "vector, model, or heuristic evidence cannot admit a transition"
            )
        object.__setattr__(self, "evidence_class", evidence)
        object.__setattr__(
            self, "wave_receipt_cid", _cid(self.wave_receipt_cid, "wave_receipt_cid")
        )
        object.__setattr__(
            self,
            "validation_result_cid",
            _cid(self.validation_result_cid, "validation_result_cid"),
        )
        object.__setattr__(
            self, "proof_receipt_cid", _cid(self.proof_receipt_cid, "proof_receipt_cid")
        )
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        sources = _cids(self.raw_source_cids, "raw_source_cids", required=True)
        object.__setattr__(self, "raw_source_cids", sources)
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self, "validation_commands", _commands(self.validation_commands)
        )
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise RefactorMemoryError("transition analyzer_id must remain SPAR-033")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if _bool(self.raw_source_required, "raw_source_required") is not True:
            raise RefactorMemoryError("raw_source_required cannot be disabled")
        object.__setattr__(self, "network", _network_value(self.network))
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "raw_source_required", True)

    @property
    def tree_id(self) -> str:
        return self.reuse_key.tree_id

    @property
    def negative_episode(self) -> bool:
        return self.outcome == TransitionOutcome.REJECTED.value

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
    def similarity_authoritative(self) -> bool:
        return False

    @property
    def vector_authoritative(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_TRANSITION_SCHEMA,
            "interface": REFACTOR_TRANSITION_INTERFACE,
            "reuse_key": self.reuse_key.to_dict(),
            "outcome": self.outcome,
            "negative_episode": self.negative_episode,
            "wave_receipt_cid": self.wave_receipt_cid,
            "validation_result_cid": self.validation_result_cid,
            "proof_receipt_cid": self.proof_receipt_cid,
            "packet_cid": self.packet_cid,
            "raw_source_cids": list(self.raw_source_cids),
            "write_paths": list(self.write_paths),
            "validation_commands": list(self.validation_commands),
            "evidence_class": self.evidence_class,
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "raw_source_required": True,
            "network": NETWORK_DENY,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "similarity_authoritative": False,
            "vector_authoritative": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def transition_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["transition_cid"] = self.transition_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorTransition":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("transition_cid")
        if payload.pop("schema") != REFACTOR_TRANSITION_SCHEMA:
            raise RefactorMemoryError("unsupported RefactorTransition schema")
        if payload.pop("interface") != REFACTOR_TRANSITION_INTERFACE:
            raise RefactorMemoryError("unsupported RefactorTransition interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("raw_source_required") is not True:
            raise RefactorMemoryError("raw_source_required cannot be disabled")
        if payload.pop("adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise RefactorMemoryError("transition analyzer_id must remain SPAR-033")
        if payload.pop("network") != NETWORK_DENY:
            raise RefactorMemoryError("network must remain deny")
        payload.pop("negative_episode")
        result = cls(**payload)
        _verify_cid(claimed, result.transition_cid, "transition_cid")
        return result


def compile_refactor_transition(**fields: Any) -> RefactorTransition:
    """Compile one RefactorTransition@1 episode or fail closed."""

    return RefactorTransition(**fields)


@dataclass(frozen=True, slots=True)
class RefactorReuseDecision:
    """Exact reuse decision. Similarity cannot admit reuse."""

    query_key_cid: str
    decision: str
    reasons: Sequence[str] = ()
    matched_transition_cid: str = ""
    context_transition_cids: Sequence[str] = ()
    retained_negative_cids: Sequence[str] = ()
    exact_match: bool = False
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    similarity_authoritative: bool = False
    vector_authoritative: bool = False

    interface: ClassVar[str] = REFACTOR_REUSE_DECISION_INTERFACE
    schema: ClassVar[str] = REFACTOR_REUSE_DECISION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "query_key_cid",
            "decision",
            "reasons",
            "matched_transition_cid",
            "context_transition_cids",
            "retained_negative_cids",
            "exact_match",
            "analyzer_id",
            "adapter_is_nomination_only",
            "similarity_authoritative",
            "vector_authoritative",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "decision_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "query_key_cid", _cid(self.query_key_cid, "query_key_cid")
        )
        decision = _decision_value(self.decision)
        object.__setattr__(self, "decision", decision)
        reasons = _reasons(self.reasons)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "matched_transition_cid",
            _optional_cid(self.matched_transition_cid, "matched_transition_cid"),
        )
        object.__setattr__(
            self,
            "context_transition_cids",
            _unique_ordered_text(
                list(self.context_transition_cids),
                "context_transition_cids",
                limit=MAX_EPISODES,
            ),
        )
        object.__setattr__(
            self,
            "retained_negative_cids",
            _unique_ordered_text(
                list(self.retained_negative_cids),
                "retained_negative_cids",
                limit=MAX_EPISODES,
            ),
        )
        exact = _bool(self.exact_match, "exact_match")
        object.__setattr__(self, "exact_match", exact)
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if _bool(self.similarity_authoritative, "similarity_authoritative") is not False:
            raise RefactorMemoryError("similarity cannot be authoritative")
        if _bool(self.vector_authoritative, "vector_authoritative") is not False:
            raise RefactorMemoryError("vector similarity is not authority")
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise RefactorMemoryError("decision analyzer_id must remain SPAR-033")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "similarity_authoritative", False)
        object.__setattr__(self, "vector_authoritative", False)
        if decision == ReuseDecisionKind.REUSE.value:
            if exact is not True:
                raise RefactorMemoryError("reuse requires an exact key match")
            if not self.matched_transition_cid:
                raise RefactorMemoryError("reuse requires a matched transition")
            if reasons:
                raise RefactorMemoryError("reuse cannot carry revoke reasons")
        if decision == ReuseDecisionKind.REVOKE.value and not reasons:
            raise RefactorMemoryError("revoke requires at least one reason")
        if decision == ReuseDecisionKind.CONTEXT_ONLY.value:
            if exact is True:
                raise RefactorMemoryError("exact match cannot be context_only")
            if RevokeReason.SIMILARITY_NOT_EXACT.value not in reasons:
                raise RefactorMemoryError("context_only requires similarity_not_exact")

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
            "schema": REFACTOR_REUSE_DECISION_SCHEMA,
            "interface": REFACTOR_REUSE_DECISION_INTERFACE,
            "query_key_cid": self.query_key_cid,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "matched_transition_cid": self.matched_transition_cid,
            "context_transition_cids": list(self.context_transition_cids),
            "retained_negative_cids": list(self.retained_negative_cids),
            "exact_match": self.exact_match,
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "similarity_authoritative": False,
            "vector_authoritative": False,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def decision_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["decision_cid"] = self.decision_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorReuseDecision":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("decision_cid")
        if payload.pop("schema") != REFACTOR_REUSE_DECISION_SCHEMA:
            raise RefactorMemoryError("unsupported RefactorReuseDecision schema")
        if payload.pop("interface") != REFACTOR_REUSE_DECISION_INTERFACE:
            raise RefactorMemoryError("unsupported RefactorReuseDecision interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise RefactorMemoryError("decision analyzer_id must remain SPAR-033")
        result = cls(**payload)
        _verify_cid(claimed, result.decision_cid, "decision_cid")
        return result


def _coerce_key(value: Any, name: str = "reuse_key") -> RefactorReuseKey:
    if isinstance(value, RefactorReuseKey):
        return value
    if isinstance(value, Mapping):
        if "key_cid" in value:
            return RefactorReuseKey.from_dict(value)
        return compile_reuse_key(**dict(value))
    raise RefactorMemoryError(f"{name} must be a RefactorReuseKey")


def _coerce_transition(value: Any, name: str = "transition") -> RefactorTransition:
    if isinstance(value, RefactorTransition):
        return value
    if isinstance(value, Mapping):
        if "transition_cid" in value:
            return RefactorTransition.from_dict(value)
        return compile_refactor_transition(**dict(value))
    raise RefactorMemoryError(f"{name} must be a RefactorTransition")


def _coerce_episodes(values: Any) -> tuple[RefactorTransition, ...]:
    if values in (None, ()):
        return ()
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise RefactorMemoryError("episodes must be a list")
    episodes: list[RefactorTransition] = []
    seen: set[str] = set()
    for item in values:
        episode = _coerce_transition(item, "episodes")
        if episode.transition_cid in seen:
            continue
        seen.add(episode.transition_cid)
        episodes.append(episode)
    if len(episodes) > MAX_EPISODES:
        raise RefactorMemoryError("episodes exceed maximum length")
    return tuple(episodes)


def _similar_context_cids(similar_hits: Any) -> tuple[str, ...]:
    if similar_hits in (None, ()):
        return ()
    items = _mapping_sequence(similar_hits, "similar_hits")
    cids: list[str] = []
    seen: set[str] = set()
    for item in items:
        evidence = str(item.get("evidence_class") or item.get("channel") or "")
        if evidence in _NON_ADMITTING_EVIDENCE or evidence in {
            "lexical",
            "graph",
            "vector",
            "vector_candidate",
            "analogous",
        }:
            claimed = item.get("transition_cid") or item.get("record_cid") or item.get("hit_cid")
            if claimed in (None, ""):
                continue
            cid = _cid(claimed, "similar_hits")
            if cid not in seen:
                seen.add(cid)
                cids.append(cid)
            continue
        if item.get("exact") is True or item.get("channel") == "exact":
            raise RefactorMemoryError(
                "similar_hits cannot claim exact reuse; exact identity is a key match"
            )
    return tuple(cids)


def decide_exact_reuse(
    query_key: RefactorReuseKey | Mapping[str, Any],
    episodes: Sequence[RefactorTransition | Mapping[str, Any]] = (),
    *,
    similar_hits: Sequence[Mapping[str, Any]] = (),
) -> RefactorReuseDecision:
    """Decide exact reuse. Similarity is context only; mismatch revokes."""

    query = _coerce_key(query_key, "query_key")
    stored = _coerce_episodes(episodes)
    context_cids = _similar_context_cids(similar_hits)
    exact: list[RefactorTransition] = [
        item for item in stored if item.reuse_key.key_cid == query.key_cid
    ]
    negatives = tuple(
        item.transition_cid for item in exact if item.negative_episode
    )
    accepted = [item for item in exact if not item.negative_episode]
    mismatch_context: list[str] = []
    mismatch: list[str] = []
    for item in stored:
        if item.reuse_key.key_cid == query.key_cid:
            continue
        reasons = mismatch_reasons(query, item.reuse_key)
        if reasons:
            for reason in reasons:
                if reason not in mismatch:
                    mismatch.append(reason)
            mismatch_context.append(item.transition_cid)

    if exact and negatives:
        return RefactorReuseDecision(
            query_key_cid=query.key_cid,
            decision=ReuseDecisionKind.REVOKE.value,
            reasons=(RevokeReason.NEGATIVE_EPISODE_BLOCKS_REUSE.value,),
            matched_transition_cid=negatives[0],
            context_transition_cids=context_cids,
            retained_negative_cids=negatives,
            exact_match=True,
        )
    if accepted:
        return RefactorReuseDecision(
            query_key_cid=query.key_cid,
            decision=ReuseDecisionKind.REUSE.value,
            reasons=(),
            matched_transition_cid=accepted[0].transition_cid,
            context_transition_cids=context_cids,
            retained_negative_cids=(),
            exact_match=True,
        )
    if context_cids:
        return RefactorReuseDecision(
            query_key_cid=query.key_cid,
            decision=ReuseDecisionKind.CONTEXT_ONLY.value,
            reasons=(RevokeReason.SIMILARITY_NOT_EXACT.value,),
            matched_transition_cid="",
            context_transition_cids=context_cids,
            retained_negative_cids=tuple(
                item.transition_cid for item in stored if item.negative_episode
            ),
            exact_match=False,
        )
    if mismatch:
        return RefactorReuseDecision(
            query_key_cid=query.key_cid,
            decision=ReuseDecisionKind.REVOKE.value,
            reasons=tuple(mismatch),
            matched_transition_cid="",
            context_transition_cids=tuple(mismatch_context),
            retained_negative_cids=tuple(
                item.transition_cid for item in stored if item.negative_episode
            ),
            exact_match=False,
        )
    return RefactorReuseDecision(
        query_key_cid=query.key_cid,
        decision=ReuseDecisionKind.REVOKE.value,
        reasons=(RevokeReason.NO_EXACT_MATCH.value,),
        matched_transition_cid="",
        context_transition_cids=(),
        retained_negative_cids=(),
        exact_match=False,
    )


@dataclass(frozen=True, slots=True)
class RefactorMemoryStore:
    """Append-only exact episode store. Negative episodes are retained."""

    episodes: Sequence[RefactorTransition] = ()
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    negative_episodes_retained: bool = True

    interface: ClassVar[str] = REFACTOR_MEMORY_STORE_INTERFACE
    schema: ClassVar[str] = REFACTOR_MEMORY_STORE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "episodes",
            "analyzer_id",
            "adapter_is_nomination_only",
            "negative_episodes_retained",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "store_cid",
        }
    )

    def __post_init__(self) -> None:
        episodes = _coerce_episodes(self.episodes)
        object.__setattr__(self, "episodes", episodes)
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise RefactorMemoryError("store analyzer_id must remain SPAR-033")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if _bool(self.negative_episodes_retained, "negative_episodes_retained") is not True:
            raise RefactorMemoryError("negative episodes must be retained")
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "negative_episodes_retained", True)

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
    def accepted_transition_cids(self) -> tuple[str, ...]:
        return tuple(
            item.transition_cid for item in self.episodes if not item.negative_episode
        )

    @property
    def rejected_transition_cids(self) -> tuple[str, ...]:
        return tuple(
            item.transition_cid for item in self.episodes if item.negative_episode
        )

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_MEMORY_STORE_SCHEMA,
            "interface": REFACTOR_MEMORY_STORE_INTERFACE,
            "episodes": [item.to_dict() for item in self.episodes],
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "negative_episodes_retained": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def store_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["store_cid"] = self.store_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorMemoryStore":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("store_cid")
        if payload.pop("schema") != REFACTOR_MEMORY_STORE_SCHEMA:
            raise RefactorMemoryError("unsupported RefactorMemoryStore schema")
        if payload.pop("interface") != REFACTOR_MEMORY_STORE_INTERFACE:
            raise RefactorMemoryError("unsupported RefactorMemoryStore interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if payload.pop("negative_episodes_retained") is not True:
            raise RefactorMemoryError("negative episodes must be retained")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise RefactorMemoryError("store analyzer_id must remain SPAR-033")
        result = cls(**payload)
        _verify_cid(claimed, result.store_cid, "store_cid")
        return result


def remember_transition(
    store: RefactorMemoryStore | Mapping[str, Any] | None,
    transition: RefactorTransition | Mapping[str, Any],
) -> RefactorMemoryStore:
    """Append one episode. Duplicate identities are idempotent."""

    current = (
        RefactorMemoryStore()
        if store in (None, ())
        else (
            store
            if isinstance(store, RefactorMemoryStore)
            else RefactorMemoryStore.from_dict(_as_mapping(store, "store"))
        )
    )
    episode = _coerce_transition(transition)
    return RefactorMemoryStore(episodes=(*current.episodes, episode))


def retain_negative_episode(
    store: RefactorMemoryStore | Mapping[str, Any] | None,
    transition: RefactorTransition | Mapping[str, Any],
) -> RefactorMemoryStore:
    """Retain a rejected episode. Accepted input is rejected."""

    episode = _coerce_transition(transition)
    if not episode.negative_episode:
        raise RefactorMemoryError("retain_negative_episode requires a rejected outcome")
    return remember_transition(store, episode)


def lookup_exact_reuse(
    store: RefactorMemoryStore | Mapping[str, Any],
    query_key: RefactorReuseKey | Mapping[str, Any],
    *,
    similar_hits: Sequence[Mapping[str, Any]] = (),
) -> RefactorReuseDecision:
    """Look up exact reuse against a retained store."""

    current = (
        store
        if isinstance(store, RefactorMemoryStore)
        else RefactorMemoryStore.from_dict(_as_mapping(store, "store"))
    )
    return decide_exact_reuse(
        query_key, current.episodes, similar_hits=similar_hits
    )


@dataclass(frozen=True, slots=True)
class RefactorMemoryReceipt:
    """Body-free SPAR-033 memory receipt. Nomination-only."""

    store_cid: str
    decision_cid: str
    query_key_cid: str
    decision: str
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    mutated: bool = False
    deterministic: bool = True

    interface: ClassVar[str] = REFACTOR_MEMORY_RECEIPT_INTERFACE
    schema: ClassVar[str] = REFACTOR_MEMORY_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "store_cid",
            "decision_cid",
            "query_key_cid",
            "decision",
            "analyzer_id",
            "adapter_is_nomination_only",
            "mutated",
            "deterministic",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_cid", _cid(self.store_cid, "store_cid"))
        object.__setattr__(self, "decision_cid", _cid(self.decision_cid, "decision_cid"))
        object.__setattr__(
            self, "query_key_cid", _cid(self.query_key_cid, "query_key_cid")
        )
        object.__setattr__(self, "decision", _decision_value(self.decision))
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise RefactorMemoryError("receipt analyzer_id must remain SPAR-033")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if _bool(self.mutated, "mutated") is not False:
            raise RefactorMemoryError("memory cannot mutate")
        if _bool(self.deterministic, "deterministic") is not True:
            raise RefactorMemoryError("memory must remain deterministic")
        object.__setattr__(self, "adapter_is_nomination_only", True)
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

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_MEMORY_RECEIPT_SCHEMA,
            "interface": REFACTOR_MEMORY_RECEIPT_INTERFACE,
            "store_cid": self.store_cid,
            "decision_cid": self.decision_cid,
            "query_key_cid": self.query_key_cid,
            "decision": self.decision,
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "mutated": False,
            "deterministic": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorMemoryReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != REFACTOR_MEMORY_RECEIPT_SCHEMA:
            raise RefactorMemoryError("unsupported RefactorMemoryReceipt schema")
        if payload.pop("interface") != REFACTOR_MEMORY_RECEIPT_INTERFACE:
            raise RefactorMemoryError("unsupported RefactorMemoryReceipt interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise RefactorMemoryError("adapter must remain nomination_only")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise RefactorMemoryError("receipt analyzer_id must remain SPAR-033")
        result = cls(**payload)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


def compile_memory_receipt(
    store: RefactorMemoryStore,
    decision: RefactorReuseDecision,
) -> RefactorMemoryReceipt:
    return RefactorMemoryReceipt(
        store_cid=store.store_cid,
        decision_cid=decision.decision_cid,
        query_key_cid=decision.query_key_cid,
        decision=decision.decision,
    )


class RefactorMemory:
    """SPAR-033 exact reuse adapter. Nomination-only; dry-run never mutates."""

    def remember(
        self,
        store: RefactorMemoryStore | Mapping[str, Any] | None,
        transition: RefactorTransition | Mapping[str, Any],
    ) -> RefactorMemoryStore:
        return remember_transition(store, transition)

    def retain_negative(
        self,
        store: RefactorMemoryStore | Mapping[str, Any] | None,
        transition: RefactorTransition | Mapping[str, Any],
    ) -> RefactorMemoryStore:
        return retain_negative_episode(store, transition)

    def decide(
        self,
        query_key: RefactorReuseKey | Mapping[str, Any],
        episodes: Sequence[RefactorTransition | Mapping[str, Any]] = (),
        *,
        similar_hits: Sequence[Mapping[str, Any]] = (),
    ) -> RefactorReuseDecision:
        return decide_exact_reuse(query_key, episodes, similar_hits=similar_hits)

    def lookup(
        self,
        store: RefactorMemoryStore | Mapping[str, Any],
        query_key: RefactorReuseKey | Mapping[str, Any],
        *,
        similar_hits: Sequence[Mapping[str, Any]] = (),
    ) -> RefactorReuseDecision:
        return lookup_exact_reuse(store, query_key, similar_hits=similar_hits)

    def receipt(
        self,
        store: RefactorMemoryStore,
        decision: RefactorReuseDecision,
    ) -> RefactorMemoryReceipt:
        return compile_memory_receipt(store, decision)

    def dry_run(
        self,
        query_key: RefactorReuseKey | Mapping[str, Any],
        episodes: Sequence[RefactorTransition | Mapping[str, Any]] = (),
        *,
        similar_hits: Sequence[Mapping[str, Any]] = (),
    ) -> RefactorReuseDecision:
        return self.decide(query_key, episodes, similar_hits=similar_hits)


def dry_run_exact_reuse(
    query_key: RefactorReuseKey | Mapping[str, Any],
    episodes: Sequence[RefactorTransition | Mapping[str, Any]] = (),
    *,
    similar_hits: Sequence[Mapping[str, Any]] = (),
) -> RefactorReuseDecision:
    """Deterministic dry-run. Never mutates."""

    return decide_exact_reuse(query_key, episodes, similar_hits=similar_hits)


def encode_canonical_key(key: RefactorReuseKey) -> dict[str, Any]:
    return key.to_dict()


def decode_canonical_key(payload: Mapping[str, Any]) -> RefactorReuseKey:
    return RefactorReuseKey.from_dict(payload)


def encode_canonical_transition(transition: RefactorTransition) -> dict[str, Any]:
    return transition.to_dict()


def decode_canonical_transition(payload: Mapping[str, Any]) -> RefactorTransition:
    return RefactorTransition.from_dict(payload)


def encode_canonical_decision(decision: RefactorReuseDecision) -> dict[str, Any]:
    return decision.to_dict()


def decode_canonical_decision(payload: Mapping[str, Any]) -> RefactorReuseDecision:
    return RefactorReuseDecision.from_dict(payload)


def encode_canonical_store(store: RefactorMemoryStore) -> dict[str, Any]:
    return store.to_dict()


def decode_canonical_store(payload: Mapping[str, Any]) -> RefactorMemoryStore:
    return RefactorMemoryStore.from_dict(payload)


def encode_canonical_receipt(receipt: RefactorMemoryReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> RefactorMemoryReceipt:
    return RefactorMemoryReceipt.from_dict(payload)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise RefactorMemoryError(
            f"refactor memory must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_IS_NOMINATION_ONLY",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "DECLARED_DECISIONS",
    "DECLARED_OUTCOMES",
    "DECLARED_REVOKE_REASONS",
    "DIMENSION_MISMATCH_REASONS",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXACT_KEY_REQUIRED_FOR_REUSE",
    "FORBIDDEN_REUSE_NAMES",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MEMORY_CAN_AUTHORIZE_COMPLETION",
    "MEMORY_CAN_AUTHORIZE_TRANSITION",
    "MEMORY_CAN_CREATE_AUTHORITY",
    "MEMORY_CONTRACT_VERSION",
    "MISMATCHED_EVIDENCE_REVOKES_REUSE",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "NEGATIVE_EPISODES_RETAINED",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "PROGRAM",
    "PROOF_TRANSFER_ACROSS_STALE_BINDINGS",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REFACTOR_MEMORY_INTERFACE",
    "REFACTOR_MEMORY_RECEIPT_INTERFACE",
    "REFACTOR_MEMORY_STORE_INTERFACE",
    "REFACTOR_REUSE_DECISION_INTERFACE",
    "REFACTOR_REUSE_KEY_INTERFACE",
    "REFACTOR_TRANSITION_INTERFACE",
    "REUSE_KEY_DIMENSIONS",
    "SIMILARITY_YIELDS_CONTEXT_ONLY",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "STALE_EVIDENCE_REVOKES_REUSE",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TEST_PASS_IS_NOT_PROOF",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "RefactorMemory",
    "RefactorMemoryError",
    "RefactorMemoryReceipt",
    "RefactorMemoryStore",
    "RefactorReuseDecision",
    "RefactorReuseKey",
    "RefactorTransition",
    "ReuseDecisionKind",
    "RevokeReason",
    "TransitionOutcome",
    "assert_not_competing_capsule_family",
    "compile_memory_receipt",
    "compile_refactor_transition",
    "compile_reuse_key",
    "decode_canonical_decision",
    "decode_canonical_key",
    "decode_canonical_receipt",
    "decode_canonical_store",
    "decode_canonical_transition",
    "decide_exact_reuse",
    "dry_run_exact_reuse",
    "encode_canonical_decision",
    "encode_canonical_key",
    "encode_canonical_receipt",
    "encode_canonical_store",
    "encode_canonical_transition",
    "lookup_exact_reuse",
    "mismatch_reasons",
    "provider_free_exports",
    "refactor_memory_cid_profile",
    "refactor_memory_descriptor",
    "remember_transition",
    "retain_negative_episode",
]
