"""SPAR-035 minimal semantic context and residual model routing.

This module extends the current ContextCompiler/compression harness with
``SemanticRefactorContextAdapter@1``.  It packs named unresolved questions,
exact affected slices, contracts, counterexamples, evidence, analogous
refactors, and allowed effects, then emits a residual route receipt.

ContextCompiler remains the only context-compilation authority.  This adapter
never replaces it, never dumps a full task, and never creates a competing
context, scheduler, vector, proof, or completion authority.  Exact SPAR-033
reuse, verified procedures, deterministic analysis/transform, and proof
search close without a general model.  A general model is invoked only for
one named unresolved typed residual.  Repeated identical failures do not
retry without new evidence.  Model output is proposal-only and cannot
weaken validation.

The adapter is nomination-only.  Vector, model, and heuristic evidence
cannot admit a route or suppress raw-source fallback.  Observational
metadata is excluded from identity.  Dry-run is deterministic and never
mutates.  Network is denied.
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


TASK_ID: Final[str] = "SPAR-035"
GOAL_ID: Final[str] = "SPAR-G063"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "route/context decisions"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.context_adapter@1"
)
CONTEXT_COMPILER_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.context.context_compiler"
)

SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE: Final[str] = (
    "SemanticRefactorContextAdapter@1"
)
SEMANTIC_REFACTOR_CONTEXT_INTERFACE: Final[str] = "SemanticRefactorContext@1"
NAMED_UNRESOLVED_QUESTION_INTERFACE: Final[str] = "NamedUnresolvedQuestion@1"
AFFECTED_SLICE_INTERFACE: Final[str] = "AffectedSlice@1"
RESIDUAL_MODEL_ROUTE_INTERFACE: Final[str] = "ResidualModelRoute@1"
RESIDUAL_ROUTE_RECEIPT_INTERFACE: Final[str] = "ResidualRouteReceipt@1"

SEMANTIC_REFACTOR_CONTEXT_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-refactor-context-adapter@1"
)
SEMANTIC_REFACTOR_CONTEXT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-refactor-context@1"
)
NAMED_UNRESOLVED_QUESTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/named-unresolved-question@1"
)
AFFECTED_SLICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/affected-slice@1"
)
RESIDUAL_MODEL_ROUTE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-model-route@1"
)
RESIDUAL_ROUTE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/residual-route-receipt@1"
)

CONTEXT_CONTRACT_VERSION: Final[str] = "1"

CONTEXT_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
CONTEXT_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
CONTEXT_CAN_CREATE_AUTHORITY: Final[bool] = False
CONTEXT_CAN_REPLACE_COMPILER: Final[bool] = False
CONTEXT_CAN_WEAKEN_VALIDATION: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
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
CONTEXT_COMPILER_REMAINS_AUTHORITY: Final[bool] = True
ONE_RESIDUAL_GENERAL_MODEL_QUESTION: Final[bool] = True
INDEPENDENT_VALIDATION_REQUIRED: Final[bool] = True
IDENTICAL_FAILURE_RETRY_WITHOUT_EVIDENCE: Final[bool] = False
MAX_GENERAL_MODEL_QUESTIONS: Final[int] = 1

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_QUESTIONS: Final[int] = 16
MAX_SLICES: Final[int] = 64
MAX_CONTRACTS: Final[int] = 64
MAX_COUNTEREXAMPLES: Final[int] = 64
MAX_ANALOGS: Final[int] = 64
MAX_REASONS: Final[int] = 64
MAX_FAILURES: Final[int] = 256

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
    "can_replace_compiler",
    "can_weaken_validation",
    "projection_is_authority",
    "context_is_authority",
    "similarity_authoritative",
    "vector_authoritative",
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
        "full_source",
        "full_task_body",
        "full_task_dump",
        "prompt",
        "repository_dump",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "task_prose",
        "transcript",
        "snippet",
    }
)

FORBIDDEN_CONTEXT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "compile_full_task_dump",
        "invent_unresolved_question",
        "weaken_validation",
        "retry_identical_failure",
        "route_multiple_residuals",
        "replace_context_compiler",
        "suppress_raw_source",
        "authorize_completion",
        "open_network",
        "admit_by_similarity",
    }
)

DEFAULT_ALLOWED_EFFECTS: Final[tuple[str, ...]] = (
    "bounded_source_edit",
    "bounded_test_edit",
    "isolated_validation",
)
DECLARED_ALLOWED_EFFECTS: Final[frozenset[str]] = frozenset(DEFAULT_ALLOWED_EFFECTS)
DECLARED_FORBIDDEN_EFFECTS: Final[frozenset[str]] = frozenset(
    {
        "network",
        "install",
        "model_download",
        "protected_branch",
        "credential",
        "production",
        "unrestricted_diff",
        "hidden_dynamic_frontier",
        "direct_multiprocess_duckdb",
    }
)

NO_MODEL_STEPS: Final[tuple[str, ...]] = (
    "exact_reuse",
    "verified_procedure",
    "deterministic_analysis",
    "proof_search",
    "deterministic_transform",
)
MODEL_ROUTE_CLASS: Final[tuple[str, ...]] = (
    "exact_reuse",
    "verified_procedure",
    "deterministic_analysis",
    "proof_search",
    "deterministic_transform",
    "specialist_ranking",
    "residual_general_model",
)
DECLARED_REUSE_DECISIONS: Final[frozenset[str]] = frozenset(
    {"reuse", "revoke", "context_only"}
)
DECLARED_ANALYSIS_KINDS: Final[frozenset[str]] = frozenset(
    {"ast", "cst", "graph", "state", "effect", "contract"}
)
DECLARED_RANKING_CHANNELS: Final[frozenset[str]] = frozenset(
    {"vector", "hybrid", "lexical"}
)


class ContextAdapterError(ValueError):
    """Fail-closed violation of a SPAR-035 context/route contract."""

    def __init__(self, message: str, *, reason_code: str = "malformed") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)


class RouteKind(str, Enum):
    EXACT_REUSE = "exact_reuse"
    VERIFIED_PROCEDURE = "verified_procedure"
    DETERMINISTIC_ANALYSIS = "deterministic_analysis"
    PROOF_SEARCH = "proof_search"
    DETERMINISTIC_TRANSFORM = "deterministic_transform"
    SPECIALIST_RANKING = "specialist_ranking"
    RESIDUAL_GENERAL_MODEL = "residual_general_model"
    NO_MODEL = "no_model"
    BLOCKED = "blocked"


class RouteStatus(str, Enum):
    NOMINATED = "nominated"
    REUSED = "reused"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"
    NO_MODEL = "no_model"


class QuestionKind(str, Enum):
    BOUNDARY_CONTRACT = "boundary_contract"
    STATE_OWNERSHIP = "state_ownership"
    INITIALIZATION_ORDER = "initialization_order"
    EFFECT_SCOPE = "effect_scope"
    COMPATIBILITY = "compatibility"
    PARTITION_CUT = "partition_cut"
    VALIDATION_GAP = "validation_gap"


class AnalogChannel(str, Enum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


DECLARED_ROUTE_KINDS: Final[frozenset[str]] = frozenset(item.value for item in RouteKind)
DECLARED_ROUTE_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in RouteStatus
)
DECLARED_QUESTION_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in QuestionKind
)
DECLARED_ANALOG_CHANNELS: Final[frozenset[str]] = frozenset(
    item.value for item in AnalogChannel
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise ContextAdapterError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ContextAdapterError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise ContextAdapterError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise ContextAdapterError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise ContextAdapterError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise ContextAdapterError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ContextAdapterError(f"{name} must be a boolean")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise ContextAdapterError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise ContextAdapterError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise ContextAdapterError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise ContextAdapterError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise ContextAdapterError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise ContextAdapterError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _reject_body(payload: Mapping[str, Any], name: str) -> None:
    present = _FORBIDDEN_BODY_KEYS & set(payload)
    if present:
        raise ContextAdapterError(
            f"{name} must remain body-free; forbidden keys: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise ContextAdapterError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise ContextAdapterError(f"{name} does not verify")


def _unique_ordered_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ContextAdapterError(f"{name} must be a list")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if len(ordered) > limit:
        raise ContextAdapterError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _cids(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContextAdapterError(f"{name} must be a list of CIDs")
    ordered = tuple(sorted(_cid(item, name) for item in values))
    if required and not ordered:
        raise ContextAdapterError(f"{name} must not be empty")
    if len(ordered) != len(set(ordered)):
        raise ContextAdapterError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise ContextAdapterError(f"{name} exceeds maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise ContextAdapterError(f"{name} exceeds path bound")
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
        raise ContextAdapterError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise ContextAdapterError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContextAdapterError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise ContextAdapterError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise ContextAdapterError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ContextAdapterError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise ContextAdapterError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise ContextAdapterError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise ContextAdapterError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise ContextAdapterError(f"missing {name}")
    if isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray)):
        _reject_excluded(value, name)
        _reject_body(value, name)
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping) and not isinstance(
            payload, (str, bytes, bytearray)
        ):
            _reject_excluded(payload, name)
            _reject_body(payload, name)
            return dict(payload)
    raise ContextAdapterError(f"{name} must be a mapping")


def _optional_mapping(value: Any, name: str) -> dict[str, Any] | None:
    if value in (None, {}, ()):
        return None
    return _as_mapping(value, name)


def _mapping_sequence(value: Any, name: str, *, limit: int) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContextAdapterError(f"{name} must be a list")
    items: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping) and not isinstance(item, (str, bytes, bytearray)):
            _reject_body(item, name)
            items.append(dict(item))
            continue
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, Mapping):
                _reject_body(payload, name)
                items.append(dict(payload))
                continue
        raise ContextAdapterError(f"{name} items must be objects")
    if len(items) > limit:
        raise ContextAdapterError(f"{name} exceeds maximum length")
    return tuple(items)


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise ContextAdapterError(f"{name} cannot claim {flag}")


def _network_value(value: Any) -> str:
    text = _text(value, "network")
    if text != NETWORK_DENY:
        raise ContextAdapterError("network is denied")
    return NETWORK_DENY


def _question_kind(value: Any) -> str:
    if isinstance(value, QuestionKind):
        return value.value
    text = _text(getattr(value, "value", value), "kind")
    if text not in DECLARED_QUESTION_KINDS:
        raise ContextAdapterError(f"unsupported question kind {text!r}")
    return text


def _route_kind(value: Any) -> str:
    if isinstance(value, RouteKind):
        return value.value
    text = _text(getattr(value, "value", value), "route_kind")
    if text not in DECLARED_ROUTE_KINDS:
        raise ContextAdapterError(f"unsupported route_kind {text!r}")
    return text


def _route_status(value: Any) -> str:
    if isinstance(value, RouteStatus):
        return value.value
    text = _text(getattr(value, "value", value), "status")
    if text not in DECLARED_ROUTE_STATUSES:
        raise ContextAdapterError(f"unsupported status {text!r}")
    return text


def _attr_or_key(payload: Mapping[str, Any] | Any, *names: str) -> Any:
    if isinstance(payload, Mapping):
        for name in names:
            if name in payload:
                return payload[name]
    for name in names:
        if hasattr(payload, name):
            return getattr(payload, name)
    return None


def context_adapter_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def context_adapter_descriptor() -> dict[str, Any]:
    return {
        "interface": SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "predicted_symbols": (
            SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE,
            RESIDUAL_ROUTE_RECEIPT_INTERFACE,
        ),
        "context_compiler_authority": CONTEXT_COMPILER_AUTHORITY,
        "context_compiler_remains_authority": True,
        "nomination_only": True,
        "raw_source_required": True,
        "network": NETWORK_DENY,
        "one_residual_general_model_question": True,
        "independent_validation_required": True,
        "identical_failure_retry_without_evidence": False,
        "model_output_is_proposal_only": True,
        "can_weaken_validation": False,
        "no_model_steps": NO_MODEL_STEPS,
        "model_route_class": MODEL_ROUTE_CLASS,
        "forbids": tuple(sorted(FORBIDDEN_CONTEXT_NAMES)),
    }


def _packet_write_paths(packet: Mapping[str, Any]) -> tuple[str, ...]:
    if "write_paths" in packet:
        return _exact_paths(packet["write_paths"], "write_paths")
    scope = packet.get("effect_scope")
    if isinstance(scope, Mapping) and "write_paths" in scope:
        return _exact_paths(scope["write_paths"], "write_paths")
    raise ContextAdapterError("SPAR-019 packet write_paths are required")


def _packet_source_cids(packet: Mapping[str, Any]) -> tuple[str, ...]:
    preimage = packet.get("preimage")
    sources = None
    if isinstance(preimage, Mapping):
        sources = preimage.get("source_cids")
    if sources in (None, ()):
        sources = packet.get("raw_source_cids") or packet.get("source_cids")
    if sources in (None, ()):
        raise ContextAdapterError("raw source required: packet source_cids")
    ordered = tuple(sorted(_cid(item, "source_cids") for item in sources))
    if not ordered:
        raise ContextAdapterError("raw source required: packet source_cids")
    if len(ordered) != len(set(ordered)):
        raise ContextAdapterError("source_cids must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise ContextAdapterError("source_cids exceed maximum length")
    return ordered


def _packet_validation_commands(packet: Mapping[str, Any]) -> tuple[str, ...]:
    commands = packet.get("validation_commands")
    if commands in (None, ()):
        raise ContextAdapterError("validation_commands are required")
    return _commands(commands)


def _packet_cid(packet: Mapping[str, Any]) -> str:
    claimed = packet.get("packet_cid")
    if claimed in (None, ""):
        raise ContextAdapterError("packet_cid is required")
    return _cid(claimed, "packet_cid")


def _packet_tree(packet: Mapping[str, Any]) -> str:
    return _tree_id(packet.get("tree_id"))


def _reuse_decision(value: Any) -> dict[str, Any]:
    payload = _as_mapping(value, "SPAR-033 reuse_decision")
    decision = _text(payload.get("decision"), "reuse_decision.decision")
    if decision not in DECLARED_REUSE_DECISIONS:
        raise ContextAdapterError(f"unsupported SPAR-033 decision {decision!r}")
    exact = payload.get("exact_match")
    if exact is None:
        exact = decision == "reuse"
    exact_match = _bool(exact, "reuse_decision.exact_match")
    if decision == "reuse" and exact_match is not True:
        raise ContextAdapterError("SPAR-033 reuse requires exact key match")
    query = _cid(payload.get("query_key_cid"), "reuse_decision.query_key_cid")
    matched = _optional_cid(
        payload.get("matched_transition_cid"), "reuse_decision.matched_transition_cid"
    )
    return {
        "decision": decision,
        "exact_match": exact_match,
        "query_key_cid": query,
        "matched_transition_cid": matched,
        "decision_cid": _optional_cid(
            payload.get("decision_cid"), "reuse_decision.decision_cid"
        ),
    }


def _capability_can_close(value: Any, name: str) -> dict[str, Any] | None:
    payload = _optional_mapping(value, name)
    if payload is None:
        return None
    evidence = _text(payload.get("evidence_class") or "exact_static_fact", "evidence_class")
    can_close = _bool(payload.get("can_close", False), f"{name}.can_close")
    available = _bool(payload.get("available", can_close), f"{name}.available")
    complete = _bool(payload.get("complete", can_close), f"{name}.complete")
    if evidence in _NON_ADMITTING_EVIDENCE and (can_close or complete):
        raise ContextAdapterError(
            f"{name} vector, model, or heuristic evidence cannot admit a route"
        )
    closed = bool(can_close or (available and complete))
    return {
        "can_close": closed,
        "available": available,
        "complete": complete,
        "evidence_class": evidence,
        "capability_cid": _optional_cid(
            payload.get("procedure_cid")
            or payload.get("analysis_cid")
            or payload.get("proof_cid")
            or payload.get("transform_cid")
            or payload.get("capability_cid"),
            f"{name}.capability_cid",
        ),
        "kind": _text(payload.get("kind") or "", f"{name}.kind", empty=True),
    }


def _analysis_capability(value: Any) -> dict[str, Any] | None:
    payload = _capability_can_close(value, "analysis")
    if payload is None:
        return None
    kind = payload["kind"]
    if payload["can_close"]:
        if kind not in DECLARED_ANALYSIS_KINDS:
            raise ContextAdapterError("unsupported analysis kind")
    return payload


def _ranking(value: Any) -> dict[str, Any] | None:
    payload = _optional_mapping(value, "ranking")
    if payload is None:
        return None
    if _bool(payload.get("nomination_only", True), "ranking.nomination_only") is not True:
        raise ContextAdapterError("specialist ranking must remain nomination_only")
    if payload.get("can_close") is True:
        raise ContextAdapterError("specialist ranking cannot close a required residual")
    if payload.get("suppress_residual") is True:
        raise ContextAdapterError("specialist ranking cannot suppress residual routing")
    channel = _text(payload.get("channel") or AnalogChannel.VECTOR.value, "ranking.channel")
    if channel not in DECLARED_RANKING_CHANNELS:
        raise ContextAdapterError(f"unsupported ranking channel {channel!r}")
    return {"channel": channel, "nomination_only": True, "can_close": False}


def _compiler_receipt(value: Any) -> dict[str, Any] | None:
    payload = _optional_mapping(value, "compiler_receipt")
    if payload is None:
        return None
    if payload.get("replaced") is True or payload.get("context_compiler_replaced") is True:
        raise ContextAdapterError("adapter cannot replace ContextCompiler")
    if payload.get("adapter_is_context_compiler") is True:
        raise ContextAdapterError("adapter cannot replace ContextCompiler")
    if payload.get("can_authorize_completion") is True:
        raise ContextAdapterError("compiler_receipt cannot claim can_authorize_completion")
    receipt_cid = _optional_cid(
        payload.get("receipt_cid") or payload.get("compiler_receipt_cid"),
        "compiler_receipt.receipt_cid",
    )
    return {
        "receipt_cid": receipt_cid,
        "context_compiler_remains_authority": True,
    }


def _allowed_effects(values: Any) -> tuple[str, ...]:
    if values in (None, (), []):
        return DEFAULT_ALLOWED_EFFECTS
    ordered = _unique_ordered_text(values, "allowed_effects", limit=MAX_MEMBERS)
    forbidden = set(ordered) & DECLARED_FORBIDDEN_EFFECTS
    if forbidden:
        raise ContextAdapterError(f"forbidden effects are not allowed: {sorted(forbidden)}")
    extra = set(ordered) - DECLARED_ALLOWED_EFFECTS
    if extra:
        raise ContextAdapterError(f"undeclared allowed effect: {sorted(extra)}")
    return ordered


def _reasons(values: Any) -> tuple[str, ...]:
    if values in (None, ()):
        return ()
    return _unique_ordered_text(values, "reasons", limit=MAX_REASONS)


@dataclass(frozen=True, slots=True)
class AffectedSlice:
    """Exact affected source slice bound to a write path and raw-source CID."""

    slice_id: str
    path: str
    symbol: str
    source_cid: str

    interface: ClassVar[str] = AFFECTED_SLICE_INTERFACE
    schema: ClassVar[str] = AFFECTED_SLICE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "slice_id",
            "path",
            "symbol",
            "source_cid",
            "slice_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_id", _text(self.slice_id, "slice_id"))
        object.__setattr__(self, "path", _exact_path(self.path, "path"))
        object.__setattr__(self, "symbol", _text(self.symbol, "symbol"))
        object.__setattr__(self, "source_cid", _cid(self.source_cid, "source_cid"))

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": AFFECTED_SLICE_SCHEMA,
            "interface": AFFECTED_SLICE_INTERFACE,
            "slice_id": self.slice_id,
            "path": self.path,
            "symbol": self.symbol,
            "source_cid": self.source_cid,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def slice_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["slice_cid"] = self.slice_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AffectedSlice":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("slice_cid")
        if payload.pop("schema") != AFFECTED_SLICE_SCHEMA:
            raise ContextAdapterError("unsupported AffectedSlice schema")
        if payload.pop("interface") != AFFECTED_SLICE_INTERFACE:
            raise ContextAdapterError("unsupported AffectedSlice interface")
        result = cls(**payload)
        _verify_cid(claimed, result.slice_cid, "slice_cid")
        return result


def compile_affected_slice(**fields: Any) -> AffectedSlice:
    return AffectedSlice(**fields)


@dataclass(frozen=True, slots=True)
class NamedUnresolvedQuestion:
    """One typed unresolved residual question bound to an exact slice."""

    question_id: str
    kind: str
    statement: str
    slice_id: str
    contract_ids: Sequence[str] = ()
    evidence_cids: Sequence[str] = ()
    typed: bool = True
    unresolved: bool = True

    interface: ClassVar[str] = NAMED_UNRESOLVED_QUESTION_INTERFACE
    schema: ClassVar[str] = NAMED_UNRESOLVED_QUESTION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "question_id",
            "kind",
            "statement",
            "slice_id",
            "contract_ids",
            "evidence_cids",
            "typed",
            "unresolved",
            "question_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _text(self.question_id, "question_id"))
        object.__setattr__(self, "kind", _question_kind(self.kind))
        object.__setattr__(self, "statement", _text(self.statement, "statement"))
        object.__setattr__(self, "slice_id", _text(self.slice_id, "slice_id"))
        object.__setattr__(
            self,
            "contract_ids",
            _unique_ordered_text(self.contract_ids, "contract_ids", limit=MAX_CONTRACTS),
        )
        object.__setattr__(
            self,
            "evidence_cids",
            _cids(self.evidence_cids, "evidence_cids", required=False),
        )
        if _bool(self.typed, "typed") is not True:
            raise ContextAdapterError("residual question must remain typed")
        if _bool(self.unresolved, "unresolved") is not True:
            raise ContextAdapterError("named residual question must remain unresolved")
        object.__setattr__(self, "typed", True)
        object.__setattr__(self, "unresolved", True)

    @property
    def evidence_fingerprint(self) -> str:
        payload = {
            "question_id": self.question_id,
            "evidence_cids": list(self.evidence_cids),
        }
        return cid_for_dag_json(payload)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": NAMED_UNRESOLVED_QUESTION_SCHEMA,
            "interface": NAMED_UNRESOLVED_QUESTION_INTERFACE,
            "question_id": self.question_id,
            "kind": self.kind,
            "statement": self.statement,
            "slice_id": self.slice_id,
            "contract_ids": list(self.contract_ids),
            "evidence_cids": list(self.evidence_cids),
            "typed": True,
            "unresolved": True,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def question_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["question_cid"] = self.question_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NamedUnresolvedQuestion":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("question_cid")
        if payload.pop("schema") != NAMED_UNRESOLVED_QUESTION_SCHEMA:
            raise ContextAdapterError("unsupported NamedUnresolvedQuestion schema")
        if payload.pop("interface") != NAMED_UNRESOLVED_QUESTION_INTERFACE:
            raise ContextAdapterError("unsupported NamedUnresolvedQuestion interface")
        if payload.pop("typed") is not True:
            raise ContextAdapterError("residual question must remain typed")
        if payload.pop("unresolved") is not True:
            raise ContextAdapterError("named residual question must remain unresolved")
        result = cls(**payload)
        _verify_cid(claimed, result.question_cid, "question_cid")
        return result


def compile_named_question(**fields: Any) -> NamedUnresolvedQuestion:
    return NamedUnresolvedQuestion(**fields)


def _compile_slices(
    values: Any,
    *,
    write_paths: Sequence[str],
    source_cids: Sequence[str],
) -> tuple[AffectedSlice, ...]:
    items = _mapping_sequence(values, "slices", limit=MAX_SLICES)
    slices: list[AffectedSlice] = []
    seen: set[str] = set()
    write = set(write_paths)
    sources = set(source_cids)
    for item in items:
        extra = {key: item[key] for key in ("slice_id", "path", "symbol", "source_cid") if key in item}
        compiled = AffectedSlice.from_dict(item) if "slice_cid" in item else AffectedSlice(**extra)
        if compiled.slice_id in seen:
            raise ContextAdapterError("slices must not contain duplicate slice_id")
        seen.add(compiled.slice_id)
        if compiled.path not in write:
            raise ContextAdapterError("slice path must be an owned write path")
        if compiled.source_cid not in sources:
            raise ContextAdapterError("slice source_cid must be a declared raw source")
        slices.append(compiled)
    return tuple(slices)


def _compile_questions(
    values: Any,
    *,
    slices: Sequence[AffectedSlice],
    contract_ids: Sequence[str],
) -> tuple[NamedUnresolvedQuestion, ...]:
    items = _mapping_sequence(values, "questions", limit=MAX_QUESTIONS)
    questions: list[NamedUnresolvedQuestion] = []
    seen: set[str] = set()
    slice_ids = {item.slice_id for item in slices}
    known_contracts = set(contract_ids)
    for item in items:
        compiled = (
            NamedUnresolvedQuestion.from_dict(item)
            if "question_cid" in item
            else NamedUnresolvedQuestion(
                question_id=item.get("question_id"),
                kind=item.get("kind"),
                statement=item.get("statement"),
                slice_id=item.get("slice_id"),
                contract_ids=item.get("contract_ids") or (),
                evidence_cids=item.get("evidence_cids") or (),
                typed=item.get("typed", True),
                unresolved=item.get("unresolved", True),
            )
        )
        if compiled.question_id in seen:
            raise ContextAdapterError("questions must not contain duplicate question_id")
        seen.add(compiled.question_id)
        if compiled.slice_id not in slice_ids:
            raise ContextAdapterError("question slice_id must refer to an affected slice")
        unknown = set(compiled.contract_ids) - known_contracts
        if unknown:
            raise ContextAdapterError("question contract_ids must refer to packed contracts")
        questions.append(compiled)
    return tuple(questions)


def _compile_contracts(values: Any) -> tuple[dict[str, Any], ...]:
    items = _mapping_sequence(values, "contracts", limit=MAX_CONTRACTS)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        contract_id = _text(item.get("contract_id"), "contract_id")
        if contract_id in seen:
            raise ContextAdapterError("contracts must not contain duplicate contract_id")
        seen.add(contract_id)
        ordered.append(
            {
                "contract_id": contract_id,
                "contract_cid": _cid(item.get("contract_cid"), "contract_cid"),
                "kind": _text(item.get("kind") or "boundary", "contract.kind"),
            }
        )
    return tuple(ordered)


def _compile_counterexamples(values: Any) -> tuple[dict[str, Any], ...]:
    items = _mapping_sequence(values, "counterexamples", limit=MAX_COUNTEREXAMPLES)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        cex_id = _text(item.get("counterexample_id"), "counterexample_id")
        if cex_id in seen:
            raise ContextAdapterError("counterexamples must not contain duplicate ids")
        seen.add(cex_id)
        replayed = _bool(item.get("replayed", False), "counterexample.replayed")
        authority = _text(
            item.get("source_authority") or "specification",
            "counterexample.source_authority",
        )
        if authority in _NON_ADMITTING_EVIDENCE:
            raise ContextAdapterError(
                "vector, model, or heuristic evidence cannot admit a counterexample"
            )
        ordered.append(
            {
                "counterexample_id": cex_id,
                "evidence_cid": _cid(item.get("evidence_cid"), "counterexample.evidence_cid"),
                "kind": _text(item.get("kind") or "boundary", "counterexample.kind"),
                "replayed": replayed,
                "source_authority": authority,
            }
        )
    return tuple(ordered)


def _compile_evidence(values: Any) -> tuple[dict[str, Any], ...]:
    items = _mapping_sequence(values, "evidence", limit=MAX_MEMBERS)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        evidence_id = _text(item.get("evidence_id"), "evidence_id")
        if evidence_id in seen:
            raise ContextAdapterError("evidence must not contain duplicate evidence_id")
        seen.add(evidence_id)
        evidence_class = _text(
            item.get("evidence_class") or "exact_static_fact", "evidence_class"
        )
        ordered.append(
            {
                "evidence_id": evidence_id,
                "evidence_cid": _cid(item.get("evidence_cid"), "evidence_cid"),
                "evidence_class": evidence_class,
            }
        )
    return tuple(ordered)


def _compile_analogs(values: Any) -> tuple[dict[str, Any], ...]:
    items = _mapping_sequence(values, "analogous_refactors", limit=MAX_ANALOGS)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        analog_id = _text(item.get("analog_id"), "analog_id")
        if analog_id in seen:
            raise ContextAdapterError("analogous_refactors must not contain duplicate ids")
        seen.add(analog_id)
        channel = _text(item.get("channel") or AnalogChannel.VECTOR.value, "analog.channel")
        if channel == "exact":
            raise ContextAdapterError("analogous refactors cannot claim exact reuse")
        if channel not in DECLARED_ANALOG_CHANNELS:
            raise ContextAdapterError(f"unsupported analog channel {channel!r}")
        evidence_class = _text(
            item.get("evidence_class") or "vector_candidate", "analog.evidence_class"
        )
        ordered.append(
            {
                "analog_id": analog_id,
                "transition_cid": _cid(item.get("transition_cid"), "analog.transition_cid"),
                "channel": channel,
                "evidence_class": evidence_class,
                "context_only": True,
            }
        )
    return tuple(ordered)


def _prior_failures(values: Any) -> tuple[dict[str, Any], ...]:
    items = _mapping_sequence(values, "prior_failures", limit=MAX_FAILURES)
    ordered: list[dict[str, Any]] = []
    for item in items:
        ordered.append(
            {
                "question_id": _text(item.get("question_id"), "prior_failure.question_id"),
                "evidence_fingerprint": _cid(
                    item.get("evidence_fingerprint") or item.get("evidence_cid"),
                    "prior_failure.evidence_fingerprint",
                ),
            }
        )
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class SemanticRefactorContext:
    """Bounded semantic context for residual routing. Nomination-only."""

    tree_id: str
    packet_cid: str
    query_key_cid: str
    slices: Sequence[AffectedSlice]
    questions: Sequence[NamedUnresolvedQuestion]
    contracts: Sequence[Mapping[str, Any]]
    counterexamples: Sequence[Mapping[str, Any]]
    evidence: Sequence[Mapping[str, Any]]
    analogous_refactors: Sequence[Mapping[str, Any]]
    allowed_effects: Sequence[str]
    raw_source_cids: Sequence[str]
    write_paths: Sequence[str]
    validation_commands: Sequence[str]
    compiler_receipt_cid: str = ""
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    raw_source_required: bool = True
    independent_validation_required: bool = True
    context_compiler_remains_authority: bool = True
    network: str = NETWORK_DENY

    interface: ClassVar[str] = SEMANTIC_REFACTOR_CONTEXT_INTERFACE
    schema: ClassVar[str] = SEMANTIC_REFACTOR_CONTEXT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "query_key_cid",
            "slices",
            "questions",
            "contracts",
            "counterexamples",
            "evidence",
            "analogous_refactors",
            "allowed_effects",
            "raw_source_cids",
            "write_paths",
            "validation_commands",
            "compiler_receipt_cid",
            "analyzer_id",
            "adapter_is_nomination_only",
            "raw_source_required",
            "independent_validation_required",
            "context_compiler_remains_authority",
            "network",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_replace_compiler",
            "can_weaken_validation",
            "context_is_authority",
            "context_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(self, "query_key_cid", _cid(self.query_key_cid, "query_key_cid"))
        object.__setattr__(
            self,
            "compiler_receipt_cid",
            _optional_cid(self.compiler_receipt_cid, "compiler_receipt_cid"),
        )
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(self, "questions", tuple(self.questions))
        object.__setattr__(self, "contracts", tuple(dict(item) for item in self.contracts))
        object.__setattr__(
            self, "counterexamples", tuple(dict(item) for item in self.counterexamples)
        )
        object.__setattr__(self, "evidence", tuple(dict(item) for item in self.evidence))
        object.__setattr__(
            self,
            "analogous_refactors",
            tuple(dict(item) for item in self.analogous_refactors),
        )
        object.__setattr__(self, "allowed_effects", _allowed_effects(self.allowed_effects))
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
            raise ContextAdapterError("context analyzer_id must remain SPAR-035")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise ContextAdapterError("adapter must remain nomination_only")
        if _bool(self.raw_source_required, "raw_source_required") is not True:
            raise ContextAdapterError("raw_source_required cannot be disabled")
        if _bool(
            self.independent_validation_required, "independent_validation_required"
        ) is not True:
            raise ContextAdapterError("independent validation cannot be weakened")
        if _bool(
            self.context_compiler_remains_authority,
            "context_compiler_remains_authority",
        ) is not True:
            raise ContextAdapterError("ContextCompiler remains the context authority")
        object.__setattr__(self, "network", _network_value(self.network))
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "raw_source_required", True)
        object.__setattr__(self, "independent_validation_required", True)
        object.__setattr__(self, "context_compiler_remains_authority", True)

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
    def can_replace_compiler(self) -> bool:
        return False

    @property
    def can_weaken_validation(self) -> bool:
        return False

    @property
    def context_is_authority(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": SEMANTIC_REFACTOR_CONTEXT_SCHEMA,
            "interface": SEMANTIC_REFACTOR_CONTEXT_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "query_key_cid": self.query_key_cid,
            "slices": [item.to_dict() for item in self.slices],
            "questions": [item.to_dict() for item in self.questions],
            "contracts": [dict(item) for item in self.contracts],
            "counterexamples": [dict(item) for item in self.counterexamples],
            "evidence": [dict(item) for item in self.evidence],
            "analogous_refactors": [dict(item) for item in self.analogous_refactors],
            "allowed_effects": list(self.allowed_effects),
            "raw_source_cids": list(self.raw_source_cids),
            "write_paths": list(self.write_paths),
            "validation_commands": list(self.validation_commands),
            "compiler_receipt_cid": self.compiler_receipt_cid,
            "analyzer_id": ANALYZER_ID,
            "adapter_is_nomination_only": True,
            "raw_source_required": True,
            "independent_validation_required": True,
            "context_compiler_remains_authority": True,
            "network": NETWORK_DENY,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_replace_compiler": False,
            "can_weaken_validation": False,
            "context_is_authority": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def context_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["context_cid"] = self.context_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticRefactorContext":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("context_cid")
        if payload.pop("schema") != SEMANTIC_REFACTOR_CONTEXT_SCHEMA:
            raise ContextAdapterError("unsupported SemanticRefactorContext schema")
        if payload.pop("interface") != SEMANTIC_REFACTOR_CONTEXT_INTERFACE:
            raise ContextAdapterError("unsupported SemanticRefactorContext interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ContextAdapterError("adapter must remain nomination_only")
        if payload.pop("raw_source_required") is not True:
            raise ContextAdapterError("raw_source_required cannot be disabled")
        if payload.pop("independent_validation_required") is not True:
            raise ContextAdapterError("independent validation cannot be weakened")
        if payload.pop("context_compiler_remains_authority") is not True:
            raise ContextAdapterError("ContextCompiler remains the context authority")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise ContextAdapterError("context analyzer_id must remain SPAR-035")
        payload["network"] = _network_value(payload.get("network"))
        payload["slices"] = tuple(
            AffectedSlice.from_dict(item) if isinstance(item, Mapping) else item
            for item in payload.get("slices") or ()
        )
        payload["questions"] = tuple(
            NamedUnresolvedQuestion.from_dict(item) if isinstance(item, Mapping) else item
            for item in payload.get("questions") or ()
        )
        result = cls(**payload)
        _verify_cid(claimed, result.context_cid, "context_cid")
        return result


@dataclass(frozen=True, slots=True)
class ResidualModelRoute:
    """Deterministic residual/no-model route. Nomination-only."""

    route_kind: str
    status: str
    residual_question_id: str = ""
    residual_question_cid: str = ""
    general_model_invoked: bool = False
    ranking_applied: bool = False
    retry_blocked: bool = False
    no_model_step: str = ""
    reasons: Sequence[str] = ()
    typed_terminal: bool = False
    model_output_is_proposal_only: bool = True
    independent_validation_required: bool = True

    interface: ClassVar[str] = RESIDUAL_MODEL_ROUTE_INTERFACE
    schema: ClassVar[str] = RESIDUAL_MODEL_ROUTE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "route_kind",
            "status",
            "residual_question_id",
            "residual_question_cid",
            "general_model_invoked",
            "ranking_applied",
            "retry_blocked",
            "no_model_step",
            "reasons",
            "typed_terminal",
            "model_output_is_proposal_only",
            "independent_validation_required",
            "can_weaken_validation",
            "can_authorize_completion",
            "route_cid",
        }
    )

    def __post_init__(self) -> None:
        kind = _route_kind(self.route_kind)
        status = _route_status(self.status)
        object.__setattr__(self, "route_kind", kind)
        object.__setattr__(self, "status", status)
        question_id = _text(
            self.residual_question_id, "residual_question_id", empty=True
        )
        question_cid = _optional_cid(
            self.residual_question_cid, "residual_question_cid"
        )
        invoked = _bool(self.general_model_invoked, "general_model_invoked")
        retry_blocked = _bool(self.retry_blocked, "retry_blocked")
        ranking_applied = _bool(self.ranking_applied, "ranking_applied")
        typed_terminal = _bool(self.typed_terminal, "typed_terminal")
        object.__setattr__(self, "residual_question_id", question_id)
        object.__setattr__(self, "residual_question_cid", question_cid)
        object.__setattr__(self, "general_model_invoked", invoked)
        object.__setattr__(self, "retry_blocked", retry_blocked)
        object.__setattr__(self, "ranking_applied", ranking_applied)
        object.__setattr__(self, "typed_terminal", typed_terminal)
        object.__setattr__(
            self, "no_model_step", _text(self.no_model_step, "no_model_step", empty=True)
        )
        object.__setattr__(self, "reasons", _reasons(self.reasons))
        if _bool(self.model_output_is_proposal_only, "model_output_is_proposal_only") is not True:
            raise ContextAdapterError("model output must remain proposal-only")
        if _bool(
            self.independent_validation_required, "independent_validation_required"
        ) is not True:
            raise ContextAdapterError("independent validation cannot be weakened")
        object.__setattr__(self, "model_output_is_proposal_only", True)
        object.__setattr__(self, "independent_validation_required", True)
        if invoked and kind != RouteKind.RESIDUAL_GENERAL_MODEL.value:
            raise ContextAdapterError("general model may be invoked only for one residual")
        if kind == RouteKind.RESIDUAL_GENERAL_MODEL.value:
            if not question_id:
                raise ContextAdapterError("residual_general_model requires one named question")
            if invoked is not True:
                raise ContextAdapterError("residual_general_model must invoke the general model")
            if typed_terminal:
                raise ContextAdapterError("residual_general_model is not a typed terminal")
        if kind == RouteKind.BLOCKED.value:
            if invoked:
                raise ContextAdapterError("blocked route cannot invoke a general model")
            if not typed_terminal:
                raise ContextAdapterError("blocked route is a typed terminal")
        if kind in NO_MODEL_STEPS or kind == RouteKind.NO_MODEL.value:
            if invoked:
                raise ContextAdapterError("no-model route cannot invoke a general model")
            if question_id:
                raise ContextAdapterError("no-model route cannot bind a residual question")
        if kind == RouteKind.SPECIALIST_RANKING.value:
            if invoked:
                raise ContextAdapterError("specialist ranking cannot invoke a general model")
            if ranking_applied is not True:
                raise ContextAdapterError("specialist ranking route requires ranking_applied")

    @property
    def can_weaken_validation(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": RESIDUAL_MODEL_ROUTE_SCHEMA,
            "interface": RESIDUAL_MODEL_ROUTE_INTERFACE,
            "route_kind": self.route_kind,
            "status": self.status,
            "residual_question_id": self.residual_question_id,
            "residual_question_cid": self.residual_question_cid,
            "general_model_invoked": self.general_model_invoked,
            "ranking_applied": self.ranking_applied,
            "retry_blocked": self.retry_blocked,
            "no_model_step": self.no_model_step,
            "reasons": list(self.reasons),
            "typed_terminal": self.typed_terminal,
            "model_output_is_proposal_only": True,
            "independent_validation_required": True,
            "can_weaken_validation": False,
            "can_authorize_completion": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def route_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["route_cid"] = self.route_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResidualModelRoute":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("route_cid")
        if payload.pop("schema") != RESIDUAL_MODEL_ROUTE_SCHEMA:
            raise ContextAdapterError("unsupported ResidualModelRoute schema")
        if payload.pop("interface") != RESIDUAL_MODEL_ROUTE_INTERFACE:
            raise ContextAdapterError("unsupported ResidualModelRoute interface")
        if payload.pop("can_weaken_validation") is not False:
            raise ContextAdapterError("route cannot claim can_weaken_validation")
        if payload.pop("can_authorize_completion") is not False:
            raise ContextAdapterError("route cannot claim can_authorize_completion")
        if payload.pop("model_output_is_proposal_only") is not True:
            raise ContextAdapterError("model output must remain proposal-only")
        if payload.pop("independent_validation_required") is not True:
            raise ContextAdapterError("independent validation cannot be weakened")
        result = cls(**payload)
        _verify_cid(claimed, result.route_cid, "route_cid")
        return result


def _identical_retry(
    question: NamedUnresolvedQuestion,
    prior: Sequence[Mapping[str, Any]],
) -> bool:
    fingerprint = question.evidence_fingerprint
    for item in prior:
        if item["question_id"] != question.question_id:
            continue
        if item["evidence_fingerprint"] == fingerprint:
            return True
    return False


def _select_route(
    *,
    reuse: Mapping[str, Any],
    procedure: Mapping[str, Any] | None,
    analysis: Mapping[str, Any] | None,
    proof: Mapping[str, Any] | None,
    transform: Mapping[str, Any] | None,
    ranking: Mapping[str, Any] | None,
    questions: Sequence[NamedUnresolvedQuestion],
    prior: Sequence[Mapping[str, Any]],
) -> ResidualModelRoute:
    ranking_applied = ranking is not None
    if reuse["decision"] == "reuse":
        return ResidualModelRoute(
            route_kind=RouteKind.EXACT_REUSE.value,
            status=RouteStatus.REUSED.value,
            no_model_step=RouteKind.EXACT_REUSE.value,
            ranking_applied=ranking_applied,
        )
    if procedure is not None and procedure["can_close"]:
        return ResidualModelRoute(
            route_kind=RouteKind.VERIFIED_PROCEDURE.value,
            status=RouteStatus.NO_MODEL.value,
            no_model_step=RouteKind.VERIFIED_PROCEDURE.value,
            ranking_applied=ranking_applied,
        )
    if analysis is not None and analysis["can_close"]:
        return ResidualModelRoute(
            route_kind=RouteKind.DETERMINISTIC_ANALYSIS.value,
            status=RouteStatus.NO_MODEL.value,
            no_model_step=RouteKind.DETERMINISTIC_ANALYSIS.value,
            ranking_applied=ranking_applied,
        )
    if proof is not None and proof["can_close"]:
        return ResidualModelRoute(
            route_kind=RouteKind.PROOF_SEARCH.value,
            status=RouteStatus.NO_MODEL.value,
            no_model_step=RouteKind.PROOF_SEARCH.value,
            ranking_applied=ranking_applied,
        )
    if transform is not None and transform["can_close"]:
        return ResidualModelRoute(
            route_kind=RouteKind.DETERMINISTIC_TRANSFORM.value,
            status=RouteStatus.NO_MODEL.value,
            no_model_step=RouteKind.DETERMINISTIC_TRANSFORM.value,
            ranking_applied=ranking_applied,
        )
    unresolved = tuple(item for item in questions if item.unresolved)
    if len(unresolved) > MAX_GENERAL_MODEL_QUESTIONS:
        return ResidualModelRoute(
            route_kind=RouteKind.BLOCKED.value,
            status=RouteStatus.BLOCKED.value,
            ranking_applied=ranking_applied,
            typed_terminal=True,
            reasons=("multiple_residuals",),
        )
    if len(unresolved) == 1:
        question = unresolved[0]
        if _identical_retry(question, prior):
            return ResidualModelRoute(
                route_kind=RouteKind.BLOCKED.value,
                status=RouteStatus.BLOCKED.value,
                residual_question_id=question.question_id,
                residual_question_cid=question.question_cid,
                ranking_applied=ranking_applied,
                retry_blocked=True,
                typed_terminal=True,
                reasons=("identical_failure_without_new_evidence",),
            )
        return ResidualModelRoute(
            route_kind=RouteKind.RESIDUAL_GENERAL_MODEL.value,
            status=RouteStatus.NOMINATED.value,
            residual_question_id=question.question_id,
            residual_question_cid=question.question_cid,
            general_model_invoked=True,
            ranking_applied=ranking_applied,
            reasons=("one_named_unresolved_residual",),
        )
    if ranking_applied:
        return ResidualModelRoute(
            route_kind=RouteKind.SPECIALIST_RANKING.value,
            status=RouteStatus.NOMINATED.value,
            ranking_applied=True,
            reasons=("specialist_ranking_nomination_only",),
        )
    return ResidualModelRoute(
        route_kind=RouteKind.BLOCKED.value,
        status=RouteStatus.BLOCKED.value,
        ranking_applied=False,
        typed_terminal=True,
        reasons=("missing_named_residual",),
    )


@dataclass(frozen=True, slots=True)
class ResidualRouteReceipt:
    """Content-addressed SPAR-035 route receipt. Nomination-only."""

    tree_id: str
    context: SemanticRefactorContext
    route: ResidualModelRoute
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    mutated: bool = False
    deterministic: bool = True
    network: str = NETWORK_DENY

    interface: ClassVar[str] = RESIDUAL_ROUTE_RECEIPT_INTERFACE
    schema: ClassVar[str] = RESIDUAL_ROUTE_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "context_cid",
            "route_cid",
            "route_kind",
            "status",
            "residual_question_id",
            "general_model_invoked",
            "retry_blocked",
            "typed_terminal",
            "model_output_is_proposal_only",
            "independent_validation_required",
            "context_compiler_remains_authority",
            "adapter_is_nomination_only",
            "mutated",
            "deterministic",
            "network",
            "analyzer_id",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_weaken_validation",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.context, SemanticRefactorContext):
            raise ContextAdapterError("context must be a SemanticRefactorContext")
        if not isinstance(self.route, ResidualModelRoute):
            raise ContextAdapterError("route must be a ResidualModelRoute")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        if self.tree_id != self.context.tree_id:
            raise ContextAdapterError("receipt tree_id does not match context")
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise ContextAdapterError("receipt analyzer_id must remain SPAR-035")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise ContextAdapterError("adapter must remain nomination_only")
        if _bool(self.mutated, "mutated") is not False:
            raise ContextAdapterError("dry-run/receipt must not mutate")
        if _bool(self.deterministic, "deterministic") is not True:
            raise ContextAdapterError("route receipt must remain deterministic")
        object.__setattr__(self, "network", _network_value(self.network))
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

    @property
    def can_weaken_validation(self) -> bool:
        return False

    @property
    def route_kind(self) -> str:
        return self.route.route_kind

    @property
    def status(self) -> str:
        return self.route.status

    @property
    def residual_question_id(self) -> str:
        return self.route.residual_question_id

    @property
    def general_model_invoked(self) -> bool:
        return self.route.general_model_invoked

    @property
    def retry_blocked(self) -> bool:
        return self.route.retry_blocked

    @property
    def typed_terminal(self) -> bool:
        return self.route.typed_terminal

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": RESIDUAL_ROUTE_RECEIPT_SCHEMA,
            "interface": RESIDUAL_ROUTE_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "context_cid": self.context.context_cid,
            "route_cid": self.route.route_cid,
            "route_kind": self.route.route_kind,
            "status": self.route.status,
            "residual_question_id": self.route.residual_question_id,
            "general_model_invoked": self.route.general_model_invoked,
            "retry_blocked": self.route.retry_blocked,
            "typed_terminal": self.route.typed_terminal,
            "model_output_is_proposal_only": True,
            "independent_validation_required": True,
            "context_compiler_remains_authority": True,
            "adapter_is_nomination_only": True,
            "mutated": False,
            "deterministic": True,
            "network": NETWORK_DENY,
            "analyzer_id": ANALYZER_ID,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_weaken_validation": False,
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
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        context: SemanticRefactorContext,
        route: ResidualModelRoute,
    ) -> "ResidualRouteReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != RESIDUAL_ROUTE_RECEIPT_SCHEMA:
            raise ContextAdapterError("unsupported ResidualRouteReceipt schema")
        if payload.pop("interface") != RESIDUAL_ROUTE_RECEIPT_INTERFACE:
            raise ContextAdapterError("unsupported ResidualRouteReceipt interface")
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("adapter_is_nomination_only") is not True:
            raise ContextAdapterError("adapter must remain nomination_only")
        if payload.pop("model_output_is_proposal_only") is not True:
            raise ContextAdapterError("model output must remain proposal-only")
        if payload.pop("independent_validation_required") is not True:
            raise ContextAdapterError("independent validation cannot be weakened")
        if payload.pop("context_compiler_remains_authority") is not True:
            raise ContextAdapterError("ContextCompiler remains the context authority")
        if payload.pop("mutated") is not False:
            raise ContextAdapterError("dry-run/receipt must not mutate")
        if payload.pop("deterministic") is not True:
            raise ContextAdapterError("route receipt must remain deterministic")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise ContextAdapterError("receipt analyzer_id must remain SPAR-035")
        if payload.pop("context_cid") != context.context_cid:
            raise ContextAdapterError("receipt context_cid does not match context")
        if payload.pop("route_cid") != route.route_cid:
            raise ContextAdapterError("receipt route_cid does not match route")
        payload.pop("route_kind")
        payload.pop("status")
        payload.pop("residual_question_id")
        payload.pop("general_model_invoked")
        payload.pop("retry_blocked")
        payload.pop("typed_terminal")
        payload.pop("network")
        result = cls(tree_id=payload["tree_id"], context=context, route=route)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


def compile_context_receipt(
    context: SemanticRefactorContext,
    route: ResidualModelRoute,
) -> ResidualRouteReceipt:
    return ResidualRouteReceipt(
        tree_id=context.tree_id, context=context, route=route
    )


def adapt_semantic_refactor_context(
    *,
    packet: Mapping[str, Any] | None,
    reuse_decision: Mapping[str, Any] | None,
    questions: Sequence[Mapping[str, Any] | NamedUnresolvedQuestion] = (),
    slices: Sequence[Mapping[str, Any] | AffectedSlice] = (),
    contracts: Sequence[Mapping[str, Any]] = (),
    counterexamples: Sequence[Mapping[str, Any]] = (),
    evidence: Sequence[Mapping[str, Any]] = (),
    analogous_refactors: Sequence[Mapping[str, Any]] = (),
    allowed_effects: Sequence[str] | None = None,
    procedure: Mapping[str, Any] | None = None,
    analysis: Mapping[str, Any] | None = None,
    proof: Mapping[str, Any] | None = None,
    transform: Mapping[str, Any] | None = None,
    ranking: Mapping[str, Any] | None = None,
    compiler_receipt: Mapping[str, Any] | None = None,
    prior_failures: Sequence[Mapping[str, Any]] = (),
    vector_evidence: Mapping[str, Any] | None = None,
    network: str = NETWORK_DENY,
    skip_validation: bool = False,
    weaken_validation: bool = False,
) -> ResidualRouteReceipt:
    """Pack bounded context and select one residual/no-model route."""

    if packet is None:
        raise ContextAdapterError("missing SPAR-019 packet")
    if reuse_decision is None:
        raise ContextAdapterError("missing SPAR-033 reuse_decision")
    packet_map = _as_mapping(packet, "packet")
    _network_value(network)
    if skip_validation is True or weaken_validation is True:
        raise ContextAdapterError("model output cannot weaken validation")
    if vector_evidence is not None:
        vector_map = _as_mapping(vector_evidence, "vector_evidence")
        _text(
            vector_map.get("evidence_class") or "vector_candidate",
            "vector_evidence.evidence_class",
        )
        if vector_map.get("suppress_raw_source") is True:
            raise ContextAdapterError("vectors cannot suppress raw-source fallback")
        if vector_map.get("skip_full_suite") is True:
            raise ContextAdapterError("vectors cannot skip full-suite validation")
        if vector_map.get("weaken_validation") is True:
            raise ContextAdapterError("model output cannot weaken validation")
        if vector_map.get("admit_route") is True:
            raise ContextAdapterError(
                "vector, model, or heuristic evidence cannot admit a route"
            )

    tree_id = _packet_tree(packet_map)
    write_paths = _packet_write_paths(packet_map)
    source_cids = _packet_source_cids(packet_map)
    validation_commands = _packet_validation_commands(packet_map)
    reuse = _reuse_decision(reuse_decision)
    compiled_contracts = _compile_contracts(contracts)
    compiled_slices = _compile_slices(
        slices, write_paths=write_paths, source_cids=source_cids
    )
    compiled_questions = _compile_questions(
        questions,
        slices=compiled_slices,
        contract_ids=tuple(item["contract_id"] for item in compiled_contracts),
    )
    compiled_counterexamples = _compile_counterexamples(counterexamples)
    compiled_evidence = _compile_evidence(evidence)
    compiled_analogs = _compile_analogs(analogous_refactors)
    compiler = _compiler_receipt(compiler_receipt)
    procedure_cap = _capability_can_close(procedure, "procedure")
    analysis_cap = _analysis_capability(analysis)
    proof_cap = _capability_can_close(proof, "proof")
    transform_cap = _capability_can_close(transform, "transform")
    ranking_cap = _ranking(ranking)
    prior = _prior_failures(prior_failures)

    context = SemanticRefactorContext(
        tree_id=tree_id,
        packet_cid=_packet_cid(packet_map),
        query_key_cid=reuse["query_key_cid"],
        slices=compiled_slices,
        questions=compiled_questions,
        contracts=compiled_contracts,
        counterexamples=compiled_counterexamples,
        evidence=compiled_evidence,
        analogous_refactors=compiled_analogs,
        allowed_effects=_allowed_effects(allowed_effects),
        raw_source_cids=source_cids,
        write_paths=write_paths,
        validation_commands=validation_commands,
        compiler_receipt_cid="" if compiler is None else compiler["receipt_cid"],
    )
    route = _select_route(
        reuse=reuse,
        procedure=procedure_cap,
        analysis=analysis_cap,
        proof=proof_cap,
        transform=transform_cap,
        ranking=ranking_cap,
        questions=compiled_questions,
        prior=prior,
    )
    return compile_context_receipt(context, route)


def dry_run_context_route(**fields: Any) -> ResidualRouteReceipt:
    """Deterministic dry-run alias; never mutates."""

    return adapt_semantic_refactor_context(**fields)


def encode_canonical_context(context: SemanticRefactorContext) -> dict[str, Any]:
    return context.to_dict()


def decode_canonical_context(payload: Mapping[str, Any]) -> SemanticRefactorContext:
    return SemanticRefactorContext.from_dict(payload)


def encode_canonical_route(route: ResidualModelRoute) -> dict[str, Any]:
    return route.to_dict()


def decode_canonical_route(payload: Mapping[str, Any]) -> ResidualModelRoute:
    return ResidualModelRoute.from_dict(payload)


def encode_canonical_receipt(receipt: ResidualRouteReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
    *,
    context: SemanticRefactorContext,
    route: ResidualModelRoute,
) -> ResidualRouteReceipt:
    return ResidualRouteReceipt.from_dict(payload, context=context, route=route)


class SemanticRefactorContextAdapter:
    """Nomination-only adapter over ContextCompiler residual routing."""

    interface: ClassVar[str] = SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE
    schema: ClassVar[str] = SEMANTIC_REFACTOR_CONTEXT_ADAPTER_SCHEMA

    def adapt(self, **fields: Any) -> ResidualRouteReceipt:
        return adapt_semantic_refactor_context(**fields)

    def dry_run(self, **fields: Any) -> ResidualRouteReceipt:
        return dry_run_context_route(**fields)

    def receipt(
        self,
        context: SemanticRefactorContext,
        route: ResidualModelRoute,
    ) -> ResidualRouteReceipt:
        return compile_context_receipt(context, route)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise ContextAdapterError(
            f"context adapter must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_IS_NOMINATION_ONLY",
    "AFFECTED_SLICE_INTERFACE",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "CONTEXT_CAN_AUTHORIZE_COMPLETION",
    "CONTEXT_CAN_AUTHORIZE_TRANSITION",
    "CONTEXT_CAN_CREATE_AUTHORITY",
    "CONTEXT_CAN_REPLACE_COMPILER",
    "CONTEXT_CAN_WEAKEN_VALIDATION",
    "CONTEXT_COMPILER_AUTHORITY",
    "CONTEXT_COMPILER_REMAINS_AUTHORITY",
    "CONTEXT_CONTRACT_VERSION",
    "DECLARED_ALLOWED_EFFECTS",
    "DECLARED_QUESTION_KINDS",
    "DECLARED_ROUTE_KINDS",
    "DECLARED_ROUTE_STATUSES",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "FORBIDDEN_CONTEXT_NAMES",
    "GOAL_ID",
    "IDENTICAL_FAILURE_RETRY_WITHOUT_EVIDENCE",
    "IDENTITY_EXCLUDED_FIELDS",
    "INDEPENDENT_VALIDATION_REQUIRED",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MAX_GENERAL_MODEL_QUESTIONS",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "MODEL_ROUTE_CLASS",
    "NAMED_UNRESOLVED_QUESTION_INTERFACE",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "NO_MODEL_STEPS",
    "ONE_RESIDUAL_GENERAL_MODEL_QUESTION",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "RESIDUAL_MODEL_ROUTE_INTERFACE",
    "RESIDUAL_ROUTE_RECEIPT_INTERFACE",
    "SEMANTIC_REFACTOR_CONTEXT_ADAPTER_INTERFACE",
    "SEMANTIC_REFACTOR_CONTEXT_INTERFACE",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "AffectedSlice",
    "AnalogChannel",
    "ContextAdapterError",
    "NamedUnresolvedQuestion",
    "QuestionKind",
    "ResidualModelRoute",
    "ResidualRouteReceipt",
    "RouteKind",
    "RouteStatus",
    "SemanticRefactorContext",
    "SemanticRefactorContextAdapter",
    "adapt_semantic_refactor_context",
    "assert_not_competing_capsule_family",
    "compile_affected_slice",
    "compile_context_receipt",
    "compile_named_question",
    "context_adapter_cid_profile",
    "context_adapter_descriptor",
    "decode_canonical_context",
    "decode_canonical_receipt",
    "decode_canonical_route",
    "dry_run_context_route",
    "encode_canonical_context",
    "encode_canonical_receipt",
    "encode_canonical_route",
    "provider_free_exports",
]
