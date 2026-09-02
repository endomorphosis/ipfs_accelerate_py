"""SPAR-026 datasets test/proof selection adapter and full-suite fallback.

This module extends current supervisor validation orchestration with
``RefactorValidationSelectionAdapter@1``.  It consumes datasets-owned
``TestSelection@1`` mappings bound to SPAR-007 graph, SPAR-008 dynamic
frontier, SPAR-011 compatibility inventory, and SPAR-019 packet preimages.

Datasets remains the only semantic selection authority.  This adapter never
re-selects tests, never traverses graph edges, never invents pytest node IDs,
and never weakens a producer fallback.  Uncertainty and missing raw source
force the declared full-suite / raw-source fallback.  Vector, model, and
heuristic evidence cannot suppress those gates.

The adapter is nomination-only.  It cannot authorize a transition,
completion, or competing authority.  Observational metadata is excluded
from identity.
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


TASK_ID: Final[str] = "SPAR-026"
GOAL_ID: Final[str] = "SPAR-G051"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "validation orchestration"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.selection_adapter@1"
)
DATASETS_SELECTION_AUTHORITY: Final[str] = "ipfs_datasets_py"

REFACTOR_VALIDATION_SELECTION_ADAPTER_INTERFACE: Final[str] = (
    "RefactorValidationSelectionAdapter@1"
)
REFACTOR_VALIDATION_SELECTION_INTERFACE: Final[str] = (
    "RefactorValidationSelection@1"
)
REFACTOR_VALIDATION_SELECTION_RECEIPT_INTERFACE: Final[str] = (
    "RefactorValidationSelectionReceipt@1"
)

REFACTOR_VALIDATION_SELECTION_ADAPTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-validation-selection-adapter@1"
)
REFACTOR_VALIDATION_SELECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-validation-selection@1"
)
REFACTOR_VALIDATION_SELECTION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/refactor-validation-selection-receipt@1"
)

SELECTION_CONTRACT_VERSION: Final[str] = "1"

SELECTION_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
SELECTION_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
SELECTION_CAN_CREATE_AUTHORITY: Final[bool] = False
SELECTION_CAN_WEAKEN_PRODUCER_FALLBACK: Final[bool] = False
SELECTION_CAN_RESELECT: Final[bool] = False
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
DATASETS_OWNS_SELECTION: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_MEMBERS: Final[int] = 16_384
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024

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
    "can_weaken_producer_fallback",
    "can_reselect",
    "projection_is_authority",
)

_OPAQUE_CONFIDENCE: Final[frozenset[str]] = frozenset({"opaque", "heuristic"})
_UNKNOWN_PRESENCE: Final[frozenset[str]] = frozenset({"unknown"})
_REQUIRED_UNSUPPORTED_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {
        "undispositioned",
        "unsupported",
    }
)

FORBIDDEN_RESELECTION_NAMES: Final[frozenset[str]] = frozenset(
    {
        "run_impact_selected",
        "select_tests_and_proofs",
        "traverse_edges",
        "guess_node_id",
        "collect_tests",
    }
)

FALLBACK_NONE: Final[str] = "none"
FALLBACK_FULL_PYTEST: Final[str] = "full_pytest"
FALLBACK_FULL_PROOFS: Final[str] = "full_proofs"
FALLBACK_BOTH: Final[str] = "both"
_PRODUCER_FALLBACKS: Final[frozenset[str]] = frozenset(
    {
        FALLBACK_NONE,
        FALLBACK_FULL_PYTEST,
        FALLBACK_FULL_PROOFS,
        FALLBACK_BOTH,
    }
)


class SelectionAdapterError(ValueError):
    """Fail-closed violation of a SPAR-026 selection-adapter contract."""


class SelectionFallbackKind(str, Enum):
    NONE = FALLBACK_NONE
    FULL_PYTEST = FALLBACK_FULL_PYTEST
    FULL_PROOFS = FALLBACK_FULL_PROOFS
    BOTH = FALLBACK_BOTH


class FallbackReason(str, Enum):
    PRODUCER_FULL_PYTEST = "producer_full_pytest"
    PRODUCER_FULL_PROOFS = "producer_full_proofs"
    DYNAMIC_PYTHON_FRONTIER = "dynamic_python_frontier"
    UNRESOLVED_GRAPH_FRONTIER = "unresolved_graph_frontier"
    UNRESOLVED_OBLIGATIONS = "unresolved_obligations"
    UNKNOWN_TEST_UNIVERSE = "unknown_test_universe"
    OPAQUE_OR_HEURISTIC_EVIDENCE = "opaque_or_heuristic_evidence"
    UNDISPOSITIONED_COMPATIBILITY = "undispositioned_compatibility"
    DYNAMIC_PYTEST_PLUGIN = "dynamic_pytest_plugin"
    INSUFFICIENT_GRAPH_EVIDENCE = "insufficient_graph_evidence"
    MISSING_RAW_SOURCE = "missing_raw_source"
    VECTOR_CANNOT_SUPPRESS_RAW_SOURCE = "vector_cannot_suppress_raw_source"


DECLARED_FALLBACK_KINDS: Final[frozenset[str]] = frozenset(
    kind.value for kind in SelectionFallbackKind
)
DECLARED_FALLBACK_REASONS: Final[frozenset[str]] = frozenset(
    reason.value for reason in FallbackReason
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise SelectionAdapterError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise SelectionAdapterError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise SelectionAdapterError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise SelectionAdapterError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise SelectionAdapterError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise SelectionAdapterError(f"{name} must be a valid CID") from exc


def _optional_cid(value: Any, name: str) -> str:
    if value in (None, ""):
        return ""
    return _cid(value, name)


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SelectionAdapterError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise SelectionAdapterError(f"{name} must be a non-negative integer")
    return value


def _tree_id(value: Any) -> str:
    text = _text(value, "tree_id")
    if len(text) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in text
    ):
        raise SelectionAdapterError(
            "tree_id must be a lowercase hex Git tree identity"
        )
    return text


def _closed(data: Mapping[str, Any], fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or isinstance(data, (str, bytes, bytearray)):
        raise SelectionAdapterError(f"{name} must be an object")
    extra = set(data) - fields
    missing = fields - set(data)
    if extra & IDENTITY_EXCLUDED_FIELDS:
        raise SelectionAdapterError(
            f"{name} identity excludes observational fields: "
            f"{sorted(extra & IDENTITY_EXCLUDED_FIELDS)}"
        )
    if extra:
        raise SelectionAdapterError(f"unknown {name} field: {sorted(extra)}")
    if missing:
        raise SelectionAdapterError(f"missing {name} field: {sorted(missing)}")
    return dict(data)


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise SelectionAdapterError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _require_dag_json(value: Any, name: str) -> None:
    try:
        cid_for_dag_json(value)
    except Exception as exc:
        raise SelectionAdapterError(f"{name} must be strict DAG-JSON") from exc


def _verify_cid(claimed: Any, computed: str, name: str) -> None:
    cid = _cid(claimed, name)
    if cid != computed:
        raise SelectionAdapterError(f"{name} does not verify")


def _unique_sorted_text(values: Any, name: str, *, limit: int) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise SelectionAdapterError(f"{name} must be a list")
    ordered = tuple(sorted(_text(item, name) for item in values))
    if len(ordered) > limit:
        raise SelectionAdapterError(f"{name} exceeds maximum length")
    if len(ordered) != len(set(ordered)):
        raise SelectionAdapterError(f"{name} must not contain duplicates")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise SelectionAdapterError(f"{name} exceeds path bound")
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
        raise SelectionAdapterError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise SelectionAdapterError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str, *, required: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise SelectionAdapterError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if required and not ordered:
        raise SelectionAdapterError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise SelectionAdapterError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise SelectionAdapterError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise SelectionAdapterError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise SelectionAdapterError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise SelectionAdapterError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        raise SelectionAdapterError(f"missing {name}")
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
    raise SelectionAdapterError(f"{name} must be a mapping")


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
    raise SelectionAdapterError(f"{name} must be an object")


def _mapping_sequence(value: Any, name: str) -> tuple[dict[str, Any], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise SelectionAdapterError(f"{name} must be a list")
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
        raise SelectionAdapterError(f"{name} items must be objects")
    if len(items) > MAX_MEMBERS:
        raise SelectionAdapterError(f"{name} exceeds maximum length")
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


def selection_adapter_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise SelectionAdapterError(f"{name} cannot claim {flag}")


def _fallback_value(value: Any, name: str = "fallback") -> str:
    if isinstance(value, SelectionFallbackKind):
        return value.value
    raw = getattr(value, "value", value)
    text = _text(raw, name)
    if text not in _PRODUCER_FALLBACKS:
        raise SelectionAdapterError(f"unsupported {name} {text!r}")
    return text


def _needs_full_pytest(fallback: str) -> bool:
    return fallback in {FALLBACK_FULL_PYTEST, FALLBACK_BOTH}


def _needs_full_proofs(fallback: str) -> bool:
    return fallback in {FALLBACK_FULL_PROOFS, FALLBACK_BOTH}


def combine_fallbacks(*fallbacks: str) -> str:
    """Return the least upper bound of producer and adapter fallbacks."""

    need_pytest = any(_needs_full_pytest(_fallback_value(item)) for item in fallbacks)
    need_proofs = any(_needs_full_proofs(_fallback_value(item)) for item in fallbacks)
    if need_pytest and need_proofs:
        return FALLBACK_BOTH
    if need_pytest:
        return FALLBACK_FULL_PYTEST
    if need_proofs:
        return FALLBACK_FULL_PROOFS
    return FALLBACK_NONE


def assert_fallback_not_weakened(*, producer: str, effective: str) -> None:
    """Fail closed when an effective fallback drops producer pytest/proof force."""

    producer_fb = _fallback_value(producer, "producer_fallback")
    effective_fb = _fallback_value(effective, "effective_fallback")
    if _needs_full_pytest(producer_fb) and not _needs_full_pytest(effective_fb):
        raise SelectionAdapterError(
            f"cannot weaken producer fallback {producer_fb!r} to {effective_fb!r}"
        )
    if _needs_full_proofs(producer_fb) and not _needs_full_proofs(effective_fb):
        raise SelectionAdapterError(
            f"cannot weaken producer fallback {producer_fb!r} to {effective_fb!r}"
        )


def selection_adapter_descriptor() -> dict[str, Any]:
    return {
        "interface": REFACTOR_VALIDATION_SELECTION_ADAPTER_INTERFACE,
        "adapter_id": ANALYZER_ID,
        "datasets_owns_selection": True,
        "raw_source_required": True,
        "nomination_only": True,
        "forbids": (
            "run_impact_selected",
            "graph_traversal",
            "weaken_producer_fallback",
            "invent_pytest_node_ids",
            "suppress_raw_source",
        ),
    }


def _require_tree(actual: Any, expected: str, name: str) -> str:
    tree = _tree_id(actual)
    if tree != expected:
        raise SelectionAdapterError(f"{name} tree_id does not match packet tree_id")
    return tree


def _producer_node_ids(selection: Mapping[str, Any]) -> tuple[str, ...]:
    nodes = _attr_or_key(selection, "selected_pytest_node_ids")
    if nodes is None:
        return ()
    return _unique_sorted_text(list(nodes), "selected_pytest_node_ids", limit=MAX_MEMBERS)


def _producer_proof_ids(selection: Mapping[str, Any]) -> tuple[str, ...]:
    proofs = _attr_or_key(selection, "selected_proof_ids")
    if proofs is None:
        return ()
    return _unique_sorted_text(list(proofs), "selected_proof_ids", limit=MAX_MEMBERS)


def _producer_fallback_reasons(selection: Mapping[str, Any]) -> tuple[str, ...]:
    reasons = _attr_or_key(selection, "fallback_reasons")
    if reasons in (None, ()):
        return ()
    return _unique_sorted_text(list(reasons), "fallback_reasons", limit=MAX_MEMBERS)


def _packet_write_paths(packet: Mapping[str, Any]) -> tuple[str, ...]:
    if "write_paths" in packet:
        return _exact_paths(packet["write_paths"], "write_paths")
    scope = _nested_mapping(packet.get("effect_scope"), "effect_scope")
    if "write_paths" in scope:
        return _exact_paths(scope["write_paths"], "write_paths")
    raise SelectionAdapterError("SPAR-019 packet write_paths are required")


def _packet_source_cids(packet: Mapping[str, Any]) -> tuple[str, ...]:
    preimage = _nested_mapping(packet.get("preimage"), "preimage")
    sources = preimage.get("source_cids", packet.get("source_cids"))
    if sources in (None, ()):
        raise SelectionAdapterError("raw source required: SPAR-019 preimage source_cids")
    ordered = tuple(sorted(_cid(item, "source_cids") for item in sources))
    if not ordered:
        raise SelectionAdapterError("raw source required: SPAR-019 preimage source_cids")
    if len(ordered) != len(set(ordered)):
        raise SelectionAdapterError("source_cids must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise SelectionAdapterError("source_cids exceed maximum length")
    return ordered


def _packet_validation_commands(packet: Mapping[str, Any]) -> tuple[str, ...]:
    commands = packet.get("validation_commands")
    if commands in (None, ()):
        raise SelectionAdapterError("SPAR-019 validation_commands are required")
    return _commands(commands)


def _packet_cid(packet: Mapping[str, Any]) -> str:
    claimed = packet.get("packet_cid")
    if claimed in (None, ""):
        raise SelectionAdapterError("SPAR-019 packet_cid is required")
    return _cid(claimed, "packet_cid")


def _graph_view_cid(graph: Mapping[str, Any]) -> str:
    claimed = graph.get("graph_view_cid") or graph.get("graph_cid")
    if claimed in (None, ""):
        raise SelectionAdapterError("SPAR-007 graph_view_cid is required")
    return _cid(claimed, "graph_view_cid")


def _graph_unresolved(graph: Mapping[str, Any]) -> bool:
    frontier = graph.get("unresolved_frontier")
    if frontier in (None, "", ()):
        return False
    if isinstance(frontier, Sequence) and not isinstance(
        frontier, (str, bytes, bytearray, Mapping)
    ):
        return len(frontier) > 0
    payload = _nested_mapping(frontier, "unresolved_frontier")
    items = payload.get("items")
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return len(items) > 0
    count = payload.get("unresolved_count")
    if type(count) is int and not isinstance(count, bool):
        return count > 0
    return False


def _graph_opaque(graph: Mapping[str, Any]) -> bool:
    frontier = graph.get("unresolved_frontier")
    payload = _nested_mapping(frontier, "unresolved_frontier") if frontier else {}
    items = _mapping_sequence(payload.get("items"), "unresolved_frontier.items")
    for item in items:
        confidence = str(item.get("confidence") or "")
        evidence = str(item.get("evidence_class") or "")
        reason = str(item.get("reason") or "")
        if confidence in _OPAQUE_CONFIDENCE:
            return True
        if evidence in _NON_ADMITTING_EVIDENCE:
            return True
        if reason in {"opaque_capability", "missing_source", "analyzer_gap"}:
            return True
    return False


def _frontier_unresolved(frontier: Mapping[str, Any]) -> bool:
    count = frontier.get("unresolved_count")
    if type(count) is int and not isinstance(count, bool) and count > 0:
        return True
    unknown = frontier.get("unknown_kinds")
    if isinstance(unknown, Sequence) and not isinstance(unknown, (str, bytes, bytearray)):
        if len(unknown) > 0:
            return True
    findings = _mapping_sequence(frontier.get("findings"), "findings")
    for finding in findings:
        if finding.get("unresolved") is True:
            return True
        presence = str(finding.get("presence") or "")
        if presence in _UNKNOWN_PRESENCE:
            return True
        confidence = str(finding.get("confidence") or "")
        if confidence in _OPAQUE_CONFIDENCE:
            return True
        evidence = str(finding.get("evidence_class") or "")
        if evidence in _NON_ADMITTING_EVIDENCE:
            return True
        kind = str(finding.get("kind") or finding.get("inventory_label") or "")
        if "pytest_plugin" in kind or kind == "dynamic_pytest_plugin":
            return True
    return False


def _frontier_pytest_plugin(frontier: Mapping[str, Any]) -> bool:
    findings = _mapping_sequence(frontier.get("findings"), "findings")
    for finding in findings:
        kind = str(finding.get("kind") or finding.get("inventory_label") or "")
        if "pytest_plugin" in kind or kind == "dynamic_pytest_plugin":
            return True
        family = str(finding.get("family") or "")
        if "pytest" in family and finding.get("unresolved") is True:
            return True
    reasons = frontier.get("fallback_reasons")
    if isinstance(reasons, Sequence) and "dynamic_pytest_plugin" in reasons:
        return True
    return False


def _inventory_undispositioned(inventory: Mapping[str, Any]) -> bool:
    consumers = _mapping_sequence(
        inventory.get("consumers") or inventory.get("consumer_plans"),
        "consumers",
    )
    for consumer in consumers:
        required = consumer.get("required", True)
        if required is not True:
            continue
        disposition = str(consumer.get("disposition") or "")
        if disposition in _REQUIRED_UNSUPPORTED_DISPOSITIONS:
            return True
    obligations = _mapping_sequence(inventory.get("obligations"), "obligations")
    for obligation in obligations:
        required = obligation.get("required", True)
        if required is not True:
            continue
        disposition = str(obligation.get("disposition") or "")
        if disposition in _REQUIRED_UNSUPPORTED_DISPOSITIONS:
            return True
    return bool(inventory.get("undispositioned_required") is True)


def _reject_vector_suppression(vector_evidence: Any) -> None:
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
            raise SelectionAdapterError(
                "vectors/projections cannot suppress raw-source fallback"
            )
        if item.get("skip_full_suite") is True or item.get("weaken_fallback") is True:
            raise SelectionAdapterError(
                "vectors/projections cannot weaken full-suite fallback"
            )
        if evidence in _NON_ADMITTING_EVIDENCE and item.get("admits_selection") is True:
            raise SelectionAdapterError(
                "vector, model, or heuristic evidence cannot admit selection"
            )


def _reject_non_admitting_selection(selection: Mapping[str, Any]) -> None:
    evidence = str(selection.get("evidence_class") or "")
    if evidence in _NON_ADMITTING_EVIDENCE:
        raise SelectionAdapterError(
            "vector, model, or heuristic evidence cannot admit selection"
        )


@dataclass(frozen=True, slots=True)
class RefactorValidationSelection:
    """Bound datasets selection plus SPAR raw-source and full-suite fallback."""

    tree_id: str
    packet_cid: str
    producer_selection_cid: str
    current_root_cid: str
    graph_view_cid: str
    frontier_cid: str
    inventory_cid: str
    producer_fallback: str
    effective_fallback: str
    fallback_reasons: Sequence[str]
    selected_pytest_node_ids: Sequence[str]
    selected_proof_ids: Sequence[str]
    covered_seed_obligation_ids: Sequence[str]
    unresolved_obligation_ids: Sequence[str]
    raw_source_cids: Sequence[str]
    write_paths: Sequence[str]
    validation_commands: Sequence[str]
    previous_root_cid: str = ""
    known_test_universe_cid: str = ""
    known_test_universe_count: int = 0
    raw_source_required: bool = True
    adapter_is_nomination_only: bool = True
    datasets_owns_selection: bool = True

    interface: ClassVar[str] = REFACTOR_VALIDATION_SELECTION_INTERFACE
    schema: ClassVar[str] = REFACTOR_VALIDATION_SELECTION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "packet_cid",
            "producer_selection_cid",
            "previous_root_cid",
            "current_root_cid",
            "graph_view_cid",
            "frontier_cid",
            "inventory_cid",
            "producer_fallback",
            "effective_fallback",
            "fallback_reasons",
            "selected_pytest_node_ids",
            "selected_proof_ids",
            "covered_seed_obligation_ids",
            "unresolved_obligation_ids",
            "raw_source_cids",
            "write_paths",
            "validation_commands",
            "known_test_universe_cid",
            "known_test_universe_count",
            "raw_source_required",
            "adapter_is_nomination_only",
            "datasets_owns_selection",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_weaken_producer_fallback",
            "can_reselect",
            "validation_selection_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        object.__setattr__(self, "packet_cid", _cid(self.packet_cid, "packet_cid"))
        object.__setattr__(
            self,
            "producer_selection_cid",
            _cid(self.producer_selection_cid, "producer_selection_cid"),
        )
        object.__setattr__(
            self,
            "previous_root_cid",
            _optional_cid(self.previous_root_cid, "previous_root_cid"),
        )
        object.__setattr__(
            self, "current_root_cid", _cid(self.current_root_cid, "current_root_cid")
        )
        object.__setattr__(
            self, "graph_view_cid", _cid(self.graph_view_cid, "graph_view_cid")
        )
        object.__setattr__(
            self, "frontier_cid", _cid(self.frontier_cid, "frontier_cid")
        )
        object.__setattr__(
            self, "inventory_cid", _cid(self.inventory_cid, "inventory_cid")
        )
        producer = _fallback_value(self.producer_fallback, "producer_fallback")
        effective = _fallback_value(self.effective_fallback, "effective_fallback")
        assert_fallback_not_weakened(producer=producer, effective=effective)
        object.__setattr__(self, "producer_fallback", producer)
        object.__setattr__(self, "effective_fallback", effective)
        object.__setattr__(
            self,
            "fallback_reasons",
            _unique_sorted_text(
                list(self.fallback_reasons), "fallback_reasons", limit=MAX_MEMBERS
            ),
        )
        object.__setattr__(
            self,
            "selected_pytest_node_ids",
            _unique_sorted_text(
                list(self.selected_pytest_node_ids),
                "selected_pytest_node_ids",
                limit=MAX_MEMBERS,
            ),
        )
        object.__setattr__(
            self,
            "selected_proof_ids",
            _unique_sorted_text(
                list(self.selected_proof_ids),
                "selected_proof_ids",
                limit=MAX_MEMBERS,
            ),
        )
        object.__setattr__(
            self,
            "covered_seed_obligation_ids",
            _unique_sorted_text(
                list(self.covered_seed_obligation_ids),
                "covered_seed_obligation_ids",
                limit=MAX_MEMBERS,
            ),
        )
        object.__setattr__(
            self,
            "unresolved_obligation_ids",
            _unique_sorted_text(
                list(self.unresolved_obligation_ids),
                "unresolved_obligation_ids",
                limit=MAX_MEMBERS,
            ),
        )
        sources = tuple(sorted(_cid(item, "raw_source_cids") for item in self.raw_source_cids))
        if not sources:
            raise SelectionAdapterError("raw source required")
        object.__setattr__(self, "raw_source_cids", sources)
        object.__setattr__(
            self, "write_paths", _exact_paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self,
            "validation_commands",
            _commands(self.validation_commands),
        )
        object.__setattr__(
            self,
            "known_test_universe_cid",
            _optional_cid(self.known_test_universe_cid, "known_test_universe_cid"),
        )
        object.__setattr__(
            self,
            "known_test_universe_count",
            _nonneg_int(self.known_test_universe_count, "known_test_universe_count"),
        )
        if _bool(self.raw_source_required, "raw_source_required") is not True:
            raise SelectionAdapterError("raw_source_required cannot be disabled")
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise SelectionAdapterError("adapter must remain nomination_only")
        if _bool(self.datasets_owns_selection, "datasets_owns_selection") is not True:
            raise SelectionAdapterError("datasets remains the selection authority")
        object.__setattr__(self, "raw_source_required", True)
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "datasets_owns_selection", True)

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
    def can_weaken_producer_fallback(self) -> bool:
        return False

    @property
    def can_reselect(self) -> bool:
        return False

    @property
    def full_suite_required(self) -> bool:
        return self.effective_fallback != FALLBACK_NONE

    @property
    def requires_full_pytest(self) -> bool:
        return _needs_full_pytest(self.effective_fallback)

    @property
    def requires_full_proofs(self) -> bool:
        return _needs_full_proofs(self.effective_fallback)

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_VALIDATION_SELECTION_SCHEMA,
            "interface": REFACTOR_VALIDATION_SELECTION_INTERFACE,
            "tree_id": self.tree_id,
            "packet_cid": self.packet_cid,
            "producer_selection_cid": self.producer_selection_cid,
            "previous_root_cid": self.previous_root_cid,
            "current_root_cid": self.current_root_cid,
            "graph_view_cid": self.graph_view_cid,
            "frontier_cid": self.frontier_cid,
            "inventory_cid": self.inventory_cid,
            "producer_fallback": self.producer_fallback,
            "effective_fallback": self.effective_fallback,
            "fallback_reasons": list(self.fallback_reasons),
            "selected_pytest_node_ids": list(self.selected_pytest_node_ids),
            "selected_proof_ids": list(self.selected_proof_ids),
            "covered_seed_obligation_ids": list(self.covered_seed_obligation_ids),
            "unresolved_obligation_ids": list(self.unresolved_obligation_ids),
            "raw_source_cids": list(self.raw_source_cids),
            "write_paths": list(self.write_paths),
            "validation_commands": list(self.validation_commands),
            "known_test_universe_cid": self.known_test_universe_cid,
            "known_test_universe_count": self.known_test_universe_count,
            "raw_source_required": True,
            "adapter_is_nomination_only": True,
            "datasets_owns_selection": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_weaken_producer_fallback": False,
            "can_reselect": False,
        }
        _require_dag_json(payload, self.__class__.__name__)
        return payload

    @property
    def validation_selection_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["validation_selection_cid"] = self.validation_selection_cid
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefactorValidationSelection":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("validation_selection_cid")
        if payload.pop("schema") != REFACTOR_VALIDATION_SELECTION_SCHEMA:
            raise SelectionAdapterError("unsupported RefactorValidationSelection schema")
        if payload.pop("interface") != REFACTOR_VALIDATION_SELECTION_INTERFACE:
            raise SelectionAdapterError(
                "unsupported RefactorValidationSelection interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("raw_source_required") is not True:
            raise SelectionAdapterError("raw_source_required cannot be disabled")
        if payload.pop("adapter_is_nomination_only") is not True:
            raise SelectionAdapterError("adapter must remain nomination_only")
        if payload.pop("datasets_owns_selection") is not True:
            raise SelectionAdapterError("datasets remains the selection authority")
        result = cls(**payload)
        _verify_cid(claimed, result.validation_selection_cid, "validation_selection_cid")
        return result


@dataclass(frozen=True, slots=True)
class RefactorValidationSelectionReceipt:
    """Content-addressed SPAR-026 adapter receipt. Nomination-only."""

    tree_id: str
    selection: RefactorValidationSelection

    interface: ClassVar[str] = REFACTOR_VALIDATION_SELECTION_RECEIPT_INTERFACE
    schema: ClassVar[str] = REFACTOR_VALIDATION_SELECTION_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "tree_id",
            "validation_selection_cid",
            "producer_selection_cid",
            "packet_cid",
            "effective_fallback",
            "full_suite_required",
            "raw_source_required",
            "adapter_is_nomination_only",
            "can_authorize_completion",
            "can_authorize_transition",
            "analyzer_id",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.selection, RefactorValidationSelection):
            raise SelectionAdapterError("selection must be a RefactorValidationSelection")
        object.__setattr__(self, "tree_id", _tree_id(self.tree_id))
        if self.tree_id != self.selection.tree_id:
            raise SelectionAdapterError("receipt tree_id does not match selection")

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_authorize_transition(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "schema": REFACTOR_VALIDATION_SELECTION_RECEIPT_SCHEMA,
            "interface": REFACTOR_VALIDATION_SELECTION_RECEIPT_INTERFACE,
            "tree_id": self.tree_id,
            "validation_selection_cid": self.selection.validation_selection_cid,
            "producer_selection_cid": self.selection.producer_selection_cid,
            "packet_cid": self.selection.packet_cid,
            "effective_fallback": self.selection.effective_fallback,
            "full_suite_required": self.selection.full_suite_required,
            "raw_source_required": True,
            "adapter_is_nomination_only": True,
            "can_authorize_completion": False,
            "can_authorize_transition": False,
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
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        selection: RefactorValidationSelection | None = None,
    ) -> "RefactorValidationSelectionReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != REFACTOR_VALIDATION_SELECTION_RECEIPT_SCHEMA:
            raise SelectionAdapterError(
                "unsupported RefactorValidationSelectionReceipt schema"
            )
        if payload.pop("interface") != REFACTOR_VALIDATION_SELECTION_RECEIPT_INTERFACE:
            raise SelectionAdapterError(
                "unsupported RefactorValidationSelectionReceipt interface"
            )
        _pop_authority_flags(payload, cls.__name__)
        if payload.pop("raw_source_required") is not True:
            raise SelectionAdapterError("raw_source_required cannot be disabled")
        if payload.pop("adapter_is_nomination_only") is not True:
            raise SelectionAdapterError("adapter must remain nomination_only")
        if payload.pop("analyzer_id") != ANALYZER_ID:
            raise SelectionAdapterError("receipt analyzer_id must remain SPAR-026")
        if selection is None:
            raise SelectionAdapterError("receipt reconstruction requires selection")
        if selection.validation_selection_cid != payload.pop("validation_selection_cid"):
            raise SelectionAdapterError("receipt validation_selection_cid does not match")
        if selection.producer_selection_cid != payload.pop("producer_selection_cid"):
            raise SelectionAdapterError("receipt producer_selection_cid does not match")
        if selection.packet_cid != payload.pop("packet_cid"):
            raise SelectionAdapterError("receipt packet_cid does not match")
        if selection.effective_fallback != payload.pop("effective_fallback"):
            raise SelectionAdapterError("receipt effective_fallback does not match")
        if selection.full_suite_required != payload.pop("full_suite_required"):
            raise SelectionAdapterError("receipt full_suite_required does not match")
        result = cls(tree_id=payload.pop("tree_id"), selection=selection)
        _verify_cid(claimed, result.receipt_cid, "receipt_cid")
        return result


class RefactorValidationSelectionAdapter:
    """Consume datasets-owned selection and enforce SPAR fallbacks."""

    interface: ClassVar[str] = REFACTOR_VALIDATION_SELECTION_ADAPTER_INTERFACE
    schema: ClassVar[str] = REFACTOR_VALIDATION_SELECTION_ADAPTER_SCHEMA
    analyzer_id: ClassVar[str] = ANALYZER_ID

    def adapt(
        self,
        *,
        selection: Mapping[str, Any] | Any,
        packet: Mapping[str, Any] | Any,
        graph: Mapping[str, Any] | Any,
        frontier: Mapping[str, Any] | Any,
        inventory: Mapping[str, Any] | Any,
        vector_evidence: Any = None,
    ) -> RefactorValidationSelection:
        return adapt_refactor_validation_selection(
            selection=selection,
            packet=packet,
            graph=graph,
            frontier=frontier,
            inventory=inventory,
            vector_evidence=vector_evidence,
        )


def adapt_refactor_validation_selection(
    *,
    selection: Mapping[str, Any] | Any,
    packet: Mapping[str, Any] | Any,
    graph: Mapping[str, Any] | Any,
    frontier: Mapping[str, Any] | Any,
    inventory: Mapping[str, Any] | Any,
    vector_evidence: Any = None,
) -> RefactorValidationSelection:
    """Bind datasets selection to SPAR roots/obligations and enforce fallbacks."""

    packet_map = _as_mapping(packet, "SPAR-019 packet")
    graph_map = _as_mapping(graph, "SPAR-007 graph")
    frontier_map = _as_mapping(frontier, "SPAR-008 frontier")
    inventory_map = _as_mapping(inventory, "SPAR-011 inventory")
    selection_map = _as_mapping(selection, "datasets TestSelection")
    _reject_non_admitting_selection(selection_map)
    _reject_vector_suppression(vector_evidence)

    tree_id = _tree_id(packet_map.get("tree_id"))
    binding = _nested_mapping(graph_map.get("binding"), "SPAR-007 binding")
    graph_tree = binding.get("tree_id", graph_map.get("tree_id"))
    _require_tree(graph_tree, tree_id, "SPAR-007")
    _require_tree(frontier_map.get("tree_id"), tree_id, "SPAR-008")
    _require_tree(inventory_map.get("tree_id"), tree_id, "SPAR-011")

    producer_selection_cid = _attr_or_key(selection_map, "selection_cid")
    if producer_selection_cid in (None, ""):
        raise SelectionAdapterError("datasets selection_cid is required")
    producer_selection_cid = _cid(producer_selection_cid, "selection_cid")
    current_root = _attr_or_key(
        selection_map, "current_root_cid", "current_semantic_state_root_cid"
    )
    if current_root in (None, ""):
        raise SelectionAdapterError("datasets current_root_cid is required")
    previous_root = _attr_or_key(
        selection_map, "previous_root_cid", "previous_semantic_state_root_cid"
    )
    producer_fallback = _fallback_value(
        _attr_or_key(selection_map, "fallback") or FALLBACK_NONE,
        "producer_fallback",
    )
    node_ids = _producer_node_ids(selection_map)
    proof_ids = _producer_proof_ids(selection_map)
    covered = _attr_or_key(selection_map, "covered_seed_obligation_ids") or ()
    unresolved = _attr_or_key(selection_map, "unresolved_obligation_ids") or ()
    universe_cid = _attr_or_key(selection_map, "known_test_universe_cid")
    universe_count = _attr_or_key(selection_map, "known_test_universe_count")
    if universe_count is None:
        universe_count = 0

    reasons = list(_producer_fallback_reasons(selection_map))
    extras: list[str] = []
    if _needs_full_pytest(producer_fallback):
        extras.append(FallbackReason.PRODUCER_FULL_PYTEST.value)
    if _needs_full_proofs(producer_fallback):
        extras.append(FallbackReason.PRODUCER_FULL_PROOFS.value)

    escalations = [producer_fallback]
    if _graph_unresolved(graph_map):
        extras.append(FallbackReason.UNRESOLVED_GRAPH_FRONTIER.value)
        escalations.append(FALLBACK_BOTH)
    if _graph_opaque(graph_map):
        extras.append(FallbackReason.OPAQUE_OR_HEURISTIC_EVIDENCE.value)
        extras.append(FallbackReason.INSUFFICIENT_GRAPH_EVIDENCE.value)
        escalations.append(FALLBACK_BOTH)
    if _frontier_unresolved(frontier_map):
        extras.append(FallbackReason.DYNAMIC_PYTHON_FRONTIER.value)
        escalations.append(FALLBACK_BOTH)
    if _frontier_pytest_plugin(frontier_map):
        extras.append(FallbackReason.DYNAMIC_PYTEST_PLUGIN.value)
        escalations.append(FALLBACK_FULL_PYTEST)
    if _inventory_undispositioned(inventory_map):
        extras.append(FallbackReason.UNDISPOSITIONED_COMPATIBILITY.value)
        escalations.append(FALLBACK_FULL_PYTEST)
    unresolved_ids = _unique_sorted_text(
        list(unresolved), "unresolved_obligation_ids", limit=MAX_MEMBERS
    )
    if unresolved_ids:
        extras.append(FallbackReason.UNRESOLVED_OBLIGATIONS.value)
        escalations.append(FALLBACK_BOTH)
    if universe_cid in (None, ""):
        extras.append(FallbackReason.UNKNOWN_TEST_UNIVERSE.value)
        escalations.append(FALLBACK_FULL_PYTEST)

    effective = combine_fallbacks(*escalations)
    assert_fallback_not_weakened(producer=producer_fallback, effective=effective)
    merged_reasons = _unique_sorted_text(
        reasons + extras, "fallback_reasons", limit=MAX_MEMBERS
    )

    frontier_cid = frontier_map.get("frontier_cid")
    if frontier_cid in (None, ""):
        raise SelectionAdapterError("SPAR-008 frontier_cid is required")
    inventory_cid = inventory_map.get("inventory_cid")
    if inventory_cid in (None, ""):
        raise SelectionAdapterError("SPAR-011 inventory_cid is required")

    return RefactorValidationSelection(
        tree_id=tree_id,
        packet_cid=_packet_cid(packet_map),
        producer_selection_cid=producer_selection_cid,
        previous_root_cid=_optional_cid(previous_root, "previous_root_cid"),
        current_root_cid=_cid(current_root, "current_root_cid"),
        graph_view_cid=_graph_view_cid(graph_map),
        frontier_cid=_cid(frontier_cid, "frontier_cid"),
        inventory_cid=_cid(inventory_cid, "inventory_cid"),
        producer_fallback=producer_fallback,
        effective_fallback=effective,
        fallback_reasons=merged_reasons,
        selected_pytest_node_ids=node_ids,
        selected_proof_ids=proof_ids,
        covered_seed_obligation_ids=_unique_sorted_text(
            list(covered), "covered_seed_obligation_ids", limit=MAX_MEMBERS
        ),
        unresolved_obligation_ids=unresolved_ids,
        raw_source_cids=_packet_source_cids(packet_map),
        write_paths=_packet_write_paths(packet_map),
        validation_commands=_packet_validation_commands(packet_map),
        known_test_universe_cid=_optional_cid(universe_cid, "known_test_universe_cid"),
        known_test_universe_count=_nonneg_int(universe_count, "known_test_universe_count"),
    )


def compile_selection_receipt(
    selection: RefactorValidationSelection | Mapping[str, Any],
) -> RefactorValidationSelectionReceipt:
    resolved = (
        selection
        if isinstance(selection, RefactorValidationSelection)
        else RefactorValidationSelection.from_dict(selection)
    )
    return RefactorValidationSelectionReceipt(
        tree_id=resolved.tree_id, selection=resolved
    )


def encode_canonical_selection(selection: RefactorValidationSelection) -> dict[str, Any]:
    return selection.to_dict()


def decode_canonical_selection(payload: Mapping[str, Any]) -> RefactorValidationSelection:
    return RefactorValidationSelection.from_dict(payload)


def encode_canonical_receipt(receipt: RefactorValidationSelectionReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(
    payload: Mapping[str, Any],
    *,
    selection: RefactorValidationSelection,
) -> RefactorValidationSelectionReceipt:
    return RefactorValidationSelectionReceipt.from_dict(payload, selection=selection)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise SelectionAdapterError(
            f"selection adapter must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ADAPTER_IS_NOMINATION_ONLY",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "DATASETS_OWNS_SELECTION",
    "DATASETS_SELECTION_AUTHORITY",
    "DECLARED_FALLBACK_KINDS",
    "DECLARED_FALLBACK_REASONS",
    "DUCKLAKE_IS_AUTHORITY",
    "FALLBACK_BOTH",
    "FALLBACK_FULL_PROOFS",
    "FALLBACK_FULL_PYTEST",
    "FALLBACK_NONE",
    "FORBIDDEN_RESELECTION_NAMES",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "REFACTOR_VALIDATION_SELECTION_ADAPTER_INTERFACE",
    "REFACTOR_VALIDATION_SELECTION_INTERFACE",
    "REFACTOR_VALIDATION_SELECTION_RECEIPT_INTERFACE",
    "SELECTION_CAN_AUTHORIZE_COMPLETION",
    "SELECTION_CAN_AUTHORIZE_TRANSITION",
    "SELECTION_CAN_CREATE_AUTHORITY",
    "SELECTION_CAN_RESELECT",
    "SELECTION_CAN_WEAKEN_PRODUCER_FALLBACK",
    "SELECTION_CONTRACT_VERSION",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "FallbackReason",
    "RefactorValidationSelection",
    "RefactorValidationSelectionAdapter",
    "RefactorValidationSelectionReceipt",
    "SelectionAdapterError",
    "SelectionFallbackKind",
    "adapt_refactor_validation_selection",
    "assert_fallback_not_weakened",
    "assert_not_competing_capsule_family",
    "combine_fallbacks",
    "compile_selection_receipt",
    "decode_canonical_receipt",
    "decode_canonical_selection",
    "encode_canonical_receipt",
    "encode_canonical_selection",
    "provider_free_exports",
    "selection_adapter_cid_profile",
    "selection_adapter_descriptor",
]
