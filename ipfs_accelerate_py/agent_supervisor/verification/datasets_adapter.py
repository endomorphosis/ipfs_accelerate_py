"""Lazy fail-closed semantic input adapter for optional ``ipfs_datasets_py``.

``DatasetsVerificationInputAdapter`` normalizes canonical mapping inputs
(``RepositoryState``, ``InvalidationPlan``, ``SemanticCapsule``,
``ContextPack``) into supervisor-owned views.  Upstream types may be registered
explicitly when they land; the adapter never invents or re-exports absent
datasets classes, never calls arbitrary ``to_dict``, never walks arbitrary
attributes, and never performs network or install side effects.

Importing this module never imports ``ipfs_datasets_py``.  Exact leaf-module
and leaf-symbol probes are required for capability checks: a top-level
namespace-package import alone is not evidence that the dependency is
operational.

Authority rules
---------------
* All adapter outputs are explicitly non-authoritative.
* Datasets ``repository_tree_id`` is opaque selector evidence and is never
  promoted to a receipt ``repository_tree_cid``.
* Validation IDs require an exact reviewed validation-ID → pytest-node-ID
  mapping; absence forces broader selection.
* Missing identities, unknown schemas, malformed CIDs, opaque/uncovered/
  truncated edges, and absent leaf modules produce typed observations.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    MultiformatsIdentityError,
    validate_cid,
)

# ---------------------------------------------------------------------------
# Interface / schema constants
# ---------------------------------------------------------------------------

DATASETS_VERIFICATION_INPUT_ADAPTER_INTERFACE: Final[str] = (
    "DatasetsVerificationInputAdapter@1"
)
DATASETS_VERIFICATION_INPUT_ADAPTER_VERSION: Final[int] = 1

REPOSITORY_STATE_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-repository-state-view@1"
)
INVALIDATION_PLAN_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-invalidation-plan-view@1"
)
SEMANTIC_CAPSULE_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-semantic-capsule-view@1"
)
CONTEXT_PACK_VIEW_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-context-pack-view@1"
)
ADAPTER_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-adapter-observation@1"
)
LEAF_CAPABILITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-leaf-capability@1"
)
VALIDATION_SELECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-validation-selection@1"
)

# Canonical mapping schemas accepted without upstream types.
DATASETS_REPOSITORY_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-repository-state@1"
)
DATASETS_INVALIDATION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-invalidation-plan@1"
)
DATASETS_SEMANTIC_CAPSULE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-semantic-capsule@1"
)
DATASETS_CONTEXT_PACK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-context-pack@1"
)
DATASETS_DEPENDENCY_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/datasets-dependency-edge@1"
)

# Reviewed CodeEvidenceCorpusAdapter impact schemas (dependency gap record).
CODE_IMPACT_INDEX_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.code-impact-index@1"
)
CODE_IMPACT_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.code-impact-result@1"
)

# Exact leaf seams workers may probe (plan §5.1).  Never mutate nested repo.
CODE_EVIDENCE_LEAF_MODULE: Final[str] = (
    "ipfs_datasets_py.knowledge_graphs.adapters.code_evidence"
)
CODE_EVIDENCE_LEAF_SYMBOLS: Final[tuple[str, ...]] = (
    "CodeEvidenceCorpusAdapter",
    "impact_from_index",
    "normalize_impact_index",
)
BOUNDED_TOOL_RUNNER_LEAF_MODULE: Final[str] = (
    "ipfs_datasets_py.logic.backends.process"
)
BOUNDED_TOOL_RUNNER_LEAF_SYMBOL: Final[str] = "BoundedToolRunner"
TOP_LEVEL_NAMESPACE_MODULE: Final[str] = "ipfs_datasets_py"

# Canonical datasets classes are not present yet; only Mapping (+ registry).
DATASETS_CANONICAL_TYPES_GAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "RepositoryState": "absent",
        "InvalidationPlan": "absent",
        "SemanticCapsule": "absent",
        "ContextPack": "absent",
    }
)

_VERSIONED_SCHEMA_RE: Final[re.Pattern[str]] = re.compile(
    r"^[^\x00\r\n]{1,512}@[1-9][0-9]*$"
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_PYTEST_NODE_RE: Final[re.Pattern[str]] = re.compile(
    r"^[^\x00\r\n:]+\.py(?:::[^\x00\r\n]+)+$"
)

_ALLOWED_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "depends_on",
        "tested_by",
        "proved_by",
        "imports",
        "calls",
        "configures",
        "fixtures",
        "opaque",
        "dynamic",
        "unknown",
    }
)
_OPAQUE_EDGE_KINDS: Final[frozenset[str]] = frozenset({"opaque", "dynamic", "unknown"})

_MAX_TEXT: Final[int] = 4096
_MAX_SEQUENCE: Final[int] = 10_000
_MAX_TOKEN_ESTIMATE: Final[int] = 10_000_000


# ---------------------------------------------------------------------------
# Errors / observation vocabulary
# ---------------------------------------------------------------------------


class DatasetsAdapterError(ValueError):
    """Raised only for programmer contract misuse (not soft input failures)."""


class ObservationKind(str, Enum):
    """Typed non-authoritative observation kinds produced by the adapter."""

    PRESENT = "present"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNKNOWN_SCHEMA = "unknown_schema"
    MALFORMED = "malformed"
    MISSING_IDENTITY = "missing_identity"
    OPAQUE = "opaque"
    UNCOVERED = "uncovered"
    TRUNCATED = "truncated"
    LEAF_MODULE_ABSENT = "leaf_module_absent"
    BROADER_SELECTION_REQUIRED = "broader_selection_required"
    DEPENDENCY_GAP = "dependency_gap"


class InputKind(str, Enum):
    REPOSITORY_STATE = "repository_state"
    INVALIDATION_PLAN = "invalidation_plan"
    SEMANTIC_CAPSULE = "semantic_capsule"
    CONTEXT_PACK = "context_pack"
    IMPACT_INDEX = "impact_index"
    IMPACT_RESULT = "impact_result"


class EdgeDisposition(str, Enum):
    EXACT = "exact"
    CONSERVATIVE = "conservative"
    OPAQUE = "opaque"
    UNCOVERED = "uncovered"
    TRUNCATED = "truncated"
    MISSING = "missing"


# ---------------------------------------------------------------------------
# Protocols (typing only — never used for runtime attribute discovery)
# ---------------------------------------------------------------------------


@runtime_checkable
class RepositoryStateProtocol(Protocol):
    """Typing-only protocol.  Runtime acceptance requires Mapping or registry."""



@runtime_checkable
class InvalidationPlanProtocol(Protocol):
    pass


@runtime_checkable
class SemanticCapsuleProtocol(Protocol):
    pass


@runtime_checkable
class ContextPackProtocol(Protocol):
    pass


# ---------------------------------------------------------------------------
# Pure helpers (no network, no install, no arbitrary traversal)
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = _MAX_TEXT,
) -> str:
    if value is None:
        if required:
            raise DatasetsAdapterError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise DatasetsAdapterError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise DatasetsAdapterError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise DatasetsAdapterError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > maximum:
        raise DatasetsAdapterError(f"{field_name} exceeds {maximum} UTF-8 bytes")
    return text


def _optional_text(value: Any, *, field_name: str, maximum: int = _MAX_TEXT) -> str:
    if value is None or value == "":
        return ""
    return _text(value, field_name=field_name, required=True, maximum=maximum)


def _boolean(value: Any, *, field_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise DatasetsAdapterError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int = _MAX_TOKEN_ESTIMATE,
    default: int | None = None,
) -> int:
    if value is None:
        if default is None:
            raise DatasetsAdapterError(f"{field_name} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetsAdapterError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise DatasetsAdapterError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


def _versioned_schema(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, maximum=512)
    if not _VERSIONED_SCHEMA_RE.fullmatch(text):
        raise DatasetsAdapterError(
            f"{field_name} must be an explicitly versioned @N schema"
        )
    return text


def _cid(value: Any, *, field_name: str, required: bool = True) -> str:
    text = _optional_text(value, field_name=field_name, maximum=256)
    if not text:
        if required:
            raise DatasetsAdapterError(f"{field_name} is required")
        return ""
    try:
        return validate_cid(text, codecs=("raw", "dag-json"))
    except MultiformatsIdentityError as exc:
        raise DatasetsAdapterError(f"{field_name} is not a valid CID: {exc}") from exc


def _string_tuple(
    value: Any,
    *,
    field_name: str,
    maximum_items: int = _MAX_SEQUENCE,
    sort: bool = True,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = tuple(value)
    else:
        raise DatasetsAdapterError(f"{field_name} must be a sequence of strings")
    if len(items) > maximum_items:
        raise DatasetsAdapterError(f"{field_name} exceeds {maximum_items} items")
    out: list[str] = []
    for index, item in enumerate(items):
        out.append(_text(item, field_name=f"{field_name}[{index}]"))
    if sort:
        return tuple(sorted(set(out)))
    # Preserve order but drop duplicates deterministically (first wins).
    seen: set[str] = set()
    ordered: list[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetsAdapterError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise DatasetsAdapterError(f"{field_name} keys must be strings")
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray))


# ---------------------------------------------------------------------------
# Observations and views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdapterObservation:
    """Typed non-authoritative observation produced by the adapter."""

    kind: ObservationKind
    reason_code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ADAPTER_OBSERVATION_SCHEMA
    authoritative: bool = False
    completion_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(self, "completion_authority", False)
        object.__setattr__(
            self,
            "details",
            _frozen_mapping(dict(self.details)),
        )
        object.__setattr__(
            self,
            "kind",
            self.kind
            if isinstance(self.kind, ObservationKind)
            else ObservationKind(self.kind),
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, field_name="reason_code", maximum=256),
        )
        object.__setattr__(
            self,
            "message",
            _text(self.message, field_name="message", maximum=_MAX_TEXT),
        )

    @property
    def is_present(self) -> bool:
        return self.kind is ObservationKind.PRESENT

    @property
    def forces_broader_selection(self) -> bool:
        return self.kind in {
            ObservationKind.BROADER_SELECTION_REQUIRED,
            ObservationKind.OPAQUE,
            ObservationKind.UNCOVERED,
            ObservationKind.TRUNCATED,
            ObservationKind.MISSING_IDENTITY,
            ObservationKind.MALFORMED,
            ObservationKind.UNKNOWN_SCHEMA,
            ObservationKind.UNAVAILABLE,
            ObservationKind.LEAF_MODULE_ABSENT,
            ObservationKind.UNSUPPORTED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "reason_code": self.reason_code,
            "message": self.message,
            "details": dict(self.details),
            "authoritative": False,
            "completion_authority": False,
        }


def _observation(
    observation_kind: ObservationKind,
    reason_code: str,
    message: str,
    **details: Any,
) -> AdapterObservation:
    return AdapterObservation(
        kind=observation_kind,
        reason_code=reason_code,
        message=message,
        details=details,
    )


@dataclass(frozen=True, slots=True)
class DependencyEdgeView:
    """Normalized dependency / test / proof edge (non-authoritative)."""

    edge_id: str
    source: str
    target: str
    kind: str
    disposition: EdgeDisposition
    authoritative: bool = False
    truncated: bool = False
    opaque: bool = False
    uncovered: bool = False
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(
            self,
            "disposition",
            self.disposition
            if isinstance(self.disposition, EdgeDisposition)
            else EdgeDisposition(self.disposition),
        )
        object.__setattr__(self, "record", _frozen_mapping(dict(self.record)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "disposition": self.disposition.value,
            "authoritative": False,
            "truncated": self.truncated,
            "opaque": self.opaque,
            "uncovered": self.uncovered,
            "record": dict(self.record),
        }


@dataclass(frozen=True, slots=True)
class SourceSpanView:
    path: str
    start_line: int
    end_line: int
    symbol: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class RepositoryStateView:
    """Normalized repository state.  ``repository_tree_id`` is opaque."""

    schema: str
    repository_tree_id: str
    semantic_state_root_cid: str
    environment_root_cid: str
    dependency_lock_root_cid: str
    observations: tuple[AdapterObservation, ...] = ()
    # Explicitly never populated from datasets repository_tree_id.
    repository_tree_cid: str = ""
    authoritative: bool = False
    view_schema: str = REPOSITORY_STATE_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        # Hard separation: opaque datasets tree id must not alias receipt CID.
        if self.repository_tree_cid and self.repository_tree_cid == self.repository_tree_id:
            object.__setattr__(self, "repository_tree_cid", "")
        object.__setattr__(self, "observations", tuple(self.observations))

    @property
    def usable(self) -> bool:
        return bool(self.repository_tree_id) and bool(self.semantic_state_root_cid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_schema": self.view_schema,
            "schema": self.schema,
            "repository_tree_id": self.repository_tree_id,
            "repository_tree_cid": self.repository_tree_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "environment_root_cid": self.environment_root_cid,
            "dependency_lock_root_cid": self.dependency_lock_root_cid,
            "observations": [item.to_dict() for item in self.observations],
            "authoritative": False,
            "opaque_tree_id_is_not_receipt_tree_cid": True,
        }


@dataclass(frozen=True, slots=True)
class InvalidationPlanView:
    schema: str
    repository_tree_id: str
    semantic_state_root_cid: str
    changed_symbols: tuple[str, ...]
    changed_paths: tuple[str, ...]
    edges: tuple[DependencyEdgeView, ...]
    uncertainty: Mapping[str, Any]
    spans: tuple[SourceSpanView, ...]
    contracts: tuple[Mapping[str, Any], ...]
    uncovered_symbols: tuple[str, ...]
    uncovered_paths: tuple[str, ...]
    truncated: bool
    requires_broader_selection: bool
    observations: tuple[AdapterObservation, ...] = ()
    repository_tree_cid: str = ""
    authoritative: bool = False
    view_schema: str = INVALIDATION_PLAN_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        if self.repository_tree_cid and self.repository_tree_cid == self.repository_tree_id:
            object.__setattr__(self, "repository_tree_cid", "")
        object.__setattr__(self, "uncertainty", _frozen_mapping(dict(self.uncertainty)))
        object.__setattr__(self, "changed_symbols", tuple(self.changed_symbols))
        object.__setattr__(self, "changed_paths", tuple(self.changed_paths))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(
            self,
            "contracts",
            tuple(_frozen_mapping(dict(item)) for item in self.contracts),
        )
        object.__setattr__(self, "uncovered_symbols", tuple(self.uncovered_symbols))
        object.__setattr__(self, "uncovered_paths", tuple(self.uncovered_paths))
        object.__setattr__(self, "observations", tuple(self.observations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_schema": self.view_schema,
            "schema": self.schema,
            "repository_tree_id": self.repository_tree_id,
            "repository_tree_cid": self.repository_tree_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "changed_symbols": list(self.changed_symbols),
            "changed_paths": list(self.changed_paths),
            "edges": [edge.to_dict() for edge in self.edges],
            "uncertainty": dict(self.uncertainty),
            "spans": [span.to_dict() for span in self.spans],
            "contracts": [dict(item) for item in self.contracts],
            "uncovered_symbols": list(self.uncovered_symbols),
            "uncovered_paths": list(self.uncovered_paths),
            "truncated": self.truncated,
            "requires_broader_selection": self.requires_broader_selection,
            "observations": [item.to_dict() for item in self.observations],
            "authoritative": False,
            "opaque_tree_id_is_not_receipt_tree_cid": True,
        }


@dataclass(frozen=True, slots=True)
class SemanticCapsuleView:
    schema: str
    semantic_state_root_cid: str
    repository_tree_id: str
    edges: tuple[DependencyEdgeView, ...]
    spans: tuple[SourceSpanView, ...]
    contracts: tuple[Mapping[str, Any], ...]
    fixture_references: tuple[str, ...]
    truncated: bool
    requires_broader_selection: bool
    observations: tuple[AdapterObservation, ...] = ()
    repository_tree_cid: str = ""
    authoritative: bool = False
    view_schema: str = SEMANTIC_CAPSULE_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        if self.repository_tree_cid and self.repository_tree_cid == self.repository_tree_id:
            object.__setattr__(self, "repository_tree_cid", "")
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "spans", tuple(self.spans))
        object.__setattr__(
            self,
            "contracts",
            tuple(_frozen_mapping(dict(item)) for item in self.contracts),
        )
        object.__setattr__(self, "fixture_references", tuple(self.fixture_references))
        object.__setattr__(self, "observations", tuple(self.observations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_schema": self.view_schema,
            "schema": self.schema,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "repository_tree_id": self.repository_tree_id,
            "repository_tree_cid": self.repository_tree_cid,
            "edges": [edge.to_dict() for edge in self.edges],
            "spans": [span.to_dict() for span in self.spans],
            "contracts": [dict(item) for item in self.contracts],
            "fixture_references": list(self.fixture_references),
            "truncated": self.truncated,
            "requires_broader_selection": self.requires_broader_selection,
            "observations": [item.to_dict() for item in self.observations],
            "authoritative": False,
            "opaque_tree_id_is_not_receipt_tree_cid": True,
        }


@dataclass(frozen=True, slots=True)
class ContextPackView:
    schema: str
    repository_tree_id: str
    semantic_state_root_cid: str
    environment_root_cid: str
    dependency_lock_root_cid: str
    token_estimate: int
    fixture_task_references: tuple[str, ...]
    contracts: tuple[Mapping[str, Any], ...]
    observations: tuple[AdapterObservation, ...] = ()
    repository_tree_cid: str = ""
    authoritative: bool = False
    view_schema: str = CONTEXT_PACK_VIEW_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        if self.repository_tree_cid and self.repository_tree_cid == self.repository_tree_id:
            object.__setattr__(self, "repository_tree_cid", "")
        object.__setattr__(
            self, "fixture_task_references", tuple(self.fixture_task_references)
        )
        object.__setattr__(
            self,
            "contracts",
            tuple(_frozen_mapping(dict(item)) for item in self.contracts),
        )
        object.__setattr__(self, "observations", tuple(self.observations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_schema": self.view_schema,
            "schema": self.schema,
            "repository_tree_id": self.repository_tree_id,
            "repository_tree_cid": self.repository_tree_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "environment_root_cid": self.environment_root_cid,
            "dependency_lock_root_cid": self.dependency_lock_root_cid,
            "token_estimate": self.token_estimate,
            "fixture_task_references": list(self.fixture_task_references),
            "contracts": [dict(item) for item in self.contracts],
            "observations": [item.to_dict() for item in self.observations],
            "authoritative": False,
            "opaque_tree_id_is_not_receipt_tree_cid": True,
        }


@dataclass(frozen=True, slots=True)
class ValidationSelectionView:
    """Result of mapping validation IDs to exact pytest node IDs."""

    schema: str = VALIDATION_SELECTION_SCHEMA
    validation_ids: tuple[str, ...] = ()
    mapped_pytest_node_ids: tuple[str, ...] = ()
    unmapped_validation_ids: tuple[str, ...] = ()
    requires_broader_selection: bool = False
    observations: tuple[AdapterObservation, ...] = ()
    authoritative: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)
        object.__setattr__(self, "validation_ids", tuple(self.validation_ids))
        object.__setattr__(
            self, "mapped_pytest_node_ids", tuple(self.mapped_pytest_node_ids)
        )
        object.__setattr__(
            self, "unmapped_validation_ids", tuple(self.unmapped_validation_ids)
        )
        object.__setattr__(self, "observations", tuple(self.observations))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "validation_ids": list(self.validation_ids),
            "mapped_pytest_node_ids": list(self.mapped_pytest_node_ids),
            "unmapped_validation_ids": list(self.unmapped_validation_ids),
            "requires_broader_selection": self.requires_broader_selection,
            "observations": [item.to_dict() for item in self.observations],
            "authoritative": False,
        }


@dataclass(frozen=True, slots=True)
class LeafCapability:
    """Exact leaf-module/symbol capability receipt."""

    schema: str
    module: str
    symbol: str
    available: bool
    reason_code: str
    message: str
    authoritative: bool = False
    top_level_namespace_insufficient: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "authoritative", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "module": self.module,
            "symbol": self.symbol,
            "available": self.available,
            "reason_code": self.reason_code,
            "message": self.message,
            "authoritative": False,
            "top_level_namespace_insufficient": True,
        }


@dataclass(frozen=True, slots=True)
class NormalizeResult:
    """Envelope for a single normalize call (view or typed failure)."""

    input_kind: InputKind
    observation: AdapterObservation
    view: (
        RepositoryStateView
        | InvalidationPlanView
        | SemanticCapsuleView
        | ContextPackView
        | None
    ) = None

    @property
    def ok(self) -> bool:
        return self.observation.is_present and self.view is not None

    @property
    def requires_broader_selection(self) -> bool:
        if self.observation.forces_broader_selection:
            return True
        view = self.view
        if isinstance(view, (InvalidationPlanView, SemanticCapsuleView)):
            return view.requires_broader_selection
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_kind": self.input_kind.value,
            "observation": self.observation.to_dict(),
            "view": None if self.view is None else self.view.to_dict(),
            "ok": self.ok,
            "requires_broader_selection": self.requires_broader_selection,
            "authoritative": False,
        }


# ---------------------------------------------------------------------------
# Upstream type registry (explicit only; does not change authority)
# ---------------------------------------------------------------------------


UpstreamConverter = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class _RegisteredUpstreamType:
    type_object: type
    input_kind: InputKind
    converter: UpstreamConverter
    module_name: str
    symbol_name: str


# ---------------------------------------------------------------------------
# Normalization internals
# ---------------------------------------------------------------------------


def _normalize_edge(
    payload: Mapping[str, Any],
    *,
    index: int,
) -> tuple[DependencyEdgeView, list[AdapterObservation]]:
    observations: list[AdapterObservation] = []
    prefix = f"edges[{index}]"
    source = _text(payload.get("source") or payload.get("source_id"), field_name=f"{prefix}.source")
    target = _text(payload.get("target") or payload.get("target_id"), field_name=f"{prefix}.target")
    kind = _optional_text(payload.get("kind") or payload.get("edge_kind"), field_name=f"{prefix}.kind")
    if not kind:
        kind = "unknown"
    kind = kind.strip().lower()
    if kind not in _ALLOWED_EDGE_KINDS:
        # Unknown kinds are accepted as opaque non-authoritative edges.
        observations.append(
            _observation(
                ObservationKind.OPAQUE,
                "unknown_edge_kind",
                f"edge kind {kind!r} treated as opaque",
                kind=kind,
            )
        )
        kind = "opaque"

    truncated = _boolean(payload.get("truncated"), field_name=f"{prefix}.truncated")
    uncovered = _boolean(payload.get("uncovered"), field_name=f"{prefix}.uncovered")
    opaque_flag = _boolean(payload.get("opaque"), field_name=f"{prefix}.opaque")
    opaque = opaque_flag or kind in _OPAQUE_EDGE_KINDS

    if truncated:
        disposition = EdgeDisposition.TRUNCATED
        observations.append(
            _observation(
                ObservationKind.TRUNCATED,
                "truncated_edge",
                f"edge {source}->{target} is truncated",
                source=source,
                target=target,
            )
        )
    elif uncovered:
        disposition = EdgeDisposition.UNCOVERED
        observations.append(
            _observation(
                ObservationKind.UNCOVERED,
                "uncovered_edge",
                f"edge {source}->{target} is uncovered",
                source=source,
                target=target,
            )
        )
    elif opaque:
        disposition = EdgeDisposition.OPAQUE
        observations.append(
            _observation(
                ObservationKind.OPAQUE,
                "opaque_edge",
                f"edge {source}->{target} is opaque",
                source=source,
                target=target,
                kind=kind,
            )
        )
    else:
        disposition = EdgeDisposition.EXACT

    edge_id = _optional_text(payload.get("edge_id"), field_name=f"{prefix}.edge_id")
    if not edge_id:
        edge_id = f"edge:{kind}:{source}->{target}"

    record_raw = payload.get("record")
    if record_raw is None:
        record: Mapping[str, Any] = {}
    else:
        record = dict(_mapping(record_raw, field_name=f"{prefix}.record"))

    edge = DependencyEdgeView(
        edge_id=edge_id,
        source=source,
        target=target,
        kind=kind,
        disposition=disposition,
        truncated=truncated,
        opaque=opaque,
        uncovered=uncovered,
        record=record,
    )
    return edge, observations


def _normalize_edges(
    value: Any,
    *,
    field_name: str = "edges",
) -> tuple[tuple[DependencyEdgeView, ...], list[AdapterObservation], bool]:
    observations: list[AdapterObservation] = []
    if value is None:
        return (), observations, False
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DatasetsAdapterError(f"{field_name} must be a sequence of mappings")
    if len(value) > _MAX_SEQUENCE:
        raise DatasetsAdapterError(f"{field_name} exceeds {_MAX_SEQUENCE} items")

    edges: list[DependencyEdgeView] = []
    broader = False
    for index, item in enumerate(value):
        mapping = _mapping(item, field_name=f"{field_name}[{index}]")
        edge, edge_obs = _normalize_edge(mapping, index=index)
        edges.append(edge)
        observations.extend(edge_obs)
        if edge.disposition in {
            EdgeDisposition.OPAQUE,
            EdgeDisposition.UNCOVERED,
            EdgeDisposition.TRUNCATED,
            EdgeDisposition.MISSING,
        }:
            broader = True

    # Deterministic order: kind, source, target, edge_id.
    edges.sort(key=lambda e: (e.kind, e.source, e.target, e.edge_id))
    return tuple(edges), observations, broader


def _normalize_spans(
    value: Any,
    *,
    field_name: str = "spans",
) -> tuple[SourceSpanView, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DatasetsAdapterError(f"{field_name} must be a sequence of mappings")
    spans: list[SourceSpanView] = []
    for index, item in enumerate(value):
        mapping = _mapping(item, field_name=f"{field_name}[{index}]")
        path = _text(mapping.get("path"), field_name=f"{field_name}[{index}].path")
        start = _integer(
            mapping.get("start_line", mapping.get("line", 1)),
            field_name=f"{field_name}[{index}].start_line",
            minimum=1,
            maximum=10_000_000,
        )
        end = _integer(
            mapping.get("end_line", start),
            field_name=f"{field_name}[{index}].end_line",
            minimum=1,
            maximum=10_000_000,
        )
        if end < start:
            raise DatasetsAdapterError(
                f"{field_name}[{index}].end_line must be >= start_line"
            )
        symbol = _optional_text(
            mapping.get("symbol"), field_name=f"{field_name}[{index}].symbol"
        )
        spans.append(
            SourceSpanView(path=path, start_line=start, end_line=end, symbol=symbol)
        )
    spans.sort(key=lambda s: (s.path, s.start_line, s.end_line, s.symbol))
    return tuple(spans)


def _normalize_contracts(
    value: Any,
    *,
    field_name: str = "contracts",
) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise DatasetsAdapterError(f"{field_name} must be a sequence of mappings")
    contracts: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        mapping = _mapping(item, field_name=f"{field_name}[{index}]")
        # Only retain string-keyed shallow copies; no nested execution.
        contracts.append(MappingProxyType(dict(mapping)))
    return tuple(contracts)


def _uncertainty_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    mapping = _mapping(value, field_name="uncertainty")
    return MappingProxyType(dict(mapping))


def _soft_normalize(
    input_kind: InputKind,
    fn: Callable[[], Any],
) -> NormalizeResult:
    try:
        view = fn()
    except DatasetsAdapterError as exc:
        message = str(exc)
        lower = message.lower()
        if "schema" in lower and (
            "versioned" in lower or "unsupported" in lower or "unknown" in lower
        ):
            kind = ObservationKind.UNKNOWN_SCHEMA
            reason = "unknown_schema"
        elif "required" in lower or "must not be empty" in lower:
            kind = ObservationKind.MISSING_IDENTITY
            reason = "missing_identity"
        elif "not a valid cid" in lower or "cid is truncated" in lower:
            kind = ObservationKind.MALFORMED
            reason = "malformed_cid"
        else:
            kind = ObservationKind.MALFORMED
            reason = "malformed_input"
        return NormalizeResult(
            input_kind=input_kind,
            observation=_observation(kind, reason, message),
            view=None,
        )
    if isinstance(view, NormalizeResult):
        return view
    return NormalizeResult(
        input_kind=input_kind,
        observation=_observation(
            ObservationKind.PRESENT,
            "normalized",
            f"{input_kind.value} normalized",
        ),
        view=view,
    )


# ---------------------------------------------------------------------------
# Leaf probes
# ---------------------------------------------------------------------------


def _default_importer(module_name: str) -> Any:
    return importlib.import_module(module_name)


def probe_leaf_symbol(
    module_name: str,
    symbol_name: str,
    *,
    importer: Callable[[str], Any] | None = None,
) -> LeafCapability:
    """Probe one exact leaf module attribute.  Never installs or networks."""

    import_fn = importer or _default_importer
    try:
        module = import_fn(module_name)
    except Exception as exc:  # noqa: BLE001 — typed unavailable for any import failure
        return LeafCapability(
            schema=LEAF_CAPABILITY_SCHEMA,
            module=module_name,
            symbol=symbol_name,
            available=False,
            reason_code="leaf_module_absent",
            message=f"{module_name} unavailable: {type(exc).__name__}: {exc}",
        )
    if not hasattr(module, symbol_name):
        return LeafCapability(
            schema=LEAF_CAPABILITY_SCHEMA,
            module=module_name,
            symbol=symbol_name,
            available=False,
            reason_code="leaf_symbol_absent",
            message=f"{module_name}.{symbol_name} is absent",
        )
    symbol = getattr(module, symbol_name)
    if symbol is None:
        return LeafCapability(
            schema=LEAF_CAPABILITY_SCHEMA,
            module=module_name,
            symbol=symbol_name,
            available=False,
            reason_code="leaf_symbol_absent",
            message=f"{module_name}.{symbol_name} is None",
        )
    return LeafCapability(
        schema=LEAF_CAPABILITY_SCHEMA,
        module=module_name,
        symbol=symbol_name,
        available=True,
        reason_code="leaf_symbol_present",
        message=f"{module_name}.{symbol_name} is available",
    )


def probe_top_level_namespace_alone(
    *,
    importer: Callable[[str], Any] | None = None,
) -> LeafCapability:
    """A top-level namespace import alone is never treated as availability."""

    import_fn = importer or _default_importer
    try:
        module = import_fn(TOP_LEVEL_NAMESPACE_MODULE)
    except Exception as exc:  # noqa: BLE001
        return LeafCapability(
            schema=LEAF_CAPABILITY_SCHEMA,
            module=TOP_LEVEL_NAMESPACE_MODULE,
            symbol="",
            available=False,
            reason_code="top_level_namespace_unavailable",
            message=(
                f"{TOP_LEVEL_NAMESPACE_MODULE} namespace unavailable: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    # Even when the namespace import succeeds, it is insufficient evidence.
    _ = module
    return LeafCapability(
        schema=LEAF_CAPABILITY_SCHEMA,
        module=TOP_LEVEL_NAMESPACE_MODULE,
        symbol="",
        available=False,
        reason_code="top_level_namespace_insufficient",
        message=(
            f"import of {TOP_LEVEL_NAMESPACE_MODULE} alone is not evidence "
            "that leaf capabilities are operational"
        ),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DatasetsVerificationInputAdapter:
    """Lazy fail-closed adapter for datasets semantic inputs.

    Construction and mapping-only normalization never import
    ``ipfs_datasets_py``.  Leaf probes import only on demand.
    """

    interface: Final[str] = DATASETS_VERIFICATION_INPUT_ADAPTER_INTERFACE
    version: Final[int] = DATASETS_VERIFICATION_INPUT_ADAPTER_VERSION

    def __init__(
        self,
        *,
        importer: Callable[[str], Any] | None = None,
    ) -> None:
        self._importer = importer or _default_importer
        self._registry: list[_RegisteredUpstreamType] = []
        # capability cache: (module, symbol) -> LeafCapability
        self._leaf_cache: dict[tuple[str, str], LeafCapability] = {}

    # -- discovery / gap -------------------------------------------------

    def canonical_types_gap(self) -> Mapping[str, str]:
        """Record that canonical datasets classes are not re-exported here."""

        return dict(DATASETS_CANONICAL_TYPES_GAP)

    def code_evidence_impact_schemas(self) -> Mapping[str, str]:
        """Document the reviewed CodeEvidenceCorpusAdapter impact schemas."""

        return {
            "code_impact_index": CODE_IMPACT_INDEX_SCHEMA,
            "code_impact_result": CODE_IMPACT_RESULT_SCHEMA,
        }

    def capability_declaration(self) -> Mapping[str, Any]:
        """Local capability declaration without probing leaf modules."""

        return {
            "interface": self.interface,
            "version": self.version,
            "lazy": True,
            "authoritative": False,
            "completion_authority": False,
            "accepts_mappings": True,
            "accepts_registered_upstream_types": True,
            "invokes_to_dict": False,
            "arbitrary_attribute_traversal": False,
            "network_side_effects": False,
            "install_side_effects": False,
            "canonical_types_gap": dict(DATASETS_CANONICAL_TYPES_GAP),
            "code_evidence_impact_schemas": dict(self.code_evidence_impact_schemas()),
            "leaf_probes": {
                "code_evidence": {
                    "module": CODE_EVIDENCE_LEAF_MODULE,
                    "symbols": list(CODE_EVIDENCE_LEAF_SYMBOLS),
                },
                "bounded_tool_runner": {
                    "module": BOUNDED_TOOL_RUNNER_LEAF_MODULE,
                    "symbol": BOUNDED_TOOL_RUNNER_LEAF_SYMBOL,
                },
            },
            "top_level_namespace_insufficient": True,
            "repository_tree_id_is_opaque": True,
            "repository_tree_id_not_receipt_tree_cid": True,
        }

    # -- upstream registration -------------------------------------------

    def register_upstream_type(
        self,
        type_object: type,
        *,
        input_kind: InputKind | str,
        converter: UpstreamConverter,
        module_name: str,
        symbol_name: str,
    ) -> None:
        """Register an exact upstream type without changing authority.

        The converter must return a strict canonical mapping.  The adapter
        never calls ``to_dict`` on the instance and never walks attributes.
        """

        if not isinstance(type_object, type):
            raise DatasetsAdapterError("type_object must be a type")
        if not callable(converter):
            raise DatasetsAdapterError("converter must be callable")
        kind = (
            input_kind
            if isinstance(input_kind, InputKind)
            else InputKind(str(input_kind))
        )
        module = _text(module_name, field_name="module_name", maximum=512)
        symbol = _text(symbol_name, field_name="symbol_name", maximum=256)
        # Replace existing registration for the same type+kind.
        self._registry = [
            entry
            for entry in self._registry
            if not (entry.type_object is type_object and entry.input_kind is kind)
        ]
        self._registry.append(
            _RegisteredUpstreamType(
                type_object=type_object,
                input_kind=kind,
                converter=converter,
                module_name=module,
                symbol_name=symbol,
            )
        )

    def registered_upstream_types(self) -> tuple[Mapping[str, str], ...]:
        return tuple(
            {
                "module_name": entry.module_name,
                "symbol_name": entry.symbol_name,
                "input_kind": entry.input_kind.value,
                "type_name": entry.type_object.__name__,
            }
            for entry in self._registry
        )

    # -- leaf probes -----------------------------------------------------

    def probe_code_evidence(self, *, use_cache: bool = True) -> Mapping[str, LeafCapability]:
        results: dict[str, LeafCapability] = {}
        for symbol in CODE_EVIDENCE_LEAF_SYMBOLS:
            results[symbol] = self._probe_cached(
                CODE_EVIDENCE_LEAF_MODULE, symbol, use_cache=use_cache
            )
        return results

    def probe_bounded_tool_runner(self, *, use_cache: bool = True) -> LeafCapability:
        return self._probe_cached(
            BOUNDED_TOOL_RUNNER_LEAF_MODULE,
            BOUNDED_TOOL_RUNNER_LEAF_SYMBOL,
            use_cache=use_cache,
        )

    def probe_top_level_namespace(self) -> LeafCapability:
        return probe_top_level_namespace_alone(importer=self._importer)

    def _probe_cached(
        self,
        module_name: str,
        symbol_name: str,
        *,
        use_cache: bool,
    ) -> LeafCapability:
        key = (module_name, symbol_name)
        if use_cache and key in self._leaf_cache:
            return self._leaf_cache[key]
        capability = probe_leaf_symbol(
            module_name, symbol_name, importer=self._importer
        )
        self._leaf_cache[key] = capability
        return capability

    # -- input coercion (no to_dict, no attr walk) ------------------------

    def _coerce_to_mapping(
        self,
        value: Any,
        *,
        input_kind: InputKind,
    ) -> tuple[Mapping[str, Any] | None, AdapterObservation | None]:
        if value is None:
            return None, _observation(
                ObservationKind.MISSING_IDENTITY,
                "missing_input",
                f"{input_kind.value} input is required",
            )
        if _is_mapping(value):
            return value, None

        for entry in self._registry:
            if entry.input_kind is input_kind and isinstance(value, entry.type_object):
                try:
                    converted = entry.converter(value)
                except Exception as exc:  # noqa: BLE001
                    return None, _observation(
                        ObservationKind.MALFORMED,
                        "upstream_converter_failed",
                        (
                            f"registered converter for {entry.module_name}."
                            f"{entry.symbol_name} failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                        module_name=entry.module_name,
                        symbol_name=entry.symbol_name,
                    )
                if not _is_mapping(converted):
                    return None, _observation(
                        ObservationKind.MALFORMED,
                        "upstream_converter_non_mapping",
                        "registered converter must return a mapping",
                        module_name=entry.module_name,
                        symbol_name=entry.symbol_name,
                    )
                return converted, None

        # Forbidden: do not call to_dict, do not walk attributes.
        type_name = type(value).__name__
        if DATASETS_CANONICAL_TYPES_GAP.get(
            type_name if type_name in DATASETS_CANONICAL_TYPES_GAP else "",
            "",
        ) == "absent" or type_name in DATASETS_CANONICAL_TYPES_GAP:
            return None, _observation(
                ObservationKind.DEPENDENCY_GAP,
                "canonical_type_absent",
                (
                    f"canonical datasets type {type_name} is not available; "
                    "provide a strict mapping or register an upstream type"
                ),
                type_name=type_name,
            )
        return None, _observation(
            ObservationKind.UNSUPPORTED,
            "unregistered_input_type",
            (
                f"unsupported {input_kind.value} input type {type_name}; "
                "accept Mapping or an explicitly registered upstream type only"
            ),
            type_name=type_name,
        )

    # -- public normalize APIs -------------------------------------------

    def normalize_repository_state(self, value: Any) -> NormalizeResult:
        def _build() -> RepositoryStateView:
            mapping, error = self._coerce_to_mapping(
                value, input_kind=InputKind.REPOSITORY_STATE
            )
            if error is not None:
                return NormalizeResult(
                    input_kind=InputKind.REPOSITORY_STATE,
                    observation=error,
                    view=None,
                )
            assert mapping is not None
            schema = _versioned_schema(
                mapping.get("schema") or DATASETS_REPOSITORY_STATE_SCHEMA,
                field_name="schema",
            )
            if schema != DATASETS_REPOSITORY_STATE_SCHEMA:
                raise DatasetsAdapterError(f"unsupported repository state schema: {schema}")

            repository_tree_id = _text(
                mapping.get("repository_tree_id"),
                field_name="repository_tree_id",
            )
            semantic = _cid(
                mapping.get("semantic_state_root_cid"),
                field_name="semantic_state_root_cid",
            )
            environment = _cid(
                mapping.get("environment_root_cid"),
                field_name="environment_root_cid",
                required=False,
            )
            lock = _cid(
                mapping.get("dependency_lock_root_cid"),
                field_name="dependency_lock_root_cid",
                required=False,
            )
            # Caller may supply a separate receipt tree CID; never copy tree_id.
            raw_receipt_tree = mapping.get("repository_tree_cid")
            if (
                raw_receipt_tree is None
                or raw_receipt_tree == ""
                or raw_receipt_tree == repository_tree_id
            ):
                receipt_tree_cid = ""
            else:
                receipt_tree_cid = _cid(
                    raw_receipt_tree,
                    field_name="repository_tree_cid",
                    required=False,
                )
                if receipt_tree_cid == repository_tree_id:
                    receipt_tree_cid = ""

            observations: list[AdapterObservation] = []
            if not environment:
                observations.append(
                    _observation(
                        ObservationKind.MISSING_IDENTITY,
                        "missing_environment_root",
                        "environment_root_cid absent; non-authoritative",
                    )
                )
            if not lock:
                observations.append(
                    _observation(
                        ObservationKind.MISSING_IDENTITY,
                        "missing_dependency_lock_root",
                        "dependency_lock_root_cid absent; non-authoritative",
                    )
                )

            return RepositoryStateView(
                schema=schema,
                repository_tree_id=repository_tree_id,
                semantic_state_root_cid=semantic,
                environment_root_cid=environment,
                dependency_lock_root_cid=lock,
                repository_tree_cid=receipt_tree_cid,
                observations=tuple(observations),
            )

        return _soft_normalize(InputKind.REPOSITORY_STATE, _build)

    def normalize_invalidation_plan(self, value: Any) -> NormalizeResult:
        def _build() -> InvalidationPlanView | NormalizeResult:
            mapping, error = self._coerce_to_mapping(
                value, input_kind=InputKind.INVALIDATION_PLAN
            )
            if error is not None:
                return NormalizeResult(
                    input_kind=InputKind.INVALIDATION_PLAN,
                    observation=error,
                    view=None,
                )
            assert mapping is not None
            schema = _versioned_schema(
                mapping.get("schema") or DATASETS_INVALIDATION_PLAN_SCHEMA,
                field_name="schema",
            )
            if schema != DATASETS_INVALIDATION_PLAN_SCHEMA:
                raise DatasetsAdapterError(
                    f"unsupported invalidation plan schema: {schema}"
                )

            repository_tree_id = _text(
                mapping.get("repository_tree_id"),
                field_name="repository_tree_id",
            )
            semantic = _cid(
                mapping.get("semantic_state_root_cid"),
                field_name="semantic_state_root_cid",
            )
            changed_symbols = _string_tuple(
                mapping.get("changed_symbols"), field_name="changed_symbols"
            )
            changed_paths = _string_tuple(
                mapping.get("changed_paths"), field_name="changed_paths"
            )
            edges, edge_obs, edge_broader = _normalize_edges(mapping.get("edges"))
            spans = _normalize_spans(mapping.get("spans"))
            contracts = _normalize_contracts(mapping.get("contracts"))
            uncertainty = _uncertainty_mapping(mapping.get("uncertainty"))
            uncovered_symbols = _string_tuple(
                mapping.get("uncovered_symbols"), field_name="uncovered_symbols"
            )
            uncovered_paths = _string_tuple(
                mapping.get("uncovered_paths"), field_name="uncovered_paths"
            )
            truncated = _boolean(mapping.get("truncated"), field_name="truncated")
            uncovered_impact = _boolean(
                mapping.get("uncovered_impact"), field_name="uncovered_impact"
            )
            if uncovered_symbols or uncovered_paths:
                uncovered_impact = True

            observations = list(edge_obs)
            broader = edge_broader or truncated or uncovered_impact
            if truncated:
                observations.append(
                    _observation(
                        ObservationKind.TRUNCATED,
                        "truncated_plan",
                        "invalidation plan reports truncated frontiers",
                    )
                )
            if uncovered_impact:
                observations.append(
                    _observation(
                        ObservationKind.UNCOVERED,
                        "uncovered_impact",
                        "invalidation plan reports uncovered impact",
                        uncovered_symbols=list(uncovered_symbols),
                        uncovered_paths=list(uncovered_paths),
                    )
                )
            if broader:
                observations.append(
                    _observation(
                        ObservationKind.BROADER_SELECTION_REQUIRED,
                        "uncertain_selector",
                        "opaque, uncovered, or truncated edges force broader selection",
                    )
                )

            raw_receipt_tree = mapping.get("repository_tree_cid")
            if (
                raw_receipt_tree is None
                or raw_receipt_tree == ""
                or raw_receipt_tree == repository_tree_id
            ):
                receipt_tree_cid = ""
            else:
                receipt_tree_cid = _cid(
                    raw_receipt_tree,
                    field_name="repository_tree_cid",
                    required=False,
                )
                if receipt_tree_cid == repository_tree_id:
                    receipt_tree_cid = ""

            return InvalidationPlanView(
                schema=schema,
                repository_tree_id=repository_tree_id,
                semantic_state_root_cid=semantic,
                changed_symbols=changed_symbols,
                changed_paths=changed_paths,
                edges=edges,
                uncertainty=uncertainty,
                spans=spans,
                contracts=contracts,
                uncovered_symbols=uncovered_symbols,
                uncovered_paths=uncovered_paths,
                truncated=truncated,
                requires_broader_selection=broader,
                observations=tuple(observations),
                repository_tree_cid=receipt_tree_cid,
            )

        return _soft_normalize(InputKind.INVALIDATION_PLAN, _build)

    def normalize_semantic_capsule(self, value: Any) -> NormalizeResult:
        def _build() -> SemanticCapsuleView | NormalizeResult:
            mapping, error = self._coerce_to_mapping(
                value, input_kind=InputKind.SEMANTIC_CAPSULE
            )
            if error is not None:
                return NormalizeResult(
                    input_kind=InputKind.SEMANTIC_CAPSULE,
                    observation=error,
                    view=None,
                )
            assert mapping is not None
            schema = _versioned_schema(
                mapping.get("schema") or DATASETS_SEMANTIC_CAPSULE_SCHEMA,
                field_name="schema",
            )
            if schema != DATASETS_SEMANTIC_CAPSULE_SCHEMA:
                raise DatasetsAdapterError(
                    f"unsupported semantic capsule schema: {schema}"
                )

            semantic = _cid(
                mapping.get("semantic_state_root_cid"),
                field_name="semantic_state_root_cid",
            )
            repository_tree_id = _optional_text(
                mapping.get("repository_tree_id"),
                field_name="repository_tree_id",
            )
            edges, edge_obs, edge_broader = _normalize_edges(mapping.get("edges"))
            spans = _normalize_spans(mapping.get("spans"))
            contracts = _normalize_contracts(mapping.get("contracts"))
            fixtures = _string_tuple(
                mapping.get("fixture_references")
                or mapping.get("fixture_task_references"),
                field_name="fixture_references",
            )
            truncated = _boolean(mapping.get("truncated"), field_name="truncated")
            observations = list(edge_obs)
            broader = edge_broader or truncated
            if truncated:
                observations.append(
                    _observation(
                        ObservationKind.TRUNCATED,
                        "truncated_capsule",
                        "semantic capsule reports truncated edges",
                    )
                )
            if broader:
                observations.append(
                    _observation(
                        ObservationKind.BROADER_SELECTION_REQUIRED,
                        "uncertain_capsule",
                        "opaque/uncovered/truncated capsule edges force broader selection",
                    )
                )

            raw_receipt_tree = mapping.get("repository_tree_cid")
            if (
                raw_receipt_tree is None
                or raw_receipt_tree == ""
                or raw_receipt_tree == repository_tree_id
            ):
                receipt_tree_cid = ""
            else:
                receipt_tree_cid = _cid(
                    raw_receipt_tree,
                    field_name="repository_tree_cid",
                    required=False,
                )
                if receipt_tree_cid and receipt_tree_cid == repository_tree_id:
                    receipt_tree_cid = ""

            return SemanticCapsuleView(
                schema=schema,
                semantic_state_root_cid=semantic,
                repository_tree_id=repository_tree_id,
                edges=edges,
                spans=spans,
                contracts=contracts,
                fixture_references=fixtures,
                truncated=truncated,
                requires_broader_selection=broader,
                observations=tuple(observations),
                repository_tree_cid=receipt_tree_cid,
            )

        return _soft_normalize(InputKind.SEMANTIC_CAPSULE, _build)

    def normalize_context_pack(self, value: Any) -> NormalizeResult:
        def _build() -> ContextPackView | NormalizeResult:
            mapping, error = self._coerce_to_mapping(
                value, input_kind=InputKind.CONTEXT_PACK
            )
            if error is not None:
                return NormalizeResult(
                    input_kind=InputKind.CONTEXT_PACK,
                    observation=error,
                    view=None,
                )
            assert mapping is not None
            schema = _versioned_schema(
                mapping.get("schema") or DATASETS_CONTEXT_PACK_SCHEMA,
                field_name="schema",
            )
            if schema != DATASETS_CONTEXT_PACK_SCHEMA:
                raise DatasetsAdapterError(
                    f"unsupported context pack schema: {schema}"
                )

            repository_tree_id = _text(
                mapping.get("repository_tree_id"),
                field_name="repository_tree_id",
            )
            semantic = _cid(
                mapping.get("semantic_state_root_cid"),
                field_name="semantic_state_root_cid",
            )
            environment = _cid(
                mapping.get("environment_root_cid"),
                field_name="environment_root_cid",
                required=False,
            )
            lock = _cid(
                mapping.get("dependency_lock_root_cid"),
                field_name="dependency_lock_root_cid",
                required=False,
            )
            token_estimate = _integer(
                mapping.get("token_estimate", mapping.get("tokens", 0)),
                field_name="token_estimate",
                minimum=0,
                maximum=_MAX_TOKEN_ESTIMATE,
                default=0,
            )
            fixtures = _string_tuple(
                mapping.get("fixture_task_references")
                or mapping.get("fixture_references"),
                field_name="fixture_task_references",
            )
            contracts = _normalize_contracts(mapping.get("contracts"))
            raw_receipt_tree = mapping.get("repository_tree_cid")
            if (
                raw_receipt_tree is None
                or raw_receipt_tree == ""
                or raw_receipt_tree == repository_tree_id
            ):
                receipt_tree_cid = ""
            else:
                receipt_tree_cid = _cid(
                    raw_receipt_tree,
                    field_name="repository_tree_cid",
                    required=False,
                )
                if receipt_tree_cid == repository_tree_id:
                    receipt_tree_cid = ""

            observations: list[AdapterObservation] = []
            if not environment or not lock:
                observations.append(
                    _observation(
                        ObservationKind.MISSING_IDENTITY,
                        "incomplete_context_roots",
                        "context pack missing environment or lock root",
                    )
                )

            return ContextPackView(
                schema=schema,
                repository_tree_id=repository_tree_id,
                semantic_state_root_cid=semantic,
                environment_root_cid=environment,
                dependency_lock_root_cid=lock,
                token_estimate=token_estimate,
                fixture_task_references=fixtures,
                contracts=contracts,
                observations=tuple(observations),
                repository_tree_cid=receipt_tree_cid,
            )

        return _soft_normalize(InputKind.CONTEXT_PACK, _build)

    def normalize_impact_index(self, value: Any) -> NormalizeResult:
        """Normalize a CodeEvidence impact index mapping (opaque tree id)."""

        def _build() -> InvalidationPlanView | NormalizeResult:
            if not _is_mapping(value):
                return NormalizeResult(
                    input_kind=InputKind.IMPACT_INDEX,
                    observation=_observation(
                        ObservationKind.UNSUPPORTED,
                        "impact_index_requires_mapping",
                        "impact index must be a strict mapping",
                    ),
                    view=None,
                )
            assert isinstance(value, Mapping)
            schema = _text(
                value.get("schema") or CODE_IMPACT_INDEX_SCHEMA,
                field_name="schema",
                maximum=512,
            )
            if schema != CODE_IMPACT_INDEX_SCHEMA:
                raise DatasetsAdapterError(
                    f"unsupported impact index schema: {schema}"
                )
            repository_tree_id = _text(
                value.get("repository_tree_id"),
                field_name="repository_tree_id",
            )
            # Impact indexes do not carry a semantic CID; missing => observation.
            semantic = _cid(
                value.get("semantic_state_root_cid"),
                field_name="semantic_state_root_cid",
                required=False,
            )
            observations: list[AdapterObservation] = []
            if not semantic:
                observations.append(
                    _observation(
                        ObservationKind.MISSING_IDENTITY,
                        "impact_index_missing_semantic_root",
                        "impact index lacks semantic_state_root_cid",
                    )
                )
            symbol_paths = value.get("symbol_paths") or {}
            if not isinstance(symbol_paths, Mapping):
                raise DatasetsAdapterError("symbol_paths must be a mapping")
            validation_targets = value.get("validation_targets") or {}
            if not isinstance(validation_targets, Mapping):
                raise DatasetsAdapterError("validation_targets must be a mapping")

            # Represent validation targets as tested_by edges (non-authoritative).
            edges: list[DependencyEdgeView] = []
            for validation_id, targets in sorted(validation_targets.items()):
                vid = _text(str(validation_id), field_name="validation_id")
                if not isinstance(targets, Sequence) or isinstance(
                    targets, (str, bytes, bytearray)
                ):
                    raise DatasetsAdapterError(
                        "validation_targets values must be sequences"
                    )
                for target in targets:
                    t = _text(str(target), field_name="validation_target")
                    edges.append(
                        DependencyEdgeView(
                            edge_id=f"validation:{vid}->{t}",
                            source=t,
                            target=vid,
                            kind="tested_by",
                            disposition=EdgeDisposition.EXACT,
                        )
                    )
            edges.sort(key=lambda e: (e.kind, e.source, e.target, e.edge_id))

            return InvalidationPlanView(
                schema=schema,
                repository_tree_id=repository_tree_id,
                semantic_state_root_cid=semantic,
                changed_symbols=(),
                changed_paths=(),
                edges=tuple(edges),
                uncertainty=MappingProxyType(
                    {
                        "source": "code_impact_index",
                        "symbol_count": len(symbol_paths),
                        "validation_count": len(validation_targets),
                    }
                ),
                spans=(),
                contracts=(),
                uncovered_symbols=(),
                uncovered_paths=(),
                truncated=False,
                requires_broader_selection=False,
                observations=tuple(observations),
                repository_tree_cid="",
            )

        return _soft_normalize(InputKind.IMPACT_INDEX, _build)

    def map_validation_ids_to_pytest_nodes(
        self,
        validation_ids: Sequence[str] | None,
        validation_id_to_pytest_node: Mapping[str, Any] | None,
    ) -> ValidationSelectionView:
        """Require exact validation-ID → pytest-node mapping or broaden."""

        ids = _string_tuple(validation_ids, field_name="validation_ids", sort=True)
        observations: list[AdapterObservation] = []
        if not ids:
            return ValidationSelectionView(
                validation_ids=(),
                mapped_pytest_node_ids=(),
                unmapped_validation_ids=(),
                requires_broader_selection=False,
                observations=(),
            )

        mapping: Mapping[str, Any]
        if validation_id_to_pytest_node is None:
            mapping = {}
        elif not isinstance(validation_id_to_pytest_node, Mapping):
            return ValidationSelectionView(
                validation_ids=ids,
                mapped_pytest_node_ids=(),
                unmapped_validation_ids=ids,
                requires_broader_selection=True,
                observations=(
                    _observation(
                        ObservationKind.MALFORMED,
                        "invalid_validation_mapping",
                        "validation_id_to_pytest_node must be a mapping",
                    ),
                ),
            )
        else:
            mapping = validation_id_to_pytest_node

        mapped: list[str] = []
        unmapped: list[str] = []
        for validation_id in ids:
            raw = mapping.get(validation_id)
            if raw is None or raw == "":
                unmapped.append(validation_id)
                continue
            if not isinstance(raw, str):
                unmapped.append(validation_id)
                observations.append(
                    _observation(
                        ObservationKind.MALFORMED,
                        "pytest_node_not_string",
                        f"mapping for {validation_id} is not a string",
                        validation_id=validation_id,
                    )
                )
                continue
            node = raw.strip()
            if not node or "\x00" in node:
                unmapped.append(validation_id)
                observations.append(
                    _observation(
                        ObservationKind.MALFORMED,
                        "malformed_pytest_node",
                        f"mapping for {validation_id} is malformed",
                        validation_id=validation_id,
                    )
                )
                continue
            # Exact pytest node id: path.py::name (parametrized ok).
            if not _PYTEST_NODE_RE.fullmatch(node):
                unmapped.append(validation_id)
                observations.append(
                    _observation(
                        ObservationKind.MALFORMED,
                        "pytest_node_shape_invalid",
                        f"mapping for {validation_id} is not a pytest node id",
                        validation_id=validation_id,
                        pytest_node_id=node,
                    )
                )
                continue
            mapped.append(node)

        broader = bool(unmapped)
        if broader:
            observations.append(
                _observation(
                    ObservationKind.BROADER_SELECTION_REQUIRED,
                    "validation_id_mapping_incomplete",
                    "validation IDs without exact pytest-node mapping force broader selection",
                    unmapped_validation_ids=list(unmapped),
                )
            )

        # Deterministic unique mapped nodes.
        unique_mapped = tuple(sorted(set(mapped)))
        return ValidationSelectionView(
            validation_ids=ids,
            mapped_pytest_node_ids=unique_mapped,
            unmapped_validation_ids=tuple(unmapped),
            requires_broader_selection=broader,
            observations=tuple(observations),
        )

    def cross_check_identity_roots(
        self,
        *,
        repository_state: RepositoryStateView | None,
        invalidation_plan: InvalidationPlanView | None,
        context_pack: ContextPackView | None,
        semantic_capsule: SemanticCapsuleView | None = None,
    ) -> tuple[AdapterObservation, ...]:
        """Cross-check tree/semantic roots across inputs (fail closed)."""

        observations: list[AdapterObservation] = []
        tree_ids = {
            name: getattr(view, "repository_tree_id", "")
            for name, view in (
                ("repository_state", repository_state),
                ("invalidation_plan", invalidation_plan),
                ("context_pack", context_pack),
                ("semantic_capsule", semantic_capsule),
            )
            if view is not None and getattr(view, "repository_tree_id", "")
        }
        if tree_ids:
            unique_trees = set(tree_ids.values())
            if len(unique_trees) > 1:
                observations.append(
                    _observation(
                        ObservationKind.MISSING_IDENTITY,
                        "repository_tree_id_mismatch",
                        "datasets repository_tree_id values disagree across inputs",
                        trees=dict(tree_ids),
                    )
                )

        semantic_ids = {
            name: getattr(view, "semantic_state_root_cid", "")
            for name, view in (
                ("repository_state", repository_state),
                ("invalidation_plan", invalidation_plan),
                ("context_pack", context_pack),
                ("semantic_capsule", semantic_capsule),
            )
            if view is not None and getattr(view, "semantic_state_root_cid", "")
        }
        if semantic_ids:
            unique_semantic = set(semantic_ids.values())
            if len(unique_semantic) > 1:
                observations.append(
                    _observation(
                        ObservationKind.MISSING_IDENTITY,
                        "semantic_state_root_mismatch",
                        "semantic_state_root_cid values disagree across inputs",
                        roots=dict(semantic_ids),
                    )
                )

        # Never allow opaque tree id to be treated as receipt tree cid.
        for name, view in (
            ("repository_state", repository_state),
            ("invalidation_plan", invalidation_plan),
            ("context_pack", context_pack),
            ("semantic_capsule", semantic_capsule),
        ):
            if view is None:
                continue
            tree_id = getattr(view, "repository_tree_id", "")
            tree_cid = getattr(view, "repository_tree_cid", "")
            if tree_id and tree_cid and tree_id == tree_cid:
                observations.append(
                    _observation(
                        ObservationKind.MALFORMED,
                        "tree_id_cid_collision",
                        (
                            f"{name}: repository_tree_id must remain opaque and "
                            "must not equal repository_tree_cid"
                        ),
                    )
                )
        return tuple(observations)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def create_datasets_verification_input_adapter(
    *,
    importer: Callable[[str], Any] | None = None,
) -> DatasetsVerificationInputAdapter:
    return DatasetsVerificationInputAdapter(importer=importer)


__all__ = [
    "ADAPTER_OBSERVATION_SCHEMA",
    "BOUNDED_TOOL_RUNNER_LEAF_MODULE",
    "BOUNDED_TOOL_RUNNER_LEAF_SYMBOL",
    "CODE_EVIDENCE_LEAF_MODULE",
    "CODE_EVIDENCE_LEAF_SYMBOLS",
    "CODE_IMPACT_INDEX_SCHEMA",
    "CODE_IMPACT_RESULT_SCHEMA",
    "CONTEXT_PACK_VIEW_SCHEMA",
    "DATASETS_CANONICAL_TYPES_GAP",
    "DATASETS_CONTEXT_PACK_SCHEMA",
    "DATASETS_DEPENDENCY_EDGE_SCHEMA",
    "DATASETS_INVALIDATION_PLAN_SCHEMA",
    "DATASETS_REPOSITORY_STATE_SCHEMA",
    "DATASETS_SEMANTIC_CAPSULE_SCHEMA",
    "DATASETS_VERIFICATION_INPUT_ADAPTER_INTERFACE",
    "DATASETS_VERIFICATION_INPUT_ADAPTER_VERSION",
    "LEAF_CAPABILITY_SCHEMA",
    "REPOSITORY_STATE_VIEW_SCHEMA",
    "SEMANTIC_CAPSULE_VIEW_SCHEMA",
    "TOP_LEVEL_NAMESPACE_MODULE",
    "VALIDATION_SELECTION_SCHEMA",
    "AdapterObservation",
    "ContextPackProtocol",
    "ContextPackView",
    "DatasetsAdapterError",
    "DatasetsVerificationInputAdapter",
    "DependencyEdgeView",
    "EdgeDisposition",
    "InputKind",
    "InvalidationPlanProtocol",
    "InvalidationPlanView",
    "LeafCapability",
    "NormalizeResult",
    "ObservationKind",
    "RepositoryStateProtocol",
    "RepositoryStateView",
    "SemanticCapsuleProtocol",
    "SemanticCapsuleView",
    "SourceSpanView",
    "ValidationSelectionView",
    "create_datasets_verification_input_adapter",
    "probe_leaf_symbol",
    "probe_top_level_namespace_alone",
]
