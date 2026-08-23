"""Bounded, non-authoritative repository world projection.

``RepositoryWorldState`` is deliberately a planning projection.  It retains
the identities of the admitted ``SupervisorWorldSnapshot`` and the exact
``analysis.repository_snapshot.RepositorySnapshot`` from which it was built,
but it is not a replacement for either authority.  In particular, neither a
world-state projection nor a delta extracted from two projections can grant
authority, establish proof, or establish completion.

The module has a cold import boundary: semantic-state and repository-snapshot
modules are imported only by :class:`RepositoryWorldModel` when a caller asks
to project their already-created records.  Merely importing this module does
not scan a repository, initialize DuckDB, or initialize sibling packages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .contracts import (
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ProcedureBoundsError,
    ProcedureContractError,
    _bounded,
    _decode_fields,
    _enum,
    _identifier,
    _nested,
    _nonnegative_int,
    _schema_name,
    _strings,
    _text,
    _verify_identity,
)

MAX_WORLD_REFERENCES: Final[int] = 256
MAX_WORLD_COUNTER: Final[int] = 2**63 - 1
WORLD_PROJECTION_ROLE: Final[str] = "bounded_planning_projection_only"


class WorldModelError(ProcedureContractError):
    """A repository world projection or delta is malformed."""


class WorldProjectionError(WorldModelError):
    """An authoritative source could not be safely projected."""


class WorldProjectionRole(str, Enum):
    """Closed authority disposition for world-model artifacts."""

    PLANNING_PROJECTION_ONLY = WORLD_PROJECTION_ROLE


class WorldProjectionStatus(str, Enum):
    CURRENT = "current"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"


class ArtifactClass(str, Enum):
    PYTHON_SOURCE = "python_source"
    TEST = "test"
    DOCUMENTATION = "documentation"
    SCHEMA = "schema"
    MANIFEST = "manifest"
    LOCKFILE = "lockfile"
    CONFIGURATION = "configuration"
    GENERATED = "generated"
    OTHER = "other"


class TransitionClass(str, Enum):
    """Closed classes of transitions represented by the initial world model."""

    SOURCE_EDIT = "source_edit"
    SCHEMA_CHANGE = "schema_change"
    DEPENDENCY_CHANGE = "dependency_change"
    TEST_RESULT = "test_result"
    PROOF_RESULT = "proof_result"
    PROVIDER_STATE_CHANGE = "provider_state_change"
    LEASE_ACQUISITION = "lease_acquisition"
    LEASE_EXPIRY = "lease_expiry"
    MERGE = "merge"
    ROLLBACK = "rollback"
    RECEIPT_ADMISSION = "receipt_admission"
    POLICY_CHANGE = "policy_change"
    PROCEDURE_PROMOTION = "procedure_promotion"
    PROCEDURE_REVOCATION = "procedure_revocation"
    UNKNOWN = "unknown"


class WorldDimension(str, Enum):
    """Closed reference dimensions which deterministic extraction may compare."""

    REPOSITORY = "repository"
    WORLD_SNAPSHOT = "world_snapshot"
    REPOSITORY_COMMIT = "repository_commit"
    TREE = "tree"
    REPOSITORY_SNAPSHOT = "repository_snapshot"
    ANALYSIS_HEAD_TREE = "analysis_head_tree"
    ANALYSIS_INDEX_TREE = "analysis_index_tree"
    PACKAGE_GRAPH = "package_graph"
    IMPORT_GRAPH = "import_graph"
    DEPENDENCY_GRAPH = "dependency_graph"
    INTERFACE_GRAPH = "interface_graph"
    EFFECT_GRAPH = "effect_graph"
    ACCEPTANCE_STATE = "acceptance_state"
    TASK_DEPENDENCIES = "task_dependencies"
    PROOF_STATUS = "proof_status"
    TEST_STATUS = "test_status"
    CAPABILITY_STATE = "capability_state"
    PROVIDER_CAPACITY = "provider_capacity"
    WORKTREES = "worktrees"
    LEASES = "leases"
    MERGE_QUEUE = "merge_queue"
    CACHE_STATE = "cache_state"
    ARTIFACT_PRESSURE = "artifact_pressure"
    RESOURCE_BUDGET = "resource_budget"
    PROCEDURE_REGISTRY = "procedure_registry"
    CONTRACT = "contract"
    POLICY = "policy"
    ENVIRONMENT = "environment"


_REFERENCE_FIELDS: Final[tuple[tuple[WorldDimension, str], ...]] = (
    (WorldDimension.REPOSITORY, "repository_reference"),
    (WorldDimension.WORLD_SNAPSHOT, "world_snapshot_cid"),
    (WorldDimension.REPOSITORY_COMMIT, "repository_commit"),
    (WorldDimension.TREE, "tree_id"),
    (WorldDimension.REPOSITORY_SNAPSHOT, "repository_snapshot_id"),
    (WorldDimension.ANALYSIS_HEAD_TREE, "analysis_head_tree_id"),
    (WorldDimension.ANALYSIS_INDEX_TREE, "analysis_index_tree_id"),
    (WorldDimension.PACKAGE_GRAPH, "package_graph_id"),
    (WorldDimension.IMPORT_GRAPH, "import_graph_id"),
    (WorldDimension.DEPENDENCY_GRAPH, "dependency_graph_id"),
    (WorldDimension.INTERFACE_GRAPH, "interface_graph_id"),
    (WorldDimension.EFFECT_GRAPH, "effect_graph_id"),
    (WorldDimension.ACCEPTANCE_STATE, "acceptance_state_id"),
    (WorldDimension.TASK_DEPENDENCIES, "task_dependency_state_id"),
    (WorldDimension.PROOF_STATUS, "proof_status_id"),
    (WorldDimension.TEST_STATUS, "test_status_id"),
    (WorldDimension.CAPABILITY_STATE, "capability_state_id"),
    (WorldDimension.PROVIDER_CAPACITY, "provider_capacity_id"),
    (WorldDimension.MERGE_QUEUE, "merge_queue_id"),
    (WorldDimension.CACHE_STATE, "cache_state_id"),
    (WorldDimension.ARTIFACT_PRESSURE, "artifact_pressure_id"),
    (WorldDimension.RESOURCE_BUDGET, "resource_budget_id"),
    (WorldDimension.PROCEDURE_REGISTRY, "procedure_registry_id"),
    (WorldDimension.CONTRACT, "contract_revision"),
    (WorldDimension.POLICY, "policy_revision"),
    (WorldDimension.ENVIRONMENT, "environment_id"),
)


def _optional_identifier(value: Any, field_name: str) -> str:
    return _identifier(value, field_name, required=False)


def _bounded_counter(value: Any, field_name: str) -> int:
    return _nonnegative_int(value, field_name, maximum=MAX_WORLD_COUNTER)


def _signed_counter(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorldModelError(f"{field_name} must be an integer")
    if not -MAX_WORLD_COUNTER <= value <= MAX_WORLD_COUNTER:
        raise ProcedureBoundsError(f"{field_name} exceeds its numeric bound")
    return value


def _bounded_texts(
    values: Any,
    field_name: str,
    *,
    identifiers: bool = False,
) -> tuple[str, ...]:
    """Return a sorted, unique, bounded sequence without accepting a string."""

    return _strings(
        values,
        field_name,
        limit=MAX_WORLD_REFERENCES,
        identifiers=identifiers,
        preserve_order=False,
    )


def _paths(values: Any, field_name: str) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise WorldModelError(f"{field_name} must be a sequence of paths")
    if len(raw) > MAX_WORLD_REFERENCES:
        raise ProcedureBoundsError(f"{field_name} exceeds its item bound")
    result: set[str] = set()
    for value in raw:
        text = _text(value, field_name)
        candidate = PurePosixPath(text)
        normalized = candidate.as_posix()
        if (
            candidate.is_absolute()
            or text in {"", "."}
            or "\\" in text
            or ".." in candidate.parts
            or normalized != text
        ):
            raise WorldModelError(f"{field_name} must contain canonical repository paths")
        result.add(normalized)
    return tuple(sorted(result))


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


@dataclass(frozen=True)
class RepositoryWorldState(CanonicalContract):
    """Compact planning state bound to, but never replacing, source authorities."""

    SCHEMA: ClassVar[str] = _schema_name("RepositoryWorldState")

    bindings: ArtifactBindings
    world_snapshot_cid: str
    repository_reference: str
    repository_snapshot_id: str
    analysis_head_tree_id: str
    analysis_index_tree_id: str
    changed_files: tuple[str, ...] = ()
    changed_symbols: tuple[str, ...] = ()
    package_graph_id: str = ""
    import_graph_id: str = ""
    dependency_graph_id: str = ""
    interface_graph_id: str = ""
    effect_graph_id: str = ""
    acceptance_state_id: str = ""
    active_task_ids: tuple[str, ...] = ()
    task_dependency_ids: tuple[str, ...] = ()
    task_dependency_state_id: str = ""
    proof_status_id: str = ""
    test_status_id: str = ""
    capability_state_id: str = ""
    provider_capacity_id: str = ""
    worktree_ids: tuple[str, ...] = ()
    lease_ids: tuple[str, ...] = ()
    merge_queue_id: str = ""
    cache_state_id: str = ""
    artifact_pressure_id: str = ""
    token_budget_remaining: int = 0
    resource_budget_id: str = ""
    known_failure_signature_ids: tuple[str, ...] = ()
    procedure_registry_revision: int = 0
    procedure_registry_id: str = ""
    source_evidence_ids: tuple[str, ...] = ()
    unavailable_dimensions: tuple[str, ...] = ()
    projection_status: WorldProjectionStatus = WorldProjectionStatus.CURRENT
    projection_role: WorldProjectionRole = WorldProjectionRole.PLANNING_PROJECTION_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "world_snapshot_cid",
            "repository_reference",
            "repository_snapshot_id",
            "analysis_head_tree_id",
            "analysis_index_tree_id",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.bindings.tree_id != self.tree_id:
            raise WorldModelError("bindings.tree_id must match the projected tree_id")
        object.__setattr__(self, "changed_files", _paths(self.changed_files, "changed_files"))
        object.__setattr__(
            self,
            "changed_symbols",
            _bounded_texts(self.changed_symbols, "changed_symbols"),
        )
        for name in (
            "package_graph_id",
            "import_graph_id",
            "dependency_graph_id",
            "interface_graph_id",
            "effect_graph_id",
            "acceptance_state_id",
            "task_dependency_state_id",
            "proof_status_id",
            "test_status_id",
            "capability_state_id",
            "provider_capacity_id",
            "merge_queue_id",
            "cache_state_id",
            "artifact_pressure_id",
            "resource_budget_id",
            "procedure_registry_id",
        ):
            object.__setattr__(self, name, _optional_identifier(getattr(self, name), name))
        for name in (
            "active_task_ids",
            "task_dependency_ids",
            "worktree_ids",
            "lease_ids",
            "known_failure_signature_ids",
            "source_evidence_ids",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_texts(getattr(self, name), name, identifiers=True),
            )
        unavailable = _bounded_texts(
            self.unavailable_dimensions,
            "unavailable_dimensions",
        )
        allowed_dimensions = {item.value for item in WorldDimension}
        if set(unavailable).difference(allowed_dimensions):
            raise WorldModelError("unavailable_dimensions contains an unknown dimension")
        object.__setattr__(self, "unavailable_dimensions", unavailable)
        object.__setattr__(
            self,
            "token_budget_remaining",
            _bounded_counter(self.token_budget_remaining, "token_budget_remaining"),
        )
        object.__setattr__(
            self,
            "procedure_registry_revision",
            _bounded_counter(self.procedure_registry_revision, "procedure_registry_revision"),
        )
        object.__setattr__(
            self,
            "projection_status",
            _enum(self.projection_status, WorldProjectionStatus, "projection_status"),
        )
        object.__setattr__(
            self,
            "projection_role",
            _enum(self.projection_role, WorldProjectionRole, "projection_role"),
        )
        if self.projection_role is not WorldProjectionRole.PLANNING_PROJECTION_ONLY:
            raise WorldModelError("repository world state cannot become an authority")
        _bounded(self, "RepositoryWorldState")

    @property
    def repository_id(self) -> str:
        return self.bindings.repository_id

    @property
    def repository_commit(self) -> str:
        return self.bindings.repository_commit

    @property
    def tree_id(self) -> str:
        return self.bindings.tree_id

    @property
    def objective_id(self) -> str:
        return self.bindings.objective_id

    @property
    def task_id(self) -> str:
        return self.bindings.task_id

    @property
    def contract_revision(self) -> str:
        return self.bindings.contract_revision

    @property
    def policy_revision(self) -> str:
        return self.bindings.policy_revision

    @property
    def environment_id(self) -> str:
        return self.bindings.environment_id

    @property
    def is_authoritative(self) -> bool:
        return False

    @property
    def can_grant_authority(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "world_snapshot_cid": self.world_snapshot_cid,
            "repository_reference": self.repository_reference,
            "repository_snapshot_id": self.repository_snapshot_id,
            "analysis_head_tree_id": self.analysis_head_tree_id,
            "analysis_index_tree_id": self.analysis_index_tree_id,
            "changed_files": self.changed_files,
            "changed_symbols": self.changed_symbols,
            "package_graph_id": self.package_graph_id,
            "import_graph_id": self.import_graph_id,
            "dependency_graph_id": self.dependency_graph_id,
            "interface_graph_id": self.interface_graph_id,
            "effect_graph_id": self.effect_graph_id,
            "acceptance_state_id": self.acceptance_state_id,
            "active_task_ids": self.active_task_ids,
            "task_dependency_ids": self.task_dependency_ids,
            "task_dependency_state_id": self.task_dependency_state_id,
            "proof_status_id": self.proof_status_id,
            "test_status_id": self.test_status_id,
            "capability_state_id": self.capability_state_id,
            "provider_capacity_id": self.provider_capacity_id,
            "worktree_ids": self.worktree_ids,
            "lease_ids": self.lease_ids,
            "merge_queue_id": self.merge_queue_id,
            "cache_state_id": self.cache_state_id,
            "artifact_pressure_id": self.artifact_pressure_id,
            "token_budget_remaining": self.token_budget_remaining,
            "resource_budget_id": self.resource_budget_id,
            "known_failure_signature_ids": self.known_failure_signature_ids,
            "procedure_registry_revision": self.procedure_registry_revision,
            "procedure_registry_id": self.procedure_registry_id,
            "source_evidence_ids": self.source_evidence_ids,
            "unavailable_dimensions": self.unavailable_dimensions,
            "projection_status": self.projection_status.value,
            "projection_role": self.projection_role.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepositoryWorldState":
        fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        fields = tuple(name for name in fields if name != "SCHEMA")
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class AbstractRepositoryState(CanonicalContract):
    """Bounded task-family abstraction which retains all authority bindings."""

    SCHEMA: ClassVar[str] = _schema_name("AbstractRepositoryState")

    bindings: ArtifactBindings
    source_world_state_id: str
    changed_file_classes: tuple[ArtifactClass, ...] = ()
    changed_file_count: int = 0
    changed_symbol_count: int = 0
    active_task_count: int = 0
    dependency_count: int = 0
    known_effect_classes: tuple[str, ...] = ()
    known_failure_family_ids: tuple[str, ...] = ()
    proof_status_id: str = ""
    test_status_id: str = ""
    capability_state_id: str = ""
    provider_capacity_id: str = ""
    resource_budget_id: str = ""
    procedure_registry_revision: int = 0
    projection_role: WorldProjectionRole = WorldProjectionRole.PLANNING_PROJECTION_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(
            self,
            "source_world_state_id",
            _identifier(self.source_world_state_id, "source_world_state_id"),
        )
        classes: list[ArtifactClass] = []
        for value in self.changed_file_classes:
            item = _enum(value, ArtifactClass, "changed_file_classes")
            if item not in classes:
                classes.append(item)
        object.__setattr__(
            self, "changed_file_classes", tuple(sorted(classes, key=lambda item: item.value))
        )
        for name in (
            "changed_file_count",
            "changed_symbol_count",
            "active_task_count",
            "dependency_count",
            "procedure_registry_revision",
        ):
            object.__setattr__(self, name, _bounded_counter(getattr(self, name), name))
        for name in ("known_effect_classes", "known_failure_family_ids"):
            object.__setattr__(
                self, name, _bounded_texts(getattr(self, name), name, identifiers=True)
            )
        for name in (
            "proof_status_id",
            "test_status_id",
            "capability_state_id",
            "provider_capacity_id",
            "resource_budget_id",
        ):
            object.__setattr__(self, name, _optional_identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "projection_role",
            _enum(self.projection_role, WorldProjectionRole, "projection_role"),
        )
        if self.projection_role is not WorldProjectionRole.PLANNING_PROJECTION_ONLY:
            raise WorldModelError("abstract state cannot become an authority")
        _bounded(self, "AbstractRepositoryState")

    @property
    def is_authoritative(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "source_world_state_id": self.source_world_state_id,
            "changed_file_classes": tuple(item.value for item in self.changed_file_classes),
            "changed_file_count": self.changed_file_count,
            "changed_symbol_count": self.changed_symbol_count,
            "active_task_count": self.active_task_count,
            "dependency_count": self.dependency_count,
            "known_effect_classes": self.known_effect_classes,
            "known_failure_family_ids": self.known_failure_family_ids,
            "proof_status_id": self.proof_status_id,
            "test_status_id": self.test_status_id,
            "capability_state_id": self.capability_state_id,
            "provider_capacity_id": self.provider_capacity_id,
            "resource_budget_id": self.resource_budget_id,
            "procedure_registry_revision": self.procedure_registry_revision,
            "projection_role": self.projection_role.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AbstractRepositoryState":
        fields = (
            "bindings",
            "source_world_state_id",
            "changed_file_classes",
            "changed_file_count",
            "changed_symbol_count",
            "active_task_count",
            "dependency_count",
            "known_effect_classes",
            "known_failure_family_ids",
            "proof_status_id",
            "test_status_id",
            "capability_state_id",
            "provider_capacity_id",
            "resource_budget_id",
            "procedure_registry_revision",
            "projection_role",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class StateReferenceChange(CanonicalContract):
    """One closed world-reference comparison, never an open mapping."""

    SCHEMA: ClassVar[str] = _schema_name("StateReferenceChange")

    dimension: WorldDimension
    before_id: str = ""
    after_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _enum(self.dimension, WorldDimension, "dimension"))
        for name in ("before_id", "after_id"):
            object.__setattr__(self, name, _optional_identifier(getattr(self, name), name))
        if self.before_id == self.after_id:
            raise WorldModelError("a state reference change must change its value")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "dimension": self.dimension.value,
            "before_id": self.before_id,
            "after_id": self.after_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateReferenceChange":
        fields = ("dimension", "before_id", "after_id")
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


def _reference_changes(values: Any) -> tuple[StateReferenceChange, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray, memoryview)
    ):
        raw = values
    else:
        raise WorldModelError("reference_changes must be a sequence")
    if len(raw) > len(WorldDimension):
        raise ProcedureBoundsError("reference_changes exceeds its closed dimension bound")
    items = tuple(
        item
        if isinstance(item, StateReferenceChange)
        else StateReferenceChange.from_dict(item)
        if isinstance(item, Mapping) and "schema" in item
        else StateReferenceChange(**item)
        if isinstance(item, Mapping)
        else None
        for item in raw
    )
    if any(item is None for item in items):
        raise WorldModelError("reference_changes contains a malformed record")
    typed = tuple(item for item in items if isinstance(item, StateReferenceChange))
    dimensions = [item.dimension for item in typed]
    if len(dimensions) != len(set(dimensions)):
        raise WorldModelError("reference_changes contains duplicate dimensions")
    return tuple(sorted(typed, key=lambda item: item.dimension.value))


@dataclass(frozen=True)
class WorldStateDelta(CanonicalContract):
    """Exact deterministic comparison of two bounded world projections."""

    SCHEMA: ClassVar[str] = _schema_name("WorldStateDelta")

    bindings: ArtifactBindings
    before_state_id: str
    after_state_id: str
    transition_class: TransitionClass
    added_changed_files: tuple[str, ...] = ()
    removed_changed_files: tuple[str, ...] = ()
    added_changed_symbols: tuple[str, ...] = ()
    removed_changed_symbols: tuple[str, ...] = ()
    added_active_task_ids: tuple[str, ...] = ()
    removed_active_task_ids: tuple[str, ...] = ()
    added_task_dependency_ids: tuple[str, ...] = ()
    removed_task_dependency_ids: tuple[str, ...] = ()
    added_worktree_ids: tuple[str, ...] = ()
    removed_worktree_ids: tuple[str, ...] = ()
    added_lease_ids: tuple[str, ...] = ()
    removed_lease_ids: tuple[str, ...] = ()
    added_failure_signature_ids: tuple[str, ...] = ()
    removed_failure_signature_ids: tuple[str, ...] = ()
    reference_changes: tuple[StateReferenceChange, ...] = ()
    token_budget_delta: int = 0
    before_registry_revision: int = 0
    after_registry_revision: int = 0
    evidence_ids: tuple[str, ...] = ()
    projection_role: WorldProjectionRole = WorldProjectionRole.PLANNING_PROJECTION_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("before_state_id", "after_state_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "transition_class",
            _enum(self.transition_class, TransitionClass, "transition_class"),
        )
        for name in ("added_changed_files", "removed_changed_files"):
            object.__setattr__(self, name, _paths(getattr(self, name), name))
        for name in ("added_changed_symbols", "removed_changed_symbols"):
            object.__setattr__(self, name, _bounded_texts(getattr(self, name), name))
        for name in (
            "added_active_task_ids",
            "removed_active_task_ids",
            "added_task_dependency_ids",
            "removed_task_dependency_ids",
            "added_worktree_ids",
            "removed_worktree_ids",
            "added_lease_ids",
            "removed_lease_ids",
            "added_failure_signature_ids",
            "removed_failure_signature_ids",
            "evidence_ids",
        ):
            object.__setattr__(
                self, name, _bounded_texts(getattr(self, name), name, identifiers=True)
            )
        object.__setattr__(self, "reference_changes", _reference_changes(self.reference_changes))
        object.__setattr__(
            self,
            "token_budget_delta",
            _signed_counter(self.token_budget_delta, "token_budget_delta"),
        )
        for name in ("before_registry_revision", "after_registry_revision"):
            object.__setattr__(self, name, _bounded_counter(getattr(self, name), name))
        object.__setattr__(
            self,
            "projection_role",
            _enum(self.projection_role, WorldProjectionRole, "projection_role"),
        )
        if self.projection_role is not WorldProjectionRole.PLANNING_PROJECTION_ONLY:
            raise WorldModelError("a world delta cannot become an authority")
        _bounded(self, "WorldStateDelta")

    @property
    def is_authoritative(self) -> bool:
        return False

    @property
    def changed_file_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.added_changed_files) | set(self.removed_changed_files)))

    @property
    def changed_symbol_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.added_changed_symbols) | set(self.removed_changed_symbols)))

    @property
    def has_changes(self) -> bool:
        return bool(
            self.changed_file_ids
            or self.changed_symbol_ids
            or self.reference_changes
            or self.added_active_task_ids
            or self.removed_active_task_ids
            or self.added_task_dependency_ids
            or self.removed_task_dependency_ids
            or self.added_worktree_ids
            or self.removed_worktree_ids
            or self.added_lease_ids
            or self.removed_lease_ids
            or self.added_failure_signature_ids
            or self.removed_failure_signature_ids
            or self.token_budget_delta
            or self.before_registry_revision != self.after_registry_revision
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "before_state_id": self.before_state_id,
            "after_state_id": self.after_state_id,
            "transition_class": self.transition_class.value,
            "added_changed_files": self.added_changed_files,
            "removed_changed_files": self.removed_changed_files,
            "added_changed_symbols": self.added_changed_symbols,
            "removed_changed_symbols": self.removed_changed_symbols,
            "added_active_task_ids": self.added_active_task_ids,
            "removed_active_task_ids": self.removed_active_task_ids,
            "added_task_dependency_ids": self.added_task_dependency_ids,
            "removed_task_dependency_ids": self.removed_task_dependency_ids,
            "added_worktree_ids": self.added_worktree_ids,
            "removed_worktree_ids": self.removed_worktree_ids,
            "added_lease_ids": self.added_lease_ids,
            "removed_lease_ids": self.removed_lease_ids,
            "added_failure_signature_ids": self.added_failure_signature_ids,
            "removed_failure_signature_ids": self.removed_failure_signature_ids,
            "reference_changes": self.reference_changes,
            "token_budget_delta": self.token_budget_delta,
            "before_registry_revision": self.before_registry_revision,
            "after_registry_revision": self.after_registry_revision,
            "evidence_ids": self.evidence_ids,
            "projection_role": self.projection_role.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorldStateDelta":
        fields = (
            "bindings",
            "before_state_id",
            "after_state_id",
            "transition_class",
            "added_changed_files",
            "removed_changed_files",
            "added_changed_symbols",
            "removed_changed_symbols",
            "added_active_task_ids",
            "removed_active_task_ids",
            "added_task_dependency_ids",
            "removed_task_dependency_ids",
            "added_worktree_ids",
            "removed_worktree_ids",
            "added_lease_ids",
            "removed_lease_ids",
            "added_failure_signature_ids",
            "removed_failure_signature_ids",
            "reference_changes",
            "token_budget_delta",
            "before_registry_revision",
            "after_registry_revision",
            "evidence_ids",
            "projection_role",
        )
        values = _decode_fields(payload, cls.SCHEMA, fields, cls.__name__)
        if "bindings" in values:
            values["bindings"] = _bindings(values["bindings"])
        if "reference_changes" in values:
            values["reference_changes"] = _reference_changes(values["reference_changes"])
        record = cls(**values)
        _verify_identity(payload, record)
        return record


def classify_artifact_path(path: str) -> ArtifactClass:
    """Deterministically classify a repository-relative path for abstraction."""

    normalized = _paths((path,), "path")[0]
    pure = PurePosixPath(normalized)
    name = pure.name.lower()
    suffix = pure.suffix.lower()
    parts = {part.lower() for part in pure.parts}
    if "test" in parts or "tests" in parts or name.startswith("test_"):
        return ArtifactClass.TEST
    if "generated" in parts or name.endswith("_generated.py"):
        return ArtifactClass.GENERATED
    if suffix == ".py":
        return ArtifactClass.PYTHON_SOURCE
    if suffix in {".md", ".rst", ".txt"}:
        return ArtifactClass.DOCUMENTATION
    if suffix in {".jsonschema", ".proto", ".avsc"} or "schema" in name:
        return ArtifactClass.SCHEMA
    if name in {"uv.lock", "poetry.lock", "package-lock.json", "cargo.lock"}:
        return ArtifactClass.LOCKFILE
    if name in {"pyproject.toml", "package.json", "cargo.toml", "setup.cfg"}:
        return ArtifactClass.MANIFEST
    if suffix in {".toml", ".yaml", ".yml", ".ini", ".cfg"}:
        return ArtifactClass.CONFIGURATION
    return ArtifactClass.OTHER


def abstract_repository_state(
    state: RepositoryWorldState,
    *,
    known_effect_classes: Sequence[str] = (),
    known_failure_family_ids: Sequence[str] = (),
) -> AbstractRepositoryState:
    if not isinstance(state, RepositoryWorldState):
        raise WorldModelError("state must be a RepositoryWorldState")
    classes = tuple(classify_artifact_path(path) for path in state.changed_files)
    return AbstractRepositoryState(
        bindings=state.bindings,
        source_world_state_id=state.content_id,
        changed_file_classes=classes,
        changed_file_count=len(state.changed_files),
        changed_symbol_count=len(state.changed_symbols),
        active_task_count=len(state.active_task_ids),
        dependency_count=len(state.task_dependency_ids),
        known_effect_classes=tuple(known_effect_classes),
        known_failure_family_ids=tuple(known_failure_family_ids),
        proof_status_id=state.proof_status_id,
        test_status_id=state.test_status_id,
        capability_state_id=state.capability_state_id,
        provider_capacity_id=state.provider_capacity_id,
        resource_budget_id=state.resource_budget_id,
        procedure_registry_revision=state.procedure_registry_revision,
    )


def _set_delta(
    before: Sequence[str], after: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    before_set = set(before)
    after_set = set(after)
    return tuple(sorted(after_set - before_set)), tuple(sorted(before_set - after_set))


def infer_transition_class(
    before: RepositoryWorldState,
    after: RepositoryWorldState,
) -> TransitionClass:
    """Return a deterministic conservative class, abstaining when ambiguous."""

    if before.bindings.policy_revision != after.bindings.policy_revision:
        return TransitionClass.POLICY_CHANGE
    if before.procedure_registry_revision != after.procedure_registry_revision:
        if after.procedure_registry_revision > before.procedure_registry_revision:
            return TransitionClass.PROCEDURE_PROMOTION
        return TransitionClass.PROCEDURE_REVOCATION
    added_leases, removed_leases = _set_delta(before.lease_ids, after.lease_ids)
    if added_leases and not removed_leases:
        return TransitionClass.LEASE_ACQUISITION
    if removed_leases and not added_leases:
        return TransitionClass.LEASE_EXPIRY
    if before.provider_capacity_id != after.provider_capacity_id:
        return TransitionClass.PROVIDER_STATE_CHANGE
    if before.proof_status_id != after.proof_status_id:
        return TransitionClass.PROOF_RESULT
    if before.test_status_id != after.test_status_id:
        return TransitionClass.TEST_RESULT
    if before.dependency_graph_id != after.dependency_graph_id:
        return TransitionClass.DEPENDENCY_CHANGE
    if before.bindings.contract_revision != after.bindings.contract_revision:
        return TransitionClass.SCHEMA_CHANGE
    if before.tree_id != after.tree_id:
        if before.merge_queue_id != after.merge_queue_id:
            return TransitionClass.MERGE
        return TransitionClass.SOURCE_EDIT
    if before.cache_state_id != after.cache_state_id:
        return TransitionClass.RECEIPT_ADMISSION
    return TransitionClass.UNKNOWN


def extract_world_state_delta(
    before: RepositoryWorldState,
    after: RepositoryWorldState,
    *,
    transition_class: TransitionClass | str | None = None,
    evidence_ids: Sequence[str] = (),
) -> WorldStateDelta:
    """Compare two projections deterministically without asserting real effects."""

    if not isinstance(before, RepositoryWorldState) or not isinstance(after, RepositoryWorldState):
        raise WorldModelError("before and after must be RepositoryWorldState records")
    if before.repository_id != after.repository_id:
        raise WorldModelError("world-state delta cannot cross repository identities")
    if before.objective_id != after.objective_id or before.task_id != after.task_id:
        raise WorldModelError("world-state delta cannot cross objective or task bindings")
    reference_changes = tuple(
        StateReferenceChange(
            dimension=dimension,
            before_id=getattr(before, field_name),
            after_id=getattr(after, field_name),
        )
        for dimension, field_name in _REFERENCE_FIELDS
        if getattr(before, field_name) != getattr(after, field_name)
    )
    added_files, removed_files = _set_delta(before.changed_files, after.changed_files)
    added_symbols, removed_symbols = _set_delta(before.changed_symbols, after.changed_symbols)
    added_tasks, removed_tasks = _set_delta(before.active_task_ids, after.active_task_ids)
    added_dependencies, removed_dependencies = _set_delta(
        before.task_dependency_ids, after.task_dependency_ids
    )
    added_worktrees, removed_worktrees = _set_delta(before.worktree_ids, after.worktree_ids)
    added_leases, removed_leases = _set_delta(before.lease_ids, after.lease_ids)
    added_failures, removed_failures = _set_delta(
        before.known_failure_signature_ids, after.known_failure_signature_ids
    )
    chosen_class = (
        infer_transition_class(before, after)
        if transition_class is None
        else _enum(transition_class, TransitionClass, "transition_class")
    )
    return WorldStateDelta(
        # Delta bindings represent the source state.  The after-state identity
        # is retained independently and must be checked before application.
        bindings=before.bindings,
        before_state_id=before.content_id,
        after_state_id=after.content_id,
        transition_class=chosen_class,
        added_changed_files=added_files,
        removed_changed_files=removed_files,
        added_changed_symbols=added_symbols,
        removed_changed_symbols=removed_symbols,
        added_active_task_ids=added_tasks,
        removed_active_task_ids=removed_tasks,
        added_task_dependency_ids=added_dependencies,
        removed_task_dependency_ids=removed_dependencies,
        added_worktree_ids=added_worktrees,
        removed_worktree_ids=removed_worktrees,
        added_lease_ids=added_leases,
        removed_lease_ids=removed_leases,
        added_failure_signature_ids=added_failures,
        removed_failure_signature_ids=removed_failures,
        reference_changes=reference_changes,
        token_budget_delta=after.token_budget_remaining - before.token_budget_remaining,
        before_registry_revision=before.procedure_registry_revision,
        after_registry_revision=after.procedure_registry_revision,
        evidence_ids=tuple(evidence_ids),
    )


def _component(snapshot: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    components = snapshot.get("components")
    if not isinstance(components, Mapping) or not isinstance(components.get(name), Mapping):
        raise WorldProjectionError(f"world snapshot is missing component {name}")
    return components[name]


def _snapshot_value(snapshot: Any, name: str) -> Any:
    if isinstance(snapshot, Mapping):
        return snapshot.get(name)
    return getattr(snapshot, name, None)


class RepositoryWorldModel:
    """Pure projector and comparator over independently admitted source records."""

    projection_role: Final[WorldProjectionRole] = WorldProjectionRole.PLANNING_PROJECTION_ONLY

    @staticmethod
    def project(
        world_snapshot: Mapping[str, Any],
        repository_snapshot: Any,
        *,
        objective_id: str,
        task_id: str,
        changed_symbols: Sequence[str] = (),
        package_graph_id: str = "",
        import_graph_id: str = "",
        dependency_graph_id: str = "",
        interface_graph_id: str = "",
        effect_graph_id: str = "",
        active_task_ids: Sequence[str] = (),
        task_dependency_ids: Sequence[str] = (),
        task_dependency_state_id: str = "",
        proof_status_id: str = "",
        test_status_id: str = "",
        provider_capacity_id: str = "",
        worktree_ids: Sequence[str] = (),
        lease_ids: Sequence[str] = (),
        cache_state_id: str = "",
        artifact_pressure_id: str = "",
        token_budget_remaining: int = 0,
        known_failure_signature_ids: Sequence[str] = (),
        procedure_registry_revision: int = 0,
        procedure_registry_id: str = "",
        source_evidence_ids: Sequence[str] = (),
    ) -> RepositoryWorldState:
        """Project the two existing authority records without creating authority.

        ``world_snapshot`` may be either ``SupervisorWorldSnapshot@1`` itself
        or the wrapper returned by ``WorldSnapshotBuilder@1``.  The repository
        input is consumed through the public identity attributes of
        ``analysis.repository_snapshot.RepositorySnapshot``; a canonical
        mapping with the same fields is also accepted for durable replay.
        """

        # Lazy imports keep import/collection hermetic when semantic-state or
        # analysis sibling packages are absent or intentionally uninitialized.
        try:
            from ..semantic_state.world_snapshot_contracts import parse_world_snapshot
        except ImportError as exc:  # pragma: no cover - qualified trees include it
            raise WorldProjectionError("SupervisorWorldSnapshot authority is unavailable") from exc

        candidate = world_snapshot.get("snapshot", world_snapshot)
        try:
            admitted = parse_world_snapshot(candidate)
        except (TypeError, ValueError) as exc:
            raise WorldProjectionError("world snapshot did not pass its canonical parser") from exc

        repository_snapshot_id = _snapshot_value(repository_snapshot, "snapshot_id")
        if not repository_snapshot_id:
            repository_snapshot_id = _snapshot_value(repository_snapshot, "repository_snapshot_id")
        head_commit_id = _snapshot_value(repository_snapshot, "head_commit_id")
        head_tree_id = _snapshot_value(repository_snapshot, "head_tree_id")
        index_tree_id = _snapshot_value(repository_snapshot, "index_tree_id")
        if not all((repository_snapshot_id, head_commit_id, head_tree_id, index_tree_id)):
            raise WorldProjectionError(
                "repository snapshot must expose snapshot, commit, head-tree, "
                "and index-tree identities"
            )

        dispositions = _snapshot_value(repository_snapshot, "dispositions") or ()
        if isinstance(dispositions, (str, bytes, bytearray)) or not isinstance(
            dispositions, Sequence
        ):
            raise WorldProjectionError("repository snapshot dispositions must be a sequence")
        if len(dispositions) > MAX_WORLD_REFERENCES * 128:
            raise ProcedureBoundsError("repository disposition projection exceeds its scan bound")
        changed_files: list[str] = []
        for item in dispositions:
            status = _snapshot_value(item, "git_status")
            status_value = getattr(status, "value", status)
            path = _snapshot_value(item, "path")
            if status_value not in (None, "", "clean") and path:
                changed_files.append(str(path))
        if len(changed_files) > MAX_WORLD_REFERENCES:
            raise ProcedureBoundsError("changed-file projection exceeds its item bound")

        repository_component = _component(admitted, "repository")
        tree_component = _component(admitted, "repository_tree")
        component_statuses = {
            name: str(_component(admitted, name).get("status") or "")
            for name in (
                "repository",
                "repository_tree",
                "objectives",
                "contract_root",
                "environment_bindings",
                "policy_root",
            )
        }
        status = (
            WorldProjectionStatus.CURRENT
            if all(value == "current" for value in component_statuses.values())
            else WorldProjectionStatus.INCOMPLETE
        )
        unavailable: list[str] = []
        optional_refs = {
            WorldDimension.PACKAGE_GRAPH: package_graph_id,
            WorldDimension.IMPORT_GRAPH: import_graph_id,
            WorldDimension.DEPENDENCY_GRAPH: dependency_graph_id,
            WorldDimension.INTERFACE_GRAPH: interface_graph_id,
            WorldDimension.EFFECT_GRAPH: effect_graph_id,
            WorldDimension.PROOF_STATUS: proof_status_id,
            WorldDimension.TEST_STATUS: test_status_id,
            WorldDimension.PROVIDER_CAPACITY: provider_capacity_id,
            WorldDimension.CACHE_STATE: cache_state_id,
            WorldDimension.ARTIFACT_PRESSURE: artifact_pressure_id,
            WorldDimension.PROCEDURE_REGISTRY: procedure_registry_id,
        }
        unavailable.extend(
            dimension.value for dimension, value in optional_refs.items() if not value
        )

        components = admitted["components"]
        bindings = ArtifactBindings(
            repository_id=str(admitted["repository_id"]),
            repository_commit=str(head_commit_id),
            tree_id=str(tree_component["cid"]),
            objective_id=objective_id,
            task_id=task_id,
            contract_revision=str(components["contract_root"]["cid"]),
            policy_revision=str(components["policy_root"]["cid"]),
            environment_id=str(components["environment_bindings"]["cid"]),
        )
        evidence = tuple(source_evidence_ids) + (
            str(admitted["snapshot_cid"]),
            str(repository_snapshot_id),
        )
        return RepositoryWorldState(
            bindings=bindings,
            world_snapshot_cid=str(admitted["snapshot_cid"]),
            repository_reference=str(repository_component["cid"]),
            repository_snapshot_id=str(repository_snapshot_id),
            analysis_head_tree_id=str(head_tree_id),
            analysis_index_tree_id=str(index_tree_id),
            changed_files=tuple(changed_files),
            changed_symbols=tuple(changed_symbols),
            package_graph_id=package_graph_id,
            import_graph_id=import_graph_id,
            dependency_graph_id=dependency_graph_id,
            interface_graph_id=interface_graph_id,
            effect_graph_id=effect_graph_id,
            acceptance_state_id=str(components["completion_root"]["cid"]),
            active_task_ids=tuple(active_task_ids),
            task_dependency_ids=tuple(task_dependency_ids),
            task_dependency_state_id=(
                task_dependency_state_id or str(components["accepted_plan_root"]["cid"])
            ),
            proof_status_id=proof_status_id,
            test_status_id=test_status_id,
            capability_state_id=str(components["capability_snapshot"]["cid"]),
            provider_capacity_id=provider_capacity_id,
            worktree_ids=tuple(worktree_ids),
            lease_ids=tuple(lease_ids),
            merge_queue_id=str(components["merge_root"]["cid"]),
            cache_state_id=cache_state_id,
            artifact_pressure_id=artifact_pressure_id,
            token_budget_remaining=token_budget_remaining,
            resource_budget_id=str(components["resource_snapshot"]["cid"]),
            known_failure_signature_ids=tuple(known_failure_signature_ids),
            procedure_registry_revision=procedure_registry_revision,
            procedure_registry_id=procedure_registry_id,
            source_evidence_ids=evidence,
            unavailable_dimensions=tuple(unavailable),
            projection_status=status,
        )

    @staticmethod
    def abstract(
        state: RepositoryWorldState,
        *,
        known_effect_classes: Sequence[str] = (),
        known_failure_family_ids: Sequence[str] = (),
    ) -> AbstractRepositoryState:
        return abstract_repository_state(
            state,
            known_effect_classes=known_effect_classes,
            known_failure_family_ids=known_failure_family_ids,
        )

    @staticmethod
    def delta(
        before: RepositoryWorldState,
        after: RepositoryWorldState,
        *,
        transition_class: TransitionClass | str | None = None,
        evidence_ids: Sequence[str] = (),
    ) -> WorldStateDelta:
        return extract_world_state_delta(
            before,
            after,
            transition_class=transition_class,
            evidence_ids=evidence_ids,
        )


__all__ = [
    "AbstractRepositoryState",
    "ArtifactClass",
    "RepositoryWorldModel",
    "RepositoryWorldState",
    "StateReferenceChange",
    "TransitionClass",
    "WorldDimension",
    "WorldModelError",
    "WorldProjectionError",
    "WorldProjectionRole",
    "WorldProjectionStatus",
    "WorldStateDelta",
    "abstract_repository_state",
    "classify_artifact_path",
    "extract_world_state_delta",
    "infer_transition_class",
]
