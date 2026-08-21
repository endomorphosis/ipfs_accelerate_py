"""State-ownership model for mutable semantic facts (PCAR-014).

`StateOwnershipModel` classifies DuckDB tables, JSON and Markdown files,
in-memory registries, events, caches, worktree metadata, leases, and
provider/goal/task/completion/receipt state as `authoritative`,
`materialized_projection`, `cache`, `historical_event`, `fixture`,
`legacy`, or `unknown`. Every mutable semantic fact has exactly one
authoritative store or a typed hard conflict. Unknown ownership is never
accepted as resolved. Projections and caches are rebuildable and cannot
satisfy production state authority. Migration uses a closed, bounded phase
sequence and cannot grant authority or leave indefinite dual ownership.

The model records existing store authority. It cannot mutate stores, grant
authority, or authorize code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ipfs_accelerate_py.utils.cid_utils import (
    canonical_dag_json_bytes,
    cid_for_dag_json,
    validate_cid,
)

from .contracts import (
    ArchitectureContractError,
    Confidence,
    SourceFactIdentity,
    SourceSpan,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
)

STATE_OWNERSHIP_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/state-ownership-model@1"
)
STATE_OWNERSHIP_VERSION = 1
STATE_OWNERSHIP_EVIDENCE = "pcar/state-ownership-model@1"
STATE_ITEM_SCHEMA = "ipfs_accelerate_py/agent-supervisor/state-item@1"
STATE_ITEM_VERSION = 1
STATE_CONFLICT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/state-conflict@1"
STATE_CONFLICT_VERSION = 1
STATE_MIGRATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/state-migration-plan@1"
)
STATE_MIGRATION_VERSION = 1
EXTRACTOR_IDENTITY = "pcar-014-state-ownership-model"
TASK_ID = "PCAR-014"
DEFAULT_FRESHNESS = "pcar-014-state-ownership"
DEFAULT_REPOSITORY_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
DUCKDB_CONTROL_STORE_ID = "duckdb:control"
DUCKDB_CONTROL_LOCATOR = "{state_root}/control.duckdb"
EFFECT_CLASS = "read_only_analysis_and_immutable_migration_proposals"
CAN_AUTHORIZE_CHANGES = False
CAN_GRANT_AUTHORITY = False
CAN_CREATE_INDEFINITE_DUAL_OWNERSHIP = False
MARKDOWN_IS_NOT_AUTHORITY = True
DUCKLAKE_IS_NOT_AUTHORITY = True
PROJECTION_IS_NOT_AUTHORITY = True
CACHE_IS_NOT_AUTHORITY = True
UNKNOWN_OWNER_ACCEPTANCE_PROHIBITED = True
INVENTORY_RELATIVE_PATH = (
    "docs/architecture/architecture_refactorer_inventory/current_state_stores.json"
)
BASELINE_RELATIVE_PATH = (
    "docs/architecture/architecture_refactorer_inventory/state_store_baseline.json"
)
DUAL_WRITE_BOUND = "explicit-cutover-required"

_UNKNOWN_FIELD_MESSAGE = "unknown state-ownership field"
_MISSING_FIELD_MESSAGE = "missing state-ownership field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")


class StateOwnershipError(ArchitectureContractError):
    """Fail-closed state-ownership contract violation."""


class StateOwnershipAuthorityError(StateOwnershipError):
    """Raised when the model is asked to authorize, grant, or dual-own."""


class StateDisposition(str, Enum):
    """Closed store-class vocabulary (PCAR-PLAN-R1)."""

    AUTHORITATIVE = "authoritative"
    MATERIALIZED_PROJECTION = "materialized_projection"
    CACHE = "cache"
    HISTORICAL_EVENT = "historical_event"
    FIXTURE = "fixture"
    LEGACY = "legacy"
    UNKNOWN = "unknown"


CLOSED_STATE_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in StateDisposition
)
NON_AUTHORITATIVE_DISPOSITIONS: frozenset[StateDisposition] = frozenset(
    {
        StateDisposition.MATERIALIZED_PROJECTION,
        StateDisposition.CACHE,
        StateDisposition.HISTORICAL_EVENT,
        StateDisposition.FIXTURE,
        StateDisposition.LEGACY,
        StateDisposition.UNKNOWN,
    }
)
REBUILDABLE_DISPOSITIONS: frozenset[StateDisposition] = frozenset(
    {
        StateDisposition.MATERIALIZED_PROJECTION,
        StateDisposition.CACHE,
    }
)


class StoreKind(str, Enum):
    """Closed inventory store-kind vocabulary (PCAR-001 / PCAR-014)."""

    DUCKDB_TABLES = "DuckDB tables"
    JSON_FILES = "JSON files"
    MARKDOWN_TASK_BOARDS = "Markdown task boards"
    IN_MEMORY_REGISTRIES = "in-memory registries"
    EVENT_LOGS = "event logs"
    CACHE_NAMESPACES = "cache namespaces"
    WORKTREE_METADATA = "worktree metadata"
    LEASE_RECORDS = "lease records"
    PROVIDER_STATE = "provider state"
    GOAL_STATE = "goal state"
    TASK_STATE = "task state"
    COMPLETION_STATE = "completion state"
    RECEIPT_STATE = "receipt state"


REQUIRED_STORE_KINDS: tuple[StoreKind, ...] = tuple(StoreKind)
CLOSED_STORE_KINDS: frozenset[str] = frozenset(item.value for item in StoreKind)


class SemanticFactKind(str, Enum):
    """Closed mutable-or-derived semantic-fact vocabulary."""

    CONTROL_PLANE_STORE = "control-plane store"
    GOAL_STATE = "goal state"
    TASK_STATE = "task state"
    LEASE_RECORDS = "lease records"
    WORKTREE_METADATA = "worktree metadata"
    PROVIDER_STATE = "provider state"
    COMPLETION_STATE = "completion state"
    RECEIPT_STATE = "receipt state"
    REGISTRY_MEMBERSHIP = "in-memory registries"
    EVENT_HISTORY = "event logs"
    CACHE_NAMESPACE = "cache namespaces"
    JSON_PROJECTION = "JSON files"
    MARKDOWN_PROJECTION = "Markdown task boards"


CLOSED_SEMANTIC_FACTS: frozenset[str] = frozenset(
    item.value for item in SemanticFactKind
)
MUTABLE_SEMANTIC_FACTS: frozenset[SemanticFactKind] = frozenset(
    {
        SemanticFactKind.GOAL_STATE,
        SemanticFactKind.TASK_STATE,
        SemanticFactKind.LEASE_RECORDS,
        SemanticFactKind.WORKTREE_METADATA,
        SemanticFactKind.PROVIDER_STATE,
        SemanticFactKind.COMPLETION_STATE,
        SemanticFactKind.RECEIPT_STATE,
        SemanticFactKind.REGISTRY_MEMBERSHIP,
    }
)
DERIVED_SEMANTIC_FACTS: frozenset[SemanticFactKind] = frozenset(
    {
        SemanticFactKind.JSON_PROJECTION,
        SemanticFactKind.MARKDOWN_PROJECTION,
        SemanticFactKind.CACHE_NAMESPACE,
    }
)
HISTORICAL_SEMANTIC_FACTS: frozenset[SemanticFactKind] = frozenset(
    {SemanticFactKind.EVENT_HISTORY}
)

_KIND_TO_FACT: dict[StoreKind, SemanticFactKind] = {
    StoreKind.DUCKDB_TABLES: SemanticFactKind.CONTROL_PLANE_STORE,
    StoreKind.JSON_FILES: SemanticFactKind.JSON_PROJECTION,
    StoreKind.MARKDOWN_TASK_BOARDS: SemanticFactKind.MARKDOWN_PROJECTION,
    StoreKind.IN_MEMORY_REGISTRIES: SemanticFactKind.REGISTRY_MEMBERSHIP,
    StoreKind.EVENT_LOGS: SemanticFactKind.EVENT_HISTORY,
    StoreKind.CACHE_NAMESPACES: SemanticFactKind.CACHE_NAMESPACE,
    StoreKind.WORKTREE_METADATA: SemanticFactKind.WORKTREE_METADATA,
    StoreKind.LEASE_RECORDS: SemanticFactKind.LEASE_RECORDS,
    StoreKind.PROVIDER_STATE: SemanticFactKind.PROVIDER_STATE,
    StoreKind.GOAL_STATE: SemanticFactKind.GOAL_STATE,
    StoreKind.TASK_STATE: SemanticFactKind.TASK_STATE,
    StoreKind.COMPLETION_STATE: SemanticFactKind.COMPLETION_STATE,
    StoreKind.RECEIPT_STATE: SemanticFactKind.RECEIPT_STATE,
}


class MigrationPhase(str, Enum):
    """Closed bounded-migration phase vocabulary (PCAR-PLAN-R1)."""

    SNAPSHOT = "snapshot"
    DUAL_READ_SHADOW = "dual_read_shadow"
    DUAL_WRITE = "dual_write"
    CUTOVER = "cutover"
    VALIDATION = "validation"
    READ_ONLY_LEGACY = "read_only_legacy"
    RETIREMENT = "retirement"


CLOSED_MIGRATION_PHASES: frozenset[str] = frozenset(
    item.value for item in MigrationPhase
)
BOUNDED_MIGRATION_PHASES: tuple[MigrationPhase, ...] = (
    MigrationPhase.SNAPSHOT,
    MigrationPhase.DUAL_READ_SHADOW,
    MigrationPhase.DUAL_WRITE,
    MigrationPhase.CUTOVER,
    MigrationPhase.VALIDATION,
    MigrationPhase.READ_ONLY_LEGACY,
    MigrationPhase.RETIREMENT,
)
DUAL_AUTHORITY_PHASES: frozenset[MigrationPhase] = frozenset(
    {
        MigrationPhase.DUAL_READ_SHADOW,
        MigrationPhase.DUAL_WRITE,
    }
)


class StateConflictKind(str, Enum):
    """Closed hard-conflict vocabulary for unresolved state ownership."""

    UNKNOWN_OWNER = "unknown_owner"
    UNKNOWN_CONFLICT = "unknown_conflict"
    MULTIPLE_AUTHORITATIVE_STORES = "multiple_authoritative_stores"
    PROJECTION_CLAIMED_AUTHORITY = "projection_claimed_authority"
    CACHE_CLAIMED_AUTHORITY = "cache_claimed_authority"
    MARKDOWN_CLAIMED_AUTHORITY = "markdown_claimed_authority"
    DUCKLAKE_CLAIMED_AUTHORITY = "ducklake_claimed_authority"
    NON_REBUILDABLE_PROJECTION = "non_rebuildable_projection"
    INDEFINITE_DUAL_AUTHORITY = "indefinite_dual_authority"
    CONFLICTING_DISPOSITION = "conflicting_disposition"
    MISSING_KIND = "missing_kind"
    LEGACY_CLAIMED_AUTHORITY = "legacy_claimed_authority"
    FIXTURE_CLAIMED_AUTHORITY = "fixture_claimed_authority"
    EVENT_CLAIMED_MUTABLE_AUTHORITY = "event_claimed_mutable_authority"


CLOSED_STATE_CONFLICTS: frozenset[str] = frozenset(
    item.value for item in StateConflictKind
)

_ITEM_FIELDS = frozenset(
    {
        "content_identity",
        "disposition",
        "item_id",
        "kind",
        "nominated_owner",
        "path",
        "physical_store_id",
        "provenance",
        "rebuild_from_store_id",
        "rebuildable",
        "schema",
        "semantic_fact",
        "store_locator",
        "tables",
        "uncertainty",
        "version",
    }
)
_CONFLICT_FIELDS = frozenset(
    {
        "content_identity",
        "item_ids",
        "kind",
        "message",
        "schema",
        "semantic_fact",
        "store_ids",
        "version",
    }
)
_PHASE_FIELDS = frozenset(
    {
        "dual_authority",
        "dual_authority_bounded",
        "grants_authority",
        "phase",
        "terminal",
        "writes_legacy",
        "writes_target",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "can_create_indefinite_dual_ownership",
        "can_grant_authority",
        "content_identity",
        "dual_write_bound",
        "ends_dual_authority",
        "phases",
        "plan_id",
        "schema",
        "semantic_fact",
        "source_store_id",
        "target_store_id",
        "version",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "can_authorize_changes",
        "can_create_indefinite_dual_ownership",
        "can_grant_authority",
        "conflicts",
        "content_identity",
        "freshness",
        "items",
        "migrations",
        "repository_tree",
        "schema",
        "version",
    }
)
_NOMINATION_FIELDS = frozenset(
    {
        "disposition",
        "item_id",
        "kind",
        "nominated_owner",
        "path",
        "provenance",
        "rebuildable",
        "store_locator",
        "tables",
        "uncertainty",
    }
)


def _content_identity(payload: Mapping[str, Any]) -> str:
    return cid_for_dag_json(payload)


def _validate_dag_json_cid(value: str) -> str:
    try:
        return validate_cid(value, codecs=("dag-json",))
    except (TypeError, ValueError) as exc:
        raise StateOwnershipError(
            "content identity must be a dag-json CIDv1"
        ) from exc


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    extra = sorted(set(payload) - set(allowed))
    if extra:
        raise StateOwnershipError(f"{_UNKNOWN_FIELD_MESSAGE}: {extra}")


def _require_fields(payload: Mapping[str, Any], allowed: Iterable[str]) -> None:
    allowed_fields = set(allowed)
    _reject_unknown(payload, allowed_fields)
    missing = sorted(allowed_fields - set(payload))
    if missing:
        raise StateOwnershipError(f"{_MISSING_FIELD_MESSAGE}: {missing}")


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise StateOwnershipError(f"{name} must be a boolean")
    return value


def _require_optional_text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if type(value) is not str or "\x00" in value:
        raise StateOwnershipError(f"{name} must be a string")
    return value


def _require_text_sequence(
    value: Any,
    name: str,
    *,
    unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StateOwnershipError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=StateOwnershipError)
        for item in value
    )
    if unique and len(set(items)) != len(items):
        raise StateOwnershipError(f"{name} must not contain duplicates")
    return items


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _item_id_for(kind: StoreKind, item_id: str = "") -> str:
    if item_id:
        return _require_text(item_id, "item_id", error_type=StateOwnershipError)
    return kind.value.lower().replace(" ", "-")


def semantic_fact_for(kind: StoreKind | str) -> SemanticFactKind:
    """Map a closed store kind onto its semantic fact."""

    store_kind = _closed_enum(
        kind, StoreKind, "store kind", error_type=StateOwnershipError
    )
    return _KIND_TO_FACT[store_kind]


def physical_store_id(locator: str) -> str:
    """Normalize a store locator to a physical store identity."""

    text = _require_text(
        locator, "store locator", error_type=StateOwnershipError
    ).replace("\\", "/")
    lowered = text.lower()
    if "ducklake" in lowered:
        return "ducklake"
    if text.startswith("duckdb:"):
        return DUCKDB_CONTROL_STORE_ID if text == "duckdb:control" else text
    if "control.duckdb" in lowered:
        return DUCKDB_CONTROL_STORE_ID
    if lowered.endswith(".duckdb") or ".duckdb#" in lowered:
        base = text.split("#", 1)[0]
        name = base.rsplit("/", 1)[-1]
        if name.endswith(".duckdb"):
            name = name[: -len(".duckdb")]
        return f"duckdb:{name}"
    if text.endswith(".md") or lowered.endswith(".todo.md"):
        return f"markdown:{text}"
    if text.endswith(".jsonl"):
        return f"jsonl:{text}"
    if text.endswith(".json") or "architecture_refactorer_inventory" in text:
        return f"json:{text}"
    if "cache" in lowered:
        return f"cache:{text}"
    return f"path:{text}"


def _inferred_rebuildable(disposition: StateDisposition) -> bool:
    return disposition in REBUILDABLE_DISPOSITIONS


def _wrap_contract(exc: ArchitectureContractError) -> StateOwnershipError:
    if isinstance(exc, StateOwnershipError):
        return exc
    return StateOwnershipError(str(exc))


@dataclass(frozen=True)
class StateStoreBinding:
    """Current-tree source binding for one required store kind."""

    kind: StoreKind
    path: str
    store_locator: str
    disposition: StateDisposition
    nominated_owner: str
    start_line: int
    end_line: int
    uncertainty: str = ""
    tables: tuple[str, ...] = ()
    rebuildable: bool | None = None


CURRENT_TREE_STORE_BINDINGS: tuple[StateStoreBinding, ...] = (
    StateStoreBinding(
        StoreKind.DUCKDB_TABLES,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        DUCKDB_CONTROL_LOCATOR,
        StateDisposition.AUTHORITATIVE,
        "DuckDB plus Quack exclusive state-owner boundary",
        12,
        12,
        "runtime_state_root_binding",
        ("goals", "tasks", "leases", "completion_receipts", "domain_events", "worktrees"),
        False,
    ),
    StateStoreBinding(
        StoreKind.JSON_FILES,
        "docs/architecture/architecture_refactorer_inventory/state_store_baseline.json",
        "docs/architecture/architecture_refactorer_inventory",
        StateDisposition.MATERIALIZED_PROJECTION,
        DUCKDB_CONTROL_STORE_ID,
        1,
        1,
        "multiple_json_sinks_exist_this_inventory_nominates_without_migrating",
        (),
        True,
    ),
    StateStoreBinding(
        StoreKind.MARKDOWN_TASK_BOARDS,
        "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        StateDisposition.MATERIALIZED_PROJECTION,
        DUCKDB_CONTROL_STORE_ID,
        108,
        108,
        "",
        (),
        True,
    ),
    StateStoreBinding(
        StoreKind.IN_MEMORY_REGISTRIES,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py",
        "ipfs_accelerate_py/agent_supervisor",
        StateDisposition.UNKNOWN,
        "",
        84,
        84,
        "dynamic_registry_membership",
        (),
        False,
    ),
    StateStoreBinding(
        StoreKind.EVENT_LOGS,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        f"{DUCKDB_CONTROL_LOCATOR}#domain_events",
        StateDisposition.HISTORICAL_EVENT,
        DUCKDB_CONTROL_STORE_ID,
        827,
        827,
        "jsonl_sidecars_remain_legacy_append_streams",
        ("domain_events",),
        False,
    ),
    StateStoreBinding(
        StoreKind.CACHE_NAMESPACES,
        "ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py",
        StateDisposition.CACHE,
        DUCKDB_CONTROL_STORE_ID,
        846,
        846,
        "cache_directories_are_runtime_layout",
        (),
        True,
    ),
    StateStoreBinding(
        StoreKind.WORKTREE_METADATA,
        "ipfs_accelerate_py/agent_supervisor/merge/database_worktree_registry.py",
        f"{DUCKDB_CONTROL_LOCATOR}#worktrees",
        StateDisposition.AUTHORITATIVE,
        DUCKDB_CONTROL_STORE_ID,
        1632,
        1632,
        "git_worktree_directories_are_os_bootstrap_not_task_authority",
        ("worktrees",),
        False,
    ),
    StateStoreBinding(
        StoreKind.LEASE_RECORDS,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        f"{DUCKDB_CONTROL_LOCATOR}#leases",
        StateDisposition.AUTHORITATIVE,
        DUCKDB_CONTROL_STORE_ID,
        339,
        339,
        "duckdb_table_binding",
        ("leases",),
        False,
    ),
    StateStoreBinding(
        StoreKind.PROVIDER_STATE,
        "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
        "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
        StateDisposition.UNKNOWN,
        "",
        376,
        376,
        "provider_runtime",
        (),
        False,
    ),
    StateStoreBinding(
        StoreKind.GOAL_STATE,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        f"{DUCKDB_CONTROL_LOCATOR}#goals",
        StateDisposition.AUTHORITATIVE,
        DUCKDB_CONTROL_STORE_ID,
        407,
        407,
        "duckdb_table_binding",
        ("goals",),
        False,
    ),
    StateStoreBinding(
        StoreKind.TASK_STATE,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        f"{DUCKDB_CONTROL_LOCATOR}#tasks",
        StateDisposition.AUTHORITATIVE,
        DUCKDB_CONTROL_STORE_ID,
        470,
        470,
        "duckdb_table_binding",
        ("tasks",),
        False,
    ),
    StateStoreBinding(
        StoreKind.COMPLETION_STATE,
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/0001_control_plane.sql",
        f"{DUCKDB_CONTROL_LOCATOR}#completion_receipts",
        StateDisposition.AUTHORITATIVE,
        DUCKDB_CONTROL_STORE_ID,
        809,
        809,
        "duckdb_table_binding",
        ("completion_receipts",),
        False,
    ),
    StateStoreBinding(
        StoreKind.RECEIPT_STATE,
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/authoritative_completion.py",
        f"{DUCKDB_CONTROL_LOCATOR}#completion_receipts",
        StateDisposition.AUTHORITATIVE,
        DUCKDB_CONTROL_STORE_ID,
        178,
        178,
        "duckdb_table_binding",
        ("completion_receipts",),
        False,
    ),
)


@dataclass(frozen=True)
class StateItem:
    """One classified store or projection bound to source evidence."""

    item_id: str
    kind: StoreKind
    path: str
    disposition: StateDisposition
    provenance: SourceFactIdentity
    store_locator: str = ""
    physical_store_id: str = ""
    semantic_fact: SemanticFactKind | str = ""
    nominated_owner: str = ""
    rebuildable: bool = False
    rebuild_from_store_id: str = ""
    uncertainty: str = ""
    tables: tuple[str, ...] = ()
    schema: str = STATE_ITEM_SCHEMA
    version: int = STATE_ITEM_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=StateOwnershipError)
        if schema != STATE_ITEM_SCHEMA:
            raise StateOwnershipError("unexpected state-item schema")
        version = _require_int(self.version, "version", error_type=StateOwnershipError)
        if version != STATE_ITEM_VERSION:
            raise StateOwnershipError("unexpected state-item version")
        item_id = _require_text(
            self.item_id, "item_id", error_type=StateOwnershipError
        )
        if _looks_like_content_identity(item_id):
            raise StateOwnershipError(
                "content identity is not inferred to be a store owner"
            )
        kind = _closed_enum(
            self.kind, StoreKind, "store kind", error_type=StateOwnershipError
        )
        disposition = _closed_enum(
            self.disposition,
            StateDisposition,
            "state disposition",
            error_type=StateOwnershipError,
        )
        path = _require_text(self.path, "path", error_type=StateOwnershipError)
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        store_locator = _require_optional_text(self.store_locator, "store_locator") or path
        physical = _require_optional_text(self.physical_store_id, "physical_store_id")
        if not physical:
            physical = physical_store_id(store_locator)
        fact = self.semantic_fact or semantic_fact_for(kind)
        fact = _closed_enum(
            fact,
            SemanticFactKind,
            "semantic fact",
            error_type=StateOwnershipError,
        )
        rebuildable = _require_bool(self.rebuildable, "rebuildable")
        rebuild_from = _require_optional_text(
            self.rebuild_from_store_id, "rebuild_from_store_id"
        )
        nominated = _require_optional_text(self.nominated_owner, "nominated_owner")
        if rebuildable and not rebuild_from:
            if nominated == DUCKDB_CONTROL_STORE_ID:
                rebuild_from = DUCKDB_CONTROL_STORE_ID
            elif disposition is not StateDisposition.AUTHORITATIVE:
                rebuild_from = DUCKDB_CONTROL_STORE_ID
        uncertainty = _require_optional_text(self.uncertainty, "uncertainty")
        tables = _require_text_sequence(self.tables, "tables", unique=True)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "store_locator", store_locator)
        object.__setattr__(self, "physical_store_id", physical)
        object.__setattr__(self, "semantic_fact", fact)
        object.__setattr__(self, "nominated_owner", nominated)
        object.__setattr__(self, "rebuildable", rebuildable)
        object.__setattr__(self, "rebuild_from_store_id", rebuild_from)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "tables", tables)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=StateOwnershipError,
                )
            )
            if claimed != identity:
                raise StateOwnershipError("state-item content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "item_id": self.item_id,
            "kind": self.kind.value,
            "nominated_owner": self.nominated_owner,
            "path": self.path,
            "physical_store_id": self.physical_store_id,
            "provenance": self.provenance.to_dict(),
            "rebuild_from_store_id": self.rebuild_from_store_id,
            "rebuildable": self.rebuildable,
            "schema": self.schema,
            "semantic_fact": self.semantic_fact.value,
            "store_locator": self.store_locator,
            "tables": list(self.tables),
            "uncertainty": self.uncertainty,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise StateOwnershipError("state-item content identity mismatch")
        return {**payload, "content_identity": identity}

    @property
    def is_authoritative(self) -> bool:
        return self.disposition is StateDisposition.AUTHORITATIVE

    @property
    def is_projection(self) -> bool:
        return self.disposition is StateDisposition.MATERIALIZED_PROJECTION

    @property
    def is_unknown(self) -> bool:
        return self.disposition is StateDisposition.UNKNOWN

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateItem":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _ITEM_FIELDS)
        try:
            item = cls(
                item_id=mapping["item_id"],
                kind=mapping["kind"],
                path=mapping["path"],
                disposition=mapping["disposition"],
                provenance=mapping["provenance"],
                store_locator=mapping["store_locator"],
                physical_store_id=mapping["physical_store_id"],
                semantic_fact=mapping["semantic_fact"],
                nominated_owner=mapping["nominated_owner"],
                rebuildable=mapping["rebuildable"],
                rebuild_from_store_id=mapping["rebuild_from_store_id"],
                uncertainty=mapping["uncertainty"],
                tables=mapping["tables"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != item.content_identity:
            raise StateOwnershipError("state-item content identity mismatch")
        return item

    from_dict = from_mapping


@dataclass(frozen=True)
class StateNomination:
    """Reviewed classification input before conflict detection."""

    kind: StoreKind
    path: str
    disposition: StateDisposition
    provenance: SourceFactIdentity
    item_id: str = ""
    store_locator: str = ""
    nominated_owner: str = ""
    uncertainty: str = ""
    tables: tuple[str, ...] = ()
    rebuildable: bool | None = None

    def __post_init__(self) -> None:
        kind = _closed_enum(
            self.kind, StoreKind, "store kind", error_type=StateOwnershipError
        )
        disposition = _closed_enum(
            self.disposition,
            StateDisposition,
            "state disposition",
            error_type=StateOwnershipError,
        )
        path = _require_text(self.path, "path", error_type=StateOwnershipError)
        try:
            provenance = (
                self.provenance
                if isinstance(self.provenance, SourceFactIdentity)
                else SourceFactIdentity.from_mapping(self.provenance)
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        item_id = _item_id_for(kind, self.item_id)
        store_locator = _require_optional_text(self.store_locator, "store_locator")
        nominated = _require_optional_text(self.nominated_owner, "nominated_owner")
        uncertainty = _require_optional_text(self.uncertainty, "uncertainty")
        tables = _require_text_sequence(self.tables, "tables", unique=True)
        rebuildable = self.rebuildable
        if rebuildable is None:
            rebuildable = _inferred_rebuildable(disposition)
        else:
            rebuildable = _require_bool(rebuildable, "rebuildable")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "store_locator", store_locator)
        object.__setattr__(self, "nominated_owner", nominated)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "rebuildable", rebuildable)

    def to_item(self) -> StateItem:
        locator = self.store_locator or self.path
        rebuild_from = ""
        if self.rebuildable:
            rebuild_from = (
                DUCKDB_CONTROL_STORE_ID
                if self.disposition is not StateDisposition.AUTHORITATIVE
                else ""
            )
            if self.nominated_owner == DUCKDB_CONTROL_STORE_ID:
                rebuild_from = DUCKDB_CONTROL_STORE_ID
        return StateItem(
            item_id=self.item_id,
            kind=self.kind,
            path=self.path,
            disposition=self.disposition,
            provenance=self.provenance,
            store_locator=locator,
            semantic_fact=semantic_fact_for(self.kind),
            nominated_owner=self.nominated_owner,
            rebuildable=bool(self.rebuildable),
            rebuild_from_store_id=rebuild_from,
            uncertainty=self.uncertainty,
            tables=self.tables,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "item_id": self.item_id,
            "kind": self.kind.value,
            "nominated_owner": self.nominated_owner,
            "path": self.path,
            "provenance": self.provenance.to_dict(),
            "rebuildable": bool(self.rebuildable),
            "store_locator": self.store_locator,
            "tables": list(self.tables),
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateNomination":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _NOMINATION_FIELDS)
        return cls(
            kind=mapping["kind"],
            path=mapping["path"],
            disposition=mapping["disposition"],
            provenance=mapping["provenance"],
            item_id=mapping["item_id"],
            store_locator=mapping["store_locator"],
            nominated_owner=mapping["nominated_owner"],
            uncertainty=mapping["uncertainty"],
            tables=mapping["tables"],
            rebuildable=mapping["rebuildable"],
        )

    from_dict = from_mapping


@dataclass(frozen=True)
class StateConflict:
    """Typed hard conflict that prevents unique store ownership."""

    kind: StateConflictKind
    semantic_fact: SemanticFactKind
    message: str
    item_ids: tuple[str, ...] = ()
    store_ids: tuple[str, ...] = ()
    schema: str = STATE_CONFLICT_SCHEMA
    version: int = STATE_CONFLICT_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=StateOwnershipError)
        if schema != STATE_CONFLICT_SCHEMA:
            raise StateOwnershipError("unexpected state-conflict schema")
        version = _require_int(self.version, "version", error_type=StateOwnershipError)
        if version != STATE_CONFLICT_VERSION:
            raise StateOwnershipError("unexpected state-conflict version")
        kind = _closed_enum(
            self.kind,
            StateConflictKind,
            "state conflict kind",
            error_type=StateOwnershipError,
        )
        fact = _closed_enum(
            self.semantic_fact,
            SemanticFactKind,
            "semantic fact",
            error_type=StateOwnershipError,
        )
        message = _require_text(self.message, "message", error_type=StateOwnershipError)
        item_ids = tuple(
            sorted(set(_require_text_sequence(self.item_ids, "item_ids")))
        )
        store_ids = tuple(
            sorted(set(_require_text_sequence(self.store_ids, "store_ids")))
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "semantic_fact", fact)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "store_ids", store_ids)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=StateOwnershipError,
                )
            )
            if claimed != identity:
                raise StateOwnershipError("state-conflict content identity mismatch")
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "item_ids": list(self.item_ids),
            "kind": self.kind.value,
            "message": self.message,
            "schema": self.schema,
            "semantic_fact": self.semantic_fact.value,
            "store_ids": list(self.store_ids),
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise StateOwnershipError("state-conflict content identity mismatch")
        return {**payload, "content_identity": identity}

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateConflict":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _CONFLICT_FIELDS)
        conflict = cls(
            kind=mapping["kind"],
            semantic_fact=mapping["semantic_fact"],
            message=mapping["message"],
            item_ids=mapping["item_ids"],
            store_ids=mapping["store_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != conflict.content_identity:
            raise StateOwnershipError("state-conflict content identity mismatch")
        return conflict

    from_dict = from_mapping


@dataclass(frozen=True)
class StateMigrationPhase:
    """One closed migration phase with explicit dual-authority bounds."""

    phase: MigrationPhase
    dual_authority: bool
    dual_authority_bounded: bool
    grants_authority: bool = False
    writes_legacy: bool = False
    writes_target: bool = False
    terminal: bool = False

    def __post_init__(self) -> None:
        phase = _closed_enum(
            self.phase,
            MigrationPhase,
            "migration phase",
            error_type=StateOwnershipError,
        )
        dual_authority = _require_bool(self.dual_authority, "dual_authority")
        dual_authority_bounded = _require_bool(
            self.dual_authority_bounded, "dual_authority_bounded"
        )
        grants_authority = _require_bool(self.grants_authority, "grants_authority")
        writes_legacy = _require_bool(self.writes_legacy, "writes_legacy")
        writes_target = _require_bool(self.writes_target, "writes_target")
        terminal = _require_bool(self.terminal, "terminal")
        if grants_authority:
            raise StateOwnershipError(
                "migration phases cannot grant authority"
            )
        expected_dual = phase in DUAL_AUTHORITY_PHASES
        if dual_authority is not expected_dual:
            raise StateOwnershipError(
                f"{phase.value} dual-authority flag must be {expected_dual}"
            )
        if dual_authority and not dual_authority_bounded:
            raise StateOwnershipError(
                "dual-authority migration phases must be formally bounded"
            )
        if not dual_authority and dual_authority_bounded:
            raise StateOwnershipError(
                "non-dual phases cannot claim a dual-authority bound"
            )
        expected_legacy_write = phase is MigrationPhase.DUAL_WRITE
        expected_target_write = phase in {
            MigrationPhase.DUAL_WRITE,
            MigrationPhase.CUTOVER,
        }
        if writes_legacy is not expected_legacy_write:
            raise StateOwnershipError(
                f"{phase.value} writes_legacy must be {expected_legacy_write}"
            )
        if writes_target is not expected_target_write:
            raise StateOwnershipError(
                f"{phase.value} writes_target must be {expected_target_write}"
            )
        expected_terminal = phase is MigrationPhase.RETIREMENT
        if terminal is not expected_terminal:
            raise StateOwnershipError(
                f"{phase.value} terminal flag must be {expected_terminal}"
            )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "dual_authority", dual_authority)
        object.__setattr__(self, "dual_authority_bounded", dual_authority_bounded)
        object.__setattr__(self, "grants_authority", False)
        object.__setattr__(self, "writes_legacy", writes_legacy)
        object.__setattr__(self, "writes_target", writes_target)
        object.__setattr__(self, "terminal", terminal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dual_authority": self.dual_authority,
            "dual_authority_bounded": self.dual_authority_bounded,
            "grants_authority": False,
            "phase": self.phase.value,
            "terminal": self.terminal,
            "writes_legacy": self.writes_legacy,
            "writes_target": self.writes_target,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateMigrationPhase":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _PHASE_FIELDS)
        return cls(
            phase=mapping["phase"],
            dual_authority=mapping["dual_authority"],
            dual_authority_bounded=mapping["dual_authority_bounded"],
            grants_authority=mapping["grants_authority"],
            writes_legacy=mapping["writes_legacy"],
            writes_target=mapping["writes_target"],
            terminal=mapping["terminal"],
        )

    from_dict = from_mapping


def closed_migration_phases() -> tuple[StateMigrationPhase, ...]:
    """Return the closed snapshot-to-retirement migration sequence."""

    return (
        StateMigrationPhase(
            phase=MigrationPhase.SNAPSHOT,
            dual_authority=False,
            dual_authority_bounded=False,
            writes_legacy=False,
            writes_target=False,
            terminal=False,
        ),
        StateMigrationPhase(
            phase=MigrationPhase.DUAL_READ_SHADOW,
            dual_authority=True,
            dual_authority_bounded=True,
            writes_legacy=False,
            writes_target=False,
            terminal=False,
        ),
        StateMigrationPhase(
            phase=MigrationPhase.DUAL_WRITE,
            dual_authority=True,
            dual_authority_bounded=True,
            writes_legacy=True,
            writes_target=True,
            terminal=False,
        ),
        StateMigrationPhase(
            phase=MigrationPhase.CUTOVER,
            dual_authority=False,
            dual_authority_bounded=False,
            writes_legacy=False,
            writes_target=True,
            terminal=False,
        ),
        StateMigrationPhase(
            phase=MigrationPhase.VALIDATION,
            dual_authority=False,
            dual_authority_bounded=False,
            writes_legacy=False,
            writes_target=False,
            terminal=False,
        ),
        StateMigrationPhase(
            phase=MigrationPhase.READ_ONLY_LEGACY,
            dual_authority=False,
            dual_authority_bounded=False,
            writes_legacy=False,
            writes_target=False,
            terminal=False,
        ),
        StateMigrationPhase(
            phase=MigrationPhase.RETIREMENT,
            dual_authority=False,
            dual_authority_bounded=False,
            writes_legacy=False,
            writes_target=False,
            terminal=True,
        ),
    )


@dataclass(frozen=True)
class StateMigrationPlan:
    """Immutable bounded migration that ends dual-read/write."""

    plan_id: str
    semantic_fact: SemanticFactKind
    source_store_id: str
    target_store_id: str
    phases: tuple[StateMigrationPhase, ...] = ()
    dual_write_bound: str = DUAL_WRITE_BOUND
    ends_dual_authority: bool = True
    can_grant_authority: bool = False
    can_create_indefinite_dual_ownership: bool = False
    schema: str = STATE_MIGRATION_SCHEMA
    version: int = STATE_MIGRATION_VERSION
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=StateOwnershipError)
        if schema != STATE_MIGRATION_SCHEMA:
            raise StateOwnershipError("unexpected state-migration schema")
        version = _require_int(self.version, "version", error_type=StateOwnershipError)
        if version != STATE_MIGRATION_VERSION:
            raise StateOwnershipError("unexpected state-migration version")
        plan_id = _require_text(self.plan_id, "plan_id", error_type=StateOwnershipError)
        fact = _closed_enum(
            self.semantic_fact,
            SemanticFactKind,
            "semantic fact",
            error_type=StateOwnershipError,
        )
        source = _require_text(
            self.source_store_id, "source_store_id", error_type=StateOwnershipError
        )
        target = _require_text(
            self.target_store_id, "target_store_id", error_type=StateOwnershipError
        )
        if source == target:
            raise StateOwnershipError(
                "migration source and target must be distinct stores"
            )
        if target == "ducklake" or target.startswith("ducklake"):
            raise StateOwnershipError("DuckLake cannot be a migration target authority")
        if target.startswith("markdown:"):
            raise StateOwnershipError("Markdown cannot be a migration target authority")
        if target.startswith("json:") or target.startswith("cache:"):
            raise StateOwnershipError(
                "projections and caches cannot be migration target authorities"
            )
        if self.can_grant_authority is not False:
            raise StateOwnershipError("migration plans cannot grant authority")
        if self.can_create_indefinite_dual_ownership is not False:
            raise StateOwnershipError(
                "migration plans cannot create indefinite dual ownership"
            )
        ends = _require_bool(self.ends_dual_authority, "ends_dual_authority")
        if ends is not True:
            raise StateOwnershipError("migration plans must end dual authority")
        bound = _require_text(
            self.dual_write_bound, "dual_write_bound", error_type=StateOwnershipError
        )
        if isinstance(self.phases, (str, bytes, bytearray)) or not isinstance(
            self.phases, Sequence
        ):
            raise StateOwnershipError("phases must be a sequence")
        phases = tuple(
            item
            if isinstance(item, StateMigrationPhase)
            else StateMigrationPhase.from_mapping(item)
            for item in self.phases
        )
        expected = closed_migration_phases()
        if tuple(item.phase for item in phases) != tuple(item.phase for item in expected):
            raise StateOwnershipError(
                "bounded migration requires the closed snapshot-to-retirement sequence"
            )
        if phases != expected:
            raise StateOwnershipError(
                "bounded migration phases must match the closed dual-authority contract"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "semantic_fact", fact)
        object.__setattr__(self, "source_store_id", source)
        object.__setattr__(self, "target_store_id", target)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "dual_write_bound", bound)
        object.__setattr__(self, "ends_dual_authority", True)
        object.__setattr__(self, "can_grant_authority", False)
        object.__setattr__(self, "can_create_indefinite_dual_ownership", False)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=StateOwnershipError,
                )
            )
            if claimed != identity:
                raise StateOwnershipError(
                    "state-migration content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "can_create_indefinite_dual_ownership": False,
            "can_grant_authority": False,
            "dual_write_bound": self.dual_write_bound,
            "ends_dual_authority": True,
            "phases": [item.to_dict() for item in self.phases],
            "plan_id": self.plan_id,
            "schema": self.schema,
            "semantic_fact": self.semantic_fact.value,
            "source_store_id": self.source_store_id,
            "target_store_id": self.target_store_id,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise StateOwnershipError("state-migration content identity mismatch")
        return {**payload, "content_identity": identity}

    @property
    def terminal_phase(self) -> StateMigrationPhase:
        return self.phases[-1]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateMigrationPlan":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _PLAN_FIELDS)
        plan = cls(
            plan_id=mapping["plan_id"],
            semantic_fact=mapping["semantic_fact"],
            source_store_id=mapping["source_store_id"],
            target_store_id=mapping["target_store_id"],
            phases=mapping["phases"],
            dual_write_bound=mapping["dual_write_bound"],
            ends_dual_authority=mapping["ends_dual_authority"],
            can_grant_authority=mapping["can_grant_authority"],
            can_create_indefinite_dual_ownership=mapping[
                "can_create_indefinite_dual_ownership"
            ],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != plan.content_identity:
            raise StateOwnershipError("state-migration content identity mismatch")
        return plan

    from_dict = from_mapping


def plan_state_migration(
    semantic_fact: SemanticFactKind | str,
    source_store_id: str,
    target_store_id: str,
    *,
    plan_id: str = "",
    dual_write_bound: str = DUAL_WRITE_BOUND,
) -> StateMigrationPlan:
    """Build the closed bounded migration that ends dual authority."""

    fact = _closed_enum(
        semantic_fact,
        SemanticFactKind,
        "semantic fact",
        error_type=StateOwnershipError,
    )
    identity = plan_id or f"migrate-{fact.value.lower().replace(' ', '-')}"
    return StateMigrationPlan(
        plan_id=identity,
        semantic_fact=fact,
        source_store_id=source_store_id,
        target_store_id=target_store_id,
        phases=closed_migration_phases(),
        dual_write_bound=dual_write_bound,
    )


def _normalize_items(
    items: Sequence[StateItem | StateNomination | Mapping[str, Any]],
) -> tuple[StateItem, ...]:
    if isinstance(items, (str, bytes, bytearray)) or not isinstance(items, Sequence):
        raise StateOwnershipError("items must be a sequence")
    parsed: list[StateItem] = []
    for item in items:
        if isinstance(item, StateItem):
            parsed.append(item)
        elif isinstance(item, StateNomination):
            parsed.append(item.to_item())
        else:
            mapping = _require_mapping(item, error_type=StateOwnershipError)
            if "physical_store_id" in mapping:
                parsed.append(StateItem.from_mapping(mapping))
            else:
                parsed.append(StateNomination.from_mapping(mapping).to_item())
    ids = [item.item_id for item in parsed]
    if len(set(ids)) != len(ids):
        raise StateOwnershipError("state items must have unique item_id values")
    return tuple(parsed)


def _normalize_plans(
    plans: Sequence[StateMigrationPlan | Mapping[str, Any]] | None,
) -> tuple[StateMigrationPlan, ...]:
    if plans is None:
        return ()
    if isinstance(plans, (str, bytes, bytearray)) or not isinstance(plans, Sequence):
        raise StateOwnershipError("migrations must be a sequence")
    parsed = tuple(
        item
        if isinstance(item, StateMigrationPlan)
        else StateMigrationPlan.from_mapping(item)
        for item in plans
    )
    ids = [item.plan_id for item in parsed]
    if len(set(ids)) != len(ids):
        raise StateOwnershipError("migration plans must have unique plan_id values")
    return parsed


def _conflict(
    kind: StateConflictKind,
    fact: SemanticFactKind,
    message: str,
    items: Iterable[StateItem] = (),
    store_ids: Iterable[str] = (),
) -> StateConflict:
    records = tuple(items)
    return StateConflict(
        kind=kind,
        semantic_fact=fact,
        message=message,
        item_ids=tuple(item.item_id for item in records),
        store_ids=tuple(store_ids),
    )


def detect_state_conflicts(items: Sequence[StateItem]) -> tuple[StateConflict, ...]:
    """Derive hard conflicts from classified store items."""

    conflicts: list[StateConflict] = []
    by_fact: dict[SemanticFactKind, list[StateItem]] = {
        fact: [] for fact in SemanticFactKind
    }
    for item in items:
        by_fact[item.semantic_fact].append(item)
        if item.disposition is StateDisposition.UNKNOWN:
            conflicts.append(
                _conflict(
                    StateConflictKind.UNKNOWN_CONFLICT,
                    item.semantic_fact,
                    "unknown state ownership cannot be accepted as resolved",
                    (item,),
                    (item.physical_store_id,) if item.physical_store_id else (),
                )
            )
        if item.is_authoritative and (
            item.kind is StoreKind.MARKDOWN_TASK_BOARDS
            or item.physical_store_id.startswith("markdown:")
        ):
            conflicts.append(
                _conflict(
                    StateConflictKind.MARKDOWN_CLAIMED_AUTHORITY,
                    item.semantic_fact,
                    "Markdown is a sealed bootstrap/human projection and never authority",
                    (item,),
                    (item.physical_store_id,),
                )
            )
        if item.physical_store_id == "ducklake" and item.is_authoritative:
            conflicts.append(
                _conflict(
                    StateConflictKind.DUCKLAKE_CLAIMED_AUTHORITY,
                    item.semantic_fact,
                    "DuckLake is an optional non-authoritative history projection",
                    (item,),
                    (item.physical_store_id,),
                )
            )
        if item.semantic_fact in DERIVED_SEMANTIC_FACTS:
            if item.is_authoritative:
                kind = (
                    StateConflictKind.CACHE_CLAIMED_AUTHORITY
                    if item.kind is StoreKind.CACHE_NAMESPACES
                    else StateConflictKind.PROJECTION_CLAIMED_AUTHORITY
                )
                if item.kind is StoreKind.MARKDOWN_TASK_BOARDS:
                    kind = StateConflictKind.MARKDOWN_CLAIMED_AUTHORITY
                conflicts.append(
                    _conflict(
                        kind,
                        item.semantic_fact,
                        "projections and caches cannot be authoritative stores",
                        (item,),
                        (item.physical_store_id,),
                    )
                )
            if not item.rebuildable:
                conflicts.append(
                    _conflict(
                        StateConflictKind.NON_REBUILDABLE_PROJECTION,
                        item.semantic_fact,
                        "projections and caches must be rebuildable from the authoritative store",
                        (item,),
                        (item.physical_store_id,),
                    )
                )
        if (
            item.semantic_fact in HISTORICAL_SEMANTIC_FACTS
            and item.is_authoritative
        ):
            conflicts.append(
                _conflict(
                    StateConflictKind.EVENT_CLAIMED_MUTABLE_AUTHORITY,
                    item.semantic_fact,
                    "historical events are not mutable current-state authority",
                    (item,),
                    (item.physical_store_id,),
                )
            )
    present_kinds = {item.kind for item in items}
    missing = [kind for kind in REQUIRED_STORE_KINDS if kind not in present_kinds]
    if missing:
        conflicts.append(
            _conflict(
                StateConflictKind.MISSING_KIND,
                SemanticFactKind.CONTROL_PLANE_STORE,
                f"missing required store kinds: {[item.value for item in missing]}",
            )
        )

    for fact in MUTABLE_SEMANTIC_FACTS:
        group = by_fact[fact]
        authoritative = [item for item in group if item.is_authoritative]
        unknown = [item for item in group if item.is_unknown]
        store_ids = tuple(
            sorted({item.physical_store_id for item in authoritative})
        )
        if len(store_ids) > 1:
            conflicts.append(
                _conflict(
                    StateConflictKind.MULTIPLE_AUTHORITATIVE_STORES,
                    fact,
                    "each mutable semantic fact must have exactly one authoritative store",
                    authoritative,
                    store_ids,
                )
            )
        dispositions = {item.disposition for item in group if item.is_authoritative}
        if StateDisposition.AUTHORITATIVE in dispositions and unknown:
            conflicts.append(
                _conflict(
                    StateConflictKind.CONFLICTING_DISPOSITION,
                    fact,
                    "unknown ownership conflicts with an authoritative store for the same fact",
                    (*authoritative, *unknown),
                    store_ids,
                )
            )
        if not authoritative and not unknown:
            if group:
                dispositions = {item.disposition for item in group}
                if dispositions <= {StateDisposition.FIXTURE}:
                    conflict_kind = StateConflictKind.FIXTURE_CLAIMED_AUTHORITY
                    message = "fixture stores cannot satisfy production state authority"
                elif dispositions <= {StateDisposition.LEGACY}:
                    conflict_kind = StateConflictKind.LEGACY_CLAIMED_AUTHORITY
                    message = "legacy stores cannot remain the unique owner"
                else:
                    conflict_kind = StateConflictKind.UNKNOWN_OWNER
                    message = "mutable semantic fact has no authoritative store"
                conflicts.append(
                    _conflict(
                        conflict_kind,
                        fact,
                        message,
                        group,
                        tuple(sorted({item.physical_store_id for item in group})),
                    )
                )
            else:
                conflicts.append(
                    _conflict(
                        StateConflictKind.UNKNOWN_OWNER,
                        fact,
                        "mutable semantic fact is unclassified",
                    )
                )
        elif not authoritative and unknown:
            conflicts.append(
                _conflict(
                    StateConflictKind.UNKNOWN_OWNER,
                    fact,
                    "unknown state ownership is a hard blocker, not an accepted owner",
                    unknown,
                    tuple(sorted({item.physical_store_id for item in unknown})),
                )
            )

    unique: dict[tuple[str, str, tuple[str, ...]], StateConflict] = {}
    for conflict in conflicts:
        key = (conflict.kind.value, conflict.semantic_fact.value, conflict.item_ids)
        unique.setdefault(key, conflict)
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.kind.value, item.semantic_fact.value, item.item_ids),
        )
    )


@dataclass(frozen=True)
class StateOwnershipModel:
    """Reviewed ownership of every inventoried mutable semantic fact."""

    repository_tree: str
    freshness: str
    items: tuple[StateItem, ...]
    conflicts: tuple[StateConflict, ...] = ()
    migrations: tuple[StateMigrationPlan, ...] = ()
    schema: str = STATE_OWNERSHIP_SCHEMA
    version: int = STATE_OWNERSHIP_VERSION
    can_authorize_changes: bool = CAN_AUTHORIZE_CHANGES
    can_grant_authority: bool = CAN_GRANT_AUTHORITY
    can_create_indefinite_dual_ownership: bool = CAN_CREATE_INDEFINITE_DUAL_OWNERSHIP
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=StateOwnershipError)
        if schema != STATE_OWNERSHIP_SCHEMA:
            raise StateOwnershipError("unexpected state-ownership schema")
        version = _require_int(self.version, "version", error_type=StateOwnershipError)
        if version != STATE_OWNERSHIP_VERSION:
            raise StateOwnershipError("unexpected state-ownership version")
        if self.can_authorize_changes is not False:
            raise StateOwnershipError(
                "state ownership model cannot authorize changes"
            )
        if self.can_grant_authority is not False:
            raise StateOwnershipError("state ownership model cannot grant authority")
        if self.can_create_indefinite_dual_ownership is not False:
            raise StateOwnershipError(
                "state ownership model cannot create indefinite dual ownership"
            )
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=StateOwnershipError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=StateOwnershipError
        )
        items = _normalize_items(self.items)
        migrations = _normalize_plans(self.migrations)
        derived = detect_state_conflicts(items)
        if self.conflicts:
            if isinstance(self.conflicts, (str, bytes, bytearray)) or not isinstance(
                self.conflicts, Sequence
            ):
                raise StateOwnershipError("conflicts must be a sequence")
            provided = tuple(
                item
                if isinstance(item, StateConflict)
                else StateConflict.from_mapping(item)
                for item in self.conflicts
            )
            provided_ids = [item.to_dict() for item in provided]
            derived_ids = [item.to_dict() for item in derived]
            if provided_ids != derived_ids:
                raise StateOwnershipError("state-ownership conflicts projection mismatch")
            conflicts = provided
        else:
            conflicts = derived
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "migrations", migrations)
        object.__setattr__(self, "can_authorize_changes", False)
        object.__setattr__(self, "can_grant_authority", False)
        object.__setattr__(self, "can_create_indefinite_dual_ownership", False)
        identity = _content_identity(self._identity_payload())
        if self.content_identity:
            claimed = _validate_dag_json_cid(
                _require_text(
                    self.content_identity,
                    "content_identity",
                    error_type=StateOwnershipError,
                )
            )
            if claimed != identity:
                raise StateOwnershipError(
                    "state-ownership content identity mismatch"
                )
        object.__setattr__(self, "content_identity", identity)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "can_authorize_changes": False,
            "can_create_indefinite_dual_ownership": False,
            "can_grant_authority": False,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "freshness": self.freshness,
            "items": [item.to_dict() for item in self.items],
            "migrations": [item.to_dict() for item in self.migrations],
            "repository_tree": self.repository_tree,
            "schema": self.schema,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise StateOwnershipError("state-ownership content identity mismatch")
        return {**payload, "content_identity": identity}

    def to_json(self) -> str:
        return canonical_dag_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def covers_required_kinds(self) -> bool:
        return {item.kind for item in self.items} >= set(REQUIRED_STORE_KINDS)

    @property
    def fails_closed(self) -> bool:
        return bool(self.conflicts)

    @property
    def one_authoritative_store(self) -> bool:
        """Every mutable fact has exactly one owner or a recorded hard conflict."""

        for fact in MUTABLE_SEMANTIC_FACTS:
            owners = self.authoritative_items(fact)
            store_ids = {item.physical_store_id for item in owners}
            has_conflict = bool(self.conflicts_for(fact))
            if len(store_ids) == 1 and not has_conflict:
                continue
            if has_conflict:
                continue
            return False
        return True

    def items_of(self, kind: StoreKind | str) -> tuple[StateItem, ...]:
        store_kind = _closed_enum(
            kind, StoreKind, "store kind", error_type=StateOwnershipError
        )
        return tuple(item for item in self.items if item.kind is store_kind)

    def items_with(self, disposition: StateDisposition | str) -> tuple[StateItem, ...]:
        value = _closed_enum(
            disposition,
            StateDisposition,
            "state disposition",
            error_type=StateOwnershipError,
        )
        return tuple(item for item in self.items if item.disposition is value)

    def items_for_fact(
        self, fact: SemanticFactKind | str
    ) -> tuple[StateItem, ...]:
        kind = _closed_enum(
            fact, SemanticFactKind, "semantic fact", error_type=StateOwnershipError
        )
        return tuple(item for item in self.items if item.semantic_fact is kind)

    def authoritative_items(
        self, fact: SemanticFactKind | str
    ) -> tuple[StateItem, ...]:
        return tuple(
            item for item in self.items_for_fact(fact) if item.is_authoritative
        )

    def conflicts_for(
        self, fact: SemanticFactKind | str
    ) -> tuple[StateConflict, ...]:
        kind = _closed_enum(
            fact, SemanticFactKind, "semantic fact", error_type=StateOwnershipError
        )
        return tuple(item for item in self.conflicts if item.semantic_fact is kind)

    def rebuildable_projections(self) -> tuple[StateItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.rebuildable and item.disposition in REBUILDABLE_DISPOSITIONS
        )

    def authoritative_store_for(self, fact: SemanticFactKind | str) -> str:
        kind = _closed_enum(
            fact, SemanticFactKind, "semantic fact", error_type=StateOwnershipError
        )
        if self.conflicts_for(kind):
            raise StateOwnershipError(
                f"{kind.value} has no unique authoritative store"
            )
        owners = self.authoritative_items(kind)
        store_ids = {item.physical_store_id for item in owners}
        if len(store_ids) != 1:
            raise StateOwnershipError(
                f"{kind.value} has no unique authoritative store"
            )
        return next(iter(store_ids))

    def migration_for(
        self, fact: SemanticFactKind | str
    ) -> StateMigrationPlan | None:
        kind = _closed_enum(
            fact, SemanticFactKind, "semantic fact", error_type=StateOwnershipError
        )
        matches = [item for item in self.migrations if item.semantic_fact is kind]
        if len(matches) > 1:
            raise StateOwnershipError(
                f"duplicate migration plans for {kind.value}"
            )
        return matches[0] if matches else None

    def authorize_change(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_state_authorization("change")

    def grant_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_grant("grant")

    def create_indefinite_dual_ownership(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_indefinite_dual_authority()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateOwnershipModel":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _MODEL_FIELDS)
        model = cls(
            repository_tree=mapping["repository_tree"],
            freshness=mapping["freshness"],
            items=mapping["items"],
            conflicts=mapping["conflicts"],
            migrations=mapping["migrations"],
            schema=mapping["schema"],
            version=mapping["version"],
            can_authorize_changes=mapping["can_authorize_changes"],
            can_grant_authority=mapping["can_grant_authority"],
            can_create_indefinite_dual_ownership=mapping[
                "can_create_indefinite_dual_ownership"
            ],
        )
        if mapping["content_identity"] != model.content_identity:
            raise StateOwnershipError("state-ownership content identity mismatch")
        return model

    from_dict = from_mapping

    @classmethod
    def from_json(cls, payload: str) -> "StateOwnershipModel":
        if type(payload) is not str or not payload:
            raise StateOwnershipError(
                "state-ownership JSON must be a nonempty string"
            )
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StateOwnershipError("state-ownership JSON is malformed") from exc
        if not isinstance(decoded, Mapping):
            raise StateOwnershipError(
                "state-ownership JSON must contain an object"
            )
        return cls.from_mapping(decoded)


def refuse_state_authorization(action: str) -> None:
    """Reject attempts to treat the model as change authority."""

    name = _require_text(action, "action", error_type=StateOwnershipError)
    raise StateOwnershipAuthorityError(
        f"state ownership model cannot authorize {name}"
    )


def refuse_authority_grant(action: str) -> None:
    """Reject attempts to grant store authority through this model."""

    name = _require_text(action, "action", error_type=StateOwnershipError)
    raise StateOwnershipAuthorityError(
        f"state ownership model cannot {name} authority"
    )


def refuse_indefinite_dual_authority() -> None:
    """Reject unbounded dual-write or dual-authority proposals."""

    raise StateOwnershipAuthorityError(
        "state ownership model cannot create indefinite dual ownership"
    )


def accept_unknown_owner(*_args: Any, **_kwargs: Any) -> None:
    """Unknown ownership is a hard conflict, never an accepted owner."""

    raise StateOwnershipError("unknown-owner acceptance is prohibited")


def _binding_provenance(
    binding: StateStoreBinding,
    *,
    repository_tree: str,
    freshness: str,
    confidence: Confidence = Confidence.EXACT,
) -> SourceFactIdentity:
    try:
        return SourceFactIdentity(
            extractor_identity=EXTRACTOR_IDENTITY,
            span=SourceSpan(binding.path, binding.start_line, binding.end_line),
            confidence=confidence,
            freshness=freshness,
            repository_tree=repository_tree,
        )
    except ArchitectureContractError as exc:
        raise _wrap_contract(exc) from exc


def nominations_from_bindings(
    bindings: Sequence[StateStoreBinding] = CURRENT_TREE_STORE_BINDINGS,
    *,
    repository_tree: str = DEFAULT_REPOSITORY_TREE,
    freshness: str = DEFAULT_FRESHNESS,
) -> tuple[StateNomination, ...]:
    """Project reviewed current-tree bindings into classification nominations."""

    return tuple(
        StateNomination(
            kind=binding.kind,
            path=binding.path,
            disposition=binding.disposition,
            provenance=_binding_provenance(
                binding, repository_tree=repository_tree, freshness=freshness
            ),
            store_locator=binding.store_locator,
            nominated_owner=binding.nominated_owner,
            uncertainty=binding.uncertainty,
            tables=binding.tables,
            rebuildable=binding.rebuildable,
        )
        for binding in bindings
    )


def default_unknown_migrations(
    items: Sequence[StateItem],
) -> tuple[StateMigrationPlan, ...]:
    """Propose bounded migrations that would end unknown dual-read/write."""

    plans: list[StateMigrationPlan] = []
    for item in items:
        if item.disposition is not StateDisposition.UNKNOWN:
            continue
        if item.semantic_fact not in MUTABLE_SEMANTIC_FACTS:
            continue
        plans.append(
            plan_state_migration(
                item.semantic_fact,
                item.physical_store_id,
                DUCKDB_CONTROL_STORE_ID,
                plan_id=f"migrate-{item.item_id}-to-duckdb",
            )
        )
    return tuple(plans)


def classify_state_ownership(
    items: Sequence[StateItem | StateNomination | Mapping[str, Any]],
    *,
    repository_tree: str = DEFAULT_REPOSITORY_TREE,
    freshness: str = DEFAULT_FRESHNESS,
    migrations: Sequence[StateMigrationPlan | Mapping[str, Any]] | None = None,
) -> StateOwnershipModel:
    """Classify nominated stores and record one-owner or hard-conflict outcomes."""

    parsed = _normalize_items(items)
    plans = _normalize_plans(migrations)
    return StateOwnershipModel(
        repository_tree=repository_tree,
        freshness=freshness,
        items=parsed,
        migrations=plans,
    )


def build_state_ownership_model(
    items: Sequence[StateItem | StateNomination | Mapping[str, Any]],
    *,
    repository_tree: str = DEFAULT_REPOSITORY_TREE,
    freshness: str = DEFAULT_FRESHNESS,
    migrations: Sequence[StateMigrationPlan | Mapping[str, Any]] | None = None,
) -> StateOwnershipModel:
    """Alias for :func:`classify_state_ownership`."""

    return classify_state_ownership(
        items,
        repository_tree=repository_tree,
        freshness=freshness,
        migrations=migrations,
    )


def classify_current_tree_state_ownership(
    *,
    repository_tree: str = DEFAULT_REPOSITORY_TREE,
    freshness: str = DEFAULT_FRESHNESS,
    include_unknown_migrations: bool = True,
) -> StateOwnershipModel:
    """Classify the reviewed current-tree store inventory."""

    nominations = nominations_from_bindings(
        repository_tree=repository_tree, freshness=freshness
    )
    items = tuple(item.to_item() for item in nominations)
    migrations = (
        default_unknown_migrations(items) if include_unknown_migrations else ()
    )
    return classify_state_ownership(
        items,
        repository_tree=repository_tree,
        freshness=freshness,
        migrations=migrations,
    )


def state_items_from_inventory(
    payload: Mapping[str, Any],
    *,
    repository_tree: str = DEFAULT_REPOSITORY_TREE,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[StateNomination, ...]:
    """Project a PCAR-001 state-store inventory into nominations."""

    mapping = _require_mapping(payload, error_type=StateOwnershipError)
    stores = mapping.get("stores")
    if isinstance(stores, (str, bytes, bytearray)) or not isinstance(stores, Sequence):
        raise StateOwnershipError("state-store inventory stores must be a list")
    nominations: list[StateNomination] = []
    for raw in stores:
        record = _require_mapping(raw, error_type=StateOwnershipError)
        kind = _closed_enum(
            record.get("kind"),
            StoreKind,
            "store kind",
            error_type=StateOwnershipError,
        )
        disposition = _closed_enum(
            record.get("disposition"),
            StateDisposition,
            "state disposition",
            error_type=StateOwnershipError,
        )
        span_payload = record.get("source_span")
        if not isinstance(span_payload, Mapping):
            raise StateOwnershipError("inventory source_span must be an object")
        span = SourceSpan.from_mapping(
            {
                "path": span_payload.get("path"),
                "start_line": span_payload.get("start_line"),
                "end_line": span_payload.get("end_line"),
            }
        )
        locator = _require_optional_text(record.get("path"), "path")
        if kind is StoreKind.RECEIPT_STATE:
            locator = f"{DUCKDB_CONTROL_LOCATOR}#completion_receipts"
        tables = record.get("tables") or ()
        uncertainty = record.get("uncertainty")
        nominated = record.get("nominated_owner") or ""
        if disposition is StateDisposition.AUTHORITATIVE and not nominated:
            nominated = DUCKDB_CONTROL_STORE_ID
        if disposition in REBUILDABLE_DISPOSITIONS and not nominated:
            nominated = DUCKDB_CONTROL_STORE_ID
        nominations.append(
            StateNomination(
                kind=kind,
                path=span.path,
                disposition=disposition,
                provenance=SourceFactIdentity(
                    extractor_identity=extractor_identity,
                    span=span,
                    confidence=(
                        Confidence.EXACT
                        if disposition is not StateDisposition.UNKNOWN
                        else Confidence.CONSERVATIVE
                    ),
                    freshness=freshness,
                    repository_tree=repository_tree,
                ),
                store_locator=locator,
                nominated_owner=str(nominated),
                uncertainty="" if uncertainty is None else str(uncertainty),
                tables=tuple(tables),
                rebuildable=_inferred_rebuildable(disposition),
            )
        )
    return tuple(nominations)
