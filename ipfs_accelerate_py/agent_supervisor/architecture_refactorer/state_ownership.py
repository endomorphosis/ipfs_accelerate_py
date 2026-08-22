"""State-ownership model for mutable semantic facts (PCAR-014).

`StateOwnershipModel` classifies DuckDB tables, JSON and Markdown files,
in-memory registries, events, caches, worktree metadata, leases, and
provider/goal/task/completion/receipt state as `authoritative`,
`materialized_projection`, `cache`, `historical_event`, `fixture`,
`legacy`, or `unknown`. Every mutable semantic fact has exactly one
authoritative store or a typed hard blocker. Migration uses a closed,
bounded phase sequence and never leaves indefinite dual authority.
Projections and caches are rebuildable and cannot be authoritative.
The model records existing store authority; it cannot mutate stores,
grant authority, or treat Markdown/dashboard surfaces as owners.
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

from .architecture_ir import ArchitectureIR
from .contracts import (
    ArchitectureContractError,
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
    _closed_enum,
    _require_int,
    _require_mapping,
    _require_text,
    NON_PROBATIVE_CONFIDENCE,
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
EFFECT_CLASS = "read_only_analysis"
MODEL_CAN_MUTATE_STORES = False
MODEL_CAN_GRANT_AUTHORITY = False
MODEL_CAN_CREATE_DUAL_AUTHORITY = False
MARKDOWN_CAN_BE_AUTHORITATIVE = False
DASHBOARD_CAN_BE_AUTHORITATIVE = False
PROJECTION_CAN_BE_AUTHORITATIVE = False
CACHE_CAN_BE_AUTHORITATIVE = False
UNKNOWN_OWNER_ACCEPTED = False
INDEFINITE_DUAL_AUTHORITY_PROHIBITED = True
CONTENT_IDENTITY_IS_NOT_AUTHORITY = True
DUCKDB_QUACK_OWNER = "DuckDB plus Quack exclusive state-owner boundary"

_UNKNOWN_FIELD_MESSAGE = "unknown state-ownership field"
_MISSING_FIELD_MESSAGE = "missing state-ownership field"
_CID_PREFIXES = ("bagu", "bafy", "bafk", "sha256:")
_OWNER_EDGE_KINDS = frozenset(
    {EdgeKind.PERSISTS, EdgeKind.WRITES, EdgeKind.MUTATES}
)
_DASHBOARD_MARKERS = ("dashboard",)

FACT_CONTROL_PLANE_STORE = "control_plane.store"
FACT_JSON_INVENTORY = "inventory.classification"
FACT_TASK_RECORD = "task.record"
FACT_GOAL_RECORD = "goal.record"
FACT_LEASE_RECORD = "lease.record"
FACT_WORKTREE_BINDING = "worktree.binding"
FACT_PROVIDER_INVOCATION = "provider.invocation"
FACT_COMPLETION_RECORD = "completion.record"
FACT_RECEIPT_RECORD = "receipt.record"
FACT_DOMAIN_EVENT = "event.domain"
FACT_ANALYSIS_CACHE = "analysis.receipt"
FACT_DAEMON_REGISTRY = "registry.daemon.membership"

REQUIRED_MUTABLE_FACTS: tuple[str, ...] = (
    FACT_CONTROL_PLANE_STORE,
    FACT_GOAL_RECORD,
    FACT_TASK_RECORD,
    FACT_LEASE_RECORD,
    FACT_WORKTREE_BINDING,
    FACT_PROVIDER_INVOCATION,
    FACT_COMPLETION_RECORD,
    FACT_RECEIPT_RECORD,
)


class StateOwnershipError(ArchitectureContractError):
    """Fail-closed state-ownership contract violation."""


class StateOwnershipAuthorityError(StateOwnershipError):
    """Raised when the model is asked to mutate stores or grant authority."""


class StoreKind(str, Enum):
    """Closed store-class vocabulary accepted by the state inventory."""

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


class StateDisposition(str, Enum):
    """Closed store-disposition vocabulary (PCAR-PLAN-R1)."""

    AUTHORITATIVE = "authoritative"
    MATERIALIZED_PROJECTION = "materialized_projection"
    CACHE = "cache"
    HISTORICAL_EVENT = "historical_event"
    FIXTURE = "fixture"
    LEGACY = "legacy"
    UNKNOWN = "unknown"


CLOSED_DISPOSITIONS: frozenset[str] = frozenset(
    item.value for item in StateDisposition
)
REQUIRED_DISPOSITIONS: tuple[StateDisposition, ...] = tuple(StateDisposition)
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
PROHIBITED_AUTHORITATIVE_KINDS: frozenset[StoreKind] = frozenset(
    {
        StoreKind.JSON_FILES,
        StoreKind.MARKDOWN_TASK_BOARDS,
        StoreKind.IN_MEMORY_REGISTRIES,
        StoreKind.EVENT_LOGS,
        StoreKind.CACHE_NAMESPACES,
    }
)


class MigrationPhase(str, Enum):
    """Closed bounded-migration phase vocabulary (PCAR-PLAN-R1)."""

    SNAPSHOT = "snapshot"
    DUAL_READ_SHADOW = "dual_read_shadow"
    CONTROLLED_DUAL_WRITE = "controlled_dual_write"
    CUTOVER = "cutover"
    VALIDATION = "validation"
    READ_ONLY_LEGACY = "read_only_legacy"
    RETIREMENT = "retirement"


REQUIRED_MIGRATION_PHASES: tuple[MigrationPhase, ...] = tuple(MigrationPhase)
CLOSED_MIGRATION_PHASES: frozenset[str] = frozenset(
    item.value for item in MigrationPhase
)


class StateConflictKind(str, Enum):
    """Closed hard-blocker vocabulary for unresolved state ownership."""

    UNKNOWN_OWNER = "unknown_owner"
    MULTIPLE_AUTHORITATIVE_STORES = "multiple_authoritative_stores"
    MISSING_AUTHORITATIVE_STORE = "missing_authoritative_store"
    PROJECTION_CLAIMED_AUTHORITY = "projection_claimed_authority"
    CACHE_CLAIMED_AUTHORITY = "cache_claimed_authority"
    MARKDOWN_CLAIMED_AUTHORITY = "markdown_claimed_authority"
    DASHBOARD_CLAIMED_AUTHORITY = "dashboard_claimed_authority"
    INDEFINITE_DUAL_AUTHORITY = "indefinite_dual_authority"
    UNBOUNDED_MIGRATION = "unbounded_migration"
    MISSING_MIGRATION_PHASE = "missing_migration_phase"
    NON_REBUILDABLE_PROJECTION = "non_rebuildable_projection"
    UNCLASSIFIED_STORE = "unclassified_store"
    CONFLICTING_DISPOSITION = "conflicting_disposition"
    LEGACY_WITHOUT_CUTOVER = "legacy_without_cutover"
    UNKNOWN_PRODUCTION_OWNER = "unknown_production_owner"
    NON_PROBATIVE_AUTHORITY = "non_probative_authority"
    GRAPH_OWNER_AMBIGUITY = "graph_owner_ambiguity"


CLOSED_CONFLICT_KINDS: frozenset[str] = frozenset(
    item.value for item in StateConflictKind
)


_ITEM_FIELDS = frozenset(
    {
        "content_identity",
        "disposition",
        "fact_id",
        "item_id",
        "kind",
        "nominated_owner",
        "path",
        "provenance",
        "rebuildable",
        "schema",
        "tables",
        "uncertainty",
        "version",
        "writable",
    }
)
_PHASE_FIELDS = frozenset(
    {
        "dual_read",
        "dual_write",
        "legacy_read_only",
        "phase",
        "retired",
        "source_writable",
        "target_writable",
    }
)
_PLAN_FIELDS = frozenset(
    {
        "content_identity",
        "ends_dual_read_write",
        "fact_id",
        "grants_authority",
        "indefinite_dual_authority",
        "phases",
        "plan_id",
        "schema",
        "source_item_id",
        "target_item_id",
        "version",
    }
)
_CONFLICT_FIELDS = frozenset(
    {
        "content_identity",
        "fact_id",
        "item_ids",
        "kind",
        "message",
        "schema",
        "version",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "architecture_ir_identity",
        "can_create_dual_authority",
        "can_grant_authority",
        "can_mutate_stores",
        "conflicts",
        "content_identity",
        "effect_class",
        "freshness",
        "items",
        "migrations",
        "repository_tree",
        "schema",
        "version",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "disposition",
        "fact_id",
        "item_id",
        "kind",
        "nominated_owner",
        "nominated_symbol",
        "path",
        "rebuildable",
        "source_path",
        "start_line",
        "end_line",
        "tables",
        "uncertainty",
        "writable",
    }
)
_INVENTORY_STORE_FIELDS = frozenset(
    {
        "disposition",
        "kind",
        "path",
        "source_span",
    }
)
_INVENTORY_OPTIONAL_FIELDS = frozenset(
    {
        "nominated_owner",
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


def _require_text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StateOwnershipError(f"{name} must be a list of strings")
    items = tuple(
        _require_text(item, f"{name} item", error_type=StateOwnershipError)
        for item in value
    )
    return tuple(sorted(set(items)))


def _looks_like_content_identity(value: str) -> bool:
    return value.startswith(_CID_PREFIXES)


def _wrap_contract(exc: ArchitectureContractError) -> StateOwnershipError:
    if isinstance(exc, StateOwnershipError):
        return exc
    return StateOwnershipError(str(exc))


def _optional_text(value: Any, name: str) -> str:
    if value is None:
        return ""
    if type(value) is not str or "\x00" in value:
        raise StateOwnershipError(f"{name} must be a string")
    return value


def _is_dashboard_path(path: str) -> bool:
    lowered = path.replace("\\", "/").lower()
    return any(marker in lowered for marker in _DASHBOARD_MARKERS)


def _is_markdown_path(path: str) -> bool:
    return path.replace("\\", "/").lower().endswith(".md")


def _record_tuple(
    value: Any,
    record_type: type[Any],
    name: str,
) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise StateOwnershipError(f"{name} must be a sequence")
    records = tuple(
        item if isinstance(item, record_type) else record_type.from_mapping(item)
        for item in value
    )
    return records


@dataclass(frozen=True)
class StateSourceBinding:
    """Current-tree observational binding for one classified store."""

    item_id: str
    kind: StoreKind
    fact_id: str
    path: str
    disposition: StateDisposition
    nominated_owner: str
    nominated_symbol: str
    source_path: str
    start_line: int
    end_line: int
    rebuildable: bool
    writable: bool
    tables: tuple[str, ...] = ()
    uncertainty: str = ""


CURRENT_STATE_BINDINGS: tuple[StateSourceBinding, ...] = (
    StateSourceBinding(
        item_id="duckdb-control",
        kind=StoreKind.DUCKDB_TABLES,
        fact_id=FACT_CONTROL_PLANE_STORE,
        path="{state_root}/control.duckdb",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="schema_contracts",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=12,
        end_line=12,
        rebuildable=False,
        writable=True,
        tables=(
            "goals",
            "tasks",
            "leases",
            "completion_receipts",
            "domain_events",
            "worktrees",
            "provider_invocations",
        ),
        uncertainty="runtime_state_root_binding",
    ),
    StateSourceBinding(
        item_id="json-inventory",
        kind=StoreKind.JSON_FILES,
        fact_id=FACT_JSON_INVENTORY,
        path="docs/architecture/architecture_refactorer_inventory",
        disposition=StateDisposition.MATERIALIZED_PROJECTION,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="required_kinds",
        source_path=(
            "docs/architecture/architecture_refactorer_inventory/"
            "state_store_baseline.json"
        ),
        start_line=1,
        end_line=1,
        rebuildable=True,
        writable=False,
        uncertainty="multiple_json_sinks_exist_this_inventory_nominates_without_migrating",
    ),
    StateSourceBinding(
        item_id="markdown-todo",
        kind=StoreKind.MARKDOWN_TASK_BOARDS,
        fact_id=FACT_TASK_RECORD,
        path="docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        disposition=StateDisposition.MATERIALIZED_PROJECTION,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="PCAR-001",
        source_path=(
            "docs/architecture/agent_supervisor_architecture_refactorer.todo.md"
        ),
        start_line=108,
        end_line=108,
        rebuildable=True,
        writable=False,
    ),
    StateSourceBinding(
        item_id="registry-daemons",
        kind=StoreKind.IN_MEMORY_REGISTRIES,
        fact_id=FACT_DAEMON_REGISTRY,
        path="ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py",
        disposition=StateDisposition.FIXTURE,
        nominated_owner="TodoDaemonRegistration catalog",
        nominated_symbol="DEFAULT_DAEMON_REGISTRATIONS",
        source_path="ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py",
        start_line=27,
        end_line=38,
        rebuildable=False,
        writable=False,
        uncertainty="dynamic_registry_membership",
    ),
    StateSourceBinding(
        item_id="events-domain",
        kind=StoreKind.EVENT_LOGS,
        fact_id=FACT_DOMAIN_EVENT,
        path="{state_root}/control.duckdb#domain_events",
        disposition=StateDisposition.HISTORICAL_EVENT,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="domain_events",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=827,
        end_line=827,
        rebuildable=False,
        writable=False,
        tables=("domain_events",),
        uncertainty="jsonl_sidecars_remain_legacy_append_streams",
    ),
    StateSourceBinding(
        item_id="cache-analysis",
        kind=StoreKind.CACHE_NAMESPACES,
        fact_id=FACT_ANALYSIS_CACHE,
        path="ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py",
        disposition=StateDisposition.CACHE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="AnalysisCache",
        source_path="ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py",
        start_line=846,
        end_line=846,
        rebuildable=True,
        writable=False,
        uncertainty="cache_directories_are_runtime_layout",
    ),
    StateSourceBinding(
        item_id="worktree-bindings",
        kind=StoreKind.WORKTREE_METADATA,
        fact_id=FACT_WORKTREE_BINDING,
        path="{state_root}/control.duckdb#worktrees",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="DatabaseWorktreeRegistry",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/merge/"
            "database_worktree_registry.py"
        ),
        start_line=1632,
        end_line=1632,
        rebuildable=False,
        writable=True,
        tables=("worktrees",),
        uncertainty="git_worktree_directories_are_os_bootstrap_not_task_authority",
    ),
    StateSourceBinding(
        item_id="leases",
        kind=StoreKind.LEASE_RECORDS,
        fact_id=FACT_LEASE_RECORD,
        path="{state_root}/control.duckdb#leases",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="leases",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=339,
        end_line=339,
        rebuildable=False,
        writable=True,
        tables=("leases",),
        uncertainty="duckdb_table_binding",
    ),
    StateSourceBinding(
        item_id="provider-invocations",
        kind=StoreKind.PROVIDER_STATE,
        fact_id=FACT_PROVIDER_INVOCATION,
        path="{state_root}/control.duckdb#provider_invocations",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="provider_invocations",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=722,
        end_line=722,
        rebuildable=False,
        writable=True,
        tables=("provider_invocations",),
        uncertainty="duckdb_table_binding",
    ),
    StateSourceBinding(
        item_id="provider-attempt-store",
        kind=StoreKind.PROVIDER_STATE,
        fact_id=FACT_PROVIDER_INVOCATION,
        path="ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
        disposition=StateDisposition.LEGACY,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="ProviderAttemptReservation",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py"
        ),
        start_line=376,
        end_line=376,
        rebuildable=False,
        writable=False,
        uncertainty="provider_runtime",
    ),
    StateSourceBinding(
        item_id="goals",
        kind=StoreKind.GOAL_STATE,
        fact_id=FACT_GOAL_RECORD,
        path="{state_root}/control.duckdb#goals",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="goals",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=407,
        end_line=407,
        rebuildable=False,
        writable=True,
        tables=("goals",),
        uncertainty="duckdb_table_binding",
    ),
    StateSourceBinding(
        item_id="tasks",
        kind=StoreKind.TASK_STATE,
        fact_id=FACT_TASK_RECORD,
        path="{state_root}/control.duckdb#tasks",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="tasks",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=470,
        end_line=470,
        rebuildable=False,
        writable=True,
        tables=("tasks",),
        uncertainty="duckdb_table_binding",
    ),
    StateSourceBinding(
        item_id="completion-receipts",
        kind=StoreKind.COMPLETION_STATE,
        fact_id=FACT_COMPLETION_RECORD,
        path="{state_root}/control.duckdb#completion_receipts",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="completion_receipts",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=809,
        end_line=809,
        rebuildable=False,
        writable=True,
        tables=("completion_receipts",),
        uncertainty="duckdb_table_binding",
    ),
    StateSourceBinding(
        item_id="receipts",
        kind=StoreKind.RECEIPT_STATE,
        fact_id=FACT_RECEIPT_RECORD,
        path="{state_root}/control.duckdb#completion_receipts",
        disposition=StateDisposition.AUTHORITATIVE,
        nominated_owner=DUCKDB_QUACK_OWNER,
        nominated_symbol="completion_receipts",
        source_path=(
            "ipfs_accelerate_py/agent_supervisor/task_sources/sql/"
            "0001_control_plane.sql"
        ),
        start_line=809,
        end_line=809,
        rebuildable=False,
        writable=True,
        tables=("completion_receipts",),
        uncertainty="duckdb_table_binding",
    ),
)


def _phase_defaults(phase: MigrationPhase) -> dict[str, bool]:
    if phase is MigrationPhase.SNAPSHOT:
        return {
            "dual_read": False,
            "dual_write": False,
            "source_writable": True,
            "target_writable": False,
            "legacy_read_only": False,
            "retired": False,
        }
    if phase is MigrationPhase.DUAL_READ_SHADOW:
        return {
            "dual_read": True,
            "dual_write": False,
            "source_writable": True,
            "target_writable": False,
            "legacy_read_only": False,
            "retired": False,
        }
    if phase is MigrationPhase.CONTROLLED_DUAL_WRITE:
        return {
            "dual_read": True,
            "dual_write": True,
            "source_writable": True,
            "target_writable": True,
            "legacy_read_only": False,
            "retired": False,
        }
    if phase is MigrationPhase.CUTOVER:
        return {
            "dual_read": True,
            "dual_write": False,
            "source_writable": False,
            "target_writable": True,
            "legacy_read_only": False,
            "retired": False,
        }
    if phase is MigrationPhase.VALIDATION:
        return {
            "dual_read": True,
            "dual_write": False,
            "source_writable": False,
            "target_writable": True,
            "legacy_read_only": False,
            "retired": False,
        }
    if phase is MigrationPhase.READ_ONLY_LEGACY:
        return {
            "dual_read": True,
            "dual_write": False,
            "source_writable": False,
            "target_writable": True,
            "legacy_read_only": True,
            "retired": False,
        }
    return {
        "dual_read": False,
        "dual_write": False,
        "source_writable": False,
        "target_writable": True,
        "legacy_read_only": True,
        "retired": True,
    }


@dataclass(frozen=True)
class StateMigrationPhase:
    """One bounded migration phase with explicit dual-read/write flags."""

    phase: MigrationPhase
    dual_read: bool
    dual_write: bool
    source_writable: bool
    target_writable: bool
    legacy_read_only: bool
    retired: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase",
            _closed_enum(
                self.phase,
                MigrationPhase,
                "migration phase",
                error_type=StateOwnershipError,
            ),
        )
        dual_read = _require_bool(self.dual_read, "dual_read")
        dual_write = _require_bool(self.dual_write, "dual_write")
        source_writable = _require_bool(self.source_writable, "source_writable")
        target_writable = _require_bool(self.target_writable, "target_writable")
        legacy_read_only = _require_bool(self.legacy_read_only, "legacy_read_only")
        retired = _require_bool(self.retired, "retired")
        expected = _phase_defaults(self.phase)
        actual = {
            "dual_read": dual_read,
            "dual_write": dual_write,
            "source_writable": source_writable,
            "target_writable": target_writable,
            "legacy_read_only": legacy_read_only,
            "retired": retired,
        }
        if actual != expected:
            raise StateOwnershipError(
                f"migration phase {self.phase.value} flags are not bounded"
            )
        object.__setattr__(self, "dual_read", dual_read)
        object.__setattr__(self, "dual_write", dual_write)
        object.__setattr__(self, "source_writable", source_writable)
        object.__setattr__(self, "target_writable", target_writable)
        object.__setattr__(self, "legacy_read_only", legacy_read_only)
        object.__setattr__(self, "retired", retired)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dual_read": self.dual_read,
            "dual_write": self.dual_write,
            "legacy_read_only": self.legacy_read_only,
            "phase": self.phase.value,
            "retired": self.retired,
            "source_writable": self.source_writable,
            "target_writable": self.target_writable,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateMigrationPhase":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _PHASE_FIELDS)
        return cls(
            phase=mapping["phase"],
            dual_read=mapping["dual_read"],
            dual_write=mapping["dual_write"],
            source_writable=mapping["source_writable"],
            target_writable=mapping["target_writable"],
            legacy_read_only=mapping["legacy_read_only"],
            retired=mapping["retired"],
        )

    from_dict = from_mapping

    @classmethod
    def for_phase(cls, phase: MigrationPhase | str) -> "StateMigrationPhase":
        parsed = _closed_enum(
            phase, MigrationPhase, "migration phase", error_type=StateOwnershipError
        )
        return cls(phase=parsed, **_phase_defaults(parsed))


@dataclass(frozen=True)
class StateItem:
    """One classified store bound to a single semantic fact."""

    item_id: str
    kind: StoreKind
    fact_id: str
    path: str
    disposition: StateDisposition
    nominated_owner: str
    provenance: SourceFactIdentity
    rebuildable: bool
    writable: bool
    tables: tuple[str, ...] = ()
    uncertainty: str = ""
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
        item_id = _require_text(self.item_id, "item_id", error_type=StateOwnershipError)
        if _looks_like_content_identity(item_id):
            raise StateOwnershipError("content identity is not inferred to be authority")
        kind = _closed_enum(
            self.kind, StoreKind, "store kind", error_type=StateOwnershipError
        )
        fact_id = _require_text(self.fact_id, "fact_id", error_type=StateOwnershipError)
        path = _require_text(self.path, "path", error_type=StateOwnershipError)
        disposition = _closed_enum(
            self.disposition,
            StateDisposition,
            "state disposition",
            error_type=StateOwnershipError,
        )
        nominated_owner = _require_text(
            self.nominated_owner, "nominated_owner", error_type=StateOwnershipError
        )
        if _looks_like_content_identity(nominated_owner):
            raise StateOwnershipError("content identity is not inferred to be authority")
        provenance = (
            self.provenance
            if isinstance(self.provenance, SourceFactIdentity)
            else SourceFactIdentity.from_mapping(self.provenance)
        )
        rebuildable = _require_bool(self.rebuildable, "rebuildable")
        writable = _require_bool(self.writable, "writable")
        tables = _require_text_tuple(self.tables, "tables")
        uncertainty = _optional_text(self.uncertainty, "uncertainty")
        _validate_item_invariants(
            kind=kind,
            path=path,
            disposition=disposition,
            provenance=provenance,
            rebuildable=rebuildable,
            writable=writable,
        )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fact_id", fact_id)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "nominated_owner", nominated_owner)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "rebuildable", rebuildable)
        object.__setattr__(self, "writable", writable)
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "uncertainty", uncertainty)
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
            "fact_id": self.fact_id,
            "item_id": self.item_id,
            "kind": self.kind.value,
            "nominated_owner": self.nominated_owner,
            "path": self.path,
            "provenance": self.provenance.to_dict(),
            "rebuildable": self.rebuildable,
            "schema": self.schema,
            "tables": list(self.tables),
            "uncertainty": self.uncertainty,
            "version": self.version,
            "writable": self.writable,
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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateItem":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _ITEM_FIELDS)
        try:
            item = cls(
                item_id=mapping["item_id"],
                kind=mapping["kind"],
                fact_id=mapping["fact_id"],
                path=mapping["path"],
                disposition=mapping["disposition"],
                nominated_owner=mapping["nominated_owner"],
                provenance=mapping["provenance"],
                rebuildable=mapping["rebuildable"],
                writable=mapping["writable"],
                tables=mapping["tables"],
                uncertainty=mapping["uncertainty"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != item.content_identity:
            raise StateOwnershipError("state-item content identity mismatch")
        return item

    from_dict = from_mapping


def _validate_item_invariants(
    *,
    kind: StoreKind,
    path: str,
    disposition: StateDisposition,
    provenance: SourceFactIdentity,
    rebuildable: bool,
    writable: bool,
) -> None:
    if disposition is StateDisposition.AUTHORITATIVE:
        if kind in PROHIBITED_AUTHORITATIVE_KINDS:
            if kind is StoreKind.MARKDOWN_TASK_BOARDS:
                raise StateOwnershipError("Markdown cannot be an authoritative store")
            if kind is StoreKind.CACHE_NAMESPACES:
                raise StateOwnershipError("cache cannot be an authoritative store")
            if kind is StoreKind.JSON_FILES:
                raise StateOwnershipError(
                    "JSON projections cannot be an authoritative store"
                )
            if kind is StoreKind.EVENT_LOGS:
                raise StateOwnershipError(
                    "historical events cannot be an authoritative store"
                )
            raise StateOwnershipError(
                f"{kind.value} cannot be an authoritative store"
            )
        if _is_markdown_path(path) or kind is StoreKind.MARKDOWN_TASK_BOARDS:
            raise StateOwnershipError("Markdown cannot be an authoritative store")
        if _is_dashboard_path(path):
            raise StateOwnershipError("dashboard cannot be an authoritative store")
        if rebuildable:
            raise StateOwnershipError(
                "authoritative stores are not rebuildable projections"
            )
        if not writable:
            raise StateOwnershipError("authoritative stores must be writable")
        if provenance.confidence in NON_PROBATIVE_CONFIDENCE:
            raise StateOwnershipError(
                "heuristic or opaque facts cannot prove store authority"
            )
        return
    if writable:
        raise StateOwnershipError(
            f"{disposition.value} stores cannot accept production writes"
        )
    if disposition in REBUILDABLE_DISPOSITIONS and not rebuildable:
        raise StateOwnershipError(
            f"{disposition.value} stores must be rebuildable projections"
        )
    if disposition not in REBUILDABLE_DISPOSITIONS and rebuildable:
        raise StateOwnershipError(
            f"{disposition.value} stores are not rebuildable projections"
        )


@dataclass(frozen=True)
class StateConflict:
    """Typed hard blocker that prevents one-owner closure."""

    kind: StateConflictKind
    fact_id: str
    message: str
    item_ids: tuple[str, ...] = ()
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
        fact_id = _require_text(self.fact_id, "fact_id", error_type=StateOwnershipError)
        message = _require_text(self.message, "message", error_type=StateOwnershipError)
        item_ids = _require_text_tuple(self.item_ids, "item_ids")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fact_id", fact_id)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "item_ids", item_ids)
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
            "fact_id": self.fact_id,
            "item_ids": list(self.item_ids),
            "kind": self.kind.value,
            "message": self.message,
            "schema": self.schema,
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
            fact_id=mapping["fact_id"],
            message=mapping["message"],
            item_ids=mapping["item_ids"],
            schema=mapping["schema"],
            version=mapping["version"],
        )
        if mapping["content_identity"] != conflict.content_identity:
            raise StateOwnershipError("state-conflict content identity mismatch")
        return conflict

    from_dict = from_mapping


@dataclass(frozen=True)
class StateMigrationPlan:
    """Bounded snapshot-to-retirement migration that ends dual authority."""

    plan_id: str
    fact_id: str
    source_item_id: str
    target_item_id: str
    phases: tuple[StateMigrationPhase, ...]
    ends_dual_read_write: bool = True
    grants_authority: bool = False
    indefinite_dual_authority: bool = False
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
        fact_id = _require_text(self.fact_id, "fact_id", error_type=StateOwnershipError)
        source_item_id = _require_text(
            self.source_item_id, "source_item_id", error_type=StateOwnershipError
        )
        target_item_id = _require_text(
            self.target_item_id, "target_item_id", error_type=StateOwnershipError
        )
        if source_item_id == target_item_id:
            raise StateOwnershipError("migration source and target must differ")
        phases = _record_tuple(self.phases, StateMigrationPhase, "phases")
        observed = tuple(item.phase for item in phases)
        if observed != REQUIRED_MIGRATION_PHASES:
            missing = [
                item.value
                for item in REQUIRED_MIGRATION_PHASES
                if item not in set(observed)
            ]
            if missing:
                raise StateOwnershipError(
                    f"missing migration phase: {missing}"
                )
            raise StateOwnershipError(
                "migration phases must be the closed bounded sequence"
            )
        dual_write_phases = tuple(
            item.phase.value for item in phases if item.dual_write
        )
        if dual_write_phases != (MigrationPhase.CONTROLLED_DUAL_WRITE.value,):
            raise StateOwnershipError(
                "dual-write is allowed only during formally controlled dual-write"
            )
        terminal = phases[-1]
        if not terminal.retired or terminal.dual_read or terminal.dual_write:
            raise StateOwnershipError("migration must retire dual-read/write")
        ends = _require_bool(self.ends_dual_read_write, "ends_dual_read_write")
        grants = _require_bool(self.grants_authority, "grants_authority")
        indefinite = _require_bool(
            self.indefinite_dual_authority, "indefinite_dual_authority"
        )
        if not ends:
            raise StateOwnershipError("migration plans must end dual-read/write")
        if grants:
            raise StateOwnershipAuthorityError(
                "state-ownership model cannot grant authority"
            )
        if indefinite:
            raise StateOwnershipError("indefinite dual authority is prohibited")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "fact_id", fact_id)
        object.__setattr__(self, "source_item_id", source_item_id)
        object.__setattr__(self, "target_item_id", target_item_id)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "ends_dual_read_write", True)
        object.__setattr__(self, "grants_authority", False)
        object.__setattr__(self, "indefinite_dual_authority", False)
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
            "ends_dual_read_write": True,
            "fact_id": self.fact_id,
            "grants_authority": False,
            "indefinite_dual_authority": False,
            "phases": [item.to_dict() for item in self.phases],
            "plan_id": self.plan_id,
            "schema": self.schema,
            "source_item_id": self.source_item_id,
            "target_item_id": self.target_item_id,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        identity = _content_identity(payload)
        if self.content_identity != identity:
            raise StateOwnershipError("state-migration content identity mismatch")
        return {**payload, "content_identity": identity}

    def dual_write_window(self) -> tuple[MigrationPhase, ...]:
        return tuple(item.phase for item in self.phases if item.dual_write)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateMigrationPlan":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _PLAN_FIELDS)
        try:
            plan = cls(
                plan_id=mapping["plan_id"],
                fact_id=mapping["fact_id"],
                source_item_id=mapping["source_item_id"],
                target_item_id=mapping["target_item_id"],
                phases=mapping["phases"],
                ends_dual_read_write=mapping["ends_dual_read_write"],
                grants_authority=mapping["grants_authority"],
                indefinite_dual_authority=mapping["indefinite_dual_authority"],
                schema=mapping["schema"],
                version=mapping["version"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
        if mapping["content_identity"] != plan.content_identity:
            raise StateOwnershipError("state-migration content identity mismatch")
        return plan

    from_dict = from_mapping


def _conflict_sort_key(item: StateConflict) -> tuple[str, str, str]:
    return (item.kind.value, item.fact_id, item.message)


def _item_sort_key(item: StateItem) -> str:
    return item.item_id


def _plan_sort_key(item: StateMigrationPlan) -> str:
    return item.plan_id


def detect_state_conflicts(
    items: Sequence[StateItem],
    migrations: Sequence[StateMigrationPlan] = (),
) -> tuple[StateConflict, ...]:
    """Return hard blockers for unknown, dual, or missing authority."""

    ordered = tuple(sorted(items, key=_item_sort_key))
    by_id = {item.item_id: item for item in ordered}
    if len(by_id) != len(ordered):
        raise StateOwnershipError("state item ids must be unique")
    by_fact: dict[str, list[StateItem]] = {}
    for item in ordered:
        by_fact.setdefault(item.fact_id, []).append(item)
    migrated_legacy = {
        plan.source_item_id
        for plan in migrations
        if plan.ends_dual_read_write and not plan.indefinite_dual_authority
    }
    conflicts: list[StateConflict] = []
    for fact_id, group in by_fact.items():
        unknown = tuple(
            item for item in group if item.disposition is StateDisposition.UNKNOWN
        )
        authoritative = tuple(item for item in group if item.is_authoritative)
        legacy = tuple(
            item for item in group if item.disposition is StateDisposition.LEGACY
        )
        if unknown:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.UNKNOWN_OWNER,
                    fact_id=fact_id,
                    message="unknown mutable-state ownership is a hard blocker",
                    item_ids=tuple(item.item_id for item in unknown),
                )
            )
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.UNKNOWN_PRODUCTION_OWNER,
                    fact_id=fact_id,
                    message="unknown production ownership cannot be accepted",
                    item_ids=tuple(item.item_id for item in unknown),
                )
            )
        if len(authoritative) > 1:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.MULTIPLE_AUTHORITATIVE_STORES,
                    fact_id=fact_id,
                    message="mutable semantic fact has more than one authoritative store",
                    item_ids=tuple(item.item_id for item in authoritative),
                )
            )
        if fact_id in REQUIRED_MUTABLE_FACTS and not authoritative:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.MISSING_AUTHORITATIVE_STORE,
                    fact_id=fact_id,
                    message="mutable semantic fact has no authoritative store",
                    item_ids=tuple(item.item_id for item in group),
                )
            )
        elif (
            not authoritative
            and legacy
            and fact_id not in REQUIRED_MUTABLE_FACTS
        ):
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.MISSING_AUTHORITATIVE_STORE,
                    fact_id=fact_id,
                    message="legacy store has no authoritative owner",
                    item_ids=tuple(item.item_id for item in legacy),
                )
            )
        for item in legacy:
            if item.item_id not in migrated_legacy:
                conflicts.append(
                    StateConflict(
                        kind=StateConflictKind.LEGACY_WITHOUT_CUTOVER,
                        fact_id=fact_id,
                        message="legacy store requires a bounded cutover plan",
                        item_ids=(item.item_id,),
                    )
                )
        dispositions = {item.disposition for item in group if item.is_authoritative}
        if StateDisposition.AUTHORITATIVE in dispositions and unknown:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.CONFLICTING_DISPOSITION,
                    fact_id=fact_id,
                    message="unknown owner cannot coexist with an accepted authority",
                    item_ids=tuple(item.item_id for item in (*authoritative, *unknown)),
                )
            )
    for fact_id in REQUIRED_MUTABLE_FACTS:
        if fact_id not in by_fact:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.MISSING_AUTHORITATIVE_STORE,
                    fact_id=fact_id,
                    message="mutable semantic fact has no authoritative store",
                )
            )
    covered_kinds = {item.kind for item in ordered}
    missing_kinds = [
        kind.value for kind in REQUIRED_STORE_KINDS if kind not in covered_kinds
    ]
    if missing_kinds:
        conflicts.append(
            StateConflict(
                kind=StateConflictKind.UNCLASSIFIED_STORE,
                fact_id="store.classes",
                message=f"missing required store classes: {missing_kinds}",
            )
        )
    for plan in migrations:
        if plan.source_item_id not in by_id or plan.target_item_id not in by_id:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.UNBOUNDED_MIGRATION,
                    fact_id=plan.fact_id,
                    message="migration references an unclassified store",
                    item_ids=(plan.source_item_id, plan.target_item_id),
                )
            )
            continue
        source = by_id[plan.source_item_id]
        target = by_id[plan.target_item_id]
        if source.fact_id != plan.fact_id or target.fact_id != plan.fact_id:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.UNBOUNDED_MIGRATION,
                    fact_id=plan.fact_id,
                    message="migration must stay on one semantic fact",
                    item_ids=(source.item_id, target.item_id),
                )
            )
        if not target.is_authoritative:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.MISSING_AUTHORITATIVE_STORE,
                    fact_id=plan.fact_id,
                    message="migration target is not the authoritative store",
                    item_ids=(target.item_id,),
                )
            )
    unique: dict[tuple[str, str, str, tuple[str, ...]], StateConflict] = {}
    for item in conflicts:
        unique[(item.kind.value, item.fact_id, item.message, item.item_ids)] = item
    return tuple(sorted(unique.values(), key=_conflict_sort_key))


def classify_architecture_state(
    graph: ArchitectureIR | Mapping[str, Any],
) -> tuple[StateConflict, ...]:
    """Fail closed when ArchitectureIR STATE nodes lack exactly one owner."""

    architecture = (
        graph if isinstance(graph, ArchitectureIR) else ArchitectureIR.from_mapping(graph)
    )
    nodes = {node.node_id: node for node in architecture.nodes}
    owners: dict[str, set[str]] = {
        node.node_id: set()
        for node in architecture.nodes
        if node.kind is NodeKind.STATE
    }
    if not owners:
        return ()
    for edge in architecture.edges:
        if edge.kind not in _OWNER_EDGE_KINDS:
            continue
        if edge.target not in owners:
            continue
        owners[edge.target].add(edge.source)
    conflicts: list[StateConflict] = []
    for node_id, sources in owners.items():
        node = nodes[node_id]
        fact_id = node.node_id
        if len(sources) == 1:
            continue
        if not sources:
            conflicts.append(
                StateConflict(
                    kind=StateConflictKind.GRAPH_OWNER_AMBIGUITY,
                    fact_id=fact_id,
                    message="ArchitectureIR STATE node has no persist/write owner",
                    item_ids=(node_id,),
                )
            )
            continue
        conflicts.append(
            StateConflict(
                kind=StateConflictKind.GRAPH_OWNER_AMBIGUITY,
                fact_id=fact_id,
                message="ArchitectureIR STATE node has multiple persist/write owners",
                item_ids=tuple(sorted({node_id, *sources})),
            )
        )
    return tuple(sorted(conflicts, key=_conflict_sort_key))


def plan_bounded_migration(
    *,
    plan_id: str,
    fact_id: str,
    source_item_id: str,
    target_item_id: str,
) -> StateMigrationPlan:
    """Return the closed snapshot-to-retirement migration sequence."""

    return StateMigrationPlan(
        plan_id=plan_id,
        fact_id=fact_id,
        source_item_id=source_item_id,
        target_item_id=target_item_id,
        phases=tuple(
            StateMigrationPhase.for_phase(phase) for phase in REQUIRED_MIGRATION_PHASES
        ),
        ends_dual_read_write=True,
        grants_authority=False,
        indefinite_dual_authority=False,
    )


def classify_state_item(
    item: StateItem | Mapping[str, Any] | StateSourceBinding,
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
    confidence: Confidence = Confidence.EXACT,
) -> StateItem:
    """Bind one store to exactly one disposition and provenance."""

    if isinstance(item, StateItem):
        return item
    if isinstance(item, StateSourceBinding):
        return _item_from_binding(
            item,
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
            confidence=confidence,
        )
    return StateItem.from_mapping(item)


def _item_from_binding(
    binding: StateSourceBinding,
    *,
    repository_tree: str,
    freshness: str,
    extractor_identity: str,
    confidence: Confidence,
) -> StateItem:
    return StateItem(
        item_id=binding.item_id,
        kind=binding.kind,
        fact_id=binding.fact_id,
        path=binding.path,
        disposition=binding.disposition,
        nominated_owner=binding.nominated_owner,
        provenance=SourceFactIdentity(
            extractor_identity=extractor_identity,
            span=SourceSpan(
                binding.source_path, binding.start_line, binding.end_line
            ),
            confidence=confidence,
            freshness=freshness,
            repository_tree=repository_tree,
        ),
        rebuildable=binding.rebuildable,
        writable=binding.writable,
        tables=binding.tables,
        uncertainty=binding.uncertainty,
    )


def _slug_kind(kind: StoreKind) -> str:
    return kind.value.lower().replace(" ", "-")


def items_from_inventory(
    inventory: Mapping[str, Any],
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[StateItem, ...]:
    """Classify an accepted state-store inventory without migrating it."""

    mapping = _require_mapping(inventory, error_type=StateOwnershipError)
    stores = mapping.get("stores")
    if isinstance(stores, (str, bytes, bytearray)) or not isinstance(stores, Sequence):
        raise StateOwnershipError("inventory stores must be a sequence")
    items: list[StateItem] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(stores, start=1):
        store = _require_mapping(raw, error_type=StateOwnershipError)
        _reject_unknown(store, _INVENTORY_STORE_FIELDS | _INVENTORY_OPTIONAL_FIELDS)
        missing = sorted(_INVENTORY_STORE_FIELDS - set(store))
        if missing:
            raise StateOwnershipError(f"{_MISSING_FIELD_MESSAGE}: {missing}")
        kind = _closed_enum(
            store["kind"], StoreKind, "store kind", error_type=StateOwnershipError
        )
        span_payload = store["source_span"]
        if not isinstance(span_payload, Mapping):
            raise StateOwnershipError("inventory source_span must be an object")
        span = SourceSpan.from_mapping(span_payload)
        item_id = f"inventory-{index:02d}-{_slug_kind(kind)}"
        if item_id in seen_ids:
            raise StateOwnershipError("inventory store ids must be unique")
        seen_ids.add(item_id)
        disposition = _closed_enum(
            store["disposition"],
            StateDisposition,
            "state disposition",
            error_type=StateOwnershipError,
        )
        rebuildable = disposition in REBUILDABLE_DISPOSITIONS
        writable = disposition is StateDisposition.AUTHORITATIVE
        if disposition is StateDisposition.AUTHORITATIVE and (
            kind in PROHIBITED_AUTHORITATIVE_KINDS
            or _is_markdown_path(str(store["path"]))
            or _is_dashboard_path(str(store["path"]))
        ):
            # Closed authority rules outrank inventory nominations.
            disposition = StateDisposition.UNKNOWN
            rebuildable = False
            writable = False
        tables_value = store.get("tables", ())
        tables = (
            _require_text_tuple(tables_value, "tables")
            if tables_value not in (None, "")
            else ()
        )
        uncertainty = store.get("uncertainty")
        nominated = store.get("nominated_owner") or DUCKDB_QUACK_OWNER
        items.append(
            StateItem(
                item_id=item_id,
                kind=kind,
                fact_id=_slug_kind(kind),
                path=store["path"],
                disposition=disposition,
                nominated_owner=nominated
                if isinstance(nominated, str) and nominated
                else DUCKDB_QUACK_OWNER,
                provenance=SourceFactIdentity(
                    extractor_identity=extractor_identity,
                    span=span,
                    confidence=Confidence.EXACT,
                    freshness=freshness,
                    repository_tree=repository_tree,
                ),
                rebuildable=rebuildable,
                writable=writable,
                tables=tables,
                uncertainty="" if uncertainty is None else str(uncertainty),
            )
        )
    return tuple(sorted(items, key=_item_sort_key))


def current_state_items(
    *,
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> tuple[StateItem, ...]:
    """Classify the current-tree store bindings."""

    return tuple(
        _item_from_binding(
            binding,
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
            confidence=Confidence.EXACT,
        )
        for binding in CURRENT_STATE_BINDINGS
    )


def current_state_migrations() -> tuple[StateMigrationPlan, ...]:
    """Bounded cutover for the remaining legacy provider sidecar."""

    return (
        plan_bounded_migration(
            plan_id="migrate-provider-attempt-store",
            fact_id=FACT_PROVIDER_INVOCATION,
            source_item_id="provider-attempt-store",
            target_item_id="provider-invocations",
        ),
    )


@dataclass(frozen=True)
class StateOwnershipModel:
    """Canonical classification of stores for one repository tree."""

    repository_tree: str
    freshness: str
    items: tuple[StateItem, ...]
    conflicts: tuple[StateConflict, ...] = ()
    migrations: tuple[StateMigrationPlan, ...] = ()
    schema: str = STATE_OWNERSHIP_SCHEMA
    version: int = STATE_OWNERSHIP_VERSION
    effect_class: str = EFFECT_CLASS
    can_mutate_stores: bool = MODEL_CAN_MUTATE_STORES
    can_grant_authority: bool = MODEL_CAN_GRANT_AUTHORITY
    can_create_dual_authority: bool = MODEL_CAN_CREATE_DUAL_AUTHORITY
    architecture_ir_identity: str = ""
    content_identity: str = ""

    def __post_init__(self) -> None:
        schema = _require_text(self.schema, "schema", error_type=StateOwnershipError)
        if schema != STATE_OWNERSHIP_SCHEMA:
            raise StateOwnershipError("unexpected state-ownership schema")
        version = _require_int(self.version, "version", error_type=StateOwnershipError)
        if version != STATE_OWNERSHIP_VERSION:
            raise StateOwnershipError("unexpected state-ownership version")
        repository_tree = _require_text(
            self.repository_tree, "repository_tree", error_type=StateOwnershipError
        )
        freshness = _require_text(
            self.freshness, "freshness", error_type=StateOwnershipError
        )
        effect_class = _require_text(
            self.effect_class, "effect_class", error_type=StateOwnershipError
        )
        if effect_class != EFFECT_CLASS:
            raise StateOwnershipError(
                "state-ownership effect class is read_only_analysis"
            )
        if self.can_mutate_stores is not False:
            raise StateOwnershipAuthorityError(
                "state-ownership model cannot mutate stores"
            )
        if self.can_grant_authority is not False:
            raise StateOwnershipAuthorityError(
                "state-ownership model cannot grant authority"
            )
        if self.can_create_dual_authority is not False:
            raise StateOwnershipAuthorityError(
                "state-ownership model cannot create dual authority"
            )
        architecture_ir_identity = self.architecture_ir_identity
        if architecture_ir_identity:
            architecture_ir_identity = _validate_dag_json_cid(
                _require_text(
                    architecture_ir_identity,
                    "architecture_ir_identity",
                    error_type=StateOwnershipError,
                )
            )
        else:
            architecture_ir_identity = ""
        items = tuple(
            sorted(
                _record_tuple(self.items, StateItem, "items"),
                key=_item_sort_key,
            )
        )
        item_ids = tuple(item.item_id for item in items)
        if len(item_ids) != len(set(item_ids)):
            raise StateOwnershipError("state item ids must be unique")
        for item in items:
            if item.provenance.repository_tree != repository_tree:
                raise StateOwnershipError(
                    "item provenance repository_tree must match the model"
                )
            if item.provenance.freshness != freshness:
                raise StateOwnershipError(
                    "item provenance freshness must match the model"
                )
        migrations = tuple(
            sorted(
                _record_tuple(self.migrations, StateMigrationPlan, "migrations"),
                key=_plan_sort_key,
            )
        )
        declared = tuple(
            sorted(
                _record_tuple(self.conflicts, StateConflict, "conflicts"),
                key=_conflict_sort_key,
            )
        )
        detected = detect_state_conflicts(items, migrations)
        declared_keys = {
            (item.kind, item.fact_id, item.message, item.item_ids) for item in declared
        }
        missing = [
            item
            for item in detected
            if (item.kind, item.fact_id, item.message, item.item_ids)
            not in declared_keys
        ]
        if missing:
            raise StateOwnershipError(
                "state-ownership conflicts must include detected hard blockers"
            )
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "repository_tree", repository_tree)
        object.__setattr__(self, "freshness", freshness)
        object.__setattr__(self, "effect_class", effect_class)
        object.__setattr__(self, "can_mutate_stores", False)
        object.__setattr__(self, "can_grant_authority", False)
        object.__setattr__(self, "can_create_dual_authority", False)
        object.__setattr__(self, "architecture_ir_identity", architecture_ir_identity)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "migrations", migrations)
        object.__setattr__(self, "conflicts", declared)
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
            "architecture_ir_identity": self.architecture_ir_identity,
            "can_create_dual_authority": False,
            "can_grant_authority": False,
            "can_mutate_stores": False,
            "conflicts": [item.to_dict() for item in self.conflicts],
            "effect_class": self.effect_class,
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
    def covers_required_store_kinds(self) -> bool:
        return {item.kind for item in self.items} >= set(REQUIRED_STORE_KINDS)

    @property
    def covers_required_dispositions(self) -> bool:
        present = {item.disposition for item in self.items}
        # Unknown is legal vocabulary but is a hard blocker, not a required
        # current-tree disposition once every fact is classified.
        required = set(REQUIRED_DISPOSITIONS) - {StateDisposition.UNKNOWN}
        return required <= present

    @property
    def fails_closed(self) -> bool:
        return bool(self.conflicts)

    @property
    def one_authoritative_store_holds(self) -> bool:
        blocking = {
            StateConflictKind.UNKNOWN_OWNER,
            StateConflictKind.UNKNOWN_PRODUCTION_OWNER,
            StateConflictKind.MULTIPLE_AUTHORITATIVE_STORES,
            StateConflictKind.MISSING_AUTHORITATIVE_STORE,
            StateConflictKind.CONFLICTING_DISPOSITION,
            StateConflictKind.GRAPH_OWNER_AMBIGUITY,
        }
        return not any(item.kind in blocking for item in self.conflicts)

    @property
    def unknown_ownership_count(self) -> int:
        return sum(
            1
            for item in self.items
            if item.disposition is StateDisposition.UNKNOWN
        )

    def items_for(self, kind: StoreKind | str) -> tuple[StateItem, ...]:
        parsed = _closed_enum(
            kind, StoreKind, "store kind", error_type=StateOwnershipError
        )
        return tuple(item for item in self.items if item.kind is parsed)

    def items_for_fact(self, fact_id: str) -> tuple[StateItem, ...]:
        fact = _require_text(fact_id, "fact_id", error_type=StateOwnershipError)
        return tuple(item for item in self.items if item.fact_id == fact)

    def authoritative_owner(self, fact_id: str) -> StateItem:
        group = self.items_for_fact(fact_id)
        owners = tuple(item for item in group if item.is_authoritative)
        unknown = tuple(
            item for item in group if item.disposition is StateDisposition.UNKNOWN
        )
        if unknown or len(owners) != 1:
            raise StateOwnershipError(
                f"{fact_id} does not have exactly one authoritative store"
            )
        return owners[0]

    def conflicts_for(self, fact_id: str) -> tuple[StateConflict, ...]:
        fact = _require_text(fact_id, "fact_id", error_type=StateOwnershipError)
        return tuple(item for item in self.conflicts if item.fact_id == fact)

    def mutate_store(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_store_mutation("mutate")

    def grant_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_authority_grant("grant")

    def create_dual_authority(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_dual_authority("dual-write")

    def authorize_markdown(self, *_args: Any, **_kwargs: Any) -> None:
        refuse_markdown_authority("markdown")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StateOwnershipModel":
        mapping = _require_mapping(payload, error_type=StateOwnershipError)
        _require_fields(mapping, _MODEL_FIELDS)
        try:
            model = cls(
                repository_tree=mapping["repository_tree"],
                freshness=mapping["freshness"],
                items=mapping["items"],
                conflicts=mapping["conflicts"],
                migrations=mapping["migrations"],
                schema=mapping["schema"],
                version=mapping["version"],
                effect_class=mapping["effect_class"],
                can_mutate_stores=mapping["can_mutate_stores"],
                can_grant_authority=mapping["can_grant_authority"],
                can_create_dual_authority=mapping["can_create_dual_authority"],
                architecture_ir_identity=mapping["architecture_ir_identity"],
            )
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
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
            raise StateOwnershipError("state-ownership JSON must contain an object")
        return cls.from_mapping(decoded)


def refuse_store_mutation(action: str) -> None:
    """Reject attempts to treat the model as a store writer."""

    _require_text(action, "action", error_type=StateOwnershipError)
    raise StateOwnershipAuthorityError("state-ownership model cannot mutate stores")


def refuse_authority_grant(action: str) -> None:
    """Reject attempts to treat the model as an authority granter."""

    _require_text(action, "action", error_type=StateOwnershipError)
    raise StateOwnershipAuthorityError(
        "state-ownership model cannot grant authority"
    )


def refuse_dual_authority(action: str) -> None:
    """Reject attempts to leave indefinite dual write/authority."""

    _require_text(action, "action", error_type=StateOwnershipError)
    raise StateOwnershipAuthorityError(
        "state-ownership model cannot create dual authority"
    )


def refuse_markdown_authority(action: str) -> None:
    """Reject Markdown or dashboard authority claims."""

    _require_text(action, "action", error_type=StateOwnershipError)
    raise StateOwnershipAuthorityError("Markdown cannot be an authoritative store")


def build_state_ownership_model(
    items: Sequence[StateItem | Mapping[str, Any] | StateSourceBinding] | None = None,
    *,
    migrations: Sequence[StateMigrationPlan | Mapping[str, Any]] | None = None,
    inventory: Mapping[str, Any] | None = None,
    architecture: ArchitectureIR | Mapping[str, Any] | None = None,
    extra_conflicts: Sequence[StateConflict | Mapping[str, Any]] = (),
    repository_tree: str,
    freshness: str = DEFAULT_FRESHNESS,
    extractor_identity: str = EXTRACTOR_IDENTITY,
) -> StateOwnershipModel:
    """Classify stores, record conflicts, and keep migration bounded."""

    graph: ArchitectureIR | None
    if architecture is None:
        graph = None
    elif isinstance(architecture, ArchitectureIR):
        graph = architecture
    else:
        try:
            graph = ArchitectureIR.from_mapping(architecture)
        except ArchitectureContractError as exc:
            raise _wrap_contract(exc) from exc
    if graph is not None:
        if graph.repository_tree != repository_tree:
            raise StateOwnershipError(
                "ArchitectureIR repository_tree must match the state-ownership model"
            )
        if graph.freshness != freshness:
            raise StateOwnershipError(
                "ArchitectureIR freshness must match the state-ownership model"
            )
    if inventory is not None:
        parsed_items = items_from_inventory(
            inventory,
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
        )
        parsed_migrations: tuple[StateMigrationPlan, ...] = ()
    elif items is None:
        parsed_items = current_state_items(
            repository_tree=repository_tree,
            freshness=freshness,
            extractor_identity=extractor_identity,
        )
        parsed_migrations = (
            current_state_migrations() if migrations is None else ()
        )
    else:
        parsed_items = tuple(
            classify_state_item(
                item,
                repository_tree=repository_tree,
                freshness=freshness,
                extractor_identity=extractor_identity,
            )
            for item in items
        )
        parsed_migrations = ()
    if migrations is not None:
        parsed_migrations = tuple(
            item
            if isinstance(item, StateMigrationPlan)
            else StateMigrationPlan.from_mapping(item)
            for item in migrations
        )
    extra = tuple(
        item if isinstance(item, StateConflict) else StateConflict.from_mapping(item)
        for item in extra_conflicts
    )
    graph_conflicts = classify_architecture_state(graph) if graph is not None else ()
    detected = detect_state_conflicts(parsed_items, parsed_migrations)
    merged: dict[tuple[str, str, str, tuple[str, ...]], StateConflict] = {}
    for item in (*detected, *graph_conflicts, *extra):
        merged[(item.kind.value, item.fact_id, item.message, item.item_ids)] = item
    conflicts = tuple(sorted(merged.values(), key=_conflict_sort_key))
    return StateOwnershipModel(
        repository_tree=repository_tree,
        freshness=freshness,
        items=parsed_items,
        conflicts=conflicts,
        migrations=parsed_migrations,
        architecture_ir_identity="" if graph is None else graph.content_identity,
    )


build_current_state_ownership_model = build_state_ownership_model


__all__ = [
    "CACHE_CAN_BE_AUTHORITATIVE",
    "CLOSED_CONFLICT_KINDS",
    "CLOSED_DISPOSITIONS",
    "CLOSED_MIGRATION_PHASES",
    "CLOSED_STORE_KINDS",
    "CONTENT_IDENTITY_IS_NOT_AUTHORITY",
    "CURRENT_STATE_BINDINGS",
    "DASHBOARD_CAN_BE_AUTHORITATIVE",
    "DEFAULT_FRESHNESS",
    "DUCKDB_QUACK_OWNER",
    "EFFECT_CLASS",
    "EXTRACTOR_IDENTITY",
    "FACT_ANALYSIS_CACHE",
    "FACT_COMPLETION_RECORD",
    "FACT_CONTROL_PLANE_STORE",
    "FACT_DAEMON_REGISTRY",
    "FACT_DOMAIN_EVENT",
    "FACT_GOAL_RECORD",
    "FACT_JSON_INVENTORY",
    "FACT_LEASE_RECORD",
    "FACT_PROVIDER_INVOCATION",
    "FACT_RECEIPT_RECORD",
    "FACT_TASK_RECORD",
    "FACT_WORKTREE_BINDING",
    "INDEFINITE_DUAL_AUTHORITY_PROHIBITED",
    "MARKDOWN_CAN_BE_AUTHORITATIVE",
    "MODEL_CAN_CREATE_DUAL_AUTHORITY",
    "MODEL_CAN_GRANT_AUTHORITY",
    "MODEL_CAN_MUTATE_STORES",
    "NON_AUTHORITATIVE_DISPOSITIONS",
    "PROHIBITED_AUTHORITATIVE_KINDS",
    "PROJECTION_CAN_BE_AUTHORITATIVE",
    "REBUILDABLE_DISPOSITIONS",
    "REQUIRED_DISPOSITIONS",
    "REQUIRED_MIGRATION_PHASES",
    "REQUIRED_MUTABLE_FACTS",
    "REQUIRED_STORE_KINDS",
    "STATE_CONFLICT_SCHEMA",
    "STATE_CONFLICT_VERSION",
    "STATE_ITEM_SCHEMA",
    "STATE_ITEM_VERSION",
    "STATE_MIGRATION_SCHEMA",
    "STATE_MIGRATION_VERSION",
    "STATE_OWNERSHIP_EVIDENCE",
    "STATE_OWNERSHIP_SCHEMA",
    "STATE_OWNERSHIP_VERSION",
    "StoreKind",
    "StateConflict",
    "StateConflictKind",
    "StateDisposition",
    "StateItem",
    "StateMigrationPhase",
    "StateMigrationPlan",
    "StateOwnershipAuthorityError",
    "StateOwnershipError",
    "StateOwnershipModel",
    "StateSourceBinding",
    "TASK_ID",
    "UNKNOWN_OWNER_ACCEPTED",
    "build_current_state_ownership_model",
    "build_state_ownership_model",
    "classify_architecture_state",
    "classify_state_item",
    "current_state_items",
    "current_state_migrations",
    "detect_state_conflicts",
    "items_from_inventory",
    "plan_bounded_migration",
    "refuse_authority_grant",
    "refuse_dual_authority",
    "refuse_markdown_authority",
    "refuse_store_mutation",
]
