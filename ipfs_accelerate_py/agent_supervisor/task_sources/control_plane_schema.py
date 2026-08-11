"""Normalized control-plane schema inventory and install helpers (DQP-005).

Interface: ``ControlPlaneSchema@1``

Owns the physical table/view inventory for ``control.duckdb`` domain SQL
(``sql/0001_control_plane.sql``), join-critical identity columns, existing
task-CID / lease compatibility contracts, and the optional supervisor DuckDB
dependency pin. Schema DDL is applied only through the checksum-bound migration
catalog; this module never invents runtime ad-hoc tables.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_migrations import (
    ControlPlaneMigrationRunner,
    MigrationCatalog,
    MigrationRunReport,
    compute_schema_fingerprint,
    duckdb_available,
    load_default_catalog,
)
from .duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Interface / version identities
# ---------------------------------------------------------------------------

CONTROL_PLANE_SCHEMA_INTERFACE: Final = "ControlPlaneSchema@1"
CONTROL_PLANE_SCHEMA_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-schema@1"
)
CONTROL_PLANE_SCHEMA_VERSION: Final[int] = 1
CONTROL_PLANE_SCHEMA_REVISION: Final[int] = 1
CONTROL_PLANE_MIGRATION_ID: Final = "0001_control_plane"
CONTROL_PLANE_MIGRATION_VERSION: Final[int] = 1
CONTROL_PLANE_SQL_FILENAME: Final = "0001_control_plane.sql"

# Optional supervisor service dependency profile (pyproject extra).
SUPERVISOR_OPTIONAL_EXTRA: Final = "agent-supervisor"
PINNED_DUCKDB_VERSION_SPEC: Final = "duckdb>=1.5.0,<1.6.0"
PINNED_DUCKDB_MAJOR: Final = 1
PINNED_DUCKDB_MINOR: Final = 5
PINNED_DUCKDB_VERSION_PREFIX: Final = "1.5"
PINNED_QUACK_EXTENSION: Final = "quack"
PINNED_QUACK_EXTENSION_API: Final = "quack@1"
PINNED_PROFILE_ID: Final = "agent-supervisor-duckdb-quack-1.5"

SCHEMA_DOMAINS: Final[tuple[str, ...]] = (
    "meta",
    "intent",
    "schedule",
    "runtime",
    "git",
    "code",
    "evidence",
    "cache",
    "control",
    "improve",
)

# Bookkeeping tables installed by the migration runner (not domain SQL).
BOOKKEEPING_TABLES: Final[tuple[str, ...]] = (
    "control_plane_metadata",
    "schema_migrations",
    "schema_migration_attempts",
)

# Domain table inventory (main schema). Order is documentation-only.
_DOMAIN_TABLES: Final[dict[str, tuple[str, ...]]] = {
    "meta": (
        "schema_contracts",
        "store_generations",
    ),
    "intent": (
        "objectives",
        "objective_revisions",
        "goals",
        "goal_edges",
        "plans",
        "plan_revisions",
        "planning_decisions",
        "plan_candidates",
        "tasks",
        "task_revisions",
        "task_dependencies",
        "task_outputs",
        "task_acceptance",
        "task_validations",
        "artifacts",
    ),
    "schedule": (
        "task_assignments",
        "task_blocks",
        "refill_epochs",
        "findings",
        "finding_dispositions",
    ),
    "runtime": (
        "supervisor_instances",
        "daemon_instances",
        "daemon_sessions",
        "heartbeats",
        "health_samples",
        "stall_detections",
        "restart_decisions",
        "task_attempts",
        "attempt_phases",
        "task_claims",
        "provider_invocations",
        "validation_runs",
        "validation_results",
        "merge_attempts",
        "recovery_actions",
        "idempotency_records",
        "effect_claims",
        "completion_receipts",
    ),
    "git": (
        "repositories",
        "repository_revisions",
        "submodule_edges",
        "worktrees",
        "worktree_snapshots",
        "worktree_paths",
        "dirty_overlays",
        "branches",
        "git_refs",
        "merge_bases",
        "merge_queue_entries",
        "resource_claims",
        "path_claims",
        "leases",
        "lease_events",
        "token_history",
    ),
    "code": (
        "source_snapshots",
        "source_files",
        "file_versions",
        "parse_runs",
        "symbols",
        "symbol_versions",
        "ast_nodes",
        "ast_edges",
        "imports",
        "calls",
        "references",
        "definitions",
        "type_relations",
        "mutations",
        "mutation_files",
        "mutation_hunks",
        "ast_mutations",
        "impact_edges",
        "impact_closures",
        "repair_candidates",
        "repair_applications",
    ),
    "evidence": (
        "proof_obligations",
        "proof_attempts",
        "counterexamples",
        "evidence_nodes",
    ),
    "cache": (
        "context_manifests",
        "context_members",
        "context_deltas",
        "prompt_templates",
        "prompt_instances",
        "prompt_inputs",
        "provider_calls",
        "provider_responses",
        "failure_signatures",
        "decision_cache_entries",
        "replay_suppressions",
    ),
    "control": (
        "state_servers",
        "server_epochs",
        "client_sessions",
        "capability_snapshots",
        "credentials",
        "authorization_roles",
        "authorization_grants",
        "backup_snapshots",
        "restore_receipts",
        "maintenance_leases",
        "domain_events",
        "structured_logs",
    ),
    "improve": (
        "metrics",
        "metric_samples",
        "budget_reservations",
        "budget_consumption",
        "quack_query_telemetry",
        "churn_metrics",
    ),
}

DOMAIN_TABLES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {key: tuple(value) for key, value in _DOMAIN_TABLES.items()}
)

DIAGNOSTIC_VIEWS: Final[tuple[str, ...]] = (
    "ready_task_context_v1",
    "diagnostic_schema_inventory_v1",
    "diagnostic_lease_surface_v1",
)

# Existing DuckDB task-source / lease-coordinator identity contracts.
TASK_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "task_alias",
    "goal_cid",
    "ordinal",
    "status",
    "revision",
)

LEASE_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "claim_cid",
    "resolution_cid",
    "claimant_did",
    "logical_epoch",
    "fencing_token",
    "expires_at_ms",
    "attempt",
    "state",
    "started_at_ms",
    "release_reason",
    "retry_not_before_ms",
)

# Columns that must exist as real columns for joins/claims/auth — never JSON-only.
JOIN_CRITICAL_IDENTITIES: Final[tuple[tuple[str, str], ...]] = (
    ("tasks", "task_cid"),
    ("tasks", "goal_cid"),
    ("task_dependencies", "task_cid"),
    ("task_dependencies", "dependency_task_cid"),
    ("leases", "task_cid"),
    ("leases", "claim_cid"),
    ("leases", "claimant_did"),
    ("leases", "fencing_token"),
    ("lease_events", "task_cid"),
    ("token_history", "task_cid"),
    ("heartbeats", "task_cid"),
    ("task_claims", "task_cid"),
    ("task_claims", "claim_id"),
    ("task_attempts", "task_cid"),
    ("task_attempts", "attempt_id"),
    ("completion_receipts", "task_cid"),
    ("completion_receipts", "receipt_cid"),
    ("domain_events", "event_id"),
    ("domain_events", "stream_id"),
    ("domain_events", "sequence"),
    ("path_claims", "task_cid"),
    ("path_claims", "path"),
    ("resource_claims", "resource_id"),
    ("worktrees", "worktree_id"),
    ("worktrees", "repository_id"),
    ("merge_queue_entries", "task_cid"),
    ("repositories", "repository_id"),
    ("goals", "goal_cid"),
    ("objectives", "objective_id"),
    ("plans", "plan_cid"),
    ("mutations", "mutation_id"),
    ("mutations", "task_cid"),
    ("proof_obligations", "obligation_id"),
    ("evidence_nodes", "evidence_id"),
    ("context_manifests", "manifest_cid"),
    ("context_manifests", "task_cid"),
    ("idempotency_records", "idempotency_key"),
    ("state_servers", "server_id"),
    ("client_sessions", "session_id"),
    ("maintenance_leases", "lease_id"),
    ("artifacts", "cid"),
)

_OPAQUE_JSON_COLUMN_RE: Final = re.compile(
    r"(?:_json|payload_json|body_json|identity_json|extension_json|"
    r"effect_json|policy_json|argv_json|provenance_json)$",
    re.IGNORECASE,
)


class ControlPlaneSchemaError(RuntimeError):
    """Base class for fail-closed control-plane schema errors."""


class ControlPlaneSchemaInstallError(ControlPlaneSchemaError):
    """Schema installation or fingerprint verification failed."""


class ControlPlaneSchemaCompatibilityError(ControlPlaneSchemaError):
    """Existing task CID / lease semantics are not preserved."""


class ControlPlaneSchemaIdentityError(ControlPlaneSchemaError):
    """A join-critical identity is missing or only present in opaque JSON."""


@dataclass(frozen=True)
class DuckDBQuackDependencyProfile:
    """Pinned install profile for the optional supervisor service."""

    extra_name: str = SUPERVISOR_OPTIONAL_EXTRA
    duckdb_spec: str = PINNED_DUCKDB_VERSION_SPEC
    duckdb_major: int = PINNED_DUCKDB_MAJOR
    duckdb_minor: int = PINNED_DUCKDB_MINOR
    duckdb_version_prefix: str = PINNED_DUCKDB_VERSION_PREFIX
    extension_name: str = PINNED_QUACK_EXTENSION
    extension_api: str = PINNED_QUACK_EXTENSION_API
    profile_id: str = PINNED_PROFILE_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_name": self.extra_name,
            "duckdb_spec": self.duckdb_spec,
            "duckdb_major": int(self.duckdb_major),
            "duckdb_minor": int(self.duckdb_minor),
            "duckdb_version_prefix": self.duckdb_version_prefix,
            "extension_name": self.extension_name,
            "extension_api": self.extension_api,
            "profile_id": self.profile_id,
        }


@dataclass(frozen=True)
class ControlPlaneSchema:
    """Canonical inventory for the normalized control-plane physical schema.

    Interface: ``ControlPlaneSchema@1``.
    """

    INTERFACE: ClassVar[str] = CONTROL_PLANE_SCHEMA_INTERFACE
    SCHEMA: ClassVar[str] = CONTROL_PLANE_SCHEMA_SCHEMA

    schema_revision: int = CONTROL_PLANE_SCHEMA_REVISION
    migration_id: str = CONTROL_PLANE_MIGRATION_ID
    migration_version: int = CONTROL_PLANE_MIGRATION_VERSION
    domains: tuple[str, ...] = SCHEMA_DOMAINS
    domain_tables: Mapping[str, tuple[str, ...]] = DOMAIN_TABLES
    diagnostic_views: tuple[str, ...] = DIAGNOSTIC_VIEWS
    bookkeeping_tables: tuple[str, ...] = BOOKKEEPING_TABLES
    task_identity_columns: tuple[str, ...] = TASK_IDENTITY_COLUMNS
    lease_identity_columns: tuple[str, ...] = LEASE_IDENTITY_COLUMNS
    join_critical_identities: tuple[tuple[str, str], ...] = JOIN_CRITICAL_IDENTITIES
    dependency_profile: DuckDBQuackDependencyProfile = DuckDBQuackDependencyProfile()

    def __post_init__(self) -> None:
        if int(self.schema_revision) < 1:
            raise ControlPlaneSchemaError("schema_revision must be >= 1")
        if tuple(self.domains) != SCHEMA_DOMAINS:
            raise ControlPlaneSchemaError(
                "domains must match the closed SCHEMA_DOMAINS vocabulary"
            )
        missing = [name for name in SCHEMA_DOMAINS if name not in self.domain_tables]
        if missing:
            raise ControlPlaneSchemaError(
                f"domain_tables missing domains: {missing}"
            )

    @property
    def all_domain_tables(self) -> tuple[str, ...]:
        tables: list[str] = []
        seen: set[str] = set()
        for domain in self.domains:
            for table in self.domain_tables[domain]:
                if table not in seen:
                    tables.append(table)
                    seen.add(table)
        return tuple(tables)

    def sql_path(self) -> Path:
        return Path(__file__).resolve().parent / "sql" / CONTROL_PLANE_SQL_FILENAME

    def sql_text(self) -> str:
        path = self.sql_path()
        if not path.is_file():
            raise ControlPlaneSchemaInstallError(
                f"control-plane schema SQL is missing: {path}"
            )
        return path.read_text(encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "schema_revision": int(self.schema_revision),
            "migration_id": self.migration_id,
            "migration_version": int(self.migration_version),
            "domains": list(self.domains),
            "domain_tables": {
                domain: list(tables) for domain, tables in self.domain_tables.items()
            },
            "diagnostic_views": list(self.diagnostic_views),
            "bookkeeping_tables": list(self.bookkeeping_tables),
            "task_identity_columns": list(self.task_identity_columns),
            "lease_identity_columns": list(self.lease_identity_columns),
            "join_critical_identities": [
                {"table": table, "column": column}
                for table, column in self.join_critical_identities
            ],
            "dependency_profile": self.dependency_profile.to_dict(),
            "sql_filename": CONTROL_PLANE_SQL_FILENAME,
        }


def default_control_plane_schema() -> ControlPlaneSchema:
    """Return the package ControlPlaneSchema@1 inventory."""

    return ControlPlaneSchema()


def default_dependency_profile() -> DuckDBQuackDependencyProfile:
    """Return the pinned DuckDB/Quack optional-service profile."""

    return DuckDBQuackDependencyProfile()


def package_sql_directory() -> Path:
    return Path(__file__).resolve().parent / "sql"


def load_control_plane_catalog(
    sql_directory: Path | str | None = None,
) -> MigrationCatalog:
    """Load the default migration catalog (includes 0001_control_plane)."""

    return load_default_catalog(sql_directory or package_sql_directory())


def install_control_plane_schema(
    database_path: Path | str,
    *,
    catalog: MigrationCatalog | None = None,
    application_version: str | None = None,
    tool_version: str | None = None,
    owner_id: str | None = None,
) -> MigrationRunReport:
    """Apply the control-plane catalog to ``database_path`` (idempotent)."""

    if not duckdb_available():
        raise ControlPlaneSchemaInstallError(
            "DuckDB is required to install the control-plane schema"
        )
    resolved_catalog = catalog or load_control_plane_catalog()
    if resolved_catalog.latest_version < CONTROL_PLANE_MIGRATION_VERSION:
        raise ControlPlaneSchemaInstallError(
            "control-plane catalog is missing migration "
            f"{CONTROL_PLANE_MIGRATION_ID}"
        )
    migration = resolved_catalog.get(CONTROL_PLANE_MIGRATION_VERSION)
    if migration.migration_id != CONTROL_PLANE_MIGRATION_ID:
        raise ControlPlaneSchemaInstallError(
            f"expected migration_id {CONTROL_PLANE_MIGRATION_ID}, "
            f"got {migration.migration_id}"
        )
    runner = ControlPlaneMigrationRunner.for_database(
        database_path,
        catalog=resolved_catalog,
        application_version=application_version,
        tool_version=tool_version,
        owner_id=owner_id,
    )
    return runner.apply()


def prove_fresh_and_upgraded_equivalence(
    left_database_path: Path | str,
    right_database_path: Path | str,
    *,
    catalog: MigrationCatalog | None = None,
    application_version: str = "0.0.45",
    tool_version: str = "1.5.2",
) -> dict[str, Any]:
    """Prove empty-to-latest fingerprints match on two independent databases."""

    resolved = catalog or load_control_plane_catalog()
    left_runner = ControlPlaneMigrationRunner.for_database(
        left_database_path,
        catalog=resolved,
        application_version=application_version,
        tool_version=tool_version,
        owner_id="schema-proof-left",
    )
    return left_runner.prove_empty_to_latest_equivalence(
        other_database_path=right_database_path,
    )


def _table_columns(connection: Any, table_name: str) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table_name],
    ).fetchall()
    out: dict[str, str] = {}
    for row in rows:
        if isinstance(row, Mapping):
            out[str(row["column_name"])] = str(row["data_type"])
        else:
            out[str(row[0])] = str(row[1])
    return out


def _relation_exists(
    connection: Any,
    name: str,
    *,
    table_type: str | None = None,
) -> bool:
    sql = """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
    """
    params: list[Any] = [name]
    if table_type is not None:
        sql += " AND table_type = ?"
        params.append(table_type)
    sql += " LIMIT 1"
    return connection.execute(sql, params).fetchone() is not None


def verify_installed_schema(
    database_path: Path | str,
    *,
    schema: ControlPlaneSchema | None = None,
) -> dict[str, Any]:
    """Verify domain tables, views, join-critical columns, and lease/task shapes."""

    inventory = schema or default_control_plane_schema()
    if not duckdb_available():
        raise ControlPlaneSchemaInstallError(
            "DuckDB is required to verify the control-plane schema"
        )
    report: dict[str, Any] = {
        "database_path": str(database_path),
        "schema_revision": inventory.schema_revision,
        "tables_ok": [],
        "views_ok": [],
        "join_critical_ok": [],
        "task_columns_ok": [],
        "lease_columns_ok": [],
        "opaque_json_only_identities": [],
    }
    with open_duckdb_connection(database_path) as connection:
        for table in inventory.bookkeeping_tables:
            if not _relation_exists(connection, table):
                raise ControlPlaneSchemaInstallError(
                    f"bookkeeping table missing: {table}"
                )
            report["tables_ok"].append(table)
        for table in inventory.all_domain_tables:
            if not _relation_exists(connection, table):
                raise ControlPlaneSchemaInstallError(
                    f"domain table missing: {table}"
                )
            report["tables_ok"].append(table)
        for view in inventory.diagnostic_views:
            if not _relation_exists(connection, view):
                raise ControlPlaneSchemaInstallError(
                    f"diagnostic view missing: {view}"
                )
            report["views_ok"].append(view)

        for table, column in inventory.join_critical_identities:
            columns = _table_columns(connection, table)
            if column not in columns:
                raise ControlPlaneSchemaIdentityError(
                    f"join-critical column {table}.{column} is missing"
                )
            if _OPAQUE_JSON_COLUMN_RE.search(column):
                raise ControlPlaneSchemaIdentityError(
                    f"join-critical identity {table}.{column} must not be "
                    "an opaque JSON column"
                )
            report["join_critical_ok"].append(f"{table}.{column}")

        # Ensure no join-critical identity is represented only by a JSON twin.
        for table, column in inventory.join_critical_identities:
            columns = _table_columns(connection, table)
            json_twin = f"{column}_json"
            if json_twin in columns and column not in columns:
                report["opaque_json_only_identities"].append(f"{table}.{column}")
                raise ControlPlaneSchemaIdentityError(
                    f"join-critical identity {table}.{column} exists only as "
                    f"opaque JSON column {json_twin}"
                )

        task_columns = _table_columns(connection, "tasks")
        for column in inventory.task_identity_columns:
            if column not in task_columns:
                raise ControlPlaneSchemaCompatibilityError(
                    f"tasks.{column} required for task CID semantics is missing"
                )
            report["task_columns_ok"].append(column)

        lease_columns = _table_columns(connection, "leases")
        for column in inventory.lease_identity_columns:
            if column not in lease_columns:
                raise ControlPlaneSchemaCompatibilityError(
                    f"leases.{column} required for lease semantics is missing"
                )
            report["lease_columns_ok"].append(column)

        # task_cid must be the lease primary key surface (unique identity).
        if "task_cid" not in lease_columns:
            raise ControlPlaneSchemaCompatibilityError(
                "leases.task_cid is required"
            )
        report["schema_fingerprint"] = compute_schema_fingerprint(connection)
    return report


def assert_dependency_profile_pinned(
    pyproject_text: str,
    *,
    profile: DuckDBQuackDependencyProfile | None = None,
) -> None:
    """Fail closed unless ``pyproject.toml`` pins the supervisor DuckDB extra."""

    expected = profile or default_dependency_profile()
    text = str(pyproject_text)
    if f"{expected.extra_name}" not in text:
        raise ControlPlaneSchemaInstallError(
            f"pyproject.toml must declare optional extra {expected.extra_name!r}"
        )
    # Accept either the exact spec or an equivalent 1.5.x pin.
    has_spec = expected.duckdb_spec in text
    has_prefix_pin = (
        "duckdb" in text
        and "1.5" in text
        and ("<1.6" in text or "~=1.5" in text or "==1.5" in text)
    )
    if not (has_spec or has_prefix_pin):
        raise ControlPlaneSchemaInstallError(
            "pyproject.toml must pin DuckDB 1.5.x for the optional supervisor "
            f"service (expected {expected.duckdb_spec!r})"
        )
    if "agent-supervisor" not in text and expected.extra_name not in text:
        raise ControlPlaneSchemaInstallError(
            "pyproject.toml must name the optional supervisor service extra"
        )


def read_pyproject_text(path: Path | str | None = None) -> str:
    if path is None:
        path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    return Path(path).read_text(encoding="utf-8")


__all__ = [
    "BOOKKEEPING_TABLES",
    "CONTROL_PLANE_MIGRATION_ID",
    "CONTROL_PLANE_MIGRATION_VERSION",
    "CONTROL_PLANE_SCHEMA_INTERFACE",
    "CONTROL_PLANE_SCHEMA_REVISION",
    "CONTROL_PLANE_SCHEMA_SCHEMA",
    "CONTROL_PLANE_SCHEMA_VERSION",
    "CONTROL_PLANE_SQL_FILENAME",
    "ControlPlaneSchema",
    "ControlPlaneSchemaCompatibilityError",
    "ControlPlaneSchemaError",
    "ControlPlaneSchemaIdentityError",
    "ControlPlaneSchemaInstallError",
    "DIAGNOSTIC_VIEWS",
    "DOMAIN_TABLES",
    "DuckDBQuackDependencyProfile",
    "JOIN_CRITICAL_IDENTITIES",
    "LEASE_IDENTITY_COLUMNS",
    "PINNED_DUCKDB_MAJOR",
    "PINNED_DUCKDB_MINOR",
    "PINNED_DUCKDB_VERSION_PREFIX",
    "PINNED_DUCKDB_VERSION_SPEC",
    "PINNED_PROFILE_ID",
    "PINNED_QUACK_EXTENSION",
    "PINNED_QUACK_EXTENSION_API",
    "SCHEMA_DOMAINS",
    "SUPERVISOR_OPTIONAL_EXTRA",
    "TASK_IDENTITY_COLUMNS",
    "assert_dependency_profile_pinned",
    "default_control_plane_schema",
    "default_dependency_profile",
    "install_control_plane_schema",
    "load_control_plane_catalog",
    "package_sql_directory",
    "prove_fresh_and_upgraded_equivalence",
    "read_pyproject_text",
    "verify_installed_schema",
]
