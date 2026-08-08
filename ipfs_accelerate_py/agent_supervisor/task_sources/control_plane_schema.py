"""Normalized control-plane schema catalog and install surface (DQP-005).

Interface: ``ControlPlaneSchema@1``.

This module is the domain-schema companion to the checksum-bound migration
runner. It enumerates the admitted tables and views installed by
``sql/0001_control_plane.sql``, records the pinned DuckDB/Quack dependency
profile for the optional supervisor service, and applies the package catalog
to a physical ``control.duckdb`` while proving empty-to-latest fingerprint
equivalence.

Join-critical identities (task CIDs, lease fencing tokens, revisions, epochs)
are first-class typed columns. Opaque JSON is allowed only as a bounded
extension payload, never as the sole home of an identity used in joins,
claims, retention, or authorization.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_migrations import (
    ControlPlaneMigrationRunner,
    MigrationCatalog,
    MigrationRunReport,
    compute_schema_fingerprint,
    load_default_catalog,
)
from .duckdb_state import open_duckdb_connection
from .quack_capabilities import (
    PINNED_DUCKDB_MAJOR,
    PINNED_DUCKDB_MINOR,
    PINNED_DUCKDB_VERSION_PREFIX,
    PINNED_EXTENSION_API,
    PINNED_EXTENSION_NAME,
    default_compatibility_profile,
)


CONTROL_PLANE_SCHEMA_VERSION: Final[int] = 1
CONTROL_PLANE_SCHEMA_INTERFACE: Final[str] = "ControlPlaneSchema@1"
CONTROL_PLANE_SCHEMA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-schema@1"
)
CONTROL_PLANE_DEPENDENCY_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-dependency-profile@1"
)

BASE_MIGRATION_VERSION: Final[int] = 1
BASE_MIGRATION_ID: Final[str] = "0001_control_plane"
BASE_MIGRATION_FILENAME: Final[str] = f"{BASE_MIGRATION_ID}.sql"

# Optional-extra pin for the supervisor service (mirrors pyproject.toml).
PINNED_DUCKDB_REQUIREMENT: Final[str] = "duckdb>=1.5.0,<1.6.0"
PINNED_OPTIONAL_EXTRA: Final[str] = "agent-supervisor"
PINNED_PROFILE_ID: Final[str] = "agent-supervisor-duckdb-quack-1.5"

# Bookkeeping tables installed by the migration runner (not domain SQL).
BOOKKEEPING_TABLES: Final[tuple[str, ...]] = (
    "control_plane_metadata",
    "schema_migrations",
    "schema_migration_attempts",
)

# Closed domain vocabulary from the control-plane plan / DQP-005 effects.
class SchemaDomain(str, Enum):
    META = "meta"
    INTENT = "intent"
    SCHEDULE = "schedule"
    RUNTIME = "runtime"
    GIT = "git"
    CODE = "code"
    EVIDENCE = "evidence"
    CACHE = "cache"
    CONTROL = "control"
    IMPROVE = "improve"
    DIAGNOSTIC = "diagnostic"
    CONTEXT = "context"


DOMAIN_TABLES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        SchemaDomain.META.value: (
            "schema_contracts",
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
        ),
        SchemaDomain.GIT.value: (
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
        ),
        SchemaDomain.INTENT.value: (
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
            "task_assignments",
            "task_blocks",
            "refill_epochs",
            "findings",
            "finding_dispositions",
        ),
        SchemaDomain.SCHEDULE.value: (
            "schedule_policies",
            "schedule_lanes",
            "schedule_queue_entries",
        ),
        SchemaDomain.RUNTIME.value: (
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
        SchemaDomain.EVIDENCE.value: (
            "domain_events",
            "structured_logs",
            "metrics",
            "metric_samples",
            "budget_reservations",
            "budget_consumption",
            "quack_query_telemetry",
            "evidence_nodes",
            "artifacts",
        ),
        SchemaDomain.CODE.value: (
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
            "symbol_references",
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
            "proof_obligations",
            "proof_attempts",
            "counterexamples",
        ),
        SchemaDomain.CACHE.value: (
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
            "churn_metrics",
        ),
        SchemaDomain.CONTROL.value: (
            "control_surfaces",
            "control_operations",
            "control_authorization_decisions",
        ),
        SchemaDomain.IMPROVE.value: (
            "improve_experiments",
            "improve_rollouts",
            "improve_decisions",
        ),
    }
)

DIAGNOSTIC_VIEWS: Final[tuple[str, ...]] = (
    "ready_task_context_v1",
    "active_lease_v1",
    "open_task_dependency_v1",
    "schema_domain_catalog_v1",
)

# Columns that must remain first-class (never JSON-only) for lease/task identity.
TASK_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "task_alias",
    "goal_cid",
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
    "retry_not_before_ms",
)

_REQUIREMENT_RE: Final = re.compile(
    r"^duckdb\s*>=\s*1\.5(?:\.0)?\s*,\s*<\s*1\.6(?:\.0)?\s*$",
    re.IGNORECASE,
)


class ControlPlaneSchemaError(RuntimeError):
    """Fail-closed schema catalog or install error."""


class ControlPlaneSchemaDriftError(ControlPlaneSchemaError):
    """Installed schema does not match the admitted catalog."""


def package_sql_directory() -> Path:
    """Return the package directory that owns admitted migration SQL."""

    return Path(__file__).resolve().parent / "sql"


def base_migration_sql_path() -> Path:
    return package_sql_directory() / BASE_MIGRATION_FILENAME


def load_base_migration_sql() -> str:
    path = base_migration_sql_path()
    if not path.is_file():
        raise ControlPlaneSchemaError(
            f"base control-plane migration is missing: {path}"
        )
    return path.read_text(encoding="utf-8")


def all_domain_tables() -> tuple[str, ...]:
    tables: list[str] = []
    for domain in (
        SchemaDomain.META,
        SchemaDomain.GIT,
        SchemaDomain.INTENT,
        SchemaDomain.SCHEDULE,
        SchemaDomain.RUNTIME,
        SchemaDomain.EVIDENCE,
        SchemaDomain.CODE,
        SchemaDomain.CACHE,
        SchemaDomain.CONTROL,
        SchemaDomain.IMPROVE,
    ):
        tables.extend(DOMAIN_TABLES[domain.value])
    return tuple(tables)


def required_tables() -> tuple[str, ...]:
    """Tables expected after bookkeeping + base domain migration."""

    return BOOKKEEPING_TABLES + all_domain_tables()


def required_views() -> tuple[str, ...]:
    return DIAGNOSTIC_VIEWS


@dataclass(frozen=True)
class ControlPlaneDependencyProfile:
    """Pinned optional-service dependency profile for DuckDB/Quack.

    Interface companion to ``QuackCompatibilityProfile@1``; package pins live
    here and in ``pyproject.toml`` under the ``agent-supervisor`` extra.
    """

    SCHEMA: ClassVar[str] = CONTROL_PLANE_DEPENDENCY_PROFILE_SCHEMA

    profile_id: str = PINNED_PROFILE_ID
    optional_extra: str = PINNED_OPTIONAL_EXTRA
    duckdb_requirement: str = PINNED_DUCKDB_REQUIREMENT
    duckdb_major: int = PINNED_DUCKDB_MAJOR
    duckdb_minor: int = PINNED_DUCKDB_MINOR
    duckdb_version_prefix: str = PINNED_DUCKDB_VERSION_PREFIX
    extension_name: str = PINNED_EXTENSION_NAME
    extension_api: str = PINNED_EXTENSION_API

    def __post_init__(self) -> None:
        if not str(self.profile_id).strip():
            raise ControlPlaneSchemaError("profile_id is required")
        if not str(self.optional_extra).strip():
            raise ControlPlaneSchemaError("optional_extra is required")
        requirement = str(self.duckdb_requirement).strip()
        if not _REQUIREMENT_RE.fullmatch(requirement):
            raise ControlPlaneSchemaError(
                "duckdb_requirement must pin the 1.5.x supervisor window "
                f"(got {requirement!r})"
            )
        object.__setattr__(self, "duckdb_requirement", requirement)
        if int(self.duckdb_major) != PINNED_DUCKDB_MAJOR:
            raise ControlPlaneSchemaError("duckdb_major must match pin")
        if int(self.duckdb_minor) != PINNED_DUCKDB_MINOR:
            raise ControlPlaneSchemaError("duckdb_minor must match pin")
        if str(self.duckdb_version_prefix) != PINNED_DUCKDB_VERSION_PREFIX:
            raise ControlPlaneSchemaError(
                "duckdb_version_prefix must match pin"
            )
        if str(self.extension_name) != PINNED_EXTENSION_NAME:
            raise ControlPlaneSchemaError("extension_name must match pin")
        if str(self.extension_api) != PINNED_EXTENSION_API:
            raise ControlPlaneSchemaError("extension_api must match pin")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "profile_id": self.profile_id,
            "optional_extra": self.optional_extra,
            "duckdb_requirement": self.duckdb_requirement,
            "duckdb_major": int(self.duckdb_major),
            "duckdb_minor": int(self.duckdb_minor),
            "duckdb_version_prefix": self.duckdb_version_prefix,
            "extension_name": self.extension_name,
            "extension_api": self.extension_api,
        }

    @classmethod
    def default(cls) -> "ControlPlaneDependencyProfile":
        return cls()


@dataclass(frozen=True)
class ControlPlaneSchema:
    """Admitted control-plane schema catalog (``ControlPlaneSchema@1``)."""

    SCHEMA: ClassVar[str] = CONTROL_PLANE_SCHEMA_SCHEMA
    INTERFACE: ClassVar[str] = CONTROL_PLANE_SCHEMA_INTERFACE

    version: int = CONTROL_PLANE_SCHEMA_VERSION
    migration_id: str = BASE_MIGRATION_ID
    migration_version: int = BASE_MIGRATION_VERSION
    domains: tuple[str, ...] = tuple(
        domain.value
        for domain in (
            SchemaDomain.META,
            SchemaDomain.INTENT,
            SchemaDomain.SCHEDULE,
            SchemaDomain.RUNTIME,
            SchemaDomain.GIT,
            SchemaDomain.CODE,
            SchemaDomain.EVIDENCE,
            SchemaDomain.CACHE,
            SchemaDomain.CONTROL,
            SchemaDomain.IMPROVE,
        )
    )
    tables: tuple[str, ...] = all_domain_tables()
    views: tuple[str, ...] = DIAGNOSTIC_VIEWS
    bookkeeping_tables: tuple[str, ...] = BOOKKEEPING_TABLES
    dependency_profile: ControlPlaneDependencyProfile = (
        ControlPlaneDependencyProfile()
    )
    sql_path: str = str(base_migration_sql_path())

    def __post_init__(self) -> None:
        if int(self.version) != CONTROL_PLANE_SCHEMA_VERSION:
            raise ControlPlaneSchemaError(
                f"unsupported ControlPlaneSchema version {self.version}"
            )
        if str(self.migration_id) != BASE_MIGRATION_ID:
            raise ControlPlaneSchemaError(
                f"base migration_id must be {BASE_MIGRATION_ID}"
            )
        if int(self.migration_version) != BASE_MIGRATION_VERSION:
            raise ControlPlaneSchemaError(
                f"base migration_version must be {BASE_MIGRATION_VERSION}"
            )
        expected_tables = all_domain_tables()
        if tuple(self.tables) != expected_tables:
            raise ControlPlaneSchemaError(
                "tables catalog drifted from admitted DOMAIN_TABLES"
            )
        if tuple(self.views) != DIAGNOSTIC_VIEWS:
            raise ControlPlaneSchemaError(
                "views catalog drifted from admitted DIAGNOSTIC_VIEWS"
            )
        missing_domains = [
            name
            for name in (
                SchemaDomain.META.value,
                SchemaDomain.INTENT.value,
                SchemaDomain.SCHEDULE.value,
                SchemaDomain.RUNTIME.value,
                SchemaDomain.GIT.value,
                SchemaDomain.CODE.value,
                SchemaDomain.EVIDENCE.value,
                SchemaDomain.CACHE.value,
                SchemaDomain.CONTROL.value,
                SchemaDomain.IMPROVE.value,
            )
            if name not in self.domains
        ]
        if missing_domains:
            raise ControlPlaneSchemaError(
                f"schema domains missing: {missing_domains}"
            )
        if not isinstance(self.dependency_profile, ControlPlaneDependencyProfile):
            raise ControlPlaneSchemaError(
                "dependency_profile must be ControlPlaneDependencyProfile"
            )
        # Fail closed if the SQL file is absent from the package tree.
        if not Path(self.sql_path).is_file():
            raise ControlPlaneSchemaError(
                f"schema SQL is not present at {self.sql_path}"
            )

    @classmethod
    def default(cls) -> "ControlPlaneSchema":
        return cls()

    def domain_tables(self, domain: str | SchemaDomain) -> tuple[str, ...]:
        key = domain.value if isinstance(domain, SchemaDomain) else str(domain)
        try:
            return DOMAIN_TABLES[key]
        except KeyError as exc:
            raise ControlPlaneSchemaError(
                f"unknown schema domain: {key}"
            ) from exc

    def catalog(self) -> MigrationCatalog:
        catalog = load_default_catalog()
        if catalog.latest_version < self.migration_version:
            raise ControlPlaneSchemaError(
                "package migration catalog is missing the base control-plane "
                f"migration (latest={catalog.latest_version})"
            )
        migration = catalog.get(self.migration_version)
        if migration.migration_id != self.migration_id:
            raise ControlPlaneSchemaError(
                f"expected migration_id {self.migration_id}, "
                f"got {migration.migration_id}"
            )
        return catalog

    def runner_for(
        self,
        database_path: Path | str,
        *,
        application_version: str | None = None,
        tool_version: str | None = None,
        owner_id: str | None = None,
    ) -> ControlPlaneMigrationRunner:
        return ControlPlaneMigrationRunner.for_database(
            database_path,
            catalog=self.catalog(),
            application_version=application_version,
            tool_version=tool_version,
            owner_id=owner_id,
        )

    def install(
        self,
        database_path: Path | str,
        *,
        application_version: str | None = None,
        tool_version: str | None = None,
        owner_id: str | None = None,
    ) -> MigrationRunReport:
        """Apply the package catalog through the migration runner."""

        runner = self.runner_for(
            database_path,
            application_version=application_version,
            tool_version=tool_version,
            owner_id=owner_id,
        )
        report = runner.apply()
        self.verify_installed(database_path)
        return report

    def prove_empty_to_latest_equivalence(
        self,
        left_database_path: Path | str,
        right_database_path: Path | str,
        *,
        application_version: str = "0.0.0",
        tool_version: str = "1.5.2",
    ) -> dict[str, Any]:
        runner = self.runner_for(
            left_database_path,
            application_version=application_version,
            tool_version=tool_version,
            owner_id="schema-equivalence-left",
        )
        return runner.prove_empty_to_latest_equivalence(
            other_database_path=right_database_path
        )

    def schema_fingerprint(self, database_path: Path | str) -> str:
        with open_duckdb_connection(database_path) as connection:
            return compute_schema_fingerprint(connection)

    def verify_installed(self, database_path: Path | str) -> dict[str, Any]:
        """Assert required tables, views, and lease/task identity columns."""

        with open_duckdb_connection(database_path) as connection:
            table_rows = connection.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
            present_tables = {
                str(row["table_name"] if isinstance(row, Mapping) else row[0])
                for row in table_rows
            }
            missing_tables = [
                name for name in required_tables() if name not in present_tables
            ]
            if missing_tables:
                raise ControlPlaneSchemaDriftError(
                    f"installed schema missing tables: {missing_tables}"
                )

            present_views = _list_views(connection)
            missing_views = [
                name for name in required_views() if name not in present_views
            ]
            if missing_views:
                raise ControlPlaneSchemaDriftError(
                    f"installed schema missing views: {missing_views}"
                )

            task_columns = _table_columns(connection, "tasks")
            lease_columns = _table_columns(connection, "leases")
            missing_task = [
                name for name in TASK_IDENTITY_COLUMNS if name not in task_columns
            ]
            missing_lease = [
                name
                for name in LEASE_IDENTITY_COLUMNS
                if name not in lease_columns
            ]
            if missing_task:
                raise ControlPlaneSchemaDriftError(
                    f"tasks identity columns missing: {missing_task}"
                )
            if missing_lease:
                raise ControlPlaneSchemaDriftError(
                    f"leases identity columns missing: {missing_lease}"
                )

            fingerprint = compute_schema_fingerprint(connection)
            return {
                "schema": self.SCHEMA,
                "interface": self.INTERFACE,
                "table_count": len(present_tables),
                "view_count": len(present_views),
                "schema_fingerprint": fingerprint,
                "task_identity_columns": list(TASK_IDENTITY_COLUMNS),
                "lease_identity_columns": list(LEASE_IDENTITY_COLUMNS),
            }

    def quack_profile(self) -> Mapping[str, Any]:
        """Return the capability profile bound to this dependency pin."""

        profile = default_compatibility_profile()
        payload = profile.to_dict()
        payload["dependency_profile"] = self.dependency_profile.to_dict()
        return MappingProxyType(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "version": int(self.version),
            "migration_id": self.migration_id,
            "migration_version": int(self.migration_version),
            "domains": list(self.domains),
            "tables": list(self.tables),
            "views": list(self.views),
            "bookkeeping_tables": list(self.bookkeeping_tables),
            "dependency_profile": self.dependency_profile.to_dict(),
            "sql_path": self.sql_path,
            "task_identity_columns": list(TASK_IDENTITY_COLUMNS),
            "lease_identity_columns": list(LEASE_IDENTITY_COLUMNS),
        }


def _table_columns(connection: Any, table_name: str) -> set[str]:
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table_name],
    ).fetchall()
    return {
        str(row["column_name"] if isinstance(row, Mapping) else row[0])
        for row in rows
    }


def _list_views(connection: Any) -> set[str]:
    """Return view names from information_schema (tables and/or views)."""

    names: set[str] = set()
    errors: list[Exception] = []
    queries = (
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'VIEW'
        """,
        """
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = 'main'
        """,
    )
    for sql in queries:
        try:
            rows = connection.execute(sql).fetchall()
        except Exception as exc:  # DuckDB catalog surface differs by build.
            errors.append(exc)
            continue
        names.update(
            str(row["table_name"] if isinstance(row, Mapping) else row[0])
            for row in rows
        )
    if not names and errors:
        raise ControlPlaneSchemaError(
            "unable to enumerate installed views from information_schema"
        ) from errors[0]
    return names


def assert_sql_preserves_identity_columns(sql_text: str | None = None) -> None:
    """Static check: base SQL declares task/lease identity columns."""

    text = sql_text if sql_text is not None else load_base_migration_sql()
    upper = text.upper()
    if "CREATE TABLE TASKS" not in upper:
        raise ControlPlaneSchemaError("base SQL must create tasks")
    if "CREATE TABLE LEASES" not in upper:
        raise ControlPlaneSchemaError("base SQL must create leases")
    for column in TASK_IDENTITY_COLUMNS:
        if column not in text:
            raise ControlPlaneSchemaError(
                f"tasks identity column {column!r} missing from base SQL"
            )
    for column in LEASE_IDENTITY_COLUMNS:
        if column not in text:
            raise ControlPlaneSchemaError(
                f"leases identity column {column!r} missing from base SQL"
            )
    # Join-critical identity must not exist only inside opaque JSON keys.
    for token in ("task_cid_json", "lease_json_only"):
        if token in text:
            raise ControlPlaneSchemaError(
                f"forbidden JSON-only identity token present: {token}"
            )


def assert_pyproject_pins_supervisor_duckdb(
    pyproject_path: Path | str | None = None,
) -> dict[str, Any]:
    """Verify ``pyproject.toml`` pins the optional supervisor DuckDB extra."""

    if pyproject_path is None:
        # control_plane_schema.py -> task_sources -> agent_supervisor ->
        # ipfs_accelerate_py -> repository root
        root = Path(__file__).resolve().parents[3]
        path = root / "pyproject.toml"
    else:
        path = Path(pyproject_path)
    if not path.is_file():
        raise ControlPlaneSchemaError(f"pyproject.toml not found: {path}")
    text = path.read_text(encoding="utf-8")
    if f"{PINNED_OPTIONAL_EXTRA} = [" not in text and f'{PINNED_OPTIONAL_EXTRA} = [' not in text:
        # TOML may use agent-supervisor = [
        if "agent-supervisor" not in text:
            raise ControlPlaneSchemaError(
                "pyproject.toml must declare optional extra agent-supervisor"
            )
    if "duckdb>=1.5" not in text and "duckdb >= 1.5" not in text:
        raise ControlPlaneSchemaError(
            "pyproject.toml must pin duckdb 1.5.x for the supervisor service"
        )
    if "<1.6" not in text and "< 1.6" not in text:
        raise ControlPlaneSchemaError(
            "pyproject.toml must upper-bound duckdb below 1.6 for the pin"
        )
    if "*.sql" not in text and "sql/*.sql" not in text:
        raise ControlPlaneSchemaError(
            "pyproject.toml package-data must include control-plane SQL files"
        )
    return {
        "path": str(path),
        "optional_extra": PINNED_OPTIONAL_EXTRA,
        "duckdb_requirement": PINNED_DUCKDB_REQUIREMENT,
    }


__all__ = [
    "BASE_MIGRATION_FILENAME",
    "BASE_MIGRATION_ID",
    "BASE_MIGRATION_VERSION",
    "BOOKKEEPING_TABLES",
    "CONTROL_PLANE_DEPENDENCY_PROFILE_SCHEMA",
    "CONTROL_PLANE_SCHEMA_INTERFACE",
    "CONTROL_PLANE_SCHEMA_SCHEMA",
    "CONTROL_PLANE_SCHEMA_VERSION",
    "ControlPlaneDependencyProfile",
    "ControlPlaneSchema",
    "ControlPlaneSchemaDriftError",
    "ControlPlaneSchemaError",
    "DIAGNOSTIC_VIEWS",
    "DOMAIN_TABLES",
    "LEASE_IDENTITY_COLUMNS",
    "PINNED_DUCKDB_REQUIREMENT",
    "PINNED_OPTIONAL_EXTRA",
    "PINNED_PROFILE_ID",
    "SchemaDomain",
    "TASK_IDENTITY_COLUMNS",
    "all_domain_tables",
    "assert_pyproject_pins_supervisor_duckdb",
    "assert_sql_preserves_identity_columns",
    "base_migration_sql_path",
    "load_base_migration_sql",
    "package_sql_directory",
    "required_tables",
    "required_views",
]
