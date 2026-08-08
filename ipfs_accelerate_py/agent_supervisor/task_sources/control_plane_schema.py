"""Installable normalized control-plane schema (ControlPlaneSchema@1).

This module is the Python companion to ``sql/0001_control_plane.sql``. It
exposes a closed domain inventory, required table/view names, the pinned
DuckDB dependency profile for the optional agent-supervisor service, and
helpers that install the schema through the checksum-bound migration runner.

Join-critical identities (``task_cid``, lease fencing fields, repository and
evidence digests, revisions, epochs) are first-class columns in SQL. Opaque
JSON is allowed only for bounded extension payloads.

Import is side-effect free: no filesystem, database, network, provider, or
process action occurs at module load.
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

CONTROL_PLANE_SCHEMA_INTERFACE: Final[str] = "ControlPlaneSchema@1"
CONTROL_PLANE_SCHEMA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-schema@1"
)
CONTROL_PLANE_SCHEMA_VERSION: Final[int] = 1
CONTROL_PLANE_MIGRATION_VERSION: Final[int] = 1
CONTROL_PLANE_MIGRATION_ID: Final[str] = "0001_control_plane"

# Logical domains installed by 0001_control_plane.sql.
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

# Domain tables required after applying migration 0001.
REQUIRED_DOMAIN_TABLES: Final[tuple[str, ...]] = (
    # meta
    "schema_contracts",
    "state_servers",
    "server_epochs",
    "client_sessions",
    "capability_snapshots",
    # control
    "authorization_roles",
    "authorization_grants",
    "credentials",
    "backup_snapshots",
    "restore_receipts",
    "maintenance_leases",
    # git
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
    # intent
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
    "findings",
    "finding_dispositions",
    # schedule
    "task_assignments",
    "task_blocks",
    "refill_epochs",
    # runtime
    "supervisor_instances",
    "daemon_instances",
    "daemon_sessions",
    "heartbeats",
    "stall_detections",
    "restart_decisions",
    "leases",
    "lease_events",
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
    "health_samples",
    "domain_events",
    "structured_logs",
    "metrics",
    "metric_samples",
    "budget_reservations",
    "budget_consumption",
    "quack_query_telemetry",
    # code
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
    "references_graph",
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
    # evidence
    "proof_obligations",
    "proof_attempts",
    "counterexamples",
    "evidence_nodes",
    # cache
    "context_manifests",
    "context_members",
    "context_deltas",
    "prompt_templates",
    "prompt_instances",
    "prompt_inputs",
    "decision_cache_entries",
    "replay_suppressions",
    # improve
    "provider_calls",
    "provider_responses",
    "failure_signatures",
    "churn_metrics",
)

REQUIRED_VIEWS: Final[tuple[str, ...]] = (
    "ready_task_context_v1",
    "live_sessions_v1",
    "expiring_leases_v1",
    "stuck_phases_v1",
    "schema_domain_inventory_v1",
)

# Columns that must remain first-class (not only inside JSON).
TASK_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "task_alias",
    "goal_cid",
    "status",
    "revision",
    "fence_epoch",
)

LEASE_SEMANTIC_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "claim_cid",
    "owner_session_id",
    "fence_epoch",
    "fencing_token",
    "attempt",
    "state",
    "expires_at",
    "revision",
)

# Optional supervisor service pin (DQP-005 / Quack 1.5.x profile).
PINNED_DUCKDB_MAJOR: Final[int] = 1
PINNED_DUCKDB_MINOR: Final[int] = 5
PINNED_DUCKDB_VERSION_PREFIX: Final[str] = "1.5"
PINNED_DUCKDB_REQUIREMENT: Final[str] = "duckdb>=1.5.0,<1.6.0"
PINNED_OPTIONAL_EXTRA: Final[str] = "agent-supervisor"
PINNED_EXTENSION_NAME: Final[str] = "quack"

_DOMAIN_TABLE_MAP: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "meta": (
            "schema_contracts",
            "state_servers",
            "server_epochs",
            "client_sessions",
            "capability_snapshots",
        ),
        "control": (
            "authorization_roles",
            "authorization_grants",
            "credentials",
            "backup_snapshots",
            "restore_receipts",
            "maintenance_leases",
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
            "findings",
            "finding_dispositions",
        ),
        "schedule": (
            "task_assignments",
            "task_blocks",
            "refill_epochs",
        ),
        "runtime": (
            "supervisor_instances",
            "daemon_instances",
            "daemon_sessions",
            "heartbeats",
            "stall_detections",
            "restart_decisions",
            "leases",
            "lease_events",
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
            "health_samples",
            "domain_events",
            "structured_logs",
            "metrics",
            "metric_samples",
            "budget_reservations",
            "budget_consumption",
            "quack_query_telemetry",
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
            "references_graph",
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
            "decision_cache_entries",
            "replay_suppressions",
        ),
        "improve": (
            "provider_calls",
            "provider_responses",
            "failure_signatures",
            "churn_metrics",
        ),
    }
)

_VERSION_RE: Final = re.compile(
    r"^\s*v?(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?"
)


class ControlPlaneSchemaError(RuntimeError):
    """Base class for fail-closed control-plane schema errors."""


class ControlPlaneSchemaNotInstalledError(ControlPlaneSchemaError):
    """The required domain schema is missing or incomplete."""


class ControlPlaneSchemaCompatibilityError(ControlPlaneSchemaError):
    """Existing tables are incompatible with the normalized schema."""


class ControlPlaneSchemaDependencyError(ControlPlaneSchemaError):
    """The pinned DuckDB dependency profile is unavailable or mismatched."""


def package_sql_path() -> Path:
    """Return the package SQL directory containing ordered migrations."""

    return Path(__file__).resolve().parent / "sql"


def control_plane_sql_path() -> Path:
    """Return the path to migration ``0001_control_plane.sql``."""

    return package_sql_path() / f"{CONTROL_PLANE_MIGRATION_ID}.sql"


def load_control_plane_sql() -> str:
    """Load the normalized control-plane SQL migration text."""

    path = control_plane_sql_path()
    if not path.is_file():
        raise ControlPlaneSchemaError(
            f"control-plane SQL migration is missing: {path}"
        )
    return path.read_text(encoding="utf-8")


def default_schema_catalog() -> MigrationCatalog:
    """Load the package migration catalog (includes domain SQL when present)."""

    return load_default_catalog(package_sql_path())


def domain_table_map() -> Mapping[str, tuple[str, ...]]:
    """Return the closed domain → table inventory."""

    return _DOMAIN_TABLE_MAP


def required_tables() -> tuple[str, ...]:
    """Return bookkeeping plus domain tables required after install."""

    return BOOKKEEPING_TABLES + REQUIRED_DOMAIN_TABLES


def required_views() -> tuple[str, ...]:
    """Return constrained diagnostic/context views required after install."""

    return REQUIRED_VIEWS


def pinned_duckdb_requirement() -> str:
    """Return the explicit DuckDB pin for the optional supervisor service."""

    return PINNED_DUCKDB_REQUIREMENT


def parse_duckdb_version(version: str) -> tuple[int, int, int] | None:
    """Parse a DuckDB version string into (major, minor, patch)."""

    match = _VERSION_RE.match(str(version or ""))
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch") or 0),
    )


def duckdb_version_matches_pin(version: str) -> bool:
    """Return whether ``version`` is inside the pinned 1.5.x window."""

    parsed = parse_duckdb_version(version)
    if parsed is None:
        return False
    major, minor, _patch = parsed
    return major == PINNED_DUCKDB_MAJOR and minor == PINNED_DUCKDB_MINOR


def _row_value(row: Any, index: int = 0, key: str | None = None) -> Any:
    if row is None:
        return None
    if isinstance(row, Mapping):
        if key is not None and key in row:
            return row[key]
        return next(iter(row.values()))
    try:
        return row[index]
    except Exception:
        return None


def _table_names(connection: Any) -> set[str]:
    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND upper(table_type) IN ('BASE TABLE', 'TABLE')
        """
    ).fetchall()
    names: set[str] = set()
    for row in rows:
        value = _row_value(row, 0, "table_name")
        if value is not None:
            names.add(str(value))
    return names


def _view_names(connection: Any) -> set[str]:
    rows = connection.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND upper(table_type) = 'VIEW'
        """
    ).fetchall()
    names: set[str] = set()
    for row in rows:
        value = _row_value(row, 0, "table_name")
        if value is not None:
            names.add(str(value))
    return names


def _column_names(connection: Any, table_name: str) -> set[str]:
    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table_name],
    ).fetchall()
    names: set[str] = set()
    for row in rows:
        value = _row_value(row, 0, "column_name")
        if value is not None:
            names.add(str(value))
    return names


@dataclass(frozen=True)
class ControlPlaneSchema:
    """ControlPlaneSchema@1 installer and inspector over the migration catalog."""

    INTERFACE: ClassVar[str] = CONTROL_PLANE_SCHEMA_INTERFACE
    SCHEMA: ClassVar[str] = CONTROL_PLANE_SCHEMA_SCHEMA
    VERSION: ClassVar[int] = CONTROL_PLANE_SCHEMA_VERSION

    catalog: MigrationCatalog
    application_version: str = "control-plane-schema"
    tool_version: str | None = None

    @classmethod
    def default(cls, *, tool_version: str | None = None) -> "ControlPlaneSchema":
        """Construct from the package SQL catalog."""

        return cls(catalog=default_schema_catalog(), tool_version=tool_version)

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, MigrationCatalog):
            raise TypeError("catalog must be a MigrationCatalog")
        if self.catalog.latest_version < CONTROL_PLANE_MIGRATION_VERSION:
            raise ControlPlaneSchemaError(
                "package catalog is missing 0001_control_plane domain SQL; "
                f"latest_version={self.catalog.latest_version}"
            )
        migration = self.catalog.get(CONTROL_PLANE_MIGRATION_VERSION)
        if migration.migration_id != CONTROL_PLANE_MIGRATION_ID:
            raise ControlPlaneSchemaError(
                f"expected migration_id {CONTROL_PLANE_MIGRATION_ID}, "
                f"got {migration.migration_id}"
            )

    def runner_for(
        self,
        database_path: Path | str,
        *,
        owner_id: str | None = None,
    ) -> ControlPlaneMigrationRunner:
        """Return a migration runner bound to this schema catalog."""

        return ControlPlaneMigrationRunner.for_database(
            database_path,
            catalog=self.catalog,
            application_version=self.application_version,
            tool_version=self.tool_version,
            owner_id=owner_id,
        )

    def install(
        self,
        database_path: Path | str,
        *,
        owner_id: str | None = None,
    ) -> MigrationRunReport:
        """Apply migrations through the latest control-plane schema version."""

        runner = self.runner_for(database_path, owner_id=owner_id)
        report = runner.apply(target_version=CONTROL_PLANE_MIGRATION_VERSION)
        self.verify_installed(database_path)
        return report

    def schema_fingerprint(self, database_path: Path | str) -> str:
        """Return the canonical information-schema fingerprint."""

        runner = self.runner_for(database_path)
        return runner.schema_fingerprint()

    def prove_empty_to_latest_equivalence(
        self,
        left_database_path: Path | str,
        right_database_path: Path | str,
    ) -> dict[str, Any]:
        """Prove two empty databases install to the same fingerprint."""

        left = self.runner_for(left_database_path, owner_id="schema-proof-left")
        return left.prove_empty_to_latest_equivalence(
            other_database_path=right_database_path
        )

    def inspect(self, database_path: Path | str) -> dict[str, Any]:
        """Return an inspectable snapshot of installed schema state."""

        runner = self.runner_for(database_path)
        base = runner.inspect()
        with open_duckdb_connection(database_path) as connection:
            tables = sorted(_table_names(connection))
            views = sorted(_view_names(connection))
            domains = {
                domain: list(members)
                for domain, members in _DOMAIN_TABLE_MAP.items()
            }
            missing_tables = sorted(
                set(required_tables()) - set(tables)
            )
            missing_views = sorted(set(required_views()) - set(views))
            task_columns = sorted(_column_names(connection, "tasks"))
            lease_columns = sorted(_column_names(connection, "leases"))
        return {
            **base,
            "interface": self.INTERFACE,
            "schema": self.SCHEMA,
            "schema_version": self.VERSION,
            "domains": domains,
            "tables": tables,
            "views": views,
            "missing_tables": missing_tables,
            "missing_views": missing_views,
            "task_identity_columns": task_columns,
            "lease_semantic_columns": lease_columns,
            "pinned_duckdb_requirement": PINNED_DUCKDB_REQUIREMENT,
            "installed": not missing_tables and not missing_views,
        }

    def verify_installed(self, database_path: Path | str) -> dict[str, Any]:
        """Fail closed when required tables, views, or identity columns are absent."""

        snapshot = self.inspect(database_path)
        if snapshot["missing_tables"] or snapshot["missing_views"]:
            raise ControlPlaneSchemaNotInstalledError(
                "control-plane schema is incomplete: "
                f"missing_tables={snapshot['missing_tables']}, "
                f"missing_views={snapshot['missing_views']}"
            )
        task_columns = set(snapshot["task_identity_columns"])
        missing_task = sorted(set(TASK_IDENTITY_COLUMNS) - task_columns)
        if missing_task:
            raise ControlPlaneSchemaCompatibilityError(
                f"tasks table missing identity columns: {missing_task}"
            )
        lease_columns = set(snapshot["lease_semantic_columns"])
        missing_lease = sorted(set(LEASE_SEMANTIC_COLUMNS) - lease_columns)
        if missing_lease:
            raise ControlPlaneSchemaCompatibilityError(
                f"leases table missing semantic columns: {missing_lease}"
            )
        if int(snapshot["current_version"]) < CONTROL_PLANE_MIGRATION_VERSION:
            raise ControlPlaneSchemaNotInstalledError(
                "control-plane migration 0001 is not applied"
            )
        return snapshot

    def verify_task_and_lease_semantics(
        self,
        database_path: Path | str,
    ) -> dict[str, Any]:
        """Exercise task_cid / lease insert and unique identity constraints."""

        self.verify_installed(database_path)
        now = "2020-01-01T00:00:00Z"
        with open_duckdb_connection(database_path) as connection:
            connection.execute(
                """
                INSERT INTO goals (
                    goal_cid, goal_alias, parent_goal_cid, objective_id,
                    title, status, ordinal, created_at, updated_at, revision
                ) VALUES (
                    'goal:cid:schema-probe', 'GOAL-SCHEMA', NULL, NULL,
                    'schema probe goal', 'active', 1, ?, ?, 1
                )
                """,
                [now, now],
            )
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_id, status, priority,
                    ordinal, track, created_at, updated_at, revision, fence_epoch
                ) VALUES (
                    'task:cid:schema-probe', 'TASK-SCHEMA',
                    'goal:cid:schema-probe', NULL, 'ready', 'P0',
                    1, 'schema', ?, ?, 1, 0
                )
                """,
                [now, now],
            )
            connection.execute(
                """
                INSERT INTO leases (
                    task_cid, claim_cid, resolution_cid, owner_session_id,
                    claimant_did, fence_epoch, fencing_token, attempt, state,
                    acquired_at, expires_at, started_at, release_reason,
                    retry_not_before, revision
                ) VALUES (
                    'task:cid:schema-probe', 'claim:cid:schema-probe', '',
                    'session:schema-probe', 'did:schema-probe', 1, 1, 1,
                    'accepted', ?, ?, ?, '', NULL, 1
                )
                """,
                [now, "2020-01-01T01:00:00Z", now],
            )
            # task_cid is the durable primary key for both tasks and leases.
            task_row = connection.execute(
                "SELECT task_cid, status, revision FROM tasks WHERE task_cid = ?",
                ["task:cid:schema-probe"],
            ).fetchone()
            lease_row = connection.execute(
                """
                SELECT task_cid, fencing_token, fence_epoch, state
                FROM leases WHERE task_cid = ?
                """,
                ["task:cid:schema-probe"],
            ).fetchone()
            view_row = connection.execute(
                """
                SELECT task_cid FROM ready_task_context_v1
                WHERE task_cid = ?
                """,
                ["task:cid:schema-probe"],
            ).fetchone()
            # Accepted lease removes the task from the ready context view.
            if view_row is not None:
                raise ControlPlaneSchemaCompatibilityError(
                    "ready_task_context_v1 must exclude accepted leases"
                )
            # Duplicate task_cid must fail (unique/primary key).
            duplicate_failed = False
            try:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_cid, task_alias, goal_cid, plan_id, status,
                        priority, ordinal, track, created_at, updated_at,
                        revision, fence_epoch
                    ) VALUES (
                        'task:cid:schema-probe', 'TASK-SCHEMA-DUP',
                        'goal:cid:schema-probe', NULL, 'ready', 'P0',
                        2, 'schema', ?, ?, 1, 0
                    )
                    """,
                    [now, now],
                )
            except Exception:
                duplicate_failed = True
            if not duplicate_failed:
                raise ControlPlaneSchemaCompatibilityError(
                    "tasks.task_cid must reject duplicates"
                )
        return {
            "task_cid": str(_row_value(task_row, 0, "task_cid")),
            "task_status": str(_row_value(task_row, 1, "status")),
            "task_revision": int(_row_value(task_row, 2, "revision")),
            "lease_task_cid": str(_row_value(lease_row, 0, "task_cid")),
            "lease_fencing_token": int(_row_value(lease_row, 1, "fencing_token")),
            "lease_fence_epoch": int(_row_value(lease_row, 2, "fence_epoch")),
            "lease_state": str(_row_value(lease_row, 3, "state")),
            "ready_view_excludes_accepted_lease": True,
            "task_cid_unique": True,
        }

    def verify_existing_table_compatibility(
        self,
        database_path: Path | str,
        *,
        existing_task_cids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Confirm installed schema can host pre-existing task CID identities."""

        self.verify_installed(database_path)
        retained: list[str] = []
        now = "2020-01-02T00:00:00Z"
        with open_duckdb_connection(database_path) as connection:
            connection.execute(
                """
                INSERT INTO goals (
                    goal_cid, goal_alias, parent_goal_cid, objective_id,
                    title, status, ordinal, created_at, updated_at, revision
                ) VALUES (
                    'goal:cid:compat', 'GOAL-COMPAT', NULL, NULL,
                    'compatibility goal', 'active', 100, ?, ?, 1
                )
                ON CONFLICT (goal_cid) DO NOTHING
                """,
                [now, now],
            )
            for index, task_cid in enumerate(existing_task_cids):
                cid = str(task_cid).strip()
                if not cid:
                    raise ControlPlaneSchemaCompatibilityError(
                        "existing task_cid values must be non-empty"
                    )
                connection.execute(
                    """
                    INSERT INTO tasks (
                        task_cid, task_alias, goal_cid, plan_id, status,
                        priority, ordinal, track, created_at, updated_at,
                        revision, fence_epoch
                    ) VALUES (?, ?, 'goal:cid:compat', NULL, 'ready', 'P1',
                              ?, 'compat', ?, ?, 1, 0)
                    ON CONFLICT (task_cid) DO NOTHING
                    """,
                    [cid, f"COMPAT-{index}", 1000 + index, now, now],
                )
                row = connection.execute(
                    "SELECT task_cid FROM tasks WHERE task_cid = ?",
                    [cid],
                ).fetchone()
                if row is None:
                    raise ControlPlaneSchemaCompatibilityError(
                        f"failed to retain existing task_cid: {cid}"
                    )
                retained.append(cid)
        return {
            "retained_task_cids": retained,
            "count": len(retained),
            "compatible": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable schema contract summary."""

        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": self.VERSION,
            "migration_id": CONTROL_PLANE_MIGRATION_ID,
            "migration_version": CONTROL_PLANE_MIGRATION_VERSION,
            "domains": list(SCHEMA_DOMAINS),
            "required_tables": list(required_tables()),
            "required_views": list(required_views()),
            "task_identity_columns": list(TASK_IDENTITY_COLUMNS),
            "lease_semantic_columns": list(LEASE_SEMANTIC_COLUMNS),
            "pinned_duckdb_requirement": PINNED_DUCKDB_REQUIREMENT,
            "pinned_optional_extra": PINNED_OPTIONAL_EXTRA,
            "pinned_extension_name": PINNED_EXTENSION_NAME,
            "catalog_latest_version": self.catalog.latest_version,
            "catalog_fingerprint": self.catalog.fingerprint(),
        }


def assert_pyproject_pins_supervisor_duckdb(pyproject_text: str) -> None:
    """Fail closed when pyproject.toml lacks the optional supervisor DuckDB pin."""

    text = str(pyproject_text)
    if PINNED_OPTIONAL_EXTRA not in text and "agent_supervisor" not in text:
        raise ControlPlaneSchemaDependencyError(
            "pyproject.toml must declare an optional agent-supervisor extra"
        )
    if "duckdb>=1.5.0" not in text and "duckdb >= 1.5.0" not in text:
        raise ControlPlaneSchemaDependencyError(
            "pyproject.toml must pin duckdb>=1.5.0 for the supervisor service"
        )
    if "<1.6.0" not in text and "< 1.6.0" not in text:
        raise ControlPlaneSchemaDependencyError(
            "pyproject.toml must upper-bound duckdb to the 1.5.x line (<1.6.0)"
        )


__all__ = [
    "BOOKKEEPING_TABLES",
    "CONTROL_PLANE_MIGRATION_ID",
    "CONTROL_PLANE_MIGRATION_VERSION",
    "CONTROL_PLANE_SCHEMA_INTERFACE",
    "CONTROL_PLANE_SCHEMA_SCHEMA",
    "CONTROL_PLANE_SCHEMA_VERSION",
    "ControlPlaneSchema",
    "ControlPlaneSchemaCompatibilityError",
    "ControlPlaneSchemaDependencyError",
    "ControlPlaneSchemaError",
    "ControlPlaneSchemaNotInstalledError",
    "LEASE_SEMANTIC_COLUMNS",
    "PINNED_DUCKDB_MAJOR",
    "PINNED_DUCKDB_MINOR",
    "PINNED_DUCKDB_REQUIREMENT",
    "PINNED_DUCKDB_VERSION_PREFIX",
    "PINNED_EXTENSION_NAME",
    "PINNED_OPTIONAL_EXTRA",
    "REQUIRED_DOMAIN_TABLES",
    "REQUIRED_VIEWS",
    "SCHEMA_DOMAINS",
    "TASK_IDENTITY_COLUMNS",
    "assert_pyproject_pins_supervisor_duckdb",
    "control_plane_sql_path",
    "default_schema_catalog",
    "domain_table_map",
    "duckdb_available",
    "duckdb_version_matches_pin",
    "load_control_plane_sql",
    "package_sql_path",
    "parse_duckdb_version",
    "pinned_duckdb_requirement",
    "required_tables",
    "required_views",
    "compute_schema_fingerprint",
]
