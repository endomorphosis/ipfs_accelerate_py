"""Versioned mutable external-agent control-plane schema (EAAEF-090).

Interface: ``ExternalAgentControlSchema@1``

Owns the closed collection inventory for the complete mutable coordination
plane. DuckDB plus one fenced Quack owner is the sole current authority.
DuckLake is a history-only projection and never grants a claim, lease, fence,
write owner, resume right, merge authority, or finalization decision.

Physical names reuse the existing control-plane inventory where those
relations already exist. Families that the operational profile does not yet
normalize receive new closed collection names here; this module does not
rewrite ``control_plane_schema`` or ``eaaef_operational_schema``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .task_identity import canonical_content_cid

EXTERNAL_AGENT_CONTROL_SCHEMA_INTERFACE: Final = "ExternalAgentControlSchema@1"
EXTERNAL_AGENT_CONTROL_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-control-plane@1"
)
EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION: Final[int] = 1
EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION_HISTORY: Final[tuple[int, ...]] = (1,)
EXTERNAL_AGENT_CONTROL_MIGRATION_ID: Final = "0001_external_agent_control_plane"

MUTABLE_COORDINATION_AUTHORITY: Final = "one_fenced_quack_owner"
COLLECTION_AUTHORITY_MUTABLE: Final = "mutable"
COLLECTION_AUTHORITY_HISTORY_ONLY: Final = "history_only"
DUCKLAKE_HISTORY_ONLY_MARKER: Final = "history_only"
DUCKLAKE_GRANTS_CURRENT_AUTHORITY: Final = False
DUCKLAKE_SOURCE_NAME: Final = "ducklake"

# Objective families. Order is documentation-only and fingerprint-stable.
REQUIRED_COLLECTION_FAMILIES: Final[tuple[str, ...]] = (
    "repositories",
    "handoffs",
    "sessions",
    "runs",
    "goal_revisions",
    "plan_revisions",
    "task_revisions",
    "conflicts",
    "processes",
    "containers",
    "claims",
    "leases",
    "reservations",
    "approvals",
    "events",
    "checkpoints",
    "validations",
    "proofs",
    "merge",
    "artifacts",
    "migrations",
    "cursors",
)

# Mutable DuckDB relations. Existing names are reused; new names complete the
# objective inventory without expanding DuckLake authority.
_MUTABLE_COLLECTIONS: Final[dict[str, tuple[str, ...]]] = {
    "repositories": ("repositories", "repository_revisions"),
    "handoffs": ("handoffs",),
    "sessions": ("daemon_sessions", "client_sessions"),
    "runs": ("runs",),
    "goal_revisions": ("goals", "goal_revisions"),
    "plan_revisions": ("plans", "plan_revisions"),
    "task_revisions": ("tasks", "task_revisions"),
    "conflicts": ("conflicts",),
    "processes": ("processes",),
    "containers": ("containers",),
    "claims": (
        "task_claims",
        "resource_claims",
        "path_claims",
        "effect_claims",
    ),
    "leases": ("leases", "lease_events", "maintenance_leases"),
    "reservations": ("budget_reservations",),
    "approvals": ("approvals",),
    "events": ("domain_events",),
    "checkpoints": ("checkpoints",),
    "validations": (
        "task_validations",
        "validation_runs",
        "validation_results",
    ),
    "proofs": ("completion_receipts", "evidence_nodes"),
    "merge": ("merge_attempts", "merge_queue_entries", "merge_bases"),
    "artifacts": ("artifacts",),
    "migrations": ("schema_migrations", "schema_migration_attempts"),
    "cursors": ("cursors", "outbox_cursors"),
}

MUTABLE_COLLECTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {family: tuple(tables) for family, tables in _MUTABLE_COLLECTIONS.items()}
)

# DuckLake projections of committed history. None of these may be treated as
# live coordination authority.
HISTORY_ONLY_COLLECTIONS: Final[tuple[str, ...]] = (
    "ducklake_epochs",
    "ducklake_task_history",
    "ducklake_event_history",
    "ducklake_audit_history",
    "ducklake_snapshots",
    "ducklake_lineage",
    "ducklake_benchmarks",
    "ducklake_recovery_manifests",
)

JOIN_CRITICAL_IDENTITIES: Final[tuple[tuple[str, str], ...]] = (
    ("repositories", "repository_id"),
    ("repository_revisions", "repository_id"),
    ("handoffs", "handoff_id"),
    ("daemon_sessions", "session_id"),
    ("client_sessions", "session_id"),
    ("runs", "run_id"),
    ("goals", "goal_cid"),
    ("goal_revisions", "goal_cid"),
    ("plans", "plan_cid"),
    ("plan_revisions", "plan_cid"),
    ("tasks", "task_cid"),
    ("task_revisions", "task_cid"),
    ("conflicts", "conflict_id"),
    ("processes", "process_id"),
    ("containers", "container_id"),
    ("task_claims", "claim_id"),
    ("task_claims", "task_cid"),
    ("resource_claims", "resource_id"),
    ("path_claims", "path"),
    ("effect_claims", "effect_id"),
    ("leases", "task_cid"),
    ("leases", "fencing_token"),
    ("lease_events", "task_cid"),
    ("maintenance_leases", "lease_id"),
    ("budget_reservations", "reservation_id"),
    ("approvals", "approval_id"),
    ("domain_events", "event_id"),
    ("domain_events", "stream_id"),
    ("domain_events", "sequence"),
    ("checkpoints", "checkpoint_id"),
    ("task_validations", "task_cid"),
    ("validation_runs", "run_id"),
    ("validation_results", "result_id"),
    ("completion_receipts", "receipt_cid"),
    ("completion_receipts", "task_cid"),
    ("evidence_nodes", "evidence_id"),
    ("merge_attempts", "merge_attempt_id"),
    ("merge_queue_entries", "task_cid"),
    ("merge_bases", "repository_id"),
    ("artifacts", "cid"),
    ("schema_migrations", "version"),
    ("cursors", "cursor_id"),
    ("outbox_cursors", "cursor_id"),
)

_DUCKLAKE_TOKENS: Final[frozenset[str]] = frozenset(
    {"ducklake", "duck_lake", "lakehouse"}
)
_AUTHORITY_CLAIM_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "authoritative",
        "current_authority",
        "mutable",
        "coordination",
        "claim",
        "lease",
        "fence",
        "merge",
    }
)


class ExternalAgentControlSchemaError(RuntimeError):
    """Closed control-plane schema inventory is invalid or misused."""


def _flatten_tables(collections: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    tables: list[str] = []
    seen: set[str] = set()
    for family in REQUIRED_COLLECTION_FAMILIES:
        for table in collections[family]:
            name = str(table)
            if name in seen:
                raise ExternalAgentControlSchemaError(
                    f"duplicate mutable collection {name!r}"
                )
            tables.append(name)
            seen.add(name)
    return tuple(tables)


REQUIRED_COLLECTIONS: Final[tuple[str, ...]] = _flatten_tables(MUTABLE_COLLECTIONS)
REQUIRED_COLLECTION_SET: Final[frozenset[str]] = frozenset(REQUIRED_COLLECTIONS)
HISTORY_ONLY_COLLECTION_SET: Final[frozenset[str]] = frozenset(
    HISTORY_ONLY_COLLECTIONS
)


def _normalized_name(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        raise ExternalAgentControlSchemaError(f"{field_name} is required")
    return text


def _is_ducklake_token(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return False
    if text in _DUCKLAKE_TOKENS or text in HISTORY_ONLY_COLLECTION_SET:
        return True
    return any(token in text.split("_") for token in _DUCKLAKE_TOKENS) or text.startswith(
        "ducklake_"
    )


def required_collections(
    schema: ExternalAgentControlSchema | None = None,
) -> tuple[str, ...]:
    """Return the closed mutable collection names in fingerprint order."""

    inventory = schema or default_external_agent_control_schema()
    return inventory.all_mutable_collections


def assert_schema_version_monotonic(
    versions: Sequence[int] | None = None,
    *,
    current_version: int | None = None,
) -> tuple[int, ...]:
    """Reject empty, non-positive, receding, or skipped schema versions."""

    history = (
        EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION_HISTORY
        if versions is None
        else tuple(int(value) for value in versions)
    )
    if not history:
        raise ExternalAgentControlSchemaError("schema version history is empty")
    expected = 1
    for version in history:
        if version != expected:
            raise ExternalAgentControlSchemaError(
                "schema version history must be strictly monotonic integers "
                f"starting at 1; expected {expected}, got {version}"
            )
        expected += 1
    bound = (
        EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION
        if current_version is None
        else int(current_version)
    )
    if bound < 1:
        raise ExternalAgentControlSchemaError("schema_version must be >= 1")
    if history[-1] != bound:
        raise ExternalAgentControlSchemaError(
            "schema_version must equal the latest monotonic history entry"
        )
    return history


def admit_schema_version(
    candidate: int,
    *,
    previous: int | None = None,
) -> int:
    """Accept replay of the current version or the immediate successor."""

    prior = (
        EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION
        if previous is None
        else int(previous)
    )
    value = int(candidate)
    if prior < 1 or value < 1:
        raise ExternalAgentControlSchemaError("schema_version must be >= 1")
    if value < prior:
        raise ExternalAgentControlSchemaError(
            "schema version cannot recede"
        )
    if value > prior + 1:
        raise ExternalAgentControlSchemaError(
            "schema version cannot skip"
        )
    return value


def collection_authority(name: str) -> str:
    """Return ``mutable`` or ``history_only`` for a closed collection name."""

    table = _normalized_name(name, field_name="collection")
    if table in REQUIRED_COLLECTION_SET:
        if _is_ducklake_token(table):
            raise ExternalAgentControlSchemaError(
                f"{table} cannot be both mutable and a DuckLake projection"
            )
        return COLLECTION_AUTHORITY_MUTABLE
    if table in HISTORY_ONLY_COLLECTION_SET or _is_ducklake_token(table):
        return COLLECTION_AUTHORITY_HISTORY_ONLY
    raise ExternalAgentControlSchemaError(
        f"unknown control-plane collection: {name}"
    )


def reject_ducklake_authority(
    source: Any = DUCKLAKE_SOURCE_NAME,
    *,
    grants_current_authority: bool | None = None,
    role: str | None = None,
) -> Mapping[str, Any]:
    """Seal DuckLake as history-only and refuse current-authority claims."""

    token = _normalized_name(source, field_name="source")
    claimed_role = str(role or "").strip().lower()
    authority_claimed = grants_current_authority is True or (
        claimed_role in _AUTHORITY_CLAIM_TOKENS
    )
    if _is_ducklake_token(token) and authority_claimed:
        raise ExternalAgentControlSchemaError(
            "DuckLake never grants current authority"
        )
    if _is_ducklake_token(token) and token in REQUIRED_COLLECTION_SET:
        raise ExternalAgentControlSchemaError(
            "DuckLake collections cannot join the mutable inventory"
        )
    if not _is_ducklake_token(token):
        raise ExternalAgentControlSchemaError(
            "reject_ducklake_authority applies only to DuckLake sources"
        )
    marker = {
        "schema": EXTERNAL_AGENT_CONTROL_SCHEMA,
        "source": DUCKLAKE_SOURCE_NAME,
        "role": DUCKLAKE_HISTORY_ONLY_MARKER,
        "authority": COLLECTION_AUTHORITY_HISTORY_ONLY,
        "grants_current_authority": False,
        "mutable_coordination_authority": MUTABLE_COORDINATION_AUTHORITY,
        "history_only_collections": list(HISTORY_ONLY_COLLECTIONS),
    }
    return MappingProxyType(
        {**marker, "marker_cid": canonical_content_cid(marker)}
    )


@dataclass(frozen=True)
class ExternalAgentControlSchema:
    """Canonical inventory for the mutable external-agent control plane.

    Interface: ``ExternalAgentControlSchema@1``.
    """

    INTERFACE: ClassVar[str] = EXTERNAL_AGENT_CONTROL_SCHEMA_INTERFACE
    SCHEMA: ClassVar[str] = EXTERNAL_AGENT_CONTROL_SCHEMA

    schema_version: int = EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION
    version_history: tuple[int, ...] = EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION_HISTORY
    migration_id: str = EXTERNAL_AGENT_CONTROL_MIGRATION_ID
    families: tuple[str, ...] = REQUIRED_COLLECTION_FAMILIES
    collections: Mapping[str, tuple[str, ...]] = MUTABLE_COLLECTIONS
    history_only_collections: tuple[str, ...] = HISTORY_ONLY_COLLECTIONS
    join_critical_identities: tuple[tuple[str, str], ...] = JOIN_CRITICAL_IDENTITIES
    mutable_coordination_authority: str = MUTABLE_COORDINATION_AUTHORITY
    ducklake_role: str = DUCKLAKE_HISTORY_ONLY_MARKER
    ducklake_grants_current_authority: bool = DUCKLAKE_GRANTS_CURRENT_AUTHORITY

    def __post_init__(self) -> None:
        assert_schema_version_monotonic(
            self.version_history,
            current_version=int(self.schema_version),
        )
        if tuple(self.families) != REQUIRED_COLLECTION_FAMILIES:
            raise ExternalAgentControlSchemaError(
                "families must match the closed REQUIRED_COLLECTION_FAMILIES"
            )
        missing = [
            family for family in REQUIRED_COLLECTION_FAMILIES if family not in self.collections
        ]
        if missing:
            raise ExternalAgentControlSchemaError(
                f"collections missing families: {missing}"
            )
        extra = sorted(set(self.collections) - set(REQUIRED_COLLECTION_FAMILIES))
        if extra:
            raise ExternalAgentControlSchemaError(
                f"collections contain unknown families: {extra}"
            )
        tables = _flatten_tables(self.collections)
        ducklake_mutable = [
            name for name in tables if _is_ducklake_token(name)
        ]
        if ducklake_mutable:
            raise ExternalAgentControlSchemaError(
                "DuckLake collections cannot be mutable authority: "
                f"{ducklake_mutable}"
            )
        history = tuple(
            _normalized_name(name, field_name="history_only_collection")
            for name in self.history_only_collections
        )
        if history != HISTORY_ONLY_COLLECTIONS:
            raise ExternalAgentControlSchemaError(
                "history_only_collections must match the closed DuckLake inventory"
            )
        overlap = sorted(set(tables) & set(history))
        if overlap:
            raise ExternalAgentControlSchemaError(
                f"history-only collections overlap mutable inventory: {overlap}"
            )
        if self.ducklake_grants_current_authority is not False:
            raise ExternalAgentControlSchemaError(
                "DuckLake never grants current authority"
            )
        if self.ducklake_role != DUCKLAKE_HISTORY_ONLY_MARKER:
            raise ExternalAgentControlSchemaError(
                "DuckLake role must be the history-only marker"
            )
        if self.mutable_coordination_authority != MUTABLE_COORDINATION_AUTHORITY:
            raise ExternalAgentControlSchemaError(
                "mutable coordination authority must be the fenced Quack owner"
            )
        object.__setattr__(self, "history_only_collections", history)

    @property
    def all_mutable_collections(self) -> tuple[str, ...]:
        return _flatten_tables(self.collections)

    def fingerprint(self) -> str:
        return canonical_content_cid(self._fingerprint_payload())

    def _fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "schema_version": int(self.schema_version),
            "version_history": list(self.version_history),
            "migration_id": self.migration_id,
            "families": list(self.families),
            "collections": {
                family: list(self.collections[family]) for family in self.families
            },
            "history_only_collections": list(self.history_only_collections),
            "join_critical_identities": [
                {"table": table, "column": column}
                for table, column in self.join_critical_identities
            ],
            "mutable_coordination_authority": self.mutable_coordination_authority,
            "ducklake_role": self.ducklake_role,
            "ducklake_grants_current_authority": False,
        }

    def to_dict(self) -> Mapping[str, Any]:
        payload = {
            **self._fingerprint_payload(),
            "schema_fingerprint": self.fingerprint(),
            "required_collections": list(self.all_mutable_collections),
        }
        return MappingProxyType(payload)


def default_external_agent_control_schema() -> ExternalAgentControlSchema:
    """Return the package ExternalAgentControlSchema@1 inventory."""

    return ExternalAgentControlSchema()


def schema_fingerprint(
    schema: ExternalAgentControlSchema | None = None,
) -> str:
    """Return the stable content identity of the closed schema inventory."""

    return (schema or default_external_agent_control_schema()).fingerprint()


__all__ = (
    "COLLECTION_AUTHORITY_HISTORY_ONLY",
    "COLLECTION_AUTHORITY_MUTABLE",
    "DUCKLAKE_GRANTS_CURRENT_AUTHORITY",
    "DUCKLAKE_HISTORY_ONLY_MARKER",
    "DUCKLAKE_SOURCE_NAME",
    "EXTERNAL_AGENT_CONTROL_MIGRATION_ID",
    "EXTERNAL_AGENT_CONTROL_SCHEMA",
    "EXTERNAL_AGENT_CONTROL_SCHEMA_INTERFACE",
    "EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION",
    "EXTERNAL_AGENT_CONTROL_SCHEMA_VERSION_HISTORY",
    "HISTORY_ONLY_COLLECTIONS",
    "HISTORY_ONLY_COLLECTION_SET",
    "JOIN_CRITICAL_IDENTITIES",
    "MUTABLE_COLLECTIONS",
    "MUTABLE_COORDINATION_AUTHORITY",
    "REQUIRED_COLLECTIONS",
    "REQUIRED_COLLECTION_FAMILIES",
    "REQUIRED_COLLECTION_SET",
    "ExternalAgentControlSchema",
    "ExternalAgentControlSchemaError",
    "admit_schema_version",
    "assert_schema_version_monotonic",
    "collection_authority",
    "default_external_agent_control_schema",
    "reject_ducklake_authority",
    "required_collections",
    "schema_fingerprint",
)
