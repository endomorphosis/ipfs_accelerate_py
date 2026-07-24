"""Lease-safe Profile G adapters and DuckDB coordination for daemon lanes.

Bundle supervisors are separate processes, while DuckDB permits one external
writer process at a time. Each operation therefore takes a process-shared file
lock, opens a short-lived DuckDB connection, and checks the accepted claim and
fencing token inside one transaction. An expired worker cannot publish progress
or a terminal receipt after a takeover.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

from .duckdb_state import (
    DuckDBConnection as _DuckConnection,
    DuckDBCursor as _DuckCursor,
    DuckDBRow as _DuckRow,
    exclusive_file_lock as _exclusive_file_lock,
)
from .task_identity import canonical_bundle_identity

MIN_LEASE_MS = 5_000
MAX_LEASE_MS = 300_000
PROVIDER_VERSION = "3.2.0"
MAX_PERSISTED_DEPENDENCY_REPAIRS = 256
STRUCTURAL_DEPENDENCY_REPAIR_KINDS = frozenset(
    {"missing_dependency", "dependency_cycle", "duplicate_alias", "duplicate_task"}
)
READY_BUNDLE_TASK_STATUSES = frozenset({"todo", "ready", "needed", "queued", "in_progress"})
COORDINATION_STORE_SCHEMA = "ipfs_accelerate_py.agent_supervisor.lease-coordination-duckdb@1"
COORDINATION_LOCK_TIMEOUT_SECONDS = 30.0
COORDINATION_DUCKDB_MEMORY_LIMIT = "256MB"
MAX_PERSISTED_HEARTBEATS_PER_LEASE = 8
SMALL_STORE_FULL_ARTIFACT_LIMIT = 10_000


def _coordinator_operation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Open one flock-serialized DuckDB connection for a public operation."""

    @wraps(method)
    def wrapped(self: LeaseCoordinator, *args: Any, **kwargs: Any) -> Any:
        with self._database_operation():
            return method(self, *args, **kwargs)

    return wrapped


def _duckdb_path_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


class LeaseError(RuntimeError):
    """Base error for lease protocol failures."""

    code = "G_CLAIM_CONFLICT"


class LeaseConflictError(LeaseError):
    """Raised when another non-expired claim owns the task."""


class ExecutionScopeConflictError(LeaseConflictError):
    """Raised when another task revision owns the same execution scope."""

    code = "G_EXECUTION_SCOPE_CONFLICT"


class DependencyNotReadyError(LeaseConflictError):
    """Raised when a task's prerequisite receipts have not all succeeded."""

    code = "G_DEPENDENCY_NOT_READY"

    def __init__(self, message: str, *, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


class LeaseExpiredError(LeaseError):
    """Raised when a caller's accepted lease is no longer current."""

    code = "G_LEASE_EXPIRED"


class StaleFencingTokenError(LeaseError):
    """Raised when execution uses a superseded fencing token."""

    code = "G_CLAIM_CONFLICT"


def canonical_profile_g_bytes(value: Any) -> bytes:
    """Encode canonical DAG-JSON-compatible bytes and reject floats."""

    def check(item: Any) -> None:
        if item is None or isinstance(item, str | bool | int):
            return
        if isinstance(item, float):
            raise ValueError("Profile G artifacts cannot contain floats")
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Profile G object keys must be strings")
            for child in item.values():
                check(child)
            return
        raise ValueError(f"unsupported Profile G value: {type(item).__name__}")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def profile_g_cid(value: Any) -> str:
    """Return CIDv1 DAG-JSON/sha2-256 without requiring an optional codec package."""

    digest = hashlib.sha256(canonical_profile_g_bytes(value)).digest()
    # CIDv1 + dag-json (0x0129 varint) + sha2-256 multihash.
    raw = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def _link(value: Any) -> str:
    """Create a content link for adapter inputs that are not already artifacts."""

    return profile_g_cid(value)


def _bundle_task_statuses(bundle: Mapping[str, Any]) -> set[str]:
    """Return normalized explicit member statuses from a bundle payload."""

    tasks = bundle.get("tasks")
    if not isinstance(tasks, (list, tuple)):
        return set()
    statuses: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("status") in (None, ""):
            continue
        status = str(task["status"]).strip().lower().replace("-", "_").replace(" ", "_")
        if status in {"done", "complete"}:
            status = "completed"
        elif status in {"active"}:
            status = "in_progress"
        statuses.add(status)
    return statuses


def _reopens_blocked_bundle(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Return whether authoritative discovery reopened previously blocked work."""

    previous_statuses = _bundle_task_statuses(previous)
    current_statuses = _bundle_task_statuses(current)
    return "blocked" in previous_statuses and bool(current_statuses & READY_BUNDLE_TASK_STATUSES)


def _bundle_execution_tasks(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only members owned by this bundle execution slice.

    Member aliases are receipt authorities, so a later slice must not remap
    members completed by an earlier slice. Bundles without slice metadata retain
    the legacy all-member behavior; an explicit empty slice owns no members.
    """

    tasks = bundle.get("tasks")
    members = (
        [item for item in tasks if isinstance(item, Mapping)]
        if isinstance(tasks, (list, tuple))
        else []
    )
    if (
        "execution_slice_task_cids" not in bundle
        and "execution_slice_task_ids" not in bundle
    ):
        return members

    def values(raw: Any) -> set[str]:
        if isinstance(raw, str):
            items = raw.split(",")
        elif isinstance(raw, (list, tuple, set)):
            items = raw
        elif raw in (None, ""):
            items = ()
        else:
            items = (raw,)
        return {str(item).strip() for item in items if str(item).strip()}

    selected_cids = values(bundle.get("execution_slice_task_cids"))
    selected_ids = values(bundle.get("execution_slice_task_ids"))
    return [
        item
        for item in members
        if str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
        in selected_cids
        or str(item.get("task_id") or "").strip() in selected_ids
    ]


def _dependency_task_cids(bundle: Mapping[str, Any]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Return normalized prerequisite CIDs and their bounded source provenance.

    Objective-graph payloads have existed in a few compatible shapes.  Accepting
    those shapes here keeps the lease boundary strict without coupling it to the
    planner implementation (which would also introduce an import cycle).
    """

    found: dict[str, list[dict[str, Any]]] = {}

    def add(value: Any, source: str, *, edge: Mapping[str, Any] | None = None) -> None:
        values: list[Any]
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            values = list(value)
        elif value in (None, ""):
            values = []
        else:
            values = [value]
        for item in values:
            if isinstance(item, Mapping):
                cid = str(
                    item.get("dependency_task_cid")
                    or item.get("prerequisite_task_cid")
                    or item.get("task_cid")
                    or item.get("cid")
                    or ""
                ).strip()
            else:
                cid = str(item).strip()
            if not cid:
                continue
            provenance: dict[str, Any] = {"source": source}
            if edge is not None:
                for key in ("kind", "reason", "source_path", "source_task_cid", "target_task_cid"):
                    if edge.get(key) not in (None, ""):
                        provenance[key] = edge[key]
                edge_provenance = edge.get("provenance")
                if isinstance(edge_provenance, Mapping):
                    provenance["edge_provenance"] = dict(edge_provenance)
            records = found.setdefault(cid, [])
            if provenance not in records and len(records) < 16:
                records.append(provenance)

    # Presence matters here: the objective planner emits an explicit bundle-
    # scoped projection, and an empty projection means every prerequisite is
    # internal to this execution unit.  Falling through to member or embedded
    # dependencies would reintroduce CIDs that never receive their own lease.
    has_bundle_dependency_projection = "dependency_task_cids" in bundle
    add(bundle.get("dependency_task_cids"), "bundle.dependency_task_cids")
    embedded = bundle.get("profile_g")
    if not has_bundle_dependency_projection and isinstance(embedded, Mapping):
        embedded_task = embedded.get("task")
        if isinstance(embedded_task, Mapping):
            add(embedded_task.get("dependency_task_cids"), "bundle.profile_g.task.dependency_task_cids")
    edges = bundle.get("dependency_edges")
    if not has_bundle_dependency_projection and isinstance(edges, (list, tuple)):
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                continue
            add(
                edge.get("dependency_task_cid")
                or edge.get("prerequisite_task_cid")
                or edge.get("source_task_cid"),
                f"bundle.dependency_edges[{index}]",
                edge=edge,
            )
    # Bundle execution CIDs are the receipt authority.  Member dependencies
    # remain a compatibility fallback only when no aggregate bundle edge was
    # supplied; mixing both identities would require two receipts for one
    # logical prerequisite and could permanently block otherwise-ready work.
    tasks = bundle.get("tasks")
    if not has_bundle_dependency_projection and not found and isinstance(tasks, (list, tuple)):
        for index, task in enumerate(tasks):
            if not isinstance(task, Mapping):
                continue
            for key in ("dependency_task_cids", "prerequisite_task_cids"):
                add(task.get(key), f"bundle.tasks[{index}].{key}")
    explicit_provenance = bundle.get("dependency_provenance")
    if not isinstance(explicit_provenance, Mapping) and isinstance(embedded, Mapping):
        explicit_provenance = embedded.get("dependency_provenance")
    if isinstance(explicit_provenance, Mapping):
        for raw_cid, raw_items in explicit_provenance.items():
            cid = str(raw_cid).strip()
            if cid not in found:
                continue
            items = raw_items if isinstance(raw_items, (list, tuple)) else [raw_items]
            for item in items:
                record = dict(item) if isinstance(item, Mapping) else {"detail": str(item)}
                record.setdefault("source", "bundle.dependency_provenance")
                if record not in found[cid] and len(found[cid]) < 16:
                    found[cid].append(record)
    return sorted(found), found


def _dependency_repair_evidence(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Return a bounded copy of planner-produced dependency repair records."""

    value = bundle.get("dependency_repair_evidence")
    embedded = bundle.get("profile_g")
    if not isinstance(value, (list, tuple)) and isinstance(embedded, Mapping):
        value = embedded.get("dependency_repair_evidence")
    records = [dict(item) for item in value or [] if isinstance(item, Mapping)] if isinstance(value, (list, tuple)) else []
    return records[:MAX_PERSISTED_DEPENDENCY_REPAIRS], len(records)


def adapt_goal_bundle(bundle: Mapping[str, Any], *, created_at_ms: int | None = None) -> dict[str, Any]:
    """Adapt one objective bundle payload into a canonical Goal/Subgoal/TaskSpec chain."""

    now = int(time.time() * 1000) if created_at_ms is None else int(created_at_ms)
    bundle_key = str(bundle.get("bundle_key") or "objective/general")
    correlation = str(bundle.get("correlation_id") or bundle_key)[:128]
    owner_did = str(bundle.get("owner_did") or "did:web:ipfs-accelerate.local")
    canonical_identity = canonical_bundle_identity(bundle)
    objective = {
        "bundle_key": bundle_key,
        "source_todo": str(bundle.get("source_todo") or ""),
        "task_ids": sorted(
            str(item.get("task_id"))
            for item in bundle.get("tasks", [])
            if isinstance(item, Mapping) and item.get("task_id")
        ),
    }
    objective_cid = _link(objective)
    policy_cid = str(bundle.get("policy_cid") or _link({"policy": "accelerator-daemon-lane-v1"}))
    goal = {
        "schema": "mcp++/profile-g/goal@1",
        "created_at_ms": now,
        "parents": [],
        "correlation_id": correlation,
        "owner_did": owner_did,
        "objective_cid": objective_cid,
        "policy_cid": policy_cid,
        "parent_goal_cids": [],
        "labels": sorted({"accelerator", "daemon-lane"}),
    }
    goal_cid = profile_g_cid(goal)
    subgoal = {
        "schema": "mcp++/profile-g/subgoal@1",
        "created_at_ms": now,
        "parents": [goal_cid],
        "correlation_id": correlation,
        "goal_cid": goal_cid,
        "parent_subgoal_cid": None,
        "objective_cid": objective_cid,
        "decomposition_method": "objective-bundle-v1",
        "decomposer_cid": _link({"adapter": "ipfs_accelerate_py.agent_supervisor"}),
        "selection_cid": None,
    }
    subgoal_cid = profile_g_cid(subgoal)
    template_cid = _link({"tool": "codex.todo_bundle", "bundle_key": bundle_key})
    plan = {
        "schema": "mcp++/profile-g/plan-branch@1",
        "created_at_ms": now,
        "parents": [subgoal_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "candidate_input_cids": [_link(dict(bundle))],
        "task_template_cids": [template_cid],
        "evaluator_cid": _link({"evaluator": "objective-bundle-priority-v1"}),
        "score_millionths": 1_000_000,
        "explanation_cid": _link({"reason": "selected generated objective bundle"}),
    }
    plan_cid = profile_g_cid(plan)
    selection = {
        "schema": "mcp++/profile-g/plan-selection@1",
        "created_at_ms": now,
        "parents": [plan_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "plan_branch_cid": plan_cid,
        "selector_did": owner_did,
        "proof_cid": str(bundle.get("proof_cid") or _link({"proof": bundle_key})),
        "policy_decision_cid": str(bundle.get("policy_decision_cid") or _link({"decision": "allow"})),
        "reason_cid": _link({"reason": "bundle emitted by accepted objective graph"}),
    }
    selection_cid = profile_g_cid(selection)
    # Subgoal.selection_cid remains null because making both immutable objects
    # point at one another would create an impossible content-addressed cycle.
    # PlanSelection is the authoritative selected-branch record.
    dependency_task_cids, dependency_provenance = _dependency_task_cids(bundle)
    dependency_repairs, dependency_repair_count = _dependency_repair_evidence(bundle)
    task = {
        "schema": "mcp++/profile-g/task@1",
        "created_at_ms": now,
        "parents": [selection_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "plan_branch_cid": plan_cid,
        "selection_cid": selection_cid,
        "interface_cid": _link({"interface": "codex.todo_bundle@1"}),
        "input_cid": _link(dict(bundle)),
        "tool": "codex.todo_bundle",
        "dependency_task_cids": dependency_task_cids,
        "idempotency_key": canonical_identity.semantic_fingerprint[:32],
        "canonical_task_key": canonical_identity.canonical_task_key,
        "canonical_task_cid": canonical_identity.canonical_task_cid,
        "resource_class": str(bundle.get("resource_class") or "cpu-small"),
        "deadline_ms": int(bundle.get("deadline_ms") or now + 86_400_000),
        "expected_value_millionths": int(bundle.get("expected_value_millionths") or 500_000),
        "max_attempts": int(bundle.get("max_attempts") or 3),
        "execution_mode": "idempotent",
    }
    task_spec_cid = profile_g_cid(task)
    artifacts = {profile_g_cid(item): item for item in (goal, subgoal, plan, selection, task)}
    return {
        "goal": goal,
        "goal_cid": goal_cid,
        "subgoal": subgoal,
        "subgoal_cid": subgoal_cid,
        "plan_branch": plan,
        "plan_branch_cid": plan_cid,
        "selection": selection,
        "selection_cid": selection_cid,
        "task": task,
        "task_cid": task_spec_cid,
        "task_spec_cid": task_spec_cid,
        "canonical_task_key": canonical_identity.canonical_task_key,
        "canonical_task_cid": canonical_identity.canonical_task_cid,
        "dependency_provenance": dependency_provenance,
        "dependency_repair_evidence": dependency_repairs,
        "dependency_repair_evidence_count": dependency_repair_count,
        "artifacts": artifacts,
    }


@dataclass(frozen=True)
class LeaseGrant:
    task_cid: str
    goal_cid: str
    subgoal_cid: str
    claim_cid: str
    resolution_cid: str
    claimant_did: str
    logical_epoch: int
    fencing_token: int
    lease_expires_at_ms: int
    attempt: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskLeaseState:
    """Authoritative scheduler projection for one registered task.

    ``state`` is deliberately scheduler-oriented: expired and voluntarily
    released lease records project as ``ready``.  The durable lease outcome is
    retained in ``lease_state`` and ``release_reason`` for diagnostics.
    """

    task_cid: str
    goal_cid: str
    subgoal_cid: str
    task_id: str
    bundle: dict[str, Any]
    state: str
    lease_state: str | None
    claim_cid: str | None
    resolution_cid: str | None
    claimant_did: str | None
    logical_epoch: int
    fencing_token: int
    lease_expires_at_ms: int | None
    attempt: int
    max_attempts: int
    release_reason: str | None
    retry_not_before_ms: int
    registered_at_ms: int
    updated_at_ms: int

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def to_dict(self) -> dict[str, Any]:
        payload = {definition.name: getattr(self, definition.name) for definition in fields(self)}
        payload["bundle"] = dict(self.bundle)
        return payload


class LeaseCoordinator:
    """Durable accepted-lease registry for independent daemon processes."""

    def __init__(self, path: str | Path, *, clock_ms: Callable[[], int] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            with self.path.open("rb") as stream:
                if stream.read(16) == b"SQLite format 3\0":
                    raise ValueError(
                        f"legacy SQLite coordination store requires migration: {self.path}"
                    )
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.RLock()
        self._operation_state = threading.local()
        self._connection: _DuckConnection | None = None
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        with self._database_operation():
            self._init_schema()

    @contextmanager
    def _database_operation(self) -> Iterator[None]:
        """Serialize short-lived DuckDB connections across lane processes."""

        with self._lock:
            depth = int(getattr(self._operation_state, "depth", 0))
            if depth:
                self._operation_state.depth = depth + 1
                try:
                    yield
                finally:
                    self._operation_state.depth = depth
                return

            with _exclusive_file_lock(
                self._lock_path,
                timeout_seconds=COORDINATION_LOCK_TIMEOUT_SECONDS,
            ):
                try:
                    import duckdb
                except ImportError as exc:
                    raise RuntimeError(
                        "DuckDB is required for lease coordination"
                    ) from exc
                duckdb_connection = duckdb.connect(str(self.path))
                duckdb_connection.execute("SET threads=1")
                duckdb_connection.execute(
                    f"SET memory_limit='{COORDINATION_DUCKDB_MEMORY_LIMIT}'"
                )
                self._connection = _DuckConnection.wrap(
                    duckdb_connection,
                    transaction_on_context=True,
                )
                self._operation_state.depth = 1
                try:
                    yield
                finally:
                    self._operation_state.depth = 0
                    connection = self._connection
                    self._connection = None
                    if connection is not None:
                        connection.close()

    def _init_schema(self) -> None:
        assert self._connection is not None
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                  cid TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
                  created_at_ms BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                  task_cid TEXT PRIMARY KEY, goal_cid TEXT NOT NULL, subgoal_cid TEXT NOT NULL,
                  task_id TEXT NOT NULL, bundle_json TEXT NOT NULL,
                  registered_at_ms BIGINT NOT NULL DEFAULT 0,
                  updated_at_ms BIGINT NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS task_aliases (
                  alias_task_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_dependencies (
                  task_cid TEXT NOT NULL, dependency_task_cid TEXT NOT NULL,
                  provenance_json TEXT NOT NULL,
                  PRIMARY KEY(task_cid, dependency_task_cid)
                );
                CREATE TABLE IF NOT EXISTS task_dependency_repairs (
                  task_cid TEXT NOT NULL, repair_index BIGINT NOT NULL,
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY(task_cid, repair_index)
                );
                CREATE TABLE IF NOT EXISTS task_dependency_repair_state (
                  task_cid TEXT PRIMARY KEY, source_count BIGINT NOT NULL,
                  stored_count BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                  task_cid TEXT PRIMARY KEY, claim_cid TEXT NOT NULL, resolution_cid TEXT NOT NULL,
                  claimant_did TEXT NOT NULL, logical_epoch BIGINT NOT NULL,
                  fencing_token BIGINT NOT NULL, expires_at_ms BIGINT NOT NULL,
                  attempt BIGINT NOT NULL, state TEXT NOT NULL, started_at_ms BIGINT NOT NULL,
                  release_reason TEXT,
                  retry_not_before_ms BIGINT NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS token_history (
                  task_cid TEXT NOT NULL, fencing_token BIGINT NOT NULL,
                  PRIMARY KEY(task_cid, fencing_token)
                );
                CREATE TABLE IF NOT EXISTS heartbeats (
                  heartbeat_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL, claimant_did TEXT NOT NULL,
                  fencing_token BIGINT NOT NULL, observed_at_ms BIGINT NOT NULL,
                  expires_at_ms BIGINT NOT NULL, capacity_millionths BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                  receipt_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL, goal_cid TEXT NOT NULL,
                  subgoal_cid TEXT NOT NULL, claim_cid TEXT NOT NULL, fencing_token BIGINT NOT NULL,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coordination_metadata (
                  metadata_key TEXT PRIMARY KEY, value_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_dependencies_dependency_idx
                  ON task_dependencies(dependency_task_cid);
                CREATE INDEX IF NOT EXISTS receipts_task_order_idx
                  ON receipts(task_cid, receipt_cid);
                """
            )
            # REF-036 databases remain valid after migration. Keep schema
            # evolution additive so existing DuckDB stores upgrade in place.
            task_columns = {
                str(row["name"]) for row in self._connection.execute("PRAGMA table_info(tasks)")
            }
            if "registered_at_ms" not in task_columns:
                self._connection.execute(
                    "ALTER TABLE tasks ADD COLUMN registered_at_ms BIGINT NOT NULL DEFAULT 0"
                )
            if "updated_at_ms" not in task_columns:
                self._connection.execute(
                    "ALTER TABLE tasks ADD COLUMN updated_at_ms BIGINT NOT NULL DEFAULT 0"
                )
            lease_columns = {
                str(row["name"]) for row in self._connection.execute("PRAGMA table_info(leases)")
            }
            if "release_reason" not in lease_columns:
                self._connection.execute("ALTER TABLE leases ADD COLUMN release_reason TEXT")
            if "retry_not_before_ms" not in lease_columns:
                self._connection.execute(
                    "ALTER TABLE leases ADD COLUMN retry_not_before_ms BIGINT NOT NULL DEFAULT 0"
                )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_leases_scheduler_state "
                "ON leases(state, expires_at_ms, retry_not_before_ms)"
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO coordination_metadata VALUES(?,?)",
                (
                    "store",
                    json.dumps(
                        {
                            "schema": COORDINATION_STORE_SCHEMA,
                            "backend": "duckdb",
                        },
                        sort_keys=True,
                    ),
                ),
            )

    def close(self) -> None:
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None

    def compact(self) -> dict[str, Any]:
        """Atomically rewrite live rows into a compact DuckDB store."""

        with self._lock:
            if int(getattr(self._operation_state, "depth", 0)):
                raise RuntimeError(
                    "coordination compaction cannot run inside a database operation"
                )
            with _exclusive_file_lock(
                self._lock_path,
                timeout_seconds=COORDINATION_LOCK_TIMEOUT_SECONDS,
            ):
                source_bytes = self.path.stat().st_size
                temporary = self.path.with_name(
                    f".{self.path.name}.compact-{os.getpid()}-"
                    f"{threading.get_ident()}.tmp"
                )
                temporary.unlink(missing_ok=True)
                Path(f"{temporary}.wal").unlink(missing_ok=True)
                try:
                    try:
                        import duckdb
                    except ImportError as exc:
                        raise RuntimeError(
                            "DuckDB is required for lease coordination"
                        ) from exc
                    connection = duckdb.connect(":memory:")
                    try:
                        connection.execute("SET threads=1")
                        connection.execute(
                            f"SET memory_limit='{COORDINATION_DUCKDB_MEMORY_LIMIT}'"
                        )
                        connection.execute(
                            "ATTACH "
                            f"{_duckdb_path_literal(self.path)} "
                            "AS source_store (READ_ONLY)"
                        )
                        connection.execute(
                            "ATTACH "
                            f"{_duckdb_path_literal(temporary)} "
                            "AS target_store"
                        )
                        connection.execute(
                            "COPY FROM DATABASE source_store TO target_store"
                        )
                        connection.execute("DETACH target_store")
                        connection.execute("DETACH source_store")
                    finally:
                        connection.close()
                    compacted = duckdb.connect(str(temporary))
                    try:
                        compacted.execute("SET threads=1")
                        compacted.execute(
                            f"SET memory_limit='{COORDINATION_DUCKDB_MEMORY_LIMIT}'"
                        )
                        compacted.execute(
                            "INSERT OR REPLACE INTO coordination_metadata "
                            "VALUES(?,?)",
                            (
                                "last_compaction",
                                json.dumps(
                                    {
                                        "compacted_at_ms": self._clock_ms(),
                                        "source_bytes": source_bytes,
                                    },
                                    sort_keys=True,
                                ),
                            ),
                        )
                        compacted.execute("CHECKPOINT")
                    finally:
                        compacted.close()
                    target_bytes = temporary.stat().st_size
                    os.replace(temporary, self.path)
                finally:
                    temporary.unlink(missing_ok=True)
                    Path(f"{temporary}.wal").unlink(missing_ok=True)
                return {
                    "source_bytes": source_bytes,
                    "target_bytes": target_bytes,
                    "reclaimed_bytes": max(0, source_bytes - target_bytes),
                    "compacted_at_ms": self._clock_ms(),
                }

    def __enter__(self) -> LeaseCoordinator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _put_artifact(self, connection: _DuckConnection, kind: str, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        cid = profile_g_cid(body)
        connection.execute(
            "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
            (cid, kind, canonical_profile_g_bytes(body).decode("utf-8"), int(body.get("created_at_ms") or self._clock_ms())),
        )
        return cid

    @_coordinator_operation
    def register_bundle(self, bundle: Mapping[str, Any], *, created_at_ms: int | None = None) -> dict[str, Any]:
        embedded = bundle.get("profile_g")
        adapted = dict(embedded) if isinstance(embedded, Mapping) and embedded.get("task_cid") else adapt_goal_bundle(bundle, created_at_ms=created_at_ms)
        canonical_identity = canonical_bundle_identity(bundle)
        canonical_task_cid = str(adapted.get("canonical_task_cid") or canonical_identity.canonical_task_cid)
        task_spec_cid = str(adapted.get("task_spec_cid") or adapted.get("task_cid") or "")
        adapted["canonical_task_key"] = str(
            adapted.get("canonical_task_key") or canonical_identity.canonical_task_key
        )
        adapted["canonical_task_cid"] = canonical_task_cid
        adapted["task_spec_cid"] = task_spec_cid
        dependency_task_cids, dependency_provenance = _dependency_task_cids(bundle)
        adapted_task = adapted.get("task")
        if "dependency_task_cids" not in bundle and isinstance(adapted_task, Mapping):
            embedded_cids, _ = _dependency_task_cids(
                {"dependency_task_cids": adapted_task.get("dependency_task_cids")}
            )
            for cid in embedded_cids:
                if cid not in dependency_task_cids:
                    dependency_task_cids.append(cid)
                    dependency_provenance.setdefault(cid, []).append(
                        {"source": "profile_g.task.dependency_task_cids"}
                    )
        dependency_task_cids.sort()
        adapted["dependency_task_cids"] = dependency_task_cids
        adapted["dependency_provenance"] = dependency_provenance
        dependency_repairs, dependency_repair_count = _dependency_repair_evidence(bundle)
        adapted["dependency_repair_evidence"] = dependency_repairs
        adapted["dependency_repair_evidence_count"] = dependency_repair_count
        # Coordination is keyed by semantic execution identity. The immutable
        # Profile-G TaskSpec CID remains available separately for provenance.
        adapted["task_cid"] = canonical_task_cid
        registered_at = self._clock_ms() if created_at_ms is None else int(created_at_ms)
        attempt_budget_reset = False
        task_id = ",".join(
            str(item.get("task_id"))
            for item in bundle.get("tasks", [])
            if isinstance(item, Mapping)
        ) or str(bundle.get("bundle_key") or adapted["task_cid"])
        bundle_json = json.dumps(dict(bundle), sort_keys=True)
        all_member_aliases = {
            str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
            for item in (
                bundle.get("tasks", [])
                if isinstance(bundle.get("tasks"), (list, tuple))
                else []
            )
            if isinstance(item, Mapping)
            and str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
        }
        execution_member_aliases = {
            str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
            for item in _bundle_execution_tasks(bundle)
            if str(item.get("canonical_task_cid") or item.get("task_cid") or "").strip()
        }
        aliases = {canonical_task_cid, task_spec_cid, *execution_member_aliases}

        def sync_aliases() -> None:
            # Repair databases written before slice-scoped identities without
            # deleting an alias now owned by a different, completed slice.
            for alias in sorted(all_member_aliases - execution_member_aliases):
                self._connection.execute(
                    "DELETE FROM task_aliases WHERE alias_task_cid=? AND task_cid=?",
                    (alias, canonical_task_cid),
                )
            for alias in sorted(aliases):
                if alias:
                    self._connection.execute(
                        "INSERT OR REPLACE INTO task_aliases VALUES(?,?)",
                        (alias, canonical_task_cid),
                    )

        with self._lock, self._connection:
            previous_task = self._connection.execute(
                "SELECT bundle_json FROM tasks WHERE task_cid=?",
                (canonical_task_cid,),
            ).fetchone()
            registration_complete = self._connection.execute(
                "SELECT 1 FROM task_dependency_repair_state WHERE task_cid=?",
                (canonical_task_cid,),
            ).fetchone()
            if (
                previous_task is not None
                and registration_complete is not None
                and str(previous_task["bundle_json"]) == bundle_json
            ):
                sync_aliases()
                adapted["attempt_budget_reset"] = False
                return adapted

            for cid, artifact in adapted["artifacts"].items():
                self._connection.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                    (cid, str(artifact["schema"]), canonical_profile_g_bytes(artifact).decode("utf-8"), artifact["created_at_ms"]),
                )
            previous_bundle = (
                json.loads(str(previous_task["bundle_json"]))
                if previous_task is not None
                else {}
            )
            reopened = isinstance(previous_bundle, Mapping) and _reopens_blocked_bundle(
                previous_bundle,
                bundle,
            )
            # Keep the original immutable Goal/Subgoal provenance while
            # refreshing mutable discovery metadata on every scheduler poll.
            self._connection.execute(
                """INSERT INTO tasks(
                       task_cid, goal_cid, subgoal_cid, task_id, bundle_json,
                       registered_at_ms, updated_at_ms
                   ) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(task_cid) DO UPDATE SET
                     task_id=excluded.task_id,
                     bundle_json=excluded.bundle_json,
                     updated_at_ms=excluded.updated_at_ms""",
                (
                    canonical_task_cid,
                    adapted["goal_cid"],
                    adapted["subgoal_cid"],
                    task_id,
                    bundle_json,
                    registered_at,
                    registered_at,
                ),
            )
            if reopened:
                reset = self._connection.execute(
                    """UPDATE leases
                       SET state='released', attempt=0,
                           release_reason='requeued:bundle_status_reopened',
                           retry_not_before_ms=0
                       WHERE task_cid=?
                         AND state IN ('released','expired')
                         AND release_reason LIKE 'receipt:%:blocked'""",
                    (canonical_task_cid,),
                )
                attempt_budget_reset = reset.rowcount > 0
            # A dependency DAG names the immutable work-item CIDs while the
            # coordination lease is bundle-scoped.  Aliases bridge those two
            # identities so the slice's successful receipt unlocks only the
            # members that it executed. Earlier slice aliases must remain
            # mapped to their completed receipt authorities.
            sync_aliases()
            self._connection.execute("DELETE FROM task_dependencies WHERE task_cid=?", (canonical_task_cid,))
            for dependency_task_cid in dependency_task_cids:
                self._connection.execute(
                    "INSERT INTO task_dependencies VALUES(?,?,?)",
                    (
                        canonical_task_cid,
                        dependency_task_cid,
                        json.dumps(dependency_provenance.get(dependency_task_cid, []), sort_keys=True),
                    ),
                )
            self._connection.execute("DELETE FROM task_dependency_repairs WHERE task_cid=?", (canonical_task_cid,))
            for index, repair in enumerate(dependency_repairs):
                self._connection.execute(
                    "INSERT INTO task_dependency_repairs VALUES(?,?,?)",
                    (canonical_task_cid, index, json.dumps(repair, sort_keys=True)),
                )
            self._connection.execute(
                "INSERT OR REPLACE INTO task_dependency_repair_state VALUES(?,?,?)",
                (canonical_task_cid, dependency_repair_count, len(dependency_repairs)),
            )
        adapted["attempt_budget_reset"] = attempt_budget_reset
        return adapted

    @_coordinator_operation
    def register_bundles(
        self,
        bundles: Iterable[Mapping[str, Any]],
        *,
        created_at_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Register a discovery batch idempotently.

        This convenience API intentionally delegates one bundle at a time so
        embedded Profile G artifacts and canonical identity follow exactly the
        same compatibility path as :meth:`register_bundle`.
        """

        with self._lock, self._connection:
            return [
                self.register_bundle(bundle, created_at_ms=created_at_ms)
                for bundle in bundles
            ]

    @_coordinator_operation
    def requeue_exhausted_blocked(self, task_cid: str, *, reason: str) -> bool:
        """Reset a blocked attempt budget after authoritative work is reopened.

        This operation is deliberately narrow: it cannot disturb accepted or
        completed leases, and only resets an exhausted lease whose last
        terminal receipt classified the work as blocked.
        """

        normalized_reason = str(reason or "authoritative_source_reopened").strip().replace(" ", "_")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                resolved_task_cid = self._resolve_task_cid(connection, task_cid)
                if resolved_task_cid is None:
                    connection.commit()
                    return False
                row = connection.execute(
                    """SELECT t.bundle_json, l.state, l.attempt, l.release_reason
                       FROM tasks AS t
                       JOIN leases AS l ON l.task_cid=t.task_cid
                       WHERE t.task_cid=?""",
                    (resolved_task_cid,),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return False
                exhausted = int(row["attempt"] or 0) >= self._max_attempts(row)
                blocked_receipt = str(row["release_reason"] or "").startswith("receipt:") and str(
                    row["release_reason"] or ""
                ).endswith(":blocked")
                if row["state"] not in {"released", "expired"} or not exhausted or not blocked_receipt:
                    connection.commit()
                    return False
                connection.execute(
                    """UPDATE leases
                       SET state='released', attempt=0, retry_not_before_ms=0,
                           release_reason=?
                       WHERE task_cid=?""",
                    (f"requeued:{normalized_reason}"[:256], resolved_task_cid),
                )
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _resolve_task_cid(connection: _DuckConnection, task_cid: str) -> str | None:
        row = connection.execute(
            "SELECT task_cid FROM task_aliases WHERE alias_task_cid=?",
            (task_cid,),
        ).fetchone()
        if row is not None:
            return str(row[0])
        row = connection.execute("SELECT task_cid FROM tasks WHERE task_cid=?", (task_cid,)).fetchone()
        return str(row[0]) if row is not None else None

    def _dependency_cycles(
        self,
        connection: _DuckConnection,
        task_cid: str,
        *,
        max_nodes: int,
        max_cycles: int,
    ) -> tuple[list[list[str]], bool]:
        """Find a bounded set of dependency cycles reachable from ``task_cid``."""

        cycles: list[list[str]] = []
        visited: set[str] = set()
        active: list[str] = []
        active_set: set[str] = set()
        truncated = False

        def visit(current: str) -> None:
            nonlocal truncated
            if truncated or len(cycles) >= max_cycles:
                truncated = True
                return
            if len(visited) >= max_nodes and current not in visited:
                truncated = True
                return
            if current in active_set:
                start = active.index(current)
                cycle = active[start:] + [current]
                if cycle not in cycles:
                    cycles.append(cycle)
                return
            if current in visited:
                return
            visited.add(current)
            active.append(current)
            active_set.add(current)
            rows = connection.execute(
                "SELECT dependency_task_cid FROM task_dependencies WHERE task_cid=? ORDER BY dependency_task_cid",
                (current,),
            ).fetchall()
            for row in rows:
                resolved = self._resolve_task_cid(connection, str(row[0]))
                if resolved is not None:
                    visit(resolved)
            active.pop()
            active_set.remove(current)

        visit(task_cid)
        return cycles, truncated

    def _claimability(
        self,
        connection: _DuckConnection,
        task_cid: str,
        *,
        max_evidence: int,
    ) -> dict[str, Any]:
        resolved_task_cid = self._resolve_task_cid(connection, task_cid)
        if resolved_task_cid is None:
            raise KeyError(f"unknown task CID: {task_cid}")
        rows = connection.execute(
            "SELECT dependency_task_cid, provenance_json FROM task_dependencies "
            "WHERE task_cid=? ORDER BY dependency_task_cid",
            (resolved_task_cid,),
        ).fetchall()
        dependencies = [str(row["dependency_task_cid"]) for row in rows]
        provenance = {
            str(row["dependency_task_cid"]): json.loads(row["provenance_json"])
            for row in rows
        }
        satisfied: list[str] = []
        blocked: list[str] = []
        missing: list[str] = []
        evidence: list[dict[str, Any]] = []
        evidence_truncated = False

        def add_evidence(item: dict[str, Any]) -> None:
            nonlocal evidence_truncated
            if len(evidence) < max_evidence:
                evidence.append(item)
            else:
                evidence_truncated = True

        for dependency_cid in dependencies:
            receipt_task_cid = self._resolve_task_cid(connection, dependency_cid)
            if receipt_task_cid is None:
                missing.append(dependency_cid)
                blocked.append(dependency_cid)
                add_evidence(
                    {
                        "kind": "missing_dependency",
                        "dependency_task_cid": dependency_cid,
                        "provenance": provenance.get(dependency_cid, []),
                        "repair": "register the prerequisite task or repair the dependency edge",
                    }
                )
                continue
            receipt = connection.execute(
                "SELECT receipt_cid, payload_json FROM receipts WHERE task_cid=? ORDER BY rowid DESC LIMIT 1",
                (receipt_task_cid,),
            ).fetchone()
            receipt_payload = json.loads(receipt["payload_json"]) if receipt is not None else {}
            latest_status = str(receipt_payload.get("status") or "missing")
            if latest_status == "succeeded":
                satisfied.append(dependency_cid)
                continue
            blocked.append(dependency_cid)
            add_evidence(
                {
                    "kind": "prerequisite_receipt_not_succeeded",
                    "dependency_task_cid": dependency_cid,
                    "resolved_task_cid": receipt_task_cid,
                    "latest_receipt_cid": str(receipt["receipt_cid"]) if receipt is not None else None,
                    "latest_status": latest_status,
                    "provenance": provenance.get(dependency_cid, []),
                    "repair": "complete and merge the prerequisite successfully before claiming this task",
                }
            )

        cycles, cycle_search_truncated = self._dependency_cycles(
            connection,
            resolved_task_cid,
            max_nodes=max(64, max_evidence * 8),
            max_cycles=max_evidence,
        )
        for cycle in cycles:
            add_evidence(
                {
                    "kind": "dependency_cycle",
                    "cycle_task_cids": cycle,
                    "repair": "remove or redirect at least one cyclic prerequisite edge",
                }
            )
        planner_repairs: list[dict[str, Any]] = []
        repair_rows = connection.execute(
            "SELECT payload_json FROM task_dependency_repairs WHERE task_cid=? ORDER BY repair_index",
            (resolved_task_cid,),
        ).fetchall()
        for row in repair_rows:
            repair = json.loads(row["payload_json"])
            kind = str(repair.get("kind") or "").strip().lower().replace("-", "_")
            if kind not in STRUCTURAL_DEPENDENCY_REPAIR_KINDS:
                continue
            repair["kind"] = kind
            repair.setdefault(
                "repair",
                "repair the planner dependency metadata and regenerate the bundle schedule",
            )
            planner_repairs.append(repair)
            add_evidence(repair)
        repair_state = connection.execute(
            "SELECT source_count, stored_count FROM task_dependency_repair_state WHERE task_cid=?",
            (resolved_task_cid,),
        ).fetchone()
        persisted_repairs_truncated = bool(
            repair_state is not None and int(repair_state["source_count"]) > int(repair_state["stored_count"])
        )
        return {
            "schema": "ipfs_accelerate_py/dependency-claimability@1",
            "task_cid": resolved_task_cid,
            "claimable": not blocked and not cycles and not planner_repairs,
            "dependency_task_cids": dependencies,
            "satisfied_dependency_task_cids": satisfied,
            "blocked_dependency_task_cids": blocked,
            "missing_dependency_task_cids": missing,
            "dependency_cycles": cycles,
            "structural_dependency_repairs": planner_repairs[:max_evidence],
            "repair_evidence": evidence,
            "evidence_truncated": evidence_truncated or cycle_search_truncated or persisted_repairs_truncated,
            "planner_repair_evidence_count": int(repair_state["source_count"]) if repair_state is not None else 0,
        }

    @_coordinator_operation
    def claimability(self, task_cid: str, *, max_evidence: int = 32) -> dict[str, Any]:
        """Explain whether all prerequisite tasks have successful receipts.

        Evidence is deliberately bounded so malformed or cyclic planner input
        becomes actionable repair data instead of an unbounded walk or deadlock.
        """

        limit = int(max_evidence)
        if not 1 <= limit <= 256:
            raise ValueError("max_evidence must be in [1, 256]")
        with self._lock:
            return self._claimability(self._connection, task_cid, max_evidence=limit)

    def _expire(self, connection: _DuckConnection, task_cid: str, now: int) -> None:
        row = connection.execute(
            "SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms<=?",
            (task_cid, now),
        ).fetchone()
        if row is None:
            return
        resolution = self._resolution_payload(row, outcome="expired", now=now)
        resolution_cid = self._put_artifact(connection, "ClaimResolution", resolution)
        connection.execute(
            """UPDATE leases
               SET state='expired', resolution_cid=?, release_reason='expired'
               WHERE task_cid=? AND claim_cid=? AND fencing_token=?""",
            (resolution_cid, task_cid, row["claim_cid"], row["fencing_token"]),
        )

    @staticmethod
    def _max_attempts(task: _DuckRow | Mapping[str, Any]) -> int:
        bundle = json.loads(task["bundle_json"])
        return max(1, int(bundle.get("max_attempts") or 3))

    @staticmethod
    def _execution_scope(task: _DuckRow) -> str:
        """Return the stable bundle scope that must have at most one live lease."""

        bundle = json.loads(task["bundle_json"])
        return str(bundle.get("bundle_key") or "").strip()

    def _active_execution_scope_conflict(
        self,
        connection: _DuckConnection,
        task: _DuckRow,
        *,
        now: int,
    ) -> _DuckRow | None:
        """Find a live lease for another revision of this task's bundle."""

        execution_scope = self._execution_scope(task)
        if not execution_scope:
            return None
        rows = connection.execute(
            """SELECT l.*, t.bundle_json, t.task_id
               FROM leases AS l
               JOIN tasks AS t ON t.task_cid=l.task_cid
               WHERE l.task_cid<>?
                 AND l.state='accepted'
                 AND l.expires_at_ms>?""",
            (str(task["task_cid"]), now),
        ).fetchall()
        return next(
            (row for row in rows if self._execution_scope(row) == execution_scope),
            None,
        )

    def _task_projection(
        self,
        task: _DuckRow,
        lease: _DuckRow | None,
        *,
        now: int,
    ) -> TaskLeaseState:
        bundle = json.loads(task["bundle_json"])
        lease_state = str(lease["state"]) if lease is not None else None
        max_attempts = self._max_attempts(task)
        attempt = int(lease["attempt"] or 0) if lease is not None else 0
        retry_not_before = int(lease["retry_not_before_ms"] or 0) if lease is not None else 0
        if lease_state == "accepted" and int(lease["expires_at_ms"]) > now:
            state = "accepted"
        elif lease_state == "completed":
            state = "completed"
        elif attempt >= max_attempts or retry_not_before > now:
            state = "blocked"
        else:
            state = "ready"
        return TaskLeaseState(
            task_cid=str(task["task_cid"]),
            goal_cid=str(task["goal_cid"]),
            subgoal_cid=str(task["subgoal_cid"]),
            task_id=str(task["task_id"]),
            bundle=bundle,
            state=state,
            lease_state=lease_state,
            claim_cid=str(lease["claim_cid"]) if lease is not None else None,
            resolution_cid=str(lease["resolution_cid"]) if lease is not None else None,
            claimant_did=str(lease["claimant_did"]) if lease is not None and state == "accepted" else None,
            logical_epoch=int(lease["logical_epoch"] or 0) if lease is not None else 0,
            fencing_token=int(lease["fencing_token"] or 0) if lease is not None else 0,
            lease_expires_at_ms=(
                int(lease["expires_at_ms"]) if lease is not None and state == "accepted" else None
            ),
            attempt=attempt,
            max_attempts=max_attempts,
            release_reason=(
                str(lease["release_reason"]) if lease is not None and lease["release_reason"] else None
            ),
            retry_not_before_ms=retry_not_before,
            registered_at_ms=int(task["registered_at_ms"] or 0),
            updated_at_ms=int(task["updated_at_ms"] or 0),
        )

    @staticmethod
    def _projection_dict(state: TaskLeaseState) -> dict[str, Any]:
        result = state.to_dict()
        result["bundle_key"] = str(state.bundle.get("bundle_key") or state.task_id)
        # Common spelling used by the lane manifest.
        result["expires_at_ms"] = state.lease_expires_at_ms
        return result

    def _projection_with_claimability(
        self,
        connection: _DuckConnection,
        state: TaskLeaseState,
        *,
        max_evidence: int,
    ) -> dict[str, Any]:
        """Attach bounded dependency evidence to a scheduler projection."""

        result = self._projection_dict(state)
        readiness = self._claimability(
            connection,
            state.task_cid,
            max_evidence=max_evidence,
        )
        result.update(
            {
                "claimable": bool(readiness["claimable"]),
                "dependency_task_cids": list(readiness["dependency_task_cids"]),
                "satisfied_dependency_task_cids": list(
                    readiness["satisfied_dependency_task_cids"]
                ),
                "blocked_dependency_task_cids": list(
                    readiness["blocked_dependency_task_cids"]
                ),
                "blocking_task_cids": list(readiness["blocked_dependency_task_cids"]),
                "missing_dependency_task_cids": list(
                    readiness["missing_dependency_task_cids"]
                ),
                "dependency_cycles": list(readiness["dependency_cycles"]),
                "dependency_repair_evidence": list(readiness["repair_evidence"]),
                "claimability_evidence_truncated": bool(
                    readiness["evidence_truncated"]
                ),
            }
        )
        if result["state"] == "ready" and not result["claimable"]:
            result["state"] = "blocked"
            result["blocked_reason"] = "dependency_not_ready"
        return result

    @_coordinator_operation
    def list_tasks(
        self,
        *,
        task_cids: Iterable[str] | None = None,
        now_ms: int | None = None,
        include_claimability: bool = False,
        max_claimability_evidence: int = 8,
    ) -> list[dict[str, Any]]:
        """Return a consistent live projection, optionally scoped to current tasks."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        evidence_limit = int(max_claimability_evidence)
        if not 1 <= evidence_limit <= 256:
            raise ValueError("max_claimability_evidence must be in [1, 256]")
        selected = None if task_cids is None else sorted({str(item) for item in task_cids})
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                expired = connection.execute(
                    "SELECT task_cid FROM leases WHERE state='accepted' AND expires_at_ms<=?",
                    (now,),
                ).fetchall()
                for row in expired:
                    self._expire(connection, str(row["task_cid"]), now)
                query = """SELECT t.*, l.claim_cid, l.resolution_cid, l.claimant_did,
                                  l.logical_epoch, l.fencing_token, l.expires_at_ms,
                                  l.attempt, l.state, l.started_at_ms,
                                  l.release_reason, l.retry_not_before_ms
                           FROM tasks AS t
                           LEFT JOIN leases AS l ON l.task_cid=t.task_cid"""
                if selected is None:
                    rows = connection.execute(
                        f"{query} ORDER BY t.registered_at_ms, t.task_cid"
                    ).fetchall()
                else:
                    rows = []
                    for offset in range(0, len(selected), 500):
                        chunk = selected[offset : offset + 500]
                        placeholders = ",".join("?" for _item in chunk)
                        rows.extend(
                            connection.execute(
                                f"{query} WHERE t.task_cid IN ({placeholders})",
                                chunk,
                            ).fetchall()
                        )
                    rows.sort(key=lambda row: (int(row["registered_at_ms"]), str(row["task_cid"])))
                states = [
                    self._task_projection(
                        row,
                        row if row["state"] is not None else None,
                        now=now,
                    )
                    for row in rows
                ]
                result = [
                    (
                        self._projection_with_claimability(
                            connection,
                            state,
                            max_evidence=evidence_limit,
                        )
                        if include_claimability
                        else self._projection_dict(state)
                    )
                    for state in states
                ]
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def task_state(
        self,
        task_cid: str,
        *,
        now_ms: int | None = None,
        include_claimability: bool = False,
        max_claimability_evidence: int = 8,
    ) -> dict[str, Any] | None:
        """Return the live scheduler projection for ``task_cid``."""

        now = self._clock_ms() if now_ms is None else int(now_ms)
        evidence_limit = int(max_claimability_evidence)
        if not 1 <= evidence_limit <= 256:
            raise ValueError("max_claimability_evidence must be in [1, 256]")
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT * FROM tasks WHERE task_cid=?", (task_cid,)
                ).fetchone()
                if task is None:
                    connection.commit()
                    return None
                self._expire(connection, task_cid, now)
                lease = connection.execute(
                    "SELECT * FROM leases WHERE task_cid=?", (task_cid,)
                ).fetchone()
                state = self._task_projection(task, lease, now=now)
                result = (
                    self._projection_with_claimability(
                        connection,
                        state,
                        max_evidence=evidence_limit,
                    )
                    if include_claimability
                    else self._projection_dict(state)
                )
                connection.commit()
                return result
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def claim(
        self,
        task_cid: str,
        claimant_did: str,
        *,
        requested_lease_ms: int = 60_000,
        now_ms: int | None = None,
    ) -> LeaseGrant:
        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                resolved_task_cid = self._resolve_task_cid(conn, task_cid)
                if resolved_task_cid is None:
                    raise KeyError(f"unknown task CID: {task_cid}")
                task_cid = resolved_task_cid
                task = conn.execute("SELECT * FROM tasks WHERE task_cid=?", (task_cid,)).fetchone()
                assert task is not None
                readiness = self._claimability(conn, task_cid, max_evidence=32)
                if not readiness["claimable"]:
                    count = len(readiness["blocked_dependency_task_cids"])
                    cycles = len(readiness["dependency_cycles"])
                    repairs = len(readiness["structural_dependency_repairs"])
                    raise DependencyNotReadyError(
                        f"task has {count} unsatisfied prerequisite(s), {cycles} dependency cycle(s), "
                        f"and {repairs} structural dependency repair(s)",
                        evidence=readiness,
                    )
                grant = self._claim_in_transaction(
                    conn, task, claimant_did, duration=duration, now=now
                )
                conn.commit()
                return grant
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def claim_ready(
        self,
        claimant_did: str,
        *,
        requested_lease_ms: int = 60_000,
        exclude_task_cids: Iterable[str] = (),
        eligible_task_cids: Iterable[str] | None = None,
        now_ms: int | None = None,
    ) -> LeaseGrant | None:
        """Atomically select and accept the oldest ready task.

        Selection and acceptance share one ``BEGIN IMMEDIATE`` transaction;
        two scheduler processes can therefore observe the same discovery set
        without ever accepting two leases for one task. Exclusions are applied
        in Python to avoid an unbounded dynamic SQL clause.
        """

        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        excluded = {str(item) for item in exclude_task_cids}
        eligible = (
            None
            if eligible_task_cids is None
            else {str(item) for item in eligible_task_cids}
        )
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                expired = connection.execute(
                    "SELECT task_cid FROM leases WHERE state='accepted' AND expires_at_ms<=?",
                    (now,),
                ).fetchall()
                for row in expired:
                    self._expire(connection, str(row["task_cid"]), now)
                candidates = connection.execute(
                    """SELECT t.*
                       FROM tasks AS t
                       LEFT JOIN leases AS l ON l.task_cid=t.task_cid
                       WHERE l.task_cid IS NULL
                          OR (l.state IN ('released','expired')
                              AND l.retry_not_before_ms<=?)
                       ORDER BY t.registered_at_ms, t.task_cid""",
                    (now,),
                ).fetchall()
                for task in candidates:
                    candidate_cid = str(task["task_cid"])
                    if candidate_cid in excluded or (
                        eligible is not None and candidate_cid not in eligible
                    ):
                        continue
                    # Discovery order is only a scheduling hint. Re-evaluate
                    # dependency receipts in this transaction so a dynamic
                    # worker can never claim a stale or blocked plan entry.
                    readiness = self._claimability(
                        connection, candidate_cid, max_evidence=32
                    )
                    if not readiness["claimable"]:
                        continue
                    # A finite attempt budget prevents a permanently failing
                    # task from monopolizing newly idle lanes.
                    lease = connection.execute(
                        "SELECT attempt FROM leases WHERE task_cid=?", (task["task_cid"],)
                    ).fetchone()
                    if lease is not None and int(lease["attempt"]) >= self._max_attempts(task):
                        continue
                    if self._active_execution_scope_conflict(
                        connection, task, now=now
                    ) is not None:
                        continue
                    grant = self._claim_in_transaction(
                        connection, task, claimant_did, duration=duration, now=now
                    )
                    connection.commit()
                    return grant
                connection.commit()
                return None
            except Exception:
                connection.rollback()
                raise

    @_coordinator_operation
    def steal(
        self,
        task_cid: str,
        claimant_did: str,
        *,
        requested_lease_ms: int = 60_000,
        now_ms: int | None = None,
    ) -> LeaseGrant:
        """Take over an expired or released task with a new fencing token.

        An active lease is never pre-empted, even when the requesting DID is
        the current owner. This stricter behavior makes work-stealing safe to
        call from a competing idle lane.
        """

        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                task = connection.execute(
                    "SELECT * FROM tasks WHERE task_cid=?", (task_cid,)
                ).fetchone()
                if task is None:
                    raise KeyError(f"unknown task CID: {task_cid}")
                self._expire(connection, task_cid, now)
                lease = connection.execute(
                    "SELECT * FROM leases WHERE task_cid=?", (task_cid,)
                ).fetchone()
                if lease is None:
                    raise LeaseConflictError("unclaimed ready tasks must use claim or claim_ready")
                if lease["state"] == "accepted":
                    raise LeaseConflictError(f"task is leased by {lease['claimant_did']}")
                if lease["state"] not in {"expired", "released"}:
                    raise LeaseConflictError(f"task cannot be stolen from state {lease['state']}")
                grant = self._claim_in_transaction(
                    connection, task, claimant_did, duration=duration, now=now
                )
                connection.commit()
                return grant
            except Exception:
                connection.rollback()
                raise

    def _claim_in_transaction(
        self,
        connection: _DuckConnection,
        task: _DuckRow,
        claimant_did: str,
        *,
        duration: int,
        now: int,
    ) -> LeaseGrant:
        """Accept one claim inside a caller-owned immediate transaction."""

        task_cid = str(task["task_cid"])
        self._expire(connection, task_cid, now)
        active = connection.execute(
            "SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms>?",
            (task_cid, now),
        ).fetchone()
        if active is not None:
            if active["claimant_did"] == claimant_did:
                return self._grant(active, task)
            raise LeaseConflictError(f"task is leased by {active['claimant_did']}")
        scope_conflict = self._active_execution_scope_conflict(
            connection, task, now=now
        )
        if scope_conflict is not None:
            execution_scope = self._execution_scope(task)
            raise ExecutionScopeConflictError(
                f"bundle execution scope {execution_scope!r} is leased by task "
                f"{scope_conflict['task_cid']} ({scope_conflict['claimant_did']})"
            )
        prior = connection.execute(
            "SELECT * FROM leases WHERE task_cid=?", (task_cid,)
        ).fetchone()
        if prior is not None and prior["state"] == "completed":
            raise LeaseConflictError("task already has a successful terminal receipt")
        if prior is not None and int(prior["attempt"] or 0) >= self._max_attempts(task):
            raise LeaseConflictError("task attempt budget is exhausted")
        retry_not_before = int(prior["retry_not_before_ms"] or 0) if prior is not None else 0
        if retry_not_before > now:
            raise LeaseConflictError(f"task is cooling down until {retry_not_before}")
        token = int(prior["fencing_token"] or 0) + 1 if prior is not None else 1
        epoch = int(prior["logical_epoch"] or 0) + 1 if prior is not None else 1
        attempt = int(prior["attempt"] or 0) + 1 if prior is not None else 1
        claim = self._claim_payload(task, claimant_did, epoch, attempt, duration, now)
        claim_cid = self._put_artifact(connection, "TaskClaim", claim)
        expires = now + duration
        resolution = {
            "schema": "mcp++/profile-g/claim-resolution@1",
            "created_at_ms": now,
            "parents": [claim_cid],
            "correlation_id": claim["correlation_id"],
            "task_cid": task_cid,
            "logical_epoch": epoch,
            "considered_claim_cids": [claim_cid],
            "accepted_claim_cid": claim_cid,
            "outcome": "accepted",
            "fencing_token": token,
            "lease_expires_at_ms": expires,
            "attestation_cids": [],
            "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}),
            "policy_decision_cid": claim["policy_decision_cid"],
            "coordination_receipt_cid": None,
            "retry_not_before_ms": 0,
            "resolver_did": "did:web:ipfs-accelerate.local",
        }
        resolution_cid = self._put_artifact(connection, "ClaimResolution", resolution)
        connection.execute(
            """INSERT INTO leases(
                   task_cid, claim_cid, resolution_cid, claimant_did,
                   logical_epoch, fencing_token, expires_at_ms, attempt,
                   state, started_at_ms, release_reason, retry_not_before_ms
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(task_cid) DO UPDATE SET
                 claim_cid=excluded.claim_cid,
                 resolution_cid=excluded.resolution_cid,
                 claimant_did=excluded.claimant_did,
                 logical_epoch=excluded.logical_epoch,
                 fencing_token=excluded.fencing_token,
                 expires_at_ms=excluded.expires_at_ms,
                 attempt=excluded.attempt,
                 state='accepted',
                 started_at_ms=excluded.started_at_ms,
                 release_reason=NULL,
                 retry_not_before_ms=0""",
            (
                task_cid,
                claim_cid,
                resolution_cid,
                claimant_did,
                epoch,
                token,
                expires,
                attempt,
                "accepted",
                now,
                None,
                0,
            ),
        )
        connection.execute("INSERT INTO token_history VALUES(?,?)", (task_cid, token))
        return LeaseGrant(
            task_cid,
            task["goal_cid"],
            task["subgoal_cid"],
            claim_cid,
            resolution_cid,
            claimant_did,
            epoch,
            token,
            expires,
            attempt,
        )

    def _claim_payload(self, task: _DuckRow, claimant: str, epoch: int, attempt: int, duration: int, now: int) -> dict[str, Any]:
        bundle = json.loads(task["bundle_json"])
        correlation = str(bundle.get("correlation_id") or bundle.get("bundle_key") or task["task_id"])[:128]
        return {
            "schema": "mcp++/profile-g/task-claim@1", "created_at_ms": now, "parents": [],
            "correlation_id": correlation, "task_cid": task["task_cid"],
            "proposal_cid": _link({"proposal": task["task_cid"], "epoch": epoch}),
            "claimant_did": claimant, "record_cid": _link({"peer": claimant}), "logical_epoch": epoch,
            "requested_lease_ms": duration, "risk_bucket": 0, "capability_fit_millionths": 1_000_000,
            "expected_finish_ms": now + duration, "proof_cid": str(bundle.get("proof_cid") or _link({"proof": claimant})),
            "policy_decision_cid": str(bundle.get("policy_decision_cid") or _link({"decision": "allow"})), "attempt": attempt,
        }

    def _grant(self, lease: _DuckRow, task: _DuckRow | None = None) -> LeaseGrant:
        task = task or self._connection.execute("SELECT * FROM tasks WHERE task_cid=?", (lease["task_cid"],)).fetchone()
        assert task is not None
        return LeaseGrant(lease["task_cid"], task["goal_cid"], task["subgoal_cid"], lease["claim_cid"], lease["resolution_cid"], lease["claimant_did"], lease["logical_epoch"], lease["fencing_token"], lease["expires_at_ms"], lease["attempt"])

    def _current(self, connection: _DuckConnection, grant: LeaseGrant, now: int) -> _DuckRow:
        self._expire(connection, grant.task_cid, now)
        row = connection.execute("SELECT * FROM leases WHERE task_cid=?", (grant.task_cid,)).fetchone()
        if row is None or row["state"] != "accepted" or row["expires_at_ms"] <= now:
            raise LeaseExpiredError("lease has expired or was released")
        if row["claim_cid"] != grant.claim_cid or row["claimant_did"] != grant.claimant_did:
            raise StaleFencingTokenError("claim is no longer accepted")
        if row["fencing_token"] != grant.fencing_token:
            raise StaleFencingTokenError("fencing token is stale")
        return row

    @_coordinator_operation
    def validate(self, grant: LeaseGrant, *, now_ms: int | None = None) -> LeaseGrant:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(self._connection, grant, now)
                result = self._grant(row)
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    @_coordinator_operation
    def renew(self, grant: LeaseGrant, *, requested_lease_ms: int = 60_000, now_ms: int | None = None) -> LeaseGrant:
        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                expires = now + duration
                claim = json.loads(conn.execute("SELECT payload_json FROM artifacts WHERE cid=?", (grant.claim_cid,)).fetchone()[0])
                event = {
                    "schema": "mcp++/profile-g/claim-resolution@1", "created_at_ms": now,
                    "parents": [row["resolution_cid"]], "correlation_id": claim["correlation_id"],
                    "task_cid": grant.task_cid, "logical_epoch": grant.logical_epoch,
                    "considered_claim_cids": [grant.claim_cid], "accepted_claim_cid": grant.claim_cid,
                    "outcome": "accepted", "fencing_token": grant.fencing_token,
                    "lease_expires_at_ms": expires, "attestation_cids": [],
                    "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}),
                    "policy_decision_cid": claim["policy_decision_cid"], "coordination_receipt_cid": None,
                    "retry_not_before_ms": 0, "resolver_did": "did:web:ipfs-accelerate.local",
                }
                renewal_cid = self._put_artifact(conn, "ClaimResolution", event)
                conn.execute("UPDATE leases SET expires_at_ms=?, resolution_cid=? WHERE task_cid=?", (expires, renewal_cid, grant.task_cid))
                conn.commit()
                return LeaseGrant(grant.task_cid, grant.goal_cid, grant.subgoal_cid, grant.claim_cid, renewal_cid, grant.claimant_did, grant.logical_epoch, grant.fencing_token, expires, grant.attempt)
            except Exception:
                conn.rollback()
                raise

    def _resolution_payload(self, row: _DuckRow, *, outcome: str, now: int) -> dict[str, Any]:
        claim = json.loads(self._connection.execute("SELECT payload_json FROM artifacts WHERE cid=?", (row["claim_cid"],)).fetchone()[0])
        return {
            "schema": "mcp++/profile-g/claim-resolution@1", "created_at_ms": now, "parents": [row["resolution_cid"]],
            "correlation_id": claim["correlation_id"], "task_cid": row["task_cid"], "logical_epoch": row["logical_epoch"],
            "considered_claim_cids": [row["claim_cid"]], "accepted_claim_cid": None, "outcome": outcome,
            "fencing_token": row["fencing_token"], "lease_expires_at_ms": None, "attestation_cids": [],
            "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}), "policy_decision_cid": claim["policy_decision_cid"],
            "coordination_receipt_cid": None, "retry_not_before_ms": 0, "resolver_did": "did:web:ipfs-accelerate.local",
        }

    @_coordinator_operation
    def release(
        self,
        grant: LeaseGrant,
        *,
        reason: str = "released",
        now_ms: int | None = None,
    ) -> str:
        """Voluntarily return accepted work to the ready pool.

        ``reason`` is scheduler metadata (for example ``drained`` or
        ``blocked``), not lease authority. The accepted claim and fencing token
        are still checked atomically before the state transition.
        """

        now = self._clock_ms() if now_ms is None else int(now_ms)
        reason = str(reason or "released")[:256]
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                cid = self._put_artifact(conn, "ClaimResolution", self._resolution_payload(row, outcome="released", now=now))
                conn.execute(
                    """UPDATE leases SET state='released', resolution_cid=?, release_reason=?
                       WHERE task_cid=? AND claim_cid=? AND fencing_token=?""",
                    (cid, reason, grant.task_cid, grant.claim_cid, grant.fencing_token),
                )
                conn.commit()
                return cid
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def heartbeat(
        self,
        grant: LeaseGrant,
        *,
        capacity_millionths: int,
        ttl_ms: int = 15_000,
        now_ms: int | None = None,
        active_phase: str | None = None,
        cpu_millionths: int | None = None,
        cpu_percent: int | None = None,
        memory_percent: int | None = None,
        disk_percent: int | None = None,
        memory_used_bytes: int | None = None,
        memory_available_bytes: int | None = None,
        memory_total_bytes: int | None = None,
        disk_used_bytes: int | None = None,
        disk_available_bytes: int | None = None,
        disk_total_bytes: int | None = None,
        occupied_workers: int | None = None,
        available_workers: int | None = None,
        resource_class: str | None = None,
        provider_id: str | None = None,
        provider_capacity: Mapping[str, Any] | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish a live, fenced worker heartbeat.

        Resource measurements use integers so the resulting Profile G artifact
        remains canonical DAG-JSON.  Callers that sample fractional percentages
        should use fixed-point units (``cpu_millionths`` is one million for one
        fully occupied CPU).  Optional provider and detail mappings are retained
        in ``payload_json`` without widening the stable DuckDB table.
        """

        def optional_integer(name: str, value: int | None) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            return value

        def optional_text(name: str, value: str | None) -> str | None:
            if value is None:
                return None
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            return value

        now = self._clock_ms() if now_ms is None else int(now_ms)
        capacity = optional_integer("capacity_millionths", capacity_millionths)
        assert capacity is not None  # Required by the public signature.
        if not 0 <= capacity <= 1_000_000:
            raise ValueError("capacity_millionths must be in [0, 1000000]")
        ttl = optional_integer("ttl_ms", ttl_ms)
        assert ttl is not None
        measurements = {
            "cpu_millionths": optional_integer("cpu_millionths", cpu_millionths),
            "cpu_percent": optional_integer("cpu_percent", cpu_percent),
            "memory_percent": optional_integer("memory_percent", memory_percent),
            "disk_percent": optional_integer("disk_percent", disk_percent),
            "memory_used_bytes": optional_integer("memory_used_bytes", memory_used_bytes),
            "memory_available_bytes": optional_integer("memory_available_bytes", memory_available_bytes),
            "memory_total_bytes": optional_integer("memory_total_bytes", memory_total_bytes),
            "disk_used_bytes": optional_integer("disk_used_bytes", disk_used_bytes),
            "disk_available_bytes": optional_integer("disk_available_bytes", disk_available_bytes),
            "disk_total_bytes": optional_integer("disk_total_bytes", disk_total_bytes),
            "occupied_workers": optional_integer("occupied_workers", occupied_workers),
            "available_workers": optional_integer("available_workers", available_workers),
        }
        text_fields = {
            "active_phase": optional_text("active_phase", active_phase),
            "resource_class": optional_text("resource_class", resource_class),
            "provider_id": optional_text("provider_id", provider_id),
        }
        if provider_capacity is not None and not isinstance(provider_capacity, Mapping):
            raise ValueError("provider_capacity must be a mapping")
        if detail is not None and not isinstance(detail, Mapping):
            raise ValueError("detail must be a mapping")
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._current(conn, grant, now)
                payload = {"schema": "ipfs_accelerate_py/daemon-heartbeat@1", "created_at_ms": now,
                           "task_cid": grant.task_cid, "goal_cid": grant.goal_cid, "subgoal_cid": grant.subgoal_cid,
                           "claim_cid": grant.claim_cid, "claimant_did": grant.claimant_did,
                           "fencing_token": grant.fencing_token, "capacity_millionths": capacity,
                           "expires_at_ms": min(grant.lease_expires_at_ms, now + ttl)}
                payload.update({key: value for key, value in measurements.items() if value is not None})
                payload.update({key: value for key, value in text_fields.items() if value is not None})
                if provider_capacity is not None:
                    payload["provider_capacity"] = dict(provider_capacity)
                if detail is not None:
                    payload["detail"] = dict(detail)
                # Validate the complete payload before touching either artifact
                # table. This rejects nested floats and unsupported containers.
                canonical_profile_g_bytes(payload)
                cid = self._put_artifact(conn, "DaemonHeartbeat", payload)
                conn.execute("INSERT OR REPLACE INTO heartbeats VALUES(?,?,?,?,?,?,?,?)",
                             (cid, grant.task_cid, grant.claimant_did, grant.fencing_token, now, payload["expires_at_ms"], capacity, json.dumps(payload, sort_keys=True)))
                self._prune_heartbeat_history(conn, grant)
                conn.commit()
                return {**payload, "heartbeat_cid": cid}
            except Exception:
                conn.rollback()
                raise

    @_coordinator_operation
    def latest_heartbeats(
        self,
        *,
        task_cids: Iterable[str] | None = None,
        provider_id: str | None = None,
        include_expired: bool = False,
        now_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the newest heartbeat for each task.

        Expired advertisements are excluded by default so callers cannot use a
        dead worker's resource or provider capacity for admission decisions.
        Historical inspection may opt in with ``include_expired=True``.
        ``provider_id`` matches the explicit provider telemetry identifier, not
        the lease claimant DID; legacy heartbeats therefore do not match it.
        """

        now = self._clock_ms() if now_ms is None else int(now_ms)
        selected = None if task_cids is None else {str(item) for item in task_cids}
        if selected == set():
            return []
        if provider_id is not None and not isinstance(provider_id, str):
            raise ValueError("provider_id must be a string")
        with self._lock:
            query = "SELECT * FROM heartbeats"
            clauses: list[str] = []
            parameters: list[Any] = []
            if not include_expired:
                clauses.append("expires_at_ms>?")
                parameters.append(now)
            if selected is not None:
                placeholders = ",".join("?" for _item in selected)
                clauses.append(f"task_cid IN ({placeholders})")
                parameters.extend(sorted(selected))
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY observed_at_ms DESC, heartbeat_cid DESC"
            rows = self._connection.execute(query, parameters).fetchall()

        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            task_cid = str(row["task_cid"])
            if task_cid in latest:
                continue
            try:
                payload = json.loads(str(row["payload_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if provider_id is not None and payload.get("provider_id") != provider_id:
                continue
            payload["heartbeat_cid"] = str(row["heartbeat_cid"])
            latest[task_cid] = payload
        return [latest[task_cid] for task_cid in sorted(latest)]

    @_coordinator_operation
    def latest_heartbeat(
        self,
        task_cid: str,
        *,
        include_expired: bool = False,
        now_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Return the newest current heartbeat for one task, if any."""

        items = self.latest_heartbeats(
            task_cids=(task_cid,),
            include_expired=include_expired,
            now_ms=now_ms,
        )
        return items[0] if items else None

    @_coordinator_operation
    def receipt(
        self, grant: LeaseGrant, *, status: str, output: Mapping[str, Any] | None = None,
        failure_class: str = "none", started_at_ms: int | None = None, now_ms: int | None = None,
    ) -> dict[str, Any]:
        """Publish a terminal receipt for a fenced execution.

        ``succeeded`` is the merge-success authority used by downstream
        dependency gates.  Callers must therefore emit it only after the task's
        outputs have merged and its required validation has passed.
        """

        now = self._clock_ms() if now_ms is None else int(now_ms)
        status = str(status)
        if status not in {"succeeded", "failed", "cancelled", "compensated"}:
            raise ValueError("invalid receipt status")
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                output_cid = _link(dict(output or {})) if output is not None else None
                payload = {
                    "schema": "mcp++/profile-g/task-receipt@1", "created_at_ms": now, "parents": [row["resolution_cid"]],
                    "correlation_id": json.loads(conn.execute("SELECT payload_json FROM artifacts WHERE cid=?", (grant.claim_cid,)).fetchone()[0])["correlation_id"],
                    "task_cid": grant.task_cid, "claim_cid": grant.claim_cid, "resolution_cid": row["resolution_cid"],
                    "fencing_token": grant.fencing_token, "profile_b_receipt_cid": _link({"task": grant.task_cid, "token": grant.fencing_token, "finished": now}),
                    "output_cid": output_cid, "status": status, "failure_class": failure_class, "attempt": grant.attempt,
                    "started_at_ms": int(started_at_ms if started_at_ms is not None else row["started_at_ms"]), "finished_at_ms": now,
                    "resource_use_cid": _link({"heartbeats": self._heartbeat_count(conn, grant)}), "provider": "ipfs_accelerate_py",
                    "provider_version": PROVIDER_VERSION, "next_state": "complete" if status == "succeeded" else "ready",
                }
                if status == "succeeded" and output is None:
                    raise ValueError("successful receipt requires output")
                cid = self._put_artifact(conn, "TaskReceipt", payload)
                conn.execute("INSERT INTO receipts VALUES(?,?,?,?,?,?,?)", (cid, grant.task_cid, grant.goal_cid, grant.subgoal_cid, grant.claim_cid, grant.fencing_token, json.dumps(payload, sort_keys=True)))
                terminal = "completed" if status == "succeeded" else "released"
                release_reason = None if status == "succeeded" else f"receipt:{status}:{failure_class}"[:256]
                conn.execute(
                    "UPDATE leases SET state=?, release_reason=? WHERE task_cid=?",
                    (terminal, release_reason, grant.task_cid),
                )
                conn.commit()
                return {"receipt_cid": cid, "goal_cid": grant.goal_cid, "subgoal_cid": grant.subgoal_cid, "receipt": payload}
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _prune_heartbeat_history(
        connection: _DuckConnection,
        grant: LeaseGrant,
    ) -> None:
        stale_rows = connection.execute(
            """SELECT heartbeat_cid
               FROM (
                 SELECT heartbeat_cid,
                        row_number() OVER (
                          ORDER BY observed_at_ms DESC, heartbeat_cid DESC
                        ) AS history_rank
                 FROM heartbeats
                 WHERE task_cid=? AND fencing_token=?
               )
               WHERE history_rank>?""",
            (
                grant.task_cid,
                grant.fencing_token,
                MAX_PERSISTED_HEARTBEATS_PER_LEASE,
            ),
        ).fetchall()
        stale_cids = [(str(row[0]),) for row in stale_rows]
        if not stale_cids:
            return
        connection.executemany(
            "DELETE FROM heartbeats WHERE heartbeat_cid=?",
            stale_cids,
        )
        connection.executemany(
            "DELETE FROM artifacts "
            "WHERE cid=? AND kind='DaemonHeartbeat'",
            stale_cids,
        )

    @staticmethod
    def _heartbeat_count(connection: _DuckConnection, grant: LeaseGrant) -> int:
        return int(connection.execute("SELECT COUNT(*) FROM heartbeats WHERE task_cid=? AND fencing_token=?", (grant.task_cid, grant.fencing_token)).fetchone()[0])

    @_coordinator_operation
    def active_lease(self, task_cid: str, *, now_ms: int | None = None) -> LeaseGrant | None:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock, self._connection:
            self._expire(self._connection, task_cid, now)
            row = self._connection.execute("SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms>?", (task_cid, now)).fetchone()
            return self._grant(row) if row is not None else None

    @_coordinator_operation
    def list_receipts(self, task_cid: str) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM receipts WHERE task_cid=? ORDER BY rowid", (task_cid,)).fetchall()
        return [{"receipt_cid": row["receipt_cid"], "goal_cid": row["goal_cid"], "subgoal_cid": row["subgoal_cid"], "receipt": json.loads(row["payload_json"])} for row in rows]

    @_coordinator_operation
    def get_artifact(self, cid: str) -> dict[str, Any] | None:
        """Return a stored coordination artifact by CID."""

        row = self._connection.execute("SELECT payload_json FROM artifacts WHERE cid=?", (cid,)).fetchone()
        return json.loads(row[0]) if row is not None else None


_LEGACY_COORDINATION_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("artifacts", ("cid", "kind", "payload_json", "created_at_ms")),
    (
        "tasks",
        (
            "task_cid",
            "goal_cid",
            "subgoal_cid",
            "task_id",
            "bundle_json",
            "registered_at_ms",
            "updated_at_ms",
        ),
    ),
    ("task_aliases", ("alias_task_cid", "task_cid")),
    (
        "task_dependencies",
        ("task_cid", "dependency_task_cid", "provenance_json"),
    ),
    (
        "task_dependency_repairs",
        ("task_cid", "repair_index", "payload_json"),
    ),
    (
        "task_dependency_repair_state",
        ("task_cid", "source_count", "stored_count"),
    ),
    (
        "leases",
        (
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
        ),
    ),
    ("token_history", ("task_cid", "fencing_token")),
    (
        "heartbeats",
        (
            "heartbeat_cid",
            "task_cid",
            "claimant_did",
            "fencing_token",
            "observed_at_ms",
            "expires_at_ms",
            "capacity_millionths",
            "payload_json",
        ),
    ),
    (
        "receipts",
        (
            "receipt_cid",
            "task_cid",
            "goal_cid",
            "subgoal_cid",
            "claim_cid",
            "fencing_token",
            "payload_json",
        ),
    ),
)


def migrate_sqlite_coordination_store(
    source_path: str | Path,
    target_path: str | Path,
    *,
    replace: bool = False,
    batch_size: int = 512,
    current_task_cids: Iterable[str] | None = None,
    heartbeat_history_per_lease: int = 8,
    preserve_all_history: bool = False,
) -> dict[str, Any]:
    """Atomically migrate a bounded legacy SQLite lease store into DuckDB.

    Migration is intentionally explicit. A live SQLite file is never opened as
    DuckDB, and the source remains untouched for rollback or audit. By default,
    the latest registration batch, its dependency closure, authoritative lease
    and receipt rows, and bounded heartbeat history are retained. The legacy
    source remains the cold audit archive.
    """

    source = Path(source_path)
    target = Path(target_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as stream:
        if stream.read(16) != b"SQLite format 3\0":
            raise ValueError(f"not a SQLite coordination store: {source}")
    if target.exists() and not replace:
        raise FileExistsError(target)
    limit = int(batch_size)
    if not 1 <= limit <= 10_000:
        raise ValueError("batch_size must be in [1, 10000]")
    heartbeat_limit = int(heartbeat_history_per_lease)
    if not 0 <= heartbeat_limit <= 10_000:
        raise ValueError("heartbeat_history_per_lease must be in [0, 10000]")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.migration-{threading.get_ident()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    temporary_lock = temporary.with_name(f".{temporary.name}.lock")
    counts: dict[str, int] = {}
    source_connection = sqlite3.connect(str(source))
    source_connection.row_factory = sqlite3.Row

    def table_exists(table: str) -> bool:
        return (
            source_connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            is not None
        )

    def chunks(values: Iterable[str], size: int = 400) -> Iterator[list[str]]:
        items = sorted({str(value) for value in values if str(value)})
        for offset in range(0, len(items), size):
            yield items[offset : offset + size]

    def rows_for_task_cids(
        table: str,
        columns: tuple[str, ...],
        task_ids: set[str],
    ) -> list[tuple[Any, ...]]:
        if not task_ids or not table_exists(table):
            return []
        selected = ", ".join(columns)
        rows: list[tuple[Any, ...]] = []
        for batch in chunks(task_ids):
            placeholders = ", ".join("?" for _ in batch)
            rows.extend(
                tuple(row)
                for row in source_connection.execute(
                    f"SELECT {selected} FROM {table} "
                    f"WHERE task_cid IN ({placeholders})",
                    batch,
                ).fetchall()
            )
        return rows

    def compact_historical_bundle(raw: str) -> str:
        if len(raw) <= 1_000_000:
            return raw
        try:
            bundle = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return json.dumps(
                {
                    "schema": "ipfs_accelerate_py.agent_supervisor.legacy-bundle-tombstone@1",
                    "source_bytes": len(raw),
                },
                sort_keys=True,
            )
        tasks = []
        for item in bundle.get("tasks", []) if isinstance(bundle, Mapping) else []:
            if not isinstance(item, Mapping):
                continue
            tasks.append(
                {
                    key: item[key]
                    for key in (
                        "task_id",
                        "task_cid",
                        "canonical_task_cid",
                        "status",
                        "depends_on",
                    )
                    if item.get(key) not in (None, "", [], {})
                }
            )
        return json.dumps(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.legacy-bundle-tombstone@1",
                "bundle_key": bundle.get("bundle_key")
                if isinstance(bundle, Mapping)
                else "",
                "source_todo": bundle.get("source_todo")
                if isinstance(bundle, Mapping)
                else "",
                "tasks": tasks,
                "source_bytes": len(raw),
            },
            sort_keys=True,
        )

    try:
        source_connection.execute("BEGIN")
        task_columns = dict(_LEGACY_COORDINATION_COLUMNS)["tasks"]
        retained_task_cids = {
            str(task_cid)
            for task_cid in (current_task_cids or ())
            if str(task_cid)
        }
        if table_exists("tasks"):
            if preserve_all_history:
                retained_task_cids.update(
                    str(row[0])
                    for row in source_connection.execute(
                        "SELECT task_cid FROM tasks"
                    ).fetchall()
                )
            elif not retained_task_cids:
                latest = source_connection.execute(
                    "SELECT max(updated_at_ms) FROM tasks"
                ).fetchone()[0]
                if latest is not None:
                    retained_task_cids.update(
                        str(row[0])
                        for row in source_connection.execute(
                            "SELECT task_cid FROM tasks WHERE updated_at_ms>=?",
                            (max(0, int(latest) - 60_000),),
                        ).fetchall()
                    )
        if table_exists("leases"):
            retained_task_cids.update(
                str(row[0])
                for row in source_connection.execute(
                    "SELECT task_cid FROM leases WHERE state='accepted'"
                ).fetchall()
            )

        dependency_aliases: set[str] = set()
        frontier = set(retained_task_cids)
        while frontier and table_exists("task_dependencies"):
            discovered_aliases: set[str] = set()
            for batch in chunks(frontier):
                placeholders = ", ".join("?" for _ in batch)
                discovered_aliases.update(
                    str(row[0])
                    for row in source_connection.execute(
                        "SELECT dependency_task_cid FROM task_dependencies "
                        f"WHERE task_cid IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
            dependency_aliases.update(discovered_aliases)
            resolved: set[str] = set()
            if discovered_aliases and table_exists("task_aliases"):
                for batch in chunks(discovered_aliases):
                    placeholders = ", ".join("?" for _ in batch)
                    resolved.update(
                        str(row[0])
                        for row in source_connection.execute(
                            "SELECT task_cid FROM task_aliases "
                            f"WHERE alias_task_cid IN ({placeholders})",
                            batch,
                        ).fetchall()
                    )
            if discovered_aliases and table_exists("tasks"):
                for batch in chunks(discovered_aliases):
                    placeholders = ", ".join("?" for _ in batch)
                    resolved.update(
                        str(row[0])
                        for row in source_connection.execute(
                            "SELECT task_cid FROM tasks "
                            f"WHERE task_cid IN ({placeholders})",
                            batch,
                        ).fetchall()
                    )
            frontier = resolved - retained_task_cids
            retained_task_cids.update(resolved)

        table_rows: dict[str, list[tuple[Any, ...]]] = {}
        raw_task_rows = rows_for_task_cids(
            "tasks", task_columns, retained_task_cids
        )
        seed_task_cids = {
            str(task_cid)
            for task_cid in (current_task_cids or retained_task_cids)
            if str(task_cid)
        }
        task_rows: list[tuple[Any, ...]] = []
        artifact_cids: set[str] = set()
        required_aliases = set(retained_task_cids) | dependency_aliases
        for row in raw_task_rows:
            values = list(row)
            task_cid = str(values[0])
            artifact_cids.update((str(values[1]), str(values[2])))
            if task_cid not in seed_task_cids:
                values[4] = compact_historical_bundle(str(values[4]))
            try:
                bundle = json.loads(str(values[4]))
            except (TypeError, ValueError, json.JSONDecodeError):
                bundle = {}
            profile = (
                bundle.get("profile_g")
                if isinstance(bundle, Mapping)
                and isinstance(bundle.get("profile_g"), Mapping)
                else {}
            )
            for key in (
                "goal_cid",
                "subgoal_cid",
                "plan_branch_cid",
                "selection_cid",
                "task_spec_cid",
            ):
                if profile.get(key):
                    artifact_cids.add(str(profile[key]))
                    required_aliases.add(str(profile[key]))
            artifacts = profile.get("artifacts")
            if isinstance(artifacts, Mapping):
                artifact_cids.update(str(cid) for cid in artifacts)
            for item in bundle.get("tasks", []) if isinstance(bundle, Mapping) else []:
                if not isinstance(item, Mapping):
                    continue
                for key in ("canonical_task_cid", "task_cid"):
                    if item.get(key):
                        required_aliases.add(str(item[key]))
            task_rows.append(tuple(values))
        table_rows["tasks"] = task_rows

        alias_columns = dict(_LEGACY_COORDINATION_COLUMNS)["task_aliases"]
        aliases = rows_for_task_cids(
            "task_aliases", alias_columns, retained_task_cids
        )
        if not preserve_all_history:
            aliases = [
                row
                for row in aliases
                if str(row[0]) in required_aliases or str(row[0]) == str(row[1])
            ]
        table_rows["task_aliases"] = aliases
        artifact_cids.update(str(row[0]) for row in aliases)

        for table in (
            "task_dependencies",
            "task_dependency_repairs",
            "task_dependency_repair_state",
            "leases",
            "token_history",
            "receipts",
        ):
            columns = dict(_LEGACY_COORDINATION_COLUMNS)[table]
            table_rows[table] = rows_for_task_cids(
                table, columns, retained_task_cids
            )
        for row in table_rows["leases"]:
            artifact_cids.update((str(row[1]), str(row[2])))
        for row in table_rows["receipts"]:
            artifact_cids.update((str(row[0]), str(row[4])))

        heartbeat_columns = dict(_LEGACY_COORDINATION_COLUMNS)["heartbeats"]
        heartbeats: list[tuple[Any, ...]] = []
        if retained_task_cids and table_exists("heartbeats"):
            for batch in chunks(retained_task_cids):
                placeholders = ", ".join("?" for _ in batch)
                selected = ", ".join(f"h.{column}" for column in heartbeat_columns)
                query = (
                    f"SELECT {selected} FROM ("
                    "SELECT h.*, row_number() OVER ("
                    "PARTITION BY h.task_cid, h.fencing_token "
                    "ORDER BY h.observed_at_ms DESC, h.heartbeat_cid DESC"
                    ") AS history_rank FROM heartbeats h "
                    f"WHERE h.task_cid IN ({placeholders})"
                    ") h LEFT JOIN leases l ON l.task_cid=h.task_cid "
                    "WHERE h.history_rank<=? OR "
                    "(l.state='accepted' AND l.fencing_token=h.fencing_token)"
                )
                heartbeats.extend(
                    tuple(row)
                    for row in source_connection.execute(
                        query,
                        [*batch, heartbeat_limit],
                    ).fetchall()
                )
        table_rows["heartbeats"] = heartbeats
        artifact_cids.update(str(row[0]) for row in heartbeats)

        artifact_columns = dict(_LEGACY_COORDINATION_COLUMNS)["artifacts"]
        artifacts: list[tuple[Any, ...]] = []
        artifact_row_count = 0
        if table_exists("artifacts"):
            artifact_row_count = int(
                source_connection.execute(
                    "SELECT count(*) FROM artifacts"
                ).fetchone()[0]
            )
        retain_all_artifacts = preserve_all_history or (
            artifact_row_count <= SMALL_STORE_FULL_ARTIFACT_LIMIT
        )
        if retain_all_artifacts and artifact_row_count:
            artifacts = [
                tuple(row)
                for row in source_connection.execute(
                    "SELECT cid, kind, payload_json, created_at_ms FROM artifacts"
                ).fetchall()
            ]
        elif artifact_cids and artifact_row_count:
            for batch in chunks(artifact_cids):
                placeholders = ", ".join("?" for _ in batch)
                artifacts.extend(
                    tuple(row)
                    for row in source_connection.execute(
                        "SELECT cid, kind, payload_json, created_at_ms "
                        f"FROM artifacts WHERE cid IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
        table_rows["artifacts"] = artifacts

        with LeaseCoordinator(temporary) as coordinator:
            with coordinator._database_operation():
                connection = coordinator._connection
                assert connection is not None
                connection.execute("BEGIN TRANSACTION")
                try:
                    for table, columns in _LEGACY_COORDINATION_COLUMNS:
                        selected = ", ".join(columns)
                        placeholders = ", ".join("?" for _ in columns)
                        copied = 0
                        rows = table_rows.get(table, [])
                        for offset in range(0, len(rows), limit):
                            batch = rows[offset : offset + limit]
                            connection.executemany(
                                f"INSERT OR REPLACE INTO {table} "
                                f"({selected}) VALUES ({placeholders})",
                                batch,
                            )
                            copied += len(batch)
                        counts[table] = copied
                    connection.execute(
                        "INSERT OR REPLACE INTO coordination_metadata VALUES(?,?)",
                        (
                            "legacy_sqlite_migration",
                            json.dumps(
                                {
                                    "schema": COORDINATION_STORE_SCHEMA,
                                    "source_path": str(source.resolve()),
                                    "source_bytes": source.stat().st_size,
                                    "migrated_at_ms": int(time.time() * 1000),
                                    "preserve_all_history": bool(
                                        preserve_all_history
                                    ),
                                    "retained_task_count": len(
                                        retained_task_cids
                                    ),
                                    "heartbeat_history_per_lease": heartbeat_limit,
                                    "retained_all_artifacts": retain_all_artifacts,
                                    "row_counts": counts,
                                },
                                sort_keys=True,
                            ),
                        ),
                    )
                    connection.commit()
                    connection.execute("CHECKPOINT")
                except Exception:
                    connection.rollback()
                    raise
        source_connection.rollback()
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        source_connection.close()
        temporary_lock.unlink(missing_ok=True)

    return {
        "schema": COORDINATION_STORE_SCHEMA,
        "source_path": str(source),
        "target_path": str(target),
        "source_bytes": source.stat().st_size,
        "target_bytes": target.stat().st_size,
        "retained_task_count": len(retained_task_cids),
        "heartbeat_history_per_lease": heartbeat_limit,
        "preserve_all_history": bool(preserve_all_history),
        "retained_all_artifacts": retain_all_artifacts,
        "row_counts": counts,
    }


@dataclass(frozen=True)
class LeasedQueuedTask:
    """A queue task paired with the only grant that authorizes its execution."""

    task: Any
    grant: LeaseGrant


class LeaseQueueBridge:
    """Bind the legacy DuckDB queue claim lifecycle to Profile G leases.

    Queue ownership alone is never execution authority.  A bridge claim is
    returned only after the embedded canonical TaskSpec has an accepted lease;
    a conflicting queue claim is immediately returned to the queue.
    """

    def __init__(
        self,
        queue: Any,
        coordinator: LeaseCoordinator,
        *,
        worker_id: str,
        claimant_did: str,
        lease_ms: int = 60_000,
    ) -> None:
        self.queue = queue
        self.coordinator = coordinator
        self.worker_id = worker_id
        self.claimant_did = claimant_did
        self.lease_ms = lease_ms

    def claim_next(self, *, supported_task_types: list[str] | None = None) -> LeasedQueuedTask | None:
        task = self.queue.claim_next(worker_id=self.worker_id, supported_task_types=supported_task_types)
        if task is None:
            return None
        payload = task.payload if isinstance(task.payload, Mapping) else {}
        try:
            adapted = self.coordinator.register_bundle(payload)
            grant = self.coordinator.claim(
                adapted["task_cid"],
                self.claimant_did,
                requested_lease_ms=self.lease_ms,
            )
        except Exception:
            self.queue.release(task_id=task.task_id, worker_id=self.worker_id, reason="profile-g lease not accepted")
            raise
        return LeasedQueuedTask(task=task, grant=grant)

    def renew(self, leased: LeasedQueuedTask) -> LeasedQueuedTask:
        return LeasedQueuedTask(
            task=leased.task,
            grant=self.coordinator.renew(leased.grant, requested_lease_ms=self.lease_ms),
        )

    def release(self, leased: LeasedQueuedTask, *, reason: str = "released") -> bool:
        self.coordinator.release(leased.grant, reason=reason)
        return bool(self.queue.release(task_id=leased.task.task_id, worker_id=self.worker_id, reason=reason))

    def complete(
        self,
        leased: LeasedQueuedTask,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        failure_class: str = "none",
    ) -> dict[str, Any]:
        receipt = self.coordinator.receipt(
            leased.grant,
            status=status,
            output=output,
            failure_class=failure_class,
        )
        queue_status = "completed" if status == "succeeded" else "failed"
        self.queue.complete(
            task_id=leased.task.task_id,
            status=queue_status,
            result={"profile_g": receipt},
            error=None if status == "succeeded" else failure_class,
        )
        return receipt


__all__ = [
    "DependencyNotReadyError", "LeaseConflictError", "LeaseCoordinator", "LeaseError", "LeaseExpiredError", "LeaseGrant",
    "LeaseQueueBridge", "LeasedQueuedTask",
    "MAX_LEASE_MS", "MIN_LEASE_MS", "StaleFencingTokenError", "TaskLeaseState", "adapt_goal_bundle",
    "canonical_profile_g_bytes", "profile_g_cid",
]
