"""Typed Quack control-plane client with connection caching and safe SQL.

Interface: ``QuackStateClient@1``

Clients never receive an open SQL surface. Every statement is a named,
parameter-bound template from a closed registry. Identifiers are fixed in the
template text; callers supply only typed parameter values. Attach sessions
verify store/server identity (database UUID, schema fingerprint, generation,
extension fingerprint) and refuse mismatched peers rather than retrying through
an LLM path.

Transport modes:

* ``embedded`` — open ``control.duckdb`` through the existing exclusive-lock
  helper (hermetic tests and single-process tooling);
* ``quack`` — connect to the loopback owner's typed Unix-socket command
  gateway; generic SQL ``ATTACH`` is intentionally unavailable to clients.

Transaction, CAS, fence, generation, and idempotency semantics live in
``control_plane_transactions.StateTransaction``.
"""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..federation.event_wait import (
    AdaptiveLongPollEventWaitClient,
    EventSource,
    EventWaitError,
)
from ..federation.events import EventBatch, EventWaitRequest
from .control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    ControlPlaneBounds,
    ControlPlaneContractError,
    ControlPlaneIdentityError,
    ControlPlaneStoreIdentity,
    StateAuthorityClass,
    StateCommand,
    StoreGeneration,
    canonical_json_bytes,
    content_identity,
)
from .control_plane_transactions import (
    CASResult,
    FenceMismatchError,
    IdempotencyConflictError,
    OptimisticConflictError,
    RetryPolicy,
    StaleGenerationError,
    StateTransaction,
    TransactionConflictKind,
    TransactionError,
    TransientTransactionError,
    default_retry_policy,
    run_with_retry,
)
from .database_task_source import TYPED_DEFERRAL_BUDGET_BLOCK_OPERATION
from .duckdb_state import open_duckdb_connection
from .typed_state_owner import (
    TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_COMMAND,
    TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_OPERATION,
    TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_REASON,
    TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_SCHEMA,
    TYPED_DATABASE_BLOCKED_RETRY_TERMINAL_OPERATION,
    TYPED_DATABASE_CLAIM_PROCESS_SCHEMA,
    TYPED_DATABASE_CLAIM_RECOVERY_COMMAND,
    TYPED_DATABASE_CLAIM_RECOVERY_OPERATION,
    TYPED_DATABASE_CLAIM_RECOVERY_REASON,
    TYPED_DATABASE_CLAIM_RECOVERY_SCHEMA,
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TYPED_RETRY_COOLDOWN_SCHEMA,
    TypedStateOwnerError,
    _process_birth_content_id,
    _process_runtime_facts,
    _validated_stored_retry_cooldown,
    open_typed_state_owner_connection,
)

QUACK_STATE_CLIENT_INTERFACE: Final = "QuackStateClient@1"
QUACK_STATE_CLIENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-state-client@1"
)
QUACK_STATE_CLIENT_VERSION: Final[int] = 1
CLIENT_SESSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/client-session@1"
)
STATEMENT_TEMPLATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/statement-template@1"
)
PAGE_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/page-result@1"
)

DEFAULT_STORE_ID: Final = "control.duckdb"
DEFAULT_PAGE_LIMIT: Final = 50
MAX_PAGE_LIMIT: Final = 500
DEFAULT_CONNECT_TIMEOUT_SECONDS: Final = 30.0
_PROTECTED_REOPENED_TASK_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "proposed",
        "admitted",
        "pending",
        "ready",
        "todo",
        "queued",
        "retrying",
        "claimed",
        "in_progress",
        "running",
    }
)
_SAFE_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDER_RE: Final = re.compile(r"\?")
# Reject attempts to smuggle multi-statement or comment-terminated SQL.
_FORBIDDEN_SQL_FRAGMENT_RE: Final = re.compile(
    r";|--|/\*|\*/|\b(ATTACH|DETACH|COPY|INSTALL|LOAD|PRAGMA|CALL|EXPORT|"
    r"IMPORT|DROP|ALTER|CREATE|TRUNCATE|VACUUM|FORCE|"
    r"READ_CSV(?:_AUTO)?|READ_JSON(?:_AUTO)?|READ_PARQUET|PYTHON_EVAL)\b|"
    r"['\"]/|['\"]\.\./",
    re.IGNORECASE,
)


def _schema_fingerprint_digest(value: str) -> str:
    """Return the SHA-256 identity carried by a canonical schema CID.

    Migration metadata stores the canonical DAG-JSON CID, while the closed
    control-plane store identity contract admits digest strings.  The state
    owner performs the same lossless conversion before publishing identity;
    attached clients must compare that digest instead of an unrelated
    fallback derived from store coordinates.
    """

    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("sha256:"):
        digest = text.removeprefix("sha256:")
        if len(digest) == 64:
            try:
                bytes.fromhex(digest)
            except ValueError:
                pass
            else:
                return f"sha256:{digest.lower()}"
    if text.startswith("b"):
        try:
            encoded = text[1:].upper()
            encoded += "=" * ((8 - len(encoded) % 8) % 8)
            raw = base64.b32decode(encoded)
        except (ValueError, TypeError):
            raw = b""
        dag_json_sha256_prefix = b"\x01\xa9\x02\x12\x20"
        if raw.startswith(dag_json_sha256_prefix) and len(raw) == (
            len(dag_json_sha256_prefix) + 32
        ):
            return f"sha256:{raw[len(dag_json_sha256_prefix):].hex()}"
    return ""


class QuackClientError(ControlPlaneContractError):
    """Base error for the typed Quack client.

    Inherits ``ControlPlaneContractError`` (a ``ValueError``) so identity and
    SQL boundary failures share one exception lattice without mixing
    ``RuntimeError`` and ``ValueError`` bases.
    """


class QuackClientIdentityError(QuackClientError, ControlPlaneIdentityError):
    """Server/store identity verification failed."""


class QuackClientSQLError(QuackClientError):
    """Caller attempted raw SQL, identifier interpolation, or an unknown template."""


class QuackClientTransportError(QuackClientError):
    """Transport/connection failure (may be retried by higher layers)."""


class TransportMode(str, Enum):
    EMBEDDED = "embedded"
    QUACK = "quack"


class StatementKind(str, Enum):
    QUERY = "query"
    MUTATION = "mutation"
    META = "meta"


@dataclass(frozen=True)
class StatementTemplate:
    """Closed, parameter-bound SQL template. Identifiers are fixed in ``sql``."""

    SCHEMA: ClassVar[str] = STATEMENT_TEMPLATE_SCHEMA

    name: str
    sql: str
    parameter_names: tuple[str, ...] = ()
    kind: StatementKind = StatementKind.QUERY
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name or not _SAFE_IDENTIFIER_RE.fullmatch(name):
            raise QuackClientSQLError(f"invalid template name: {self.name!r}")
        sql = str(self.sql or "").strip()
        if not sql:
            raise QuackClientSQLError(f"template {name} has empty SQL")
        if _FORBIDDEN_SQL_FRAGMENT_RE.search(sql):
            # Templates themselves may contain CREATE only in internal seeds;
            # public registry forbids DDL/admin verbs.
            if name not in _INTERNAL_SEED_TEMPLATES:
                raise QuackClientSQLError(
                    f"template {name} contains forbidden SQL surface"
                )
        placeholders = len(_PLACEHOLDER_RE.findall(sql))
        params = tuple(str(item).strip() for item in self.parameter_names)
        if any(not item or not _SAFE_IDENTIFIER_RE.fullmatch(item) for item in params):
            raise QuackClientSQLError(
                f"template {name} has invalid parameter names"
            )
        if placeholders != len(params):
            raise QuackClientSQLError(
                f"template {name} placeholder count {placeholders} != "
                f"parameter count {len(params)}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sql", sql)
        object.__setattr__(self, "parameter_names", params)
        kind = self.kind if isinstance(self.kind, StatementKind) else StatementKind(str(self.kind))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "description", str(self.description or ""))

    def bind(self, parameters: Mapping[str, Any] | Sequence[Any] | None) -> list[Any]:
        """Return ordered parameter values; reject unknown/missing names."""

        if parameters is None:
            values: Mapping[str, Any] | Sequence[Any] = {}
        else:
            values = parameters
        if isinstance(values, Mapping):
            unknown = set(values) - set(self.parameter_names)
            if unknown:
                raise QuackClientSQLError(
                    f"template {self.name} received unknown parameters: "
                    f"{sorted(unknown)}"
                )
            missing = [name for name in self.parameter_names if name not in values]
            if missing:
                raise QuackClientSQLError(
                    f"template {self.name} missing parameters: {missing}"
                )
            ordered = [values[name] for name in self.parameter_names]
        elif isinstance(values, Sequence) and not isinstance(
            values, (str, bytes, bytearray)
        ):
            if len(values) != len(self.parameter_names):
                raise QuackClientSQLError(
                    f"template {self.name} expected {len(self.parameter_names)} "
                    f"parameters, got {len(values)}"
                )
            ordered = list(values)
        else:
            raise QuackClientSQLError(
                f"template {self.name} parameters must be a mapping or sequence"
            )
        for index, value in enumerate(ordered):
            _assert_bound_value(value, self.parameter_names[index])
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "name": self.name,
            "sql": self.sql,
            "parameter_names": list(self.parameter_names),
            "kind": self.kind.value,
            "description": self.description,
        }


def _assert_bound_value(value: Any, name: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
            raise QuackClientSQLError(f"parameter {name} must be finite")
        if isinstance(value, str):
            if "\x00" in value:
                raise QuackClientSQLError(f"parameter {name} must not contain NUL")
            if len(value.encode("utf-8")) > 1_048_576:
                raise QuackClientSQLError(f"parameter {name} exceeds byte bound")
        return
    raise QuackClientSQLError(
        f"parameter {name} has unsupported type {type(value).__name__}"
    )


# Internal-only templates that may touch admin-ish surfaces during seed.
_INTERNAL_SEED_TEMPLATES: Final[frozenset[str]] = frozenset(
    {
        "seed_store_generation",
        "seed_client_session",
        "upsert_task_status",
    }
)


def _default_templates() -> dict[str, StatementTemplate]:
    return {
        "whoami_metadata": StatementTemplate(
            name="whoami_metadata",
            sql=(
                "SELECT key, value FROM control_plane_metadata "
                "WHERE key IN ('database_uuid', 'schema_fingerprint', "
                "'schema_version', 'application_version', 'tool_version') "
                "ORDER BY key"
            ),
            parameter_names=(),
            kind=StatementKind.META,
            description="Read store identity metadata keys",
        ),
        "load_store_generation": StatementTemplate(
            name="load_store_generation",
            sql=(
                "SELECT generation, schema_revision, fence_epoch, revision, "
                "database_uuid, birth_id FROM store_generations "
                "ORDER BY generation DESC LIMIT 1"
            ),
            parameter_names=(),
            kind=StatementKind.META,
            description="Load latest store generation head",
        ),
        "seed_store_generation": StatementTemplate(
            name="seed_store_generation",
            sql=(
                "INSERT INTO store_generations ("
                "generation, schema_revision, fence_epoch, revision, "
                "database_uuid, birth_id, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "generation",
                "schema_revision",
                "fence_epoch",
                "revision",
                "database_uuid",
                "birth_id",
                "created_at",
            ),
            kind=StatementKind.MUTATION,
            description="Seed initial store generation (bootstrap only)",
        ),
        "seed_client_session": StatementTemplate(
            name="seed_client_session",
            sql=(
                "INSERT INTO client_sessions ("
                "session_id, server_id, owner_id, process_birth_id, "
                "attached_at, last_seen_at, fence_epoch, generation, "
                "status, revision"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "session_id",
                "server_id",
                "owner_id",
                "process_birth_id",
                "attached_at",
                "last_seen_at",
                "fence_epoch",
                "generation",
                "status",
                "revision",
            ),
            kind=StatementKind.MUTATION,
            description="Register an attached client session",
        ),
        "touch_client_session": StatementTemplate(
            name="touch_client_session",
            sql=(
                "UPDATE client_sessions SET last_seen_at = ?, revision = revision + 1 "
                "WHERE session_id = ?"
            ),
            parameter_names=("last_seen_at", "session_id"),
            kind=StatementKind.MUTATION,
            description="Heartbeat an attached session",
        ),
        "select_task_by_cid": StatementTemplate(
            name="select_task_by_cid",
            sql=(
                "SELECT task_cid, task_alias, goal_cid, status, revision, "
                "ordinal, body_json FROM tasks WHERE task_cid = ? LIMIT 1"
            ),
            parameter_names=("task_cid",),
            kind=StatementKind.QUERY,
            description="Fetch one task by content id",
        ),
        "list_tasks_page": StatementTemplate(
            name="list_tasks_page",
            sql=(
                "SELECT task_cid, task_alias, goal_cid, status, revision, "
                "ordinal FROM tasks WHERE ordinal > ? "
                "ORDER BY ordinal ASC, task_cid ASC LIMIT ?"
            ),
            parameter_names=("after_ordinal", "limit"),
            kind=StatementKind.QUERY,
            description="Cursor page of tasks ordered by ordinal",
        ),
        "executor_task_projection_page": StatementTemplate(
            name="executor_task_projection_page",
            sql=(
                "SELECT t.task_cid, t.task_alias, t.goal_cid, t.plan_cid, "
                "t.objective_id, t.ordinal, t.status, t.revision, t.priority, "
                "t.identity_json, t.body_json, "
                "COALESCE((SELECT to_json(list(d.dependency_task_cid ORDER BY "
                "d.dependency_task_cid)) FROM task_dependencies AS d WHERE "
                "d.task_cid = t.task_cid), '[]') AS dependencies_json, "
                "COALESCE((SELECT to_json(list(struct_pack(ordinal := o.ordinal, "
                "path := o.path, effect := o.effect_json) ORDER BY o.ordinal)) "
                "FROM task_outputs AS o WHERE o.task_cid = t.task_cid), '[]') "
                "AS outputs_json, COALESCE((SELECT to_json(list(struct_pack("
                "ordinal := a.ordinal, criterion := a.criterion, evidence_policy "
                ":= a.evidence_policy_json) ORDER BY a.ordinal)) FROM "
                "task_acceptance AS a WHERE a.task_cid = t.task_cid), '[]') "
                "AS acceptance_json, COALESCE((SELECT to_json(list(struct_pack("
                "ordinal := v.ordinal, argv := v.argv_json, policy := "
                "v.policy_json) ORDER BY v.ordinal)) FROM task_validations AS v "
                "WHERE v.task_cid = t.task_cid), '[]') AS validations_json "
                "FROM tasks AS t ORDER BY t.ordinal ASC, t.task_cid ASC "
                "LIMIT ? OFFSET ?"
            ),
            parameter_names=("limit", "offset"),
            kind=StatementKind.QUERY,
            description=(
                "Bounded full-fidelity task projection for an admitted executor"
            ),
        ),
        "executor_task_projection_by_identity": StatementTemplate(
            name="executor_task_projection_by_identity",
            sql=(
                "SELECT t.task_cid, t.task_alias, t.goal_cid, t.plan_cid, "
                "t.objective_id, t.ordinal, t.status, t.revision, t.priority, "
                "t.identity_json, t.body_json, "
                "COALESCE((SELECT to_json(list(d.dependency_task_cid ORDER BY "
                "d.dependency_task_cid)) FROM task_dependencies AS d WHERE "
                "d.task_cid = t.task_cid), '[]') AS dependencies_json, "
                "COALESCE((SELECT to_json(list(struct_pack(ordinal := o.ordinal, "
                "path := o.path, effect := o.effect_json) ORDER BY o.ordinal)) "
                "FROM task_outputs AS o WHERE o.task_cid = t.task_cid), '[]') "
                "AS outputs_json, COALESCE((SELECT to_json(list(struct_pack("
                "ordinal := a.ordinal, criterion := a.criterion, evidence_policy "
                ":= a.evidence_policy_json) ORDER BY a.ordinal)) FROM "
                "task_acceptance AS a WHERE a.task_cid = t.task_cid), '[]') "
                "AS acceptance_json, COALESCE((SELECT to_json(list(struct_pack("
                "ordinal := v.ordinal, argv := v.argv_json, policy := "
                "v.policy_json) ORDER BY v.ordinal)) FROM task_validations AS v "
                "WHERE v.task_cid = t.task_cid), '[]') AS validations_json "
                "FROM tasks AS t WHERE t.task_cid = ? OR t.task_alias = ? "
                "ORDER BY t.task_cid ASC LIMIT 2"
            ),
            parameter_names=("task_identity", "task_alias"),
            kind=StatementKind.QUERY,
            description=(
                "Exact full-fidelity task projection for an admitted executor"
            ),
        ),
        "executor_control_snapshot": StatementTemplate(
            name="executor_control_snapshot",
            sql=(
                "SELECT (SELECT COUNT(*) FROM objectives) AS objective_count, "
                "(SELECT COUNT(*) FROM goals) AS goal_count, "
                "(SELECT COUNT(*) FROM plans) AS plan_count, "
                "(SELECT COUNT(*) FROM tasks) AS task_count, "
                "(SELECT COUNT(*) FROM task_dependencies) AS dependency_count, "
                "(SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events) "
                "AS event_watermark, COALESCE((SELECT to_json(list(struct_pack("
                "goal_cid := g.goal_cid, status := g.status, revision := "
                "g.revision) ORDER BY g.goal_cid)) FROM goals AS g), '[]') "
                "AS goals_json, COALESCE((SELECT to_json(list(struct_pack("
                "plan_cid := p.plan_cid, status := p.status, revision := "
                "p.revision) ORDER BY p.plan_cid)) FROM plans AS p), '[]') "
                "AS plans_json, COALESCE((SELECT to_json(list(struct_pack("
                "task_cid := t.task_cid, status := t.status, revision := "
                "t.revision) ORDER BY t.task_cid)) FROM tasks AS t), '[]') "
                "AS tasks_json"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Bounded authoritative executor control-plane snapshot",
        ),
        "executor_retry_cooldown_by_task": StatementTemplate(
            name="executor_retry_cooldown_by_task",
            sql=(
                "SELECT task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, state, "
                "started_at_ms, release_reason, retry_not_before_ms, "
                "owner_session_id, fence_epoch, revision, extension_schema, "
                "extension_json FROM leases WHERE task_cid = ? LIMIT 2"
            ),
            parameter_names=("task_cid",),
            kind=StatementKind.QUERY,
            description=(
                "Read one complete retry cooldown row for an admitted executor"
            ),
        ),
        "executor_retry_cooldown_page": StatementTemplate(
            name="executor_retry_cooldown_page",
            sql=(
                "SELECT task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, state, "
                "started_at_ms, release_reason, retry_not_before_ms, "
                "owner_session_id, fence_epoch, revision, extension_schema, "
                "extension_json FROM leases ORDER BY task_cid LIMIT ? OFFSET ?"
            ),
            parameter_names=("limit", "offset"),
            kind=StatementKind.QUERY,
            description=(
                "Read every bounded lease row so foreign cooldowns fail closed"
            ),
        ),
        "executor_insert_retry_cooldown": StatementTemplate(
            name="executor_insert_retry_cooldown",
            sql=(
                "INSERT INTO leases (task_cid, claim_cid, resolution_cid, "
                "claimant_did, logical_epoch, fencing_token, expires_at_ms, "
                "attempt, state, started_at_ms, release_reason, "
                "retry_not_before_ms, owner_session_id, fence_epoch, revision, "
                "extension_schema, extension_json) SELECT ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ? WHERE ? = -1 RETURNING task_cid, "
                "claim_cid, resolution_cid, claimant_did, logical_epoch, "
                "fencing_token, expires_at_ms, attempt, state, started_at_ms, "
                "release_reason, retry_not_before_ms, owner_session_id, "
                "fence_epoch, revision, extension_schema, extension_json"
            ),
            parameter_names=(
                "task_cid",
                "claim_id",
                "resolution_cid",
                "claimant_did",
                "logical_epoch",
                "fencing_token",
                "expires_at_ms",
                "attempt_number",
                "state",
                "started_at_ms",
                "reason",
                "retry_not_before_ms",
                "owner_session_id",
                "fence_epoch",
                "new_queue_revision",
                "extension_schema",
                "extension_json",
                "expected_queue_revision_for_insert",
            ),
            kind=StatementKind.MUTATION,
            description=(
                "Insert one claim-bound executor cooldown with expected absence"
            ),
        ),
        "executor_update_retry_cooldown": StatementTemplate(
            name="executor_update_retry_cooldown",
            sql=(
                "UPDATE leases SET claim_cid = ?, resolution_cid = ?, "
                "claimant_did = ?, logical_epoch = ?, fencing_token = ?, "
                "expires_at_ms = ?, attempt = ?, state = ?, started_at_ms = ?, "
                "release_reason = ?, retry_not_before_ms = ?, owner_session_id "
                "= ?, fence_epoch = ?, revision = ?, extension_schema = ?, "
                "extension_json = ? WHERE task_cid = ? AND revision = ? AND "
                "attempt = ? AND attempt < ? AND extension_schema = ? RETURNING "
                "task_cid, claim_cid, resolution_cid, claimant_did, "
                "logical_epoch, fencing_token, expires_at_ms, attempt, state, "
                "started_at_ms, release_reason, retry_not_before_ms, "
                "owner_session_id, fence_epoch, revision, extension_schema, "
                "extension_json"
            ),
            parameter_names=(
                "claim_id",
                "resolution_cid",
                "claimant_did",
                "logical_epoch",
                "fencing_token",
                "expires_at_ms",
                "attempt_number",
                "state",
                "started_at_ms",
                "reason",
                "retry_not_before_ms",
                "owner_session_id",
                "fence_epoch",
                "new_queue_revision",
                "extension_schema",
                "extension_json",
                "task_cid",
                "expected_queue_revision",
                "expected_queue_attempt",
                "new_attempt_guard",
                "expected_existing_extension_schema",
            ),
            kind=StatementKind.MUTATION,
            description=(
                "Replace one older typed cooldown by exact lease revision"
            ),
        ),
        "insert_task": StatementTemplate(
            name="insert_task",
            sql=(
                "INSERT INTO tasks ("
                "task_cid, task_alias, goal_cid, plan_cid, objective_id, "
                "ordinal, status, revision, priority, created_at, updated_at, "
                "identity_json, body_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "task_cid",
                "task_alias",
                "goal_cid",
                "plan_cid",
                "objective_id",
                "ordinal",
                "status",
                "revision",
                "priority",
                "created_at",
                "updated_at",
                "identity_json",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Insert a task row",
        ),
        "cas_task_status": StatementTemplate(
            name="cas_task_status",
            sql=(
                "UPDATE tasks SET status = ?, revision = ?, updated_at = ? "
                "WHERE task_cid = ? AND revision = ?"
            ),
            parameter_names=(
                "status",
                "new_revision",
                "updated_at",
                "task_cid",
                "expected_revision",
            ),
            kind=StatementKind.MUTATION,
            description="CAS update task status by expected revision",
        ),
        "executor_insert_validation_run": StatementTemplate(
            name="executor_insert_validation_run",
            sql=(
                "INSERT INTO validation_runs (run_id, task_cid, attempt_id, "
                "started_at, finished_at, status, command_digest, body_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "run_id",
                "task_cid",
                "attempt_id",
                "started_at",
                "finished_at",
                "status",
                "command_digest",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Insert one executor validation run",
        ),
        "executor_insert_validation_result": StatementTemplate(
            name="executor_insert_validation_result",
            sql=(
                "INSERT INTO validation_results (result_id, run_id, task_cid, "
                "ordinal, outcome, evidence_digest, body_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "result_id",
                "run_id",
                "task_cid",
                "ordinal",
                "outcome",
                "evidence_digest",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Insert one executor validation result",
        ),
        "executor_insert_validation_evidence": StatementTemplate(
            name="executor_insert_validation_evidence",
            sql=(
                "INSERT INTO evidence_nodes (evidence_id, parent_evidence_id, "
                "task_cid, evidence_kind, digest, created_at, body_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "evidence_id",
                "parent_evidence_id",
                "task_cid",
                "evidence_kind",
                "digest",
                "created_at",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Insert passing executor validation evidence",
        ),
        "executor_cas_task_status_receipt": StatementTemplate(
            name="executor_cas_task_status_receipt",
            sql=(
                "UPDATE tasks SET status = ?, revision = ?, updated_at = ?, "
                "body_json = ? WHERE task_cid = ? AND revision = ? "
                "RETURNING revision"
            ),
            parameter_names=(
                "status",
                "new_revision",
                "updated_at",
                "body_json",
                "task_cid",
                "expected_task_revision",
            ),
            kind=StatementKind.MUTATION,
            description=(
                "CAS task status while retaining its authoritative transition receipt"
            ),
        ),
        "insert_goal": StatementTemplate(
            name="insert_goal",
            sql=(
                "INSERT INTO goals ("
                "goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal, "
                "title, status, created_at, updated_at, revision, body_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "goal_cid",
                "goal_alias",
                "objective_id",
                "parent_goal_cid",
                "ordinal",
                "title",
                "status",
                "created_at",
                "updated_at",
                "revision",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Insert a goal row",
        ),
        "lookup_idempotency": StatementTemplate(
            name="lookup_idempotency",
            sql=(
                "SELECT idempotency_key, command_kind, command_id, store_id, "
                "session_id, result_digest, created_at, expires_at, body_json "
                "FROM idempotency_records WHERE idempotency_key = ? LIMIT 1"
            ),
            parameter_names=("idempotency_key",),
            kind=StatementKind.QUERY,
            description="Lookup a prior idempotent command result",
        ),
        "count_tasks": StatementTemplate(
            name="count_tasks",
            sql="SELECT COUNT(*) AS task_count FROM tasks",
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Count tasks",
        ),
        "list_ready_task_aliases": StatementTemplate(
            name="list_ready_task_aliases",
            sql=(
                "SELECT t.task_alias FROM tasks AS t "
                "WHERE lower(t.status) IN ('admitted','pending','proposed','queued',"
                "'ready','retrying','todo','unstarted') "
                "AND NOT EXISTS ("
                "SELECT 1 FROM task_dependencies AS d "
                "JOIN tasks AS prerequisite "
                "ON prerequisite.task_cid = d.dependency_task_cid "
                "WHERE d.task_cid = t.task_cid "
                "AND lower(prerequisite.status) NOT IN "
                "('complete','completed','done','skipped')"
                ") ORDER BY t.ordinal ASC, t.task_cid ASC LIMIT 500"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Bounded configured-board ready frontier",
        ),
        "max_event_watermark": StatementTemplate(
            name="max_event_watermark",
            sql=(
                "SELECT COALESCE(MAX(global_sequence), 0) AS event_watermark "
                "FROM domain_events"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Latest authoritative domain-event watermark",
        ),
    }


DEFAULT_STATEMENT_TEMPLATES: Final[Mapping[str, StatementTemplate]] = MappingProxyType(
    _default_templates()
)


@dataclass(frozen=True)
class ClientSession:
    """Attached client session identity."""

    SCHEMA: ClassVar[str] = CLIENT_SESSION_SCHEMA

    session_id: str
    server_id: str
    owner_id: str
    process_birth_id: str
    store_id: str
    generation: int
    fence_epoch: int
    attached_at: str
    transport_mode: TransportMode
    endpoint: str
    store_identity: ControlPlaneStoreIdentity | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "session_id": self.session_id,
            "server_id": self.server_id,
            "owner_id": self.owner_id,
            "process_birth_id": self.process_birth_id,
            "store_id": self.store_id,
            "generation": self.generation,
            "fence_epoch": self.fence_epoch,
            "attached_at": self.attached_at,
            "transport_mode": self.transport_mode.value,
            "endpoint": self.endpoint,
            "store_identity": (
                None if self.store_identity is None else self.store_identity.to_dict()
            ),
        }


@dataclass(frozen=True)
class PageResult:
    """Cursor page for bounded list queries."""

    SCHEMA: ClassVar[str] = PAGE_RESULT_SCHEMA

    items: tuple[Mapping[str, Any], ...]
    next_cursor: int | None
    limit: int
    exhausted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "items": [dict(item) for item in self.items],
            "next_cursor": self.next_cursor,
            "limit": self.limit,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class QuackEndpoint:
    """Resolved client endpoint."""

    mode: TransportMode
    target: str
    database_path: Path | None = None
    quack_uri: str | None = None
    secret_handle: str = ""

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, TransportMode) else TransportMode(str(self.mode))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "target", str(self.target or "").strip())
        if not self.target:
            raise QuackClientError("endpoint target must not be empty")
        if self.secret_handle and not str(self.secret_handle).startswith(
            ("env://", "vault://", "handle:", "secret-handle:")
        ):
            raise QuackClientError(
                "endpoint secret must be an opaque handle, not raw credential"
            )


def resolve_endpoint(
    target: str | Path,
    *,
    mode: TransportMode | str | None = None,
    secret_handle: str = "",
) -> QuackEndpoint:
    """Resolve an embedded path or quack:// URI into a typed endpoint."""

    text = str(target).strip()
    if not text:
        raise QuackClientError("endpoint target is required")
    selected_mode: TransportMode
    if mode is not None:
        selected_mode = mode if isinstance(mode, TransportMode) else TransportMode(str(mode))
    elif text.startswith("quack:"):
        selected_mode = TransportMode.QUACK
    else:
        selected_mode = TransportMode.EMBEDDED

    if selected_mode is TransportMode.EMBEDDED:
        path = Path(text)
        return QuackEndpoint(
            mode=selected_mode,
            target=str(path),
            database_path=path,
            secret_handle=secret_handle,
        )
    if not text.startswith("quack:"):
        raise QuackClientError(
            "quack transport requires a quack: URI (loopback only by default)"
        )
    # Accept quack:127.0.0.1:PORT or quack://127.0.0.1:PORT
    return QuackEndpoint(
        mode=selected_mode,
        target=text,
        quack_uri=text,
        secret_handle=secret_handle,
    )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_birth_id() -> str:
    return f"birth:{uuid.uuid4()}"


def _new_session_id() -> str:
    return f"session:{uuid.uuid4()}"


def _row_mapping(columns: Sequence[str], row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return {
            str(columns[index] if index < len(columns) else index): value
            for index, value in enumerate(row)
        }
    try:
        return {
            str(columns[index]): row[index]  # type: ignore[index]
            for index in range(len(columns))
        }
    except Exception:
        return {"value": row}


def _result_columns(result: Any) -> tuple[str, ...]:
    # DuckDBCursor stores columns on ``_columns``; native results use description.
    direct = getattr(result, "_columns", None)
    if isinstance(direct, Sequence) and direct and not isinstance(
        direct, (str, bytes, bytearray)
    ):
        return tuple(str(item) for item in direct)
    description = getattr(result, "description", None) or ()
    columns: list[str] = []
    for item in description:
        if isinstance(item, Sequence) and item and not isinstance(
            item, (str, bytes, bytearray)
        ):
            columns.append(str(item[0]))
        else:
            columns.append(str(item))
    return tuple(columns)


def _fetch_all(result: Any) -> list[Any]:
    if result is None:
        return []
    fetchall = getattr(result, "fetchall", None)
    if callable(fetchall):
        return list(fetchall() or [])
    if isinstance(result, list):
        return result
    return []


def _fetch_one(result: Any) -> Any | None:
    if result is None:
        return None
    fetchone = getattr(result, "fetchone", None)
    if callable(fetchone):
        return fetchone()
    rows = _fetch_all(result)
    return rows[0] if rows else None


def _reject_protected_typed_deferral_reopen(
    txn: StateTransaction,
    *,
    task_cid: str,
    requested_status: str,
) -> None:
    """Reject generic blocked reopens while the task transaction is held."""

    if requested_status.strip().lower() not in _PROTECTED_REOPENED_TASK_STATUSES:
        return
    result = txn.execute_named_operation("select_task_by_cid", (task_cid,))
    row = _fetch_one(result)
    current = _row_mapping(
        (
            "task_cid",
            "task_alias",
            "goal_cid",
            "status",
            "revision",
            "ordinal",
            "body_json",
        ),
        row,
    )
    if str(current.get("status") or "").strip().lower() != "blocked":
        return
    try:
        prior_body = json.loads(str(current.get("body_json") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QuackClientError(
            "blocked task status CAS prior receipt is malformed"
        ) from exc
    prior_receipt = (
        prior_body.get("completion_receipt")
        if isinstance(prior_body, Mapping)
        else None
    )
    if (
        isinstance(prior_receipt, Mapping)
        and prior_receipt.get("operation")
        == TYPED_DEFERRAL_BUDGET_BLOCK_OPERATION
    ):
        raise QuackClientError(
            "protected typed-deferral task cannot be reopened by generic CAS"
        )


class _ConnectionAdapter:
    """Normalize DuckDBConnection / native duckdb connections for transactions."""

    def __init__(
        self,
        connection: Any,
        *,
        embedded_operations: Mapping[str, StatementTemplate] | None = None,
    ) -> None:
        self._connection = connection
        self._owns_close = False
        self._embedded_operations = MappingProxyType(
            dict(embedded_operations or {})
        )

    @property
    def raw(self) -> Any:
        return self._connection

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Any:
        if parameters is None:
            return self._connection.execute(sql)
        return self._connection.execute(sql, parameters)

    def execute_operation(
        self,
        operation: str,
        parameters: Sequence[Any] | None = None,
    ) -> Any:
        execute = getattr(self._connection, "execute_operation", None)
        if callable(execute):
            return execute(operation, parameters)
        template = self._embedded_operations.get(str(operation or ""))
        if template is None:
            raise AttributeError("connection has no typed owner operation surface")
        bound = template.bind(parameters)
        return self.execute(template.sql, bound if bound else None)

    def commit(self) -> None:
        commit = getattr(self._connection, "commit", None)
        if callable(commit):
            commit()
            # DuckDBConnection tracks BEGIN via execute(); a successful
            # commit() clears that flag. Also issue SQL COMMIT for native
            # connections that started the txn via BEGIN but whose commit()
            # is a no-op outside their own flag.
            try:
                self._connection.execute("COMMIT")
            except Exception:
                pass
            return
        try:
            self._connection.execute("COMMIT")
        except Exception:
            pass

    def rollback(self) -> None:
        rollback = getattr(self._connection, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception:
                pass
        try:
            self._connection.execute("ROLLBACK")
        except Exception:
            pass

    def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if callable(close):
            close()


class QuackStateClient:
    """Typed, fail-closed Quack/DuckDB control-plane client.

    Interface: ``QuackStateClient@1``.
    """

    INTERFACE: ClassVar[str] = QUACK_STATE_CLIENT_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_STATE_CLIENT_SCHEMA
    VERSION: ClassVar[int] = QUACK_STATE_CLIENT_VERSION

    def __init__(
        self,
        *,
        owner_id: str,
        store_id: str = DEFAULT_STORE_ID,
        expected_identity: ControlPlaneStoreIdentity | None = None,
        templates: Mapping[str, StatementTemplate] | None = None,
        retry_policy: RetryPolicy | None = None,
        bounds: ControlPlaneBounds | None = None,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        process_birth_id: str | None = None,
        clock: Callable[[], str] | None = None,
        connection_factory: Callable[[QuackEndpoint], Any] | None = None,
        secret_resolver: Callable[[str], str] | None = None,
    ) -> None:
        owner = str(owner_id or "").strip()
        if not owner:
            raise QuackClientError("owner_id is required")
        self.owner_id = owner
        self.store_id = str(store_id or DEFAULT_STORE_ID).strip() or DEFAULT_STORE_ID
        self.expected_identity = expected_identity
        self.bounds = bounds or ControlPlaneBounds()
        self.retry_policy = retry_policy or default_retry_policy(self.bounds)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.process_birth_id = process_birth_id or _new_birth_id()
        self._clock = clock or _utc_now
        self._connection_factory = connection_factory
        self._secret_resolver = secret_resolver
        self._templates: dict[str, StatementTemplate] = dict(
            templates or DEFAULT_STATEMENT_TEMPLATES
        )
        self._templates_sealed = False
        self._lock = threading.RLock()
        self._endpoint: QuackEndpoint | None = None
        self._adapter: _ConnectionAdapter | None = None
        self._session: ClientSession | None = None
        self._store_generation: StoreGeneration | None = None
        self._closed = False
        self._event_wait_source: EventSource | None = None
        self._event_wait_owner_boundary: Any | None = None
        self._event_wait_minimum_interval_seconds = 0.25
        self._event_wait_maximum_interval_seconds = 5.0
        self._event_wait_backoff_multiplier = 2.0

    @property
    def attached(self) -> bool:
        return self._session is not None and self._adapter is not None and not self._closed

    @property
    def session(self) -> ClientSession | None:
        return self._session

    @property
    def store_generation(self) -> StoreGeneration | None:
        return self._store_generation

    def list_templates(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def get_template(self, name: str) -> StatementTemplate:
        key = str(name or "").strip()
        if key not in self._templates:
            raise QuackClientSQLError(f"unknown statement template: {name!r}")
        return self._templates[key]

    def register_template(self, template: StatementTemplate) -> None:
        """Register an additional closed template (trusted code only)."""

        if not isinstance(template, StatementTemplate):
            raise QuackClientSQLError("template must be a StatementTemplate")
        with self._lock:
            if self._templates_sealed:
                raise QuackClientSQLError(
                    "statement template catalog is sealed for this client"
                )
            self._templates[template.name] = template

    def seal_templates(self) -> tuple[str, ...]:
        """Prevent runtime enlargement of the named statement catalog.

        Federation state-owner facades call this after installing their closed,
        trusted templates.  The method is monotonic and idempotent; it never
        removes an existing template or changes SQL text.
        """

        with self._lock:
            self._templates_sealed = True
            return tuple(sorted(self._templates))

    @property
    def templates_sealed(self) -> bool:
        with self._lock:
            return bool(self._templates_sealed)

    # ------------------------------------------------------------------
    # Typed event wait boundary
    # ------------------------------------------------------------------

    def bind_event_wait_source(
        self,
        source: EventSource,
        *,
        owner_boundary: Any | None = None,
        minimum_interval_seconds: float = 0.25,
        maximum_interval_seconds: float = 5.0,
        backoff_multiplier: float = 2.0,
    ) -> Mapping[str, object]:
        """Bind a closed event source and optional owner-local wait service.

        ``owner_boundary`` is normally :class:`QuackStateServer` in the state
        owner process and provides the real shared condition.  A remote Quack
        client cannot receive server push with the current extension; without
        that boundary this method enables only the bounded, backing-off,
        explicitly unqualified compatibility path.
        """

        if not callable(getattr(source, "events_for_subscription", None)) or not callable(
            getattr(source, "store_generation", None)
        ):
            raise QuackClientError(
                "event source must expose the closed subscription and generation interfaces"
            )
        if owner_boundary is not None:
            if not callable(getattr(owner_boundary, "wait_for_events", None)) or not callable(
                getattr(owner_boundary, "event_wait_capability", None)
            ):
                raise QuackClientError(
                    "owner event boundary does not expose the typed wait interface"
                )
        try:
            minimum = float(minimum_interval_seconds)
            maximum = float(maximum_interval_seconds)
            multiplier = float(backoff_multiplier)
            # Reuse the compatibility implementation's closed bound checks.
            AdaptiveLongPollEventWaitClient(
                lambda request: self._fetch_event_batch(source, request),
                minimum_interval_seconds=minimum,
                maximum_interval_seconds=maximum,
                backoff_multiplier=multiplier,
            )
        except (TypeError, ValueError, EventWaitError) as exc:
            raise QuackClientError("adaptive event wait bounds are invalid") from exc
        with self._lock:
            if not self.attached:
                raise QuackClientError(
                    "event wait source requires an attached state client"
                )
            if self._event_wait_source is not None and self._event_wait_source is not source:
                raise QuackClientError(
                    "event wait source is already bound for this client session"
                )
            if (
                self._event_wait_owner_boundary is not None
                and self._event_wait_owner_boundary is not owner_boundary
            ):
                raise QuackClientError(
                    "event wait owner boundary is already bound for this client session"
                )
            self._event_wait_source = source
            self._event_wait_owner_boundary = owner_boundary
            self._event_wait_minimum_interval_seconds = minimum
            self._event_wait_maximum_interval_seconds = maximum
            self._event_wait_backoff_multiplier = multiplier
        return MappingProxyType(self.event_wait_capability())

    def clear_event_wait_binding(self) -> None:
        """Release client-side references without changing owner state."""

        with self._lock:
            self._event_wait_source = None
            self._event_wait_owner_boundary = None

    def _fetch_event_batch(
        self,
        source: EventSource,
        request: EventWaitRequest,
    ) -> EventBatch:
        events = source.events_for_subscription(
            consumer_id=request.consumer_id,
            subscription_id=request.subscription_id,
            subscription_revision=request.subscription_revision,
            after_cursor=request.after_cursor,
            maximum_events=request.maximum_events,
        )
        return EventBatch(
            consumer_id=request.consumer_id,
            subscription_id=request.subscription_id,
            subscription_revision=request.subscription_revision,
            after_cursor=request.after_cursor,
            next_cursor=(events[-1].global_sequence if events else request.after_cursor),
            store_generation=source.store_generation(),
            events=events,
            timed_out=False,
            cancelled=False,
            server_shutdown=False,
        )

    @staticmethod
    def _validate_event_batch(
        request: EventWaitRequest,
        batch: EventBatch,
        *,
        expected_store_generation: int,
    ) -> EventBatch:
        if not isinstance(batch, EventBatch):
            raise QuackClientError("event wait boundary returned an untyped batch")
        if (
            batch.consumer_id != request.consumer_id
            or batch.subscription_id != request.subscription_id
            or batch.subscription_revision != request.subscription_revision
            or batch.after_cursor != request.after_cursor
            or len(batch.events) > request.maximum_events
            or batch.store_generation != expected_store_generation
        ):
            raise QuackClientIdentityError(
                "event wait batch differs from the bounded request identity"
            )
        return batch

    def wait_for_events(self, request: EventWaitRequest) -> EventBatch:
        """Wait through the owner condition or explicit adaptive fallback."""

        if not isinstance(request, EventWaitRequest):
            raise QuackClientError("event wait requires EventWaitRequest")
        with self._lock:
            session = self._require_session()
            adapter = self._require_adapter()
            source = self._event_wait_source
            owner_boundary = self._event_wait_owner_boundary
            minimum = self._event_wait_minimum_interval_seconds
            maximum = self._event_wait_maximum_interval_seconds
            multiplier = self._event_wait_backoff_multiplier
        remote_wait = getattr(adapter.raw, "wait_for_events", None)
        if (
            session.transport_mode is TransportMode.QUACK
            and callable(remote_wait)
            and bool(getattr(adapter.raw, "supports_event_wait", False))
        ):
            return self._validate_event_batch(
                request,
                remote_wait(request),
                expected_store_generation=session.generation,
            )
        if source is None:
            raise QuackClientError("typed event wait source is not bound")
        if owner_boundary is not None:
            return self._validate_event_batch(
                request,
                owner_boundary.wait_for_events(request),
                expected_store_generation=session.generation,
            )
        if session.transport_mode is not TransportMode.QUACK:
            raise QuackClientError(
                "embedded event waits require the server-owned condition boundary"
            )
        compatibility = AdaptiveLongPollEventWaitClient(
            lambda candidate: self._fetch_event_batch(source, candidate),
            minimum_interval_seconds=minimum,
            maximum_interval_seconds=maximum,
            backoff_multiplier=multiplier,
        )
        return self._validate_event_batch(
            request,
            compatibility.wait_for_events(request),
            expected_store_generation=session.generation,
        )

    def cancel_event_wait(self, consumer_id: str) -> None:
        """Cancel through the owner boundary; adaptive fallback has no push."""

        consumer = str(consumer_id or "").strip()
        if not consumer:
            raise QuackClientError("consumer_id is required")
        with self._lock:
            adapter = self._adapter
            owner_boundary = self._event_wait_owner_boundary
        cancel = getattr(owner_boundary, "cancel_event_wait", None)
        if not callable(cancel) and adapter is not None:
            remote_cancel = getattr(adapter.raw, "cancel_event_wait", None)
            if callable(remote_cancel) and bool(
                getattr(adapter.raw, "supports_event_wait", False)
            ):
                remote_cancel(consumer)
                return
        if not callable(cancel):
            raise QuackClientError(
                "remote adaptive event wait cancellation is unavailable"
            )
        cancel(consumer)

    def clear_event_wait_cancellation(self, consumer_id: str) -> None:
        """Clear an owner-side cancellation before a later wait."""

        consumer = str(consumer_id or "").strip()
        if not consumer:
            raise QuackClientError("consumer_id is required")
        with self._lock:
            adapter = self._adapter
            owner_boundary = self._event_wait_owner_boundary
        clear = getattr(owner_boundary, "clear_event_wait_cancellation", None)
        if not callable(clear) and adapter is not None:
            remote_clear = getattr(
                adapter.raw,
                "clear_event_wait_cancellation",
                None,
            )
            if callable(remote_clear) and bool(
                getattr(adapter.raw, "supports_event_wait", False)
            ):
                remote_clear(consumer)
                return
        if not callable(clear):
            raise QuackClientError(
                "remote adaptive event wait cancellation is unavailable"
            )
        clear(consumer)

    def event_wait_capability(self) -> dict[str, object]:
        """Describe the selected wait path without claiming promotion."""

        with self._lock:
            source = self._event_wait_source
            owner_boundary = self._event_wait_owner_boundary
            session = self._session
            adapter = self._adapter
        if (
            session is not None
            and session.transport_mode is TransportMode.QUACK
            and adapter is not None
            and callable(getattr(adapter.raw, "wait_for_events", None))
            and bool(getattr(adapter.raw, "supports_event_wait", False))
        ):
            return {
                "available": True,
                "interface": "TypedStateOwnerEventWait@1",
                "client_interface": "QuackStateClientEventWait@1",
                "transport": "typed_state_owner_bounded_long_wait",
                "server_owned": True,
                "blocking_condition": True,
                "adaptive_polling": False,
                "event_driven_qualified": True,
            }
        if source is None:
            return {
                "available": False,
                "interface": "QuackStateClientEventWait@1",
                "event_driven_qualified": False,
                "reason": "typed event source is not bound",
            }
        if owner_boundary is not None:
            capability = dict(owner_boundary.event_wait_capability())
            capability.update(
                {
                    "client_interface": "QuackStateClientEventWait@1",
                    "transport": "owner_local_condition",
                    "event_driven_qualified": False,
                }
            )
            return capability
        if session is not None and session.transport_mode is TransportMode.QUACK:
            capability = dict(AdaptiveLongPollEventWaitClient.capability())
            capability.update(
                {
                    "available": True,
                    "client_interface": "QuackStateClientEventWait@1",
                    "transport": "quack_adaptive_long_poll",
                    "event_driven_qualified": False,
                }
            )
            return capability
        return {
            "available": False,
            "interface": "QuackStateClientEventWait@1",
            "event_driven_qualified": False,
            "reason": "embedded mode requires an owner-local wait boundary",
        }

    def attach(
        self,
        target: str | Path | QuackEndpoint,
        *,
        mode: TransportMode | str | None = None,
        secret_handle: str = "",
        server_id: str = "server:local",
        seed_generation: bool = False,
        expected_identity: ControlPlaneStoreIdentity | None = None,
    ) -> ClientSession:
        """Attach to a store, verify identity, and cache the connection."""

        with self._lock:
            if self._closed:
                raise QuackClientError("client is closed")
            if self.attached:
                raise QuackClientError("client is already attached; detach first")
            endpoint = (
                target
                if isinstance(target, QuackEndpoint)
                else resolve_endpoint(target, mode=mode, secret_handle=secret_handle)
            )
            if endpoint.mode is TransportMode.QUACK and secret_handle:
                # Credentials remain handle-only; never materialize into argv.
                endpoint = QuackEndpoint(
                    mode=endpoint.mode,
                    target=endpoint.target,
                    database_path=endpoint.database_path,
                    quack_uri=endpoint.quack_uri,
                    secret_handle=secret_handle,
                )
            adapter = self._open_connection(endpoint)
            try:
                if endpoint.mode is TransportMode.QUACK:
                    owner_identity = getattr(adapter.raw, "identity", None)
                    observed_server_id = (
                        str(owner_identity.get("server_id") or "")
                        if isinstance(owner_identity, Mapping)
                        else ""
                    )
                    if (
                        not observed_server_id
                        and self._connection_factory is not None
                    ):
                        # Hermetic tests may inject an in-memory DB-API object;
                        # it is never the default Quack authority path.
                        observed_server_id = str(server_id or "")
                    if not observed_server_id:
                        raise QuackClientIdentityError(
                            "typed state-owner handshake returned no server identity"
                        )
                    if server_id not in {"", "server:local", observed_server_id}:
                        raise QuackClientIdentityError(
                            "requested server identity differs from the typed owner"
                        )
                    server_id = observed_server_id
                if seed_generation:
                    self._seed_generation_if_missing(adapter)
                generation = self._load_generation(adapter)
                identity = self._observe_store_identity(adapter, generation)
                expected = expected_identity or self.expected_identity
                if expected is not None:
                    self._verify_identity(expected, identity, generation)
                owner_session_id = str(
                    getattr(adapter.raw, "session_id", "") or ""
                )
                session_id = owner_session_id or _new_session_id()
                attached_at = self._clock()
                if not owner_session_id:
                    self._execute_template(
                        adapter,
                        "seed_client_session",
                        {
                            "session_id": session_id,
                            "server_id": server_id,
                            "owner_id": self.owner_id,
                            "process_birth_id": self.process_birth_id,
                            "attached_at": attached_at,
                            "last_seen_at": attached_at,
                            "fence_epoch": generation.fence_epoch,
                            "generation": generation.generation,
                            "status": "attached",
                            "revision": 0,
                        },
                    )
                    # Embedded/test adapters retain their existing session
                    # registration path. Quack sessions are server-issued.
                    adapter.commit()
                session = ClientSession(
                    session_id=session_id,
                    server_id=server_id,
                    owner_id=self.owner_id,
                    process_birth_id=self.process_birth_id,
                    store_id=self.store_id,
                    generation=generation.generation,
                    fence_epoch=generation.fence_epoch,
                    attached_at=attached_at,
                    transport_mode=endpoint.mode,
                    endpoint=endpoint.target,
                    store_identity=identity,
                )
            except Exception:
                adapter.close()
                raise
            self._endpoint = endpoint
            self._adapter = adapter
            self._session = session
            self._store_generation = generation
            return session

    def detach(self) -> None:
        with self._lock:
            adapter = self._adapter
            self._adapter = None
            self._session = None
            self._store_generation = None
            self._endpoint = None
            if adapter is not None:
                adapter.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.detach()
        self.clear_event_wait_binding()

    def __enter__(self) -> QuackStateClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def reconnect(self) -> ClientSession:
        """Drop the cached connection and re-attach to the same endpoint."""

        with self._lock:
            if self._endpoint is None or self._session is None:
                raise QuackClientError("cannot reconnect without a prior attach")
            endpoint = self._endpoint
            expected = (
                self._session.store_identity
                if self._session is not None
                else self.expected_identity
            )
            server_id = self._session.server_id
            self.detach()
            return self.attach(
                endpoint,
                server_id=server_id,
                expected_identity=expected,
            )

    def execute(
        self,
        template_name: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Execute a named template with bound parameters only."""

        with self._lock:
            adapter = self._require_adapter()
            return self._execute_template(adapter, template_name, parameters)

    def execute_sql(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Rejected escape hatch: callers cannot run arbitrary SQL."""

        raise QuackClientSQLError(
            "arbitrary SQL is forbidden; use a named statement template"
        )

    def paginate(
        self,
        template_name: str = "list_tasks_page",
        *,
        cursor: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
        parameters: Mapping[str, Any] | None = None,
        cursor_parameter: str = "after_ordinal",
        limit_parameter: str = "limit",
        cursor_field: str = "ordinal",
    ) -> PageResult:
        """Fetch one cursor page from a list template."""

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise QuackClientError("cursor must be a non-negative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_PAGE_LIMIT
        ):
            raise QuackClientError(f"limit must be between 1 and {MAX_PAGE_LIMIT}")
        params = dict(parameters or {})
        params[cursor_parameter] = cursor
        params[limit_parameter] = limit + 1  # probe for more rows
        rows = self.execute(template_name, params)
        exhausted = len(rows) <= limit
        page_rows = rows[:limit]
        next_cursor: int | None = None
        if not exhausted and page_rows:
            last = page_rows[-1]
            if cursor_field not in last:
                raise QuackClientError(
                    f"page row missing cursor field {cursor_field!r}"
                )
            next_cursor = int(last[cursor_field])
        return PageResult(
            items=tuple(page_rows),
            next_cursor=next_cursor,
            limit=limit,
            exhausted=exhausted,
        )

    def load_generation(self) -> StoreGeneration:
        with self._lock:
            adapter = self._require_adapter()
            generation = self._load_generation(adapter)
            self._store_generation = generation
            return generation

    def transaction(
        self,
        *,
        expected_generation: StoreGeneration | None = None,
    ) -> StateTransaction:
        """Open a StateTransaction against the cached connection."""

        with self._lock:
            adapter = self._require_adapter()
            session = self._session
            generation = expected_generation or self._store_generation
            return StateTransaction(
                adapter,
                store_id=self.store_id,
                expected_generation=generation,
                session_id="" if session is None else session.session_id,
                retry_policy=self.retry_policy,
                now_iso=self._clock,
            )

    def submit_command(
        self,
        command: StateCommand,
        *,
        apply: Callable[[StateTransaction, StateCommand, StoreGeneration], Mapping[str, Any]]
        | None = None,
        refresh_on_conflict: bool = True,
    ) -> CASResult:
        """Submit a fenced idempotent command with jittered conflict retry."""

        if not isinstance(command, StateCommand):
            raise QuackClientError("command must be a StateCommand")
        apply_fn = apply or self._default_task_status_apply

        def _operation(attempt: int) -> CASResult:
            with self._lock:
                adapter = self._require_adapter()
                live = self._load_generation(adapter)
                active = command
                if attempt > 1 and refresh_on_conflict:
                    active = StateCommand(
                        command_id=command.command_id,
                        command_kind=command.command_kind,
                        store_id=command.store_id,
                        session_id=command.session_id or (
                            self._session.session_id if self._session else ""
                        ),
                        expected_generation=live.generation,
                        expected_revision=live.revision,
                        fence_epoch=live.fence_epoch,
                        idempotency_key=command.idempotency_key,
                        authority_class=command.authority_class,
                        parameters=dict(command.parameters),
                        secret_handle=command.secret_handle,
                    )
                txn = StateTransaction(
                    adapter,
                    store_id=self.store_id,
                    expected_generation=StoreGeneration(
                        store_id=self.store_id,
                        generation=active.expected_generation,
                        schema_revision=live.schema_revision,
                        fence_epoch=active.fence_epoch,
                        revision=active.expected_revision,
                        database_uuid=live.database_uuid,
                        birth_id=live.birth_id,
                    ),
                    session_id=active.session_id,
                    retry_policy=self.retry_policy,
                    now_iso=self._clock,
                )
                try:
                    prepare = getattr(adapter.raw, "prepare_command", None)
                    if callable(prepare):
                        prepare(active)
                    result = txn.execute_command(active, apply=apply_fn)
                except OptimisticConflictError as exc:
                    return CASResult(
                        outcome=CommandOutcome.CONFLICT,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.OPTIMISTIC,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except StaleGenerationError as exc:
                    return CASResult(
                        outcome=CommandOutcome.STALE,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.STALE_GENERATION,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except FenceMismatchError as exc:
                    return CASResult(
                        outcome=CommandOutcome.STALE,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.FENCE_MISMATCH,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except IdempotencyConflictError:
                    # Non-retryable: surface the typed failure to the caller.
                    raise
                except TransientTransactionError as exc:
                    return CASResult(
                        outcome=CommandOutcome.CONFLICT,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.TRANSIENT,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except TransactionError as exc:
                    if exc.retryable:
                        return CASResult(
                            outcome=CommandOutcome.CONFLICT,
                            changed=False,
                            revision=live.revision,
                            generation=live.generation,
                            fence_epoch=live.fence_epoch,
                            result={"error": str(exc)},
                            conflict_kind=exc.kind,
                            attempts=attempt,
                            idempotency_key=command.idempotency_key,
                            command_id=command.command_id,
                        )
                    raise
                self._store_generation = self._load_generation(adapter)
                return CASResult(
                    outcome=result.outcome,
                    changed=result.changed,
                    revision=result.revision,
                    generation=result.generation,
                    fence_epoch=result.fence_epoch,
                    result=dict(result.result),
                    conflict_kind=result.conflict_kind,
                    attempts=attempt,
                    idempotency_key=result.idempotency_key,
                    command_id=result.command_id,
                    result_digest=result.result_digest,
                )

        return run_with_retry(_operation, policy=self.retry_policy)

    def apply_command_in_transaction(
        self,
        transaction: StateTransaction,
        command: StateCommand,
        live: StoreGeneration,
    ) -> Mapping[str, Any]:
        """Apply the built-in mutation inside a caller-owned transaction.

        This narrow seam lets the command fabric atomically couple authority
        consumption and its private receipt to the existing domain CAS and
        idempotency record.  It exposes no caller-supplied SQL or mutation
        callback.
        """

        if not isinstance(transaction, StateTransaction):
            raise QuackClientError("transaction must be StateTransaction@1")
        if not isinstance(command, StateCommand):
            raise QuackClientError("command must be StateCommand@1")
        if not isinstance(live, StoreGeneration):
            raise QuackClientError("live generation must be StoreGeneration@1")
        # This compatibility seam is one exact operation, not a generic
        # ``StateCommand`` interpreter.  In particular, an independently
        # authorized OBSERVE/MIGRATE effect must never become a task mutation
        # merely because its signed parameters happen to contain ``status``.
        # Broader lifecycle transitions belong to their dedicated owner
        # operations and contracts.
        parameters = dict(command.parameters)
        if command.command_kind is not CommandKind.CLAIM:
            raise QuackClientError(
                "built-in owner task mutation requires command_kind=claim"
            )
        if set(parameters) != {
            "task_cid",
            "expected_task_revision",
            "status",
        }:
            raise QuackClientError(
                "built-in owner task claim requires the exact closed parameter set"
            )
        if parameters.get("status") != "claimed":
            raise QuackClientError(
                "command_kind=claim authorizes only status=claimed"
            )
        return self._default_task_status_apply(transaction, command, live)

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        new_status: str,
        idempotency_key: str,
        command_id: str | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> CASResult:
        """Convenience CAS for task status using the closed template set."""

        session = self._require_session()
        live = self.load_generation()
        body_json = (
            canonical_json_bytes(dict(body)).decode("utf-8")
            if body is not None
            else ""
        )
        operation = "task.status.cas.receipt" if body is not None else "task.status.cas"
        command = StateCommand(
            command_id=command_id
            or (
                f"cmd:cas-status-receipt:{task_cid}:{expected_task_revision}"
                if body is not None
                else f"cmd:cas-status:{task_cid}:{expected_task_revision}"
            ),
            command_kind=CommandKind.CLAIM,
            store_id=self.store_id,
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key=idempotency_key,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters={
                "operation": operation,
                "task_cid": task_cid,
                "expected_task_revision": expected_task_revision,
                "status": new_status,
                **({"body_json": body_json} if body is not None else {}),
            },
        )
        if body is None:
            return self.submit_command(command)

        def apply_receipt(
            txn: StateTransaction,
            active: StateCommand,
            generation: StoreGeneration,
        ) -> Mapping[str, Any]:
            parameters = dict(active.parameters)
            expected = int(parameters["expected_task_revision"])
            # The remote typed owner performs this semantic check at its
            # transaction boundary.  Embedded mode has no separate owner, so
            # enforce the same closed reopen policy inside this transaction.
            if session.transport_mode is TransportMode.EMBEDDED:
                _reject_protected_typed_deferral_reopen(
                    txn,
                    task_cid=str(parameters["task_cid"]),
                    requested_status=str(parameters["status"]),
                )
            result = txn.execute_named_operation(
                "executor_cas_task_status_receipt",
                (
                    str(parameters["status"]),
                    expected + 1,
                    self._clock(),
                    str(parameters["body_json"]),
                    str(parameters["task_cid"]),
                    expected,
                ),
            )
            row = _fetch_one(result)
            if row is None:
                raise OptimisticConflictError(
                    "task status receipt CAS failed",
                    details={
                        "task_cid": str(parameters["task_cid"]),
                        "expected_task_revision": expected,
                    },
                )
            return {
                "task_cid": str(parameters["task_cid"]),
                "status": str(parameters["status"]),
                "task_revision": expected + 1,
                "store_revision_before": generation.revision,
                "command_id": active.command_id,
                "receipt_persisted": True,
            }

        return self.submit_command(command, apply=apply_receipt)

    def claim_process_attestation(self) -> Mapping[str, Any]:
        """Return the active owner-grant birth tuple for a typed claim.

        The caller merely reproduces the handshake material in its receipt;
        the exclusive owner independently re-reads procfs and validates every
        field against the active kernel-bound grant before admitting a claim.
        """

        adapter = self._require_adapter()
        grant = getattr(adapter.raw, "grant", None)
        if not isinstance(grant, Mapping):
            raise QuackClientError(
                "database claim process attestation requires a typed owner grant"
            )
        required = {
            "grant_id",
            "client_id",
            "process_birth_id",
            "peer_pid",
            "peer_uid",
            "peer_start_time_ticks",
        }
        if any(name not in grant for name in required):
            raise QuackClientError(
                "typed owner grant omits database claim process identity"
            )
        try:
            peer_pid = grant["peer_pid"]
            peer_uid = grant["peer_uid"]
            peer_start = grant["peer_start_time_ticks"]
            if (
                type(peer_pid) is not int
                or peer_pid < 1
                or type(peer_uid) is not int
                or peer_uid < 0
                or type(peer_start) is not int
                or peer_start < 0
            ):
                raise ValueError("invalid typed owner grant process scalar")
            start_time, parent_pid, boot_id = _process_runtime_facts(peer_pid)
        except (TypedStateOwnerError, TypeError, ValueError) as exc:
            raise QuackClientError(
                "typed owner grant process identity is unavailable"
            ) from exc
        birth_id = _process_birth_content_id(
            peer_pid,
            start_time,
            boot_id,
            parent_pid,
        )
        if (
            start_time != peer_start
            or grant.get("process_birth_id") != birth_id
            or grant.get("client_id") != self.owner_id
            or grant.get("process_birth_id") != self.process_birth_id
        ):
            raise QuackClientError(
                "typed owner grant differs from the active process birth"
            )
        return MappingProxyType(
            {
                "schema": TYPED_DATABASE_CLAIM_PROCESS_SCHEMA,
                "grant_id": str(grant["grant_id"]),
                "client_id": str(grant["client_id"]),
                "process_birth_id": birth_id,
                "pid": peer_pid,
                "uid": peer_uid,
                "start_time_ticks": start_time,
                "boot_id": boot_id,
                "parent_pid": parent_pid,
            }
        )

    def recover_dead_claim_reservation(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        task_body: Mapping[str, Any],
        reservation_receipt: Mapping[str, Any],
        now_ms: int | None = None,
    ) -> CASResult:
        """Atomically reopen one reservation whose attested process is dead."""

        task = str(task_cid or "").strip()
        if (
            not task
            or isinstance(expected_task_revision, bool)
            or not isinstance(expected_task_revision, int)
            or expected_task_revision < 1
        ):
            raise QuackClientError(
                "dead claim recovery task revision is invalid"
            )
        body = dict(task_body)
        prior = dict(reservation_receipt)
        if (
            body.get("completion_receipt") != prior
            or prior.get("operation") != "database_claim"
            or prior.get("claim_phase_schema")
            != TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
        ):
            raise QuackClientError(
                "dead claim recovery requires an exact typed reservation"
            )
        text_identity: dict[str, str] = {}
        for name in (
            "claim_id",
            "attempt_id",
            "lease_id",
            "owner_session_id",
        ):
            value = prior.get(name)
            if (
                type(value) is not str
                or not value.strip()
                or value != value.strip()
                or len(value.encode("utf-8")) > 1_024
                or any(marker in value for marker in ("\x00", "\n", "\r"))
            ):
                raise QuackClientError(
                    f"dead claim recovery {name} is invalid"
                )
            text_identity[name] = value
        integer_identity: dict[str, int] = {}
        for name in ("attempt_number", "fencing_token", "fence_epoch"):
            value = prior.get(name)
            if type(value) is not int or value < 1:
                raise QuackClientError(
                    f"dead claim recovery {name} is invalid"
                )
            integer_identity[name] = value
        claimed_from_revision = prior.get("claimed_from_revision")
        historic_attestation = prior.get("claim_process_attestation")
        execution_route = prior.get("execution_route_binding")
        if (
            type(claimed_from_revision) is not int
            or claimed_from_revision != expected_task_revision - 1
            or not isinstance(historic_attestation, Mapping)
            or not isinstance(execution_route, Mapping)
            or prior.get("execution_route_policy_id")
            != execution_route.get("policy_id")
            or prior.get("execution_route_origin_revision")
            != execution_route.get("task_revision")
            or execution_route.get("task_cid") != task
        ):
            raise QuackClientError(
                "dead claim recovery reservation lineage is invalid"
            )
        current_attestation = dict(self.claim_process_attestation())
        exact_identity = {**text_identity, **integer_identity}
        reason = TYPED_DATABASE_CLAIM_RECOVERY_REASON
        started_at_ms = int(time.time() * 1_000) if now_ms is None else now_ms
        if (
            isinstance(started_at_ms, bool)
            or not isinstance(started_at_ms, int)
            or started_at_ms < 0
        ):
            raise QuackClientError("dead claim recovery now_ms is invalid")
        reservation_cid = content_identity(
            {"typed_database_claim_reservation": prior}
        )
        recovery_receipt = {
            "schema": TYPED_DATABASE_CLAIM_RECOVERY_SCHEMA,
            "operation": TYPED_DATABASE_CLAIM_RECOVERY_OPERATION,
            **exact_identity,
            "recovered_claim_phase_schema": (
                TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA
            ),
            "recovered_claimed_from_revision": claimed_from_revision,
            "recovered_reservation_cid": reservation_cid,
            "recovered_claim_process_attestation": dict(
                historic_attestation
            ),
            "recovery_process_attestation": current_attestation,
            "recovered_from_revision": expected_task_revision,
            "queue_reason": reason,
            "backoff_ms": 0,
            "retry_not_before_ms": started_at_ms,
            "control_expected_revision": expected_task_revision,
            "execution_route_binding": dict(execution_route),
            "execution_route_policy_id": execution_route["policy_id"],
            "execution_route_origin_revision": int(
                execution_route["task_revision"]
            ),
        }
        body["completion_receipt"] = recovery_receipt
        body_json = canonical_json_bytes(body).decode("utf-8")
        extension = {
            "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "task_cid": task,
            "expected_task_revision": expected_task_revision,
            **exact_identity,
            "delay_ms": 0,
            "started_at_ms": started_at_ms,
            "retry_not_before_ms": started_at_ms,
            "selection_penalty": 0,
            "consecutive_failures": integer_identity["attempt_number"],
            "reason": reason,
            "expected_queue_revision": -1,
            "expected_queue_attempt": 0,
        }
        extension_json = canonical_json_bytes(extension).decode("utf-8")
        resolution_cid = content_identity(
            {
                "typed_retry_cooldown": extension,
                "started_at_ms": started_at_ms,
            }
        )
        parameters = {
            **extension,
            "operation": TYPED_DATABASE_CLAIM_RECOVERY_COMMAND,
            "expected_task_status": "in_progress",
            "resolution_cid": resolution_cid,
            "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "extension_json": extension_json,
            "status": "retrying",
            "body_json": body_json,
        }
        command_digest = hashlib.sha256(
            canonical_json_bytes(parameters)
        ).hexdigest()
        session = self._require_session()
        live = self.load_generation()
        command = StateCommand(
            command_id=f"cmd:dead-claim-recovery:{command_digest}",
            command_kind=CommandKind.CLAIM,
            store_id=self.store_id,
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key=(
                f"executor-dead-claim-recovery:{command_digest}"
            ),
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters=parameters,
        )

        def apply_recovery(
            txn: StateTransaction,
            active: StateCommand,
            generation: StoreGeneration,
        ) -> Mapping[str, Any]:
            values = dict(active.parameters)
            observed = _fetch_all(
                txn.execute_named_operation(
                    "executor_retry_cooldown_by_task",
                    (values["task_cid"],),
                )
            )
            if observed:
                raise OptimisticConflictError(
                    "dead claim recovery cooldown absence became stale"
                )
            queue_result = txn.execute_named_operation(
                "executor_insert_retry_cooldown",
                (
                    values["task_cid"],
                    values["claim_id"],
                    values["resolution_cid"],
                    values["owner_session_id"],
                    values["fence_epoch"],
                    values["fencing_token"],
                    0,
                    values["attempt_number"],
                    "released",
                    values["started_at_ms"],
                    values["reason"],
                    values["retry_not_before_ms"],
                    values["owner_session_id"],
                    values["fence_epoch"],
                    1,
                    values["extension_schema"],
                    values["extension_json"],
                    -1,
                ),
            )
            if _fetch_one(queue_result) is None:
                raise OptimisticConflictError(
                    "dead claim recovery cooldown absence CAS failed"
                )
            expected = int(values["expected_task_revision"])
            task_result = txn.execute_named_operation(
                "executor_cas_task_status_receipt",
                (
                    "retrying",
                    expected + 1,
                    self._clock(),
                    values["body_json"],
                    values["task_cid"],
                    expected,
                ),
            )
            if _fetch_one(task_result) is None:
                raise OptimisticConflictError(
                    "dead claim recovery task CAS failed"
                )
            return {
                "schema": TYPED_DATABASE_CLAIM_RECOVERY_SCHEMA,
                "operation": TYPED_DATABASE_CLAIM_RECOVERY_COMMAND,
                "task_cid": values["task_cid"],
                "attempt_id": values["attempt_id"],
                "attempt_number": values["attempt_number"],
                "task_revision": expected + 1,
                "queue_revision": 1,
                "retry_not_before_ms": values["retry_not_before_ms"],
                "historic_liveness": "dead",
                "store_revision_before": generation.revision,
            }

        return self.submit_command(command, apply=apply_recovery)

    def recover_blocked_task_retry(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        task_body: Mapping[str, Any],
        terminal_receipt: Mapping[str, Any],
        max_task_attempts_before: int,
        max_task_attempts_after: int,
        operator_handoff_receipt_id: str,
        sidecar_evidence_id: str,
        now_ms: int,
    ) -> CASResult:
        """Atomically admit one operator-sealed blocked retry.

        The caller must retain the exact blocked task body and fixed
        ``now_ms``.  They are part of the command digest, so a restart can
        submit the same identity after the task has advanced and receive the
        durable idempotent result without re-evaluating stale predecessor
        state.  This method is intentionally not surfaced by
        ``TypedDatabaseTaskSource`` or any daemon-wide grant.
        """

        task = str(task_cid or "").strip()
        if (
            not task
            or isinstance(expected_task_revision, bool)
            or not isinstance(expected_task_revision, int)
            or expected_task_revision < 1
        ):
            raise QuackClientError(
                "blocked retry recovery task revision is invalid"
            )
        body = dict(task_body)
        prior = dict(terminal_receipt)
        if body.get("completion_receipt") != prior:
            raise QuackClientError(
                "blocked retry recovery requires the exact terminal receipt"
            )
        terminal_reason = prior.get("reason")
        if (
            prior.get("operation")
            != TYPED_DATABASE_BLOCKED_RETRY_TERMINAL_OPERATION
            or type(terminal_reason) is not str
            or not terminal_reason.strip()
            or terminal_reason != terminal_reason.strip()
            or len(terminal_reason.encode("utf-8")) > 2_048
            or prior.get("retryable") is not False
            or prior.get("control_expected_status") != "in_progress"
            or prior.get("control_expected_revision")
            != expected_task_revision - 1
        ):
            raise QuackClientError(
                "blocked retry recovery terminal lineage is invalid"
            )
        normalized_references: dict[str, str] = {}
        for name, value in {
            "operator_handoff_receipt_id": operator_handoff_receipt_id,
            "sidecar_evidence_id": sidecar_evidence_id,
        }.items():
            selected = str(value or "").strip()
            if (
                not selected
                or selected != value
                or len(selected.encode("utf-8")) > 1_024
                or any(marker in selected for marker in ("\x00", "\n", "\r"))
            ):
                raise QuackClientError(
                    f"blocked retry recovery {name} is invalid"
                )
            normalized_references[name] = selected
        text_identity: dict[str, str] = {}
        for name in (
            "claim_id",
            "attempt_id",
            "lease_id",
            "owner_session_id",
        ):
            value = prior.get(name)
            if (
                type(value) is not str
                or not value.strip()
                or value != value.strip()
                or len(value.encode("utf-8")) > 1_024
                or any(marker in value for marker in ("\x00", "\n", "\r"))
            ):
                raise QuackClientError(
                    f"blocked retry recovery {name} is invalid"
                )
            text_identity[name] = value
        integer_identity: dict[str, int] = {}
        for name in ("attempt_number", "fencing_token", "fence_epoch"):
            value = prior.get(name)
            if type(value) is not int or value < 1:
                raise QuackClientError(
                    f"blocked retry recovery {name} is invalid"
                )
            integer_identity[name] = value
        attempt_number = integer_identity["attempt_number"]
        fresh_attempt_number = attempt_number + 1
        if (
            type(max_task_attempts_before) is not int
            or max_task_attempts_before != attempt_number
            or type(max_task_attempts_after) is not int
            or max_task_attempts_after != fresh_attempt_number
        ):
            raise QuackClientError(
                "blocked retry recovery must admit exactly one fresh attempt"
            )
        execution_route = prior.get("execution_route_binding")
        if not isinstance(execution_route, Mapping):
            raise QuackClientError(
                "blocked retry recovery execution route is absent"
            )
        route = dict(execution_route)
        route_policy_id = route.get("policy_id")
        route_origin_revision = route.get("task_revision")
        if (
            route.get("task_cid") != task
            or type(route_policy_id) is not str
            or not route_policy_id
            or type(route_origin_revision) is not int
            or route_origin_revision < 1
            or prior.get("execution_route_policy_id") != route_policy_id
            or prior.get("execution_route_origin_revision")
            != route_origin_revision
        ):
            raise QuackClientError(
                "blocked retry recovery execution route lineage is invalid"
            )
        if type(now_ms) is not int or now_ms < 0:
            raise QuackClientError(
                "blocked retry recovery now_ms must be a fixed integer"
            )

        source_completion_receipt_id = "sha256:" + hashlib.sha256(
            canonical_json_bytes(prior)
        ).hexdigest()
        route_binding_cid = content_identity(
            {"task_execution_route_binding": route}
        )
        exact_identity = {**text_identity, **integer_identity}
        reason = TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_REASON
        recovery_receipt = {
            "schema": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_SCHEMA,
            "operation": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_OPERATION,
            **exact_identity,
            "terminal_operation": prior["operation"],
            "terminal_reason": terminal_reason,
            "source_completion_receipt_id": (
                source_completion_receipt_id
            ),
            **normalized_references,
            "recovered_from_revision": expected_task_revision,
            "max_task_attempts_before": max_task_attempts_before,
            "max_task_attempts_after": max_task_attempts_after,
            "attempt_refunded": False,
            "fresh_attempt_number": fresh_attempt_number,
            "queue_reason": reason,
            "backoff_ms": 0,
            "retry_not_before_ms": now_ms,
            "control_expected_status": "blocked",
            "control_expected_revision": expected_task_revision,
            "execution_route_binding": route,
            "execution_route_binding_cid": route_binding_cid,
            "execution_route_policy_id": route_policy_id,
            "execution_route_origin_revision": route_origin_revision,
        }
        body["completion_receipt"] = recovery_receipt
        body_json = canonical_json_bytes(body).decode("utf-8")
        extension = {
            "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "task_cid": task,
            "expected_task_revision": expected_task_revision,
            **exact_identity,
            "delay_ms": 0,
            "started_at_ms": now_ms,
            "retry_not_before_ms": now_ms,
            "selection_penalty": 0,
            "consecutive_failures": attempt_number,
            "reason": reason,
            "expected_queue_revision": -1,
            "expected_queue_attempt": 0,
        }
        extension_json = canonical_json_bytes(extension).decode("utf-8")
        resolution_cid = content_identity(
            {
                "typed_retry_cooldown": extension,
                "started_at_ms": now_ms,
            }
        )
        parameters = {
            **extension,
            "schema": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_SCHEMA,
            "operation": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_COMMAND,
            "expected_task_status": "blocked",
            "terminal_operation": prior["operation"],
            "terminal_reason": terminal_reason,
            "source_completion_receipt_id": (
                source_completion_receipt_id
            ),
            **normalized_references,
            "execution_route_binding_cid": route_binding_cid,
            "execution_route_policy_id": route_policy_id,
            "execution_route_origin_revision": route_origin_revision,
            "fresh_attempt_number": fresh_attempt_number,
            "max_task_attempts_before": max_task_attempts_before,
            "max_task_attempts_after": max_task_attempts_after,
            "attempt_refunded": False,
            "reason": reason,
            "resolution_cid": resolution_cid,
            "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "extension_json": extension_json,
            "status": "retrying",
            "body_json": body_json,
        }
        command_digest = hashlib.sha256(
            canonical_json_bytes(parameters)
        ).hexdigest()
        session = self._require_session()
        live = self.load_generation()
        command = StateCommand(
            command_id=f"cmd:blocked-retry-recovery:{command_digest}",
            command_kind=CommandKind.CLAIM,
            store_id=self.store_id,
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key=(
                f"executor-blocked-retry-recovery:{command_digest}"
            ),
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters=parameters,
        )

        def apply_recovery(
            txn: StateTransaction,
            active: StateCommand,
            generation: StoreGeneration,
        ) -> Mapping[str, Any]:
            values = dict(active.parameters)
            observed = _fetch_all(
                txn.execute_named_operation(
                    "executor_retry_cooldown_by_task",
                    (values["task_cid"],),
                )
            )
            if observed:
                raise OptimisticConflictError(
                    "blocked retry recovery cooldown absence became stale"
                )
            queue_result = txn.execute_named_operation(
                "executor_insert_retry_cooldown",
                (
                    values["task_cid"],
                    values["claim_id"],
                    values["resolution_cid"],
                    values["owner_session_id"],
                    values["fence_epoch"],
                    values["fencing_token"],
                    0,
                    values["attempt_number"],
                    "released",
                    values["started_at_ms"],
                    values["reason"],
                    values["retry_not_before_ms"],
                    values["owner_session_id"],
                    values["fence_epoch"],
                    1,
                    values["extension_schema"],
                    values["extension_json"],
                    -1,
                ),
            )
            if _fetch_one(queue_result) is None:
                raise OptimisticConflictError(
                    "blocked retry recovery cooldown absence CAS failed"
                )
            expected = int(values["expected_task_revision"])
            task_result = txn.execute_named_operation(
                "executor_cas_task_status_receipt",
                (
                    "retrying",
                    expected + 1,
                    self._clock(),
                    values["body_json"],
                    values["task_cid"],
                    expected,
                ),
            )
            if _fetch_one(task_result) is None:
                raise OptimisticConflictError(
                    "blocked retry recovery task CAS failed"
                )
            return {
                "schema": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_SCHEMA,
                "operation": TYPED_DATABASE_BLOCKED_RETRY_RECOVERY_COMMAND,
                "task_cid": values["task_cid"],
                "attempt_id": values["attempt_id"],
                "attempt_number": values["attempt_number"],
                "fresh_attempt_number": values["fresh_attempt_number"],
                "task_revision": expected + 1,
                "queue_revision": 1,
                "retry_not_before_ms": values["retry_not_before_ms"],
                "source_completion_receipt_id": values[
                    "source_completion_receipt_id"
                ],
                "operator_handoff_receipt_id": values[
                    "operator_handoff_receipt_id"
                ],
                "sidecar_evidence_id": values["sidecar_evidence_id"],
                "max_task_attempts_before": values[
                    "max_task_attempts_before"
                ],
                "max_task_attempts_after": values[
                    "max_task_attempts_after"
                ],
                "attempt_refunded": False,
                "execution_route_binding_cid": values[
                    "execution_route_binding_cid"
                ],
                "execution_route_policy_id": values[
                    "execution_route_policy_id"
                ],
                "execution_route_origin_revision": values[
                    "execution_route_origin_revision"
                ],
                "store_revision_before": generation.revision,
            }

        return self.submit_command(command, apply=apply_recovery)

    def record_task_retry_cooldown(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        expected_task_status: str,
        attempt_id: str,
        claim_id: str,
        lease_id: str,
        owner_session_id: str,
        attempt_number: int,
        fencing_token: int,
        fence_epoch: int,
        delay_ms: int,
        reason: str,
        selection_penalty: int = 0,
        now_ms: int | None = None,
    ) -> CASResult:
        """Record one task-revision and claim-bound retry cooldown.

        The queue row is written only by the exclusive typed owner.  Both an
        expected absence and a replacement of an older typed row are explicit
        CAS states; an untyped or newer row is never overwritten, while an
        exact same-attempt replay reproduces the original command identity.
        """

        text_fields = {
            "task_cid": task_cid,
            "attempt_id": attempt_id,
            "claim_id": claim_id,
            "lease_id": lease_id,
            "owner_session_id": owner_session_id,
            "reason": reason,
        }
        normalized: dict[str, str] = {}
        for name, value in text_fields.items():
            selected = str(value or "").strip()
            maximum = 2_048 if name == "reason" else 1_024
            if (
                not selected
                or len(selected.encode("utf-8")) > maximum
                or any(marker in selected for marker in ("\x00", "\n", "\r"))
            ):
                raise QuackClientError(f"retry cooldown {name} is invalid")
            normalized[name] = selected
        expected_status = str(expected_task_status or "").strip().lower()
        if expected_status == "blocked":
            raise QuackClientError(
                "typed blocked recovery requires coordination-coupled owner "
                "authority"
            )
        if expected_status not in {"in_progress", "retrying"}:
            raise QuackClientError(
                "retry cooldown requires an exact claimed control state"
            )

        positive_values = {
            "attempt_number": attempt_number,
            "fencing_token": fencing_token,
            "fence_epoch": fence_epoch,
        }
        if (
            isinstance(expected_task_revision, bool)
            or not isinstance(expected_task_revision, int)
            or expected_task_revision < 0
        ):
            raise QuackClientError(
                "retry cooldown expected_task_revision is invalid"
            )
        normalized_ints: dict[str, int] = {
            "expected_task_revision": int(expected_task_revision)
        }
        for name, value in positive_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise QuackClientError(f"retry cooldown {name} is invalid")
            normalized_ints[name] = int(value)
        if (
            isinstance(delay_ms, bool)
            or not isinstance(delay_ms, int)
            or not 0 <= delay_ms <= 86_400_000
        ):
            raise QuackClientError("retry cooldown delay_ms is outside its bound")
        if (
            isinstance(selection_penalty, bool)
            or not isinstance(selection_penalty, int)
            or not 0 <= selection_penalty <= 1_000_000
        ):
            raise QuackClientError(
                "retry cooldown selection_penalty is outside its bound"
            )
        started_at_ms = int(time.time() * 1_000) if now_ms is None else now_ms
        if (
            isinstance(started_at_ms, bool)
            or not isinstance(started_at_ms, int)
            or started_at_ms < 0
        ):
            raise QuackClientError("retry cooldown now_ms is invalid")
        retry_not_before_ms = int(started_at_ms) + int(delay_ms)

        prior_rows = self.execute(
            "executor_retry_cooldown_by_task",
            {"task_cid": normalized["task_cid"]},
        )
        if len(prior_rows) > 1:
            raise QuackClientError("retry cooldown queue identity is ambiguous")
        prior: dict[str, Any] = {}
        if prior_rows:
            try:
                prior = _validated_stored_retry_cooldown(
                    prior_rows[0],
                    task_cid=normalized["task_cid"],
                )
            except TypedStateOwnerError as exc:
                raise QuackClientError(
                    "retry cooldown prior queue state is malformed"
                ) from exc
        expected_queue_revision = -1
        expected_queue_attempt = 0
        if prior:
            prior_revision = int(prior["revision"])
            prior_attempt = int(prior["attempt"])
            prior_extension = dict(prior["extension"])
            if prior_attempt == normalized_ints["attempt_number"]:
                replay_identity = {
                    "task_cid": normalized["task_cid"],
                    "expected_task_revision": normalized_ints[
                        "expected_task_revision"
                    ],
                    "attempt_id": normalized["attempt_id"],
                    "claim_id": normalized["claim_id"],
                    "lease_id": normalized["lease_id"],
                    "owner_session_id": normalized["owner_session_id"],
                    "attempt_number": normalized_ints["attempt_number"],
                    "fencing_token": normalized_ints["fencing_token"],
                    "fence_epoch": normalized_ints["fence_epoch"],
                    "selection_penalty": int(selection_penalty),
                    "consecutive_failures": normalized_ints["attempt_number"],
                    "reason": normalized["reason"],
                    "delay_ms": int(delay_ms),
                }
                if any(
                    prior_extension.get(name) != expected
                    for name, expected in replay_identity.items()
                ):
                    raise QuackClientError(
                        "retry cooldown same-attempt replay identity differs"
                    )
                try:
                    started_at_ms = int(prior_extension["started_at_ms"])
                    retry_not_before_ms = int(
                        prior_extension["retry_not_before_ms"]
                    )
                    expected_queue_revision = int(
                        prior_extension["expected_queue_revision"]
                    )
                    expected_queue_attempt = int(
                        prior_extension["expected_queue_attempt"]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise QuackClientError(
                        "retry cooldown replay receipt is malformed"
                    ) from exc
            elif prior_attempt < normalized_ints["attempt_number"]:
                expected_queue_revision = prior_revision
                expected_queue_attempt = prior_attempt
            else:
                raise QuackClientError(
                    "retry cooldown refuses a newer queue row"
                )

        extension = {
            "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
            "task_cid": normalized["task_cid"],
            "expected_task_revision": normalized_ints["expected_task_revision"],
            "attempt_id": normalized["attempt_id"],
            "claim_id": normalized["claim_id"],
            "lease_id": normalized["lease_id"],
            "owner_session_id": normalized["owner_session_id"],
            "attempt_number": normalized_ints["attempt_number"],
            "fencing_token": normalized_ints["fencing_token"],
            "fence_epoch": normalized_ints["fence_epoch"],
            "delay_ms": int(delay_ms),
            "started_at_ms": int(started_at_ms),
            "retry_not_before_ms": retry_not_before_ms,
            "selection_penalty": int(selection_penalty),
            "consecutive_failures": normalized_ints["attempt_number"],
            "reason": normalized["reason"],
            "expected_queue_revision": expected_queue_revision,
            "expected_queue_attempt": expected_queue_attempt,
        }
        extension_json = canonical_json_bytes(extension).decode("utf-8")
        resolution_cid = content_identity(
            {
                "typed_retry_cooldown": extension,
                "started_at_ms": int(started_at_ms),
            }
        )
        material = {
            **extension,
            "operation": "task.retry.cooldown.record",
            "expected_task_status": expected_status,
            "resolution_cid": resolution_cid,
        }
        command_digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        session = self._require_session()
        live = self.load_generation()
        command = StateCommand(
            command_id=f"cmd:retry-cooldown:{command_digest}",
            command_kind=CommandKind.APPEND,
            store_id=self.store_id,
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key=f"executor-retry-cooldown:{command_digest}",
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters={
                **material,
                "extension_schema": TYPED_RETRY_COOLDOWN_SCHEMA,
                "extension_json": extension_json,
            },
        )

        def apply_retry_cooldown(
            txn: StateTransaction,
            active: StateCommand,
            generation: StoreGeneration,
        ) -> Mapping[str, Any]:
            values = dict(active.parameters)
            observed_result = txn.execute_named_operation(
                "executor_retry_cooldown_by_task",
                (values["task_cid"],),
            )
            observed_rows = _fetch_all(observed_result)
            if len(observed_rows) > 1:
                raise OptimisticConflictError(
                    "retry cooldown queue identity became ambiguous"
                )
            observed = (
                _row_mapping(_result_columns(observed_result), observed_rows[0])
                if observed_rows
                else {}
            )
            expected_revision = int(values["expected_queue_revision"])
            expected_attempt = int(values["expected_queue_attempt"])
            if (
                (expected_revision == -1 and observed)
                or (expected_revision >= 0 and not observed)
                or (
                    observed
                    and (
                        int(observed.get("revision") or -1) != expected_revision
                        or int(observed.get("attempt") or -1) != expected_attempt
                        or str(observed.get("extension_schema") or "")
                        != TYPED_RETRY_COOLDOWN_SCHEMA
                    )
                )
            ):
                raise OptimisticConflictError(
                    "retry cooldown expected queue revision is stale"
                )
            new_queue_revision = 1 if expected_revision == -1 else expected_revision + 1
            common_values = (
                values["claim_id"],
                values["resolution_cid"],
                values["owner_session_id"],
                values["fence_epoch"],
                values["fencing_token"],
                0,
                values["attempt_number"],
                "released",
                values["started_at_ms"],
                values["reason"],
                values["retry_not_before_ms"],
                values["owner_session_id"],
                values["fence_epoch"],
                new_queue_revision,
                values["extension_schema"],
                values["extension_json"],
            )
            if expected_revision == -1:
                operation = "executor_insert_retry_cooldown"
                mutation_parameters = (
                    values["task_cid"],
                    *common_values,
                    expected_revision,
                )
            else:
                operation = "executor_update_retry_cooldown"
                mutation_parameters = (
                    *common_values,
                    values["task_cid"],
                    expected_revision,
                    expected_attempt,
                    values["attempt_number"],
                    TYPED_RETRY_COOLDOWN_SCHEMA,
                )
            result = txn.execute_named_operation(
                operation,
                mutation_parameters,
            )
            row = _fetch_one(result)
            if row is None:
                raise OptimisticConflictError(
                    "retry cooldown absence/revision CAS failed"
                )
            written = _row_mapping(_result_columns(result), row)
            expected_written = {
                "task_cid": values["task_cid"],
                "claim_cid": values["claim_id"],
                "resolution_cid": values["resolution_cid"],
                "claimant_did": values["owner_session_id"],
                "logical_epoch": values["fence_epoch"],
                "fencing_token": values["fencing_token"],
                "expires_at_ms": 0,
                "attempt": values["attempt_number"],
                "state": "released",
                "started_at_ms": values["started_at_ms"],
                "release_reason": values["reason"],
                "retry_not_before_ms": values["retry_not_before_ms"],
                "owner_session_id": values["owner_session_id"],
                "fence_epoch": values["fence_epoch"],
                "revision": new_queue_revision,
                "extension_schema": values["extension_schema"],
                "extension_json": values["extension_json"],
            }
            if any(
                written.get(name) != expected
                for name, expected in expected_written.items()
            ):
                raise OptimisticConflictError(
                    "retry cooldown mutation returned inconsistent state"
                )
            return {
                "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
                "operation": "task.retry.cooldown.record",
                "task_cid": str(values["task_cid"]),
                "expected_task_revision": int(values["expected_task_revision"]),
                "attempt_id": str(values["attempt_id"]),
                "claim_id": str(values["claim_id"]),
                "attempt_number": int(values["attempt_number"]),
                "queue_revision": new_queue_revision,
                "retry_not_before_ms": int(values["retry_not_before_ms"]),
                "reason": str(values["reason"]),
                "store_revision_before": generation.revision,
            }

        return self.submit_command(command, apply=apply_retry_cooldown)

    def record_task_validation(
        self,
        *,
        task_cid: str,
        outcome: str,
        evidence_digest: str,
        argv: Sequence[str] | None = None,
        attempt_id: str = "",
        body: Mapping[str, Any] | None = None,
        idempotency_key: str,
        command_id: str | None = None,
    ) -> CASResult:
        """Record one bounded validation result through an admitted command."""

        task = str(task_cid or "").strip()
        selected_outcome = str(outcome or "").strip().lower()
        digest = str(evidence_digest or "").strip().lower()
        if not task:
            raise QuackClientError("validation task_cid is required")
        if selected_outcome not in {"passed", "failed", "error", "skipped"}:
            raise QuackClientError("validation outcome is outside the closed vocabulary")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise QuackClientError("validation evidence_digest must be sha256")
        command_argv = tuple(str(item) for item in (argv or ("database-validation",)))
        if not command_argv or len(command_argv) > 128:
            raise QuackClientError("validation argv is empty or exceeds its bound")
        if any(len(item.encode("utf-8")) > 8_192 for item in command_argv):
            raise QuackClientError("validation argv item exceeds its byte bound")
        result_body = dict(body or {})
        now = self._clock()
        material = {
            "task_cid": task,
            "attempt_id": str(attempt_id or ""),
            "outcome": selected_outcome,
            "evidence_digest": digest,
            "argv": list(command_argv),
            "body": result_body,
            "idempotency_key": str(idempotency_key),
        }
        run_id = content_identity({"validation_run": material})
        result_id = content_identity({"validation_result": material})
        evidence_id = content_identity({"validation_evidence": material})
        effective_attempt_id = str(attempt_id or f"validation:{run_id}")
        command_digest = "sha256:" + hashlib.sha256(
            canonical_json_bytes(list(command_argv))
        ).hexdigest()
        run_body_json = canonical_json_bytes(
            {"argv": list(command_argv), "body": result_body}
        ).decode("utf-8")
        result_body_json = canonical_json_bytes(result_body).decode("utf-8")
        evidence_body_json = canonical_json_bytes(
            {
                "outcome": selected_outcome,
                "run_id": run_id,
                "result_id": result_id,
                "body": result_body,
            }
        ).decode("utf-8")
        operation = (
            "task.validation.record.passed"
            if selected_outcome == "passed"
            else "task.validation.record.nonpassing"
        )
        session = self._require_session()
        live = self.load_generation()
        parameters = {
            "operation": operation,
            "task_cid": task,
            "run_id": run_id,
            "result_id": result_id,
            "evidence_id": evidence_id,
            "attempt_id": effective_attempt_id,
            "outcome": selected_outcome,
            "evidence_digest": digest,
            "started_at": now,
            "finished_at": now,
            "command_digest": command_digest,
            "run_body_json": run_body_json,
            "result_body_json": result_body_json,
            "evidence_body_json": evidence_body_json,
        }
        command = StateCommand(
            command_id=command_id or f"cmd:validation:{result_id}",
            command_kind=CommandKind.APPEND,
            store_id=self.store_id,
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key=idempotency_key,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters=parameters,
        )

        def apply_validation(
            txn: StateTransaction,
            active: StateCommand,
            generation: StoreGeneration,
        ) -> Mapping[str, Any]:
            values = dict(active.parameters)
            txn.execute_named_operation(
                "executor_insert_validation_run",
                (
                    values["run_id"],
                    values["task_cid"],
                    values["attempt_id"],
                    values["started_at"],
                    values["finished_at"],
                    values["outcome"],
                    values["command_digest"],
                    values["run_body_json"],
                ),
            )
            txn.execute_named_operation(
                "executor_insert_validation_result",
                (
                    values["result_id"],
                    values["run_id"],
                    values["task_cid"],
                    0,
                    values["outcome"],
                    values["evidence_digest"],
                    values["result_body_json"],
                ),
            )
            if values["outcome"] == "passed":
                txn.execute_named_operation(
                    "executor_insert_validation_evidence",
                    (
                        values["evidence_id"],
                        "",
                        values["task_cid"],
                        "validation",
                        values["evidence_digest"],
                        values["finished_at"],
                        values["evidence_body_json"],
                    ),
                )
            return {
                "task_cid": str(values["task_cid"]),
                "run_id": str(values["run_id"]),
                "result_id": str(values["result_id"]),
                "evidence_id": (
                    str(values["evidence_id"])
                    if values["outcome"] == "passed"
                    else ""
                ),
                "outcome": str(values["outcome"]),
                "store_revision_before": generation.revision,
            }

        return self.submit_command(command, apply=apply_validation)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_adapter(self) -> _ConnectionAdapter:
        if self._closed:
            raise QuackClientError("client is closed")
        if self._adapter is None:
            raise QuackClientError("client is not attached")
        return self._adapter

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise QuackClientError("client is not attached")
        return self._session

    def _open_connection(self, endpoint: QuackEndpoint) -> _ConnectionAdapter:
        if self._connection_factory is not None:
            connection = self._connection_factory(endpoint)
            return _ConnectionAdapter(
                connection,
                embedded_operations=(
                    self._templates
                    if endpoint.mode is TransportMode.EMBEDDED
                    else None
                ),
            )
        if endpoint.mode is TransportMode.EMBEDDED:
            if endpoint.database_path is None:
                raise QuackClientError("embedded endpoint requires a database path")
            connection = open_duckdb_connection(
                endpoint.database_path,
                timeout_seconds=self.connect_timeout_seconds,
            )
            return _ConnectionAdapter(
                connection,
                embedded_operations=self._templates,
            )
        return self._open_quack_connection(endpoint)

    def _open_quack_connection(self, endpoint: QuackEndpoint) -> _ConnectionAdapter:
        uri = endpoint.quack_uri or endpoint.target
        if not self._is_loopback_quack_uri(uri):
            raise QuackClientError(
                "non-loopback Quack bind requires a separately reviewed policy"
            )
        try:
            # The Quack state owner exposes a closed server-side operation
            # catalog over its authenticated local command socket.  Clients do
            # not receive a generic READ_WRITE ATTACH and cannot submit SQL.
            connection = open_typed_state_owner_connection(
                store_id=self.store_id,
                client_id=self.owner_id,
                process_birth_id=self.process_birth_id,
                timeout_seconds=self.connect_timeout_seconds,
            )
            return _ConnectionAdapter(connection)
        except (OSError, TypedStateOwnerError) as exc:
            raise QuackClientTransportError(
                "failed to attach the typed Quack state-owner boundary "
                f"({type(exc).__name__})"
            ) from exc

    @staticmethod
    def _is_loopback_quack_uri(uri: str) -> bool:
        text = str(uri or "").strip().lower()
        if not text.startswith("quack:"):
            return False
        rest = text[len("quack:") :]
        if rest.startswith("//"):
            rest = rest[2:]
        host = rest.split(":", 1)[0].split("/", 1)[0]
        return host in {"127.0.0.1", "localhost", "::1"}

    @staticmethod
    def _validated_quack_uri_literal(uri: str) -> str:
        text = str(uri or "").strip()
        if not re.fullmatch(
            r"quack:(?://)?(?:127\.0\.0\.1|localhost|::1):\d{1,5}",
            text,
            flags=re.IGNORECASE,
        ):
            raise QuackClientError(f"invalid or non-loopback quack URI: {uri!r}")
        if "'" in text or ";" in text or "\x00" in text:
            raise QuackClientError("quack URI contains forbidden characters")
        return text

    def _execute_template(
        self,
        adapter: _ConnectionAdapter,
        template_name: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None,
    ) -> tuple[Mapping[str, Any], ...]:
        template = self.get_template(template_name)
        bound = template.bind(parameters)
        try:
            execute_operation = getattr(adapter.raw, "execute_operation", None)
            if callable(execute_operation):
                result = execute_operation(template.name, bound)
            else:
                result = adapter.execute(template.sql, bound if bound else None)
        except QuackClientError:
            raise
        except Exception as exc:
            raise QuackClientTransportError(
                f"template {template.name} failed: {exc}"
            ) from exc
        columns = _result_columns(result)
        rows = _fetch_all(result)
        if not rows:
            return tuple()
        if not columns:
            # Prefer mapping rows (DuckDBRow); otherwise treat as DML with no projection.
            if isinstance(rows[0], Mapping):
                return tuple(_row_mapping((), row) for row in rows)
            return tuple()
        return tuple(_row_mapping(columns, row) for row in rows)

    def _load_generation(self, adapter: _ConnectionAdapter) -> StoreGeneration:
        rows = self._execute_template(adapter, "load_store_generation", None)
        if not rows:
            raise StaleGenerationError(
                "store generation is missing; seed or migrate the database first"
            )
        row = rows[0]
        return StoreGeneration(
            store_id=self.store_id,
            generation=int(row["generation"]),
            schema_revision=int(row["schema_revision"]),
            fence_epoch=int(row["fence_epoch"]),
            revision=int(row["revision"]),
            database_uuid=str(row["database_uuid"]),
            birth_id=str(row.get("birth_id") or ""),
        )

    def _seed_generation_if_missing(self, adapter: _ConnectionAdapter) -> None:
        rows = self._execute_template(adapter, "load_store_generation", None)
        if rows:
            return
        meta = {
            item["key"]: item["value"]
            for item in self._execute_template(adapter, "whoami_metadata", None)
        }
        database_uuid = str(meta.get("database_uuid") or str(uuid.uuid4()))
        try:
            schema_revision = int(meta.get("schema_version") or 1)
        except (TypeError, ValueError) as exc:
            raise QuackClientIdentityError(
                "control-plane schema_version metadata is not an integer"
            ) from exc
        if schema_revision < 1:
            raise QuackClientIdentityError(
                "control-plane schema_version metadata must be positive"
            )
        self._execute_template(
            adapter,
            "seed_store_generation",
            {
                "generation": 1,
                "schema_revision": schema_revision,
                "fence_epoch": 1,
                "revision": 0,
                "database_uuid": database_uuid,
                "birth_id": self.process_birth_id,
                "created_at": self._clock(),
            },
        )
        adapter.commit()

    def _observe_store_identity(
        self,
        adapter: _ConnectionAdapter,
        generation: StoreGeneration,
    ) -> ControlPlaneStoreIdentity:
        meta_rows = self._execute_template(adapter, "whoami_metadata", None)
        meta = {str(row["key"]): str(row["value"]) for row in meta_rows}
        schema_fingerprint = _schema_fingerprint_digest(
            str(meta.get("schema_fingerprint") or "")
        )
        if not schema_fingerprint:
            # Derive a stable fingerprint from available identity material so
            # hermetic stores without migration metadata still verify.
            material = {
                "database_uuid": generation.database_uuid,
                "schema_revision": generation.schema_revision,
                "store_id": self.store_id,
            }
            digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
            schema_fingerprint = f"sha256:{digest}"
        extension_fingerprint = ""
        if self.expected_identity is not None:
            extension_fingerprint = self.expected_identity.extension_fingerprint
        return ControlPlaneStoreIdentity(
            repository_id=(
                self.expected_identity.repository_id
                if self.expected_identity is not None
                else "repository:local"
            ),
            database_uuid=generation.database_uuid,
            store_id=self.store_id,
            schema_revision=generation.schema_revision,
            generation=generation.generation,
            schema_fingerprint=schema_fingerprint,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            server_birth_id=generation.birth_id,
            extension_fingerprint=extension_fingerprint,
        )

    def _verify_identity(
        self,
        expected: ControlPlaneStoreIdentity,
        observed: ControlPlaneStoreIdentity,
        generation: StoreGeneration,
    ) -> None:
        if expected.store_id != observed.store_id:
            raise QuackClientIdentityError(
                f"store_id mismatch: expected {expected.store_id}, "
                f"observed {observed.store_id}"
            )
        if expected.database_uuid != observed.database_uuid:
            raise QuackClientIdentityError(
                f"database_uuid mismatch: expected {expected.database_uuid}, "
                f"observed {observed.database_uuid}"
            )
        if expected.schema_fingerprint and (
            expected.schema_fingerprint != observed.schema_fingerprint
        ):
            raise QuackClientIdentityError(
                "schema_fingerprint mismatch between client expectation and store"
            )
        if expected.generation and expected.generation != generation.generation:
            raise QuackClientIdentityError(
                f"generation mismatch: expected {expected.generation}, "
                f"observed {generation.generation}"
            )
        if expected.extension_fingerprint and observed.extension_fingerprint:
            if expected.extension_fingerprint != observed.extension_fingerprint:
                raise QuackClientIdentityError(
                    "extension_fingerprint mismatch; refuse mismatched Quack peer"
                )

    def _default_task_status_apply(
        self,
        txn: StateTransaction,
        command: StateCommand,
        live: StoreGeneration,
    ) -> Mapping[str, Any]:
        params = dict(command.parameters)
        task_cid = str(params.get("task_cid") or "").strip()
        status = str(params.get("status") or "").strip()
        expected_task_revision = params.get("expected_task_revision")
        if not task_cid or not status:
            raise QuackClientError("task_cid and status parameters are required")
        if (
            isinstance(expected_task_revision, bool)
            or not isinstance(expected_task_revision, int)
            or expected_task_revision < 0
        ):
            raise QuackClientError("expected_task_revision must be a non-negative int")
        _reject_protected_typed_deferral_reopen(
            txn,
            task_cid=task_cid,
            requested_status=status,
        )
        new_revision = txn.cas_row_revision(
            table="tasks",
            key_column="task_cid",
            key_value=task_cid,
            expected_revision=int(expected_task_revision),
            assignments={
                "status": status,
                "updated_at": self._clock(),
            },
        )
        return {
            "task_cid": task_cid,
            "status": status,
            "task_revision": new_revision,
            "store_revision_before": live.revision,
            "command_id": command.command_id,
        }


def open_embedded_client(
    database_path: str | Path,
    *,
    owner_id: str,
    store_id: str = DEFAULT_STORE_ID,
    expected_identity: ControlPlaneStoreIdentity | None = None,
    seed_generation: bool = True,
    retry_policy: RetryPolicy | None = None,
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> QuackStateClient:
    """Attach an embedded client to ``database_path`` and return it open."""

    client = QuackStateClient(
        owner_id=owner_id,
        store_id=store_id,
        expected_identity=expected_identity,
        retry_policy=retry_policy,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    client.attach(
        database_path,
        mode=TransportMode.EMBEDDED,
        seed_generation=seed_generation,
        expected_identity=expected_identity,
    )
    return client


__all__ = [
    "CLIENT_SESSION_SCHEMA",
    "DEFAULT_STATEMENT_TEMPLATES",
    "DEFAULT_STORE_ID",
    "PAGE_RESULT_SCHEMA",
    "QUACK_STATE_CLIENT_INTERFACE",
    "QUACK_STATE_CLIENT_SCHEMA",
    "ClientSession",
    "PageResult",
    "QuackClientError",
    "QuackClientIdentityError",
    "QuackClientSQLError",
    "QuackClientTransportError",
    "QuackEndpoint",
    "QuackStateClient",
    "StatementKind",
    "StatementTemplate",
    "TransportMode",
    "open_embedded_client",
    "resolve_endpoint",
]
