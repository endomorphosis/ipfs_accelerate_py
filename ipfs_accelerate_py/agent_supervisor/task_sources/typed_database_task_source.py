"""Database task-source projection over the closed typed state-owner gateway.

This adapter is intentionally not a remote DuckDB compatibility layer.  It
uses only the fixed named operations registered in ``QuackStateClient`` and
the birth-bound grant installed for one managed implementation daemon.  No
database path, ATTACH credential, SQL text, or generic query surface crosses
the process boundary.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import (
    ControlPlaneStoreIdentity,
    StoreGeneration,
    canonical_json_bytes,
    content_identity,
)
from .database_task_source import (
    DATABASE_TASK_SOURCE_SCHEMA,
    TYPED_DEFERRAL_BUDGET_BLOCK_OPERATION,
    TaskPage,
    TaskRecord,
    TaskSourceBoundsError,
    TaskSourceConflictError,
    TaskSourceIntegrityError,
    TaskSourceSnapshot,
)
from .database_task_source import (
    MAX_QUERY_LIMIT as TASK_SOURCE_MAX_QUERY_LIMIT,
)
from .database_task_source import (
    CASResult as DatabaseCASResult,
)
from .intent_repository import (
    MAX_PLAN_PROJECTION_BYTES,
    MAX_PROJECTION_RECORDS,
    TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
    IntentReceipt,
    QueueEntry,
)
from .quack_state_client import ClientSession, QuackStateClient, TransportMode
from .state_owner_bootstrap import StateOwnerBootstrapCredentials
from .task_execution_route_policy import (
    TaskExecutionRouteBinding,
    TaskExecutionRoutePolicy,
    task_execution_contract_cid,
)
from .typed_state_owner import (
    TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
    TYPED_DATABASE_CLAIM_RECOVERY_OPERATION,
    TYPED_DATABASE_CLAIM_RECOVERY_SCHEMA,
    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
    TYPED_DATABASE_STRICT_RESUME_REQUEUE_OPERATION,
    TYPED_RETRY_COOLDOWN_SCHEMA,
    TYPED_RETRYING_RECEIPT_OPERATIONS,
    TYPED_TASK_STATUS_VOCABULARY,
    TypedStateOwnerConnection,
    TypedStateOwnerError,
    _validated_database_strict_resume_rejection_receipt,
    _validated_stored_retry_cooldown,
)

TYPED_DATABASE_TASK_SOURCE_INTERFACE: Final = "TypedDatabaseTaskSource@1"
TYPED_DATABASE_TASK_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-task-source@1"
)
DEFAULT_QUERY_LIMIT: Final = 50
_TRANSPORT_PAGE_LIMIT: Final = 500
_TASK_HISTORY_PAGE_LIMIT: Final = 16
_MAX_JSON_BYTES: Final = 262_144
_READY_STATUSES: Final[frozenset[str]] = frozenset(
    {"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"}
)
_COMPLETED_STATUSES: Final[frozenset[str]] = frozenset({"completed", "skipped", "complete", "done"})
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {
        *_COMPLETED_STATUSES,
        "cancelled",
        "canceled",
        "failed",
        "quarantined",
        "rejected",
    }
)
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
_DAEMON_REQUIRED_OWNER_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "whoami_metadata",
        "load_store_generation",
        "executor_task_projection_page",
        "executor_control_snapshot",
        "executor_task_projection_by_identity",
        "executor_task_revision_history_by_cid",
        "executor_retry_cooldown_by_task",
        "executor_retry_cooldown_page",
        "txn_load_generation",
        "txn_lookup_idempotency",
        "txn_advance_store_revision",
        "txn_record_idempotency",
        "txn_cas_task_status",
        "executor_cas_task_status_receipt",
        "executor_insert_task_revision",
        "executor_insert_retry_cooldown",
        "executor_update_retry_cooldown",
        "executor_insert_validation_run",
        "executor_insert_validation_result",
        "executor_insert_validation_evidence",
    }
)
_DAEMON_REQUIRED_OWNER_COMMAND_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "task.status.cas",
        "task.status.cas.receipt",
        "task.retry.cooldown.record",
        "task.claim.reservation.recover",
        "task.validation.record.passed",
        "task.validation.record.nonpassing",
    }
)


def _bounded_json(value: Any, *, noun: str) -> Any:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TaskSourceIntegrityError(f"{noun} is not encoded JSON")
    if len(value.encode("utf-8")) > _MAX_JSON_BYTES:
        raise TaskSourceBoundsError(f"{noun} exceeds its byte bound")
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TaskSourceIntegrityError(f"{noun} is malformed") from exc


def _mapping_json(value: Any, *, noun: str) -> dict[str, Any]:
    parsed = _bounded_json(value, noun=noun)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        raise TaskSourceIntegrityError(f"{noun} is not a JSON object")
    return parsed


def _list_json(value: Any, *, noun: str) -> list[Any]:
    parsed = _bounded_json(value, noun=noun)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise TaskSourceIntegrityError(f"{noun} is not a JSON array")
    return parsed


def _record_from_row(row: Mapping[str, Any]) -> tuple[TaskRecord, Mapping[str, Any]]:
    required = {
        "task_cid",
        "task_alias",
        "goal_cid",
        "plan_cid",
        "objective_id",
        "ordinal",
        "status",
        "revision",
        "priority",
        "identity_json",
        "body_json",
        "dependencies_json",
        "outputs_json",
        "acceptance_json",
        "validations_json",
    }
    if set(row) != required:
        raise TaskSourceIntegrityError("typed task projection differs from its schema")
    identity = _mapping_json(row["identity_json"], noun="task identity")
    body = _mapping_json(row["body_json"], noun="task body")
    dependencies_raw = _list_json(row["dependencies_json"], noun="task dependencies")
    if any(not isinstance(item, str) or not item for item in dependencies_raw):
        raise TaskSourceIntegrityError("task dependencies contain a non-identity")

    outputs: list[Mapping[str, Any]] = []
    for item in _list_json(row["outputs_json"], noun="task outputs"):
        if not isinstance(item, dict):
            raise TaskSourceIntegrityError("task output is not an object")
        normalized = dict(item)
        effect = normalized.get("effect")
        if isinstance(effect, str):
            normalized["effect"] = _mapping_json(effect, noun="task output effect")
        outputs.append(MappingProxyType(normalized))

    acceptance: list[Mapping[str, Any]] = []
    for item in _list_json(row["acceptance_json"], noun="task acceptance"):
        if not isinstance(item, dict):
            raise TaskSourceIntegrityError("task acceptance item is not an object")
        normalized = dict(item)
        policy = normalized.get("evidence_policy")
        if isinstance(policy, str):
            normalized["evidence_policy"] = _mapping_json(policy, noun="acceptance evidence policy")
        acceptance.append(MappingProxyType(normalized))

    validations: list[Mapping[str, Any]] = []
    for item in _list_json(row["validations_json"], noun="task validations"):
        if not isinstance(item, dict):
            raise TaskSourceIntegrityError("task validation item is not an object")
        normalized = dict(item)
        argv = normalized.get("argv")
        if isinstance(argv, str):
            parsed_argv = _list_json(argv, noun="task validation argv")
            if any(not isinstance(value, str) for value in parsed_argv):
                raise TaskSourceIntegrityError("task validation argv is malformed")
            normalized["argv"] = parsed_argv
        policy = normalized.get("policy")
        if isinstance(policy, str):
            normalized["policy"] = _mapping_json(policy, noun="task validation policy")
        validations.append(MappingProxyType(normalized))

    task_cid = str(row["task_cid"] or "").strip()
    task_alias = str(row["task_alias"] or "").strip()
    if not task_cid or not task_alias:
        raise TaskSourceIntegrityError("task projection lacks canonical identity")
    return (
        TaskRecord(
            task_cid=task_cid,
            task_alias=task_alias,
            goal_cid=str(row["goal_cid"] or ""),
            plan_cid=str(row["plan_cid"] or ""),
            objective_id=str(row["objective_id"] or ""),
            ordinal=int(row["ordinal"]),
            status=str(row["status"] or "").strip().lower(),
            revision=int(row["revision"]),
            priority=str(row["priority"] or ""),
            body=MappingProxyType(body),
            dependencies=tuple(dependencies_raw),
            outputs=tuple(outputs),
            acceptance=tuple(acceptance),
            validations=tuple(validations),
        ),
        MappingProxyType(identity),
    )


def _cursor_encode(revision: int, offset: int) -> str:
    encoded = canonical_json_bytes({"v": 1, "revision": int(revision), "offset": int(offset)})
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _cursor_decode(cursor: str, *, revision: int) -> int:
    text = str(cursor or "").strip()
    if not text:
        return 0
    try:
        payload = json.loads(
            base64.urlsafe_b64decode((text + "=" * (-len(text) % 4)).encode("ascii")).decode(
                "utf-8"
            )
        )
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskSourceConflictError("typed task cursor is malformed") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("v") != 1
        or payload.get("revision") != revision
        or isinstance(payload.get("offset"), bool)
        or not isinstance(payload.get("offset"), int)
        or int(payload["offset"]) < 0
    ):
        raise TaskSourceConflictError("typed task cursor is stale or malformed")
    return int(payload["offset"])


class TypedDatabaseTaskSource:
    """Closed named-operation adapter consumed by DatabaseImplementationDaemon."""

    INTERFACE: ClassVar[str] = TYPED_DATABASE_TASK_SOURCE_INTERFACE
    SCHEMA: ClassVar[str] = TYPED_DATABASE_TASK_SOURCE_SCHEMA

    def __init__(
        self,
        client: QuackStateClient,
        *,
        execution_route_policy: TaskExecutionRoutePolicy
        | Mapping[str, Any]
        | None = None,
        owns_client: bool = True,
        clock_ms: Any | None = None,
    ) -> None:
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise TaskSourceIntegrityError(
                "typed database task source requires an attached QuackStateClient"
            )
        if not isinstance(owns_client, bool):
            raise TaskSourceIntegrityError(
                "typed database task source client ownership is invalid"
            )
        if clock_ms is not None and not callable(clock_ms):
            raise TaskSourceIntegrityError(
                "typed database task source clock is invalid"
            )
        self._client = client
        self._owns_client = owns_client
        self._closed = False
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self.path = Path("typed-state-owner")
        self.database_path = self.path
        self._execution_route_policy = (
            execution_route_policy
            if isinstance(execution_route_policy, TaskExecutionRoutePolicy)
            else TaskExecutionRoutePolicy.from_dict(execution_route_policy)
            if execution_route_policy is not None
            else None
        )
        if self._execution_route_policy is not None:
            self._validate_execution_route_policy_population()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._owns_client:
                self._client.close()

    def __enter__(self) -> TypedDatabaseTaskSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise TaskSourceIntegrityError("typed database task source is closed")

    def require_quack_authority_binding(
        self,
        *,
        expected_endpoint: str,
        expected_process_instance_id: str,
        bootstrap_credentials: StateOwnerBootstrapCredentials | None = None,
    ) -> Mapping[str, Any]:
        """Return the exact attached Quack authority bound to this adapter.

        The database implementation daemon uses this closed check before it
        derives lane-private coordination or execution sidecars.  It proves
        that task authority remains on one attached typed Quack owner; it does
        not expose a database path, token, credential, or generic SQL surface.
        """

        self._require_open()
        endpoint = str(expected_endpoint or "").strip()
        process_instance_id = str(expected_process_instance_id or "").strip()
        client = self._client
        if (
            type(self) is not TypedDatabaseTaskSource
            or type(expected_endpoint) is not str
            or not endpoint.startswith("quack:")
            or type(expected_process_instance_id) is not str
            or not process_instance_id
            or type(client) is not QuackStateClient
            or not client.attached
            or getattr(client, "_connection_factory", None) is not None
        ):
            raise TaskSourceIntegrityError(
                "typed database task source is not bound to the exact Quack authority"
            )
        try:
            live_store_generation = client.load_generation()
        except Exception as exc:
            raise TaskSourceIntegrityError(
                "typed database task source Quack authority is not live"
            ) from exc
        session = client.session if type(client) is QuackStateClient else None
        adapter = (
            getattr(client, "_adapter", None)
            if type(client) is QuackStateClient
            else None
        )
        connection = getattr(adapter, "raw", None)
        grant = (
            getattr(connection, "grant", None)
            if type(connection) is TypedStateOwnerConnection
            else None
        )
        owner_identity = (
            getattr(connection, "identity", None)
            if type(connection) is TypedStateOwnerConnection
            else None
        )
        store_identity = getattr(session, "store_identity", None)
        store_generation = live_store_generation
        route_policy = self._execution_route_policy
        grant_mapping = grant if isinstance(grant, Mapping) else {}
        grant_operations = frozenset(
            str(item) for item in grant_mapping.get("allowed_operations") or ()
        )
        grant_command_operations = frozenset(
            str(item)
            for item in grant_mapping.get("allowed_command_operations") or ()
        )
        grant_entity_scopes = grant_mapping.get("entity_scopes") or {}
        if (
            type(connection) is not TypedStateOwnerConnection
            or not isinstance(owner_identity, Mapping)
            or type(session) is not ClientSession
            or session.transport_mode is not TransportMode.QUACK
            or session.endpoint != endpoint
            or not session.session_id
            or not session.server_id
            or not session.store_id
            or not session.process_birth_id
            or session.store_id != client.store_id
            or session.process_birth_id != client.process_birth_id
            or session.process_birth_id != process_instance_id
            or owner_identity.get("server_id") != session.server_id
            or owner_identity.get("store_id") != session.store_id
            or owner_identity.get("generation") != session.generation
            or owner_identity.get("fence_epoch") != session.fence_epoch
            or type(store_identity) is not ControlPlaneStoreIdentity
            or type(store_generation) is not StoreGeneration
            or store_generation.store_id != session.store_id
            or store_generation.database_uuid != store_identity.database_uuid
            or int(store_generation.generation) != int(session.generation)
            or int(store_generation.fence_epoch) != int(session.fence_epoch)
            or route_policy is None
            or not route_policy.policy_id
            or not isinstance(grant, Mapping)
            or grant.get("client_id") != client.owner_id
            or grant.get("process_birth_id") != process_instance_id
            or grant_operations != _DAEMON_REQUIRED_OWNER_OPERATIONS
            or grant_command_operations
            != _DAEMON_REQUIRED_OWNER_COMMAND_OPERATIONS
            or str(grant.get("tenant_id") or "").strip()
            or str(grant.get("federation_id") or "").strip()
            or not isinstance(grant_entity_scopes, Mapping)
            or bool(grant_entity_scopes)
            or str(grant.get("authority_profile") or "").strip()
        ):
            raise TaskSourceIntegrityError(
                "typed database task source is not bound to the exact Quack authority"
            )
        if bootstrap_credentials is not None and (
            type(bootstrap_credentials) is not StateOwnerBootstrapCredentials
            or bootstrap_credentials.endpoint != endpoint
            or type(bootstrap_credentials.socket_path) is not str
            or connection.bootstrap_socket_path
            != os.path.abspath(bootstrap_credentials.socket_path)
            or bootstrap_credentials.store_id != client.store_id
            or bootstrap_credentials.server_id != session.server_id
            or bootstrap_credentials.client_id != client.owner_id
            or bootstrap_credentials.process_birth_id != process_instance_id
            or type(bootstrap_credentials.token) is not str
            or not hmac.compare_digest(
                connection.bootstrap_token_digest,
                hashlib.sha256(
                    bootstrap_credentials.token.encode("utf-8")
                ).hexdigest(),
            )
            or connection.status_bootstrap
            or bootstrap_credentials.execution_route_policy != route_policy
        ):
            raise TaskSourceIntegrityError(
                "typed task authority differs from its process-bound bootstrap"
            )
        stable_body = {
            "interface": "TypedDatabaseTaskSourceStableQuackAuthority@1",
            "store_id": store_identity.store_id,
            "database_uuid": store_identity.database_uuid,
            "schema_fingerprint": store_identity.schema_fingerprint,
            "repository_id": store_identity.repository_id,
            "schema_revision": int(store_identity.schema_revision),
            "route_policy_id": route_policy.policy_id,
            "plan_root_cid": route_policy.plan_root_cid,
            "repository_tree_id": route_policy.repository_tree_id,
            "source_projection_cid": route_policy.source_projection_cid,
        }
        return MappingProxyType(
            {
                "interface": "TypedDatabaseTaskSourceQuackAuthorityBinding@1",
                "stable_binding_id": content_identity(stable_body),
                "stable_authority": MappingProxyType(stable_body),
                "endpoint": session.endpoint,
                "server_id": session.server_id,
                "session_id": session.session_id,
                "store_id": session.store_id,
                "process_birth_id": session.process_birth_id,
                "generation": int(session.generation),
                "fence_epoch": int(session.fence_epoch),
            }
        )

    def _all_records(
        self, *, expected_count: int
    ) -> tuple[tuple[TaskRecord, Mapping[str, Any]], ...]:
        self._require_open()
        if expected_count < 0 or expected_count > TASK_SOURCE_MAX_QUERY_LIMIT:
            raise TaskSourceBoundsError(
                "typed task population exceeds its admitted projection bound"
            )
        rows: list[Mapping[str, Any]] = []
        while len(rows) < expected_count:
            page = self._client.execute(
                "executor_task_projection_page",
                {
                    "limit": min(_TRANSPORT_PAGE_LIMIT, expected_count - len(rows)),
                    "offset": len(rows),
                },
            )
            if not page:
                break
            rows.extend(page)
        if len(rows) != expected_count:
            raise TaskSourceConflictError(
                "typed task population changed during bounded projection"
            )
        return tuple(_record_from_row(row) for row in rows)

    def _snapshot_material(
        self,
    ) -> tuple[Mapping[str, Any], tuple[tuple[TaskRecord, Mapping[str, Any]], ...], int]:
        for _attempt in range(4):
            before = self._client.load_generation()
            rows = self._client.execute("executor_control_snapshot")
            if len(rows) != 1:
                raise TaskSourceIntegrityError("typed control snapshot is absent or ambiguous")
            row = rows[0]
            records = self._all_records(expected_count=int(row.get("task_count") or 0))
            after = self._client.load_generation()
            if before.content_id == after.content_id:
                if int(row.get("task_count") or 0) != len(records):
                    raise TaskSourceBoundsError(
                        "typed task population exceeds its admitted projection bound"
                    )
                return row, records, after.revision
        raise TaskSourceConflictError("typed control projection changed during bounded snapshot")

    @staticmethod
    def _validated_retry_cooldown_row(
        raw: Mapping[str, Any],
        *,
        task_cid: str,
    ) -> Mapping[str, Any]:
        """Validate every persisted field in one typed cooldown row."""

        task = str(task_cid or "").strip()
        if not task:
            raise TaskSourceIntegrityError("retry cooldown task identity is empty")
        row = dict(raw)
        required = {
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
            "owner_session_id",
            "fence_epoch",
            "revision",
            "extension_schema",
            "extension_json",
        }
        if set(row) != required or str(row.get("task_cid") or "") != task:
            raise TaskSourceIntegrityError(
                "retry cooldown owner row differs from its closed projection"
            )
        try:
            validated = _validated_stored_retry_cooldown(
                row,
                task_cid=task,
            )
        except TypedStateOwnerError as exc:
            raise TaskSourceIntegrityError(
                "retry cooldown row is foreign or differs from its receipt"
            ) from exc
        return MappingProxyType(validated)

    def _retry_cooldown_row(
        self,
        task_cid: str,
    ) -> Mapping[str, Any] | None:
        """Return one complete owner row, including its lease CAS revision."""

        task = str(task_cid or "").strip()
        if not task:
            raise TaskSourceIntegrityError("retry cooldown task identity is empty")
        rows = self._client.execute(
            "executor_retry_cooldown_by_task",
            {"task_cid": task},
        )
        if len(rows) > 1:
            raise TaskSourceIntegrityError(
                "retry cooldown queue identity is ambiguous"
            )
        if not rows:
            return None
        return self._validated_retry_cooldown_row(rows[0], task_cid=task)

    def _stable_ready_material(
        self,
    ) -> tuple[
        Mapping[str, Any],
        tuple[tuple[TaskRecord, Mapping[str, Any]], ...],
        int,
        Mapping[str, Mapping[str, Any]],
    ]:
        """Read tasks and typed cooldowns under one generation identity."""

        for _attempt in range(4):
            before = self._client.load_generation()
            snapshot_row, records, revision = self._snapshot_material()
            cooldowns: dict[str, Mapping[str, Any]] = {}
            offset = 0
            while offset <= len(records):
                rows = self._client.execute(
                    "executor_retry_cooldown_page",
                    {
                        "limit": min(_TRANSPORT_PAGE_LIMIT, len(records) + 1),
                        "offset": offset,
                    },
                )
                if not rows:
                    break
                for raw in rows:
                    row = dict(raw)
                    task_cid = str(row.get("task_cid") or "")
                    if not task_cid or task_cid in cooldowns:
                        raise TaskSourceIntegrityError(
                            "retry cooldown page is malformed or duplicated"
                        )
                    validated = self._validated_retry_cooldown_row(
                        row,
                        task_cid=task_cid,
                    )
                    cooldowns[task_cid] = validated
                offset += len(rows)
                if len(rows) < min(_TRANSPORT_PAGE_LIMIT, len(records) + 1):
                    break
                if len(cooldowns) > len(records):
                    raise TaskSourceBoundsError(
                        "retry cooldown population exceeds task population"
                    )
            after = self._client.load_generation()
            if (
                before.content_id == after.content_id
                and revision == after.revision
            ):
                task_cids = {record.task_cid for record, _identity in records}
                if not set(cooldowns).issubset(task_cids):
                    raise TaskSourceIntegrityError(
                        "retry cooldown projection contains a foreign task"
                    )
                for record, _identity in records:
                    if record.status == "retrying":
                        cooldown = cooldowns.get(record.task_cid)
                        if cooldown is None:
                            raise TaskSourceIntegrityError(
                                "retrying task has no typed cooldown receipt"
                            )
                        self._validate_retrying_cooldown_binding(
                            record,
                            cooldown,
                        )
                return snapshot_row, records, revision, MappingProxyType(cooldowns)
        raise TaskSourceConflictError(
            "typed task/cooldown projection changed during bounded snapshot"
        )

    @staticmethod
    def _validate_retrying_cooldown_binding(
        task: TaskRecord,
        cooldown: Mapping[str, Any],
    ) -> None:
        """Bind ready retry admission to its exact task control receipt."""

        body = task.body if isinstance(task.body, Mapping) else {}
        receipt = body.get("completion_receipt")
        extension = cooldown.get("extension")
        if not isinstance(receipt, Mapping) or not isinstance(
            extension, Mapping
        ):
            raise TaskSourceIntegrityError(
                "retrying task has no complete typed cooldown binding"
            )
        receipt_values = dict(receipt)
        extension_values = dict(extension)
        operation = receipt_values.get("operation")
        if type(operation) is not str or operation not in (
            TYPED_RETRYING_RECEIPT_OPERATIONS
        ):
            raise TaskSourceIntegrityError(
                "retrying task receipt is not an admitted retry transition"
            )
        expected_task_revision = extension_values.get(
            "expected_task_revision"
        )
        if (
            isinstance(expected_task_revision, bool)
            or not isinstance(expected_task_revision, int)
            or expected_task_revision != task.revision - 1
        ):
            raise TaskSourceIntegrityError(
                "retry cooldown differs from the task revision lineage"
            )
        exact_bindings = {
            "attempt_id": extension_values.get("attempt_id"),
            "claim_id": extension_values.get("claim_id"),
            "lease_id": extension_values.get("lease_id"),
            "owner_session_id": extension_values.get("owner_session_id"),
            "attempt_number": extension_values.get("attempt_number"),
            "fencing_token": extension_values.get("fencing_token"),
            "fence_epoch": extension_values.get("fence_epoch"),
            "queue_reason": extension_values.get("reason"),
            "backoff_ms": extension_values.get("delay_ms"),
            "retry_not_before_ms": extension_values.get(
                "retry_not_before_ms"
            ),
            "control_expected_revision": task.revision - 1,
        }
        if any(
            type(receipt_values.get(name)) is not type(expected)
            or receipt_values.get(name) != expected
            for name, expected in exact_bindings.items()
        ):
            raise TaskSourceIntegrityError(
                "retrying task receipt differs from its typed cooldown"
            )

    def _snapshot_from_material(
        self,
        row: Mapping[str, Any],
        records: tuple[tuple[TaskRecord, Mapping[str, Any]], ...],
        revision: int,
    ) -> TaskSourceSnapshot:
        tasks = [record for record, _identity in records]
        plan_cids = {task.plan_cid for task in tasks if task.plan_cid}
        plan_root = next(iter(plan_cids)) if len(plan_cids) == 1 else ""
        repository_trees = {
            str(identity.get("repository_tree_id") or "").strip()
            for _task, identity in records
            if str(identity.get("repository_tree_id") or "").strip()
        }
        if len(repository_trees) > 1:
            raise TaskSourceIntegrityError("typed task population spans multiple repository trees")
        repository_tree_id = next(iter(repository_trees)) if repository_trees else ""
        goals = _list_json(row.get("goals_json"), noun="goal snapshot")
        plans = _list_json(row.get("plans_json"), noun="plan snapshot")
        task_heads = _list_json(row.get("tasks_json"), noun="task head snapshot")
        projection = {
            "schema": TYPED_DATABASE_TASK_SOURCE_SCHEMA,
            "store_revision": revision,
            "goals": goals,
            "plans": plans,
            "task_heads": task_heads,
            "tasks": [task.to_dict() for task in tasks],
            "repository_tree_id": repository_tree_id,
            "plan_root_cid": plan_root,
        }
        projection_cid = content_identity(projection)
        terminal = bool(tasks) and all(task.status in _TERMINAL_STATUSES for task in tasks)
        source_identity = content_identity(
            {
                "plan_root_cid": plan_root,
                "repository_tree_id": repository_tree_id,
                "projection_cid": projection_cid,
            }
        )
        return TaskSourceSnapshot(
            source_schema=DATABASE_TASK_SOURCE_SCHEMA,
            schema_version=1,
            plan_root_cid=plan_root,
            repository_tree_id=repository_tree_id,
            projection_cid=projection_cid,
            formal_plan_id=plan_root,
            source_identity=source_identity,
            revision=max(1, revision),
            event_cursor=int(row.get("event_watermark") or 0),
            goal_count=int(row.get("goal_count") or 0),
            task_count=int(row.get("task_count") or 0),
            dependency_count=int(row.get("dependency_count") or 0),
            terminal=terminal,
            objective_count=int(row.get("objective_count") or 0),
            plan_count=int(row.get("plan_count") or 0),
        )

    def snapshot(self) -> TaskSourceSnapshot:
        row, records, revision = self._snapshot_material()
        return self._snapshot_from_material(row, records, revision)

    @property
    def execution_route_policy(self) -> TaskExecutionRoutePolicy | None:
        """Return the immutable launch policy, never an ambient projection."""

        return self._execution_route_policy

    def seal_execution_route_policy(
        self,
        execution_modes: Mapping[str, str],
    ) -> TaskExecutionRoutePolicy:
        """Seal all current tasks and modes from one generation-stable read."""

        row, records, revision = self._snapshot_material()
        snapshot = self._snapshot_from_material(row, records, revision)
        return TaskExecutionRoutePolicy.seal(
            snapshot=snapshot,
            tasks=tuple(record for record, _identity in records),
            execution_modes=execution_modes,
        )

    def _validate_execution_route_policy_population(self) -> None:
        policy = self._execution_route_policy
        if policy is None:
            return
        row, records, revision = self._snapshot_material()
        snapshot = self._snapshot_from_material(row, records, revision)
        tasks = tuple(record for record, _identity in records)
        entries = policy.entries_by_cid
        if (
            snapshot.plan_root_cid != policy.plan_root_cid
            or snapshot.repository_tree_id != policy.repository_tree_id
            or len(tasks) != len(entries)
            or {task.task_cid for task in tasks} != set(entries)
        ):
            raise TaskSourceIntegrityError(
                "launch execution route policy differs from current typed population"
            )
        for task in tasks:
            entry = entries[task.task_cid]
            if (
                task.task_alias != entry.task_alias
                or task.revision < entry.task_revision
                or task_execution_contract_cid(task) != entry.task_contract_cid
            ):
                raise TaskSourceIntegrityError(
                    "typed task changed after launch execution-route admission"
                )

    def _require_execution_route_plan_root(self) -> TaskExecutionRoutePolicy:
        policy = self._execution_route_policy
        if policy is None:
            raise TaskSourceIntegrityError(
                "typed executor has no launch execution route policy"
            )
        snapshot = self.snapshot()
        if (
            snapshot.plan_root_cid != policy.plan_root_cid
            or snapshot.repository_tree_id != policy.repository_tree_id
        ):
            raise TaskSourceIntegrityError(
                "typed task plan root changed after execution-route admission"
            )
        return policy

    def execution_route_binding_for_task(
        self,
        task: TaskRecord,
    ) -> Mapping[str, Any]:
        """Bind a task to its exact launch route or its carried retry lineage."""

        policy = self._require_execution_route_plan_root()
        entry = policy.entries_by_cid.get(str(task.task_cid or ""))
        if entry is None:
            raise TaskSourceIntegrityError(
                "task is absent from the launch route policy"
            )
        if task.revision == entry.task_revision:
            return MappingProxyType(policy.binding_for_task(task).to_dict())
        body = task.body if isinstance(task.body, Mapping) else {}
        receipt = body.get("completion_receipt")
        route = (
            receipt.get("execution_route_binding")
            if isinstance(receipt, Mapping)
            else None
        )
        if not isinstance(route, Mapping):
            raise TaskSourceIntegrityError(
                "advanced task revision has no carried execution-route binding"
            )
        return self.validate_execution_route_binding(
            route,
            task=task,
            allow_claim_revision=True,
        )

    def validate_execution_route_binding(
        self,
        value: Mapping[str, Any],
        *,
        task: TaskRecord,
        allow_claim_revision: bool = False,
    ) -> Mapping[str, Any]:
        """Validate an attempt binding and its exact shared claim lineage."""

        policy = self._require_execution_route_plan_root()
        binding: TaskExecutionRouteBinding = policy.validate_binding(value)
        if (
            task.task_cid != binding.task_cid
            or task.task_alias != binding.task_alias
            or task_execution_contract_cid(task) != binding.task_contract_cid
        ):
            raise TaskSourceIntegrityError(
                "attempt execution route differs from its authoritative task"
            )
        if task.revision == binding.task_revision:
            return MappingProxyType(binding.to_dict())
        if not allow_claim_revision or task.revision <= binding.task_revision:
            raise TaskSourceIntegrityError(
                "authoritative task revision differs from its execution route"
            )
        body = task.body if isinstance(task.body, Mapping) else {}
        receipt = body.get("completion_receipt")
        route = (
            receipt.get("execution_route_binding")
            if isinstance(receipt, Mapping)
            else None
        )
        if (
            not isinstance(receipt, Mapping)
            or not isinstance(route, Mapping)
            or dict(route) != binding.to_dict()
            or receipt.get("execution_route_policy_id") != binding.policy_id
            or receipt.get("execution_route_origin_revision")
            != binding.task_revision
        ):
            raise TaskSourceIntegrityError(
                "advanced task revision has no exact carried execution-route lineage"
            )
        return MappingProxyType(binding.to_dict())

    def get_task(self, task_cid_or_alias: Any) -> TaskRecord | None:
        self._require_open()
        if isinstance(task_cid_or_alias, TaskRecord):
            key = task_cid_or_alias.task_cid
        elif isinstance(task_cid_or_alias, Mapping):
            key = str(
                task_cid_or_alias.get("task_cid") or task_cid_or_alias.get("task_alias") or ""
            ).strip()
        else:
            key = str(task_cid_or_alias or "").strip()
        if not key:
            raise TaskSourceIntegrityError("task identity must not be empty")
        rows = self._client.execute(
            "executor_task_projection_by_identity",
            {"task_identity": key, "task_alias": key},
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise TaskSourceIntegrityError("task identity is ambiguous")
        return _record_from_row(rows[0])[0]

    get = get_task

    def task_revision_history_projection(
        self,
        task_cid_or_alias: str,
    ) -> Mapping[str, Any]:
        """Return canonical task history through the closed typed owner.

        The post-merge crash fence consumes this same projection contract from
        local DuckDB task sources.  Typed execution must obtain it through a
        fixed read-only operation so it never falls back to a second database
        authority or weakens the fence when a task reaches later revisions.
        """

        key = str(task_cid_or_alias or "").strip()
        if not key:
            raise TaskSourceIntegrityError("task identity must not be empty")
        for _attempt in range(4):
            before = self._client.load_generation()
            task = self.get_task(key)
            if task is None:
                raise KeyError(key)
            rows: list[Mapping[str, Any]] = []
            offset = 0
            while offset <= MAX_PROJECTION_RECORDS:
                page = self._client.execute(
                    "executor_task_revision_history_by_cid",
                    {
                        "task_cid": task.task_cid,
                        "limit": min(
                            _TASK_HISTORY_PAGE_LIMIT,
                            MAX_PROJECTION_RECORDS + 1 - offset,
                        ),
                        "offset": offset,
                    },
                )
                if not page:
                    break
                rows.extend(page)
                offset += len(page)
                if len(rows) > MAX_PROJECTION_RECORDS:
                    raise TaskSourceBoundsError(
                        "task revision history exceeds projection bound"
                    )
                if len(page) < _TASK_HISTORY_PAGE_LIMIT:
                    break
            after = self._client.load_generation()
            if before.content_id != after.content_id:
                continue

            revisions: list[dict[str, Any]] = []
            for raw in rows:
                row = dict(raw)
                if set(row) != {"revision", "status", "body_json"}:
                    raise TaskSourceIntegrityError(
                        "typed task revision differs from its closed projection"
                    )
                revision = row.get("revision")
                status = row.get("status")
                if (
                    isinstance(revision, bool)
                    or not isinstance(revision, int)
                    or revision < 1
                    or not isinstance(status, str)
                    or not status.strip()
                ):
                    raise TaskSourceIntegrityError(
                        "typed task revision history is malformed"
                    )
                revisions.append(
                    {
                        "revision": revision,
                        "status": status,
                        "body": _mapping_json(
                            row.get("body_json"),
                            noun="task revision body",
                        ),
                    }
                )
            material = {
                "schema": TASK_REVISION_HISTORY_PROJECTION_SCHEMA,
                "task_cid": task.task_cid,
                "revisions": revisions,
            }
            if len(canonical_json_bytes(material)) > MAX_PLAN_PROJECTION_BYTES:
                raise TaskSourceBoundsError(
                    "task revision history projection exceeds its byte bound"
                )
            return MappingProxyType(
                {
                    **material,
                    "projection_cid": content_identity(material),
                }
            )
        raise TaskSourceConflictError(
            "typed task revision history changed during bounded projection"
        )

    def list_tasks(
        self,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= TASK_SOURCE_MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(
                f"limit must be in [1, {TASK_SOURCE_MAX_QUERY_LIMIT}]"
            )
        snapshot_row, stable_records, revision = self._snapshot_material()
        snapshot = self._snapshot_from_material(snapshot_row, stable_records, revision)
        offset = _cursor_decode(cursor, revision=snapshot.revision) if cursor else 0
        records = [record for record, _identity in stable_records]
        if status is not None:
            selected = (
                {str(status).strip().lower()}
                if isinstance(status, str)
                else {str(item).strip().lower() for item in status}
            )
            records = [record for record in records if record.status in selected]
        page = records[offset : offset + limit]
        has_more = offset + len(page) < len(records)
        return TaskPage(
            tasks=tuple(page),
            revision=snapshot.revision,
            next_cursor=(_cursor_encode(snapshot.revision, offset + len(page)) if has_more else ""),
        )

    def ready_tasks(
        self,
        completed_ids: Iterable[str] = (),
        blocked_ids: Iterable[str] = (),
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= TASK_SOURCE_MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(
                f"limit must be in [1, {TASK_SOURCE_MAX_QUERY_LIMIT}]"
            )
        completed = {str(item).strip() for item in completed_ids if str(item).strip()}
        blocked = {str(item).strip() for item in blocked_ids if str(item).strip()}
        if completed & blocked:
            raise ValueError("completed_ids and blocked_ids must be disjoint")
        snapshot_row, stable_records, revision, cooldowns = (
            self._stable_ready_material()
        )
        snapshot = self._snapshot_from_material(snapshot_row, stable_records, revision)
        records = [record for record, _identity in stable_records]
        now_ms = getattr(
            self, "_clock_ms", lambda: int(time.time() * 1_000)
        )()
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise TaskSourceIntegrityError("typed task-source clock is invalid")
        by_identity = {
            identity: record
            for record in records
            for identity in (record.task_cid, record.task_alias)
        }
        ready: list[TaskRecord] = []
        for record in records:
            identities = {record.task_cid, record.task_alias}
            if (
                identities & (completed | blocked)
                or record.status not in _READY_STATUSES
                or int(
                    cooldowns.get(record.task_cid, {}).get(
                        "retry_not_before_ms",
                        0,
                    )
                )
                > now_ms
            ):
                continue
            if all(
                dependency in completed
                or (
                    dependency in by_identity
                    and by_identity[dependency].status in _COMPLETED_STATUSES
                )
                for dependency in record.dependencies
            ):
                ready.append(record)
                if len(ready) >= limit:
                    break
        return TaskPage(tasks=tuple(ready), revision=snapshot.revision)

    readiness = ready_tasks
    select_ready_tasks = ready_tasks

    def compare_and_set_status(
        self,
        task_cid_or_alias: Any,
        expected_revision: int,
        status: str,
        receipt: Mapping[str, Any] | None = None,
        *,
        evidence_digests: Sequence[str] | None = None,
    ) -> DatabaseCASResult:
        prior = self.get_task(task_cid_or_alias)
        if prior is None:
            raise KeyError(str(task_cid_or_alias))
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise TaskSourceConflictError("expected task revision is invalid")
        if prior.revision != expected_revision:
            raise TaskSourceConflictError("task revision CAS failed")
        requested_status = str(status or "").strip().lower()
        if requested_status not in TYPED_TASK_STATUS_VOCABULARY:
            raise TaskSourceIntegrityError(
                "task status is outside the closed typed vocabulary"
            )
        prior_receipt = prior.body.get("completion_receipt")
        if (
            prior.status == "blocked"
            and requested_status in _PROTECTED_REOPENED_TASK_STATUSES
            and isinstance(prior_receipt, Mapping)
            and prior_receipt.get("operation")
            == TYPED_DEFERRAL_BUDGET_BLOCK_OPERATION
        ):
            raise TaskSourceConflictError(
                "protected typed-deferral task cannot be reopened by generic CAS"
            )
        merged_body = dict(prior.body)
        if receipt is not None:
            merged_body["completion_receipt"] = dict(receipt)
        material = {
            "task_cid": prior.task_cid,
            "expected_revision": expected_revision,
            "status": requested_status,
            "receipt": dict(receipt or {}),
            "evidence_digests": list(evidence_digests or ()),
        }
        digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        result = self._client.cas_task_status(
            task_cid=prior.task_cid,
            expected_task_revision=expected_revision,
            new_status=requested_status,
            idempotency_key=f"executor-cas:{digest}",
            command_id=f"executor-cas:{digest}",
            body=merged_body,
        )
        if not result.accepted:
            raise TaskSourceConflictError(
                str(result.result.get("error") or "task status CAS was not accepted")
            )
        updated = self.get_task(prior.task_cid)
        if updated is None:
            raise TaskSourceIntegrityError("task disappeared after status CAS")
        if updated.status != requested_status:
            raise TaskSourceIntegrityError("task status CAS returned inconsistent state")
        return DatabaseCASResult(
            task=updated,
            previous_status=prior.status,
            revision=updated.revision,
            event_cursor=self.snapshot().event_cursor,
            changed=bool(result.changed),
            receipt_cid=str(result.result_digest or ""),
        )

    cas_status = compare_and_set_status

    def claim_process_attestation(self) -> Mapping[str, Any]:
        """Return the active owner-derived process tuple for claim receipts."""

        attestation = self._client.claim_process_attestation()
        return MappingProxyType(dict(attestation))

    def recover_dead_claim_reservation(
        self,
        task_cid_or_alias: str,
        *,
        expected_task_revision: int,
        now_ms: int | None = None,
    ) -> IntentReceipt:
        """Atomically reopen one unpromoted claim owned by a dead process."""

        prior = self.get(task_cid_or_alias)
        if prior is None:
            raise KeyError(str(task_cid_or_alias))
        if (
            prior.status != "in_progress"
            or isinstance(expected_task_revision, bool)
            or not isinstance(expected_task_revision, int)
            or prior.revision != expected_task_revision
        ):
            raise TaskSourceConflictError(
                "dead claim recovery task revision is stale"
            )
        prior_body = prior.body if isinstance(prior.body, Mapping) else {}
        reservation = prior_body.get("completion_receipt")
        if not isinstance(reservation, Mapping):
            raise TaskSourceIntegrityError(
                "dead claim recovery task has no reservation receipt"
            )
        selected_now = self._clock_ms() if now_ms is None else now_ms
        if (
            isinstance(selected_now, bool)
            or not isinstance(selected_now, int)
            or selected_now < 0
        ):
            raise TaskSourceIntegrityError("typed task-source clock is invalid")
        result = self._client.recover_dead_claim_reservation(
            task_cid=prior.task_cid,
            expected_task_revision=expected_task_revision,
            task_body=prior_body,
            reservation_receipt=reservation,
            now_ms=selected_now,
        )
        if not result.accepted:
            raise TaskSourceConflictError(
                str(
                    result.result.get("error")
                    or "dead claim recovery was not accepted"
                )
            )
        updated = self.get(prior.task_cid)
        row = self._retry_cooldown_row(prior.task_cid)
        if updated is None or updated.status != "retrying" or row is None:
            raise TaskSourceIntegrityError(
                "dead claim recovery post-state is incomplete"
            )
        self._validate_retrying_cooldown_binding(updated, row)
        receipt = updated.body.get("completion_receipt")
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("operation")
            != TYPED_DATABASE_CLAIM_RECOVERY_OPERATION
            or receipt.get("schema")
            != TYPED_DATABASE_CLAIM_RECOVERY_SCHEMA
            or receipt.get("attempt_number")
            != reservation.get("attempt_number")
        ):
            raise TaskSourceIntegrityError(
                "dead claim recovery receipt is inconsistent"
            )
        details = MappingProxyType(dict(result.result))
        return IntentReceipt(
            event_id=str(result.result_digest or content_identity(dict(details))),
            event_type="TASK_DEAD_CLAIM_RESERVATION_RECOVERED",
            global_sequence=0,
            recorded_at="typed-state-owner",
            subject_id=prior.task_cid,
            revision=int(updated.revision),
            changed=bool(result.changed),
            details=details,
        )

    def record_validation_result(
        self,
        *,
        task_cid: str,
        outcome: str,
        evidence_digest: str,
        argv: Sequence[str] | None = None,
        attempt_id: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> IntentReceipt:
        material = {
            "task_cid": str(task_cid),
            "outcome": str(outcome),
            "evidence_digest": str(evidence_digest),
            "argv": list(argv or ()),
            "attempt_id": str(attempt_id),
            "body": dict(body or {}),
        }
        digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        result = self._client.record_task_validation(
            task_cid=str(task_cid),
            outcome=str(outcome),
            evidence_digest=str(evidence_digest),
            argv=argv,
            attempt_id=attempt_id,
            body=body,
            idempotency_key=f"executor-validation:{digest}",
            command_id=f"executor-validation:{digest}",
        )
        if not result.accepted:
            raise TaskSourceConflictError(
                str(result.result.get("error") or "validation write was not accepted")
            )
        details = MappingProxyType(dict(result.result))
        return IntentReceipt(
            event_id=str(result.result_digest or content_identity(dict(details))),
            event_type="TASK_VALIDATION_RECORDED",
            global_sequence=0,
            recorded_at="typed-state-owner",
            subject_id=str(task_cid),
            revision=int(result.revision),
            changed=bool(result.changed),
            details=details,
        )

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
    ) -> IntentReceipt:
        """Persist and reproduce one owner-mediated typed retry cooldown."""

        if str(expected_task_status or "").strip().lower() == "blocked":
            raise TaskSourceConflictError(
                "typed blocked recovery requires coordination-coupled owner "
                "authority"
            )
        selected_now = self._clock_ms() if now_ms is None else now_ms
        if (
            isinstance(selected_now, bool)
            or not isinstance(selected_now, int)
            or selected_now < 0
        ):
            raise TaskSourceIntegrityError("typed task-source clock is invalid")
        result = self._client.record_task_retry_cooldown(
            task_cid=task_cid,
            expected_task_revision=expected_task_revision,
            expected_task_status=expected_task_status,
            attempt_id=attempt_id,
            claim_id=claim_id,
            lease_id=lease_id,
            owner_session_id=owner_session_id,
            attempt_number=attempt_number,
            fencing_token=fencing_token,
            fence_epoch=fence_epoch,
            delay_ms=delay_ms,
            reason=reason,
            selection_penalty=selection_penalty,
            now_ms=selected_now,
        )
        if not result.accepted:
            raise TaskSourceConflictError(
                str(
                    result.result.get("error")
                    or "typed retry cooldown write was not accepted"
                )
            )
        details = dict(result.result)
        row = self._retry_cooldown_row(str(task_cid))
        extension = dict(row.get("extension") or {}) if row is not None else {}
        checks = (
            ("row.present", row is not None, True),
            (
                "row.extension_schema",
                None if row is None else row.get("extension_schema"),
                TYPED_RETRY_COOLDOWN_SCHEMA,
            ),
            ("row.claim_id", None if row is None else row.get("claim_cid"), str(claim_id)),
            (
                "row.owner_session_id",
                None if row is None else row.get("owner_session_id"),
                str(owner_session_id),
            ),
            ("row.attempt", None if row is None else row.get("attempt"), int(attempt_number)),
            (
                "row.fencing_token",
                None if row is None else row.get("fencing_token"),
                int(fencing_token),
            ),
            (
                "row.fence_epoch",
                None if row is None else row.get("fence_epoch"),
                int(fence_epoch),
            ),
            (
                "row.retry_not_before_ms",
                None if row is None else row.get("retry_not_before_ms"),
                details.get("retry_not_before_ms"),
            ),
            (
                "row.revision",
                None if row is None else row.get("revision"),
                details.get("queue_revision"),
            ),
            ("extension.schema", extension.get("schema"), TYPED_RETRY_COOLDOWN_SCHEMA),
            ("extension.task_cid", extension.get("task_cid"), str(task_cid)),
            (
                "extension.expected_task_revision",
                extension.get("expected_task_revision"),
                int(expected_task_revision),
            ),
            ("extension.attempt_id", extension.get("attempt_id"), str(attempt_id)),
            ("extension.claim_id", extension.get("claim_id"), str(claim_id)),
            ("extension.lease_id", extension.get("lease_id"), str(lease_id)),
            (
                "extension.attempt_number",
                extension.get("attempt_number"),
                int(attempt_number),
            ),
            ("extension.reason", extension.get("reason"), str(reason)),
            ("details.schema", details.get("schema"), TYPED_RETRY_COOLDOWN_SCHEMA),
            (
                "details.operation",
                details.get("operation"),
                "task.retry.cooldown.record",
            ),
            ("details.task_cid", details.get("task_cid"), str(task_cid)),
            (
                "details.expected_task_revision",
                details.get("expected_task_revision"),
                int(expected_task_revision),
            ),
            ("details.attempt_id", details.get("attempt_id"), str(attempt_id)),
            ("details.claim_id", details.get("claim_id"), str(claim_id)),
            (
                "details.attempt_number",
                details.get("attempt_number"),
                int(attempt_number),
            ),
            ("details.reason", details.get("reason"), str(reason)),
        )
        mismatches = tuple(
            name
            for name, observed, expected in checks
            if observed != expected
            or (
                isinstance(expected, int)
                and not isinstance(expected, bool)
                and isinstance(observed, bool)
            )
        )
        if mismatches:
            raise TaskSourceIntegrityError(
                "typed retry cooldown post-state differs from its receipt: "
                + ", ".join(mismatches)
            )
        frozen = MappingProxyType(details)
        return IntentReceipt(
            event_id=str(result.result_digest or content_identity(details)),
            event_type="TASK_RETRY_COOLDOWN_RECORDED",
            global_sequence=0,
            recorded_at="typed-state-owner",
            subject_id=str(task_cid),
            revision=int(row["revision"]),
            changed=bool(result.changed),
            details=frozen,
        )

    @staticmethod
    def _queue_entry_from_cooldown_row(
        row: Mapping[str, Any],
    ) -> QueueEntry:
        extension = dict(row.get("extension") or {})
        for name in ("selection_penalty", "consecutive_failures"):
            value = extension.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TaskSourceIntegrityError(
                    f"retry cooldown {name} is invalid"
                )
        return QueueEntry(
            task_cid=str(row["task_cid"]),
            attempt=int(row["attempt"]),
            retry_not_before_ms=int(row["retry_not_before_ms"]),
            selection_penalty=int(extension.get("selection_penalty") or 0),
            consecutive_failures=int(
                extension.get("consecutive_failures") or 0
            ),
            state=str(row["state"]),
            reason=str(row["release_reason"] or extension.get("reason") or ""),
        )

    def validate_retrying_task_cooldown(
        self,
        task_cid_or_alias: str,
        *,
        expected_attempt_identity: Mapping[str, Any] | None = None,
        expected_reason: str | None = None,
        expected_delay_ms: int | None = None,
    ) -> QueueEntry:
        """Return an exact retrying task/queue binding under one generation."""

        for _attempt in range(4):
            before = self._client.load_generation()
            task = self.get(task_cid_or_alias)
            if task is None or task.status != "retrying":
                raise TaskSourceIntegrityError(
                    "typed cooldown validation requires a retrying task"
                )
            row = self._retry_cooldown_row(task.task_cid)
            after = self._client.load_generation()
            if before.content_id != after.content_id:
                continue
            if row is None:
                raise TaskSourceIntegrityError(
                    "retrying task has no typed cooldown receipt"
                )
            self._validate_retrying_cooldown_binding(task, row)
            extension = dict(row.get("extension") or {})
            if expected_attempt_identity is not None:
                required_identity = {
                    "attempt_id",
                    "claim_id",
                    "lease_id",
                    "owner_session_id",
                    "attempt_number",
                    "fencing_token",
                    "fence_epoch",
                }
                supplied_identity = dict(expected_attempt_identity)
                if set(supplied_identity) != required_identity:
                    raise TaskSourceIntegrityError(
                        "typed cooldown expected attempt identity is invalid"
                    )
                mismatches = [
                    name
                    for name in sorted(required_identity)
                    if type(extension.get(name))
                    is not type(supplied_identity.get(name))
                    or extension.get(name) != supplied_identity.get(name)
                ]
                if mismatches:
                    raise TaskSourceIntegrityError(
                        "retrying task cooldown differs from the expected "
                        "attempt: " + ", ".join(mismatches)
                    )
            if expected_reason is not None and (
                type(extension.get("reason")) is not str
                or extension.get("reason") != expected_reason
            ):
                raise TaskSourceIntegrityError(
                    "retrying task cooldown differs from the expected reason"
                )
            if expected_delay_ms is not None and (
                isinstance(expected_delay_ms, bool)
                or not isinstance(expected_delay_ms, int)
                or type(extension.get("delay_ms")) is not int
                or extension.get("delay_ms") != expected_delay_ms
            ):
                raise TaskSourceIntegrityError(
                    "retrying task cooldown differs from the expected delay"
                )
            return self._queue_entry_from_cooldown_row(row)
        raise TaskSourceConflictError(
            "typed retrying task/cooldown changed during bounded validation"
        )

    def validate_strict_resume_requeue_attempt_floor(
        self,
        task_cid_or_alias: str,
    ) -> int:
        """Return only the owner-validated scheduling floor of a strict requeue."""

        for _attempt in range(4):
            before = self._client.load_generation()
            task = self.get(task_cid_or_alias)
            if task is None or task.status != "ready":
                raise TaskSourceIntegrityError(
                    "typed strict-resume floor requires a ready task"
                )
            task_body = task.body if isinstance(task.body, Mapping) else {}
            receipt = task_body.get("completion_receipt")
            validated = _validated_database_strict_resume_rejection_receipt(
                receipt
            )
            after = self._client.load_generation()
            if before.content_id != after.content_id:
                continue
            shared = validated["shared_claim_binding"]
            if (
                validated["operation"]
                != TYPED_DATABASE_STRICT_RESUME_REQUEUE_OPERATION
                or validated["task_cid"] != task.task_cid
                or validated["rejected_task_alias"] != task.task_alias
                or validated["rejected_task_revision"] + 1
                != int(task.revision)
                or validated["attempt_budget_exhausted"] is not False
                or shared.get("claim_phase_schema")
                not in {
                    TYPED_DATABASE_CLAIM_RESERVATION_SCHEMA,
                    TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA,
                }
                or not validated["execution_route_binding"]
            ):
                raise TaskSourceIntegrityError(
                    "typed strict-resume ready floor differs from control truth"
                )
            return int(validated["attempt_number"])
        raise TaskSourceConflictError(
            "typed strict-resume ready floor changed during bounded validation"
        )

    def get_queue_entry(self, task_cid: str) -> QueueEntry | None:
        """Return canonical selection state through the closed owner query."""

        row = self._retry_cooldown_row(task_cid)
        if row is None:
            return None
        return self._queue_entry_from_cooldown_row(row)


__all__ = [
    "TYPED_DATABASE_TASK_SOURCE_INTERFACE",
    "TYPED_DATABASE_TASK_SOURCE_SCHEMA",
    "TypedDatabaseTaskSource",
]
