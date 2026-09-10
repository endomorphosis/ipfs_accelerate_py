"""Closed operational port for an already migrated legacy ``merge_requests``.

Binding borrows a live typed gateway's handle. It neither opens a file nor
installs schema, imports receipts, grants admission, or authorizes task/goal
completion. Legacy queue transitions remain the implementation of record.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from typing import Any, Mapping

from .merge_queue import (
    MERGE_TARGET_BINDING_SCHEMA,
    MAX_MERGE_QUEUE_DEFERRAL_SECONDS,
    MergeQueue,
    MergeQueueFenceError,
    _MERGE_QUEUE_SETTLEMENT_COLUMNS,
)

OPERATIONS = frozenset(
    {
        "get",
        "enqueue",
        "claim",
        "dequeue",
        "owns_claim",
        "complete",
        "requeue",
        "quarantine",
        "defer",
        "pending_requests",
        "processing_requests",
        "completed_requests",
        "quarantined_requests",
        "has_pending_for_task",
    }
)
SERVICE_OPERATIONS = frozenset("legacy.merge_queue." + name for name in OPERATIONS)
SCHEMA = "ipfs_accelerate_py/legacy-owner-merge-queue@1"
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024


class OwnerMergeQueueError(RuntimeError):
    """Queue binding, transaction, or closed request was not admitted."""


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if (
        type(value) is not str
        or len(value) > 4096
        or "\0" in value
        or value != value.strip()
        or (not empty and not value)
    ):
        raise OwnerMergeQueueError(f"invalid {name}")
    return value


def _metadata(value: Any) -> dict[str, Any]:
    if type(value) is not str or len(value.encode("utf-8")) > 65536:
        raise OwnerMergeQueueError("metadata_json must be bounded JSON text")

    def reject_constant(_value):
        raise OwnerMergeQueueError("non-finite metadata is not admitted")

    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise OwnerMergeQueueError("duplicate metadata key")
            result[key] = item
        return result

    value = json.loads(value, parse_constant=reject_constant, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise OwnerMergeQueueError("metadata must be an object")
    json.dumps(value, allow_nan=False)
    return value


class _BorrowedConnection:
    """One legacy context owns only transactions it successfully began.

    Never enters/exits/closes the gateway handle. Failed SQL/commit cannot be
    converted to a successful dedupe lookup by legacy exception recovery. An
    unfinished transaction rolls back; a failed rollback freezes the shared owner
    binding and this service.
    """

    def __init__(self, service):
        self.service = service
        self.connection = service._connection
        self.owned = False
        self.failure = None

    def __enter__(self):
        if self.service._retired or self.connection.in_transaction:
            raise OwnerMergeQueueError(
                "queue cannot inherit a transaction or retired binding"
            )
        return self

    def execute(self, sql, parameters=None):
        if self.failure is not None:
            raise self.failure
        try:
            normalized = " ".join(sql.strip().upper().split())
            if not normalized.startswith(("SELECT ", "BEGIN ")) and not self.owned:
                raise OwnerMergeQueueError("queue write requires its own transaction")
            result = self.connection._execute_once(sql, parameters)
            if sql.strip().upper() in {"BEGIN IMMEDIATE", "BEGIN TRANSACTION"}:
                self.owned = True
                self.service._validate_owner()
            return result
        except Exception as exc:
            if (
                self.connection._poisoned
                or self.connection._connection is not self.service._raw_connection
            ):
                # BEGIN can itself fail its native rollback before this loan
                # records ownership. Freeze that uncertainty as well.
                self.service._retire_owner_binding()
            self.failure = OwnerMergeQueueError("owner queue SQL failed")
            raise self.failure from exc

    def commit(self):
        if self.failure is not None:
            raise self.failure
        if not self.owned:
            raise OwnerMergeQueueError(
                "queue cannot commit a transaction it does not own"
            )
        # The gateway transaction lock already excludes store/session writers.
        # Revalidate those coordinates, then serialize the final lease check
        # and COMMIT against revoke_grant. Never call the locking grant checker
        # while holding its non-reentrant lock.
        try:
            self.service._validate_owner()
            with self.service._gateway._grants_lock:
                self.service._validate_commit_grant_locked()
                try:
                    self.connection._execute_once("COMMIT", None)
                    self.owned = False
                except Exception as exc:
                    # Commit outcome may be uncertain; do not reopen or retry locally.
                    # A healthy, still-open transaction can roll back once.
                    # A natively poisoned handle must never enter rollback's
                    # implicit recovery path. Either outcome freezes the shared
                    # binding before another gateway service can acquire it.
                    try:
                        self.rollback()
                    finally:
                        self.service._retire_owner_binding()
                    self.failure = OwnerMergeQueueError(
                        "owner queue commit failed; binding retired"
                    )
                    raise self.failure from exc
        except Exception as exc:
            if self.failure is not None:
                raise
            # Even an admission denial must poison this borrowed operation:
            # legacy enqueue must not acknowledge its post-rollback dedupe lookup.
            self.failure = OwnerMergeQueueError("owner queue commit admission denied")
            raise self.failure from exc

    def rollback(self):
        if not self.owned:
            return
        # A failed rollback is uncertain, not permission to retry it during
        # legacy exception cleanup and again during context exit.
        self.owned = False
        try:
            if (
                self.connection._poisoned
                or self.connection._connection is not self.service._raw_connection
            ):
                raise OwnerMergeQueueError("borrowed native handle became uncertain")
            self.connection.rollback()
        except Exception as exc:
            self.service._retire_owner_binding()
            self.failure = OwnerMergeQueueError(
                "owner queue rollback failed; binding retired"
            )
            raise self.failure from exc

    def __exit__(self, exc_type, exc, traceback):
        unfinished = self.owned
        self.rollback()
        if self.failure is not None:
            raise self.failure
        if unfinished and exc_type is None:
            raise OwnerMergeQueueError("queue operation omitted its commit")

    def close(self):
        """The gateway exclusively owns handle lifetime."""


class _OwnerQueue(MergeQueue):
    def __init__(
        self,
        service,
        *,
        max_age_seconds,
        max_queue_size,
        max_processing,
        max_attempts,
        max_worktree_bytes,
        worktree_usage,
    ):
        # Do not call MergeQueue.__init__: it opens files and imports projections.
        self.service = service
        self.max_age_seconds = max_age_seconds
        self.max_queue_size = max_queue_size
        self.max_processing = max_processing
        self.max_attempts = max_attempts
        self.max_worktree_bytes = max_worktree_bytes
        self._worktree_usage = worktree_usage
        self.priority_aging_seconds = 300
        self._clock = time.time
        self.target_repository_id = ""
        self.target_branch = ""
        self.require_target_binding = False
        self.bind_target(service.repository_id, service.target_branch, required=True)
        self.completed_dir = None

    def _connect(self):
        return _BorrowedConnection(self.service)

    def _request_from_row(self, row):
        self._require_row_target(
            row, operation="owner_queue", request_id=str(row["request_id"])
        )
        request = super()._request_from_row(row)
        if request.dedupe_key != str(row["dedupe_key"] or ""):
            raise OwnerMergeQueueError(
                "legacy request dedupe identity differs from its preserved coordinates"
            )
        return request

    def _find_by_dedupe_key(self, connection, dedupe_key):
        row = super()._find_by_dedupe_key(connection, dedupe_key)
        if row is not None:
            # Validate before legacy enqueue's idempotent COMMIT, not after it.
            self._request_from_row(row)
        return row

    def has_pending_for_task(self, task_id, *, commit_sha=None):
        # Pending cooldowns and expired processing claims still own work.
        # Page readers intentionally omit cooldowns, so they cannot answer this
        # dispatch-suppression question. Validate preserved identities as well.
        target_sql, target_parameters = self._target_binding_sql()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM merge_requests "
                "WHERE status IN ('pending','processing')" + target_sql,
                target_parameters,
            ).fetchall()
        identity = task_id.casefold()
        found = False
        for row in rows:
            request = self._request_from_row(row)
            if identity in {
                request.task_id.casefold(),
                request.canonical_task_id.casefold(),
                request.canonical_task_key.casefold(),
            } and (
                commit_sha is None
                or request.commit_sha.casefold() == commit_sha.casefold()
            ):
                found = True
        return found

    def _stage_path(self, request):
        return None

    def _write_stage_receipt(self, request):
        return None

    def _prune_receipts(self, directory, *, keep):
        pass

    def _purge_stale(self):
        # An expired lease denies mutation; it does not establish callback
        # closure or authorize recovering another consumer's provider work.
        return 0


class _OwnerMergeQueueService:
    def __init__(
        self,
        gateway,
        *,
        expected_identity,
        repository_id,
        target_branch,
        max_age_seconds,
        max_queue_size,
        max_processing,
        max_attempts,
        max_worktree_bytes,
        worktree_usage,
    ):
        from ..task_sources.duckdb_state import DuckDBConnection

        gateway._require_live_server_binding()
        if dict(expected_identity) != dict(gateway.identity):
            raise OwnerMergeQueueError("queue binding differs from admitted owner")
        if not isinstance(gateway._connection, DuckDBConnection):
            raise OwnerMergeQueueError(
                "queue requires the owner's existing tracked handle"
            )
        self._gateway = gateway
        self._connection = gateway._connection
        self._raw_connection = self._connection._connection
        self._request_admission = None
        self.identity = dict(gateway.identity)
        self.repository_id = _text(repository_id, "repository_id")
        self.target_branch = _text(target_branch, "target_branch")
        self._retired = False
        for value in (max_age_seconds, max_queue_size, max_processing, max_attempts):
            if type(value) is not int or not 1 <= value <= 1000000:
                raise OwnerMergeQueueError(
                    "queue policy requires bounded positive integers"
                )
        if max_worktree_bytes is not None and (
            type(max_worktree_bytes) is not int or max_worktree_bytes < 0
        ):
            raise OwnerMergeQueueError("invalid worktree capacity")
        if worktree_usage is not None and not callable(worktree_usage):
            raise OwnerMergeQueueError("worktree usage must be an owner-side callable")
        with _BorrowedConnection(self) as connection:
            connection.execute("BEGIN TRANSACTION")
            self._validate_schema()
            connection.commit()
        self._queue = _OwnerQueue(
            self,
            max_age_seconds=max_age_seconds,
            max_queue_size=max_queue_size,
            max_processing=max_processing,
            max_attempts=max_attempts,
            max_worktree_bytes=max_worktree_bytes,
            worktree_usage=worktree_usage,
        )

    def _retire_owner_binding(self):
        self._retired = True
        self._gateway._retire_borrowed_owner_connection(self._connection)

    def _validate_owner(self):
        self._gateway._require_live_server_binding()
        if (
            self._retired
            or self._gateway._connection is not self._connection
            or self._connection._connection is not self._raw_connection
            or dict(self._gateway.identity) != self.identity
            or self._gateway.store_id != self.identity.get("store_id")
        ):
            raise OwnerMergeQueueError("queue owner binding changed")
        row = self._connection._execute_once(
            "SELECT generation, schema_revision, fence_epoch, database_uuid, birth_id "
            "FROM store_generations ORDER BY generation DESC LIMIT 1",
            None,
        ).fetchone()
        expected = tuple(
            self.identity.get(key)
            for key in (
                "generation",
                "schema_revision",
                "fence_epoch",
                "database_uuid",
                "process_birth_id",
            )
        )
        if (
            row is None
            or tuple(row[index] for index in range(len(row))) != expected
            or self._connection._connection is not self._raw_connection
        ):
            raise OwnerMergeQueueError("queue store generation or fence is stale")
        if self._request_admission is not None:
            grant, session_id, peer_identity = self._request_admission
            grant = self._gateway._require_active_grant(
                grant, peer_identity=peer_identity
            )
            session = self._connection._execute_once(
                "SELECT owner_id, process_birth_id, server_id, generation, fence_epoch, status "
                "FROM client_sessions WHERE session_id=?",
                [session_id],
            ).fetchall()
            expected_session = (
                grant.client_id,
                grant.process_birth_id,
                self.identity["server_id"],
                self.identity["generation"],
                self.identity["fence_epoch"],
                "attached",
            )
            if (
                len(session) != 1
                or tuple(session[0][index] for index in range(6)) != expected_session
            ):
                raise OwnerMergeQueueError("queue client session is detached or stale")

    def _validate_commit_grant_locked(self):
        """Caller holds gateway grants lock through the ensuing COMMIT."""
        if self._request_admission is None:
            return  # Binding's schema-only transaction has no client grant.
        grant, _session_id, peer = self._request_admission
        current = tuple(
            candidate
            for candidate in self._gateway._grants.values()
            if candidate.grant_id == grant.grant_id
        )
        if (
            grant.grant_id in self._gateway._revoked_grants
            or len(current) != 1
            or int(time.time() * 1000) >= current[0].expires_at
        ):
            raise OwnerMergeQueueError("queue commit grant expired or revoked")
        active = current[0]
        if (
            active.allowed_operations != grant.allowed_operations
            or active.entity_scopes != grant.entity_scopes
            or active.client_id != grant.client_id
            or active.process_birth_id != grant.process_birth_id
            or (active.peer_pid, active.peer_uid, active.peer_start_time_ticks) != peer
        ):
            raise OwnerMergeQueueError("queue commit grant authority changed")

    def _validate_schema(self):
        execute = self._connection._execute_once
        namespace = execute(
            "SELECT current_schema(), current_schemas(false)", None
        ).fetchone()
        shadows = execute(
            "SELECT table_name FROM information_schema.tables WHERE table_catalog='temp' "
            "AND table_name IN ('merge_requests', 'agent_supervisor_store_metadata', "
            "'store_generations', 'client_sessions')",
            None,
        ).fetchall()
        if (
            namespace[0] != "main"
            or list(namespace[1]) not in ([], ["main"])
            or shadows
        ):
            raise OwnerMergeQueueError("queue namespace changed or is shadowed")
        for table, expected in _MERGE_QUEUE_SETTLEMENT_COLUMNS.items():
            tables = execute(
                "SELECT table_type FROM information_schema.tables "
                "WHERE table_catalog=current_database() AND table_schema='main' AND table_name=?",
                [table],
            ).fetchall()
            columns = execute(
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                "WHERE table_catalog=current_database() AND table_schema='main' AND table_name=? "
                "ORDER BY ordinal_position",
                [table],
            ).fetchall()
            if [tuple(row[index] for index in range(len(row))) for row in tables] != [
                ("BASE TABLE",)
            ] or tuple(
                tuple(row[index] for index in range(len(row))) for row in columns
            ) != expected:
                raise OwnerMergeQueueError(
                    "exact existing legacy queue schema is required"
                )
        constraints = execute(
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE database_name=current_database() AND schema_name='main' "
            "AND table_name='merge_requests' AND constraint_type='PRIMARY KEY'",
            None,
        ).fetchall()
        indexes = execute(
            "SELECT expressions FROM duckdb_indexes() WHERE database_name=current_database() "
            "AND schema_name='main' AND table_name='merge_requests' AND is_unique",
            None,
        ).fetchall()
        if [list(row[0]) for row in constraints] != [["request_id"]] or not any(
            str(row[0]) == "[dedupe_key]" for row in indexes
        ):
            raise OwnerMergeQueueError("legacy queue identity constraints are required")

    def execute(self, payload: Mapping[str, Any], *, grant, session_id, peer_identity):
        # Gateway holds its transaction lock and has just revalidated peer/TTL.
        if self._request_admission is not None or self._connection.in_transaction:
            raise OwnerMergeQueueError("queue operation cannot inherit a transaction")
        self._request_admission = (grant, session_id, peer_identity)
        try:
            return self._execute_request(payload, grant=grant)
        except OwnerMergeQueueError:
            raise
        except Exception as exc:
            # Do not enter the gateway's generic native-fatal reopening path.
            # An uncertain queue transaction requires fresh owner qualification.
            raise OwnerMergeQueueError("owner queue operation rejected") from exc
        finally:
            self._request_admission = None

    def _execute_request(self, payload, *, grant):
        self._validate_owner()
        self._validate_schema()
        required = {
            "schema",
            "operation",
            "owner_identity",
            "repository_id",
            "target_branch",
            "consumer_id",
            "arguments",
        }
        if set(payload) != required or payload.get("schema") != SCHEMA:
            raise OwnerMergeQueueError("closed queue request schema is required")
        operation = payload["operation"]
        if type(operation) is not str or operation not in OPERATIONS:
            raise OwnerMergeQueueError("queue operation is not admitted")
        scopes = dict(grant.entity_scopes)
        if (
            set(scopes) != {"repository_id", "target_branch", "consumer_id"}
            or any(payload.get(key) != value for key, value in scopes.items())
            or scopes["repository_id"] != self.repository_id
            or scopes["target_branch"] != self.target_branch
            or type(payload["owner_identity"]) is not dict
            or payload["owner_identity"] != self.identity
            or any(
                type(payload["owner_identity"].get(key)) is not type(value)
                for key, value in self.identity.items()
            )
            or "legacy.merge_queue." + operation not in grant.allowed_operations
        ):
            raise OwnerMergeQueueError(
                "queue operation differs from the exact owner grant"
            )
        args = payload["arguments"]
        if type(args) is not dict:
            raise OwnerMergeQueueError("queue arguments must be an object")
        consumer = _text(scopes["consumer_id"], "consumer_id")
        result = None
        owns_claim = None
        requests_json = None
        has_pending = None
        if operation == "has_pending_for_task":
            if set(args) != {"task_id", "commit_sha"}:
                raise OwnerMergeQueueError(
                    "active task fields differ from closed contract"
                )
            task_id = _text(args["task_id"], "task_id")
            commit_sha = args["commit_sha"]
            if commit_sha is not None:
                commit_sha = _text(commit_sha, "commit_sha", empty=True)
            has_pending = self._queue.has_pending_for_task(
                task_id, commit_sha=commit_sha
            )
            self._validate_owner()
        elif operation in {
            "pending_requests",
            "processing_requests",
            "quarantined_requests",
            "completed_requests",
        }:
            if operation == "completed_requests":
                text_fields = {
                    "metadata_schema",
                    "completion_schema",
                    "completion_reason",
                    "canonical_task_id",
                    "database_task_cid",
                    "reopen_schema",
                    "reopen_reason",
                    "before_request_id",
                }
                bool_fields = {"require_completion_absent", "ordered_by_request_id"}
                if set(args) != {"limit"} | text_fields | bool_fields:
                    raise OwnerMergeQueueError(
                        "completion page fields differ from closed contract"
                    )
                arguments = {
                    key: _text(args[key], key, empty=True) for key in text_fields
                }
                for key in bool_fields:
                    if type(args[key]) is not bool:
                        raise OwnerMergeQueueError(
                            "completion page flags must be boolean"
                        )
                    arguments[key] = args[key]
            else:
                if set(args) != {"limit", "after_request_id"}:
                    raise OwnerMergeQueueError(
                        "snapshot fields differ from closed contract"
                    )
                cursor = args["after_request_id"]
                if cursor is not None:
                    cursor = _text(cursor, "after_request_id", empty=True)
                arguments = {"after_request_id": cursor}
            limit = args["limit"]
            if type(limit) is not int or not 1 <= limit <= 256:
                raise OwnerMergeQueueError(
                    "snapshot limit must be an integer from 1 to 256"
                )
            rows = getattr(self._queue, operation)(limit=limit, **arguments)
            # One page is an observation, never a lease recovery or settlement.
            # Retain the native fair/oldest order and explicit ID cursor order.
            encoded = []
            size = 2
            for row in rows:
                item = json.dumps(row.to_dict(), sort_keys=True, allow_nan=False)
                size += len(item.encode("utf-8")) + 2
                if size > MAX_SNAPSHOT_BYTES:
                    raise OwnerMergeQueueError(
                        "snapshot byte budget exceeded; use a smaller page"
                    )
                encoded.append(item)
            requests_json = "[" + ", ".join(encoded) + "]"
            # A grant revoked/expired or a detached session during the read
            # cannot supply a successful observation from the former binding.
            self._validate_owner()
        elif operation == "enqueue":
            allowed = {
                "branch_name",
                "task_id",
                "priority",
                "lane_id",
                "commit_sha",
                "canonical_task_id",
                "canonical_task_key",
                "metadata_json",
            }
            if set(args) != allowed:
                raise OwnerMergeQueueError("enqueue fields differ from closed contract")
            arguments = {
                key: _text(
                    value,
                    key,
                    empty=key in {"lane_id", "canonical_task_id", "canonical_task_key"},
                )
                for key, value in args.items()
                if key != "metadata_json"
            }
            result = self._queue.enqueue(
                **arguments, metadata=_metadata(args["metadata_json"])
            )
        elif operation == "dequeue":
            if args:
                raise OwnerMergeQueueError("dequeue accepts no arguments")
            result = self._queue.dequeue(consumer_id=consumer)
        elif operation in {"get", "claim"}:
            if set(args) != {"request_id"}:
                raise OwnerMergeQueueError("request_id is required")
            request_id = _text(args["request_id"], "request_id")
            result = self._queue.get(request_id)
            if result is not None:
                if not result.has_target_binding or (
                    result.target_repository_id,
                    result.target_branch,
                ) != (self.repository_id, self.target_branch):
                    raise MergeQueueFenceError(
                        "request target differs from queue binding"
                    )
                if operation == "claim":
                    result = self._queue.claim_pending_request(
                        request_id, consumer_id=consumer
                    )
        else:
            fields = {"request_id", "claim_token", "claim_generation"}
            if operation in {"complete", "requeue", "quarantine", "defer"}:
                fields.add("metadata_json")
            if operation in {"requeue", "quarantine", "defer"}:
                fields.add("reason")
            if operation == "defer":
                fields.add("delay_seconds_json")
            if set(args) != fields:
                raise OwnerMergeQueueError(
                    "claim operation fields differ from closed contract"
                )
            request_id = _text(args["request_id"], "request_id")
            token = _text(args["claim_token"], "claim_token")
            generation = args["claim_generation"]
            if type(generation) is not int or generation < 1:
                raise OwnerMergeQueueError("invalid claim generation")
            current = self._queue.get(request_id)
            if current is None:
                raise MergeQueueFenceError("claimed request is unavailable")
            request = replace(
                current,
                consumer_id=consumer,
                claim_token=token,
                claim_generation=generation,
            )
            if operation == "owns_claim":
                owns_claim = self._queue.owns_claim(request, consumer_id=consumer)
            else:
                metadata = _metadata(args["metadata_json"])
                target = {
                    "target_repository_id": self.repository_id,
                    "target_branch": self.target_branch,
                    "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
                }
                if any(
                    key in metadata and metadata[key] != value
                    for key, value in target.items()
                ):
                    raise OwnerMergeQueueError(
                        "transition metadata cannot change target binding"
                    )
                if operation == "complete":
                    self._queue.complete(request, metadata=metadata)
                elif operation == "requeue":
                    self._queue.requeue(
                        request,
                        reason=_text(args["reason"], "reason", empty=True),
                        metadata=metadata,
                    )
                elif operation == "defer":
                    delay_text = _text(args["delay_seconds_json"], "delay_seconds_json")
                    delay = json.loads(delay_text)
                    if (
                        type(delay) not in {int, float}
                        or not math.isfinite(delay)
                        or not 0 <= delay <= MAX_MERGE_QUEUE_DEFERRAL_SECONDS
                    ):
                        raise OwnerMergeQueueError(
                            "deferral delay is outside the native bound"
                        )
                    self._queue.defer(
                        request,
                        reason=_text(args["reason"], "reason", empty=True),
                        delay_seconds=delay,
                        metadata=metadata,
                    )
                else:
                    self._queue.quarantine(
                        request,
                        reason=_text(args["reason"], "reason", empty=True),
                        metadata=metadata,
                    )
                result = self._queue.get(request_id)
        # Legacy floating timestamps/metadata travel losslessly as JSON text:
        # the typed control-plane envelope intentionally forbids floats.
        response = {
            "schema": SCHEMA,
            "owner_identity": dict(self.identity),
            "operation": operation,
            "completion_authority": False,
            "owns_claim": owns_claim,
            "request_json": None
            if result is None
            else json.dumps(result.to_dict(), sort_keys=True, allow_nan=False),
        }
        if has_pending is not None:
            response["has_pending"] = has_pending
        if requests_json is not None:
            response["requests_json"] = requests_json
        return response


class OwnerMergeQueueClient:
    """Use an already admitted TypedStateOwnerConnection; never open a file."""

    def __init__(
        self, connection, *, repository_id: str, target_branch: str, consumer_id: str
    ):
        self.connection = connection
        self.repository_id = _text(repository_id, "repository_id")
        self.target_branch = _text(target_branch, "target_branch")
        self.consumer_id = _text(consumer_id, "consumer_id")

    def call(self, operation: str, **arguments):
        return self.connection.legacy_merge_queue(
            {
                "schema": SCHEMA,
                "operation": operation,
                "owner_identity": dict(self.connection.identity),
                "repository_id": self.repository_id,
                "target_branch": self.target_branch,
                "consumer_id": self.consumer_id,
                "arguments": arguments,
            }
        )
