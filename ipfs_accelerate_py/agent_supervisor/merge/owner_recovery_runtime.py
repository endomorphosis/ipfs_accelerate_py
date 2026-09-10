"""Owner-held recovery cursors, consumer custody and versioned train receipts.

Provisioning is an explicit local migration action. Binding and client calls
only use the already admitted gateway handle; none opens a database or imports
filesystem projections. Receipts remain observations, never task acceptance.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

from .owner_merge_queue import (
    OwnerMergeQueueError,
    _BorrowedConnection,
    _OwnerMergeQueueService,
    _text,
)

SCHEMA = "ipfs_accelerate_py/legacy-owner-recovery-runtime@1"
STAGES = (
    "priority_task_cids",
    "completed_requests",
    "false_completed_requests",
    "false_pending_requests",
    "false_processing_requests",
    "pending_requests",
    "quarantined_requests",
    "processing_requests",
)
OPERATIONS = frozenset(
    {
        "describe_scope",
        "load_cursors",
        "cas_cursors",
        "acquire_consumer_lease",
        "renew_consumer_lease",
        "release_consumer_lease",
        "get_receipt",
        "publish_receipt",
    }
)
SERVICE_OPERATIONS = frozenset("legacy.merge_recovery." + x for x in OPERATIONS)
_SCOPE_FIELDS = frozenset(
    {"board_namespace", "config_cid", "plan_cid", "lane_id", "attempt_root"}
)
MAX_JSON_BYTES = 512 * 1024

# Exact versioned relations, intentionally independent of CASF event acks.
_TABLES = {
    "legacy_merge_recovery_migrations": "migration_id VARCHAR PRIMARY KEY, payload_cid VARCHAR NOT NULL",
    "legacy_merge_recovery_scopes": "scope_cid VARCHAR PRIMARY KEY, store_id VARCHAR NOT NULL, repository_id VARCHAR NOT NULL, target_branch VARCHAR NOT NULL, scope_json VARCHAR NOT NULL, migration_id VARCHAR NOT NULL",
    "legacy_merge_recovery_cursors": "scope_cid VARCHAR PRIMARY KEY, revision BIGINT NOT NULL, state_cid VARCHAR NOT NULL, cursors_json VARCHAR NOT NULL",
    "legacy_merge_recovery_cursor_history": "scope_cid VARCHAR NOT NULL, revision BIGINT NOT NULL, state_cid VARCHAR NOT NULL, cursors_json VARCHAR NOT NULL, PRIMARY KEY (scope_cid, revision)",
    "legacy_merge_recovery_operations": "scope_cid VARCHAR NOT NULL, consumer_id VARCHAR NOT NULL, operation_id VARCHAR NOT NULL, request_cid VARCHAR NOT NULL, result_json VARCHAR NOT NULL, PRIMARY KEY (scope_cid, consumer_id, operation_id)",
    "legacy_merge_recovery_leases": "repository_id VARCHAR NOT NULL, target_branch VARCHAR NOT NULL, lease_id VARCHAR NOT NULL, fence_epoch BIGINT NOT NULL, consumer_id VARCHAR NOT NULL, scope_cid VARCHAR NOT NULL, peer_json VARCHAR NOT NULL, owner_json VARCHAR NOT NULL, expires_at BIGINT NOT NULL, state VARCHAR NOT NULL, PRIMARY KEY (repository_id, target_branch)",
    "legacy_merge_recovery_receipt_heads": "repository_id VARCHAR NOT NULL, target_branch VARCHAR NOT NULL, receipt_key VARCHAR NOT NULL, revision BIGINT NOT NULL, receipt_cid VARCHAR NOT NULL, PRIMARY KEY (repository_id, target_branch, receipt_key)",
    "legacy_merge_recovery_receipt_versions": "repository_id VARCHAR NOT NULL, target_branch VARCHAR NOT NULL, receipt_key VARCHAR NOT NULL, revision BIGINT NOT NULL, receipt_cid VARCHAR NOT NULL, receipt_json VARCHAR NOT NULL, scope_cid VARCHAR NOT NULL, consumer_id VARCHAR NOT NULL, migration_id VARCHAR NOT NULL, PRIMARY KEY (repository_id, target_branch, receipt_key, revision)",
}


class OwnerRecoveryRuntimeError(OwnerMergeQueueError):
    """Recovery operation lacks exact admitted state/custody."""


def _json(value: Any) -> str:
    try:
        value = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise OwnerRecoveryRuntimeError("recovery value is not bounded JSON") from exc
    if len(value.encode()) > MAX_JSON_BYTES:
        raise OwnerRecoveryRuntimeError("recovery JSON exceeds bound")
    return value


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode()).hexdigest()


def _json_object(raw: Any) -> dict:
    if type(raw) is not str or len(raw.encode()) > MAX_JSON_BYTES:
        raise OwnerRecoveryRuntimeError("recovery wire JSON exceeds bound")
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise OwnerRecoveryRuntimeError("recovery wire JSON is malformed") from exc
    if type(value) is not dict or _json(value) != raw:
        raise OwnerRecoveryRuntimeError("recovery wire JSON is not a canonical object")
    return value


def _integer(value, name, *, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise OwnerRecoveryRuntimeError("invalid " + name)
    return value


def _closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise OwnerRecoveryRuntimeError("recovery fields differ from closed contract")


def _cursors(value):
    _closed(value, STAGES)
    return {name: _text(value[name], name, empty=True) for name in STAGES}


def _scope(binding):
    _closed(binding, _SCOPE_FIELDS)
    result = {key: _text(binding[key], key) for key in sorted(_SCOPE_FIELDS)}
    root = Path(result["attempt_root"])
    if (
        not root.is_absolute()
        or str(root) != result["attempt_root"]
        or ".." in root.parts
    ):
        raise OwnerRecoveryRuntimeError(
            "attempt root must be an admitted absolute namespace"
        )
    return result


def recovery_scope_cid(*, store_id, repository_id, target_branch, scope_binding):
    return _cid(
        {
            "schema": SCHEMA,
            "store_id": _text(store_id, "store_id"),
            "repository_id": _text(repository_id, "repository_id"),
            "target_branch": _text(target_branch, "target_branch"),
            "scope": _scope(scope_binding),
        }
    )


def _tuple(row):
    return tuple(row[index] for index in range(len(row)))


def _schema_columns(definition):
    columns = []
    for part in definition.split(", PRIMARY KEY (", 1)[0].split(", "):
        if part.startswith("PRIMARY KEY") or part.endswith(")"):
            continue
        words = part.split()
        columns.append(
            (
                words[0],
                words[1],
                "NO" if "NOT NULL" in part or "PRIMARY KEY" in part else "YES",
            )
        )
    # Columns forming composite primary keys are NOT NULL in DuckDB.
    if "PRIMARY KEY (" in definition:
        keys = definition.split("PRIMARY KEY (", 1)[1].rstrip(")").split(", ")
        columns = [
            (name, kind, "NO" if name in keys else nullable)
            for name, kind, nullable in columns
        ]
    return tuple(columns)


class _OwnerRecoveryRuntimeService(_OwnerMergeQueueService):
    def __init__(
        self,
        gateway,
        *,
        expected_identity,
        repository_id,
        target_branch,
        validate_schema=True,
    ):
        gateway._require_live_server_binding()
        queue = gateway._legacy_merge_queue_service
        if (
            queue is None
            or dict(expected_identity) != dict(gateway.identity)
            or queue.repository_id != repository_id
            or queue.target_branch != target_branch
        ):
            raise OwnerRecoveryRuntimeError(
                "recovery requires the exact already bound owner queue"
            )
        queue._validate_owner()
        self._gateway = gateway
        self._connection = queue._connection
        self._raw_connection = queue._raw_connection
        self._request_admission = None
        self.identity = dict(gateway.identity)
        self.repository_id = _text(repository_id, "repository_id")
        self.target_branch = _text(target_branch, "target_branch")
        self._retired = False
        if validate_schema:
            with _BorrowedConnection(self) as connection:
                connection.execute("BEGIN TRANSACTION")
                self._validate_schema()
                connection.commit()

    def _validate_schema(self):
        super()._validate_schema()
        run = self._connection._execute_once
        for table, definition in _TABLES.items():
            rows = run(
                "SELECT table_catalog, table_schema, table_type FROM information_schema.tables WHERE table_name=?",
                [table],
            ).fetchall()
            database = run("SELECT current_database()", None).fetchone()[0]
            if [_tuple(row) for row in rows] != [(database, "main", "BASE TABLE")]:
                raise OwnerRecoveryRuntimeError(
                    "explicit complete recovery schema migration required"
                )
            columns = run(
                "SELECT column_name,data_type,is_nullable FROM information_schema.columns WHERE table_catalog=current_database() AND table_schema='main' AND table_name=? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            if tuple(_tuple(row) for row in columns) != _schema_columns(definition):
                raise OwnerRecoveryRuntimeError("recovery schema columns differ")
            expected_keys = (
                definition.split("PRIMARY KEY (", 1)[1].rstrip(")").split(", ")
                if "PRIMARY KEY (" in definition
                else [definition.split()[0]]
            )
            keys = run(
                "SELECT constraint_column_names FROM duckdb_constraints() WHERE database_name=current_database() AND schema_name='main' AND table_name=? AND constraint_type='PRIMARY KEY'",
                [table],
            ).fetchall()
            if [list(row[0]) for row in keys] != [expected_keys]:
                raise OwnerRecoveryRuntimeError(
                    "recovery schema identity constraint differs"
                )

    def _scope_row(self, scope_cid):
        rows = self._connection._execute_once(
            "SELECT store_id,repository_id,target_branch,scope_json FROM legacy_merge_recovery_scopes WHERE scope_cid=?",
            [scope_cid],
        ).fetchall()
        if len(rows) != 1:
            raise OwnerRecoveryRuntimeError(
                "recovery scope has not been owner provisioned"
            )
        store, repo, target, raw = _tuple(rows[0])
        scope = _scope(json.loads(raw))
        if (
            store != self._gateway.store_id
            or repo != self.repository_id
            or target != self.target_branch
            or recovery_scope_cid(
                store_id=store,
                repository_id=repo,
                target_branch=target,
                scope_binding=scope,
            )
            != scope_cid
        ):
            raise OwnerRecoveryRuntimeError("recovery scope binding differs")
        return scope

    def _cursor_head(self, scope):
        rows = self._connection._execute_once(
            "SELECT revision,state_cid,cursors_json FROM legacy_merge_recovery_cursors WHERE scope_cid=?",
            [scope],
        ).fetchall()
        if len(rows) != 1:
            raise OwnerRecoveryRuntimeError("provisioned cursor is missing")
        revision, cid, raw = _tuple(rows[0])
        _integer(revision, "cursor revision")
        latest = self._connection._execute_once(
            "SELECT MIN(revision),MAX(revision),COUNT(*) FROM legacy_merge_recovery_cursor_history WHERE scope_cid=?",
            [scope],
        ).fetchone()
        if _tuple(latest) != (0, revision, revision + 1):
            raise OwnerRecoveryRuntimeError(
                "cursor head is not the latest preserved revision"
            )
        value = _cursors(json.loads(raw))
        if _cid(value) != cid:
            raise OwnerRecoveryRuntimeError("cursor content identity differs")
        preserved = self._connection._execute_once(
            "SELECT state_cid,cursors_json FROM legacy_merge_recovery_cursor_history WHERE scope_cid=? AND revision=?",
            [scope, revision],
        ).fetchall()
        if len(preserved) != 1 or _tuple(preserved[0]) != (cid, raw):
            raise OwnerRecoveryRuntimeError(
                "exact cursor revision history is missing or differs"
            )
        return {"revision": revision, "state_cid": cid, "cursors": value}

    def _receipt_head(self, key, revision=None):
        run = self._connection._execute_once
        if revision is None:
            rows = run(
                "SELECT revision,receipt_cid FROM legacy_merge_recovery_receipt_heads WHERE repository_id=? AND target_branch=? AND receipt_key=?",
                [self.repository_id, self.target_branch, key],
            ).fetchall()
            if not rows:
                preserved = run(
                    "SELECT 1 FROM legacy_merge_recovery_receipt_versions WHERE repository_id=? AND target_branch=? AND receipt_key=? LIMIT 1",
                    [self.repository_id, self.target_branch, key],
                ).fetchone()
                if preserved is not None:
                    raise OwnerRecoveryRuntimeError(
                        "preserved train receipt has no admitted head"
                    )
                return None
            revision, head_cid = _tuple(rows[0])
            _integer(revision, "receipt revision", minimum=1)
            latest = run(
                "SELECT MIN(revision),MAX(revision),COUNT(*) FROM legacy_merge_recovery_receipt_versions WHERE repository_id=? AND target_branch=? AND receipt_key=?",
                [self.repository_id, self.target_branch, key],
            ).fetchone()
            if _tuple(latest) != (1, revision, revision):
                raise OwnerRecoveryRuntimeError(
                    "receipt head is not the latest preserved revision"
                )
        else:
            head_cid = None
        rows = run(
            "SELECT receipt_cid,receipt_json FROM legacy_merge_recovery_receipt_versions WHERE repository_id=? AND target_branch=? AND receipt_key=? AND revision=?",
            [self.repository_id, self.target_branch, key, revision],
        ).fetchall()
        if len(rows) != 1:
            raise OwnerRecoveryRuntimeError("exact preserved train receipt is missing")
        cid, raw = _tuple(rows[0])
        receipt = json.loads(raw)
        if (
            type(receipt) is not dict
            or _cid(receipt) != cid
            or (head_cid is not None and cid != head_cid)
        ):
            raise OwnerRecoveryRuntimeError("train receipt immutable identity differs")
        return {"revision": revision, "receipt_cid": cid, "receipt": receipt}

    def _lease_row(self):
        row = self._connection._execute_once(
            "SELECT lease_id,fence_epoch,consumer_id,scope_cid,peer_json,owner_json,expires_at,state FROM legacy_merge_recovery_leases WHERE repository_id=? AND target_branch=?",
            [self.repository_id, self.target_branch],
        ).fetchone()
        result = (
            None
            if row is None
            else dict(
                zip(
                    (
                        "lease_id",
                        "fence_epoch",
                        "consumer_id",
                        "scope_cid",
                        "peer_json",
                        "owner_json",
                        "expires_at",
                        "state",
                    ),
                    _tuple(row),
                )
            )
        )
        if result is None:
            return None
        if result["state"] not in {"active", "released"}:
            raise OwnerRecoveryRuntimeError("consumer custody state is unknown")
        _integer(result["fence_epoch"], "consumer fence", minimum=1)
        _integer(result["expires_at"], "consumer expiry")
        for key in ("lease_id", "consumer_id", "scope_cid"):
            _text(result[key], key)
        self._scope_row(result["scope_cid"])
        owner = _json_object(result["owner_json"])
        if (
            set(owner) != set(self.identity)
            or any(
                type(owner[key]) is not type(value)
                for key, value in self.identity.items()
            )
            or owner["store_id"] != self.identity["store_id"]
            or owner["database_uuid"] != self.identity["database_uuid"]
        ):
            raise OwnerRecoveryRuntimeError(
                "consumer custody owner binding is malformed"
            )
        for key, value in owner.items():
            if type(value) is int:
                _integer(
                    value, key, minimum=1 if key in {"generation", "fence_epoch"} else 0
                )
            elif type(value) is str:
                _text(value, key)
        try:
            peer = json.loads(result["peer_json"])
        except (ValueError, RecursionError) as exc:
            raise OwnerRecoveryRuntimeError(
                "consumer custody peer binding is malformed"
            ) from exc
        if (
            type(peer) is not list
            or len(peer) != 3
            or _json(peer) != result["peer_json"]
        ):
            raise OwnerRecoveryRuntimeError(
                "consumer custody peer binding is malformed"
            )
        for index, value in enumerate(peer):
            _integer(value, "consumer peer", minimum=0 if index == 1 else 1)
        return result

    def _require_lease(self, args, scope, consumer, *, allow_released=False):
        row = self._lease_row()
        peer = self._request_admission[2]
        if (
            row is None
            or row["state"]
            not in ({"active", "released"} if allow_released else {"active"})
            or row["lease_id"] != args["lease_id"]
            or row["fence_epoch"] != args["fence_epoch"]
            or row["consumer_id"] != consumer
            or row["scope_cid"] != scope
            or row["peer_json"] != _json(list(peer))
            or row["owner_json"] != _json(self.identity)
        ):
            raise OwnerRecoveryRuntimeError(
                "exact current owner/peer consumer lease required"
            )
        # Expiry is a diagnostic, never proof that a retained callback is gone.
        return row

    def _replay(self, scope, consumer, operation_id, request):
        row = self._connection._execute_once(
            "SELECT request_cid,result_json FROM legacy_merge_recovery_operations WHERE scope_cid=? AND consumer_id=? AND operation_id=?",
            [scope, consumer, operation_id],
        ).fetchone()
        if row is None:
            return None
        if row[0] != _cid({k: v for k, v in request.items() if k != "owner_identity"}):
            raise OwnerRecoveryRuntimeError(
                "operation id already binds different recovery request"
            )
        return json.loads(row[1])

    def _record(self, connection, scope, consumer, operation_id, request, result):
        connection.execute(
            "INSERT INTO legacy_merge_recovery_operations VALUES (?,?,?,?,?)",
            [
                scope,
                consumer,
                operation_id,
                _cid({k: v for k, v in request.items() if k != "owner_identity"}),
                _json(result),
            ],
        )

    def _execute_request(self, payload, *, grant):
        self._validate_owner()
        self._validate_schema()
        _closed(
            payload,
            {
                "schema",
                "operation",
                "owner_identity",
                "repository_id",
                "target_branch",
                "consumer_id",
                "recovery_scope_cid",
                "arguments",
            },
        )
        operation = payload["operation"]
        scopes = dict(grant.entity_scopes)
        if (
            payload["schema"] != SCHEMA
            or type(operation) is not str
            or operation not in OPERATIONS
            or set(scopes)
            != {"repository_id", "target_branch", "consumer_id", "recovery_scope_cid"}
            or any(payload[k] != v for k, v in scopes.items())
            or payload["repository_id"] != self.repository_id
            or payload["target_branch"] != self.target_branch
            or type(payload["owner_identity"]) is not dict
            or payload["owner_identity"] != self.identity
            or any(
                type(payload["owner_identity"].get(k)) is not type(v)
                for k, v in self.identity.items()
            )
            or "legacy.merge_recovery." + operation not in grant.allowed_operations
        ):
            raise OwnerRecoveryRuntimeError(
                "recovery request differs from exact issued scope"
            )
        scope = _text(scopes["recovery_scope_cid"], "recovery_scope_cid")
        consumer = _text(scopes["consumer_id"], "consumer_id")
        binding = self._scope_row(scope)
        args = payload["arguments"]
        fields = {
            "describe_scope": (),
            "load_cursors": (),
            "cas_cursors": (
                "expected_revision",
                "expected_state_cid",
                "cursors",
                "operation_id",
            ),
            "acquire_consumer_lease": ("operation_id", "ttl_seconds"),
            "renew_consumer_lease": (
                "lease_id",
                "fence_epoch",
                "ttl_seconds",
                "operation_id",
            ),
            "release_consumer_lease": ("lease_id", "fence_epoch", "operation_id"),
            "get_receipt": ("receipt_key", "revision"),
            "publish_receipt": (
                "receipt_key",
                "receipt_json",
                "expected_revision",
                "expected_receipt_cid",
                "lease_id",
                "fence_epoch",
                "operation_id",
            ),
        }
        _closed(args, fields[operation])
        args = dict(args)
        if "receipt_json" in args:
            args["receipt"] = _json_object(args.pop("receipt_json"))
        for key in ("operation_id", "lease_id", "receipt_key", "expected_state_cid"):
            if key in args:
                _text(args[key], key)
        for key in ("expected_revision", "fence_epoch"):
            if key in args:
                _integer(args[key], key)
        if "ttl_seconds" in args and (
            type(args["ttl_seconds"]) not in (int, float)
            or not math.isfinite(args["ttl_seconds"])
            or not 1 <= args["ttl_seconds"] <= 3600
        ):
            raise OwnerRecoveryRuntimeError("lease TTL outside bounded contract")
        if "cursors" in args:
            _cursors(args["cursors"])
        if "receipt" in args and type(args["receipt"]) is not dict:
            raise OwnerRecoveryRuntimeError("receipt must be an object")
        if "receipt" in args:
            _json(args["receipt"])
        if "expected_receipt_cid" in args:
            _text(args["expected_receipt_cid"], "expected_receipt_cid", empty=True)
        if "revision" in args and args["revision"] is not None:
            _integer(args["revision"], "revision", minimum=1)
        with _BorrowedConnection(self) as connection:
            connection.execute("BEGIN TRANSACTION")
            if operation in {
                "renew_consumer_lease",
                "release_consumer_lease",
                "publish_receipt",
            }:
                self._require_lease(
                    args,
                    scope,
                    consumer,
                    allow_released=operation == "release_consumer_lease",
                )
            result = None
            changed = False
            if "operation_id" in args:
                result = self._replay(scope, consumer, args["operation_id"], payload)
            if (
                operation == "acquire_consumer_lease"
                and result is not None
                and result.get("acquired") is True
            ):
                # An old successful reply is not a fresh custody grant after
                # release, peer replacement, or an owner-generation change.
                self._require_lease(result, scope, consumer)
            if result is None:
                if operation == "describe_scope":
                    result = {"recovery_scope_cid": scope, "scope": binding}
                elif operation == "load_cursors":
                    result = self._cursor_head(scope)
                elif operation == "cas_cursors":
                    head = self._cursor_head(scope)
                    new = _cursors(args["cursors"])
                    if (args["expected_revision"], args["expected_state_cid"]) != (
                        head["revision"],
                        head["state_cid"],
                    ):
                        result = {**head, "conflict": True, "changed": False}
                    elif new == head["cursors"]:
                        result = {**head, "conflict": False, "changed": False}
                    else:
                        revision = head["revision"] + 1
                        cid = _cid(new)
                        raw = _json(new)
                        connection.execute(
                            "UPDATE legacy_merge_recovery_cursors SET revision=?,state_cid=?,cursors_json=? WHERE scope_cid=? AND revision=? AND state_cid=?",
                            [
                                revision,
                                cid,
                                raw,
                                scope,
                                head["revision"],
                                head["state_cid"],
                            ],
                        )
                        connection.execute(
                            "INSERT INTO legacy_merge_recovery_cursor_history VALUES (?,?,?,?)",
                            [scope, revision, cid, raw],
                        )
                        changed = True
                        result = {
                            "revision": revision,
                            "state_cid": cid,
                            "cursors": new,
                            "conflict": False,
                            "changed": True,
                        }
                elif operation == "get_receipt":
                    result = {
                        "head": self._receipt_head(
                            args["receipt_key"], args["revision"]
                        )
                    }
                elif operation == "acquire_consumer_lease":
                    row = self._lease_row()
                    if row is not None and row["state"] == "active":
                        result = {
                            "acquired": False,
                            "reason": "consumer_custody_retained",
                            "lease_id": None,
                            "fence_epoch": None,
                        }
                    else:
                        fence = 1 if row is None else row["fence_epoch"] + 1
                        lease = "lease:" + uuid.uuid4().hex
                        expires = int(time.time() * 1000 + args["ttl_seconds"] * 1000)
                        values = [
                            lease,
                            fence,
                            consumer,
                            scope,
                            _json(list(self._request_admission[2])),
                            _json(self.identity),
                            expires,
                            "active",
                            self.repository_id,
                            self.target_branch,
                        ]
                        if row is None:
                            connection.execute(
                                "INSERT INTO legacy_merge_recovery_leases (lease_id,fence_epoch,consumer_id,scope_cid,peer_json,owner_json,expires_at,state,repository_id,target_branch) VALUES (?,?,?,?,?,?,?,?,?,?)",
                                values,
                            )
                        else:
                            connection.execute(
                                "UPDATE legacy_merge_recovery_leases SET lease_id=?,fence_epoch=?,consumer_id=?,scope_cid=?,peer_json=?,owner_json=?,expires_at=?,state=? WHERE repository_id=? AND target_branch=?",
                                values,
                            )
                        result = {
                            "acquired": True,
                            "lease_id": lease,
                            "fence_epoch": fence,
                            "expires_at": expires,
                        }
                        changed = True
                elif operation == "renew_consumer_lease":
                    expires = int(time.time() * 1000 + args["ttl_seconds"] * 1000)
                    connection.execute(
                        "UPDATE legacy_merge_recovery_leases SET expires_at=? WHERE repository_id=? AND target_branch=?",
                        [expires, self.repository_id, self.target_branch],
                    )
                    changed = True
                    result = {
                        "renewed": True,
                        "lease_id": args["lease_id"],
                        "fence_epoch": args["fence_epoch"],
                        "expires_at": expires,
                    }
                elif operation == "release_consumer_lease":
                    connection.execute(
                        "UPDATE legacy_merge_recovery_leases SET state='released' WHERE repository_id=? AND target_branch=?",
                        [self.repository_id, self.target_branch],
                    )
                    changed = True
                    result = {
                        "released": True,
                        "lease_id": args["lease_id"],
                        "fence_epoch": args["fence_epoch"],
                    }
                elif operation == "publish_receipt":
                    head = self._receipt_head(args["receipt_key"])
                    old_revision = 0 if head is None else head["revision"]
                    old_cid = "" if head is None else head["receipt_cid"]
                    if (args["expected_revision"], args["expected_receipt_cid"]) != (
                        old_revision,
                        old_cid,
                    ):
                        result = {"head": head, "conflict": True, "changed": False}
                    elif head is not None and head["receipt"] == args["receipt"]:
                        result = {"head": head, "conflict": False, "changed": False}
                    else:
                        revision = old_revision + 1
                        cid = _cid(args["receipt"])
                        connection.execute(
                            "INSERT INTO legacy_merge_recovery_receipt_versions VALUES (?,?,?,?,?,?,?,?,?)",
                            [
                                self.repository_id,
                                self.target_branch,
                                args["receipt_key"],
                                revision,
                                cid,
                                _json(args["receipt"]),
                                scope,
                                consumer,
                                "",
                            ],
                        )
                        if head is None:
                            connection.execute(
                                "INSERT INTO legacy_merge_recovery_receipt_heads VALUES (?,?,?,?,?)",
                                [
                                    self.repository_id,
                                    self.target_branch,
                                    args["receipt_key"],
                                    revision,
                                    cid,
                                ],
                            )
                        else:
                            connection.execute(
                                "UPDATE legacy_merge_recovery_receipt_heads SET revision=?,receipt_cid=? WHERE repository_id=? AND target_branch=? AND receipt_key=?",
                                [
                                    revision,
                                    cid,
                                    self.repository_id,
                                    self.target_branch,
                                    args["receipt_key"],
                                ],
                            )
                        result = {
                            "head": {
                                "revision": revision,
                                "receipt_cid": cid,
                                "receipt": args["receipt"],
                            },
                            "conflict": False,
                            "changed": True,
                        }
                        changed = True
                if changed:
                    self._record(
                        connection,
                        scope,
                        consumer,
                        args["operation_id"],
                        payload,
                        result,
                    )
            connection.commit()
        return {
            "schema": SCHEMA,
            "operation": operation,
            "owner_identity": self.identity,
            "repository_id": self.repository_id,
            "target_branch": self.target_branch,
            "consumer_id": consumer,
            "recovery_scope_cid": scope,
            "completion_authority": False,
            "result_json": _json(result),
        }


def provision_legacy_merge_recovery_schema(
    gateway,
    *,
    expected_identity,
    repository_id,
    target_branch,
    migration_id,
    scope_bindings,
    receipt_imports=(),
    cursor_imports=(),
):
    """Explicit local owner migration; never reachable from the worker protocol.

    The native operator must qualify preserved-state/source admission before this
    call. This function neither supplies that admission nor issues client grants.
    """
    service = _OwnerRecoveryRuntimeService(
        gateway,
        expected_identity=expected_identity,
        repository_id=repository_id,
        target_branch=target_branch,
        validate_schema=False,
    )
    migration_id = _text(migration_id, "migration_id")
    if (
        not isinstance(scope_bindings, (list, tuple))
        or not 1 <= len(scope_bindings) <= 256
        or not isinstance(receipt_imports, (list, tuple))
        or len(receipt_imports) > 10000
        or not isinstance(cursor_imports, (list, tuple))
        or len(cursor_imports) > 256
    ):
        raise OwnerRecoveryRuntimeError("migration population exceeds bound")
    scopes = [_scope(x) for x in scope_bindings]
    scope_ids = [
        recovery_scope_cid(
            store_id=gateway.store_id,
            repository_id=repository_id,
            target_branch=target_branch,
            scope_binding=scope,
        )
        for scope in scopes
    ]
    if len(set(scope_ids)) != len(scope_ids):
        raise OwnerRecoveryRuntimeError("migration repeats a recovery scope")
    imported_cursors = {}
    for item in cursor_imports:
        _closed(item, {"scope_cid", "cursors", "state_cid"})
        scope_cid = _text(item["scope_cid"], "scope_cid")
        cursors = _cursors(item["cursors"])
        if (
            scope_cid not in scope_ids
            or scope_cid in imported_cursors
            or _cid(cursors) != item["state_cid"]
        ):
            raise OwnerRecoveryRuntimeError(
                "preserved cursor scope or content identity differs"
            )
        imported_cursors[scope_cid] = cursors
    payload = {
        "repository_id": repository_id,
        "target_branch": target_branch,
        "scope_bindings": scopes,
        "receipt_imports": list(receipt_imports),
        "cursor_imports": list(cursor_imports),
    }
    payload_cid = _cid(payload)
    with _BorrowedConnection(service) as connection:
        connection.execute("BEGIN TRANSACTION")
        names = [
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_name LIKE 'legacy_merge_recovery_%'"
            ).fetchall()
        ]
        if not names:
            for name, definition in _TABLES.items():
                connection.execute("CREATE TABLE " + name + " (" + definition + ")")
        service._validate_schema()
        old = connection.execute(
            "SELECT payload_cid FROM legacy_merge_recovery_migrations WHERE migration_id=?",
            [migration_id],
        ).fetchone()
        if old is not None:
            if old[0] != payload_cid:
                raise OwnerRecoveryRuntimeError(
                    "migration id already binds different preserved state"
                )
            # Replaying admission must verify the preserved coordinates, not
            # merely a surviving migration marker. Later heads are legitimate.
            for cid in scope_ids:
                service._scope_row(cid)
                origin = connection.execute(
                    "SELECT migration_id FROM legacy_merge_recovery_scopes WHERE scope_cid=?",
                    [cid],
                ).fetchone()[0]
                initial = connection.execute(
                    "SELECT state_cid,cursors_json FROM legacy_merge_recovery_cursor_history WHERE scope_cid=? AND revision=0",
                    [cid],
                ).fetchall()
                if len(initial) != 1:
                    raise OwnerRecoveryRuntimeError(
                        "preserved initial cursor history is missing"
                    )
                initial_cid, initial_raw = _tuple(initial[0])
                initial_value = _cursors(json.loads(initial_raw))
                if _cid(initial_value) != initial_cid:
                    raise OwnerRecoveryRuntimeError(
                        "preserved initial cursor identity differs"
                    )
                if origin == migration_id:
                    expected_initial = imported_cursors.get(
                        cid, {stage: "" for stage in STAGES}
                    )
                    if initial_value != expected_initial or initial_raw != _json(
                        expected_initial
                    ):
                        raise OwnerRecoveryRuntimeError(
                            "migration initial cursor differs from preserved import"
                        )
                elif cid in imported_cursors:
                    raise OwnerRecoveryRuntimeError(
                        "imported cursor scope migration identity differs"
                    )
                service._cursor_head(cid)
            for item in receipt_imports:
                _closed(item, {"receipt_key", "revision", "receipt_cid", "receipt"})
                preserved = service._receipt_head(item["receipt_key"], item["revision"])
                if preserved != {
                    "revision": item["revision"],
                    "receipt_cid": item["receipt_cid"],
                    "receipt": item["receipt"],
                }:
                    raise OwnerRecoveryRuntimeError(
                        "preserved imported receipt differs"
                    )
                origin = connection.execute(
                    "SELECT migration_id FROM legacy_merge_recovery_receipt_versions WHERE repository_id=? AND target_branch=? AND receipt_key=? AND revision=?",
                    [
                        repository_id,
                        target_branch,
                        item["receipt_key"],
                        item["revision"],
                    ],
                ).fetchone()[0]
                if origin != migration_id:
                    raise OwnerRecoveryRuntimeError(
                        "imported receipt migration identity differs"
                    )
                service._receipt_head(item["receipt_key"])
            connection.commit()
            return {
                "migration_id": migration_id,
                "replayed": True,
                "scope_cids": [
                    recovery_scope_cid(
                        store_id=gateway.store_id,
                        repository_id=repository_id,
                        target_branch=target_branch,
                        scope_binding=s,
                    )
                    for s in scopes
                ],
            }
        scope_ids = []
        for scope in scopes:
            cid = recovery_scope_cid(
                store_id=gateway.store_id,
                repository_id=repository_id,
                target_branch=target_branch,
                scope_binding=scope,
            )
            scope_ids.append(cid)
            existing = connection.execute(
                "SELECT scope_json FROM legacy_merge_recovery_scopes WHERE scope_cid=?",
                [cid],
            ).fetchone()
            if existing is not None:
                service._scope_row(cid)
                if cid in imported_cursors:
                    raise OwnerRecoveryRuntimeError(
                        "cursor import cannot replace an already provisioned scope"
                    )
                continue
            connection.execute(
                "INSERT INTO legacy_merge_recovery_scopes VALUES (?,?,?,?,?,?)",
                [
                    cid,
                    gateway.store_id,
                    repository_id,
                    target_branch,
                    _json(scope),
                    migration_id,
                ],
            )
            empty = imported_cursors.get(cid, {stage: "" for stage in STAGES})
            raw = _json(empty)
            state = _cid(empty)
            connection.execute(
                "INSERT INTO legacy_merge_recovery_cursors VALUES (?,?,?,?)",
                [cid, 0, state, raw],
            )
            connection.execute(
                "INSERT INTO legacy_merge_recovery_cursor_history VALUES (?,?,?,?)",
                [cid, 0, state, raw],
            )
        for item in receipt_imports:
            _closed(item, {"receipt_key", "revision", "receipt_cid", "receipt"})
            key = _text(item["receipt_key"], "receipt_key")
            revision = _integer(item["revision"], "revision", minimum=1)
            if (
                type(item["receipt"]) is not dict
                or _cid(item["receipt"]) != item["receipt_cid"]
            ):
                raise OwnerRecoveryRuntimeError(
                    "imported receipt content identity differs"
                )
            head = service._receipt_head(key)
            expected = 1 if head is None else head["revision"] + 1
            if revision != expected:
                raise OwnerRecoveryRuntimeError(
                    "receipt import requires complete contiguous preserved history"
                )
            connection.execute(
                "INSERT INTO legacy_merge_recovery_receipt_versions VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    repository_id,
                    target_branch,
                    key,
                    revision,
                    item["receipt_cid"],
                    _json(item["receipt"]),
                    "",
                    "",
                    migration_id,
                ],
            )
            if head is None:
                connection.execute(
                    "INSERT INTO legacy_merge_recovery_receipt_heads VALUES (?,?,?,?,?)",
                    [repository_id, target_branch, key, revision, item["receipt_cid"]],
                )
            else:
                connection.execute(
                    "UPDATE legacy_merge_recovery_receipt_heads SET revision=?,receipt_cid=? WHERE repository_id=? AND target_branch=? AND receipt_key=?",
                    [revision, item["receipt_cid"], repository_id, target_branch, key],
                )
        connection.execute(
            "INSERT INTO legacy_merge_recovery_migrations VALUES (?,?)",
            [migration_id, payload_cid],
        )
        connection.commit()
    return {"migration_id": migration_id, "replayed": False, "scope_cids": scope_ids}


class OwnerRecoveryRuntimeClient:
    """Closed grant-bound adapter; no file path or independent DB fallback."""

    def __init__(
        self,
        connection,
        *,
        repository_id,
        target_branch,
        consumer_id,
        recovery_scope_cid,
    ):
        self.connection = connection
        self.repository_id = _text(repository_id, "repository_id")
        self.target_branch = _text(target_branch, "target_branch")
        self.consumer_id = _text(consumer_id, "consumer_id")
        self.recovery_scope_cid = _text(recovery_scope_cid, "recovery_scope_cid")

    def call(self, operation, **arguments):
        if operation == "publish_receipt" and "receipt" in arguments:
            arguments["receipt_json"] = _json(arguments.pop("receipt"))
        response = self.connection.legacy_merge_recovery(
            {
                "schema": SCHEMA,
                "operation": operation,
                "owner_identity": dict(self.connection.identity),
                "repository_id": self.repository_id,
                "target_branch": self.target_branch,
                "consumer_id": self.consumer_id,
                "recovery_scope_cid": self.recovery_scope_cid,
                "arguments": arguments,
            }
        )
        return response["result"]

    def describe_scope(self):
        return self.call("describe_scope")["scope"]

    def load_cursors(self):
        return self.call("load_cursors")

    def cas_cursors(
        self, *, expected_revision, expected_state_cid, cursors, operation_id
    ):
        return self.call(
            "cas_cursors",
            expected_revision=expected_revision,
            expected_state_cid=expected_state_cid,
            cursors=cursors,
            operation_id=operation_id,
        )

    def acquire_consumer_lease(self, *, operation_id, ttl_seconds=300):
        return self.call(
            "acquire_consumer_lease", operation_id=operation_id, ttl_seconds=ttl_seconds
        )

    def renew_consumer_lease(
        self, *, lease_id, fence_epoch, ttl_seconds=300, operation_id
    ):
        return self.call(
            "renew_consumer_lease",
            lease_id=lease_id,
            fence_epoch=fence_epoch,
            ttl_seconds=ttl_seconds,
            operation_id=operation_id,
        )

    def release_consumer_lease(self, *, lease_id, fence_epoch, operation_id):
        return self.call(
            "release_consumer_lease",
            lease_id=lease_id,
            fence_epoch=fence_epoch,
            operation_id=operation_id,
        )

    def get_receipt(self, receipt_key, *, revision=None):
        return self.call("get_receipt", receipt_key=receipt_key, revision=revision)[
            "head"
        ]

    def publish_receipt(
        self,
        receipt_key,
        receipt,
        *,
        expected_revision,
        expected_receipt_cid,
        lease_id,
        fence_epoch,
        operation_id,
    ):
        return self.call(
            "publish_receipt",
            receipt_key=receipt_key,
            receipt=receipt,
            expected_revision=expected_revision,
            expected_receipt_cid=expected_receipt_cid,
            lease_id=lease_id,
            fence_epoch=fence_epoch,
            operation_id=operation_id,
        )
