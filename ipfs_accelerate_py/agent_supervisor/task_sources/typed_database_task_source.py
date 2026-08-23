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
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import canonical_json_bytes, content_identity
from .database_task_source import (
    DATABASE_TASK_SOURCE_SCHEMA,
    TaskPage,
    TaskRecord,
    TaskSourceBoundsError,
    TaskSourceConflictError,
    TaskSourceIntegrityError,
    TaskSourceSnapshot,
)
from .database_task_source import (
    CASResult as DatabaseCASResult,
)
from .intent_repository import IntentReceipt
from .quack_state_client import QuackStateClient

TYPED_DATABASE_TASK_SOURCE_INTERFACE: Final = "TypedDatabaseTaskSource@1"
TYPED_DATABASE_TASK_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/typed-database-task-source@1"
)
DEFAULT_QUERY_LIMIT: Final = 50
MAX_QUERY_LIMIT: Final = 500
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

    def __init__(self, client: QuackStateClient) -> None:
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise TaskSourceIntegrityError(
                "typed database task source requires an attached QuackStateClient"
            )
        self._client = client
        self._closed = False
        self.path = Path("typed-state-owner")
        self.database_path = self.path

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.close()

    def __enter__(self) -> TypedDatabaseTaskSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise TaskSourceIntegrityError("typed database task source is closed")

    def _all_records(self) -> tuple[tuple[TaskRecord, Mapping[str, Any]], ...]:
        self._require_open()
        rows = self._client.execute(
            "executor_task_projection_page", {"limit": MAX_QUERY_LIMIT, "offset": 0}
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
            records = self._all_records()
            after = self._client.load_generation()
            if before.content_id == after.content_id:
                row = rows[0]
                if int(row.get("task_count") or 0) != len(records):
                    raise TaskSourceBoundsError(
                        "typed task population exceeds its admitted projection bound"
                    )
                return row, records, after.revision
        raise TaskSourceConflictError("typed control projection changed during bounded snapshot")

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

    def list_tasks(
        self,
        status: str | Iterable[str] | None = None,
        cursor: str = "",
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> TaskPage:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
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
            or not 1 <= limit <= MAX_QUERY_LIMIT
        ):
            raise TaskSourceBoundsError(f"limit must be in [1, {MAX_QUERY_LIMIT}]")
        completed = {str(item).strip() for item in completed_ids if str(item).strip()}
        blocked = {str(item).strip() for item in blocked_ids if str(item).strip()}
        if completed & blocked:
            raise ValueError("completed_ids and blocked_ids must be disjoint")
        snapshot_row, stable_records, revision = self._snapshot_material()
        snapshot = self._snapshot_from_material(snapshot_row, stable_records, revision)
        records = [record for record, _identity in stable_records]
        by_identity = {
            identity: record
            for record in records
            for identity in (record.task_cid, record.task_alias)
        }
        ready: list[TaskRecord] = []
        for record in records:
            identities = {record.task_cid, record.task_alias}
            if identities & (completed | blocked) or record.status not in _READY_STATUSES:
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
        if not requested_status:
            raise TaskSourceIntegrityError("task status must not be empty")
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

    def get_queue_entry(self, _task_cid: str) -> None:
        """No cooldown row exists until a typed retry operation creates one."""

        return None


__all__ = [
    "TYPED_DATABASE_TASK_SOURCE_INTERFACE",
    "TYPED_DATABASE_TASK_SOURCE_SCHEMA",
    "TypedDatabaseTaskSource",
]
