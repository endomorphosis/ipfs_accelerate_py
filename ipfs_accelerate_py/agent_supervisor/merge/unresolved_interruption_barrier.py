"""Native task-scope barriers for UNKNOWN interruption cross-store recovery.

The existing lease event journal holds the barrier. No completion row, task
attempt status, or earlier event is repurposed. Ordinary native claimability
honors every pending barrier until the exact external admission is recorded.
"""
from __future__ import annotations

from typing import Any, Mapping
import re

SCHEMA = "ipfs_accelerate_py/native-unresolved-interruption-barrier@1"
PREPARED = "unresolved_interruption_prepared"
ADMITTED = "unresolved_interruption_admitted"


def _identity(value: Mapping[str, Any]) -> str:
    from .database_coordination import canonical_content_cid
    return canonical_content_cid(dict(value))


def _receipt_id(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _claim_identity(coordinator: Any, value: Any) -> dict[str, Any]:
    from . import database_coordination as dc
    raw = value.to_dict() if isinstance(value, dc.TaskClaim) else dict(value)
    for name in ("attempt_number", "fencing_token", "fence_epoch"):
        if type(raw.get(name)) is not int or raw[name] < 1:
            raise dc.DatabaseCoordinationStaleFenceError("native interruption claim scalar is invalid")
    return coordinator._task_claim_identity(raw)


def states(coordinator: Any, connection: Any, task_cid: str) -> dict[str, dict[str, Any]]:
    from . import database_coordination as dc
    scope = dc.exclusive_scope_key(lease_kind=dc.LeaseKind.TASK, scope=task_cid, task_cid=task_cid)
    rows = connection.execute(
        "SELECT event_id,event_type,lease_id,fencing_token,fence_epoch,body_json FROM lease_events "
        "WHERE scope_key=? AND event_type IN (?,?) LIMIT 8193", [scope, PREPARED, ADMITTED],
    ).fetchall()
    if len(rows) > 8192:
        raise dc.DatabaseCoordinationError("unresolved interruption journal bound exceeded")
    prepared: dict[str, dict[str, Any]] = {}
    admitted: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = dc._row_mapping(row)
        get = lambda name: dc._row_get(item, name)
        kind = str(get("event_type"))
        body = dc._decode_coordination_body(get("body_json"), table="lease_events", identity=str(get("event_id")))
        common = {"schema", "operation", "identity", "reservation_receipt_id", "control_revision", "barrier_id"}
        fields = common if kind == PREPARED else common | {"admission_receipt_id", "admission_control_revision", "record_id"}
        if (
            set(body) != fields or body.get("schema") != SCHEMA or body.get("operation") != kind
            or not _receipt_id(body.get("reservation_receipt_id"))
            or type(body.get("control_revision")) is not int or body["control_revision"] < 1
        ):
            raise dc.DatabaseCoordinationError("unresolved interruption barrier shape is invalid")
        claim = _claim_identity(coordinator, body["identity"])
        if (
            claim != body["identity"] or claim["task_cid"] != task_cid
            or claim["lease_id"] != get("lease_id")
            or claim["fencing_token"] != int(get("fencing_token"))
            or claim["fence_epoch"] != int(get("fence_epoch"))
        ):
            raise dc.DatabaseCoordinationStaleFenceError("unresolved interruption barrier identity is crossed")
        original = {key:value for key,value in body.items() if key not in {"barrier_id", "record_id", "admission_receipt_id", "admission_control_revision"}}
        original["operation"] = PREPARED
        barrier_id = _identity(original)
        if body["barrier_id"] != barrier_id:
            raise dc.DatabaseCoordinationError("unresolved interruption barrier digest differs")
        target = prepared if kind == PREPARED else admitted
        if barrier_id in target:
            raise dc.DatabaseCoordinationError("duplicate native interruption barrier event")
        if kind == ADMITTED and (
            not _receipt_id(body.get("admission_receipt_id"))
            or type(body.get("admission_control_revision")) is not int
            or body["admission_control_revision"] != body["control_revision"] + 2
            or body["record_id"] != _identity({key:value for key,value in body.items() if key != "record_id"})
        ):
            raise dc.DatabaseCoordinationError("unresolved interruption admission digest differs")
        target[barrier_id] = body
    if set(admitted) - set(prepared):
        raise dc.DatabaseCoordinationError("native interruption admission has no preparation")
    for key, value in admitted.items():
        paired = {name:item for name,item in value.items() if name not in {"admission_receipt_id", "admission_control_revision", "record_id"}}
        paired["operation"] = PREPARED
        if paired != prepared[key]:
            raise dc.DatabaseCoordinationError("native interruption admission preparation is crossed")
    return {
        key: {"prepared": value, "admitted": admitted.get(key)}
        for key, value in prepared.items()
    }


def pending(coordinator: Any, connection: Any, task_cid: str) -> bool:
    return any(value["admitted"] is None for value in states(coordinator, connection, task_cid).values())


def _guard(coordinator: Any, connection: Any, claim: Mapping[str, Any], now: int) -> Any:
    from . import database_coordination as dc
    scope = dc.exclusive_scope_key(lease_kind=dc.LeaseKind.TASK, scope=claim["task_cid"], task_cid=claim["task_cid"])
    coordinator._expire_scope(connection, scope, now)
    return coordinator._protect_task_claim_unlocked(
        connection, identity=claim, now=now, expected_attempt_status=dc.AttemptStatus.EXPIRED,
        expected_lease_state=dc.LeaseState.EXPIRED, allow_logically_completed=False, record_event=False,
    )


def prepare(coordinator: Any, claim: Any, *, reservation_receipt_id: str, control_revision: int, now_ms: int | None = None) -> dict[str, Any]:
    from . import database_coordination as dc
    value = _claim_identity(coordinator, claim)
    if type(control_revision) is not int or control_revision < 1 or not _receipt_id(reservation_receipt_id):
        raise dc.DatabaseCoordinationError("native interruption preparation binding is invalid")
    body = {"schema": SCHEMA, "operation": PREPARED, "identity": value,
            "reservation_receipt_id": reservation_receipt_id, "control_revision": control_revision}
    body["barrier_id"] = _identity(body)
    now = coordinator._now_ms() if now_ms is None else dc._nonneg_int(int(now_ms), "now_ms")
    with coordinator._lock:
        connection = coordinator._require()
        coordinator._begin(connection)
        try:
            existing = states(coordinator, connection, value["task_cid"])
            if body["barrier_id"] in existing:
                prior = existing[body["barrier_id"]]
                if prior["prepared"] != body:
                    raise dc.DatabaseCoordinationError("native interruption preparation conflicts")
                if prior["admitted"] is None:
                    _guard(coordinator, connection, value, now)
                coordinator._commit_if_idle(connection)
                return body
            if any(item["admitted"] is None for item in existing.values()):
                raise dc.DatabaseCoordinationConflictError("task already has another pending interruption")
            lease = _guard(coordinator, connection, value, now)
            coordinator._record_event(connection, lease_id=value["lease_id"], scope_key=lease.scope_key,
                event_type=PREPARED, fencing_token=value["fencing_token"], fence_epoch=value["fence_epoch"], observed_at_ms=now, body=body)
            coordinator._commit_if_idle(connection)
            return body
        except BaseException:
            coordinator._rollback_if_open(connection)
            raise


def get(coordinator: Any, claim: Any, reservation_receipt_id: str) -> dict[str, Any] | None:
    value = _claim_identity(coordinator, claim)
    with coordinator._lock:
        matches = [item for item in states(coordinator, coordinator._require(), value["task_cid"]).values()
            if item["prepared"]["identity"] == value and item["prepared"]["reservation_receipt_id"] == reservation_receipt_id]
        if len(matches) > 1:
            from .database_coordination import DatabaseCoordinationError
            raise DatabaseCoordinationError("ambiguous native interruption barrier")
        return matches[0] if matches else None


def admit(coordinator: Any, claim: Any, *, reservation_receipt_id: str, admission: Any, now_ms: int | None = None) -> dict[str, Any]:
    from . import database_coordination as dc
    from ..todo_daemon.unresolved_interruption import NativeContinuationAdmission
    if type(admission) is not NativeContinuationAdmission:
        raise dc.DatabaseCoordinationError("retained native continuation admission is required")
    value = _claim_identity(coordinator, claim)
    now = coordinator._now_ms() if now_ms is None else dc._nonneg_int(int(now_ms), "now_ms")
    with coordinator._lock:
        connection = coordinator._require()
        coordinator._begin(connection)
        try:
            matches = [item for item in states(coordinator, connection, value["task_cid"]).values()
                if item["prepared"]["identity"] == value and item["prepared"]["reservation_receipt_id"] == reservation_receipt_id]
            if len(matches) != 1:
                raise dc.DatabaseCoordinationError("exact native interruption preparation is required")
            state = matches[0]
            prepared = state["prepared"]
            admission_receipt, control_task = admission.require_current(coordinator)
            if (
                admission_receipt.get("operation") != "native_unresolved_continuation_admitted"
                or admission_receipt.get("reservation_receipt_id") != reservation_receipt_id
                or admission_receipt.get("callback_outcome") != "unknown"
                or admission_receipt.get("attempt_consumed") is not True
                or admission_receipt.get("settlement_authority") is not False
                or admission_receipt.get("completion_authority") is not False
                or admission_receipt.get("retry_authorized") is not True
                or admission_receipt.get("prior_evidence_reused") is not False
                or not _receipt_id(admission_receipt.get("receipt_id"))
                or _claim_identity(coordinator, admission_receipt.get("attempt", {})) != value
                or control_task.get("task_cid") != value["task_cid"]
                or control_task.get("status") != "retrying"
                or control_task.get("revision") != prepared["control_revision"] + 2
                or control_task.get("body", {}).get("completion_receipt") != dict(admission_receipt)
            ):
                raise dc.DatabaseCoordinationStaleFenceError("native interruption external admission is crossed")
            body = {**prepared, "operation": ADMITTED,
                "admission_receipt_id": admission_receipt["receipt_id"],
                "admission_control_revision": control_task["revision"]}
            body["record_id"] = _identity(body)
            if state["admitted"] is not None:
                if state["admitted"] != body:
                    raise dc.DatabaseCoordinationError("native interruption admission conflicts")
                coordinator._commit_if_idle(connection)
                return body
            lease = _guard(coordinator, connection, value, now)
            if admission.require_current(coordinator) != (admission_receipt, control_task):
                raise dc.DatabaseCoordinationStaleFenceError("native continuation admission changed before commit")
            coordinator._record_event(connection, lease_id=value["lease_id"], scope_key=lease.scope_key,
                event_type=ADMITTED, fencing_token=value["fencing_token"], fence_epoch=value["fence_epoch"], observed_at_ms=now, body=body)
            coordinator._commit_if_idle(connection)
            return body
        except BaseException:
            coordinator._rollback_if_open(connection)
            raise
