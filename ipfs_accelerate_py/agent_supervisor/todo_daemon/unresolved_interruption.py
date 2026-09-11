"""Distinct native UNKNOWN interruption reservation and continuation CAS.

Only the declared native Doctor producer is currently supported. Historical
unrestricted callbacks are refused. Original execution rows, phases, provider
receipts and spent attempt ordinals are never rewritten by these operations.
"""
from __future__ import annotations

import json
from typing import Any

from .doctor_interruption_custody import DoctorInterruptionCustody, observations
from .native_doctor_callback import NativeDoctorCallback, DoctorCallbackDenied, identity, canonical, PROFILE_KEY
from ..merge.worktree_lifecycle import WorktreeLifecycleStore

SCHEMA = "ipfs_accelerate_py/native-unresolved-interruption@1"
RESERVED = "native_unresolved_interruption_reserved"
ADMITTED = "native_unresolved_continuation_admitted"


def _event_id(kind: str, attempt_id: str) -> str:
    return identity({"schema": SCHEMA, "kind": kind, "attempt_id": attempt_id})


def _record(daemon: Any, kind: str, attempt: Any, body: dict[str, Any]) -> dict[str, Any]:
    existing = _read(daemon, kind, attempt)
    if existing is not None:
        if existing != body:
            raise DoctorCallbackDenied("native interruption record conflicts")
        return existing
    daemon._require_connection().execute(
        "INSERT INTO daemon_execution_events "
        "(event_id,attempt_id,task_cid,event_type,recorded_at_ms,body_json) VALUES (?,?,?,?,?,?)",
        [_event_id(kind, attempt.attempt_id), attempt.attempt_id, attempt.task_cid,
         kind, daemon._now_ms(), canonical(body).decode()],
    )
    return body


def _read(daemon: Any, kind: str, attempt: Any) -> dict[str, Any] | None:
    row = daemon._require_connection().execute(
        "SELECT attempt_id,task_cid,event_type,body_json FROM daemon_execution_events WHERE event_id=?",
        [_event_id(kind, attempt.attempt_id)],
    ).fetchone()
    if row is None:
        return None
    body = json.loads(row[3])
    common = {"schema", "operation", "attempt", "callback_outcome", "attempt_consumed",
              "settlement_authority", "completion_authority", "receipt_id",
              "prior_control_revision", "resulting_control_revision"}
    fields = common | ({"scope"} if kind == RESERVED else {
        "reservation_receipt_id", "retry_authorized", "prior_evidence_reused",
    })
    if (
        type(body) is not dict or set(body) != fields
        or type(body.get("prior_control_revision")) is not int
        or type(body.get("resulting_control_revision")) is not int
        or body["prior_control_revision"] < 1
        or body["resulting_control_revision"] != body["prior_control_revision"] + 1
        or
        row[0] != attempt.attempt_id or row[1] != attempt.task_cid or row[2] != kind
        or body.get("schema") != SCHEMA or body.get("operation") != kind
        or body.get("attempt") != attempt.to_dict()
        or body.get("callback_outcome") != "unknown"
        or body.get("settlement_authority") is not False
        or body.get("completion_authority") is not False
        or body.get("attempt_consumed") is not True
        or (kind == ADMITTED and (body.get("retry_authorized") is not True or body.get("prior_evidence_reused") is not False))
        or body.get("receipt_id") != identity({key:value for key,value in body.items() if key != "receipt_id"})
    ):
        raise DoctorCallbackDenied("native interruption record is not exact")
    return body


def _body(kind: str, attempt: Any, **values: Any) -> dict[str, Any]:
    body = {
        "schema": SCHEMA, "operation": kind, "attempt": attempt.to_dict(),
        "callback_outcome": "unknown", "attempt_consumed": True,
        "settlement_authority": False, "completion_authority": False,
        **values,
    }
    return {**body, "receipt_id": identity(body)}


def _require_unknown(daemon: Any, attempt: Any) -> Any:
    current = daemon.get_attempt(attempt.attempt_id)
    if (
        current != attempt or attempt.status != "running"
        or attempt.committed_phase != "context" or attempt.revision != 2
        or PROFILE_KEY not in attempt.body
        or [row["phase"] for row in daemon.phase_history(attempt.attempt_id)] != ["claimed", "context"]
    ):
        raise DoctorCallbackDenied("exact declared unknown context attempt is required")
    counts = daemon._attempt_execution_evidence_counts(attempt.attempt_id)
    if any(int(counts[name]) for name in ("provider_invocation_count", "effect_claim_count")):
        raise DoctorCallbackDenied("accepted execution evidence requires its existing settlement route")
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    if claim is None or not daemon._claim_matches_execution_attempt(claim, attempt):
        raise DoctorCallbackDenied("native interruption claim identity changed")
    return claim


def _cas(daemon: Any, attempt: Any, *, expected: int, before: str, after: str, receipt: dict[str, Any]) -> Any:
    task = daemon.task_source.get(attempt.task_cid)
    if (
        task is not None and task.status == after and task.revision == expected + 1
        and task.body.get("completion_receipt") == receipt
    ):
        return task
    if task is None or task.status != before or task.revision != expected:
        raise DoctorCallbackDenied("interruption control CAS observation changed")
    try:
        result = daemon._cas_task_status_database(
            attempt.task_cid, expected_revision=expected, new_status=after, receipt=receipt,
        )
    except Exception:
        observed = daemon.task_source.get(attempt.task_cid)
        if (
            observed is None or observed.status != after or observed.revision != expected + 1
            or observed.body.get("completion_receipt") != receipt
        ):
            raise
        return observed
    observed = getattr(result, "task", None)
    if (
        observed is None or observed.status != after or observed.revision != expected + 1
        or observed.body.get("completion_receipt") != receipt
    ):
        raise DoctorCallbackDenied("interruption CAS did not produce exact native task")
    return observed


def reserve(daemon: Any, attempt: Any, callback: NativeDoctorCallback) -> dict[str, Any]:
    """Fence one elapsed unknown attempt; no callback or completion is replayed."""
    with daemon._lock:
        claim = _require_unknown(daemon, attempt)
        with DoctorInterruptionCustody(daemon, attempt, callback) as held:
            scope = held.require_current()
            expected = int(attempt.body["control_binding"]["control_expected_revision"])
            task = daemon.task_source.get(attempt.task_cid)
            existing = _read(daemon, RESERVED, attempt)
            receipt = _body(
                RESERVED, attempt, scope=scope,
                prior_control_revision=expected, resulting_control_revision=expected + 1,
            )
            if existing is not None and existing != receipt:
                raise DoctorCallbackDenied("interruption reservation changed")
            if task is None:
                raise DoctorCallbackDenied("native control task is missing")
            if task.status == "in_progress" and (
                task.revision != expected or not daemon._task_has_exact_database_claim_receipt(task, claim)
            ):
                raise DoctorCallbackDenied("interruption does not bind the current native control claim")
            # This native operation checks elapsed time AND the latest exact
            # claim/fence even when expiry was already recorded.
            daemon.coordinator.prepare_unresolved_interruption(
                claim, reservation_receipt_id=receipt["receipt_id"],
                control_revision=expected, now_ms=daemon._now_ms(),
            )
            held.quarantine({"schema": SCHEMA, "reservation_receipt_id": receipt["receipt_id"]})
            held.require_current()
            _cas(daemon, attempt, expected=expected, before="in_progress", after="blocked", receipt=receipt)
            return _record(daemon, RESERVED, attempt, receipt)


def admit_continuation(daemon: Any, attempt: Any, callback: NativeDoctorCallback) -> dict[str, Any]:
    """A distinct operator action admits a new attempt after exact custody."""
    with daemon._lock:
        prior_admission = _read(daemon, ADMITTED, attempt)
        if prior_admission is not None and continuation_admitted(daemon, attempt):
            return prior_admission
        claim = _require_unknown(daemon, attempt)
        limits = [callback.max_attempts]
        configured = getattr(daemon, "max_task_attempts", 0)
        if type(configured) is not int or configured < 0:
            raise DoctorCallbackDenied("native continuation attempt budget is invalid")
        if configured:
            limits.append(configured)
        if attempt.attempt_number >= min(limits):
            raise DoctorCallbackDenied("native continuation spent attempt budget is exhausted")
        reserved = _read(daemon, RESERVED, attempt)
        if reserved is None:
            raise DoctorCallbackDenied("continuation requires a durable native reservation")
        with DoctorInterruptionCustody(daemon, attempt, callback) as held:
            if held.require_current() != reserved["scope"]:
                raise DoctorCallbackDenied("continuation scope changed")
            held.quarantine({"schema": SCHEMA, "reservation_receipt_id": reserved["receipt_id"]})
            daemon.coordinator.expire_task_claim(claim, now_ms=daemon._now_ms())
            expected = reserved["resulting_control_revision"]
            receipt = _body(
                ADMITTED, attempt, reservation_receipt_id=reserved["receipt_id"],
                prior_control_revision=expected, resulting_control_revision=expected + 1,
                retry_authorized=True, prior_evidence_reused=False,
            )
            _cas(daemon, attempt, expected=expected, before="blocked", after="retrying", receipt=receipt)
            held.require_current()
            admitted = _record(daemon, ADMITTED, attempt, receipt)
            daemon.coordinator.admit_unresolved_interruption(
                claim, reservation_receipt_id=reserved["receipt_id"],
                admission=NativeContinuationAdmission(daemon, attempt, held),
                now_ms=daemon._now_ms(),
            )
            return admitted


class NativeContinuationAdmission:
    """Non-serializable custody and native-store reader, never a JSON receipt.

    Construction grants no authority by itself. Every use requires the retained
    native writer lock, full unchanged candidate, exact quarantine, original
    unknown execution history, both native events and the actual control CAS.
    """

    def __init__(self, daemon: Any, attempt: Any, custody: DoctorInterruptionCustody):
        from .implementation_daemon import DatabaseImplementationDaemon
        if (
            type(daemon) is not DatabaseImplementationDaemon
            or type(custody) is not DoctorInterruptionCustody or custody.attempt != attempt
            or custody.daemon is not daemon
        ):
            raise DoctorCallbackDenied("exact native continuation custody is required")
        self.daemon, self.attempt, self.custody = daemon, attempt, custody
        self.require_current()

    def require_current(self, coordinator: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
        from ..merge.database_coordination import DatabaseCoordinator, ProcessSerializedDatabaseCoordinator
        from .implementation_daemon import DatabaseImplementationDaemon
        daemon, attempt, held = self.daemon, self.attempt, self.custody
        if type(daemon) is not DatabaseImplementationDaemon or held.daemon is not daemon:
            raise DoctorCallbackDenied("native continuation daemon identity changed")
        bound = daemon.coordinator
        if coordinator is not None and not (
            type(coordinator) is DatabaseCoordinator
            and (coordinator is bound or (
                type(bound) is ProcessSerializedDatabaseCoordinator and bound.is_open
                and bound.database_path == coordinator.database_path
            ))
        ):
            raise DoctorCallbackDenied("native continuation coordinator binding differs")
        budget = held.callback.max_attempts
        configured = daemon.max_task_attempts
        if type(configured) is not int or configured < 0:
            raise DoctorCallbackDenied("native continuation attempt budget is invalid")
        if attempt.attempt_number >= min(budget, configured or budget):
            raise DoctorCallbackDenied("native continuation spent attempt budget is exhausted")
        if (
            type(held) is not DoctorInterruptionCustody or held.attempt != attempt
            or daemon.get_attempt(attempt.attempt_id) != attempt
            or attempt.status != "running" or attempt.committed_phase != "context"
            or [row["phase"] for row in daemon.phase_history(attempt.attempt_id)] != ["claimed", "context"]
        ):
            raise DoctorCallbackDenied("native continuation execution custody changed")
        counts = daemon._attempt_execution_evidence_counts(attempt.attempt_id)
        if any(int(counts[name]) for name in ("provider_invocation_count", "effect_claim_count")):
            raise DoctorCallbackDenied("native continuation has accepted execution evidence")
        reserved = _read(daemon, RESERVED, attempt)
        admitted = _read(daemon, ADMITTED, attempt)
        if (
            reserved is None or admitted is None
            or admitted["reservation_receipt_id"] != reserved["receipt_id"]
            or admitted["prior_control_revision"] != reserved["resulting_control_revision"]
            or reserved["scope"] != held.require_current()
        ):
            raise DoctorCallbackDenied("native continuation execution admission is missing or crossed")
        started, closed = observations(daemon, attempt)
        if started != held.started or closed != held.closed:
            raise DoctorCallbackDenied("native continuation callback observation changed")
        lifecycle = WorktreeLifecycleStore(repo_root=started["profile"]["repository_root"])
        quarantine = lifecycle.load_quarantine(started["worktree_path"])
        if (
            quarantine is None or quarantine["lifecycle_record"] != started["lifecycle"]
            or quarantine["reason"] != "native_unresolved_interruption"
            or quarantine["fence_authority"] != {"schema": SCHEMA, "reservation_receipt_id": reserved["receipt_id"]}
        ):
            raise DoctorCallbackDenied("native continuation quarantine admission differs")
        task = daemon.task_source.get(attempt.task_cid)
        if (
            task is None or task.status != "retrying"
            or task.revision != admitted["resulting_control_revision"]
            or task.body.get("completion_receipt") != admitted
        ):
            raise DoctorCallbackDenied("native continuation control admission is not current")
        return admitted, task.to_dict()


def continuation_admitted(daemon: Any, attempt: Any) -> bool:
    """Read native admission history; JSON status files never enter this route."""
    admitted = _read(daemon, ADMITTED, attempt)
    if admitted is None:
        return False
    reserved = _read(daemon, RESERVED, attempt)
    if (
        reserved is None or admitted.get("reservation_receipt_id") != reserved["receipt_id"]
        or admitted.get("retry_authorized") is not True
        or admitted.get("prior_evidence_reused") is not False
        or admitted.get("prior_control_revision") != reserved.get("resulting_control_revision")
        or admitted.get("resulting_control_revision") != admitted["prior_control_revision"] + 1
    ):
        raise DoctorCallbackDenied("native continuation lineage is invalid")
    barrier = daemon.coordinator.get_unresolved_interruption(
        attempt.to_dict(), reserved["receipt_id"],
    )
    if barrier is None or barrier["admitted"] is None:
        return False
    if barrier["admitted"]["admission_receipt_id"] != admitted["receipt_id"]:
        raise DoctorCallbackDenied("native continuation task barrier is crossed")
    started, closed = observations(daemon, attempt)
    scope = reserved.get("scope")
    if (
        type(scope) is not dict
        or scope.get("started_observation_id") != started["observation_id"]
        or scope.get("closed_observation_id") != closed["observation_id"]
        or scope.get("candidate_id") != identity(closed["observed_candidate"])
        or scope.get("lifecycle_record_id") != started["lifecycle"]["record_id"]
        or reserved["prior_control_revision"]
        != attempt.body["control_binding"]["control_expected_revision"]
    ):
        raise DoctorCallbackDenied("native continuation evidence binding changed")
    lifecycle = WorktreeLifecycleStore(repo_root=started["profile"]["repository_root"])
    quarantine = lifecycle.load_quarantine(started["worktree_path"])
    if (
        quarantine is None or quarantine["lifecycle_record"] != started["lifecycle"]
        or quarantine["reason"] != "native_unresolved_interruption"
        or quarantine["fence_authority"] != {
            "schema": SCHEMA, "reservation_receipt_id": reserved["receipt_id"],
        }
    ):
        raise DoctorCallbackDenied("native continuation quarantine binding changed")
    # Admission remains historical after a distinct claim advances the task.
    # It never authorizes the old callback, task completion or evidence reuse.
    return True


def reconcile_declared(daemon: Any) -> list[dict[str, Any]]:
    """Recover only future callbacks admitted with the bounded native profile."""
    callback = daemon._provider_fn
    if type(callback) is not NativeDoctorCallback:
        return []
    results = []
    for attempt in daemon.list_running_attempts():
        if PROFILE_KEY not in attempt.body:
            continue
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        if claim is None or int(claim.expires_at_ms) > daemon._now_ms():
            continue
        try:
            reserved = _read(daemon, RESERVED, attempt)
            if reserved is None:
                reserved = reserve(daemon, attempt, callback)
            admitted = admit_continuation(daemon, attempt, callback)
            results.append({"attempt_id": attempt.attempt_id, "continuation_receipt_id": admitted["receipt_id"], "retry_authorized": True})
        except (DoctorCallbackDenied, OSError) as exc:
            # No callback is retried by this error path. Partial durable CASes
            # remain visible and their exact stages can resume on the next pass.
            results.append({"attempt_id": attempt.attempt_id, "reason": str(exc), "retry_authorized": False})
    return results
