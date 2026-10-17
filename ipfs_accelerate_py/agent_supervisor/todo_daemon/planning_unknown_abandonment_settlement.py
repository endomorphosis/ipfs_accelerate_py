"""Audit-bound native local-authority settlement; UNKNOWN rows stay UNKNOWN.

This module requires genuine native handles and stopped-store custody supplied
by an independently admitted caller. It creates no owner, callback, lease or
connection replacement. A later operator rearm or successor is outside scope.
"""
from __future__ import annotations

from collections.abc import Mapping
import fcntl
import os
from pathlib import Path
import stat
from typing import Any

from . import planning_unknown_abandonment as policy
from ..task_sources.control_plane_contracts import content_identity


def native_stores(daemon: Any) -> dict[str, str]:
    from .implementation_daemon import DatabaseImplementationDaemon
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..task_sources.intent_repository import IntentRepository
    from ..task_sources.duckdb_state import DuckDBConnection
    from ..merge.database_coordination import DatabaseCoordinator, ProcessSerializedDatabaseCoordinator
    import duckdb

    policy.need(type(daemon) is DatabaseImplementationDaemon and daemon.require_real_execution is True,
                "abandonment requires the actual real-execution daemon")
    daemon.open()
    source, coordinator = daemon.task_source, daemon.coordinator
    policy.need(type(source) is DatabaseTaskSource and type(source.intent) is IntentRepository
                and not source.intent.uses_quack_transport
                and type(coordinator) in (DatabaseCoordinator, ProcessSerializedDatabaseCoordinator),
                "abandonment requires genuine native source/coordinator adapters")
    coordinator.open()
    lock = daemon._embedded_writer_lock_handle
    policy.need(lock is not None and not lock.closed, "native execution writer custody absent")
    witness = os.fstat(lock.fileno())
    observed = daemon._embedded_writer_lock_path.stat()
    policy.need((witness.st_dev, witness.st_ino) == (observed.st_dev, observed.st_ino)
                and stat.S_ISREG(witness.st_mode) and witness.st_uid == os.geteuid()
                and witness.st_nlink == 1 and not witness.st_mode & 0o002,
                "native execution writer lock identity differs")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def physical(connection):
        policy.need(type(connection) in (DuckDBConnection, duckdb.DuckDBPyConnection),
                    "replacement database connection refused")
        if type(connection) is DuckDBConnection:
            policy.need(type(connection._connection) is duckdb.DuckDBPyConnection and not connection._quack_uri,
                        "native physical file connection required")
        rows = connection.execute("SELECT path FROM duckdb_databases() WHERE database_name=current_database()").fetchall()
        policy.need(len(rows) == 1, "native physical store ambiguous")
        path = Path(rows[0]["path"] if isinstance(rows[0], Mapping) else rows[0][0])
        policy.need(path.is_absolute() and path.resolve(strict=True) == path, "native store path aliased")
        return str(path)
    with source.intent._connection(write=False) as connection:
        canonical = physical(connection)
    execution = physical(daemon._require_connection())
    if type(coordinator) is DatabaseCoordinator:
        coordination = physical(coordinator._require())
    else:
        stack, inner = coordinator._coordinator_context()
        with stack:
            coordination = physical(inner._require())
    result = {"canonical": canonical, "execution": execution, "coordination": coordination}
    policy.need(result == {"canonical": str(daemon.database_path.absolute()),
        "execution": str(daemon.execution_path.absolute()), "coordination": str(daemon.coordination_path.absolute())},
        "actual daemon handle/store crosswalk differs")
    return result


def provider_rows(daemon: Any, attempt_id: str) -> list[dict[str, Any]]:
    columns = ("invocation_id", "attempt_id", "task_cid", "idempotency_key", "owner_session_id", "recorded_at_ms", "result_json")
    rows = daemon._require_connection().execute(
        "SELECT invocation_id,attempt_id,task_cid,idempotency_key,owner_session_id,recorded_at_ms,result_json "
        "FROM provider_invocations WHERE attempt_id=? ORDER BY invocation_id LIMIT 3", [attempt_id]).fetchall()
    return [dict(row) if isinstance(row, Mapping) else dict(zip(columns, row)) for row in rows]


def verify_current(daemon: Any, attempt: Any, audit: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    context, snapshot = audit["context"], audit["snapshot"]
    policy.need(native_stores(daemon) == context["stores"], "ADMIN audit has another native physical store scope")
    original = snapshot["attempt"]
    actual = policy.plain(attempt.to_dict())
    mutable = {"status", "committed_phase", "revision", "finished_at_ms", "updated_at_ms"}
    policy.need({k: v for k, v in actual.items() if k not in mutable}
                == {k: v for k, v in original.items() if k not in mutable}, "native attempt immutable preimage changed")
    policy.need(provider_rows(daemon, attempt.attempt_id) == snapshot["providers"],
                "raw UNKNOWN provider row changed or accepted result appeared")
    effects = daemon._require_connection().execute(
        "SELECT effect_id FROM effect_claims WHERE attempt_id=? LIMIT 2", [attempt.attempt_id]).fetchall()
    policy.need(effects == [] and snapshot["effects"] == [], "incompatible execution effects appeared")
    phases = policy.plain(daemon.phase_history(attempt.attempt_id))
    if attempt.status == "running":
        policy.need(actual == original and phases == snapshot["phases"], "running native phase preimage changed")
    else:
        policy.need(attempt.status == "failed" and attempt.revision == original["revision"] + 1
                    and attempt.committed_phase == "failed" and attempt.finished_at_ms is not None
                    and phases[:-1] == snapshot["phases"] and len(phases) == len(snapshot["phases"]) + 1
                    and phases[-1] == {"phase": "failed", "revision": attempt.revision,
                        "fencing_token": original["fencing_token"], "fence_epoch": original["fence_epoch"],
                        "committed_at_ms": attempt.finished_at_ms, "body": dict(receipt)},
                    "native FAILED crash stage has another receipt")
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    lease = daemon.coordinator.get_lease(attempt.lease_id)
    coordinated = daemon.coordinator.get_task_attempt(attempt.attempt_id)
    policy.need(claim is not None and lease is not None and coordinated is not None, "native coordination authority disappeared")
    for record, keys in ((claim, policy.IDENTITY_FIELDS), (lease, policy.IDENTITY_FIELDS),
                         (coordinated, tuple(k for k in policy.IDENTITY_FIELDS if k not in {"claim_id", "lease_id"}))):
        observed = record.to_dict()
        policy.need(all(type(observed.get(k)) is type(original[k]) and observed.get(k) == original[k] for k in keys),
                    "native coordination identity/fence differs")
    states = tuple(str(getattr(value, "value", value)) for value in (claim.state, lease.state, coordinated.status))
    policy.need(states in {("accepted", "accepted", "running"), ("expired", "expired", "expired"),
                          ("released", "released", "failed")}, "coordination is outside exact failure crash stages")
    if attempt.status == "running":
        policy.need(states == ("accepted", "accepted", "running")
                    and policy.plain(claim.to_dict()) == snapshot["coordination"]["claim"]
                    and policy.plain(lease.to_dict()) == snapshot["coordination"]["lease"]
                    and policy.plain(coordinated.to_dict()) == snapshot["coordination"]["attempt"],
                    "original coordination preimage changed before FAILED")
    # Direct fixed reads use the already authenticated genuine coordinator
    # handle. They close the newer-owner window before changing execution.
    def check_latest(connection):
        rows = connection.execute("SELECT attempt_id,attempt_number FROM task_attempts WHERE task_cid=? "
            "ORDER BY attempt_number LIMIT 3", [attempt.task_cid]).fetchall()
        expected_history = [(previous["attempt_id"], previous["attempt_number"])
                            for previous in snapshot["coordination"]["previous_attempts"]]
        expected_history.append((attempt.attempt_id, attempt.attempt_number))
        policy.need([(row[0], int(row[1])) for row in rows] == expected_history
                    and [number for _, number in expected_history] == [1, 2],
                    "complete coordinator ordinal inventory differs or newer owner exists")
    from ..merge.database_coordination import DatabaseCoordinator
    if type(daemon.coordinator) is DatabaseCoordinator:
        check_latest(daemon.coordinator._require())
    else:
        stack, inner = daemon.coordinator._coordinator_context()
        with stack: check_latest(inner._require())
    for earlier in snapshot["coordination"]["previous_attempts"]:
        current = daemon.coordinator.get_task_attempt(earlier["attempt_id"])
        policy.need(current is not None and policy.plain(current.to_dict()) == earlier,
                    "historical attempt ordinal/fence/preimage changed")
    task = daemon.task_source.get_task(attempt.task_cid)
    intent = daemon.task_source.intent.get_task(attempt.task_cid)
    policy.need(task is not None and intent is not None, "canonical task disappeared")
    expected_intent = policy.plain(snapshot["intent_task"])
    if task.status == "in_progress":
        policy.need(task.to_dict() == snapshot["task"] and policy.plain(intent) == expected_intent,
                    "canonical claim-bound source preimage changed")
    else:
        policy.need(attempt.status == "failed" and states == ("released", "released", "failed")
                    and task.status == "blocked" and task.revision == context["expected_task_revision"] + 1,
                    "canonical task is newer, rearmed or outside BLOCKED crash stage")
        expected_intent["status"] = "blocked"
        expected_intent["revision"] += 1
        expected_intent["updated_at"] = intent["updated_at"]
        expected_intent["body"]["completion_receipt"] = dict(receipt)
        policy.need(policy.plain(intent) == expected_intent, "BLOCKED full Intent identity/body differs")
    population = policy.canonical_population(daemon.task_source)
    original_population = snapshot["canonical_population"]
    policy.need({k: v for k, v in population.items() if k != "tasks"}
                == {k: v for k, v in original_population.items() if k != "tasks"}, "canonical plan/goal/graph changed")
    policy.need([row for row in population["tasks"] if row["task_cid"] != attempt.task_cid]
                == [row for row in original_population["tasks"] if row["task_cid"] != attempt.task_cid],
                "unrelated canonical task changed")
    now_row = next(row for row in population["tasks"] if row["task_cid"] == attempt.task_cid)
    old_row = next(row for row in original_population["tasks"] if row["task_cid"] == attempt.task_cid)
    allowed = {"status", "revision", "updated_at", "body_json"}
    policy.need({k: v for k, v in now_row.items() if k not in allowed}
                == {k: v for k, v in old_row.items() if k not in allowed}, "raw canonical identity/creation fields changed")


def build_failure(daemon: Any, attempt: Any) -> dict[str, Any]:
    from .implementation_daemon import DATABASE_PORTAL_FAILURE_SETTLEMENT_SCHEMA
    binding = daemon._control_binding_for_attempt(attempt)
    audit = policy.exact_audit(daemon.task_source, attempt=attempt, binding=binding)
    receipt = {"schema": DATABASE_PORTAL_FAILURE_SETTLEMENT_SCHEMA, "operation": "database_task_claim_failure",
        "failure_kind": policy.FAILURE_KIND, "failure_payload_digest": policy.failure_payload(audit, binding),
        **{key: getattr(attempt, key) for key in policy.IDENTITY_FIELDS},
        "control_expected_status": binding["control_expected_status"],
        "control_expected_revision": binding["control_expected_revision"],
        **daemon._attempt_execution_evidence_counts(attempt.attempt_id), "automatic_retry_admitted": False}
    receipt["settlement_id"] = content_identity(receipt)
    verify_current(daemon, attempt, audit, receipt)
    return receipt


def validate_failure(daemon: Any, attempt: Any, receipt: Mapping[str, Any]) -> None:
    # No call to _portal_failure_phase_receipt or a receipt builder here: both
    # already invoke the daemon validator on crash replay.
    binding = daemon._control_binding_for_attempt(attempt)
    audit = policy.exact_audit(daemon.task_source, attempt=attempt, binding=binding)
    policy.need(receipt["failure_kind"] == policy.FAILURE_KIND
                and receipt["failure_payload_digest"] == policy.failure_payload(audit, binding),
                "failure kind/digest is not backed by its exact durable ADMIN audit")
    verify_current(daemon, attempt, audit, receipt)


def settle_planning_unknown_administrative_abandonment(daemon: Any, *, authority_context: Mapping[str, Any]) -> dict[str, Any]:
    policy.load_context(authority_context, check_closed_stores=False)
    stores = native_stores(daemon)
    snapshot = policy.load_context(authority_context, check_closed_stores=False)
    attempt = daemon.get_attempt(snapshot["attempt"]["attempt_id"])
    policy.need(attempt is not None, "registered native attempt disappeared")
    binding = daemon._control_binding_for_attempt(attempt)
    audit = policy.exact_audit(daemon.task_source, attempt=attempt, binding=binding)
    policy.need(audit["context"] == dict(authority_context), "settlement supplied another registered audit context")
    failure = build_failure(daemon, attempt)
    daemon._validate_portal_failure_settlement_receipt(failure, attempt=attempt)
    before_provider = provider_rows(daemon, attempt.attempt_id)
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    attempt = daemon._commit_portal_failure_phase(attempt, failure)
    daemon._fail_coordination_claim(claim, failure)
    control = daemon._settle_portal_failure_control_task(attempt, failure)
    policy.need(control["superseded"] is False and control["status"] == "blocked"
                and control["revision"] == authority_context["expected_task_revision"] + 1,
                "administrative closeout never accepts a later control generation")
    result = {"schema": "planning-unknown-administrative-native-settlement@1",
        "settlement_id": failure["settlement_id"], "audit_evidence_id": audit["evidence_id"],
        "audit_event_id": audit["event"]["event_id"], "context_cid": authority_context["context_cid"],
        "task_cid": attempt.task_cid, "attempt_id": attempt.attempt_id,
        "control_status": "blocked", "control_revision": control["revision"],
        "execution_status": "failed", "claim_state": "released", "coordination_attempt_status": "failed",
        "actual_store_bindings": stores, "failure_receipt": failure,
        "provider_outcome": "unknown", "possible_external_effects": "unverified",
        "callback_closed": False, "remote_cancellation_proved": False, "nonreachability_asserted": False,
        "raw_unknown_rows_preserved": True, "automatic_retry_admitted": False,
        "operator_rearm_performed": False, "successor_attempt_admitted": False, "completion_authority": False}
    if not daemon._terminal_portal_failure_event_exists(attempt.attempt_id, failure["settlement_id"]):
        daemon._record_event("terminal_portal_failure_settled", attempt_id=attempt.attempt_id,
                             task_cid=attempt.task_cid, body=result)
    terminal = daemon._require_connection().execute(
        "SELECT task_cid,event_type,body_json FROM daemon_execution_events WHERE attempt_id=? AND event_type=? LIMIT 2",
        [attempt.attempt_id, "terminal_portal_failure_settled"]).fetchall()
    policy.need(len(terminal) == 1 and terminal[0][0] == attempt.task_cid
                and terminal[0][1] == "terminal_portal_failure_settled"
                and policy.decode(terminal[0][2]) == result,
                "native administrative terminal event missing, ambiguous or different")
    final_attempt = daemon.get_attempt(attempt.attempt_id)
    verify_current(daemon, final_attempt, audit, failure)
    policy.need(provider_rows(daemon, attempt.attempt_id) == before_provider, "UNKNOWN raw row changed during closeout")
    return result
