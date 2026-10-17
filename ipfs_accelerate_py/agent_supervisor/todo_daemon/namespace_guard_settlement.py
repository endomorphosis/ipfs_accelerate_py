"""Separate native failure settlement after an exact ADMIN adjudication audit.

The audit transaction must precede daemon construction/open: native open is
allowed to update its own metadata. This adapter checks durable target records
after that legitimate open, and calls the existing crash-resumable native
failure settlement. It never edits an UNKNOWN invocation or admits a retry.
The caller must run under the original finite owner with exclusive custody of
the closed execution/coordinator stores; this module does not create an owner,
credential, callback, replacement connection or task claim.
"""
from __future__ import annotations

from collections.abc import Mapping
import fcntl
import os
from pathlib import Path
import stat
from typing import Any

from . import namespace_guard_adjudication as proof

SETTLEMENT_SCHEMA = "namespace-guard-adjudicated-native-settlement@1"


def _native_stores(daemon: Any) -> dict[str, str]:
    """Authenticate real native adapters and their live physical connections.

    Return physical paths explicitly. The production owner must bind these to
    its original registered stores; an isolated-copy qualification reports
    its copy paths and is never a campaign recovery receipt.
    """
    from .implementation_daemon import DatabaseImplementationDaemon
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..task_sources.intent_repository import IntentRepository
    from ..task_sources.duckdb_state import DuckDBConnection
    from ..merge.database_coordination import DatabaseCoordinator, ProcessSerializedDatabaseCoordinator
    import duckdb

    proof._need(type(daemon) is DatabaseImplementationDaemon and daemon.require_real_execution is True,
                "recovery requires the original real native daemon")
    daemon.open()
    source = daemon.task_source
    coordinator = daemon.coordinator
    proof._need(type(source) is DatabaseTaskSource and type(source.intent) is IntentRepository
                and not source.intent.uses_quack_transport
                and type(coordinator) in (DatabaseCoordinator, ProcessSerializedDatabaseCoordinator),
                "recovery requires actual native source/coordinator adapters")
    # Injecting a genuine coordinator into the public daemon constructor does
    # not open it. Administrative recovery must use its real lifecycle API.
    coordinator.open()
    lock = daemon._embedded_writer_lock_handle
    proof._need(lock is not None and not lock.closed, "native execution writer custody is absent")
    fd = lock.fileno()
    identity = os.fstat(fd)
    observed = daemon._embedded_writer_lock_path.stat()
    proof._need((identity.st_dev, identity.st_ino) == (observed.st_dev, observed.st_ino)
                and stat.S_ISREG(identity.st_mode) and identity.st_uid == os.geteuid()
                and identity.st_nlink == 1 and not identity.st_mode & 0o002,
                "native execution writer lock inode differs")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def physical(connection: Any) -> str:
        proof._need(type(connection) in (DuckDBConnection, duckdb.DuckDBPyConnection),
                    "database connection is not an original native connection")
        if type(connection) is DuckDBConnection:
            proof._need(type(connection._connection) is duckdb.DuckDBPyConnection
                        and not connection._quack_uri,
                        "administrative recovery requires genuine exclusive file custody")
        rows = connection.execute("SELECT path FROM duckdb_databases() WHERE database_name=current_database()").fetchall()
        proof._need(len(rows) == 1, "native physical database binding is ambiguous")
        row = rows[0]
        value = row["path"] if isinstance(row, Mapping) else row[0]
        path = Path(value)
        proof._need(path.is_absolute() and path.resolve(strict=True) == path,
                    "native database physical path is aliased")
        return str(path)
    with source.intent._connection(write=False) as connection:
        canonical = physical(connection)
    execution = physical(daemon._require_connection())
    coordination = str(coordinator.database_path.absolute())
    if type(coordinator) is DatabaseCoordinator:
        actual_coordination = physical(coordinator._require())
    else:
        coordinator_stack, inner = coordinator._coordinator_context()
        with coordinator_stack:
            actual_coordination = physical(inner._require())
    proof._need(canonical == str(daemon.database_path.absolute())
                and execution == str(daemon.execution_path.absolute())
                and coordination == str(daemon.coordination_path.absolute())
                and actual_coordination == coordination
                and Path(coordination).resolve(strict=True) == Path(coordination),
                "native daemon stores differ from their actual adapters")
    return {"canonical": canonical, "execution": execution, "coordination": coordination}


def _rows(daemon: Any, table: str, attempt_id: str) -> list[dict[str, Any]]:
    # Literal private native-table reads, never caller SQL. The object is an
    # actual DatabaseImplementationDaemon with its original public open().
    statements = {
        "provider_invocations": "SELECT invocation_id,attempt_id,task_cid,idempotency_key,owner_session_id,recorded_at_ms,result_json FROM provider_invocations WHERE attempt_id = ? ORDER BY invocation_id LIMIT 3",
        "effect_claims": "SELECT effect_id,attempt_id,task_cid,effect_key,idempotency_key,owner_session_id,recorded_at_ms,result_json FROM effect_claims WHERE attempt_id = ? ORDER BY effect_id LIMIT 2",
    }
    column_names = {
        "provider_invocations": ("invocation_id","attempt_id","task_cid","idempotency_key","owner_session_id","recorded_at_ms","result_json"),
        "effect_claims": ("effect_id","attempt_id","task_cid","effect_key","idempotency_key","owner_session_id","recorded_at_ms","result_json"),
    }
    connection = daemon._require_connection()
    cursor = connection.execute(statements[table], [attempt_id])
    columns = column_names[table]
    return [dict(row) if isinstance(row, Mapping) else dict(zip(columns, row))
            for row in cursor.fetchall()]


def _find_native_audit(daemon: Any, adjudication: Mapping[str, Any]) -> dict[str, Any]:
    evidence_id = proof._cid({"namespace": proof.EVIDENCE_KIND,
                              "task_cid": proof.TASK_CID, "proof_id": adjudication["proof_id"]})
    # Administrative adjudication is immutable history, not expiring current
    # completion evidence. Preserve the repository's normal freshness policy;
    # read only this exact protected audit through its real native connection.
    columns = ("evidence_id", "parent_evidence_id", "task_cid", "evidence_kind",
               "digest", "created_at", "body_json")
    with daemon.task_source.intent._connection(write=False) as connection:
        rows = connection.execute(
            "SELECT evidence_id,parent_evidence_id,task_cid,evidence_kind,digest,created_at,body_json "
            "FROM evidence_nodes WHERE evidence_id = ? AND task_cid = ? AND evidence_kind = ? LIMIT 2",
            [evidence_id, proof.TASK_CID, proof.EVIDENCE_KIND],
        ).fetchall()
    actual = []
    for row in rows:
        record = dict(row) if isinstance(row, Mapping) else dict(zip(columns, row))
        record["body"] = proof._decode(record.pop("body_json"))
        actual.append(record)
    audit = proof._one(actual, noun="durable canonical ADMIN audit")
    proof._need(audit["task_cid"] == proof.TASK_CID and audit["evidence_kind"] == proof.EVIDENCE_KIND
                and audit["parent_evidence_id"] == "" and audit["digest"] == proof._cid(adjudication)
                and dict(audit["body"]) == adjudication, "canonical ADMIN audit binding differs")
    return audit


def _audit_event(daemon: Any, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Read the unique real native event, without manufacturing an event ID."""
    evidence_id = audit["evidence_id"]
    matches = []
    after = 0
    for _page in range(16):
        rows = daemon.task_source.intent.list_events(after_global_sequence=after, limit=256)
        if not rows:
            break
        for row in rows:
            if row["body"].get("subject_id") == evidence_id:
                matches.append(dict(row))
        after = rows[-1]["global_sequence"]
    else:
        proof._need(False, "native audit event scan exceeds bound")
    event = proof._one(matches, noun="native ADMIN audit event")
    proof._need(event["event_type"] == "intent.evidence_recorded"
                and event["task_cid"] == proof.TASK_CID
                and event["body"]["body"]["evidence_id"] == evidence_id,
                "native ADMIN audit event subject differs")
    envelope = event["body"]
    proof._need(set(envelope) == {"schema", "event_type", "subject_id", "body", "recorded_at", "owner_id"}
                and envelope["event_type"] == event["event_type"]
                and envelope["recorded_at"] == event["recorded_at"]
                and envelope["body"] == {"evidence_id": evidence_id, "parent_evidence_id": "",
                    "task_cid": proof.TASK_CID, "evidence_kind": proof.EVIDENCE_KIND,
                    "digest": audit["digest"], "body": audit["body"],
                    "created_at": audit["created_at"], "revision": 0}
                and event["event_id"] == proof._cid({key: event[key] for key in
                    ("stream_id", "sequence", "global_sequence", "event_type", "body")}),
                "native ADMIN audit event/content binding differs")
    return event


def settle_namespace_guard_adjudicated_attempt(
    daemon: Any, *, authority_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Settle only the administrator-adjudicated retained native attempt.

    Crash stages remain the original native ones: durable FAILED execution
    phase, coordinator FAILED/released claim, canonical BLOCKED CAS, then its
    unique terminal settlement event. An exact interrupted stage resumes;
    later operator rearming is a separate operation and is refused here.
    """
    from .implementation_daemon import DatabaseImplementationDaemon
    from .database_portal_bridge import DatabasePortalBridgeError

    stores = _native_stores(daemon)
    # The real audit is already durable. Full historical proof inputs remain
    # pinned; whole old sidecar file hashes are intentionally not reused after
    # the original native open has legitimately written its own metadata.
    snapshot, identity = proof._load_authority(authority_context, check_live_stores=False)
    adjudication = proof._adjudication_receipt(snapshot, identity, authority_context)
    audit = _find_native_audit(daemon, adjudication)
    audit_event = _audit_event(daemon, audit)
    expected = snapshot["execution"]["2"]
    before_provider = _rows(daemon, "provider_invocations", proof.ATTEMPT_ID)
    before_effect = _rows(daemon, "effect_claims", proof.ATTEMPT_ID)
    proof._need(before_provider == expected["provider_invocations"] and before_effect == [],
                "preserved native UNKNOWN/effect rows changed after adjudication")
    original = expected["database_task_attempts"][0]
    attempt = daemon.get_attempt(proof.ATTEMPT_ID)
    proof._need(attempt is not None, "adjudicated execution attempt disappeared")
    proof._same_identity(attempt.to_dict(), identity)
    proof._need(attempt.task_alias == original["task_alias"]
                and attempt.started_at_ms == original["started_at_ms"]
                and dict(attempt.body) == proof._decode(original["body_json"]),
                "native attempt immutable preimage changed")
    history = [dict(item) for item in daemon.phase_history(proof.ATTEMPT_ID)]
    # Original public phase_history returns decoded native bodies, so compare
    # precisely against the archived SQL row preimage rather than JSON syntax.
    expected_phases = [{**{k: v for k, v in row.items() if k not in ("body_json", "attempt_id")},
                        "body": proof._decode(row["body_json"])} for row in expected["attempt_phases"]]
    proof._need(history[:2] == expected_phases, "native claimed/context phase history changed")
    failure = DatabasePortalBridgeError(proof.GUARD_REASON)
    expected_failure = daemon._build_portal_failure_settlement_receipt(attempt, failure=failure)
    proof._need(expected_failure["provider_invocation_count"] == 1
                and expected_failure["effect_claim_count"] == 0, "native failure custody count differs")
    if attempt.status == "running":
        proof._need(attempt.revision == 2 and attempt.committed_phase == "context"
                    and attempt.finished_at_ms is None and len(history) == 2,
                    "unsettled native attempt phase changed")
    else:
        proof._need(attempt.status == "failed" and attempt.revision == 3
                    and attempt.committed_phase == "failed" and attempt.finished_at_ms is not None
                    and len(history) == 3 and history[-1]["phase"] == "failed"
                    and history[-1]["revision"] == 3 and history[-1]["body"] == expected_failure,
                    "crash-resume native FAILED receipt differs")
    claim = daemon.coordinator.get_task_claim(proof.CLAIM_ID)
    coord_attempt = daemon.coordinator.get_task_attempt(proof.ATTEMPT_ID)
    lease = daemon.coordinator.get_lease(identity["lease_id"])
    proof._need(claim is not None and coord_attempt is not None and lease is not None,
                "adjudicated coordination identity disappeared")
    proof._same_identity(claim.to_dict(), identity)
    proof._same_identity(lease.to_dict(), identity)
    proof._same_identity(coord_attempt.to_dict(), identity,
                         tuple(k for k in proof._IDENTITY_FIELDS if k not in ("claim_id", "lease_id")))
    state = str(getattr(claim.state, "value", claim.state))
    coord_state = str(getattr(coord_attempt.status, "value", coord_attempt.status))
    lease_state = str(getattr(lease.state, "value", lease.state))
    proof._need((state, coord_state, lease_state) in {
                    ("accepted", "running", "accepted"), ("released", "failed", "released")},
                "coordination is outside native failure crash stages")
    if attempt.status == "running":
        proof._need(state == "accepted"
                    and claim.to_dict() == snapshot["coordination"]["native_claim_record_projection"],
                    "original retained claim changed before first native settlement")
    task = daemon.task_source.get_task(proof.TASK_CID)
    proof._need(task is not None, "canonical adjudicated task disappeared")
    if task.status == "in_progress":
        proof._need(task.to_dict() == snapshot["task"], "canonical source contract changed before settlement")
    else:
        body = dict(snapshot["task"]["body"])
        body["completion_receipt"] = expected_failure
        proof._need(attempt.status == "failed" and task.status == "blocked" and task.revision == 8
                    and dict(task.body) == body, "control task is outside native settlement crash stages")
    result = daemon._settle_terminal_portal_failure(attempt, failure=failure)
    after_provider = _rows(daemon, "provider_invocations", proof.ATTEMPT_ID)
    after_effect = _rows(daemon, "effect_claims", proof.ATTEMPT_ID)
    proof._need(after_provider == before_provider and after_effect == before_effect
                and result["provider_invocation_count"] == 1 and result["effect_claim_count"] == 0
                and result["execution_status"] == "failed" and result["control_status"] == "blocked"
                and result["control_revision"] == 8 and result["control_superseded"] is False,
                "native settlement did not preserve exact UNKNOWN history")
    final_attempt = daemon.get_attempt(proof.ATTEMPT_ID)
    final_task = daemon.task_source.get_task(proof.TASK_CID)
    proof._need(final_attempt is not None and final_task is not None, "native terminal readback missing")
    terminal_receipt = daemon._portal_failure_phase_receipt(final_attempt)
    proof._need(terminal_receipt == expected_failure, "native terminal settlement identity changed")
    return {
        "schema": SETTLEMENT_SCHEMA, "adjudication_id": adjudication["adjudication_id"],
        "actual_store_bindings": stores,
        "audit_evidence_id": audit["evidence_id"], "audit_digest": audit["digest"],
        "audit_event_id": audit_event["event_id"], "audit_event": audit_event,
        "native_settlement": dict(result), "native_failure_receipt": dict(terminal_receipt),
        "task": final_task.to_dict(), "attempt": final_attempt.to_dict(),
        "provider_invocations": after_provider, "effect_claims": after_effect,
        "unknown_preserved": True, "spent_attempt_ordinal_preserved": True,
        "operator_rearm_performed": False, "successor_attempt_admitted": False,
        "completion_authority": False, "provider_exit_inferred": False,
    }


def rearm_namespace_guard_settled_attempt(
    daemon: Any, *, authority_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Perform the distinct operator-control rearm after native settlement.

    This deliberately is not called by settlement or adjudication. The caller
    explicitly chooses this later operation under administrator custody. Its
    original native stages are BLOCKED→RETRYING control CAS followed by exact
    coordinator rearm; either response-loss boundary resumes idempotently.
    Attempt ordinal1 and its UNKNOWN invocation remain historical rows. The
    later packet/budget update and successor dispatch are separate operations.
    """
    from .implementation_daemon import DatabaseImplementationDaemon
    from .database_portal_bridge import DatabasePortalBridgeError

    stores = _native_stores(daemon)
    snapshot, identity = proof._load_authority(authority_context, check_live_stores=False)
    adjudication = proof._adjudication_receipt(snapshot, identity, authority_context)
    audit = _find_native_audit(daemon, adjudication)
    event = _audit_event(daemon, audit)
    original = snapshot["execution"]["2"]
    provider = _rows(daemon, "provider_invocations", proof.ATTEMPT_ID)
    effects = _rows(daemon, "effect_claims", proof.ATTEMPT_ID)
    proof._need(provider == original["provider_invocations"] and effects == [],
                "UNKNOWN history/effects changed before operator rearm")
    attempt = daemon.get_attempt(proof.ATTEMPT_ID)
    proof._need(attempt is not None, "settled native attempt missing")
    proof._same_identity(attempt.to_dict(), identity)
    proof._need(attempt.status == "failed" and attempt.committed_phase == "failed"
                and attempt.revision == 3 and attempt.finished_at_ms is not None,
                "operator rearm requires native FAILED execution")
    failure = daemon._portal_failure_phase_receipt(attempt)
    expected_failure = daemon._build_portal_failure_settlement_receipt(
        attempt, failure=DatabasePortalBridgeError(proof.GUARD_REASON))
    proof._need(failure == expected_failure and failure["provider_invocation_count"] == 1
                and failure["effect_claim_count"] == 0, "operator rearm failure receipt differs")
    task = daemon.task_source.get_task(proof.TASK_CID)
    proof._need(task is not None, "operator rearm target disappeared")
    operator_receipt = {"operation": "operator_control_plane_repair", "settlement_id": failure["settlement_id"]}
    expected_body = dict(snapshot["task"]["body"])
    if task.status == "blocked":
        expected_body["completion_receipt"] = failure
        proof._need(task.revision == 8 and dict(task.body) == expected_body,
                    "operator rearm requires the exact settled canonical block")
        cas = daemon.task_source.rearm_blocked_task(task, receipt=operator_receipt)
        proof._need(cas.changed is True and cas.previous_status == "blocked" and cas.revision == 9,
                    "operator rearm control CAS changed")
        control = cas.to_dict()
    else:
        expected_body["completion_receipt"] = operator_receipt
        proof._need(task.status == "retrying" and task.revision == 9 and dict(task.body) == expected_body,
                    "operator rearm response-loss preimage differs")
        control = task.to_dict()
    rearm = dict(daemon.coordinator.rearm_failed_task(
        failure_receipt=failure, control_task_observation=control))
    final_task = daemon.task_source.get_task(proof.TASK_CID)
    proof._need(final_task is not None and final_task.status == "retrying" and final_task.revision == 9
                and final_task.body.get("completion_receipt") == operator_receipt
                and rearm["ready"] is True and rearm["control_revision"] == 9
                and rearm["failure_settlement_id"] == failure["settlement_id"],
                "native operator rearm readback differs")
    after_provider = _rows(daemon, "provider_invocations", proof.ATTEMPT_ID)
    after_effect = _rows(daemon, "effect_claims", proof.ATTEMPT_ID)
    final_attempt = daemon.get_attempt(proof.ATTEMPT_ID)
    proof._need(after_provider == provider and after_effect == effects
                and final_attempt is not None and final_attempt.to_dict() == attempt.to_dict(),
                "operator rearm rewrote spent native execution history")
    return {
        "schema": "legal-ir-namespace-guard-recovery-qualified/v1",
        "actual_store_bindings": stores,
        "task_cid": proof.TASK_CID, "post_rearm_task": final_task.to_dict(),
        "audit_evidence_id": audit["evidence_id"], "audit_event_id": event["event_id"],
        "settlement_id": failure["settlement_id"], "rearm_id": rearm["rearm_id"],
        "adjudication_id": adjudication["adjudication_id"],
        "unknown_invocation_count": 1, "effect_claim_count": 0, "preserved_attempt_ordinal": 1,
        "stage_receipts": {"adjudication": adjudication, "audit_evidence": audit,
                           "audit_event": event, "native_failure": failure,
                           "operator_control": operator_receipt, "coordinator_rearm": rearm},
        "preserved_attempt": final_attempt.to_dict(), "provider_invocations": after_provider,
        "effect_claims": after_effect, "unknown_preserved": True,
        "provider_exit_inferred": False, "completion_authority": False,
        "successor_attempt_launched": False, "stage_pins": [],
    }
