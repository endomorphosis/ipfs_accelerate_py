"""Native local acknowledgement for retained, unresolved task custody.

A central event alone never grants independent work. The current exclusive
execution owner must prove the same retained bytes and install a coordinator
fence first. This protocol does not terminalize or retry the retained attempt.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..merge import owner_task_quarantine as custody
from ..merge import workspace_quarantine as workspace
from ..task_sources import owner_task_quarantine as central
from ..task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ..merge.worktree_lifecycle import current_process_birth

ACK_KEY = "owner_task_quarantine_ack@1"
EXECUTION_TABLES = (
    "database_task_attempts",
    "attempt_phases",
    "provider_invocations",
    "effect_claims",
    "attempt_dispatch_journal",
    "attempt_recovery_dispatch_fences",
    "database_portal_attempt_bindings",
    "database_portal_terminal_reconciliations",
    "daemon_execution_events",
)


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def require_owner(daemon: Any, *, require_idle: bool = True) -> None:
    central.require(
        not daemon._closed
        and daemon._connection is not None
        and daemon._retained_embedded_writer_fence_current()
        and current_process_birth() == daemon.process_birth
        and (not require_idle or daemon._active_external_callbacks == 0),
        "quarantine_requires_idle_execution_owner",
    )
    process = daemon._database_process_instance_record(daemon.process_instance_id)
    central.require(
        process is not None
        and process.get("state") == "active"
        and process.get("owner_session_id") == daemon.owner_session_id
        and process.get("process_birth") == daemon.process_birth.to_dict(),
        "quarantine_execution_owner_changed",
    )


def execution_snapshot(connection: Any, task_cid: str) -> dict[str, Any]:
    groups = {}
    remaining_rows, remaining_bytes = 8192, 4_194_304
    for table in EXECUTION_TABLES:
        where = (
            "attempt_id IN (SELECT attempt_id FROM database_task_attempts WHERE task_cid = ?)"
            if table == "attempt_phases"
            else "task_cid = ?"
        )
        population = connection.execute(
            f"SELECT count(*), COALESCE(sum(octet_length(encode(to_json(t)))),0) FROM (SELECT * FROM {table} WHERE {where}) AS t",
            [task_cid],
        ).fetchone()
        count, size = population[0], population[1]
        central.require(
            type(count) is int
            and type(size) is int
            and 0 <= count <= remaining_rows
            and 0 <= size <= remaining_bytes,
            "quarantine_execution_population_bound",
        )
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE {where} ORDER BY ALL LIMIT ?",
            [task_cid, remaining_rows + 1],
        ).fetchall()
        central.require(len(rows) == count, "quarantine_execution_population_changed")
        values = [[row[index] for index in range(len(row))] for row in rows]
        groups[table] = {"count": count, "cid": content_identity(values)}
        remaining_rows -= count
        remaining_bytes -= size
    return groups


def capture(daemon: Any, attempt: Any, *, require_idle: bool = True) -> dict[str, Any]:
    """Diagnose only the two proved native failure classes; all else stays blocked."""
    require_owner(daemon, require_idle=require_idle)
    central.require(
        attempt.owner_session_id == daemon.owner_session_id
        and daemon.get_attempt(attempt.attempt_id).to_dict() == attempt.to_dict(),
        "quarantine_attempt_changed",
    )
    bridge = daemon._database_portal_bridge
    central.require(bridge is not None, "quarantine_native_bridge_missing")
    binding = daemon._database_portal_attempt_binding(attempt)
    central.require(binding is not None, "quarantine_binding_missing")
    paths = bridge._paths(attempt)
    # Native projection/binding validation; no state file or worktree creation.
    task = daemon.task_source.get(attempt.task_cid)
    central.require(task is not None, "quarantine_task_missing")
    saga = daemon._database_portal_terminal_reconciliation_saga(attempt)
    if attempt.status in {"failed", "superseded"}:
        central.require(
            saga is not None
            and saga["stage"] == "commit_barrier"
            and saga["intended_database_disposition"] == "superseded_attempt_revoked",
            "quarantine_terminal_diagnosis_unproved",
        )
        evidence = daemon._terminal_reconciliation_evidence_from_saga(
            attempt=attempt, saga=saga, bridge=bridge
        )
        failed = [
            row
            for row in daemon.phase_history(attempt.attempt_id)
            if row.get("phase") == "failed"
        ]
        central.require(
            len(failed) == 1
            and failed[0]["body"].get("terminal_reconciliation") == evidence
            and failed[0]["body"].get("database_disposition")
            == "terminalized_for_retry",
            "quarantine_terminal_conflict_unproved",
        )
        diagnosis = "terminal_disposition_conflict"
        basis = {"evidence": evidence, "binding": dict(binding)}
    else:
        boundary = daemon._database_callback_boundary_state(attempt)
        central.require(
            attempt.status == "running"
            and saga is None
            and binding.get("stage") == "portal_entered"
            and not boundary["callback_receipt_errors"]
            and boundary["durable_provider_result"] is None
            and boundary["durable_effect_result"] is None
            and boundary["provider_dispatch"] is not None
            and boundary["provider_dispatch"].get("outcome") == "started"
            and boundary["provider_dispatch"].get("body")
            == {
                "schema": "ipfs_accelerate_py/agent-supervisor/database-callback-dispatch@1",
                "outcome": "unknown_until_callback_returns",
            }
            and boundary["effect_dispatch"] is None,
            "quarantine_entered_diagnosis_unproved",
        )
        try:
            paths.state.lstat()
        except FileNotFoundError:
            pass
        else:
            raise central.QuarantineDenied("quarantine_nested_state_present")
        diagnosis = "entered_callback_state_missing"
        basis = {
            "binding": dict(binding),
            "boundary": dict(boundary),
            "state_path": str(paths.state),
        }
    connection = daemon._require_connection()
    central.require(
        not connection.in_transaction, "quarantine_execution_transaction_active"
    )
    connection.execute("BEGIN TRANSACTION")
    try:
        execution = execution_snapshot(connection, attempt.task_cid)
        require_owner(daemon, require_idle=require_idle)
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    with daemon.coordinator._lock:
        coordination = custody.snapshot(daemon.coordinator._require(), attempt.task_cid)
    central.require(
        bridge.workspace_repository_root is not None
        and bridge.workspace_root is not None,
        "quarantine_workspace_configuration_unbound",
    )
    workspace_plan = workspace.plan(
        bridge.workspace_repository_root, bridge.workspace_root
    )
    retained = {
        "workspace_custody_cid": workspace_plan["cid"],
        "workspace_root": workspace_plan["root"],
        "fresh_workspace_root": workspace_plan["fresh_root"],
        "task_cid": attempt.task_cid,
        "task_revision": task.revision,
        "attempt_id": attempt.attempt_id,
        "execution_store_id": daemon.execution_store_identity,
        "execution_owner_id": daemon.owner_session_id,
        "diagnosis": diagnosis,
        "retained_state_cid": content_identity(
            {
                "execution": execution,
                "coordination": coordination,
                "basis": plain(basis),
                "task": plain(task.to_dict()),
                "workspace": workspace_plan,
            }
        ),
    }
    return {
        "retained": retained,
        "execution": execution,
        "coordination": coordination,
        "workspace": workspace_plan,
    }


def acknowledge(
    daemon: Any, *, attempt_id: str, revoke: bool = False
) -> dict[str, Any]:
    """The sole native entry point: fresh capture, central CAS, local custody, ack."""
    with daemon._lock:
        attempt = daemon.get_attempt(attempt_id)
        central.require(attempt is not None, "quarantine_attempt_missing")
        observed = capture(daemon, attempt)
        bridge = daemon._database_portal_bridge
        frozen = workspace.freeze(
            bridge.workspace_repository_root,
            bridge.workspace_root,
            expected=observed["workspace"]["snapshot"],
        )
        central.require(
            frozen == observed["workspace"], "quarantine_workspace_freeze_changed"
        )
        intent = daemon.task_source.intent
        prior = intent.owner_task_quarantines().get(attempt_id)
        if prior is not None:
            central.require(
                all(prior[key] == value for key, value in observed["retained"].items()),
                "quarantine_retained_state_changed",
            )
        with intent._connection() as control_connection:
            binding = getattr(control_connection, "_quack_mutation_binding", None)
        if prior is None or revoke or prior["owner_binding"] != binding:
            head = intent.quarantine_retained_task(
                retained=observed["retained"],
                expected_event_id=prior["event_id"] if prior else "",
                revoke=revoke,
            )
        else:
            head = prior
        central.require(
            capture(daemon, attempt) == observed,
            "quarantine_custody_changed_after_central_ack",
        )
        coordinator = daemon.coordinator
        with coordinator._lock:
            connection = coordinator._require()
            coordinator._begin(connection)
            try:
                custody.install(
                    connection,
                    task_cid=attempt.task_cid,
                    event_id=head["event_id"],
                    retained_state_cid=head["retained_state_cid"],
                    expected_snapshot=observed["coordination"],
                )
                coordinator._commit_if_idle(connection)
            except BaseException:
                coordinator._rollback_if_open(connection)
                raise
        connection = daemon._require_connection()
        ack = {
            "schema": ACK_KEY,
            "head": dict(head),
            "execution": observed["execution"],
            "coordination": observed["coordination"],
            "process_instance_id": daemon.process_instance_id,
            "process_birth": daemon.process_birth.to_dict(),
        }
        key = ACK_KEY + ":" + attempt_id
        connection.execute(
            "INSERT INTO daemon_execution_metadata (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            [key, canonical_json_bytes(ack).decode()],
        )
        daemon._database_portal_reconciliation_checked = False
        daemon._owner_quarantine_independent_admission = None
        return ack


def current(daemon: Any) -> dict[str, Any]:
    """Return only exact, current, locally acknowledged active fences.

    A central fence belonging to this execution store without its local ack is
    a global barrier. Foreign lanes still observe it through central mutation
    guards but have no authority to acknowledge the retained execution store.
    """
    result = {}
    intent = getattr(daemon.task_source, "intent", None)
    if intent is None:
        return result
    with intent._connection() as connection:
        heads = central.heads(connection)
        if not heads:
            return result
        binding = getattr(connection, "_quack_mutation_binding", None)
    for attempt_id, head in heads.items():
        central.require(
            head["state"] == "active" and head["owner_binding"] == binding,
            "quarantine_owner_generation_or_state_changed",
        )
        if head["execution_store_id"] != daemon.execution_store_identity:
            continue
        central.require(
            head["execution_owner_id"] == daemon.owner_session_id,
            "quarantine_local_owner_changed",
        )
        require_owner(daemon, require_idle=False)
        row = (
            daemon._require_connection()
            .execute(
                "SELECT value FROM daemon_execution_metadata WHERE key = ?",
                [ACK_KEY + ":" + attempt_id],
            )
            .fetchone()
        )
        central.require(
            row is not None and type(row[0]) is str and len(row[0].encode()) <= 262144,
            "quarantine_local_ack_missing",
        )
        ack = central.strict_json(row[0])
        central.require(
            type(ack) is dict
            and set(ack)
            == {
                "schema",
                "head",
                "execution",
                "coordination",
                "process_instance_id",
                "process_birth",
            }
            and ack["schema"] == ACK_KEY
            and ack["head"] == head
            and ack["process_instance_id"] == daemon.process_instance_id
            and ack["process_birth"] == daemon.process_birth.to_dict(),
            "quarantine_local_ack_changed",
        )
        bridge = daemon._database_portal_bridge
        frozen = workspace.verify(
            bridge.workspace_repository_root, bridge.workspace_root
        )
        central.require(
            frozen["cid"] == head["workspace_custody_cid"],
            "quarantine_workspace_ack_changed",
        )
        with daemon._lock:
            observed = capture(
                daemon, daemon.get_attempt(attempt_id), require_idle=False
            )
        central.require(
            all(head[key] == value for key, value in observed["retained"].items())
            and observed["execution"] == ack["execution"]
            and observed["coordination"] == ack["coordination"],
            "quarantine_custody_changed",
        )
        with daemon.coordinator._lock:
            record = custody.verify(daemon.coordinator._require()).get(head["task_cid"])
        central.require(
            record is not None
            and record["event_id"] == head["event_id"]
            and record["retained_state_cid"] == head["retained_state_cid"],
            "quarantine_coordination_ack_missing",
        )
        result[attempt_id] = head
    return result


def refresh(daemon: Any) -> dict[str, Any]:
    """Re-acknowledge a recorded fence after restart from the same native rows."""
    intent = getattr(daemon.task_source, "intent", None)
    if intent is None:
        return {}
    heads = intent.owner_task_quarantines()
    for attempt_id, head in heads.items():
        if head["execution_store_id"] == daemon.execution_store_identity:
            central.require(head["state"] == "active", "quarantine_revoked")
            acknowledge(daemon, attempt_id=attempt_id)
    return current(daemon)


def admit_known_blockers(daemon: Any, result: dict[str, Any]) -> list[str]:
    """Nomination is diagnostic; capture itself must prove the native failure."""
    if daemon.authority_mode != "quack" or result.get("blocked") is not True:
        return []
    candidates = result.get("attempts")
    central.require(
        type(candidates) is list and len(candidates) <= 100,
        "quarantine_reconciliation_population_incomplete",
    )
    nominated = sorted(
        {
            item["attempt_id"]
            for item in candidates
            if type(item) is dict
            and item.get("blocked") is True
            and type(item.get("attempt_id")) is str
        }
    )
    admitted = []
    for attempt_id in nominated:
        attempt = daemon.get_attempt(attempt_id)
        if attempt is None:
            continue
        try:
            capture(daemon, attempt)
        except central.QuarantineDenied:
            continue
        acknowledge(daemon, attempt_id=attempt_id)
        admitted.append(attempt_id)
    return admitted


def independent_workspace_root(daemon: Any, repo_root: Any, configured: Any) -> Any:
    from pathlib import Path

    root = Path(configured)
    if not root.is_absolute():
        root = Path(repo_root) / root
    intent = getattr(daemon.task_source, "intent", None)
    if intent is None:
        return configured
    heads = intent.owner_task_quarantines()
    if not heads:
        return configured
    current(daemon)
    frozen = workspace.verify(Path(repo_root), root.resolve())
    central.require(
        all(
            head["workspace_custody_cid"] == frozen["cid"]
            and head["workspace_root"] == frozen["root"]
            and head["fresh_workspace_root"] == frozen["fresh_root"]
            for head in heads.values()
        ),
        "quarantine_independent_workspace_scope_changed",
    )
    return Path(frozen["fresh_root"])
