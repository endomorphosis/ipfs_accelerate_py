"""Later revisions and shutdown diagnostics cannot manufacture recovery authority."""

from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_RETRY_BUDGET_SCHEMA,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.runtime import multi_supervisor_runner


PINS = (
    *DATABASE_FENCED_PROVIDER_RETAINED_MANIFEST_PINS,
    DATABASE_PCTDD005_SUCCESSOR_MANIFEST_PIN,
)


@pytest.mark.parametrize("pin", PINS, ids=lambda pin: str(pin["task_alias"]))
@pytest.mark.parametrize("revision_delta", (1, 14))
@pytest.mark.parametrize("terminal_link", (False, True))
@pytest.mark.parametrize(
    "operation,reason",
    (
        ("database_unknown_outcome_blocked", "callback_authority_incomplete_blocked"),
        ("database_unknown_outcome_blocked", "provider_dispatch_outcome_unknown"),
        ("database_retry_exhausted", "portal_provider_failed"),
    ),
)
def test_later_revision_without_recovery_authority_never_mutates_task(
    tmp_path,
    monkeypatch,
    pin,
    revision_delta,
    terminal_link,
    operation,
    reason,
):
    """A newer revision cannot bypass a current native claim denial."""

    daemon = _open_daemon(tmp_path, session="session:later-revision-authority")
    receipt = {
        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
        "operation": operation,
        "reason": reason,
        "forced_block": True,
        "authority_outcome": "unknown",
        "retry_exhausted": True,
        "attempts_used": 2,
        "attempt_number": 99,
        "unknown_outcome_rearm_count": 3,
        "owner_session_id": "session:prior-owner",
        "process_instance_id": "process:prior-owner",
    }
    if terminal_link:
        receipt["terminal_reconciliation"] = {"schema": "unverified-later-attempt"}
    task = SimpleNamespace(
        task_cid=str(pin["task_cid"]),
        task_alias=str(pin["task_alias"]),
        status="blocked",
        revision=int(pin["blocked_task_revision"]) + revision_delta,
        body={"completion_receipt": receipt},
    )
    mutations = []
    try:
        daemon._database_portal_bridge = object()
        monkeypatch.setattr(
            daemon.task_source,
            "list_tasks",
            lambda **kwargs: SimpleNamespace(
                tasks=(task,) if kwargs.get("status") == "blocked" else ()
            ),
        )
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(
            daemon, "_automatic_claim_forbidden_current", lambda task: True
        )
        monkeypatch.setattr(
            daemon, "_database_portal_no_provider_rearm_evidence", lambda *args: None
        )

        def track_mutation(task_cid, **kwargs):
            mutations.append((task_cid, kwargs))
            return SimpleNamespace(
                task_cid=task_cid,
                status=kwargs["new_status"],
                revision=task.revision + 1,
                body={"completion_receipt": kwargs["receipt"]},
            )

        monkeypatch.setattr(daemon, "_cas_task_status_database", track_mutation)
        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert mutations == []
        assert not any(item.get("rearmed") for item in outcomes)
        assert task.body["completion_receipt"] == receipt
    finally:
        daemon.close()


def test_live_provider_tree_cannot_be_reported_as_fenced(tmp_path, monkeypatch):
    """A retained live tree must block owner replacement and source cutover."""

    runner = multi_supervisor_runner
    track = runner.SupervisorTrack(
        name="later-revision-lane",
        script_path=tmp_path / "script.py",
        log_path=tmp_path / "run.log",
        supervisor_pid_path=tmp_path / "supervisor.pid",
        daemon_pid_path=tmp_path / "daemon.pid",
    )
    process = SimpleNamespace(pid=271828, wait=lambda **kwargs: None)
    monkeypatch.setattr(
        runner, "_restarting_track_must_preserve_extra_gate_grok", lambda *args: True
    )
    monkeypatch.setattr(
        runner,
        "_terminate_managed_process",
        lambda *args, **kwargs: (False, (process.pid,)),
    )
    payload = runner.stop_tracks(
        [track],
        {track.name: process},
        repo_root=tmp_path,
        grace_seconds=0.1,
        output=lambda message: None,
    )
    assert payload["all_trees_fenced"] is False
    assert payload["stopped_count"] == 0
    assert payload["removed_runtime_markers"] == []


@pytest.mark.parametrize("pin", PINS, ids=lambda pin: str(pin["task_alias"]))
def test_terminal_disposition_conflict_is_reported_for_every_task(monkeypatch, pin):
    """The native repair page must retain a failed terminal validator result."""

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.owner_session_id = "session:terminal-conflict"
    attempt = SimpleNamespace(
        attempt_id="attempt:terminal-conflict",
        claim_id="claim:terminal-conflict",
        task_cid=str(pin["task_cid"]),
        task_alias=str(pin["task_alias"]),
        to_dict=lambda: {"attempt_id": "attempt:terminal-conflict"},
    )
    row = (*([None] * 15), "{}")
    connection = SimpleNamespace(
        execute=lambda *args: SimpleNamespace(fetchall=lambda: [row])
    )
    monkeypatch.setattr(daemon, "_require_connection", lambda: connection)
    monkeypatch.setattr(daemon, "_attempt_from_row", lambda row: attempt)

    def reject_changed_disposition(attempt):
        raise DatabaseImplementationConflictError(
            "terminal phase changed its actual database disposition"
        )

    monkeypatch.setattr(
        daemon,
        "_database_portal_terminal_reconciliation_saga",
        reject_changed_disposition,
    )
    outcomes = daemon._repair_database_portal_terminal_receipts(
        bridge=object(), trigger="test-terminal-authority", exact_attempt=attempt
    )
    assert len(outcomes) == 1
    assert outcomes[0]["blocked"] is True
    assert outcomes[0]["reason"] == "terminal_reconciliation_receipt_repair_failed"
    assert (
        outcomes[0]["error"] == "terminal phase changed its actual database disposition"
    )
