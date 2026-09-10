"""A control-row revision race defers the real tick without replaying work."""
import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    QuackTransportContentionError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as module,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


def test_real_reconciliation_conflict_defers_whole_tick_then_reads_current_row(
    tmp_path, monkeypatch,
):
    daemon = _open_daemon(tmp_path, session="session:revision-race")
    calls, claim_calls = [], []
    try:
        daemon.materialize_population(_population(1))
        original = daemon.task_source.get("task:cid:001")

        def reconcile():
            current = daemon.task_source.get(original.task_cid)
            calls.append(current.revision)
            if len(calls) == 1:
                # An actual committed prefix occurs inside this callback.
                # A later stale CAS cannot prove this entire tick unchanged.
                daemon._cas_task_status_database(
                    current.task_cid, expected_revision=current.revision,
                    new_status="blocked", receipt={"operation": "race_winner"},
                )
            daemon._cas_task_status_database(
                current.task_cid, expected_revision=current.revision,
                new_status="retrying", receipt={"operation": "fresh_recovery"},
            )
            return [{"changed": True, "task_cid": current.task_cid}]

        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", reconcile)
        monkeypatch.setattr(daemon, "claim_next", lambda: claim_calls.append(True))
        result = daemon.run_once()
        assert result["reason"] == "reconciliation_task_revision_conflict"
        assert result["deferred"] is True
        assert result["unchanged"] is None
        assert "write_count" not in result
        assert result["changed"] is False  # This diagnostic grants no mutation.
        assert result["completion_authority"] is False
        assert result["provider_dispatched"] == "unknown"
        assert result["recovery_provider_dispatched"] is False
        assert result["conflict_error_type"] == "TaskSourceConflictError"
        assert result["reconciliation_step"] == "reconcile"
        assert "post_merge_recovery" in result["recovery_prefix"]
        assert calls == [original.revision]
        assert claim_calls == []
        current = daemon.task_source.get(original.task_cid)
        assert current.revision == original.revision + 1
        assert current.status == "blocked"
        assert current.body["completion_receipt"] == {"operation": "race_winner"}

        next_result = daemon.run_once()
        current = daemon.task_source.get(original.task_cid)
        assert calls == [original.revision, original.revision + 1]
        assert current.revision == original.revision + 2
        assert current.status == "retrying"
        assert current.body["completion_receipt"] == {"operation": "fresh_recovery"}
        assert next_result.get("reason") != "reconciliation_task_revision_conflict"
        assert claim_calls == [True]
    finally:
        daemon.close()


@pytest.mark.parametrize("error_type", [
    RuntimeError, OSError, TimeoutError, module.DatabaseImplementationAuthorityError,
    module.DatabaseImplementationConflictError, QuackTransportContentionError,
])
def test_wrapper_preserves_non_task_conflict_failure_identity(error_type):
    daemon = object.__new__(module.DatabaseImplementationDaemon)
    error = error_type("task revision CAS is stale")

    def callback():
        raise error

    with pytest.raises(error_type) as caught:
        daemon._run_reconciliation_step(callback)
    assert caught.value is error


def test_wrapper_retains_specific_missing_coordination_authority_policy():
    daemon = object.__new__(module.DatabaseImplementationDaemon)

    def callback():
        raise module.DatabaseImplementationAuthorityError(
            "typed blocked recovery is unavailable without coordination-coupled owner authority"
        )

    assert daemon._run_reconciliation_step(callback) == []


def test_attach_contention_from_actual_reconciliation_still_aborts_whole_tick(
    tmp_path, monkeypatch,
):
    daemon = _open_daemon(tmp_path, session="session:reconciliation-attach")
    calls = []

    def callback():
        calls.append("reconciliation")
        raise QuackTransportContentionError(
            "quack control-plane attach contended: Authentication failed"
        )

    try:
        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", callback)
        monkeypatch.setattr(
            daemon, "reconcile_expired_running_attempts",
            lambda: calls.append("guard_expiry") or [],
        )
        monkeypatch.setattr(
            daemon, "reconcile_landed_merged_tasks",
            lambda: pytest.fail("attach contention continued the normal tick"),
        )
        monkeypatch.setattr(
            daemon, "claim_next",
            lambda: pytest.fail("attach contention admitted a fresh claim"),
        )
        result = daemon.run_once()
        assert result["reason"] == "quack_attach_contended"
        assert result["deferred"] is True
        # The existing outer attach guard retains its separate expiry pass.
        assert calls == ["reconciliation", "guard_expiry"]
    finally:
        daemon.close()


@pytest.mark.parametrize("conflict_type", [TaskSourceConflictError, module.TaskSourceConflictError])
def test_wrapper_signal_retains_exact_original_conflict_as_cause(conflict_type):
    daemon = object.__new__(module.DatabaseImplementationDaemon)
    original = conflict_type("task revision CAS is stale")

    def callback():
        raise original

    with pytest.raises(module._DatabaseReconciliationConflictDeferral) as caught:
        daemon._run_reconciliation_step(callback)
    assert caught.value.__cause__ is original
    assert caught.value.step == "callback"
