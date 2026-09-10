"""Real control-row CAS races must defer a native tick without stale effects."""
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceUnknownOutcomeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


def _blocked_successor(tmp_path: Path, task_count: int = 1):
    seed = _open_daemon(tmp_path, session="session:predecessor", max_task_attempts=2)
    try:
        seed.materialize_population(_population(task_count))
        for index in range(1, task_count + 1):
            task = seed.task_source.get(f"task:cid:{index:03d}")
            receipt = seed._retry_budget_receipt(
                task, attempts_used=2, operation="database_retry_exhausted",
                reason="portal_provider_failed",
            )
            receipt["retry_exhausted"] = True
            seed._cas_task_status_database(
                task.task_cid, expected_revision=task.revision,
                new_status="blocked", receipt=receipt,
            )
    finally:
        seed.close()
    calls = []
    daemon = _open_daemon(
        tmp_path, session="session:successor", max_task_attempts=2,
        provider_calls=calls,
    )
    return daemon, calls


@pytest.mark.parametrize("tasks", [1, 2])
@pytest.mark.parametrize("new_authority", ["retryable", "unknown_callback"])
def test_real_rearm_cas_race_defers_tick_and_rereads_revision(
    tmp_path, monkeypatch, tasks, new_authority,
):
    daemon, providers = _blocked_successor(tmp_path, tasks)
    conflict_cid = f"task:cid:{tasks:03d}"
    original_cas = daemon._cas_task_status_database
    before = daemon.task_source.get(conflict_cid)
    sibling_receipt = dict(before.body["completion_receipt"])
    sibling_receipt["unknown_outcome_rearm_count"] = 1
    if new_authority == "unknown_callback":
        sibling_receipt.update(
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
            forced_block=True, authority_outcome="unknown",
        )
    fired = []

    def race(task_cid, **kwargs):
        if task_cid == conflict_cid and not fired:
            fired.append(kwargs["expected_revision"])
            # A real competing mutation wins after the scan, before its CAS.
            original_cas(
                task_cid, expected_revision=before.revision,
                new_status="retrying", receipt=sibling_receipt,
            )
            original_cas(
                task_cid, expected_revision=before.revision + 1,
                new_status="blocked", receipt=sibling_receipt,
            )
        return original_cas(task_cid, **kwargs)

    try:
        monkeypatch.setattr(daemon, "_cas_task_status_database", race)
        result = daemon.run_once()
        assert result["deferred"] is True
        assert result["selection_idle_reason"] == "database_unknown_outcome_rearm_conflict"
        assert result["write_count"] == tasks - 1
        assert result["unchanged"] is (tasks == 1)
        assert result["implementation_result"] is None
        assert providers == []
        assert daemon.list_running_attempts() == []
        current = daemon.task_source.get(conflict_cid)
        assert current.revision == before.revision + 2
        assert current.status == "blocked"
        assert current.body["completion_receipt"] == sibling_receipt
        conflict = result["unknown_outcome_rearms"][-1]
        assert conflict["rearmed"] is False
        assert conflict["expected_revision"] == before.revision

        monkeypatch.setattr(daemon, "_cas_task_status_database", original_cas)
        # Exercise the public production tick again, without a daemon restart.
        next_result = daemon.run_once()
        current = daemon.task_source.get(conflict_cid)
        if new_authority == "retryable":
            assert next_result["selection_idle_reason"] == "database_unknown_outcomes_rearmed"
            assert current.revision == before.revision + 3
            assert current.status == "retrying"
            assert current.body["completion_receipt"]["unknown_outcome_rearm_count"] == 2
            assert providers == []
        else:
            assert current.revision == before.revision + 2
            assert current.status == "blocked"
            assert current.body["completion_receipt"] == sibling_receipt
            assert conflict_cid not in providers
        assert fired == [before.revision]
    finally:
        daemon.close()


@pytest.mark.parametrize("error_type", [
    RuntimeError, OSError, TimeoutError,
    DatabaseImplementationAuthorityError, TaskSourceUnknownOutcomeError,
])
def test_non_conflict_failure_is_not_relabelled_as_rearm_race(
    tmp_path, monkeypatch, error_type,
):
    daemon, providers = _blocked_successor(tmp_path)
    before = daemon.task_source.get("task:cid:001")

    def failure(*args, **kwargs):
        raise error_type("task revision CAS is stale")

    try:
        monkeypatch.setattr(daemon, "_cas_task_status_database", failure)
        with pytest.raises(error_type, match="task revision CAS is stale"):
            daemon.run_once()
        after = daemon.task_source.get(before.task_cid)
        assert after.revision == before.revision
        assert after.body == before.body
        assert providers == []
    finally:
        daemon.close()
