import pytest

from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


@pytest.mark.parametrize("alias", ["DQP-T001", "PCTDD-006"])
def test_unknown_callback_requires_evidence_even_after_predecessor_changes(
    tmp_path, alias
):
    seed = _open_daemon(tmp_path, session="stable-session", max_task_attempts=2)
    try:
        population = _population(1)
        population["tasks"][0]["task_id"] = alias
        seed.materialize_population(population)
        task = seed.task_source.get("task:cid:001")
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=2,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        receipt.update(
            forced_block=True,
            authority_outcome="unknown",
            retry_exhausted=True,
            unknown_outcome_rearm_count=3,
            process_instance_id="predecessor-process",
        )
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=task.revision,
            new_status="blocked",
            receipt=receipt,
        )
    finally:
        seed.close()
    successor = _open_daemon(tmp_path, session="stable-session", max_task_attempts=2)
    try:
        before = successor.task_source.get("task:cid:001")
        results = successor.reconcile_blocked_unknown_outcome_tasks()
        after = successor.task_source.get("task:cid:001")
        assert (after.status, after.revision, after.body) == (
            before.status,
            before.revision,
            before.body,
        ), {
            "rearm_without_no_effect_evidence": results,
            "prior_budget": before.body.get("completion_receipt"),
            "new_budget": after.body.get("completion_receipt"),
        }
    finally:
        successor.close()


@pytest.mark.parametrize("fence", ["shared", "saga"])
def test_unresolved_native_recovery_barrier_prevents_other_task_rearm(
    tmp_path, monkeypatch, fence
):
    seed = _open_daemon(tmp_path, session="predecessor", max_task_attempts=2)
    try:
        seed.materialize_population(_population(1))
        task = seed.task_source.get("task:cid:001")
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_unknown_outcome_blocked",
            reason="provider_outcome_unknown",
        )
        receipt.update(
            forced_block=True,
            authority_outcome="unknown",
            process_instance_id="predecessor-process",
        )
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=task.revision,
            new_status="blocked",
            receipt=receipt,
        )
    finally:
        seed.close()
    successor = _open_daemon(tmp_path, session="successor", max_task_attempts=2)
    try:
        blocked = {
            "task_cid": "other-task",
            "recovery_required": True,
            "reason": "native_recovery_fence_still_open",
        }
        name = (
            "_reconcile_shared_no_provider_rearm_fences"
            if fence == "shared"
            else "_reconcile_database_no_provider_rearm_sagas"
        )
        monkeypatch.setattr(successor, name, lambda: [blocked])
        before = successor.task_source.get("task:cid:001")
        results = successor.reconcile_blocked_unknown_outcome_tasks()
        after = successor.task_source.get("task:cid:001")
        assert (after.status, after.revision, after.body) == (
            before.status,
            before.revision,
            before.body,
        ), {"recovery_results": results}
    finally:
        successor.close()


@pytest.mark.parametrize("alias", ["DQP-T001", "PCTDD-006"])
def test_failed_terminal_recovery_keeps_candidate_quarantined(
    tmp_path, monkeypatch, alias
):
    seed = _open_daemon(tmp_path, session="predecessor", max_task_attempts=2)
    try:
        population = _population(1)
        population["tasks"][0]["task_id"] = alias
        seed.materialize_population(population)
        task = seed.task_source.get("task:cid:001")
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_unknown_outcome_blocked",
            reason="provider_outcome_unknown",
        )
        receipt.update(
            forced_block=True,
            authority_outcome="unknown",
            process_instance_id="predecessor-process",
        )
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=task.revision,
            new_status="blocked",
            receipt=receipt,
        )
    finally:
        seed.close()
    successor = _open_daemon(tmp_path, session="successor", max_task_attempts=2)
    try:
        candidate = {
            "task_cid": "task:cid:001",
            "task_alias": alias,
            "recovered": False,
            "error": "receipt_stale",
        }
        monkeypatch.setattr(
            successor, "reconcile_blocked_terminal_landed_tasks", lambda: [candidate]
        )
        before = successor.task_source.get("task:cid:001")
        results = successor.reconcile_blocked_unknown_outcome_tasks()
        after = successor.task_source.get("task:cid:001")
        assert (after.status, after.revision, after.body) == (
            before.status,
            before.revision,
            before.body,
        ), results
    finally:
        successor.close()
