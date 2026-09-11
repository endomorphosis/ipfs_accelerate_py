"""Reconcile an ordinary finalizer interrupted after its canonical task CAS."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_database_portal_bridge import (
    _seed_interrupted_database_portal_attempt,
)


@pytest.mark.parametrize("force_block,cap", [(False, 3), (False, 1), (True, 3)])
def test_reconcile_after_ordinary_finalizer_cas(tmp_path, monkeypatch, force_block, cap):
    _, daemon, bridge, attempt, paths = _seed_interrupted_database_portal_attempt(
        tmp_path, max_task_attempts=cap,
    )
    daemon._begin_callback_dispatch(
        attempt, dispatch_kind="provider", idempotency_key=f"provider:{attempt.attempt_id}",
    )
    daemon._record_database_portal_attempt_binding(
        attempt, json.loads(paths.binding.read_text()), "portal_entered",
    )
    original = daemon.coordinator.release

    def crash(*args, **kwargs):
        raise RuntimeError("crash after canonical CAS")

    def forbidden(*args, **kwargs):
        raise AssertionError("reconciliation dispatched a callback")

    try:
        monkeypatch.setattr(daemon.coordinator, "release", crash)
        with pytest.raises(RuntimeError, match="canonical CAS"):
            daemon._finalize_failed_attempt(
                attempt, reason="ordinary_failure", force_block=force_block,
                unknown_authority=force_block,
            )
        task = daemon.task_source.get(attempt.task_cid)
        assert task.status == ("blocked" if force_block or cap == 1 else "retrying")
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
        before = json.dumps(task.to_dict(), sort_keys=True)
        monkeypatch.setattr(daemon.coordinator, "release", original)
        daemon._provider_fn = daemon._effect_fn = daemon._validation_fn = forbidden
        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="database_daemon_startup", force=True,
        )
        assert result["reconciled"], json.dumps(result, indent=2)
        expected = "blocked_unknown_outcome" if force_block else "terminalized_for_retry"
        assert result["attempts"][0]["database_disposition"] == expected
        assert json.dumps(daemon.task_source.get(attempt.task_cid).to_dict(), sort_keys=True) == before
        for _ in range(2):
            repair = daemon._repair_database_portal_terminal_receipts(
                bridge=bridge, trigger="test-repeat",
                exact_attempt=daemon.get_attempt(attempt.attempt_id),
            )
            assert repair and all(x["reconciled"] for x in repair), repair
    finally:
        daemon.close()


@pytest.mark.parametrize("changed", [
    "revision", "validation", "execution", "claim_id", "attempt_id", "lease_id",
    "owner_session_id", "attempt_number", "fencing_token", "fence_epoch",
    "task_cid", "terminal_reconciliation", "attempt_consumed", "operation",
])
def test_finalized_observation_rejects_replacement_or_unbound_receipt(tmp_path, changed):
    _, daemon, _, attempt, _ = _seed_interrupted_database_portal_attempt(tmp_path)
    try:
        daemon._finalize_failed_attempt(attempt, reason="ordinary_failure")
        task = daemon.task_source.get(attempt.task_cid)
        assert daemon._database_attempt_finalized_disposition(attempt, task) == "terminalized_for_retry"
        candidate = SimpleNamespace(**{
            "task_cid": task.task_cid, "revision": task.revision,
            "status": task.status, "body": deepcopy(task.to_dict()["body"]),
        })
        if changed == "revision":
            candidate.revision += 1
        elif changed in {"validation", "execution"}:
            # The attempt's immutable input binding must still equal the task.
            from dataclasses import replace
            body = deepcopy(attempt.to_dict()["body"])
            body["control_claim"][changed + "_spec_cid"] = "changed"
            attempt = replace(attempt, body=body)
        elif changed == "terminal_reconciliation":
            candidate.body["completion_receipt"][changed] = {"unverified": True}
        elif changed == "attempt_consumed":
            candidate.body["completion_receipt"][changed] = False
        else:
            candidate.body["completion_receipt"][changed] = "changed"
        assert daemon._database_attempt_finalized_disposition(attempt, candidate) == ""
    finally:
        daemon.close()
