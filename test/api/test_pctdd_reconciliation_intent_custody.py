"""A retained supersession saga cannot borrow an old finalizer's authority."""

import json

import pytest

from test.api.test_agent_supervisor_database_portal_bridge import (
    _seed_interrupted_database_portal_attempt,
)


@pytest.mark.parametrize("old_unknown", [False, True])
def test_superseded_saga_retains_custody_when_old_finalized_receipt_reappears(
    tmp_path,
    monkeypatch,
    old_unknown,
):
    _, daemon, bridge, attempt, _ = _seed_interrupted_database_portal_attempt(tmp_path)
    try:
        original = daemon.task_source.get(attempt.task_cid)
        _, binding = bridge._ensure_attempt_projection(attempt, original)
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
            outcome="deferred",
        )
        daemon._record_database_portal_attempt_binding(
            attempt, binding, "portal_entered"
        )

        release = daemon.coordinator.release

        def crash_after_cas(*args, **kwargs):
            raise RuntimeError("after canonical CAS")

        monkeypatch.setattr(daemon.coordinator, "release", crash_after_cas)
        with pytest.raises(RuntimeError, match="after canonical CAS"):
            daemon._finalize_failed_attempt(
                attempt,
                reason="old ordinary failure",
                force_block=old_unknown,
                unknown_authority=old_unknown,
            )
        finalized = daemon.task_source.get(attempt.task_cid)
        monkeypatch.setattr(daemon.coordinator, "release", release)

        daemon.materialize_population(
            {
                "repository_tree_id": "tree:replacement-before-saga",
                "tasks": [
                    {
                        "task_cid": attempt.task_cid,
                        "task_id": "PCTDD-001",
                        "goal_cid": "goal:pctdd",
                        "title": "Later replacement",
                        "status": "ready",
                        "validation_commands": ["pytest replacement.py"],
                    }
                ],
            }
        )
        finalize = daemon._finalize_failed_attempt

        def crash_after_barrier(*args, **kwargs):
            saga = daemon._database_portal_terminal_reconciliation_saga(attempt)
            assert saga["intended_database_disposition"] == "superseded_attempt_revoked"
            raise RuntimeError("after supersession barrier")

        monkeypatch.setattr(daemon, "_finalize_failed_attempt", crash_after_barrier)
        with pytest.raises(RuntimeError, match="after supersession barrier"):
            daemon.reconcile_quiesced_database_portal_attempts(
                trigger="test", force=True
            )
        monkeypatch.setattr(daemon, "_finalize_failed_attempt", finalize)

        # A real later CAS carries an old complete receipt. It cannot turn the
        # already admitted supersession into an ordinary retry finalizer.
        replacement = daemon.task_source.get(attempt.task_cid)
        daemon._cas_task_status_database(
            attempt.task_cid,
            expected_revision=replacement.revision,
            new_status=finalized.status,
            receipt=dict(finalized.body["completion_receipt"]),
        )
        task_before = json.dumps(
            daemon.task_source.get(attempt.task_cid).to_dict(), sort_keys=True
        )
        before = daemon.get_attempt(attempt.attempt_id).to_dict()
        claim_before = daemon.coordinator.get_task_claim(attempt.claim_id)
        results = daemon._reconcile_database_portal_post_cas_transitions(
            bridge=bridge,
            trigger="test-replay",
        )
        assert results and results[0]["blocked"] is True
        assert (
            results[0]["error"]
            == "terminal finalizer contradicts its prepared disposition"
        )
        assert daemon.get_attempt(attempt.attempt_id).to_dict() == before
        assert daemon.coordinator.get_task_claim(attempt.claim_id) == claim_before
        assert (
            json.dumps(
                daemon.task_source.get(attempt.task_cid).to_dict(), sort_keys=True
            )
            == task_before
        )
        assert (
            daemon._database_portal_terminal_reconciliation_saga(attempt)["stage"]
            == "commit_barrier"
        )
    finally:
        daemon.close()


def test_entered_unknown_callback_without_state_retains_exact_custody(tmp_path):
    _, daemon, _, attempt, paths = _seed_interrupted_database_portal_attempt(
        tmp_path,
        seed_nested_state=False,
    )
    try:
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_database_portal_attempt_binding(
            attempt,
            json.loads(paths.binding.read_text()),
            "portal_entered",
        )
        connection = daemon._require_connection()
        tables = (
            "database_task_attempts",
            "attempt_phases",
            "attempt_dispatch_journal",
            "provider_invocations",
            "effect_claims",
            "database_portal_attempt_bindings",
        )

        def snapshot():
            return {
                table: connection.execute(
                    f"SELECT * FROM {table} WHERE attempt_id = ? ORDER BY 1",
                    [attempt.attempt_id],
                ).fetchall()
                for table in tables
            }

        before = snapshot()
        task_before = daemon.task_source.get(attempt.task_cid).to_dict()
        claim_before = daemon.coordinator.get_task_claim(attempt.claim_id)
        for _ in range(2):
            result = daemon.reconcile_quiesced_database_portal_attempts(
                trigger="test", force=True
            )
            assert result["blocked"] is True
            item = result["attempts"][0]
            assert item["terminal_provider_evidence"] is False
            assert item["nested_state"]["present"] is False
            assert (
                item["portal_reconciliation"]["provider_forbidden_terminal_recovery"][
                    "state_reason"
                ]
                == "missing_state_file"
            )
            assert snapshot() == before
            assert daemon.task_source.get(attempt.task_cid).to_dict() == task_before
            assert daemon.coordinator.get_task_claim(attempt.claim_id) == claim_before
            assert not paths.state.exists()
    finally:
        daemon.close()
