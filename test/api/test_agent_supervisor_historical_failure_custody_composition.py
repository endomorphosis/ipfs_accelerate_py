"""Historical terminal replay must stop before retrying unsettled successors."""

from test.api.test_agent_supervisor_historical_failure_replay import (
    execution_snapshot, setup_daemon_history, snapshot,
)


def test_historical_marker_precedes_custody_deferral_without_retry(tmp_path, monkeypatch):
    daemon, old, newer, receipt, calls = setup_daemon_history(tmp_path, monkeypatch)
    try:
        daemon.require_real_execution = True
        claim = daemon.coordinator.get_task_claim(newer.claim_id)
        monkeypatch.setattr(daemon, "_clock_ms", lambda: claim.expires_at_ms + 1)
        before = snapshot(daemon.coordinator), execution_snapshot(daemon)
        canonical = daemon.task_source.get(old.task_cid)
        first = daemon.run_once()
        second = daemon.run_once()
        for outcome in (first, second):
            assert outcome["reason"] == "production_attempt_settlement_required"
            assert outcome["retry_authorized"] is False
            assert outcome["provider_dispatched"] == "unknown"
            assert outcome["recovery_provider_dispatched"] is False
            assert outcome["retained_attempt_evidence"]["attempt_id"] == newer.attempt_id
        reconciled = first["recovery_prefix"]["portal_failure_reconciliations"]
        assert len(reconciled) == 1 and reconciled[0]["attempt_id"] == old.attempt_id
        assert second["recovery_prefix"]["portal_failure_reconciliations"] == []
        assert (snapshot(daemon.coordinator), execution_snapshot(daemon)) == before
        assert daemon.task_source.get(old.task_cid) == canonical
        assert daemon.get_attempt(newer.attempt_id) == newer
        assert calls == [old.attempt_id]
        assert daemon._terminal_portal_failure_event_exists(old.attempt_id, receipt["settlement_id"])
    finally:
        daemon.close()
