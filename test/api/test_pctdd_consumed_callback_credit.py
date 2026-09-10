"""A no-worker diagnostic cannot refund an admitted, consumed callback credit."""

import pytest

from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


@pytest.mark.parametrize("force_block", [False, True])
def test_unstarted_consumed_credit_stays_blocked(tmp_path, monkeypatch, force_block):
    calls = []
    daemon = _open_daemon(
        tmp_path,
        session="consumed-credit",
        provider_calls=calls,
        max_task_attempts=0,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        # Model an independently admitted exact pair at the finalization
        # boundary. Its verifier has separate positive/negative coverage.
        monkeypatch.setattr(
            daemon, "_retained_recovery_pair_binds_attempt", lambda *_a, **_k: True
        )
        monkeypatch.setattr(
            daemon, "_retained_recovery_pair_is_exact_for_attempt",
            lambda *_a, **_k: True,
        )
        # Even the strongest absence-of-worker diagnostics cannot recreate
        # the credit already consumed by the canonical claim.
        monkeypatch.setattr(daemon, "_task_alias_is_extra_gate", lambda *_a: True)
        monkeypatch.setattr(
            daemon, "_extra_gate_claim_has_not_started_worker",
            lambda *_a: True, raising=False,
        )
        monkeypatch.setattr(
            daemon, "_extra_gate_running_attempt_is_live_local",
            lambda *_a: False, raising=False,
        )
        _, receipt = daemon._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=force_block,
            unknown_authority=True,
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task.status == "blocked"
        assert receipt["retry_exhausted"] is True
        assert receipt["attempts_used"] == 1
        assert calls == []
    finally:
        daemon.close()
