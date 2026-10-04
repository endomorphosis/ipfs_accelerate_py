"""Native daemon completion must retain the admitted attempt identity."""

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceConflictError,
)
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


def test_completion_forwards_exact_current_attempt_to_validation_owner(
    tmp_path: Path, monkeypatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="completion-attempt-binding")
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        native_record = daemon.task_source.record_validation_result
        calls = []
        native_cas = daemon.task_source.compare_and_set_status
        cas_calls = []
        admitted = daemon.task_source.get(attempt.task_cid)
        expected_control = dict(admitted.body["completion_receipt"])

        def retain_native_record(**kwargs):
            calls.append(dict(kwargs))
            return native_record(**kwargs)

        monkeypatch.setattr(
            daemon.task_source, "record_validation_result", retain_native_record,
        )

        def retain_native_cas(*args, **kwargs):
            cas_calls.append(dict(kwargs))
            return native_cas(*args, **kwargs)

        monkeypatch.setattr(
            daemon.task_source, "compare_and_set_status", retain_native_cas,
        )
        result = daemon.complete_attempt(
            attempt,
            validation_result={
                "outcome": "passed",
                "evidence_digest": "sha256:" + "d" * 64,
                "argv": ["independent-test-validation"],
            },
        )
        assert len(calls) == 1
        assert calls[0]["attempt_id"] == attempt.attempt_id
        assert calls[0]["task_cid"] == attempt.task_cid
        assert len(cas_calls) == 1
        assert cas_calls[0]["expected_revision"] == admitted.revision
        assert cas_calls[0]["expected_control_receipt"] == expected_control
        assert expected_control["attempt_id"] == attempt.attempt_id
        assert result.status == "succeeded"
        assert daemon.task_source.get(attempt.task_cid).status == "completed"
    finally:
        daemon.close()


@pytest.mark.parametrize("changed", ["receipt", "revision"])
def test_completion_refuses_changed_admitted_cas_binding(
    tmp_path: Path, monkeypatch, changed: str,
) -> None:
    daemon = _open_daemon(tmp_path, session="completion-stale-binding")
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)
        original = daemon.task_source.get(attempt.task_cid)
        native_cas = daemon.task_source.compare_and_set_status

        def stale_native_cas(*args, **kwargs):
            if changed == "receipt":
                kwargs["expected_control_receipt"] = {
                    **kwargs["expected_control_receipt"],
                    "attempt_id": "foreign-attempt",
                }
            else:
                kwargs["expected_revision"] -= 1
            return native_cas(*args, **kwargs)

        monkeypatch.setattr(
            daemon.task_source, "compare_and_set_status", stale_native_cas,
        )
        with pytest.raises(TaskSourceConflictError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "a" * 64,
                    "argv": ["independent-test-validation"],
                },
            )
        after = daemon.task_source.get(attempt.task_cid)
        assert after.status == "in_progress"
        assert after.revision == original.revision
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
    finally:
        daemon.close()
