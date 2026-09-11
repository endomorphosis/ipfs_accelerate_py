"""Legacy unknown callbacks through actual typed admission and the public tick.

The owner-issued gateway, task/claim/attempt stores and callback journal are real
disposable fixtures. The existing fixture uses a TCP Quack transport double and
an injected provider process interruption; neither supplies settlement authority.
"""

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    candidate_rejection_closure as closure,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as impl,
)
from test.api import test_candidate_rejection_closure as fixtures
from test.api.test_database_attempt_diagnostic_feedback import prompt_daemon


class InterruptedProcess(BaseException):
    pass


@pytest.mark.parametrize("origin", ["interrupted", "cleanup_unavailable"])
def test_public_tick_retains_legacy_unknown_callback_custody(
    tmp_path, monkeypatch, origin
):
    route_fixture = fixtures.routes._reviewed_route(tmp_path)
    calls = []

    def factory(paths, alias):
        daemon = prompt_daemon(monkeypatch)
        daemon.repo_root = route_fixture[0]
        daemon.bind_launch_task_execution_route = lambda _: None
        daemon.close_event_runtime = lambda: None

        def run():
            calls.append(alias)
            if origin == "interrupted":
                # Normal Portal records its start before invoking the runner.
                # Retain that real chain prefix so this is known legacy history;
                # absent/unreadable history has a separate journal-unknown guard.
                binding = json.loads(paths.binding.read_text())
                canonical_key, canonical_cid = bridge._portal_completion_event_identity(
                    paths=paths,
                    projection_text=paths.task_projection.read_text(),
                    binding=binding,
                )
                impl.append_jsonl_event(
                    paths.events,
                    "implementation_started",
                    {
                        "task_id": alias,
                        "attempt": 1,
                        "canonical_task_key": canonical_key,
                        "canonical_task_cid": canonical_cid,
                    },
                )
                raise InterruptedProcess("provider callback process interrupted")
            task = impl.parse_task_text(
                paths.task_projection.read_text(),
                path=paths.task_projection,
                task_header_prefix=f"## {alias}",
            )[0]
            return fixtures._preserved_candidate_pass(
                tmp_path,
                monkeypatch,
                bridge=bridge,
                paths=paths,
                task=task,
                route_fixture=route_fixture,
                complete=False,
            )

        daemon.run_once = run
        return daemon

    with fixtures._typed_outer(
        tmp_path, monkeypatch, repository=route_fixture[0], factory=factory, tick=True
    ) as (outer, bridge, source):
        attempt = outer.claim_next()
        assert attempt is not None
        with pytest.raises(
            InterruptedProcess
            if origin == "interrupted"
            else closure.CandidateClosureObservationUnknown
        ):
            outer._resume_attempt_without_process_crash(attempt)
        key = f"provider:{attempt.attempt_id}"
        original = outer.provider_invocation_recorded(
            attempt.attempt_id, idempotency_key=key
        )
        assert original["callback_state"] == "started_outcome_unknown"
        assert original["provider_effect_state"] == "unknown_may_have_started"
        task = source.get_task(attempt.task_cid)
        before_task = (task.revision, task.status, dict(task.body))
        before_attempt = outer.get_attempt(attempt.attempt_id)
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        )

        for _ in range(2):
            result = outer.run_once()
            task = source.get_task(attempt.task_cid)
            current_attempt = outer.get_attempt(attempt.attempt_id)
            assert (task.revision, task.status, dict(task.body)) == before_task
            assert (
                outer.coordinator.get_task_claim(attempt.claim_id).state.value
                == "accepted"
            )
            assert current_attempt.status == before_attempt.status
            assert current_attempt.committed_phase == before_attempt.committed_phase
            assert (
                outer.provider_invocation_recorded(
                    attempt.attempt_id, idempotency_key=key
                )
                == original
            )
            assert len(calls) == 1
            assert result["reason"] == "database_provider_callback_outcome_unknown"
            assert result["deferred"] is True
            assert result["active_task_id"] == attempt.task_cid
            assert result["attempt_id"] == attempt.attempt_id
            assert result["implementation_result"] is None
            assert result["unchanged"] is None
            assert "write_count" not in result
            assert result["provider_dispatched"] == "unknown"
            assert result["attempt_consumed"] == "unknown"
            assert result["completion_authority"] is False
            assert result["backoff_seconds"] > 0
            assert (
                result["recovery_prefix"]["post_merge_recovery"]["reason"]
                != "post_merge_recovery_callback_failed"
            )
            assert outer._idle_recovery_prefix is None
        assert outer.claim_next() is None
        assert len(calls) == 1


def test_exact_released_unstarted_attempt_still_retires(tmp_path, monkeypatch):
    repository = fixtures.routes._reviewed_route(tmp_path)[0]
    with fixtures._typed_outer(
        tmp_path,
        monkeypatch,
        repository=repository,
        factory=lambda *_: pytest.fail("unstarted stale work must not dispatch"),
        tick=True,
    ) as (outer, _bridge, _source):
        attempt = outer.claim_next()
        assert attempt is not None
        key = f"provider:{attempt.attempt_id}"
        assert (
            outer.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key)
            is None
        )
        claim = outer.coordinator.get_task_claim(attempt.claim_id)
        lease = outer._protect_attempt_claim(attempt, claim)
        outer.coordinator.release(
            lease,
            reason="fixture_unstarted_dispatch_abandoned",
            expected_fencing_token=attempt.fencing_token,
            expected_fence_epoch=attempt.fence_epoch,
            now_ms=outer._now_ms(),
        )
        result = outer._resume_attempt_without_process_crash(attempt)
        assert result["status"] == "failed"
        assert result["reason"] != "database_provider_callback_outcome_unknown"
        assert outer.get_attempt(attempt.attempt_id).status == "failed"
        assert (
            outer.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        )
        assert (
            outer.provider_invocation_recorded(attempt.attempt_id, idempotency_key=key)
            is None
        )
