"""Bound consumed candidate retry intent without granting settlement authority."""

from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalCandidateRetry,
    DatabasePortalExecutionBridge,
)
from test.api.test_agent_supervisor_database_portal_bridge import (
    _DatabaseAttemptAuthorityPortal,
    _TaskSource,
    _attempt,
    _record,
)


def _run(tmp_path: Path, result: dict, *, limit=4, database_attempt=1):
    class Portal(_DatabaseAttemptAuthorityPortal):
        closed = False
        calls = 0

        def run_once(self):
            self.calls += 1
            return result

        def close_event_runtime(self):
            self.closed = True

    portal = Portal()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: portal,
        max_task_attempts=limit,
    )
    with pytest.raises(DatabasePortalBridgeError) as caught:
        bridge.run_provider(replace(_attempt(), attempt_number=database_attempt))
    assert portal.closed and portal.calls == 1
    paths = bridge._paths(_attempt())
    assert "- Status: ready" in paths.task_projection.read_text()
    assert not paths.events.exists()  # Failure intent never fabricates a durable event.
    return caught.value


def _candidate(**updates):
    result = {
        "returncode": 78,
        "attempt": 1,
        "attempt_consumed": True,
        "provider_dispatched": True,
        "validation_result": {
            "attempted": True,
            "passed": False,
            "reason": "proposal_gate_failed",
            "proposal_gate": {
                "reason_codes": [
                    "validation_channel_tampering_forbidden",
                    "sk_private_fixture",
                ],
                "raw_provider_output": "private fixture output",
            },
        },
    }
    result.update(updates)
    return {"implementation_result": result}


@pytest.mark.parametrize("limit", [True, False, -1, 10001, 1.0, "4", None])
def test_attempt_budget_rejects_invalid_limits(tmp_path, limit):
    with pytest.raises(ValueError, match="max_task_attempts"):
        DatabasePortalExecutionBridge(
            task_source=_TaskSource(_record()),
            attempt_root=tmp_path / "attempts",
            portal_factory=lambda *_: None,
            max_task_attempts=limit,
        )
    assert not (tmp_path / "attempts").exists()


@pytest.mark.parametrize("database_attempt,portal_attempt", [(1, 1), (3, 1), (1, 3)])
def test_candidate_retry_is_consumed_and_bound_to_both_ordinals(
    tmp_path, database_attempt, portal_attempt
):
    error = _run(
        tmp_path, _candidate(attempt=portal_attempt), database_attempt=database_attempt
    )
    assert type(error) is DatabasePortalCandidateRetry
    assert error.reason == "proposal_gate_failed"
    assert error.attempt_consumed is error.provider_dispatched is True
    assert error.backoff_seconds == 0
    assert error.result["attempt_budget"] == {
        "budget_kind": "coordination_attempt_safety_cap",
        "database_attempt_number": database_attempt,
        "portal_attempt_number": portal_attempt,
        "bounded_attempt_ordinal": max(database_attempt, portal_attempt),
        "max_task_attempts": 4,
        "retry_allowed": True,
        "completion_authority": False,
        "claim_release_authority": False,
    }
    assert error.result["implementation"]["validation_result"]["proposal_gate"] == {
        "reason_codes": ["validation_channel_tampering_forbidden"]
    }
    assert "private" not in str(error.result)


@pytest.mark.parametrize("database_attempt,portal_attempt,limit", [
    (4, 1, 4), (1, 4, 4), (5, 1, 4), (1, 5, 4), (1, 1, 0), (1, 1, 1),
])
def test_exhausted_budget_cannot_be_reset_by_new_projection(
    tmp_path, database_attempt, portal_attempt, limit
):
    error = _run(
        tmp_path, _candidate(attempt=portal_attempt),
        database_attempt=database_attempt, limit=limit,
    )
    assert type(error) is DatabasePortalBridgeError
    assert str(error) == "proposal_gate_failed"
    assert error.result["attempt_budget"]["retry_allowed"] is False
    assert error.result["implementation"]["attempt_consumed"] is True


@pytest.mark.parametrize("portal_attempt", [True, None, 0, -1, "1", 65536])
def test_candidate_requires_typed_bounded_local_ordinal(tmp_path, portal_attempt):
    error = _run(tmp_path, _candidate(attempt=portal_attempt))
    assert type(error) is DatabasePortalBridgeError
    assert "invalid Portal attempt ordinal" in str(error)


def test_deferred_flag_cannot_erase_consumed_candidate(tmp_path):
    error = _run(tmp_path, _candidate(deferred=True))
    assert type(error) is DatabasePortalCandidateRetry
    assert error.attempt_consumed is True


@pytest.mark.parametrize("updates", [
    {"returncode": True}, {"returncode": "78"}, {"attempt_consumed": 1},
    {"attempt_consumed": False}, {"provider_dispatched": 1},
    {"provider_dispatched": False},
])
def test_malformed_candidate_flags_never_grant_retry(tmp_path, updates):
    # deferred keeps malformed return codes on the failure path, without
    # allowing an unconsumed result to qualify as a pre-dispatch deferral.
    error = _run(tmp_path, _candidate(deferred=True, **updates))
    assert type(error) is DatabasePortalBridgeError


@pytest.mark.parametrize("provider_flags", [
    {"provider_dispatched": False}, {"provider_call_allowed": False},
])
def test_explicit_non_dispatch_deferral_preserves_backoff(tmp_path, provider_flags):
    error = _run(tmp_path, {"implementation_result": {
        "returncode": 1, "reason": "validation_project_dependency_preflight_failed",
        "deferred": True, "attempt_consumed": False, "backoff_seconds": 7,
        **provider_flags,
    }})
    assert type(error) is DatabasePortalBridgeDeferred
    assert error.backoff_seconds == 7
    assert error.attempt_consumed is error.provider_dispatched is False
    assert error.result["implementation"]["backoff_seconds"] == 7


@pytest.mark.parametrize("backoff", [True, -1, 86401, "7"])
def test_malformed_deferral_backoff_is_rejected(tmp_path, backoff):
    error = _run(tmp_path, {"implementation_result": {
        "returncode": 1, "deferred": True, "attempt_consumed": False,
        "provider_dispatched": False, "backoff_seconds": backoff,
    }})
    assert type(error) is DatabasePortalBridgeError
    assert "backoff" in str(error)


@pytest.mark.parametrize("call_allowed", [True, "false", 0])
def test_contradictory_or_untyped_provider_admission_never_defers(tmp_path, call_allowed):
    error = _run(tmp_path, {"implementation_result": {
        "returncode": 1, "deferred": True, "attempt_consumed": False,
        "provider_dispatched": False, "provider_call_allowed": call_allowed,
    }})
    assert type(error) is DatabasePortalBridgeError


@pytest.mark.parametrize("reason", [
    "custom_deferred", "capacity", "backoff", "resource_claim",
    "worktree_lifecycle_claim_exists", "inflight_process",
])
def test_reason_text_alone_is_not_non_consuming_retry_authority(tmp_path, reason):
    error = _run(tmp_path, {"implementation_result": {
        "reason": reason, "returncode": 1,
    }})
    assert type(error) is DatabasePortalBridgeError
    assert str(error) == reason


def test_declared_validation_cannot_use_candidate_retry_receipt(tmp_path):
    candidate = _candidate(reason="proposal_gate_failed")
    candidate["implementation_result"]["validation_result"]["reason"] = (
        "declared_validation_failed"
    )
    assert type(_run(tmp_path, candidate)) is DatabasePortalBridgeError


def test_candidate_exception_refuses_untyped_reason_or_backoff():
    with pytest.raises(ValueError, match="closed candidate"):
        DatabasePortalCandidateRetry("arbitrary_failure")
    with pytest.raises(ValueError, match="backoff"):
        DatabasePortalCandidateRetry("proposal_gate_failed", backoff_seconds=True)


def test_mixed_pre_dispatch_deferrals_are_not_reported_as_provider_consumption(tmp_path):
    """The conservative ordinal ceiling is explicitly distinct from model usage."""
    deferred = {"implementation_result": {
        "returncode": 1, "reason": "worktree_lifecycle_claim_exists",
        "deferred": True, "attempt_consumed": False,
        "provider_dispatched": False,
    }}
    observations = []
    for ordinal, result in ((1, deferred), (2, deferred), (3, _candidate())):
        attempt_root = tmp_path / str(ordinal)
        attempt_root.mkdir()
        observations.append(_run(attempt_root, result, limit=3, database_attempt=ordinal))
    for error in observations[:2]:
        assert type(error) is DatabasePortalBridgeDeferred
        assert error.attempt_consumed is False
        assert "attempt_budget" not in error.result
    failure = observations[-1]
    assert type(failure) is DatabasePortalBridgeError
    assert failure.result["attempt_budget"]["budget_kind"] == "coordination_attempt_safety_cap"
    assert failure.result["attempt_budget"]["retry_allowed"] is False
    # Only the third observed result says a model was dispatched. Reaching
    # coordination ordinal three must not manufacture three paid model calls.
    assert sum(error.result["implementation"]["provider_dispatched"] for error in observations) == 1


def test_runner_binds_parsed_attempt_limit_to_real_bridge(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon_runner as runner,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
        PortalImplementationDaemon,
        parse_args,
    )

    parsed = parse_args([
        "--task-source-kind", "duckdb", "--authority-mode", "embedded_exclusive",
        "--database-path", str(tmp_path / "control.duckdb"),
        "--state-dir", str(tmp_path / "state"), "--state-prefix", "budget",
        "--max-task-attempts", "3", "--implement", "--once",
    ])
    monkeypatch.setattr(runner, "_mirror_database_portal_execution_binding", lambda _: None)
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        authority_mode="embedded_exclusive", task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        bridge = runner.bind_database_portal_execution_from_args(
            daemon, parsed, repo_root=tmp_path,
            portal_daemon_class=PortalImplementationDaemon,
        )
        assert type(bridge) is DatabasePortalExecutionBridge
        assert bridge.task_source is daemon.task_source
        assert bridge.max_task_attempts == 3
        assert daemon.execution_callbacks_bound is True
        assert daemon._provider_fn.__self__ is bridge
    finally:
        daemon.close()
