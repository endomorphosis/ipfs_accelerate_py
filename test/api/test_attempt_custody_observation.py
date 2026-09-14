"""Native custody reads cannot become completion or retry authority."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import attempt_custody_observation as obs
from ipfs_accelerate_py.agent_supervisor.todo_daemon.expired_attempt_custody import (
    ExpiredExecutionCustodyPending, guard_generic_retirement,
)
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population
from test.api.test_agent_supervisor_expired_execution_custody import retained_rows


@pytest.fixture
def retained(tmp_path):
    now = {"ms": 1000}
    daemon = _open_daemon(tmp_path, lease_ms=5000, clock_ms=lambda: now["ms"])
    daemon.require_real_execution = True
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.commit_phase(daemon.claim_next(), "context", body={"private": "not on wire"})
        now["ms"] = 7000
        yield daemon, attempt
    finally:
        daemon.close()


def test_exact_expired_read_preserves_all_native_rows(retained):
    daemon, attempt = retained
    before = retained_rows(daemon)
    value = obs.observe_attempt(daemon, attempt)
    assert value["attempt"] == {k: getattr(attempt, k) for k in obs.IDENTITY}
    assert value["execution"]["committed_phase"] == "context"
    assert value["claim"]["state"] == "accepted"
    assert value["claim"]["expires_at_ms"] == 6000
    assert value["coordination_attempt"]["status"] == "running"
    assert value["receipt_counts"] == {"provider_invocation_count": 0, "effect_claim_count": 0}
    assert value["callback_outcome"] == "unknown"
    assert all(value[k] is False for k in obs.DENIALS)
    assert b"not on wire" not in obs._bytes(value)
    assert retained_rows(daemon) == before
    assert obs.observe_attempt(daemon, attempt) == value


@pytest.mark.parametrize("reader", ["get_attempt", "get_task_claim", "get_task_attempt"])
def test_missing_native_row_is_unavailable_not_no_effects(retained, monkeypatch, reader):
    daemon, attempt = retained
    target = daemon if reader == "get_attempt" else daemon.coordinator
    monkeypatch.setattr(target, reader, lambda _: None)
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.observe_attempt(daemon, attempt)


@pytest.mark.parametrize("reader", ["get_attempt", "get_task_claim", "get_task_attempt"])
def test_cross_attempt_native_row_rejected(retained, monkeypatch, reader):
    daemon, attempt = retained
    target = daemon if reader == "get_attempt" else daemon.coordinator
    original = getattr(target, reader)
    monkeypatch.setattr(target, reader, lambda key: replace(original(key), attempt_id="attempt:foreign"))
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.observe_attempt(daemon, attempt)


@pytest.mark.parametrize("reader", ["get_attempt", "get_task_claim", "get_task_attempt", "phase_history",
                                    "_attempt_execution_evidence_counts"])
def test_second_read_change_rejected(retained, monkeypatch, reader):
    daemon, attempt = retained
    target = daemon.coordinator if reader in {"get_task_claim", "get_task_attempt"} else daemon
    original = getattr(target, reader)
    calls = []

    def changed(key):
        value = original(key)
        calls.append(key)
        if len(calls) == 2:
            if reader == "phase_history":
                value[0]["body"]["changed"] = True
            elif reader == "_attempt_execution_evidence_counts":
                value["effect_claim_count"] += 1
            else:
                value = replace(value, revision=value.revision + 1)
        return value

    monkeypatch.setattr(target, reader, changed)
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.observe_attempt(daemon, attempt)


def test_reader_cannot_observe_another_session(retained):
    daemon, attempt = retained
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.observe_attempt(daemon, replace(attempt, owner_session_id="session:foreign"))


@pytest.mark.parametrize("change", ["extra", "retry", "fence", "body_hash", "bool_count", "phase_revision"])
def test_closed_relay_rejects_bad_evidence_even_with_recomputed_digest(retained, change):
    daemon, attempt = retained
    value = obs.observe_attempt(daemon, attempt)
    if change == "extra":
        value["extra"] = "unknown"
    elif change == "retry":
        value["retry_authorized"] = True
    elif change == "fence":
        value["phases"][0]["fence_epoch"] += 1
    elif change == "body_hash":
        value["claim"]["body_sha256"] = "made up"
    elif change == "bool_count":
        value["receipt_counts"]["effect_claim_count"] = False
    else:
        value["phases"][-1]["revision"] += 1
    value["observation_sha256"] = obs._digest({k: v for k, v in value.items() if k != "observation_sha256"})
    with pytest.raises(obs.AttemptObservationUnavailable):
        obs.validate_observation(value)


def test_deferral_reports_exact_custody_and_preserves_unknown_effects(retained):
    daemon, attempt = retained
    reports = []
    daemon._native_dispatch_control = SimpleNamespace(
        retained_attempt_custody=lambda d, a: reports.append(obs.observe_attempt(d, a)),
    )
    before = retained_rows(daemon)
    with pytest.raises(ExpiredExecutionCustodyPending):
        guard_generic_retirement(daemon, attempt, reason="claim_authority_expired")
    assert reports[0]["attempt"]["attempt_id"] == attempt.attempt_id
    assert retained_rows(daemon) == before


def test_reader_failure_never_bypasses_retirement_guard(retained):
    daemon, attempt = retained

    def unavailable(*_):
        raise OSError("native observation unavailable")

    daemon._native_dispatch_control = SimpleNamespace(retained_attempt_custody=unavailable)
    before = retained_rows(daemon)
    with pytest.raises(ExpiredExecutionCustodyPending):
        guard_generic_retirement(daemon, attempt, reason="claim_authority_expired")
    assert retained_rows(daemon) == before


def test_native_client_clears_prior_observation_when_reader_becomes_unavailable(retained, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.native_dispatch_drain import NativeDispatchClient

    daemon, attempt = retained
    client = object.__new__(NativeDispatchClient)
    messages = []
    client.exchange = lambda operation, body: messages.append((operation, body))
    client.retained_attempt_custody(daemon, attempt)
    assert messages[-1][0] == "custody_boundary"
    assert messages[-1][1]["observation"]["attempt"]["attempt_id"] == attempt.attempt_id
    monkeypatch.setattr(daemon.coordinator, "get_task_claim", lambda _: None)
    client.retained_attempt_custody(daemon, attempt)
    assert messages[-1] == ("custody_boundary", {"observation": None})
