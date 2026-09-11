"""Terminal task state must not retire a live owner's health monitor."""

from types import SimpleNamespace
import threading

import pytest

from scripts import run_agent_supervisor_efficiency_state_hardening as operator


def terminal_receipt():
    return {
        "terminal": True,
        "healthy": True,
        "scheduler_alive": True,
        "owner_ready": True,
        "broker_ready": True,
    }


def test_terminal_monitor_resamples_and_detects_owner_loss(monkeypatch):
    samples = iter([
        {"observed_at": 10.0, "authority": {"available": True}},
        {"observed_at": 20.0, "authority": {"available": True}},
        {"observed_at": 30.0, "authority": {"available": True}},
    ])
    receipts = iter([
        terminal_receipt(), terminal_receipt(),
        {**terminal_receipt(), "healthy": False, "owner_ready": False},
    ])
    published = []
    failures = []
    pairs = []
    failure_event = threading.Event()

    def record_failure(*args, **kwargs):
        failures.append(kwargs)
        failure_event.set()

    monkeypatch.setattr(operator, "_status_sample", lambda *a: next(samples))
    monkeypatch.setattr(operator, "_authoritative_progress_between", lambda *a: False)
    def health_receipt(*args, **kwargs):
        pairs.append(kwargs["samples"])
        assert kwargs["last_progress_at"] == 0.0
        return next(receipts)

    monkeypatch.setattr(operator, "_health_receipt", health_receipt)
    monkeypatch.setattr(operator, "_atomic_json", lambda p, r: published.append(r))
    monkeypatch.setattr(operator, "_record_control_failure", record_failure)
    operator._status_monitor_loop(
        SimpleNamespace(payload={}), {"status_receipt": "unused"}, None, None,
        launched_at=0.0, previous={"authority": {"available": True}},
        last_progress_at=0.0, stop=SimpleNamespace(wait=lambda _: False, is_set=lambda: False),
        failure={}, failure_event=failure_event,
    )
    assert [pair[0].get("observed_at") for pair in pairs] == [None, 10.0, 20.0]
    assert [pair[1]["observed_at"] for pair in pairs] == [10.0, 20.0, 30.0]
    assert len(published) == 3
    assert failures == [{
        "reason_code": "authoritative_owner_not_ready",
        "error_type": "ASEHHealthGateFailure",
    }]


@pytest.mark.parametrize("field,reason", [
    ("scheduler_alive", "authoritative_scheduler_not_live"),
    ("owner_ready", "authoritative_owner_not_ready"),
    ("broker_ready", "authoritative_broker_not_ready"),
    ("healthy", "authoritative_terminal_not_admitted"),
])
def test_terminal_state_does_not_bypass_health_admission(field, reason):
    receipt = {**terminal_receipt(), field: False}
    assert operator._post_admission_health_action(
        receipt, prior_available=True, current_available=True, unhealthy_edges=0,
    ) == ("fail", reason, 0)


def test_legacy_terminal_return_cannot_retire_retained_observation(monkeypatch):
    original = operator._post_admission_health_action

    def legacy_terminal_return(receipt, **kwargs):
        result = original(receipt, **kwargs)
        if receipt.get("healthy") is True and receipt.get("terminal") is True:
            return "stop", "", 0
        return result

    monkeypatch.setattr(operator, "_post_admission_health_action", legacy_terminal_return)
    test_terminal_monitor_resamples_and_detects_owner_loss(monkeypatch)


def test_retained_observation_preserves_outage_budget(monkeypatch):
    failure_event = threading.Event()
    samples, failures = [], []
    unavailable = {"authority": {"available": False}}

    def sample(*args):
        samples.append(len(samples) + 1)
        assert len(samples) <= 3, "outage budget reset between samples"
        return {**unavailable, "observed_at": float(len(samples))}

    def record_failure(*args, **kwargs):
        failures.append(kwargs)
        failure_event.set()

    monkeypatch.setattr(operator, "_status_sample", sample)
    monkeypatch.setattr(operator, "_authoritative_progress_between", lambda *a: False)
    monkeypatch.setattr(operator, "_health_receipt", lambda *a, **k: {
        "scheduler_alive": True, "owner_ready": True, "broker_ready": True,
        "healthy": False,
    })
    monkeypatch.setattr(operator, "_atomic_json", lambda *a: None)
    monkeypatch.setattr(operator, "_record_control_failure", record_failure)
    operator._status_monitor_loop(
        SimpleNamespace(payload={}), {"status_receipt": "unused"}, None, None,
        launched_at=0.0, previous=unavailable, last_progress_at=0.0,
        stop=SimpleNamespace(wait=lambda _: False, is_set=lambda: False),
        failure={}, failure_event=failure_event,
    )
    assert len(samples) == 3
    assert failures == [{
        "reason_code": "authoritative_status_unavailable_two_samples",
        "error_type": "ASEHHealthQueryFailure",
    }]


def test_refresh_request_consumed_only_by_existing_retained_sampler(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime.owner_observation_request import (
        OwnerObservationRequests, observation_scope,
    )
    request_scope = observation_scope(
        program_id=operator.PROGRAM, owner_identity={"generation": 1},
        source_head="a" * 40, source_tree="b" * 40,
        launch_admission_id="sha256:test",
    )
    requests = OwnerObservationRequests(request_scope)
    stop, failed = threading.Event(), threading.Event()
    requests.pending.set()
    pairs = []
    caller_thread = threading.get_ident()

    def sample(*args):
        assert threading.get_ident() == caller_thread
        assert not requests.pending.is_set()
        return {"observed_at": 2.0, "authority": {"available": True}}

    def receipt(*args, **kwargs):
        pairs.append(kwargs["samples"])
        assert kwargs["last_progress_at"] == 0.5
        return terminal_receipt()

    monkeypatch.setattr(operator, "_status_sample", sample)
    monkeypatch.setattr(operator, "_authoritative_progress_between", lambda *a: False)
    monkeypatch.setattr(operator, "_health_receipt", receipt)
    monkeypatch.setattr(operator, "_atomic_json", lambda *a: stop.set())
    try:
        operator._status_monitor_loop(
            SimpleNamespace(payload={}), {"status_receipt": "unused"}, None, None,
            launched_at=0.0, previous={"observed_at": 1.0}, last_progress_at=0.5,
            stop=SimpleNamespace(wait=lambda _: stop.is_set(), is_set=stop.is_set),
            failure={}, failure_event=failed, observation_requests=requests,
        )
    finally:
        requests.close()
    assert len(pairs) == 1
    assert pairs[0][0]["observed_at"] == 1.0
    assert pairs[0][1]["observed_at"] == 2.0
    assert not failed.is_set()


def test_old_owner_launch_has_no_refresh_admission(monkeypatch):
    from pathlib import Path
    launch = {"schema": "ipfs_accelerate_py/agent-supervisor/aseh-owner-launch@1"}
    launch["receipt_cid"] = operator._identity(launch)
    monkeypatch.setattr(operator, "_load", lambda p: (object(), {}))
    monkeypatch.setattr(operator, "_paths", lambda b: {"evidence": Path("unused")})
    monkeypatch.setattr(operator, "_secure_runtime_json", lambda *a, **k: launch)
    with pytest.raises(operator.OperatorError, match="no observation request route"):
        operator.request_status_refresh(Path("unused"))


def test_repeated_refresh_requests_do_not_reset_outage_budget(monkeypatch):
    original = operator._status_monitor_loop
    starts = []
    requests = SimpleNamespace(sample_started=lambda: starts.append(len(starts)))
    def with_requests(*args, **kwargs):
        return original(*args, **kwargs, observation_requests=requests)
    monkeypatch.setattr(operator, "_status_monitor_loop", with_requests)
    test_retained_observation_preserves_outage_budget(monkeypatch)
    assert len(starts) == 3
