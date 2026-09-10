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
