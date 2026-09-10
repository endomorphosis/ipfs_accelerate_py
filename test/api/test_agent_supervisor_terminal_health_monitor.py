"""Terminal task state must not retire a live owner's health monitor."""

from types import SimpleNamespace

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
    monkeypatch.setattr(operator, "_status_sample", lambda *a: next(samples))
    monkeypatch.setattr(operator, "_authoritative_progress_between", lambda *a: False)
    monkeypatch.setattr(operator, "_health_receipt", lambda *a, **k: next(receipts))
    monkeypatch.setattr(operator, "_atomic_json", lambda p, r: published.append(r))
    monkeypatch.setattr(operator, "_record_control_failure", lambda *a, **k: failures.append(k))
    operator._status_monitor_loop(
        SimpleNamespace(payload={}), {"status_receipt": "unused"}, None, None,
        launched_at=0.0, previous={"authority": {"available": True}},
        last_progress_at=0.0, stop=SimpleNamespace(wait=lambda _: False),
        failure={}, failure_event=None,
    )
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
