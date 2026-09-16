from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_reduce import (
    TRACE_LABELS,
    classify_event_deterministic,
    redact_event,
    reduce_events,
    reduce_jsonl,
)


def test_redact_event_drops_payloads() -> None:
    redacted = redact_event(
        {
            "operation": "workflow_preview",
            "status": "succeeded",
            "error_code": "",
            "authority": "proposal",
            "grant_ids": ["secret-grant"],
            "receipt_id": "sha256:abc",
            "prompt": "do not store",
        }
    )
    dumped = str(redacted)
    assert "secret-grant" not in dumped
    assert "do not store" not in dumped
    assert redacted["operation"] == "workflow_preview"


def test_deterministic_labels_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    report = reduce_events(
        (
            {"operation": "capabilities", "status": "succeeded"},
            {"operation": "prove", "status": "failed", "error_code": "kernel_timeout"},
            {"operation": "generate", "status": "failed", "error_code": "provider_429"},
        )
    )
    assert report.may_complete_task is False
    assert report.source == "deterministic"
    assert report.labels[0] == "success"
    assert report.labels[1] == "stall"
    assert report.labels[2] == "provider_down"
    assert report.counts["success"] == 1
    assert report.counts["stall"] == 1
    assert report.counts["provider_down"] == 1


def test_typesafe_labels_composed_in_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_trace_reduce.typesafe_permitted",
        lambda **_kwargs: True,
    )
    labels = iter(["flaky_fail", "invented"])

    class _Result:
        def __init__(self, choice: str) -> None:
            self.choices = {"label": SimpleNamespace(choice=choice, confidence=0.8)}
            self.nouls = {"retryable": SimpleNamespace(noul=0.9)}

    def fake_system_one(state, questions, **kwargs):
        return _Result(next(labels))

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        fake_system_one,
    )
    report = reduce_events(
        (
            {"status": "failed", "error_code": "flaky"},
            {"status": "failed", "error_code": "kernel_error"},
        )
    )
    assert report.labels[0] == "flaky_fail"
    assert report.labels[1] == "kernel_wait"
    assert report.may_complete_task is False
    assert "invented" not in report.labels
    assert report.retryable == (True, True)


def test_reduce_jsonl_and_never_completes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "control-audit.jsonl"
    path.write_text(
        '{"operation":"capabilities","status":"succeeded"}\n'
        '{"operation":"admit","status":"failed","error_code":"denied"}\n',
        encoding="utf-8",
    )
    report = reduce_jsonl(path)
    assert report.may_complete_task is False
    assert report.to_dict()["may_complete_task"] is False
    assert set(TRACE_LABELS) == set(report.counts)
    assert classify_event_deterministic({"status": "failed", "error_code": "denied"}) == "real_fail"
