from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_watchdog import (
    classify_watchdog_symptom,
    last_watchdog_classification,
)


def test_watchdog_without_key_does_not_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    receipt = classify_watchdog_symptom(
        {"phase": "implement", "stalled": True, "worker_count": 0}
    )
    assert receipt["kills_process"] is False
    assert receipt["rewrites_supervisor"] is False
    assert receipt["may_complete_task"] is False
    assert receipt["accepted_as_authority"] is False


def test_false_missing_noul_does_not_authorize_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_watchdog.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"label": SimpleNamespace(choice="stalled", confidence=0.9)}
        nouls = {"false_missing": SimpleNamespace(noul=0.95)}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = classify_watchdog_symptom(
        {"phase": "implement", "stalled": True, "worker_count": 1}
    )
    assert receipt["label"] == "false_missing"
    assert receipt["kills_process"] is False
    assert last_watchdog_classification()["kills_process"] is False
