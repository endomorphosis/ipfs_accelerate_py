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
    assert receipt["writes_board"] is False
    assert receipt["writes_locks"] is False
    assert receipt["may_complete_task"] is False
    assert receipt["accepted_as_authority"] is False
    assert receipt["unstall_action"] == "preserve"
    assert receipt["unstall_writes_board"] is False


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
    assert receipt["writes_board"] is False
    assert last_watchdog_classification()["kills_process"] is False


def test_stalled_nominates_unstall_without_kill_or_board_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_watchdog.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    def fake_system_one(_state, questions, **_kwargs):
        if "label" in questions:
            return SimpleNamespace(
                choices={"label": SimpleNamespace(choice="stalled", confidence=0.9)},
                nouls={"false_missing": SimpleNamespace(noul=0.1)},
            )
        return SimpleNamespace(
            choices={"answer": SimpleNamespace(choice="replan_suffix", confidence=0.9)},
            nouls={
                "stale_evidence": SimpleNamespace(noul=0.2),
                "provider_down": SimpleNamespace(noul=0.1),
                "needs_human": SimpleNamespace(noul=0.1),
            },
        )

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        fake_system_one,
    )
    receipt = classify_watchdog_symptom(
        {
            "phase": "implement",
            "stalled": True,
            "worker_count": 0,
            "declared_meta_actions": ("REPLAN_AFFECTED_SUFFIX",),
            "possible_resolution_action_ids": ("action-replan",),
            "action_meta_by_id": {"action-replan": "REPLAN_AFFECTED_SUFFIX"},
        }
    )
    assert receipt["label"] == "stalled"
    assert receipt["unstall_action"] == "replan_suffix"
    assert receipt["kills_process"] is False
    assert receipt["writes_board"] is False
    assert receipt["unstall_writes_board"] is False
    assert last_watchdog_classification()["kills_process"] is False
