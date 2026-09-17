from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall import (
    last_unstall_nomination,
    nominate_unstall_action,
)


def test_unstall_without_key_preserves(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    receipt = nominate_unstall_action({"reason": "stale_evidence"})
    assert receipt["action"] == "preserve"
    assert receipt["meta_action"] == "NO_OP"
    assert receipt["writes_board"] is False
    assert receipt["writes_locks"] is False
    assert receipt["may_complete_task"] is False
    assert receipt["accepted_as_authority"] is False


def test_unknown_choice_and_low_confidence_preserve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {
            "answer": SimpleNamespace(choice="delete_locks", confidence=0.99),
        }
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action({"reason": "stale_evidence"})
    assert receipt["action"] == "preserve"
    assert "unknown_choice_preserve" in receipt["reason_codes"]


def test_stale_noul_nominates_invalidate_not_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="replan_suffix", confidence=0.9)}
        nouls = {
            "stale_evidence": SimpleNamespace(noul=0.92),
            "provider_down": SimpleNamespace(noul=0.1),
            "needs_human": SimpleNamespace(noul=0.1),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action({"reason": "stale_evidence"})
    assert receipt["action"] == "invalidate_stale_evidence"
    assert receipt["meta_action"] == "NO_OP"
    assert receipt["may_complete_task"] is False
    assert last_unstall_nomination()["writes_board"] is False
