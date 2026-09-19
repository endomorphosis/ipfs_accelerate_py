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
    assert receipt["invents_meta_action"] is False
    assert receipt["uncertain"] is False


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


def test_low_confidence_replan_preserves_when_undeclared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="replan_suffix", confidence=0.2)}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action({"reason": "stale_evidence"})
    assert receipt["action"] == "preserve"
    assert "low_confidence_preserve" in receipt["reason_codes"]


def test_low_confidence_keeps_declared_replan_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="replan_suffix", confidence=0.2)}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action(
        {"reason": "stale_evidence"},
        possible_resolution_action_ids=("action:replan",),
        action_meta_by_id={"action:replan": "REPLAN_AFFECTED_SUFFIX"},
    )
    assert receipt["action"] == "replan_suffix"
    assert receipt["meta_action"] == "REPLAN_AFFECTED_SUFFIX"
    assert "prefer_declared_meta_action" in receipt["reason_codes"]
    assert receipt["invents_meta_action"] is False
    assert receipt["writes_board"] is False
    assert receipt["may_complete_task"] is False


def test_undeclared_meta_preserves_and_does_not_invent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="request_human", confidence=0.95)}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall import (
        collect_declared_meta_actions,
    )

    declared = collect_declared_meta_actions(
        ("action:static", "CALL_REMOTE_STRONG_MODEL"),
        declared_meta_actions=("CALL_REMOTE_STRONG_MODEL",),
        action_meta_by_id={"action:static": "RUN_LOCAL_STATIC_ANALYSIS"},
    )
    assert "CALL_REMOTE_STRONG_MODEL" not in declared
    assert "RUN_LOCAL_STATIC_ANALYSIS" not in declared
    assert "REQUEST_HUMAN_DECISION" not in declared
    receipt = nominate_unstall_action(
        {"reason": "stale_evidence"},
        possible_resolution_action_ids=("action:static",),
        action_meta_by_id={"action:static": "RUN_LOCAL_STATIC_ANALYSIS"},
    )
    assert receipt["action"] == "preserve"
    assert receipt["meta_action"] == "NO_OP"
    assert "undeclared_meta_preserve" in receipt["reason_codes"]
    assert receipt["invents_meta_action"] is False


def test_without_key_preserves_even_when_replan_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    receipt = nominate_unstall_action(
        {"reason": "stale_evidence"},
        possible_resolution_action_ids=("REPLAN_AFFECTED_SUFFIX",),
    )
    assert receipt["action"] == "preserve"
    assert receipt["meta_action"] == "NO_OP"
    assert receipt["invents_meta_action"] is False


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


def test_low_confidence_keeps_declared_meta_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="replan_suffix", confidence=0.2)}
        nouls = {
            "stale_evidence": SimpleNamespace(noul=0.95),
            "provider_down": SimpleNamespace(noul=0.1),
            "needs_human": SimpleNamespace(noul=0.1),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action(
        {"reason": "stale_evidence"},
        possible_resolution_action_ids=("action-replan",),
        declared_meta_actions=("REPLAN_AFFECTED_SUFFIX",),
        action_meta_by_id={"action-replan": "REPLAN_AFFECTED_SUFFIX"},
    )
    assert receipt["action"] == "replan_suffix"
    assert receipt["meta_action"] == "REPLAN_AFFECTED_SUFFIX"
    assert "prefer_declared_meta_action" in receipt["reason_codes"]
    assert receipt["writes_board"] is False
    assert receipt["may_complete_task"] is False


def test_undeclared_meta_action_preserves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="replan_suffix", confidence=0.95)}
        nouls = {
            "stale_evidence": SimpleNamespace(noul=0.1),
            "provider_down": SimpleNamespace(noul=0.1),
            "needs_human": SimpleNamespace(noul=0.1),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action(
        {"reason": "stale_evidence"},
        possible_resolution_action_ids=("action-static",),
        declared_meta_actions=("RUN_LOCAL_STATIC_ANALYSIS",),
        action_meta_by_id={"action-static": "RUN_LOCAL_STATIC_ANALYSIS"},
    )
    assert receipt["action"] == "preserve"
    assert receipt["meta_action"] == "NO_OP"
    assert "undeclared_meta_preserve" in receipt["reason_codes"]
    assert receipt["writes_board"] is False


def test_unknown_choice_does_not_invent_declared_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="delete_locks", confidence=0.99)}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action(
        {"reason": "stale_evidence"},
        declared_meta_actions=("REPLAN_AFFECTED_SUFFIX",),
    )
    assert receipt["action"] == "preserve"
    assert "unknown_choice_preserve" in receipt["reason_codes"]
    assert receipt["meta_action"] == "NO_OP"


def test_unstall_mid_band_noul_is_uncertain_not_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="preserve", confidence=0.9)}
        nouls = {
            "stale_evidence": SimpleNamespace(noul=0.5),
            "provider_down": SimpleNamespace(noul=0.1),
            "needs_human": SimpleNamespace(noul=0.1),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_a, **_k: _Result(),
    )
    receipt = nominate_unstall_action({"reason": "unknown"})
    assert receipt["action"] == "preserve"
    assert receipt["uncertain"] is True
    assert "uncertain_band" in receipt["reason_codes"]
    assert "stale_evidence_noul" not in receipt["reason_codes"]
    assert receipt["accepted_as_authority"] is False


def test_unstall_choice_criteria_exposes_meta_action_subtree() -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall import (
        UNSTALL_CHOICE_CRITERIA,
    )

    replan = UNSTALL_CHOICE_CRITERIA["replan_suffix"]["maps_to"]
    assert "REPLAN_AFFECTED_SUFFIX" in replan
    assert "does not complete a task" in replan["REPLAN_AFFECTED_SUFFIX"]
    human = UNSTALL_CHOICE_CRITERIA["request_human"]["maps_to"]
    assert "REQUEST_HUMAN_DECISION" in human
    assert "CALL_REMOTE_STRONG_MODEL" not in UNSTALL_CHOICE_CRITERIA["preserve"]["maps_to"]
