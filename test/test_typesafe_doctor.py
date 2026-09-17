from __future__ import annotations

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import HIGH_CONFIDENCE
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_doctor import (
    filter_candidates_for_tactician,
    hammer_timeout_hint,
    last_hammer_hint,
    order_candidates_for_hammer,
    select_retrieve_ids_for_tactician,
)


def test_no_key_keeps_all_retrieve_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    rows = (
        {"id": "cand-a", "path": "src/a.py"},
        {"id": "cand-b", "path": "src/b.py"},
    )
    assert select_retrieve_ids_for_tactician(rows) == ("cand-a", "cand-b")
    assert filter_candidates_for_tactician(rows) == rows


def test_drops_only_high_confidence_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_doctor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    def fake_score(*, candidate_id, **_kwargs):
        if candidate_id == "cand-junk":
            return SimpleNamespace(
                action="scored",
                score=0.0,
                confidence=HIGH_CONFIDENCE,
            )
        return SimpleNamespace(action="scored", score=1.8, confidence=0.9)

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.score_synthesis_candidate",
        fake_score,
    )
    rows = (
        {"id": "cand-good", "path": "src/a.py"},
        {"id": "cand-junk", "path": "tmp/noise.py"},
    )
    assert select_retrieve_ids_for_tactician(rows) == ("cand-good",)


def test_dropping_everyone_keeps_original(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_doctor.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.score_synthesis_candidate",
        lambda **_kwargs: SimpleNamespace(
            action="scored", score=0.0, confidence=HIGH_CONFIDENCE
        ),
    )
    rows = ({"id": "cand-a"}, {"id": "cand-b"})
    assert select_retrieve_ids_for_tactician(rows) == ("cand-a", "cand-b")
    assert filter_candidates_for_tactician(rows) == rows


def test_hammer_timeout_hint_fail_open_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert hammer_timeout_hint(finding_id="finding-1") == {}


def test_hammer_timeout_hint_is_advisory_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_doctor.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.triage_smt",
        lambda **_kwargs: SimpleNamespace(claim_status="unsat", confidence=0.81),
    )
    hint = hammer_timeout_hint(finding_id="finding-1", english="forall x. P x -> P x")
    assert hint["typesafe_hint_only"] is True
    assert hint["claim_status"] == "unsat"
    assert hint["accepted_as_authority"] is False
    assert last_hammer_hint()["accepted_as_authority"] is False


def test_order_candidates_for_hammer_fail_open_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "TYPESAFE_API_KEY",
        "ipfs_accelerate_py_TYPESAFE_API_KEY",
        "IPFS_ACCELERATE_PY_TYPESAFE_API_KEY",
        "IPFS_DATASETS_PY_TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    rows = ({"id": "low"}, {"id": "high"})
    assert order_candidates_for_hammer(rows) == rows


def test_order_candidates_for_hammer_puts_higher_score_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_doctor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    def fake_score(*, candidate_id, **_kwargs):
        score = 1.9 if candidate_id == "high" else 0.2
        return SimpleNamespace(action="scored", score=score, confidence=0.8)

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.score_synthesis_candidate",
        fake_score,
    )
    rows = ({"id": "low", "path": "b.py"}, {"id": "high", "path": "a.py"})
    ordered = order_candidates_for_hammer(rows)
    assert [row["id"] for row in ordered] == ["high", "low"]
    hint = last_hammer_hint()
    assert hint["typesafe_hint_only"] is True
    assert hint["accepted_as_authority"] is False
    assert hint["ordered_ids"] == ["high", "low"]
