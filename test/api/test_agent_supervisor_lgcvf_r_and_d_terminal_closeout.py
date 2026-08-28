"""Focused tests for the append-only LGCVF terminal R&D closeout."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from scripts import validate_lgcvf_r_and_d_terminal_closeout as terminal


def _cid(label: str) -> str:
    return content_identity({"fixture": label})


def _predecessor_validation() -> dict[str, Any]:
    return {
        "schema": "lgcvf-implementation-report-validation@1",
        "valid": True,
        "report_sha256": "sha256:" + "1" * 64,
        "successor_tasks_cid": _cid("predecessor"),
        "benchmark_cid": _cid("benchmark"),
        "benchmark_authority_cid": _cid("benchmark-authority"),
        "qualification_cid": _cid("qualification"),
        "qualification_authority_cid": _cid("qualification-authority"),
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "benchmark_replay_cid": _cid("volatile-benchmark-replay"),
        "qualification_replay_cid": _cid("volatile-qualification-replay"),
        "validation_cid": _cid("volatile-validation"),
    }


def _install_reconstruction(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    predecessor = {
        "plan_cid": _cid("plan"),
        "successor_tasks_cid": _cid("predecessor"),
    }
    qualification = {
        "result_cid": _cid("qualification"),
        "checkout_fingerprint_cid": _cid("checkout"),
    }
    benchmark = {"report_cid": _cid("benchmark"), "overall_disposition": "partial"}
    external = {"receipt_cid": _cid("external")}
    production = {"receipt_cid": _cid("production")}
    tasks = []
    for task_id in terminal.EXPECTED_TASK_IDS:
        evidence = {"evidence_cid": _cid(task_id + "-evidence")}
        tasks.append(
            {
                "task_id": task_id,
                "disposition": terminal.EXPECTED_DISPOSITIONS[task_id],
                "predecessor_task_cid": _cid(task_id + "-predecessor"),
                "task_resolution_cid": _cid(task_id + "-resolution"),
                "evidence": evidence,
            }
        )
    resolution = {"resolution_cid": _cid("resolution"), "tasks": tasks}
    by_path = {
        terminal.PREDECESSOR_PATH: predecessor,
        terminal.QUALIFICATION_PATH: qualification,
        terminal.BENCHMARK_PATH: benchmark,
        terminal.EXTERNAL_RECEIPT_PATH: external,
        terminal.PRODUCTION_RECEIPT_PATH: production,
        terminal.RESOLUTION_PATH: resolution,
    }

    monkeypatch.setattr(
        terminal,
        "_load_object",
        lambda path, *, label: copy.deepcopy(by_path[path]),
    )
    monkeypatch.setattr(
        terminal,
        "verify_current_successor_resolution",
        lambda: {"check_cid": _cid("successor-check")},
    )
    roots = {
        "ipfs_accelerate_py": {"head": "1" * 40, "tree": "2" * 40},
        "ipfs_datasets_py": {
            "head": "3" * 40,
            "tree": "4" * 40,
            "gitlink": "3" * 40,
        },
    }
    monkeypatch.setattr(
        terminal,
        "current_source_revisions",
        lambda: SimpleNamespace(to_dict=lambda: roots),
    )
    monkeypatch.setattr(
        terminal,
        "load_trust_policy",
        lambda: SimpleNamespace(
            key_id=_cid("key"),
            identity="Benjamin Barber",
            role="sole R&D verifier and operator",
        ),
    )
    monkeypatch.setattr(
        terminal,
        "_sha256_file",
        lambda path: "sha256:" + ("a" if path == terminal.RELEASE_PATH else "b") * 64,
    )
    monkeypatch.setattr(terminal, "_input_snapshots", dict)
    monkeypatch.setattr(
        terminal, "_require_snapshots_unchanged", lambda snapshots: None
    )
    return _predecessor_validation()


def _reseal(value: dict[str, Any]) -> None:
    value.pop("closeout_cid", None)
    value["closeout_cid"] = content_identity(value)


def test_builds_exact_terminal_r_and_d_state(monkeypatch: pytest.MonkeyPatch) -> None:
    predecessor = _install_reconstruction(monkeypatch)
    closeout = terminal.build_terminal_closeout(predecessor_validation=predecessor)

    assert closeout["task_implementation_complete"] is True
    assert closeout["test_qualification_complete"] is True
    assert closeout["objective_complete"] is False
    assert closeout["release_qualified"] is False
    assert closeout["production_authorized"] is False
    assert [item["disposition"] for item in closeout["resolved_tasks"]] == [
        "self_verified_r_and_d",
        "production_declined_r_and_d",
        "completed",
    ]
    assert terminal.validate_terminal_closeout(closeout) == closeout["closeout_cid"]


def test_rejects_resealed_authority_and_task_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closeout = terminal.build_terminal_closeout(
        predecessor_validation=_install_reconstruction(monkeypatch)
    )
    raised = copy.deepcopy(closeout)
    raised["production_authorized"] = True
    _reseal(raised)
    with pytest.raises(terminal.TerminalCloseoutError, match="authority state"):
        terminal.validate_terminal_closeout(raised)

    wrong_task = copy.deepcopy(closeout)
    wrong_task["resolved_tasks"][0]["disposition"] = "externally_qualified"
    _reseal(wrong_task)
    with pytest.raises(terminal.TerminalCloseoutError, match="task differs"):
        terminal.validate_terminal_closeout(wrong_task)


def test_stable_predecessor_projection_ignores_fresh_replay_ids() -> None:
    first = _predecessor_validation()
    second = copy.deepcopy(first)
    second["benchmark_replay_cid"] = _cid("different-benchmark-replay")
    second["qualification_replay_cid"] = _cid("different-qualification-replay")
    second["validation_cid"] = _cid("different-validation")

    assert terminal._predecessor_authority_cid(
        first
    ) == terminal._predecessor_authority_cid(second)


def test_terminal_schema_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    closeout = terminal.build_terminal_closeout(
        predecessor_validation=_install_reconstruction(monkeypatch)
    )
    closeout["release_notes"] = "open schema injection"
    _reseal(closeout)
    with pytest.raises(terminal.TerminalCloseoutError, match="fields differ"):
        terminal.validate_terminal_closeout(closeout)
