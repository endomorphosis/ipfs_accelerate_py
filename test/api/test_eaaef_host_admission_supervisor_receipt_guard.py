"""Hermetic receipt gates for the offline EAAEF host-admission runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.validation import eaaef_host_admission

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts/run_eaaef_host_admission_supervisor.py"
IDENTITY = {
    "source_head": "1" * 40,
    "source_tree": "2" * 40,
    "board_namespace": "external-agent-autonomous-execution-fabric-v1",
    "board_cid": "sha256:" + "3" * 64,
}


def _load_runner(name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, RUNNER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _FakeSource:
    def __init__(self, tasks: dict[str, str]) -> None:
        self.tasks = {
            alias: SimpleNamespace(
                task_cid=f"cid:{alias}",
                revision=1,
                status=status,
            )
            for alias, status in tasks.items()
        }
        self.cas_calls: list[dict[str, Any]] = []
        self.validation_calls: list[dict[str, Any]] = []
        self.evidence_calls: list[dict[str, Any]] = []

    def get_task(self, alias: str) -> Any:
        return self.tasks.get(alias)

    def record_validation_result(self, **kwargs: Any) -> None:
        self.validation_calls.append(kwargs)

    def record_evidence(self, **kwargs: Any) -> None:
        self.evidence_calls.append(kwargs)

    def compare_and_set_status(
        self,
        task_cid: str,
        revision: int,
        status: str,
        receipt: dict[str, Any],
        *,
        evidence_digests: list[str] | None = None,
    ) -> Any:
        alias = task_cid.removeprefix("cid:")
        task = self.tasks[alias]
        assert task.revision == revision
        task.status = status
        task.revision += 1
        self.cas_calls.append(
            {
                "task_id": alias,
                "status": status,
                "receipt": receipt,
                "evidence_digests": evidence_digests,
            }
        )
        return SimpleNamespace(
            task=task,
            changed=True,
            receipt_cid=f"receipt:{alias}",
        )


def _receipt_contract(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    return receipt_dir, {
        f"EAAEF-{number}": f"eaaef-{number}.json" for number in range(180, 192)
    }


def _write_receipt(receipt_dir: Path, filename: str, decision: str) -> Path:
    path = receipt_dir / filename
    path.write_text(json.dumps({"decision": decision}) + "\n", encoding="utf-8")
    return path


def test_canonical_verifier_receives_exact_current_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("tested_eaaef_verifier_api")
    observed: dict[str, Any] = {}

    def verify(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {"valid": True, "decision": "inventory", "blockers": []}

    monkeypatch.setattr(
        eaaef_host_admission,
        "verify_host_admission_task_receipt",
        verify,
        raising=False,
    )

    result = runner._verify_host_admission_task_receipt(
        task_id="EAAEF-180",
        receipt_dir=tmp_path,
        expected_identity=IDENTITY,
    )

    assert result == {"valid": True, "decision": "inventory", "blockers": []}
    assert observed == {
        "task_id": "EAAEF-180",
        "receipt_dir": tmp_path,
        "expected_source_head": IDENTITY["source_head"],
        "expected_source_tree": IDENTITY["source_tree"],
        "expected_board_namespace": IDENTITY["board_namespace"],
        "expected_board_cid": IDENTITY["board_cid"],
    }


@pytest.mark.parametrize(
    ("alias", "valid", "decision", "create_receipt"),
    [
        ("EAAEF-180", False, "inventory", True),
        ("EAAEF-181", False, "", False),
        ("EAAEF-182", True, "typed_missing", True),
        ("EAAEF-183", True, "no_go", True),
        ("EAAEF-191", True, "no_go", True),
    ],
)
def test_stale_missing_typed_missing_and_no_go_cannot_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    valid: bool,
    decision: str,
    create_receipt: bool,
) -> None:
    runner = _load_runner(f"tested_eaaef_rejected_{alias}")
    receipt_dir, files = _receipt_contract(tmp_path)
    if create_receipt:
        _write_receipt(receipt_dir, files[alias], decision)
    source = _FakeSource({alias: "todo"})
    monkeypatch.setattr(runner, "_receipt_contract", lambda: (receipt_dir, files))
    monkeypatch.setattr(
        runner,
        "_verify_host_admission_task_receipt",
        lambda **_kwargs: {
            "valid": valid,
            "decision": decision,
            "blockers": [] if valid else ["stale or missing receipt"],
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_argv",
        lambda *_args, **_kwargs: pytest.fail("invalid receipt reached validation"),
    )

    result = runner._complete_s_task(source, alias, IDENTITY)

    assert result["status"].startswith("waiting_")
    assert result["decision"] == decision
    assert source.validation_calls == []
    assert source.evidence_calls == []
    assert source.cas_calls == []


def test_every_completed_host_task_is_reopened_and_receipt_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("tested_eaaef_reopen_all")
    receipt_dir, files = _receipt_contract(tmp_path)
    aliases = sorted(files)
    for alias in aliases[1:]:
        _write_receipt(receipt_dir, files[alias], "typed_missing")
    source = _FakeSource({alias: "completed" for alias in aliases})
    monkeypatch.setattr(runner, "_receipt_contract", lambda: (receipt_dir, files))
    monkeypatch.setattr(
        runner,
        "_verify_host_admission_task_receipt",
        lambda **kwargs: {
            "valid": False,
            "decision": (
                ""
                if kwargs["task_id"] == aliases[0]
                else "no_go" if kwargs["task_id"] == "EAAEF-191" else "typed_missing"
            ),
            "blockers": ["receipt is stale for current source"],
        },
    )

    reopened = runner._reopen_invalid_host_admission_tasks(source, IDENTITY)

    assert [item["task_id"] for item in reopened] == aliases
    assert all(item["receipt_quarantined"] is True for item in reopened)
    assert all(source.tasks[alias].status == "todo" for alias in aliases)
    assert len(source.cas_calls) == 12
    for call in source.cas_calls:
        assert call["status"] == "todo"
        assert call["receipt"]["receipt_quarantined"] is True
        assert call["receipt"]["receipt_blockers"]


def test_current_completion_decisions_remain_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("tested_eaaef_valid_completed")
    receipt_dir, files = _receipt_contract(tmp_path)
    decisions = {
        "EAAEF-180": "inventory",
        "EAAEF-181": "bound_unadmitted",
        **{f"EAAEF-{number}": "admitted" for number in range(182, 192)},
    }
    for alias, decision in decisions.items():
        _write_receipt(receipt_dir, files[alias], decision)
    source = _FakeSource({alias: "completed" for alias in decisions})
    monkeypatch.setattr(runner, "_receipt_contract", lambda: (receipt_dir, files))
    monkeypatch.setattr(
        runner,
        "_verify_host_admission_task_receipt",
        lambda **kwargs: {
            "valid": True,
            "decision": decisions[kwargs["task_id"]],
            "blockers": [],
        },
    )

    assert runner._reopen_invalid_host_admission_tasks(source, IDENTITY) == []
    assert source.cas_calls == []


def test_completion_rechecks_and_binds_receipt_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("tested_eaaef_completion_binding")
    receipt_dir, files = _receipt_contract(tmp_path)
    alias = "EAAEF-182"
    _write_receipt(receipt_dir, files[alias], "admitted")
    source = _FakeSource({alias: "todo"})
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_receipt_contract", lambda: (receipt_dir, files))
    monkeypatch.setattr(
        runner,
        "_verify_host_admission_task_receipt",
        lambda **_kwargs: {
            "valid": True,
            "decision": "admitted",
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        runner,
        "_run_argv",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="passed", stderr=""
        ),
    )

    result = runner._complete_s_task(source, alias, IDENTITY)

    assert result["status"] == "completed"
    assert len(source.validation_calls) == 1
    assert len(source.evidence_calls) == 1
    assert source.evidence_calls[0]["body"]["source_head"] == IDENTITY["source_head"]
    assert source.evidence_calls[0]["body"]["decision"] == "admitted"
    assert len(source.cas_calls) == 1
    digests = source.cas_calls[0]["evidence_digests"]
    assert digests is not None and len(digests) == 2
    assert digests[1] == source.evidence_calls[0]["digest"]


def test_receipt_mutation_during_validation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("tested_eaaef_receipt_toctou")
    receipt_dir, files = _receipt_contract(tmp_path)
    alias = "EAAEF-182"
    receipt_path = _write_receipt(receipt_dir, files[alias], "admitted")
    source = _FakeSource({alias: "todo"})
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_receipt_contract", lambda: (receipt_dir, files))
    monkeypatch.setattr(
        runner,
        "_verify_host_admission_task_receipt",
        lambda **_kwargs: {
            "valid": True,
            "decision": "admitted",
            "blockers": [],
        },
    )

    def mutate_receipt(*args: Any, **kwargs: Any) -> Any:
        receipt_path.write_text('{"decision":"revoked"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="passed", stderr=""
        )

    monkeypatch.setattr(runner, "_run_argv", mutate_receipt)

    result = runner._complete_s_task(source, alias, IDENTITY)

    assert result["status"] == "receipt_changed_or_revoked"
    assert any("changed during validation" in item for item in result["blockers"])
    assert source.evidence_calls == []
    assert source.cas_calls == []


def test_plan_r2_gate_uses_current_receipt_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("tested_eaaef_plan_r2_guard")
    receipt_dir, files = _receipt_contract(tmp_path)
    _write_receipt(receipt_dir, files["EAAEF-190"], "admitted")
    monkeypatch.setattr(runner, "_receipt_contract", lambda: (receipt_dir, files))
    monkeypatch.setattr(
        runner,
        "_verify_host_admission_task_receipt",
        lambda **_kwargs: {
            "valid": False,
            "decision": "admitted",
            "blockers": ["receipt source is stale"],
        },
    )

    assert runner._plan_r2_remote_owner_admitted(IDENTITY) is False
