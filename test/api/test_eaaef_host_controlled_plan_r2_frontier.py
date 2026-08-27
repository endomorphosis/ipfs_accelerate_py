"""Host-controlled Plan-R2 frontier admit without live launch."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_eaaef_host_admission_supervisor.py"
RECEIPT = (
    ROOT
    / "docs/architecture/external_agent_autonomous_execution_fabric"
    / "receipts"
    / "host_admission"
    / "host_controlled_plan_r2_frontier.json"
)


def _load_script(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _FakePage:
    def __init__(self, tasks: list[object]) -> None:
        self.tasks = tasks


class _FakeCAS:
    def __init__(self, task: SimpleNamespace, changed: bool) -> None:
        self.task = task
        self.changed = changed
        self.receipt_cid = "cid:receipt"


class _FakeSource:
    def __init__(self, tasks: list[SimpleNamespace]) -> None:
        self._tasks = {task.task_alias: task for task in tasks}

    def list_tasks(self, limit: int = 1000) -> _FakePage:
        del limit
        return _FakePage(list(self._tasks.values()))

    def compare_and_set_status(
        self,
        task_cid: str,
        expected_revision: int,
        status: str,
        receipt: dict,
    ) -> _FakeCAS:
        del expected_revision, receipt
        for task in self._tasks.values():
            if task.task_cid == task_cid:
                task.status = status
                task.revision += 1
                return _FakeCAS(task, True)
        raise KeyError(task_cid)


def _task(alias: str, status: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_alias=alias,
        task_cid=f"cid:{alias}",
        status=status,
        revision=1,
    )


def test_admit_held_plan_r2_waits_for_bootstrap(tmp_path: Path) -> None:
    runner = _load_script(RUNNER, "tested_eaaef_plan_r2_admit_wait")
    runner.PLAN_R2_APPLY_RECEIPT = tmp_path / "plan_r2.json"
    source = _FakeSource(
        [
            _task("EAAEF-000", "completed"),
            _task("EAAEF-009", "ready"),
            _task("EAAEF-191", "completed"),
            _task("EAAEF-141", "blocked"),
        ]
    )
    payload = runner.admit_held_plan_r2_frontier(source)
    assert payload["applied"] is False
    assert payload["reason"] == "bootstrap_incomplete"
    assert payload["live_launch_allowed"] is False
    assert payload["configured_board_launch"] is False
    assert source._tasks["EAAEF-141"].status == "blocked"


def test_admit_held_plan_r2_promotes_blocked_after_bootstrap(tmp_path: Path) -> None:
    runner = _load_script(RUNNER, "tested_eaaef_plan_r2_admit_apply")
    runner.PLAN_R2_APPLY_RECEIPT = tmp_path / "plan_r2.json"
    bootstrap = [f"EAAEF-{number:03d}" for number in range(0, 10)] + [
        f"EAAEF-{number}" for number in range(180, 192)
    ]
    tasks = [_task(alias, "completed") for alias in bootstrap]
    tasks.extend(
        [
            _task("EAAEF-141", "blocked"),
            _task("EAAEF-144", "blocked"),
            _task("EAAEF-145", "ready"),
        ]
    )
    source = _FakeSource(tasks)
    payload = runner.admit_held_plan_r2_frontier(source)
    assert payload["applied"] is True
    assert payload["reason"] == "bootstrap_complete"
    assert payload["live_launch_allowed"] is False
    assert payload["process_started"] is False
    assert payload["configured_board_launch"] is False
    assert set(payload["admitted"]) == {"EAAEF-141", "EAAEF-144"}
    assert source._tasks["EAAEF-141"].status == "ready"
    assert source._tasks["EAAEF-144"].status == "ready"
    assert source._tasks["EAAEF-145"].status == "ready"


def test_host_controlled_plan_r2_receipt_forbids_live_launch() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["schema"] == (
        "ipfs_accelerate_py/agent-supervisor/"
        "eaaef-host-controlled-plan-r2-frontier-admit@1"
    )
    assert payload["applied"] is True
    assert payload["configured_board_launch"] is False
    assert payload["live_launch_allowed"] is False
    assert payload["live_multi_supervisor"] is False
    assert payload["process_started"] is False
