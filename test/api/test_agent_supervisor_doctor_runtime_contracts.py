from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.analysis.doctor_runtime_contracts import (
    BASE,
    CONSUMERS,
    PROVIDER,
    inspect_runtime_contracts,
)


def fixture(root, provider):
    for name in (PROVIDER, *CONSUMERS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    bridge = root / "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py"
    bridge.parent.mkdir(parents=True, exist_ok=True)
    bridge.write_text(
        "class DatabasePortalExecutionBridge:\n    def __init__(self, *, max_task_attempts=0): pass\n"
    )
    (root / PROVIDER).write_text(provider)
    (root / (BASE + "database_task_source.py")).write_text(
        "from .intent_repository import SCHEMA\n"
        "class Adapter:\n"
        "    def project(self):\n"
        "        return self._intent.plan_projection()\n"
    )


def test_detects_missing_export_and_method_without_importing_target(tmp_path):
    fixture(tmp_path, "raise RuntimeError('must not execute')\nclass IntentRepository: pass\n")
    report = inspect_runtime_contracts(tmp_path)
    assert report["status"] == "failed"
    assert {item["reason_code"] for item in report["findings"]} == {
        "missing_intent_export",
        "missing_intent_method",
    }
    assert not report["automatic_repair_attempted"]


def test_fixed_surface_passes_without_claiming_behavioral_qualification(tmp_path):
    fixture(
        tmp_path, "SCHEMA = 'one'\nclass IntentRepository:\n    def plan_projection(self): pass\n"
    )
    report = inspect_runtime_contracts(tmp_path)
    assert report["status"] == "passed"
    assert len(report["inputs"]) == len(CONSUMERS) + 2
    assert not report["behavioral_tests_run"]
    assert not report["full_system_qualified"]


def test_missing_provider_fails_closed(tmp_path):
    fixture(tmp_path, "")
    (tmp_path / PROVIDER).unlink()
    assert inspect_runtime_contracts(tmp_path)["status"] == "failed"


def test_bridge_without_attempt_budget_fails_closed(tmp_path):
    fixture(
        tmp_path, "SCHEMA = 'one'\nclass IntentRepository:\n    def plan_projection(self): pass\n"
    )
    bridge = tmp_path / "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py"
    bridge.write_text("class DatabasePortalExecutionBridge:\n    def __init__(self): pass\n")
    report = inspect_runtime_contracts(tmp_path)
    assert report["status"] == "failed"
    assert any(
        f["reason_code"] == "missing_bridge_attempt_budget_contract" for f in report["findings"]
    )
