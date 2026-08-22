from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / "scripts/validate_agent_supervisor_residual_intelligence_board.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("vrif_board_validator", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_board_is_complete_acyclic_and_parallel_ready() -> None:
    module = _validator_module()
    report = module.validate_program()
    assert report["valid"], report["errors"]
    assert report["task_count"] == 33
    assert report["goal_count"] == 9
    assert report["root_goal"] == "VRIF-G000"
    assert report["ready_frontier"] == [
        "VRIF-009",
        "VRIF-010",
        "VRIF-011",
        "VRIF-012",
    ]


def test_validator_json_surface_is_machine_readable() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(completed.stdout)
    assert report["valid"] is True
    assert report["program_id"] == ("agent-supervisor-verified-residual-intelligence-foundry-v1")


def test_validator_check_all_surface_matches_scheduler_contract() -> None:
    completed = subprocess.run(
        [sys.executable, str(VALIDATOR), "--check-all"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(completed.stdout)
    assert report["valid"] is True
    assert report["ready_frontier"] == [
        "VRIF-009",
        "VRIF-010",
        "VRIF-011",
        "VRIF-012",
    ]


def test_parser_rejects_duplicate_bold_fields() -> None:
    module = _validator_module()
    records, errors = module._parse_records(
        "\n".join(
            (
                "## VRIF-000 Example",
                "",
                "- **Status:** todo",
                "- **Status:** completed",
            )
        ),
        module._TASK_HEADER,
    )
    assert len(records) == 1
    assert errors == ["VRIF-000: duplicate field 'status'"]


def test_declared_dependency_contract_has_no_unknowns_or_cycle() -> None:
    module = _validator_module()
    known = set(module.TASK_IDS)
    assert all(set(dependencies) <= known for dependencies in module.EXPECTED_DEPENDENCIES.values())
    acyclic, cycle = module._acyclic(module.EXPECTED_DEPENDENCIES)
    assert acyclic is True
    assert cycle == []
