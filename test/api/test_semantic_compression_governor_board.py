"""Fail-closed tests for the Semantic Compression Governor planning board."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "scripts/validate_semantic_compression_governor_board.py"
LAUNCHER_PATH = (
    REPO_ROOT
    / "scripts/ops/agent_supervisor/semantic_compression_governor_scheduler.py"
)
CONTROL_PATHS = (
    "docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md",
    "docs/architecture/semantic_compression_governor.objectives.md",
    "docs/architecture/semantic_compression_governor.todo.md",
    "config/semantic_compression_governor_scheduler.json",
)


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("scg_board_validator_test", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _load_launcher() -> ModuleType:
    name = "scg_scheduler_test"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LAUNCHER = _load_launcher()


@pytest.fixture
def control_root(tmp_path: Path) -> Path:
    for relative in CONTROL_PATHS:
        source = REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Mutation tests exercise a stable initial projection even after the live
    # board has made dependency-closed progress.
    todo = tmp_path / "docs/architecture/semantic_compression_governor.todo.md"
    text = todo.read_text(encoding="utf-8")
    seen = 0

    def initial_status(match: re.Match[str]) -> str:
        nonlocal seen
        status = "completed" if seen == 0 else "todo"
        seen += 1
        return f"- Status: {status}"

    text = re.sub(r"(?m)^- Status: (?:todo|completed)$", initial_status, text)
    assert seen == 49
    todo.write_text(text, encoding="utf-8")
    return tmp_path


def _replace_once(root: Path, relative: str, old: str, new: str) -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, (relative, old, text.count(old))
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _mutate_config(root: Path, callback: Any) -> None:
    path = root / "config/semantic_compression_governor_scheduler.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _errors(root: Path) -> list[str]:
    report = VALIDATOR.validate(root)
    assert report["valid"] is False
    return report["errors"]


def test_current_board_is_closed_valid_and_has_exact_frontier() -> None:
    report = VALIDATOR.validate(REPO_ROOT)
    assert report["schema"].endswith("semantic-compression-governor-board-validation@1")
    assert report["board_namespace"] == "semantic-compression-governor-v1"
    assert report["valid"] is True, report["errors"]
    assert report["task_count"] == 49
    assert report["goal_count"] == 10
    assert report["completed_task_ids"][0] == "SCG-000"
    assert report["terminal_task_id"] == "SCG-048"
    if report["completed_task_ids"] == ["SCG-000"]:
        assert report["ready_task_ids"] == [
            "SCG-001",
            "SCG-002",
            "SCG-003",
            "SCG-004",
        ]


def test_check_all_cli_emits_one_machine_readable_report() -> None:
    run = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--check-all"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert run.stderr == ""
    report = json.loads(run.stdout)
    assert report["valid"] is True
    assert report["completed_task_ids"][0] == "SCG-000"


def test_cli_requires_explicit_check_all() -> None:
    run = subprocess.run(
        [sys.executable, str(VALIDATOR_PATH)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 2
    assert json.loads(run.stdout)["errors"] == ["explicit --check-all is required"]


def test_task_population_and_progress_frontier_are_closed(control_root: Path) -> None:
    todo = "docs/architecture/semantic_compression_governor.todo.md"
    initial_todo = (control_root / todo).read_text(encoding="utf-8")
    _replace_once(
        control_root,
        todo,
        "## SCG-048 Run terminal current-tree qualification and publish the final report",
        "## SCG-049 Run terminal current-tree qualification and publish the final report",
    )
    errors = _errors(control_root)
    assert any("task IDs/order differ" in error for error in errors)

    # A dependency-closed early completion is valid and advances the frontier.
    (control_root / todo).write_text(initial_todo, encoding="utf-8")
    text = (control_root / todo).read_text(encoding="utf-8")
    marker = "## SCG-001 Inventory accelerate"
    prefix, suffix = text.split(marker, 1)
    suffix = suffix.replace("- Status: todo", "- Status: completed", 1)
    (control_root / todo).write_text(prefix + marker + suffix, encoding="utf-8")
    report = VALIDATOR.validate(control_root)
    assert report["valid"] is True, report["errors"]
    assert report["completed_task_ids"] == ["SCG-000", "SCG-001"]
    assert report["ready_task_ids"] == ["SCG-002", "SCG-003", "SCG-004"]

    # A later task may not claim completion before every declared dependency.
    (control_root / todo).write_text(initial_todo, encoding="utf-8")
    text = (control_root / todo).read_text(encoding="utf-8")
    marker = "## SCG-005 Synthesize and test the authority consumption matrix"
    prefix, suffix = text.split(marker, 1)
    suffix = suffix.replace("- Status: todo", "- Status: completed", 1)
    (control_root / todo).write_text(prefix + marker + suffix, encoding="utf-8")
    errors = _errors(control_root)
    assert any("SCG-005 is completed before dependencies" in error for error in errors)

    (control_root / todo).write_text(initial_todo, encoding="utf-8")
    text = initial_todo
    marker = "## SCG-048 Run terminal current-tree qualification"
    prefix, suffix = text.split(marker, 1)
    suffix = suffix.replace("- Status: todo", "- Status: completed", 1)
    (control_root / todo).write_text(prefix + marker + suffix, encoding="utf-8")
    errors = _errors(control_root)
    assert any("SCG-048 is completed before transitive dependencies" in error for error in errors)


def test_task_dependency_cycle_and_terminal_fan_in_fail(control_root: Path) -> None:
    todo = "docs/architecture/semantic_compression_governor.todo.md"
    text = (control_root / todo).read_text(encoding="utf-8")
    marker = "## SCG-001 Inventory accelerate"
    prefix, suffix = text.split(marker, 1)
    suffix = suffix.replace("- Depends on: SCG-000", "- Depends on: SCG-005", 1)
    (control_root / todo).write_text(prefix + marker + suffix, encoding="utf-8")
    errors = _errors(control_root)
    assert any("SCG-001 dependencies differ" in error for error in errors)
    assert any("task dependency graph has a cycle" in error for error in errors)

    shutil.copy2(REPO_ROOT / todo, control_root / todo)
    _replace_once(
        control_root,
        todo,
        "- Depends on: SCG-046, SCG-047\n- Goal id: SCG-G090",
        "- Depends on: SCG-047\n- Goal id: SCG-G090",
    )
    errors = _errors(control_root)
    assert any("SCG-048 terminal fan-in" in error for error in errors)


def test_parallel_waves_are_derived_from_the_current_dependency_graph(
    control_root: Path,
) -> None:
    plan = "docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md"
    _replace_once(
        control_root,
        plan,
        "W3  SCG-006 | SCG-040",
        "W3  SCG-006",
    )
    errors = _errors(control_root)
    assert any("plan parallel waves differ from the closed task DAG" in error for error in errors)


def test_goal_population_model_count_and_lineage_are_closed(control_root: Path) -> None:
    objectives = "docs/architecture/semantic_compression_governor.objectives.md"
    _replace_once(
        control_root,
        objectives,
        '"required_models":23',
        '"required_models":22',
    )
    errors = _errors(control_root)
    assert any("required_models must bind all 23" in error for error in errors)

    shutil.copy2(REPO_ROOT / objectives, control_root / objectives)
    _replace_once(
        control_root,
        objectives,
        "- Depends on: SCG-G030, SCG-G050, SCG-G070",
        "- Depends on: SCG-G030, SCG-G050",
    )
    errors = _errors(control_root)
    assert any("SCG-G080 dependencies differ" in error for error in errors)
    assert any("SCG-044 dependency SCG-037" in error for error in errors)


def test_required_fields_and_namespace_fail_closed(control_root: Path) -> None:
    todo = "docs/architecture/semantic_compression_governor.todo.md"
    _replace_once(control_root, todo, "- Provider role: operator-only\n", "")
    # Provider role is optional; removing a required adjacent field must fail.
    _replace_once(control_root, todo, "- Context budget tokens: 0\n", "")
    _replace_once(control_root, todo, "- LLM context budget bytes: 0\n", "")
    errors = _errors(control_root)
    assert any("SCG-000 is missing task fields" in error for error in errors)

    shutil.copy2(REPO_ROOT / todo, control_root / todo)
    text = (control_root / todo).read_text(encoding="utf-8")
    marker = "## SCG-000 Seal the supervisor-native governor program"
    prefix, suffix = text.split(marker, 1)
    suffix = suffix.replace(
        "- Board namespace: semantic-compression-governor-v1",
        "- Board namespace: attacker-board-v1",
        1,
    )
    (control_root / todo).write_text(prefix + marker + suffix, encoding="utf-8")
    errors = _errors(control_root)
    assert any("SCG-000 has wrong board namespace" in error for error in errors)


def test_plan_safety_claim_and_named_artifact_population_are_required(
    control_root: Path,
) -> None:
    plan = "docs/architecture/SEMANTIC_COMPRESSION_GOVERNOR_PLAN.md"
    _replace_once(
        control_root,
        plan,
        "treat model agreement as proof",
        "treat model agreement as advisory evidence",
    )
    errors = _errors(control_root)
    assert any("missing safety claim" in error and "model agreement" in error for error in errors)

    shutil.copy2(REPO_ROOT / plan, control_root / plan)
    _replace_once(
        control_root,
        plan,
        "\nGovernorRunReceipt\n",
        "\nGovernorExecutionLog\n",
    )
    errors = _errors(control_root)
    assert any("GovernorRunReceipt" in error for error in errors)


def test_source_pins_and_config_duplicate_keys_fail_closed(control_root: Path) -> None:
    def alter_pin(payload: dict[str, Any]) -> None:
        payload["source_binding"]["ipfs_datasets_planning_revision"] = "0" * 40

    _mutate_config(control_root, alter_pin)
    errors = _errors(control_root)
    assert any("source binding differs" in error for error in errors)

    shutil.copy2(
        REPO_ROOT / "config/semantic_compression_governor_scheduler.json",
        control_root / "config/semantic_compression_governor_scheduler.json",
    )
    config_path = control_root / "config/semantic_compression_governor_scheduler.json"
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace("{", '{"schema":"forged",', 1), encoding="utf-8"
    )
    errors = _errors(control_root)
    assert any("duplicate JSON key 'schema'" in error for error in errors)


def test_protected_control_set_and_worker_ownership_fail_closed(control_root: Path) -> None:
    def remove_control(payload: dict[str, Any]) -> None:
        payload["protected_paths"].remove(
            "docs/architecture/semantic_compression_governor.todo.md"
        )

    _mutate_config(control_root, remove_control)
    errors = _errors(control_root)
    assert any("protected controls differ" in error for error in errors)

    shutil.copy2(
        REPO_ROOT / "config/semantic_compression_governor_scheduler.json",
        control_root / "config/semantic_compression_governor_scheduler.json",
    )
    todo = "docs/architecture/semantic_compression_governor.todo.md"
    text = (control_root / todo).read_text(encoding="utf-8")
    marker = "## SCG-001 Inventory accelerate"
    prefix, suffix = text.split(marker, 1)
    old = (
        "docs/architecture/semantic_compression_governor_inventory/accelerate.json, "
        "docs/architecture/semantic_compression_governor_inventory/accelerate.md"
    )
    new = "docs/architecture/semantic_compression_governor.todo.md"
    assert suffix.count(old) == 2
    suffix = suffix.replace(old, new, 2)
    (control_root / todo).write_text(prefix + marker + suffix, encoding="utf-8")
    errors = _errors(control_root)
    assert any("SCG-001 owns protected controls" in error for error in errors)


def test_initial_projection_and_lane_parity_are_bound(control_root: Path) -> None:
    def alter_projection(payload: dict[str, Any]) -> None:
        payload["initial_projection"]["ready_task_ids"] = ["SCG-001"]

    _mutate_config(control_root, alter_projection)
    errors = _errors(control_root)
    assert any("initial projection differs" in error for error in errors)

    shutil.copy2(
        REPO_ROOT / "config/semantic_compression_governor_scheduler.json",
        control_root / "config/semantic_compression_governor_scheduler.json",
    )

    def alter_lane(payload: dict[str, Any]) -> None:
        payload["lanes"][0]["initial_task_ids"] = ["SCG-004"]

    _mutate_config(control_root, alter_lane)
    errors = _errors(control_root)
    assert any("lane mapping differs" in error for error in errors)
    assert any("lane parity mismatch for SCG-004" in error for error in errors)


def test_safety_policies_and_timeout_ceiling_cannot_be_weakened(control_root: Path) -> None:
    def weaken_privacy(payload: dict[str, Any]) -> None:
        payload["capability_policy"][
            "unapproved_external_expanded_shadow_allowed"
        ] = True

    _mutate_config(control_root, weaken_privacy)
    errors = _errors(control_root)
    assert any("capability safety claims differ" in error for error in errors)

    shutil.copy2(
        REPO_ROOT / "config/semantic_compression_governor_scheduler.json",
        control_root / "config/semantic_compression_governor_scheduler.json",
    )

    def lower_timeout(payload: dict[str, Any]) -> None:
        payload["implementation_max_timeout_seconds"] = 7200

    _mutate_config(control_root, lower_timeout)
    errors = _errors(control_root)
    assert any("timeout exceeds scheduler hard maximum" in error for error in errors)


def test_unsafe_paths_and_unknown_scheduler_fields_are_rejected(control_root: Path) -> None:
    todo = "docs/architecture/semantic_compression_governor.todo.md"
    text = (control_root / todo).read_text(encoding="utf-8")
    marker = "## SCG-001 Inventory accelerate"
    prefix, suffix = text.split(marker, 1)
    old = (
        "docs/architecture/semantic_compression_governor_inventory/accelerate.json, "
        "docs/architecture/semantic_compression_governor_inventory/accelerate.md"
    )
    assert suffix.count(old) == 2
    suffix = suffix.replace(old, "../escape", 2)
    (control_root / todo).write_text(prefix + marker + suffix, encoding="utf-8")
    errors = _errors(control_root)
    assert any("unsafe path '../escape'" in error for error in errors)

    shutil.copy2(REPO_ROOT / todo, control_root / todo)

    def add_field(payload: dict[str, Any]) -> None:
        payload["silently_relax_assurance"] = True

    _mutate_config(control_root, add_field)
    errors = _errors(control_root)
    assert any("scheduler has unknown fields" in error for error in errors)


def test_validator_has_no_product_import_or_network_dependency() -> None:
    source = VALIDATOR_PATH.read_text(encoding="utf-8")
    assert "from ipfs_accelerate_py" not in source
    assert "import ipfs_accelerate_py" not in source
    assert "subprocess" not in source
    assert "requests" not in source


def test_scheduler_launch_plan_semantically_binds_three_strict_lanes() -> None:
    board = LAUNCHER.load_board(
        REPO_ROOT / "config/semantic_compression_governor_scheduler.json",
        repo_root=REPO_ROOT,
    )
    plan = LAUNCHER.launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=60,
        stamp="scg-test",
        expected_task_count=49,
    )
    mapping = LAUNCHER._validate_launch_plan(board, plan)
    assert mapping == {
        "common_arg_count": 65,
        "lane_count": 3,
        "strict_shards": [
            {"count": 3, "index": 0},
            {"count": 3, "index": 1},
            {"count": 3, "index": 2},
        ],
    }
    assert plan["expected_task_count"] == 49
    assert plan["strict_task_sharding"] is True
    assert plan["runtime_root"].endswith(
        "data/agent_supervisor/semantic_compression_governor/run-scg-v1"
    )


def test_scheduler_requires_explicit_implementation_authority() -> None:
    board = LAUNCHER.load_board(
        REPO_ROOT / "config/semantic_compression_governor_scheduler.json",
        repo_root=REPO_ROOT,
    )
    with pytest.raises(LAUNCHER.SCGSchedulerError, match="explicit --implement"):
        LAUNCHER.launch_plan(
            board,
            implement=False,
            foreground=False,
            duration_seconds=60,
        )
