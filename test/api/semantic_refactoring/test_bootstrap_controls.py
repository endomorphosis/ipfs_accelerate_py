"""Focused, side-effect-free qualification of the sealed SPAR controls."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py"


def _spec():
    definition = importlib.util.spec_from_file_location("spar_control_spec_test", SPEC_PATH)
    assert definition is not None and definition.loader is not None
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    definition.loader.exec_module(module)
    return module


def _materializer():
    path = ROOT / "scripts/materialize_semantic_preserving_remodularization_program.py"
    definition = importlib.util.spec_from_file_location("spar_materializer_test", path)
    assert definition is not None and definition.loader is not None
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    definition.loader.exec_module(module)
    return module


def test_exact_goal_task_and_dependency_population() -> None:
    spec = _spec()
    assert tuple(task.task_id for task in spec.TASKS) == tuple(
        f"SPAR-{index:03d}" for index in range(51)
    )
    assert len(spec.GOALS) == 32
    assert spec.GOALS[0].goal_id == "SPAR-G000"
    assert len(spec.WAVES) == 26
    dependencies = [
        (dependency, task.task_id)
        for task in spec.TASKS
        for dependency in task.dependencies
    ]
    assert len(dependencies) == 109
    assert {task.task_id for task in spec.TASKS if not task.dependencies} == {"SPAR-000"}


def test_operator_task_is_not_worker_schedulable() -> None:
    spec = _spec()
    board = spec.render_taskboard()
    block = board.split("## SPAR-000 ", 1)[1].split("## SPAR-001 ", 1)[0]
    assert "- Completion: operator" in block
    assert "- Is schedulable: false" in block
    assert "- Review only: true" in block
    assert "`SPAR-000` is operator-only" in board


def test_rendered_controls_are_deterministic_and_sealed() -> None:
    spec = _spec()
    assert spec.render(ROOT, check=True)["valid"] is True
    seal = json.loads(
        (ROOT / "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json").read_text()
    )
    claimed = seal.pop("seal_cid")
    assert claimed == spec.identity(seal)
    assert seal["dependency_root_cid"] == spec.identity(
        sorted(
            [
                (dependency, task.task_id)
                for task in spec.TASKS
                for dependency in task.dependencies
            ]
        )
    )


def test_scheduler_authority_and_rollout_are_fail_closed() -> None:
    config = json.loads(
        (ROOT / "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json").read_text()
    )
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["task_source_kind"] == "duckdb"
    assert config["database_program"]["failover_policy"] == "fail_closed"
    assert config["operational_control_plane"]["direct_multi_process_duckdb_file_open_permitted"] is False
    assert config["ducklake_projection_program"]["authority"] is False
    assert config["ducklake_projection_program"]["completion_prerequisite"] is False
    assert config["authority_policy"]["vector_similarity_is_authority"] is False
    assert config["authority_policy"]["worker_self_approval"] is False
    assert config["initial_projection"]["completed_task_ids"] == ["SPAR-000"]
    assert config["initial_projection"]["ready_task_ids"] == ["SPAR-001"]


def test_import_does_not_create_runtime_state() -> None:
    runtime = ROOT / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
    before = runtime.exists()
    _spec()
    assert runtime.exists() is before


def test_quack_handle_token_binding_is_private_and_same_inode(tmp_path: Path) -> None:
    materializer = _materializer()
    source = tmp_path / "typed-state-owner.token"
    source.write_text("a_private_test_token_1234567890", encoding="utf-8")
    source.chmod(0o600)
    target = materializer._publish_handle_token(tmp_path, "handle:spar-test")
    source_stat = os.stat(source, follow_symlinks=False)
    target_stat = os.stat(target, follow_symlinks=False)
    assert (source_stat.st_dev, source_stat.st_ino) == (
        target_stat.st_dev,
        target_stat.st_ino,
    )
    assert target_stat.st_mode & 0o077 == 0
    target.unlink()
