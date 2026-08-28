"""Focused, side-effect-free qualification of the sealed SPAR controls."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_quack_workers_use_birth_bound_bootstrap_not_persisted_tokens() -> None:
    materializer = _materializer()
    config = json.loads(
        (ROOT / "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json").read_text()
    )
    assert config["database_program"]["endpoint_secret_handle"] == (
        "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
    )
    assert not hasattr(materializer, "_publish_handle_token")
    assert not hasattr(materializer, "_LiveQuackTransport")
    assert materializer._SparStateOwnerBootstrapBroker.__doc__


def test_generic_bootstrap_survives_compact_track_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner as runner,
    )

    script = tmp_path / "implementation-supervisor.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    compact = runner.implementation_supervisor_compact_track_spec(
        name="semantic-preserving-autonomous-remodularization-v1",
        script_path=script,
        state_dir=state_dir,
        state_prefix="spar",
    )
    track = runner.expand_implementation_track_lanes(
        compact,
        stamp="test",
        lanes_per_track=3,
    )[0]
    assert track.database_program is None

    captured: dict[str, object] = {}

    def capture_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(runner.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(
        runner,
        "_capture_owned_popen_birth",
        lambda _process, _profile: object(),
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("\0spar-bootstrap-test-" + str(os.getpid()))
    listener.listen(4)
    listener_fd = listener.fileno()
    store_id = "data/agent_supervisor/spar-test/control.duckdb"
    common_args = (
        "--task-source-kind",
        "duckdb",
        "--authority-mode",
        "quack",
        "--state-failover-policy",
        "fail_closed",
        "--endpoint-secret-handle",
        "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "--quack-endpoint",
        "quack:127.0.0.1:46731",
        "--state-store-id",
        store_id,
        "--state-owner-bootstrap-fd",
        str(listener_fd),
        "--state-owner-bootstrap-store-id",
        store_id,
    )
    try:
        process = runner.start_track(
            track,
            repo_root=tmp_path,
            common_args=common_args,
            python_executable=sys.executable,
            output=lambda _message: None,
        )
    finally:
        listener.close()

    command = captured["command"]
    assert isinstance(command, list)
    assert process.pid == os.getpid()
    assert captured["pass_fds"] == (listener_fd,)
    assert command[-2:] == ["--database-owner-session-id", track.name]
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN=" not in " ".join(command)
