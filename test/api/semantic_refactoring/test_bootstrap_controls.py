"""Focused, side-effect-free qualification of the sealed SPAR controls."""

from __future__ import annotations

import errno
import hashlib
import importlib
import importlib.util
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from dataclasses import replace
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


def _dependency_validator():
    path = ROOT / "scripts/validate_semantic_preserving_remodularization_dependencies.py"
    definition = importlib.util.spec_from_file_location(
        "spar_dependency_validator_test",
        path,
    )
    assert definition is not None and definition.loader is not None
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    definition.loader.exec_module(module)
    return module


def _configured_board_scheduler():
    return importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.runtime."
        "configured_board_scheduler"
    )


def _nested_source_fixture(
    tmp_path: Path,
    *,
    relative: str = "ipfs_datasets_py",
) -> dict[str, object]:
    parent = tmp_path / "parent"
    nested = parent / relative
    nested.mkdir(parents=True)

    def git(repository: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    for repository in (parent, nested):
        git(repository, "init", "-q", "-b", "main")
        git(repository, "config", "user.name", "SPAR Source Test")
        git(
            repository,
            "config",
            "user.email",
            "spar-source@example.invalid",
        )
    source = nested / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    git(nested, "add", "source.py")
    git(nested, "commit", "-qm", "planning source")
    planning_commit = git(nested, "rev-parse", "HEAD")
    planning_tree = git(nested, "rev-parse", "HEAD^{tree}")
    git(
        parent,
        "update-index",
        "--add",
        "--cacheinfo",
        "160000",
        planning_commit,
        relative,
    )
    git(parent, "commit", "-qm", "planning gitlink")
    outer_planning_commit = git(parent, "rev-parse", "HEAD")

    source.write_text("VALUE = 2\n", encoding="utf-8")
    git(nested, "add", "source.py")
    git(nested, "commit", "-qm", "accepted descendant")
    current_commit = git(nested, "rev-parse", "HEAD")
    git(
        parent,
        "update-index",
        "--cacheinfo",
        "160000",
        current_commit,
        relative,
    )
    git(parent, "commit", "-qm", "accepted current gitlink")
    return {
        "parent": parent,
        "nested": nested,
        "git": git,
        "planning_commit": planning_commit,
        "planning_tree": planning_tree,
        "outer_planning_commit": outer_planning_commit,
        "current_commit": current_commit,
        "relative": relative,
    }


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
    assert {
        "pyproject.toml",
        "ipfs_accelerate_py/agent_supervisor/runtime/"
        "configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
        "pre_implementation_kernel.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/"
        "pre_implementation_provider_gate.py",
        "ipfs_accelerate_py/agent_supervisor/validation/"
        "project_dependency_preflight.py",
    }.issubset(seal["bootstrap_runtime_file_sha256"])
    scheduler = json.loads(
        (
            ROOT
            / "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"
        ).read_text()
    )
    assert scheduler["provider"]["merge_resolver_mode"] == (
        "disabled_until_sealed_residual"
    )


def test_dependency_binding_separates_planning_and_current_gitlinks(
    tmp_path: Path,
) -> None:
    validator = _dependency_validator()
    fixture = _nested_source_fixture(tmp_path)
    parent = fixture["parent"]
    nested = fixture["nested"]
    git = fixture["git"]
    assert isinstance(parent, Path)
    assert isinstance(nested, Path)
    assert callable(git)
    binding = {
        "commit": fixture["planning_commit"],
        "tree": fixture["planning_tree"],
    }

    passed, detail = validator._validate_nested_source_binding(
        root=parent,
        outer_planning_commit=str(fixture["outer_planning_commit"]),
        relative="ipfs_datasets_py",
        binding=binding,
    )
    assert passed is True
    assert detail["planning_gitlink"] is True
    assert detail["planning_is_ancestor"] is True
    assert detail["current_gitlink"] is True
    assert detail["current_head"] == fixture["current_commit"]

    wrong_tree, wrong_tree_detail = (
        validator._validate_nested_source_binding(
            root=parent,
            outer_planning_commit=str(fixture["outer_planning_commit"]),
            relative="ipfs_datasets_py",
            binding={**binding, "tree": "0" * 40},
        )
    )
    assert wrong_tree is False
    assert wrong_tree_detail["planning_tree_exact"] is False

    wrong_planning_link, wrong_planning_detail = (
        validator._validate_nested_source_binding(
            root=parent,
            outer_planning_commit=git(parent, "rev-parse", "HEAD"),
            relative="ipfs_datasets_py",
            binding=binding,
        )
    )
    assert wrong_planning_link is False
    assert wrong_planning_detail["planning_gitlink"] is False

    git(nested, "checkout", "-q", "--detach", str(fixture["planning_commit"]))
    mismatched_checkout, mismatch_detail = (
        validator._validate_nested_source_binding(
            root=parent,
            outer_planning_commit=str(fixture["outer_planning_commit"]),
            relative="ipfs_datasets_py",
            binding=binding,
        )
    )
    assert mismatched_checkout is False
    assert mismatch_detail["current_gitlink"] is False
    git(nested, "checkout", "-q", "--detach", str(fixture["current_commit"]))

    (nested / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty, dirty_detail = validator._validate_nested_source_binding(
        root=parent,
        outer_planning_commit=str(fixture["outer_planning_commit"]),
        relative="ipfs_datasets_py",
        binding=binding,
    )
    assert dirty is False
    assert dirty_detail["clean"] is False


def test_dependency_binding_requires_current_outer_planning_ancestry(
    tmp_path: Path,
) -> None:
    validator = _dependency_validator()
    fixture = _nested_source_fixture(tmp_path)
    parent = fixture["parent"]
    git = fixture["git"]
    assert isinstance(parent, Path)
    assert callable(git)
    planning_commit = str(fixture["outer_planning_commit"])
    binding = {
        "commit": planning_commit,
        "tree": git(parent, "rev-parse", f"{planning_commit}^{{tree}}"),
    }

    passed, detail = validator._validate_outer_source_binding(
        root=parent,
        binding=binding,
    )
    assert passed is True
    assert detail["planning_is_ancestor"] is True

    git(parent, "checkout", "-q", "--orphan", "unrelated")
    git(parent, "commit", "--allow-empty", "-qm", "unrelated root")
    rejected, rejected_detail = validator._validate_outer_source_binding(
        root=parent,
        binding=binding,
    )
    assert rejected is False
    assert rejected_detail["planning_is_ancestor"] is False


def test_materializer_source_forest_admits_only_clean_descendant_gitlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    fixture = _nested_source_fixture(tmp_path)
    parent = fixture["parent"]
    nested = fixture["nested"]
    git = fixture["git"]
    assert isinstance(parent, Path)
    assert isinstance(nested, Path)
    assert callable(git)
    monkeypatch.setattr(materializer, "ROOT", parent)
    config = {
        "source_binding": {
            "datasets_submodule_path": "ipfs_datasets_py",
            "datasets_planning_revision": fixture["planning_commit"],
        }
    }

    forest = materializer._source_forest(
        config,
        head=git(parent, "rev-parse", "HEAD"),
    )
    assert forest["nested_repositories"] == [
        {
            "repository": "ipfs_datasets",
            "path": "ipfs_datasets_py",
            "head": fixture["current_commit"],
            "tree": git(nested, "rev-parse", "HEAD^{tree}"),
            "planning_revision": fixture["planning_commit"],
            "planning_revision_is_ancestor": True,
            "access": "read_only_contract_audit",
        }
    ]

    git(nested, "checkout", "-q", "--detach", str(fixture["planning_commit"]))
    with pytest.raises(materializer.OperatorError, match="gitlink differs"):
        materializer._source_forest(
            config,
            head=git(parent, "rev-parse", "HEAD"),
        )


def test_materializer_source_forest_keeps_mcpplusplus_exactly_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    relative = "ipfs_accelerate_py/mcplusplus"
    fixture = _nested_source_fixture(tmp_path, relative=relative)
    parent = fixture["parent"]
    git = fixture["git"]
    assert isinstance(parent, Path)
    assert callable(git)
    monkeypatch.setattr(materializer, "ROOT", parent)
    config = {
        "source_binding": {
            "mcp_plus_plus_submodule_path": relative,
            "mcp_plus_plus_planning_revision": fixture["planning_commit"],
        }
    }

    with pytest.raises(materializer.OperatorError, match="exact read-only seal"):
        materializer._source_forest(
            config,
            head=git(parent, "rev-parse", "HEAD"),
        )

    validator = _dependency_validator()
    passed, detail = validator._validate_nested_source_binding(
        root=parent,
        outer_planning_commit=str(fixture["outer_planning_commit"]),
        relative=relative,
        binding={
            "commit": fixture["planning_commit"],
            "tree": fixture["planning_tree"],
        },
        permit_descendants=False,
    )
    assert passed is False
    assert isinstance(detail, dict)
    assert detail["advancement_policy"] == "exact_read_only_pin"
    assert detail["current_is_exact_planning_revision"] is False


def test_launch_source_forest_receipts_are_content_addressed_and_current(
    tmp_path: Path,
) -> None:
    materializer = _materializer()
    receipt_dir = tmp_path / "launch" / "source-forest"
    paths = {
        "launch_source_forest_dir": receipt_dir,
        "launch_source_forest_current": receipt_dir / "current.json",
    }
    forest = {
        "source_head": "a" * 40,
        "nested_repositories": [],
        "cross_repository_writes": False,
    }
    forest["source_forest_root"] = materializer._identity(forest)

    first = materializer._record_launch_source_forest(
        paths,
        source_head="a" * 40,
        repository_tree="b" * 40,
        source_forest=forest,
    )
    replay = materializer._record_launch_source_forest(
        paths,
        source_head="a" * 40,
        repository_tree="b" * 40,
        source_forest=forest,
    )

    assert replay == first
    assert json.loads(
        paths["launch_source_forest_current"].read_text(encoding="utf-8")
    )["receipt_id"] == first["receipt_id"]
    immutable = [
        path
        for path in receipt_dir.glob("*.json")
        if path.name != "current.json"
    ]
    assert len(immutable) == 1
    assert json.loads(immutable[0].read_text(encoding="utf-8"))[
        "source_forest_root"
    ] == forest["source_forest_root"]


def test_scheduler_authority_and_rollout_are_fail_closed() -> None:
    config_path = (
        ROOT
        / "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json"
    )
    config = json.loads(config_path.read_text())
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
    assert config["launch_source_amendment_policy"] == {
        "append_mode": "exact_plan_revision_cas",
        "attempt_source_policy": "exact_launch_forest_and_worktree_preimage",
        "authoritative_store": "DuckDB/PlanRevisionRepository@1 over Quack",
        "cli_json_is_authority": False,
        "completion_policy": "current_tree_requalification_required",
        "filesystem_projection_is_authority": False,
        "required": True,
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "launch-source-amendment@1"
        ),
        "task_history_policy": "preserve_immutable_task_cids_and_receipts",
    }

    scheduler = _configured_board_scheduler()
    board = scheduler.load_configured_board(config_path, repo_root=ROOT)
    common_args = scheduler.configured_board_common_args(
        board,
        implement=True,
    )
    assert common_args.count("--require-launch-source-amendment") == 1
    assert "--launch-source-amendment-json" not in common_args


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


def test_quack_lane_runtime_directories_are_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    monkeypatch.setattr(materializer, "ROOT", tmp_path)
    runtime = tmp_path / "runtime"
    lane = runtime / "state" / "lane-0"
    lane.mkdir(parents=True, mode=0o775)
    lane.chmod(0o775)
    board = SimpleNamespace(
        payload={
            "runtime_paths": {
                "root": "runtime",
                "state": "runtime/state",
                "merge_queue": "runtime/merge-queue",
                "logs": "runtime/logs",
                "worktrees": "runtime/worktrees",
            }
        },
        max_lanes=2,
        resolved_database_program=lambda: SimpleNamespace(
            event_store_path="runtime/events",
            runtime_registry_path="runtime/registry",
        ),
    )
    paths = {
        "runtime": runtime,
        "database": runtime / "control.duckdb",
        "owner": runtime / "quack-owner",
        "bootstrap_receipt": runtime / "evidence/bootstrap/receipt.json",
        "ducklake_catalog": runtime / "ducklake/catalog.duckdb",
        "ducklake_data": runtime / "ducklake/data",
        "launch_source_forest_dir": (
            runtime / "evidence/launch/source-forest"
        ),
        "launch_source_amendment_dir": (
            runtime / "evidence/launch/source-amendment"
        ),
    }

    materializer._harden_runtime_directories(board, paths)

    expected = (
        runtime,
        runtime / "state/lane-0",
        runtime / "state/lane-1",
        runtime / "events",
        runtime / "registry",
        runtime / "merge-queue/pending",
        runtime / "evidence/bootstrap",
        runtime / "ducklake/data",
    )
    for directory in expected:
        metadata = directory.stat()
        assert metadata.st_uid == os.geteuid()
        assert stat.S_IMODE(metadata.st_mode) == 0o700


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


def _route_restart_fixture():
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        TaskRecord,
        TaskSourceSnapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
        TaskExecutionRoutePolicy,
    )

    plan_root = "cid:plan-root"
    repository_tree = "tree:repository"
    operator_task = TaskRecord(
        task_cid="cid:spar-000",
        task_alias="SPAR-000",
        goal_cid="cid:goal-000",
        ordinal=0,
        status="completed",
        revision=2,
        body={"completion": "operator", "status": "completed"},
        plan_cid=plan_root,
    )
    implementation_task = TaskRecord(
        task_cid="cid:spar-001",
        task_alias="SPAR-001",
        goal_cid="cid:goal-010",
        ordinal=1,
        status="todo",
        revision=1,
        body={"completion": "worker", "status": "todo"},
        dependencies=(operator_task.task_cid,),
        plan_cid=plan_root,
    )
    bootstrap_snapshot = TaskSourceSnapshot(
        source_schema="task-source@1",
        schema_version=1,
        plan_root_cid=plan_root,
        repository_tree_id=repository_tree,
        projection_cid="cid:bootstrap-projection",
        formal_plan_id="SPAR-PLAN-R1",
        source_identity="cid:task-source",
        revision=140,
        event_cursor=140,
        goal_count=2,
        task_count=2,
        dependency_count=1,
        terminal=False,
    )
    policy = TaskExecutionRoutePolicy.seal(
        snapshot=bootstrap_snapshot,
        tasks=(operator_task, implementation_task),
        execution_modes={
            "SPAR-000": GROK_CODEX_EXECUTION_MODE,
            "SPAR-001": GROK_CODEX_EXECUTION_MODE,
        },
    )
    binding = policy.binding_for_task(implementation_task)
    completion_receipt = {
        "execution_route_binding": binding.to_dict(),
        "execution_route_policy_id": policy.policy_id,
        "execution_route_origin_revision": binding.task_revision,
    }
    retrying_task = replace(
        implementation_task,
        status="retrying",
        revision=4,
        body={
            **dict(implementation_task.body),
            "status": "retrying",
            "completion_receipt": completion_receipt,
        },
    )
    current_snapshot = replace(
        bootstrap_snapshot,
        projection_cid="cid:current-projection",
    )
    bootstrap = {
        "operator_completed_task_ids": ["SPAR-000"],
        "task_count": 2,
        "plan_root_cid": plan_root,
        "repository_tree_id": repository_tree,
        "projection_cid": bootstrap_snapshot.projection_cid,
    }
    return (
        bootstrap,
        current_snapshot,
        operator_task,
        implementation_task,
        retrying_task,
        policy,
    )


def test_restart_reuses_exact_owner_written_execution_route_policy() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
        TaskExecutionRoutePolicy,
    )

    materializer = _materializer()
    (
        bootstrap,
        current_snapshot,
        operator_task,
        _implementation_task,
        retrying_task,
        original_policy,
    ) = _route_restart_fixture()
    resealed = TaskExecutionRoutePolicy.seal(
        snapshot=current_snapshot,
        tasks=(operator_task, retrying_task),
        execution_modes={
            "SPAR-000": GROK_CODEX_EXECUTION_MODE,
            "SPAR-001": GROK_CODEX_EXECUTION_MODE,
        },
    )
    assert resealed.policy_id != original_policy.policy_id

    resumed = materializer._resume_execution_route_policy(
        bootstrap=bootstrap,
        snapshot=current_snapshot,
        tasks=(operator_task, retrying_task),
    )

    assert resumed == original_policy


def test_restart_allows_operational_body_after_carried_route() -> None:
    materializer = _materializer()
    (
        bootstrap,
        current_snapshot,
        operator_task,
        _implementation_task,
        retrying_task,
        original_policy,
    ) = _route_restart_fixture()
    body = dict(retrying_task.body)
    body["unknown_callback_reopen_count"] = 1
    advanced = replace(
        retrying_task,
        revision=int(retrying_task.revision) + 1,
        body=body,
    )

    resumed = materializer._resume_execution_route_policy(
        bootstrap=bootstrap,
        snapshot=current_snapshot,
        tasks=(operator_task, advanced),
    )

    assert resumed == original_policy


def test_launch_route_population_allows_operational_body_after_route() -> None:
    materializer = _materializer()
    (
        _bootstrap,
        _current_snapshot,
        operator_task,
        _implementation_task,
        retrying_task,
        original_policy,
    ) = _route_restart_fixture()
    body = dict(retrying_task.body)
    body["unknown_callback_reopen_count"] = 1
    advanced = replace(
        retrying_task,
        revision=int(retrying_task.revision) + 1,
        body=body,
    )

    entries = materializer._assert_launch_route_population(
        tasks=(operator_task, advanced),
        execution_route_policy=original_policy,
        plan_cid=original_policy.plan_root_cid,
        repository_tree_id=original_policy.repository_tree_id,
    )

    assert set(entries) == {operator_task.task_cid, advanced.task_cid}


def test_launch_route_population_rejects_contract_drift_at_sealed_revision() -> None:
    materializer = _materializer()
    (
        _bootstrap,
        _current_snapshot,
        operator_task,
        implementation_task,
        _retrying_task,
        original_policy,
    ) = _route_restart_fixture()
    drifted = replace(
        implementation_task,
        body={**dict(implementation_task.body), "description": "drifted"},
    )

    with pytest.raises(
        materializer.OperatorError,
        match="launch task population differs from its immutable execution route",
    ):
        materializer._assert_launch_route_population(
            tasks=(operator_task, drifted),
            execution_route_policy=original_policy,
            plan_cid=original_policy.plan_root_cid,
            repository_tree_id=original_policy.repository_tree_id,
        )


def test_restart_recovers_only_exact_post_merge_predecessor_route() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        canonical_json_bytes,
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_RETRY_COOLDOWN_SCHEMA,
        _post_merge_retry_queue_receipt,
    )

    materializer = _materializer()
    (
        bootstrap,
        snapshot,
        operator_task,
        implementation_task,
        _retrying_task,
        original_policy,
    ) = _route_restart_fixture()
    route = original_policy.binding_for_task(implementation_task).to_dict()
    identity = {
        "attempt_id": "attempt:post-merge-route",
        "attempt_number": 1,
        "claim_id": "claim:post-merge-route",
        "lease_id": "lease:post-merge-route",
        "owner_session_id": "session:post-merge-route",
        "fencing_token": 1,
        "fence_epoch": 1,
    }
    terminal_reason = "Portal completion lacks one exact implementation commit"
    predecessor_receipt = {
        "operation": "database_portal_terminal_failure",
        **identity,
        "execution_phase": "failed",
        "execution_revision": 7,
        "execution_finished_at_ms": 1_000,
        "reason": terminal_reason,
        "retryable": False,
        "coordination": {
            "attempt_id": identity["attempt_id"],
            "claim_id": identity["claim_id"],
            "attempt_number": identity["attempt_number"],
        },
        "control_expected_status": "in_progress",
        "control_expected_revision": 2,
        "execution_route_binding": route,
        "execution_route_policy_id": route["policy_id"],
        "execution_route_origin_revision": route["task_revision"],
    }
    seed = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-post-merge-completion-recovery-seed@1"
        ),
        "task_cid": implementation_task.task_cid,
        "task_alias": implementation_task.task_alias,
        **identity,
        "source_task_revision": 3,
        "request_id": "request:post-merge-route",
        "candidate_commit": "a" * 40,
        "qualified_target_commit": "b" * 40,
        "qualification_kind": "repair",
        "qualification_receipt_id": "receipt:post-merge-route",
        "queue_source_attempt_id": identity["attempt_id"],
        "queue_source_claim_id": identity["claim_id"],
        "queue_source_lease_id": identity["lease_id"],
        "queue_source_fencing_token": identity["fencing_token"],
        "queue_source_fence_epoch": identity["fence_epoch"],
        "queue_source_binding_id": "sha256:" + "c" * 64,
        "queue_source_projection_immutable_digest": "sha256:" + "d" * 64,
        "recovery_evidence_id": "sha256:" + "e" * 64,
        "terminal_reason": terminal_reason,
    }
    seed["seed_id"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(seed)
    ).hexdigest()
    queue_reason = (
        "database_post_merge_declared_outputs_repair:"
        "request:post-merge-route:receipt:post-merge-route"
    )
    cooldown = {
        "schema": TYPED_RETRY_COOLDOWN_SCHEMA,
        "task_cid": implementation_task.task_cid,
        "expected_task_revision": 3,
        **identity,
        "delay_ms": 100,
        "started_at_ms": 1_000,
        "retry_not_before_ms": 1_100,
        "selection_penalty": 100,
        "consecutive_failures": 1,
        "reason": queue_reason,
        "expected_queue_revision": -1,
        "expected_queue_attempt": 0,
    }
    cooldown["resolution_cid"] = content_identity(
        {
            "typed_retry_cooldown": cooldown,
            "started_at_ms": cooldown["started_at_ms"],
        }
    )
    current_receipt = {
        "operation": "database_post_merge_declared_outputs_repair_recovery",
        **identity,
        "execution_phase": "failed",
        "execution_revision": 7,
        "execution_finished_at_ms": 1_000,
        "request_id": "request:post-merge-route",
        "candidate_commit": "a" * 40,
        "source_binding_id": "sha256:" + "c" * 64,
        "source_projection_immutable_digest": "sha256:" + "d" * 64,
        "queue_reason": queue_reason,
        "queue_receipt": _post_merge_retry_queue_receipt(cooldown),
        "coordination": dict(predecessor_receipt["coordination"]),
        "control_expected_status": "blocked",
        "control_expected_revision": 3,
        "repair_commit": "b" * 40,
        "repair_receipt_id": "receipt:post-merge-route",
        "repair_evidence_id": "sha256:" + "e" * 64,
        "post_merge_completion_recovery_seed": seed,
    }
    base_body = dict(implementation_task.body)
    predecessor_body = {**base_body, "completion_receipt": predecessor_receipt}
    current_body = {**base_body, "completion_receipt": current_receipt}
    current = replace(
        implementation_task,
        status="retrying",
        revision=4,
        body=current_body,
    )
    history = [
        {"revision": 1, "status": "todo", "body": base_body},
        {"revision": 2, "status": "in_progress", "body": base_body},
        {"revision": 3, "status": "blocked", "body": predecessor_body},
        {"revision": 4, "status": "retrying", "body": current_body},
    ]

    resumed = materializer._resume_execution_route_policy(
        bootstrap=bootstrap,
        snapshot=snapshot,
        tasks=(operator_task, current),
        histories_by_task={current.task_cid: history},
    )
    assert resumed == original_policy

    corrupted = list(history)
    corrupted[-2] = {**corrupted[-2], "body": base_body}
    with pytest.raises(materializer.OperatorError, match="exact post-merge"):
        materializer._resume_execution_route_policy(
            bootstrap=bootstrap,
            snapshot=snapshot,
            tasks=(operator_task, current),
            histories_by_task={current.task_cid: corrupted},
        )


def test_fresh_board_seals_current_execution_route_policy() -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
        GROK_CODEX_EXECUTION_MODE,
        TaskExecutionRoutePolicy,
    )

    materializer = _materializer()
    (
        bootstrap,
        snapshot,
        operator_task,
        implementation_task,
        _retrying_task,
        _original_policy,
    ) = _route_restart_fixture()
    expected = TaskExecutionRoutePolicy.seal(
        snapshot=snapshot,
        tasks=(operator_task, implementation_task),
        execution_modes={
            "SPAR-000": GROK_CODEX_EXECUTION_MODE,
            "SPAR-001": GROK_CODEX_EXECUTION_MODE,
        },
    )

    assert materializer._resume_execution_route_policy(
        bootstrap=bootstrap,
        snapshot=snapshot,
        tasks=(operator_task, implementation_task),
    ) == expected


def test_restart_rejects_corrupted_execution_route_lineage() -> None:
    materializer = _materializer()
    (
        bootstrap,
        snapshot,
        operator_task,
        _implementation_task,
        retrying_task,
        _original_policy,
    ) = _route_restart_fixture()
    body = dict(retrying_task.body)
    receipt = dict(body["completion_receipt"])
    binding = dict(receipt["execution_route_binding"])
    binding["policy_id"] = "cid:corrupted-policy"
    receipt["execution_route_policy_id"] = binding["policy_id"]
    receipt["execution_route_binding"] = binding
    body["completion_receipt"] = receipt
    corrupted = replace(retrying_task, body=body)

    with pytest.raises(
        materializer.OperatorError,
        match="does not reconstruct exactly",
    ):
        materializer._resume_execution_route_policy(
            bootstrap=bootstrap,
            snapshot=snapshot,
            tasks=(operator_task, corrupted),
        )


def test_restart_rejects_advanced_task_without_execution_route_lineage() -> None:
    materializer = _materializer()
    (
        bootstrap,
        snapshot,
        operator_task,
        implementation_task,
        retrying_task,
        _original_policy,
    ) = _route_restart_fixture()
    stripped = replace(retrying_task, body=dict(implementation_task.body))

    with pytest.raises(
        materializer.OperatorError,
        match="advanced ordinary task lacks carried execution-route lineage",
    ):
        materializer._resume_execution_route_policy(
            bootstrap=bootstrap,
            snapshot=snapshot,
            tasks=(operator_task, stripped),
        )


def test_restart_rejects_partial_execution_route_receipt() -> None:
    materializer = _materializer()
    (
        bootstrap,
        snapshot,
        operator_task,
        implementation_task,
        _retrying_task,
        original_policy,
    ) = _route_restart_fixture()
    partial = replace(
        implementation_task,
        body={
            **dict(implementation_task.body),
            "completion_receipt": {
                "execution_route_policy_id": original_policy.policy_id,
            },
        },
    )

    with pytest.raises(
        materializer.OperatorError,
        match="partial execution-route receipt",
    ):
        materializer._resume_execution_route_policy(
            bootstrap=bootstrap,
            snapshot=snapshot,
            tasks=(operator_task, partial),
        )


def test_recover_poisoned_owner_connection_replaces_shared_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _materializer()
    replacement = object()

    class _Poisoned:
        def __init__(self) -> None:
            self._poisoned = True
            self.path = tmp_path / "control.duckdb"
            self.closed = False

        def close(self) -> None:
            self.closed = True

    poisoned = _Poisoned()
    gateway = SimpleNamespace(_connection=poisoned)
    server = SimpleNamespace(
        _connection=poisoned,
        _owner_transaction_lock=threading.RLock(),
        _command_gateway=gateway,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "open_quack_state_owner_connection",
        lambda path: replacement,
    )

    assert materializer._recover_poisoned_owner_connection(server) is True
    assert poisoned.closed is True
    assert server._connection is replacement
    assert gateway._connection is replacement
    assert materializer._recover_poisoned_owner_connection(server) is False


def test_recover_poisoned_owner_connection_uses_config_path_when_wrap_has_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _materializer()
    replacement = object()
    database_path = tmp_path / "control.duckdb"
    opened: list[Path] = []

    class _Wrapped:
        def __init__(self) -> None:
            self._poisoned = True
            self.path = None
            self.closed = False

        def close(self) -> None:
            self.closed = True

    wrapped = _Wrapped()
    server = SimpleNamespace(
        _connection=wrapped,
        _owner_transaction_lock=threading.RLock(),
        _command_gateway=SimpleNamespace(_connection=wrapped),
        _transport_connection=wrapped,
        config=SimpleNamespace(database_path=database_path),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "open_quack_state_owner_connection",
        lambda path: opened.append(Path(path)) or replacement,
    )

    assert materializer._recover_poisoned_owner_connection(server) is True
    assert wrapped.closed is True
    assert opened == [database_path]
    assert server._connection is replacement
    assert server._command_gateway._connection is replacement
    assert server._transport_connection is replacement


def test_recover_poisoned_owner_connection_restarts_shared_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _materializer()
    replacement = object()
    refreshed: list[bool] = []

    class _Poisoned:
        def __init__(self) -> None:
            self._poisoned = True
            self.path = tmp_path / "control.duckdb"

        def close(self) -> None:
            return None

    poisoned = _Poisoned()
    server = SimpleNamespace(
        _connection=poisoned,
        _owner_transaction_lock=threading.RLock(),
        _command_gateway=SimpleNamespace(_connection=poisoned),
        _transport_connection=poisoned,
        _refresh_read_replica=lambda: refreshed.append(True),
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "open_quack_state_owner_connection",
        lambda path: replacement,
    )

    assert materializer._recover_poisoned_owner_connection(server) is True
    assert refreshed == []
    assert server._transport_connection is replacement


def test_recover_poisoned_owner_connection_reconnects_without_second_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _materializer()
    opened: list[Path] = []

    class _Owner:
        def __init__(self) -> None:
            self._poisoned = True
            self.path = tmp_path / "control.duckdb"
            self.reconnected = 0

        def reconnect_exclusive_owner(self) -> None:
            self.reconnected += 1
            self._poisoned = False

        def close(self) -> None:
            raise AssertionError("close must not drop the exclusive thread lock")

    owner = _Owner()
    server = SimpleNamespace(
        _connection=owner,
        _owner_transaction_lock=threading.RLock(),
        _command_gateway=SimpleNamespace(_connection=owner),
        _transport_connection=owner,
        _refresh_read_replica=lambda: None,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "open_quack_state_owner_connection",
        lambda path: opened.append(Path(path)) or object(),
    )

    assert materializer._recover_poisoned_owner_connection(server, force=True) is True
    assert owner.reconnected == 1
    assert opened == []
    assert server._connection is owner
    assert server._command_gateway._connection is owner


def test_recover_force_does_not_bounce_healthy_owner_when_listener_is_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _materializer()

    class _Owner:
        def __init__(self) -> None:
            self._poisoned = False
            self.path = tmp_path / "control.duckdb"
            self.reconnected = 0

        def reconnect_exclusive_owner(self) -> None:
            self.reconnected += 1

    owner = _Owner()
    server = SimpleNamespace(
        _connection=owner,
        _owner_transaction_lock=threading.RLock(),
        _command_gateway=SimpleNamespace(_connection=owner),
        _transport_connection=owner,
        config=SimpleNamespace(
            database_path=tmp_path / "control.duckdb",
            container_bind_host="127.0.0.1",
            container_port=46731,
        ),
    )
    monkeypatch.setattr(materializer, "_owner_listener_ready", lambda _server: True)

    assert materializer._recover_poisoned_owner_connection(server, force=True) is False
    assert owner.reconnected == 0
    assert server._connection is owner
    assert server._transport_connection is owner


def test_owner_listener_ready_uses_host_port_when_container_fields_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    server = SimpleNamespace(
        config=SimpleNamespace(host="127.0.0.1", port=46731, container_bind_host="", container_port=0),
        _bound_port=0,
    )
    seen: list[tuple[str, int]] = []

    def connect(address: object, timeout: object = None) -> object:
        del timeout
        host, port = address  # type: ignore[misc]
        seen.append((str(host), int(port)))
        return socket.socket()

    monkeypatch.setattr(materializer.socket, "create_connection", connect)
    assert materializer._owner_listener_ready(server) is True
    assert seen == [("127.0.0.1", 46731)]


def test_owner_listener_ready_refuses_only_connection_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    server = SimpleNamespace(
        config=SimpleNamespace(container_bind_host="127.0.0.1", container_port=46731),
        _bound_port=46731,
    )

    def refused(_address: object, timeout: object = None) -> object:
        del timeout
        raise ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused")

    monkeypatch.setattr(materializer.socket, "create_connection", refused)
    assert materializer._owner_listener_ready(server) is False

    def timed_out(_address: object, timeout: object = None) -> object:
        del timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(materializer.socket, "create_connection", timed_out)
    assert materializer._owner_listener_ready(server) is True

    def reset(_address: object, timeout: object = None) -> object:
        del timeout
        raise ConnectionResetError(errno.ECONNRESET, "Connection reset")

    monkeypatch.setattr(materializer.socket, "create_connection", reset)
    assert materializer._owner_listener_ready(server) is True


def test_recover_force_does_not_bounce_when_listener_times_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _materializer()

    class _Owner:
        def __init__(self) -> None:
            self._poisoned = False
            self.path = tmp_path / "control.duckdb"
            self.reconnected = 0

        def reconnect_exclusive_owner(self) -> None:
            self.reconnected += 1

    owner = _Owner()
    refreshed: list[bool] = []
    server = SimpleNamespace(
        _connection=owner,
        _owner_transaction_lock=threading.RLock(),
        _command_gateway=SimpleNamespace(_connection=owner),
        _transport_connection=owner,
        _refresh_read_replica=lambda: refreshed.append(True),
        config=SimpleNamespace(
            database_path=tmp_path / "control.duckdb",
            container_bind_host="127.0.0.1",
            container_port=46731,
        ),
    )
    monkeypatch.setattr(
        materializer.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )

    assert materializer._recover_poisoned_owner_connection(server, force=True) is False
    assert owner.reconnected == 0
    assert refreshed == []
    assert server._transport_connection is owner


def test_recover_reconnects_closed_exclusive_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materializer = _materializer()

    class _Owner:
        def __init__(self) -> None:
            self._poisoned = False
            self._closed = True
            self._connection = None
            self.path = tmp_path / "control.duckdb"
            self.reconnected = 0
            self._lock_context = object()

        def reconnect_exclusive_owner(self) -> None:
            self.reconnected += 1
            self._closed = False
            self._connection = object()

    owner = _Owner()
    server = SimpleNamespace(
        _connection=owner,
        _owner_transaction_lock=threading.RLock(),
        _command_gateway=SimpleNamespace(_connection=owner),
        _transport_connection=owner,
        _refresh_read_replica=lambda: None,
        config=SimpleNamespace(
            database_path=tmp_path / "control.duckdb",
            container_bind_host="127.0.0.1",
            container_port=46731,
        ),
    )
    monkeypatch.setattr(materializer, "_owner_listener_ready", lambda _server: False)

    assert materializer._recover_poisoned_owner_connection(server, force=True) is True
    assert owner.reconnected == 1
    assert owner._closed is False
    assert server._connection is owner
    assert server._command_gateway._connection is owner


def test_reconnect_exclusive_owner_keeps_usable_handle(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnection,
    )

    path = tmp_path / "control.duckdb"
    owner = DuckDBConnection(path)
    try:
        owner.execute("CREATE TABLE t(id INTEGER)")
        owner.execute("INSERT INTO t VALUES (1)")
        with owner._execution_condition:
            owner._poison_locked()
        assert owner._poisoned is True
        owner.reconnect_exclusive_owner()
        assert owner._poisoned is False
        assert owner._closed is False
        probe = owner.execute("SELECT COUNT(*) FROM t").fetchone()
        assert probe is not None
        assert int(probe[0]) == 1
    finally:
        owner.close()


def test_process_mutations_skips_signed_owner_command_requests(
    tmp_path: Path,
) -> None:
    materializer = _materializer()
    mutation_dir = tmp_path / "mutations"
    mutation_dir.mkdir()
    request = mutation_dir / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.request.json"
    request.write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "quack-owner-command-request@1"
                ),
                "command": "compare_and_set_status",
                "sql": "DELETE FROM tasks",
            }
        ),
        encoding="utf-8",
    )
    executed: list[object] = []
    server = SimpleNamespace(
        _connection=SimpleNamespace(
            execute=lambda *args, **kwargs: executed.append(args)
        ),
        _owner_transaction_lock=threading.RLock(),
    )
    materializer._process_mutations(server, mutation_dir)
    assert request.exists()
    assert executed == []
    assert not (mutation_dir / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.done.json").exists()


def test_process_mutations_applies_board_unstall_op(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    mutation_dir = tmp_path / "mutations"
    mutation_dir.mkdir()
    request = mutation_dir / "board.request.json"
    request.write_text(
        json.dumps({"op": "board_unstall", "stale_seconds": 16_200}),
        encoding="utf-8",
    )
    applied: list[object] = []

    def _apply(connection: object, payload: object, **_kwargs: object) -> dict[str, object]:
        applied.append((connection, payload))
        return {"ok": True, "rowcount": 1, "board_unstall": {"unstalled": []}}

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state."
        "apply_owner_command_payload",
        _apply,
    )
    connection = object()
    server = SimpleNamespace(
        _connection=connection,
        _owner_transaction_lock=threading.RLock(),
    )
    materializer._process_mutations(server, mutation_dir)
    assert applied == [
        (connection, {"op": "board_unstall", "stale_seconds": 16_200})
    ]
    assert not request.exists()
    done = json.loads(
        (mutation_dir / "board.done.json").read_text(encoding="utf-8")
    )
    assert done["ok"] is True
    assert done["rowcount"] == 1


def test_owner_command_inbox_is_runtime_registry_mutations() -> None:
    materializer = _materializer()
    program = SimpleNamespace(
        runtime_registry_path=(
            "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1/"
            "registry"
        )
    )
    inbox = materializer._owner_command_inbox(program)
    assert inbox == (
        ROOT
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
        / "registry"
        / "mutations"
    )
    assert inbox.is_dir()


def test_publish_live_projection_skips_live_query_birth_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    calls: list[bool] = []

    class _Server:
        def ready(self, *, retry_transient_birth: bool = True) -> dict[str, object]:
            calls.append(retry_transient_birth)
            raise AssertionError("periodic projection must not call server.ready()")

    server = _Server()
    server._connection = object()  # type: ignore[attr-defined]
    server._owner_transaction_lock = threading.RLock()  # type: ignore[attr-defined]
    server._identity = SimpleNamespace(  # type: ignore[attr-defined]
        process_birth_id="birth:test",
        server_id="server:test",
        store_id="store:test",
        generation=1,
        schema_revision=1,
    )
    monkeypatch.setattr(
        materializer,
        "_task_status",
        lambda _connection: {"task_count": 0, "status_counts": {}},
    )
    payload = materializer._publish_live_projection(server, {"owner": tmp_path})
    assert calls == []
    assert payload["quack_authenticated_live_query"] is True
    assert payload["owner_process_birth_id"] == "birth:test"
    assert (tmp_path / "spar-live-projection.json").is_file()


def test_publish_live_projection_uses_identity_when_live_query_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()

    class _Server:
        def ready(self, *, retry_transient_birth: bool = True) -> dict[str, object]:
            del retry_transient_birth
            raise RuntimeError("authenticated remote live query failed: IOException")

    server = _Server()
    server._connection = object()  # type: ignore[attr-defined]
    server._owner_transaction_lock = threading.RLock()  # type: ignore[attr-defined]
    server._identity = SimpleNamespace(  # type: ignore[attr-defined]
        process_birth_id="birth:fallback",
        server_id="server:fallback",
        store_id="store:fallback",
        generation=42,
        schema_revision=3,
    )
    monkeypatch.setattr(materializer, "_owner_listener_ready", lambda _server: True)
    monkeypatch.setattr(
        materializer,
        "_task_status",
        lambda _connection: {
            "task_count": 51,
            "status_counts": {"completed": 18, "todo": 32, "quarantined": 1},
            "dependency_ready_task_ids": ["SPAR-018"],
            "active_task_ids": [],
            "blocked_count": 0,
            "terminal_count": 19,
        },
    )
    payload = materializer._publish_live_projection(server, {"owner": tmp_path})
    assert payload["generation"] == 42
    assert payload["owner_process_birth_id"] == "birth:fallback"
    assert payload["quack_authenticated_live_query"] is True
    assert payload["task_projection"]["dependency_ready_task_ids"] == ["SPAR-018"]


def test_owner_projection_monitor_does_not_force_bounce_when_listener_is_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    recover_force: list[bool] = []
    publishes = {"n": 0}

    def publish(*_args: object, **_kwargs: object) -> dict[str, object]:
        publishes["n"] += 1
        if publishes["n"] == 2:
            raise RuntimeError("authenticated remote live query failed: IOException")
        return {"ok": True}

    def recover(_server: object, *, force: bool = False) -> bool:
        recover_force.append(force)
        return False

    monkeypatch.setattr(materializer, "_publish_live_projection", publish)
    monkeypatch.setattr(materializer, "_process_mutations", lambda *_a, **_k: None)
    monkeypatch.setattr(materializer, "_recover_poisoned_owner_connection", recover)
    monkeypatch.setattr(materializer, "_owner_listener_ready", lambda _server: True)
    server = SimpleNamespace(
        process_mutation_inbox=lambda: None,
        _connection=SimpleNamespace(path=tmp_path / "control.duckdb"),
        config=SimpleNamespace(database_path=tmp_path / "control.duckdb"),
    )
    inbox = tmp_path / "registry" / "mutations"
    inbox.mkdir(parents=True)
    monitor = materializer._OwnerProjectionMonitor(
        server,
        {"owner": tmp_path / "owner"},
        mutation_dir=inbox,
        on_failure=lambda _exc: None,
    )
    monitor.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not recover_force:
            time.sleep(0.02)
        assert recover_force == [False]
    finally:
        monitor.stop()


def test_owner_projection_monitor_drains_signed_owner_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _materializer()
    drained = {"n": 0}

    def process_inbox() -> None:
        drained["n"] += 1

    server = SimpleNamespace(process_mutation_inbox=process_inbox)
    monkeypatch.setattr(
        materializer,
        "_publish_live_projection",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        materializer,
        "_process_mutations",
        lambda *_args, **_kwargs: None,
    )
    inbox = tmp_path / "registry" / "mutations"
    inbox.mkdir(parents=True)
    monitor = materializer._OwnerProjectionMonitor(
        server,
        {"owner": tmp_path / "owner"},
        mutation_dir=inbox,
        on_failure=lambda _exc: None,
    )
    monitor.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and drained["n"] < 1:
            time.sleep(0.02)
        assert drained["n"] >= 1
    finally:
        monitor.stop()
