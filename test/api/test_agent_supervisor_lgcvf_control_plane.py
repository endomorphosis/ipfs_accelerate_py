"""Focused conformance tests for the LGCVF one-writer control plane."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import (
    FormalWorkPlan,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    CODEX_MODEL_ENV,
    PROVIDER_ENV,
    configured_board_launch_plan,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)

from scripts import (
    materialize_logic_governed_compositional_verification_fabric_control_plane as materializer,
)

ROOT = Path(__file__).resolve().parents[2]


def _population() -> tuple[dict[str, Any], dict[str, Any]]:
    config = materializer.load_config()
    formal_path = ROOT / str(config["formal_plan_path"])
    todo_path = ROOT / str(config["taskboard_path"])
    formal = FormalWorkPlan.from_dict(json.loads(formal_path.read_text(encoding="utf-8")))
    source = {
        "accelerator_head": "1" * 40,
        "accelerator_tree": "2" * 40,
        "source_forest_root": "baguqeera-test-source-forest",
    }
    population = materializer.project_population(
        config,
        formal_plan=formal,
        todo_text=todo_path.read_text(encoding="utf-8"),
        source=source,
    )
    return config, population


def test_population_rejects_scheduler_formal_task_prefix_projection_drift() -> None:
    config = materializer.load_config()
    formal_path = ROOT / str(config["formal_plan_path"])
    todo_path = ROOT / str(config["taskboard_path"])
    formal = FormalWorkPlan.from_dict(json.loads(formal_path.read_text(encoding="utf-8")))
    drifted = copy.deepcopy(config)
    drifted["task_prefix"] = "## WRONG-"

    with pytest.raises(
        materializer.MaterializationError,
        match="Markdown task selector differs from the formal logical prefix",
    ):
        materializer.project_population(
            drifted,
            formal_plan=formal,
            todo_text=todo_path.read_text(encoding="utf-8"),
            source={
                "accelerator_head": "1" * 40,
                "accelerator_tree": "2" * 40,
                "source_forest_root": "baguqeera-test-source-forest",
            },
        )


def _revision_one_runtime(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize archived revision 1 under the revision-2 runtime config."""

    config, successor = _population()
    archive_dir = (
        ROOT
        / "data/agent_supervisor/logic_governed_compositional_verification_fabric/plan_revisions"
    )
    archive_files = tuple(archive_dir.glob("*.json"))
    assert len(archive_files) == 1
    archive_path = archive_files[0]
    predecessor = FormalWorkPlan.from_dict(
        json.loads(archive_path.read_text(encoding="utf-8"))
    )
    successor_formal = FormalWorkPlan.from_dict(
        json.loads((ROOT / str(config["formal_plan_path"])).read_text(encoding="utf-8"))
    )
    baseline = str(successor_formal.metadata["accelerator_construction_head"])
    old_todo = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.hooksPath=/dev/null",
            "show",
            baseline
            + ":docs/architecture/logic_governed_compositional_verification_fabric.todo.md",
        ],
        cwd=ROOT,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    ).stdout
    predecessor_config = copy.deepcopy(config)
    predecessor_config["plan_binding"]["formal_plan_content_id"] = predecessor.content_id
    predecessor_config["plan_binding"]["predecessor_plan_cid"] = (
        predecessor.metadata["predecessor_plan_cid"]
    )
    predecessor_config["initial_projection"]["task_count"] = 27
    source = {
        "accelerator_head": successor["source_head"],
        "accelerator_tree": str(successor["repository_tree_id"]).removeprefix(
            "git-tree:"
        ),
        "source_forest_root": successor["source_forest_root"],
    }
    predecessor_population = materializer.project_population(
        predecessor_config,
        formal_plan=predecessor,
        todo_text=old_todo,
        source=source,
    )
    runtime_config = copy.deepcopy(config)
    runtime_config["database_program"]["store_id"] = "run/control.duckdb"
    runtime_config["runtime_paths"]["state"] = "run/state"
    runtime_config["runtime_paths"]["evidence"] = "run/evidence"
    archive_target = (
        tmp_path
        / str(runtime_config["formal_plan_path"])
    ).parent / "plan_revisions" / f"{predecessor.content_id}.json"
    archive_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_path, archive_target)
    materializer.materialize(
        runtime_config,
        predecessor_population,
        root=tmp_path,
        recheck_source=False,
    )
    return runtime_config, successor


def _runtime_paths(
    config: dict[str, Any], tmp_path: Path
) -> dict[str, Path]:
    return materializer._successor_paths(config, root=tmp_path)  # noqa: SLF001


def _crash_successor_process(
    tmp_path: Path,
    config: dict[str, Any],
    population: dict[str, Any],
    *,
    fault_point: str,
) -> subprocess.CompletedProcess[str]:
    """Exit a child process at a durable plan-revision crash boundary."""

    config_path = tmp_path / "crash-config.json"
    population_path = tmp_path / "crash-population.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    population_path.write_text(json.dumps(population), encoding="utf-8")
    program = """
import json
import os
import sys
from pathlib import Path
from scripts import materialize_logic_governed_compositional_verification_fabric_control_plane as materializer

root = Path(sys.argv[1])
config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
population = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
fault_point = sys.argv[4]

def crash(point):
    if point == fault_point:
        os._exit(73)

materializer.steer_successor(
    config,
    population,
    root=root,
    fault_injector=crash,
)
"""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(tmp_path),
            str(config_path),
            str(population_path),
            fault_point,
        ],
        cwd=ROOT,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(ROOT),
        },
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


def test_lgcvf_scheduler_is_single_writer_and_protects_control_evidence() -> None:
    config = materializer.load_config()
    board = load_configured_board(
        materializer.CONFIG_PATH,
        repo_root=ROOT,
    )

    assert board.board_namespace == materializer.EXPECTED_NAMESPACE
    assert board.merge_target_branch == (
        "agent/logic-governed-compositional-verification-fabric-v1"
    )
    assert board.max_lanes == 1
    assert board.worktree_submodule_paths == ("ipfs_datasets_py",)
    assert config["bootstrap_writer_policy"] == {
        "maximum_processes": 1,
        "quack_required": False,
        "direct_multi_process_duckdb_permitted": False,
        "automatic_installation_permitted": False,
    }
    program = board.resolved_database_program()
    assert program.authority_mode == "embedded"
    assert program.task_source_kind == "duckdb"
    assert program.quack_endpoint == ""
    assert program.failover_policy == "fail_closed"
    launch = configured_board_launch_plan(
        board,
        implement=True,
        detach=False,
    )
    assert launch["environment"][PROVIDER_ENV] == "grok_cli"
    # This is an explicit one-provider route. Exporting CODEX_MODEL_ENV would
    # instead be parsed as an incomplete sealed Grok/Codex fallback tuple.
    assert CODEX_MODEL_ENV not in launch["environment"]
    protected = set(board.protected_paths)
    assert {
        str(config["formal_plan_path"]),
        str(config["taskboard_path"]),
        str(config["validator_path"]),
        str(config["materializer_path"]),
        "scripts/qualify_logic_governed_compositional_verification_fabric.py",
        "scripts/validate_logic_governed_compositional_verification_fabric_closeout.py",
        "docs/architecture/logic_governed_compositional_verification/paired_benchmark_result.json",
        "test/api/test_agent_supervisor_compositional_verification_vertical.py",
        "test/fixtures/agent_supervisor/compositional_verification/tests/test_selected.py",
        "test/fixtures/agent_supervisor/compositional_verification/tests/test_unselected.py",
    }.issubset(protected)


def test_explicit_objective_heap_keeps_optional_objective_daemon_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    optional_module = (
        "ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon"
    )
    monkeypatch.setitem(sys.modules, optional_module, None)
    holder = SimpleNamespace(
        config=SimpleNamespace(
            objective_path=(
                ROOT
                / "docs/architecture/"
                "logic_governed_compositional_verification_fabric.objectives.md"
            ),
            repo_root=ROOT,
        )
    )

    goals = PortalImplementationSupervisor._objective_goals_for_finding_mapping(holder)

    assert {goal.goal_id for goal in goals} >= {"LGCVF-G000", "LGCVF-G130"}


def test_population_preserves_formal_identities_dependencies_and_closed_gates() -> None:
    config, population = _population()
    formal = FormalWorkPlan.from_dict(
        json.loads((ROOT / str(config["formal_plan_path"])).read_text(encoding="utf-8"))
    )
    projected = {str(item["task_id"]): item for item in population["tasks"]}
    formal_tasks = {item.task_id: item for item in formal.tasks}

    assert population["plan_root_cid"] == formal.content_id
    assert len(population["objectives"]) == 14
    assert len(population["tasks"]) == 28
    for task_id, task in formal_tasks.items():
        record = projected[task_id]
        assert record["task_cid"] == task.content_id
        assert record["formal_task_content_id"] == task.content_id
        assert record["dependencies"] == [
            formal_tasks[dependency].content_id for dependency in task.depends_on
        ]

    assert projected["LGCVF-121"]["status"] == "blocked"
    assert projected["LGCVF-121"]["construction_status"] == ("blocked_external_authority")
    assert projected["LGCVF-123"]["status"] == "blocked"
    assert projected["LGCVF-123"]["construction_status"] == "blocked_manual"
    for task_id, completion in (
        ("LGCVF-121", "external-authority"),
        ("LGCVF-123", "manual"),
    ):
        assert projected[task_id]["completion"] == completion
        assert projected[task_id]["review_only"] is True
        assert projected[task_id]["is_schedulable"] is False


def test_typed_materialization_read_only_replay_and_overwrite_rejection(
    tmp_path: Path,
) -> None:
    assert DatabaseTaskSource.available(), "required DuckDB capability is unavailable"
    config, population = _population()
    temporary = copy.deepcopy(config)
    temporary["database_program"]["store_id"] = "run-v1/control.duckdb"
    temporary["runtime_paths"]["evidence"] = "run-v1/evidence"

    receipt = materializer.materialize(
        temporary,
        population,
        root=tmp_path,
        recheck_source=False,
    )
    assert receipt["maximum_writer_processes"] == 1
    assert receipt["quack_qualified"] is False
    assert receipt["plan_root_cid"] == population["plan_root_cid"]
    assert receipt["verification"]["valid"] is True
    assert receipt["verification"]["stores_unchanged"] is True

    live = materializer.verify_read_only(
        temporary,
        population,
        root=tmp_path,
        expected_stage="live",
    )
    assert live["valid"] is True
    assert live["verification_mode"] == "read_only"
    assert live["control"]["task_count"] == 28
    assert live["coordination"]["counts"]["registered_tasks"] == 28
    assert not any(live["execution"]["row_counts"].values())

    control_path = tmp_path / str(temporary["database_program"]["store_id"])
    source = DatabaseTaskSource(control_path, install_schema=False)
    try:
        ready = {item.task_alias for item in source.ready_tasks(limit=100).tasks}
        manual = source.get_task("LGCVF-123")
        external = source.get_task("LGCVF-121")
    finally:
        source.close()
    assert ready == {"LGCVF-051", "LGCVF-060", "LGCVF-070", "LGCVF-080"}
    assert manual is not None
    assert manual.status == "blocked"
    assert manual.body["construction_status"] == "blocked_manual"
    assert manual.body["completion"] == "manual"
    assert external is not None
    assert external.body["construction_status"] == "blocked_external_authority"

    with pytest.raises(materializer.MaterializationError, match="refusing to overwrite"):
        materializer.materialize(
            temporary,
            population,
            root=tmp_path,
            recheck_source=False,
        )


def test_successor_preview_is_read_only_and_emits_the_closed_delta(
    tmp_path: Path,
) -> None:
    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    before = {
        key: materializer._sha256_file(paths[key])  # noqa: SLF001
        for key in ("control", "coordination", "execution", "receipt")
    }

    preview = materializer.preview_successor(config, successor, root=tmp_path)

    after = {
        key: materializer._sha256_file(paths[key])  # noqa: SLF001
        for key in ("control", "coordination", "execution", "receipt")
    }
    assert after == before
    assert preview["write_performed"] is False
    assert len(preview["retained_task_cids"]) == 27
    assert len(preview["added_task_cids"]) == 1
    assert set(preview["amended_aliases"]) == {
        "LGCVF-081",
        "LGCVF-111",
        "LGCVF-112",
        "LGCVF-120",
        "LGCVF-122",
        "LGCVF-124",
    }
    assert set(preview["reprioritized_aliases"]) == {"LGCVF-121", "LGCVF-123"}
    delta_items = preview["delta"]["items"]
    operations = {item["operation"] for item in delta_items}
    assert operations == {
        "add_task",
        "amend_unstarted_task",
        "reprioritize_unstarted_task",
    }
    assert len([item for item in delta_items if item["operation"] == "add_task"]) == 1
    assert len(
        [item for item in delta_items if item["operation"] == "amend_unstarted_task"]
    ) == 6
    assert len(
        [
            item
            for item in delta_items
            if item["operation"] == "reprioritize_unstarted_task"
        ]
    ) == 2
    candidates = {
        item["task_id"]: item for item in preview["candidate_population"]["tasks"]
    }
    candidate_113_cid = candidates["LGCVF-113"]["task_cid"]
    assert candidate_113_cid in candidates["LGCVF-120"]["dependencies"]
    changed = {
        item["target_cid"]: item
        for item in delta_items
        if item["operation"] != "add_task"
    }
    for alias in materializer.SUCCESSOR_CHANGED_ALIASES:
        candidate = candidates[alias]
        predecessor_spec = preview["evidence"]["task_spec_cids"][alias]
        target = candidate["task_cid"]
        item = changed[target]
        assert item["before_digest"] == predecessor_spec
        assert item["expected_target_spec_revision"] == predecessor_spec
        assert item["after_record_cid"] == preview["candidate_task_spec_cids"][alias]
        assert item["affected_task_cids"] == [target]
    add_item = next(
        item for item in delta_items if item["operation"] == "add_task"
    )
    assert add_item["target_cid"] == ""
    assert add_item["after_record_cid"] == candidate_113_cid
    assert add_item["affected_task_cids"] == [candidate_113_cid]
    for alias in ("LGCVF-111", "LGCVF-112"):
        policy = candidates[alias]["markdown_metadata"]["conflict_policy"]
        assert "candidate" in policy.casefold()
        assert candidates[alias]["markdown_metadata_cid"]
    assert (
        "program_repair_synthesis.py"
        in candidates["LGCVF-081"]["markdown_metadata"]["outputs"]
    )
    assert preview["evidence"]["retained_completion_binding"]["binding_cid"]
    assert preview["evidence"]["protected_blocker_binding"]["binding_cid"]


def test_successor_steer_preserves_history_registers_only_113_and_replays(
    tmp_path: Path,
) -> None:
    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    source = DatabaseTaskSource(paths["control"], install_schema=False)
    try:
        task_080 = source.get_task("LGCVF-080")
        assert task_080 is not None
        source.compare_and_set_status(
            task_080,
            task_080.revision,
            "blocked",
            {"reason": "preserved_failed_candidate"},
        )
        before_tasks = {
            item.task_alias: item.to_dict()
            for item in source.list_tasks(limit=100).tasks
        }
    finally:
        source.close()
    execution_before = materializer._sha256_file(paths["execution"])  # noqa: SLF001
    bootstrap_before = materializer._sha256_file(paths["receipt"])  # noqa: SLF001

    receipt = materializer.steer_successor(config, successor, root=tmp_path)
    verification = materializer.verify_successor_read_only(
        config, successor, root=tmp_path
    )
    replay = materializer.steer_successor(config, successor, root=tmp_path)

    assert receipt["receipt_cid"] == replay["receipt_cid"]
    assert verification["valid"] is True
    assert verification["task_count"] == 28
    assert verification["retained_task_count"] == 27
    assert verification["added_task_count"] == 1
    assert verification["accepted_history_preserved"] is True
    assert materializer._sha256_file(paths["execution"]) == execution_before  # noqa: SLF001
    assert materializer._sha256_file(paths["receipt"]) == bootstrap_before  # noqa: SLF001
    source = DatabaseTaskSource(paths["control"], install_schema=False)
    try:
        after_tasks = {
            item.task_alias: item.to_dict()
            for item in source.list_tasks(limit=100).tasks
        }
    finally:
        source.close()
    assert after_tasks["LGCVF-080"]["status"] == "blocked"
    for alias, before in before_tasks.items():
        assert after_tasks[alias]["task_cid"] == before["task_cid"]
        assert after_tasks[alias]["status"] == before["status"]
    assert after_tasks["LGCVF-121"]["body"] == before_tasks["LGCVF-121"]["body"]
    assert after_tasks["LGCVF-123"]["body"] == before_tasks["LGCVF-123"]["body"]
    coordination = materializer.read_coordination_registry_projection(
        paths["coordination"]
    )
    registered = {
        item["task_id"]: item["task_cid"] for item in coordination["tasks"]
    }
    assert set(registered) == set(after_tasks)
    assert registered["LGCVF-113"] == after_tasks["LGCVF-113"]["task_cid"]
    dependency_edges = {
        (item["task_cid"], item["dependency_task_cid"])
        for item in coordination["dependency_edges"]
    }
    assert (
        after_tasks["LGCVF-120"]["task_cid"],
        after_tasks["LGCVF-113"]["task_cid"],
    ) in dependency_edges
    assert (
        after_tasks["LGCVF-113"]["task_cid"]
        in after_tasks["LGCVF-120"]["dependencies"]
    )


def test_successor_verify_allows_status_receipt_but_rejects_plan_spec_drift(
    tmp_path: Path,
) -> None:
    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    materializer.steer_successor(config, successor, root=tmp_path)

    source = DatabaseTaskSource(paths["control"], install_schema=False)
    try:
        task = source.get_task("LGCVF-080")
        assert task is not None
        source.compare_and_set_status(
            task,
            task.revision,
            "retrying",
            {
                "operation": "typed_portal_validation_retry_recovery",
                "receipt_id": "sha256:" + ("42" * 32),
                "authority": "database-implementation-daemon",
            },
        )
        tasks_by_alias = {
            str(item["task_alias"]): item for item in source.plan_projection()["tasks"]
        }
    finally:
        source.close()

    assert materializer.verify_successor_read_only(
        config, successor, root=tmp_path
    )["valid"] is True
    baseline = copy.deepcopy(tasks_by_alias["LGCVF-080"])

    def write_projection(task_projection: dict[str, Any]) -> None:
        with open_intent_repository(
            paths["control"],
            owner_id="lgcvf-spec-drift-test",
            install_schema=False,
        ) as repository:
            current = repository.get_task(str(task_projection["task_cid"]))
            assert current is not None
            repository.upsert_task(
                task_cid=str(task_projection["task_cid"]),
                task_alias=str(task_projection["task_alias"]),
                goal_cid=str(task_projection["goal_cid"]),
                plan_cid=str(task_projection["plan_cid"]),
                objective_id=str(task_projection["objective_id"]),
                ordinal=int(task_projection["ordinal"]),
                status=str(task_projection["status"]),
                priority=str(task_projection["priority"]),
                body=copy.deepcopy(task_projection["body"]),
                identity=copy.deepcopy(task_projection["identity"]),
                expected_revision=int(current["revision"]),
                dependencies=[
                    str(item["dependency_task_cid"])
                    for item in task_projection["dependencies"]
                ],
                outputs=[copy.deepcopy(item["effect"]) for item in task_projection["outputs"]],
                acceptance=[
                    {
                        **copy.deepcopy(item["evidence_policy"]),
                        "criterion": str(item["criterion"]),
                    }
                    for item in task_projection["acceptance"]
                ],
                validations=[
                    {
                        "argv": list(item["argv"]),
                        **copy.deepcopy(item["policy"]),
                    }
                    for item in task_projection["validations"]
                ],
            )

    dependency_candidate = next(
        item["task_cid"]
        for alias, item in sorted(tasks_by_alias.items())
        if alias != "LGCVF-080"
        and item["task_cid"]
        not in {
            dependency["dependency_task_cid"]
            for dependency in baseline["dependencies"]
        }
    )

    def drift_title(value: dict[str, Any]) -> None:
        value["body"]["title"] = "forged title"

    def drift_authority(value: dict[str, Any]) -> None:
        value["body"]["authority"] = "forged authority"

    def drift_output(value: dict[str, Any]) -> None:
        value["outputs"][0]["effect"]["effect"] = "forged-effect"

    def drift_dependency(value: dict[str, Any]) -> None:
        value["dependencies"].append(
            {"dependency_task_cid": dependency_candidate, "kind": "depends_on"}
        )

    def drift_acceptance(value: dict[str, Any]) -> None:
        value["acceptance"][0]["criterion"] += " (forged)"

    def drift_validation(value: dict[str, Any]) -> None:
        value["validations"][0]["argv"].append("--forged")

    for drift in (
        drift_title,
        drift_output,
        drift_dependency,
        drift_acceptance,
        drift_validation,
        drift_authority,
    ):
        forged = copy.deepcopy(baseline)
        drift(forged)
        write_projection(forged)
        with pytest.raises(
            materializer.MaterializationError,
            match="current task specification is stale",
        ):
            materializer.verify_successor_read_only(
                config, successor, root=tmp_path
            )
        write_projection(baseline)
        assert materializer.verify_successor_read_only(
            config, successor, root=tmp_path
        )["valid"] is True


def test_successor_preview_rejects_active_claim_authority(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )

    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    source = DatabaseTaskSource(paths["control"], install_schema=False)
    try:
        task = source.get_task("LGCVF-080")
        assert task is not None
    finally:
        source.close()
    coordinator = DatabaseCoordinator(paths["coordination"])
    try:
        coordinator.open()
        coordinator.claim_task(
            task_cid=task.task_cid,
            owner_session_id="test-active-owner",
            lease_ms=60_000,
            idempotency_key="test-active-claim",
        )
    finally:
        coordinator.close()

    with pytest.raises(materializer.MaterializationError, match="active claims"):
        materializer.preview_successor(config, successor, root=tmp_path)


def test_successor_injected_failure_restores_control_and_coordination_exactly(
    tmp_path: Path,
) -> None:
    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    before = {
        key: materializer._sha256_file(paths[key])  # noqa: SLF001
        for key in ("control", "coordination", "execution", "receipt")
    }

    def fail_after_composite_apply(point: str) -> None:
        if point == "after_duckdb":
            raise RuntimeError("injected-successor-failure")

    with pytest.raises(
        materializer.MaterializationError, match="exact operational rollback"
    ):
        materializer.steer_successor(
            config,
            successor,
            root=tmp_path,
            fault_injector=fail_after_composite_apply,
        )
    after = {
        key: materializer._sha256_file(paths[key])  # noqa: SLF001
        for key in ("control", "coordination", "execution", "receipt")
    }
    assert after == before
    source = DatabaseTaskSource(paths["control"], install_schema=False)
    try:
        assert source.get_task("LGCVF-113") is None
        active = [
            item
            for item in source.plan_projection()["plans"]
            if item["status"] == "active"
        ]
    finally:
        source.close()
    assert [item["plan_cid"] for item in active] == [
        config["plan_binding"]["predecessor_plan_cid"]
    ]
    assert not paths["revision_receipts"].exists()
    recovered = materializer.steer_successor(config, successor, root=tmp_path)
    assert recovered["plan_revision_apply_receipt"]["committed"] is True
    assert materializer.verify_successor_read_only(
        config, successor, root=tmp_path
    )["valid"] is True


def test_successor_restart_recovers_partial_composite_apply_before_preview(
    tmp_path: Path,
) -> None:
    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    before = {
        key: materializer._sha256_file(paths[key])  # noqa: SLF001
        for key in ("control", "coordination", "execution", "receipt")
    }

    crashed = _crash_successor_process(
        tmp_path,
        config,
        successor,
        fault_point="after_duckdb",
    )

    assert crashed.returncode == 73, (crashed.stdout, crashed.stderr)
    assert not paths["revision_receipts"].exists()
    receipt = materializer.steer_successor(config, successor, root=tmp_path)
    assert receipt["database_sha256_before"] == before
    assert receipt["plan_revision_apply_receipt"]["committed"] is True
    assert materializer.verify_successor_read_only(
        config, successor, root=tmp_path
    )["valid"] is True


def test_successor_restart_finalizes_committed_apply_without_reapplying(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
        PlanRevisionStore,
    )

    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    crashed = _crash_successor_process(
        tmp_path,
        config,
        successor,
        fault_point="after_revision_commit_before_external_receipt",
    )

    assert crashed.returncode == 73, (crashed.stdout, crashed.stderr)
    assert not paths["revision_receipts"].exists()
    store = PlanRevisionStore(paths["revision_store"], recover=False)
    active = store.get_active()
    assert active is not None
    assert active.plan_root_cid == successor["plan_root_cid"]
    committed_store_hashes = {
        key: materializer._sha256_file(paths[key])  # noqa: SLF001
        for key in ("control", "coordination", "execution", "receipt")
    }

    receipt = materializer.steer_successor(config, successor, root=tmp_path)

    assert receipt["plan_revision_apply_receipt"]["state"] == "committed"
    assert committed_store_hashes == {
        key: materializer._sha256_file(paths[key])  # noqa: SLF001
        for key in ("control", "coordination", "execution", "receipt")
    }
    assert materializer.verify_successor_read_only(
        config, successor, root=tmp_path
    )["valid"] is True


def test_successor_rejects_malformed_predecessor_archive(tmp_path: Path) -> None:
    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    archive = json.loads(paths["predecessor_archive"].read_text(encoding="utf-8"))
    archive["metadata"]["board_namespace"] = "forged-namespace"
    paths["predecessor_archive"].write_text(json.dumps(archive), encoding="utf-8")

    with pytest.raises(materializer.MaterializationError, match="archive identity"):
        materializer.preview_successor(config, successor, root=tmp_path)


def test_successor_rejects_rewritten_predecessor_completion_evidence(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )

    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    source = DatabaseTaskSource(paths["control"], install_schema=False)
    try:
        task = source.get_task("LGCVF-001")
        assert task is not None
    finally:
        source.close()
    coordinator = DatabaseCoordinator(paths["coordination"])
    try:
        coordinator.open()
        coordinator.mark_task_complete(
            task.task_cid,
            status="succeeded",
            body={"authority": "forged-self-certification"},
        )
    finally:
        coordinator.close()

    with pytest.raises(materializer.MaterializationError, match="stale or rewritten"):
        materializer.preview_successor(config, successor, root=tmp_path)


def test_successor_read_only_verify_preserves_all_store_fingerprints(
    tmp_path: Path,
) -> None:
    config, successor = _revision_one_runtime(tmp_path)
    paths = _runtime_paths(config, tmp_path)
    materializer.steer_successor(config, successor, root=tmp_path)
    receipt_file = next(paths["revision_receipts"].glob("*.json"))
    before_databases = {
        key: (
            paths[key].stat().st_size,
            paths[key].stat().st_mtime_ns,
            materializer._sha256_file(paths[key]),  # noqa: SLF001
        )
        for key in ("control", "coordination", "execution", "receipt")
    }
    before_revision_store = materializer._directory_fingerprint(  # noqa: SLF001
        paths["revision_store"]
    )
    before_receipt = materializer._sha256_file(receipt_file)  # noqa: SLF001

    result = materializer.verify_successor_read_only(
        config, successor, root=tmp_path
    )

    assert result["stores_unchanged"] is True
    assert before_databases == {
        key: (
            paths[key].stat().st_size,
            paths[key].stat().st_mtime_ns,
            materializer._sha256_file(paths[key]),  # noqa: SLF001
        )
        for key in ("control", "coordination", "execution", "receipt")
    }
    assert before_revision_store == materializer._directory_fingerprint(  # noqa: SLF001
        paths["revision_store"]
    )
    assert before_receipt == materializer._sha256_file(receipt_file)  # noqa: SLF001
