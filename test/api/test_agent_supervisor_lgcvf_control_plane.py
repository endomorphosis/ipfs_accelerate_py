"""Focused conformance tests for the LGCVF one-writer control plane."""

from __future__ import annotations

import copy
import json
import os
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
    ConfiguredBoardError,
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
from scripts import (
    qualify_logic_governed_compositional_verification_fabric as qualifier,
)

ROOT = Path(__file__).resolve().parents[2]


def _create_sealed_datasets_repository(repository: Path) -> None:
    """Create the minimal clean nested Git repository required by recovery."""

    nested = repository / "ipfs_datasets_py"
    package = nested / "ipfs_datasets_py"
    tests = nested / "tests"
    package.mkdir(parents=True)
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tests / ".gitkeep").write_text("", encoding="utf-8")
    (nested / ".gitignore").write_text(
        "__pycache__/\n*.py[co]\n", encoding="utf-8"
    )
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "fixture@example.invalid"),
        ("config", "user.name", "LGCVF Fixture"),
        ("add", "."),
        ("commit", "-qm", "sealed datasets source"),
    ):
        subprocess.run(
            ("git", *arguments),
            cwd=nested,
            check=True,
            capture_output=True,
            text=True,
        )


def _sealed_materializer_script(tmp_path: Path) -> Path:
    """Build a minimal clean tracked checkout for exact isolated guard tests."""

    repository = tmp_path / "sealed-materializer-source"
    script = (
        repository
        / "scripts/materialize_logic_governed_compositional_verification_fabric_control_plane.py"
    )
    script.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT
        / "scripts/materialize_logic_governed_compositional_verification_fabric_control_plane.py",
        script,
    )
    config = (
        repository
        / "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json"
    )
    config.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT
        / "config/agent_supervisor_logic_governed_compositional_verification_fabric_scheduler.json",
        config,
    )
    package = repository / "ipfs_accelerate_py/agent_supervisor"
    for relative in (
        "../__init__.py",
        "__init__.py",
        "merge/__init__.py",
        "planning/__init__.py",
        "proof/__init__.py",
        "task_sources/__init__.py",
    ):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (package / "merge/database_coordination.py").write_text(
        "def read_coordination_history_projection(*args, **kwargs): return {}\n"
        "def read_coordination_registry_projection(*args, **kwargs): return {}\n",
        encoding="utf-8",
    )
    (package / "planning/formal_planning_contracts.py").write_text(
        "FormalWorkPlan = object\n", encoding="utf-8"
    )
    revision_names = (
        "CompletionAuthority",
        "DeltaEffectClass",
        "LifecycleState",
        "MergeStrategyKind",
        "PlanAuthorityRoots",
        "PlanCompletionRule",
        "PlanConflictContract",
        "PlanDelta",
        "PlanDeltaItem",
        "PlanDeltaOperation",
        "PlanLeaseContract",
        "PlanMergeStrategy",
        "PlanOrigin",
        "PlanPopulationDigest",
        "PlanProviderContract",
        "PlanResourceContract",
        "PlanRetryContract",
        "PlanRevision",
        "PlanWorktreeContract",
        "PopulationKind",
    )
    (package / "planning/plan_revision_contracts.py").write_text(
        "\n".join(f"{name} = object" for name in revision_names) + "\n",
        encoding="utf-8",
    )
    (package / "proof/formal_verification_contracts.py").write_text(
        "def content_identity(value): return 'fixture'\n", encoding="utf-8"
    )
    (package / "task_sources/intent_repository.py").write_text(
        "def task_authority_spec_cid(*args, **kwargs): return 'fixture'\n"
        "def task_projection_spec_cid(*args, **kwargs): return 'fixture'\n",
        encoding="utf-8",
    )
    (package / "task_sources/todo_vector_index.py").write_text(
        "def parse_todo_blocks(*args, **kwargs): return []\n"
        "def split_csv(*args, **kwargs): return []\n",
        encoding="utf-8",
    )
    (repository / ".gitignore").write_text(
        "__pycache__/\n*.py[co]\n", encoding="utf-8"
    )
    (repository / "test").mkdir()
    (repository / "test/.gitkeep").write_text("", encoding="utf-8")
    _create_sealed_datasets_repository(repository)
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "fixture@example.invalid"),
        ("config", "user.name", "LGCVF Fixture"),
        ("add", "."),
        ("commit", "-qm", "sealed materializer source"),
    ):
        subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    return script


@pytest.fixture(autouse=True)
def _unit_test_recovery_runtime_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core unit tests run in pytest; subprocess tests retain the real guard."""

    monkeypatch.setattr(
        materializer,
        "_require_isolated_recovery_interpreter",
        lambda: None,
    )
    monkeypatch.setattr(
        qualifier,
        "_require_isolated_recovery_runtime",
        lambda: None,
    )


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


def _fresh_recovery_population() -> tuple[dict[str, Any], dict[str, Any]]:
    """Project revision 2 against the real immutable Git commit/tree."""

    config = materializer.load_config()
    formal = FormalWorkPlan.from_dict(
        json.loads((ROOT / str(config["formal_plan_path"])).read_text(encoding="utf-8"))
    )
    source = materializer._project_source_binding(  # noqa: SLF001
        config,
        root=ROOT,
        require_clean=False,
    )
    population = materializer.project_population(
        config,
        formal_plan=formal,
        todo_text=(ROOT / str(config["taskboard_path"])).read_text(encoding="utf-8"),
        source={
            "accelerator_head": source["accelerator_head"],
            "accelerator_tree": source["accelerator_tree"],
            "source_forest_root": source["source_forest_root"],
        },
    )
    return config, population


def _fake_recovery_qualification(preview: dict[str, Any]) -> dict[str, Any]:
    """Test-only corroboration; production exposes no receipt injection API."""

    suites: list[dict[str, Any]] = []
    for evidence in preview["merge_completion_evidence"]:
        observation = {
            "task_id": evidence["task_id"],
            "task_cid": evidence["task_cid"],
            "validation_spec": evidence["validation_spec"],
        }
        observation["observation_cid"] = materializer.content_identity(observation)
        suites.append(observation)
    omission = {
        "schema": "lgcvf-recovery-validation-projection-omission@1",
        "accelerator_head": preview["source_head"],
        "accelerator_tree": preview["source_tree"],
        "datasets_gitlink": "fixture-datasets-gitlink",
        "datasets_tree": "fixture-datasets-tree",
        "omitted_source_symlinks": [],
    }
    omission["commitment_cid"] = materializer.content_identity(omission)
    projection_evidence = {
        "schema": "lgcvf-recovery-validation-projection-evidence@1",
        "source_binding_cid": "fixture-source-binding-cid",
        "omission_root": omission["commitment_cid"],
        "ordered_suites": [],
    }
    projection_evidence["commitment_cid"] = materializer.content_identity(
        projection_evidence
    )
    receipt = {
        "schema": "test-only-recovery-qualification",
        "suites": suites,
        "validation_projection_omission_commitment": omission,
        "validation_projection_omission_root": omission["commitment_cid"],
        "validation_projection_evidence_commitment": projection_evidence,
        "validation_projection_evidence_root": projection_evidence[
            "commitment_cid"
        ],
    }
    receipt["receipt_cid"] = materializer.content_identity(receipt)
    return receipt


def _patch_recovery_judge(
    monkeypatch: pytest.MonkeyPatch,
    population: dict[str, Any],
    qualification: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        materializer,
        "_require_clean_recovery_source",
        lambda *_args, **_kwargs: (
            population["source_head"],
            str(population["repository_tree_id"]).removeprefix("git-tree:"),
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_run_and_verify_recovery_qualification",
        lambda **_kwargs: copy.deepcopy(qualification),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_recovery_qualification",
        lambda value, **_kwargs: copy.deepcopy(dict(value)),
    )


def _patch_recovery_qualification_only(
    monkeypatch: pytest.MonkeyPatch,
    qualification: dict[str, Any],
) -> None:
    """Replace only the protected judge; retain the real Git clean check."""

    monkeypatch.setattr(
        materializer,
        "_run_and_verify_recovery_qualification",
        lambda **_kwargs: copy.deepcopy(qualification),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_recovery_qualification",
        lambda value, **_kwargs: copy.deepcopy(dict(value)),
    )


def _fresh_recovery_staging_container(
    config: dict[str, Any], root: Path
) -> Path:
    target = materializer._fresh_recovery_paths(config, root=root)["target"]  # noqa: SLF001
    return target.with_name(f"{target.name}-fresh-recovery-staging")


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
    runtime_config.pop("fresh_generation_recovery")
    runtime_config["runtime_paths"].update(
        {
            "root": "run",
            "state": "run/state",
            "worktrees": "run/worktrees",
            "merge_queue": "run/merge-queue",
            "logs": "run/logs",
            "evidence": "run/evidence",
        }
    )
    runtime_config["database_program"].update(
        {
            "store_generation": "lgcvf-successor-test",
            "export_profile": "lgcvf-successor-test",
            "event_store_path": "run/events",
            "runtime_registry_path": "run/registry",
            "worktree_root": "run/worktrees",
        }
    )
    runtime_config["database_program"]["store_id"] = "run/control.duckdb"
    archive_target = (
        tmp_path
        / str(runtime_config["formal_plan_path"])
    ).parent / "plan_revisions" / f"{predecessor.content_id}.json"
    archive_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_path, archive_target)
    materializer._materialize_canonical(  # noqa: SLF001
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
from scripts import (
    materialize_logic_governed_compositional_verification_fabric_control_plane
    as materializer,
)

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
    assert config["provider"]["provider_id"] == "grok_cli"
    with pytest.raises(
        ConfiguredBoardError,
        match="fresh-generation recovery initial launch admission failed",
    ):
        # The authoritative checkout deliberately has no run-v17 yet.  The
        # scheduler must fail before it renders an executable launch plan.
        configured_board_launch_plan(
            board,
            implement=True,
            detach=False,
        )
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

    receipt = materializer._materialize_canonical(  # noqa: SLF001
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

    with pytest.raises(
        materializer.MaterializationError,
        match="generic verification is not admission authority",
    ):
        materializer.verify_read_only(
            temporary,
            population,
            root=tmp_path,
            expected_stage="live",
        )
    live = materializer._verify_read_only_core(  # noqa: SLF001
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
        materializer._materialize_canonical(  # noqa: SLF001
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


def test_fresh_recovery_preview_is_no_write_and_rejects_quarantine_drift(
    tmp_path: Path,
) -> None:
    config, population = _fresh_recovery_population()
    with pytest.raises(materializer.MaterializationError, match="ambiguous JSON"):
        materializer._decode_evidence_json(  # noqa: SLF001
            b'{"task_id":"LGCVF-080","task_id":"LGCVF-081"}',
            noun="pinned recovery evidence",
        )
    forensic_root = (
        ROOT
        / "data/agent_supervisor/logic_governed_compositional_verification_fabric/run-v16"
    )

    def forensic_inventory() -> dict[str, tuple[Any, ...]]:
        result: dict[str, tuple[Any, ...]] = {}
        for path in sorted(forensic_root.rglob("*")):
            status = path.lstat()
            relative = path.relative_to(forensic_root).as_posix()
            if path.is_symlink():
                result[relative] = ("symlink", status.st_mode, os.readlink(path))
            elif path.is_dir():
                result[relative] = ("directory", status.st_mode)
            elif path.is_file():
                result[relative] = (
                    "file",
                    status.st_mode,
                    status.st_size,
                    status.st_mtime_ns,
                    materializer._sha256_file(path),
                )
            else:
                result[relative] = ("special", status.st_mode)
        return result

    before = forensic_inventory()

    preview = materializer.preview_fresh_generation_recovery(
        config, population, root=tmp_path, source_root=ROOT
    )
    policy = config["fresh_generation_recovery"]
    assert preview["schema"] == materializer.FRESH_RECOVERY_PREVIEW_SCHEMA
    assert preview["duckdb_runtime_cid"] == policy["duckdb_runtime_cid"]
    assert preview["verification_python_executable"] == str(
        Path(sys.executable).resolve(strict=True)
    )
    assert preview["verification_python_executable_sha256"] == (
        policy["verification_python_executable_sha256"]
    )
    wrong_runtime = copy.deepcopy(config)
    wrong_runtime["fresh_generation_recovery"]["duckdb_runtime_cid"] = (
        "baguqeera-wrong-runtime"
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="DuckDB runtime differs from configuration",
    ):
        materializer._require_bound_duckdb_runtime_policy(wrong_runtime)  # noqa: SLF001
    wrong_interpreter = copy.deepcopy(config)
    wrong_interpreter["fresh_generation_recovery"][
        "verification_python_executable_sha256"
    ] = "sha256:" + "0" * 64
    with pytest.raises(
        materializer.MaterializationError,
        match="verification interpreter differs from configuration",
    ):
        materializer._require_bound_duckdb_runtime_policy(  # noqa: SLF001
            wrong_interpreter
        )
    forest_poison = copy.deepcopy(population)
    forest_poison.pop("population_root")
    forest_poison["source_forest_root"] = "baguqeera-forged-source-forest"
    forest_poison["population_root"] = materializer.content_identity(forest_poison)
    with pytest.raises(
        materializer.MaterializationError,
        match="population differs from canonical source projection",
    ):
        materializer.preview_fresh_generation_recovery(
            config,
            forest_poison,
            root=tmp_path,
            source_root=ROOT,
        )

    with pytest.raises(
        materializer.MaterializationError,
        match="canonical-only materialization is not admissible",
    ):
        materializer.materialize(
            config, population, root=tmp_path, recheck_source=False
        )

    stripped = copy.deepcopy(config)
    stripped.pop("fresh_generation_recovery")
    with pytest.raises(
        materializer.MaterializationError,
        match="canonical-only materialization is not admissible",
    ):
        materializer.materialize(
            stripped, population, root=tmp_path, recheck_source=False
        )
    with pytest.raises(
        materializer.MaterializationError,
        match="generic verification is not admission authority",
    ):
        materializer.verify_read_only(
            stripped, population, root=tmp_path, expected_stage="live"
        )

    protected_target = tmp_path / Path(
        str(config["fresh_generation_recovery"]["target_runtime_root"])
    )
    protected_target.parent.mkdir(parents=True, exist_ok=True)
    alias = tmp_path / "ordinary-runtime-alias"
    os.symlink(protected_target.relative_to(tmp_path), alias)
    aliased = copy.deepcopy(stripped)
    aliased["runtime_paths"].update(
        {
            "root": alias.name,
            "state": f"{alias.name}/state",
            "worktrees": f"{alias.name}/worktrees",
            "merge_queue": f"{alias.name}/merge-queue",
            "logs": f"{alias.name}/logs",
            "evidence": f"{alias.name}/evidence",
        }
    )
    aliased["database_program"].update(
        {
            "store_generation": "ordinary-generation",
            "export_profile": "ordinary-profile",
            "store_id": f"{alias.name}/control.duckdb",
            "event_store_path": f"{alias.name}/events",
            "runtime_registry_path": f"{alias.name}/registry",
            "worktree_root": f"{alias.name}/worktrees",
        }
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="canonical-only materialization is not admissible",
    ):
        materializer.materialize(
            aliased, population, root=tmp_path, recheck_source=False
        )
    with pytest.raises(
        materializer.MaterializationError,
        match="generic verification is not admission authority",
    ):
        materializer.verify_read_only(
            aliased, population, root=tmp_path, expected_stage="live"
        )
    for operation in (
        materializer.preview_successor,
        materializer.steer_successor,
        materializer.verify_successor_read_only,
    ):
        with pytest.raises(
            materializer.MaterializationError,
            match="reject legacy successor",
        ):
            operation(aliased, population, root=tmp_path)
    assert alias.is_symlink()
    assert not protected_target.exists()

    assert preview["write_performed"] is False
    assert preview["target_state"] == "absent"
    assert preview["completion_partition"]["completed_count"] == 13
    assert preview["completion_partition"]["todo_count"] == 13
    assert preview["completion_partition"]["blocked_count"] == 2
    assert not materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=tmp_path
    )["target"].exists()
    assert before == forensic_inventory()

    drifted = copy.deepcopy(config)
    drifted["fresh_generation_recovery"][
        "contaminated_coordination_rejected_record_set_cid"
    ] = "baguqeera-forged"
    with pytest.raises(
        materializer.MaterializationError,
        match="configuration differs from the canonical profile",
    ):
        materializer.preview_fresh_generation_recovery(
            drifted, population, root=tmp_path, source_root=ROOT
        )


def test_generic_verify_cli_routes_to_strict_fresh_recovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, population = _population()
    monkeypatch.setattr(materializer, "load_config", lambda: config)
    monkeypatch.setattr(materializer, "build_population", lambda _config: population)
    monkeypatch.setattr(
        materializer,
        "verify_fresh_generation_recovery",
        lambda *_args, **_kwargs: {"schema": "strict-recovery-test", "valid": True},
    )
    monkeypatch.setattr(
        materializer,
        "verify_read_only",
        lambda *_args, **_kwargs: pytest.fail("generic verifier must not admit run-v17"),
    )

    assert materializer.main(["verify"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "strict-recovery-test"


def test_recovery_direct_api_and_cli_require_isolated_python(tmp_path: Path) -> None:
    script = _sealed_materializer_script(tmp_path)
    probe = r'''
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

script = Path(sys.argv[1])
specification = importlib.util.spec_from_file_location("lgcvf_materializer_guard", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
if len(sys.argv) > 2 and sys.argv[2] == "spoof-write-bytecode":
    sys.dont_write_bytecode = True
elif len(sys.argv) > 2 and sys.argv[2] == "spoof-all-python-flags":
    sys.dont_write_bytecode = True
    sys.flags = SimpleNamespace(
        isolated=1,
        ignore_environment=1,
        no_site=1,
        safe_path=True,
        dont_write_bytecode=1,
    )
operations = (
    module.preview_fresh_generation_recovery,
    module.verify_fresh_generation_recovery.__wrapped__,
    module.materialize_fresh_generation_recovery.__wrapped__,
)
errors = []
for operation in operations:
    try:
        operation({}, {})
    except module.MaterializationError as exc:
        errors.append(str(exc))
    else:
        raise SystemExit("empty recovery config was admitted")
print(json.dumps(errors))
'''
    base_environment = {"HOME": str(ROOT), "LANG": "C", "PATH": "/usr/bin:/bin"}
    ordinary_api = subprocess.run(
        (sys.executable, "-c", probe, str(script)),
        cwd=ROOT,
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert ordinary_api.returncode == 0, ordinary_api.stderr
    assert json.loads(ordinary_api.stdout) == [
        "protected recovery requires python -I -S -B"
    ] * 3

    missing_b_spoof = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-c",
            probe,
            str(script),
            "spoof-write-bytecode",
        ),
        cwd=ROOT,
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert missing_b_spoof.returncode == 0, missing_b_spoof.stderr
    assert json.loads(missing_b_spoof.stdout) == [
        "protected recovery requires python -I -S -B"
    ] * 3

    replaced_flags = subprocess.run(
        (
            sys.executable,
            "-c",
            probe,
            str(script),
            "spoof-all-python-flags",
        ),
        cwd=ROOT,
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert replaced_flags.returncode == 0, replaced_flags.stderr
    assert json.loads(replaced_flags.stdout) == [
        "protected recovery requires python -I -S -B"
    ] * 3

    isolated_api = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", probe, str(script)),
        cwd=ROOT,
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert isolated_api.returncode == 0, isolated_api.stderr
    assert all(
        "protected recovery requires" not in error
        for error in json.loads(isolated_api.stdout)
    )

    ordinary_cli = subprocess.run(
        (sys.executable, str(script), "recovery-preview"),
        cwd=ROOT,
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert ordinary_cli.returncode == 2, ordinary_cli.stderr
    ordinary_result = json.loads(ordinary_cli.stdout)
    assert ordinary_result["error"] == "protected recovery requires python -I -S -B"

    isolated_cli = subprocess.run(
        (
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(script),
            "recovery-preview",
        ),
        cwd=ROOT,
        env=base_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    isolated_result = json.loads(isolated_cli.stdout)
    assert "protected recovery requires" not in str(isolated_result.get("error", ""))


@pytest.mark.parametrize("index_flag", ("--skip-worktree", "--assume-unchanged"))
def test_recovery_direct_entry_binds_config_to_ordinary_head_bytes(
    tmp_path: Path,
    index_flag: str,
) -> None:
    relative = (
        "config/agent_supervisor_logic_governed_compositional_verification_fabric_"
        "scheduler.json"
    )
    environment = {"HOME": str(tmp_path), "LANG": "C", "PATH": "/usr/bin:/bin"}
    api_script = _sealed_materializer_script(tmp_path / "api")
    api_repository = api_script.parents[1]
    api_probe = r'''
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

script = Path(sys.argv[1])
relative = sys.argv[2]
index_flag = sys.argv[3]
specification = importlib.util.spec_from_file_location("guard", script)
module = importlib.util.module_from_spec(specification)
sys.modules[specification.name] = module
specification.loader.exec_module(module)
subprocess.run(
    ("/usr/bin/git", "update-index", index_flag, "--", relative),
    cwd=script.parents[1],
    check=True,
)
config = script.parents[1] / relative
config.write_bytes(config.read_bytes() + b"\n")
try:
    module.preview_fresh_generation_recovery({}, {})
except module.MaterializationError as exc:
    print(json.dumps({"error": str(exc)}))
else:
    raise SystemExit("exceptional-index config was admitted")
'''
    direct = subprocess.run(
        (
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            "-c",
            api_probe,
            str(api_script),
            relative,
            index_flag,
        ),
        cwd=api_repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert direct.returncode == 0, direct.stderr
    assert json.loads(direct.stdout) == {
        "error": "protected recovery configuration differs from HEAD"
    }
    assert not (
        api_repository
        / "data/agent_supervisor/logic_governed_compositional_verification_fabric/run-v17"
    ).exists()

    cli_script = _sealed_materializer_script(tmp_path / "cli")
    cli_repository = cli_script.parents[1]
    cli_config = cli_repository / relative
    before_inventory = sorted(
        path.relative_to(cli_repository).as_posix()
        for path in cli_repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    subprocess.run(
        ("/usr/bin/git", "update-index", index_flag, "--", relative),
        cwd=cli_repository,
        check=True,
        capture_output=True,
        text=True,
    )
    cli_config.write_bytes(cli_config.read_bytes() + b"\n")
    cli = subprocess.run(
        (
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            str(cli_script),
            "recovery-preview",
        ),
        cwd=cli_repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert cli.returncode != 0
    assert "protected recovery configuration" in cli.stderr
    assert not (
        cli_repository
        / "data/agent_supervisor/logic_governed_compositional_verification_fabric/run-v17"
    ).exists()
    assert sorted(
        path.relative_to(cli_repository).as_posix()
        for path in cli_repository.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ) == before_inventory


@pytest.mark.parametrize("repository_scope", ("accelerator", "datasets"))
@pytest.mark.parametrize("substitution", ("grafts", "replace"))
def test_recovery_direct_entry_rejects_git_object_substitution(
    tmp_path: Path,
    repository_scope: str,
    substitution: str,
) -> None:
    script = _sealed_materializer_script(tmp_path)
    repository = script.parents[1]
    selected = (
        repository if repository_scope == "accelerator" else repository / "ipfs_datasets_py"
    )
    common = Path(
        subprocess.run(
            (
                "/usr/bin/git",
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if substitution == "grafts":
        head = subprocess.run(
            ("/usr/bin/git", "rev-parse", "HEAD"),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        grafts = common / "info/grafts"
        grafts.parent.mkdir(parents=True, exist_ok=True)
        grafts.write_text(head + "\n", encoding="utf-8")
    else:
        marker = selected / "replacement-parent-fixture"
        marker.write_text("replacement parent\n", encoding="utf-8")
        subprocess.run(
            ("/usr/bin/git", "add", marker.name),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ("/usr/bin/git", "commit", "-qm", "replacement parent fixture"),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        )
        head = subprocess.run(
            ("/usr/bin/git", "rev-parse", "HEAD"),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        parent = subprocess.run(
            ("/usr/bin/git", "rev-parse", "HEAD^"),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if repository_scope == "datasets":
            subprocess.run(
                ("/usr/bin/git", "add", "ipfs_datasets_py"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ("/usr/bin/git", "commit", "-qm", "bind nested replacement fixture"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
        subprocess.run(
            ("/usr/bin/git", "update-ref", f"refs/replace/{head}", parent),
            cwd=selected,
            check=True,
            capture_output=True,
            text=True,
        )
    with pytest.raises(RuntimeError, match="(?:object substitution|replacement refs)"):
        materializer._git_object_substitution_state(selected)  # noqa: SLF001

    completed = subprocess.run(
        (
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            str(script),
            "recovery-preview",
        ),
        cwd=repository,
        env={"HOME": str(repository), "LANG": "C", "PATH": "/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "protected recovery" in completed.stderr
    assert not (
        repository
        / "data/agent_supervisor/logic_governed_compositional_verification_fabric/run-v17"
    ).exists()


@pytest.mark.parametrize("repository_scope", ("accelerator", "datasets"))
@pytest.mark.parametrize("attack", ("fsmonitor", "filter", "same-size-mtime"))
def test_recovery_materializer_git_observations_are_raw_head_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository_scope: str,
    attack: str,
) -> None:
    def git(repository: Path, *arguments: str) -> str:
        return subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    outer = tmp_path / "accelerator"
    outer_source = outer / "package/source.py"
    outer_source.parent.mkdir(parents=True)
    outer_source.write_bytes(b"VALUE = 'AAAA'\n")
    nested = outer / "ipfs_datasets_py"
    nested_source = nested / "package/source.py"
    nested_source.parent.mkdir(parents=True)
    nested_source.write_bytes(b"VALUE = 'AAAA'\n")
    for repository in (nested, outer):
        git(repository, "init", "-q")
        git(repository, "config", "user.email", "fixture@example.invalid")
        git(repository, "config", "user.name", "LGCVF Fixture")
        git(repository, "add", ".")
        git(repository, "commit", "-qm", "sealed raw Git fixture")

    monkeypatch.setattr(
        materializer,
        "_ISOLATED_RECOVERY_PYCACHE_DIRECTORY",
        object(),
    )
    selected = outer if repository_scope == "accelerator" else nested
    source = outer_source if repository_scope == "accelerator" else nested_source
    original = source.stat()
    source.write_bytes(b"VALUE = 'BBBB'\n")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    if attack == "same-size-mtime":
        with pytest.raises(RuntimeError, match="differs from HEAD"):
            materializer._scan_isolated_recovery_import_roots(  # noqa: SLF001
                selected,
                roots=("package",),
                tracked_pathspecs=("package",),
                root_import_candidates=False,
            )
        return

    common = Path(
        git(
            selected,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    )
    marker = tmp_path / f"{repository_scope}-{attack}-executed"
    hook = common / f"{attack}-hook"
    hook.write_text(
        "#!/bin/sh\n"
        f"touch {str(marker)!r}\n"
        "cat\n",
        encoding="utf-8",
    )
    hook.chmod(0o700)
    if attack == "fsmonitor":
        git(selected, "config", "core.fsmonitor", str(hook))
        with pytest.raises(RuntimeError, match="source is not clean"):
            materializer._clean_recovery_import_source(selected)  # noqa: SLF001
    else:
        git(selected, "config", "filter.lgcvf-evil.clean", str(hook))
        attributes = common / "info/attributes"
        attributes.parent.mkdir(parents=True, exist_ok=True)
        attributes.write_text("*.py filter=lgcvf-evil\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="(?:substitution|filter drivers)"):
            materializer._git_object_substitution_state(selected)  # noqa: SLF001
    assert marker.exists() is False


@pytest.mark.parametrize("authority", ("formal_plan_path", "taskboard_path"))
def test_recovery_population_authority_rejects_raw_blob_drift(
    tmp_path: Path,
    authority: str,
) -> None:
    repository = tmp_path / "authority"
    repository.mkdir()
    relative = {
        "formal_plan_path": "formal.json",
        "taskboard_path": "taskboard.md",
    }[authority]
    path = repository / relative
    path.write_bytes(b"trusted-authority-bytes\n")
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "fixture@example.invalid"),
        ("config", "user.name", "LGCVF Fixture"),
        ("add", "."),
        ("commit", "-qm", "sealed authority fixture"),
    ):
        subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    _path, payload, _digest = materializer._read_regular_evidence_bytes(  # noqa: SLF001
        repository,
        relative,
        field=authority,
    )
    materializer._require_head_bound_recovery_bytes(  # noqa: SLF001
        repository,
        relative,
        payload,
        field=authority,
    )
    original = path.stat()
    changed = bytearray(payload)
    changed[0] = ord("T")
    path.write_bytes(changed)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    _path, changed_payload, _digest = materializer._read_regular_evidence_bytes(  # noqa: SLF001
        repository,
        relative,
        field=authority,
    )
    with pytest.raises(materializer.MaterializationError, match="index and HEAD"):
        materializer._require_head_bound_recovery_bytes(  # noqa: SLF001
            repository,
            relative,
            changed_payload,
            field=authority,
        )


def test_protected_generation_rejects_legacy_successor_routes_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, population = _fresh_recovery_population()
    target = materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=tmp_path
    )["target"]
    authoritative_target = materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=ROOT
    )["target"]
    authoritative_before = (
        materializer._directory_fingerprint(  # noqa: SLF001
            authoritative_target,
            require_private=True,
        )
        if authoritative_target.exists()
        else None
    )

    for operation in (
        materializer.preview_successor,
        materializer.steer_successor,
        materializer.verify_successor_read_only,
    ):
        with pytest.raises(
            materializer.MaterializationError,
            match="reject legacy successor",
        ):
            operation(config, population, root=tmp_path)
        assert not target.exists()
        assert not target.parent.exists()

    monkeypatch.setattr(materializer, "load_config", lambda: config)
    monkeypatch.setattr(
        materializer, "build_population", lambda _config: population
    )
    for command in ("successor-preview", "successor-steer", "successor-verify"):
        assert materializer.main([command]) == 2
        result = json.loads(capsys.readouterr().out)
        assert result["valid"] is False
        assert "reject legacy successor" in result["error"]
    assert authoritative_target.exists() is (authoritative_before is not None)
    if authoritative_before is not None:
        assert materializer._directory_fingerprint(  # noqa: SLF001
            authoritative_target,
            require_private=True,
        ) == authoritative_before


def test_fresh_recovery_atomic_idempotent_strict_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, population = _fresh_recovery_population()
    preview = materializer.preview_fresh_generation_recovery(
        config, population, root=tmp_path, source_root=ROOT
    )
    qualification = _fake_recovery_qualification(preview)
    _patch_recovery_judge(monkeypatch, population, qualification)

    receipt = materializer.materialize_fresh_generation_recovery(
        config, population, root=tmp_path, source_root=ROOT
    )
    paths = materializer._fresh_recovery_paths(config, root=tmp_path)  # noqa: SLF001
    store_paths = {
        key: path
        for key, path in paths.items()
        if key in {"control", "coordination", "execution", "receipt", "recovery_receipt"}
    }
    before_verify = {
        key: (path.stat().st_size, path.stat().st_mtime_ns, materializer._sha256_file(path))
        for key, path in store_paths.items()
    }
    report = materializer.verify_fresh_generation_recovery(
        config, population, root=tmp_path, source_root=ROOT
    )
    replay = materializer.materialize_fresh_generation_recovery(
        config, population, root=tmp_path, source_root=ROOT
    )

    assert replay["receipt_cid"] == receipt["receipt_cid"]
    assert receipt["schema"] == materializer.FRESH_RECOVERY_RECEIPT_SCHEMA
    assert report["schema"] == materializer.FRESH_RECOVERY_VERIFICATION_SCHEMA
    assert report["duckdb_runtime_cid"] == config["fresh_generation_recovery"][
        "duckdb_runtime_cid"
    ]
    assert report["validation_projection_omission_commitment"] == qualification[
        "validation_projection_omission_commitment"
    ]
    assert report["validation_projection_omission_root"] == qualification[
        "validation_projection_omission_root"
    ]
    assert report["validation_projection_evidence_commitment"] == qualification[
        "validation_projection_evidence_commitment"
    ]
    assert report["validation_projection_evidence_root"] == qualification[
        "validation_projection_evidence_root"
    ]
    assert (report["completed_count"], report["todo_count"], report["blocked_count"]) == (
        13,
        13,
        2,
    )
    assert report["ready_task_ids"] == ["LGCVF-081"]
    assert before_verify == {
        key: (path.stat().st_size, path.stat().st_mtime_ns, materializer._sha256_file(path))
        for key, path in store_paths.items()
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="generic verification is not admission authority",
    ):
        materializer.verify_read_only(
            config, population, root=tmp_path, expected_stage="live"
        )
    operational = materializer._verify_read_only_core(  # noqa: SLF001
        config, population, root=tmp_path, expected_stage="live"
    )
    observations = {
        item["task_id"]: item["observation_cid"] for item in qualification["suites"]
    }
    for evidence in operational["control"]["evidence"]:
        alias = evidence["body"]["request_id"]
        task_id = next(
            item["task_id"]
            for item in preview["merge_completion_evidence"]
            if item["request_id"] == alias
        )
        assert evidence["digest"] != observations[task_id]
        assert evidence["body"]["validation_observation_cid"] == observations[task_id]
    for completion in operational["control"]["completion_receipts"]:
        digests = completion["body"]["evidence_digests"]
        assert len(digests) == 1
        assert digests[0] not in observations.values()

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )

    coordination_backup = tmp_path / "coordination-before-history.duckdb"
    shutil.copy2(paths["coordination"], coordination_backup)
    coordinator = DatabaseCoordinator(paths["coordination"])
    try:
        coordinator.open()
        lease = coordinator.acquire(
            lease_kind="merge",
            scope="fresh-recovery-adversarial-history",
            owner_session_id="test:foreign-history",
            lease_ms=30_000,
        )
        coordinator.release(lease, reason="closed but still contaminating history")
    finally:
        coordinator.close()
    with pytest.raises(
        materializer.MaterializationError,
        match="coordination (?:projection differs|history is not empty)",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    shutil.copy2(coordination_backup, paths["coordination"])
    assert materializer.verify_fresh_generation_recovery(
        config, population, root=tmp_path, source_root=ROOT
    )["valid"] is True

    state = tmp_path / Path(str(config["runtime_paths"]["state"]))
    revision_store = state / "plan-revision-store"
    revision_store.mkdir(parents=True)
    state.chmod(0o700)
    revision_store.chmod(0o700)
    with pytest.raises(materializer.MaterializationError, match="state directory is not empty"):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    revision_store.rmdir()
    state.rmdir()

    paths["target"].chmod(0o555)
    with pytest.raises(
        materializer.MaterializationError,
        match="directory is not owner-accessible",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    paths["target"].chmod(0o700)

    paths["control"].chmod(0o444)
    with pytest.raises(
        materializer.MaterializationError,
        match="file is not owner-accessible",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    paths["control"].chmod(0o600)

    paths["control"].chmod(0o666)
    with pytest.raises(materializer.MaterializationError, match="permissions differ"):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    paths["control"].chmod(0o600)

    control_backup = tmp_path / "control-before-authority-tamper.duckdb"
    shutil.copy2(paths["control"], control_backup)

    def restore_control() -> None:
        shutil.copy2(control_backup, paths["control"])

    import duckdb  # type: ignore

    connection = duckdb.connect(str(paths["control"]))
    try:
        output_row = connection.execute(
            "SELECT task_cid, ordinal, effect_json FROM task_outputs "
            "ORDER BY task_cid, ordinal LIMIT 1"
        ).fetchone()
        assert output_row is not None
        raw_effect = str(output_row[2])
        decoded_effect = json.loads(raw_effect)
        effect_key = next(iter(decoded_effect))
        forged_effect = (
            "{"
            + json.dumps(effect_key)
            + ":\"forged-first-value\","
            + raw_effect[1:]
        )
        connection.execute(
            "UPDATE task_outputs SET effect_json = ? "
            "WHERE task_cid = ? AND ordinal = ?",
            [forged_effect, str(output_row[0]), int(output_row[1])],
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MaterializationError,
        match="control typed authority contains invalid JSON",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    connection = duckdb.connect(str(paths["control"]))
    try:
        connection.execute(
            "UPDATE tasks SET extension_json = '' WHERE task_alias = 'LGCVF-090'"
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MaterializationError,
        match="control typed authority contains invalid JSON",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    connection = duckdb.connect(str(paths["control"]))
    try:
        connection.execute(
            "UPDATE control_plane_metadata SET value = ? WHERE key = 'database_uuid'",
            ["00000000-0000-5000-8000-000000000000"],
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MaterializationError,
        match="control residual content differs",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    connection = duckdb.connect(str(paths["control"]))
    try:
        connection.execute(
            "CREATE INDEX forged_schema_contract_index "
            "ON schema_contracts(interface_name)"
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MaterializationError,
        match="control (?:schema|catalog) differs",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    baseline_catalog = materializer._read_only_duckdb_catalog(  # noqa: SLF001
        paths["control"], noun="baseline control"
    )
    connection = duckdb.connect(str(paths["control"]))
    try:
        index_name = str(
            connection.execute(
                "SELECT index_name FROM duckdb_indexes() "
                "WHERE database_name = current_database() "
                "ORDER BY index_name LIMIT 1"
            ).fetchone()[0]
        )
        quoted_index = '"' + index_name.replace('"', '""') + '"'
        connection.execute(
            f"COMMENT ON INDEX {quoted_index} IS 'forged authority comment'"
        )
    finally:
        connection.close()
    commented_catalog = materializer._read_only_duckdb_catalog(  # noqa: SLF001
        paths["control"], noun="commented control"
    )
    assert commented_catalog["catalog_root"] != baseline_catalog["catalog_root"]
    assert any(
        item["name"] == index_name
        and item["comment"] == "forged authority comment"
        for item in commented_catalog["indexes"]
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="control (?:schema|catalog) differs",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    connection = duckdb.connect(str(paths["control"]))
    try:
        connection.execute("CREATE SCHEMA hidden_authority")
        connection.execute(
            "CREATE TABLE hidden_authority.contaminated(value INTEGER)"
        )
        connection.execute("CREATE MACRO hidden_macro(value) AS value + 1")
        connection.execute("CREATE SEQUENCE hidden_sequence")
        connection.execute(
            "CREATE TYPE hidden_authority_state AS ENUM ('present', 'absent')"
        )
    finally:
        connection.close()
    hidden_catalog = materializer._read_only_duckdb_catalog(  # noqa: SLF001
        paths["control"], noun="adversarial control"
    )
    assert {item["name"] for item in hidden_catalog["schemas"]} >= {
        "hidden_authority"
    }
    assert {
        (item["schema"], item["name"]) for item in hidden_catalog["tables"]
    } >= {("hidden_authority", "contaminated")}
    assert {item["name"] for item in hidden_catalog["macros"]} >= {
        "hidden_macro"
    }
    assert {item["name"] for item in hidden_catalog["sequences"]} >= {
        "hidden_sequence"
    }
    assert {item["name"] for item in hidden_catalog["types"]} >= {
        "hidden_authority_state"
    }
    with pytest.raises(
        materializer.MaterializationError,
        match="control (?:schema|catalog) differs",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    connection = duckdb.connect(str(paths["control"]))
    try:
        connection.execute(
            "UPDATE schema_migrations SET receipt_cid = ? WHERE version = 1",
            ["baguqeera-forged-migration-receipt"],
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MaterializationError,
        match=(
            "schema_(?:migrations receipt identity|migration_attempts authority) "
            "differs"
        ),
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    connection = duckdb.connect(str(paths["control"]))
    try:
        event_row = connection.execute(
            "SELECT event_id, stream_id, sequence, global_sequence, event_type, body_json "
            "FROM domain_events WHERE event_type = 'intent.goal_edge_linked' "
            "ORDER BY global_sequence LIMIT 1"
        ).fetchone()
        wrapper = json.loads(str(event_row[5]))
        wrapper["body"]["annotation"] = {"updated_at": "forged-nested-clock"}
        event_id = materializer.content_identity(
            {
                "stream_id": str(event_row[1]),
                "sequence": int(event_row[2]),
                "global_sequence": int(event_row[3]),
                "event_type": str(event_row[4]),
                "body": wrapper,
            }
        )
        connection.execute(
            "UPDATE domain_events SET event_id = ?, body_json = ? WHERE event_id = ?",
            [
                event_id,
                json.dumps(wrapper, sort_keys=True, separators=(",", ":")),
                str(event_row[0]),
            ],
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MaterializationError,
        match="semantic event stream differs",
    ):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    restore_control()

    hardlink_origin = tmp_path / "control-hardlink-origin.duckdb"
    paths["control"].rename(hardlink_origin)
    os.link(hardlink_origin, paths["control"])
    with pytest.raises(materializer.MaterializationError, match="file identity differs"):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    paths["control"].unlink()
    hardlink_origin.rename(paths["control"])

    tampered = json.loads(paths["recovery_receipt"].read_text(encoding="utf-8"))
    tampered["unknown_authority"] = True
    tampered.pop("receipt_cid")
    tampered["receipt_cid"] = materializer.content_identity(tampered)
    materializer._atomic_write_json(paths["recovery_receipt"], tampered)  # noqa: SLF001
    with pytest.raises(materializer.MaterializationError, match="receipt fields differ"):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )

    materializer._atomic_write_json(paths["recovery_receipt"], receipt)  # noqa: SLF001
    receipt_bytes = paths["recovery_receipt"].read_bytes()
    receipt_schema = f'"schema":"{receipt["schema"]}"'.encode()
    assert receipt_schema in receipt_bytes
    paths["recovery_receipt"].write_bytes(
        receipt_bytes.replace(
            receipt_schema, receipt_schema + b"," + receipt_schema, 1
        )
    )
    with pytest.raises(materializer.MaterializationError, match="ambiguous JSON"):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    paths["recovery_receipt"].write_bytes(receipt_bytes)

    manifest_path = paths["recovery"] / f'{receipt["manifest_cid"]}.manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_schema = f'"schema":"{manifest["schema"]}"'.encode()
    assert manifest_schema in manifest_bytes
    manifest_path.write_bytes(
        manifest_bytes.replace(
            manifest_schema, manifest_schema + b"," + manifest_schema, 1
        )
    )
    with pytest.raises(materializer.MaterializationError, match="ambiguous JSON"):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )
    manifest_path.write_bytes(manifest_bytes)

    evidence_dir = Path(str(config["runtime_paths"]["evidence"]))
    lexical_evidence = tmp_path / evidence_dir
    redirected = lexical_evidence.with_name("evidence.redirected")
    lexical_evidence.rename(redirected)
    os.symlink(redirected.name, lexical_evidence)
    with pytest.raises(materializer.MaterializationError, match="contains a symlink"):
        materializer.verify_fresh_generation_recovery(
            config, population, root=tmp_path, source_root=ROOT
        )


def test_fresh_recovery_real_clean_checkout_keeps_stage_ignored_and_rechecks_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real Git clean check around the ignored atomic stage."""

    config, population = _fresh_recovery_population()
    clean_root = tmp_path / "clean-checkout"
    git_env = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
    }

    def git(*argv: str, cwd: Path = tmp_path) -> str:
        completed = subprocess.run(
            ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", *argv],
            cwd=cwd,
            env=git_env,
            capture_output=True,
            check=True,
            text=True,
            timeout=120,
        )
        return completed.stdout.strip()

    git("clone", "--shared", "--no-checkout", str(ROOT), str(clean_root))
    git(
        "checkout",
        "-B",
        str(config["source_binding"]["accelerator_required_branch"]),
        population["source_head"],
        cwd=clean_root,
    )
    datasets = clean_root / "ipfs_datasets_py"
    if datasets.exists():
        assert not any(datasets.iterdir())
        datasets.rmdir()
    datasets_head = git("rev-parse", "HEAD", cwd=ROOT / "ipfs_datasets_py")
    git(
        "clone",
        "--shared",
        "--no-checkout",
        str(ROOT / "ipfs_datasets_py"),
        str(datasets),
    )
    git("checkout", "--detach", datasets_head, cwd=datasets)

    source_run = (
        ROOT
        / "data/agent_supervisor/logic_governed_compositional_verification_fabric/run-v16"
    )
    target_run = (
        clean_root
        / "data/agent_supervisor/logic_governed_compositional_verification_fabric/run-v16"
    )
    for relative in (
        Path("evidence/plan-revisions"),
        Path("evidence/quarantine"),
        Path("merge-queue/completed"),
        Path("merge-queue/train/receipts"),
    ):
        shutil.copytree(
            source_run / relative,
            target_run / relative,
            symlinks=True,
            copy_function=shutil.copy2,
        )
    assert git("status", "--porcelain=v1", "--untracked-files=all", cwd=clean_root) == ""

    population = materializer.build_population(config, root=clean_root)
    target = materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=clean_root
    )["target"]
    config_poison = copy.deepcopy(config)
    config_poison["poll_interval_seconds"] = int(
        config_poison["poll_interval_seconds"]
    ) + 1
    with pytest.raises(
        materializer.MaterializationError,
        match="configuration differs from the canonical profile",
    ):
        materializer.materialize_fresh_generation_recovery(
            config_poison,
            population,
            root=clean_root,
            source_root=clean_root,
        )
    assert not target.exists()
    assert not _fresh_recovery_staging_container(config, clean_root).exists()
    metadata_poison = copy.deepcopy(population)
    poisoned_task = next(
        item for item in metadata_poison["tasks"] if item["task_id"] == "LGCVF-090"
    )
    poisoned_task["title"] = "forged recovery task title"
    forged_root = copy.deepcopy(population)
    forged_root["population_root"] = "baguqeera-forged-population-root"
    forged_task = copy.deepcopy(population)
    forged_task["tasks"][0]["task_cid"] = "baguqeera-forged-task-cid"
    for poisoned in (metadata_poison, forged_root, forged_task):
        with pytest.raises(
            materializer.MaterializationError,
            match="population differs from canonical source projection",
        ):
            materializer.materialize_fresh_generation_recovery(
                config,
                poisoned,
                root=clean_root,
                source_root=clean_root,
            )
        assert not target.exists()
        assert not _fresh_recovery_staging_container(config, clean_root).exists()

    preview = materializer.preview_fresh_generation_recovery(
        config, population, root=clean_root, source_root=clean_root
    )
    qualification = _fake_recovery_qualification(preview)
    _patch_recovery_qualification_only(monkeypatch, qualification)
    tracked = clean_root / "README.md"
    tracked_bytes = tracked.read_bytes()

    def drift_tracked_source(point: str) -> None:
        if point == "after_stage_verification":
            tracked.write_bytes(tracked_bytes + b"\nsource-race\n")

    with pytest.raises(
        materializer.MaterializationError,
        match="exact clean source binding.*dirty execution worktree",
    ):
        materializer.materialize_fresh_generation_recovery(
            config,
            population,
            root=clean_root,
            source_root=clean_root,
            fault_injector=drift_tracked_source,
        )
    tracked.write_bytes(tracked_bytes)
    assert not target.exists()
    assert git("status", "--porcelain=v1", "--untracked-files=all", cwd=clean_root) == ""

    merge_path = clean_root / str(
        config["fresh_generation_recovery"]["merge_completions"][0][
            "completed_record_path"
        ]
    )
    merge_bytes = merge_path.read_bytes()

    def drift_ignored_forensic_evidence(point: str) -> None:
        if point == "after_stage_verification":
            merge_path.write_bytes(merge_bytes + b"\n")

    with pytest.raises(
        materializer.MaterializationError,
        match="(?:SHA-256|bytes|evidence|record)",
    ):
        materializer.materialize_fresh_generation_recovery(
            config,
            population,
            root=clean_root,
            source_root=clean_root,
            fault_injector=drift_ignored_forensic_evidence,
        )
    merge_path.write_bytes(merge_bytes)
    assert not target.exists()
    assert git("status", "--porcelain=v1", "--untracked-files=all", cwd=clean_root) == ""

    receipt = materializer.materialize_fresh_generation_recovery(
        config, population, root=clean_root, source_root=clean_root
    )
    assert receipt["completed_count"] == 13
    assert target.is_dir()
    staging = _fresh_recovery_staging_container(config, clean_root)
    assert (staging / "recovery.lock").is_file()
    assert not list(staging.glob("stage-*"))
    assert git("status", "--porcelain=v1", "--untracked-files=all", cwd=clean_root) == ""


def test_fresh_recovery_crash_collision_lock_and_qualifier_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, population = _fresh_recovery_population()
    real_qualification_verifier = materializer._verify_recovery_qualification  # noqa: SLF001

    crash_root = tmp_path / "crash"
    crash_root.mkdir()
    preview = materializer.preview_fresh_generation_recovery(
        config, population, root=crash_root, source_root=ROOT
    )
    qualification = _fake_recovery_qualification(preview)
    _patch_recovery_judge(monkeypatch, population, qualification)

    kill_root = tmp_path / "killed-before-publish"
    kill_root.mkdir()
    kill_paths = materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=kill_root
    )

    child_input = kill_root / "child-input.json"
    materializer._atomic_write_json(  # noqa: SLF001
        child_input,
        {
            "config": config,
            "population": population,
            "qualification": qualification,
        },
    )
    child_program = """
import copy
import json
import os
import sys
from pathlib import Path
from scripts import materialize_logic_governed_compositional_verification_fabric_control_plane as materializer
from scripts import qualify_logic_governed_compositional_verification_fabric as qualifier

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = payload["config"]
population = payload["population"]
qualification = payload["qualification"]
source_root = Path(sys.argv[2])
target_root = Path(sys.argv[3])
materializer._require_clean_recovery_source = lambda *_args, **_kwargs: (
    population["source_head"],
    str(population["repository_tree_id"]).removeprefix("git-tree:"),
)
materializer._require_isolated_recovery_interpreter = lambda: None
qualifier._require_isolated_recovery_runtime = lambda: None
materializer._run_and_verify_recovery_qualification = (
    lambda **_kwargs: copy.deepcopy(qualification)
)
materializer._verify_recovery_qualification = (
    lambda value, **_kwargs: copy.deepcopy(dict(value))
)

def exit_before_publish(point):
    if point == "after_stage_verification":
        os._exit(73)

materializer.materialize_fresh_generation_recovery(
    config,
    population,
    root=target_root,
    source_root=source_root,
    fault_injector=exit_before_publish,
)
os._exit(75)
"""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            child_program,
            str(child_input),
            str(ROOT),
            str(kill_root),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=180,
    )
    assert child.returncode == 73, (child.stdout, child.stderr)
    assert not kill_paths["target"].exists()
    kill_staging = _fresh_recovery_staging_container(config, kill_root)
    stale_stages = list(kill_staging.glob("stage-*"))
    assert len(stale_stages) == 1
    assert materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=stale_stages[0]
    )["target"].is_dir()
    killed_replay = materializer.materialize_fresh_generation_recovery(
        config,
        population,
        root=kill_root,
        source_root=ROOT,
    )
    assert killed_replay["completed_count"] == 13
    assert kill_paths["target"].is_dir()
    assert not list(kill_staging.glob("stage-*"))

    redirect_root = tmp_path / "redirect"
    redirect_root.mkdir()
    redirect_target = redirect_root / Path(
        str(config["fresh_generation_recovery"]["target_runtime_root"])
    )
    redirect_target.parent.mkdir(parents=True)
    redirect_destination = redirect_target.parent / "redirected-run-v17"
    os.symlink(redirect_destination.name, redirect_target)
    with pytest.raises(materializer.MaterializationError, match="contains a symlink"):
        materializer.materialize_fresh_generation_recovery(
            config, population, root=redirect_root, source_root=ROOT
        )
    assert not redirect_destination.exists()
    assert not _fresh_recovery_staging_container(config, redirect_root).exists()

    unsafe_stale_root = tmp_path / "unsafe-stale-stage"
    unsafe_stale_root.mkdir()
    unsafe_staging = _fresh_recovery_staging_container(config, unsafe_stale_root)
    unsafe_staging.mkdir(parents=True, mode=0o700)
    unsafe_staging.chmod(0o700)
    outside = unsafe_stale_root / "outside"
    outside.mkdir()
    os.symlink(outside, unsafe_staging / "stage-forged-link")
    with pytest.raises(
        materializer.MaterializationError,
        match="stale fresh recovery stage root identity differs",
    ):
        materializer.materialize_fresh_generation_recovery(
            config,
            population,
            root=unsafe_stale_root,
            source_root=ROOT,
        )
    assert outside.is_dir()
    assert (unsafe_staging / "stage-forged-link").is_symlink()
    assert not materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=unsafe_stale_root
    )["target"].exists()

    def crash(point: str) -> None:
        if point == "after_stage_verification":
            container = _fresh_recovery_staging_container(config, crash_root)
            stages = list(container.glob("stage-*"))
            assert len(stages) == 1
            staged_target = stages[0] / Path(
                str(config["fresh_generation_recovery"]["target_runtime_root"])
            )
            verified = staged_target.with_name("run-v17.verified-but-swapped")
            staged_target.rename(verified)
            staged_target.mkdir()

    with pytest.raises(
        materializer.MaterializationError,
        match="stage (?:root identity differs|changed after strict verification)",
    ):
        materializer.materialize_fresh_generation_recovery(
            config,
            population,
            root=crash_root,
            source_root=ROOT,
            fault_injector=crash,
        )
    crash_paths = materializer._fresh_recovery_paths(config, root=crash_root)  # noqa: SLF001
    assert not crash_paths["target"].exists()
    crash_staging = _fresh_recovery_staging_container(config, crash_root)
    assert crash_staging.is_dir()
    assert not list(crash_staging.glob("stage-*"))

    inner_swap_root = tmp_path / "inner-swap"
    inner_swap_root.mkdir()

    def mutate_verified_inner_file(point: str) -> None:
        if point != "after_stage_verification":
            return
        container = _fresh_recovery_staging_container(config, inner_swap_root)
        stages = list(container.glob("stage-*"))
        assert len(stages) == 1
        staged_receipt = materializer._fresh_recovery_paths(  # noqa: SLF001
            config, root=stages[0]
        )["recovery_receipt"]
        staged_receipt.write_bytes(staged_receipt.read_bytes() + b"\n")

    with pytest.raises(
        materializer.MaterializationError,
        match="stage (?:file identity changed|changed after strict verification)",
    ):
        materializer.materialize_fresh_generation_recovery(
            config,
            population,
            root=inner_swap_root,
            source_root=ROOT,
            fault_injector=mutate_verified_inner_file,
        )
    inner_paths = materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=inner_swap_root
    )
    assert not inner_paths["target"].exists()
    inner_staging = _fresh_recovery_staging_container(config, inner_swap_root)
    assert inner_staging.is_dir()
    assert not list(inner_staging.glob("stage-*"))

    permission_root = tmp_path / "permission-swap"
    permission_root.mkdir()

    def chmod_verified_stage(point: str) -> None:
        if point != "after_stage_verification":
            return
        stages = list(
            _fresh_recovery_staging_container(
                config, permission_root
            ).glob("stage-*")
        )
        assert len(stages) == 1
        staged_target = materializer._fresh_recovery_paths(  # noqa: SLF001
            config, root=stages[0]
        )["target"]
        staged_target.chmod(0o777)

    with pytest.raises(
        materializer.MaterializationError,
        match="stage (?:root identity|changed after strict verification)",
    ):
        materializer.materialize_fresh_generation_recovery(
            config,
            population,
            root=permission_root,
            source_root=ROOT,
            fault_injector=chmod_verified_stage,
        )
    assert not materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=permission_root
    )["target"].exists()

    def crash_after_publish(point: str) -> None:
        if point == "after_publish":
            raise RuntimeError("injected-after-publish")

    with pytest.raises(RuntimeError, match="injected-after-publish"):
        materializer.materialize_fresh_generation_recovery(
            config,
            population,
            root=crash_root,
            source_root=ROOT,
            fault_injector=crash_after_publish,
        )
    assert crash_paths["target"].is_dir()
    replay = materializer.materialize_fresh_generation_recovery(
        config, population, root=crash_root, source_root=ROOT
    )
    assert replay["completed_count"] == 13

    collision_root = tmp_path / "collision"
    collision_root.mkdir()
    collision_target = materializer._fresh_recovery_paths(  # noqa: SLF001
        config, root=collision_root
    )["target"]
    collision_target.mkdir(parents=True)
    with pytest.raises(materializer.MaterializationError):
        materializer.materialize_fresh_generation_recovery(
            config, population, root=collision_root, source_root=ROOT
        )

    lock_root = tmp_path / "hardlink-lock"
    lock_root.mkdir()
    lock_target = materializer._fresh_recovery_paths(config, root=lock_root)[  # noqa: SLF001
        "target"
    ]
    lock_container = _fresh_recovery_staging_container(config, lock_root)
    lock_container.mkdir(parents=True, mode=0o700)
    lock_container.chmod(0o700)
    seed = lock_container / "lock-seed"
    seed.write_text("preserved", encoding="utf-8")
    seed.chmod(0o600)
    os.link(seed, lock_container / "recovery.lock")
    with pytest.raises(materializer.MaterializationError, match="lock identity differs"):
        materializer.materialize_fresh_generation_recovery(
            config, population, root=lock_root, source_root=ROOT
        )
    assert not lock_target.exists()

    monkeypatch.setattr(
        materializer,
        "_verify_recovery_qualification",
        real_qualification_verifier,
    )
    with pytest.raises(
        materializer.MaterializationError,
        match="qualification is absent or invalid",
    ):
        # A minimal self-authored pass cannot enter the protected verifier.
        materializer._verify_recovery_qualification(  # noqa: SLF001
            {"passed": True, "returncode": 0},
            preview=preview,
            source_root=ROOT,
        )
