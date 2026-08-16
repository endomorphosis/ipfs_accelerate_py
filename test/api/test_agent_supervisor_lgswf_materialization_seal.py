"""Focused qualification for the two-stage LGSWF bootstrap authority.

The fixtures deliberately redirect the materializer to ``tmp_path``.  They do
not construct, inspect, or mutate the configured ``run-actual-v5`` namespace.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationExpiredError,
    DatabaseCoordinator,
    open_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)

ROOT = Path(__file__).resolve().parents[2]
MATERIALIZER = ROOT / "scripts/materialize_logic_governed_semantic_work_fabric_control_plane.py"
VALIDATOR = ROOT / "scripts/validate_logic_governed_semantic_work_fabric_board.py"
requires_duckdb = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for the temporary LGSWF control-plane fixture",
)
RECEIPT_RESULT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/content-addressed-receipt-result@1"
SEAL_AUTHORITY_KEYS = (
    "claim",
    "preparation",
    "validation_receipt",
    "seal_basis_evidence_receipt",
    "control_cas",
    "coordination_promotion",
    "cross_store_guard",
    "settled_lease",
    "writer_reservation",
    "writer_release",
)


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"lgswf_materializer_test_{id(object())}", MATERIALIZER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"lgswf_validator_test_{id(object())}", VALIDATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _temporary_config() -> dict[str, Any]:
    return {
        "database_program": {
            "store_id": "runtime/control.duckdb",
            "schema_revision": "datasets-authoritative-operational-v1",
            "schema_profile": "datasets-authoritative-operational",
            "semantic_relations_permitted": False,
        },
        "runtime_paths": {"evidence": "runtime/evidence"},
        "initial_projection": {
            "ready_task_ids": ["LGSWF-006"],
            "completed_task_ids": [],
        },
    }


def _temporary_population() -> dict[str, Any]:
    aliases = ("LGSWF-006", "LGSWF-001", "LGSWF-002", "LGSWF-003")
    tasks: list[dict[str, Any]] = []
    for ordinal, alias in enumerate(aliases, start=1):
        task: dict[str, Any] = {
            "task_cid": f"task:{alias}",
            "task_id": alias,
            "task_alias": alias,
            "goal_cid": "goal:lgswf-root",
            "plan_cid": "plan:lgswf-test",
            "objective_id": "objective:lgswf-root",
            "status": "todo",
            "priority": "P0",
            "ordinal": ordinal,
            "title": alias,
            "dependencies": ([] if alias == "LGSWF-006" else ["task:LGSWF-006"]),
            "outputs": [
                {
                    "path": f"outputs/{alias}.json",
                    "effect_id": f"effect:{alias}",
                }
            ],
            "acceptance": [f"accept {alias}"],
            "validations": [f"validate {alias}"],
        }
        if alias == "LGSWF-006":
            task.update(
                {
                    "completion": "manual",
                    "review_only": "true",
                    "is_schedulable": "false",
                }
            )
        tasks.append(task)
    return {
        "repository_tree_id": "tree:lgswf-test",
        "source_head": "head:lgswf-test",
        "plan_root_cid": "plan:lgswf-test",
        "objectives": [
            {
                "objective_id": "objective:lgswf-root",
                "objective_alias": "LGSWF-G000",
                "title": "LGSWF root",
                "goal_cid": "goal:lgswf-root",
                "goal_alias": "LGSWF-G000",
                "status": "open",
            }
        ],
        "plans": [
            {
                "plan_cid": "plan:lgswf-test",
                "plan_alias": "LGSWF-PLAN-TEST",
                "goal_cid": "goal:lgswf-root",
                "status": "active",
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": {alias: f"task:{alias}" for alias in aliases},
        "goal_cids_by_alias": {"LGSWF-G000": "goal:lgswf-root"},
    }


def _materialize_temporary_plane(
    tmp_path: Path,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any], dict[str, Any]]:
    module = _load_materializer()
    module.ROOT = tmp_path
    config = _temporary_config()
    population = _temporary_population()
    module._assert_population_source_current = lambda _config, _population: {
        "source_head": _population["source_head"],
        "repository_tree_id": _population["repository_tree_id"],
        "worktree_clean": True,
        "nested_repositories": [],
        "source_forest_root": module._identity(
            {
                "source_head": _population["source_head"],
                "repository_tree_id": _population["repository_tree_id"],
            }
        ),
    }
    _install_test_qualification(module, config, population)
    receipt = module.materialize(config, population)
    return module, config, population, receipt


def _canonical_receipt(
    module: ModuleType,
    result: dict[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
    assert set(result) == {
        "schema",
        "operation",
        "canonical_receipt",
        "canonical_receipt_path",
        "operation_replayed",
    }
    assert result["schema"] == RECEIPT_RESULT_SCHEMA
    assert result["operation"] == operation
    assert isinstance(result["operation_replayed"], bool)
    assert "receipt_cid" not in result
    receipt = result["canonical_receipt"]
    assert isinstance(receipt, dict)
    claimed_cid = receipt.get("receipt_cid")
    assert isinstance(claimed_cid, str) and claimed_cid.startswith("sha256:")
    unsigned = dict(receipt)
    unsigned.pop("receipt_cid")
    assert module._identity(unsigned) == claimed_cid
    persisted_path = module.ROOT / result["canonical_receipt_path"]
    assert json.loads(persisted_path.read_text(encoding="utf-8")) == receipt
    return receipt


def _stub_launch_evidence() -> dict[str, Any]:
    return {
        "launch_plan_cid": "sha256:" + "1" * 64,
        "schema": "lgswf-test-launch-plan@1",
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "schema_revision": "datasets-authoritative-operational-v1",
        "configured_schema_profile": "datasets-authoritative-operational",
        "semantic_relations_permitted": False,
        "lanes": 1,
        "admitted_lanes": 1,
        "plan_bound_dispatch": False,
        "effective_strict_task_sharding": True,
        "plan_bound_promotion_task": "LGSWF-005",
        "implement": True,
        "process_started": False,
    }


def _stub_seal_evidence(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    population: dict[str, Any],
) -> None:
    module._load_qualification_receipt(config, population)
    monkeypatch.setattr(
        module, "_render_launch_plan_evidence", lambda _config: _stub_launch_evidence()
    )
    monkeypatch.setattr(module, "_sha256_file", lambda _path: "sha256:" + "2" * 64)
    monkeypatch.setattr(
        module,
        "_control_bundle_cid",
        lambda _config, _population: "sha256:" + "3" * 64,
    )


def _sealed_temporary_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any], dict[str, Any]]:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    sealed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    return module, config, population, sealed


def _acquire_test_bootstrap_writer(
    module: ModuleType,
    config: dict[str, Any],
    population: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    materialization = module._load_materialization_receipt(config, population)
    qualification = module._load_qualification_receipt(config, population)
    seal_basis = module._build_seal_basis(
        config=config,
        population=population,
        materialization_receipt=materialization,
        qualification_receipt=qualification,
        launch_plan=module._render_launch_plan_evidence(config),
    )
    owner_id = "lgswf-bootstrap-seal:" + module._identity(population).split(":", 1)[1][:24]
    coordinator = open_database_coordinator(module._paths(config)["coordination"])
    try:
        writer = module._acquire_bootstrap_writer(
            coordinator,
            population=population,
            task_cid=population["task_cids_by_alias"]["LGSWF-006"],
            owner_id=owner_id,
            accepted_result_cid=module._identity(seal_basis),
        )
    finally:
        coordinator.close()
    return writer, seal_basis


def _control_cross_store_guard_event_count(
    module: ModuleType,
    paths: dict[str, Path],
    population: dict[str, Any],
) -> int:
    import duckdb  # type: ignore

    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="control-guard-count",
        repository_tree_id=population["repository_tree_id"],
        plan_root_cid=population["plan_root_cid"],
        install_schema=False,
    )
    try:
        task = task_source.get(population["task_cids_by_alias"]["LGSWF-006"])
        if task is None or task.status != "completed":
            return 0
        control_result_digest = module._identity(task.to_dict())
    finally:
        task_source.close()
    connection = duckdb.connect(str(paths["coordination"]), read_only=True)
    try:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM lease_events "
                "WHERE event_type = 'cross_store_fence_guard_succeeded' "
                "AND json_extract_string(body_json, '$.control_result_digest') = ?",
                [control_result_digest],
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _rewrite_self_consistent_seal_receipt(
    module: ModuleType,
    config: dict[str, Any],
    *,
    field: str,
    value: Any,
) -> None:
    path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.pop("receipt_cid")
    receipt[field] = value
    receipt["authority_root"] = module._identity(
        {key: receipt[key] for key in SEAL_AUTHORITY_KEYS}
    )
    receipt["receipt_cid"] = module._identity(receipt)
    module._write_receipt(path, receipt)


def _install_test_qualification(
    module: ModuleType,
    config: dict[str, Any],
    population: dict[str, Any],
) -> dict[str, Any]:
    commands = module._qualification_commands()
    source_verification = module._assert_population_source_current(config, population)
    receipt = {
        "schema": module.QUALIFICATION_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": module._identity(population),
        "source_verification": source_verification,
        "command_argv": [list(argv) for _label, argv, _expected in commands],
        "qualified": True,
        "results": [
            {
                "label": label,
                "argv": list(argv),
                "returncode": 0,
                "expected_passed": expected_passed,
                "required_outcomes_valid": True,
                "stdout_sha256": "sha256:" + "a" * 64,
                "stderr_sha256": "sha256:" + "b" * 64,
            }
            for label, argv, expected_passed in commands
        ],
    }
    receipt["receipt_cid"] = module._identity(receipt)
    module._write_receipt(
        module._bootstrap_receipt_path(config, "qualification.json"),
        receipt,
    )
    assert module._load_qualification_receipt(config, population) == receipt
    return receipt


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_dirty_worktree_fails_closed_before_population_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_materializer()
    calls: list[list[str]] = []

    def dirty_status(argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(stdout=" M docs/architecture/board.md\n")

    monkeypatch.setattr(module.subprocess, "run", dirty_status)

    with pytest.raises(
        module.MaterializationError,
        match="dirty execution worktree",
    ):
        module.build_population({})

    assert calls == [["git", "status", "--porcelain=v1", "--untracked-files=all"]]


def test_population_binds_clean_execution_head_tree_and_explicit_goal_edges(
    tmp_path: Path,
) -> None:
    module = _load_materializer()
    repository = tmp_path / "accelerator"
    datasets = repository / "datasets"
    datasets.mkdir(parents=True)

    _git(datasets, "init", "-q")
    (datasets / "semantic.py").write_text("SEMANTIC_AUTHORITY = True\n", encoding="utf-8")
    _git(datasets, "add", "semantic.py")
    _git(
        datasets,
        "-c",
        "user.name=LGSWF Test",
        "-c",
        "user.email=lgswf@example.invalid",
        "commit",
        "-qm",
        "datasets authority",
    )
    datasets_head = _git(datasets, "rev-parse", "HEAD")

    board_path = repository / "board.md"
    objectives_path = repository / "objectives.md"
    plan_path = repository / "plan.md"
    config_path = repository / "scheduler.json"
    board_template = """# Board

## LGSWF-001 Execute bound task
- Status: todo
- Owning repository: ipfs_accelerate_py
- Base revision: {planning_revision}
- Subgoal ID: LGSWF-G010
- Depends on:
"""
    objectives_path.write_text(
        """# Objectives

## LGSWF-G000 Root goal
- Priority: P0

## LGSWF-G010 Child goal
- Parent: LGSWF-G000
- Priority: P0

## LGSWF-G020 Dependent goal
- Parent: LGSWF-G000
- Depends on: LGSWF-G010
- Priority: P0
""",
        encoding="utf-8",
    )
    plan_path.write_text("# Plan\n", encoding="utf-8")

    def write_inputs(planning_revision: str) -> None:
        board_path.write_text(
            board_template.format(planning_revision=planning_revision),
            encoding="utf-8",
        )
        config_path.write_text(
            json.dumps(
                {
                    "taskboard_path": "board.md",
                    "objectives_path": "objectives.md",
                    "plan_path": "plan.md",
                    "initial_projection": {"task_count": 1, "goal_count": 3},
                    "source_binding": {
                        "accelerator_required_ancestor": planning_revision,
                        "ipfs_datasets_submodule_path": "datasets",
                        "ipfs_datasets_planning_revision": datasets_head,
                    },
                    "supersedes_quarantined_plan_root_cid": "sha256:" + "0" * 64,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    _git(repository, "init", "-q")
    write_inputs("planning-lineage-placeholder")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=LGSWF Test",
        "-c",
        "user.email=lgswf@example.invalid",
        "commit",
        "-qm",
        "planning lineage",
    )
    planning_revision = _git(repository, "rev-parse", "HEAD")
    write_inputs(planning_revision)
    _git(repository, "add", "board.md", "scheduler.json")
    _git(
        repository,
        "-c",
        "user.name=LGSWF Test",
        "-c",
        "user.email=lgswf@example.invalid",
        "commit",
        "-qm",
        "execution head",
    )
    execution_head = _git(repository, "rev-parse", "HEAD")
    execution_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    assert execution_head != planning_revision
    assert _git(repository, "status", "--porcelain=v1", "--untracked-files=all") == ""

    module.ROOT = repository
    module.CONFIG_PATH = config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    population = module.build_population(config)
    source_verification = module._assert_population_source_current(config, population)

    assert population["source_head"] == execution_head
    assert population["repository_tree_id"] == execution_tree
    assert source_verification["source_head"] == execution_head
    assert source_verification["repository_tree_id"] == execution_tree
    assert source_verification["nested_repositories"][0]["head"] == datasets_head
    assert population["plans"][0]["source_head"] == execution_head
    assert population["plans"][0]["repository_tree_id"] == execution_tree
    task = population["tasks"][0]
    assert task["owning_repository"] == "ipfs_accelerate_py"
    assert task["planning_lineage_revision"] == planning_revision
    assert task["base_revision"] == execution_head
    assert task["base_repository_tree_id"] == execution_tree
    assert task["accepted_plan_root_cid"] == population["plan_root_cid"]

    task_body = {
        key: value for key, value in task.items() if key not in module._TASK_BODY_TOP_LEVEL_FIELDS
    }
    assert task_body["planning_lineage_revision"] == planning_revision
    assert task_body["base_revision"] == execution_head
    assert task_body["base_repository_tree_id"] == execution_tree

    goal_cids = population["goal_cids_by_alias"]
    assert population["goal_edges"] == [
        {
            "parent_goal_cid": goal_cids["LGSWF-G000"],
            "child_goal_cid": goal_cids["LGSWF-G010"],
            "edge_kind": "goal_parent",
        },
        {
            "parent_goal_cid": goal_cids["LGSWF-G000"],
            "child_goal_cid": goal_cids["LGSWF-G020"],
            "edge_kind": "goal_parent",
        },
        {
            "parent_goal_cid": goal_cids["LGSWF-G010"],
            "child_goal_cid": goal_cids["LGSWF-G020"],
            "edge_kind": "goal_dependency",
        },
    ]


def test_actual_launch_evidence_reports_bounded_legacy_dispatch_truthfully() -> None:
    module = _load_materializer()
    config = module._load_config()

    evidence = module._render_launch_plan_evidence(config)

    assert evidence["authority_mode"] == "embedded"
    assert evidence["task_source_kind"] == "duckdb"
    assert evidence["lanes"] == 1
    assert evidence["admitted_lanes"] == 1
    assert evidence["effective_strict_task_sharding"] is True
    assert evidence["plan_bound_dispatch"] is False
    assert evidence["plan_bound_promotion_task"] == "LGSWF-005"
    assert evidence["implement"] is True
    assert evidence["process_started"] is False


@requires_duckdb
def test_unsealed_plane_exposes_only_manual_seal_and_daemon_skips_it(
    tmp_path: Path,
) -> None:
    module, config, population, result = _materialize_temporary_plane(tmp_path)
    assert result["operation_replayed"] is False
    receipt = _canonical_receipt(module, result, operation="materialize")

    verification = receipt["verification"]
    assert verification["bootstrap_stage"] == "unsealed"
    assert verification["ready_task_aliases"] == ["LGSWF-006"]
    assert verification["completed_task_aliases"] == []
    assert verification["active_task_claim_count"] == 0

    paths = module._paths(config)
    daemon = DatabaseImplementationDaemon(
        database_path=paths["control"],
        coordination_path=paths["coordination"],
        execution_path=paths["execution"],
        owner_session_id="lgswf-test-automatic-daemon",
        authority_mode="embedded",
        task_source_kind="duckdb",
        install_schema=False,
    )
    try:
        assert daemon._automatic_claim_exclusions() == {"task:LGSWF-006"}
        assert daemon.claim_next() is None
        manual = daemon.task_source.get(population["task_cids_by_alias"]["LGSWF-006"])
        assert manual is not None
        assert manual.status == "todo"
        assert [dict(item) for item in manual.acceptance] == [
            {
                "ordinal": 0,
                "criterion": "accept LGSWF-006",
                "evidence_policy": {"criterion": "accept LGSWF-006"},
            }
        ]
        assert [dict(item) for item in manual.validations] == [
            {"ordinal": 0, "argv": ["validate LGSWF-006"], "policy": {}}
        ]
    finally:
        daemon.close()

    connection = open_duckdb_connection(paths["control"])
    try:
        connection.execute(
            "UPDATE goals SET title = ? WHERE goal_cid = ?",
            ["FORGED GOAL TITLE", "goal:lgswf-root"],
        )
    finally:
        connection.close()
    with pytest.raises(
        module.MaterializationError,
        match="control objective/goal/plan population changed",
    ):
        module._verify_control_population(config, population)


@requires_duckdb
def test_restart_resumes_exact_live_claim_before_preparation_without_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    captured: dict[str, Any] = {}

    materialization_receipt = module._load_materialization_receipt(config, population)
    qualification_receipt = module._load_qualification_receipt(config, population)
    seal_basis = module._build_seal_basis(
        config=config,
        population=population,
        materialization_receipt=materialization_receipt,
        qualification_receipt=qualification_receipt,
        launch_plan=_stub_launch_evidence(),
    )
    owner_id = "lgswf-bootstrap-seal:" + module._identity(population).split(":", 1)[1][:24]
    paths = module._paths(config)
    writer_coordinator = open_database_coordinator(paths["coordination"])
    try:
        stranded_writer = module._acquire_bootstrap_writer(
            writer_coordinator,
            population=population,
            task_cid=population["task_cids_by_alias"]["LGSWF-006"],
            owner_id=owner_id,
            accepted_result_cid=module._identity(seal_basis),
        )
    finally:
        writer_coordinator.close()

    def crash_before_preparation(
        _coordinator: DatabaseCoordinator,
        claim: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        captured["claim"] = claim.to_dict()
        writer_rows = [
            item
            for item in _coordinator.coordination_registry_projection()["resource_claims"]
            if item["state"] == "accepted"
        ]
        assert len(writer_rows) == 1
        captured["writer"] = writer_rows[0]
        raise RuntimeError("injected crash before PREPARED")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseCoordinator,
            "prepare_task_completion",
            crash_before_preparation,
        )
        with pytest.raises(RuntimeError, match="injected crash before PREPARED"):
            module.seal(config, population)

    assert captured["writer"]["lease_id"] == stranded_writer.lease_id
    assert captured["writer"]["fencing_token"] == stranded_writer.fencing_token
    assert captured["writer"]["fence_epoch"] == stranded_writer.fence_epoch
    coordinator = open_database_coordinator(paths["coordination"])
    try:
        active = coordinator.list_active_leases()
        assert len(active) == 1
        assert active[0].claim_id == captured["claim"]["claim_id"]
        assert active[0].expires_at_ms > time.time_ns() // 1_000_000
        assert coordinator.list_unsettled_task_completions(limit=100) == []
    finally:
        coordinator.close()

    resumed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    assert resumed["claim"]["claim_id"] == captured["claim"]["claim_id"]
    assert resumed["claim"]["lease_id"] == captured["claim"]["lease_id"]
    assert resumed["preparation"]["claim_id"] == captured["claim"]["claim_id"]
    assert resumed["settled_lease"]["state"] == "released"


@requires_duckdb
def test_restart_resumes_exact_live_prepared_todo_without_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    captured: dict[str, Any] = {}

    def crash_after_preparation(
        _coordinator: DatabaseCoordinator,
        claim: Any,
        **_kwargs: Any,
    ) -> Any:
        captured["claim"] = claim.to_dict()
        raise RuntimeError("injected crash after PREPARED")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseCoordinator,
            "protect_task_claim",
            crash_after_preparation,
        )
        with pytest.raises(RuntimeError, match="injected crash after PREPARED"):
            module.seal(config, population)

    paths = module._paths(config)
    task_cid = population["task_cids_by_alias"]["LGSWF-006"]
    coordinator = open_database_coordinator(paths["coordination"])
    try:
        prepared = coordinator.get_prepared_task_completion(task_cid)
        assert prepared is not None
        assert prepared["status"] == "prepared"
        assert prepared["claim_id"] == captured["claim"]["claim_id"]
        active = coordinator.list_active_leases()
        assert len(active) == 1
        assert active[0].claim_id == captured["claim"]["claim_id"]
        assert active[0].expires_at_ms > time.time_ns() // 1_000_000
    finally:
        coordinator.close()

    resumed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    assert resumed["claim"]["claim_id"] == captured["claim"]["claim_id"]
    assert resumed["preparation"]["preparation_digest"] == prepared["preparation_digest"]
    assert resumed["control_cas"]["task"]["status"] == "completed"
    assert resumed["settled_lease"]["state"] == "released"


@requires_duckdb
@pytest.mark.parametrize("crash_window", ["after_validation", "after_basis_evidence"])
def test_prepared_retry_reuses_one_attempt_bound_validation_and_basis_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_window: str,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    paths = module._paths(config)

    with monkeypatch.context() as injected:
        if crash_window == "after_validation":
            injected.setattr(
                DatabaseTaskSource,
                "record_evidence",
                lambda _self, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("injected crash after validation")
                ),
            )
            expected_error = "injected crash after validation"
        else:
            injected.setattr(
                DatabaseTaskSource,
                "compare_and_set_status",
                lambda _self, *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("injected crash after basis evidence")
                ),
            )
            expected_error = "injected crash after basis evidence"
        with pytest.raises(RuntimeError, match=expected_error):
            module.seal(config, population)

    connection = open_duckdb_connection(paths["control"])
    try:
        validation_count = int(
            connection.execute("SELECT COUNT(*) FROM validation_results").fetchone()[0]
        )
        basis_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM evidence_nodes "
                "WHERE evidence_kind = 'bootstrap_seal_basis'"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    assert validation_count == 1
    assert basis_count == (0 if crash_window == "after_validation" else 1)

    sealed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    binding = sealed["post_verification"]["completion_binding"]
    validation = sealed["validation_receipt"]
    basis = sealed["seal_basis_evidence_receipt"]
    assert validation["attempt_id"] == binding["attempt_id"]
    for field in (
        "task_cid",
        "claim_id",
        "attempt_id",
        "lease_id",
        "fencing_token",
        "fence_epoch",
    ):
        assert validation["body"][field] == binding[field]
        assert basis["body"][field] == binding[field]

    connection = open_duckdb_connection(paths["control"])
    try:
        assert int(connection.execute("SELECT COUNT(*) FROM validation_results").fetchone()[0]) == 1
        assert int(
            connection.execute(
                "SELECT COUNT(*) FROM evidence_nodes "
                "WHERE evidence_kind = 'bootstrap_seal_basis'"
            ).fetchone()[0]
        ) == 1
    finally:
        connection.close()


@requires_duckdb
@pytest.mark.parametrize("expiry_stage", ["validation", "basis_evidence"])
def test_expired_partial_evidence_is_immutably_superseded_by_later_fenced_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expiry_stage: str,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    paths = module._paths(config)
    original_now_ms = DatabaseCoordinator._now_ms
    mutation_name = (
        "record_validation_result" if expiry_stage == "validation" else "record_evidence"
    )
    original_mutation = getattr(DatabaseTaskSource, mutation_name)
    clock = {"expired": False}

    def injected_now_ms(self: DatabaseCoordinator) -> int:
        observed = original_now_ms(self)
        return observed + 600_000 if clock["expired"] else observed

    def expire_after_external_mutation(
        self: DatabaseTaskSource,
        **kwargs: Any,
    ) -> Any:
        receipt = original_mutation(self, **kwargs)
        clock["expired"] = True
        return receipt

    with monkeypatch.context() as injected:
        injected.setattr(DatabaseCoordinator, "_now_ms", injected_now_ms)
        injected.setattr(DatabaseTaskSource, mutation_name, expire_after_external_mutation)
        with pytest.raises(
            DatabaseCoordinationExpiredError,
            match="task claim .* is not accepted",
        ):
            module.seal(config, population)

    qualification = module._load_qualification_receipt(config, population)
    materialization = module._load_materialization_receipt(config, population)
    launch_plan = module._render_launch_plan_evidence(config)
    seal_basis = module._build_seal_basis(
        config=config,
        population=population,
        materialization_receipt=materialization,
        qualification_receipt=qualification,
        launch_plan=launch_plan,
    )
    accepted_result_cid = module._identity(seal_basis)
    old_validations, old_basis = module._read_manual_seal_evidence(
        control_path=paths["control"],
        task_cid=population["task_cids_by_alias"]["LGSWF-006"],
        qualification_receipt_cid=qualification["receipt_cid"],
        seal_basis_cid=accepted_result_cid,
    )
    assert len(old_validations) == 1
    assert len(old_basis) == (0 if expiry_stage == "validation" else 1)
    old_attempt_id = old_validations[0]["body"]["attempt_id"]
    old_fence = (
        old_validations[0]["body"]["fence_epoch"],
        old_validations[0]["body"]["fencing_token"],
    )

    deadline = time.time_ns() // 1_000_000 - 1
    connection = open_duckdb_connection(paths["coordination"])
    try:
        connection.execute(
            "UPDATE task_claims SET expires_at_ms = ? WHERE state = 'accepted'",
            [deadline],
        )
        connection.execute(
            "UPDATE resource_claims SET expires_at_ms = ? WHERE state = 'accepted'",
            [deadline],
        )
        connection.execute(
            "UPDATE fenced_leases SET expires_at_ms = ? WHERE state = 'accepted'",
            [deadline],
        )
    finally:
        connection.close()

    sealed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    assert sealed["claim"]["attempt_id"] != old_attempt_id
    assert (sealed["claim"]["fence_epoch"], sealed["claim"]["fencing_token"]) > old_fence
    links = sealed["validation_receipt"]["body"]["superseded_partial_evidence"]
    assert sealed["seal_basis_evidence_receipt"]["body"][
        "superseded_partial_evidence"
    ] == links
    assert len(links) == 1 + len(old_basis)
    assert {item["receipt_cid"] for item in links} == {
        module._identity(item) for item in [*old_validations, *old_basis]
    }
    admissions = {item["stage"]: item["fence_admission"] for item in links}
    if expiry_stage == "validation":
        assert admissions == {"validation": "post_fence_failed"}
    else:
        assert admissions == {
            "validation": "guarded",
            "basis_evidence": "post_fence_failed",
        }

    all_validations, all_basis = module._read_manual_seal_evidence(
        control_path=paths["control"],
        task_cid=population["task_cids_by_alias"]["LGSWF-006"],
        qualification_receipt_cid=qualification["receipt_cid"],
        seal_basis_cid=accepted_result_cid,
    )
    assert all(item in all_validations for item in old_validations)
    assert all(item in all_basis for item in old_basis)
    assert len(all_validations) == 2
    assert len(all_basis) == 1 + len(old_basis)
    projection = module._verify_live_store(config, population)
    assert projection["seal"]["receipt_cid"] == sealed["receipt_cid"]
    replay = _canonical_receipt(module, module.seal(config, population), operation="seal")
    assert replay == sealed


@requires_duckdb
def test_control_cas_before_guard_replays_idempotently_under_live_fences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    writer, _seal_basis = _acquire_test_bootstrap_writer(module, config, population)
    paths = module._paths(config)

    original_cas = DatabaseTaskSource.compare_and_set_status

    def fail_after_control_cas(
        self: DatabaseTaskSource,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_cas(self, *args, **kwargs)
        raise RuntimeError("injected crash after control CAS before guard")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseTaskSource,
            "compare_and_set_status",
            fail_after_control_cas,
        )
        with pytest.raises(
            RuntimeError,
            match="injected crash after control CAS before guard",
        ):
            module._seal_with_writer(config, population, writer_claim=writer)

    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="guard-crash-check",
        repository_tree_id=population["repository_tree_id"],
        plan_root_cid=population["plan_root_cid"],
        install_schema=False,
    )
    try:
        task = task_source.get(population["task_cids_by_alias"]["LGSWF-006"])
        assert task is not None and task.status == "completed"
    finally:
        task_source.close()
    assert _control_cross_store_guard_event_count(module, paths, population) == 0

    resumed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    assert any(
        item.get("status") == "guard_replay_required"
        for item in resumed["recovery"]
    )
    assert resumed["cross_store_guard"]["control_result_digest"] == module._identity(
        resumed["control_cas"]["task"]
    )
    assert _control_cross_store_guard_event_count(module, paths, population) == 1


@requires_duckdb
def test_guard_before_promotion_recovers_without_duplicate_guard_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    writer, _seal_basis = _acquire_test_bootstrap_writer(module, config, population)
    paths = module._paths(config)

    def fail_before_promotion(
        _self: DatabaseCoordinator,
        _claim: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        raise RuntimeError("injected crash after guard before promotion")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseCoordinator,
            "complete_task_claim",
            fail_before_promotion,
        )
        with pytest.raises(
            RuntimeError,
            match="injected crash after guard before promotion",
        ):
            module._seal_with_writer(config, population, writer_claim=writer)

    assert _control_cross_store_guard_event_count(module, paths, population) == 1
    resumed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    assert resumed["coordination_promotion"]["status"] == "succeeded"
    assert resumed["settled_lease"]["state"] == "released"
    assert _control_cross_store_guard_event_count(module, paths, population) == 1


@requires_duckdb
def test_naturally_expired_guarded_task_recovers_from_exact_durable_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    writer, _seal_basis = _acquire_test_bootstrap_writer(module, config, population)
    paths = module._paths(config)

    def fail_before_promotion(
        _self: DatabaseCoordinator,
        _claim: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        raise RuntimeError("injected crash after guard before natural task expiry")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseCoordinator,
            "complete_task_claim",
            fail_before_promotion,
        )
        with pytest.raises(
            RuntimeError,
            match="injected crash after guard before natural task expiry",
        ):
            module._seal_with_writer(config, population, writer_claim=writer)

    deadline = time.time_ns() // 1_000_000 - 1
    connection = open_duckdb_connection(paths["coordination"])
    try:
        connection.execute(
            "UPDATE task_claims SET expires_at_ms = ? WHERE state = 'accepted'",
            [deadline],
        )
        connection.execute(
            "UPDATE fenced_leases SET expires_at_ms = ? "
            "WHERE lease_kind = 'task' AND state = 'accepted'",
            [deadline],
        )
    finally:
        connection.close()

    resumed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    assert any(item.get("recovered") is True for item in resumed["recovery"])
    assert resumed["claim"]["state"] == "completed"
    assert resumed["settled_lease"]["state"] == "completed"
    assert _control_cross_store_guard_event_count(module, paths, population) == 1
    assert module._load_existing_seal_receipt(config, population) is not None

    forged_release = dict(resumed["writer_release"])
    forged_release["fencing_token"] = int(forged_release["fencing_token"]) + 1
    _rewrite_self_consistent_seal_receipt(
        module,
        config,
        field="writer_release",
        value=forged_release,
    )
    with pytest.raises(
        module.MaterializationError,
        match="durable authority differs",
    ):
        module._load_existing_seal_receipt(config, population)


@requires_duckdb
def test_expired_guard_writer_uses_exact_later_recovery_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    writer, _seal_basis = _acquire_test_bootstrap_writer(module, config, population)
    paths = module._paths(config)

    def fail_before_promotion(
        _self: DatabaseCoordinator,
        _claim: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        raise RuntimeError("injected crash after guard before writer expiry")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseCoordinator,
            "complete_task_claim",
            fail_before_promotion,
        )
        with pytest.raises(
            RuntimeError,
            match="injected crash after guard before writer expiry",
        ):
            module._seal_with_writer(config, population, writer_claim=writer)

    deadline = time.time_ns() // 1_000_000 - 1
    connection = open_duckdb_connection(paths["coordination"])
    try:
        connection.execute(
            "UPDATE resource_claims SET expires_at_ms = ? WHERE claim_id = ?",
            [deadline, writer.claim_id],
        )
        connection.execute(
            "UPDATE fenced_leases SET expires_at_ms = ? WHERE lease_id = ?",
            [deadline, writer.lease_id],
        )
    finally:
        connection.close()

    resumed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    reservation = resumed["writer_reservation"]
    assert resumed["cross_store_guard"]["writer_claim_id"] == writer.claim_id
    assert reservation["claim_id"] != writer.claim_id
    assert (
        int(reservation["fence_epoch"]),
        int(reservation["fencing_token"]),
    ) > (int(writer.fence_epoch), int(writer.fencing_token))
    assert resumed["writer_release"]["claim_id"] == reservation["claim_id"]
    assert resumed["writer_release"]["state"] == "released"
    projection = module._verify_live_store(config, population)
    assert projection["seal"]["receipt_cid"] == resumed["receipt_cid"]

    forged_reservation = dict(reservation)
    forged_reservation["fencing_token"] = int(reservation["fencing_token"]) + 100
    _rewrite_self_consistent_seal_receipt(
        module,
        config,
        field="writer_reservation",
        value=forged_reservation,
    )
    with pytest.raises(
        module.MaterializationError,
        match="writer fences cannot be reconstructed exactly",
    ):
        module._verify_live_store(config, population)


@requires_duckdb
def test_expired_prepared_control_effect_without_guard_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    writer, _seal_basis = _acquire_test_bootstrap_writer(module, config, population)
    paths = module._paths(config)

    original_cas = DatabaseTaskSource.compare_and_set_status

    def fail_after_control_cas(
        self: DatabaseTaskSource,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_cas(self, *args, **kwargs)
        raise RuntimeError("injected unguarded control effect")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseTaskSource,
            "compare_and_set_status",
            fail_after_control_cas,
        )
        with pytest.raises(RuntimeError, match="injected unguarded control effect"):
            module._seal_with_writer(config, population, writer_claim=writer)

    coordinator = open_database_coordinator(paths["coordination"])
    try:
        prepared = coordinator.list_prepared_task_completions(limit=10)
        assert len(prepared) == 1
        claim = coordinator.get_task_claim(prepared[0]["claim_id"])
        assert claim is not None
        lease = coordinator.get_lease(claim.lease_id)
        assert lease is not None
        coordinator.expire_task_claim(
            claim,
            now_ms=int(lease.expires_at_ms) + 1,
        )
    finally:
        coordinator.close()

    with pytest.raises(
        module.MaterializationError,
        match="expired manual seal completion has no durable cross-store fence guard",
    ):
        module.seal(config, population)
    assert _control_cross_store_guard_event_count(module, paths, population) == 0
    assert not module._bootstrap_receipt_path(config, "duckdb-seal.json").exists()


@requires_duckdb
@pytest.mark.parametrize("tamper", ["validation_attempt", "basis_fence"])
def test_attempt_or_fence_tamper_rejects_seal_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    sealed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    paths = module._paths(config)
    connection = open_duckdb_connection(paths["control"])
    try:
        if tamper == "validation_attempt":
            connection.execute(
                "UPDATE validation_runs SET attempt_id = 'attempt:forged' "
                "WHERE run_id = ?",
                [sealed["validation_receipt"]["run_id"]],
            )
        else:
            row = connection.execute(
                "SELECT body_json FROM evidence_nodes WHERE evidence_id = ?",
                [sealed["seal_basis_evidence_receipt"]["evidence_id"]],
            ).fetchone()
            assert row is not None
            body = json.loads(row[0])
            body["fence_epoch"] = int(body["fence_epoch"]) + 1
            connection.execute(
                "UPDATE evidence_nodes SET body_json = ? WHERE evidence_id = ?",
                [
                    json.dumps(body, sort_keys=True, separators=(",", ":")),
                    sealed["seal_basis_evidence_receipt"]["evidence_id"],
                ],
            )
    finally:
        connection.close()

    with pytest.raises(
        module.MaterializationError,
        match="bound to a different attempt or fence",
    ):
        module.seal(config, population)


@requires_duckdb
def test_foreign_resource_lease_blocks_unsealed_verification_and_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    paths = module._paths(config)
    coordinator = open_database_coordinator(paths["coordination"])
    try:
        resource = coordinator.claim_resource(
            resource_kind="gpu",
            resource_id="gpu:foreign",
            owner_session_id="foreign-supervisor",
            lease_ms=300_000,
        )
        assert resource.state.value == "accepted"
    finally:
        coordinator.close()

    try:
        with pytest.raises(module.MaterializationError) as unsealed_error:
            module._verify_store(config, population, expected_stage="unsealed")
        assert "active" in str(unsealed_error.value).lower()
        assert "lease" in str(unsealed_error.value).lower()

        with pytest.raises(module.MaterializationError) as seal_error:
            module.seal(config, population)
        assert "foreign resource" in str(seal_error.value).lower()
    finally:
        coordinator = open_database_coordinator(paths["coordination"])
        try:
            lease = coordinator.get_lease(resource.lease_id)
            assert lease is not None
            coordinator.release(lease, reason="test cleanup")
        finally:
            coordinator.close()

    with pytest.raises(
        module.MaterializationError,
        match="fresh unsealed coordination registry contains execution history",
    ):
        module._verify_store(config, population, expected_stage="unsealed")
    with pytest.raises(
        module.MaterializationError,
        match="foreign writer authority|foreign resource",
    ):
        module.seal(config, population)


@requires_duckdb
def test_trusted_seal_prepares_cas_promotes_settles_and_replays_immutably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)

    sealed_result = module.seal(config, population)
    assert sealed_result["operation_replayed"] is False
    sealed = _canonical_receipt(module, sealed_result, operation="seal")
    accepted_result_cid = sealed["accepted_result_cid"]
    assert sealed["preparation"]["status"] == "prepared"
    assert sealed["preparation"]["evidence_digest"] == accepted_result_cid
    assert (
        sealed["preparation"]["body"]["requires_cross_store_fence_guard"]
        is True
    )
    assert sealed["control_cas"]["changed"] is True
    assert sealed["control_cas"]["task"]["status"] == "completed"
    assert (
        sealed["control_cas"]["task"]["body"]["completion_receipt"]["accepted_result_cid"]
        == accepted_result_cid
    )
    assert sealed["coordination_promotion"]["status"] == "succeeded"
    assert sealed["cross_store_guard"]["preparation_digest"] == (
        sealed["preparation"]["preparation_digest"]
    )
    assert sealed["cross_store_guard"]["control_result_digest"] == module._identity(
        sealed["control_cas"]["task"]
    )
    assert sealed["settled_lease"]["state"] == "released"
    assert sealed["post_verification"]["accepted_result_cid"] == accepted_result_cid
    assert sealed["post_verification"]["ready_task_aliases"] == [
        "LGSWF-001",
        "LGSWF-002",
        "LGSWF-003",
    ]

    receipt_path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    stored_before = receipt_path.read_bytes()
    persisted = json.loads(stored_before)
    claimed_receipt_cid = persisted.pop("receipt_cid")
    assert module._identity(persisted) == claimed_receipt_cid

    replay_result = module.seal(config, population)
    assert replay_result["operation_replayed"] is True
    replay = _canonical_receipt(module, replay_result, operation="seal")
    assert replay["receipt_cid"] == sealed["receipt_cid"]
    assert replay["accepted_result_cid"] == accepted_result_cid
    assert receipt_path.read_bytes() == stored_before


@requires_duckdb
def test_forged_control_result_identity_fails_sealed_cross_authority_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    sealed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    accepted_result_cid = sealed["accepted_result_cid"]
    task_cid = population["task_cids_by_alias"]["LGSWF-006"]
    paths = module._paths(config)

    coordinator = open_database_coordinator(paths["coordination"])
    try:
        promoted = coordinator.get_prepared_task_completion(task_cid)
        assert promoted is not None
        assert promoted["status"] == "succeeded"
        assert promoted["evidence_digest"] == accepted_result_cid
        assert module._identity(promoted["body"]["seal_basis"]) == accepted_result_cid
    finally:
        coordinator.close()

    connection = open_duckdb_connection(paths["control"])
    try:
        row = connection.execute(
            "SELECT body_json FROM tasks WHERE task_cid = ?",
            [task_cid],
        ).fetchone()
        assert row is not None
        body = json.loads(row[0])
        forged_result_cid = "sha256:" + "e" * 64
        assert forged_result_cid != accepted_result_cid
        body["completion_receipt"]["accepted_result_cid"] = forged_result_cid
        connection.execute(
            "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
            [json.dumps(body, sort_keys=True, separators=(",", ":")), task_cid],
        )
    finally:
        connection.close()

    with pytest.raises(module.MaterializationError) as error:
        module._verify_store(config, population, expected_stage="sealed")
    assert "accepted result" in str(error.value).lower()
    assert "coordination" in str(error.value).lower()


@requires_duckdb
def test_post_settlement_reconstruction_writes_identity_default_verify_accepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)

    sealed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    accepted_result_cid = sealed["accepted_result_cid"]
    assert sealed["coordination_promotion"]["status"] == "succeeded"
    assert sealed["settled_lease"]["state"] == "released"
    assert (
        module._verify_store(config, population, expected_stage="sealed")["accepted_result_cid"]
        == accepted_result_cid
    )

    receipt_path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    receipt_path.unlink()
    assert not receipt_path.exists()

    reconstructed_result = module.seal(config, population)
    assert reconstructed_result["operation_replayed"] is False
    reconstructed = _canonical_receipt(
        module,
        reconstructed_result,
        operation="seal",
    )
    assert reconstructed["accepted_result_cid"] == accepted_result_cid
    assert reconstructed["post_verification"]["accepted_result_cid"] == (accepted_result_cid)
    authority = {key: reconstructed[key] for key in SEAL_AUTHORITY_KEYS}
    assert reconstructed["authority_root"] == module._identity(authority)
    assert reconstructed["claim"]["state"] == "released"
    assert reconstructed["preparation"]["evidence_digest"] == accepted_result_cid
    assert reconstructed["validation_receipt"]["outcome"] == "passed"
    assert reconstructed["seal_basis_evidence_receipt"]["digest"] == accepted_result_cid
    assert reconstructed["control_cas"]["task"]["status"] == "completed"
    assert reconstructed["coordination_promotion"]["status"] == "succeeded"
    assert reconstructed["settled_lease"]["state"] == "released"
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["accepted_result_cid"] == accepted_result_cid
    claimed_receipt_cid = persisted.pop("receipt_cid")
    assert module._identity(persisted) == claimed_receipt_cid

    monkeypatch.setattr(module, "_load_config", lambda: config)
    monkeypatch.setattr(module, "build_population", lambda _config: population)
    assert module.main(["verify"]) == 0
    verification = json.loads(capsys.readouterr().out)
    assert verification["valid"] is True
    assert verification["seal_receipt_cid"] == claimed_receipt_cid


@requires_duckdb
def test_self_consistent_receipt_with_wrong_result_identity_fails_seal_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)

    sealed = _canonical_receipt(
        module,
        module.seal(config, population),
        operation="seal",
    )
    receipt_path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered.pop("receipt_cid")
    tampered["accepted_result_cid"] = "sha256:" + "f" * 64
    assert tampered["accepted_result_cid"] != sealed["accepted_result_cid"]
    tampered["receipt_cid"] = module._identity(tampered)
    module._write_receipt(receipt_path, tampered)

    with pytest.raises(
        module.MaterializationError,
        match="existing bootstrap seal is stale: accepted_result_cid",
    ):
        module._load_existing_seal_receipt(config, population)
    with pytest.raises(
        module.MaterializationError,
        match="existing bootstrap seal is stale: accepted_result_cid",
    ):
        module.seal(config, population)


@pytest.mark.parametrize(
    ("seal_receipt", "expected_error"),
    [
        (None, "accepted bootstrap seal receipt is absent"),
        (
            {
                "receipt_cid": "sha256:" + "3" * 64,
                "accepted_result_cid": "sha256:" + "4" * 64,
            },
            "bootstrap seal receipt disagrees with control authority",
        ),
    ],
)
def test_sealed_verification_requires_matching_receipt_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seal_receipt: dict[str, str] | None,
    expected_error: str,
) -> None:
    module = _load_materializer()
    config = _temporary_config()
    population = _temporary_population()
    accepted_result_cid = "sha256:" + "5" * 64
    monkeypatch.setattr(module, "_load_config", lambda: config)
    monkeypatch.setattr(module, "build_population", lambda _config: population)
    monkeypatch.setattr(
        module,
        "_verify_store",
        lambda *_args, **_kwargs: {"accepted_result_cid": accepted_result_cid},
    )
    monkeypatch.setattr(
        module,
        "_load_existing_seal_receipt",
        lambda *_args, **_kwargs: seal_receipt,
    )

    assert module.main(["verify"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is False
    assert result["error"] == expected_error


def test_sealed_verification_accepts_only_the_exact_result_identity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_materializer()
    config = _temporary_config()
    population = _temporary_population()
    accepted_result_cid = "sha256:" + "6" * 64
    seal_receipt = {
        "receipt_cid": "sha256:" + "7" * 64,
        "accepted_result_cid": accepted_result_cid,
    }
    monkeypatch.setattr(module, "_load_config", lambda: config)
    monkeypatch.setattr(module, "build_population", lambda _config: population)
    monkeypatch.setattr(
        module,
        "_verify_store",
        lambda *_args, **_kwargs: {"accepted_result_cid": accepted_result_cid},
    )
    monkeypatch.setattr(
        module,
        "_load_existing_seal_receipt",
        lambda *_args, **_kwargs: seal_receipt,
    )

    assert module.main(["verify"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["seal_receipt_cid"] == seal_receipt["receipt_cid"]


@requires_duckdb
def test_live_verifier_accepts_exact_seal_without_mutating_any_store_or_lock_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module, config, population, sealed = _sealed_temporary_plane(tmp_path, monkeypatch)
    paths = module._paths(config)
    assert module._verify_store(config, population, expected_stage="sealed")[
        "accepted_result_cid"
    ] == sealed["accepted_result_cid"]

    for lock_path in tmp_path.rglob("*.lock"):
        lock_path.unlink()
    entries_before = sorted(
        path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
    )
    stores_before = {
        key: (path.read_bytes(), path.stat().st_mtime_ns) for key, path in paths.items()
    }

    live = module._verify_live_store(config, population)

    assert live["verification_mode"] == "live"
    assert live["seal"]["accepted_result_cid"] == sealed["accepted_result_cid"]
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == (
        entries_before
    )
    assert not list(tmp_path.rglob("*.lock"))
    assert {
        key: (path.read_bytes(), path.stat().st_mtime_ns) for key, path in paths.items()
    } == stores_before

    monkeypatch.setattr(module, "_load_config", lambda: config)
    monkeypatch.setattr(module, "build_population", lambda _config: population)
    assert module.main(["verify-live"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True
    assert report["verification"]["seal"]["receipt_cid"] == sealed["receipt_cid"]


@requires_duckdb
def test_live_verifier_allows_real_task_progress_claim_provider_and_effect_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, _sealed = _sealed_temporary_plane(tmp_path, monkeypatch)
    paths = module._paths(config)
    schema_env = "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
    prior_schema = os.environ.get(schema_env)
    os.environ[schema_env] = str(config["database_program"]["schema_revision"])
    daemon: DatabaseImplementationDaemon | None = None
    try:
        daemon = DatabaseImplementationDaemon(
            database_path=paths["control"],
            coordination_path=paths["coordination"],
            execution_path=paths["execution"],
            owner_session_id="lgswf-live-verification-daemon",
            authority_mode="embedded",
            task_source_kind="duckdb",
            install_schema=False,
        )
        attempt = daemon.claim_next()
        assert attempt is not None and attempt.task_alias == "LGSWF-001"
        attempt = daemon.commit_phase(attempt, "context", body={"live_test": True})
        attempt, provider_result, replayed = daemon.run_provider(attempt)
        assert replayed is False
        attempt, _effect_result, replayed = daemon.run_effect(attempt, provider_result)
        assert replayed is False
    finally:
        if daemon is not None:
            daemon.close()
        if prior_schema is None:
            os.environ.pop(schema_env, None)
        else:
            os.environ[schema_env] = prior_schema

    coordinator = open_database_coordinator(paths["coordination"])
    try:
        resource_claims = [
            coordinator.claim_resource(
                resource_kind=kind,
                resource_id=f"live-{kind}",
                owner_session_id="lgswf-live-verification-daemon",
                task_cid=attempt.task_cid,
                repository_id="repository:lgswf-test",
                path=("outputs/live-path.json" if kind == "path" else ""),
            )
            for kind in ("provider", "path", "prover", "merge")
        ]
        for resource_claim in resource_claims[1:]:
            lease = coordinator.get_lease(resource_claim.lease_id)
            assert lease is not None
            coordinator.release(lease, reason="live verifier fixture settled")
        maintenance = coordinator.acquire_maintenance_lease(
            owner_session_id="lgswf-live-maintenance",
            scope="live-verifier-fixture",
        )
        maintenance_lease = coordinator.get_lease(maintenance.lease_id)
        assert maintenance_lease is not None
        coordinator.release(maintenance_lease, reason="live verifier fixture settled")
    finally:
        coordinator.close()

    live = module._verify_live_store(config, population)
    assert live["control"]["counts"]["task_revisions"] > len(population["tasks"])
    assert live["coordination"]["active_task_claims"] == 1
    assert live["coordination"]["active_resource_claims"] == 1
    assert live["coordination"]["active_fenced_leases"] == 2
    assert live["execution"]["row_counts"] == {
        "attempt_phases": 4,
        "daemon_execution_events": 6,
        "database_task_attempts": 1,
        "effect_claims": 1,
        "provider_invocations": 1,
    }
    with pytest.raises(module.MaterializationError):
        module._verify_store(config, population, expected_stage="sealed")


@requires_duckdb
@pytest.mark.parametrize("forgery", ["task", "goal", "dependency"])
def test_live_verifier_rejects_forged_static_control_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    module, config, population, _sealed = _sealed_temporary_plane(tmp_path, monkeypatch)
    connection = open_duckdb_connection(module._paths(config)["control"])
    try:
        if forgery == "task":
            connection.execute(
                "UPDATE tasks SET priority = 'FORGED' WHERE task_cid = 'task:LGSWF-001'"
            )
        elif forgery == "goal":
            connection.execute(
                "UPDATE goals SET title = 'FORGED' WHERE goal_cid = 'goal:lgswf-root'"
            )
        else:
            connection.execute(
                "DELETE FROM task_dependencies WHERE task_cid = 'task:LGSWF-001'"
            )
    finally:
        connection.close()

    with pytest.raises(module.MaterializationError, match="sealed .* changed"):
        module._verify_live_store(config, population)


@requires_duckdb
def test_live_verifier_rejects_impossible_claim_history_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, _sealed = _sealed_temporary_plane(tmp_path, monkeypatch)
    paths = module._paths(config)
    schema_env = "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"
    prior_schema = os.environ.get(schema_env)
    os.environ[schema_env] = str(config["database_program"]["schema_revision"])
    daemon = DatabaseImplementationDaemon(
        database_path=paths["control"],
        coordination_path=paths["coordination"],
        execution_path=paths["execution"],
        owner_session_id="lgswf-live-forgery-daemon",
        authority_mode="embedded",
        task_source_kind="duckdb",
        install_schema=False,
    )
    try:
        attempt = daemon.claim_next()
        assert attempt is not None
    finally:
        daemon.close()
        if prior_schema is None:
            os.environ.pop(schema_env, None)
        else:
            os.environ[schema_env] = prior_schema

    connection = open_duckdb_connection(paths["coordination"])
    try:
        connection.execute(
            "UPDATE task_attempts SET fencing_token = fencing_token + 100 "
            "WHERE attempt_id = ?",
            [attempt.attempt_id],
        )
    finally:
        connection.close()
    with pytest.raises(module.MaterializationError, match="impossible identities"):
        module._verify_live_store(config, population)


@requires_duckdb
def test_live_verifier_rejects_tampered_content_addressed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, _sealed = _sealed_temporary_plane(tmp_path, monkeypatch)
    receipt_path = module._bootstrap_receipt_path(config, "duckdb-seal.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["accepted_result_cid"] = "sha256:" + "f" * 64
    module._write_receipt(receipt_path, receipt)

    with pytest.raises(module.MaterializationError, match="receipt CID does not verify"):
        module._verify_live_store(config, population)


@pytest.mark.parametrize(
    ("command_result", "expected_database_error"),
    [
        ((0, "{}", ""), None),
        ((9, "", "live rejection"), "live rejection"),
    ],
)
def test_board_check_all_delegates_to_live_verifier(
    monkeypatch: pytest.MonkeyPatch,
    command_result: tuple[int, str, str],
    expected_database_error: str | None,
) -> None:
    validator = _load_validator()
    calls: list[tuple[list[str], Path]] = []

    def command(argv: list[str], *, cwd: Path) -> tuple[int, str, str]:
        calls.append((argv, cwd))
        return command_result

    monkeypatch.setattr(validator, "_command", command)
    report = validator.validate(require_database=True)

    assert calls == [
        (
            [sys.executable, str(validator.MATERIALIZER), "verify-live"],
            validator.ROOT,
        )
    ]
    database_errors = [
        item for item in report["errors"] if item.startswith("DuckDB control-plane verification")
    ]
    if expected_database_error is None:
        assert database_errors == []
    else:
        assert len(database_errors) == 1
        assert expected_database_error in database_errors[0]
