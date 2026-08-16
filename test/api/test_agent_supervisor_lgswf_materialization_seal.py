"""Focused qualification for the two-stage LGSWF bootstrap authority.

The fixtures deliberately redirect the materializer to ``tmp_path``.  They do
not construct, inspect, or mutate the configured ``run-actual-v3`` namespace.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinator,
    open_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationDaemon,
)

ROOT = Path(__file__).resolve().parents[2]
MATERIALIZER = ROOT / "scripts/materialize_logic_governed_semantic_work_fabric_control_plane.py"
requires_duckdb = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for the temporary LGSWF control-plane fixture",
)
RECEIPT_RESULT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/content-addressed-receipt-result@1"


def _load_materializer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"lgswf_materializer_test_{id(object())}", MATERIALIZER
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
            "outputs": [],
            "acceptance": [],
            "validations": [],
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


def _install_test_qualification(
    module: ModuleType,
    config: dict[str, Any],
    population: dict[str, Any],
) -> dict[str, Any]:
    commands = module._qualification_commands()
    receipt = {
        "schema": module.QUALIFICATION_SCHEMA,
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "population_cid": module._identity(population),
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

    assert population["source_head"] == execution_head
    assert population["repository_tree_id"] == execution_tree
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
    finally:
        daemon.close()


@requires_duckdb
def test_restart_resumes_exact_live_claim_before_preparation_without_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, config, population, materialized = _materialize_temporary_plane(tmp_path)
    _canonical_receipt(module, materialized, operation="materialize")
    _stub_seal_evidence(module, monkeypatch, config, population)
    captured: dict[str, Any] = {}

    def crash_before_preparation(
        _coordinator: DatabaseCoordinator,
        claim: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        captured["claim"] = claim.to_dict()
        raise RuntimeError("injected crash before PREPARED")

    with monkeypatch.context() as injected:
        injected.setattr(
            DatabaseCoordinator,
            "prepare_task_completion",
            crash_before_preparation,
        )
        with pytest.raises(RuntimeError, match="injected crash before PREPARED"):
            module.seal(config, population)

    paths = module._paths(config)
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
        assert "active" in str(seal_error.value).lower()
        assert "lease" in str(seal_error.value).lower()
    finally:
        coordinator = open_database_coordinator(paths["coordination"])
        try:
            lease = coordinator.get_lease(resource.lease_id)
            assert lease is not None
            coordinator.release(lease, reason="test cleanup")
        finally:
            coordinator.close()


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
    assert sealed["control_cas"]["changed"] is True
    assert sealed["control_cas"]["task"]["status"] == "completed"
    assert (
        sealed["control_cas"]["task"]["body"]["completion_receipt"]["accepted_result_cid"]
        == accepted_result_cid
    )
    assert sealed["coordination_promotion"]["status"] == "succeeded"
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
