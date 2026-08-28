"""Focused tests for the operator-owned SAWM R2 controls.

These tests inspect and render controls only. They do not open the live
authority database, start Quack, probe a provider, or launch a supervisor.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(relative: str, name: str) -> ModuleType:
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reidentify_extension_projection(pin: dict[str, object]) -> None:
    body = {key: value for key, value in pin.items() if key != "projection_id"}
    pin["projection_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_static_board_gate_is_valid() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_validator_test",
    )
    report = validator.validate_program(REPO_ROOT)
    assert report["valid"] is True, report["errors"]
    assert report["task_count"] == 45
    assert report["goal_count"] == 29
    assert report["markdown_completion_is_authority"] is False


def test_dependency_gate_qualifies_the_exact_isolated_launch_stack() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_validator_test",
    )
    report = validator.validate_dependencies(REPO_ROOT, cold_import=True)
    assert report["valid"] is True, report["errors"]
    assert report["database_opened"] is True
    assert report["network_required"] is False

    checks = {item["name"]: item for item in report["checks"]}
    for name in (
        "sealed_launch_toolchain_declaration",
        "accepted_control_plane_mode_closure",
        "exact_immutable_validation_runtime_closure",
        "independent_native_dependency_authorization",
        "quack_httpfs_projection_pins",
        "isolated_launch_toolchain",
        "recomputed_native_dependency_pin",
        "isolated_duckdb_quack_httpfs_load",
        "cold_import_side_effects",
    ):
        assert checks[name]["passed"] is True, checks[name]["detail"]
    mode_closure = checks["accepted_control_plane_mode_closure"]["detail"]
    assert mode_closure["argv_flags"] == ["-I", "-S", "-B"]
    assert mode_closure["file_count"] >= 852
    assert mode_closure["errors"] == []
    assert all(
        int(mode, 8) & 0o022 == 0
        for mode in mode_closure["mode_counts"]
    )
    isolated = checks["isolated_duckdb_quack_httpfs_load"]["detail"]
    assert isolated["python_executable"] == "/usr/bin/python3.12"
    assert isolated["argv_flags"] == ["-I", "-S", "-B"]
    assert isolated["ambient_specs"] == {
        "duckdb": False,
        "_duckdb": False,
        "pytest": False,
    }
    assert isolated["native_module_origin"].startswith("/proc/self/fd/")
    assert [row[0] for row in isolated["extension_rows"]] == ["httpfs", "quack"]
    assert isolated["extension_path_reporting"] == (
        "normalized_after_exact_runtime_path_verification"
    )
    assert all(
        row[4].startswith("$ISOLATED_HOME/.duckdb/extensions/")
        for row in isolated["extension_rows"]
    )
    assert isolated["settings"] == [False, False, False, False]
    assert isolated["select_42"] == 42
    runtime = checks["exact_immutable_validation_runtime_closure"]["detail"]
    assert runtime["python_executable"] == "/usr/bin/python3.12"
    assert runtime["pythonpath_entries"] == [
        "/opt/ipfs-accelerate-aseh-validation-9b3ba6caebcf/site-packages-py-multihash",
        "/opt/ipfs-accelerate-aseh-validation-9b3ba6caebcf/site-packages",
        "/opt/ipfs-accelerate-legal-validation-7ffe92439767/site-packages",
    ]
    assert runtime["payloads"]["delta_deployment"]["manifest_entry_count"] == 4526
    assert runtime["payloads"]["base_deployment"]["manifest_entry_count"] == 28058
    assert runtime["import_probe"]["module_count"] == 39
    dependency = runtime["project_dependency_preflight"]
    assert dependency["valid"] is True
    assert dependency["passed"] is True
    assert dependency["reason"] == (
        "approved_validation_environment_satisfies_project_dependencies"
    )
    assert dependency["path_sensitive"] is True
    assert dependency["validation_command_count"] == 46
    assert dependency["requirements_count"] == 43
    assert dependency["missing_count"] == 0
    assert dependency["incompatible_count"] == 0
    assert dependency["invalid_count"] == 0
    assert dependency["failure"] is None


def test_m6_validation_runtime_tamper_and_absolute_launcher_fail_closed() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_validation_runtime_tamper_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )
    exact = validator._validation_runtime_closure(
        REPO_ROOT,
        config,
        seal,
        rehash_payloads=False,
        probe_imports=False,
        probe_dependencies=False,
    )
    assert exact["valid"] is True, exact["errors"]

    tampered = copy.deepcopy(config)
    tampered["validation_runtime"]["delta_deployment"]["artifacts"][0][
        "sha256"
    ] = "sha256:" + ("00" * 32)
    rejected = validator._validation_runtime_closure(
        REPO_ROOT,
        tampered,
        seal,
        rehash_payloads=False,
        probe_imports=False,
        probe_dependencies=False,
    )
    assert rejected["valid"] is False
    assert any("validation_runtime" in error for error in rejected["errors"])

    _historical, operational, command_errors = (
        validator._historical_and_operational_validation_commands(REPO_ROOT)
    )
    assert command_errors == []
    assert len(operational) == 46
    assert all(
        command.startswith("PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. python ")
        for command in operational
    )
    from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
        ValidationRuntimeError,
        validation_shell_command,
    )

    validation_shell_command(operational[0])
    with pytest.raises(
        ValidationRuntimeError,
        match="sealed python or python3 launcher",
    ):
        validation_shell_command(
            operational[0].replace(
                " python ", " /home/barberb/.local/bin/python ", 1
            )
        )


def test_dependency_gate_rejects_projection_identity_drift_from_native() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_native_projection_binding_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )

    projections, errors = validator._validate_extension_projection_pins(
        seal,
        config,
    )
    assert errors == []
    assert projections["quack"]["engine_version"] == "v1.5.5"
    assert projections["quack"]["platform"] == "linux_arm64"
    assert projections["httpfs"]["engine_version"] == "v1.5.5"
    assert projections["httpfs"]["platform"] == "linux_arm64"

    for field, value in (
        ("engine_version", "v9.9.9"),
        ("platform", "linux_amd64"),
    ):
        wrong = copy.deepcopy(seal)
        pin = wrong["configured_board_quack_projection"]["pin"]
        pin[field] = value
        _reidentify_extension_projection(pin)
        _, drift_errors = validator._validate_extension_projection_pins(
            wrong,
            config,
        )
        assert (
            "configured-board extension engine/platform differs from "
            "native DuckDB/toolchain"
        ) in drift_errors

    wrong_httpfs = copy.deepcopy(seal)
    wrong_httpfs_config = copy.deepcopy(config)
    httpfs_pin = wrong_httpfs["httpfs_extension_pin"]
    wrong_parent = Path(httpfs_pin["path"]).parent.parent / "linux_amd64"
    httpfs_pin["path"] = str(wrong_parent / "httpfs.duckdb_extension")
    httpfs_pin["info_path"] = str(
        wrong_parent / "httpfs.duckdb_extension.info"
    )
    wrong_httpfs_config["quack_owner"]["pinned_httpfs_extension"] = copy.deepcopy(
        httpfs_pin
    )
    _, httpfs_errors = validator._validate_extension_projection_pins(
        wrong_httpfs,
        wrong_httpfs_config,
    )
    assert (
        "httpfs extension path engine/platform differs from "
        "native DuckDB/toolchain"
    ) in httpfs_errors


def test_rendered_population_has_exact_operator_frontier() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_test",
    )
    population = materializer.build_population(REPO_ROOT)
    tasks = population["taskboard"]
    goals = population["objectives"]
    assert len(tasks) == 45
    assert len(goals) == 29
    assert [task["task_id"] for task in tasks] == [
        f"SAWM-{index:03d}" for index in range(45)
    ]
    assert tasks[0]["status"] == "ready"
    assert all(task["status"] == "todo" for task in tasks[1:])
    assert tasks[1]["depends_on"] == [tasks[0]["task_cid"]]
    assert all(str(goal["goal_cid"]).startswith("sha256:") for goal in goals)
    assert population["plan_revision"] == "SAWM-PLAN-R2"


def test_nonroot_goal_objective_projection_is_optional_and_closed() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_optional_goal_objective_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_optional_goal_objective_test",
    )
    goals = materializer.build_population(REPO_ROOT)["objectives"]
    root = next(goal for goal in goals if goal["goal_id"] == "SAWM-G000")
    children = [goal for goal in goals if goal["goal_id"] != "SAWM-G000"]

    assert root["objective_id"]
    assert children
    assert all("objective_id" not in goal for goal in children)
    assert all(str(goal.get("objective_id") or "") == "" for goal in children)

    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert 'str(expected.get("objective_id") or "")' in source
    assert 'expected["objective_id"]' not in source


def test_managed_daemon_uses_the_supervisors_explicit_board_lock(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        board_scoped_checkout_mutation_lock_path,
        board_scoped_protected_path_maintenance_lock_path,
        checkout_mutation_lock_path,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
        parse_args as parse_daemon_args,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
    )

    namespace = "semantic-addressed-world-model-v1"
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.board_namespace = namespace
    supervisor = object.__new__(PortalImplementationSupervisor)
    supervisor.board_namespace = namespace
    supervisor.config = SimpleNamespace(repo_root=tmp_path)

    expected = board_scoped_checkout_mutation_lock_path(tmp_path, namespace)
    assert daemon._repo_merge_lock_path() == expected
    assert supervisor._repo_merge_lock_path() == expected
    maintenance = daemon._protected_path_maintenance_lock_path()
    assert maintenance == (
        board_scoped_protected_path_maintenance_lock_path(tmp_path, namespace)
    )
    assert expected != checkout_mutation_lock_path(tmp_path)
    assert maintenance != checkout_mutation_lock_path(tmp_path)
    assert expected != board_scoped_checkout_mutation_lock_path(
        tmp_path,
        "independent-board-v1",
    )
    assert parse_daemon_args(
        ["--board-namespace", namespace, "--once"]
    ).board_namespace == namespace

    daemon.board_namespace = ""
    assert daemon._repo_merge_lock_path() == checkout_mutation_lock_path(
        tmp_path
    )


def test_preserved_definition_cids_rehash_from_the_prior_source_binding() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_identity_test",
    )
    population = materializer.build_population(REPO_ROOT)
    prior = population["migration_inventory"]

    for record in (*population["objectives"], *population["taskboard"]):
        assert record["definition"]["source_binding_cid"] == prior[
            "definition_source_binding_cid"
        ]
        assert materializer._identity(record["definition"]) == record["definition_cid"]
    assert materializer._identity(population["program_definition"]) == population[
        "program_definition_cid"
    ]


def test_materialization_rejects_a_dirty_or_uncommitted_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_dirty_test",
    )
    population = materializer.build_population(REPO_ROOT)
    real_git = materializer._git

    def dirty_git(root: Path, *args: str, **kwargs: object) -> str:
        if args[:2] == ("status", "--porcelain=v1"):
            return " M scripts/materialize_semantic_addressed_world_model_program.py"
        return real_git(root, *args, **kwargs)

    monkeypatch.setattr(materializer, "_git", dirty_git)
    with pytest.raises(materializer.MaterializationError, match="clean committed"):
        materializer._assert_committed_clean_source(REPO_ROOT, population)

    monkeypatch.setattr(materializer, "_git", real_git)
    stale_population = copy.deepcopy(population)
    stale_population["source_binding"]["head"] = "0" * 40
    with pytest.raises(
        materializer.MaterializationError,
        match="source binding changed",
    ):
        materializer._assert_source_binding_matches(
            REPO_ROOT,
            stale_population,
        )


def test_materialization_refuses_preexisting_successor_sidecars(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_fresh_sidecars_test",
    )
    target = tmp_path / "control.duckdb"
    (tmp_path / "control.execution.duckdb").touch(mode=0o600)
    with pytest.raises(
        materializer.MaterializationError,
        match="execution/coordination authority is not fresh",
    ):
        materializer._assert_fresh_successor_operational_state(target)


def test_append_only_source_migration_rehearsal_verifies_exactly() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_migration_test",
    )
    population = materializer.build_population(REPO_ROOT)
    migration = population["migration_inventory"]
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    prior = materializer._verify_prior_store(REPO_ROOT, config, population)
    dependency = materializer._validator_report(
        REPO_ROOT, "scripts/validate_semantic_addressed_world_model_dependencies.py"
    )
    board = materializer._validator_report(
        REPO_ROOT, "scripts/validate_semantic_addressed_world_model_board.py"
    )
    validation_digest = materializer._identity(
        {
            "dependency": dependency,
            "board": board,
            "program_definition_cid": population["program_definition_cid"],
        }
    )

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(prefix="sawm-r2-test-", dir="/tmp") as directory:
        stage = Path(directory) / "control.duckdb"
        shutil.copy2(Path(prior["database_path"]), stage)
        source = DatabaseTaskSource(
            stage,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
            owner_id="sawm-r2-source-migrator",
        )
        try:
            operator = source.get_task("SAWM-000")
            assert operator is not None
            migration_body = materializer._migration_body(
                population, config, validation_digest
            )
            assert migration_body["preworker_launch_failure"] == migration[
                "preworker_launch_failure"
            ]
            assert migration_body["preprovider_task_failure"] == migration[
                "preprovider_task_failure"
            ]
            recovery_receipt = materializer._task_recovery_receipt(population)
            assert migration_body["nonterminal_task_recovery_receipt"] == (
                recovery_receipt
            )
            migration_digest = materializer._identity(migration_body)
            source.plans.append_revision(
                plan_cid=str(population["plan_root_cid"]),
                expected_revision=int(migration["prior_plan_revision"]),
                body={
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": migration["migration_revision"],
                    "source_migration_digest": migration_digest,
                    "supersession_mode": materializer._M6_SUPERSESSION_MODE,
                },
                delta=materializer._migration_plan_delta(population),
            )
            source.record_evidence(
                task_cid=operator.task_cid,
                evidence_kind="operator_control_plane_source_migration",
                digest=migration_digest,
                body=migration_body,
            )
            operational_event_ids = {}
            for alias in materializer._operational_task_aliases():
                prior_task = source.intent.get_task(alias)
                assert prior_task is not None
                expected_status = "blocked" if alias == "SAWM-001" else "todo"
                expected_revision = 5 if alias == "SAWM-001" else 1
                assert (prior_task["status"], prior_task["revision"]) == (
                    expected_status,
                    expected_revision,
                )
                prior_validations = tuple(
                    tuple(str(part) for part in item.get("argv") or ())
                    for item in prior_task["validations"]
                )
                operational_validations, replacement_count = (
                    materializer._operational_validation_commands(
                        prior_validations,
                        task_alias=alias,
                    )
                )
                operational_receipt = materializer._operational_validation_receipt(
                    population,
                    task_alias=alias,
                    task_cid=prior_task["task_cid"],
                    expected_status=expected_status,
                    expected_revision=expected_revision,
                    prior_validations=prior_validations,
                    operational_validations=operational_validations,
                    replacement_count=replacement_count,
                )
                upsert = source.intent.upsert_task(
                    task_cid=prior_task["task_cid"],
                    task_alias=prior_task["task_alias"],
                    goal_cid=prior_task["goal_cid"],
                    ordinal=prior_task["ordinal"],
                    status=prior_task["status"],
                    priority=prior_task["priority"],
                    plan_cid=prior_task["plan_cid"],
                    objective_id=prior_task["objective_id"],
                    body={
                        **dict(prior_task["body"]),
                        "operational_validation_revision": operational_receipt,
                    },
                    identity=dict(prior_task["identity"]),
                    expected_revision=expected_revision,
                    dependencies=list(prior_task["dependencies"]),
                    outputs=[
                        dict(item["effect"]) for item in prior_task["outputs"]
                    ],
                    acceptance=[
                        dict(item["evidence_policy"])
                        for item in prior_task["acceptance"]
                    ],
                    validations=[
                        {**dict(prior["policy"]), "argv": list(argv)}
                        for prior, argv in zip(
                            prior_task["validations"],
                            operational_validations,
                            strict=True,
                        )
                    ],
                )
                operational_event_ids[alias] = upsert.event_id
            candidate = source.get_task("SAWM-001")
            assert candidate is not None
            assert (candidate.status, candidate.revision) == ("blocked", 6)
            recovery = source.compare_and_set_status(
                candidate.task_cid,
                expected_revision=6,
                status="todo",
                receipt=recovery_receipt,
            )
            assert recovery.changed is True
            assert recovery.previous_status == "blocked"
            assert (recovery.task.status, recovery.task.revision) == ("todo", 7)
        finally:
            source.close()

        verified = materializer._verify_store(
            stage,
            population,
            require_operator_complete=True,
            require_migration=True,
            migration_config=config,
            expected_validation_digest=validation_digest,
        )
        assert verified["event_watermark"] == migration["prior_event_watermark"] + 47
        assert verified["migration_event_watermark"] == 123
        assert verified["operational_validation_first_event_watermark"] == 124
        assert verified["operational_validation_last_event_watermark"] == 167
        assert verified["target_event_watermark"] == 168
        assert verified["projection_cid"] == materializer._M6_EXPECTED_PROJECTION_CID
        assert verified["statuses"]["SAWM-001"] == "todo"
        assert verified["revisions"]["SAWM-001"] == 7
        assert all(verified["revisions"][f"SAWM-{index:03d}"] == 2 for index in range(2, 45))
        assert verified["accepted_definition_changes"] == 0
        assert verified["accepted_completion_changes"] == 0
        assert verified["nonterminal_task_status_recovery_changes"] == 1
        assert verified["task_recovery_receipt"] == recovery_receipt
        assert verified["projection_matches_events"] is True
        assert materializer._store_sha256(Path(prior["database_path"])) == migration[
            "prior_control_store_sha256"
        ]
        stale_lock = stage.parent / ".migration-receipt.json.publish.lock"
        stale_lock.touch(mode=0o600)
        receipt = materializer._ensure_migration_receipt(
            stage.parent, stage, population, verified, validation_digest
        )
        receipt_path = stage.parent / "migration-receipt.json"
        assert receipt_path.is_file()
        assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@3"
        assert receipt["migration_event_watermark"] == 123
        assert receipt["operational_validation_first_event_watermark"] == 124
        assert receipt["operational_validation_last_event_watermark"] == 167
        assert receipt["target_event_watermark"] == 168
        assert receipt["migration_projection_cid"] != receipt["projection_cid"]
        assert receipt["projection_cid"] == materializer._M6_EXPECTED_PROJECTION_CID
        assert receipt["task_recovery_event_id"] == verified[
            "task_recovery_event_id"
        ]
        assert receipt["task_recovery_receipt"] == recovery_receipt
        assert receipt["nonterminal_task_status_recovery_changes"] == 1
        assert not (stage.parent / "control.execution.duckdb").exists()
        assert not (stage.parent / "control.coordination.duckdb").exists()
        assert materializer._ensure_migration_receipt(
            stage.parent, stage, population, verified, validation_digest
        ) == receipt
        receipt_path.unlink()
        assert materializer._ensure_migration_receipt(
            stage.parent, stage, population, verified, validation_digest
        ) == receipt


def test_m7_source_only_migration_rehearsal_and_tamper_gates() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m7_rehearsal_test",
    )
    population = materializer.build_population(REPO_ROOT)
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    validation_digest = materializer._m7_validation_digest(
        REPO_ROOT,
        population,
    )
    authority = config["source_repair_materialization"]
    prior_path = REPO_ROOT / authority["prior_store_id"]
    prior_hash = materializer._store_sha256(prior_path)
    prior_append_surface_digest = materializer._append_surface_digest(prior_path)
    assert prior_append_surface_digest == authority["prior_append_surface_digest"]

    class AdapterRow:
        def __init__(self, columns: tuple[str, ...], values: tuple[object, ...]):
            self.columns = columns
            self.values = values

        def __getitem__(self, index: int) -> object:
            return self.values[index]

        def __iter__(self):
            return iter(self.columns)

    adapter_row = AdapterRow(("left", "right"), ("value-left", "value-right"))
    assert tuple(adapter_row) == ("left", "right")
    assert adapter_row != ("value-left", "value-right")
    assert materializer._positional_rows([adapter_row], 2) == [
        ("value-left", "value-right")
    ]

    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m7-test-",
        dir="/tmp",
    ) as directory:
        stage = Path(directory) / "control.duckdb"
        shutil.copy2(prior_path, stage)
        before_semantic = materializer._semantic_authority_digest(stage)
        before_frozen = materializer._frozen_base_authority_digest(stage)
        source = DatabaseTaskSource(
            stage,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
            owner_id="sawm-r2-source-migrator",
        )
        try:
            operator = source.get_task("SAWM-000")
            assert operator is not None
            body = materializer._m7_migration_body(
                population,
                config,
                validation_digest,
            )
            failure = materializer._M7_PREPUBLICATION_MATERIALIZATION_FAILURE
            assert body["schema"] == (
                "sawm/operator-control-plane-source-migration@5"
            )
            assert body["prepublication_materialization_failure"] == failure
            assert body["prepublication_materialization_failure_cid"] == (
                materializer._identity(failure)
            )
            assert failure["target_published"] is False
            assert failure["implementation_provider_invoked"] is False
            digest = materializer._identity(body)
            delta = materializer._m7_migration_plan_delta(population, config)
            source.plans.append_revision(
                plan_cid=str(population["plan_root_cid"]),
                expected_revision=7,
                body={
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": materializer._M7_MIGRATION_REVISION,
                    "source_migration_digest": digest,
                    "supersession_mode": materializer._M7_SUPERSESSION_MODE,
                },
                delta=delta,
            )
            source.record_evidence(
                task_cid=operator.task_cid,
                evidence_kind="operator_control_plane_source_migration",
                digest=digest,
                body=body,
            )
        finally:
            source.close()

        verified = materializer._verify_m7_store(
            stage,
            population,
            config,
            validation_digest,
        )
        assert verified["event_watermark"] == 170
        assert verified["projection_cid"] == (
            materializer._M7_EXPECTED_PROJECTION_CID
        )
        assert verified["task_revision_changes"] == 0
        assert verified["task_status_changes"] == 0
        assert verified["accepted_definition_changes"] == 0
        assert verified["accepted_completion_changes"] == 0
        assert verified["projection_matches_events"] is True
        assert materializer._semantic_authority_digest(stage) == before_semantic
        assert materializer._frozen_base_authority_digest(stage) == before_frozen
        assert before_semantic == authority["prior_semantic_authority_digest"]
        assert before_frozen == authority["prior_frozen_base_authority_digest"]
        assert materializer._store_sha256(prior_path) == prior_hash

        import duckdb

        forged_evidence = Path(directory) / "forged-evidence.duckdb"
        shutil.copy2(stage, forged_evidence)
        connection = duckdb.connect(str(forged_evidence))
        try:
            connection.execute(
                "INSERT INTO evidence_nodes "
                "SELECT ?, parent_evidence_id, task_cid, evidence_kind, digest, "
                "created_at, body_json FROM evidence_nodes LIMIT 1",
                ["sha256:" + "1" * 64],
            )
        finally:
            connection.close()
        with pytest.raises(
            materializer.MigrationRequired,
            match="append-surface",
        ):
            materializer._verify_m7_store(
                forged_evidence,
                population,
                config,
                validation_digest,
            )

        forged_plan = Path(directory) / "forged-plan.duckdb"
        shutil.copy2(stage, forged_plan)
        connection = duckdb.connect(str(forged_plan))
        try:
            connection.execute(
                "UPDATE plans SET body_json = ?",
                ['{"forged":true}'],
            )
        finally:
            connection.close()
        with pytest.raises(
            materializer.MigrationRequired,
            match="plan body",
        ):
            materializer._verify_m7_store(
                forged_plan,
                population,
                config,
                validation_digest,
            )

        forged_state = Path(directory) / "forged-state.duckdb"
        shutil.copy2(stage, forged_state)
        connection = duckdb.connect(str(forged_state))
        try:
            connection.execute(
                "UPDATE state_servers SET status = 'running' WHERE generation = 8"
            )
        finally:
            connection.close()
        with pytest.raises(
            materializer.MigrationRequired,
            match="frozen base authority",
        ):
            materializer._verify_m7_store(
                forged_state,
                population,
                config,
                validation_digest,
            )


def test_m7_controls_remain_immutable_historical_authority() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m7_source_repair_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m7_source_repair_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    assert dependency_validator._m7_source_repair_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m7_migration_errors(config, seal, migration) == []

    changed_authority = copy.deepcopy(config)
    changed_authority["source_repair_materialization"][
        "accepted_completion_changes"
    ] = 1
    assert dependency_validator._m7_source_repair_errors(
        changed_authority,
        seal,
        migration,
    )
    drifted_seal = copy.deepcopy(seal)
    drifted_seal["source_repair_materialization_cid"] = "sha256:" + "0" * 64
    assert dependency_validator._m7_source_repair_errors(
        config,
        drifted_seal,
        migration,
    )

    # Active M8 runtime movement cannot invalidate the frozen M7 authority.
    stale_runtime = copy.deepcopy(config)
    stale_runtime["runtime_paths"]["root"] = (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m6"
    )
    assert dependency_validator._m7_source_repair_errors(
        stale_runtime,
        seal,
        migration,
    ) == []


def test_m8_controls_and_live_comparator_fail_closed() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m8_source_repair_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m8_source_repair_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m8_live_comparator_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m8_dispatch_fail_closed_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    expected_cid = (
        "sha256:01935058ca682411743904513524b3eab41369b057151db529af14d46c5c0963"
    )
    authority = config["source_repair_successor_materialization"]
    assert authority == migration["source_repair_successor_materialization"]
    assert "sha256:" + hashlib.sha256(
        json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest() == expected_cid
    assert seal["source_repair_successor_materialization_cid"] == expected_cid
    assert dependency_validator._m8_source_repair_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m8_migration_errors(config, seal, migration) == []

    # Exercise the historical M8 selector in isolation.  M11, M10, and M9
    # keys intentionally have precedence, including fail-closed malformed
    # handling.
    malformed_successor = copy.deepcopy(config)
    malformed_successor.pop("live_provider_retry_successor_materialization")
    malformed_successor.pop("live_projection_successor_materialization")
    malformed_successor.pop("live_recovery_successor_materialization")
    malformed_successor["source_repair_successor_materialization"] = []
    with pytest.raises(
        operator.OperatorError,
        match="active source-only successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed_successor)
    with pytest.raises(
        materializer.MaterializationError,
        match="M8 source-only successor authority is invalid",
    ):
        materializer._m8_successor_configured(malformed_successor)

    changed_authority = copy.deepcopy(config)
    changed_authority["source_repair_successor_materialization"][
        "accepted_completion_changes"
    ] = 1
    assert dependency_validator._m8_source_repair_errors(
        changed_authority,
        seal,
        migration,
    )

    changed_failure_config = copy.deepcopy(config)
    changed_failure_inventory = copy.deepcopy(migration)
    for controls in (changed_failure_config, changed_failure_inventory):
        controls["source_repair_successor_materialization"][
            "live_preflight_failure"
        ]["implementation_provider_invoked"] = True
    assert dependency_validator._m8_source_repair_errors(
        changed_failure_config,
        seal,
        changed_failure_inventory,
    )

    stale_runtime = copy.deepcopy(config)
    stale_runtime.pop("live_recovery_successor_materialization")
    stale_runtime["runtime_paths"]["root"] = (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m7"
    )
    assert dependency_validator._m8_source_repair_errors(
        stale_runtime,
        seal,
        migration,
    )

    drifted_seal = copy.deepcopy(seal)
    drifted_seal["source_repair_successor_materialization_cid"] = (
        "sha256:" + "0" * 64
    )
    assert dependency_validator._m8_source_repair_errors(
        config,
        drifted_seal,
        migration,
    )

    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert "materializer._verify_m6_task_projection(live, population)" in source
    assert "materializer._semantic_authority_digest_on(connection)" in source
    assert (
        'successor_key = "source_repair_successor_materialization"' in source
    )
    assert (
        'expected_event_cursor = int(active_source_repair["target_event_watermark"])'
        in source
    )
    assert 'str(expected.get("objective_id") or "")' in source
    assert 'expected["objective_id"]' not in source
    assert "checked = materializer.check_materialized(REPO_ROOT, config_path)" in source


def test_m8_materializer_control_rehearsal_and_tamper_gate() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m8_control_rehearsal_test",
    )
    population = materializer.build_population(REPO_ROOT)
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    authority = materializer._m8_source_repair_authority(population, config)
    validation_digest = materializer._m7_validation_digest(
        REPO_ROOT,
        population,
    )
    body = materializer._m8_migration_body(
        population,
        config,
        validation_digest,
    )
    delta = materializer._m8_migration_plan_delta(population, config)

    assert materializer._identity(authority) == (
        "sha256:01935058ca682411743904513524b3eab41369b057151db529af14d46c5c0963"
    )
    assert body["schema"] == "sawm/operator-control-plane-source-migration@6"
    assert body["migration_revision"] == "SAWM-R2-M8"
    assert body["supersession_reason"] == (
        "optional_nonroot_goal_objective_projection_repair"
    )
    assert body["live_preflight_failure"] == authority["live_preflight_failure"]
    assert body["accepted_goal_definitions_rewritten"] is False
    assert body["accepted_completion_changes"] == 0
    assert body["implementation_provider_invocations"] == 0
    assert body["worker_self_approval"] is False
    assert delta["kind"] == "optional_nonroot_goal_objective_projection_repair"
    assert delta["accepted_definition_changes"] == 0
    assert delta["accepted_completion_changes"] == 0
    assert delta["worker_self_approval"] is False

    prior_path = REPO_ROOT / authority["prior_store_id"]
    prior_hash = materializer._store_sha256(prior_path)
    materializer._assert_m8_prior_publication_anchor(
        REPO_ROOT,
        authority,
        prior_path.resolve(),
    )
    with tempfile.TemporaryDirectory(
        prefix="sawm-r2-m8-test-",
        dir="/tmp",
    ) as directory:
        stage = Path(directory) / "control.duckdb"
        shutil.copy2(prior_path, stage)
        before_semantic = materializer._semantic_authority_digest(stage)
        before_frozen = materializer._frozen_base_authority_digest(stage)
        before_append = materializer._append_surface_digest(stage)

        from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
            DatabaseTaskSource,
        )

        source = DatabaseTaskSource(
            stage,
            install_schema=False,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
            owner_id="sawm-r2-source-migrator",
        )
        try:
            operator_task = source.get_task("SAWM-000")
            assert operator_task is not None
            source.plans.append_revision(
                plan_cid=str(population["plan_root_cid"]),
                expected_revision=8,
                body={
                    "current_source_binding_cid": population["source_binding"][
                        "source_binding_cid"
                    ],
                    "source_migration_revision": materializer._M8_MIGRATION_REVISION,
                    "source_migration_digest": materializer._identity(body),
                    "supersession_mode": materializer._M8_SUPERSESSION_MODE,
                },
                delta=delta,
            )
            source.record_evidence(
                task_cid=operator_task.task_cid,
                evidence_kind="operator_control_plane_source_migration",
                digest=materializer._identity(body),
                body=body,
            )
        finally:
            source.close()

        verified = materializer._verify_m8_store(
            stage,
            population,
            config,
            validation_digest,
        )
        assert verified["event_watermark"] == 172
        assert verified["projection_cid"] == materializer._M8_EXPECTED_PROJECTION_CID
        assert verified["projection_matches_events"] is True
        assert verified["task_revision_changes"] == 0
        assert verified["task_status_changes"] == 0
        assert verified["accepted_definition_changes"] == 0
        assert verified["accepted_completion_changes"] == 0
        assert materializer._semantic_authority_digest(stage) == before_semantic
        assert materializer._frozen_base_authority_digest(stage) == before_frozen
        assert before_semantic == authority["prior_semantic_authority_digest"]
        assert before_frozen == authority["prior_frozen_base_authority_digest"]
        assert before_append == authority["prior_append_surface_digest"]
        assert materializer._store_sha256(prior_path) == prior_hash

    changed_config = copy.deepcopy(config)
    changed_config["source_repair_successor_materialization"][
        "accepted_completion_changes"
    ] = 1
    with pytest.raises(
        materializer.MaterializationError,
        match="differs across scheduler and inventory",
    ):
        materializer._m8_source_repair_authority(population, changed_config)


def test_scheduler_keeps_ducklake_non_authoritative() -> None:
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    assert config["max_lanes"] == 1
    assert config["database_program"]["authority_mode"] == "quack"
    assert config["database_program"]["failover_policy"] == "fail_closed"
    ducklake = config["ducklake_history_projection"]
    assert ducklake["load_local_only"] is True
    assert ducklake["install_or_network_forbidden"] is True
    assert ducklake["authority"] is False
    assert ducklake["scheduling_prerequisite"] is False
    assert ducklake["completion_prerequisite"] is False


def test_m8_nofollow_receipts_and_ducklake_retry_are_idempotent(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m8_advisory_retry_test",
    )
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "receipt.json"
    link.symlink_to(target.name)
    with pytest.raises(
        materializer.MigrationRequired,
        match="no-follow regular file",
    ):
        materializer._load_nofollow_json(
            link,
            root=tmp_path,
            noun="test receipt",
        )

    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    config["ducklake_history_projection"].update(
        {
            "catalog_path": "ducklake/history.ducklake",
            "data_path": "ducklake/data",
            "receipt_path": "ducklake-history-receipt.json",
        }
    )
    record = {
        "program_definition_cid": "sha256:" + "1" * 64,
        "projection_cid": "baguqeera" + "a" * 52,
    }
    first = materializer._ducklake_projection(tmp_path, config, record)
    second = materializer._ducklake_projection(tmp_path, config, record)
    assert second == first
    assert first["authority"] is False
    assert first["scheduling_prerequisite"] is False
    assert first["completion_prerequisite"] is False
    unhashed = dict(first)
    claimed = unhashed.pop("receipt_cid")
    assert claimed == materializer._identity(unhashed)

    receipt_path = tmp_path / "ducklake-history-receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MigrationRequired,
        match="DuckLake history receipt differs",
    ):
        materializer._ducklake_projection(tmp_path, config, record)

    if first["status"] == "available":
        import duckdb

        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET autoinstall_known_extensions = false")
            connection.execute("SET autoload_known_extensions = false")
            connection.execute("LOAD ducklake")
            catalog = tmp_path / "ducklake/history.ducklake"
            data = tmp_path / "ducklake/data"

            def literal(value: Path) -> str:
                return "'" + str(value).replace("'", "''") + "'"

            connection.execute(
                "ATTACH "
                + literal(Path("ducklake:" + str(catalog)))
                + " AS sawm_history_check (DATA_PATH "
                + literal(data)
                + ")"
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM sawm_history_check.control_history "
                "WHERE program_definition_cid = ? AND projection_cid = ?",
                [record["program_definition_cid"], record["projection_cid"]],
            ).fetchone()[0]
            assert int(count) == 1
        finally:
            connection.close()


def test_m6_migration_preserves_m1_through_m5_and_preprovider_evidence() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_chain_test",
    )
    population = materializer.build_population(REPO_ROOT)
    migration = population["migration_inventory"]
    history = migration["migration_history"]

    assert migration["migration_revision"] == "SAWM-R2-M6"
    assert migration["migration_kind"] == (
        "bounded_validation_runtime_and_operational_board_command_recovery"
    )
    assert migration["supersession_reason"] == (
        "source_authority_revision_and_settled_preprovider_task_requeue"
    )
    assert migration["prior_plan_revision"] == 6
    assert migration["target_plan_revision"] == 7
    assert migration["prior_event_watermark"] == 121
    assert len(history) == 5
    m1 = history[0]
    m2 = history[1]
    m3 = history[2]
    m4 = history[3]
    m5 = history[4]
    assert m1["migration_revision"] == "SAWM-R2-M1"
    assert m2["migration_revision"] == "SAWM-R2-M2"
    assert m3["migration_revision"] == "SAWM-R2-M3"
    assert m4["migration_revision"] == "SAWM-R2-M4"
    assert m5["migration_revision"] == "SAWM-R2-M5"
    assert all(entry["schema"] == "sawm/source-migration-history-entry@1" for entry in history[:3])
    assert m4["schema"] == "sawm/source-migration-history-entry@2"
    assert m5["schema"] == "sawm/source-migration-history-entry@3"
    assert materializer._store_sha256(REPO_ROOT / m1["prior_store_id"]) == m1[
        "prior_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m1["target_store_id"]) == m1[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m2["target_store_id"]) == m2[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m3["target_store_id"]) == m3[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m4["target_store_id"]) == m4[
        "target_control_store_sha256"
    ]
    assert materializer._store_sha256(REPO_ROOT / m5["target_store_id"]) == m5[
        "target_control_store_sha256"
    ]
    assert m5["target_control_store_sha256"] == migration[
        "prior_control_store_sha256"
    ]
    assert m5["target_event_prefix_sha256"] == migration[
        "prior_event_prefix_sha256"
    ]
    assert m5["prior_event_watermark"] == 116
    assert m5["migration_event_watermark"] == 118
    assert m5["materialization_event_watermark"] == 119
    assert m5["target_event_watermark"] == 121
    assert m5["materialization_projection_cid"] == migration[
        "prior_materialization_projection_cid"
    ]
    assert m5["projection_cid"] == migration["prior_projection_cid"]
    assert m5["migration_receipt_cid"] == migration[
        "prior_materialization_receipt_cid"
    ]
    for entry in history:
        materializer._verify_receipt_anchor(
            REPO_ROOT / entry["migration_receipt_path"],
            entry["migration_receipt_cid"],
        )
    failure = migration["preworker_launch_failure"]
    assert set(failure) == {
        "accepted_control_plane_admitted",
        "command",
        "configuration_root",
        "control_plane_admission_cid",
        "control_plane_archive_sha256",
        "control_plane_capsule_id",
        "coordinator_log_path",
        "coordinator_log_sha256",
        "coordinator_pid",
        "coordinator_pid_projection_after_failure",
        "credential_handoff_retired",
        "error_payload",
        "error_payload_cid",
        "exit_code",
        "failure_time_authority",
        "implementation_provider_invoked",
        "outer_launch_command",
        "outer_launch_exit_code",
        "owner_generation",
        "owner_process_birth_id",
        "owner_server_id",
        "phase",
        "provider_capability_probed",
        "schema",
        "source_head",
        "source_tree",
        "store_id",
        "task_claimed",
        "task_state_changed",
        "worker_started",
    }
    assert failure["schema"] == "sawm/pre-worker-launch-failure@2"
    assert failure["phase"] == "detached_coordinator_provider_entry_module_preflight"
    assert failure["outer_launch_exit_code"] == 0
    assert failure["exit_code"] == 2
    assert failure["accepted_control_plane_admitted"] is True
    assert failure["provider_capability_probed"] is True
    assert failure["coordinator_pid_projection_after_failure"] == "absent"
    assert failure["worker_started"] is False
    assert failure["task_claimed"] is False
    assert failure["task_state_changed"] is False
    assert failure["implementation_provider_invoked"] is False
    assert failure["credential_handoff_retired"] is True
    assert failure["failure_time_authority"] == "unavailable"
    preprovider = migration["preprovider_task_failure"]
    assert preprovider["schema"] == "sawm/pre-provider-task-failure@2"
    assert preprovider["task_alias"] == "SAWM-001"
    assert preprovider["task_status"] == "in_progress"
    assert preprovider["canonical_task_status"] == "blocked"
    assert preprovider["canonical_task_revision"] == 5
    assert preprovider["canonical_event_watermark"] == 121
    assert preprovider["settlement_closed"] is True
    assert preprovider["canonical_projection_cid"] == migration[
        "prior_projection_cid"
    ]
    assert preprovider["lifecycle_started"] is True
    assert preprovider["worktree_setup_occurred"] is True
    assert preprovider["provider_dispatched"] is False
    assert preprovider["implementation_provider_invoked"] is False
    assert preprovider["effect_claim_recorded"] is False
    assert preprovider["implementation_commit_created"] is False
    assert preprovider["merge_attempted"] is False
    assert preprovider["task_completed"] is False
    assert preprovider["owner_status"] == "stopped"
    verified_failure = materializer._verify_preprovider_failure_artifacts(
        REPO_ROOT, migration
    )
    assert verified_failure["provider_invocation_count"] == 0
    assert verified_failure["effect_claim_count"] == 0


def test_m6_operational_revisions_and_requeue_are_fail_closed() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m6_recovery_paths_test",
    )
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependencies_m6_recovery_paths_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    recovery_controls = {
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "test/api/test_agent_supervisor_database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
    }

    assert recovery_controls.issubset(validator.CONTROL_RELATIVE_PATHS)
    assert recovery_controls.issubset(dependency_validator.CONTROL_PATHS)
    assert tuple(config["protected_paths"]) == validator.CONTROL_RELATIVE_PATHS
    assert recovery_controls.issubset(
        config["configured_board_live_capsule"]["control_paths"]
    )
    assert recovery_controls.issubset(migration["bounded_control_plane_repair_paths"])
    assert validator._m6_migration_errors(config, seal, migration) == []
    assert dependency_validator._m6_source_migration_errors(
        config, seal, migration
    ) == []

    collapsed_m5_watermarks = copy.deepcopy(migration)
    collapsed_m5_watermarks["migration_history"][4][
        "materialization_event_watermark"
    ] = 121
    assert validator._m6_migration_errors(
        config, seal, collapsed_m5_watermarks
    )
    assert dependency_validator._m6_source_migration_errors(
        config, seal, collapsed_m5_watermarks
    )

    definition_rewrite = copy.deepcopy(config)
    definition_rewrite["prior_materialization"]["operational_validation_requeue"][
        "accepted_definition_changes"
    ] = 1
    assert validator._m6_migration_errors(definition_rewrite, seal, migration)
    assert dependency_validator._m6_source_migration_errors(
        definition_rewrite, seal, migration
    )

    drifted_seal_binding = copy.deepcopy(seal)
    drifted_seal_binding["source_migration"]["program_definition_cid"] = (
        "sha256:" + "0" * 64
    )
    assert validator._m6_migration_errors(config, drifted_seal_binding, migration)
    assert dependency_validator._m6_source_migration_errors(
        config, drifted_seal_binding, migration
    )


def test_live_owner_identity_requires_exact_canonical_replica_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_identity_test",
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        _schema_fingerprint_digest,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    raw_schema_fingerprint = (
        "baguqeerayrmgjy3yusfmd7c3zcojf6l2wg7l46dnyvlm4homwfipaf5r2nha"
    )
    identity = {
        "server_id": "server:sawm-test",
        "store_id": "store:sawm-test",
        "database_uuid": "database:sawm-test",
        "process_birth_id": "birth:sawm-test",
        "listen_uri": "quack:127.0.0.1:45247",
        "extension_fingerprint": "sha256:" + ("ab" * 32),
        "schema_revision": 1,
        "schema_fingerprint": _schema_fingerprint_digest(raw_schema_fingerprint),
        "generation": 2,
        "fence_epoch": 2,
        "revision": 0,
        "credential_generation": 2,
        "secret_handle": "env://SAWM_QUACK_TOKEN",
    }

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

    class Connection:
        state_present = True

        def execute(
            self, sql: str, parameters: list[object] | None = None
        ) -> Result:
            del parameters
            if "FROM state_servers" in sql:
                rows = [
                    (
                        identity["server_id"], identity["store_id"],
                        identity["database_uuid"], identity["process_birth_id"],
                        identity["listen_uri"], identity["extension_fingerprint"],
                        1, 2, "ready", 1,
                    )
                ] if self.state_present else []
                return Result(rows)
            if "FROM store_generations" in sql:
                return Result([(2, 1, 2, 0, identity["database_uuid"], identity["process_birth_id"])])
            if "FROM credentials" in sql:
                return Result([(
                    "cred:server:sawm-test:2", identity["secret_handle"],
                    2, "quack-auth", None, 0,
                )])
            if "FROM control_plane_metadata" in sql:
                return Result([
                    ("database_uuid", identity["database_uuid"]),
                    ("schema_fingerprint", raw_schema_fingerprint),
                    ("schema_version", "1"),
                ])
            raise AssertionError(sql)

        def close(self) -> None:
            return None

    connection = Connection()
    monkeypatch.setattr(
        duckdb_state,
        "open_quack_transport_connection",
        lambda _uri, *, token: connection,
    )
    observed = operator._remote_owner_identity(
        identity["listen_uri"], "opaque-test-token", identity
    )
    assert observed["canonical_rows_verified"] is True
    connection.state_present = False
    with pytest.raises(operator.OperatorError, match="missing or ambiguous"):
        operator._remote_owner_identity(
            identity["listen_uri"], "opaque-test-token", identity
        )


def test_board_gate_binds_dependency_validator_native_authorization_and_extensions() -> None:
    validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_dependency_binding_test",
    )
    config = json.loads(
        (REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json")
        .read_text(encoding="utf-8")
    )
    seal = json.loads(
        (REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json")
        .read_text(encoding="utf-8")
    )

    assert validator._configured_board_dependency_errors(REPO_ROOT, config, seal) == []

    wrong_validator = copy.deepcopy(config)
    wrong_validator["dependency_validator_path"] = "scripts/other_validator.py"
    assert any(
        "dependency_validator_path" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, wrong_validator, seal
        )
    )

    wrong_httpfs = copy.deepcopy(config)
    wrong_httpfs["quack_owner"]["pinned_httpfs_extension"]["version"] = "stale"
    assert any(
        "scheduler httpfs pin differs" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, wrong_httpfs, seal
        )
    )

    wrong_projection = copy.deepcopy(seal)
    wrong_projection["configured_board_quack_projection"]["pin"][
        "projection_id"
    ] = "sha256:" + ("11" * 32)
    assert any(
        "Quack projection is invalid" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, config, wrong_projection
        )
    )

    for field, value in (
        ("engine_version", "v9.9.9"),
        ("platform", "linux_amd64"),
    ):
        wrong_native_binding = copy.deepcopy(seal)
        projection_pin = wrong_native_binding[
            "configured_board_quack_projection"
        ]["pin"]
        projection_pin[field] = value
        _reidentify_extension_projection(projection_pin)
        assert any(
            "extension engine/platform differs from native DuckDB/toolchain"
            in error
            for error in validator._configured_board_dependency_errors(
                REPO_ROOT,
                config,
                wrong_native_binding,
            )
        )

    wrong_httpfs_root = copy.deepcopy(seal)
    wrong_httpfs_config = copy.deepcopy(config)
    httpfs_pin = wrong_httpfs_root["httpfs_extension_pin"]
    wrong_parent = Path(httpfs_pin["path"]).parent.parent / "linux_amd64"
    httpfs_pin["path"] = str(wrong_parent / "httpfs.duckdb_extension")
    httpfs_pin["info_path"] = str(
        wrong_parent / "httpfs.duckdb_extension.info"
    )
    wrong_httpfs_config["quack_owner"]["pinned_httpfs_extension"] = copy.deepcopy(
        httpfs_pin
    )
    assert any(
        "httpfs extension path engine/platform differs from native DuckDB/toolchain"
        in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT,
            wrong_httpfs_config,
            wrong_httpfs_root,
        )
    )

    wrong_authorization = copy.deepcopy(seal)
    wrong_authorization["configured_board_native_dependency"]["acceptance"][
        "authorization_id"
    ] = "sha256:" + ("00" * 32)
    assert any(
        "authorization does not bind" in error
        for error in validator._configured_board_dependency_errors(
            REPO_ROOT, config, wrong_authorization
        )
    )


def _synthetic_extension_pin(
    root: Path,
    name: str,
    version: str,
    payload: bytes,
) -> dict[str, object]:
    extension_root = root / "v1.5.5" / "linux_arm64"
    extension_root.mkdir(parents=True, exist_ok=True)
    extension = extension_root / f"{name}.duckdb_extension"
    info = extension_root / f"{name}.duckdb_extension.info"
    extension.write_bytes(payload)
    info_payload = f"metadata:{name}".encode()
    info.write_bytes(info_payload)
    pin: dict[str, object] = {
        "path": str(extension),
        "info_path": str(info),
        "version": version,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "info_sha256": hashlib.sha256(info_payload).hexdigest(),
        "info_size": len(info_payload),
        "network_install_allowed": False,
        "unsigned_extension_allowed": False,
    }
    if name == "quack":
        pin["service_external_access_limitation"] = (
            "canonical_writer_sealed; "
            "pinned_extension_preloaded_only_in_locked_read_only_loopback_replica"
        )
    return pin


def test_operator_loads_and_resolves_exact_extension_names(
    tmp_path: Path,
) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_extension_mapping_test",
    )
    httpfs = _synthetic_extension_pin(tmp_path, "httpfs", "httpfs-v1", b"httpfs")
    quack = _synthetic_extension_pin(tmp_path, "quack", "quack-v1", b"quack")
    transport = operator._SawmQuackTransport(
        {"pinned_httpfs_extension": httpfs, "pinned_extension": quack}
    )

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class Connection:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows
            self.statements: list[str] = []

        def execute(self, sql: str) -> Result:
            self.statements.append(sql)
            return Result(self.rows if "duckdb_extensions()" in sql else [])

    transport._ensure_extension_projection()
    install_paths = transport._sealed_extension_set.install_paths
    exact_rows = [
        ("httpfs", str(install_paths["httpfs"]), httpfs["version"]),
        ("quack", str(install_paths["quack"]), quack["version"]),
    ]
    exact = Connection(exact_rows)
    try:
        transport._load_reviewed_extensions(exact)
        assert exact.statements[:2] == ["LOAD httpfs", "LOAD quack"]

        swapped = Connection(
            [
                ("httpfs", exact_rows[1][1], quack["version"]),
                ("quack", exact_rows[0][1], httpfs["version"]),
            ]
        )
        with pytest.raises(operator.OperatorError, match="differs from the reviewed pin"):
            transport._load_reviewed_extensions(swapped)

        duplicate_name = Connection(
            [
                ("httpfs", exact_rows[0][1], httpfs["version"]),
                ("httpfs", exact_rows[1][1], quack["version"]),
            ]
        )
        with pytest.raises(operator.OperatorError, match="missing or ambiguous"):
            transport._load_reviewed_extensions(duplicate_name)

        wrong_version = Connection(
            [
                exact_rows[0],
                ("quack", exact_rows[1][1], "stale-version"),
            ]
        )
        with pytest.raises(operator.OperatorError, match="differs from the reviewed pin"):
            transport._load_reviewed_extensions(wrong_version)
    finally:
        transport.stop()


def test_operator_rechecks_extension_bytes_after_native_load(tmp_path: Path) -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_extension_tamper_test",
    )
    httpfs = _synthetic_extension_pin(tmp_path, "httpfs", "httpfs-v1", b"httpfs")
    quack = _synthetic_extension_pin(tmp_path, "quack", "quack-v1", b"quack")
    transport = operator._SawmQuackTransport(
        {"pinned_httpfs_extension": httpfs, "pinned_extension": quack}
    )
    transport._ensure_extension_projection()
    install_paths = transport._sealed_extension_set.install_paths

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self._rows

    class TamperingConnection:
        def execute(self, sql: str) -> Result:
            if sql == "LOAD quack":
                projected = install_paths["quack"]
                metadata = projected.with_name(f"{projected.name}.info")
                os.chmod(metadata, 0o600)
                metadata.write_bytes(b"tamper")
            rows = [
                ("httpfs", str(install_paths["httpfs"]), httpfs["version"]),
                ("quack", str(install_paths["quack"]), quack["version"]),
            ]
            return Result(rows if "duckdb_extensions()" in sql else [])

    try:
        with pytest.raises(operator.OperatorError, match="custody|bytes drifted"):
            transport._load_reviewed_extensions(TamperingConnection())
    finally:
        transport.stop()


def test_sawm_disables_board_extension_access_at_connection_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        BOARD_EXTENSION_INSTALL_POLICY_ENV,
        open_board_control_plane,
    )

    monkeypatch.setenv(BOARD_EXTENSION_INSTALL_POLICY_ENV, "disabled")
    plane = open_board_control_plane(
        tmp_path,
        root=tmp_path / "control",
    )
    try:
        settings = plane._conn().execute(
            "SELECT current_setting('autoinstall_known_extensions'), "
            "current_setting('autoload_known_extensions'), "
            "current_setting('enable_external_access'), "
            "current_setting('allow_unsigned_extensions'), "
            "current_setting('lock_configuration')"
        ).fetchone()
        assert tuple(settings[index] for index in range(len(settings))) == (
            False,
            False,
            False,
            False,
            True,
        )
        with pytest.raises(Exception, match="configuration has been lock"):
            plane._conn().execute("SET autoinstall_known_extensions=true")
        assert plane.quack_loaded is False
        assert plane.ducklake_loaded is False
    finally:
        plane.close()


def test_load_only_ducklake_remains_root_confined_after_policy_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        BOARD_EXTENSION_INSTALL_POLICY_ENV,
        open_board_control_plane,
    )

    root = tmp_path / "control"
    monkeypatch.setenv(BOARD_EXTENSION_INSTALL_POLICY_ENV, "load_only")
    plane = open_board_control_plane(tmp_path, root=root)
    try:
        if not plane.ducklake_loaded:
            pytest.skip("the current interpreter has no admitted DuckLake extension")
        assert plane.ducklake_attached is True
        assert plane.backend == (
            "ducklake+quack" if plane.quack_loaded else "hermetic-duckdb"
        )
        plane._project_relations_to_lake()
        settings = plane._conn().execute(
            "SELECT current_setting('enable_external_access'), "
            "current_setting('lock_configuration'), "
            "current_setting('allowed_directories')"
        ).fetchone()
        observed = tuple(settings[index] for index in range(len(settings)))
        assert observed[0:2] == (False, True)
        assert str(root.resolve()) + os.sep in observed[2]
        assert all(
            Path(item).resolve(strict=False).is_relative_to(root.resolve())
            for item in observed[2]
        )
        with pytest.raises(Exception, match="disabled|not allowed"):
            plane._conn().execute("SELECT * FROM read_text('/etc/hosts')")
    finally:
        plane.close()


def test_board_control_plane_closes_connection_when_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import board_control_plane

    class Connection:
        closed = False

        def close(self) -> None:
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(
        board_control_plane,
        "_open_extension_capable_connection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(
        board_control_plane,
        "_initialize_open_board_control_plane",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("lock failed")),
    )

    with pytest.raises(RuntimeError, match="lock failed"):
        board_control_plane.open_board_control_plane(
            tmp_path,
            root=tmp_path / "control",
        )
    assert connection.closed is True


def test_operator_stop_closes_shared_extension_custody_on_quack_stop_failure() -> None:
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_stop_cleanup_test",
    )
    transport = operator._SawmQuackTransport({})

    class Replica:
        closed = False

        def execute(self, _sql: str, _parameters: object = None) -> None:
            raise RuntimeError("quack stop failed")

        def close(self) -> None:
            self.closed = True

    class Seal:
        close_count = 0

        def close(self) -> None:
            self.close_count += 1

    replica = Replica()
    seal = Seal()
    transport._serve_uri = "quack:127.0.0.1:45123"
    transport._replica_connection = replica
    transport._sealed_extension_set = seal

    with pytest.raises(RuntimeError, match="quack stop failed"):
        transport.stop()
    assert replica.closed is True
    assert seal.close_count == 1
    assert transport._sealed_extension_set is None


def test_m10_controls_and_live_projection_comparator_fail_closed() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m10_projection_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m10_projection_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m10_projection_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m10_projection_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "live_projection_successor_materialization"
    cid_key = "live_projection_successor_materialization_cid"
    expected_cid = (
        "sha256:f27878def0ee9b406d0dbaac669728278e4f375cb267ee7f76330faaf9b14f10"
    )
    authority = config[key]
    assert authority == migration[key]
    assert "sha256:" + hashlib.sha256(
        json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest() == expected_cid
    assert seal[cid_key] == expected_cid
    assert dependency_validator._m10_live_projection_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m10_migration_errors(config, seal, migration) == []
    assert board_validator._active_successor_migration_errors(
        config,
        seal,
        migration,
    ) == []
    # M9 is still checked as immutable history, but its old active paths no
    # longer override the key-present M10 generation.
    assert dependency_validator._m9_live_recovery_errors(
        config,
        seal,
        migration,
    ) == []
    historical_config = copy.deepcopy(config)
    historical_config.pop("live_provider_retry_successor_materialization")
    historical_migration = copy.deepcopy(migration)
    historical_migration.pop("live_provider_retry_successor_materialization")
    historical_seal = copy.deepcopy(seal)
    historical_seal.pop("live_provider_retry_successor_materialization_cid")
    assert operator._active_source_repair_materialization(
        historical_config
    ) == authority

    assert authority["live_task_projection"] == {
        "schema": "sawm/live-task-projection-expectation@1",
        "task_count": 45,
        "task_revision_count": 1,
        "operator_task_alias": "SAWM-000",
        "operator_status": "completed",
        "operator_revision": 2,
        "candidate_task_alias": "SAWM-001",
        "candidate_task_cid": (
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
        "candidate_status": "retrying",
        "candidate_revision": 10,
        "candidate_completion_receipt": {
            "operation": "operator_control_plane_repair",
            "settlement_id": (
                "baguqeeransrfearh5ojrnru6ls43mn7xsx6mlhndkhokrn4k7wlhowxjnshq"
            ),
        },
        "remaining_task_status": "todo",
        "remaining_task_revision": 2,
    }
    assert authority["task_revision_changes"] == 0
    assert authority["task_status_changes"] == 0
    assert authority["coordination_semantic_changes"] == 0
    assert authority["coordination_sidecar_copied_unchanged"] is True
    assert authority["provider_strategy"]["route_changed"] is False
    repair_paths = set(authority["bounded_control_plane_repair_paths"])
    assert len(repair_paths) == 9
    assert not any("todo_daemon" in path for path in repair_paths)
    assert "test/api/test_agent_supervisor_database_portal_bridge.py" not in repair_paths

    malformed = copy.deepcopy(historical_config)
    malformed[key] = []
    with pytest.raises(
        operator.OperatorError,
        match="active M10 live-projection successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)
    with pytest.raises(
        materializer.MaterializationError,
        match="M10 live projection authority is invalid",
    ):
        materializer._m10_successor_configured(malformed)

    partial_inventory = copy.deepcopy(migration)
    partial_inventory.pop(key)
    assert any(
        "M10" in error
        for error in board_validator._active_successor_migration_errors(
            config,
            seal,
            partial_inventory,
        )
    )
    changed_config = copy.deepcopy(config)
    changed_migration = copy.deepcopy(migration)
    for control in (changed_config, changed_migration):
        control[key]["task_status_changes"] = 1
    assert dependency_validator._m10_live_projection_errors(
        changed_config,
        seal,
        changed_migration,
    )
    changed_config = copy.deepcopy(historical_config)
    changed_config["provider"]["fallback_model_id"] = "changed"
    assert dependency_validator._m10_live_projection_errors(
        changed_config,
        historical_seal,
        historical_migration,
    )

    operator_source = Path(operator.__file__).read_text(encoding="utf-8")
    materializer_source = Path(materializer.__file__).read_text(encoding="utf-8")
    assert operator_source.index(key) < operator_source.index(
        "live_recovery_successor_materialization"
    )
    assert "_verify_m9_head_task_projection" in operator_source
    assert (
        'active_source_repair["target_semantic_authority_digest"]'
        in operator_source
    )
    assert "_require_active_final_pair_marker" in operator_source
    assert f'def _m10_successor_configured' in materializer_source
    assert "_verify_m9_live_task_projection" in materializer_source
    assert 'candidate.status != "retrying"' in materializer_source
    assert "candidate.revision != 10" in materializer_source


def test_m10_projection_identity_binds_the_exact_m9_task_head() -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m10_projection_identity_test",
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )

    population = materializer.build_population(REPO_ROOT)
    material = {
        "objectives": 1,
        "goals": [
            {
                "goal_cid": str(goal["goal_cid"]),
                "status": str(goal["status"]),
                "revision": 1,
            }
            for goal in sorted(
                population["objectives"], key=lambda item: str(item["goal_cid"])
            )
        ],
        "plans": [
            {
                "plan_cid": str(population["plan_root_cid"]),
                "status": "active",
                "revision": 11,
            }
        ],
        "tasks": [
            {
                "task_cid": str(task["task_cid"]),
                "status": (
                    "completed"
                    if task["task_id"] == "SAWM-000"
                    else "retrying"
                    if task["task_id"] == "SAWM-001"
                    else "todo"
                ),
                "revision": (
                    2
                    if task["task_id"] == "SAWM-000"
                    else 10
                    if task["task_id"] == "SAWM-001"
                    else 2
                ),
            }
            for task in sorted(
                population["taskboard"], key=lambda item: str(item["task_cid"])
            )
        ],
        "dependency_count": 136,
        "event_watermark": 179,
    }
    assert content_identity(material) == (
        "baguqeerareq2bngq3hffyk5vidym2ukeleg5gehpaxhqvvdjayn7ucxplcaq"
    )


def test_m10_live_projection_comparators_accept_only_m9_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m10_live_comparator_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m10_live_comparator_test",
    )
    population = materializer.build_population(REPO_ROOT)
    rearm_receipt = materializer._m9_task_rearm_receipt()

    candidate = SimpleNamespace(
        task_cid=(
            "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
        ),
        status="retrying",
        revision=10,
        body={"completion_receipt": rearm_receipt},
    )
    candidate_source = SimpleNamespace(get_task=lambda _task: candidate)
    monkeypatch.setattr(
        materializer,
        "_verify_m6_task_projection",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    assert materializer._verify_m9_live_task_projection(
        candidate_source,
        population,
    ) == ({}, {}, {})
    candidate.status = "todo"
    with pytest.raises(
        materializer.MigrationRequired,
        match="frozen M9 task rearm projection differs",
    ):
        materializer._verify_m9_live_task_projection(
            candidate_source,
            population,
        )
    candidate.status = "retrying"

    tasks: dict[str, SimpleNamespace] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        body: dict[str, object] = {}
        if alias == "SAWM-001":
            body["completion_receipt"] = rearm_receipt
        tasks[str(expected["task_cid"])] = SimpleNamespace(
            status=(
                "completed"
                if alias == "SAWM-000"
                else "retrying"
                if alias == "SAWM-001"
                else "todo"
            ),
            revision=(
                2
                if alias == "SAWM-000"
                else 10
                if alias == "SAWM-001"
                else 2
            ),
            body=body,
        )

    task_revision_rows = [
        (
            candidate.task_cid,
            10,
            "retrying",
            json.dumps({"completion_receipt": rearm_receipt}),
        )
    ]

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

        def fetchone(self) -> tuple[object, ...]:
            return self.rows[0]

    class Connection:
        def execute(self, statement: str) -> Result:
            if "FROM task_revisions" in statement:
                return Result(task_revision_rows)
            if "FROM completion_receipts" in statement:
                return Result([(1,)])
            raise AssertionError(f"unexpected comparator query: {statement}")

    class ConnectionContext:
        def __enter__(self) -> Connection:
            return Connection()

        def __exit__(self, *_args: object) -> None:
            return None

    source = SimpleNamespace(
        get_task=lambda task_cid: tasks.get(task_cid),
        intent=SimpleNamespace(
            _connection=lambda *, write=False: ConnectionContext()
        ),
    )

    def historical_projection_mismatch(*_args: object, **_kwargs: object) -> None:
        raise materializer.MigrationRequired(
            "frozen M6 task status/revision projection differs"
        )

    monkeypatch.setattr(
        materializer,
        "_verify_m6_task_projection",
        historical_projection_mismatch,
    )
    statuses, revisions, _receipts = operator._verify_m9_head_task_projection(
        source,
        population,
        materializer,
    )
    assert statuses["SAWM-001"] == "retrying"
    assert revisions["SAWM-001"] == 10
    tasks[candidate.task_cid].revision = 7
    with pytest.raises(
        materializer.MigrationRequired,
        match="M9-head task status/revision differs: SAWM-001",
    ):
        operator._verify_m9_head_task_projection(
            source,
            population,
            materializer,
        )


def _build_m9_rehearsal_pair(
    materializer: ModuleType,
    *,
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: dict[str, object],
    config: dict[str, object],
    validation_digest: str,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    shutil.copyfile(prior_control, control)
    shutil.copyfile(prior_coordination, coordination)
    source = DatabaseTaskSource(
        control,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        owner_id="sawm-r2-runtime-recovery-migrator",
    )
    try:
        operator = source.get_task("SAWM-000")
        assert operator is not None
        body = materializer._m9_migration_body(
            population,
            config,
            validation_digest,
        )
        digest = materializer._identity(body)
        source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=9,
            body={
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": materializer._M9_MIGRATION_REVISION,
                "source_migration_digest": digest,
                "supersession_mode": materializer._M9_SUPERSESSION_MODE,
            },
            delta=materializer._m9_migration_plan_delta(population, config),
        )
        source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_runtime_recovery",
            digest=digest,
            body=body,
        )
        task_cas = source.compare_and_set_status(
            materializer._M8_LIVE_FAILURE_RECEIPT["task_cid"],
            9,
            "retrying",
            materializer._m9_task_rearm_receipt(),
        )
    finally:
        source.close()
    coordinator = DatabaseCoordinator(coordination).open()
    try:
        rearm = coordinator.rearm_failed_task(
            failure_receipt=materializer._M8_LIVE_FAILURE_RECEIPT,
            control_task_observation=task_cas.to_dict(),
            now_ms=materializer._M9_COORDINATION_REARM_OBSERVED_AT_MS,
        )
    finally:
        coordinator.close()
    assert rearm["ready"] is True
    assert rearm["replayed"] is False


def test_m9_exact_control_and_coordination_rehearsal(tmp_path: Path) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m9_exact_rehearsal_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m9_live_recovery_authority(population, config)
    assert materializer._identity(authority) == (
        "sha256:e7834d12a6150a7d4dd1f90dcb5369d73a6d8620a38bf6baa0cbbb3da17bebb9"
    )
    frozen = materializer._verify_frozen_m8_live_authority(
        REPO_ROOT,
        authority,
    )
    assert frozen["append_surface_digest"] == authority[
        "prior_append_surface_digest"
    ]
    malformed = copy.deepcopy(config)
    malformed["live_recovery_successor_materialization"] = []
    with pytest.raises(
        materializer.MaterializationError,
        match="M9 live recovery authority is invalid",
    ):
        materializer._m9_successor_configured(malformed)

    prior_control = REPO_ROOT / authority["prior_store_id"]
    prior_coordination = REPO_ROOT / authority["prior_coordination_store_id"]
    control = tmp_path / "control.duckdb"
    coordination = tmp_path / "control.coordination.duckdb"
    validation_digest = "sha256:m9-focused-rehearsal"
    before_prior = (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )

    symlink_root = tmp_path / "symlink-predecessor"
    expected_parent = (
        symlink_root
        / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m8"
    )
    expected_parent.mkdir(parents=True)
    relocated = symlink_root / "relocated-control.duckdb"
    shutil.copyfile(prior_control, relocated)
    (expected_parent / "control.duckdb").symlink_to(relocated)
    with pytest.raises(
        materializer.MigrationRequired,
        match="no-follow regular file",
    ):
        materializer._verify_frozen_m8_live_authority(
            symlink_root,
            authority,
        )
    with pytest.raises(
        materializer.MigrationRequired,
        match="no-follow regular file",
    ):
        materializer._assert_m9_prior_publication_anchor(
            symlink_root,
            authority,
        )

    _build_m9_rehearsal_pair(
        materializer,
        control=control,
        coordination=coordination,
        prior_control=prior_control,
        prior_coordination=prior_coordination,
        population=population,
        config=config,
        validation_digest=validation_digest,
    )
    before_verify = (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    report = materializer._verify_m9_store_pair_copy(
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert report["projection_cid"] == materializer._M9_EXPECTED_PROJECTION_CID
    assert report["event_watermark"] == 177
    assert report["coordination_projection_digest"] == (
        materializer._M9_COORDINATION_PROJECTION_DIGEST
    )
    assert report["coordination_event_count"] == 36
    assert report["task_revision_changes"] == 1
    assert before_verify == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    assert before_prior == (
        materializer._store_sha256(prior_control),
        materializer._store_sha256(prior_coordination),
    )

    import duckdb

    connection = duckdb.connect(str(control))
    try:
        connection.execute(
            "UPDATE goals SET title = title || '-tampered' "
            "WHERE goal_cid = (SELECT MIN(goal_cid) FROM goals)"
        )
    finally:
        connection.close()
    with pytest.raises(
        materializer.MigrationRequired,
        match="non-authorized control table",
    ):
        materializer._verify_m9_store_pair_copy(
            control,
            coordination,
            prior_control,
            prior_coordination,
            population,
            config,
            validation_digest,
        )


def test_m9_receipt_is_last_single_link_pair_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m9_receipt_recovery_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m9_live_recovery_authority(population, config)
    prior_control = tmp_path / "prior-control.duckdb"
    prior_coordination = tmp_path / "prior-control.coordination.duckdb"
    shutil.copyfile(REPO_ROOT / authority["prior_store_id"], prior_control)
    shutil.copyfile(
        REPO_ROOT / authority["prior_coordination_store_id"],
        prior_coordination,
    )
    target = tmp_path / "target"
    target.mkdir()
    control = target / "control.duckdb"
    coordination = target / "control.coordination.duckdb"
    validation_digest = "sha256:m9-receipt-recovery"
    _build_m9_rehearsal_pair(
        materializer,
        control=control,
        coordination=coordination,
        prior_control=prior_control,
        prior_coordination=prior_coordination,
        population=population,
        config=config,
        validation_digest=validation_digest,
    )
    verified = materializer._verify_m9_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_committed_clean_source",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m9_source_delta",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m9_prior_publication_anchor",
        lambda *_args, **_kwargs: (prior_control, prior_coordination),
    )
    receipt = materializer._ensure_m9_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    receipt_path = target / "migration-receipt.json"
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt_path.stat().st_nlink == 1
    assert control.stat().st_nlink == coordination.stat().st_nlink == 1
    assert not list(target.glob(".migration-receipt.json.*.tmp"))
    assert materializer._ensure_m9_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt

    receipt_path.unlink()
    pending = target / ".migration-receipt.json.999999.tmp"
    pending.write_bytes(materializer._canonical(receipt) + b"\n")
    os.link(pending, receipt_path)
    assert receipt_path.stat().st_nlink == 2
    recovered = materializer._ensure_m9_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    assert recovered == receipt
    assert receipt_path.stat().st_nlink == 1
    assert not pending.exists()

    alias = target / "mutable-control-alias.duckdb"
    os.link(control, alias)
    with pytest.raises(
        materializer.MigrationRequired,
        match="exactly 1 link",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    alias.unlink()
    execution = target / "control.execution.duckdb"
    execution.write_bytes(b"not-copied")
    with pytest.raises(
        materializer.MigrationRequired,
        match="execution sidecar exists",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    execution.unlink()

    # Reproduce a sidecar appearing after full pair verification but before
    # the receipt lock is acquired.  The final commit-input gate must refuse
    # to publish the marker.
    receipt_path.unlink()
    original_pair_verifier = materializer._verify_m9_store_pair

    def inject_execution_sidecar(*args: object, **kwargs: object) -> object:
        result = original_pair_verifier(*args, **kwargs)
        execution.write_bytes(b"appeared-during-verification")
        return result

    monkeypatch.setattr(
        materializer,
        "_verify_m9_store_pair",
        inject_execution_sidecar,
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="execution sidecar exists at receipt commit",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    assert not receipt_path.exists()
    execution.unlink()
    monkeypatch.setattr(
        materializer,
        "_verify_m9_store_pair",
        original_pair_verifier,
    )

    # If the store changes after the last pre-link recheck, the post-link
    # check removes only the marker created by this invocation and retains the
    # pending receipt bytes for crash/audit recovery.
    original_commit_gate = materializer._assert_m9_receipt_commit_inputs
    commit_gate_calls = 0

    def mutate_after_prelink_gate(*args: object, **kwargs: object) -> None:
        nonlocal commit_gate_calls
        original_commit_gate(*args, **kwargs)
        commit_gate_calls += 1
        if commit_gate_calls == 2:
            with control.open("ab") as handle:
                handle.write(b"post-gate-mutation")
                handle.flush()
                os.fsync(handle.fileno())

    monkeypatch.setattr(
        materializer,
        "_assert_m9_receipt_commit_inputs",
        mutate_after_prelink_gate,
    )
    with pytest.raises(
        materializer.MigrationRequired,
        match="store pair changed at receipt commit",
    ):
        materializer._ensure_m9_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    assert not receipt_path.exists()
    assert len(list(target.glob(".migration-receipt.json.*.tmp"))) == 1


def test_m11_controls_and_provider_retry_authority_fail_closed() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m11_provider_retry_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m11_provider_retry_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m11_provider_retry_test",
    )
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_provider_retry_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    migration = json.loads(
        (
            REPO_ROOT
            / "docs/architecture/semantic_addressed_world_model_inventory/"
            "prior_materialization_migration.json"
        ).read_text(encoding="utf-8")
    )
    seal = json.loads(
        (
            REPO_ROOT
            / "config/semantic_addressed_world_model_dependencies.seal.json"
        ).read_text(encoding="utf-8")
    )
    key = "live_provider_retry_successor_materialization"
    cid_key = "live_provider_retry_successor_materialization_cid"
    expected_cid = (
        "sha256:7f8404735adae0fb7ed890cda97dc674b78ae915efa193207e98abd95eb5193e"
    )
    failure_cid = (
        "sha256:a1fae6ef892b02a19b1155a0a5ed7c0eefa8f8255f3949e25d5a8b8b9d721676"
    )
    authority = config[key]

    assert authority == migration[key]
    assert "sha256:" + hashlib.sha256(
        json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest() == expected_cid
    assert seal[cid_key] == expected_cid
    assert materializer._expected_m11_live_provider_retry_authority() == authority
    assert materializer._identity(authority) == expected_cid
    population = materializer.build_population(REPO_ROOT)
    assert materializer._m11_live_provider_retry_authority(
        population,
        config,
    ) == authority
    assert materializer._m11_successor_configured(config) is True
    assert dependency_validator._m11_provider_retry_errors(
        config,
        seal,
        migration,
    ) == []
    assert board_validator._m11_migration_errors(config, seal, migration) == []
    assert board_validator._active_successor_migration_errors(
        config,
        seal,
        migration,
    ) == []
    assert dependency_validator._m10_live_projection_errors(
        config,
        seal,
        migration,
    ) == []
    assert operator._active_source_repair_materialization(config) == authority

    assert authority["schema"] == "sawm/provider-launch-repair-authorization@1"
    assert authority["migration_revision"] == "SAWM-R2-M11"
    assert authority["prior_event_watermark"] == 181
    assert authority["target_plan_revision"] == 12
    assert authority["target_event_watermark"] == 184
    assert authority["target_generation"] == 13
    assert authority["target_projection_cid"] == (
        "baguqeeratdygaminyfax543hh5bik3kj5admm5filgtu37a3jfnyc6snwnxa"
    )
    assert authority["prior_coordination_wal_sha256"] == (
        "938dbc7028ec438889019c352b5a2529cc8c0f0fd18a11f0eb157a6772fc5321"
    )
    assert authority["prior_coordination_wal_size"] == 24_746
    assert authority["prior_coordination_event_count"] == 55
    assert authority["target_coordination_event_count"] == 56
    assert authority["coordination_wal_replayed_on_copy"] is True
    assert authority["prior_coordination_store_mutated"] is False
    assert authority["task_rearm"]["from_status"] == "blocked"
    assert authority["task_rearm"]["from_revision"] == 12
    assert authority["task_rearm"]["to_status"] == "retrying"
    assert authority["task_rearm"]["to_revision"] == 13
    assert authority["task_rearm"]["settlement_id"] == (
        "baguqeera7cwinhpjgl2etuitwiuhs4ts6lix5txyb2pjtsbfz2npfyryznma"
    )
    failure = authority["live_implementation_failure"]
    assert authority["live_implementation_failure_cid"] == failure_cid
    assert materializer._identity(failure) == failure_cid
    assert failure["provider_execution_observed"] is True
    assert failure["provider_execution_accounting_mismatch"] is True
    assert failure["settlement_provider_invocation_count"] == 0
    assert authority["implementation_provider_invocations_observed"] == 1
    assert authority["settlement_provider_invocation_count"] == 0
    for field in (
        "effect_claim_changes",
        "implementation_commit_changes",
        "merge_attempt_changes",
        "accepted_definition_changes",
        "accepted_completion_changes",
    ):
        assert authority[field] == 0
    assert authority["execution_sidecar_copied"] is False
    assert authority["worker_self_approval"] is False
    assert authority["provider_strategy"] == {
        "capability_probe_required": True,
        "prior_route": "grok-4.6_then_codex_on_independently_verified_quota",
        "provider_result_is_completion_authority": False,
        "route_changed": False,
        "target_route": "grok-4.6_then_codex_on_independently_verified_quota",
    }

    repair_paths = set(authority["bounded_control_plane_repair_paths"])
    assert len(repair_paths) == 11
    assert authority["runner_repair"]["runner_path"] in repair_paths
    assert authority["runner_repair"]["focused_test_path"] in repair_paths
    assert repair_paths.issubset(dependency_validator.CONTROL_PATHS)
    assert repair_paths.issubset(board_validator.CONTROL_RELATIVE_PATHS)
    assert repair_paths.issubset(
        set(population["source_binding"]["control_sha256"])
    )
    assert repair_paths.issubset(set(config["protected_paths"]))
    assert repair_paths.issubset(
        set(config["configured_board_live_capsule"]["control_paths"])
    )

    malformed = copy.deepcopy(config)
    malformed[key] = []
    with pytest.raises(
        operator.OperatorError,
        match="active M11 provider-retry successor authority is invalid",
    ):
        operator._active_source_repair_materialization(malformed)
    with pytest.raises(
        materializer.MaterializationError,
        match="M11 live provider retry authority is invalid",
    ):
        materializer._m11_successor_configured(malformed)
    assert any(
        "M11" in error
        for error in board_validator._active_successor_migration_errors(
            malformed,
            seal,
            migration,
        )
    )

    partial_inventory = copy.deepcopy(migration)
    partial_inventory.pop(key)
    assert any(
        "M11" in error
        for error in board_validator._active_successor_migration_errors(
            config,
            seal,
            partial_inventory,
        )
    )
    partial_seal = copy.deepcopy(seal)
    partial_seal.pop(cid_key)
    assert any(
        "M11" in error
        for error in board_validator._active_successor_migration_errors(
            config,
            partial_seal,
            migration,
        )
    )

    def rejected(mutator: object) -> None:
        changed_config = copy.deepcopy(config)
        changed_migration = copy.deepcopy(migration)
        mutator(changed_config[key])
        mutator(changed_migration[key])
        assert dependency_validator._m11_provider_retry_errors(
            changed_config,
            seal,
            changed_migration,
        )

    mutators = (
        lambda item: item.__setitem__("prior_coordination_wal_size", 24_745),
        lambda item: item.__setitem__("target_event_watermark", 183),
        lambda item: item["task_rearm"].__setitem__("to_revision", 14),
        lambda item: item["runner_repair"].__setitem__(
            "provider_execution_observed", False
        ),
        lambda item: item.__setitem__(
            "implementation_provider_invocations_observed", 0
        ),
        lambda item: item.__setitem__("execution_sidecar_copied", True),
        lambda item: item["provider_strategy"].__setitem__(
            "route_changed", True
        ),
        lambda item: item.__setitem__("worker_self_approval", True),
        lambda item: item["bounded_control_plane_repair_paths"].pop(),
    )
    for mutator in mutators:
        rejected(mutator)

    operator_source = Path(operator.__file__).read_text(encoding="utf-8")
    materializer_source = Path(materializer.__file__).read_text(encoding="utf-8")
    assert operator_source.index(key) < operator_source.index(
        "live_projection_successor_materialization"
    )
    assert "_verify_m11_head_task_projection" in operator_source
    assert "_require_m11_final_pair_marker" in operator_source
    assert materializer_source.index("def _m11_successor_configured") < (
        materializer_source.index("def _m10_target_paths")
    )
    assert "def _assert_m11_source_delta" in materializer_source


def test_m11_live_task_comparator_requires_exact_append_only_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_comparator_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m11_comparator_test",
    )
    population = materializer.build_population(REPO_ROOT)
    candidate_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    tasks: dict[str, SimpleNamespace] = {}
    for expected in population["taskboard"]:
        alias = str(expected["task_id"])
        body: dict[str, object] = {}
        if alias == "SAWM-001":
            body["completion_receipt"] = dict(operator._M11_REARM_RECEIPT)
        tasks[str(expected["task_cid"])] = SimpleNamespace(
            status=(
                "completed"
                if alias == "SAWM-000"
                else "retrying"
                if alias == "SAWM-001"
                else "todo"
            ),
            revision=(
                2
                if alias == "SAWM-000"
                else 13
                if alias == "SAWM-001"
                else 2
            ),
            body=body,
        )
    history = [
        (candidate_cid, 10, "retrying", operator._M9_REARM_RECEIPT),
        (candidate_cid, 11, "in_progress", operator._M10_CLAIM_RECEIPT),
        (candidate_cid, 12, "blocked", operator._M10_FAILURE_RECEIPT),
        (candidate_cid, 13, "retrying", operator._M11_REARM_RECEIPT),
    ]

    class Result:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def fetchall(self) -> list[tuple[object, ...]]:
            return self.rows

        def fetchone(self) -> tuple[object, ...]:
            return self.rows[0]

    class Connection:
        def execute(self, statement: str) -> Result:
            if "FROM task_revisions" in statement:
                return Result(
                    [
                        (task_cid, revision, status, json.dumps({
                            "completion_receipt": dict(receipt)
                        }))
                        for task_cid, revision, status, receipt in history
                    ]
                )
            if "FROM completion_receipts" in statement:
                return Result([(1,)])
            raise AssertionError(f"unexpected comparator query: {statement}")

    class ConnectionContext:
        def __enter__(self) -> Connection:
            return Connection()

        def __exit__(self, *_args: object) -> None:
            return None

    source = SimpleNamespace(
        get_task=lambda task_cid: tasks.get(task_cid),
        intent=SimpleNamespace(
            _connection=lambda *, write=False: ConnectionContext()
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_m6_task_projection",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    statuses, revisions, _receipts = operator._verify_m11_head_task_projection(
        source,
        population,
        materializer,
    )
    assert statuses["SAWM-001"] == "retrying"
    assert revisions["SAWM-001"] == 13

    history.pop(1)
    with pytest.raises(
        materializer.MigrationRequired,
        match="M11-head task revision/completion history differs",
    ):
        operator._verify_m11_head_task_projection(
            source,
            population,
            materializer,
        )
    history.insert(
        1,
        (candidate_cid, 11, "in_progress", operator._M10_CLAIM_RECEIPT),
    )
    tasks[candidate_cid].revision = 12
    with pytest.raises(
        materializer.MigrationRequired,
        match="M11-head task status/revision differs: SAWM-001",
    ):
        operator._verify_m11_head_task_projection(
            source,
            population,
            materializer,
        )


def test_m11_prior_base_wal_replay_is_disposable_and_tamper_closed(
    tmp_path: Path,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_wal_replay_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m11_live_provider_retry_authority(
        population,
        config,
    )
    copied_root = tmp_path / "disposable-authority"
    copied_paths: list[Path] = []
    for field in (
        "prior_store_id",
        "prior_coordination_store_id",
        "prior_coordination_wal_id",
        "prior_materialization_receipt_path",
        "prior_owner_status_path",
    ):
        relative = Path(authority[field])
        source = REPO_ROOT / relative
        target = copied_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied_paths.append(target)
    source_hashes = {
        field: materializer._store_sha256(REPO_ROOT / authority[field])
        for field in (
            "prior_store_id",
            "prior_coordination_store_id",
            "prior_coordination_wal_id",
        )
    }

    _control, _coordination, _wal, report = (
        materializer._assert_m11_prior_publication_anchor(
            copied_root,
            authority,
            population,
        )
    )
    assert report["coordination_event_count"] == 55
    assert report["coordination_projection_digest"] == (
        "sha256:25b8bd03cfe9a684a2626a6c5e6c955b295ba6073248c7a1cf2cd83d7550576d"
    )
    assert {
        field: materializer._store_sha256(REPO_ROOT / authority[field])
        for field in source_hashes
    } == source_hashes
    assert all(path.is_file() for path in copied_paths)

    copied_wal = copied_root / authority["prior_coordination_wal_id"]
    copied_wal.write_bytes(copied_wal.read_bytes() + b"tamper")
    with pytest.raises(
        materializer.MigrationRequired,
        match="control/coordination/WAL bytes differ",
    ):
        materializer._assert_m11_prior_publication_anchor(
            copied_root,
            authority,
            population,
        )


def _build_m11_rehearsal_pair(
    materializer: ModuleType,
    *,
    control: Path,
    coordination: Path,
    prior_control: Path,
    prior_coordination: Path,
    population: dict[str, object],
    config: dict[str, object],
    validation_digest: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the exact M11 append and rearm only on disposable stores."""

    import duckdb

    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinator,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources import intent_repository
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    monkeypatch.setattr(
        intent_repository,
        "_utc_iso",
        lambda _moment=None: materializer._M11_CONTROL_RECORDED_AT,
    )
    control.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prior_control, control)
    shutil.copyfile(prior_coordination, coordination)
    prior_wal = prior_coordination.with_name(prior_coordination.name + ".wal")
    target_wal = coordination.with_name(coordination.name + ".wal")
    shutil.copyfile(prior_wal, target_wal)

    source = DatabaseTaskSource(
        control,
        install_schema=False,
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        owner_id="sawm-r2-provider-retry-migrator",
    )
    try:
        operator = source.get_task("SAWM-000")
        assert operator is not None
        body = materializer._m11_migration_body(
            population,
            config,
            validation_digest,
        )
        digest = materializer._identity(body)
        source.plans.append_revision(
            plan_cid=str(population["plan_root_cid"]),
            expected_revision=11,
            body={
                "current_source_binding_cid": population["source_binding"][
                    "source_binding_cid"
                ],
                "source_migration_revision": materializer._M11_MIGRATION_REVISION,
                "source_migration_digest": digest,
                "supersession_mode": materializer._M11_SUPERSESSION_MODE,
            },
            delta=materializer._m11_migration_plan_delta(population, config),
        )
        source.record_evidence(
            task_cid=operator.task_cid,
            evidence_kind="operator_control_plane_source_migration",
            digest=digest,
            body=body,
        )
        task_cas = source.compare_and_set_status(
            materializer._M10_LIVE_FAILURE_RECEIPT["task_cid"],
            12,
            "retrying",
            materializer._m11_task_rearm_receipt(),
        )
    finally:
        source.close()

    coordinator = DatabaseCoordinator(coordination).open()
    try:
        rearm = coordinator.rearm_failed_task(
            failure_receipt=materializer._M10_LIVE_FAILURE_RECEIPT,
            control_task_observation=task_cas.to_dict(),
            now_ms=materializer._M11_COORDINATION_REARM_OBSERVED_AT_MS,
        )
    finally:
        coordinator.close()
    assert rearm["ready"] is True
    assert rearm["replayed"] is False

    # Seal the disposable target's replayed coordination state into its base;
    # the frozen predecessor base+WAL remain byte-for-byte untouched.
    checkpoint = duckdb.connect(str(coordination))
    try:
        checkpoint.execute("CHECKPOINT")
    finally:
        checkpoint.close()
    assert not os.path.lexists(target_wal)


def test_m11_pair_receipt_last_rehearsal_is_idempotent_and_tamper_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = _load(
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "sawm_materializer_m11_pair_receipt_test",
    )
    config = json.loads(
        (
            REPO_ROOT
            / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        ).read_text(encoding="utf-8")
    )
    population = materializer.build_population(REPO_ROOT)
    authority = materializer._m11_live_provider_retry_authority(
        population,
        config,
    )
    validation_digest = "sha256:m11-disposable-pair-receipt"

    # Recreate the exact predecessor publication layout under a disposable
    # root so every anchor and receipt path is exercised without a live write.
    for field in (
        "prior_store_id",
        "prior_coordination_store_id",
        "prior_coordination_wal_id",
        "prior_materialization_receipt_path",
        "prior_owner_status_path",
    ):
        source_path = REPO_ROOT / authority[field]
        copied_path = tmp_path / authority[field]
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, copied_path)
    prior_control = tmp_path / authority["prior_store_id"]
    prior_coordination = tmp_path / authority["prior_coordination_store_id"]
    control = tmp_path / authority["target_store_id"]
    coordination = tmp_path / authority["target_coordination_store_id"]
    original_paths = tuple(
        REPO_ROOT / authority[field]
        for field in (
            "prior_store_id",
            "prior_coordination_store_id",
            "prior_coordination_wal_id",
            "prior_materialization_receipt_path",
            "prior_owner_status_path",
        )
    )
    original_hashes = {
        path: materializer._store_sha256(path) for path in original_paths
    }

    _build_m11_rehearsal_pair(
        materializer,
        control=control,
        coordination=coordination,
        prior_control=prior_control,
        prior_coordination=prior_coordination,
        population=population,
        config=config,
        validation_digest=validation_digest,
        monkeypatch=monkeypatch,
    )
    target_hashes = (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    verified = materializer._verify_m11_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    )
    assert verified["projection_cid"] == materializer._M11_EXPECTED_PROJECTION_CID
    assert verified["event_watermark"] == 184
    assert verified["coordination_event_count"] == 56
    assert verified["task_revision_changes"] == 1
    assert verified["task_status_changes"] == 1
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    assert materializer._verify_m11_store_pair(
        tmp_path,
        control,
        coordination,
        prior_control,
        prior_coordination,
        population,
        config,
        validation_digest,
    ) == verified
    anchored = materializer._assert_m11_prior_publication_anchor(
        tmp_path,
        authority,
        population,
    )

    # The pair and predecessor were each checked above through their real
    # disposable replay paths. Receipt-only fault cases should exercise the
    # linearization protocol without repeating that expensive replay.
    monkeypatch.setattr(
        materializer,
        "_verify_m11_store_pair",
        lambda *_args, **_kwargs: dict(verified),
    )
    monkeypatch.setattr(
        materializer,
        "_assert_m11_prior_publication_anchor",
        lambda *_args, **_kwargs: anchored,
    )

    # Exercise the exact committed source-delta closure independently of the
    # dirty test checkout, then retain that fail-closed fake for receipt work.
    changed_paths = list(authority["bounded_control_plane_repair_paths"])

    def exact_git(_root: Path, *args: str, **_kwargs: object) -> str:
        if args[:2] == ("merge-base", "--is-ancestor"):
            return ""
        if args and args[0] == "diff":
            return "\n".join(f"M\t{path}" for path in changed_paths)
        raise AssertionError(f"unexpected M11 source-delta git call: {args}")

    monkeypatch.setattr(materializer, "_git", exact_git)
    delta_population = copy.deepcopy(population)
    delta_population["source_binding"]["head"] = "1" * 40
    materializer._assert_m11_source_delta(
        tmp_path,
        delta_population,
        authority,
    )
    changed_paths.append("unexpected.py")
    with pytest.raises(
        materializer.MaterializationError,
        match="exact repair paths",
    ):
        materializer._assert_m11_source_delta(
            tmp_path,
            delta_population,
            authority,
        )
    changed_paths.pop()
    monkeypatch.setattr(
        materializer,
        "_assert_m11_source_delta",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        materializer,
        "_assert_committed_clean_source",
        lambda *_args, **_kwargs: None,
    )

    receipt_path = control.parent / "migration-receipt.json"
    expected = materializer._expected_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    malformed_pending = control.parent / ".migration-receipt.json.bad.tmp"
    malformed_pending.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        materializer.MigrationRequired,
        match="pending M11 receipt temporary differs",
    ):
        materializer._ensure_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    assert not receipt_path.exists()
    malformed_pending.unlink()

    # An exact orphan from a crash before hardlink publication is recovered;
    # an ambiguous or unrelated hardlink is never accepted.
    pending = control.parent / ".migration-receipt.json.999999.tmp"
    pending.write_bytes(materializer._canonical(expected) + b"\n")
    receipt = materializer._ensure_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    )
    assert receipt == expected
    assert receipt["schema"] == "sawm/non-authoritative-migration-receipt@9"
    assert receipt["receipt_is_final_pair_commit_marker"] is True
    assert receipt_path.stat().st_nlink == 1
    assert not pending.exists()
    assert not list(control.parent.glob(".migration-receipt.json.*.tmp"))
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
    assert materializer._ensure_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt
    assert materializer._verify_existing_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt

    crash_alias = control.parent / ".migration-receipt.json.123.tmp"
    os.link(receipt_path, crash_alias)
    assert receipt_path.stat().st_nlink == 2
    assert materializer._ensure_m11_migration_receipt(
        tmp_path,
        control,
        coordination,
        population,
        config,
        verified,
        validation_digest,
    ) == receipt
    assert receipt_path.stat().st_nlink == 1
    assert not crash_alias.exists()

    unrelated_alias = control.parent / "unrelated-receipt-hardlink.json"
    os.link(receipt_path, unrelated_alias)
    with pytest.raises(
        materializer.MigrationRequired,
        match="ambiguous pending hardlink",
    ):
        materializer._ensure_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    unrelated_alias.unlink()

    execution_sidecar = control.with_name("control.execution.duckdb")
    execution_sidecar.write_bytes(b"forbidden")
    with pytest.raises(
        materializer.MigrationRequired,
        match="mutable sidecar appeared before final receipt",
    ):
        materializer._ensure_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    execution_sidecar.unlink()

    pristine_receipt = receipt_path.read_bytes()
    tampered = json.loads(pristine_receipt)
    tampered["worker_self_approval"] = True
    receipt_path.write_bytes(materializer._canonical(tampered) + b"\n")
    with pytest.raises(
        materializer.MigrationRequired,
        match="final pair marker differs",
    ):
        materializer._verify_existing_m11_migration_receipt(
            tmp_path,
            control,
            coordination,
            population,
            config,
            verified,
            validation_digest,
        )
    receipt_path.write_bytes(pristine_receipt)

    assert original_hashes == {
        path: materializer._store_sha256(path) for path in original_paths
    }
    assert target_hashes == (
        materializer._store_sha256(control),
        materializer._store_sha256(coordination),
    )
