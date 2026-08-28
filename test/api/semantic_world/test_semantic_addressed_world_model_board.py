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
from types import ModuleType

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


def test_m7_controls_and_live_comparator_fail_closed() -> None:
    dependency_validator = _load(
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "sawm_dependency_m7_source_repair_test",
    )
    board_validator = _load(
        "scripts/validate_semantic_addressed_world_model_board.py",
        "sawm_board_m7_source_repair_test",
    )
    operator = _load(
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "sawm_operator_m7_live_comparator_test",
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
    stale_runtime = copy.deepcopy(config)
    stale_runtime["runtime_paths"]["root"] = (
        "data/agent_supervisor/semantic_addressed_world_model/run-r2-m6"
    )
    assert dependency_validator._m7_source_repair_errors(
        stale_runtime,
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

    source = Path(operator.__file__).read_text(encoding="utf-8")
    assert "materializer._verify_m6_task_projection(live, population)" in source
    assert "materializer._semantic_authority_digest_on(connection)" in source
    assert "live_snapshot[\"event_cursor\"] != 170" in source
    assert (
        'if isinstance(config.get("source_repair_materialization"), Mapping):'
        in source
    )
    assert "checked = materializer.check_materialized(REPO_ROOT, config_path)" in source


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
