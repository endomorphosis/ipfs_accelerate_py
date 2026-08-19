"""Contract tests for the generated EAAEF supervisor board."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "docs/architecture/external_agent_autonomous_execution_fabric"


def _load_validator():
    path = ROOT / "scripts/validate_external_agent_autonomous_execution_fabric_board.py"
    spec = importlib.util.spec_from_file_location("eaaef_validator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _board() -> dict:
    return json.loads((CAMPAIGN / "task_board.json").read_text(encoding="utf-8"))


def test_generated_board_passes_fail_closed_validator() -> None:
    report = _load_validator().validate()
    assert report["valid"] is True, report["errors"]
    assert report["counts"] == {
        "goal_count": 19,
        "task_count": 104,
        "initial_population_count": 10,
        "owned_path_count": 362,
        "owned_path_overlap_count": 29,
        "overlap_merge_contract_count": 31,
        "dependency_edge_count": 245,
    }


def test_bootstrap_is_the_only_initial_ready_task() -> None:
    board = _board()
    tasks = {task["stable_task_id"]: task for task in board["tasks"]}
    ready = [
        task["stable_task_id"]
        for task in board["tasks"]
        if task["status"] == "todo"
        and task["is_schedulable"]
        and not task["dependencies"]
    ]
    assert ready == ["EAAEF-000"]
    assert tasks["EAAEF-000"]["completion_mode"] == "manual"
    for number in range(1, 6):
        assert "EAAEF-000" in tasks[f"EAAEF-{number:03d}"]["dependencies"]


def test_future_population_is_held_until_plan_r2() -> None:
    for task in _board()["tasks"]:
        if int(task["stable_task_id"].split("-")[-1]) < 10:
            continue
        assert task["status"] == "blocked"
        assert task["is_schedulable"] is False
        assert task["population_state"] == "template_only_awaiting_plan_r2"
        assert task["blocked_reason"] == "awaiting_EAAEF-009_plan_revision"
        assert task["source_semantic_state_root"] == "REBIND_REQUIRED_BY_EAAEF-009"


def test_cross_repository_execution_validation_has_explicit_cwd_and_argv() -> None:
    expected_cwd = {
        "ipfs_accelerate_py": ".",
        "ipfs_datasets_py": "ipfs_datasets_py",
        "ipfs_kit_py": "ipfs_kit_py",
        "Mcp-Plus-Plus": "ipfs_accelerate_py/mcplusplus",
    }
    for task in _board()["tasks"]:
        commands = task["execution_validation"]
        assert commands
        for command in commands:
            assert command["working_directory"] == expected_cwd[task["owning_repository"]]
            assert isinstance(command["argv"], list) and command["argv"]
            assert ";" not in command["argv"]


def test_bootstrap_owns_and_validates_new_fail_closed_contracts() -> None:
    task = next(
        item for item in _board()["tasks"] if item["stable_task_id"] == "EAAEF-000"
    )
    required_paths = {
        "ipfs_accelerate_py/llm_router.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/contract_mismatch_analyzer.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/contract_vulnerability_rules.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/mcp_contract_catalog.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/mcp_invocation_trace.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/parser_failure_triage.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/polyglot_ast_health.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/polyglot_ast_provider.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/python_mcp_surface_extractor.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/runtime_component_catalog.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/runtime_contract_evidence_compiler.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/swissknife_contract_extractor.py",
        "ipfs_accelerate_py/agent_supervisor/control/plan_execution_store.py",
        "ipfs_accelerate_py/agent_supervisor/entrypoints/inference_runtime.py",
        "ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_train.py",
        "ipfs_accelerate_py/agent_supervisor/objectives/backlog_refinery.py",
        "ipfs_accelerate_py/agent_supervisor/objectives/objective_graph.py",
        "ipfs_accelerate_py/agent_supervisor/proof/mcp_contract_proof_cache.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/eaaef_bootstrap_gateway.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/plan_r2_remote_owner.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/external_agent_state_repository.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/persistent_task_queue.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/worker_container_execution_profile.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/eaaef_bootstrap_daemon_gateway.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/eaaef_borrowed_transaction.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/eaaef_operational_schema.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/worktrees.py",
        "ipfs_accelerate_py/agent_supervisor/validation/agent_native_dependency_admission.py",
        "ipfs_accelerate_py/agent_supervisor/validation/eaaef_bootstrap_gateway_launch.py",
        "ipfs_accelerate_py/agent_supervisor/validation/eaaef_lane_gateway_admission.py",
        "ipfs_accelerate_py/agent_supervisor/validation/plan_r2_remote_owner_admission.py",
        "ipfs_accelerate_py/agent_supervisor/validation/proof_cached_test_validation.py",
        "ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py",
        "ipfs_accelerate_py/testing/proof_reuse/default_identity_services.py",
        "ipfs_accelerate_py/testing/proof_reuse/item_identity.py",
        "scripts/extract_typescript_ast.mjs",
        "test/api/test_agent_supervisor_contract_mismatch_analyzer.py",
        "test/api/test_agent_supervisor_contract_vulnerability_rules.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "test/api/test_agent_supervisor_incremental_runtime.py",
        "test/api/test_agent_supervisor_inference_runtime.py",
        "test/api/test_agent_supervisor_ipfs_datasets_logic_provider.py",
        "test/api/test_agent_supervisor_mcp_contract_catalog.py",
        "test/api/test_agent_supervisor_mcp_contract_proof_cache.py",
        "test/api/test_agent_supervisor_mcp_invocation_trace.py",
        "test/api/test_agent_supervisor_native_dependency_admission.py",
        "test/api/test_agent_supervisor_parser_failure_triage.py",
        "test/api/test_agent_supervisor_polyglot_ast_health.py",
        "test/api/test_agent_supervisor_polyglot_ast_provider.py",
        "test/api/test_agent_supervisor_prompt_v3_resolution_hardening.py",
        "test/api/test_agent_supervisor_proof_merge_gate.py",
        "test/api/test_agent_supervisor_proof_cached_test_validation.py",
        "test/api/test_agent_supervisor_python_mcp_surface_extractor.py",
        "test/api/test_agent_supervisor_router_owned_provider_decision.py",
        "test/api/test_agent_supervisor_runtime_component_catalog.py",
        "test/api/test_agent_supervisor_runtime_contract_evidence_compiler.py",
        "test/api/test_agent_supervisor_swissknife_contract_extractor.py",
        "test/api/test_agent_supervisor_todo_daemon_port.py",
        "test/api/test_agent_supervisor_validation_scheduler.py",
        "test/api/test_eaaef_bootstrap_daemon_gateway.py",
        "test/api/test_eaaef_bootstrap_gateway_launch.py",
        "test/api/test_eaaef_bootstrap_runtime_gateway.py",
        "test/api/test_eaaef_borrowed_transaction.py",
        "test/api/test_eaaef_lane_gateway_runtime.py",
        "test/api/test_eaaef_operational_schema.py",
        "test/api/test_eaaef_quack_command_fabric.py",
        "test/api/test_eaaef_supervisor_daemon_birth_wiring.py",
        "test/api/test_external_agent_state_repository.py",
        "test/api/test_external_agent_worker_authority_propagation.py",
        "test/api/test_llm_router_agent_implementation_route.py",
        "test/api/test_llm_router_agent_supervisor_fallback_route.py",
        "test/api/test_llm_router_exact_provider_fallback.py",
        "test/api/test_proof_reuse_default_identity_services.py",
        "test/api/test_pytest_proof_reuse_item_identity.py",
        "test/api/test_plan_r2_remote_owner.py",
    }
    assert required_paths <= set(task["owned_files"])
    argv = task["execution_validation"][0]["argv"]
    assert "test/api/test_eaaef_bootstrap_daemon_gateway.py" in argv
    assert "test/api/test_eaaef_bootstrap_gateway_launch.py" in argv
    assert "test/api/test_eaaef_bootstrap_runtime_gateway.py" in argv
    assert "test/api/test_eaaef_borrowed_transaction.py" in argv
    assert "test/api/test_eaaef_lane_gateway_runtime.py" in argv
    assert "test/api/test_eaaef_operational_schema.py" in argv
    assert "test/api/test_eaaef_quack_command_fabric.py" in argv
    assert "test/api/test_eaaef_supervisor_daemon_birth_wiring.py" in argv
    assert "test/api/test_agent_supervisor_database_implementation_daemon.py" in argv
    assert "test/api/test_agent_supervisor_incremental_runtime.py" in argv
    assert "test/api/test_agent_supervisor_native_dependency_admission.py" in argv
    assert "test/api/test_agent_supervisor_validation_scheduler.py" in argv
    assert "test/api/test_external_agent_state_repository.py" in argv
    assert "test/api/test_plan_r2_remote_owner.py" in argv
    assert "test/api/test_external_agent_worker_authority_propagation.py" in argv

    plan_r2 = next(
        item for item in _board()["tasks"] if item["stable_task_id"] == "EAAEF-009"
    )
    assert "EAAEF-000" in plan_r2["dependencies"]
    assert {
        "ipfs_accelerate_py/agent_supervisor/runtime/plan_r2_remote_owner.py",
        "ipfs_accelerate_py/agent_supervisor/validation/plan_r2_remote_owner_admission.py",
        "test/api/test_plan_r2_remote_owner.py",
    } <= set(plan_r2["owned_files"])
    assert {
        contract["path"]
        for contract in plan_r2["overlap_merge_contracts"]
        if contract["predecessor_task_id"] == "EAAEF-000"
    } == {
        "ipfs_accelerate_py/agent_supervisor/task_sources/external_agent_state_repository.py",
        "test/api/test_external_agent_state_repository.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/plan_r2_remote_owner.py",
        "ipfs_accelerate_py/agent_supervisor/validation/plan_r2_remote_owner_admission.py",
        "test/api/test_plan_r2_remote_owner.py",
    }


def test_control_artifacts_are_generator_or_source_owned_not_worker_outputs() -> None:
    board = _board()
    expected = {
        "docs/architecture/external_agent_autonomous_execution_fabric/OBJECTIVES.md": (
            "generator_owned_projection"
        ),
        "docs/architecture/external_agent_autonomous_execution_fabric/PLAN.md": (
            "reviewed_source_owned_control_document"
        ),
        "docs/architecture/external_agent_autonomous_execution_fabric/TASK_BOARD.md": (
            "generator_owned_projection"
        ),
        "docs/architecture/external_agent_autonomous_execution_fabric/task_board.json": (
            "generator_owned_canonical_board"
        ),
        "docs/architecture/external_agent_autonomous_execution_fabric/stack_compatibility_manifest.json": (
            "reviewed_source_owned_board_input"
        ),
        "docs/architecture/external_agent_autonomous_execution_fabric/source_reconciliation_manifest.json": (
            "reviewed_source_owned_board_input"
        ),
        "docs/architecture/external_agent_autonomous_execution_fabric/reconciliation_report.md": (
            "reviewed_source_owned_human_projection"
        ),
        "docs/architecture/external_agent_autonomous_execution_fabric/bootstrap_materialization_attempts.json": (
            "reviewed_source_owned_evidence_ledger"
        ),
    }
    ownership = {
        item["path"]: item for item in board["control_artifact_ownership"]
    }
    assert {path: item["ownership_class"] for path, item in ownership.items()} == expected
    assert all(item["worker_mutation_admitted"] is False for item in ownership.values())
    worker_outputs = {
        path for task in board["tasks"] for path in task["execution_outputs"]
    }
    assert set(expected).isdisjoint(worker_outputs)


def test_source_manifest_records_exact_sca_merge_deletion_restore() -> None:
    source = json.loads(
        (CAMPAIGN / "source_reconciliation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    regression = source["repositories"]["ipfs_accelerate_py"][
        "semantic_contract_assurance_merge_deletion"
    ]
    assert regression["source_snapshot"]["commit"] == (
        "38cd50092d300b61327a9225e7f10cfe8acefb4f"
    )
    assert regression["retained_parent"]["commit"] == (
        "56ef4fa6479eee78cacfe7372df67a5eda329060"
    )
    assert regression["accidental_drop"]["commit"] == (
        "ea9b2af0c1e772ba445ab1589b123050484c217a"
    )
    restoration = regression["restoration"]
    assert restoration["path_count"] == len(restoration["paths"]) == 25
    restored_paths = {item["path"] for item in restoration["paths"]}
    assert "scripts/extract_typescript_ast.mjs" in restored_paths
    assert len(
        {
            path
            for path in restored_paths
            if path.startswith("test/api/test_agent_supervisor_")
        }
    ) == 12
    assert all(
        isinstance(item["blob"], str) and len(item["blob"]) == 40
        for item in restoration["paths"]
    )


def test_source_manifest_records_same_lane_lifecycle_forward_reconciliation() -> None:
    source = json.loads(
        (CAMPAIGN / "source_reconciliation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    lifecycle = source["repositories"]["ipfs_accelerate_py"][
        "worktree_lifecycle_restart_reconciliation"
    ]
    assert lifecycle["source_commit"] == (
        "9e39c6c9edb0b756f99f9857a89e70642ef1321c"
    )
    assert lifecycle["source_tree"] == (
        "ea321ea749103ece6a175c4e984372e42ac204bd"
    )
    assert lifecycle["qualification_observation"]["sealed_promotion_receipt"] is False
    assert lifecycle["paths"] == [
        "ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py",
        "test/api/test_agent_supervisor_worktree_lifecycle.py",
    ]


def test_source_manifest_records_registered_sibling_containment_repair() -> None:
    source = json.loads(
        (CAMPAIGN / "source_reconciliation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    repair = source["repositories"]["ipfs_accelerate_py"][
        "todo_daemon_registered_shared_dependency_reconciliation"
    ]
    assert repair["source_commit"] == (
        "a4413463c6e9d356b0143750db98486c0689bb0a"
    )
    assert repair["source_tree"] == (
        "f205cbf01c9677549e25ae733a7ad2fdf8610b3d"
    )
    assert "Git registers it as a worktree" in repair["containment_correction"]
    assert repair["qualification_observation"]["sealed_promotion_receipt"] is False


def test_source_manifest_records_signed_command_fabric_profile_v2_repair() -> None:
    source = json.loads(
        (CAMPAIGN / "source_reconciliation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    repair = source["repositories"]["ipfs_accelerate_py"][
        "signed_command_fabric_profile_v2_reconciliation"
    ]
    assert repair["authoritative_schema"].endswith(
        "eaaef-signed-command-fabric-profile@2"
    )
    assert repair["board_namespace"] == (
        "external-agent-autonomous-execution-fabric-v1"
    )
    assert repair["shard_id"] == "control-shard-0"
    assert repair["qualification_observation"]["sealed_promotion_receipt"] is False


def test_source_manifest_records_isolated_validator_import_root_repair() -> None:
    source = json.loads(
        (CAMPAIGN / "source_reconciliation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    repair = source["repositories"]["ipfs_accelerate_py"][
        "isolated_materializer_validator_import_root_reconciliation"
    ]
    assert repair["security_review"]["result"] == "no_p0_or_p1_finding"
    assert repair["qualification_observation"]["collected"] == 20
    assert repair["qualification_observation"]["passed"] == 20
    assert repair["qualification_observation"]["process_started"] is False
    assert repair["qualification_observation"]["sealed_promotion_receipt"] is False


def test_stack_manifest_binds_new_fail_closed_bootstrap_contracts() -> None:
    stack = json.loads(
        (CAMPAIGN / "stack_compatibility_manifest.json").read_text(encoding="utf-8")
    )
    assert stack["manifest_revision"] == 2
    assert stack["supersedes_manifest_cid"] == (
        "sha256:b79f49c80c50086ee8929f56008b4b20263a710cea0cf916904bc9d1fe540eb7"
    )
    contracts = stack["frozen_contracts"]["control_plane"][
        "external_agent_fabric_contracts"
    ]
    assert stack["frozen_contracts"]["control_plane"][
        "bootstrap_schema_revision"
    ] == "datasets-authoritative-eaaef-operational-control-plane@2"
    assert contracts["eaaef_signed_command_fabric_profile"] == (
        "ipfs_accelerate_py/agent-supervisor/"
        "eaaef-signed-command-fabric-profile@2"
    )
    assert contracts["eaaef_signed_command_fabric_profile_identity"] == {
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "shard_id": "control-shard-0",
        "schema_revision": (
            "datasets-authoritative-eaaef-operational-control-plane@2"
        ),
        "child_adapter_status": "implemented_unqualified_fail_closed",
        "production_admitted": False,
    }
    assert contracts["worker_container_execution_profile_launch"] == (
        "ipfs_accelerate_py/agent-supervisor/"
        "source-addressed-container-execution-profile-launch@1"
    )
    assert contracts["worker_image_qualification"].endswith(
        "external-agent-worker-image-qualification@1"
    )
    assert contracts["worker_container_profile"].endswith(
        "external-agent-worker-container-profile@1"
    )
    assert contracts["worker_container_profile_v2"] == (
        "ipfs_accelerate_py/agent-supervisor/"
        "external-agent-worker-container-profile@2"
    )
    assert contracts["worker_container_execution_profile_launch_v2"] == (
        "ipfs_accelerate_py/agent-supervisor/"
        "source-addressed-container-execution-profile-launch@2"
    )
    assert contracts["eaaef_bootstrap_daemon_capability"] == {
        "interface": "EAAEFBootstrapDaemonCapability@1",
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-bootstrap-daemon-capability@1"
        ),
    }
    assert contracts["eaaef_bootstrap_daemon_gateway"] == {
        "interface": "EAAEFBootstrapDaemonGateway@1",
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-bootstrap-daemon-gateway@1"
        ),
    }
    assert contracts["eaaef_bootstrap_daemon_operation_disposition"] == (
        "ipfs_accelerate_py/agent-supervisor/"
        "eaaef-bootstrap-daemon-operation-disposition@1"
    )
    assert contracts["eaaef_operational_profile"] == {
        "interface": "DatasetsAuthoritativeEAAEFOperationalProfile@2",
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "datasets-authoritative-eaaef-operational-control-plane@2"
        ),
        "migration_id": "0002_eaaef_owner_transaction_operational_extension",
        "production_admitted": False,
    }
    assert contracts["eaaef_bootstrap_daemon_operational_capability"] == {
        "interface": "EAAEFBootstrapDaemonOperationalCapability@2",
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-bootstrap-daemon-operational-capability@2"
        ),
    }
    assert contracts["eaaef_bootstrap_gateway_binding"] == {
        "interface": "EAAEFBootstrapGatewayBinding@1",
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-bootstrap-gateway-binding@1"
        ),
    }
    assert contracts["eaaef_borrowed_transaction_adapter"][
        "qualification_status"
    ] == "implemented_unqualified_fail_closed"
    assert contracts["eaaef_bootstrap_borrowed_transaction_handler"][
        "qualification_status"
    ] == "implemented_unqualified_fail_closed"
    assert contracts["eaaef_task_operation_authority"].endswith(
        "eaaef-task-operation-authority@2"
    )
    runtime_proxy = contracts["eaaef_bootstrap_execution_repository_proxy"]
    assert runtime_proxy["interface"] == "EAAEFBootstrapExecutionRepositoryProxy@2"
    assert runtime_proxy["qualification_status"] == (
        "r1_source_verified_runtime_factory_implemented"
    )
    assert runtime_proxy["source_factory_implemented"] is True
    assert runtime_proxy["source_dispatch_path_implemented"] is True
    assert runtime_proxy["direct_construction_allowed"] is False
    assert runtime_proxy["production_admitted"] is False
    assert runtime_proxy["production_blockers"] == [
        "signed_quack_client_factory_qualification_artifact_absent",
        "signed_dynamic_dispatcher_service_qualification_artifact_absent",
        "independently_signed_per_birth_lane_runtime_artifact_absent",
    ]
    qualification = stack["qualification_state"]
    assert qualification["bootstrap_daemon_operations"].startswith("no_go_")
    assert "all_31_borrowed_transaction_handlers_and_source_runtime_factory_implemented" in (
        qualification["bootstrap_daemon_operations"]
    )
    assert qualification["bootstrap_runtime_gateway"].startswith("no_go_")
    assert "source_verified_runtime_factory_implemented" in (
        qualification["bootstrap_runtime_gateway"]
    )
    assert qualification["plan_r2_remote_owner"].startswith("no_go_")
    assert "source_complete_external_signed_channel_required" in (
        qualification["plan_r2_remote_owner"]
    )
    assert qualification["provider_effect_launch"].startswith("no_go_")
    assert "external_signed_v2_artifact_admitted_engine" in (
        qualification["provider_effect_launch"]
    )


def _mutated_board_errors(board: dict) -> list[str]:
    validator = _load_validator()
    source = json.loads(
        (CAMPAIGN / "source_reconciliation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    stack = json.loads(
        (CAMPAIGN / "stack_compatibility_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    errors: list[str] = []
    validator._validate_board(board, source, stack, errors)
    return errors


def test_repeated_owned_paths_form_exact_serialized_predecessor_chains() -> None:
    last_owner: dict[tuple[str, str], str] = {}
    contract_count = 0
    for task in _board()["tasks"]:
        contracts = {
            (contract["repository"], contract["path"]): contract
            for contract in task["overlap_merge_contracts"]
        }
        expected_contract_paths: set[tuple[str, str]] = set()
        for path in task["owned_files"]:
            key = (task["owning_repository"], path)
            predecessor = last_owner.get(key)
            if predecessor is not None:
                expected_contract_paths.add(key)
                contract = contracts[key]
                assert contract == {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "external-agent-owned-path-overlap-contract@1"
                    ),
                    "repository": task["owning_repository"],
                    "path": path,
                    "predecessor_task_id": predecessor,
                    "successor_task_id": task["stable_task_id"],
                    "dependency_type": "direct",
                    "strategy": "serialized_forward_extension",
                    "merge_lane": "single_admitted_merge_lane",
                }
                assert predecessor in task["dependencies"]
                contract_count += 1
            last_owner[key] = task["stable_task_id"]
        assert set(contracts) == expected_contract_paths
    assert contract_count == 31


@pytest.mark.parametrize(
    "mutation, expected_error",
    (
        (
            "missing_contract",
            "owned-path overlap contracts differ from the exact serialized predecessor chain",
        ),
        (
            "wrong_strategy",
            "owned-path overlap contracts differ from the exact serialized predecessor chain",
        ),
        (
            "missing_direct_dependency",
            "overlap predecessor EAAEF-000 is not a direct dependency",
        ),
    ),
)
def test_overlap_contract_validator_rejects_unserialized_conflicts(
    mutation: str, expected_error: str
) -> None:
    board = copy.deepcopy(_board())
    task = next(
        item for item in board["tasks"] if item["stable_task_id"] == "EAAEF-091"
    )
    if mutation == "missing_contract":
        task["overlap_merge_contracts"].pop()
    elif mutation == "wrong_strategy":
        task["overlap_merge_contracts"][0]["strategy"] = "parallel_last_writer_wins"
    else:
        task["dependencies"].remove("EAAEF-000")
    errors = _mutated_board_errors(board)
    assert any(expected_error in error for error in errors), errors
