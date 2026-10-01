"""Run native semantic/world/Doctor integration checks in an offline container.

This exercises real databases, producer scans, provers and repair transactions.
It makes no model calls and does not produce a Terminal-Bench score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import time
import xml.etree.ElementTree as ET


SOURCE = Path(__file__).resolve().parents[3]
TESTS = (
    "test/api/semantic_state/test_semantic_context_runtime.py",
    "test/api/semantic_state/test_semantic_context_refresh.py",
    "test/api/semantic_state/test_semantic_context_retry.py",
    "test/api/semantic_state/test_semantic_worker_projection.py",
    "test/api/semantic_state/test_semantic_capsule_selection.py",
    "test/api/semantic_state/test_supervised_task_context.py",
    "test/api/semantic_state/test_code_retrieval_context.py",
    "test/api/semantic_state/test_intent_world_snapshot.py",
    "test/api/semantic_state/test_intent_progress.py",
    "test/api/semantic_state/test_world_snapshot_contracts.py",
    "test/api/semantic_state/test_published_task_context.py",
    "test/api/semantic_state/test_published_retrieval.py",
    "test/api/semantic_state/test_context_pack.py",
    "test/api/test_agent_supervisor_deterministic_doctor_runtime.py",
    "test/api/test_agent_supervisor_doctor_repair_composition.py",
    "test/api/test_doctor_repair_eligibility.py",
    "test/api/test_doctor_task_workflow.py",
    "test/api/test_doctor_candidate_runner.py",
    "test/api/test_doctor_residual_context.py",
    "test/api/test_terminal_doctor_dispatch.py",
    "test/api/test_native_doctor_interruption.py",
    "test/api/test_agent_supervisor_context_delta.py",
    "test/api/test_agent_supervisor_context_compiler.py",
    "test/api/test_context_chunk_losslessness.py",
    "test/api/test_agent_supervisor_implementation_failure_review.py",
    "test/api/test_agent_supervisor_doctor_worktree_adapter.py",
    "test/api/test_agent_supervisor_code_symbol_vector_index.py",
    "test/api/test_database_portal_attempt_budget.py",
    "test/api/test_database_attempt_diagnostic_feedback.py",
    "test/api/test_database_projection_alias_parsing.py",
    "test/api/test_database_completion_attempt_binding.py",
    "test/api/test_intent_ordinary_claim.py",
    "test/api/test_database_bounded_pre_effect_retry.py",
    "test/api/test_supervisor_planning_composition.py",
    "test/api/test_supervisor_meta_index.py",
    "test/api/test_router_implementation_runner.py",
    "test/api/test_router_workspace_contract.py",
    "test/api/test_semantic_router_translation.py",
    "test/api/test_semantic_router_integration.py",
    "test/api/test_admitted_owner_git_identity.py",
    "test/api/test_codex_tool_diagnostics.py",
    "test/api/test_agent_supervisor_local_planning_admission.py",
    "test/api/test_agent_supervisor_formal_plan_compiler.py",
    "test/api/test_agent_supervisor_prompt_directory_scanner.py",
    "test/api/test_agent_supervisor_prompt_goal_planner.py",
    "test/api/test_native_quack_owner_qualification.py",
    "test/api/test_agent_supervisor_quack_transport_native.py",
    "test/integration/test_isolated_benchmark_runtime.py",
    "test/integration/test_native_owner_live_heartbeat.py",
    "test/integration/test_admitted_benchmark_runtime.py",
    "test/integration/test_admitted_published_shutdown.py",
    "test/integration/test_large_native_task_receipt_cas.py",
    "test/api/test_local_completion_bridge.py",
    "test/api/test_local_planning_receipt_storage.py",
    "test/api/test_local_validation_python_launcher.py",
    "test/api/test_validation_memfd_portability.py",
    "benchmarks/agent_supervisor/container_coding/test_vector_index_preflight.py",
    "benchmarks/agent_supervisor/container_coding/test_local_planner_bridge.py",
    "benchmarks/agent_supervisor/container_coding/test_native_codex_baseline.py",
    "benchmarks/agent_supervisor/container_coding/test_terminal_indexed_preparation.py",
    "benchmarks/agent_supervisor/container_coding/test_terminal_initial_context.py",
    "benchmarks/agent_supervisor/container_coding/test_terminal_context_rebind.py",
    "benchmarks/agent_supervisor/container_coding/test_terminal_context_audit.py",
    "benchmarks/agent_supervisor/container_coding/test_terminal_translation_audit.py",
    "benchmarks/agent_supervisor/container_coding/test_post_stop_refresh_budget.py",
    "benchmarks/agent_supervisor/container_coding/test_benchmark_comparison.py",
    "benchmarks/agent_supervisor/container_coding/test_native_supervisor_diagnostics.py",
    "benchmarks/agent_supervisor/container_coding/test_cleanup_deadline.py",
    "benchmarks/agent_supervisor/container_coding/test_native_codex_bundle.py",
    "benchmarks/agent_supervisor/container_coding/test_native_doctor_gate.py",
    "test/integration/test_admitted_context_refresh.py",
    "test/integration/test_agent_supervisor_doctor_transaction_live.py",
    "test/integration/test_agent_supervisor_doctor_transaction_validation.py",
)


def verified_doctor_qualification(path: Path) -> dict:
    """Require positive native receipts, independently of the process exit."""
    result = json.loads(path.read_bytes())
    required = (
        "tactician_planned", "sealed_proof_verified", "native_synthesis_admitted",
        "graph_impact_closed", "transaction_committed", "publication_verified",
        "before_behavior_failed", "after_behavior_passed",
    )
    count = result.get("executed_validation_receipts")
    if (result.get("schema") != "doctor-native-composition-qualification@1"
            or any(result.get(key) is not True for key in required)
            or type(count) is not int or count <= 0
            or result.get("task_completion_authorized") is not False
            or result.get("terminal_bench_result") is not False
            or result.get("provider_calls") != 0
            or not result.get("committed_commit")
            or result.get("committed_commit") == result.get("expected_base_commit")):
        raise ValueError("Doctor qualification is absent, incomplete, or overclaims authority")
    return result


def verified_indexed_lifecycle(path: Path) -> dict:
    """Require the joined native task lifecycle, including observed checks."""
    result = json.loads(path.read_bytes())
    before, after = result.get("before_context", {}), result.get("after_context", {})
    if (result.get("schema") != "indexed-doctor-local-lifecycle-qualification@1"
            or result.get("status") != "qualified" or result.get("task_status") != "completed"
            or any(result.get(key) is not True for key in (
                "pending_acceptance_enforced", "native_prompt_context_consumed",
                "native_doctor_transaction_committed", "actual_native_completion"))
            or result.get("before_validation", {}).get("passed") is not False
            or result.get("after_validation", {}).get("passed") is not True
            or not before.get("semantic_root_cid") or not after.get("semantic_root_cid")
            or before["semantic_root_cid"] == after["semantic_root_cid"]
            or before.get("task_cid") != result.get("task_cid")
            or after.get("task_cid") != result.get("task_cid")
            or result.get("text_generation_calls") != 0
            or result.get("terminal_bench_result") is not False
            or result.get("full_supervisor_started") is not False):
        raise ValueError("indexed lifecycle is incomplete or overclaims benchmark execution")
    return result


def _read_refresh_artifact(root: Path, relative: str) -> bytes:
    """Read a bounded retained receipt without following repository links."""
    path = PurePosixPath(relative)
    if (path.is_absolute() or str(path) != relative or not path.parts
            or ".." in path.parts or path.parts[0] != ".runtime"):
        raise ValueError("published refresh path must be canonical repository runtime data")
    root = root.absolute()
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    leaf = None
    try:
        for part in (*root.parts[1:], *path.parts[:-1]):
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        leaf = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        before = os.fstat(leaf)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= 2_000_000:
            raise ValueError("published refresh must be a bounded regular file")
        with os.fdopen(leaf, "rb", closefd=False) as stream:
            raw = stream.read(2_000_001)
        after = os.fstat(leaf)
        if len(raw) > 2_000_000 or any(getattr(before, name) != getattr(after, name)
                for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("published refresh changed during read")
        return raw
    finally:
        if leaf is not None:
            os.close(leaf)
        os.close(fd)


def _verify_native_doctor_result(result: dict, refresh: dict) -> dict:
    """Check retained native observations, without granting live authority."""
    initial = result.get("initial_context", {})
    after = result.get("after_stop", {})
    rows = after.get("published_context", [])
    start, stop = result.get("start", {}), result.get("stop", {})
    if (result.get("schema") != "native-doctor-supervision-qualification@1"
            or result.get("qualified") is not True
            or type(result.get("provider_calls")) is not int or result["provider_calls"] != 0
            or result.get("benchmark_result") is not False
            or result.get("production_activation") is not False
            or result.get("token_savings") is not None
            or result.get("task") != {"status": "completed", "revision": 4}
            or type(result.get("task", {}).get("revision")) is not int
            or type(result.get("initial_public_check_exit_code")) is not int
            or result["initial_public_check_exit_code"] <= 0
            or type(result.get("final_public_check_exit_code")) is not int
            or result["final_public_check_exit_code"] != 0
            or result.get("pre_start_task_status") != "ready"
            or result.get("canonical_unchanged_before_start") is not True
            or not result.get("baseline_commit") or not result.get("published_commit")
            or result["baseline_commit"] == result["published_commit"]
            or not result.get("doctor_candidate_commit")
            or result["doctor_candidate_commit"] == result["baseline_commit"]):
        raise ValueError("native Doctor task or actual validation lifecycle is incomplete")
    for operation, receipt in (("start", start), ("stop", stop)):
        if (receipt.get("schema") != "ipfs_accelerate_py/agent-supervisor/operation-result@1"
                or receipt.get("operation") != operation or receipt.get("status") != "succeeded"
                or receipt.get("authority") != "mutation" or receipt.get("error") is not None
                or not receipt.get("audit_receipt_id") or not receipt.get("data", {}).get("receipt_id")):
            raise ValueError("native Doctor lifecycle operation receipt is incomplete")
    if (any(not start.get(key) or start.get(key) != stop.get(key)
            for key in ("repository_id", "policy_id", "objective_id"))
            or stop.get("data", {}).get("old_tree_fenced") is not True
            or stop["data"].get("new_process_identity") is not None
            or type(result.get("remaining_processes")) is not int or result["remaining_processes"] != 0
            or after.get("process_tree", {}).get("members") != []):
        raise ValueError("native Doctor STOP did not fence and empty its process tree")
    expected_stages = {"tactician": "planned", "proof": "verified",
                       "synthesis_preview": "supported", "plan_compilation": "admitted"}
    stages = result.get("doctor_stages", {})
    if (any(stages.get(name, {}).get("status") != "executed"
            or stages[name].get("disposition") != disposition or not stages[name].get("receipt_id")
            for name, disposition in expected_stages.items())
            or stages.get("proof", {}).get("mutation_capable") is not True
            or not stages.get("proof", {}).get("scope")
            or stages.get("impact", {}).get("status") != "executed"
            or stages["impact"].get("mutation_admissible") is not True
            or not stages["impact"].get("receipt_id")):
        raise ValueError("native Doctor tactician/proof/synthesis/impact receipts are incomplete")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("native Doctor requires one retained publication refresh")
    row = rows[0]
    if (row.get("status") != "refreshed" or row.get("task_revision") != 4
            or refresh.get("schema") != "supervisor-task-context-preparation@1"
            or refresh.get("refresh_schema") != "supervisor-published-task-context@1"
            or refresh.get("task_status") != "completed" or refresh.get("task_revision") != 4
            or any(item.get("task_cid") != result.get("task_cid") for item in (initial, row, refresh))
            or not result.get("task_cid") or refresh.get("published_commit") != result["published_commit"]
            or refresh.get("predecessor_bundle") != result.get("context_bundle")
            or row.get("context_bundle") != refresh.get("context_bundle")
            or any(refresh.get(key) is not False for key in (
                "execution_authority", "completion_authority", "canonical_task_mutated", "launch_nomination_mutated"))
            or row.get("completion_authority") is not False
            or type(refresh.get("text_generation_calls")) is not int or refresh["text_generation_calls"] != 0):
        raise ValueError("native Doctor refreshed task/commit/authority bindings differ")
    for name in ("semantic_root_cid", "world_snapshot_cid"):
        if (not initial.get(name) or refresh.get("predecessor_" + name) != initial[name]
                or not refresh.get(name) or refresh[name] == initial[name] or row.get(name) != refresh[name]):
            raise ValueError("native Doctor semantic/world refresh did not change the bound predecessor")
    world = refresh.get("world", {}).get("planning_context", {})
    semantic = refresh.get("semantic", {})
    if (world.get("semantic_root_cid") != refresh["semantic_root_cid"]
            or world.get("world_snapshot_cid") != refresh["world_snapshot_cid"]
            or semantic.get("semantic_root_cid") != refresh["semantic_root_cid"]
            or semantic.get("reconstruction", {}).get("nomination_matched") is not True
            or semantic.get("ducklake", {}).get("status") != "projected"
            or refresh.get("world", {}).get("metadata", {}).get("status") != "projected"
            or not any(task.get("task_cid") == result["task_cid"] and task.get("status") == "completed"
                       and task.get("revision") == 4 for task in world.get("tasks", []))):
        raise ValueError("native Doctor semantic/world reconstruction or DuckLake projection is incomplete")
    retrieval = refresh.get("retrieval", {})
    scope = refresh.get("source_scope", {})
    if (row.get("retrieval_status") != "current" or retrieval.get("status") != "current"
            or retrieval.get("reason") != "rebuilt_with_pinned_configuration"
            or not retrieval.get("index_id") or not retrieval.get("result_id")
            or retrieval.get("previous_index_id") != result.get("vector_index_id")
            or retrieval.get("previous_index_id") != initial.get("retrieval", {}).get("index_id")
            or retrieval["index_id"] == retrieval.get("previous_index_id")
            or not retrieval.get("config_id") or retrieval["config_id"] != retrieval.get("previous_config_id")
            or retrieval.get("ducklake", {}).get("status") != "projected"
            or retrieval["ducklake"].get("authoritative") is not False
            or retrieval["ducklake"].get("completion_authority") is not False
            or scope.get("scope_expanded") is not False
            or not scope.get("preserved_paths") or scope.get("declared_outputs_outside_index_scope") != []
            or scope.get("accepted_source_tree_includes_declared_outputs") is not True
            or row.get("source_scope") != scope):
        raise ValueError("native Doctor source-current pinned retrieval refresh is incomplete")
    return result


def verified_native_doctor_lifecycle(path: Path) -> dict:
    """Require native daemon completion, fenced STOP and all refreshed indexes."""
    result = json.loads(path.read_bytes())
    rows = result.get("after_stop", {}).get("published_context", [])
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("native Doctor publication refresh receipt is missing")
    row = rows[0]
    raw = _read_refresh_artifact(path.parent / "repository", row.get("refresh_artifact", ""))
    if hashlib.sha256(raw).hexdigest() != row.get("refresh_sha256"):
        raise ValueError("native Doctor publication refresh digest differs")
    return _verify_native_doctor_result(result, json.loads(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets-source", type=Path, default=SOURCE.parent / "ipfs_datasets")
    parser.add_argument("--kit-source", type=Path, default=SOURCE.parent / "ipfs_kit")
    parser.add_argument("--lean-toolchain", type=Path, required=True)
    parser.add_argument("--image", default="ipfs-supervisor-integration:local")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--ducklake-extension-dir", type=Path)
    args = parser.parse_args()
    datasets = args.datasets_source.resolve(strict=True)
    kit = args.kit_source.resolve(strict=True)
    lean = args.lean_toolchain.resolve(strict=True)
    if not (lean / "bin/lean").is_file():
        parser.error("--lean-toolchain must contain bin/lean")
    for path in TESTS:
        (SOURCE / path).resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # A rootless Docker bind mount maps the invoking user to container root.
    # Its group can write this new dedicated output; proof processes stay uid1000.
    output.chmod(0o770)

    def logged(argv, name, timeout=600):
        with (output / name).open("w") as stream:
            return subprocess.run(argv, cwd=SOURCE, stdout=stream, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode

    if args.build:
        if args.ducklake_extension_dir is None:
            parser.error("--build requires --ducklake-extension-dir for the native extension")
        extension = args.ducklake_extension_dir.resolve(strict=True)
        if logged(["docker", "build", "--build-context", f"ducklake-extension={extension}",
                   "-f", str(Path(__file__).with_name("Dockerfile.supervision")), "-t", args.image,
                   str(Path(__file__).parent)], "build.log"):
            raise SystemExit("Docker build failed; see " + str(output / "build.log"))
    image = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", args.image], text=True).strip()
    files = [*TESTS, "ipfs_accelerate_py/llm_router.py",
             "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
             "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
             "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_attempt_feedback.py",
             "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon_runner.py",
             "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
             "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
             "ipfs_accelerate_py/agent_supervisor/analysis/code_symbol_vector_index.py",
             "ipfs_accelerate_py/agent_supervisor/entrypoints/service_factory.py",
             "ipfs_accelerate_py/agent_supervisor/entrypoints/isolated_benchmark_runtime.py",
             "ipfs_accelerate_py/agent_supervisor/entrypoints/admitted_benchmark_runtime.py",
             "ipfs_accelerate_py/agent_supervisor/todo_daemon/native_owner_bootstrap.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/local_completion_bridge.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/candidate_execution.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/container_worker_boundary.py",
             "ipfs_accelerate_py/agent_supervisor/task_sources/typed_database_task_source.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/local_planning_admission.py",
             "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py",
             "ipfs_accelerate_py/agent_supervisor/prompt/prompt_directory_scanner.py",
             "ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py",
             "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/semantic_context_runtime.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/semantic_capsule_selection.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py",
             "ipfs_accelerate_py/agent_supervisor/context/context_compiler.py",
             "ipfs_accelerate_py/agent_supervisor/context/context_contracts.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/published_task_context.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/published_retrieval.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/published_learned_retrieval.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/local_learned_embedding.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/supervised_task_context.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/supervisor_meta_index.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/code_retrieval_context.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/codex_usage_receipt.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/task_context_bundle.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/doctor_repair_composition.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/doctor_task_workflow.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/doctor_candidate_runner.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/doctor_residual_context.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py",
             "ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transaction.py",
             "ipfs_accelerate_py/agent_supervisor/semantic_state/intent_world_snapshot.py",
             "ipfs_accelerate_py/agent_supervisor/semantic_state/intent_progress.py",
             "ipfs_accelerate_py/agent_supervisor/proof/deterministic_doctor_hammer.py",
             "ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py",
             "ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py",
             "ipfs_accelerate_py/agent_supervisor/analysis/deterministic_doctor_impact.py",
             "ipfs_accelerate_py/agent_supervisor/runtime/doctor_worktree_adapter.py",
             "benchmarks/agent_supervisor/container_coding/doctor_composition_smoke.py",
             "benchmarks/agent_supervisor/container_coding/indexed_doctor_lifecycle.py",
             "benchmarks/agent_supervisor/container_coding/native_doctor_supervision.py",
             "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
             "benchmarks/agent_supervisor/container_coding/terminal_context_audit.py",
             "benchmarks/agent_supervisor/container_coding/terminal_context_rebind.py",
             "benchmarks/agent_supervisor/container_coding/run_supervision_docker.py",
             "benchmarks/agent_supervisor/container_coding/fixtures/native_doctor_gate_observations.json",
             "benchmarks/agent_supervisor/container_coding/local_planning_qualification.py",
             "benchmarks/agent_supervisor/container_coding/local_live_planner.py",
             "benchmarks/agent_supervisor/container_coding/live_supervisor_qualification.py",
             "benchmarks/agent_supervisor/container_coding/native_codex_baseline.py",
             "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
             "benchmarks/agent_supervisor/container_coding/terminal_indexed_preparation.py",
             "benchmarks/agent_supervisor/container_coding/terminal_initial_context.py",
             "benchmarks/agent_supervisor/container_coding/terminal_deployment.py",
             "benchmarks/agent_supervisor/container_coding/container_worker_deployment.py",
             "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py",
             "benchmarks/agent_supervisor/container_coding/full_supervisor_benchmark.py",
             "benchmarks/agent_supervisor/container_coding/benchmark_comparison.py",
             "benchmarks/agent_supervisor/container_coding/native_quack_qualification.py",
             "benchmarks/agent_supervisor/container_coding/retrieval_prompt_smoke.py",
             "benchmarks/agent_supervisor/container_coding/vector_index_preflight.py",
             "benchmarks/agent_supervisor/container_coding/learned_vector_preflight.py",
             "benchmarks/agent_supervisor/container_coding/test_learned_vector_preflight.py",
             "test/api/semantic_state/test_published_learned_retrieval.py",
             "test/integration/test_admitted_learned_refresh.py",
             "test/integration/test_initial_learned_context_reuse.py",
             "benchmarks/agent_supervisor/container_coding/doctor_symbol_repair.py"]
    before_hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in files}
    # Include the actual external native producer and shared CID contracts.
    dependency_files = sorted([*(datasets / "ipfs_datasets_py/logic/software_contracts").rglob("*.py"),
                              datasets / "ipfs_datasets_py/logic/families/translations.py"])
    dependency_hashes = {str(path.relative_to(datasets)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in dependency_files}
    if not dependency_hashes:
        raise ValueError("native semantic producer sources were not found")
    # Freeze the resolved image ID for every command in this run.
    command = ["docker", "run", "--rm", "--network", "none", "--user", "1000:0", "--cpus", "4",
               "--memory", "4g", "--pids-limit", "512",
               "-v", f"{SOURCE}:/source:ro", "-v", f"{datasets}:/datasets:ro",
               "-v", f"{kit}:/kit:ro", "-v", f"{lean}:/toolchains/lean:ro",
               "-v", f"{output}:/results", "-e", "DOCTOR_COMPOSITION_LEAN=/toolchains/lean/bin/lean",
               "-e", "IPFS_ACCELERATE_RUN_LIVE_QUACK=1"]
    started = time.monotonic()
    tests_rc = logged([*command, image, "--junitxml=/results/tests.xml", *TESTS], "tests.log")
    smoke_rc = logged([*command, "--entrypoint", "python", image,
                       "benchmarks/agent_supervisor/container_coding/indexed_doctor_lifecycle.py",
                       "--output", "/results/indexed-lifecycle"], "indexed-lifecycle.log")
    native_doctor_rc = logged([*command, "--entrypoint", "python", image,
                              "benchmarks/agent_supervisor/container_coding/native_doctor_supervision.py",
                              "--output", "/results/native-doctor"], "native-doctor.log")
    vector_rc = logged([*command, "--entrypoint", "python", image,
                       "benchmarks/agent_supervisor/container_coding/vector_index_preflight.py",
                       "--repository", "/source", "--output", "/results/vector",
                       "--file", "ipfs_accelerate_py/agent_supervisor/todo_daemon/retained_callback_suffix.py",
                       "--query", "verified_seed_predecessor"], "vector.log")
    vector = None
    vector_error = ""
    try:
        vector = json.loads((output / "vector/result.json").read_bytes())
        if (vector.get("schema") != "native-lexical-vector-qualification@1"
                or vector.get("status") != "qualified"
                or type(vector.get("symbols")) is not int or vector["symbols"] < 1
                or vector.get("native_fact_rows_replayed") != vector["symbols"]
                or vector.get("complete_permitted_scope") is not True
                or vector.get("ducklake", {}).get("status") != "projected"
                or vector.get("provider_calls") != 0
                or vector.get("semantic_authority") is not False
                or vector.get("full_system_qualified") is not False):
            raise ValueError("native vector qualification is incomplete")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        vector, vector_error = None, str(error)
    doctor = None
    doctor_error = ""
    qualification_path = output / "indexed-lifecycle/doctor/qualification.json"
    try:
        doctor = verified_doctor_qualification(qualification_path)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        doctor_error = str(error)
    lifecycle = None
    lifecycle_error = ""
    try:
        lifecycle = verified_indexed_lifecycle(output / "indexed-lifecycle/result.json")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        lifecycle_error = str(error)
    native_doctor = None
    native_doctor_error = ""
    native_doctor_path = output / "native-doctor/result.json"
    try:
        native_doctor = verified_native_doctor_lifecycle(native_doctor_path)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        native_doctor_error = str(error)
    counts = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
    if (output / "tests.xml").is_file():
        for suite in ET.parse(output / "tests.xml").getroot().iter("testsuite"):
            for name in counts:
                counts[name] += int(suite.get(name, "0"))
    after_hashes = {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in files}
    changed = before_hashes != after_hashes or dependency_hashes != {
        str(path.relative_to(datasets)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in dependency_files}
    result = {
        "schema": "supervisor-docker-integration@1", "image_id": image,
        "container_limits": {"cpus": 4, "memory": "4g", "pids": 512},
        "status": "passed" if tests_rc == smoke_rc == vector_rc == native_doctor_rc == 0 and counts["tests"] > 0
            and not any(counts[key] for key in ("failures", "errors", "skipped"))
            and not changed and doctor is not None and vector is not None and lifecycle is not None
            and native_doctor is not None
            else "failed_or_incomplete",
        "test_exit_code": tests_rc, "doctor_exit_code": smoke_rc, "counts": counts,
        "vector_exit_code": vector_rc, "vector_qualification": vector,
        "vector_qualification_error": vector_error,
        "seconds": time.monotonic() - started,
        "source_sha256": before_hashes, "source_changed_during_run": changed,
        "datasets_source_sha256": dependency_hashes,
        "test_scope": TESTS, "full_repository_suite_run": False,
        "doctor_qualification": doctor, "doctor_qualification_error": doctor_error,
        "indexed_lifecycle": lifecycle, "indexed_lifecycle_error": lifecycle_error,
        "native_doctor_exit_code": native_doctor_rc,
        "native_doctor_lifecycle": native_doctor,
        "native_doctor_lifecycle_error": native_doctor_error,
        "native_doctor_lifecycle_sha256": hashlib.sha256(native_doctor_path.read_bytes()).hexdigest()
            if native_doctor is not None else None,
        "doctor_qualification_sha256": hashlib.sha256(qualification_path.read_bytes()).hexdigest()
            if doctor is not None else None,
        "lean_executable_sha256": hashlib.sha256((lean / "bin/lean").read_bytes()).hexdigest(),
        "provider_calls": 0, "provider_tokens": None,
        "full_indexed_benchmark_qualified": False,
        "benchmark_comparison": "not measured by this provider-free integration qualification",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
