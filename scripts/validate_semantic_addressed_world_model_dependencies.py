#!/usr/bin/env python3
"""Verify the exact, offline SAWM R2 source and dependency seal.

This gate is deliberately read-only.  It neither imports optional providers
in-process nor opens DuckDB.  The separate cold-import probe runs in a bounded
child with network, subprocess and database constructors denied.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SEAL_PATH = REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
NAMESPACE = "semantic-addressed-world-model-v1"
REVISION = "SAWM-PLAN-R2"
SCHEMA = "semantic-addressed-world-model/dependency-seal-validation@1"

# These are the only source differences admitted above the sealed base before
# implementation workers begin.  The two README files explain controls but are
# intentionally not completion authority or worker-owned outputs.
CONTROL_PATHS = frozenset(
    {
        ".gitignore",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        "docs/architecture/semantic_addressed_world_model.objectives.md",
        "docs/architecture/semantic_addressed_world_model.todo.md",
        "docs/architecture/semantic_addressed_world_model_inventory/repository_baseline.json",
        "docs/architecture/semantic_addressed_world_model_inventory/authority_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/overlap_gap_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/identity_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/interface_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/dependency_graph.json",
        "docs/architecture/semantic_addressed_world_model_inventory/capability_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/rollout_baseline.json",
        "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_quack_protocol.py",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
        "test/api/semantic_world/README.md",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/README.md",
    }
)

INTERFACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py", ("DatabaseTaskSource", "materialize", "snapshot", "get_task", "list_tasks", "ready_tasks", "compare_and_set_status", "record_evidence", "record_validation_result", "projection_matches_events")),
    ("ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py", ("install_datasets_authoritative_operational_schema", "verify_datasets_authoritative_operational_schema")),
    ("ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py", ("load_configured_board", "preflight_configured_board", "configured_board_launch_plan", "main")),
    ("ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py", ("DatabaseProgramConfig",)),
    ("ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py", ("QuackStateServer", "build_server")),
    ("ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py", ("discover_live_quack_endpoint", "DuckDBConnection")),
    ("ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py", ("build_mutation_request", "validate_mutation_request", "execute_mutation_bundle", "execute_owner_mutation", "service_mutation_inbox")),
    ("ipfs_accelerate_py/llm_router.py", ("probe_grok_codex_agent_route_readiness",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py", ("WorktreeLifecycleStore",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py", ("LeaseCoordinator",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py", ("CheckoutMutationLease",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py", ("MergeQueue",)),
    ("ipfs_accelerate_py/agent_supervisor/merge/merge_train.py", ("MergeTrain",)),
    ("ipfs_accelerate_py/agent_supervisor/integrations/ducklake_history_projection.py", ("project_history",)),
    ("ipfs_accelerate_py/agent_supervisor/context/context_compiler.py", ("ContextCompiler",)),
    ("ipfs_accelerate_py/agent_supervisor/context/decision_runtime.py", ("DecisionRuntime",)),
    ("ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py", ("SemanticCompressionHarness",)),
    ("ipfs_accelerate_py/agent_supervisor/semantic_governor/governor.py", ("SemanticCompressionGovernor",)),
    ("ipfs_accelerate_py/agent_supervisor/verification/planner.py", ("IncrementalVerificationPlanner",)),
    ("ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/sealer.py", ("IncrementalProofSealer",)),
)


def _duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _run(root: Path, *args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(args), cwd=cwd or root, check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20,
        env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
    )
    if completed.returncode:
        raise RuntimeError(f"{' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _git(root: Path, *args: str, cwd: Path | None = None) -> str:
    return _run(root, "git", *args, cwd=cwd)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_version(path: Path) -> str:
    with path.open("rb") as handle:
        project = tomllib.load(handle).get("project") or {}
    return str(project.get("version") or "")


def _status_paths(root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
        env={**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"},
    )
    if completed.returncode:
        raise RuntimeError(f"git status failed: {completed.stderr.strip()}")
    lines = completed.stdout.splitlines()
    result: list[str] = []
    for line in lines:
        text = line[3:] if len(line) >= 4 else ""
        if " -> " in text:
            text = text.split(" -> ", 1)[1]
        result.append(text.strip('"'))
    return tuple(result)


def _cold_import(root: Path, python: str, fixed: Mapping[str, Any]) -> dict[str, Any]:
    probe = r'''
import json, os, pathlib, socket, subprocess, sys, threading
sys.path[:0] = sys.argv[1:]
sys.path[:0] = json.loads(os.environ["SAWM_SOURCE_PATHS_JSON"])
before_env = dict(os.environ); before_threads = {t.ident for t in threading.enumerate()}
def denied(*a, **k): raise RuntimeError("cold_import_side_effect_denied")
class DeniedPopen(subprocess.Popen):
    def __init__(self, *a, **k): denied(*a, **k)
class DeniedSocket(socket.socket):
    def __init__(self, *a, **k): denied(*a, **k)
subprocess.Popen = DeniedPopen
for name in ("run", "call", "check_call", "check_output"):
    setattr(subprocess, name, denied)
socket.socket = DeniedSocket
try:
    import duckdb
    duckdb.connect = denied
except ImportError:
    pass
mods = [
 "ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py",
 "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source",
 "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
 "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
 "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
]
errors=[]
for mod in mods:
    try: __import__(mod)
    except Exception as exc: errors.append({"module":mod,"error":type(exc).__name__+": "+str(exc)})
after_threads={t.ident for t in threading.enumerate()}
changed={k:[before_env.get(k),v] for k,v in os.environ.items() if before_env.get(k)!=v}
removed=sorted(set(before_env)-set(os.environ))
print(json.dumps({"valid":not errors and not changed and not removed and after_threads==before_threads,
 "errors":errors,"environment_changes":changed,"environment_removed":removed,
 "new_thread_count":len(after_threads-before_threads)},sort_keys=True))
'''
    env = {str(key): str(value) for key, value in fixed.items()}
    env.update(
        {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": os.pathsep.join((str(root / "ipfs_datasets_py"), str(root / "ipfs_kit_py"), str(root))),
            "SAWM_SOURCE_PATHS_JSON": json.dumps(
                [str(root / "ipfs_datasets_py"), str(root / "ipfs_kit_py"), str(root)]
            ),
        }
    )
    completed = subprocess.run(
        [python, "-I", "-c", probe, str(root / "ipfs_datasets_py"),
         str(root / "ipfs_kit_py"), str(root)], cwd=root, env=env, check=False,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=45,
    )
    # -I ignores PYTHONPATH; seed paths explicitly without enabling user site.
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"valid": False, "returncode": completed.returncode, "stderr": completed.stderr[-2000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"valid": False, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}


def validate_dependencies(repo_root: Path | str = REPO_ROOT, *, cold_import: bool = True) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            errors.append(f"{name}: {detail}")

    try:
        seal = _load(root / SEAL_PATH.relative_to(REPO_ROOT))
    except Exception as exc:
        return {"schema": SCHEMA, "valid": False, "board_namespace": NAMESPACE,
                "plan_revision": REVISION, "errors": [f"seal_load: {type(exc).__name__}: {exc}"], "checks": []}

    check("seal_identity", seal.get("schema") == "semantic-addressed-world-model/dependency-seal@1" and seal.get("board_namespace") == NAMESPACE and seal.get("plan_revision") == REVISION and seal.get("status") == "sealed", seal.get("schema"))
    authorities = seal.get("source_authorities") if isinstance(seal.get("source_authorities"), list) else []
    by_package = {str(item.get("package")): item for item in authorities if isinstance(item, Mapping)}
    accel = by_package.get("ipfs_accelerate_py", {})
    try:
        head = _git(root, "rev-parse", "HEAD")
        branch = _git(root, "branch", "--show-current")
        ancestor_ok = subprocess.run(["git", "merge-base", "--is-ancestor", str(accel.get("head")), "HEAD"], cwd=root, check=False).returncode == 0
        origin = _git(root, "remote", "get-url", "origin")
        changed = set(_git(root, "diff", "--name-only", str(accel.get("head")), "--").splitlines())
        changed.update(_status_paths(root))
        unexpected = sorted(path for path in changed if path and path not in CONTROL_PATHS)
        check("accelerator_source", branch == accel.get("branch") and origin == accel.get("origin") and ancestor_ok and not unexpected,
              {"head": head, "branch": branch, "origin": origin, "unexpected_changes": unexpected})
    except Exception as exc:
        check("accelerator_source", False, f"{type(exc).__name__}: {exc}")

    gitlink_errors: list[str] = []
    for package in ("ipfs_datasets_py", "ipfs_kit_py"):
        authority = by_package.get(package, {})
        nested = root / package
        try:
            index_oid = _git(root, "rev-parse", f"HEAD:{package}")
            nested_head = _git(root, "rev-parse", "HEAD", cwd=nested)
            nested_tree = _git(root, "rev-parse", "HEAD^{tree}", cwd=nested)
            nested_status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", cwd=nested)
            version = _project_version(nested / "pyproject.toml")
            expected_origin = str(authority.get("origin") or "")
            actual_origin = _git(root, "remote", "get-url", "origin", cwd=nested)
            if not (
                index_oid == authority.get("gitlink_commit") == nested_head
                and nested_tree == authority.get("tree")
                and version == authority.get("package_version")
                and not nested_status
                and actual_origin == expected_origin
            ):
                gitlink_errors.append(f"{package}: index={index_oid} head={nested_head} tree={nested_tree} version={version} status={bool(nested_status)} origin={actual_origin}")
        except Exception as exc:
            gitlink_errors.append(f"{package}: {type(exc).__name__}: {exc}")
    check("gitlinks_and_nested_authorities", not gitlink_errors, gitlink_errors)

    dependency_errors: list[str] = []
    for item in seal.get("dependency_files") or ():
        if not isinstance(item, Mapping):
            dependency_errors.append("non-object dependency entry")
            continue
        path = root / str(item.get("path") or "")
        try:
            observed_sha = _sha(path)
            observed_blob = _git(root, "hash-object", str(path))
            if observed_sha != item.get("sha256") or observed_blob != item.get("git_blob_oid"):
                dependency_errors.append(f"{item.get('path')}: sha256={observed_sha} blob={observed_blob}")
        except Exception as exc:
            dependency_errors.append(f"{item.get('path')}: {type(exc).__name__}: {exc}")
    check("dependency_file_hashes", not dependency_errors, dependency_errors)

    toolchain = seal.get("toolchain") if isinstance(seal.get("toolchain"), Mapping) else {}
    versions = {}
    for distribution in ("pytest", "duckdb"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    # Bind the reviewed invocation path as well as the bytes it resolves to.
    # ``Path.resolve()`` would erase the sealed /home/barberb/.local/bin/python
    # entry-point identity because it is intentionally a symlink to /usr/bin.
    interpreter = Path(sys.executable).absolute()
    observed_toolchain = {
        "python_executable": str(interpreter), "python_sha256": _sha(interpreter),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "pytest_distribution_version": versions["pytest"],
        "duckdb_distribution_version": versions["duckdb"],
        "operating_system": platform.system(), "machine": platform.machine(),
    }
    expected_toolchain = {key: toolchain.get(key) for key in observed_toolchain}
    check("toolchain", observed_toolchain == expected_toolchain,
          {"expected": expected_toolchain, "observed": observed_toolchain})

    interface_errors: list[str] = []
    for relative, names in INTERFACES:
        try:
            tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
            observed = {node.name for node in ast.walk(tree) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            missing = sorted(set(names) - observed)
            if missing:
                interface_errors.append(f"{relative}: missing {missing}")
        except Exception as exc:
            interface_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
    check("landed_interfaces", not interface_errors, interface_errors)

    policy = seal.get("environment_policy") if isinstance(seal.get("environment_policy"), Mapping) else {}
    fixed = policy.get("fixed") if isinstance(policy.get("fixed"), Mapping) else {}
    policy_ok = (
        policy.get("secrets_source") == "environment_only"
        and policy.get("ordinary_test_network") is False
        and policy.get("cold_import_side_effects_allowed") is False
        and policy.get("arbitrary_code_deserialization_allowed") is False
        and all(str(fixed.get(key)) == value for key, value in {
            "IPFS_DATASETS_AUTO_INSTALL": "0", "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
            "IPFS_KIT_AUTO_INSTALL_DEPS": "0", "PYTHONNOUSERSITE": "1",
        }.items())
    )
    check("offline_environment_policy", policy_ok, policy)

    direction = seal.get("package_dependency_direction") if isinstance(seal.get("package_dependency_direction"), Mapping) else {}
    check("authority_dependency_direction", direction.get("accelerate_consumes_datasets_semantics") is True and direction.get("accelerate_consumes_kit_storage") is True and direction.get("datasets_may_depend_on_accelerate_operations") is False and direction.get("kit_may_decide_datasets_semantics") is False and direction.get("parallel_authority_implementation_allowed") is False, direction)
    plane = seal.get("control_plane") if isinstance(seal.get("control_plane"), Mapping) else {}
    check(
        "duckdb_quack_ducklake_boundaries",
        plane.get("authoritative_store") == "DuckDB"
        and plane.get("exclusive_mutation_owner") == "canonical DuckDB writer"
        and plane.get("multi_process_query_transport") == "Quack read-only replica"
        and plane.get("authenticated_mutation_protocol") == "closed atomic owner-inbox protocol@2"
        and plane.get("canonical_writer_served_through_quack") is False
        and plane.get("quack_replica_authoritative") is False
        and plane.get("owner_protocol_allows_arbitrary_sql") is False
        and plane.get("ducklake_authoritative") is False
        and plane.get("markdown_authoritative") is False
        and plane.get("worker_self_completion_allowed") is False,
        plane,
    )

    protocol_errors: list[str] = []
    try:
        scheduler = _load(
            root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        )
        migration = _load(
            root
            / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
        )
        configured_pin = scheduler.get("quack_owner", {}).get("pinned_extension", {})
        sealed_pin = seal.get("quack_extension_pin", {})
        pin_path = Path(str(configured_pin.get("path") or ""))
        if configured_pin != sealed_pin:
            protocol_errors.append("scheduler Quack pin differs from the dependency seal")
        if not pin_path.is_file() or _sha(pin_path) != configured_pin.get("sha256"):
            protocol_errors.append("reviewed local Quack extension bytes are unavailable or mismatched")
        if configured_pin.get("network_install_allowed") is not False:
            protocol_errors.append("Quack network installation must remain forbidden")
        if configured_pin.get("unsigned_extension_allowed") is not False:
            protocol_errors.append("unsigned Quack extensions must remain forbidden")

        sealed_migration = seal.get("source_migration", {})
        if (
            sealed_migration.get("migration_revision") != migration.get("migration_revision")
            or sealed_migration.get("prior_store_id") != migration.get("prior_store_id")
            or sealed_migration.get("target_store_id") != migration.get("target_store_id")
            or sealed_migration.get("prior_event_watermark") != migration.get("prior_event_watermark")
            or sealed_migration.get("prior_event_prefix_sha256") != migration.get("prior_event_prefix_sha256")
            or sealed_migration.get("prior_control_store_sha256") != migration.get("prior_control_store_sha256")
            or sealed_migration.get("accepted_definition_rewrite_allowed") is not False
            or sealed_migration.get("accepted_completion_replay_allowed") is not False
            or sealed_migration.get("prior_authority_preserved") is not True
        ):
            protocol_errors.append("append-only source-migration seal differs from its inventory")
        if scheduler.get("database_program", {}).get("store_generation") != "2":
            protocol_errors.append("migrated owner must acquire successor store generation 2")

        protocol_source = (
            root / "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py"
        ).read_text(encoding="utf-8")
        operator_source = (
            root / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
        ).read_text(encoding="utf-8")
        runner_source = (
            root / "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py"
        ).read_text(encoding="utf-8")
        daemon_source = (
            root / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
        ).read_text(encoding="utf-8")
        if "quack-owner-mutation-request@2" not in protocol_source:
            protocol_errors.append("closed mutation protocol revision 2 is absent")
        if "MUTATION_SQL_TEMPLATES" not in protocol_source or "execute_owner_mutation" not in protocol_source:
            protocol_errors.append("closed atomic mutation catalog is absent")
        if "read_only=True" not in operator_source or "canonical writer without loading or serving Quack" not in operator_source:
            protocol_errors.append("read-only Quack replica / sealed writer boundary is absent")
        for name in (
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_BINDING",
        ):
            if name not in runner_source:
                protocol_errors.append(f"state credential scrub list omits {name}")
        if daemon_source.count("inherit_environment=False") < 3:
            protocol_errors.append("provider subprocesses do not all use exact scrubbed environments")
    except Exception as exc:
        protocol_errors.append(f"{type(exc).__name__}: {exc}")
    check("append_only_migration_and_closed_quack_protocol", not protocol_errors, protocol_errors)

    if cold_import and not errors:
        # Use the sealed interpreter, not whichever Python happened to invoke a
        # caller.  The probe prepends explicit source roots because -I ignores
        # PYTHONPATH by design.
        python = str(toolchain.get("python_executable") or sys.executable)
        result = _cold_import(root, python, fixed)
        check("cold_import_side_effects", result.get("valid") is True, result)
    else:
        checks.append({"name": "cold_import_side_effects", "passed": None, "detail": "skipped after prior failure or by request"})

    return {"schema": SCHEMA, "valid": not errors, "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "errors": errors, "checks": checks,
            "database_opened": False, "network_required": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--skip-cold-import", action="store_true", help="diagnostic only; not a launch gate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_dependencies(args.repo_root, cold_import=not args.skip_cold_import)
    except Exception as exc:
        report = {"schema": SCHEMA, "valid": False, "board_namespace": NAMESPACE,
                  "plan_revision": REVISION, "errors": [f"unhandled_validator_error: {type(exc).__name__}: {exc}"], "checks": []}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
