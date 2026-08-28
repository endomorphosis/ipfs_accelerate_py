#!/usr/bin/env python3
"""Verify the exact, offline SAWM R2 source and dependency seal.

This gate is deliberately read-only.  It does not import optional providers in
the validator process.  A separate, hermetic child recomputes the reviewed
native DuckDB pin, consumes those bytes through a sealed anonymous descriptor,
loads the exact local HTTPFS and Quack projections, and opens only an in-memory
database for ``SELECT 42``.  Network and extension installation remain denied.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SEAL_PATH = REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
NAMESPACE = "semantic-addressed-world-model-v1"
REVISION = "SAWM-PLAN-R2"
SCHEMA = "semantic-addressed-world-model/dependency-seal-validation@1"
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_NATIVE_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "board_namespace",
        "plan_revision",
        "status",
        "scope",
        "dependency_id",
        "payload_sha256",
        "python_executable_sha256",
        "authority_basis",
        "inspection_is_authority",
        "authorization_may_claim_task_completion",
        "authorization_id",
    }
)

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
        "config/semantic_addressed_world_model_native_dependency.authorization.json",
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "ipfs_accelerate_py/agent_implementation_route.py",
        "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_extension_projection.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_live_capsule.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_attestation.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_quack_protocol.py",
        "test/api/test_agent_supervisor_configured_board_extension_projection.py",
        "test/api/test_agent_supervisor_configured_board_live_capsule.py",
        "test/api/test_agent_supervisor_configured_board_scheduler.py",
        "test/api/test_agent_supervisor_native_dependency_pin.py",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
        "test/api/semantic_world/README.md",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/README.md",
    }
)

INTERFACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py", ("DatabaseTaskSource", "materialize", "snapshot", "get_task", "list_tasks", "ready_tasks", "compare_and_set_status", "record_evidence", "record_validation_result", "projection_matches_events")),
    ("ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py", ("install_datasets_authoritative_operational_schema", "verify_datasets_authoritative_operational_schema")),
    ("ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py", ("load_configured_board", "preflight_configured_board", "configured_board_launch_plan", "main")),
    ("ipfs_accelerate_py/agent_supervisor/runtime/configured_board_live_capsule.py", ("ConfiguredBoardLiveCapsuleAdmission", "parse_configured_board_live_capsule_policy", "parse_configured_board_live_capsule_admission", "build_configured_board_live_capsule_admission", "verify_configured_board_live_capsule")),
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
        capture_output=True, text=True, timeout=20,
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


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _stable_regular_evidence(path: Path, *, maximum: int) -> dict[str, Any]:
    """Hash one no-follow regular file and reject identity changes mid-read."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ValueError(f"{path} is not stable regular evidence")
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_gid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    if offset != before.st_size or identity(before) != identity(after):
        raise ValueError(f"{path} changed while it was hashed")
    return {
        "sha256": "sha256:" + digest.hexdigest(),
        "size": before.st_size,
    }


def _copy_pinned_regular(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    """Copy one exact no-follow source into an exclusive qualification path."""

    if _SHA256.fullmatch(expected_sha256) is None or expected_size <= 0:
        raise ValueError("extension projection pin is invalid")
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    target_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = os.open(source, source_flags)
    target_fd = -1
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise ValueError(f"{source} is not the pinned regular source")
        target_fd = os.open(destination, target_flags, 0o600)
        digest = hashlib.sha256()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                source_fd,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(target_fd, view)
                if written <= 0:
                    raise ValueError("extension projection write failed")
                view = view[written:]
            offset += len(block)
        after = os.fstat(source_fd)
        if (
            offset != expected_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_gid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or "sha256:" + digest.hexdigest() != expected_sha256
        ):
            raise ValueError(f"{source} differs from its pinned bytes")
        os.fsync(target_fd)
        os.fchmod(target_fd, 0o400)
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(source_fd)


def _restore_and_remove_private_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, files in os.walk(root, topdown=False):
        current_path = Path(current)
        for name in files:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o600)
        for name in directories:
            candidate = current_path / name
            if not candidate.is_symlink():
                os.chmod(candidate, 0o700)
        os.chmod(current_path, 0o700)
    shutil.rmtree(root)


def _positive_launch_environment(
    fixed: Mapping[str, Any],
    *,
    home: Path,
) -> dict[str, str]:
    env = {str(key): str(value) for key, value in fixed.items()}
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "PYTHONUSERBASE": str(home / ".python-user-base"),
            "XDG_CACHE_HOME": str(home / ".cache"),
        }
    )
    if any(name.startswith("LD_") for name in env) or "PYTHONPATH" in env:
        raise ValueError("isolated launch environment contains an ambient loader path")
    return env


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
        capture_output=True,
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


def _validate_native_authorization(
    root: Path,
    seal: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    toolchain: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Authenticate authority independently of inspecting the native bytes."""

    errors: list[str] = []
    native_value = seal.get("configured_board_native_dependency")
    if type(native_value) is not dict or set(native_value) != {
        "schema",
        "source_path",
        "acceptance",
        "pin",
        "sealed_memfd_required",
        "ambient_site_import_allowed",
        "ambient_loader_environment_allowed",
    }:
        return {}, {}, ["configured native dependency seal is noncanonical"]
    native = dict(native_value)
    if (
        native.get("schema")
        != "semantic-addressed-world-model/configured-board-native-dependency@1"
        or native.get("sealed_memfd_required") is not True
        or native.get("ambient_site_import_allowed") is not False
        or native.get("ambient_loader_environment_allowed") is not False
    ):
        errors.append("configured native dependency policy is invalid")
    source = Path(str(native.get("source_path") or ""))
    if not source.is_absolute():
        errors.append("configured native dependency source is not absolute")

    pin = native.get("pin")
    reference = native.get("acceptance")
    if type(pin) is not dict:
        errors.append("configured native dependency pin is not an object")
        pin = {}
    if type(reference) is not dict or set(reference) != {
        "schema",
        "path",
        "sha256",
        "size",
        "authorization_id",
    }:
        errors.append("native authorization reference is noncanonical")
        return native, {}, errors
    if reference.get("schema") != (
        "semantic-addressed-world-model/"
        "native-dependency-authorization-reference@1"
    ):
        errors.append("native authorization reference schema is invalid")

    relative = Path(str(reference.get("path") or ""))
    protected = set(scheduler.get("protected_paths") or ())
    capsule = scheduler.get("configured_board_live_capsule")
    capsule_paths = set(capsule.get("control_paths") or ()) if type(capsule) is dict else set()
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() not in CONTROL_PATHS
        or relative.as_posix() not in protected
        or relative.as_posix() not in capsule_paths
    ):
        errors.append("native authorization artifact is not a protected control path")
        return native, {}, errors

    authorization_path = root / relative
    try:
        evidence = _stable_regular_evidence(authorization_path, maximum=65_536)
        authorization = _load(authorization_path)
    except Exception as exc:
        errors.append(f"native authorization artifact is unavailable: {type(exc).__name__}: {exc}")
        return native, {}, errors
    if (
        evidence.get("sha256") != reference.get("sha256")
        or evidence.get("size") != reference.get("size")
    ):
        errors.append("native authorization artifact bytes differ from the accepted reference")
    if set(authorization) != _NATIVE_AUTHORIZATION_FIELDS:
        errors.append("native authorization artifact fields are noncanonical")
        return native, authorization, errors

    unsigned = dict(authorization)
    authorization_id = unsigned.pop("authorization_id", None)
    expected_id = "sha256:" + hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if (
        authorization_id != expected_id
        or authorization_id != reference.get("authorization_id")
        or authorization.get("schema") != (
            "semantic-addressed-world-model/"
            "native-dependency-launch-authorization@1"
        )
        or authorization.get("board_namespace") != NAMESPACE
        or authorization.get("plan_revision") != REVISION
        or authorization.get("status") != "accepted"
        or authorization.get("scope") != "configured-board-live-control-plane"
        or authorization.get("dependency_id") != pin.get("dependency_id")
        or authorization.get("payload_sha256") != pin.get("payload_sha256")
        or authorization.get("python_executable_sha256")
        != pin.get("python_executable_sha256")
        or authorization.get("authority_basis")
        != (
            "operator-owned protected control inside the accepted immutable "
            "source capsule"
        )
        or authorization.get("inspection_is_authority") is not False
        or authorization.get("authorization_may_claim_task_completion") is not False
    ):
        errors.append("native dependency authorization was not independently admitted")
    expected_python_digest = "sha256:" + str(toolchain.get("launch_python_sha256") or "")
    if (
        pin.get("python_executable_sha256") != expected_python_digest
        or pin.get("distribution_version")
        != toolchain.get("duckdb_distribution_version")
    ):
        errors.append("native dependency pin differs from the launch toolchain declaration")
    return native, authorization, errors


def _validate_extension_projection_pins(
    seal: Mapping[str, Any],
    scheduler: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Rehash Quack and HTTPFS payload/metadata and bind scheduler projections."""

    errors: list[str] = []
    projections: dict[str, dict[str, Any]] = {}
    owner = scheduler.get("quack_owner")
    if type(owner) is not dict:
        return {}, ["scheduler quack_owner is absent"]

    configured = seal.get("configured_board_quack_projection")
    configured_pin: dict[str, Any] = {}
    if type(configured) is not dict or set(configured) != {
        "schema",
        "source_path",
        "info_path",
        "pin",
        "load_policy",
        "network_install_allowed",
        "unsigned_extension_allowed",
    }:
        errors.append("configured-board Quack projection is noncanonical")
    else:
        pin_value = configured.get("pin")
        if type(pin_value) is not dict or set(pin_value) != {
            "schema",
            "name",
            "engine_version",
            "platform",
            "payload_sha256",
            "payload_size",
            "info_sha256",
            "info_size",
            "projection_id",
        }:
            errors.append("configured-board Quack projection pin is noncanonical")
        else:
            configured_pin = dict(pin_value)
            identity_body = dict(configured_pin)
            observed_id = identity_body.pop("projection_id", None)
            expected_id = "sha256:" + hashlib.sha256(
                _canonical_json(identity_body)
            ).hexdigest()
            if (
                configured.get("schema")
                != "semantic-addressed-world-model/configured-board-quack-projection@1"
                or configured.get("load_policy") != "local_load_only"
                or configured.get("network_install_allowed") is not False
                or configured.get("unsigned_extension_allowed") is not False
                or configured_pin.get("schema") != (
                    "ipfs_accelerate_py.agent_supervisor."
                    "configured-board-duckdb-extension-pin@1"
                )
                or configured_pin.get("name") != "quack"
                or observed_id != expected_id
            ):
                errors.append("configured-board Quack projection identity is invalid")

    engine_version = str(configured_pin.get("engine_version") or "")
    platform_name = str(configured_pin.get("platform") or "")
    if not engine_version.startswith("v") or not platform_name:
        errors.append("extension projection engine/platform binding is invalid")

    for name, seal_key, scheduler_key in (
        ("quack", "quack_extension_pin", "pinned_extension"),
        ("httpfs", "httpfs_extension_pin", "pinned_httpfs_extension"),
    ):
        value = seal.get(seal_key)
        scheduled = owner.get(scheduler_key)
        required = {
            "path",
            "info_path",
            "version",
            "sha256",
            "size",
            "info_sha256",
            "info_size",
            "network_install_allowed",
            "unsigned_extension_allowed",
        }
        allowed = required | ({"service_external_access_limitation"} if name == "quack" else set())
        if type(value) is not dict or set(value) != allowed:
            errors.append(f"{name} extension source pin is noncanonical")
            continue
        if scheduled != value:
            errors.append(f"scheduler {name} extension pin differs from the dependency seal")
        path = Path(str(value.get("path") or ""))
        info_path = Path(str(value.get("info_path") or ""))
        size = value.get("size")
        info_size = value.get("info_size")
        if (
            not path.is_absolute()
            or not info_path.is_absolute()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or isinstance(info_size, bool)
            or not isinstance(info_size, int)
            or value.get("network_install_allowed") is not False
            or value.get("unsigned_extension_allowed") is not False
        ):
            errors.append(f"{name} extension source pin policy is invalid")
            continue
        try:
            payload_evidence = _stable_regular_evidence(path, maximum=64 * 1024 * 1024)
            info_evidence = _stable_regular_evidence(info_path, maximum=64 * 1024)
        except Exception as exc:
            errors.append(f"{name} extension source is unavailable: {type(exc).__name__}: {exc}")
            continue
        payload_sha256 = "sha256:" + str(value.get("sha256") or "")
        metadata_sha256 = "sha256:" + str(value.get("info_sha256") or "")
        if (
            payload_evidence != {"sha256": payload_sha256, "size": size}
            or info_evidence != {"sha256": metadata_sha256, "size": info_size}
        ):
            errors.append(f"{name} extension payload or metadata bytes differ")
        body: dict[str, Any] = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "configured-board-duckdb-extension-pin@1"
            ),
            "name": name,
            "engine_version": engine_version,
            "platform": platform_name,
            "payload_sha256": payload_sha256,
            "payload_size": size,
            "info_sha256": metadata_sha256,
            "info_size": info_size,
        }
        body["projection_id"] = "sha256:" + hashlib.sha256(
            _canonical_json(body)
        ).hexdigest()
        projections[name] = {
            **body,
            "source_path": str(path),
            "info_path": str(info_path),
            "version": str(value.get("version") or ""),
        }

    quack = projections.get("quack")
    if quack and configured_pin:
        quack_pin = {key: item for key, item in quack.items() if key not in {"source_path", "info_path", "version"}}
        if (
            configured.get("source_path") != quack.get("source_path")
            or configured.get("info_path") != quack.get("info_path")
            or configured_pin != quack_pin
        ):
            errors.append("configured-board Quack projection differs from exact source pins")
    return projections, errors


def _isolated_native_extension_probe(
    root: Path,
    *,
    launch_python: Path,
    expected_python_sha256: str,
    fixed: Mapping[str, Any],
    native: Mapping[str, Any],
    authorization: Mapping[str, Any],
    projections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the launch stack under exact CPython flags and a positive env."""

    probe_root = Path(tempfile.mkdtemp(prefix="sawm-dependency-probe-", dir="/tmp"))
    try:
        os.chmod(probe_root, 0o700)
        home = probe_root / "home"
        home.mkdir(mode=0o700)
        (home / ".python-user-base").mkdir(mode=0o700)
        (home / ".cache").mkdir(mode=0o700)
        quack = projections["quack"]
        extension_directory = (
            home
            / ".duckdb/extensions"
            / str(quack["engine_version"])
            / str(quack["platform"])
        )
        extension_directory.mkdir(parents=True, mode=0o700)
        expected_install_paths: dict[str, str] = {}
        for name in ("httpfs", "quack"):
            projection = projections[name]
            extension = extension_directory / f"{name}.duckdb_extension"
            metadata = extension_directory / f"{name}.duckdb_extension.info"
            _copy_pinned_regular(
                Path(str(projection["source_path"])),
                extension,
                expected_sha256=str(projection["payload_sha256"]),
                expected_size=int(projection["payload_size"]),
            )
            _copy_pinned_regular(
                Path(str(projection["info_path"])),
                metadata,
                expected_sha256=str(projection["info_sha256"]),
                expected_size=int(projection["info_size"]),
            )
            expected_install_paths[name] = str(extension)
        for directory in sorted(
            (item for item in home.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory != home / ".cache":
                os.chmod(directory, 0o500)
        os.chmod(home, 0o500)

        payload = {
            "native": native,
            "authorization_id": authorization.get("authorization_id"),
            "launch_python_sha256": expected_python_sha256,
            "expected_install_paths": expected_install_paths,
            "extension_versions": {
                name: str(projections[name]["version"])
                for name in ("httpfs", "quack")
            },
            "fixed_environment": {str(key): str(value) for key, value in fixed.items()},
            "home": str(home),
        }
        probe = r'''
import hashlib, importlib.util, json, os, pathlib, platform, sys

payload = json.load(sys.stdin)
cmdline = pathlib.Path("/proc/self/cmdline").read_bytes().split(b"\0")
ambient_specs = {
    name: importlib.util.find_spec(name) is not None
    for name in ("duckdb", "_duckdb", "pytest")
}
stdlib_path = list(sys.path)
if (
    sys.flags.isolated != 1
    or sys.flags.no_site != 1
    or sys.flags.dont_write_bytecode != 1
    or sys.flags.no_user_site != 1
    or not all(flag in cmdline for flag in (b"-I", b"-S", b"-B"))
    or any(ambient_specs.values())
    or any("site-packages" in item or "dist-packages" in item for item in stdlib_path)
    or "site" in sys.modules
    or "PYTHONPATH" in os.environ
    or any(name.startswith("LD_") for name in os.environ)
):
    raise RuntimeError("isolated launch interpreter admitted ambient packages")
for name, value in payload["fixed_environment"].items():
    if os.environ.get(name) != value:
        raise RuntimeError("isolated launch fixed environment drifted")
if os.environ.get("HOME") != payload["home"]:
    raise RuntimeError("isolated launch HOME differs from qualification projection")

executable_sha256 = hashlib.sha256(pathlib.Path("/proc/self/exe").read_bytes()).hexdigest()
if (
    sys.executable != "/usr/bin/python3.12"
    or executable_sha256 != payload["launch_python_sha256"]
):
    raise RuntimeError("isolated launch interpreter bytes differ from the seal")

root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from ipfs_accelerate_py.agent_implementation_route import (
    inspect_agent_supervisor_native_dependency_source,
    parse_agent_supervisor_native_dependency_pin,
    preload_agent_supervisor_native_dependency_from_bootstrap,
    seal_agent_supervisor_native_dependency,
)

native = payload["native"]
expected_pin = parse_agent_supervisor_native_dependency_pin(native["pin"])
observed_pin = inspect_agent_supervisor_native_dependency_source(
    native["source_path"],
    distribution_version=expected_pin.distribution_version,
    engine_version=expected_pin.engine_version,
)
if observed_pin != expected_pin:
    raise RuntimeError("native dependency source differs from its accepted pin")
launch = seal_agent_supervisor_native_dependency(
    native["source_path"],
    expected_pin=expected_pin,
    accepted_authorization_id=payload["authorization_id"],
)
try:
    duckdb = preload_agent_supervisor_native_dependency_from_bootstrap(
        *launch.bootstrap_arguments
    )
    expected_origin = f"/proc/self/fd/{launch.descriptor.descriptor}"
    if getattr(duckdb, "__file__", None) != expected_origin:
        raise RuntimeError("DuckDB was not loaded from the sealed descriptor")
    connection = duckdb.connect(
        ":memory:",
        config={
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "allow_unsigned_extensions": "false",
            "enable_external_access": "true",
            "extension_directory": str(
                pathlib.Path(payload["home"]) / ".duckdb/extensions"
            ),
        },
    )
    try:
        load_settings = connection.execute(
            "SELECT current_setting('enable_external_access'), "
            "current_setting('autoinstall_known_extensions'), "
            "current_setting('autoload_known_extensions'), "
            "current_setting('allow_unsigned_extensions')"
        ).fetchone()
        if list(load_settings or ()) != [True, False, False, False]:
            raise RuntimeError("local extension load window policy drifted")
        connection.execute("LOAD httpfs")
        connection.execute("LOAD quack")
        rows = connection.execute(
            "SELECT extension_name, extension_version, loaded, installed, install_path "
            "FROM duckdb_extensions() "
            "WHERE extension_name IN ('httpfs', 'quack') ORDER BY extension_name"
        ).fetchall()
        connection.execute("SET enable_external_access=false")
        answer = connection.execute("SELECT 42").fetchone()
        settings = connection.execute(
            "SELECT current_setting('enable_external_access'), "
            "current_setting('autoinstall_known_extensions'), "
            "current_setting('autoload_known_extensions'), "
            "current_setting('allow_unsigned_extensions')"
        ).fetchone()
    finally:
        connection.close()
finally:
    os.close(launch.descriptor.descriptor)

expected_rows = [
    [
        name,
        payload["extension_versions"][name],
        True,
        True,
        payload["expected_install_paths"][name],
    ]
    for name in ("httpfs", "quack")
]
if list(answer or ()) != [42] or [list(row) for row in rows] != expected_rows:
    raise RuntimeError("isolated DuckDB extension probe returned unexpected evidence")
if list(settings or ()) != [False, False, False, False]:
    raise RuntimeError("isolated DuckDB extension/network policy drifted")
print(json.dumps({
    "valid": True,
    "argv_flags": ["-I", "-S", "-B"],
    "ambient_specs": ambient_specs,
    "stdlib_path": stdlib_path,
    "python_executable": sys.executable,
    "python_executable_sha256": executable_sha256,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "operating_system": platform.system(),
    "machine": platform.machine(),
    "native_dependency_id": observed_pin.dependency_id,
    "native_payload_sha256": observed_pin.payload_sha256,
    "native_module_origin": expected_origin,
    "duckdb_distribution_version": str(duckdb.__version__),
    "extension_rows": [list(row) for row in rows],
    "settings": list(settings),
    "select_42": 42,
    "database_opened": True,
    "network_required": False,
}, sort_keys=True))
'''
        completed = subprocess.run(
            [str(launch_python), "-I", "-S", "-B", "-c", probe, str(root)],
            cwd=root,
            env=_positive_launch_environment(fixed, home=home),
            check=False,
            input=json.dumps(payload, sort_keys=True),
            capture_output=True,
            text=True,
            timeout=90,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {
                "valid": False,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-4000:],
            }
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "valid": False,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
    finally:
        _restore_and_remove_private_tree(probe_root)


def _cold_import(root: Path, python: str, fixed: Mapping[str, Any]) -> dict[str, Any]:
    probe = r'''
import importlib.util, json, os, socket, subprocess, sys, threading
ambient_specs={name:importlib.util.find_spec(name) is not None for name in ("duckdb","_duckdb","pytest")}
stdlib_path=list(sys.path)
if (sys.flags.isolated != 1 or sys.flags.no_site != 1
    or sys.flags.dont_write_bytecode != 1 or sys.flags.no_user_site != 1
    or any(ambient_specs.values())
    or any("site-packages" in item or "dist-packages" in item for item in stdlib_path)
    or "site" in sys.modules):
    raise RuntimeError("cold_import_ambient_package_path")
sys.path[:0] = json.loads(sys.stdin.read())
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
 "new_thread_count":len(after_threads-before_threads),"ambient_specs":ambient_specs,
 "stdlib_path":stdlib_path},sort_keys=True))
'''
    with tempfile.TemporaryDirectory(prefix="sawm-cold-import-", dir="/tmp") as directory:
        home = Path(directory)
        (home / ".python-user-base").mkdir(mode=0o700)
        (home / ".cache").mkdir(mode=0o700)
        completed = subprocess.run(
            [python, "-I", "-S", "-B", "-c", probe],
            cwd=root,
            env=_positive_launch_environment(fixed, home=home),
            check=False,
            input=json.dumps(
                [str(root / "ipfs_datasets_py"), str(root / "ipfs_kit_py"), str(root)]
            ),
            capture_output=True,
            text=True,
            timeout=45,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {
                "valid": False,
                "returncode": completed.returncode,
                "stderr": completed.stderr[-2000:],
            }
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "valid": False,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }


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

    policy = seal.get("environment_policy") if isinstance(seal.get("environment_policy"), Mapping) else {}
    fixed = policy.get("fixed") if isinstance(policy.get("fixed"), Mapping) else {}
    toolchain = seal.get("toolchain") if isinstance(seal.get("toolchain"), Mapping) else {}
    launch_python = Path(str(toolchain.get("launch_python_executable") or ""))
    legacy_python = Path(str(toolchain.get("python_executable") or ""))
    launch_toolchain_errors: list[str] = []
    try:
        launch_evidence = _stable_regular_evidence(launch_python, maximum=64 * 1024 * 1024)
    except Exception as exc:
        launch_evidence = {}
        launch_toolchain_errors.append(
            f"launch interpreter unavailable: {type(exc).__name__}: {exc}"
        )
    try:
        legacy_python_sha256 = _sha(legacy_python)
    except Exception as exc:
        legacy_python_sha256 = ""
        launch_toolchain_errors.append(
            f"reviewed interpreter entry point unavailable: {type(exc).__name__}: {exc}"
        )
    if (
        str(launch_python) != "/usr/bin/python3.12"
        or not launch_python.is_absolute()
        or launch_evidence.get("sha256")
        != "sha256:" + str(toolchain.get("launch_python_sha256") or "")
        or legacy_python_sha256 != toolchain.get("python_sha256")
        or toolchain.get("launch_python_sha256") != toolchain.get("python_sha256")
        or policy.get("python_invocation_flags") != ["-I", "-S", "-B"]
        or toolchain.get("implicit_install_allowed") is not False
        or toolchain.get("implicit_model_download_allowed") is not False
    ):
        launch_toolchain_errors.append("sealed launch interpreter declaration is invalid")
    check(
        "sealed_launch_toolchain_declaration",
        not launch_toolchain_errors,
        {
            "errors": launch_toolchain_errors,
            "launch_python_executable": str(launch_python),
            "launch_python_evidence": launch_evidence,
            "legacy_entry_point": str(legacy_python),
            "legacy_entry_point_sha256": legacy_python_sha256,
            "python_invocation_flags": policy.get("python_invocation_flags"),
            "pytest_distribution_version": {
                "declared": toolchain.get("pytest_distribution_version"),
                "authority": "test-only; excluded from launch admission",
            },
        },
    )

    try:
        scheduler = _load(
            root / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
        )
        scheduler_error = ""
    except Exception as exc:
        scheduler = {}
        scheduler_error = f"{type(exc).__name__}: {exc}"

    native, authorization, native_errors = _validate_native_authorization(
        root,
        seal,
        scheduler,
        toolchain,
    )
    if scheduler_error:
        native_errors.append(f"scheduler unavailable: {scheduler_error}")
    check(
        "independent_native_dependency_authorization",
        not native_errors,
        {
            "errors": native_errors,
            "authorization_id": authorization.get("authorization_id"),
            "inspection_is_authority": authorization.get("inspection_is_authority"),
        },
    )

    projections, projection_errors = _validate_extension_projection_pins(
        seal,
        scheduler,
    )
    if scheduler_error:
        projection_errors.append(f"scheduler unavailable: {scheduler_error}")
    check(
        "quack_httpfs_projection_pins",
        not projection_errors,
        {
            "errors": projection_errors,
            "projection_ids": {
                name: item.get("projection_id")
                for name, item in sorted(projections.items())
            },
        },
    )

    probe_ready = not launch_toolchain_errors and not native_errors and not projection_errors
    if probe_ready:
        try:
            dependency_probe = _isolated_native_extension_probe(
                root,
                launch_python=launch_python,
                expected_python_sha256=str(toolchain.get("launch_python_sha256") or ""),
                fixed=fixed,
                native=native,
                authorization=authorization,
                projections=projections,
            )
        except Exception as exc:
            dependency_probe = {
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        dependency_probe = {
            "valid": False,
            "skipped": "invalid toolchain, authorization, or projection prerequisite",
        }
    expected_isolated_toolchain = {
        "python_executable": str(launch_python),
        "python_executable_sha256": toolchain.get("launch_python_sha256"),
        "python_implementation": toolchain.get("python_implementation"),
        "python_version": toolchain.get("python_version"),
        "operating_system": toolchain.get("operating_system"),
        "machine": toolchain.get("machine"),
        "duckdb_distribution_version": toolchain.get("duckdb_distribution_version"),
    }
    observed_isolated_toolchain = {
        key: dependency_probe.get(key) for key in expected_isolated_toolchain
    }
    check(
        "isolated_launch_toolchain",
        dependency_probe.get("valid") is True
        and observed_isolated_toolchain == expected_isolated_toolchain
        and dependency_probe.get("argv_flags") == ["-I", "-S", "-B"]
        and dependency_probe.get("ambient_specs")
        == {"duckdb": False, "_duckdb": False, "pytest": False},
        {
            "expected": expected_isolated_toolchain,
            "observed": observed_isolated_toolchain,
            "argv_flags": dependency_probe.get("argv_flags"),
            "ambient_specs": dependency_probe.get("ambient_specs"),
            "failure": dependency_probe if dependency_probe.get("valid") is not True else None,
        },
    )
    native_pin = native.get("pin") if type(native.get("pin")) is dict else {}
    check(
        "recomputed_native_dependency_pin",
        dependency_probe.get("valid") is True
        and dependency_probe.get("native_dependency_id") == native_pin.get("dependency_id")
        and dependency_probe.get("native_payload_sha256") == native_pin.get("payload_sha256")
        and str(dependency_probe.get("native_module_origin") or "").startswith(
            "/proc/self/fd/"
        ),
        {
            "expected_dependency_id": native_pin.get("dependency_id"),
            "observed_dependency_id": dependency_probe.get("native_dependency_id"),
            "expected_payload_sha256": native_pin.get("payload_sha256"),
            "observed_payload_sha256": dependency_probe.get("native_payload_sha256"),
            "module_origin": dependency_probe.get("native_module_origin"),
        },
    )
    check(
        "isolated_duckdb_quack_httpfs_load",
        dependency_probe.get("valid") is True
        and dependency_probe.get("database_opened") is True
        and dependency_probe.get("network_required") is False
        and dependency_probe.get("select_42") == 42
        and dependency_probe.get("settings") == [False, False, False, False]
        and len(dependency_probe.get("extension_rows") or ()) == 2,
        dependency_probe,
    )

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

    policy_ok = (
        policy.get("secrets_source") == "environment_only"
        and policy.get("secrets_in_prompt_argv_log_receipt_board_corpus_model") is False
        and policy.get("ordinary_test_network") is False
        and policy.get("cold_import_side_effects_allowed") is False
        and policy.get("arbitrary_code_deserialization_allowed") is False
        and policy.get("python_invocation_flags") == ["-I", "-S", "-B"]
        and policy.get("duckdb_extension_home")
        == "exact_private_read_only_projection"
        and all(
            str(fixed.get(key)) == value
            for key, value in {
                "IPFS_DATASETS_AUTO_INSTALL": "0",
                "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS": "0",
                "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY": "disabled",
                "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
                "IPFS_KIT_AUTO_INSTALL_DEPS": "0",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }.items()
        )
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
            or sealed_migration.get("prior_migration_receipt_cid")
            != migration.get("prior_materialization_receipt_cid")
            or sealed_migration.get("prior_migration_count")
            != len(migration.get("migration_history") or ())
            or sealed_migration.get("target_plan_revision")
            != migration.get("target_plan_revision")
            or sealed_migration.get("accepted_definition_rewrite_allowed") is not False
            or sealed_migration.get("accepted_completion_replay_allowed") is not False
            or sealed_migration.get("prior_authority_preserved") is not True
        ):
            protocol_errors.append("append-only source-migration seal differs from its inventory")
        if scheduler.get("database_program", {}).get("store_generation") != "3":
            protocol_errors.append("migrated owner must acquire successor store generation 3")

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
        python = str(toolchain.get("launch_python_executable") or "")
        result = _cold_import(root, python, fixed)
        check("cold_import_side_effects", result.get("valid") is True, result)
    else:
        checks.append({"name": "cold_import_side_effects", "passed": None, "detail": "skipped after prior failure or by request"})

    return {"schema": SCHEMA, "valid": not errors, "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "errors": errors, "checks": checks,
            "database_opened": dependency_probe.get("database_opened") is True,
            "network_required": False}


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
