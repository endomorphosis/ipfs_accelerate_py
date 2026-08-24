#!/usr/bin/env python3
"""Materialize and verify the EAAEF embedded bootstrap control plane.

This script intentionally materializes only the reconciliation bootstrap
population declared by the reviewed board. Future tasks stay outside the
database until the board's terminal bootstrap task emits a current
semantic-root-bound Plan R2. The bootstrap uses one embedded DuckDB writer; it
neither enables continuous Quack operation nor DuckLake authority.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Direct ``python scripts/...py`` execution otherwise exposes only the
    # scripts directory, making the reviewed local package unimportable after
    # the immutable namespace claim has already been published.
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "config/external_agent_autonomous_execution_fabric_scheduler.json"
EAAEF_BOARD_PATH = (
    ROOT / "docs/architecture/external_agent_autonomous_execution_fabric/task_board.json"
)
RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-materialization@2"
)
POPULATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-population@1"
)
NAMESPACE_CLAIM_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "external-agent-autonomous-execution-fabric-namespace-claim@2"
)
SCHEDULER_CONFIG_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "external_agent_autonomous_execution_fabric.scheduler_config@2"
)
RUNTIME_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-runtime-binding@1"
)
RUNTIME_INVOCATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-runtime-invocation@1"
)
NAMESPACE_CLAIM_FIELDS = frozenset(
    {
        "schema",
        "population_cid",
        "plan_root_cid",
        "source_head",
        "source_tree",
        "source_generation_cid",
        "runtime_binding",
        "runtime_binding_cid",
        "materialization_invocation",
        "store_generation",
        "database_program_bindings",
        "database_paths",
        "maximum_writer_processes",
        "partial_effect_policy",
        "process_started",
        "claim_cid",
    }
)
MATERIALIZATION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "namespace_claim_cid",
        "authority_mode",
        "maximum_writer_processes",
        "continuous_quack_authority",
        "ducklake_authority",
        "board_validation",
        "population_cid",
        "plan_root_cid",
        "source_head",
        "source_tree",
        "source_generation",
        "runtime_binding",
        "runtime_binding_cid",
        "materialization_invocation",
        "controls",
        "database_paths",
        "database_program_bindings",
        "schema_install",
        "schema_verification",
        "operation_vocabulary_cid",
        "operational_profile_verification",
        "borrowed_transaction_handler_source_evidence",
        "control_schema_projection",
        "database_materialization",
        "control_projection",
        "coordination_projection",
        "execution_projection",
        "ready_task_aliases",
        "process_started",
        "receipt_cid",
    }
)
SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-signed-command-fabric-profile@2"
)
SIGNED_COMMAND_FABRIC_PROFILE_FIELDS = frozenset(
    {
        "schema",
        "transport_kind",
        "board_namespace",
        "shard_id",
        "ingress_endpoint",
        "ingress_secret_handle",
        "projection_endpoint",
        "projection_secret_handle",
        "store_id",
        "store_generation",
        "schema_revision",
        "owner_qualification_schema",
        "command_envelope_schema",
        "state_command_schema",
        "ingress_relation",
        "ingress_append_only",
        "ingress_accepts_signed_envelopes_only",
        "operational_database_private",
        "operational_tables_remotely_exposed",
        "one_mutable_owner",
        "owner_verifies_signed_envelopes",
        "projection_read_only",
        "projection_append_allowed",
        "atomic_plan_r2_required",
        "direct_file_fallback",
        "failover_policy",
        "child_adapter_status",
    }
)


class MaterializationError(RuntimeError):
    """Fail-closed bootstrap materialization error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _cid(value: Any) -> str:
    raw = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid JSON {path.relative_to(ROOT)}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{path.relative_to(ROOT)} must contain an object")
    return value


def _relative_path(value: Any, *, field: str) -> Path:
    raw = str(value or "")
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise MaterializationError(f"{field} must be a safe repository-relative path")
    resolved = (ROOT / path).resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise MaterializationError(f"{field} escapes the repository") from exc
    return resolved


def _git(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise MaterializationError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def _assert_clean() -> None:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise MaterializationError(
            "materialization requires a clean exact source tree; commit the board, "
            "manifests, policy and source bindings first"
        )


def _file_cid(path: Path) -> str:
    try:
        return _cid(path.read_bytes())
    except OSError as exc:
        raise MaterializationError(f"unable to read {path.relative_to(ROOT)}: {exc}") from exc


def _external_file_sha256(path: Path) -> str:
    try:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise MaterializationError(f"unable to hash runtime file {path}: {exc}") from exc


def _canonical_runtime_path(value: Any, *, field: str, directory: bool) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise MaterializationError(f"{field} must be a canonical absolute path")
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise MaterializationError(f"{field} does not exist: {path}") from exc
    if str(resolved) != value:
        raise MaterializationError(f"{field} is not a canonical resolved path")
    if directory and not resolved.is_dir():
        raise MaterializationError(f"{field} is not a directory")
    if not directory and not resolved.is_file():
        raise MaterializationError(f"{field} is not a file")
    return resolved


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise MaterializationError(f"{field} is not a full SHA-256 identity")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise MaterializationError(f"{field} is not a full SHA-256 identity") from exc
    return value


def _canonical_absent_runtime_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise MaterializationError(f"{field} must be a canonical absolute path")
    path = Path(value)
    if path.exists() or str(path.resolve(strict=False)) != value:
        raise MaterializationError(f"{field} must be canonical and absent")
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    try:
        mode = ancestor.stat()
    except OSError as exc:
        raise MaterializationError(f"{field} has no admitted existing ancestor") from exc
    if mode.st_uid != 0 or mode.st_mode & 0o022:
        raise MaterializationError(f"{field} is not beneath a root-owned non-writable ancestor")
    return path


def _runtime_binding_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the committed runtime contract without importing third-party code."""

    value = config.get("bootstrap_runtime_binding")
    if not isinstance(value, Mapping):
        raise MaterializationError("bootstrap_runtime_binding is missing")
    if set(value) != {
        "schema",
        "launcher",
        "interpreter",
        "approved_import_root",
        "duckdb",
    }:
        raise MaterializationError("bootstrap_runtime_binding shape is not canonical")
    if value.get("schema") != RUNTIME_BINDING_SCHEMA:
        raise MaterializationError("bootstrap_runtime_binding schema is not canonical")
    launcher = value.get("launcher")
    interpreter = value.get("interpreter")
    duckdb_binding = value.get("duckdb")
    if not isinstance(launcher, Mapping) or set(launcher) != {
        "resolved_path",
        "sha256",
        "argv_prefix",
        "allowed_commands",
    }:
        raise MaterializationError("bootstrap launcher binding shape is not canonical")
    if not isinstance(interpreter, Mapping) or set(interpreter) != {
        "resolved_path",
        "sha256",
        "version",
        "cache_tag",
        "platform",
        "pycache_prefix",
        "flags",
    }:
        raise MaterializationError("bootstrap interpreter binding shape is not canonical")
    flags = interpreter.get("flags")
    required_flags = {
        "isolated": 1,
        "no_site": 1,
        "dont_write_bytecode": 1,
        "no_user_site": 1,
        "ignore_environment": 1,
        "safe_path": True,
    }
    if not isinstance(flags, Mapping) or _canonical_bytes(dict(flags)) != _canonical_bytes(
        required_flags
    ):
        raise MaterializationError("bootstrap interpreter must require exact -I -S -B flags")
    if not isinstance(duckdb_binding, Mapping) or set(duckdb_binding) != {
        "distribution_name",
        "distribution_version",
        "module_version",
        "engine_version",
        "module_path",
        "module_sha256",
        "extension_path",
        "extension_sha256",
        "record_path",
        "record_sha256",
        "record_entry_count",
        "record_verified_file_count",
        "record_unhashed_pyc_count",
        "record_payload_cid",
    }:
        raise MaterializationError("bootstrap DuckDB binding shape is not canonical")
    for field in ("version", "cache_tag", "platform"):
        if not isinstance(interpreter.get(field), str) or not interpreter.get(field):
            raise MaterializationError(f"bootstrap interpreter {field} is required")
    for field in (
        "distribution_name",
        "distribution_version",
        "module_version",
        "engine_version",
    ):
        if not isinstance(duckdb_binding.get(field), str) or not duckdb_binding.get(field):
            raise MaterializationError(f"bootstrap DuckDB {field} is required")
    for field in (
        "record_entry_count",
        "record_verified_file_count",
        "record_unhashed_pyc_count",
    ):
        if type(duckdb_binding.get(field)) is not int or int(duckdb_binding[field]) < 0:
            raise MaterializationError(f"bootstrap DuckDB {field} is not a bounded integer")
    _require_sha256(
        duckdb_binding.get("record_payload_cid"),
        field="bootstrap_runtime_binding.duckdb.record_payload_cid",
    )

    interpreter_path = _canonical_runtime_path(
        interpreter.get("resolved_path"),
        field="bootstrap_runtime_binding.interpreter.resolved_path",
        directory=False,
    )
    import_root = _canonical_runtime_path(
        value.get("approved_import_root"),
        field="bootstrap_runtime_binding.approved_import_root",
        directory=True,
    )
    launcher_path = _canonical_runtime_path(
        launcher.get("resolved_path"),
        field="bootstrap_runtime_binding.launcher.resolved_path",
        directory=False,
    )
    expected_launcher_path = (
        ROOT / "scripts/launch_external_agent_autonomous_execution_fabric_materializer.py"
    ).resolve(strict=True)
    if launcher_path != expected_launcher_path:
        raise MaterializationError("bootstrap launcher path is not the reviewed repository launcher")
    argv_prefix = launcher.get("argv_prefix")
    expected_argv_prefix = [
        str(interpreter_path),
        "-I",
        "-S",
        "-B",
        str(launcher_path),
    ]
    if not isinstance(argv_prefix, list) or argv_prefix != expected_argv_prefix:
        raise MaterializationError("bootstrap launcher argv_prefix is not canonical")
    if launcher.get("allowed_commands") != [
        "build",
        "runtime-check",
        "materialize",
        "verify",
        "launch-plan",
        "configured-board-launch",
    ]:
        raise MaterializationError("bootstrap launcher allowed_commands is not canonical")
    _canonical_absent_runtime_path(
        interpreter.get("pycache_prefix"),
        field="bootstrap_runtime_binding.interpreter.pycache_prefix",
    )
    module_path = _canonical_runtime_path(
        duckdb_binding.get("module_path"),
        field="bootstrap_runtime_binding.duckdb.module_path",
        directory=False,
    )
    extension_path = _canonical_runtime_path(
        duckdb_binding.get("extension_path"),
        field="bootstrap_runtime_binding.duckdb.extension_path",
        directory=False,
    )
    record_path = _canonical_runtime_path(
        duckdb_binding.get("record_path"),
        field="bootstrap_runtime_binding.duckdb.record_path",
        directory=False,
    )
    for path, field in (
        (module_path, "module_path"),
        (extension_path, "extension_path"),
        (record_path, "record_path"),
    ):
        try:
            path.relative_to(import_root)
        except ValueError as exc:
            raise MaterializationError(
                f"bootstrap DuckDB {field} is outside the single approved import root"
            ) from exc
    for path, expected, field in (
        (launcher_path, launcher.get("sha256"), "launcher.sha256"),
        (interpreter_path, interpreter.get("sha256"), "interpreter.sha256"),
        (module_path, duckdb_binding.get("module_sha256"), "duckdb.module_sha256"),
        (
            extension_path,
            duckdb_binding.get("extension_sha256"),
            "duckdb.extension_sha256",
        ),
        (record_path, duckdb_binding.get("record_sha256"), "duckdb.record_sha256"),
    ):
        digest = _require_sha256(expected, field=f"bootstrap_runtime_binding.{field}")
        if _external_file_sha256(path) != digest:
            raise MaterializationError(f"bootstrap runtime file differs from {field}")
    return json.loads(json.dumps(value, sort_keys=True))


def _verify_duckdb_record(
    record_path: Path,
    import_root: Path,
) -> dict[str, Any]:
    """Verify the bounded wheel RECORD payload before any DuckDB import."""

    import base64
    import binascii
    import csv
    from pathlib import PurePosixPath

    try:
        with record_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise MaterializationError(f"unable to parse DuckDB RECORD: {exc}") from exc
    if not rows or len(rows) > 4096:
        raise MaterializationError("DuckDB RECORD entry population is not bounded")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    unhashed_pyc_count = 0
    record_rows = 0
    total_verified_bytes = 0
    for row in rows:
        if len(row) != 3:
            raise MaterializationError("DuckDB RECORD row shape is not canonical")
        raw_path, encoded_digest, raw_size = row
        relative = PurePosixPath(raw_path)
        if (
            not raw_path
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in raw_path
            or raw_path in seen
        ):
            raise MaterializationError("DuckDB RECORD contains an unsafe or duplicate path")
        seen.add(raw_path)
        target = (import_root / Path(*relative.parts)).resolve(strict=False)
        try:
            target.relative_to(import_root)
        except ValueError as exc:
            raise MaterializationError("DuckDB RECORD member escapes the import root") from exc
        if not encoded_digest:
            if raw_size:
                raise MaterializationError("unhashed DuckDB RECORD member declares a size")
            if target == record_path:
                record_rows += 1
                continue
            if "__pycache__" not in relative.parts or relative.suffix != ".pyc":
                raise MaterializationError("DuckDB RECORD contains an unverified executable member")
            unhashed_pyc_count += 1
            continue
        if not encoded_digest.startswith("sha256=") or not raw_size.isdecimal():
            raise MaterializationError("DuckDB RECORD member does not use SHA-256 and size")
        try:
            target = target.resolve(strict=True)
            expected_digest = base64.urlsafe_b64decode(
                encoded_digest[7:] + "=" * ((4 - len(encoded_digest[7:]) % 4) % 4)
            )
            size = target.stat().st_size
        except (OSError, ValueError, binascii.Error) as exc:
            raise MaterializationError(f"DuckDB RECORD member is invalid: {raw_path}") from exc
        if len(expected_digest) != 32 or size != int(raw_size):
            raise MaterializationError(f"DuckDB RECORD member size/digest is invalid: {raw_path}")
        total_verified_bytes += size
        if total_verified_bytes > 512 * 1024 * 1024:
            raise MaterializationError("DuckDB RECORD verified byte population exceeds policy")
        try:
            actual_digest = hashlib.sha256(target.read_bytes()).digest()
        except OSError as exc:
            raise MaterializationError(f"unable to read DuckDB RECORD member: {raw_path}") from exc
        if actual_digest != expected_digest:
            raise MaterializationError(f"DuckDB RECORD member digest differs: {raw_path}")
        verified.append(
            {
                "path": raw_path,
                "sha256": "sha256:" + actual_digest.hex(),
                "size": size,
            }
        )
    if record_rows != 1:
        raise MaterializationError("DuckDB RECORD does not contain exactly one self row")
    return {
        "record_entry_count": len(rows),
        "record_verified_file_count": len(verified),
        "record_unhashed_pyc_count": unhashed_pyc_count,
        "record_payload_cid": _cid(verified),
    }


def _observe_runtime_binding(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Observe the already-isolated interpreter and the exact DuckDB import."""

    import importlib.util
    import sysconfig
    from importlib.metadata import distribution

    expected_interpreter = expected["interpreter"]
    expected_launcher = expected["launcher"]
    expected_duckdb = expected["duckdb"]
    approved_import_root = str(expected["approved_import_root"])
    observed_flags = {
        "isolated": int(sys.flags.isolated),
        "no_site": int(sys.flags.no_site),
        "dont_write_bytecode": int(sys.flags.dont_write_bytecode),
        "no_user_site": int(sys.flags.no_user_site),
        "ignore_environment": int(sys.flags.ignore_environment),
        "safe_path": bool(sys.flags.safe_path),
    }
    if _canonical_bytes(observed_flags) != _canonical_bytes(expected_interpreter["flags"]):
        raise MaterializationError("runtime interpreter is not running with exact -I -S -B flags")
    if sys.pycache_prefix != expected_interpreter["pycache_prefix"]:
        raise MaterializationError("runtime pycache prefix is not the admitted absent path")
    if Path(str(sys.pycache_prefix)).exists():
        raise MaterializationError("runtime pycache prefix unexpectedly exists")

    stdlib = Path(sysconfig.get_path("stdlib")).resolve(strict=True)
    expected_sys_path = [
        str(ROOT.resolve(strict=True)),
        approved_import_root,
        str(stdlib.parent / f"python{sys.version_info.major}{sys.version_info.minor}.zip"),
        str(stdlib),
        str((stdlib / "lib-dynload").resolve(strict=True)),
    ]
    if sys.path != expected_sys_path:
        raise MaterializationError(
            "runtime sys.path differs from the closed repository/stdlib/import-root projection"
        )
    site_roots: list[str] = []
    for entry in sys.path:
        if not isinstance(entry, str) or not entry:
            continue
        path = Path(entry)
        if not ({"site-packages", "dist-packages"} & set(path.parts)):
            continue
        try:
            site_roots.append(str(path.resolve(strict=True)))
        except OSError as exc:
            raise MaterializationError(f"runtime import root is missing: {path}") from exc
    if site_roots != [approved_import_root]:
        raise MaterializationError(
            "runtime must expose exactly one canonical approved site-packages import root"
        )

    if "duckdb" in sys.modules or "_duckdb" in sys.modules:
        raise MaterializationError("DuckDB must not be preloaded before runtime admission")
    record_projection = _verify_duckdb_record(
        Path(str(expected_duckdb["record_path"])),
        Path(approved_import_root),
    )
    expected_record_projection = {
        key: expected_duckdb[key]
        for key in (
            "record_entry_count",
            "record_verified_file_count",
            "record_unhashed_pyc_count",
            "record_payload_cid",
        )
    }
    if _canonical_bytes(record_projection) != _canonical_bytes(expected_record_projection):
        raise MaterializationError("DuckDB RECORD payload projection differs from binding")
    for module_name, expected_path in (
        ("duckdb", str(expected_duckdb["module_path"])),
        ("_duckdb", str(expected_duckdb["extension_path"])),
    ):
        spec = importlib.util.find_spec(module_name)
        origin = "" if spec is None or spec.origin is None else str(Path(spec.origin).resolve())
        if origin != expected_path:
            raise MaterializationError(
                f"runtime {module_name} import does not resolve to the admitted path"
            )

    import _duckdb
    import duckdb

    duckdb_distribution = distribution("duckdb")
    record_entries = [
        item
        for item in duckdb_distribution.files or ()
        if item.name == "RECORD" and item.parent.name.endswith(".dist-info")
    ]
    if len(record_entries) != 1:
        raise MaterializationError("DuckDB distribution has no unique dist-info RECORD")
    record_path = Path(duckdb_distribution.locate_file(record_entries[0])).resolve(strict=True)
    interpreter_path = Path(sys.executable).resolve(strict=True)
    module_path = Path(duckdb.__file__ or "").resolve(strict=True)
    extension_path = Path(_duckdb.__file__ or "").resolve(strict=True)
    engine_version = str(duckdb.sql("SELECT version()").fetchone()[0])
    return {
        "schema": RUNTIME_BINDING_SCHEMA,
        "launcher": dict(expected_launcher),
        "interpreter": {
            "resolved_path": str(interpreter_path),
            "sha256": _external_file_sha256(interpreter_path),
            "version": sys.version,
            "cache_tag": str(sys.implementation.cache_tag or ""),
            "platform": sysconfig.get_platform(),
            "pycache_prefix": str(sys.pycache_prefix or ""),
            "flags": observed_flags,
        },
        "approved_import_root": approved_import_root,
        "duckdb": {
            "distribution_name": str(duckdb_distribution.metadata.get("Name") or ""),
            "distribution_version": str(duckdb_distribution.version),
            "module_version": str(duckdb.__version__),
            "engine_version": engine_version,
            "module_path": str(module_path),
            "module_sha256": _external_file_sha256(module_path),
            "extension_path": str(extension_path),
            "extension_sha256": _external_file_sha256(extension_path),
            "record_path": str(record_path),
            "record_sha256": _external_file_sha256(record_path),
            **record_projection,
        },
    }


def _validated_runtime_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = _runtime_binding_contract(config)
    observed = _observe_runtime_binding(expected)
    if _canonical_bytes(observed) != _canonical_bytes(expected):
        raise MaterializationError(
            "observed bootstrap runtime differs from the committed runtime binding"
        )
    return observed


def _runtime_invocation_projection(
    runtime_binding: Mapping[str, Any], command: str
) -> dict[str, Any]:
    launcher = runtime_binding.get("launcher")
    if not isinstance(launcher, Mapping) or command not in (
        launcher.get("allowed_commands") or ()
    ):
        raise MaterializationError("bootstrap runtime command is not admitted")
    argv_prefix = launcher.get("argv_prefix")
    if not isinstance(argv_prefix, list):
        raise MaterializationError("bootstrap launcher argv_prefix is missing")
    projection = {
        "schema": RUNTIME_INVOCATION_SCHEMA,
        "command": command,
        "orig_argv": [*argv_prefix, command],
        "materializer_argv": [
            str(
                (
                    ROOT
                    / "scripts/materialize_external_agent_autonomous_execution_fabric_control_plane.py"
                ).resolve(strict=True)
            ),
            command,
        ],
        "launcher_path": str(launcher.get("resolved_path") or ""),
        "launcher_sha256": str(launcher.get("sha256") or ""),
    }
    projection["invocation_cid"] = _cid(projection)
    return projection


def _validated_runtime_invocation(
    runtime_binding: Mapping[str, Any], command: str
) -> dict[str, Any]:
    expected = _runtime_invocation_projection(runtime_binding, command)
    if list(sys.orig_argv) != expected["orig_argv"]:
        raise MaterializationError("runtime sys.orig_argv differs from admitted launcher command")
    if list(sys.argv) != expected["materializer_argv"]:
        raise MaterializationError("runtime sys.argv differs from admitted materializer command")
    return expected


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    if str(config.get("schema") or "") != SCHEDULER_CONFIG_SCHEMA:
        raise MaterializationError("scheduler config schema identity is not canonical")
    program = config.get("bootstrap_database_program")
    if not isinstance(program, Mapping):
        raise MaterializationError("bootstrap_database_program is missing")
    operational = config.get("database_program")
    if not isinstance(operational, Mapping):
        raise MaterializationError("operational database_program is missing")
    if dict(program) == dict(operational):
        raise MaterializationError(
            "bootstrap and operational database programs must be distinct"
        )
    if str(program.get("authority_mode") or "") != "embedded":
        raise MaterializationError("bootstrap database authority must be embedded")
    if str(program.get("task_source_kind") or "") != "duckdb":
        raise MaterializationError("bootstrap task source must be duckdb")
    if int(program.get("maximum_writer_processes") or 0) != 1:
        raise MaterializationError("bootstrap permits exactly one writer process")
    for field in (
        "store_generation",
        "schema_revision",
        "event_store_path",
        "runtime_registry_path",
        "worktree_root",
        "export_profile",
        "failover_policy",
    ):
        if not str(program.get(field) or "").strip():
            raise MaterializationError(
                f"bootstrap_database_program.{field} is required"
            )
    if str(program.get("failover_policy")) != "fail_closed":
        raise MaterializationError("bootstrap database failover policy must be fail_closed")
    result = {
        "control": _relative_path(
            program.get("store_id"),
            field="bootstrap_database_program.store_id",
        ),
        "coordination": _relative_path(
            program.get("coordination_store_id"),
            field="bootstrap_database_program.coordination_store_id",
        ),
        "execution": _relative_path(
            program.get("execution_store_id"),
            field="bootstrap_database_program.execution_store_id",
        ),
    }
    if len(set(result.values())) != 3:
        raise MaterializationError("control, coordination and execution stores must be distinct")
    control = result["control"]
    if control.suffix.lower() not in {".duckdb", ".ddb"}:
        raise MaterializationError(
            "bootstrap_database_program.store_id must identify a DuckDB file"
        )
    expected = {
        "coordination": control.with_name(f"{control.stem}.coordination.duckdb"),
        "execution": control.with_name(f"{control.stem}.execution.duckdb"),
    }
    for name, path in expected.items():
        if result[name] != path:
            raise MaterializationError(
                f"bootstrap_database_program.{name}_store_id must equal the deterministic "
                f"DatabaseImplementationDaemon sidecar {path.relative_to(ROOT)}"
            )
    return result


def _operational_command_fabric_profile(
    config: Mapping[str, Any],
    *,
    operational_program: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the prospective split ingress/private-owner/egress topology."""

    raw = config.get("operational_command_fabric")
    if not isinstance(raw, Mapping):
        raise MaterializationError("operational_command_fabric is required")
    if set(raw) != SIGNED_COMMAND_FABRIC_PROFILE_FIELDS:
        raise MaterializationError(
            "operational_command_fabric shape is not canonical"
        )
    profile = dict(raw)
    exact = {
        "schema": SIGNED_COMMAND_FABRIC_PROFILE_SCHEMA,
        "transport_kind": "signed_command_fabric",
        "board_namespace": "external-agent-autonomous-execution-fabric-v1",
        "shard_id": "control-shard-0",
        "owner_qualification_schema": (
            "ipfs_accelerate_py/agent-supervisor/eaaef-quack-owner-qualification@1"
        ),
        "command_envelope_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        "state_command_schema": (
            "ipfs_accelerate_py/agent-supervisor/state-command@1"
        ),
        "ingress_relation": "command_inbox",
        "ingress_append_only": True,
        "ingress_accepts_signed_envelopes_only": True,
        "operational_database_private": True,
        "operational_tables_remotely_exposed": False,
        "one_mutable_owner": True,
        "owner_verifies_signed_envelopes": True,
        "projection_read_only": True,
        "projection_append_allowed": False,
        "atomic_plan_r2_required": True,
        "direct_file_fallback": False,
        "failover_policy": "fail_closed",
        "child_adapter_status": "implemented_unqualified_fail_closed",
    }
    if any(profile.get(field) != value for field, value in exact.items()):
        raise MaterializationError(
            "operational signed command fabric policy is not fail closed"
        )
    if _host_receipt_decision("EAAEF-188") == "admitted":
        profile["child_adapter_status"] = "admitted"
    ingress = str(profile.get("ingress_endpoint") or "")
    projection = str(profile.get("projection_endpoint") or "")
    if (
        ingress == projection
        or not ingress.startswith("quack:127.0.0.1:")
        or not projection.startswith("quack:127.0.0.1:")
        or not str(profile.get("ingress_secret_handle") or "").startswith(
            "secret-handle:"
        )
        or not str(profile.get("projection_secret_handle") or "").startswith(
            "secret-handle:"
        )
        or profile.get("store_id") != operational_program.get("store_id")
        or profile.get("store_generation")
        != operational_program.get("store_generation")
        or profile.get("schema_revision")
        != operational_program.get("schema_revision")
    ):
        raise MaterializationError(
            "operational signed command fabric identity is inconsistent"
        )
    if profile.get("board_namespace") != config.get("board_namespace"):
        raise MaterializationError(
            "operational signed command fabric board namespace differs"
        )
    return profile


def _database_program_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind separate bootstrap and operational programs without opening Quack."""

    bootstrap = config.get("bootstrap_database_program")
    operational = config.get("database_program")
    if not isinstance(bootstrap, Mapping) or not isinstance(operational, Mapping):
        raise MaterializationError(
            "bootstrap_database_program and operational database_program are required"
        )
    _paths(config)
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        DatabaseProgramConfig,
        DatabaseProgramConfigError,
    )

    try:
        bootstrap_program = DatabaseProgramConfig.from_mapping(bootstrap)
        operational_program = DatabaseProgramConfig.from_mapping(operational)
    except DatabaseProgramConfigError as exc:
        raise MaterializationError(f"invalid database program: {exc}") from exc
    if (
        bootstrap_program.authority_mode != "embedded"
        or bootstrap_program.task_source_kind != "duckdb"
    ):
        raise MaterializationError(
            "bootstrap_database_program must be embedded DuckDB"
        )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
        EAAEF_OPERATIONAL_PROFILE_ID,
    )

    if (
        bootstrap_program.schema_revision != EAAEF_OPERATIONAL_PROFILE_ID
        or operational_program.schema_revision != EAAEF_OPERATIONAL_PROFILE_ID
    ):
        raise MaterializationError(
            "bootstrap and operational database programs must bind the exact "
            "EAAEF operational profile @2"
        )
    if (
        operational_program.authority_mode != "quack"
        or operational_program.task_source_kind != "duckdb"
        or operational_program.failover_policy != "fail_closed"
        or not operational_program.quack_endpoint
        or not operational_program.endpoint_secret_handle
        or not operational_program.store_id
        or "/" in operational_program.store_id
        or "\\" in operational_program.store_id
        or operational_program.store_id.endswith((".duckdb", ".ddb"))
    ):
        raise MaterializationError(
            "operational database_program must be remote Quack with no direct-file fallback"
        )
    command_fabric = _operational_command_fabric_profile(
        config,
        operational_program=operational_program.to_dict(),
    )
    projection = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-database-program-bindings@1"
        ),
        "bootstrap": bootstrap_program.to_dict(),
        "bootstrap_source_cid": _cid(dict(bootstrap)),
        "bootstrap_profile_cid": _cid(bootstrap_program.to_dict()),
        "operational": operational_program.to_dict(),
        "operational_source_cid": _cid(dict(operational)),
        "operational_database_program_profile_cid": _cid(
            operational_program.to_dict()
        ),
        "operational_command_fabric": command_fabric,
        "operational_profile_cid": _cid(command_fabric),
        "operational_child_adapter_status": (
            "admitted"
            if _host_receipt_decision("EAAEF-188") == "admitted"
            else "implemented_unqualified_fail_closed"
        ),
        "materializer_opens_operational_profile": False,
        "direct_file_fallback": False,
    }
    if projection["bootstrap_profile_cid"] == projection["operational_profile_cid"]:
        raise MaterializationError("database program profile identities collide")
    projection["binding_cid"] = _cid(projection)
    return projection


def _receipt_path(config: Mapping[str, Any]) -> Path:
    registry = _relative_path(
        (config.get("bootstrap_database_program") or {}).get(
            "runtime_registry_path"
        ),
        field="bootstrap_database_program.runtime_registry_path",
    )
    return registry / "bootstrap-materialization.json"


def _claim_path(config: Mapping[str, Any]) -> Path:
    registry = _relative_path(
        (config.get("bootstrap_database_program") or {}).get(
            "runtime_registry_path"
        ),
        field="bootstrap_database_program.runtime_registry_path",
    )
    return registry / "bootstrap-materialization-claim.json"


def _namespace_artifacts(config: Mapping[str, Any]) -> tuple[Path, ...]:
    """Return every known file whose presence makes the namespace non-fresh."""

    paths = _paths(config)
    # The complete run directory is one immutable generation.  Checking only
    # the three databases would admit a stale PID, event cursor, merge queue,
    # worktree, or registry artifact from an earlier partial attempt.
    members: list[Path] = [
        paths["control"].parent,
        *paths.values(),
        _claim_path(config),
        _receipt_path(config),
    ]
    program = config.get("bootstrap_database_program") or {}
    for field in (
        "event_store_path",
        "runtime_registry_path",
        "worktree_root",
        "merge_queue_dir",
        "state_dir",
    ):
        members.append(
            _relative_path(
                program.get(field),
                field=f"bootstrap_database_program.{field}",
            )
        )
    for path in paths.values():
        members.append(Path(f"{path}.wal"))
    members.append(
        paths["execution"].with_name(f".{paths['execution'].name}.writer.lock")
    )
    return tuple(dict.fromkeys(members))


_GENERATION_RE = re.compile(r"^(?P<prefix>.+-v)(?P<n>\d+)$")
GENERATION_CURSOR_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-store-generation-cursor@1"
)
MAX_GENERATION_RECOVERIES = 3
_HOST_GATED_MARKERS = (
    "quack_owner_",
    "provider_container_qualification",
    "oci_image_qualification",
    "container_profile",
    "eaaef_scoped_provider",
    "worker_network_",
    "signed_command_fabric_child_adapter",
    "board validation has not admitted",
    "container_policy.",
    "configured-board launch admission",
    "independently signed",
    "provider/container qualification is diagnostic",
    "versioned Grok mount",
    "typed authenticated Quack",
    "external source-addressed EAAEF-000",
    "rootless",
    "DuckDB",
)
_HOST_EVIDENCE_DIRTY_PREFIXES = (
    "docs/architecture/external_agent_autonomous_execution_fabric/receipts/host_admission/",
    "data/agent_supervisor/external_agent_autonomous_execution_fabric/authority/host-evidence/",
)
_STALE_LAUNCH_BLOCKER_MARKERS = (
    "no externally signed @2 profile artifact or admitted container engine exists",
    "provider/container qualification is diagnostic-only",
    "typed authenticated Quack ingress",
    "external source-addressed EAAEF-000 operational capability",
)
_AUTO_RECOVERABLE_MARKERS = (
    "advance to a new explicit store generation",
    "bootstrap namespace claim is immutable",
    "output path is not a safe identifier",
    "refusing to overwrite existing bootstrap namespace",
    "differs from current source",
    "materialization_source_or_board_mismatch",
    "another_supervisor_holds_identity_recovery",
)


def _generation_cursor_path() -> Path:
    return (
        ROOT
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / "generation-cursor.json"
    )


def _configured_generation(config: Mapping[str, Any]) -> str:
    generation = str(
        ((config.get("bootstrap_database_program") or {}).get("store_generation"))
        or ""
    )
    if not _GENERATION_RE.fullmatch(generation):
        raise MaterializationError("store_generation is not a recoverable run-vN identity")
    return generation


def _successor_generation(generation: str) -> str:
    match = _GENERATION_RE.fullmatch(str(generation or ""))
    if match is None:
        raise MaterializationError(
            f"store_generation {generation!r} cannot be advanced automatically"
        )
    return f"{match.group('prefix')}{int(match.group('n')) + 1}"


def _rewrite_generation(value: Any, from_generation: str, to_generation: str) -> Any:
    from_match = _GENERATION_RE.fullmatch(from_generation)
    to_match = _GENERATION_RE.fullmatch(to_generation)
    if from_match is None or to_match is None:
        raise MaterializationError("generation rewrite identities are invalid")
    from_n = from_match.group("n")
    to_n = to_match.group("n")
    if isinstance(value, str):
        rewritten = value
        for old, new in (
            (from_generation, to_generation),
            (f"-run-v{from_n}", f"-run-v{to_n}"),
            (f"/run-v{from_n}", f"/run-v{to_n}"),
        ):
            rewritten = rewritten.replace(old, new)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_generation(item, from_generation, to_generation) for item in value]
    if isinstance(value, dict):
        return {
            key: _rewrite_generation(item, from_generation, to_generation)
            for key, item in value.items()
        }
    return value


def _read_generation_cursor() -> dict[str, Any] | None:
    path = _generation_cursor_path()
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != GENERATION_CURSOR_SCHEMA:
        return None
    return value


def _write_generation_cursor(cursor: Mapping[str, Any]) -> None:
    path = _generation_cursor_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(cursor)
    payload["schema"] = GENERATION_CURSOR_SCHEMA
    payload["process_started"] = False
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _active_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay a gitignored generation cursor onto the committed scheduler config."""

    working = copy.deepcopy(dict(config))
    configured = _configured_generation(working)
    cursor = _read_generation_cursor()
    if (
        not isinstance(cursor, Mapping)
        or cursor.get("configured_generation") != configured
    ):
        return working
    active = str(cursor.get("active_generation") or "")
    if not active or active == configured:
        return working
    return _rewrite_generation(working, configured, active)


def _namespace_state(config: Mapping[str, Any]) -> str:
    receipt_path = _receipt_path(config)
    claim_path = _claim_path(config)
    if receipt_path.is_file():
        try:
            receipt = _load_object(receipt_path)
            head = _git("rev-parse", "HEAD")
        except MaterializationError:
            return "failed_partial"
        if str(receipt.get("source_head") or "") == head:
            return "materialized"
        return "stale_materialized"
    if claim_path.is_file() or any(
        path.exists() for path in _namespace_artifacts(config) if path != _paths(config)["control"].parent
    ):
        return "failed_partial"
    return "fresh"


def _porcelain_paths(status: str) -> list[str]:
    paths: list[str] = []
    for line in str(status or "").splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[-1]
        if path:
            paths.append(path)
    return paths


def _is_host_evidence_only_dirty(status: str) -> bool:
    paths = _porcelain_paths(status)
    if not paths:
        return False
    return all(
        any(path.startswith(prefix) for prefix in _HOST_EVIDENCE_DIRTY_PREFIXES)
        for path in paths
    )


def _host_receipt_decision(task_id: str) -> str:
    names = {
        "EAAEF-182": "duckdb_quack_155.json",
        "EAAEF-183": "engine_mode.json",
        "EAAEF-184": "provider_authorization.json",
        "EAAEF-185": "worker_image.json",
        "EAAEF-186": "container_profile.json",
        "EAAEF-187": "worker_network.json",
        "EAAEF-188": "command_fabric_endpoints.json",
        "EAAEF-189": "native_lane_dispatcher.json",
        "EAAEF-190": "plan_r2_remote_owner.json",
        "EAAEF-191": "admission_bundle.json",
    }
    filename = names.get(task_id)
    if not filename:
        return ""
    path = (
        ROOT
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "receipts/host_admission"
        / filename
    )
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    decision = str(payload.get("decision") or "")
    if task_id != "EAAEF-191" or decision != "admitted":
        return decision
    try:
        from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
            verify_admission_bundle_receipt,
        )

        board = _load_object(EAAEF_BOARD_PATH)
        verification = verify_admission_bundle_receipt(
            receipt_dir=path.parent,
            expected_source_head=_git("rev-parse", "HEAD"),
            expected_source_tree=_git("rev-parse", "HEAD^{tree}"),
            expected_board_namespace=str(board.get("board_namespace") or ""),
            expected_board_cid=str(board.get("board_cid") or ""),
        )
    except Exception:
        return "no_go"
    return "admitted" if verification.get("admitted") is True else "no_go"


def _drop_stale_launch_blockers(blockers: list[str]) -> list[str]:
    engine_admitted = _host_receipt_decision("EAAEF-183") == "admitted"
    profile_admitted = _host_receipt_decision("EAAEF-186") == "admitted"
    image_admitted = _host_receipt_decision("EAAEF-185") == "admitted"
    fabric_admitted = _host_receipt_decision("EAAEF-188") == "admitted"
    lane_admitted = _host_receipt_decision("EAAEF-189") == "admitted"
    bundle_admitted = _host_receipt_decision("EAAEF-191") == "admitted"
    kept: list[str] = []
    for blocker in blockers:
        if (
            "no externally signed @2 profile artifact or admitted container engine exists"
            in blocker
            and engine_admitted
            and profile_admitted
        ):
            continue
        if (
            "provider/container qualification is diagnostic-only" in blocker
            and image_admitted
            and profile_admitted
        ):
            continue
        if (
            "typed authenticated Quack ingress" in blocker
            and fabric_admitted
            and lane_admitted
        ):
            continue
        if (
            "external source-addressed EAAEF-000 operational capability" in blocker
            and bundle_admitted
        ):
            continue
        if "nested checkout is dirty" in blocker and bundle_admitted:
            continue
        if "differs from current source" in blocker and bundle_admitted:
            continue
        if "unable to open authority read-only" in blocker and bundle_admitted:
            continue
        if bundle_admitted and _classify_blocker(blocker) == "host_gated_external_authority":
            continue
        kept.append(blocker)
    return kept


def _classify_blocker(text: str) -> str:
    raw = str(text or "")
    if "nested checkout is dirty" in raw:
        return "host_source_commit_required"
    if any(marker in raw for marker in _AUTO_RECOVERABLE_MARKERS):
        return "auto_recoverable"
    if any(marker in raw for marker in _HOST_GATED_MARKERS):
        return "host_gated_external_authority"
    return "unclassified"


def _source_generation(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config.get("source_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationError("source_binding is missing")
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    required_accelerator = str(binding.get("ipfs_accelerate_planning_revision") or "")
    required_accelerator_tree = str(binding.get("ipfs_accelerate_planning_tree") or "")
    if not required_accelerator or not required_accelerator_tree:
        raise MaterializationError("reviewed accelerator commit/tree binding is incomplete")
    if _git("rev-parse", f"{required_accelerator}^{{tree}}") != required_accelerator_tree:
        raise MaterializationError("reviewed accelerator integration tree differs from config")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", required_accelerator, head],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        != 0
    ):
        raise MaterializationError("reviewed accelerator integration root is not an ancestor")
    repositories = {
        "ipfs_accelerate_py": {
            "path": ROOT,
            "head": head,
            "tree": tree,
            "required_head": required_accelerator,
            "required_tree": required_accelerator_tree,
        },
        "ipfs_datasets_py": {
            "path": _relative_path(
                binding.get("ipfs_datasets_submodule_path"),
                field="source_binding.ipfs_datasets_submodule_path",
            ),
            "required_head": str(binding.get("ipfs_datasets_planning_revision") or ""),
            "required_tree": str(binding.get("ipfs_datasets_planning_tree") or ""),
        },
        "ipfs_kit_py": {
            "path": _relative_path(
                binding.get("ipfs_kit_submodule_path"),
                field="source_binding.ipfs_kit_submodule_path",
            ),
            "required_head": str(binding.get("ipfs_kit_planning_revision") or ""),
            "required_tree": str(binding.get("ipfs_kit_planning_tree") or ""),
        },
        "Mcp-Plus-Plus": {
            "path": _relative_path(
                binding.get("mcp_plus_plus_submodule_path"),
                field="source_binding.mcp_plus_plus_submodule_path",
            ),
            "required_head": str(binding.get("mcp_plus_plus_planning_revision") or ""),
            "required_tree": str(binding.get("mcp_plus_plus_planning_tree") or ""),
        },
    }
    projection: dict[str, Any] = {}
    planning_repositories: dict[str, dict[str, str]] = {}
    for name, record in repositories.items():
        path = Path(record["path"])
        nested_head = record.get("head") or _git("rev-parse", "HEAD", cwd=path)
        nested_tree = record.get("tree") or _git("rev-parse", "HEAD^{tree}", cwd=path)
        required_head = str(record.get("required_head") or "")
        required_tree = str(record.get("required_tree") or nested_tree)
        if not required_head or not required_tree:
            raise MaterializationError(f"{name} reviewed commit/tree binding is incomplete")
        if _git("rev-parse", f"{required_head}^{{tree}}", cwd=path) != required_tree:
            raise MaterializationError(f"{name} reviewed commit/tree binding is invalid")
        if name != "ipfs_accelerate_py" and (
            nested_head != required_head or nested_tree != required_tree
        ):
            descendant = (
                subprocess.run(
                    ["git", "merge-base", "--is-ancestor", required_head, nested_head],
                    cwd=path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ).returncode
                == 0
            )
            # EAAEF-191 host admission may accept a descendant overlay of the
            # reviewed nested root. The planning forest stays the R1 identities.
            if not descendant or _host_receipt_decision("EAAEF-191") != "admitted":
                raise MaterializationError(
                    f"{name} nested checkout differs from its reviewed root"
                )
        if name != "ipfs_accelerate_py":
            gitlink = _git("rev-parse", f"HEAD:{path.relative_to(ROOT).as_posix()}")
            if gitlink != nested_head:
                raise MaterializationError(f"{name} superproject gitlink differs from nested HEAD")
        nested_status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=path)
        if (
            nested_status
            and not _is_host_evidence_only_dirty(nested_status)
            and _host_receipt_decision("EAAEF-191") != "admitted"
        ):
            raise MaterializationError(f"{name} nested checkout is dirty")
        projection[name] = {
            "head": nested_head,
            "tree": nested_tree,
            "required_integration_head": required_head,
            "required_integration_tree": required_tree,
        }
        planning_repositories[name] = {"commit": required_head, "tree": required_tree}
    planning_forest = {
        "schema": "ExternalAgentSourceForest@1",
        "repositories": planning_repositories,
    }
    configured_forest_root = str(binding.get("source_forest_root") or "")
    if _cid(planning_forest) != configured_forest_root:
        raise MaterializationError("source_binding.source_forest_root differs from exact roots")
    projection["planning_source_forest_root"] = configured_forest_root
    projection["source_generation_cid"] = _cid(projection)
    return projection


def _validate_board(runtime_binding: Mapping[str, Any]) -> dict[str, Any]:
    validator = ROOT / "scripts/validate_external_agent_autonomous_execution_fabric_board.py"
    approved_raw = str(runtime_binding.get("approved_import_root") or "")
    approved_import_root = Path(approved_raw)
    try:
        approved_resolved = approved_import_root.resolve(strict=True)
    except OSError as exc:
        raise MaterializationError(
            "bootstrap approved import root is unavailable for board validation"
        ) from exc
    if (
        not approved_raw
        or not approved_import_root.is_absolute()
        or approved_resolved != approved_import_root
        or not approved_import_root.is_dir()
    ):
        raise MaterializationError(
            "bootstrap approved import root is noncanonical for board validation"
        )
    # The outer launcher has already admitted this exact import root. A fresh
    # isolated child does not inherit the launcher's sys.path mutation, so pass
    # the two closed roots as argv data and add them explicitly before loading
    # the native Markdown projection. The reviewed repository precedes the
    # approved dependency root, matching the outer launcher. Never inherit
    # PYTHONPATH or user-site discovery into the validator.
    isolated_validator = (
        "import runpy,sys;"
        "validator=sys.argv[3];"
        "sys.path[:0]=[sys.argv[2],sys.argv[1]];"
        "sys.argv=[validator,'--check-all'];"
        "runpy.run_path(validator,run_name='__main__')"
    )
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        isolated_validator,
        str(approved_import_root),
        str(ROOT),
        str(validator),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise MaterializationError(
            "board validation exceeded its 30-second sealed child deadline"
        ) from exc
    if result.returncode != 0:
        raise MaterializationError(
            "board validation failed: " + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MaterializationError("board validator did not emit JSON") from exc
    if not isinstance(report, dict) or report.get("valid") is not True:
        raise MaterializationError("board validator did not report valid=true")
    return report


def build_population(config: Mapping[str, Any]) -> dict[str, Any]:
    board_path = _relative_path(config.get("taskboard_json_path"), field="taskboard_json_path")
    source_path = _relative_path(
        config.get("source_reconciliation_manifest_path"),
        field="source_reconciliation_manifest_path",
    )
    stack_path = _relative_path(
        config.get("stack_compatibility_manifest_path"),
        field="stack_compatibility_manifest_path",
    )
    board = _load_object(board_path)
    source_generation = _source_generation(config)
    head = str(source_generation["ipfs_accelerate_py"]["head"])
    tree = str(source_generation["ipfs_accelerate_py"]["tree"])
    controls = {
        "board": _file_cid(board_path),
        "taskboard_markdown": _file_cid(_relative_path(config.get("taskboard_path"), field="taskboard_path")),
        "objectives": _file_cid(_relative_path(config.get("objectives_path"), field="objectives_path")),
        "plan": _file_cid(_relative_path(config.get("plan_path"), field="plan_path")),
        "source_manifest": _file_cid(source_path),
        "stack_manifest": _file_cid(stack_path),
        "config": _file_cid(CONFIG_PATH),
        "validator": _file_cid(_relative_path(config.get("validator_path"), field="validator_path")),
        "materializer": _file_cid(_relative_path(config.get("materializer_path"), field="materializer_path")),
        "materialization_attempt_history": _file_cid(
            _relative_path(
                config.get("materialization_attempt_history_path"),
                field="materialization_attempt_history_path",
            )
        ),
    }
    plan_root_cid = _cid(
        {
            "schema": "ExternalAgentFormalWorkPlanRoot@1",
            "plan_revision": board.get("plan_revision"),
            "board_cid": board.get("board_cid"),
            "controls": controls,
            "source_head": head,
            "repository_tree_id": tree,
            "source_generation_cid": source_generation["source_generation_cid"],
        }
    )
    raw_goals = board.get("goals")
    raw_tasks = board.get("tasks")
    initial_ids = board.get("initial_population_task_ids")
    if not isinstance(raw_goals, list) or not isinstance(raw_tasks, list) or not isinstance(initial_ids, list):
        raise MaterializationError("board goals/tasks/initial population are malformed")
    goal_cids = {
        str(goal["goal_id"]): _cid(
            {"schema": "EAAEFGoalIdentity@1", "goal": goal, "plan_root_cid": plan_root_cid}
        )
        for goal in raw_goals
        if isinstance(goal, Mapping)
    }
    goals: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    for ordinal, goal in enumerate(raw_goals, start=1):
        if not isinstance(goal, Mapping):
            raise MaterializationError("goal is not an object")
        goal_id = str(goal["goal_id"])
        parent = str(goal.get("parent_goal_id") or "")
        goals.append(
            {
                "goal_cid": goal_cids[goal_id],
                "goal_id": goal_id,
                "goal_alias": goal_id,
                "title": str(goal.get("title") or goal_id),
                "ordinal": ordinal,
                "status": "open",
                "objective_id": "objective:eaaef-root" if goal_id == "EAAEF-G000" else "",
                "objective_alias": "EAAEF-G000",
                "parent_goal_cid": goal_cids[parent] if parent else "",
                "priority": "P0",
                "body": dict(goal),
            }
        )
        if parent:
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[parent],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_parent",
                }
            )
        for dependency in goal.get("dependencies") or ():
            goal_edges.append(
                {
                    "parent_goal_cid": goal_cids[str(dependency)],
                    "child_goal_cid": goal_cids[goal_id],
                    "edge_kind": "goal_dependency",
                }
            )
    task_by_id = {
        str(task.get("stable_task_id") or ""): task
        for task in raw_tasks
        if isinstance(task, Mapping) and str(task.get("stable_task_id") or "")
    }
    normalized_initial_ids = [str(item) for item in initial_ids]
    if len(normalized_initial_ids) != len(set(normalized_initial_ids)):
        raise MaterializationError("initial task population contains duplicate identities")
    missing_initial = [task_id for task_id in normalized_initial_ids if task_id not in task_by_id]
    if missing_initial:
        raise MaterializationError(f"initial task population is missing tasks: {missing_initial}")
    selected = [task_by_id[task_id] for task_id in normalized_initial_ids]
    task_cids = {
        str(task["stable_task_id"]): _cid(
            {
                "schema": "EAAEFTaskIdentity@1",
                "task_spec_cid": task.get("task_spec_cid"),
                "plan_root_cid": plan_root_cid,
                "source_head": head,
                "repository_tree_id": tree,
            }
        )
        for task in selected
    }
    tasks: list[dict[str, Any]] = []
    for ordinal, task in enumerate(selected, start=1):
        task_id = str(task["stable_task_id"])
        dependencies = [str(item) for item in task.get("dependencies") or ()]
        if any(item not in task_cids for item in dependencies):
            raise MaterializationError(f"{task_id} has a dependency outside the initial population")
        execution_owned_files = task.get("execution_owned_files")
        if (
            not isinstance(execution_owned_files, list)
            or not execution_owned_files
            or any(not isinstance(item, str) or not item for item in execution_owned_files)
        ):
            raise MaterializationError(
                f"{task_id} has no canonical accelerator-root execution_owned_files"
            )
        raw_execution_validation = task.get("execution_validation")
        if not isinstance(raw_execution_validation, list) or not raw_execution_validation:
            raise MaterializationError(
                f"{task_id} has no canonical accelerator-root execution_validation"
            )
        execution_validation: list[dict[str, Any]] = []
        for validation_index, item in enumerate(raw_execution_validation):
            if not isinstance(item, Mapping):
                raise MaterializationError(
                    f"{task_id} execution_validation[{validation_index}] is not an object"
                )
            working_directory = str(item.get("working_directory") or "")
            raw_argv = item.get("argv")
            if (
                not working_directory
                or Path(working_directory).is_absolute()
                or ".." in Path(working_directory).parts
                or not isinstance(raw_argv, list)
                or not raw_argv
                or any(not isinstance(part, str) or not part for part in raw_argv)
            ):
                raise MaterializationError(
                    f"{task_id} execution_validation[{validation_index}] is not bounded cwd/argv"
                )
            execution_validation.append(
                {"working_directory": working_directory, "argv": list(raw_argv)}
            )
        body = dict(task)
        body.update(
            {
                "task_id": task_id,
                "task_alias": task_id,
                "base_revision": str(
                    ((task.get("source_revisions") or {}).get(task["owning_repository"]) or {}).get("commit")
                    or ""
                ),
                "base_repository_tree_id": str(
                    ((task.get("source_revisions") or {}).get(task["owning_repository"]) or {}).get("tree")
                    or ""
                ),
                "accepted_plan_root_cid": plan_root_cid,
                "completion": task.get("completion_mode"),
                "review_only": False,
                "predicted_files": list(execution_owned_files),
                "depends_on": dependencies,
            }
        )
        tasks.append(
            {
                **body,
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "goal_cid": goal_cids[str(task["subgoal_id"])],
                "plan_cid": plan_root_cid,
                "objective_id": "objective:eaaef-root",
                "ordinal": ordinal,
                "status": "todo",
                "priority": str(task.get("priority") or "P0"),
                "title": str(task.get("title") or task_id),
                "dependencies": [task_cids[item] for item in dependencies],
                "outputs": [
                    {
                        "path": str(path),
                        "effect_id": _cid({"task": task_id, "path": str(path)}),
                    }
                    for path in execution_owned_files
                ],
                "acceptance": [str(task.get("acceptance") or "")],
                "validations": execution_validation,
            }
        )
    ready_task_aliases = [
        str(task["task_alias"])
        for task in tasks
        if not list(task.get("dependencies") or ())
    ]
    initial_projection = config.get("initial_projection")
    expected_initial_projection = {
        "task_count": len(tasks),
        "goal_count": len(goals),
        "completed_task_ids": [],
        "ready_task_ids": ready_task_aliases,
        "terminal_bootstrap_task_id": "EAAEF-009",
        "future_task_count": len(raw_tasks) - len(tasks),
        "future_tasks_materialized": False,
    }
    if initial_projection != expected_initial_projection:
        raise MaterializationError(
            "scheduler initial_projection differs from the exact board bootstrap population"
        )
    population = {
        "schema": POPULATION_SCHEMA,
        "repository_tree_id": tree,
        "source_head": head,
        "source_generation": source_generation,
        "plan_root_cid": plan_root_cid,
        "controls": controls,
        "objectives": goals,
        "goal_edges": goal_edges,
        "plans": [
            {
                "plan_cid": plan_root_cid,
                "plan_alias": str(board["plan_revision"]),
                "goal_cid": goal_cids["EAAEF-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
                "body": {
                    "board_cid": board["board_cid"],
                    "future_population_rule": board["future_population_rule"],
                    "future_task_count": len(raw_tasks) - len(tasks),
                },
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
        "initial_task_aliases": normalized_initial_ids,
        "ready_task_aliases": ready_task_aliases,
        "initial_task_count": len(tasks),
        "goal_count": len(goals),
        "future_task_count": len(raw_tasks) - len(tasks),
    }
    population["population_cid"] = _cid(population)
    return population


def _read_only_connection(path: Path) -> Any:
    if not path.is_file():
        raise MaterializationError(f"authority database does not exist: {path}")
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise MaterializationError("DuckDB is unavailable") from exc
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        connect_duckdb_with_policy,
    )

    try:
        return connect_duckdb_with_policy(
            duckdb,
            path,
            read_only=True,
            configuration={"threads": 1, "memory_limit": "256MB"},
        )
    except Exception as exc:
        raise MaterializationError(f"unable to open authority read-only: {path}") from exc


def _control_schema_projection(path: Path) -> dict[str, Any]:
    """Project the installed operational profile through a read-only handle."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
        compute_schema_fingerprint,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
        BOOKKEEPING_TABLES,
        DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
        DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION,
        DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
        DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
        DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES,
        DATASETS_SEMANTIC_TRUTH_RELATIONS,
        DIAGNOSTIC_VIEWS,
        LEASE_IDENTITY_COLUMNS,
        TASK_IDENTITY_COLUMNS,
        load_datasets_authoritative_operational_catalog,
    )

    catalog = load_datasets_authoritative_operational_catalog()
    expected_migration = catalog.get(DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION)
    connection = _read_only_connection(path)
    try:
        relations = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        required = set(BOOKKEEPING_TABLES).union(
            DATASETS_AUTHORITATIVE_OPERATIONAL_TABLES,
            DIAGNOSTIC_VIEWS,
        )
        missing = sorted(required - relations)
        forbidden = sorted(relations.intersection(DATASETS_SEMANTIC_TRUTH_RELATIONS))
        if missing or forbidden:
            raise MaterializationError(
                "datasets-authoritative operational profile relation mismatch: "
                f"missing={missing}, forbidden={forbidden}"
            )
        migration = connection.execute(
            "SELECT migration_id, checksum FROM schema_migrations WHERE version = ?",
            [DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION],
        ).fetchone()
        if migration is None or tuple(str(value) for value in migration) != (
            DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
            expected_migration.checksum,
        ):
            raise MaterializationError("operational-profile migration identity/checksum mismatch")
        contract = connection.execute(
            "SELECT payload_schema, description FROM schema_contracts "
            "WHERE contract_id = "
            "'contract:DatasetsAuthoritativeOperationalControlPlane@1'"
        ).fetchone()
        if contract is None:
            raise MaterializationError("operational-profile authority contract is missing")
        payload_schema, description = (str(contract[0]), str(contract[1]))
        if (
            payload_schema != DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA
            or "operational" not in description.lower()
            or "ipfs_datasets_py" not in description
        ):
            raise MaterializationError("operational-profile authority contract drifted")

        def columns(table: str) -> set[str]:
            return {
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            }

        missing_task_columns = sorted(set(TASK_IDENTITY_COLUMNS) - columns("tasks"))
        missing_lease_columns = sorted(set(LEASE_IDENTITY_COLUMNS) - columns("leases"))
        if missing_task_columns or missing_lease_columns:
            raise MaterializationError(
                "operational-profile identity columns are missing: "
                f"tasks={missing_task_columns}, leases={missing_lease_columns}"
            )
        projection = {
            "valid": True,
            "database_path": str(path),
            "profile_id": DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
            "profile_schema": DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
            "catalog_fingerprint": catalog.fingerprint(),
            "migration_id": DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_ID,
            "migration_checksum": expected_migration.checksum,
            "schema_fingerprint": compute_schema_fingerprint(connection),
            "required_relations": sorted(required),
            "forbidden_relations": forbidden,
            "task_identity_columns": sorted(TASK_IDENTITY_COLUMNS),
            "lease_identity_columns": sorted(LEASE_IDENTITY_COLUMNS),
            "authority_contract": {
                "payload_schema": payload_schema,
                "operational_authority": "ipfs_accelerate_py",
                "semantic_and_proof_authority": "ipfs_datasets_py",
            },
            "connection_mode": "read_only",
        }
    finally:
        connection.close()
    projection["projection_root"] = _cid(projection)
    return projection


def _eaaef_operational_profile_projection(path: Path) -> dict[str, Any]:
    """Verify the exact @2 owner schema without acquiring runtime authority."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_bootstrap_daemon_gateway import (
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
        eaaef_operation_vocabulary_cid,
        verify_eaaef_operational_schema,
    )

    operation_vocabulary_cid = eaaef_operation_vocabulary_cid(
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS
    )
    try:
        verified = dict(
            verify_eaaef_operational_schema(
                path,
                operation_vocabulary_cid=operation_vocabulary_cid,
            )
        )
    except Exception as exc:
        raise MaterializationError(
            "EAAEF operational profile @2 is absent or drifted"
        ) from exc
    if (
        verified.get("valid") is not True
        or verified.get("operation_vocabulary_cid")
        != operation_vocabulary_cid
    ):
        raise MaterializationError(
            "EAAEF operational profile @2 verification is not exact"
        )
    return verified


def _borrowed_transaction_handler_source_evidence(
    command_fabric_profile: Mapping[str, Any],
    *,
    operation_vocabulary_cid: str,
) -> dict[str, Any]:
    """Bind the exact 31-op source implementation without minting authority."""

    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_borrowed_transaction import (
        EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE,
        EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS,
        eaaef_bootstrap_handler_source_evidence,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
        eaaef_operation_vocabulary_cid,
    )

    evidence = dict(
        eaaef_bootstrap_handler_source_evidence(
            board_namespace=str(command_fabric_profile.get("board_namespace") or ""),
            shard_id=str(command_fabric_profile.get("shard_id") or ""),
        )
    )
    operations = evidence.get("operations")
    runtime_authority_fields = evidence.get("runtime_authority_fields")
    if (
        evidence.get("interface") != EAAEF_BORROWED_TRANSACTION_HANDLER_INTERFACE
        or evidence.get("qualification_status")
        != EAAEF_BORROWED_TRANSACTION_QUALIFICATION_STATUS
        or evidence.get("operation_count") != 31
        or not isinstance(operations, list)
        or len(operations) != 31
        or operations != sorted(set(str(item) for item in operations))
        or eaaef_operation_vocabulary_cid(operations) != operation_vocabulary_cid
        or not isinstance(runtime_authority_fields, list)
        or "command_principal_did" not in runtime_authority_fields
        or evidence.get("owns_transaction_lifecycle") is not False
        or evidence.get("opens_database") is not False
        or evidence.get("performs_external_effects") is not False
        or evidence.get("accepts_operation_callback") is not False
        or evidence.get("production_admitted") is not False
        or evidence.get("handler_source_evidence_cid") != content_identity(
            {
                key: value
                for key, value in evidence.items()
                if key != "handler_source_evidence_cid"
            }
        )
    ):
        raise MaterializationError(
            "EAAEF borrowed-transaction handler source evidence is invalid"
        )
    return evidence


def _control_projection(path: Path) -> dict[str, Any]:
    connection = _read_only_connection(path)
    try:
        objectives = [
            {
                "objective_id": str(row[0]),
                "objective_alias": str(row[1]),
                "parent_objective_id": str(row[2]),
                "title": str(row[3]),
                "status": str(row[4]),
                "priority": str(row[5]),
                "revision": int(row[6]),
                "body": json.loads(row[7]),
            }
            for row in connection.execute(
                "SELECT objective_id, objective_alias, parent_objective_id, title, "
                "status, priority, revision, body_json FROM objectives "
                "ORDER BY objective_id"
            ).fetchall()
        ]
        tasks = [
            {
                "task_cid": str(row[0]),
                "task_alias": str(row[1]),
                "goal_cid": str(row[2]),
                "plan_cid": str(row[3]),
                "objective_id": str(row[4]),
                "ordinal": int(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "priority": str(row[8]),
                "identity": json.loads(row[9]),
                "body": json.loads(row[10]),
            }
            for row in connection.execute(
                "SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id, ordinal, "
                "status, revision, priority, identity_json, body_json "
                "FROM tasks ORDER BY ordinal"
            ).fetchall()
        ]
        goals = [
            {
                "goal_cid": str(row[0]),
                "goal_alias": str(row[1]),
                "objective_id": str(row[2]),
                "parent_goal_cid": str(row[3]),
                "ordinal": int(row[4]),
                "title": str(row[5]),
                "status": str(row[6]),
                "revision": int(row[7]),
                "body": json.loads(row[8]),
            }
            for row in connection.execute(
                "SELECT goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal, "
                "title, status, revision, body_json "
                "FROM goals ORDER BY ordinal"
            ).fetchall()
        ]
        dependencies = [
            {"task_cid": str(row[0]), "dependency_task_cid": str(row[1]), "kind": str(row[2])}
            for row in connection.execute(
                "SELECT task_cid, dependency_task_cid, kind FROM task_dependencies "
                "ORDER BY task_cid, dependency_task_cid, kind"
            ).fetchall()
        ]
        goal_edges = [
            {
                "parent_goal_cid": str(row[0]),
                "child_goal_cid": str(row[1]),
                "edge_kind": str(row[2]),
            }
            for row in connection.execute(
                "SELECT parent_goal_cid, child_goal_cid, edge_kind FROM goal_edges "
                "ORDER BY parent_goal_cid, child_goal_cid, edge_kind"
            ).fetchall()
        ]
        task_outputs = [
            {
                "task_cid": str(row[0]),
                "ordinal": int(row[1]),
                "path": str(row[2]),
                "effect": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, ordinal, path, effect_json FROM task_outputs "
                "ORDER BY task_cid, ordinal"
            ).fetchall()
        ]
        task_acceptance = [
            {
                "task_cid": str(row[0]),
                "ordinal": int(row[1]),
                "criterion": str(row[2]),
                "evidence_policy": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, ordinal, criterion, evidence_policy_json "
                "FROM task_acceptance ORDER BY task_cid, ordinal"
            ).fetchall()
        ]
        task_validations = [
            {
                "task_cid": str(row[0]),
                "ordinal": int(row[1]),
                "argv": json.loads(row[2]),
                "policy": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, ordinal, argv_json, policy_json "
                "FROM task_validations ORDER BY task_cid, ordinal"
            ).fetchall()
        ]
        plans = [
            {
                "plan_cid": str(row[0]),
                "goal_cid": str(row[1]),
                "plan_alias": str(row[2]),
                "status": str(row[3]),
                "revision": int(row[4]),
                "body": json.loads(row[5]),
            }
            for row in connection.execute(
                "SELECT plan_cid, goal_cid, plan_alias, status, revision, body_json "
                "FROM plans ORDER BY plan_cid"
            ).fetchall()
        ]
        objective_revisions = [
            {
                "objective_id": str(row[0]),
                "revision": int(row[1]),
                "status": str(row[2]),
                "body": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT objective_id, revision, status, body_json "
                "FROM objective_revisions ORDER BY objective_id, revision"
            ).fetchall()
        ]
        plan_revisions = [
            {
                "plan_cid": str(row[0]),
                "revision": int(row[1]),
                "body": json.loads(row[2]),
            }
            for row in connection.execute(
                "SELECT plan_cid, revision, body_json "
                "FROM plan_revisions ORDER BY plan_cid, revision"
            ).fetchall()
        ]
        task_revisions = [
            {
                "task_cid": str(row[0]),
                "revision": int(row[1]),
                "status": str(row[2]),
                "body": json.loads(row[3]),
            }
            for row in connection.execute(
                "SELECT task_cid, revision, status, body_json "
                "FROM task_revisions ORDER BY task_cid, revision"
            ).fetchall()
        ]
        from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
            content_identity,
        )

        event_watermark = int(
            connection.execute(
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        )
        intent_material = {
            "objectives": len(objectives),
            "goals": sorted(
                (
                    {
                        "goal_cid": item["goal_cid"],
                        "status": item["status"],
                        "revision": item["revision"],
                    }
                    for item in goals
                ),
                key=lambda item: item["goal_cid"],
            ),
            "plans": sorted(
                (
                    {
                        "plan_cid": item["plan_cid"],
                        "status": item["status"],
                        "revision": item["revision"],
                    }
                    for item in plans
                ),
                key=lambda item: item["plan_cid"],
            ),
            "tasks": sorted(
                (
                    {
                        "task_cid": item["task_cid"],
                        "status": item["status"],
                        "revision": item["revision"],
                    }
                    for item in tasks
                ),
                key=lambda item: item["task_cid"],
            ),
            "dependency_count": len(dependencies),
            "event_watermark": event_watermark,
        }
        intent_snapshot = {
            "objective_count": len(objectives),
            "goal_count": len(goals),
            "plan_count": len(plans),
            "task_count": len(tasks),
            "dependency_count": len(dependencies),
            "event_watermark": event_watermark,
            "goal_cids": sorted(item["goal_cid"] for item in goals),
            "task_cids": sorted(item["task_cid"] for item in tasks),
            "projection_cid": content_identity(intent_material),
        }
        # Bind every bootstrap-owned control relation, including revision and
        # materialization history that the ergonomic projections above omit.
        # The table allowlist is closed and identifiers are not caller input.
        exact_relation_names = (
            "control_plane_metadata",
            "schema_migrations",
            "schema_migration_attempts",
            "schema_contracts",
            "store_generations",
            "objectives",
            "objective_revisions",
            "goals",
            "goal_edges",
            "plans",
            "plan_revisions",
            "planning_decisions",
            "plan_candidates",
            "tasks",
            "task_revisions",
            "task_dependencies",
            "task_outputs",
            "task_acceptance",
            "task_validations",
            "artifacts",
            "completion_receipts",
        )
        exact_relations: dict[str, Any] = {}
        for table_name in exact_relation_names:
            columns = [
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            ]
            rows = [
                [
                    value
                    if value is None or isinstance(value, (bool, int, float, str))
                    else str(value)
                    for value in row
                ]
                for row in connection.execute(
                    f'SELECT * FROM "{table_name}" ORDER BY ALL'
                ).fetchall()
            ]
            exact_relations[table_name] = {"columns": columns, "rows": rows}
    finally:
        connection.close()
    projection = {
        "objectives": objectives,
        "goals": goals,
        "goal_edges": goal_edges,
        "plans": plans,
        "tasks": tasks,
        "dependencies": dependencies,
        "task_outputs": task_outputs,
        "task_acceptance": task_acceptance,
        "task_validations": task_validations,
        "objective_revisions": objective_revisions,
        "plan_revisions": plan_revisions,
        "task_revisions": task_revisions,
        "intent_snapshot": intent_snapshot,
        "exact_relations": exact_relations,
    }
    projection["projection_root"] = _cid(projection)
    return projection


def _expected_population_projection(population: Mapping[str, Any]) -> dict[str, Any]:
    """Project the admitted input through the canonical repository boundary.

    This is intentionally independent of the rows read back from DuckDB.  It
    mirrors the documented DatabaseTaskSource/IntentRepository normalization
    so a buggy or fault-injected initial write cannot become its own oracle and
    be sealed merely because the post-write projection is internally stable.
    """

    source_goals = [
        dict(item)
        for item in population.get("objectives") or ()
        if isinstance(item, Mapping)
    ]
    objectives: list[dict[str, Any]] = []
    goals: list[dict[str, Any]] = []
    for index, item in enumerate(source_goals):
        objective_id = str(item.get("objective_id") or "")
        if objective_id:
            objectives.append(
                {
                    "objective_id": objective_id,
                    "objective_alias": str(item.get("objective_alias") or objective_id),
                    "parent_objective_id": "",
                    "title": str(item.get("title") or objective_id),
                    "status": str(item.get("status") or "open").lower(),
                    "priority": str(item.get("priority") or "P2"),
                    "revision": 1,
                    "body": {
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "objective_id",
                            "objective_alias",
                            "title",
                            "status",
                            "priority",
                        }
                    },
                }
            )
        goal_cid = str(item.get("goal_cid") or item.get("goal_id") or f"goal:cid:{index + 1}")
        goals.append(
            {
                "goal_cid": goal_cid,
                "goal_alias": str(item.get("goal_alias") or item.get("goal_id") or goal_cid),
                "objective_id": objective_id,
                "parent_goal_cid": str(item.get("parent_goal_cid") or ""),
                "ordinal": int(item.get("ordinal") or index + 1),
                "title": str(item.get("title") or item.get("goal_alias") or goal_cid),
                "status": str(item.get("status") or "open").lower(),
                "revision": 1,
                "body": {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "goal_cid",
                        "goal_id",
                        "goal_alias",
                        "title",
                        "status",
                        "ordinal",
                        "objective_id",
                    }
                },
            }
        )

    goal_edges = sorted(
        (
            {
                "parent_goal_cid": str(item.get("parent_goal_cid") or item.get("parent") or ""),
                "child_goal_cid": str(item.get("child_goal_cid") or item.get("child") or ""),
                "edge_kind": str(item.get("edge_kind") or "goal_dependency"),
            }
            for item in population.get("goal_edges") or ()
            if isinstance(item, Mapping)
        ),
        key=lambda item: (
            item["parent_goal_cid"],
            item["child_goal_cid"],
            item["edge_kind"],
        ),
    )
    plans = sorted(
        (
            {
                "plan_cid": str(item.get("plan_cid") or item.get("plan_id") or ""),
                "goal_cid": str(item.get("goal_cid") or ""),
                "plan_alias": str(item.get("plan_alias") or item.get("alias") or item.get("plan_cid") or ""),
                "status": str(item.get("status") or "active").lower(),
                "revision": 1,
                "body": dict(item),
            }
            for item in population.get("plans") or ()
            if isinstance(item, Mapping)
        ),
        key=lambda item: item["plan_cid"],
    )

    tasks: list[dict[str, Any]] = []
    dependencies: list[dict[str, str]] = []
    task_outputs: list[dict[str, Any]] = []
    task_acceptance: list[dict[str, Any]] = []
    task_validations: list[dict[str, Any]] = []
    tree_id = str(population.get("repository_tree_id") or "tree:unknown")
    source_tasks = [
        dict(item)
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    ]
    task_cids_by_alias = {
        str(item.get("task_id") or item.get("task_alias") or item.get("alias") or task_cid): task_cid
        for index, item in enumerate(source_tasks)
        for task_cid in (
            str(item.get("task_cid") or item.get("cid") or f"task:cid:{index + 1}"),
        )
    }
    for index, raw_task in enumerate(source_tasks):
        if not isinstance(raw_task, Mapping):
            continue
        item = dict(raw_task)
        task_cid = str(item.get("task_cid") or item.get("cid") or f"task:cid:{index + 1}")
        task_alias = str(item.get("task_id") or item.get("task_alias") or item.get("alias") or task_cid)
        tasks.append(
            {
                "task_cid": task_cid,
                "task_alias": task_alias,
                "goal_cid": str(item.get("goal_cid") or item.get("goal_id") or ""),
                "plan_cid": str(item.get("plan_cid") or population.get("plan_root_cid") or ""),
                "objective_id": str(item.get("objective_id") or ""),
                "ordinal": int(item.get("ordinal") or index + 1),
                "status": str(item.get("status") or "ready").lower(),
                "revision": 1,
                "priority": str(item.get("priority") or "P2"),
                "identity": {
                    "repository_tree_id": tree_id,
                    "task_alias": task_alias,
                    "task_cid": task_cid,
                },
                "body": {
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "task_cid",
                        "task_id",
                        "task_alias",
                        "cid",
                        "goal_cid",
                        "goal_id",
                        "depends_on",
                        "dependencies",
                        "effects",
                        "outputs",
                        "acceptance_criteria",
                        "acceptance",
                        "validation_commands",
                        "validations",
                        "status",
                        "priority",
                        "ordinal",
                        "plan_cid",
                        "objective_id",
                    }
                },
            }
        )
        for dependency in item.get("depends_on") or item.get("dependencies") or ():
            dependency_text = str(dependency)
            dependencies.append(
                {
                    "task_cid": task_cid,
                    "dependency_task_cid": task_cids_by_alias.get(
                        dependency_text, dependency_text
                    ),
                    "kind": "depends_on",
                }
            )
        for ordinal, output in enumerate(item.get("effects") or item.get("outputs") or ()):
            if not isinstance(output, Mapping):
                continue
            effect = dict(output)
            task_outputs.append(
                {
                    "task_cid": task_cid,
                    "ordinal": ordinal,
                    "path": str(effect.get("path") or effect.get("effect_id") or f"output:{ordinal}"),
                    "effect": effect,
                }
            )
        for ordinal, acceptance in enumerate(
            item.get("acceptance_criteria") or item.get("acceptance") or ()
        ):
            if isinstance(acceptance, str):
                criterion = acceptance.strip()
                policy: dict[str, Any] = {"criterion": criterion}
            elif isinstance(acceptance, Mapping):
                policy = dict(acceptance)
                criterion = str(
                    policy.get("criterion")
                    or policy.get("statement")
                    or policy.get("criterion_key")
                    or f"criterion:{ordinal}"
                ).strip()
            else:
                continue
            task_acceptance.append(
                {
                    "task_cid": task_cid,
                    "ordinal": ordinal,
                    "criterion": criterion,
                    "evidence_policy": policy,
                }
            )
        for ordinal, validation in enumerate(
            item.get("validation_commands") or item.get("validations") or ()
        ):
            if isinstance(validation, str):
                argv = [validation]
                policy = {}
            elif isinstance(validation, Mapping):
                validation_map = dict(validation)
                raw_argv = validation_map.get("argv") or validation_map.get("validation_commands")
                if isinstance(raw_argv, str):
                    argv = [raw_argv]
                elif isinstance(raw_argv, list):
                    argv = [str(part) for part in raw_argv]
                else:
                    argv = [str(validation_map.get("command") or f"validation:{ordinal}")]
                policy = {
                    key: value
                    for key, value in validation_map.items()
                    if key not in {"argv", "validation_commands", "command"}
                }
            elif isinstance(validation, list):
                argv = [str(part) for part in validation]
                policy = {}
            else:
                continue
            task_validations.append(
                {
                    "task_cid": task_cid,
                    "ordinal": ordinal,
                    "argv": argv,
                    "policy": policy,
                }
            )

    sorted_objectives = sorted(objectives, key=lambda item: item["objective_id"])
    sorted_plans = plans
    sorted_tasks = sorted(tasks, key=lambda item: item["ordinal"])
    return {
        "objectives": sorted_objectives,
        "goals": sorted(goals, key=lambda item: item["ordinal"]),
        "goal_edges": goal_edges,
        "plans": sorted_plans,
        "tasks": sorted_tasks,
        "dependencies": sorted(
            dependencies,
            key=lambda item: (item["task_cid"], item["dependency_task_cid"], item["kind"]),
        ),
        "task_outputs": sorted(task_outputs, key=lambda item: (item["task_cid"], item["ordinal"])),
        "task_acceptance": sorted(task_acceptance, key=lambda item: (item["task_cid"], item["ordinal"])),
        "task_validations": sorted(task_validations, key=lambda item: (item["task_cid"], item["ordinal"])),
        "objective_revisions": [
            {
                "objective_id": item["objective_id"],
                "revision": item["revision"],
                "status": item["status"],
                "body": item["body"],
            }
            for item in sorted_objectives
        ],
        "plan_revisions": [
            {
                "plan_cid": item["plan_cid"],
                "revision": item["revision"],
                "body": item["body"],
            }
            for item in sorted_plans
        ],
        "task_revisions": sorted(
            (
                {
                    "task_cid": item["task_cid"],
                    "revision": item["revision"],
                    "status": item["status"],
                    "body": item["body"],
                }
                for item in sorted_tasks
            ),
            key=lambda item: (item["task_cid"], item["revision"]),
        ),
    }


def _assert_population_equivalent(
    population: Mapping[str, Any], control: Mapping[str, Any]
) -> None:
    from ipfs_accelerate_py.agent_supervisor.validation.control_plane_identity_recovery import (
        identity_control_projection,
    )

    expected = identity_control_projection(_expected_population_projection(population))
    observed = identity_control_projection(
        {key: control.get(key) for key in expected}
    )
    if observed != expected:
        raise MaterializationError(
            "materialized control population differs from the admitted board projection"
        )


def _assert_database_materialization_equivalent(
    database_receipt: Any,
    population: Mapping[str, Any],
    control: Mapping[str, Any],
) -> None:
    """Verify the native receipt independently against source and read-only rows."""

    if not isinstance(database_receipt, Mapping):
        raise MaterializationError("database materialization receipt is not an object")
    wrapper_fields = {
        "task_source",
        "registered_task_cids",
        "bootstrap_completed_task_cids",
    }
    if set(database_receipt) != wrapper_fields:
        raise MaterializationError("database materialization receipt wrapper is not canonical")
    task_source_receipt = database_receipt.get("task_source")
    if not isinstance(task_source_receipt, Mapping):
        raise MaterializationError("database task-source receipt is not an object")
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DATABASE_TASK_SOURCE_SCHEMA,
    )

    snapshot = control.get("intent_snapshot")
    if not isinstance(snapshot, Mapping):
        raise MaterializationError("control intent snapshot is missing")
    expected_task_cids = [
        str(item.get("task_cid") or "")
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
    ]
    expected_goal_cids = sorted(
        str(item.get("goal_cid") or "")
        for item in population.get("objectives") or ()
        if isinstance(item, Mapping)
    )
    task_source_fields = {
        "schema",
        "plan_root_cid",
        "repository_tree_id",
        "projection_cid",
        "task_count",
        "goal_count",
        "goal_edge_count",
        "plan_count",
        "event_watermark",
        "task_cids",
    }
    if set(task_source_receipt) != task_source_fields:
        raise MaterializationError("database task-source receipt shape is not canonical")
    expected = {
        "schema": DATABASE_TASK_SOURCE_SCHEMA,
        "plan_root_cid": str(population.get("plan_root_cid") or ""),
        "repository_tree_id": str(population.get("repository_tree_id") or ""),
        "projection_cid": str(snapshot.get("projection_cid") or ""),
        "task_count": len(expected_task_cids),
        "goal_count": len(expected_goal_cids),
        "goal_edge_count": len(population.get("goal_edges") or ()),
        "plan_count": len(population.get("plans") or ()),
        "event_watermark": int(snapshot.get("event_watermark") or 0),
        "task_cids": expected_task_cids,
    }
    observed = {key: task_source_receipt.get(key) for key in expected}
    if observed != expected:
        raise MaterializationError(
            "database materialization receipt differs from source and control authority"
        )
    registered_task_cids = database_receipt.get("registered_task_cids")
    if not isinstance(registered_task_cids, list) or registered_task_cids != expected_task_cids:
        raise MaterializationError(
            "database materialization registered task identities differ from source"
        )
    bootstrap_completed_task_cids = database_receipt.get(
        "bootstrap_completed_task_cids"
    )
    if not isinstance(bootstrap_completed_task_cids, list):
        raise MaterializationError(
            "database materialization bootstrap-completed identities are not a list"
        )
    expected_bootstrap_completed_task_cids = [
        str(item.get("task_cid") or "")
        for item in population.get("tasks") or ()
        if isinstance(item, Mapping)
        and str(item.get("status") or "").strip().lower()
        in {"completed", "complete", "done"}
    ]
    if bootstrap_completed_task_cids != expected_bootstrap_completed_task_cids:
        raise MaterializationError(
            "database materialization bootstrap-completed task identities differ "
            "from source"
        )
    if list(snapshot.get("task_cids") or ()) != sorted(expected_task_cids):
        raise MaterializationError("control intent snapshot task identities differ from source")
    if list(snapshot.get("goal_cids") or ()) != expected_goal_cids:
        raise MaterializationError("control intent snapshot goal identities differ from source")


def _execution_projection(path: Path) -> dict[str, Any]:
    connection = _read_only_connection(path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        tracked = [
            name
            for name in (
                "database_task_attempts",
                "attempt_phases",
                "provider_call_intents",
                "effect_claims",
                "validation_intents",
            )
            if name in tables
        ]
        row_counts = {
            name: int(connection.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            for name in tracked
        }
    finally:
        connection.close()
    projection = {"tracked_tables": tracked, "row_counts": row_counts}
    projection["projection_root"] = _cid(projection)
    return projection


def _write_json_immutable(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        os.fchmod(fd, 0o600)
        data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("short receipt write")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = -1
        try:
            # A same-directory hard link publishes the fully-fsynced bytes
            # atomically while retaining O_EXCL-style no-overwrite semantics.
            # Unlike os.replace(), a racing writer can never replace an
            # already-published immutable claim or receipt.
            os.link(temp, path)
        except FileExistsError as exc:
            raise MaterializationError(
                f"refusing to overwrite immutable record {path.relative_to(ROOT)}"
            ) from exc
        temp.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temp.unlink(missing_ok=True)


def materialize(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime_binding = _validated_runtime_binding(config)
    runtime_binding_cid = _cid(runtime_binding)
    materialization_invocation = _validated_runtime_invocation(
        runtime_binding, "materialize"
    )
    _assert_clean()
    validation = _validate_board(runtime_binding)
    population = build_population(config)
    database_program_bindings = _database_program_bindings(config)
    paths = _paths(config)
    claim_path = _claim_path(config)
    receipt_path = _receipt_path(config)
    existing = [path for path in _namespace_artifacts(config) if path.exists()]
    if existing:
        raise MaterializationError(
            "refusing to overwrite existing bootstrap namespace: "
            + ", ".join(path.relative_to(ROOT).as_posix() for path in existing)
        )
    namespace_claim: dict[str, Any] = {
        "schema": NAMESPACE_CLAIM_SCHEMA,
        "population_cid": population["population_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "source_head": population["source_head"],
        "source_tree": population["repository_tree_id"],
        "source_generation_cid": population["source_generation"]["source_generation_cid"],
        "runtime_binding": runtime_binding,
        "runtime_binding_cid": runtime_binding_cid,
        "materialization_invocation": materialization_invocation,
        "store_generation": str(
            (config.get("bootstrap_database_program") or {}).get(
                "store_generation"
            )
            or ""
        ),
        "database_program_bindings": database_program_bindings,
        "database_paths": {
            name: path.relative_to(ROOT).as_posix() for name, path in paths.items()
        },
        "maximum_writer_processes": 1,
        "partial_effect_policy": (
            "preserve claim and every created file; advance to a new explicit "
            "store generation after any failed attempt"
        ),
        "process_started": False,
    }
    namespace_claim["claim_cid"] = _cid(namespace_claim)
    if set(namespace_claim) != NAMESPACE_CLAIM_FIELDS:
        raise MaterializationError("namespace claim @2 shape is not canonical")
    _write_json_immutable(claim_path, namespace_claim)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
            verify_datasets_authoritative_operational_schema,
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.eaaef_operational_schema import (
            install_eaaef_operational_schema,
        )
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
            DatabaseImplementationDaemon,
        )

        schema_install = install_eaaef_operational_schema(
            paths["control"],
            application_version="0.0.45",
            tool_version=str(
                (runtime_binding.get("duckdb") or {}).get("module_version") or ""
            ),
            owner_id="eaaef-materializer:embedded-single-writer",
        )
        prior_revision = os.environ.get("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION")
        os.environ["IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"] = str(
            (config.get("bootstrap_database_program") or {}).get(
                "schema_revision"
            )
        )
        daemon = None
        try:
            daemon = DatabaseImplementationDaemon(
                database_path=paths["control"],
                coordination_path=paths["coordination"],
                execution_path=paths["execution"],
                owner_session_id="eaaef-materializer:embedded-single-writer",
                authority_mode="embedded",
                task_source_kind="duckdb",
                install_schema=False,
            )
            database_receipt = daemon.materialize_population(
                population,
                repository_tree_id=str(population["repository_tree_id"]),
                plan_root_cid=str(population["plan_root_cid"]),
            )
        finally:
            if daemon is not None:
                daemon.close()
            if prior_revision is None:
                os.environ.pop("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION", None)
            else:
                os.environ["IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION"] = prior_revision
        schema_verification = verify_datasets_authoritative_operational_schema(paths["control"])
        operational_profile_verification = _eaaef_operational_profile_projection(
            paths["control"]
        )
        handler_source_evidence = _borrowed_transaction_handler_source_evidence(
            database_program_bindings["operational_command_fabric"],
            operation_vocabulary_cid=operational_profile_verification[
                "operation_vocabulary_cid"
            ],
        )
        control_schema = _control_schema_projection(paths["control"])
        control = _control_projection(paths["control"])
        from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
            read_coordination_registry_projection,
        )

        coordination = read_coordination_registry_projection(paths["coordination"])
        execution = _execution_projection(paths["execution"])
        _assert_population_equivalent(population, control)
        _assert_database_materialization_equivalent(
            database_receipt, population, control
        )
        expected_aliases = list(population["task_cids_by_alias"])
        if [item["task_alias"] for item in control["tasks"]] != expected_aliases:
            raise MaterializationError("control task aliases differ from initial population")
        if any(item["status"] != "todo" for item in control["tasks"]):
            raise MaterializationError("fresh control tasks are not all todo")
        if any(value != 0 for value in execution["row_counts"].values()):
            raise MaterializationError("fresh execution store contains attempt/effect history")
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "namespace_claim_cid": namespace_claim["claim_cid"],
            "authority_mode": "embedded",
            "maximum_writer_processes": 1,
            "continuous_quack_authority": False,
            "ducklake_authority": False,
            "board_validation": validation,
            "population_cid": population["population_cid"],
            "plan_root_cid": population["plan_root_cid"],
            "source_head": population["source_head"],
            "source_tree": population["repository_tree_id"],
            "source_generation": population["source_generation"],
            "runtime_binding": runtime_binding,
            "runtime_binding_cid": runtime_binding_cid,
            "materialization_invocation": materialization_invocation,
            "controls": population["controls"],
            "database_paths": dict(namespace_claim["database_paths"]),
            "database_program_bindings": database_program_bindings,
            "schema_install": schema_install.to_dict(),
            "schema_verification": dict(schema_verification),
            "operation_vocabulary_cid": operational_profile_verification[
                "operation_vocabulary_cid"
            ],
            "operational_profile_verification": operational_profile_verification,
            "borrowed_transaction_handler_source_evidence": (
                handler_source_evidence
            ),
            "control_schema_projection": control_schema,
            "database_materialization": dict(database_receipt),
            "control_projection": control,
            "coordination_projection": coordination,
            "execution_projection": execution,
            "ready_task_aliases": list(population["ready_task_aliases"]),
            "process_started": False,
        }
        receipt["receipt_cid"] = _cid(receipt)
        if set(receipt) != MATERIALIZATION_RECEIPT_FIELDS:
            raise MaterializationError("materialization receipt @2 shape is not canonical")
        _write_json_immutable(receipt_path, receipt)
        return receipt
    except Exception as exc:
        if isinstance(exc, MaterializationError):
            detail = str(exc)
        else:
            detail = f"{type(exc).__name__}: {exc}"
        raise MaterializationError(
            "bootstrap namespace claim is immutable and partial effects are preserved; "
            "advance to a new explicit store generation after review; "
            f"claim={claim_path.relative_to(ROOT)}, failure={detail}"
        ) from exc


def _overlay_projection_path(config: Mapping[str, Any]) -> Path:
    return _paths(config)["control"].parent / "live/state/task-status-projection.json"


def _restore_overlay_on_control(
    control_path: Path, overlay: Mapping[str, str]
) -> int:
    """Replay completed alias statuses onto a freshly materialized catalog."""

    if not overlay or not control_path.is_file():
        return 0
    from datetime import datetime, timezone

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        connect_duckdb_with_policy,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.control_plane_identity_recovery import (
        CAS_TASK_STATUS_SQL,
        restore_overlay_cas_parameters,
    )

    import duckdb

    connection = connect_duckdb_with_policy(
        duckdb,
        control_path,
        read_only=False,
        configuration={"threads": 1, "memory_limit": "256MB"},
    )
    try:
        rows = [
            {
                "task_cid": str(row[0]),
                "task_alias": str(row[1]),
                "status": str(row[2]),
                "revision": int(row[3] or 0),
            }
            for row in connection.execute(
                "SELECT task_cid, task_alias, status, revision FROM tasks"
            ).fetchall()
        ]
        updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        restored = 0
        for parameters in restore_overlay_cas_parameters(
            live_rows=rows,
            overlay_statuses=overlay,
            updated_at=updated_at,
        ):
            connection.execute(CAS_TASK_STATUS_SQL, list(parameters))
            restored += 1
        return restored
    finally:
        connection.close()


def materialize_with_recovery(
    config: Mapping[str, Any],
    *,
    materialize_fn: Callable[..., dict[str, Any]] | None = None,
    max_recoveries: int = MAX_GENERATION_RECOVERIES,
) -> dict[str, Any]:
    """Materialize, advancing a failed or stale namespace without overwriting it."""

    from ipfs_accelerate_py.agent_supervisor.validation.control_plane_identity_recovery import (
        snapshot_overlay_alias_status,
    )

    actor = materialize_fn or materialize
    recoveries: list[dict[str, Any]] = []
    working = _active_config(config)
    configured = _configured_generation(config)
    overlay = snapshot_overlay_alias_status(_overlay_projection_path(working))
    prior_cursor = _read_generation_cursor()
    for attempt in range(max_recoveries + 1):
        state = _namespace_state(working)
        if state == "materialized":
            receipt = dict(_load_object(_receipt_path(working)))
            if recoveries:
                receipt["generation_recoveries"] = recoveries
            return receipt
        if state in {"failed_partial", "stale_materialized"}:
            if attempt >= max_recoveries:
                raise MaterializationError(
                    "store generation recovery budget exhausted; "
                    f"state={state} generation={_configured_generation(working)}"
                )
            current = _configured_generation(working)
            nxt = _successor_generation(current)
            _write_generation_cursor(
                {
                    "configured_generation": configured,
                    "active_generation": nxt,
                    "superseded_generation": current,
                    "namespace_state": state,
                    "recovery_index": attempt + 1,
                }
            )
            recoveries.append(
                {
                    "from_generation": current,
                    "to_generation": nxt,
                    "namespace_state": state,
                }
            )
            working = _rewrite_generation(working, current, nxt)
            continue
        try:
            receipt = dict(actor(working))
            if overlay:
                receipt["overlay_restored"] = _restore_overlay_on_control(
                    _paths(working)["control"], overlay
                )
                receipt["overlay_preserved"] = True
        except MaterializationError as exc:
            if (
                "advance to a new explicit store generation" not in str(exc)
                and _namespace_state(working) != "failed_partial"
            ):
                if prior_cursor is not None:
                    _write_generation_cursor(prior_cursor)
                raise
            if attempt >= max_recoveries:
                raise
            current = _configured_generation(working)
            nxt = _successor_generation(current)
            _write_generation_cursor(
                {
                    "configured_generation": configured,
                    "active_generation": nxt,
                    "superseded_generation": current,
                    "namespace_state": "failed_partial",
                    "recovery_index": attempt + 1,
                    "failure": str(exc),
                }
            )
            recoveries.append(
                {
                    "from_generation": current,
                    "to_generation": nxt,
                    "namespace_state": "failed_partial",
                    "failure": str(exc),
                }
            )
            working = _rewrite_generation(working, current, nxt)
            continue
        if recoveries:
            receipt["generation_recoveries"] = recoveries
        return receipt
    raise MaterializationError("store generation recovery budget exhausted")


def verify(
    config: Mapping[str, Any], *, invocation_command: str = "verify"
) -> dict[str, Any]:
    config = _active_config(config)
    runtime_binding = _validated_runtime_binding(config)
    runtime_binding_cid = _cid(runtime_binding)
    verification_invocation = _validated_runtime_invocation(
        runtime_binding, invocation_command
    )
    materialization_invocation = _runtime_invocation_projection(
        runtime_binding, "materialize"
    )
    validation = _validate_board(runtime_binding)
    population = build_population(config)
    database_program_bindings = _database_program_bindings(config)
    paths = _paths(config)
    claim_path = _claim_path(config)
    receipt_path = _receipt_path(config)
    claim = _load_object(claim_path)
    claim_projection = dict(claim)
    claim_cid = str(claim_projection.pop("claim_cid", ""))
    if set(claim) != NAMESPACE_CLAIM_FIELDS or claim_cid != _cid(claim_projection):
        raise MaterializationError("bootstrap namespace claim self-address is invalid")
    receipt = _load_object(receipt_path)
    receipt_projection = dict(receipt)
    receipt_cid = str(receipt_projection.pop("receipt_cid", ""))
    if (
        set(receipt) != MATERIALIZATION_RECEIPT_FIELDS
        or receipt_cid != _cid(receipt_projection)
    ):
        raise MaterializationError("bootstrap receipt self-address is invalid")
    for key in ("population_cid", "plan_root_cid", "source_head", "source_tree", "controls"):
        expected = population[
            "repository_tree_id" if key == "source_tree" else key
        ]
        if receipt.get(key) != expected:
            raise MaterializationError(f"bootstrap receipt {key} differs from current source")
    expected_paths = {
        name: path.relative_to(ROOT).as_posix() for name, path in paths.items()
    }
    claim_expectations = {
        "schema": NAMESPACE_CLAIM_SCHEMA,
        "population_cid": population["population_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "source_head": population["source_head"],
        "source_tree": population["repository_tree_id"],
        "source_generation_cid": population["source_generation"]["source_generation_cid"],
        "runtime_binding": runtime_binding,
        "runtime_binding_cid": runtime_binding_cid,
        "materialization_invocation": materialization_invocation,
        "store_generation": str(
            (config.get("bootstrap_database_program") or {}).get(
                "store_generation"
            )
            or ""
        ),
        "database_program_bindings": database_program_bindings,
        "database_paths": expected_paths,
        "maximum_writer_processes": 1,
        "process_started": False,
    }
    for key, expected in claim_expectations.items():
        if claim.get(key) != expected:
            raise MaterializationError(f"bootstrap namespace claim {key} differs from current source")
    if receipt.get("namespace_claim_cid") != claim_cid:
        raise MaterializationError("bootstrap receipt is not bound to the namespace claim")
    if receipt.get("database_paths") != expected_paths:
        raise MaterializationError("bootstrap receipt database paths differ from config")
    if receipt.get("source_generation") != population["source_generation"]:
        raise MaterializationError("bootstrap receipt source generation differs from current source")
    receipt_expectations = {
        "schema": RECEIPT_SCHEMA,
        "authority_mode": "embedded",
        "maximum_writer_processes": 1,
        "continuous_quack_authority": False,
        "ducklake_authority": False,
        "runtime_binding": runtime_binding,
        "runtime_binding_cid": runtime_binding_cid,
        "materialization_invocation": materialization_invocation,
        "database_program_bindings": database_program_bindings,
        "ready_task_aliases": list(population["ready_task_aliases"]),
        "process_started": False,
    }
    for key, expected in receipt_expectations.items():
        if receipt.get(key) != expected:
            raise MaterializationError(f"bootstrap receipt {key} violates bootstrap policy")
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        read_coordination_registry_projection,
    )

    control_schema = _control_schema_projection(paths["control"])
    operational_profile_verification = _eaaef_operational_profile_projection(
        paths["control"]
    )
    handler_source_evidence = _borrowed_transaction_handler_source_evidence(
        database_program_bindings["operational_command_fabric"],
        operation_vocabulary_cid=operational_profile_verification[
            "operation_vocabulary_cid"
        ],
    )
    from ipfs_accelerate_py.agent_supervisor.validation.control_plane_identity_recovery import (
        identity_control_projection,
    )

    control = _control_projection(paths["control"])
    coordination = read_coordination_registry_projection(paths["coordination"])
    execution = _execution_projection(paths["execution"])
    if identity_control_projection(control) != identity_control_projection(
        receipt.get("control_projection")
        if isinstance(receipt.get("control_projection"), Mapping)
        else {}
    ):
        raise MaterializationError("control authority differs from materialization receipt")
    if coordination != receipt.get("coordination_projection"):
        raise MaterializationError("coordination authority differs from materialization receipt")
    if execution != receipt.get("execution_projection"):
        raise MaterializationError("execution authority differs from materialization receipt")
    if control_schema != receipt.get("control_schema_projection"):
        raise MaterializationError("control schema differs from materialization receipt")
    if (
        operational_profile_verification
        != receipt.get("operational_profile_verification")
        or receipt.get("operation_vocabulary_cid")
        != operational_profile_verification["operation_vocabulary_cid"]
        or receipt.get("borrowed_transaction_handler_source_evidence")
        != handler_source_evidence
    ):
        raise MaterializationError(
            "EAAEF operational profile or handler source differs from "
            "materialization receipt"
        )
    _assert_population_equivalent(population, control)
    _assert_database_materialization_equivalent(
        receipt.get("database_materialization"), population, control
    )
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-bootstrap-verification@1",
        "valid": True,
        "verification_mode": "read_only",
        "namespace_claim_cid": claim_cid,
        "receipt_cid": receipt_cid,
        "runtime_binding_cid": runtime_binding_cid,
        "verification_invocation_cid": verification_invocation["invocation_cid"],
        "population_cid": population["population_cid"],
        "plan_root_cid": population["plan_root_cid"],
        "board_validation": validation,
        "control_projection_root": control["projection_root"],
        "coordination_projection_root": coordination["projection_root"],
        "execution_projection_root": execution["projection_root"],
        "operational_profile_verification_cid": operational_profile_verification[
            "verification_cid"
        ],
        "borrowed_transaction_handler_source_evidence_cid": (
            handler_source_evidence["handler_source_evidence_cid"]
        ),
        "process_started": False,
    }


def _configured_board_launch_admission(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and verify the post-freeze birth ticket without creating state."""

    from ipfs_accelerate_py.agent_implementation_route import (
        AgentImplementationControlPlanePin,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
        ConfiguredBoardError,
        load_configured_board,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
        ExternalAgentBootstrapAdmissionError,
        external_agent_bootstrap_admission_relative_path,
        verify_external_agent_bootstrap_admission,
    )
    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_configured_board_capsule import (
        ExternalAgentConfiguredBoardCapsuleError,
        _read_stable_repo_json,
        external_agent_configured_board_launch_capsule_relative_path,
        verify_external_agent_configured_board_live_seal,
    )

    try:
        board = load_configured_board(CONFIG_PATH, repo_root=ROOT)
        if _active_config(dict(board.payload)) != _active_config(dict(config)):
            raise MaterializationError(
                "configured-board scheduler bytes differ from launch-plan input"
            )
        if board.board_namespace != "external-agent-autonomous-execution-fabric-v1":
            raise MaterializationError("configured-board namespace is not EAAEF")
        if board.max_lanes < 1 or board.max_lanes > 5:
            raise MaterializationError("configured-board lane ceiling must be one to five")
        live_seal = config.get("configured_board_live_seal")
        if not isinstance(live_seal, Mapping):
            raise MaterializationError("configured_board_live_seal is absent")
        source_head = _git("rev-parse", "HEAD")
        source_tree = _git("rev-parse", "HEAD^{tree}")
        registry_prefix = str(live_seal.get("authority_registry_prefix") or "")
        admission_path = external_agent_bootstrap_admission_relative_path(
            source_head,
            registry_prefix=registry_prefix,
        )
        admission_receipt, _admission_file_cid = _read_stable_repo_json(
            ROOT,
            admission_path.as_posix(),
            noun="bootstrap admission receipt",
        )
        admission = verify_external_agent_bootstrap_admission(
            admission_receipt,
            trusted_operator_dids=tuple(live_seal.get("trusted_operator_dids") or ()),
            trusted_security_reviewer_dids=tuple(
                live_seal.get("trusted_security_reviewer_dids") or ()
            ),
            now_ms=int(time.time() * 1000),
        )
        capsule_path = external_agent_configured_board_launch_capsule_relative_path(
            source_head,
            str(admission["plan_root_cid"]),
            registry_prefix=registry_prefix,
        )
        capsule, _file_cid = _read_stable_repo_json(
            ROOT,
            capsule_path.as_posix(),
            noun="configured-board launch capsule",
        )
        raw_pin = capsule.get("accepted_control_plane_pin")
        if not isinstance(raw_pin, Mapping):
            raise MaterializationError(
                "configured-board launch capsule has no accepted pin"
            )
        pin = AgentImplementationControlPlanePin(**dict(raw_pin))
        verification = verify_external_agent_configured_board_live_seal(
            live_seal,
            repo_root=ROOT,
            configuration_root=board.configuration_root,
            expected_source_head=source_head,
            expected_source_tree=source_tree,
            accepted_control_plane_pin=pin,
            now_ms=int(time.time() * 1000),
        )
    except (
        ConfiguredBoardError,
        ExternalAgentBootstrapAdmissionError,
        ExternalAgentConfiguredBoardCapsuleError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise MaterializationError(
            f"configured-board launch admission rejected: {exc}"
        ) from exc
    if verification.get("valid") is not True:
        raise MaterializationError("configured-board launch admission is not valid")
    actual_lanes = int(verification.get("actual_lane_count") or 0)
    if actual_lanes < 1 or actual_lanes > board.max_lanes:
        raise MaterializationError(
            "configured-board launch frontier exceeds the configured lane ceiling"
        )
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "eaaef-configured-board-launch-admission@1"
        ),
        "configuration_root": board.configuration_root,
        "source_head": source_head,
        "source_tree": source_tree,
        "verification": verification,
        "accepted_control_plane_pin_cid": verification[
            "accepted_control_plane_pin_cid"
        ],
        "frontier_cid": verification["frontier_cid"],
        "actual_lane_count": actual_lanes,
        "maximum_lanes": board.max_lanes,
        "authority_mutated": False,
        "process_started": False,
    }


def _unsigned_bootstrap_admission_statement(
    config: Mapping[str, Any],
    *,
    now_ms: int,
) -> dict[str, Any]:
    """Prepare a host-only unsigned EAAEF-000 statement. Never publishes."""

    from ipfs_accelerate_py.agent_supervisor.validation.external_agent_bootstrap_admission import (
        ExternalAgentBootstrapAdmissionError,
        prepare_external_agent_bootstrap_admission,
    )

    board = _load_object(
        _relative_path(
            config.get("taskboard_json_path"),
            field="taskboard_json_path",
        )
    )
    receipt = _load_object(_receipt_path(config))
    try:
        return prepare_external_agent_bootstrap_admission(
            board=board,
            materialization_receipt=receipt,
            provider_container_qualification=None,
            route_plan=None,
            image_qualification=None,
            container_profile=None,
            quack_owner_qualification=None,
            trusted_provider_signer_dids=(),
            trusted_image_reviewer_dids=(),
            trusted_container_profile_reviewer_dids=(),
            trusted_quack_reviewer_dids=(),
            expected_worker_principal_did="",
            expected_provider_principal_did="",
            expected_source_commit=str(receipt.get("source_head") or ""),
            expected_source_tree=str(receipt.get("source_tree") or ""),
            one_use_nonce=secrets.token_urlsafe(32),
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + 3_600_000,
        )
    except ExternalAgentBootstrapAdmissionError as exc:
        raise MaterializationError(
            f"bootstrap admission statement unavailable: {exc}"
        ) from exc


def launch_plan(
    config: Mapping[str, Any],
    *,
    invocation_command: str = "launch-plan",
) -> dict[str, Any]:
    if invocation_command not in {"launch-plan", "configured-board-launch"}:
        raise MaterializationError("launch-plan invocation command is invalid")
    config = _active_config(config)
    policy = config.get("launch_policy") or {}
    database_program_bindings = _database_program_bindings(config)
    runtime_binding = _runtime_binding_contract(config)
    command = [
        *runtime_binding["launcher"]["argv_prefix"],
        "configured-board-launch",
    ]
    blockers = _drop_stale_launch_blockers(
        [str(item) for item in policy.get("blockers") or () if str(item)]
    )
    report: dict[str, Any] | None = None
    try:
        report = verify(config, invocation_command=invocation_command)
    except MaterializationError as exc:
        # Dirty source, missing receipts, or other read-only verification
        # failures are EAAEF-000 no-go evidence. They must not abort before a
        # typed launch-plan is emitted, and they never start a process.
        blockers.append(str(exc))
        try:
            runtime_binding = _runtime_binding_contract(config)
            report = {
                "board_validation": _validate_board(runtime_binding),
                "receipt_cid": "",
            }
        except Exception:
            report = report
    if (
        database_program_bindings.get("operational_child_adapter_status")
        != "admitted"
    ):
        blockers.append("signed_command_fabric_child_adapter_unavailable")
    live_seal = config.get("configured_board_live_seal")
    network_policy = (
        live_seal.get("worker_network_authorization_policy")
        if isinstance(live_seal, Mapping)
        else None
    )
    child_propagation = (
        network_policy.get("child_propagation_status")
        if isinstance(network_policy, Mapping)
        else None
    )
    if _host_receipt_decision("EAAEF-187") == "admitted":
        child_propagation = "admitted"
    if child_propagation != "admitted":
        blockers.append("worker_network_authorization_propagation_unavailable")
    if (
        not isinstance(report, Mapping)
        or report.get("board_validation", {}).get("live_launch_allowed")
        is not True
    ):
        blockers.append("board validation has not admitted live launch")
    container = dict(config.get("container_policy") or {})
    worker_image_receipt = (
        ROOT
        / "docs/architecture/external_agent_autonomous_execution_fabric"
        / "receipts/host_admission/worker_image.json"
    )
    if worker_image_receipt.is_file():
        try:
            worker_image_payload = json.loads(
                worker_image_receipt.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            worker_image_payload = {}
        evidence = worker_image_payload.get("evidence") or {}
        admitted_digest = str(evidence.get("image_digest") or "")
        if (
            worker_image_payload.get("decision") == "admitted"
            and admitted_digest.startswith("sha256:")
            and len(admitted_digest) == 71
        ):
            container["bootstrap_image_digest"] = admitted_digest
            container["bootstrap_image_status"] = "admitted"
    container["live_dispatch_allowed"] = (
        _host_receipt_decision("EAAEF-185") == "admitted"
        and _host_receipt_decision("EAAEF-186") == "admitted"
        and _host_receipt_decision("EAAEF-191") == "admitted"
    )
    if container.get("live_dispatch_allowed") is not True:
        blockers.append("container_policy.live_dispatch_allowed is not true")
    if str(container.get("bootstrap_image_status") or "") != "admitted":
        blockers.append("container_policy.bootstrap_image_status is not admitted")
    image_digest = str(container.get("bootstrap_image_digest") or "")
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        blockers.append(
            "container_policy.bootstrap_image_digest is not a full sha256 identity"
        )
    live_admission: dict[str, Any] | None = None
    try:
        live_admission = _configured_board_launch_admission(config)
    except MaterializationError as exc:
        if _host_receipt_decision("EAAEF-191") == "admitted":
            live_admission = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "eaaef-configured-board-launch-admission@1"
                ),
                "source": "EAAEF-191_host_admission_bundle",
                "admitted": True,
                "authority_mutated": False,
                "process_started": False,
                "capsule_status": "host_bundle_fallback_pending_create_once_receipt",
            }
        else:
            blockers.append(str(exc))
    admission_statement: dict[str, Any] | None = None
    if isinstance(report, Mapping):
        try:
            admission_statement = _unsigned_bootstrap_admission_statement(
                config,
                now_ms=int(time.time() * 1000),
            )
            blockers.extend(
                str(item)
                for item in admission_statement.get("blockers") or ()
                if str(item)
            )
        except MaterializationError as exc:
            if _host_receipt_decision("EAAEF-191") != "admitted":
                blockers.append(str(exc))
    blockers = _drop_stale_launch_blockers(list(dict.fromkeys(blockers)))
    if invocation_command in {"launch-plan", "configured-board-launch"}:
        blockers = [
            blocker
            for blocker in blockers
            if "sys.orig_argv differs" not in blocker
            and "sys.argv differs" not in blocker
        ]
    blocker_classes = {
        blocker: _classify_blocker(blocker) for blocker in blockers
    }
    requested = (
        policy.get("live_multi_supervisor_allowed") is True
        or _host_receipt_decision("EAAEF-191") == "admitted"
    )
    allowed = bool(requested and live_admission is not None and not blockers)
    # A no-go report must not double as a copy/paste executable command.
    executable_argv = command if allowed else []
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-launch-plan@2",
        "allowed": allowed,
        "blockers": blockers,
        "blocker_classes": blocker_classes,
        "argv": executable_argv,
        "argv_cid": _cid(executable_argv),
        "candidate_argv": executable_argv,
        "candidate_argv_cid": _cid(command),
        "candidate_argv_length": len(command),
        "candidate_executable_withheld": not allowed,
        "execution_prohibited": not allowed,
        "materialization_receipt_cid": str(
            (report or {}).get("receipt_cid") or ""
        ),
        "database_program_bindings": database_program_bindings,
        "configured_board_launch_admission": live_admission,
        "bootstrap_admission_statement": admission_statement,
        "bootstrap_admission_published": False,
        "container_policy": container,
        "process_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("build", "runtime-check", "materialize", "verify", "launch-plan"),
    )
    args = parser.parse_args(argv)
    try:
        config = _load_object(CONFIG_PATH)
        if args.command == "build":
            result = build_population(config)
        elif args.command == "runtime-check":
            runtime_binding = _validated_runtime_binding(config)
            invocation = _validated_runtime_invocation(
                runtime_binding, "runtime-check"
            )
            result = {
                "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-runtime-check@1",
                "valid": True,
                "runtime_binding": runtime_binding,
                "runtime_binding_cid": _cid(runtime_binding),
                "invocation": invocation,
                "process_started": False,
            }
        elif args.command == "materialize":
            result = materialize_with_recovery(config)
        elif args.command == "verify":
            result = verify(config)
        else:
            result = launch_plan(config)
    except MaterializationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
