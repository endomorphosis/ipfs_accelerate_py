#!/usr/bin/env python3
"""Bounded launcher for the PCPC Quack owner and configured supervisor.

The launcher is deliberately program-specific.  It admits one sealed scheduler
configuration, one rootless-Docker owner profile, and one detached configured-
board scheduler entry point.  Raw Quack tokens are neither accepted nor
published; readiness is proved through the configured opaque handle.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import site
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any, Final, Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (  # noqa: E402
    QUACK_ISOLATION_RECEIPT_SCHEMA,
    QUACK_STATE_SERVER_INTERFACE,
    QUACK_STATE_SERVER_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (  # noqa: E402
    canonical_json_bytes,
    content_identity,
    is_secret_handle,
)

PROGRAM: Final = "agent-supervisor-proof-carrying-procedure-compiler-v1"
CONFIG_RELATIVE: Final = "config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json"
MATERIALIZER_RELATIVE: Final = "scripts/materialize_agent_supervisor_procedure_compiler_program.py"
SCHEDULER_RELATIVE: Final = "scripts/ops/agent_supervisor/configured_board_scheduler.py"
OWNER_CONTAINER_NAME: Final = "ipfs-accelerate-pcpc-quack-owner-v1"
OWNER_CONTAINER_HOSTNAME: Final = "ipfs-accelerate-pcpc-quack-owner-v1"
OWNER_ISOLATION_FILENAME: Final = "isolation.json"
OWNER_HOME: Final = "/opt/quack-home"
OWNER_EXTENSION_TARGET: Final = f"{OWNER_HOME}/.duckdb/extensions/v1.5.5/linux_arm64"
EXPECTED_IMAGE_ID: Final = "sha256:ca52183d6e3f6d472b36092fc07a76fde0b7962da92b84dad2dc1038d93009ad"
EXPECTED_IMAGE_LABEL: Final = "2026-08-20-v3"
EXPECTED_OWNER_WRITE_ROOT: Final = (
    "state/agent_supervisor_proof_carrying_procedure_compiler/control"
)
EXPECTED_OWNER_STATE_DIR: Final = f"{EXPECTED_OWNER_WRITE_ROOT}/quack-owner"
EXPECTED_DATABASE_PATH: Final = f"{EXPECTED_OWNER_WRITE_ROOT}/control.duckdb"
EXPECTED_EXTENSION_HASHES: Final = {
    "httpfs.duckdb_extension": ("eba6e263e395a83966090f1f11ade63630b1b21422f0f2813858d179d42ea1e9"),
    "httpfs.duckdb_extension.info": (
        "69f35648f184abd1ffe5a455e1b378eaa287dfe24f0fa04deb475826128c93bd"
    ),
    "quack.duckdb_extension": ("41b2b9292bfb860c5ca8c5f818f9dd7a2c6bc24f9c750cffbc3169286fe59f08"),
    "quack.duckdb_extension.info": (
        "14ee8ddb246c590db9f8b1d090566ef159cf8a9175b3b0b7069d54435815bd89"
    ),
}
OWNER_START_RECEIPT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/pcpc-owner-launch@1"
OWNER_STATUS_RECEIPT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/pcpc-owner-status@1"
SUPERVISOR_START_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/pcpc-supervisor-launch@1"
)
REMOTE_PROBE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/pcpc-handle-readiness-probe@1"
PLAN_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/pcpc-launch-plan@1"
ERROR_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/pcpc-launch-error@1"
MATERIALIZATION_VERIFICATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler-materialization-verification@1"
)
TRUSTED_DUCKDB_HOME_ENV: Final = "IPFS_ACCELERATE_AGENT_TRUSTED_DUCKDB_HOME"
TRUSTED_RUNTIME_PATH_ENV_NAMES: Final = (
    "CUDA_CACHE_PATH",
    "XDG_CACHE_HOME",
)
TRUSTED_RUNTIME_FLAG_ENV: Final = {
    "CUDA_CACHE_DISABLE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
TRUSTED_HOME_DIRECTORY_MODE: Final = 0o500
TRUSTED_CACHE_DIRECTORY_MODE: Final = 0o700
RUNTIME_LABEL_KEY: Final = "org.ipfs-accelerate.pcpc-runtime"
RUNTIME_LABELS: Final = {
    "org.ipfs-accelerate.pcpc.role": "quack-owner",
    "org.ipfs-accelerate.pcpc.program": PROGRAM,
}
MAX_JSON_BYTES: Final = 1_048_576
MAX_OWNER_WAIT_SECONDS: Final = 120
MAX_SUPERVISOR_DURATION_SECONDS: Final = 31_536_000
MAX_FAILED_OWNER_QUARANTINES: Final = 16
MAX_EXTENSION_FILE_BYTES: Final = 128 * 1024 * 1024
HEX_64: Final = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_ID: Final = re.compile(r"^[0-9a-f]{64}$")

_OWNER_CONFIG_FIELDS: Final = frozenset(
    {
        "schema",
        "required",
        "backend",
        "runtime_executable",
        "runtime_endpoint",
        "image_id",
        "image_os",
        "image_architecture",
        "image_label",
        "python_executable",
        "network",
        "host",
        "port",
        "container_bind_host",
        "container_port",
        "owner_write_root",
        "state_dir",
        "extension_directory",
        "extension_files_sha256",
        "pids_limit",
        "memory_bytes",
        "cpus",
        "tmpfs_size_bytes",
    }
)
_DATABASE_PROGRAM_FIELDS: Final = frozenset(
    {
        "authority_mode",
        "authoritative_transactional_data_model",
        "authoritative_records_schema_cas_and_fencing",
        "task_source_kind",
        "endpoint_secret_handle",
        "quack_endpoint",
        "store_id",
        "store_generation",
        "schema_revision",
        "event_store_path",
        "runtime_registry_path",
        "worktree_root",
        "export_profile",
        "failover_policy",
        "explicit_legacy",
    }
)
_STATUS_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "lifecycle",
        "database_path",
        "state_dir",
        "host",
        "port",
        "container_bind_host",
        "container_port",
        "store_id",
        "secret_handle",
        "identity",
        "capability_status",
        "extension_fingerprint",
        "storage_schema_fingerprint",
        "read_replica",
        "owner_marker_path",
        "status_path",
    }
)
_IDENTITY_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "contract_version",
        "server_id",
        "store_id",
        "database_uuid",
        "schema_revision",
        "schema_fingerprint",
        "generation",
        "fence_epoch",
        "revision",
        "process_birth",
        "process_birth_id",
        "listen_uri",
        "extension_fingerprint",
        "credential_generation",
        "secret_handle",
        "repository_id",
        "startup_epoch",
        "started_at",
        "status",
    }
)
_READ_REPLICA_FIELDS: Final = frozenset(
    {
        "schema",
        "authority",
        "path",
        "source_database_path",
        "server_id",
        "database_uuid",
        "generation",
        "schema_revision",
        "schema_fingerprint",
        "storage_schema_fingerprint",
        "sha256",
        "size_bytes",
        "refresh_sequence",
        "refreshed_at_ms",
        "live",
    }
)
_MATERIALIZATION_FIELDS: Final = frozenset(
    {
        "schema",
        "valid",
        "repository_commit",
        "repository_tree",
        "database_path",
        "projection_cid",
        "task_count",
        "completed_task_ids",
        "ready_task_ids",
        "blocked_task_ids",
        "projection_matches_events",
        "plan_current",
        "tasks_current",
        "qualification_current",
        "freshly_qualified",
        "fresh_qualification_cid",
    }
)
_OWNER_ISOLATION_FIELDS: Final = frozenset(
    {
        "schema",
        "runtime",
        "container_id",
        "container_hostname",
        "network_mode",
        "container_bind_host",
        "container_port",
        "published_host",
        "published_port",
        "published_protocol",
        "owner_write_root",
        "database_path",
        "state_dir",
        "repository_path",
        "allowed_rw_mount_targets",
        "issuer",
        "issued_at",
        "receipt_cid",
    }
)
_SUPERVISOR_LAUNCH_FIELDS: Final = frozenset(
    {"coordinator_pid", "coordinator_pid_path", "coordinator_log"}
)


class ProgramLaunchError(RuntimeError):
    """Typed, fail-closed launch refusal."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code or "launch_refused")
        super().__init__(message)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: float = 60,
    ) -> CommandResult: ...


def _default_runner(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float = 60,
) -> CommandResult:
    completed = subprocess.run(
        tuple(str(item) for item in argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return CommandResult(
        returncode=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _decode_json_object(raw: str, *, noun: str) -> dict[str, Any]:
    if len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise ProgramLaunchError("json_too_large", f"{noun} exceeds its byte bound")
    try:
        value = json.loads(raw, object_pairs_hook=_duplicate_guard)
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ProgramLaunchError("json_invalid", f"{noun} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProgramLaunchError("json_invalid", f"{noun} must be one JSON object")
    return value


def _canonical_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProgramLaunchError("config_invalid", f"{field} must be non-empty")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ProgramLaunchError(
            "path_escape", f"{field} is not a canonical repository-relative path"
        )
    return value


def _resolved_inside(repo_root: Path, relative: str, *, field: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / _canonical_relative(relative, field=field)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProgramLaunchError("path_escape", f"{field} escapes the repository") from exc
    return candidate


def _positive_int(value: Any, *, field: str, maximum: int) -> int:
    if type(value) is not int or value < 1 or value > maximum:
        raise ProgramLaunchError("config_invalid", f"{field} must be an integer in [1, {maximum}]")
    return int(value)


@dataclass(frozen=True)
class ProgramConfig:
    repo_root: Path
    config_path: Path
    config_sha256: str
    branch: str
    ancestor: str
    board_namespace: str
    runtime_executable: str
    runtime_endpoint: str
    image_id: str
    image_os: str
    image_architecture: str
    image_label: str
    python_executable: str
    host: str
    port: int
    container_bind_host: str
    container_port: int
    owner_write_root: Path
    state_dir: Path
    database_path: Path
    store_id: str
    secret_handle: str
    endpoint: str
    extension_directory: Path
    extension_hashes: Mapping[str, str]
    pids_limit: int
    memory_bytes: int
    cpus: int
    tmpfs_size_bytes: int
    evidence_root: Path
    state_root: Path

    @property
    def isolation_receipt_path(self) -> Path:
        return self.state_dir / OWNER_ISOLATION_FILENAME

    @property
    def repository_id_prefix(self) -> str:
        return f"git:{self.board_namespace}"

    @property
    def qualification_home(self) -> Path:
        identity = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": "ipfs_accelerate_py/agent-supervisor/pcpc-duckdb-home@1",
                    "extension_files_sha256": dict(sorted(self.extension_hashes.items())),
                }
            )
        ).hexdigest()
        return self.state_root / "qualification-homes" / identity


def parse_program_config(
    payload: Mapping[str, Any], *, repo_root: Path, config_path: Path
) -> ProgramConfig:
    """Validate the program-owned configuration subset with closed fields."""

    if not isinstance(payload, Mapping):
        raise ProgramLaunchError("config_invalid", "scheduler config must be an object")
    owner = payload.get("quack_owner_isolation")
    database = payload.get("database_program")
    source = payload.get("source_binding")
    runtime_paths = payload.get("runtime_paths")
    if not isinstance(owner, Mapping) or set(owner) != _OWNER_CONFIG_FIELDS:
        raise ProgramLaunchError(
            "config_unknown_field",
            "quack_owner_isolation has unknown or missing normative fields",
        )
    if not isinstance(database, Mapping) or set(database) != _DATABASE_PROGRAM_FIELDS:
        raise ProgramLaunchError(
            "config_unknown_field",
            "database_program has unknown or missing normative fields",
        )
    if not isinstance(source, Mapping) or not isinstance(runtime_paths, Mapping):
        raise ProgramLaunchError("config_invalid", "source/runtime bindings are required")
    if (
        payload.get("schema")
        != "ipfs_accelerate_py.agent_supervisor."
        "agent-supervisor-proof-carrying-procedure-compiler-v1.scheduler_config@1"
        or payload.get("board_namespace") != PROGRAM
        or owner.get("schema") != "ipfs_accelerate_py.agent_supervisor.pcpc-quack-owner-isolation@1"
        or owner.get("required") is not True
        or owner.get("backend") != "docker"
        or owner.get("network") != "bridge"
        or owner.get("host") != "127.0.0.1"
        or owner.get("port") != 45671
        or owner.get("container_bind_host") != "0.0.0.0"
        or owner.get("container_port") != 45671
        or owner.get("image_id") != EXPECTED_IMAGE_ID
        or owner.get("image_label") != EXPECTED_IMAGE_LABEL
        or owner.get("python_executable") != "/opt/pcpc-runtime/bin/python"
        or database.get("authority_mode") != "quack"
        or database.get("task_source_kind") != "duckdb"
        or database.get("failover_policy") != "fail_closed"
        or database.get("explicit_legacy") is not False
    ):
        raise ProgramLaunchError("config_invalid", "program authority profile is invalid")
    branch = str(source.get("accelerator_required_branch") or "")
    ancestor = str(source.get("accelerator_required_ancestor") or "")
    if not branch or not re.fullmatch(r"[0-9a-f]{40}", ancestor):
        raise ProgramLaunchError("config_invalid", "source binding is malformed")
    runtime_executable = str(owner.get("runtime_executable") or "")
    runtime_endpoint = str(owner.get("runtime_endpoint") or "")
    image_id = str(owner.get("image_id") or "")
    if (
        not Path(runtime_executable).is_absolute()
        or runtime_endpoint != "unix:///run/user/1000/docker.sock"
        or not image_id.startswith("sha256:")
        or not HEX_64.fullmatch(image_id.removeprefix("sha256:"))
    ):
        raise ProgramLaunchError("config_invalid", "Docker runtime identity is malformed")
    store_id = _canonical_relative(database.get("store_id"), field="database_program.store_id")
    owner_relative = _canonical_relative(
        owner.get("owner_write_root"), field="quack_owner_isolation.owner_write_root"
    )
    state_relative = _canonical_relative(
        owner.get("state_dir"), field="quack_owner_isolation.state_dir"
    )
    owner_write_root = _resolved_inside(repo_root, owner_relative, field="owner_write_root")
    state_dir = _resolved_inside(repo_root, state_relative, field="state_dir")
    database_path = _resolved_inside(repo_root, store_id, field="store_id")
    if (
        owner_relative != EXPECTED_OWNER_WRITE_ROOT
        or state_relative != EXPECTED_OWNER_STATE_DIR
        or store_id != EXPECTED_DATABASE_PATH
        or database_path.parent != owner_write_root
        or database_path.name != "control.duckdb"
        or state_dir != owner_write_root / "quack-owner"
    ):
        raise ProgramLaunchError(
            "config_invalid", "database, owner root, and owner state are not exact"
        )
    endpoint = str(database.get("quack_endpoint") or "")
    host = str(owner.get("host") or "")
    port = _positive_int(owner.get("port"), field="port", maximum=65535)
    container_bind_host = str(owner.get("container_bind_host") or "")
    container_port = _positive_int(
        owner.get("container_port"), field="container_port", maximum=65535
    )
    if endpoint != f"quack:{host}:{port}":
        raise ProgramLaunchError("config_invalid", "Quack endpoint is not owner-bound")
    if container_bind_host != "0.0.0.0" or container_port != port:
        raise ProgramLaunchError(
            "config_invalid", "container and published Quack ports are not exact"
        )
    secret_handle = str(database.get("endpoint_secret_handle") or "")
    if not is_secret_handle(secret_handle):
        raise ProgramLaunchError("config_invalid", "endpoint secret must be an opaque handle")
    extension_directory = Path(str(owner.get("extension_directory") or ""))
    hashes = owner.get("extension_files_sha256")
    required_extensions = {
        "httpfs.duckdb_extension",
        "httpfs.duckdb_extension.info",
        "quack.duckdb_extension",
        "quack.duckdb_extension.info",
    }
    if (
        not extension_directory.is_absolute()
        or not isinstance(hashes, Mapping)
        or set(hashes) != required_extensions
        or any(
            not isinstance(value, str) or not HEX_64.fullmatch(value) for value in hashes.values()
        )
        or dict(hashes) != EXPECTED_EXTENSION_HASHES
    ):
        raise ProgramLaunchError("config_invalid", "extension allowlist is malformed")
    evidence = _canonical_relative(runtime_paths.get("evidence"), field="runtime_paths.evidence")
    state = _canonical_relative(runtime_paths.get("state"), field="runtime_paths.state")
    config_bytes = config_path.read_bytes()
    return ProgramConfig(
        repo_root=repo_root.resolve(),
        config_path=config_path.resolve(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        branch=branch,
        ancestor=ancestor,
        board_namespace=PROGRAM,
        runtime_executable=runtime_executable,
        runtime_endpoint=runtime_endpoint,
        image_id=image_id,
        image_os=str(owner.get("image_os") or ""),
        image_architecture=str(owner.get("image_architecture") or ""),
        image_label=str(owner.get("image_label") or ""),
        python_executable=str(owner.get("python_executable") or ""),
        host=host,
        port=port,
        container_bind_host=container_bind_host,
        container_port=container_port,
        owner_write_root=owner_write_root,
        state_dir=state_dir,
        database_path=database_path,
        store_id=store_id,
        secret_handle=secret_handle,
        endpoint=endpoint,
        extension_directory=extension_directory.resolve(),
        extension_hashes=dict(sorted(hashes.items())),
        pids_limit=_positive_int(owner.get("pids_limit"), field="pids_limit", maximum=4096),
        memory_bytes=_positive_int(owner.get("memory_bytes"), field="memory_bytes", maximum=2**40),
        cpus=_positive_int(owner.get("cpus"), field="cpus", maximum=64),
        tmpfs_size_bytes=_positive_int(
            owner.get("tmpfs_size_bytes"), field="tmpfs_size_bytes", maximum=2**40
        ),
        evidence_root=_resolved_inside(repo_root, evidence, field="runtime_paths.evidence"),
        state_root=_resolved_inside(repo_root, state, field="runtime_paths.state"),
    )


def load_program_config(repo_root: Path, config_path: Path | None = None) -> ProgramConfig:
    root = repo_root.resolve()
    expected = (root / CONFIG_RELATIVE).resolve()
    selected = expected if config_path is None else config_path.resolve()
    if selected != expected or not selected.is_file() or selected.is_symlink():
        raise ProgramLaunchError("config_path_invalid", f"launcher admits only {CONFIG_RELATIVE}")
    payload = _decode_json_object(selected.read_text(encoding="utf-8"), noun="scheduler config")
    return parse_program_config(payload, repo_root=root, config_path=selected)


def _regular_file_sha256(path: Path, *, noun: str) -> tuple[os.stat_result, str]:
    """Hash one bounded regular file through a stable no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProgramLaunchError("extension_invalid", f"{noun} cannot be opened") from exc
    digest = hashlib.sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_EXTENSION_FILE_BYTES:
            raise ProgramLaunchError("extension_invalid", f"{noun} is unsafe")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > MAX_EXTENSION_FILE_BYTES:
                raise ProgramLaunchError("extension_invalid", f"{noun} is oversized")
            digest.update(block)
        after = os.fstat(descriptor)
    except ProgramLaunchError:
        raise
    except OSError as exc:
        raise ProgramLaunchError(
            "extension_invalid", f"{noun} cannot be read"
        ) from exc
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if total != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise ProgramLaunchError("extension_drift", f"{noun} changed while being hashed")
    return after, digest.hexdigest()


def verify_extension_files(config: ProgramConfig) -> None:
    for name, expected in config.extension_hashes.items():
        path = config.extension_directory / name
        try:
            os.lstat(path)
        except OSError as exc:
            raise ProgramLaunchError(
                "extension_missing", f"extension file is absent: {name}"
            ) from exc
        observed, digest = _regular_file_sha256(path, noun=f"extension file {name}")
        if (
            path.is_symlink()
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_size > MAX_EXTENSION_FILE_BYTES
        ):
            raise ProgramLaunchError("extension_invalid", f"extension file is unsafe: {name}")
        if digest != expected:
            raise ProgramLaunchError("extension_drift", f"extension digest drifted: {name}")


def owner_labels(config: ProgramConfig, *, head: str, tree: str) -> dict[str, str]:
    return {
        **RUNTIME_LABELS,
        "org.ipfs-accelerate.pcpc.head": head,
        "org.ipfs-accelerate.pcpc.tree": tree,
        "org.ipfs-accelerate.pcpc.config-sha256": config.config_sha256,
    }


def build_owner_create_argv(
    config: ProgramConfig,
    *,
    head: str,
    tree: str,
    container_name: str = OWNER_CONTAINER_NAME,
) -> list[str]:
    """Return the exact secret-free Docker create command."""

    argv = [
        config.runtime_executable,
        "--host",
        config.runtime_endpoint,
        "container",
        "create",
        "--name",
        container_name,
        "--hostname",
        OWNER_CONTAINER_HOSTNAME,
        "--read-only",
        "--network",
        "bridge",
        "--publish",
        f"{config.host}:{config.port}:{config.container_port}/tcp",
        "--ipc",
        "private",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(config.pids_limit),
        "--memory",
        str(config.memory_bytes),
        "--cpus",
        str(config.cpus),
        "--user",
        "0:0",
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={config.tmpfs_size_bytes},mode=1777",
        "--tmpfs",
        f"/var/tmp:rw,noexec,nosuid,nodev,size={config.tmpfs_size_bytes},mode=1777",
        "--tmpfs",
        f"{OWNER_HOME}:rw,noexec,nosuid,nodev,size={config.tmpfs_size_bytes},mode=0700,uid=0,gid=0",
    ]
    for key, value in sorted(owner_labels(config, head=head, tree=tree).items()):
        argv.extend(("--label", f"{key}={value}"))
    argv.extend(
        (
            "--mount",
            f"type=bind,src={config.repo_root},dst={config.repo_root},readonly,bind-propagation=rprivate",
            "--mount",
            f"type=bind,src={config.owner_write_root},dst={config.owner_write_root},bind-propagation=rprivate",
        )
    )
    for name in sorted(config.extension_hashes):
        source = config.extension_directory / name
        target = f"{OWNER_EXTENSION_TARGET}/{name}"
        argv.extend(("--mount", f"type=bind,src={source},dst={target},readonly"))
    repository_id = f"{config.repository_id_prefix}:commit:{head}:tree:{tree}"
    argv.extend(
        (
            "--entrypoint",
            "/usr/bin/env",
            config.image_id,
            "-i",
            f"HOME={OWNER_HOME}",
            f"XDG_CACHE_HOME={OWNER_HOME}/.cache",
            f"XDG_CONFIG_HOME={OWNER_HOME}/.config",
            "PATH=/opt/pcpc-runtime/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONNOUSERSITE=1",
            config.python_executable,
            str(config.repo_root / "scripts/ops/agent_supervisor/quack_state_server.py"),
            "--database",
            str(config.database_path),
            "--state-dir",
            str(config.state_dir),
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--container-bind-host",
            config.container_bind_host,
            "--container-port",
            str(config.container_port),
            "--store-id",
            config.store_id,
            "--repository-id",
            repository_id,
            "--secret-handle",
            config.secret_handle,
            "--isolation-receipt-json",
            str(config.isolation_receipt_path),
            "--json",
            "start",
        )
    )
    return argv


def build_owner_isolation_receipt(
    config: ProgramConfig, *, container_id: str, issued_at: str
) -> dict[str, Any]:
    if not CONTAINER_ID.fullmatch(container_id):
        raise ProgramLaunchError(
            "container_identity_invalid", "container ID is not full SHA-256 hex"
        )
    unsigned = {
        "schema": QUACK_ISOLATION_RECEIPT_SCHEMA,
        "runtime": "docker",
        "container_id": container_id,
        "container_hostname": OWNER_CONTAINER_HOSTNAME,
        "network_mode": "bridge",
        "container_bind_host": config.container_bind_host,
        "container_port": config.container_port,
        "published_host": config.host,
        "published_port": config.port,
        "published_protocol": "tcp",
        "owner_write_root": str(config.owner_write_root),
        "database_path": str(config.database_path),
        "state_dir": str(config.state_dir),
        "repository_path": str(config.repo_root),
        "allowed_rw_mount_targets": [str(config.owner_write_root)],
        "issuer": f"{PROGRAM}:program-launcher@1",
        "issued_at": issued_at,
    }
    return {**unsigned, "receipt_cid": content_identity(unsigned)}


def validate_owner_isolation_receipt(
    receipt: Mapping[str, Any], *, config: ProgramConfig, container_id: str
) -> None:
    if set(receipt) != _OWNER_ISOLATION_FIELDS:
        raise ProgramLaunchError("receipt_unknown_field", "owner isolation receipt is not closed")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_cid", None)
    if (
        receipt.get("schema") != QUACK_ISOLATION_RECEIPT_SCHEMA
        or receipt.get("runtime") != "docker"
        or receipt.get("container_id") != container_id
        or receipt.get("container_hostname") != OWNER_CONTAINER_HOSTNAME
        or receipt.get("network_mode") != "bridge"
        or receipt.get("container_bind_host") != config.container_bind_host
        or receipt.get("container_port") != config.container_port
        or receipt.get("published_host") != config.host
        or receipt.get("published_port") != config.port
        or receipt.get("published_protocol") != "tcp"
        or receipt.get("owner_write_root") != str(config.owner_write_root)
        or receipt.get("database_path") != str(config.database_path)
        or receipt.get("state_dir") != str(config.state_dir)
        or receipt.get("repository_path") != str(config.repo_root)
        or receipt.get("allowed_rw_mount_targets") != [str(config.owner_write_root)]
        or not str(receipt.get("issuer") or "")
        or not str(receipt.get("issued_at") or "")
        or claimed != content_identity(unsigned)
    ):
        raise ProgramLaunchError("receipt_forged", "owner isolation receipt identity is invalid")


def _mount_map(inspect: Mapping[str, Any]) -> dict[str, tuple[str, bool]]:
    mounts = inspect.get("Mounts")
    if not isinstance(mounts, list) or len(mounts) > 16:
        raise ProgramLaunchError("container_inspect_invalid", "container mounts are absent")
    result: dict[str, tuple[str, bool]] = {}
    for item in mounts:
        if not isinstance(item, Mapping) or item.get("Type") != "bind":
            continue
        source = str(item.get("Source") or "")
        destination = str(item.get("Destination") or "")
        rw = item.get("RW")
        if not source or not destination or type(rw) is not bool or destination in result:
            raise ProgramLaunchError(
                "container_inspect_invalid", "container bind mount is malformed"
            )
        result[destination] = (source, rw)
    return result


def validate_owner_container_inspect(
    inspect: Mapping[str, Any],
    *,
    config: ProgramConfig,
    head: str,
    tree: str,
    container_id: str | None = None,
    container_name: str = OWNER_CONTAINER_NAME,
    require_running: bool = False,
) -> str:
    """Reject a forged, foreign, or weakened owner container projection."""

    observed_id = str(inspect.get("Id") or "")
    config_block = inspect.get("Config")
    host = inspect.get("HostConfig")
    state = inspect.get("State")
    network = inspect.get("NetworkSettings")
    port_key = f"{config.container_port}/tcp"
    expected_binding = [{"HostIp": config.host, "HostPort": str(config.port)}]
    if (
        not CONTAINER_ID.fullmatch(observed_id)
        or (container_id is not None and observed_id != container_id)
        or inspect.get("Name") != f"/{container_name}"
        or inspect.get("Image") != config.image_id
        or not isinstance(config_block, Mapping)
        or not isinstance(host, Mapping)
        or not isinstance(state, Mapping)
        or config_block.get("Image") != config.image_id
        or config_block.get("User") != "0:0"
        or config_block.get("Hostname") != OWNER_CONTAINER_HOSTNAME
        or config_block.get("Entrypoint") != ["/usr/bin/env"]
        or config_block.get("ExposedPorts") != {port_key: {}}
    ):
        raise ProgramLaunchError(
            "container_identity_mismatch", "owner container identity is foreign"
        )
    expected_argv = build_owner_create_argv(config, head=head, tree=tree)
    expected_command = expected_argv[expected_argv.index(config.image_id) + 1 :]
    if (
        config_block.get("Env") != ["PATH=/opt/pcpc-runtime/bin:/usr/local/bin:/usr/bin:/bin"]
        or config_block.get("Cmd") != expected_command
    ):
        raise ProgramLaunchError(
            "container_environment_unbounded",
            "owner image environment or command is not exact",
        )
    labels = config_block.get("Labels")
    expected_labels = owner_labels(config, head=head, tree=tree)
    if not isinstance(labels, Mapping) or any(
        labels.get(key) != value for key, value in expected_labels.items()
    ):
        raise ProgramLaunchError("container_label_mismatch", "owner container labels are foreign")
    if (
        host.get("ReadonlyRootfs") is not True
        or host.get("NetworkMode") != "bridge"
        or host.get("PortBindings") != {port_key: expected_binding}
        or host.get("PublishAllPorts") is not False
        or host.get("IpcMode") != "private"
        # Docker's closed private PID namespace is represented by the empty
        # mode; the CLI rejects a literal ``--pid private`` value.
        or host.get("PidMode") != ""
        or sorted(host.get("CapDrop") or ()) != ["ALL"]
        or "no-new-privileges" not in (host.get("SecurityOpt") or ())
        or int(host.get("PidsLimit") or 0) != config.pids_limit
        or int(host.get("Memory") or 0) != config.memory_bytes
        or int(host.get("NanoCpus") or 0) != config.cpus * 1_000_000_000
    ):
        raise ProgramLaunchError(
            "container_isolation_weakened", "owner runtime controls are weaker than configured"
        )
    expected_mounts = {
        str(config.repo_root): (str(config.repo_root), False),
        str(config.owner_write_root): (str(config.owner_write_root), True),
        **{
            f"{OWNER_EXTENSION_TARGET}/{name}": (str(config.extension_directory / name), False)
            for name in config.extension_hashes
        },
    }
    if _mount_map(inspect) != expected_mounts:
        raise ProgramLaunchError("container_mount_mismatch", "owner bind mount set is not exact")
    command = config_block.get("Cmd")
    assert isinstance(command, list)
    forbidden = ("OPENAI", "ANTHROPIC", "AUTH.JSON", "DOCKER.SOCK", "QUACK_TOKEN")
    joined = "\n".join(str(item) for item in command).upper()
    if any(fragment in joined for fragment in forbidden):
        raise ProgramLaunchError(
            "container_secret_exposure", "owner command contains a forbidden secret surface"
        )
    if require_running and state.get("Running") is not True:
        raise ProgramLaunchError("owner_not_running", "owner container is not running")
    if require_running:
        attachments = network.get("Networks") if isinstance(network, Mapping) else None
        bridge = attachments.get("bridge") if isinstance(attachments, Mapping) else None
        try:
            bridge_address = ipaddress.ip_address(str(bridge.get("IPAddress") or ""))
        except (AttributeError, ValueError):
            bridge_address = None
        if (
            not isinstance(network, Mapping)
            or network.get("Ports") != {port_key: expected_binding}
            or not isinstance(attachments, Mapping)
            or set(attachments) != {"bridge"}
            or not isinstance(bridge, Mapping)
            or not CONTAINER_ID.fullmatch(str(bridge.get("NetworkID") or ""))
            or not CONTAINER_ID.fullmatch(str(bridge.get("EndpointID") or ""))
            or not isinstance(bridge_address, ipaddress.IPv4Address)
            or bridge_address.is_loopback
            or bridge_address.is_unspecified
        ):
            raise ProgramLaunchError(
                "container_network_mismatch",
                "running owner does not have the exact loopback-published bridge attachment",
            )
    return observed_id


def _atomic_create(path: Path, payload: Mapping[str, Any], *, canonical: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = (
        canonical_json_bytes(dict(payload))
        if canonical
        else (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        written = 0
        while written < len(data):
            count = os.write(descriptor, data[written:])
            if count <= 0:
                raise OSError("short receipt write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            observed = path.read_bytes()
            if observed != data:
                raise ProgramLaunchError(
                    "existing_artifact_conflict", f"refusing to replace existing artifact: {path}"
                ) from None
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _persist_receipt(config: ProgramConfig, receipt: Mapping[str, Any]) -> Path:
    cid = str(receipt.get("receipt_cid") or "")
    if not cid or content_identity({k: v for k, v in receipt.items() if k != "receipt_cid"}) != cid:
        raise ProgramLaunchError("receipt_forged", "launcher receipt CID is invalid")
    path = config.evidence_root / "program-launcher" / f"{cid}.json"
    _atomic_create(path, receipt)
    return path


def _safe_read_json(path: Path, *, exact_fields: frozenset[str], noun: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ProgramLaunchError("artifact_unavailable", f"{noun} is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
            or info.st_size > MAX_JSON_BYTES
        ):
            raise ProgramLaunchError("artifact_unsafe", f"{noun} is not owner-only and bounded")
        raw = os.read(descriptor, MAX_JSON_BYTES + 1)
    finally:
        os.close(descriptor)
    value = _decode_json_object(raw.decode("utf-8"), noun=noun)
    if set(value) != exact_fields:
        raise ProgramLaunchError("artifact_unknown_field", f"{noun} has unknown or missing fields")
    return value


def _identity_summary(status: Mapping[str, Any]) -> dict[str, Any]:
    identity = status.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
        raise ProgramLaunchError("owner_status_invalid", "owner identity is not closed")
    return {
        "server_id": str(identity.get("server_id") or ""),
        "store_id": str(identity.get("store_id") or ""),
        "database_uuid": str(identity.get("database_uuid") or ""),
        "schema_revision": int(identity.get("schema_revision") or 0),
        "schema_fingerprint": str(identity.get("schema_fingerprint") or ""),
        "generation": int(identity.get("generation") or 0),
        "process_birth_id": str(identity.get("process_birth_id") or ""),
        "listen_uri": str(identity.get("listen_uri") or ""),
        "extension_fingerprint": str(identity.get("extension_fingerprint") or ""),
        "secret_handle": str(identity.get("secret_handle") or ""),
        "repository_id": str(identity.get("repository_id") or ""),
        "status": str(identity.get("status") or ""),
    }


def validate_owner_status_payload(
    status: Mapping[str, Any], *, config: ProgramConfig, head: str, tree: str
) -> dict[str, Any]:
    if set(status) != _STATUS_FIELDS:
        raise ProgramLaunchError("owner_status_unknown_field", "owner status is not closed")
    identity = _identity_summary(status)
    replica = status.get("read_replica")
    expected_replica_path = config.database_path.with_name(
        f"{config.database_path.stem}.read-replica{config.database_path.suffix}"
    )
    expected_repository = f"{config.repository_id_prefix}:commit:{head}:tree:{tree}"
    if (
        status.get("schema") != QUACK_STATE_SERVER_SCHEMA
        or status.get("interface") != QUACK_STATE_SERVER_INTERFACE
        or status.get("lifecycle") != "ready"
        or status.get("database_path") != str(config.database_path)
        or status.get("state_dir") != str(config.state_dir)
        or status.get("host") != config.host
        or status.get("port") != config.port
        or status.get("container_bind_host") != config.container_bind_host
        or status.get("container_port") != config.container_port
        or status.get("store_id") != config.store_id
        or status.get("secret_handle") != config.secret_handle
        or not str(status.get("storage_schema_fingerprint") or "")
        or identity["store_id"] != config.store_id
        or identity["listen_uri"] != config.endpoint
        or identity["secret_handle"] != config.secret_handle
        or identity["repository_id"] != expected_repository
        or identity["status"] != "ready"
        or not identity["server_id"]
        or not identity["database_uuid"]
        or identity["generation"] < 1
        or identity["schema_revision"] < 1
        or not isinstance(replica, Mapping)
        or set(replica) != _READ_REPLICA_FIELDS
        or replica.get("schema") != "ipfs_accelerate_py/agent-supervisor/read-replica-observation@1"
        or replica.get("authority") != "non_authoritative_read_replica"
        or replica.get("path") != str(expected_replica_path)
        or replica.get("source_database_path") != str(config.database_path)
        or replica.get("server_id") != identity["server_id"]
        or replica.get("database_uuid") != identity["database_uuid"]
        or replica.get("generation") != identity["generation"]
        or replica.get("schema_revision") != identity["schema_revision"]
        or replica.get("schema_fingerprint") != identity["schema_fingerprint"]
        or replica.get("storage_schema_fingerprint") != status.get("storage_schema_fingerprint")
        or not isinstance(replica.get("sha256"), str)
        or not HEX_64.fullmatch(str(replica.get("sha256")).removeprefix("sha256:"))
        or not str(replica.get("sha256")).startswith("sha256:")
        or type(replica.get("size_bytes")) is not int
        or not 0 < replica.get("size_bytes", 0) <= 8 * 1024 * 1024 * 1024
        or type(replica.get("refresh_sequence")) is not int
        or replica.get("refresh_sequence", 0) < 1
        or type(replica.get("refreshed_at_ms")) is not int
        or replica.get("refreshed_at_ms", 0) < 1
        or replica.get("live") is not True
    ):
        raise ProgramLaunchError("owner_status_mismatch", "owner status is stale or foreign")
    try:
        replica_info = expected_replica_path.stat()
    except OSError as exc:
        raise ProgramLaunchError(
            "owner_status_mismatch", "read-replica observation has no local artifact"
        ) from exc
    if (
        expected_replica_path.is_symlink()
        or not stat.S_ISREG(replica_info.st_mode)
        or replica_info.st_uid != os.geteuid()
        or replica_info.st_mode & 0o077
        or replica_info.st_size != replica["size_bytes"]
    ):
        raise ProgramLaunchError(
            "owner_status_mismatch", "read-replica artifact is unsafe or size-drifted"
        )
    return identity


def _bounded_environment() -> dict[str, str]:
    result = {"PATH": "/usr/bin:/bin"}
    for name in ("LANG", "LC_ALL", "LC_CTYPE", "TZ"):
        value = str(os.environ.get(name, "") or "")
        if value and len(value) <= 128 and "\x00" not in value:
            result[name] = value
    return result


def _copy_exact_extension_file(source: Path, target: Path, *, expected_sha256: str) -> None:
    """Copy one allowlisted extension without following source or target links."""

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    write_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        source_fd = os.open(source, read_flags)
    except OSError as exc:
        raise ProgramLaunchError(
            "extension_missing", f"extension file cannot be opened: {source.name}"
        ) from exc
    target_fd: int | None = None
    digest = hashlib.sha256()
    total = 0
    try:
        source_info = os.fstat(source_fd)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_size > MAX_EXTENSION_FILE_BYTES:
            raise ProgramLaunchError(
                "extension_invalid", f"extension file is unsafe: {source.name}"
            )
        target_fd = os.open(target, write_flags, 0o400)
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > MAX_EXTENSION_FILE_BYTES:
                raise ProgramLaunchError(
                    "extension_invalid", f"extension file is oversized: {source.name}"
                )
            digest.update(block)
            remaining = memoryview(block)
            while remaining:
                written = os.write(target_fd, remaining)
                if written < 1:
                    raise OSError("extension copy made no progress")
                remaining = remaining[written:]
        os.fsync(target_fd)
    except ProgramLaunchError:
        raise
    except OSError as exc:
        raise ProgramLaunchError(
            "extension_copy_failed", f"extension file cannot be isolated: {source.name}"
        ) from exc
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)
    if total != source_info.st_size or digest.hexdigest() != expected_sha256:
        raise ProgramLaunchError(
            "extension_drift", f"extension digest drifted during isolation: {source.name}"
        )


def _validate_qualification_home(config: ProgramConfig, *, home: Path | None = None) -> Path:
    home = config.qualification_home if home is None else home
    extension_directories = (
        home,
        home / ".duckdb",
        home / ".duckdb" / "extensions",
        home / ".duckdb" / "extensions" / "v1.5.5",
        home / ".duckdb" / "extensions" / "v1.5.5" / "linux_arm64",
    )
    for directory in extension_directories:
        try:
            observed = os.lstat(directory)
        except OSError as exc:
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification HOME is incomplete"
            ) from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != TRUSTED_HOME_DIRECTORY_MODE
        ):
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification HOME directory is unsafe"
            )
    extension_home = extension_directories[-1]
    cache_root = home / ".cache"
    cache_directories = (
        cache_root,
        cache_root / "cuda",
        cache_root / "ipfs_accelerate",
        cache_root / "xdg",
    )
    for directory in cache_directories:
        try:
            observed = os.lstat(directory)
        except OSError as exc:
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification cache enclave is incomplete"
            ) from exc
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != TRUSTED_CACHE_DIRECTORY_MODE
        ):
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification cache enclave is unsafe"
            )
    expected_children = {
        home: {".cache", ".duckdb"},
        cache_root: {"cuda", "ipfs_accelerate", "xdg"},
        extension_directories[1]: {"extensions"},
        extension_directories[2]: {"v1.5.5"},
        extension_directories[3]: {"linux_arm64"},
        extension_home: set(config.extension_hashes),
    }
    for directory, names in expected_children.items():
        try:
            observed_names = {entry.name for entry in directory.iterdir()}
        except OSError as exc:
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification HOME cannot be inspected"
            ) from exc
        if observed_names != names:
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification HOME contains undeclared content"
            )
    for name, expected_sha256 in config.extension_hashes.items():
        target = extension_home / name
        try:
            observed = os.lstat(target)
        except OSError as exc:
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification extension is unavailable"
            ) from exc
        try:
            stable, digest = _regular_file_sha256(
                target,
                noun=f"qualification extension {name}",
            )
        except ProgramLaunchError as exc:
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification extension cannot be hashed"
            ) from exc
        if (
            not stat.S_ISREG(observed.st_mode)
            or stat.S_ISLNK(observed.st_mode)
            or observed.st_uid != os.geteuid()
            or stat.S_IMODE(observed.st_mode) != 0o400
            or observed.st_nlink != 1
            or observed.st_size > MAX_EXTENSION_FILE_BYTES
            or stable.st_ino != observed.st_ino
            or stable.st_dev != observed.st_dev
            or digest != expected_sha256
        ):
            raise ProgramLaunchError(
                "qualification_home_invalid", "qualification extension identity drifted"
            )
    return home


def _private_program_directory(path: Path, *, noun: str) -> None:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise ProgramLaunchError(
            "qualification_home_invalid", f"{noun} cannot be inspected"
        ) from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ProgramLaunchError("qualification_home_invalid", f"{noun} is not private")


def _fsync_program_directory(path: Path) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ProgramLaunchError(
            "qualification_home_invalid", "qualification HOME publication is not durable"
        ) from exc


def _quarantine_qualification_home(config: ProgramConfig) -> None:
    home = config.qualification_home
    try:
        observed = os.lstat(home)
    except OSError:
        return
    if (
        not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or observed.st_uid != os.geteuid()
    ):
        raise ProgramLaunchError(
            "qualification_home_invalid", "invalid qualification HOME is not an owned directory"
        )
    quarantine_root = config.state_root / "qualification-home-quarantine"
    try:
        quarantine_root.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise ProgramLaunchError(
            "qualification_home_invalid", "qualification HOME quarantine is unavailable"
        ) from exc
    _private_program_directory(quarantine_root, noun="qualification HOME quarantine")
    if len(tuple(islice(quarantine_root.iterdir(), MAX_FAILED_OWNER_QUARANTINES + 1))) >= (
        MAX_FAILED_OWNER_QUARANTINES
    ):
        raise ProgramLaunchError(
            "qualification_home_invalid", "qualification HOME quarantine is at capacity"
        )
    target = quarantine_root / f"{home.name}-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            home,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        stable = os.fstat(descriptor)
        if (
            stable.st_dev != observed.st_dev
            or stable.st_ino != observed.st_ino
            or not stat.S_ISDIR(stable.st_mode)
            or stable.st_uid != os.geteuid()
        ):
            raise OSError("qualification HOME changed during quarantine")
        # A 0500 directory cannot be moved between parents on every supported
        # filesystem. The private parent and stable no-follow descriptor admit
        # this bounded recovery-only permission change.
        os.fchmod(descriptor, 0o700)
        os.rename(home, target)
    except OSError as exc:
        raise ProgramLaunchError(
            "qualification_home_invalid", "invalid qualification HOME cannot be quarantined"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _fsync_program_directory(quarantine_root)
    _fsync_program_directory(config.state_root)


def _build_qualification_home(config: ProgramConfig) -> Path:
    home = config.qualification_home
    homes_root = home.parent
    try:
        homes_root.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise ProgramLaunchError(
            "qualification_home_invalid", "qualification HOME root cannot be prepared"
        ) from exc
    _private_program_directory(homes_root, noun="qualification HOME root")
    try:
        return _validate_qualification_home(config)
    except ProgramLaunchError:
        if home.exists() or home.is_symlink():
            _quarantine_qualification_home(config)
    with tempfile.TemporaryDirectory(
        prefix=f".{home.name}.staging-", dir=homes_root
    ) as raw_staging:
        staging = Path(raw_staging)
        staging.chmod(0o700)
        extension_home = staging / ".duckdb" / "extensions" / "v1.5.5" / "linux_arm64"
        extension_home.mkdir(parents=True, mode=0o700)
        for parent in (
            staging / ".duckdb",
            staging / ".duckdb" / "extensions",
            extension_home.parent,
        ):
            parent.chmod(0o700)
        for name, expected_sha256 in config.extension_hashes.items():
            _copy_exact_extension_file(
                config.extension_directory / name,
                extension_home / name,
                expected_sha256=expected_sha256,
            )
        cache_root = staging / ".cache"
        for directory in (
            cache_root,
            cache_root / "cuda",
            cache_root / "ipfs_accelerate",
            cache_root / "xdg",
        ):
            directory.mkdir(mode=TRUSTED_CACHE_DIRECTORY_MODE)
        for directory in (
            staging / ".duckdb",
            staging / ".duckdb" / "extensions",
            extension_home.parent,
            extension_home,
            staging,
        ):
            directory.chmod(TRUSTED_HOME_DIRECTORY_MODE)
        _validate_qualification_home(config, home=staging)
        try:
            os.rename(staging, home)
        except OSError:
            # A concurrent launcher may have published the same content-bound
            # HOME. Admit only the independently revalidated exact artifact.
            return _validate_qualification_home(config)
        _fsync_program_directory(homes_root)
    return _validate_qualification_home(config)


def _qualification_environment(config: ProgramConfig) -> dict[str, str]:
    """Build a persistent private HOME containing only the pinned DuckDB extensions."""

    verify_extension_files(config)
    try:
        config.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        state_info = os.lstat(config.state_root)
    except OSError as exc:
        raise ProgramLaunchError(
            "qualification_home_invalid", "program state root cannot be prepared"
        ) from exc
    if (
        not stat.S_ISDIR(state_info.st_mode)
        or stat.S_ISLNK(state_info.st_mode)
        or state_info.st_uid != os.geteuid()
        or stat.S_IMODE(state_info.st_mode) != 0o700
    ):
        raise ProgramLaunchError(
            "qualification_home_invalid", "program state root is not private"
        )
    home = _build_qualification_home(config)
    environment = _bounded_environment()
    environment["HOME"] = str(home)
    environment[TRUSTED_DUCKDB_HOME_ENV] = str(home)
    user_base = Path(site.getuserbase()).resolve()
    user_site = Path(site.getusersitepackages()).resolve()
    if (
        not user_base.is_dir()
        or not user_site.is_dir()
        or str(user_site) not in sys.path
        or user_base not in user_site.parents
    ):
        raise ProgramLaunchError(
            "python_environment_invalid",
            "active Python user site is unavailable for isolated qualification",
        )
    environment["PYTHONUSERBASE"] = str(user_base)
    cache_root = home / ".cache"
    cache_paths = {
        "CUDA_CACHE_PATH": cache_root / "cuda",
        "XDG_CACHE_HOME": cache_root / "xdg",
    }
    for name in TRUSTED_RUNTIME_PATH_ENV_NAMES:
        path = cache_paths[name]
        environment[name] = str(path)
    environment.update(TRUSTED_RUNTIME_FLAG_ENV)
    return environment


def _revalidate_qualification_home_after_child(config: ProgramConfig) -> None:
    try:
        _validate_qualification_home(config)
    except ProgramLaunchError:
        if config.qualification_home.exists() or config.qualification_home.is_symlink():
            _quarantine_qualification_home(config)
        raise


def _default_remote_probe(config: ProgramConfig, *, tree: str) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    names = {
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": None,
        "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": config.secret_handle,
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": config.store_id,
        "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(config.repo_root),
    }
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name, value in names.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        with DatabaseTaskSource(
            config.endpoint,
            owner_id=f"{PROGRAM}:launcher-readiness",
            repository_tree_id=tree,
            install_schema=False,
        ) as source:
            snapshot = source.snapshot()
            page = source.list_tasks(limit=64)
            ready = source.ready_tasks(limit=64)
            statuses = Counter(item.status for item in page.tasks)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    unsigned = {
        "schema": REMOTE_PROBE_SCHEMA,
        "transport": "quack",
        "endpoint": config.endpoint,
        "secret_handle": config.secret_handle,
        "repository_tree": tree,
        "projection_cid": snapshot.projection_cid,
        "task_count": snapshot.task_count,
        "ready_task_ids": [item.task_alias for item in ready.tasks],
        "blocked_task_ids": [item.task_alias for item in page.tasks if item.status == "blocked"],
        "status_counts": dict(sorted(statuses.items())),
        "authenticated": True,
        "query_surface": "DatabaseTaskSource.snapshot+list_tasks+ready_tasks",
    }
    return {**unsigned, "probe_cid": content_identity(unsigned)}


RemoteProbe = Callable[[ProgramConfig], Mapping[str, Any]]


class ProcedureCompilerProgramLauncher:
    def __init__(
        self,
        *,
        repo_root: Path = REPO_ROOT,
        config_path: Path | None = None,
        runner: CommandRunner = _default_runner,
        clock: Callable[[], str] = _utc_now,
        sleeper: Callable[[float], None] = time.sleep,
        remote_probe: Callable[[ProgramConfig, str], Mapping[str, Any]] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config = load_program_config(self.repo_root, config_path)
        self.runner = runner
        self.clock = clock
        self.sleeper = sleeper
        self.remote_probe = remote_probe

    def _run(
        self,
        argv: Sequence[str],
        *,
        timeout: float = 60,
        env: Mapping[str, str] | None = None,
    ) -> CommandResult:
        return self.runner(argv, cwd=self.repo_root, env=env, timeout=timeout)

    def _git(self, *args: str) -> str:
        result = self._run(("git", *args), timeout=60, env=_bounded_environment())
        if result.returncode:
            raise ProgramLaunchError("git_failed", result.stderr.strip() or "Git command failed")
        return result.stdout.strip()

    def repository_identity(self, *, require_clean: bool) -> tuple[str, str]:
        if self._git("rev-parse", "--show-toplevel") != str(self.repo_root):
            raise ProgramLaunchError("repository_mismatch", "repository root is foreign")
        branch = self._git("branch", "--show-current")
        head = self._git("rev-parse", "HEAD")
        tree = self._git("rev-parse", "HEAD^{tree}")
        if branch != self.config.branch:
            raise ProgramLaunchError("branch_mismatch", f"required branch {self.config.branch!r}")
        ancestor = self._run(
            ("git", "merge-base", "--is-ancestor", self.config.ancestor, "HEAD"),
            env=_bounded_environment(),
        )
        if ancestor.returncode:
            raise ProgramLaunchError(
                "ancestor_mismatch", "required starting commit is not an ancestor"
            )
        tracked = self._run(
            ("git", "ls-files", "--error-unmatch", CONFIG_RELATIVE),
            env=_bounded_environment(),
        )
        if tracked.returncode:
            raise ProgramLaunchError("config_untracked", "scheduler config is not committed")
        working_blob = self._git("hash-object", CONFIG_RELATIVE)
        head_blob = self._git("rev-parse", f"HEAD:{CONFIG_RELATIVE}")
        if working_blob != head_blob:
            raise ProgramLaunchError("config_drift", "scheduler config differs from HEAD")
        if require_clean and self._git("status", "--porcelain=v1", "--untracked-files=all"):
            raise ProgramLaunchError("checkout_dirty", "launch requires a clean checkout")
        return head, tree

    def _verify_runtime(self) -> None:
        executable = Path(self.config.runtime_executable)
        if (
            executable.is_symlink()
            or not executable.is_file()
            or not os.access(executable, os.X_OK)
        ):
            raise ProgramLaunchError(
                "runtime_invalid", "Docker executable is not a real executable"
            )
        info_result = self._run(
            (
                self.config.runtime_executable,
                "--host",
                self.config.runtime_endpoint,
                "info",
                "--format",
                "{{json .}}",
            ),
            env=_bounded_environment(),
        )
        if info_result.returncode:
            raise ProgramLaunchError(
                "runtime_unavailable", "rootless Docker endpoint is unavailable"
            )
        info = _decode_json_object(info_result.stdout.strip(), noun="Docker info")
        security = info.get("SecurityOptions")
        architecture = str(info.get("Architecture") or "")
        expected_arch = self.config.image_architecture
        arch_ok = architecture == expected_arch or {architecture, expected_arch} == {
            "aarch64",
            "arm64",
        }
        if (
            not isinstance(security, list)
            or "name=rootless" not in security
            or info.get("OSType") != self.config.image_os
            or not arch_ok
        ):
            raise ProgramLaunchError(
                "runtime_not_rootless", "Docker endpoint is not the admitted rootless runtime"
            )
        image_result = self._run(
            (
                self.config.runtime_executable,
                "--host",
                self.config.runtime_endpoint,
                "image",
                "inspect",
                self.config.image_id,
            ),
            env=_bounded_environment(),
        )
        if image_result.returncode:
            raise ProgramLaunchError("image_unavailable", "exact owner image is unavailable")
        image_values = json.loads(image_result.stdout)
        if (
            not isinstance(image_values, list)
            or len(image_values) != 1
            or not isinstance(image_values[0], Mapping)
        ):
            raise ProgramLaunchError("image_inspect_invalid", "Docker image inspect is malformed")
        image = image_values[0]
        image_config = image.get("Config")
        labels = image_config.get("Labels") if isinstance(image_config, Mapping) else None
        if (
            image.get("Id") != self.config.image_id
            or image.get("Os") != self.config.image_os
            or image.get("Architecture") != self.config.image_architecture
            or not isinstance(labels, Mapping)
            or labels.get(RUNTIME_LABEL_KEY) != self.config.image_label
        ):
            raise ProgramLaunchError(
                "image_identity_mismatch", "owner image identity or label drifted"
            )
        verify_extension_files(self.config)

    def _inspect_container(self, *, allow_absent: bool) -> Mapping[str, Any] | None:
        result = self._run(
            (
                self.config.runtime_executable,
                "--host",
                self.config.runtime_endpoint,
                "container",
                "inspect",
                OWNER_CONTAINER_NAME,
            ),
            env=_bounded_environment(),
        )
        if result.returncode:
            text = f"{result.stdout}\n{result.stderr}".lower()
            if allow_absent and "no such" in text:
                return None
            raise ProgramLaunchError(
                "container_inspect_failed", "cannot inspect the owner container"
            )
        values = json.loads(result.stdout)
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], Mapping):
            raise ProgramLaunchError("container_inspect_invalid", "container inspect is malformed")
        return values[0]

    def _quarantine_failed_owner_state(self, *, attempt_identity: str) -> Path:
        """Atomically preserve one failed attempt without touching authority data."""

        if re.fullmatch(r"(?:[0-9a-f]{64}|attempt-[0-9a-f]{32})", attempt_identity) is None:
            raise ProgramLaunchError(
                "owner_cleanup_failed", "failed attempt identity is not closed"
            )
        try:
            state_info = os.lstat(self.config.state_dir)
        except OSError as exc:
            raise ProgramLaunchError(
                "owner_cleanup_failed", "failed owner state directory is unavailable"
            ) from exc
        if (
            not stat.S_ISDIR(state_info.st_mode)
            or stat.S_ISLNK(state_info.st_mode)
            or state_info.st_uid != os.geteuid()
            or stat.S_IMODE(state_info.st_mode) != 0o700
        ):
            raise ProgramLaunchError(
                "owner_cleanup_failed", "failed owner state directory is unsafe"
            )
        quarantine_root = self.config.owner_write_root / "quack-owner-quarantine"
        try:
            quarantine_root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        try:
            quarantine_info = os.lstat(quarantine_root)
            entries = tuple(islice(quarantine_root.iterdir(), MAX_FAILED_OWNER_QUARANTINES + 1))
        except OSError as exc:
            raise ProgramLaunchError(
                "owner_cleanup_failed", "owner quarantine cannot be inspected"
            ) from exc
        if (
            not stat.S_ISDIR(quarantine_info.st_mode)
            or stat.S_ISLNK(quarantine_info.st_mode)
            or quarantine_info.st_uid != os.geteuid()
            or stat.S_IMODE(quarantine_info.st_mode) != 0o700
            or len(entries) >= MAX_FAILED_OWNER_QUARANTINES
        ):
            raise ProgramLaunchError(
                "owner_cleanup_failed", "owner quarantine is unsafe or at capacity"
            )
        target = quarantine_root / attempt_identity
        if target.exists() or target.is_symlink():
            raise ProgramLaunchError(
                "owner_cleanup_failed", "failed owner quarantine target already exists"
            )
        try:
            os.rename(self.config.state_dir, target)
            for directory_path in (quarantine_root, self.config.owner_write_root):
                descriptor = os.open(directory_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except OSError as exc:
            raise ProgramLaunchError(
                "owner_cleanup_failed", "failed owner state could not be quarantined"
            ) from exc
        return target

    def _cleanup_failed_owner_attempt(
        self,
        *,
        head: str,
        tree: str,
        container_id: str,
        attempt_identity: str,
    ) -> Path:
        """Remove only the inspect-proven attempt container, then quarantine state."""

        inspect = self._inspect_container(allow_absent=True)
        quarantine_identity = container_id or attempt_identity
        if inspect is not None:
            observed_id = validate_owner_container_inspect(
                inspect,
                config=self.config,
                head=head,
                tree=tree,
                container_id=container_id or None,
                require_running=False,
            )
            quarantine_identity = observed_id
            state = inspect.get("State")
            if not isinstance(state, Mapping):
                raise ProgramLaunchError(
                    "owner_cleanup_failed", "failed owner state is not inspect-bound"
                )
            if state.get("Running") is True:
                stopped = self._run(
                    (
                        self.config.runtime_executable,
                        "--host",
                        self.config.runtime_endpoint,
                        "container",
                        "stop",
                        "--time",
                        "5",
                        observed_id,
                    ),
                    timeout=30,
                    env=_bounded_environment(),
                )
                if stopped.returncode or stopped.stdout.strip() != observed_id:
                    raise ProgramLaunchError(
                        "owner_cleanup_failed", "failed owner container did not stop"
                    )
            removed = self._run(
                (
                    self.config.runtime_executable,
                    "--host",
                    self.config.runtime_endpoint,
                    "container",
                    "rm",
                    observed_id,
                ),
                timeout=30,
                env=_bounded_environment(),
            )
            if removed.returncode or removed.stdout.strip() != observed_id:
                raise ProgramLaunchError(
                    "owner_cleanup_failed", "failed owner container was not removed"
                )
        remaining = self._inspect_container(allow_absent=True)
        if remaining is not None:
            raise ProgramLaunchError(
                "owner_cleanup_failed", "an owner container remains after cleanup"
            )
        return self._quarantine_failed_owner_state(attempt_identity=quarantine_identity)

    def _materialization_verify(self, *, head: str, tree: str) -> dict[str, Any]:
        result = self._run(
            (sys.executable, str(self.repo_root / MATERIALIZER_RELATIVE), "--verify"),
            timeout=3600,
            env=_qualification_environment(self.config),
        )
        _revalidate_qualification_home_after_child(self.config)
        if result.returncode:
            raise ProgramLaunchError(
                "materialization_unqualified", "current-tree materialization verification failed"
            )
        value = _decode_json_object(result.stdout, noun="materialization verification")
        if set(value) != _MATERIALIZATION_FIELDS:
            raise ProgramLaunchError(
                "materialization_invalid", "materialization verification is not closed"
            )
        if (
            value.get("schema") != MATERIALIZATION_VERIFICATION_SCHEMA
            or value.get("valid") is not True
            or value.get("repository_commit") != head
            or value.get("repository_tree") != tree
            or value.get("database_path") != self.config.store_id
            or value.get("task_count") != 32
            or value.get("blocked_task_ids") != []
            or value.get("projection_matches_events") is not True
            or value.get("plan_current") is not True
            or value.get("tasks_current") is not True
            or value.get("qualification_current") is not True
            or value.get("freshly_qualified") is not True
        ):
            raise ProgramLaunchError(
                "materialization_stale", "materialization is not exact-current-tree qualified"
            )
        return value

    def _probe(self, *, tree: str) -> dict[str, Any]:
        if self.remote_probe is None:
            value = dict(_default_remote_probe(self.config, tree=tree))
        else:
            value = dict(self.remote_probe(self.config, tree))
        if (
            value.get("schema") != REMOTE_PROBE_SCHEMA
            or value.get("endpoint") != self.config.endpoint
            or value.get("secret_handle") != self.config.secret_handle
            or value.get("repository_tree") != tree
            or value.get("authenticated") is not True
            or type(value.get("task_count")) is not int
            or value.get("task_count", 0) < 1
            or value.get("blocked_task_ids") != []
        ):
            raise ProgramLaunchError(
                "remote_probe_failed", "handle-only remote task-source probe is invalid"
            )
        return value

    def _read_owner_status(self, *, head: str, tree: str) -> tuple[dict[str, Any], dict[str, Any]]:
        status = _safe_read_json(
            self.config.state_dir / "quack-state-server.status.json",
            exact_fields=_STATUS_FIELDS,
            noun="owner status",
        )
        identity = validate_owner_status_payload(status, config=self.config, head=head, tree=tree)
        return status, identity

    def owner_plan(self) -> dict[str, Any]:
        head, tree = self.repository_identity(require_clean=True)
        self._verify_runtime()
        argv = build_owner_create_argv(self.config, head=head, tree=tree)
        unsigned = {
            "schema": PLAN_SCHEMA,
            "program": PROGRAM,
            "action": "owner-start",
            "repository_commit": head,
            "repository_tree": tree,
            "config_sha256": self.config.config_sha256,
            "container_name": OWNER_CONTAINER_NAME,
            "image_id": self.config.image_id,
            "owner_write_root": str(self.config.owner_write_root),
            "database_path": str(self.config.database_path),
            "state_dir": str(self.config.state_dir),
            "endpoint": self.config.endpoint,
            "secret_handle": self.config.secret_handle,
            "argv": argv,
            "mutates": False,
        }
        return {**unsigned, "receipt_cid": content_identity(unsigned)}

    def owner_start(self) -> dict[str, Any]:
        head, tree = self.repository_identity(require_clean=True)
        self._verify_runtime()
        existing = self._inspect_container(allow_absent=True)
        if existing is not None:
            container_id = validate_owner_container_inspect(
                existing,
                config=self.config,
                head=head,
                tree=tree,
                require_running=True,
            )
            isolation = _safe_read_json(
                self.config.isolation_receipt_path,
                exact_fields=_OWNER_ISOLATION_FIELDS,
                noun="owner isolation receipt",
            )
            validate_owner_isolation_receipt(
                isolation, config=self.config, container_id=container_id
            )
            status, identity = self._read_owner_status(head=head, tree=tree)
            probe = self._probe(tree=tree)
            unsigned = {
                "schema": OWNER_START_RECEIPT_SCHEMA,
                "program": PROGRAM,
                "repository_commit": head,
                "repository_tree": tree,
                "config_sha256": self.config.config_sha256,
                "image_id": self.config.image_id,
                "container_name": OWNER_CONTAINER_NAME,
                "container_id": container_id,
                "owner_isolation_receipt_cid": isolation["receipt_cid"],
                "owner_identity": identity,
                "remote_probe": probe,
                "materialization_verification": None,
                "reused": True,
                "started_at": str(status.get("identity", {}).get("started_at") or ""),
            }
            receipt = {**unsigned, "receipt_cid": content_identity(unsigned)}
            _persist_receipt(self.config, receipt)
            return receipt
        if self.config.isolation_receipt_path.exists():
            raise ProgramLaunchError(
                "orphaned_owner_state",
                "owner isolation receipt exists without its exact container; refusing takeover",
            )
        if not self.config.database_path.is_file() or self.config.database_path.is_symlink():
            raise ProgramLaunchError(
                "database_unavailable", "materialized control database is absent"
            )
        materialization = self._materialization_verify(head=head, tree=tree)
        # The exact port must be unowned before Docker create.  This observation
        # cannot authorize a bind; it only detects an obvious conflicting owner.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex((self.config.host, self.config.port)) == 0:
                raise ProgramLaunchError(
                    "endpoint_in_use", "configured Quack endpoint is already in use"
                )
        self.config.owner_write_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.config.state_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        attempt_identity = f"attempt-{uuid.uuid4().hex}"
        container_id = ""
        try:
            create = self._run(
                build_owner_create_argv(self.config, head=head, tree=tree),
                timeout=120,
                env=_bounded_environment(),
            )
            if create.returncode:
                raise ProgramLaunchError(
                    "container_create_failed", "Docker refused owner container creation"
                )
            raw_container_id = create.stdout.strip()
            if not CONTAINER_ID.fullmatch(raw_container_id):
                raise ProgramLaunchError(
                    "container_identity_invalid",
                    "Docker create did not return a full container ID",
                )
            container_id = raw_container_id
            inspect = self._inspect_container(allow_absent=False)
            assert inspect is not None
            validate_owner_container_inspect(
                inspect,
                config=self.config,
                head=head,
                tree=tree,
                container_id=container_id,
                require_running=False,
            )
            isolation = build_owner_isolation_receipt(
                self.config, container_id=container_id, issued_at=self.clock()
            )
            _atomic_create(self.config.isolation_receipt_path, isolation)
            start = self._run(
                (
                    self.config.runtime_executable,
                    "--host",
                    self.config.runtime_endpoint,
                    "container",
                    "start",
                    container_id,
                ),
                timeout=60,
                env=_bounded_environment(),
            )
            if start.returncode or start.stdout.strip() != container_id:
                raise ProgramLaunchError(
                    "container_start_failed", "Docker owner container did not start"
                )
            deadline = time.monotonic() + MAX_OWNER_WAIT_SECONDS
            last_error = "owner readiness was not published"
            while time.monotonic() < deadline:
                try:
                    running = self._inspect_container(allow_absent=False)
                    assert running is not None
                    validate_owner_container_inspect(
                        running,
                        config=self.config,
                        head=head,
                        tree=tree,
                        container_id=container_id,
                        require_running=True,
                    )
                    _status, identity = self._read_owner_status(head=head, tree=tree)
                    probe = self._probe(tree=tree)
                    break
                except ProgramLaunchError as exc:
                    last_error = f"{exc.code}: {exc}"
                    self.sleeper(0.5)
            else:
                raise ProgramLaunchError("owner_readiness_timeout", last_error)
            unsigned = {
                "schema": OWNER_START_RECEIPT_SCHEMA,
                "program": PROGRAM,
                "repository_commit": head,
                "repository_tree": tree,
                "config_sha256": self.config.config_sha256,
                "image_id": self.config.image_id,
                "container_name": OWNER_CONTAINER_NAME,
                "container_id": container_id,
                "owner_isolation_receipt_cid": isolation["receipt_cid"],
                "owner_identity": identity,
                "remote_probe": probe,
                "materialization_verification": materialization,
                "reused": False,
                "started_at": self.clock(),
            }
            receipt = {**unsigned, "receipt_cid": content_identity(unsigned)}
            _persist_receipt(self.config, receipt)
            return receipt
        except Exception as exc:
            launch_error = (
                exc
                if isinstance(exc, ProgramLaunchError)
                else ProgramLaunchError("owner_launch_failed", type(exc).__name__)
            )
            try:
                quarantine_path = self._cleanup_failed_owner_attempt(
                    head=head,
                    tree=tree,
                    container_id=container_id,
                    attempt_identity=attempt_identity,
                )
            except ProgramLaunchError as cleanup_error:
                raise ProgramLaunchError(
                    "owner_cleanup_failed",
                    f"{launch_error.code}; {cleanup_error}",
                ) from exc
            raise ProgramLaunchError(
                launch_error.code,
                f"{launch_error}; failed attempt quarantined at {quarantine_path}",
            ) from exc

    def owner_status(self) -> dict[str, Any]:
        head, tree = self.repository_identity(require_clean=False)
        self._verify_runtime()
        inspect = self._inspect_container(allow_absent=True)
        if inspect is None:
            unsigned = {
                "schema": OWNER_STATUS_RECEIPT_SCHEMA,
                "program": PROGRAM,
                "repository_commit": head,
                "repository_tree": tree,
                "config_sha256": self.config.config_sha256,
                "image_id": self.config.image_id,
                "container_name": OWNER_CONTAINER_NAME,
                "container_id": "",
                "owner_isolation_receipt_cid": None,
                "running": False,
                "ready": False,
                "owner_identity": None,
                "remote_probe": None,
                "observed_at": self.clock(),
            }
            return {**unsigned, "receipt_cid": content_identity(unsigned)}
        container_id = validate_owner_container_inspect(
            inspect, config=self.config, head=head, tree=tree, require_running=True
        )
        isolation = _safe_read_json(
            self.config.isolation_receipt_path,
            exact_fields=_OWNER_ISOLATION_FIELDS,
            noun="owner isolation receipt",
        )
        validate_owner_isolation_receipt(isolation, config=self.config, container_id=container_id)
        _status, identity = self._read_owner_status(head=head, tree=tree)
        probe = self._probe(tree=tree)
        unsigned = {
            "schema": OWNER_STATUS_RECEIPT_SCHEMA,
            "program": PROGRAM,
            "repository_commit": head,
            "repository_tree": tree,
            "config_sha256": self.config.config_sha256,
            "image_id": self.config.image_id,
            "container_name": OWNER_CONTAINER_NAME,
            "container_id": container_id,
            "owner_isolation_receipt_cid": isolation["receipt_cid"],
            "running": True,
            "ready": True,
            "owner_identity": identity,
            "remote_probe": probe,
            "observed_at": self.clock(),
        }
        return {**unsigned, "receipt_cid": content_identity(unsigned)}

    def supervisor_plan(self) -> dict[str, Any]:
        head, tree = self.repository_identity(require_clean=True)
        argv = [
            sys.executable,
            str(self.repo_root / SCHEDULER_RELATIVE),
            "--repo-root",
            str(self.repo_root),
            "--config",
            str(self.config.config_path),
            "launch",
            "--implement",
            "--duration-seconds",
            str(MAX_SUPERVISOR_DURATION_SECONDS),
        ]
        unsigned = {
            "schema": PLAN_SCHEMA,
            "program": PROGRAM,
            "action": "supervisor-start",
            "repository_commit": head,
            "repository_tree": tree,
            "config_sha256": self.config.config_sha256,
            "argv": argv,
            "mutates": False,
        }
        return {**unsigned, "receipt_cid": content_identity(unsigned)}

    def supervisor_start(self, *, owner_receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
        head, tree = self.repository_identity(require_clean=True)
        supplied = None if owner_receipt is None else dict(owner_receipt)
        if supplied is not None:
            supplied_cid = supplied.get("receipt_cid")
            supplied_unsigned = dict(supplied)
            supplied_unsigned.pop("receipt_cid", None)
            if (
                supplied.get("schema") != OWNER_START_RECEIPT_SCHEMA
                or supplied.get("program") != PROGRAM
                or supplied.get("repository_commit") != head
                or supplied.get("repository_tree") != tree
                or supplied_cid != content_identity(supplied_unsigned)
            ):
                raise ProgramLaunchError(
                    "owner_not_admitted", "supplied owner launch receipt is invalid"
                )
        # A CID proves only receipt identity. Re-observe Docker, isolation,
        # owner status, and the authenticated handle-only query at admission.
        owner = self.owner_status()
        probe = owner.get("remote_probe")
        if (
            owner.get("schema") != OWNER_STATUS_RECEIPT_SCHEMA
            or owner.get("program") != PROGRAM
            or owner.get("repository_commit") != head
            or owner.get("repository_tree") != tree
            or owner.get("container_id") in {None, ""}
            or owner.get("running") is not True
            or owner.get("ready") is not True
            or not isinstance(probe, Mapping)
            or probe.get("blocked_task_ids") != []
            or (supplied is not None and supplied.get("container_id") != owner.get("container_id"))
        ):
            raise ProgramLaunchError("owner_not_admitted", "exact ready owner receipt is required")
        existing_pid = self.config.state_root / "configured-board-master.pid"
        if existing_pid.exists():
            raise ProgramLaunchError(
                "coordinator_exists", "coordinator PID projection already exists"
            )
        argv = self.supervisor_plan()["argv"]
        result = self._run(argv, timeout=180, env=_qualification_environment(self.config))
        _revalidate_qualification_home_after_child(self.config)
        if result.returncode:
            raise ProgramLaunchError(
                "supervisor_launch_failed", "configured scheduler launch failed"
            )
        plan = _decode_json_object(result.stdout, noun="configured scheduler launch receipt")
        if set(plan) != _SUPERVISOR_LAUNCH_FIELDS:
            raise ProgramLaunchError(
                "supervisor_receipt_invalid",
                "configured scheduler launch receipt is not closed",
            )
        pid = plan.get("coordinator_pid")
        pid_path = Path(str(plan.get("coordinator_pid_path") or ""))
        log_path = Path(str(plan.get("coordinator_log") or ""))
        expected_pid_path = self.config.state_root / "configured-board-master.pid"
        if (
            type(pid) is not int
            or pid < 2
            or pid_path != expected_pid_path
            or not log_path.is_absolute()
            or log_path.parent
            != _resolved_inside(
                self.repo_root,
                _canonical_relative(
                    _decode_json_object(
                        self.config.config_path.read_text(encoding="utf-8"), noun="scheduler config"
                    )["runtime_paths"]["logs"],
                    field="runtime_paths.logs",
                ),
                field="runtime_paths.logs",
            )
        ):
            raise ProgramLaunchError(
                "supervisor_receipt_invalid", "configured scheduler launch identity is invalid"
            )
        try:
            info = os.lstat(pid_path)
            log_info = os.lstat(log_path)
            pid_raw = pid_path.read_bytes()
            os.kill(pid, 0)
        except (OSError, ValueError) as exc:
            raise ProgramLaunchError("coordinator_not_live", "coordinator PID is not live") from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or pid_raw != f"{pid}\n".encode("ascii")
            or stat.S_ISLNK(log_info.st_mode)
            or not stat.S_ISREG(log_info.st_mode)
            or log_info.st_uid != os.geteuid()
            or stat.S_IMODE(log_info.st_mode) != 0o600
            or log_info.st_nlink != 1
        ):
            raise ProgramLaunchError(
                "coordinator_pid_forged", "coordinator PID projection is unsafe"
            )
        self.sleeper(1.0)
        try:
            os.kill(pid, 0)
        except OSError as exc:
            raise ProgramLaunchError(
                "coordinator_early_exit", "coordinator exited during launch admission"
            ) from exc
        probe = self._probe(tree=tree)
        if probe.get("blocked_task_ids") != []:
            raise ProgramLaunchError("supervisor_blocked", "control state contains blocked tasks")
        unsigned = {
            "schema": SUPERVISOR_START_RECEIPT_SCHEMA,
            "program": PROGRAM,
            "repository_commit": head,
            "repository_tree": tree,
            "config_sha256": self.config.config_sha256,
            "owner_status_receipt_cid": owner.get("receipt_cid"),
            "owner_start_receipt_cid": (None if supplied is None else supplied.get("receipt_cid")),
            "owner_container_id": owner.get("container_id"),
            "coordinator_pid": pid,
            "coordinator_pid_path": str(pid_path),
            "coordinator_log": str(log_path),
            "scheduler_argv": argv,
            "remote_probe": probe,
            "implement": True,
            "detached": True,
            "started_at": self.clock(),
        }
        receipt = {**unsigned, "receipt_cid": content_identity(unsigned)}
        _persist_receipt(self.config, receipt)
        return receipt

    def start(self) -> dict[str, Any]:
        owner = self.owner_start()
        supervisor = self.supervisor_start(owner_receipt=owner)
        unsigned = {
            "schema": "ipfs_accelerate_py/agent-supervisor/pcpc-program-start@1",
            "program": PROGRAM,
            "owner": owner,
            "supervisor": supervisor,
        }
        return {**unsigned, "receipt_cid": content_identity(unsigned)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("owner-start", "supervisor-start", "start"):
        child = subparsers.add_parser(name)
        child.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("owner-status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        launcher = ProcedureCompilerProgramLauncher(
            repo_root=args.repo_root,
            config_path=args.config,
        )
        if args.command == "owner-status":
            result = launcher.owner_status()
        elif args.command == "owner-start":
            result = launcher.owner_plan() if args.dry_run else launcher.owner_start()
        elif args.command == "supervisor-start":
            result = launcher.supervisor_plan() if args.dry_run else launcher.supervisor_start()
        elif args.command == "start" and args.dry_run:
            unsigned = {
                "schema": PLAN_SCHEMA,
                "program": PROGRAM,
                "action": "start",
                "owner": launcher.owner_plan(),
                "supervisor": launcher.supervisor_plan(),
                "mutates": False,
            }
            result = {**unsigned, "receipt_cid": content_identity(unsigned)}
        else:
            result = launcher.start()
        sys.stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        return 0
    except ProgramLaunchError as exc:
        unsigned = {
            "schema": ERROR_SCHEMA,
            "program": PROGRAM,
            "valid": False,
            "error_code": exc.code,
            "error": str(exc),
        }
        payload = {**unsigned, "receipt_cid": content_identity(unsigned)}
        sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        return 2
    except Exception as exc:  # noqa: BLE001 - closed CLI failure projection
        unsigned = {
            "schema": ERROR_SCHEMA,
            "program": PROGRAM,
            "valid": False,
            "error_code": "unexpected_launch_failure",
            "error": type(exc).__name__,
        }
        payload = {**unsigned, "receipt_cid": content_identity(unsigned)}
        sys.stdout.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
