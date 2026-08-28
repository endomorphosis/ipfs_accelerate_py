"""Board-parameterized admission for a sealed configured-board control plane.

The accepted control-plane archive already provides the immutable Python
closure used by the scheduler, runner, supervisor, and daemon.  This adjacent
contract binds that archive to one exact configured board and to the
operator-owned controls that may influence its execution.  It deliberately
does not carry credentials and grants no task-completion authority.

The older ``--require-configured-board-live-seal`` path remains an unconditional
NO-GO.  Callers must opt in through a closed scheduler policy and present this
complete admission together with an ``accepted-control-plane@2`` descriptor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ...agent_implementation_route import (
    AgentSupervisorNativeDependencyLaunch,
    verify_agent_supervisor_native_dependency_sealed_fd,
)
from ...llm_router import (
    AgentImplementationControlPlanePin,
    verify_agent_implementation_sealed_control_plane,
)
from ..core.multiformats_identity import cid_for_dag_json, validate_cid
from .configured_board_extension_projection import (
    CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
    ConfiguredBoardExtensionPin,
    parse_configured_board_extension_pin,
    verify_configured_board_extension_home,
)

CONFIGURED_BOARD_LIVE_CAPSULE_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "configured-board-live-control-capsule-policy@1"
)
CONFIGURED_BOARD_LIVE_CAPSULE_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "configured-board-live-control-capsule-admission@1"
)

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_POLICY_FIELDS: Final = frozenset({"schema", "required", "control_paths"})
_ARTIFACT_FIELDS: Final = frozenset({"path", "sha256", "size"})
_ADMISSION_FIELDS: Final = frozenset(
    {
        "schema",
        "board_namespace",
        "plan_revision",
        "task_prefix",
        "config_path",
        "configuration_root",
        "source_head",
        "source_tree",
        "control_plane_capsule_id",
        "control_plane_archive_sha256",
        "native_authorization_id",
        "native_dependency_id",
        "native_python_executable_sha256",
        "quack_extension_projection",
        "database_authority",
        "max_lanes",
        "strict_task_sharding",
        "control_artifacts",
        "admission_cid",
    }
)
_DATABASE_FIELDS: Final = frozenset(
    {
        "authority_mode",
        "task_source_kind",
        "schema_revision",
        "failover_policy",
        "store_id",
        "store_generation",
        "endpoint_secret_handle",
    }
)


class ConfiguredBoardLiveCapsuleError(ValueError):
    """The live configured-board capsule is incomplete or has drifted."""


def _reject_duplicate_json_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfiguredBoardLiveCapsuleError(
                f"configured-board live capsule repeats JSON key {key!r}"
            )
        result[key] = value
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _cid(value: Mapping[str, Any]) -> str:
    return cid_for_dag_json(dict(value), for_identity=True)


def _text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in "\x00\r\n")
    ):
        raise ConfiguredBoardLiveCapsuleError(f"{field} is invalid")
    return value


def _relative(value: object, field: str) -> str:
    rendered = _text(value, field)
    path = PurePosixPath(rendered)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in rendered
        or path.as_posix() != rendered
    ):
        raise ConfiguredBoardLiveCapsuleError(f"{field} is not a safe relative path")
    return rendered


def _exact_sha256(value: object, field: str) -> str:
    rendered = _text(value, field)
    if _SHA256.fullmatch(rendered) is None:
        raise ConfiguredBoardLiveCapsuleError(f"{field} is not a sha256 identity")
    return rendered


def _exact_content_id(value: object, field: str) -> str:
    rendered = _text(value, field)
    try:
        validated = validate_cid(rendered, codecs=("dag-json",))
    except ValueError as exc:
        raise ConfiguredBoardLiveCapsuleError(
            f"{field} is not a canonical CIDv1 DAG-JSON identity"
        ) from exc
    return validated


def _exact_git(value: object, field: str) -> str:
    rendered = _text(value, field)
    if _GIT_OBJECT.fullmatch(rendered) is None:
        raise ConfiguredBoardLiveCapsuleError(f"{field} is not a Git object")
    return rendered


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfiguredBoardLiveCapsuleError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfiguredBoardLiveCapsuleError(
            f"{field} must be a nonnegative integer"
        )
    return value


def parse_configured_board_live_capsule_policy(
    value: object,
) -> tuple[str, ...]:
    """Return the exact sorted control paths for a required live policy."""

    if type(value) is not dict or set(value) != _POLICY_FIELDS:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule policy fields are noncanonical"
        )
    if value.get("schema") != CONFIGURED_BOARD_LIVE_CAPSULE_POLICY_SCHEMA:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule policy schema is unsupported"
        )
    if value.get("required") is not True:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule policy must fail closed as required"
        )
    raw_paths = value.get("control_paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule control_paths are empty"
        )
    paths = tuple(
        _relative(item, f"control_paths[{index}]")
        for index, item in enumerate(raw_paths)
    )
    if paths != tuple(sorted(set(paths))):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule control_paths are noncanonical"
        )
    return paths


@dataclass(frozen=True, slots=True)
class ConfiguredBoardLiveCapsuleAdmission:
    """Immutable board/control binding carried across every process birth."""

    schema: str
    board_namespace: str
    plan_revision: str
    task_prefix: str
    config_path: str
    configuration_root: str
    source_head: str
    source_tree: str
    control_plane_capsule_id: str
    control_plane_archive_sha256: str
    native_authorization_id: str
    native_dependency_id: str
    native_python_executable_sha256: str
    quack_extension_projection: ConfiguredBoardExtensionPin
    database_authority: Mapping[str, object]
    max_lanes: int
    strict_task_sharding: bool
    control_artifacts: tuple[Mapping[str, object], ...]
    admission_cid: str

    def as_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "board_namespace": self.board_namespace,
            "plan_revision": self.plan_revision,
            "task_prefix": self.task_prefix,
            "config_path": self.config_path,
            "configuration_root": self.configuration_root,
            "source_head": self.source_head,
            "source_tree": self.source_tree,
            "control_plane_capsule_id": self.control_plane_capsule_id,
            "control_plane_archive_sha256": self.control_plane_archive_sha256,
            "native_authorization_id": self.native_authorization_id,
            "native_dependency_id": self.native_dependency_id,
            "native_python_executable_sha256": (
                self.native_python_executable_sha256
            ),
            "quack_extension_projection": (
                self.quack_extension_projection.as_dict()
            ),
            "database_authority": dict(self.database_authority),
            "max_lanes": self.max_lanes,
            "strict_task_sharding": self.strict_task_sharding,
            "control_artifacts": [dict(item) for item in self.control_artifacts],
        }
        if include_identity:
            payload["admission_cid"] = self.admission_cid
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.as_dict()).decode("utf-8")


def parse_configured_board_live_capsule_admission(
    value: str | Mapping[str, object],
) -> ConfiguredBoardLiveCapsuleAdmission:
    """Parse a closed admission and recompute its self identity."""

    try:
        payload = (
            json.loads(value, object_pairs_hook=_reject_duplicate_json_keys)
            if isinstance(value, str)
            else value
        )
    except json.JSONDecodeError as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule admission is not JSON"
        ) from exc
    if type(payload) is not dict or set(payload) != _ADMISSION_FIELDS:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule admission fields are noncanonical"
        )
    if payload.get("schema") != CONFIGURED_BOARD_LIVE_CAPSULE_ADMISSION_SCHEMA:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule admission schema is unsupported"
        )
    authority = payload.get("database_authority")
    if type(authority) is not dict or set(authority) != _DATABASE_FIELDS:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board database authority fields are noncanonical"
        )
    normalized_authority = {
        field: _text(authority.get(field), f"database_authority.{field}")
        for field in _DATABASE_FIELDS - {"store_generation"}
    }
    normalized_authority["store_generation"] = _positive_int(
        authority.get("store_generation"),
        "database_authority.store_generation",
    )
    if normalized_authority["authority_mode"] != "quack":
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live authority must remain Quack"
        )
    if normalized_authority["task_source_kind"] != "duckdb":
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live task source must remain DuckDB"
        )
    if normalized_authority["failover_policy"] != "fail_closed":
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live authority must fail closed"
        )
    artifacts = payload.get("control_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live control artifacts are empty"
        )
    normalized_artifacts: list[dict[str, object]] = []
    for index, artifact in enumerate(artifacts):
        if type(artifact) is not dict or set(artifact) != _ARTIFACT_FIELDS:
            raise ConfiguredBoardLiveCapsuleError(
                "configured-board live control artifact fields are noncanonical"
            )
        normalized_artifacts.append(
            {
                "path": _relative(artifact.get("path"), f"artifact[{index}].path"),
                "sha256": _exact_sha256(
                    artifact.get("sha256"), f"artifact[{index}].sha256"
                ),
                "size": _nonnegative_int(
                    artifact.get("size"), f"artifact[{index}].size"
                ),
            }
        )
    if [item["path"] for item in normalized_artifacts] != sorted(
        {str(item["path"]) for item in normalized_artifacts}
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live control artifacts are noncanonical"
        )
    strict = payload.get("strict_task_sharding")
    if not isinstance(strict, bool):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live strict_task_sharding must be boolean"
        )
    admission = ConfiguredBoardLiveCapsuleAdmission(
        schema=CONFIGURED_BOARD_LIVE_CAPSULE_ADMISSION_SCHEMA,
        board_namespace=_text(payload.get("board_namespace"), "board_namespace"),
        plan_revision=_text(payload.get("plan_revision"), "plan_revision"),
        task_prefix=_text(payload.get("task_prefix"), "task_prefix"),
        config_path=_relative(payload.get("config_path"), "config_path"),
        configuration_root=_exact_content_id(
            payload.get("configuration_root"), "configuration_root"
        ),
        source_head=_exact_git(payload.get("source_head"), "source_head"),
        source_tree=_exact_git(payload.get("source_tree"), "source_tree"),
        control_plane_capsule_id=_exact_sha256(
            payload.get("control_plane_capsule_id"), "control_plane_capsule_id"
        ),
        control_plane_archive_sha256=_exact_sha256(
            payload.get("control_plane_archive_sha256"),
            "control_plane_archive_sha256",
        ),
        native_authorization_id=_exact_sha256(
            payload.get("native_authorization_id"),
            "native_authorization_id",
        ),
        native_dependency_id=_exact_sha256(
            payload.get("native_dependency_id"),
            "native_dependency_id",
        ),
        native_python_executable_sha256=_exact_sha256(
            payload.get("native_python_executable_sha256"),
            "native_python_executable_sha256",
        ),
        quack_extension_projection=parse_configured_board_extension_pin(
            payload.get("quack_extension_projection")
        ),
        database_authority=normalized_authority,
        max_lanes=_positive_int(payload.get("max_lanes"), "max_lanes"),
        strict_task_sharding=strict,
        control_artifacts=tuple(normalized_artifacts),
        admission_cid=_exact_content_id(
            payload.get("admission_cid"), "admission_cid"
        ),
    )
    expected = _cid(admission.as_dict(include_identity=False))
    if admission.admission_cid != expected:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule admission identity drifted"
        )
    return admission


def _stable_regular_bytes(path: Path, *, maximum: int = 8 * 1024 * 1024) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum
        ):
            raise ConfiguredBoardLiveCapsuleError(
                "configured-board live control is not a bounded regular file"
            )
        raw = bytearray()
        while len(raw) <= maximum:
            block = os.read(descriptor, min(65_536, maximum + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) > maximum:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live control changed while it was read"
        )
    return bytes(raw)


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    environment = {
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        env=environment,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ConfiguredBoardLiveCapsuleError(
            f"configured-board Git proof failed: {' '.join(arguments)}"
        )
    return completed.stdout


def _source_generation(root: Path) -> tuple[str, str]:
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board live capsule requires a clean accepted checkout"
        )
    return _exact_git(head, "source_head"), _exact_git(tree, "source_tree")


def _artifact_records(
    root: Path,
    *,
    source_head: str,
    control_paths: Sequence[str],
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for relative in control_paths:
        admitted = _relative(relative, "control path")
        raw = _stable_regular_bytes(root / admitted)
        tracked = _git(root, "show", f"{source_head}:{admitted}")
        if raw != tracked:
            raise ConfiguredBoardLiveCapsuleError(
                f"configured-board control differs from accepted HEAD: {admitted}"
            )
        records.append(
            {
                "path": admitted,
                "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    return tuple(records)


def build_configured_board_live_capsule_admission(
    *,
    repo_root: Path,
    board_namespace: str,
    plan_revision: str,
    task_prefix: str,
    config_path: str,
    configuration_root: str,
    control_paths: Sequence[str],
    control_plane_pin: AgentImplementationControlPlanePin,
    native_authorization_id: str,
    native_dependency_id: str,
    native_python_executable_sha256: str,
    quack_extension_projection: ConfiguredBoardExtensionPin,
    database_authority: Mapping[str, object],
    max_lanes: int,
    strict_task_sharding: bool,
) -> ConfiguredBoardLiveCapsuleAdmission:
    """Bind a clean accepted Git generation and exact controls to a capsule."""

    root = Path(repo_root).resolve(strict=True)
    source_head, source_tree = _source_generation(root)
    if (
        source_head != control_plane_pin.source_head
        or source_tree != control_plane_pin.source_tree
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board source differs from the accepted control plane"
        )
    artifacts = _artifact_records(
        root,
        source_head=source_head,
        control_paths=tuple(control_paths),
    )
    body: dict[str, object] = {
        "schema": CONFIGURED_BOARD_LIVE_CAPSULE_ADMISSION_SCHEMA,
        "board_namespace": board_namespace,
        "plan_revision": plan_revision,
        "task_prefix": task_prefix,
        "config_path": config_path,
        "configuration_root": configuration_root,
        "source_head": source_head,
        "source_tree": source_tree,
        "control_plane_capsule_id": control_plane_pin.capsule_id,
        "control_plane_archive_sha256": control_plane_pin.archive_sha256,
        "native_authorization_id": native_authorization_id,
        "native_dependency_id": native_dependency_id,
        "native_python_executable_sha256": native_python_executable_sha256,
        "quack_extension_projection": quack_extension_projection.as_dict(),
        "database_authority": dict(database_authority),
        "max_lanes": max_lanes,
        "strict_task_sharding": strict_task_sharding,
        "control_artifacts": [dict(item) for item in artifacts],
    }
    body["admission_cid"] = _cid(body)
    return parse_configured_board_live_capsule_admission(body)


def verify_configured_board_live_capsule(
    admission: ConfiguredBoardLiveCapsuleAdmission | str | Mapping[str, object],
    *,
    control_plane_pin: AgentImplementationControlPlanePin,
    control_plane_descriptor: int,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch,
    repo_root: Path,
    expected_board_namespace: str = "",
    expected_config_path: str = "",
) -> ConfiguredBoardLiveCapsuleAdmission:
    """Revalidate descriptor, source generation, and exact controls."""

    parsed = parse_configured_board_live_capsule_admission(
        admission.as_dict()
        if isinstance(admission, ConfiguredBoardLiveCapsuleAdmission)
        else admission
    )
    verified = verify_agent_implementation_sealed_control_plane(
        control_plane_pin,
        control_plane_descriptor,
    )
    if verified != f"/proc/self/fd/{control_plane_descriptor}":
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board control-plane descriptor drifted"
        )
    try:
        native_executable = verify_agent_supervisor_native_dependency_sealed_fd(
            native_dependency_launch
        )
    except (OSError, ValueError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board native dependency descriptor is invalid"
        ) from exc
    native_descriptor = native_dependency_launch.descriptor.descriptor
    if (
        native_descriptor == control_plane_descriptor
        or native_executable != f"/proc/self/fd/{native_descriptor}"
        or parsed.native_authorization_id
        != native_dependency_launch.accepted_authorization_id
        or parsed.native_dependency_id
        != native_dependency_launch.pin.dependency_id
        or parsed.native_python_executable_sha256
        != native_dependency_launch.pin.python_executable_sha256
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board native dependency binding drifted"
        )
    extension_directory = Path(
        str(os.environ.get(CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV, "") or "")
    )
    if (
        not extension_directory.is_absolute()
        or extension_directory.name != "extensions"
        or extension_directory.parent.name != ".duckdb"
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board extension projection environment is invalid"
        )
    extension_home = extension_directory.parent.parent
    try:
        verified_home = verify_configured_board_extension_home(
            parsed.quack_extension_projection,
            extension_home,
        )
    except (OSError, ValueError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board extension projection is invalid"
        ) from exc
    if verified_home / ".duckdb/extensions" != extension_directory:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board extension projection directory drifted"
        )
    if (
        parsed.control_plane_capsule_id != control_plane_pin.capsule_id
        or parsed.control_plane_archive_sha256 != control_plane_pin.archive_sha256
        or parsed.source_head != control_plane_pin.source_head
        or parsed.source_tree != control_plane_pin.source_tree
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board admission differs from the control-plane capsule"
        )
    if expected_board_namespace and parsed.board_namespace != expected_board_namespace:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board admission names a different board"
        )
    if expected_config_path and parsed.config_path != expected_config_path:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board admission names a different config"
        )
    root = Path(repo_root).resolve(strict=True)
    if _source_generation(root) != (parsed.source_head, parsed.source_tree):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source generation drifted"
        )
    expected_artifacts = _artifact_records(
        root,
        source_head=parsed.source_head,
        control_paths=tuple(str(item["path"]) for item in parsed.control_artifacts),
    )
    if expected_artifacts != parsed.control_artifacts:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected controls drifted"
        )
    return parsed


__all__ = (
    "CONFIGURED_BOARD_LIVE_CAPSULE_ADMISSION_SCHEMA",
    "CONFIGURED_BOARD_LIVE_CAPSULE_POLICY_SCHEMA",
    "ConfiguredBoardLiveCapsuleAdmission",
    "ConfiguredBoardLiveCapsuleError",
    "build_configured_board_live_capsule_admission",
    "parse_configured_board_live_capsule_admission",
    "parse_configured_board_live_capsule_policy",
    "verify_configured_board_live_capsule",
)
