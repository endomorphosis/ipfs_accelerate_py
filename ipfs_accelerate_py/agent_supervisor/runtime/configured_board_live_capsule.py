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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ...agent_implementation_route import (
    AgentSupervisorNativeDependencyLaunch,
    parse_agent_supervisor_native_dependency_pin,
    verify_agent_supervisor_native_dependency_sealed_fd,
)
from ...llm_router import (
    AgentImplementationControlPlanePin,
    verify_agent_implementation_sealed_control_plane,
)
from ..core.multiformats_identity import cid_for_dag_json, validate_cid
from ..merge.checkout_lock import checkout_repository_id
from .configured_board_extension_projection import (
    CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV,
    CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV,
    ConfiguredBoardExtensionPin,
    ConfiguredBoardExtensionSetPin,
    build_configured_board_extension_set_pin,
    parse_configured_board_extension_pin,
    parse_configured_board_extension_set_pin,
    parse_configured_board_extension_set_pin_json,
    verify_configured_board_extension_set_home,
)

CONFIGURED_BOARD_LIVE_CAPSULE_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor."
    "configured-board-live-control-capsule-policy@1"
)
CONFIGURED_BOARD_LIVE_CAPSULE_ADMISSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "configured-board-live-control-capsule-admission@1"
)
CONFIGURED_BOARD_ACCEPTED_SOURCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "configured-board-accepted-source@1"
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
        "extension_set_pin",
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
_NATIVE_DEPENDENCY_FIELDS: Final = frozenset(
    {
        "schema",
        "source_path",
        "acceptance",
        "pin",
        "sealed_memfd_required",
        "ambient_site_import_allowed",
        "ambient_loader_environment_allowed",
    }
)
_NATIVE_AUTHORIZATION_REFERENCE_FIELDS: Final = frozenset(
    {"schema", "path", "sha256", "size", "authorization_id"}
)
_NATIVE_AUTHORIZATION_FIELDS: Final = frozenset(
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
_QUACK_PROJECTION_FIELDS: Final = frozenset(
    {
        "schema",
        "source_path",
        "info_path",
        "pin",
        "load_policy",
        "network_install_allowed",
        "unsigned_extension_allowed",
    }
)
_HTTPFS_PIN_FIELDS: Final = frozenset(
    {
        "path",
        "sha256",
        "size",
        "info_path",
        "info_sha256",
        "info_size",
        "version",
        "network_install_allowed",
        "unsigned_extension_allowed",
    }
)


class ConfiguredBoardLiveCapsuleError(ValueError):
    """The live configured-board capsule is incomplete or has drifted."""


CanonicalSourceTransitionLoader = Callable[
    [str, Mapping[str, object]], Mapping[str, object]
]


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
    extension_set_pin: ConfiguredBoardExtensionSetPin
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
            "extension_set_pin": self.extension_set_pin.as_dict(),
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
        extension_set_pin=parse_configured_board_extension_set_pin(
            payload.get("extension_set_pin")
        ),
        database_authority=normalized_authority,
        max_lanes=_positive_int(payload.get("max_lanes"), "max_lanes"),
        strict_task_sharding=strict,
        control_artifacts=tuple(normalized_artifacts),
        admission_cid=_exact_content_id(
            payload.get("admission_cid"), "admission_cid"
        ),
    )
    if (
        admission.extension_set_pin.pins["quack"]
        != admission.quack_extension_projection
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board Quack projection differs from its extension set"
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


def _control_artifact(
    admission: ConfiguredBoardLiveCapsuleAdmission,
    relative: object,
) -> Mapping[str, object]:
    admitted = _relative(relative, "protected control path")
    matches = tuple(
        artifact
        for artifact in admission.control_artifacts
        if artifact.get("path") == admitted
    )
    if len(matches) != 1:
        raise ConfiguredBoardLiveCapsuleError(
            f"configured-board protected control is not admitted: {admitted}"
        )
    return matches[0]


def _protected_control_bytes(
    root: Path,
    admission: ConfiguredBoardLiveCapsuleAdmission,
    relative: object,
    *,
    maximum: int,
) -> bytes:
    artifact = _control_artifact(admission, relative)
    admitted = str(artifact["path"])
    candidate = root / admitted
    current = root
    for part in PurePosixPath(admitted).parts[:-1]:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ConfiguredBoardLiveCapsuleError(
                f"configured-board protected control parent is unavailable: {admitted}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ConfiguredBoardLiveCapsuleError(
                f"configured-board protected control parent is unsafe: {admitted}"
            )
    raw = _stable_regular_bytes(candidate, maximum=maximum)
    if (
        len(raw) != artifact["size"]
        or "sha256:" + hashlib.sha256(raw).hexdigest() != artifact["sha256"]
    ):
        raise ConfiguredBoardLiveCapsuleError(
            f"configured-board protected control differs from admission: {admitted}"
        )
    return raw


def _protected_control_json(
    root: Path,
    admission: ConfiguredBoardLiveCapsuleAdmission,
    relative: object,
    *,
    maximum: int,
) -> tuple[dict[str, object], bytes]:
    raw = _protected_control_bytes(
        root,
        admission,
        relative,
        maximum=maximum,
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected control is not canonical JSON"
        ) from exc
    if type(payload) is not dict:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected control is not a JSON object"
        )
    return dict(payload), raw


def _verify_protected_native_and_quack_authority(
    admission: ConfiguredBoardLiveCapsuleAdmission,
    *,
    native_dependency_launch: AgentSupervisorNativeDependencyLaunch,
    root: Path,
) -> None:
    """Authenticate the protected authority chain at this process birth."""

    config, config_raw = _protected_control_json(
        root,
        admission,
        admission.config_path,
        maximum=4 * 1024 * 1024,
    )
    dependency_seal_path = _relative(
        config.get("dependency_seal_path"),
        "dependency_seal_path",
    )
    raw_program = config.get("database_program")
    raw_generation = (
        raw_program.get("store_generation")
        if isinstance(raw_program, dict)
        else None
    )
    try:
        expected_database_authority = {
            field: _text(
                raw_program.get(field) if isinstance(raw_program, dict) else None,
                f"database_program.{field}",
            )
            for field in _DATABASE_FIELDS - {"store_generation"}
        }
        generation_text = _text(
            str(raw_generation) if raw_generation is not None else None,
            "database_program.store_generation",
        )
        if not generation_text.isascii() or not generation_text.isdecimal():
            raise ConfiguredBoardLiveCapsuleError(
                "database_program.store_generation is invalid"
            )
        expected_database_authority["store_generation"] = _positive_int(
            int(generation_text),
            "database_program.store_generation",
        )
        expected_task_prefix = _text(config.get("task_prefix"), "task_prefix")
        expected_max_lanes = _positive_int(config.get("max_lanes"), "max_lanes")
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected scheduler authority is invalid"
        ) from exc
    expected_strict_sharding = config.get("strict_task_sharding")
    if type(expected_strict_sharding) is not bool:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected scheduler sharding is invalid"
        )
    expected_configuration_root = _cid(
        {"bytes_sha256": hashlib.sha256(config_raw).hexdigest()}
    )
    if (
        config.get("board_namespace") != admission.board_namespace
        or config.get("plan_revision") != admission.plan_revision
        or admission.task_prefix != expected_task_prefix
        or admission.configuration_root != expected_configuration_root
        or dict(admission.database_authority)
        != expected_database_authority
        or admission.max_lanes != expected_max_lanes
        or admission.strict_task_sharding is not expected_strict_sharding
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board scheduler identity differs from admission"
        )
    seal, _seal_raw = _protected_control_json(
        root,
        admission,
        dependency_seal_path,
        maximum=4 * 1024 * 1024,
    )
    if (
        seal.get("schema") != "semantic-addressed-world-model/dependency-seal@1"
        or seal.get("board_namespace") != admission.board_namespace
        or seal.get("plan_revision") != admission.plan_revision
        or seal.get("status") != "sealed"
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected dependency seal is invalid"
        )

    native = seal.get("configured_board_native_dependency")
    if type(native) is not dict or set(native) != _NATIVE_DEPENDENCY_FIELDS:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected native dependency is noncanonical"
        )
    if (
        native.get("schema")
        != "semantic-addressed-world-model/configured-board-native-dependency@1"
        or native.get("sealed_memfd_required") is not True
        or native.get("ambient_site_import_allowed") is not False
        or native.get("ambient_loader_environment_allowed") is not False
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected native dependency policy is invalid"
        )
    try:
        sealed_pin = parse_agent_supervisor_native_dependency_pin(native.get("pin"))
        launch_pin = parse_agent_supervisor_native_dependency_pin(
            native_dependency_launch.pin.as_dict()
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected native dependency pin is invalid"
        ) from exc
    if sealed_pin != launch_pin:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board native launch pin differs from protected authority"
        )

    reference = native.get("acceptance")
    if (
        type(reference) is not dict
        or set(reference) != _NATIVE_AUTHORIZATION_REFERENCE_FIELDS
        or reference.get("schema")
        != (
            "semantic-addressed-world-model/"
            "native-dependency-authorization-reference@1"
        )
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board native authorization reference is invalid"
        )
    authorization_path = _relative(
        reference.get("path"),
        "native authorization path",
    )
    authorization, authorization_raw = _protected_control_json(
        root,
        admission,
        authorization_path,
        maximum=65_536,
    )
    if (
        type(reference.get("size")) is not int
        or reference.get("size") != len(authorization_raw)
        or reference.get("sha256")
        != "sha256:" + hashlib.sha256(authorization_raw).hexdigest()
        or type(authorization) is not dict
        or set(authorization) != _NATIVE_AUTHORIZATION_FIELDS
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board native authorization artifact differs"
        )
    unsigned_authorization = dict(authorization)
    authorization_id = str(
        unsigned_authorization.pop("authorization_id", "") or ""
    )
    expected_authorization_id = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned_authorization,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        authorization_id != expected_authorization_id
        or authorization_id != reference.get("authorization_id")
        or authorization_id != native_dependency_launch.accepted_authorization_id
        or authorization_id != admission.native_authorization_id
        or authorization.get("schema")
        != (
            "semantic-addressed-world-model/"
            "native-dependency-launch-authorization@1"
        )
        or authorization.get("board_namespace") != admission.board_namespace
        or authorization.get("plan_revision") != admission.plan_revision
        or authorization.get("status") != "accepted"
        or authorization.get("scope")
        != "configured-board-live-control-plane"
        or authorization.get("dependency_id") != sealed_pin.dependency_id
        or authorization.get("payload_sha256") != sealed_pin.payload_sha256
        or authorization.get("python_executable_sha256")
        != sealed_pin.python_executable_sha256
        or authorization.get("authority_basis")
        != (
            "operator-owned protected control inside the accepted immutable "
            "source capsule"
        )
        or authorization.get("inspection_is_authority") is not False
        or authorization.get("authorization_may_claim_task_completion") is not False
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board native authorization was not admitted"
        )

    projection = seal.get("configured_board_quack_projection")
    if type(projection) is not dict or set(projection) != _QUACK_PROJECTION_FIELDS:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected Quack projection is noncanonical"
        )
    try:
        projection_pin = parse_configured_board_extension_pin(projection.get("pin"))
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected Quack projection pin is invalid"
        ) from exc
    if (
        projection.get("schema")
        != "semantic-addressed-world-model/configured-board-quack-projection@1"
        or projection.get("load_policy") != "local_load_only"
        or projection.get("network_install_allowed") is not False
        or projection.get("unsigned_extension_allowed") is not False
        or projection_pin != admission.quack_extension_projection
        or projection_pin.engine_version != sealed_pin.engine_version
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board Quack projection differs from protected authority"
        )

    httpfs = seal.get("httpfs_extension_pin")
    if type(httpfs) is not dict or set(httpfs) != _HTTPFS_PIN_FIELDS:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected HTTPFS pin is noncanonical"
        )
    admitted_pins = admission.extension_set_pin.pins
    httpfs_pin = admitted_pins["httpfs"]
    if (
        admitted_pins["quack"] != projection_pin
        or httpfs_pin.engine_version != projection_pin.engine_version
        or httpfs_pin.platform != projection_pin.platform
        or httpfs_pin.payload_sha256
        != f"sha256:{str(httpfs.get('sha256') or '')}"
        or httpfs_pin.payload_size != httpfs.get("size")
        or httpfs_pin.info_sha256
        != f"sha256:{str(httpfs.get('info_sha256') or '')}"
        or httpfs_pin.info_size != httpfs.get("info_size")
        or httpfs.get("network_install_allowed") is not False
        or httpfs.get("unsigned_extension_allowed") is not False
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board extension set differs from protected authority"
        )
    owner = config.get("quack_owner")
    if type(owner) is not dict:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected Quack owner is noncanonical"
        )
    configured_pins = {
        "httpfs": owner.get("pinned_httpfs_extension"),
        "quack": owner.get("pinned_extension"),
    }
    if any(type(value) is not dict for value in configured_pins.values()):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected extension owner pins are noncanonical"
        )
    protected_pins: dict[str, Mapping[str, object]] = {
        "httpfs": httpfs,
        "quack": {
            "path": projection.get("source_path"),
            "info_path": projection.get("info_path"),
            "sha256": projection_pin.payload_sha256.removeprefix("sha256:"),
            "size": projection_pin.payload_size,
            "info_sha256": projection_pin.info_sha256.removeprefix("sha256:"),
            "info_size": projection_pin.info_size,
            "network_install_allowed": False,
            "unsigned_extension_allowed": False,
        },
    }
    for name, protected in protected_pins.items():
        configured = configured_pins[name]
        assert isinstance(configured, dict)
        for field in (
            "path",
            "info_path",
            "sha256",
            "size",
            "info_sha256",
            "info_size",
            "network_install_allowed",
            "unsigned_extension_allowed",
        ):
            if configured.get(field) != protected.get(field):
                raise ConfiguredBoardLiveCapsuleError(
                    f"configured-board protected {name} owner pin drifted"
                )
    versions = {
        name: str(configured.get("version") or "")
        for name, configured in configured_pins.items()
        if isinstance(configured, dict)
    }
    if versions.get("httpfs") != httpfs.get("version"):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected HTTPFS version drifted"
        )
    try:
        protected_set_pin = build_configured_board_extension_set_pin(
            admitted_pins,
            versions=versions,
        )
    except ValueError as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected extension set is invalid"
        ) from exc
    if protected_set_pin != admission.extension_set_pin:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board extension set admission drifted"
        )


def _git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    environment = {
        "PATH": "/usr/bin:/bin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", *arguments],
            cwd=root,
            env=environment,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board Git proof is unavailable"
        ) from exc
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


def _pinned_scheduler_payload(
    root: Path,
    admission: ConfiguredBoardLiveCapsuleAdmission,
) -> dict[str, object]:
    """Return the exact scheduler payload admitted by the source capsule."""

    raw = _git(root, "show", f"{admission.source_head}:{admission.config_path}")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board pinned scheduler is not canonical JSON"
        ) from exc
    if type(payload) is not dict:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board pinned scheduler is not an object"
        )
    if (
        payload.get("board_namespace") != admission.board_namespace
        or payload.get("plan_revision") != admission.plan_revision
        or payload.get("task_prefix") != admission.task_prefix
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board pinned scheduler identity drifted"
        )
    return payload


def _default_canonical_source_transition(
    merge_commit: str,
    scheduler: Mapping[str, object],
) -> Mapping[str, object]:
    """Resolve one source transition from canonical Quack completion state."""

    program = scheduler.get("database_program")
    if type(program) is not dict:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source has no database program"
        )
    endpoint = str(program.get("quack_endpoint") or "").strip()
    handle = str(program.get("endpoint_secret_handle") or "").strip()
    generation = program.get("store_generation")
    if (
        not endpoint.startswith("quack:127.0.0.1:")
        or not handle.startswith("env://")
        or isinstance(generation, bool)
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source has invalid Quack authority"
        )
    try:
        expected_generation = int(generation)
    except (TypeError, ValueError) as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source has invalid store generation"
        ) from exc
    token_name = handle.removeprefix("env://").strip()
    token = str(
        os.environ.get(token_name, "")
        or os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "")
        or ""
    ).strip()
    if not token:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source cannot resolve Quack authority"
        )
    try:
        from ..task_sources.duckdb_state import open_quack_transport_connection

        connection = open_quack_transport_connection(endpoint, token=token)
        try:
            generations = connection.execute(
                "SELECT generation, database_uuid FROM store_generations "
                "ORDER BY generation DESC LIMIT 1"
            ).fetchall()
            tasks = connection.execute(
                "SELECT task_cid, task_alias, status, revision, body_json "
                "FROM tasks WHERE status IN ('completed', 'complete', 'done') "
                "ORDER BY ordinal, task_cid LIMIT 4096"
            ).fetchall()
        finally:
            connection.close()
    except Exception as exc:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source Quack lookup failed"
        ) from exc
    if (
        len(generations) != 1
        or int(generations[0][0]) != expected_generation
        or not str(generations[0][1] or "").strip()
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source Quack identity drifted"
        )
    matches: list[dict[str, object]] = []
    for row in tasks:
        try:
            body = json.loads(
                str(row[4]), object_pairs_hook=_reject_duplicate_json_keys
            )
        except (TypeError, json.JSONDecodeError) as exc:
            raise ConfiguredBoardLiveCapsuleError(
                "configured-board accepted task body is invalid"
            ) from exc
        if type(body) is not dict:
            raise ConfiguredBoardLiveCapsuleError(
                "configured-board accepted task body is not an object"
            )
        completion = body.get("completion_receipt")
        if not isinstance(completion, Mapping):
            continue
        validation = completion.get("validation")
        transition = (
            validation.get("accepted_source_transition")
            if isinstance(validation, Mapping)
            else None
        )
        if isinstance(transition, Mapping) and transition.get("merge_commit") == merge_commit:
            matches.append(
                {
                    "task_cid": str(row[0]),
                    "task_alias": str(row[1]),
                    "status": str(row[2]),
                    "revision": int(row[3]),
                    "completion_receipt": dict(completion),
                    "transition": dict(transition),
                    "store_generation": expected_generation,
                    "database_uuid": str(generations[0][1]),
                }
            )
    if len(matches) != 1:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board accepted source has no unique canonical transition"
        )
    return matches[0]


def _verify_canonical_source_transition(
    *,
    repo_root: Path,
    board_namespace: str,
    admission_cid: str,
    target_repository_id: str,
    prior_head: str,
    merge_head: str,
    implementation_head: str,
    target_branch: str,
    authority: Mapping[str, object],
) -> Mapping[str, str]:
    completion = authority.get("completion_receipt")
    transition = authority.get("transition")
    validation = completion.get("validation") if isinstance(completion, Mapping) else None
    if not isinstance(transition, Mapping):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board canonical source transition is absent"
        )
    normalized = dict(transition)
    transition_cid = str(normalized.pop("transition_cid", "") or "")
    proof = transition.get("integration_commit_proof")
    invariant = transition.get("declared_output_invariant")
    alias = str(authority.get("task_alias") or "")
    expected_fields = {
        "schema",
        "board_namespace",
        "configured_board_admission_cid",
        "task_alias",
        "database_task_cid",
        "attempt_id",
        "attempt_number",
        "portal_attempt_number",
        "claim_id",
        "fencing_token",
        "database_attempt_binding",
        "canonical_task_cid",
        "canonical_task_key",
        "request_id",
        "merge_request_digest",
        "merge_request_dedupe_key",
        "target_repository_id",
        "implementation_commit",
        "implementation_tree",
        "merge_commit",
        "merge_tree",
        "target_branch",
        "changed_path_diff_sha256",
        "integration_commit_proof",
        "declared_output_invariant",
        "portal_event_log_sha256",
        "authority",
        "task_completion_authority",
        "worker_self_approval",
    }
    transition_schema = str(transition.get("schema") or "")
    source_binding_valid = False
    proof_topology_valid = True
    legacy_direct_schema = (
        "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@1"
    )
    legacy_reconciled_schema = (
        "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@2"
    )
    target_advanced_schema = (
        "ipfs_accelerate_py/agent-supervisor/accepted-source-transition@3"
    )
    reconciliation_fields = {
        "source_event_mode",
        "queued_implementation_event_id",
        "reconciliation_event_id",
        "merge_queue_terminal_status",
        "merge_queue_attempt",
        "merge_queue_cancellation_reason",
        "completion_persistence",
    }
    reconciled_transition = bool(
        transition_schema == legacy_reconciled_schema
        or (
            transition_schema == target_advanced_schema
            and transition.get("source_event_mode") is not None
        )
    )
    reconciliation_binding_valid = True
    if reconciled_transition:
        expected_fields.update(reconciliation_fields)
        completion_persistence = transition.get("completion_persistence")
        queue_attempt = transition.get("merge_queue_attempt")
        portal_attempt = transition.get("portal_attempt_number")
        reconciliation_binding_valid = bool(
            transition.get("source_event_mode")
            == "queued_merge_reconciliation"
            and _SHA256.fullmatch(
                str(transition.get("queued_implementation_event_id") or "")
            )
            is not None
            and _SHA256.fullmatch(
                str(transition.get("reconciliation_event_id") or "")
            )
            is not None
            and transition.get("merge_queue_terminal_status") == "cancelled"
            and isinstance(queue_attempt, int)
            and not isinstance(queue_attempt, bool)
            and isinstance(portal_attempt, int)
            and not isinstance(portal_attempt, bool)
            and queue_attempt >= portal_attempt >= 1
            and transition.get("merge_queue_cancellation_reason")
            == "stale_quarantined_merge"
            and isinstance(completion_persistence, Mapping)
            and completion_persistence.get("passed") is True
            and completion_persistence.get("reason")
            == "completion_persisted"
            and completion_persistence.get("durable_update") is True
            and completion_persistence.get("status_persisted") is True
        )
    if transition_schema in {legacy_direct_schema, legacy_reconciled_schema}:
        expected_fields.add("baseline_ref")
        source_binding_valid = transition.get("baseline_ref") == prior_head
        proof_topology_valid = reconciliation_binding_valid
    elif transition_schema == target_advanced_schema:
        expected_fields.update(
            {"candidate_baseline_ref", "integration_base_commit"}
        )
        candidate_baseline = str(
            transition.get("candidate_baseline_ref") or ""
        )
        integration_base = str(
            transition.get("integration_base_commit") or ""
        )
        if (
            re.fullmatch(r"[0-9a-f]{40}", candidate_baseline)
            and integration_base == prior_head
        ):
            try:
                _git(
                    repo_root,
                    "merge-base",
                    "--is-ancestor",
                    candidate_baseline,
                    integration_base,
                )
                _git(
                    repo_root,
                    "merge-base",
                    "--is-ancestor",
                    candidate_baseline,
                    implementation_head,
                )
            except ConfiguredBoardLiveCapsuleError:
                source_binding_valid = False
            else:
                source_binding_valid = True
        proof_topology_valid = bool(
            isinstance(proof, Mapping)
            and proof.get("candidate_baseline_ref") == candidate_baseline
            and proof.get("integration_base_commit") == integration_base
            and proof.get("exact_two_parent_merge") is True
            and reconciliation_binding_valid
        )
    implementation_tree = _git(
        repo_root, "rev-parse", f"{implementation_head}^{{tree}}"
    ).decode("ascii").strip()
    merge_tree = _git(
        repo_root, "rev-parse", f"{merge_head}^{{tree}}"
    ).decode("ascii").strip()
    changed_path_diff_sha256 = "sha256:" + hashlib.sha256(
        _git(
            repo_root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            prior_head,
            merge_head,
        )
    ).hexdigest()
    database_binding = transition.get("database_attempt_binding")
    normalized_database_binding = (
        dict(database_binding) if isinstance(database_binding, Mapping) else {}
    )
    database_binding_id = str(
        normalized_database_binding.pop("binding_id", "") or ""
    )
    expected_database_binding_fields = {
        "schema",
        "interface",
        "attempt_id",
        "claim_id",
        "task_cid",
        "task_alias",
        "goal_cid",
        "plan_cid",
        "task_revision",
        "fencing_token",
        "fence_epoch",
        "lease_id",
        "task_body_digest",
        "projection_seed_digest",
        "projection_immutable_digest",
        "authoritative_task_store",
        "projection_authority",
    }
    if (
        set(normalized) != expected_fields
        or str(authority.get("status") or "").lower()
        not in {"completed", "complete", "done"}
        or not str(authority.get("task_cid") or "").startswith("sha256:")
        or not isinstance(completion, Mapping)
        or completion.get("operation") != "database_complete"
        or not isinstance(validation, Mapping)
        or validation.get("outcome") != "passed"
        or validation.get("task_cid") != authority.get("task_cid")
        or validation.get("attempt_id") != transition.get("attempt_id")
        or validation.get("accepted_source_transition") != transition
        or not source_binding_valid
        or not proof_topology_valid
        or transition.get("database_task_cid") != authority.get("task_cid")
        or transition.get("task_alias") != alias
        or transition.get("board_namespace") != board_namespace
        or transition.get("configured_board_admission_cid") != admission_cid
        or transition.get("target_repository_id") != target_repository_id
        or transition.get("implementation_commit") != implementation_head
        or transition.get("implementation_tree") != implementation_tree
        or transition.get("merge_commit") != merge_head
        or transition.get("merge_tree") != merge_tree
        or transition.get("target_branch") != target_branch
        or transition.get("changed_path_diff_sha256")
        != changed_path_diff_sha256
        or not str(transition.get("request_id") or "")
        or not str(transition.get("canonical_task_key") or "").startswith(
            "task/v1/"
        )
        or not str(transition.get("canonical_task_cid") or "").startswith(
            "baguq"
        )
        or not str(transition.get("attempt_id") or "")
        or not str(transition.get("claim_id") or "")
        or isinstance(transition.get("attempt_number"), bool)
        or not isinstance(transition.get("attempt_number"), int)
        or int(transition.get("attempt_number") or 0) < 1
        or isinstance(transition.get("portal_attempt_number"), bool)
        or not isinstance(transition.get("portal_attempt_number"), int)
        or int(transition.get("portal_attempt_number") or 0) < 1
        or isinstance(transition.get("fencing_token"), bool)
        or not isinstance(transition.get("fencing_token"), int)
        or int(transition.get("fencing_token") or 0) < 1
        or set(normalized_database_binding) != expected_database_binding_fields
        or database_binding.get("schema")
        != (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-attempt-binding@1"
        )
        or database_binding.get("interface")
        != "DatabasePortalExecutionBridge@1"
        or database_binding.get("attempt_id") != transition.get("attempt_id")
        or database_binding.get("claim_id") != transition.get("claim_id")
        or database_binding.get("task_cid")
        != transition.get("database_task_cid")
        or database_binding.get("task_alias") != transition.get("task_alias")
        or database_binding.get("fencing_token")
        != transition.get("fencing_token")
        or database_binding.get("authoritative_task_store") != "duckdb"
        or database_binding.get("projection_authority") is not False
        or database_binding_id
        != "sha256:"
        + hashlib.sha256(_canonical_json(normalized_database_binding)).hexdigest()
        or _SHA256.fullmatch(
            str(transition.get("merge_request_digest") or "")
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(transition.get("merge_request_dedupe_key") or ""),
        )
        is None
        or transition.get("authority")
        != "database_completion_cas_after_portal_and_git_verification"
        or transition.get("task_completion_authority") is not False
        or transition.get("worker_self_approval") is not False
        or transition_cid
        != "sha256:" + hashlib.sha256(_canonical_json(normalized)).hexdigest()
        or not isinstance(proof, Mapping)
        or proof.get("passed") is not True
        or proof.get("implementation_commit") != implementation_head
        or proof.get("integration_commit") != merge_head
        or proof.get("integration_ref") != merge_head
        or proof.get("target_branch") != target_branch
        or not isinstance(invariant, Mapping)
        or invariant.get("passed") is not True
        or invariant.get("repository_ref") != merge_head
    ):
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board canonical source transition is inconsistent"
        )
    return {
        "task_alias": alias,
        "database_task_cid": str(authority.get("task_cid") or ""),
        "transition_cid": transition_cid,
        "request_id": str(transition.get("request_id") or ""),
        "implementation_commit": implementation_head,
    }


def verify_configured_board_accepted_source(
    admission: ConfiguredBoardLiveCapsuleAdmission | str | Mapping[str, object],
    *,
    repo_root: Path,
    transition_loader: CanonicalSourceTransitionLoader | None = None,
    admitted_live_capsule_restart: bool = False,
) -> Mapping[str, object]:
    """Admit exact source, or a receipt-backed chain of supervisor merges.

    This is deliberately narrower than descendant admission.  Every first-parent
    successor must be a two-parent merge, must retain every protected control,
    and must bind the implementation parent to a source-transition packet that
    was admitted inside the canonical Quack task-completion transaction.

    Isolated child restarts of an already-running wave may set
    ``admitted_live_capsule_restart``.  That path still requires the sealed
    capsule source and protected controls, but allows a Git descendant HEAD
    that is not a two-parent supervisor merge.  The child continues to execute
    the admitted capsule, not the drifted worktree.
    """

    parsed = parse_configured_board_live_capsule_admission(
        admission.as_dict()
        if isinstance(admission, ConfiguredBoardLiveCapsuleAdmission)
        else admission
    )
    root = Path(repo_root).resolve(strict=True)
    current_head, current_tree = _source_generation(root)
    expected_artifacts = _artifact_records(
        root,
        source_head=parsed.source_head,
        control_paths=tuple(str(item["path"]) for item in parsed.control_artifacts),
    )
    if expected_artifacts != parsed.control_artifacts:
        raise ConfiguredBoardLiveCapsuleError(
            "configured-board protected controls drifted"
        )
    kind = "exact"
    merge_commits: list[str] = []
    implementation_commits: list[str] = []
    task_aliases: list[str] = []
    database_task_cids: list[str] = []
    transition_cids: list[str] = []
    request_ids: list[str] = []
    database_uuid = ""
    store_generation = 0
    target_repository_id = checkout_repository_id(root)
    if (current_head, current_tree) != (parsed.source_head, parsed.source_tree):
        if admitted_live_capsule_restart:
            try:
                _git(
                    root,
                    "merge-base",
                    "--is-ancestor",
                    parsed.source_head,
                    current_head,
                )
            except ConfiguredBoardLiveCapsuleError as exc:
                raise ConfiguredBoardLiveCapsuleError(
                    "configured-board admitted restart is not a source descendant"
                ) from exc
            kind = "admitted_live_capsule_restart"
        else:
            kind = "accepted_supervisor_merge_successor"
        if kind == "accepted_supervisor_merge_successor":
            try:
                chain = tuple(
                    line
                    for line in _git(
                        root,
                        "rev-list",
                        "--first-parent",
                        "--reverse",
                        f"{parsed.source_head}..{current_head}",
                    ).decode("ascii").splitlines()
                    if line
                )
            except (UnicodeError, ConfiguredBoardLiveCapsuleError) as exc:
                raise ConfiguredBoardLiveCapsuleError(
                    "configured-board accepted source is not a Git descendant"
                ) from exc
            if not chain:
                raise ConfiguredBoardLiveCapsuleError(
                    "configured-board accepted source generation drifted"
                )
            if len(chain) > 4_096:
                raise ConfiguredBoardLiveCapsuleError(
                    "configured-board accepted source chain is unbounded"
                )
            scheduler = _pinned_scheduler_payload(root, parsed)
            target_branch = _text(
                scheduler.get("merge_target_branch"), "merge target branch"
            )
            current_branch = _git(root, "symbolic-ref", "--short", "HEAD").decode(
                "utf-8"
            ).strip()
            if current_branch != target_branch:
                raise ConfiguredBoardLiveCapsuleError(
                    "configured-board accepted source target branch drifted"
                )
            loader = transition_loader or _default_canonical_source_transition
            prior = parsed.source_head
            for merge_head in chain:
                parents = _git(root, "rev-list", "--parents", "-n", "1", merge_head)
                parent_fields = parents.decode("ascii").strip().split()
                if (
                    len(parent_fields) != 3
                    or parent_fields[0] != merge_head
                    or parent_fields[1] != prior
                ):
                    raise ConfiguredBoardLiveCapsuleError(
                        "configured-board accepted source contains a non-supervisor merge"
                    )
                implementation_head = parent_fields[2]
                authority = loader(merge_head, scheduler)
                verified = _verify_canonical_source_transition(
                    repo_root=root,
                    board_namespace=parsed.board_namespace,
                    admission_cid=parsed.admission_cid,
                    target_repository_id=target_repository_id,
                    prior_head=prior,
                    merge_head=merge_head,
                    implementation_head=implementation_head,
                    target_branch=target_branch,
                    authority=authority,
                )
                observed_uuid = str(authority.get("database_uuid") or "")
                observed_generation = authority.get("store_generation")
                if (
                    not observed_uuid
                    or isinstance(observed_generation, bool)
                    or not isinstance(observed_generation, int)
                    or observed_generation < 1
                    or (database_uuid and observed_uuid != database_uuid)
                    or (store_generation and observed_generation != store_generation)
                    or verified["task_alias"] in task_aliases
                    or verified["database_task_cid"] in database_task_cids
                    or verified["transition_cid"] in transition_cids
                    or verified["request_id"] in request_ids
                ):
                    raise ConfiguredBoardLiveCapsuleError(
                        "configured-board accepted source authority is ambiguous"
                    )
                database_uuid = observed_uuid
                store_generation = observed_generation
                task_aliases.append(verified["task_alias"])
                database_task_cids.append(verified["database_task_cid"])
                transition_cids.append(verified["transition_cid"])
                request_ids.append(verified["request_id"])
                implementation_commits.append(verified["implementation_commit"])
                merge_commits.append(merge_head)
                prior = merge_head
            if prior != current_head:
                raise ConfiguredBoardLiveCapsuleError(
                    "configured-board accepted source chain is incomplete"
                )
            if _source_generation(root) != (current_head, current_tree):
                raise ConfiguredBoardLiveCapsuleError(
                    "configured-board accepted source changed during verification"
                )
    body: dict[str, object] = {
        "schema": CONFIGURED_BOARD_ACCEPTED_SOURCE_SCHEMA,
        "kind": kind,
        "board_namespace": parsed.board_namespace,
        "admission_cid": parsed.admission_cid,
        "source_head": parsed.source_head,
        "source_tree": parsed.source_tree,
        "current_head": current_head,
        "current_tree": current_tree,
        "merge_commits": merge_commits,
        "implementation_commits": implementation_commits,
        "task_aliases": task_aliases,
        "database_task_cids": database_task_cids,
        "transition_cids": transition_cids,
        "request_ids": request_ids,
        "target_repository_id": target_repository_id,
        "database_uuid": database_uuid,
        "store_generation": store_generation,
        "protected_control_count": len(parsed.control_artifacts),
        "authority": (
            "exact_capsule_source"
            if kind == "exact"
            else "exact_capsule_source_plus_descendant_head"
            if kind == "admitted_live_capsule_restart"
            else "git_merge_plus_quack_admitted_source_transition"
        ),
        "task_completion_authority": False,
    }
    body["receipt_cid"] = _cid(body)
    return body


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
    extension_set_pin: ConfiguredBoardExtensionSetPin,
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
        "extension_set_pin": extension_set_pin.as_dict(),
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
    admitted_live_capsule_restart: bool = False,
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
    extension_set_pin_json = str(
        os.environ.get(CONFIGURED_BOARD_EXTENSION_SET_PIN_ENV, "") or ""
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
        environment_set_pin = parse_configured_board_extension_set_pin_json(
            extension_set_pin_json
        )
        if environment_set_pin != parsed.extension_set_pin:
            raise ConfiguredBoardLiveCapsuleError(
                "configured-board extension set environment drifted"
            )
        verified_home = verify_configured_board_extension_set_home(
            parsed.extension_set_pin.pins,
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
    verify_configured_board_accepted_source(
        parsed,
        repo_root=root,
        admitted_live_capsule_restart=admitted_live_capsule_restart,
    )
    _verify_protected_native_and_quack_authority(
        parsed,
        native_dependency_launch=native_dependency_launch,
        root=root,
    )
    return parsed


__all__ = (
    "CONFIGURED_BOARD_LIVE_CAPSULE_ADMISSION_SCHEMA",
    "CONFIGURED_BOARD_ACCEPTED_SOURCE_SCHEMA",
    "CONFIGURED_BOARD_LIVE_CAPSULE_POLICY_SCHEMA",
    "ConfiguredBoardLiveCapsuleAdmission",
    "ConfiguredBoardLiveCapsuleError",
    "build_configured_board_live_capsule_admission",
    "parse_configured_board_live_capsule_admission",
    "parse_configured_board_live_capsule_policy",
    "verify_configured_board_accepted_source",
    "verify_configured_board_live_capsule",
)
