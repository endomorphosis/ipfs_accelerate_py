"""Source-addressed launch projection for qualified EAAEF worker containers.

The bootstrap admission contract freezes an independently reviewed
``external-agent-worker-container-profile@1`` CID.  That profile deliberately
does not name a host engine endpoint.  Runtime code must not fill that gap from
``DOCKER_HOST``, a caller path, or a hard-coded rootful socket.

This module loads one small, separately signed launch envelope from the
reviewer-owned invocation profile directory.  Its pathname is derived solely
from the admitted profile CID.  The envelope embeds the complete admitted
profile, binds the exact source revision and engine endpoint, and has its own
content identity.  A verified object is therefore useful only as a cache of
expected values: every effect boundary calls :func:`reverify_worker_container_execution_profile`
and stable-reads the source artifact again.

No function in this module starts a process or contacts a container engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.entrypoints.local_profile import (
    LocalProfileTampered,
    verify_did_key_signature,
)
from ipfs_accelerate_py.agent_supervisor.validation.external_agent_fabric_bootstrap import (
    EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA,
    EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
    validate_eaaef_worker_container_profile_artifact,
)

from .worker_network import _stable_private_json
from .worker_network_dispatch import parse_worker_network_launch_authority

EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "source-addressed-container-execution-profile-launch@1"
)
EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "source-addressed-container-execution-profile-launch@2"
)
EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE: Final = (
    "independent_container_execution_reviewer"
)
EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2: Final = (
    "independent_grok_container_execution_reviewer"
)
EAAEF_GROK_PROMPT_MOUNT_TARGET: Final = "/run/eaaef/grok/prompt.txt"
EAAEF_GROK_POLICY_MOUNT_TARGET: Final = "/opt/codex-home/sandbox.toml"
EAAEF_GROK_PROVIDER_HOME_MOUNT_TARGET: Final = "/opt/codex-home"

_GROK_MOUNT_KINDS = frozenset(
    {"grok_prompt", "grok_policy", "grok_provider_home"}
)
_GROK_PROVIDER_HOME_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/grok-provider-home-source@1"
)
_MAX_MOUNT_SOURCE_BYTES = 2 * 1024 * 1024

_CID = re.compile(r"sha256:[0-9a-f]{64}")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_MAX_LIFETIME_MS = 24 * 60 * 60 * 1000
_FIELDS = frozenset(
    {
        "schema",
        "source_head",
        "source_tree",
        "accepted_control_plane_capsule_id",
        "qualified_worker_image_digest",
        "qualified_worker_container_profile_cid",
        "engine_endpoint",
        "profile",
        "issued_at_ms",
        "expires_at_ms",
        "signer_identity_did",
        "signer_role",
        "signature",
        "artifact_cid",
    }
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def worker_container_execution_profile_signing_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """Return canonical bytes signed by the independent execution reviewer."""

    body = dict(value)
    body.pop("artifact_cid", None)
    body.pop("signature", None)
    return _canonical_bytes(body)


def worker_container_execution_profile_relative_path(profile_cid: str) -> Path:
    """Return the only invocation-profile-relative path for one admitted CID."""

    normalized = str(profile_cid).strip()
    if _CID.fullmatch(normalized) is None:
        raise ValueError("qualified worker container profile CID is invalid")
    return Path("container-execution-profiles") / (
        normalized.removeprefix("sha256:") + ".json"
    )


def _engine_endpoint(profile: Mapping[str, Any], value: object) -> str:
    endpoint = str(value or "")
    mode = str(profile.get("execution_mode") or "")
    if mode == "rootless_engine":
        expected = f"unix:///run/user/{os.geteuid()}/docker.sock"
        if (
            endpoint != expected
            or profile.get("rootless") is not True
            or profile.get("rootless_supported") is not True
            or profile.get("rootful_fallback_admitted") is not False
        ):
            raise ValueError(
                "rootless worker profile cannot use a rootful or caller-selected endpoint"
            )
        return endpoint
    if mode == "rootful_daemon_nonroot_worker":
        if (
            endpoint != "unix:///var/run/docker.sock"
            or profile.get("rootless") is not False
            or profile.get("rootful_fallback_admitted") is not True
        ):
            raise ValueError("rootful worker endpoint is not independently admitted")
        return endpoint
    raise ValueError("worker container execution mode is invalid")


@dataclass(frozen=True, slots=True)
class WorkerContainerExecutionMount:
    """One signed mount class; source paths remain separately effect-bound."""

    source_identity: str
    target: str
    read_only: bool
    kind: str


@dataclass(frozen=True, slots=True)
class WorkerContainerExecutionProfile:
    """Immutable projection of one freshly verified launch artifact."""

    schema: str
    profile_schema: str
    artifact_cid: str
    source_file_cid: str
    profile_cid: str
    image_digest: str
    source_head: str
    source_tree: str
    accepted_control_plane_capsule_id: str
    runtime: str
    execution_mode: str
    engine_endpoint: str
    daemon_identity_cid: str
    daemon_policy_cid: str
    nonroot_user: str
    read_only_base: bool
    network_mode: str
    cap_drop: tuple[str, ...]
    no_new_privileges: bool
    pids_limit: int
    cpu_limit: float
    memory_limit_bytes: int
    disk_limit_bytes: int
    maximum_parallel_workers: int
    maximum_parallel_containers: int
    gpu_mode: str
    gpu_device_ids: tuple[str, ...]
    gpu_memory_limit_bytes: int
    privileged: bool
    host_pid: bool
    host_ipc: bool
    devices: tuple[str, ...]
    docker_socket_mounted: bool
    inherit_host_environment: bool
    environment: tuple[tuple[str, str], ...]
    mounts: tuple[WorkerContainerExecutionMount, ...]
    signer_identity_did: str
    issued_at_ms: int
    expires_at_ms: int
    source_path: Path
    trusted_root: Path

    def container_environment(self) -> dict[str, str]:
        return dict(self.environment)

    def mount_for_kind(self, kind: str) -> WorkerContainerExecutionMount | None:
        matches = tuple(item for item in self.mounts if item.kind == kind)
        if len(matches) > 1:
            raise ValueError(f"worker profile has ambiguous {kind} mounts")
        return matches[0] if matches else None

    def mounts_for_provider(
        self,
        provider: str,
    ) -> tuple[WorkerContainerExecutionMount, ...]:
        """Project only mounts admitted for one provider boundary."""

        normalized = str(provider).strip().lower()
        if normalized == "grok":
            if (
                self.schema
                != EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2
                or self.profile_schema
                != EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2
            ):
                raise ValueError(
                    "qualified Grok requires the signed execution profile @2"
                )
            return self.mounts
        if normalized == "codex":
            return tuple(
                mount
                for mount in self.mounts
                if mount.kind in {"worktree", "provider_auth", "secret"}
            )
        raise ValueError("worker execution profile provider is invalid")


def _stable_private_file_bytes(path: Path) -> bytes:
    """Read one private regular source without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("worker execution mount source is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o077
            or before.st_size < 0
            or before.st_size > _MAX_MOUNT_SOURCE_BYTES
        ):
            raise ValueError("worker execution mount source is not private")
        chunks: list[bytes] = []
        remaining = _MAX_MOUNT_SOURCE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > _MAX_MOUNT_SOURCE_BYTES
            or (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
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
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise ValueError("worker execution mount source changed while read")
        return payload
    finally:
        os.close(descriptor)


def worker_container_execution_file_source_identity(path: Path) -> str:
    """Return the source address of one private immutable mount file."""

    return "sha256:" + hashlib.sha256(_stable_private_file_bytes(path)).hexdigest()


def worker_container_execution_grok_provider_home_source_identity(
    provider_home: Path,
) -> str:
    """Address the complete private tree mounted as writable Grok state.

    A started provider may add session state.  Such a change deliberately
    invalidates this launch source address: restart requires a newly reviewed
    exact state root rather than trusting an unrecorded writable residue.
    """

    try:
        before = os.lstat(provider_home)
    except OSError as exc:
        raise ValueError("Grok provider-home source is unavailable") from exc
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o700
        or not provider_home.name.startswith("asref-grok-home-")
    ):
        raise ValueError("Grok provider-home source is not private")
    entries: list[dict[str, object]] = []
    directory_snapshots: dict[Path, tuple[int, ...]] = {
        provider_home: (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
    }
    try:
        for root, directory_names, file_names in os.walk(
            provider_home,
            topdown=True,
            followlinks=False,
        ):
            directory_names.sort()
            file_names.sort()
            root_path = Path(root)
            for name in directory_names:
                path = root_path / name
                metadata = os.lstat(path)
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.geteuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                ):
                    raise ValueError("Grok provider-home directory is not private")
                entries.append(
                    {
                        "path": str(path.relative_to(provider_home)),
                        "kind": "directory",
                        "mode": "0700",
                    }
                )
                directory_snapshots[path] = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_uid,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
            for name in file_names:
                path = root_path / name
                relative = str(path.relative_to(provider_home))
                metadata = os.lstat(path)
                if stat.S_ISLNK(metadata.st_mode):
                    if relative != "auth.json":
                        raise ValueError(
                            "Grok provider-home contains an unapproved symlink"
                        )
                    target = path.resolve(strict=True)
                    _stable_private_file_bytes(target)
                    entries.append(
                        {
                            "path": relative,
                            "kind": "auth_symlink",
                            "target": str(target),
                        }
                    )
                elif stat.S_ISREG(metadata.st_mode):
                    entries.append(
                        {
                            "path": relative,
                            "kind": "file",
                            "source_identity": (
                                worker_container_execution_file_source_identity(path)
                            ),
                        }
                    )
                else:
                    raise ValueError(
                        "Grok provider-home contains an unsupported entry"
                    )
    except OSError as exc:
        raise ValueError("Grok provider-home source is unavailable") from exc
    required_files = {
        "alternate-provider-deny-sentinel",
        "config.toml",
        "sandbox.toml",
    }
    observed_files = {
        str(entry["path"])
        for entry in entries
        if entry.get("kind") == "file"
    }
    if not required_files.issubset(observed_files):
        raise ValueError("Grok provider-home launch controls are incomplete")
    try:
        for directory, expected in directory_snapshots.items():
            current = os.lstat(directory)
            observed = (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_uid,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            if observed != expected:
                raise ValueError("Grok provider-home source changed while read")
    except OSError as exc:
        raise ValueError("Grok provider-home source changed while read") from exc
    return _content_id(
        {
            "schema": _GROK_PROVIDER_HOME_IDENTITY_SCHEMA,
            "entries": entries,
        }
    )


def validate_worker_container_execution_grok_mount_sources(
    profile: WorkerContainerExecutionProfile,
    *,
    workspace: Path,
    prompt_path: Path,
    policy_path: Path,
    provider_home: Path,
) -> None:
    """Bind concrete Grok sources to the three independently signed CIDs."""

    if (
        profile.schema != EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2
        or profile.profile_schema != EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2
    ):
        raise ValueError("qualified Grok requires the signed execution profile @2")
    concrete = {
        "workspace": Path(workspace),
        "prompt": Path(prompt_path),
        "policy": Path(policy_path),
        "home": Path(provider_home),
    }
    if any(
        not path.is_absolute()
        or path != path.resolve(strict=True)
        or "\x00" in str(path)
        or "," in str(path)
        for path in concrete.values()
    ):
        raise ValueError("qualified Grok mount source path is invalid")
    if (
        concrete["policy"].parent != concrete["home"]
        or concrete["prompt"].is_relative_to(concrete["workspace"])
        or concrete["home"].is_relative_to(concrete["workspace"])
        or concrete["prompt"].is_relative_to(concrete["home"])
    ):
        raise ValueError("qualified Grok mount source path is not isolated")
    auth_link = concrete["home"] / "auth.json"
    try:
        auth_target = auth_link.resolve(strict=True)
    except OSError as exc:
        raise ValueError("qualified Grok provider auth source is unavailable") from exc
    if not auth_link.is_symlink() or auth_target.is_relative_to(concrete["workspace"]):
        raise ValueError("qualified Grok provider auth source path is invalid")
    identities = {
        "provider_auth": worker_container_execution_file_source_identity(auth_target),
        "grok_prompt": worker_container_execution_file_source_identity(
            concrete["prompt"]
        ),
        "grok_policy": worker_container_execution_file_source_identity(
            concrete["policy"]
        ),
        "grok_provider_home": (
            worker_container_execution_grok_provider_home_source_identity(
                concrete["home"]
            )
        ),
    }
    for kind, identity in identities.items():
        mount = profile.mount_for_kind(kind)
        if mount is None or mount.source_identity != identity:
            raise ValueError(f"qualified Grok {kind} source identity drifted")


def _project(
    *,
    value: Mapping[str, Any],
    source_file_cid: str,
    source_path: Path,
    trusted_root: Path,
) -> WorkerContainerExecutionProfile:
    profile = value["profile"]
    assert isinstance(profile, Mapping)
    gpu = profile["gpu"]
    mounts = profile["mounts"]
    environment = profile["environment"]
    assert isinstance(gpu, Mapping)
    assert isinstance(mounts, list)
    assert isinstance(environment, Mapping)
    return WorkerContainerExecutionProfile(
        schema=str(value["schema"]),
        profile_schema=str(profile["schema"]),
        artifact_cid=str(value["artifact_cid"]),
        source_file_cid=source_file_cid,
        profile_cid=str(profile["profile_cid"]),
        image_digest=str(profile["image_digest"]),
        source_head=str(value["source_head"]),
        source_tree=str(value["source_tree"]),
        accepted_control_plane_capsule_id=str(
            value["accepted_control_plane_capsule_id"]
        ),
        runtime=str(profile["runtime"]),
        execution_mode=str(profile["execution_mode"]),
        engine_endpoint=str(value["engine_endpoint"]),
        daemon_identity_cid=str(profile["daemon_identity_cid"]),
        daemon_policy_cid=str(profile["daemon_policy_cid"]),
        nonroot_user=str(profile["nonroot_user"]),
        read_only_base=bool(profile["read_only_base"]),
        network_mode=str(profile["network_mode"]),
        cap_drop=tuple(str(item) for item in profile["cap_drop"]),
        no_new_privileges=bool(profile["no_new_privileges"]),
        pids_limit=int(profile["pids_limit"]),
        cpu_limit=float(profile["cpu_limit"]),
        memory_limit_bytes=int(profile["memory_limit_bytes"]),
        disk_limit_bytes=int(profile["disk_limit_bytes"]),
        maximum_parallel_workers=int(profile["maximum_parallel_workers"]),
        maximum_parallel_containers=int(profile["maximum_parallel_containers"]),
        gpu_mode=str(gpu["mode"]),
        gpu_device_ids=tuple(str(item) for item in gpu["device_ids"]),
        gpu_memory_limit_bytes=int(gpu["memory_limit_bytes"]),
        privileged=bool(profile["privileged"]),
        host_pid=bool(profile["host_pid"]),
        host_ipc=bool(profile["host_ipc"]),
        devices=tuple(str(item) for item in profile["devices"]),
        docker_socket_mounted=bool(profile["docker_socket_mounted"]),
        inherit_host_environment=bool(profile["inherit_host_environment"]),
        environment=tuple(
            sorted((str(name), str(item)) for name, item in environment.items())
        ),
        mounts=tuple(
            WorkerContainerExecutionMount(
                source_identity=str(item["source_identity"]),
                target=str(item["target"]),
                read_only=bool(item["read_only"]),
                kind=str(item["kind"]),
            )
            for item in mounts
        ),
        signer_identity_did=str(value["signer_identity_did"]),
        issued_at_ms=int(value["issued_at_ms"]),
        expires_at_ms=int(value["expires_at_ms"]),
        source_path=source_path,
        trusted_root=trusted_root,
    )


def _verify_loaded(
    value: Mapping[str, Any],
    *,
    source_file_cid: str,
    source_path: Path,
    trusted_root: Path,
    launch_authority: Mapping[str, object],
    invocation_binding: object | None,
    expected_artifact_cid: str,
    now_ms: int,
) -> WorkerContainerExecutionProfile:
    if set(value) != _FIELDS:
        raise ValueError("worker container execution profile shape is invalid")
    profile = value.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("worker container execution profile body is invalid")
    body = {key: item for key, item in value.items() if key != "artifact_cid"}
    issued = value.get("issued_at_ms")
    expires = value.get("expires_at_ms")
    signer = str(value.get("signer_identity_did") or "")
    profile_cid = str(
        launch_authority.get("qualified_worker_container_profile_cid") or ""
    )
    image_digest = str(
        launch_authority.get("qualified_worker_image_digest") or ""
    )
    schema_pair = (value.get("schema"), profile.get("schema"))
    expected_signer_role = (
        EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2
        if schema_pair
        == (
            EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2,
            EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
        )
        else EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE
    )
    if (
        schema_pair
        not in {
            (
                EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA,
                EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA,
            ),
            (
                EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2,
                EAAEF_WORKER_CONTAINER_PROFILE_SCHEMA_V2,
            ),
        }
        or value.get("artifact_cid") != _content_id(body)
        or (
            expected_artifact_cid
            and value.get("artifact_cid") != expected_artifact_cid
        )
        or value.get("source_head") != launch_authority.get("source_head")
        or value.get("source_tree") != launch_authority.get("source_tree")
        or value.get("accepted_control_plane_capsule_id")
        != launch_authority.get("accepted_control_plane_capsule_id")
        or value.get("qualified_worker_image_digest") != image_digest
        or value.get("qualified_worker_container_profile_cid") != profile_cid
        or profile.get("profile_cid") != profile_cid
        or profile.get("image_digest") != image_digest
        or profile.get("worker_principal_did")
        != launch_authority.get("worker_principal_did")
        or profile.get("provider_principal_did")
        != launch_authority.get("provider_principal_did")
        or signer != profile.get("reviewer_identity_did")
        or signer
        in {
            launch_authority.get("worker_principal_did"),
            launch_authority.get("provider_principal_did"),
        }
        or value.get("signer_role") != expected_signer_role
        or isinstance(issued, bool)
        or not isinstance(issued, int)
        or isinstance(expires, bool)
        or not isinstance(expires, int)
        or issued <= 0
        or issued > now_ms
        or now_ms >= expires
        or expires - issued > _MAX_LIFETIME_MS
        or expires > int(profile.get("expires_at_ms") or 0)
        or not _GIT_OBJECT.fullmatch(str(value.get("source_head") or ""))
        or not _GIT_OBJECT.fullmatch(str(value.get("source_tree") or ""))
        or any(
            _CID.fullmatch(str(value.get(name) or "")) is None
            for name in (
                "accepted_control_plane_capsule_id",
                "qualified_worker_image_digest",
                "qualified_worker_container_profile_cid",
                "artifact_cid",
            )
        )
    ):
        raise ValueError("worker container execution profile binding is invalid")
    invocation_expiry = int(getattr(invocation_binding, "expires_at_ms", 0) or 0)
    if invocation_binding is not None and (
        str(getattr(invocation_binding, "resource_cid", "")) != profile_cid
        or (
            invocation_expiry > 0
            and expires > invocation_expiry
        )
    ):
        raise ValueError("worker container execution profile invocation drifted")
    reason = validate_eaaef_worker_container_profile_artifact(
        profile,
        expected_profile_cid=profile_cid,
        expected_image_digest=image_digest,
        expected_worker_principal_did=str(
            launch_authority.get("worker_principal_did") or ""
        ),
        expected_provider_principal_did=str(
            launch_authority.get("provider_principal_did") or ""
        ),
        now_ms=now_ms,
    )
    if reason:
        raise ValueError(reason)
    if invocation_binding is not None:
        expected_worktree_id = str(
            getattr(invocation_binding, "worktree_id", "") or ""
        )
        worktree_mounts = tuple(
            item
            for item in profile.get("mounts", ())
            if isinstance(item, Mapping) and item.get("kind") == "worktree"
        )
        if (
            _CID.fullmatch(expected_worktree_id) is None
            or len(worktree_mounts) != 1
            or worktree_mounts[0].get("source_identity")
            != expected_worktree_id
        ):
            raise ValueError(
                "worker container execution profile worktree identity drifted"
            )
    _engine_endpoint(profile, value.get("engine_endpoint"))
    signature = value.get("signature")
    if not isinstance(signature, str) or not signature:
        raise ValueError("worker container execution profile signature is invalid")
    try:
        verify_did_key_signature(
            identity_did=signer,
            payload={
                key: item
                for key, item in value.items()
                if key not in {"signature", "artifact_cid"}
            },
            signature=signature,
        )
    except (LocalProfileTampered, ValueError) as exc:
        raise ValueError(
            "worker container execution profile signature is invalid"
        ) from exc
    return _project(
        value=value,
        source_file_cid=source_file_cid,
        source_path=source_path,
        trusted_root=trusted_root,
    )


def load_worker_container_execution_profile(
    *,
    launch_authority: str | Mapping[str, object],
    invocation_binding: object,
    now_ms: int | None = None,
    expected_artifact_cid: str = "",
) -> WorkerContainerExecutionProfile:
    """Stable-read the only profile artifact admitted for an invocation."""

    launch = parse_worker_network_launch_authority(
        launch_authority,
        accepted_control_plane_pin=getattr(invocation_binding, "control_plane", None),
        require_admitted=True,
    )
    profile_cid = str(launch["qualified_worker_container_profile_cid"])
    trusted_root = Path(str(getattr(invocation_binding, "profile_dir", "")))
    if not trusted_root.is_absolute():
        raise ValueError(
            f"{EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA} root is not absolute"
        )
    relative = worker_container_execution_profile_relative_path(profile_cid)
    source_path = trusted_root / relative
    try:
        value, source_file_cid = _stable_private_json(
            source_path,
            trusted_root=trusted_root,
        )
    except OSError as exc:
        raise ValueError(
            f"{EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA} artifact is unavailable"
        ) from exc
    return _verify_loaded(
        value,
        source_file_cid=source_file_cid,
        source_path=source_path,
        trusted_root=trusted_root,
        launch_authority=launch,
        invocation_binding=invocation_binding,
        expected_artifact_cid=expected_artifact_cid,
        now_ms=int(now_ms if now_ms is not None else time.time() * 1000),
    )


def reverify_worker_container_execution_profile(
    profile: WorkerContainerExecutionProfile,
    *,
    launch_authority: str | Mapping[str, object],
    now_ms: int | None = None,
) -> WorkerContainerExecutionProfile:
    """Re-read a profile artifact and require byte/content identity stability."""

    if not isinstance(profile, WorkerContainerExecutionProfile):
        raise ValueError("worker container execution profile is absent")
    launch = parse_worker_network_launch_authority(
        launch_authority,
        require_admitted=True,
    )
    expected_path = profile.trusted_root / (
        worker_container_execution_profile_relative_path(profile.profile_cid)
    )
    if profile.source_path != expected_path:
        raise ValueError("worker container execution profile source path drifted")
    try:
        value, source_file_cid = _stable_private_json(
            expected_path,
            trusted_root=profile.trusted_root,
        )
    except OSError as exc:
        raise ValueError(
            f"{EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA} artifact is unavailable"
        ) from exc
    verified = _verify_loaded(
        value,
        source_file_cid=source_file_cid,
        source_path=expected_path,
        trusted_root=profile.trusted_root,
        launch_authority=launch,
        invocation_binding=None,
        expected_artifact_cid=profile.artifact_cid,
        now_ms=int(now_ms if now_ms is not None else time.time() * 1000),
    )
    if verified != profile:
        raise ValueError("worker container execution profile projection drifted")
    return verified


def reverify_worker_container_execution_grok_mounts(
    profile: WorkerContainerExecutionProfile,
    *,
    launch_authority: str | Mapping[str, object],
    workspace: Path,
    prompt_path: Path,
    policy_path: Path,
    provider_home: Path,
    now_ms: int | None = None,
) -> WorkerContainerExecutionProfile:
    """Reverify the signed @2 artifact and all concrete Grok mount sources."""

    verified = reverify_worker_container_execution_profile(
        profile,
        launch_authority=launch_authority,
        now_ms=now_ms,
    )
    validate_worker_container_execution_grok_mount_sources(
        verified,
        workspace=workspace,
        prompt_path=prompt_path,
        policy_path=policy_path,
        provider_home=provider_home,
    )
    return verified


__all__ = (
    "EAAEF_GROK_POLICY_MOUNT_TARGET",
    "EAAEF_GROK_PROMPT_MOUNT_TARGET",
    "EAAEF_GROK_PROVIDER_HOME_MOUNT_TARGET",
    "EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA",
    "EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SCHEMA_V2",
    "EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE",
    "EAAEF_WORKER_CONTAINER_EXECUTION_PROFILE_SIGNER_ROLE_V2",
    "WorkerContainerExecutionMount",
    "WorkerContainerExecutionProfile",
    "load_worker_container_execution_profile",
    "reverify_worker_container_execution_grok_mounts",
    "reverify_worker_container_execution_profile",
    "validate_worker_container_execution_grok_mount_sources",
    "worker_container_execution_file_source_identity",
    "worker_container_execution_grok_provider_home_source_identity",
    "worker_container_execution_profile_relative_path",
    "worker_container_execution_profile_signing_bytes",
)
