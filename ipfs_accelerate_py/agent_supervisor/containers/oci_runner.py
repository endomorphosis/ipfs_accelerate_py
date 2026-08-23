"""Default-deny OCI launch planner for EAAEF-051.

This module turns an EAAEF-050 :class:`ContainerExecutionProfile` into a
structured argv / run spec.  It does not inspect, create, start, or otherwise
contact a live container engine.  Workers are planned as nonroot, read-only
base, capability-dropped, no-new-privileges processes with PID/CPU/RAM/GPU/
disk/time bounds, no Docker socket, and network deny.

A rootless engine is selected whenever the admission record says it is
supported and verified.  A rootful-host-daemon / nonroot-worker fallback is
admitted only by an explicit independent policy flag that is off by default.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    ContainerContractError,
    ContainerExecutionProfile,
    ContainerMount,
    ContainerTrustError,
    IsolationPolicy,
    MountKind,
    NetworkPolicy,
    ResourceBounds,
    _contains_docker_socket,
)


DEFAULT_NONROOT_USER: Final[str] = "65532:65532"
DEFAULT_PIDS_LIMIT: Final[int] = 256
DEFAULT_TMPFS_SIZE_BYTES: Final[int] = 67_108_864
DEFAULT_ENGINE_BINARY: Final[str] = "docker"
ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT: Final[bool] = False
ROOTLESS_ENGINE_MODE: Final[str] = "rootless_engine"
ROOTFUL_ENGINE_MODE: Final[str] = "rootful_daemon_nonroot_worker"
ROOTFUL_ENGINE_ENDPOINT: Final[str] = "unix:///var/run/docker.sock"

_DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_ENGINE_TOKENS: Final[tuple[str, ...]] = (
    "--privileged",
    "--cap-add",
    "--network=host",
    "--network=bridge",
    "--network=container",
    "--pid=host",
    "--ipc=host",
    "--uts=host",
    "--userns=host",
    "--security-opt=seccomp=unconfined",
    "--security-opt=apparmor=unconfined",
    "--device",
    "--gpus=all",
    "--runtime=sysbox",
    "--runtime=kata",
)


class OciRunnerError(ContainerContractError):
    """The OCI runner rejected a launch plan."""


class OciRunnerTrustError(OciRunnerError, ContainerTrustError):
    """The planned worker would violate default-deny isolation."""


class OciEngineAdmissionError(OciRunnerTrustError):
    """Rootful fallback was requested without independent policy admission."""


class EngineMode(str, Enum):
    """Closed engine vocabulary for the planned worker."""

    ROOTLESS = ROOTLESS_ENGINE_MODE
    ROOTFUL_DAEMON_NONROOT_WORKER = ROOTFUL_ENGINE_MODE


@dataclass(frozen=True)
class EngineAdmission:
    """Independent engine-mode admission.  Rootful fallback is off by default."""

    rootless_supported: bool = True
    rootless_verified: bool = True
    rootful_fallback_admitted: bool = ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT
    independent_security_approval: bool = False
    daemon_identity_cid: str = ""
    daemon_policy_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rootless_supported", _require_bool(self.rootless_supported, "rootless_supported")
        )
        object.__setattr__(
            self, "rootless_verified", _require_bool(self.rootless_verified, "rootless_verified")
        )
        object.__setattr__(
            self,
            "rootful_fallback_admitted",
            _require_bool(self.rootful_fallback_admitted, "rootful_fallback_admitted"),
        )
        object.__setattr__(
            self,
            "independent_security_approval",
            _require_bool(
                self.independent_security_approval, "independent_security_approval"
            ),
        )
        object.__setattr__(
            self,
            "daemon_identity_cid",
            str(self.daemon_identity_cid or "").strip(),
        )
        object.__setattr__(
            self, "daemon_policy_cid", str(self.daemon_policy_cid or "").strip()
        )


@dataclass(frozen=True)
class OciRunSpec:
    """Structured argv / run spec.  Building it never invokes an engine."""

    argv: tuple[str, ...]
    engine_binary: str
    engine_mode: EngineMode
    engine_endpoint: str
    image: str
    user: str
    read_only: bool
    network: str
    cap_drop: tuple[str, ...]
    no_new_privileges: bool
    privileged: bool
    docker_socket_mounted: bool
    inherit_host_environment: bool
    host_pid: bool
    host_ipc: bool
    pids_limit: int
    cpu_millicores: int
    ram_mib: int
    disk_mib: int
    gpu_count: int
    timeout_seconds: int
    rootful_fallback_admitted: bool
    rootless_preferred: bool
    mounts: tuple[ContainerMount, ...]
    live_engine_invoked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "engine_binary": self.engine_binary,
            "engine_mode": self.engine_mode.value,
            "engine_endpoint": self.engine_endpoint,
            "image": self.image,
            "user": self.user,
            "read_only": self.read_only,
            "network": self.network,
            "cap_drop": list(self.cap_drop),
            "no_new_privileges": self.no_new_privileges,
            "privileged": self.privileged,
            "docker_socket_mounted": self.docker_socket_mounted,
            "inherit_host_environment": self.inherit_host_environment,
            "host_pid": self.host_pid,
            "host_ipc": self.host_ipc,
            "pids_limit": self.pids_limit,
            "cpu_millicores": self.cpu_millicores,
            "ram_mib": self.ram_mib,
            "disk_mib": self.disk_mib,
            "gpu_count": self.gpu_count,
            "timeout_seconds": self.timeout_seconds,
            "rootful_fallback_admitted": self.rootful_fallback_admitted,
            "rootless_preferred": self.rootless_preferred,
            "mounts": [mount.to_dict() for mount in self.mounts],
            "live_engine_invoked": False,
        }


class OciRunner:
    """Plan default-deny OCI workers without contacting a live engine."""

    DEFAULT_NONROOT_USER: ClassVar[str] = DEFAULT_NONROOT_USER
    ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT: ClassVar[bool] = (
        ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT
    )

    def __init__(self, *, engine_admission: EngineAdmission | None = None) -> None:
        self._engine_admission = (
            engine_admission if engine_admission is not None else EngineAdmission()
        )

    def build_spec(
        self,
        profile: ContainerExecutionProfile | Mapping[str, Any],
        **kwargs: Any,
    ) -> OciRunSpec:
        admission = kwargs.pop("engine_admission", self._engine_admission)
        return build_oci_run_spec(profile, engine_admission=admission, **kwargs)


def default_engine_admission() -> EngineAdmission:
    """Return the closed default: rootless preferred, rootful fallback off."""

    return EngineAdmission()


def select_engine_mode(admission: EngineAdmission | None = None) -> EngineMode:
    """Prefer a verified rootless engine; otherwise require independent admission."""

    policy = admission if admission is not None else EngineAdmission()
    if policy.rootless_supported and policy.rootless_verified:
        return EngineMode.ROOTLESS
    if policy.rootless_supported and not policy.rootless_verified:
        raise OciEngineAdmissionError(
            "rootless engine is supported but not independently verified"
        )
    if _rootful_fallback_is_admitted(policy):
        return EngineMode.ROOTFUL_DAEMON_NONROOT_WORKER
    raise OciEngineAdmissionError(
        "rootful-host-daemon fallback requires independent policy admission"
    )


def build_oci_run_spec(
    profile: ContainerExecutionProfile | Mapping[str, Any],
    *,
    command: Sequence[str] = (),
    worktree_source: str = "",
    extra_mounts: Sequence[ContainerMount | Mapping[str, Any]] = (),
    extra_args: Sequence[str] = (),
    engine_admission: EngineAdmission | None = None,
    engine_binary: str = DEFAULT_ENGINE_BINARY,
    user: str = DEFAULT_NONROOT_USER,
    pids_limit: int = DEFAULT_PIDS_LIMIT,
    privileged: bool = False,
) -> OciRunSpec:
    """Build a default-deny OCI argv / run spec without invoking an engine."""

    resolved = _coerce_profile(profile)
    policy = resolved.policy
    resources = resolved.resources
    _assert_default_deny_policy(policy)
    if privileged or policy.privileged or any(
        _is_privileged_flag(item) for item in extra_args
    ):
        raise OciRunnerTrustError("privileged workers are prohibited")
    _reject_forbidden_engine_text(*extra_args, artifact_name="oci extra args")
    for item in extra_args:
        text = str(item).strip()
        if text in {"-v", "--volume", "--mount"} or _looks_like_mount(text):
            raise OciRunnerTrustError("oci extra args must not add bind mounts")

    nonroot = _require_nonroot(user)
    pids = _require_positive_int(pids_limit, "pids_limit")
    mounts = _collect_mounts(
        policy.mounts,
        extra_mounts,
        worktree_source=worktree_source,
    )
    admission = engine_admission if engine_admission is not None else EngineAdmission()
    engine_mode = select_engine_mode(admission)
    endpoint = _engine_endpoint(engine_mode)
    binary = _require_engine_binary(engine_binary)
    argv = _build_argv(
        binary=binary,
        endpoint=endpoint,
        image=resolved.image_digest,
        user=nonroot,
        resources=resources,
        pids_limit=pids,
        mounts=mounts,
        command=command,
        extra_args=extra_args,
    )
    _reject_forbidden_engine_text(*argv, artifact_name="oci argv")
    _reject_docker_socket_mounts(argv, mounts)
    if "--privileged" in argv:
        raise OciRunnerTrustError("privileged workers are prohibited")

    return OciRunSpec(
        argv=argv,
        engine_binary=binary,
        engine_mode=engine_mode,
        engine_endpoint=endpoint,
        image=resolved.image_digest,
        user=nonroot,
        read_only=True,
        network="none",
        cap_drop=("ALL",),
        no_new_privileges=True,
        privileged=False,
        docker_socket_mounted=False,
        inherit_host_environment=False,
        host_pid=False,
        host_ipc=False,
        pids_limit=pids,
        cpu_millicores=resources.cpu_millicores,
        ram_mib=resources.ram_mib,
        disk_mib=resources.disk_mib,
        gpu_count=resources.gpu_count,
        timeout_seconds=resources.timeout_seconds,
        rootful_fallback_admitted=(
            engine_mode is EngineMode.ROOTFUL_DAEMON_NONROOT_WORKER
        ),
        rootless_preferred=True,
        mounts=mounts,
        live_engine_invoked=False,
    )


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OciRunnerError(f"{name} must be a boolean")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OciRunnerError(f"{name} must be a positive integer")
    return value


def _require_digest(value: str, name: str) -> str:
    text = str(value or "").strip()
    if _DIGEST_RE.fullmatch(text) is None:
        raise OciEngineAdmissionError(f"{name} must be a sha256 digest")
    return text


def _require_nonroot(user: str) -> str:
    text = str(user or "").strip()
    uid_text, separator, gid_text = text.partition(":")
    if not separator or not uid_text.isdigit() or not gid_text.isdigit():
        raise OciRunnerTrustError("nonroot user must be uid:gid")
    if int(uid_text) == 0 or int(gid_text) == 0:
        raise OciRunnerTrustError("nonroot uid is required")
    return f"{int(uid_text)}:{int(gid_text)}"


def _require_engine_binary(value: str) -> str:
    text = str(value or "").strip()
    if text not in {"docker", "podman"}:
        raise OciRunnerError("engine binary must be docker or podman")
    return text


def _rootful_fallback_is_admitted(policy: EngineAdmission) -> bool:
    if policy.rootful_fallback_admitted is not True:
        return False
    if policy.independent_security_approval is not True:
        raise OciEngineAdmissionError(
            "rootful-host-daemon fallback requires independent policy admission"
        )
    if policy.rootless_supported or policy.rootless_verified:
        raise OciEngineAdmissionError(
            "rootful-host-daemon fallback requires an unsupported rootless engine"
        )
    identity = _require_digest(policy.daemon_identity_cid, "daemon_identity_cid")
    admitted_policy = _require_digest(policy.daemon_policy_cid, "daemon_policy_cid")
    if identity == admitted_policy:
        raise OciEngineAdmissionError(
            "rootful daemon identity must be distinct from daemon policy"
        )
    return True


def _engine_endpoint(mode: EngineMode) -> str:
    if mode is EngineMode.ROOTLESS:
        return f"unix:///run/user/{os.geteuid()}/docker.sock"
    return ROOTFUL_ENGINE_ENDPOINT


def _coerce_profile(
    value: ContainerExecutionProfile | Mapping[str, Any],
) -> ContainerExecutionProfile:
    if isinstance(value, ContainerExecutionProfile):
        return value
    if isinstance(value, Mapping):
        return ContainerExecutionProfile.from_dict(value)
    raise OciRunnerError("profile must be a ContainerExecutionProfile")


def _assert_default_deny_policy(policy: IsolationPolicy) -> None:
    if policy.network_policy is not NetworkPolicy.DENY:
        raise OciRunnerTrustError("network policy must default-deny")
    if policy.docker_socket_mounted:
        raise OciRunnerTrustError("docker.sock mounts are prohibited")
    if not policy.no_new_privileges:
        raise OciRunnerTrustError("no-new-privileges is required")
    if not policy.read_only_base:
        raise OciRunnerTrustError("read-only base filesystem is required")
    if policy.privileged:
        raise OciRunnerTrustError("privileged workers are prohibited")


def _is_engine_host_flag(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered == "--host" or lowered.startswith("--host=")


def _is_privileged_flag(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    return lowered == "--privileged" or lowered.startswith("--privileged=")


def _reject_forbidden_engine_text(*values: str, artifact_name: str) -> None:
    for value in values:
        text = str(value or "")
        lowered = text.strip().lower()
        if not lowered or _is_engine_host_flag(text):
            continue
        if _contains_docker_socket(text):
            raise OciRunnerTrustError(f"{artifact_name} docker.sock mounts are prohibited")
        if any(token in lowered for token in _FORBIDDEN_ENGINE_TOKENS):
            raise OciRunnerTrustError(f"{artifact_name} isolation escape is prohibited")


def _looks_like_mount(value: str) -> bool:
    lowered = value.strip().lower().replace("\\", "/")
    if _is_engine_host_flag(lowered):
        return False
    return (
        lowered.startswith("-v")
        or lowered.startswith("--volume")
        or lowered.startswith("--mount")
        or "type=bind" in lowered
    )


def _collect_mounts(
    policy_mounts: Sequence[ContainerMount],
    extra_mounts: Sequence[ContainerMount | Mapping[str, Any]],
    *,
    worktree_source: str,
) -> tuple[ContainerMount, ...]:
    collected: list[ContainerMount] = []
    seen: set[str] = set()
    items: list[ContainerMount | Mapping[str, Any]] = list(policy_mounts)
    items.extend(extra_mounts)
    source = str(worktree_source or "").strip()
    if source:
        items.append(
            {
                "source": source,
                "target": "/workspace",
                "read_only": False,
                "kind": MountKind.WORKTREE.value,
            }
        )
    for item in items:
        mount = _coerce_mount(item)
        _reject_docker_socket_text(mount.source, mount.target)
        key = f"{mount.source}->{mount.target}"
        if key in seen:
            raise OciRunnerError("mounts must not contain duplicate targets")
        seen.add(key)
        collected.append(mount)
    return tuple(collected)


def _coerce_mount(value: ContainerMount | Mapping[str, Any]) -> ContainerMount:
    try:
        if isinstance(value, ContainerMount):
            return value
        if isinstance(value, Mapping):
            return ContainerMount.from_dict(value)
    except ContainerTrustError as exc:
        raise OciRunnerTrustError(str(exc)) from exc
    raise OciRunnerError("mount must be a ContainerMount object")


def _reject_docker_socket_text(*values: str) -> None:
    for value in values:
        if _contains_docker_socket(value):
            raise OciRunnerTrustError("docker.sock mounts are prohibited")


def _reject_docker_socket_mounts(
    argv: Sequence[str], mounts: Sequence[ContainerMount]
) -> None:
    for mount in mounts:
        _reject_docker_socket_text(mount.source, mount.target)
    for index, part in enumerate(argv):
        text = str(part)
        previous = str(argv[index - 1]) if index else ""
        if _is_engine_host_flag(text) or _is_engine_host_flag(previous):
            continue
        if previous in {"-v", "--volume", "--mount"} or _looks_like_mount(text):
            _reject_forbidden_engine_text(text, artifact_name="oci argv")
        elif _contains_docker_socket(text):
            raise OciRunnerTrustError("docker.sock mounts are prohibited")


def _cpus_flag(millicores: int) -> str:
    whole, fraction = divmod(millicores, 1000)
    if fraction == 0:
        return f"--cpus={whole}"
    return f"--cpus={(millicores / 1000):.3f}".rstrip("0").rstrip(".")


def _bytes_from_mib(mib: int) -> int:
    return mib * 1024 * 1024


def _mount_flag(mount: ContainerMount) -> str:
    source = mount.source
    target = mount.target
    if not source or not target.startswith("/") or ".." in source or ".." in target:
        raise OciRunnerTrustError("mount path is not admitted")
    if "," in source or "," in target or "\x00" in source or "\x00" in target:
        raise OciRunnerTrustError("mount path is not admitted")
    _reject_docker_socket_text(source, target)
    suffix = ",ro=true" if mount.read_only else ""
    return f"type=bind,src={source},dst={target}{suffix}"


def _build_argv(
    *,
    binary: str,
    endpoint: str,
    image: str,
    user: str,
    resources: ResourceBounds,
    pids_limit: int,
    mounts: Sequence[ContainerMount],
    command: Sequence[str],
    extra_args: Sequence[str],
) -> tuple[str, ...]:
    memory_bytes = _bytes_from_mib(resources.ram_mib)
    disk_bytes = _bytes_from_mib(resources.disk_mib)
    uid, _, gid = user.partition(":")
    argv: list[str] = [
        binary,
        f"--host={endpoint}",
        "create",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--pids-limit={pids_limit}",
        _cpus_flag(resources.cpu_millicores),
        f"--memory={memory_bytes}",
        f"--memory-swap={memory_bytes}",
        f"--storage-opt=size={disk_bytes}",
        f"--stop-timeout={resources.timeout_seconds}",
        f"--user={user}",
        (
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,"
            f"size={DEFAULT_TMPFS_SIZE_BYTES},mode=1777,uid={uid},gid={gid}"
        ),
    ]
    if resources.gpu_count:
        argv.append(f"--gpus={resources.gpu_count}")
    for extra in extra_args:
        flag = str(extra)
        if not flag or flag.startswith("-"):
            _reject_forbidden_engine_text(flag, artifact_name="oci extra args")
        argv.append(flag)
    for mount in mounts:
        argv.extend(["--mount", _mount_flag(mount)])
    argv.append(image)
    argv.extend(str(item) for item in command)
    if any(not part for part in argv):
        raise OciRunnerError("oci argv must not contain empty tokens")
    return tuple(argv)


__all__ = (
    "DEFAULT_ENGINE_BINARY",
    "DEFAULT_NONROOT_USER",
    "DEFAULT_PIDS_LIMIT",
    "ROOTFUL_ENGINE_ENDPOINT",
    "ROOTFUL_ENGINE_MODE",
    "ROOTFUL_FALLBACK_ADMITTED_BY_DEFAULT",
    "ROOTLESS_ENGINE_MODE",
    "EngineAdmission",
    "EngineMode",
    "OciEngineAdmissionError",
    "OciRunSpec",
    "OciRunner",
    "OciRunnerError",
    "OciRunnerTrustError",
    "build_oci_run_spec",
    "default_engine_admission",
    "select_engine_mode",
)
