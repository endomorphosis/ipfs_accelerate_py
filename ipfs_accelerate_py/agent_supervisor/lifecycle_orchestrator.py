"""Fenced process lifecycle orchestration for agent supervisors.

The control plane owns authorization, idempotency, and the outer mutation
transaction.  This module is the process-effect adapter used inside that
transaction.  It deliberately does not accept a bare PID: processes are
selected by an immutable launch profile and inherited run markers, and every
signal is preceded and followed by an exact ``pid + start-time`` check.

The small append-only saga journal is not a competing control transaction.  It
records the process checkpoints that the shared transaction cannot infer:
whether an old tree was proved absent, which new process was launched, and
whether health remained good for the requested window.  Those checkpoints make
an interrupted restart safely resumable without ever starting a second tree.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .control_contracts import Operation, OperationRequest
from .control_plane import (
    BackendConflictError,
    BackendResponse,
    PartialMutationError,
    StaleLeaseError,
    TransactionConflictError,
)


LIFECYCLE_PROFILE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lifecycle-profile@1"
)
PROCESS_IDENTITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/process-identity@1"
)
PROCESS_TREE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/process-tree@1"
LIFECYCLE_INTENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lifecycle-transition-intent@1"
)
LIFECYCLE_SAGA_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lifecycle-transition-saga@1"
)
LIFECYCLE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lifecycle-transition-receipt@1"
)

RUN_ID_ENV = "IPFS_ACCELERATE_LIFECYCLE_RUN_ID"
PROFILE_ID_ENV = "IPFS_ACCELERATE_LIFECYCLE_PROFILE_ID"
TARGET_ID_ENV = "IPFS_ACCELERATE_LIFECYCLE_TARGET_ID"
REPOSITORY_ROOT_ENV = "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT"
STATE_ROOT_ENV = "IPFS_ACCELERATE_LIFECYCLE_STATE_ROOT"
RUN_ROOT_ENV = "IPFS_ACCELERATE_LIFECYCLE_RUN_ROOT"
FENCING_EPOCH_ENV = "IPFS_ACCELERATE_LIFECYCLE_FENCING_EPOCH"
CONFIGURATION_ROOT_ENV = "IPFS_ACCELERATE_LIFECYCLE_CONFIGURATION_ROOT"

_MARKER_NAMES = (
    RUN_ID_ENV,
    PROFILE_ID_ENV,
    TARGET_ID_ENV,
    REPOSITORY_ROOT_ENV,
    STATE_ROOT_ENV,
    RUN_ROOT_ENV,
    FENCING_EPOCH_ENV,
    CONFIGURATION_ROOT_ENV,
)


def _canonical_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _absolute(value: str | Path, name: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return str(path.resolve(strict=False))


def _under(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve(strict=False).relative_to(
            Path(root).resolve(strict=False)
        )
        return True
    except ValueError:
        return False


def _text(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result or "\x00" in result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _read_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except (OSError, UnicodeError):
        return ""


class LifecycleAction(str, Enum):
    START = "start"
    STOP = "stop"
    RESTART = "restart"


class LifecycleSagaPhase(str, Enum):
    PREPARED = "prepared"
    STOPPING_OLD = "stopping_old"
    OLD_FENCED = "old_fenced"
    STARTING_NEW = "starting_new"
    VERIFYING_HEALTH = "verifying_health"
    COMMITTED = "committed"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            LifecycleSagaPhase.COMMITTED,
            LifecycleSagaPhase.FAILED,
        }


class LifecycleOrchestrationError(BackendConflictError):
    """A lifecycle precondition or postcondition did not hold."""


class ProcessIdentityMismatch(LifecycleOrchestrationError):
    """A PID now names a different OS process."""


class ProcessTreeNotFenced(PartialMutationError):
    """Shutdown left one or more exact descendants alive."""


class LifecycleProfileChanged(LifecycleOrchestrationError):
    """A restart attempted to replace the validated launch profile."""


class SplitBrainError(LifecycleOrchestrationError):
    """More than one root process exists for a single run/profile."""


@dataclass(frozen=True)
class LifecycleProfile:
    """Immutable, content-addressed launch configuration."""

    target_id: str
    run_id: str
    configuration_root: str
    repository_root: str
    state_root: str
    run_root: str
    argv: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...] = ()
    health_path: str = ""
    health_stale_ms: int = 30_000
    profile_id: str = ""

    def __post_init__(self) -> None:
        for name in ("target_id", "run_id", "configuration_root"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("repository_root", "state_root", "run_root", "cwd"):
            object.__setattr__(
                self, name, _absolute(getattr(self, name), name)
            )
        if not _under(self.cwd, self.repository_root):
            raise ValueError("profile cwd must be inside repository_root")
        if not _under(self.run_root, self.state_root):
            raise ValueError("profile run_root must be inside state_root")
        argv = tuple(_text(item, "argv item") for item in self.argv)
        if not argv or len(argv) > 512:
            raise ValueError("argv must contain between 1 and 512 items")
        object.__setattr__(self, "argv", argv)
        environment = tuple(
            sorted(
                (
                    _text(name, "environment name"),
                    str(value),
                )
                for name, value in self.environment
            )
        )
        names = [name for name, _value in environment]
        if len(names) != len(set(names)):
            raise ValueError("environment names must be unique")
        if set(names).intersection(_MARKER_NAMES):
            raise ValueError("profile environment cannot override lifecycle markers")
        object.__setattr__(self, "environment", environment)
        health_path = str(self.health_path).strip()
        if health_path:
            health_path = _absolute(health_path, "health_path")
            if not _under(health_path, self.state_root):
                raise ValueError("health_path must be inside state_root")
        object.__setattr__(self, "health_path", health_path)
        _positive_int(self.health_stale_ms, "health_stale_ms")
        expected = _canonical_id(self._payload())
        if self.profile_id and self.profile_id != expected:
            raise ValueError("profile_id does not match launch configuration")
        object.__setattr__(self, "profile_id", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_PROFILE_SCHEMA,
            "target_id": self.target_id,
            "run_id": self.run_id,
            "configuration_root": self.configuration_root,
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "run_root": self.run_root,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "environment": [list(item) for item in self.environment],
            "health_path": self.health_path,
            "health_stale_ms": self.health_stale_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "profile_id": self.profile_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleProfile":
        if payload.get("schema") not in (None, LIFECYCLE_PROFILE_SCHEMA):
            raise ValueError("unsupported lifecycle profile schema")
        environment = payload.get("environment") or ()
        if isinstance(environment, Mapping):
            environment = tuple(environment.items())
        return cls(
            target_id=payload.get("target_id", ""),
            run_id=payload.get("run_id", ""),
            configuration_root=payload.get("configuration_root", ""),
            repository_root=payload.get("repository_root", ""),
            state_root=payload.get("state_root", ""),
            run_root=payload.get("run_root", ""),
            argv=tuple(payload.get("argv") or ()),
            cwd=payload.get("cwd", ""),
            environment=tuple(tuple(item) for item in environment),
            health_path=payload.get("health_path", ""),
            health_stale_ms=payload.get("health_stale_ms", 30_000),
            profile_id=payload.get("profile_id", ""),
        )

    def launch_environment(self, fencing_epoch: int) -> dict[str, str]:
        result = dict(os.environ)
        result.update(dict(self.environment))
        result.update(
            {
                RUN_ID_ENV: self.run_id,
                PROFILE_ID_ENV: self.profile_id,
                TARGET_ID_ENV: self.target_id,
                REPOSITORY_ROOT_ENV: self.repository_root,
                STATE_ROOT_ENV: self.state_root,
                RUN_ROOT_ENV: self.run_root,
                FENCING_EPOCH_ENV: str(fencing_epoch),
                CONFIGURATION_ROOT_ENV: self.configuration_root,
            }
        )
        return result


# Descriptive alias retained for callers that prefer the acceptance terminology.
ValidatedLifecycleProfile = LifecycleProfile


@dataclass(frozen=True)
class ProcessIdentity:
    """PID-reuse-resistant identity for one Linux process."""

    pid: int
    start_time_ticks: int
    parent_pid: int
    process_group_id: int
    session_id: int
    boot_id: str
    argv: tuple[str, ...]
    cwd: str
    executable: str
    run_id: str
    profile_id: str
    target_id: str
    repository_root: str
    state_root: str
    run_root: str
    fencing_epoch: int
    configuration_root: str
    identity_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "pid",
            "start_time_ticks",
            "process_group_id",
            "session_id",
        ):
            _positive_int(getattr(self, name), name)
        _positive_int(self.parent_pid, "parent_pid", allow_zero=True)
        _positive_int(self.fencing_epoch, "fencing_epoch", allow_zero=True)
        for name in (
            "run_id",
            "profile_id",
            "target_id",
            "repository_root",
            "state_root",
            "run_root",
            "configuration_root",
        ):
            _text(getattr(self, name), name)
        expected = _canonical_id(self._payload())
        if self.identity_id and self.identity_id != expected:
            raise ValueError("identity_id does not match process identity")
        object.__setattr__(self, "identity_id", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROCESS_IDENTITY_SCHEMA,
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "parent_pid": self.parent_pid,
            "process_group_id": self.process_group_id,
            "session_id": self.session_id,
            "boot_id": self.boot_id,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "executable": self.executable,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "target_id": self.target_id,
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "run_root": self.run_root,
            "fencing_epoch": self.fencing_epoch,
            "configuration_root": self.configuration_root,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "identity_id": self.identity_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcessIdentity":
        if payload.get("schema") not in (None, PROCESS_IDENTITY_SCHEMA):
            raise ValueError("unsupported process identity schema")
        return cls(
            pid=payload.get("pid", 0),
            start_time_ticks=payload.get("start_time_ticks", 0),
            parent_pid=payload.get("parent_pid", 0),
            process_group_id=payload.get("process_group_id", 0),
            session_id=payload.get("session_id", 0),
            boot_id=payload.get("boot_id", ""),
            argv=tuple(payload.get("argv") or ()),
            cwd=payload.get("cwd", ""),
            executable=payload.get("executable", ""),
            run_id=payload.get("run_id", ""),
            profile_id=payload.get("profile_id", ""),
            target_id=payload.get("target_id", ""),
            repository_root=payload.get("repository_root", ""),
            state_root=payload.get("state_root", ""),
            run_root=payload.get("run_root", ""),
            fencing_epoch=payload.get("fencing_epoch", -1),
            configuration_root=payload.get("configuration_root", ""),
            identity_id=payload.get("identity_id", ""),
        )


@dataclass(frozen=True)
class ProcessTreeSnapshot:
    """A complete marker-selected tree at one observation point."""

    profile_id: str
    run_id: str
    members: tuple[ProcessIdentity, ...] = ()
    captured_at_ms: int = 0
    tree_id: str = ""

    def __post_init__(self) -> None:
        _text(self.profile_id, "profile_id")
        _text(self.run_id, "run_id")
        _positive_int(self.captured_at_ms, "captured_at_ms", allow_zero=True)
        members = tuple(sorted(self.members, key=lambda item: item.pid))
        if len({item.pid for item in members}) != len(members):
            raise ValueError("process tree contains duplicate PIDs")
        if any(
            item.profile_id != self.profile_id or item.run_id != self.run_id
            for item in members
        ):
            raise ValueError("process tree member binding mismatch")
        object.__setattr__(self, "members", members)
        expected = _canonical_id(self._payload())
        if self.tree_id and self.tree_id != expected:
            raise ValueError("tree_id does not match process tree")
        object.__setattr__(self, "tree_id", expected)

    @property
    def roots(self) -> tuple[ProcessIdentity, ...]:
        member_pids = {item.pid for item in self.members}
        return tuple(
            item for item in self.members if item.parent_pid not in member_pids
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROCESS_TREE_SCHEMA,
            "profile_id": self.profile_id,
            "run_id": self.run_id,
            "members": [item.to_dict() for item in self.members],
            "captured_at_ms": self.captured_at_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "tree_id": self.tree_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcessTreeSnapshot":
        if payload.get("schema") not in (None, PROCESS_TREE_SCHEMA):
            raise ValueError("unsupported process tree schema")
        return cls(
            profile_id=payload.get("profile_id", ""),
            run_id=payload.get("run_id", ""),
            members=tuple(
                ProcessIdentity.from_dict(item)
                for item in payload.get("members") or ()
            ),
            captured_at_ms=payload.get("captured_at_ms", 0),
            tree_id=payload.get("tree_id", ""),
        )


class ProcessAdapter(Protocol):
    def snapshot(self, profile: LifecycleProfile) -> ProcessTreeSnapshot:
        ...

    def launch(
        self, profile: LifecycleProfile, *, fencing_epoch: int
    ) -> ProcessIdentity:
        ...

    def terminate(
        self,
        tree: ProcessTreeSnapshot,
        *,
        grace_seconds: float,
        deadline_ms: int,
    ) -> None:
        ...

    def identity_alive(self, identity: ProcessIdentity) -> bool:
        ...

    def healthy(
        self,
        profile: LifecycleProfile,
        tree: ProcessTreeSnapshot,
        *,
        fencing_epoch: int,
        now_ms: int,
    ) -> bool:
        ...


class LinuxProcessAdapter:
    """Linux ``/proc`` implementation with exact inherited run markers."""

    def __init__(
        self,
        *,
        clock_ms: Callable[[], int] | None = None,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._popen = popen

    @staticmethod
    def _stat(pid: int) -> tuple[int, int, int, int]:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        if not fields or fields[0] == "Z":
            raise ProcessLookupError(pid)
        # state, ppid, pgrp, session, ... starttime is field 22 overall.
        return int(fields[1]), int(fields[2]), int(fields[3]), int(fields[19])

    @staticmethod
    def _environ(pid: int) -> dict[str, str]:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
        result: dict[str, str] = {}
        for item in raw.split(b"\0"):
            name, separator, value = item.partition(b"=")
            if separator:
                result[name.decode("utf-8", "surrogateescape")] = value.decode(
                    "utf-8", "surrogateescape"
                )
        return result

    @staticmethod
    def _argv(pid: int) -> tuple[str, ...]:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return tuple(
            item.decode("utf-8", "surrogateescape")
            for item in raw.rstrip(b"\0").split(b"\0")
            if item
        )

    def _identity(
        self, pid: int, profile: LifecycleProfile
    ) -> ProcessIdentity:
        parent, group, session, started = self._stat(pid)
        environment = self._environ(pid)
        markers = {
            RUN_ID_ENV: profile.run_id,
            PROFILE_ID_ENV: profile.profile_id,
            TARGET_ID_ENV: profile.target_id,
            REPOSITORY_ROOT_ENV: profile.repository_root,
            STATE_ROOT_ENV: profile.state_root,
            RUN_ROOT_ENV: profile.run_root,
            CONFIGURATION_ROOT_ENV: profile.configuration_root,
        }
        if any(environment.get(name) != value for name, value in markers.items()):
            raise ProcessIdentityMismatch(
                f"process {pid} does not belong to the selected run/profile"
            )
        try:
            fence = int(environment[FENCING_EPOCH_ENV])
        except (KeyError, ValueError) as exc:
            raise ProcessIdentityMismatch(
                f"process {pid} has no valid lifecycle fence"
            ) from exc
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            executable = os.readlink(f"/proc/{pid}/exe")
        except OSError as exc:
            raise ProcessLookupError(pid) from exc
        return ProcessIdentity(
            pid=pid,
            start_time_ticks=started,
            parent_pid=parent,
            process_group_id=group,
            session_id=session,
            boot_id=_read_boot_id(),
            argv=self._argv(pid),
            cwd=str(Path(cwd).resolve(strict=False)),
            executable=str(Path(executable).resolve(strict=False)),
            run_id=profile.run_id,
            profile_id=profile.profile_id,
            target_id=profile.target_id,
            repository_root=profile.repository_root,
            state_root=profile.state_root,
            run_root=profile.run_root,
            fencing_epoch=fence,
            configuration_root=profile.configuration_root,
        )

    def snapshot(self, profile: LifecycleProfile) -> ProcessTreeSnapshot:
        members: list[ProcessIdentity] = []
        try:
            entries = tuple(Path("/proc").iterdir())
        except OSError:
            entries = ()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                environment = self._environ(pid)
                if environment.get(RUN_ID_ENV) != profile.run_id:
                    continue
                if environment.get(TARGET_ID_ENV) != profile.target_id:
                    continue
                expected_markers = {
                    PROFILE_ID_ENV: profile.profile_id,
                    REPOSITORY_ROOT_ENV: profile.repository_root,
                    STATE_ROOT_ENV: profile.state_root,
                    RUN_ROOT_ENV: profile.run_root,
                    CONFIGURATION_ROOT_ENV: profile.configuration_root,
                }
                if any(
                    environment.get(name) != value
                    for name, value in expected_markers.items()
                ):
                    raise ProcessIdentityMismatch(
                        "the selected run already has a foreign root or "
                        "changed lifecycle configuration"
                    )
                members.append(self._identity(pid, profile))
            except (
                OSError,
                UnicodeError,
                ValueError,
                ProcessLookupError,
            ):
                continue
        return ProcessTreeSnapshot(
            profile_id=profile.profile_id,
            run_id=profile.run_id,
            members=tuple(members),
            captured_at_ms=self._clock_ms(),
        )

    def launch(
        self, profile: LifecycleProfile, *, fencing_epoch: int
    ) -> ProcessIdentity:
        Path(profile.run_root).mkdir(parents=True, exist_ok=True)
        process = self._popen(
            list(profile.argv),
            cwd=profile.cwd,
            env=profile.launch_environment(fencing_epoch),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return self._identity(process.pid, profile)
        except Exception:
            try:
                process.kill()
            except OSError:
                pass
            raise

    def identity_alive(self, identity: ProcessIdentity) -> bool:
        if identity.boot_id and _read_boot_id() != identity.boot_id:
            return False
        try:
            _parent, _group, _session, started = self._stat(identity.pid)
            return started == identity.start_time_ticks
        except (OSError, ValueError, ProcessLookupError):
            return False

    @staticmethod
    def _signal_exact(identity: ProcessIdentity, signum: int) -> None:
        try:
            _parent, group, _session, started = LinuxProcessAdapter._stat(
                identity.pid
            )
        except (OSError, ValueError, ProcessLookupError):
            return
        if started != identity.start_time_ticks:
            raise ProcessIdentityMismatch(
                f"PID {identity.pid} was reused before signal"
            )
        # Signal the exact process.  Group signaling is unsafe here because an
        # unrelated process can join a group after the snapshot.
        del group
        os.kill(identity.pid, signum)

    def terminate(
        self,
        tree: ProcessTreeSnapshot,
        *,
        grace_seconds: float,
        deadline_ms: int,
    ) -> None:
        # Descendants first prevents a cooperative root from exiting and
        # orphaning children before they receive the fence signal.
        member_pids = {item.pid for item in tree.members}

        def depth(item: ProcessIdentity) -> int:
            result = 0
            parent = item.parent_pid
            by_pid = {member.pid: member for member in tree.members}
            while parent in member_pids and result <= len(member_pids):
                result += 1
                parent = by_pid[parent].parent_pid
            return result

        ordered = sorted(tree.members, key=lambda item: (depth(item), item.pid), reverse=True)
        for member in ordered:
            try:
                self._signal_exact(member, signal.SIGTERM)
            except ProcessLookupError:
                pass
        wait_until = min(
            time.monotonic() + max(0.0, grace_seconds),
            time.monotonic() + max(0.0, deadline_ms / 1000.0),
        )
        while (
            any(self.identity_alive(member) for member in ordered)
            and time.monotonic() < wait_until
        ):
            time.sleep(0.02)
        for member in ordered:
            if self.identity_alive(member):
                try:
                    self._signal_exact(member, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def healthy(
        self,
        profile: LifecycleProfile,
        tree: ProcessTreeSnapshot,
        *,
        fencing_epoch: int,
        now_ms: int,
    ) -> bool:
        if len(tree.roots) != 1 or not tree.members:
            return False
        if any(not self.identity_alive(item) for item in tree.members):
            return False
        if not profile.health_path:
            # Launch alone is not health.  Production start/restart profiles
            # must name an authoritative heartbeat/status document.
            return False
        try:
            payload = json.loads(
                Path(profile.health_path).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, Mapping):
            return False
        state = str(
            payload.get("state")
            or payload.get("lifecycle_state")
            or payload.get("status")
            or ""
        ).lower()
        if state not in {"healthy", "running", "ok", "alive"}:
            return False
        root = tree.roots[0]
        try:
            status_pid = int(payload.get("pid"))
        except (TypeError, ValueError):
            return False
        if status_pid != root.pid:
            return False
        for name, expected in (
            ("run_id", profile.run_id),
            ("profile_id", profile.profile_id),
            ("target_id", profile.target_id),
            ("configuration_root", profile.configuration_root),
        ):
            if payload.get(name) not in (None, "", expected):
                return False
        if payload.get("fencing_epoch") not in (None, ""):
            try:
                if int(payload["fencing_epoch"]) != fencing_epoch:
                    return False
            except (TypeError, ValueError):
                return False
        updated = payload.get("heartbeat_at_ms", payload.get("updated_at_ms", 0))
        try:
            updated_ms = int(updated)
        except (TypeError, ValueError):
            return False
        return 0 <= now_ms - updated_ms <= profile.health_stale_ms


@dataclass(frozen=True)
class LifecycleTransitionIntent:
    """Durable intent written before the first process effect."""

    action: LifecycleAction
    request_id: str
    target_id: str
    repository_root: str
    state_root: str
    run_root: str
    run_id: str
    configuration_root: str
    profile_id: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str
    authorization_decision_id: str
    idempotency_key: str
    lease_id: str
    fencing_epoch: int
    expected_revision: int
    deadline_ms: int
    health_window_ms: int
    expected_effect_ids: tuple[str, ...]
    created_at_ms: int
    transition_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            self.action
            if isinstance(self.action, LifecycleAction)
            else LifecycleAction(str(self.action)),
        )
        expected = _canonical_id(self._payload())
        if self.transition_id and self.transition_id != expected:
            raise ValueError("transition_id does not match intent")
        object.__setattr__(self, "transition_id", expected)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_INTENT_SCHEMA,
            **{
                name: (
                    getattr(self, name).value
                    if isinstance(getattr(self, name), Enum)
                    else list(getattr(self, name))
                    if isinstance(getattr(self, name), tuple)
                    else getattr(self, name)
                )
                for name in (
                    "action",
                    "request_id",
                    "target_id",
                    "repository_root",
                    "state_root",
                    "run_root",
                    "run_id",
                    "configuration_root",
                    "profile_id",
                    "repository_id",
                    "tree_id",
                    "objective_id",
                    "objective_revision",
                    "policy_id",
                    "policy_revision",
                    "caller",
                    "authorization_decision_id",
                    "idempotency_key",
                    "lease_id",
                    "fencing_epoch",
                    "expected_revision",
                    "deadline_ms",
                    "health_window_ms",
                    "expected_effect_ids",
                    "created_at_ms",
                )
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "transition_id": self.transition_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleTransitionIntent":
        if payload.get("schema") not in (None, LIFECYCLE_INTENT_SCHEMA):
            raise ValueError("unsupported lifecycle intent schema")
        values = dict(payload)
        values.pop("schema", None)
        values["expected_effect_ids"] = tuple(
            values.get("expected_effect_ids") or ()
        )
        return cls(**values)


@dataclass(frozen=True)
class LifecycleTransitionReceipt:
    """Post-verification receipt for a complete or bounded partial transition."""

    intent: LifecycleTransitionIntent
    phase: LifecycleSagaPhase
    revision: int
    old_tree: ProcessTreeSnapshot | None
    new_tree: ProcessTreeSnapshot | None
    old_tree_fenced: bool
    health_window_started_at_ms: int
    health_window_completed_at_ms: int
    expected_effect_ids: tuple[str, ...]
    observed_effects: tuple[str, ...]
    compensation: tuple[str, ...]
    failure_code: str
    completed_at_ms: int
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase",
            self.phase
            if isinstance(self.phase, LifecycleSagaPhase)
            else LifecycleSagaPhase(str(self.phase)),
        )
        expected = _canonical_id(self._payload())
        if self.receipt_id and self.receipt_id != expected:
            raise ValueError("receipt_id does not match transition receipt")
        object.__setattr__(self, "receipt_id", expected)

    @property
    def succeeded(self) -> bool:
        return self.phase is LifecycleSagaPhase.COMMITTED

    @property
    def partial(self) -> bool:
        return self.phase is LifecycleSagaPhase.PARTIAL_FAILURE

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_RECEIPT_SCHEMA,
            "intent": self.intent.to_dict(),
            "phase": self.phase.value,
            "revision": self.revision,
            "old_tree": self.old_tree.to_dict() if self.old_tree else None,
            "new_tree": self.new_tree.to_dict() if self.new_tree else None,
            "old_tree_fenced": self.old_tree_fenced,
            "health_window_started_at_ms": self.health_window_started_at_ms,
            "health_window_completed_at_ms": self.health_window_completed_at_ms,
            "expected_effect_ids": list(self.expected_effect_ids),
            "observed_effects": list(self.observed_effects),
            "compensation": list(self.compensation),
            "failure_code": self.failure_code,
            "completed_at_ms": self.completed_at_ms,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleTransitionReceipt":
        if payload.get("schema") not in (None, LIFECYCLE_RECEIPT_SCHEMA):
            raise ValueError("unsupported lifecycle receipt schema")
        return cls(
            intent=LifecycleTransitionIntent.from_dict(payload["intent"]),
            phase=payload.get("phase", ""),
            revision=payload.get("revision", 0),
            old_tree=(
                ProcessTreeSnapshot.from_dict(payload["old_tree"])
                if isinstance(payload.get("old_tree"), Mapping)
                else None
            ),
            new_tree=(
                ProcessTreeSnapshot.from_dict(payload["new_tree"])
                if isinstance(payload.get("new_tree"), Mapping)
                else None
            ),
            old_tree_fenced=payload.get("old_tree_fenced", False),
            health_window_started_at_ms=payload.get(
                "health_window_started_at_ms", 0
            ),
            health_window_completed_at_ms=payload.get(
                "health_window_completed_at_ms", 0
            ),
            expected_effect_ids=tuple(payload.get("expected_effect_ids") or ()),
            observed_effects=tuple(payload.get("observed_effects") or ()),
            compensation=tuple(payload.get("compensation") or ()),
            failure_code=payload.get("failure_code", ""),
            completed_at_ms=payload.get("completed_at_ms", 0),
            receipt_id=payload.get("receipt_id", ""),
        )


@dataclass(frozen=True)
class _SagaState:
    intent: LifecycleTransitionIntent
    phase: LifecycleSagaPhase = LifecycleSagaPhase.PREPARED
    revision: int = 0
    old_tree: ProcessTreeSnapshot | None = None
    new_tree: ProcessTreeSnapshot | None = None
    old_tree_fenced: bool = False
    health_window_started_at_ms: int = 0
    observed_effects: tuple[str, ...] = ()
    compensation: tuple[str, ...] = ()
    failure_code: str = ""
    receipt: LifecycleTransitionReceipt | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LIFECYCLE_SAGA_SCHEMA,
            "intent": self.intent.to_dict(),
            "phase": self.phase.value,
            "revision": self.revision,
            "old_tree": self.old_tree.to_dict() if self.old_tree else None,
            "new_tree": self.new_tree.to_dict() if self.new_tree else None,
            "old_tree_fenced": self.old_tree_fenced,
            "health_window_started_at_ms": self.health_window_started_at_ms,
            "observed_effects": list(self.observed_effects),
            "compensation": list(self.compensation),
            "failure_code": self.failure_code,
            "receipt": self.receipt.to_dict() if self.receipt else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_SagaState":
        if payload.get("schema") != LIFECYCLE_SAGA_SCHEMA:
            raise ValueError("unsupported lifecycle saga schema")
        return cls(
            intent=LifecycleTransitionIntent.from_dict(payload["intent"]),
            phase=LifecycleSagaPhase(payload.get("phase", "")),
            revision=payload.get("revision", 0),
            old_tree=(
                ProcessTreeSnapshot.from_dict(payload["old_tree"])
                if isinstance(payload.get("old_tree"), Mapping)
                else None
            ),
            new_tree=(
                ProcessTreeSnapshot.from_dict(payload["new_tree"])
                if isinstance(payload.get("new_tree"), Mapping)
                else None
            ),
            old_tree_fenced=payload.get("old_tree_fenced", False),
            health_window_started_at_ms=payload.get(
                "health_window_started_at_ms", 0
            ),
            observed_effects=tuple(payload.get("observed_effects") or ()),
            compensation=tuple(payload.get("compensation") or ()),
            failure_code=payload.get("failure_code", ""),
            receipt=(
                LifecycleTransitionReceipt.from_dict(payload["receipt"])
                if isinstance(payload.get("receipt"), Mapping)
                else None
            ),
        )


class LifecycleSagaStore:
    """Fsync'd append-only process-saga journal with a target-wide lock."""

    def __init__(
        self,
        state_root: str | Path,
        *,
        filename: str = "lifecycle-transitions.jsonl",
    ) -> None:
        self.state_root = Path(_absolute(state_root, "state_root"))
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("lifecycle journal filename must be relative")
        self.path = self.state_root / filename
        self.lock_path = self.state_root / ".lifecycle-transition.lock"

    @contextmanager
    def locked(self) -> Iterable[None]:
        self.state_root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def history(self) -> tuple[_SagaState, ...]:
        result: list[_SagaState] = []
        if not self.path.exists():
            return ()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise TransactionConflictError(
                "lifecycle saga journal is unreadable"
            ) from exc
        for line in lines:
            try:
                raw = json.loads(line)
                state = _SagaState.from_dict(raw)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise TransactionConflictError(
                    "lifecycle saga journal contains an invalid record"
                ) from exc
            result.append(state)
        return tuple(result)

    def latest(self) -> dict[str, _SagaState]:
        result: dict[str, _SagaState] = {}
        for state in self.history():
            current = result.get(state.intent.target_id)
            if current is None or state.revision > current.revision:
                result[state.intent.target_id] = state
            elif state.revision == current.revision and state != current:
                raise TransactionConflictError(
                    "lifecycle saga has divergent target revisions"
                )
        return result

    def find_idempotency(
        self, *, target_id: str, idempotency_key: str
    ) -> _SagaState | None:
        matching = [
            state
            for state in self.history()
            if state.intent.target_id == target_id
            and state.intent.idempotency_key == idempotency_key
        ]
        return max(matching, key=lambda item: item.revision) if matching else None

    def append(self, state: _SagaState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            state.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())


class LifecycleOrchestrator:
    """Bounded start/stop/restart handler for ``SupervisorControlService``."""

    def __init__(
        self,
        *,
        state_root: str | Path,
        profiles: Mapping[str, LifecycleProfile] | Sequence[LifecycleProfile],
        process_adapter: ProcessAdapter | None = None,
        clock_ms: Callable[[], int] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval_ms: int = 50,
        stop_grace_ms: int = 5_000,
    ) -> None:
        self.store = LifecycleSagaStore(state_root)
        values = (
            tuple(profiles.values())
            if isinstance(profiles, Mapping)
            else tuple(profiles)
        )
        self._profiles = {item.configuration_root: item for item in values}
        if len(self._profiles) != len(values):
            raise ValueError("configuration_root values must be unique")
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._monotonic = monotonic
        self._sleep = sleep
        self._poll_interval_ms = _positive_int(
            poll_interval_ms, "poll_interval_ms"
        )
        self._stop_grace_ms = _positive_int(
            stop_grace_ms, "stop_grace_ms", allow_zero=True
        )
        self._process = process_adapter or LinuxProcessAdapter(
            clock_ms=self._clock_ms
        )

    def _profile(self, request: OperationRequest) -> LifecycleProfile:
        configuration_root = str(
            request.parameters.get("configuration_root") or ""
        ).strip()
        profile = self._profiles.get(configuration_root)
        if profile is None:
            raise LifecycleProfileChanged(
                "configuration_root is not a registered validated profile"
            )
        bindings = {
            "target_id": str(
                request.parameters.get("target_id") or profile.target_id
            ),
            "run_id": str(request.parameters.get("run_id") or profile.run_id),
            "repository_root": request.repository_root,
            "state_root": request.state_root,
        }
        for name, actual in bindings.items():
            if getattr(profile, name) != actual:
                raise LifecycleProfileChanged(
                    f"profile {name} does not match the request"
                )
        if Path(request.state_root).resolve() != self.store.state_root:
            raise LifecycleProfileChanged(
                "orchestrator state root does not match the request"
            )
        return profile

    def _intent(
        self,
        request: OperationRequest,
        profile: LifecycleProfile,
        action: LifecycleAction,
    ) -> LifecycleTransitionIntent:
        if request.dry_run:
            raise LifecycleOrchestrationError(
                "process orchestrator cannot execute a dry-run"
            )
        if request.authorization is None:
            raise LifecycleOrchestrationError(
                "lifecycle request has no bound authorization decision"
            )
        authorization = request.authorization
        now_ms = self._clock_ms()
        if authorization.evaluated_at_ms > now_ms:
            raise LifecycleOrchestrationError(
                "authorization decision is from the future"
            )
        if (
            authorization.expires_at_ms is not None
            and now_ms >= authorization.expires_at_ms
        ):
            raise StaleLeaseError("lifecycle authorization has expired")
        deadline_ms = int(
            request.parameters.get("deadline_ms") or request.bounds.timeout_ms
        )
        health_window_ms = int(
            request.parameters.get("health_window_ms") or 0
        )
        if action in {LifecycleAction.START, LifecycleAction.RESTART}:
            _positive_int(health_window_ms, "health_window_ms")
        _positive_int(deadline_ms, "deadline_ms")
        deadline_ms = min(deadline_ms, request.bounds.timeout_ms)
        return LifecycleTransitionIntent(
            action=action,
            request_id=request.request_id,
            target_id=profile.target_id,
            repository_root=request.repository_root,
            state_root=request.state_root,
            run_root=profile.run_root,
            run_id=profile.run_id,
            configuration_root=profile.configuration_root,
            profile_id=profile.profile_id,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_id=request.objective_id,
            objective_revision=request.objective_revision,
            policy_id=request.policy_id,
            policy_revision=request.policy_revision,
            caller=request.caller,
            authorization_decision_id=authorization.content_id,
            idempotency_key=request.idempotency_key,
            lease_id=request.lease_id,
            fencing_epoch=request.fencing_epoch
            if request.fencing_epoch is not None
            else -1,
            expected_revision=int(
                request.parameters.get("expected_revision") or 0
            ),
            deadline_ms=deadline_ms,
            health_window_ms=health_window_ms,
            expected_effect_ids=tuple(
                item.effect_id for item in request.expected_effects
            ),
            created_at_ms=self._clock_ms(),
        )

    def _reserve(self, intent: LifecycleTransitionIntent) -> _SagaState:
        with self.store.locked():
            replay = self.store.find_idempotency(
                target_id=intent.target_id,
                idempotency_key=intent.idempotency_key,
            )
            if replay is not None:
                if replay.intent.request_id != intent.request_id:
                    raise TransactionConflictError(
                        "idempotency key is bound to a changed lifecycle request"
                    )
                return replay
            latest = self.store.latest().get(intent.target_id)
            if latest is not None:
                if not latest.phase.terminal:
                    raise TransactionConflictError(
                        "another lifecycle transition is still active"
                    )
                current_lifecycle_revision = (
                    latest.receipt.revision
                    if latest.receipt is not None
                    else latest.intent.expected_revision
                )
                if current_lifecycle_revision != intent.expected_revision:
                    raise TransactionConflictError(
                        f"stale lifecycle revision {intent.expected_revision}; "
                        f"current revision is {current_lifecycle_revision}"
                    )
                if intent.fencing_epoch < latest.intent.fencing_epoch:
                    raise StaleLeaseError("lifecycle fencing epoch is stale")
                if (
                    intent.fencing_epoch == latest.intent.fencing_epoch
                    and intent.lease_id != latest.intent.lease_id
                ):
                    raise StaleLeaseError(
                        "fencing epoch is already owned by another lease"
                    )
                if (
                    intent.action is LifecycleAction.RESTART
                    and latest.intent.profile_id != intent.profile_id
                ):
                    raise LifecycleProfileChanged(
                        "restart configuration changed since the prior transition"
                    )
            elif intent.expected_revision != 0:
                raise TransactionConflictError(
                    f"stale lifecycle revision {intent.expected_revision}; "
                    "current revision is 0"
                )
            state = _SagaState(
                intent=intent,
                # Journal/CAS revisions advance for every checkpoint.  The
                # public lifecycle revision in the receipt advances once per
                # complete operator transition.
                revision=0 if latest is None else latest.revision + 1,
            )
            # The PREPARED intent is fsync'd before _run performs any process
            # discovery that can lead to a signal or launch.
            self.store.append(state)
            return state

    def _advance(
        self,
        state: _SagaState,
        phase: LifecycleSagaPhase,
        **changes: Any,
    ) -> _SagaState:
        with self.store.locked():
            latest = self.store.latest().get(state.intent.target_id)
            if latest is None or latest.intent.transition_id != state.intent.transition_id:
                raise TransactionConflictError("lifecycle transition reservation was lost")
            if latest.revision != state.revision:
                raise TransactionConflictError("lifecycle transition revision is stale")
            updated = replace(
                latest,
                phase=phase,
                revision=latest.revision + 1,
                **changes,
            )
            self.store.append(updated)
            return updated

    def _remaining_ms(self, deadline: float) -> int:
        return max(0, int((deadline - self._monotonic()) * 1000))

    def _assert_single_tree(
        self, tree: ProcessTreeSnapshot, *, allow_empty: bool
    ) -> None:
        if not tree.members:
            if allow_empty:
                return
            raise LifecycleOrchestrationError(
                "no exact process tree exists for the requested run/profile"
            )
        if len(tree.roots) != 1:
            raise SplitBrainError(
                "multiple process roots exist for the requested run/profile"
            )

    def _prove_absent(
        self,
        profile: LifecycleProfile,
        old_tree: ProcessTreeSnapshot,
    ) -> tuple[bool, ProcessTreeSnapshot]:
        exact_alive = tuple(
            member
            for member in old_tree.members
            if self._process.identity_alive(member)
        )
        observed = self._process.snapshot(profile)
        return not exact_alive and not observed.members, observed

    def _stop_old(
        self,
        state: _SagaState,
        profile: LifecycleProfile,
        deadline: float,
        *,
        require_running: bool,
    ) -> _SagaState:
        tree = state.old_tree or self._process.snapshot(profile)
        self._assert_single_tree(tree, allow_empty=not require_running)
        if any(
            member.fencing_epoch > state.intent.fencing_epoch
            for member in tree.members
        ):
            raise StaleLeaseError(
                "request fence is older than the observed process tree"
            )
        state = self._advance(
            state,
            LifecycleSagaPhase.STOPPING_OLD,
            old_tree=tree,
        )
        if tree.members:
            self._process.terminate(
                tree,
                grace_seconds=min(
                    self._stop_grace_ms,
                    self._remaining_ms(deadline),
                )
                / 1000.0,
                deadline_ms=self._remaining_ms(deadline),
            )
        while self._remaining_ms(deadline) > 0:
            absent, observed = self._prove_absent(profile, tree)
            if absent:
                stop_observation = (
                    "old_process_tree_terminated"
                    if tree.members
                    else "old_process_tree_already_absent"
                )
                return self._advance(
                    state,
                    LifecycleSagaPhase.OLD_FENCED,
                    old_tree_fenced=True,
                    observed_effects=tuple(
                        sorted(
                            set(state.observed_effects)
                            | {stop_observation, "run_fenced"}
                        )
                    ),
                )
            del observed
            self._sleep(min(self._poll_interval_ms, self._remaining_ms(deadline)) / 1000)
        alive = tuple(
            member.identity_id
            for member in tree.members
            if self._process.identity_alive(member)
        )
        state = self._advance(
            state,
            LifecycleSagaPhase.PARTIAL_FAILURE,
            failure_code="descendants_remain",
            compensation=("repair_or_quarantine_remaining_process_tree",),
            observed_effects=tuple(
                sorted(set(state.observed_effects) | set(alive))
            ),
        )
        raise ProcessTreeNotFenced(
            "shutdown left exact descendants alive",
            applied_effect_ids=state.intent.expected_effect_ids,
            recovery="repair",
        )

    def _start_new(
        self,
        state: _SagaState,
        profile: LifecycleProfile,
        deadline: float,
    ) -> _SagaState:
        current = self._process.snapshot(profile)
        if current.members:
            self._assert_single_tree(current, allow_empty=False)
            if state.new_tree is None:
                raise SplitBrainError(
                    "a process tree appeared before the authorized launch"
                )
            expected_ids = {item.identity_id for item in state.new_tree.members}
            if not expected_ids.issubset(
                {item.identity_id for item in current.members}
            ):
                raise ProcessIdentityMismatch(
                    "resumed startup observes a different process tree"
                )
        elif (
            state.phase is LifecycleSagaPhase.PARTIAL_FAILURE
            and state.new_tree is not None
        ):
            # Prior health compensation proved the unhealthy tree absent.
            # Retain the old-tree fence checkpoint but create a fresh identity.
            state = self._advance(
                state,
                LifecycleSagaPhase.PARTIAL_FAILURE,
                new_tree=None,
                health_window_started_at_ms=0,
            )
        if state.new_tree is None:
            state = self._advance(state, LifecycleSagaPhase.STARTING_NEW)
            try:
                launched = self._process.launch(
                    profile, fencing_epoch=state.intent.fencing_epoch
                )
            except Exception as exc:
                state = self._advance(
                    state,
                    LifecycleSagaPhase.PARTIAL_FAILURE,
                    failure_code="launch_failed",
                    compensation=(
                        (
                            "resume_start_after_verified_old_tree_fence"
                            if state.old_tree_fenced
                            else "retry_start_without_prior_process_effect"
                        ),
                    ),
                )
                raise PartialMutationError(
                    f"process launch failed: {exc}",
                    applied_effect_ids=(
                        state.intent.expected_effect_ids
                        if state.old_tree_fenced
                        else ()
                    ),
                    recovery="repair",
                ) from exc
            observed = self._process.snapshot(profile)
            if not any(
                item.identity_id == launched.identity_id
                for item in observed.members
            ):
                # A child which only forked and exited is never startup.
                observed = ProcessTreeSnapshot(
                    profile_id=profile.profile_id,
                    run_id=profile.run_id,
                    members=(launched,)
                    if self._process.identity_alive(launched)
                    else (),
                    captured_at_ms=self._clock_ms(),
                )
            self._assert_single_tree(observed, allow_empty=False)
            state = self._advance(
                state,
                LifecycleSagaPhase.VERIFYING_HEALTH,
                new_tree=observed,
                health_window_started_at_ms=0,
                observed_effects=tuple(
                    sorted(set(state.observed_effects) | {"new_process_launched"})
                ),
            )

        healthy_since = state.health_window_started_at_ms
        while self._remaining_ms(deadline) > 0:
            observed = self._process.snapshot(profile)
            try:
                self._assert_single_tree(observed, allow_empty=False)
            except LifecycleOrchestrationError:
                if healthy_since:
                    state = self._advance(
                        state,
                        LifecycleSagaPhase.VERIFYING_HEALTH,
                        health_window_started_at_ms=0,
                    )
                healthy_since = 0
            else:
                now_ms = self._clock_ms()
                if self._process.healthy(
                    profile,
                    observed,
                    fencing_epoch=state.intent.fencing_epoch,
                    now_ms=now_ms,
                ):
                    if not healthy_since:
                        healthy_since = now_ms
                        state = self._advance(
                            state,
                            LifecycleSagaPhase.VERIFYING_HEALTH,
                            new_tree=observed,
                            health_window_started_at_ms=healthy_since,
                        )
                    if now_ms - healthy_since >= state.intent.health_window_ms:
                        return replace(state, new_tree=observed)
                else:
                    if healthy_since:
                        state = self._advance(
                            state,
                            LifecycleSagaPhase.VERIFYING_HEALTH,
                            health_window_started_at_ms=0,
                        )
                    healthy_since = 0
            self._sleep(min(self._poll_interval_ms, self._remaining_ms(deadline)) / 1000)

        compensation: list[str] = ["terminate_unhealthy_new_process_tree"]
        current = self._process.snapshot(profile)
        if current.members:
            self._process.terminate(
                current,
                grace_seconds=min(
                    self._stop_grace_ms,
                    self._remaining_ms(deadline),
                )
                / 1000.0,
                deadline_ms=self._remaining_ms(deadline),
            )
        absent, _observed = self._prove_absent(profile, current)
        compensation.append(
            "unhealthy_new_process_tree_terminated"
            if absent
            else "unhealthy_new_process_tree_remains"
        )
        state = self._advance(
            state,
            LifecycleSagaPhase.PARTIAL_FAILURE,
            failure_code="sustained_health_not_proved",
            compensation=tuple(compensation),
        )
        raise PartialMutationError(
            "startup did not prove sustained health before its deadline",
            applied_effect_ids=(
                state.intent.expected_effect_ids if state.old_tree_fenced else ()
            ),
            recovery="repair",
        )

    def _commit(self, state: _SagaState) -> LifecycleTransitionReceipt:
        now_ms = self._clock_ms()
        observed = set(state.observed_effects)
        if state.intent.action in {
            LifecycleAction.START,
            LifecycleAction.RESTART,
        }:
            observed.add("sustained_health_verified")
        receipt = LifecycleTransitionReceipt(
            intent=state.intent,
            phase=LifecycleSagaPhase.COMMITTED,
            revision=state.intent.expected_revision + 1,
            old_tree=state.old_tree,
            new_tree=state.new_tree,
            old_tree_fenced=state.old_tree_fenced,
            health_window_started_at_ms=state.health_window_started_at_ms,
            health_window_completed_at_ms=(
                now_ms
                if state.intent.action
                in {LifecycleAction.START, LifecycleAction.RESTART}
                else 0
            ),
            expected_effect_ids=state.intent.expected_effect_ids,
            observed_effects=tuple(sorted(observed)),
            compensation=state.compensation,
            failure_code="",
            completed_at_ms=now_ms,
        )
        committed = self._advance(
            state,
            LifecycleSagaPhase.COMMITTED,
            observed_effects=receipt.observed_effects,
            receipt=receipt,
        )
        if committed.receipt is None:
            raise TransactionConflictError("committed lifecycle receipt is absent")
        return committed.receipt

    def execute(self, request: OperationRequest) -> LifecycleTransitionReceipt:
        action = LifecycleAction(request.operation.value)
        profile = self._profile(request)
        intent = self._intent(request, profile, action)
        state = self._reserve(intent)
        if state.receipt is not None:
            return state.receipt
        deadline = self._monotonic() + intent.deadline_ms / 1000.0

        if action is LifecycleAction.START:
            if state.phase is LifecycleSagaPhase.PREPARED:
                current = self._process.snapshot(profile)
                if current.members:
                    self._assert_single_tree(current, allow_empty=False)
                    raise SplitBrainError(
                        "start rejected because an exact run tree already exists"
                    )
            state = self._start_new(state, profile, deadline)
        elif action is LifecycleAction.STOP:
            if not state.old_tree_fenced:
                state = self._stop_old(
                    state, profile, deadline, require_running=False
                )
        else:
            if not state.old_tree_fenced:
                state = self._stop_old(
                    state, profile, deadline, require_running=True
                )
            # A restart always revalidates absence immediately before launch.
            old_tree = state.old_tree or self._process.snapshot(profile)
            absent, _observed = self._prove_absent(profile, old_tree)
            if not absent:
                raise ProcessTreeNotFenced(
                    "old process tree was not fenced before restart launch",
                    applied_effect_ids=state.intent.expected_effect_ids,
                    recovery="repair",
                )
            state = self._start_new(state, profile, deadline)
        return self._commit(state)

    def start(self, request: OperationRequest) -> LifecycleTransitionReceipt:
        if request.operation is not Operation.START:
            raise ValueError("start requires Operation.START")
        return self.execute(request)

    def stop(self, request: OperationRequest) -> LifecycleTransitionReceipt:
        if request.operation is not Operation.STOP:
            raise ValueError("stop requires Operation.STOP")
        return self.execute(request)

    def restart(self, request: OperationRequest) -> LifecycleTransitionReceipt:
        if request.operation is not Operation.RESTART:
            raise ValueError("restart requires Operation.RESTART")
        return self.execute(request)

    def __call__(self, request: OperationRequest) -> BackendResponse:
        receipt = self.execute(request)
        checks = [
            "authorization_bound_by_control_transaction",
            "configuration_profile_unchanged",
            "exact_process_tree_resolved",
            "lease_and_fence_bound",
            "post_effect_receipt_persisted",
        ]
        if receipt.intent.action in {
            LifecycleAction.STOP,
            LifecycleAction.RESTART,
        }:
            checks.append("old_tree_fenced")
        if receipt.intent.action in {
            LifecycleAction.START,
            LifecycleAction.RESTART,
        }:
            checks.append("sustained_health_verified")
        return BackendResponse(
            data={
                "transition": receipt.to_dict(),
                "transition_id": receipt.intent.transition_id,
                "receipt_id": receipt.receipt_id,
                "old_tree_fenced": receipt.old_tree_fenced,
                "new_process_identity": (
                    receipt.new_tree.roots[0].to_dict()
                    if receipt.new_tree and len(receipt.new_tree.roots) == 1
                    else None
                ),
                "health_window_ms": receipt.intent.health_window_ms,
            },
            changed=True,
            applied_effect_ids=receipt.expected_effect_ids,
            checks=tuple(checks),
        )


FencedLifecycleOrchestrator = LifecycleOrchestrator


def resolve_exact_process_tree(
    profile: LifecycleProfile,
    *,
    adapter: ProcessAdapter | None = None,
) -> ProcessTreeSnapshot:
    """Public read-only helper used by runner/watchdog diagnostics."""

    return (adapter or LinuxProcessAdapter()).snapshot(profile)


__all__ = [
    "CONFIGURATION_ROOT_ENV",
    "FENCING_EPOCH_ENV",
    "FencedLifecycleOrchestrator",
    "LifecycleAction",
    "LifecycleOrchestrationError",
    "LifecycleOrchestrator",
    "LifecycleProfile",
    "LifecycleProfileChanged",
    "LifecycleSagaPhase",
    "LifecycleSagaStore",
    "LifecycleTransitionIntent",
    "LifecycleTransitionReceipt",
    "LinuxProcessAdapter",
    "ProcessAdapter",
    "ProcessIdentity",
    "ProcessIdentityMismatch",
    "ProcessTreeNotFenced",
    "ProcessTreeSnapshot",
    "RUN_ID_ENV",
    "SplitBrainError",
    "ValidatedLifecycleProfile",
    "resolve_exact_process_tree",
]
