"""Bounded recovery of one registered, leased foreground systemd service.

Observation is body-free and never grants completion. Recovery restarts only
the pinned user unit; its existing checkout lease and cgroup own replacement
and cleanup. Heartbeats, growing logs and native process activity cannot renew
a declared task deadline or replenish the persisted recovery budget.
"""

from __future__ import annotations

import argparse
import errno
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .watchdog_guardian import acquire_lock, atomic_json
from . import semantic_recovery
from .maintenance_hold import review_hold
from .storage_pressure import (
    check_storage_capacity, clear_emergency_status, is_storage_pressure,
    publish_emergency_status, read_emergency_status,
)

SCHEMA = "ipfs_accelerate_py/leased-service-watchdog@1"
STOP_NAMES = ("HOLD", "OPERATOR_STOP", "watchdog.disabled", "hold", "operator.hold")
DONE = frozenset({"completed", "complete", "done"})
READY = frozenset({"ready", "pending", "in_progress"})
MAX_FILE_BYTES = 16 * 1024 * 1024


class ObservationError(ValueError):
    """Ownership or configuration is not sufficiently known for recovery."""


def _json(path: Path, *, missing: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ObservationError("JSON input exceeds observation bound")
        value = json.loads(path.read_text())
    except FileNotFoundError:
        if missing:
            return {}
        raise ObservationError("required observation input is absent") from None
    if not isinstance(value, dict):
        raise ObservationError("JSON input must be an object")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    try:
        if not path.is_file():
            return "absent"
        if path.stat().st_size > MAX_FILE_BYTES:
            return "oversized"
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unreadable"


def _path(root: Path, raw: str) -> Path:
    path = Path(raw)
    path = path if path.is_absolute() else root / path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ObservationError("campaign path escapes the accepted repository") from exc
    return path.resolve()


@dataclass(frozen=True)
class Config:
    unit: str
    repo_root: Path
    lane_id: str
    board_path: Path
    task_prefix: str
    status_path: Path
    lease_owner_path: Path
    inventory_path: Path
    state_dir: Path
    unit_fragment_path: Path
    unit_fragment_sha256: str
    expected_argv: tuple[str, ...]
    task_spec_path: Path | None = None
    owned_hashes_path: Path | None = None
    log_path: Path | None = None
    attempt_path: Path | None = None
    nested_repositories: tuple[str, ...] = ()
    repair_paths: tuple[str, ...] = ()
    progress_globs: tuple[str, ...] = ()
    hold_paths: tuple[str, ...] = ()
    native_executables: tuple[str, ...] = ("grok*", "codex*", "claude*")
    expected_environment: Mapping[str, str] = field(default_factory=lambda: {"IPFS_ACCELERATE_AGENT_MAX_DIRTY_ATTEMPTS": "0"})
    lease_namespace: str = "swissknife-supervisor-checkout-lease-v1"
    poll_seconds: float = 30
    startup_grace_seconds: float = 180
    deadline_grace_seconds: float = 120
    heartbeat_stale_seconds: float = 120
    idle_stall_seconds: float = 300
    max_backoff_seconds: float = 1200
    default_task_timeout_seconds: float = 1800
    max_task_timeout_seconds: float = 28800
    restart_budget: int = 3
    restart_cooldown_seconds: float = 300
    restart_timeout_seconds: float = 180
    failure_loop_attempts: int = 3
    semantic_stall_seconds: float = 1800

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Config":
        fields = set(cls.__dataclass_fields__)
        if set(value) - fields:
            raise ObservationError("unknown watchdog configuration fields")
        data = dict(value)
        root = Path(data["repo_root"]).resolve(strict=True)
        data["repo_root"] = root
        for name in ("board_path", "status_path", "inventory_path", "state_dir", "task_spec_path", "owned_hashes_path", "log_path", "attempt_path"):
            if data.get(name) is not None:
                data[name] = _path(root, data[name])
        for name in ("lease_owner_path", "unit_fragment_path"):
            data[name] = Path(data[name]).resolve()
        for name in ("expected_argv", "nested_repositories", "repair_paths", "progress_globs", "hold_paths", "native_executables"):
            if name in data:
                if not isinstance(data[name], list) or any(type(item) is not str or not item for item in data[name]):
                    raise ObservationError(f"{name} must be an array of nonempty strings")
                data[name] = tuple(data[name])
        config = cls(**data)
        config.validate()
        return config

    def validate(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", self.unit):
            raise ObservationError("one exact systemd service name is required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.unit_fragment_sha256):
            raise ObservationError("unit fragment SHA-256 is required")
        if not self.expected_argv or not Path(self.expected_argv[0]).is_absolute():
            raise ObservationError("exact executable argv with an absolute executable is required")
        for name in ("poll_seconds", "startup_grace_seconds", "deadline_grace_seconds", "heartbeat_stale_seconds",
                     "idle_stall_seconds", "max_backoff_seconds", "default_task_timeout_seconds",
                     "max_task_timeout_seconds", "restart_cooldown_seconds", "restart_timeout_seconds", "semantic_stall_seconds"):
            item = getattr(self, name)
            if type(item) not in (int, float) or not math.isfinite(item) or item <= 0:
                raise ObservationError(f"{name} must be finite and positive")
        if type(self.restart_budget) is not int or not 1 <= self.restart_budget <= 100:
            raise ObservationError("restart_budget must be a bounded positive integer")
        if type(self.failure_loop_attempts) is not int or not 2 <= self.failure_loop_attempts <= 100:
            raise ObservationError("failure_loop_attempts must be a bounded integer")
        if self.default_task_timeout_seconds > self.max_task_timeout_seconds:
            raise ObservationError("task timeout bounds are inconsistent")
        if self.state_dir == self.repo_root or any(self.state_dir == p or self.state_dir in p.parents for p in (
                self.board_path, self.status_path, self.inventory_path)):
            raise ObservationError("watchdog telemetry must have its own separate directory")
        if not self.task_prefix.startswith("## ") or not self.lane_id:
            raise ObservationError("task prefix and lane identity are required")
        for raw in (*self.nested_repositories, *self.repair_paths, *self.progress_globs, *self.hold_paths):
            _path(self.repo_root, raw.replace("{task_id}", "TASK-001"))
        if not isinstance(self.expected_environment, Mapping) or any(
                type(key) is not str or type(item) is not str for key, item in self.expected_environment.items()):
            raise ObservationError("expected environment must contain string bindings")


def parse_board(text: str, prefix: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = None
            if line.startswith(prefix):
                identity = line[3:].split(None, 1)[0]
                if identity in result:
                    raise ObservationError("duplicate task identity")
                current = {}
                result[identity] = current
        elif current is not None and line.startswith("- ") and ":" in line:
            key, item = line[2:].split(":", 1)
            current[re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")] = item.strip()
    if not result:
        raise ObservationError("taskboard is empty")
    seen, visiting = set(), set()

    def visit(identity: str) -> None:
        if identity in visiting:
            raise ObservationError("task dependency cycle")
        if identity in seen:
            return
        visiting.add(identity)
        for dependency in dependencies(result[identity]):
            if dependency not in result:
                raise ObservationError("task dependency is missing")
            visit(dependency)
        visiting.remove(identity)
        seen.add(identity)

    for identity in result:
        if result[identity].get("status", "").lower() not in DONE | READY | {"blocked", "failed"}:
            raise ObservationError("unsupported task status")
        visit(identity)
    return result


def dependencies(fields: Mapping[str, str]) -> list[str]:
    return [part for part in re.split(r"[,\s]+", fields.get("depends_on", "")) if part and part.lower() not in {"none", "n/a", "-"}]


class Host:
    """Read bounded OS metadata; the only effect is restart of the pinned unit."""

    def _run(self, argv: list[str], timeout: float = 30) -> str:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode:
            raise ObservationError("system inspection or unit action failed")
        return result.stdout

    def unit(self, config: Config) -> dict[str, Any]:
        properties = ("Id", "ActiveState", "SubState", "MainPID", "FragmentPath", "DropInPaths",
                      "NeedDaemonReload", "WorkingDirectory", "KillMode", "ControlGroup")
        raw = self._run(["systemctl", "--user", "show", config.unit, *("--property=" + name for name in properties)])
        unit = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        encoded = "".join(char if char.isascii() and char.isalnum() else "_" + format(ord(char), "02x") for char in config.unit)
        value = json.loads(self._run(["busctl", "--user", "--json=short", "get-property", "org.freedesktop.systemd1",
                                     "/org/freedesktop/systemd1/unit/" + encoded, "org.freedesktop.systemd1.Service", "ExecStart"]))
        rows = value.get("data", [])
        if len(rows) != 1 or not isinstance(rows[0], list) or len(rows[0]) < 2:
            raise ObservationError("loaded ExecStart identity is ambiguous")
        unit["argv"] = rows[0][1]
        return unit

    def process(self, pid: int, *, environment: bool = False) -> dict[str, Any] | None:
        directory = Path("/proc") / str(pid)
        try:
            before = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if before[0] in {"Z", "X"}:
                return None
            uid = directory.stat().st_uid
            argv = (directory / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
            cgroups = [line.split(":", 2)[2] for line in (directory / "cgroup").read_text().splitlines() if ":" in line]
            current = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
            if before[19] != current[19] or current[0] in {"Z", "X"}:
                raise ObservationError("process identity changed during inspection")
            ticks = os.sysconf("SC_CLK_TCK")
            boot = time.time() - float(Path("/proc/uptime").read_text().split()[0])
            return {"pid": pid, "ppid": int(current[1]), "start_ticks": current[19],
                    "cpu_ticks": int(current[11]) + int(current[12]), "uid": uid, "argv": argv,
                    "cwd": str((directory / "cwd").resolve(strict=True)), "cgroups": cgroups,
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "pid_namespace": os.readlink(directory / "ns/pid"),
                    "start_wall": boot + int(current[19]) / ticks,
                    "environment": dict(part.split("=", 1) for part in (directory / "environ").read_bytes().decode(errors="replace").split("\0") if "=" in part) if environment else {}}
        except (FileNotFoundError, ProcessLookupError):
            return None

    def members(self, cgroup: str) -> list[int]:
        if not cgroup.startswith("/") or ".." in Path(cgroup).parts:
            raise ObservationError("unit cgroup is invalid")
        directory = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        if not directory.exists():
            return []
        paths = [directory / "cgroup.procs", *directory.glob("*/cgroup.procs")]
        if len(paths) > 128:
            raise ObservationError("unit cgroup observation exceeds bound")
        pids = {int(raw) for path in paths for raw in path.read_text().split()}
        if len(pids) > 1024:
            raise ObservationError("unit process population exceeds bound")
        return sorted(pids)

    def exact_processes(self, config: Config, foreground_argv: tuple[str, ...]) -> list[int]:
        found = []
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                if directory.stat().st_uid != os.getuid():
                    continue
                argv = (directory / "cmdline").read_bytes().decode(errors="replace").rstrip("\0").split("\0")
                if tuple(argv) not in (config.expected_argv, foreground_argv):
                    continue
                process = self.process(int(directory.name))
                if process and process["cwd"] == str(config.repo_root):
                    found.append(process["pid"])
            except (FileNotFoundError, ProcessLookupError):
                continue
        return found

    def dirty_paths(self, config: Config) -> list[str]:
        paths = set()
        nested = set(config.nested_repositories)
        for relative in ("", *config.nested_repositories):
            root = config.repo_root / relative
            for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_LOG", "rebase-merge", "rebase-apply"):
                raw = self._run(["git", "-C", str(root), "rev-parse", "--git-path", marker]).strip()
                if (Path(raw) if Path(raw).is_absolute() else root / raw).exists():
                    raise ObservationError("repository has an interrupted Git operation")
            for argv in (("diff", "--no-renames", "--name-only", "HEAD", "-z"),
                         ("ls-files", "--others", "--exclude-standard", "-z")):
                for name in self._run(["git", "-C", str(root), *argv]).split("\0"):
                    if name and not (not relative and name in nested):
                        paths.add((relative + "/" if relative else "") + name)
        return sorted(paths)

    def restart(self, config: Config) -> None:
        self._run(["systemctl", "--user", "restart", config.unit], config.restart_timeout_seconds)


def _identity(process: dict[str, Any] | None, expected: Mapping[str, Any]) -> bool:
    return bool(process and expected.get("method") == "linux-procfs-starttime-v1"
                and str(expected.get("startTicks")) == process["start_ticks"]
                and expected.get("bootId") == process["boot_id"]
                and expected.get("pidNamespace") == process["pid_namespace"]
                and expected.get("uid") == process["uid"] == os.getuid())


def _in_cgroup(process: Mapping[str, Any], cgroup: str) -> bool:
    return bool(cgroup and any(item == cgroup or item.startswith(cgroup + "/") for item in process["cgroups"]))


def _old_owner_absent(process: Mapping[str, Any] | None, identity: Mapping[str, Any]) -> bool:
    if (identity.get("method") != "linux-procfs-starttime-v1"
            or not all(isinstance(identity.get(key), str) and identity[key] for key in ("startTicks", "bootId", "pidNamespace"))):
        raise ObservationError("recorded process birth is incomplete")
    if process is None:
        return True
    if process["pid_namespace"] != identity["pidNamespace"]:
        raise ObservationError("recorded owner PID namespace is unverifiable")
    return process["boot_id"] != identity["bootId"] or process["start_ticks"] != identity["startTicks"]


def _descends(host: Host, process: Mapping[str, Any], ancestor: Mapping[str, Any]) -> bool:
    current = dict(process)
    seen: set[int] = set()
    for _ in range(128):
        if current["pid"] == ancestor["pid"]:
            return current["start_ticks"] == ancestor["start_ticks"]
        if current["pid"] in seen or current["uid"] != os.getuid():
            return False
        seen.add(current["pid"])
        parent = host.process(current["ppid"])
        again = host.process(current["pid"])
        if (not parent or not again or again["start_ticks"] != current["start_ticks"]
                or again["ppid"] != current["ppid"]):
            return False
        current = parent
    return False


class LeasedServiceWatchdog:
    def __init__(self, config: Config, *, host: Host | None = None) -> None:
        config.validate()
        self.config, self.host = config, host or Host()
        self.state_path, self.status_path = config.state_dir / "state.json", config.state_dir / "status.json"
        self.state = _json(self.state_path, missing=True)

    def _capacity(self) -> dict[str, Any]:
        c = self.config
        return check_storage_capacity([c.repo_root, c.board_path, c.status_path, self.state_path, self.status_path])

    def _storage_wait(self, result: dict[str, Any], reason: str, error: BaseException | None = None) -> dict[str, Any]:
        original = result.get("action", "observe")
        result.update(storage_pressure=True, durable_persistence=False, health="degraded",
                      storage_original_action=original, storage_original_phase=result.get("phase", ""),
                      phase="storage_pressure")
        if original != "operator_hold":
            result.update(action="storage_wait", reason=reason)
        publish_emergency_status(self.status_path, result, error=error)
        return result

    def _registration(self) -> tuple[str, ...]:
        c = self.config
        inventory = _json(c.inventory_path)
        lanes = [lane for lane in inventory.get("lanes", []) if lane.get("id") == c.lane_id]
        if len(lanes) != 1:
            raise ObservationError("lane registration is absent or ambiguous")
        lane = lanes[0]
        command = lane.get("launch", {}).get("command", [])
        if (lane.get("board") != c.board_path.relative_to(c.repo_root).as_posix()
                or lane.get("taskPrefix") != c.task_prefix or lane.get("leaseRequiredForImplementation") is not True
                or lane.get("launch", {}).get("commandPolicy") != "exact"
                or not command or "--" not in command
                or tuple(command[1:]) != c.expected_argv[1:]):
            raise ObservationError("unit invocation does not match exact registered lane")
        return tuple(command[command.index("--") + 1:])

    def _owners(self) -> dict[str, Any]:
        c, h = self.config, self.host
        foreground_argv = self._registration()
        unit = h.unit(c)
        if (unit.get("Id") != c.unit or unit.get("WorkingDirectory") != str(c.repo_root)
                or unit.get("FragmentPath") != str(c.unit_fragment_path) or unit.get("DropInPaths")
                or unit.get("NeedDaemonReload") != "no" or unit.get("KillMode") != "control-group"
                or _file_hash(c.unit_fragment_path) != c.unit_fragment_sha256
                or tuple(unit.get("argv", [])) != c.expected_argv):
            raise ObservationError("loaded systemd unit differs from the pinned service")
        main_pid = int(unit.get("MainPID") or 0)
        main = h.process(main_pid, environment=True) if main_pid else None
        pointer = _json(c.lease_owner_path, missing=True)
        lease = {}
        if pointer:
            namespace, lane = pointer.get("namespace", {}), pointer.get("lane", {})
            if namespace.get("name") != c.lease_namespace or lane.get("id") != c.lane_id:
                raise ObservationError("foreign or unverifiable checkout lease")
            if (Path(namespace.get("superproject", "")).resolve() != c.repo_root
                    or Path(namespace.get("commonDirectory", "")).resolve() != c.lease_owner_path.parent.parent
                    or lane.get("board") != c.board_path.relative_to(c.repo_root).as_posix()
                    or tuple(pointer.get("command", [])) != foreground_argv):
                raise ObservationError("checkout lease does not bind this registered service")
            candidates = []
            for path in c.lease_owner_path.parent.glob(".owner-state-*.json"):
                candidate = _json(path)
                if candidate.get("leaseId") == pointer.get("leaseId"):
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise ObservationError("checkout lease generation state is absent or ambiguous")
            lease = candidates[0]
            for key in ("schemaVersion", "leaseId", "namespace", "owner", "lane", "command", "acquiredAt"):
                if lease.get(key) != pointer.get(key):
                    raise ObservationError("lease generation is not owned by its pointer")
            if lease.get("owner", {}).get("hostname") != socket.gethostname():
                raise ObservationError("checkout lease belongs to another host")
        child, protected = None, None
        if main:
            if tuple(main["argv"]) != c.expected_argv or main["cwd"] != str(c.repo_root) or not _in_cgroup(main, unit.get("ControlGroup", "")):
                raise ObservationError("systemd main process identity is not the registered owner")
            if any(main["environment"].get(key) != item for key, item in c.expected_environment.items()):
                raise ObservationError("service environment does not preserve the configured ownership policy")
            if not lease or lease.get("owner", {}).get("pid") != main_pid or not _identity(main, lease.get("owner", {}).get("processIdentity", {})):
                raise ObservationError("systemd owner does not hold the verified checkout lease")
            protected_pid = lease.get("childPid")
            protected = h.process(protected_pid) if type(protected_pid) is int and protected_pid > 0 else None
            protected_identity = lease.get("childProcessIdentity", {})
            if protected and (not _identity(protected, protected_identity)
                              or tuple(protected["argv"]) != tuple(protected_identity.get("command", []))
                              or protected["cwd"] != str(c.repo_root) or protected["ppid"] != main_pid
                              or not _in_cgroup(protected, unit["ControlGroup"])):
                raise ObservationError("protected lease child does not match its recorded birth and invocation")
            status_pid = _json(c.status_path, missing=True).get("pid")
            if type(status_pid) is int and status_pid > 0:
                child = h.process(status_pid)
            if not child and protected:
                # A first status commit can itself fail under pressure. Find
                # only one exact, birth-fenced protected descendant so its
                # private storage marker can be read; ordinary observations
                # still require the canonical status/PID join below.
                matches = [process for pid in h.members(unit["ControlGroup"])
                           if (process := h.process(pid)) and tuple(process["argv"]) == foreground_argv
                           and process["cwd"] == str(c.repo_root) and process["uid"] == os.getuid()
                           and _in_cgroup(process, unit["ControlGroup"]) and _descends(h, process, protected)]
                if len(matches) > 1:
                    raise ObservationError("leased foreground without canonical status is ambiguous")
                child = matches[0] if matches else None
            if child and (not protected or tuple(child["argv"]) != foreground_argv
                          or child["cwd"] != str(c.repo_root) or child["uid"] != os.getuid()
                          or not _in_cgroup(child, unit["ControlGroup"]) or not _descends(h, child, protected)):
                raise ObservationError("foreground is not the exact leased invocation and protected descendant")
        else:
            if h.exact_processes(c, foreground_argv):
                raise ObservationError("an unmanaged matching owner still exists")
            if lease:
                old_owner = h.process(lease.get("owner", {}).get("pid", 0))
                old_child = h.process(lease.get("childPid") or 0)
                owner_absent = _old_owner_absent(old_owner, lease.get("owner", {}).get("processIdentity", {}))
                child_absent = _old_owner_absent(old_child, lease.get("childProcessIdentity", {})) if lease.get("childPid") else True
                if not owner_absent or not child_absent:
                    raise ObservationError("lease owner absence cannot be proven")
        fingerprint = _digest({"unit": c.unit, "active": unit.get("ActiveState"), "main_pid": main_pid,
                               "main_birth": main.get("start_ticks") if main else None,
                               "child_pid": child.get("pid") if child else None,
                               "child_birth": child.get("start_ticks") if child else None,
                               "protected_pid": protected.get("pid") if protected else None,
                               "protected_birth": protected.get("start_ticks") if protected else None,
                               "lease": lease.get("leaseId"), "argv": unit["argv"],
                               "fragment": c.unit_fragment_sha256})
        return {"unit": unit, "main": main, "child": child, "protected": protected, "lease": lease, "fingerprint": fingerprint}

    def _holds(self, *, apply: bool = False, now: float | None = None) -> bool:
        c = self.config
        directories = {c.repo_root, c.state_dir, c.status_path.parent}
        for directory in c.status_path.parents:
            if directory == c.repo_root or c.repo_root in directory.parents:
                directories.add(directory)
        paths = {directory / name for directory in directories for name in STOP_NAMES}
        paths.update(_path(c.repo_root, raw) for raw in c.hold_paths)
        self.hold_reviews = []
        held = False
        for path in sorted(paths):
            if not os.path.lexists(path):
                continue
            review = review_hold(path, apply=apply, now=now)
            self.hold_reviews.append(review)
            held = held or os.path.lexists(path)
        return held

    def _scope(self, task_id: str, tasks: Mapping[str, Mapping[str, str]]) -> None:
        c = self.config
        paths = []
        if c.task_spec_path:
            rows = _json(c.task_spec_path).get("tasks", [])
            paths = [item for task in rows if task.get("task_id") == task_id for item in task.get("edit_paths", [])]
        if not paths:
            paths = [part.strip() for part in tasks.get(task_id, {}).get("predicted_files", "").split(",") if part.strip()]
        dirty = self.host.dirty_paths(c)
        owned = _json(c.owned_hashes_path, missing=True) if c.owned_hashes_path else {}
        for raw in paths:
            path = _path(c.repo_root, raw)
            relative = path.relative_to(c.repo_root).as_posix()
            if any(item == relative or item.startswith(relative + "/") or relative.startswith(item + "/") for item in dirty):
                claimed = owned.get(relative)
                if (not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed)
                        or not path.is_file() or claimed != _file_hash(path)):
                    raise ObservationError("declared task output overlaps foreign dirty work")

    def _activity(self, owners: Mapping[str, Any], active: str, task: Mapping[str, str]) -> dict[str, Any]:
        c = self.config
        processes = []
        cgroup = owners["unit"].get("ControlGroup", "")
        if cgroup:
            for pid in self.host.members(cgroup):
                process = self.host.process(pid)
                if process and _in_cgroup(process, cgroup):
                    processes.append(process)
        try:
            declared = shlex.split(task.get("deterministic_command", ""))
        except ValueError:
            declared = []
        workers = [p for p in processes if declared and p["argv"][1:] == declared[1:]
                   and Path(p["argv"][0]).name == Path(declared[0]).name]
        native = [p for p in processes if any(fnmatch.fnmatch(Path(p["argv"][0]).name, pattern) for pattern in c.native_executables)]
        outputs = [part.strip() for part in task.get("predicted_files", "").split(",") if part.strip()]
        artifacts = {raw: _file_hash(_path(c.repo_root, raw)) for raw in outputs[:64]}
        native_progress = []
        for pattern in c.progress_globs:
            relative = _path(c.repo_root, pattern.replace("{task_id}", active or "NO-ACTIVE-TASK")).relative_to(c.repo_root).as_posix()
            matches = list(c.repo_root.glob(relative))
            if len(matches) > 128:
                raise ObservationError("native progress artifact bound exceeded")
            for path in matches:
                if not path.is_file():
                    continue
                try:
                    progress = _json(path)
                except (OSError, ValueError):
                    continue
                if progress.get("task_id") != active or progress.get("pid") not in {p["pid"] for p in processes}:
                    continue
                state = progress.get("state")
                state = state if state in {"starting", "running", "completed", "failed", "timeout", "cancelled"} else "unknown"
                stream = progress.get("stream") or {}
                stream = stream if isinstance(stream, dict) else {}
                calls = stream.get("tool_name_counts") or {}
                calls = calls if isinstance(calls, dict) else {}
                native_progress.append({"state": state, "pid": progress["pid"],
                                        "frames": stream.get("frames") if type(stream.get("frames")) is int else None,
                                        "read_file_calls": calls.get("read_file", 0) if type(calls.get("read_file", 0)) is int else None,
                                        "stream_tainted": stream.get("stream_tainted") is True})
        tail = ""
        if c.log_path and c.log_path.exists():
            with c.log_path.open("rb") as handle:
                handle.seek(max(0, c.log_path.stat().st_size - 32768))
                tail = handle.read(32768).decode(errors="replace")
        # A later healthy task must not inherit another task's old failures.
        # Keep historical codes separately; current incidents use only the
        # latest matching producer section (and its subsequent validation).
        section, task_sections, collecting = [], [], False
        for line in tail.splitlines():
            if line.startswith("$ "):
                try:
                    command = shlex.split(line[2:], comments=True)
                except ValueError:
                    command = []
                if command == declared:
                    section, collecting = [], True
                elif collecting and command != shlex.split(task.get("validation", "")):
                    collecting = False
            elif collecting:
                section.append(line)
                task_sections.append(line)
        patterns = (
            ("provider_timeout", r"timeoutexpired|timed out|provider.{0,40}timeout"),
            ("native_incomplete", r"named.session.{0,60}(?:incomplete|did not complete)|native.{0,30}incomplete|grok cli streaming response is incomplete or invalid"),
            ("provider_quota_or_auth", r"quota.exhaust|authentication.{0,20}fail|payment required"),
            ("task_failure", r"deterministic_command_failed|validation_failed|repair attempt [123]:"),
        )
        failures = [name for name, pattern in patterns if re.search(pattern, "\n".join(section), re.I)]
        task_failures = [name for name, pattern in patterns if re.search(pattern, "\n".join(task_sections), re.I)]
        history = [name for name, pattern in patterns if re.search(pattern, tail, re.I)]
        return {"owned_process_count": len(processes), "native_process_count": len(native),
                "native_activity_signature": _digest({"processes": [(p["pid"], p["start_ticks"], p["cpu_ticks"]) for p in native], "progress": native_progress}),
                "work_signature": _digest(artifacts), "provider_failure_codes": failures,
                "historical_provider_failure_codes": history, "native_progress": native_progress,
                "task_provider_failure_codes": task_failures,
                "worker_started_at": min((p["start_wall"] for p in workers), default=None)}

    def _semantic_failures(self, tasks: Mapping[str, Mapping[str, str]], ready: list[str],
                           active: str, activity: Mapping[str, Any], now: float) -> list[dict[str, Any]]:
        """Project durable task/contract failure history independently of PIDs."""
        from .lane_resume import _command_contract, _unresolved_execution_failure, parse_deterministic_command, semantic_failure_limit

        c = self.config
        ledger = _json(c.attempt_path or c.status_path.with_name("lane_resume_attempts.json"), missing=True)
        records = self.state.setdefault("task_progress", {})
        incidents = []
        for task_id in dict.fromkeys([*ready, *([active] if active else [])]):
            fields = tasks[task_id]
            if fields.get("status", "").lower() in DONE:
                records.pop(task_id, None)
                continue
            command = parse_deterministic_command(fields.get("deterministic_command") or fields.get("deterministic_repair") or "", repo_root=c.repo_root)
            validation = parse_deterministic_command(fields.get("validation") or "", repo_root=c.repo_root)
            if command is None:
                continue
            try:
                contract = _command_contract(fields, command, validation, repo_root=c.repo_root)
            except ValueError:
                incidents.append({"code": "semantic_recovery_task_contract_invalid", "task_id": task_id,
                                  "recovery": "repair_declared_task_policy", "unit_restart_is_not_a_semantic_repair": True})
                continue
            fingerprint = _digest({"task_id": task_id, "contract": contract})
            previous = records.get(task_id) or {}
            same = previous.get("contract") == contract
            raw = ledger.get(task_id, 0)
            raw = raw if type(raw) is int and raw >= 0 else 0
            task_progress = (ledger.get("_validation_fixed_points") or {}).get(task_id) or {}
            progress = task_progress.get("execution_progress") or {}
            qualified = progress.get("contract") == contract
            proof = (ledger.get("_command_evidence") or {}).get(task_id) or {}
            proof_qualified = proof.get("retry_contract", proof.get("contract")) == contract and type(proof.get("exit_code")) is int and proof["exit_code"] not in (0, 75)
            unresolved = _unresolved_execution_failure(task_progress, proof)
            if not progress:
                unresolved = unresolved or previous.get("unresolved_prior_failure") is True or int(previous.get("observed_failures") or 0) > 0
            if not same:
                previous = {"contract": contract, "fingerprint": fingerprint,
                            "first_seen": self.state.get("first_seen", now), "baseline_attempts": raw,
                            "observed_failures": 0, "legacy_unqualified": raw > 0 and not qualified and not proof_qualified}
                # A known contract repair, unlike a PID/clock change, is
                # explicitly eligible for one new bounded evaluation.
                if records.get(task_id):
                    previous.update(first_seen=now, legacy_unqualified=False)
            previous = dict(previous)
            if task_id == active:
                previous["was_active"] = True
            observed = max(int(previous.get("observed_failures") or 0), max(0, raw - int(previous.get("baseline_attempts") or 0)))
            qualified_failures = int(progress.get("semantic_failures") or 0) if qualified else raw if proof_qualified else 0
            failures = max(observed, qualified_failures)
            if qualified and progress.get("state") == "validated":
                failures = 0
            abandoned = int(progress.get("abandoned_attempts") or 0) if qualified else 0
            gates = int(progress.get("gate_refusals") or 0) if qualified else 0
            if qualified:
                previous["legacy_unqualified"] = False
            previous.update(observed_failures=failures, last_attempt_count=raw,
                            qualified=qualified or proof_qualified, abandoned_attempts=abandoned,
                            unresolved_prior_failure=unresolved)
            if qualified:
                previous["execution_started_at_ms"] = progress.get("started_at_ms", 0)
            try:
                recovery = semantic_recovery.summary(c.attempt_path or c.status_path.with_name("lane_resume_attempts.json"), task_id)
            except semantic_recovery.SemanticRecoveryError:
                recovery = {"state": "blocked", "reason": "recovery_journal_invalid", "completion_authority": False}
            if recovery.get("spent"):
                previous["semantic_recovery"] = recovery
            if unresolved:
                previous["unresolved_prior_failure_reason"] = "prior_semantic_failure_not_validated"
            else:
                previous.pop("unresolved_prior_failure_reason", None)
            records[task_id] = previous

            descendants = []
            def depends(identity: str, seen: set[str]) -> bool:
                if identity in seen:
                    return False
                seen.add(identity)
                direct = dependencies(tasks[identity])
                return task_id in direct or any(depends(dep, seen) for dep in direct)
            for identity, item in tasks.items():
                if identity != task_id and item.get("status", "").lower() not in DONE and depends(identity, set()):
                    descendants.append(identity)
            legacy = bool(previous.get("legacy_unqualified"))
            reported = max(failures, raw if legacy else 0)
            cap = semantic_failure_limit(fields)
            code = ""
            if descendants and (reported >= cap or gates >= c.failure_loop_attempts or qualified and progress.get("escalated")):
                code = "exhausted_required_barrier"
            elif qualified and (progress.get("escalated") or reported >= cap):
                code = "exhausted_semantic_failure_budget"
            elif max(reported, abandoned) >= c.failure_loop_attempts:
                codes = activity.get("task_provider_failure_codes", []) if active == task_id else []
                code = "provider_failure_loop" if codes else "repeated_semantic_failure_loop"
            timeout = float(fields.get("implementation_timeout_seconds") or c.default_task_timeout_seconds)
            since = float(previous["first_seen"])
            if not code and (previous.get("was_active") or reported) and now - since > max(timeout, c.semantic_stall_seconds) + c.deadline_grace_seconds:
                code = "sustained_no_semantic_progress"
            if not code and legacy:
                code = "unqualified_prior_failure_history"
            if recovery.get("reason") == "recovery_journal_invalid":
                code = "bounded_semantic_recovery_journal_invalid"
            if recovery.get("blocked_contract") == contract:
                code = "bounded_semantic_recovery_exhausted" if recovery.get("spent", 0) >= recovery.get("recovery_limit", 1) else "bounded_semantic_recovery_needs_changed_diagnostics"
            if code:
                incidents.append({"code": code, "task_id": task_id, "contract_fingerprint": contract,
                                  "incident_fingerprint": fingerprint, "failure_count": reported,
                                  "contract_qualified": qualified or proof_qualified,
                                  "blocked_descendant_count": len(descendants), "blocked_task_ids": descendants[:64],
                                  "since": since, "recovery": "repair_declared_execution_inputs_or_admit_bounded_repair",
                                  "unit_restart_is_not_a_semantic_repair": True,
                                  **({"foreground_recovery": recovery} if recovery.get("spent") else {})})
        return incidents

    def observe(self, *, apply: bool = False, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        c = self.config
        result: dict[str, Any] = {"schema": SCHEMA, "unit": c.unit, "lane_id": c.lane_id,
                                  "observed_at": now, "action": "observe", "incidents": [],
                                  "work_complete": False, "completion_authority": False, "body_free": True}
        try:
            held = self._holds(apply=apply, now=now)
            if self.hold_reviews:
                result["hold_reviews"] = self.hold_reviews
            if held:
                result.update(action="operator_hold", reason="operator_hold")
                return self._finish(result)
            capacity = self._capacity()
            if not capacity["ok"]:
                return self._storage_wait(result, "storage_pressure:" + capacity["reason"])
            tasks = parse_board(c.board_path.read_text(), c.task_prefix)
            ready = [identity for identity, fields in tasks.items() if fields.get("status", "").lower() in READY
                     and all(tasks[dep].get("status", "").lower() in DONE for dep in dependencies(fields))]
            pending = [identity for identity, fields in tasks.items() if fields.get("status", "").lower() not in DONE]
            repair = {raw: _file_hash(_path(c.repo_root, raw)) for raw in c.repair_paths}
            progress_key = _digest({"tasks": {identity: fields.get("status", "") for identity, fields in tasks.items()}, "repair": repair})
            if self.state.get("progress_key") != progress_key:
                cleanup = self.state.get("pending_recovery")
                task_progress = self.state.get("task_progress")
                self.state = {"progress_key": progress_key, "restart_attempts": 0, "first_seen": now}
                if cleanup:
                    self.state["pending_recovery"] = cleanup
                if task_progress:
                    self.state["task_progress"] = task_progress
            owners = self._owners()
            pending_recovery = self.state.get("pending_recovery")
            if pending_recovery:
                old_alive = [pid for pid, birth in pending_recovery.get("process_births", {}).items()
                             if (process := self.host.process(int(pid))) and process["start_ticks"] == birth]
                if not old_alive and owners["child"] and owners["fingerprint"] != pending_recovery["fingerprint"]:
                    result["cleanup_verified"] = True
                    self.state["last_cleanup_verified_at"] = now
                    self.state.pop("pending_recovery", None)
                elif old_alive and not (pending_recovery.get("action_failed")
                                         and owners["fingerprint"] == pending_recovery["fingerprint"]):
                    result.update(action="cleanup_unverified", reason="prior_service_processes_still_present")
                    result["incidents"].append({"code": "service_cleanup_unverified", "recovery": "inspect_exact_systemd_cgroup"})
                    return self._finish(result)
            result.update(owner_fingerprint=owners["fingerprint"], task_count=len(tasks), pending_count=len(pending),
                          ready_task_ids=ready, restart_attempts=int(self.state.get("restart_attempts", 0)))
            if not pending:
                result.update(action="board_tasks_terminal", reason="no_nonterminal_tasks")
                return self._finish(result)
            if not ready:
                result["incidents"].append({"code": "blocked_ready_frontier", "pending_task_ids": pending[:64],
                                            "recovery": "repair_dependencies_or_admit_missing_work"})
                result.update(action="attention_required", reason="no_selectable_pending_task")
                return self._finish(result)
            unit, main, child = owners["unit"], owners["main"], owners["child"]
            if unit.get("ActiveState") in {"activating", "deactivating", "reloading"}:
                result.update(action="service_transition", reason="systemd_owns_current_transition")
                return self._finish(result)
            if child:
                def same_foreground(pid: int) -> bool:
                    current = self.host.process(pid)
                    return bool(current and current["start_ticks"] == child["start_ticks"]
                                and current["argv"] == child["argv"])
                emergency = read_emergency_status(c.status_path, expected_pid=child["pid"],
                                                  identity_check=same_foreground)
                if emergency:
                    emergency_active = emergency["payload"].get("active_task_id", "")
                    if emergency_active and emergency_active not in tasks:
                        raise ObservationError("storage telemetry names a foreign task")
                    result.update(active_task_id=emergency_active,
                                  owner_storage_telemetry=True)
                    return self._finish(self._storage_wait(result, "verified_foreground_storage_pressure"))
            status = _json(c.status_path, missing=True)
            active = str(status.get("active_task_id") or "") if child else ""
            if active and active not in tasks:
                raise ObservationError("status names a foreign task")
            if child and status.get("pid") != child["pid"]:
                raise ObservationError("lane status is not bound to the leased foreground PID")
            result.update(active_task_id=active, phase=status.get("phase", ""))
            task = tasks.get(active, {})
            activity = self._activity(owners, active, task)
            result.update(activity)
            if activity["provider_failure_codes"]:
                result["incidents"].append({"code": "provider_failures_observed", "task_id": active,
                                            "failure_codes": activity["provider_failure_codes"],
                                            "recovery": "inspect_pinned_provider_receipts_and_remaining_task_budget"})
            semantic_incidents = self._semantic_failures(tasks, ready, active, activity, now)
            result["incidents"].extend(semantic_incidents)
            progress_records = self.state.get("task_progress", {})
            recovery = (progress_records.get(active) or {}).get("semantic_recovery") or {}
            if recovery:
                result["foreground_semantic_recovery"] = recovery
            unresolved_tasks = [identity for identity in dict.fromkeys([*ready, *([active] if active else [])])
                                if (progress_records.get(identity) or {}).get("unresolved_prior_failure") is True]
            prior_failure = bool(unresolved_tasks) or any(int((progress_records.get(identity) or {}).get("observed_failures") or 0) > 0
                                for identity in [*ready, *([active] if active else [])])
            result["unresolved_prior_failure"] = bool(unresolved_tasks)
            if unresolved_tasks:
                result.update(unresolved_prior_failure_reason="prior_semantic_failure_not_validated",
                              unresolved_prior_failure_task_ids=unresolved_tasks[:64])
            result["health"] = "degraded" if semantic_incidents or prior_failure or activity["provider_failure_codes"] else "operational"
            if semantic_incidents:
                result["semantic_progress_state"] = semantic_incidents[0]["code"]
            owner_epoch = (main or {}).get("start_ticks", "absent")
            epoch_key = str(owner_epoch) + ":" + active
            if self.state.get("active_epoch") != epoch_key:
                self.state.update(active_epoch=epoch_key, active_seen_at=now, idle_seen_at=now, backoff_seen_at=now)
                self.state.pop("task_anchor", None)
            reason = ""
            if not main or not child:
                since = self.state.setdefault("owner_absent_since", now)
                if now - since >= c.startup_grace_seconds:
                    reason = "healthy_foreground_owner_absent"
                else:
                    result.update(action="startup_grace", reason="awaiting_verified_foreground_owner")
            else:
                self.state.pop("owner_absent_since", None)
                heartbeat = status.get("heartbeat_at_ms", 0)
                heartbeat_age = max(0, now - float(heartbeat) / 1000) if isinstance(heartbeat, (int, float)) else None
                result["heartbeat_age_seconds"] = heartbeat_age
                recovery_wait = str(status.get("phase") or "").startswith("semantic_recovery_")
                if active and status.get("phase") != "semantic_failure_wait" and not recovery_wait:
                    try:
                        timeout = float(task.get("implementation_timeout_seconds") or c.default_task_timeout_seconds)
                    except ValueError as exc:
                        raise ObservationError("task deadline is malformed") from exc
                    if not math.isfinite(timeout) or not 0 < timeout <= c.max_task_timeout_seconds:
                        raise ObservationError("task deadline exceeds configured bound")
                    start = activity["worker_started_at"]
                    if status.get("phase") == "semantic_recovery":
                        if recovery.get("state") != "running" or not 0 < recovery.get("spent", 0) <= semantic_recovery.MAX_RECOVERIES:
                            raise ObservationError("foreground recovery admission is not verified")
                        start = float(recovery.get("started_at_ms", 0)) / 1000
                        timeout = float(recovery.get("timeout_seconds", 0))
                        if not 0 < start <= now or not 0 < timeout <= min(c.max_task_timeout_seconds, 1200):
                            raise ObservationError("foreground recovery deadline is invalid")
                        recovery_epoch = str(recovery.get("qualification_fingerprint")) + ":" + str(recovery.get("spent"))
                        if self.state.get("foreground_recovery_epoch") != recovery_epoch:
                            self.state.update(foreground_recovery_epoch=recovery_epoch, task_anchor=start)
                    elif recovery.get("state") == "applied" and recovery.get("approved_contract") == (progress_records.get(active) or {}).get("contract"):
                        # One genuine original-producer admission after a repair
                        # can have waited under HOLD. Its persisted start, rather
                        # than a PID or fresh heartbeat, grants this sole anchor.
                        producer_start = (progress_records.get(active) or {}).get("execution_started_at_ms", 0)
                        probe_epoch = str(recovery.get("qualification_fingerprint")) + ":" + str(recovery.get("approved_contract"))
                        if (isinstance(producer_start, (int, float)) and recovery.get("finished_at_ms", 0) <= producer_start <= now * 1000
                                and producer_start > 0 and self.state.get("foreground_recovery_probe_epoch") != probe_epoch):
                            self.state.update(foreground_recovery_probe_epoch=probe_epoch, task_anchor=producer_start / 1000)
                            start = producer_start / 1000
                    if start is None:
                        transition = status.get("last_task_transition_at_ms", 0)
                        start = max(main["start_wall"], float(transition) / 1000) if isinstance(transition, (int, float)) and 0 < transition <= now * 1000 else self.state["active_seen_at"]
                    # Persist the earliest valid anchor for this task/owner.
                    anchor = min(self.state.get("task_anchor", start), start)
                    self.state["task_anchor"] = anchor
                    deadline = anchor + timeout + c.deadline_grace_seconds
                    result.update(task_deadline_at=deadline, task_deadline_remaining_seconds=deadline - now,
                                  native_progress_observed=self.state.get("native_signature") is not None and self.state["native_signature"] != activity["native_activity_signature"],
                                  real_work_observed=self.state.get("work_signature") is not None and self.state["work_signature"] != activity["work_signature"])
                    if heartbeat_age is not None and heartbeat_age > c.heartbeat_stale_seconds:
                        result["incidents"].append({"code": "heartbeat_stale", "task_id": active,
                                                    "recovery": "observe_declared_deadline_before_unit_recovery"})
                    if status.get("phase") in {"attempt_retry_wait", "gate_retry_wait"}:
                        backoff_epoch = epoch_key + ":" + status["phase"]
                        if self.state.get("backoff_epoch") != backoff_epoch:
                            self.state.update(backoff_epoch=backoff_epoch, backoff_seen_at=now)
                        retry_at = status.get("next_retry_at_ms", 0)
                        maximum = self.state["backoff_seen_at"] + c.max_backoff_seconds
                        declared_retry = float(retry_at) / 1000 if isinstance(retry_at, (int, float)) and retry_at > 0 else maximum
                        bounded_until = min(declared_retry, maximum)
                        result["bounded_retry_at"] = bounded_until
                        if now > bounded_until + c.deadline_grace_seconds:
                            reason = "bounded_retry_wait_exceeded"
                        else:
                            result.update(action="bounded_retry_wait", reason="declared_retry_is_not_a_stall")
                    else:
                        self.state.pop("backoff_epoch", None)
                        if now >= deadline:
                            reason = "declared_task_deadline_exceeded"
                        else:
                            result.update(action="active_with_prior_semantic_failure" if prior_failure else
                                          "active_with_provider_failure" if activity["provider_failure_codes"] else "active_within_deadline",
                                          reason="retrying_after_semantic_failure" if prior_failure else "native_activity_cannot_extend_task_deadline")
                elif status.get("phase") == "semantic_failure_wait":
                    result.update(action="attention_required", reason="unchanged_semantic_failure_budget_exhausted")
                elif recovery_wait:
                    result.update(action="attention_required", reason="foreground_recovery_" + str(recovery.get("state", "unverified")))
                elif status.get("phase") in {"attempt_retry_wait", "gate_retry_wait"}:
                    # A deliberately empty active_task_id during cooldown is
                    # not an idle producer. Keep its original bound even if
                    # systemd changes the service PID during that wait.
                    wait_key = _digest({"tasks": ready, "contracts": {
                        identity: (self.state.get("task_progress", {}).get(identity) or {}).get("contract")
                        for identity in ready}, "phase": status["phase"]})
                    if self.state.get("idle_backoff_key") != wait_key:
                        self.state.update(idle_backoff_key=wait_key, idle_backoff_seen_at=now)
                    ledger = _json(c.attempt_path or c.status_path.with_name("lane_resume_attempts.json"), missing=True)
                    retry_values = [stamp for identity, stamp in (ledger.get("_retry_after_ms") or {}).items()
                                    if identity in ready and type(stamp) is int and stamp > 0]
                    retry_at = status.get("next_retry_at_ms") or min(retry_values, default=0)
                    maximum = self.state["idle_backoff_seen_at"] + c.max_backoff_seconds
                    declared = retry_at / 1000 if isinstance(retry_at, (int, float)) and retry_at > 0 else maximum
                    bounded_until = min(declared, maximum)
                    result["bounded_retry_at"] = bounded_until
                    if now > bounded_until + c.deadline_grace_seconds:
                        reason = "bounded_retry_wait_exceeded"
                    else:
                        result.update(action="bounded_retry_wait", reason="declared_retry_is_not_a_stall")
                elif now - self.state["idle_seen_at"] > c.idle_stall_seconds:
                    reason = "selectable_work_without_active_task"
                else:
                    result.update(action="idle_grace", reason="selectable_work_awaiting_next_cycle")
            self.state.update(native_signature=activity["native_activity_signature"], work_signature=activity["work_signature"])
            if reason:
                result["incidents"].append({"code": reason, "task_id": active,
                                            "recovery": "restart_only_the_registered_leased_service"})
                result.update(action="recovery_due", reason=reason)
                if int(self.state.get("restart_attempts", 0)) >= c.restart_budget:
                    result.update(action="attention_required", reason="persisted_restart_budget_exhausted")
                elif now < float(self.state.get("retry_after", 0)):
                    result.update(action="recovery_cooldown", reason="persisted_restart_cooldown")
                elif apply:
                    self._scope(active or ready[0], tasks)
                    if self._holds() or self._owners()["fingerprint"] != owners["fingerprint"]:
                        raise ObservationError("recovery fence changed before service action")
                    capacity = self._capacity()
                    if not capacity["ok"]:
                        return self._storage_wait(result, "storage_pressure:" + capacity["reason"])
                    previous_state = dict(self.state)
                    self.state["restart_attempts"] = int(self.state.get("restart_attempts", 0)) + 1
                    self.state["retry_after"] = now + c.restart_cooldown_seconds
                    self.state["last_restart_reason"] = reason
                    old_members = self.host.members(owners["unit"]["ControlGroup"]) if owners["unit"].get("ControlGroup") else []
                    self.state["pending_recovery"] = {"fingerprint": owners["fingerprint"], "process_births": {
                        str(pid): process["start_ticks"] for pid in old_members if (process := self.host.process(pid))}}
                    try:
                        atomic_json(self.state_path, self.state)
                    except OSError:
                        # No service action was dispatched: do not retain a
                        # failed reservation or cleanup fence in memory.
                        self.state = previous_state
                        raise
                    try:
                        self.host.restart(c)
                    except (ObservationError, OSError, subprocess.SubprocessError):
                        self.state["pending_recovery"]["action_failed"] = True
                        raise
                    result.update(action="service_restart_requested", restart_attempts=self.state["restart_attempts"],
                                  recovery_verified=False, reason=reason)
            elif semantic_incidents:
                result.update(action="attention_required", reason=semantic_incidents[0]["code"])
        except (ObservationError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, subprocess.SubprocessError) as exc:
            if is_storage_pressure(exc):
                return self._storage_wait(result, "storage_pressure:" + str(exc.errno), exc)
            result.update(action="observation_blocked", reason=str(exc) if isinstance(exc, ObservationError) else type(exc).__name__)
            result["incidents"].append({"code": "ownership_or_observation_unverified", "recovery": "repair_configuration_or_wait_for_verified_owner"})
        return self._finish(result)

    def _finish(self, result: dict[str, Any]) -> dict[str, Any]:
        try:
            result["durable_persistence"] = True
            atomic_json(self.state_path, self.state)
            atomic_json(self.status_path, result)
            if result.get("incidents"):
                semantic = [item for item in result["incidents"] if item.get("incident_fingerprint")]
                if semantic:
                    for incident in semantic:
                        atomic_json(self.config.state_dir / ("incident-" + incident["incident_fingerprint"][:20] + ".json"),
                                    {**result, "incident": incident})
                else:
                    identity = _digest({"reason": result.get("reason"), "task": result.get("active_task_id"), "progress": self.state.get("progress_key")})
                    atomic_json(self.config.state_dir / ("incident-" + identity[:20] + ".json"), result)
        except OSError as error:
            if not is_storage_pressure(error):
                raise
            return self._storage_wait(result, "telemetry_storage_pressure", error)
        clear_emergency_status(self.status_path)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Authorize restart of the exact registered unit")
    parser.add_argument("--once", action="store_true", help="Observe once; never restart")
    args = parser.parse_args(argv)
    try:
        config = Config.from_dict(_json(args.config))
        while True:
            try:
                capacity = check_storage_capacity([config.repo_root, config.state_dir, config.status_path])
                if not capacity["ok"]:
                    raise OSError(capacity.get("errno", errno.ENOSPC), "watchdog startup storage pressure")
                lock = acquire_lock(config.state_dir / "watchdog.lock")
                break
            except OSError as error:
                if not is_storage_pressure(error):
                    raise
                result = {"schema": SCHEMA, "unit": config.unit, "lane_id": config.lane_id,
                          "action": "storage_wait", "reason": "watchdog_startup_storage_pressure",
                          "storage_pressure": True, "health": "degraded", "work_complete": False,
                          "completion_authority": False}
                publish_emergency_status(config.state_dir / "status.json", result, error=error)
                print(json.dumps(result, sort_keys=True), flush=True)
                if args.once:
                    return 0
                time.sleep(config.poll_seconds)
        if lock is None:
            return 0
        with lock:
            watchdog = LeasedServiceWatchdog(config)
            while True:
                result = watchdog.observe(apply=args.apply and not args.once)
                print(json.dumps(result, sort_keys=True), flush=True)
                if args.once:
                    return 0
                time.sleep(config.poll_seconds)
    except (ObservationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"schema": SCHEMA, "action": "configuration_blocked", "error_type": type(exc).__name__,
                          "work_complete": False, "completion_authority": False}), flush=True)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
