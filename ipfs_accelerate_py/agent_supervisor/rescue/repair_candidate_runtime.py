"""Service adapter for isolated repair candidates under the existing job unit.

This module allocates no scheduler or storage reservation.  It verifies the
original repair-worker lock and bounded systemd job before using the original
checkout lease.  Provider children use the existing inherited write boundary;
only pinned, configured owner commands can validate or adopt a frozen result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping

from . import repair_candidate as candidate
from ..merge.checkout_lock import (
    acquire_checkout_mutation_lease, checkout_lock_metadata,
    checkout_mutation_lock_path, release_checkout_mutation_lease,
)
from ..todo_daemon.engine import run_command

SCHEMA = "agent-supervisor/repair-candidate-runtime@1"
PENDING = frozenset({"candidate_ready", "validation_pending", "adoption_pending",
                     "candidate_pending", "adopted_pending_probe"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_MAX_METADATA = 8 * 1024 * 1024


def _need(value: bool, message: str) -> None:
    if not value:
        raise candidate.CandidateError(message)


def _wire(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def _read(path: Path, maximum: int = _MAX_METADATA) -> bytes:
    _need(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
          "aliased runtime metadata/source")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
              and before.st_nlink == 1 and 0 <= before.st_size <= maximum,
              "bounded ordinary parent-owned file required")
        raw = b""
        while True:
            part = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not part:
                break
            raw += part
            _need(len(raw) <= maximum, "runtime file byte bound")
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink,
                           s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        _need(len(raw) == before.st_size and stamp(before) == stamp(os.fstat(fd))
              == stamp(path.lstat()), "runtime file changed while observed")
        return raw
    finally:
        os.close(fd)


def _json(raw: bytes) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate runtime JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    _need(type(value) is dict, "runtime JSON object required")
    return value


def _pin(path: Path, maximum: int = _MAX_METADATA) -> dict[str, Any]:
    raw = _read(path, maximum)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _checked(ref: Mapping[str, Any]) -> bytes:
    _need(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}
          and type(ref["bytes"]) is int and 0 <= ref["bytes"] <= _MAX_METADATA
          and type(ref["sha256"]) is str and _SHA.fullmatch(ref["sha256"]) is not None,
          "closed typed source/receipt descriptor required")
    raw = _read(Path(ref["path"]))
    _need(len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
          "source/receipt descriptor changed")
    return raw


def _new(path: Path, value: Any) -> dict[str, Any]:
    raw = _wire(value)
    _need(len(raw) <= _MAX_METADATA, "runtime result byte bound")
    with path.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o400)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return _pin(path)


def _integer(value: Any, maximum: int, name: str) -> int:
    _need(type(value) is int and 0 < value <= maximum, "finite " + name + " required")
    return value


@dataclass(frozen=True)
class OwnerCommand:
    argv: tuple[str, ...]
    verify_argv: tuple[str, ...]
    source_descriptors: tuple[dict[str, Any], ...]
    timeout_seconds: int


def _owner(value: Any, stage: str) -> OwnerCommand:
    _need(type(value) is dict and set(value) == {
        "argv", "verify_argv", "source_descriptors", "timeout_seconds"},
        "configured original " + stage + " owner unavailable")
    commands = []
    permitted = {"{candidate_root}", "{freeze_path}", "{receipt_path}",
                 "{source_commitment}", "{validation_path}", "{invocation_root}"}
    for key in ("argv", "verify_argv"):
        command = value[key]
        _need(type(command) is list and 0 < len(command) <= 128
              and all(type(s) is str and s and "\0" not in s for s in command),
              "exact original owner argv required")
        _need(Path(command[0]).is_absolute(), "absolute original owner executable required")
        _need(all(("{" not in s and "}" not in s) or s in permitted for s in command),
              "only exact bounded owner argument substitutions allowed")
        _need("{receipt_path}" in command, "original owner receipt argument missing")
        commands.append(tuple(command))
    _need("{freeze_path}" in commands[0], "original owner must consume exact frozen candidate")
    _need(stage != "adoption" or "{validation_path}" in commands[0],
          "original adoption owner must consume the accepted validation")
    refs = value["source_descriptors"]
    _need(type(refs) is list and 0 < len(refs) <= 128
          and len({r["path"] for r in refs}) == len(refs), "closed original owner source population")
    for ref in refs:
        _checked(ref)
    for command in commands:
        _need(any(ref["path"] in command for ref in refs), "owner command lacks pinned source")
    return OwnerCommand(*commands, tuple(refs), _integer(value["timeout_seconds"], 1200, "owner deadline"))


@dataclass(frozen=True)
class RuntimePolicy:
    sources: tuple[dict[str, Any], ...]
    validation_owner: OwnerCommand
    adoption_owner: OwnerCommand
    bounds: dict[str, int]
    checkout_timeout_seconds: int


def parse_runtime_policy(worker: Mapping[str, Any], policy: candidate.CandidatePolicy) -> RuntimePolicy:
    value = worker.get("candidate_runtime")
    _need(type(value) is dict and set(value) == {
        "schema", "source_descriptors", "validation_owner", "adoption_owner",
        "resource_bounds", "checkout_timeout_seconds"} and value["schema"] == SCHEMA,
        "original candidate runtime owners are not configured")
    refs = value["source_descriptors"]
    _need(type(refs) is list and len(refs) == len(policy.source_paths)
          and {r["path"] for r in refs} == {str(policy.repository_root / p) for p in policy.source_paths},
          "exact candidate source manifest required")
    for ref in refs:
        _checked(ref)
    _need(sum(r["bytes"] for r in refs) <= policy.max_source_bytes, "source manifest byte bound")
    bounds = value["resource_bounds"]
    _need(type(bounds) is dict and set(bounds) == {
        "memory_max_bytes", "cpu_quota_percent", "tasks_max", "runtime_max_seconds"},
        "closed existing job-unit bounds required")
    caps = {"memory_max_bytes": 16 * 1024 ** 3, "cpu_quota_percent": 100,
            "tasks_max": 64, "runtime_max_seconds": 2400}
    for name, maximum in caps.items():
        _integer(bounds[name], maximum, name)
    validation, adoption = (_owner(value[k], stage) for k, stage in
                            (("validation_owner", "validation"), ("adoption_owner", "adoption")))
    worker_deadline = _integer(worker.get("timeout_seconds", 2400), 2400, "original worker deadline")
    effective_deadline = min(worker_deadline, bounds["runtime_max_seconds"])
    _need(all(command.timeout_seconds * 2 + 60 <= effective_deadline for command in (validation, adoption)),
          "original job deadline cannot accommodate owner plus verifier and cleanup")
    return RuntimePolicy(tuple(refs), validation, adoption, dict(bounds),
                         _integer(value["checkout_timeout_seconds"], 30, "checkout wait"))


def candidate_runtime_preflight(config: Mapping[str, Any]) -> str | None:
    """Pure configuration/source preflight for the outer worker before coding charge."""
    try:
        policy = candidate.parse_candidate_policy(config["repair_worker"])
        if policy is None:
            return None
        _need(policy.state_root == Path(config["state_dir"]), "candidate protected state differs from fleet state")
        parse_runtime_policy(config["repair_worker"], policy)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "repair_candidate_owner_unavailable:" + type(error).__name__
    return None


def _proc(pid: int) -> dict[str, Any] | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
        tail = raw[raw.rfind(")") + 2:].split()
        if tail[0] == "Z":
            return None
        return {"pid": pid, "birth": tail[19],
                "argv": (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")}
    except FileNotFoundError:
        return None


def _flock_owner(path: Path) -> dict[str, Any]:
    info = path.lstat()
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1,
          "original global worker lock custody differs")
    owners = []
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
            continue
        device = fields[5].split(":")
        if len(device) == 3 and (int(device[0], 16), int(device[1], 16), int(device[2])) == (
                os.major(info.st_dev), os.minor(info.st_dev), info.st_ino):
            owners.append(int(fields[4]))
    _need(len(owners) == 1, "original global repair worker ownership is unknown")
    process = _proc(owners[0])
    _need(process is not None and any(b"fleet_repair" in v for v in process["argv"])
          and b"run" in process["argv"], "global lock is not held by the original repair worker")
    return {"path": str(path), "pid": process["pid"], "birth": process["birth"],
            "device": info.st_dev, "inode": info.st_ino}


def _duration(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(us|ms|s|min|h)?", value)
    _need(match is not None, "finite actual systemd runtime limit unavailable")
    return float(match[1]) * {None: .000001, "us": .000001, "ms": .001,
                              "s": 1, "min": 60, "h": 3600}[match[2]]


def _unit_snapshot(unit: str, bounds: Mapping[str, int], runner) -> dict[str, Any]:
    result = runner({"argv": ["systemctl", "--user", "show", unit,
        "--property=MainPID,InvocationID,ControlGroup,ActiveState,KillMode,RuntimeMaxUSec,ActiveEnterTimestampMonotonic"]}, cwd="/", timeout=10)
    _need(result["returncode"] == 0, "original repair cgroup observation failed")
    props = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
    _need(props.get("ActiveState") == "active" and props.get("MainPID") == str(os.getpid())
          and props.get("KillMode") == "control-group" and re.fullmatch(r"[0-9a-f]{32}", props.get("InvocationID", "")),
          "caller is not the actual original bounded systemd job owner")
    group = props.get("ControlGroup", "")
    _need(group.startswith("/") and ".." not in Path(group).parts, "actual job cgroup unavailable")
    root = Path("/sys/fs/cgroup") / group.lstrip("/")
    memberships = [s.split(":", 2)[2] for s in Path("/proc/self/cgroup").read_text().splitlines()
                   if s.startswith("0::")]
    _need(memberships == [group], "caller is outside the original job cgroup")
    memory = (root / "memory.max").read_text().strip()
    tasks = (root / "pids.max").read_text().strip()
    quota, period = (root / "cpu.max").read_text().split()
    _need(memory.isdecimal() and 0 < int(memory) <= bounds["memory_max_bytes"]
          and tasks.isdecimal() and 0 < int(tasks) <= bounds["tasks_max"]
          and quota.isdecimal() and period.isdecimal() and int(period) > 0
          and 0 < int(quota) * 100 <= bounds["cpu_quota_percent"] * int(period),
          "actual cgroup limits do not meet configured finite bounds")
    _need(0 < _duration(props.get("RuntimeMaxUSec", "")) <= bounds["runtime_max_seconds"],
          "original systemd job deadline is not bounded")
    entered = props.get("ActiveEnterTimestampMonotonic", "")
    _need(entered.isdecimal() and int(entered) > 0, "original unit start clock is unknown")
    return {"unit": unit, "invocation_id": props["InvocationID"], "control_group": group,
            "cgroup_root": str(root), "memory_max_bytes": int(memory), "tasks_max": int(tasks),
            "cpu_max": [int(quota), int(period)], "runtime_max_seconds": _duration(props["RuntimeMaxUSec"]),
            "active_enter_monotonic_usec": int(entered)}


class CandidateRuntime:
    """Concrete authority adapter held inside one original repair-job invocation."""
    def __init__(self, config: dict[str, Any], board: dict[str, Any], job_path: Path,
                 token: str, *, runner=None, command_runner=run_command):
        from . import fleet_repair as fleet
        self.fleet = fleet
        self.runner = runner or fleet.command
        self.command_runner = command_runner
        self.config, self.board, self.job_path, self.token = config, board, job_path, token
        self.policy = candidate.parse_candidate_policy(config["repair_worker"])
        _need(self.policy is not None, "isolated candidate policy required")
        self.runtime_policy = parse_runtime_policy(config["repair_worker"], self.policy)
        _need(self.policy.state_root == Path(config["state_dir"]), "protected fleet state differs")
        _need(type(token) is str and _TOKEN.fullmatch(token) is not None, "exact attempt token required")
        _need(job_path == self.policy.state_root / "repairs" / board["id"] / "job.json",
              "job does not belong to the original queue")
        self.config_ref = _pin(Path(config["_config_path"]))
        self.job = self._job()
        self.attempt_number = _integer(self.job["attempts"], 1000000, "existing coding attempt")
        self.unit = fleet.repair_job_unit(config)
        self.global_owner = _flock_owner(self.policy.state_root / "repair-worker.lock")
        self.unit_owner = _unit_snapshot(self.unit, self.runtime_policy.bounds, self.runner)
        self.storage_policy = {**config["repair_worker"], "cwd": str(self.policy.repository_root)}
        _need(fleet._storage_preflight(self.storage_policy, job_path, str(self.policy.state_root)) is None,
              "original repair storage preflight refused")
        self.directory = self.policy.state_root / "repair-runtime" / board["id"] / token / self.unit_owner["invocation_id"]
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.counter = 0
        self.lease = None
        self.session = None
        self.provider_closed = False
        self.checkout_lease_closed = False
        self.resuming = False
        self.accepted: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def _job(self, *, execution: bool = True) -> dict[str, Any]:
        job = _json(_read(self.job_path))
        _need(job.get("board_id") == self.board["id"] and job.get("repair_unit_token") == self.token
              and job.get("status") in {"running", *PENDING}, "current original job/token was replaced")
        _need(os.environ.get("IPFS_TASKBOARD_REPAIR_BOARD_ID") == self.board["id"]
              and os.environ.get("IPFS_TASKBOARD_REPAIR_JOB_TOKEN") == self.token,
              "job cgroup environment does not carry the original attempt")
        if hasattr(self, "attempt_number"):
            _need(type(job.get("attempts")) is int and job["attempts"] == self.attempt_number,
                  "coding attempt changed")
        if execution:
            _need(not self.fleet.hold_paths(self.board)
                  and not self.fleet.observation_only(self.board)
                  and not self.fleet._recovery_denial(self.config, self.board),
                  "original hold or recovery policy no longer permits this stage")
        return job

    def _receipt(self, kind: str, stage: str, binding: Mapping[str, Any], details: Any):
        self.counter += 1
        ref = _new(self.directory / f"guard-{self.counter:04d}-{kind}.json",
                   {"schema": SCHEMA, "kind": kind, "stage": stage,
                    "binding_sha256": candidate.digest(binding), "details": details})
        return candidate.OwnerGuardReceipt(kind, ref["path"], candidate.digest(binding))

    def _source_guard(self, stage: str, binding: Mapping[str, Any]):
        self._job()
        _checked(self.config_ref)
        owner = _flock_owner(self.policy.state_root / "repair-worker.lock")
        _need(owner == self.global_owner, "original global repair worker changed")
        result = self.command_runner(["git", "rev-parse", "--verify", "HEAD"],
            cwd=self.policy.repository_root, timeout_seconds=self.policy.git_timeout_seconds)
        _need(result.returncode == 0 and result.stdout.strip() == self.policy.base_commit,
              "original selected source HEAD changed")
        for ref in self.runtime_policy.sources:
            _checked(ref)
        for command in (self.runtime_policy.validation_owner, self.runtime_policy.adoption_owner):
            for ref in command.source_descriptors:
                _checked(ref)
        return self._receipt("source", stage, binding, {"global_worker": owner,
            "base_commit": self.policy.base_commit, "manifest_sha256": candidate.digest(self.runtime_policy.sources)})

    def _resource_guard(self, stage: str, binding: Mapping[str, Any]):
        self._job()
        unit = _unit_snapshot(self.unit, self.runtime_policy.bounds, self.runner)
        _need(unit == self.unit_owner, "original systemd invocation changed")
        _need(self.fleet._storage_preflight(self.storage_policy, self.job_path,
              str(self.policy.state_root)) is None, "original storage preflight refused")
        if stage in {"freeze", "validate", "adopt", "resume"}:
            _need(self.provider_closed, "provider outcome/cgroup closure is unknown")
            root = Path(unit["cgroup_root"])
            _need({int(p) for p in (root / "cgroup.procs").read_text().split()} == {os.getpid()}
                  and not any(p.is_dir() for p in root.iterdir()), "original job still contains provider/owner children")
        if stage == "resume":
            terminal_path = Path(binding["metadata_root"]) / "provider-terminal.json"
            _need(not terminal_path.lstat().st_mode & 0o222, "retained provider terminal is not parent-frozen")
            terminal = _json(_read(terminal_path))
            _need(type(terminal.get("returncode")) is int
                  and terminal.get("binding_sha256") == candidate.digest(binding)
                  and terminal.get("known_cgroup_empty_except_original_owner") is True,
                  "retained original provider closeout is missing or mismatched")
            self._boundary(_json(_checked(terminal["boundary"])))
        return self._receipt("resource", stage, binding, {"unit": unit,
            "storage_preflight_only_not_disk_reservation": True, "provider_closed": self.provider_closed})

    def _owner_guard(self, stage: str, binding: Mapping[str, Any], receipt: Mapping[str, Any]):
        if stage == "validate" and stage not in self.accepted and getattr(self, "resuming", False):
            self._rehydrate_validation(binding, receipt)
        _need(stage in self.accepted, "no actual configured original owner result")
        observed, ref = self.accepted[stage]
        _need(_wire(receipt) == _wire(observed) and _wire(_json(_checked(ref))) == _wire(observed),
              "parent-owned owner receipt changed")
        _need(not Path(ref["path"]).lstat().st_mode & 0o222, "owner receipt is not frozen")
        _checked(observed["original_receipt_descriptor"])
        return candidate.OwnerGuardReceipt(stage, ref["path"],
            candidate.digest(dict(binding=binding, receipt=receipt)))

    def __enter__(self):
        metadata = checkout_lock_metadata(kind="fleet-repair-candidate", repo_root=self.policy.repository_root,
            task_id="infrastructure-repair:" + self.board["id"], attempt=self.attempt_number,
            extra={"repair_unit_token": self.token, "process_birth": _proc(os.getpid())["birth"]})
        def active(old):
            process = _proc(int(old.get("pid") or 0))
            return process is not None and (not old.get("process_birth") or old["process_birth"] == process["birth"])
        self.lease, reason, _, _ = acquire_checkout_mutation_lease(
            checkout_mutation_lock_path(self.policy.repository_root), metadata,
            owner_active=active, timeout_seconds=self.runtime_policy.checkout_timeout_seconds)
        _need(self.lease is not None, "original checkout mutation lease unavailable: " + reason)
        self.authority = candidate.ExistingRepairAuthority(checkout_lease=self.lease,
            source_guard=self._source_guard, resource_guard=self._resource_guard,
            validation_guard=lambda b, r: self._owner_guard("validate", b, r),
            adoption_guard=lambda b, r: self._owner_guard("adopt", b, r))
        return self

    def __exit__(self, *_):
        if self.lease is not None:
            _need(release_checkout_mutation_lease(self.lease, require_exact_metadata=True),
                  "original checkout lease release is unknown")
            self.checkout_lease_closed = True

    def _stage(self, stage: str, **details: Any) -> dict[str, Any]:
        value = {"schema": SCHEMA, "stage": stage, "board_id": self.board["id"],
                 "attempt_token": self.token, "attempts": self.attempt_number,
                 "coding_attempts_incremented": False, "completion_authority": False, **details}
        self.counter += 1
        ref = _new(self.directory / f"stage-{self.counter:04d}-{stage}.json", value)
        with self.fleet.lock(self.job_path.parent / "queue.lock") as acquired:
            _need(acquired, "original repair queue lock busy")
            job = self._job(execution=False)  # Retaining a closeout is allowed during an operator hold.
            job.update(status=stage, candidate_runtime={"receipt": ref,
                "candidate_metadata": str(self.session.metadata_root) if self.session else "",
                "attempt_token": self.token, "attempt_number": self.attempt_number})
            self.fleet.write_json(self.job_path, job)
        return value

    def _owner_command(self, stage: str) -> dict[str, Any] | None:
        command = self.runtime_policy.validation_owner if stage == "validate" else self.runtime_policy.adoption_owner
        self.authority.require_current(stage, self.session.binding)
        invocation = self.session.metadata_root / ("original-" + stage + "-invocation")
        if invocation.exists():
            # Never repeat an owner operation after an ambiguous or interrupted write.
            terminal = invocation / "terminal.json"
            if not terminal.exists():
                return None
            previous = _json(_read(terminal))
            if (type(previous.get("command_returncode")) is not int
                    or type(previous.get("verifier_returncode")) is not int
                    or previous["command_returncode"] != 0 or previous["verifier_returncode"] != 0):
                return None
        else:
            if self._remaining_seconds() < command.timeout_seconds * 2 + 30:
                return None  # A later original unit resumes this stage, without recoding.
            invocation.mkdir(mode=0o700)
            receipt_path = invocation / "receipt.json"
            values = {"{candidate_root}": str(self.session.candidate_root),
                "{freeze_path}": str(self.session.metadata_root / "candidate-ready.json"),
                "{receipt_path}": str(receipt_path),
                "{source_commitment}": self.session.freeze["source_commitment"],
                "{validation_path}": str(self.session.metadata_root / "adoption-pending.json"),
                "{invocation_root}": str(invocation)}
            argv = [values.get(v, v) for v in command.argv]
            verify = [values.get(v, v) for v in command.verify_argv]
            _new(invocation / "intent.json", {"schema": SCHEMA, "stage": stage,
                "argv": argv, "verify_argv": verify, "freeze_sha256": self.session.freeze["freeze_sha256"],
                "source_commitment": self.session.freeze["source_commitment"],
                "owner_source_descriptors": list(command.source_descriptors),
                "runtime_policy_sha256": candidate.digest(self.config["repair_worker"]["candidate_runtime"]),
                "attempt_token": self.token, "attempt_number": self.attempt_number})
            result = self.command_runner(argv, cwd=self.policy.repository_root,
                timeout_seconds=command.timeout_seconds)
            if result.returncode != 0:
                _new(invocation / "terminal.json", {"command_returncode": result.returncode,
                    "verifier_returncode": None})
                return None
            self.authority.require_current(stage, self.session.binding)
            # The configured source-pinned original verifier, not an agent's passed flag,
            # authenticates its own original native/source/process/storage closeout.
            check = self.command_runner(verify, cwd=self.policy.repository_root,
                timeout_seconds=command.timeout_seconds)
            _new(invocation / "terminal.json", {"command_returncode": result.returncode,
                "verifier_returncode": check.returncode})
            if check.returncode != 0:
                return None
            self.authority.require_current(stage, self.session.binding)
        receipt_path = invocation / "receipt.json"
        receipt = _json(_read(receipt_path))
        original = Path(receipt.get("original_receipt_reference", ""))
        _need(original.is_absolute() and invocation in original.parents,
              "original closeout must be inside the fresh parent-owned owner invocation")
        original_ref = _pin(original)
        _need(receipt.get("original_receipt_descriptor") == original_ref,
              "original owner closeout descriptor missing or changed")
        _need(receipt.get("freeze_sha256") == self.session.freeze["freeze_sha256"]
              and receipt.get("source_commitment") == self.session.freeze["source_commitment"],
              "original owner result belongs to another candidate")
        # Parent-only publisher seals both files after the original owner/verifier closed.
        os.chmod(original, 0o400)
        os.chmod(receipt_path, 0o400)
        ref = _pin(receipt_path)
        self.accepted[stage] = (receipt, ref)
        return receipt

    def _remaining_seconds(self) -> float:
        return (self.unit_owner["active_enter_monotonic_usec"] / 1_000_000
                + self.unit_owner["runtime_max_seconds"] - time.monotonic())

    def _rehydrate_validation(self, binding: Mapping[str, Any], supplied: Mapping[str, Any]) -> None:
        """Recheck the prior original validation, without rerunning its owner command."""
        _need(self.resuming, "prior validation is only reusable on original resume")
        metadata = Path(binding["metadata_root"])
        _need(metadata == self.policy.state_root / "repair-candidates" / self.board["id"] / self.token,
              "prior validation belongs to a different original attempt")
        self.authority.require_current("resume", binding)
        invocation = metadata / "original-validate-invocation"
        intent = _json(_read(invocation / "intent.json"))
        terminal = _json(_read(invocation / "terminal.json"))
        _need(type(terminal.get("command_returncode")) is int
              and type(terminal.get("verifier_returncode")) is int
              and terminal["command_returncode"] == terminal["verifier_returncode"] == 0,
              "prior original validation terminal is missing, failed or unknown")
        frozen = _json(_read(metadata / "candidate-ready.json"))
        values = {"{candidate_root}": binding["candidate_root"],
            "{freeze_path}": str(metadata / "candidate-ready.json"),
            "{receipt_path}": str(invocation / "receipt.json"),
            "{source_commitment}": frozen["source_commitment"],
            "{validation_path}": str(metadata / "adoption-pending.json"),
            "{invocation_root}": str(invocation)}
        command = self.runtime_policy.validation_owner
        expected_argv = [values.get(v, v) for v in command.argv]
        expected_verify = [values.get(v, v) for v in command.verify_argv]
        _need(intent.get("schema") == SCHEMA and intent.get("stage") == "validate"
              and intent.get("argv") == expected_argv and intent.get("verify_argv") == expected_verify
              and _wire(intent.get("owner_source_descriptors")) == _wire(list(command.source_descriptors))
              and intent.get("runtime_policy_sha256") == candidate.digest(self.config["repair_worker"]["candidate_runtime"])
              and intent.get("attempt_token") == self.token
              and type(intent.get("attempt_number")) is int and intent["attempt_number"] == self.attempt_number
              and intent.get("freeze_sha256") == frozen["freeze_sha256"]
              and intent.get("source_commitment") == frozen["source_commitment"],
              "prior original validation command/source/freeze scope changed")
        for ref in command.source_descriptors:
            _checked(ref)
        receipt_path = invocation / "receipt.json"
        _need(not receipt_path.lstat().st_mode & 0o222, "prior validation receipt is not parent-frozen")
        receipt_ref = _pin(receipt_path)
        receipt = _json(_checked(receipt_ref))
        _need(_wire(receipt) == _wire(supplied)
              and receipt.get("freeze_sha256") == frozen["freeze_sha256"]
              and receipt.get("source_commitment") == frozen["source_commitment"],
              "retained adoption-pending validation differs from original receipt")
        original = Path(receipt.get("original_receipt_reference", ""))
        _need(original.is_absolute() and invocation in original.parents
              and not original.lstat().st_mode & 0o222,
              "prior original validation closeout is outside its immutable parent invocation")
        _need(receipt.get("original_receipt_descriptor") == _pin(original),
              "prior original validation closeout descriptor changed")
        _need(self._remaining_seconds() > command.timeout_seconds + 30,
              "insufficient original resume deadline to recheck validation")
        # Verification is read-only original-owner authentication, not a new
        # validation run or provider/owner-command replay.
        check = self.command_runner(expected_verify, cwd=self.policy.repository_root,
            timeout_seconds=command.timeout_seconds)
        _need(type(check.returncode) is int and check.returncode == 0,
              "original validation receipt verifier refused resume")
        self.authority.require_current("resume", binding)
        _need(_pin(receipt_path) == receipt_ref, "prior validation receipt changed during recheck")
        self.accepted["validate"] = (receipt, receipt_ref)
        self.counter += 1
        _new(self.directory / f"validation-recheck-{self.counter:04d}.json", {
            "schema": SCHEMA, "receipt": receipt_ref, "verify_argv": expected_verify,
            "verifier_returncode": 0, "freeze_sha256": frozen["freeze_sha256"],
            "source_commitment": frozen["source_commitment"], "validation_owner_replayed": False,
            "provider_started": False, "coding_attempts_incremented": False})

    def _boundary(self, fenced: Mapping[str, Any]) -> None:
        _need(fenced.get("schema") == "ipfs_accelerate_py/agent-supervisor/provider-filesystem-boundary@1"
              and fenced.get("task_id") == "infrastructure-repair:" + self.board["id"]
              and type(fenced.get("attempt")) is int and fenced["attempt"] == self.attempt_number
              and fenced.get("stage") == "repair" and fenced.get("applied") is True
              and fenced.get("provider_descendants_fenced") is True
              and fenced.get("shared_git_metadata_writable") is False,
              "actual inherited provider boundary was not acknowledged")

    def _provider_usage(self, roots: tuple[Path, ...]) -> dict[str, int]:
        """Bounded, no-follow observations; this is not a filesystem quota."""
        used, entries = 0, 0
        for root in roots:
            for directory, names, files in os.walk(root, followlinks=False):
                _need(len(Path(directory).relative_to(root).parts) <= 32, "provider scratch depth exceeded")
                for name in (*names, *files):
                    entries += 1
                    _need(entries <= self.policy.max_files * 8 + 256, "provider scratch entry bound exceeded")
                    path = Path(directory) / name
                    try:
                        info = path.lstat()
                    except FileNotFoundError:
                        continue  # A provider may finish/remove its own temporary file.
                    _need(info.st_uid == os.getuid() and (stat.S_ISREG(info.st_mode)
                          or stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)),
                          "provider scratch contains foreign/special storage")
                    if not stat.S_ISDIR(info.st_mode):
                        used += info.st_size
                    _need(used <= self.policy.max_candidate_bytes, "provider candidate/private/log byte bound exceeded")
        return {"bytes": used, "entries": entries, "poll_interval_seconds": 10,
                "filesystem_quota_claimed": False}

    def _run_provider(self, argv: list[str], prompt: Path, environment: dict[str, str]):
        temporary = self.fleet._repair_temporary_directory(self.config["repair_worker"])
        private = temporary / ("ipfs-repair-provider-" + self.token + "-" + self.unit_owner["invocation_id"])
        private.mkdir(mode=0o700, exist_ok=False)
        environment.update({key: str(private) for key in ("TMPDIR", "TEMP", "TMP")})
        environment.pop("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR", None)
        stdout, stderr = self.directory / "provider.stdout", self.directory / "provider.stderr"
        process = None
        reason = None
        observed = None
        try:
            with prompt.open("rb") as inp, stdout.open("xb") as out, stderr.open("xb") as err:
                os.fchmod(out.fileno(), 0o600)
                os.fchmod(err.fileno(), 0o600)
                self._provider_usage((self.session.candidate_root, private, self.directory))
                process = subprocess.Popen(argv, cwd=self.session.candidate_root, env=environment,
                    stdin=inp, stdout=out, stderr=err, start_new_session=True, umask=0o077)
                observed = _proc(process.pid)
                deadline = time.monotonic() + max(1, self._remaining_seconds() - 60)
                while process.poll() is None:
                    self._job()
                    _need(_flock_owner(self.policy.state_root / "repair-worker.lock") == self.global_owner,
                          "original global worker disappeared during provider execution")
                    self._provider_usage((self.session.candidate_root, private, self.directory))
                    _need(self.fleet._storage_preflight(self.storage_policy, self.job_path,
                          str(self.policy.state_root)) is None, "provider storage preflight refused")
                    _need(time.monotonic() < deadline, "provider original job deadline expired")
                    try:
                        process.wait(timeout=min(10, max(.1, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                self._provider_usage((self.session.candidate_root, private, self.directory))
        except BaseException as error:
            reason = type(error).__name__ + ":" + str(error)[:500]
            if process is not None and process.poll() is None:
                current = _proc(process.pid)
                if current is not None and observed is not None and current["birth"] == observed["birth"]:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
            raise
        finally:
            _new(self.directory / "provider-execution.json", {"schema": SCHEMA,
                "argv": argv, "private_root": str(private), "stdout": str(stdout), "stderr": str(stderr),
                "provider_pid": process.pid if process is not None else None,
                "provider_birth": observed["birth"] if observed is not None else None,
                "returncode": process.returncode if process is not None else None,
                "child_reaped": process is not None and process.returncode is not None,
                "failure": reason, "poll_interval_seconds": 10,
                "candidate_and_scratch_bound": self.policy.max_candidate_bytes,
                "entire_cgroup_closed_claimed": False})
        return process

    def advance(self) -> dict[str, Any]:
        if self.session.stage == "candidate_ready":
            self._stage("validation_pending", freeze_sha256=self.session.freeze["freeze_sha256"])
            receipt = self._owner_command("validate")
            if receipt is None:
                return self._stage("validation_pending", reason="original_validation_pending_or_unknown")
            candidate.record_validation(self.session, receipt)
        if self.session.stage == "adoption_pending":
            self._stage("adoption_pending", validation_sha256=candidate.digest(self.session.validation))
            receipt = self._owner_command("adopt")
            if receipt is None:
                return self._stage("adoption_pending", reason="original_adoption_pending_or_unknown")
            candidate.record_adoption(self.session, receipt)
        return self._stage(self.session.stage, fresh_observer_probe_required=True)

    def run(self, prompt: Path) -> dict[str, Any]:
        _need(self.job["status"] == "running", "coding stage already completed; resume owner stages only")
        self.session = candidate.prepare_candidate(policy=self.policy, board_id=self.board["id"],
            attempt_id=self.token, attempt_number=self.attempt_number,
            active_root=self.policy.repository_root, authority=self.authority,
            run_command_fn=self.command_runner)
        environment = os.environ.copy()
        environment.update(candidate.provider_environment(self.session))
        route_root = self.policy.state_root / "state" / self.policy.namespace / "provider_route_receipts"
        route_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        route = route_root / ("provider-route-" + self.token + ".json")
        scratch = self.session.candidate_root / (".repair-last-message-" + self.token + ".txt")
        _need(not scratch.exists() and scratch.name not in self.policy.source_paths
              and scratch.name not in self.policy.allowed_paths, "provider scratch aliases source")
        argv = self.fleet.llm_router_repair_argv(self.config["repair_worker"], prompt, scratch,
            workspace=self.session.candidate_root)
        argv += ["--route-receipt-path", str(route), "--route-task-id", "infrastructure-repair:" + self.board["id"],
                 "--route-attempt", str(self.attempt_number), "--route-stage", "repair"]
        _read(prompt)  # Refuse aliases/oversized input before any provider process.
        result = self._run_provider(argv, prompt, environment)
        self.provider_closed = True
        boundary = route.with_name("provider-filesystem-boundary-" + self.token + ".json")
        fenced = _json(_read(boundary))
        self._boundary(fenced)
        os.chmod(boundary, 0o400)
        self._resource_guard("freeze", self.session.binding)
        if scratch.exists():
            scratch_ref = _pin(scratch)
            destination = self.session.metadata_root / "provider-last-message.txt"
            _need(not destination.exists(), "retained provider scratch already exists")
            os.rename(scratch, destination)
            _new(self.session.metadata_root / "provider-scratch-relocation.json",
                 {"source": scratch_ref, "destination": _pin(destination), "authority": False})
        _new(self.session.metadata_root / "provider-terminal.json",
             {"returncode": result.returncode, "boundary": _pin(boundary),
              "binding_sha256": candidate.digest(self.session.binding),
              "original_unit_owner": self.unit_owner,
              "known_cgroup_empty_except_original_owner": True, "agent_text_is_acceptance": False})
        frozen = candidate.freeze_candidate(self.session)
        self._stage("candidate_ready", freeze_sha256=frozen["freeze_sha256"])
        return self.advance()

    def resume(self) -> dict[str, Any]:
        _need(self.job["status"] in PENDING, "job has no frozen original attempt to resume")
        if self.job["status"] == "adopted_pending_probe":
            return {"stage": "adopted_pending_probe", "fresh_observer_probe_required": True,
                    "attempt_token": self.token, "coding_attempts_incremented": False,
                    "completion_authority": False}
        _need(hasattr(candidate, "resume_candidate"), "original candidate dead-owner recovery is unavailable")
        # No provider runs here. Original helper authenticates retained closure,
        # and the live resource guard separately requires this new cgroup empty.
        self.provider_closed = True
        self.resuming = True
        self.session = candidate.resume_candidate(policy=self.policy, board_id=self.board["id"],
            attempt_id=self.token, attempt_number=self.attempt_number,
            active_root=self.policy.repository_root, authority=self.authority,
            run_command_fn=self.command_runner)
        return self.advance()

    def publish_closed_bootstrap(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Publish the final descriptor only after the original lease actually closed."""
        _need(self.checkout_lease_closed, "cannot publish a positive original checkout closeout")
        prior = None
        if self.session is None and result["stage"] == "adopted_pending_probe":
            prior_ref = self._job(execution=False)["candidate_runtime"]["bootstrap_result"]
            prior = _json(_checked(prior_ref))
            _need(prior.get("attempt_token") == self.token
                  and type(prior.get("attempts")) is int and prior["attempts"] == self.attempt_number
                  and prior.get("stage") == "adopted_pending_probe"
                  and prior.get("original_checkout_lease_closed") is True
                  and prior.get("provider_known_closed") is True,
                  "prior adopted bootstrap closeout is not authenticated")
        body = {**result, "schema": SCHEMA, "board_id": self.board["id"],
                "attempt_token": self.token, "attempts": self.attempt_number,
                "invocation_id": self.unit_owner["invocation_id"],
                "original_checkout_lease_closed": True,
                "provider_known_closed": self.provider_closed if prior is None else True,
                "binding": self.session.binding if self.session is not None else prior.get("binding") if prior else None,
                "freeze_sha256": self.session.freeze["freeze_sha256"] if self.session and self.session.freeze
                                 else prior.get("freeze_sha256") if prior else None,
                "coding_attempts_incremented": False, "completion_authority": False}
        ref = _new(self.directory / "bootstrap-result.json", body)
        with self.fleet.lock(self.job_path.parent / "queue.lock") as acquired:
            _need(acquired, "original queue lock unavailable for final closeout")
            job = self._job(execution=False)
            _need(job["status"] == result["stage"], "candidate stage changed before final closeout")
            state = dict(job.get("candidate_runtime", {}))
            _need(state.get("attempt_token") == self.token
                  and state.get("attempt_number") == self.attempt_number,
                  "original candidate identity changed before final closeout")
            state["bootstrap_result"] = ref
            job["candidate_runtime"] = state
            self.fleet.write_json(self.job_path, job)
        return ref


def run_service_candidate(config: dict[str, Any], job_path: Path, token: str,
                          *, mode: str, prompt: Path | None = None) -> dict[str, Any]:
    """Actual inner CLI path; never invoke from the outer worker/cgroup."""
    job = _json(_read(job_path))
    board = next((b for b in config["boards"] if b["id"] == job.get("board_id")), None)
    _need(board is not None, "candidate board not in original fleet")
    runtime = None
    try:
        runtime = CandidateRuntime(config, board, job_path, token)
        with runtime:
            if mode == "run":
                _need(prompt is not None, "original prompt required")
                result = runtime.run(prompt)
            else:
                result = runtime.resume()
        ref = runtime.publish_closed_bootstrap(result)
        return {"schema": SCHEMA, "status": result["stage"], "receipt": ref,
                "attempt_token": token, "coding_attempts_incremented": False}
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        if runtime is not None:
            # Keep all partial candidate/lifecycle/owner evidence. Do not restart a provider.
            result = runtime._stage("candidate_pending", reason=type(error).__name__,
                                    detail=str(error)[:1000], original_recovery_required=True)
            ref = _new(runtime.directory / "bootstrap-result.json", result)
            return {"schema": SCHEMA, "status": "candidate_pending", "receipt": ref,
                    "attempt_token": token, "coding_attempts_incremented": False}
        return {"schema": SCHEMA, "status": "candidate_pending", "attempt_token": token,
                "provider_started": False, "reason": type(error).__name__,
                "detail": str(error)[:1000], "coding_attempts_incremented": False,
                "completion_authority": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "resume"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--attempt-token", required=True)
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args(argv)
    from .fleet_watchdog import load_config
    config = load_config(args.config)
    config["_config_path"] = str(args.config)
    try:
        result = run_service_candidate(config, args.job, args.attempt_token,
                                       mode=args.mode, prompt=args.prompt)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        # No usable original owner is not permission to fall back to direct coding.
        # The outer original worker captures this terminal and preserves its token.
        result = {"schema": SCHEMA, "status": "candidate_pending",
                  "attempt_token": args.attempt_token, "provider_started": None,
                  "reason": type(error).__name__, "detail": str(error)[:1000],
                  "coding_attempts_incremented": False, "completion_authority": False}
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
