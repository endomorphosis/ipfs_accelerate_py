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
from .repair_authorization import repair_job_directory, require_current_authorization
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
    validation_admission_retry: dict[str, Any] | None = None


def parse_runtime_policy(worker: Mapping[str, Any], policy: candidate.CandidatePolicy) -> RuntimePolicy:
    value = worker.get("candidate_runtime")
    fields = {"schema", "source_descriptors", "validation_owner", "adoption_owner", "resource_bounds", "checkout_timeout_seconds"}
    _need(type(value) is dict and set(value) in (fields, fields | {"validation_admission_retry"}) and value["schema"] == SCHEMA,
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
    # Physical parent limits must accommodate unchanged native proof headroom.
    # These caps grant no extra native CPU, memory, process slots or repair attempts.
    caps = {"memory_max_bytes": 32 * 1024 ** 3, "cpu_quota_percent": 100,
            "tasks_max": 512, "runtime_max_seconds": 2400}
    for name, maximum in caps.items():
        _integer(bounds[name], maximum, name)
    validation, adoption = (_owner(value[k], stage) for k, stage in
                            (("validation_owner", "validation"), ("adoption_owner", "adoption")))
    worker_deadline = _integer(worker.get("timeout_seconds", 2400), 2400, "original worker deadline")
    effective_deadline = min(worker_deadline, bounds["runtime_max_seconds"])
    _need(all(command.timeout_seconds * 2 + 60 <= effective_deadline for command in (validation, adoption)),
          "original job deadline cannot accommodate owner plus verifier and cleanup")
    retry = None
    if "validation_admission_retry" in value:
        from .repair_candidate_owners import checked_validation_admission_retry
        retry = checked_validation_admission_retry(value["validation_admission_retry"])
        argv = validation.argv
        _need(argv.count("--policy") == argv.count("--policy-sha256") == 1, "validation retry requires an original public owner policy")
        selected = argv[argv.index("--policy") + 1]
        policies = [ref for ref in validation.source_descriptors if ref["path"] == selected]
        _need(len(policies) == 1 and policies[0]["sha256"] == argv[argv.index("--policy-sha256") + 1],
              "validation retry owner policy is unpinned")
        owner_policy = _json(_checked(policies[0]))
        _need(owner_policy["validation"].get("validation_admission_retry") == retry
              and "external_fixture_profile" in owner_policy["validation"]
              and validation.timeout_seconds >= retry["source_probe_timeout_seconds"] + owner_policy["validation"]["timeout_seconds"] + 60,
              "validation retry owner/runtime authority or original deadline differs")
    return RuntimePolicy(tuple(refs), validation, adoption, dict(bounds),
                         _integer(value["checkout_timeout_seconds"], 30, "checkout wait"), retry)


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



NEGATIVE_VALIDATION_SCHEMA = "agent-supervisor/isolated-repair-candidate-negative-validation@1"


def _checked_negative_observation(value, *, binding, freeze_sha256,
                                  source_commitment, original_returncode):
    """Check a pinned original observer's fully closed negative disposition.

    The immutable original proof may truthfully retain storage_closed=False.
    Only the separate original read-only observer can join a later genuine
    public disposition. Neither a return code nor a rewritten flag is closure.
    """
    _need(type(value) is dict and set(value) == {"schema", "negative_validation", "closure_observation"}
          and value["schema"] == "agent-supervisor/verified-negative-validation@1",
          "closed original negative observer result required")
    pin = value["negative_validation"]
    negative = _json(_checked(pin))
    _need(not Path(pin["path"]).lstat().st_mode & 0o222,
          "original negative validation proof is not immutable")
    _need(negative.get("schema") == NEGATIVE_VALIDATION_SCHEMA
          and negative.get("passed") is False and negative.get("semantic_negative") is True
          and negative.get("negative_profile") in {"trusted_pytest_call_failure", "changed_candidate_source_syntax_error"}
          and type(original_returncode) is int and original_returncode != 0
          and type(negative.get("original_owner_returncode")) is int
          and negative["original_owner_returncode"] == original_returncode
          and negative.get("binding") == binding
          and negative.get("freeze_sha256") == freeze_sha256
          and negative.get("source_commitment") == source_commitment,
          "negative validation belongs to another original frozen attempt")
    _need(all(negative.get(k) is True for k in ("source_closed", "native_closed", "processes_closed"))
          and type(negative.get("storage_closed")) is bool and negative.get("cleanup_errors") == [],
          "original negative source/native/process custody is incomplete")
    identity = negative.get("original_owner_identity")
    _need(type(identity) is dict and set(identity) == {"pid", "birth"}
          and type(identity["pid"]) is int and identity["pid"] > 0
          and type(identity["birth"]) is str and identity["birth"].isdecimal(),
          "negative validation original PID/birth required")
    for key in ("reservation_id", "native_lease_id"):
        _need(type(negative.get(key)) is str and _TOKEN.fullmatch(negative[key]) is not None,
              "negative validation original resource identity required")
    for key in ("request", "original_failure_terminal", "original_runner_wait", "trusted_semantic_report"):
        _checked(negative[key])
    closed = value["closure_observation"]
    joins = ("negative_profile", "freeze_sha256", "source_commitment", "binding", "request", "original_owner_returncode",
             "original_owner_identity", "reservation_id", "native_lease_id")
    _need(type(closed) is dict
          and closed.get("schema") == "agent-supervisor/negative-validation-closure-observation@1"
          and closed.get("negative_validation") == pin
          and all(closed.get(k) == negative[k] for k in joins)
          and all(closed.get(k) is True for k in ("source_closed", "native_closed", "processes_closed", "storage_closed"))
          and closed.get("cleanup_errors") == []
          and type(closed.get("storage_disposition")) is dict and bool(closed["storage_disposition"])
          and type(closed.get("observed_at")) in (int, float)
          and 0 < closed["observed_at"] < float("inf"),
          "original read-only observer did not prove this exact failed claim fully closed")
    return negative


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
        _need(job_path == repair_job_directory(self.policy.state_root, board) / "job.json",
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
        require_current_authorization(self.config, self.board, job=job)
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
                "candidate_metadata": str(self.session.metadata_root) if self.session else getattr(self, "retained_candidate_metadata", ""),
                "attempt_token": self.token, "attempt_number": self.attempt_number})
            self.fleet.write_json(self.job_path, job)
        return value


    def _closed_negative_validation(self, command, invocation, terminal):
        """Read-only authentication after a genuine nonzero owner terminal.

        A positive command with a refused verifier never enters this path.
        Missing or ambiguous evidence stays pending without owner replay.
        """
        code = terminal.get("command_returncode")
        if type(code) is not int or code == 0 or terminal.get("verifier_returncode") is not None:
            return None
        epoch = require_current_authorization(self.config, self.board, job=self._job())
        if epoch is None or self._remaining_seconds() < command.timeout_seconds + 30:
            return None
        values = {"{candidate_root}": str(self.session.candidate_root),
            "{freeze_path}": str(self.session.metadata_root / "candidate-ready.json"),
            "{receipt_path}": str(invocation / "receipt.json"),
            "{source_commitment}": self.session.freeze["source_commitment"],
            "{validation_path}": str(self.session.metadata_root / "adoption-pending.json"),
            "{invocation_root}": str(invocation)}
        original_argv = [values.get(v, v) for v in command.argv]
        original_verify = [values.get(v, v) for v in command.verify_argv]
        intent = _json(_read(invocation / "intent.json"))
        _need(intent.get("schema") == SCHEMA and intent.get("stage") == "validate"
              and intent.get("argv") == original_argv and intent.get("verify_argv") == original_verify
              and intent.get("owner_source_descriptors") == list(command.source_descriptors)
              and intent.get("runtime_policy_sha256") == candidate.digest(self.config["repair_worker"]["candidate_runtime"])
              and intent.get("attempt_token") == self.token
              and type(intent.get("attempt_number")) is int and intent["attempt_number"] == self.attempt_number
              and intent.get("freeze_sha256") == self.session.freeze["freeze_sha256"]
              and intent.get("source_commitment") == self.session.freeze["source_commitment"],
              "negative observer is not bound to the exact original owner invocation")
        for pin in command.source_descriptors:
            _checked(pin)
        _need(original_verify.count("verify-validation") == 1
              and "verify-negative-validation" not in original_verify,
              "one explicit maintained negative-verifier action required")
        argv = ["verify-negative-validation" if v == "verify-validation" else v for v in original_verify]
        self.authority.require_current("validate", self.session.binding)
        observed = self.command_runner(argv, cwd=self.policy.repository_root,
            timeout_seconds=command.timeout_seconds)
        witness = {"schema": "agent-supervisor/negative-validation-observer-wait@1", "argv": argv,
            "returncode": observed.returncode, "stdout": observed.stdout, "stderr": observed.stderr,
            "original_terminal": _pin(invocation / "terminal.json"),
            "original_intent": _pin(invocation / "intent.json"),
            "owner_source_descriptors": list(command.source_descriptors)}
        self.counter += 1
        witness_pin = _new(self.directory / f"negative-observer-{self.counter:04d}.json", witness)
        if type(observed.returncode) is not int or observed.returncode != 0:
            return None
        self.authority.require_current("validate", self.session.binding)
        for pin in command.source_descriptors:
            _checked(pin)
        value = _json(observed.stdout.encode())
        negative = _checked_negative_observation(value, binding=self.session.binding,
            freeze_sha256=self.session.freeze["freeze_sha256"],
            source_commitment=self.session.freeze["source_commitment"], original_returncode=code)
        _need(Path(value["negative_validation"]["path"]).parent == invocation,
              "negative proof is outside its original owner invocation")
        self.known_negative_validation = (value, witness_pin, negative)
        return negative

    def _negative_failure_transition(self, job, bootstrap_ref):
        """One closed spent attempt, same epoch and original cumulative cap."""
        _need(self.checkout_lease_closed and self.provider_closed,
              "original checkout/provider must close before a failed attempt can requeue")
        epoch = require_current_authorization(self.config, self.board, job=job)
        _need(epoch is not None and job["attempts"] == self.attempt_number
              and job["repair_unit_token"] == self.token,
              "known negative cannot mint an epoch, reset attempts or replace the original token")
        value, observation_pin, negative = self.known_negative_validation
        observation = _json(_checked(observation_pin))
        _need(observation["returncode"] == 0 and _json(observation["stdout"].encode()) == value,
              "original read-only verifier observation changed")
        _checked_negative_observation(value, binding=negative["binding"],
            freeze_sha256=negative["freeze_sha256"], source_commitment=negative["source_commitment"],
            original_returncode=negative["original_owner_returncode"])
        history = job.get("candidate_validation_failures", [])
        _need(type(history) is list and len(history) < min(self.attempt_number, 3)
              and all(type(r) is dict and type(r.get("attempt_number")) is int
                      and 0 < r["attempt_number"] < self.attempt_number for r in history)
              and len({r["attempt_number"] for r in history}) == len(history),
              "failed-attempt history is unknown, forked or exceeds the original spent cap")
        failure = {"schema": "agent-supervisor/known-closed-negative-repair-attempt@1",
            "attempt_number": self.attempt_number, "attempt_token": self.token,
            "repair_authorization_sha256": epoch.commitment,
            "negative_validation": value["negative_validation"], "original_observer_wait": observation_pin,
            "closed_bootstrap": bootstrap_ref, "candidate_metadata": str(self.session.metadata_root),
            "freeze_sha256": negative["freeze_sha256"], "source_commitment": negative["source_commitment"],
            "coding_attempts_incremented": False, "completion_authority": False}
        job["candidate_validation_failures"] = [*history, failure]
        now = time.time()
        due = job.get("next_attempt_at", 0)
        _need(type(due) in (int, float) and 0 <= due < float("inf"), "original repair cooldown is invalid")
        job.update(status="queued", returncode=negative["original_owner_returncode"], finished_at=now,
            candidate_failure_reason="known_closed_negative_validation", next_attempt_at=max(due, now + 300))
        limit = self.fleet._repair_limits(self.config)[self.board["id"]]
        _need(type(limit) is int and 0 < limit <= epoch.value["max_attempts"] <= 3,
              "known negative cannot widen the original finite repair cap")
        self.fleet._escalate_repair_budget(job, limit, now)
        self.negative_requeue_status = job["status"]
        return job

    def _validation_admission_context(self, index, previous):
        profile = self.runtime_policy.validation_admission_retry
        _need(profile is not None and type(index) is int and 0 <= index <= 2,
              "bounded validation admission retry context required")
        epoch = require_current_authorization(self.config, self.board, job=self._job())
        _need(epoch is not None, "validation admission retry requires the existing finite epoch")
        return {"schema": "agent-supervisor/validation-admission-retry-context@1",
            "index": index, "previous": previous, "profile": profile,
            "epoch_sha256": epoch.commitment, "binding": self.session.binding,
            "freeze_sha256": self.session.freeze["freeze_sha256"],
            "source_commitment": self.session.freeze["source_commitment"],
            "attempt_token": self.token, "attempt_number": self.attempt_number,
            "runtime_policy_sha256": candidate.digest(self.config["repair_worker"]["candidate_runtime"])}

    def _validation_admission_owner_policy(self, command):
        argv = command.argv
        _need(argv.count("--policy") == argv.count("--policy-sha256") == 1,
              "validation admission retry requires one original public policy")
        path, sha = argv[argv.index("--policy") + 1], argv[argv.index("--policy-sha256") + 1]
        refs = [ref for ref in command.source_descriptors if ref["path"] == path]
        _need(len(refs) == 1 and refs[0]["sha256"] == sha, "validation admission policy is not source-pinned")
        policy = _json(_checked(refs[0]))
        _need(policy["validation"].get("validation_admission_retry") == self.runtime_policy.validation_admission_retry,
              "validation admission owner/runtime profiles differ")
        return policy

    def _validation_admission_observation(self, command, invocation, terminal, *, create, require_absent=True):
        from .repair_candidate_owners import (SOURCE_ADMISSION_VERIFIED_SCHEMA, SOURCE_ADMISSION_DEFER_SCHEMA,
            require_validation_unstarted)
        code = terminal.get("command_returncode")
        if type(code) is not int or code != 75 or terminal.get("verifier_returncode") is not None:
            return None
        wait_ref = terminal.get("original_owner_wait")
        if wait_ref is None:
            return None
        wait = _json(_checked(wait_ref)); intent_ref = _pin(invocation / "intent.json")
        intent = _json(_checked(intent_ref))
        _need(all(not Path(ref["path"]).lstat().st_mode & 0o222 for ref in
              (wait_ref, intent_ref, _pin(invocation / "terminal.json"))),
              "original validation admission history must be immutable")
        values = {"{candidate_root}": str(self.session.candidate_root),
            "{freeze_path}": str(self.session.metadata_root / "candidate-ready.json"),
            "{receipt_path}": str(invocation / "receipt.json"),
            "{source_commitment}": self.session.freeze["source_commitment"],
            "{validation_path}": str(self.session.metadata_root / "adoption-pending.json"),
            "{invocation_root}": str(invocation)}
        expected_argv = [values.get(value, value) for value in command.argv]
        expected_verify = [values.get(value, value) for value in command.verify_argv]
        _need(intent.get("schema") == SCHEMA and intent.get("stage") == "validate"
              and intent.get("argv") == expected_argv and intent.get("verify_argv") == expected_verify
              and intent.get("owner_source_descriptors") == list(command.source_descriptors)
              and intent.get("runtime_policy_sha256") == candidate.digest(self.config["repair_worker"]["candidate_runtime"])
              and intent.get("attempt_token") == self.token and type(intent.get("attempt_number")) is int
              and intent["attempt_number"] == self.attempt_number
              and intent.get("freeze_sha256") == self.session.freeze["freeze_sha256"]
              and intent.get("source_commitment") == self.session.freeze["source_commitment"],
              "validation source-admission intent differs from its configured original command")
        for source in command.source_descriptors:
            _checked(source)
        _need(wait.get("schema") == "agent-supervisor/validation-owner-wait@1"
              and wait.get("argv") == expected_argv and wait.get("owning_command_timeout_seconds") == command.timeout_seconds
              and wait.get("provider_started") is False and wait.get("coding_attempts_incremented") is False
              and wait["intent"] == intent_ref and wait["actual_Popen_returned"] is True
              and wait["actual_process_waited"] is True and type(wait["actual_process_returncode"]) is int
              and wait["actual_process_returncode"] == 75 and type(wait["returncode"]) is int and wait["returncode"] == 75,
              "validation admission defer lacks the genuine original wait")
        _need(all(type(wait[key]) in (int, float) and 0 < wait[key] < float("inf") for key in
              ("command_started_monotonic", "command_finished_monotonic"))
              and wait["command_finished_monotonic"] >= wait["command_started_monotonic"],
              "validation admission owning-command clock is invalid")
        _checked(wait["engine_source"])
        _need(wait["engine_source"] in self.runtime_policy.sources,
              "validation admission wait used an unpinned active engine")
        summary = _json(wait["stdout"].encode())
        _need(summary["schema"] == "agent-supervisor/validation-source-admission-return@1",
              "nonzero validation command is not an original source-admission defer")
        deferred_ref = summary["defer"]; deferred = _json(_checked(deferred_ref))
        _need(Path(deferred_ref["path"]) == invocation / "source-admission-deferred.json"
              and not Path(deferred_ref["path"]).lstat().st_mode & 0o222
              and deferred["schema"] == SOURCE_ADMISSION_DEFER_SCHEMA
              and deferred["owner"] == wait["runner_identity"] and deferred["original_intent"] == intent_ref,
              "validation admission source proof belongs to another original process")
        policy = self._validation_admission_owner_policy(command)
        if require_absent:
            _need(deferred["unstarted_paths"] == require_validation_unstarted(policy, self.session.freeze, invocation),
                  "a D execution plan, request or attempt already exists")
        record_path = invocation / "source-admission-observation.json"
        verify = intent["verify_argv"]
        _need(verify.count("verify-validation") == 1, "one explicit validation admission observer is required")
        observer_argv = ["verify-validation-admission" if item == "verify-validation" else item for item in verify]
        if not record_path.exists():
            if not create or not require_absent or self._remaining_seconds() < command.timeout_seconds + 30:
                return None
            _need(self.command_runner is run_command,
                  "validation admission observer requires the original actual process runner")
            self.authority.require_current("validate", self.session.binding)
            observed = self.command_runner(observer_argv, cwd=self.policy.repository_root,
                timeout_seconds=command.timeout_seconds)
            if type(observed.returncode) is not int or observed.returncode != 0:
                return None
            self.authority.require_current("validate", self.session.binding)
            value = _json(observed.stdout.encode())
            now = time.time()
            _new(record_path, {"schema": "agent-supervisor/validation-admission-observation@1",
                "observer_argv": observer_argv, "observer_returncode": 0, "observer": value,
                "intent": intent_ref, "terminal": _pin(invocation / "terminal.json"),
                "original_owner_wait": wait_ref, "observed_at": now,
                "eligible_after": now + self.runtime_policy.validation_admission_retry["cooldown_seconds"],
                "coding_attempts_incremented": False, "provider_started": False})
        ref = _pin(record_path); value = _json(_checked(ref)); observed = value["observer"]
        _need(not record_path.lstat().st_mode & 0o222
              and value["schema"] == "agent-supervisor/validation-admission-observation@1"
              and value["observer_argv"] == observer_argv
              and value["intent"] == intent_ref and value["terminal"] == _pin(invocation / "terminal.json")
              and value["original_owner_wait"] == wait_ref and type(value["observer_returncode"]) is int
              and value["observer_returncode"] == 0 and value["coding_attempts_incremented"] is False
              and value["provider_started"] is False
              and observed["schema"] == SOURCE_ADMISSION_VERIFIED_SCHEMA and observed["defer"] == deferred_ref
              and observed["source_admission"] == deferred and observed["read_only"] is True
              and type(observed["D_owner_calls"]) is int and observed["D_owner_calls"] == 0
              and observed["zero_D_calls_scope"] == "public_validation_CLI_and_validation_compute_or_storage_allocation_only"
              and type(observed["original_source_lease_calls"]) is int and observed["original_source_lease_calls"] == 0
              and observed["resource_recovery_performed"] is False
              and observed["coding_attempts_incremented"] is False,
              "validation admission readonly observation changed")
        for key in ("observed_at", "eligible_after"):
            _need(type(value[key]) in (int, float) and 0 < value[key] < float("inf"),
                  "finite validation admission cooldown required")
        _need(value["eligible_after"] == value["observed_at"] + 60,
              "validation admission cooldown was reset or widened")
        return ref, value

    def _validation_invocation(self, *, create_observation=True):
        """Select one of three immutable slots; never repeat an unknown slot."""
        base = self.session.metadata_root
        names = ["original-validate-invocation", "original-validate-source-admission-retry-01",
                 "original-validate-source-admission-retry-02"]
        command = self.runtime_policy.validation_owner
        previous = None
        for index, name in enumerate(names):
            invocation = base / name
            context = self._validation_admission_context(index, previous)
            if not invocation.exists():
                _need(not any((base / following).exists() for following in names[index + 1:]),
                      "validation admission retry history has a gap or fork")
                if index and time.time() < previous["eligible_after"]:
                    self.validation_admission_wait = {"status": "backoff", "eligible_after": previous["eligible_after"],
                        "next_slot": index, "prior_observation": previous["observation"], "coding_attempts_incremented": False}
                    return None
                self.validation_admission_selected_context = context
                return invocation
            intent_path, terminal_path = invocation / "intent.json", invocation / "terminal.json"
            if not intent_path.exists() or not terminal_path.exists():
                self.validation_admission_wait = {"status": "unknown_original_invocation", "invocation": str(invocation)}
                return None  # An uncertain original owner operation is never replayed.
            intent = _json(_read(intent_path)); terminal = _json(_read(terminal_path))
            _need(not intent_path.lstat().st_mode & 0o222 and not terminal_path.lstat().st_mode & 0o222
                  and intent.get("validation_admission_retry") == context,
                  "validation admission retry chain/epoch/candidate changed")
            selected_at = intent.get("validation_admission_invocation_started_at")
            _need(type(selected_at) in (int, float) and 0 < selected_at < float("inf")
                  and (not index or selected_at >= previous["eligible_after"]),
                  "validation admission retry started before its immutable cooldown")
            self.validation_admission_selected_context = context
            if terminal.get("command_returncode") != 75:
                _need(not any((base / following).exists() for following in names[index + 1:]),
                      "non-source-deferred validation has an orphan retry successor")
                return invocation
            has_successor = any((base / following).exists() for following in names[index + 1:])
            observation = self._validation_admission_observation(command, invocation, terminal,
                create=create_observation and not has_successor, require_absent=not has_successor)
            if observation is None:
                return None
            ref, value = observation
            if index == 2:
                self.validation_admission_wait = {"status": "finite_retry_exhausted", "additional_slots_spent": 2,
                    "last_observation": ref, "coding_attempts_incremented": False}
                return None  # Two additional slots are spent; no budget is replenished.
            previous = {"invocation": str(invocation), "intent": _pin(intent_path),
                "terminal": _pin(terminal_path), "observation": ref,
                "eligible_after": value["eligible_after"]}
        return None

    def _validation_command_wait(self, argv, invocation, timeout):
        from .repair_candidate_owners import original_runner_wait
        _need(self.command_runner is run_command,
              "validation admission retry requires the original actual process runner")
        started = time.monotonic()
        completed, witness = original_runner_wait({"source_descriptors": list(self.runtime_policy.sources)},
            argv, self.policy.repository_root, timeout)
        ref = _new(invocation / "original-owner-wait.json", {"schema": "agent-supervisor/validation-owner-wait@1",
            **witness, "intent": _pin(invocation / "intent.json"), "argv": argv,
            "owning_command_timeout_seconds": timeout, "command_started_monotonic": started,
            "command_finished_monotonic": time.monotonic(),
            "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
            "completed_at": time.time(), "provider_started": False, "coding_attempts_incremented": False})
        return completed, ref


    def _owner_command(self, stage: str) -> dict[str, Any] | None:
        command = self.runtime_policy.validation_owner if stage == "validate" else self.runtime_policy.adoption_owner
        self.authority.require_current(stage, self.session.binding)
        retry = stage == "validate" and getattr(self.runtime_policy, "validation_admission_retry", None) is not None
        invocation = self._validation_invocation() if retry else self.session.metadata_root / ("original-" + stage + "-invocation")
        if invocation is None:
            return None
        if invocation.exists():
            # Never repeat an owner operation after an ambiguous or interrupted write.
            terminal = invocation / "terminal.json"
            if not terminal.exists():
                return None
            previous = _json(_read(terminal))
            if type(previous.get("command_returncode")) is not int:
                return None
            if previous["command_returncode"] != 0:
                return self._closed_negative_validation(command, invocation, previous) if stage == "validate" else None
            if type(previous.get("verifier_returncode")) is not int or previous["verifier_returncode"] != 0:
                return None  # A refused positive verifier never becomes a negative result.
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
            intent = {"schema": SCHEMA, "stage": stage,
                "argv": argv, "verify_argv": verify, "freeze_sha256": self.session.freeze["freeze_sha256"],
                "source_commitment": self.session.freeze["source_commitment"],
                "owner_source_descriptors": list(command.source_descriptors),
                "runtime_policy_sha256": candidate.digest(self.config["repair_worker"]["candidate_runtime"]),
                "attempt_token": self.token, "attempt_number": self.attempt_number}
            if retry:
                intent["validation_admission_retry"] = self.validation_admission_selected_context
                intent["validation_admission_invocation_started_at"] = time.time()
            _new(invocation / "intent.json", intent)
            owner_wait = None
            if retry:
                result, owner_wait = self._validation_command_wait(argv, invocation, command.timeout_seconds)
            else:
                result = self.command_runner(argv, cwd=self.policy.repository_root, timeout_seconds=command.timeout_seconds)
            if result.returncode != 0:
                terminal = {"command_returncode": result.returncode, "verifier_returncode": None}
                if retry:
                    terminal["original_owner_wait"] = owner_wait
                _new(invocation / "terminal.json", terminal)
                if retry and type(result.returncode) is int and result.returncode == 75:
                    self._validation_admission_observation(command, invocation, terminal, create=True)
                    return None
                return self._closed_negative_validation(command, invocation, terminal) if stage == "validate" else None
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
        if stage == "adopt":
            from .repair_candidate_owners import DEFERRED_SCHEMA, consume_deferred_adoption
            if receipt.get("schema") == DEFERRED_SCHEMA:
                self.deferred_adoption = _pin(receipt_path)
                resolved = consume_deferred_adoption(receipt, invocation)
                if resolved is None:
                    return None
                receipt, receipt_path = resolved
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
        if getattr(self.runtime_policy, "validation_admission_retry", None) is not None:
            invocation = self._validation_invocation(create_observation=False)
            _need(invocation is not None, "no closed authorized validation invocation is available")
        else:
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
                admission = getattr(self, "validation_admission_wait", None)
                if admission is not None:
                    return self._stage("validation_pending", reason="validation_source_admission_" + admission["status"],
                        validation_admission_retry=admission)
                return self._stage("validation_pending", reason="original_validation_pending_or_unknown")
            if receipt.get("schema") == NEGATIVE_VALIDATION_SCHEMA:
                return self._stage("validation_pending", reason="known_negative_waiting_original_checkout_closeout",
                    known_negative_validation=self.known_negative_validation[1])
            candidate.record_validation(self.session, receipt)
        if self.session.stage == "adoption_pending":
            self._stage("adoption_pending", validation_sha256=candidate.digest(self.session.validation))
            receipt = self._owner_command("adopt")
            if receipt is None:
                return self._stage("adoption_pending", reason="original_adoption_pending_or_unknown",
                    **({"deferred_adoption": self.deferred_adoption} if hasattr(self, "deferred_adoption") else {}))
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
        bootstrap_pin = self.job.get("candidate_runtime", {}).get("bootstrap_result")
        if self.job["status"] == "adoption_pending" and bootstrap_pin:
            bootstrap = _json(_checked(bootstrap_pin))
            pending_pin = bootstrap.get("deferred_adoption")
            if pending_pin:
                from .repair_candidate_owners import consume_deferred_adoption
                pending = _json(_checked(pending_pin))
                _need(bootstrap.get("original_checkout_lease_closed") is True
                      and bootstrap.get("provider_known_closed") is True,
                      "prior original deferred bootstrap is not closed")
                self.provider_closed = True
                self.deferred_prior_bootstrap = bootstrap_pin
                self.retained_candidate_metadata = str(self.policy.state_root / "repair-candidates" / self.board["id"] / self.token)
                resolved = consume_deferred_adoption(pending, Path(pending_pin["path"]).parent)
                if resolved is None:
                    return self._stage("adoption_pending", reason="external_actual_activation_pending",
                        deferred_adoption=pending_pin)
                receipt, receipt_path = resolved
                self.provider_closed = True
                self.accepted["adopt"] = (receipt, _pin(receipt_path))
                self.deferred_prior_bootstrap = bootstrap_pin
                self.retained_candidate_metadata = str(self.policy.state_root / "repair-candidates" / self.board["id"] / self.token)
                links = self.job.get("repair_source_successors", [])
                _need(links, "actual source successor has not been authenticated/published")
                value = candidate.observe_verified_deferred_adoption(policy=self.policy,
                    board_id=self.board["id"], attempt_id=self.token, attempt_number=self.attempt_number,
                    authority=self.authority, deferred_receipt=pending, activation_receipt=receipt,
                    source_successor=_json(_checked(links[-1])))
                return self._stage(value["stage"], deferred_adoption=pending_pin,
                    source_successor=links[-1], fresh_observer_probe_required=True)
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
        if self.session is None and (result["stage"] == "adopted_pending_probe" or hasattr(self, "deferred_prior_bootstrap")):
            prior_ref = getattr(self, "deferred_prior_bootstrap", None) or self._job(execution=False)["candidate_runtime"]["bootstrap_result"]
            prior = _json(_checked(prior_ref))
            _need(prior.get("attempt_token") == self.token
                  and type(prior.get("attempts")) is int and prior["attempts"] == self.attempt_number
                   and prior.get("stage") in {"adoption_pending", "adopted_pending_probe"}
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
            if hasattr(self, "known_negative_validation"):
                job = self._negative_failure_transition(job, ref)
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
        return {"schema": SCHEMA, "status": getattr(runtime, "negative_requeue_status", result["stage"]), "receipt": ref,
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
