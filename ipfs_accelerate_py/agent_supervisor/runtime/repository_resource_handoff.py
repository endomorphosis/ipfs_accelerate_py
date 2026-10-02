"""Private native parent-lease delegation to an independently admitted worker.

The grant conveys bounded resource capacity only. It never conveys repository,
task, proof or completion authority and must never enter model context or logs.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from .repository_resource_bridge import RepositoryHostReservation, RepositoryPhaseDemand
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    ResourceLeaseToken, ResourceLane, get_global_resource_scheduler, LeaseTimeoutError,
)

SCHEMA = "repository-private-resource-grant@1"
FIELDS = {"schema", "repository_id", "task_cid", "phase", "parent_token", "owner_pid",
          "boot_id", "expires_monotonic", "state_path", "authority"}
AUTHORITY = {"resource_capacity_only": True, "execution_authority": False,
             "proof_authority": False, "completion_authority": False}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _directory(path):
    path = Path(path)
    _require(path.is_absolute() and path.resolve(strict=True) == path,
             "canonical private resource grant directory required")
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o077,
             "resource grant directory must be private to its owner")
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def write_repository_resource_grant(*, envelope, directory, task_cid, demand):
    """Create an exclusive private capability file; return only its digest/path."""
    _require(type(envelope) is RepositoryHostReservation and type(demand) is RepositoryPhaseDemand
        and demand.phase == "validation", "live native parent and validation-capable worker demand required")
    _require(type(task_cid) is str and 0 < len(task_cid) <= 256, "bounded task binding required")
    envelope.remaining()
    for name in ("cpu_slots", "memory_mb", "process_slots", "threads_per_process", "disk_bytes"):
        _require(getattr(demand, name) <= getattr(envelope.budget, name), "worker demand exceeds parent " + name)
    token = asdict(envelope.native.token)
    value = dict(schema=SCHEMA, repository_id=envelope.repository_id, task_cid=task_cid,
        phase=asdict(demand), parent_token=token, owner_pid=os.getpid(), boot_id=_boot(),
        expires_monotonic=envelope.deadline, state_path=str(envelope.shared.state_path), authority=AUTHORITY)
    raw = _wire(value); digest = hashlib.sha256(raw).hexdigest()
    fd = _directory(directory)
    name = digest + ".json"
    try:
        target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        try:
            with os.fdopen(target, "wb", closefd=False) as out:
                out.write(raw); out.flush(); os.fsync(target)
            os.fchmod(target, 0o400)
        finally:
            os.close(target)
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(schema=SCHEMA, artifact=str(Path(directory)/name), sha256=digest,
        parent_lease_id=envelope.native.lease_id, task_cid=task_cid,
        sensitive_capability_file=True, **AUTHORITY)


def _load(path, digest, repository_id, task_cid):
    path = Path(path)
    _require(path.name == digest + ".json" and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest), "exact resource grant digest required")
    fd = _directory(path.parent)
    try:
        source = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            info = os.fstat(source)
            _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o400 and info.st_size <= 8192,
                "immutable private bounded resource grant required")
            with os.fdopen(source, "rb", closefd=False) as stream:raw = stream.read(8193)
        finally:os.close(source)
    finally:os.close(fd)
    _require(hashlib.sha256(raw).hexdigest() == digest, "resource grant digest differs")
    def unique(pairs):
        result = {}
        for key,value in pairs:
            _require(key not in result, "duplicate resource grant field")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    _require(type(value) is dict and set(value) == FIELDS and value["schema"] == SCHEMA
        and value["repository_id"] == repository_id and value["task_cid"] == task_cid
        and value["authority"] == AUTHORITY and value["boot_id"] == _boot(), "resource grant binding differs")
    _require(type(value["expires_monotonic"]) in (int,float) and math.isfinite(value["expires_monotonic"])
        and 0 < value["expires_monotonic"]-time.monotonic() <= 3600,
        "resource grant deadline expired or malformed")
    demand = RepositoryPhaseDemand(**value["phase"])
    _require(demand.phase == "validation", "worker resource phase differs")
    token = ResourceLeaseToken(**value["parent_token"])
    shared = get_global_resource_scheduler()
    _require(token.state_path == value["state_path"] == str(shared.state_path),
             "resource grant belongs to another host admission authority")
    rows = [row for row in shared.active_leases() if row["lease_id"] == token.lease_id]
    _require(len(rows) == 1 and rows[0]["owner_pid"] == value["owner_pid"]
        and not rows[0].get("parent_lease_id"), "resource grant parent is no longer live")
    return value, demand, token, shared


class DelegatedRepositoryPhase:
    def __init__(self, value, demand, lease):
        self._value, self.demand, self.native = value, demand, lease

    def remaining(self):
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
        if self.native.cancelled:raise LeaseCancelledError("delegated repository phase cancelled")
        left = self._value["expires_monotonic"]-time.monotonic()
        if left <= 0:raise LeaseTimeoutError("delegated repository phase deadline expired")
        return left

    def native_options(self):
        return dict(parent_lease=self.native, cancel_event=self.native.cancellation_signal,
            timeout_seconds=min(90.,self.remaining()))

    def receipt(self):
        return dict(schema="repository-delegated-resource-phase@1", repository_id=self._value["repository_id"],
            task_cid=self._value["task_cid"], demand=asdict(self.demand), lease=self.native.to_dict(),
            parent_lease_id=self._value["parent_token"]["lease_id"], private_token_disclosed=False, **AUTHORITY)


@contextmanager
def delegated_repository_phase(*, artifact, expected_sha256, repository_id, task_cid):
    value, demand, token, shared = _load(artifact, expected_sha256, repository_id, task_cid)
    # Native token validation and capacity admission remain the sole host owner.
    with shared.acquire(parent_lease=token, lane=ResourceLane.VALIDATION,
            cpu_slots=demand.cpu_slots, memory_mb=demand.memory_mb,
            child_process_slots=demand.process_slots,
            timeout=min(90.,value["expires_monotonic"]-time.monotonic()),
            request_id="repository-delegated:"+expected_sha256) as lease:
        phase = DelegatedRepositoryPhase(value,demand,lease)
        yield phase
        phase.remaining()


PIPELINE_SCHEMA = "repository-private-resource-grant@2"
PIPELINE_FIELDS = FIELDS | {"root_lease_id", "bridge_phase_lease_id", "daemon_reservation_id"}


def _read_private_version(path, digest):
    """Authenticate private file storage and bytes before examining its schema."""
    path = Path(path)
    _require(type(digest) is str and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
        and path.name == digest + ".json", "exact resource grant digest required")
    fd = _directory(path.parent)
    try:
        source = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            info = os.fstat(source)
            _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                and info.st_nlink == 1 and stat.S_IMODE(info.st_mode) == 0o400 and info.st_size <= 8192,
                "immutable private bounded resource grant required")
            with os.fdopen(source, "rb", closefd=False) as stream:
                raw = stream.read(8193)
        finally:
            os.close(source)
    finally:
        os.close(fd)
    _require(hashlib.sha256(raw).hexdigest() == digest, "resource grant digest differs")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate resource grant field")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite resource grant")))
    _require(type(value) is dict, "closed resource grant object required")
    return value


def _pipeline_ancestry(value, shared, *, consumer_id=None):
    """The canonical read recovers expired/dead ancestors before joining IDs."""
    rows = {row["lease_id"]: row for row in shared.active_leases()}
    chain = (value["root_lease_id"], value["bridge_phase_lease_id"], value["parent_token"]["lease_id"])
    _require(len(set(chain)) == 3, "pipeline resource ancestor identities must be distinct")
    for index, lease_id in enumerate(chain):
        row = rows.get(lease_id)
        _require(row is not None and not row["cancelled"] and row["owner_pid"] == value["owner_pid"]
                 and row.get("parent_lease_id") == (None if index == 0 else chain[index-1]),
                 "pipeline resource ancestor is no longer live or bound")
    _require(rows[chain[0]]["lane"] == ResourceLane.ORCHESTRATION.value
        and rows[chain[1]]["lane"] == ResourceLane.VALIDATION.value
        and rows[chain[2]]["request_id"] == "daemon:" + value["daemon_reservation_id"],
        "pipeline resource owner or protected lane differs")
    for key, native_key in (("cpu_slots", "cpu_slots"), ("memory_mb", "memory_mb"),
                            ("process_slots", "child_process_slots")):
        _require(rows[chain[1]][native_key] == rows[chain[2]][native_key] == value["phase"][key],
                 "pipeline grant differs from admitted phase capacity")
    if consumer_id is not None:
        row = rows.get(consumer_id)
        _require(row is not None and not row["cancelled"] and row.get("parent_lease_id") == chain[-1],
                 "pipeline consumer is no longer live or bound")


def write_pipeline_resource_grant(*, phase, directory, task_cid):
    """Delegate only an already admitted protected phase's daemon capacity."""
    from .repository_pipeline_resources import PipelinePhase
    _require(type(phase) is PipelinePhase and phase.demand.phase == "validation",
             "exact live protected validation phase required")
    _require(type(task_cid) is str and 0 < len(task_cid) <= 256, "bounded task binding required")
    lease = phase.native_options()["parent_lease"]
    envelope = phase.pipeline.parent
    value = dict(schema=PIPELINE_SCHEMA, repository_id=envelope.repository_id, task_cid=task_cid,
        phase=asdict(phase.demand), parent_token=asdict(lease.token), owner_pid=os.getpid(), boot_id=_boot(),
        expires_monotonic=envelope.deadline, state_path=str(envelope.shared.state_path), authority=AUTHORITY,
        root_lease_id=envelope.native.lease_id, bridge_phase_lease_id=phase.native.native.lease_id,
        daemon_reservation_id=phase.daemon.reservation_id)
    _pipeline_ancestry(value, envelope.shared)
    raw = _wire(value)
    _require(len(raw) <= 8192, "bounded private pipeline grant required")
    digest = hashlib.sha256(raw).hexdigest()
    fd = _directory(directory)
    try:
        target = os.open(digest + ".json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        try:
            with os.fdopen(target, "wb", closefd=False) as stream:
                stream.write(raw); stream.flush(); os.fsync(target)
            os.fchmod(target, 0o400)
        finally:
            os.close(target)
        os.fsync(fd)
    finally:
        os.close(fd)
    return dict(schema=PIPELINE_SCHEMA, artifact=str(Path(directory)/(digest + ".json")), sha256=digest,
        parent_lease_id=lease.lease_id, root_lease_id=envelope.native.lease_id,
        bridge_phase_lease_id=phase.native.native.lease_id, daemon_reservation_id=phase.daemon.reservation_id,
        task_cid=task_cid, sensitive_capability_file=True, **AUTHORITY)


def _load_pipeline(path, digest, repository_id, task_cid):
    value = _read_private_version(path, digest)
    _require(set(value) == PIPELINE_FIELDS and value["schema"] == PIPELINE_SCHEMA
        and value["repository_id"] == repository_id and value["task_cid"] == task_cid
        and value["authority"] == AUTHORITY and value["boot_id"] == _boot(),
        "pipeline resource grant binding differs")
    _require(type(value["expires_monotonic"]) in (int, float) and math.isfinite(value["expires_monotonic"])
        and 0 < value["expires_monotonic"]-time.monotonic() <= 3600,
        "pipeline resource grant deadline expired or malformed")
    _require(type(value["owner_pid"]) is int and value["owner_pid"] > 0
        and all(type(value[key]) is str and 0 < len(value[key]) <= 256
                for key in ("root_lease_id", "bridge_phase_lease_id", "daemon_reservation_id")),
        "bounded exact pipeline resource identities required")
    demand = RepositoryPhaseDemand(**value["phase"])
    _require(demand.phase == "validation", "pipeline worker resource phase differs")
    token = ResourceLeaseToken(**value["parent_token"])
    shared = get_global_resource_scheduler()
    _require(token.state_path == value["state_path"] == str(shared.state_path),
             "pipeline grant belongs to another host admission authority")
    _pipeline_ancestry(value, shared)
    return value, demand, token, shared


class DelegatedPipelinePhase(DelegatedRepositoryPhase):
    def __init__(self, value, demand, lease, shared):
        super().__init__(value, demand, lease)
        self.shared = shared

    def is_set(self):
        try:
            _pipeline_ancestry(self._value, self.shared, consumer_id=self.native.lease_id)
        except (ValueError, OSError):
            return True
        return self.native.cancelled

    def remaining(self):
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
        if self.is_set():
            raise LeaseCancelledError("delegated pipeline ancestry cancelled or unavailable")
        return super().remaining()

    def native_options(self):
        return dict(parent_lease=self.native, cancel_event=self,
                    timeout_seconds=min(90., self.remaining()))

    def receipt(self):
        return {**super().receipt(), "schema": "repository-delegated-pipeline-phase@1",
            "grant_schema": PIPELINE_SCHEMA, "root_lease_id": self._value["root_lease_id"],
            "bridge_phase_lease_id": self._value["bridge_phase_lease_id"],
            "daemon_reservation_id": self._value["daemon_reservation_id"],
            "external_process_rss_captured": False}


@contextmanager
def delegated_pipeline_phase(*, artifact, expected_sha256, repository_id, task_cid):
    value, demand, token, shared = _load_pipeline(artifact, expected_sha256, repository_id, task_cid)
    # This authenticates the native capability, in addition to the public
    # ancestry joins. Public lease IDs alone never admit a consumer.
    with shared.acquire(parent_lease=token, lane=ResourceLane.VALIDATION,
            cpu_slots=demand.cpu_slots, memory_mb=demand.memory_mb,
            child_process_slots=demand.process_slots,
            timeout=min(90., value["expires_monotonic"]-time.monotonic()),
            request_id="repository-pipeline-delegated:"+expected_sha256) as lease:
        phase = DelegatedPipelinePhase(value, demand, lease, shared)
        phase.remaining()
        yield phase
        phase.remaining()


@contextmanager
def delegated_supported_repository_phase(*, artifact, expected_sha256, repository_id, task_cid):
    """Closed v1/v2 dispatch, after private-file/digest verification."""
    version = _read_private_version(artifact, expected_sha256).get("schema")
    _require(version in {SCHEMA, PIPELINE_SCHEMA}, "unsupported private resource grant version")
    context = delegated_repository_phase if version == SCHEMA else delegated_pipeline_phase
    with context(artifact=artifact, expected_sha256=expected_sha256,
                 repository_id=repository_id, task_cid=task_cid) as phase:
        yield phase
