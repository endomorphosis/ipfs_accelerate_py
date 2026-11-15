"""Capacity-only composition for the bounded native metadata executor.

An inherited sealed descriptor conveys a real enclosing datasets capability.
It cannot authorize task execution, providers, proofs or completion. Original
active-parent inspection and original nested native acquire remain mandatory.
No state/credential bytes are source pins, receipts or model context.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
import time

FD_ENV = "IPFS_ACCELERATE_NATIVE_METADATA_PARENT_FD"
SCHEMA = "native-metadata-sealed-parent-capacity/v1"
AUTHORITY = {"resource_capacity_only": True, "execution_authority": False,
    "provider_authority": False, "proof_authority": False, "completion_authority": False}
FIELDS = frozenset({"schema", "authority", "owner_identity", "child_identity", "owner_uid",
    "boot_id", "native_lease_token", "scheduler_policy_sha256", "disk_ledger_path",
    "disk_reservation_id", "expected_phase_lease_id", "storage_roots", "request_pin",
    "expected_request_sha256", "execution_request_pin", "owner_source", "attempt_identity",
    "sealed_work_deadline_monotonic", "sealed_native_deadline_monotonic", "allocated_envelope"})
_DEMAND_FIELDS = frozenset({"cpu_slots", "memory_mb", "child_process_slots", "storage_bytes"})
_CAPTURE_LOCK = threading.RLock()
_CAPTURED = None
_CAPTURE_ATTEMPTED = False
_CAPTURE_FAILED = False
_OPERATION = threading.local()


class NativeMetadataCapacityError(RuntimeError):
    """A fixed, credential-free capacity failure before any task authority."""


class NativeMetadataCapacityDeferred(NativeMetadataCapacityError):
    def __init__(self, reason):
        self.reason = str(reason)
        super().__init__("native_metadata_capacity_deferred:" + self.reason)


def _need(value, reason):
    if not value:
        raise NativeMetadataCapacityError(reason)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        _need(key not in value, "duplicate_parent_capacity_field")
        value[key] = item
    return value


def _identity(pid):
    raw = Path("/proc/" + str(pid) + "/stat").read_text()
    return {"pid": pid, "birth": raw.rsplit(")", 1)[1].split()[19]}


def _path(value):
    _need(type(value) is str and value and len(value) <= 4096
          and "\x00" not in value, "invalid_parent_capacity_path")
    path = Path(value)
    _need(path.is_absolute() and ".." not in path.parts
          and not any(part.is_symlink() for part in (path, *path.parents)),
          "aliased_parent_capacity_path")
    return path


def _pin(value):
    _need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
          and type(value["bytes"]) is int and 0 < value["bytes"] <= 4 * 1024**2
          and type(value["sha256"]) is str and len(value["sha256"]) == 64
          and all(c in "0123456789abcdef" for c in value["sha256"]),
          "invalid_parent_capacity_pin")
    _path(value["path"])
    return value


def _decode_fd(fd):
    _need(type(fd) is int and fd > 2, "invalid_parent_capacity_fd")
    info = os.fstat(fd)
    seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
    _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 0
          and stat.S_IMODE(info.st_mode) == 0o400 and 0 < info.st_size <= 8192
          and os.readlink("/proc/self/fd/" + str(fd)).startswith("/memfd:native-metadata-parent-capacity-v1")
          and fcntl.fcntl(fd, fcntl.F_GET_SEALS) & seals == seals,
          "unsealed_parent_capacity_descriptor")
    os.set_inheritable(fd, False)
    raw = os.pread(fd, info.st_size + 1, 0)
    _need(len(raw) == info.st_size, "parent_capacity_descriptor_truncated")
    value = json.loads(raw, object_pairs_hook=_pairs,
        parse_constant=lambda _: _need(False, "nonfinite_parent_capacity_json"))
    _need(type(value) is dict and set(value) == FIELDS and value["schema"] == SCHEMA
          and value["authority"] == AUTHORITY
          and all(type(item) is bool for item in value["authority"].values()),
          "parent_capacity_closed_schema_required")
    _need(value["owner_uid"] == os.geteuid() and type(value["owner_uid"]) is int
          and value["child_identity"] == _identity(os.getpid())
          and value["owner_identity"] == _identity(os.getppid())
          and value["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
          "parent_capacity_process_binding_changed")
    for identity in (value["owner_identity"], value["child_identity"]):
        _need(type(identity) is dict and set(identity) == {"pid", "birth"}
              and type(identity["pid"]) is int and identity["pid"] > 0
              and type(identity["birth"]) is str and identity["birth"].isdecimal(),
              "invalid_parent_capacity_identity")
    token = value["native_lease_token"]
    _need(type(token) is dict and set(token) == {"lease_id", "lease_key", "state_path"}
          and all(type(item) is str and 0 < len(item) <= 4096 for item in token.values()),
          "native_parent_capability_required")
    for name in ("disk_ledger_path",):
        _path(value[name])
    _path(token["state_path"])
    _need(type(value["storage_roots"]) is list and 1 <= len(value["storage_roots"]) <= 8,
          "bounded_parent_storage_roots_required")
    for path in value["storage_roots"]:
        _path(path)
    for name in ("request_pin", "execution_request_pin", "owner_source"):
        _pin(value[name])
    _need(value["expected_request_sha256"] == value["request_pin"]["sha256"],
          "parent_capacity_request_digest_differs")
    envelope = value["allocated_envelope"]
    _need(type(envelope) is dict and set(envelope) == _DEMAND_FIELDS
          and all(type(item) is int and item > 0 for item in envelope.values()),
          "parent_capacity_envelope_invalid")
    deadline = value["sealed_work_deadline_monotonic"]
    _need(type(deadline) in (int, float) and math.isfinite(deadline)
          and time.monotonic() < deadline
          and type(value["sealed_native_deadline_monotonic"]) in (int, float)
          and math.isfinite(value["sealed_native_deadline_monotonic"])
          and deadline <= value["sealed_native_deadline_monotonic"],
          "parent_capacity_deadline_expired")
    after = os.fstat(fd)
    _need((after.st_dev, after.st_ino, after.st_mode, after.st_uid, after.st_nlink, after.st_size)
          == (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink, info.st_size),
          "parent_capacity_descriptor_changed")
    return value, hashlib.sha256(raw).hexdigest()


def _current_parent():
    global _CAPTURE_ATTEMPTED, _CAPTURED, _CAPTURE_FAILED
    with _CAPTURE_LOCK:
        if _CAPTURE_ATTEMPTED:
            _need(not _CAPTURE_FAILED, "explicit_parent_capacity_capture_failed")
            return _CAPTURED
        text = os.environ.pop(FD_ENV, None)
        _CAPTURE_ATTEMPTED = True
        if text is None:
            return None
        # An explicit but invalid transport poisons this process. It must
        # never downgrade a later call to an unrelated root A admission.
        _CAPTURE_FAILED = True
        _need(type(text) is str and text.isdecimal() and len(text) <= 10,
              "explicit_parent_capacity_descriptor_required")
        fd = int(text)
        try:
            value, digest = _decode_fd(fd)
        finally:
            if fd > 2:
                os.close(fd)
        _CAPTURED = _ParentEnvelope(value, digest)
        _CAPTURE_FAILED = False
        return _CAPTURED


@contextmanager
def _operation(envelope, kind, *, work_deadline=None):
    previous = getattr(_OPERATION, "value", None)
    _OPERATION.value = (envelope, kind)
    try:
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import resource_admission_deadline, resource_cleanup_deadline
        value = envelope._value
        scope = (resource_cleanup_deadline(native_deadline_monotonic=value["sealed_native_deadline_monotonic"])
                 if kind == "release" else resource_admission_deadline(
                     work_deadline_monotonic=min(value["sealed_work_deadline_monotonic"], work_deadline)
                         if work_deadline is not None else value["sealed_work_deadline_monotonic"],
                     native_deadline_monotonic=value["sealed_native_deadline_monotonic"]))
        with scope:
            yield
    finally:
        _OPERATION.value = previous


@dataclass(repr=False)
class _DGrant:
    envelope: object
    lease: object
    scheduler: object
    demand: dict
    paths: tuple
    deadline: float
    request_id: str
    released: bool = False
    heartbeat_thread: object = None
    heartbeat_ident: int | None = None
    heartbeat_error: str | None = None


class _Deadline:
    def __init__(self, deadline):
        self.deadline = deadline
    def is_set(self):
        return time.monotonic() >= self.deadline


class _ParentEnvelope:
    def __init__(self, value, digest):
        self._value = value
        self.digest = digest
        self.identity = _identity(os.getpid())
        self.pid = os.getpid()
        self._authenticated = False
        self._grants = {}

    def inspect(self, demand, paths, deadline=None):
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLeaseToken
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_parented_resource_inspection import inspect_active_parented_reservation
        # These are ordinary public imports through the active raw-source
        # observer; the IO guard authenticates exact original module objects.
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_resource_recovery, autoencoder_daemon_resources, existing_resource_scheduler
        from .native_metadata_parent_io import authenticate_parent_capacity_io_sources
        authenticate_parent_capacity_io_sources()
        value = self._value
        _need(_identity(os.getpid()) == self.identity, "parent_capacity_must_not_cross_fork")
        self._authenticated = False
        with _operation(self, "inspect", work_deadline=deadline):
            result = inspect_active_parented_reservation(value["disk_ledger_path"],
                roots=tuple(Path(p) for p in value["storage_roots"]),
                reservation_id=value["disk_reservation_id"],
                execution_request_pin=value["execution_request_pin"],
                expected_request_sha256=value["expected_request_sha256"],
                owner_identity=value["owner_identity"], child_identity=value["child_identity"],
                native_lease_token=ResourceLeaseToken(**value["native_lease_token"]),
                expected_phase_lease_id=value["expected_phase_lease_id"],
                expected_scheduler_policy_sha256=value["scheduler_policy_sha256"],
                demand=demand, writable_paths=paths,
                sealed_work_deadline_monotonic=min(value["sealed_work_deadline_monotonic"], deadline)
                    if deadline is not None else value["sealed_work_deadline_monotonic"],
                expected_boot_id=value["boot_id"], lock_timeout_seconds=2)
        _need(result["binding_verified"] is True
              and result["native_capability_authenticated"] is True,
              "enclosing_parent_capacity_not_authenticated")
        self._authenticated = True
        return result

    def _heartbeat(self, grant, interval):
        # Original loop, original TTL and original stop Event. Only this held
        # original child and same-process thread receive an IO operation scope.
        grant.heartbeat_ident = threading.get_ident()
        try:
            with _operation(self, "renew", work_deadline=grant.deadline):
                grant.lease._heartbeat_loop(interval)
        except BaseException as error:
            grant.heartbeat_error = type(error).__name__

    def acquire(self, demand, paths, deadline, request_id):
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLeaseToken, ResourceLane
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
        inspection = self.inspect(demand, paths, deadline)
        _need(inspection["binding_verified"] is True
              and inspection["native_capability_authenticated"] is True,
              "enclosing_parent_capacity_not_authenticated")
        _need(inspection["effective_work_deadline_monotonic"] > time.monotonic(),
              "parent_capacity_live_ancestor_horizon_expired")
        value = self._value
        # Ancestor expires_at is renewable, not the immutable operation clock.
        effective = min(deadline, value["sealed_work_deadline_monotonic"])
        _need(effective > time.monotonic(), "parent_capacity_deadline_expired")
        with _operation(self, "allocate", work_deadline=effective):
            scheduler = open_existing_scheduler(value["native_lease_token"]["state_path"],
                                                value["scheduler_policy_sha256"])
            original_policy = scheduler.config.persisted_dict()
            automatic = scheduler.config.auto_renew_leases
            # Operational-only switch; the persisted safety policy and TTL
            # remain exact. The original loop is enrolled below with a scope.
            scheduler.config.auto_renew_leases = False
            _need(scheduler.config.persisted_dict() == original_policy,
                  "native_child_persisted_policy_changed")
            lease = scheduler.acquire(lane=ResourceLane.VALIDATION,
                parent_lease=ResourceLeaseToken(**value["native_lease_token"]),
                cpu_slots=demand["cpu_slots"], memory_mb=demand["memory_mb"],
                child_process_slots=demand["child_process_slots"], requires_gpu=False,
                timeout=min(2.0, effective - time.monotonic()),
                cancel_event=_Deadline(effective), request_id=request_id)
        grant = _DGrant(self, lease, scheduler, dict(demand), tuple(paths), effective, request_id)
        self._grants[lease.lease_id] = grant
        try:
            if automatic:
                interval = max(0.001, min(30.0, scheduler.config.lease_ttl_seconds / 3.0))
                grant.heartbeat_thread = threading.Thread(target=self._heartbeat,
                    args=(grant, interval), name="native-metadata-resource-" + lease.lease_id[:8],
                    daemon=True)
                grant.heartbeat_thread.start()
            self.check(grant)
        except BaseException:
            # This is still before preparation returns, the task journal, and
            # any scheduler borrow. The newly returned original child is owned.
            self.release(grant)
            raise
        return grant

    def check(self, grant):
        _need(type(grant) is _DGrant and grant.envelope is self and not grant.released
              and grant.lease.owner_pid == os.getpid() and not grant.lease.released
              and self._grants.get(grant.lease.lease_id) is grant
              and grant.heartbeat_error is None
              and (grant.heartbeat_thread is None or grant.heartbeat_thread.is_alive())
              and time.monotonic() < grant.deadline,
              "prepared_native_child_grant_not_live")
        inspection = self.inspect(grant.demand, grant.paths, grant.deadline)
        _need(inspection["binding_verified"] is True
              and inspection["native_capability_authenticated"] is True,
              "prepared_native_parent_binding_changed")
        with _operation(self, "inspect", work_deadline=grant.deadline):
            state = grant.scheduler.inspect_lease(grant.lease.lease_id,
                expected_request_id=grant.request_id, expected_lane="validation",
                expected_owner_pid=os.getpid(), expected_owner_birth_marker=_identity(os.getpid())["birth"],
                bounded_lock_timeout=2)
        _need(state["status"] == "present" and state["binding_verified"] is True
              and state["owner_birth_binding_verified"] is True
              and not any(state["lease"][k] for k in ("expired", "cancelled", "release_requested")),
              "prepared_native_child_lease_changed")

    def release(self, grant):
        _need(type(grant) is _DGrant and grant.envelope is self,
              "foreign_prepared_native_child_grant")
        if grant.released:
            return
        with _operation(self, "release"):
            released = grant.lease.release()
            _need(released is True, "original_D_child_release_not_confirmed")
        if grant.heartbeat_thread is not None:
            remaining = self._value["sealed_native_deadline_monotonic"] - time.monotonic()
            _need(remaining > 0, "original_D_child_cleanup_deadline_expired")
            grant.heartbeat_thread.join(timeout=min(2.0, remaining))
            _need(not grant.heartbeat_thread.is_alive(), "original_D_child_heartbeat_stop_unknown")
        grant.released = True
        self._grants.pop(grant.lease.lease_id, None)

    def provenance(self):
        value = self._value
        return {"schema": "native-metadata-prepared-parent-provenance/v1",
            "authority": dict(AUTHORITY), "transport_sha256": self.digest,
            "request_sha256": value["expected_request_sha256"],
            "execution_request_sha256": value["execution_request_pin"]["sha256"],
            "disk_reservation_id": value["disk_reservation_id"],
            "native_parent_lease_id": value["native_lease_token"]["lease_id"],
            "active_parent_and_child_suballocation_required": True,
            "new_A_host_reservation": False}


@dataclass(repr=False)
class _AGrant:
    probe: object
    lease: object
    pid: int
    deadline: float
    released: bool = False


def make_prepared_validation_admission(*, task_cid, contract_digest, workspace_path,
                                      commands, budget, deadline_monotonic,
                                      demand, writable_paths):
    from ..validation.validation_scheduler import PreparedValidationAdmission, ValidationScheduler
    from ..validation.validation_commands import build_validation_commands
    _need(type(demand) is dict and set(demand) == _DEMAND_FIELDS
          and all(type(v) is int and v > 0 for v in demand.values()),
          "prepared_metadata_demand_invalid")
    parent = _current_parent()
    deadline = deadline_monotonic
    if parent is not None:
        first = parent.inspect(demand, writable_paths, deadline)
        _need(first["effective_work_deadline_monotonic"] > time.monotonic(),
              "parent_capacity_live_ancestor_horizon_expired")
        deadline = min(deadline, parent._value["sealed_work_deadline_monotonic"])
    _need(deadline > time.monotonic(), "metadata_preparation_deadline_expired")
    # The nominal fixed packet budget is shared verbatim across both commands.
    # The effective parent deadline is a clamp, never a new relative deadline.
    effective_budget = budget
    if parent is not None:
        request_id = "native-metadata-prepared:" + hashlib.sha256(
            (task_cid + "\x00" + contract_digest).encode()).hexdigest()
        acquire = lambda: parent.acquire(demand, tuple(writable_paths), deadline, request_id)
        check, release, provenance = parent.check, parent.release, parent.provenance()
    else:
        # Explicit caps keep the original truthful live A admission path. No
        # implicit legacy zero-host snapshot or changed disk threshold exists.
        probe = ValidationScheduler(max_workers=1, resource_budget=1,
            resource_lease_budget=effective_budget,
            resource_admission_timeout_seconds=min(2.0, max(.001, deadline - time.monotonic())))
        spec = tuple(build_validation_commands([commands[0]]))[0]
        def acquire():
            decision, lease = probe._acquire_resource(spec, workspace_path=Path(workspace_path))
            if lease is None:
                reason = ",".join(decision.reasons) or "original_A_admission_refused"
                raise NativeMetadataCapacityDeferred(reason)
            return _AGrant(probe, lease, os.getpid(), deadline)
        def check(grant):
            _need(type(grant) is _AGrant and grant.probe is probe and grant.pid == os.getpid()
                  and not grant.released and time.monotonic() < grant.deadline
                  and any(held is grant.lease for held in probe.resource_scheduler.active_leases),
                  "original_A_prepared_lease_not_live")
        def release(grant):
            _need(type(grant) is _AGrant and grant.probe is probe,
                  "foreign_original_A_prepared_lease")
            if grant.released:
                return
            held = tuple(probe.resource_scheduler.active_leases)
            _need(not any(lease.lease_id == grant.lease.lease_id and lease is not grant.lease for lease in held),
                  "original_A_lease_identity_changed")
            returned = probe.resource_scheduler.release(grant.lease, reason="native_metadata_prepared_close")
            _need(returned or not any(lease is grant.lease for lease in probe.resource_scheduler.active_leases),
                  "original_A_lease_release_unknown")
            grant.released = True
        provenance = {"schema": "native-metadata-prepared-A-provenance/v1",
            "authority": dict(AUTHORITY), "original_live_A_preclaim_admission_required": True,
            "implicit_legacy_host_snapshot": False, "parent_capacity_delegated": False}
    return PreparedValidationAdmission(task_cid=task_cid, contract_digest=contract_digest,
        workspace_path=Path(workspace_path), commands=tuple(commands), budget=effective_budget,
        deadline_monotonic=deadline, acquire=acquire, check=check, release=release,
        provenance=provenance)
