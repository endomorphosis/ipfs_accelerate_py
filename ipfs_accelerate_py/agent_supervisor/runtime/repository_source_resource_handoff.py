"""Private resource-only delegation for the fixed repository stdio helper.

An explicit live caller lease is required. The receiver opens exactly its
existing native scheduler policy; neither side creates/reconfigures an
allocator. Grant bytes contain a capability and must not enter receipts or
model input. Only their path and digest leave the private invocation store.
This cooperative lifetime covers the synchronous helper, not escaped workers
or a hard process memory limit. A grant may be reused while its owner is live;
the native parent capacity bounds simultaneous consumers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

SCHEMA = "repository-source-private-resource-grant@1"
NAME = "resource-grant.json"
MAX_BYTES = 8192
DEMAND = {"cpu_slots": 1, "memory_mb": 512, "child_process_slots": 1}
AUTHORITY = {"resource_capacity_only": True, "execution_authority": False,
             "proof_authority": False, "completion_authority": False}
FIELDS = {"schema", "parent_token", "scheduler_policy_sha256", "owner_pid", "owner_uid",
          "owner_birth_marker", "boot_id", "deadline_monotonic", "workspace", "config_sha256",
          "helper_sha256", "invocation_dir", "directory_identity", "demand", "authority"}


def _need(value):
    if not value:
        raise ValueError("invalid_repository_source_resource_grant")


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _live_deadline(value):
    # Compare before subtraction or float conversion. A JSON integer can fit
    # the bounded grant yet overflow math.isfinite/float; ordered comparison
    # safely rejects it, as well as NaN/infinity and Boolean lookalikes.
    _need(type(value) in (int, float))
    now = time.monotonic()
    _need(now < value <= now + 3600)
    return value


def _path(value):
    _need(type(value) is str and 0 < len(value.encode("utf-8")) <= 4096
          and not any(ord(char) < 32 for char in value))
    path = Path(value)
    _need(path.is_absolute() and str(path) == value and ".." not in path.parts)
    return path


def _boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _birth(pid):
    # Linux /proc stat's comm may contain spaces and parentheses. starttime is
    # field22, after the final closing parenthesis and state(field3).
    text = Path(f"/proc/{pid}/stat").read_text()
    return text.rsplit(")", 1)[1].split()[19]


def _directory(path):
    _need(path.resolve(strict=True) == path)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
              and stat.S_IMODE(info.st_mode) == 0o700)
        current = path.lstat()
        _need((current.st_dev, current.st_ino) == (info.st_dev, info.st_ino))
        return descriptor, (info.st_dev, info.st_ino)
    except BaseException:
        os.close(descriptor)
        raise


def _read(descriptor, name, *, mode, maximum):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
              and stat.S_IMODE(before.st_mode) == mode and before.st_nlink == 1
              and 0 < before.st_size <= maximum)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(maximum + 1)
        after = os.fstat(fd)
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        _need(len(data) == before.st_size and identity(before) == identity(after) == identity(current))
        return data
    finally:
        os.close(fd)


def validate_resource_reference(value, *, invocation_dir, materialized=False):
    """Validate a public path/digest pair without revealing capability bytes."""
    _need(type(value) is dict and set(value) == {"path", "sha256"} and _digest(value["sha256"]))
    directory = _path(str(invocation_dir))
    _need(_path(value["path"]) == directory / NAME)
    if materialized:
        descriptor, _identity = _directory(directory)
        try:
            _need(_sha(_read(descriptor, NAME, mode=0o400, maximum=MAX_BYTES)) == value["sha256"])
        finally:
            os.close(descriptor)
    return {"path": value["path"], "sha256": value["sha256"]}


def validate_owner_resources(*, parent_lease, expected_policy_sha256, deadline_monotonic):
    """Validate explicit trusted controls before creating an invocation store."""
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLease
    _need(type(parent_lease) is ResourceLease and parent_lease.owner_pid == os.getpid()
          and not parent_lease.released and _digest(expected_policy_sha256))
    _live_deadline(deadline_monotonic)
    _need(not parent_lease.cancelled)
    _need(all(getattr(parent_lease, key) >= value for key, value in DEMAND.items()))


def install_repository_source_resources(binding, *, parent_lease, expected_policy_sha256,
                                        deadline_monotonic):
    """Install a private capability after the ordinary runtime store exists.

    The binding is updated only after writing and rechecking the original
    owner. The caller owns cleanup of the invocation store and parent lease.
    """
    from .repository_source_policy import validate_binding
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler

    _need(type(binding) is dict and "resource_grant" not in binding)
    validate_owner_resources(parent_lease=parent_lease, expected_policy_sha256=expected_policy_sha256,
                             deadline_monotonic=deadline_monotonic)
    validate_binding(binding, materialized=True)
    token = parent_lease.token
    shared = open_existing_scheduler(token.state_path, expected_policy_sha256)
    _need(not shared.is_cancelled(token.lease_id, missing_is_cancelled=True))
    descriptor, identity = _directory(Path(binding["invocation_dir"]))
    try:
        value = dict(schema=SCHEMA, parent_token=asdict(token), scheduler_policy_sha256=expected_policy_sha256,
            owner_pid=os.getpid(), owner_uid=os.geteuid(), owner_birth_marker=_birth(os.getpid()),
            boot_id=_boot(), deadline_monotonic=deadline_monotonic, workspace=binding["workspace"],
            config_sha256=binding["config_sha256"], helper_sha256=binding["helper_sha256"],
            invocation_dir=binding["invocation_dir"], directory_identity=list(identity),
            demand=dict(DEMAND), authority=dict(AUTHORITY))
        raw = _wire(value)
        _need(len(raw) <= MAX_BYTES)
        target = os.open(NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
        try:
            with os.fdopen(target, "wb", closefd=False) as stream:
                stream.write(raw); stream.flush(); os.fsync(target)
            os.fchmod(target, 0o400)
        finally:
            os.close(target)
        os.fsync(descriptor)
        validate_owner_resources(parent_lease=parent_lease, expected_policy_sha256=expected_policy_sha256,
                                 deadline_monotonic=deadline_monotonic)
        _need(not shared.is_cancelled(token.lease_id, missing_is_cancelled=True))
        reference = {"path": str(Path(binding["invocation_dir"]) / NAME), "sha256": _sha(raw)}
        validate_resource_reference(reference, invocation_dir=binding["invocation_dir"], materialized=True)
        binding["resource_grant"] = reference
        return dict(reference)
    finally:
        os.close(descriptor)


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        _need(key not in value)
        value[key] = item
    return value


def _load(*, artifact, expected_sha256, config_sha256, workspace):
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLeaseToken
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler

    path = _path(str(artifact))
    directory = path.parent
    validate_resource_reference({"path": str(path), "sha256": expected_sha256}, invocation_dir=directory)
    _need(_digest(config_sha256) and re.fullmatch(r"/tmp/supervisor-repository-source-[0-9a-f]{32}", str(directory)))
    descriptor, identity = _directory(directory)
    try:
        raw = _read(descriptor, NAME, mode=0o400, maximum=MAX_BYTES)
        _need(_sha(raw) == expected_sha256)
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: _need(False))
        _need(type(value) is dict and set(value) == FIELDS and value["schema"] == SCHEMA
              and value["invocation_dir"] == str(directory) and value["directory_identity"] == list(identity)
              and type(value["directory_identity"]) is list
              and all(type(item) is int for item in value["directory_identity"])
              and type(value["owner_pid"]) is int and value["owner_pid"] > 0
              and type(value["owner_uid"]) is int and value["owner_uid"] == os.geteuid()
              and type(value["owner_birth_marker"]) is str and value["owner_birth_marker"] == _birth(value["owner_pid"])
              and value["boot_id"] == _boot() and value["authority"] == AUTHORITY
              and type(value["authority"]) is dict and all(type(item) is bool for item in value["authority"].values())
              and type(value["demand"]) is dict and value["demand"] == DEMAND
              and all(type(item) is int for item in value["demand"].values())
              and _path(value["workspace"]) == _path(str(workspace))
              and value["config_sha256"] == config_sha256 and _digest(value["helper_sha256"])
              and _digest(value["scheduler_policy_sha256"]))
        _live_deadline(value["deadline_monotonic"])
        config_raw = _read(descriptor, "config.json", mode=0o600, maximum=16_384)
        _need(_sha(config_raw) == config_sha256)
        config = json.loads(config_raw, object_pairs_hook=_pairs)
        _need(type(config) is dict and config.get("workspace") == value["workspace"]
              and config.get("state_dir") == str(directory / "state")
              and config.get("audit_store") == str(directory / "operations.jsonl"))
        helper = Path(__file__).with_name("repository_source_mcp.py")
        _need(helper.resolve(strict=True) == helper and helper.stat().st_size <= 256_000
              and _sha(helper.read_bytes()) == value["helper_sha256"])
        record = value["parent_token"]
        _need(type(record) is dict and set(record) == {"lease_id", "lease_key", "state_path"}
              and all(type(item) is str and 0 < len(item) <= 4096 for item in record.values()))
        _path(record["state_path"])
        token = ResourceLeaseToken(**record)
        shared = open_existing_scheduler(token.state_path, value["scheduler_policy_sha256"])
        rows = [row for row in shared.active_leases() if row["lease_id"] == token.lease_id]
        _need(len(rows) == 1 and rows[0]["owner_pid"] == value["owner_pid"]
              and rows[0].get("owner_birth_marker") == value["owner_birth_marker"]
              and rows[0].get("owner_boot_id") == value["boot_id"]
              and not shared.is_cancelled(token.lease_id, missing_is_cancelled=True))
        return value, token, shared
    finally:
        os.close(descriptor)


class _DeadlineSignal:
    def __init__(self, deadline, native=None):
        self.deadline, self.native = deadline, native

    def is_set(self):
        return (time.monotonic() >= self.deadline or
                self.native is not None and (self.native.released or self.native.cancelled))


class DelegatedRepositorySourceResources:
    def __init__(self, lease, deadline):
        self.native = lease
        self.deadline_monotonic = deadline
        self._signal = _DeadlineSignal(deadline, lease)

    def native_options(self):
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
        if self._signal.is_set():
            raise LeaseCancelledError("repository_source_resources_expired_or_cancelled")
        return dict(parent_lease=self.native, cancel_event=self._signal,
                    deadline_monotonic=self.deadline_monotonic)


@contextmanager
def delegated_repository_source_resources(*, artifact, expected_sha256, config_sha256, workspace):
    """Own one delegated lease until the synchronous server has stopped.

    The caller must close its session before leaving this context. Native
    release still defers accounting for any remaining child leases. Grant
    parsing failures never fall back to independent/default admission.
    """
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
    value, token, shared = _load(artifact=artifact, expected_sha256=expected_sha256,
                                config_sha256=config_sha256, workspace=workspace)
    deadline = value["deadline_monotonic"]
    with shared.acquire(lane=ResourceLane.VALIDATION, parent_lease=token, **DEMAND,
            timeout=min(10.0, max(0.0, deadline - time.monotonic())),
            cancel_event=_DeadlineSignal(deadline), request_id="repository-source:" + expected_sha256) as lease:
        phase = DelegatedRepositorySourceResources(lease, deadline)
        phase.native_options()
        yield phase
