"""Real shared-scheduler subleases from a sealed, externally owned harness FD.

There is deliberately no standalone scheduler, policy or root-lease fallback.
These fixtures must run under the owner's existing qualification reservation.
The private parent token is never printed or included in test observations.
The owner qualification wrapper supplies the sealed FD and independent SHA256;
ordinary suites skip when neither is present, and incomplete routing fails shut.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import SimpleNamespace

import pytest

from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceLease, ResourceLeaseToken,
    get_global_resource_scheduler,
)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError("duplicate harness parent field")
        result[name] = value
    return result


def _harness_parent():
    selected_fd = os.environ.get("IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_FD")
    expected = os.environ.get("IPFS_ACCELERATE_VALIDATION_RESOURCE_PARENT_SHA256")
    if selected_fd is None and expected is None:
        pytest.skip("requires sealed existing-owner resource capability qualification harness")
    if (selected_fd is None or re.fullmatch(r"[0-9]{1,10}", selected_fd) is None
            or expected is None or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
        pytest.fail("resource qualification requires the sealed owner parent FD and SHA256")
    descriptor = int(selected_fd)
    before = os.fstat(descriptor)
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW
    if (fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals != seals
            or not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o400
            or not 2 <= before.st_size <= 8192):
        pytest.fail("resource qualification parent FD lacks bounded immutable custody")
    raw = os.pread(descriptor, 8193, 0)
    def identity(value):
        return tuple(getattr(value, name) for name in ("st_dev", "st_ino", "st_mode", "st_nlink",
            "st_size", "st_uid", "st_gid", "st_mtime_ns", "st_ctime_ns"))
    if len(raw) != before.st_size or identity(os.fstat(descriptor)) != identity(before) or sha(raw) != expected:
        pytest.fail("resource qualification parent bytes differ from the independent pin")
    value = json.loads(raw, object_pairs_hook=_unique,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite parent JSON")))
    if (type(value) is not dict or set(value) != {"schema", "token", "policy_sha256"}
            or value["schema"] != "candidate-resource-qualification-parent@1"
            or wire(value) != raw or type(value["token"]) is not dict
            or set(value["token"]) != {"lease_id", "lease_key", "state_path"}
            or type(value["policy_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value["policy_sha256"]) is None):
        pytest.fail("resource qualification parent envelope is not the closed owner schema")
    token = value["token"]
    if any(type(token[name]) is not str or re.fullmatch(r"[0-9a-f]{32}", token[name]) is None
            for name in ("lease_id", "lease_key")):
        pytest.fail("resource qualification parent token is incomplete")
    path = Path(token["state_path"])
    if not path.is_absolute() or path.resolve(strict=True) != path:
        pytest.fail("resource qualification requires the existing canonical shared ledger")
    shared = get_global_resource_scheduler()
    if (not isinstance(shared, GlobalResourceScheduler) or shared.state_path != path
            or sha(wire(shared.config.persisted_dict())) != value["policy_sha256"]):
        pytest.fail("resource qualification scheduler differs from the independently pinned owner policy")
    return shared, ResourceLeaseToken(**token), value["policy_sha256"]


@contextmanager
def validation_parent():
    shared, token, policy_sha256 = _harness_parent()
    parent = GlobalResourceScheduler.acquire(shared, lane=ResourceLane.VALIDATION,
        parent_lease=token, cpu_slots=1, memory_mb=1024, child_process_slots=2,
        timeout=10, request_id="candidate-resource-qualification-parent")
    assert type(parent) is ResourceLease and parent.owner_pid == os.getpid()
    assert parent.parent_lease_id == token.lease_id
    try:
        yield SimpleNamespace(scheduler=shared, parent=parent, policy_sha256=policy_sha256,
            harness_parent_id=token.lease_id)
    finally:
        parent.release()


@pytest.fixture
def resource_parent():
    with validation_parent() as value:
        yield value


def lease_record(owner, lease_id):
    """Read only the selected public record, excluding the private lease key."""
    raw = owner.scheduler.state_path.read_bytes()
    assert len(raw) <= 16 * 1024 * 1024
    record = json.loads(raw)["leases"].get(lease_id)
    return None if record is None else {name: value for name, value in record.items() if name != "lease_key"}


def reservation_options(owner, worker_attempt, *, repository_id="candidate-resource-unit", **changes):
    return dict(parent_lease=owner.parent, worker_attempt=worker_attempt,
        repository_id=repository_id, cpu_slots=1, memory_mb=512,
        child_process_slots=1, threads_per_process=1) | changes
