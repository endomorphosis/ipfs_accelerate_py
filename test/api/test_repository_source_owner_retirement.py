"""Authored lifecycle state with real protected files, never native release proof.

The exact lifecycle class supplies its actual receipt method, while its closed
lease/session history is authored. No scheduler, admission, worker or provider
is used. Full filesystem custody checks run under a private home directory.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_repository_owner_audit import authored_owner_evidence
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_retirement as retirement
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_owner_lifecycle import RepositorySourceOwnerLifecycle


def _sealed(path, raw):
    path.write_bytes(raw)
    path.chmod(0o400)


@pytest.fixture
def authored():
    with tempfile.TemporaryDirectory(prefix="repository-owner-retirement-", dir=Path.home()) as name:
        root = Path(name)
        parent = root / "worker-observations"
        parent.mkdir(mode=0o750)
        parent.chmod(0o750)
        audit = json.loads(authored_owner_evidence()["audit_raw"])
        descriptor = audit["session"]
        descriptor.update(owner_pid=os.getpid(), workspace=str(root / "workspace"),
                          socket_path=str(parent / "repository.sock"))
        descriptor_sha = hashlib.sha256(protocol.canonical_json(descriptor)).hexdigest()
        audit["descriptor_sha256"] = descriptor_sha
        audit["native_child"]["owner_pid"] = os.getpid()
        audit_path = parent / ("repository-mcp-" + descriptor["session_id"] + "-audit.json")
        _sealed(audit_path, protocol.canonical_json(audit))
        owner = RepositorySourceOwnerLifecycle(repository=root / "repository", workspace=root / "workspace",
            boundary=root / "boundary.json", scheduler_state_path=root / "never-opened-native.json",
            expected_policy_sha256="a" * 64, deadline_monotonic=time.monotonic() + 60)
        owner._lease = SimpleNamespace(lease_id=audit["native_child"]["parent_lease_id"],
            parent_lease_id=None, owner_pid=os.getpid(), released=True,
            cpu_slots=1, memory_mb=512, child_process_slots=1, lease_key="AUTHORED-PRIVATE-KEY")
        owner._broker = SimpleNamespace(descriptor=deepcopy(descriptor), descriptor_sha256=descriptor_sha,
            audit_path=audit_path, boundary=owner.boundary, parent=parent)
        owner._broker_audit = audit
        owner._start_attempted = owner._cleanup_complete = True
        owner._state = "closed"
        owner._stop.set()
        target = parent / ("repository-mcp-" + descriptor["session_id"] + "-lifecycle.json")
        yield SimpleNamespace(owner=owner, parent=parent, audit=audit, audit_path=audit_path,
                              target=target, root=root)


def _retain(authored):
    return retirement.retain_repository_source_owner_retirement(authored.owner)


def test_seals_exact_public_snapshot_without_native_state_access(authored):
    before = authored.audit_path.read_bytes()
    result = _retain(authored)
    raw = authored.target.read_bytes()
    envelope = json.loads(raw)
    assert set(envelope) == {"schema", "session_id", "descriptor_sha256", "session_audit_sha256",
                            "session_audit_bytes", "lifecycle"}
    assert envelope["schema"] == retirement.SCHEMA
    assert envelope["session_audit_sha256"] == hashlib.sha256(before).hexdigest()
    assert envelope["session_audit_bytes"] == len(before)
    assert envelope["lifecycle"] == authored.owner._receipt()
    assert protocol.canonical_json(envelope) == raw
    assert stat.S_IMODE(authored.target.stat().st_mode) == 0o400
    assert authored.target.stat().st_nlink == 1
    assert authored.audit_path.read_bytes() == before
    assert not authored.owner.scheduler_state_path.exists()
    assert b"AUTHORED-PRIVATE-KEY" not in raw and b"lease_key" not in raw
    assert result == dict(schema=retirement.RETENTION_SCHEMA, status="retained", reason=None,
        name=authored.target.name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def test_exact_repeat_preserves_inode_and_bytes(authored):
    first = _retain(authored)
    before = authored.target.stat()
    raw = authored.target.read_bytes()
    assert _retain(authored) == first
    after = authored.target.stat()
    assert (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns)
    assert authored.target.read_bytes() == raw


def test_existing_different_receipt_is_never_replaced(authored):
    _sealed(authored.target, b"foreign evidence")
    before = authored.target.stat()
    with pytest.raises(ValueError):
        _retain(authored)
    assert authored.target.read_bytes() == b"foreign evidence"
    assert authored.target.stat().st_ino == before.st_ino


@pytest.mark.parametrize("value", [None, object(), SimpleNamespace(_owner_pid=os.getpid())])
def test_requires_exact_lifecycle_before_any_file_access(value, monkeypatch):
    monkeypatch.setattr(retirement.os, "open", lambda *a, **k: pytest.fail("opened evidence for wrong type"))
    with pytest.raises(ValueError, match="exact_repository_source_owner_required"):
        retirement.retain_repository_source_owner_retirement(value)


def test_lifecycle_subclass_is_refused(authored):
    class Other(RepositorySourceOwnerLifecycle):
        pass
    value = Other.__new__(Other)
    value.__dict__.update(authored.owner.__dict__)
    with pytest.raises(ValueError, match="exact_repository_source_owner_required"):
        retirement.retain_repository_source_owner_retirement(value)


@pytest.mark.parametrize("pid", [None, True, -1, "123"])
def test_foreign_or_malformed_process_identity_refused(authored, pid):
    authored.owner._owner_pid = pid
    with pytest.raises(ValueError, match="belongs_to_another_process"):
        _retain(authored)
    assert not authored.target.exists()


@pytest.mark.parametrize("field,value", [("_state", "cleanup_pending"), ("_cleanup_complete", False),
    ("_start_attempted", False), ("_admission_uncertain", True), ("_admission_inflight", True),
    ("_cleanup_error", "failed")])
def test_incomplete_or_uncertain_lifecycle_cannot_export(authored, field, value):
    setattr(authored.owner, field, value)
    with pytest.raises(ValueError):
        _retain(authored)
    assert not authored.target.exists()


@pytest.mark.parametrize("change", ["absent_parent", "unreleased_parent", "wrong_parent_pid",
    "wrong_child_parent", "wrong_child_pid", "wrong_parent_demand", "missing_session", "descriptor_changed"])
def test_parent_child_session_identity_is_required(authored, change):
    owner = authored.owner
    if change == "absent_parent": owner._lease = None
    elif change == "unreleased_parent": owner._lease.released = False
    elif change == "wrong_parent_pid": owner._lease.owner_pid += 1
    elif change == "wrong_child_parent": owner._broker_audit["native_child"]["parent_lease_id"] = "foreign"
    elif change == "wrong_child_pid": owner._broker_audit["native_child"]["owner_pid"] += 1
    elif change == "wrong_parent_demand": owner._lease.memory_mb = 1024
    elif change == "missing_session": owner._broker_audit["session"] = None
    elif change == "descriptor_changed": owner._broker.descriptor["config_sha256"] = "f" * 64
    with pytest.raises((ValueError, TypeError)):
        _retain(authored)
    assert not authored.target.exists()


def test_borrowed_parent_identity_and_renewal_gap_are_retained_without_release_claim(authored):
    authored.owner._caller_parent = SimpleNamespace(lease_id="borrowed-parent", released=False)
    authored.owner._lease.parent_lease_id = "borrowed-parent"
    authored.owner._renewal_unproven = True
    _retain(authored)
    receipt = json.loads(authored.target.read_bytes())["lifecycle"]
    assert receipt["renewal_unproven"] and receipt["borrowed_parent_present"]
    assert receipt["native_parent"]["parent_lease_id"] == "borrowed-parent"
    assert authored.owner._caller_parent.released is False
    assert receipt["worker_reaping_performed"] is receipt["worker_cleanup_verified"] is False


def test_changed_first_stopped_session_snapshot_refuses_final_lifecycle_export(authored):
    authored.owner._broker_audit["service_error"] = "EarlierFailure"
    before = authored.audit_path.read_bytes()
    with pytest.raises(ValueError, match="retained_owner_session_audit_differs"):
        _retain(authored)
    assert authored.audit_path.read_bytes() == before and not authored.target.exists()


@pytest.mark.parametrize("kind", ["audit", "retirement"])
@pytest.mark.parametrize("mutation", ["symlink", "hardlink", "writable", "directory"])
def test_unsafe_evidence_never_read_or_overwritten(authored, kind, mutation):
    path = authored.audit_path if kind == "audit" else authored.target
    if kind == "retirement": _retain(authored)
    before = path.read_bytes()
    if mutation == "symlink":
        other = authored.root / "foreign"
        _sealed(other, before)
        path.unlink()
        path.symlink_to(other)
    elif mutation == "hardlink": os.link(path, authored.root / "second-link")
    elif mutation == "writable": path.chmod(0o600)
    elif mutation == "directory":
        path.unlink()
        path.mkdir()
    with pytest.raises((ValueError, OSError)):
        _retain(authored)
    if mutation != "directory": assert path.read_bytes() == before


@pytest.mark.parametrize("mode", [0o700, 0o770, 0o777])
def test_parent_requires_exact_protected_mode(authored, mode):
    authored.parent.chmod(mode)
    with pytest.raises(ValueError):
        _retain(authored)
    assert not authored.target.exists()


def test_writable_ancestor_refused(authored):
    authored.root.chmod(0o770)
    with pytest.raises(ValueError):
        _retain(authored)
    assert not authored.target.exists()


@pytest.mark.parametrize("change", ["audit_name", "boundary", "parent"])
def test_no_arbitrary_destination_can_be_supplied_through_broker(authored, change):
    broker = authored.owner._broker
    if change == "audit_name": broker.audit_path = authored.parent / "unrelated.json"
    elif change == "boundary": broker.boundary = authored.root / "other-boundary.json"
    elif change == "parent": broker.parent = authored.root
    with pytest.raises(ValueError, match="fixed_owner_retirement_parent_required"):
        _retain(authored)
    assert not authored.target.exists()


def test_private_field_in_snapshot_refused_before_export(authored):
    authored.owner._broker_audit["lease_key"] = "never-export"
    with pytest.raises(ValueError):
        _retain(authored)
    assert not authored.target.exists()


def test_export_holds_the_existing_lifecycle_lock(authored, monkeypatch):
    original = authored.owner._receipt
    def receipt():
        assert authored.owner._lock.locked()
        return original()
    monkeypatch.setattr(authored.owner, "_receipt", receipt)
    _retain(authored)
    assert not authored.owner._lock.locked()


def test_failed_write_only_removes_own_new_file(authored, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("authored full filesystem")
    monkeypatch.setattr(retirement.os, "write", fail)
    with pytest.raises(OSError):
        _retain(authored)
    assert not authored.target.exists() and authored.audit_path.exists()


def test_failed_creation_cannot_unlink_replacement_inode(authored, monkeypatch):
    original = retirement._check_parent
    def replace(fd, path, expected, owner):
        if authored.target.exists():
            authored.target.unlink()
            _sealed(authored.target, b"foreign replacement")
            raise ValueError("authored replacement")
        original(fd, path, expected, owner)
    monkeypatch.setattr(retirement, "_check_parent", replace)
    with pytest.raises(ValueError):
        _retain(authored)
    assert authored.target.read_bytes() == b"foreign replacement"


def test_noncanonical_original_audit_is_not_rewritten(authored):
    authored.audit_path.chmod(0o600)
    before = json.dumps(authored.audit).encode()
    _sealed(authored.audit_path, before)
    with pytest.raises(ValueError, match="retained_owner_session_audit_differs"):
        _retain(authored)
    assert authored.audit_path.read_bytes() == before and not authored.target.exists()


def test_snapshot_byte_bound_refuses_before_creating_retirement(authored):
    authored.owner._broker_audit["service_error"] = "a" * retirement.MAX_BYTES
    with pytest.raises(ValueError):
        _retain(authored)
    assert not authored.target.exists()


def test_partial_writes_finish_the_exact_bounded_snapshot(authored, monkeypatch):
    original = retirement.os.write
    calls = []
    def partial(fd, value):
        calls.append(len(value))
        return original(fd, value[:97])
    monkeypatch.setattr(retirement.os, "write", partial)
    result = _retain(authored)
    raw = authored.target.read_bytes()
    assert len(calls) > 1 and all(left > right for left, right in zip(calls, calls[1:]))
    assert result["bytes"] == len(raw) and result["sha256"] == hashlib.sha256(raw).hexdigest()
