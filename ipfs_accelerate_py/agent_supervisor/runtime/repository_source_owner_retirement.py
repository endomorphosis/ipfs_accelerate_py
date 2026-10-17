"""Seal one completed source owner's public lifecycle receipt, without state access.

Only the existing, same-process lifecycle can supply a receipt or destination.
Retention is evidence accounting after cleanup, never resource or execution
authority. Failure leaves cleanup unchanged and does not justify another run.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat

from . import repository_source_owner_protocol as protocol
from .repository_source_owner_lifecycle import RepositorySourceOwnerLifecycle, SCHEMA as LIFECYCLE_SCHEMA
from .repository_source_resource_handoff import DEMAND
from .source_formalization_context import _protected_directory


SCHEMA = "repository-source-owner-retirement@1"
RETENTION_SCHEMA = "repository-source-owner-retirement-retention@1"
MAX_BYTES = protocol.MAX_FRAME_BYTES + 65536
_LIFECYCLE_FIELDS = frozenset({"schema", "state", "ready", "started", "worker_spawn_registered",
    "cleanup_complete", "cleanup_pending", "native_parent_acquired", "native_parent_released",
    "native_parent", "admission_uncertain", "observation_pending", "admission_request_id",
    "broker_cleanup_complete", "broker_audit", "renewal_unproven", "retained_custody", "error",
    "cleanup_error", "borrowed_parent_present", "resource_scope", "worker_reaping_performed",
    "worker_cleanup_verified", "automatic_terminal_wiring", "cross_uid_deployment_qualified",
    "provider_calls", "token_savings_measured", "proof_authority", "execution_authority",
    "completion_authority"})
_PARENT_FIELDS = frozenset({"lease_id", "owner_pid", "parent_lease_id", *DEMAND})
_FORBIDDEN_FIELDS = frozenset({"lease_key", "parent_token", "lease_token", "resource_grant",
    "state_owner_token", "capability", "private_key"})


def _need(condition, reason="invalid_repository_source_retirement"):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory_identity(info):
    # Creating our own file changes directory timestamps and size.
    return info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid


def _text(value):
    return (type(value) is str and 0 < len(value.encode("utf-8")) <= 256
            and not any(ord(character) < 32 or ord(character) == 127 for character in value))


def _public_tree(value, depth=0):
    _need(depth <= 32)
    if type(value) is dict:
        _need(not set(value) & _FORBIDDEN_FIELDS)
        for item in value.values():
            _public_tree(item, depth + 1)
    elif type(value) is list:
        for item in value:
            _public_tree(item, depth + 1)


def _snapshot(owner):
    receipt = owner._receipt()
    _need(type(receipt) is dict and set(receipt) == _LIFECYCLE_FIELDS)
    _need(receipt["schema"] == LIFECYCLE_SCHEMA and receipt["state"] == "closed"
          and receipt["resource_scope"] == "repository_source_service_only")
    for field in ("started", "cleanup_complete", "native_parent_acquired", "native_parent_released",
                  "broker_cleanup_complete"):
        _need(receipt[field] is True)
    for field in ("ready", "cleanup_pending", "admission_uncertain", "observation_pending",
                  "retained_custody", "worker_reaping_performed", "worker_cleanup_verified",
                  "automatic_terminal_wiring", "cross_uid_deployment_qualified", "token_savings_measured",
                  "proof_authority", "execution_authority", "completion_authority"):
        _need(receipt[field] is False)
    for field in ("worker_spawn_registered", "renewal_unproven", "borrowed_parent_present"):
        _need(type(receipt[field]) is bool)
    _need(receipt["cleanup_error"] is None and type(receipt["provider_calls"]) is int
          and receipt["provider_calls"] == 0
          and type(receipt["admission_request_id"]) is str
          and re.fullmatch("repository-source-lifecycle:[0-9a-f]{32}", receipt["admission_request_id"]))
    _need(receipt["error"] is None or type(receipt["error"]) is str
          and re.fullmatch("[a-z][a-z0-9_]{0,127}", receipt["error"]))
    parent = receipt["native_parent"]
    _need(type(parent) is dict and set(parent) == _PARENT_FIELDS
          and _text(parent["lease_id"]) and type(parent["owner_pid"]) is int
          and parent["owner_pid"] == os.getpid()
          and (parent["parent_lease_id"] is None or _text(parent["parent_lease_id"]))
          and parent["parent_lease_id"] != parent["lease_id"]
          and receipt["borrowed_parent_present"] is (parent["parent_lease_id"] is not None)
          and all(type(parent[name]) is int and parent[name] == demand for name, demand in DEMAND.items()))
    lease, broker = owner._lease, owner._broker
    _need(lease is not None and lease.released is True and lease.lease_id == parent["lease_id"]
          and broker is not None)
    audit = receipt["broker_audit"]
    _need(type(audit) is dict and audit.get("schema") in {
              "repository-source-owner-session@1", "repository-source-owner-session@2"}
          and audit.get("cleanup_complete") is True and audit.get("service_stopped") is True
          and audit.get("native_child_acquired") is True and audit.get("native_child_released") is True
          and audit.get("native_child_acquisition_uncertain") is False)
    descriptor = protocol.validate_descriptor(audit["session"])
    _need(type(descriptor["owner_pid"]) is int and descriptor["owner_pid"] == os.getpid()
          and descriptor["owner_uid"] == os.getuid() == os.geteuid() == os.getgid()
          and descriptor["owner_uid"] > 0
          and protocol.canonical_json(descriptor) == protocol.canonical_json(broker.descriptor)
          and audit["descriptor_sha256"] == broker.descriptor_sha256 == _sha(protocol.canonical_json(descriptor)))
    child = audit["native_child"]
    _need(type(child) is dict and child.get("parent_lease_id") == parent["lease_id"]
          and type(child.get("owner_pid")) is int and child["owner_pid"] == parent["owner_pid"]
          and child.get("released") is True)
    _public_tree(receipt)
    raw = protocol.canonical_json(audit)
    _need(0 < len(raw) <= protocol.MAX_FRAME_BYTES)
    # The receipt is freshly constructed under the lifecycle lock. Canonical
    # serialization later makes this public snapshot independent of references.
    return receipt, descriptor, raw


def _read_sealed(fd, name, *, owner, maximum):
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(handle)
        _need(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o400
              and before.st_uid == before.st_gid == owner and before.st_nlink == 1
              and 0 < before.st_size <= maximum, "sealed_owner_retirement_evidence_required")
        with os.fdopen(handle, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(handle)
        named = os.stat(name, dir_fd=fd, follow_symlinks=False)
        _need(len(raw) == before.st_size and _identity(before) == _identity(after) == _identity(named),
              "owner_retirement_evidence_changed")
        return raw
    finally:
        os.close(handle)


def _check_parent(fd, path, expected, owner):
    _protected_directory(path, owner=owner, group=owner)
    _need(_directory_identity(os.fstat(fd)) == expected == _directory_identity(path.lstat()),
          "owner_retirement_parent_changed")


def _write_sealed(fd, name, raw, *, owner, check_parent):
    created = None
    handle = None
    try:
        try:
            handle = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=fd)
        except FileExistsError:
            _need(_read_sealed(fd, name, owner=owner, maximum=MAX_BYTES) == raw,
                  "different_owner_retirement_already_retained")
            check_parent()
            return
        info = os.fstat(handle)
        created = info.st_dev, info.st_ino
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
              and info.st_uid == info.st_gid == owner)
        view = memoryview(raw)
        while view:
            written = os.write(handle, view)
            _need(written > 0, "owner_retirement_write_incomplete")
            view = view[written:]
        os.fchmod(handle, 0o400)
        os.fsync(handle)
        _need((os.fstat(handle).st_dev, os.fstat(handle).st_ino) == created)
        _need(_read_sealed(fd, name, owner=owner, maximum=MAX_BYTES) == raw)
        check_parent()
        os.fsync(fd)
    except BaseException:
        if created is not None:
            try:
                current = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == created:
                    os.unlink(name, dir_fd=fd)
            except OSError:
                pass
        raise
    finally:
        if handle is not None:
            os.close(handle)


def retain_repository_source_owner_retirement(owner):
    """Retain this exact owner's closed public receipt, or raise without cleanup.

    No receipt, filename, scheduler state or capability is accepted from a
    caller. An absent session cannot become synthetic release evidence.
    """
    _need(type(owner) is RepositorySourceOwnerLifecycle, "exact_repository_source_owner_required")
    _need(type(owner._owner_pid) is int and owner._owner_pid == os.getpid(),
          "repository_source_owner_belongs_to_another_process")
    with owner._lock:
        receipt, descriptor, audit_raw = _snapshot(owner)
        broker = owner._broker
        session = descriptor["session_id"]
        parent = owner.boundary.parent / "worker-observations"
        audit_name = "repository-mcp-" + session + "-audit.json"
        _need(isinstance(broker.audit_path, Path))
        _need(broker.audit_path == parent / audit_name and broker.boundary == owner.boundary
              and broker.parent == parent and parent.is_absolute(), "fixed_owner_retirement_parent_required")
        uid = descriptor["owner_uid"]
        _protected_directory(parent, owner=uid, group=uid)
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            identity = _directory_identity(os.fstat(fd))
            check_parent = lambda: _check_parent(fd, parent, identity, uid)
            check_parent()
            _need(_read_sealed(fd, audit_name, owner=uid, maximum=protocol.MAX_FRAME_BYTES) == audit_raw,
                  "retained_owner_session_audit_differs")
            envelope = dict(schema=SCHEMA, session_id=session,
                descriptor_sha256=receipt["broker_audit"]["descriptor_sha256"],
                session_audit_sha256=_sha(audit_raw), session_audit_bytes=len(audit_raw), lifecycle=receipt)
            raw = protocol.canonical_json(envelope)
            _need(0 < len(raw) <= MAX_BYTES, "owner_retirement_exceeds_bound")
            name = "repository-mcp-" + session + "-lifecycle.json"
            _write_sealed(fd, name, raw, owner=uid, check_parent=check_parent)
            return dict(schema=RETENTION_SCHEMA, status="retained", reason=None,
                        name=name, sha256=_sha(raw), bytes=len(raw))
        finally:
            os.close(fd)
