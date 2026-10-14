"""Private exclusive retention of already issued world-packet input artifacts.

Independent receipt descriptors identify retained bytes. They confer neither
live packet issuance nor source, provider, token, proof or completion authority.
Failure preserves any created file for inspection; foreign files are not removed.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat


SCHEMA = "repository-world-invocation-retention@1"
PACKET_MAX_BYTES = 2 * 1024 * 1024
INVOCATION_MAX_BYTES = 3 * 1024 * 1024
FALSE_FIELDS = frozenset({"source_currentness_verified", "owner_signature_replayed",
    "provider_delivery_verified", "token_counts_replayed", "proof_authority",
    "execution_authority", "completion_authority"})
REASONS = frozenset({"retention_not_requested", "packet_retention_failed",
    "invocation_retention_pending", "invocation_retention_failed"})


class RepositoryWorldArtifactRetentionError(ValueError):
    """Closed error codes without source, paths or exception messages."""


def _need(value, code):
    if not value:
        raise RepositoryWorldArtifactRetentionError(code)


def _invocation(value):
    _need(type(value) is str and re.fullmatch(r"[0-9a-f]{32}", value) is not None,
          "invalid_world_artifact_invocation")
    return value


def artifact_name(invocation_id, kind):
    _invocation(invocation_id)
    _need(type(kind) is str and kind in {"packet", "invocation"}, "invalid_world_artifact_kind")
    return invocation_id + "-world-" + kind + ".json"


def validate_world_packet_artifacts(value, *, invocation_id):
    """Validate closed receipt bindings; do not open files or attest delivery."""
    _invocation(invocation_id)
    _need(type(value) is dict and set(value) == {
        "schema", "status", "packet", "invocation", "reason", *FALSE_FIELDS},
        "invalid_world_artifact_receipt")
    _need(value["schema"] == SCHEMA and all(value[key] is False for key in FALSE_FIELDS),
          "invalid_world_artifact_authority")
    _need(type(value["status"]) is str and value["status"] in {"retained", "unavailable"},
          "invalid_world_artifact_status")
    copied = {key: value[key] for key in value}
    for kind, limit in (("packet", PACKET_MAX_BYTES), ("invocation", INVOCATION_MAX_BYTES)):
        row = value[kind]
        if row is None:
            continue
        _need(type(row) is dict and set(row) == {"name", "bytes", "sha256"}
              and row["name"] == artifact_name(invocation_id, kind)
              and type(row["bytes"]) is int and 0 < row["bytes"] <= limit
              and type(row["sha256"]) is str
              and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
              "invalid_world_artifact_descriptor")
        copied[kind] = dict(row)
    if value["status"] == "retained":
        _need(value["reason"] is None and all(value[kind] is not None for kind in ("packet", "invocation")),
              "incomplete_world_artifact_receipt")
    else:
        _need(type(value["reason"]) is str and value["reason"] in REASONS
              and value["invocation"] is None, "invalid_world_artifact_unavailable")
        if value["reason"] in {"retention_not_requested", "packet_retention_failed"}:
            _need(value["packet"] is None, "invalid_world_artifact_unavailable")
        if value["reason"] == "invocation_retention_pending":
            _need(value["packet"] is not None, "invalid_world_artifact_unavailable")
    return copied


def unavailable_world_packet_artifacts(*, invocation_id, reason, packet=None):
    return validate_world_packet_artifacts(dict(schema=SCHEMA, status="unavailable",
        packet=packet, invocation=None, reason=reason,
        **dict.fromkeys(FALSE_FIELDS, False)), invocation_id=invocation_id)


def _canonical_path(value):
    _need(isinstance(value, Path) and value.is_absolute() and ".." not in value.parts
          and 0 < len(os.fsencode(value)) <= 4096, "invalid_world_artifact_root")
    return value


def _open_directory(path):
    """Follow no symlink in any component, including the supplied root."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                            | os.O_NONBLOCK, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _directory_identity(info):
    return info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode)


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class RepositoryWorldArtifactRetention:
    """Hold one independently selected directory through one live invocation."""

    def __init__(self, root: Path, *, workspace: Path, invocation_id: str):
        self.root = _canonical_path(root)
        workspace = _canonical_path(workspace)
        _need(not self.root.is_relative_to(workspace) and not workspace.is_relative_to(self.root),
              "world_artifact_root_overlaps_workspace")
        workspace_fd = _open_directory(workspace)
        os.close(workspace_fd)
        self.invocation_id = _invocation(invocation_id)
        self._fd = None
        self._packet = self._packet_bytes = None
        descriptor = _open_directory(self.root)
        try:
            info = os.fstat(descriptor)
            _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
                  and stat.S_IMODE(info.st_mode) == 0o700, "world_artifact_root_not_private")
            self._root_identity = _directory_identity(info)
            self._fd = descriptor
            self._guard()
        except BaseException:
            self._fd = None
            os.close(descriptor)
            raise

    def _guard(self):
        _need(self._fd is not None, "world_artifact_retention_closed")
        held = os.fstat(self._fd)
        check = _open_directory(self.root)
        try:
            _need(_directory_identity(held) == self._root_identity
                  == _directory_identity(os.fstat(check)), "world_artifact_root_changed")
        finally:
            os.close(check)

    def _write(self, kind, data):
        limit = PACKET_MAX_BYTES if kind == "packet" else INVOCATION_MAX_BYTES
        _need(type(data) is bytes and 0 < len(data) <= limit, "world_artifact_byte_limit")
        self._guard()
        name = artifact_name(self.invocation_id, kind)
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                     | os.O_NONBLOCK, 0o600, dir_fd=self._fd)
        try:
            os.fchmod(fd, 0o600)
            offset = 0
            while offset < len(data):
                written = os.write(fd, memoryview(data)[offset:])
                _need(written > 0, "world_artifact_short_write")
                offset += written
            os.fsync(fd)
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
                  and stat.S_IMODE(before.st_mode) == 0o600 and before.st_nlink == 1
                  and before.st_size == len(data), "world_artifact_file_changed")
            os.lseek(fd, 0, os.SEEK_SET)
            remaining, digest = len(data), hashlib.sha256()
            while remaining:
                raw = os.read(fd, min(65536, remaining))
                _need(bool(raw), "world_artifact_short_read")
                remaining -= len(raw)
                digest.update(raw)
            _need(not os.read(fd, 1) and digest.hexdigest() == hashlib.sha256(data).hexdigest(),
                  "world_artifact_readback_mismatch")
            _need(_file_identity(before) == _file_identity(os.fstat(fd))
                  == _file_identity(os.stat(name, dir_fd=self._fd, follow_symlinks=False)),
                  "world_artifact_file_changed")
            os.fsync(self._fd)
            self._guard()
            return dict(name=name, bytes=len(data), sha256=digest.hexdigest())
        finally:
            os.close(fd)

    def retain_packet(self, packet):
        from .repository_world_packet_replay import freeze_repository_world_packet
        _need(self._packet_bytes is None, "world_packet_already_retained")
        data = freeze_repository_world_packet(packet)
        descriptor = self._write("packet", data)
        self._packet, self._packet_bytes = descriptor, data
        return unavailable_world_packet_artifacts(invocation_id=self.invocation_id,
            reason="invocation_retention_pending", packet=descriptor)

    def _check_packet(self):
        """Recheck the retained first artifact before publishing a joined pair."""
        self._guard()
        row = self._packet
        fd = os.open(row["name"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._fd)
        try:
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
                  and stat.S_IMODE(before.st_mode) == 0o600 and before.st_nlink == 1
                  and before.st_size == row["bytes"], "world_artifact_file_changed")
            digest, remaining = hashlib.sha256(), row["bytes"]
            while remaining:
                chunk = os.read(fd, min(65536, remaining))
                _need(bool(chunk), "world_artifact_short_read")
                remaining -= len(chunk)
                digest.update(chunk)
            _need(not os.read(fd, 1) and digest.hexdigest() == row["sha256"],
                  "world_artifact_readback_mismatch")
            _need(_file_identity(before) == _file_identity(os.fstat(fd))
                  == _file_identity(os.stat(row["name"], dir_fd=self._fd, follow_symlinks=False)),
                  "world_artifact_file_changed")
            self._guard()
        finally:
            os.close(fd)

    def finish(self, invocation_receipt):
        from .repository_world_invocation_replay import freeze_repository_world_invocation
        _need(self._packet_bytes is not None, "world_packet_not_retained")
        _need(type(invocation_receipt) is dict
              and invocation_receipt.get("invocation_id") == self.invocation_id,
              "world_artifact_invocation_mismatch")
        self._check_packet()
        data = freeze_repository_world_invocation(packet_artifact=self._packet_bytes,
            expected_packet_sha256=self._packet["sha256"], invocation_receipt=invocation_receipt)
        descriptor = self._write("invocation", data)
        self._check_packet()
        return validate_world_packet_artifacts(dict(schema=SCHEMA, status="retained",
            packet=self._packet, invocation=descriptor, reason=None,
            **dict.fromkeys(FALSE_FIELDS, False)), invocation_id=self.invocation_id)

    def unavailable(self, reason):
        return unavailable_world_packet_artifacts(invocation_id=self.invocation_id,
            reason=reason, packet=self._packet)

    def close(self):
        if self._fd is not None:
            fd, self._fd = self._fd, None
            os.close(fd)
