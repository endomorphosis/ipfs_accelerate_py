"""Exclusive private attempt records, independent of provider and replay truth.

The start record exists durably before router entry. Failed writes remain for
inspection. Descriptors identify bytes; they do not authenticate chronology,
omitted attempts, provider events, usage, or a historical file's original inode.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import weakref

from .repository_world_artifact_retention import (
    _canonical_path, _directory_identity, _file_identity, _open_directory,
)

START_SCHEMA = "repository-world-attempt-start@1"
TERMINAL_SCHEMA = "repository-world-attempt-terminal@1"
SCHEMA = "repository-world-attempt-journal@1"
START_MAX_BYTES = TERMINAL_MAX_BYTES = 64 * 1024
RECEIPT_MAX_BYTES = 2 * 1024 * 1024
MAX_NODES, MAX_DEPTH = 65536, 40
CONTROL_FIELDS = frozenset({"control_sha256", "provider", "model", "model_config_sha256",
    "budgets_sha256", "cache_policy_sha256", "cache_state_sha256",
    "execution_policy_sha256", "verifier_policy_sha256", "source_sha256",
    "task_sha256", "acceptance_sha256", "requirements_sha256", "codec_sha256"})
FALSE_FIELDS = frozenset({"source_currentness_verified", "original_inode_replayed",
    "provider_delivery_verified", "provider_ingestion_verified", "billing_verified",
    "token_counts_verified", "independent_campaign_census_verified",
    "execution_authority", "completion_authority"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ERROR = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_ISSUED = {}


class RepositoryWorldAttemptJournalError(ValueError):
    """Closed codes that do not expose private content or exception strings."""


def _need(condition, reason):
    if not condition:
        raise RepositoryWorldAttemptJournalError("world_attempt_journal_" + reason)


def _wire(value, maximum):
    """Bound exact JSON literals before traversal, copying, or serialization."""
    pending, active = [(value, 0, False)], set()
    nodes = text_bytes = 0
    try:
        while pending:
            item, depth, leaving = pending.pop()
            if leaving:
                active.remove(id(item))
                continue
            nodes += 1
            _need(nodes <= MAX_NODES and depth <= MAX_DEPTH, "structure_bound")
            if type(item) is str:
                _need(len(item) <= maximum, "byte_bound")
                text_bytes += len(item.encode("utf-8", "strict"))
                _need(text_bytes <= maximum, "byte_bound")
            elif type(item) in (dict, list):
                _need(id(item) not in active, "cyclic_json")
                active.add(id(item))
                pending.append((item, depth, True))
                if type(item) is dict:
                    _need(nodes + len(pending) + 2 * len(item) <= MAX_NODES, "structure_bound")
                    _need(all(type(key) is str for key in item), "json_type")
                    children = [child for pair in item.items() for child in pair]
                else:
                    _need(nodes + len(pending) + len(item) <= MAX_NODES, "structure_bound")
                    children = item
                pending.extend((child, depth + 1, False) for child in children)
            elif type(item) is int:
                _need(item.bit_length() <= 64, "integer_bound")
            elif type(item) is float:
                _need(math.isfinite(item), "nonfinite_json")
            else:
                _need(item is None or type(item) is bool, "json_type")
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode("utf-8", "strict")
    except (TypeError, UnicodeError, RecursionError) as error:
        raise RepositoryWorldAttemptJournalError("world_attempt_journal_json_type") from error
    _need(0 < len(raw) <= maximum, "byte_bound")
    return raw


def _start(value):
    raw = _wire(value, START_MAX_BYTES)
    value = json.loads(raw)
    _need(type(value) is dict and set(value) == {"schema", "attempt_id", "case_id", "arm",
        "parent_attempt_id", "observed_controls"} and value["schema"] == START_SCHEMA,
        "start_fields")
    for key in ("attempt_id", "case_id"):
        _need(type(value[key]) is str and _ID.fullmatch(value[key]), "start_identity")
    parent = value["parent_attempt_id"]
    _need(parent is None or type(parent) is str and _ID.fullmatch(parent)
        and parent != value["attempt_id"], "start_parent")
    _need(type(value["arm"]) is str and value["arm"] in {"baseline", "candidate"}, "start_arm")
    controls = value["observed_controls"]
    _need(type(controls) is dict and set(controls) == CONTROL_FIELDS, "start_controls")
    for key, item in controls.items():
        if item is None:
            continue
        if key in {"provider", "model"}:
            _need(type(item) is str and _ID.fullmatch(item), "start_controls")
        else:
            _need(type(item) is str and _SHA.fullmatch(item), "start_controls")
    return raw


def _router_receipt(value):
    raw = _wire(value, RECEIPT_MAX_BYTES)
    value = json.loads(raw)
    _need(type(value) is dict and value.get("schema") == "router-implementation-invocation@1"
        and value.get("provider") == "codex_cli" and value.get("purpose") == "coding"
        and type(value.get("model")) is str and _ID.fullmatch(value["model"])
        and value.get("status") in ("provider_returned", "failed")
        and value.get("completion_authority") is False
        and type(value.get("invocation_id")) is str
        and re.fullmatch(r"[0-9a-f]{32}", value["invocation_id"]) is not None,
        "router_receipt")
    error_type = value.get("error_type")
    _need(error_type is None or type(error_type) is str and _ID.fullmatch(error_type),
        "router_error_type")
    packet = value.get("world_task_packet")
    _need(type(packet) is dict and "validation_profile" not in packet
        and type(packet.get("validation_count")) is int and packet["validation_count"] > 0
        and type(packet.get("dispatch_rechecked")) is bool
        and type(packet.get("provider_attempted")) is bool
        and (not packet["provider_attempted"] or packet["dispatch_rechecked"])
        and (value["status"] != "provider_returned" or packet["provider_attempted"]),
        "strict_packet_observation")
    return raw


def _state(journal):
    _need(type(journal) is RepositoryWorldAttemptJournal, "exact_issued_required")
    row = _ISSUED.get(id(journal))
    _need(row is not None and row["reference"]() is journal, "exact_issued_required")
    try:
        _need(object.__getattribute__(journal, "_seal") is row["seal"], "issued_material_changed")
    except AttributeError as error:
        raise RepositoryWorldAttemptJournalError("world_attempt_journal_issued_material_changed") from error
    return row


def _guard(row):
    _need(row["fd"] is not None, "closed")
    check = _open_directory(row["root"])
    try:
        _need(_directory_identity(os.fstat(row["fd"])) == row["root_identity"]
            == _directory_identity(os.fstat(check)), "root_changed")
    finally:
        os.close(check)


def _check(row, name):
    _guard(row)
    descriptor, initial = row["files"][name]
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=row["fd"])
    try:
        before = os.fstat(fd)
        _need(_file_identity(before) == initial, "file_changed")
        digest, remaining = hashlib.sha256(), descriptor["bytes"]
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            _need(bool(chunk), "short_read")
            remaining -= len(chunk)
            digest.update(chunk)
        _need(not os.read(fd, 1) and digest.hexdigest() == descriptor["sha256"], "readback_mismatch")
        _need(initial == _file_identity(os.fstat(fd))
            == _file_identity(os.stat(name, dir_fd=row["fd"], follow_symlinks=False)), "file_changed")
        _guard(row)
    finally:
        os.close(fd)


def _check_all(row):
    _guard(row)
    for name in row["files"]:
        _check(row, name)


def _write(row, name, raw):
    _check_all(row)
    fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        | os.O_NONBLOCK, 0o600, dir_fd=row["fd"])
    try:
        os.fchmod(fd, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(fd, memoryview(raw)[offset:])
            _need(written > 0, "short_write")
            offset += written
        os.fsync(fd)
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1
            and info.st_size == len(raw), "file_changed")
        descriptor = dict(name=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        initial = _file_identity(info)
        # Check the held creation FD before closing; _check also verifies the
        # directory entry is still this exact inode, not a same-byte substitute.
        os.lseek(fd, 0, os.SEEK_SET)
        digest, remaining = hashlib.sha256(), len(raw)
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            _need(bool(chunk), "short_read")
            remaining -= len(chunk)
            digest.update(chunk)
        _need(not os.read(fd, 1) and digest.hexdigest() == descriptor["sha256"], "readback_mismatch")
        _need(initial == _file_identity(os.fstat(fd))
            == _file_identity(os.stat(name, dir_fd=row["fd"], follow_symlinks=False)), "file_changed")
        os.fsync(row["fd"])
        _guard(row)
        row["files"][name] = descriptor, initial
        try:
            _check_all(row)
        except BaseException:
            row["files"].pop(name, None)
            raise
        return descriptor
    finally:
        os.close(fd)


class RepositoryWorldAttemptJournal:
    """One exact-issued journal in an existing private, disjoint directory."""
    __slots__ = ("_seal", "__weakref__")

    def __setattr__(self, _name, _value):
        raise AttributeError("attempt journals are immutable")

    def __delattr__(self, _name):
        raise AttributeError("attempt journals are immutable")

    def __init__(self, root: Path, *, workspace: Path, start: dict):
        _need(id(self) not in _ISSUED, "already_issued")
        raw = _start(start)
        start = json.loads(raw)
        root, workspace = _canonical_path(root), _canonical_path(workspace)
        _need(not root.is_relative_to(workspace) and not workspace.is_relative_to(root),
            "root_overlaps_workspace")
        workspace_fd = _open_directory(workspace)
        os.close(workspace_fd)
        fd = _open_directory(root)
        try:
            info = os.fstat(fd)
            _need(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700,
                "root_not_private")
            # A dedicated directory avoids mixing attempts or adopting prior
            # partial evidence. Never remove or rewrite occupied entries.
            with os.scandir(fd) as entries:
                _need(next(entries, None) is None, "root_not_empty")
            row = dict(fd=fd, root=root, workspace=workspace, root_identity=_directory_identity(info), files={},
                attempt_id=start["attempt_id"], capture_error=False, terminal=False)
            _write(row, "start.json", raw)
            seal = object()
            object.__setattr__(self, "_seal", seal)
            identity = id(self)
            def release(reference):
                found = _ISSUED.get(identity)
                if found is not None and found["reference"] is reference:
                    _ISSUED.pop(identity, None)
                    held, found["fd"] = found["fd"], None
                    if held is not None:
                        try:
                            os.close(held)
                        except OSError:
                            pass
            row.update(seal=seal, reference=weakref.ref(self, release))
            _ISSUED[identity] = row
        except BaseException:
            os.close(fd)
            raise

    def require_open(self, *, workspace=None):
        row = _state(self)
        _need(not row["terminal"], "already_terminal")
        if workspace is not None:
            workspace = _canonical_path(workspace)
            _need(workspace == row["workspace"] and not row["root"].is_relative_to(workspace)
                and not workspace.is_relative_to(row["root"]), "workspace_mismatch")
            workspace_fd = _open_directory(workspace)
            os.close(workspace_fd)
        _check_all(row)

    def retain_receipt(self, receipt):
        row = _state(self)
        RepositoryWorldAttemptJournal.require_open(self)
        _need("receipt.json" not in row["files"], "receipt_already_retained")
        raw = _router_receipt(receipt)
        _write(row, "receipt.json", raw)
        return RepositoryWorldAttemptJournal.receipt(self)

    def mark_capture_error(self):
        _state(self)["capture_error"] = True

    def finish(self, *, outcome, error_type=None):
        row = _state(self)
        RepositoryWorldAttemptJournal.require_open(self)
        _need(type(outcome) is str and outcome in {"returned", "raised", "interrupted"}, "outcome")
        _need(error_type is None or type(error_type) is str and _ERROR.fullmatch(error_type), "error_type")
        _need(outcome != "returned" or error_type is None, "error_type")
        receipt = row["files"].get("receipt.json")
        terminal = dict(schema=TERMINAL_SCHEMA, attempt_id=row["attempt_id"],
            start_sha256=row["files"]["start.json"][0]["sha256"],
            receipt_sha256=None if receipt is None else receipt[0]["sha256"],
            outcome=outcome, error_type=error_type)
        _write(row, "terminal.json", _wire(terminal, TERMINAL_MAX_BYTES))
        _check_all(row)
        row["terminal"] = True
        return RepositoryWorldAttemptJournal.receipt(self)

    def receipt(self):
        row = _state(self)
        def descriptor(name):
            return None if name not in row["files"] else dict(row["files"][name][0])
        return dict(schema=SCHEMA, attempt_id=row["attempt_id"],
            status="terminal_retained" if row["terminal"] else
                "receipt_retained" if "receipt.json" in row["files"] else "started",
            closed=row["fd"] is None, capture_error=row["capture_error"],
            start=descriptor("start.json"), receipt=descriptor("receipt.json"),
            terminal=descriptor("terminal.json"), **dict.fromkeys(FALSE_FIELDS, False))

    def close(self):
        row = _state(self)
        fd, row["fd"] = row["fd"], None
        if fd is not None:
            os.close(fd)

    def __enter__(self):
        RepositoryWorldAttemptJournal.require_open(self)
        return self

    def __exit__(self, _kind, _value, _traceback):
        RepositoryWorldAttemptJournal.close(self)
