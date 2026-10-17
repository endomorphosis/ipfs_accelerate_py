"""Bounded, worker-reported forwarding handoff through an invocation outbox.

This standard-library-only helper owns no admission, subprocess, owner state or
authority. The fixed-slot scan describes recovered files, not the population of
helpers or bytes consumed by a provider. A same-UID worker can forge this data.
The router projects validated content-free records into its existing receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading

TARGET_SCHEMA = "repository-source-worker-outbox@1"
CAPTURE_SCHEMA = "repository-source-worker-forwarding-capture@1"
RECEIPT_SCHEMA = "repository-source-worker-forwarding@1"
TARGET_FLAG = "--worker-forwarding-target"
MAX_SLOTS = 16
MAX_RECEIPT_BYTES = 65536
MAX_TARGET_BYTES = 8192
_TARGET_FIELDS = {"schema", "directory", "device", "inode", "uid", "session_id",
                  "descriptor_sha256", "invocation_id"}
_FALSE_FIELDS = {"independently_verified", "provider_ingestion_verified", "token_savings_measured",
                 "proof_authority", "execution_authority", "completion_authority", "raw_content_exported"}
_FLAGS = dict(worker_reported=True, observer_provider_calls=0, **dict.fromkeys(_FALSE_FIELDS, False))
_REASONS = {"invalid_request", "peer_refused", "stale_context", "budget_exhausted", "session_closed",
            "retrieval_refused", "deadline_exceeded", "cancelled", "transport_error", "invalid_response"}
_EVENT_FIELDS = {"exchange_ordinal", "request_sha256", "owner_response_sha256", "owner_response_bytes",
    "owner_ok", "owner_reason_code", "output_kind", "stdout_sha256", "stdout_bytes", "forwarding_status"}
_RECEIPT_FIELDS = {"schema", "status", "complete", "reason", "bridge_id", "session_id", "descriptor_sha256",
    "returncode", "owner_exchange_attempts", "owner_exchange_unconfirmed", "observed_responses", "events", *_FLAGS}
_CAPTURE_FIELDS = {"schema", "status", "reason", "session_id", "descriptor_sha256", "invocation_id",
    "target_sha256", "receipts", "receipt_count", "receipt_bytes", "slot_scan_complete", "cleanup_complete", *_FLAGS}


def _need(ok):
    if not ok:
        raise ValueError("invalid_worker_forwarding_capture")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _loads(raw, maximum):
    _need(type(raw) is bytes and len(raw) <= maximum)
    def pairs(values):
        result = {}
        for key, value in values:
            _need(key not in result)
            result[key] = value
        return result
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid_worker_forwarding_capture")))


def _hex(value, size):
    return type(value) is str and re.fullmatch("[0-9a-f]{" + str(size) + "}", value) is not None


def _int(value, maximum, minimum=0):
    return type(value) is int and minimum <= value <= maximum


def _flags(value):
    _need(value["worker_reported"] is True and all(value[key] is False for key in _FALSE_FIELDS)
          and type(value["observer_provider_calls"]) is int and value["observer_provider_calls"] == 0)


def _path(value):
    _need(type(value) is str and 0 < len(value.encode("utf-8")) <= 4096
          and not any(ord(char) < 32 or ord(char) == 127 for char in value)
          and not value.startswith("//") and "\\" not in value and value != "/")
    path = Path(value)
    _need(path.is_absolute() and ".." not in path.parts and str(path) == value)
    return path


def validate_target(target, *, workspace=None, session_id=None, descriptor_sha256=None, invocation_id=None):
    """Validate public target metadata only; historical validation opens nothing."""
    _need(type(target) is dict and set(target) == _TARGET_FIELDS and target["schema"] == TARGET_SCHEMA)
    directory = _path(target["directory"])
    _need(_int(target["device"], 2**64 - 1) and _int(target["inode"], 2**64 - 1, 1)
          and _int(target["uid"], 2**32 - 1) and _hex(target["session_id"], 32)
          and _hex(target["descriptor_sha256"], 64) and _hex(target["invocation_id"], 32))
    for key, value in (("session_id", session_id), ("descriptor_sha256", descriptor_sha256),
                       ("invocation_id", invocation_id)):
        _need(value is None or target[key] == value)
    if workspace is not None:
        _need(not directory.is_relative_to(_path(str(workspace))))
    return _loads(canonical_json(target), MAX_TARGET_BYTES)


def parse_target(raw):
    _need(type(raw) is str and len(raw) <= MAX_TARGET_BYTES)
    return validate_target(_loads(raw.encode("utf-8"), MAX_TARGET_BYTES))


def _receipt_raw(receipt, target):
    """Reject arbitrary strings/content before projecting any worker data."""
    _need(type(receipt) is dict and set(receipt) == _RECEIPT_FIELDS
          and receipt["schema"] == RECEIPT_SCHEMA and type(receipt["complete"]) is bool
          and type(receipt["status"]) is str and receipt["status"] in {"not_started", "observed", "incomplete"}
          and (receipt["reason"] is None or type(receipt["reason"]) is str and receipt["reason"] in
               {"not_started", "descriptor_unavailable", "observation_incomplete", "observation_failed"})
          and _hex(receipt["bridge_id"], 32))
    _flags(receipt)
    _need(receipt["session_id"] is None or receipt["session_id"] == target["session_id"])
    _need(receipt["descriptor_sha256"] is None or receipt["descriptor_sha256"] == target["descriptor_sha256"])
    _need((receipt["session_id"] is None) == (receipt["descriptor_sha256"] is None))
    _need(receipt["returncode"] is None or type(receipt["returncode"]) is int and receipt["returncode"] in (0, 2))
    if receipt["complete"]:
        _need(receipt["status"] == "observed" and receipt["reason"] is None
              and receipt["session_id"] is not None and receipt["returncode"] is not None)
    else:
        _need(receipt["status"] != "observed" and receipt["reason"] is not None)
    events = receipt["events"]
    _need(type(events) is list and len(events) <= 64)
    for key in ("owner_exchange_attempts", "owner_exchange_unconfirmed", "observed_responses"):
        _need(receipt[key] is None or _int(receipt[key], 64))
    if receipt["observed_responses"] is not None:
        _need(receipt["observed_responses"] == len(events))
    if all(receipt[key] is not None for key in
           ("owner_exchange_attempts", "owner_exchange_unconfirmed", "observed_responses")):
        _need(receipt["owner_exchange_attempts"] >= receipt["observed_responses"] + receipt["owner_exchange_unconfirmed"])
    previous = 0
    for event in events:
        _need(type(event) is dict and set(event) == _EVENT_FIELDS
              and _int(event["exchange_ordinal"], 64, previous + 1)
              and _hex(event["request_sha256"], 64) and _hex(event["owner_response_sha256"], 64)
              and _int(event["owner_response_bytes"], 2 * 1024 * 1024 + 4096, 1)
              and type(event["owner_ok"]) is bool
              and type(event["output_kind"]) is str and event["output_kind"] in {"owner_message", "local_error", "no_output"}
              and type(event["forwarding_status"]) is str and event["forwarding_status"] in
                  {"write_completed", "no_output", "budget_refused", "unconfirmed"})
        previous = event["exchange_ordinal"]
        _need(event["owner_reason_code"] is None if event["owner_ok"] else
              type(event["owner_reason_code"]) is str and event["owner_reason_code"] in _REASONS)
        if event["output_kind"] == "no_output":
            _need(event["stdout_sha256"] is None and type(event["stdout_bytes"]) is int
                  and event["stdout_bytes"] == 0 and event["forwarding_status"] in {"no_output", "budget_refused"})
        else:
            _need(event["output_kind"] == ("owner_message" if event["owner_ok"] else "local_error")
                  and _hex(event["stdout_sha256"], 64) and _int(event["stdout_bytes"], 2 * 1024 * 1024 + 1, 1)
                  and event["forwarding_status"] != "no_output")
    raw = canonical_json(receipt)
    _need(len(raw) <= MAX_RECEIPT_BYTES)
    return raw


def unavailable_capture(*, session_id, descriptor_sha256, invocation_id,
                        reason="capture_failed", target_sha256=None):
    _need(_hex(session_id, 32) and _hex(descriptor_sha256, 64) and _hex(invocation_id, 32)
          and reason in {"capture_failed", "cleanup_failed", "capture_unavailable"}
          and (target_sha256 is None or _hex(target_sha256, 64)))
    return dict(schema=CAPTURE_SCHEMA, status="unavailable", reason=reason,
        session_id=session_id, descriptor_sha256=descriptor_sha256, invocation_id=invocation_id,
        target_sha256=target_sha256, receipts=[], receipt_count=None, receipt_bytes=None,
        slot_scan_complete=False, cleanup_complete=False, **_FLAGS)


def captured_receipt_bytes(capture, target):
    """Pure closed-schema check; unavailable evidence never becomes zero usage."""
    target = validate_target(target)
    _need(type(capture) is dict and set(capture) == _CAPTURE_FIELDS and capture["schema"] == CAPTURE_SCHEMA
          and capture["status"] == "captured" and capture["reason"] is None
          and capture["slot_scan_complete"] is True and capture["cleanup_complete"] is True
          and capture["target_sha256"] == hashlib.sha256(canonical_json(target)).hexdigest())
    _flags(capture)
    _need(all(capture[key] == target[key] for key in ("session_id", "descriptor_sha256", "invocation_id")))
    entries = capture["receipts"]
    _need(type(entries) is list and 1 <= len(entries) <= MAX_SLOTS)
    raws, slots, bridges = [], set(), set()
    for entry in entries:
        _need(type(entry) is dict and set(entry) == {"receipt", "sha256", "bytes", "slot"}
              and _int(entry["slot"], MAX_SLOTS - 1) and entry["slot"] not in slots)
        raw = _receipt_raw(entry["receipt"], target)
        _need(_hex(entry["sha256"], 64) and entry["sha256"] == hashlib.sha256(raw).hexdigest()
              and type(entry["bytes"]) is int and entry["bytes"] == len(raw)
              and entry["receipt"]["bridge_id"] not in bridges)
        slots.add(entry["slot"])
        bridges.add(entry["receipt"]["bridge_id"])
        raws.append(raw)
    _need(type(capture["receipt_count"]) is int and capture["receipt_count"] == len(raws)
          and type(capture["receipt_bytes"]) is int and capture["receipt_bytes"] == sum(map(len, raws)))
    return raws


def _snapshot(info):
    return tuple(getattr(info, key) for key in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
        "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _open_directory(path):
    """Walk using no-follow descriptors, including every ancestor component."""
    path = _path(str(path))
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in path.parts[1:]:
            following = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = following
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_directory(fd, target):
    info = os.fstat(fd)
    _need(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
          and info.st_uid == target["uid"] == os.getuid()
          and (info.st_dev, info.st_ino) == (target["device"], target["inode"]))


def _check_name(fd, target):
    _check_directory(fd, target)
    named = _open_directory(target["directory"])
    try:
        _check_directory(named, target)
    finally:
        os.close(named)


def _file_safe(info):
    _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
          and info.st_uid == os.getuid() and info.st_nlink == 1 and 0 < info.st_size <= MAX_RECEIPT_BYTES)


def _slot(index):
    return "receipt-" + format(index, "02d") + ".json"


def _unlink_same(directory_fd, name, info):
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    _need((current.st_dev, current.st_ino) == (info.st_dev, info.st_ino))
    os.unlink(name, dir_fd=directory_fd)


def publish_receipt(target, receipt):
    """Publish one canonical receipt; no overwrite or directory enumeration.

    The caller treats any ordinary exception as advisory. Exhausted slots return
    False. Existing files are never changed, including invalid or partial ones.
    """
    target = validate_target(target)
    raw = _receipt_raw(receipt, target)
    directory_fd = _open_directory(target["directory"])
    try:
        _check_name(directory_fd, target)
        for index in range(MAX_SLOTS):
            name = _slot(index)
            try:
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                             0o600, dir_fd=directory_fd)
            except FileExistsError:
                continue
            created = None
            successful = False
            try:
                created = os.fstat(fd)
                os.fchmod(fd, 0o600)
                offset = 0
                while offset < len(raw):
                    count = os.write(fd, raw[offset:])
                    _need(count > 0)
                    offset += count
                os.fsync(fd)
                final = os.fstat(fd)
                _file_safe(final)
                _need(final.st_size == len(raw) and _snapshot(final)
                      == _snapshot(os.stat(name, dir_fd=directory_fd, follow_symlinks=False)))
                _check_name(directory_fd, target)
                os.fsync(directory_fd)
                successful = True
                return True
            finally:
                os.close(fd)
                # Without the first metadata snapshot, the slot identity is
                # unknown. Close its FD but leave the name for bounded capture.
                if not successful and created is not None:
                    try:
                        _unlink_same(directory_fd, name, created)
                    except OSError:
                        pass
                    except ValueError:
                        pass
        return False
    finally:
        os.close(directory_fd)


class _Outbox:
    """Router-owned handle. Cleanup touches only inspected fixed slot names."""
    def __init__(self, target, directory_fd):
        self._target, self._fd = target, directory_fd
        self._result = None
        self._lock = threading.Lock()

    @property
    def target(self):
        return validate_target(self._target)

    def collect_and_close(self):
        with self._lock:
            if self._result is not None:
                return _loads(canonical_json(self._result), MAX_SLOTS * MAX_RECEIPT_BYTES + 16384)
            target, entries, seen = self._target, [], []
            result = unavailable_capture(session_id=target["session_id"], descriptor_sha256=target["descriptor_sha256"],
                invocation_id=target["invocation_id"], target_sha256=hashlib.sha256(canonical_json(target)).hexdigest())
            scan_ok, cleanup_ok, interruption = True, True, None
            try:
                _check_name(self._fd, target)
                for index in range(MAX_SLOTS):
                    name = _slot(index)
                    try:
                        before = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    try:
                        _file_safe(before)
                        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=self._fd)
                        try:
                            first = os.fstat(fd)
                            _need(_snapshot(first) == _snapshot(before))
                            raw = b""
                            while len(raw) <= MAX_RECEIPT_BYTES:
                                part = os.read(fd, MAX_RECEIPT_BYTES + 1 - len(raw))
                                if not part:
                                    break
                                raw += part
                            _need(len(raw) == first.st_size and _snapshot(first) == _snapshot(os.fstat(fd))
                                  == _snapshot(os.stat(name, dir_fd=self._fd, follow_symlinks=False)))
                            # Invalid content is not evidence, but a stable,
                            # safe fixed-slot regular file remains ours to
                            # remove without retaining its untrusted text.
                            seen.append((name, first))
                            receipt = _loads(raw, MAX_RECEIPT_BYTES)
                            _need(raw == _receipt_raw(receipt, target))
                            entries.append(dict(receipt=receipt, sha256=hashlib.sha256(raw).hexdigest(),
                                                bytes=len(raw), slot=index))
                        finally:
                            os.close(fd)
                    except Exception:
                        scan_ok = False
                _check_name(self._fd, target)
            except Exception:
                scan_ok = False
            except BaseException as error:
                scan_ok = False
                interruption = error
            finally:
                try:
                    _check_name(self._fd, target)
                    for name, info in seen:
                        try:
                            # Same size/times as the validated read, not merely
                            # the inode: a later mutation invalidates the scan.
                            _need(_snapshot(os.stat(name, dir_fd=self._fd, follow_symlinks=False)) == _snapshot(info))
                            _unlink_same(self._fd, name, info)
                        except Exception:
                            scan_ok = cleanup_ok = False
                    parent = Path(target["directory"]).parent
                    parent_fd = _open_directory(parent)
                    try:
                        named = os.stat(Path(target["directory"]).name, dir_fd=parent_fd, follow_symlinks=False)
                        _need((named.st_dev, named.st_ino) == (target["device"], target["inode"]))
                        os.rmdir(Path(target["directory"]).name, dir_fd=parent_fd)
                    finally:
                        os.close(parent_fd)
                except Exception:
                    cleanup_ok = False
                except BaseException as error:
                    cleanup_ok = False
                    if interruption is None:
                        interruption = error
                finally:
                    try:
                        os.close(self._fd)
                    except Exception:
                        cleanup_ok = False
                    except BaseException as error:
                        cleanup_ok = False
                        if interruption is None:
                            interruption = error
                    self._fd = None
                result.update(receipts=entries, slot_scan_complete=scan_ok, cleanup_complete=cleanup_ok)
                # Store the fail-closed snapshot before optional validation:
                # even a control-flow interruption after FD retirement must
                # make a repeated collect return unavailable, never retry IO.
                self._result = result
                if scan_ok and cleanup_ok and entries and interruption is None:
                    candidate = dict(result, status="captured", reason=None, receipt_count=len(entries),
                                     receipt_bytes=sum(entry["bytes"] for entry in entries))
                    try:
                        captured_receipt_bytes(candidate, target)
                    except Exception:
                        pass
                    except BaseException as error:
                        interruption = error
                    else:
                        result = candidate
                elif not cleanup_ok and scan_ok:
                    result["reason"] = "cleanup_failed"
                elif scan_ok and cleanup_ok and not entries:
                    result["reason"] = "capture_unavailable"
                self._result = result
            if interruption is not None:
                raise interruption
            return _loads(canonical_json(result), MAX_SLOTS * MAX_RECEIPT_BYTES + 16384)


def create_outbox(*, workspace, session_id, descriptor_sha256, invocation_id):
    """Create one worker-private temporary directory outside the task checkout."""
    workspace = _path(str(workspace))
    _need(_hex(session_id, 32) and _hex(descriptor_sha256, 64) and _hex(invocation_id, 32))
    parent = Path(tempfile.gettempdir()).resolve(strict=True)
    _need(not parent.is_relative_to(workspace))
    directory = Path(tempfile.mkdtemp(prefix="repository-worker-" + invocation_id + "-", dir=parent))
    created = directory.lstat()
    fd = None
    try:
        fd = _open_directory(directory)
        opened = os.fstat(fd)
        _need(stat.S_ISDIR(opened.st_mode) and opened.st_uid == created.st_uid == os.getuid()
              and (opened.st_dev, opened.st_ino) == (created.st_dev, created.st_ino))
        os.fchmod(fd, 0o700)
        info = os.fstat(fd)
        target = validate_target(dict(schema=TARGET_SCHEMA, directory=str(directory), device=info.st_dev,
            inode=info.st_ino, uid=info.st_uid, session_id=session_id, descriptor_sha256=descriptor_sha256,
            invocation_id=invocation_id), workspace=workspace)
        _check_name(fd, target)
        return _Outbox(target, fd)
    except BaseException:
        if fd is not None:
            os.close(fd)
        # No recursively removed children and no pathname fallback after a
        # replacement: rmdir only the exact freshly created directory inode.
        try:
            if directory.lstat().st_ino == created.st_ino and directory.lstat().st_dev == created.st_dev:
                directory.rmdir()
        except OSError:
            pass
        raise
