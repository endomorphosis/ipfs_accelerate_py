"""Opt-in, bounded maintenance pauses; manual stop markers never expire.

Keep watchdogs running during maintenance. Only this protocol's unchanged,
owned marker may be archived after both its deadline and its owner's death.
Releasing a pause grants no task, checkout, provider or completion authority.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

SCHEMA = "ipfs_accelerate_py/maintenance-hold@1"
MAX_SECONDS = 1800
MAX_BYTES = 16384
FIELDS = {"schema", "scope", "token", "path", "reason", "created_at", "expires_at", "owner"}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _identity(pid):
    """None means proven absent; inspection errors remain errors."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("positive maintenance owner PID required")
    directory = Path("/proc") / str(pid)
    try:
        before = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        identity = {"pid": pid, "start_ticks": before[19], "uid": directory.stat().st_uid,
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "pid_namespace": os.readlink(directory / "ns/pid")}
        after = (directory / "stat").read_text().rsplit(") ", 1)[1].split()
        if before[19] != after[19]:
            raise ValueError("maintenance owner changed during observation")
        return None if after[0] in {"Z", "X"} else identity
    except (FileNotFoundError, ProcessLookupError):
        return None


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_mode & 0o077 or before.st_size > MAX_BYTES):
            raise ValueError("owned private ordinary maintenance marker required")
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink", "st_mode")
    current = path.lstat()
    if len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k)
            or getattr(before, k) != getattr(current, k) for k in keys):
        raise ValueError("maintenance marker changed during read")
    return raw, current


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _lock(path):
    if path.parent.resolve() != path.parent:
        raise ValueError("physical maintenance directory required")
    directory = path.parent.stat()
    if directory.st_uid != os.getuid() or directory.st_mode & 0o022:
        raise ValueError("owned maintenance directory required")
    lock = path.with_name("." + path.name + ".maintenance.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise ValueError("owned maintenance lock required")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _members(path, token):
    return (path.with_name("." + path.name + ".maintenance-" + token),
            path.with_name(path.name + ".released-" + token + ".json"))


def _registration(reserve):
    return reserve.with_name(reserve.name + ".json")


def _write_new(path, raw):
    fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _load(path, now):
    raw, info = _read(path)
    payload = json.loads(raw)
    if (type(payload) is not dict or set(payload) != FIELDS or payload["schema"] != SCHEMA
            or payload["scope"] != "maintenance_pause" or payload["path"] != str(path)
            or not re.fullmatch("[0-9a-f]{32}", str(payload["token"]))
            or type(payload["reason"]) is not str or not payload["reason"]):
        raise ValueError("explicit maintenance lease required")
    created, expires = payload["created_at"], payload["expires_at"]
    if (not all(_finite(v) for v in (created, expires, now))
            or not 0 < created < expires <= created + MAX_SECONDS or now < created):
        raise ValueError("finite bounded maintenance deadline required")
    owner = payload["owner"]
    if (type(owner) is not dict or set(owner) != {"pid", "start_ticks", "uid", "boot_id", "pid_namespace"}
            or type(owner["pid"]) is not int or owner["pid"] <= 0
            or type(owner["uid"]) is not int or owner["uid"] != os.getuid()
            or not all(type(owner[k]) is str and owner[k] for k in ("start_ticks", "boot_id", "pid_namespace"))):
        raise ValueError("exact maintenance owner identity required")
    reserve, archive = _members(path, payload["token"])
    registration, registered_info = _read(_registration(reserve))
    expected = {"body_sha256": hashlib.sha256(raw).hexdigest(), "device": info.st_dev,
                "inode": info.st_ino, "mtime_ns": info.st_mtime_ns, "size": len(raw)}
    if registered_info.st_nlink != 1 or json.loads(registration) != expected:
        raise ValueError("maintenance registration changed")
    members = [p for p in (path, reserve, archive) if os.path.lexists(p)]
    # The retained reserve authenticates publication and crash windows. An
    # arbitrary JSON file with this schema is never adopted as a leased hold.
    if reserve not in members:
        raise ValueError("registered maintenance inode missing")
    for member in members:
        data, item = _read(member)
        if (data != raw or (item.st_dev, item.st_ino) != (info.st_dev, info.st_ino)
                or item.st_nlink != len(members)):
            raise ValueError("maintenance inode custody changed")
    return payload, raw, info, reserve, archive


def create_hold(path: Path, *, reason: str, ttl_seconds: float, now=None):
    """Pause from the maintenance process itself; never replace any stop marker."""
    path = Path(path).absolute()
    now = time.time() if now is None else now
    if (not _finite(ttl_seconds) or not 0 < ttl_seconds <= MAX_SECONDS
            or not _finite(now) or now <= 0 or now + ttl_seconds <= now
            or type(reason) is not str or not reason or len(reason.encode()) > 4096):
        raise ValueError("bounded maintenance reason and deadline required")
    token = uuid.uuid4().hex
    payload = {"schema": SCHEMA, "scope": "maintenance_pause", "token": token, "path": str(path),
               "reason": reason, "created_at": now, "expires_at": now + ttl_seconds,
               "owner": _identity(os.getpid())}
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    reserve, _archive = _members(path, token)
    with _lock(path):
        if os.path.lexists(path):
            raise FileExistsError("existing stop marker requires explicit release")
        _write_new(reserve, raw)
        info = reserve.stat()
        _write_new(_registration(reserve), json.dumps({"body_sha256": hashlib.sha256(raw).hexdigest(),
            "device": info.st_dev, "inode": info.st_ino, "mtime_ns": info.st_mtime_ns,
            "size": len(raw)}, sort_keys=True).encode())
        _sync(path.parent)
        os.link(reserve, path, follow_symlinks=False)
        _sync(path.parent)
    return payload


def hold_binding(path: Path, *, now=None):
    """Read-only binding for a native paused-state projection."""
    path = Path(path).absolute()
    try:
        payload, raw, _info, _reserve, _archive = _load(path, time.time() if now is None else now)
        return {"path": str(path), "token": payload["token"], "sha256": hashlib.sha256(raw).hexdigest()}
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError):
        return {}


def binding_released(binding):
    """An old native hold flag can clear only with its exact archived release."""
    try:
        if (type(binding) is not dict or set(binding) != {"path", "token", "sha256"}
                or type(binding["path"]) is not str or not Path(binding["path"]).is_absolute()
                or not re.fullmatch("[0-9a-f]{32}", str(binding["token"]))
                or not re.fullmatch("[0-9a-f]{64}", str(binding["sha256"]))):
            return False
        path = Path(binding["path"])
        if path.parent.resolve() != path.parent or os.path.lexists(path):
            return False
        _reserve, archive = _members(path, binding["token"])
        raw, info = _read(archive)
        registration, registered_info = _read(archive.with_name(archive.name + ".lease.json"))
        payload = json.loads(raw)
        return bool(hashlib.sha256(raw).hexdigest() == binding["sha256"]
            and payload.get("schema") == SCHEMA and payload.get("scope") == "maintenance_pause"
            and payload.get("path") == binding["path"] and payload.get("token") == binding["token"]
            and registered_info.st_nlink == 1 and json.loads(registration) == {
                "body_sha256": binding["sha256"], "device": info.st_dev, "inode": info.st_ino,
                "mtime_ns": info.st_mtime_ns, "size": len(raw)})
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError):
        return False


def review_hold(path: Path, *, apply=False, now=None, owner_token=None):
    """Archive only an expired dead owner's lease, or the exact current owner's."""
    path = Path(path).absolute()
    result = {"status": "retained", "path": str(path), "completion_authority": False}
    if not os.path.lexists(path):
        return {**result, "status": "absent"}
    now = time.time() if now is None else now
    try:
        # Unknown/manual holds cause no lock creation or other side effects.
        observed = _load(path, now)
        if not apply:
            # Observation must not acquire/create a writable lock. Reuse the
            # fully verified read; uncertainty retains the marker below.
            payload = observed[0]
            owner = payload["owner"]
            current = _identity(os.getpid())
            explicit_release = owner_token == payload["token"] and current == owner
            if not explicit_release:
                if now < payload["expires_at"]:
                    return {**result, "reason": "maintenance_deadline_pending"}
                if current["boot_id"] == owner["boot_id"]:
                    if current["pid_namespace"] != owner["pid_namespace"]:
                        return {**result, "reason": "maintenance_owner_namespace_unverified"}
                    owner_now = _identity(owner["pid"])
                    if owner_now == owner:
                        return {**result, "reason": "maintenance_owner_alive"}
                    if owner_now and (owner_now["boot_id"], owner_now["pid_namespace"]) != (owner["boot_id"], owner["pid_namespace"]):
                        return {**result, "reason": "maintenance_owner_identity_unverified"}
            return {**result, "status": "release_due", "reason": "maintenance_lease_expired_owner_absent"}
        with _lock(path):
            payload, raw, info, reserve, archive = _load(path, now)
            owner = payload["owner"]
            current = _identity(os.getpid())
            explicit_release = owner_token == payload["token"] and current == owner
            if not explicit_release:
                if now < payload["expires_at"]:
                    return {**result, "reason": "maintenance_deadline_pending"}
                if current["boot_id"] == owner["boot_id"]:
                    if current["pid_namespace"] != owner["pid_namespace"]:
                        return {**result, "reason": "maintenance_owner_namespace_unverified"}
                    observed = _identity(owner["pid"])
                    if observed == owner:
                        return {**result, "reason": "maintenance_owner_alive"}
                    if observed and (observed["boot_id"], observed["pid_namespace"]) != (owner["boot_id"], owner["pid_namespace"]):
                        return {**result, "reason": "maintenance_owner_identity_unverified"}
            if not apply:
                return {**result, "status": "release_due", "reason": "maintenance_lease_expired_owner_absent"}
            if not os.path.lexists(archive):
                os.link(reserve, archive, follow_symlinks=False)
                _sync(path.parent)
            registration, _ = _read(_registration(reserve))
            archived_registration = archive.with_name(archive.name + ".lease.json")
            if os.path.lexists(archived_registration):
                if _read(archived_registration)[0] != registration:
                    raise ValueError("maintenance archive registration conflict")
            else:
                _write_new(archived_registration, registration)
                _sync(path.parent)
            # Re-read the entire binding after archiving and before removal.
            check, check_raw, check_info, _, _ = _load(path, now)
            if check != payload or check_raw != raw or (check_info.st_dev, check_info.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("maintenance hold changed before release")
            path.unlink()
            _sync(path.parent)
            reserve.unlink()
            _registration(reserve).unlink()
            _sync(path.parent)
            return {**result, "status": "released", "reason": "maintenance_owner_released" if explicit_release else "maintenance_lease_expired_owner_absent",
                    "archive": str(archive), "sha256": hashlib.sha256(raw).hexdigest()}
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError) as error:
        return {**result, "reason": "maintenance_lease_unverified", "error_type": type(error).__name__}


@contextmanager
def maintenance_hold(path: Path, *, reason: str, ttl_seconds=MAX_SECONDS):
    lease = create_hold(path, reason=reason, ttl_seconds=ttl_seconds)
    try:
        yield lease
    finally:
        review_hold(path, apply=True, owner_token=lease["token"])
