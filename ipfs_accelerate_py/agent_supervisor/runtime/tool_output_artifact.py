"""Opt-in command output previews with bounded, exact local replay.

Commands run as argv with stdin closed. stdout and stderr remain separate.
Every command's process group is fenced before returning; this is not a
sandbox for commands which deliberately escape their process group. Only
complete captures are retained. Previews are lossy UTF-8 displays; replay
returns exact base64 bytes. The private store belongs to one invocation and
must be allocated outside its checkout by the supervisor owner.
Catchable CLI cancellation fences the group. SIGKILL cannot run cleanup;
the owner must retain its worker/container process cleanup boundary.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import uuid

MAX_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_STORE_BYTES = 32 * 1024 * 1024
MAX_ARTIFACTS = 32
MAX_CAPTURES = 32
MAX_PREVIEW_BYTES = 8192
MAX_READ_BYTES = 8192
MAX_RECEIPT_BYTES = 8192
MAX_TIMEOUT_SECONDS = 120
LOCK_TIMEOUT_SECONDS = 2.0
PIPE_STALL_SECONDS = 1.0
_HASH = re.compile(r"[0-9a-f]{64}")
_RECEIPT = re.compile(r"capture-[0-9a-f]{32}\.json")
_PARTIAL = re.compile(r"\.partial-[0-9a-f]{32}\.(stdout|stderr)")
_TEMP_RECEIPT = re.compile(r"\.receipt-[0-9a-f]{32}")


class _CaptureCancelled(Exception):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _load_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate receipt key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (RecursionError, UnicodeError) as error:
        raise ValueError("invalid bounded receipt JSON") from error


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(name + " exceeds its integer bound")
    return value


def _directory(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or str(path) != os.fspath(path):
        raise ValueError("absolute canonical store path required")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        info = os.fstat(fd)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("store must be owned by this user with mode 0700")
        return fd
    except BaseException:
        os.close(fd)
        raise


def create_store(path):
    """Create one new private store; its existing parent must be a real directory."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.name in {"", ".", ".."}:
        raise ValueError("absolute store path required")
    # Parent directories may have ordinary permissions, but never be symlinks.
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parent.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = nxt
        os.mkdir(path.name, 0o700, dir_fd=parent)
    finally:
        os.close(parent)
    with _store(path) as fd:
        info = os.fstat(fd)
    return {"schema": "tool-output-artifact-store@1", "created": True,
            "store_identity": {"device": info.st_dev, "inode": info.st_ino}}


def _regular(info):
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("private singleton regular store file required")


def _identity(info):
    return tuple(getattr(info, key) for key in
        ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _read(fd, name, maximum):
    handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(handle)
        _regular(before)
        if before.st_size > maximum:
            raise ValueError("store file exceeds its byte bound")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(handle, min(65536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if len(raw) != before.st_size or _identity(before) != _identity(os.fstat(handle)):
            raise ValueError("store file changed during read")
        return bytes(raw)
    finally:
        os.close(handle)


def _inventory(fd, *, cleanup=False):
    total, artifacts, receipts = 0, [], []
    with os.scandir(fd) as entries:
        for index, entry in enumerate(entries):
            if index >= 2 * MAX_CAPTURES + MAX_ARTIFACTS + 4:
                raise ValueError("store entry count exceeds bound")
            name = entry.name
            info = entry.stat(follow_symlinks=False)
            _regular(info)
            total += info.st_size
            if total > MAX_STORE_BYTES:
                raise ValueError("store byte quota exceeded")
            if name == ".lock":
                if info.st_size:
                    raise ValueError("store lock must be empty")
            elif name == "reads.json":
                if info.st_size > MAX_RECEIPT_BYTES:
                    raise ValueError("read receipt exceeds bound")
            elif name.endswith(".bin") and _HASH.fullmatch(name[:-4]):
                if info.st_size > MAX_CAPTURE_BYTES:
                    raise ValueError("capture artifact exceeds bound")
                artifacts.append(name)
            elif _RECEIPT.fullmatch(name):
                if info.st_size > MAX_RECEIPT_BYTES:
                    raise ValueError("capture receipt exceeds bound")
                receipts.append(name)
            elif cleanup and (_PARTIAL.fullmatch(name) or _TEMP_RECEIPT.fullmatch(name)):
                limit = MAX_CAPTURE_BYTES if _PARTIAL.fullmatch(name) else MAX_RECEIPT_BYTES
                if info.st_size > limit:
                    raise ValueError("unfinished store artifact exceeds bound")
            else:
                # A crashed writer's partial capture is deliberately not replayable.
                raise ValueError("unrecognized or unfinished store entry")
    if len(artifacts) > MAX_ARTIFACTS or len(receipts) > MAX_CAPTURES:
        raise ValueError("store artifact or capture quota exceeded")
    return total, artifacts, receipts


@contextmanager
def _store(path, *, cleanup=False):
    fd, lock = _directory(path), None
    try:
        lock = os.open(".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                       0o600, dir_fd=fd)
        _regular(os.fstat(lock))
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError("store lock acquisition timed out")
                time.sleep(0.02)
        if _identity(os.fstat(lock)) != _identity(os.stat(".lock", dir_fd=fd, follow_symlinks=False)):
            raise ValueError("store lock identity changed")
        _inventory(fd, cleanup=cleanup)
        yield fd
    finally:
        if lock is not None:
            os.close(lock)
        os.close(fd)


def _write_receipt(fd, name, value, *, replace=False):
    raw = _json(value)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("receipt exceeds bound")
    temporary = ".receipt-" + uuid.uuid4().hex
    handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=fd)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if not replace:
            os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        else:
            os.rename(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass


def _alive_group(group):
    """Linux group membership excludes dead zombies awaiting their parent's reap."""
    if not Path("/proc").is_dir():
        try:
            os.killpg(group, 0)
            return True
        except ProcessLookupError:
            return False
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                with open(entry.path + "/stat", "rb") as stream:
                    raw = stream.read(4096)
                fields = raw.rsplit(b")", 1)[1].split()
                if int(fields[2]) == group and fields[0] not in {b"Z", b"X"}:
                    return True
            except (FileNotFoundError, ProcessLookupError):
                continue
    return False


def _fence(process):
    # Do not reap the leader until group signaling is over: retain its PID.
    observation_failed = False
    def alive():
        nonlocal observation_failed
        try:
            return _alive_group(process.pid)
        except (OSError, ValueError, IndexError):
            observation_failed = True
            return None
    def send(signum):
        nonlocal observation_failed
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        except OSError:
            observation_failed = True
    send(signal.SIGTERM)
    deadline = time.monotonic() + 0.2
    while alive() is True and time.monotonic() < deadline:
        time.sleep(0.01)
    if alive() is not False:
        # Failed observation must never skip the owned group's final signal.
        send(signal.SIGKILL)
    deadline = time.monotonic() + 1.0
    while alive() is True and time.monotonic() < deadline:
        time.sleep(0.01)
    complete = alive() is False and not observation_failed
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        complete = False
    return complete


def _exited_without_reaping(process):
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _preview(head, tail, size, limit):
    if size <= limit:
        raw, tail = head, b""
    else:
        raw = head[: (limit + 1) // 2]
        tail = tail[-(limit // 2):] if limit // 2 else b""
    result = {"head_text": raw.decode("utf-8", errors="replace"),
            "tail_text": tail.decode("utf-8", errors="replace"),
            "display_encoding": "utf-8",
            "omitted_bytes": max(0, size - len(raw) - len(tail)),
            "preview_bytes": len(raw) + len(tail)}
    try:
        raw.decode("utf-8", errors="strict")
        tail.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        result.update(display_encoding="utf-8-with-replacement",
            head_base64=base64.b64encode(raw).decode("ascii"),
            tail_base64=base64.b64encode(tail).decode("ascii"))
    return result


def run(*, store, argv, timeout=60, preview_bytes=2048, cancel_check=None):
    """Capture a single real command without reruns or implicit shell evaluation."""
    if (sys.platform != "linux" or not callable(getattr(os, "waitid", None))
            or not hasattr(os, "WNOWAIT") or not Path("/proc").is_dir()):
        raise ValueError("output capture requires Linux /proc and waitid WNOWAIT")
    if (not isinstance(argv, (list, tuple)) or not 1 <= len(argv) <= 256
            or any(type(a) is not str or "\x00" in a or len(a.encode()) > 8192 for a in argv)
            or not argv[0] or sum(len(a.encode()) for a in argv) > 65536):
        raise ValueError("bounded explicit command argv required")
    if (type(timeout) not in {int, float} or not math.isfinite(timeout)
            or not 0 < timeout <= MAX_TIMEOUT_SECONDS):
        raise ValueError("timeout must be positive and at most 120 seconds")
    _integer(preview_bytes, 0, MAX_PREVIEW_BYTES, "preview_bytes")
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable")
    def cancelled():
        return cancel_check is not None and cancel_check()
    with _store(store) as fd:
        used, artifacts, receipts = _inventory(fd)
        if len(receipts) >= MAX_CAPTURES or len(artifacts) + 2 > MAX_ARTIFACTS:
            raise ValueError("store capture or artifact quota exhausted")
        # Reserve a capture receipt and a replacement read-count receipt.
        capacity = min(MAX_CAPTURE_BYTES, MAX_STORE_BYTES - used - 2 * MAX_RECEIPT_BYTES)
        if capacity < 1:
            raise ValueError("store byte quota exhausted")
        identity = uuid.uuid4().hex
        streams, process, selector = {}, None, selectors.DefaultSelector()
        observed, reason, cleanup = 0, None, False
        started, parent_exit = time.monotonic(), None
        try:
            for name in ("stdout", "stderr"):
                temporary = ".partial-" + identity + "." + name
                handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o600, dir_fd=fd)
                streams[name] = dict(temporary=temporary, handle=handle, size=0,
                    digest=hashlib.sha256(), head=bytearray(), tail=bytearray())
            if cancelled():
                raise _CaptureCancelled()
            process = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, start_new_session=True)
            if cancelled():
                raise _CaptureCancelled()
            for name in streams:
                pipe = getattr(process, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            while selector.get_map():
                now = time.monotonic()
                if cancelled():
                    reason = "cancelled"
                    break
                if now - started >= timeout:
                    reason = "timeout"
                    break
                if parent_exit is None and _exited_without_reaping(process):
                    parent_exit = now
                if parent_exit is not None and now - parent_exit >= PIPE_STALL_SECONDS:
                    reason = "descendant_pipe_stall"
                    break
                for key, _ in selector.select(min(0.05, max(0, timeout - (now - started)))):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    observed += len(chunk)
                    if observed > capacity:
                        reason = "output_limit" if capacity == MAX_CAPTURE_BYTES else "store_limit"
                        break
                    value = streams[key.data]
                    view = memoryview(chunk)
                    while view:
                        count = os.write(value["handle"], view)
                        if count <= 0:
                            raise OSError("capture write made no progress")
                        view = view[count:]
                    value["digest"].update(chunk)
                    value["size"] += len(chunk)
                    limit = (preview_bytes + (key.data == "stdout")) // 2
                    value["head"].extend(chunk[:max(0, limit - len(value["head"]))])
                    if limit:
                        value["tail"].extend(chunk)
                        del value["tail"][:-limit]
                if reason:
                    break
            if reason is None:
                # Pipes can close before the child exits; still honor its deadline.
                while not _exited_without_reaping(process):
                    if cancelled():
                        reason = "cancelled"
                        break
                    if time.monotonic() - started >= timeout:
                        reason = "timeout"
                        break
                    time.sleep(0.01)
        except OSError:
            reason = "capture_io_error" if process is not None else "spawn_error"
        except _CaptureCancelled:
            reason = "cancelled"
        except BaseException:
            # Cancellation must still discard every incomplete artifact; the
            # finally block below fences and reaps the owned group first.
            for value in streams.values():
                try:
                    os.unlink(value["temporary"], dir_fd=fd)
                except FileNotFoundError:
                    pass
            raise
        finally:
            if process is not None:
                cleanup = _fence(process)
                for name in streams:
                    pipe = getattr(process, name, None)
                    if pipe is not None:
                        pipe.close()
            else:
                cleanup = True
            selector.close()
            for value in streams.values():
                os.close(value["handle"])
        if cancelled():
            reason = "cancelled"
        complete = reason is None and cleanup and process is not None
        if not cleanup:
            reason = "cleanup_failed"
        result = {"schema": "tool-output-artifact-capture@1", "capture_id": identity,
            "complete": complete, "termination_reason": reason,
            "exit_code": process.returncode if process is not None else None,
            "cleanup_complete": cleanup, "observed_bytes": observed,
            "retained_bytes": sum(v["size"] for v in streams.values()) if complete else 0,
            "preview_budget_bytes": preview_bytes, "streams": {},
            "proof_authority": False, "completion_authority": False}
        try:
            for name, value in streams.items():
                digest = value["digest"].hexdigest()
                if complete:
                    destination = digest + ".bin"
                    try:
                        os.link(value["temporary"], destination, src_dir_fd=fd, dst_dir_fd=fd,
                                follow_symlinks=False)
                    except FileExistsError:
                        _verify_artifact(fd, digest, offset=0, length=0)
                    os.unlink(value["temporary"], dir_fd=fd)
                    _verify_artifact(fd, digest, offset=0, length=0)
                limit = (preview_bytes + (name == "stdout")) // 2
                result["streams"][name] = {"sha256": digest if complete else None,
                    "bytes": value["size"], "replayable": complete,
                    **_preview(bytes(value["head"]), bytes(value["tail"]), value["size"], limit)}
            receipt = {**result, "streams": {name: {key: value[key] for key in
                ("sha256", "bytes", "replayable", "omitted_bytes", "preview_bytes")}
                for name, value in result["streams"].items()}}
            _write_receipt(fd, "capture-" + identity + ".json", receipt)
        finally:
            for value in streams.values():
                try:
                    os.unlink(value["temporary"], dir_fd=fd)
                except FileNotFoundError:
                    pass
        _inventory(fd)
        return result


def _verify_artifact(fd, digest, *, offset, length):
    if type(digest) is not str or not _HASH.fullmatch(digest):
        raise ValueError("exact artifact SHA256 required")
    handle = os.open(digest + ".bin", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(handle)
        _regular(before)
        if before.st_size > MAX_CAPTURE_BYTES or offset > before.st_size:
            raise ValueError("artifact or read offset exceeds bound")
        actual, selected, position = hashlib.sha256(), bytearray(), 0
        while True:
            chunk = os.read(handle, 65536)
            if not chunk:
                break
            actual.update(chunk)
            start, end = max(0, offset - position), min(len(chunk), offset + length - position)
            if start < end:
                selected.extend(chunk[start:end])
            position += len(chunk)
            if position > MAX_CAPTURE_BYTES:
                raise ValueError("artifact grew beyond capture bound")
        after = os.fstat(handle)
        named = os.stat(digest + ".bin", dir_fd=fd, follow_symlinks=False)
        if (_identity(before) != _identity(after) or _identity(after) != _identity(named)
                or position != before.st_size or actual.hexdigest() != digest):
            raise ValueError("artifact identity or digest changed")
        return bytes(selected), position
    finally:
        os.close(handle)


def _read_counts(fd):
    try:
        value = _load_json(_read(fd, "reads.json", MAX_RECEIPT_BYTES))
    except FileNotFoundError:
        return {"read_count": 0, "read_bytes": 0}
    if type(value) is not dict or set(value) != {"read_count", "read_bytes"}:
        raise ValueError("invalid read-count receipt")
    _integer(value["read_count"], 0, 100000, "read_count")
    _integer(value["read_bytes"], 0, 1024**3, "read_bytes")
    return value


def read(*, store, sha256, offset=0, length=4096):
    """Hash the complete artifact before returning one exact bounded byte page."""
    _integer(offset, 0, MAX_CAPTURE_BYTES, "offset")
    _integer(length, 1, MAX_READ_BYTES, "length")
    with _store(store) as fd:
        data, total = _verify_artifact(fd, sha256, offset=offset, length=length)
        counts = _read_counts(fd)
        if counts["read_count"] >= 100000 or counts["read_bytes"] + len(data) > 1024**3:
            raise ValueError("invocation retrieval quota exhausted")
        counts["read_count"] += 1
        counts["read_bytes"] += len(data)
        _write_receipt(fd, "reads.json", counts, replace=True)
        result = {"schema": "tool-output-artifact-page@1", "sha256": sha256,
            "offset": offset, "length": len(data), "total_bytes": total,
            "next_offset": offset + len(data), "eof": offset + len(data) == total,
            "data_base64": base64.b64encode(data).decode("ascii"),
            "proof_authority": False, "completion_authority": False}
        try:
            result["text"] = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            result["text"] = None
        return result


def summarize_store(path):
    """Content-free usage observations; worker-owned records are not proof."""
    with _store(path) as fd:
        total, artifacts, receipts = _inventory(fd)
        result = {"schema": "tool-output-artifact-usage@1", "capture_count": len(receipts),
            "complete_capture_count": 0, "incomplete_capture_count": 0,
            "command_failure_count": 0, "observed_bytes": 0, "retained_bytes": 0,
            "preview_bytes": 0, "omitted_bytes": 0, "artifact_count": len(artifacts),
            "store_bytes": total, **_read_counts(fd), "raw_content_exported": False,
            "worker_owned_observation": True, "proof_authority": False,
            "completion_authority": False}
        verified = {}
        for name in receipts:
            value = _load_json(_read(fd, name, MAX_RECEIPT_BYTES))
            if (type(value) is not dict or set(value) != {"schema", "capture_id", "complete",
                    "termination_reason", "exit_code", "cleanup_complete", "observed_bytes",
                    "retained_bytes", "preview_budget_bytes", "streams", "proof_authority",
                    "completion_authority"}
                    or value.get("schema") != "tool-output-artifact-capture@1"
                    or value.get("capture_id") != name[8:-5]
                    or type(value.get("complete")) is not bool
                    or type(value.get("cleanup_complete")) is not bool
                    or type(value.get("streams")) is not dict
                    or not set(value["streams"]) <= {"stdout", "stderr"}
                    or value.get("proof_authority") is not False
                    or value.get("completion_authority") is not False
                    or value.get("termination_reason") is not None and type(value.get("termination_reason")) is not str
                    or value.get("termination_reason") not in {None, "timeout", "descendant_pipe_stall",
                        "output_limit", "store_limit", "capture_io_error", "spawn_error", "cleanup_failed", "cancelled"}):
                raise ValueError("invalid capture receipt")
            exit_code = value["exit_code"]
            if exit_code is not None:
                _integer(exit_code, -255, 255, "exit_code")
            _integer(value["preview_budget_bytes"], 0, MAX_PREVIEW_BYTES, "preview_budget_bytes")
            for key in ("observed_bytes", "retained_bytes"):
                _integer(value[key], 0, MAX_CAPTURE_BYTES + 65536, key)
            size, preview = 0, 0
            for stream in value["streams"].values():
                if (type(stream) is not dict or set(stream) != {"sha256", "bytes", "replayable",
                        "omitted_bytes", "preview_bytes"}
                        or type(stream["replayable"]) is not bool
                        or stream["replayable"] != value["complete"]):
                    raise ValueError("invalid capture stream receipt")
                amount = _integer(stream["bytes"], 0, MAX_CAPTURE_BYTES, "stream bytes")
                shown = _integer(stream["preview_bytes"], 0, MAX_PREVIEW_BYTES, "preview bytes")
                omitted = _integer(stream["omitted_bytes"], 0, MAX_CAPTURE_BYTES, "omitted bytes")
                if shown + omitted != amount:
                    raise ValueError("capture stream byte census differs")
                size += amount
                preview += shown
                digest = stream["sha256"]
                if value["complete"]:
                    if type(digest) is not str or not _HASH.fullmatch(digest):
                        raise ValueError("complete stream lacks artifact identity")
                    if digest not in verified:
                        verified[digest] = _verify_artifact(fd, digest, offset=0, length=0)[1]
                    if verified[digest] != amount:
                        raise ValueError("capture stream size differs from artifact")
                elif digest is not None:
                    raise ValueError("incomplete stream claims replay identity")
            if (size > MAX_CAPTURE_BYTES or preview > value["preview_budget_bytes"]
                    or value["observed_bytes"] < size
                    or value["retained_bytes"] != (size if value["complete"] else 0)
                    or value["complete"] and (set(value["streams"]) != {"stdout", "stderr"}
                        or value["observed_bytes"] != size or not value["cleanup_complete"]
                        or value["termination_reason"] is not None or exit_code is None)
                    or not value["complete"] and value["termination_reason"] is None):
                raise ValueError("capture completeness or byte census differs")
            result["complete_capture_count" if value["complete"] else "incomplete_capture_count"] += 1
            result["command_failure_count"] += value.get("exit_code") not in {0, None}
            for key in ("observed_bytes", "retained_bytes"):
                result[key] += _integer(value.get(key), 0, MAX_CAPTURE_BYTES + 65536, key)
            for stream in value["streams"].values():
                for key in ("preview_bytes", "omitted_bytes"):
                    result[key] += _integer(stream.get(key), 0, MAX_CAPTURE_BYTES, key)
        return result


def remove_store(path, *, expected_identity):
    """Remove only the original bounded invocation store, without recursion.

    The owner calls this after the provider returns. A changed root or unknown
    entry causes explicit failure; this helper never follows links or removes
    unrelated directories as a fallback.
    """
    if (type(expected_identity) is not dict or set(expected_identity) != {"device", "inode"}
            or any(type(value) is not int or value < 0 for value in expected_identity.values())):
        raise ValueError("original store identity required for cleanup")
    path = Path(path)
    with _store(path, cleanup=True) as fd:
        info = os.fstat(fd)
        if expected_identity != {"device": info.st_dev, "inode": info.st_ino}:
            raise ValueError("store root identity changed before cleanup")
        parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parent.parts[1:]:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = nxt
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("store directory changed during cleanup")
            # Inventory already proved bounded singleton regular files. Unlink
            # removes a substituted symlink itself, never its target.
            names = []
            with os.scandir(fd) as entries:
                for index, entry in enumerate(entries):
                    if index >= 2 * MAX_CAPTURES + MAX_ARTIFACTS + 4:
                        raise ValueError("store grew during cleanup")
                    names.append(entry.name)
            for name in names:
                if name != ".lock":
                    os.unlink(name, dir_fd=fd)
            os.unlink(".lock", dir_fd=fd)
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("store directory changed during cleanup")
            os.rmdir(path.name, dir_fd=parent)
        finally:
            os.close(parent)
    return {"schema": "tool-output-artifact-cleanup@1", "removed": True}


def _preview_parts(stream):
    """Recover exact bounded preview segments from a trusted version-one result."""
    encoding = stream["display_encoding"]
    parts = []
    for name in ("head", "tail"):
        if encoding == "utf-8":
            text = stream[name + "_text"]
            if type(text) is not str or len(text) > MAX_PREVIEW_BYTES:
                raise ValueError("bounded preview text required")
            raw = text.encode("utf-8", errors="strict")
        elif encoding == "utf-8-with-replacement":
            encoded = stream[name + "_base64"]
            if type(encoded) is not str or len(encoded) > 4 * ((MAX_PREVIEW_BYTES + 2) // 3):
                raise ValueError("bounded exact preview base64 required")
            raw = base64.b64decode(encoded, validate=True)
        else:
            raise ValueError("unsupported preview encoding")
        parts.append(raw)
    if sum(map(len, parts)) > MAX_PREVIEW_BYTES:
        raise ValueError("preview projection exceeds byte bound")
    return parts


def format_response(result, response_format="legacy"):
    """Project a trusted internal result without I/O or changing stored receipts.

    Legacy returns the original dictionary unchanged. Compact uses exactly one
    representation for each complete byte sequence. Partial UTF-8 code points
    stay exact base64; compact responses never introduce replacement text.
    """
    if type(response_format) is not str or response_format not in {"legacy", "compact"}:
        raise ValueError("explicit supported response format required")
    if response_format == "legacy":
        return result
    if type(result) is not dict:
        raise ValueError("structured internal tool output result required")
    schema = result.get("schema")
    if schema == "tool-output-artifact-capture@1":
        projected = {"schema": "tool-output-artifact-capture@2", **{key: result[key] for key in
            ("complete", "exit_code", "cleanup_complete", "termination_reason",
             "observed_bytes", "retained_bytes", "proof_authority", "completion_authority")},
             "streams": {}}
        streams = result["streams"]
        if type(streams) is not dict or not set(streams) <= {"stdout", "stderr"}:
            raise ValueError("bounded stdout/stderr result required")
        preview_total = 0
        for name, stream in streams.items():
            head, tail = _preview_parts(stream)
            preview_total += len(head) + len(tail)
            size = _integer(stream["bytes"], 0, MAX_CAPTURE_BYTES, "stream bytes")
            omitted = _integer(stream["omitted_bytes"], 0, size, "omitted bytes")
            if len(head) + len(tail) != size - omitted:
                raise ValueError("preview projection byte census differs")
            item = {"bytes": stream["bytes"], "omitted_bytes": stream["omitted_bytes"]}
            if stream["omitted_bytes"] == 0:
                raw = head + tail
                try:
                    item["text"] = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    item["data_base64"] = base64.b64encode(raw).decode("ascii")
            else:
                try:
                    head_text = head.decode("utf-8", errors="strict")
                    tail_text = tail.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    item.update(head_base64=base64.b64encode(head).decode("ascii"),
                                tail_base64=base64.b64encode(tail).decode("ascii"))
                else:
                    item.update(head_text=head_text, tail_text=tail_text)
                if result["complete"]:
                    item["sha256"] = stream["sha256"]
            projected["streams"][name] = item
        if preview_total > MAX_PREVIEW_BYTES:
            raise ValueError("combined preview projection exceeds byte bound")
        return projected
    if schema == "tool-output-artifact-page@1":
        projected = {"schema": "tool-output-artifact-page@2", **{key: result[key] for key in
            ("sha256", "offset", "length", "total_bytes", "next_offset", "eof",
             "proof_authority", "completion_authority")}}
        encoded = result["data_base64"]
        if type(encoded) is not str or len(encoded) > 4 * ((MAX_READ_BYTES + 2) // 3):
            raise ValueError("bounded exact page base64 required")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_READ_BYTES or len(raw) != result["length"]:
            raise ValueError("page projection exceeds byte bound")
        try:
            projected["text"] = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            projected["data_base64"] = encoded
        return projected
    raise ValueError("unsupported internal tool output schema")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    capture = sub.add_parser("run")
    capture.add_argument("--store", required=True)
    capture.add_argument("--timeout", type=float, default=60)
    capture.add_argument("--preview-bytes", type=int, default=2048)
    capture.add_argument("--response-format", choices=("legacy", "compact"), default="legacy")
    capture.add_argument("argv", nargs=argparse.REMAINDER)
    page = sub.add_parser("read")
    page.add_argument("--store", required=True)
    page.add_argument("--sha256", required=True)
    page.add_argument("--offset", type=int, default=0)
    page.add_argument("--length", type=int, default=4096)
    page.add_argument("--response-format", choices=("legacy", "compact"), default="legacy")
    options = vars(parser.parse_args(argv))
    operation = options.pop("operation")
    response_format = options.pop("response_format")
    handlers = {}
    cancellation = [False]
    if operation == "run":
        def cancel(signum, frame):
            # Set a flag only: an asynchronous exception inside Popen or
            # process fencing could otherwise interrupt child custody.
            cancellation[0] = True
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, cancel)
    try:
        if operation == "run":
            if options["argv"][:1] == ["--"]:
                options["argv"] = options["argv"][1:]
            result = run(**options, cancel_check=lambda: cancellation[0])
        else:
            result = read(**options)
        print(_json(format_response(result, response_format)).decode("ascii"))
        return 0 if operation == "read" or (result["complete"] and result["exit_code"] == 0) else 1
    except (ValueError, OSError) as error:
        # Do not include command argv, arbitrary paths, or output in diagnostics.
        print(_json({"schema": "tool-output-artifact-error@1", "error": type(error).__name__,
                     "complete": False}).decode("ascii"))
        return 2
    except _CaptureCancelled:
        print(_json({"schema": "tool-output-artifact-error@1", "error": "cancelled",
                     "complete": False}).decode("ascii"))
        return 2
    finally:
        for number, previous in handlers.items():
            signal.signal(number, previous)


if __name__ == "__main__":
    raise SystemExit(main())
