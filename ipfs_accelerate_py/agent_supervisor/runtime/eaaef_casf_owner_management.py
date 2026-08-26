"""Private durable management for one committed EAAEF CASF bootstrap owner.

This module is deliberately not part of the public reconciliation-owner
interface.  It exposes two owner-local operations only: a typed cached status
snapshot and an idempotent stop request.  The channel never carries a database
path, a Quack credential, SQL, a signing key, or provider authority.

The endpoint is an abstract Unix socket so deeply nested sealed generation
paths cannot exceed the platform pathname limit.  Its random name is not an
authenticator.  Authentication requires a fresh 256-bit generation key stored
separately from a self-addressed capsule in the owner's 0700 state directory,
plus exact Linux peer credentials and a corroborated process birth on both
sides.  Stale artifacts therefore fail closed after reboot or PID reuse.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import stat
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from ..task_sources.eaaef_casf_bootstrap_owner import (
    EAAEFCASFBootstrapOwnerError,
)

EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE: Final = (
    "EAAEFCASFBootstrapOwnerManagement@1"
)
EAAEF_CASF_OWNER_MANAGEMENT_CAPSULE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-management-capsule@1"
)
EAAEF_CASF_OWNER_MANAGEMENT_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-management-request@1"
)
EAAEF_CASF_OWNER_MANAGEMENT_RESPONSE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-management-response@1"
)
EAAEF_CASF_OWNER_MANAGEMENT_STATUS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-management-status@1"
)
EAAEF_CASF_OWNER_STOP_INTENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-stop-intent@1"
)
EAAEF_CASF_OWNER_STOP_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-casf-owner-stop-result@1"
)

MANAGEMENT_KEY_NAME: Final = "management.key"
MANAGEMENT_CAPSULE_NAME: Final = "management-capsule.json"
MANAGEMENT_STOP_INTENT_NAME: Final = "management-stop-intent.json"
MANAGEMENT_STOP_RESULT_NAME: Final = "management-stop-result.json"

_MAX_FRAME_BYTES: Final = 64 * 1024
_MAX_ARTIFACT_BYTES: Final = 64 * 1024
_MAX_STATUS_REQUEST_NONCES: Final = 4096
_MAX_CONNECTION_WORKERS: Final = 8
_DEFAULT_AUTHENTICATION_TIMEOUT_SECONDS: Final = 1.0
_FRAME_HEADER = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_SHA256_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_64_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_RE: Final = re.compile(r"^eaaef-[a-z0-9][a-z0-9.-]{7,95}$")
_PROCESS_BIRTH_FIELDS: Final = frozenset(
    {"pid", "start_time_ticks", "parent_pid", "boot_id", "argv_sha256"}
)
_IMMUTABLE_PROCESS_BIRTH_FIELDS: Final = (
    "pid",
    "start_time_ticks",
    "boot_id",
    "argv_sha256",
)
_CAPSULE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "binding_cid",
        "snapshot_bindings_cid",
        "endpoint_nonce",
        "owner_process_birth",
        "key_sha256",
        "capsule_nonce",
        "capsule_cid",
    }
)
_REQUEST_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "capsule_cid",
        "client_process_birth",
        "request_nonce",
        "operation",
        "arguments",
        "request_cid",
        "mac",
    }
)
_RESPONSE_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "capsule_cid",
        "request_cid",
        "operation",
        "ok",
        "error_code",
        "result",
        "response_cid",
        "mac",
    }
)
_STATUS_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "owner_process_birth",
        "phase",
        "owner_committed",
        "owner_process_alive",
        "provider_process_started",
        "task_state_mutated",
        "owner_start_receipt_cid",
        "final_record_cid",
        "commit_receipt_cid",
        "status_cid",
    }
)
_STOP_INTENT_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "binding_cid",
        "snapshot_bindings_cid",
        "capsule_cid",
        "owner_process_birth",
        "owner_start_receipt_cid",
        "final_record_cid",
        "commit_receipt_cid",
        "stop_request_id",
        "stop_request_cid",
        "intent_cid",
        "intent_mac",
    }
)
_STOP_RESULT_FIELDS: Final = frozenset(
    {
        "schema",
        "interface",
        "generation_id",
        "binding_cid",
        "snapshot_bindings_cid",
        "capsule_cid",
        "owner_process_birth",
        "owner_start_receipt_cid",
        "final_record_cid",
        "commit_receipt_cid",
        "stop_request_id",
        "stop_request_cid",
        "stop_intent_cid",
        "committed_owner_stopped",
        "broker_process_exit_pending",
        "exclusive_owner_lease_released",
        "task_state_mutated",
        "result_cid",
        "result_mac",
    }
)
_ARGUMENT_FIELDS: Final = {
    "status.snapshot": frozenset(),
    "stop": frozenset(
        {
            "stop_request_id",
            "owner_start_receipt_cid",
            "final_record_cid",
            "commit_receipt_cid",
        }
    ),
}
_ERROR_CODES: Final = frozenset(
    {
        "owner_not_committed",
        "stop_request_diverged",
        "stop_request_failed",
        "stop_result_unavailable",
        "request_nonce_replayed",
        "request_nonce_capacity_exhausted",
    }
)


class EAAEFCASFOwnerManagementError(EAAEFCASFBootstrapOwnerError):
    """The private committed-owner management channel failed closed."""


def _canonical_bytes(value: Any, *, noun: str) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise EAAEFCASFOwnerManagementError(
            f"{noun} is not canonical JSON"
        ) from exc
    if not raw or len(raw) > _MAX_FRAME_BYTES:
        raise EAAEFCASFOwnerManagementError(f"{noun} exceeds its byte bound")
    return raw


def _cid(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_bytes(value, noun="management content identity")
    ).hexdigest()


def _decode_object(raw: object, *, noun: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_FRAME_BYTES:
        raise EAAEFCASFOwnerManagementError(
            f"{noun} is not bounded canonical bytes"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EAAEFCASFOwnerManagementError(
            f"{noun} is not canonical JSON"
        ) from exc
    if type(value) is not dict or not all(type(key) is str for key in value):
        raise EAAEFCASFOwnerManagementError(f"{noun} is not an exact object")
    if _canonical_bytes(value, noun=noun) != raw:
        raise EAAEFCASFOwnerManagementError(f"{noun} bytes are not canonical")
    return value


def _verified_cid(
    raw: Mapping[str, Any],
    *,
    fields: frozenset[str],
    cid_field: str,
    noun: str,
) -> dict[str, Any]:
    value = dict(raw)
    body = dict(value)
    claimed = body.pop(cid_field, "")
    if set(value) != fields or type(claimed) is not str or claimed != _cid(body):
        raise EAAEFCASFOwnerManagementError(f"{noun} identity differs")
    return value


def _exact_process_birth(raw: object, *, noun: str) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != _PROCESS_BIRTH_FIELDS:
        raise EAAEFCASFOwnerManagementError(f"{noun} shape differs")
    if (
        any(
            type(raw[field]) is not int
            for field in ("pid", "start_time_ticks", "parent_pid")
        )
        or any(type(raw[field]) is not str for field in ("boot_id", "argv_sha256"))
        or int(raw["pid"]) < 1
        or int(raw["start_time_ticks"]) < 1
        or int(raw["parent_pid"]) < 0
        or not str(raw["boot_id"])
        or not _SHA256_RE.fullmatch(str(raw["argv_sha256"]))
    ):
        raise EAAEFCASFOwnerManagementError(f"{noun} types differ")
    return dict(raw)


def _observed_process_birth(pid: int) -> dict[str, Any] | None:
    from . import eaaef_reconciliation_lifecycle as reconciliation

    observed = reconciliation.inspect_process_birth(pid)
    return None if observed is None else observed.to_dict()


def _corroborates_process_birth(sealed: Mapping[str, Any]) -> bool:
    """Check immutable birth fields while retaining the sealed birth PPID.

    Linux reparents a surviving child when its caller exits, so ``parent_pid``
    is provenance from process creation rather than a stable liveness field.
    PID, start ticks, boot ID, and argv digest jointly remain PID-reuse- and
    reboot-resistant.
    """

    observed = _observed_process_birth(int(sealed["pid"]))
    return observed is not None and all(
        observed[field] == sealed[field]
        for field in _IMMUTABLE_PROCESS_BIRTH_FIELDS
    )


def _current_process_birth() -> dict[str, Any]:
    observed = _observed_process_birth(os.getpid())
    if observed is None:
        raise EAAEFCASFOwnerManagementError(
            "management client process birth is unavailable"
        )
    return observed


def _private_directory(path: Path) -> tuple[Path, int]:
    lexical = Path(os.path.abspath(path))
    if lexical.resolve(strict=False) != lexical:
        raise EAAEFCASFOwnerManagementError(
            "management state directory is not sealed"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lexical, flags)
        metadata = os.fstat(descriptor)
        current = os.lstat(lexical)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or (metadata.st_dev, metadata.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise EAAEFCASFOwnerManagementError(
                "management state directory is not private"
            )
        return lexical, descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except (NameError, OSError):
            pass
        raise


def _write_once(state_dir: Path, name: str, payload: bytes) -> None:
    if type(payload) is not bytes or not payload or len(payload) > _MAX_ARTIFACT_BYTES:
        raise EAAEFCASFOwnerManagementError(
            "management artifact is not bounded bytes"
        )
    _lexical, directory_fd = _private_directory(state_dir)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
            ):
                raise EAAEFCASFOwnerManagementError(
                    "management artifact reservation is unsafe"
                )
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written < 1:
                    raise EAAEFCASFOwnerManagementError(
                        "management artifact write made no progress"
                    )
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise EAAEFCASFOwnerManagementError(
            f"management artifact {name} already exists"
        ) from exc
    except OSError as exc:
        raise EAAEFCASFOwnerManagementError(
            f"management artifact {name} is unavailable"
        ) from exc
    finally:
        os.close(directory_fd)


def _read_private(state_dir: Path, name: str) -> bytes:
    _lexical, directory_fd = _private_directory(state_dir)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or not 0 < metadata.st_size <= _MAX_ARTIFACT_BYTES
            ):
                raise EAAEFCASFOwnerManagementError(
                    f"management artifact {name} is not private"
                )
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65536))
                if not chunk:
                    raise EAAEFCASFOwnerManagementError(
                        f"management artifact {name} was truncated"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                raise EAAEFCASFOwnerManagementError(
                    f"management artifact {name} grew while reading"
                )
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (metadata.st_dev, metadata.st_ino, metadata.st_size) != (
                current.st_dev,
                current.st_ino,
                current.st_size,
            ):
                raise EAAEFCASFOwnerManagementError(
                    f"management artifact {name} changed while reading"
                )
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise EAAEFCASFOwnerManagementError(
            f"management artifact {name} is unavailable"
        ) from exc
    finally:
        os.close(directory_fd)


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = channel.recv(size - len(output))
        if not chunk:
            raise EAAEFCASFOwnerManagementError(
                "owner management channel closed early"
            )
        output.extend(chunk)
    return bytes(output)


def _recv_packet(channel: socket.socket) -> bytes:
    header = _recv_exact(channel, _FRAME_HEADER.size)
    (size,) = _FRAME_HEADER.unpack(header)
    if not 0 < size <= _MAX_FRAME_BYTES:
        raise EAAEFCASFOwnerManagementError(
            "owner management frame length is invalid"
        )
    return _recv_exact(channel, size)


def _recv_exact_before(
    channel: socket.socket,
    size: int,
    *,
    deadline: float,
) -> bytes:
    """Receive exact bytes under one monotonic deadline, including drip feed."""

    output = bytearray()
    while len(output) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EAAEFCASFOwnerManagementError(
                "owner management authentication deadline expired"
            )
        channel.settimeout(remaining)
        try:
            chunk = channel.recv(size - len(output))
        except TimeoutError as exc:
            raise EAAEFCASFOwnerManagementError(
                "owner management authentication deadline expired"
            ) from exc
        if not chunk:
            raise EAAEFCASFOwnerManagementError(
                "owner management channel closed during authentication"
            )
        output.extend(chunk)
    return bytes(output)


def _recv_packet_before(channel: socket.socket, *, deadline: float) -> bytes:
    header = _recv_exact_before(
        channel,
        _FRAME_HEADER.size,
        deadline=deadline,
    )
    (size,) = _FRAME_HEADER.unpack(header)
    if not 0 < size <= _MAX_FRAME_BYTES:
        raise EAAEFCASFOwnerManagementError(
            "owner management frame length is invalid"
        )
    return _recv_exact_before(channel, size, deadline=deadline)


def _send_packet(channel: socket.socket, raw: bytes) -> None:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_FRAME_BYTES:
        raise EAAEFCASFOwnerManagementError(
            "owner management frame is not bounded bytes"
        )
    channel.sendall(_FRAME_HEADER.pack(len(raw)) + raw)


def _mac(key: bytes, value: Mapping[str, Any]) -> str:
    return hmac.new(
        key,
        _canonical_bytes(dict(value), noun="owner management authenticator"),
        hashlib.sha256,
    ).hexdigest()


def _build_capsule(
    *,
    generation_id: str,
    binding_cid: str,
    snapshot_bindings_cid: str,
    endpoint_nonce: str,
    owner_process_birth: Mapping[str, Any],
    key: bytes,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_OWNER_MANAGEMENT_CAPSULE_SCHEMA,
        "interface": EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE,
        "generation_id": generation_id,
        "binding_cid": binding_cid,
        "snapshot_bindings_cid": snapshot_bindings_cid,
        "endpoint_nonce": endpoint_nonce,
        "owner_process_birth": dict(owner_process_birth),
        "key_sha256": hashlib.sha256(key).hexdigest(),
        "capsule_nonce": secrets.token_hex(32),
    }
    value["capsule_cid"] = _cid(value)
    return value


def _validate_capsule(
    raw: bytes,
    *,
    generation_id: str,
    binding_cid: str,
    snapshot_bindings_cid: str,
    key: bytes,
    require_live: bool = True,
) -> dict[str, Any]:
    if type(require_live) is not bool:
        raise EAAEFCASFOwnerManagementError(
            "owner management capsule liveness policy differs"
        )
    value = _verified_cid(
        _decode_object(raw, noun="owner management capsule"),
        fields=_CAPSULE_FIELDS,
        cid_field="capsule_cid",
        noun="owner management capsule",
    )
    birth = _exact_process_birth(
        value.get("owner_process_birth"), noun="management owner process birth"
    )
    if (
        value.get("schema") != EAAEF_CASF_OWNER_MANAGEMENT_CAPSULE_SCHEMA
        or value.get("interface") != EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE
        or value.get("generation_id") != generation_id
        or value.get("binding_cid") != binding_cid
        or value.get("snapshot_bindings_cid") != snapshot_bindings_cid
        or type(value.get("endpoint_nonce")) is not str
        or not _HEX_64_RE.fullmatch(str(value["endpoint_nonce"]))
        or type(value.get("capsule_nonce")) is not str
        or not _HEX_64_RE.fullmatch(str(value["capsule_nonce"]))
        or value.get("key_sha256") != hashlib.sha256(key).hexdigest()
        or (require_live and not _corroborates_process_birth(birth))
    ):
        raise EAAEFCASFOwnerManagementError(
            "owner management capsule is stale or divergent"
        )
    return value


def _endpoint(capsule: Mapping[str, Any]) -> str:
    return "\0ipfs-accelerate-eaaef-casf-" + str(capsule["endpoint_nonce"])


def _build_request(
    *,
    generation_id: str,
    capsule_cid: str,
    key: bytes,
    operation: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    if operation not in _ARGUMENT_FIELDS or set(arguments) != _ARGUMENT_FIELDS[operation]:
        raise EAAEFCASFOwnerManagementError(
            "owner management request arguments differ"
        )
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_OWNER_MANAGEMENT_REQUEST_SCHEMA,
        "interface": EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE,
        "generation_id": generation_id,
        "capsule_cid": capsule_cid,
        "client_process_birth": _current_process_birth(),
        "request_nonce": secrets.token_hex(32),
        "operation": operation,
        "arguments": dict(arguments),
    }
    value["request_cid"] = _cid(value)
    value["mac"] = _mac(key, value)
    return value


def _validate_request(
    raw: bytes,
    *,
    generation_id: str,
    capsule_cid: str,
    key: bytes,
    peer_pid: int,
) -> dict[str, Any]:
    value = _decode_object(raw, noun="owner management request")
    unsigned = dict(value)
    claimed_mac = unsigned.pop("mac", "")
    body = dict(unsigned)
    claimed_cid = body.pop("request_cid", "")
    operation = value.get("operation")
    arguments = value.get("arguments")
    birth = _exact_process_birth(
        value.get("client_process_birth"), noun="management client process birth"
    )
    if (
        set(value) != _REQUEST_FIELDS
        or value.get("schema") != EAAEF_CASF_OWNER_MANAGEMENT_REQUEST_SCHEMA
        or value.get("interface") != EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE
        or value.get("generation_id") != generation_id
        or value.get("capsule_cid") != capsule_cid
        or type(value.get("request_nonce")) is not str
        or not _HEX_64_RE.fullmatch(str(value["request_nonce"]))
        or type(operation) is not str
        or operation not in _ARGUMENT_FIELDS
        or type(arguments) is not dict
        or set(arguments) != _ARGUMENT_FIELDS[operation]
        or type(claimed_cid) is not str
        or claimed_cid != _cid(body)
        or type(claimed_mac) is not str
        or not hmac.compare_digest(claimed_mac, _mac(key, unsigned))
        or int(birth["pid"]) != peer_pid
        or not _corroborates_process_birth(birth)
    ):
        raise EAAEFCASFOwnerManagementError(
            "owner management request authentication differs"
        )
    if operation == "stop":
        stop_request_id = arguments.get("stop_request_id")
        if (
            type(stop_request_id) is not str
            or not _HEX_64_RE.fullmatch(stop_request_id)
            or any(
                type(arguments.get(field)) is not str
                or not _SHA256_RE.fullmatch(str(arguments[field]))
                for field in (
                    "owner_start_receipt_cid",
                    "final_record_cid",
                    "commit_receipt_cid",
                )
            )
        ):
            raise EAAEFCASFOwnerManagementError(
                "owner management stop request identity differs"
            )
    return value


def _build_response(
    *,
    generation_id: str,
    capsule_cid: str,
    key: bytes,
    request_cid: str,
    operation: str,
    ok: bool,
    result: Mapping[str, Any] | None = None,
    error_code: str = "",
) -> dict[str, Any]:
    if (
        type(ok) is not bool
        or (ok and error_code)
        or (not ok and error_code not in _ERROR_CODES)
    ):
        raise EAAEFCASFOwnerManagementError(
            "owner management response inputs differ"
        )
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_OWNER_MANAGEMENT_RESPONSE_SCHEMA,
        "interface": EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE,
        "generation_id": generation_id,
        "capsule_cid": capsule_cid,
        "request_cid": request_cid,
        "operation": operation,
        "ok": ok,
        "error_code": error_code,
        "result": {} if result is None else dict(result),
    }
    value["response_cid"] = _cid(value)
    value["mac"] = _mac(key, value)
    return value


def _validate_response(
    raw: bytes,
    *,
    generation_id: str,
    capsule_cid: str,
    key: bytes,
    request_cid: str,
    operation: str,
) -> dict[str, Any]:
    value = _decode_object(raw, noun="owner management response")
    unsigned = dict(value)
    claimed_mac = unsigned.pop("mac", "")
    body = dict(unsigned)
    claimed_cid = body.pop("response_cid", "")
    if (
        set(value) != _RESPONSE_FIELDS
        or value.get("schema") != EAAEF_CASF_OWNER_MANAGEMENT_RESPONSE_SCHEMA
        or value.get("interface") != EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE
        or value.get("generation_id") != generation_id
        or value.get("capsule_cid") != capsule_cid
        or value.get("request_cid") != request_cid
        or value.get("operation") != operation
        or type(value.get("ok")) is not bool
        or type(value.get("error_code")) is not str
        or type(value.get("result")) is not dict
        or (value["ok"] is True and value["error_code"])
        or (value["ok"] is False and value["error_code"] not in _ERROR_CODES)
        or type(claimed_cid) is not str
        or claimed_cid != _cid(body)
        or type(claimed_mac) is not str
        or not hmac.compare_digest(claimed_mac, _mac(key, unsigned))
    ):
        raise EAAEFCASFOwnerManagementError(
            "owner management response authentication differs"
        )
    return value


def _status_snapshot(
    *,
    generation_id: str,
    owner_process_birth: Mapping[str, Any],
    phase: str,
    owner_start_receipt_cid: str,
    final_record_cid: str,
    commit_receipt_cid: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_OWNER_MANAGEMENT_STATUS_SCHEMA,
        "interface": EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE,
        "generation_id": generation_id,
        "owner_process_birth": dict(owner_process_birth),
        "phase": phase,
        "owner_committed": phase in {"committed", "stopping", "stopped"},
        "owner_process_alive": phase != "stopped",
        "provider_process_started": False,
        "task_state_mutated": False,
        "owner_start_receipt_cid": owner_start_receipt_cid,
        "final_record_cid": final_record_cid,
        "commit_receipt_cid": commit_receipt_cid,
    }
    value["status_cid"] = _cid(value)
    return value


def _validate_status(raw: Mapping[str, Any], *, generation_id: str) -> dict[str, Any]:
    value = _verified_cid(
        raw,
        fields=_STATUS_FIELDS,
        cid_field="status_cid",
        noun="owner management status",
    )
    birth = _exact_process_birth(
        value.get("owner_process_birth"), noun="status owner process birth"
    )
    phase = value.get("phase")
    cid_fields = (
        value.get("owner_start_receipt_cid"),
        value.get("final_record_cid"),
        value.get("commit_receipt_cid"),
    )
    if (
        value.get("schema") != EAAEF_CASF_OWNER_MANAGEMENT_STATUS_SCHEMA
        or value.get("interface") != EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE
        or value.get("generation_id") != generation_id
        or phase not in {"provisional", "committed", "stopping", "stopped"}
        or type(value.get("owner_committed")) is not bool
        or type(value.get("owner_process_alive")) is not bool
        or value.get("provider_process_started") is not False
        or value.get("task_state_mutated") is not False
        or (phase == "provisional" and any(cid_fields))
        or (
            phase != "provisional"
            and any(type(item) is not str or not _SHA256_RE.fullmatch(item) for item in cid_fields)
        )
        or value["owner_committed"] is not (phase != "provisional")
        or value["owner_process_alive"] is not (phase != "stopped")
        or (phase != "stopped" and not _corroborates_process_birth(birth))
    ):
        raise EAAEFCASFOwnerManagementError(
            "owner management status identity differs"
        )
    return value


def _stop_intent(
    *,
    generation_id: str,
    binding_cid: str,
    snapshot_bindings_cid: str,
    capsule_cid: str,
    owner_process_birth: Mapping[str, Any],
    owner_start_receipt_cid: str,
    final_record_cid: str,
    commit_receipt_cid: str,
    key: bytes,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_OWNER_STOP_INTENT_SCHEMA,
        "interface": EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE,
        "generation_id": generation_id,
        "binding_cid": binding_cid,
        "snapshot_bindings_cid": snapshot_bindings_cid,
        "capsule_cid": capsule_cid,
        "owner_process_birth": dict(owner_process_birth),
        "owner_start_receipt_cid": owner_start_receipt_cid,
        "final_record_cid": final_record_cid,
        "commit_receipt_cid": commit_receipt_cid,
        "stop_request_id": request["arguments"]["stop_request_id"],
        "stop_request_cid": request["request_cid"],
    }
    value["intent_cid"] = _cid(value)
    value["intent_mac"] = _mac(key, value)
    return value


def _validate_stop_intent(
    raw: bytes,
    *,
    generation_id: str,
    binding_cid: str,
    snapshot_bindings_cid: str,
    capsule_cid: str,
    key: bytes,
) -> dict[str, Any]:
    decoded = _decode_object(raw, noun="owner stop intent")
    unsigned = dict(decoded)
    claimed_mac = unsigned.pop("intent_mac", "")
    value = _verified_cid(
        unsigned,
        fields=_STOP_INTENT_FIELDS - {"intent_mac"},
        cid_field="intent_cid",
        noun="owner stop intent",
    )
    _exact_process_birth(
        value.get("owner_process_birth"), noun="stop intent owner process birth"
    )
    if (
        value.get("schema") != EAAEF_CASF_OWNER_STOP_INTENT_SCHEMA
        or value.get("interface") != EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE
        or value.get("generation_id") != generation_id
        or value.get("binding_cid") != binding_cid
        or value.get("snapshot_bindings_cid") != snapshot_bindings_cid
        or value.get("capsule_cid") != capsule_cid
        or any(
            type(value.get(field)) is not str
            or not _SHA256_RE.fullmatch(str(value[field]))
            for field in (
                "owner_start_receipt_cid",
                "final_record_cid",
                "commit_receipt_cid",
            )
        )
        or type(value.get("stop_request_id")) is not str
        or not _HEX_64_RE.fullmatch(str(value["stop_request_id"]))
        or type(value.get("stop_request_cid")) is not str
        or not _SHA256_RE.fullmatch(str(value["stop_request_cid"]))
        or type(claimed_mac) is not str
        or not hmac.compare_digest(claimed_mac, _mac(key, unsigned))
    ):
        raise EAAEFCASFOwnerManagementError("owner stop intent differs")
    return decoded


def _stop_result(
    *,
    generation_id: str,
    owner_process_birth: Mapping[str, Any],
    intent: Mapping[str, Any],
    key: bytes,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": EAAEF_CASF_OWNER_STOP_RESULT_SCHEMA,
        "interface": EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE,
        "generation_id": generation_id,
        "binding_cid": intent["binding_cid"],
        "snapshot_bindings_cid": intent["snapshot_bindings_cid"],
        "capsule_cid": intent["capsule_cid"],
        "owner_process_birth": dict(owner_process_birth),
        "owner_start_receipt_cid": intent["owner_start_receipt_cid"],
        "final_record_cid": intent["final_record_cid"],
        "commit_receipt_cid": intent["commit_receipt_cid"],
        "stop_request_id": intent["stop_request_id"],
        "stop_request_cid": intent["stop_request_cid"],
        "stop_intent_cid": intent["intent_cid"],
        "committed_owner_stopped": True,
        # The state server and its lease are gone before this result is
        # published.  The small broker process must remain alive long enough
        # to transmit it, so callers separately corroborate the exact birth's
        # disappearance before reporting terminal shutdown.
        "broker_process_exit_pending": True,
        "exclusive_owner_lease_released": True,
        "task_state_mutated": False,
    }
    value["result_cid"] = _cid(value)
    value["result_mac"] = _mac(key, value)
    return value


def _validate_stop_result(
    raw: bytes | Mapping[str, Any],
    *,
    generation_id: str,
    binding_cid: str,
    snapshot_bindings_cid: str,
    capsule_cid: str,
    key: bytes,
) -> dict[str, Any]:
    value_raw = (
        dict(raw)
        if isinstance(raw, Mapping)
        else _decode_object(raw, noun="owner stop result")
    )
    unsigned = dict(value_raw)
    claimed_mac = unsigned.pop("result_mac", "")
    value = _verified_cid(
        unsigned,
        fields=_STOP_RESULT_FIELDS - {"result_mac"},
        cid_field="result_cid",
        noun="owner stop result",
    )
    _exact_process_birth(
        value.get("owner_process_birth"), noun="stop result owner process birth"
    )
    if (
        value.get("schema") != EAAEF_CASF_OWNER_STOP_RESULT_SCHEMA
        or value.get("interface") != EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE
        or value.get("generation_id") != generation_id
        or value.get("binding_cid") != binding_cid
        or value.get("snapshot_bindings_cid") != snapshot_bindings_cid
        or value.get("capsule_cid") != capsule_cid
        or any(
            type(value.get(field)) is not str
            or not _SHA256_RE.fullmatch(str(value[field]))
            for field in (
                "owner_start_receipt_cid",
                "final_record_cid",
                "commit_receipt_cid",
                "stop_intent_cid",
            )
        )
        or type(value.get("stop_request_id")) is not str
        or not _HEX_64_RE.fullmatch(str(value["stop_request_id"]))
        or type(value.get("stop_request_cid")) is not str
        or not _SHA256_RE.fullmatch(str(value["stop_request_cid"]))
        or value.get("committed_owner_stopped") is not True
        or value.get("broker_process_exit_pending") is not True
        or value.get("exclusive_owner_lease_released") is not True
        or value.get("task_state_mutated") is not False
        or type(claimed_mac) is not str
        or not hmac.compare_digest(claimed_mac, _mac(key, unsigned))
    ):
        raise EAAEFCASFOwnerManagementError("owner stop result differs")
    return value_raw


class CASFOwnerManagementServer:
    """Broker-owned authenticated status/stop service."""

    def __init__(
        self,
        *,
        generation_id: str,
        binding_cid: str,
        snapshot_bindings_cid: str,
        state_dir: Path,
        owner_process_birth: Mapping[str, Any],
        request_stop: Callable[[], None],
        stop_timeout_seconds: float,
        authentication_timeout_seconds: float = (
            _DEFAULT_AUTHENTICATION_TIMEOUT_SECONDS
        ),
    ) -> None:
        if (
            type(generation_id) is not str
            or not _GENERATION_RE.fullmatch(generation_id)
            or type(binding_cid) is not str
            or not _SHA256_RE.fullmatch(binding_cid)
            or type(snapshot_bindings_cid) is not str
            or not _SHA256_RE.fullmatch(snapshot_bindings_cid)
            or not callable(request_stop)
            or isinstance(stop_timeout_seconds, bool)
            or not isinstance(stop_timeout_seconds, (int, float))
            or not 0.1 <= float(stop_timeout_seconds) <= 600.0
            or isinstance(authentication_timeout_seconds, bool)
            or not isinstance(authentication_timeout_seconds, (int, float))
            or not 0.05 <= float(authentication_timeout_seconds) <= 5.0
        ):
            raise EAAEFCASFOwnerManagementError(
                "owner management server inputs differ"
            )
        self.generation_id = generation_id
        self.binding_cid = binding_cid
        self.snapshot_bindings_cid = snapshot_bindings_cid
        self.state_dir = Path(state_dir)
        self.owner_process_birth = _exact_process_birth(
            owner_process_birth, noun="management server owner process birth"
        )
        if not _corroborates_process_birth(self.owner_process_birth):
            raise EAAEFCASFOwnerManagementError(
                "management server owner process birth is stale"
            )
        self.request_stop = request_stop
        self.stop_timeout_seconds = float(stop_timeout_seconds)
        self.authentication_timeout_seconds = float(
            authentication_timeout_seconds
        )
        self.key = b""
        self.capsule: dict[str, Any] = {}
        self.listener: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.closed = threading.Event()
        self.stop_requested = threading.Event()
        self.stopped = threading.Event()
        self.stop_response_sent = threading.Event()
        self._gate = threading.Lock()
        self._phase = "provisional"
        self._owner_start_receipt_cid = ""
        self._final_record_cid = ""
        self._commit_receipt_cid = ""
        self._intent: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None
        self._status_request_nonces: OrderedDict[str, None] = OrderedDict()
        self._stop_request_nonce: str | None = None
        self._connection_slots = threading.BoundedSemaphore(
            _MAX_CONNECTION_WORKERS
        )
        self._connection_gate = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._connection_threads: set[threading.Thread] = set()

    def start(self) -> None:
        if self.listener is not None:
            raise EAAEFCASFOwnerManagementError(
                "owner management server is one-shot"
            )
        if not hasattr(socket, "SO_PEERCRED"):
            raise EAAEFCASFOwnerManagementError(
                "Linux owner management peer credentials are unavailable"
            )
        self.key = secrets.token_bytes(32)
        endpoint_nonce = secrets.token_hex(32)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind("\0ipfs-accelerate-eaaef-casf-" + endpoint_nonce)
            listener.listen(_MAX_CONNECTION_WORKERS)
            listener.settimeout(0.2)
            capsule = _build_capsule(
                generation_id=self.generation_id,
                binding_cid=self.binding_cid,
                snapshot_bindings_cid=self.snapshot_bindings_cid,
                endpoint_nonce=endpoint_nonce,
                owner_process_birth=self.owner_process_birth,
                key=self.key,
            )
            _write_once(self.state_dir, MANAGEMENT_KEY_NAME, self.key)
            _write_once(
                self.state_dir,
                MANAGEMENT_CAPSULE_NAME,
                _canonical_bytes(capsule, noun="owner management capsule"),
            )
        except BaseException:
            listener.close()
            # A partial key/capsule reservation quarantines this generation.
            # Do not retain usable secret material or a listener in memory;
            # the caller must bootstrap a fresh generation.
            self.key = b""
            self.capsule = {}
            raise
        self.capsule = capsule
        self.listener = listener
        self.thread = threading.Thread(
            target=self._serve,
            name="eaaef-casf-owner-management",
            daemon=True,
        )
        try:
            self.thread.start()
        except BaseException:
            listener.close()
            self.listener = None
            self.thread = None
            self.key = b""
            self.capsule = {}
            raise

    def mark_committed(
        self,
        *,
        owner_start_receipt_cid: str,
        final_record_cid: str,
        commit_receipt_cid: str,
    ) -> None:
        selected = (
            owner_start_receipt_cid,
            final_record_cid,
            commit_receipt_cid,
        )
        if any(type(item) is not str or not _SHA256_RE.fullmatch(item) for item in selected):
            raise EAAEFCASFOwnerManagementError(
                "owner management commit identities differ"
            )
        with self._gate:
            if self._phase != "provisional":
                raise EAAEFCASFOwnerManagementError(
                    "owner management commit phase differs"
                )
            self._owner_start_receipt_cid = owner_start_receipt_cid
            self._final_record_cid = final_record_cid
            self._commit_receipt_cid = commit_receipt_cid
            self._phase = "committed"

    def _snapshot(self) -> dict[str, Any]:
        with self._gate:
            return _status_snapshot(
                generation_id=self.generation_id,
                owner_process_birth=self.owner_process_birth,
                phase=self._phase,
                owner_start_receipt_cid=self._owner_start_receipt_cid,
                final_record_cid=self._final_record_cid,
                commit_receipt_cid=self._commit_receipt_cid,
            )

    def _peer_pid(self, connection: socket.socket) -> int:
        try:
            raw = connection.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                _PEER_CREDENTIALS.size,
            )
            peer_pid, peer_uid, peer_gid = _PEER_CREDENTIALS.unpack(raw)
        except (OSError, struct.error) as exc:
            raise EAAEFCASFOwnerManagementError(
                "owner management peer credentials are unavailable"
            ) from exc
        if (
            peer_pid < 1
            or peer_uid != os.geteuid()
            or peer_gid != os.getegid()
        ):
            raise EAAEFCASFOwnerManagementError(
                "owner management peer credentials differ"
            )
        return peer_pid

    def _handle_stop(self, request: Mapping[str, Any]) -> dict[str, Any]:
        with self._gate:
            if self._phase == "provisional":
                raise EAAEFCASFOwnerManagementError("owner is not committed")
            requested = _stop_intent(
                generation_id=self.generation_id,
                binding_cid=self.binding_cid,
                snapshot_bindings_cid=self.snapshot_bindings_cid,
                capsule_cid=str(self.capsule["capsule_cid"]),
                owner_process_birth=self.owner_process_birth,
                owner_start_receipt_cid=self._owner_start_receipt_cid,
                final_record_cid=self._final_record_cid,
                commit_receipt_cid=self._commit_receipt_cid,
                key=self.key,
                request=request,
            )
            if self._intent is None:
                _write_once(
                    self.state_dir,
                    MANAGEMENT_STOP_INTENT_NAME,
                    _canonical_bytes(requested, noun="owner stop intent"),
                )
                self._intent = requested
                self._phase = "stopping"
                self.stop_requested.set()
                try:
                    self.request_stop()
                except BaseException as exc:
                    raise EAAEFCASFOwnerManagementError(
                        "owner stop callback failed"
                    ) from exc
            elif self._intent != requested:
                raise EAAEFCASFOwnerManagementError(
                    "owner stop request diverged"
                )
        stop_deadline = time.monotonic() + self.stop_timeout_seconds
        while not self.stopped.is_set():
            remaining = stop_deadline - time.monotonic()
            if remaining <= 0 or self.closed.is_set():
                raise EAAEFCASFOwnerManagementError(
                    "owner stop result is unavailable"
                )
            self.stopped.wait(min(remaining, 0.05))
        with self._gate:
            if self._result is None:
                raise EAAEFCASFOwnerManagementError(
                    "owner stop result is unavailable"
                )
            return dict(self._result)

    def _admit_request_nonce(self, request: Mapping[str, Any]) -> None:
        request_nonce = str(request["request_nonce"])
        operation = str(request["operation"])
        with self._gate:
            if operation == "stop" and self._phase != "provisional":
                arguments = request["arguments"]
                if (
                    arguments["owner_start_receipt_cid"]
                    != self._owner_start_receipt_cid
                    or arguments["final_record_cid"] != self._final_record_cid
                    or arguments["commit_receipt_cid"]
                    != self._commit_receipt_cid
                ):
                    raise EAAEFCASFOwnerManagementError(
                        "owner management stop commit binding diverged"
                    )
                if request_nonce in self._status_request_nonces:
                    raise EAAEFCASFOwnerManagementError(
                        "owner management request nonce was replayed"
                    )
                if self._stop_request_nonce is None:
                    self._stop_request_nonce = request_nonce
                    return
                if self._stop_request_nonce == request_nonce:
                    raise EAAEFCASFOwnerManagementError(
                        "owner management request nonce was replayed"
                    )
                raise EAAEFCASFOwnerManagementError(
                    "owner management stop request diverged"
                )
            if request_nonce in self._status_request_nonces:
                raise EAAEFCASFOwnerManagementError(
                    "owner management request nonce was replayed"
                )
            if len(self._status_request_nonces) >= _MAX_STATUS_REQUEST_NONCES:
                self._status_request_nonces.popitem(last=False)
            self._status_request_nonces[request_nonce] = None

    def _serve_connection(self, connection: socket.socket) -> None:
        peer_pid = self._peer_pid(connection)
        authentication_deadline = (
            time.monotonic() + self.authentication_timeout_seconds
        )
        request_raw = _recv_packet_before(
            connection,
            deadline=authentication_deadline,
        )
        request = _validate_request(
            request_raw,
            generation_id=self.generation_id,
            capsule_cid=self.capsule["capsule_cid"],
            key=self.key,
            peer_pid=peer_pid,
        )
        if time.monotonic() > authentication_deadline:
            raise EAAEFCASFOwnerManagementError(
                "owner management authentication deadline expired"
            )
        # The long timeout is available only after exact frame, HMAC, peer,
        # and process-birth authentication.  It covers committed-owner stop
        # completion and the final bounded response, never authentication.
        connection.settimeout(self.stop_timeout_seconds)
        operation = str(request["operation"])
        ok = True
        error_code = ""
        result: dict[str, Any] = {}
        try:
            self._admit_request_nonce(request)
            if operation == "status.snapshot":
                result = self._snapshot()
            else:
                result = self._handle_stop(request)
        except EAAEFCASFOwnerManagementError as exc:
            ok = False
            message = str(exc)
            if "not committed" in message:
                error_code = "owner_not_committed"
            elif "replayed" in message:
                error_code = "request_nonce_replayed"
            elif "capacity" in message:
                error_code = "request_nonce_capacity_exhausted"
            elif "diverged" in message:
                error_code = "stop_request_diverged"
            elif "unavailable" in message:
                error_code = "stop_result_unavailable"
            else:
                error_code = "stop_request_failed"
        response = _build_response(
            generation_id=self.generation_id,
            capsule_cid=self.capsule["capsule_cid"],
            key=self.key,
            request_cid=request["request_cid"],
            operation=operation,
            ok=ok,
            result=result,
            error_code=error_code,
        )
        _send_packet(
            connection,
            _canonical_bytes(response, noun="owner management response"),
        )
        if operation == "stop" and ok:
            self.stop_response_sent.set()

    def _serve_worker(self, connection: socket.socket) -> None:
        try:
            self._serve_connection(connection)
        except BaseException:
            # Unauthenticated or malformed requests receive no oracle.
            pass
        finally:
            try:
                connection.close()
            except OSError:
                pass
            with self._connection_gate:
                self._connections.discard(connection)
                self._connection_threads.discard(threading.current_thread())
            self._connection_slots.release()

    def _serve(self) -> None:
        listener = self.listener
        if listener is None:
            return
        while not self.closed.is_set():
            try:
                connection, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            if not self._connection_slots.acquire(blocking=False):
                try:
                    connection.close()
                except OSError:
                    pass
                continue
            worker = threading.Thread(
                target=self._serve_worker,
                args=(connection,),
                name="eaaef-casf-owner-management-connection",
                daemon=True,
            )
            with self._connection_gate:
                if self.closed.is_set():
                    self._connection_slots.release()
                    connection.close()
                    break
                self._connections.add(connection)
                self._connection_threads.add(worker)
            try:
                worker.start()
            except BaseException:
                with self._connection_gate:
                    self._connections.discard(connection)
                    self._connection_threads.discard(worker)
                self._connection_slots.release()
                connection.close()
                raise

    def mark_stopped(self) -> Mapping[str, Any]:
        with self._gate:
            if self._phase != "stopping" or self._intent is None:
                raise EAAEFCASFOwnerManagementError(
                    "owner management stop phase differs"
                )
            result = _stop_result(
                generation_id=self.generation_id,
                owner_process_birth=self.owner_process_birth,
                intent=self._intent,
                key=self.key,
            )
            _write_once(
                self.state_dir,
                MANAGEMENT_STOP_RESULT_NAME,
                _canonical_bytes(result, noun="owner stop result"),
            )
            self._result = result
            self._phase = "stopped"
            self.stopped.set()
            return dict(result)

    def close(self) -> None:
        self.closed.set()
        listener = self.listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(self.stop_timeout_seconds, 2.0))
        with self._connection_gate:
            connections = tuple(self._connections)
            workers = tuple(self._connection_threads)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(
                    timeout=min(self.authentication_timeout_seconds + 0.5, 2.0)
                )


def _load_generation_identity(
    *,
    generation_id: str,
    binding_cid: str,
    snapshot_bindings_cid: str,
    state_dir: Path,
    require_live: bool,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    if (
        type(generation_id) is not str
        or not _GENERATION_RE.fullmatch(generation_id)
        or type(binding_cid) is not str
        or not _SHA256_RE.fullmatch(binding_cid)
        or type(snapshot_bindings_cid) is not str
        or not _SHA256_RE.fullmatch(snapshot_bindings_cid)
        or type(require_live) is not bool
    ):
        raise EAAEFCASFOwnerManagementError(
            "owner management generation identity differs"
        )
    key = _read_private(Path(state_dir), MANAGEMENT_KEY_NAME)
    if len(key) != 32:
        raise EAAEFCASFOwnerManagementError(
            "owner management key length differs"
        )
    capsule = _validate_capsule(
        _read_private(Path(state_dir), MANAGEMENT_CAPSULE_NAME),
        generation_id=generation_id,
        binding_cid=binding_cid,
        snapshot_bindings_cid=snapshot_bindings_cid,
        key=key,
        require_live=require_live,
    )
    owner_process_birth = _exact_process_birth(
        capsule["owner_process_birth"],
        noun="management capsule owner process birth",
    )
    return key, capsule, owner_process_birth


def _validated_stop_artifacts(
    *,
    generation_id: str,
    binding_cid: str,
    snapshot_bindings_cid: str,
    state_dir: Path,
    capsule: Mapping[str, Any],
    key: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    capsule_cid = str(capsule["capsule_cid"])
    intent = _validate_stop_intent(
        _read_private(Path(state_dir), MANAGEMENT_STOP_INTENT_NAME),
        generation_id=generation_id,
        binding_cid=binding_cid,
        snapshot_bindings_cid=snapshot_bindings_cid,
        capsule_cid=capsule_cid,
        key=key,
    )
    result = _validate_stop_result(
        _read_private(Path(state_dir), MANAGEMENT_STOP_RESULT_NAME),
        generation_id=generation_id,
        binding_cid=binding_cid,
        snapshot_bindings_cid=snapshot_bindings_cid,
        capsule_cid=capsule_cid,
        key=key,
    )
    linked_fields = (
        "binding_cid",
        "snapshot_bindings_cid",
        "capsule_cid",
        "owner_process_birth",
        "owner_start_receipt_cid",
        "final_record_cid",
        "commit_receipt_cid",
        "stop_request_id",
        "stop_request_cid",
    )
    if (
        any(result[field] != intent[field] for field in linked_fields)
        or intent["owner_process_birth"] != capsule["owner_process_birth"]
        or result["stop_intent_cid"] != intent["intent_cid"]
    ):
        raise EAAEFCASFOwnerManagementError(
            "owner stop intent and result binding differs"
        )
    return intent, result


class CASFOwnerManagementClient:
    """Reattachable client holding no database or provider authority."""

    def __init__(
        self,
        *,
        generation_id: str,
        binding_cid: str,
        snapshot_bindings_cid: str,
        state_dir: Path,
        timeout_seconds: float,
    ) -> None:
        if (
            type(generation_id) is not str
            or not _GENERATION_RE.fullmatch(generation_id)
            or type(binding_cid) is not str
            or not _SHA256_RE.fullmatch(binding_cid)
            or type(snapshot_bindings_cid) is not str
            or not _SHA256_RE.fullmatch(snapshot_bindings_cid)
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0.1 <= float(timeout_seconds) <= 600.0
        ):
            raise EAAEFCASFOwnerManagementError(
                "owner management client inputs differ"
            )
        self.generation_id = generation_id
        self.binding_cid = binding_cid
        self.snapshot_bindings_cid = snapshot_bindings_cid
        self.state_dir = Path(state_dir)
        self.timeout_seconds = float(timeout_seconds)
        self.key, self.capsule, self.owner_process_birth = _load_generation_identity(
            generation_id=generation_id,
            binding_cid=binding_cid,
            snapshot_bindings_cid=snapshot_bindings_cid,
            state_dir=self.state_dir,
            require_live=True,
        )

    @classmethod
    def adopt_completed_stop(
        cls,
        *,
        generation_id: str,
        binding_cid: str,
        snapshot_bindings_cid: str,
        state_dir: Path,
    ) -> Mapping[str, Any]:
        """Verify terminal stop proof without reviving a sealed generation.

        This path intentionally cannot connect, issue a new request, or reuse
        owner authority.  It accepts only the write-once, MAC-bound intent and
        result after the exact capsule-bound owner birth has disappeared.
        """

        key, capsule, owner_process_birth = _load_generation_identity(
            generation_id=generation_id,
            binding_cid=binding_cid,
            snapshot_bindings_cid=snapshot_bindings_cid,
            state_dir=Path(state_dir),
            require_live=False,
        )
        current_birth = _current_process_birth()
        if owner_process_birth["boot_id"] != current_birth["boot_id"]:
            raise EAAEFCASFOwnerManagementError(
                "owner management terminal proof boot differs"
            )
        observed_owner = _observed_process_birth(
            int(owner_process_birth["pid"])
        )
        if observed_owner is not None and all(
            observed_owner[field] == owner_process_birth[field]
            for field in _IMMUTABLE_PROCESS_BIRTH_FIELDS
        ):
            raise EAAEFCASFOwnerManagementError(
                "owner management terminal proof owner remains live"
            )
        if observed_owner is not None:
            raise EAAEFCASFOwnerManagementError(
                "owner management terminal proof PID was reused"
            )
        _intent, result = _validated_stop_artifacts(
            generation_id=generation_id,
            binding_cid=binding_cid,
            snapshot_bindings_cid=snapshot_bindings_cid,
            state_dir=Path(state_dir),
            capsule=capsule,
            key=key,
        )
        return result

    def _connect(self) -> socket.socket:
        if not _corroborates_process_birth(self.owner_process_birth):
            raise EAAEFCASFOwnerManagementError(
                "owner management process birth is stale"
            )
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            channel.settimeout(self.timeout_seconds)
            channel.connect(_endpoint(self.capsule))
            raw = channel.getsockopt(
                socket.SOL_SOCKET,
                socket.SO_PEERCRED,
                _PEER_CREDENTIALS.size,
            )
            peer_pid, peer_uid, peer_gid = _PEER_CREDENTIALS.unpack(raw)
            if (
                peer_pid != int(self.owner_process_birth["pid"])
                or peer_uid != os.geteuid()
                or peer_gid != os.getegid()
                or not _corroborates_process_birth(self.owner_process_birth)
            ):
                raise EAAEFCASFOwnerManagementError(
                    "owner management endpoint process birth differs"
                )
            return channel
        except BaseException:
            channel.close()
            raise

    def _round_trip(self, request: Mapping[str, Any]) -> dict[str, Any]:
        channel = self._connect()
        try:
            _send_packet(
                channel,
                _canonical_bytes(request, noun="owner management request"),
            )
            return _validate_response(
                _recv_packet(channel),
                generation_id=self.generation_id,
                capsule_cid=self.capsule["capsule_cid"],
                key=self.key,
                request_cid=str(request["request_cid"]),
                operation=str(request["operation"]),
            )
        except (OSError, TimeoutError) as exc:
            raise EAAEFCASFOwnerManagementError(
                "owner management response is unavailable"
            ) from exc
        finally:
            channel.close()

    def status_snapshot(self) -> Mapping[str, Any]:
        request = _build_request(
            generation_id=self.generation_id,
            capsule_cid=self.capsule["capsule_cid"],
            key=self.key,
            operation="status.snapshot",
            arguments={},
        )
        response = self._round_trip(request)
        if response["ok"] is not True:
            raise EAAEFCASFOwnerManagementError(
                "owner management status was rejected: "
                + str(response["error_code"])
            )
        status = _validate_status(
            response["result"], generation_id=self.generation_id
        )
        if status["owner_process_birth"] != self.owner_process_birth:
            raise EAAEFCASFOwnerManagementError(
                "owner management status process birth differs"
            )
        return status

    def _adopt_stop_result(
        self,
        request: Mapping[str, Any],
        *,
        deadline: float,
    ) -> dict[str, Any]:
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                _intent, result = _validated_stop_artifacts(
                    generation_id=self.generation_id,
                    binding_cid=self.binding_cid,
                    snapshot_bindings_cid=self.snapshot_bindings_cid,
                    state_dir=self.state_dir,
                    capsule=self.capsule,
                    key=self.key,
                )
                if (
                    result["stop_request_id"]
                    != request["arguments"]["stop_request_id"]
                    or result["stop_request_cid"] != request["request_cid"]
                    or result["owner_process_birth"] != self.owner_process_birth
                    or any(
                        result[field] != request["arguments"][field]
                        for field in (
                            "owner_start_receipt_cid",
                            "final_record_cid",
                            "commit_receipt_cid",
                        )
                    )
                ):
                    raise EAAEFCASFOwnerManagementError(
                        "owner stop result does not adopt the exact request"
                    )
                return result
            except EAAEFCASFOwnerManagementError as exc:
                last_error = exc
                time.sleep(0.01)
        raise EAAEFCASFOwnerManagementError(
            "owner stop response and durable result are unavailable"
        ) from last_error

    def stop(self) -> Mapping[str, Any]:
        committed_status = self.status_snapshot()
        if committed_status["phase"] != "committed":
            raise EAAEFCASFOwnerManagementError(
                "owner management stop was rejected: owner_not_committed"
            )
        request = _build_request(
            generation_id=self.generation_id,
            capsule_cid=self.capsule["capsule_cid"],
            key=self.key,
            operation="stop",
            arguments={
                "stop_request_id": secrets.token_hex(32),
                "owner_start_receipt_cid": committed_status[
                    "owner_start_receipt_cid"
                ],
                "final_record_cid": committed_status["final_record_cid"],
                "commit_receipt_cid": committed_status["commit_receipt_cid"],
            },
        )
        try:
            response = self._round_trip(request)
        except EAAEFCASFOwnerManagementError:
            return self._adopt_stop_result(
                request,
                deadline=time.monotonic() + self.timeout_seconds,
            )
        if response["ok"] is not True:
            if response["error_code"] == "stop_result_unavailable":
                return self._adopt_stop_result(
                    request,
                    deadline=time.monotonic() + self.timeout_seconds,
                )
            raise EAAEFCASFOwnerManagementError(
                "owner management stop was rejected: "
                + str(response["error_code"])
            )
        result = _validate_stop_result(
            response["result"],
            generation_id=self.generation_id,
            binding_cid=self.binding_cid,
            snapshot_bindings_cid=self.snapshot_bindings_cid,
            capsule_cid=str(self.capsule["capsule_cid"]),
            key=self.key,
        )
        _intent, retained = _validated_stop_artifacts(
            generation_id=self.generation_id,
            binding_cid=self.binding_cid,
            snapshot_bindings_cid=self.snapshot_bindings_cid,
            state_dir=self.state_dir,
            capsule=self.capsule,
            key=self.key,
        )
        if (
            result != retained
            or result["stop_request_id"] != request["arguments"]["stop_request_id"]
            or result["stop_request_cid"] != request["request_cid"]
            or result["owner_process_birth"] != self.owner_process_birth
            or any(
                result[field] != request["arguments"][field]
                for field in (
                    "owner_start_receipt_cid",
                    "final_record_cid",
                    "commit_receipt_cid",
                )
            )
        ):
            raise EAAEFCASFOwnerManagementError(
                "owner management stop result differs"
            )
        return result

    def wait_dead(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _corroborates_process_birth(self.owner_process_birth):
                return True
            time.sleep(0.01)
        return False

    def is_alive(self) -> bool:
        """Corroborate the exact capsule-bound owner birth without connecting."""

        return _corroborates_process_birth(self.owner_process_birth)


__all__ = (
    "CASFOwnerManagementClient",
    "CASFOwnerManagementServer",
    "EAAEFCASFOwnerManagementError",
    "EAAEF_CASF_OWNER_MANAGEMENT_CAPSULE_SCHEMA",
    "EAAEF_CASF_OWNER_MANAGEMENT_INTERFACE",
    "EAAEF_CASF_OWNER_MANAGEMENT_REQUEST_SCHEMA",
    "EAAEF_CASF_OWNER_MANAGEMENT_RESPONSE_SCHEMA",
    "EAAEF_CASF_OWNER_MANAGEMENT_STATUS_SCHEMA",
    "EAAEF_CASF_OWNER_STOP_INTENT_SCHEMA",
    "EAAEF_CASF_OWNER_STOP_RESULT_SCHEMA",
    "MANAGEMENT_CAPSULE_NAME",
    "MANAGEMENT_KEY_NAME",
    "MANAGEMENT_STOP_INTENT_NAME",
    "MANAGEMENT_STOP_RESULT_NAME",
)
