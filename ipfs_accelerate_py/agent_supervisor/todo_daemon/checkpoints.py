"""DaemonCheckpoint@1 and typed stale-stop lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

CHECKPOINT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/daemon-checkpoint@1"
MAX_CHECKPOINT_BYTES = 1_048_576
_REQUIRED = (
    "attempt_id",
    "packet_cid",
    "tree_cid",
    "fence_epoch",
    "effects",
    "obligations",
)
_STALE_BINDINGS = {
    "attempt_id": "stale-scope",
    "packet_cid": "stale-plan",
    "tree_cid": "stale-root",
    "fence_epoch": "stale-fence",
}

TRANSITIONS = {
    "ready": ("running", "cancelled"),
    "running": ("checkpointed", "stale-stop", "completed", "cancelled"),
    "checkpointed": ("running", "stale-stop"),
    "stale-stop": (),
    "completed": (),
    "cancelled": (),
}
STALE_REASONS = (
    "stale-plan",
    "stale-root",
    "stale-lease",
    "stale-fence",
    "stale-state-owner-epoch",
    "stale-scope",
    "cancel",
    "already-accepted",
)


class CheckpointError(ValueError):
    """Lifecycle or checkpoint rejected."""


def transition(state: str, action: str) -> str:
    actions = {
        "start": "running",
        "checkpoint": "checkpointed",
        "resume": "running",
        "stale-stop": "stale-stop",
        "complete": "completed",
        "cancel": "cancelled",
    }
    if (
        not isinstance(state, str)
        or not isinstance(action, str)
        or state not in TRANSITIONS
        or action not in actions
    ):
        raise CheckpointError("unknown checkpoint state or action")
    nxt = actions[action]
    if nxt not in TRANSITIONS[state]:
        raise CheckpointError(f"illegal {state}->{nxt}")
    return nxt


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise CheckpointError("checkpoint must contain finite JSON values") from exc


def _validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise CheckpointError("checkpoint record must be an object")
    missing = [name for name in _REQUIRED if name not in record]
    if missing:
        raise CheckpointError(f"checkpoint missing {missing}")
    if record.get("as_completion"):
        raise CheckpointError("checkpoint cannot be completion")
    for name in ("attempt_id", "packet_cid", "tree_cid"):
        value = record[name]
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise CheckpointError(f"{name} must be non-empty text")
    epoch = record["fence_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CheckpointError("fence_epoch must be a non-negative integer")
    for name in ("effects", "obligations"):
        if not isinstance(record[name], (list, tuple)):
            raise CheckpointError(f"{name} must be a sequence")
    reason = record.get("stale_reason")
    if reason not in (None, "") and reason not in STALE_REASONS:
        raise CheckpointError("unknown stale reason")
    if record.get("corrupt"):
        raise CheckpointError("corrupt checkpoint")
    # JSON normalization makes tuple-based legacy mappings round-trip exactly
    # through the versioned envelope without relying on Python repr/eval.
    return json.loads(_canonical_bytes(dict(record)))


def write_checkpoint(record: Mapping[str, Any], path: str | Path) -> Mapping[str, Any]:
    payload = _validate_record(record)
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    encoded = (
        _canonical_bytes(
            {"schema": CHECKPOINT_SCHEMA, "record": payload, "record_sha256": digest}
        )
        + b"\n"
    )
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise CheckpointError("checkpoint exceeds persistence bound")
    target = Path(path)
    temporary: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise CheckpointError("checkpoint path must not be a symlink")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise CheckpointError("checkpoint persistence failed") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return MappingProxyType({"ok": True, "path": str(target), "record_sha256": digest})


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError("duplicate checkpoint field")
        result[key] = value
    return result


def read_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Read a bounded, versioned checkpoint and verify its saved record hash."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise CheckpointError("checkpoint must be a regular file")
            encoded = stream.read(MAX_CHECKPOINT_BYTES + 1)
        if len(encoded) > MAX_CHECKPOINT_BYTES:
            raise CheckpointError("checkpoint exceeds persistence bound")
        envelope = json.loads(encoded, object_pairs_hook=_unique_object)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError("corrupt or unreadable checkpoint") from exc
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"schema", "record", "record_sha256"}
        or envelope.get("schema") != CHECKPOINT_SCHEMA
    ):
        raise CheckpointError("unsupported checkpoint envelope")
    payload = _validate_record(envelope["record"])
    if (
        envelope["record_sha256"]
        != hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    ):
        raise CheckpointError("corrupt checkpoint digest")
    return MappingProxyType(payload)


def resume_checkpoint(
    record: Mapping[str, Any] | str | Path,
    *,
    expected_bindings: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if isinstance(record, (str, Path)):
        record = read_checkpoint(record)
    if not isinstance(record, Mapping):
        raise CheckpointError("checkpoint record must be an object")
    if record.get("corrupt"):
        raise CheckpointError("corrupt checkpoint")
    reason = record.get("stale_reason")
    if reason not in (None, ""):
        if reason not in STALE_REASONS:
            raise CheckpointError("unknown stale reason")
        return MappingProxyType({"resumed": False, "stopped": True, "reason": reason})
    payload = _validate_record(record)
    if expected_bindings is not None and not isinstance(expected_bindings, Mapping):
        raise CheckpointError("expected_bindings must be an object")
    for name, expected in (expected_bindings or {}).items():
        if name not in _STALE_BINDINGS:
            raise CheckpointError(f"unknown checkpoint binding {name}")
        if type(payload[name]) is not type(expected) or payload[name] != expected:
            return MappingProxyType(
                {"resumed": False, "stopped": True, "reason": _STALE_BINDINGS[name]}
            )
    return MappingProxyType({"resumed": True, "stopped": False, "reason": ""})


def stale_stop(reason: str) -> Mapping[str, Any]:
    if reason not in STALE_REASONS:
        raise CheckpointError(f"unknown stale reason {reason}")
    return MappingProxyType({"stopped": True, "effect_after": False, "reason": reason})
