"""Authenticate retained checkpoint bytes without interpreting a decoder.

Exact persisted catalog nominations and streamed original-file observations
are separate authorities. A byte match does not verify tensor shape, ABI,
encoder/token/span/inventory contracts, quality, or runtime readiness. Native
catalog records and their false authority fields are returned unchanged.

This uses cooperative catalog and file endpoint checks, not an atomic snapshot
of a live catalog and filesystem. Consumers must repeat authentication when
using an artifact; retaining this observation does not lease a checkpoint.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import stat

from .task_ir_selection import resolve_task_ir_selections

SCHEMA = "task-ir-checkpoint-authentication/v1"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_TOTAL_CHECKPOINT_BYTES = 1024 * 1024 * 1024
MAX_SELECTIONS = 16
READ_CHUNK_BYTES = 1024 * 1024
_SELECTOR_FIELDS = (
    "record_id", "ir_family_id", "dimension", "dimension_role", "schema_version",
    "task_id", "profile_id", "format_id", "checkpoint_sha256", "role",
)
_STAT_FIELDS = (
    "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns",
)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError("checkpoint requires an existing regular single-link file")
    return tuple(getattr(value, name) for name in _STAT_FIELDS)


def _path_identity(path: Path) -> tuple[int, ...]:
    try:
        if not path.is_absolute() or ".." in path.parts or path.resolve(strict=True) != path:
            raise ValueError("checkpoint requires an exact canonical absolute path without symlinks")
        return _identity(path.lstat())
    except (OSError, RuntimeError) as error:
        raise ValueError("checkpoint file is unavailable or unsafe") from error


def _pin_plan(resolution: dict) -> tuple[Path, dict, tuple[int, ...]]:
    pin = resolution["selected_binding"]["declaration"]["original_checkpoint_pin"]
    if (type(pin) is not dict or set(pin) != {"path", "bytes", "sha256"}
            or type(pin["path"]) is not str or type(pin["bytes"]) is not int
            or not 0 < pin["bytes"] <= MAX_CHECKPOINT_BYTES
            or pin["sha256"] != resolution["selected_binding"]["checkpoint_sha256"]):
        raise ValueError("checkpoint requires a closed, typed pin within the per-file byte bound")
    path = Path(pin["path"])
    if str(path) != pin["path"]:
        raise ValueError("checkpoint requires an exact canonical absolute path without symlinks")
    identity = _path_identity(path)
    if identity[4] != pin["bytes"]:
        raise ValueError("checkpoint file size differs from the declared byte pin")
    return path, deepcopy(pin), identity


def _check_file(path: Path, descriptor: int, expected: tuple[int, ...]) -> None:
    try:
        observed = _identity(os.fstat(descriptor))
    except OSError as error:
        raise ValueError("checkpoint descriptor became unavailable") from error
    if observed != expected or _path_identity(path) != expected:
        raise ValueError("checkpoint file identity changed during authentication")


def _read_pin(descriptor: int, pin: dict) -> None:
    digest = hashlib.sha256()
    size = 0
    while True:
        # One extra byte is sufficient to reject growth; this never buffers a
        # checkpoint or follows an unbounded stream to discover its length.
        block = os.read(descriptor, min(READ_CHUNK_BYTES, pin["bytes"] - size + 1))
        if not block:
            break
        size += len(block)
        if size > pin["bytes"]:
            raise ValueError("checkpoint grew beyond its declared byte pin")
        digest.update(block)
    if size != pin["bytes"] or digest.hexdigest() != pin["sha256"]:
        raise ValueError("checkpoint bytes differ from the declared SHA256 or size pin")


def authenticate_task_ir_checkpoints(*, catalog_path: Path, selections: list[dict]) -> list[dict]:
    """Observe 1–16 exact nominations and authenticate their original file bytes.

    All selected paths, file types, declared sizes, and the conservative 1 GiB
    aggregate budget are checked before opening any checkpoint. Read-only
    descriptors stay open through the closing catalog and all-file checks.
    No checkpoint JSON/tensor parsing, model imports, store writes, providers,
    training, inference, or runtime admission occurs.
    """
    if type(selections) is not list or not 1 <= len(selections) <= MAX_SELECTIONS:
        raise ValueError("checkpoint authentication requires 1 to 16 exact catalog selectors")
    resolutions = resolve_task_ir_selections(catalog_path=catalog_path, selections=selections)
    requests = [{name: resolution["selected_binding"][name] for name in _SELECTOR_FIELDS}
                for resolution in resolutions]
    plans = [_pin_plan(resolution) for resolution in resolutions]
    if sum(pin["bytes"] for _, pin, _ in plans) > MAX_TOTAL_CHECKPOINT_BYTES:
        raise ValueError("checkpoint selections exceed the aggregate byte bound")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    opened = []
    with ExitStack() as stack:
        for path, pin, expected in plans:
            if _path_identity(path) != expected:
                raise ValueError("checkpoint file identity changed before opening")
            try:
                descriptor = os.open(path, flags)
            except OSError as error:
                raise ValueError("checkpoint file cannot be opened safely for read-only authentication") from error
            stack.callback(os.close, descriptor)
            opened.append((path, pin, expected, descriptor))
            _check_file(path, descriptor, expected)
            try:
                _read_pin(descriptor, pin)
            except OSError as error:
                raise ValueError("checkpoint file could not be read") from error
            _check_file(path, descriptor, expected)
        closing = resolve_task_ir_selections(catalog_path=catalog_path, selections=requests)
        if closing != resolutions:
            raise ValueError("task IR catalog generation changed during checkpoint authentication")
        for path, _, expected, descriptor in opened:
            _check_file(path, descriptor, expected)
        return [
            {
                "schema": SCHEMA,
                "selectors": request,
                "native_resolution": resolution,
                "original_checkpoint_pin": pin,
                "file_witness": {
                    "path": str(path), "bytes": pin["bytes"], "sha256": pin["sha256"],
                    "device": expected[0], "inode": expected[1], "mode": expected[2],
                    "nlink": expected[3], "mtime_ns": expected[5], "ctime_ns": expected[6],
                },
                "checkpoint_bytes_authenticated": True,
                "observation_scope": "cooperative-endpoint-checks",
                "authority": {
                    "model_manager_constructed": False, "model_loaded": False,
                    "inference_executed": False, "training_executed": False,
                    "runtime_admitted": False, "teacher_qualified": False,
                    "proof_authority": False,
                },
            }
            for request, resolution, (path, pin, expected, _) in zip(requests, resolutions, opened)
        ]
