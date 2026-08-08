"""Durable once-only fallback reservations.

This module deliberately has no provider classification or retry behaviour.
It only atomically reserves (or adopts) a router-authorized logical attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import fcntl

CAS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/provider-attempt-cas@1"

class ProviderAttemptStoreError(ValueError): pass

def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()

def _token(value: object, name: str) -> str:
    value = str(value or "").strip()
    if not value or "\x00" in value or "\n" in value or "\r" in value: raise ProviderAttemptStoreError(f"{name} must be nonempty")
    return value

def _owned_regular(path: Path) -> None:
    try: metadata = path.lstat()
    except OSError as exc: raise ProviderAttemptStoreError("attempt reservation is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ProviderAttemptStoreError("attempt reservation is not an owned private regular file")

@dataclass(frozen=True)
class ProviderAttemptReservation:
    logical_attempt_id: str; route_id: str; decision_id: str; task_id: str; worktree_id: str; reservation_id: str; state: str; created_at_ms: int
    @property
    def content_id(self) -> str: return "sha256:" + hashlib.sha256(_canonical(asdict(self))).hexdigest()

@dataclass(frozen=True)
class ProviderAttemptCASResult:
    reservation: ProviderAttemptReservation; created: bool; adopted: bool

class DurableProviderAttemptCAS:
    """File-lock backed compare-and-swap for one fallback per logical attempt."""
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)

    def _path(self, logical_attempt_id: str) -> Path:
        safe = hashlib.sha256(_token(logical_attempt_id, "logical_attempt_id").encode()).hexdigest()
        return self.directory / (safe + ".json")

    def reserve_or_adopt(self, *, logical_attempt_id: str, route_id: str, decision_id: str, task_id: str, worktree_id: str, authorized: bool, now_ms: int | None = None) -> ProviderAttemptCASResult:
        if authorized is not True: raise ProviderAttemptStoreError("router decision did not authorize fallback")
        values = {name: _token(value, name) for name, value in {"logical_attempt_id": logical_attempt_id, "route_id": route_id, "decision_id": decision_id, "task_id": task_id, "worktree_id": worktree_id}.items()}
        path = self._path(values["logical_attempt_id"]); lock_path = path.with_suffix(".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.fchmod(fd, 0o600)
            lock_stat = os.fstat(fd)
            if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.geteuid() or lock_stat.st_nlink != 1:
                raise ProviderAttemptStoreError("attempt reservation lock is invalid")
            fcntl.flock(fd, fcntl.LOCK_EX)
            if path.exists():
                _owned_regular(path)
                data = json.loads(path.read_text(encoding="utf-8"))
                reservation = ProviderAttemptReservation(**data)
                if {name: getattr(reservation, name) for name in values} != values or reservation.state not in {"reserved", "completed"}:
                    raise ProviderAttemptStoreError("existing fallback reservation does not match logical attempt")
                return ProviderAttemptCASResult(reservation, created=False, adopted=True)
            reservation = ProviderAttemptReservation(**values, reservation_id="sha256:" + hashlib.sha256((values["logical_attempt_id"] + "\0" + secrets.token_hex(16)).encode()).hexdigest(), state="reserved", created_at_ms=int(time.time() * 1000) if now_ms is None else int(now_ms))
            temporary = path.with_name("." + path.name + "." + secrets.token_hex(8))
            try:
                with open(temporary, "xb") as stream:
                    stream.write(_canonical(asdict(reservation)) + b"\n"); stream.flush(); os.fsync(stream.fileno())
                os.chmod(temporary, 0o600); os.replace(temporary, path)
            finally:
                try: temporary.unlink()
                except FileNotFoundError: pass
            return ProviderAttemptCASResult(reservation, created=True, adopted=False)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)

    def complete(self, reservation: ProviderAttemptReservation) -> ProviderAttemptReservation:
        """Record terminal state without restoring capacity or altering counters."""
        path = self._path(reservation.logical_attempt_id)
        if not path.exists(): raise ProviderAttemptStoreError("fallback reservation is absent")
        _owned_regular(path)
        current = ProviderAttemptReservation(**json.loads(path.read_text(encoding="utf-8")))
        if current != reservation: raise ProviderAttemptStoreError("fallback reservation changed")
        completed = ProviderAttemptReservation(**{**asdict(current), "state": "completed"})
        temporary = path.with_name("." + path.name + "." + secrets.token_hex(8))
        with open(temporary, "xb") as stream:
            stream.write(_canonical(asdict(completed)) + b"\n"); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o600); os.replace(temporary, path)
        return completed
