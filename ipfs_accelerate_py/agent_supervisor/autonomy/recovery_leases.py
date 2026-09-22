"""Multi-writer recovery leases: one owner per task, lane, or checkout.

TTL expiry is stealable. Malformed or unreadable records are not stolen.
Cross-process updates take an advisory flock. TypeSafe is not consulted.

Environment (optional, fail-open if unset):

- ``AUTONOMY_RECOVERY_LEASE_PATH`` — shared JSON table
- ``AGENT_SUPERVISOR_STATE_ROOT`` / ``AUTONOMY_STATE_ROOT`` — default
  ``{root}/recovery-leases.json``
- ``AUTONOMY_OWNER_ID`` — stable lane owner (else ``pid:<pid>``)
- ``AUTONOMY_RECOVERY_LEASE_TTL_SECONDS`` — daemon TTL (default 300)
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]

RECOVERY_LEASE_KINDS = ("task", "lane", "checkout")
DEFAULT_RECOVERY_LEASE_TTL_SECONDS = 30.0
DEFAULT_DAEMON_RECOVERY_LEASE_TTL_SECONDS = 300.0
_MAX_RESOURCE_ID_BYTES = 256


def _bounded_id(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text or len(text.encode("utf-8")) > _MAX_RESOURCE_ID_BYTES or "\x00" in text:
        raise ValueError(f"{name} must be a bounded identifier")
    return text


def _lease_key(kind: str, resource_id: str) -> str:
    chosen = str(kind or "").strip()
    if chosen not in RECOVERY_LEASE_KINDS:
        raise ValueError("recovery lease kind is not closed")
    return f"{chosen}:{_bounded_id(resource_id, 'resource_id')}"


def _expires_at(record: Mapping[str, Any] | None) -> float:
    if not isinstance(record, Mapping):
        return 0.0
    try:
        return float(record.get("expires_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class RecoveryLease:
    kind: str
    resource_id: str
    owner_id: str
    lease_id: str
    expires_at: float
    acquired_at: float
    pid: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "resource_id": self.resource_id,
            "owner_id": self.owner_id,
            "lease_id": self.lease_id,
            "expires_at": self.expires_at,
            "acquired_at": self.acquired_at,
            "pid": int(self.pid),
        }


def recovery_lease_ttl_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    default: float = DEFAULT_RECOVERY_LEASE_TTL_SECONDS,
) -> float:
    env = os.environ if environ is None else environ
    raw = str(env.get("AUTONOMY_RECOVERY_LEASE_TTL_SECONDS") or "").strip()
    if not raw:
        return float(default)
    try:
        ttl = float(raw)
    except ValueError:
        return float(default)
    return ttl if ttl > 0 else float(default)


def bundle_recovery_lease_ttl_seconds(lease_ms: int | float) -> float:
    """Daemon TTL is at least the default and at least 2x the DuckDB lease."""

    try:
        seconds = float(lease_ms) / 1000.0
    except (TypeError, ValueError):
        seconds = 0.0
    return max(DEFAULT_DAEMON_RECOVERY_LEASE_TTL_SECONDS, seconds * 2.0)


def recovery_leases_from_env(
    environ: Mapping[str, str] | None = None,
) -> RecoveryLeaseTable:
    """Share a file-backed table when ``AUTONOMY_RECOVERY_LEASE_PATH`` is set."""

    env = os.environ if environ is None else environ
    path = str(env.get("AUTONOMY_RECOVERY_LEASE_PATH") or "").strip()
    if not path:
        root = str(
            env.get("AGENT_SUPERVISOR_STATE_ROOT")
            or env.get("AUTONOMY_STATE_ROOT")
            or ""
        ).strip()
        if root:
            path = str(Path(root) / "recovery-leases.json")
    return RecoveryLeaseTable(path or None)


class RecoveryLeaseTable:
    """In-memory or file-backed exclusive leases with steal-on-expiry."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._leases: dict[str, dict[str, Any]] = {}

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        with self._lock:
            if self._path is None:
                yield
                return
            guard = self._path.with_name(f".{self._path.name}.update.lock")
            guard.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_CREAT | os.O_RDWR
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(guard, flags, 0o600)
            locked = False
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    locked = True
                elif msvcrt is not None:  # pragma: no cover
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    locked = True
                yield
            finally:
                try:
                    if locked and fcntl is not None:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    elif locked and msvcrt is not None:  # pragma: no cover
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                finally:
                    os.close(fd)

    def _load_unlocked(self) -> dict[str, dict[str, Any]]:
        if self._path is None:
            return dict(self._leases)
        try:
            if not self._path.exists():
                return {}
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {"__unreadable__": {"kind": "malformed"}}
        leases = payload.get("leases") if isinstance(payload, dict) else None
        if not isinstance(leases, dict):
            return {"__unreadable__": {"kind": "malformed"}}
        return {
            str(key): dict(value)
            for key, value in leases.items()
            if isinstance(value, Mapping)
        }

    def _store_unlocked(self, leases: Mapping[str, Mapping[str, Any]]) -> None:
        clean = {str(key): dict(value) for key, value in leases.items()}
        if self._path is None:
            self._leases = clean
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"leases": clean}, sort_keys=True, indent=2) + "\n"
        tmp = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(body.encode("utf-8"))
        with tmp.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self._path)

    def try_acquire(
        self,
        kind: str,
        resource_id: str,
        owner_id: str,
        *,
        now: float | None = None,
        ttl_seconds: float = DEFAULT_RECOVERY_LEASE_TTL_SECONDS,
    ) -> tuple[bool, dict[str, Any]]:
        """Acquire or renew. Steal only when the incumbent is past TTL."""

        key = _lease_key(kind, resource_id)
        owner = _bounded_id(owner_id, "owner_id")
        ttl = float(ttl_seconds)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        stamp = float(time.time() if now is None else now)
        with self._exclusive():
            leases = self._load_unlocked()
            if "__unreadable__" in leases:
                return False, {
                    "blocked": True,
                    "reason": "recovery_lease_unreadable",
                    "key": key,
                }
            existing = leases.get(key)
            leases = {
                item_key: dict(item)
                for item_key, item in leases.items()
                if item_key == key or _expires_at(item) > stamp
            }
            if isinstance(existing, Mapping):
                incumbent = str(existing.get("owner_id") or "")
                expires = _expires_at(existing)
                if incumbent == owner:
                    existing = dict(existing)
                    existing["expires_at"] = stamp + ttl
                    leases[key] = existing
                    self._store_unlocked(leases)
                    return True, {
                        "blocked": False,
                        "reason": "recovery_lease_renewed",
                        "key": key,
                        "lease": dict(existing),
                    }
                if expires > stamp:
                    return False, {
                        "blocked": True,
                        "reason": "recovery_lease_held",
                        "key": key,
                        "owner_id": incumbent,
                        "expires_at": expires,
                    }
            seed = f"{owner}:{key}:{stamp}:{os.getpid()}"
            record = RecoveryLease(
                kind=str(kind),
                resource_id=str(resource_id),
                owner_id=owner,
                lease_id=hashlib.sha1(seed.encode("utf-8")).hexdigest(),
                expires_at=stamp + ttl,
                acquired_at=stamp,
                pid=os.getpid(),
            ).to_dict()
            if existing is not None:
                record["stolen_from"] = str(existing.get("owner_id") or "")
            leases[key] = record
            self._store_unlocked(leases)
            return True, {
                "blocked": False,
                "reason": (
                    "recovery_lease_stolen"
                    if existing is not None
                    else "recovery_lease_acquired"
                ),
                "key": key,
                "lease": record,
            }

    def release(
        self,
        kind: str,
        resource_id: str,
        owner_id: str,
    ) -> dict[str, Any]:
        key = _lease_key(kind, resource_id)
        owner = _bounded_id(owner_id, "owner_id")
        with self._exclusive():
            leases = self._load_unlocked()
            if "__unreadable__" in leases:
                return {"released": False, "reason": "recovery_lease_unreadable", "key": key}
            existing = leases.get(key)
            if not isinstance(existing, Mapping):
                return {"released": True, "reason": "recovery_lease_absent", "key": key}
            if str(existing.get("owner_id") or "") != owner:
                return {
                    "released": False,
                    "reason": "recovery_lease_not_owner",
                    "key": key,
                }
            leases.pop(key, None)
            self._store_unlocked(leases)
            return {"released": True, "reason": "recovery_lease_released", "key": key}

    def inspect(self, kind: str, resource_id: str) -> dict[str, Any] | None:
        key = _lease_key(kind, resource_id)
        with self._exclusive():
            leases = self._load_unlocked()
            existing = leases.get(key)
            return dict(existing) if isinstance(existing, Mapping) else None


__all__ = [
    "DEFAULT_RECOVERY_LEASE_TTL_SECONDS",
    "RECOVERY_LEASE_KINDS",
    "RecoveryLease",
    "RecoveryLeaseTable",
    "recovery_leases_from_env",
    "recovery_lease_ttl_from_env",
    "bundle_recovery_lease_ttl_seconds",
    "DEFAULT_DAEMON_RECOVERY_LEASE_TTL_SECONDS",
]
