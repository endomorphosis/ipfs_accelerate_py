"""Exclusive L3 campaign leases with monotonic fencing.

Named corpus, split, tokenizer, checkpoint, run, proof, evaluation,
promotion, and publication mutations reuse the existing checkout-lock CAS
guard.  This module does not introduce a second scheduler or checkpoint
store.  Duplicate live writers are rejected; a write without the current
fence is rejected; promotion remains a distinct exclusive key and is never
implied by holding a checkpoint or run lease.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from ..runtime.learning_checkpoint import (
    CAMPAIGN_DURABILITY_REQUIREMENT_ID,
    LEASE_DEFAULT_DURATION_MS,
    LEASE_DEFAULT_HEARTBEAT_MS,
    LEASE_DEFAULT_MAX_ATTEMPTS,
    L3ResourceKind,
    NAMED_L3_RESOURCES,
    StaleFenceError,
    assert_distinct_l3_lease_keys,
    exclusive_lease_key,
)
from .checkout_lock import serialized_lock_update


CAMPAIGN_LEASE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/campaign-l3-lease@1"
CAMPAIGN_LEASE_DIRNAME: Final = "agent-campaign-l3-leases"


class CampaignLeaseError(RuntimeError):
    """Malformed or unsafe L3 lease operation."""


class DuplicateWriterError(CampaignLeaseError):
    """Another unexpired owner already holds the exclusive key."""


class AttemptBoundError(CampaignLeaseError):
    """LEASE-DEFAULT attempt budget was exhausted for this key."""


class LeaseExpiredError(CampaignLeaseError):
    """The caller no longer holds a live lease."""


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not str(value).strip():
        raise CampaignLeaseError(f"{name} must be a non-empty string")
    text = str(value).strip()
    if "\x00" in text:
        raise CampaignLeaseError(f"{name} must not contain NUL")
    return text


def _required_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignLeaseError(f"{name} must be an integer")
    if value < minimum:
        raise CampaignLeaseError(f"{name} must be at least {minimum}")
    return value


def _now_ms(clock: Callable[[], float]) -> int:
    return int(clock() * 1000)


def _safe_key_name(key: str) -> str:
    return "".join(character if character.isalnum() or character in "-._" else "-" for character in key)


@dataclass(frozen=True)
class CampaignLease:
    """One exclusive L3 ownership record bound to a monotonic fence."""

    lease_key: str
    resource: L3ResourceKind
    owner_id: str
    lease_id: str
    fence: int
    attempt: int
    issued_at_ms: int
    heartbeat_at_ms: int
    expires_at_ms: int
    resource_id: str = ""
    schema: str = CAMPAIGN_LEASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CAMPAIGN_LEASE_SCHEMA:
            raise CampaignLeaseError("unsupported campaign lease schema")
        object.__setattr__(self, "lease_key", _required_text(self.lease_key, "lease_key"))
        selected = (
            self.resource
            if isinstance(self.resource, L3ResourceKind)
            else L3ResourceKind(str(self.resource))
        )
        object.__setattr__(self, "resource", selected)
        object.__setattr__(self, "owner_id", _required_text(self.owner_id, "owner_id"))
        object.__setattr__(self, "lease_id", _required_text(self.lease_id, "lease_id"))
        object.__setattr__(self, "fence", _required_int(self.fence, "fence", minimum=1))
        object.__setattr__(self, "attempt", _required_int(self.attempt, "attempt", minimum=1))
        for name in ("issued_at_ms", "heartbeat_at_ms", "expires_at_ms"):
            object.__setattr__(self, name, _required_int(getattr(self, name), name, minimum=0))
        object.__setattr__(self, "resource_id", str(self.resource_id or "").strip())
        expected_key = exclusive_lease_key(self.resource, resource_id=self.resource_id)
        if self.lease_key != expected_key:
            raise CampaignLeaseError("lease_key does not match the named L3 resource")

    def is_expired(self, now_ms: int) -> bool:
        return int(now_ms) >= self.expires_at_ms

    def heartbeat_overdue(self, now_ms: int, *, heartbeat_ms: int = LEASE_DEFAULT_HEARTBEAT_MS) -> bool:
        return int(now_ms) - self.heartbeat_at_ms > int(heartbeat_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requirement_id": CAMPAIGN_DURABILITY_REQUIREMENT_ID,
            "lease_key": self.lease_key,
            "resource": self.resource.value,
            "resource_id": self.resource_id,
            "owner_id": self.owner_id,
            "lease_id": self.lease_id,
            "fence": self.fence,
            "attempt": self.attempt,
            "issued_at_ms": self.issued_at_ms,
            "heartbeat_at_ms": self.heartbeat_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignLease":
        if not isinstance(payload, Mapping):
            raise CampaignLeaseError("campaign lease must be an object")
        return cls(
            lease_key=str(payload.get("lease_key") or ""),
            resource=L3ResourceKind(str(payload.get("resource") or "")),
            owner_id=str(payload.get("owner_id") or ""),
            lease_id=str(payload.get("lease_id") or ""),
            fence=payload.get("fence", 0),  # type: ignore[arg-type]
            attempt=payload.get("attempt", 0),  # type: ignore[arg-type]
            issued_at_ms=payload.get("issued_at_ms", 0),  # type: ignore[arg-type]
            heartbeat_at_ms=payload.get("heartbeat_at_ms", 0),  # type: ignore[arg-type]
            expires_at_ms=payload.get("expires_at_ms", 0),  # type: ignore[arg-type]
            resource_id=str(payload.get("resource_id") or ""),
            schema=str(payload.get("schema") or CAMPAIGN_LEASE_SCHEMA),
        )


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                    "utf-8"
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


class CampaignLeaseCoordinator:
    """CAS coordinator for the closed L3 exclusive-key catalog."""

    def __init__(
        self,
        root: Path | str,
        *,
        lease_duration_ms: int = LEASE_DEFAULT_DURATION_MS,
        heartbeat_ms: int = LEASE_DEFAULT_HEARTBEAT_MS,
        max_attempts: int = LEASE_DEFAULT_MAX_ATTEMPTS,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if (
            isinstance(lease_duration_ms, bool)
            or not isinstance(lease_duration_ms, int)
            or lease_duration_ms < 1
        ):
            raise CampaignLeaseError("lease_duration_ms must be a positive integer")
        if isinstance(heartbeat_ms, bool) or not isinstance(heartbeat_ms, int) or heartbeat_ms < 1:
            raise CampaignLeaseError("heartbeat_ms must be a positive integer")
        if heartbeat_ms > LEASE_DEFAULT_HEARTBEAT_MS:
            raise CampaignLeaseError("LEASE-DEFAULT heartbeat must be at most 60 seconds")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise CampaignLeaseError("max_attempts must be a positive integer")
        if max_attempts > LEASE_DEFAULT_MAX_ATTEMPTS:
            raise CampaignLeaseError("LEASE-DEFAULT allows at most three attempts")
        self.root = Path(root)
        self.lease_duration_ms = lease_duration_ms
        self.heartbeat_ms = heartbeat_ms
        self.max_attempts = max_attempts
        self.clock = clock or time.time
        assert_distinct_l3_lease_keys()

    @property
    def catalog(self) -> dict[L3ResourceKind, str]:
        return {kind: exclusive_lease_key(kind) for kind in NAMED_L3_RESOURCES}

    def path_for(
        self,
        kind: L3ResourceKind | str,
        *,
        resource_id: str = "",
    ) -> Path:
        key = exclusive_lease_key(kind, resource_id=resource_id)
        digest = content_identity({"kind": "campaign-l3-lease-path", "lease_key": key})
        return self.root / CAMPAIGN_LEASE_DIRNAME / f"{_safe_key_name(key)}-{digest[-16:]}.json"

    def load(
        self,
        kind: L3ResourceKind | str,
        *,
        resource_id: str = "",
    ) -> CampaignLease | None:
        path = self.path_for(kind, resource_id=resource_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CampaignLeaseError("campaign lease record is malformed") from exc
        if not isinstance(payload, Mapping):
            raise CampaignLeaseError("campaign lease record must be an object")
        return CampaignLease.from_dict(payload)

    def _publish(self, record: CampaignLease) -> CampaignLease:
        _atomic_write(self.path_for(record.resource, resource_id=record.resource_id), record.to_dict())
        return record

    def acquire(
        self,
        kind: L3ResourceKind | str,
        *,
        owner_id: str,
        resource_id: str = "",
        reclaim_expired: bool = True,
    ) -> CampaignLease:
        """Acquire the exclusive key, fencing any expired predecessor."""

        owner = _required_text(owner_id, "owner_id")
        selected = kind if isinstance(kind, L3ResourceKind) else L3ResourceKind(str(kind))
        path = self.path_for(selected, resource_id=resource_id)
        with serialized_lock_update(path):
            now_ms = _now_ms(self.clock)
            current = None
            try:
                raw = path.read_text(encoding="utf-8")
                payload = json.loads(raw)
                if isinstance(payload, Mapping):
                    current = CampaignLease.from_dict(payload)
            except FileNotFoundError:
                current = None
            except (OSError, UnicodeError, json.JSONDecodeError, CampaignLeaseError):
                current = None
            if current is not None and not current.is_expired(now_ms):
                if current.owner_id != owner:
                    raise DuplicateWriterError(
                        f"duplicate writer denied for {current.lease_key}: held by {current.owner_id}"
                    )
                # Same owner renews in place without consuming another attempt.
                renewed = CampaignLease(
                    lease_key=current.lease_key,
                    resource=current.resource,
                    owner_id=owner,
                    lease_id=current.lease_id,
                    fence=current.fence + 1,
                    attempt=current.attempt,
                    issued_at_ms=current.issued_at_ms,
                    heartbeat_at_ms=now_ms,
                    expires_at_ms=now_ms + self.lease_duration_ms,
                    resource_id=current.resource_id,
                )
                return self._publish(renewed)
            if current is not None and current.is_expired(now_ms) and not reclaim_expired:
                raise LeaseExpiredError(f"{current.lease_key} is expired")
            next_attempt = 1 if current is None else current.attempt + 1
            if next_attempt > self.max_attempts:
                raise AttemptBoundError(
                    f"{exclusive_lease_key(selected, resource_id=resource_id)} exceeded "
                    f"{self.max_attempts} attempts"
                )
            record = CampaignLease(
                lease_key=exclusive_lease_key(selected, resource_id=resource_id),
                resource=selected,
                owner_id=owner,
                lease_id=content_identity(
                    {
                        "kind": "campaign-l3-lease",
                        "lease_key": exclusive_lease_key(selected, resource_id=resource_id),
                        "owner_id": owner,
                        "issued_at_ms": now_ms,
                        "predecessor": None if current is None else current.lease_id,
                    }
                ),
                fence=1 if current is None else current.fence + 1,
                attempt=next_attempt,
                issued_at_ms=now_ms,
                heartbeat_at_ms=now_ms,
                expires_at_ms=now_ms + self.lease_duration_ms,
                resource_id=str(resource_id or "").strip(),
            )
            return self._publish(record)

    def heartbeat(self, lease: CampaignLease, *, expected_fence: int) -> CampaignLease:
        """Renew the 30-minute window after proving the caller still holds the fence."""

        self.assert_write_fence(lease, expected_fence)
        path = self.path_for(lease.resource, resource_id=lease.resource_id)
        with serialized_lock_update(path):
            current = self.load(lease.resource, resource_id=lease.resource_id)
            now_ms = _now_ms(self.clock)
            if current is None or current.lease_id != lease.lease_id:
                raise LeaseExpiredError("campaign lease is no longer current")
            if current.fence != expected_fence:
                raise StaleFenceError(
                    f"stale fence {expected_fence} for {current.lease_key}; current is {current.fence}"
                )
            if current.is_expired(now_ms):
                raise LeaseExpiredError(f"{current.lease_key} expired before heartbeat")
            renewed = CampaignLease(
                lease_key=current.lease_key,
                resource=current.resource,
                owner_id=current.owner_id,
                lease_id=current.lease_id,
                fence=current.fence + 1,
                attempt=current.attempt,
                issued_at_ms=current.issued_at_ms,
                heartbeat_at_ms=now_ms,
                expires_at_ms=now_ms + self.lease_duration_ms,
                resource_id=current.resource_id,
            )
            return self._publish(renewed)

    def assert_write_fence(self, lease: CampaignLease, expected_fence: int) -> CampaignLease:
        """Reject overwrite or resume that does not carry the current fence."""

        if isinstance(expected_fence, bool) or not isinstance(expected_fence, int):
            raise StaleFenceError("expected_fence must be an integer")
        current = self.load(lease.resource, resource_id=lease.resource_id)
        if current is None:
            raise LeaseExpiredError("campaign lease is missing")
        if current.lease_id != lease.lease_id:
            raise StaleFenceError("campaign lease identity was rotated")
        if current.fence != expected_fence:
            raise StaleFenceError(
                f"stale fence {expected_fence} for {current.lease_key}; current is {current.fence}"
            )
        if current.is_expired(_now_ms(self.clock)):
            raise LeaseExpiredError(f"{current.lease_key} expired")
        return current

    def release(self, lease: CampaignLease, *, expected_fence: int) -> bool:
        path = self.path_for(lease.resource, resource_id=lease.resource_id)
        with serialized_lock_update(path):
            current = self.load(lease.resource, resource_id=lease.resource_id)
            if (
                current is None
                or current.lease_id != lease.lease_id
                or current.fence != expected_fence
            ):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            return True


__all__ = (
    "CAMPAIGN_LEASE_DIRNAME",
    "CAMPAIGN_LEASE_SCHEMA",
    "AttemptBoundError",
    "CampaignLease",
    "CampaignLeaseCoordinator",
    "CampaignLeaseError",
    "DuplicateWriterError",
    "LeaseExpiredError",
)
