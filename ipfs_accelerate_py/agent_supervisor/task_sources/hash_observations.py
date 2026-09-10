"""Short-lived hash observations and fenced production on the existing owner.

Only the database owner calls this module, under its connection transaction
lock. File reads happen in clients, outside these short transactions. These
rows are derived cache observations, never proof or executable admission.
The additive control-plane migration owns DDL; opening this service creates
neither another database nor another writer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import stat
import time
from collections.abc import Callable, Mapping
from typing import Any

IDENTITY_SCHEMA = "ipfs-accelerate/hash-file-identity@1"
IDENTITY_PROFILE = "file-sha256@1"
DEFAULT_TTL_SECONDS = 86_400
MAX_TTL_SECONDS = 604_800
DEFAULT_LEASE_SECONDS = 30
MAX_LEASE_SECONDS = 300
DEFAULT_MAX_ROWS = 4096
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NUMERIC_IDENTITY_FIELDS = frozenset(
    {"dev", "ino", "mode", "uid", "gid", "nlink", "size", "mtime_ns", "ctime_ns"}
)
_IDENTITY_FIELDS = _NUMERIC_IDENTITY_FIELDS | {
    "schema", "algorithm", "profile", "host_boot_id", "mount_id"
}
_COLUMNS = (
    "cache_key, identity_json, owner_generation, state, principal, lease_token, "
    "fence, claimed_at_ms, lease_expires_ms, ttl_ms, sha256, expires_at_ms"
)


class HashObservationError(ValueError):
    """Malformed request, stale producer, or unusable observation."""


class HashObservationUnavailableError(HashObservationError):
    """The owner's admitted schema does not include hash observations."""


def _text(value: Any, name: str, *, limit: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or "\x00" in value:
        raise HashObservationError(f"invalid {name}")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise HashObservationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _identity_json(identity: Mapping[str, Any]) -> str:
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
        raise HashObservationError("identity fields do not match the file witness schema")
    if (
        identity["schema"] != IDENTITY_SCHEMA
        or identity["algorithm"] != "sha256"
        or identity["profile"] != IDENTITY_PROFILE
    ):
        raise HashObservationError("unsupported hash identity profile")
    _text(identity["host_boot_id"], "host_boot_id", limit=256)
    _text(identity["mount_id"], "mount_id", limit=256)
    for field in _NUMERIC_IDENTITY_FIELDS:
        value = identity[field]
        if type(value) is not int or not 0 <= value <= (2**63 - 1):
            raise HashObservationError(f"invalid identity {field}")
    if not stat.S_ISREG(identity["mode"]) or identity["nlink"] < 1:
        raise HashObservationError("hash observations require a linked regular file")
    return json.dumps(dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity_key(identity: Mapping[str, Any]) -> str:
    """Return the canonical key shared by clients and the database owner."""
    return hashlib.sha256(_identity_json(identity).encode("utf-8")).hexdigest()


def _duration_ms(value: Any, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not 0.001 <= value <= maximum or not math.isfinite(value)
    ):
        raise HashObservationError(f"{name} must be between 0.001 and {maximum} seconds")
    return int(value * 1000)


def _request_duration_ms(
    request: Mapping[str, Any], prefix: str, default: int, maximum: int,
) -> int:
    """Accept integer wire durations or the legacy direct seconds API."""
    milliseconds, seconds = f"{prefix}_ms", f"{prefix}_seconds"
    if milliseconds in request:
        if seconds in request:
            raise HashObservationError(f"provide only one {prefix} duration unit")
        value = request[milliseconds]
        if type(value) is not int or not 1 <= value <= maximum * 1000:
            raise HashObservationError(
                f"{milliseconds} must be an integer between 1 and {maximum * 1000}"
            )
        return value
    return _duration_ms(request.get(seconds, default), seconds, maximum)


class HashObservationStore:
    """Bounded owner-managed observations with one fenced producer per key."""

    def __init__(
        self, connection: Any, *, generation: str,
        clock_ms: Callable[[], int] | None = None,
        monotonic_ms: Callable[[], int] | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self._connection = connection
        self._base_generation = _text(generation, "generation")
        self._generation = self._base_generation
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._monotonic_ms = monotonic_ms or (lambda: time.monotonic_ns() // 1_000_000)
        if type(max_rows) is not int or not 1 <= max_rows <= 65_536:
            raise HashObservationError("max_rows must be between 1 and 65536")
        self._max_rows = max_rows
        self._last_wall: int | None = None
        self._last_monotonic: int | None = None
        self._clock_epoch = 0
        self._clock_invalidated = False
        try:
            connection.execute(f"SELECT {_COLUMNS} FROM hash_observations LIMIT 0").fetchall()
        except Exception as exc:
            raise HashObservationUnavailableError(
                "owner schema has no compatible hash observations migration"
            ) from exc

    def _now(self) -> int:
        wall, monotonic = self._clock_ms(), self._monotonic_ms()
        if type(wall) is not int or not 0 <= wall < (2**63 - 1) - MAX_TTL_SECONDS * 1000:
            raise HashObservationError("owner clock is invalid")
        if type(monotonic) is not int or monotonic < 0:
            raise HashObservationError("owner monotonic clock is invalid")
        if self._last_wall is not None and self._last_monotonic is not None:
            # A rollback or lost wall time cannot lengthen the cache lifetime.
            # Tolerate <=1s of sampling skew / normal clock-rate correction.
            if (
                wall < self._last_wall or monotonic < self._last_monotonic
                or wall - self._last_wall + 1000 < monotonic - self._last_monotonic
            ):
                self._clock_epoch += 1
                self._generation = f"{self._base_generation}:clock-{self._clock_epoch}"
                self._clock_invalidated = True
        self._last_wall, self._last_monotonic = wall, monotonic
        return wall

    def handle(self, request: Mapping[str, Any], *, principal: str) -> dict[str, Any]:
        """Apply one authenticated request inside a short owner transaction.

        TTL and lease clocks come exclusively from this owner. Cache hits do
        not move the observation time. Complete is fenced to its identity,
        principal, owner generation, random token, and unexpired lease.
        """
        principal = _text(principal, "principal", limit=2048)
        if not isinstance(request, Mapping):
            raise HashObservationError("request must be a mapping")
        action = request.get("action")
        if not isinstance(action, str) or action not in {"claim", "lookup", "complete", "abort"}:
            raise HashObservationError("unsupported hash observation action")
        allowed = {"action", "key", "identity"}
        if action in {"claim", "lookup"}:
            allowed |= {"ttl_seconds", "ttl_ms"}
            if action == "claim":
                allowed |= {"lease_seconds", "lease_ms"}
        else:
            allowed |= {"generation", "lease_token", "fence"}
            if action == "complete":
                allowed |= {"sha256"}
        if set(request) - allowed:
            raise HashObservationError("unknown hash observation request fields")
        identity = _identity_json(request.get("identity"))
        key = _digest(request.get("key"), "key")
        if key != hashlib.sha256(identity.encode("utf-8")).hexdigest():
            raise HashObservationError("key does not match the file identity")
        ttl_ms = _request_duration_ms(request, "ttl", DEFAULT_TTL_SECONDS, MAX_TTL_SECONDS)
        lease_ms = _request_duration_ms(request, "lease", DEFAULT_LEASE_SECONDS, MAX_LEASE_SECONDS)
        if action in {"complete", "abort"}:
            _text(request.get("generation"), "generation")
            _digest(request.get("lease_token"), "lease_token")
            if type(request.get("fence")) is not int or not 1 <= request["fence"] < 2**63:
                raise HashObservationError("invalid producer fence")
            if action == "complete":
                _digest(request.get("sha256"), "sha256")
        now = self._now()
        connection = self._connection
        connection.execute("BEGIN TRANSACTION")
        try:
            if self._clock_invalidated:
                connection.execute("DELETE FROM hash_observations")
            result = self._handle_transaction(request, principal, identity, key, ttl_ms, lease_ms, now)
            connection.execute("COMMIT")
            self._clock_invalidated = False
            return result
        except BaseException:
            connection.execute("ROLLBACK")
            raise

    def _handle_transaction(
        self, request: Mapping[str, Any], principal: str, identity: str,
        key: str, ttl_ms: int, lease_ms: int, now: int,
    ) -> dict[str, Any]:
        connection = self._connection
        values = connection.execute(
            f"SELECT {_COLUMNS} FROM hash_observations WHERE cache_key = ?", [key]
        ).fetchone()
        names = [name.strip() for name in _COLUMNS.split(",")]
        row = (
            dict(values) if isinstance(values, Mapping)
            else dict(zip(names, values, strict=True)) if values is not None else None
        )
        if row is not None and row["identity_json"] != identity:
            raise HashObservationError("stored hash identity does not match its key")
        live = row is not None and row["owner_generation"] == self._generation
        action = request["action"]
        if action in {"claim", "lookup"}:
            if (
                row is not None and row["state"] == "complete" and row["claimed_at_ms"] <= now
                and now < min(row["expires_at_ms"], row["claimed_at_ms"] + ttl_ms)
            ):
                _digest(row["sha256"], "stored sha256")
                return self._observation(row)
            if live and row["state"] == "claimed" and now < row["lease_expires_ms"]:
                return {
                    "status": "busy", "generation": self._generation,
                    "lease_expires_ms": row["lease_expires_ms"],
                    "retry_after_ms": min(100, max(1, row["lease_expires_ms"] - now)),
                }
            if action == "lookup":
                return {"status": "miss", "generation": self._generation}
            fence = int(row["fence"]) + 1 if row is not None else 1
            if fence >= 2**63:
                raise HashObservationError("producer fence exhausted")
            self._make_room(key, now, replacing=row is not None)
            token = secrets.token_hex(32)
            connection.execute("DELETE FROM hash_observations WHERE cache_key = ?", [key])
            connection.execute(
                f"INSERT INTO hash_observations ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [key, identity, self._generation, "claimed", principal, token,
                 fence, now, now + lease_ms, ttl_ms, "", now + ttl_ms],
            )
            return {
                "status": "claimed", "generation": self._generation,
                "lease_token": token, "fence": fence, "lease_expires_ms": now + lease_ms,
                "observed_at_ms": now, "expires_at_ms": now + ttl_ms,
            }
        if (
            not live or row["state"] != "claimed" or row["principal"] != principal
            or request["generation"] != self._generation
            or request["fence"] != row["fence"]
            or not secrets.compare_digest(request["lease_token"], row["lease_token"])
            or now >= row["lease_expires_ms"] or now < row["claimed_at_ms"]
        ):
            raise HashObservationError("stale or foreign hash producer lease")
        if action == "abort":
            # Preserve the fence until bounded cleanup / replacement.
            connection.execute(
                "UPDATE hash_observations SET state = 'aborted', lease_token = '', "
                "lease_expires_ms = ?, expires_at_ms = ? WHERE cache_key = ?", [now, now, key]
            )
            return {"status": "aborted", "generation": self._generation}
        if now >= row["expires_at_ms"]:
            raise HashObservationError("hash observation TTL expired before publication")
        connection.execute(
            "UPDATE hash_observations SET state = 'complete', sha256 = ?, lease_token = '' "
            "WHERE cache_key = ?", [request["sha256"], key]
        )
        row["sha256"] = request["sha256"]
        return self._observation(row)

    def _observation(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "hit", "generation": self._generation,
            "sha256": row["sha256"], "observed_at_ms": row["claimed_at_ms"],
            "expires_at_ms": row["expires_at_ms"],
        }

    def _make_room(self, key: str, now: int, *, replacing: bool) -> None:
        connection = self._connection
        # Exclude the current key to preserve its fence during replacement.
        connection.execute(
            "DELETE FROM hash_observations WHERE cache_key <> ? AND "
            "((state <> 'complete' AND owner_generation <> ?) OR "
            "(state <> 'claimed' AND expires_at_ms <= ?) OR "
            "(state = 'claimed' AND lease_expires_ms <= ?))",
            [key, self._generation, now, now],
        )
        count_row = connection.execute("SELECT COUNT(*) AS row_count FROM hash_observations").fetchone()
        count = int(count_row["row_count"] if isinstance(count_row, Mapping) else count_row[0])
        if replacing or count < self._max_rows:
            return
        # Evict a completed observation before ever displacing a live producer.
        victim = connection.execute(
            "SELECT cache_key FROM hash_observations WHERE state <> 'claimed' "
            "ORDER BY expires_at_ms, cache_key LIMIT 1"
        ).fetchone()
        if victim is None:
            raise HashObservationError("hash observation producer capacity is full")
        victim_key = victim["cache_key"] if isinstance(victim, Mapping) else victim[0]
        connection.execute("DELETE FROM hash_observations WHERE cache_key = ?", [victim_key])
