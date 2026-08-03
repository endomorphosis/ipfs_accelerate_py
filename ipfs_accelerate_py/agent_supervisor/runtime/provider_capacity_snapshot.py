"""Private, fresh capacity snapshots for independent Grok/Codex review.

The trust boundary is deliberately local and explicit: an operator-owned
0700 directory and 0600, single-link snapshot file.  The digest detects torn
or corrupted content; it is not a third-party signature.  Production callers
must name this file explicitly and must not promote ambient JSON telemetry to
the same authority.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import logging
import os
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .resource_scheduler import ProviderCapacity, normalize_provider_capacities

logger = logging.getLogger(__name__)

PROVIDER_CAPACITY_SNAPSHOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/provider-capacity-snapshot@1"
)
PROVIDER_CAPACITY_TRUST: Final = "local-owner-private-file-v1"
PROVIDER_CAPACITY_BUDGET_SEMANTICS: Final = (
    "operator-admission-budget-not-provider-reported-quota"
)
DUAL_REVIEW_PROVIDER_ID: Final = "grok-codex-review-pair"
DUAL_REVIEW_PROVIDER_IDS: Final = ("grok_cli", "codex_cli")
GROK_TERRA_CANDIDATE_PROVIDER_ID: Final = "grok-terra-candidate-route"
GROK_TERRA_CANDIDATE_CAPABILITIES: Final = (
    "candidate-only",
    "codex-cli",
    "terra-implement",
)
DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES: Final = {
    "grok_cli": frozenset({"grok-cli", "grok-implement"}),
    "codex_cli": frozenset(
        {"codex-cli", "codex-review", "independent-review"}
    ),
}
DUAL_REVIEW_PROVIDER_CAPABILITIES: Final = tuple(
    sorted(
        set().union(*DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES.values())
    )
)
DUAL_REVIEW_REQUIRED_CAPABILITIES: Final = tuple(
    f"llm:{item}" for item in DUAL_REVIEW_PROVIDER_CAPABILITIES
)
DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS: Final = 30_000
MAX_PROVIDER_CAPACITY_SNAPSHOT_BYTES: Final = 64 * 1024
_PROVIDER_CAPACITY_FIELDS: Final = frozenset(
    {
        "provider_id",
        "healthy",
        "quota_remaining",
        "latency_ms",
        "context_window_tokens",
        "token_budget_remaining",
        "max_concurrency",
        "active_requests",
        "capabilities",
        "observed_at_ms",
        "retry_after_ms",
    }
)
_ENVELOPE_FIELDS: Final = frozenset(
    {
        "schema",
        "trust",
        "budget_semantics",
        "observed_at_ms",
        "expires_at_ms",
        "providers",
        "snapshot_id",
    }
)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate provider capacity field {key!r}")
        result[key] = value
    return result


def _positive_age(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("provider capacity max age must be a positive integer")
    return value


def _private_directory(path: Path, *, create: bool) -> Path:
    directory = Path(os.path.abspath(os.fspath(path))).parent
    if create:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        info = os.lstat(directory)
    except OSError as exc:
        raise ValueError("provider capacity directory is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError("provider capacity directory must be a real directory")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise ValueError("provider capacity directory owner is invalid")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("provider capacity directory permissions must be 0700")
    return directory


def _open_private_file(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        return os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("provider capacity snapshot target is unsafe") from exc
        raise


def _read_private_bytes(path: Path) -> bytes:
    target = Path(os.path.abspath(os.fspath(path)))
    _private_directory(target, create=False)
    descriptor = _open_private_file(target)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("provider capacity snapshot must be a regular file")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise ValueError("provider capacity snapshot owner is invalid")
        if before.st_nlink != 1:
            raise ValueError("provider capacity snapshot cannot be hard-linked")
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise ValueError(
                "provider capacity snapshot permissions must be 0600 or stricter"
            )
        if before.st_size <= 0 or before.st_size > MAX_PROVIDER_CAPACITY_SNAPSHOT_BYTES:
            raise ValueError("provider capacity snapshot size is invalid")
        raw = b""
        while len(raw) <= MAX_PROVIDER_CAPACITY_SNAPSHOT_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    65_536,
                    MAX_PROVIDER_CAPACITY_SNAPSHOT_BYTES + 1 - len(raw),
                ),
            )
            if not chunk:
                break
            raw += chunk
        after = os.fstat(descriptor)
        if (
            len(raw) > MAX_PROVIDER_CAPACITY_SNAPSHOT_BYTES
            or len(raw) != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise ValueError("provider capacity snapshot changed while being read")
        return raw
    finally:
        os.close(descriptor)


def _record(capacity: ProviderCapacity) -> dict[str, Any]:
    return {
        "provider_id": capacity.provider_id,
        "healthy": capacity.healthy,
        "quota_remaining": capacity.quota_remaining,
        "latency_ms": capacity.latency_ms,
        "context_window_tokens": capacity.context_window_tokens,
        "token_budget_remaining": capacity.token_budget_remaining,
        "max_concurrency": capacity.max_concurrency,
        "active_requests": capacity.active_requests,
        "capabilities": list(capacity.capabilities),
        "observed_at_ms": capacity.observed_at_ms,
        "retry_after_ms": capacity.retry_after_ms,
    }


def _snapshot_id(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "snapshot_id"}
    return "sha256:" + hashlib.sha256(_json_bytes(body)).hexdigest()


def _parse_snapshot(
    raw: bytes,
    *,
    max_age_ms: int,
    now_ms: int,
    require_fresh: bool,
) -> tuple[dict[str, Any], tuple[ProviderCapacity, ...]]:
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite provider capacity value {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("provider capacity snapshot JSON is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != _ENVELOPE_FIELDS:
        raise ValueError("provider capacity snapshot shape is invalid")
    if payload.get("schema") != PROVIDER_CAPACITY_SNAPSHOT_SCHEMA:
        raise ValueError("provider capacity snapshot schema is invalid")
    if payload.get("trust") != PROVIDER_CAPACITY_TRUST:
        raise ValueError("provider capacity snapshot trust boundary is invalid")
    if payload.get("budget_semantics") != PROVIDER_CAPACITY_BUDGET_SEMANTICS:
        raise ValueError("provider capacity budget semantics are invalid")
    for field_name in ("observed_at_ms", "expires_at_ms"):
        value = payload.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"provider capacity {field_name} is invalid")
    observed_at_ms = int(payload["observed_at_ms"])
    expires_at_ms = int(payload["expires_at_ms"])
    if expires_at_ms <= observed_at_ms or expires_at_ms - observed_at_ms > max_age_ms:
        raise ValueError("provider capacity snapshot TTL is invalid")
    if require_fresh and (
        observed_at_ms > now_ms
        or now_ms > expires_at_ms
        or now_ms - observed_at_ms > max_age_ms
    ):
        raise ValueError("provider capacity snapshot is stale")
    if payload.get("snapshot_id") != _snapshot_id(payload):
        raise ValueError("provider capacity snapshot digest is invalid")

    raw_providers = payload.get("providers")
    if not isinstance(raw_providers, dict) or set(raw_providers) != set(
        DUAL_REVIEW_PROVIDER_IDS
    ):
        raise ValueError(
            "provider capacity snapshot inventory must be exactly Grok and Codex"
        )
    capacities: list[ProviderCapacity] = []
    for provider_id, value in raw_providers.items():
        if (
            not isinstance(provider_id, str)
            or not provider_id
            or provider_id != provider_id.strip().lower()
            or not isinstance(value, dict)
            or set(value) != _PROVIDER_CAPACITY_FIELDS
            or value.get("provider_id") != provider_id
        ):
            raise ValueError("provider capacity record shape is invalid")
        if not isinstance(value.get("healthy"), bool):
            raise ValueError("provider capacity health must be boolean")
        for field_name in _PROVIDER_CAPACITY_FIELDS - {
            "provider_id",
            "healthy",
            "capabilities",
        }:
            field_value = value.get(field_name)
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise ValueError(
                    f"provider capacity {field_name} must be an integer"
                )
        capabilities = value.get("capabilities")
        if not isinstance(capabilities, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in capabilities
        ):
            raise ValueError("provider capacity capabilities are invalid")
        capacity = ProviderCapacity.from_mapping(value)
        if require_fresh and (
            capacity.observed_at_ms <= 0
            or capacity.observed_at_ms > now_ms
            or now_ms - capacity.observed_at_ms > max_age_ms
        ):
            raise ValueError("provider capacity observation is stale")
        capacities.append(capacity)
    if min(item.observed_at_ms for item in capacities) != observed_at_ms:
        raise ValueError(
            "provider capacity envelope does not bind its oldest sample"
        )
    return payload, tuple(sorted(capacities, key=lambda item: item.provider_id))


def load_provider_capacity_snapshot(
    path: Path,
    *,
    max_age_ms: int = DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
    now_ms: int | None = None,
) -> tuple[ProviderCapacity, ...]:
    """Load one bounded, owner-private, non-stale capacity snapshot."""

    hard_ttl = _positive_age(max_age_ms)
    current = int(time.time() * 1000) if now_ms is None else int(now_ms)
    _payload, capacities = _parse_snapshot(
        _read_private_bytes(path),
        max_age_ms=hard_ttl,
        now_ms=current,
        require_fresh=True,
    )
    return capacities


def provider_capacity_observation_floor(
    path: Path,
    *,
    max_age_ms: int = DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
) -> int:
    """Return the newest securely parsed provider observation, even if stale.

    Capacity publishers use this as a strict wall-clock CAS floor across
    process restarts. The same owner/private-file and digest checks as the
    production reader still apply; only freshness relative to wall clock is
    omitted.
    """

    hard_ttl = _positive_age(max_age_ms)
    _payload, capacities = _parse_snapshot(
        _read_private_bytes(path),
        max_age_ms=hard_ttl,
        now_ms=0,
        require_fresh=False,
    )
    return max(item.observed_at_ms for item in capacities)


def _locked_snapshot_file(target: Path) -> tuple[int, Path]:
    lock_path = target.with_name(f".{target.name}.lock")
    descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or (hasattr(os, "geteuid") and info.st_uid != os.geteuid())
    ):
        os.close(descriptor)
        raise ValueError("provider capacity publication lock is unsafe")
    os.fchmod(descriptor, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor, lock_path


def write_provider_capacity_snapshot(
    path: Path,
    providers: Mapping[str, Any] | Sequence[ProviderCapacity | Mapping[str, Any]],
    *,
    max_age_ms: int = DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Atomically publish a monotonic owner-private Grok/Codex snapshot.

    Quota, token, and context values are operator admission budgets. They are
    never represented as provider-reported account balances.
    """

    hard_ttl = _positive_age(max_age_ms)
    current = int(time.time() * 1000) if now_ms is None else int(now_ms)
    supplied = normalize_provider_capacities(providers)
    by_id = {item.provider_id: item for item in supplied}
    if set(by_id) != set(DUAL_REVIEW_PROVIDER_IDS):
        raise ValueError(
            "provider capacity snapshot inventory must be exactly Grok and Codex"
        )
    normalized = tuple(by_id[provider_id] for provider_id in DUAL_REVIEW_PROVIDER_IDS)
    if any(item.observed_at_ms <= 0 for item in normalized):
        raise ValueError("provider capacity observations require timestamps")
    if any(item.observed_at_ms > current for item in normalized):
        raise ValueError("provider capacity observation cannot be in the future")
    observed_at_ms = min(item.observed_at_ms for item in normalized)
    expires_at_ms = observed_at_ms + hard_ttl
    if current > expires_at_ms:
        raise ValueError("provider capacity observations are already stale")
    payload: dict[str, Any] = {
        "schema": PROVIDER_CAPACITY_SNAPSHOT_SCHEMA,
        "trust": PROVIDER_CAPACITY_TRUST,
        "budget_semantics": PROVIDER_CAPACITY_BUDGET_SEMANTICS,
        "observed_at_ms": observed_at_ms,
        "expires_at_ms": expires_at_ms,
        "providers": {
            item.provider_id: _record(item) for item in normalized
        },
    }
    payload["snapshot_id"] = _snapshot_id(payload)
    encoded = _json_bytes(payload)
    if len(encoded) > MAX_PROVIDER_CAPACITY_SNAPSHOT_BYTES:
        raise ValueError("provider capacity snapshot exceeds its byte bound")

    target = Path(os.path.abspath(os.fspath(path)))
    directory = _private_directory(target, create=True)
    lock_descriptor, _lock_path = _locked_snapshot_file(target)
    try:
        try:
            existing_raw = _read_private_bytes(target)
        except FileNotFoundError:
            existing_payload = None
        else:
            existing_payload, existing_capacities = _parse_snapshot(
                existing_raw,
                max_age_ms=hard_ttl,
                now_ms=current,
                require_fresh=False,
            )
        if existing_payload is not None:
            previous_by_id = {
                item.provider_id: item for item in existing_capacities
            }
            unchanged = True
            for provider_id in DUAL_REVIEW_PROVIDER_IDS:
                candidate = by_id[provider_id]
                previous = previous_by_id[provider_id]
                if candidate.observed_at_ms < previous.observed_at_ms:
                    raise ValueError(
                        "provider capacity publication would move observation "
                        f"time backward for {provider_id}"
                    )
                if candidate.observed_at_ms == previous.observed_at_ms:
                    if _record(candidate) != _record(previous):
                        raise ValueError(
                            "provider capacity publication conflicts at the "
                            f"same observation time for {provider_id}"
                        )
                else:
                    unchanged = False
            if unchanged:
                return dict(existing_payload)

        temporary = directory / (
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                os.fchmod(descriptor, 0o600)
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError(
                            "short write while publishing provider capacity"
                        )
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, target)
            directory_descriptor = os.open(
                directory,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
    load_provider_capacity_snapshot(
        target,
        max_age_ms=hard_ttl,
        now_ms=current,
    )
    return payload


def synthesize_dual_review_provider_capacity(
    providers: Any,
    *,
    max_age_ms: int = DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
    now_ms: int | None = None,
) -> tuple[ProviderCapacity, ...]:
    """Add one fail-closed reservation target for both provider roles."""

    hard_ttl = _positive_age(max_age_ms)
    current = int(time.time() * 1000) if now_ms is None else int(now_ms)
    try:
        normalized = tuple(
            item
            for item in normalize_provider_capacities(providers)
            if item.provider_id != DUAL_REVIEW_PROVIDER_ID
        )
    except Exception:
        logger.warning("Invalid provider telemetry; dual-review capacity is closed")
        normalized = ()
    by_id = {item.provider_id: item for item in normalized}
    required = [by_id.get(provider_id) for provider_id in DUAL_REVIEW_PROVIDER_IDS]
    present = [item for item in required if item is not None]
    complete = len(present) == len(DUAL_REVIEW_PROVIDER_IDS)
    fresh = complete and all(
        item.observed_at_ms > 0
        and item.observed_at_ms <= current
        and current - item.observed_at_ms <= hard_ttl
        for item in present
    )
    known_limits = complete and all(
        item.quota_remaining >= 0
        and item.context_window_tokens >= 0
        and item.token_budget_remaining >= 0
        and item.active_requests <= item.max_concurrency
        for item in present
    )
    role_capable = complete and all(
        DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES[item.provider_id].issubset(
            item.capabilities
        )
        for item in present
    )
    usable = complete and fresh and known_limits and role_capable and all(
        item.healthy for item in present
    )
    observed = min(
        (item.observed_at_ms for item in present if item.observed_at_ms > 0),
        default=0,
    )
    latency = max((item.latency_ms for item in present), default=0)
    retry_after = max((item.retry_after_ms for item in present), default=0)
    if usable:
        total_slots = min(item.max_concurrency for item in present)
        free_slots = min(item.available_concurrency for item in present)
        pair = ProviderCapacity(
            provider_id=DUAL_REVIEW_PROVIDER_ID,
            healthy=True,
            quota_remaining=min(item.quota_remaining for item in present),
            latency_ms=latency,
            context_window_tokens=min(
                item.context_window_tokens for item in present
            ),
            token_budget_remaining=min(
                item.token_budget_remaining for item in present
            ),
            max_concurrency=total_slots,
            active_requests=max(0, total_slots - free_slots),
            capabilities=DUAL_REVIEW_PROVIDER_CAPABILITIES,
            observed_at_ms=observed,
            retry_after_ms=retry_after,
        )
    else:
        # An explicit unhealthy candidate prevents the generic compatibility
        # switch for missing telemetry from bypassing independent review.
        pair = ProviderCapacity(
            provider_id=DUAL_REVIEW_PROVIDER_ID,
            healthy=False,
            quota_remaining=0,
            latency_ms=latency,
            context_window_tokens=0,
            token_budget_remaining=0,
            max_concurrency=0,
            active_requests=0,
            capabilities=DUAL_REVIEW_PROVIDER_CAPABILITIES,
            observed_at_ms=observed,
            retry_after_ms=retry_after,
        )
    # This is deliberately a separate, non-review provider.  It only opens
    # scheduler admission far enough for the child daemon to obtain its own
    # typed hard-quota evidence from Grok and, if that evidence verifies, run
    # the exact Terra implementation fallback.  It must never make the
    # independent-review pair healthy or advertise review capabilities.
    grok = by_id.get("grok_cli")
    codex = by_id.get("codex_cli")
    candidate_usable = bool(
        complete
        and fresh
        and grok is not None
        and codex is not None
        and grok.quota_remaining == 0
        and codex.healthy
        and codex.retry_after_ms == 0
        and codex.quota_remaining > 0
        and codex.context_window_tokens >= 0
        and codex.token_budget_remaining >= 0
        and codex.active_requests <= codex.max_concurrency
        and "codex-cli" in codex.capabilities
    )
    candidate = ProviderCapacity(
        provider_id=GROK_TERRA_CANDIDATE_PROVIDER_ID,
        healthy=candidate_usable,
        quota_remaining=(codex.quota_remaining if candidate_usable else 0),
        latency_ms=(codex.latency_ms if codex is not None else latency),
        context_window_tokens=(
            codex.context_window_tokens if candidate_usable else 0
        ),
        token_budget_remaining=(
            codex.token_budget_remaining if candidate_usable else 0
        ),
        max_concurrency=(codex.max_concurrency if candidate_usable else 0),
        active_requests=(codex.active_requests if candidate_usable else 0),
        capabilities=GROK_TERRA_CANDIDATE_CAPABILITIES,
        observed_at_ms=observed,
        retry_after_ms=(codex.retry_after_ms if codex is not None else retry_after),
    )
    return tuple(
        sorted((*normalized, pair, candidate), key=lambda item: item.provider_id)
    )


__all__ = [
    "DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS",
    "DUAL_REVIEW_PROVIDER_CAPABILITIES",
    "DUAL_REVIEW_PROVIDER_ID",
    "DUAL_REVIEW_PROVIDER_IDS",
    "DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES",
    "DUAL_REVIEW_REQUIRED_CAPABILITIES",
    "GROK_TERRA_CANDIDATE_CAPABILITIES",
    "GROK_TERRA_CANDIDATE_PROVIDER_ID",
    "PROVIDER_CAPACITY_BUDGET_SEMANTICS",
    "PROVIDER_CAPACITY_SNAPSHOT_SCHEMA",
    "PROVIDER_CAPACITY_TRUST",
    "load_provider_capacity_snapshot",
    "provider_capacity_observation_floor",
    "synthesize_dual_review_provider_capacity",
    "write_provider_capacity_snapshot",
]
