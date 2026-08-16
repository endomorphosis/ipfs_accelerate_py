"""Control-plane transaction, CAS, idempotency, and jittered retry primitives.

Interface: ``StateTransaction@1``

DuckDB concurrency is optimistic. Control-plane mutations therefore run as short
transactions that:

* bind expected store generation, fence epoch, and row revision;
* record idempotency keys so a lost response can be replayed exactly once;
* classify conflicts (optimistic, stale generation, fence, idempotency); and
* apply bounded, jittered retries for retryable outcomes only.

This module is transport-agnostic. Callers supply a connection that implements
``execute`` / ``commit`` / ``rollback`` (DuckDB native, ``DuckDBConnection``,
or a Quack-attached session). Arbitrary model-supplied SQL never enters here.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, Protocol, runtime_checkable

from .control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    ControlPlaneBounds,
    ControlPlaneContractError,
    ControlPlaneGenerationError,
    ControlPlaneIdentityError,
    MAX_INT,
    MIN_FENCE_EPOCH,
    MIN_GENERATION,
    MIN_REVISION,
    StateCommand,
    StoreGeneration,
    canonical_json_bytes,
)

STATE_TRANSACTION_INTERFACE: Final = "StateTransaction@1"
STATE_TRANSACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/state-transaction@1"
)
STATE_TRANSACTION_VERSION: Final[int] = 1
CAS_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/cas-result@1"
)
RETRY_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/retry-policy@1"
)

DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = 8
DEFAULT_BASE_DELAY_SECONDS: Final[float] = 0.01
DEFAULT_MAX_DELAY_SECONDS: Final[float] = 0.5
DEFAULT_JITTER_RATIO: Final[float] = 0.25


class TransactionConflictKind(str, Enum):
    """Closed classification of control-plane transaction conflicts."""

    OPTIMISTIC = "optimistic_conflict"
    STALE_GENERATION = "stale_generation"
    FENCE_MISMATCH = "fence_mismatch"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    IDENTITY_MISMATCH = "identity_mismatch"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class TransactionError(RuntimeError):
    """Base class for typed state-transaction failures."""

    def __init__(
        self,
        message: str,
        *,
        kind: TransactionConflictKind = TransactionConflictKind.UNKNOWN,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind if isinstance(kind, TransactionConflictKind) else (
            TransactionConflictKind(str(kind))
        )
        self.retryable = bool(retryable)
        self.details = MappingProxyType(dict(details or {}))


class OptimisticConflictError(TransactionError):
    """Row revision CAS failed because another writer advanced the row."""

    def __init__(
        self,
        message: str = "optimistic conflict: expected revision is stale",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            kind=TransactionConflictKind.OPTIMISTIC,
            retryable=True,
            details=details,
        )


class StaleGenerationError(TransactionError):
    """Caller generation does not match the live store generation."""

    def __init__(
        self,
        message: str = "stale store generation",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            kind=TransactionConflictKind.STALE_GENERATION,
            retryable=False,
            details=details,
        )


class FenceMismatchError(TransactionError):
    """Caller fence epoch is behind the live fence."""

    def __init__(
        self,
        message: str = "fence epoch mismatch",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            kind=TransactionConflictKind.FENCE_MISMATCH,
            retryable=False,
            details=details,
        )


class IdempotencyConflictError(TransactionError):
    """Idempotency key reused with a different command payload."""

    def __init__(
        self,
        message: str = "idempotency key conflict",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            kind=TransactionConflictKind.IDEMPOTENCY_CONFLICT,
            retryable=False,
            details=details,
        )


class TransientTransactionError(TransactionError):
    """Retryable transport or lock contention failure."""

    def __init__(
        self,
        message: str = "transient transaction failure",
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            kind=TransactionConflictKind.TRANSIENT,
            retryable=True,
            details=details,
        )


class _IdempotentReplaySignal(Exception):
    """Internal signal: concurrent writer already committed this command."""

    def __init__(
        self,
        *,
        existing: Mapping[str, Any] | None = None,
        idempotency_key: str = "",
        command: StateCommand | None = None,
    ) -> None:
        super().__init__("idempotent replay required")
        self.existing = existing
        self.idempotency_key = idempotency_key
        self.command = command


@runtime_checkable
class TransactionConnection(Protocol):
    """Minimal connection surface used by StateTransaction."""

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _bounded_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_INT,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlPlaneContractError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ControlPlaneContractError(f"{name} is outside supported bounds")
    return value


def _bounded_float(
    value: Any,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: float = 3600.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlPlaneContractError(f"{name} must be a finite float") from exc
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ControlPlaneContractError(f"{name} is outside supported bounds")
    return number


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ControlPlaneContractError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise ControlPlaneIdentityError(f"{name} must not be empty")
    if "\x00" in text:
        raise ControlPlaneContractError(f"{name} must not contain NUL")
    return text


def _fetch_rows(result: Any) -> list[Any]:
    if result is None:
        return []
    fetchall = getattr(result, "fetchall", None)
    if callable(fetchall):
        rows = fetchall()
        return list(rows or [])
    if isinstance(result, list):
        return result
    return []


def _fetch_one(result: Any) -> Any | None:
    if result is None:
        return None
    fetchone = getattr(result, "fetchone", None)
    if callable(fetchone):
        return fetchone()
    rows = _fetch_rows(result)
    return rows[0] if rows else None


def _row_value(row: Any, index: int, key: str | None = None) -> Any:
    if row is None:
        return None
    if isinstance(row, Mapping):
        if key is not None and key in row:
            return row[key]
        values = list(row.values())
        return values[index] if index < len(values) else None
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return row[index] if index < len(row) else None
    getter = getattr(row, "__getitem__", None)
    if callable(getter):
        try:
            if key is not None:
                return getter(key)
        except Exception:
            pass
        return getter(index)
    return None


def result_digest(payload: Mapping[str, Any] | Sequence[Any] | str | int | bool | None) -> str:
    """Return a stable digest for an idempotent command result body."""

    if isinstance(payload, Mapping):
        body: Any = dict(payload)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        body = list(payload)
    else:
        body = payload
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return f"sha256:{digest}"


def classify_exception(exc: BaseException) -> TransactionConflictKind:
    """Map an exception into a closed conflict classification."""

    if isinstance(exc, TransactionError):
        return exc.kind
    if isinstance(exc, ControlPlaneGenerationError):
        message = str(exc).lower()
        if "fence" in message:
            return TransactionConflictKind.FENCE_MISMATCH
        return TransactionConflictKind.STALE_GENERATION
    if isinstance(exc, IdempotencyConflictError):
        return TransactionConflictKind.IDEMPOTENCY_CONFLICT
    text = str(exc).lower()
    if "revision" in text or "optimistic" in text or "conflict" in text:
        return TransactionConflictKind.OPTIMISTIC
    if "generation" in text or "stale" in text:
        return TransactionConflictKind.STALE_GENERATION
    if "fence" in text:
        return TransactionConflictKind.FENCE_MISMATCH
    if "idempoten" in text:
        return TransactionConflictKind.IDEMPOTENCY_CONFLICT
    if any(
        token in text
        for token in (
            "timeout",
            "busy",
            "locked",
            "contention",
            "connection",
            "reset",
            "broken pipe",
            "temporarily",
        )
    ):
        return TransactionConflictKind.TRANSIENT
    return TransactionConflictKind.UNKNOWN


def is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, TransactionError):
        return bool(exc.retryable)
    kind = classify_exception(exc)
    return kind in {
        TransactionConflictKind.OPTIMISTIC,
        TransactionConflictKind.TRANSIENT,
    }


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with full jitter.

    Interface fragment of ``StateTransaction@1``.
    """

    SCHEMA: ClassVar[str] = RETRY_POLICY_SCHEMA

    max_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS
    jitter_ratio: float = DEFAULT_JITTER_RATIO
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_attempts",
            _bounded_int(self.max_attempts, "max_attempts", minimum=1, maximum=128),
        )
        object.__setattr__(
            self,
            "base_delay_seconds",
            _bounded_float(self.base_delay_seconds, "base_delay_seconds"),
        )
        object.__setattr__(
            self,
            "max_delay_seconds",
            _bounded_float(self.max_delay_seconds, "max_delay_seconds"),
        )
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ControlPlaneContractError(
                "max_delay_seconds must be >= base_delay_seconds"
            )
        object.__setattr__(
            self,
            "jitter_ratio",
            _bounded_float(self.jitter_ratio, "jitter_ratio", maximum=1.0),
        )
        if self.seed is not None:
            object.__setattr__(
                self,
                "seed",
                _bounded_int(self.seed, "seed", minimum=0),
            )

    def delay_for_attempt(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Return sleep seconds before the next attempt (0-based attempt index)."""

        index = _bounded_int(attempt, "attempt", minimum=0, maximum=256)
        if index <= 0:
            return 0.0
        expo = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** (index - 1)),
        )
        if self.jitter_ratio <= 0.0:
            return expo
        generator = rng if rng is not None else random.Random(self.seed)
        # Full jitter in [expo * (1 - ratio), expo].
        low = expo * (1.0 - self.jitter_ratio)
        return low + generator.random() * (expo - low)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "max_attempts": self.max_attempts,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "jitter_ratio": self.jitter_ratio,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class CASResult:
    """Outcome of one compare-and-swap or idempotent command execution."""

    SCHEMA: ClassVar[str] = CAS_RESULT_SCHEMA

    outcome: CommandOutcome
    changed: bool
    revision: int
    generation: int
    fence_epoch: int
    result: Mapping[str, Any] = field(default_factory=dict)
    conflict_kind: TransactionConflictKind | None = None
    attempts: int = 1
    idempotency_key: str = ""
    command_id: str = ""
    result_digest: str = ""

    def __post_init__(self) -> None:
        outcome = self.outcome
        if not isinstance(outcome, CommandOutcome):
            outcome = CommandOutcome(str(outcome))
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "changed", bool(self.changed))
        object.__setattr__(
            self,
            "revision",
            _bounded_int(self.revision, "revision", minimum=MIN_REVISION),
        )
        object.__setattr__(
            self,
            "generation",
            _bounded_int(self.generation, "generation", minimum=MIN_GENERATION),
        )
        object.__setattr__(
            self,
            "fence_epoch",
            _bounded_int(self.fence_epoch, "fence_epoch", minimum=MIN_FENCE_EPOCH),
        )
        object.__setattr__(
            self,
            "result",
            MappingProxyType(dict(self.result or {})),
        )
        if self.conflict_kind is not None and not isinstance(
            self.conflict_kind, TransactionConflictKind
        ):
            object.__setattr__(
                self,
                "conflict_kind",
                TransactionConflictKind(str(self.conflict_kind)),
            )
        object.__setattr__(
            self,
            "attempts",
            _bounded_int(self.attempts, "attempts", minimum=1, maximum=256),
        )
        object.__setattr__(
            self, "idempotency_key", _text(self.idempotency_key, "idempotency_key", required=False)
        )
        object.__setattr__(
            self, "command_id", _text(self.command_id, "command_id", required=False)
        )
        object.__setattr__(
            self,
            "result_digest",
            _text(self.result_digest, "result_digest", required=False),
        )

    @property
    def accepted(self) -> bool:
        return self.outcome in {
            CommandOutcome.ACCEPTED,
            CommandOutcome.IDEMPOTENT_REPLAY,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": STATE_TRANSACTION_INTERFACE,
            "outcome": self.outcome.value,
            "changed": self.changed,
            "revision": self.revision,
            "generation": self.generation,
            "fence_epoch": self.fence_epoch,
            "result": dict(self.result),
            "conflict_kind": (
                None if self.conflict_kind is None else self.conflict_kind.value
            ),
            "attempts": self.attempts,
            "idempotency_key": self.idempotency_key,
            "command_id": self.command_id,
            "result_digest": self.result_digest,
        }


def default_retry_policy(
    bounds: ControlPlaneBounds | None = None,
    *,
    seed: int | None = None,
) -> RetryPolicy:
    max_attempts = (
        bounds.max_conflict_retries if bounds is not None else DEFAULT_MAX_RETRY_ATTEMPTS
    )
    return RetryPolicy(max_attempts=max_attempts, seed=seed)


def run_with_retry(
    operation: Callable[[int], CASResult],
    *,
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> CASResult:
    """Execute ``operation(attempt)`` with jittered retry on retryable failures.

    ``operation`` should raise ``TransactionError`` (or compatible) for
    conflicts, or return a ``CASResult`` with a conflict outcome.
    """

    selected = policy or default_retry_policy()
    generator = rng if rng is not None else random.Random(selected.seed)
    last_error: BaseException | None = None
    last_result: CASResult | None = None
    for attempt in range(1, selected.max_attempts + 1):
        try:
            result = operation(attempt)
        except BaseException as exc:
            last_error = exc
            if not is_retryable_exception(exc) or attempt >= selected.max_attempts:
                if isinstance(exc, TransactionError):
                    raise
                raise TransactionError(
                    str(exc),
                    kind=classify_exception(exc),
                    retryable=is_retryable_exception(exc),
                ) from exc
            delay = selected.delay_for_attempt(attempt, rng=generator)
            if delay > 0:
                sleep(delay)
            continue
        if not isinstance(result, CASResult):
            raise TransactionError(
                "retry operation must return CASResult",
                kind=TransactionConflictKind.UNKNOWN,
                retryable=False,
            )
        last_result = result
        if result.accepted:
            return result
        if (
            result.conflict_kind
            in {
                TransactionConflictKind.OPTIMISTIC,
                TransactionConflictKind.TRANSIENT,
            }
            and attempt < selected.max_attempts
        ):
            delay = selected.delay_for_attempt(attempt, rng=generator)
            if delay > 0:
                sleep(delay)
            continue
        return result
    if last_result is not None:
        return last_result
    if last_error is not None:
        raise last_error
    raise TransactionError(
        "retry budget exhausted without a result",
        kind=TransactionConflictKind.TRANSIENT,
        retryable=False,
    )


class StateTransaction:
    """Short-lived CAS transaction against a control-plane connection.

    Interface: ``StateTransaction@1``.
    """

    INTERFACE: ClassVar[str] = STATE_TRANSACTION_INTERFACE
    SCHEMA: ClassVar[str] = STATE_TRANSACTION_SCHEMA

    def __init__(
        self,
        connection: TransactionConnection,
        *,
        store_id: str,
        expected_generation: StoreGeneration | None = None,
        session_id: str = "",
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], float] | None = None,
        now_iso: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(connection, TransactionConnection) and not hasattr(
            connection, "execute"
        ):
            raise ControlPlaneContractError(
                "connection must provide execute/commit/rollback"
            )
        self._connection = connection
        self.store_id = _text(store_id, "store_id")
        self.session_id = _text(session_id, "session_id", required=False)
        self.expected_generation = expected_generation
        self.retry_policy = retry_policy or default_retry_policy()
        self._clock = clock or time.monotonic
        self._now_iso = now_iso or (
            lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )
        self._active = False
        self._committed = False
        self._rolled_back = False

    @property
    def active(self) -> bool:
        return self._active and not self._committed and not self._rolled_back

    def begin(self) -> "StateTransaction":
        if self._active:
            raise TransactionError(
                "transaction already active",
                kind=TransactionConflictKind.UNKNOWN,
                retryable=False,
            )
        self._connection.execute("BEGIN TRANSACTION")
        self._active = True
        self._committed = False
        self._rolled_back = False
        return self

    def commit(self) -> None:
        if not self._active:
            raise TransactionError(
                "no active transaction to commit",
                kind=TransactionConflictKind.UNKNOWN,
                retryable=False,
            )
        try:
            self._connection.commit()
        except Exception as exc:
            # Adapters without a working commit() can still accept SQL COMMIT.
            try:
                self._connection.execute("COMMIT")
            except Exception as sql_exc:
                self._active = False
                self._rolled_back = True
                raise TransientTransactionError(
                    f"failed to commit transaction: {exc}; {sql_exc}",
                    details={"error": str(exc), "sql_error": str(sql_exc)},
                ) from sql_exc
        self._active = False
        self._committed = True

    def rollback(self) -> None:
        if not self._active:
            return
        try:
            try:
                self._connection.rollback()
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except Exception:
                    pass
        finally:
            self._active = False
            self._rolled_back = True

    def __enter__(self) -> "StateTransaction":
        return self.begin()

    def __exit__(self, exc_type: Any, _exc: Any, _tb: Any) -> None:
        if exc_type is None and self._active:
            self.commit()
        else:
            self.rollback()

    def load_generation(self) -> StoreGeneration:
        """Load the latest store generation row for this store."""

        result = self._connection.execute(
            """
            SELECT generation, schema_revision, fence_epoch, revision,
                   database_uuid, birth_id
            FROM store_generations
            ORDER BY generation DESC
            LIMIT 1
            """
        )
        row = _fetch_one(result)
        if row is None:
            raise StaleGenerationError(
                "store generation row is missing",
                details={"store_id": self.store_id},
            )
        return StoreGeneration(
            store_id=self.store_id,
            generation=int(_row_value(row, 0, "generation")),
            schema_revision=int(_row_value(row, 1, "schema_revision")),
            fence_epoch=int(_row_value(row, 2, "fence_epoch")),
            revision=int(_row_value(row, 3, "revision")),
            database_uuid=str(_row_value(row, 4, "database_uuid")),
            birth_id=str(_row_value(row, 5, "birth_id") or ""),
        )

    def assert_expected_generation(
        self,
        generation: StoreGeneration | None = None,
    ) -> StoreGeneration:
        """Fail closed when the caller's expected generation is stale."""

        live = generation if generation is not None else self.load_generation()
        expected = self.expected_generation
        if expected is None:
            return live
        if live.store_id != expected.store_id:
            raise StaleGenerationError(
                "store_id mismatch",
                details={
                    "expected_store_id": expected.store_id,
                    "live_store_id": live.store_id,
                },
            )
        if live.database_uuid != expected.database_uuid:
            raise StaleGenerationError(
                "database_uuid mismatch",
                details={
                    "expected_database_uuid": expected.database_uuid,
                    "live_database_uuid": live.database_uuid,
                },
            )
        if live.generation != expected.generation:
            raise StaleGenerationError(
                "store generation mismatch",
                details={
                    "expected_generation": expected.generation,
                    "live_generation": live.generation,
                },
            )
        if live.fence_epoch < expected.fence_epoch:
            raise FenceMismatchError(
                "live fence epoch is behind expected fence",
                details={
                    "expected_fence_epoch": expected.fence_epoch,
                    "live_fence_epoch": live.fence_epoch,
                },
            )
        if live.fence_epoch > expected.fence_epoch:
            raise FenceMismatchError(
                "caller fence epoch is stale",
                details={
                    "expected_fence_epoch": expected.fence_epoch,
                    "live_fence_epoch": live.fence_epoch,
                },
            )
        if live.revision < expected.revision:
            # Live should never move backwards; treat as generation drift.
            raise StaleGenerationError(
                "live revision moved backwards relative to expected",
                details={
                    "expected_revision": expected.revision,
                    "live_revision": live.revision,
                },
            )
        return live

    def lookup_idempotency(
        self,
        idempotency_key: str,
    ) -> Mapping[str, Any] | None:
        key = _text(idempotency_key, "idempotency_key")
        result = self._connection.execute(
            """
            SELECT idempotency_key, command_kind, command_id, store_id,
                   session_id, result_digest, created_at, expires_at, body_json
            FROM idempotency_records
            WHERE idempotency_key = ?
            LIMIT 1
            """,
            [key],
        )
        row = _fetch_one(result)
        if row is None:
            return None
        body_raw = _row_value(row, 8, "body_json")
        body: Any = {}
        if isinstance(body_raw, str) and body_raw:
            try:
                body = json.loads(body_raw)
            except json.JSONDecodeError as exc:
                raise TransactionError(
                    "idempotency body_json is corrupt",
                    kind=TransactionConflictKind.UNKNOWN,
                    retryable=False,
                ) from exc
        return {
            "idempotency_key": str(_row_value(row, 0, "idempotency_key")),
            "command_kind": str(_row_value(row, 1, "command_kind")),
            "command_id": str(_row_value(row, 2, "command_id")),
            "store_id": str(_row_value(row, 3, "store_id")),
            "session_id": str(_row_value(row, 4, "session_id")),
            "result_digest": str(_row_value(row, 5, "result_digest")),
            "created_at": str(_row_value(row, 6, "created_at") or ""),
            "expires_at": _row_value(row, 7, "expires_at"),
            "body": body if isinstance(body, Mapping) else {"value": body},
        }

    def record_idempotency(
        self,
        *,
        command: StateCommand,
        result_body: Mapping[str, Any],
        result_digest_value: str | None = None,
        expires_at: str | None = None,
    ) -> str:
        digest = result_digest_value or result_digest(result_body)
        body_json = json.dumps(
            dict(result_body),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        try:
            self._connection.execute(
                """
                INSERT INTO idempotency_records (
                    idempotency_key, command_kind, command_id, store_id,
                    session_id, result_digest, created_at, expires_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    command.idempotency_key,
                    command.command_kind.value
                    if isinstance(command.command_kind, CommandKind)
                    else str(command.command_kind),
                    command.command_id,
                    command.store_id,
                    command.session_id,
                    digest,
                    self._now_iso(),
                    expires_at,
                    body_json,
                ],
            )
        except Exception as exc:
            # Concurrent first-writers can race past the pre-insert lookup.
            # The current transaction may be aborted after a constraint error,
            # so outer handling must rollback before re-reading the winner.
            if not _looks_like_unique_violation(exc):
                raise
            raise _IdempotentReplaySignal(
                idempotency_key=command.idempotency_key,
                command=command,
            ) from exc
        return digest

    def advance_store_revision(
        self,
        live: StoreGeneration,
        *,
        next_revision: int | None = None,
    ) -> StoreGeneration:
        """Advance the store generation head revision after a successful CAS."""

        new_revision = (
            live.revision + 1 if next_revision is None else int(next_revision)
        )
        if new_revision <= live.revision:
            raise OptimisticConflictError(
                "next store revision must advance",
                details={
                    "live_revision": live.revision,
                    "next_revision": new_revision,
                },
            )
        self._connection.execute(
            """
            UPDATE store_generations
            SET revision = ?
            WHERE generation = ? AND revision = ? AND fence_epoch = ?
            """,
            [new_revision, live.generation, live.revision, live.fence_epoch],
        )
        # DuckDB may not expose rowcount reliably; re-read.
        refreshed = self.load_generation()
        if refreshed.revision != new_revision:
            raise OptimisticConflictError(
                "store revision CAS lost the race",
                details={
                    "expected_revision": live.revision,
                    "attempted_revision": new_revision,
                    "live_revision": refreshed.revision,
                },
            )
        return refreshed

    def cas_row_revision(
        self,
        *,
        table: str,
        key_column: str,
        key_value: str,
        revision_column: str = "revision",
        expected_revision: int,
        assignments: Mapping[str, Any],
    ) -> int:
        """CAS-update one row and return the new revision.

        ``table``, ``key_column``, and ``revision_column`` must be caller-fixed
        constants from trusted code paths (statement templates), never
        model-supplied identifiers.
        """

        _assert_safe_sql_identifier(table, "table")
        _assert_safe_sql_identifier(key_column, "key_column")
        _assert_safe_sql_identifier(revision_column, "revision_column")
        if not assignments:
            raise ControlPlaneContractError("CAS assignments must not be empty")
        new_revision = expected_revision + 1
        set_parts: list[str] = []
        params: list[Any] = []
        for column, value in assignments.items():
            _assert_safe_sql_identifier(column, "assignment column")
            if column == revision_column:
                continue
            set_parts.append(f'"{column}" = ?')
            params.append(value)
        set_parts.append(f'"{revision_column}" = ?')
        params.append(new_revision)
        params.extend([key_value, expected_revision])
        sql = (
            f'UPDATE "{table}" SET {", ".join(set_parts)} '
            f'WHERE "{key_column}" = ? AND "{revision_column}" = ? '
            f'RETURNING "{revision_column}"'
        )
        row = _fetch_one(self._connection.execute(sql, params))
        if row is None:
            raise OptimisticConflictError(
                f"row revision CAS failed for {table}",
                details={
                    "table": table,
                    "key": key_value,
                    "expected_revision": expected_revision,
                },
            )
        returned_revision = int(_row_value(row, 0, revision_column))
        if returned_revision != new_revision:
            raise OptimisticConflictError(
                f"row revision CAS returned an unexpected revision for {table}",
                details={
                    "table": table,
                    "key": key_value,
                    "expected_revision": expected_revision,
                    "returned_revision": returned_revision,
                },
            )
        return new_revision

    def execute_command(
        self,
        command: StateCommand,
        *,
        apply: Callable[["StateTransaction", StateCommand, StoreGeneration], Mapping[str, Any]],
        auto_commit: bool = True,
    ) -> CASResult:
        """Execute one fenced, idempotent command under generation checks.

        ``apply`` performs the domain mutation and returns the result body.
        Idempotent replay short-circuits before ``apply`` when the stored
        command identity matches.
        """

        if not isinstance(command, StateCommand):
            raise ControlPlaneContractError("command must be a StateCommand")
        owns_txn = False
        if not self._active:
            self.begin()
            owns_txn = True
        try:
            live = self.assert_expected_generation()
            if (
                command.expected_generation != live.generation
                or command.fence_epoch != live.fence_epoch
            ):
                # Prefer specific typed failures over a generic generation error.
                if command.expected_generation != live.generation:
                    raise StaleGenerationError(
                        "command expected_generation is stale",
                        details={
                            "expected_generation": command.expected_generation,
                            "live_generation": live.generation,
                        },
                    )
                raise FenceMismatchError(
                    "command fence_epoch is stale",
                    details={
                        "expected_fence_epoch": command.fence_epoch,
                        "live_fence_epoch": live.fence_epoch,
                    },
                )
            if command.store_id != self.store_id:
                raise TransactionError(
                    "command store_id does not match transaction store",
                    kind=TransactionConflictKind.IDENTITY_MISMATCH,
                    retryable=False,
                    details={
                        "command_store_id": command.store_id,
                        "transaction_store_id": self.store_id,
                    },
                )

            existing = self.lookup_idempotency(command.idempotency_key)
            if existing is not None:
                same_command = (
                    existing["command_id"] == command.command_id
                    and existing["command_kind"]
                    == (
                        command.command_kind.value
                        if isinstance(command.command_kind, CommandKind)
                        else str(command.command_kind)
                    )
                    and existing["store_id"] == command.store_id
                )
                if not same_command:
                    raise IdempotencyConflictError(
                        "idempotency key already bound to a different command",
                        details={
                            "idempotency_key": command.idempotency_key,
                            "existing_command_id": existing["command_id"],
                            "command_id": command.command_id,
                        },
                    )
                body = existing.get("body") or {}
                if not isinstance(body, Mapping):
                    body = {"value": body}
                if auto_commit and owns_txn:
                    self.commit()
                return CASResult(
                    outcome=CommandOutcome.IDEMPOTENT_REPLAY,
                    changed=False,
                    revision=live.revision,
                    generation=live.generation,
                    fence_epoch=live.fence_epoch,
                    result=dict(body),
                    attempts=1,
                    idempotency_key=command.idempotency_key,
                    command_id=command.command_id,
                    result_digest=str(existing.get("result_digest") or ""),
                )

            if command.expected_revision != live.revision:
                raise OptimisticConflictError(
                    "command expected_revision is stale",
                    details={
                        "expected_revision": command.expected_revision,
                        "live_revision": live.revision,
                    },
                )

            body = apply(self, command, live)
            if not isinstance(body, Mapping):
                raise ControlPlaneContractError("apply() must return a mapping body")
            advanced = self.advance_store_revision(live)
            digest = self.record_idempotency(command=command, result_body=body)
            if auto_commit and owns_txn:
                self.commit()
            return CASResult(
                outcome=CommandOutcome.ACCEPTED,
                changed=True,
                revision=advanced.revision,
                generation=advanced.generation,
                fence_epoch=advanced.fence_epoch,
                result=dict(body),
                attempts=1,
                idempotency_key=command.idempotency_key,
                command_id=command.command_id,
                result_digest=digest,
            )
        except _IdempotentReplaySignal as signal:
            # Another writer committed first; drop local mutations then re-read.
            if self._active:
                self.rollback()
            key = signal.idempotency_key or command.idempotency_key
            existing = signal.existing or self.lookup_idempotency(key)
            if existing is None:
                raise IdempotencyConflictError(
                    "idempotency insert raced and no record is visible",
                    details={"idempotency_key": key},
                ) from signal
            same_command = (
                existing["command_id"] == command.command_id
                and existing["command_kind"]
                == (
                    command.command_kind.value
                    if isinstance(command.command_kind, CommandKind)
                    else str(command.command_kind)
                )
                and existing["store_id"] == command.store_id
            )
            if not same_command:
                raise IdempotencyConflictError(
                    "idempotency key already bound to a different command",
                    details={
                        "idempotency_key": key,
                        "existing_command_id": existing["command_id"],
                        "command_id": command.command_id,
                    },
                ) from signal
            body = existing.get("body") or {}
            if not isinstance(body, Mapping):
                body = {"value": body}
            live_after = self.load_generation()
            return CASResult(
                outcome=CommandOutcome.IDEMPOTENT_REPLAY,
                changed=False,
                revision=live_after.revision,
                generation=live_after.generation,
                fence_epoch=live_after.fence_epoch,
                result=dict(body),
                attempts=1,
                idempotency_key=command.idempotency_key,
                command_id=command.command_id,
                result_digest=str(existing.get("result_digest") or ""),
            )
        except Exception:
            if owns_txn and self._active:
                self.rollback()
            raise

    def execute_command_with_retry(
        self,
        command: StateCommand,
        *,
        apply: Callable[["StateTransaction", StateCommand, StoreGeneration], Mapping[str, Any]],
        refresh_command: Callable[[StateCommand, StoreGeneration], StateCommand] | None = None,
        load_live: Callable[[], StoreGeneration] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> CASResult:
        """Retry an optimistic CAS command with optional command refresh."""

        def _operation(attempt: int) -> CASResult:
            # Each attempt uses a fresh transaction boundary.
            txn = StateTransaction(
                self._connection,
                store_id=self.store_id,
                expected_generation=self.expected_generation,
                session_id=self.session_id,
                retry_policy=self.retry_policy,
                clock=self._clock,
                now_iso=self._now_iso,
            )
            live_loader = load_live or txn.load_generation
            active_command = command
            if attempt > 1 and refresh_command is not None:
                live = live_loader()
                active_command = refresh_command(command, live)
                txn.expected_generation = StoreGeneration(
                    store_id=live.store_id,
                    generation=live.generation,
                    schema_revision=live.schema_revision,
                    fence_epoch=live.fence_epoch,
                    revision=live.revision,
                    database_uuid=live.database_uuid,
                    birth_id=live.birth_id,
                )
            result = txn.execute_command(active_command, apply=apply)
            return CASResult(
                outcome=result.outcome,
                changed=result.changed,
                revision=result.revision,
                generation=result.generation,
                fence_epoch=result.fence_epoch,
                result=dict(result.result),
                conflict_kind=result.conflict_kind,
                attempts=attempt,
                idempotency_key=result.idempotency_key,
                command_id=result.command_id,
                result_digest=result.result_digest,
            )

        return run_with_retry(
            _operation,
            policy=self.retry_policy,
            sleep=sleep,
            rng=rng,
        )


_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _looks_like_unique_violation(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "unique",
            "duplicate",
            "constraint",
            "primary key",
            "already exists",
        )
    )


def _assert_safe_sql_identifier(value: str, name: str) -> str:
    text = _text(value, name)
    if not _SAFE_IDENTIFIER_RE.fullmatch(text):
        raise ControlPlaneIdentityError(
            f"{name} is not a safe SQL identifier: {text!r}"
        )
    if text.upper() in {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "ATTACH",
        "DETACH",
        "COPY",
        "PRAGMA",
        "CALL",
    }:
        raise ControlPlaneIdentityError(f"{name} collides with a SQL keyword")
    return text


__all__ = [
    "CASResult",
    "FenceMismatchError",
    "IdempotencyConflictError",
    "OptimisticConflictError",
    "RetryPolicy",
    "STATE_TRANSACTION_INTERFACE",
    "STATE_TRANSACTION_SCHEMA",
    "StateTransaction",
    "StaleGenerationError",
    "TransactionConflictKind",
    "TransactionConnection",
    "TransactionError",
    "TransientTransactionError",
    "classify_exception",
    "default_retry_policy",
    "is_retryable_exception",
    "result_digest",
    "run_with_retry",
]
