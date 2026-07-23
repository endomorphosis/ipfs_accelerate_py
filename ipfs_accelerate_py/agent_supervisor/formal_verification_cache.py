"""Trust-aware, process-safe cache for formal-verification receipts.

This cache is deliberately not a trust root.  Every hit is reconstructed as a
typed :class:`ProofReceipt`, its assurance is derived again from immutable
evidence, and all semantic/execution bindings are checked against the cache
key.  Provider status text and claimed assurance are never used to admit an
entry.

SQLite supplies the small transactional boundary needed by parallel
supervisors.  In addition to durable proof receipts, the database contains a
short-lived single-flight lease and outcome channel.  The latter lets threads
and processes share one active expensive computation without treating its
possibly non-authoritative result as a proof-cache hit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeVar

from .formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    ContractValidationError,
    EvidenceFreshness,
    EvidenceKind,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    assurance_satisfies,
    canonical_json,
)


FORMAL_VERIFICATION_CACHE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-verification-cache-entry@1"
)
FORMAL_VERIFICATION_CACHE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-verification-cache-key@1"
)
FORMAL_VERIFICATION_FLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/formal-verification-flight@1"
)
DEFAULT_CACHE_TTL_SECONDS: Final = 24 * 60 * 60
DEFAULT_LEASE_SECONDS: Final = 5 * 60
DEFAULT_WAIT_TIMEOUT_SECONDS: Final = 10 * 60


class CacheLookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"


class CacheRejectionReason(str, Enum):
    """Stable reason codes suitable for audit records and metrics."""

    CACHE_MISS = "cache_miss"
    MALFORMED_ENTRY = "malformed_cache_entry"
    POISONED_ENTRY = "poisoned_cache_entry"
    STALE_ENTRY = "stale_cache_entry"
    PARTIAL_ENTRY = "partial_cache_entry"
    SOLVER_ONLY_ENTRY = "solver_only_cache_entry"
    SIMULATED_ATTESTATION = "simulated_attestation"
    INSUFFICIENT_ASSURANCE = "required_assurance_not_satisfied"
    FRESHNESS_NOT_SATISFIED = "freshness_requirement_not_satisfied"
    BINDING_MISMATCH = "cache_binding_mismatch"

    # Short compatibility spellings.
    MALFORMED = "malformed_cache_entry"
    POISONED = "poisoned_cache_entry"
    STALE = "stale_cache_entry"
    PARTIAL = "partial_cache_entry"
    SOLVER_ONLY = "solver_only_cache_entry"


class SingleFlightError(RuntimeError):
    """Base class for single-flight coordination failures."""


class SingleFlightTimeout(SingleFlightError, TimeoutError):
    """Raised when a follower cannot observe an outcome before its deadline."""


class SingleFlightExecutionError(SingleFlightError):
    """A leader failed while producing the shared outcome."""


def _json_value(value: Any, *, field_name: str) -> Any:
    """Round-trip a value through the proof contract's canonical JSON subset."""

    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError, ContractValidationError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must contain canonical JSON values") from exc


def _identity_component(value: Any, *, field_name: str) -> Any:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{field_name} must not be empty")
    return _json_value(value, field_name=field_name)


def _premises(value: Any) -> tuple[Any, ...]:
    if value is None:
        raise ValueError("premises is required (use an empty array when there are none)")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("premises must be an array")
    normalized = [_json_value(item, field_name="premises") for item in value]

    def sort_key(item: Any) -> bytes:
        return canonical_json(item).encode("utf-8")

    return tuple(sorted(normalized, key=sort_key))


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProofCacheKey:
    """Content-addressed identity of every input capable of changing a proof."""

    obligation: Any
    premises: tuple[Any, ...]
    translator: Any
    solver: Any
    kernel: Any
    toolchain: Any
    theorem_registry: Any
    policy: Any
    resource_budget: Any
    candidate_tree: Any

    def __post_init__(self) -> None:
        for name in (
            "obligation",
            "translator",
            "solver",
            "kernel",
            "toolchain",
            "theorem_registry",
            "policy",
            "resource_budget",
            "candidate_tree",
        ):
            object.__setattr__(
                self,
                name,
                _identity_component(getattr(self, name), field_name=name),
            )
        object.__setattr__(self, "premises", _premises(self.premises))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FORMAL_VERIFICATION_CACHE_KEY_SCHEMA,
            "obligation": self.obligation,
            "premises": list(self.premises),
            "translator": self.translator,
            "solver": self.solver,
            "kernel": self.kernel,
            "toolchain": self.toolchain,
            "theorem_registry": self.theorem_registry,
            "policy": self.policy,
            "resource_budget": self.resource_budget,
            "candidate_tree": self.candidate_tree,
        }

    @property
    def key_id(self) -> str:
        return f"proof-cache-key:sha256:{_sha256(self.to_dict())}"

    @property
    def cache_key(self) -> str:
        return self.key_id

    @property
    def digest(self) -> str:
        return self.key_id

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofCacheKey":
        if not isinstance(value, Mapping):
            raise ValueError("cache key must be an object")
        schema = value.get("schema_version")
        if schema not in (None, FORMAL_VERIFICATION_CACHE_KEY_SCHEMA):
            raise ValueError("unsupported formal-verification cache-key schema")
        return cls(
            obligation=value.get("obligation"),
            premises=tuple(value.get("premises") or ()),
            translator=value.get("translator"),
            solver=value.get("solver"),
            kernel=value.get("kernel"),
            toolchain=value.get("toolchain"),
            theorem_registry=value.get("theorem_registry"),
            policy=value.get("policy"),
            resource_budget=value.get("resource_budget"),
            candidate_tree=value.get("candidate_tree"),
        )


def build_proof_cache_key(
    *,
    obligation: Any,
    premises: Sequence[Any],
    translator: Any,
    solver: Any,
    kernel: Any,
    toolchain: Any,
    theorem_registry: Any,
    policy: Any,
    resource_budget: Any,
    candidate_tree: Any | None = None,
    candidate_tree_id: Any | None = None,
) -> ProofCacheKey:
    """Build a cache key while accepting the common ``*_tree_id`` spelling."""

    if candidate_tree is not None and candidate_tree_id is not None:
        if _json_value(candidate_tree, field_name="candidate_tree") != _json_value(
            candidate_tree_id, field_name="candidate_tree_id"
        ):
            raise ValueError("candidate_tree and candidate_tree_id disagree")
    tree = candidate_tree if candidate_tree is not None else candidate_tree_id
    return ProofCacheKey(
        obligation=obligation,
        premises=tuple(premises),
        translator=translator,
        solver=solver,
        kernel=kernel,
        toolchain=toolchain,
        theorem_registry=theorem_registry,
        policy=policy,
        resource_budget=resource_budget,
        candidate_tree=tree,
    )


# Descriptive compatibility spelling.
make_proof_cache_key = build_proof_cache_key
FormalVerificationCacheKey = ProofCacheKey


@dataclass(frozen=True)
class CacheRequirements:
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED
    required_freshness: EvidenceFreshness = EvidenceFreshness.CURRENT
    max_age_seconds: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required_assurance", AssuranceLevel(self.required_assurance)
        )
        object.__setattr__(
            self, "required_freshness", EvidenceFreshness(self.required_freshness)
        )
        if self.max_age_seconds is not None and (
            isinstance(self.max_age_seconds, bool)
            or not isinstance(self.max_age_seconds, int)
            or self.max_age_seconds < 0
        ):
            raise ValueError("max_age_seconds must be a nonnegative integer or None")


@dataclass(frozen=True)
class ProofCacheEntry:
    key: ProofCacheKey
    receipt: ProofReceipt
    created_at_ms: int
    expires_at_ms: int
    complete: bool = True
    entry_digest: str = ""

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FORMAL_VERIFICATION_CACHE_SCHEMA,
            "key_id": self.key.key_id,
            "key": self.key.to_dict(),
            "receipt_id": self.receipt.receipt_id,
            "receipt": self.receipt.to_dict(),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "complete": self.complete,
        }

    @property
    def computed_digest(self) -> str:
        return f"sha256:{_sha256(self._unsigned_dict())}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_dict(),
            "entry_digest": self.entry_digest or self.computed_digest,
        }

    @classmethod
    def create(
        cls,
        key: ProofCacheKey,
        receipt: ProofReceipt,
        *,
        created_at_ms: int,
        expires_at_ms: int,
        complete: bool = True,
    ) -> "ProofCacheEntry":
        entry = cls(
            key=key,
            receipt=receipt,
            created_at_ms=created_at_ms,
            expires_at_ms=expires_at_ms,
            complete=complete,
        )
        return cls(**{**entry.__dict__, "entry_digest": entry.computed_digest})


@dataclass(frozen=True)
class CacheLookupResult:
    status: CacheLookupStatus
    key: ProofCacheKey
    entry: ProofCacheEntry | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.status is CacheLookupStatus.HIT

    @property
    def receipt(self) -> ProofReceipt | None:
        return self.entry.receipt if self.entry is not None and self.hit else None

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""


@dataclass(frozen=True)
class CacheStoreResult:
    stored: bool
    key: ProofCacheKey
    entry: ProofCacheEntry | None = None
    reason_codes: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.stored


@dataclass(frozen=True)
class SingleFlightLease:
    """A fenced lease returned for both leaders and followers."""

    cache: "FormalVerificationCache" = field(repr=False, compare=False)
    key: ProofCacheKey
    owner_id: str
    token: str
    fencing_token: int
    acquired_at_ms: int
    expires_at_ms: int
    acquired: bool

    @property
    def is_leader(self) -> bool:
        return self.acquired

    def renew(self, *, lease_seconds: int = DEFAULT_LEASE_SECONDS) -> "SingleFlightLease":
        return self.cache.renew_lease(self, lease_seconds=lease_seconds)

    def release(self) -> bool:
        return self.cache.release_lease(self)

    def __enter__(self) -> "SingleFlightLease":
        if not self.acquired:
            raise SingleFlightError("cannot enter a single-flight lease not owned by caller")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def _strict_json_loads(value: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=pairs)


def _component_identifier(value: Any, *names: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate:
                return candidate
    return ""


def _obligation_identifier(value: Any) -> str:
    direct = _component_identifier(value, "obligation_id", "content_id", "id")
    if direct:
        return direct
    if isinstance(value, Mapping):
        try:
            return CodeProofObligation.from_dict(value).obligation_id
        except (TypeError, ValueError, ContractValidationError):
            return ""
    return ""


def _premise_identifiers(values: Sequence[Any]) -> tuple[str, ...]:
    identifiers = []
    for value in values:
        identifier = _component_identifier(
            value, "premise_id", "content_id", "id"
        )
        if not identifier and isinstance(value, str):
            identifier = value
        if identifier:
            identifiers.append(identifier)
    return tuple(sorted(set(identifiers)))


def _binding_reasons(key: ProofCacheKey, receipt: ProofReceipt) -> set[str]:
    reasons: set[str] = set()

    expected_obligation = _obligation_identifier(key.obligation)
    if expected_obligation and expected_obligation != receipt.obligation_id:
        reasons.add(CacheRejectionReason.BINDING_MISMATCH.value)

    expected_premises = _premise_identifiers(key.premises)
    if expected_premises and expected_premises != tuple(sorted(receipt.premise_ids)):
        reasons.add(CacheRejectionReason.BINDING_MISMATCH.value)

    bindings = (
        (key.translator, receipt.translator_id, ("translator_id", "id")),
        (key.kernel, receipt.kernel_id, ("kernel_id", "id")),
        (key.toolchain, receipt.toolchain_id, ("toolchain_id", "id")),
        (
            key.theorem_registry,
            receipt.theorem_registry_id,
            ("theorem_registry_id", "registry_id", "id"),
        ),
        (key.policy, receipt.policy_id, ("policy_id", "id")),
        (
            key.candidate_tree,
            receipt.repository_tree_id,
            ("candidate_tree_id", "repository_tree_id", "tree_id", "id"),
        ),
    )
    for component, actual, names in bindings:
        expected = _component_identifier(component, *names)
        if expected and expected != actual:
            reasons.add(CacheRejectionReason.BINDING_MISMATCH.value)

    solver_ids: tuple[str, ...] = ()
    if isinstance(key.solver, Mapping):
        raw_ids = key.solver.get("solver_ids") or key.solver.get("allowed_solvers")
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, str):
            solver_ids = tuple(str(item) for item in raw_ids)
    expected_solver = _component_identifier(key.solver, "solver_id", "id")
    if expected_solver and expected_solver != receipt.solver_id:
        reasons.add(CacheRejectionReason.BINDING_MISMATCH.value)
    elif solver_ids and receipt.solver_id not in solver_ids:
        reasons.add(CacheRejectionReason.BINDING_MISMATCH.value)

    try:
        expected_budget = ResourceBudget.from_dict(key.resource_budget).to_dict()
    except (TypeError, ValueError, ContractValidationError):
        expected_budget = key.resource_budget
    if expected_budget != receipt.resource_budget.to_dict():
        reasons.add(CacheRejectionReason.BINDING_MISMATCH.value)
    return reasons


def _trust_reasons(
    key: ProofCacheKey,
    receipt: ProofReceipt,
    *,
    complete: bool,
    requirements: CacheRequirements,
) -> set[str]:
    reasons = _binding_reasons(key, receipt)
    if not complete or receipt.verdict is not ProofVerdict.PROVED:
        reasons.add(CacheRejectionReason.PARTIAL_ENTRY.value)
    if receipt.metadata.get("partial") is True:
        reasons.add(CacheRejectionReason.PARTIAL_ENTRY.value)

    simulated_attestation = any(
        evidence.kind is EvidenceKind.CRYPTOGRAPHIC_ATTESTATION
        and evidence.simulated
        for evidence in receipt.evidence
    )
    if simulated_attestation:
        reasons.add(CacheRejectionReason.SIMULATED_ATTESTATION.value)

    actual = receipt.authoritative_assurance
    if actual.rank <= AssuranceLevel.SOLVER_CHECKED.rank:
        reasons.add(CacheRejectionReason.SOLVER_ONLY_ENTRY.value)
    if not assurance_satisfies(actual, requirements.required_assurance):
        reasons.add(CacheRejectionReason.INSUFFICIENT_ASSURANCE.value)
    if (
        requirements.required_freshness is EvidenceFreshness.CURRENT
        and receipt.freshness is not EvidenceFreshness.CURRENT
    ):
        reasons.add(CacheRejectionReason.FRESHNESS_NOT_SATISFIED.value)
    return reasons


T = TypeVar("T")


class FormalVerificationCache:
    """Durable receipt cache and cross-process single-flight coordinator."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        default_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        sqlite_timeout_seconds: int = 30,
    ) -> None:
        if isinstance(default_ttl_seconds, bool) or default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be a positive integer")
        if sqlite_timeout_seconds <= 0:
            raise ValueError("sqlite_timeout_seconds must be positive")
        if path is None:
            root = Path(
                tempfile.mkdtemp(prefix="formal-verification-cache-")
            )
            self.path = root / "cache.sqlite3"
        else:
            supplied = Path(path)
            self.path = (
                supplied
                if supplied.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
                else supplied / "formal_verification_cache.sqlite3"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.default_ttl_seconds = int(default_ttl_seconds)
        self._clock = clock
        self._sqlite_timeout_seconds = sqlite_timeout_seconds
        self._schema_lock = threading.Lock()
        self._initialize()

    @property
    def db_path(self) -> Path:
        return self.path

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=self._sqlite_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self._sqlite_timeout_seconds * 1000}")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._schema_lock:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS proof_cache_entries (
                        key_id TEXT PRIMARY KEY,
                        key_json TEXT NOT NULL,
                        entry_json TEXT NOT NULL,
                        created_at_ms INTEGER NOT NULL,
                        expires_at_ms INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS proof_cache_expiry_idx
                        ON proof_cache_entries(expires_at_ms);
                    CREATE TABLE IF NOT EXISTS proof_single_flights (
                        key_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        token TEXT NOT NULL,
                        fencing_token INTEGER NOT NULL,
                        acquired_at_ms INTEGER NOT NULL,
                        expires_at_ms INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS proof_flight_expiry_idx
                        ON proof_single_flights(expires_at_ms);
                    CREATE TABLE IF NOT EXISTS proof_flight_outcomes (
                        key_id TEXT PRIMARY KEY,
                        fencing_token INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        outcome_json TEXT NOT NULL,
                        outcome_digest TEXT NOT NULL,
                        created_at_ms INTEGER NOT NULL,
                        expires_at_ms INTEGER NOT NULL
                    );
                    """
                )
            finally:
                connection.close()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _coerce_key(self, key: ProofCacheKey | Mapping[str, Any]) -> ProofCacheKey:
        return key if isinstance(key, ProofCacheKey) else ProofCacheKey.from_dict(key)

    def put(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        receipt: ProofReceipt | Mapping[str, Any],
        *,
        ttl_seconds: int | None = None,
        complete: bool = True,
    ) -> CacheStoreResult:
        """Validate and atomically store an authoritative proof receipt."""

        cache_key = self._coerce_key(key)
        try:
            typed_receipt = (
                receipt
                if isinstance(receipt, ProofReceipt)
                else ProofReceipt.from_dict(receipt)
            )
        except (TypeError, ValueError, ContractValidationError):
            return CacheStoreResult(
                False,
                cache_key,
                reason_codes=(CacheRejectionReason.MALFORMED_ENTRY.value,),
            )
        reasons = _trust_reasons(
            cache_key,
            typed_receipt,
            complete=complete,
            requirements=CacheRequirements(
                required_assurance=AssuranceLevel.KERNEL_VERIFIED
            ),
        )
        if reasons:
            return CacheStoreResult(
                False, cache_key, reason_codes=tuple(sorted(reasons))
            )
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        now = self._now_ms()
        entry = ProofCacheEntry.create(
            cache_key,
            typed_receipt,
            created_at_ms=now,
            expires_at_ms=now + ttl * 1000,
            complete=complete,
        )
        key_json = canonical_json(cache_key.to_dict())
        entry_json = canonical_json(entry.to_dict())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO proof_cache_entries(
                    key_id, key_json, entry_json, created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET
                    key_json=excluded.key_json,
                    entry_json=excluded.entry_json,
                    created_at_ms=excluded.created_at_ms,
                    expires_at_ms=excluded.expires_at_ms
                """,
                (
                    cache_key.key_id,
                    key_json,
                    entry_json,
                    entry.created_at_ms,
                    entry.expires_at_ms,
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return CacheStoreResult(True, cache_key, entry=entry)

    store = put

    def _decode_entry(
        self, key: ProofCacheKey, row: sqlite3.Row
    ) -> tuple[ProofCacheEntry | None, tuple[str, ...]]:
        try:
            stored_key_payload = _strict_json_loads(str(row["key_json"]))
            stored_payload = _strict_json_loads(str(row["entry_json"]))
            if not isinstance(stored_key_payload, Mapping) or not isinstance(
                stored_payload, Mapping
            ):
                raise ValueError("entry envelope must contain objects")
            stored_key = ProofCacheKey.from_dict(stored_key_payload)
            receipt_payload = stored_payload.get("receipt")
            if not isinstance(receipt_payload, Mapping):
                raise ValueError("entry receipt must be an object")
            receipt = ProofReceipt.from_dict(receipt_payload)
            created_at_ms = int(stored_payload["created_at_ms"])
            expires_at_ms = int(stored_payload["expires_at_ms"])
            complete = stored_payload["complete"]
            if not isinstance(complete, bool):
                raise ValueError("entry complete must be a boolean")
            digest = str(stored_payload["entry_digest"])
            entry = ProofCacheEntry(
                key=stored_key,
                receipt=receipt,
                created_at_ms=created_at_ms,
                expires_at_ms=expires_at_ms,
                complete=complete,
                entry_digest=digest,
            )
        except (KeyError, TypeError, ValueError, ContractValidationError, json.JSONDecodeError):
            return None, (CacheRejectionReason.MALFORMED_ENTRY.value,)

        poisoned = (
            stored_key.key_id != key.key_id
            or canonical_json(stored_key.to_dict()) != canonical_json(key.to_dict())
            or str(stored_payload.get("key_id")) != key.key_id
            or str(stored_payload.get("receipt_id")) != receipt.receipt_id
            or entry.entry_digest != entry.computed_digest
            or int(row["created_at_ms"]) != entry.created_at_ms
            or int(row["expires_at_ms"]) != entry.expires_at_ms
        )
        if poisoned:
            return None, (CacheRejectionReason.POISONED_ENTRY.value,)
        return entry, ()

    def lookup(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        *,
        required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
        required_freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
        max_age_seconds: int | None = None,
        requirements: CacheRequirements | None = None,
    ) -> CacheLookupResult:
        cache_key = self._coerce_key(key)
        requested = requirements or CacheRequirements(
            required_assurance=required_assurance,
            required_freshness=required_freshness,
            max_age_seconds=max_age_seconds,
        )
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM proof_cache_entries WHERE key_id=?",
                (cache_key.key_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return CacheLookupResult(
                CacheLookupStatus.MISS,
                cache_key,
                reason_codes=(CacheRejectionReason.CACHE_MISS.value,),
            )
        entry, decoding_reasons = self._decode_entry(cache_key, row)
        if entry is None:
            return CacheLookupResult(
                CacheLookupStatus.REJECTED,
                cache_key,
                reason_codes=decoding_reasons,
            )

        now = self._now_ms()
        reasons = _trust_reasons(
            cache_key,
            entry.receipt,
            complete=entry.complete,
            requirements=requested,
        )
        if now >= entry.expires_at_ms:
            reasons.add(CacheRejectionReason.STALE_ENTRY.value)
            reasons.add(CacheRejectionReason.FRESHNESS_NOT_SATISFIED.value)
        if (
            requested.max_age_seconds is not None
            and now - entry.created_at_ms > requested.max_age_seconds * 1000
        ):
            reasons.add(CacheRejectionReason.STALE_ENTRY.value)
            reasons.add(CacheRejectionReason.FRESHNESS_NOT_SATISFIED.value)
        if reasons:
            return CacheLookupResult(
                CacheLookupStatus.REJECTED,
                cache_key,
                entry=entry,
                reason_codes=tuple(sorted(reasons)),
            )
        return CacheLookupResult(CacheLookupStatus.HIT, cache_key, entry=entry)

    def get(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        **requirements: Any,
    ) -> ProofReceipt | None:
        """Compatibility helper returning only the accepted receipt."""

        return self.lookup(key, **requirements).receipt

    def delete(self, key: ProofCacheKey | Mapping[str, Any]) -> bool:
        cache_key = self._coerce_key(key)
        connection = self._connect()
        try:
            cursor = connection.execute(
                "DELETE FROM proof_cache_entries WHERE key_id=?",
                (cache_key.key_id,),
            )
            return cursor.rowcount > 0
        finally:
            connection.close()

    def purge_expired(self) -> int:
        now = self._now_ms()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            first = connection.execute(
                "DELETE FROM proof_cache_entries WHERE expires_at_ms<=?", (now,)
            ).rowcount
            connection.execute(
                "DELETE FROM proof_flight_outcomes WHERE expires_at_ms<=?", (now,)
            )
            connection.execute(
                "DELETE FROM proof_single_flights WHERE expires_at_ms<=?", (now,)
            )
            connection.commit()
            return first
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def acquire_lease(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        *,
        owner_id: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> SingleFlightLease:
        if isinstance(lease_seconds, bool) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        cache_key = self._coerce_key(key)
        owner = (owner_id or f"{os.getpid()}:{threading.get_ident()}").strip()
        if not owner:
            raise ValueError("owner_id must not be empty")
        now = self._now_ms()
        token = uuid.uuid4().hex
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM proof_single_flights WHERE key_id=?",
                (cache_key.key_id,),
            ).fetchone()
            if row is not None and int(row["expires_at_ms"]) > now:
                connection.commit()
                return SingleFlightLease(
                    cache=self,
                    key=cache_key,
                    owner_id=str(row["owner_id"]),
                    token=str(row["token"]),
                    fencing_token=int(row["fencing_token"]),
                    acquired_at_ms=int(row["acquired_at_ms"]),
                    expires_at_ms=int(row["expires_at_ms"]),
                    acquired=False,
                )
            fencing = int(row["fencing_token"]) + 1 if row is not None else 1
            expires = now + lease_seconds * 1000
            connection.execute(
                """
                INSERT INTO proof_single_flights(
                    key_id, owner_id, token, fencing_token,
                    acquired_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    token=excluded.token,
                    fencing_token=excluded.fencing_token,
                    acquired_at_ms=excluded.acquired_at_ms,
                    expires_at_ms=excluded.expires_at_ms
                """,
                (cache_key.key_id, owner, token, fencing, now, expires),
            )
            # An outcome from an older fencing generation must never satisfy a
            # follower of the new computation.
            connection.execute(
                "DELETE FROM proof_flight_outcomes WHERE key_id=?",
                (cache_key.key_id,),
            )
            connection.commit()
            return SingleFlightLease(
                cache=self,
                key=cache_key,
                owner_id=owner,
                token=token,
                fencing_token=fencing,
                acquired_at_ms=now,
                expires_at_ms=expires,
                acquired=True,
            )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    acquire_single_flight = acquire_lease

    def renew_lease(
        self,
        lease: SingleFlightLease,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
    ) -> SingleFlightLease:
        if not lease.acquired:
            raise SingleFlightError("only the lease owner can renew a lease")
        if isinstance(lease_seconds, bool) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = self._now_ms()
        expires = now + lease_seconds * 1000
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE proof_single_flights SET expires_at_ms=?
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                  AND expires_at_ms>?
                """,
                (
                    expires,
                    lease.key.key_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise SingleFlightError("single-flight lease is expired or fenced")
            connection.commit()
        finally:
            connection.close()
        return SingleFlightLease(
            cache=self,
            key=lease.key,
            owner_id=lease.owner_id,
            token=lease.token,
            fencing_token=lease.fencing_token,
            acquired_at_ms=lease.acquired_at_ms,
            expires_at_ms=expires,
            acquired=True,
        )

    def release_lease(self, lease: SingleFlightLease) -> bool:
        if not lease.acquired:
            return False
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                DELETE FROM proof_single_flights
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                """,
                (
                    lease.key.key_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                ),
            )
            return cursor.rowcount == 1
        finally:
            connection.close()

    def _publish_outcome(
        self,
        lease: SingleFlightLease,
        *,
        status: str,
        value: Any,
        ttl_seconds: int,
    ) -> None:
        if not lease.acquired:
            raise SingleFlightError("only the lease owner can publish an outcome")
        envelope = {
            "schema_version": FORMAL_VERIFICATION_FLIGHT_SCHEMA,
            "status": status,
            "value": _json_value(value, field_name="single-flight outcome"),
        }
        encoded = canonical_json(envelope)
        digest = f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
        now = self._now_ms()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM proof_single_flights
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                  AND expires_at_ms>?
                """,
                (
                    lease.key.key_id,
                    lease.owner_id,
                    lease.token,
                    lease.fencing_token,
                    now,
                ),
            ).fetchone()
            if active is None:
                connection.rollback()
                raise SingleFlightError("cannot publish from an expired or fenced lease")
            connection.execute(
                """
                INSERT INTO proof_flight_outcomes(
                    key_id, fencing_token, status, outcome_json,
                    outcome_digest, created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET
                    fencing_token=excluded.fencing_token,
                    status=excluded.status,
                    outcome_json=excluded.outcome_json,
                    outcome_digest=excluded.outcome_digest,
                    created_at_ms=excluded.created_at_ms,
                    expires_at_ms=excluded.expires_at_ms
                """,
                (
                    lease.key.key_id,
                    lease.fencing_token,
                    status,
                    encoded,
                    digest,
                    now,
                    now + ttl_seconds * 1000,
                ),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _read_outcome(self, key_id: str, fencing_token: int) -> tuple[bool, Any]:
        now = self._now_ms()
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM proof_flight_outcomes
                WHERE key_id=? AND fencing_token=? AND expires_at_ms>?
                """,
                (key_id, fencing_token, now),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return False, None
        try:
            payload = _strict_json_loads(str(row["outcome_json"]))
            encoded = canonical_json(payload)
            digest = f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
            if digest != str(row["outcome_digest"]):
                raise ValueError("single-flight outcome digest mismatch")
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema_version") != FORMAL_VERIFICATION_FLIGHT_SCHEMA
                or payload.get("status") != row["status"]
            ):
                raise ValueError("single-flight outcome envelope mismatch")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SingleFlightExecutionError("single-flight outcome is malformed") from exc
        if row["status"] == "error":
            error = payload.get("value")
            message = (
                str(error.get("message"))
                if isinstance(error, Mapping)
                else str(error)
            )
            raise SingleFlightExecutionError(message or "single-flight leader failed")
        return True, payload.get("value")

    def single_flight(
        self,
        key: ProofCacheKey | Mapping[str, Any],
        execute: Callable[[], T],
        *,
        owner_id: str | None = None,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        wait_timeout_seconds: int = DEFAULT_WAIT_TIMEOUT_SECONDS,
        poll_interval_seconds: float = 0.02,
        outcome_ttl_seconds: int = 60,
    ) -> T:
        """Execute once across threads/processes and share the exact outcome.

        Shared outcomes are a coordination channel, not proof-cache entries;
        callers must still use :meth:`lookup` for an authoritative cached
        receipt.
        """

        if not callable(execute):
            raise ValueError("execute must be callable")
        if (
            wait_timeout_seconds <= 0
            or poll_interval_seconds <= 0
            or outcome_ttl_seconds <= 0
        ):
            raise ValueError(
                "wait timeout, poll interval, and outcome TTL must be positive"
            )
        cache_key = self._coerce_key(key)
        deadline = time.monotonic() + wait_timeout_seconds
        while True:
            lease = self.acquire_lease(
                cache_key, owner_id=owner_id, lease_seconds=lease_seconds
            )
            if lease.acquired:
                heartbeat_stop = threading.Event()
                heartbeat_failures: list[BaseException] = []

                def heartbeat() -> None:
                    interval = max(0.1, lease_seconds / 3)
                    while not heartbeat_stop.wait(interval):
                        try:
                            self.renew_lease(
                                lease, lease_seconds=lease_seconds
                            )
                        except BaseException as exc:
                            heartbeat_failures.append(exc)
                            return

                heartbeat_thread = threading.Thread(
                    target=heartbeat,
                    name=f"proof-flight-heartbeat-{lease.fencing_token}",
                    daemon=True,
                )
                heartbeat_thread.start()
                try:
                    value = execute()
                    heartbeat_stop.set()
                    heartbeat_thread.join()
                    if heartbeat_failures:
                        raise SingleFlightError(
                            "single-flight lease heartbeat was fenced"
                        ) from heartbeat_failures[0]
                    self._publish_outcome(
                        lease,
                        status="ok",
                        value=value,
                        ttl_seconds=outcome_ttl_seconds,
                    )
                    return value
                except BaseException as exc:
                    heartbeat_stop.set()
                    heartbeat_thread.join()
                    try:
                        self._publish_outcome(
                            lease,
                            status="error",
                            value={
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            ttl_seconds=outcome_ttl_seconds,
                        )
                    except BaseException:
                        # Preserve the leader's original failure.  Followers
                        # will take over after the fenced/expired lease when
                        # an error outcome cannot be published.
                        pass
                    raise
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join()
                    lease.release()

            observed_fence = lease.fencing_token
            while time.monotonic() < deadline:
                found, value = self._read_outcome(
                    cache_key.key_id, observed_fence
                )
                if found:
                    return value
                connection = self._connect()
                try:
                    active = connection.execute(
                        """
                        SELECT fencing_token, expires_at_ms
                        FROM proof_single_flights WHERE key_id=?
                        """,
                        (cache_key.key_id,),
                    ).fetchone()
                finally:
                    connection.close()
                if (
                    active is None
                    or int(active["expires_at_ms"]) <= self._now_ms()
                    or int(active["fencing_token"]) != observed_fence
                ):
                    break
                time.sleep(poll_interval_seconds)
            if time.monotonic() >= deadline:
                raise SingleFlightTimeout(
                    f"timed out waiting for proof work {cache_key.key_id}"
                )

    execute_single_flight = single_flight
    run_single_flight = single_flight


# Concise compatibility name for callers that already operate in the formal
# verification package.
ProofCache = FormalVerificationCache
TrustAwareProofCache = FormalVerificationCache


__all__ = [
    "FORMAL_VERIFICATION_CACHE_SCHEMA",
    "FORMAL_VERIFICATION_CACHE_KEY_SCHEMA",
    "FORMAL_VERIFICATION_FLIGHT_SCHEMA",
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_WAIT_TIMEOUT_SECONDS",
    "CacheLookupStatus",
    "CacheRejectionReason",
    "CacheRequirements",
    "ProofCacheKey",
    "FormalVerificationCacheKey",
    "ProofCacheEntry",
    "CacheLookupResult",
    "CacheStoreResult",
    "SingleFlightLease",
    "SingleFlightError",
    "SingleFlightTimeout",
    "SingleFlightExecutionError",
    "FormalVerificationCache",
    "ProofCache",
    "TrustAwareProofCache",
    "build_proof_cache_key",
    "make_proof_cache_key",
]
