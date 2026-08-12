"""Exact-key verification receipt cache admission, lookup, and invalidation.

``VerificationReceiptCache@1`` is the fail-closed production admission layer on
top of :class:`VerificationReceiptStore`:

* exact ``VerificationReceiptKey`` lookup only (no semantic approximation);
* every hit re-derives key identity, content CID, kind, and terminal status;
* production reuse requires a current successful terminal receipt;
* stale / simulated / timeout / unavailable / invalid / malformed /
  kind-mismatched / key-mismatched / corrupt candidates never satisfy
  production;
* unrelated full-tree edits preserve the old immutable receipt under its old
  key without publishing a scoped-staleness tombstone, while the new tree key
  cannot reuse it;
* concurrent writers merge via generation CAS retry and never overwrite peers.

Cache presence is never authority.  The store cannot upgrade evidence or
mutate immutable receipt bytes in place.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .contracts import (
    PROOF_RECEIPT_SCHEMA,
    STATIC_ANALYSIS_RECEIPT_SCHEMA,
    TEST_RECEIPT_SCHEMA,
    TYPE_CHECK_RECEIPT_SCHEMA,
    CacheReuseDecision,
    CacheReuseDisposition,
    ProofReceipt,
    StaticAnalysisReceipt,
    TerminalStatus,
    TestReceipt,
    TypeCheckReceipt,
    VerificationContractError,
    VerificationIdentityError,
    VerificationReceipt,
    VerificationReceiptKey,
)
from .receipt_store import (
    CompareAndSwapResult,
    GCMetadata,
    IndexEntry,
    IndexSnapshot,
    ReceiptStoreError,
    ReceiptStoreIntegrityError,
    TombstoneRecord,
    VerificationReceiptStore,
    cas_publish_entry,
    mapping_cid,
)

# ---------------------------------------------------------------------------
# Evidence / interface constants
# ---------------------------------------------------------------------------

VERIFICATION_RECEIPT_CACHE_INTERFACE: Final[str] = "VerificationReceiptCache@1"
RECEIPT_CACHE_EVIDENCE: Final[str] = "ivp/receipt-cache@1"
CONCURRENT_WRITER_EVIDENCE: Final[str] = "ivp/concurrent-writer@1"
REPLAY_CORRUPTION_EVIDENCE: Final[str] = "ivp/replay-corruption@1"

DEFAULT_MAX_CAS_RETRIES: Final[int] = 32

REASON_EXACT_CURRENT_PRODUCTION: Final[str] = "exact_current_production_receipt"
REASON_CACHE_MISS: Final[str] = "cache_miss"
REASON_TOMBSTONED: Final[str] = "scoped_staleness_tombstone"
REASON_STALE_STATUS: Final[str] = "stale_terminal_status"
REASON_SIMULATED: Final[str] = "simulated_not_production"
REASON_TIMEOUT: Final[str] = "timeout_not_production"
REASON_UNAVAILABLE: Final[str] = "unavailable_not_production"
REASON_INVALID: Final[str] = "invalid_not_production"
REASON_TERMINAL_REJECTED: Final[str] = "terminal_status_not_production"
REASON_MALFORMED: Final[str] = "malformed_candidate"
REASON_CORRUPT: Final[str] = "corrupt_candidate"
REASON_KIND_MISMATCH: Final[str] = "kind_mismatched"
REASON_KEY_MISMATCH: Final[str] = "key_mismatched"
REASON_BODY_CID_MISMATCH: Final[str] = "body_cid_mismatch"
REASON_RECEIPT_CID_MISMATCH: Final[str] = "receipt_cid_mismatch"
REASON_STORE_UNAVAILABLE: Final[str] = "store_unavailable"
REASON_NON_AUTHORITATIVE: Final[str] = "non_authoritative_candidate"
REASON_ADMITTED: Final[str] = "admitted_exact_key"
REASON_ADMIT_REJECTED: Final[str] = "admit_rejected_not_production_eligible"
REASON_MARK_STALE: Final[str] = "mark_stale_tombstone"
REASON_CAS_EXHAUSTED: Final[str] = "cas_retries_exhausted"

_RECEIPT_TYPES_BY_SCHEMA: Final[Mapping[str, type]] = {
    STATIC_ANALYSIS_RECEIPT_SCHEMA: StaticAnalysisReceipt,
    TYPE_CHECK_RECEIPT_SCHEMA: TypeCheckReceipt,
    TEST_RECEIPT_SCHEMA: TestReceipt,
    PROOF_RECEIPT_SCHEMA: ProofReceipt,
}

_PRODUCTION_SUCCESS: Final[frozenset[TerminalStatus]] = frozenset(
    {TerminalStatus.PASSED, TerminalStatus.PROVED}
)

_STATUS_DISPOSITION: Final[Mapping[TerminalStatus, tuple[CacheReuseDisposition, str]]] = {
    TerminalStatus.STALE: (CacheReuseDisposition.STALE, REASON_STALE_STATUS),
    TerminalStatus.SIMULATED: (CacheReuseDisposition.SIMULATED, REASON_SIMULATED),
    TerminalStatus.TIMEOUT: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_TIMEOUT,
    ),
    TerminalStatus.UNAVAILABLE: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_UNAVAILABLE,
    ),
    TerminalStatus.INVALID: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_INVALID,
    ),
    TerminalStatus.FAILED: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_TERMINAL_REJECTED,
    ),
    TerminalStatus.DISPROVED: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_TERMINAL_REJECTED,
    ),
    TerminalStatus.UNKNOWN: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_TERMINAL_REJECTED,
    ),
    TerminalStatus.NOT_MODELED: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_TERMINAL_REJECTED,
    ),
    TerminalStatus.CANCELLED: (
        CacheReuseDisposition.TERMINAL_STATUS_REJECTED,
        REASON_TERMINAL_REJECTED,
    ),
}


# ---------------------------------------------------------------------------
# Errors and result types
# ---------------------------------------------------------------------------


class ReceiptCacheError(RuntimeError):
    """Base operational failure for the verification receipt cache."""


class ReceiptCacheIntegrityError(ReceiptCacheError, ValueError):
    """Candidate failed integrity or identity revalidation."""


class ReceiptCacheAdmitError(ReceiptCacheError, ValueError):
    """Receipt rejected before durable admission."""


class ProductionEligibility(str, Enum):
    """Whether a candidate may satisfy production reuse."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"


@dataclass(frozen=True, slots=True)
class AdmitResult:
    """Outcome of attempting to admit a receipt into the exact-key index."""

    success: bool
    key_id: str
    receipt_cid: str
    created: bool
    cas: CompareAndSwapResult | None
    reason: str
    production_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "key_id": self.key_id,
            "receipt_cid": self.receipt_cid,
            "created": self.created,
            "cas": self.cas.to_dict() if self.cas is not None else None,
            "reason": self.reason,
            "production_eligible": self.production_eligible,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms(clock: Callable[[], int] | None = None) -> int:
    if clock is not None:
        return int(clock())
    import time

    return int(time.time() * 1000)


def production_eligible(receipt: VerificationReceipt) -> bool:
    """Return True when a well-formed receipt may satisfy production reuse."""

    status = receipt.status
    if status not in _PRODUCTION_SUCCESS:
        return False
    if isinstance(receipt, TestReceipt):
        source_key = receipt.test_execution_key
        if source_key is not None:
            eligibility = getattr(source_key, "eligibility_class", None)
            eligibility_value = getattr(eligibility, "value", eligibility)
            if eligibility_value == "non_reusable":
                return False
            components = getattr(source_key, "components", None) or {}
            if bool(components.get("non_reusable_reason")):
                return False
    return True


def decode_verification_receipt(value: Any) -> VerificationReceipt:
    """Decode a canonical receipt record; raise on unsupported/malformed input."""

    if isinstance(
        value, (StaticAnalysisReceipt, TypeCheckReceipt, TestReceipt, ProofReceipt)
    ):
        return value
    if not isinstance(value, Mapping):
        raise ReceiptCacheIntegrityError("receipt must be a canonical receipt record")
    schema = value.get("schema")
    receipt_type = _RECEIPT_TYPES_BY_SCHEMA.get(str(schema) if schema is not None else "")
    if receipt_type is None:
        raise ReceiptCacheIntegrityError(
            f"receipt has an unsupported schema: {schema!r}"
        )
    try:
        result = receipt_type.from_dict(value)  # type: ignore[attr-defined]
    except (VerificationContractError, VerificationIdentityError, ValueError, TypeError) as exc:
        raise ReceiptCacheIntegrityError(f"malformed receipt: {exc}") from exc
    assert isinstance(
        result, (StaticAnalysisReceipt, TypeCheckReceipt, TestReceipt, ProofReceipt)
    )
    return result


def _decision(
    key: VerificationReceiptKey,
    disposition: CacheReuseDisposition,
    *reason_codes: str,
    candidate: VerificationReceipt | None = None,
) -> CacheReuseDecision:
    reasons = tuple(code for code in reason_codes if code)
    if not reasons:
        reasons = ("unspecified",)
    # CacheReuseDecision forbids candidates on MISSING; for SIMULATED requires
    # matching status.  Other dispositions may carry candidates for audit.
    if disposition is CacheReuseDisposition.MISSING:
        candidate = None
    if disposition is CacheReuseDisposition.REUSED and candidate is None:
        raise ReceiptCacheIntegrityError("reused decision requires a candidate")
    try:
        return CacheReuseDecision(
            key_cid=key.key_id,
            disposition=disposition,
            reason_codes=reasons,
            candidate_receipt=candidate,
        )
    except (VerificationContractError, VerificationIdentityError) as exc:
        # Fall back to a policy rejection if the candidate cannot be attached
        # under contract rules (e.g. non-successful reused attempt).
        if disposition is CacheReuseDisposition.REUSED:
            return CacheReuseDecision(
                key_cid=key.key_id,
                disposition=CacheReuseDisposition.POLICY_REJECTED,
                reason_codes=(REASON_TERMINAL_REJECTED, f"contract:{exc}"),
                candidate_receipt=None,
            )
        raise


def _load_envelope_body(
    store: VerificationReceiptStore,
    receipt_cid: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Return (envelope, body) after verifying body_cid integrity."""

    try:
        envelope = store.get_receipt_envelope(receipt_cid)
    except ReceiptStoreIntegrityError as exc:
        raise ReceiptCacheIntegrityError(f"corrupt envelope: {exc}") from exc
    except ReceiptStoreError as exc:
        raise ReceiptCacheError(f"store unavailable for {receipt_cid}: {exc}") from exc

    body = envelope.get("body")
    if not isinstance(body, Mapping):
        raise ReceiptCacheIntegrityError(REASON_MALFORMED)
    body_dict = dict(body)
    expected_body_cid = envelope.get("body_cid")
    recomputed = mapping_cid(body_dict)
    if not isinstance(expected_body_cid, str) or not expected_body_cid:
        raise ReceiptCacheIntegrityError(REASON_BODY_CID_MISMATCH)
    if expected_body_cid != recomputed:
        raise ReceiptCacheIntegrityError(REASON_BODY_CID_MISMATCH)
    return envelope, body_dict


def classify_candidate(
    key: VerificationReceiptKey,
    *,
    receipt: VerificationReceipt | None = None,
    receipt_cid: str | None = None,
    error: BaseException | None = None,
    tombstoned: bool = False,
    for_production: bool = True,
) -> CacheReuseDecision:
    """Classify a candidate into a :class:`CacheReuseDecision` (fail closed)."""

    if tombstoned:
        return _decision(
            key,
            CacheReuseDisposition.STALE,
            REASON_TOMBSTONED,
            candidate=receipt,
        )

    if error is not None:
        message = str(error).lower()
        if REASON_MALFORMED in message or "malformed" in message:
            return _decision(key, CacheReuseDisposition.CORRUPT, REASON_MALFORMED)
        if REASON_BODY_CID_MISMATCH in message or "body_cid" in message:
            return _decision(key, CacheReuseDisposition.CORRUPT, REASON_BODY_CID_MISMATCH)
        if "unsupported schema" in message or "corrupt" in message:
            return _decision(key, CacheReuseDisposition.CORRUPT, REASON_CORRUPT)
        return _decision(key, CacheReuseDisposition.CORRUPT, REASON_CORRUPT)

    if receipt is None:
        return _decision(key, CacheReuseDisposition.MISSING, REASON_CACHE_MISS)

    # Exact key identity revalidation on every hit.
    if receipt.key.key_id != key.key_id:
        return _decision(
            key,
            CacheReuseDisposition.MISMATCHED,
            REASON_KEY_MISMATCH,
            candidate=receipt,
        )
    # Full structural key equality (not just content id).
    if receipt.key != key:
        return _decision(
            key,
            CacheReuseDisposition.MISMATCHED,
            REASON_KEY_MISMATCH,
            candidate=receipt,
        )
    if receipt.key.receipt_kind is not key.receipt_kind:
        return _decision(
            key,
            CacheReuseDisposition.MISMATCHED,
            REASON_KIND_MISMATCH,
            candidate=receipt,
        )
    if receipt_cid is not None and receipt.receipt_id and receipt_cid != receipt.receipt_id:
        # Index points at envelope CID; receipt_id is content id of the body.
        # They differ by design (envelope wraps body).  Only flag when the
        # caller claimed a body/receipt content CID that disagrees.
        pass

    status = receipt.status
    if status in _STATUS_DISPOSITION:
        disposition, reason = _STATUS_DISPOSITION[status]
        return _decision(key, disposition, reason, candidate=receipt)

    if for_production:
        if not production_eligible(receipt):
            return _decision(
                key,
                CacheReuseDisposition.POLICY_REJECTED,
                REASON_TERMINAL_REJECTED,
                candidate=receipt,
            )
        return _decision(
            key,
            CacheReuseDisposition.REUSED,
            REASON_EXACT_CURRENT_PRODUCTION,
            candidate=receipt,
        )

    # Non-production advisory lookup: still require exact key match and
    # successful terminal for reuse disposition; otherwise report status.
    if status in _PRODUCTION_SUCCESS:
        return _decision(
            key,
            CacheReuseDisposition.REUSED,
            REASON_EXACT_CURRENT_PRODUCTION,
            candidate=receipt,
        )
    return _decision(
        key,
        CacheReuseDisposition.NON_AUTHORITATIVE,
        REASON_NON_AUTHORITATIVE,
        candidate=receipt,
    )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class VerificationReceiptCache:
    """Durable exact-key verification receipt cache over a store protocol."""

    INTERFACE: Final[str] = VERIFICATION_RECEIPT_CACHE_INTERFACE

    def __init__(
        self,
        store: VerificationReceiptStore,
        *,
        max_cas_retries: int = DEFAULT_MAX_CAS_RETRIES,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if store is None:
            raise ReceiptCacheError("VerificationReceiptCache requires a store")
        if (
            isinstance(max_cas_retries, bool)
            or not isinstance(max_cas_retries, int)
            or max_cas_retries < 1
        ):
            raise ReceiptCacheError("max_cas_retries must be a positive int")
        self._store = store
        self._max_cas_retries = int(max_cas_retries)
        self._clock = clock

    @property
    def store(self) -> VerificationReceiptStore:
        return self._store

    # -- lookup -------------------------------------------------------------

    def lookup(
        self,
        key: VerificationReceiptKey,
        *,
        for_production: bool = True,
        touch_access: bool = True,
    ) -> CacheReuseDecision:
        """Exact-key lookup with full revalidation on every hit.

        Production reuse requires an exact key match and a successful terminal
        status.  Cache presence alone never upgrades evidence.
        """

        if not isinstance(key, VerificationReceiptKey):
            raise ReceiptCacheError("lookup requires a VerificationReceiptKey")

        try:
            index = self._store.current_index()
        except ReceiptStoreError:
            return _decision(
                key,
                CacheReuseDisposition.MISSING,
                REASON_STORE_UNAVAILABLE,
            )

        entry_map = index.entry_map()
        tombstone_keys = {item.key_id for item in index.tombstones}

        if key.key_id in tombstone_keys and key.key_id not in entry_map:
            return classify_candidate(key, tombstoned=True, for_production=for_production)

        entry = entry_map.get(key.key_id)
        if entry is None:
            return classify_candidate(key, for_production=for_production)

        try:
            _envelope, body = _load_envelope_body(self._store, entry.receipt_cid)
            receipt = decode_verification_receipt(body)
            if touch_access:
                try:
                    self._store.record_access(entry.receipt_cid)
                except ReceiptStoreError:
                    # Access metadata is best-effort for GC; never upgrades
                    # authority and never blocks a fail-closed decision.
                    pass
            return classify_candidate(
                key,
                receipt=receipt,
                receipt_cid=entry.receipt_cid,
                for_production=for_production,
            )
        except ReceiptCacheIntegrityError as exc:
            return classify_candidate(key, error=exc, for_production=for_production)
        except (VerificationContractError, VerificationIdentityError, ValueError, TypeError) as exc:
            return classify_candidate(
                key,
                error=ReceiptCacheIntegrityError(f"{REASON_MALFORMED}: {exc}"),
                for_production=for_production,
            )
        except ReceiptStoreError as exc:
            return classify_candidate(
                key,
                error=ReceiptCacheIntegrityError(f"{REASON_CORRUPT}: {exc}"),
                for_production=for_production,
            )

    def lookup_many(
        self,
        keys: Sequence[VerificationReceiptKey],
        *,
        for_production: bool = True,
    ) -> tuple[CacheReuseDecision, ...]:
        """Lookup each key independently; order preserved."""

        return tuple(
            self.lookup(key, for_production=for_production) for key in keys
        )

    # -- admit --------------------------------------------------------------

    def admit(
        self,
        receipt: VerificationReceipt,
        *,
        for_production: bool = True,
        require_production_eligible: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AdmitResult:
        """Durably store an immutable receipt and index it under its exact key.

        Concurrent writers merge via generation CAS retry so peer entries are
        never overwritten.  Immutable receipt bytes are never mutated in place.

        When ``require_production_eligible`` is true (default when
        ``for_production`` is true), only production-eligible successful
        receipts are indexed for production reuse.  Structural validation
        always runs.
        """

        if require_production_eligible is None:
            require_production_eligible = for_production

        try:
            normalized = decode_verification_receipt(receipt)
        except ReceiptCacheIntegrityError as exc:
            raise ReceiptCacheAdmitError(str(exc)) from exc

        eligible = production_eligible(normalized)
        if require_production_eligible and not eligible:
            return AdmitResult(
                success=False,
                key_id=normalized.key.key_id,
                receipt_cid="",
                created=False,
                cas=None,
                reason=REASON_ADMIT_REJECTED,
                production_eligible=False,
            )

        body = normalized.to_record()
        try:
            put = self._store.put_receipt_envelope(
                body,
                stored_at_ms=_now_ms(self._clock),
                metadata=dict(metadata or {}),
            )
        except ReceiptStoreError as exc:
            raise ReceiptCacheError(f"failed to put receipt envelope: {exc}") from exc

        entry = IndexEntry(
            key_id=normalized.key.key_id,
            receipt_cid=put.cid,
            kind="receipt",
            metadata={
                "receipt_id": normalized.receipt_id,
                "receipt_kind": normalized.key.receipt_kind.value,
                "status": normalized.status.value,
                "production_eligible": eligible,
                **dict(metadata or {}),
            },
        )
        cas = cas_publish_entry(
            self._store,
            entry,
            max_retries=self._max_cas_retries,
            clock=self._clock,
        )
        if not cas.success:
            return AdmitResult(
                success=False,
                key_id=normalized.key.key_id,
                receipt_cid=put.cid,
                created=put.created,
                cas=cas,
                reason=cas.reason or REASON_CAS_EXHAUSTED,
                production_eligible=eligible,
            )

        try:
            self._store.record_access(put.cid, at_ms=_now_ms(self._clock))
        except ReceiptStoreError:
            pass

        return AdmitResult(
            success=True,
            key_id=normalized.key.key_id,
            receipt_cid=put.cid,
            created=put.created,
            cas=cas,
            reason=REASON_ADMITTED,
            production_eligible=eligible,
        )

    # -- invalidation / tombstones ------------------------------------------

    def mark_stale(
        self,
        key: VerificationReceiptKey | str,
        *,
        reason: str = REASON_MARK_STALE,
        prior_receipt_cid: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CompareAndSwapResult:
        """Publish a scoped-staleness tombstone for ``key``.

        Immutable history is retained: prior receipt bytes remain readable and
        prior index generations remain in ``replay()``.  Unrelated full-tree
        identity changes must **not** call this method; they simply use a new
        key and leave the old immutable entry untouched.
        """

        key_id = key.key_id if isinstance(key, VerificationReceiptKey) else str(key)
        if not key_id.strip():
            raise ReceiptCacheError("mark_stale requires a key id")
        if not isinstance(reason, str) or not reason.strip():
            raise ReceiptCacheError("mark_stale requires a non-empty reason")

        last: CompareAndSwapResult | None = None
        for _ in range(self._max_cas_retries):
            current = self._store.current_index()
            entry_map = current.entry_map()
            entry = entry_map.get(key_id)
            resolved_prior = prior_receipt_cid
            if resolved_prior is None:
                if entry is None:
                    # Nothing live to tombstone; treat as success no-op with
                    # a synthetic non-conflict result describing current head.
                    return CompareAndSwapResult(
                        success=True,
                        conflict=False,
                        generation=current.generation,
                        root_cid=(
                            current.root_cid
                            if current.generation > 0
                            else "empty"
                        ),
                        expected_generation=current.generation,
                        expected_root_cid=(
                            current.root_cid if current.generation > 0 else None
                        ),
                        snapshot=current,
                        reason="already_absent",
                    )
                resolved_prior = entry.receipt_cid

            expected_generation = current.generation
            expected_root = current.root_cid if current.generation > 0 else None
            tomb = TombstoneRecord(
                key_id=key_id,
                prior_receipt_cid=resolved_prior,
                reason=reason.strip(),
                tombstoned_at_ms=_now_ms(self._clock),
                metadata=dict(metadata or {}),
            )
            last = self._store.publish_tombstone(
                tomb,
                expected_generation=expected_generation,
                expected_root_cid=expected_root,
            )
            if last.success or not last.conflict:
                return last
            # Conflict: retry from a freshly read root; never overwrite peers.
        assert last is not None
        return last

    # Alias used by objective wording.
    def tombstone(
        self,
        key: VerificationReceiptKey | str,
        *,
        reason: str = REASON_MARK_STALE,
        prior_receipt_cid: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CompareAndSwapResult:
        return self.mark_stale(
            key,
            reason=reason,
            prior_receipt_cid=prior_receipt_cid,
            metadata=metadata,
        )

    # -- history / GC -------------------------------------------------------

    def replay(self) -> tuple[IndexSnapshot, ...]:
        """Replay immutable index generations from the underlying store."""

        return self._store.replay_history()

    def collect_gc_metadata(self) -> tuple[GCMetadata, ...]:
        """Return reachability / last-access GC metadata from the store."""

        return self._store.collect_gc_metadata()

    def gc_metadata(self) -> tuple[GCMetadata, ...]:
        """Alias for :meth:`collect_gc_metadata` (todo interface name)."""

        return self.collect_gc_metadata()

    def current_index(self) -> IndexSnapshot:
        return self._store.current_index()

    def get_historical(
        self,
        key: VerificationReceiptKey | str,
    ) -> VerificationReceipt | None:
        """Return the live receipt for ``key`` without production filtering.

        Used for immutable history inspection.  Does not upgrade authority and
        does not treat the candidate as production-reusable.
        """

        key_id = key.key_id if isinstance(key, VerificationReceiptKey) else str(key)
        index = self._store.current_index()
        entry = index.entry_map().get(key_id)
        if entry is None:
            return None
        try:
            _envelope, body = _load_envelope_body(self._store, entry.receipt_cid)
            return decode_verification_receipt(body)
        except (ReceiptCacheIntegrityError, ReceiptStoreError):
            return None


__all__ = [
    "AdmitResult",
    "CONCURRENT_WRITER_EVIDENCE",
    "DEFAULT_MAX_CAS_RETRIES",
    "ProductionEligibility",
    "REASON_ADMITTED",
    "REASON_ADMIT_REJECTED",
    "REASON_BODY_CID_MISMATCH",
    "REASON_CACHE_MISS",
    "REASON_CORRUPT",
    "REASON_EXACT_CURRENT_PRODUCTION",
    "REASON_INVALID",
    "REASON_KEY_MISMATCH",
    "REASON_KIND_MISMATCH",
    "REASON_MALFORMED",
    "REASON_MARK_STALE",
    "REASON_SIMULATED",
    "REASON_STALE_STATUS",
    "REASON_TIMEOUT",
    "REASON_TOMBSTONED",
    "REASON_UNAVAILABLE",
    "RECEIPT_CACHE_EVIDENCE",
    "REPLAY_CORRUPTION_EVIDENCE",
    "ReceiptCacheAdmitError",
    "ReceiptCacheError",
    "ReceiptCacheIntegrityError",
    "VERIFICATION_RECEIPT_CACHE_INTERFACE",
    "VerificationReceiptCache",
    "classify_candidate",
    "decode_verification_receipt",
    "production_eligible",
]
