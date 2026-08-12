"""Bounded candidate lookup and pytest skip application.

The locator index is only a hint.  This module implements two admission paths:

1. **Legacy / certificate-only** (:class:`ProofReuseLookup`): delegates immutable
   candidate admission to :class:`TestProofCache`, which rehashes retained
   certificate bytes and checks the exact current locator, execution key,
   policy, revocation state, and local proof verifier.

2. **Two-stage warm path** (:class:`ProofReuseTwoStageLookup`, PTR-145): begins
   with locator + current collected item only, loads retained bytes from a
   dedicated :class:`TestCandidateContextStore`, rehashes every component,
   resolves the retained runtime frontier against admitted live roots, rebuilds
   fresh current identity without fixture or test execution, requires the
   current execution key to match the candidate, and only then hands off to
   certificate-cache verification.  Revalidation alone can never skip.

Every optional-boundary failure is converted to an explicit ``RUN`` decision.
No lookup path in this module calls a prover or an issuer handle.
"""

from __future__ import annotations

import inspect
import json
import queue
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from ...agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    ReuseDecision,
    ReuseReasonCode,
    TestExecutionKey,
    TestLocatorKey,
    decision_from_exception,
    reuse_run,
)
from ...agent_supervisor.proof.test_proof_cache import (
    DEFAULT_MAX_BLOB_BYTES,
    DEFAULT_MAX_CANDIDATES,
    TestProofCache,
    TestProofCacheLookupStatus,
)

PROOF_REUSE_LOOKUP_INTERFACE: Final = "ProofReuseLookup@1"
# Production two-stage interface upgraded by PTR-164 for signed-receipt trust.
TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE: Final = "TwoStageCandidateLookup@2"
# Historical alias retained for importers that still name the PTR-145 surface.
PROOF_REUSE_TWO_STAGE_LOOKUP_INTERFACE: Final = TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE
SIGNED_RECEIPT_TRUST_VERIFIER_INTERFACE: Final = "SignedReceiptTrustVerifier@1"
ITEM_DECISION_ATTRIBUTE: Final = "_ipfs_proof_reuse_decision"
ITEM_LOOKUP_REQUEST_ATTRIBUTE: Final = "_ipfs_proof_reuse_lookup_request"
SKIP_REASON_PREFIX: Final = "proof-cache-hit:"

DEFAULT_LOOKUP_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_MAX_BATCH_ITEMS: Final = 4096
MAX_DIAGNOSTIC_TEXT: Final = 128
MAX_ATTESTATION_BYTES: Final = 64 * 1024

_USER_PROPERTY_KEYS: Final = frozenset(
    {
        "proof_reuse_action",
        "proof_reuse_reason",
        "proof_reuse_certificate_cid",
        "proof_reuse_receipt_cid",
    }
)


@dataclass(frozen=True)
class ProofReuseLookupRequest:
    """One collected item's current identity and optional eligibility gate."""

    item: Any
    locator: Any
    execution_key: Any
    eligibility: Any = None
    current_policy: Mapping[str, Any] | None = None
    now_ms: int | None = None


@dataclass(frozen=True)
class RevalidatedProofReuseLookupRequest:
    """Locator-first warm lookup request (execution key optional until revalidated).

    Warm admission begins with the stable locator and the current collected
    item only.  A final execution key may be absent; when present it is still
    checked for exact agreement with the retained candidate after fresh
    current-context rebuild.
    """

    item: Any
    locator: Any
    execution_key: Any = None
    eligibility: Any = None
    current_policy: Mapping[str, Any] | None = None
    now_ms: int | None = None
    allowed_roots: Mapping[str, Any] | None = None

    def to_lookup_request(self) -> ProofReuseLookupRequest:
        return ProofReuseLookupRequest(
            item=self.item,
            locator=self.locator,
            execution_key=self.execution_key,
            eligibility=self.eligibility,
            current_policy=self.current_policy,
            now_ms=self.now_ms,
        )


class _CandidateStoreError(RuntimeError):
    """A bounded, internal marker for an unavailable candidate store."""


class _LookupTimedOut(TimeoutError):
    """The entire read-and-verify operation exceeded its collection budget."""


def _bounded_type_name(value: Any) -> str:
    return type(value).__name__[:MAX_DIAGNOSTIC_TEXT]


def _run_for_exception(exc: BaseException, *, stage: str) -> ReuseDecision:
    reason = (
        ReuseReasonCode.TIMEOUT
        if isinstance(exc, (TimeoutError, _LookupTimedOut))
        else ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN
    )
    return decision_from_exception(
        exc,
        reason_code=reason,
        diagnostics={"stage": stage[:MAX_DIAGNOSTIC_TEXT]},
    )


def _bounded_call(function: Callable[[], Any], timeout_seconds: float) -> Any:
    """Run an optional-boundary call without allowing it to block collection.

    A daemon thread is intentional: cancelling a Python call cannot safely
    kill it, while waiting for an executor shutdown would defeat the bound.
    Lookup is read-only, so a timed-out worker has no publication authority.
    """

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result = (True, function())
        except BaseException as exc:  # optional providers must fail open
            result = (False, exc)
        try:
            result_queue.put_nowait(result)
        except queue.Full:  # pragma: no cover - only possible after corruption
            pass

    worker = threading.Thread(
        target=_worker,
        name="proof-reuse-lookup",
        daemon=True,
    )
    worker.start()
    try:
        succeeded, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        raise _LookupTimedOut("proof reuse lookup exceeded its time budget") from None
    if succeeded:
        return value
    raise value


def _call_candidate_method(
    method: Callable[..., Any],
    locator: TestLocatorKey,
    *,
    max_candidates: int,
) -> Any:
    """Invoke common store adapters while passing a candidate bound when supported."""

    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_bound = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or parameter.name == "max_candidates"
        for parameter in parameters
    )
    locator_value: Any = locator.locator_id
    if getattr(method, "__name__", "") == "candidate_provider":
        locator_value = locator
    if accepts_bound:
        return method(locator_value, max_candidates=max_candidates)
    return method(locator_value)


def _materialize_store_result(value: Any) -> Any:
    """Unwrap typed store results without trusting their status as authority."""

    if value is None:
        return ()
    candidates = getattr(value, "candidates", None)
    if candidates is not None:
        status = str(getattr(value, "status", "")).lower()
        if not candidates and status.endswith(("error", "rejected")):
            raise _CandidateStoreError("candidate store lookup failed")
        return candidates
    if isinstance(value, Mapping) and "candidates" in value:
        candidates = value.get("candidates")
        if candidates is None:
            return ()
        return candidates
    return value


def _normalise_locator(value: Any) -> tuple[TestLocatorKey | None, ReuseDecision | None]:
    if isinstance(value, TestLocatorKey):
        locator = value
    else:
        reusable = getattr(value, "reusable", None)
        locator = getattr(value, "locator", None)
        if reusable is False:
            return None, reuse_run(ReuseReasonCode.NON_REUSABLE)
        if reusable is not True or not isinstance(locator, TestLocatorKey):
            return None, reuse_run(
                ReuseReasonCode.UNSUPPORTED,
                diagnostics={"locator_type": _bounded_type_name(value)},
            )
        if getattr(value, "locator_cid", "") != locator.locator_id:
            return None, reuse_run(ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED)
    if locator.non_reusable_reason:
        return None, reuse_run(ReuseReasonCode.NON_REUSABLE)
    return locator, None


def _normalise_execution_key(
    value: Any,
) -> tuple[TestExecutionKey | None, ReuseDecision | None]:
    if isinstance(value, TestExecutionKey):
        execution_key = value
    else:
        reusable = getattr(value, "reusable", None)
        execution_key = getattr(value, "execution_key", None)
        if reusable is False:
            return None, reuse_run(ReuseReasonCode.NON_REUSABLE)
        if reusable is not True or not isinstance(execution_key, TestExecutionKey):
            return None, reuse_run(
                ReuseReasonCode.UNSUPPORTED,
                diagnostics={"execution_key_type": _bounded_type_name(value)},
            )
        if getattr(value, "execution_cid", "") != execution_key.execution_key_id:
            return None, reuse_run(ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED)
    return execution_key, None


def _eligibility_decision(
    eligibility: Any,
    execution_key: TestExecutionKey,
) -> ReuseDecision | None:
    if eligibility is None or eligibility is True:
        return None
    if eligibility is False:
        return reuse_run(ReuseReasonCode.ELIGIBILITY_DENIED)
    reusable = getattr(eligibility, "reusable", None)
    if reusable is not True:
        reason_method = getattr(eligibility, "as_reuse_reason", None)
        try:
            reason = reason_method() if callable(reason_method) else None
        except BaseException as exc:
            return _run_for_exception(exc, stage="eligibility_reason")
        if not isinstance(reason, ReuseReasonCode) or reason is ReuseReasonCode.PROOF_CACHE_HIT:
            reason = ReuseReasonCode.ELIGIBILITY_DENIED
        return reuse_run(reason)

    verify = getattr(eligibility, "verify", None)
    if not callable(verify):
        return reuse_run(ReuseReasonCode.UNSUPPORTED)
    try:
        verified = verify()
    except BaseException as exc:
        return _run_for_exception(exc, stage="eligibility_verify")
    if verified is not eligibility or getattr(verified, "reusable", None) is not True:
        return reuse_run(ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED)

    for attribute in (
        "repository_forest_cid",
        "static_trace_root_cid",
        "runtime_trace_root_cid",
    ):
        eligible_value = getattr(verified, attribute, "")
        current_value = getattr(execution_key, attribute, "")
        if eligible_value and eligible_value != current_value:
            return reuse_run(ReuseReasonCode.INVALIDATION)
    return None


class ProofReuseLookup:
    """Fail-open orchestrator around the authoritative proof-cache admission."""

    interface = PROOF_REUSE_LOOKUP_INTERFACE

    def __init__(
        self,
        candidate_store: Any = None,
        certificate_provider: Any = None,
        *,
        store: Any = None,
        provider: Any = None,
        verifier: Any = None,
        current_policy: Mapping[str, Any] | None = None,
        policy_provider: Callable[
            [TestLocatorKey, TestExecutionKey], Mapping[str, Any]
        ]
        | None = None,
        revocation_checker: Callable[..., Any] | None = None,
        clock: Callable[[], int] | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS,
        timeout_seconds: float = DEFAULT_LOOKUP_TIMEOUT_SECONDS,
    ) -> None:
        if candidate_store is not None and store is not None:
            raise ValueError("specify candidate_store or store, not both")
        if certificate_provider is not None and provider is not None:
            raise ValueError("specify certificate_provider or provider, not both")
        for name, value in (
            ("max_candidates", max_candidates),
            ("max_blob_bytes", max_blob_bytes),
            ("max_batch_items", max_batch_items),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")

        self._candidate_store = candidate_store if store is None else store
        self._certificate_provider = (
            certificate_provider if provider is None else provider
        )
        self._verifier = verifier
        self._current_policy = (
            dict(current_policy) if current_policy is not None else None
        )
        self._policy_provider = policy_provider
        self._revocation_checker = revocation_checker
        self._clock = clock
        self.max_candidates = max_candidates
        self.max_blob_bytes = max_blob_bytes
        self.max_batch_items = max_batch_items
        self.timeout_seconds = float(timeout_seconds)

    def _candidate_verifier(self) -> Any:
        if self._verifier is not None:
            return self._verifier
        provider = self._certificate_provider
        if provider is None:
            return None
        adapter = getattr(provider, "as_cache_verifier", None)
        if callable(adapter):
            return adapter()
        verify = getattr(provider, "verify", None)
        if callable(verify):
            # TestProofCache accepts verifier objects but requires exact True.
            return provider
        return None

    def _candidates(self, locator: TestLocatorKey) -> Any:
        store = self._candidate_store
        if store is None:
            return None
        method = None
        for name in (
            "lookup_test_candidates",
            "lookup_candidates",
            "lookup",
            "candidate_provider",
        ):
            candidate = getattr(store, name, None)
            if callable(candidate):
                method = candidate
                break
        if method is None and callable(store):
            method = store
        if method is None:
            raise TypeError("candidate store is unsupported")
        return _materialize_store_result(
            _call_candidate_method(
                method,
                locator,
                max_candidates=self.max_candidates,
            )
        )

    def _lookup_unbounded(
        self,
        locator: TestLocatorKey,
        execution_key: TestExecutionKey,
        *,
        current_policy: Mapping[str, Any] | None,
        now_ms: int | None,
    ) -> ReuseDecision:
        candidates = self._candidates(locator)
        cache = TestProofCache(
            verifier=self._candidate_verifier(),
            policy_provider=self._policy_provider,
            revocation_checker=self._revocation_checker,
            current_policy=self._current_policy,
            clock=self._clock,
            max_candidates=self.max_candidates,
            max_blob_bytes=self.max_blob_bytes,
        )
        result = cache.lookup(
            locator,
            execution_key,
            candidates=candidates,
            current_policy=current_policy,
            now_ms=now_ms,
        )
        decision = result.decision
        if (
            result.status is TestProofCacheLookupStatus.HIT
            and result.admission is not None
            and result.admission.authoritative
            and isinstance(decision, ReuseDecision)
            and decision.is_skip
        ):
            # Reconstruct at the orchestration boundary to re-enforce the
            # closed RUN/SKIP contract before it reaches pytest.
            return ReuseDecision.from_dict(decision.to_dict())
        if isinstance(decision, ReuseDecision) and decision.is_run:
            return ReuseDecision.from_dict(decision.to_dict())
        return reuse_run(ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN)

    def lookup(
        self,
        locator: Any,
        execution_key: Any,
        *,
        eligibility: Any = None,
        current_policy: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> ReuseDecision:
        """Return SKIP only after exact, current, authoritative verification."""

        try:
            current_locator, rejected = _normalise_locator(locator)
            if rejected is not None:
                return rejected
            current_execution_key, rejected = _normalise_execution_key(execution_key)
            if rejected is not None:
                return rejected
            assert current_locator is not None and current_execution_key is not None
            if current_execution_key.locator_cid != current_locator.locator_id:
                return reuse_run(ReuseReasonCode.EXECUTION_KEY_MISMATCH)
            rejected = _eligibility_decision(eligibility, current_execution_key)
            if rejected is not None:
                return rejected
            return _bounded_call(
                lambda: self._lookup_unbounded(
                    current_locator,
                    current_execution_key,
                    current_policy=current_policy,
                    now_ms=now_ms,
                ),
                self.timeout_seconds,
            )
        except BaseException as exc:
            return _run_for_exception(exc, stage="lookup")

    def batch_lookup(
        self,
        requests: Iterable[Any],
        *,
        apply_skips: bool = True,
    ) -> tuple[ReuseDecision, ...]:
        return batch_lookup_reuse_decisions(
            self,
            requests,
            apply_skips=apply_skips,
        )


def _request_from_value(value: Any) -> ProofReuseLookupRequest | None:
    if isinstance(value, RevalidatedProofReuseLookupRequest):
        return value.to_lookup_request()
    if isinstance(value, ProofReuseLookupRequest):
        return value
    if isinstance(value, Mapping):
        return ProofReuseLookupRequest(
            item=value.get("item"),
            locator=value.get("locator"),
            execution_key=value.get("execution_key"),
            eligibility=value.get("eligibility"),
            current_policy=value.get("current_policy"),
            now_ms=value.get("now_ms"),
        )
    if isinstance(value, (tuple, list)) and 3 <= len(value) <= 6:
        padded = list(value) + [None] * (6 - len(value))
        return ProofReuseLookupRequest(*padded)
    attached = getattr(value, ITEM_LOOKUP_REQUEST_ATTRIBUTE, None)
    if isinstance(attached, RevalidatedProofReuseLookupRequest):
        converted = attached.to_lookup_request()
        if converted.item is None:
            return ProofReuseLookupRequest(
                item=value,
                locator=converted.locator,
                execution_key=converted.execution_key,
                eligibility=converted.eligibility,
                current_policy=converted.current_policy,
                now_ms=converted.now_ms,
            )
        return converted
    if isinstance(attached, ProofReuseLookupRequest):
        if attached.item is None:
            return ProofReuseLookupRequest(
                item=value,
                locator=attached.locator,
                execution_key=attached.execution_key,
                eligibility=attached.eligibility,
                current_policy=attached.current_policy,
                now_ms=attached.now_ms,
            )
        return attached
    locator = getattr(value, "_ipfs_proof_reuse_locator", None)
    execution_key = getattr(value, "_ipfs_proof_reuse_execution_key", None)
    if locator is not None or execution_key is not None:
        return ProofReuseLookupRequest(
            item=value,
            locator=locator,
            execution_key=execution_key,
            eligibility=getattr(value, "_ipfs_proof_reuse_eligibility", None),
            current_policy=getattr(value, "_ipfs_proof_reuse_policy", None),
        )
    return None


def _attach_decision(item: Any, decision: ReuseDecision) -> None:
    if item is None:
        return
    try:
        setattr(item, ITEM_DECISION_ATTRIBUTE, decision)
    except BaseException:
        return
    properties = getattr(item, "user_properties", None)
    if not isinstance(properties, list):
        return
    retained = [
        entry
        for entry in properties
        if not (
            isinstance(entry, tuple)
            and len(entry) == 2
            and entry[0] in _USER_PROPERTY_KEYS
        )
    ]
    retained.extend(
        (
            ("proof_reuse_action", decision.action.value),
            ("proof_reuse_reason", decision.reason_code.value),
        )
    )
    if decision.is_skip:
        retained.extend(
            (
                ("proof_reuse_certificate_cid", decision.certificate_cid),
                ("proof_reuse_receipt_cid", decision.receipt_cid),
            )
        )
    try:
        properties[:] = retained
    except BaseException:
        pass


def apply_verified_skip(item: Any, decision: Any) -> bool:
    """Attach a standard pytest skip marker for a revalidated proof hit only."""

    if not isinstance(decision, ReuseDecision):
        safe_decision = reuse_run(
            ReuseReasonCode.MALFORMED_ARTIFACT,
            diagnostics={"decision_type": _bounded_type_name(decision)},
        )
        _attach_decision(item, safe_decision)
        return False
    try:
        safe_decision = ReuseDecision.from_dict(decision.to_dict())
    except BaseException as exc:
        _attach_decision(item, _run_for_exception(exc, stage="apply_skip"))
        return False
    _attach_decision(item, safe_decision)
    if not (
        safe_decision.is_skip
        and safe_decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
        and safe_decision.authority is CertificateAuthority.AUTHORITATIVE
        and safe_decision.certificate_cid
        and safe_decision.receipt_cid
    ):
        return False
    reason = f"{SKIP_REASON_PREFIX}{safe_decision.certificate_cid}"
    try:
        import pytest

        item.add_marker(pytest.mark.skip(reason=reason))
    except BaseException:
        # A malformed/unsupported pytest item or marker failure must execute.
        _attach_decision(
            item,
            reuse_run(
                ReuseReasonCode.UNSUPPORTED,
                diagnostics={"stage": "apply_skip"},
            ),
        )
        return False
    return True


def batch_lookup_reuse_decisions(
    lookup_or_requests: ProofReuseLookup | Iterable[Any],
    requests: Iterable[Any] | None = None,
    *,
    lookup: ProofReuseLookup | None = None,
    apply_skips: bool = True,
) -> tuple[ReuseDecision, ...]:
    """Evaluate a bounded collection batch and attach decisions to its items.

    Both ``batch_lookup_reuse_decisions(lookup, requests)`` and
    ``batch_lookup_reuse_decisions(requests, lookup=lookup)`` are supported.
    Items beyond the configured batch bound are never looked up or skipped.
    """

    if isinstance(lookup_or_requests, ProofReuseLookup):
        service = lookup_or_requests
        request_values = requests
        if lookup is not None and lookup is not service:
            raise ValueError("conflicting lookup services")
    else:
        service = lookup
        request_values = lookup_or_requests
        if requests is not None:
            raise TypeError("requests were supplied twice")
    if not isinstance(service, ProofReuseLookup):
        raise TypeError("lookup must be a ProofReuseLookup")
    if request_values is None:
        raise TypeError("requests are required")

    decisions: list[ReuseDecision] = []
    try:
        iterator = iter(request_values)
    except BaseException as exc:
        return (_run_for_exception(exc, stage="batch_iter"),)

    for index in range(service.max_batch_items + 1):
        try:
            value = next(iterator)
        except StopIteration:
            break
        except BaseException as exc:
            decisions.append(_run_for_exception(exc, stage="batch_next"))
            break
        request = _request_from_value(value)
        item = request.item if request is not None else value
        if index >= service.max_batch_items:
            decision = reuse_run(ReuseReasonCode.OVER_BUDGET)
        elif request is None:
            decision = reuse_run(
                ReuseReasonCode.UNSUPPORTED,
                diagnostics={"item_type": _bounded_type_name(value)},
            )
        else:
            if isinstance(service, ProofReuseTwoStageLookup):
                decision = service.lookup(
                    request.locator,
                    request.execution_key,
                    eligibility=request.eligibility,
                    current_policy=request.current_policy,
                    now_ms=request.now_ms,
                    item=item,
                )
            else:
                decision = service.lookup(
                    request.locator,
                    request.execution_key,
                    eligibility=request.eligibility,
                    current_policy=request.current_policy,
                    now_ms=request.now_ms,
                )
        if apply_skips:
            apply_verified_skip(item, decision)
            attached = getattr(item, ITEM_DECISION_ATTRIBUTE, decision)
            if isinstance(attached, ReuseDecision):
                decision = attached
        else:
            _attach_decision(item, decision)
        decisions.append(decision)
    return tuple(decisions)


# ---------------------------------------------------------------------------
# Signed-receipt trust (PTR-164)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedReceiptTrustResult:
    """Outcome of pre-proof signed-receipt trust admission.

    Never authorizes ``SKIP`` by itself.  A verified result only permits the
    caller to proceed to local proof verification.
    """

    verified: bool
    reason: str
    signed_receipt: Any = None
    checks: Mapping[str, bool] = None  # type: ignore[assignment]
    diagnostics: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checks",
            dict(self.checks or {}),
        )
        object.__setattr__(
            self,
            "diagnostics",
            dict(self.diagnostics or {}),
        )

    @property
    def interface(self) -> str:
        return SIGNED_RECEIPT_TRUST_VERIFIER_INTERFACE

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def may_proceed_to_proof_verification(self) -> bool:
        return bool(self.verified)


def _mapping_get(value: Any, *names: str) -> Any:
    if value is None:
        return None
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        attr = getattr(value, name, None)
        if attr is not None:
            return attr
    return None


def _as_bytes(value: Any, *, max_bytes: int = MAX_ATTESTATION_BYTES) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return raw if 0 < len(raw) <= max_bytes else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Accept raw utf-8 or lowercase hex of even length.
        try:
            if len(text) % 2 == 0 and all(
                ch in "0123456789abcdefABCDEF" for ch in text
            ):
                raw = bytes.fromhex(text)
                if 0 < len(raw) <= max_bytes:
                    return raw
        except ValueError:
            pass
        encoded = text.encode("utf-8")
        return encoded if 0 < len(encoded) <= max_bytes else None
    canonical = getattr(value, "canonical_bytes", None)
    if callable(canonical):
        try:
            raw = bytes(canonical())
            return raw if 0 < len(raw) <= max_bytes else None
        except Exception:
            return None
    return None


def _load_receipt_from_material(value: Any) -> Any | None:
    from ...agent_supervisor.proof.test_execution_contracts import TestPassReceipt

    if isinstance(value, TestPassReceipt):
        return value
    raw = _as_bytes(value)
    if raw is not None:
        try:
            text = raw.decode("utf-8")
            payload = json.loads(text)
            if isinstance(payload, Mapping):
                return TestPassReceipt.from_dict(payload)
        except Exception:
            return None
    if isinstance(value, Mapping):
        try:
            return TestPassReceipt.from_dict(value)
        except Exception:
            return None
    return None


class SignedReceiptTrustVerifier:
    """Fail-closed pre-proof trust gate for warm lookup (PTR-164).

    Authority order before any ZK / certificate proof verification:

    1. Immutable receipt and attestation bytes rehash to their CIDs.
    2. Domain-separated Ed25519 signature verifies under the pinned key.
    3. Key material is valid for ``pytest-pass-attestation`` usage.
    4. Key is not revoked and matches the active epoch.
    5. Local trust-policy CID, trust domain, and epoch agree.
    6. Only then may the caller proceed to proof verification.

    Any gap returns a non-verified result.  This class never authorizes skip.
    """

    interface = SIGNED_RECEIPT_TRUST_VERIFIER_INTERFACE

    def __init__(
        self,
        *,
        trust_policy: Any = None,
        pinned_policy_cid: str = "",
        pinned_public_key_material: bytes | None = None,
        nonce_registry: Any = None,
        clock: Callable[[], int] | None = None,
        require_attestation: bool = False,
    ) -> None:
        self._trust_policy = trust_policy
        self._pinned_policy_cid = str(pinned_policy_cid or "")
        self._pinned_public_key_material = (
            bytes(pinned_public_key_material)
            if pinned_public_key_material is not None
            else None
        )
        self._nonce_registry = nonce_registry
        self._clock = clock
        self._require_attestation = bool(require_attestation)

    @property
    def may_authorize_skip(self) -> bool:
        return False

    @property
    def require_attestation(self) -> bool:
        return self._require_attestation

    def verify(
        self,
        *,
        receipt: Any = None,
        receipt_bytes: Any = None,
        attestation: Any = None,
        attestation_bytes: Any = None,
        current_execution_key_cid: str = "",
        current_candidate_context_cid: str = "",
        now: int | None = None,
    ) -> SignedReceiptTrustResult:
        """Verify signed-receipt trust; never raises into the lookup path."""

        checks = {
            "immutable_bytes": False,
            "signature": False,
            "key_validity": False,
            "revocation": False,
            "epoch": False,
            "policy": False,
        }
        diagnostics: dict[str, Any] = {"stage": "signed_receipt_trust"}

        try:
            return self._verify_unbounded(
                receipt=receipt,
                receipt_bytes=receipt_bytes,
                attestation=attestation,
                attestation_bytes=attestation_bytes,
                current_execution_key_cid=current_execution_key_cid,
                current_candidate_context_cid=current_candidate_context_cid,
                now=now,
                checks=checks,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            diagnostics["exception_type"] = _bounded_type_name(exc)
            return SignedReceiptTrustResult(
                False,
                "signed_receipt_trust_exception",
                checks=checks,
                diagnostics=diagnostics,
            )

    def _verify_unbounded(
        self,
        *,
        receipt: Any,
        receipt_bytes: Any,
        attestation: Any,
        attestation_bytes: Any,
        current_execution_key_cid: str,
        current_candidate_context_cid: str,
        now: int | None,
        checks: dict[str, bool],
        diagnostics: dict[str, Any],
    ) -> SignedReceiptTrustResult:
        from .runner_pass_attestation import (
            RunnerPassAttestation,
            RunnerTrustPolicy,
            verify_runner_pass_attestation,
            verify_runner_pass_attestation_with_key,
        )

        policy = self._trust_policy
        if policy is None:
            return SignedReceiptTrustResult(
                False,
                "trust_policy_unavailable",
                checks=checks,
                diagnostics=diagnostics,
            )
        if not isinstance(policy, RunnerTrustPolicy):
            # Accept already-validated policy-like objects that expose cid.
            if not hasattr(policy, "cid") or not hasattr(policy, "key_for"):
                return SignedReceiptTrustResult(
                    False,
                    "trust_policy_invalid",
                    checks=checks,
                    diagnostics=diagnostics,
                )

        typed_receipt = _load_receipt_from_material(receipt)
        if typed_receipt is None:
            typed_receipt = _load_receipt_from_material(receipt_bytes)
        if typed_receipt is None:
            return SignedReceiptTrustResult(
                False,
                "receipt_material_missing",
                checks=checks,
                diagnostics=diagnostics,
            )

        receipt_raw = _as_bytes(receipt_bytes) or _as_bytes(typed_receipt)
        if receipt_raw is None:
            return SignedReceiptTrustResult(
                False,
                "receipt_bytes_missing",
                checks=checks,
                diagnostics=diagnostics,
            )
        # Immutable receipt bytes must rehash to the typed receipt CID.
        try:
            if typed_receipt.canonical_bytes() != receipt_raw:
                return SignedReceiptTrustResult(
                    False,
                    "receipt_immutable_bytes_mismatch",
                    checks=checks,
                    diagnostics=diagnostics,
                )
        except Exception:
            return SignedReceiptTrustResult(
                False,
                "receipt_immutable_bytes_error",
                checks=checks,
                diagnostics=diagnostics,
            )
        checks["immutable_bytes"] = True

        att_raw = _as_bytes(attestation_bytes) or _as_bytes(attestation)
        if att_raw is None and attestation is not None:
            # Already-decoded attestation object path.
            if isinstance(attestation, RunnerPassAttestation):
                try:
                    att_raw = attestation.canonical_bytes()
                except Exception:
                    att_raw = None
        if att_raw is None:
            return SignedReceiptTrustResult(
                False,
                "attestation_material_missing",
                checks=checks,
                diagnostics=diagnostics,
            )

        try:
            if isinstance(attestation, RunnerPassAttestation):
                candidate_att = attestation
                if candidate_att.canonical_bytes() != att_raw:
                    return SignedReceiptTrustResult(
                        False,
                        "attestation_immutable_bytes_mismatch",
                        checks=checks,
                        diagnostics=diagnostics,
                    )
            else:
                candidate_att = RunnerPassAttestation.from_bytes(att_raw)
        except Exception:
            return SignedReceiptTrustResult(
                False,
                "attestation_malformed",
                checks=checks,
                diagnostics=diagnostics,
            )

        pinned_policy_cid = self._pinned_policy_cid or str(
            getattr(policy, "cid", "") or ""
        )
        if not pinned_policy_cid:
            return SignedReceiptTrustResult(
                False,
                "pinned_policy_cid_missing",
                checks=checks,
                diagnostics=diagnostics,
            )

        execution_key_cid = str(
            current_execution_key_cid
            or getattr(typed_receipt, "execution_key_cid", "")
            or ""
        )
        candidate_context_cid = str(
            current_candidate_context_cid
            or getattr(candidate_att, "candidate_context_cid", "")
            or ""
        )
        if not execution_key_cid or not candidate_context_cid:
            return SignedReceiptTrustResult(
                False,
                "trust_context_cids_missing",
                checks=checks,
                diagnostics=diagnostics,
            )

        current_time = (
            int(now)
            if now is not None
            else (int(self._clock()) if self._clock is not None else None)
        )

        if self._pinned_public_key_material is not None:
            verification = verify_runner_pass_attestation_with_key(
                candidate_att,
                receipt=typed_receipt,
                policy=policy,
                pinned_policy_cid=pinned_policy_cid,
                current_execution_key_cid=execution_key_cid,
                current_candidate_context_cid=candidate_context_cid,
                pinned_public_key_material=self._pinned_public_key_material,
                now=current_time,
                nonce_registry=self._nonce_registry,
            )
        else:
            verification = verify_runner_pass_attestation(
                candidate_att,
                receipt=typed_receipt,
                policy=policy,
                pinned_policy_cid=pinned_policy_cid,
                current_execution_key_cid=execution_key_cid,
                current_candidate_context_cid=candidate_context_cid,
                now=current_time,
                nonce_registry=self._nonce_registry,
            )

        if not getattr(verification, "valid", False):
            reason = str(getattr(verification, "reason", "") or "attestation_rejected")
            lowered = reason.lower()
            if "revok" in lowered:
                checks["revocation"] = False
            if "epoch" in lowered:
                checks["epoch"] = False
            if "policy" in lowered or "trust domain" in lowered:
                checks["policy"] = False
            if "signature" in lowered or "ed25519" in lowered:
                checks["signature"] = False
            if "key" in lowered:
                checks["key_validity"] = False
            diagnostics["attestation_reason"] = reason[:MAX_DIAGNOSTIC_TEXT]
            return SignedReceiptTrustResult(
                False,
                "signed_receipt_trust_rejected",
                checks=checks,
                diagnostics=diagnostics,
            )

        # Successful path: mark every trust facet checked by the attestation
        # verifier (policy.key_for covers validity/epoch/revocation).
        checks["signature"] = True
        checks["key_validity"] = True
        checks["revocation"] = True
        checks["epoch"] = True
        checks["policy"] = True
        signed = getattr(verification, "signed_receipt", None)
        return SignedReceiptTrustResult(
            True,
            "verified",
            signed_receipt=signed,
            checks=checks,
            diagnostics=diagnostics,
        )


def build_signed_receipt_trust_verifier(
    *,
    trust_policy: Any = None,
    pinned_policy_cid: str = "",
    pinned_public_key_material: bytes | None = None,
    nonce_registry: Any = None,
    clock: Callable[[], int] | None = None,
    require_attestation: bool = False,
) -> SignedReceiptTrustVerifier:
    """Factory for the production pre-proof signed-receipt trust verifier."""

    return SignedReceiptTrustVerifier(
        trust_policy=trust_policy,
        pinned_policy_cid=pinned_policy_cid,
        pinned_public_key_material=pinned_public_key_material,
        nonce_registry=nonce_registry,
        clock=clock,
        require_attestation=require_attestation,
    )


def _extract_attestation_material(
    candidate: Any,
    component_bytes: Mapping[str, bytes] | None = None,
) -> tuple[Any | None, bytes | None, Any | None, bytes | None]:
    """Pull receipt/attestation material from retained candidate surfaces."""

    components = dict(component_bytes or {})
    receipt_material = (
        components.get("pass_receipt")
        or components.get("receipt")
        or _mapping_get(candidate, "receipt_bytes", "pass_receipt", "receipt")
    )
    attestation_material = (
        components.get("runner_attestation")
        or components.get("attestation")
        or _mapping_get(
            candidate,
            "attestation_bytes",
            "runner_attestation_bytes",
            "runner_attestation",
            "attestation",
        )
    )
    # Certificate metadata may retain public attestation linkage only.
    metadata = _mapping_get(candidate, "metadata")
    if attestation_material is None and isinstance(metadata, Mapping):
        attestation_material = (
            metadata.get("runner_attestation_bytes")
            or metadata.get("attestation_bytes")
            or metadata.get("runner_attestation")
        )
    certificate = _mapping_get(candidate, "certificate")
    if attestation_material is None and certificate is not None:
        cert_meta = _mapping_get(certificate, "metadata")
        if isinstance(cert_meta, Mapping):
            attestation_material = (
                cert_meta.get("runner_attestation_bytes")
                or cert_meta.get("attestation_bytes")
            )
    return receipt_material, _as_bytes(receipt_material), attestation_material, _as_bytes(
        attestation_material
    )


def _map_revalidation_to_reuse_reason(reason: Any) -> ReuseReasonCode:
    """Map RuntimeContextRevalidator reasons onto closed ReuseReasonCode values."""

    try:
        from .runtime_revalidation import (
            RevalidationReason,
            map_revalidation_reason_to_reuse_code,
        )

        if isinstance(reason, RevalidationReason):
            code = map_revalidation_reason_to_reuse_code(reason)
        else:
            code = map_revalidation_reason_to_reuse_code(str(reason))
        try:
            mapped = ReuseReasonCode(code)
        except ValueError:
            mapped = ReuseReasonCode.UNKNOWN
    except Exception:
        mapped = ReuseReasonCode.UNKNOWN
    if mapped is ReuseReasonCode.PROOF_CACHE_HIT:
        return ReuseReasonCode.UNSUPPORTED
    return mapped


def _execution_key_from_component_bytes(
    component_bytes: Mapping[str, bytes] | None,
    *,
    claimed_cid: str = "",
) -> TestExecutionKey | None:
    """Decode a retained execution-key component when present and well-formed."""

    if not component_bytes:
        return None
    raw = component_bytes.get("execution_key")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    try:
        key = TestExecutionKey.from_dict(payload)
    except Exception:
        return None
    if claimed_cid and key.execution_key_id != claimed_cid:
        return None
    return key


def _execution_key_from_candidate_fields(
    candidate: Any,
    locator: TestLocatorKey,
) -> TestExecutionKey | None:
    """Synthesize a minimal current execution key from admitted candidate CIDs.

    Used only after revalidation has confirmed exact identity agreement.  The
    key is the identity surface the certificate cache admits against; it is
    never skip authority by itself.
    """

    try:
        return TestExecutionKey(
            locator_cid=locator.locator_id,
            repository_forest_cid=str(
                getattr(candidate, "repository_forest_cid", "") or ""
            ),
            test_ast_cid=str(getattr(candidate, "test_ast_cid", "") or ""),
            static_trace_root_cid=str(
                getattr(candidate, "static_trace_root_cid", "") or ""
            ),
            runtime_trace_root_cid=str(
                getattr(candidate, "runtime_trace_root_cid", "") or ""
            ),
            runtime_completeness_policy="complete-v1",
            environment_cid=str(getattr(candidate, "environment_cid", "") or ""),
            policy_cid=str(getattr(candidate, "policy_cid", "") or ""),
            dependency_lock_cid=str(
                getattr(candidate, "dependency_lock_cid", "") or ""
            ),
            installed_distributions_cid=str(
                getattr(candidate, "installed_distributions_cid", "") or ""
            ),
            platform_cid=str(getattr(candidate, "platform_cid", "") or ""),
            hardware_capability_cid=str(
                getattr(candidate, "capability_root_cid", "") or ""
            ),
            external_snapshot_cids=tuple(
                getattr(candidate, "external_snapshot_cids", ()) or ()
            ),
        )
    except Exception:
        return None


class ProofReuseTwoStageLookup(ProofReuseLookup):
    """Locator-first warm lookup with signed-receipt trust before proof cache.

    Implements ``TwoStageCandidateLookup@2`` (PTR-164).

    Authority sequence:

    1. Accept locator (+ optional collected item) only.
    2. Load retained candidate bytes from a dedicated
       ``TestCandidateContextStore`` and rehash every component.
    3. Resolve the retained runtime frontier against admitted live roots.
    4. Rebuild current AST/static/fixtures/hooks/parameters/forest/locks/
       distributions/environment/capabilities/snapshots/policy without
       fixture or test execution.
    5. Require the final current execution key to match the candidate.
    6. When signed-receipt material or a trust verifier is present, check
       immutable bytes, signature, key validity, revocation, epoch and
       policy **before** local proof verification.
    7. Only then admit through the certificate proof cache.
    8. Revalidation or trust alone never returns ``SKIP``.
    9. Every miss, mismatch, unknown, timeout, corruption, provider absence,
       or exception returns ``RUN``.
    """

    interface = TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE

    def __init__(
        self,
        candidate_context_store: Any = None,
        certificate_provider: Any = None,
        *,
        proof_cache_store: Any = None,
        store: Any = None,
        provider: Any = None,
        revalidator: Any = None,
        current_context_provider: Any = None,
        identity_services: Any = None,
        live_identity_compiler: Callable[..., Any] | None = None,
        allowed_roots: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
        verifier: Any = None,
        current_policy: Mapping[str, Any] | None = None,
        policy_provider: Callable[
            [TestLocatorKey, TestExecutionKey], Mapping[str, Any]
        ]
        | None = None,
        revocation_checker: Callable[..., Any] | None = None,
        signed_receipt_trust_verifier: Any = None,
        trust_policy: Any = None,
        clock: Callable[[], int] | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS,
        timeout_seconds: float = DEFAULT_LOOKUP_TIMEOUT_SECONDS,
        require_runtime_frontier: bool = True,
    ) -> None:
        # Stage-2 certificate candidates use proof_cache_store / store.
        # The dedicated candidate-context store is stage-1 only.
        if proof_cache_store is not None and store is not None:
            raise ValueError("specify proof_cache_store or store, not both")
        cert_store = proof_cache_store if store is None else store
        super().__init__(
            candidate_store=cert_store,
            certificate_provider=certificate_provider,
            provider=provider,
            verifier=verifier,
            current_policy=current_policy,
            policy_provider=policy_provider,
            revocation_checker=revocation_checker,
            clock=clock,
            max_candidates=max_candidates,
            max_blob_bytes=max_blob_bytes,
            max_batch_items=max_batch_items,
            timeout_seconds=timeout_seconds,
        )
        self._candidate_context_store = candidate_context_store
        self._current_context_provider = current_context_provider
        self._identity_services = identity_services
        self._live_identity_compiler = live_identity_compiler
        self._allowed_roots = dict(allowed_roots or {})
        self._environ = environ
        self._require_runtime_frontier = bool(require_runtime_frontier)
        self._revalidator = revalidator
        self._revalidator_lock = threading.RLock()
        self._signed_receipt_trust_verifier = signed_receipt_trust_verifier
        self._trust_policy = trust_policy

    @property
    def may_authorize_skip_from_revalidation_alone(self) -> bool:
        return False

    @property
    def candidate_context_store(self) -> Any:
        return self._candidate_context_store

    @property
    def current_context_provider(self) -> Any:
        return self._current_context_provider

    @property
    def signed_receipt_trust_verifier(self) -> Any:
        return self._signed_receipt_trust_verifier

    def _ensure_trust_verifier(self) -> Any | None:
        if self._signed_receipt_trust_verifier is not None:
            return self._signed_receipt_trust_verifier
        if self._trust_policy is None:
            return None
        self._signed_receipt_trust_verifier = build_signed_receipt_trust_verifier(
            trust_policy=self._trust_policy,
            clock=(
                (lambda: int(self._clock() // 1000))
                if self._clock is not None
                else None
            ),
        )
        return self._signed_receipt_trust_verifier

    def _apply_signed_receipt_trust(
        self,
        *,
        revalidation: Any,
        current_locator: TestLocatorKey,
        verified_key: TestExecutionKey,
        now_ms: int | None,
    ) -> ReuseDecision | None:
        """Run pre-proof trust when material or a verifier is present.

        Returns ``None`` when trust is not applicable (no verifier and no
        attestation material) or when trust verified successfully so certificate
        admission may proceed.  Returns a ``RUN`` decision on any trust gap.
        """

        trust_verifier = self._ensure_trust_verifier()
        component_bytes = getattr(revalidation, "component_bytes", None)
        candidate = getattr(revalidation, "candidate", None)
        (
            receipt_material,
            receipt_raw,
            attestation_material,
            attestation_raw,
        ) = _extract_attestation_material(candidate, component_bytes)

        # Certificate-store candidates may retain public attestation bytes.
        if attestation_raw is None and self._candidate_store is not None:
            try:
                store_candidates = self._candidates(current_locator)
                if store_candidates:
                    for entry in store_candidates:
                        _rm, _rr, _am, ar = _extract_attestation_material(entry)
                        if ar is not None:
                            attestation_material = _am
                            attestation_raw = ar
                            if receipt_raw is None and _rr is not None:
                                receipt_material = _rm
                                receipt_raw = _rr
                            break
            except Exception:
                pass

        require = bool(
            trust_verifier is not None
            and getattr(trust_verifier, "require_attestation", False)
        )
        has_material = attestation_raw is not None or attestation_material is not None
        if trust_verifier is None and not has_material:
            return None
        if trust_verifier is None and has_material:
            return reuse_run(
                ReuseReasonCode.TRUST_POLICY_REJECTED,
                diagnostics={
                    "stage": "signed_receipt_trust",
                    "reason": "trust_verifier_unavailable_with_material",
                },
            )
        if trust_verifier is not None and not has_material:
            if require:
                return reuse_run(
                    ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN,
                    diagnostics={
                        "stage": "signed_receipt_trust",
                        "reason": "attestation_material_required",
                    },
                )
            # Soft mode: no material means trust is not yet claimable; still
            # allow certificate stage which independently fails open.
            return None

        candidate_context_cid = str(
            getattr(candidate, "candidate_context_cid", "")
            or getattr(candidate, "content_id", "")
            or getattr(candidate, "cid", "")
            or ""
        )
        now_s = None
        if now_ms is not None:
            try:
                now_s = int(now_ms) // 1000
            except Exception:
                now_s = None
        trust = trust_verifier.verify(
            receipt=receipt_material,
            receipt_bytes=receipt_raw,
            attestation=attestation_material,
            attestation_bytes=attestation_raw,
            current_execution_key_cid=verified_key.execution_key_id,
            current_candidate_context_cid=candidate_context_cid,
            now=now_s,
        )
        if not getattr(trust, "verified", False):
            reason = str(getattr(trust, "reason", "") or "signed_receipt_trust_rejected")
            code = ReuseReasonCode.TRUST_POLICY_REJECTED
            lowered = reason.lower()
            if "revok" in lowered or "expired" in lowered:
                code = ReuseReasonCode.EXPIRED_OR_REVOKED
            elif "malform" in lowered or "bytes" in lowered:
                code = ReuseReasonCode.MALFORMED_ARTIFACT
            elif "missing" in lowered or "unavailable" in lowered:
                code = ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN
            return reuse_run(
                code,
                diagnostics={
                    "stage": "signed_receipt_trust",
                    "reason": reason[:MAX_DIAGNOSTIC_TEXT],
                    "checks": dict(getattr(trust, "checks", {}) or {}),
                },
            )
        # Verified: proceed to proof verification (never skip from trust alone).
        return None

    def _ensure_revalidator(self) -> Any:
        if self._revalidator is not None:
            return self._revalidator
        with self._revalidator_lock:
            if self._revalidator is not None:
                return self._revalidator
            from .runtime_revalidation import build_runtime_context_revalidator

            provider = self._current_context_provider
            if provider is None and (
                self._identity_services is not None
                or self._live_identity_compiler is not None
            ):
                from .current_context_provider import (
                    build_default_current_context_provider,
                )

                provider = build_default_current_context_provider(
                    identity_services=self._identity_services,
                    live_identity_compiler=self._live_identity_compiler,
                    allowed_roots=self._allowed_roots,
                    environ=self._environ,
                    clock=self._clock,
                )
                self._current_context_provider = provider

            self._revalidator = build_runtime_context_revalidator(
                candidate_store=self._candidate_context_store,
                current_context_provider=provider,
                allowed_roots=self._allowed_roots,
                environ=self._environ,
                clock=self._clock,
                require_runtime_frontier=self._require_runtime_frontier,
            )
            return self._revalidator

    def revalidate_only(
        self,
        locator: Any,
        *,
        item: Any = None,
        now_ms: int | None = None,
        max_candidates: int | None = None,
    ) -> Any:
        """Run stage-1 revalidation only (never authorizes SKIP)."""

        from .runtime_revalidation import (
            RevalidationAction,
            RuntimeRevalidationResult,
            revalidation_result_to_run_decision,
        )

        try:
            revalidator = self._ensure_revalidator()
            provider = self._current_context_provider
            if provider is not None and item is not None:
                bind = getattr(provider, "bind_collected_item", None)
                if callable(bind):
                    bind(item)
            result = revalidator.revalidate(
                locator,
                max_candidates=max_candidates or self.max_candidates,
                now_ms=now_ms,
            )
            if not isinstance(result, RuntimeRevalidationResult):
                return result
            # Fence: revalidation action must never be treated as skip.
            assert result.may_authorize_skip is False
            if result.action is RevalidationAction.PROCEED_TO_CERTIFICATE_VERIFICATION:
                # Still not a skip — expose the proceed result for stage 2.
                return result
            return result
        except BaseException as exc:
            return _run_for_exception(exc, stage="revalidate_only")

    def lookup(
        self,
        locator: Any,
        execution_key: Any = None,
        *,
        eligibility: Any = None,
        current_policy: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
        item: Any = None,
    ) -> ReuseDecision:
        """Two-stage warm lookup: revalidate, then certificate cache.

        ``execution_key`` may be omitted for locator-first warm admission; when
        supplied it must match the retained candidate after fresh rebuild.
        """

        try:
            return _bounded_call(
                lambda: self._two_stage_lookup_unbounded(
                    locator,
                    execution_key,
                    eligibility=eligibility,
                    current_policy=current_policy,
                    now_ms=now_ms,
                    item=item,
                ),
                self.timeout_seconds,
            )
        except BaseException as exc:
            return _run_for_exception(exc, stage="two_stage_lookup")

    def _two_stage_lookup_unbounded(
        self,
        locator: Any,
        execution_key: Any,
        *,
        eligibility: Any,
        current_policy: Mapping[str, Any] | None,
        now_ms: int | None,
        item: Any,
    ) -> ReuseDecision:
        from .runtime_revalidation import (
            RevalidationAction,
            RevalidationReason,
            RuntimeRevalidationResult,
        )

        # --- Stage 0: normalize locator (execution key optional) ---
        current_locator, rejected = _normalise_locator(locator)
        if rejected is not None:
            return rejected
        assert current_locator is not None

        provided_execution_key: TestExecutionKey | None = None
        if execution_key is not None:
            provided_execution_key, rejected = _normalise_execution_key(execution_key)
            if rejected is not None:
                return rejected
            assert provided_execution_key is not None
            if provided_execution_key.locator_cid != current_locator.locator_id:
                return reuse_run(ReuseReasonCode.EXECUTION_KEY_MISMATCH)
            rejected = _eligibility_decision(eligibility, provided_execution_key)
            if rejected is not None:
                return rejected
        elif eligibility not in (None, True):
            # Without an execution key, only hard-false eligibility is checked.
            if eligibility is False:
                return reuse_run(ReuseReasonCode.ELIGIBILITY_DENIED)
            reusable = getattr(eligibility, "reusable", None)
            if reusable is False:
                return reuse_run(ReuseReasonCode.ELIGIBILITY_DENIED)

        # --- Stage 1: dedicated candidate-context store + fresh revalidation ---
        if self._candidate_context_store is None and self._revalidator is None:
            return reuse_run(
                ReuseReasonCode.CACHE_UNAVAILABLE,
                diagnostics={"stage": "candidate_context_store_absent"},
            )

        revalidator = self._ensure_revalidator()
        provider = self._current_context_provider
        if provider is not None and item is not None:
            bind = getattr(provider, "bind_collected_item", None)
            if callable(bind):
                try:
                    bind(item)
                except Exception as exc:
                    return _run_for_exception(exc, stage="bind_item")

        try:
            revalidation = revalidator.revalidate(
                current_locator,
                max_candidates=self.max_candidates,
                now_ms=now_ms,
            )
        except Exception as exc:
            return _run_for_exception(exc, stage="revalidate")

        if not isinstance(revalidation, RuntimeRevalidationResult):
            return reuse_run(
                ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                diagnostics={"stage": "revalidation_type"},
            )

        # Invariant: revalidation alone never authorizes skip.
        if revalidation.may_authorize_skip:
            return reuse_run(
                ReuseReasonCode.ILLEGAL_AUTHORITY,
                diagnostics={"stage": "revalidation_claimed_skip"},
            )

        if revalidation.action is not RevalidationAction.PROCEED_TO_CERTIFICATE_VERIFICATION:
            return reuse_run(
                _map_revalidation_to_reuse_reason(revalidation.reason),
                diagnostics={
                    "stage": "revalidation",
                    "revalidation_action": revalidation.action.value,
                    "revalidation_reason": revalidation.reason.value,
                    **{
                        key: value
                        for key, value in dict(revalidation.diagnostics).items()
                        if key
                        not in {
                            "stage",
                            "revalidation_action",
                            "revalidation_reason",
                        }
                    },
                },
            )

        if revalidation.reason is not RevalidationReason.CONTEXT_UNCHANGED:
            return reuse_run(
                ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                diagnostics={
                    "stage": "proceed_reason",
                    "revalidation_reason": revalidation.reason.value,
                },
            )

        candidate = revalidation.candidate
        current_context = revalidation.current
        if candidate is None or current_context is None:
            return reuse_run(
                ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN,
                diagnostics={"stage": "revalidation_missing_contexts"},
            )

        # Exact current execution key must match the candidate before proof
        # verification.
        if (
            not current_context.execution_key_cid
            or current_context.execution_key_cid != candidate.execution_key_cid
        ):
            return reuse_run(
                ReuseReasonCode.EXECUTION_KEY_MISMATCH,
                diagnostics={
                    "stage": "execution_key_match",
                    "candidate_execution_key_cid": candidate.execution_key_cid[
                        :MAX_DIAGNOSTIC_TEXT
                    ],
                    "current_execution_key_cid": current_context.execution_key_cid[
                        :MAX_DIAGNOSTIC_TEXT
                    ],
                },
            )

        # --- Stage 2: resolve the verified execution key for proof cache ---
        verified_key = provided_execution_key
        if verified_key is not None:
            if verified_key.execution_key_id != candidate.execution_key_cid:
                return reuse_run(
                    ReuseReasonCode.EXECUTION_KEY_MISMATCH,
                    diagnostics={
                        "stage": "provided_execution_key",
                        "provided": verified_key.execution_key_id[
                            :MAX_DIAGNOSTIC_TEXT
                        ],
                        "candidate": candidate.execution_key_cid[
                            :MAX_DIAGNOSTIC_TEXT
                        ],
                    },
                )
        else:
            verified_key = _execution_key_from_component_bytes(
                revalidation.component_bytes,
                claimed_cid=candidate.execution_key_cid,
            )
            if verified_key is None:
                verified_key = _execution_key_from_candidate_fields(
                    candidate, current_locator
                )
            if verified_key is None:
                return reuse_run(
                    ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN,
                    diagnostics={"stage": "execution_key_materialize"},
                )
            # When synthesizing from candidate fields, content id may differ
            # from the retained execution_key_cid unless the full key payload
            # is present.  Prefer component bytes; if only fields are available
            # require the synthesized key's id to match when possible, else
            # still proceed only when the revalidated current key matches.
            if (
                verified_key.execution_key_id != candidate.execution_key_cid
                and "execution_key" not in (revalidation.component_bytes or {})
            ):
                # Field-synthesized keys rarely rehash to the retained CID.
                # Stage-1 already confirmed current.execution_key_cid match;
                # build a key surface for the proof cache that carries the
                # verified CIDs and bind via the current context identity.
                # The proof cache still requires exact receipt/certificate
                # agreement on execution_key_cid — so without retained key
                # bytes we cannot invent authority.  Fail open to RUN.
                return reuse_run(
                    ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN,
                    diagnostics={
                        "stage": "execution_key_bytes_required",
                        "candidate_execution_key_cid": candidate.execution_key_cid[
                            :MAX_DIAGNOSTIC_TEXT
                        ],
                    },
                )

        if eligibility is not None and provided_execution_key is None:
            rejected = _eligibility_decision(eligibility, verified_key)
            if rejected is not None:
                return rejected

        # --- Stage 1.5: signed-receipt trust before proof verification ---
        trust_rejected = self._apply_signed_receipt_trust(
            revalidation=revalidation,
            current_locator=current_locator,
            verified_key=verified_key,
            now_ms=now_ms,
        )
        if trust_rejected is not None:
            return trust_rejected

        # Certificate-cache admission (authoritative stage).
        # When no proof-cache store/provider is configured, stage-1 success
        # still cannot skip.
        if (
            self._candidate_store is None
            and self._certificate_provider is None
            and self._verifier is None
        ):
            return reuse_run(
                ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
                diagnostics={
                    "stage": "certificate_stage",
                    "revalidation": "proceed",
                    "may_proceed_to_certificate_verification": True,
                    "signed_receipt_trust": "passed_or_not_applicable",
                },
            )

        return self._lookup_unbounded(
            current_locator,
            verified_key,
            current_policy=current_policy,
            now_ms=now_ms,
        )


def build_proof_reuse_two_stage_lookup(
    *,
    candidate_context_store: Any = None,
    certificate_provider: Any = None,
    proof_cache_store: Any = None,
    revalidator: Any = None,
    current_context_provider: Any = None,
    identity_services: Any = None,
    live_identity_compiler: Callable[..., Any] | None = None,
    allowed_roots: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    verifier: Any = None,
    current_policy: Mapping[str, Any] | None = None,
    policy_provider: Callable[
        [TestLocatorKey, TestExecutionKey], Mapping[str, Any]
    ]
    | None = None,
    revocation_checker: Callable[..., Any] | None = None,
    signed_receipt_trust_verifier: Any = None,
    trust_policy: Any = None,
    clock: Callable[[], int] | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS,
    timeout_seconds: float = DEFAULT_LOOKUP_TIMEOUT_SECONDS,
    require_runtime_frontier: bool = True,
) -> ProofReuseTwoStageLookup:
    """Factory for the production two-stage warm lookup service."""

    return ProofReuseTwoStageLookup(
        candidate_context_store=candidate_context_store,
        certificate_provider=certificate_provider,
        proof_cache_store=proof_cache_store,
        revalidator=revalidator,
        current_context_provider=current_context_provider,
        identity_services=identity_services,
        live_identity_compiler=live_identity_compiler,
        allowed_roots=allowed_roots,
        environ=environ,
        verifier=verifier,
        current_policy=current_policy,
        policy_provider=policy_provider,
        revocation_checker=revocation_checker,
        signed_receipt_trust_verifier=signed_receipt_trust_verifier,
        trust_policy=trust_policy,
        clock=clock,
        max_candidates=max_candidates,
        max_blob_bytes=max_blob_bytes,
        max_batch_items=max_batch_items,
        timeout_seconds=timeout_seconds,
        require_runtime_frontier=require_runtime_frontier,
    )


__all__ = [
    "DEFAULT_LOOKUP_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BATCH_ITEMS",
    "ITEM_DECISION_ATTRIBUTE",
    "ITEM_LOOKUP_REQUEST_ATTRIBUTE",
    "PROOF_REUSE_LOOKUP_INTERFACE",
    "PROOF_REUSE_TWO_STAGE_LOOKUP_INTERFACE",
    "SIGNED_RECEIPT_TRUST_VERIFIER_INTERFACE",
    "SKIP_REASON_PREFIX",
    "TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE",
    "ProofReuseLookup",
    "ProofReuseLookupRequest",
    "ProofReuseTwoStageLookup",
    "RevalidatedProofReuseLookupRequest",
    "SignedReceiptTrustResult",
    "SignedReceiptTrustVerifier",
    "apply_verified_skip",
    "batch_lookup_reuse_decisions",
    "build_proof_reuse_two_stage_lookup",
    "build_signed_receipt_trust_verifier",
]
