"""Bounded candidate lookup and pytest skip application.

The locator index is only a hint.  This module delegates immutable candidate
admission to :class:`TestProofCache`, which rehashes the retained bytes and
checks the exact current locator, execution key, policy, revocation state, and
local proof verifier.  Every optional-boundary failure is converted to an
explicit ``RUN`` decision.

No lookup path in this module calls a prover or an issuer handle.
"""

from __future__ import annotations

import inspect
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
ITEM_DECISION_ATTRIBUTE: Final = "_ipfs_proof_reuse_decision"
ITEM_LOOKUP_REQUEST_ATTRIBUTE: Final = "_ipfs_proof_reuse_lookup_request"
SKIP_REASON_PREFIX: Final = "proof-cache-hit:"

DEFAULT_LOOKUP_TIMEOUT_SECONDS: Final = 5.0
DEFAULT_MAX_BATCH_ITEMS: Final = 4096
MAX_DIAGNOSTIC_TEXT: Final = 128

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


__all__ = [
    "DEFAULT_LOOKUP_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BATCH_ITEMS",
    "ITEM_DECISION_ATTRIBUTE",
    "ITEM_LOOKUP_REQUEST_ATTRIBUTE",
    "PROOF_REUSE_LOOKUP_INTERFACE",
    "SKIP_REASON_PREFIX",
    "ProofReuseLookup",
    "ProofReuseLookupRequest",
    "apply_verified_skip",
    "batch_lookup_reuse_decisions",
]
