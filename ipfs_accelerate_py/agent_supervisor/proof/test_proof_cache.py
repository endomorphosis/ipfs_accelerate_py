"""Trust-aware admission for proof-backed test-reuse candidates.

``TestProofCache@1`` is deliberately an authority adapter, not a storage
implementation.  A locator index or remote cache may suggest immutable
receipt/certificate bytes, but every lookup:

* decodes retained canonical bytes and recomputes both CIDs;
* binds the receipt to the exact current locator and execution key;
* applies the caller's current issuer, epoch, revocation, circuit, key,
  statement, proof-system, and policy requirements; and
* invokes a local verifier.

No serialized ``trusted``/``authoritative`` flag or mutable index metadata is
an authority input.  Absence and ordinary failures produce typed ``RUN``
decisions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Final

from .test_execution_contracts import (
    CertificateAuthority,
    EligibilityClass,
    ProofBackendMode,
    ReuseDecision,
    ReuseReasonCode,
    TestExecutionContractError,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
    decision_from_absence,
    decision_from_exception,
    reuse_run,
    reuse_skip,
)

TEST_PROOF_CACHE_INTERFACE: Final = "TestProofCache@1"
DEFAULT_MAX_CANDIDATES: Final = 32
DEFAULT_MAX_BLOB_BYTES: Final = 1_048_576

_REQUIRED_POLICY_IDS: Final = (
    "policy_cid",
    "statement_cid",
    "circuit_cid",
    "verifying_key_cid",
    "proof_system_id",
)
_REQUIRED_PUBLIC_INPUTS: Final = (
    "receipt_cid",
    "execution_key_cid",
    "policy_cid",
    "statement_cid",
    "circuit_cid",
    "verifying_key_cid",
    "proof_system_id",
    "issuer_id",
    "issuer_key_id",
    "epoch",
    "setup_outcome",
    "call_outcome",
    "teardown_outcome",
)
_PRIVATE_MARKERS: Final = (
    "secret",
    "private",
    "password",
    "passwd",
    "token",
    "credential",
    "witness",
    "stdout",
    "stderr",
    "source_body",
)
_ERROR_REASONS: Final = frozenset(
    {
        ReuseReasonCode.CACHE_UNAVAILABLE,
        ReuseReasonCode.CID_PROVIDER_UNAVAILABLE,
        ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE,
        ReuseReasonCode.VERIFIER_UNAVAILABLE,
        ReuseReasonCode.KEY_UNAVAILABLE,
        ReuseReasonCode.CIRCUIT_UNAVAILABLE,
        ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
        ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN,
        ReuseReasonCode.TIMEOUT,
    }
)


class TestProofCacheLookupStatus(StrEnum):
    """Closed lookup result states."""

    HIT = "hit"
    MISS = "miss"
    ERROR = "error"

    __test__ = False


@dataclass(frozen=True)
class TestProofCacheAdmission:
    """Result of re-deriving authority for one immutable candidate."""

    __test__: ClassVar[bool] = False

    admitted: bool
    reason_code: ReuseReasonCode
    receipt: TestPassReceipt | None = None
    certificate: TestProofCertificate | None = None
    receipt_cid: str = ""
    certificate_cid: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.admitted

    @property
    def authoritative(self) -> bool:
        return bool(
            self.admitted and self.certificate is not None and self.certificate.can_authorize_skip
        )


@dataclass(frozen=True)
class TestProofCacheLookup:
    """Typed lookup outcome; only ``HIT`` can contain a ``SKIP`` decision."""

    __test__: ClassVar[bool] = False

    status: TestProofCacheLookupStatus
    decision: ReuseDecision
    admission: TestProofCacheAdmission | None = None
    reason_codes: tuple[ReuseReasonCode, ...] = ()
    candidates_considered: int = 0

    def __post_init__(self) -> None:
        if self.status is TestProofCacheLookupStatus.HIT:
            if self.admission is None or not self.admission.authoritative:
                raise ValueError("cache HIT requires an authoritative admission")
            if not self.decision.is_skip:
                raise ValueError("cache HIT requires a SKIP decision")
        elif not self.decision.is_run:
            raise ValueError("cache MISS/ERROR requires a RUN decision")

    def __bool__(self) -> bool:
        return self.hit

    @property
    def hit(self) -> bool:
        return self.status is TestProofCacheLookupStatus.HIT

    @property
    def reason_code(self) -> ReuseReasonCode:
        if self.reason_codes:
            return self.reason_codes[0]
        return self.decision.reason_code

    @property
    def receipt(self) -> TestPassReceipt | None:
        return self.admission.receipt if self.admission is not None else None

    @property
    def certificate(self) -> TestProofCertificate | None:
        return self.admission.certificate if self.admission is not None else None


class _DuplicateKey(ValueError):
    pass


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("duplicate JSON member")
        result[key] = value
    return result


def _candidate_value(candidate: Any, name: str, default: Any = None) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _string_set(value: Any) -> frozenset[str] | None:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    result: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        result.add(item.strip())
    return frozenset(result)


def _has_private_material(value: Any, *, depth: int = 0) -> bool:
    """Inspect names only; never reflect candidate values into diagnostics."""

    if depth > 16:
        return True
    if isinstance(value, Mapping):
        if len(value) > 256:
            return True
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if any(marker in key for marker in _PRIVATE_MARKERS):
                return True
            if _has_private_material(item, depth=depth + 1):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 256:
            return True
        return any(_has_private_material(item, depth=depth + 1) for item in value)
    return False


def _decode_contract(
    raw: Any,
    contract_type: type[TestPassReceipt] | type[TestProofCertificate],
    *,
    max_blob_bytes: int,
) -> tuple[TestPassReceipt | TestProofCertificate | None, ReuseReasonCode | None]:
    if not isinstance(raw, (bytes, bytearray)):
        return None, ReuseReasonCode.MALFORMED_ARTIFACT
    immutable_bytes = bytes(raw)
    if not immutable_bytes or len(immutable_bytes) > max_blob_bytes:
        reason = (
            ReuseReasonCode.OVER_BUDGET
            if len(immutable_bytes) > max_blob_bytes
            else ReuseReasonCode.MALFORMED_ARTIFACT
        )
        return None, reason
    try:
        text = immutable_bytes.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
        if not isinstance(payload, Mapping):
            return None, ReuseReasonCode.MALFORMED_ARTIFACT
        contract = contract_type.from_dict(payload)
    except TestExecutionContractError as exc:
        if "private" in str(exc).lower():
            return None, ReuseReasonCode.PRIVATE_MATERIAL
        if "illegal-authority" in str(exc):
            return None, ReuseReasonCode.ILLEGAL_AUTHORITY
        return None, ReuseReasonCode.MALFORMED_ARTIFACT
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, TypeError, ValueError):
        return None, ReuseReasonCode.MALFORMED_ARTIFACT
    except Exception:
        return None, ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN

    # Canonical bytes are the immutable authority.  Semantically equivalent
    # whitespace, reordered keys, or duplicate-key JSON is not admitted.
    if contract.canonical_bytes() != immutable_bytes:
        return None, ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED
    return contract, None


class TestProofCache:
    """Revalidate test proof candidates under current, caller-owned authority.

    ``candidate_provider`` is a retrieval adapter only.  ``current_policy`` can
    be supplied per call or by ``policy_provider``; per-call policy takes
    precedence.  The class never writes a trust database or promotes provider
    status flags.
    """

    __test__ = False
    interface = TEST_PROOF_CACHE_INTERFACE

    def __init__(
        self,
        *,
        verifier: Callable[[TestProofCertificate, TestPassReceipt, Mapping[str, Any]], Any]
        | Any
        | None = None,
        candidate_provider: Any | None = None,
        policy_provider: Callable[[TestLocatorKey, TestExecutionKey], Mapping[str, Any]]
        | None = None,
        revocation_checker: Callable[
            [TestProofCertificate, TestPassReceipt, Mapping[str, Any]], Any
        ]
        | None = None,
        current_policy: Mapping[str, Any] | None = None,
        clock: Callable[[], int] | None = None,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    ) -> None:
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")
        if (
            isinstance(max_blob_bytes, bool)
            or not isinstance(max_blob_bytes, int)
            or max_blob_bytes <= 0
        ):
            raise ValueError("max_blob_bytes must be a positive integer")
        self._verifier = verifier
        self._candidate_provider = candidate_provider
        self._policy_provider = policy_provider
        self._revocation_checker = revocation_checker
        self._current_policy = dict(current_policy) if current_policy is not None else None
        self._clock = clock or (lambda: time.time_ns() // 1_000_000)
        self._max_candidates = max_candidates
        self._max_blob_bytes = max_blob_bytes

    @staticmethod
    def candidate(
        receipt: TestPassReceipt,
        certificate: TestProofCertificate,
        *,
        created_at_ms: int | None = None,
        expires_at_ms: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a storage-neutral candidate from immutable canonical bytes."""

        result: dict[str, Any] = {
            "receipt_bytes": receipt.canonical_bytes(),
            "certificate_bytes": certificate.canonical_bytes(),
            "receipt_cid": receipt.receipt_id,
            "certificate_cid": certificate.certificate_id,
            "metadata": dict(metadata or {}),
        }
        if created_at_ms is not None:
            result["created_at_ms"] = created_at_ms
        if expires_at_ms is not None:
            result["expires_at_ms"] = expires_at_ms
        return result

    @staticmethod
    def _reject(
        reason_code: ReuseReasonCode,
        *,
        receipt: TestPassReceipt | None = None,
        certificate: TestProofCertificate | None = None,
    ) -> TestProofCacheAdmission:
        return TestProofCacheAdmission(
            admitted=False,
            reason_code=reason_code,
            receipt=receipt,
            certificate=certificate,
            receipt_cid=receipt.receipt_id if receipt is not None else "",
            certificate_cid=(certificate.certificate_id if certificate is not None else ""),
        )

    def _resolve_policy(
        self,
        locator: TestLocatorKey,
        execution_key: TestExecutionKey,
        current_policy: Mapping[str, Any] | None,
    ) -> tuple[Mapping[str, Any] | None, Exception | None]:
        if current_policy is not None:
            return current_policy, None
        if self._policy_provider is not None:
            try:
                return self._policy_provider(locator, execution_key), None
            except Exception as exc:
                return None, exc
        return self._current_policy, None

    @staticmethod
    def _policy_requirements(
        policy: Any,
    ) -> tuple[dict[str, Any] | None, ReuseReasonCode | None]:
        if not isinstance(policy, Mapping):
            return None, ReuseReasonCode.TRUST_POLICY_REJECTED
        requirements = dict(policy)
        for name in _REQUIRED_POLICY_IDS:
            value = requirements.get(name)
            if not isinstance(value, str) or not value.strip():
                if name == "verifying_key_cid":
                    return None, ReuseReasonCode.KEY_UNAVAILABLE
                if name == "circuit_cid":
                    return None, ReuseReasonCode.CIRCUIT_UNAVAILABLE
                return None, ReuseReasonCode.TRUST_POLICY_REJECTED
            requirements[name] = value.strip()

        for name in (
            "trusted_issuer_ids",
            "allowed_epochs",
            "revoked_issuer_ids",
            "revoked_certificate_cids",
            "revoked_receipt_cids",
        ):
            default: tuple[str, ...] = () if name.startswith("revoked_") else ()
            normalized = _string_set(requirements.get(name, default))
            if normalized is None:
                return None, ReuseReasonCode.TRUST_POLICY_REJECTED
            requirements[name] = normalized
        if not requirements["trusted_issuer_ids"] or not requirements["allowed_epochs"]:
            return None, ReuseReasonCode.TRUST_POLICY_REJECTED

        max_age = requirements.get("max_age_ms")
        if max_age is not None and (
            isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 0
        ):
            return None, ReuseReasonCode.TRUST_POLICY_REJECTED
        return requirements, None

    def admit(
        self,
        candidate: Any,
        *,
        locator: TestLocatorKey,
        execution_key: TestExecutionKey,
        current_policy: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> TestProofCacheAdmission:
        """Re-derive authority for one candidate without trusting cache state."""

        policy, policy_error = self._resolve_policy(locator, execution_key, current_policy)
        if policy_error is not None:
            return self._reject(ReuseReasonCode.CACHE_UNAVAILABLE)
        requirements, reason = self._policy_requirements(policy)
        if reason is not None or requirements is None:
            return self._reject(reason or ReuseReasonCode.TRUST_POLICY_REJECTED)

        metadata = _candidate_value(candidate, "metadata", {})
        if _has_private_material(metadata):
            return self._reject(ReuseReasonCode.PRIVATE_MATERIAL)

        receipt_value, receipt_error = _decode_contract(
            _candidate_value(candidate, "receipt_bytes"),
            TestPassReceipt,
            max_blob_bytes=self._max_blob_bytes,
        )
        if receipt_error is not None or not isinstance(receipt_value, TestPassReceipt):
            return self._reject(receipt_error or ReuseReasonCode.MALFORMED_ARTIFACT)
        receipt = receipt_value

        certificate_value, certificate_error = _decode_contract(
            _candidate_value(candidate, "certificate_bytes"),
            TestProofCertificate,
            max_blob_bytes=self._max_blob_bytes,
        )
        if certificate_error is not None or not isinstance(certificate_value, TestProofCertificate):
            return self._reject(
                certificate_error or ReuseReasonCode.MALFORMED_ARTIFACT,
                receipt=receipt,
            )
        certificate = certificate_value

        claimed_receipt_cid = _candidate_value(candidate, "receipt_cid")
        claimed_certificate_cid = _candidate_value(candidate, "certificate_cid")
        if (
            not isinstance(claimed_receipt_cid, str)
            or not isinstance(claimed_certificate_cid, str)
            or claimed_receipt_cid != receipt.receipt_id
            or claimed_certificate_cid != certificate.certificate_id
        ):
            return self._reject(
                ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED,
                receipt=receipt,
                certificate=certificate,
            )

        try:
            current_ms = self._clock() if now_ms is None else now_ms
        except Exception:
            return self._reject(
                ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                receipt=receipt,
                certificate=certificate,
            )
        if isinstance(current_ms, bool) or not isinstance(current_ms, int) or current_ms < 0:
            return self._reject(
                ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                receipt=receipt,
                certificate=certificate,
            )
        created_at = _candidate_value(candidate, "created_at_ms")
        expires_at = _candidate_value(candidate, "expires_at_ms")
        for timestamp in (created_at, expires_at):
            if timestamp is not None and (
                isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0
            ):
                return self._reject(
                    ReuseReasonCode.MALFORMED_ARTIFACT,
                    receipt=receipt,
                    certificate=certificate,
                )
        if (
            (created_at is not None and created_at > current_ms)
            or (expires_at is not None and expires_at <= current_ms)
            or (created_at is not None and expires_at is not None and expires_at <= created_at)
        ):
            return self._reject(
                ReuseReasonCode.EXPIRED_OR_REVOKED,
                receipt=receipt,
                certificate=certificate,
            )
        max_age = requirements.get("max_age_ms")
        if max_age is not None and (created_at is None or current_ms - created_at > max_age):
            return self._reject(
                ReuseReasonCode.EXPIRED_OR_REVOKED,
                receipt=receipt,
                certificate=certificate,
            )

        if locator.non_reusable_reason:
            return self._reject(
                ReuseReasonCode.NON_REUSABLE,
                receipt=receipt,
                certificate=certificate,
            )
        if execution_key.eligibility_class is EligibilityClass.NON_REUSABLE:
            return self._reject(
                ReuseReasonCode.ELIGIBILITY_DENIED,
                receipt=receipt,
                certificate=certificate,
            )
        if execution_key.locator_cid != locator.locator_id:
            return self._reject(
                ReuseReasonCode.EXECUTION_KEY_MISMATCH,
                receipt=receipt,
                certificate=certificate,
            )
        if (
            execution_key.policy_cid != requirements["policy_cid"]
            or receipt.policy_cid != requirements["policy_cid"]
            or certificate.policy_cid != requirements["policy_cid"]
        ):
            return self._reject(
                ReuseReasonCode.POLICY_MISMATCH,
                receipt=receipt,
                certificate=certificate,
            )

        if not receipt.admitted or not receipt.all_phases_pass or receipt.disqualifying_states:
            return self._reject(
                ReuseReasonCode.RECEIPT_MISMATCH,
                receipt=receipt,
                certificate=certificate,
            )
        if (
            receipt.execution_key_cid != execution_key.execution_key_id
            or receipt.locator_cid != locator.locator_id
            or certificate.execution_key_cid != execution_key.execution_key_id
            or certificate.receipt_cid != receipt.receipt_id
        ):
            return self._reject(
                ReuseReasonCode.EXECUTION_KEY_MISMATCH,
                receipt=receipt,
                certificate=certificate,
            )
        if (
            not execution_key.static_trace_root_cid
            or not execution_key.runtime_trace_root_cid
            or not execution_key.runtime_completeness_policy
            or not receipt.completeness_receipt_cid
            or receipt.static_trace_root_cid != execution_key.static_trace_root_cid
            or receipt.runtime_trace_root_cid != execution_key.runtime_trace_root_cid
            or receipt.dependency_forest_cid != execution_key.repository_forest_cid
        ):
            return self._reject(
                ReuseReasonCode.INCOMPLETE_TRACE,
                receipt=receipt,
                certificate=certificate,
            )

        if (
            certificate.backend_mode is ProofBackendMode.SIMULATED
            or certificate.authority is CertificateAuthority.NON_ATTESTED
            or not certificate.can_authorize_skip
        ):
            return self._reject(
                ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
                receipt=receipt,
                certificate=certificate,
            )
        if (
            certificate.statement_cid != requirements["statement_cid"]
            or certificate.circuit_cid != requirements["circuit_cid"]
            or certificate.verifying_key_cid != requirements["verifying_key_cid"]
            or certificate.proof_system_id != requirements["proof_system_id"]
        ):
            return self._reject(
                ReuseReasonCode.POLICY_MISMATCH,
                receipt=receipt,
                certificate=certificate,
            )
        if (
            certificate.issuer_id not in requirements["trusted_issuer_ids"]
            or certificate.epoch not in requirements["allowed_epochs"]
        ):
            return self._reject(
                ReuseReasonCode.TRUST_POLICY_REJECTED,
                receipt=receipt,
                certificate=certificate,
            )
        if certificate.issuer_id in requirements["revoked_issuer_ids"]:
            return self._reject(
                ReuseReasonCode.ISSUER_REVOKED,
                receipt=receipt,
                certificate=certificate,
            )
        if (
            certificate.certificate_id in requirements["revoked_certificate_cids"]
            or receipt.receipt_id in requirements["revoked_receipt_cids"]
        ):
            return self._reject(
                ReuseReasonCode.EXPIRED_OR_REVOKED,
                receipt=receipt,
                certificate=certificate,
            )

        expected_public_inputs = {
            "receipt_cid": receipt.receipt_id,
            "execution_key_cid": execution_key.execution_key_id,
            "policy_cid": requirements["policy_cid"],
            "statement_cid": requirements["statement_cid"],
            "circuit_cid": requirements["circuit_cid"],
            "verifying_key_cid": requirements["verifying_key_cid"],
            "proof_system_id": requirements["proof_system_id"],
            "issuer_id": certificate.issuer_id,
            "issuer_key_id": receipt.issuer_key_id,
            "epoch": certificate.epoch,
            "setup_outcome": receipt.setup_outcome.value,
            "call_outcome": receipt.call_outcome.value,
            "teardown_outcome": receipt.teardown_outcome.value,
        }
        if _has_private_material(certificate.public_inputs) or any(
            certificate.public_inputs.get(name) != expected_public_inputs[name]
            for name in _REQUIRED_PUBLIC_INPUTS
        ):
            private = _has_private_material(certificate.public_inputs)
            return self._reject(
                (ReuseReasonCode.PRIVATE_MATERIAL if private else ReuseReasonCode.POLICY_MISMATCH),
                receipt=receipt,
                certificate=certificate,
            )

        if self._revocation_checker is not None:
            try:
                revoked = self._revocation_checker(certificate, receipt, requirements)
            except TimeoutError:
                return self._reject(
                    ReuseReasonCode.TIMEOUT,
                    receipt=receipt,
                    certificate=certificate,
                )
            except Exception:
                return self._reject(
                    ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                    receipt=receipt,
                    certificate=certificate,
                )
            if revoked is not False:
                return self._reject(
                    (
                        ReuseReasonCode.EXPIRED_OR_REVOKED
                        if revoked is True
                        else ReuseReasonCode.TRUST_POLICY_REJECTED
                    ),
                    receipt=receipt,
                    certificate=certificate,
                )

        verifier = self._verifier
        verify = getattr(verifier, "verify", None)
        if not callable(verify):
            verify = verifier if callable(verifier) else None
        if verify is None:
            return self._reject(
                ReuseReasonCode.VERIFIER_UNAVAILABLE,
                receipt=receipt,
                certificate=certificate,
            )
        try:
            verified = verify(certificate, receipt, requirements)
        except TimeoutError:
            return self._reject(
                ReuseReasonCode.TIMEOUT,
                receipt=receipt,
                certificate=certificate,
            )
        except Exception:
            return self._reject(
                ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
                receipt=receipt,
                certificate=certificate,
            )
        # Exact True is required.  Mappings or objects carrying historical
        # ``verified`` / ``authoritative`` flags are deliberately not trusted.
        if verified is not True:
            return self._reject(
                ReuseReasonCode.TRUST_POLICY_REJECTED,
                receipt=receipt,
                certificate=certificate,
            )

        return TestProofCacheAdmission(
            admitted=True,
            reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
            receipt=receipt,
            certificate=certificate,
            receipt_cid=receipt.receipt_id,
            certificate_cid=certificate.certificate_id,
        )

    def _load_candidates(
        self, locator: TestLocatorKey, candidates: Iterable[Any] | Any | None
    ) -> tuple[Any, Exception | None]:
        if candidates is not None:
            return candidates, None
        provider = self._candidate_provider
        if provider is None:
            return None, None
        lookup = getattr(provider, "lookup_test_candidates", None)
        if not callable(lookup):
            lookup = getattr(provider, "lookup_candidates", None)
        if not callable(lookup):
            lookup = provider if callable(provider) else None
        if lookup is None:
            return None, TypeError("candidate provider is unsupported")
        try:
            return lookup(locator.locator_id), None
        except Exception as exc:
            return None, exc

    def lookup(
        self,
        locator: TestLocatorKey,
        execution_key: TestExecutionKey,
        *,
        candidates: Iterable[Any] | Any | None = None,
        current_policy: Mapping[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> TestProofCacheLookup:
        """Lookup candidates and return a typed RUN/SKIP decision."""

        loaded, load_error = self._load_candidates(locator, candidates)
        if load_error is not None:
            decision = decision_from_exception(
                load_error, reason_code=ReuseReasonCode.CACHE_UNAVAILABLE
            )
            return TestProofCacheLookup(
                status=TestProofCacheLookupStatus.ERROR,
                decision=decision,
                reason_codes=(ReuseReasonCode.CACHE_UNAVAILABLE,),
            )
        if loaded is None:
            decision = decision_from_absence(ReuseReasonCode.CANDIDATE_MISSING)
            return TestProofCacheLookup(
                status=TestProofCacheLookupStatus.MISS,
                decision=decision,
                reason_codes=(ReuseReasonCode.CANDIDATE_MISSING,),
            )

        if isinstance(loaded, Mapping) or not isinstance(loaded, Iterable):
            candidate_items = [loaded]
        elif isinstance(loaded, (str, bytes, bytearray)):
            candidate_items = [loaded]
        else:
            try:
                candidate_items = []
                iterator = iter(loaded)
                for _index in range(self._max_candidates + 1):
                    try:
                        candidate_items.append(next(iterator))
                    except StopIteration:
                        break
            except Exception as exc:
                decision = decision_from_exception(
                    exc, reason_code=ReuseReasonCode.CACHE_UNAVAILABLE
                )
                return TestProofCacheLookup(
                    status=TestProofCacheLookupStatus.ERROR,
                    decision=decision,
                    reason_codes=(ReuseReasonCode.CACHE_UNAVAILABLE,),
                )

        if not candidate_items:
            decision = decision_from_absence(ReuseReasonCode.CANDIDATE_MISSING)
            return TestProofCacheLookup(
                status=TestProofCacheLookupStatus.MISS,
                decision=decision,
                reason_codes=(ReuseReasonCode.CANDIDATE_MISSING,),
            )
        if len(candidate_items) > self._max_candidates:
            decision = reuse_run(ReuseReasonCode.OVER_BUDGET)
            return TestProofCacheLookup(
                status=TestProofCacheLookupStatus.MISS,
                decision=decision,
                reason_codes=(ReuseReasonCode.OVER_BUDGET,),
                candidates_considered=self._max_candidates,
            )

        reasons: list[ReuseReasonCode] = []
        last_admission: TestProofCacheAdmission | None = None
        for candidate in candidate_items:
            # Even an earlier Admission object cannot authorize this lookup;
            # only its immutable candidate bytes would be useful, and admission
            # records intentionally do not retain those bytes.
            admission = self.admit(
                candidate,
                locator=locator,
                execution_key=execution_key,
                current_policy=current_policy,
                now_ms=now_ms,
            )
            last_admission = admission
            if admission.admitted:
                assert admission.receipt is not None
                assert admission.certificate is not None
                decision = reuse_skip(
                    certificate_cid=admission.certificate_cid,
                    receipt_cid=admission.receipt_cid,
                )
                return TestProofCacheLookup(
                    status=TestProofCacheLookupStatus.HIT,
                    decision=decision,
                    admission=admission,
                    reason_codes=(ReuseReasonCode.PROOF_CACHE_HIT,),
                    candidates_considered=len(reasons) + 1,
                )
            reasons.append(admission.reason_code)

        reason_code = reasons[0] if reasons else ReuseReasonCode.CANDIDATE_MISSING
        decision = reuse_run(reason_code)
        status = (
            TestProofCacheLookupStatus.ERROR
            if reason_code in _ERROR_REASONS
            else TestProofCacheLookupStatus.MISS
        )
        return TestProofCacheLookup(
            status=status,
            decision=decision,
            admission=last_admission,
            reason_codes=tuple(reasons),
            candidates_considered=len(candidate_items),
        )


__all__ = [
    "DEFAULT_MAX_BLOB_BYTES",
    "DEFAULT_MAX_CANDIDATES",
    "TEST_PROOF_CACHE_INTERFACE",
    "TestProofCache",
    "TestProofCacheAdmission",
    "TestProofCacheLookup",
    "TestProofCacheLookupStatus",
]
