from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    CertificateAuthority,
    PhaseOutcome,
    ProofBackendMode,
    ReuseAction,
    ReuseReasonCode,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
    reuse_run,
    reuse_skip,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import TestProofCache
from ipfs_accelerate_py.testing.proof_reuse.lookup import (
    ITEM_DECISION_ATTRIBUTE,
    ProofReuseLookup,
    ProofReuseLookupRequest,
    apply_verified_skip,
    batch_lookup_reuse_decisions,
)

NOW_MS = 10_000


def _locator(*, node_id: str = "test/api/test_example.py::test_example") -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:example",
        package_identity="package:example",
        node_id=node_id,
    )


def _execution_key(
    locator: TestLocatorKey,
    *,
    policy_cid: str = "cid:policy",
) -> TestExecutionKey:
    return TestExecutionKey(
        locator_cid=locator.locator_id,
        repository_forest_cid="cid:repository-forest",
        static_trace_root_cid="cid:static-trace",
        runtime_trace_root_cid="cid:runtime-trace",
        runtime_completeness_policy="complete-v1",
        policy_cid=policy_cid,
    )


def _receipt(
    locator: TestLocatorKey,
    execution_key: TestExecutionKey,
) -> TestPassReceipt:
    return TestPassReceipt(
        execution_key_cid=execution_key.execution_key_id,
        locator_cid=locator.locator_id,
        setup_outcome=PhaseOutcome.PASS,
        call_outcome=PhaseOutcome.PASS,
        teardown_outcome=PhaseOutcome.PASS,
        static_trace_root_cid=execution_key.static_trace_root_cid,
        runtime_trace_root_cid=execution_key.runtime_trace_root_cid,
        completeness_receipt_cid="cid:completeness-receipt",
        dependency_forest_cid=execution_key.repository_forest_cid,
        issuer_key_id="key:issuer",
        policy_cid=execution_key.policy_cid,
    )


def _certificate(
    receipt: TestPassReceipt,
    execution_key: TestExecutionKey,
) -> TestProofCertificate:
    return TestProofCertificate(
        receipt_cid=receipt.receipt_id,
        execution_key_cid=execution_key.execution_key_id,
        policy_cid=execution_key.policy_cid,
        statement_cid="cid:statement",
        circuit_cid="cid:circuit",
        verifying_key_cid="cid:verifying-key",
        proof_artifact_cid="cid:proof",
        issuer_id="issuer:trusted",
        epoch="epoch:7",
        proof_system_id="groth16",
        backend_mode=ProofBackendMode.CRYPTOGRAPHIC,
        authority=CertificateAuthority.AUTHORITATIVE,
        public_inputs={
            "receipt_cid": receipt.receipt_id,
            "execution_key_cid": execution_key.execution_key_id,
            "policy_cid": execution_key.policy_cid,
            "statement_cid": "cid:statement",
            "circuit_cid": "cid:circuit",
            "verifying_key_cid": "cid:verifying-key",
            "proof_system_id": "groth16",
            "issuer_id": "issuer:trusted",
            "issuer_key_id": "key:issuer",
            "epoch": "epoch:7",
            "setup_outcome": "pass",
            "call_outcome": "pass",
            "teardown_outcome": "pass",
        },
    )


def _policy(**changes: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "policy_cid": "cid:policy",
        "statement_cid": "cid:statement",
        "circuit_cid": "cid:circuit",
        "verifying_key_cid": "cid:verifying-key",
        "proof_system_id": "groth16",
        "trusted_issuer_ids": ("issuer:trusted",),
        "allowed_epochs": ("epoch:7",),
        "revoked_issuer_ids": (),
        "revoked_receipt_cids": (),
        "revoked_certificate_cids": (),
    }
    result.update(changes)
    return result


class _Store:
    def __init__(self, candidates: Any) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def lookup(self, locator_cid: str, *, max_candidates: int) -> Any:
        self.calls.append((locator_cid, max_candidates))
        return self.candidates


class _Provider:
    def __init__(self, result: Any = True, error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.verify_calls = 0
        self.prove_calls = 0

    def as_cache_verifier(self):
        def _verify(*_args: Any) -> Any:
            self.verify_calls += 1
            if self.error is not None:
                raise self.error
            return self.result

        return _verify

    def prove(self, *_args: Any, **_kwargs: Any) -> None:
        self.prove_calls += 1
        raise AssertionError("lookup must never prove")


class _Item:
    def __init__(self, nodeid: str = "test_example.py::test_example") -> None:
        self.nodeid = nodeid
        self.user_properties: list[tuple[str, Any]] = []
        self.markers: list[Any] = []

    def add_marker(self, marker: Any) -> None:
        self.markers.append(marker)


def _fixture(
    *,
    candidate_changes: Mapping[str, Any] | None = None,
    provider: _Provider | None = None,
    store: Any = None,
    max_candidates: int = 32,
    timeout_seconds: float = 1.0,
) -> tuple[
    ProofReuseLookup,
    TestLocatorKey,
    TestExecutionKey,
    TestPassReceipt,
    TestProofCertificate,
    dict[str, Any],
    _Provider,
]:
    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)
    candidate = TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=9_000,
        expires_at_ms=11_000,
    )
    candidate.update(candidate_changes or {})
    provider_value = provider or _Provider()
    lookup = ProofReuseLookup(
        _Store((candidate,)) if store is None else store,
        provider_value,
        current_policy=_policy(),
        max_candidates=max_candidates,
        timeout_seconds=timeout_seconds,
    )
    return (
        lookup,
        locator,
        execution_key,
        receipt,
        certificate,
        candidate,
        provider_value,
    )


def _assert_run(decision: Any, reason: ReuseReasonCode | None = None) -> None:
    assert decision.action is ReuseAction.RUN
    assert not decision.certificate_cid
    assert not decision.receipt_cid
    if reason is not None:
        assert decision.reason_code is reason


def test_exact_verified_candidate_is_the_only_skip_and_never_proves() -> None:
    lookup, locator, execution_key, receipt, certificate, _candidate, provider = (
        _fixture()
    )

    decision = lookup.lookup(locator, execution_key, now_ms=NOW_MS)

    assert decision.action is ReuseAction.SKIP
    assert decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
    assert decision.authority is CertificateAuthority.AUTHORITATIVE
    assert decision.certificate_cid == certificate.certificate_id
    assert decision.receipt_cid == receipt.receipt_id
    assert provider.verify_calls == 1
    assert provider.prove_calls == 0


@pytest.mark.parametrize(
    ("candidate_changes", "reason"),
    (
        ({"expires_at_ms": NOW_MS}, ReuseReasonCode.EXPIRED_OR_REVOKED),
        ({"receipt_bytes": b"not-json"}, ReuseReasonCode.MALFORMED_ARTIFACT),
        ({"receipt_cid": "cid:forged"}, ReuseReasonCode.CANDIDATE_INTEGRITY_FAILED),
    ),
)
def test_stale_parse_error_and_poisoned_cid_all_run(
    candidate_changes: Mapping[str, Any],
    reason: ReuseReasonCode,
) -> None:
    lookup, locator, execution_key, *_rest = _fixture(
        candidate_changes=candidate_changes
    )

    _assert_run(lookup.lookup(locator, execution_key, now_ms=NOW_MS), reason)


def test_miss_runs_without_invoking_verifier() -> None:
    provider = _Provider()
    lookup, locator, execution_key, *_rest = _fixture(
        provider=provider,
        store=_Store(()),
    )

    _assert_run(
        lookup.lookup(locator, execution_key, now_ms=NOW_MS),
        ReuseReasonCode.CANDIDATE_MISSING,
    )
    assert provider.verify_calls == 0
    assert provider.prove_calls == 0


def test_provider_error_and_unexpected_store_error_run() -> None:
    provider = _Provider(error=RuntimeError("provider secret must not escape"))
    lookup, locator, execution_key, *_rest = _fixture(provider=provider)
    provider_decision = lookup.lookup(locator, execution_key, now_ms=NOW_MS)

    _assert_run(
        provider_decision,
        ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
    )
    assert "provider secret" not in repr(dict(provider_decision.diagnostics))
    assert provider.prove_calls == 0

    class BrokenStore:
        def lookup(self, *_args: Any, **_kwargs: Any) -> None:
            raise LookupError("unavailable")

    lookup, locator, execution_key, *_rest = _fixture(store=BrokenStore())
    _assert_run(
        lookup.lookup(locator, execution_key, now_ms=NOW_MS),
        ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN,
    )


def test_lookup_timeout_is_wall_clock_bounded_and_runs() -> None:
    class SlowStore:
        def lookup(self, *_args: Any, **_kwargs: Any) -> tuple[()]:
            time.sleep(0.2)
            return ()

    lookup, locator, execution_key, *_rest = _fixture(
        store=SlowStore(),
        timeout_seconds=0.02,
    )
    started = time.monotonic()

    decision = lookup.lookup(locator, execution_key, now_ms=NOW_MS)

    elapsed = time.monotonic() - started
    _assert_run(decision, ReuseReasonCode.TIMEOUT)
    assert elapsed < 0.15


def test_exact_current_locator_and_execution_binding_is_required() -> None:
    lookup, locator, _execution_key_value, *_rest = _fixture()
    other_locator = _locator(node_id="test/api/test_other.py::test_other")
    stale_execution_key = _execution_key(other_locator)

    _assert_run(
        lookup.lookup(locator, stale_execution_key, now_ms=NOW_MS),
        ReuseReasonCode.EXECUTION_KEY_MISMATCH,
    )
    assert not lookup._candidate_store.calls


def test_false_or_stale_eligibility_runs_before_store_lookup() -> None:
    lookup, locator, execution_key, *_rest = _fixture()

    _assert_run(
        lookup.lookup(
            locator,
            execution_key,
            eligibility=False,
            now_ms=NOW_MS,
        ),
        ReuseReasonCode.ELIGIBILITY_DENIED,
    )

    class StaleEligibility:
        reusable = True
        repository_forest_cid = "cid:old-repository"
        static_trace_root_cid = execution_key.static_trace_root_cid
        runtime_trace_root_cid = execution_key.runtime_trace_root_cid

        def verify(self) -> Any:
            return self

    _assert_run(
        lookup.lookup(
            locator,
            execution_key,
            eligibility=StaleEligibility(),
            now_ms=NOW_MS,
        ),
        ReuseReasonCode.INVALIDATION,
    )
    assert not lookup._candidate_store.calls


def test_truthy_or_non_attested_verifier_results_cannot_skip() -> None:
    class Truthy:
        def __bool__(self) -> bool:
            return True

    for result in (Truthy(), {"verified": True, "authoritative": True}, 1):
        provider = _Provider(result=result)
        lookup, locator, execution_key, *_rest = _fixture(provider=provider)
        _assert_run(
            lookup.lookup(locator, execution_key, now_ms=NOW_MS),
            ReuseReasonCode.TRUST_POLICY_REJECTED,
        )
        assert provider.prove_calls == 0


def test_candidate_count_and_batch_size_are_bounded() -> None:
    lookup, locator, execution_key, _receipt_value, _cert, candidate, provider = (
        _fixture(max_candidates=2)
    )
    yielded = 0

    def candidates():
        nonlocal yielded
        for _index in range(100):
            yielded += 1
            yield candidate

    lookup._candidate_store = _Store(candidates())
    _assert_run(
        lookup.lookup(locator, execution_key, now_ms=NOW_MS),
        ReuseReasonCode.OVER_BUDGET,
    )
    assert yielded == 3
    assert provider.verify_calls == 0
    assert provider.prove_calls == 0

    lookup.max_batch_items = 1
    first = _Item("first")
    second = _Item("second")
    third = _Item("third")
    decisions = lookup.batch_lookup(
        (
            ProofReuseLookupRequest(first, locator, execution_key, now_ms=NOW_MS),
            ProofReuseLookupRequest(second, locator, execution_key, now_ms=NOW_MS),
            ProofReuseLookupRequest(third, locator, execution_key, now_ms=NOW_MS),
        )
    )
    assert len(decisions) == 2
    _assert_run(decisions[1], ReuseReasonCode.OVER_BUDGET)
    assert not second.markers
    assert not third.markers


def test_verified_skip_application_attaches_exact_marker_and_properties() -> None:
    item = _Item()
    decision = reuse_skip(
        certificate_cid="cid:certificate",
        receipt_cid="cid:receipt",
    )

    assert apply_verified_skip(item, decision)

    assert len(item.markers) == 1
    marker = item.markers[0]
    assert marker.name == "skip"
    assert marker.kwargs == {"reason": "proof-cache-hit:cid:certificate"}
    assert getattr(item, ITEM_DECISION_ATTRIBUTE) == decision
    assert ("proof_reuse_action", "SKIP") in item.user_properties
    assert (
        "proof_reuse_certificate_cid",
        "cid:certificate",
    ) in item.user_properties


def test_run_malformed_decision_and_marker_error_never_skip() -> None:
    item = _Item()
    assert not apply_verified_skip(
        item,
        reuse_run(ReuseReasonCode.CANDIDATE_MISSING),
    )
    assert not item.markers

    assert not apply_verified_skip(
        item,
        {
            "action": "SKIP",
            "reason_code": "proof_cache_hit",
            "certificate_cid": "cid:forged",
            "receipt_cid": "cid:forged",
        },
    )
    assert not item.markers
    _assert_run(
        getattr(item, ITEM_DECISION_ATTRIBUTE),
        ReuseReasonCode.MALFORMED_ARTIFACT,
    )

    class BrokenItem(_Item):
        def add_marker(self, marker: Any) -> None:
            raise RuntimeError("unsupported")

    broken = BrokenItem()
    assert not apply_verified_skip(
        broken,
        reuse_skip(
            certificate_cid="cid:certificate",
            receipt_cid="cid:receipt",
        ),
    )
    _assert_run(
        getattr(broken, ITEM_DECISION_ATTRIBUTE),
        ReuseReasonCode.UNSUPPORTED,
    )


def test_batch_lookup_attaches_hit_and_unsupported_run_decisions() -> None:
    lookup, locator, execution_key, *_rest = _fixture()
    hit_item = _Item("hit")
    unsupported_item = _Item("unsupported")

    decisions = batch_lookup_reuse_decisions(
        (
            ProofReuseLookupRequest(
                hit_item,
                locator,
                execution_key,
                now_ms=NOW_MS,
            ),
            unsupported_item,
        ),
        lookup=lookup,
    )

    assert decisions[0].is_skip
    assert hit_item.markers[0].kwargs["reason"].startswith("proof-cache-hit:")
    _assert_run(decisions[1], ReuseReasonCode.UNSUPPORTED)
    assert not unsupported_item.markers
    assert (
        getattr(unsupported_item, ITEM_DECISION_ATTRIBUTE).reason_code
        is ReuseReasonCode.UNSUPPORTED
    )
