"""Locator-only warm path with signed-receipt trust (PTR-164).

Acceptance covered:

* An unmodified item reaches lookup before setup (locator only; no execution key).
* Current AST/fixture/hook/config/dependency/environment/policy context is rebuilt.
* Each warm lookup checks immutable bytes, signature, key validity, revocation,
  epoch and policy **before** proof verification.
* Any gap returns RUN (never skip from revalidation or trust alone).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_candidate_context_store import (
    TestCandidateContextStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    PhaseOutcome,
    ReuseAction,
    ReuseReasonCode,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
)
from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
    CandidateExecutionContext,
    CurrentExecutionContext,
)
from ipfs_accelerate_py.testing.proof_reuse.current_context_provider import (
    DefaultCurrentContextProvider,
    current_context_from_candidate_identities,
)
from ipfs_accelerate_py.testing.proof_reuse.lookup import (
    SIGNED_RECEIPT_TRUST_VERIFIER_INTERFACE,
    TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE,
    ProofReuseTwoStageLookup,
    RevalidatedProofReuseLookupRequest,
    SignedReceiptTrustVerifier,
    batch_lookup_reuse_decisions,
    build_proof_reuse_two_stage_lookup,
    build_signed_receipt_trust_verifier,
)
from ipfs_accelerate_py.testing.proof_reuse.runner_pass_attestation import (
    AttestationNonceRegistry,
    RunnerKeyRecord,
    RunnerPublicKey,
    RunnerTrustPolicy,
    attest_test_pass_receipt,
    dag_cbor_cid,
)


NOW_S = 1_800_000_000.0
NOW_MS = int(NOW_S * 1000)


def _cid(label: str) -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/test/locator-only-label@1",
            "label": label,
        }
    )


def _component_bytes(label: str) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "ipfs_accelerate_py/test/candidate-component@1",
            "label": label,
            "version": 1,
        }
    )


def _component_cid(label: str) -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/test/candidate-component@1",
            "label": label,
            "version": 1,
        }
    )


def _complete_runtime_trace(
    *,
    files: list[dict[str, Any]] | None = None,
    complete: bool = True,
) -> dict[str, Any]:
    dependencies: dict[str, list[dict[str, Any]]] = {
        "modules": [],
        "code_objects": [],
        "files": list(files or []),
        "environment": [],
        "subprocesses": [],
        "services": [],
        "policies": [],
        "capabilities": [],
    }
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/runtime-test-dependency-trace@1",
        "interface": "RuntimeTestDependencyTrace@1",
        "eligibility_profile": "pure",
        "completeness": {
            "status": "complete" if complete else "incomplete",
            "complete": complete,
            "reasons": [] if complete else ["private_event"],
        },
        "dependencies": dependencies,
        "health": {
            "audit_hook_healthy": True,
            "profile_healthy": True,
            "started": True,
            "stopped": True,
            "observed_event_count": sum(len(v) for v in dependencies.values()),
            "recorded_fact_count": sum(len(v) for v in dependencies.values()),
            "dropped_event_count": 0,
            "unsupported_event_kinds": [],
            "private_event_kinds": [],
            "internal_failure_kinds": [],
        },
    }


def _locator(
    *,
    node_id: str = "test/api/test_locator_only.py::test_example",
) -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:locator-only",
        package_identity="package:locator-only",
        node_id=node_id,
    )


def _execution_key(
    locator: TestLocatorKey,
    *,
    policy_cid: str = "cid:policy",
    forest_cid: str = "cid:repository-forest",
    static_cid: str = "cid:static-trace",
    runtime_cid: str = "cid:runtime-trace",
    **overrides: Any,
) -> TestExecutionKey:
    fields = dict(
        locator_cid=locator.locator_id,
        repository_forest_cid=forest_cid,
        static_trace_root_cid=static_cid,
        runtime_trace_root_cid=runtime_cid,
        runtime_completeness_policy="complete-v1",
        policy_cid=policy_cid,
        test_ast_cid="cid:ast",
        environment_cid="cid:env",
    )
    fields.update(overrides)
    return TestExecutionKey(**fields)


class _Item:
    def __init__(self, nodeid: str = "test_locator_only.py::test_example") -> None:
        self.nodeid = nodeid
        self.user_properties: list[tuple[str, Any]] = []
        self.markers: list[Any] = []
        self.path = None
        self._ipfs_proof_reuse_locator = None

    def add_marker(self, marker: Any) -> None:
        self.markers.append(marker)


def _matching_warm_bundle(tmp_path: Path, *, tag: str = "warm") -> dict[str, Any]:
    payload = b"locator-only-payload"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(exist_ok=True)
    data = fixtures / "payload.bin"
    data.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    runtime_trace = _complete_runtime_trace(
        files=[
            {
                "root_id": "repo",
                "path": "fixtures/payload.bin",
                "size_bytes": len(payload),
                "content_sha256": digest,
            }
        ]
    )
    runtime_cid = content_identity(runtime_trace)
    locator = _locator(node_id=f"test/api/test_{tag}.py::test_{tag}")
    labels = {
        "static_trace": f"static-{tag}",
        "repository_forest": f"forest-{tag}",
        "environment": f"env-{tag}",
        "policy": f"policy-{tag}",
        "pass_receipt": f"receipt-{tag}",
        "test_ast": f"ast-{tag}",
        "dependency_lock": f"lock-{tag}",
        "installed_distributions": f"dist-{tag}",
        "capability_root": f"cap-{tag}",
        "fixtures": f"fixtures-{tag}",
        "hooks": f"hooks-{tag}",
        "parameters": f"params-{tag}",
        "source": f"source-{tag}",
    }
    cids = {name: _component_cid(label) for name, label in labels.items()}
    execution_key = _execution_key(
        locator,
        policy_cid=cids["policy"],
        forest_cid=cids["repository_forest"],
        static_cid=cids["static_trace"],
        runtime_cid=runtime_cid,
        test_ast_cid=cids["test_ast"],
        environment_cid=cids["environment"],
        dependency_lock_cid=cids["dependency_lock"],
        installed_distributions_cid=cids["installed_distributions"],
        hardware_capability_cid=cids["capability_root"],
    )
    candidate = CandidateExecutionContext(
        locator_cid=locator.locator_id,
        execution_key_cid=execution_key.execution_key_id,
        pass_receipt_cid=cids["pass_receipt"],
        repository_forest_cid=cids["repository_forest"],
        test_ast_cid=cids["test_ast"],
        static_trace_root_cid=cids["static_trace"],
        runtime_trace_root_cid=runtime_cid,
        environment_cid=cids["environment"],
        policy_cid=cids["policy"],
        dependency_lock_cid=cids["dependency_lock"],
        installed_distributions_cid=cids["installed_distributions"],
        capability_root_cid=cids["capability_root"],
        component_cids={
            "execution_key": execution_key.execution_key_id,
            "static_trace": cids["static_trace"],
            "runtime_trace": runtime_cid,
            "repository_forest": cids["repository_forest"],
            "environment": cids["environment"],
            "policy": cids["policy"],
            "pass_receipt": cids["pass_receipt"],
            "test_ast": cids["test_ast"],
            "fixtures": cids["fixtures"],
            "hooks": cids["hooks"],
            "parameters": cids["parameters"],
            "source": cids["source"],
        },
        external_snapshot_cids=(),
        retained_at_ms=NOW_MS,
    )
    components = {
        "execution_key": canonical_json_bytes(execution_key.to_dict()),
        "static_trace": _component_bytes(labels["static_trace"]),
        "runtime_trace": canonical_json_bytes(runtime_trace),
        "repository_forest": _component_bytes(labels["repository_forest"]),
        "environment": _component_bytes(labels["environment"]),
        "policy": _component_bytes(labels["policy"]),
        "pass_receipt": _component_bytes(labels["pass_receipt"]),
        "test_ast": _component_bytes(labels["test_ast"]),
    }
    current = current_context_from_candidate_identities(
        candidate,
        rebuild_source="fresh_live_rebuild",
        rebuilt_at_ms=NOW_MS,
    )
    item = _Item(nodeid=locator.node_id)
    item._ipfs_proof_reuse_locator = locator
    return {
        "locator": locator,
        "candidate": candidate,
        "current": current,
        "components": components,
        "runtime_trace": runtime_trace,
        "root": tmp_path,
        "execution_key": execution_key,
        "item": item,
    }


def _live_matching_compiler(current: CurrentExecutionContext):
    def compiler(**_kwargs: Any) -> CurrentExecutionContext:
        return current

    return compiler


def _trust_material() -> tuple[
    Ed25519PrivateKey, RunnerPublicKey, RunnerTrustPolicy
]:
    private = Ed25519PrivateKey.generate()
    public = RunnerPublicKey.from_public_key(private.public_key())
    policy = RunnerTrustPolicy(
        trust_domain="pytest.local",
        active_key_epoch="epoch-7",
        keys=(
            RunnerKeyRecord(
                public_key_cid=public.cid,
                public_key_material=public.material,
                key_epoch="epoch-7",
                not_before=int(NOW_S) - 60,
                not_after=int(NOW_S) + 3600,
            ),
        ),
        policy_epoch="policy-3",
    )
    return private, public, policy


def _admitted_receipt(
    locator: TestLocatorKey,
    execution_key: TestExecutionKey,
    policy: RunnerTrustPolicy,
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
        policy_cid=policy.cid,
        trust_domain=policy.trust_domain,
        admitted=True,
    )


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


def test_two_stage_interface_is_signed_receipt_trust_v2() -> None:
    assert TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE == "TwoStageCandidateLookup@2"
    assert (
        SIGNED_RECEIPT_TRUST_VERIFIER_INTERFACE == "SignedReceiptTrustVerifier@1"
    )
    lookup = build_proof_reuse_two_stage_lookup()
    assert lookup.interface == TWO_STAGE_CANDIDATE_LOOKUP_INTERFACE
    assert lookup.may_authorize_skip_from_revalidation_alone is False
    assert isinstance(lookup, ProofReuseTwoStageLookup)


# ---------------------------------------------------------------------------
# Locator-only warm path
# ---------------------------------------------------------------------------


def test_unmodified_item_reaches_lookup_with_locator_only(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    ctx_store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    put = ctx_store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    assert put.stored

    compile_calls: list[str] = []

    def compiler(**_kwargs: Any) -> CurrentExecutionContext:
        compile_calls.append("rebuilt")
        return bundle["current"]

    provider = DefaultCurrentContextProvider(
        live_identity_compiler=compiler,
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_S,
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=ctx_store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    # Locator + item only — no execution key (collection seed style).
    request = RevalidatedProofReuseLookupRequest(
        item=bundle["item"],
        locator=bundle["locator"],
        execution_key=None,
        now_ms=NOW_MS,
    )
    decisions = batch_lookup_reuse_decisions(
        lookup,
        (request,),
        apply_skips=False,
    )
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE
    assert compile_calls, "current context must be rebuilt before proof stage"
    assert provider.fixtures_executed is False
    assert provider.test_body_executed is False


def test_direct_locator_only_lookup_rebuilds_context(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path, tag="direct")
    ctx_store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    ctx_store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_S,
    )
    lookup = build_proof_reuse_two_stage_lookup(
        candidate_context_store=ctx_store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    decision = lookup.lookup(
        bundle["locator"],
        None,
        item=bundle["item"],
        now_ms=NOW_MS,
    )
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE
    assert provider.fixtures_executed is False


# ---------------------------------------------------------------------------
# Signed-receipt trust gate
# ---------------------------------------------------------------------------


def test_signed_receipt_trust_verifier_ordered_checks() -> None:
    private, public, policy = _trust_material()
    locator = _locator()
    execution_key = _execution_key(locator, policy_cid=policy.cid)
    receipt = _admitted_receipt(locator, execution_key, policy)
    registry = AttestationNonceRegistry()
    candidate_context_cid = dag_cbor_cid({"candidate": "trust-v1"})
    attestation = attest_test_pass_receipt(
        receipt,
        private_key=private,
        policy=policy,
        candidate_context_cid=candidate_context_cid,
        issuance_nonce="nonce-trust-ordered",
        issued_at=int(NOW_S),
        nonce_registry=registry,
    )
    trust = build_signed_receipt_trust_verifier(
        trust_policy=policy,
        pinned_policy_cid=policy.cid,
        pinned_public_key_material=public.material,
        nonce_registry=registry,
        clock=lambda: int(NOW_S),
    )
    assert trust.interface == SIGNED_RECEIPT_TRUST_VERIFIER_INTERFACE
    assert trust.may_authorize_skip is False

    result = trust.verify(
        receipt=receipt,
        receipt_bytes=receipt.canonical_bytes(),
        attestation=attestation,
        attestation_bytes=attestation.canonical_bytes(),
        current_execution_key_cid=execution_key.execution_key_id,
        current_candidate_context_cid=candidate_context_cid,
        now=int(NOW_S),
    )
    assert result.verified is True
    assert result.may_authorize_skip is False
    assert result.may_proceed_to_proof_verification is True
    for name in (
        "immutable_bytes",
        "signature",
        "key_validity",
        "revocation",
        "epoch",
        "policy",
    ):
        assert result.checks.get(name) is True, name
    assert result.signed_receipt is not None
    assert result.signed_receipt.runner_attestation_cid == attestation.cid


def test_trust_rejects_tampered_bytes_and_bad_signature() -> None:
    private, _public, policy = _trust_material()
    locator = _locator()
    execution_key = _execution_key(locator, policy_cid=policy.cid)
    receipt = _admitted_receipt(locator, execution_key, policy)
    registry = AttestationNonceRegistry()
    candidate_context_cid = dag_cbor_cid({"candidate": "tamper-v1"})
    attestation = attest_test_pass_receipt(
        receipt,
        private_key=private,
        policy=policy,
        candidate_context_cid=candidate_context_cid,
        issuance_nonce="nonce-tamper",
        issued_at=int(NOW_S),
        nonce_registry=registry,
    )
    trust = SignedReceiptTrustVerifier(
        trust_policy=policy,
        pinned_policy_cid=policy.cid,
        clock=lambda: int(NOW_S),
    )

    # Immutable receipt bytes mismatch.
    mismatched = trust.verify(
        receipt=receipt,
        receipt_bytes=b'{"not":"the-receipt"}',
        attestation=attestation,
        attestation_bytes=attestation.canonical_bytes(),
        current_execution_key_cid=execution_key.execution_key_id,
        current_candidate_context_cid=candidate_context_cid,
        now=int(NOW_S),
    )
    assert mismatched.verified is False
    assert mismatched.checks.get("immutable_bytes") is False

    # Tampered attestation signature envelope.
    bad = bytearray(attestation.canonical_bytes())
    bad[-1] ^= 0xFF
    sig_fail = trust.verify(
        receipt=receipt,
        receipt_bytes=receipt.canonical_bytes(),
        attestation_bytes=bytes(bad),
        current_execution_key_cid=execution_key.execution_key_id,
        current_candidate_context_cid=candidate_context_cid,
        now=int(NOW_S),
    )
    assert sig_fail.verified is False


def test_trust_rejects_wrong_epoch_and_policy_pin() -> None:
    private, public, policy = _trust_material()
    locator = _locator()
    execution_key = _execution_key(locator, policy_cid=policy.cid)
    receipt = _admitted_receipt(locator, execution_key, policy)
    registry = AttestationNonceRegistry()
    candidate_context_cid = dag_cbor_cid({"candidate": "epoch-v1"})
    attestation = attest_test_pass_receipt(
        receipt,
        private_key=private,
        policy=policy,
        candidate_context_cid=candidate_context_cid,
        issuance_nonce="nonce-epoch",
        issued_at=int(NOW_S),
        nonce_registry=registry,
    )
    other_policy = RunnerTrustPolicy(
        trust_domain="other.domain",
        active_key_epoch="epoch-99",
        keys=(
            RunnerKeyRecord(
                public_key_cid=public.cid,
                public_key_material=public.material,
                key_epoch="epoch-99",
                not_before=int(NOW_S) - 60,
                not_after=int(NOW_S) + 3600,
            ),
        ),
        policy_epoch="policy-other",
    )
    trust = SignedReceiptTrustVerifier(
        trust_policy=other_policy,
        pinned_policy_cid=other_policy.cid,
        clock=lambda: int(NOW_S),
    )
    result = trust.verify(
        receipt=receipt,
        receipt_bytes=receipt.canonical_bytes(),
        attestation=attestation,
        attestation_bytes=attestation.canonical_bytes(),
        current_execution_key_cid=execution_key.execution_key_id,
        current_candidate_context_cid=candidate_context_cid,
        now=int(NOW_S),
    )
    assert result.verified is False


def test_warm_lookup_runs_when_attestation_required_but_missing(
    tmp_path: Path,
) -> None:
    private, _public, policy = _trust_material()
    del private
    bundle = _matching_warm_bundle(tmp_path, tag="missing-att")
    ctx_store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    ctx_store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    trust = build_signed_receipt_trust_verifier(
        trust_policy=policy,
        pinned_policy_cid=policy.cid,
        clock=lambda: int(NOW_S),
        require_attestation=True,
    )
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_S,
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=ctx_store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        signed_receipt_trust_verifier=trust,
        timeout_seconds=2.0,
    )
    decision = lookup.lookup(
        bundle["locator"],
        None,
        item=bundle["item"],
        now_ms=NOW_MS,
    )
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN
    assert decision.diagnostics.get("stage") == "signed_receipt_trust"


def test_any_gap_runs_and_trust_never_authorizes_skip() -> None:
    trust = SignedReceiptTrustVerifier(require_attestation=True)
    result = trust.verify()
    assert result.verified is False
    assert result.may_authorize_skip is False
    assert result.may_proceed_to_proof_verification is False

    lookup = build_proof_reuse_two_stage_lookup(timeout_seconds=1.0)
    decision = lookup.lookup(
        _locator(),
        None,
        item=_Item(),
        now_ms=NOW_MS,
    )
    assert decision.action is ReuseAction.RUN
    assert decision.reason_code is ReuseReasonCode.CACHE_UNAVAILABLE
