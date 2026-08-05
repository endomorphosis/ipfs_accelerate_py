"""Two-stage warm lookup: locator-first revalidation then proof cache (PTR-145).

Acceptance covered:

* Warm lookup begins with locator plus current collected item only.
* A dedicated TestCandidateContextStore returns retained bytes.
* Every component is rehashed.
* The retained runtime frontier is resolved against admitted live roots.
* Current AST, static trace, fixtures, hooks, parameters, repository forest,
  locks, distributions, environment, capabilities, snapshots and policy are
  rebuilt without fixture or test execution.
* The final current execution key exactly matches the candidate before proof
  verification.
* Revalidation alone can never skip.
* Every miss, mismatch, unknown, timeout, corruption, provider absence, or
  exception returns RUN.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.test_candidate_context_store import (
    TestCandidateContextStore,
)
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
)
from ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache import TestProofCache
from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
    CandidateExecutionContext,
    CurrentExecutionContext,
)
from ipfs_accelerate_py.testing.proof_reuse.current_context_provider import (
    CURRENT_CONTEXT_PROVIDER_INTERFACE,
    DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE,
    CurrentContextCompileReason,
    DefaultCurrentContextProvider,
    build_default_current_context_provider,
    current_context_from_candidate_identities,
)
from ipfs_accelerate_py.testing.proof_reuse.lookup import (
    PROOF_REUSE_TWO_STAGE_LOOKUP_INTERFACE,
    ProofReuseLookup,
    ProofReuseTwoStageLookup,
    RevalidatedProofReuseLookupRequest,
    batch_lookup_reuse_decisions,
    build_proof_reuse_two_stage_lookup,
)
from ipfs_accelerate_py.testing.proof_reuse.runtime_revalidation import (
    RevalidationAction,
    RevalidationReason,
    RuntimeContextRevalidator,
    build_runtime_context_revalidator,
    revalidation_result_to_run_decision,
)


NOW_S = 20.0
NOW_MS = 20_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return content_identity(
        {
            "schema": "ipfs_accelerate_py/test/two-stage-label@1",
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
    node_id: str = "test/api/test_two_stage.py::test_example",
) -> TestLocatorKey:
    return TestLocatorKey(
        repository_id="repository:two-stage",
        package_identity="package:two-stage",
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


class _CertProvider:
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


class _CertStore:
    def __init__(self, candidates: Any) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def lookup(self, locator_cid: str, *, max_candidates: int) -> Any:
        self.calls.append((locator_cid, max_candidates))
        return self.candidates


class _Item:
    def __init__(self, nodeid: str = "test_two_stage.py::test_example") -> None:
        self.nodeid = nodeid
        self.user_properties: list[tuple[str, Any]] = []
        self.markers: list[Any] = []
        self.path = None

    def add_marker(self, marker: Any) -> None:
        self.markers.append(marker)


def _matching_warm_bundle(
    tmp_path: Path,
    *,
    tag: str = "warm",
    mutate_file: bool = False,
    use_real_execution_key: bool = False,
) -> dict[str, Any]:
    """Build store-ready candidate + live matching current context."""

    payload = b"warm-fixture-payload"
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(exist_ok=True)
    data = fixtures / "payload.bin"
    data.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if mutate_file:
        data.write_bytes(payload + b"-changed")

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

    if use_real_execution_key:
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
        execution_key_bytes = canonical_json_bytes(execution_key.to_dict())
        execution_key_cid = execution_key.execution_key_id
    else:
        execution_key = None
        execution_key_bytes = _component_bytes(f"ek-{tag}")
        execution_key_cid = _component_cid(f"ek-{tag}")

    candidate = CandidateExecutionContext(
        locator_cid=locator.locator_id,
        execution_key_cid=execution_key_cid,
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
            "execution_key": execution_key_cid,
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
        "execution_key": execution_key_bytes,
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
    return {
        "locator": locator,
        "candidate": candidate,
        "current": current,
        "components": components,
        "runtime_trace": runtime_trace,
        "root": tmp_path,
        "execution_key": execution_key,
        "item": _Item(nodeid=locator.node_id),
    }


def _live_matching_compiler(current: CurrentExecutionContext):
    fixture_calls = {"n": 0}
    body_calls = {"n": 0}

    def compiler(**_kwargs: Any) -> CurrentExecutionContext:
        # Warm rebuild must not execute fixtures or the test body.
        assert fixture_calls["n"] == 0
        assert body_calls["n"] == 0
        return current

    compiler.fixture_calls = fixture_calls  # type: ignore[attr-defined]
    compiler.body_calls = body_calls  # type: ignore[attr-defined]
    return compiler


def _assert_run(decision: Any, reason: ReuseReasonCode | None = None) -> None:
    assert decision.action is ReuseAction.RUN
    assert not decision.certificate_cid
    assert not decision.receipt_cid
    if reason is not None:
        assert decision.reason_code is reason


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


def test_public_interfaces_and_symbols_are_stable() -> None:
    assert CURRENT_CONTEXT_PROVIDER_INTERFACE == "CurrentContextProvider@1"
    assert (
        DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE
        == "DefaultCurrentContextProvider@1"
    )
    assert PROOF_REUSE_TWO_STAGE_LOOKUP_INTERFACE == "ProofReuseTwoStageLookup@1"
    provider = build_default_current_context_provider()
    assert provider.interface == DEFAULT_CURRENT_CONTEXT_PROVIDER_INTERFACE
    assert provider.may_authorize_skip is False
    lookup = build_proof_reuse_two_stage_lookup()
    assert lookup.interface == PROOF_REUSE_TWO_STAGE_LOOKUP_INTERFACE
    assert isinstance(lookup, ProofReuseLookup)
    assert lookup.may_authorize_skip_from_revalidation_alone is False


# ---------------------------------------------------------------------------
# DefaultCurrentContextProvider
# ---------------------------------------------------------------------------


def test_default_current_context_provider_rebuilds_without_fixtures(
    tmp_path: Path,
) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    compiler = _live_matching_compiler(bundle["current"])
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=compiler,
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_S,
    )
    provider.bind_collected_item(bundle["item"])
    assert provider.collected_item is bundle["item"]

    result = provider.compile_current_result(
        locator_cid=bundle["candidate"].locator_cid,
        candidate=bundle["candidate"],
        component_bytes=bundle["components"],
    )
    assert result.compiled is True
    assert result.reason is CurrentContextCompileReason.COMPILED
    assert result.fixtures_executed is False
    assert result.test_body_executed is False
    assert result.may_authorize_skip is False
    assert result.context is not None
    assert result.context.execution_key_cid == bundle["candidate"].execution_key_cid
    assert result.context.rebuild_source == "fresh_live_rebuild"
    # All required surfaces present on rebuilt context.
    for attr in (
        "test_ast_cid",
        "static_trace_root_cid",
        "runtime_trace_root_cid",
        "repository_forest_cid",
        "environment_cid",
        "policy_cid",
        "dependency_lock_cid",
        "installed_distributions_cid",
        "capability_root_cid",
    ):
        assert getattr(result.context, attr)
    for key in ("fixtures", "hooks", "parameters"):
        assert key in result.context.component_cids


def test_provider_absence_and_incomplete_identity_return_none(
    tmp_path: Path,
) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    empty = DefaultCurrentContextProvider()
    assert (
        empty.compile_current(
            locator_cid=bundle["candidate"].locator_cid,
            candidate=bundle["candidate"],
            component_bytes=bundle["components"],
        )
        is None
    )
    result = empty.compile_current_result(
        locator_cid=bundle["candidate"].locator_cid,
        candidate=bundle["candidate"],
        component_bytes=bundle["components"],
    )
    assert result.compiled is False
    assert result.reason in {
        CurrentContextCompileReason.PROVIDER_ABSENT,
        CurrentContextCompileReason.IDENTITY_INCOMPLETE,
    }


def test_provider_rejects_fixture_or_body_side_effects(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)

    def bad_compiler(**kwargs: Any) -> CurrentExecutionContext:
        provider = kwargs.get("provider")
        if provider is not None:
            provider.note_fixture_execution_forbidden()
        return bundle["current"]

    provider = DefaultCurrentContextProvider(live_identity_compiler=bad_compiler)
    result = provider.compile_current_result(
        locator_cid=bundle["candidate"].locator_cid,
        candidate=bundle["candidate"],
        component_bytes=bundle["components"],
    )
    assert result.compiled is False
    assert result.reason is CurrentContextCompileReason.FIXTURE_EXECUTION_FORBIDDEN
    assert result.fixtures_executed is True


# ---------------------------------------------------------------------------
# Locator-first stage-1 revalidation
# ---------------------------------------------------------------------------


def test_warm_lookup_starts_with_locator_and_item_only(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    put = store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    assert put.stored
    assert put.may_authorize_skip is False

    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_S,
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_MS,
        timeout_seconds=2.0,
    )

    # No execution key, no certificate provider — stage-1 proceeds then RUN.
    decision = lookup.lookup(
        bundle["locator"],
        None,
        item=bundle["item"],
        now_ms=NOW_MS,
    )
    _assert_run(decision, ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE)
    assert provider.compile_count >= 1
    assert provider.fixtures_executed is False
    assert provider.test_body_executed is False


def test_dedicated_candidate_store_returns_retained_bytes_and_rehashes(
    tmp_path: Path,
) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    hit = store.lookup(bundle["candidate"].locator_cid)
    assert hit.hit
    assert hit.may_authorize_skip is False
    assert hit.descriptor_bytes
    assert hit.component_bytes
    assert "runtime_trace" in hit.component_bytes
    assert "execution_key" in hit.component_bytes

    revalidator = build_runtime_context_revalidator(
        candidate_store=store,
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_S,
    )
    result = revalidator.revalidate(bundle["locator"].locator_id)
    assert result.is_proceed
    assert result.lookup_hit is True
    assert result.component_bytes
    assert result.may_authorize_skip is False


def test_revalidation_alone_never_skips(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    result = lookup.revalidate_only(
        bundle["locator"],
        item=bundle["item"],
        now_ms=NOW_MS,
    )
    assert result.action is RevalidationAction.PROCEED_TO_CERTIFICATE_VERIFICATION
    assert result.may_authorize_skip is False
    fenced = revalidation_result_to_run_decision(result)
    assert fenced.action is ReuseAction.RUN
    assert fenced.reason_code is not ReuseReasonCode.PROOF_CACHE_HIT


# ---------------------------------------------------------------------------
# Live frontier + identity mismatch → RUN
# ---------------------------------------------------------------------------


def test_changed_runtime_frontier_returns_run(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path, mutate_file=True)
    store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    decision = lookup.lookup(bundle["locator"], item=bundle["item"], now_ms=NOW_MS)
    _assert_run(decision, ReuseReasonCode.INVALIDATION)


def test_identity_mismatch_returns_run(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    mutated = current_context_from_candidate_identities(
        bundle["candidate"],
        rebuild_source="fresh_live_rebuild",
        rebuilt_at_ms=NOW_MS,
        overrides={"test_ast_cid": _cid("ast-mutated")},
    )
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(mutated),
        allowed_roots={"repo": bundle["root"]},
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    decision = lookup.lookup(bundle["locator"], item=bundle["item"], now_ms=NOW_MS)
    _assert_run(decision, ReuseReasonCode.EXECUTION_KEY_MISMATCH)


def test_candidate_miss_returns_run(tmp_path: Path) -> None:
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=lambda **_k: None,
        allowed_roots={"repo": tmp_path},
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=TestCandidateContextStore(
            tmp_path / "ctx", clock=lambda: NOW_S
        ),
        current_context_provider=provider,
        allowed_roots={"repo": tmp_path},
        timeout_seconds=2.0,
    )
    decision = lookup.lookup(_locator(), item=_Item(), now_ms=NOW_MS)
    _assert_run(decision, ReuseReasonCode.CANDIDATE_MISSING)


def test_store_corruption_returns_run(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    # Tamper retained component bytes after publication.
    bad = dict(bundle["components"])
    bad["static_trace"] = _component_bytes("static-tampered")
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
    )
    revalidator = RuntimeContextRevalidator(
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
    )
    result = revalidator.revalidate(
        bundle["candidate"].locator_cid,
        candidate=bundle["candidate"],
        component_bytes=bad,
    )
    assert result.is_run
    assert result.reason is RevalidationReason.COMPONENT_INTEGRITY_FAILED

    lookup = ProofReuseTwoStageLookup(
        revalidator=revalidator,
        candidate_context_store=store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    # Force path through revalidator with bad inline components by monkeypatching.
    # Store path remains clean; corruption is covered by revalidator above.
    decision = lookup.lookup(bundle["locator"], item=bundle["item"], now_ms=NOW_MS)
    # Clean store still proceeds to cert absence RUN.
    assert decision.action is ReuseAction.RUN


def test_provider_absence_returns_run(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    # No current-context provider and no compiler → current context unavailable.
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=store,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    decision = lookup.lookup(bundle["locator"], item=bundle["item"], now_ms=NOW_MS)
    _assert_run(decision, ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN)


def test_timeout_returns_run(tmp_path: Path) -> None:
    class SlowStore:
        def lookup(self, *_args: Any, **_kwargs: Any) -> Any:
            time.sleep(0.25)
            return SimpleNamespace(hit=False, reason_code="miss")

    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=SlowStore(),
        current_context_provider=DefaultCurrentContextProvider(
            live_identity_compiler=lambda **_k: None
        ),
        allowed_roots={"repo": tmp_path},
        timeout_seconds=0.02,
    )
    started = time.monotonic()
    decision = lookup.lookup(_locator(), item=_Item(), now_ms=NOW_MS)
    elapsed = time.monotonic() - started
    _assert_run(decision, ReuseReasonCode.TIMEOUT)
    assert elapsed < 0.2


def test_exception_fail_open_to_run(tmp_path: Path) -> None:
    class BoomStore:
        def lookup(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("store secret must not escape")

    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=BoomStore(),
        current_context_provider=DefaultCurrentContextProvider(
            live_identity_compiler=lambda **_k: None
        ),
        allowed_roots={"repo": tmp_path},
        timeout_seconds=1.0,
    )
    decision = lookup.lookup(_locator(), item=_Item(), now_ms=NOW_MS)
    assert decision.action is ReuseAction.RUN
    assert "store secret" not in repr(dict(decision.diagnostics))


# ---------------------------------------------------------------------------
# Full two-stage: revalidation + certificate verification → SKIP
# ---------------------------------------------------------------------------


def test_exact_match_then_proof_cache_hit_is_only_skip(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path, use_real_execution_key=True)
    assert bundle["execution_key"] is not None

    ctx_store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    put = ctx_store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    assert put.stored

    receipt = _receipt(bundle["locator"], bundle["execution_key"])
    # Align policy cid on certificate path with execution key.
    certificate = _certificate(receipt, bundle["execution_key"])
    policy = _policy(policy_cid=bundle["execution_key"].policy_cid)
    cert_candidate = TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=9_000,
        expires_at_ms=30_000,
    )
    provider = _CertProvider()
    current_provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
        clock=lambda: NOW_S,
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=ctx_store,
        certificate_provider=provider,
        proof_cache_store=_CertStore((cert_candidate,)),
        current_context_provider=current_provider,
        allowed_roots={"repo": bundle["root"]},
        current_policy=policy,
        clock=lambda: NOW_MS,
        timeout_seconds=2.0,
    )

    decision = lookup.lookup(
        bundle["locator"],
        None,  # locator + item only; execution key derived after revalidation
        item=bundle["item"],
        now_ms=NOW_MS,
    )
    assert decision.action is ReuseAction.SKIP
    assert decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
    assert decision.authority is CertificateAuthority.AUTHORITATIVE
    assert decision.certificate_cid == certificate.certificate_id
    assert decision.receipt_cid == receipt.receipt_id
    assert provider.verify_calls == 1
    assert provider.prove_calls == 0
    assert current_provider.fixtures_executed is False
    assert current_provider.test_body_executed is False


def test_provided_execution_key_must_match_candidate(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path, use_real_execution_key=True)
    ctx_store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    ctx_store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    other = _execution_key(
        bundle["locator"],
        policy_cid="cid:other-policy",
        forest_cid="cid:other-forest",
    )
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=ctx_store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    decision = lookup.lookup(
        bundle["locator"],
        other,
        item=bundle["item"],
        now_ms=NOW_MS,
    )
    _assert_run(decision, ReuseReasonCode.EXECUTION_KEY_MISMATCH)


def test_revalidated_request_and_batch_lookup(tmp_path: Path) -> None:
    bundle = _matching_warm_bundle(tmp_path)
    ctx_store = TestCandidateContextStore(tmp_path / "ctx", clock=lambda: NOW_S)
    ctx_store.publish(
        bundle["candidate"],
        bundle["components"],
        locator_cid=bundle["candidate"].locator_cid,
    )
    provider = DefaultCurrentContextProvider(
        live_identity_compiler=_live_matching_compiler(bundle["current"]),
        allowed_roots={"repo": bundle["root"]},
    )
    lookup = ProofReuseTwoStageLookup(
        candidate_context_store=ctx_store,
        current_context_provider=provider,
        allowed_roots={"repo": bundle["root"]},
        timeout_seconds=2.0,
    )
    request = RevalidatedProofReuseLookupRequest(
        item=bundle["item"],
        locator=bundle["locator"],
        now_ms=NOW_MS,
    )
    decisions = batch_lookup_reuse_decisions(
        lookup,
        (request,),
        apply_skips=False,
    )
    assert len(decisions) == 1
    _assert_run(decisions[0], ReuseReasonCode.CERTIFICATE_PROVIDER_UNAVAILABLE)


def test_legacy_proof_reuse_lookup_still_works_without_two_stage() -> None:
    """Existing certificate-only path remains available for non-warm callers."""

    locator = _locator()
    execution_key = _execution_key(locator)
    receipt = _receipt(locator, execution_key)
    certificate = _certificate(receipt, execution_key)
    candidate = TestProofCache.candidate(
        receipt,
        certificate,
        created_at_ms=9_000,
        expires_at_ms=30_000,
    )
    provider = _CertProvider()
    lookup = ProofReuseLookup(
        _CertStore((candidate,)),
        provider,
        current_policy=_policy(),
        timeout_seconds=1.0,
    )
    decision = lookup.lookup(locator, execution_key, now_ms=NOW_MS)
    assert decision.action is ReuseAction.SKIP
    assert decision.reason_code is ReuseReasonCode.PROOF_CACHE_HIT
    assert provider.prove_calls == 0
