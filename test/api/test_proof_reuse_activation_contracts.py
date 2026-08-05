"""PTR-131: seal automatic runtime activation and candidate-context contracts."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from ipfs_accelerate_py.testing.proof_reuse.activation_contracts import (
    ACTIVATION_AUTHORITY_SEQUENCE,
    ACTIVATION_CONTRACT_VERSION,
    CANDIDATE_EXECUTION_CONTEXT_INTERFACE,
    CURRENT_EXECUTION_CONTEXT_INTERFACE,
    LOCATOR_HINT_INTERFACE,
    PROOF_REUSE_ACTIVATION_CONTRACT_INTERFACE,
    RUNTIME_REUSE_DISPOSITION_INTERFACE,
    SKIP_REQUIRED_COMPARISON_DIMENSIONS,
    ActivationContractError,
    ArtifactRole,
    AuthoritativeCertificateBinding,
    CandidateExecutionContext,
    CurrentExecutionContext,
    DeferredProofRequest,
    LocatorHint,
    OptionalCapabilityFaultKind,
    PostPassRuntimeObservation,
    ProofReuseActivationContract,
    RuntimeReuseAction,
    RuntimeReuseDisposition,
    SkipComparisonDimension,
    TrustedPassReceiptBinding,
    admit_content_addressed_boundary,
    artifact_role_of,
    compare_contexts_for_skip,
    disposition_for_optional_capability_fault,
    disposition_run,
    disposition_skip,
    record_post_pass_runtime_observation,
    rehash_retained_canonical_bytes,
    require_content_addressed_boundary,
    roles_are_distinct,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _locator_hint(**changes: object) -> LocatorHint:
    values: dict[str, object] = {
        "locator_cid": "cid:locator:alpha",
        "candidate_context_cid": "cid:candidate:alpha",
        "certificate_cid": "cid:cert:alpha",
        "index_generation": 3,
        "repository_id": "repository:sha256:demo",
        "node_id": "test/api/test_demo.py::test_alpha",
        "selection_semantics": "exact_node",
    }
    values.update(changes)
    return LocatorHint(**values)  # type: ignore[arg-type]


def _candidate(**changes: object) -> CandidateExecutionContext:
    values: dict[str, object] = {
        "locator_cid": "cid:locator:alpha",
        "execution_key_cid": "cid:execution:alpha",
        "pass_receipt_cid": "cid:receipt:alpha",
        "repository_forest_cid": "cid:forest:v1",
        "test_ast_cid": "cid:ast:alpha",
        "static_trace_root_cid": "cid:static:alpha",
        "runtime_trace_root_cid": "cid:runtime:alpha",
        "environment_cid": "cid:env:alpha",
        "policy_cid": "cid:policy:alpha",
        "dependency_lock_cid": "cid:lock:alpha",
        "retained_at_ms": 1_700_000_000_000,
    }
    values.update(changes)
    return CandidateExecutionContext(**values)  # type: ignore[arg-type]


def _current(**changes: object) -> CurrentExecutionContext:
    values: dict[str, object] = {
        "locator_cid": "cid:locator:alpha",
        "execution_key_cid": "cid:execution:alpha",
        "repository_forest_cid": "cid:forest:v1",
        "test_ast_cid": "cid:ast:alpha",
        "static_trace_root_cid": "cid:static:alpha",
        "runtime_trace_root_cid": "cid:runtime:alpha",
        "environment_cid": "cid:env:alpha",
        "policy_cid": "cid:policy:alpha",
        "dependency_lock_cid": "cid:lock:alpha",
        "rebuild_source": "fresh_live_rebuild",
        "rebuilt_at_ms": 1_700_000_000_100,
    }
    values.update(changes)
    return CurrentExecutionContext(**values)  # type: ignore[arg-type]


def _receipt_binding(**changes: object) -> TrustedPassReceiptBinding:
    values: dict[str, object] = {
        "receipt_cid": "cid:receipt:alpha",
        "execution_key_cid": "cid:execution:alpha",
        "locator_cid": "cid:locator:alpha",
        "candidate_context_cid": "cid:candidate:alpha",
        "runtime_trace_root_cid": "cid:runtime:alpha",
        "admitted": True,
    }
    values.update(changes)
    return TrustedPassReceiptBinding(**values)  # type: ignore[arg-type]


def _deferred(**changes: object) -> DeferredProofRequest:
    values: dict[str, object] = {
        "receipt_cid": "cid:receipt:alpha",
        "execution_key_cid": "cid:execution:alpha",
        "candidate_context_cid": "cid:candidate:alpha",
        "statement_cid": "cid:statement:v1",
        "policy_cid": "cid:policy:alpha",
        "circuit_cid": "cid:circuit:v1",
        "verifying_key_cid": "cid:vk:v1",
        "issuer_id": "issuer:local",
        "epoch": "epoch:1",
        "locator_cid": "cid:locator:alpha",
        "public_inputs": {
            "receipt_cid": "cid:receipt:alpha",
            "execution_key_cid": "cid:execution:alpha",
        },
    }
    values.update(changes)
    return DeferredProofRequest(**values)  # type: ignore[arg-type]


def _certificate(
    candidate: CandidateExecutionContext | None = None, **changes: object
) -> AuthoritativeCertificateBinding:
    cand = candidate or _candidate()
    values: dict[str, object] = {
        "certificate_cid": "cid:cert:alpha",
        "receipt_cid": cand.pass_receipt_cid,
        "execution_key_cid": cand.execution_key_cid,
        "candidate_context_cid": cand.candidate_context_id,
        "statement_cid": "cid:statement:v1",
        "circuit_cid": "cid:circuit:v1",
        "verifying_key_cid": "cid:vk:v1",
        "policy_cid": cand.policy_cid,
        "issuer_id": "issuer:local",
        "epoch": "epoch:1",
        "authoritative": True,
        "simulated": False,
        "locally_verified": True,
    }
    values.update(changes)
    return AuthoritativeCertificateBinding(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Artifact role distinction
# ---------------------------------------------------------------------------


def test_contract_distinguishes_six_artifact_roles() -> None:
    roles = roles_are_distinct()
    assert roles == {
        "LOCATOR_HINT": "locator_hint",
        "IMMUTABLE_CANDIDATE_CONTEXT": "immutable_candidate_context",
        "FRESH_CURRENT_CONTEXT": "fresh_current_context",
        "TRUSTED_PASS_RECEIPT": "trusted_pass_receipt",
        "DEFERRED_PROOF_REQUEST": "deferred_proof_request",
        "AUTHORITATIVE_CERTIFICATE": "authoritative_certificate",
    }

    hint = _locator_hint()
    candidate = _candidate()
    current = _current()
    receipt = _receipt_binding()
    deferred = _deferred()
    certificate = _certificate(candidate)

    assert artifact_role_of(hint) is ArtifactRole.LOCATOR_HINT
    assert artifact_role_of(candidate) is ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT
    assert artifact_role_of(current) is ArtifactRole.FRESH_CURRENT_CONTEXT
    assert artifact_role_of(receipt) is ArtifactRole.TRUSTED_PASS_RECEIPT
    assert artifact_role_of(deferred) is ArtifactRole.DEFERRED_PROOF_REQUEST
    assert artifact_role_of(certificate) is ArtifactRole.AUTHORITATIVE_CERTIFICATE

    for artifact in (hint, candidate, current, receipt, deferred):
        assert artifact.may_authorize_skip is False
    assert certificate.may_authorize_skip is True

    # Roles are not interchangeable via wrong interface payloads.
    with pytest.raises(ActivationContractError):
        CandidateExecutionContext.from_dict(
            {**candidate.to_dict(), "role": ArtifactRole.LOCATOR_HINT.value}
        )


def test_round_trips_and_deterministic_identities() -> None:
    candidate = _candidate(component_cids={"b": "cid:b", "a": "cid:a"})
    again = _candidate(component_cids={"a": "cid:a", "b": "cid:b"})
    assert candidate.to_json() == again.to_json()
    assert candidate.candidate_context_id == again.candidate_context_id
    assert candidate.interface == CANDIDATE_EXECUTION_CONTEXT_INTERFACE
    assert CandidateExecutionContext.from_dict(candidate.to_dict()) == candidate

    current = _current()
    assert current.interface == CURRENT_EXECUTION_CONTEXT_INTERFACE
    assert CurrentExecutionContext.from_dict(current.to_dict()) == current

    hint = _locator_hint()
    assert hint.interface == LOCATOR_HINT_INTERFACE
    assert LocatorHint.from_dict(hint.to_dict()) == hint

    deferred = _deferred()
    assert DeferredProofRequest.from_dict(deferred.to_dict()) == deferred

    receipt = _receipt_binding()
    assert TrustedPassReceiptBinding.from_dict(receipt.to_dict()) == receipt

    certificate = _certificate(candidate)
    assert AuthoritativeCertificateBinding.from_dict(certificate.to_dict()) == certificate

    sealed = ProofReuseActivationContract.sealed()
    assert sealed.interface == PROOF_REUSE_ACTIVATION_CONTRACT_INTERFACE
    assert ProofReuseActivationContract.from_dict(sealed.to_dict()) == sealed
    assert sealed.contract_id.startswith("b")


# ---------------------------------------------------------------------------
# Content-addressed boundary: canonical bytes + CID rehash
# ---------------------------------------------------------------------------


def test_content_addressed_boundary_requires_canonical_bytes_and_rehash() -> None:
    candidate = _candidate()
    data = candidate.canonical_bytes()
    actual = rehash_retained_canonical_bytes(data)
    assert actual == candidate.candidate_context_id

    admitted = admit_content_addressed_boundary(
        role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
        claimed_cid=candidate.candidate_context_id,
        canonical_bytes=data,
    )
    assert admitted.admitted is True
    assert admitted.actual_cid == candidate.candidate_context_id
    assert admitted.byte_length == len(data)

    strict = require_content_addressed_boundary(
        role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
        claimed_cid=candidate.candidate_context_id,
        canonical_bytes=data,
    )
    assert strict.admitted is True

    # Tampered bytes fail rehash equality.
    tampered = data[:-1] + (b"X" if data[-1:] != b"X" else b"Y")
    rejected = admit_content_addressed_boundary(
        role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
        claimed_cid=candidate.candidate_context_id,
        canonical_bytes=tampered,
    )
    assert rejected.admitted is False
    assert rejected.reason_code == "candidate_integrity_failed"

    # Wrong claimed CID fails closed without raising.
    wrong = admit_content_addressed_boundary(
        role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
        claimed_cid="cid:forged",
        canonical_bytes=data,
    )
    assert wrong.admitted is False
    assert wrong.actual_cid == candidate.candidate_context_id

    with pytest.raises(ActivationContractError):
        require_content_addressed_boundary(
            role=ArtifactRole.IMMUTABLE_CANDIDATE_CONTEXT,
            claimed_cid="cid:forged",
            canonical_bytes=data,
        )

    # Non-canonical formatting (pretty JSON) is rejected.
    pretty = json.dumps(json.loads(data.decode("utf-8")), indent=2).encode("utf-8")
    noncanonical = admit_content_addressed_boundary(
        role=ArtifactRole.TRUSTED_PASS_RECEIPT,
        claimed_cid=candidate.candidate_context_id,
        canonical_bytes=pretty,
    )
    assert noncanonical.admitted is False


def test_every_activation_artifact_is_content_addressed() -> None:
    artifacts = [
        _locator_hint(),
        _candidate(),
        _current(),
        _receipt_binding(),
        _deferred(),
        _certificate(),
        ProofReuseActivationContract.sealed(),
        disposition_run("candidate_missing"),
    ]
    for artifact in artifacts:
        data = artifact.canonical_bytes()
        assert rehash_retained_canonical_bytes(data) == artifact.content_id
        assert artifact.content_id.startswith("b")


# ---------------------------------------------------------------------------
# Pre-SKIP comparison of current AST/static/runtime/environment/policy
# ---------------------------------------------------------------------------


def test_skip_requires_current_ast_static_runtime_environment_policy_match() -> None:
    assert tuple(dim.value for dim in SKIP_REQUIRED_COMPARISON_DIMENSIONS) == (
        "ast",
        "static",
        "runtime",
        "environment",
        "policy",
    )

    candidate = _candidate()
    current = _current()
    comparison = compare_contexts_for_skip(candidate, current)
    assert comparison.matched is True

    for dimension, field_name in (
        (SkipComparisonDimension.AST, "test_ast_cid"),
        (SkipComparisonDimension.STATIC, "static_trace_root_cid"),
        (SkipComparisonDimension.RUNTIME, "runtime_trace_root_cid"),
        (SkipComparisonDimension.ENVIRONMENT, "environment_cid"),
        (SkipComparisonDimension.POLICY, "policy_cid"),
    ):
        mutated = _current(**{field_name: "cid:changed"})
        result = compare_contexts_for_skip(candidate, mutated)
        assert result.matched is False
        assert dimension.value in result.mismatched_dimensions

    sealed = ProofReuseActivationContract.sealed()
    certificate = _certificate(candidate)
    skip = sealed.evaluate_skip_admission(
        candidate=candidate,
        current=current,
        certificate=certificate,
        candidate_bytes=candidate.canonical_bytes(),
    )
    assert skip.action is RuntimeReuseAction.SKIP
    assert skip.collection_failed is False

    run = sealed.evaluate_skip_admission(
        candidate=candidate,
        current=_current(test_ast_cid="cid:ast:mutated"),
        certificate=certificate,
        candidate_bytes=candidate.canonical_bytes(),
    )
    assert run.action is RuntimeReuseAction.RUN
    assert run.is_skip is False


def test_current_context_rejects_historical_trace_relabel() -> None:
    with pytest.raises(ActivationContractError, match="fresh source"):
        _current(rebuild_source="historical_trace")
    with pytest.raises(ActivationContractError, match="fresh source"):
        _current(rebuild_source="prior_pass_trace")


def test_simulated_certificate_never_skips() -> None:
    candidate = _candidate()
    current = _current()
    with pytest.raises(ActivationContractError, match="simulated"):
        _certificate(candidate, simulated=True, authoritative=True)

    simulated = _certificate(
        candidate, simulated=True, authoritative=False, locally_verified=True
    )
    assert simulated.may_authorize_skip is False
    disposition = ProofReuseActivationContract.sealed().evaluate_skip_admission(
        candidate=candidate,
        current=current,
        certificate=simulated,
    )
    assert disposition.action is RuntimeReuseAction.RUN
    assert disposition.reason_code == "certificate_non_attested"


# ---------------------------------------------------------------------------
# Post-pass observations without duplicating the test call
# ---------------------------------------------------------------------------


def test_post_pass_runtime_observation_forbids_duplicate_test_call() -> None:
    observation = record_post_pass_runtime_observation(
        locator_cid="cid:locator:alpha",
        execution_key_cid="cid:execution:alpha",
        runtime_trace_root_cid="cid:runtime:alpha",
        pass_receipt_cid="cid:receipt:alpha",
        test_call_count=1,
        setup_call_count=1,
        teardown_call_count=1,
        observed_at_ms=42,
    )
    assert observation.test_call_count == 1
    assert observation.duplicate_test_call_forbidden is True
    assert observation.observation_source == "post_pass_lifecycle"
    assert PostPassRuntimeObservation.from_dict(observation.to_dict()) == observation

    with pytest.raises(ActivationContractError, match="exactly one test call"):
        record_post_pass_runtime_observation(
            locator_cid="cid:locator:alpha",
            execution_key_cid="cid:execution:alpha",
            runtime_trace_root_cid="cid:runtime:alpha",
            pass_receipt_cid="cid:receipt:alpha",
            test_call_count=2,
        )

    with pytest.raises(ActivationContractError, match="post-pass"):
        PostPassRuntimeObservation(
            locator_cid="cid:locator:alpha",
            execution_key_cid="cid:execution:alpha",
            runtime_trace_root_cid="cid:runtime:alpha",
            pass_receipt_cid="cid:receipt:alpha",
            observation_source="pre_execution_prediction",
        )


# ---------------------------------------------------------------------------
# Optional capability faults → RUN or DEFERRED, never collection failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fault",
    list(OptionalCapabilityFaultKind),
)
def test_optional_capability_faults_map_to_run_or_deferred_without_collection_failure(
    fault: OptionalCapabilityFaultKind,
) -> None:
    # Without a retained receipt, every fault is RUN.
    run = disposition_for_optional_capability_fault(
        fault,
        capability="groth16_endpoint_or_binary",
        receipt_retained=False,
    )
    assert run.action is RuntimeReuseAction.RUN
    assert run.collection_failed is False
    assert run.interface == RUNTIME_REUSE_DISPOSITION_INTERFACE
    assert "optional_capability_" in run.reason_code

    # With a retained receipt, missing/incompatible/timed_out prefer DEFERRED.
    retained = disposition_for_optional_capability_fault(
        fault,
        capability="provekit_binary_and_artifacts",
        receipt_retained=True,
        receipt_cid="cid:receipt:alpha",
        candidate_context_cid="cid:candidate:alpha",
    )
    assert retained.collection_failed is False
    if fault in {
        OptionalCapabilityFaultKind.MISSING,
        OptionalCapabilityFaultKind.INCOMPATIBLE,
        OptionalCapabilityFaultKind.TIMED_OUT,
    }:
        assert retained.action is RuntimeReuseAction.DEFERRED
        assert retained.receipt_cid == "cid:receipt:alpha"
    else:
        assert retained.action is RuntimeReuseAction.RUN

    # SKIP is unreachable from optional capability mapping.
    assert run.action is not RuntimeReuseAction.SKIP
    assert retained.action is not RuntimeReuseAction.SKIP


def test_disposition_never_allows_collection_failed_true() -> None:
    with pytest.raises(ActivationContractError, match="never fail collection"):
        RuntimeReuseDisposition(
            action=RuntimeReuseAction.RUN,
            reason_code="candidate_missing",
            collection_failed=True,
        )


def test_skip_disposition_requires_certificate_and_receipt() -> None:
    with pytest.raises(ActivationContractError, match="certificate_cid"):
        disposition_skip(certificate_cid="", receipt_cid="cid:receipt:alpha")
    with pytest.raises(ActivationContractError, match="receipt_cid"):
        disposition_skip(certificate_cid="cid:cert:alpha", receipt_cid="")


# ---------------------------------------------------------------------------
# Sealed activation contract doctrine
# ---------------------------------------------------------------------------


def test_sealed_activation_contract_authority_sequence_and_gates() -> None:
    sealed = ProofReuseActivationContract.sealed()
    assert sealed.authority_sequence == ACTIVATION_AUTHORITY_SEQUENCE
    assert sealed.content_addressed_rehash_required is True
    assert sealed.post_pass_observation_without_duplicate_call is True
    assert sealed.optional_capability_collection_failure_forbidden is True
    assert sealed.skip_required_dimensions == (
        "ast",
        "static",
        "runtime",
        "environment",
        "policy",
    )
    assert sealed.role_may_authorize_skip(ArtifactRole.LOCATOR_HINT) is False
    assert sealed.role_may_authorize_skip(ArtifactRole.TRUSTED_PASS_RECEIPT) is False
    assert (
        sealed.role_may_authorize_skip(ArtifactRole.AUTHORITATIVE_CERTIFICATE) is True
    )

    with pytest.raises(ActivationContractError, match="sealed activation sequence"):
        ProofReuseActivationContract(
            authority_sequence=ACTIVATION_AUTHORITY_SEQUENCE[:-1]
        )

    with pytest.raises(ActivationContractError, match="rehash is mandatory"):
        ProofReuseActivationContract(content_addressed_rehash_required=False)


def test_private_material_rejected_from_public_artifacts() -> None:
    with pytest.raises(ActivationContractError):
        _candidate(metadata={"api_key": "should-not-appear"})
    with pytest.raises(ActivationContractError):
        _deferred(public_inputs={"private_witness": "w"})


def test_versionless_and_unknown_fields_fail_closed() -> None:
    candidate = _candidate()
    payload = candidate.to_dict()
    stripped = {
        key: value
        for key, value in payload.items()
        if key not in {"interface", "contract_version"}
    }
    with pytest.raises(ActivationContractError, match="versionless"):
        CandidateExecutionContext.from_dict(stripped)

    forged = deepcopy(payload)
    forged["unexpected_field"] = "x"
    with pytest.raises(ActivationContractError, match="unsupported fields"):
        CandidateExecutionContext.from_dict(forged)

    forged_version = deepcopy(payload)
    forged_version["contract_version"] = ACTIVATION_CONTRACT_VERSION + 1
    with pytest.raises(ActivationContractError, match="contract_version"):
        CandidateExecutionContext.from_dict(forged_version)


def test_evaluate_skip_admission_rejects_rehash_mismatch_without_raising() -> None:
    candidate = _candidate()
    current = _current()
    certificate = _certificate(candidate)
    disposition = ProofReuseActivationContract.sealed().evaluate_skip_admission(
        candidate=candidate,
        current=current,
        certificate=certificate,
        candidate_bytes=b'{"not":"matching"}',
    )
    assert disposition.action is RuntimeReuseAction.RUN
    assert disposition.collection_failed is False
