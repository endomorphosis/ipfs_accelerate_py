"""Contracts for program-analysis ZK public inputs, witness policy, and traces."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.program_analysis_zkp import (
    PROGRAM_ZKP_EVIDENCE_TRACE_STATEMENT,
    PROGRAM_ZKP_EVIDENCE_VERIFICATION_RECEIPT,
    PROGRAM_ZKP_G080_EVIDENCE_TERMS,
    PUBLIC_COMMITMENT_KEYS,
    PUBLIC_INPUT_CODEC_ID,
    PUBLIC_INPUT_CODEC_VERSION,
    TRACE_VALIDITY_DOES_NOT_PROVE,
    TRACE_VALIDITY_SCOPE_STATEMENT,
    PrivateProgramAnalysisWitness,
    ProgramAnalysisZkpError,
    ProgramZkpBackendMode,
    ProgramZkpClaimPromotionError,
    ProgramZkpPublicInputs,
    ProgramZkpReplayError,
    ProgramZkpShadowEnvelope,
    ProgramZkpStatement,
    ProgramZkpTamperError,
    ProgramZkpTrace,
    ProgramZkpTraceError,
    ProgramZkpTraceStep,
    ProgramZkpTrust,
    ProgramZkpVerdict,
    ProgramZkpVersionError,
    ProgramZkpWitnessDisclosureError,
    ProgramZkpWitnessPolicy,
    assert_trace_non_claims,
    build_canonical_program_zkp_trace,
    build_program_zkp_public_inputs,
    claim_level_for_verified_trace,
    commitment_identity,
    create_program_zkp_shadow_envelope,
    encode_public_input_vector,
    next_trace_state,
    prepare_program_analysis_zkp,
    public_artifact_contains,
    public_input_vector_digest,
    public_program_zkp_artifact,
    record_program_zkp_verification,
    reject_illegal_zk_claim_promotion,
    reject_private_witness_from_public_payload,
    supported_transition_table,
    trace_validity_does_not_prove,
    TraceState,
    TraceTransitionKind,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import (
    ClaimLevel,
    ClaimPromotionError,
    SemanticAuthorityError,
)


def _public_inputs(**overrides: str) -> ProgramZkpPublicInputs:
    base = {
        "forest_commitment": commitment_identity("forest", {"root": "repo:alpha"}),
        "inventory_commitment": commitment_identity(
            "inventory", {"files": ["a.py", "b.py"]}
        ),
        "contract_commitment": commitment_identity(
            "contract", {"symbol": "pkg.api.call"}
        ),
        "call_slice_commitment": commitment_identity(
            "call_slice", {"path": ["main", "pkg.api.call"]}
        ),
        "assumptions_commitment": commitment_identity(
            "assumptions", {"items": ["finite_bounds", "hermetic_fixture"]}
        ),
        "analyzer_version": "analyzer:program-graph@1.0.0",
        "resolver_version": "resolver:call@2.1.0",
        "translator_version": "translator:contract-ir@1.3.0",
        "prover_version": "prover:shadow-trace@0.1.0",
        "result_commitment": commitment_identity(
            "result", {"status": "contract_check_ok", "finite": True}
        ),
        "circuit_id": "circuit:program-contract-trace@1",
        "proving_key_id": "pk:program-contract-trace@1:sha256-pk",
        "verifying_key_id": "vk:program-contract-trace@1:sha256-vk",
        "ceremony_id": "ceremony:program-contract-trace@1",
        "public_input_codec_id": PUBLIC_INPUT_CODEC_ID,
        "public_input_codec_version": PUBLIC_INPUT_CODEC_VERSION,
    }
    base.update(overrides)
    return build_program_zkp_public_inputs(**base)


def _witness(secret: str = "opening-secret-never-publish") -> PrivateProgramAnalysisWitness:
    return PrivateProgramAnalysisWitness(
        {
            "source_text": "def call():\n    return secret\n",
            "ast_node": {"kind": "FunctionDef", "name": "call"},
            "proof_trace": {"steps": ["open", "check", "commit"]},
            "commitment_opening": secret,
        }
    )


def _request(secret: str = "opening-secret-never-publish"):
    return prepare_program_analysis_zkp(
        _public_inputs(),
        witness=_witness(secret),
        backend_mode=ProgramZkpBackendMode.SHADOW,
    )


def _envelope(secret: str = "opening-secret-never-publish") -> ProgramZkpShadowEnvelope:
    return create_program_zkp_shadow_envelope(
        _request(secret),
        proof_artifact_id="artifact:zk-proof-shadow",
        proof_digest="sha256:" + ("ab" * 32),
        prover_id="prover:shadow-worker",
    )


def _receipt(
    *,
    verdict: ProgramZkpVerdict = ProgramZkpVerdict.VERIFIED,
) -> object:
    return record_program_zkp_verification(
        _envelope(),
        verdict=verdict,
        verifier_id="verifier:program-analysis-zkp@1",
    )


# ---------------------------------------------------------------------------
# Public commitments and codec
# ---------------------------------------------------------------------------


def test_public_inputs_bind_all_canonical_commitments() -> None:
    inputs = _public_inputs()
    public = inputs.public_inputs

    assert tuple(public.keys()) == PUBLIC_COMMITMENT_KEYS
    for key in PUBLIC_COMMITMENT_KEYS:
        assert public[key]
    assert inputs.public_input_codec_id == PUBLIC_INPUT_CODEC_ID
    assert inputs.public_input_codec_version == PUBLIC_INPUT_CODEC_VERSION
    assert inputs.public_input_digest.startswith("b")
    assert inputs.content_id.startswith("b")
    assert ProgramZkpPublicInputs.from_dict(inputs.to_dict()) == inputs
    vector = encode_public_input_vector(public)
    assert len(vector) == len(PUBLIC_COMMITMENT_KEYS)
    assert public_input_vector_digest(public) == inputs.public_input_digest


def test_every_public_commitment_changes_identity() -> None:
    baseline = _public_inputs()
    for key in PUBLIC_COMMITMENT_KEYS:
        if key in {"public_input_codec_id", "public_input_codec_version"}:
            continue
        changed = baseline.with_overrides(**{key: "tampered:%s" % key})
        assert changed.public_input_digest != baseline.public_input_digest
        assert changed.content_id != baseline.content_id


def test_public_inputs_reject_wrong_codec_version() -> None:
    with pytest.raises(ProgramZkpVersionError, match="public_input_codec_version"):
        _public_inputs(public_input_codec_version="999")
    with pytest.raises(ProgramZkpVersionError, match="public_input_codec_id"):
        _public_inputs(public_input_codec_id="codec:wrong")


def test_encode_public_input_vector_rejects_missing_or_extra_keys() -> None:
    public = dict(_public_inputs().public_inputs)
    del public["forest_commitment"]
    with pytest.raises(ProgramAnalysisZkpError, match="keys mismatch"):
        encode_public_input_vector(public)

    public = dict(_public_inputs().public_inputs)
    public["extra_slot"] = "nope"
    with pytest.raises(ProgramAnalysisZkpError, match="keys mismatch"):
        encode_public_input_vector(public)


def test_trace_validity_non_claims_are_normative() -> None:
    assert "inventory_completeness" in TRACE_VALIDITY_DOES_NOT_PROVE
    assert "translator_soundness" in TRACE_VALIDITY_DOES_NOT_PROVE
    assert "arbitrary_runtime_semantics" in TRACE_VALIDITY_DOES_NOT_PROVE
    assert "theorem_beyond_committed_supported_result" in TRACE_VALIDITY_DOES_NOT_PROVE
    for claim in TRACE_VALIDITY_DOES_NOT_PROVE:
        assert trace_validity_does_not_prove(claim)
    assert not trace_validity_does_not_prove("commitment_openings")
    assert "inventory completeness" in TRACE_VALIDITY_SCOPE_STATEMENT.lower()
    inputs = _public_inputs()
    assert_trace_non_claims(inputs)
    statement = prepare_program_analysis_zkp(
        inputs, witness=_witness()
    ).statement
    assert_trace_non_claims(statement)


# ---------------------------------------------------------------------------
# Deterministic trace transitions
# ---------------------------------------------------------------------------


def test_supported_transition_table_is_complete_and_deterministic() -> None:
    table = supported_transition_table()
    assert len(table) == 8
    assert table[0]["source"] == TraceState.INIT.value
    assert table[-1]["target"] == TraceState.TERMINAL.value
    # Each source appears once (linear chain).
    sources = [row["source"] for row in table]
    assert len(sources) == len(set(sources))


def test_canonical_trace_follows_supported_transitions() -> None:
    inputs = _public_inputs()
    trace = build_canonical_program_zkp_trace(inputs)

    assert trace.is_complete
    assert trace.terminal_state is TraceState.TERMINAL
    assert trace.result_commitment == inputs.result_commitment
    assert trace.public_input_digest == inputs.public_input_digest
    assert ProgramZkpTrace.from_dict(trace.to_dict()) == trace
    assert_trace_non_claims(trace)


def test_trace_rejects_reordered_steps() -> None:
    inputs = _public_inputs()
    trace = build_canonical_program_zkp_trace(inputs)
    steps = list(trace.steps)
    # Each step remains an individually supported transition triple, but the
    # sequence is reordered so kinds and chained states no longer match.
    swapped = list(steps)
    swapped[1], swapped[2] = steps[2], steps[1]
    reindexed = tuple(
        ProgramZkpTraceStep(
            index=index,
            kind=step.kind,
            source_state=step.source_state,
            target_state=step.target_state,
            binding_commitment=step.binding_commitment,
        )
        for index, step in enumerate(swapped)
    )
    with pytest.raises(ProgramZkpTraceError):
        ProgramZkpTrace(
            steps=reindexed,
            result_commitment=inputs.result_commitment,
            public_input_digest=inputs.public_input_digest,
        )


def test_trace_rejects_omitted_steps() -> None:
    inputs = _public_inputs()
    trace = build_canonical_program_zkp_trace(inputs)
    shortened = trace.steps[:-1]
    with pytest.raises(ProgramZkpTraceError, match="exactly the supported"):
        ProgramZkpTrace(
            steps=shortened,
            result_commitment=inputs.result_commitment,
            public_input_digest=inputs.public_input_digest,
        )


def test_trace_rejects_illegal_single_transition() -> None:
    with pytest.raises(ProgramZkpTraceError, match="unsupported transition"):
        next_trace_state(TraceState.INIT, TraceTransitionKind.COMMIT_RESULT)
    with pytest.raises(ProgramZkpTraceError, match="not a supported transition"):
        ProgramZkpTraceStep(
            index=0,
            kind=TraceTransitionKind.TERMINATE,
            source_state=TraceState.INIT,
            target_state=TraceState.TERMINAL,
        )


def test_trace_rejects_forged_result_binding() -> None:
    inputs = _public_inputs()
    with pytest.raises(ProgramZkpTraceError, match="committed result"):
        build_canonical_program_zkp_trace(
            inputs,
            binding_commitments={
                TraceTransitionKind.COMMIT_RESULT.value: "result:forged"
            },
        )


# ---------------------------------------------------------------------------
# Statement and shadow envelope
# ---------------------------------------------------------------------------


def test_statement_is_non_authoritative_and_non_semantic() -> None:
    request = _request()
    statement = request.statement

    assert statement.claim_level is ClaimLevel.ZK_TRACE_ATTESTED
    assert statement.semantic_proof is False
    assert statement.authoritative is False
    assert statement.trust is ProgramZkpTrust.NON_AUTHORITATIVE
    assert set(statement.does_not_prove) == set(TRACE_VALIDITY_DOES_NOT_PROVE)
    assert claim_level_for_verified_trace() is ClaimLevel.ZK_TRACE_ATTESTED
    assert ProgramZkpStatement.from_dict(statement.to_dict()) == statement


def test_vfs_g080_evidence_terms_are_published_on_statement_and_receipt() -> None:
    """Cover vfs/zk-trace-statement@1 and vfs/zk-verification-receipt@1."""

    assert PROGRAM_ZKP_G080_EVIDENCE_TERMS == (
        "vfs/zk-trace-statement@1",
        "vfs/zk-verification-receipt@1",
    )
    assert PROGRAM_ZKP_EVIDENCE_TRACE_STATEMENT == "vfs/zk-trace-statement@1"
    assert (
        PROGRAM_ZKP_EVIDENCE_VERIFICATION_RECEIPT == "vfs/zk-verification-receipt@1"
    )

    statement = _request().statement
    receipt = _receipt()

    assert statement.evidence == PROGRAM_ZKP_EVIDENCE_TRACE_STATEMENT
    assert receipt.evidence == PROGRAM_ZKP_EVIDENCE_VERIFICATION_RECEIPT
    assert statement.backend_mode is ProgramZkpBackendMode.SHADOW
    assert receipt.backend_mode is ProgramZkpBackendMode.SHADOW

    statement_payload = statement.to_public_artifact()
    receipt_payload = receipt.to_public_artifact()
    assert statement_payload["evidence"] == "vfs/zk-trace-statement@1"
    assert receipt_payload["evidence"] == "vfs/zk-verification-receipt@1"
    assert receipt_payload["statement_evidence"] == "vfs/zk-trace-statement@1"
    # Bound circuit / key / codec / ceremony identities remain public.
    for key in (
        "circuit_id",
        "proving_key_id",
        "verifying_key_id",
        "ceremony_id",
        "public_input_codec_id",
        "public_input_codec_version",
    ):
        assert statement_payload[key]
        assert statement_payload[key] == getattr(statement.public_inputs, key)
    # Round-trip preserves evidence and identity.
    assert ProgramZkpStatement.from_dict(statement_payload).evidence == (
        PROGRAM_ZKP_EVIDENCE_TRACE_STATEMENT
    )
    assert type(receipt).from_dict(receipt_payload).evidence == (
        PROGRAM_ZKP_EVIDENCE_VERIFICATION_RECEIPT
    )


def test_statement_rejects_forged_evidence_identity() -> None:
    statement = _request().statement
    forged = dict(statement.to_dict())
    forged["evidence"] = "vfs/zk-capability-conformance@1"
    with pytest.raises(ProgramZkpTamperError, match="statement evidence"):
        ProgramZkpStatement.from_dict(forged)


def test_receipt_rejects_forged_evidence_identity() -> None:
    receipt = _receipt()
    forged = dict(receipt.to_dict())
    forged["evidence"] = "vfs/zk-trace-statement@1"
    with pytest.raises(ProgramZkpTamperError, match="receipt evidence"):
        type(receipt).from_dict(forged)


def test_statement_rejects_semantic_proof_flag() -> None:
    inputs = _public_inputs()
    trace = build_canonical_program_zkp_trace(inputs)
    with pytest.raises(SemanticAuthorityError, match="semantic proof"):
        ProgramZkpStatement(
            public_inputs=inputs,
            trace_id=trace.trace_id,
            semantic_proof=True,
        )


def test_shadow_envelope_is_non_authoritative() -> None:
    envelope = _envelope()
    payload = envelope.to_public_artifact()

    assert envelope.authoritative is False
    assert envelope.simulated is True
    assert payload["authoritative"] is False
    assert "shadow" in payload["non_authoritative_reason"]
    assert ProgramZkpShadowEnvelope.from_dict(payload) == envelope


# ---------------------------------------------------------------------------
# Tampering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    (
        "forest_commitment",
        "inventory_commitment",
        "contract_commitment",
        "call_slice_commitment",
        "assumptions_commitment",
        "result_commitment",
        "circuit_id",
        "proving_key_id",
        "verifying_key_id",
        "ceremony_id",
        "analyzer_version",
        "resolver_version",
        "translator_version",
        "prover_version",
    ),
)
def test_tampered_public_input_field_is_rejected_on_load(field_name: str) -> None:
    inputs = _public_inputs()
    forged = inputs.to_dict()
    forged[field_name] = "tampered-value"
    # Keep the original digest → tamper detection.
    with pytest.raises(ProgramZkpTamperError, match="public-input digest"):
        ProgramZkpPublicInputs.from_dict(forged)


def test_tampered_statement_identity_is_rejected() -> None:
    statement = _request().statement
    forged = statement.to_public_artifact()
    forged["trace_id"] = "trace:forged"
    with pytest.raises(ProgramZkpTamperError, match="statement identity"):
        ProgramZkpStatement.from_dict(forged)


def test_tampered_statement_digest_is_rejected() -> None:
    statement = _request().statement
    forged = statement.to_public_artifact()
    forged["public_input_digest"] = "digest:forged"
    with pytest.raises(ProgramZkpTamperError, match="public-input digest"):
        ProgramZkpStatement.from_dict(forged)


def test_envelope_cannot_assert_authority() -> None:
    envelope = _envelope()
    forged = envelope.to_public_artifact()
    forged["authoritative"] = True
    with pytest.raises(ProgramZkpTamperError, match="cannot assert authority"):
        ProgramZkpShadowEnvelope.from_dict(forged)


def test_statement_cannot_assert_authority() -> None:
    statement = _request().statement
    forged = statement.to_public_artifact()
    forged["authoritative"] = True
    with pytest.raises(ProgramZkpTamperError, match="cannot assert authority"):
        ProgramZkpStatement.from_dict(forged)


def test_receipt_rejects_mismatched_key_or_circuit_pins() -> None:
    envelope = _envelope()
    with pytest.raises(ProgramZkpTamperError, match="verifying_key_id"):
        record_program_zkp_verification(
            envelope,
            verdict=ProgramZkpVerdict.VERIFIED,
            verifier_id="verifier:x",
        ).__class__(
            statement=envelope.statement,
            verdict=ProgramZkpVerdict.VERIFIED,
            verifier_id="verifier:x",
            verifying_key_id="vk:wrong",
            circuit_id=envelope.statement.public_inputs.circuit_id,
            public_input_digest=envelope.statement.public_input_digest,
            ceremony_id=envelope.statement.public_inputs.ceremony_id,
            public_input_codec_version=PUBLIC_INPUT_CODEC_VERSION,
            backend_mode=ProgramZkpBackendMode.SHADOW,
        )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_verification_receipt_replays_against_exact_inputs() -> None:
    receipt = _receipt()
    inputs = _public_inputs()
    receipt.require_replay(
        public_inputs=inputs,
        verifying_key_id=inputs.verifying_key_id,
        circuit_id=inputs.circuit_id,
        ceremony_id=inputs.ceremony_id,
        public_input_codec_version=PUBLIC_INPUT_CODEC_VERSION,
    )


def test_replay_rejects_drifted_public_inputs() -> None:
    receipt = _receipt()
    drifted = _public_inputs(
        forest_commitment=commitment_identity("forest", {"root": "repo:other"})
    )
    with pytest.raises(ProgramZkpReplayError, match="public-input digest"):
        receipt.require_replay(
            public_inputs=drifted,
            verifying_key_id=drifted.verifying_key_id,
            circuit_id=drifted.circuit_id,
            ceremony_id=drifted.ceremony_id,
        )


def test_replay_rejects_stale_verifying_key() -> None:
    receipt = _receipt()
    inputs = _public_inputs()
    with pytest.raises(ProgramZkpReplayError, match="verifying_key_id"):
        receipt.require_replay(
            public_inputs=inputs,
            verifying_key_id="vk:stale",
            circuit_id=inputs.circuit_id,
            ceremony_id=inputs.ceremony_id,
        )


def test_replay_rejects_wrong_circuit_or_ceremony() -> None:
    receipt = _receipt()
    inputs = _public_inputs()
    with pytest.raises(ProgramZkpReplayError, match="circuit_id"):
        receipt.require_replay(
            public_inputs=inputs,
            verifying_key_id=inputs.verifying_key_id,
            circuit_id="circuit:other@9",
            ceremony_id=inputs.ceremony_id,
        )
    with pytest.raises(ProgramZkpReplayError, match="ceremony_id"):
        receipt.require_replay(
            public_inputs=inputs,
            verifying_key_id=inputs.verifying_key_id,
            circuit_id=inputs.circuit_id,
            ceremony_id="ceremony:other",
        )


def test_replay_rejects_codec_version_drift() -> None:
    receipt = _receipt()
    inputs = _public_inputs()
    with pytest.raises(ProgramZkpVersionError, match="codec version"):
        receipt.require_replay(
            public_inputs=inputs,
            verifying_key_id=inputs.verifying_key_id,
            circuit_id=inputs.circuit_id,
            ceremony_id=inputs.ceremony_id,
            public_input_codec_version="2",
        )


def test_replay_rejects_stale_proving_key_via_public_input_digest() -> None:
    """Stale proving-key identity drifts the public-input digest and fails replay."""

    receipt = _receipt()
    stale_pk = _public_inputs(proving_key_id="pk:program-contract-trace@1:stale")
    with pytest.raises(ProgramZkpReplayError, match="public-input digest"):
        receipt.require_replay(
            public_inputs=stale_pk,
            verifying_key_id=stale_pk.verifying_key_id,
            circuit_id=stale_pk.circuit_id,
            ceremony_id=stale_pk.ceremony_id,
        )


def test_shadow_verified_receipt_is_still_non_authoritative() -> None:
    receipt = _receipt(verdict=ProgramZkpVerdict.VERIFIED)
    assert receipt.verified is True
    assert receipt.authoritative is False
    assert receipt.trust is ProgramZkpTrust.NON_AUTHORITATIVE
    assert receipt.claim_level is ClaimLevel.ZK_TRACE_ATTESTED
    payload = receipt.to_public_artifact()
    assert payload["semantic_proof"] is False
    assert payload["authoritative"] is False


# ---------------------------------------------------------------------------
# Privacy leak
# ---------------------------------------------------------------------------


def test_private_witness_is_redacted_non_mapping_and_non_serializable() -> None:
    secret = "never-log-opening-9f2937"
    witness = _witness(secret)

    assert secret not in repr(witness)
    assert "commitment_opening" not in repr(witness)
    assert not isinstance(witness, Mapping)
    assert witness.redacted() == {"private_witness_redacted": True}
    with pytest.raises(ProgramZkpWitnessDisclosureError):
        witness.to_dict()
    with pytest.raises(ProgramZkpWitnessDisclosureError):
        pickle.dumps(witness)
    with pytest.raises(ProgramZkpWitnessDisclosureError):
        public_program_zkp_artifact(witness)


def test_proving_callback_can_use_read_only_witness_without_publication() -> None:
    request = _request(secret="callback-only-secret")

    def consume(values: Mapping[str, object]) -> tuple[str, bool]:
        with pytest.raises(TypeError):
            values["extra"] = "forbidden"  # type: ignore[index]
        return str(values["commitment_opening"]), "source_text" in values

    assert request.use_witness(consume) == ("callback-only-secret", True)


def test_witness_fields_are_absent_from_logs_context_and_public_artifacts() -> None:
    secret = "s3cr3t-witness-value-unique"
    request = _request(secret=secret)

    for artifact in (
        request.to_dict(),
        request.to_log_record(),
        request.to_context_capsule(),
        public_program_zkp_artifact({"request": request}),
        request.statement.to_public_artifact(),
        _envelope(secret).to_public_artifact(),
        _receipt().to_public_artifact(),
    ):
        encoded = json.dumps(artifact, sort_keys=True)
        assert secret not in encoded
        assert "commitment_opening" not in encoded
        assert "source_text" not in encoded
        assert "proof_trace" not in encoded
        assert not public_artifact_contains(artifact, secret)

    assert secret not in repr(request)
    with pytest.raises(ProgramZkpWitnessDisclosureError, match="cannot be cached"):
        request.to_cache_record()
    with pytest.raises(ProgramZkpWitnessDisclosureError):
        pickle.dumps(request)


def test_reject_private_witness_from_public_payload() -> None:
    with pytest.raises(ProgramZkpWitnessDisclosureError):
        reject_private_witness_from_public_payload(
            {"private_witness": {"opening": "x"}}
        )
    with pytest.raises(ProgramZkpWitnessDisclosureError):
        reject_private_witness_from_public_payload(
            {"nested": {"witness_opening": "x"}}
        )
    # Safe redaction marker is allowed.
    reject_private_witness_from_public_payload({"private_witness_redacted": True})


def test_witness_policy_requires_redaction_and_limits_fields() -> None:
    with pytest.raises(ProgramAnalysisZkpError, match="redact"):
        ProgramZkpWitnessPolicy(redact_from_public_artifacts=False)
    with pytest.raises(ProgramAnalysisZkpError, match="redact"):
        ProgramZkpWitnessPolicy(redact_from_logs=False)
    with pytest.raises(ProgramAnalysisZkpError, match="redact"):
        ProgramZkpWitnessPolicy(redact_from_cache=False)

    policy = ProgramZkpWitnessPolicy(allow_source_openings=False, max_opening_fields=2)
    with pytest.raises(ProgramAnalysisZkpError, match="not admitted"):
        PrivateProgramAnalysisWitness({"source_text": "x"}, policy=policy)
    with pytest.raises(ProgramAnalysisZkpError, match="max_opening_fields"):
        PrivateProgramAnalysisWitness(
            {"a": 1, "b": 2, "c": 3},
            policy=policy,
        )
    assert ProgramZkpWitnessPolicy.from_dict(policy.to_dict()) == policy


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_contract_version_is_pinned_in_payloads() -> None:
    inputs = _public_inputs()
    trace = build_canonical_program_zkp_trace(inputs)
    statement = prepare_program_analysis_zkp(inputs, witness=_witness()).statement
    envelope = _envelope()
    receipt = _receipt()

    for payload in (
        inputs.to_dict(),
        trace.to_dict(),
        statement.to_dict(),
        envelope.to_dict(),
        receipt.to_dict(),
    ):
        assert payload["contract_version"] == 1
        assert payload["schema"].endswith("@1")


def test_version_fields_are_bound_in_public_inputs() -> None:
    inputs = _public_inputs(
        analyzer_version="analyzer@9",
        resolver_version="resolver@9",
        translator_version="translator@9",
        prover_version="prover@9",
    )
    vector = inputs.public_input_vector
    assert "analyzer@9" in vector
    assert "resolver@9" in vector
    assert "translator@9" in vector
    assert "prover@9" in vector
    # Changing any toolchain version changes the digest.
    other = inputs.with_overrides(analyzer_version="analyzer@10")
    assert other.public_input_digest != inputs.public_input_digest


def test_receipt_rejects_codec_version_mismatch_on_construction() -> None:
    envelope = _envelope()
    with pytest.raises(ProgramZkpVersionError, match="codec version"):
        type(_receipt())(
            statement=envelope.statement,
            verdict=ProgramZkpVerdict.VERIFIED,
            verifier_id="verifier:x",
            verifying_key_id=envelope.statement.public_inputs.verifying_key_id,
            circuit_id=envelope.statement.public_inputs.circuit_id,
            public_input_digest=envelope.statement.public_input_digest,
            ceremony_id=envelope.statement.public_inputs.ceremony_id,
            public_input_codec_version="999",
            backend_mode=ProgramZkpBackendMode.SHADOW,
        )


# ---------------------------------------------------------------------------
# Illegal claim promotion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    (
        ClaimLevel.MODEL_PROVED,
        ClaimLevel.MODEL_DISPROVED,
        ClaimLevel.RUNTIME_WITNESSED,
        ClaimLevel.OBSERVED_SYNTAX,
        ClaimLevel.RESOLVED_STATIC,
    ),
)
def test_zk_trace_cannot_be_promoted_to_semantic_claims(target: ClaimLevel) -> None:
    with pytest.raises(ProgramZkpClaimPromotionError):
        reject_illegal_zk_claim_promotion(ClaimLevel.ZK_TRACE_ATTESTED, target)
    # Same-level is allowed.
    reject_illegal_zk_claim_promotion(
        ClaimLevel.ZK_TRACE_ATTESTED, ClaimLevel.ZK_TRACE_ATTESTED
    )


@pytest.mark.parametrize(
    "source",
    (
        ClaimLevel.OBSERVED_SYNTAX,
        ClaimLevel.RESOLVED_STATIC,
        ClaimLevel.MODEL_PROVED,
        ClaimLevel.RUNTIME_WITNESSED,
    ),
)
def test_other_claims_cannot_be_promoted_to_zk_trace(source: ClaimLevel) -> None:
    with pytest.raises(ProgramZkpClaimPromotionError):
        reject_illegal_zk_claim_promotion(source, ClaimLevel.ZK_TRACE_ATTESTED)


def test_statement_rejects_non_zk_claim_level() -> None:
    inputs = _public_inputs()
    trace = build_canonical_program_zkp_trace(inputs)
    with pytest.raises(ProgramZkpClaimPromotionError, match="zk_trace_attested"):
        ProgramZkpStatement(
            public_inputs=inputs,
            trace_id=trace.trace_id,
            claim_level=ClaimLevel.MODEL_PROVED,
        )


def test_general_claim_promotion_still_rejects_cross_level() -> None:
    with pytest.raises((ProgramZkpClaimPromotionError, ClaimPromotionError)):
        reject_illegal_zk_claim_promotion(
            ClaimLevel.OBSERVED_SYNTAX, ClaimLevel.MODEL_PROVED
        )


def test_prepare_round_trip_binds_trace_and_public_inputs() -> None:
    inputs = _public_inputs()
    request = prepare_program_analysis_zkp(inputs, witness=_witness())
    assert request.statement.public_input_digest == inputs.public_input_digest
    assert request.trace.trace_id == request.statement.trace_id
    assert request.trace.result_commitment == inputs.result_commitment
    envelope = create_program_zkp_shadow_envelope(
        request,
        proof_artifact_id="artifact:p",
        proof_digest="sha256:deadbeef",
    )
    receipt = record_program_zkp_verification(
        envelope,
        verdict=ProgramZkpVerdict.VERIFIED,
        verifier_id="verifier:local",
    )
    assert receipt.public_input_digest == inputs.public_input_digest
    assert_trace_non_claims(receipt)
