"""IPS-031: bounded hermetic prover and verifier adapters."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.provers import (
    DEFAULT_MAX_OUTPUT_BYTES,
    KNOWN_PROVER_BACKEND_IDS,
    PROVER_ADAPTER_EVIDENCE,
    CancellationToken,
    ExternalEngineResult,
    HermeticHmacEngine,
    IncrementalProofBackendAdapter,
    ProgramRegistry,
    ProverError,
    ProverInvocation,
    ProverOutcome,
    ProverReasonCode,
    ProverStatus,
    RegisteredProgram,
    VerificationInvocation,
    assert_no_sensitive_material,
    closed_known_prover_backend_ids,
    closed_prover_reason_codes,
    closed_prover_statuses,
    default_hermetic_program_registry,
    prove,
    public_input_cid_of,
    scrub_sensitive_mapping,
    verify,
    witness_safe_log_line,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.trust import (
    ProvingKeyHandle,
    ProvingKeyRecord,
    SetupOrigin,
    TrustedProofPolicy,
    VerificationKeyRecord,
    build_production_policy,
)

_PROGRAM_ID = "program:ips-hermetic-hmac@1"
_CIRCUIT_ID = "circuit:ips-hermetic-hmac@1"
_BACKEND = "hermetic_hmac"
_VK_ID = "vk/hermetic-1"
_PK_ID = "pk/hermetic-1"
_VK_CID = "bafybeigverificationkeyhermetic0000000000000001"
_PK_CID = "bafybeigprovingkeyhermetic000000000000000000001"
_PUBLIC = b"ips-prover-public-input-v1\ncommitted\n"
_WITNESS = b"ips-prover-witness-secret-v1\nnever-export\n"


def _handle(**overrides) -> ProvingKeyHandle:
    payload = {
        "key_id": _PK_ID,
        "key_cid": _PK_CID,
        "circuit_ids": frozenset({_CIRCUIT_ID}),
        "setup_origin": SetupOrigin.OPERATOR_REVIEWED,
        "test_only": False,
        "paired_verification_key_id": _VK_ID,
        "epoch": 1,
    }
    payload.update(overrides)
    return ProvingKeyHandle(**payload)


def _prove_invocation(**overrides) -> ProverInvocation:
    payload = {
        "program_id": _PROGRAM_ID,
        "circuit_id": _CIRCUIT_ID,
        "public_input": _PUBLIC,
        "witness": _WITNESS,
        "proving_key_handle": _handle(),
        "verification_key_id": _VK_ID,
        "verification_key_cid": _VK_CID,
        "backend_id": _BACKEND,
        "proof_unit_id": "unit/hermetic-1",
        "production": True,
    }
    payload.update(overrides)
    return ProverInvocation(**payload)


def _verify_invocation(
    proof_bytes: bytes,
    *,
    public_input: bytes = _PUBLIC,
    **overrides,
) -> VerificationInvocation:
    payload = {
        "program_id": _PROGRAM_ID,
        "circuit_id": _CIRCUIT_ID,
        "public_input": public_input,
        "proof_bytes": proof_bytes,
        "verification_key_id": _VK_ID,
        "verification_key_cid": _VK_CID,
        "backend_id": _BACKEND,
        "proof_unit_id": "unit/hermetic-1",
        "production": True,
        "metadata": {"proving_key_cid": _PK_CID},
    }
    payload.update(overrides)
    return VerificationInvocation(**payload)


def _adapter(**kwargs) -> IncrementalProofBackendAdapter:
    return IncrementalProofBackendAdapter(**kwargs)


def _policy() -> TrustedProofPolicy:
    return build_production_policy(
        verification_keys=(
            VerificationKeyRecord(
                key_id=_VK_ID,
                key_cid=_VK_CID,
                circuit_ids=frozenset({_CIRCUIT_ID}),
                setup_origin=SetupOrigin.OPERATOR_REVIEWED,
                test_only=False,
                epoch=1,
            ),
        ),
        proving_keys=(
            ProvingKeyRecord(
                key_id=_PK_ID,
                key_cid=_PK_CID,
                circuit_ids=frozenset({_CIRCUIT_ID}),
                setup_origin=SetupOrigin.OPERATOR_REVIEWED,
                test_only=False,
                paired_verification_key_id=_VK_ID,
                epoch=1,
            ),
        ),
        signers=(),
        current_epoch=1,
        minimum_key_epoch=1,
    )


class _AmbiguousEngine:
    """External engine that completes without durable proof bytes."""

    def prove(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(
            completed=False,
            proof_bytes=None,
            verified=None,
            durable_artifact_present=False,
            error_message="engine exit status unclear",
        )

    def verify(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(
            completed=True,
            proof_bytes=invocation.proof_bytes,
            verified=None,
            durable_artifact_present=True,
            error_message="verify verdict missing",
        )


class _TimeoutEngine:
    def prove(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(completed=False, timed_out=True)

    def verify(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(completed=False, timed_out=True)


class _CancelEngine:
    def prove(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(completed=False, cancelled=True)

    def verify(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(completed=False, cancelled=True)


class _UnavailableEngine:
    def prove(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(completed=False, unavailable=True)

    def verify(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(completed=False, unavailable=True)


class _OversizedEngine:
    def prove(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(
            completed=True,
            proof_bytes=b"\x00" * (DEFAULT_MAX_OUTPUT_BYTES + 1),
            durable_artifact_present=True,
        )

    def verify(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(
            completed=True,
            proof_bytes=invocation.proof_bytes,
            verified=False,
        )


class _CompletedWithoutBytesEngine:
    def prove(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(
            completed=True,
            proof_bytes=None,
            durable_artifact_present=False,
        )

    def verify(self, program, invocation) -> ExternalEngineResult:
        return ExternalEngineResult(
            completed=True,
            proof_bytes=None,
            verified=True,
        )


# ---------------------------------------------------------------------------
# Closed vocabularies / evidence
# ---------------------------------------------------------------------------


def test_evidence_subset_and_closed_vocabularies() -> None:
    assert PROVER_ADAPTER_EVIDENCE == "ips/prover-adapters@1"
    assert closed_known_prover_backend_ids() == KNOWN_PROVER_BACKEND_IDS
    statuses = closed_prover_statuses()
    for required in (
        "proved",
        "failed",
        "timeout",
        "unavailable",
        "cancelled",
        "ambiguous",
        "verification_failed",
        "invalid",
        "simulated",
    ):
        assert required in statuses
    reasons = closed_prover_reason_codes()
    for required in (
        "proved",
        "public_input_mismatch",
        "invalid_cryptography",
        "ambiguous_external_completion",
        "arbitrary_executable_rejected",
        "output_bound_exceeded",
        "timeout",
        "cancelled",
        "unavailable",
        "unknown_program",
    ):
        assert required in reasons


# ---------------------------------------------------------------------------
# Happy path: prove + verify round trip
# ---------------------------------------------------------------------------


def test_hermetic_prove_and_verify_round_trip() -> None:
    adapter = _adapter()
    prove_outcome = adapter.prove(_prove_invocation())
    assert prove_outcome.status is ProverStatus.PROVED
    assert prove_outcome.proved is True
    assert prove_outcome.verified is True
    assert prove_outcome.success is True
    assert prove_outcome.ambiguous is False
    assert prove_outcome.proof_bytes is not None
    assert prove_outcome.proof_cid is not None
    assert prove_outcome.reason_code == ProverReasonCode.PROVED.value
    assert prove_outcome.bounded is True

    verify_outcome = adapter.verify(
        _verify_invocation(
            prove_outcome.proof_bytes,
            metadata={"proving_key_cid": _PK_CID},
        )
    )
    assert verify_outcome.status is ProverStatus.PROVED
    assert verify_outcome.proved is True
    assert verify_outcome.verified is True
    assert verify_outcome.reason_code == ProverReasonCode.VERIFIED.value


def test_module_level_prove_verify_helpers() -> None:
    outcome = prove(_prove_invocation())
    assert outcome.proved is True
    recheck = verify(
        _verify_invocation(outcome.proof_bytes, metadata={"proving_key_cid": _PK_CID})
    )
    assert recheck.proved is True


# ---------------------------------------------------------------------------
# Modified public input / invalid cryptography fail
# ---------------------------------------------------------------------------


def test_modified_public_input_fails_verification() -> None:
    adapter = _adapter()
    proved = adapter.prove(_prove_invocation())
    assert proved.proved is True

    tampered = adapter.verify(
        _verify_invocation(
            proved.proof_bytes,
            public_input=b"ips-prover-public-input-v1\nTAMPERED\n",
            metadata={"proving_key_cid": _PK_CID},
        )
    )
    assert tampered.proved is False
    assert tampered.verified is False
    assert tampered.status is ProverStatus.VERIFICATION_FAILED
    assert tampered.reason_code == ProverReasonCode.INVALID_CRYPTOGRAPHY.value


def test_invalid_proof_bytes_fail_verification() -> None:
    adapter = _adapter()
    garbage = adapter.verify(
        _verify_invocation(
            b"\x00" * 64,
            metadata={"proving_key_cid": _PK_CID},
        )
    )
    assert garbage.proved is False
    assert garbage.status is ProverStatus.VERIFICATION_FAILED
    assert garbage.reason_code == ProverReasonCode.INVALID_CRYPTOGRAPHY.value

    truncated = adapter.verify(
        _verify_invocation(
            b"\x11" * 16,
            metadata={"proving_key_cid": _PK_CID},
        )
    )
    assert truncated.proved is False
    assert truncated.status is ProverStatus.VERIFICATION_FAILED


def test_public_input_cid_mismatch_rejected_at_construction() -> None:
    with pytest.raises(ProverError, match="public_input_cid"):
        _prove_invocation(public_input_cid="sha256:" + ("00" * 32))


# ---------------------------------------------------------------------------
# Witness / proving-key secrecy
# ---------------------------------------------------------------------------


def test_witness_and_proving_key_absent_from_receipts_and_logs() -> None:
    adapter = _adapter()
    outcome = adapter.prove(_prove_invocation())
    assert outcome.proved is True

    receipt = outcome.to_receipt()
    canonical = outcome.to_canonical()
    canonical_json = outcome.to_canonical_json()
    invocation = _prove_invocation().to_canonical()

    for surface in (receipt, canonical, invocation):
        assert surface["witness_exported"] is False
        assert surface["proving_key_exported"] is False
        assert "witness" not in surface
        assert "witness_bytes" not in surface
        assert "proving_key_bytes" not in surface
        assert "private_key" not in surface
        assert_no_sensitive_material(surface)

    assert "never-export" not in canonical_json
    assert _WITNESS.hex() not in canonical_json
    assert "witness=" not in canonical_json

    for line in outcome.log_lines:
        assert "never-export" not in line
        assert "witness=" not in line.casefold()
        assert "proving_key_bytes" not in line.casefold()
        assert_no_sensitive_material(line)


def test_scrub_sensitive_mapping_and_log_helper() -> None:
    scrubbed = scrub_sensitive_mapping(
        {
            "public_input_cid": "sha256:abc",
            "witness": _WITNESS,
            "proving_key_bytes": b"secret",
            "nested": {"trapdoor": "x", "ok": 1},
        }
    )
    assert scrubbed["witness"] == "<redacted>"
    assert scrubbed["proving_key_bytes"] == "<redacted>"
    assert scrubbed["nested"]["trapdoor"] == "<redacted>"
    assert scrubbed["nested"]["ok"] == 1

    line = witness_safe_log_line(
        "prove.start",
        program_id=_PROGRAM_ID,
        public_input_cid=public_input_cid_of(_PUBLIC),
    )
    assert "prove.start" in line
    assert "witness" not in line

    with pytest.raises(ProverError):
        witness_safe_log_line("bad witness=secret")


def test_proving_key_handle_never_exports_bytes() -> None:
    handle = _handle()
    with pytest.raises(Exception, match="nonexportable|forbidden"):
        handle.export_bytes()
    public = handle.to_public_api()
    assert public["exportable"] is False
    assert public["proving_key_exported"] is False


# ---------------------------------------------------------------------------
# Ambiguous external completion never reported as proved
# ---------------------------------------------------------------------------


def test_ambiguous_external_completion_never_proved() -> None:
    registry = default_hermetic_program_registry()
    # Rebind hermetic program to a fake backend id is not allowed for unknown;
    # instead inject ambiguous engine under hermetic_hmac.
    adapter = IncrementalProofBackendAdapter(
        programs=registry,
        engines={"hermetic_hmac": _AmbiguousEngine()},
    )
    outcome = adapter.prove(_prove_invocation())
    assert outcome.proved is False
    assert outcome.success is False
    assert outcome.ambiguous is True
    assert outcome.status is ProverStatus.AMBIGUOUS
    assert (
        outcome.reason_code == ProverReasonCode.AMBIGUOUS_EXTERNAL_COMPLETION.value
    )
    assert "not reporting proved" in outcome.message or "ambiguous" in outcome.message.casefold()

    # Verify path with missing boolean verdict is also ambiguous, never proved.
    engine = HermeticHmacEngine()
    good = engine.prove(
        registry.require(_PROGRAM_ID),
        _prove_invocation(),
    )
    assert good.proof_bytes is not None
    verify_outcome = adapter.verify(
        _verify_invocation(good.proof_bytes, metadata={"proving_key_cid": _PK_CID})
    )
    assert verify_outcome.proved is False
    assert verify_outcome.ambiguous is True
    assert verify_outcome.status is ProverStatus.AMBIGUOUS


def test_completed_without_proof_bytes_is_ambiguous() -> None:
    adapter = IncrementalProofBackendAdapter(
        engines={"hermetic_hmac": _CompletedWithoutBytesEngine()},
    )
    outcome = adapter.prove(_prove_invocation())
    assert outcome.proved is False
    assert outcome.ambiguous is True
    assert outcome.status is ProverStatus.AMBIGUOUS


def test_outcome_rejects_proved_true_with_ambiguous_status() -> None:
    with pytest.raises(ProverError, match="never set proved"):
        ProverOutcome(
            schema="test",
            status=ProverStatus.AMBIGUOUS,
            proved=True,
            reason_code="ambiguous_external_completion",
            message="bad",
            backend_id=_BACKEND,
            program_id=_PROGRAM_ID,
            circuit_id=_CIRCUIT_ID,
            public_input_cid=public_input_cid_of(_PUBLIC),
            proof_unit_id="unit/x",
            verified=True,
            ambiguous=True,
        )


# ---------------------------------------------------------------------------
# Timeout / cancel / unavailable
# ---------------------------------------------------------------------------


def test_timeout_outcome() -> None:
    adapter = IncrementalProofBackendAdapter(
        engines={"hermetic_hmac": _TimeoutEngine()},
    )
    outcome = adapter.prove(_prove_invocation())
    assert outcome.proved is False
    assert outcome.status is ProverStatus.TIMEOUT
    assert outcome.reason_code == ProverReasonCode.TIMEOUT.value


def test_cancelled_before_and_by_engine() -> None:
    token = CancellationToken()
    token.cancel()
    adapter = _adapter()
    early = adapter.prove(_prove_invocation(cancellation=token))
    assert early.proved is False
    assert early.status is ProverStatus.CANCELLED

    adapter2 = IncrementalProofBackendAdapter(
        engines={"hermetic_hmac": _CancelEngine()},
    )
    late = adapter2.prove(_prove_invocation())
    assert late.proved is False
    assert late.status is ProverStatus.CANCELLED


def test_unavailable_backend() -> None:
    adapter = IncrementalProofBackendAdapter(
        available_backends={"hermetic_hmac": False},
    )
    outcome = adapter.prove(_prove_invocation())
    assert outcome.proved is False
    assert outcome.status is ProverStatus.UNAVAILABLE
    assert outcome.reason_code == ProverReasonCode.UNAVAILABLE.value

    adapter2 = IncrementalProofBackendAdapter(
        engines={"hermetic_hmac": _UnavailableEngine()},
    )
    outcome2 = adapter2.prove(_prove_invocation())
    assert outcome2.status is ProverStatus.UNAVAILABLE
    assert outcome2.proved is False


# ---------------------------------------------------------------------------
# Static registry: no arbitrary executable/path; unknown program/backend
# ---------------------------------------------------------------------------


def test_unknown_program_rejected() -> None:
    adapter = _adapter()
    outcome = adapter.prove(_prove_invocation(program_id="program:not-registered@1"))
    assert outcome.proved is False
    assert outcome.status is ProverStatus.INVALID
    assert outcome.reason_code == ProverReasonCode.UNKNOWN_PROGRAM.value


def test_unknown_backend_rejected() -> None:
    adapter = _adapter()
    outcome = adapter.prove(_prove_invocation(backend_id="exotic-snark"))
    assert outcome.proved is False
    assert outcome.status is ProverStatus.UNKNOWN
    assert outcome.reason_code == ProverReasonCode.UNKNOWN_BACKEND.value


def test_circuit_mismatch_rejected() -> None:
    adapter = _adapter()
    outcome = adapter.prove(_prove_invocation(circuit_id="circuit:other@1"))
    assert outcome.proved is False
    assert outcome.reason_code == ProverReasonCode.UNREGISTERED_CIRCUIT.value


def test_caller_supplied_executable_rejected() -> None:
    adapter = _adapter()
    outcome = adapter.prove(
        _prove_invocation(metadata={"executable_path": "/tmp/evil-prover"})
    )
    assert outcome.proved is False
    assert outcome.reason_code == ProverReasonCode.ARBITRARY_EXECUTABLE_REJECTED.value

    outcome2 = adapter.prove(
        _prove_invocation(metadata={"network_url": "https://example.invalid/keys"})
    )
    assert outcome2.proved is False
    assert outcome2.reason_code == ProverReasonCode.NETWORK_FORBIDDEN.value

    outcome3 = adapter.prove(
        _prove_invocation(metadata={"setup_generate": True})
    )
    assert outcome3.proved is False
    assert outcome3.reason_code == ProverReasonCode.SETUP_GENERATION_FORBIDDEN.value


def test_registered_program_rejects_path_argv() -> None:
    with pytest.raises(ProverError, match="path or shell"):
        RegisteredProgram(
            program_id="program:bad@1",
            circuit_id="circuit:bad@1",
            backend_id="hermetic_hmac",
            argv=("/usr/bin/evil",),
        )


def test_output_bound_exceeded() -> None:
    adapter = IncrementalProofBackendAdapter(
        engines={"hermetic_hmac": _OversizedEngine()},
    )
    outcome = adapter.prove(_prove_invocation(max_output_bytes=128))
    assert outcome.proved is False
    assert outcome.reason_code == ProverReasonCode.OUTPUT_BOUND_EXCEEDED.value


# ---------------------------------------------------------------------------
# Trust policy integration
# ---------------------------------------------------------------------------


def test_unallowlisted_verification_key_rejected_with_policy() -> None:
    adapter = IncrementalProofBackendAdapter(policy=_policy())
    outcome = adapter.prove(
        _prove_invocation(verification_key_id="vk/unknown", verification_key_cid=_VK_CID)
    )
    assert outcome.proved is False
    assert outcome.reason_code == ProverReasonCode.KEY_TRUST_REJECTED.value


def test_allowlisted_keys_accepted_with_policy() -> None:
    adapter = IncrementalProofBackendAdapter(policy=_policy())
    outcome = adapter.prove(_prove_invocation())
    assert outcome.proved is True
    assert outcome.details["handle_only"] is True


def test_simulated_backend_forbidden_in_production() -> None:
    registry = ProgramRegistry(
        (
            RegisteredProgram(
                program_id="program:sim@1",
                circuit_id="circuit:sim@1",
                backend_id="simulated",
                production_allowed=True,
            ),
        )
    )
    adapter = IncrementalProofBackendAdapter(programs=registry)
    outcome = adapter.prove(
        _prove_invocation(
            program_id="program:sim@1",
            circuit_id="circuit:sim@1",
            backend_id="simulated",
            production=True,
        )
    )
    assert outcome.proved is False
    assert outcome.status is ProverStatus.SIMULATED
    assert outcome.reason_code == ProverReasonCode.SIMULATED_FORBIDDEN.value


# ---------------------------------------------------------------------------
# Registry / engine defaults
# ---------------------------------------------------------------------------


def test_default_registry_contains_hermetic_program() -> None:
    registry = default_hermetic_program_registry()
    assert _PROGRAM_ID in registry
    program = registry.require(_PROGRAM_ID)
    assert program.backend_id == _BACKEND
    assert program.circuit_id == _CIRCUIT_ID
    payload = registry.to_canonical()
    assert payload["evidence_subset"] == PROVER_ADAPTER_EVIDENCE
    assert "witness" not in payload["programs"][_PROGRAM_ID]


def test_invocation_canonical_omits_witness_bytes() -> None:
    inv = _prove_invocation()
    payload = inv.to_canonical()
    assert payload["witness_present"] is True
    assert payload["witness_exported"] is False
    assert payload["witness_byte_length"] == len(_WITNESS)
    assert "witness" not in payload
    assert payload["public_input_cid"] == public_input_cid_of(_PUBLIC)
