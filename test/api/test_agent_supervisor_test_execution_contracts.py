"""PTR-001: test-execution proof contract schemas and decision boundary."""

from __future__ import annotations

import json
import math
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    MAX_DIAGNOSTIC_KEYS,
    MAX_SEQUENCE_ITEMS,
    MAX_TEXT_CHARS,
    CertificateAuthority,
    EligibilityClass,
    PhaseOutcome,
    ProofBackendMode,
    REUSE_DECISION_INTERFACE,
    ReuseAction,
    ReuseDecision,
    ReuseReasonCode,
    TEST_EXECUTION_CONTRACT_VERSION,
    TEST_EXECUTION_KEY_INTERFACE,
    TEST_LOCATOR_KEY_INTERFACE,
    TEST_PASS_RECEIPT_INTERFACE,
    TEST_PROOF_CERTIFICATE_INTERFACE,
    TestExecutionContractError,
    TestExecutionKey,
    TestLocatorKey,
    TestPassReceipt,
    TestProofCertificate,
    certificate_may_skip,
    coerce_lookup_result,
    decision_from_absence,
    decision_from_exception,
    reuse_run,
    reuse_skip,
    skip_reason_for_certificate,
)


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _locator(**changes: object) -> TestLocatorKey:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:demo",
        "package_identity": "ipfs_accelerate_py",
        "node_id": "test/api/test_demo.py::test_alpha",
        "collection_schema_version": "1",
        "root_identity": "root:repo",
        "selection_semantics": "exact_node",
    }
    values.update(changes)
    return TestLocatorKey(**values)  # type: ignore[arg-type]


def _execution_key(locator: TestLocatorKey | None = None, **changes: object) -> TestExecutionKey:
    loc = locator or _locator()
    values: dict[str, object] = {
        "locator_cid": loc.locator_id,
        "repository_forest_cid": "cid:forest:v1",
        "git_commit_id": "deadbeef",
        "git_tree_id": "tree:abc",
        "test_module_cid": "cid:module",
        "test_function_cid": "cid:function",
        "test_ast_cid": "cid:ast",
        "static_trace_root_cid": "cid:static",
        "runtime_trace_root_cid": "cid:runtime",
        "pytest_version": "8.2.0",
        "python_version": "3.11.9",
        "policy_cid": "cid:policy",
        "eligibility_class": EligibilityClass.REPOSITORY_FOREST_BOUND,
        "markers": ("slow", "unit"),
        "fixture_cids": ("cid:fix-b", "cid:fix-a"),
    }
    values.update(changes)
    return TestExecutionKey(**values)  # type: ignore[arg-type]


def _receipt(
    key: TestExecutionKey | None = None,
    locator: TestLocatorKey | None = None,
    **changes: object,
) -> TestPassReceipt:
    loc = locator or _locator()
    ek = key or _execution_key(loc)
    values: dict[str, object] = {
        "execution_key_cid": ek.execution_key_id,
        "locator_cid": loc.locator_id,
        "setup_outcome": PhaseOutcome.PASS,
        "call_outcome": PhaseOutcome.PASS,
        "teardown_outcome": PhaseOutcome.PASS,
        "setup_duration_ms": 1,
        "call_duration_ms": 2,
        "teardown_duration_ms": 1,
        "runner_identity": "runner:pytest@8.2",
        "trust_domain": "trust:local",
        "issuer_key_id": "key:issuer-1",
        "nonce": "nonce-1",
        "policy_cid": "cid:policy",
        "admitted": True,
    }
    values.update(changes)
    return TestPassReceipt(**values)  # type: ignore[arg-type]


def _certificate(
    receipt: TestPassReceipt | None = None,
    key: TestExecutionKey | None = None,
    **changes: object,
) -> TestProofCertificate:
    r = receipt or _receipt()
    values: dict[str, object] = {
        "receipt_cid": r.receipt_id,
        "execution_key_cid": r.execution_key_cid if key is None else key.execution_key_id,
        "statement_cid": "cid:statement:TestPassStatementV1",
        "circuit_cid": "cid:circuit:v1",
        "verifying_key_cid": "cid:vk:v1",
        "proof_system_id": "groth16",
        "proof_artifact_cid": "cid:proof",
        "proof_digest": "sha256:proof",
        "backend_mode": ProofBackendMode.CRYPTOGRAPHIC,
        "authority": CertificateAuthority.AUTHORITATIVE,
        "issuer_id": "issuer:local",
        "policy_cid": "cid:policy",
        "epoch": "epoch:1",
        "public_inputs": {
            "receipt_cid": r.receipt_id,
            "execution_key_cid": r.execution_key_cid,
        },
    }
    values.update(changes)
    return TestProofCertificate(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy-path round-trips and deterministic serialization
# ---------------------------------------------------------------------------


def test_locator_key_round_trip_and_identity() -> None:
    first = _locator(metadata={"b": 1, "a": 2})
    second = _locator(metadata={"a": 2, "b": 1})

    assert first.interface == TEST_LOCATOR_KEY_INTERFACE
    assert first.to_json() == second.to_json()
    assert first.locator_id == second.locator_id
    assert first.locator_id.startswith("b")
    assert TestLocatorKey.from_dict(first.to_dict()) == first
    assert json.loads(first.to_json())["interface"] == TEST_LOCATOR_KEY_INTERFACE
    assert json.loads(first.to_json())["contract_version"] == TEST_EXECUTION_CONTRACT_VERSION


def test_execution_key_deterministic_fixture_and_marker_order() -> None:
    a = _execution_key(fixture_cids=("cid:fix-b", "cid:fix-a"), markers=("z", "a"))
    b = _execution_key(fixture_cids=("cid:fix-a", "cid:fix-b"), markers=("a", "z"))

    assert a.to_json() == b.to_json()
    assert a.execution_key_id == b.execution_key_id
    assert list(json.loads(a.to_json())["fixture_cids"]) == ["cid:fix-a", "cid:fix-b"]
    assert TestExecutionKey.from_dict(a.to_dict()) == a
    assert a.interface == TEST_EXECUTION_KEY_INTERFACE


def test_pass_receipt_and_certificate_round_trip() -> None:
    receipt = _receipt()
    certificate = _certificate(receipt)

    assert receipt.interface == TEST_PASS_RECEIPT_INTERFACE
    assert certificate.interface == TEST_PROOF_CERTIFICATE_INTERFACE
    assert receipt.all_phases_pass is True
    assert TestPassReceipt.from_dict(receipt.to_dict()) == receipt
    assert TestProofCertificate.from_dict(certificate.to_dict()) == certificate
    assert certificate.can_authorize_skip is True
    assert certificate_may_skip(certificate) is True
    assert skip_reason_for_certificate(certificate).startswith("proof-cache-hit:")


def test_reuse_decision_run_and_skip_explicit() -> None:
    cert = _certificate()
    run = reuse_run(ReuseReasonCode.CANDIDATE_MISSING, diagnostics={"k": "v"})
    skip = reuse_skip(
        certificate_cid=cert.certificate_id,
        receipt_cid=cert.receipt_cid,
        validation_receipt_cid="cid:validation",
    )

    assert run.action is ReuseAction.RUN
    assert run.is_run is True
    assert run.is_skip is False
    assert skip.action is ReuseAction.SKIP
    assert skip.is_skip is True
    assert skip.authority is CertificateAuthority.AUTHORITATIVE
    assert ReuseDecision.from_dict(run.to_dict()) == run
    assert ReuseDecision.from_dict(skip.to_dict()) == skip
    assert run.interface == REUSE_DECISION_INTERFACE


def test_serialization_is_byte_stable_across_key_insertion_order() -> None:
    key = _execution_key(
        components={"z": "cid:z", "a": "cid:a"},
        metadata={"m2": True, "m1": False},
    )
    again = TestExecutionKey.from_dict(json.loads(key.to_json()))
    assert key.canonical_bytes() == again.canonical_bytes()
    assert key.content_id == again.content_id


# ---------------------------------------------------------------------------
# Rejection: nonfinite, unbounded, private, malformed, versionless, illegal-authority
# ---------------------------------------------------------------------------


def test_rejects_nonfinite_floats_in_metadata() -> None:
    with pytest.raises(TestExecutionContractError, match="float|nonfinite|canonical"):
        _locator(metadata={"score": float("nan")})
    with pytest.raises(TestExecutionContractError, match="float|nonfinite|canonical"):
        _locator(metadata={"score": float("inf")})
    with pytest.raises(TestExecutionContractError, match="float|nonfinite|canonical"):
        _locator(metadata={"score": math.pi})


def test_rejects_unbounded_sequences_and_text() -> None:
    with pytest.raises(TestExecutionContractError, match="bounded"):
        _execution_key(fixture_cids=tuple(f"cid:{i}" for i in range(MAX_SEQUENCE_ITEMS + 1)))
    with pytest.raises(TestExecutionContractError, match="bounded"):
        _locator(node_id="x" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(TestExecutionContractError, match="bounded"):
        reuse_run(
            ReuseReasonCode.UNKNOWN,
            diagnostics={f"k{i}": "v" for i in range(MAX_DIAGNOSTIC_KEYS + 1)},
        )


def test_rejects_private_material_in_public_payloads() -> None:
    with pytest.raises(TestExecutionContractError, match="private"):
        _locator(metadata={"api_key": "synthetic-secret"})
    with pytest.raises(TestExecutionContractError, match="private"):
        _certificate(public_inputs={"private_witness": "w"})
    with pytest.raises(TestExecutionContractError, match="private"):
        reuse_run(ReuseReasonCode.UNKNOWN, diagnostics={"password": "x"})


def test_rejects_malformed_types_and_unknown_fields() -> None:
    with pytest.raises(TestExecutionContractError):
        TestLocatorKey(
            repository_id="repo",
            package_identity="pkg",
            node_id="n",
            metadata=["not", "a", "mapping"],  # type: ignore[arg-type]
        )
    payload = _locator().to_dict()
    payload["extra_hostile_field"] = "x"
    with pytest.raises(TestExecutionContractError, match="unsupported fields"):
        TestLocatorKey.from_dict(payload)
    with pytest.raises(TestExecutionContractError):
        ReuseDecision(
            action="MAYBE",  # type: ignore[arg-type]
            reason_code=ReuseReasonCode.UNKNOWN,
        )


def test_rejects_versionless_and_wrong_interface_payloads() -> None:
    base = _locator().to_dict()
    versionless = {
        k: v
        for k, v in base.items()
        if k not in {"interface", "contract_version", "schema"}
    }
    # Keep schema so schema check does not fire first; still versionless.
    versionless["schema"] = base["schema"]
    with pytest.raises(TestExecutionContractError, match="versionless"):
        TestLocatorKey.from_dict(versionless)

    wrong = deepcopy(base)
    wrong["interface"] = "TestLocatorKey@0"
    with pytest.raises(TestExecutionContractError, match="interface"):
        TestLocatorKey.from_dict(wrong)

    wrong_version = deepcopy(base)
    wrong_version["contract_version"] = 99
    with pytest.raises(TestExecutionContractError, match="contract_version"):
        TestLocatorKey.from_dict(wrong_version)

    for cls, builder in (
        (TestExecutionKey, _execution_key),
        (TestPassReceipt, _receipt),
        (TestProofCertificate, _certificate),
        (ReuseDecision, lambda: reuse_run(ReuseReasonCode.MODE_OFF)),
    ):
        payload = builder().to_dict()
        stripped = {
            k: v
            for k, v in payload.items()
            if k not in {"interface", "contract_version"}
        }
        with pytest.raises(TestExecutionContractError, match="versionless"):
            cls.from_dict(stripped)


def test_rejects_illegal_authority_on_certificate_and_skip() -> None:
    with pytest.raises(TestExecutionContractError, match="illegal-authority"):
        _certificate(
            backend_mode=ProofBackendMode.SIMULATED,
            authority=CertificateAuthority.AUTHORITATIVE,
        )

    simulated = _certificate(
        backend_mode=ProofBackendMode.SIMULATED,
        authority=CertificateAuthority.NON_ATTESTED,
    )
    assert simulated.authority is CertificateAuthority.NON_ATTESTED
    assert simulated.can_authorize_skip is False
    assert certificate_may_skip(simulated) is False

    with pytest.raises(TestExecutionContractError, match="illegal-authority|authoritative"):
        reuse_skip(
            certificate_cid=simulated.certificate_id,
            receipt_cid=simulated.receipt_cid,
            authority=CertificateAuthority.NON_ATTESTED,
        )

    with pytest.raises(TestExecutionContractError, match="illegal-authority"):
        _certificate(authority=CertificateAuthority.UNKNOWN)


def test_admitted_receipt_rejects_disqualifying_outcomes() -> None:
    with pytest.raises(TestExecutionContractError, match="admitted"):
        _receipt(call_outcome=PhaseOutcome.FAIL, admitted=True)
    with pytest.raises(TestExecutionContractError, match="admitted"):
        _receipt(disqualifying_states=("xfail",), admitted=True)

    not_admitted = _receipt(
        call_outcome=PhaseOutcome.SKIP,
        admitted=False,
        disqualifying_states=("skip",),
    )
    assert not_admitted.admitted is False
    assert not_admitted.all_phases_pass is False


def test_forged_content_identity_is_rejected() -> None:
    loc = _locator()
    forged = loc.to_dict()
    forged["content_id"] = "cid:forged"
    with pytest.raises(TestExecutionContractError, match="content identity"):
        TestLocatorKey.from_dict(forged)


# ---------------------------------------------------------------------------
# Decision action: absence and exceptions cannot coerce to SKIP
# ---------------------------------------------------------------------------


def test_absence_and_exception_map_only_to_run() -> None:
    absent = decision_from_absence()
    assert absent.action is ReuseAction.RUN
    assert absent.reason_code is ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN

    absent2 = decision_from_absence(ReuseReasonCode.CACHE_UNAVAILABLE)
    assert absent2.action is ReuseAction.RUN
    assert absent2.reason_code is ReuseReasonCode.CACHE_UNAVAILABLE

    # Attacker cannot force skip reason through absence helper.
    forced = decision_from_absence(ReuseReasonCode.PROOF_CACHE_HIT)
    assert forced.action is ReuseAction.RUN
    assert forced.reason_code is ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN

    exc_decision = decision_from_exception(RuntimeError("boom"))
    assert exc_decision.action is ReuseAction.RUN
    assert exc_decision.reason_code is ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN
    assert "boom" not in json.dumps(exc_decision.diagnostics)
    assert exc_decision.diagnostics.get("exception_type") == "RuntimeError"

    forced_exc = decision_from_exception(
        ValueError("x"), reason_code=ReuseReasonCode.PROOF_CACHE_HIT
    )
    assert forced_exc.action is ReuseAction.RUN
    assert forced_exc.reason_code is ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN


def test_coerce_lookup_result_never_invents_skip() -> None:
    assert coerce_lookup_result(None).action is ReuseAction.RUN
    assert coerce_lookup_result(RuntimeError("x")).action is ReuseAction.RUN
    assert coerce_lookup_result(123).action is ReuseAction.RUN

    # Mapping without action cannot become SKIP.
    with pytest.raises(TestExecutionContractError):
        # Direct decode rejects absence of action.
        ReuseDecision.from_dict(
            {
                "schema": reuse_run().to_dict()["schema"],
                "interface": REUSE_DECISION_INTERFACE,
                "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
                "reason_code": ReuseReasonCode.PROOF_CACHE_HIT.value,
                "certificate_cid": "c",
                "receipt_cid": "r",
                "authority": CertificateAuthority.AUTHORITATIVE.value,
            }
        )

    coerced = coerce_lookup_result(
        {
            "schema": reuse_run().to_dict()["schema"],
            "interface": REUSE_DECISION_INTERFACE,
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            # deliberately omit action
            "reason_code": ReuseReasonCode.PROOF_CACHE_HIT.value,
            "certificate_cid": "c",
            "receipt_cid": "r",
            "authority": CertificateAuthority.AUTHORITATIVE.value,
        }
    )
    assert coerced.action is ReuseAction.RUN

    good_skip = reuse_skip(certificate_cid="c", receipt_cid="r")
    assert coerce_lookup_result(good_skip).action is ReuseAction.SKIP
    assert coerce_lookup_result(good_skip.to_dict()).action is ReuseAction.SKIP


def test_skip_requires_explicit_run_or_skip_action_and_binding_cids() -> None:
    with pytest.raises(TestExecutionContractError, match="explicitly RUN or SKIP|absence"):
        ReuseDecision.from_dict(
            {
                "schema": reuse_run().to_dict()["schema"],
                "interface": REUSE_DECISION_INTERFACE,
                "contract_version": 1,
                "action": "",
                "reason_code": "candidate_missing",
            }
        )
    with pytest.raises(TestExecutionContractError, match="certificate_cid"):
        ReuseDecision(
            action=ReuseAction.SKIP,
            reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
            certificate_cid="",
            receipt_cid="r",
            authority=CertificateAuthority.AUTHORITATIVE,
        )
    with pytest.raises(TestExecutionContractError, match="proof_cache_hit|reason"):
        ReuseDecision(
            action=ReuseAction.SKIP,
            reason_code=ReuseReasonCode.CANDIDATE_MISSING,
            certificate_cid="c",
            receipt_cid="r",
            authority=CertificateAuthority.AUTHORITATIVE,
        )
    # Factory remaps skip-only reasons to fail-open RUN (never raises into SKIP).
    remapped = reuse_run(ReuseReasonCode.PROOF_CACHE_HIT)
    assert remapped.action is ReuseAction.RUN
    assert remapped.reason_code is ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN

    with pytest.raises(TestExecutionContractError, match="proof_cache_hit|skip-only"):
        ReuseDecision(
            action=ReuseAction.RUN,
            reason_code=ReuseReasonCode.PROOF_CACHE_HIT,
        )


def test_parameterized_locator_requires_values_or_non_reusable_reason() -> None:
    with pytest.raises(TestExecutionContractError, match="parameter"):
        _locator(parameter_id="p0")
    ok = _locator(parameter_id="p0", parameter_values_cid="cid:params")
    assert ok.parameter_id == "p0"
    ok2 = _locator(parameter_id="p0", non_reusable_reason="unserializable")
    assert ok2.non_reusable_reason == "unserializable"


def test_degradation_reason_codes_are_run_only() -> None:
    for code in (
        ReuseReasonCode.PLUGIN_UNAVAILABLE,
        ReuseReasonCode.CACHE_UNAVAILABLE,
        ReuseReasonCode.CID_PROVIDER_UNAVAILABLE,
        ReuseReasonCode.CERTIFICATE_NON_ATTESTED,
        ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN,
        ReuseReasonCode.TIMEOUT,
        ReuseReasonCode.UNSUPPORTED,
    ):
        decision = reuse_run(code)
        assert decision.action is ReuseAction.RUN
        assert decision.reason_code is code
