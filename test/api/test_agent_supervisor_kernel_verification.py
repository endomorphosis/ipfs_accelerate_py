from __future__ import annotations

import copy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
    ContractValidationError,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.kernel_verification import (
    IndependentKernelVerifier,
    KernelFailureCode,
    KernelTarget,
    KernelVerificationBindings,
    KernelVerificationResult,
    KernelVerificationStatus,
    _upstream_content_digest,
    build_kernel_verified_receipt,
    kernel_unavailable_result,
    verify_kernel_reconstruction,
)


def _source_and_output(target: str) -> tuple[str, str, str, str]:
    if target == "lean":
        return (
            "theorem exact_statement : True := by trivial\n"
            "#print axioms exact_statement\n",
            "by trivial",
            json.dumps(
                {
                    "severity": "information",
                    "data": "'exact_statement' does not depend on any axioms",
                }
            ),
            "",
        )
    if target == "coq":
        return (
            "Theorem exact_statement : True. Proof. exact I. Qed.\n"
            "Print Assumptions exact_statement.\n",
            "exact I.",
            "Closed under the global context\n",
            "",
        )
    return (
        'theory Exact imports Main begin\n'
        'theorem exact_statement: "True" by simp\n'
        "end\n",
        "by simp",
        "Finished theory Exact\n",
        "",
    )


def _packet(target: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source, proof, stdout, stderr = _source_and_output(target)
    command = [f"/tools/{target}-1.0", "check", f"Reconstruction.{target}"]
    source_digest = _upstream_content_digest({"checked_source": source})
    output_digest = _upstream_content_digest({"stdout": stdout, "stderr": stderr})
    record = {
        "schema_version": "1.0.0",
        "reconstruction_id": f"reconstruction-{target}",
        "request_id": "request-1",
        "candidate_id": "candidate-1",
        "target_itp": target,
        "environment_lock_id": f"lock-{target}",
        "kernel_command": " ".join(command),
        "kernel_accepted": True,
        "kernel_output_digest": output_digest,
        "started_at": "2026-07-23T00:00:00+00:00",
        "finished_at": "2026-07-23T00:00:01+00:00",
        "failure_reason": None,
    }
    evidence = {
        "schema_version": "1.0.0",
        "reconstruction_id": f"reconstruction-{target}",
        "request_id": "request-1",
        "candidate_id": "candidate-1",
        "itp": target,
        "command": command,
        "checked_source": source,
        "checked_source_digest": source_digest,
        "reconstructed_proof_text": proof,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": 0,
        "timed_out": False,
        "wall_time_seconds": "0.01",
        "raw_output_digest": output_digest,
    }
    lock = {
        "schema_version": "1.0.0",
        "lock_id": f"lock-{target}",
        "itp": target,
        "itp_version": "1.0.0",
        "kernel_command_template": "{kernel} check {source_file}",
        "solver_versions": {},
        "executable_paths": {target: f"/tools/{target}-1.0"},
        "os_info": "test",
        "pinned_at": "2026-07-23T00:00:00+00:00",
    }
    return record, evidence, lock


def _bindings(target: str) -> KernelVerificationBindings:
    return KernelVerificationBindings(
        obligation_id="obligation-1",
        request_id="request-1",
        candidate_id="candidate-1",
        kernel_id=f"kernel:{target}@1.0.0",
        toolchain_id=f"lock-{target}",
        expected_statement="True",
    )


def _obligation() -> CodeProofObligation:
    return CodeProofObligation(
        repository_id="repository-1",
        repository_tree_id="tree-1",
        ast_scope_ids=("scope-1",),
        statement="True",
        template_id="truth",
        template_version="1",
        template_semantic_hash="sha256:truth",
    )


def _receipt(result) -> ProofReceipt:
    return build_kernel_verified_receipt(
        result,
        plan_id="plan-1",
        attempt_id="attempt-1",
        repository_id="repository-1",
        repository_tree_id="tree-1",
        ast_scope_ids=("scope-1",),
        translator_id="translator-1",
        solver_id="solver-1",
        policy_id="policy-1",
        resource_budget=ResourceBudget(
            wall_time_ms=1_000,
            cpu_time_ms=1_000,
            memory_bytes=64 * 1024 * 1024,
            max_processes=1,
            network_allowed=False,
        ),
        provider_id="provider:untrusted",
        provider_claimed_assurance=AssuranceLevel.ATTESTED,
    )


@pytest.mark.parametrize("target", ("lean", "coq", "isabelle"))
def test_maps_supported_upstream_reconstructions_without_weakening_trust(
    target: str,
) -> None:
    record, evidence, lock = _packet(target)
    result = verify_kernel_reconstruction(
        record,
        evidence,
        lock,
        bindings=_bindings(target),
        provider_status="provider says verified",
        independent=True,
    )

    assert result.status is KernelVerificationStatus.ACCEPTED
    assert result.target.value == target
    assert result.verdict is ProofVerdict.PROVED
    assert result.assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.evidence.metadata["bindings_verified"] is True
    assert result.evidence.metadata["statement_verified"] is True
    assert KernelVerificationResult.from_dict(result.to_dict()) == result
    receipt = _receipt(result)
    assert receipt.verdict is ProofVerdict.PROVED
    assert receipt.authoritative_verdict is ProofVerdict.PROVED
    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert receipt.is_kernel_verified


def test_provider_status_and_claimed_assurance_cannot_upgrade_rejection() -> None:
    record, evidence, lock = _packet("lean")
    record["kernel_accepted"] = False
    record["failure_reason"] = "candidate proof did not close the goal"
    result = verify_kernel_reconstruction(
        record,
        evidence,
        lock,
        bindings=_bindings("lean"),
        provider_status="VERIFIED / kernel_verified / proved",
        independent=True,
    )
    receipt = _receipt(result)

    assert result.status is KernelVerificationStatus.REJECTED
    assert result.verdict is ProofVerdict.INCONCLUSIVE
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert receipt.provider_claimed_assurance is AssuranceLevel.ATTESTED
    assert receipt.authoritative_verdict is ProofVerdict.INCONCLUSIVE
    assert receipt.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert not receipt.is_kernel_verified


def test_provider_originated_reconstruction_packet_is_not_independent() -> None:
    record, evidence, lock = _packet("lean")

    result = verify_kernel_reconstruction(
        record,
        evidence,
        lock,
        bindings=_bindings("lean"),
        provider_status="verified",
    )

    assert result.failure_code is KernelFailureCode.BINDING_MISMATCH
    assert result.verdict is not ProofVerdict.PROVED
    assert not _receipt(result).is_kernel_verified


@pytest.mark.parametrize(
    ("mutation", "failure_code"),
    (
        ("timeout", KernelFailureCode.KERNEL_TIMED_OUT),
        ("candidate_mismatch", KernelFailureCode.BINDING_MISMATCH),
        ("environment_mismatch", KernelFailureCode.ENVIRONMENT_MISMATCH),
        ("digest_mismatch", KernelFailureCode.DIGEST_MISMATCH),
        ("changed_statement", KernelFailureCode.STATEMENT_MISMATCH),
        ("forbidden_declaration", KernelFailureCode.FORBIDDEN_DECLARATION),
        ("sorry", KernelFailureCode.INCOMPLETE_PROOF),
        ("corrupt_output", KernelFailureCode.KERNEL_REJECTED),
    ),
)
def test_timeout_mismatch_forbidden_and_corrupt_fixtures_fail_closed(
    mutation: str, failure_code: KernelFailureCode
) -> None:
    record, evidence, lock = _packet("lean")
    if mutation == "timeout":
        evidence["timed_out"] = True
    elif mutation == "candidate_mismatch":
        evidence["candidate_id"] = "different-candidate"
    elif mutation == "environment_mismatch":
        lock["itp"] = "coq"
    elif mutation == "digest_mismatch":
        evidence["checked_source"] = str(evidence["checked_source"]) + "\n"
    elif mutation == "changed_statement":
        evidence["checked_source"] = str(evidence["checked_source"]).replace(
            ": True", ": False"
        )
        evidence["checked_source_digest"] = _upstream_content_digest(
            {"checked_source": evidence["checked_source"]}
        )
    elif mutation == "forbidden_declaration":
        evidence["checked_source"] = (
            "axiom forged : False\n" + str(evidence["checked_source"])
        )
        evidence["checked_source_digest"] = _upstream_content_digest(
            {"checked_source": evidence["checked_source"]}
        )
    elif mutation == "sorry":
        evidence["checked_source"] = str(evidence["checked_source"]).replace(
            "by trivial", "sorry\n-- reconstructed snippet: by trivial"
        )
        evidence["checked_source_digest"] = _upstream_content_digest(
            {"checked_source": evidence["checked_source"]}
        )
    else:
        evidence["stdout"] = json.dumps(
            {"severity": "error", "data": "declaration has type mismatch"}
        )
        evidence["raw_output_digest"] = _upstream_content_digest(
            {"stdout": evidence["stdout"], "stderr": evidence["stderr"]}
        )
        record["kernel_output_digest"] = evidence["raw_output_digest"]

    result = verify_kernel_reconstruction(
        record,
        evidence,
        lock,
        bindings=_bindings("lean"),
        independent=True,
    )
    receipt = _receipt(result)

    assert result.failure_code is failure_code
    assert not result.accepted
    assert result.verdict is not ProofVerdict.PROVED
    assert result.assurance is not AssuranceLevel.KERNEL_VERIFIED
    assert not receipt.is_kernel_verified


@pytest.mark.parametrize(
    ("target", "replacement"),
    (
        ("coq", "admit."),
        ("isabelle", "sorry"),
    ),
)
def test_admit_and_sorry_never_produce_kernel_verified_receipts(
    target: str, replacement: str
) -> None:
    record, evidence, lock = _packet(target)
    proof = str(evidence["reconstructed_proof_text"])
    evidence["checked_source"] = str(evidence["checked_source"]).replace(
        proof, replacement + f"\n(* reconstructed snippet: {proof} *)"
    )
    evidence["checked_source_digest"] = _upstream_content_digest(
        {"checked_source": evidence["checked_source"]}
    )

    result = verify_kernel_reconstruction(
        record,
        evidence,
        lock,
        bindings=_bindings(target),
        independent=True,
    )

    assert result.failure_code is KernelFailureCode.INCOMPLETE_PROOF
    assert result.authoritative_verdict is not ProofVerdict.PROVED
    assert result.authoritative_assurance is not AssuranceLevel.KERNEL_VERIFIED
    assert not _receipt(result).is_kernel_verified


def test_request_and_obligation_bind_the_exact_theorem() -> None:
    obligation = _obligation()
    record, evidence, lock = _packet("coq")
    request = {
        "request_id": "request-1",
        "theorem_id": obligation.obligation_id,
        "goal_statement": "changed theorem",
    }
    result = verify_kernel_reconstruction(
        record,
        evidence,
        lock,
        obligation=obligation,
        request=request,
        candidate_id="candidate-1",
        kernel_id="kernel:coq@1.0.0",
        toolchain_id="lock-coq",
        independent=True,
    )

    assert result.failure_code is KernelFailureCode.STATEMENT_MISMATCH
    assert result.verdict is not ProofVerdict.PROVED


def test_kernel_unavailability_is_typed_and_never_verified() -> None:
    result = kernel_unavailable_result(
        target=KernelTarget.ISABELLE,
        bindings=_bindings("isabelle"),
        provider_status="verified",
    )

    assert result.status is KernelVerificationStatus.UNAVAILABLE
    assert result.failure_code is KernelFailureCode.KERNEL_UNAVAILABLE
    assert result.verdict is ProofVerdict.UNSUPPORTED
    assert result.assurance is AssuranceLevel.UNVERIFIED
    assert not _receipt(result).is_kernel_verified


def test_authoritative_verdict_is_checked_during_receipt_round_trip() -> None:
    record, evidence, lock = _packet("lean")
    result = IndependentKernelVerifier().verify(
        record,
        evidence,
        lock,
        bindings=_bindings("lean"),
        independent=True,
    )
    receipt = _receipt(result)
    payload = copy.deepcopy(receipt.to_dict())
    payload["authoritative_verdict"] = "disproved"

    with pytest.raises(ContractValidationError, match="authoritative verdict"):
        ProofReceipt.from_dict(payload)
