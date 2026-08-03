"""PDR-041 mutation-authority proof boundary and forgery matrix."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    ProgramLogicAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
    DOCTOR_AUTHORITATIVE_PROOF_INTERFACE,
    DeterministicDoctorHammer,
    DoctorAuthoritativeProofReceipt,
    DoctorExactLoweringReceipt,
    DoctorExecutableRole,
    DoctorHammerAuthorityError,
    DoctorHammerReasonCode,
    DoctorPinnedExecutable,
    DoctorProofAuthorityDisposition,
    DoctorReviewedTheorem,
)
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorSealedReceiptError,
    DoctorSealedReceiptRef,
    DoctorSealedReceiptStore,
)

_SOLVER_PROGRAM = r"""
import base64,hashlib,json,sys
c=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def cid(v):
 d=hashlib.sha256(c(v)).digest()
 return 'b'+base64.b32encode(b'\x01\xa9\x02\x12\x20'+d).decode().rstrip('=').lower()
d=json.loads(sys.stdin.buffer.read());t=d['theorem'];l=d['lowering'];tc=cid(t);lc=cid(l)
p='proof:'+hashlib.sha256((tc+'|'+lc).encode()).hexdigest()
o={'status':'proved','theorem_cid':tc,'lowering_cid':lc,'property_id':t['property_id'],'consequence_ref':t['consequence_ref'],'premise_ids':t['premise_ids'],'proof_object':p}
sys.stdout.write(c(o).decode())
""".strip()

_KERNEL_PROGRAM = r"""
import base64,hashlib,json,sys
c=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def cid(v):
 d=hashlib.sha256(c(v)).digest()
 return 'b'+base64.b32encode(b'\x01\xa9\x02\x12\x20'+d).decode().rstrip('=').lower()
d=json.loads(sys.stdin.buffer.read());t=d['theorem'];l=d['lowering'];n=d['native_receipt'];tc=cid(t);lc=cid(l)
p='proof:'+hashlib.sha256((tc+'|'+lc).encode()).hexdigest()
if n['proof_object']!=p: raise SystemExit(7)
pc=cid({'proof_object':p});nc=cid(n)
o={'status':'kernel_verified','theorem_cid':tc,'lowering_cid':lc,'native_receipt_cid':nc,'property_id':t['property_id'],'consequence_ref':t['consequence_ref'],'premise_ids':t['premise_ids'],'proof_object_cid':pc,'kernel_id':'kernel:doctor-json-replay@1'}
sys.stdout.write(c(o).decode())
""".strip()


def _roots(**overrides: str) -> ProgramLogicAuthorityRoots:
    values = {
        "repository_id": "repository:fixture",
        "objective_id": "objective:pdr-041",
        "trace_id": "trace:proof",
        "change_id": "change:repair",
        "consumer_id": "consumer:mutation-gate",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "graph_id": "graph:fixture",
        "index_id": "index:fixture",
        "corpus_id": "corpus:fixture",
        "model_id": "model:none",
        "translator_id": "translator:doctor-logic-ir@1",
        "toolchain_id": "toolchain:python-json-kernel@1",
        "policy_id": "policy:fixture",
        "environment_id": "environment:sealed-validation",
    }
    values.update(overrides)
    return ProgramLogicAuthorityRoots(**values)


def _theorem(roots: ProgramLogicAuthorityRoots | None = None) -> DoctorReviewedTheorem:
    roots = roots or _roots()
    return DoctorReviewedTheorem(
        roots=roots,
        theorem_id="theorem:repair-preserves-contract",
        property_id="property:consumer-contract-preserved",
        claim_id="claim:add-required-context",
        consequence_ref="consequence:add-context-argument",
        theorem_body=(
            "theorem theorem:repair-preserves-contract "
            "property property:consumer-contract-preserved "
            "claim claim:add-required-context "
            "consequence:add-context-argument forall event, "
            "apply_repair(event) = reviewed_contract(event)"
        ),
        body_format="doctor-logic-ir",
        premise_ids=("premise:reviewed-contract", "premise:current-call-graph"),
        assumption_ids=("assumption:closed-world-scope",),
        review_receipt_id="review:operator:pdr-041",
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        policy_id=roots.policy_id,
    )


def _lowering(theorem: DoctorReviewedTheorem) -> DoctorExactLoweringReceipt:
    anchors = (
        f"{theorem.theorem_id} {theorem.property_id} {theorem.claim_id} "
        f"{theorem.consequence_ref}"
    )
    return DoctorExactLoweringReceipt.create(
        theorem,
        logic_ir_statement=f"assert {anchors} -> reviewed_contract_preserved",
        native_statement=f"theorem {anchors} := reviewed_contract_preserved",
    )


def _pins(roots: ProgramLogicAuthorityRoots) -> tuple[DoctorPinnedExecutable, DoctorPinnedExecutable]:
    executable = "/usr/bin/python3.12"
    digest = "sha256:" + hashlib.sha256(Path(executable).read_bytes()).hexdigest()
    solver = DoctorPinnedExecutable(
        role=DoctorExecutableRole.SOLVER,
        executable_path=executable,
        executable_sha256=digest,
        argv=("-I", "-S", "-c", _SOLVER_PROGRAM),
        verifier_id="solver:doctor-json-proof@1",
        toolchain_id=roots.toolchain_id,
        environment_id=roots.environment_id,
    )
    kernel = DoctorPinnedExecutable(
        role=DoctorExecutableRole.KERNEL,
        executable_path=executable,
        executable_sha256=digest,
        argv=("-I", "-S", "-c", _KERNEL_PROGRAM),
        verifier_id="kernel:doctor-json-replay@1",
        toolchain_id=roots.toolchain_id,
        environment_id=roots.environment_id,
    )
    return solver, kernel


def _verified(tmp_path):
    roots = _roots()
    theorem = _theorem(roots)
    lowering = _lowering(theorem)
    solver, kernel = _pins(roots)
    store = DoctorSealedReceiptStore(
        tmp_path / "proof-authority.sqlite",
        authority_id="operator-authority:pdr-041",
    )
    hammer = DeterministicDoctorHammer(
        authoritative_store=store,
        trusted_executable_pins=(solver, kernel),
    )
    receipt = hammer.verify_authoritative(
        theorem,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    return roots, theorem, lowering, solver, kernel, store, hammer, receipt


def test_native_execution_sealed_reload_and_fresh_kernel_replay(tmp_path) -> None:
    roots, theorem, _, _, _, _, hammer, receipt = _verified(tmp_path)
    assert DOCTOR_AUTHORITATIVE_PROOF_INTERFACE == "DoctorAuthoritativeProofReceipt@1"
    assert type(receipt) is DoctorAuthoritativeProofReceipt
    assert receipt.disposition is DoctorProofAuthorityDisposition.VERIFIED
    assert receipt.mutation_capable is True
    assert receipt.write_authority is False
    assert receipt.selected_consequence_ref == theorem.consequence_ref
    assert receipt.property_id == theorem.property_id
    assert receipt.toolchain_id == roots.toolchain_id
    assert receipt.native_execution is not None
    assert receipt.native_execution.executed is True
    assert receipt.kernel_replay is not None
    assert receipt.kernel_replay.kernel_verified is True
    assert receipt.native_store_ref is not None
    assert receipt.kernel_store_ref is not None
    assert receipt.authority_store_ref is not None
    assert receipt.native_store_ref.entry_id != receipt.kernel_store_ref.entry_id
    restored = DoctorAuthoritativeProofReceipt.from_dict(receipt.to_record())
    assert restored.content_id == receipt.content_id
    assert restored.mutation_capable is False
    reverified = hammer.reverify_authoritative(restored, current_roots=roots)
    assert reverified.content_id == receipt.content_id
    assert reverified.mutation_capable is True


def test_nonempty_reviewed_semantic_body_and_exact_lowering_are_mandatory() -> None:
    roots = _roots()
    with pytest.raises(DoctorHammerAuthorityError):
        replace(_theorem(roots), theorem_body=" ")
    with pytest.raises(DoctorHammerAuthorityError):
        replace(
            _theorem(roots),
            theorem_body=(
                "theorem:repair-preserves-contract "
                "property:consumer-contract-preserved "
                "claim:add-required-context consequence:add-context-argument"
            ),
        )
    theorem = _theorem(roots)
    lowering = _lowering(theorem)
    forged = lowering.to_record()
    forged["native_statement"] = forged["native_statement"].replace(
        theorem.theorem_id, "theorem:wrong"
    )
    with pytest.raises(DoctorHammerAuthorityError):
        DoctorExactLoweringReceipt.from_dict(forged)


@pytest.mark.parametrize(
    "candidate",
    [
        {"theorem_id": "caller-map", "reviewed": True},
        type("DuckTheorem", (), {"reviewed": True, "theorem_body": "x = x"})(),
        True,
    ],
)
def test_raw_mapping_duck_boolean_and_test_injection_cannot_admit(
    tmp_path, candidate
) -> None:
    roots = _roots()
    theorem = _theorem(roots)
    lowering = _lowering(theorem)
    solver, kernel = _pins(roots)
    store = DoctorSealedReceiptStore(
        tmp_path / "raw.sqlite", authority_id="operator-authority:pdr-041"
    )
    hammer = DeterministicDoctorHammer(
        authoritative_store=store,
        trusted_executable_pins=(solver, kernel),
    )
    receipt = hammer.verify_authoritative(  # type: ignore[arg-type]
        candidate,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    assert receipt.mutation_capable is False
    assert receipt.disposition is DoctorProofAuthorityDisposition.REJECTED


def test_provider_local_or_untrusted_receipts_cannot_admit(tmp_path) -> None:
    roots = _roots()
    theorem = _theorem(roots)
    lowering = _lowering(theorem)
    solver, kernel = _pins(roots)
    without_store = DeterministicDoctorHammer(
        trusted_executable_pins=(solver, kernel)
    ).verify_authoritative(
        theorem,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    assert without_store.disposition is DoctorProofAuthorityDisposition.ABSTAINED
    assert (
        DoctorHammerReasonCode.SEALED_RECEIPT_REQUIRED.value
        in without_store.reason_codes
    )
    store = DoctorSealedReceiptStore(
        tmp_path / "untrusted.sqlite",
        authority_id="operator-authority:pdr-041",
    )
    untrusted = DeterministicDoctorHammer(authoritative_store=store)
    receipt = untrusted.verify_authoritative(
        theorem,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    assert receipt.mutation_capable is False
    assert DoctorHammerReasonCode.EXECUTABLE_NOT_PINNED.value in receipt.reason_codes


def test_wrong_tool_tree_policy_uniqueness_and_executable_digest_reject(tmp_path) -> None:
    roots = _roots()
    theorem = _theorem(roots)
    lowering = _lowering(theorem)
    solver, kernel = _pins(roots)
    store = DoctorSealedReceiptStore(
        tmp_path / "wrong-bindings.sqlite",
        authority_id="operator-authority:pdr-041",
    )
    hammer = DeterministicDoctorHammer(
        authoritative_store=store,
        trusted_executable_pins=(solver, kernel),
    )
    stale = hammer.verify_authoritative(
        theorem,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=_roots(tree_id="tree:changed"),
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    assert stale.mutation_capable is False
    stale_policy = hammer.verify_authoritative(
        theorem,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=_roots(policy_id="policy:changed"),
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    assert stale_policy.mutation_capable is False
    wrong_unique = hammer.verify_authoritative(
        theorem,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref, "consequence:forged"),
    )
    assert wrong_unique.mutation_capable is False
    wrong_theorem = replace(
        theorem,
        theorem_id="theorem:other",
        theorem_body=theorem.theorem_body.replace(
            theorem.theorem_id,
            "theorem:other",
        ),
    )
    mismatched_theorem = hammer.verify_authoritative(
        wrong_theorem,
        lowering,
        solver_pin=solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(wrong_theorem.consequence_ref,),
    )
    assert mismatched_theorem.mutation_capable is False
    wrong_tool_solver = replace(solver, toolchain_id="toolchain:wrong")
    wrong_tool_hammer = DeterministicDoctorHammer(
        authoritative_store=store,
        trusted_executable_pins=(wrong_tool_solver, kernel),
    )
    wrong_tool = wrong_tool_hammer.verify_authoritative(
        theorem,
        lowering,
        solver_pin=wrong_tool_solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    assert wrong_tool.mutation_capable is False
    assert DoctorHammerReasonCode.WRONG_TOOLCHAIN.value in wrong_tool.reason_codes
    bad_solver = replace(solver, executable_sha256="sha256:" + "0" * 64)
    bad_hammer = DeterministicDoctorHammer(
        authoritative_store=store,
        trusted_executable_pins=(bad_solver, kernel),
    )
    substituted = bad_hammer.verify_authoritative(
        theorem,
        lowering,
        solver_pin=bad_solver,
        kernel_pin=kernel,
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    assert substituted.mutation_capable is False
    assert (
        DoctorHammerReasonCode.KERNEL_REPLAY_FAILED.value
        in substituted.reason_codes
    )


def test_forged_cid_stale_current_roots_and_sealed_store_tamper_fail(tmp_path) -> None:
    _, _, _, _, _, store, hammer, receipt = _verified(tmp_path)
    forged = receipt.to_record()
    forged["content_id"] = "baforged"
    with pytest.raises(DoctorHammerAuthorityError):
        DoctorAuthoritativeProofReceipt.from_dict(forged)
    with pytest.raises(DoctorHammerAuthorityError):
        hammer.reverify_authoritative(
            receipt, current_roots=_roots(policy_id="policy:changed")
        )
    assert receipt.native_store_ref is not None
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE doctor_sealed_receipts
            SET receipt_json = replace(receipt_json, 'property:', 'forged:')
            WHERE sequence = ?
            """,
            (receipt.native_store_ref.sequence,),
        )
    with pytest.raises(DoctorSealedReceiptError):
        store.reload(receipt.native_store_ref, type(receipt.native_execution))
    with pytest.raises(DoctorSealedReceiptError):
        store.reload(  # type: ignore[arg-type]
            receipt.native_store_ref.to_dict(), type(receipt.native_execution)
        )


def test_forged_store_reference_and_identity_only_flags_are_not_authority(
    tmp_path,
) -> None:
    _, _, _, _, _, store, _, receipt = _verified(tmp_path)
    assert receipt.native_store_ref is not None
    forged_ref = DoctorSealedReceiptRef.from_dict(
        {
            **receipt.native_store_ref.to_dict(),
            "seal": "hmac-sha256:" + "0" * 64,
        }
    )
    with pytest.raises(DoctorSealedReceiptError):
        store.reload(forged_ref, type(receipt.native_execution))
    assert receipt.authority_store_ref is not None
    forged_authority_ref = DoctorSealedReceiptRef.from_dict(
        {
            **receipt.authority_store_ref.to_dict(),
            "seal": "hmac-sha256:" + "f" * 64,
        }
    )
    assert (
        replace(
            receipt,
            authority_store_ref=forged_authority_ref,
        ).mutation_capable
        is False
    )
    assert replace(receipt, receipt_id="receipt:forged").mutation_capable is False
    lookalike = {
        "schema": receipt.schema,
        "disposition": "verified",
        "uniqueness_satisfied": True,
        "round_trip_ok": True,
        "kernel_checked": True,
        "identity_verified": True,
    }
    with pytest.raises((TypeError, DoctorHammerAuthorityError, DoctorSealedReceiptError)):
        store.append(lookalike)
