from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    AUTHORITY_CLASS,
    HIGH_CONFIDENCE,
    KernelSpend,
    SmtTriageAction,
    TypesafeKernelSkip,
    advise_decision_question,
    advise_proof_draft,
    escalation_meta_action,
    evaluate_closed_question,
    is_trap_family,
    maybe_verify_leanstral_draft,
    residual_uncertainty_bp,
    score_synthesis_candidate,
    triage_smt,
    typesafe_permitted,
)


def test_privacy_blocks_remote_typesafe() -> None:
    assert typesafe_permitted(privacy_class="local_only", remote_disclosure_permitted=True) is False
    assert typesafe_permitted(
        privacy_class="repository_private", remote_disclosure_permitted=False
    ) is False


def test_trap_family_detects_float_and_bitvector() -> None:
    assert is_trap_family(smtlib="(set-logic QF_FP)\n(fp.add RNE a b)")
    assert is_trap_family(case_id="float32_point_one_plus_point_two")
    assert is_trap_family(complexity="trap")
    assert not is_trap_family(smtlib="(set-logic UF)\n(check-sat)", case_id="fol_identity")


def test_proof_advice_skips_kernel_on_confident_abstain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Verdict:
        disposition = "abstain"
        claim_status = "unknown"
        confidence = HIGH_CONFIDENCE
        is_proof_body = 0.01
        well_formed = 0.01
        uses_forbidden = 0.01
        candidate_quality = 0.0
        parsed = SimpleNamespace(kind="abstain")

    monkeypatch.setattr(
        "ipfs_accelerate_py.leanstral_typesafe.solve_with_typesafe",
        lambda *_args, **_kwargs: _Verdict(),
    )
    receipt = advise_proof_draft(
        goal_id="g1",
        declaration="theorem t : True",
        draft_text="ABSTAIN",
    )
    assert receipt.accepted_as_authority is False
    assert receipt.authority_class == AUTHORITY_CLASS
    assert receipt.action == KernelSpend.SKIP.value


def test_proof_advice_spends_kernel_on_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Verdict:
        disposition = "reject"
        claim_status = "sat"
        confidence = 0.4
        is_proof_body = 0.2
        well_formed = 0.1
        uses_forbidden = 0.8
        candidate_quality = 0.1
        parsed = SimpleNamespace(kind="incomplete")

    monkeypatch.setattr(
        "ipfs_accelerate_py.leanstral_typesafe.solve_with_typesafe",
        lambda *_args, **_kwargs: _Verdict(),
    )
    receipt = advise_proof_draft(
        goal_id="g1",
        declaration="theorem t : True",
        draft_text="<|im_start|>thought>\nmaybe\n",
    )
    assert receipt.action == KernelSpend.SPEND.value
    assert "low_confidence" in receipt.reason_codes


def test_smt_triage_forces_z3_on_trap_even_if_confident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )
    called: list[int] = []
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_z3_benchmark.run_typesafe",
        lambda _case, timeout=30.0: called.append(1)
        or SimpleNamespace(
            status="unsat", confidence=0.99, usage={"input_tokens": 10, "output_tokens": 2}
        ),
    )
    receipt = triage_smt(
        english="fp32 0.1+0.2=0.3",
        smtlib="(set-logic QF_FP)\n(assert (fp.eq (fp.add RNE a b) c))\n(check-sat)\n",
        case_id="float32_point_one_plus_point_two",
        complexity="trap",
    )
    assert receipt.action == SmtTriageAction.RUN_Z3.value
    assert receipt.trap_family is True
    assert receipt.accepted_as_authority is False
    assert called == []
    assert "skip_typesafe_http" in receipt.reason_codes


def test_smt_triage_skips_z3_when_confident_and_not_trap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
        clear_samples,
    )

    clear_samples()
    monkeypatch.setenv("TYPESAFE_Z3_SPOT_CHECK_RATE", "0")
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_z3_benchmark.run_typesafe",
        lambda _case, timeout=30.0: SimpleNamespace(
            status="unsat", confidence=0.92, usage={}
        ),
    )
    receipt = triage_smt(
        english="forall x. P x -> P x",
        smtlib="(set-logic UF)\n(assert (not (forall ((x U)) (=> (P x) (P x)))))\n(check-sat)\n",
        case_id="fol_identity",
        complexity="easy",
    )
    assert receipt.action == SmtTriageAction.SKIP_Z3.value
    assert receipt.claim_status == "unsat"


def test_closed_question_rejects_choice_outside_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {
            "answer": SimpleNamespace(choice="invented_file.py", confidence=0.9),
        }
        scores = {}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_args, **_kwargs: _Result(),
    )
    receipt = evaluate_closed_question(
        question_id="q1",
        question_type="which_files_are_affected",
        alternatives=("src/a.py", "src/b.py"),
        state={"objective_id": "obj-1"},
    )
    assert receipt.action == "abstain"
    assert "choice_not_in_allowlist" in receipt.reason_codes


def test_whether_question_maps_noul_to_yes_no(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        nouls = {"answer": SimpleNamespace(noul=0.91)}
        choices = {}
        scores = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_args, **_kwargs: _Result(),
    )
    receipt = evaluate_closed_question(
        question_id="q2",
        question_type="whether_replan_is_required",
        state={"failed_step": "tests"},
    )
    assert receipt.action == "answered"
    assert receipt.choice == "yes"
    assert receipt.accepted_as_authority is False


def test_whether_with_alternatives_uses_allowlist_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="replan_suffix", confidence=0.4)}
        scores = {}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_args, **_kwargs: _Result(),
    )
    advice = advise_decision_question(
        question_id="q-replan",
        question_type="whether_replan_is_required",
        alternatives=("preserve", "replan_suffix"),
        state={"failed_step": "tests"},
    )
    assert advice.can_resolve is False
    assert advice.nominated_answer == "replan_suffix"
    assert advice.residual_uncertainty_bp >= 1
    assert advice.next_action == "CALL_REMOTE_STRONG_MODEL"
    assert advice.receipt.accepted_as_authority is False


def test_replan_atomic_signals_override_choice_in_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="preserve", confidence=0.9)}
        scores = {}
        nouls = {
            "mandatory_check_failed": SimpleNamespace(noul=0.92),
            "stale_evidence": SimpleNamespace(noul=0.05),
            "suffix_still_matches_tree": SimpleNamespace(noul=0.1),
        }

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_args, **_kwargs: _Result(),
    )
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
        planning_atomic_questions,
    )

    questions = planning_atomic_questions(
        "whether_replan_is_required",
        ("preserve", "replan_suffix"),
    )
    assert "mandatory_check_failed" in questions
    assert "stale_evidence" in questions
    assert "suffix_still_matches_tree" in questions
    assert "answer" in questions
    advice = advise_decision_question(
        question_id="q-replan-atomic",
        question_type="whether_replan_is_required",
        alternatives=("preserve", "replan_suffix"),
        state={"failure": {"step": "tests"}},
    )
    assert advice.nominated_answer == "replan_suffix"
    assert "failed_and_suffix_mismatch" in advice.receipt.reason_codes
    assert advice.can_resolve is False


def test_proof_question_escalates_to_smt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.typesafe_permitted",
        lambda **_kwargs: True,
    )

    class _Result:
        choices = {"answer": SimpleNamespace(choice="obl-1", confidence=0.99)}
        scores = {}
        nouls = {}

    monkeypatch.setattr(
        "ipfs_accelerate_py.typesafe_inference.system_one",
        lambda *_args, **_kwargs: _Result(),
    )
    advice = advise_decision_question(
        question_id="q-proof",
        question_type="which_proof_obligation_applies",
        alternatives=("obl-1", "obl-2"),
        state={"obligation_ids": ["obl-1", "obl-2"]},
    )
    assert advice.nominated_answer == "obl-1"
    assert advice.next_action == "RUN_SMT_OR_PROVER"
    assert advice.can_resolve is False


def test_human_question_escalates_to_human() -> None:
    assert (
        escalation_meta_action("whether_human_choice_is_irreducible", confidence=0.99, answered=True)
        == "REQUEST_HUMAN_DECISION"
    )


def test_residual_uncertainty_never_zero() -> None:
    assert residual_uncertainty_bp(1.0) == 1
    assert residual_uncertainty_bp(0.0) == 10_000
    assert residual_uncertainty_bp(0.4) > residual_uncertainty_bp(0.9)


def test_synthesis_score_refuses_unknown_candidate() -> None:
    receipt = score_synthesis_candidate(
        candidate_id="cand-9",
        allowlisted_ids=("cand-1", "cand-2"),
        state={},
    )
    assert receipt.action == "abstain"
    assert "candidate_not_allowlisted" in receipt.reason_codes


def test_maybe_verify_skips_kernel_without_calling_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"kernel": False}
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.advise_proof_draft",
        lambda **_kwargs: type(
            "R",
            (),
            {
                "action": KernelSpend.SKIP.value,
                "to_dict": lambda self: {"action": KernelSpend.SKIP.value},
            },
        )(),
    )

    def _kernel(*_args, **_kwargs):
        called["kernel"] = True
        raise AssertionError("kernel must not run")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider.verify_leanstral_draft",
        _kernel,
    )
    with pytest.raises(TypesafeKernelSkip):
        maybe_verify_leanstral_draft(
            SimpleNamespace(draft_text="ABSTAIN"),
            SimpleNamespace(theorem_id="t", obligation_id="o"),
            typesafe_precheck=True,
            native_source="theorem t : True := sorry",
            bindings=None,
        )
    assert called["kernel"] is False


IDENTITY_DECL = (
    "theorem recovery_goal (U : Type) (P : U → Prop) : ∀ x, P x → P x"
)
IDENTITY_SOURCE = IDENTITY_DECL + " := sorry\n#print axioms recovery_goal\n"


def _identity_theorem() -> object:
    from ipfs_accelerate_py.agent_supervisor.proof.proof_context import FixedTheoremIdentity

    digest = "sha256:" + hashlib.sha256(IDENTITY_SOURCE.encode("utf-8")).hexdigest()
    return FixedTheoremIdentity(
        theorem_id="recovery_goal",
        obligation_id="obligation-identity",
        declaration_name="recovery_goal",
        assumptions=("P",),
        conclusion="∀ x, P x → P x",
        template_id="identity",
        template_version="1",
        source_scope=("recovery_goal",),
        canonical_source_digest=digest,
    )


def _identity_draft(proof_text: str):
    from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
        LEANSTRAL_DRAFT_SCHEMA_VERSION,
        LeanstralProofDraft,
    )

    theorem = _identity_theorem()
    output_sha256 = hashlib.sha256(proof_text.encode("utf-8")).hexdigest()
    identity = {
        "schema_version": LEANSTRAL_DRAFT_SCHEMA_VERSION,
        "llm_provider": "leanstral_local",
        "model": "Leanstral",
        "obligation_ids": [theorem.obligation_id],
        "canonical_source_digest": theorem.canonical_source_digest,
        "theorem_id": theorem.theorem_id,
        "theorem_equivalence_key": theorem.equivalence_key,
        "context_capsule_id": "capsule-identity",
        "proposal_kind": "proof",
        "prompt_sha256": "b" * 64,
        "output_sha256": output_sha256,
    }
    artifact_id = (
        "leanstral-draft-"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
    )
    return LeanstralProofDraft(
        artifact_id=artifact_id,
        draft_text=proof_text,
        request_id="request-identity",
        llm_provider="leanstral_local",
        model="Leanstral",
        obligation_ids=(theorem.obligation_id,),
        canonical_source_digest=theorem.canonical_source_digest,
        prompt_sha256="b" * 64,
        output_sha256=output_sha256,
        timeout_ms=5_000,
        token_budget=256,
        theorem_id=theorem.theorem_id,
        theorem_equivalence_key=theorem.equivalence_key,
        context_capsule_id="capsule-identity",
        proposal_kind="proof",
    )


def _identity_bindings():
    from ipfs_accelerate_py.agent_supervisor.proof.kernel_verification import (
        KernelVerificationBindings,
    )

    return KernelVerificationBindings(
        obligation_id="obligation-identity",
        request_id="request-identity",
        candidate_id="candidate-identity",
        kernel_id="kernel:lean@test",
        toolchain_id="toolchain:lean@test",
    )


def _recording_kernel_runner(calls: list):
    def kernel_runner(**kwargs):
        calls.append(kwargs.get("source") or "")
        return {
            "command": ["/tools/lean", "--json", "Reconstruction.lean"],
            "stdout": json.dumps(
                {
                    "severity": "information",
                    "data": "'recovery_goal' does not depend on any axioms",
                }
            ),
            "stderr": "",
            "returncode": 0,
            "version": "Lean 4.test",
            "executable": "/tools/lean",
        }

    return kernel_runner


def test_provider_verify_draft_typesafe_precheck_spends_kernel_for_identity_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
        LeanstralGateStatus,
        create_leanstral_proof_provider,
    )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor.advise_proof_draft",
        lambda **_kwargs: SimpleNamespace(
            action=KernelSpend.SPEND.value,
            disposition="accept_candidate",
            claim_status="unsat",
            confidence=0.9,
        ),
    )
    calls: list = []
    provider = create_leanstral_proof_provider()
    result = provider.verify_draft(
        _identity_draft("by\n  intro x h\n  exact h"),
        _identity_theorem(),
        typesafe_precheck=True,
        native_source=IDENTITY_SOURCE,
        bindings=_identity_bindings(),
        kernel_runner=_recording_kernel_runner(calls),
    )
    assert calls, "kernel must run when TypeSafe spends"
    assert result.admission.accepted is True
    assert result.status in {LeanstralGateStatus.ACCEPTED, LeanstralGateStatus.REJECTED}


def test_live_identity_draft_typesafe_precheck_then_mocked_kernel() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
        LeanstralGateStatus,
        create_leanstral_proof_provider,
    )
    from ipfs_accelerate_py.leanstral_typesafe import (
        PROOF_PROMPT,
        discover_leanstral_base_url,
        leanstral_chat,
        parse_leanstral_output,
    )
    from ipfs_accelerate_py.typesafe_inference import typesafe_configured

    if not typesafe_configured():
        pytest.skip("TYPESAFE_API_KEY is not set")
    if not discover_leanstral_base_url():
        pytest.skip("Leanstral HTTP endpoint is not reachable")
    chat = leanstral_chat(
        PROOF_PROMPT.format(declaration=IDENTITY_DECL),
        base_url=discover_leanstral_base_url(),
        max_tokens=128,
        timeout=90.0,
    )
    parsed = parse_leanstral_output(str(chat["content"]))
    advice = advise_proof_draft(
        goal_id="h.fol_identity",
        declaration=IDENTITY_DECL,
        draft_text=str(chat["content"]),
        remote_disclosure_permitted=True,
    )
    print(
        json.dumps(
            {
                "parsed_kind": parsed.kind,
                "parsed_body": parsed.body,
                "advice": advice.to_dict(),
                "leanstral_usage": chat.get("usage"),
            },
            indent=2,
        )
    )
    assert advice.accepted_as_authority is False
    assert advice.action in {KernelSpend.SPEND.value, KernelSpend.SKIP.value}

    if parsed.kind != "proof_body" or advice.action == KernelSpend.SKIP.value:
        return

    calls: list = []
    provider = create_leanstral_proof_provider()
    result = provider.verify_draft(
        _identity_draft(parsed.body),
        _identity_theorem(),
        typesafe_precheck=True,
        native_source=IDENTITY_SOURCE,
        bindings=_identity_bindings(),
        kernel_runner=_recording_kernel_runner(calls),
    )
    assert calls, "identity proof_body should spend the kernel"
    assert result.admission.accepted is True
    assert result.status in {LeanstralGateStatus.ACCEPTED, LeanstralGateStatus.REJECTED}
    assert result.kernel_verification.accepted == (result.status is LeanstralGateStatus.ACCEPTED)
