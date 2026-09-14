"""SAWM-019 e-graph normalization and relation-promotion tests."""

from __future__ import annotations

import ast
import importlib
import inspect
import threading
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_relations import (
    ContradictionDisposition,
    RelationAuthorityStatus,
    RelationKind,
    RelationScope,
    RelationScopeKind,
    RelationValidationVerdict,
)

from ipfs_accelerate_py.agent_supervisor.analysis.program_egraph import (
    BUILTIN_THEORY_ID,
    PROGRAM_EGRAPH_NORMALIZER_INTERFACE,
    PROGRAM_WORLD_EGRAPH_INTERFACE,
    SAWM_EGRAPH_NORMALIZATION_EVIDENCE,
    EqualitySaturationPlan,
    EvidenceBasis,
    FragmentPurity,
    ProgramEGraphAuthorityError,
    ProgramEGraphNormalizer,
    ProgramEGraphStaleError,
    PromotionDisposition,
    RelationPromotionProposal,
    RewriteRule,
    RewriteSoundness,
    RewriteTheory,
    SaturationBounds,
    SaturationDisposition,
    Term,
    builtin_python_pure_theory,
    normalize_program_fragment,
    propose_relation_promotion,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EGRAPH_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/analysis/program_egraph.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _env() -> str:
    return _cid("env-v1")


def _scope(*, theory_cid: str | None = None) -> RelationScope:
    return RelationScope(
        scope_kind=RelationScopeKind.THEORY,
        language="python",
        subject_cids=(_cid("subject:pure-fragment"),),
        theory_or_policy_cid=theory_cid or builtin_python_pure_theory().theory_cid,
        environment_binding_cid=_env(),
    )


def _normalize(fragment: str, peer: str | None = None, **kwargs: Any) -> EqualitySaturationPlan:
    return normalize_program_fragment(
        fragment,
        peer_fragment=peer,
        environment_binding_cid=kwargs.pop("environment_binding_cid", _env()),
        **kwargs,
    )


def _promote(left: str, right: str, **kwargs: Any) -> RelationPromotionProposal:
    theory = kwargs.get("theory")
    theory_cid = theory.theory_cid if isinstance(theory, RewriteTheory) else None
    scope = kwargs.pop("scope", _scope(theory_cid=theory_cid))
    return propose_relation_promotion(
        left,
        right,
        scope=scope,
        environment_binding_cid=kwargs.pop("environment_binding_cid", _env()),
        **kwargs,
    )


def test_public_interfaces_and_symbols_are_present() -> None:
    tree = ast.parse(EGRAPH_PATH.read_text(encoding="utf-8"))
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "normalize_program_fragment" in functions
    assert "propose_relation_promotion" in functions
    assert "ProgramEGraphNormalizer" in classes
    assert "EqualitySaturationPlan" in classes
    assert "RelationPromotionProposal" in classes
    assert PROGRAM_WORLD_EGRAPH_INTERFACE == "ProgramWorldEGraph@1"
    assert PROGRAM_EGRAPH_NORMALIZER_INTERFACE == "ProgramEGraphNormalizer@1"
    assert SAWM_EGRAPH_NORMALIZATION_EVIDENCE == "sawm/egraph-normalization@1"


def test_import_has_no_io_or_thread_side_effects() -> None:
    before = {thread.name for thread in threading.enumerate()}
    imported = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.analysis.program_egraph"
    )
    after = {thread.name for thread in threading.enumerate()}
    assert after == before
    assert inspect.isfunction(imported.normalize_program_fragment)
    assert inspect.isfunction(imported.propose_relation_promotion)
    assert inspect.isclass(imported.ProgramEGraphNormalizer)


def test_rewrite_rule_identity_is_stable_and_soundness_is_required() -> None:
    theory = builtin_python_pure_theory()
    assert theory.theory_id == BUILTIN_THEORY_ID
    again = builtin_python_pure_theory()
    assert theory.theory_cid == again.theory_cid
    assert {rule.rule_id for rule in theory.rules}
    for rule in theory.rules:
        assert rule.rule_cid
        assert rule.soundness == RewriteSoundness.ALGEBRAIC_IDENTITY.value
        assert rule.review_ref == SAWM_EGRAPH_NORMALIZATION_EVIDENCE
        rebuilt = RewriteRule.from_dict(rule.to_dict())
        assert rebuilt.rule_cid == rule.rule_cid
    with pytest.raises(ProgramEGraphAuthorityError):
        RewriteRule(
            rule_id="sim",
            lhs=Term.pat("x"),
            rhs=Term.pat("x"),
            soundness="similarity",
            theory_id="bad",
        )


def test_deterministic_normal_form_for_identities_and_constants() -> None:
    zero = _normalize("x + 0")
    assert zero.disposition == SaturationDisposition.NORMALIZED.value
    assert zero.normal_form == "x"
    assert zero.equivalent
    assert zero.saturation_fixed_point
    assert "add-right-identity" in zero.applied_rule_ids or "add-left-identity" in zero.applied_rule_ids or zero.normal_form == "x"
    folded = _normalize("1 + 2")
    assert folded.normal_form == "3"
    assert "constant-fold" in folded.applied_rule_ids
    double = _normalize("not (not True)")
    assert double.normal_form == "True"
    again = _normalize("x + 0")
    assert again.plan_cid == zero.plan_cid
    assert again.to_dict()["plan_cid"] == zero.plan_cid


def test_commutativity_and_associativity_prove_equivalence() -> None:
    comm = _normalize("x + y", peer="y + x")
    assert comm.disposition == SaturationDisposition.EQUIVALENT.value
    assert comm.equivalent
    assert comm.supports_equivalence_promotion
    assoc = _normalize("(x + y) + z", peer="x + (y + z)")
    assert assoc.equivalent
    mixed = _normalize("(x + 0) + y", peer="y + x")
    assert mixed.equivalent


def test_false_equivalence_negatives_do_not_promote() -> None:
    shifted = _normalize("x + 1", peer="x + 2")
    assert shifted.disposition == SaturationDisposition.INEQUIVALENT_UNPROVED.value
    assert not shifted.equivalent
    assert not shifted.supports_equivalence_promotion
    proposal = _promote("x + 1", "x + 2")
    assert proposal.disposition == PromotionDisposition.ABSTAINED.value
    assert proposal.claim is None
    assert proposal.independently_admitted is False
    assert proposal.justifies_repair is False
    and_or = _promote("a and b", "a or b")
    assert and_or.disposition == PromotionDisposition.ABSTAINED.value


def test_sound_normalization_proposes_candidate_relation_only() -> None:
    proposal = _promote("x + 0", "x")
    assert proposal.disposition == PromotionDisposition.PROPOSED.value
    assert proposal.evidence_basis == EvidenceBasis.NORMALIZATION.value
    assert proposal.proposal_only is True
    assert proposal.independently_admitted is False
    assert proposal.grants_semantic_authority is False
    assert proposal.may_influence_planning is False
    assert proposal.claim is not None
    assert proposal.claim.authority_status == RelationAuthorityStatus.CANDIDATE.value
    assert proposal.claim.relation_kind == RelationKind.LOGICAL_EQUIVALENCE.value
    assert proposal.recommended_authority_status == RelationAuthorityStatus.VALIDATED.value
    assert proposal.saturation_plan_cid
    assert proposal.justifies_repair is True
    encoded = proposal.to_dict()
    assert encoded["independently_admitted"] is False
    assert encoded["proposal_only"] is True


def test_reflexive_fragments_promote_from_equality() -> None:
    proposal = _promote("x", "x", relation_kind=RelationKind.EQUALITY)
    assert proposal.disposition == PromotionDisposition.PROPOSED.value
    assert proposal.evidence_basis in {
        EvidenceBasis.REFLEXIVITY.value,
        EvidenceBasis.NORMALIZATION.value,
    }
    assert proposal.claim is not None
    assert proposal.claim.left_cid == proposal.claim.right_cid


def test_similarity_and_model_output_cannot_support_promotion() -> None:
    with pytest.raises(ProgramEGraphAuthorityError):
        _promote("x + 0", "x", similarity={"kind": "similarity", "score": 91})
    with pytest.raises(ProgramEGraphAuthorityError):
        _promote("x + 0", "x", model_output="the model says these are equal")
    with pytest.raises(ProgramEGraphAuthorityError):
        _promote(
            "x + 0",
            "x",
            evidence={"kind": "embedding", "nearest": "x"},
        )
    with pytest.raises(ProgramEGraphAuthorityError):
        propose_relation_promotion(
            "foo(x)",
            "bar(x)",
            scope=_scope(),
            environment_binding_cid=_env(),
            relation_kind="similarity",
        )


def test_unsupported_effects_concurrency_and_opaque_calls_abstain() -> None:
    opened = _normalize("open('path')")
    assert opened.disposition == SaturationDisposition.UNSUPPORTED.value
    assert opened.purity == FragmentPurity.EFFECTFUL.value
    assert any(code.startswith("effectful_call:") for code in opened.reason_codes)
    threaded = _normalize("threading")
    assert threaded.purity == FragmentPurity.CONCURRENT.value
    opaque = _normalize("mystery(x)")
    assert opaque.disposition == SaturationDisposition.UNSUPPORTED.value
    assert opaque.purity == FragmentPurity.OPAQUE.value
    awaited = _normalize("await x")
    assert awaited.disposition == SaturationDisposition.UNSUPPORTED.value
    attr = _normalize("x.attr")
    assert attr.disposition == SaturationDisposition.UNSUPPORTED.value
    proposal = _promote("open('path')", "open('path')")
    assert proposal.disposition == PromotionDisposition.UNSUPPORTED.value
    assert proposal.justifies_repair is False
    assert proposal.independently_admitted is False


def test_unsupported_language_and_floats_are_typed() -> None:
    js = _normalize("x + 0", language="javascript")
    assert js.disposition == SaturationDisposition.UNSUPPORTED.value
    assert "language_unavailable:javascript" in js.reason_codes
    floating = _normalize("1.5 + 2.5")
    assert floating.disposition == SaturationDisposition.UNSUPPORTED.value
    assert "unsupported_constant" in floating.reason_codes


def test_conflicting_rules_abstain_and_never_justify_repair() -> None:
    x = Term.pat("x")
    theory = RewriteTheory(
        theory_id="conflict-theory@1",
        review_refs=(SAWM_EGRAPH_NORMALIZATION_EVIDENCE,),
        rules=(
            RewriteRule(
                rule_id="to-true",
                lhs=x,
                rhs=Term.const(True),
                soundness=RewriteSoundness.DECLARED_AXIOM,
                theory_id="conflict-theory@1",
                review_ref="review:conflict",
            ),
            RewriteRule(
                rule_id="to-false",
                lhs=x,
                rhs=Term.const(False),
                soundness=RewriteSoundness.DECLARED_AXIOM,
                theory_id="conflict-theory@1",
                review_ref="review:conflict",
            ),
        ),
    )
    static_conflicts = theory.oriented_conflicts()
    assert static_conflicts
    plan = _normalize("x", theory=theory)
    assert plan.disposition == SaturationDisposition.CONFLICT.value
    assert not plan.equivalent
    assert not plan.justifies_repair
    assert plan.ex_falso_admission is False
    proposal = _promote("x", "y", theory=theory, scope=_scope(theory_cid=theory.theory_cid))
    assert proposal.disposition == PromotionDisposition.CONFLICT.value
    assert proposal.justifies_repair is False
    assert proposal.contradiction_disposition == ContradictionDisposition.ABSTENTION.value
    assert proposal.validation_verdict == RelationValidationVerdict.CONFLICT.value
    assert proposal.claim is None


def test_inconsistent_egraph_does_not_ex_falso_promote() -> None:
    x = Term.pat("x")
    theory = RewriteTheory(
        theory_id="collapse-theory@1",
        review_refs=(SAWM_EGRAPH_NORMALIZATION_EVIDENCE,),
        rules=(
            RewriteRule(
                rule_id="any-to-true",
                lhs=Term.app("add", x, Term.const(0)),
                rhs=Term.const(True),
                soundness=RewriteSoundness.DECLARED_AXIOM,
                theory_id="collapse-theory@1",
                review_ref="review:collapse",
            ),
            RewriteRule(
                rule_id="any-to-false",
                lhs=Term.app("add", x, Term.const(0)),
                rhs=Term.const(False),
                soundness=RewriteSoundness.DECLARED_AXIOM,
                theory_id="collapse-theory@1",
                review_ref="review:collapse",
            ),
        ),
    )
    plan = _normalize("x + 0", peer="True", theory=theory)
    assert plan.disposition == SaturationDisposition.CONFLICT.value
    assert plan.ex_falso_admission is False
    assert "contradiction_abstention" in plan.reason_codes or "conflicting_rules" in plan.reason_codes
    proposal = _promote(
        "x + 0",
        "y + 0",
        theory=theory,
        scope=_scope(theory_cid=theory.theory_cid),
    )
    assert proposal.disposition in {
        PromotionDisposition.CONFLICT.value,
        PromotionDisposition.ABSTAINED.value,
    }
    assert proposal.justifies_repair is False
    assert proposal.claim is None


def test_expanding_rule_hits_saturation_bounds() -> None:
    x = Term.pat("x")
    theory = RewriteTheory(
        theory_id="expand-theory@1",
        review_refs=(SAWM_EGRAPH_NORMALIZATION_EVIDENCE,),
        rules=(
            RewriteRule(
                rule_id="expand",
                lhs=x,
                rhs=Term.app("add", x, Term.const(0)),
                soundness=RewriteSoundness.DECLARED_AXIOM,
                theory_id="expand-theory@1",
                review_ref="review:expand",
            ),
        ),
    )
    plan = _normalize(
        "x",
        theory=theory,
        bounds=SaturationBounds(
            max_iterations=1,
            max_eclasses=4,
            max_enodes=8,
            max_matches_per_rule=4,
        ),
    )
    assert plan.disposition == SaturationDisposition.BOUND_EXHAUSTED.value
    assert "saturation_bound_exhausted" in plan.reason_codes
    assert not plan.equivalent
    proposal = _promote(
        "x",
        "x + 0",
        theory=theory,
        scope=_scope(theory_cid=theory.theory_cid),
        bounds=SaturationBounds(
            max_iterations=1,
            max_eclasses=4,
            max_enodes=8,
            max_matches_per_rule=4,
        ),
    )
    assert proposal.disposition in {
        PromotionDisposition.UNSUPPORTED.value,
        PromotionDisposition.ABSTAINED.value,
    }
    assert proposal.justifies_repair is False


def test_kernel_proof_supports_promotion_but_does_not_self_admit() -> None:
    proof = {
        "disposition": "admitted_proof",
        "proof_status": "kernel_verified",
        "reconstruction_id": "native:theorem:x-plus-zero",
        "model_authored": False,
        "admitted": True,
        "tree_id": "tree:current",
        "admission_id": "adm:kernel-1",
        "compilation_id": "comp:1",
    }
    proposal = _promote("opaque_left(x)", "opaque_right(x)", proof_admission=proof)
    assert proposal.disposition == PromotionDisposition.PROPOSED.value
    assert proposal.evidence_basis == EvidenceBasis.KERNEL_PROOF.value
    assert proposal.recommended_authority_status == RelationAuthorityStatus.PROVED.value
    assert proposal.independently_admitted is False
    assert proposal.claim is not None
    assert proposal.claim.authority_status == RelationAuthorityStatus.CANDIDATE.value
    assert proposal.proof_receipt_cid
    model_proof = dict(proof)
    model_proof["model_authored"] = True
    with pytest.raises(ProgramEGraphAuthorityError):
        _promote("opaque_left(x)", "opaque_right(x)", proof_admission=model_proof)


def test_admitted_refutation_does_not_promote() -> None:
    proof = {
        "disposition": "admitted_refutation",
        "proof_status": "validated_refuted",
        "replay_id": "replay:1",
        "model_authored": False,
        "admitted": True,
        "tree_id": "tree:current",
        "admission_id": "adm:refute-1",
    }
    proposal = _promote("x + 1", "x + 2", proof_admission=proof)
    assert proposal.disposition == PromotionDisposition.REFUTED.value
    assert proposal.claim is None
    assert proposal.justifies_repair is False
    assert proposal.validation_verdict == RelationValidationVerdict.REFUTED.value


def test_stale_environment_fails_closed() -> None:
    with pytest.raises(ProgramEGraphStaleError):
        normalize_program_fragment(
            "x + 0",
            environment_binding_cid=_env(),
            expected_environment_binding_cid=_cid("env-other"),
        )
    with pytest.raises(ProgramEGraphStaleError):
        propose_relation_promotion(
            "x + 0",
            "x",
            scope=_scope(),
            environment_binding_cid=_cid("env-other"),
        )


def test_normalizer_class_binds_freshness_and_delegates() -> None:
    theory = builtin_python_pure_theory()
    normalizer = ProgramEGraphNormalizer(
        expected_environment_binding_cid=_env(),
        expected_theory_cid=theory.theory_cid,
        theory=theory,
    )
    plan = normalizer.normalize("x * 1", environment_binding_cid=_env())
    assert plan.disposition == SaturationDisposition.NORMALIZED.value
    assert plan.normal_form == "x"
    proposal = normalizer.propose_promotion(
        "(x * 1) + 0",
        "x",
        scope=_scope(theory_cid=theory.theory_cid),
        environment_binding_cid=_env(),
    )
    assert proposal.disposition == PromotionDisposition.PROPOSED.value
    with pytest.raises(ProgramEGraphStaleError):
        normalizer.normalize("x", environment_binding_cid=_cid("env-stale"))


def test_proved_rule_requires_proof_receipt_and_rejects_unbound_vars() -> None:
    with pytest.raises(Exception):
        RewriteRule(
            rule_id="proved-missing",
            lhs=Term.pat("x"),
            rhs=Term.pat("x"),
            soundness=RewriteSoundness.PROVED,
            theory_id="t",
        )
    with pytest.raises(Exception):
        RewriteRule(
            rule_id="unbound",
            lhs=Term.pat("x"),
            rhs=Term.pat("y"),
            soundness=RewriteSoundness.ALGEBRAIC_IDENTITY,
            theory_id="t",
        )


def test_payloads_reject_similarity_fields_and_floats() -> None:
    plan = _normalize("x")
    payload = plan.to_dict()
    assert "score" not in payload
    assert "similarity" not in payload
    assert payload["grants_semantic_authority"] is False
    with pytest.raises(ProgramEGraphAuthorityError):
        EqualitySaturationPlan(
            **{
                **{
                    key: value
                    for key, value in payload.items()
                    if key not in {"plan_cid", "schema", "interface", "evidence", "producer_id", "version", "bounds"}
                },
                "bounds": plan.bounds,
                "grants_semantic_authority": True,
            }
        )
