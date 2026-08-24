"""Equality-saturation hardening for ProgramRepairSynthesizer@1 (LGCVF-080).

Required evidence: congruence, side-condition, budget, extraction, replay,
and invalid-rewrite coverage. Unavailable features are recorded on the
receipt rather than silently claimed.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import (
    DeclaredEqualityTheory,
    EqualityEGraph,
    EqualityFeatureStatus,
    EqualityRewriteStatus,
    EqualityRule,
    ProgramRepairAuthorityError,
    ProgramRepairBounds,
    ProgramRepairDisposition,
    ProgramRepairMode,
    ProgramRepairReason,
    ProgramRepairRequest,
    ProgramRepairSynthesisError,
    equality_saturation_capabilities,
    extract_under_equality_theory,
    parse_equality_term,
    prove_equality_under_theory,
    replay_equality_rewrites,
    synthesize_program_repair,
)


def roots() -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id="repository:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        overlay_id="overlay:fixture",
        file_root_id="file-root:fixture",
        ast_root_id="ast:fixture",
        graph_id="graph:fixture",
        corpus_id="corpus:fixture",
        index_id="index:fixture",
        model_id="model:fixture",
        cache_id="cache:fixture",
        operator_registry_id="operators:fixture",
        translator_id="translator:fixture",
        solver_id="solver:fixture",
        kernel_id="kernel:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        sandbox_id="sandbox:fixture",
        environment_id="environment:fixture",
        lease_id="lease:fixture",
    )


def arith_theory(**overrides: object) -> DeclaredEqualityTheory:
    values: dict[str, object] = {
        "theory_id": "theory:arith@1",
        "review_refs": ("review:equality_theory@1", "review:equality_rewrite@1"),
        "rules": (
            EqualityRule(
                rule_id="rule:add-zero",
                lhs="(+ x 0)",
                rhs="x",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:arith@1",
            ),
            EqualityRule(
                rule_id="rule:commute-add",
                lhs="(+ a b)",
                rhs="(+ b a)",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:arith@1",
            ),
        ),
        "repository_id": "repository:fixture",
        "tree_id": "tree:fixture",
        "operator_costs": {"+": 2, "*": 2, "x": 1, "y": 1, "0": 1},
    }
    values.update(overrides)
    return DeclaredEqualityTheory(**values)  # type: ignore[arg-type]


def _assert_available(receipt, *features: str) -> None:
    inventory = receipt.capability_map()
    for feature in features:
        assert feature in inventory
        assert inventory[feature].status is EqualityFeatureStatus.AVAILABLE


def _assert_unavailable(receipt, feature: str, *, note_substr: str) -> None:
    item = receipt.capability_map()[feature]
    assert item.status is EqualityFeatureStatus.UNAVAILABLE
    assert note_substr in item.note


def test_typed_eclasses_carry_operator_sorts() -> None:
    theory = arith_theory()
    graph = EqualityEGraph(theory)
    plus = graph.add_term("(+ x 0)")
    zero = graph.add_term("0")
    negated = graph.add_term("(not true)")
    assert graph._sorts[graph._find(plus)] == "Int"
    assert graph._sorts[graph._find(zero)] == "Int"
    assert graph._sorts[graph._find(negated)] == "Bool"
    assert graph._find(plus) != graph._find(negated)


def test_congruence_rebuild_unites_contexts() -> None:
    theory = arith_theory()
    proved = prove_equality_under_theory(theory, "(* (+ x 0) y)", "(* x y)")
    assert proved.proved
    assert proved.congruence_merges >= 1
    assert proved.rebuild_count >= 1
    assert proved.eclass_count >= 1
    assert proved.source_sort == "Int"
    assert proved.target_sort == "Int"
    assert "rule:add-zero" in proved.applied_rule_ids
    _assert_available(proved, "typed_eclasses", "congruence_rebuild")


def test_pattern_variable_rewrite_is_reviewed() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:pat@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:add-zero-var",
                lhs="(+ ?x 0)",
                rhs="?x",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:pat@1",
            ),
        ),
    )
    proved = prove_equality_under_theory(theory, "(+ y 0)", "y")
    assert proved.proved
    assert proved.replay_steps
    assert proved.replay_steps[0].substitution == (("?x", "y"),)
    assert "review:equality_rewrite@1" in proved.applied_review_refs


def test_unreviewed_side_condition_blocks_rewrite() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:unreviewed@1",
        review_refs=("review:equality_theory@1",),
        rules=(
            EqualityRule(
                rule_id="rule:unreviewed",
                lhs="(+ x 0)",
                rhs="x",
                review_ref="review:not-declared@1",
                theory_id="theory:unreviewed@1",
            ),
        ),
    )
    receipt = prove_equality_under_theory(theory, "(+ x 0)", "x")
    assert not receipt.proved
    assert receipt.status is EqualityRewriteStatus.UNPROVED
    assert any("reviewed:failed" in item for item in receipt.side_condition_results)
    _assert_available(receipt, "reviewed_side_conditions", "provenance")


def test_unbound_pattern_variable_is_a_side_condition_failure() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:open@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:open-rhs",
                lhs="(+ ?x 0)",
                rhs="?y",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:open@1",
            ),
        ),
    )
    receipt = prove_equality_under_theory(theory, "(+ x 0)", "z")
    assert not receipt.proved
    assert any("closed_vars:failed" in item for item in receipt.side_condition_results)


def test_unknown_side_condition_is_rejected_at_construction() -> None:
    with pytest.raises(ProgramRepairSynthesisError, match="unknown reviewed side condition"):
        EqualityRule(
            rule_id="rule:bad-sc",
            lhs="a",
            rhs="b",
            review_ref="review:equality_rewrite@1",
            theory_id="theory:bad@1",
            side_conditions=("not_a_real_condition",),
        )


def test_node_budget_exhaustion_is_recorded() -> None:
    theory = arith_theory()
    receipt = prove_equality_under_theory(
        theory, "(+ x 0)", "x", max_nodes=2
    )
    assert not receipt.proved
    assert receipt.status is EqualityRewriteStatus.BUDGET_EXHAUSTED
    assert receipt.reason_code == ProgramRepairReason.BOUNDS_EXCEEDED.value
    assert receipt.egraph_node_count <= 2
    _assert_available(receipt, "bounded_saturation")


def test_rewrite_depth_budget_exhaustion_is_recorded() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:chain@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:a-b",
                lhs="a",
                rhs="b",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:chain@1",
            ),
            EqualityRule(
                rule_id="rule:b-c",
                lhs="b",
                rhs="c",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:chain@1",
            ),
            EqualityRule(
                rule_id="rule:c-d",
                lhs="c",
                rhs="d",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:chain@1",
            ),
        ),
    )
    exhausted = prove_equality_under_theory(theory, "a", "d", max_depth=1)
    assert not exhausted.proved
    assert exhausted.status is EqualityRewriteStatus.BUDGET_EXHAUSTED
    closed = prove_equality_under_theory(theory, "a", "d", max_depth=4)
    assert closed.proved
    assert closed.rewrite_depth >= 1


def test_extraction_prefers_cheaper_representative() -> None:
    theory = arith_theory()
    extracted, cost = extract_under_equality_theory(theory, "(+ x 0)")
    assert extracted == "x"
    assert cost == 1
    proved = prove_equality_under_theory(theory, "(+ x 0)", "x")
    assert proved.proved
    assert proved.extracted_term == "x"
    assert proved.extraction_cost == 1
    assert proved.extraction_cost < 3
    _assert_available(proved, "extraction_cost")


def test_replay_reconstructs_extracted_target() -> None:
    theory = arith_theory()
    proved = prove_equality_under_theory(theory, "(* (+ x 0) y)", "(* x y)")
    assert proved.proved
    assert proved.replay_steps
    replayed = replay_equality_rewrites(
        proved.source_term, proved.replay_steps, theory
    )
    assert replayed == proved.target_term
    assert proved.independent_equivalence.startswith("passed")
    _assert_available(proved, "extraction_replay", "independent_equivalence_check")


def test_invalid_sort_mismatch_is_rejected() -> None:
    theory = arith_theory()
    receipt = prove_equality_under_theory(theory, "(not true)", "0")
    assert not receipt.proved
    assert receipt.status is EqualityRewriteStatus.INVALID
    assert receipt.reason_code == ProgramRepairReason.EQUALITY_TYPE_MISMATCH.value
    assert receipt.source_sort == "Bool"
    assert receipt.target_sort == "Int"


def test_invalid_effect_changing_rewrite_is_rejected() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:effects@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:write-intro",
                lhs="x",
                rhs="(write x)",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:effects@1",
            ),
        ),
        operator_effects={"write": ("file_write",)},
    )
    receipt = prove_equality_under_theory(theory, "x", "(write x)")
    assert not receipt.proved
    assert receipt.status is EqualityRewriteStatus.INVALID
    assert receipt.reason_code == ProgramRepairReason.EQUALITY_EFFECT_CHANGE.value
    assert "file_write" in receipt.independent_effect
    _assert_available(receipt, "independent_effect_check")


def test_effectful_rule_fails_reviewed_side_condition() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:pure@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:impure",
                lhs="x",
                rhs="(write x)",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:pure@1",
                side_conditions=("pure",),
                effects=("file_write",),
            ),
        ),
        operator_effects={"write": ("file_write",)},
    )
    # Source and target already differ in effects, so the independent
    # effect gate rejects before saturation. The same theory also refuses
    # the impure rule when the terms themselves are effect-neutral.
    direct = prove_equality_under_theory(theory, "x", "(write x)")
    assert direct.status is EqualityRewriteStatus.INVALID
    same = prove_equality_under_theory(theory, "x", "y")
    assert not same.proved
    assert any("pure:failed" in item or "no_undeclared_effects:failed" in item
               for item in same.side_condition_results)


def test_authority_claiming_theory_is_rejected() -> None:
    with pytest.raises(ProgramRepairAuthorityError):
        DeclaredEqualityTheory(
            theory_id="theory:auth@1",
            review_refs=("review:equality_theory@1",),
            rules=(
                EqualityRule(
                    rule_id="rule:x",
                    lhs="a",
                    rhs="b",
                    review_ref="review:equality_rewrite@1",
                    theory_id="theory:auth@1",
                ),
            ),
            grants_semantic_authority=True,
        )


def test_receipt_records_unavailable_features_truthfully() -> None:
    inventory = {item.feature: item for item in equality_saturation_capabilities()}
    for feature in (
        "typed_eclasses",
        "congruence_rebuild",
        "reviewed_side_conditions",
        "provenance",
        "bounded_saturation",
        "extraction_cost",
        "extraction_replay",
        "independent_equivalence_check",
        "independent_effect_check",
    ):
        assert inventory[feature].status is EqualityFeatureStatus.AVAILABLE
    assert inventory["smt_semantic_side_conditions"].status is EqualityFeatureStatus.UNAVAILABLE
    assert "no_solver" in inventory["smt_semantic_side_conditions"].note
    assert inventory["kernel_checked_equivalence"].status is EqualityFeatureStatus.UNAVAILABLE
    assert inventory["external_egg_runtime"].status is EqualityFeatureStatus.UNAVAILABLE
    proved = prove_equality_under_theory(arith_theory(), "(+ x 0)", "x")
    _assert_unavailable(proved, "smt_semantic_side_conditions", note_substr="no_solver")
    _assert_unavailable(proved, "kernel_checked_equivalence", note_substr="no_kernel")
    _assert_unavailable(proved, "external_egg_runtime", note_substr="egg")


def test_synthesizer_equality_mode_uses_hardened_path() -> None:
    theory = arith_theory()
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:eq",),
        target_paths=("pkg/expr.py",),
        mode=ProgramRepairMode.EQUALITY_REWRITE,
        equality_theory=theory,
        source_term="(* (+ x 0) y)",
        target_term="(* x y)",
        postcondition_refs=("post:equivalent_under_declared_theory",),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.SUPPORTED
    assert receipt.equality_receipt is not None
    assert receipt.equality_receipt.proved
    assert receipt.equality_receipt.congruence_merges >= 1
    assert receipt.equality_receipt.extracted_term == "(* x y)"
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.proposal_only is True
    assert receipt.selected_candidate.write_authority is False
    assert receipt.deterministic_zero_model_calls is True
    assert receipt.llm_invocation_count == 0
    replayed = replay_equality_rewrites(
        request.source_term,
        receipt.equality_receipt.replay_steps,
        theory,
    )
    assert replayed == request.target_term


def test_synthesizer_abstains_on_invalid_rewrite() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:effects@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:write-intro",
                lhs="x",
                rhs="(write x)",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:effects@1",
            ),
        ),
        operator_effects={"write": ("file_write",)},
        repository_id="repository:fixture",
        tree_id="tree:fixture",
    )
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:eq",),
        target_paths=("pkg/expr.py",),
        mode=ProgramRepairMode.EQUALITY_REWRITE,
        equality_theory=theory,
        source_term="x",
        target_term="(write x)",
        bounds=ProgramRepairBounds(max_egraph_nodes=32),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.ABSTAIN
    assert receipt.equality_receipt is not None
    assert receipt.equality_receipt.status is EqualityRewriteStatus.INVALID
    assert ProgramRepairReason.EQUALITY_EFFECT_CHANGE.value in receipt.reason_codes


def test_synthesizer_reports_budget_exhausted() -> None:
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:eq",),
        target_paths=("pkg/expr.py",),
        mode=ProgramRepairMode.EQUALITY_REWRITE,
        equality_theory=arith_theory(),
        source_term="(+ x 0)",
        target_term="x",
        bounds=ProgramRepairBounds(max_egraph_nodes=2),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.BUDGET_EXHAUSTED
    assert receipt.equality_receipt is not None
    assert receipt.equality_receipt.status is EqualityRewriteStatus.BUDGET_EXHAUSTED


def test_existing_concrete_add_zero_still_proves() -> None:
    proved = prove_equality_under_theory(arith_theory(), "(+ x 0)", "x")
    assert proved.proved
    assert proved.proposal_only is True
    assert proved.grants_write_authority is False
    assert proved.grants_semantic_authority is False
    assert "rule:add-zero" in proved.applied_rule_ids
    assert parse_equality_term("(+  x  0)").render() == "(+ x 0)"


def test_unrelated_terms_remain_unproved() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:narrow@1",
        review_refs=("review:equality_theory@1",),
        rules=(
            EqualityRule(
                rule_id="rule:only-a",
                lhs="aaa",
                rhs="bbb",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:narrow@1",
            ),
        ),
    )
    receipt = prove_equality_under_theory(theory, "unrelated", "other")
    assert not receipt.proved
    assert receipt.status.value in {"unproved", "budget_exhausted"}


def test_duplicate_redexes_replay_after_congruence() -> None:
    theory = arith_theory()
    proved = prove_equality_under_theory(
        theory, "(* (+ x 0) (+ x 0))", "(* x x)"
    )
    assert proved.proved
    assert proved.congruence_merges >= 1
    replayed = replay_equality_rewrites(
        proved.source_term, proved.replay_steps, theory
    )
    assert replayed == "(* x x)"
    assert proved.independent_equivalence.startswith("passed")
    _assert_available(proved, "congruence_rebuild", "extraction_replay")


def test_all_enodes_in_eclass_are_ematched() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:two-enode@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:plus-eq",
                lhs="(+ y 0)",
                rhs="(+ z 0)",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:two-enode@1",
            ),
            EqualityRule(
                rule_id="rule:add-zero-var",
                lhs="(+ ?x 0)",
                rhs="?x",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:two-enode@1",
            ),
        ),
        operator_costs={"+": 2, "y": 1, "z": 1, "0": 1},
    )
    # Target `z` is not present as a plus-redex until plus-eq inserts `(+ z 0)`
    # into the same e-class as `(+ y 0)`. Proving `(+ y 0) ≡ z` therefore
    # requires ematching every plus-node in that class, not just the first.
    proved = prove_equality_under_theory(theory, "(+ y 0)", "z")
    assert proved.proved
    assert "rule:add-zero-var" in proved.applied_rule_ids
    assert "rule:plus-eq" in proved.applied_rule_ids
    replayed = replay_equality_rewrites(
        proved.source_term, proved.replay_steps, theory
    )
    assert replayed == "z"
    _assert_available(proved, "typed_eclasses", "congruence_rebuild", "extraction_replay")
