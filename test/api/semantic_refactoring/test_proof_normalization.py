"""Independent contract tests for SPAR-031 proof-normalization adapter."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.proof_normalization import (
    ADAPTER_IS_NOMINATION_ONLY,
    ANALYZER_ID,
    AUTHORITY,
    AUTHORITY_OWNER,
    ABSTRACTION_REFINEMENT_INTERFACE,
    BOUNDARY_SUMMARY_INTERFACE,
    DECLARED_ABSTRACTION_KINDS,
    DECLARED_EXPRESSION_KINDS,
    DECLARED_NORMALIZATION_STATUSES,
    DECLARED_STEP_KINDS,
    DRY_RUN_IS_DETERMINISTIC,
    DRY_RUN_MUTATES,
    DUCKLAKE_IS_AUTHORITY,
    EGRAPH_SATURATION_INTERFACE,
    EQUALITY_REWRITE_INTERFACE,
    FORBIDDEN_NORMALIZATION_NAMES,
    FORMS_REMAIN_UNPROMOTED,
    GENERAL_PYTHON_EQUIVALENCE_CLAIMED,
    GOAL_ID,
    GUESSED_AXIOMS_REJECTED,
    IDENTITY_EXCLUDED_FIELDS,
    IMPLICIT_INSTALL_FORBIDDEN,
    IMPLICIT_NETWORK_FORBIDDEN,
    INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS,
    INTERPOLANT_INTERFACE,
    MARKDOWN_IS_NOT_COMPLETION,
    MODEL_OUTPUT_IS_PROPOSAL_ONLY,
    NETWORK_DENIED,
    NETWORK_DENY,
    NORMALIZATION_FORM_INTERFACE,
    NORMALIZED_FORMS_REENTER_VALIDATION,
    NORMAL_FORM_CANNOT_ADMIT_PROOFS,
    PROGRAM,
    PROOF_NORMALIZATION_CAN_AUTHORIZE_COMPLETION,
    PROOF_NORMALIZATION_CAN_AUTHORIZE_TRANSITION,
    PROOF_NORMALIZATION_CAN_CREATE_AUTHORITY,
    PROOF_NORMALIZATION_CAN_CREATE_PROOF_AUTHORITY,
    PROOF_NORMALIZATION_CONTRACT_VERSION,
    PROOF_NORMALIZATION_RECEIPT_INTERFACE,
    PROJECTION_CLUSTERING_IS_AUTHORITY,
    RAW_COUNTERMODEL_CANNOT_REFUTE,
    RAW_SOURCE_REQUIRED,
    REFACTOR_PROOF_NORMALIZATION_ADAPTER_INTERFACE,
    REWRITE_SELECTOR_DETERMINISTIC,
    SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS,
    SUITABLE_EXPRESSION_INTERFACE,
    TASK_ID,
    TEST_PASS_IS_NOT_COMPLETION,
    TEST_PASS_IS_NOT_PROOF,
    UNKNOWN_REMAINS_UNKNOWN,
    UNVALIDATED_INTERPOLANTS_FAIL_CLOSED,
    VALIDATION_REENTRY_INTERFACE,
    VECTOR_SIMILARITY_IS_AUTHORITY,
    WORKER_SELF_APPROVAL,
    AbstractionKind,
    AbstractionRefinement,
    ExpressionKind,
    FeatureStatus,
    Interpolant,
    NormalizationStatus,
    ProofNormalizationError,
    ProofNormalizationReceipt,
    RefactorProofNormalizationAdapter,
    StepKind,
    SuitableExpression,
    ValidationReentryStatus,
    assert_not_competing_capsule_family,
    compile_abstraction_refinement,
    compile_equality_rewrite,
    compile_interpolant,
    compile_suitable_expression,
    decode_canonical_receipt,
    dry_run_proof_normalization,
    encode_canonical_receipt,
    interpolate_boundary,
    proof_normalization_capabilities,
    proof_normalization_cid_profile,
    proof_normalization_descriptor,
    provider_free_exports,
    refine_abstraction,
    run_proof_normalization,
    saturate_egraph,
)


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "semantic_refactoring"
    / "proof_normalization.py"
)
TEST_PATH = Path(__file__).resolve()
WRITE_SCOPE = (
    "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/proof_normalization.py",
    "test/api/semantic_refactoring/test_proof_normalization.py",
)
PROTECTED_PATHS = (
    ".gitignore",
    "benchmarks/agent_supervisor/semantic_refactoring/preregistration.json",
    "config/agent_supervisor_semantic_preserving_remodularization_scheduler.json",
    "config/semantic_preserving_autonomous_remodularization_dependencies.seal.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_PLAN.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.objectives.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization.todo.md",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/authority_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/benchmark_preregistration.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/dynamic_python_risk_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/identity_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/interface_inventory.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/overlap_gap_matrix.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/repository_baseline.json",
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/rollout_baseline.json",
    "scripts/materialize_semantic_preserving_remodularization_program.py",
    "scripts/ops/agent_supervisor/semantic_preserving_remodularization.py",
    "scripts/validate_semantic_preserving_remodularization_board.py",
    "scripts/validate_semantic_preserving_remodularization_dependencies.py",
    "test/api/semantic_refactoring/test_bootstrap_controls.py",
)
CAPSULE_TYPES = (
    "FunctionSemanticCapsule",
    "MethodSemanticCapsule",
    "ClassSemanticCapsule",
    "TopLevelBlockCapsule",
    "ModuleSemanticCapsule",
    "PackageSemanticCapsule",
    "CallsiteSemanticCapsule",
    "StateOwnerCapsule",
    "RegistrationCapsule",
    "ResourceLifecycleCapsule",
)
TREE_ID = "fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051"
OTHER_TREE = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
WRITE_PATHS = ("pkg/mod.py", "pkg/extracted.py")
VALIDATION = ("python3 -m pytest -q tests/test_mod.py",)
SOURCE_CID_LABEL = "source"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _wave(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "receipt_cid": _cid("wave"),
        "packet_cids": [_cid("packet")],
        "write_paths": list(WRITE_PATHS),
        "worktree_id": _cid("worktree"),
        "status": "applied",
        "writes_repository": False,
        "executor_is_nomination_only": True,
    }
    fields.update(overrides)
    return fields


def _selection(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "tree_id": TREE_ID,
        "validation_selection_cid": _cid("selection"),
        "packet_cid": _cid("packet"),
        "raw_source_cids": [_cid(SOURCE_CID_LABEL)],
        "write_paths": list(WRITE_PATHS),
        "validation_commands": list(VALIDATION),
        "raw_source_required": True,
        "adapter_is_nomination_only": True,
        "datasets_owns_selection": True,
    }
    fields.update(overrides)
    return fields


def _term(label: str) -> str:
    return _cid(f"term:{label}")


def _expression(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "expression_id": "expr:lhs",
        "term_id": _term("lhs"),
        "kind": ExpressionKind.EXPRESSION.value,
        "tree_id": TREE_ID,
        "source_cid": _cid(SOURCE_CID_LABEL),
        "source_authority": "specification",
        "operator": "",
        "arg_term_ids": [],
        "finite": True,
        "required": True,
        "cost": 2,
        "semantics": "",
    }
    fields.update(overrides)
    return fields


def _rewrite(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "rule_id": "rule:commute",
        "lhs_term_id": _term("lhs"),
        "rhs_term_id": _term("rhs"),
        "review_ref": "review:eq-1",
        "justification": "reconstructed_proof",
        "tree_id": TREE_ID,
        "reconstruction_cid": _cid("reconstruction"),
        "oriented": True,
        "cost": 1,
        "theory_id": "theory:eq",
    }
    fields.update(overrides)
    return fields


def _interpolant(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "interpolant_id": "itp:boundary",
        "predicate_id": "pred:shared",
        "tree_id": TREE_ID,
        "a_term_ids": [_term("lhs")],
        "b_term_ids": [_term("rhs")],
        "justification": "reconstructed_proof",
        "justification_cid": _cid("reconstruction"),
        "validated": True,
    }
    fields.update(overrides)
    return fields


def _refinement(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "predicate_id": "pred:state",
        "kind": AbstractionKind.STATE.value,
        "tree_id": TREE_ID,
        "source_cid": _cid("replay"),
        "grain": "refined",
        "distinguishing": True,
        "replayed": True,
    }
    fields.update(overrides)
    return fields


def _validate_ok(**request: Any) -> dict[str, Any]:
    return {
        "status": "validated",
        "result_cid": _cid("validation"),
        "tree_id": request["tree_id"],
        "network": request["network"],
    }


def _validate_reject(**request: Any) -> dict[str, Any]:
    return {
        "status": "rejected",
        "result_cid": _cid("validation-reject"),
        "tree_id": request["tree_id"],
        "reason_code": "translation_rejected",
        "network": request["network"],
    }


def _run(**overrides: Any) -> ProofNormalizationReceipt:
    fields: dict[str, Any] = {
        "wave": _wave(),
        "selection": _selection(),
        "expressions": [
            _expression(),
            _expression(expression_id="expr:rhs", term_id=_term("rhs"), cost=1),
        ],
        "rewrites": [_rewrite()],
        "validate": _validate_ok,
    }
    fields.update(overrides)
    return run_proof_normalization(**fields)


def test_owned_paths_and_task_identity_are_exact() -> None:
    assert TASK_ID == "SPAR-031"
    assert GOAL_ID == "SPAR-G053"
    assert PROGRAM == "semantic-preserving-autonomous-remodularization-v1"
    assert (
        REFACTOR_PROOF_NORMALIZATION_ADAPTER_INTERFACE
        == "RefactorProofNormalizationAdapter@1"
    )
    assert SUITABLE_EXPRESSION_INTERFACE == "SuitableExpression@1"
    assert EQUALITY_REWRITE_INTERFACE == "EqualityRewrite@1"
    assert EGRAPH_SATURATION_INTERFACE == "EGraphSaturation@1"
    assert INTERPOLANT_INTERFACE == "Interpolant@1"
    assert ABSTRACTION_REFINEMENT_INTERFACE == "AbstractionRefinement@1"
    assert BOUNDARY_SUMMARY_INTERFACE == "BoundarySummary@1"
    assert NORMALIZATION_FORM_INTERFACE == "NormalizationForm@1"
    assert PROOF_NORMALIZATION_RECEIPT_INTERFACE == "ProofNormalizationReceipt@1"
    assert VALIDATION_REENTRY_INTERFACE == "ValidationReentry@1"
    assert PROOF_NORMALIZATION_CONTRACT_VERSION == "1"
    assert ANALYZER_ID.endswith("proof_normalization@1")
    assert MODULE_PATH.is_file()
    assert TEST_PATH.is_file()
    for relative in WRITE_SCOPE:
        assert (ROOT / relative).is_file()


def test_authority_flags_cannot_self_authorize() -> None:
    assert AUTHORITY == "proof normalization"
    assert AUTHORITY_OWNER == "ipfs_accelerate_py"
    assert PROOF_NORMALIZATION_CAN_AUTHORIZE_COMPLETION is False
    assert PROOF_NORMALIZATION_CAN_AUTHORIZE_TRANSITION is False
    assert PROOF_NORMALIZATION_CAN_CREATE_AUTHORITY is False
    assert PROOF_NORMALIZATION_CAN_CREATE_PROOF_AUTHORITY is False
    assert VECTOR_SIMILARITY_IS_AUTHORITY is False
    assert PROJECTION_CLUSTERING_IS_AUTHORITY is False
    assert MODEL_OUTPUT_IS_PROPOSAL_ONLY is True
    assert TEST_PASS_IS_NOT_COMPLETION is True
    assert TEST_PASS_IS_NOT_PROOF is True
    assert NORMAL_FORM_CANNOT_ADMIT_PROOFS is True
    assert RAW_COUNTERMODEL_CANNOT_REFUTE is True
    assert UNVALIDATED_INTERPOLANTS_FAIL_CLOSED is True
    assert MARKDOWN_IS_NOT_COMPLETION is True
    assert WORKER_SELF_APPROVAL is False
    assert DUCKLAKE_IS_AUTHORITY is False
    assert SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS is True
    assert ADAPTER_IS_NOMINATION_ONLY is True
    assert RAW_SOURCE_REQUIRED is True
    assert NETWORK_DENIED is True
    assert DRY_RUN_IS_DETERMINISTIC is True
    assert DRY_RUN_MUTATES is False
    assert UNKNOWN_REMAINS_UNKNOWN is True
    assert GUESSED_AXIOMS_REJECTED is True
    assert INCOMPLETE_CONTRACTS_ARE_TYPED_TERMINALS is True
    assert GENERAL_PYTHON_EQUIVALENCE_CLAIMED is False
    assert IMPLICIT_INSTALL_FORBIDDEN is True
    assert IMPLICIT_NETWORK_FORBIDDEN is True
    assert NORMALIZED_FORMS_REENTER_VALIDATION is True
    assert FORMS_REMAIN_UNPROMOTED is True
    assert DECLARED_NORMALIZATION_STATUSES == {
        "normalized",
        "interpolated",
        "refined",
        "unsupported",
        "incomplete",
        "unknown",
        "rejected",
        "timeout",
        "stale",
    }
    assert ExpressionKind.EXPRESSION.value in DECLARED_EXPRESSION_KINDS
    assert StepKind.EQUALITY_SATURATION.value in DECLARED_STEP_KINDS
    assert AbstractionKind.STATE.value in DECLARED_ABSTRACTION_KINDS
    profile = proof_normalization_cid_profile()
    assert profile["codec"] == "dag-json"
    assert "not universal meaning" in profile["rule"]


def test_module_defines_predicted_symbols_not_capsule_family() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "RefactorProofNormalizationAdapter" in names
    assert "SuitableExpression" in names
    assert "EqualityRewrite" in names
    assert "EGraphSaturation" in names
    assert "Interpolant" in names
    assert "AbstractionRefinement" in names
    assert "BoundarySummary" in names
    assert "ProofNormalizationReceipt" in names
    for capsule in CAPSULE_TYPES:
        assert capsule not in names
    assert_not_competing_capsule_family()
    exports = provider_free_exports()
    assert "RefactorProofNormalizationAdapter" in exports
    assert "run_proof_normalization" in exports
    assert "dry_run_proof_normalization" in exports
    assert "saturate_egraph" in exports
    assert "interpolate_boundary" in exports
    assert "refine_abstraction" in exports


def test_protected_paths_are_not_owned_write_scope() -> None:
    owned = set(WRITE_SCOPE)
    for relative in PROTECTED_PATHS:
        assert relative not in owned
        assert (ROOT / relative).exists()


def test_module_does_not_claim_proof_authority_or_open_network() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not (names & FORBIDDEN_NORMALIZATION_NAMES)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("tactician_hammer_coordinator" in name for name in imported)
    assert not any("procedure_compiler" in name for name in imported)
    descriptor = proof_normalization_descriptor()
    assert descriptor["interface"] == REFACTOR_PROOF_NORMALIZATION_ADAPTER_INTERFACE
    assert descriptor["network"] == NETWORK_DENY
    assert descriptor["nomination_only"] is True
    assert descriptor["claims_general_equivalence"] is False
    assert descriptor["normal_form_cannot_admit_proofs"] is True
    assert descriptor["raw_countermodel_cannot_refute"] is True
    assert descriptor["unvalidated_interpolants_fail_closed"] is True
    assert descriptor["unknown_remains_unknown"] is True
    assert descriptor["guessed_axioms_rejected"] is True
    assert descriptor["normalized_forms_reenter_validation"] is True
    assert descriptor["forms_remain_unpromoted"] is True
    assert descriptor["rewrite_selector"] == REWRITE_SELECTOR_DETERMINISTIC
    forbids = set(descriptor["forbids"])
    assert "claim_general_equivalence" in forbids
    assert "admit_vector_proof" in forbids
    assert "open_network" in forbids


def test_equality_saturation_nominates_unpromoted_normal_form() -> None:
    receipt = _run()
    assert receipt.status == NormalizationStatus.NORMALIZED.value
    assert receipt.adapter_is_nomination_only is True
    assert receipt.can_authorize_completion is False
    assert receipt.can_authorize_transition is False
    assert receipt.can_create_proof_authority is False
    assert receipt.network == NETWORK_DENY
    assert receipt.mutated is False
    assert receipt.deterministic is True
    assert receipt.claims_general_equivalence is False
    assert receipt.forms_remain_unpromoted is True
    assert len(receipt.forms) == 1
    assert receipt.forms[0].unpromoted is True
    assert receipt.forms[0].representative_term_id == _term("rhs")
    assert set(receipt.forms[0].member_term_ids) == {_term("lhs"), _term("rhs")}
    assert receipt.reentry is not None
    assert receipt.reentry.status == ValidationReentryStatus.VALIDATED.value
    assert receipt.reentry.can_authorize_completion is False
    assert receipt.typed_terminal is False
    assert receipt.saturation is not None
    assert receipt.saturation.rewrite_cids
    restored = decode_canonical_receipt(encode_canonical_receipt(receipt))
    assert restored == receipt
    adapter = RefactorProofNormalizationAdapter()
    again = adapter.normalize(
        wave=_wave(),
        selection=_selection(),
        expressions=[
            _expression(),
            _expression(expression_id="expr:rhs", term_id=_term("rhs"), cost=1),
        ],
        rewrites=[_rewrite()],
        validate=_validate_ok,
    )
    assert again.receipt_cid == receipt.receipt_cid


def test_dry_run_is_deterministic_and_non_mutating() -> None:
    first = dry_run_proof_normalization(
        wave=_wave(),
        selection=_selection(),
        expressions=[
            _expression(),
            _expression(expression_id="expr:rhs", term_id=_term("rhs"), cost=1),
        ],
        rewrites=[_rewrite()],
        validate=_validate_ok,
    )
    second = dry_run_proof_normalization(
        wave=_wave(),
        selection=_selection(),
        expressions=[
            _expression(),
            _expression(expression_id="expr:rhs", term_id=_term("rhs"), cost=1),
        ],
        rewrites=[_rewrite()],
        validate=_validate_ok,
    )
    assert first.receipt_cid == second.receipt_cid
    assert first.mutated is False
    assert first.deterministic is True
    with pytest.raises(ProofNormalizationError, match="cannot mutate"):
        run_proof_normalization(
            wave=_wave(),
            selection=_selection(),
            expressions=[_expression()],
            mutate=True,
        )


def test_congruence_rebuild_merges_matching_operators() -> None:
    arg_a = _term("arg-a")
    arg_b = _term("arg-b")
    parent_left = _term("parent-left")
    parent_right = _term("parent-right")
    expressions = [
        _expression(expression_id="expr:arg-a", term_id=arg_a, cost=1),
        _expression(expression_id="expr:arg-b", term_id=arg_b, cost=1),
        _expression(
            expression_id="expr:parent-left",
            term_id=parent_left,
            operator="adapter",
            arg_term_ids=[arg_a],
            kind=ExpressionKind.ADAPTER.value,
        ),
        _expression(
            expression_id="expr:parent-right",
            term_id=parent_right,
            operator="adapter",
            arg_term_ids=[arg_b],
            kind=ExpressionKind.ADAPTER.value,
        ),
    ]
    rewrites = [
        _rewrite(
            rule_id="rule:args",
            lhs_term_id=arg_a,
            rhs_term_id=arg_b,
        )
    ]
    saturation = saturate_egraph(
        tree_id=TREE_ID,
        expressions=tuple(compile_suitable_expression(**item) for item in expressions),
        rewrites=tuple(compile_equality_rewrite(**item) for item in rewrites),
    )
    members = {tuple(item.member_term_ids) for item in saturation.classes}
    assert any(parent_left in group and parent_right in group for group in members)


def test_guessed_and_vector_expressions_cannot_become_axioms() -> None:
    with pytest.raises(ProofNormalizationError, match="cannot become axioms"):
        compile_suitable_expression(**_expression(source_authority="guessed"))
    with pytest.raises(ProofNormalizationError, match="cannot become axioms"):
        compile_suitable_expression(**_expression(source_authority="vector_candidate"))
    with pytest.raises(ProofNormalizationError, match="cannot become axioms"):
        compile_suitable_expression(**_expression(source_authority="model_hypothesis"))
    with pytest.raises(ProofNormalizationError, match="cannot admit"):
        _run(vector_evidence={"evidence_class": "heuristic", "admit_proof": True})


def test_unsupported_required_semantics_are_typed_terminals() -> None:
    with pytest.raises(ProofNormalizationError, match="unsupported required"):
        compile_suitable_expression(**_expression(semantics="higher_order"))
    with pytest.raises(ProofNormalizationError, match="unsupported required"):
        compile_suitable_expression(**_expression(finite=False))


def test_reconstructed_proof_justifies_rewrite_and_interpolant() -> None:
    proof_receipt = {
        "tree_id": TREE_ID,
        "reconstructions": [
            {
                "reconstruction_cid": _cid("reconstruction"),
                "kernel_checked": True,
            }
        ],
        "replays": [],
        "steps": [],
    }
    receipt = _run(
        interpolants=[_interpolant()],
        proof_receipt=proof_receipt,
    )
    assert receipt.status == NormalizationStatus.NORMALIZED.value
    assert len(receipt.interpolants) == 1
    assert receipt.interpolants[0].validated is True
    assert any(item.kind == StepKind.INTERPOLATION.value for item in receipt.steps)


def test_unvalidated_interpolants_fail_closed() -> None:
    with pytest.raises(ProofNormalizationError, match="unvalidated interpolants"):
        compile_interpolant(**_interpolant(validated=False))


def test_raw_countermodel_cannot_refine_until_replay() -> None:
    with pytest.raises(ProofNormalizationError, match="cannot refine"):
        compile_abstraction_refinement(**_refinement(replayed=False))
    with pytest.raises(ProofNormalizationError, match="cannot refine"):
        _run(
            differential_receipt={
                "tree_id": TREE_ID,
                "comparisons": [
                    {
                        "mismatch": True,
                        "abstraction_gap": True,
                        "replayed": False,
                        "independently_observed": False,
                        "source_cid": _cid("raw-cm"),
                        "predicate_id": "pred:raw",
                    }
                ],
            }
        )


def test_replayed_differential_mismatch_refines_abstraction() -> None:
    receipt = _run(
        rewrites=[],
        expressions=[
            _expression(
                expression_id="expr:summary",
                kind=ExpressionKind.BOUNDARY_SUMMARY.value,
            )
        ],
        differential_receipt={
            "tree_id": TREE_ID,
            "comparisons": [
                {
                    "mismatch": True,
                    "abstraction_gap": True,
                    "replayed": True,
                    "independently_observed": True,
                    "source_cid": _cid("replay"),
                    "predicate_id": "pred:diff",
                    "kind": AbstractionKind.BOUNDARY.value,
                }
            ],
        },
    )
    assert receipt.status == NormalizationStatus.REFINED.value
    assert len(receipt.refinements) == 1
    assert receipt.refinements[0].replayed is True
    assert receipt.summaries[0].smaller is True
    assert receipt.summaries[0].refinement_cids


def test_boundary_summary_is_smaller_and_content_addressed() -> None:
    receipt = _run(
        expressions=[
            _expression(),
            _expression(expression_id="expr:rhs", term_id=_term("rhs"), cost=1),
            _expression(
                expression_id="expr:summary",
                term_id=_term("summary"),
                kind=ExpressionKind.BOUNDARY_SUMMARY.value,
            ),
        ],
        interpolants=[_interpolant()],
        proof_receipt={
            "tree_id": TREE_ID,
            "reconstructions": [
                {"reconstruction_cid": _cid("reconstruction"), "kernel_checked": True}
            ],
        },
    )
    assert receipt.summaries
    assert all(item.smaller is True for item in receipt.summaries)
    again = decode_canonical_receipt(encode_canonical_receipt(receipt))
    assert again.summaries == receipt.summaries


def test_unknown_remains_unknown_without_justified_work() -> None:
    receipt = _run(expressions=[], rewrites=[], validate=None)
    assert receipt.status == NormalizationStatus.UNKNOWN.value
    assert receipt.unknown_remains_unknown is True
    assert receipt.typed_terminal is True
    assert any(item.kind == StepKind.UNKNOWN.value for item in receipt.steps)


def test_stale_proof_receipt_is_rejected() -> None:
    receipt = _run(
        proof_receipt={
            "tree_id": TREE_ID,
            "reconstructions": [
                {"reconstruction_cid": _cid("reconstruction"), "kernel_checked": True}
            ],
            "steps": [{"kind": "stale"}],
        }
    )
    assert receipt.status == NormalizationStatus.STALE.value
    assert any(item.kind == StepKind.STALE.value for item in receipt.steps)


def test_timeout_is_a_typed_non_conclusive_terminal() -> None:
    extra_rewrites = [
        _rewrite(
            rule_id=f"rule:{index}",
            lhs_term_id=_term(f"a{index}"),
            rhs_term_id=_term(f"b{index}"),
            reconstruction_cid=_cid("reconstruction"),
        )
        for index in range(3)
    ]
    extra_expressions = [
        _expression(expression_id=f"expr:a{index}", term_id=_term(f"a{index}"))
        for index in range(3)
    ] + [
        _expression(expression_id=f"expr:b{index}", term_id=_term(f"b{index}"), cost=1)
        for index in range(3)
    ]
    receipt = _run(
        expressions=extra_expressions,
        rewrites=extra_rewrites,
        max_steps=1,
        validate=None,
    )
    assert receipt.status == NormalizationStatus.TIMEOUT.value
    assert receipt.steps_used == 1
    assert receipt.saturation is not None
    assert receipt.saturation.timed_out is True


def test_missing_predecessors_fail_closed() -> None:
    with pytest.raises(ProofNormalizationError, match="SPAR-025"):
        run_proof_normalization(wave=None, selection=_selection())
    with pytest.raises(ProofNormalizationError, match="SPAR-026"):
        run_proof_normalization(wave=_wave(), selection=None)


def test_tree_mismatch_fails_closed() -> None:
    with pytest.raises(ProofNormalizationError, match="SPAR-026 tree_id"):
        _run(selection=_selection(tree_id=OTHER_TREE))
    with pytest.raises(ProofNormalizationError, match="expression tree_id"):
        _run(expressions=[_expression(tree_id=OTHER_TREE)])


def test_write_paths_must_match_and_stay_exact() -> None:
    with pytest.raises(ProofNormalizationError, match="write_paths"):
        _run(wave=_wave(write_paths=["pkg/mod.py"]))
    with pytest.raises(ProofNormalizationError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=[]))
    with pytest.raises(ProofNormalizationError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=["pkg/*.py"]))
    with pytest.raises(ProofNormalizationError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=["/tmp/pkg/mod.py"]))
    with pytest.raises(ProofNormalizationError, match="unrestricted scope"):
        _run(selection=_selection(write_paths=["pkg/../secret.py"]))


def test_missing_raw_source_is_typed_terminal() -> None:
    with pytest.raises(ProofNormalizationError, match="raw source"):
        _run(selection=_selection(raw_source_cids=[]))
    with pytest.raises(ProofNormalizationError, match="declared raw source"):
        _run(expressions=[_expression(source_cid=_cid("other-source"))])


def test_vectors_cannot_admit_proofs_or_suppress_raw_source() -> None:
    with pytest.raises(ProofNormalizationError, match="raw-source"):
        _run(
            vector_evidence={
                "evidence_class": "vector_candidate",
                "suppress_raw_source": True,
            }
        )
    with pytest.raises(ProofNormalizationError, match="cannot admit"):
        _run(
            vector_evidence={
                "evidence_class": "model_hypothesis",
                "admit_proof": True,
            }
        )


def test_body_free_artifacts_reject_proof_dumps() -> None:
    with pytest.raises(ProofNormalizationError, match="body-free"):
        _run(expressions=[_expression(proof_body="theorem T : True := by trivial")])
    with pytest.raises(ProofNormalizationError, match="body-free"):
        _run(rewrites=[_rewrite(source="axiom true")])


def test_network_is_denied() -> None:
    with pytest.raises(ProofNormalizationError, match="network is denied"):
        _run(network="allow")


def test_validation_rejection_cannot_complete() -> None:
    receipt = _run(validate=_validate_reject)
    assert receipt.status == NormalizationStatus.REJECTED.value
    assert receipt.reentry is not None
    assert receipt.reentry.status == ValidationReentryStatus.REJECTED.value
    assert receipt.can_authorize_completion is False
    assert receipt.forms[0].unpromoted is True
    assert receipt.typed_terminal is True


def test_proof_candidate_rewrite_justification_is_rejected() -> None:
    with pytest.raises(ProofNormalizationError, match="unsupported rewrite justification"):
        compile_equality_rewrite(**_rewrite(justification="proof_candidate"))


def test_stale_reconstruction_binding_fails_closed() -> None:
    with pytest.raises(ProofNormalizationError, match="SPAR-030 reconstruction_cid"):
        _run(
            proof_receipt={
                "tree_id": TREE_ID,
                "reconstructions": [
                    {
                        "reconstruction_cid": _cid("other-reconstruction"),
                        "kernel_checked": True,
                    }
                ],
            }
        )


def test_capabilities_inventory_keeps_unavailable_features_unknown() -> None:
    capabilities = {item.feature: item for item in proof_normalization_capabilities()}
    assert capabilities["typed_egraph"].status == FeatureStatus.AVAILABLE.value
    assert capabilities["equality_saturation"].available is True
    assert capabilities["finite_interpolation"].available is True
    assert capabilities["abstraction_refinement_from_replayed_ce"].available is True
    assert capabilities["general_python_equivalence"].available is False
    assert capabilities["external_egg_runtime"].status == FeatureStatus.UNAVAILABLE.value
    assert capabilities["kernel_equivalence"].available is False


def test_identity_excludes_observational_fields() -> None:
    assert "timestamp" in IDENTITY_EXCLUDED_FIELDS
    assert "model_output" in IDENTITY_EXCLUDED_FIELDS
    with pytest.raises(ProofNormalizationError, match="observational"):
        ProofNormalizationReceipt.from_dict(
            {
                **_run().to_dict(),
                "timestamp": "now",
            }
        )


def test_adapter_methods_remain_nomination_only() -> None:
    adapter = RefactorProofNormalizationAdapter()
    expression = adapter.compile_expression(**_expression())
    rewrite = adapter.compile_rewrite(**_rewrite())
    assert isinstance(expression, SuitableExpression)
    assert expression.source_authority == "specification"
    saturation = adapter.saturate(
        tree_id=TREE_ID,
        expressions=(
            expression,
            compile_suitable_expression(
                **_expression(expression_id="expr:rhs", term_id=_term("rhs"), cost=1)
            ),
        ),
        rewrites=(rewrite,),
    )
    assert saturation.timed_out is False
    interpolant = adapter.interpolate(
        _interpolant(),
        tree_id=TREE_ID,
        known_term_ids=(_term("lhs"), _term("rhs")),
        reconstruction_cids=(_cid("reconstruction"),),
    )
    assert isinstance(interpolant, Interpolant)
    refinement = adapter.refine(_refinement(), tree_id=TREE_ID)
    assert isinstance(refinement, AbstractionRefinement)
    dry = adapter.dry_run(
        wave=_wave(),
        selection=_selection(),
        expressions=[
            _expression(),
            _expression(expression_id="expr:rhs", term_id=_term("rhs"), cost=1),
        ],
        rewrites=[_rewrite()],
        validate=_validate_ok,
    )
    assert dry.adapter_is_nomination_only is True
    assert dry.can_create_authority is False
