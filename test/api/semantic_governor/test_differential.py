"""Tests for SCG-027 semantic differential comparison beyond text.

Acceptance criteria enforced here:

* Equivalent valid patches classify as ``equivalent_success``.
* ``compressed_failed_expanded_succeeded`` is a distinct comparative outcome.
* Inconclusive verification stays ``verification_inconclusive``.
* Text difference alone cannot classify failure; structural/verification
  evidence is required for semantic nonequivalence.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    AttemptTerminalStatus,
    ComparativeOutcome,
    CostTimingProjection,
    OutcomeClassificationBasis,
    PairedAttemptRecord,
    SemanticEditClass,
    ShadowAttemptRole,
    ShadowExecutionPlan,
    ShadowExecutionResult,
    ShadowSelectionReason,
    VerificationProjection,
    non_text_classification_bases,
    verify_comparison_identity,
    verify_report_identity,
    verify_result_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.differential import (
    COMPARE_SHADOW_RESULTS_INTERFACE,
    SCG_DIFFERENTIAL_EVIDENCE,
    SEMANTIC_DIFFERENTIAL_OUTCOME_INTERFACE,
    AttemptStructuralProjection,
    SemanticDifferentialOutcome,
    SemanticGovernorDifferentialError,
    StructuralComparisonEvidence,
    classify_comparative_outcome,
    compare_shadow_results,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/differential.py"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _header(artifact_kind: str, **overrides: Any) -> GovernorArtifactHeader:
    compressed = _cid("context-pack-compressed")
    fields: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": compressed,
        "verification_bundle_cid": _cid("verification-bundle"),
        "generator": GeneratorIdentity(
            generator_id="shadow_execution",
            generator_version="1.0.0",
            interface_id="create_shadow_plan@1",
        ),
        "provenance": ArtifactProvenance(
            producer_id="semantic_governor",
            producer_version="1",
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=(_cid("input-a"),),
            tool_ids=("shadow.v1",),
            policy_cid=_cid("policy"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="isolated_worktree",
                kind=AssumptionKind.ENVIRONMENT,
                statement="Paired shadow runs use disposable evaluation worktrees",
                supporting_cids=(_cid("worktree-policy"),),
            ),
        ),
        "metadata": {"task": "SCG-027"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)


def _cost(**overrides: Any) -> CostTimingProjection:
    fields: dict[str, Any] = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "wall_time_ms": 1500,
        "model_spend_micros": 25000,
        "verification_time_ms": 300,
    }
    fields.update(overrides)
    return CostTimingProjection(**fields)


def _verification(**overrides: Any) -> VerificationProjection:
    fields: dict[str, Any] = {
        "verification_bundle_cid": _cid("verification-bundle"),
        "selected_tests_passed": True,
        "full_suite_passed": True,
        "proofs_passed": True,
        "static_checks_passed": True,
        "counterexample_present": False,
        "acceptance_matrix_satisfied": True,
        "production_eligible": False,
    }
    fields.update(overrides)
    return VerificationProjection(**fields)


def _attempt(
    role: str = ShadowAttemptRole.COMPRESSED.value,
    **overrides: Any,
) -> PairedAttemptRecord:
    defaults: dict[str, Any] = {
        "role": role,
        "execution_mode": ExecutionMode.LIVE,
        "context_pack_cid": (
            _cid("context-pack-compressed")
            if role == ShadowAttemptRole.COMPRESSED.value
            else _cid("context-pack-expanded")
        ),
        "route_id": "route.default",
        "attempt_status": AttemptTerminalStatus.SUCCEEDED,
        "acceptance_disposition": (
            AcceptanceDisposition.CANDIDATE_ONLY
            if role == ShadowAttemptRole.EXPANDED.value
            else AcceptanceDisposition.NOT_ACCEPTED
        ),
        "cost_timing": _cost(),
        "verification": _verification(),
        "patch_cid": _cid(f"patch-{role}"),
        "worktree_id": f"worktree-{role}",
        "failure_reason_codes": (),
        "notes": None,
    }
    defaults.update(overrides)
    return PairedAttemptRecord(**defaults)


def _plan(**overrides: Any) -> ShadowExecutionPlan:
    compressed = _cid("context-pack-compressed")
    fields: dict[str, Any] = {
        "header": _header("shadow_execution_plan", context_pack_cid=compressed),
        "task_id": "SCG-027",
        "audit_policy_cid": _cid("audit-policy"),
        "compressed_context_pack_cid": compressed,
        "expanded_context_pack_cid": _cid("context-pack-expanded"),
        "compressed_route_id": "route.compressed",
        "expanded_route_id": "route.expanded",
        "selection_reasons": (ShadowSelectionReason.RISK_CLASS_MANDATORY.value,),
        "max_wall_time_ms": 120_000,
        "max_model_spend_micros": 5_000_000,
        "max_expansion_token_budget": 50_000,
        "isolated_evaluation_worktree_required": True,
        "expanded_is_oracle_candidate_only": True,
        "allow_external_expanded_disclosure": False,
        "metadata": {"evidence": SCG_DIFFERENTIAL_EVIDENCE},
    }
    fields.update(overrides)
    return ShadowExecutionPlan(**fields)


def _shadow_result(
    *,
    compressed: PairedAttemptRecord | None = None,
    expanded: PairedAttemptRecord | None = None,
    **overrides: Any,
) -> ShadowExecutionResult:
    plan = _plan()
    fields: dict[str, Any] = {
        "header": _header("shadow_execution_result"),
        "plan_cid": plan.plan_cid,
        "compressed_attempt": compressed
        or _attempt(ShadowAttemptRole.COMPRESSED.value),
        "expanded_attempt": expanded
        if expanded is not None
        else _attempt(ShadowAttemptRole.EXPANDED.value),
        "both_attempts_isolated": True,
        "expanded_skipped_reason": None,
        "metadata": {},
    }
    fields.update(overrides)
    result = ShadowExecutionResult(**fields)
    verify_result_identity(result)
    return result


def _equivalent_structural(
    *,
    text_differs: bool = True,
) -> StructuralComparisonEvidence:
    """Structural evidence for two valid, semantically equivalent patches."""

    text_c = "text-digest-compressed-reformat"
    text_e = "text-digest-expanded-reformat" if text_differs else text_c
    shared_files = ("src/module.py",)
    shared_symbols = ("module.fn",)
    return StructuralComparisonEvidence(
        compressed=AttemptStructuralProjection(
            text_digest=text_c,
            file_ids=shared_files,
            symbol_ids=shared_symbols,
            interface_ids=("module.fn:signature",),
            side_effect_ids=(),
            exception_contracts=("ValueError",),
            schema_ids=("schema.v1",),
            ast_edit_classes=(SemanticEditClass.EQUIVALENT_REFORMAT.value,),
        ),
        expanded=AttemptStructuralProjection(
            text_digest=text_e,
            file_ids=shared_files,
            symbol_ids=shared_symbols,
            interface_ids=("module.fn:signature",),
            side_effect_ids=(),
            exception_contracts=("ValueError",),
            schema_ids=("schema.v1",),
            ast_edit_classes=(SemanticEditClass.EQUIVALENT_REFORMAT.value,),
        ),
        pairwise_ast_edit_classes=(SemanticEditClass.EQUIVALENT_REFORMAT.value,),
    )


def _divergent_structural() -> StructuralComparisonEvidence:
    return StructuralComparisonEvidence(
        compressed=AttemptStructuralProjection(
            text_digest="text-a",
            file_ids=("src/module.py",),
            symbol_ids=("module.fn",),
            interface_ids=("module.fn:signature-v1",),
            side_effect_ids=("io.write",),
            exception_contracts=("ValueError",),
            schema_ids=("schema.v1",),
            ast_edit_classes=(SemanticEditClass.MODIFY_LOGIC.value,),
        ),
        expanded=AttemptStructuralProjection(
            text_digest="text-b",
            file_ids=("src/module.py", "src/extra.py"),
            symbol_ids=("module.fn", "module.helper"),
            interface_ids=("module.fn:signature-v2",),
            side_effect_ids=(),
            exception_contracts=("ValueError", "KeyError"),
            schema_ids=("schema.v2",),
            ast_edit_classes=(SemanticEditClass.INTERFACE_CHANGE.value,),
        ),
        pairwise_ast_edit_classes=(
            SemanticEditClass.MODIFY_LOGIC.value,
            SemanticEditClass.INTERFACE_CHANGE.value,
            SemanticEditClass.ADD.value,
        ),
    )


# ---------------------------------------------------------------------------
# Module surface / evidence / import safety
# ---------------------------------------------------------------------------


def test_evidence_and_interfaces_are_stable() -> None:
    assert SCG_DIFFERENTIAL_EVIDENCE == "scg/differential@1"
    assert COMPARE_SHADOW_RESULTS_INTERFACE == "compare_shadow_results@1"
    assert SEMANTIC_DIFFERENTIAL_OUTCOME_INTERFACE == "SemanticDifferentialOutcome@1"


def test_module_import_performs_no_io() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"open", "urlopen", "system", "Popen", "connect", "create_connection"}
    for node in tree.body:
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else "")
                )
                assert name not in forbidden


# ---------------------------------------------------------------------------
# Acceptance: equivalent valid patches classify equivalent
# ---------------------------------------------------------------------------


def test_equivalent_valid_patches_classify_equivalent() -> None:
    """Both succeeded, structural equivalence-preserving edits → equivalent_success."""

    shared_patch = _cid("patch-shared-equivalent")
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        patch_cid=shared_patch,
        cost_timing=_cost(input_tokens=800, model_spend_micros=20000),
        verification=_verification(production_eligible=False),
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        patch_cid=shared_patch,
        cost_timing=_cost(input_tokens=4000, model_spend_micros=80000),
        verification=_verification(production_eligible=False),
    )
    result = _shadow_result(compressed=compressed, expanded=expanded)
    structural = _equivalent_structural(text_differs=False)

    outcome = compare_shadow_results(
        shadow_result=result,
        structural_evidence=structural,
    )

    assert isinstance(outcome, SemanticDifferentialOutcome)
    assert outcome.comparative_outcome == ComparativeOutcome.EQUIVALENT_SUCCESS.value
    assert outcome.semantic_equivalent is True
    assert outcome.failure_classified is False
    assert outcome.report.textual_difference_is_not_semantic_failure is True
    assert (
        OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value
        in outcome.comparison.classification_bases
    )
    assert (
        OutcomeClassificationBasis.AST_EDIT_CLASSES.value
        in outcome.comparison.classification_bases
    )
    verify_report_identity(outcome.report)
    verify_comparison_identity(outcome.comparison)


def test_text_differs_but_structurally_equivalent_is_still_equivalent() -> None:
    """Textual reformat alone must not prevent equivalent_success."""

    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        patch_cid=_cid("patch-compressed-reformat"),
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        patch_cid=_cid("patch-expanded-reformat"),
    )
    result = _shadow_result(compressed=compressed, expanded=expanded)
    structural = _equivalent_structural(text_differs=True)

    outcome = compare_shadow_results(
        result,
        structural_evidence=structural,
    )

    assert outcome.report.text_differs is True
    assert outcome.report.files_differ is False
    assert outcome.report.symbols_differ is False
    assert outcome.report.interfaces_differ is False
    assert outcome.semantic_equivalent is True
    assert outcome.comparative_outcome == ComparativeOutcome.EQUIVALENT_SUCCESS.value
    assert outcome.failure_classified is False
    # Text may be recorded, but failure is not classified from it.
    assert OutcomeClassificationBasis.TEXT_DIFF.value in (
        outcome.report.classification_bases
    )


def test_identical_patch_cids_without_structural_evidence_are_equivalent() -> None:
    shared = _cid("identical-patch")
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value, patch_cid=shared
    )
    expanded = _attempt(ShadowAttemptRole.EXPANDED.value, patch_cid=shared)
    result = _shadow_result(compressed=compressed, expanded=expanded)

    outcome = compare_shadow_results(shadow_result=result)

    assert outcome.semantic_equivalent is True
    assert outcome.comparative_outcome == ComparativeOutcome.EQUIVALENT_SUCCESS.value
    assert outcome.report.text_differs is False


# ---------------------------------------------------------------------------
# Acceptance: compressed-failed / expanded-succeeded is distinct
# ---------------------------------------------------------------------------


def test_compressed_failed_expanded_succeeded_is_distinct() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.FAILED,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        failure_reason_codes=("selected_tests_failed", "omission_suspected"),
        verification=_verification(
            selected_tests_passed=False,
            full_suite_passed=False,
            acceptance_matrix_satisfied=False,
            counterexample_present=True,
            production_eligible=False,
        ),
        patch_cid=_cid("patch-compressed-bad"),
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
        acceptance_disposition=AcceptanceDisposition.CANDIDATE_ONLY,
        verification=_verification(
            selected_tests_passed=True,
            full_suite_passed=True,
            acceptance_matrix_satisfied=True,
            counterexample_present=False,
            production_eligible=False,
        ),
        patch_cid=_cid("patch-expanded-good"),
    )
    result = _shadow_result(compressed=compressed, expanded=expanded)

    outcome = compare_shadow_results(
        shadow_result=result,
        structural_evidence=_divergent_structural(),
    )

    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
    )
    assert outcome.failure_classified is True
    assert outcome.semantic_equivalent is False
    assert non_text_classification_bases(outcome.report.classification_bases)
    # Must not collapse into both_failed or equivalent.
    assert outcome.comparative_outcome != ComparativeOutcome.EQUIVALENT_SUCCESS.value
    assert outcome.comparative_outcome != (
        ComparativeOutcome.BOTH_FAILED_SAME_REASON.value
    )
    assert outcome.comparative_outcome != (
        ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value
    )
    assert outcome.comparative_outcome != (
        ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value
    )


def test_compressed_succeeded_expanded_failed_is_distinct() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
        verification=_verification(acceptance_matrix_satisfied=True),
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.FAILED,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        failure_reason_codes=("proof_failed",),
        verification=_verification(
            proofs_passed=False,
            acceptance_matrix_satisfied=False,
            counterexample_present=True,
            production_eligible=False,
        ),
    )
    result = _shadow_result(compressed=compressed, expanded=expanded)
    outcome = compare_shadow_results(shadow_result=result)

    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.COMPRESSED_SUCCEEDED_EXPANDED_FAILED.value
    )
    assert outcome.failure_classified is True
    assert non_text_classification_bases(outcome.comparison.classification_bases)


def test_both_failed_same_vs_different_reason() -> None:
    compressed_same = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.FAILED,
        failure_reason_codes=("timeout", "resource"),
        verification=_verification(
            selected_tests_passed=False,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
    )
    expanded_same = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.FAILED,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        failure_reason_codes=("resource", "timeout"),
        verification=_verification(
            selected_tests_passed=False,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
    )
    same = compare_shadow_results(
        shadow_result=_shadow_result(
            compressed=compressed_same, expanded=expanded_same
        )
    )
    assert (
        same.comparative_outcome
        == ComparativeOutcome.BOTH_FAILED_SAME_REASON.value
    )

    expanded_diff = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.FAILED,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        failure_reason_codes=("proof_failed",),
        verification=_verification(
            proofs_passed=False,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
    )
    different = compare_shadow_results(
        shadow_result=_shadow_result(
            compressed=compressed_same, expanded=expanded_diff
        )
    )
    assert (
        different.comparative_outcome
        == ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value
    )


# ---------------------------------------------------------------------------
# Acceptance: inconclusive verification stays inconclusive
# ---------------------------------------------------------------------------


def test_inconclusive_attempt_status_stays_inconclusive() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.INCONCLUSIVE,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        verification=_verification(
            selected_tests_passed=None,
            full_suite_passed=None,
            proofs_passed=None,
            static_checks_passed=None,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
        patch_cid=None,
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
        verification=_verification(acceptance_matrix_satisfied=True),
    )
    result = _shadow_result(compressed=compressed, expanded=expanded)
    outcome = compare_shadow_results(shadow_result=result)

    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value
    )
    assert outcome.failure_classified is False
    # Must not be upgraded to compressed_failed_expanded_succeeded.
    assert outcome.comparative_outcome != (
        ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
    )
    assert outcome.comparative_outcome != ComparativeOutcome.EQUIVALENT_SUCCESS.value


def test_evaluation_failed_stays_inconclusive() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.EVALUATION_FAILED,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        verification=_verification(
            selected_tests_passed=None,
            full_suite_passed=None,
            proofs_passed=None,
            static_checks_passed=None,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
        patch_cid=None,
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.EVALUATION_FAILED,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        verification=_verification(
            selected_tests_passed=None,
            full_suite_passed=None,
            proofs_passed=None,
            static_checks_passed=None,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
        patch_cid=None,
    )
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded)
    )
    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value
    )


def test_incomplete_verification_flags_stay_inconclusive() -> None:
    """Succeeded status with incomplete matrix evidence stays inconclusive."""

    incomplete = _verification(
        selected_tests_passed=True,
        full_suite_passed=None,
        proofs_passed=None,
        static_checks_passed=None,
        acceptance_matrix_satisfied=False,
        production_eligible=False,
    )
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
        verification=incomplete,
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
        verification=incomplete,
    )
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded)
    )
    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.VERIFICATION_INCONCLUSIVE.value
    )


# ---------------------------------------------------------------------------
# Text alone cannot classify failure / nonequivalence without non-text bases
# ---------------------------------------------------------------------------


def test_text_difference_alone_does_not_classify_failure() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        patch_cid=_cid("patch-text-a"),
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        patch_cid=_cid("patch-text-b"),
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
    )
    # No structural evidence — only patch CIDs differ (text observation).
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded)
    )

    assert outcome.report.text_differs is True
    assert outcome.failure_classified is False
    # Without structural proof, cannot claim semantic_equivalent=false from text.
    assert outcome.semantic_equivalent is not False or non_text_classification_bases(
        outcome.report.classification_bases
    )
    # Must not be a failure-like comparative outcome driven by text alone.
    assert outcome.comparative_outcome not in {
        ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value,
        ComparativeOutcome.COMPRESSED_SUCCEEDED_EXPANDED_FAILED.value,
        ComparativeOutcome.BOTH_FAILED_SAME_REASON.value,
        ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value,
    }


def test_structural_divergence_classifies_both_valid_different() -> None:
    compressed = _attempt(ShadowAttemptRole.COMPRESSED.value)
    expanded = _attempt(ShadowAttemptRole.EXPANDED.value)
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded),
        structural_evidence=_divergent_structural(),
    )
    assert outcome.semantic_equivalent is False
    assert outcome.comparative_outcome == (
        ComparativeOutcome.BOTH_VALID_DIFFERENT.value
    )
    assert outcome.report.interfaces_differ is True
    assert outcome.report.files_differ is True
    assert OutcomeClassificationBasis.INTERFACE_DIFF.value in (
        outcome.report.classification_bases
    )


# ---------------------------------------------------------------------------
# Expanded never accepted / missing expanded / human review
# ---------------------------------------------------------------------------


def test_expanded_acceptance_never_accepted() -> None:
    compressed = _attempt(ShadowAttemptRole.COMPRESSED.value)
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        acceptance_disposition=AcceptanceDisposition.CANDIDATE_ONLY,
    )
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded),
        structural_evidence=_equivalent_structural(text_differs=False),
    )
    assert outcome.comparison.expanded_acceptance != (
        AcceptanceDisposition.ACCEPTED.value
    )


def test_missing_expanded_requires_human_review() -> None:
    plan = _plan()
    compressed = _attempt(ShadowAttemptRole.COMPRESSED.value)
    result = ShadowExecutionResult(
        header=_header("shadow_execution_result"),
        plan_cid=plan.plan_cid,
        compressed_attempt=compressed,
        expanded_attempt=None,
        both_attempts_isolated=False,
        expanded_skipped_reason=ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value,
        metadata={},
    )
    outcome = compare_shadow_results(shadow_result=result)
    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value
    )
    assert outcome.comparison.human_review_required is True


def test_human_review_disposition_forces_review_outcome() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        acceptance_disposition=AcceptanceDisposition.HUMAN_REVIEW_REQUIRED,
    )
    expanded = _attempt(ShadowAttemptRole.EXPANDED.value)
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded)
    )
    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.HUMAN_REVIEW_REQUIRED.value
    )


# ---------------------------------------------------------------------------
# Relative quality (expanded/compressed better)
# ---------------------------------------------------------------------------


def test_expanded_better_when_verification_strictly_superior() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
        verification=_verification(
            selected_tests_passed=True,
            full_suite_passed=False,
            proofs_passed=False,
            static_checks_passed=True,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
        verification=_verification(
            selected_tests_passed=True,
            full_suite_passed=True,
            proofs_passed=True,
            static_checks_passed=True,
            acceptance_matrix_satisfied=True,
            production_eligible=False,
        ),
    )
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded)
    )
    assert outcome.comparative_outcome == ComparativeOutcome.EXPANDED_BETTER.value
    assert outcome.failure_classified is True
    assert outcome.report.tests_differ is True or outcome.report.proofs_differ is True


# ---------------------------------------------------------------------------
# API surface: plan-style args, mapping round-trip, classify helper
# ---------------------------------------------------------------------------


def test_compare_with_explicit_attempts_and_cids() -> None:
    compressed = _attempt(ShadowAttemptRole.COMPRESSED.value, patch_cid=_cid("p1"))
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value, patch_cid=_cid("p1")
    )
    plan = _plan()
    # Build a real shadow result CID by constructing a result, then compare
    # via the plan-style positional API.
    shadow = _shadow_result(compressed=compressed, expanded=expanded)
    outcome = compare_shadow_results(
        compressed,
        expanded,
        plan_cid=plan.plan_cid,
        shadow_result_cid=shadow.result_cid,
        structural_evidence=_equivalent_structural(text_differs=False),
        header_seed=shadow.header,
    )
    assert outcome.comparative_outcome == ComparativeOutcome.EQUIVALENT_SUCCESS.value
    assert outcome.report.plan_cid == plan.plan_cid
    assert outcome.report.shadow_result_cid == shadow.result_cid


def test_outcome_to_dict_round_trip_fields() -> None:
    result = _shadow_result()
    outcome = compare_shadow_results(
        shadow_result=result,
        structural_evidence=_equivalent_structural(text_differs=False),
    )
    payload = outcome.to_dict()
    assert payload["evidence"] == SCG_DIFFERENTIAL_EVIDENCE
    assert payload["comparative_outcome"] == outcome.comparative_outcome
    assert payload["report"]["report_cid"] == outcome.report.report_cid
    assert payload["comparison"]["comparison_cid"] == outcome.comparison.comparison_cid
    assert "outcome_cid" in payload


def test_structural_evidence_from_dict_round_trip() -> None:
    original = _equivalent_structural(text_differs=True)
    restored = StructuralComparisonEvidence.from_dict(original.to_dict())
    assert restored.compressed.text_digest == original.compressed.text_digest
    assert list(restored.pairwise_ast_edit_classes) == list(
        original.pairwise_ast_edit_classes
    )


def test_classify_comparative_outcome_helper() -> None:
    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.FAILED,
        failure_reason_codes=("selected_tests_failed",),
        verification=_verification(
            selected_tests_passed=False,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
    )
    expanded = _attempt(
        ShadowAttemptRole.EXPANDED.value,
        attempt_status=AttemptTerminalStatus.SUCCEEDED,
    )
    classified = classify_comparative_outcome(compressed, expanded)
    assert classified == (
        ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
    )


def test_rejects_role_mismatch() -> None:
    with pytest.raises(SemanticGovernorDifferentialError, match="role"):
        compare_shadow_results(
            _attempt(ShadowAttemptRole.EXPANDED.value),
            _attempt(ShadowAttemptRole.EXPANDED.value),
            plan_cid=_cid("plan"),
            shadow_result_cid=_cid("result"),
        )


def test_rejects_missing_plan_cid_without_shadow_result() -> None:
    with pytest.raises(SemanticGovernorDifferentialError, match="plan_cid"):
        compare_shadow_results(
            _attempt(ShadowAttemptRole.COMPRESSED.value),
            _attempt(ShadowAttemptRole.EXPANDED.value),
            shadow_result_cid=_cid("result"),
        )


def test_determinism_same_inputs_same_cids() -> None:
    result = _shadow_result()
    structural = _equivalent_structural(text_differs=False)
    left = compare_shadow_results(shadow_result=result, structural_evidence=structural)
    right = compare_shadow_results(shadow_result=result, structural_evidence=structural)
    assert left.report.report_cid == right.report.report_cid
    assert left.comparison.comparison_cid == right.comparison.comparison_cid
    assert left.outcome_cid == right.outcome_cid


def test_verification_evidence_override_does_not_upgrade_failure() -> None:
    """Passing shared verification evidence cannot erase a failed attempt status."""

    compressed = _attempt(
        ShadowAttemptRole.COMPRESSED.value,
        attempt_status=AttemptTerminalStatus.FAILED,
        failure_reason_codes=("selected_tests_failed",),
        verification=_verification(
            selected_tests_passed=False,
            acceptance_matrix_satisfied=False,
            production_eligible=False,
        ),
    )
    expanded = _attempt(ShadowAttemptRole.EXPANDED.value)
    # Optimistic evidence must not rewrite the failed terminal status.
    optimistic = _verification(
        selected_tests_passed=True,
        full_suite_passed=True,
        acceptance_matrix_satisfied=True,
        production_eligible=False,
    )
    outcome = compare_shadow_results(
        shadow_result=_shadow_result(compressed=compressed, expanded=expanded),
        verification_evidence=optimistic,
    )
    assert (
        outcome.comparative_outcome
        == ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
    )
