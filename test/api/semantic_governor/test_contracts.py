"""Contract vectors for shadow execution and semantic differentials (SCG-008).

Acceptance criteria enforced here:

* Text difference alone cannot classify failure.
* Expanded output is never marked accepted by construction.
* Simulated/live provenance is unambiguous.
"""

from __future__ import annotations

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    AttemptTerminalStatus,
    ComparativeOutcome,
    CostTimingProjection,
    DIFFERENTIAL_PATCH_REPORT_INTERFACE,
    DifferentialPatchReport,
    ExecutionMode,
    OutcomeClassificationBasis,
    PairedAttemptRecord,
    SEMANTIC_GOVERNOR_EXECUTION_INTERFACE,
    SEMANTIC_OUTCOME_COMPARISON_INTERFACE,
    SHADOW_EXECUTION_PLAN_INTERFACE,
    SHADOW_EXECUTION_RESULT_INTERFACE,
    SCG_EXECUTION_CONTRACTS_EVIDENCE,
    SemanticEditClass,
    SemanticGovernorExecutionError,
    SemanticOutcomeComparison,
    ShadowAttemptRole,
    ShadowExecutionPlan,
    ShadowExecutionResult,
    ShadowSelectionReason,
    VerificationProjection,
    assert_expanded_never_accepted,
    assert_failure_classification_not_text_alone,
    comparative_outcomes,
    outcome_classification_bases,
    verify_comparison_identity,
    verify_plan_identity,
    verify_report_identity,
    verify_result_identity,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AuthoritySource,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
    AssumptionKind,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _generator(**overrides: object) -> GeneratorIdentity:
    fields = {
        "generator_id": "shadow_execution",
        "generator_version": "1.0.0",
        "interface_id": "create_shadow_plan@1",
    }
    fields.update(overrides)
    return GeneratorIdentity(**fields)  # type: ignore[arg-type]


def _provenance(**overrides: object) -> ArtifactProvenance:
    fields = {
        "producer_id": "semantic_governor",
        "producer_version": "1",
        "execution_mode": ExecutionMode.LIVE,
        "authority_source": AuthoritySource.DETERMINISTIC,
        "input_cids": (_cid("input-a"),),
        "tool_ids": ("shadow.v1",),
        "policy_cid": _cid("policy"),
        "notes": None,
    }
    fields.update(overrides)
    return ArtifactProvenance(**fields)  # type: ignore[arg-type]


def _header(artifact_kind: str, **overrides: object) -> GovernorArtifactHeader:
    context = _cid("context-pack-compressed")
    fields = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": context,
        "verification_bundle_cid": _cid("verification-bundle"),
        "generator": _generator(),
        "provenance": _provenance(),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="isolated_worktree",
                kind=AssumptionKind.ENVIRONMENT,
                statement="Paired shadow runs use disposable evaluation worktrees",
                supporting_cids=(_cid("worktree-policy"),),
            ),
        ),
        "metadata": {"task": "SCG-008"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)  # type: ignore[arg-type]


def _cost(**overrides: object) -> CostTimingProjection:
    fields = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "wall_time_ms": 1500,
        "model_spend_micros": 25000,
        "verification_time_ms": 300,
    }
    fields.update(overrides)
    return CostTimingProjection(**fields)  # type: ignore[arg-type]


def _verification(**overrides: object) -> VerificationProjection:
    fields = {
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
    return VerificationProjection(**fields)  # type: ignore[arg-type]


def _attempt(
    role: str = ShadowAttemptRole.COMPRESSED.value,
    **overrides: object,
) -> PairedAttemptRecord:
    defaults: dict[str, object] = {
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
    return PairedAttemptRecord(**defaults)  # type: ignore[arg-type]


def _plan(**overrides: object) -> ShadowExecutionPlan:
    compressed = _cid("context-pack-compressed")
    fields = {
        "header": _header("shadow_execution_plan", context_pack_cid=compressed),
        "task_id": "SCG-008",
        "audit_policy_cid": _cid("audit-policy"),
        "compressed_context_pack_cid": compressed,
        "expanded_context_pack_cid": _cid("context-pack-expanded"),
        "compressed_route_id": "route.compressed",
        "expanded_route_id": "route.expanded",
        "selection_reasons": (
            ShadowSelectionReason.RISK_CLASS_MANDATORY.value,
            ShadowSelectionReason.RANDOM_QUALITY_CONTROL.value,
        ),
        "max_wall_time_ms": 120_000,
        "max_model_spend_micros": 5_000_000,
        "max_expansion_token_budget": 50_000,
        "isolated_evaluation_worktree_required": True,
        "expanded_is_oracle_candidate_only": True,
        "allow_external_expanded_disclosure": False,
        "metadata": {"evidence": SCG_EXECUTION_CONTRACTS_EVIDENCE},
    }
    fields.update(overrides)
    return ShadowExecutionPlan(**fields)  # type: ignore[arg-type]


def _result(**overrides: object) -> ShadowExecutionResult:
    plan = _plan()
    fields = {
        "header": _header("shadow_execution_result"),
        "plan_cid": plan.plan_cid,
        "compressed_attempt": _attempt(ShadowAttemptRole.COMPRESSED.value),
        "expanded_attempt": _attempt(ShadowAttemptRole.EXPANDED.value),
        "both_attempts_isolated": True,
        "expanded_skipped_reason": None,
        "metadata": {},
    }
    fields.update(overrides)
    return ShadowExecutionResult(**fields)  # type: ignore[arg-type]


def _report(**overrides: object) -> DifferentialPatchReport:
    result = _result()
    fields = {
        "header": _header("differential_patch_report"),
        "plan_cid": result.plan_cid,
        "shadow_result_cid": result.result_cid,
        "text_differs": True,
        "files_differ": False,
        "symbols_differ": False,
        "interfaces_differ": False,
        "side_effects_differ": False,
        "exceptions_differ": False,
        "schemas_differ": False,
        "tests_differ": False,
        "proofs_differ": False,
        "counterexamples_differ": False,
        "static_analysis_differ": False,
        "performance_differ": False,
        "acceptance_differ": False,
        "human_review_required": False,
        "ast_edit_classes": (SemanticEditClass.EQUIVALENT_REFORMAT.value,),
        "compressed_input_tokens": 1000,
        "expanded_input_tokens": 4000,
        "compressed_output_tokens": 200,
        "expanded_output_tokens": 220,
        "compressed_wall_time_ms": 1500,
        "expanded_wall_time_ms": 3000,
        "compressed_model_spend_micros": 25000,
        "expanded_model_spend_micros": 80000,
        "semantic_equivalent": True,
        "failure_classified": False,
        "classification_bases": (OutcomeClassificationBasis.TEXT_DIFF.value,),
        "textual_difference_is_not_semantic_failure": True,
        "metadata": {},
    }
    fields.update(overrides)
    return DifferentialPatchReport(**fields)  # type: ignore[arg-type]


def _comparison(**overrides: object) -> SemanticOutcomeComparison:
    report = _report()
    fields = {
        "header": _header("semantic_outcome_comparison"),
        "plan_cid": report.plan_cid,
        "shadow_result_cid": report.shadow_result_cid,
        "differential_report_cid": report.report_cid,
        "comparative_outcome": ComparativeOutcome.EQUIVALENT_SUCCESS,
        "compressed_acceptance": AcceptanceDisposition.NOT_ACCEPTED,
        "expanded_acceptance": AcceptanceDisposition.CANDIDATE_ONLY,
        "human_review_required": False,
        "classification_bases": (
            OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value,
            OutcomeClassificationBasis.AST_EDIT_CLASSES.value,
        ),
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return SemanticOutcomeComparison(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Closed vocabularies and interface pins
# ---------------------------------------------------------------------------


def test_comparative_outcomes_are_exactly_ten() -> None:
    expected = (
        "equivalent_success",
        "compressed_better",
        "expanded_better",
        "both_valid_different",
        "compressed_failed_expanded_succeeded",
        "compressed_succeeded_expanded_failed",
        "both_failed_same_reason",
        "both_failed_different_reason",
        "verification_inconclusive",
        "human_review_required",
    )
    assert comparative_outcomes() == expected
    assert len(ComparativeOutcome) == 10
    for value in expected:
        assert ComparativeOutcome(value).value == value


def test_interfaces_and_evidence_pins() -> None:
    assert SEMANTIC_GOVERNOR_EXECUTION_INTERFACE == "SemanticGovernorExecution@1"
    assert SHADOW_EXECUTION_PLAN_INTERFACE == "ShadowExecutionPlan@1"
    assert SHADOW_EXECUTION_RESULT_INTERFACE == "ShadowExecutionResult@1"
    assert DIFFERENTIAL_PATCH_REPORT_INTERFACE == "DifferentialPatchReport@1"
    assert SEMANTIC_OUTCOME_COMPARISON_INTERFACE == "SemanticOutcomeComparison@1"
    assert SCG_EXECUTION_CONTRACTS_EVIDENCE == "scg/execution-contracts@1"
    assert OutcomeClassificationBasis.TEXT_DIFF.value in outcome_classification_bases()


# ---------------------------------------------------------------------------
# Deterministic identity / round-trip
# ---------------------------------------------------------------------------


def test_shadow_plan_identity_is_deterministic() -> None:
    left = _plan()
    right = _plan()
    assert left.plan_cid == right.plan_cid
    assert left.to_dict() == right.to_dict()
    assert verify_plan_identity(left) == left.plan_cid
    restored = ShadowExecutionPlan.from_dict(left.to_dict())
    assert restored == left
    assert restored.plan_cid == left.plan_cid


def test_shadow_result_and_comparison_round_trip() -> None:
    result = _result()
    restored_result = ShadowExecutionResult.from_dict(result.to_dict())
    assert restored_result == result
    assert verify_result_identity(result) == result.result_cid

    report = _report(shadow_result_cid=result.result_cid, plan_cid=result.plan_cid)
    restored_report = DifferentialPatchReport.from_dict(report.to_dict())
    assert restored_report == report
    assert verify_report_identity(report) == report.report_cid

    comparison = _comparison(
        plan_cid=result.plan_cid,
        shadow_result_cid=result.result_cid,
        differential_report_cid=report.report_cid,
    )
    restored_comparison = SemanticOutcomeComparison.from_dict(comparison.to_dict())
    assert restored_comparison == comparison
    assert verify_comparison_identity(comparison) == comparison.comparison_cid


def test_selection_reason_order_does_not_change_plan_identity() -> None:
    left = _plan(
        selection_reasons=(
            ShadowSelectionReason.RANDOM_QUALITY_CONTROL.value,
            ShadowSelectionReason.RISK_CLASS_MANDATORY.value,
        )
    )
    right = _plan(
        selection_reasons=(
            ShadowSelectionReason.RISK_CLASS_MANDATORY.value,
            ShadowSelectionReason.RANDOM_QUALITY_CONTROL.value,
        )
    )
    assert left.plan_cid == right.plan_cid
    assert list(left.selection_reasons) == sorted(left.selection_reasons)


# ---------------------------------------------------------------------------
# Acceptance: text difference alone cannot classify failure
# ---------------------------------------------------------------------------


def test_text_difference_alone_cannot_classify_failure_on_report() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="text difference alone cannot classify"
    ):
        _report(
            text_differs=True,
            failure_classified=True,
            semantic_equivalent=None,
            classification_bases=(OutcomeClassificationBasis.TEXT_DIFF.value,),
            ast_edit_classes=(SemanticEditClass.EQUIVALENT_REFORMAT.value,),
        )


def test_text_only_cannot_mark_semantic_nonequivalent() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="not semantic failure|non-text evidence"
    ):
        _report(
            text_differs=True,
            failure_classified=False,
            semantic_equivalent=False,
            classification_bases=(OutcomeClassificationBasis.TEXT_DIFF.value,),
            ast_edit_classes=(SemanticEditClass.EQUIVALENT_REFORMAT.value,),
        )


def test_failure_with_verification_evidence_is_allowed() -> None:
    report = _report(
        text_differs=True,
        tests_differ=True,
        failure_classified=True,
        semantic_equivalent=False,
        classification_bases=(
            OutcomeClassificationBasis.TEXT_DIFF.value,
            OutcomeClassificationBasis.TEST_RESULT_DIFF.value,
            OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value,
        ),
        ast_edit_classes=(SemanticEditClass.MODIFY_LOGIC.value,),
    )
    assert report.failure_classified is True
    assert report.textual_difference_is_not_semantic_failure is True


def test_assert_failure_helper_rejects_text_alone() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="text difference alone"):
        assert_failure_classification_not_text_alone(
            [OutcomeClassificationBasis.TEXT_DIFF.value],
            failure_classified=True,
        )
    assert_failure_classification_not_text_alone(
        [OutcomeClassificationBasis.PROOF_RECEIPTS.value],
        failure_classified=True,
    )


def test_failure_like_outcome_requires_non_text_bases() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="text difference alone"):
        _comparison(
            comparative_outcome=ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED,
            compressed_acceptance=AcceptanceDisposition.NOT_ACCEPTED,
            expanded_acceptance=AcceptanceDisposition.CANDIDATE_ONLY,
            classification_bases=(OutcomeClassificationBasis.TEXT_DIFF.value,),
        )


def test_textual_difference_policy_flag_must_remain_true() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError,
        match="textual_difference_is_not_semantic_failure",
    ):
        _report(textual_difference_is_not_semantic_failure=False)


# ---------------------------------------------------------------------------
# Acceptance: expanded output is never marked accepted
# ---------------------------------------------------------------------------


def test_expanded_attempt_cannot_be_accepted() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="never marked accepted"
    ):
        _attempt(
            ShadowAttemptRole.EXPANDED.value,
            acceptance_disposition=AcceptanceDisposition.ACCEPTED,
            verification=_verification(
                production_eligible=True,
                acceptance_matrix_satisfied=True,
            ),
        )


def test_expanded_attempt_cannot_be_production_eligible() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="cannot be production_eligible"
    ):
        _attempt(
            ShadowAttemptRole.EXPANDED.value,
            acceptance_disposition=AcceptanceDisposition.CANDIDATE_ONLY,
            verification=_verification(
                production_eligible=True,
                acceptance_matrix_satisfied=True,
            ),
        )


def test_expanded_acceptance_on_comparison_never_accepted() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="never marked accepted"):
        _comparison(expanded_acceptance=AcceptanceDisposition.ACCEPTED)


def test_assert_expanded_helper() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="never marked accepted"):
        assert_expanded_never_accepted(
            AcceptanceDisposition.ACCEPTED.value,
            role=ShadowAttemptRole.EXPANDED.value,
        )
    assert_expanded_never_accepted(
        AcceptanceDisposition.CANDIDATE_ONLY.value,
        role=ShadowAttemptRole.EXPANDED.value,
    )


def test_plan_requires_expanded_oracle_only_and_isolated_worktree() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="expanded_is_oracle_candidate_only"
    ):
        _plan(expanded_is_oracle_candidate_only=False)
    with pytest.raises(
        SemanticGovernorExecutionError, match="isolated_evaluation_worktree_required"
    ):
        _plan(isolated_evaluation_worktree_required=False)


def test_valid_expanded_is_candidate_only() -> None:
    attempt = _attempt(ShadowAttemptRole.EXPANDED.value)
    assert attempt.acceptance_disposition == AcceptanceDisposition.CANDIDATE_ONLY.value
    assert attempt.verification.production_eligible is False


# ---------------------------------------------------------------------------
# Acceptance: simulated / live provenance is unambiguous
# ---------------------------------------------------------------------------


def test_execution_mode_is_closed_and_unambiguous() -> None:
    live = _attempt(execution_mode=ExecutionMode.LIVE)
    simulated = _attempt(
        execution_mode=ExecutionMode.SIMULATED,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        verification=_verification(production_eligible=False),
    )
    replay = _attempt(
        execution_mode=ExecutionMode.REPLAY,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
        verification=_verification(production_eligible=False),
    )
    assert live.execution_mode == "live"
    assert simulated.execution_mode == "simulated"
    assert replay.execution_mode == "replay"
    with pytest.raises(SemanticGovernorExecutionError, match="unsupported value"):
        _attempt(execution_mode="maybe_live")


def test_simulated_attempt_cannot_be_accepted_or_production_eligible() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="simulated attempt"):
        _attempt(
            execution_mode=ExecutionMode.SIMULATED,
            acceptance_disposition=AcceptanceDisposition.ACCEPTED,
            verification=_verification(
                production_eligible=True,
                acceptance_matrix_satisfied=True,
            ),
        )
    with pytest.raises(SemanticGovernorExecutionError, match="simulated attempt"):
        _attempt(
            execution_mode=ExecutionMode.SIMULATED,
            acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
            verification=_verification(
                production_eligible=True,
                acceptance_matrix_satisfied=True,
            ),
        )


def test_simulated_header_requires_simulated_attempts() -> None:
    sim_header = _header(
        "shadow_execution_result",
        provenance=_provenance(execution_mode=ExecutionMode.SIMULATED),
        terminal_status=GovernorTerminalStatus.SIMULATED,
    )
    with pytest.raises(
        SemanticGovernorExecutionError, match="simulated header provenance"
    ):
        _result(
            header=sim_header,
            compressed_attempt=_attempt(execution_mode=ExecutionMode.LIVE),
            expanded_attempt=_attempt(
                ShadowAttemptRole.EXPANDED.value,
                execution_mode=ExecutionMode.LIVE,
            ),
        )
    sealed = _result(
        header=sim_header,
        compressed_attempt=_attempt(
            execution_mode=ExecutionMode.SIMULATED,
            verification=_verification(production_eligible=False),
        ),
        expanded_attempt=_attempt(
            ShadowAttemptRole.EXPANDED.value,
            execution_mode=ExecutionMode.SIMULATED,
            verification=_verification(production_eligible=False),
        ),
    )
    assert sealed.header.provenance.execution_mode == "simulated"
    assert sealed.compressed_attempt.execution_mode == "simulated"
    assert sealed.expanded_attempt is not None
    assert sealed.expanded_attempt.execution_mode == "simulated"


def test_simulated_comparison_cannot_accept_compressed() -> None:
    sim_header = _header(
        "semantic_outcome_comparison",
        provenance=_provenance(execution_mode=ExecutionMode.SIMULATED),
        terminal_status=GovernorTerminalStatus.SIMULATED,
    )
    with pytest.raises(
        SemanticGovernorExecutionError, match="simulated provenance cannot accept"
    ):
        _comparison(
            header=sim_header,
            comparative_outcome=ComparativeOutcome.EQUIVALENT_SUCCESS,
            compressed_acceptance=AcceptanceDisposition.ACCEPTED,
            expanded_acceptance=AcceptanceDisposition.CANDIDATE_ONLY,
            classification_bases=(
                OutcomeClassificationBasis.VERIFICATION_RECEIPTS.value,
            ),
        )


def test_live_and_simulated_are_distinct_in_identity() -> None:
    live = _attempt(execution_mode=ExecutionMode.LIVE)
    simulated = _attempt(
        execution_mode=ExecutionMode.SIMULATED,
        verification=_verification(production_eligible=False),
    )
    assert live.attempt_cid != simulated.attempt_cid
    assert live.to_dict()["execution_mode"] == "live"
    assert simulated.to_dict()["execution_mode"] == "simulated"


# ---------------------------------------------------------------------------
# Fail-closed: unknown fields, forged CIDs, floats, private data
# ---------------------------------------------------------------------------


def test_unknown_fields_fail_closed() -> None:
    payload = _plan().to_dict()
    payload["extra"] = True
    with pytest.raises(SemanticGovernorExecutionError, match="fields must be exactly"):
        ShadowExecutionPlan.from_dict(payload)


def test_forged_plan_cid_fails_closed() -> None:
    payload = _plan().to_dict()
    payload["plan_cid"] = _cid("forged-plan")
    with pytest.raises(SemanticGovernorExecutionError, match="does not verify"):
        ShadowExecutionPlan.from_dict(payload)


def test_floats_fail_closed_in_metadata() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="DAG-JSON|float"):
        _plan(metadata={"score": 0.5})


def test_private_data_rejected() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="private data"):
        _plan(metadata={"raw_source": "def x():\n  pass\n"})


def test_unsupported_outcome_fails_closed() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="unsupported value"):
        _comparison(comparative_outcome="model_says_same")


def test_failed_attempt_requires_reason_codes() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="failure_reason_code"):
        _attempt(
            attempt_status=AttemptTerminalStatus.FAILED,
            acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED,
            failure_reason_codes=(),
            verification=_verification(
                selected_tests_passed=False,
                full_suite_passed=False,
                acceptance_matrix_satisfied=False,
                production_eligible=False,
            ),
        )


def test_skipped_expanded_requires_reason() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="expanded_skipped_reason"):
        _result(expanded_attempt=None, expanded_skipped_reason=None)
    skipped = _result(
        expanded_attempt=None,
        expanded_skipped_reason=ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value,
        both_attempts_isolated=False,
    )
    assert skipped.expanded_attempt is None
    assert skipped.expanded_skipped_reason == "disclosure_forbidden_skip"


def test_disclosure_forbidden_cannot_allow_external() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="disclosure_forbidden"):
        _plan(
            selection_reasons=(ShadowSelectionReason.DISCLOSURE_FORBIDDEN_SKIP.value,),
            allow_external_expanded_disclosure=True,
        )


def test_header_context_pack_must_match_compressed() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="compressed_context_pack_cid"
    ):
        _plan(
            header=_header(
                "shadow_execution_plan",
                context_pack_cid=_cid("other-pack"),
            )
        )


def test_production_eligible_requires_acceptance_matrix() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="acceptance_matrix_satisfied"
    ):
        _verification(
            production_eligible=True,
            acceptance_matrix_satisfied=False,
        )


def test_compressed_accepted_requires_production_eligible() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="production_eligible"
    ):
        _attempt(
            ShadowAttemptRole.COMPRESSED.value,
            acceptance_disposition=AcceptanceDisposition.ACCEPTED,
            verification=_verification(
                production_eligible=False,
                acceptance_matrix_satisfied=True,
            ),
        )


def test_human_review_outcome_consistency() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="human_review_required"):
        _comparison(
            comparative_outcome=ComparativeOutcome.HUMAN_REVIEW_REQUIRED,
            human_review_required=False,
            compressed_acceptance=AcceptanceDisposition.HUMAN_REVIEW_REQUIRED,
            expanded_acceptance=AcceptanceDisposition.CANDIDATE_ONLY,
            classification_bases=(OutcomeClassificationBasis.HUMAN_REVIEW.value,),
        )
    sealed = _comparison(
        comparative_outcome=ComparativeOutcome.HUMAN_REVIEW_REQUIRED,
        human_review_required=True,
        compressed_acceptance=AcceptanceDisposition.HUMAN_REVIEW_REQUIRED,
        expanded_acceptance=AcceptanceDisposition.HUMAN_REVIEW_REQUIRED,
        classification_bases=(OutcomeClassificationBasis.HUMAN_REVIEW.value,),
    )
    assert sealed.comparative_outcome == "human_review_required"


def test_artifact_kinds_must_match_record_type() -> None:
    with pytest.raises(SemanticGovernorExecutionError, match="artifact_kind"):
        _plan(header=_header("shadow_execution_result"))


def test_paired_execution_requires_isolation() -> None:
    with pytest.raises(
        SemanticGovernorExecutionError, match="both_attempts_isolated"
    ):
        _result(both_attempts_isolated=False)
