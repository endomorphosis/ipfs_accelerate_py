"""CBP-130: closed-loop quality, coverage, token, and proof-cost gates."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.code_claim_contracts import (
    ClaimFamily,
    ClaimStatus,
    EvidenceTier,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.supervisor_code_proof_benchmark import (
    CODEBASE_PROOF_BENCHMARK_CORPUS_VERSION,
    CODEBASE_PROOF_BENCHMARK_INTERFACE,
    CODEBASE_PROOF_EFFICIENCY_REQUIREMENT_ID,
    CODEBASE_PROOF_OBJECTIVE_ID,
    MIN_INPUT_TOKEN_REDUCTION_BPS,
    MIN_RETRY_TOKEN_REDUCTION_BPS,
    REQUIRED_CLAIM_FAMILIES,
    REQUIRED_CLAIM_STATUSES,
    ClaimOutcomeObservation,
    CodeProofArmObservation,
    CodeProofGateName,
    CodeProofPairedCase,
    CodebaseProofBenchmark,
    CodebaseProofBenchmarkError,
    CodebaseProofBenchmarkSuite,
    ContextPath,
    MutationSeedCase,
    MutationSeedKind,
    ResultChannel,
    build_preregistered_codebase_proof_suite,
    evaluate_codebase_proof_benchmark,
    run_codebase_proof_efficiency_gates,
    verify_codebase_proof_benchmark_report,
)
from ipfs_accelerate_py.agent_supervisor.supervisor_efficiency_metrics import (
    CODE_PROOF_EFFICIENCY_EVIDENCE_ID,
    CODE_PROOF_EFFICIENCY_EVIDENCE_PRODUCERS,
    CODE_PROOF_MIN_INPUT_TOKEN_REDUCTION_BPS,
    CODE_PROOF_MIN_RETRY_TOKEN_REDUCTION_BPS,
    CodeProofEfficiencyCase,
    CodeProofEfficiencyReport,
    build_code_proof_efficiency_report,
    build_code_proof_paired_receipts,
)


def test_preregistered_suite_spans_required_families_and_mutation_kinds():
    suite = build_preregistered_codebase_proof_suite()
    assert suite.interface == CODEBASE_PROOF_BENCHMARK_INTERFACE
    assert suite.corpus_version == CODEBASE_PROOF_BENCHMARK_CORPUS_VERSION
    assert suite.objective_id == CODEBASE_PROOF_OBJECTIVE_ID
    families = {case.claim_family for case in suite.paired_cases}
    assert set(REQUIRED_CLAIM_FAMILIES).issubset(families)
    kinds = {seed.kind for seed in suite.mutation_seeds}
    assert MutationSeedKind.FALSE_ADMIT in kinds
    assert MutationSeedKind.FALSE_REFUTE in kinds
    assert MutationSeedKind.STALE_EVIDENCE in kinds
    assert MutationSeedKind.FIRST_PASS_REPAIR in kinds
    assert MutationSeedKind.EVENTUAL_REPAIR in kinds
    assert MutationSeedKind.ACCEPTED_PATCH_REGRESSION in kinds
    assert MutationSeedKind.WARM_CACHE_DOMINATED in kinds
    assert MutationSeedKind.REQUIRED_COVERAGE in kinds
    for case in suite.paired_cases:
        assert case.bulk.path is ContextPath.BULK_SOURCE
        assert case.obligation_first.path is ContextPath.OBLIGATION_FIRST
        assert case.bulk.channel is ResultChannel.DETERMINISTIC_FIXTURE
        assert (
            case.obligation_first.channel is ResultChannel.DETERMINISTIC_FIXTURE
        )
        assert case.bulk.task_reference == case.obligation_first.task_reference


def test_fixture_gates_pass_with_required_thresholds():
    report = run_codebase_proof_efficiency_gates()
    assert report.passed
    assert report.fixture_gates_authoritative
    assert report.false_admit_count == 0
    assert report.required_coverage_loss_count == 0
    assert (
        report.tokens_per_criterion_reduction_bps
        >= MIN_INPUT_TOKEN_REDUCTION_BPS
    )
    assert report.retry_token_reduction_bps >= MIN_RETRY_TOKEN_REDUCTION_BPS
    assert report.warm_prove_cost_reduction_bps > 0
    assert CODEBASE_PROOF_EFFICIENCY_REQUIREMENT_ID in report.evidence_claim_references
    assert CODE_PROOF_EFFICIENCY_EVIDENCE_ID in report.evidence_claim_references

    by_name = {gate.name: gate for gate in report.gates}
    assert by_name[CodeProofGateName.ZERO_FALSE_AUTHORITATIVE_ADMISSIONS].passed
    assert by_name[CodeProofGateName.NO_REQUIRED_COVERAGE_LOSS].passed
    assert by_name[CodeProofGateName.INPUT_TOKEN_REDUCTION].passed
    assert by_name[CodeProofGateName.RETRY_TOKEN_REDUCTION].passed
    assert by_name[CodeProofGateName.WARM_PROVE_COST_IMPROVEMENT].passed
    assert by_name[CodeProofGateName.REQUIRED_FAMILY_COVERAGE].passed
    assert by_name[CodeProofGateName.FIXTURE_CHANNEL_ISOLATION].passed


def test_report_covers_status_family_tier_and_assurance_dimensions():
    report = run_codebase_proof_efficiency_gates()
    for status in REQUIRED_CLAIM_STATUSES:
        assert status.value in report.status_counts
    # Fixture population includes the six lifecycle statuses used by claims.
    observed_statuses = {
        key for key, value in report.status_counts.items() if value > 0
    }
    assert ClaimStatus.SATISFIED.value in observed_statuses
    assert ClaimStatus.OPEN.value in observed_statuses
    assert ClaimStatus.STALE.value in observed_statuses

    for family in REQUIRED_CLAIM_FAMILIES:
        assert family.value in report.family_coverage
        bucket = report.family_coverage[family.value]
        assert bucket["required"] >= 1
        assert bucket["satisfied"] >= 1

    assert report.evidence_tier_counts.get(EvidenceTier.KERNEL_PROOF.value, 0) >= 1
    assert (
        report.assurance_counts.get(AssuranceLevel.KERNEL_VERIFIED.value, 0) >= 1
    )


def test_report_includes_all_required_efficiency_and_quality_metrics():
    report = run_codebase_proof_efficiency_gates()
    payload = report.to_dict()
    required_keys = {
        "false_admit_count",
        "false_refute_count",
        "false_admit_rate_bps",
        "false_refute_rate_bps",
        "stale_evidence_detected",
        "stale_evidence_expected",
        "first_pass_success_count",
        "first_pass_success_rate_bps",
        "eventual_repair_success_count",
        "eventual_repair_success_rate_bps",
        "accepted_patch_regression_count",
        "accepted_patch_regression_rate_bps",
        "input_tokens_per_accepted_criterion_bulk",
        "input_tokens_per_accepted_criterion_obligation",
        "bulk_retry_tokens",
        "obligation_retry_tokens",
        "bulk_provider_calls",
        "obligation_provider_calls",
        "cache_hit_rate_bps_obligation",
        "cache_reject_rate_bps_obligation",
        "bulk_wall_time_ms",
        "obligation_wall_time_ms",
        "bulk_proof_cost_microunits",
        "obligation_proof_cost_microunits",
        "tokens_per_criterion_reduction_bps",
        "retry_token_reduction_bps",
        "warm_prove_cost_reduction_bps",
    }
    assert required_keys.issubset(payload)
    assert report.stale_evidence_detected == report.stale_evidence_expected
    assert report.stale_evidence_detected >= 1
    assert report.accepted_patch_regression_count == 0
    assert report.mutation_seed_match_count == report.mutation_seed_total
    assert report.bulk_input_tokens > report.obligation_input_tokens
    assert report.bulk_retry_tokens > report.obligation_retry_tokens
    assert report.bulk_proof_cost_microunits > report.obligation_proof_cost_microunits


def test_live_model_channel_is_reported_separately_from_fixture_gates():
    suite = build_preregistered_codebase_proof_suite(
        include_live_model_channel=True
    )
    assert suite.live_model_channel is not None
    assert suite.live_model_channel.to_dict()["authoritative_for_fixture_gates"] is False
    report = evaluate_codebase_proof_benchmark(suite)
    assert report.passed
    assert report.live_model_summary["present"] is True
    assert report.live_model_summary["authoritative_for_fixture_gates"] is False
    assert report.live_model_summary["case_count"] >= 1
    # Live observations do not inflate fixture false-admit counts.
    assert report.false_admit_count == 0


def test_report_is_recomputed_and_tampering_fails_closed():
    suite = build_preregistered_codebase_proof_suite()
    report = evaluate_codebase_proof_benchmark(suite)
    assert verify_codebase_proof_benchmark_report(report, suite)
    assert verify_codebase_proof_benchmark_report(report.to_dict(), suite)

    tampered = report.to_dict()
    tampered["false_admit_count"] = 99
    tampered["report_id"] = report.report_id
    assert not verify_codebase_proof_benchmark_report(tampered, suite)

    payload = suite.to_dict()
    payload["paired_cases"] = payload["paired_cases"][1:]
    with pytest.raises(CodebaseProofBenchmarkError, match="preregistration_digest|required claim families"):
        CodebaseProofBenchmarkSuite.from_dict(payload)


def test_suite_round_trip_and_alias():
    suite = build_preregistered_codebase_proof_suite()
    restored = CodebaseProofBenchmarkSuite.from_dict(suite.to_dict())
    assert restored.suite_id == suite.suite_id
    assert isinstance(suite, CodebaseProofBenchmark)
    encoded = json.dumps(suite.to_dict())
    assert "prompt" not in encoded.lower() or "prompt" not in suite.to_dict()
    # Forbidden payload keys must not appear as field names.
    blob = json.dumps(suite.to_dict())
    for forbidden in ("source_body", "decoded_output", "proof_body", "patch\""):
        assert forbidden not in blob


def test_efficiency_metrics_extension_meets_cbp_thresholds():
    suite = build_preregistered_codebase_proof_suite()
    efficiency = build_code_proof_efficiency_report(suite.paired_cases)
    assert isinstance(efficiency, CodeProofEfficiencyReport)
    assert efficiency.passed
    assert (
        efficiency.tokens_per_criterion_reduction_bps
        >= CODE_PROOF_MIN_INPUT_TOKEN_REDUCTION_BPS
    )
    assert (
        efficiency.retry_token_reduction_bps
        >= CODE_PROOF_MIN_RETRY_TOKEN_REDUCTION_BPS
    )
    assert efficiency.warm_prove_cost_reduction_bps > 0
    assert efficiency.required_coverage_preserved
    assert CODE_PROOF_EFFICIENCY_EVIDENCE_ID in efficiency.evidence_claim_references
    assert (
        CODE_PROOF_EFFICIENCY_EVIDENCE_PRODUCERS[CODE_PROOF_EFFICIENCY_EVIDENCE_ID]
        == "supervisor_efficiency_metrics.build_code_proof_efficiency_report"
    )

    # Receipt projection path also produces a passing report.
    receipts = build_code_proof_paired_receipts(suite.paired_cases)
    from_receipts = build_code_proof_efficiency_report(receipts)
    assert from_receipts.required_coverage_preserved
    assert from_receipts.cases


def test_false_admit_seed_fails_zero_false_admission_gate():
    suite = build_preregistered_codebase_proof_suite()
    poisoned_cases = []
    for case in suite.paired_cases:
        if case.claim_family is ClaimFamily.SECURITY_PROPERTY:
            bad_claim = ClaimOutcomeObservation(
                claim_reference="claim:poison:false-admit",
                claim_family=ClaimFamily.SECURITY_PROPERTY,
                evidence_tier=EvidenceTier.QUERY_FACT,
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                status=ClaimStatus.SATISFIED,
                authoritative_admission=True,
                false_admit=True,
                false_refute=False,
                required_for_coverage=False,
            )
            bulk = CodeProofArmObservation.from_dict(case.bulk.to_dict())
            bulk_payload = bulk.to_dict()
            bulk_payload["claims"] = list(bulk_payload["claims"]) + [
                bad_claim.to_dict()
            ]
            # Rebuild arm without schema-derived keys that confuse from_dict.
            for key in (
                "input_tokens_per_accepted_criterion",
                "cache_hit_rate_bps",
                "cache_reject_rate_bps",
            ):
                bulk_payload.pop(key, None)
            new_bulk = CodeProofArmObservation.from_dict(bulk_payload)
            poisoned_cases.append(
                CodeProofPairedCase(
                    task_reference=case.task_reference,
                    claim_family=case.claim_family,
                    bulk=new_bulk,
                    obligation_first=case.obligation_first,
                    warm_cache_dominated=case.warm_cache_dominated,
                )
            )
        else:
            poisoned_cases.append(case)

    poisoned = CodebaseProofBenchmarkSuite(
        corpus_version=suite.corpus_version,
        repository_id=suite.repository_id,
        tree_id=suite.tree_id,
        policy_id=suite.policy_id,
        policy_revision=suite.policy_revision,
        objective_id=suite.objective_id,
        objective_revision=suite.objective_revision,
        paired_cases=tuple(poisoned_cases),
        mutation_seeds=suite.mutation_seeds,
    )
    report = evaluate_codebase_proof_benchmark(poisoned)
    assert report.false_admit_count >= 1
    assert not report.passed
    gate = next(
        g
        for g in report.gates
        if g.name is CodeProofGateName.ZERO_FALSE_AUTHORITATIVE_ADMISSIONS
    )
    assert not gate.passed
    assert report.evidence_claim_references == ()


def test_coverage_loss_fails_gate():
    suite = build_preregistered_codebase_proof_suite()
    degraded = []
    for case in suite.paired_cases:
        if case.task_reference.endswith(":dependency"):
            obl_payload = case.obligation_first.to_dict()
            # Drop the only required satisfied claim on the obligation arm.
            obl_payload["claims"] = [
                c
                for c in obl_payload["claims"]
                if not (
                    c.get("required_for_coverage")
                    and c.get("status") == ClaimStatus.SATISFIED.value
                )
            ]
            for key in (
                "input_tokens_per_accepted_criterion",
                "cache_hit_rate_bps",
                "cache_reject_rate_bps",
            ):
                obl_payload.pop(key, None)
            degraded.append(
                CodeProofPairedCase(
                    task_reference=case.task_reference,
                    claim_family=case.claim_family,
                    bulk=case.bulk,
                    obligation_first=CodeProofArmObservation.from_dict(obl_payload),
                    warm_cache_dominated=case.warm_cache_dominated,
                )
            )
        else:
            degraded.append(case)
    degraded_suite = CodebaseProofBenchmarkSuite(
        corpus_version=suite.corpus_version,
        repository_id=suite.repository_id,
        tree_id=suite.tree_id,
        policy_id=suite.policy_id,
        policy_revision=suite.policy_revision,
        objective_id=suite.objective_id,
        objective_revision=suite.objective_revision,
        paired_cases=tuple(degraded),
        mutation_seeds=suite.mutation_seeds,
    )
    report = evaluate_codebase_proof_benchmark(degraded_suite)
    assert report.required_coverage_loss_count >= 1
    assert not report.passed


def test_token_and_retry_gates_reject_insufficient_reduction():
    suite = build_preregistered_codebase_proof_suite()
    weak = []
    for case in suite.paired_cases:
        bulk_payload = case.bulk.to_dict()
        obl_payload = case.obligation_first.to_dict()
        # Force nearly-identical token usage so gates fail.
        obl_payload["input_tokens"] = bulk_payload["input_tokens"]
        obl_payload["retry_tokens"] = bulk_payload["retry_tokens"]
        for payload in (bulk_payload, obl_payload):
            for key in (
                "input_tokens_per_accepted_criterion",
                "cache_hit_rate_bps",
                "cache_reject_rate_bps",
            ):
                payload.pop(key, None)
        weak.append(
            CodeProofPairedCase(
                task_reference=case.task_reference,
                claim_family=case.claim_family,
                bulk=CodeProofArmObservation.from_dict(bulk_payload),
                obligation_first=CodeProofArmObservation.from_dict(obl_payload),
                warm_cache_dominated=case.warm_cache_dominated,
            )
        )
    weak_suite = CodebaseProofBenchmarkSuite(
        corpus_version=suite.corpus_version,
        repository_id=suite.repository_id,
        tree_id=suite.tree_id,
        policy_id=suite.policy_id,
        policy_revision=suite.policy_revision,
        objective_id=suite.objective_id,
        objective_revision=suite.objective_revision,
        paired_cases=tuple(weak),
        mutation_seeds=suite.mutation_seeds,
    )
    report = evaluate_codebase_proof_benchmark(weak_suite)
    assert report.tokens_per_criterion_reduction_bps < MIN_INPUT_TOKEN_REDUCTION_BPS
    assert report.retry_token_reduction_bps < MIN_RETRY_TOKEN_REDUCTION_BPS
    assert not report.passed


def test_claim_outcome_rejects_incoherent_false_flags():
    with pytest.raises(CodebaseProofBenchmarkError, match="false_admit"):
        ClaimOutcomeObservation(
            claim_reference="claim:bad",
            claim_family=ClaimFamily.API_CONTRACT,
            evidence_tier=EvidenceTier.KERNEL_PROOF,
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
            status=ClaimStatus.OPEN,
            authoritative_admission=False,
            false_admit=True,
            false_refute=False,
        )
    with pytest.raises(CodebaseProofBenchmarkError, match="false_refute"):
        ClaimOutcomeObservation(
            claim_reference="claim:bad2",
            claim_family=ClaimFamily.API_CONTRACT,
            evidence_tier=EvidenceTier.KERNEL_PROOF,
            required_assurance=AssuranceLevel.KERNEL_VERIFIED,
            status=ClaimStatus.SATISFIED,
            authoritative_admission=True,
            false_admit=False,
            false_refute=True,
        )


def test_mutation_seed_match_and_mismatch():
    matched = MutationSeedCase(
        seed_id="mut.demo.match",
        kind=MutationSeedKind.STALE_EVIDENCE,
        claim_family=ClaimFamily.DEPENDENCY_REACHABILITY,
        task_reference="task:demo",
        expected_false_admit=False,
        expected_false_refute=False,
        expected_stale_detection=True,
        expected_first_pass_success=True,
        expected_eventual_repair_success=True,
        expected_accepted_patch_regression=False,
        observed_false_admit=False,
        observed_false_refute=False,
        observed_stale_detection=True,
        observed_first_pass_success=True,
        observed_eventual_repair_success=True,
        observed_accepted_patch_regression=False,
    )
    assert matched.seed_matched
    mismatched = MutationSeedCase(
        seed_id="mut.demo.mismatch",
        kind=MutationSeedKind.FALSE_ADMIT,
        claim_family=ClaimFamily.SECURITY_PROPERTY,
        task_reference="task:demo",
        expected_false_admit=False,
        expected_false_refute=False,
        expected_stale_detection=False,
        expected_first_pass_success=True,
        expected_eventual_repair_success=True,
        expected_accepted_patch_regression=False,
        observed_false_admit=True,
        observed_false_refute=False,
        observed_stale_detection=False,
        observed_first_pass_success=True,
        observed_eventual_repair_success=True,
        observed_accepted_patch_regression=False,
    )
    assert not mismatched.seed_matched


def test_code_proof_efficiency_case_round_trip():
    case = CodeProofEfficiencyCase(
        task_reference="task:rt",
        claim_family="api_contract",
        bulk_receipt_id="receipt:bulk",
        obligation_receipt_id="receipt:obl",
        bulk_input_tokens=10_000,
        obligation_input_tokens=5_000,
        bulk_retry_tokens=4_000,
        obligation_retry_tokens=1_000,
        bulk_proof_cost_microunits=2_000,
        obligation_proof_cost_microunits=500,
        bulk_accepted_criteria=2,
        obligation_accepted_criteria=2,
        bulk_provider_calls=4,
        obligation_provider_calls=2,
        bulk_cache_hits=0,
        obligation_cache_hits=3,
        bulk_cache_rejects=1,
        obligation_cache_rejects=0,
        bulk_wall_time_ms=10_000,
        obligation_wall_time_ms=4_000,
        required_evidence_references=("evidence:a", "evidence:b"),
        bulk_covered_evidence_references=("evidence:a", "evidence:b"),
        obligation_covered_evidence_references=("evidence:a", "evidence:b"),
        warm_cache_dominated=True,
    )
    restored = CodeProofEfficiencyCase.from_dict(case.to_dict())
    assert restored.case_id == case.case_id
    assert restored.tokens_per_criterion_reduction_bps >= 4_000
    assert restored.retry_token_reduction_bps >= 6_000
    assert restored.required_coverage_preserved

    report = CodeProofEfficiencyReport(cases=(case,))
    assert report.passed
    restored_report = CodeProofEfficiencyReport.from_dict(report.to_dict())
    assert restored_report.report_id == report.report_id


def test_compare_bulk_and_obligation_on_identical_tasks():
    suite = build_preregistered_codebase_proof_suite()
    report = evaluate_codebase_proof_benchmark(suite)
    for case in suite.paired_cases:
        assert case.bulk.task_reference == case.obligation_first.task_reference
        assert case.input_token_reduction_bps > 0
        assert case.retry_token_reduction_bps > 0
        assert case.required_coverage_preserved
    assert report.bulk_provider_calls >= report.obligation_provider_calls
    assert report.cache_hit_rate_bps_obligation >= 0
    assert report.efficiency_report is not None
    assert report.efficiency_report.passed
