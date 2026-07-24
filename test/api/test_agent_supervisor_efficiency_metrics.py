from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context_compiler import (
    DELTA_RETRY_EVIDENCE_ID,
    REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID as COMPILER_REQUIRED_CONTEXT_ID,
    ContextCompiler,
)
from ipfs_accelerate_py.agent_supervisor.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)
from ipfs_accelerate_py.agent_supervisor.supervisor_efficiency_metrics import (
    DELTA_RETRY_CONTEXT_EVIDENCE_ID,
    DELTA_RETRY_PROMOTION_REPORT_SCHEMA,
    EFFICIENCY_CONTRACT_VERSION,
    EFFICIENCY_RECEIPT_SCHEMA,
    EFFICIENCY_REPORT_SCHEMA,
    PAIRED_EFFICIENCY_CASE_SCHEMA,
    PAIRED_EFFICIENCY_REPORT_SCHEMA,
    REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,
    REQUIRED_CONTEXT_PROMOTION_REPORT_SCHEMA,
    TERMINAL_ACCEPTED_WORK_EVIDENCE_ID,
    TERMINAL_ACCEPTED_WORK_EVIDENCE_SCHEMA,
    MAX_ARTIFACT_REFERENCES,
    MAX_CHANGED_PATHS,
    MAX_DURATION_MS,
    MAX_EVIDENCE_REFERENCES,
    MAX_STAGES,
    MAX_TEXT_BYTES,
    MAX_TOKENS,
    ArtifactReference,
    CacheDisposition,
    CacheObservation,
    ChangedScope,
    DeltaRetryProofBinding,
    DeltaRetryPromotionReport,
    EfficiencyReceipt,
    EfficiencyReport,
    EfficiencyScenario,
    EfficiencyValidationError,
    EvidenceDelta,
    PairedEfficiencyCase,
    PairedEfficiencyReport,
    RetryObservation,
    RequiredContextProofBinding,
    RequiredContextPromotionReport,
    StageName,
    StageTiming,
    TerminalAcceptance,
    TerminalAcceptedWorkEvidence,
    TerminalOutcome,
    TokenUsage,
    WorkCost,
    WorkStatus,
    aggregate_efficiency_receipts,
    build_efficiency_baseline_fixtures,
    build_delta_retry_promotion_report,
    build_paired_efficiency_report,
    build_terminal_accepted_work_evidence,
    build_required_context_promotion_report,
)


def _fixtures() -> dict[str, EfficiencyReceipt]:
    return build_efficiency_baseline_fixtures()


def _delta_retry_fixture(*, requested_only: bool = False):
    """Compile a fresh ASI-G092 receipt on paired-report-compatible IDs."""

    tree_digest = "sha256:" + "1" * 64
    policy_digest = "sha256:" + "2" * 64
    compiler = ContextCompiler(
        ContextBudget(
            max_input_tokens=4_000,
            reserved_output_tokens=200,
            reserved_tool_tokens=100,
            max_items=64,
            max_serialized_bytes=262_144,
        ),
        tokenizer=lambda text: max(1, len(text.encode("utf-8")) // 24),
        provider_context_window=4_500,
    )

    def reference(
        reference_id: str,
        content_id: str,
        *,
        required: bool = False,
    ) -> ContextReference:
        return ContextReference(
            reference_id=reference_id,
            kind="benchmark-evidence",
            tier=(
                ContextTier.INVARIANT if required else ContextTier.EVIDENCE
            ),
            referenced_content_id=content_id,
            repository_id="repo:efficiency-delta",
            tree_id=tree_digest,
            token_count=60,
            metadata={
                "required": required,
                "coverage_ids": (
                    "coverage:required",
                )
                if required
                else (f"coverage:{reference_id}",),
            },
        )

    required = reference(
        "required",
        "sha256:" + "3" * 64,
        required=True,
    )
    optional = tuple(
        reference(
            f"optional-{index}",
            "sha256:" + f"{index + 4:x}" * 64,
        )
        for index in range(8)
    )
    parent = compiler.compile(
        repository_id="repo:efficiency-delta",
        tree_id=tree_digest,
        objective_id="ASI-G092",
        objective_revision="sha256:" + "a" * 64,
        policy_id="policy:supervisor",
        policy_revision=policy_digest,
        caller="supervisor:efficiency-test",
        stage="implementation",
        goal={"id": "ASI-G092", "summary": "Use retry deltas"},
        authority={"mode": "proposal", "allowed_paths": ["src"]},
        scope={"paths": ["src/context.py"]},
        acceptance={"criteria": ["retain required coverage"]},
        evidence=(required, *optional),
    ).capsule
    if requested_only:
        result = compiler.compile_delta(
            parent,
            evidence=(required, *optional),
            requested_reference_ids=("optional-0",),
        )
    else:
        changed = reference(
            "optional-0",
            "sha256:" + "f" * 64,
        )
        result = compiler.compile_delta(
            parent,
            evidence=(required, changed, *optional[1:]),
        )
    return result


def _required_context_fixture():
    """Compile a fresh ASI-G091 result for the typed promotion join."""

    tree_digest = "sha256:" + "7" * 64
    policy_digest = "sha256:" + "8" * 64
    compiler = ContextCompiler(
        ContextBudget(
            max_input_tokens=4_000,
            reserved_output_tokens=200,
            reserved_tool_tokens=100,
            max_items=64,
            max_serialized_bytes=262_144,
        ),
        tokenizer=lambda text: max(1, len(text.encode("utf-8")) // 24),
        provider_context_window=4_500,
    )
    required = ContextReference(
        reference_id="required",
        kind="benchmark-evidence",
        tier=ContextTier.INVARIANT,
        referenced_content_id="sha256:" + "9" * 64,
        repository_id="repo:efficiency-context",
        tree_id=tree_digest,
        token_count=60,
        metadata={
            "required": True,
            "coverage_ids": ("coverage:required",),
        },
    )
    return compiler.compile(
        repository_id="repo:efficiency-context",
        tree_id=tree_digest,
        objective_id="ASI-G091",
        objective_revision="sha256:" + "a" * 64,
        policy_id="policy:supervisor",
        policy_revision=policy_digest,
        caller="supervisor:efficiency-test",
        stage="implementation",
        goal={"id": "ASI-G091", "summary": "Preserve required context"},
        authority={"mode": "proposal", "allowed_paths": ["src"]},
        scope={"paths": ["src/context.py"]},
        acceptance={"criteria": ["retain required coverage"]},
        evidence=(required,),
    )


def _paired_required_context_report(context_result, *, verified: bool = False):
    candidate_input = context_result.receipt.input_tokens
    baseline_input = max(candidate_input * 2, candidate_input + 1)
    baseline = replace(
        _fixtures()["cold"],
        task_reference="task:required-context",
        goal_reference=context_result.receipt.objective_id,
        repository_tree_digest=context_result.receipt.tree_id,
        policy_digest=context_result.receipt.policy_revision,
        tokens=TokenUsage(input_tokens=baseline_input, output_tokens=200),
        evidence=EvidenceDelta(
            baseline_references=("coverage:required",),
            terminal_references=("coverage:required",),
        ),
    )
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=candidate_input, output_tokens=100),
        output_digest="e" * 64,
    )
    builder = (
        build_terminal_accepted_work_evidence
        if verified
        else build_paired_efficiency_report
    )
    return builder(
        (baseline,),
        (candidate,),
        required_evidence_by_task={
            "task:required-context": ("coverage:required",),
        },
    )


def _paired_delta_report(
    delta_result,
    *,
    common_input_tokens: int = 0,
    verified: bool = False,
):
    delta_receipt = delta_result.receipt
    baseline = replace(
        _fixtures()["cold"],
        task_reference="task:delta-retry",
        goal_reference=delta_receipt.objective_id,
        repository_tree_digest=delta_receipt.tree_id,
        policy_digest=delta_receipt.policy_revision,
        tokens=TokenUsage(
            input_tokens=(
                common_input_tokens + delta_receipt.full_replay_tokens
            ),
            output_tokens=200,
        ),
        evidence=EvidenceDelta(
            baseline_references=("coverage:required",),
            terminal_references=("coverage:required",),
        ),
    )
    candidate = replace(
        baseline,
        tokens=TokenUsage(
            input_tokens=common_input_tokens + delta_receipt.delta_tokens,
            output_tokens=100,
        ),
        output_digest="f" * 64,
    )
    builder = (
        build_terminal_accepted_work_evidence
        if verified
        else build_paired_efficiency_report
    )
    return builder(
        (baseline,),
        (candidate,),
        required_evidence_by_task={
            baseline.task_reference: ("coverage:required",)
        },
    )


def test_fixture_baselines_cover_required_end_to_end_scenarios() -> None:
    fixtures = _fixtures()

    assert tuple(fixtures) == (
        "cold",
        "warm",
        "failed",
        "repaired",
        "parallel-independent",
        "conflicting",
    )
    assert fixtures["cold"].scenario is EfficiencyScenario.COLD
    assert fixtures["cold"].reused_tokens == 0
    assert {
        item.disposition for item in fixtures["cold"].cache_observations
    } == {CacheDisposition.MISS}

    assert fixtures["warm"].scenario is EfficiencyScenario.WARM
    assert fixtures["warm"].reused_tokens > 0
    assert CacheDisposition.HIT in {
        item.disposition for item in fixtures["warm"].cache_observations
    }

    assert fixtures["failed"].terminal.outcome is TerminalOutcome.FAILED
    assert not fixtures["failed"].accepted
    assert (
        fixtures["failed"].validation.status is WorkStatus.FAILED
    )

    assert fixtures["repaired"].accepted
    assert fixtures["repaired"].retry_count == 1
    assert fixtures["repaired"].retries[0].delta_context_digest

    parallel = fixtures["parallel-independent"]
    assert parallel.related_task_references == ("task:parallel-b",)
    assert not parallel.conflict_references

    conflicting = fixtures["conflicting"]
    assert conflicting.terminal.outcome is TerminalOutcome.CONFLICTED
    assert conflicting.conflict_references == ("conflict:path-overlap",)


def test_receipt_joins_every_required_measurement_dimension() -> None:
    receipt = _fixtures()["repaired"]

    assert receipt.schema == EFFICIENCY_RECEIPT_SCHEMA
    assert receipt.schema_version == EFFICIENCY_CONTRACT_VERSION
    assert receipt.queue_delay_ms == 1_000
    assert receipt.stage_latency_ms(StageName.INFERENCE) == 4_000
    assert receipt.input_tokens == 4_900
    assert receipt.output_tokens == 780
    assert receipt.reused_tokens == 300
    assert receipt.cache_observations
    assert receipt.retry_count == 1
    assert receipt.validation.cost_microunits == 400
    assert receipt.proof.status is WorkStatus.NOT_REQUIRED
    assert receipt.changed_scope.paths
    assert receipt.artifacts
    assert receipt.terminal.accepted
    assert receipt.total_cost_microunits == (
        receipt.inference_cost_microunits
        + receipt.validation.cost_microunits
        + receipt.proof.cost_microunits
    )
    assert receipt.accepted_evidence_gain == 2
    assert receipt.evidence_gain_per_thousand_input_tokens == pytest.approx(
        2 * 1000 / 4_900
    )


def test_receipt_is_canonical_order_independent_and_round_trips() -> None:
    receipt = _fixtures()["cold"]
    reordered = replace(
        receipt,
        stages=tuple(reversed(receipt.stages)),
        artifacts=tuple(reversed(receipt.artifacts)),
        evidence=EvidenceDelta(
            baseline_references=tuple(
                reversed(receipt.evidence.baseline_references)
            ),
            terminal_references=tuple(
                reversed(receipt.evidence.terminal_references)
            ),
        ),
    )

    assert reordered == receipt
    assert reordered.receipt_id == receipt.receipt_id
    encoded = receipt.to_json()
    assert EfficiencyReceipt.from_json(encoded) == receipt
    assert EfficiencyReceipt.from_json(encoded.encode()) == receipt

    identified = receipt.to_dict(include_receipt_id=True)
    assert identified["receipt_id"] == receipt.receipt_id
    assert EfficiencyReceipt.from_dict(identified) == receipt


def test_receipt_rejects_tampered_identity_and_derived_claims() -> None:
    receipt = _fixtures()["cold"]
    payload = receipt.to_dict(include_receipt_id=True)
    payload["queue_delay_ms"] += 1
    with pytest.raises(EfficiencyValidationError, match="identity"):
        EfficiencyReceipt.from_dict(payload)

    payload = receipt.to_dict()
    payload["total_cost_microunits"] += 1
    with pytest.raises(EfficiencyValidationError, match="total_cost"):
        EfficiencyReceipt.from_dict(payload)

    payload = receipt.to_dict()
    payload["terminal"]["accepted"] = False
    with pytest.raises(EfficiencyValidationError, match="accepted claim"):
        EfficiencyReceipt.from_dict(payload)


def test_wire_receipt_contains_only_digests_and_bounded_references() -> None:
    receipt = _fixtures()["repaired"]
    wire = receipt.to_dict(include_receipt_id=True)
    encoded = json.dumps(wire, sort_keys=True)

    assert "prompt" not in encoded
    assert "source_body" not in encoded
    assert "decoded_output" not in encoded
    assert "artifact_graph" not in encoded
    assert receipt.context_digest.startswith("sha256:")
    assert receipt.input_digest.startswith("sha256:")
    assert receipt.output_digest.startswith("sha256:")
    assert all(item.digest.startswith("sha256:") for item in receipt.artifacts)
    assert len(receipt.canonical_bytes()) < 262_144

    wire["prompt"] = "private prompt"
    with pytest.raises(EfficiencyValidationError, match="unsupported fields"):
        EfficiencyReceipt.from_dict(wire)

    artifact = receipt.artifacts[0].to_dict()
    artifact["body"] = {"nested": ["artifact graph"]}
    with pytest.raises(EfficiencyValidationError, match="unsupported fields"):
        ArtifactReference.from_dict(artifact)


def test_aggregation_charges_all_attempts_but_rewards_only_acceptance() -> None:
    fixtures = _fixtures()
    receipts = tuple(fixtures.values())
    report = aggregate_efficiency_receipts(receipts)

    assert report.schema == EFFICIENCY_REPORT_SCHEMA
    assert report.receipt_count == 6
    assert report.accepted_receipt_count == 4
    assert report.accepted_task_count == 4
    assert report.total_cost_microunits == sum(
        receipt.total_cost_microunits for receipt in receipts
    )
    assert report.total_input_tokens == sum(
        receipt.input_tokens for receipt in receipts
    )
    assert report.stage_latency_ms["inference"] == sum(
        receipt.stage_latency_ms(StageName.INFERENCE) for receipt in receipts
    )
    assert report.stage_invocation_counts["validation"] == 6
    assert report.total_cache_bytes_reused == 2_048
    assert report.total_validation_duration_ms == 12_000
    assert report.total_proof_duration_ms == 0
    assert report.total_changed_file_count == 4
    assert report.total_changed_symbol_count == 4
    assert report.total_lines_added == 80
    assert report.total_lines_deleted == 12
    assert report.artifact_reference_count == 4
    assert report.accepted_evidence_gain == sum(
        receipt.evidence.gain for receipt in receipts if receipt.accepted
    )
    assert report.cost_per_accepted_task_microunits == pytest.approx(
        report.total_cost_microunits / 4
    )
    assert (
        report.evidence_gain_per_thousand_input_tokens
        == pytest.approx(
            report.accepted_evidence_gain
            * 1000
            / report.total_input_tokens
        )
    )
    assert report.cache_outcome_counts == {
        "bypass": 0,
        "error": 0,
        "hit": 1,
        "invalidated": 0,
        "miss": 5,
    }


def test_aggregate_is_deterministic_and_report_round_trips() -> None:
    receipts = tuple(_fixtures().values())
    forward = aggregate_efficiency_receipts(receipts)
    reverse = aggregate_efficiency_receipts(reversed(receipts))

    assert forward == reverse
    assert forward.report_id == reverse.report_id
    assert EfficiencyReport.from_json(forward.to_json()) == forward
    identified = forward.to_dict(include_report_id=True)
    assert EfficiencyReport.from_dict(identified) == forward

    tampered = forward.to_dict()
    tampered["total_cost_microunits"] += 1
    with pytest.raises(EfficiencyValidationError, match="total cost"):
        EfficiencyReport.from_dict(tampered)


def test_failed_attempt_before_repair_is_in_cost_for_one_accepted_task() -> None:
    fixtures = _fixtures()
    failed = replace(
        fixtures["failed"],
        task_reference=fixtures["repaired"].task_reference,
    )
    repaired = fixtures["repaired"]
    report = aggregate_efficiency_receipts((failed, repaired))

    assert report.accepted_task_count == 1
    assert report.receipt_count == 2
    assert report.cost_per_accepted_task_microunits == (
        failed.total_cost_microunits + repaired.total_cost_microunits
    )
    assert report.accepted_evidence_gain == repaired.evidence.gain
    assert report.total_input_tokens == (
        failed.input_tokens + repaired.input_tokens
    )


def test_empty_aggregate_has_defined_zero_projections() -> None:
    report = aggregate_efficiency_receipts(())

    assert report.receipt_count == 0
    assert report.accepted_task_count == 0
    assert report.total_cost_microunits == 0
    assert not report.cost_per_accepted_task_ratio.defined
    assert report.cost_per_accepted_task_microunits == 0.0
    assert not report.evidence_gain_per_thousand_input_tokens_ratio.defined
    assert report.evidence_gain_per_thousand_input_tokens == 0.0


def test_aggregation_rejects_duplicate_or_double_accepted_receipts() -> None:
    cold = _fixtures()["cold"]
    with pytest.raises(EfficiencyValidationError, match="duplicate receipt"):
        aggregate_efficiency_receipts((cold, cold))

    second = replace(
        _fixtures()["warm"],
        task_reference=cold.task_reference,
    )
    with pytest.raises(
        EfficiencyValidationError, match="only one accepted receipt"
    ):
        aggregate_efficiency_receipts((cold, second))


def test_paired_report_measures_only_terminal_accepted_tasks_and_charges_attempts() -> None:
    fixtures = _fixtures()
    baseline_terminal = fixtures["cold"]
    failed_attempt = replace(
        fixtures["failed"],
        task_reference=baseline_terminal.task_reference,
        goal_reference=baseline_terminal.goal_reference,
        repository_tree_digest=baseline_terminal.repository_tree_digest,
        policy_digest=baseline_terminal.policy_digest,
    )
    failed_only = replace(
        fixtures["failed"],
        task_reference="task:failed-only",
    )
    candidate_terminal = replace(
        baseline_terminal,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
    )

    report = build_paired_efficiency_report(
        (failed_attempt, baseline_terminal, failed_only),
        (candidate_terminal, failed_only),
    )

    assert report.schema == PAIRED_EFFICIENCY_REPORT_SCHEMA
    assert report.paired_task_count == 1
    assert report.population_complete
    assert report.terminal_accepted_work_accounting_proven
    # The compact report is a detached calculation.  Only the replayable
    # source-population witness below may claim ASI-G093.
    assert not report.evidence_claim_references
    case = report.cases[0]
    assert case.schema == PAIRED_EFFICIENCY_CASE_SCHEMA
    assert case.task_reference == baseline_terminal.task_reference
    assert len(case.baseline_receipt_ids) == 2
    assert failed_attempt.receipt_id in case.baseline_receipt_ids
    assert failed_only.task_reference not in {
        item.task_reference for item in report.cases
    }
    assert case.baseline_input_tokens == (
        failed_attempt.input_tokens + baseline_terminal.input_tokens
    )
    assert case.candidate_input_tokens == candidate_terminal.input_tokens
    assert report.median_input_token_reduction_bps == 7_500
    assert report.token_gate_passed
    assert report.coverage_gate_passed
    assert report.passed


def test_terminal_accepted_work_evidence_replays_complete_source_populations() -> None:
    fixtures = _fixtures()
    baseline_terminal = fixtures["cold"]
    failed_attempt = replace(
        fixtures["failed"],
        task_reference=baseline_terminal.task_reference,
        goal_reference=baseline_terminal.goal_reference,
        repository_tree_digest=baseline_terminal.repository_tree_digest,
        policy_digest=baseline_terminal.policy_digest,
    )
    failed_only = replace(
        fixtures["failed"],
        task_reference="task:failed-only",
        goal_reference=baseline_terminal.goal_reference,
        repository_tree_digest=baseline_terminal.repository_tree_digest,
        policy_digest=baseline_terminal.policy_digest,
    )
    candidate_terminal = replace(
        baseline_terminal,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
    )

    evidence = build_terminal_accepted_work_evidence(
        (failed_only, baseline_terminal, failed_attempt),
        (candidate_terminal, failed_only),
    )

    assert evidence.schema == TERMINAL_ACCEPTED_WORK_EVIDENCE_SCHEMA
    assert evidence.proved_requirement_ids == (
        TERMINAL_ACCEPTED_WORK_EVIDENCE_ID,
    )
    assert evidence.evidence_claim_references == (
        TERMINAL_ACCEPTED_WORK_EVIDENCE_ID,
    )
    assert evidence.result == "passed"
    assert evidence.promotion_eligible
    assert evidence.source_receipt_count == 5
    assert evidence.task_references == (baseline_terminal.task_reference,)
    assert evidence.repository_tree_digest == (
        baseline_terminal.repository_tree_digest
    )
    case = evidence.paired_report.cases[0]
    assert failed_attempt.receipt_id in case.baseline_receipt_ids
    assert baseline_terminal.receipt_id in case.baseline_receipt_ids
    assert failed_only.receipt_id not in case.baseline_receipt_ids
    assert case.baseline_input_tokens == (
        failed_attempt.input_tokens + baseline_terminal.input_tokens
    )

    # Source ordering is canonical and the complete typed population survives
    # serialization so the report can be independently replayed.
    reordered = build_terminal_accepted_work_evidence(
        tuple(reversed((failed_only, baseline_terminal, failed_attempt))),
        tuple(reversed((candidate_terminal, failed_only))),
    )
    assert reordered == evidence
    assert reordered.evidence_id == evidence.evidence_id
    assert TerminalAcceptedWorkEvidence.from_json(
        evidence.to_json()
    ) == evidence
    identified = evidence.to_dict(include_evidence_id=True)
    assert TerminalAcceptedWorkEvidence.from_dict(identified) == evidence


def test_detached_or_tampered_terminal_accounting_cannot_claim_evidence() -> None:
    baseline = _fixtures()["cold"]
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
    )
    detached = build_paired_efficiency_report((baseline,), (candidate,))
    assert detached.terminal_accepted_work_accounting_proven
    assert detached.passed
    assert not detached.evidence_claim_references

    evidence = build_terminal_accepted_work_evidence(
        (baseline,),
        (candidate,),
    )
    payload = json.loads(evidence.to_json())
    payload["baseline_receipts"][0]["tokens"]["input_tokens"] += 1
    with pytest.raises(
        EfficiencyValidationError,
        match="does not match replayed source",
    ):
        TerminalAcceptedWorkEvidence.from_dict(payload)

    omitted = json.loads(evidence.to_json())
    omitted["baseline_receipts"] = []
    with pytest.raises(
        EfficiencyValidationError,
        match="requires both source arms",
    ):
        TerminalAcceptedWorkEvidence.from_dict(omitted)

    wrong_requirement = json.loads(evidence.to_json())
    wrong_requirement["requirement_id"] = "not-the-objective"
    with pytest.raises(EfficiencyValidationError, match="unexpected requirement"):
        TerminalAcceptedWorkEvidence.from_dict(wrong_requirement)


def test_terminal_accounting_evidence_rejects_unpaired_or_stale_populations() -> None:
    fixtures = _fixtures()
    baseline = fixtures["cold"]
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
    )
    candidate_only = replace(
        fixtures["warm"],
        repository_tree_digest=baseline.repository_tree_digest,
        policy_digest=baseline.policy_digest,
        goal_reference=baseline.goal_reference,
    )
    with pytest.raises(
        EfficiencyValidationError,
        match="population-complete",
    ):
        build_terminal_accepted_work_evidence(
            (baseline,),
            (candidate, candidate_only),
        )

    with pytest.raises(
        EfficiencyValidationError,
        match="repository_tree_digest",
    ):
        build_terminal_accepted_work_evidence(
            (baseline,),
            (replace(candidate, repository_tree_digest="f" * 64),),
        )


def test_paired_report_couples_token_reduction_to_required_coverage() -> None:
    baseline = _fixtures()["cold"]
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
        evidence=EvidenceDelta(
            baseline_references=("evidence:syntax",),
            terminal_references=(
                "evidence:syntax",
                "evidence:acceptance",
            ),
        ),
    )

    report = build_paired_efficiency_report(
        (baseline,),
        (candidate,),
        required_evidence_by_task={
            baseline.task_reference: (
                "evidence:syntax",
                "evidence:unit",
                "evidence:acceptance",
            )
        },
    )

    case = report.cases[0]
    assert report.median_input_token_reduction_bps == 5_000
    assert report.token_gate_passed
    assert case.baseline_coverage_bps == 10_000
    assert case.candidate_coverage_bps == 6_666
    assert not case.coverage_preserved
    assert not case.candidate_has_full_required_coverage
    assert report.coverage_regression_count == 1
    assert report.candidate_incomplete_coverage_count == 1
    assert not report.coverage_gate_passed
    assert not report.passed


def test_paired_report_uses_median_same_task_reduction() -> None:
    baseline = _fixtures()["cold"]
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
    )
    template = build_paired_efficiency_report(
        (baseline,),
        (candidate,),
    ).cases[0]
    report = PairedEfficiencyReport(
        cases=(
            replace(
                template,
                task_reference="task:paired-a",
                baseline_input_tokens=100,
                candidate_input_tokens=100,
            ),
            replace(
                template,
                task_reference="task:paired-b",
                baseline_input_tokens=1_000,
                candidate_input_tokens=650,
            ),
            replace(
                template,
                task_reference="task:paired-c",
                baseline_input_tokens=10_000,
                candidate_input_tokens=9_000,
            ),
        ),
    )

    # The old ratio-of-medians calculation was 35%; preserving each pair
    # reveals that the median task improved by only 10%.
    assert report.median_baseline_input_tokens == 1_000
    assert report.median_candidate_input_tokens == 650
    assert report.median_input_token_reduction_bps == 1_000
    assert not report.token_gate_passed
    assert not report.passed


def test_paired_report_discloses_population_mismatch_and_round_trips() -> None:
    fixtures = _fixtures()
    baseline = fixtures["cold"]
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
    )
    candidate_only = fixtures["warm"]

    report = build_paired_efficiency_report(
        (baseline,),
        (candidate, candidate_only),
    )

    assert report.candidate_unpaired_accepted_task_references == (
        candidate_only.task_reference,
    )
    assert not report.population_complete
    assert not report.terminal_accepted_work_accounting_proven
    assert not report.evidence_claim_references
    assert not report.passed
    assert PairedEfficiencyReport.from_json(report.to_json()) == report
    identified = report.to_dict(include_report_id=True)
    assert PairedEfficiencyReport.from_dict(identified) == report
    assert PairedEfficiencyCase.from_dict(
        report.cases[0].to_dict(include_case_id=True)
    ) == report.cases[0]

    tampered = report.to_dict()
    tampered["median_input_token_reduction_bps"] += 1
    with pytest.raises(EfficiencyValidationError, match="reduction"):
        PairedEfficiencyReport.from_dict(tampered)


def test_paired_report_rejects_unfrozen_or_ambiguous_populations() -> None:
    baseline = _fixtures()["cold"]
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
    )

    with pytest.raises(EfficiencyValidationError, match="define every paired"):
        build_paired_efficiency_report(
            (baseline,),
            (candidate,),
            required_evidence_by_task={},
        )

    with pytest.raises(EfficiencyValidationError, match="repository_tree_digest"):
        build_paired_efficiency_report(
            (baseline,),
            (
                replace(
                    candidate,
                    repository_tree_digest="f" * 64,
                ),
            ),
        )

    with pytest.raises(EfficiencyValidationError, match="only one accepted"):
        build_paired_efficiency_report(
            (baseline,),
            (candidate, replace(candidate, output_digest="e" * 64)),
        )


def test_required_context_promotion_binds_capsule_to_same_task_gate() -> None:
    result = _required_context_fixture()
    paired = _paired_required_context_report(result, verified=True)

    report = build_required_context_promotion_report(
        paired,
        {"task:required-context": (result,)},
    )

    assert report.schema == REQUIRED_CONTEXT_PROMOTION_REPORT_SCHEMA
    assert REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID == (
        COMPILER_REQUIRED_CONTEXT_ID
    )
    assert report.proof_population_complete
    assert report.coverage_requirements_consistent
    assert report.token_accounting_consistent
    assert report.typed_context_gate_passed
    assert report.paired_efficiency_gate_passed
    assert report.terminal_work_evidence == paired
    assert report.evidence_claim_references == (
        REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,
    )
    assert report.promotion_eligible

    binding = report.proof_bindings[0]
    assert binding.capsule_id == result.capsule.capsule_id
    assert binding.receipt_id == result.receipt.receipt_id
    assert binding.evidence_id == result.receipt.evidence.content_id
    assert binding.required_reference_ids == ("required",)
    assert binding.required_coverage_ids == ("coverage:required",)
    assert binding.required_references_preserved
    assert binding.required_coverage_preserved
    assert RequiredContextProofBinding.from_context_compile_result(
        "task:required-context",
        result,
    ) == binding

    assert RequiredContextPromotionReport.from_json(
        report.to_json()
    ) == report
    assert RequiredContextPromotionReport.from_dict(
        report.to_dict(include_report_id=True)
    ) == report


def test_required_context_promotion_fails_closed_for_gap_or_forgery() -> None:
    result = _required_context_fixture()
    paired = _paired_required_context_report(result)

    detached = build_required_context_promotion_report(
        paired,
        {"task:required-context": (result,)},
    )
    assert detached.typed_context_gate_passed
    assert not detached.paired_efficiency_gate_passed
    assert not detached.evidence_claim_references
    assert not detached.promotion_eligible

    missing = build_required_context_promotion_report(paired, {})
    assert missing.missing_proof_task_references == (
        "task:required-context",
    )
    assert not missing.typed_context_gate_passed
    assert not missing.evidence_claim_references
    assert not missing.promotion_eligible

    unexplained_tokens = replace(
        paired,
        cases=(
            replace(
                paired.cases[0],
                candidate_input_tokens=(
                    paired.cases[0].candidate_input_tokens + 1
                ),
            ),
        ),
    )
    inconsistent = build_required_context_promotion_report(
        unexplained_tokens,
        {"task:required-context": (result,)},
    )
    assert not inconsistent.token_accounting_consistent
    assert not inconsistent.evidence_claim_references
    assert not inconsistent.promotion_eligible

    assert result.receipt.evidence is not None
    with pytest.raises(EfficiencyValidationError, match="artifact digest"):
        forged_result = replace(
            result,
            receipt=replace(
                result.receipt,
                evidence=replace(
                    result.receipt.evidence,
                    artifact_digest="sha256:" + "0" * 64,
                ),
            ),
        )

    with pytest.raises(EfficiencyValidationError, match="coverage"):
        build_required_context_promotion_report(
            replace(
                paired,
                cases=(
                    replace(
                        paired.cases[0],
                        required_evidence_references=("coverage:other",),
                        baseline_covered_evidence_references=(
                            "coverage:other",
                        ),
                        candidate_covered_evidence_references=(
                            "coverage:other",
                        ),
                    ),
                ),
            ),
            {"task:required-context": (result,)},
        )


def test_delta_retry_promotion_binds_typed_result_to_same_task_gate() -> None:
    result = _delta_retry_fixture()
    receipt = result.receipt
    paired = _paired_delta_report(result, verified=True)

    report = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (result,)},
    )

    assert report.schema == DELTA_RETRY_PROMOTION_REPORT_SCHEMA
    assert DELTA_RETRY_CONTEXT_EVIDENCE_ID == DELTA_RETRY_EVIDENCE_ID
    assert paired.paired_report.median_input_token_reduction_bps >= 3_500
    assert paired.paired_report.coverage_gate_passed
    assert report.proof_population_complete
    assert report.token_accounting_consistent
    assert report.median_delta_input_token_reduction_bps >= 3_500
    assert report.typed_delta_gate_passed
    assert report.paired_efficiency_gate_passed
    assert report.terminal_work_evidence == paired
    assert report.evidence_claim_references == (
        DELTA_RETRY_EVIDENCE_ID,
    )
    assert report.promotion_eligible

    binding = report.proof_bindings[0]
    assert binding.parent_context_capsule == result.parent_capsule
    assert binding.context_delta_capsule == result.delta_capsule
    assert (
        binding.reconstructed_context_capsule
        == result.reconstructed_capsule
    )
    assert binding.receipt_id == receipt.receipt_id
    assert binding.evidence_id == receipt.evidence.content_id
    assert binding.parent_capsule_id == receipt.parent_capsule_id
    assert binding.delta_capsule_id == receipt.delta_capsule_id
    assert binding.reconstructed_capsule_id == (
        receipt.reconstructed_capsule_id
    )
    assert binding.required_fields == (
        "acceptance",
        "authority",
        "goal",
        "scope",
    )
    assert binding.coverage_preserved
    assert DeltaRetryProofBinding.from_context_delta_result(
        "task:delta-retry",
        result,
    ) == binding

    verifiers = {receipt.receipt_id: result.verifier}
    with pytest.raises(
        EfficiencyValidationError,
        match="provider_tokens_verified",
    ):
        DeltaRetryPromotionReport.from_json(report.to_json())
    assert DeltaRetryPromotionReport.from_json(
        report.to_json(),
        verifiers_by_receipt=verifiers,
    ) == report
    identified = report.to_dict(include_report_id=True)
    assert DeltaRetryPromotionReport.from_dict(
        identified,
        verifiers_by_receipt=verifiers,
    ) == report


def test_delta_retry_promotion_fails_closed_for_missing_stale_or_unverified_proof() -> None:
    result = _delta_retry_fixture()
    receipt = result.receipt
    paired = _paired_delta_report(result)

    detached = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (result,)},
    )
    assert detached.typed_delta_gate_passed
    assert not detached.paired_efficiency_gate_passed
    assert not detached.evidence_claim_references
    assert not detached.promotion_eligible

    incomplete = build_delta_retry_promotion_report(paired, {})
    assert incomplete.missing_proof_task_references == (
        "task:delta-retry",
    )
    assert not incomplete.typed_delta_gate_passed
    assert not incomplete.evidence_claim_references
    assert not incomplete.promotion_eligible

    incomplete_population = build_delta_retry_promotion_report(
        replace(
            paired,
            candidate_unpaired_accepted_task_references=("task:unpaired",),
        ),
        {"task:delta-retry": (result,)},
    )
    assert incomplete_population.typed_delta_gate_passed
    assert not incomplete_population.paired_efficiency_gate_passed
    assert not incomplete_population.evidence_claim_references
    assert not incomplete_population.promotion_eligible

    with pytest.raises(EfficiencyValidationError, match="outside"):
        build_delta_retry_promotion_report(
            paired,
            {"task:stale": (result,)},
        )
    forged_evidence = replace(
        receipt.evidence,
        artifact_digest="sha256:" + "0" * 64,
    )
    forged_receipt = replace(receipt, evidence=forged_evidence)
    with pytest.raises(
        EfficiencyValidationError,
        match="ContextDeltaResult",
    ):
        build_delta_retry_promotion_report(
            paired,
            {"task:delta-retry": (forged_receipt,)},
        )
    with pytest.raises(
        EfficiencyValidationError,
        match="provider-token verifier",
    ):
        build_delta_retry_promotion_report(
            paired,
            {"task:delta-retry": (replace(result, verifier=None),)},
        )
    assert receipt.evidence is not None
    forged_token_evidence = replace(
        receipt.evidence,
        delta_tokens=receipt.delta_tokens - 1,
    )
    forged_token_receipt = replace(
        receipt,
        delta_tokens=receipt.delta_tokens - 1,
        evidence=forged_token_evidence,
    )
    with pytest.raises(
        EfficiencyValidationError,
        match="not reproducible",
    ):
        replace(result, receipt=forged_token_receipt)
    with pytest.raises(EfficiencyValidationError, match="objective"):
        wrong_objective_paired = replace(
            paired,
            cases=(
                replace(
                    paired.cases[0],
                    goal_reference="ASI-G091",
                ),
            ),
        )
        build_delta_retry_promotion_report(
            wrong_objective_paired,
            {"task:delta-retry": (result,)},
        )

    forged = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (result,)},
    ).to_dict()
    forged["typed_delta_gate_passed"] = False
    with pytest.raises(EfficiencyValidationError, match="typed_delta"):
        DeltaRetryPromotionReport.from_dict(
            forged,
            verifiers_by_receipt={receipt.receipt_id: result.verifier},
        )

    verified_report = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (result,)},
    )
    unverified_binding = replace(
        verified_report.proof_bindings[0],
        verifier=None,
    )
    unverified_report = replace(
        verified_report,
        proof_bindings=(unverified_binding,),
    )
    assert not unverified_binding.provider_tokens_verified
    assert not unverified_report.typed_delta_gate_passed
    assert not unverified_report.evidence_claim_references
    assert not unverified_report.promotion_eligible

    forged_receipt = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (result,)},
    ).to_dict()
    forged_receipt["proof_bindings"][0]["context_delta_receipt"][
        "delta_tokens"
    ] += 1
    with pytest.raises(
        EfficiencyValidationError,
        match="bound|identity|not reproducible",
    ):
        DeltaRetryPromotionReport.from_dict(
            forged_receipt,
            verifiers_by_receipt={receipt.receipt_id: result.verifier},
        )

    forged_parent = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (result,)},
    ).to_dict()
    forged_parent["proof_bindings"][0]["parent_context_capsule"]["goal"][
        "summary"
    ] = "forged parent"
    with pytest.raises(
        EfficiencyValidationError,
        match="parent|reconstruct|identity",
    ):
        DeltaRetryPromotionReport.from_dict(
            forged_parent,
            verifiers_by_receipt={receipt.receipt_id: result.verifier},
        )


def test_delta_retry_gate_accepts_requested_only_and_enforces_35_percent() -> None:
    requested_result = _delta_retry_fixture(requested_only=True)
    paired = _paired_delta_report(requested_result, verified=True)
    report = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (requested_result,)},
    )

    binding = report.proof_bindings[0]
    assert not binding.changed_reference_ids
    assert binding.requested_reference_ids == ("optional-0",)
    assert binding.retained_reference_ids
    assert report.promotion_eligible

    stricter_threshold = binding.input_token_reduction_bps + 1
    inefficient_paired_report = replace(
        paired.paired_report,
        minimum_input_token_reduction_bps=stricter_threshold,
    )
    inefficient_paired = replace(
        paired,
        paired_report=inefficient_paired_report,
    )
    inefficient = build_delta_retry_promotion_report(
        inefficient_paired,
        {"task:delta-retry": (requested_result,)},
    )

    assert (
        inefficient.median_delta_input_token_reduction_bps
        < stricter_threshold
    )
    assert not inefficient.typed_delta_gate_passed
    assert not inefficient.evidence_claim_references
    assert not inefficient.promotion_eligible

    unexplained = build_delta_retry_promotion_report(
        replace(
            paired.paired_report,
            cases=(
                replace(
                    paired.paired_report.cases[0],
                    candidate_input_tokens=(
                        paired.paired_report.cases[
                            0
                        ].candidate_input_tokens
                        + 1
                    ),
                ),
            ),
        ),
        {"task:delta-retry": (requested_result,)},
    )
    assert not unexplained.token_accounting_consistent
    assert not unexplained.evidence_claim_references
    assert not unexplained.promotion_eligible


def test_delta_retry_gate_rejects_unattributed_lifecycle_input() -> None:
    result = _delta_retry_fixture()
    paired = _paired_delta_report(result, common_input_tokens=275)

    report = build_delta_retry_promotion_report(
        paired,
        {"task:delta-retry": (result,)},
    )

    case = paired.cases[0]
    binding = report.proof_bindings[0]
    assert case.baseline_input_tokens - binding.full_replay_tokens == 275
    assert case.candidate_input_tokens - binding.delta_tokens == 275
    assert not report.token_accounting_consistent
    assert not report.evidence_claim_references
    assert not report.promotion_eligible


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: TokenUsage(input_tokens=1, reused_tokens=2),
            "reused_tokens",
        ),
        (
            lambda: TokenUsage(input_tokens=True),
            "integer",
        ),
        (
            lambda: TokenUsage(input_tokens=MAX_TOKENS + 1),
            "between",
        ),
        (
            lambda: StageTiming(StageName.ANALYSIS, MAX_DURATION_MS + 1),
            "between",
        ),
        (
            lambda: CacheObservation(
                "analysis",
                CacheDisposition.MISS,
                "0" * 64,
                bytes_reused=1,
            ),
            "only a cache hit",
        ),
        (
            lambda: ArtifactReference("artifact:a", "not-a-digest", "patch"),
            "SHA-256",
        ),
        (
            lambda: WorkCost(
                status=WorkStatus.PASSED,
                operation_count=1,
            ),
            "reference",
        ),
        (
            lambda: ChangedScope(paths=("../escape.py",)),
            "repository-relative",
        ),
        (
            lambda: TerminalAcceptance(
                TerminalOutcome.FAILED,
                ("failed",),
                "0" * 64,
            ),
            "non-accepted",
        ),
        (
            lambda: TerminalAcceptance(
                TerminalOutcome.ACCEPTED,
                ("accepted",),
            ),
            "acceptance_digest",
        ),
    ],
)
def test_component_invalid_states_fail_closed(factory: object, message: str) -> None:
    with pytest.raises(EfficiencyValidationError, match=message):
        factory()  # type: ignore[operator]


def test_receipt_rejects_cross_field_invalid_states() -> None:
    fixtures = _fixtures()
    cold = fixtures["cold"]

    with pytest.raises(EfficiencyValidationError, match="passed validation"):
        replace(
            cold,
            validation=WorkCost(
                status=WorkStatus.FAILED,
                operation_count=1,
            ),
        )

    with pytest.raises(EfficiencyValidationError, match="attempt must equal"):
        replace(cold, attempt=2)

    with pytest.raises(EfficiencyValidationError, match="unique stage"):
        replace(cold, stages=cold.stages + (cold.stages[0],))

    with pytest.raises(EfficiencyValidationError, match="queue_delay"):
        replace(cold, queue_delay_ms=cold.elapsed_ms + 1)

    with pytest.raises(EfficiencyValidationError, match="input_digest"):
        replace(cold, input_digest="")

    with pytest.raises(EfficiencyValidationError, match="cold scenario"):
        replace(
            cold,
            cache_observations=(
                CacheObservation(
                    "analysis",
                    CacheDisposition.HIT,
                    "0" * 64,
                    bytes_reused=1,
                ),
            ),
        )

    with pytest.raises(
        EfficiencyValidationError, match="retry token accounting"
    ):
        replace(
            fixtures["repaired"],
            retries=(
                replace(
                    fixtures["repaired"].retries[0],
                    tokens=TokenUsage(
                        fixtures["repaired"].input_tokens + 1,
                        0,
                        0,
                    ),
                ),
            ),
        )


def test_collection_and_text_bounds_are_enforced_before_serialization() -> None:
    cold = _fixtures()["cold"]

    with pytest.raises(EfficiencyValidationError, match="stage"):
        replace(
            cold,
            scenario=EfficiencyScenario.OBSERVED,
            stages=tuple(
                StageTiming(StageName.ANALYSIS, index)
                for index in range(MAX_STAGES + 1)
            ),
        )

    with pytest.raises(EfficiencyValidationError, match="paths"):
        ChangedScope(
            paths=tuple(
                f"src/file_{index}.py"
                for index in range(MAX_CHANGED_PATHS + 1)
            )
        )

    with pytest.raises(EfficiencyValidationError, match="byte bound"):
        ArtifactReference(
            "a" * (MAX_TEXT_BYTES + 1),
            "0" * 64,
            "patch",
        )

    with pytest.raises(EfficiencyValidationError, match="artifacts"):
        replace(
            cold,
            scenario=EfficiencyScenario.OBSERVED,
            artifacts=tuple(
                ArtifactReference(
                    f"artifact:{index}",
                    f"{index:064x}",
                    "patch",
                )
                for index in range(MAX_ARTIFACT_REFERENCES + 1)
            ),
        )

    with pytest.raises(EfficiencyValidationError, match="terminal_references"):
        EvidenceDelta(
            terminal_references=tuple(
                f"evidence:{index}"
                for index in range(MAX_EVIDENCE_REFERENCES + 1)
            )
        )


def test_retry_records_are_contiguous_compact_and_accounted_in_totals() -> None:
    repaired = _fixtures()["repaired"]
    retry = repaired.retries[0]

    assert retry.attempt == 2
    assert retry.reason_code == "validation_failure"
    assert retry.diagnostic_digest.startswith("sha256:")
    assert retry.delta_context_digest.startswith("sha256:")
    assert "diagnostic" not in retry.to_dict()
    assert retry.tokens.input_tokens <= repaired.tokens.input_tokens

    with pytest.raises(EfficiencyValidationError, match="contiguous"):
        replace(
            repaired,
            attempt=2,
            retries=(replace(retry, attempt=3),),
        )
