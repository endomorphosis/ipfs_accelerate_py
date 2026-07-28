from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.analyzer_health import (
    AnalyzerHealthReport,
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
)
from ipfs_accelerate_py.agent_supervisor.context_compiler import (
    DELTA_RETRY_ACCEPTANCE_CRITERIA,
    DELTA_RETRY_EVIDENCE_ID,
    DELTA_RETRY_OBJECTIVE_ID,
    REQUIRED_CONTEXT_ACCEPTANCE_CRITERIA,
    REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID as COMPILER_REQUIRED_CONTEXT_ID,
    ContextCompiler,
)
from ipfs_accelerate_py.agent_supervisor.context_contracts import (
    ContextBudget,
    ContextReference,
    ContextTier,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.goal_coverage import (
    AcceptanceCoverage,
    CoverageStatus,
    GoalCoverageMap,
)
from ipfs_accelerate_py.agent_supervisor.scan_receipts import (
    ExhaustionBinding,
    evaluate_exhaustion_quorum,
)
from ipfs_accelerate_py.agent_supervisor.supervisor_efficiency_metrics import (
    DELTA_RETRY_CONTEXT_EVIDENCE_ID,
    DELTA_RETRY_PROMOTION_REPORT_SCHEMA,
    EFFICIENCY_CONTRACT_VERSION,
    EFFICIENCY_EVIDENCE_PRODUCERS,
    EFFICIENCY_RECEIPT_SCHEMA,
    EFFICIENCY_REPORT_SCHEMA,
    PAIRED_EFFICIENCY_CASE_SCHEMA,
    PAIRED_EFFICIENCY_REPORT_SCHEMA,
    REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,
    REQUIRED_CONTEXT_PROMOTION_REPORT_SCHEMA,
    TERMINAL_ACCEPTED_WORK_ACCEPTANCE_CRITERIA,
    TERMINAL_ACCEPTED_WORK_EVIDENCE_ID,
    TERMINAL_ACCEPTED_WORK_EVIDENCE_SCHEMA,
    TERMINAL_ACCEPTED_WORK_OBJECTIVE_ID,
    TOKEN_EFFICIENCY_ACCEPTANCE_CRITERIA,
    TOKEN_EFFICIENCY_CHILD_GOAL_IDS,
    TOKEN_EFFICIENCY_OBJECTIVE_ID,
    TOKEN_EFFICIENCY_OBJECTIVE_REVISION,
    TOKEN_EFFICIENCY_PRODUCING_TASK_IDS,
    MAX_ARTIFACT_REFERENCES,
    MAX_CHANGED_PATHS,
    MAX_DURATION_MS,
    MAX_EVIDENCE_REFERENCES,
    MAX_SERIALIZED_REPORT_BYTES,
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
    evaluate_token_efficiency_completion,
    verify_terminal_accepted_work_evidence,
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
    assert reordered.report_id == evidence.report_id
    assert (
        reordered.benchmark_input_digest
        == evidence.benchmark_input_digest
    )
    assert (
        len(evidence.to_json().encode("utf-8"))
        <= MAX_SERIALIZED_REPORT_BYTES
    )
    assert TerminalAcceptedWorkEvidence.from_json(
        evidence.to_json()
    ) == evidence
    identified = evidence.to_dict(include_evidence_id=True)
    assert TerminalAcceptedWorkEvidence.from_dict(identified) == evidence

    substituted_candidate = replace(
        candidate_terminal,
        tokens=TokenUsage(input_tokens=2_001, output_tokens=300),
    )
    substituted = build_terminal_accepted_work_evidence(
        (failed_only, baseline_terminal, failed_attempt),
        (substituted_candidate, failed_only),
    )
    assert substituted.report_id != evidence.report_id
    assert (
        substituted.benchmark_input_digest
        != evidence.benchmark_input_digest
    )
    assert substituted.evidence_id != evidence.evidence_id


def test_terminal_evidence_verifier_requires_the_independent_complete_cohort() -> None:
    fixtures = _fixtures()
    terminal = fixtures["cold"]
    failed_attempt = replace(
        fixtures["failed"],
        task_reference=terminal.task_reference,
        goal_reference=terminal.goal_reference,
        repository_tree_digest=terminal.repository_tree_digest,
        policy_digest=terminal.policy_digest,
    )
    candidate = replace(
        terminal,
        tokens=TokenUsage(input_tokens=2_000, output_tokens=300),
        inference_cost_microunits=2_600,
    )
    baseline_population = (failed_attempt, terminal)
    evidence = build_terminal_accepted_work_evidence(
        baseline_population,
        (candidate,),
    )
    expected_binding = {
        "expected_goal_reference": terminal.goal_reference,
        "expected_repository_tree_digest": terminal.repository_tree_digest,
        "expected_policy_digest": terminal.policy_digest,
    }

    # Independent population enumeration may use a different order or a
    # serialized evidence value without changing the verified identity.
    assert verify_terminal_accepted_work_evidence(
        evidence.to_dict(include_evidence_id=True),
        tuple(reversed(baseline_population)),
        (candidate,),
        **expected_binding,
    ) == evidence

    # The embedded report remains internally replayable after an upstream
    # enumerator accidentally drops an attempt.  The external-population
    # verifier is the boundary that detects that omission.
    with pytest.raises(
        EfficiencyValidationError,
        match="independently supplied source populations",
    ):
        verify_terminal_accepted_work_evidence(
            evidence,
            (terminal,),
            (candidate,),
            **expected_binding,
        )

    with pytest.raises(
        EfficiencyValidationError,
        match="duplicate receipt identit",
    ):
        verify_terminal_accepted_work_evidence(
            evidence,
            (failed_attempt, terminal, failed_attempt),
            (candidate,),
            **expected_binding,
        )

    changed_attempt = replace(
        failed_attempt,
        tokens=TokenUsage(
            input_tokens=failed_attempt.input_tokens + 1,
            output_tokens=failed_attempt.output_tokens,
        ),
    )
    with pytest.raises(
        EfficiencyValidationError,
        match="independently supplied source populations",
    ):
        verify_terminal_accepted_work_evidence(
            evidence,
            (changed_attempt, terminal),
            (candidate,),
            **expected_binding,
        )

    with pytest.raises(
        EfficiencyValidationError,
        match="completion gate's expected goal",
    ):
        verify_terminal_accepted_work_evidence(
            evidence,
            baseline_population,
            (candidate,),
            **{
                **expected_binding,
                "expected_policy_digest": "f" * 64,
            },
        )


def test_g093_completion_requires_current_cohort_health_quorum_and_two_phases() -> None:
    fixtures = _fixtures()
    terminal = fixtures["cold"]
    failed_attempt = replace(
        fixtures["failed"],
        task_reference=terminal.task_reference,
        goal_reference=terminal.goal_reference,
        repository_tree_digest=terminal.repository_tree_digest,
        policy_digest=terminal.policy_digest,
    )
    # Accounting is authoritative independently of the separate 35% token
    # promotion gate.
    candidate = replace(
        terminal,
        tokens=TokenUsage(input_tokens=6_000, output_tokens=300),
        inference_cost_microunits=7_500,
    )
    baseline_population = (failed_attempt, terminal)
    candidate_population = (candidate,)
    terminal_evidence = build_terminal_accepted_work_evidence(
        baseline_population,
        candidate_population,
    )
    assert not terminal_evidence.promotion_eligible

    now = datetime(2026, 7, 24, 15, 0, tzinfo=timezone.utc)
    repository_id = "repo:agent-supervisor"
    objective_revision = "sha256:" + "a" * 64
    command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_efficiency_metrics.py "
        "test/api/test_agent_supervisor_context_compiler.py "
        "test/api/test_agent_supervisor_context_delta.py -q"
    )
    validations = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-074",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": terminal.repository_tree_digest,
                "command": command,
                "terminal_evidence_id": terminal_evidence.evidence_id,
            },
            validation_passed=True,
            repository_id=repository_id,
            repository_tree=terminal.repository_tree_digest,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-074:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            TERMINAL_ACCEPTED_WORK_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "repository_tree": terminal.repository_tree_digest,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "supervisor_efficiency_metrics.py"
                ),
                "validation": (
                    "test/api/test_agent_supervisor_efficiency_metrics.py"
                ),
                "validation_receipt_ids": [
                    f"validation:asi-074:{index}"
                ],
            }
            for index, criterion in enumerate(
                TERMINAL_ACCEPTED_WORK_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    analyzer_version = "terminal-accounting-completion@1"
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
        "analyzer_version": analyzer_version,
    }
    binding = {
        "repository_id": repository_id,
        "tree_id": terminal.repository_tree_digest,
        "objective_id": TERMINAL_ACCEPTED_WORK_OBJECTIVE_ID,
        "objective_revision": objective_revision,
        "goal_reference": terminal.goal_reference,
        "policy_revision": terminal.policy_digest,
        "requirement_id": TERMINAL_ACCEPTED_WORK_EVIDENCE_ID,
        "terminal_evidence_id": terminal_evidence.evidence_id,
        "paired_report_id": terminal_evidence.report_id,
        "benchmark_input_digest": (
            terminal_evidence.benchmark_input_digest
        ),
        "analyzer_version": analyzer_version,
        "configuration_revision": "sha256:completion-config",
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-074-exhaustive-a",
                "evidence_channel": "receipt-population",
                "receipt_cid": "scan:asi-074:exhaustive-a",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-074-population",
                "implementation": "terminal-receipt-population",
                "child_receipt_binding": "cohort:asi-074:population",
                "child_receipt_sha256": "sha256:" + "1" * 64,
                "aggregate_tree_binding": terminal.repository_tree_digest,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-074-exhaustive-b",
                "evidence_channel": "accounting-lifecycle",
                "receipt_cid": "scan:asi-074:exhaustive-b",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-074-accounting",
                "implementation": "terminal-accounting-lifecycle",
                "child_receipt_binding": "cohort:asi-074:accounting",
                "child_receipt_sha256": "sha256:" + "2" * 64,
                "aggregate_tree_binding": terminal.repository_tree_digest,
                "finished_at": now.isoformat(),
            },
        ],
    }
    expected = {
        "expected_repository_id": repository_id,
        "expected_goal_reference": terminal.goal_reference,
        "expected_repository_tree_digest": (
            terminal.repository_tree_digest
        ),
        "expected_policy_digest": terminal.policy_digest,
        "expected_objective_revision": objective_revision,
    }
    values = {
        **expected,
        "evidence": validations,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    provisional = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert provisional.acceptance_criteria == (
        TERMINAL_ACCEPTED_WORK_ACCEPTANCE_CRITERIA
    )
    assert provisional.gate is not None and provisional.gate.passed
    assert "provisional_transition_required" in provisional.reason_codes

    verified = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    typed_coverage = GoalCoverageMap(
        criteria=[
            AcceptanceCoverage(
                criterion_id=f"ASI-G093:{index}",
                goal_id=TERMINAL_ACCEPTED_WORK_OBJECTIVE_ID,
                criterion=criterion,
                status=CoverageStatus.VERIFIED,
                changed_files=[
                    "ipfs_accelerate_py/agent_supervisor/"
                    "supervisor_efficiency_metrics.py"
                ],
                validation_receipt_ids=[
                    f"validation:asi-074:{index}"
                ],
            )
            for index, criterion in enumerate(
                TERMINAL_ACCEPTED_WORK_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
        edges=[],
        receipts=[],
        finding_assignments=[],
        registered_goal_ids=[TERMINAL_ACCEPTED_WORK_OBJECTIVE_ID],
        evaluated_at=now.isoformat(),
        repository_tree=terminal.repository_tree_digest,
    )
    typed_health = AnalyzerHealthReport(
        status=AnalyzerHealthStatus.HEALTHY,
        reasons=(),
        thresholds=AnalyzerHealthThresholds(),
        metrics={
            "objective_id": TERMINAL_ACCEPTED_WORK_OBJECTIVE_ID,
            "repository_tree": terminal.repository_tree_digest,
        },
    )
    typed_binding = ExhaustionBinding(
        repository_id=repository_id,
        tree_id=terminal.repository_tree_digest,
        analyzer_version=analyzer_version,
        configuration_revision=binding["configuration_revision"],
        objective_revision=objective_revision,
    )
    typed_quorum = evaluate_exhaustion_quorum(
        (
            {
                "receipt_cid": "scan:asi-074:typed-population",
                "terminal_reason": "exhausted",
                "scan_mode": "exhaustive",
                "finished_at": now.isoformat(),
                "metadata": {
                    "analyzer_health": {"status": "healthy"},
                    "coverage_complete": True,
                    "evidence_channel": "typed-receipt-population",
                },
            },
            {
                "receipt_cid": "scan:asi-074:typed-accounting",
                "terminal_reason": "exhausted",
                "scan_mode": "audit",
                "finished_at": now.isoformat(),
                "metadata": {
                    "analyzer_health": {"status": "healthy"},
                    "coverage_complete": True,
                    "evidence_channel": "typed-accounting-lifecycle",
                },
            },
        ),
        binding=typed_binding,
        required_members=2,
    )
    assert typed_quorum.satisfied
    typed_proof = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "coverage": typed_coverage,
            "analyzer_health": typed_health,
            "exhaustion_quorum": typed_quorum,
        },
    )
    assert typed_proof.state is GoalState.VERIFIED_COMPLETE
    assert typed_proof.gate is not None and typed_proof.gate.passed

    # The completion bridge independently enumerates the cohort; an omitted
    # charged attempt cannot be hidden by an otherwise valid artifact.
    incomplete = terminal_evidence.evaluate_objective_completion(
        (terminal,),
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert not incomplete.verified
    assert "coverage_unverified" in incomplete.reason_codes

    reordered = terminal_evidence.evaluate_objective_completion(
        tuple(reversed(baseline_population)),
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert reordered.verified

    missing_validation = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": validations[:-1]},
    )
    assert not missing_validation.verified
    assert (
        TERMINAL_ACCEPTED_WORK_ACCEPTANCE_CRITERIA[-1]
        in missing_validation.missing_criteria
    )

    failed_validation = replace(
        validations[0],
        validation_passed=False,
        validation_receipt={
            "status": "failed",
            "tree_id": terminal.repository_tree_digest,
            "command": command,
        },
        provenance_cid="validation:asi-074:failed",
    )
    rejected_validation = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "evidence": (failed_validation, *validations[1:]),
        },
    )
    assert not rejected_validation.verified
    assert "failed_validation" in rejected_validation.reason_codes

    stale_validation = replace(
        validations[0],
        observed_at=now - timedelta(seconds=301),
        provenance_cid="validation:asi-074:stale",
    )
    rejected_stale = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "evidence": (stale_validation, *validations[1:]),
        },
    )
    assert not rejected_stale.verified
    assert "stale_evidence" in rejected_stale.reason_codes

    # Exact criterion mapping is required: duplicated rows cannot substitute
    # for a missing mandatory criterion.
    duplicate_rows = {
        **coverage,
        "criteria": [
            *coverage["criteria"][:-1],
            coverage["criteria"][0],
        ],
    }
    unmapped = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "coverage": duplicate_rows},
    )
    assert not unmapped.verified
    assert "coverage_missing" in unmapped.reason_codes

    for invalid_coverage in (
        {
            **coverage,
            "criteria": [
                {
                    **coverage["criteria"][0],
                    "validation_receipt_ids": ["validation:foreign"],
                },
                *coverage["criteria"][1:],
            ],
        },
        {
            **coverage,
            "criteria": [
                {
                    **coverage["criteria"][0],
                    "validation_receipt_ids": [
                        "validation:asi-074:2"
                    ],
                },
                *coverage["criteria"][1:],
            ],
        },
        {
            **coverage,
            "repository_tree": "sha256:" + "0" * 64,
        },
        {
            **coverage,
            "evaluated_at": (
                now - timedelta(seconds=301)
            ).isoformat(),
        },
    ):
        rejected_coverage = (
            terminal_evidence.evaluate_objective_completion(
                baseline_population,
                candidate_population,
                current_state=GoalState.PROVISIONALLY_COMPLETE,
                **{**values, "coverage": invalid_coverage},
            )
        )
        assert not rejected_coverage.verified
        assert any(
            code in rejected_coverage.reason_codes
            for code in (
                "coverage_unverified",
                "coverage_tree_mismatch",
                "coverage_stale",
            )
        )

    unsafe = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "analyzer_health": {
                "status": "healthy",
                "healthy": True,
                "analyzer_version": analyzer_version,
            },
        },
    )
    assert not unsafe.verified
    assert "analyzer_unhealthy" in unsafe.reason_codes

    # A submitted quorum cannot weaken the trusted configured count or reuse
    # an identity/channel to manufacture independence.
    weak_quorums = (
        {
            **quorum,
            "required_members": 1,
            "member_count": 1,
            "members": quorum["members"][:1],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "receipt_cid": "scan:asi-074:exhaustive-a",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "member_id": "asi-074-exhaustive-a",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "conclusive": False,
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "scan_mode": "audit"},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "healthy": False},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "safe_for_completion_reasoning": False,
                },
            ],
        },
        {
            **quorum,
            "binding": {
                **binding,
                "terminal_evidence_id": "sha256:detached",
            },
        },
        {
            **quorum,
            "members": [
                {
                    **quorum["members"][0],
                    "finished_at": (
                        now - timedelta(seconds=301)
                    ).isoformat(),
                },
                quorum["members"][1],
            ],
        },
    )
    for weak_quorum in weak_quorums:
        rejected = terminal_evidence.evaluate_objective_completion(
            baseline_population,
            candidate_population,
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "exhaustion_quorum": weak_quorum},
        )
        assert not rejected.verified
        assert any(
            code.startswith("exhaustion_quorum")
            for code in rejected.reason_codes
        )

    configured_three = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        required_exhaustive_receipts=3,
        **values,
    )
    assert not configured_three.verified
    assert any(
        code.startswith("exhaustion_quorum")
        for code in configured_three.reason_codes
    )

    wrong_tree = terminal_evidence.evaluate_objective_completion(
        baseline_population,
        candidate_population,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "expected_repository_tree_digest": "sha256:" + "0" * 64,
        },
    )
    assert not wrong_tree.verified
    assert not wrong_tree.tasks_complete


def test_terminal_accounting_proof_is_independent_of_promotion_gates() -> None:
    baseline = _fixtures()["cold"]
    # More candidate input fails the token-reduction gate, while retaining
    # identical accepted-task and evidence populations.
    candidate = replace(
        baseline,
        tokens=TokenUsage(input_tokens=5_000, output_tokens=300),
        inference_cost_microunits=6_500,
    )

    evidence = build_terminal_accepted_work_evidence(
        (baseline,),
        (candidate,),
    )

    assert evidence.evidence_claim_references == (
        TERMINAL_ACCEPTED_WORK_EVIDENCE_ID,
    )
    assert evidence.paired_report.terminal_accepted_work_accounting_proven
    assert not evidence.paired_report.token_gate_passed
    assert not evidence.promotion_eligible
    assert EFFICIENCY_EVIDENCE_PRODUCERS == {
        TERMINAL_ACCEPTED_WORK_EVIDENCE_ID: (
            "supervisor_efficiency_metrics."
            "build_terminal_accepted_work_evidence"
        )
    }


def test_terminal_accounting_rejects_nonempty_arms_without_accepted_work() -> None:
    failed = _fixtures()["failed"]

    with pytest.raises(
        EfficiencyValidationError,
        match="non-empty, population-complete",
    ):
        build_terminal_accepted_work_evidence(
            (failed,),
            (failed,),
        )


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

    verifiers = {binding.receipt_id: result.verifier}
    assert RequiredContextPromotionReport.from_json(
        report.to_json(),
        verifiers_by_receipt=verifiers,
    ) == report
    assert RequiredContextPromotionReport.from_dict(
        report.to_dict(include_report_id=True),
        verifiers_by_receipt=verifiers,
    ) == report
    with pytest.raises(
        EfficiencyValidationError,
        match="provider_tokens_verified",
    ):
        RequiredContextPromotionReport.from_json(report.to_json())


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

    foreign_measurement = replace(
        paired,
        cases=(
            replace(
                paired.cases[0],
                repository_tree_digest="sha256:" + "f" * 64,
            ),
        ),
    )
    rebound_without_builder = RequiredContextPromotionReport(
        paired_report=foreign_measurement,
        proof_bindings=detached.proof_bindings,
    )
    assert not rebound_without_builder.source_bindings_consistent
    assert not rebound_without_builder.typed_context_gate_passed
    assert not rebound_without_builder.evidence_claim_references

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


def test_g091_completion_requires_current_tree_health_quorum_and_two_phases() -> None:
    result = _required_context_fixture()
    terminal = _paired_required_context_report(result, verified=True)
    report = build_required_context_promotion_report(
        terminal,
        {"task:required-context": (result,)},
    )
    assert report.promotion_eligible
    assert report.source_bindings_consistent

    now = datetime(2026, 7, 24, 15, 0, tzinfo=timezone.utc)
    command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_efficiency_metrics.py "
        "test/api/test_agent_supervisor_context_compiler.py "
        "test/api/test_agent_supervisor_context_delta.py -q"
    )
    repository_id = result.receipt.repository_id
    tree_id = result.receipt.tree_id
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-061",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": tree_id,
                "command": command,
                "promotion_report_id": report.report_id,
            },
            validation_passed=True,
            repository_id=repository_id,
            repository_tree=tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-061:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            REQUIRED_CONTEXT_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "repository_tree": tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    + (
                        "context_contracts.py"
                        if index == 1
                        else "context_compiler.py"
                        if index < 6
                        else "supervisor_efficiency_metrics.py"
                    )
                ),
                "validation": (
                    "test/api/test_agent_supervisor_context_compiler.py"
                    if index < 6
                    else (
                        "test/api/"
                        "test_agent_supervisor_efficiency_metrics.py"
                    )
                ),
            }
            for index, criterion in enumerate(
                REQUIRED_CONTEXT_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    analyzer_version = "required-context-completion@1"
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
        "analyzer_version": analyzer_version,
    }
    binding = {
        "repository_id": repository_id,
        "tree_id": tree_id,
        "analyzer_version": analyzer_version,
        "configuration_revision": "sha256:completion-config",
        "objective_revision": result.capsule.objective_revision,
        "policy_revision": result.receipt.policy_revision,
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-061-exhaustive-a",
                "evidence_channel": "compiler-and-contracts",
                "receipt_cid": "scan:asi-061:exhaustive-a",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-061-compiler",
                "implementation": "required-context-compiler",
                "child_receipt_binding": "task:asi-061:compiler",
                "child_receipt_sha256": "sha256:" + "3" * 64,
                "aggregate_tree_binding": tree_id,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-061-exhaustive-b",
                "evidence_channel": "promotion-and-lifecycle",
                "receipt_cid": "scan:asi-061:exhaustive-b",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-061-promotion",
                "implementation": "required-context-promotion",
                "child_receipt_binding": "task:asi-061:promotion",
                "child_receipt_sha256": "sha256:" + "4" * 64,
                "aggregate_tree_binding": tree_id,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    provisional = report.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert provisional.acceptance_criteria == (
        REQUIRED_CONTEXT_ACCEPTANCE_CRITERIA
    )
    assert provisional.gate is not None and provisional.gate.passed
    assert "provisional_transition_required" in provisional.reason_codes

    verified = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    no_validations = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": ()},
    )
    assert not no_validations.verified
    assert no_validations.missing_criteria == (
        REQUIRED_CONTEXT_ACCEPTANCE_CRITERIA
    )

    failed = replace(
        evidence[0],
        provenance_cid="validation:asi-061:failed",
        validation_passed=False,
        validation_receipt={
            "status": "failed",
            "tree_id": tree_id,
            "command": command,
        },
    )
    failed_validation = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (*evidence, failed)},
    )
    assert not failed_validation.verified
    assert "failed_validation" in failed_validation.reason_codes

    unmapped = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "coverage": {
                **coverage,
                "criteria": [
                    *coverage["criteria"][:-1],
                    {**coverage["criteria"][-1], "validation": ""},
                ],
            },
        },
    )
    assert not unmapped.verified
    assert "coverage_unverified" in unmapped.reason_codes

    unsafe = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "analyzer_health": {
                "status": "healthy",
                "healthy": True,
                "analyzer_version": analyzer_version,
            },
        },
    )
    assert not unsafe.verified
    assert "analyzer_unhealthy" in unsafe.reason_codes

    invalid_quorums = (
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "receipt_cid": "scan:asi-061:exhaustive-a",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "scan_mode": "audit"},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "healthy": False},
            ],
        },
        {
            **quorum,
            "binding": {**binding, "tree_id": "sha256:foreign"},
        },
    )
    for invalid_quorum in invalid_quorums:
        rejected = report.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "exhaustion_quorum": invalid_quorum},
        )
        assert not rejected.verified
        assert any(
            code.startswith("exhaustion_quorum")
            for code in rejected.reason_codes
        )

    detached = replace(report, terminal_work_evidence=None)
    assert not detached.promotion_eligible
    rejected = detached.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert not rejected.verified


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
    forged_binding_claim = dict(identified)
    forged_binding_claim["source_bindings_consistent"] = False
    with pytest.raises(
        EfficiencyValidationError,
        match="source_bindings_consistent",
    ):
        DeltaRetryPromotionReport.from_dict(
            forged_binding_claim,
            verifiers_by_receipt=verifiers,
        )


def test_g092_completion_requires_current_tree_health_quorum_and_two_phases() -> None:
    result = _delta_retry_fixture()
    terminal = _paired_delta_report(result, verified=True)
    report = build_delta_retry_promotion_report(
        terminal,
        {"task:delta-retry": (result,)},
    )
    assert report.promotion_eligible
    assert report.source_bindings_consistent

    now = datetime(2026, 7, 24, 15, 0, tzinfo=timezone.utc)
    command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_efficiency_metrics.py "
        "test/api/test_agent_supervisor_context_compiler.py "
        "test/api/test_agent_supervisor_context_delta.py -q"
    )
    receipt = result.receipt
    repository_id = receipt.repository_id
    tree_id = receipt.tree_id
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-060",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": tree_id,
                "command": command,
                "promotion_report_id": report.report_id,
            },
            validation_passed=True,
            repository_id=repository_id,
            repository_tree=tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-060:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            DELTA_RETRY_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "repository_tree": tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    + (
                        "context_contracts.py"
                        if index == 1
                        else "context_compiler.py"
                        if index < 5
                        else "supervisor_efficiency_metrics.py"
                    )
                ),
                "validation": (
                    "test/api/test_agent_supervisor_context_delta.py"
                    if index < 5
                    else (
                        "test/api/"
                        "test_agent_supervisor_efficiency_metrics.py"
                    )
                ),
            }
            for index, criterion in enumerate(
                DELTA_RETRY_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    analyzer_version = "delta-retry-completion@1"
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
        "analyzer_version": analyzer_version,
    }
    binding = {
        "repository_id": repository_id,
        "tree_id": tree_id,
        "objective_id": DELTA_RETRY_OBJECTIVE_ID,
        "objective_revision": result.parent_capsule.objective_revision,
        "policy_revision": receipt.policy_revision,
        "requirement_id": DELTA_RETRY_CONTEXT_EVIDENCE_ID,
        "promotion_report_id": report.report_id,
        "analyzer_version": analyzer_version,
        "configuration_revision": "sha256:completion-config",
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-060-exhaustive-a",
                "evidence_channel": "delta-contract-and-compiler",
                "receipt_cid": "scan:asi-060:exhaustive-a",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-060-delta",
                "implementation": "delta-context-compiler",
                "child_receipt_binding": "task:asi-060:delta",
                "child_receipt_sha256": "sha256:" + "5" * 64,
                "aggregate_tree_binding": tree_id,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-060-exhaustive-b",
                "evidence_channel": "promotion-and-lifecycle",
                "receipt_cid": "scan:asi-060:exhaustive-b",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-060-promotion",
                "implementation": "delta-retry-promotion",
                "child_receipt_binding": "task:asi-060:promotion",
                "child_receipt_sha256": "sha256:" + "6" * 64,
                "aggregate_tree_binding": tree_id,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    provisional = report.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert provisional.acceptance_criteria == DELTA_RETRY_ACCEPTANCE_CRITERIA
    assert provisional.gate is not None and provisional.gate.passed
    assert "provisional_transition_required" in provisional.reason_codes

    verified = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    reordered = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "evidence": tuple(reversed(evidence)),
            "coverage": {
                **coverage,
                "criteria": list(reversed(coverage["criteria"])),
            },
        },
    )
    assert reordered.verified
    assert reordered.missing_criteria == verified.missing_criteria == ()
    assert reordered.invalid_criteria == verified.invalid_criteria == ()

    no_validations = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": ()},
    )
    assert not no_validations.verified
    assert no_validations.missing_criteria == (
        DELTA_RETRY_ACCEPTANCE_CRITERIA
    )

    failed = replace(
        evidence[0],
        provenance_cid="validation:asi-060:failed",
        validation_passed=False,
        validation_receipt={
            "status": "failed",
            "tree_id": tree_id,
            "command": command,
        },
    )
    failed_validation = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (*evidence, failed)},
    )
    assert not failed_validation.verified
    assert "failed_validation" in failed_validation.reason_codes

    stale = replace(
        evidence[0],
        provenance_cid="validation:asi-060:stale",
        observed_at=now - timedelta(seconds=301),
    )
    stale_validation = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (stale, *evidence[1:])},
    )
    assert not stale_validation.verified
    assert "stale_evidence" in stale_validation.reason_codes

    unmapped = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "coverage": {
                **coverage,
                "criteria": [
                    *coverage["criteria"][:-1],
                    {**coverage["criteria"][-1], "validation": ""},
                ],
            },
        },
    )
    assert not unmapped.verified
    assert "coverage_unverified" in unmapped.reason_codes

    duplicate_mapping = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "coverage": {
                **coverage,
                "criteria": [
                    *coverage["criteria"],
                    coverage["criteria"][0],
                ],
            },
        },
    )
    assert not duplicate_mapping.verified
    assert "coverage_unverified" in duplicate_mapping.reason_codes

    unsafe = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "analyzer_health": {
                "status": "healthy",
                "healthy": True,
                "analyzer_version": analyzer_version,
            },
        },
    )
    assert not unsafe.verified
    assert "analyzer_unhealthy" in unsafe.reason_codes

    invalid_quorums = (
        {**quorum, "required_members": 1},
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "receipt_cid": "scan:asi-060:exhaustive-a",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "member_id": "asi-060-exhaustive-a",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "uncontradicted": False,
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "scan_mode": "audit"},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "healthy": False},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "finished_at": (
                        now - timedelta(seconds=301)
                    ).isoformat(),
                },
            ],
        },
        {
            **quorum,
            "binding": {
                **binding,
                "tree_id": "sha256:" + "0" * 64,
            },
        },
        {
            **quorum,
            "binding": {
                **binding,
                "promotion_report_id": "sha256:" + "0" * 64,
            },
        },
    )
    for invalid_quorum in invalid_quorums:
        rejected = report.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "exhaustion_quorum": invalid_quorum},
        )
        assert not rejected.verified
        assert any(
            code.startswith("exhaustion_quorum")
            for code in rejected.reason_codes
        )

    configured_three = report.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        required_exhaustive_receipts=3,
        **values,
    )
    assert not configured_three.verified
    assert any(
        code.startswith("exhaustion_quorum")
        for code in configured_three.reason_codes
    )

    detached = replace(report, terminal_work_evidence=None)
    assert not detached.promotion_eligible
    rejected = detached.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert not rejected.verified


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


def test_g010_parent_completion_closes_producers_children_and_proof_gate() -> None:
    now = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
    repository_id = "repo:token-efficiency"
    tree_id = "sha256:" + "a" * 64
    command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_efficiency_metrics.py "
        "test/api/test_agent_supervisor_context_compiler.py "
        "test/api/test_agent_supervisor_context_delta.py -q"
    )
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-088",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": tree_id,
                "command": command,
            },
            validation_passed=True,
            repository_id=repository_id,
            repository_tree=tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-088:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            TOKEN_EFFICIENCY_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "repository_tree": tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    + (
                        "context_contracts.py"
                        if index == 1
                        else "context_compiler.py"
                        if index < 4
                        else "supervisor_efficiency_metrics.py"
                    )
                ),
                "validation": (
                    "test/api/test_agent_supervisor_context_compiler.py"
                    if index < 3
                    else "test/api/test_agent_supervisor_context_delta.py"
                    if index == 3
                    else (
                        "test/api/"
                        "test_agent_supervisor_efficiency_metrics.py"
                    )
                ),
                "validation_receipt_id": evidence[
                    index - 1
                ].provenance_cid,
            }
            for index, criterion in enumerate(
                TOKEN_EFFICIENCY_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    analyzer_version = "token-efficiency-completion@1"
    binding = {
        "repository_id": repository_id,
        "tree_id": tree_id,
        "objective_id": TOKEN_EFFICIENCY_OBJECTIVE_ID,
        "objective_revision": TOKEN_EFFICIENCY_OBJECTIVE_REVISION,
        "analyzer_version": analyzer_version,
        "configuration_revision": "sha256:g010-completion-config",
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
        "analyzer_version": analyzer_version,
        "binding": binding,
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-088-exhaustive-a",
                "evidence_channel": "contracts-compiler",
                "receipt_cid": "scan:asi-088:exhaustive-a",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-088-contracts",
                "implementation": "token-efficiency-contracts",
                "child_receipt_binding": "task:asi-088:contracts",
                "child_receipt_sha256": "sha256:" + "7" * 64,
                "aggregate_tree_binding": tree_id,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-088-exhaustive-b",
                "evidence_channel": "delta-measurement",
                "receipt_cid": "scan:asi-088:exhaustive-b",
                "binding": binding,
                "scan_mode": "exhaustive",
                "passed": True,
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "exhaustive": True,
                "conclusive": True,
                "uncontradicted": True,
                "analyzer_version": analyzer_version,
                "producer_id": "asi-088-measurement",
                "implementation": "token-efficiency-measurement",
                "child_receipt_binding": "task:asi-088:measurement",
                "child_receipt_sha256": "sha256:" + "8" * 64,
                "aggregate_tree_binding": tree_id,
                "finished_at": now.isoformat(),
            },
        ],
    }
    producing_tasks = [
        {"task_id": task_id, "status": "completed"}
        for task_id in TOKEN_EFFICIENCY_PRODUCING_TASK_IDS
    ]
    child_goals = [
        {
            "goal_id": goal_id,
            "state": "verified_complete",
            "verified": True,
            "proof_requirements": [
                {
                    "goal_id": goal_id,
                    "acceptance_criterion": "typed producer proof",
                    "obligation_id": f"proof:{goal_id}",
                    "proof_receipt_id": f"receipt:{goal_id}",
                    "required_assurance": "kernel_verified",
                    "authoritative_assurance": "kernel_verified",
                    "proof_verdict": "proved",
                    "freshness": "current",
                    "assurance_satisfied": True,
                }
            ],
            "completion_gate": {
                "passed": True,
                "evaluated_evidence": {
                    "repository_id": repository_id,
                    "repository_tree": tree_id,
                    "evaluated_at": now.isoformat(),
                    "validation_evidence": [
                        {
                            "valid": True,
                            "evidence": {
                                "repository_id": repository_id,
                                "repository_tree": tree_id,
                            },
                        }
                    ],
                },
            },
        }
        for goal_id in TOKEN_EFFICIENCY_CHILD_GOAL_IDS
    ]
    values = {
        "repository_id": repository_id,
        "repository_tree": tree_id,
        "producing_tasks": producing_tasks,
        "child_goals": child_goals,
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    provisional = evaluate_token_efficiency_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert provisional.acceptance_criteria == (
        TOKEN_EFFICIENCY_ACCEPTANCE_CRITERIA
    )
    assert provisional.gate is not None and provisional.gate.passed

    verified = evaluate_token_efficiency_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    typed_coverage = GoalCoverageMap(
        criteria=[
            AcceptanceCoverage(
                criterion_id=f"ASI-G010:{index}",
                goal_id=TOKEN_EFFICIENCY_OBJECTIVE_ID,
                criterion=criterion,
                status=CoverageStatus.VERIFIED,
                changed_files=[
                    "ipfs_accelerate_py/agent_supervisor/"
                    "supervisor_efficiency_metrics.py"
                ],
                validation_receipt_ids=[
                    evidence[index - 1].provenance_cid
                ],
            )
            for index, criterion in enumerate(
                TOKEN_EFFICIENCY_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
        edges=[],
        receipts=[],
        finding_assignments=[],
        registered_goal_ids=[TOKEN_EFFICIENCY_OBJECTIVE_ID],
        evaluated_at=now.isoformat(),
        repository_tree=tree_id,
    )
    typed_health = AnalyzerHealthReport(
        status=AnalyzerHealthStatus.HEALTHY,
        reasons=(),
        thresholds=AnalyzerHealthThresholds(),
        metrics={"objective_id": TOKEN_EFFICIENCY_OBJECTIVE_ID},
    )
    typed_quorum = evaluate_exhaustion_quorum(
        (
            {
                "receipt_cid": "scan:asi-088:typed-contracts",
                "terminal_reason": "exhausted",
                "scan_mode": "exhaustive",
                "finished_at": now.isoformat(),
                "metadata": {
                    "analyzer_health": {"status": "healthy"},
                    "coverage_complete": True,
                    "evidence_channel": "typed-contracts",
                },
            },
            {
                "receipt_cid": "scan:asi-088:typed-measurement",
                "terminal_reason": "exhausted",
                "scan_mode": "audit",
                "finished_at": now.isoformat(),
                "metadata": {
                    "analyzer_health": {"status": "healthy"},
                    "coverage_complete": True,
                    "evidence_channel": "typed-measurement",
                },
            },
        ),
        binding=ExhaustionBinding(
            repository_id=repository_id,
            tree_id=tree_id,
            analyzer_version=analyzer_version,
            configuration_revision=binding["configuration_revision"],
            objective_revision=TOKEN_EFFICIENCY_OBJECTIVE_REVISION,
        ),
        required_members=2,
    )
    typed_verified = evaluate_token_efficiency_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "coverage": typed_coverage,
            "analyzer_health": typed_health,
            "exhaustion_quorum": typed_quorum,
        },
    )
    assert typed_verified.state is GoalState.VERIFIED_COMPLETE
    assert typed_verified.verified

    failed = replace(
        evidence[0],
        provenance_cid="validation:asi-088:failed",
        validation_passed=False,
        validation_receipt={"status": "failed", "tree_id": tree_id},
    )
    invalid_cases = (
        {"producing_tasks": producing_tasks[:-1]},
        {
            "child_goals": [
                {
                    **child_goals[0],
                    "completion_gate": {
                        **child_goals[0]["completion_gate"],
                        "evaluated_evidence": {
                            **child_goals[0]["completion_gate"][
                                "evaluated_evidence"
                            ],
                            "evaluated_at": (
                                now - timedelta(seconds=301)
                            ).isoformat(),
                        },
                    },
                },
                *child_goals[1:],
            ]
        },
        {"evidence": (*evidence, failed)},
        {
            "coverage": {
                **coverage,
                "criteria": [
                    {
                        **coverage["criteria"][0],
                        "validation_receipt_id": "validation:detached",
                    },
                    *coverage["criteria"][1:],
                ],
            }
        },
        {
            "analyzer_health": {
                **health,
                "safe_for_completion_reasoning": False,
            }
        },
        {
            "exhaustion_quorum": {
                **quorum,
                "members": [
                    quorum["members"][0],
                    {
                        **quorum["members"][1],
                        "receipt_cid": quorum["members"][0]["receipt_cid"],
                    },
                ],
            }
        },
    )
    for change in invalid_cases:
        rejected = evaluate_token_efficiency_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, **change},
        )
        assert not rejected.verified

    with pytest.raises(ValueError, match="configured ASI-G010 count"):
        evaluate_token_efficiency_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            required_exhaustive_receipts=1,
            **values,
        )
