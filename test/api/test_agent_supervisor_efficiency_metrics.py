from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.supervisor_efficiency_metrics import (
    EFFICIENCY_CONTRACT_VERSION,
    EFFICIENCY_RECEIPT_SCHEMA,
    EFFICIENCY_REPORT_SCHEMA,
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
    EfficiencyReceipt,
    EfficiencyReport,
    EfficiencyScenario,
    EfficiencyValidationError,
    EvidenceDelta,
    RetryObservation,
    StageName,
    StageTiming,
    TerminalAcceptance,
    TerminalOutcome,
    TokenUsage,
    WorkCost,
    WorkStatus,
    aggregate_efficiency_receipts,
    build_efficiency_baseline_fixtures,
)


def _fixtures() -> dict[str, EfficiencyReceipt]:
    return build_efficiency_baseline_fixtures()


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
