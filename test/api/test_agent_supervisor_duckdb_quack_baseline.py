"""DQP-009: state, latency, and LLM-churn baselines.

Acceptance (from the sealed DuckDB/Quack board):

* Baseline binds tree, environment, workload and metric definitions
* Distinguishes missing from zero
* Counts rejected/retry/abandoned provider usage
* Cannot be regenerated with weakened safety, durability, or quality criteria

Interfaces: ``SupervisorStateBaseline@1``, ``LLMChurnBaseline@1``
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline import (
    BASELINE_CONTRACT_VERSION,
    BASELINE_EVIDENCE,
    BASELINE_GOAL_ID,
    BASELINE_TASK_ID,
    BaselineBinding,
    BaselineStratum,
    DEFAULT_DURABILITY_FLOORS,
    DEFAULT_QUALITY_FLOORS,
    DEFAULT_SAFETY_FLOORS,
    DuckDBQuackBaseline,
    DuckDBQuackBaselineError,
    HERMETIC_WORKLOAD_SENSOR,
    LLM_CHURN_BASELINE_INTERFACE,
    LLMChurnBaseline,
    LockedCriteria,
    MISSING_TELEMETRY_SENSOR,
    MetricObservation,
    PROVIDER_LEDGER_SENSOR,
    ProviderUsageCharge,
    ProviderUsageDisposition,
    SUPERVISOR_STATE_BASELINE_INTERFACE,
    SampleStatus,
    SupervisorStateBaseline,
    UnavailableReason,
    charge_provider_usage,
    content_identity,
    default_llm_churn_metric_definitions,
    default_metric_definitions,
    default_state_metric_definitions,
    establish_baseline,
    establish_llm_churn_baseline,
    establish_state_baseline,
    hermetic_fixture_baseline,
    metric_definitions_digest,
    regenerate_baseline,
)


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_and_task_identities() -> None:
    assert SUPERVISOR_STATE_BASELINE_INTERFACE == "SupervisorStateBaseline@1"
    assert LLM_CHURN_BASELINE_INTERFACE == "LLMChurnBaseline@1"
    assert BASELINE_TASK_ID == "DQP-009"
    assert BASELINE_GOAL_ID == "DQP-G050"
    assert BASELINE_CONTRACT_VERSION == 1
    assert BASELINE_EVIDENCE == "dqp/duckdb-quack-baseline@1"
    assert SupervisorStateBaseline.INTERFACE == SUPERVISOR_STATE_BASELINE_INTERFACE
    assert LLMChurnBaseline.INTERFACE == LLM_CHURN_BASELINE_INTERFACE


def test_default_metric_catalog_covers_state_latency_and_churn() -> None:
    state_names = {item.name for item in default_state_metric_definitions()}
    churn_names = {item.name for item in default_llm_churn_metric_definitions()}
    all_names = {item.name for item in default_metric_definitions()}

    for required in (
        "file_reads",
        "file_writes",
        "file_parses",
        "independent_db_opens",
        "lock_wait_ms",
        "noop_poll_count",
        "task_claim_latency_ms",
        "queue_latency_ms",
        "accepted_mutation_quality_bps",
        "rollback_rate_bps",
        "failure_rate_bps",
    ):
        assert required in state_names

    for required in (
        "context_bytes",
        "provider_calls",
        "provider_input_tokens",
        "provider_output_tokens",
        "duplicate_semantic_inputs",
        "cache_reuse_count",
        "rejected_provider_usage",
        "retry_provider_usage",
        "abandoned_provider_usage",
        "accepted_provider_usage",
    ):
        assert required in churn_names

    assert state_names | churn_names == all_names
    assert metric_definitions_digest().startswith("sha256:")


# ---------------------------------------------------------------------------
# Binding: tree, environment, workload, metric definitions
# ---------------------------------------------------------------------------


def test_baseline_binds_tree_environment_workload_and_metrics() -> None:
    baseline = hermetic_fixture_baseline()

    assert baseline.binding.tree_id == "tree:dqp-009-hermetic"
    assert baseline.binding.environment_id == "environment:hermetic-validation"
    assert baseline.binding.workload_id == "workload:dqp-009-fixed-hermetic@1"
    assert baseline.binding.metric_definitions_digest == metric_definitions_digest(
        baseline.metric_definitions
    )
    assert baseline.binding.seed == "seed:dqp-009"

    # Nested interfaces share identity fields with the combined binding.
    for component in (baseline.state, baseline.llm_churn):
        assert component.binding.tree_id == baseline.binding.tree_id
        assert component.binding.environment_id == baseline.binding.environment_id
        assert component.binding.workload_id == baseline.binding.workload_id
        assert component.binding.seed == baseline.binding.seed
        assert component.INTERFACE in (
            SUPERVISOR_STATE_BASELINE_INTERFACE,
            LLM_CHURN_BASELINE_INTERFACE,
        )

def test_binding_rejects_non_sha_metric_digest() -> None:
    with pytest.raises(DuckDBQuackBaselineError, match="sha256"):
        BaselineBinding(
            tree_id="tree:1",
            environment_id="env:1",
            workload_id="work:1",
            metric_definitions_digest="not-a-digest",
        )


def test_state_baseline_rejects_digest_mismatch() -> None:
    definitions = default_state_metric_definitions()
    binding = BaselineBinding(
        tree_id="tree:1",
        environment_id="env:1",
        workload_id="work:1",
        metric_definitions_digest=metric_definitions_digest(
            default_llm_churn_metric_definitions()
        ),
    )
    with pytest.raises(DuckDBQuackBaselineError, match="metric_definitions_digest"):
        SupervisorStateBaseline(
            binding=binding,
            metric_definitions=definitions,
            observations=(),
            locked_criteria=LockedCriteria.defaults(),
            strata=(BaselineStratum.COLD,),
        )


def test_establish_state_baseline_round_trip() -> None:
    baseline = establish_state_baseline(
        tree_id="tree:rt",
        environment_id="env:rt",
        workload_id="work:rt",
        observations=[
            MetricObservation.measured(
                "file_reads",
                10,
                sensor_id=HERMETIC_WORKLOAD_SENSOR,
                stratum=BaselineStratum.COLD,
            )
        ],
        strata=(BaselineStratum.COLD,),
    )
    restored = SupervisorStateBaseline.from_dict(baseline.to_dict())
    assert restored.content_id == baseline.content_id
    assert restored.measured_or_none("file_reads") == 10


# ---------------------------------------------------------------------------
# Missing vs zero
# ---------------------------------------------------------------------------


def test_measured_zero_is_not_missing() -> None:
    zero = MetricObservation.measured(
        "noop_poll_count",
        0,
        sensor_id=HERMETIC_WORKLOAD_SENSOR,
        stratum=BaselineStratum.COLD,
    )
    assert zero.is_measured
    assert zero.measured_value() == 0
    assert zero.status is SampleStatus.MEASURED
    assert "value" in zero.to_dict()
    assert "reason_code" not in zero.to_dict()


def test_unavailable_is_not_encoded_as_zero() -> None:
    missing = MetricObservation.unavailable(
        "lock_wait_ms",
        UnavailableReason.TELEMETRY_MISSING,
        stratum=BaselineStratum.WARM,
    )
    assert missing.is_unavailable
    assert missing.measured_value() is None  # missing, not zero
    assert missing.value == 0  # internal placeholder only
    payload = missing.to_dict()
    assert "value" not in payload
    assert payload["reason_code"] == "telemetry-missing"
    assert payload["sensor_id"] == MISSING_TELEMETRY_SENSOR


def test_unavailable_rejects_numeric_encoding() -> None:
    with pytest.raises(DuckDBQuackBaselineError, match="must not encode a numeric"):
        MetricObservation(
            metric_name="file_reads",
            status=SampleStatus.UNAVAILABLE,
            sensor_id=MISSING_TELEMETRY_SENSOR,
            stratum=BaselineStratum.COLD,
            value=5,
            reason_code=UnavailableReason.TELEMETRY_MISSING.value,
        )


def test_missing_required_metric_becomes_unavailable_not_zero() -> None:
    baseline = establish_state_baseline(
        tree_id="tree:miss",
        environment_id="env:miss",
        workload_id="work:miss",
        observations=[],  # nothing measured
        strata=(BaselineStratum.COLD,),
    )
    observation = baseline.observation("file_reads")
    assert observation is not None
    assert observation.is_unavailable
    assert baseline.measured_or_none("file_reads") is None
    assert "file_reads" in baseline.unavailable_metrics()


def test_hermetic_fixture_preserves_measured_zeros() -> None:
    baseline = hermetic_fixture_baseline()
    # Explicit measured zeros from the hermetic workload sensor.
    assert baseline.state.measured_or_none("noop_poll_count") == 0
    assert baseline.llm_churn.measured_or_none("cache_reuse_count") == 0
    noop = baseline.state.observation("noop_poll_count")
    assert noop is not None and noop.is_measured
    assert noop.sensor_id == HERMETIC_WORKLOAD_SENSOR


# ---------------------------------------------------------------------------
# Rejected / retry / abandoned provider usage charged
# ---------------------------------------------------------------------------


def test_charge_provider_usage_counts_rejected_retry_abandoned() -> None:
    totals = charge_provider_usage(
        [
            ProviderUsageCharge(
                disposition=ProviderUsageDisposition.ACCEPTED,
                call_count=2,
                input_tokens=100,
                output_tokens=40,
            ),
            ProviderUsageCharge(
                disposition=ProviderUsageDisposition.REJECTED,
                call_count=3,
                input_tokens=90,
                output_tokens=0,
            ),
            ProviderUsageCharge(
                disposition=ProviderUsageDisposition.RETRY,
                call_count=4,
                input_tokens=80,
                output_tokens=10,
            ),
            ProviderUsageCharge(
                disposition=ProviderUsageDisposition.ABANDONED,
                call_count=5,
                input_tokens=70,
                output_tokens=0,
            ),
        ]
    )
    assert totals.accepted_calls == 2
    assert totals.rejected_calls == 3
    assert totals.retry_calls == 4
    assert totals.abandoned_calls == 5
    assert totals.total_calls == 14
    assert totals.non_accepted_calls == 12
    assert totals.to_dict()["all_dispositions_charged"] is True
    # Tokens remain charged for non-accepted work.
    assert totals.rejected_tokens == 90
    assert totals.retry_tokens == 90
    assert totals.abandoned_tokens == 70
    assert totals.total_tokens == 100 + 40 + 90 + 90 + 70


def test_llm_churn_baseline_records_non_accepted_usage() -> None:
    baseline = establish_llm_churn_baseline(
        tree_id="tree:churn",
        environment_id="env:churn",
        workload_id="work:churn",
        provider_charges=[
            {
                "disposition": "accepted",
                "call_count": 1,
                "input_tokens": 50,
                "output_tokens": 10,
            },
            {
                "disposition": "rejected",
                "call_count": 2,
                "input_tokens": 40,
                "output_tokens": 0,
            },
            {
                "disposition": "retry",
                "call_count": 3,
                "input_tokens": 30,
                "output_tokens": 5,
            },
            {
                "disposition": "abandoned",
                "call_count": 4,
                "input_tokens": 20,
                "output_tokens": 0,
            },
        ],
        strata=(BaselineStratum.COLD,),
    )
    usage = baseline.provider_usage
    assert usage.rejected_calls == 2
    assert usage.retry_calls == 3
    assert usage.abandoned_calls == 4
    assert usage.total_calls == 10
    assert baseline.measured_or_none("rejected_provider_usage") == 2
    assert baseline.measured_or_none("retry_provider_usage") == 3
    assert baseline.measured_or_none("abandoned_provider_usage") == 4
    assert baseline.measured_or_none("provider_calls") == 10
    # Sensor proves the charges came from the provider ledger, not invention.
    rejected = baseline.observation("rejected_provider_usage")
    assert rejected is not None
    assert rejected.sensor_id == PROVIDER_LEDGER_SENSOR


def test_provider_calls_must_include_non_accepted_charges() -> None:
    usage = charge_provider_usage(
        [
            ProviderUsageCharge(
                disposition=ProviderUsageDisposition.REJECTED,
                call_count=2,
                input_tokens=10,
            )
        ]
    )
    definitions = default_llm_churn_metric_definitions()
    binding = BaselineBinding(
        tree_id="tree:x",
        environment_id="env:x",
        workload_id="work:x",
        metric_definitions_digest=metric_definitions_digest(definitions),
    )
    # Measured provider_calls that omit rejected work is rejected.
    with pytest.raises(DuckDBQuackBaselineError, match="provider_calls"):
        LLMChurnBaseline(
            binding=binding,
            metric_definitions=definitions,
            observations=(
                MetricObservation.measured(
                    "provider_calls",
                    0,  # pretends rejected work is free
                    sensor_id=PROVIDER_LEDGER_SENSOR,
                    stratum=BaselineStratum.COLD,
                ),
                MetricObservation.measured(
                    "rejected_provider_usage",
                    2,
                    sensor_id=PROVIDER_LEDGER_SENSOR,
                    stratum=BaselineStratum.COLD,
                ),
                MetricObservation.measured(
                    "retry_provider_usage",
                    0,
                    sensor_id=PROVIDER_LEDGER_SENSOR,
                    stratum=BaselineStratum.COLD,
                ),
                MetricObservation.measured(
                    "abandoned_provider_usage",
                    0,
                    sensor_id=PROVIDER_LEDGER_SENSOR,
                    stratum=BaselineStratum.COLD,
                ),
                MetricObservation.measured(
                    "accepted_provider_usage",
                    0,
                    sensor_id=PROVIDER_LEDGER_SENSOR,
                    stratum=BaselineStratum.COLD,
                ),
            ),
            provider_usage=usage,
            locked_criteria=LockedCriteria.defaults(),
            strata=(BaselineStratum.COLD,),
        )


# ---------------------------------------------------------------------------
# Locked criteria: cannot regenerate with weakened floors
# ---------------------------------------------------------------------------


def test_default_locked_criteria_include_safety_durability_quality() -> None:
    criteria = LockedCriteria.defaults()
    for key, value in DEFAULT_SAFETY_FLOORS.items():
        assert criteria.safety_floors[key] == value == 0
    for key, value in DEFAULT_DURABILITY_FLOORS.items():
        assert criteria.durability_floors[key] == value == 0
    for key, value in DEFAULT_QUALITY_FLOORS.items():
        assert criteria.quality_floors[key] == value
    payload = criteria.to_dict()
    assert payload["floor_kinds"]["safety_floors"] == "maximum"
    assert payload["floor_kinds"]["durability_floors"] == "maximum"
    assert payload["floor_kinds"]["quality_floors"] == "minimum"


def test_regenerate_rejects_weakened_safety_floor() -> None:
    prior = hermetic_fixture_baseline()
    weakened = LockedCriteria(
        safety_floors={
            **dict(prior.locked_criteria.safety_floors),
            "false_completions": 1,  # raised allowance
        },
        durability_floors=dict(prior.locked_criteria.durability_floors),
        quality_floors=dict(prior.locked_criteria.quality_floors),
    )
    with pytest.raises(
        DuckDBQuackBaselineError,
        match="weakened safety, durability, or quality criteria",
    ):
        regenerate_baseline(prior, locked_criteria=weakened)


def test_regenerate_rejects_weakened_durability_floor() -> None:
    prior = hermetic_fixture_baseline()
    weakened = LockedCriteria(
        safety_floors=dict(prior.locked_criteria.safety_floors),
        durability_floors={
            **dict(prior.locked_criteria.durability_floors),
            "accepted_state_losses": 1,
        },
        quality_floors=dict(prior.locked_criteria.quality_floors),
    )
    with pytest.raises(DuckDBQuackBaselineError, match="durability_floors"):
        regenerate_baseline(prior, locked_criteria=weakened)


def test_regenerate_rejects_lowered_quality_floor() -> None:
    prior = hermetic_fixture_baseline()
    weakened = LockedCriteria(
        safety_floors=dict(prior.locked_criteria.safety_floors),
        durability_floors=dict(prior.locked_criteria.durability_floors),
        quality_floors={
            **dict(prior.locked_criteria.quality_floors),
            "accepted_mutation_quality_bps": 1_000,  # lowered minimum
        },
    )
    with pytest.raises(DuckDBQuackBaselineError, match="quality_floors"):
        regenerate_baseline(prior, locked_criteria=weakened)


def test_regenerate_allows_equal_or_stricter_criteria_and_new_measurements() -> None:
    prior = hermetic_fixture_baseline()
    stricter = LockedCriteria(
        safety_floors=dict(prior.locked_criteria.safety_floors),
        durability_floors=dict(prior.locked_criteria.durability_floors),
        quality_floors={
            **dict(prior.locked_criteria.quality_floors),
            "accepted_mutation_quality_bps": 9_000,  # higher minimum is stricter
        },
    )
    regenerated = regenerate_baseline(
        prior,
        locked_criteria=stricter,
        state_observations=[
            MetricObservation.measured(
                "file_reads",
                99,
                sensor_id=HERMETIC_WORKLOAD_SENSOR,
                stratum=BaselineStratum.COLD,
            )
        ],
        provider_charges=[
            ProviderUsageCharge(
                disposition=ProviderUsageDisposition.ACCEPTED,
                call_count=1,
                input_tokens=10,
                output_tokens=5,
            )
        ],
        strata=(BaselineStratum.COLD,),
    )
    assert regenerated.locked_criteria.quality_floors[
        "accepted_mutation_quality_bps"
    ] == 9_000
    assert regenerated.state.measured_or_none("file_reads") == 99
    assert regenerated.llm_churn.provider_usage.total_calls == 1
    assert regenerated.binding.tree_id == prior.binding.tree_id


def test_regenerate_with_same_criteria_succeeds() -> None:
    prior = hermetic_fixture_baseline()
    again = regenerate_baseline(prior, locked_criteria=prior.locked_criteria)
    assert again.locked_criteria.content_id == prior.locked_criteria.content_id
    assert again.binding.tree_id == prior.binding.tree_id


# ---------------------------------------------------------------------------
# Combined baseline / hermetic fixture
# ---------------------------------------------------------------------------


def test_hermetic_fixture_combined_baseline_interfaces() -> None:
    baseline = hermetic_fixture_baseline()
    assert isinstance(baseline, DuckDBQuackBaseline)
    assert baseline.task_id == "DQP-009"
    assert baseline.goal_id == "DQP-G050"
    assert baseline.state.INTERFACE == SUPERVISOR_STATE_BASELINE_INTERFACE
    assert baseline.llm_churn.INTERFACE == LLM_CHURN_BASELINE_INTERFACE
    assert baseline.llm_churn.provider_usage.rejected_calls == 1
    assert baseline.llm_churn.provider_usage.retry_calls == 1
    assert baseline.llm_churn.provider_usage.abandoned_calls == 1
    assert baseline.llm_churn.provider_usage.total_calls == 5
    assert baseline.state.sample_count == 3
    assert baseline.state.confidence_bps == 9_500

    payload = baseline.to_dict()
    restored = DuckDBQuackBaseline.from_dict(payload)
    assert restored.content_id == baseline.content_id
    assert SUPERVISOR_STATE_BASELINE_INTERFACE in payload["interfaces"]
    assert LLM_CHURN_BASELINE_INTERFACE in payload["interfaces"]

def test_establish_baseline_binds_full_metric_catalog() -> None:
    baseline = establish_baseline(
        tree_id="tree:full",
        environment_id="env:full",
        workload_id="work:full",
        strata=(BaselineStratum.COLD, BaselineStratum.WARM),
        state_observations=[
            MetricObservation.measured(
                "file_reads",
                1,
                sensor_id=HERMETIC_WORKLOAD_SENSOR,
                stratum=BaselineStratum.COLD,
            )
        ],
        provider_charges=[],
    )
    assert len(baseline.metric_definitions) == len(default_metric_definitions())
    assert baseline.binding.metric_definitions_digest == metric_definitions_digest()
    # Unobserved warm stratum metrics are unavailable, not zero-invented.
    warm_reads = baseline.state.observation(
        "file_reads", stratum=BaselineStratum.WARM
    )
    assert warm_reads is not None
    assert warm_reads.is_unavailable
    assert warm_reads.measured_value() is None


def test_negative_counts_rejected() -> None:
    with pytest.raises(DuckDBQuackBaselineError, match="non-negative"):
        MetricObservation.measured(
            "file_reads",
            -1,
            sensor_id=HERMETIC_WORKLOAD_SENSOR,
        )
    with pytest.raises(DuckDBQuackBaselineError, match="non-negative"):
        ProviderUsageCharge(
            disposition=ProviderUsageDisposition.REJECTED,
            call_count=-1,
        )


def test_content_identity_is_stable() -> None:
    left = hermetic_fixture_baseline()
    right = hermetic_fixture_baseline()
    assert left.content_id == right.content_id
    assert left.content_id == content_identity(left.to_dict())


def test_closed_provider_disposition_vocabulary() -> None:
    assert {item.value for item in ProviderUsageDisposition} == {
        "accepted",
        "rejected",
        "retry",
        "abandoned",
    }
    for disposition in ProviderUsageDisposition:
        assert disposition.is_charged is True


def test_strata_vocabulary() -> None:
    assert {item.value for item in BaselineStratum} == {
        "cold",
        "warm",
        "restart",
        "parallel",
    }


def test_combined_baseline_rejects_identity_mismatch() -> None:
    baseline = hermetic_fixture_baseline()
    foreign = establish_state_baseline(
        tree_id="tree:foreign",
        environment_id=baseline.binding.environment_id,
        workload_id=baseline.binding.workload_id,
        strata=(BaselineStratum.COLD,),
        seed=baseline.binding.seed,
    )
    with pytest.raises(DuckDBQuackBaselineError, match="tree_id"):
        DuckDBQuackBaseline(
            binding=baseline.binding,
            state=foreign,
            llm_churn=baseline.llm_churn,
            locked_criteria=baseline.locked_criteria,
            metric_definitions=baseline.metric_definitions,
        )
