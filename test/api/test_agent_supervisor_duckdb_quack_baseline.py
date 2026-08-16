"""DQP-009: state, latency, and LLM-churn baseline hermetic tests.

Interfaces: ``SupervisorStateBaseline@1``, ``LLMChurnBaseline@1``

Acceptance:

* Baseline binds tree, environment, workload and metric definitions
* Distinguishes missing from zero
* Counts rejected / retry / abandoned provider usage
* Cannot be regenerated with weakened safety, durability, or quality criteria
"""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline import (
    BASELINE_CONTRACT_VERSION,
    DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS,
    DEFAULT_MIN_SAMPLES,
    DURABILITY_FLOOR_KEYS,
    EVIDENCE,
    GOAL_ID,
    LLM_CHURN_BASELINE_INTERFACE,
    LLM_CHURN_METRIC_NAMES,
    SAFETY_FLOOR_KEYS,
    STATE_METRIC_NAMES,
    SUPERVISOR_STATE_BASELINE_INTERFACE,
    TASK_ID,
    BaselineBinding,
    BaselineCriteria,
    BaselineEnvironment,
    BaselineStratum,
    BaselineVerdict,
    DuckDBQuackBaselineError,
    LLMChurnBaseline,
    MetricSample,
    ProviderUsageCounters,
    SupervisorStateBaseline,
    TelemetryStatus,
    UnavailableReason,
    WorkloadDefinition,
    assert_criteria_not_weakened,
    content_identity,
    default_llm_metric_definitions,
    default_state_metric_definitions,
    default_workload,
    establish_duckdb_quack_baselines,
    establish_llm_churn_baseline,
    establish_supervisor_state_baseline,
    metric_sample_distinguishes_missing_from_zero,
    regenerate_baseline_with_criteria,
)


TREE_ID = "tree:sha256:dqp009-fixture-tree"


def _fixed_environment() -> BaselineEnvironment:
    return BaselineEnvironment(
        python_version="3.12.0",
        platform_name="Linux-fixed",
        implementation="CPython",
        path_fingerprint="sha256:" + ("ab" * 32),
        duckdb_version="1.5.2",
        extra={"machine": "x86_64", "system": "Linux"},
    )


def _fixed_workload(*, sample_count: int = 8) -> WorkloadDefinition:
    return default_workload(seed=0xD009, sample_count=sample_count)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert SUPERVISOR_STATE_BASELINE_INTERFACE == "SupervisorStateBaseline@1"
    assert LLM_CHURN_BASELINE_INTERFACE == "LLMChurnBaseline@1"
    assert SupervisorStateBaseline.INTERFACE == SUPERVISOR_STATE_BASELINE_INTERFACE
    assert LLMChurnBaseline.INTERFACE == LLM_CHURN_BASELINE_INTERFACE
    assert BASELINE_CONTRACT_VERSION == 1
    assert TASK_ID == "DQP-009"
    assert GOAL_ID == "DQP-G050"
    assert EVIDENCE == "dqp/duckdb-quack-baseline@1"


# ---------------------------------------------------------------------------
# Binding: tree, environment, workload, metric definitions
# ---------------------------------------------------------------------------


def test_baseline_binds_tree_environment_workload_and_metric_definitions() -> None:
    env = _fixed_environment()
    workload = _fixed_workload()
    report = establish_duckdb_quack_baselines(
        tree_id=TREE_ID,
        environment=env,
        workload=workload,
        repository_id="repository:dqp-009",
    )
    assert report.verdict is BaselineVerdict.ESTABLISHED

    state = report.state_baseline
    llm = report.llm_churn_baseline

    for baseline in (state, llm):
        assert baseline.binding.tree_id == TREE_ID
        assert baseline.binding.repository_id == "repository:dqp-009"
        assert baseline.binding.environment.identity_id == env.identity_id
        assert baseline.binding.workload.identity_id == workload.identity_id
        assert baseline.binding.metric_definitions
        assert baseline.binding.criteria.identity_id == report.criteria.identity_id
        # All required strata present.
        strata = {
            item.stratum.value if hasattr(item.stratum, "value") else item.stratum
            for item in baseline.strata
        }
        assert strata == {s.value for s in BaselineStratum}

    state_names = {item.name for item in state.binding.metric_definitions}
    llm_names = {item.name for item in llm.binding.metric_definitions}
    assert state_names == set(STATE_METRIC_NAMES)
    assert llm_names == set(LLM_CHURN_METRIC_NAMES)
    assert state.binding.metric_definition_identity == content_identity(
        [item.to_dict() for item in default_state_metric_definitions()]
    )
    assert llm.binding.metric_definition_identity == content_identity(
        [item.to_dict() for item in default_llm_metric_definitions()]
    )


def test_binding_round_trip_preserves_identity() -> None:
    env = _fixed_environment()
    workload = _fixed_workload()
    criteria = BaselineCriteria.sealed_defaults()
    binding = BaselineBinding(
        tree_id=TREE_ID,
        environment=env,
        workload=workload,
        metric_definitions=default_state_metric_definitions(),
        criteria=criteria,
    )
    restored = BaselineBinding.from_dict(binding.to_dict())
    assert restored.identity_id == binding.identity_id
    assert restored.tree_id == TREE_ID
    assert restored.criteria.identity_id == criteria.identity_id


# ---------------------------------------------------------------------------
# Missing vs zero
# ---------------------------------------------------------------------------


def test_metric_sample_distinguishes_missing_from_zero() -> None:
    assert metric_sample_distinguishes_missing_from_zero() is True

    zero = MetricSample.measured("rollback_count", 0)
    missing = MetricSample.unavailable(
        "rollback_count", UnavailableReason.TELEMETRY_MISSING
    )
    assert zero.status is TelemetryStatus.MEASURED
    assert zero.value == 0
    assert zero.measured_value() == 0
    assert "value" in zero.to_dict()
    assert "reason_code" not in zero.to_dict()

    assert missing.status is TelemetryStatus.UNAVAILABLE
    assert missing.measured_value() is None
    assert "value" not in missing.to_dict()
    assert missing.to_dict()["reason_code"] == "telemetry-missing"
    assert zero.to_dict() != missing.to_dict()


def test_unavailable_sample_rejects_numeric_encoding() -> None:
    with pytest.raises(DuckDBQuackBaselineError, match="must not encode"):
        MetricSample(
            metric_name="provider_calls",
            status=TelemetryStatus.UNAVAILABLE,
            sensor_id="sensor:x",
            value=3,
            reason_code="telemetry-missing",
        )


def test_measured_zero_is_not_treated_as_missing_in_state_baseline() -> None:
    baseline = establish_supervisor_state_baseline(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
    )
    rollback = baseline.aggregates["rollback_count"]
    failure = baseline.aggregates["failure_count"]
    assert rollback.is_measured
    assert rollback.value == 0
    assert failure.is_measured
    assert failure.value == 0
    assert baseline.measured_aggregate("rollback_count") == 0
    assert baseline.verdict is BaselineVerdict.ESTABLISHED


def test_missing_telemetry_is_unavailable_not_zero() -> None:
    baseline = establish_supervisor_state_baseline(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        missing_metrics=("lock_waits_ms",),
    )
    sample = baseline.aggregates["lock_waits_ms"]
    assert sample.is_unavailable
    assert sample.measured_value() is None
    # Measured zeros elsewhere remain measured zeros.
    assert baseline.aggregates["rollback_count"].is_measured
    assert baseline.aggregates["rollback_count"].value == 0


def test_llm_missing_quality_rejects_baseline() -> None:
    baseline = establish_llm_churn_baseline(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
        missing_metrics=("accepted_mutation_quality_bps",),
    )
    assert baseline.aggregates["accepted_mutation_quality_bps"].is_unavailable
    assert baseline.verdict is BaselineVerdict.REJECTED
    assert any("missing_metric" in code for code in baseline.reason_codes)


# ---------------------------------------------------------------------------
# Provider usage: rejected / retry / abandoned
# ---------------------------------------------------------------------------


def test_llm_churn_counts_rejected_retry_abandoned_provider_usage() -> None:
    baseline = establish_llm_churn_baseline(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(sample_count=16),
    )
    usage = baseline.provider_usage
    assert isinstance(usage, ProviderUsageCounters)
    assert usage.total == (
        usage.accepted + usage.rejected + usage.retry + usage.abandoned
    )
    # Hermetic workload must exercise non-accepted outcomes.
    assert usage.rejected >= 0
    assert usage.retry >= 0
    assert usage.abandoned >= 0
    # With sample_count=16 across 4 strata (4 each), rejected/retry are positive.
    assert usage.rejected > 0
    assert usage.retry > 0
    assert usage.abandoned > 0

    assert baseline.aggregates["rejected_provider_calls"].value == usage.rejected
    assert baseline.aggregates["retry_provider_calls"].value == usage.retry
    assert baseline.aggregates["abandoned_provider_calls"].value == usage.abandoned
    assert baseline.aggregates["provider_calls"].value == usage.total
    assert set(usage.to_dict()["outcomes"]) == {
        "accepted",
        "rejected",
        "retry",
        "abandoned",
    }
    assert baseline.verdict is BaselineVerdict.ESTABLISHED


def test_provider_usage_round_trip() -> None:
    usage = ProviderUsageCounters(accepted=3, rejected=1, retry=2, abandoned=1)
    restored = ProviderUsageCounters.from_dict(usage.to_dict())
    assert restored.total == 7
    assert restored.to_dict()["accepted"] == 3


# ---------------------------------------------------------------------------
# Criteria sealing: cannot regenerate with weakened floors
# ---------------------------------------------------------------------------


def test_sealed_criteria_identity_is_stable() -> None:
    a = BaselineCriteria.sealed_defaults()
    b = BaselineCriteria.sealed_defaults()
    assert a.identity_id == b.identity_id
    assert set(a.safety_floors) == set(SAFETY_FLOOR_KEYS)
    assert all(value == 0 for value in a.safety_floors.values())
    assert set(a.durability_floors) == set(DURABILITY_FLOOR_KEYS)
    assert all(a.durability_floors.values())


def test_cannot_regenerate_with_weakened_safety_criteria() -> None:
    sealed = BaselineCriteria.sealed_defaults()
    weakened = BaselineCriteria(
        safety_floors={
            **{key: 0 for key in SAFETY_FLOOR_KEYS},
            "false_completion": 1,
        },
        require_zero_safety_floors=False,
        durability_floors={key: True for key in DURABILITY_FLOOR_KEYS},
    )
    with pytest.raises(DuckDBQuackBaselineError, match="weaken"):
        assert_criteria_not_weakened(sealed, weakened)
    with pytest.raises(DuckDBQuackBaselineError, match="weaken"):
        regenerate_baseline_with_criteria(
            tree_id=TREE_ID,
            sealed_criteria=sealed,
            candidate_criteria=weakened,
            environment=_fixed_environment(),
            workload=_fixed_workload(),
        )


def test_cannot_regenerate_with_weakened_durability_criteria() -> None:
    sealed = BaselineCriteria.sealed_defaults()
    weakened = BaselineCriteria(
        safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
        durability_floors={
            **{key: True for key in DURABILITY_FLOOR_KEYS},
            "fsync_on_commit": False,
        },
    )
    weakened_flag, reasons = weakened.is_weakened_relative_to(sealed)
    assert weakened_flag is True
    assert any("durability_disabled" in code for code in reasons)
    with pytest.raises(DuckDBQuackBaselineError, match="weaken"):
        regenerate_baseline_with_criteria(
            tree_id=TREE_ID,
            sealed_criteria=sealed,
            candidate_criteria=weakened,
            environment=_fixed_environment(),
            workload=_fixed_workload(),
        )


def test_cannot_regenerate_with_weakened_quality_criteria() -> None:
    sealed = BaselineCriteria.sealed_defaults()
    weakened = BaselineCriteria(
        safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
        durability_floors={key: True for key in DURABILITY_FLOOR_KEYS},
        min_accepted_mutation_quality_bps=DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS
        - 1_000,
    )
    with pytest.raises(DuckDBQuackBaselineError, match="quality_floor_lowered|weaken"):
        assert_criteria_not_weakened(sealed, weakened)


def test_cannot_establish_baseline_with_weakened_criteria() -> None:
    weakened = BaselineCriteria(
        safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
        durability_floors={key: True for key in DURABILITY_FLOOR_KEYS},
        min_samples=1,  # lower than sealed default
    )
    with pytest.raises(DuckDBQuackBaselineError, match="weaken|min_samples"):
        establish_supervisor_state_baseline(
            tree_id=TREE_ID,
            environment=_fixed_environment(),
            workload=_fixed_workload(),
            criteria=weakened,
        )


def test_equal_or_stricter_criteria_may_regenerate() -> None:
    sealed = BaselineCriteria.sealed_defaults()
    stricter = BaselineCriteria(
        safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
        durability_floors={key: True for key in DURABILITY_FLOOR_KEYS},
        min_accepted_mutation_quality_bps=DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS
        + 500,
        max_rollback_rate_bps=1_000,
        max_failure_rate_bps=1_000,
        min_samples=DEFAULT_MIN_SAMPLES + 2,
    )
    assert_criteria_not_weakened(sealed, stricter)
    report = regenerate_baseline_with_criteria(
        tree_id=TREE_ID,
        sealed_criteria=sealed,
        candidate_criteria=stricter,
        environment=_fixed_environment(),
        workload=_fixed_workload(sample_count=16),
    )
    assert report.verdict is BaselineVerdict.ESTABLISHED
    assert (
        report.criteria.min_accepted_mutation_quality_bps
        == DEFAULT_MIN_ACCEPTED_MUTATION_QUALITY_BPS + 500
    )


# ---------------------------------------------------------------------------
# Determinism, round-trip, hermetic properties
# ---------------------------------------------------------------------------


def test_baseline_establishment_is_deterministic() -> None:
    env = _fixed_environment()
    workload = _fixed_workload()
    first = establish_duckdb_quack_baselines(
        tree_id=TREE_ID, environment=env, workload=workload
    )
    second = establish_duckdb_quack_baselines(
        tree_id=TREE_ID, environment=env, workload=workload
    )
    assert first.identity_id == second.identity_id
    assert first.state_baseline.identity_id == second.state_baseline.identity_id
    assert first.llm_churn_baseline.identity_id == second.llm_churn_baseline.identity_id
    assert first.to_dict() == second.to_dict()


def test_state_and_llm_reports_round_trip() -> None:
    report = establish_duckdb_quack_baselines(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
    )
    state = SupervisorStateBaseline.from_dict(report.state_baseline.to_dict())
    llm = LLMChurnBaseline.from_dict(report.llm_churn_baseline.to_dict())
    assert state.identity_id == report.state_baseline.identity_id
    assert llm.identity_id == report.llm_churn_baseline.identity_id
    # JSON-serializable for durable evidence.
    encoded = json.dumps(report.to_dict(), sort_keys=True)
    assert "SupervisorStateBaseline@1" in encoded
    assert "LLMChurnBaseline@1" in encoded


def test_warm_stratum_reduces_file_and_context_churn_vs_cold() -> None:
    baseline = establish_supervisor_state_baseline(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
    )
    by_stratum = {
        (
            item.stratum.value if hasattr(item.stratum, "value") else item.stratum
        ): item
        for item in baseline.strata
    }
    cold_reads = by_stratum["cold"].metrics["file_reads"].value
    warm_reads = by_stratum["warm"].metrics["file_reads"].value
    assert warm_reads < cold_reads

    llm = establish_llm_churn_baseline(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(),
    )
    llm_by = {
        (
            item.stratum.value if hasattr(item.stratum, "value") else item.stratum
        ): item
        for item in llm.strata
    }
    assert (
        llm_by["warm"].metrics["context_bytes"].value
        < llm_by["cold"].metrics["context_bytes"].value
    )
    assert (
        llm_by["warm"].metrics["duplicate_semantic_inputs"].value
        <= llm_by["cold"].metrics["duplicate_semantic_inputs"].value
    )


def test_insufficient_samples_verdict() -> None:
    # Sealed min_samples is 4; use criteria with min_samples=4 and only 1 sample
    # via workload — but establishment rejects weakened min_samples below sealed.
    # Instead force insufficient by using sealed criteria and a workload that
    # still meets establishment but evaluate path: sealed min is 4, workload
    # sample_count=4 yields 1 per stratum * 4 strata = 4 samples → established.
    # Use sample_count=3 → 3 samples total after integer division? 
    # samples_per_stratum = max(1, 3//4) = 1, 4 strata → 4 samples.
    # Always at least 4 with 4 strata. Use custom evaluation via missing quality.
    # Direct unit: establish with sample path that has min samples satisfied
    # is covered; insufficient is covered when min_samples high via stricter criteria.
    strict = BaselineCriteria(
        safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
        durability_floors={key: True for key in DURABILITY_FLOOR_KEYS},
        min_samples=100,
    )
    baseline = establish_supervisor_state_baseline(
        tree_id=TREE_ID,
        environment=_fixed_environment(),
        workload=_fixed_workload(sample_count=8),
        criteria=strict,
    )
    assert baseline.verdict is BaselineVerdict.INSUFFICIENT
    assert "insufficient_samples" in baseline.reason_codes


def test_negative_counts_rejected() -> None:
    with pytest.raises(DuckDBQuackBaselineError, match="non-negative"):
        MetricSample.measured("file_reads", -1)
    with pytest.raises(DuckDBQuackBaselineError, match="non-negative"):
        ProviderUsageCounters(accepted=-1)


def test_workload_requires_all_strata() -> None:
    with pytest.raises(DuckDBQuackBaselineError, match="missing"):
        WorkloadDefinition(
            workload_id="w1",
            seed=1,
            sample_count=4,
            strata=("cold", "warm"),
            operations=("read",),
        )


def test_unknown_safety_floor_key_rejected() -> None:
    with pytest.raises(DuckDBQuackBaselineError, match="unknown safety floor"):
        BaselineCriteria(safety_floors={"not_a_real_floor": 0})


def test_baseline_not_completion_or_mutation_authority() -> None:
    criteria = BaselineCriteria.sealed_defaults()
    assert criteria.completion_authoritative is False
    assert criteria.mutation_authorized is False
    with pytest.raises(DuckDBQuackBaselineError, match="completion or mutation|authority"):
        weakened = BaselineCriteria(
            safety_floors={key: 0 for key in SAFETY_FLOOR_KEYS},
            durability_floors={key: True for key in DURABILITY_FLOOR_KEYS},
            completion_authoritative=True,
        )
        weakened.assert_establishment_safe()
