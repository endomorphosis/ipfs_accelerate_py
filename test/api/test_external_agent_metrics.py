"""EAAEF-131: estimates are not observations."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.observability.external_metrics import (
    ExternalMetrics,
    MetricsError,
)


def test_observed_values_are_readable() -> None:
    metrics = ExternalMetrics(
        run_id="run-1",
        task_id="EAAEF-131",
        observed={"tokens": 12, "proofs": 1},
        estimated={"wall_ms": 4000},
    )
    assert metrics.as_observation("tokens") == 12
    with pytest.raises(MetricsError, match="estimate"):
        metrics.as_observation("wall_ms")


def test_overlap_and_negative_fail() -> None:
    with pytest.raises(MetricsError, match="overlap"):
        ExternalMetrics(
            run_id="run-1",
            task_id="t",
            observed={"tokens": 1},
            estimated={"tokens": 2},
        )
    with pytest.raises(MetricsError, match="nonnegative"):
        ExternalMetrics(run_id="r", task_id="t", observed={"n": -1}, estimated={})
