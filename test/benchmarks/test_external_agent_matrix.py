"""EAAEF-151: A-D matrix observations, no estimates-as-live."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.external_agent_fabric.run_matrix import MatrixError, measure_matrix


OBS = {
    "completed": 1,
    "accepted_patches": 1,
    "wall_ms": 10,
    "first_useful_ms": 4,
    "tokens": 8,
    "proofs": 0,
    "duplicates": 0,
    "conflicts": 0,
    "retries": 0,
}


def test_measures_all_four_configurations() -> None:
    digest = "sha256:" + ("b" * 64)
    rows = measure_matrix(
        task_id="t",
        repository_id="r",
        authority_id="a",
        image_digest=digest,
        model_id="m",
        provider_id="p",
        prover_id="pr",
        budget_id="b",
        observations={letter: dict(OBS) for letter in "ABCD"},
    )
    assert [row["configuration"] for row in rows] == ["A", "B", "C", "D"]
    assert all(row["estimated"] == {} for row in rows)


def test_missing_configuration_fails() -> None:
    digest = "sha256:" + ("b" * 64)
    with pytest.raises(MatrixError, match="A-D"):
        measure_matrix(
            task_id="t",
            repository_id="r",
            authority_id="a",
            image_digest=digest,
            model_id="m",
            provider_id="p",
            prover_id="pr",
            budget_id="b",
            observations={"A": dict(OBS)},
        )
