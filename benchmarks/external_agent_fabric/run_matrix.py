"""Measure configurations A-D (EAAEF-151)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from .harness import CONFIGURATIONS, BenchmarkRun, matrix_for


MATRIX_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-benchmark-matrix@1"


class MatrixError(ValueError):
    """Benchmark matrix is incomplete."""


def measure(run: BenchmarkRun, *, observed: Mapping[str, int]) -> Mapping[str, Any]:
    required = (
        "completed",
        "accepted_patches",
        "wall_ms",
        "first_useful_ms",
        "tokens",
        "proofs",
        "duplicates",
        "conflicts",
        "retries",
    )
    missing = [key for key in required if key not in observed]
    if missing:
        raise MatrixError(f"missing observation {missing[0]}")
    if any(int(observed[key]) < 0 for key in required):
        raise MatrixError("observations must be nonnegative")
    return MappingProxyType(
        {
            "schema": MATRIX_SCHEMA,
            "configuration": run.configuration,
            "observed": {key: int(observed[key]) for key in required},
            "estimated": {},
        }
    )


def measure_matrix(
    *,
    task_id: str,
    repository_id: str,
    authority_id: str,
    image_digest: str,
    model_id: str,
    provider_id: str,
    prover_id: str,
    budget_id: str,
    observations: Mapping[str, Mapping[str, int]],
) -> tuple[Mapping[str, Any], ...]:
    runs = matrix_for(
        task_id=task_id,
        repository_id=repository_id,
        authority_id=authority_id,
        image_digest=image_digest,
        model_id=model_id,
        provider_id=provider_id,
        prover_id=prover_id,
        budget_id=budget_id,
    )
    if set(observations) != set(CONFIGURATIONS):
        raise MatrixError("observations must cover configurations A-D exactly")
    return tuple(measure(run, observed=observations[run.configuration]) for run in runs)
