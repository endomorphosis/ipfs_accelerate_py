"""EAAEF-150: harness preserves identities across configurations A-D."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.external_agent_fabric.harness import CONFIGURATIONS, HarnessError, matrix_for


def test_matrix_shares_all_identities() -> None:
    digest = "sha256:" + ("a" * 64)
    runs = matrix_for(
        task_id="EAAEF-150",
        repository_id="repo",
        authority_id="did:key:op",
        image_digest=digest,
        model_id="grok-4.6",
        provider_id="grok_cli",
        prover_id="none",
        budget_id="budget-1",
    )
    assert tuple(run.configuration for run in runs) == CONFIGURATIONS
    keys = [
        (run.task_id, run.repository_id, run.authority_id, run.image_digest, run.model_id)
        for run in runs
    ]
    assert len(set(keys)) == 1


def test_missing_digest_fails() -> None:
    with pytest.raises(HarnessError):
        matrix_for(
            task_id="t",
            repository_id="r",
            authority_id="a",
            image_digest="latest",
            model_id="m",
            provider_id="p",
            prover_id="pr",
            budget_id="b",
        )
