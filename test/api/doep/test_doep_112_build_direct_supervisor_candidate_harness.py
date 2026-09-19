"""DOEP-112 direct-supervisor candidate harness."""

from __future__ import annotations

import pytest

from benchmarks.agent_supervisor.doep.candidate import (
    CandidateHarnessError,
    run_direct_supervisor_candidate_harness,
)


def test_candidates_are_proposal_only() -> None:
    result = run_direct_supervisor_candidate_harness(
        [{"candidate_id": "c1", "supervisor": "doep"}]
    )
    assert result["proposal_only"] is True
    assert result["completion_authority"] is False
    assert result["duckdb_written"] is False
    assert result["candidate_ids"] == ["c1"]


def test_harness_cannot_write_duckdb() -> None:
    with pytest.raises(CandidateHarnessError, match="cannot write DuckDB"):
        run_direct_supervisor_candidate_harness(
            [{"candidate_id": "c1", "supervisor": "doep"}],
            write_duckdb=True,
        )
