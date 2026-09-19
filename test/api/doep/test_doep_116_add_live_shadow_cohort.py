"""DOEP-116 live shadow cohort."""

from __future__ import annotations

import pytest

from benchmarks.agent_supervisor.doep.live_shadow import (
    LiveShadowError,
    run_live_shadow_cohort,
)


def test_shadow_does_not_influence_live_or_complete() -> None:
    result = run_live_shadow_cohort(
        [
            {"case_id": "c1", "live_decision": "todo", "shadow_decision": "todo"},
            {"case_id": "c2", "live_decision": "todo", "shadow_decision": "retrying"},
        ]
    )
    assert result["n"] == 2
    assert result["mismatches"] == 1
    assert result["influences_live"] is False
    assert result["completion_authority"] is False


def test_shadow_cannot_write_duckdb() -> None:
    with pytest.raises(LiveShadowError, match="cannot write DuckDB"):
        run_live_shadow_cohort([], write_duckdb=True)
