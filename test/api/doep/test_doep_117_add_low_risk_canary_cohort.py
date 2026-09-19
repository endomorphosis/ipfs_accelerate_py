"""DOEP-117 low-risk canary cohort."""

from __future__ import annotations

import pytest

from benchmarks.agent_supervisor.doep.low_risk_canary import (
    LowRiskCanaryError,
    run_low_risk_canary,
)


def test_canary_cannot_complete_or_write_duckdb() -> None:
    result = run_low_risk_canary([{"case_id": "c1", "risk_class": "R4"}])
    assert result["completion_authority"] is False
    assert result["authority_expanded"] is False
    with pytest.raises(LowRiskCanaryError, match="cannot complete"):
        run_low_risk_canary([{"case_id": "c1", "risk_class": "R4"}], complete_task=True)
    with pytest.raises(LowRiskCanaryError, match="cannot write DuckDB"):
        run_low_risk_canary([{"case_id": "c1", "risk_class": "R4"}], write_duckdb=True)
