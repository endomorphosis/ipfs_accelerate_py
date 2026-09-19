"""DOEP-117 low-risk canary cohort.

Canaries cannot complete tasks, write DuckDB, or expand authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class LowRiskCanaryError(ValueError):
    """Closed canary contract violation."""


@dataclass(frozen=True, slots=True)
class LowRiskCanary:
    case_id: str
    risk_class: str


def run_low_risk_canary(
    cases: Sequence[Mapping[str, Any]],
    *,
    complete_task: bool = False,
    write_duckdb: bool = False,
) -> dict[str, Any]:
    if complete_task:
        raise LowRiskCanaryError("canary cannot complete tasks")
    if write_duckdb:
        raise LowRiskCanaryError("canary cannot write DuckDB")
    parsed = tuple(
        LowRiskCanary(case_id=str(item["case_id"]), risk_class=str(item.get("risk_class") or "R4"))
        for item in cases
    )
    if any(item.risk_class not in {"R4", "low"} for item in parsed):
        raise LowRiskCanaryError("canary cohort must stay low-risk")
    return {
        "n": len(parsed),
        "completion_authority": False,
        "authority_expanded": False,
    }
