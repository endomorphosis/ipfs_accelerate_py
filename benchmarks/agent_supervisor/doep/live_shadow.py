"""DOEP-116 live shadow cohort.

Shadow observations sit beside live dispatch. They cannot complete tasks or
write DuckDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class LiveShadowError(ValueError):
    """Closed live-shadow contract violation."""


@dataclass(frozen=True, slots=True)
class LiveShadowCohort:
    case_id: str
    live_decision: str
    shadow_decision: str


def run_live_shadow_cohort(
    cases: Sequence[Mapping[str, Any]],
    *,
    write_duckdb: bool = False,
) -> dict[str, Any]:
    if write_duckdb:
        raise LiveShadowError("live shadow cannot write DuckDB")
    parsed = tuple(
        LiveShadowCohort(
            case_id=str(item["case_id"]),
            live_decision=str(item["live_decision"]),
            shadow_decision=str(item["shadow_decision"]),
        )
        for item in cases
    )
    mismatches = sum(
        1 for item in parsed if item.live_decision != item.shadow_decision
    )
    return {
        "n": len(parsed),
        "mismatches": mismatches,
        "influences_live": False,
        "completion_authority": False,
    }
