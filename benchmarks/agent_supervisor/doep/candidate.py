"""DOEP-112 direct-supervisor candidate harness.

Candidates are proposal-only. The harness does not write DuckDB or complete
tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class CandidateHarnessError(ValueError):
    """Closed candidate-harness contract violation."""


@dataclass(frozen=True, slots=True)
class DirectSupervisorCandidate:
    candidate_id: str
    supervisor: str
    proposal_only: bool = True


def run_direct_supervisor_candidate_harness(
    candidates: Sequence[Mapping[str, Any]],
    *,
    write_duckdb: bool = False,
) -> dict[str, Any]:
    if write_duckdb:
        raise CandidateHarnessError("candidate harness cannot write DuckDB")
    if not candidates:
        raise CandidateHarnessError("empty candidate set")
    parsed = tuple(
        DirectSupervisorCandidate(
            candidate_id=str(item["candidate_id"]),
            supervisor=str(item["supervisor"]),
        )
        for item in candidates
    )
    return {
        "n": len(parsed),
        "candidate_ids": [item.candidate_id for item in parsed],
        "proposal_only": True,
        "completion_authority": False,
        "duckdb_written": False,
    }
