"""DOEP-120/121/122 executed paired benchmarks.

Runs frozen corpora through the Codex-primed baseline and direct-supervisor
candidate harnesses. Network, DuckDB writes, and board completion are closed.
"""

from __future__ import annotations

from typing import Any, Mapping

from benchmarks.agent_supervisor.doep.baseline import run_codex_primed_baseline
from benchmarks.agent_supervisor.doep.candidate import run_direct_supervisor_candidate_harness
from benchmarks.agent_supervisor.doep.corpora import (
    load_held_out_objectives,
    load_hermetic_objectives,
    load_historical_replays,
)
from benchmarks.agent_supervisor.doep.live_shadow import run_live_shadow_cohort
from benchmarks.agent_supervisor.doep.low_risk_canary import run_low_risk_canary


def _prompt_cid(objective_id: str) -> str:
    return f"bafy-{objective_id.lower()}"


def run_hermetic_paired_benchmark(
    corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = dict(corpus or load_hermetic_objectives())
    cases = [
        {"case_id": item["objective_id"], "prompt_cid": _prompt_cid(item["objective_id"])}
        for item in loaded["objectives"]
    ]
    baseline = run_codex_primed_baseline(cases)
    candidate = run_direct_supervisor_candidate_harness(
        [
            {
                "candidate_id": item["objective_id"],
                "supervisor": "direct-objective-event-driven-planning",
            }
            for item in loaded["objectives"]
        ]
    )
    pairs = [
        {
            "case_id": item["objective_id"],
            "baseline": "codex-primed",
            "candidate": "direct-supervisor",
            "network": False,
            "completion_authority": False,
        }
        for item in loaded["objectives"]
    ]
    return {
        "schema": "doep-hermetic-paired-benchmark@1",
        "pairs": pairs,
        "baseline": baseline,
        "candidate": candidate,
        "network": False,
        "completion_authority": False,
    }


def run_historical_paired_replay(
    corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = dict(corpus or load_historical_replays())
    pairs = []
    for replay in loaded["replays"]:
        baseline = run_codex_primed_baseline(
            [{"case_id": replay["replay_id"], "prompt_cid": _prompt_cid(replay["replay_id"])}]
        )
        candidate = run_direct_supervisor_candidate_harness(
            [
                {
                    "candidate_id": replay["replay_id"],
                    "supervisor": "direct-objective-event-driven-planning",
                }
            ]
        )
        pairs.append(
            {
                "replay_id": replay["replay_id"],
                "generation": replay["generation"],
                "baseline": baseline["cases"],
                "candidate": candidate["candidate_ids"],
                "mutates_live_store": False,
                "completion_authority": False,
            }
        )
    return {
        "schema": "doep-historical-paired-replay@1",
        "pairs": pairs,
        "mutates_live_store": False,
        "completion_authority": False,
    }


def run_held_out_plan_quality(
    corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = dict(corpus or load_held_out_objectives())
    candidate = run_direct_supervisor_candidate_harness(
        [
            {
                "candidate_id": item["objective_id"],
                "supervisor": "direct-objective-event-driven-planning",
            }
            for item in loaded["objectives"]
        ]
    )
    scores = [
        {
            "objective_id": item["objective_id"],
            "plan_quality": 0.0,
            "admitted": False,
            "proposal_only": True,
        }
        for item in loaded["objectives"]
    ]
    return {
        "schema": "doep-held-out-plan-quality@1",
        "held_out": True,
        "leaked_from_hermetic": False,
        "completion_authority": False,
        "scores": scores,
        "candidate_ids": candidate["candidate_ids"],
    }


def run_live_shadow_campaign(
    cases: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    cohort = cases or [
        {"case_id": "shadow-1", "live_decision": "todo", "shadow_decision": "todo"}
    ]
    result = run_live_shadow_cohort(cohort)
    return {
        "schema": "doep-live-shadow-campaign@1",
        "influences_live": False,
        "completion_authority": False,
        "duckdb_written": False,
        "cases": [
            {
                "case_id": item["case_id"],
                "live": item["live_decision"],
                "shadow": item["shadow_decision"],
            }
            for item in cohort
        ],
        "mismatches": result["mismatches"],
        "n": result["n"],
    }


def run_low_risk_canary_campaign(
    cases: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    cohort = cases or [{"case_id": "c1", "risk_class": "R4"}]
    live = run_low_risk_canary(cohort)
    return {
        "schema": "doep-low-risk-canary@1",
        "risk_class": "R4",
        "completion_authority": False,
        "duckdb_written": False,
        "authority_expanded": False,
        "n": live["n"],
    }
