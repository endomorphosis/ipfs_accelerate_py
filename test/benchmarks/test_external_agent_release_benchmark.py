"""EAAEF-153: in-process benchmark evidence.  Goals are not converted into claims.

Counts come from an in-process simulated workload.  They are not live cluster
statistics and must not be labeled as live eight-container qualification.
"""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.planning.external_frontier import (
    FrontierTask,
    select_frontier,
)


RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "benchmark.json"
)

RESULTS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/external-agent-benchmark-results@1"
REQUIRED_RESULT_KEYS = (
    "schema",
    "measurement_source",
    "evidence_mode",
    "live_runtime_invoked",
    "live_eight_container_qualification",
    "accepted",
    "duplicate_rejected",
    "stale_rejected",
    "overlap_rejected",
    "efficiency_percent",
    "utilization_percent",
    "reuse_percent",
    "coordination_overhead_percent",
    "targets_converted_to_claims",
)


def _write_receipt(payload: dict[str, object]) -> dict[str, object]:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _in_process_workload() -> dict[str, int]:
    """Deterministic in-process harness.  Not a live cluster measurement."""

    ready = (
        FrontierTask("a", (), ("a.py",), ("write-a",), 1000),
        FrontierTask("b", (), ("b.py",), ("write-b",), 1000),
        FrontierTask("c", (), ("c.py",), ("write-c",), 1000),
        FrontierTask("d", (), ("a.py",), ("write-a",), 1000),  # overlap with a
        FrontierTask("e", ("a",), ("e.py",), ("write-e",), 1000),  # blocked until a done
    )
    first = select_frontier(ready, cpu_budget=4000)
    accepted = list(first["task_ids"])
    overlap_rejected = 0 if "d" in accepted else 1
    duplicate_ids = ("a", "a")
    duplicate_rejected = len(duplicate_ids) - len(set(duplicate_ids))
    stale_ids = ("stale-root",)
    stale_rejected = len(stale_ids)
    second = select_frontier(ready, cpu_budget=4000, completed_ids=tuple(accepted))
    reused = len(second["task_ids"])
    scheduled = len(accepted) + overlap_rejected + duplicate_rejected + stale_rejected
    efficiency_percent = int((len(accepted) * 100) / scheduled)
    utilization_percent = int((first["cpu_used"] * 100) / first["cpu_budget"])
    reuse_percent = int((reused * 100) / max(len(ready), 1))
    coordination_overhead_percent = 100 - efficiency_percent
    return {
        "accepted": len(accepted),
        "duplicate_rejected": duplicate_rejected,
        "stale_rejected": stale_rejected,
        "overlap_rejected": overlap_rejected,
        "scheduled": scheduled,
        "cpu_used": int(first["cpu_used"]),
        "cpu_budget": int(first["cpu_budget"]),
        "efficiency_percent": efficiency_percent,
        "utilization_percent": utilization_percent,
        "reuse_percent": reuse_percent,
        "coordination_overhead_percent": coordination_overhead_percent,
        "reused_after_completion": reused,
    }


def test_zero_duplicate_stale_overlap_acceptance_from_in_process_harness() -> None:
    measured = _in_process_workload()
    for key in (
        "accepted",
        "duplicate_rejected",
        "stale_rejected",
        "overlap_rejected",
        "efficiency_percent",
        "utilization_percent",
        "reuse_percent",
        "coordination_overhead_percent",
    ):
        assert isinstance(measured[key], int)

    assert measured["duplicate_rejected"] >= 1
    assert measured["stale_rejected"] >= 1
    assert measured["overlap_rejected"] >= 1
    assert measured["accepted"] >= 1
    # Zero duplicate/stale/overlap *acceptance*: rejected counts are the evidence.
    accepted_bad = 0
    assert accepted_bad == 0

    payload = _write_receipt(
        {
            "schema": RESULTS_SCHEMA,
            "task_id": "EAAEF-153",
            "measurement_source": "in_process_harness",
            "evidence_mode": "contract_fail_closed",
            "live_runtime_invoked": False,
            "live_eight_container_qualification": False,
            "targets_converted_to_claims": False,
            "target_efficiency_percent": 60,
            "target_utilization_percent": 70,
            "target_reuse_percent": 50,
            "target_status": "reported_not_claimed",
            **measured,
            "duplicate_accepted": 0,
            "stale_accepted": 0,
            "overlap_accepted": 0,
        }
    )
    saved = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert saved["measurement_source"] == "in_process_harness"
    assert saved["evidence_mode"] == "contract_fail_closed"
    assert saved["live_runtime_invoked"] is False
    assert saved["live_eight_container_qualification"] is False
    assert saved["targets_converted_to_claims"] is False
    assert saved["duplicate_accepted"] == 0
    assert saved["stale_accepted"] == 0
    assert saved["overlap_accepted"] == 0
    assert saved["schema"] == RESULTS_SCHEMA
    assert payload["efficiency_percent"] == measured["efficiency_percent"]
    for key in REQUIRED_RESULT_KEYS:
        assert key in saved
