from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.artifact_store import (
    PROOF_METRICS_KIND,
    query_artifact,
    read_proof_metrics_artifact,
)
from ipfs_accelerate_py.agent_supervisor.proof_metrics import (
    ASSURANCE_LEVELS,
    PROOF_LATENCY_FIELDS,
    PROOF_METRIC_DIMENSIONS,
    PROOF_METRICS_SCHEMA,
    PROOF_OPERATIONAL_COUNT_FIELDS,
    PROOF_RATE_FIELDS,
    build_proof_metrics_snapshot,
    normalize_proof_metric_identity,
    write_proof_metrics_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.scheduler_metrics import (
    build_scheduler_snapshot,
)


IDENTITY = {
    "goal_cid": "goal:g11",
    "subgoal_cid": "subgoal:g11.s4",
    "task_cid": "task:ref-260",
}


def _plan() -> dict[str, object]:
    return {
        "plan_id": "plan:proof-metrics",
        "repository_tree_id": "tree:candidate",
        "task_id": IDENTITY["task_cid"],
        "obligation_ids": ["obligation:lease"],
        "metadata": {**IDENTITY},
        "steps": [
            {
                "step_id": "solve",
                "obligation_id": "obligation:lease",
                "stage": "solve",
                "provider_id": "provider:hammer",
                "resource_class": "cpu-proof-solver",
                "depends_on": [],
                "metadata": {"template_id": "template:lease-uniqueness"},
            },
            {
                "step_id": "kernel",
                "obligation_id": "obligation:lease",
                "stage": "kernel_verify",
                "provider_id": "provider:lean",
                "resource_class": "cpu-proof-kernel",
                "depends_on": ["solve"],
                "metadata": {"template_id": "template:lease-uniqueness"},
            },
        ],
    }


def test_proof_snapshot_exposes_all_records_dimensions_and_latencies() -> None:
    snapshot = build_proof_metrics_snapshot(
        _plan(),
        attempts=[
            {
                "attempt_id": "attempt:solve",
                "plan_id": "plan:proof-metrics",
                "step_id": "solve",
                "obligation_id": "obligation:lease",
                "provider_id": "provider:hammer",
                "stage": "solve",
                "status": "succeeded",
                "started_at": "2026-07-23T00:00:00Z",
                "finished_at": "2026-07-23T00:00:02Z",
                "resource_usage": {"cpu_ms": 1200, "tokens": 50},
            },
            {
                "attempt_id": "attempt:kernel",
                "plan_id": "plan:proof-metrics",
                "step_id": "kernel",
                "obligation_id": "obligation:lease",
                "provider_id": "provider:lean",
                "stage": "kernel_verify",
                "status": "succeeded",
                "started_at": "2026-07-23T00:00:02Z",
                "finished_at": "2026-07-23T00:00:05Z",
            },
        ],
        receipts=[
            {
                "receipt_id": "receipt:kernel",
                "attempt_id": "attempt:kernel",
                "plan_id": "plan:proof-metrics",
                "obligation_id": "obligation:lease",
                "repository_tree_id": "tree:candidate",
                "provider_id": "provider:lean",
                "template_id": "template:lease-uniqueness",
                "resource_class": "cpu-proof-kernel",
                **IDENTITY,
                "verdict": "proved",
                "authoritative_verdict": "proved",
                "authoritative_assurance": "kernel_verified",
                "freshness": "current",
                "solver_id": "hammer/v1",
                "kernel_id": "lean/v4",
                "toolchain_id": "lean-toolchain:test",
                "policy_id": "policy:g11",
                "started_at": "2026-07-23T00:00:02Z",
                "finished_at": "2026-07-23T00:00:05Z",
            }
        ],
        cache_outcomes=[
            {
                **IDENTITY,
                "repository_tree_id": "tree:candidate",
                "provider_id": "provider:lean",
                "template_id": "template:lease-uniqueness",
                "resource_class": "cpu-proof-kernel",
                "cache_key": "cache:key",
                "obligation_id": "obligation:lease",
                "outcome": "hit",
                "cache_latency_ms": 7,
            }
        ],
        resource_samples=[
            {
                **IDENTITY,
                "repository_tree_id": "tree:candidate",
                "provider_id": "provider:lean",
                "template_id": "template:lease-uniqueness",
                "resource_class": "cpu-proof-kernel",
                "observed_at_ms": 1784764800000,
                "cpu_percent": 40,
                "memory_percent": 55,
                "disk_percent": 20,
                "memory_used_bytes": 1024,
                "disk_used_bytes": 2048,
            }
        ],
        events=[
            {
                **IDENTITY,
                "repository_tree_id": "tree:candidate",
                "provider_id": "provider:lean",
                "template_id": "template:lease-uniqueness",
                "resource_class": "cpu-proof-kernel",
                **{
                    field: index + 1
                    for index, field in enumerate(PROOF_LATENCY_FIELDS)
                },
            }
        ],
        generated_at="2026-07-23T00:00:06Z",
    )

    assert snapshot["schema"] == PROOF_METRICS_SCHEMA
    assert snapshot["source_counts"] == {
        "obligations": 1,
        "attempts": 2,
        "receipts": 1,
        "dependencies": 1,
        "cache_outcomes": 1,
        "resource_samples": 1,
        "events": 1,
    }
    assert snapshot["dependencies"][0]["target_step_id"] == "solve"
    assert snapshot["totals"]["attempt_count"] == 2
    assert snapshot["totals"]["receipt_count"] == 1
    assert snapshot["totals"]["cache_hit_count"] == 1
    assert snapshot["totals"]["resource_sample_count"] == 1
    assert {
        row["assurance"] for row in snapshot["assurance_counts"]
    } == set(ASSURANCE_LEVELS)
    assert sum(
        row["receipt_count"]
        for row in snapshot["assurance_counts"]
        if row["assurance"] == "kernel_verified"
    ) == 1
    for row in snapshot["metrics"]:
        assert all(row[name] != "unknown" for name in PROOF_METRIC_DIMENSIONS)
    for field in PROOF_LATENCY_FIELDS:
        assert snapshot["totals"][field] > 0


def test_json_and_duckdb_projections_are_equivalent_and_query_bounded(
    tmp_path: Path,
) -> None:
    snapshot = build_proof_metrics_snapshot(
        _plan(),
        cache_outcomes=[
            {
                **IDENTITY,
                "repository_tree_id": "tree:candidate",
                "provider_id": "provider:hammer",
                "template_id": "template:lease-uniqueness",
                "resource_class": "cpu-proof-solver",
                "outcome": "miss",
                "cache_latency_ms": 5,
            }
        ],
    )
    path = tmp_path / "proof_metrics.json"

    write_proof_metrics_snapshot(path, snapshot)

    assert read_proof_metrics_artifact(path)["query_store"]["artifact_kind"] == (
        PROOF_METRICS_KIND
    )
    assert (tmp_path / "proof_metrics.duckdb").exists()
    for table, expected in (
        ("proof_obligations", len(snapshot["obligations"])),
        ("proof_attempts", len(snapshot["attempts"])),
        ("proof_receipts", len(snapshot["receipts"])),
        ("proof_dependencies", len(snapshot["dependencies"])),
        ("proof_cache_outcomes", len(snapshot["cache_outcomes"])),
        ("proof_resource_samples", len(snapshot["resource_samples"])),
        ("proof_assurance_counts", len(snapshot["assurance_counts"])),
        ("proof_metrics", len(snapshot["metrics"])),
    ):
        json_query = query_artifact(path, table=table, limit=100)
        database_query = query_artifact(
            tmp_path / "proof_metrics.duckdb", table=table, limit=100
        )
        assert json_query["rows"] == database_query["rows"]
        assert json_query["row_count"] == expected


def test_query_projection_never_contains_witness_or_transcript_material(
    tmp_path: Path,
) -> None:
    secret = "PRIVATE-WITNESS-MATERIAL"
    transcript = "UNBOUNDED-SOLVER-TRANSCRIPT" * 10_000
    snapshot = build_proof_metrics_snapshot(
        _plan(),
        attempts=[
            {
                **IDENTITY,
                "attempt_id": "attempt:unsafe",
                "plan_id": "plan:proof-metrics",
                "step_id": "solve",
                "obligation_id": "obligation:lease",
                "provider_id": "provider:hammer",
                "stage": "solve",
                "status": "failed",
                "metadata": {
                    "hidden_witness": secret,
                    "solver_transcript": transcript,
                    "stdout": transcript,
                },
                "error_message": transcript,
            }
        ],
        receipts=[
            {
                **IDENTITY,
                "receipt_id": "receipt:unsafe",
                "attempt_id": "attempt:unsafe",
                "plan_id": "plan:proof-metrics",
                "obligation_id": "obligation:lease",
                "repository_tree_id": "tree:candidate",
                "provider_id": "provider:hammer",
                "template_id": "template:lease-uniqueness",
                "resource_class": "cpu-proof-solver",
                "verdict": "inconclusive",
                "evidence": [
                    {"metadata": {"private_premise": secret, "proof_log": transcript}}
                ],
            }
        ],
    )
    path = tmp_path / "safe.json"
    write_proof_metrics_snapshot(path, snapshot)

    json_text = path.read_text(encoding="utf-8")
    assert secret not in json_text
    assert "UNBOUNDED-SOLVER-TRANSCRIPT" not in json_text
    for table in ("proof_attempts", "proof_receipts", "proof_metrics"):
        encoded = json.dumps(query_artifact(path, table=table), sort_keys=True)
        assert secret not in encoded
        assert "UNBOUNDED-SOLVER-TRANSCRIPT" not in encoded


def test_general_scheduler_metrics_carry_proof_dimensions_and_latencies() -> None:
    event = {
        "type": "proof_solver_finished",
        **IDENTITY,
        "lane_id": "lane:g11.s4",
        "repository_tree_id": "tree:candidate",
        "provider_id": "provider:hammer",
        "template_id": "template:lease-uniqueness",
        "resource_class": "cpu-proof-solver",
        "solver_latency_ms": 125,
        "cache_latency_seconds": 0.01,
    }

    row = build_scheduler_snapshot([event])["metrics"][0]

    assert {name: row[name] for name in PROOF_METRIC_DIMENSIONS} == {
        "goal_cid": "goal:g11",
        "subgoal_cid": "subgoal:g11.s4",
        "task_cid": "task:ref-260",
        "repository_tree_id": "tree:candidate",
        "provider_id": "provider:hammer",
        "template_id": "template:lease-uniqueness",
        "resource_class": "cpu-proof-solver",
    }
    assert row["solver_latency_ms"] == 125
    assert row["cache_latency_ms"] == 10
    assert normalize_proof_metric_identity(event)["tree_id"] == "tree:candidate"


def test_all_persistence_and_rebuild_paths_reject_private_extensions(
    tmp_path: Path,
) -> None:
    snapshot = build_proof_metrics_snapshot(_plan())
    tampered = snapshot.to_dict()
    tampered["hidden_witness"] = "must-never-be-persisted"

    with pytest.raises(ValueError, match="private proof material|unsupported fields"):
        write_proof_metrics_snapshot(
            tmp_path / "json-only.json",
            tampered,
            queryable=False,
        )

    path = tmp_path / "tampered.json"
    write_proof_metrics_snapshot(path, snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["solver_transcript"] = "must-never-be-queryable"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="private proof material|unsupported fields"):
        query_artifact(path, kind=PROOF_METRICS_KIND, table="proof_metrics")


def test_receipts_fail_closed_and_distinct_attempts_are_not_collapsed() -> None:
    snapshot = build_proof_metrics_snapshot(
        _plan(),
        attempts=[
            {
                "attempt_id": attempt_id,
                "plan_id": "plan:proof-metrics",
                "step_id": "solve",
                "provider_id": "provider:hammer",
                "obligation_id": "obligation:lease",
                "stage": "solve",
                "status": "failed",
            }
            for attempt_id in ("attempt:one", "attempt:two")
        ],
        receipts=[
            {
                **IDENTITY,
                "receipt_id": "receipt:claim",
                "plan_id": "plan:proof-metrics",
                "attempt_id": "attempt:one",
                "obligation_id": "obligation:lease",
                "verdict": "proved",
                "assurance": "attested",
            }
        ],
    )

    assert len(snapshot["attempts"]) == 2
    assert snapshot["receipts"][0]["verdict"] == "inconclusive"
    assert snapshot["receipts"][0]["assurance"] == "unverified"
    assert snapshot["receipts"][0]["authoritative"] is False
    assert snapshot["totals"]["authoritative_receipt_count"] == 0


def test_multiple_plans_are_aggregated_with_plan_scoped_step_identity() -> None:
    first = _plan()
    second = {
        **_plan(),
        "plan_id": "plan:second",
        "repository_tree_id": "tree:second",
        "obligation_ids": ["obligation:second"],
        "steps": [
            {
                **dict(_plan()["steps"][0]),
                "obligation_id": "obligation:second",
                "provider_id": "provider:second",
            }
        ],
    }
    snapshot = build_proof_metrics_snapshot(plans=(first, second))

    assert snapshot["plan_ids"] == ["plan:proof-metrics", "plan:second"]
    assert {row["repository_tree_id"] for row in snapshot["obligations"]} == {
        "tree:candidate",
        "tree:second",
    }
    assert {row["provider_id"] for row in snapshot["obligations"]} == {
        "provider:lean",
        "provider:second",
    }


def test_operational_quality_metrics_are_dimensioned_and_sum_safe(
    tmp_path: Path,
) -> None:
    common = {
        **IDENTITY,
        "repository_tree_id": "tree:candidate",
        "template_id": "template:lease-uniqueness",
        "resource_class": "llm-proof-draft",
    }
    snapshot = build_proof_metrics_snapshot(
        _plan(),
        attempts=[
            {
                **common,
                "attempt_id": "attempt:model",
                "plan_id": "plan:proof-metrics",
                "step_id": "model",
                "obligation_id": "obligation:lease",
                "provider_id": "provider:route-a",
                "stage": "model_draft",
                "status": "succeeded",
                "schema_accepted": True,
                "deterministic_fallback": True,
                "repair_attempts": 2,
                "repair_converged": True,
                "unsupported_semantics": ["temporal:unbounded", "deontic:exception"],
                "resource_usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": 100,
                },
            }
        ],
        receipts=[
            {
                **common,
                "receipt_id": "receipt:closed",
                "attempt_id": "attempt:model",
                "plan_id": "plan:proof-metrics",
                "obligation_id": "obligation:lease",
                "provider_id": "provider:route-a",
                "authoritative_verdict": "proved",
                "authoritative_assurance": "kernel_verified",
            }
        ],
        cache_outcomes=[
            {**common, "provider_id": "provider:route-a", "outcome": "hit"},
            {**common, "provider_id": "provider:route-a", "outcome": "miss"},
        ],
        events=[
            {
                **common,
                "provider_id": "provider:route-a",
                "type": "route_availability",
                "availability_check_count": 1,
                "availability_success_count": 1,
            },
            {
                **common,
                "provider_id": "provider:route-b",
                "type": "route_availability",
                "availability_check_count": 3,
                "availability_success_count": 1,
                "availability_failure_count": 2,
                "schema_accepted": False,
                "false_completion_prevented": True,
            },
        ],
    )

    totals = snapshot["totals"]
    assert set(PROOF_OPERATIONAL_COUNT_FIELDS) <= set(totals)
    assert set(PROOF_RATE_FIELDS) <= set(totals)
    assert totals["availability_check_count"] == 4
    assert totals["availability_success_count"] == 2
    assert totals["availability_failure_count"] == 2
    assert totals["availability_rate"] == 0.5
    assert totals["schema_validation_count"] == 2
    assert totals["schema_acceptance_count"] == 1
    assert totals["schema_rejection_count"] == 1
    assert totals["schema_acceptance_rate"] == 0.5
    assert totals["proof_closure_count"] == 1
    assert totals["proof_closure_rate"] == 1.0
    assert totals["fallback_count"] == 1
    assert totals["fallback_rate"] == 1.0
    assert totals["repair_attempt_count"] == 2
    assert totals["repair_convergence_count"] == 1
    assert totals["repair_convergence_rate"] == 0.5
    assert totals["input_token_count"] == 80
    assert totals["output_token_count"] == 20
    assert totals["token_count"] == 100
    assert totals["cache_hit_rate"] == 0.5
    assert totals["unsupported_semantics_count"] == 2
    assert totals["false_completion_prevention_count"] == 1

    route_rows = {
        row["provider_id"]: row
        for row in snapshot["metrics"]
        if row["availability_check_count"]
    }
    assert route_rows["provider:route-a"]["availability_rate"] == 1.0
    assert route_rows["provider:route-b"]["availability_rate"] == pytest.approx(
        1 / 3, abs=1e-6
    )
    # The total is recomputed from additive counters, never by averaging the
    # two provider-local ratios.
    assert totals["availability_rate"] != pytest.approx(
        sum(row["availability_rate"] for row in route_rows.values())
        / len(route_rows)
    )

    path = tmp_path / "operational-proof-metrics.json"
    write_proof_metrics_snapshot(path, snapshot)
    persisted_metrics = query_artifact(
        path,
        table="proof_metrics",
        limit=100,
    )["rows"]
    persisted_attempts = query_artifact(
        path,
        table="proof_attempts",
        limit=100,
    )["rows"]
    assert persisted_metrics == snapshot["metrics"]
    assert persisted_attempts[0]["input_token_count"] == 80
    assert persisted_attempts[0]["output_token_count"] == 20
