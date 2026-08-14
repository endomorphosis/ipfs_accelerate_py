"""IPS-052: deterministic forty-transition benchmark workload.

Stable seed/input produces the same task sequence and expected unit sets.
Every transition records required/reused/invalidated/new counts plus
cache-hit, cost, and provenance fields.  Estimates are never labeled
measured.  Simulated required units never count as production proving.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO / "benchmarks/agent_supervisor/incremental_proof_sealer.py"
_SPEC = importlib.util.spec_from_file_location("ips_benchmark_workload", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["ips_benchmark_workload"] = _MOD
_SPEC.loader.exec_module(_MOD)

IncrementalProofBenchmark = _MOD.IncrementalProofBenchmark
BENCHMARK_ID = _MOD.BENCHMARK_ID
CSV_FIELDS = _MOD.CSV_FIELDS
DEFAULT_SEED = _MOD.DEFAULT_SEED
FULL_TRANSITIONS = _MOD.FULL_TRANSITIONS
METRICS = _MOD.METRICS
SCENARIOS = _MOD.SCENARIOS
TRANSITION_COUNT = _MOD.TRANSITION_COUNT
WORKLOAD_EVIDENCE = _MOD.WORKLOAD_EVIDENCE


def _bench(seed: int = DEFAULT_SEED) -> IncrementalProofBenchmark:
    return IncrementalProofBenchmark(seed=seed)


def test_evidence_subset_and_closed_workload_identity() -> None:
    assert WORKLOAD_EVIDENCE == "ips/benchmark-workload@1"
    assert BENCHMARK_ID == "incremental-proof-sealer-40-transition@1"
    assert TRANSITION_COUNT == 40
    assert len(SCENARIOS) == 40
    assert DEFAULT_SEED == 20260811
    assert 0 in FULL_TRANSITIONS
    assert 39 in FULL_TRANSITIONS


def test_stable_seed_repeats_task_sequence_and_unit_sets() -> None:
    first = _bench().evaluate()
    second = _bench().evaluate()
    assert [row["scenario"] for row in first] == list(SCENARIOS)
    assert [row["index"] for row in first] == list(range(40))
    assert [row["scenario"] for row in first] == [row["scenario"] for row in second]
    assert [row["required_units"] for row in first] == [row["required_units"] for row in second]
    assert [row["reused_units"] for row in first] == [row["reused_units"] for row in second]
    assert [row["invalidated_units"] for row in first] == [
        row["invalidated_units"] for row in second
    ]
    assert [row["added_units"] for row in first] == [row["added_units"] for row in second]
    assert [row["newly_proved_units"] for row in first] == [
        row["newly_proved_units"] for row in second
    ]
    assert [row["full_seal_root"] for row in first] == [row["full_seal_root"] for row in second]


def test_seed_changes_revisions_and_roots_but_not_scenario_order() -> None:
    default_rows = _bench(DEFAULT_SEED).evaluate()
    other_rows = _bench(DEFAULT_SEED + 1).evaluate()
    assert [row["scenario"] for row in default_rows] == [row["scenario"] for row in other_rows]
    assert [row["repository_revision"] for row in default_rows] != [
        row["repository_revision"] for row in other_rows
    ]
    assert [row["full_seal_root"] for row in default_rows] != [
        row["full_seal_root"] for row in other_rows
    ]


@pytest.mark.parametrize("index", range(40))
def test_every_transition_records_counts_and_provenance(index: int) -> None:
    row = _bench().evaluate()[index]
    assert row["index"] == index
    assert row["scenario"] == SCENARIOS[index]
    assert row["unit_count_provenance"] == "observed_planner_output"
    assert row["newly_proved_units"] == row["invalidated_units"] + row["added_units"]
    assert row["required_units"] == row["reused_units"] + row["newly_proved_units"]
    required = row["required_units"]
    expected_hit = 0.0 if required == 0 else row["reused_units"] / required
    assert row["cache_hit_rate"] == pytest.approx(expected_hit)
    assert set(row["metric_provenance"]) == set(METRICS)
    for metric in METRICS:
        source = row["metric_provenance"][metric]
        assert source in {"measured", "estimated", "unavailable"}
        if source == "unavailable":
            assert row[metric] is None
        else:
            assert source == "estimated"
            assert isinstance(row[metric], (int, float))
            assert row[metric] >= 0
            assert source != "measured"
    assert row["measurement_provenance"] in {"estimated", "mixed"}
    assert "measured" not in set(row["metric_provenance"].values())
    assert row["simulated_required_units"] == 0
    assert row["deterministic_roots_match"] is True
    assert row["full_seal_root"] == row["incremental_seal_root"]
    assert str(row["full_seal_root"]).startswith("sha256:")
    if row["seal_status"] == "sealed_full":
        assert isinstance(row["fallback_reason"], str) and row["fallback_reason"]
        assert row["reused_units"] == 0
    else:
        assert row["fallback_reason"] is None
        assert row["seal_status"] == "sealed_incremental"
    if index == 0:
        assert row["parent_seal"] is None
        assert row["seal_status"] == "sealed_full"
        assert row["fallback_reason"] == "first_state"
    else:
        assert str(row["parent_seal"]).startswith("sha256:")
    if index == 37:
        assert row["rejected_attempts"] == [
            {"kind": "wrong_parent", "terminal_status": "stale_parent"}
        ]
    else:
        assert row["rejected_attempts"] == []


def test_mandatory_full_checkpoints_and_honest_conditional_rows() -> None:
    rows = _bench().evaluate()
    for index in FULL_TRANSITIONS:
        assert rows[index]["seal_status"] == "sealed_full"
        assert rows[index]["fallback_reason"]
    for index in (17, 29, 38):
        assert rows[index]["seal_status"] in {"sealed_full", "sealed_incremental"}
        if rows[index]["seal_status"] == "sealed_full":
            assert rows[index]["fallback_reason"]
        else:
            assert rows[index]["fallback_reason"] is None
    for index in (14, 36):
        assert rows[index]["seal_status"] == "sealed_incremental"
        assert rows[index]["fallback_reason"] is None


def test_docs_and_unrelated_edits_reuse_instead_of_reprove() -> None:
    rows = _bench().evaluate()
    docs = rows[2]
    ordinary = rows[11]
    docs_only = rows[21]
    later_docs = rows[34]
    for row in (docs, ordinary, docs_only, later_docs):
        assert row["seal_status"] == "sealed_incremental"
        assert row["reused_units"] >= 1
        assert row["cache_hit_rate"] >= 0.7
        assert row["newly_proved_units"] == 0


def test_localized_source_invalidates_only_affected_module() -> None:
    row = _bench().evaluate()[1]
    assert row["seal_status"] == "sealed_incremental"
    assert row["invalidated_units"] >= 1
    assert row["reused_units"] >= 1
    assert set(row["invalidated_unit_ids"]).isdisjoint(set(row["reused_unit_ids"]))
    assert "unit/static_a" in row["invalidated_unit_ids"]
    assert "unit/static_b" in row["reused_unit_ids"]


def test_addition_and_authorized_deletion_update_expected_sets() -> None:
    rows = _bench().evaluate()
    added = rows[8]
    deleted = rows[9]
    assert added["added_units"] == 1
    assert "unit/test_c" in added["added_unit_ids"]
    assert added["required_units"] == added["reused_units"] + added["newly_proved_units"]
    assert deleted["removed_units"] == 1
    assert "unit/test_c" in deleted["removed_unit_ids"]
    assert "unit/test_c" not in deleted["required_unit_ids"]


def test_report_schema_and_cli_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"
    argv = [
        "python",
        "benchmarks/agent_supervisor/incremental_proof_sealer.py",
        "--seed",
        str(DEFAULT_SEED),
        "--transitions",
        "40",
        "--json-output",
        str(json_path),
        "--csv-output",
        str(csv_path),
    ]
    payload = _bench().report(json_output=str(json_path), csv_output=str(csv_path), argv=argv)
    assert payload["schema_version"] == "incremental-proof-sealer-benchmark-results@2"
    assert payload["benchmark_id"] == BENCHMARK_ID
    assert payload["seed"] == DEFAULT_SEED
    assert payload["transition_count"] == 40
    assert "evidence_subset" not in payload
    assert payload["capabilities"]["real_prover_available"] is False
    assert payload["capabilities"]["gpu_available"] is False
    assert "simulated" in payload["capabilities"]["notes"].lower()
    assert json_path.is_file()
    assert csv_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["transition_count"] == 40
    assert len(loaded["transitions"]) == 40
    text = csv_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text), strict=True)
    csv_rows = list(reader)
    assert reader.fieldnames == list(CSV_FIELDS)
    assert len(csv_rows) == 40
    assert csv_rows[0]["scenario"] == SCENARIOS[0]
    assert csv_rows[37]["index"] == "37"


def test_cli_main_writes_both_outputs(tmp_path: Path) -> None:
    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"
    rc = _MOD.main(
        [
            "--seed",
            str(DEFAULT_SEED),
            "--transitions",
            "40",
            "--json-output",
            str(json_path),
            "--csv-output",
            str(csv_path),
            "--repo-root",
            str(_REPO),
        ]
    )
    assert rc == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["execution_context"]["process_observed"] is True
    assert payload["execution_context"]["test_execution_cryptographically_proven"] is False
    assert payload["transitions"][0]["seal_status"] == "sealed_full"
    assert payload["transitions"][37]["rejected_attempts"][0]["kind"] == "wrong_parent"


def test_estimates_are_never_sold_as_measurements() -> None:
    for row in _bench().evaluate():
        assert row["measurement_provenance"] != "measured"
        assert all(source != "measured" for source in row["metric_provenance"].values())
        full = row["full_proof_cost"]
        incremental = row["incremental_proof_cost"]
        expected = 0.0 if full == 0 else (full - incremental) / full * 100.0
        assert row["compute_saved_percent"] == pytest.approx(expected)
