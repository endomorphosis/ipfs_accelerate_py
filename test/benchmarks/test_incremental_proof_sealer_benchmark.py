"""IPS-052: deterministic forty-transition IncrementalProofSealer benchmark."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    REPO_ROOT / "benchmarks" / "agent_supervisor" / "incremental_proof_sealer.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "ips_052_incremental_proof_sealer_benchmark", BENCHMARK_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

DEFAULT_SEED = bench.DEFAULT_SEED
SCENARIOS = bench.SCENARIOS
FULL_TRANSITIONS = bench.FULL_TRANSITIONS
CONDITIONAL_FULL_TRANSITIONS = bench.CONDITIONAL_FULL_TRANSITIONS
METRIC_FIELDS = bench.METRIC_FIELDS
CSV_FIELDS = bench.CSV_FIELDS
IncrementalProofBenchmark = bench.IncrementalProofBenchmark
BenchmarkError = bench.BenchmarkError


def _bindings(seed: int = DEFAULT_SEED) -> tuple[str, dict[str, str], dict[str, str]]:
    parent = hashlib.sha1(f"parent:{seed}".encode()).hexdigest()
    revisions = {
        "accelerate": parent,
        "datasets": hashlib.sha1(f"datasets:{seed}".encode()).hexdigest(),
        "kit": hashlib.sha1(f"kit:{seed}".encode()).hexdigest(),
    }
    trees = {
        "accelerate": hashlib.sha1(f"tree-a:{seed}".encode()).hexdigest(),
        "datasets": hashlib.sha1(f"tree-d:{seed}".encode()).hexdigest(),
        "kit": hashlib.sha1(f"tree-k:{seed}".encode()).hexdigest(),
    }
    return parent, revisions, trees


def _run_payload(
    *,
    seed: int = DEFAULT_SEED,
    transitions: int = 40,
    gpu_available: bool = False,
) -> dict[str, Any]:
    parent, revisions, trees = _bindings(seed)
    benchmark = IncrementalProofBenchmark(
        seed=seed,
        transition_count=transitions,
        gpu_available=gpu_available,
    )
    return benchmark.run(
        benchmark_worktree_parent_revision=parent,
        source_revisions=revisions,
        source_trees=trees,
    )


def test_module_surface_and_evidence() -> None:
    assert BENCHMARK_PATH.is_file()
    assert bench.EVIDENCE_SUBSET == "ips/benchmark-workload@1"
    assert bench.BENCHMARK_INTERFACE == "IncrementalProofBenchmark@1"
    assert bench.BENCHMARK_SCHEMA == "incremental-proof-sealer-benchmark-results@2"
    assert bench.BENCHMARK_ID == "incremental-proof-sealer-40-transition@1"
    assert len(SCENARIOS) == 40
    assert 0 in FULL_TRANSITIONS and 39 in FULL_TRANSITIONS
    assert CONDITIONAL_FULL_TRANSITIONS == frozenset({17, 29, 38})


def test_stable_seed_produces_identical_task_sequence_and_unit_sets() -> None:
    first = IncrementalProofBenchmark(seed=DEFAULT_SEED, transition_count=40)
    second = IncrementalProofBenchmark(seed=DEFAULT_SEED, transition_count=40)
    other_seed = IncrementalProofBenchmark(seed=DEFAULT_SEED + 99, transition_count=40)

    seq_a = first.task_sequence()
    seq_b = second.task_sequence()
    assert len(seq_a) == 40
    assert [(item.index, item.scenario, item.kind) for item in seq_a] == [
        (item.index, item.scenario, item.kind) for item in seq_b
    ]
    assert [item.scenario for item in seq_a] == list(SCENARIOS)
    # Workload identity is seed-independent; only estimated metrics/roots salt.
    assert first.workload_fingerprint() == second.workload_fingerprint()
    assert first.workload_fingerprint() == other_seed.workload_fingerprint()

    units_a = first.expected_unit_sets()
    units_b = second.expected_unit_sets()
    assert len(units_a) == 40
    for left, right in zip(units_a, units_b):
        assert left == right
        assert left.required == left.reused | left.newly_proved
        assert left.newly_proved == left.invalidated | left.added
        assert not (left.reused & left.newly_proved)
        assert left.seal_status in {"sealed_full", "sealed_incremental"}


def test_different_seed_keeps_sequence_but_changes_metric_roots() -> None:
    parent_a, rev_a, trees_a = _bindings(DEFAULT_SEED)
    parent_b, rev_b, trees_b = _bindings(DEFAULT_SEED + 1)
    a = IncrementalProofBenchmark(seed=DEFAULT_SEED).run(
        benchmark_worktree_parent_revision=parent_a,
        source_revisions=rev_a,
        source_trees=trees_a,
    )
    b = IncrementalProofBenchmark(seed=DEFAULT_SEED + 1).run(
        benchmark_worktree_parent_revision=parent_b,
        source_revisions=rev_b,
        source_trees=trees_b,
    )
    assert [row["scenario"] for row in a["transitions"]] == [
        row["scenario"] for row in b["transitions"]
    ]
    assert a["transitions"][1]["full_seal_root"] != b["transitions"][1]["full_seal_root"]
    assert (
        a["transitions"][1]["prover_cpu_seconds"]
        != b["transitions"][1]["prover_cpu_seconds"]
    )


def test_every_transition_records_counts_metrics_and_provenance() -> None:
    payload = _run_payload()
    assert payload["schema_version"] == bench.BENCHMARK_SCHEMA
    assert payload["benchmark_id"] == bench.BENCHMARK_ID
    assert payload["seed"] == DEFAULT_SEED
    assert payload["transition_count"] == 40
    assert len(payload["transitions"]) == 40

    required_row_keys = {
        "index",
        "scenario",
        "repository_revision",
        "parent_seal",
        "seal_status",
        "required_units",
        "reused_units",
        "invalidated_units",
        "added_units",
        "removed_units",
        "newly_proved_units",
        "unit_count_provenance",
        "cache_hit_rate",
        *METRIC_FIELDS,
        "metric_provenance",
        "measurement_provenance",
        "compute_saved_percent",
        "chain_depth",
        "fallback_reason",
        "full_seal_root",
        "incremental_seal_root",
        "deterministic_roots_match",
        "simulated_required_units",
        "rejected_attempts",
    }

    prior_root = None
    for index, row in enumerate(payload["transitions"]):
        assert set(row) == required_row_keys
        assert row["index"] == index
        assert row["scenario"] == SCENARIOS[index]
        assert row["unit_count_provenance"] == "observed_planner_output"
        assert row["newly_proved_units"] == (
            row["invalidated_units"] + row["added_units"]
        )
        assert row["required_units"] == (
            row["reused_units"] + row["newly_proved_units"]
        )
        expected_hit = (
            0.0
            if row["required_units"] == 0
            else row["reused_units"] / row["required_units"]
        )
        assert abs(row["cache_hit_rate"] - expected_hit) < 1e-12
        assert row["simulated_required_units"] == 0
        assert row["deterministic_roots_match"] is True
        assert row["full_seal_root"] == row["incremental_seal_root"]
        assert row["full_seal_root"].startswith("sha256:")
        assert len(row["full_seal_root"]) == 7 + 64
        assert row["parent_seal"] == prior_root
        prior_root = row["full_seal_root"]

        provenance = row["metric_provenance"]
        assert set(provenance) == set(METRIC_FIELDS)
        for metric in METRIC_FIELDS:
            source = provenance[metric]
            assert source in {"measured", "estimated", "unavailable"}
            value = row[metric]
            if source == "unavailable":
                assert value is None
            else:
                assert isinstance(value, (int, float)) and value >= 0

        sources = set(provenance.values())
        if sources == {"measured"}:
            assert row["measurement_provenance"] == "measured"
        elif sources == {"estimated"}:
            assert row["measurement_provenance"] == "estimated"
        else:
            assert row["measurement_provenance"] == "mixed"

        full_cost = row["full_proof_cost"]
        inc_cost = row["incremental_proof_cost"]
        if full_cost is not None and inc_cost is not None:
            expected_savings = (
                0.0
                if full_cost == 0
                else (full_cost - inc_cost) / full_cost * 100.0
            )
            assert abs(row["compute_saved_percent"] - expected_savings) < 1e-9

        if index in FULL_TRANSITIONS or (
            index in CONDITIONAL_FULL_TRANSITIONS and row["seal_status"] == "sealed_full"
        ):
            if index in FULL_TRANSITIONS:
                assert row["seal_status"] == "sealed_full"
            assert isinstance(row["fallback_reason"], str) and row["fallback_reason"]
        elif row["seal_status"] == "sealed_incremental":
            assert row["fallback_reason"] is None

        if index == 37:
            assert row["rejected_attempts"] == [
                {"kind": "wrong_parent", "terminal_status": "stale_parent"}
            ]
        else:
            assert row["rejected_attempts"] == []


def test_mandatory_and_conditional_seal_statuses() -> None:
    payload = _run_payload()
    for index, row in enumerate(payload["transitions"]):
        status = row["seal_status"]
        if index in FULL_TRANSITIONS:
            assert status == "sealed_full"
        elif index in CONDITIONAL_FULL_TRANSITIONS:
            assert status in {"sealed_full", "sealed_incremental"}
        else:
            assert status == "sealed_incremental"
    # Closed honest decisions for the conditional trio.
    assert payload["transitions"][17]["seal_status"] == "sealed_full"
    assert payload["transitions"][29]["seal_status"] == "sealed_incremental"
    assert payload["transitions"][38]["seal_status"] == "sealed_incremental"


def test_documentation_transitions_have_near_total_reuse() -> None:
    units = IncrementalProofBenchmark(seed=DEFAULT_SEED).expected_unit_sets()
    for index in (2, 11, 21, 34):
        row = units[index]
        assert row.seal_status == "sealed_incremental"
        assert len(row.invalidated) == 0
        assert len(row.added) == 0
        assert len(row.reused) == len(row.required)
        assert len(row.required) > 0


def test_localized_source_invalidates_only_core_module() -> None:
    units = IncrementalProofBenchmark(seed=DEFAULT_SEED).expected_unit_sets()
    for index in (1, 13, 23):
        row = units[index]
        assert "unit/module/core" in row.invalidated
        assert row.newly_proved == row.invalidated
        assert len(row.reused) >= len(row.invalidated)


def test_gpu_unavailable_is_honest() -> None:
    payload = _run_payload(gpu_available=False)
    row = payload["transitions"][0]
    assert row["metric_provenance"]["prover_gpu_seconds"] == "unavailable"
    assert row["prover_gpu_seconds"] is None
    assert row["measurement_provenance"] == "mixed"
    assert payload["capabilities"]["gpu_available"] is False
    assert payload["capabilities"]["real_prover_available"] is False
    assert "estimated" in payload["capabilities"]["notes"].lower()


def test_gpu_available_marks_estimated_gpu_counter() -> None:
    payload = _run_payload(gpu_available=True)
    row = payload["transitions"][1]
    assert row["metric_provenance"]["prover_gpu_seconds"] == "estimated"
    assert isinstance(row["prover_gpu_seconds"], float)
    assert row["prover_gpu_seconds"] >= 0
    assert row["measurement_provenance"] == "estimated"


def test_csv_is_exact_ordered_scalar_projection(tmp_path: Path) -> None:
    parent, revisions, trees = _bindings()
    benchmark = IncrementalProofBenchmark(seed=DEFAULT_SEED)
    payload = benchmark.run(
        benchmark_worktree_parent_revision=parent,
        source_revisions=revisions,
        source_trees=trees,
    )
    json_path = tmp_path / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"
    benchmark.write_artifacts(payload, json_output=json_path, csv_output=csv_path)

    raw = json_path.read_bytes()
    assert raw == bench._canonical_json_bytes(payload) + b"\n"
    text = csv_path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text), strict=True)
    assert list(reader.fieldnames or []) == list(CSV_FIELDS)
    rows = list(reader)
    assert len(rows) == 40
    for index, (csv_row, json_row) in enumerate(zip(rows, payload["transitions"])):
        for field in CSV_FIELDS:
            actual = csv_row[field]
            expected = json_row[field]
            if expected is None:
                assert actual == ""
            elif isinstance(expected, bool):
                assert actual == str(expected).lower()
            elif isinstance(expected, int):
                assert actual == str(expected)
            elif isinstance(expected, float):
                assert float(actual) == expected
            else:
                assert actual == str(expected)
        assert int(csv_row["index"]) == index


def test_run_is_byte_stable_for_fixed_bindings() -> None:
    parent, revisions, trees = _bindings()
    first = IncrementalProofBenchmark(seed=DEFAULT_SEED).run(
        benchmark_worktree_parent_revision=parent,
        source_revisions=revisions,
        source_trees=trees,
    )
    second = IncrementalProofBenchmark(seed=DEFAULT_SEED).run(
        benchmark_worktree_parent_revision=parent,
        source_revisions=revisions,
        source_trees=trees,
    )
    assert bench._canonical_json_bytes(first) == bench._canonical_json_bytes(second)


def test_reject_mismatched_source_bindings() -> None:
    parent, revisions, trees = _bindings()
    with pytest.raises(BenchmarkError):
        IncrementalProofBenchmark(seed=DEFAULT_SEED).run(
            benchmark_worktree_parent_revision="0" * 40,
            source_revisions=revisions,
            source_trees=trees,
        )
    bad = dict(revisions)
    bad.pop("kit")
    with pytest.raises(BenchmarkError):
        IncrementalProofBenchmark(seed=DEFAULT_SEED).run(
            benchmark_worktree_parent_revision=parent,
            source_revisions=bad,
            source_trees=trees,
        )


def test_cli_writes_artifacts_when_git_available(tmp_path: Path) -> None:
    # Resolve real bindings from the workspace; skip if git roots are absent.
    try:
        parent, revisions, trees = bench.resolve_source_bindings(REPO_ROOT)
    except BenchmarkError as exc:
        pytest.skip(f"git source bindings unavailable: {exc}")

    json_out = tmp_path / "out" / "benchmark.json"
    csv_out = tmp_path / "out" / "benchmark.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK_PATH),
            "--seed",
            str(DEFAULT_SEED),
            "--transitions",
            "40",
            "--json-output",
            str(json_out),
            "--csv-output",
            str(csv_out),
            "--repo-root",
            str(REPO_ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["benchmark_worktree_parent_revision"] == parent
    assert payload["source_revisions"] == revisions
    assert payload["source_trees"] == trees
    assert payload["seed"] == DEFAULT_SEED
    assert len(payload["transitions"]) == 40
    assert csv_out.is_file()
    assert payload["execution_context"]["runner_id"] == bench.PROTECTED_RUNNER_ID
    assert payload["execution_context"]["argv"] == bench.protected_benchmark_argv()


def test_protected_argv_matches_validator_contract() -> None:
    argv = bench.protected_benchmark_argv()
    assert argv[1] == "benchmarks/agent_supervisor/incremental_proof_sealer.py"
    assert argv[2:6] == ["--seed", str(DEFAULT_SEED), "--transitions", "40"]
    assert argv[6] == "--json-output"
    assert argv[7].endswith("artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json")
    assert argv[8] == "--csv-output"
    assert argv[9].endswith("artifacts/agent_supervisor/incremental_proof_sealer/benchmark.csv")


def test_expected_unit_sets_cover_add_remove_and_full_reproof() -> None:
    units = IncrementalProofBenchmark(seed=DEFAULT_SEED).expected_unit_sets()
    genesis = units[0]
    assert genesis.seal_status == "sealed_full"
    assert len(genesis.added) == len(genesis.required) > 0
    assert len(genesis.reused) == 0

    added = units[8]
    assert "unit/test/t3" in added.added
    assert added.seal_status == "sealed_incremental"

    deleted = units[9]
    assert "unit/test/t2" in deleted.removed
    assert "unit/test/t2" not in deleted.required

    lock = units[12]
    assert lock.seal_status == "sealed_full"
    assert len(lock.reused) == 0
    assert len(lock.newly_proved) == len(lock.required)
