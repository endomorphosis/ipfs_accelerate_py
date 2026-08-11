"""DCR-100: deterministic repair precision, safety, and cost benchmark.

Acceptance:
* Zero false completion, unauthorized mutation, mixed-root publication,
  unobserved transition, and model/provider calls.
* Abstention counted separately from false success.
* Cache excluded from primary metrics.
* Rollout authority never granted by measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.evaluation.dcr_benchmark import (
    DEFAULT_BENCHMARK_PATH,
    DETERMINISTIC_REPAIR_BENCHMARK_INTERFACE,
    DCR_BENCHMARK_EVIDENCE,
    DCR_TASK_ID,
    SAFETY_FLOORS,
    DeterministicRepairBenchmark,
    RepairSafetyMetrics,
    run_deterministic_repair_benchmark,
    materialize_benchmark,
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[4], here.parents[3], Path.cwd()):
        if (candidate / "config" / "deterministic_contract_repair_services.json").is_file():
            return candidate
    return here.parents[4]


@pytest.fixture(scope="module")
def dcr_benchmark() -> DeterministicRepairBenchmark:
    return run_deterministic_repair_benchmark(repo_root=_repo_root())


def test_interfaces_and_floors() -> None:
    assert DETERMINISTIC_REPAIR_BENCHMARK_INTERFACE == "DeterministicRepairBenchmark@1"
    assert DeterministicRepairBenchmark.INTERFACE == DETERMINISTIC_REPAIR_BENCHMARK_INTERFACE
    assert RepairSafetyMetrics.INTERFACE == "RepairSafetyMetrics@1"
    assert DCR_TASK_ID == "DCR-100"
    assert DCR_BENCHMARK_EVIDENCE == "dcr/deterministic-repair-benchmark@1"
    assert SAFETY_FLOORS["false_completion"] == 0
    assert SAFETY_FLOORS["model_calls"] == 0
    assert callable(run_deterministic_repair_benchmark)


def test_benchmark_passes_safety_floors(dcr_benchmark: DeterministicRepairBenchmark) -> None:
    assert dcr_benchmark.passed is True
    assert dcr_benchmark.safety.floors_held is True
    assert dcr_benchmark.safety.false_completion == 0
    assert dcr_benchmark.safety.unauthorized_mutation == 0
    assert dcr_benchmark.safety.mixed_root_publication == 0
    assert dcr_benchmark.safety.unobserved_transition == 0
    assert dcr_benchmark.safety.model_calls == 0
    assert dcr_benchmark.safety.provider_calls == 0
    assert dcr_benchmark.safety.mutation_survivors == 0
    assert dcr_benchmark.safety.false_success == 0
    assert dcr_benchmark.runtime_model_calls == 0
    assert dcr_benchmark.provider_calls == 0
    assert dcr_benchmark.zero_llm_enforced is True


def test_abstention_separate_from_false_success(
    dcr_benchmark: DeterministicRepairBenchmark,
) -> None:
    assert dcr_benchmark.safety.abstentions >= 1
    assert dcr_benchmark.detection.abstention >= 1
    assert "abstention_counted_separately" in dcr_benchmark.reason_codes


def test_detection_and_repair_matrices(dcr_benchmark: DeterministicRepairBenchmark) -> None:
    assert dcr_benchmark.detection.false_positive == 0
    assert dcr_benchmark.detection.false_negative == 0
    assert dcr_benchmark.detection.true_positive >= 15
    assert dcr_benchmark.repair.false_positive == 0
    assert dcr_benchmark.repair.true_positive >= 3


def test_cache_excluded_and_resources(dcr_benchmark: DeterministicRepairBenchmark) -> None:
    assert dcr_benchmark.resources.cache_excluded is True
    assert dcr_benchmark.resources.wall_time_ms >= 0
    assert dcr_benchmark.resources.cold_import_ms >= 0
    assert dcr_benchmark.proof_reuse_hits == 0  # cache not credited
    assert dcr_benchmark.proof_reuse_misses >= 1
    assert "cache_excluded_from_primary_metrics" in dcr_benchmark.reason_codes


def test_no_rollout_authority(dcr_benchmark: DeterministicRepairBenchmark) -> None:
    payload = dcr_benchmark.to_dict()
    assert payload["rollout_authority_granted"] is False
    assert "rollout_authority_not_granted" in dcr_benchmark.reason_codes


def test_cold_import_and_declared_versions(
    dcr_benchmark: DeterministicRepairBenchmark,
) -> None:
    assert dcr_benchmark.cold_import.get("ok") is True
    assert dcr_benchmark.cold_import.get("newer_stdlib_leakage") is False
    assert "3.12" in dcr_benchmark.declared_python_versions or any(
        v.startswith("3.") for v in dcr_benchmark.declared_python_versions
    )


def test_materialize_benchmark(tmp_path: Path) -> None:
    dest = tmp_path / "benchmark.json"
    payload = materialize_benchmark(repo_root=_repo_root(), destination=dest)
    assert dest.is_file()
    on_disk = json.loads(dest.read_text(encoding="utf-8"))
    assert on_disk["interface"] == DETERMINISTIC_REPAIR_BENCHMARK_INTERFACE
    assert on_disk["task_id"] == DCR_TASK_ID
    assert on_disk["result"]["passed"] is True
    assert on_disk["rollout_authority_granted"] is False
    assert payload["result"]["passed"] is True


def test_default_path() -> None:
    assert DEFAULT_BENCHMARK_PATH.endswith("benchmark.json")
