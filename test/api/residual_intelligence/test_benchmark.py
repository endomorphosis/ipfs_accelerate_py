from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    FrozenBenchmarkCase,
    PairedBenchmarkRunner,
    ResidualBenchmarkManifest,
    load_manifest,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import ResidualTaskFamily

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks/agent_supervisor/residual_intelligence/manifest.json"


def test_frozen_manifest_covers_all_families_and_hidden_tests_stay_out_of_training() -> None:
    raw = load_manifest(MANIFEST)
    families = tuple(ResidualTaskFamily(item) for item in raw["task_families"])
    manifest = ResidualBenchmarkManifest(
        families=families,
        partitions=tuple(raw["partitions"]),
        frozen_root="bench:fixture",
    )
    cases = []
    for family in ResidualTaskFamily:
        cases.append(
            FrozenBenchmarkCase(
                family=family,
                partition="held_out",
                kind="boundary",
                case_id=f"case:{family.value}:boundary",
            )
        )
        cases.append(
            FrozenBenchmarkCase(
                family=family,
                partition="held_out",
                kind="unknown_ood",
                case_id=f"case:{family.value}:ood",
                hidden_test=True,
            )
        )
    result = PairedBenchmarkRunner().evaluate(
        manifest,
        cases,
        prior={"accept": 1},
        current={"accept": 1},
    )
    assert result["denominators"][ResidualTaskFamily.TASK_CLASSIFICATION.value] >= 1
    assert result["candidate_only"] is True
