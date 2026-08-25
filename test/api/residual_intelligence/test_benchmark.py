from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    IDENTITY_FIELDS,
    PARTITIONS,
    REQUIRED_KINDS,
    FrozenBenchmarkCase,
    PairedBenchmarkRunner,
    load_cases,
    load_frozen_benchmark,
    load_manifest,
    validate_frozen_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition,
    ResidualIntelligenceError,
    ResidualTaskFamily,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
CASES = ROOT / "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"


def test_frozen_catalog_covers_every_family_partition_and_case_class() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)

    assert len(cases) == len(ResidualTaskFamily) * len(PARTITIONS) * len(REQUIRED_KINDS)
    assert {
        (case.family, case.partition, case.kind) for case in cases
    } == {
        (family, partition, kind)
        for family in ResidualTaskFamily
        for partition in PARTITIONS
        for kind in REQUIRED_KINDS
    }
    assert manifest.frozen_root == manifest.computed_frozen_root
    assert manifest.training_admission == "training_unavailable"


def test_every_case_binds_identities_and_denies_hidden_tests_to_training() -> None:
    cases = load_cases(CASES)

    for case in cases:
        assert all(getattr(case, field) for field in IDENTITY_FIELDS)
        assert case.hidden_test is (case.partition in {"held_out", "adversarial"})
        if case.kind == "cross_repository":
            assert case.cross_repository_identity
            assert case.cross_repository_identity != case.repository_identity
            assert case.expected_disposition is ExpertDisposition.REJECT_INPUT
        elif case.kind == "unknown_ood":
            assert case.expected_disposition is ExpertDisposition.OUT_OF_DISTRIBUTION
        else:
            assert case.cross_repository_identity == ""


def test_frozen_roots_and_semantic_lineage_fail_closed_on_tampering() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    training_case = next(case for case in cases if case.partition == "training")
    held_out_case = next(case for case in cases if case.partition == "held_out")

    altered_identity = replace(training_case, tokenizer_identity="tokenizer:changed@1")
    with pytest.raises(ResidualIntelligenceError, match="catalog root"):
        validate_frozen_benchmark(manifest, (altered_identity, *cases[1:]))

    mixed_lineage = replace(held_out_case, lineage_group=training_case.lineage_group)
    replaced = tuple(mixed_lineage if item.case_id == held_out_case.case_id else item for item in cases)
    with pytest.raises(ResidualIntelligenceError, match="semantic lineage crosses"):
        validate_frozen_benchmark(manifest, replaced)


def test_paired_runner_preserves_complete_frozen_denominators() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    result = PairedBenchmarkRunner().evaluate(
        manifest,
        cases,
        prior={"accept": 100, "abstain": 284},
        current={"accept": 100, "abstain": 284},
    )

    assert result["candidate_only"] is True
    assert result["frozen_root"] == manifest.frozen_root
    assert result["total_denominator"] == len(cases)
    assert set(result["denominators"]) == {family.value for family in ResidualTaskFamily}
    assert set(result["denominators"].values()) == {len(PARTITIONS) * len(REQUIRED_KINDS)}
    with pytest.raises(ResidualIntelligenceError, match="identical metrics"):
        PairedBenchmarkRunner().evaluate(
            manifest,
            cases,
            prior={"accept": 1},
            current={"abstain": 1},
        )


def test_strict_loader_rejects_unpinned_or_duplicate_json_fields(tmp_path: Path) -> None:
    raw = load_manifest(MANIFEST)
    assert raw["frozen_roots"]["benchmark"]
    duplicate_case = tmp_path / "duplicate.jsonl"
    duplicate_case.write_text(
        '{"family":"TASK_CLASSIFICATION","family":"RISK_CLASSIFICATION"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ResidualIntelligenceError, match="duplicate key"):
        load_cases(duplicate_case)

    sample = next(case for case in load_cases(CASES) if case.kind == "cross_repository")
    with pytest.raises(ResidualIntelligenceError, match="distinct repository"):
        replace(sample, cross_repository_identity=sample.repository_identity)


def test_case_contract_requires_all_partition_appropriate_hidden_flags() -> None:
    sample = next(case for case in load_cases(CASES) if case.partition == "training")
    with pytest.raises(ResidualIntelligenceError, match="hidden_test"):
        replace(sample, hidden_test=True)
