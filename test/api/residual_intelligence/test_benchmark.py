from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    PARTITIONS, REQUIRED_KINDS, PairedBenchmarkRunner, ResidualIntelligenceError,
    load_cases, load_manifest, manifest_from_dict, validate_frozen_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ExpertDisposition, ResidualTaskFamily,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "benchmarks/agent_supervisor/residual_intelligence/manifest.json"
CASES = ROOT / "benchmarks/agent_supervisor/residual_intelligence/cases.jsonl"


def _frozen_benchmark():
    return manifest_from_dict(load_manifest(MANIFEST)), load_cases(CASES)


def test_published_benchmark_has_complete_family_partition_kind_coverage() -> None:
    manifest, cases = _frozen_benchmark()
    validate_frozen_benchmark(manifest, cases)
    assert manifest.families == tuple(ResidualTaskFamily)
    assert manifest.partitions == PARTITIONS
    assert len(cases) == len(ResidualTaskFamily) * len(PARTITIONS) * len(REQUIRED_KINDS)
    assert {(case.family, case.partition, case.kind) for case in cases} == {
        (family, partition, kind)
        for family in ResidualTaskFamily
        for partition in PARTITIONS
        for kind in REQUIRED_KINDS
    }


def test_hidden_and_cross_repository_cases_preserve_denial_and_lineage() -> None:
    manifest, cases = _frozen_benchmark()
    hidden = [case for case in cases if case.hidden_test]
    assert {case.partition for case in hidden} == {"held_out", "adversarial"}
    assert all(case.permitted_use == "evaluation" for case in hidden)
    assert not any("payload" in json.dumps(case.to_dict()).lower() for case in cases)
    assert all(
        case.repository_identity in manifest.bindings.cross_repository_identities
        for case in cases if case.kind == "cross_repository"
    )
    groups: dict[str, set[str]] = {}
    for case in cases:
        groups.setdefault(case.group_id, set()).add(case.partition)
    assert all(len(partitions) == 1 for partitions in groups.values())


def test_tampering_with_case_lineage_or_root_is_rejected() -> None:
    manifest, cases = _frozen_benchmark()
    altered = cases[0].to_dict()
    altered["group_id"] = cases[1].group_id
    altered["case_commitment"] = "sha256:" + "0" * 64
    with pytest.raises(ResidualIntelligenceError):
        type(cases[0]).from_dict(altered)
    bad = manifest.__class__(
        families=manifest.families, partitions=manifest.partitions, bindings=manifest.bindings,
        case_root=manifest.case_root, frozen_root="sha256:" + "0" * 64,
        paired_baseline_identity=manifest.paired_baseline_identity,
        hidden_test_commitment=manifest.hidden_test_commitment,
    )
    with pytest.raises(ResidualIntelligenceError, match="frozen root"):
        validate_frozen_benchmark(bad, cases)


def test_paired_runner_requires_exact_case_denominators_and_scores_both_candidates() -> None:
    manifest, cases = _frozen_benchmark()
    prior = {case.case_id: case.expected_outcome.value for case in cases}
    current = dict(prior)
    first = cases[0]
    current[first.case_id] = ExpertDisposition.ABSTAIN.value
    result = PairedBenchmarkRunner().evaluate(manifest, cases, prior=prior, current=current)
    assert result["denominators"]["all_cases"] == len(cases)
    assert result["prior"]["scored_cases"] == len(cases)
    assert result["current"]["correct_by_family"][first.family.value] == (
        result["prior"]["correct_by_family"][first.family.value] - 1
    )
    with pytest.raises(ResidualIntelligenceError, match="every exact frozen case"):
        PairedBenchmarkRunner().evaluate(manifest, cases, prior=prior, current={})
