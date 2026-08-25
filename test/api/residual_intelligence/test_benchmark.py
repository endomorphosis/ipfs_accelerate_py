from __future__ import annotations

import copy
import re
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    IDENTITY_FIELDS,
    PARTITIONS,
    REQUIRED_BINDINGS,
    REQUIRED_KINDS,
    PairedBenchmarkRunner,
    ResidualBenchmarkManifest,
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
SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


def test_frozen_catalog_covers_every_family_and_owner_scheduled_partition() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)

    assert len(cases) == len(ResidualTaskFamily) * len(PARTITIONS)
    assert [(case.family, case.partition, case.kind) for case in cases] == [
        (family, partition, kind)
        for family in ResidualTaskFamily
        for partition, kind in zip(PARTITIONS, REQUIRED_KINDS, strict=True)
    ]
    assert manifest.frozen_root == manifest.computed_frozen_root
    assert manifest.case_catalog_root == manifest.benchmark_freeze["case_root"]
    assert manifest.training_admission == "training_unavailable"


def test_cases_are_group_lineage_safe_and_hidden_inputs_are_denied() -> None:
    _, cases = load_frozen_benchmark(MANIFEST, CASES)

    assert len({case.group_id for case in cases}) == len(cases)
    for case in cases:
        assert all(SHA256_ID.fullmatch(getattr(case, field)) for field in IDENTITY_FIELDS)
        assert case.hidden_test is (case.partition in {"held_out", "adversarial"})
        assert case.input_disposition == "payload_unavailable_training_unavailable"
        assert case.expected_outcome is ExpertDisposition.CAPABILITY_UNAVAILABLE

    cross_repository = [case for case in cases if case.kind == "cross_repository"]
    unknown_ood = [case for case in cases if case.kind == "unknown_ood"]
    assert len(cross_repository) == len(ResidualTaskFamily)
    assert {case.partition for case in cross_repository} == {"held_out"}
    assert all(case.hidden_test for case in cross_repository)
    assert len(unknown_ood) == len(ResidualTaskFamily)
    assert {case.partition for case in unknown_ood} == {"adversarial"}
    assert all(case.hidden_test for case in unknown_ood)


def test_manifest_freezes_all_required_identity_bindings_and_roots() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    freeze = manifest.benchmark_freeze

    assert set(freeze["bindings"]) == set(REQUIRED_BINDINGS)
    assert all(SHA256_ID.fullmatch(value) for value in freeze["bindings"].values())
    assert re.fullmatch(r"[0-9a-f]{40}", freeze["source"]["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", freeze["source"]["tree"])
    assert freeze["source"]["commit"] == manifest.source_revision
    assert freeze["case_count"] == len(cases) == 96
    assert freeze["fault_schedule"]["entries"] == [
        {
            "family": case.family.value,
            "partition": case.partition,
            "kind": case.kind,
            "hidden_test": case.hidden_test,
            "group_id": case.group_id,
        }
        for case in cases
    ]


def test_frozen_roots_and_group_lineage_fail_closed_on_tampering() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    altered_input = replace(cases[0], input_identity="sha256:" + "0" * 64)
    with pytest.raises(ResidualIntelligenceError, match="96-case schedule"):
        validate_frozen_benchmark(manifest, (altered_input, *cases[1:]))

    held_out = next(case for case in cases if case.partition == "held_out")
    training = next(case for case in cases if case.partition == "training")
    mixed_lineage = replace(held_out, group_id=training.group_id)
    replaced = tuple(mixed_lineage if case.case_id == held_out.case_id else case for case in cases)
    with pytest.raises(ResidualIntelligenceError, match="96-case schedule"):
        validate_frozen_benchmark(manifest, replaced)

    altered_manifest = copy.deepcopy(manifest.to_dict())
    altered_manifest["benchmark_freeze"]["bindings"]["validation_policy"] = (
        "sha256:" + "f" * 64
    )
    with pytest.raises(ResidualIntelligenceError, match="binding set"):
        validate_frozen_benchmark(ResidualBenchmarkManifest.from_dict(altered_manifest), cases)


def test_paired_baseline_preserves_complete_frozen_all_abstain_denominators() -> None:
    manifest, cases = load_frozen_benchmark(MANIFEST, CASES)
    expected = manifest.benchmark_freeze["paired_baseline"]
    runner = PairedBenchmarkRunner()

    assert runner.evaluate(manifest, cases) == expected
    assert runner.evaluate(
        manifest,
        cases,
        prior=expected["before"],
        current=expected["after"],
    ) == expected
    assert expected["candidate_only"] is True
    assert expected["training_performed"] is False
    assert expected["case_count"] == len(cases)
    assert expected["before"] == expected["after"]
    assert expected["before"]["accept"] == 0
    assert expected["before"]["abstain"] == len(cases)
    assert set(expected["before"]["denominators_by_family"]) == {
        family.value for family in ResidualTaskFamily
    }
    assert set(expected["before"]["denominators_by_family"].values()) == {len(PARTITIONS)}
    with pytest.raises(ResidualIntelligenceError, match="all-abstain"):
        runner.evaluate(manifest, cases, prior={"accept": 1}, current={"accept": 1})


def test_strict_loader_rejects_duplicate_or_legacy_fields_and_bad_hidden_flags(
    tmp_path: Path,
) -> None:
    raw = load_manifest(MANIFEST)
    assert set(raw) == set(ResidualBenchmarkManifest._FIELDS)
    assert raw["benchmark_freeze"]["freeze_id"]

    duplicate_case = tmp_path / "duplicate.jsonl"
    duplicate_case.write_text(
        '{"family":"TASK_CLASSIFICATION","family":"RISK_CLASSIFICATION"}\\n',
        encoding="utf-8",
    )
    with pytest.raises(ResidualIntelligenceError, match="duplicate key"):
        load_cases(duplicate_case)

    legacy_manifest = dict(raw)
    legacy_manifest["frozen_roots"] = {}
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        ResidualBenchmarkManifest.from_dict(legacy_manifest)

    training = next(case for case in load_cases(CASES) if case.partition == "training")
    with pytest.raises(ResidualIntelligenceError, match="hidden_test"):
        replace(training, hidden_test=True)
