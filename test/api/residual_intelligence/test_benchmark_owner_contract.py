from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.benchmark import (
    CASE_SCHEMA,
    MANIFEST_SCHEMA,
    PARTITIONS,
    REQUIRED_KINDS,
    PairedBenchmarkRunner,
    ResidualBenchmarkManifest,
    build_frozen_benchmark,
    build_frozen_benchmark_contract,
    load_frozen_benchmark,
    validate_frozen_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
)
from scripts.run_agent_supervisor_residual_intelligence import (
    _vrif_frozen_benchmark_contract as owner_frozen_benchmark_contract,
)

SOURCE_COMMIT = "a" * 40
SOURCE_TREE = "b" * 40
SPLIT_ROOT = "sha256:" + "8" * 64
BASE_BINDINGS = {
    "repository_states": "sha256:" + "1" * 64,
    "objective_revisions": "sha256:" + "2" * 64,
    "operation_catalog": "sha256:" + "3" * 64,
    "provider_policy": "sha256:" + "4" * 64,
    "tokenizer": "sha256:" + "5" * 64,
    "model_versions": "sha256:" + "6" * 64,
    "validation_policy": "sha256:" + "7" * 64,
}
FAMILIES = tuple(ResidualTaskFamily)


@pytest.fixture
def frozen_bundle():
    return build_frozen_benchmark(
        task_families=FAMILIES,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        split_root=SPLIT_ROOT,
        base_bindings=BASE_BINDINGS,
    )


def test_pure_builder_matches_owner_exact_known_projection() -> None:
    contract = build_frozen_benchmark_contract(
        task_families=FAMILIES,
        source_commit=SOURCE_COMMIT,
        source_tree=SOURCE_TREE,
        split_root=SPLIT_ROOT,
        base_bindings=BASE_BINDINGS,
    )

    # These identities were independently produced by the operator helper for
    # this fixed input. They guard canonicalization as well as the nested shape.
    assert contract["benchmark_freeze"]["freeze_id"] == (
        "sha256:777cbf38ccb7cc35db7950bbcb1bf92c4885d4f1c53f2b2e39369fc59fe2c131"
    )
    assert contract["fault_schedule"]["schedule_id"] == (
        "sha256:b9279b78fa2723a00376abc6bd5f29f5884ca2b5313fffa6d26878bd35d91c03"
    )
    assert contract["benchmark_freeze"]["case_root"] == (
        "sha256:8bf6ca216deaff76e007d6dd807995e4f7a424d404cb02f7e8032aec462836bb"
    )
    assert len(contract["cases"]) == 96
    assert contract["partitions"] == list(PARTITIONS)
    assert contract["case_kinds"] == list(REQUIRED_KINDS)
    assert {
        (item["family"], item["partition"], item["kind"])
        for item in contract["cases"]
    } == {
        (family.value, partition, kind)
        for family in FAMILIES
        for partition, kind in zip(PARTITIONS, REQUIRED_KINDS, strict=True)
    }


def test_pure_builder_equals_the_independent_operator_helper() -> None:
    kwargs = {
        "task_families": FAMILIES,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "split_root": SPLIT_ROOT,
        "base_bindings": BASE_BINDINGS,
    }

    assert build_frozen_benchmark_contract(**kwargs) == owner_frozen_benchmark_contract(
        **kwargs
    )


def test_manifest_and_cases_have_only_owner_exact_fields(frozen_bundle) -> None:
    manifest, cases = frozen_bundle

    assert set(manifest.to_dict()) == {
        "schema",
        "program_identifier",
        "status",
        "owner_task",
        "source_revision",
        "partitions",
        "required_case_kinds",
        "task_families",
        "training_admission",
        "weights_committed",
        "large_corpus_committed",
        "promotion_evidence",
        "benchmark_freeze",
    }
    assert manifest.schema == MANIFEST_SCHEMA
    assert manifest.computed_frozen_root == manifest.frozen_root
    assert all(
        set(item.to_dict())
        == {
            "schema",
            "family",
            "partition",
            "kind",
            "hidden_test",
            "group_id",
            "input_identity",
            "input_disposition",
            "expected_outcome",
            "case_id",
        }
        for item in cases
    )
    assert all(item.schema == CASE_SCHEMA for item in cases)
    assert all(item.expected_outcome.value == "CAPABILITY_UNAVAILABLE" for item in cases)
    assert all(
        item.hidden_test is (item.partition in {"held_out", "adversarial"})
        for item in cases
    )


def test_strict_round_trip_loads_the_owner_exact_bundle(
    frozen_bundle,
    tmp_path: Path,
) -> None:
    manifest, cases = frozen_bundle
    manifest_path = tmp_path / "manifest.json"
    cases_path = tmp_path / "cases.jsonl"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cases_path.write_text(
        "\n".join(json.dumps(item.to_dict(), sort_keys=True) for item in cases) + "\n",
        encoding="utf-8",
    )

    loaded_manifest, loaded_cases = load_frozen_benchmark(manifest_path, cases_path)

    assert loaded_manifest.to_dict() == manifest.to_dict()
    assert loaded_cases == cases


def test_bindings_fault_schedule_and_paired_baseline_fail_closed(frozen_bundle) -> None:
    manifest, cases = frozen_bundle

    binding_tamper = copy.deepcopy(manifest.to_dict())
    binding_tamper["benchmark_freeze"]["bindings"]["provider_policy"] = (
        "sha256:" + "9" * 64
    )
    with pytest.raises(ResidualIntelligenceError, match="binding set"):
        validate_frozen_benchmark(
            ResidualBenchmarkManifest.from_dict(binding_tamper),
            cases,
        )

    schedule_tamper = copy.deepcopy(manifest.to_dict())
    schedule_tamper["benchmark_freeze"]["fault_schedule"]["entries"][0]["kind"] = (
        "negative"
    )
    with pytest.raises(ResidualIntelligenceError, match="fault schedule"):
        validate_frozen_benchmark(
            ResidualBenchmarkManifest.from_dict(schedule_tamper),
            cases,
        )

    paired_tamper = copy.deepcopy(manifest.to_dict())
    paired_tamper["benchmark_freeze"]["paired_baseline"]["evaluated_source"][
        "tree"
    ] = "c" * 40
    with pytest.raises(ResidualIntelligenceError, match="paired benchmark baseline"):
        validate_frozen_benchmark(
            ResidualBenchmarkManifest.from_dict(paired_tamper),
            cases,
        )


def test_case_identity_and_legacy_manifest_fields_fail_closed(frozen_bundle) -> None:
    manifest, cases = frozen_bundle
    altered_cases = (replace(cases[0], input_identity="sha256:" + "0" * 64), *cases[1:])
    with pytest.raises(ResidualIntelligenceError, match="96-case schedule"):
        validate_frozen_benchmark(manifest, altered_cases)

    legacy_manifest = manifest.to_dict()
    legacy_manifest["source_tree"] = SOURCE_TREE
    with pytest.raises(ResidualIntelligenceError, match="unknown fields"):
        ResidualBenchmarkManifest.from_dict(legacy_manifest)


def test_runner_only_accepts_the_frozen_all_abstain_pair(frozen_bundle) -> None:
    manifest, cases = frozen_bundle
    runner = PairedBenchmarkRunner()
    expected = manifest.benchmark_freeze["paired_baseline"]

    assert runner.evaluate(manifest, cases) == expected
    assert runner.evaluate(
        manifest,
        cases,
        prior=expected["before"],
        current=expected["after"],
    ) == expected
    with pytest.raises(ResidualIntelligenceError, match="all-abstain"):
        runner.evaluate(
            manifest,
            cases,
            prior={"accept": 1},
            current={"accept": 1},
        )
