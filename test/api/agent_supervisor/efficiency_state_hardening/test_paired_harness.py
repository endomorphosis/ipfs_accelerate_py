"""ASEH-013: sealed hermetic fixtures and fail-closed paired harness."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    EQUAL_CONTROL_FIELDS,
    PAIRED_ARMS,
    STATISTIC_FIELDS,
    admit_paired_benchmark_manifest,
    canonical_bytes,
    content_identity,
)


ROOT = Path(__file__).resolve().parents[4]
HARNESS_PATH = (
    ROOT / "benchmarks/agent_supervisor/efficiency_state_hardening/paired_harness.py"
)
VECTORS_PATH = (
    ROOT / "benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl"
)
HERMETIC_MANIFEST_PATH = (
    ROOT / "benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_manifest.json"
)
PAIRED_MANIFEST_PATH = (
    ROOT / "benchmarks/agent_supervisor/efficiency_state_hardening/manifest.json"
)


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aseh_efficiency_state_hardening_paired_harness",
        HARNESS_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


@pytest.fixture(scope="session", autouse=True)
def seal_hermetic_artifacts() -> None:
    HARNESS.materialize_sealed_artifacts()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _recipes() -> list[dict[str, Any]]:
    return HARNESS.load_hermetic_recipes()


def test_hermetic_manifest_contains_at_least_60_unique_bounded_fixtures() -> None:
    recipes = _recipes()
    manifest = _load_json(HERMETIC_MANIFEST_PATH)
    fixture_ids = [str(item["fixture_id"]) for item in recipes]
    assert len(recipes) >= 60
    assert len(set(fixture_ids)) == len(fixture_ids)
    assert manifest["count"] == len(recipes)
    assert manifest["minimum"] == 60
    assert manifest["status"] == "sealed"
    assert manifest["bounded"] is True
    assert manifest["live"] is False
    assert manifest["population_kind"] == "hermetic_development"
    assert manifest["hermetic_sufficient_for_production_promotion"] is False
    assert manifest["fixture_ids"] == fixture_ids
    assert set(manifest["task_classes"]) == set(HARNESS.TASK_CLASSES)
    for recipe in recipes:
        assert recipe["bounded"] is True
        assert recipe["live"] is False
        assert type(recipe["seed"]) is int
        assert recipe["task_class"] in HARNESS.TASK_CLASSES


def test_vectors_are_compact_recipes_and_match_the_generator() -> None:
    raw = VECTORS_PATH.read_text(encoding="utf-8")
    generated = HARNESS.serialize_vectors(HARNESS.generate_hermetic_recipes())
    assert raw == generated
    for line in raw.splitlines():
        payload = json.loads(line)
        assert isinstance(payload, dict)
        assert "arms" not in payload
        assert "receipt_cid" not in payload
        assert "model_use" not in payload
        assert set(payload) == {
            "bounded",
            "fixture_id",
            "live",
            "outcome",
            "patch_disposition",
            "seed",
            "task_class",
        }


def test_three_arms_share_identical_controls() -> None:
    result = HARNESS.run_hermetic_pairing(_recipes())
    controls = result["controls"]
    assert tuple(controls) == EQUAL_CONTROL_FIELDS
    for pair in result["pairs"]:
        assert tuple(pair) == PAIRED_ARMS
        shared = HARNESS.assert_equal_controls(tuple(pair[arm] for arm in PAIRED_ARMS))
        assert shared == controls
        for arm in PAIRED_ARMS:
            assert pair[arm]["controls_identity"] == result["controls_identity"]
            assert pair[arm]["live"] is False
            assert type(pair[arm]["audit_and_verification_overhead"]) is int


def test_pairing_rejects_unequal_controls() -> None:
    result = HARNESS.run_hermetic_pairing(_recipes()[:3])
    replacements = {
        "repository_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "objective": "ASEH-G080",
        "task_inputs": content_identity({"other": "inputs"}),
        "acceptance_tests": content_identity({"other": "tests"}),
        "available_providers_and_models": content_identity({"other": "providers"}),
        "price_accounting": content_identity({"other": "prices"}),
        "resource_limits": content_identity({"other": "limits"}),
        "maximum_retries": 1,
        "human_intervention_policy": content_identity({"other": "human"}),
    }
    for field, value in replacements.items():
        pair = copy.deepcopy(result["pairs"][0])
        mutated = dict(pair[HARNESS.CANDIDATE_ARM])
        controls = dict(mutated["controls"])
        controls[field] = value
        mutated["controls"] = controls
        mutated["controls_identity"] = HARNESS.controls_identity(controls)
        pair[HARNESS.CANDIDATE_ARM] = mutated
        with pytest.raises(HARNESS.UnequalControlsError, match="unequal controls"):
            HARNESS.assert_equal_controls(tuple(pair[arm] for arm in PAIRED_ARMS))
        with pytest.raises(HARNESS.UnequalControlsError, match="unequal controls"):
            HARNESS.compute_paired_statistics([pair])


def test_pairing_rejects_missing_paired_inputs() -> None:
    result = HARNESS.run_hermetic_pairing(_recipes()[:2])
    incomplete = dict(result["pairs"][0])
    del incomplete[HARNESS.DIRECT_ARM]
    with pytest.raises(HARNESS.PairedHarnessError, match="missing paired inputs"):
        HARNESS.compute_paired_statistics([incomplete])


def test_pairing_rejects_unseeded_nondeterminism() -> None:
    recipes = copy.deepcopy(_recipes()[:1])
    del recipes[0]["seed"]
    with pytest.raises(HARNESS.PairedHarnessError, match="unseeded"):
        HARNESS.run_hermetic_pairing(recipes)


def test_pairing_rejects_fabricated_live_labels() -> None:
    recipes = copy.deepcopy(_recipes()[:1])
    recipes[0]["live"] = True
    with pytest.raises(HARNESS.PairedHarnessError, match="live"):
        HARNESS.run_hermetic_pairing(recipes)
    result = HARNESS.run_hermetic_pairing(_recipes()[:1])
    pair = copy.deepcopy(result["pairs"][0])
    live_arm = dict(pair[HARNESS.SEALED_ARM])
    live_arm["live"] = True
    pair[HARNESS.SEALED_ARM] = live_arm
    with pytest.raises(HARNESS.PairedHarnessError, match="live"):
        HARNESS.assert_equal_controls(tuple(pair[arm] for arm in PAIRED_ARMS))


def test_pairing_rejects_incomplete_audit_cost() -> None:
    result = HARNESS.run_hermetic_pairing(_recipes()[:1])
    pair = copy.deepcopy(result["pairs"][0])
    broken = dict(pair[HARNESS.CANDIDATE_ARM])
    broken["audit_and_verification_overhead"] = None
    pair[HARNESS.CANDIDATE_ARM] = broken
    with pytest.raises(HARNESS.PairedHarnessError, match="audit"):
        HARNESS.assert_equal_controls(tuple(pair[arm] for arm in PAIRED_ARMS))


def test_pairing_computes_required_statistics_and_audit_overhead() -> None:
    result = HARNESS.run_hermetic_pairing()
    stats = result["statistics"]
    assert result["pair_count"] >= 60
    assert stats["pair_count"] == result["pair_count"]
    for field in (
        "median_difference",
        "mean_difference",
        "per_task_ratios",
        "bootstrap_confidence_intervals",
        "distribution_by_task_class",
        "outlier_analysis",
        "quality_adjusted_cost",
        "accepted_patch_rate",
        "time_to_terminal_outcome",
    ):
        assert field in stats
    assert tuple(result["manifest"]["statistics"]) == STATISTIC_FIELDS
    assert type(stats["median_difference"]) is int
    assert type(stats["mean_difference"]) is int
    assert stats["per_task_ratios"]
    assert all(type(item) is int for item in stats["per_task_ratios"])
    difference_ci = stats["bootstrap_confidence_intervals"]["difference"]
    assert difference_ci["low"] <= difference_ci["high"]
    assert difference_ci["width"] == difference_ci["high"] - difference_ci["low"]
    assert set(stats["distribution_by_task_class"]) == set(HARNESS.TASK_CLASSES)
    assert stats["outlier_analysis"]["count"] >= 1
    assert type(stats["quality_adjusted_cost"]["candidate"]) is int
    assert type(stats["accepted_patch_rate"]["candidate"]) is int
    assert type(stats["time_to_terminal_outcome"]["candidate"]) is int
    for pair in result["pairs"]:
        candidate = pair[HARNESS.CANDIDATE_ARM]
        audit = candidate["audit_and_verification_overhead"]
        charge = candidate["token_based_estimated_charge"]
        quality = candidate["accepted_patch_quality"]
        qac = candidate["quality_adjusted_cost"]
        if quality > 0:
            assert qac == (charge + audit) * HARNESS.MILLIONTHS // quality
        else:
            assert qac is None


def test_bootstrap_is_seeded_and_deterministic() -> None:
    values = [12, 18, 9, 40, 22, 17, 11, 25]
    first = HARNESS.bootstrap_interval(values, seed=20260823, domain="qac-difference")
    second = HARNESS.bootstrap_interval(values, seed=20260823, domain="qac-difference")
    third = HARNESS.bootstrap_interval(values, seed=20260823, domain="other")
    assert first == second
    assert first != third


def test_sealed_manifests_round_trip_and_remain_non_promotional() -> None:
    hermetic = _load_json(HERMETIC_MANIFEST_PATH)
    paired = _load_json(PAIRED_MANIFEST_PATH)
    body = {key: value for key, value in hermetic.items() if key != "identity"}
    assert hermetic["identity"] == content_identity(body)
    admitted = admit_paired_benchmark_manifest(paired)
    assert admitted.canonical_bytes() == canonical_bytes(paired)
    assert paired["hermetic_sufficient_for_production_promotion"] is False
    assert paired["promotion_without_paired_campaign"] is False
    assert paired["populations"]["hermetic_development"]["status"] == "sealed"
    assert paired["populations"]["hermetic_development"]["count"]["value"] >= 60
    assert paired["populations"]["historical_exact_tree_replay"]["status"] == "unavailable"
    assert paired["populations"]["new_live_shadow_canary"]["status"] == "unavailable"
    for field in STATISTIC_FIELDS:
        node = paired["statistics"][field]
        assert node["truth_state"] in {"measured", "estimated"}
        assert "value" in node
        assert type(node["value"]) is int
        assert node["value"] >= 0
    result = HARNESS.load_sealed_pairing()
    assert result["manifest"] == paired
    assert result["live"] is False


def test_unavailable_statistics_never_encode_numeric_zero() -> None:
    payload = copy.deepcopy(_load_json(PAIRED_MANIFEST_PATH))
    payload["statistics"]["median_difference"] = {
        "truth_state": "unavailable",
        "reason_code": "not_yet_measured",
        "value": 0,
    }
    with pytest.raises(Exception):
        admit_paired_benchmark_manifest(payload)


def test_cli_hermetic_cohort_emits_honest_non_live_result(tmp_path: Path) -> None:
    output = tmp_path / "hermetic.json"
    qualification = tmp_path / "qualification.json"
    assert HARNESS.main(
        [
            "--cohort",
            "hermetic",
            "--minimum-tasks",
            "60",
            "--output",
            str(output),
            "--qualification-output",
            str(qualification),
            "--allow-honest-nonpromotion",
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["live"] is False
    assert payload["pair_count"] >= 60
    assert payload["hermetic_sufficient_for_production_promotion"] is False
    assert "median_difference" in payload["statistics"]
    historical = HARNESS.run_cohort(
        "historical",
        minimum_tasks=20,
        allow_honest_nonpromotion=True,
    )
    assert historical["status"] == "unavailable"
    assert historical["truth_state"] == "unavailable"
    assert historical["live"] is False
