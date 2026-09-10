"""ASEH-013 paired hermetic harness, sealed fixtures, and statistics."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    EQUAL_CONTROL_FIELDS,
    HERMETIC_MINIMUM,
    PAIRED_ARMS,
    STATISTIC_FIELDS,
    admit_paired_benchmark_manifest,
)


ROOT = Path(__file__).resolve().parents[4]
HARNESS_PATH = (
    ROOT
    / "benchmarks"
    / "agent_supervisor"
    / "efficiency_state_hardening"
    / "paired_harness.py"
)
HERMETIC_MANIFEST_PATH = HARNESS_PATH.with_name("hermetic_manifest.json")
HERMETIC_VECTORS_PATH = HARNESS_PATH.with_name("hermetic_vectors.jsonl")
CAMPAIGN_MANIFEST_PATH = HARNESS_PATH.with_name("manifest.json")


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aseh_013_paired_harness",
        HARNESS_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()
HARNESS.write_sealed_artifacts()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_hermetic_manifest_seals_at_least_sixty_unique_bounded_fixtures() -> None:
    manifest = _load_json(HERMETIC_MANIFEST_PATH)
    recipes = HARNESS.load_vectors(HERMETIC_VECTORS_PATH)
    identities = [HARNESS.recipe_identity(item) for item in recipes]
    fixture_ids = [item["fixture_id"] for item in recipes]
    listed_ids = [item["fixture_id"] for item in manifest["fixtures"]]
    listed_identities = [item["identity"] for item in manifest["fixtures"]]

    assert manifest["schema"] == HARNESS.HERMETIC_MANIFEST_SCHEMA
    assert manifest["minimum"] == HERMETIC_MINIMUM
    assert manifest["count"] >= HERMETIC_MINIMUM
    assert manifest["count"] == len(recipes) == len(manifest["fixtures"])
    assert len(set(fixture_ids)) == len(fixture_ids) >= HERMETIC_MINIMUM
    assert len(set(identities)) == len(identities) >= HERMETIC_MINIMUM
    assert listed_ids == fixture_ids
    assert listed_identities == identities
    assert manifest["live"] is False
    assert manifest["authority"] is False
    assert manifest["hermetic_sufficient_for_production_promotion"] is False
    assert manifest["population_kind"] == "hermetic_development"
    assert manifest["status"] == "sealed"
    assert set(manifest["task_classes"]) == set(HARNESS.TASK_CLASSES)
    assert set(manifest["arms"]) == set(PAIRED_ARMS)
    assert tuple(manifest["equal_control_fields"]) == EQUAL_CONTROL_FIELDS
    for recipe, listed in zip(recipes, manifest["fixtures"], strict=True):
        admitted = HARNESS.admit_recipe(recipe)
        assert admitted["live"] is False
        assert admitted["token_budget"] <= HARNESS.MAX_TOKENS
        assert admitted["retry_budget"] <= HARNESS.MAX_RETRIES
        assert admitted["duration_us"] <= HARNESS.MAX_DURATION_US
        assert admitted["bytes_bound"] <= HARNESS.MAX_BYTES
        assert admitted["quality_bps"] <= HARNESS.MAX_QUALITY_BPS
        assert listed["bounded"] is True
        assert listed["live"] is False
        encoded = HARNESS.compact_json(admitted)
        assert "receipt_cid" not in encoded
        assert "model_use" not in encoded
        assert len(encoded.encode("utf-8")) <= HARNESS.MAX_SERIALIZED_RECIPE_BYTES


def test_sealed_vectors_and_manifests_match_the_recipe_generator() -> None:
    verified = HARNESS.verify_sealed_artifacts()
    assert verified["fixture_count"] >= HERMETIC_MINIMUM
    assert verified["unique_identities"] == verified["fixture_count"]
    generated = HARNESS.generate_fixture_recipes()
    loaded = HARNESS.load_vectors(HERMETIC_VECTORS_PATH)
    assert tuple(item["fixture_id"] for item in loaded) == tuple(
        item["fixture_id"] for item in generated
    )
    assert [HARNESS.recipe_identity(item) for item in loaded] == [
        HARNESS.recipe_identity(item) for item in generated
    ]


def test_pairing_runs_three_arms_with_identical_controls() -> None:
    recipes = HARNESS.generate_fixture_recipes()
    vectors_identity = HARNESS.content_identity({"recipes": list(recipes)})
    controls = HARNESS.shared_equal_controls(task_inputs=vectors_identity)
    campaign = HARNESS.run_paired_campaign(recipes, controls=controls)
    assert campaign["arms"] == list(PAIRED_ARMS)
    assert campaign["fixture_count"] == len(recipes)
    assert campaign["live"] is False
    assert campaign["authority"] is False
    assert campaign["hermetic_sufficient_for_production_promotion"] is False
    assert campaign["promotion_without_paired_campaign"] is False
    assert campaign["controls"] == controls
    assert set(campaign["controls"]) == set(EQUAL_CONTROL_FIELDS)
    for field in EQUAL_CONTROL_FIELDS:
        values = {
            item["controls"][field] for item in campaign["paired_fixtures"]
        }
        assert values == {controls[field]}
    first = campaign["paired_fixtures"][0]
    assert set(first["observations"]) == set(PAIRED_ARMS)
    for arm_id, observation in first["observations"].items():
        assert observation["arm_id"] == arm_id
        assert observation["live"] is False
        assert observation["simulated"] is True
        assert observation["truth_state"] == "simulated"
        assert observation["audit_overhead_microusd"] > 0


def test_pairing_rejects_unequal_controls() -> None:
    recipes = HARNESS.generate_fixture_recipes()
    vectors_identity = HARNESS.content_identity({"recipes": list(recipes)})
    shared = HARNESS.shared_equal_controls(task_inputs=vectors_identity)
    arm_controls = HARNESS.controls_for_arms(shared)
    arm_controls["candidate_optimized_supervisor"] = dict(
        arm_controls["candidate_optimized_supervisor"]
    )
    arm_controls["candidate_optimized_supervisor"]["maximum_retries"] = (
        shared["maximum_retries"] + 1
    )
    with pytest.raises(HARNESS.UnequalControlError, match="unequal controls"):
        HARNESS.run_paired_campaign(recipes, arm_controls=arm_controls)

    missing = dict(arm_controls)
    missing.pop("direct_minimal_orchestration_baseline")
    with pytest.raises(HARNESS.UnequalControlError, match="three closed arms"):
        HARNESS.require_equal_controls(missing)

    mutated = HARNESS.controls_for_arms(shared)
    mutated["sealed_current_supervisor_baseline"] = dict(
        mutated["sealed_current_supervisor_baseline"]
    )
    mutated["sealed_current_supervisor_baseline"]["task_inputs"] = (
        HARNESS.content_identity({"other": "inputs"})
    )
    with pytest.raises(HARNESS.UnequalControlError, match="task_inputs"):
        HARNESS.require_equal_controls(mutated)


def test_pairing_rejects_missing_paired_inputs_and_unseeded_bootstrap() -> None:
    recipe = HARNESS.generate_fixture_recipes()[0]
    controls = HARNESS.shared_equal_controls(
        task_inputs=HARNESS.recipe_identity(recipe)
    )
    observations = {
        "direct_minimal_orchestration_baseline": HARNESS.simulate_arm(
            recipe, arm_id="direct_minimal_orchestration_baseline"
        ),
        "candidate_optimized_supervisor": HARNESS.simulate_arm(
            recipe, arm_id="candidate_optimized_supervisor"
        ),
    }
    with pytest.raises(HARNESS.PairedHarnessError, match="missing paired inputs"):
        HARNESS.pair_fixture_observations(observations, controls=controls)
    with pytest.raises(HARNESS.PairedHarnessError, match="explicit seed"):
        HARNESS.run_paired_campaign(
            HARNESS.generate_fixture_recipes(),
            bootstrap_seed=None,
        )
    with pytest.raises(HARNESS.PairedHarnessError, match="empty population"):
        HARNESS.compute_paired_statistics(
            [],
            baseline_arm="sealed_current_supervisor_baseline",
            candidate_arm="candidate_optimized_supervisor",
        )


def test_pairing_computes_required_statistics_and_audit_overhead() -> None:
    campaign = HARNESS.run_paired_campaign()
    stats = campaign["statistics"]
    for field in STATISTIC_FIELDS:
        assert field in stats
        assert stats[field] is not None
    assert stats["median_difference"]["cost_microusd"] != 0
    assert stats["mean_difference"]["cost_microusd"] != 0
    assert stats["per_task_ratios"]["median_ratio_millionths"] > 0
    assert stats["bootstrap_confidence_intervals"]["seed"] == HARNESS.BOOTSTRAP_SEED
    assert stats["bootstrap_confidence_intervals"]["resamples"] == HARNESS.BOOTSTRAP_RESAMPLES
    assert "lower" in stats["bootstrap_confidence_intervals"]
    assert "upper" in stats["bootstrap_confidence_intervals"]
    assert set(stats["distribution_by_task_class"]) == set(HARNESS.TASK_CLASSES)
    assert stats["outlier_analysis"]["count"] >= 1
    assert stats["outlier_analysis"]["fixture_ids"]
    assert stats["quality_adjusted_cost"]["includes_audit_overhead"] is True
    assert stats["quality_adjusted_cost"]["median_candidate_microusd"] > 0
    assert 0 < stats["accepted_patch_rate"]["value"] < HARNESS.MILLIONTHS
    assert stats["time_to_terminal_outcome"]["median_us"] > 0
    assert campaign["audit_overhead"]["total_microusd"] > 0
    assert campaign["audit_overhead"]["includes_audit_in_quality_adjusted_cost"] is True

    sample = campaign["paired_fixtures"][0]["observations"][
        "candidate_optimized_supervisor"
    ]
    expected_qac = (
        sample["net_cost_microusd"] * HARNESS.BASIS_POINTS
    ) // sample["quality_bps"]
    assert sample["net_cost_microusd"] == (
        sample["gross_cost_microusd"] + sample["audit_overhead_microusd"]
    )
    assert sample["quality_adjusted_cost_microusd"] == expected_qac
    assert sample["quality_adjusted_cost_microusd"] > (
        sample["gross_cost_microusd"] * HARNESS.BASIS_POINTS
    ) // sample["quality_bps"]


def test_bootstrap_intervals_are_seeded_and_deterministic() -> None:
    values = [12, 18, 21, 40, 44, 90, 91]
    first = HARNESS.bootstrap_interval(values, seed=13, resamples=63)
    second = HARNESS.bootstrap_interval(values, seed=13, resamples=63)
    third = HARNESS.bootstrap_interval(values, seed=17, resamples=63)
    assert first == second
    assert first != third
    assert first["lower"] <= first["upper"]
    with pytest.raises(HARNESS.PairedHarnessError, match="explicit seed"):
        HARNESS.bootstrap_interval(values, seed=None)  # type: ignore[arg-type]


def test_campaign_manifest_admits_and_cannot_promote_from_hermetic_evidence() -> None:
    payload = _load_json(CAMPAIGN_MANIFEST_PATH)
    admitted = admit_paired_benchmark_manifest(payload)
    encoded = admitted.to_dict()
    assert encoded["schema"] == HARNESS.CAMPAIGN_MANIFEST_SCHEMA
    assert encoded["promotion_without_paired_campaign"] is False
    assert encoded["hermetic_sufficient_for_production_promotion"] is False
    assert encoded["populations"]["hermetic_development"]["status"] == "sealed"
    assert encoded["populations"]["hermetic_development"]["count"]["value"] >= HERMETIC_MINIMUM
    assert encoded["populations"]["historical_exact_tree_replay"]["status"] == "unavailable"
    assert encoded["populations"]["new_live_shadow_canary"]["status"] == "unavailable"
    assert set(encoded["arms"]) == set(PAIRED_ARMS)
    assert encoded["arms"]["candidate_optimized_supervisor"]["shadow_before_mutation"] is True
    assert encoded["equal_controls"]["repository_revision"] == HARNESS.REPOSITORY_TREE
    for field in STATISTIC_FIELDS:
        node = encoded["statistics"][field]
        assert node["truth_state"] == "measured"
        assert "value" in node
        assert node["sensor_id"] == HARNESS.SENSOR_ID
    replayed = admit_paired_benchmark_manifest(
        CAMPAIGN_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert replayed.manifest_cid == admitted.manifest_cid


def test_hermetic_observations_cannot_be_represented_as_live() -> None:
    recipe = dict(HARNESS.generate_fixture_recipes()[0])
    recipe["live"] = True
    with pytest.raises(HARNESS.PairedHarnessError, match="cannot be live"):
        HARNESS.admit_recipe(recipe)
    live_recipe = HARNESS.generate_fixture_recipes()[0]
    observation = HARNESS.simulate_arm(
        live_recipe, arm_id="candidate_optimized_supervisor"
    )
    observation["live"] = True
    controls = HARNESS.shared_equal_controls(
        task_inputs=HARNESS.recipe_identity(live_recipe)
    )
    observations = {
        arm_id: HARNESS.simulate_arm(live_recipe, arm_id=arm_id)
        for arm_id in PAIRED_ARMS
    }
    observations["candidate_optimized_supervisor"] = observation
    with pytest.raises(HARNESS.PairedHarnessError, match="cannot mark an arm live"):
        HARNESS.pair_fixture_observations(observations, controls=controls)


def test_class_distributions_and_outliers_cover_the_sealed_corpus() -> None:
    campaign = HARNESS.run_paired_campaign()
    distribution = campaign["statistics"]["distribution_by_task_class"]
    assert sum(distribution.values()) == campaign["fixture_count"]
    for task_class in HARNESS.TASK_CLASSES:
        assert distribution[task_class] == len(HARNESS.OUTCOMES)
    outlier_ids = set(campaign["statistics"]["outlier_analysis"]["fixture_ids"])
    recipes = {item["fixture_id"]: item for item in campaign["recipes"]}
    assert outlier_ids
    assert any(recipes[item]["token_budget"] >= 40_000 for item in outlier_ids)
