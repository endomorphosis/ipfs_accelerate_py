"""ASEH-035 ContextPack paired current-tree benchmark."""

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
    PAIRED_ARMS,
    collect_unavailable_fields,
)


ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_PATH = (
    ROOT
    / "benchmarks"
    / "agent_supervisor"
    / "efficiency_state_hardening"
    / "context_pack_benchmark.py"
)
MANIFEST_PATH = BENCHMARK_PATH.with_name("context_pack_manifest.json")


def _load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "aseh_035_context_pack_benchmark",
        BENCHMARK_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = _load_benchmark()


@pytest.fixture(scope="module")
def campaign() -> dict[str, Any]:
    result = BENCHMARK.run_paired_campaign()
    BENCHMARK.write_sealed_artifacts(campaign=result)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_required_metrics_present_on_paired_current_tree_measures(
    campaign: dict[str, Any],
) -> None:
    assert campaign["required_metrics"] == list(BENCHMARK.REQUIRED_METRICS)
    assert campaign["fixture_count"] == len(campaign["paired_fixtures"])
    assert campaign["paired_fixtures"], "paired current-tree measures must be per-fixture"
    for item in campaign["paired_fixtures"]:
        assert set(item["observations"]) == set(PAIRED_ARMS)
        for arm_id, observation in item["observations"].items():
            assert observation["arm_id"] == arm_id
            assert observation["truth_state"] == "measured"
            metrics = observation["metrics"]
            for name in BENCHMARK.REQUIRED_METRICS:
                assert name in metrics, f"{item['fixture_id']}.{arm_id} missing {name}"
                node = metrics[name]
                assert node["truth_state"] in {"measured", "estimated", "unavailable"}
    aggregates = campaign["aggregates"]
    for name in BENCHMARK.REQUIRED_METRICS:
        assert name in aggregates["metrics"]
        assert set(aggregates["metrics"][name]) == set(PAIRED_ARMS)


def test_unavailable_data_is_retained_and_never_encoded_as_zero(
    campaign: dict[str, Any],
) -> None:
    listed = tuple(campaign["explicit_unavailable_fields"])
    collected = collect_unavailable_fields(campaign)
    assert listed == collected
    assert listed, "historical/live/provider-reported fields must stay unavailable"
    assert any("provider_reported_charge" in path for path in listed)
    assert any("historical_exact_tree_replay" in path for path in listed)
    assert any("new_live_shadow_canary" in path for path in listed)
    for path, node, state in _iter_unavailable(campaign):
        assert state == "unavailable"
        assert "reason_code" in node
        for forbidden in ("value", "count", "unit", "sensor_id", "estimator_id"):
            assert forbidden not in node, f"{path} encoded {forbidden} on unavailable"
        assert node.get("value", "unavailable") != 0
        assert node.get("count", "unavailable") != 0
    manifest = BENCHMARK.build_context_pack_manifest(campaign)
    assert tuple(manifest["explicit_unavailable_fields"]) == collect_unavailable_fields(
        manifest
    )
    assert manifest["populations"]["historical_exact_tree_replay"]["status"] == "unavailable"
    assert manifest["populations"]["new_live_shadow_canary"]["status"] == "unavailable"
    assert manifest["populations"]["historical_exact_tree_replay"]["count"]["truth_state"] == (
        "unavailable"
    )


def _iter_unavailable(value: Any, path: str = ""):
    if isinstance(value, dict):
        if value.get("truth_state") == "unavailable":
            yield path, value, "unavailable"
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_unavailable(child, child_path)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from _iter_unavailable(child, child_path)


def test_seeded_critical_omissions_are_always_rejected(campaign: dict[str, Any]) -> None:
    seeded = [
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "critical_omission"
    ]
    assert seeded, "critical omissions must be seeded"
    for item in seeded:
        for observation in item["observations"].values():
            assert observation["critical_omission_seeded"] is True
            assert observation["critical_omission_accepted"] is False
            assert observation["disposition"] == "rejected"
            assert observation["reused"] is False
            node = observation["metrics"]["critical_omission_detection"]
            assert node["truth_state"] == "measured"
            assert node["value"] == 1
    negative = campaign["negative_cases"]["seeded_critical_omissions"]
    assert negative["accepted"] == 0
    assert negative["rejected"] == negative["count"] == negative["detected"]
    assert campaign["aggregates"]["seeded_critical_omissions_accepted"] == 0


def test_seeded_stale_packs_are_always_rejected(campaign: dict[str, Any]) -> None:
    seeded = [
        item for item in campaign["paired_fixtures"] if item["scenario"] == "stale_pack"
    ]
    assert len(seeded) == len(BENCHMARK.STALE_FIELDS)
    fields = {
        recipe["stale_field"]
        for recipe in campaign["recipes"]
        if recipe["scenario"] == "stale_pack"
    }
    assert fields == set(BENCHMARK.STALE_FIELDS)
    for item in seeded:
        for observation in item["observations"].values():
            assert observation["stale_seeded"] is True
            assert observation["stale_admitted"] is False
            assert observation["disposition"] == "rejected"
            assert observation["reused"] is False
            node = observation["metrics"]["stale_rejection"]
            assert node["truth_state"] == "measured"
            assert node["value"] == 1
    negative = campaign["negative_cases"]["seeded_stale_packs"]
    assert negative["admitted"] == 0
    assert negative["rejected"] == negative["count"]
    assert campaign["aggregates"]["seeded_stale_packs_admitted"] == 0


def test_fixture_as_live_is_rejected_and_campaign_is_not_live(
    campaign: dict[str, Any],
) -> None:
    assert campaign["live"] is False
    assert campaign["simulated_as_live"] is False
    assert campaign["authority"] is False
    masquerade = [
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "fixture_as_live"
    ]
    assert masquerade
    for item in masquerade:
        for observation in item["observations"].values():
            assert observation["live"] is False
            assert observation["simulated_as_live"] is False
            assert observation["disposition"] == "rejected"
            assert observation["reused"] is False
    recipe = dict(BENCHMARK.generate_fixture_recipes()[0])
    recipe["live"] = True
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="cannot be live"):
        BENCHMARK.admit_recipe(recipe)


def test_audit_overhead_is_measured_on_every_observation(campaign: dict[str, Any]) -> None:
    total = 0
    for item in campaign["paired_fixtures"]:
        for observation in item["observations"].values():
            node = observation["metrics"]["audit_overhead"]
            assert node["truth_state"] == "measured"
            assert node["unit"] == "microusd"
            assert node["sensor_id"] == BENCHMARK.SENSOR_ID
            assert node["value"] > 0
            total += node["value"]
    assert campaign["aggregates"]["audit_overhead_total_microusd"] == total
    assert total > 0


def test_results_are_not_aggregate_only(campaign: dict[str, Any]) -> None:
    assert len(campaign["paired_fixtures"]) >= 12
    scenarios = {item["scenario"] for item in campaign["paired_fixtures"]}
    assert scenarios == set(BENCHMARK.SCENARIO_KINDS)
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="aggregate-only"):
        BENCHMARK.admit_campaign_result(
            {
                **campaign,
                "paired_fixtures": [],
            }
        )


def test_hermetic_evidence_cannot_promote_or_grant_live_reuse(
    campaign: dict[str, Any],
) -> None:
    manifest = BENCHMARK.build_context_pack_manifest(campaign)
    assert campaign["hermetic_sufficient_for_production_promotion"] is False
    assert campaign["hermetic_sufficient_for_live_reuse"] is False
    assert campaign["promotion_without_paired_campaign"] is False
    assert manifest["hermetic_sufficient_for_production_promotion"] is False
    assert manifest["hermetic_sufficient_for_live_reuse"] is False
    assert manifest["live"] is False
    assert manifest["authority"] is False
    mutated = dict(campaign)
    mutated["hermetic_sufficient_for_production_promotion"] = True
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="promotion"):
        BENCHMARK.admit_campaign_result(mutated)
    mutated = dict(campaign)
    mutated["hermetic_sufficient_for_live_reuse"] = True
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="live reuse"):
        BENCHMARK.admit_campaign_result(mutated)


def test_accepting_a_seeded_critical_omission_fails_closed(
    campaign: dict[str, Any],
) -> None:
    mutated = json.loads(json.dumps(campaign))
    target = next(
        item
        for item in mutated["paired_fixtures"]
        if item["scenario"] == "critical_omission"
    )
    observation = target["observations"]["candidate_optimized_supervisor"]
    observation["critical_omission_accepted"] = True
    observation["disposition"] = "reuse"
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="critical omission"):
        BENCHMARK.admit_campaign_result(mutated)


def test_stale_identity_admission_fails_closed(campaign: dict[str, Any]) -> None:
    mutated = json.loads(json.dumps(campaign))
    target = next(
        item for item in mutated["paired_fixtures"] if item["scenario"] == "stale_pack"
    )
    observation = target["observations"]["candidate_optimized_supervisor"]
    observation["stale_admitted"] = True
    observation["disposition"] = "reuse"
    observation["reused"] = True
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="stale"):
        BENCHMARK.admit_campaign_result(mutated)


def test_missing_audit_cost_fails_closed(campaign: dict[str, Any]) -> None:
    mutated = json.loads(json.dumps(campaign))
    observation = mutated["paired_fixtures"][0]["observations"][
        "candidate_optimized_supervisor"
    ]
    observation["metrics"]["audit_overhead"] = {
        "truth_state": "unavailable",
        "reason_code": "not_yet_measured",
    }
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="audit"):
        BENCHMARK.admit_campaign_result(mutated)


def test_three_arms_share_equal_controls(campaign: dict[str, Any]) -> None:
    assert campaign["arms"] == list(PAIRED_ARMS)
    assert set(campaign["controls"]) == set(EQUAL_CONTROL_FIELDS)
    controls = campaign["controls"]
    for field in EQUAL_CONTROL_FIELDS:
        values = {item["controls"][field] for item in campaign["paired_fixtures"]}
        assert values == {controls[field]}
    arm_controls = BENCHMARK.controls_for_arms(controls)
    arm_controls["candidate_optimized_supervisor"] = dict(
        arm_controls["candidate_optimized_supervisor"]
    )
    arm_controls["candidate_optimized_supervisor"]["maximum_retries"] = (
        controls["maximum_retries"] + 1
    )
    with pytest.raises(BENCHMARK.UnequalControlError, match="unequal controls"):
        BENCHMARK.run_paired_campaign(
            campaign["recipes"],
            arm_controls=arm_controls,
        )


def test_eligible_reuse_and_token_reduction_are_paired(campaign: dict[str, Any]) -> None:
    reuse_cases = [
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "eligible_reuse"
    ]
    assert reuse_cases
    for item in reuse_cases:
        direct = item["observations"]["direct_minimal_orchestration_baseline"]
        sealed = item["observations"]["sealed_current_supervisor_baseline"]
        candidate = item["observations"]["candidate_optimized_supervisor"]
        assert direct["reused"] is False
        assert direct["reuse_eligible"] is True
        assert sealed["reused"] is True
        assert candidate["reused"] is True
        before = candidate["metrics"]["context_tokens_before"]["value"]
        after = candidate["metrics"]["context_tokens_after"]["value"]
        assert after < before
        assert candidate["metrics"]["eligible_reuse"]["value"] == 1
        assert direct["metrics"]["eligible_reuse"]["value"] == 0


def test_expansion_precision_and_recall_are_measured_for_incremental_cases(
    campaign: dict[str, Any],
) -> None:
    expansions = [
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "incremental_expansion"
    ]
    assert expansions
    for item in expansions:
        candidate = item["observations"]["candidate_optimized_supervisor"]
        precision = candidate["metrics"]["expansion_precision"]
        recall = candidate["metrics"]["expansion_recall"]
        assert precision["truth_state"] == "measured"
        assert recall["truth_state"] == "measured"
        assert precision["unit"] == "ratio_millionths"
        assert recall["unit"] == "ratio_millionths"
        assert precision["value"] == BENCHMARK.MILLIONTHS
        assert recall["value"] == BENCHMARK.MILLIONTHS
        direct = item["observations"]["direct_minimal_orchestration_baseline"]
        assert direct["metrics"]["expansion_precision"]["value"] < BENCHMARK.MILLIONTHS


def test_paired_receipt_quantities_stay_nonnegative(campaign: dict[str, Any]) -> None:
    for item in campaign["paired_fixtures"]:
        for observation in item["observations"].values():
            for name, node in observation["metrics"].items():
                if node.get("truth_state") in {"measured", "estimated"}:
                    assert type(node["value"]) is int
                    assert node["value"] >= 0, (
                        f"{item['fixture_id']}.{observation['arm_id']}.{name}"
                    )
    for name, arms in campaign["aggregates"]["metrics"].items():
        for arm_id, node in arms.items():
            if node.get("truth_state") == "unavailable":
                continue
            assert node["mean"] >= 0, f"aggregates.{name}.{arm_id}.mean"
            assert node["median"] >= 0, f"aggregates.{name}.{arm_id}.median"


def test_net_provider_cost_effect_is_estimated_and_provider_charge_stays_unavailable(
    campaign: dict[str, Any],
) -> None:
    reuse_cases = [
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "eligible_reuse"
    ]
    candidate = reuse_cases[0]["observations"]["candidate_optimized_supervisor"]
    effect = candidate["metrics"]["net_provider_cost_effect"]
    assert effect["truth_state"] == "estimated"
    assert effect["method"] == "local_compute_model"
    assert effect["price_snapshot_identity"] == "unavailable"
    assert effect["value"] > 0
    charge = candidate["metrics"]["provider_reported_charge"]
    assert charge["truth_state"] == "unavailable"
    assert "value" not in charge


def test_sealed_manifest_matches_generator_and_retains_metrics(
    campaign: dict[str, Any],
) -> None:
    BENCHMARK.write_sealed_artifacts(campaign=campaign)
    verified = BENCHMARK.verify_sealed_artifacts(campaign=campaign)
    payload = _load_json(MANIFEST_PATH)
    assert payload["schema"] == BENCHMARK.MANIFEST_SCHEMA
    assert payload["interface"] == BENCHMARK.BENCHMARK_INTERFACE
    assert payload["task_id"] == "ASEH-035"
    assert payload["live"] is False
    assert payload["authority"] is False
    assert payload["required_metrics"] == list(BENCHMARK.REQUIRED_METRICS)
    assert payload["manifest_cid"] == verified["manifest_cid"]
    assert payload["fixture_count"] == verified["fixture_count"]
    assert payload["negative_cases"]["seeded_critical_omissions"]["accepted"] == 0
    assert payload["negative_cases"]["seeded_stale_packs"]["admitted"] == 0
    for item in payload["paired_fixtures"]:
        for observation in item["observations"].values():
            for name in BENCHMARK.REQUIRED_METRICS:
                assert name in observation["metrics"]
            assert "pack" not in observation
            assert "envelope" not in observation
    replayed = BENCHMARK.build_context_pack_manifest(campaign)
    assert replayed["manifest_cid"] == payload["manifest_cid"]
