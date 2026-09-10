"""ASEH-035 ContextPack reuse, omission, and net-savings benchmark."""

from __future__ import annotations

import copy
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
BENCHMARK.write_sealed_artifacts()
_CAMPAIGN: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _campaign() -> dict[str, Any]:
    global _CAMPAIGN
    if _CAMPAIGN is None:
        _CAMPAIGN = BENCHMARK.run_paired_campaign()
    return _CAMPAIGN


def test_manifest_seals_paired_current_tree_measures_for_every_required_metric() -> None:
    manifest = _load_json(MANIFEST_PATH)
    campaign = _campaign()
    recipes = BENCHMARK.generate_fixture_recipes()

    assert manifest["schema"] == BENCHMARK.MANIFEST_SCHEMA
    assert manifest["interface"] == BENCHMARK.BENCHMARK_INTERFACE
    assert manifest["task_id"] == "ASEH-035"
    assert manifest["live"] is False
    assert manifest["authority"] is False
    assert manifest["live_reuse_granted"] is False
    assert manifest["hermetic_sufficient_for_production_promotion"] is False
    assert manifest["promotion_without_paired_campaign"] is False
    assert manifest["critical_omissions_accepted"] == 0
    assert manifest["stale_packs_admitted"] == 0
    assert manifest["count"] == len(recipes) == campaign["fixture_count"]
    assert tuple(manifest["required_metrics"]) == BENCHMARK.REQUIRED_METRICS
    assert tuple(manifest["arms"]) == PAIRED_ARMS
    assert tuple(manifest["equal_control_fields"]) == EQUAL_CONTROL_FIELDS
    assert set(manifest["controls"]) == set(EQUAL_CONTROL_FIELDS)
    assert manifest["audit_overhead"]["total_compute_units"] >= BENCHMARK.AUDIT_OVERHEAD_FLOOR
    assert "paired_fixtures" in campaign
    assert campaign["paired_fixtures"], "paired fixtures are required"

    listed_ids = [item["fixture_id"] for item in manifest["fixtures"]]
    recipe_ids = [item["fixture_id"] for item in recipes]
    assert listed_ids == recipe_ids
    assert len(set(listed_ids)) == len(listed_ids)

    for pair in campaign["paired_fixtures"]:
        assert pair["live"] is False
        assert set(pair["observations"]) == set(PAIRED_ARMS)
        assert set(pair["controls"]) == set(EQUAL_CONTROL_FIELDS)
        for arm_id, observation in pair["observations"].items():
            assert observation["arm_id"] == arm_id
            assert observation["live"] is False
            assert observation["live_reuse_granted"] is False
            assert observation["simulated"] is True
            metrics = observation["metrics"]
            for name in BENCHMARK.REQUIRED_METRICS:
                assert name in metrics
            BENCHMARK._require_audit(
                metrics["audit_overhead"], fixture_id=pair["fixture_id"]
            )
            charge = metrics["net_provider_cost_effect"]["provider_reported_charge"]
            assert charge["truth_state"] == "unavailable"
            assert "value" not in charge
            assert charge.get("value", "unavailable") != 0

    for listed in manifest["fixtures"]:
        for name in BENCHMARK.REQUIRED_METRICS:
            assert name in listed["metrics"]
        assert listed["live"] is False


def test_unavailable_provider_cost_is_retained_and_never_encoded_as_zero() -> None:
    campaign = _campaign()
    unavailable_paths = collect_unavailable_fields(campaign)
    assert unavailable_paths
    assert any("provider_reported_charge" in path for path in unavailable_paths)
    assert any("token_based_estimated_charge" in path for path in unavailable_paths)

    charge_fixture = next(
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "unavailable_provider_charge"
    )
    for observation in charge_fixture["observations"].values():
        net = observation["metrics"]["net_provider_cost_effect"]
        assert net["provider_reported_charge"]["truth_state"] == "unavailable"
        assert net["token_based_estimated_charge"]["truth_state"] == "unavailable"
        assert net["unit_prices"]["truth_state"] == "unavailable"
        assert "value" not in net["provider_reported_charge"]
        assert net["provider_reported_charge"].get("reason_code") == "not_reported"
        expansion = observation["metrics"]["expansion_precision"]
        assert expansion["truth_state"] == "unavailable"
        assert "value" not in expansion
        assert expansion.get("value", "unavailable") != 0

    mutated = copy.deepcopy(charge_fixture)
    candidate = mutated["observations"]["candidate_optimized_supervisor"]
    candidate["metrics"]["net_provider_cost_effect"]["provider_reported_charge"] = {
        "truth_state": "unavailable",
        "reason_code": "not_reported",
        "value": 0,
    }
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="numeric zero"):
        BENCHMARK.admit_arm_observation(candidate)


def test_seeded_critical_omissions_are_always_rejected() -> None:
    campaign = _campaign()
    seeded = [
        item
        for item in campaign["paired_fixtures"]
        if item["seeded_critical_omission"] is True
    ]
    assert seeded
    for pair in seeded:
        assert pair["admitted"] is False
        for observation in pair["observations"].values():
            omission = observation["metrics"]["critical_omission_detection"]
            assert omission["truth_state"] == "measured"
            assert omission["seeded"] is True
            assert omission["detected"] is True
            assert omission["accepted"] is False
            assert observation["admitted"] is False
            assert observation["metrics"]["context_tokens_after"]["truth_state"] == (
                "unavailable"
            )
    assert campaign["critical_omissions_accepted"] == 0
    assert campaign["statistics"]["critical_omission_detection"]["accepted"] == 0

    mutated = copy.deepcopy(seeded[0])
    candidate = mutated["observations"]["candidate_optimized_supervisor"]
    candidate["metrics"]["critical_omission_detection"]["accepted"] = True
    with pytest.raises(
        BENCHMARK.AcceptedCriticalOmissionError, match="accepted critical omission"
    ):
        BENCHMARK.admit_arm_observation(candidate)


def test_seeded_stale_packs_are_always_rejected() -> None:
    campaign = _campaign()
    seeded = [
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "stale_pack"
    ]
    assert len(seeded) == len(BENCHMARK.STALE_FIELDS)
    seen_fields: set[str] = set()
    for pair in seeded:
        assert pair["admitted"] is False
        assert pair["seeded_stale"] is True
        field = pair["recipe"]["stale_field"]
        seen_fields.add(field)
        for observation in pair["observations"].values():
            stale = observation["metrics"]["stale_pack_rejection"]
            assert stale["truth_state"] == "measured"
            assert stale["seeded"] is True
            assert stale["rejected"] is True
            assert stale["admitted"] is False
            assert field in stale["stale_fields"]
            assert observation["admitted"] is False
            assert observation["metrics"]["eligible_reuse"]["reused"] is False
    assert seen_fields == set(BENCHMARK.STALE_FIELDS)
    assert campaign["stale_packs_admitted"] == 0

    mutated = copy.deepcopy(seeded[0])
    candidate = mutated["observations"]["candidate_optimized_supervisor"]
    candidate["metrics"]["stale_pack_rejection"]["admitted"] = True
    candidate["metrics"]["stale_pack_rejection"]["rejected"] = False
    with pytest.raises(
        BENCHMARK.StaleIdentityAdmissionError, match="stale identity admission"
    ):
        BENCHMARK.admit_arm_observation(candidate)


def test_fixture_pack_cannot_masquerade_as_live_or_grant_live_reuse() -> None:
    campaign = _campaign()
    pair = next(
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "fixture_as_live"
    )
    assert pair["live"] is False
    assert pair["admitted"] is False
    candidate = pair["observations"]["candidate_optimized_supervisor"]
    stale = candidate["metrics"]["stale_pack_rejection"]
    assert stale["rejected"] is True
    assert stale["admitted"] is False
    assert "fixture_as_live" in stale["masquerade_reasons"]
    assert candidate["live_reuse_granted"] is False
    assert campaign["live_reuse_granted"] is False
    assert campaign["hermetic_sufficient_for_production_promotion"] is False

    live_recipe = dict(BENCHMARK.generate_fixture_recipes()[0])
    live_recipe["live"] = True
    with pytest.raises(BENCHMARK.FixtureAsLiveError, match="cannot be live"):
        BENCHMARK.admit_recipe(live_recipe)

    mutated = copy.deepcopy(candidate)
    mutated["live"] = True
    with pytest.raises(BENCHMARK.FixtureAsLiveError, match="cannot mark an arm live"):
        BENCHMARK.admit_arm_observation(mutated)

    reused = copy.deepcopy(candidate)
    reused["live_reuse_granted"] = True
    with pytest.raises(BENCHMARK.FixtureAsLiveError, match="cannot grant live reuse"):
        BENCHMARK.admit_arm_observation(reused)


def test_eligible_reuse_and_before_after_tokens_are_measured_per_fixture() -> None:
    campaign = _campaign()
    reuse = next(
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "eligible_reuse"
    )
    tokens = next(
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "token_before_after"
    )
    candidate_reuse = reuse["observations"]["candidate_optimized_supervisor"]["metrics"]
    assert candidate_reuse["eligible_reuse"]["reused"] is True
    assert candidate_reuse["eligible_reuse"]["live_reuse_granted"] is False
    assert candidate_reuse["context_tokens_before"]["truth_state"] == "measured"
    assert candidate_reuse["context_tokens_after"]["truth_state"] == "measured"
    assert (
        candidate_reuse["context_tokens_after"]["value"]
        < candidate_reuse["context_tokens_before"]["value"]
    )
    assert (
        candidate_reuse["context_tokens_before"]["value"]
        <= BENCHMARK.DATASETS_COVERAGE_BUDGET_TOKENS
    )
    bloated = BENCHMARK.build_bloated()
    assert BENCHMARK.coverage_tokens(bloated) <= BENCHMARK.DATASETS_COVERAGE_BUDGET_TOKENS
    assert len(bloated.capsule_cids) <= BENCHMARK.MAX_COVERED_CAPSULES

    candidate_tokens = tokens["observations"]["candidate_optimized_supervisor"]["metrics"]
    direct_tokens = tokens["observations"]["direct_minimal_orchestration_baseline"][
        "metrics"
    ]
    assert candidate_tokens["eligible_reuse"]["reused"] is False
    assert (
        candidate_tokens["context_tokens_after"]["value"]
        < direct_tokens["context_tokens_after"]["value"]
    )
    assert (
        candidate_tokens["net_provider_cost_effect"]["token_delta"]["value"]
        > direct_tokens["net_provider_cost_effect"]["token_delta"]["value"]
    )


def test_expansion_precision_and_recall_are_exact_integer_ratios() -> None:
    campaign = _campaign()
    expansions = [
        item
        for item in campaign["paired_fixtures"]
        if item["scenario"] == "expansion_precision_recall"
    ]
    assert len(expansions) == len(BENCHMARK.NAMED_MISSING_KINDS)
    kinds = {item["recipe"]["named_missing_kind"] for item in expansions}
    assert kinds == set(BENCHMARK.NAMED_MISSING_KINDS)
    for pair in expansions:
        metrics = pair["observations"]["candidate_optimized_supervisor"]["metrics"]
        assert metrics["expansion_precision"]["truth_state"] == "measured"
        assert metrics["expansion_recall"]["truth_state"] == "measured"
        assert metrics["expansion_precision"]["value"] == BENCHMARK.MILLIONTHS
        assert metrics["expansion_recall"]["value"] == BENCHMARK.MILLIONTHS
        assert metrics["retrieval_cost"]["truth_state"] == "measured"
        assert metrics["retrieval_cost"]["value"] >= 1
        sealed = pair["observations"]["sealed_current_supervisor_baseline"]["metrics"]
        assert sealed["expansion_precision"]["truth_state"] == "unavailable"
        assert "value" not in sealed["expansion_precision"]


def test_pairing_rejects_unequal_controls_and_aggregate_only_results() -> None:
    recipes = BENCHMARK.generate_fixture_recipes()
    shared = BENCHMARK.shared_equal_controls(
        task_inputs=BENCHMARK.content_identity({"recipes": list(recipes)})
    )
    arm_controls = BENCHMARK.controls_for_arms(shared)
    arm_controls["candidate_optimized_supervisor"] = dict(
        arm_controls["candidate_optimized_supervisor"]
    )
    arm_controls["candidate_optimized_supervisor"]["maximum_retries"] = (
        shared["maximum_retries"] + 1
    )
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="unequal controls"):
        BENCHMARK.run_paired_campaign(recipes, arm_controls=arm_controls)

    missing = dict(arm_controls)
    missing.pop("direct_minimal_orchestration_baseline")
    with pytest.raises(BENCHMARK.ContextPackBenchmarkError, match="three closed arms"):
        BENCHMARK.require_equal_controls(missing)

    with pytest.raises(BENCHMARK.AggregateOnlyError, match="aggregate-only"):
        BENCHMARK.admit_campaign(
            {
                "live": False,
                "authority": False,
                "hermetic_sufficient_for_production_promotion": False,
                "live_reuse_granted": False,
                "critical_omissions_accepted": 0,
                "stale_packs_admitted": 0,
                "audit_overhead": {"total_compute_units": 12},
                "statistics": {"pair_count": 1},
            }
        )


def test_missing_audit_cost_invalidates_the_benchmark() -> None:
    campaign = _campaign()
    pair = campaign["paired_fixtures"][0]
    candidate = copy.deepcopy(
        pair["observations"]["candidate_optimized_supervisor"]
    )
    candidate["metrics"]["audit_overhead"] = BENCHMARK.unavailable("not_yet_measured")
    with pytest.raises(BENCHMARK.MissingAuditCostError, match="missing audit cost"):
        BENCHMARK.admit_arm_observation(candidate)

    zeroed = copy.deepcopy(pair["observations"]["candidate_optimized_supervisor"])
    zeroed["metrics"]["audit_overhead"]["value"] = 0
    with pytest.raises(BENCHMARK.MissingAuditCostError, match="missing audit cost"):
        BENCHMARK.admit_arm_observation(zeroed)

    mutated = copy.deepcopy(campaign)
    mutated["audit_overhead"] = {"total_compute_units": 0, "unit": "compute_units"}
    with pytest.raises(BENCHMARK.MissingAuditCostError, match="missing audit cost"):
        BENCHMARK.admit_campaign(mutated)


def test_sealed_manifest_matches_the_recipe_generator_and_cannot_promote() -> None:
    verified = BENCHMARK.verify_sealed_artifacts()
    assert verified["fixture_count"] == len(BENCHMARK.generate_fixture_recipes())
    assert verified["unique_identities"] == verified["fixture_count"]
    manifest = _load_json(MANIFEST_PATH)
    sealed = BENCHMARK.seal_artifacts()
    assert manifest == sealed["manifest"]
    assert manifest["manifest_cid"] == sealed["manifest"]["identity"]
    assert manifest["hermetic_sufficient_for_production_promotion"] is False
    assert manifest["live_reuse_granted"] is False
    replayed = BENCHMARK.build_context_pack_manifest(sealed["campaign"])
    assert replayed["manifest_cid"] == manifest["manifest_cid"]


def test_build_and_retrieval_costs_are_present_on_paired_arms() -> None:
    campaign = _campaign()
    for pair in campaign["paired_fixtures"]:
        for observation in pair["observations"].values():
            build_cost = observation["metrics"]["build_cost"]
            assert build_cost["truth_state"] == "measured"
            assert build_cost["value"] >= 1
            retrieval = observation["metrics"]["retrieval_cost"]
            assert retrieval["truth_state"] in {"measured", "unavailable"}
            if retrieval["truth_state"] == "unavailable":
                assert "value" not in retrieval
            audit = observation["metrics"]["audit_overhead"]
            assert audit["value"] >= 1
            assert audit["unit"] == "compute_units"
    assert campaign["audit_overhead"]["includes_audit_in_net_effect"] is True
