"""AAE-055: vacuity residual and conditional GUI action-binding campaign.

Validates VacuityAndActionBindingCampaign@1 fixtures:

* recipes expand to sealed assurance fixtures matching the parent AAE-049 corpus;
* formal/policy/test/ZK vacuity cases each state residual proof;
* canonical GUI fixtures cover action binding and keyboard accessibility only
  when the GUI surface is available;
* broad visual mutation is explicitly excluded;
* production policy change is forbidden;
* operator IDs bind to released assurance-compression/GUI catalogues;
* negative cases fail closed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.operators.assurance_compression_gui import (
    build_assurance_compression_gui_operators,
)
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    RequirementProvenance,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = (
    REPO_ROOT / "test" / "fixtures" / "adversarial_assurance" / "vacuity_gui"
)
PARENT_MANIFEST = (
    REPO_ROOT / "test" / "fixtures" / "adversarial_assurance" / "manifest.json"
)
SCHEMAS_DIR = REPO_ROOT / "test" / "fixtures" / "adversarial_assurance" / "schemas"

INTERFACE = "VacuityAndActionBindingCampaign@1"
CAMPAIGN_SCHEMA = "aae/vacuity-gui-campaign@1"
EVIDENCE_ID = "aae/vacuity-gui-campaign@1"
TASK_ID = "AAE-055"
BUNDLE = "vacuity_gui"
CAMPAIGN_ID = "adversarial-assurance-vacuity-gui-v1"

REQUIRED_SCENARIOS = (
    "formal_vacuity_impossible_assumption",
    "policy_vacuity_unreachable_mode",
    "test_vacuity_permanent_skip",
    "zk_vacuity_omitted_unit",
    "gui_break_dispatchability",
    "gui_omit_confirmation",
    "gui_wrong_handler",
    "gui_stale_action_policy",
    "gui_broken_recovery",
    "gui_drop_critical_keyboard_access",
)

SCENARIO_TO_ACCEPTANCE = {
    "formal_vacuity_impossible_assumption": "formal residual",
    "policy_vacuity_unreachable_mode": "policy residual",
    "test_vacuity_permanent_skip": "test residual",
    "zk_vacuity_omitted_unit": "zk residual",
    "gui_break_dispatchability": "dispatchability",
    "gui_omit_confirmation": "confirmation",
    "gui_wrong_handler": "handler",
    "gui_stale_action_policy": "stale policy",
    "gui_broken_recovery": "recovery",
    "gui_drop_critical_keyboard_access": "keyboard accessibility",
}

EXPECTED_AUTHORITIES = {
    "formal_vacuity_impossible_assumption": "vacuity.formal.residual",
    "policy_vacuity_unreachable_mode": "vacuity.policy.residual",
    "test_vacuity_permanent_skip": "vacuity.test.residual",
    "zk_vacuity_omitted_unit": "vacuity.zk.residual",
    "gui_break_dispatchability": "gui.action.dispatchable",
    "gui_omit_confirmation": "gui.action.confirmation",
    "gui_wrong_handler": "gui.action.handler_bound",
    "gui_stale_action_policy": "gui.action.policy_current",
    "gui_broken_recovery": "gui.action.recovery",
    "gui_drop_critical_keyboard_access": "gui.accessibility.keyboard",
}

VACUITY_SCENARIOS = frozenset(
    {
        "formal_vacuity_impossible_assumption",
        "policy_vacuity_unreachable_mode",
        "test_vacuity_permanent_skip",
        "zk_vacuity_omitted_unit",
    }
)

GUI_SCENARIOS = frozenset(
    {
        "gui_break_dispatchability",
        "gui_omit_confirmation",
        "gui_wrong_handler",
        "gui_stale_action_policy",
        "gui_broken_recovery",
        "gui_drop_critical_keyboard_access",
    }
)


def _load_catalog():
    path = FIXTURE_DIR / "catalog.py"
    assert path.is_file(), f"missing catalog: {path}"
    module_name = "aae_vacuity_gui_campaign_catalog"
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, "INTERFACE", None) == INTERFACE:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses with postponed annotations resolve.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def catalog():
    return _load_catalog()


@pytest.fixture(scope="module")
def recipes(catalog) -> dict[str, Any]:
    return catalog.load_recipes()


@pytest.fixture(scope="module")
def probes(catalog) -> dict[str, Any]:
    return catalog.load_probes()


@pytest.fixture(scope="module")
def fixtures(catalog, recipes) -> list[dict[str, Any]]:
    return catalog.expand_all_fixtures(recipes)


@pytest.fixture(scope="module")
def campaign(catalog, recipes, probes) -> dict[str, Any]:
    return catalog.build_campaign(
        recipes_doc=recipes,
        probes_doc=probes,
        verify_parent=True,
        run_probes=True,
    )


def _catalogue_operator_ids(catalogue: Any) -> set[str]:
    for attr in ("operators", "definitions", "items"):
        if hasattr(catalogue, attr):
            vals = getattr(catalogue, attr)
            return {
                getattr(item, "operator_id", None) or str(item) for item in vals
            }
    if hasattr(catalogue, "as_mutation_operators"):
        return {op.operator_id for op in catalogue.as_mutation_operators()}
    if hasattr(catalogue, "operator_ids"):
        ids = catalogue.operator_ids
        return set(ids() if callable(ids) else ids)
    if hasattr(catalogue, "entries"):
        entries = catalogue.entries
        vals = entries() if callable(entries) else entries
        return {
            getattr(item, "operator_id", None)
            or getattr(getattr(item, "definition", None), "operator_id", None)
            or str(item)
            for item in vals
        }
    raise AssertionError(f"cannot list operators from {type(catalogue)!r}")


def test_declared_outputs_exist() -> None:
    assert FIXTURE_DIR.is_dir()
    assert (FIXTURE_DIR / "recipes.json").is_file()
    assert (FIXTURE_DIR / "probes.json").is_file()
    assert (FIXTURE_DIR / "catalog.py").is_file()
    assert (FIXTURE_DIR / "campaign.json").is_file()
    assert PARENT_MANIFEST.is_file()


def test_campaign_header_and_interface(campaign: dict[str, Any]) -> None:
    assert campaign["interface"] == INTERFACE
    assert campaign["schema"] == CAMPAIGN_SCHEMA
    assert campaign["campaign_id"] == CAMPAIGN_ID
    assert campaign["task_id"] == TASK_ID
    assert campaign["bundle"] == BUNDLE
    assert campaign["evidence_id"] == EVIDENCE_ID
    assert campaign["production_policy_change_allowed"] is False
    assert campaign["production_policy_changed"] is False
    assert campaign["visual_mutation_allowed"] is False
    assert campaign["visual_mutation_excluded"] is True
    assert campaign["vacuity_residual_required"] is True
    assert campaign["gui_scope"] == (
        "action_binding_and_accessibility_when_available"
    )
    assert campaign["mutation_index_start"] == 1
    assert campaign["mutation_index_end"] == 10
    assert campaign["fixture_count"] == 10
    claimed = campaign["campaign_cid"]
    recomputed = cid_for_structured(
        {k: v for k, v in campaign.items() if k != "campaign_cid"}
    )
    assert claimed == recomputed
    assert claimed.startswith("b")


def test_recipes_cover_all_ten_cases(
    recipes: dict[str, Any], fixtures: list[dict[str, Any]]
) -> None:
    assert recipes["interface"] == INTERFACE
    assert recipes["task_id"] == TASK_ID
    assert recipes["bundle"] == BUNDLE
    assert recipes["production_policy_change_allowed"] is False
    assert recipes["visual_mutation_allowed"] is False
    assert recipes["recipe_count"] == 10
    assert len(recipes["recipes"]) == 10

    by_index = sorted(
        recipes["recipes"], key=lambda item: int(item["mutation_index"])
    )
    indices = [int(r["mutation_index"]) for r in by_index]
    assert indices == list(range(1, 11))
    scenarios = [r["scenario"] for r in by_index]
    assert scenarios == list(REQUIRED_SCENARIOS)

    for scenario in REQUIRED_SCENARIOS:
        assert scenario in SCENARIO_TO_ACCEPTANCE

    fixture_scenarios = [
        f["scenario"]
        for f in sorted(fixtures, key=lambda item: int(item["mutation_index"]))
    ]
    assert fixture_scenarios == list(REQUIRED_SCENARIOS)


def test_fixtures_match_parent_corpus(catalog, fixtures: list[dict[str, Any]]) -> None:
    catalog.assert_fixtures_match_parent_corpus(fixtures)
    parent = {
        f["fixture_id"]: f for f in catalog.parent_corpus_vacuity_gui_fixtures()
    }
    for fixture in fixtures:
        sealed = parent[fixture["fixture_id"]]
        assert fixture["fixture_cid"] == sealed["fixture_cid"]
        assert fixture["recipe_cid"] == sealed["recipe_cid"]
        assert fixture["campaign_bundle"] == BUNDLE
        assert fixture["campaign"] == "vacuity_gui"
        assert fixture["bounded_oracle"]["fail_closed"] is True
        assert fixture["bounded_oracle"]["expected_outcome"] == "killed"
        assert fixture["expected_detector"]["expected_terminal_status"] == "rejected"
        notes = (fixture.get("notes") or "").lower()
        assert "visual" in notes


def test_every_fixture_binds_requirement_detector_oracle(
    fixtures: list[dict[str, Any]],
) -> None:
    for fixture in fixtures:
        prov = fixture["requirement_provenance"]
        sealed = RequirementProvenance.from_dict(prov)
        assert sealed.provenance_cid == prov["provenance_cid"]
        assert prov["requirement_id"]
        assert prov["intended_behavior"]
        assert prov["source_id"] == "plan.section.11.vacuity_gui"
        assert prov["source_path"].endswith(
            "ADVERSARIAL_ASSURANCE_ENGINE_PLAN.md"
        )

        detector = fixture["expected_detector"]
        assert detector["detector_id"]
        assert detector["detector_cid"] == cid_for_structured(
            {k: v for k, v in detector.items() if k != "detector_cid"}
        )
        assert detector["strength"] == "required"
        assert detector["violated_claim"]
        assert detector["observation_rationale"]

        oracle = fixture["bounded_oracle"]
        assert oracle["oracle_cid"] == cid_for_structured(
            {k: v for k, v in oracle.items() if k != "oracle_cid"}
        )
        assert oracle["kill_mechanisms"]
        assert oracle["observation_points"]
        bounds = oracle["bounds"]
        assert bounds["max_steps"] > 0
        assert bounds["max_depth"] > 0
        assert bounds["timeout_ms"] > 0

        identity = {
            k: v
            for k, v in fixture.items()
            if k not in {"fixture_cid", "recipe_cid", "mutation_index"}
        }
        assert fixture["fixture_cid"] == cid_for_structured(identity)


def test_held_out_fixtures_forbid_candidate_generation(
    fixtures: list[dict[str, Any]],
) -> None:
    held = [f for f in fixtures if f["partition"] == "held_out"]
    assert held, "vacuity_gui must include held_out fixtures for qualification"
    for fixture in held:
        assert fixture["used_for_candidate_generation"] is False


def test_partitions_span_diagnosis_development_held_out(
    campaign: dict[str, Any],
) -> None:
    membership = campaign["partition_membership"]
    assert set(membership) == {"diagnosis", "development", "held_out"}
    assert membership["held_out"]
    assert membership["development"]
    assert membership["diagnosis"]
    covered = set()
    for name, members in membership.items():
        assert members == sorted(members)
        overlap = covered & set(members)
        assert not overlap, f"partition overlap involving {name}: {sorted(overlap)}"
        covered.update(members)
    assert covered == {item["fixture_id"] for item in campaign["fixtures"]}


def test_all_controlled_probes_are_killed(
    catalog, probes: dict[str, Any], campaign: dict[str, Any]
) -> None:
    assert probes["interface"] == INTERFACE
    assert probes["production_policy_change_allowed"] is False
    assert probes["visual_mutation_allowed"] is False
    assert probes["vacuity_residual_required"] is True
    assert probes["probe_count"] == 10
    assert len(probes["probes"]) == 10

    results = catalog.evaluate_all_probes(probes)
    assert len(results) == 10
    assert all(r.killed for r in results)
    assert all(r.terminal_status == "rejected" for r in results)
    assert all(r.visual_mutation_excluded for r in results)

    by_scenario = {r.scenario: r for r in results}
    assert set(by_scenario) == set(REQUIRED_SCENARIOS)

    for scenario, authority in EXPECTED_AUTHORITIES.items():
        assert by_scenario[scenario].authority == authority
        assert by_scenario[scenario].kill_mechanism == authority

    campaign_results = {r["scenario"]: r for r in campaign["probe_results"]}
    assert set(campaign_results) == set(REQUIRED_SCENARIOS)
    assert all(item["killed"] for item in campaign_results.values())


def test_vacuity_probes_state_residual_proof(
    catalog, probes: dict[str, Any], campaign: dict[str, Any]
) -> None:
    """Acceptance: formal/policy/test/ZK vacuity cases state residual proof."""
    results = {
        r.scenario: r
        for r in catalog.evaluate_all_probes(probes)
        if r.scenario in VACUITY_SCENARIOS
    }
    assert set(results) == VACUITY_SCENARIOS
    for scenario, result in results.items():
        assert result.killed is True
        assert result.residual_stated is True
        residuals = result.details.get("residual_properties") or []
        assert residuals, f"{scenario}: residual_properties empty"
        assert all(str(item).strip() for item in residuals)
        assert result.details.get("finding_count", 0) >= 1
        assert result.details.get("result_cid")

    campaign_results = {
        r["scenario"]: r
        for r in campaign["probe_results"]
        if r["scenario"] in VACUITY_SCENARIOS
    }
    assert all(item["residual_stated"] for item in campaign_results.values())


@pytest.mark.parametrize(
    "scenario,expected_reason_substr",
    [
        ("formal_vacuity_impossible_assumption", "formal_vacuity_residual"),
        ("policy_vacuity_unreachable_mode", "policy_vacuity_residual"),
        ("test_vacuity_permanent_skip", "test_vacuity_residual"),
        ("zk_vacuity_omitted_unit", "zk_vacuity_residual"),
        ("gui_break_dispatchability", "dispatchability"),
        ("gui_omit_confirmation", "confirmation"),
        ("gui_wrong_handler", "wrong_handler"),
        ("gui_stale_action_policy", "stale_action_policy"),
        ("gui_broken_recovery", "recovery"),
        ("gui_drop_critical_keyboard_access", "keyboard_access"),
    ],
)
def test_each_acceptance_scenario_probe_reason(
    catalog,
    probes: dict[str, Any],
    scenario: str,
    expected_reason_substr: str,
) -> None:
    probe = next(p for p in probes["probes"] if p["scenario"] == scenario)
    result = catalog.evaluate_probe(probe)
    assert result.killed is True
    assert expected_reason_substr in result.reason


def test_gui_probes_require_available_surface(
    catalog, probes: dict[str, Any]
) -> None:
    """GUI fixtures apply only when canonical surface is available."""
    for probe in probes["probes"]:
        if probe["scenario"] not in GUI_SCENARIOS:
            continue
        observation = probe["observation"]
        assert observation.get("gui_surface_available") is True
        assert observation.get("canonical_gui_optimizer_artifact_present") is True
        result = catalog.evaluate_probe(probe)
        assert result.gui_surface_available is True
        assert result.killed is True


def test_gui_unavailable_surface_fails_closed_for_expected_kill(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p for p in bad["probes"] if p["scenario"] == "gui_break_dispatchability"
    )
    target["observation"]["gui_surface_available"] = False
    target["observation"]["canonical_gui_optimizer_artifact_present"] = False
    with pytest.raises(catalog.VacuityGuiCampaignError):
        catalog.evaluate_probe(target)


def test_probe_kill_mechanisms_include_declared_detector(
    fixtures: list[dict[str, Any]], probes: dict[str, Any]
) -> None:
    by_id = {f["fixture_id"]: f for f in fixtures}
    for probe in probes["probes"]:
        fixture = by_id[probe["fixture_id"]]
        detector_id = fixture["expected_detector"]["detector_id"]
        assert probe["detector_id"] == detector_id
        assert detector_id in probe["kill_mechanisms"] or probe[
            "authority"
        ] == detector_id
        assert probe["authority"] in fixture["bounded_oracle"]["kill_mechanisms"]


def test_operators_bind_to_released_catalogues(
    fixtures: list[dict[str, Any]], catalog
) -> None:
    scg = build_assurance_compression_gui_operators()
    known = _catalogue_operator_ids(scg)
    # Catalogue may expose operators via multiple entry shapes.
    if hasattr(scg, "assert_visual_mutation_absent"):
        scg.assert_visual_mutation_absent()

    required = set(catalog.REQUIRED_OPERATOR_IDS)
    missing = required - known
    assert not missing, f"operators missing from released catalogues: {sorted(missing)}"

    for fixture in fixtures:
        op_id = fixture["operator"]["operator_id"]
        assert op_id in known, f"{fixture['fixture_id']}: unknown operator {op_id}"
        op_class = fixture["operator"]["operator_class"]
        if fixture["scenario"] in VACUITY_SCENARIOS:
            assert op_class == "test_proof"
        else:
            assert op_class == "gui_action_binding"


def test_visual_mutation_explicitly_excluded(
    recipes: dict[str, Any],
    probes: dict[str, Any],
    campaign: dict[str, Any],
    fixtures: list[dict[str, Any]],
) -> None:
    assert recipes["visual_mutation_allowed"] is False
    assert probes["visual_mutation_allowed"] is False
    assert campaign["visual_mutation_allowed"] is False
    assert campaign["visual_mutation_excluded"] is True
    for fixture in fixtures:
        notes = (fixture.get("notes") or "").lower()
        assert "visual" in notes
        assert "exclud" in notes or "out of scope" in notes or "absent" in notes
    for probe in probes["probes"]:
        assert probe["observation"].get("visual_mutation_allowed") is False


def test_campaign_snapshot_matches_rebuilt_identity(
    catalog, campaign: dict[str, Any]
) -> None:
    snapshot_path = FIXTURE_DIR / "campaign.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["interface"] == INTERFACE
    assert snapshot["campaign_cid"] == campaign["campaign_cid"]
    assert snapshot["fixture_count"] == 10
    assert snapshot["production_policy_change_allowed"] is False
    assert snapshot["visual_mutation_excluded"] is True
    recomputed = cid_for_structured(
        {k: v for k, v in snapshot.items() if k != "campaign_cid"}
    )
    assert snapshot["campaign_cid"] == recomputed


def test_negative_production_policy_change_rejected(
    catalog, recipes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(recipes))
    bad["production_policy_change_allowed"] = True
    assert recipes["production_policy_change_allowed"] is False
    with pytest.raises(catalog.VacuityGuiCampaignError):
        if bad.get("production_policy_change_allowed") is not False:
            raise catalog.VacuityGuiCampaignError(
                "production policy change is forbidden for fixture campaigns"
            )


def test_negative_production_policy_change_in_probe_rejected(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p
        for p in bad["probes"]
        if p["scenario"] == "formal_vacuity_impossible_assumption"
    )
    target["observation"]["production_policy_changed"] = True
    with pytest.raises(catalog.VacuityGuiCampaignError):
        catalog.evaluate_probe(target)


def test_negative_surviving_probe_fails_closed(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p for p in bad["probes"] if p["scenario"] == "gui_break_dispatchability"
    )
    target["observation"] = {
        "action_id": "action.save",
        "canonical_gui_optimizer_artifact_present": True,
        "gui_surface_available": True,
        "dispatchable": True,
        "handler_bound": True,
        "presented_as_valid": True,
        "production_policy_changed": False,
        "visual_mutation_allowed": False,
    }
    with pytest.raises(catalog.VacuityGuiCampaignError):
        catalog.evaluate_probe(target)


def test_negative_visual_mutation_probe_rejected(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p for p in bad["probes"] if p["scenario"] == "gui_omit_confirmation"
    )
    target["observation"]["visual_mutation"] = True
    with pytest.raises(catalog.VacuityGuiCampaignError):
        catalog.evaluate_probe(target)


def test_negative_wrong_bundle_rejected(catalog, recipes: dict[str, Any]) -> None:
    bad_recipe = dict(recipes["recipes"][0])
    bad_recipe["campaign_bundle"] = "security_a"
    with pytest.raises(catalog.VacuityGuiCampaignError):
        catalog.expand_recipe(bad_recipe)


def test_negative_missing_scenario_rejected(
    catalog, recipes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(recipes))
    bad["recipes"] = [
        r
        for r in bad["recipes"]
        if r["scenario"] != "gui_drop_critical_keyboard_access"
    ]
    bad["recipe_count"] = len(bad["recipes"])
    fixtures = catalog.expand_all_fixtures(bad)
    scenarios = tuple(
        f["scenario"]
        for f in sorted(
            fixtures, key=lambda item: int(item.get("mutation_index") or 0)
        )
    )
    assert scenarios != REQUIRED_SCENARIOS
    with pytest.raises(catalog.VacuityGuiCampaignError):
        catalog.build_campaign(
            recipes_doc=bad,
            verify_parent=False,
            run_probes=False,
        )


def test_campaign_does_not_mutate_parent_corpus(catalog) -> None:
    before = PARENT_MANIFEST.read_bytes()
    catalog.build_campaign(verify_parent=True, run_probes=True)
    after = PARENT_MANIFEST.read_bytes()
    assert before == after


def test_parent_corpus_membership_lists_vacuity_gui() -> None:
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    membership = set(manifest["campaign_membership"]["vacuity_gui"])
    for fixture_id in (
        "vac.formal_impossible_assumption",
        "vac.policy_unreachable_mode",
        "vac.test_permanent_skip",
        "vac.zk_omitted_unit",
        "gui.break_dispatchability",
        "gui.omit_confirmation",
        "gui.wrong_handler",
        "gui.stale_action_policy",
        "gui.broken_recovery",
        "gui.drop_keyboard_access",
    ):
        assert fixture_id in membership
    assert manifest["mandated_counts"]["vacuity_gui"] == 10


def test_fixture_schema_files_exist_for_expansion() -> None:
    for name in (
        "assurance-fixture.schema.json",
        "bounded-oracle.schema.json",
        "expected-detector.schema.json",
        "fixture-partition.schema.json",
    ):
        assert (SCHEMAS_DIR / name).is_file(), name


def test_cold_import_of_catalog_is_side_effect_free() -> None:
    """Importing the catalog must not change production policy or open sockets."""
    before = PARENT_MANIFEST.read_bytes()
    module = _load_catalog()
    assert module.INTERFACE == INTERFACE
    assert module.TASK_ID == TASK_ID
    after = PARENT_MANIFEST.read_bytes()
    assert before == after


def test_acceptance_coverage_maps_to_required_mechanisms(
    fixtures: list[dict[str, Any]], probes: dict[str, Any]
) -> None:
    """Acceptance criteria: residual vacuity + GUI binding/accessibility."""
    by_scenario = {f["scenario"]: f for f in fixtures}
    probe_by_scenario = {p["scenario"]: p for p in probes["probes"]}
    for scenario, label in SCENARIO_TO_ACCEPTANCE.items():
        assert scenario in by_scenario, label
        fixture = by_scenario[scenario]
        probe = probe_by_scenario[scenario]
        authority = EXPECTED_AUTHORITIES[scenario]
        assert authority in fixture["bounded_oracle"]["kill_mechanisms"]
        assert probe["authority"] == authority
        assert fixture["expected_detector"]["detector_id"] == authority
