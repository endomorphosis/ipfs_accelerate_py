"""AAE-052: semantic-compression mutation campaign (SemanticCompressionAssuranceCampaign@1).

Validates the semantic_compression campaign fixtures:

* recipes expand to sealed assurance fixtures matching the parent AAE-049 corpus;
* all eight dependency/exception/fixture/stale/heuristic/opaque/selection/
  expanded-context cases run;
* each controlled probe is killed by its declared SCG mechanism / authority;
* every probe produces non-authoritative SCG calibration evidence;
* production policy change is forbidden;
* operator IDs bind to released assurance-compression catalogues;
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
    REPO_ROOT
    / "test"
    / "fixtures"
    / "adversarial_assurance"
    / "semantic_compression"
)
PARENT_MANIFEST = (
    REPO_ROOT / "test" / "fixtures" / "adversarial_assurance" / "manifest.json"
)
SCHEMAS_DIR = REPO_ROOT / "test" / "fixtures" / "adversarial_assurance" / "schemas"

INTERFACE = "SemanticCompressionAssuranceCampaign@1"
CAMPAIGN_SCHEMA = "aae/semantic-compression-campaign@1"
EVIDENCE_ID = "aae/semantic-compression-campaign@1"
TASK_ID = "AAE-052"
BUNDLE = "semantic_compression"
CAMPAIGN_ID = "adversarial-assurance-semantic-compression-v1"

REQUIRED_SCENARIOS = (
    "omit_required_side_effect",
    "omit_exception_path",
    "omit_result_changing_fixture",
    "stale_capsule_conceals_schema",
    "heuristic_substituted_for_raw",
    "opaque_plugin_as_exact",
    "miss_relevant_selected_test",
    "expanded_context_succeeds_compressed_fails",
)

SCENARIO_TO_ACCEPTANCE = {
    "omit_required_side_effect": "dependency",
    "omit_exception_path": "exception",
    "omit_result_changing_fixture": "fixture",
    "stale_capsule_conceals_schema": "stale",
    "heuristic_substituted_for_raw": "heuristic",
    "opaque_plugin_as_exact": "opaque",
    "miss_relevant_selected_test": "selection",
    "expanded_context_succeeds_compressed_fails": "expanded-context",
}

EXPECTED_AUTHORITIES = {
    "omit_required_side_effect": "scg.context.side_effect",
    "omit_exception_path": "scg.context.exception",
    "omit_result_changing_fixture": "scg.context.fixture",
    "stale_capsule_conceals_schema": "scg.capsule.root_freshness",
    "heuristic_substituted_for_raw": "scg.confidence.exactness",
    "opaque_plugin_as_exact": "scg.confidence.opaque",
    "miss_relevant_selected_test": "scg.selection.coverage",
    "expanded_context_succeeds_compressed_fails": (
        "scg.calibration.expanded_vs_compressed"
    ),
}


def _load_catalog():
    path = FIXTURE_DIR / "catalog.py"
    assert path.is_file(), f"missing catalog: {path}"
    module_name = "aae_semantic_compression_campaign_catalog"
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
    assert campaign["scg_calibration_authoritative"] is False
    assert campaign["mutation_index_start"] == 1
    assert campaign["mutation_index_end"] == 8
    assert campaign["fixture_count"] == 8
    claimed = campaign["campaign_cid"]
    recomputed = cid_for_structured(
        {k: v for k, v in campaign.items() if k != "campaign_cid"}
    )
    assert claimed == recomputed
    assert claimed.startswith("b")


def test_recipes_cover_all_eight_cases(
    recipes: dict[str, Any], fixtures: list[dict[str, Any]]
) -> None:
    assert recipes["interface"] == INTERFACE
    assert recipes["task_id"] == TASK_ID
    assert recipes["bundle"] == BUNDLE
    assert recipes["production_policy_change_allowed"] is False
    assert recipes["recipe_count"] == 8
    assert len(recipes["recipes"]) == 8

    by_index = sorted(
        recipes["recipes"], key=lambda item: int(item["mutation_index"])
    )
    indices = [int(r["mutation_index"]) for r in by_index]
    assert indices == list(range(1, 9))
    scenarios = [r["scenario"] for r in by_index]
    assert scenarios == list(REQUIRED_SCENARIOS)

    for scenario in REQUIRED_SCENARIOS:
        assert scenario in SCENARIO_TO_ACCEPTANCE

    fixture_scenarios = [
        f["scenario"]
        for f in sorted(fixtures, key=lambda item: int(item["mutation_index"]))
    ]
    assert fixture_scenarios == list(REQUIRED_SCENARIOS)

    acceptance_classes = [
        SCENARIO_TO_ACCEPTANCE[s] for s in REQUIRED_SCENARIOS
    ]
    assert acceptance_classes == [
        "dependency",
        "exception",
        "fixture",
        "stale",
        "heuristic",
        "opaque",
        "selection",
        "expanded-context",
    ]


def test_fixtures_match_parent_corpus(catalog, fixtures: list[dict[str, Any]]) -> None:
    catalog.assert_fixtures_match_parent_corpus(fixtures)
    parent = {
        f["fixture_id"]: f
        for f in catalog.parent_corpus_semantic_compression_fixtures()
    }
    for fixture in fixtures:
        sealed = parent[fixture["fixture_id"]]
        assert fixture["fixture_cid"] == sealed["fixture_cid"]
        assert fixture["recipe_cid"] == sealed["recipe_cid"]
        assert fixture["campaign_bundle"] == BUNDLE
        assert fixture["campaign"] == "semantic_compression"
        assert fixture["bounded_oracle"]["fail_closed"] is True
        assert fixture["bounded_oracle"]["expected_outcome"] == "killed"
        assert fixture["expected_detector"]["expected_terminal_status"] == "rejected"
        kill = fixture["bounded_oracle"]["kill_mechanisms"]
        assert "scg.calibration" in kill
        assert any(m.startswith("scg.") for m in kill)


def test_every_fixture_binds_requirement_detector_oracle(
    fixtures: list[dict[str, Any]],
) -> None:
    for fixture in fixtures:
        prov = fixture["requirement_provenance"]
        sealed = RequirementProvenance.from_dict(prov)
        assert sealed.provenance_cid == prov["provenance_cid"]
        assert prov["requirement_id"]
        assert prov["intended_behavior"]
        assert prov["source_id"] == "plan.section.11.semantic_compression"
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
    assert held, "semantic_compression must include held_out fixtures"
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
    assert probes["probe_count"] == 8
    assert len(probes["probes"]) == 8

    results = catalog.evaluate_all_probes(probes)
    assert len(results) == 8
    assert all(r.killed for r in results)
    assert all(r.terminal_status == "rejected" for r in results)

    by_scenario = {r.scenario: r for r in results}
    assert set(by_scenario) == set(REQUIRED_SCENARIOS)

    for scenario, authority in EXPECTED_AUTHORITIES.items():
        assert by_scenario[scenario].authority == authority
        assert by_scenario[scenario].kill_mechanism == authority

    campaign_results = {r["scenario"]: r for r in campaign["probe_results"]}
    assert set(campaign_results) == set(REQUIRED_SCENARIOS)
    assert all(item["killed"] for item in campaign_results.values())


@pytest.mark.parametrize(
    "scenario,expected_reason_substr",
    [
        ("omit_required_side_effect", "side_effect"),
        ("omit_exception_path", "exception"),
        ("omit_result_changing_fixture", "fixture"),
        ("stale_capsule_conceals_schema", "stale_capsule"),
        ("heuristic_substituted_for_raw", "heuristic"),
        ("opaque_plugin_as_exact", "opaque"),
        ("miss_relevant_selected_test", "selected_test"),
        ("expanded_context_succeeds_compressed_fails", "expanded_beats"),
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
        assert "scg.calibration" in probe["kill_mechanisms"]


def test_operators_bind_to_released_catalogues(
    fixtures: list[dict[str, Any]], catalog
) -> None:
    sc_ids = _catalogue_operator_ids(build_assurance_compression_gui_operators())
    known = {op for op in sc_ids if op.startswith("sc_")}

    required = set(catalog.REQUIRED_OPERATOR_IDS)
    missing = required - known
    assert not missing, f"operators missing from released catalogues: {sorted(missing)}"

    for fixture in fixtures:
        op_id = fixture["operator"]["operator_id"]
        assert op_id in known, f"{fixture['fixture_id']}: unknown operator {op_id}"
        assert fixture["operator"]["operator_class"] == "semantic_compression"


def test_scg_calibration_evidence_for_all_eight_cases(
    catalog, probes: dict[str, Any], campaign: dict[str, Any]
) -> None:
    bundle = catalog.collect_scg_calibration_evidence(probes)
    assert bundle["record_count"] == 8
    assert bundle["production_policy_change_allowed"] is False
    assert bundle["production_policy_changed"] is False
    assert bundle["authoritative_for_production_policy"] is False
    assert bundle["consumer"] == "SemanticCompressionGovernor"
    assert bundle["acceptance_classes"] == list(SCENARIO_TO_ACCEPTANCE.values())
    assert bundle["scenarios"] == list(REQUIRED_SCENARIOS)
    claimed = bundle["calibration_bundle_cid"]
    recomputed = cid_for_structured(
        {k: v for k, v in bundle.items() if k != "calibration_bundle_cid"}
    )
    assert claimed == recomputed

    by_scenario = {r["scenario"]: r for r in bundle["records"]}
    assert set(by_scenario) == set(REQUIRED_SCENARIOS)
    for scenario, record in by_scenario.items():
        assert record["killed"] is True
        assert record["production_policy_changed"] is False
        assert record["authoritative_for_production_policy"] is False
        assert record["production_policy_change_allowed"] is False
        assert record["acceptance_class"] == SCENARIO_TO_ACCEPTANCE[scenario]
        assert record["authority"] == EXPECTED_AUTHORITIES[scenario]
        assert record["consumer"] == "SemanticCompressionGovernor"
        assert record["evidence_cid"] == cid_for_structured(
            {k: v for k, v in record.items() if k != "evidence_cid"}
        )

    campaign_cal = campaign["scg_calibration"]
    assert campaign_cal is not None
    assert campaign_cal["calibration_bundle_cid"] == bundle["calibration_bundle_cid"]
    assert campaign_cal["record_count"] == 8
    assert all(r["killed"] for r in campaign["probe_results"])
    for result in campaign["probe_results"]:
        assert "calibration_evidence" in result
        assert result["calibration_evidence"]["killed"] is True
        assert result["calibration_evidence"]["production_policy_changed"] is False


def test_campaign_snapshot_matches_rebuilt_identity(
    catalog, campaign: dict[str, Any]
) -> None:
    snapshot_path = FIXTURE_DIR / "campaign.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["interface"] == INTERFACE
    assert snapshot["campaign_cid"] == campaign["campaign_cid"]
    assert snapshot["fixture_count"] == 8
    assert snapshot["production_policy_change_allowed"] is False
    assert snapshot["production_policy_changed"] is False
    assert snapshot["scg_calibration_authoritative"] is False
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
    with pytest.raises(catalog.SemanticCompressionCampaignError):
        if bad.get("production_policy_change_allowed") is not False:
            raise catalog.SemanticCompressionCampaignError(
                "production policy change is forbidden for fixture campaigns"
            )


def test_negative_production_policy_change_in_probe_rejected(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p for p in bad["probes"] if p["scenario"] == "omit_required_side_effect"
    )
    target["observation"]["production_policy_changed"] = True
    with pytest.raises(catalog.SemanticCompressionCampaignError):
        catalog.evaluate_probe(target)


def test_negative_surviving_probe_fails_closed(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p for p in bad["probes"] if p["scenario"] == "omit_required_side_effect"
    )
    # Neutralize the omission so the detector would not kill.
    target["observation"] = {
        "side_effect_required": True,
        "side_effect_in_context": True,
        "capsule_admitted": True,
        "production_policy_changed": False,
    }
    with pytest.raises(catalog.SemanticCompressionCampaignError):
        catalog.evaluate_probe(target)


def test_negative_wrong_bundle_rejected(catalog, recipes: dict[str, Any]) -> None:
    bad_recipe = dict(recipes["recipes"][0])
    bad_recipe["campaign_bundle"] = "security_a"
    with pytest.raises(catalog.SemanticCompressionCampaignError):
        catalog.expand_recipe(bad_recipe)


def test_negative_missing_scenario_rejected(
    catalog, recipes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(recipes))
    bad["recipes"] = [
        r
        for r in bad["recipes"]
        if r["scenario"] != "expanded_context_succeeds_compressed_fails"
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
    with pytest.raises(catalog.SemanticCompressionCampaignError):
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


def test_parent_corpus_membership_lists_semantic_compression() -> None:
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    membership = set(manifest["campaign_membership"]["semantic_compression"])
    for fixture_id in (
        "sc.omit_required_side_effect",
        "sc.omit_exception",
        "sc.omit_result_changing_fixture",
        "sc.stale_capsule_schema",
        "sc.heuristic_for_raw",
        "sc.opaque_plugin_as_exact",
        "sc.miss_selected_test",
        "sc.expanded_beats_compressed",
    ):
        assert fixture_id in membership
    assert manifest["mandated_counts"]["semantic_compression"] == 8


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
    """Acceptance criteria: each named class has a fixture, kill, and calibration."""
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
        assert "scg.calibration" in fixture["bounded_oracle"]["kill_mechanisms"]
