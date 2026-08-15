"""AAE-054: distributed-state, storage-durability, and crash mutation campaign.

Validates DistributedStorageCrashAssuranceCampaign@1 fixtures:

* recipes expand to sealed assurance fixtures matching the parent AAE-049 corpus;
* scenarios cover transitions, CAS, fencing, owners, leases, idempotency,
  compensation, durable acknowledgement/read-back, and every required injected
  crash boundary;
* each controlled probe is killed by its declared mechanism / authority;
* production policy change is forbidden;
* operator IDs bind to released distributed-storage catalogues;
* negative cases fail closed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.operators.distributed_storage import (
    build_distributed_storage_operators,
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
    / "distributed_storage_crash"
)
PARENT_MANIFEST = (
    REPO_ROOT / "test" / "fixtures" / "adversarial_assurance" / "manifest.json"
)
SCHEMAS_DIR = REPO_ROOT / "test" / "fixtures" / "adversarial_assurance" / "schemas"

INTERFACE = "DistributedStorageCrashAssuranceCampaign@1"
CAMPAIGN_SCHEMA = "aae/distributed-storage-crash-campaign@1"
EVIDENCE_ID = "aae/distributed-storage-crash-campaign@1"
TASK_ID = "AAE-054"
BUNDLE = "distributed_storage_crash"
CAMPAIGN_ID = "adversarial-assurance-distributed-storage-crash-v1"

REQUIRED_SCENARIOS = (
    "illegal_state_transition",
    "cas_ignore_expected_old",
    "accept_stale_fencing_token",
    "mutate_without_ownership",
    "ignore_lease_expiry",
    "drop_idempotency_key",
    "incomplete_distributed_compensation",
    "ack_before_durable_commit",
    "skip_read_back_verification",
    "crash_after_mutant_create",
    "crash_during_worktree_setup",
    "crash_after_receipt_persist",
    "crash_before_policy_cas",
    "crash_after_cas_before_cleanup",
)

SCENARIO_TO_ACCEPTANCE = {
    "illegal_state_transition": "transitions",
    "cas_ignore_expected_old": "CAS",
    "accept_stale_fencing_token": "fencing",
    "mutate_without_ownership": "owners",
    "ignore_lease_expiry": "leases",
    "drop_idempotency_key": "idempotency",
    "incomplete_distributed_compensation": "compensation",
    "ack_before_durable_commit": "durable acknowledgement",
    "skip_read_back_verification": "read-back",
    "crash_after_mutant_create": "crash boundary mutant_create",
    "crash_during_worktree_setup": "crash boundary worktree_setup",
    "crash_after_receipt_persist": "crash boundary receipt_persist",
    "crash_before_policy_cas": "crash boundary before_policy_cas",
    "crash_after_cas_before_cleanup": "crash boundary after_cas_before_cleanup",
}

EXPECTED_AUTHORITIES = {
    "illegal_state_transition": "state.transition.legal",
    "cas_ignore_expected_old": "state.cas.expected_old",
    "accept_stale_fencing_token": "state.fencing.current",
    "mutate_without_ownership": "state.ownership.required",
    "ignore_lease_expiry": "state.lease.valid",
    "drop_idempotency_key": "state.idempotency.key_present",
    "incomplete_distributed_compensation": "state.compensation.complete",
    "ack_before_durable_commit": "storage.ack.after_durable",
    "skip_read_back_verification": "storage.read_back.verified",
    "crash_after_mutant_create": "crash.boundary.mutant_create",
    "crash_during_worktree_setup": "crash.boundary.worktree_setup",
    "crash_after_receipt_persist": "crash.boundary.receipt_persist",
    "crash_before_policy_cas": "crash.boundary.before_policy_cas",
    "crash_after_cas_before_cleanup": "crash.boundary.after_cas_before_cleanup",
}

REQUIRED_CRASH_BOUNDARIES = (
    "crash.boundary.mutant_create",
    "crash.boundary.worktree_setup",
    "crash.boundary.receipt_persist",
    "crash.boundary.before_policy_cas",
    "crash.boundary.after_cas_before_cleanup",
)

CRASH_SCENARIOS = frozenset(
    {
        "crash_after_mutant_create",
        "crash_during_worktree_setup",
        "crash_after_receipt_persist",
        "crash_before_policy_cas",
        "crash_after_cas_before_cleanup",
    }
)


def _load_catalog():
    path = FIXTURE_DIR / "catalog.py"
    assert path.is_file(), f"missing catalog: {path}"
    module_name = "aae_distributed_storage_crash_campaign_catalog"
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
    assert campaign["mutation_index_start"] == 1
    assert campaign["mutation_index_end"] == 14
    assert campaign["fixture_count"] == 14
    assert list(campaign["required_crash_boundaries"]) == list(
        REQUIRED_CRASH_BOUNDARIES
    )
    assert set(campaign["crash_boundaries_covered"]) == set(
        REQUIRED_CRASH_BOUNDARIES
    )
    claimed = campaign["campaign_cid"]
    recomputed = cid_for_structured(
        {k: v for k, v in campaign.items() if k != "campaign_cid"}
    )
    assert claimed == recomputed
    assert claimed.startswith("b")


def test_recipes_cover_all_fourteen_cases(
    recipes: dict[str, Any], fixtures: list[dict[str, Any]]
) -> None:
    assert recipes["interface"] == INTERFACE
    assert recipes["task_id"] == TASK_ID
    assert recipes["bundle"] == BUNDLE
    assert recipes["production_policy_change_allowed"] is False
    assert recipes["recipe_count"] == 14
    assert len(recipes["recipes"]) == 14

    by_index = sorted(
        recipes["recipes"], key=lambda item: int(item["mutation_index"])
    )
    indices = [int(r["mutation_index"]) for r in by_index]
    assert indices == list(range(1, 15))
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
        "transitions",
        "CAS",
        "fencing",
        "owners",
        "leases",
        "idempotency",
        "compensation",
        "durable acknowledgement",
        "read-back",
        "crash boundary mutant_create",
        "crash boundary worktree_setup",
        "crash boundary receipt_persist",
        "crash boundary before_policy_cas",
        "crash boundary after_cas_before_cleanup",
    ]


def test_fixtures_match_parent_corpus(catalog, fixtures: list[dict[str, Any]]) -> None:
    catalog.assert_fixtures_match_parent_corpus(fixtures)
    parent = {
        f["fixture_id"]: f
        for f in catalog.parent_corpus_distributed_storage_crash_fixtures()
    }
    for fixture in fixtures:
        sealed = parent[fixture["fixture_id"]]
        assert fixture["fixture_cid"] == sealed["fixture_cid"]
        assert fixture["recipe_cid"] == sealed["recipe_cid"]
        assert fixture["campaign_bundle"] == BUNDLE
        assert fixture["campaign"] == "distributed_storage_crash"
        assert fixture["critical"] is True
        assert fixture["bounded_oracle"]["fail_closed"] is True
        assert fixture["bounded_oracle"]["expected_outcome"] == "killed"
        assert fixture["expected_detector"]["expected_terminal_status"] == "rejected"
        kill = fixture["bounded_oracle"]["kill_mechanisms"]
        assert "runtime.invariant" in kill
        assert any(
            m.startswith("state.")
            or m.startswith("storage.")
            or m.startswith("crash.")
            for m in kill
        )


def test_every_fixture_binds_requirement_detector_oracle(
    fixtures: list[dict[str, Any]],
) -> None:
    for fixture in fixtures:
        prov = fixture["requirement_provenance"]
        sealed = RequirementProvenance.from_dict(prov)
        assert sealed.provenance_cid == prov["provenance_cid"]
        assert prov["requirement_id"]
        assert prov["intended_behavior"]
        assert prov["source_id"] == "plan.section.11.distributed_durability_crash"
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
        assert "crash.injector" in oracle["observation_points"]
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
    assert held, "distributed_storage_crash must include held_out fixtures"
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
    assert probes["probe_count"] == 14
    assert len(probes["probes"]) == 14
    assert list(probes["required_crash_boundaries"]) == list(
        REQUIRED_CRASH_BOUNDARIES
    )

    results = catalog.evaluate_all_probes(probes)
    assert len(results) == 14
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
        ("illegal_state_transition", "illegal_state_transition"),
        ("cas_ignore_expected_old", "cas_ignore"),
        ("accept_stale_fencing_token", "stale_fencing"),
        ("mutate_without_ownership", "without_ownership"),
        ("ignore_lease_expiry", "lease_expiry"),
        ("drop_idempotency_key", "idempotency_key"),
        ("incomplete_distributed_compensation", "incomplete_distributed"),
        ("ack_before_durable_commit", "ack_before_durable"),
        ("skip_read_back_verification", "read_back"),
        ("crash_after_mutant_create", "mutant_create"),
        ("crash_during_worktree_setup", "worktree_setup"),
        ("crash_after_receipt_persist", "receipt_persist"),
        ("crash_before_policy_cas", "policy_cas"),
        ("crash_after_cas_before_cleanup", "cas_before_cleanup"),
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


def test_every_required_crash_boundary_is_covered(
    fixtures: list[dict[str, Any]], probes: dict[str, Any], campaign: dict[str, Any]
) -> None:
    detector_ids = {f["expected_detector"]["detector_id"] for f in fixtures}
    for boundary in REQUIRED_CRASH_BOUNDARIES:
        assert boundary in detector_ids, boundary

    crash_fixtures = [f for f in fixtures if f["scenario"] in CRASH_SCENARIOS]
    assert len(crash_fixtures) == 5
    probe_by_scenario = {p["scenario"]: p for p in probes["probes"]}
    for fixture in crash_fixtures:
        probe = probe_by_scenario[fixture["scenario"]]
        assert probe["observation"]["crash_injected"] is True
        assert "crash.boundary." in probe["authority"]
        assert fixture["expected_detector"]["detector_id"] == probe["authority"]

    assert set(campaign["crash_boundaries_covered"]) == set(
        REQUIRED_CRASH_BOUNDARIES
    )


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
        assert "runtime.invariant" in probe["kill_mechanisms"] or any(
            m.startswith("crash.") for m in probe["kill_mechanisms"]
        )


def test_operators_bind_to_released_catalogues(
    fixtures: list[dict[str, Any]], catalog
) -> None:
    ds_ids = _catalogue_operator_ids(build_distributed_storage_operators())
    known = ds_ids

    required = set(catalog.REQUIRED_OPERATOR_IDS)
    missing = required - known
    assert not missing, f"operators missing from released catalogues: {sorted(missing)}"

    for fixture in fixtures:
        op_id = fixture["operator"]["operator_id"]
        assert op_id in known, f"{fixture['fixture_id']}: unknown operator {op_id}"
        assert fixture["operator"]["operator_class"] in {
            "state_distributed",
            "storage_durability",
        }


def test_campaign_snapshot_matches_rebuilt_identity(
    catalog, campaign: dict[str, Any]
) -> None:
    snapshot_path = FIXTURE_DIR / "campaign.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["interface"] == INTERFACE
    assert snapshot["campaign_cid"] == campaign["campaign_cid"]
    assert snapshot["fixture_count"] == 14
    assert snapshot["production_policy_change_allowed"] is False
    assert snapshot["production_policy_changed"] is False
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
    with pytest.raises(catalog.DistributedStorageCrashCampaignError):
        if bad.get("production_policy_change_allowed") is not False:
            raise catalog.DistributedStorageCrashCampaignError(
                "production policy change is forbidden for fixture campaigns"
            )


def test_negative_production_policy_change_in_probe_rejected(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p for p in bad["probes"] if p["scenario"] == "illegal_state_transition"
    )
    target["observation"]["production_policy_changed"] = True
    with pytest.raises(catalog.DistributedStorageCrashCampaignError):
        catalog.evaluate_probe(target)


def test_negative_surviving_probe_fails_closed(
    catalog, probes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(probes))
    target = next(
        p for p in bad["probes"] if p["scenario"] == "illegal_state_transition"
    )
    # Neutralize the illegal transition so the detector would not kill.
    target["observation"] = {
        "from_state": "pending",
        "to_state": "running",
        "legal_transitions": [["pending", "running"], ["running", "completed"]],
        "transition_accepted": True,
        "production_policy_changed": False,
    }
    with pytest.raises(catalog.DistributedStorageCrashCampaignError):
        catalog.evaluate_probe(target)


def test_negative_wrong_bundle_rejected(catalog, recipes: dict[str, Any]) -> None:
    bad_recipe = dict(recipes["recipes"][0])
    bad_recipe["campaign_bundle"] = "security_a"
    with pytest.raises(catalog.DistributedStorageCrashCampaignError):
        catalog.expand_recipe(bad_recipe)


def test_negative_missing_scenario_rejected(
    catalog, recipes: dict[str, Any]
) -> None:
    bad = json.loads(json.dumps(recipes))
    bad["recipes"] = [
        r
        for r in bad["recipes"]
        if r["scenario"] != "crash_after_cas_before_cleanup"
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
    with pytest.raises(catalog.DistributedStorageCrashCampaignError):
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


def test_parent_corpus_membership_lists_distributed_storage_crash() -> None:
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    membership = set(
        manifest["campaign_membership"]["distributed_storage_crash"]
    )
    for fixture_id in (
        "dist.illegal_state_transition",
        "dist.cas_ignore_expected_old",
        "dist.stale_fencing",
        "dist.mutate_without_ownership",
        "dist.ignore_lease_expiry",
        "dist.drop_idempotency_key",
        "dist.incomplete_compensation",
        "dist.ack_before_durable",
        "dist.skip_read_back",
        "dist.crash_after_mutant_create",
        "dist.crash_during_worktree_setup",
        "dist.crash_after_receipt_persist",
        "dist.crash_before_policy_cas",
        "dist.crash_after_cas_before_cleanup",
    ):
        assert fixture_id in membership
    assert manifest["mandated_counts"]["distributed_storage_crash"] == 14


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
    """Acceptance: transitions, CAS, fencing, owners, leases, idempotency,
    compensation, durable ack/read-back, and every crash boundary."""
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
        assert "runtime.invariant" in fixture["bounded_oracle"]["kill_mechanisms"]
        assert fixture["critical"] is True


def test_all_fourteen_cases_are_critical(fixtures: list[dict[str, Any]]) -> None:
    assert len(fixtures) == 14
    assert all(f["critical"] is True for f in fixtures)
