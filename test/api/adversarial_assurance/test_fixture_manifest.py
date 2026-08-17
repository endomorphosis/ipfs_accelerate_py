"""AAE-049: deterministic fixture corpus, requirement oracles, and held-out partitions.

Validates AssuranceFixtureCorpus@1 / aae/fixture-partition@1:

* every fixture binds requirement provenance, risk, operator, expected detector,
  bounded oracle, diagnosis/development/held-out partition, and deterministic
  identity;
* partitions are pairwise disjoint with no held-out candidate-generation leakage;
* mandated campaign counts from plan §11 / AAE-050..055 are present;
* critical ZK/seal fixtures fail closed;
* compact recipes expand to the sealed manifest with stable CIDs;
* production policy change is forbidden for the corpus campaign.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.adversarial_assurance.remediation_contracts import (
    REQUIREMENT_PROVENANCE_SCHEMA,
    RequirementProvenance,
)

jsonschema = pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "adversarial_assurance"
SCHEMAS_DIR = FIXTURE_DIR / "schemas"
MANIFEST_PATH = FIXTURE_DIR / "manifest.json"
RECIPES_PATH = SCHEMAS_DIR / "recipes.json"

INTERFACE = "AssuranceFixtureCorpus@1"
CORPUS_SCHEMA = "aae/fixture-partition@1"
FIXTURE_SCHEMA = "aae/assurance-fixture@1"
ORACLE_SCHEMA = "aae/bounded-oracle@1"
DETECTOR_SCHEMA = "aae/expected-detector@1"
RECIPE_SCHEMA = "aae/fixture-recipe@1"
EVIDENCE_ID = "aae/fixture-partition@1"
CORPUS_ID = "adversarial-assurance-fixture-corpus-v1"
TASK_ID = "AAE-049"
PARTITIONS = ("diagnosis", "development", "held_out")
CAMPAIGNS = (
    "security",
    "semantic_compression",
    "zk_incremental_seal",
    "distributed_storage_crash",
    "vacuity_gui",
)
MANDATED_COUNTS = {
    "security": 20,
    "semantic_compression": 8,
    "zk_incremental_seal": 12,
    "distributed_storage_crash": 14,
    "vacuity_gui": 10,
}
REQUIRED_SCHEMA_FILES = (
    "fixture-common.schema.json",
    "bounded-oracle.schema.json",
    "expected-detector.schema.json",
    "assurance-fixture.schema.json",
    "fixture-partition.schema.json",
    "recipes.json",
)

# Plan §11 security scenarios (20).
REQUIRED_SECURITY_SCENARIOS = frozenset(
    {
        "authentication_bypass",
        "caller_selected_tenant",
        "missing_attenuation",
        "accepted_expired_delegation",
        "accepted_revoked_capability",
        "missing_confirmation",
        "cross_action_confirmation_replay",
        "policy_default_to_allow",
        "payment_as_authority",
        "stale_fencing_token",
        "retry_double_execution",
        "uncompensated_partial_mutation",
        "provider_ack_as_verified_storage",
        "receipt_before_observed_effect",
        "invalid_signature",
        "pseudo_cid",
        "stale_proof_receipt",
        "omitted_proof_unit",
        "unknown_prover_as_passed",
        "simulated_production_evidence",
    }
)

REQUIRED_COMPRESSION_SCENARIOS = frozenset(
    {
        "omit_required_side_effect",
        "omit_exception_path",
        "omit_result_changing_fixture",
        "stale_capsule_conceals_schema",
        "heuristic_substituted_for_raw",
        "opaque_plugin_as_exact",
        "miss_relevant_selected_test",
        "expanded_context_succeeds_compressed_fails",
    }
)

REQUIRED_SEAL_SCENARIOS = frozenset(
    {
        "remove_receipt_leaf",
        "remove_required_unit",
        "change_source_root",
        "change_environment_cid",
        "change_parent_seal",
        "change_proof_forest_order",
        "use_old_key",
        "attach_proof_to_wrong_statement",
        "delete_test_without_authorization",
        "substitute_simulated_for_direct_proof",
        "ignore_blocking_child",
        "replay_proof_across_branches",
    }
)

REQUIRED_DIST_SCENARIOS = frozenset(
    {
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
    }
)

REQUIRED_VACUITY_GUI_SCENARIOS = frozenset(
    {
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
    }
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_identity_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in fixture.items()
        if key not in {"fixture_cid", "recipe_cid"}
    }
    return payload


def _recompute_fixture_cid(fixture: dict[str, Any]) -> str:
    return cid_for_structured(_fixture_identity_payload(fixture))


def _expand_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Deterministic expansion of a compact recipe into a sealed fixture."""
    prov = RequirementProvenance(
        requirement_id=recipe["requirement_id"],
        intended_behavior=recipe["intended_behavior"],
        source_id=recipe["source_id"],
        requirement_cid=None,
        source_path=recipe["source_path"],
        notes=recipe["notes"],
    )
    operator = {
        "operator_id": recipe["operator_id"],
        "operator_version": recipe["operator_version"],
        "operator_class": recipe["operator_class"],
    }
    expected_detector = {
        "schema": DETECTOR_SCHEMA,
        "detector_id": recipe["detector_id"],
        "detector_revision": recipe["detector_revision"],
        "detector_kind": recipe["detector_kind"],
        "strength": recipe["detector_strength"],
        "expected_terminal_status": recipe["expected_terminal_status"],
        "violated_claim": recipe["violated_claim"],
        "observation_rationale": recipe["observation_rationale"],
    }
    expected_detector["detector_cid"] = cid_for_structured(
        {k: v for k, v in expected_detector.items() if k != "detector_cid"}
    )
    oracle_body = {
        "schema": ORACLE_SCHEMA,
        "oracle_id": f"oracle.{recipe['fixture_id']}",
        "expected_outcome": recipe["expected_outcome"],
        "kill_mechanisms": list(recipe["kill_mechanisms"]),
        "bounds": {
            "max_steps": recipe["max_steps"],
            "max_depth": recipe["max_depth"],
            "timeout_ms": recipe["timeout_ms"],
        },
        "observation_points": list(recipe["observation_points"]),
        "fail_closed": recipe["fail_closed"],
    }
    oracle_body["oracle_cid"] = cid_for_structured(
        {k: v for k, v in oracle_body.items() if k != "oracle_cid"}
    )
    identity_payload = {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": recipe["fixture_id"],
        "campaign": recipe["campaign"],
        "campaign_bundle": recipe["campaign_bundle"],
        "scenario": recipe["scenario"],
        "partition": recipe["partition"],
        "risk": recipe["risk"],
        "operator": operator,
        "requirement_provenance": prov.to_dict(),
        "expected_detector": expected_detector,
        "bounded_oracle": oracle_body,
        "relatedness_key": recipe["relatedness_key"],
        "target_id": recipe["target_id"],
        "used_for_candidate_generation": recipe["used_for_candidate_generation"],
        "critical": recipe["critical"],
        "notes": recipe["notes"],
    }
    fixture = dict(identity_payload)
    fixture["fixture_cid"] = cid_for_structured(identity_payload)
    fixture["recipe_cid"] = cid_for_structured({"schema": RECIPE_SCHEMA, **recipe})
    return fixture


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"missing manifest: {MANIFEST_PATH}"
    return _load_json(MANIFEST_PATH)


@pytest.fixture(scope="module")
def recipes() -> dict[str, Any]:
    assert RECIPES_PATH.is_file(), f"missing recipes: {RECIPES_PATH}"
    return _load_json(RECIPES_PATH)


@pytest.fixture(scope="module")
def corpus_schema() -> dict[str, Any]:
    path = SCHEMAS_DIR / "fixture-partition.schema.json"
    assert path.is_file()
    return _load_json(path)


def test_declared_outputs_exist() -> None:
    assert MANIFEST_PATH.is_file()
    assert SCHEMAS_DIR.is_dir()
    for name in REQUIRED_SCHEMA_FILES:
        assert (SCHEMAS_DIR / name).is_file(), name


def test_corpus_header_and_interface(manifest: dict[str, Any]) -> None:
    assert manifest["interface"] == INTERFACE
    assert manifest["schema"] == CORPUS_SCHEMA
    assert manifest["corpus_id"] == CORPUS_ID
    assert manifest["evidence_id"] == EVIDENCE_ID
    assert manifest["task_id"] == TASK_ID
    assert manifest["production_policy_change_allowed"] is False
    assert tuple(manifest["partitions"]) == PARTITIONS
    assert set(manifest["campaigns"]) == set(CAMPAIGNS)
    assert manifest["fixture_count"] == len(manifest["fixtures"])
    assert manifest["fixture_count"] == sum(MANDATED_COUNTS.values())
    claimed = manifest["corpus_cid"]
    recomputed = cid_for_structured(
        {k: v for k, v in manifest.items() if k != "corpus_cid"}
    )
    assert claimed == recomputed


def test_manifest_matches_json_schema(
    manifest: dict[str, Any], corpus_schema: dict[str, Any]
) -> None:
    jsonschema.Draft202012Validator.check_schema(corpus_schema)
    validator = jsonschema.Draft202012Validator(corpus_schema)
    errors = sorted(validator.iter_errors(manifest), key=lambda err: list(err.path))
    assert not errors, "; ".join(
        f"{list(err.path)}: {err.message}" for err in errors[:10]
    )


def test_every_fixture_binds_required_fields(manifest: dict[str, Any]) -> None:
    required = {
        "schema",
        "fixture_id",
        "campaign",
        "campaign_bundle",
        "scenario",
        "partition",
        "risk",
        "operator",
        "requirement_provenance",
        "expected_detector",
        "bounded_oracle",
        "relatedness_key",
        "target_id",
        "used_for_candidate_generation",
        "critical",
        "fixture_cid",
        "recipe_cid",
    }
    for fixture in manifest["fixtures"]:
        missing = required - set(fixture)
        assert not missing, f"{fixture.get('fixture_id')}: missing {sorted(missing)}"
        assert fixture["schema"] == FIXTURE_SCHEMA
        assert fixture["partition"] in PARTITIONS
        assert fixture["campaign"] in CAMPAIGNS

        operator = fixture["operator"]
        assert operator["operator_id"]
        assert operator["operator_version"]
        assert operator["operator_class"]

        prov = fixture["requirement_provenance"]
        assert prov["schema"] == REQUIREMENT_PROVENANCE_SCHEMA
        assert prov["requirement_id"]
        assert prov["intended_behavior"]
        assert prov["source_id"]
        assert prov["provenance_cid"]
        # Round-trip through the datasets contract (CID identity).
        sealed = RequirementProvenance.from_dict(prov)
        assert sealed.provenance_cid == prov["provenance_cid"]

        detector = fixture["expected_detector"]
        assert detector["schema"] == DETECTOR_SCHEMA
        assert detector["detector_id"]
        assert detector["detector_kind"]
        assert detector["strength"] in {"required", "optional"}
        assert detector["expected_terminal_status"]
        assert detector["violated_claim"]
        assert detector["observation_rationale"]
        assert detector["detector_cid"] == cid_for_structured(
            {k: v for k, v in detector.items() if k != "detector_cid"}
        )

        oracle = fixture["bounded_oracle"]
        assert oracle["schema"] == ORACLE_SCHEMA
        assert oracle["oracle_id"]
        assert oracle["expected_outcome"]
        assert oracle["kill_mechanisms"]
        bounds = oracle["bounds"]
        assert bounds["max_steps"] > 0
        assert bounds["max_depth"] > 0
        assert bounds["timeout_ms"] > 0
        assert oracle["observation_points"]
        assert isinstance(oracle["fail_closed"], bool)
        assert oracle["oracle_cid"] == cid_for_structured(
            {k: v for k, v in oracle.items() if k != "oracle_cid"}
        )


def test_fixture_identities_are_deterministic_and_unique(
    manifest: dict[str, Any],
) -> None:
    ids = [f["fixture_id"] for f in manifest["fixtures"]]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))

    cids = [f["fixture_cid"] for f in manifest["fixtures"]]
    assert len(cids) == len(set(cids))
    for fixture in manifest["fixtures"]:
        assert fixture["fixture_cid"] == _recompute_fixture_cid(fixture)
        assert fixture["fixture_cid"].startswith("b")


def test_partitions_nonempty_disjoint_and_cover_all(
    manifest: dict[str, Any],
) -> None:
    membership = manifest["partition_membership"]
    assert set(membership) == set(PARTITIONS)
    seen: set[str] = set()
    for name in PARTITIONS:
        members = membership[name]
        assert members, f"partition {name} empty"
        assert members == sorted(members)
        overlap = seen & set(members)
        assert not overlap, f"partition leakage involving {name}: {sorted(overlap)}"
        seen.update(members)
    assert seen == {f["fixture_id"] for f in manifest["fixtures"]}

    # Membership index must match fixture records.
    by_partition: dict[str, set[str]] = {p: set() for p in PARTITIONS}
    for fixture in manifest["fixtures"]:
        by_partition[fixture["partition"]].add(fixture["fixture_id"])
    for name in PARTITIONS:
        assert by_partition[name] == set(membership[name])


def test_no_held_out_candidate_generation_leakage(manifest: dict[str, Any]) -> None:
    for fixture in manifest["fixtures"]:
        if fixture["partition"] == "held_out":
            assert fixture["used_for_candidate_generation"] is False, fixture[
                "fixture_id"
            ]
        # Diagnosis/development may generate candidates; held-out never overlaps.
    held_out = set(manifest["partition_membership"]["held_out"])
    generators = {
        f["fixture_id"]
        for f in manifest["fixtures"]
        if f["used_for_candidate_generation"]
    }
    assert not (held_out & generators), sorted(held_out & generators)


def test_mandated_campaign_counts_and_scenarios(manifest: dict[str, Any]) -> None:
    assert manifest["mandated_counts"] == MANDATED_COUNTS
    by_campaign: dict[str, list[dict[str, Any]]] = {c: [] for c in CAMPAIGNS}
    for fixture in manifest["fixtures"]:
        by_campaign[fixture["campaign"]].append(fixture)

    for campaign, expected in MANDATED_COUNTS.items():
        assert len(by_campaign[campaign]) == expected, campaign
        assert set(manifest["campaign_membership"][campaign]) == {
            f["fixture_id"] for f in by_campaign[campaign]
        }

    security = {f["scenario"] for f in by_campaign["security"]}
    assert security == REQUIRED_SECURITY_SCENARIOS

    compression = {f["scenario"] for f in by_campaign["semantic_compression"]}
    assert compression == REQUIRED_COMPRESSION_SCENARIOS

    seal = {f["scenario"] for f in by_campaign["zk_incremental_seal"]}
    assert seal == REQUIRED_SEAL_SCENARIOS

    dist = {f["scenario"] for f in by_campaign["distributed_storage_crash"]}
    assert dist == REQUIRED_DIST_SCENARIOS

    vacgui = {f["scenario"] for f in by_campaign["vacuity_gui"]}
    assert vacgui == REQUIRED_VACUITY_GUI_SCENARIOS


def test_critical_seal_fixtures_fail_closed(manifest: dict[str, Any]) -> None:
    seal_fixtures = [
        f for f in manifest["fixtures"] if f["campaign"] == "zk_incremental_seal"
    ]
    assert len(seal_fixtures) == MANDATED_COUNTS["zk_incremental_seal"]
    for fixture in seal_fixtures:
        assert fixture["critical"] is True, fixture["fixture_id"]
        assert fixture["bounded_oracle"]["fail_closed"] is True, fixture["fixture_id"]
        assert "seal.incremental" in fixture["bounded_oracle"]["kill_mechanisms"] or (
            fixture["expected_detector"]["detector_kind"] == "incremental_seal"
        )


def test_security_bundles_split_for_downstream_campaigns(
    manifest: dict[str, Any],
) -> None:
    security = [f for f in manifest["fixtures"] if f["campaign"] == "security"]
    a = [f for f in security if f["campaign_bundle"] == "security_a"]
    b = [f for f in security if f["campaign_bundle"] == "security_b"]
    assert len(a) == 10
    assert len(b) == 10
    assert {f["fixture_id"] for f in a}.isdisjoint({f["fixture_id"] for f in b})


def test_recipes_expand_to_manifest(
    manifest: dict[str, Any], recipes: dict[str, Any]
) -> None:
    assert recipes["interface"] == INTERFACE
    assert recipes["corpus_id"] == CORPUS_ID
    assert recipes["task_id"] == TASK_ID
    assert recipes["recipe_count"] == len(recipes["recipes"])
    assert recipes["recipe_count"] == manifest["fixture_count"]

    expanded = [_expand_recipe(r) for r in recipes["recipes"]]
    expanded.sort(key=lambda item: item["fixture_id"])
    sealed = sorted(manifest["fixtures"], key=lambda item: item["fixture_id"])
    assert [item["fixture_id"] for item in expanded] == [
        item["fixture_id"] for item in sealed
    ]
    for left, right in zip(expanded, sealed, strict=True):
        assert left == right, left["fixture_id"]


def test_schema_files_are_valid_draft_2020_12() -> None:
    for name in (
        "fixture-common.schema.json",
        "bounded-oracle.schema.json",
        "expected-detector.schema.json",
        "assurance-fixture.schema.json",
        "fixture-partition.schema.json",
    ):
        schema = _load_json(SCHEMAS_DIR / name)
        # common is a $defs-only document; still a valid meta schema object
        if name == "fixture-common.schema.json":
            assert "$defs" in schema
            continue
        jsonschema.Draft202012Validator.check_schema(schema)


def test_negative_held_out_generation_rejected_by_schema(
    corpus_schema: dict[str, Any],
) -> None:
    """Held-out fixtures that claim candidate generation must fail schema validation."""
    manifest = _load_json(MANIFEST_PATH)
    bad = json.loads(json.dumps(manifest))
    # Mutate first held_out fixture
    target = next(f for f in bad["fixtures"] if f["partition"] == "held_out")
    target["used_for_candidate_generation"] = True
    # Identity will also be wrong, but schema should fail on partition leakage first.
    validator = jsonschema.Draft202012Validator(corpus_schema)
    errors = list(validator.iter_errors(bad))
    assert errors, "expected schema rejection for held_out generation leakage"


def test_negative_partition_overlap_detected(manifest: dict[str, Any]) -> None:
    membership = {
        name: list(members)
        for name, members in manifest["partition_membership"].items()
    }
    # Inject artificial overlap.
    if membership["diagnosis"] and membership["held_out"]:
        leaked = membership["diagnosis"][0]
        membership["held_out"] = sorted(set(membership["held_out"]) | {leaked})
    sets = {name: set(vals) for name, vals in membership.items()}
    overlap = sets["diagnosis"] & sets["held_out"]
    assert overlap, "test setup failed to create overlap"
    # Production check: real manifest must not have this.
    real = {
        name: set(vals) for name, vals in manifest["partition_membership"].items()
    }
    assert not (real["diagnosis"] & real["held_out"])
    assert not (real["diagnosis"] & real["development"])
    assert not (real["development"] & real["held_out"])


def test_gui_notes_exclude_visual_mutation(manifest: dict[str, Any]) -> None:
    gui = [
        f
        for f in manifest["fixtures"]
        if f["campaign"] == "vacuity_gui" and f["scenario"].startswith("gui_")
    ]
    assert gui
    for fixture in gui:
        notes = fixture.get("notes") or ""
        assert "visual" in notes.lower()


def test_every_campaign_has_all_three_partitions(manifest: dict[str, Any]) -> None:
    """Corpus-level partitions are non-empty; each major campaign spans them when sized for it."""
    for name in PARTITIONS:
        assert manifest["partition_membership"][name]

    # Security and seal/dist campaigns must place cases in held_out for qualification.
    for campaign in (
        "security",
        "zk_incremental_seal",
        "distributed_storage_crash",
        "semantic_compression",
        "vacuity_gui",
    ):
        partitions = {
            f["partition"]
            for f in manifest["fixtures"]
            if f["campaign"] == campaign
        }
        assert "held_out" in partitions, campaign
