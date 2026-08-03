"""Executable contract for ProofTacticianCorpus@1 (FVT-004 / FVT-G020).

Validates that
``ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json`` is a
compact golden corpus that:

* covers the twelve required goal/proof-gap/counterexample scenario families;
* includes solvable, mutated, impossible, ambiguous, unsupported, and
  unavailable variants;
* measures end-goal formalization, proof-gap recovery, proof-chain authority,
  counterexample replay/minimization, and honest failure;
* binds licenses, provenance, and expected authority ceilings;
* never embeds private witnesses, secrets, raw source/stdout, or credentials;
* never labels offline injected expectations as live verification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

import pytest


REPO_ROOT: Final = Path(__file__).resolve().parents[2]
MANIFEST_PATH: Final = (
    REPO_ROOT
    / "ipfs_datasets_py"
    / "tests"
    / "fixtures"
    / "logic"
    / "proof_tactician"
    / "manifest.json"
)
OBJECTIVES_PATH: Final = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "formal_verification_tactician_readiness.objectives.md"
)

INTERFACE: Final = "ProofTacticianCorpus@1"
SCHEMA_VERSION: Final = "proof-tactician-corpus/v1"
CORPUS_ID: Final = "proof-tactician-golden-v1"
GOAL_ID: Final = "FVT-G020"
TASK_ID: Final = "FVT-004"

REQUIRED_FAMILIES: Final = (
    "missing_loop_invariant",
    "callee_contract_frame",
    "lease_safety",
    "fairness_ambiguity",
    "impossible_target_core",
    "smt_model",
    "runtime_trace",
    "protocol_attack",
    "hypertrace",
    "kernel_rejection",
    "bridge_lemma",
    "legal_evidence_routing",
)

REQUIRED_VARIANTS: Final = (
    "solvable",
    "mutated",
    "impossible",
    "ambiguous",
    "unsupported",
    "unavailable",
)

PER_FAMILY_MINIMUM_VARIANTS: Final = (
    "solvable",
    "mutated",
    "unsupported",
)

REQUIRED_MEASUREMENTS: Final = (
    "end_goal_formalization",
    "proof_gap_recovery",
    "proof_chain_authority",
    "counterexample_replay",
    "counterexample_minimization",
    "honest_failure",
)

REQUIRED_MANIFEST_KEYS: Final = (
    "schema_version",
    "interface",
    "corpus_id",
    "corpus_revision",
    "goal_id",
    "task_id",
    "description",
    "license",
    "privacy",
    "non_execution",
    "conflict_policy",
    "authority_vocabulary",
    "variant_vocabulary",
    "family_vocabulary",
    "family_catalog",
    "measurement_vocabulary",
    "disposition_vocabulary",
    "minimization_guarantee_vocabulary",
    "forbidden_public_fields",
    "required_coverage",
    "acceptance",
    "case_ids",
    "cases",
    "coverage_index",
    "evidence_paths",
)

REQUIRED_CASE_KEYS: Final = (
    "case_id",
    "family_id",
    "variant",
    "description",
    "property_kind",
    "logic_family",
    "proof_gap_kind",
    "measures",
    "expected_authority",
    "evidence_authority_ceiling",
    "expected_disposition",
    "expected_outcome_class",
    "minimization_guarantee",
    "live_verification",
    "evidence_class",
    "private_witness_embedded",
    "license_expression",
    "provenance",
    "recipe",
    "forbidden_public_fields",
)

REQUIRED_PROVENANCE_KEYS: Final = (
    "source_class",
    "origin",
    "license_expression",
    "attribution",
    "reviewed",
)

FORBIDDEN_FIELD_MARKERS: Final = (
    "hidden_witness",
    "private_witness",
    "credential",
    "password",
    "api_key",
    "secret",
    "raw_source",
    "stdout",
    "private_channel",
    "token",
)

PRIVATE_VALUE_PATTERNS: Final = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)bearer\s+[a-z0-9._\-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|password|secret)\s*[:=]\s*\S+"),
)

EVIDENCE_AUTHORITIES: Final = frozenset(
    {"independently_checkable", "bounded", "advisory"}
)
MINIMIZATION_GUARANTEES: Final = frozenset(
    {"none", "normalized", "bounded", "locally_minimal", "globally_minimal"}
)
EVIDENCE_CLASSES: Final = frozenset(
    {"synthetic_offline_fixture", "offline", "synthetic"}
)


def _load_manifest() -> dict[str, Any]:
    assert MANIFEST_PATH.is_file(), f"missing corpus manifest: {MANIFEST_PATH}"
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return _load_manifest()


@pytest.fixture(scope="module")
def cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw = manifest["cases"]
    assert isinstance(raw, list) and raw, "cases must be a non-empty list"
    for case in raw:
        assert isinstance(case, dict)
    return raw  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Schema, goal binding, and vocabulary
# ---------------------------------------------------------------------------


def test_manifest_schema_interface_and_goal_binding(manifest: dict[str, Any]) -> None:
    for key in REQUIRED_MANIFEST_KEYS:
        assert key in manifest, f"manifest missing required key {key!r}"

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["interface"] == INTERFACE
    assert manifest["corpus_id"] == CORPUS_ID
    assert manifest["goal_id"] == GOAL_ID
    assert manifest["task_id"] == TASK_ID
    assert isinstance(manifest["corpus_revision"], str) and manifest["corpus_revision"]
    assert isinstance(manifest["description"], str) and len(manifest["description"]) > 40

    assert OBJECTIVES_PATH.is_file()
    objectives = OBJECTIVES_PATH.read_text(encoding="utf-8")
    assert f"## {GOAL_ID} " in objectives
    assert "goal/proof-gap/counterexample golden corpus" in objectives
    assert "ProofTacticianCorpus@1" in objectives


def test_vocabularies_are_closed_and_cover_required_sets(
    manifest: dict[str, Any],
) -> None:
    assert list(manifest["family_vocabulary"]) == list(REQUIRED_FAMILIES)
    assert set(manifest["variant_vocabulary"]) == set(REQUIRED_VARIANTS)
    assert set(manifest["measurement_vocabulary"]) == set(REQUIRED_MEASUREMENTS)

    assert set(REQUIRED_FAMILIES) <= set(manifest["required_coverage"]["families"])
    assert set(REQUIRED_VARIANTS) <= set(manifest["required_coverage"]["variants"])
    assert set(REQUIRED_MEASUREMENTS) <= set(
        manifest["required_coverage"]["measurements"]
    )
    assert set(PER_FAMILY_MINIMUM_VARIANTS) <= set(
        manifest["required_coverage"]["per_family_minimum_variants"]
    )

    for authority in manifest["authority_vocabulary"]:
        assert isinstance(authority, str) and authority

    for guarantee in manifest["minimization_guarantee_vocabulary"]:
        assert guarantee in MINIMIZATION_GUARANTEES

    for ceiling in manifest.get("evidence_authority_vocabulary", []):
        assert ceiling in EVIDENCE_AUTHORITIES


def test_family_catalog_matches_required_families(manifest: dict[str, Any]) -> None:
    catalog = manifest["family_catalog"]
    assert isinstance(catalog, list)
    catalog_ids = [row["family_id"] for row in catalog]
    assert catalog_ids == list(REQUIRED_FAMILIES)
    for row in catalog:
        assert set(PER_FAMILY_MINIMUM_VARIANTS) <= set(row["required_variants"])
        assert isinstance(row["title"], str) and row["title"]
        assert isinstance(row["proof_gap_kind"], str) and row["proof_gap_kind"]
        assert isinstance(row["default_authority"], str) and row["default_authority"]
        assert row["default_authority"] in set(manifest["authority_vocabulary"])


# ---------------------------------------------------------------------------
# License, privacy, non-execution, acceptance
# ---------------------------------------------------------------------------


def test_license_provenance_and_privacy_forbid_private_data(
    manifest: dict[str, Any],
) -> None:
    license_meta = manifest["license"]
    privacy = manifest["privacy"]

    assert license_meta["license_expression"]
    assert license_meta["source_class"] == "synthetic_fixture"
    assert license_meta["license_url"]

    assert privacy["privacy_class"] == "public_synthetic"
    assert privacy["contains_pii"] is False
    assert privacy["contains_secrets"] is False
    assert privacy["private_witnesses_allowed"] is False
    assert privacy["private_data_allowed"] is False
    assert privacy["raw_source_allowed"] is False
    assert privacy["raw_stdout_allowed"] is False
    assert privacy["credentials_allowed"] is False
    assert privacy["tokens_allowed"] is False
    assert privacy["hidden_channels_allowed"] is False
    assert privacy["network_required"] is False

    non_execution = manifest["non_execution"]
    assert non_execution["live_verification"] is False
    assert non_execution["requires_network"] is False
    assert non_execution["requires_optional_solver"] is False
    assert non_execution["labels_injected_results_as_live"] is False
    assert non_execution["installs_or_fetches_tools"] is False
    assert non_execution["side_effects"] is False

    forbidden = set(manifest["forbidden_public_fields"])
    for marker in FORBIDDEN_FIELD_MARKERS:
        assert marker in forbidden or marker.replace("_", "") in {
            item.replace("_", "") for item in forbidden
        }


def test_acceptance_flags_match_goal_criteria(manifest: dict[str, Any]) -> None:
    acceptance = manifest["acceptance"]
    required_true = (
        "covers_twelve_required_scenario_families",
        "variants_include_solvable_mutated_impossible_ambiguous_unsupported_unavailable",
        "measures_end_goal_formalization",
        "measures_proof_gap_recovery",
        "measures_proof_chain_authority",
        "measures_counterexample_replay_and_minimization",
        "measures_honest_failure",
        "binds_licenses_and_provenance",
        "binds_expected_authority",
        "no_private_witnesses_embedded",
        "not_labeled_as_live_verification",
    )
    for key in required_true:
        assert acceptance[key] is True, key

    conflict = manifest["conflict_policy"].lower()
    assert "live verification" in conflict
    assert "fixture" in conflict


def test_evidence_paths_bind_declared_outputs(manifest: dict[str, Any]) -> None:
    paths = set(manifest["evidence_paths"])
    assert (
        "ipfs_datasets_py/tests/fixtures/logic/proof_tactician/manifest.json" in paths
    )
    assert "test/api/test_formal_verification_tactician_corpus_contract.py" in paths
    for relative in paths:
        assert (REPO_ROOT / relative).is_file(), relative


# ---------------------------------------------------------------------------
# Case index integrity and required coverage
# ---------------------------------------------------------------------------


def test_case_index_is_consistent(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    case_ids = [case["case_id"] for case in cases]
    assert manifest["case_ids"] == case_ids
    assert len(case_ids) == len(set(case_ids))
    assert len(case_ids) >= len(REQUIRED_FAMILIES) * len(PER_FAMILY_MINIMUM_VARIANTS)

    coverage = manifest["coverage_index"]
    assert coverage["case_count"] == len(cases)
    assert set(coverage["families_present"]) == set(REQUIRED_FAMILIES)
    assert set(REQUIRED_VARIANTS) <= set(coverage["variants_present"])
    assert set(REQUIRED_MEASUREMENTS) <= set(coverage["measurements_present"])

    variants_by_family = coverage["variants_by_family"]
    for family_id in REQUIRED_FAMILIES:
        assert family_id in variants_by_family
        assert set(PER_FAMILY_MINIMUM_VARIANTS) <= set(variants_by_family[family_id])


def test_required_families_variants_and_measurements_are_covered(
    cases: list[dict[str, Any]],
) -> None:
    families = {case["family_id"] for case in cases}
    variants = {case["variant"] for case in cases}
    measures = {measure for case in cases for measure in case["measures"]}

    assert set(REQUIRED_FAMILIES) <= families
    assert set(REQUIRED_VARIANTS) <= variants
    assert set(REQUIRED_MEASUREMENTS) <= measures

    by_family: dict[str, set[str]] = {family: set() for family in REQUIRED_FAMILIES}
    for case in cases:
        if case["family_id"] in by_family:
            by_family[case["family_id"]].add(case["variant"])

    for family_id, present in by_family.items():
        missing = set(PER_FAMILY_MINIMUM_VARIANTS) - present
        assert not missing, f"{family_id} missing variants {sorted(missing)}"

    # Named families that specifically exercise ambiguity / impossibility.
    assert any(
        case["family_id"] == "fairness_ambiguity" and case["variant"] == "ambiguous"
        for case in cases
    )
    assert any(
        case["family_id"] == "impossible_target_core"
        and case["variant"] == "impossible"
        for case in cases
    )


def test_each_case_binds_authority_license_and_measures(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    authorities = set(manifest["authority_vocabulary"])
    dispositions = set(manifest["disposition_vocabulary"])
    guarantees = set(manifest["minimization_guarantee_vocabulary"])
    measurements = set(manifest["measurement_vocabulary"])

    for case in cases:
        for key in REQUIRED_CASE_KEYS:
            assert key in case, f"{case.get('case_id')}: missing {key}"

        assert case["case_id"] == f"{case['family_id']}.{case['variant']}"
        assert case["family_id"] in set(REQUIRED_FAMILIES) or case[
            "family_id"
        ] in set(manifest["family_vocabulary"])
        assert case["variant"] in set(REQUIRED_VARIANTS)
        assert case["expected_authority"] in authorities
        assert case["evidence_authority_ceiling"] in EVIDENCE_AUTHORITIES
        assert case["expected_disposition"] in dispositions
        assert case["minimization_guarantee"] in guarantees
        assert case["live_verification"] is False
        assert case["evidence_class"] in EVIDENCE_CLASSES
        assert case["private_witness_embedded"] is False
        assert case["license_expression"]

        provenance = case["provenance"]
        for key in REQUIRED_PROVENANCE_KEYS:
            assert key in provenance, f"{case['case_id']}: provenance missing {key}"
        assert provenance["source_class"] == "synthetic_fixture"
        assert provenance["license_expression"]
        assert provenance["reviewed"] is True

        assert isinstance(case["measures"], list) and case["measures"]
        for measure in case["measures"]:
            assert measure in measurements

        recipe = case["recipe"]
        assert isinstance(recipe, dict)
        assert recipe.get("end_goal_prose")
        assert recipe.get("proof_gap_kind") == case["proof_gap_kind"]


def test_variant_specific_recipe_hooks(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        recipe = case["recipe"]
        variant = case["variant"]
        if variant == "mutated":
            mutation = recipe["mutation"]
            assert mutation and mutation["semantic_change"] is True
            assert mutation["mutation_id"]
        if variant == "unsupported":
            assert recipe["unsupported_fragment"]
            assert recipe["unsupported_fragment"]["fragment_id"]
        if variant == "unavailable":
            tool = recipe["unavailable_tool"]
            assert tool and tool["tool_id"]
            assert "unavailable" in tool["policy"].lower() or "absence" in tool[
                "policy"
            ].lower()
        if variant == "ambiguous":
            ambiguity = recipe["ambiguity"]
            assert ambiguity and ambiguity["requires_material_selection"] is True
            assert len(ambiguity["alternatives"]) >= 2
        if variant == "impossible":
            hint = recipe["inconsistency_hint"]
            assert hint and hint["must_not_prove"] is True


def test_counterexample_public_contract_never_embeds_private_witnesses(
    cases: list[dict[str, Any]],
) -> None:
    for case in cases:
        contract = case.get("counterexample_public_contract")
        if contract is None:
            continue
        assert contract["private_witness_embedded"] is False
        assert contract["raw_artifact_policy"] == "digest_reference_only"
        assert "kind" in contract["public_fields"]
        assert "authority" in contract["public_fields"]
        assert contract["minimization_guarantee"] in MINIMIZATION_GUARANTEES
        for field in contract["public_fields"]:
            assert field not in FORBIDDEN_FIELD_MARKERS


# Policy / inventory / acceptance keys may *name* forbidden concepts without
# embedding payloads under those names.
_ALLOWED_POLICY_KEY_FRAGMENTS: Final = (
    "private_witness",
    "private_data",
    "raw_source",
    "raw_stdout",
    "credential",
    "token",
    "secret",
    "hidden_channel",
    "forbidden_public",
    "raw_artifact",
)


def _is_policy_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _ALLOWED_POLICY_KEY_FRAGMENTS)


def test_no_private_witness_or_secret_payloads(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    # Disallow payload containers that would hold private material under
    # non-policy names (e.g. a free-form "witness_blob" object).
    for key in _walk_keys(manifest):
        lowered = key.lower()
        if _is_policy_key(key):
            continue
        if key in FORBIDDEN_FIELD_MARKERS:
            continue
        assert "witness_blob" not in lowered
        assert "raw_stdout_body" not in lowered
        assert "private_key_pem" not in lowered

    # Values must not look like PEM keys, bearer tokens, or secret assignments.
    inventory = set(manifest["forbidden_public_fields"]) | set(FORBIDDEN_FIELD_MARKERS)
    for text in _walk_strings(manifest):
        if text in inventory:
            continue
        for pattern in PRIVATE_VALUE_PATTERNS:
            assert pattern.search(text) is None, text[:120]

    for case in cases:
        assert case["private_witness_embedded"] is False
        assert case["live_verification"] is False


def test_legal_evidence_routing_delegates_without_private_documents(
    cases: list[dict[str, Any]],
) -> None:
    legal_cases = [
        case for case in cases if case["family_id"] == "legal_evidence_routing"
    ]
    assert legal_cases
    for case in legal_cases:
        blob = json.dumps(case, sort_keys=True).lower()
        assert "legal" in blob
        assert "private tenant" not in blob
        assert "ssn" not in blob
        assert case["private_witness_embedded"] is False
        assert case["expected_authority"] in {
            "authorization",
            "candidate",
            "attestation",
        }


def test_honest_failure_variants_do_not_claim_proof_success(
    cases: list[dict[str, Any]],
) -> None:
    for case in cases:
        if case["variant"] in {"impossible", "unsupported", "unavailable", "ambiguous"}:
            disposition = case["expected_disposition"].lower()
            outcome = case["expected_outcome_class"].lower()
            assert "prove" not in disposition or "not" in disposition or "must_not" in json.dumps(
                case["recipe"]
            ).lower()
            assert "success" not in outcome or "honest" in outcome
            assert "honest_failure" in case["measures"] or case["variant"] == "ambiguous"
            assert case["evidence_authority_ceiling"] in {"advisory", "bounded"}
            assert case["live_verification"] is False


def test_mutated_cases_cannot_reuse_elevated_authority(
    cases: list[dict[str, Any]],
) -> None:
    for case in cases:
        if case["variant"] != "mutated":
            continue
        assert case["expected_authority"] == "candidate"
        assert case["evidence_authority_ceiling"] == "advisory"
        assert case["recipe"]["mutation"]["semantic_change"] is True


def test_coverage_index_matches_live_case_scan(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
) -> None:
    index = manifest["coverage_index"]
    assert index["case_count"] == len(cases)
    assert set(index["families_present"]) == {case["family_id"] for case in cases}
    assert set(index["variants_present"]) == {case["variant"] for case in cases}
    assert set(index["measurements_present"]) == {
        measure for case in cases for measure in case["measures"]
    }
    assert set(index["authorities_present"]) == {
        case["expected_authority"] for case in cases
    }

    rebuilt: dict[str, list[str]] = {}
    for case in cases:
        rebuilt.setdefault(case["family_id"], []).append(case["variant"])
    for family_id, variants in rebuilt.items():
        assert sorted(variants) == list(index["variants_by_family"][family_id])
