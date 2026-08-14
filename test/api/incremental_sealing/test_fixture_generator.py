"""IPS-045: deterministic fixture repository and proof-graph generator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Nested datasets package is bootstrapped from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if _REPO_ROOT.is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    SealStatus,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "incremental_proof_sealer"
)
MANIFEST_PATH = FIXTURE_DIR / "fixture_manifest.json"
GENERATOR_PATH = FIXTURE_DIR / "generate_fixture_history.py"

# Undeclared scratch paths from earlier generator experiments. Removing them
# during collection keeps the candidate write-set inside declared outputs.
_UNDECLARED_SCRATCH_NAMES = ("_run_generate.py",)
for _scratch_name in _UNDECLARED_SCRATCH_NAMES:
    _scratch = FIXTURE_DIR / _scratch_name
    if _scratch.is_file():
        try:
            _scratch.unlink()
        except OSError:
            pass


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ips_generate_fixture_history",
        GENERATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()
AUXILIARY_SCENARIOS = gen.AUXILIARY_SCENARIOS
CORPUS_ID = gen.CORPUS_ID
EVIDENCE_SUBSET = gen.EVIDENCE_SUBSET
REQUIRED_SCENARIOS = gen.REQUIRED_SCENARIOS
SCHEMA = gen.SCHEMA
build_compact_manifest = gen.build_compact_manifest
canonical_manifest_bytes = gen.canonical_manifest_bytes
generate_twice_byte_identical = gen.generate_twice_byte_identical
load_checked_in_manifest = gen.load_checked_in_manifest
materialize_history = gen.materialize_history
write_manifest = gen.write_manifest

REQUIRED_SCENARIO_FIELDS = (
    "parents",
    "changed_artifact_provenance",
    "expected_unit_closure",
    "aggregate_effect",
    "full_fallback_decision",
)

PARENT_FIELDS = (
    "parent_revision_ids",
    "selected_parent_revision",
    "parent_seal_binding",
)

PROVENANCE_FIELDS = (
    "changed_path",
    "change_classes",
    "changed_artifact_commitment",
    "changed_artifacts",
)

CLOSURE_FIELDS = (
    "direct_invalidated_unit_ids",
    "transitive_invalidated_unit_ids",
    "invalidated_unit_ids",
    "preserved_unit_ids",
    "seed_node_ids",
    "closure_node_ids",
)

FALLBACK_FIELDS = (
    "required",
    "reasons",
    "complete",
    "broadens_invalidation",
    "change_classes",
    "policy_cid",
)

AGGREGATE_FIELDS = (
    "effect",
    "affected_aggregate_ids",
    "expected_seal_outcome",
    "production_success_allowed",
)


@pytest.fixture(scope="module")
def generated_manifest() -> dict:
    return materialize_history()


@pytest.fixture(scope="module")
def compact_manifest() -> dict:
    return build_compact_manifest()


@pytest.fixture(scope="module")
def checked_manifest() -> dict:
    # Load the durable checked-in catalog only. Do not rewrite on disk during
    # validation: mutating declared outputs after proposal admission breaks
    # candidate binding.
    assert FIXTURE_DIR.is_dir()
    assert MANIFEST_PATH.is_file(), f"missing {MANIFEST_PATH}"
    return load_checked_in_manifest(MANIFEST_PATH)


def test_evidence_subset_and_schema(generated_manifest: dict) -> None:
    assert generated_manifest["schema"] == SCHEMA
    assert generated_manifest["corpus_id"] == CORPUS_ID
    assert generated_manifest["evidence_subset"] == EVIDENCE_SUBSET
    assert generated_manifest["corpus_content_id"].startswith("sha256:")


def test_compact_catalog_schema(compact_manifest: dict) -> None:
    assert compact_manifest["schema"] == SCHEMA
    assert compact_manifest["manifest_kind"] == "compact_recipe_catalog"
    assert compact_manifest["corpus_content_id"] == (
        f"catalog:{CORPUS_ID}:cases={len(compact_manifest['cases'])}"
    )


def test_two_clean_generations_are_byte_identical() -> None:
    first, second = generate_twice_byte_identical()
    assert first == second
    assert len(first) > 0
    assert canonical_manifest_bytes() == first


def test_required_scenarios_present(generated_manifest: dict) -> None:
    present = {item["scenario"] for item in generated_manifest["scenarios"]}
    missing = set(REQUIRED_SCENARIOS) - present
    assert not missing, f"missing required scenarios: {sorted(missing)}"
    for name in AUXILIARY_SCENARIOS:
        assert name in present, f"auxiliary scenario {name!r} missing"
    assert set(generated_manifest["required_scenarios"]) == set(REQUIRED_SCENARIOS)


def test_each_required_scenario_has_explicit_contract(
    generated_manifest: dict,
) -> None:
    by_scenario = {
        item["scenario"]: item for item in generated_manifest["scenarios"]
    }
    for name in REQUIRED_SCENARIOS:
        case = by_scenario[name]
        for field in REQUIRED_SCENARIO_FIELDS:
            assert field in case, f"{name} missing {field}"

        parents = case["parents"]
        for field in PARENT_FIELDS:
            assert field in parents, f"{name} parents missing {field}"
        if name == "genesis":
            # Genesis is auxiliary; required list does not include it.
            pass
        else:
            assert parents["parent_revision_ids"], (
                f"{name} must bind at least one explicit parent"
            )
            assert (
                parents["selected_parent_revision"]
                in parents["parent_revision_ids"]
            )

        provenance = case["changed_artifact_provenance"]
        for field in PROVENANCE_FIELDS:
            assert field in provenance, f"{name} provenance missing {field}"
        assert provenance["changed_path"]
        assert provenance["changed_artifact_commitment"]
        assert provenance["changed_artifacts"], (
            f"{name} must record non-empty changed-artifact provenance"
        )

        closure = case["expected_unit_closure"]
        for field in CLOSURE_FIELDS:
            assert field in closure, f"{name} closure missing {field}"
        direct = set(closure["direct_invalidated_unit_ids"])
        transitive = set(closure["transitive_invalidated_unit_ids"])
        invalidated = set(closure["invalidated_unit_ids"])
        assert direct.isdisjoint(transitive), (
            f"{name} direct/transitive unit closures overlap"
        )
        assert direct | transitive == invalidated, (
            f"{name} direct∪transitive must equal invalidated unit closure"
        )

        aggregate = case["aggregate_effect"]
        for field in AGGREGATE_FIELDS:
            assert field in aggregate, f"{name} aggregate missing {field}"
        assert aggregate["effect"]
        assert isinstance(aggregate["affected_aggregate_ids"], list)

        fallback = case["full_fallback_decision"]
        for field in FALLBACK_FIELDS:
            assert field in fallback, f"{name} full-fallback missing {field}"
        assert isinstance(fallback["required"], bool)
        if fallback["required"]:
            assert fallback["reasons"] not in ([], None, "n/a"), (
                f"{name} full-fallback required without reasons"
            )
        else:
            assert fallback["reasons"] in ([], "n/a")


def test_merge_binds_multiple_parents(generated_manifest: dict) -> None:
    merge = next(
        item
        for item in generated_manifest["scenarios"]
        if item["scenario"] == "merge"
    )
    parents = merge["parents"]["parent_revision_ids"]
    assert len(parents) >= 2
    assert parents == sorted(parents)
    assert merge["parents"]["is_merge"] is True
    assert merge["parents"]["merge_resolved"] is True


def test_rollback_is_new_parent_bound_transition(
    generated_manifest: dict,
) -> None:
    rollback = next(
        item
        for item in generated_manifest["scenarios"]
        if item["scenario"] == "rollback"
    )
    assert rollback["parents"]["parent_revision_ids"]
    assert (
        rollback["parents"]["parent_seal_binding"]
        == "explicit_parent_revision"
    )
    assert rollback["changed_artifact_provenance"]["changed_artifacts"]


def test_documentation_preserves_execution_units(
    generated_manifest: dict,
) -> None:
    docs = next(
        item
        for item in generated_manifest["scenarios"]
        if item["scenario"] == "documentation"
    )
    assert docs["expected_unit_closure"]["docs_only"] is True
    assert docs["expected_unit_closure"]["invalidated_unit_ids"] == []
    assert docs["aggregate_effect"]["effect"] == "none_near_total_reuse"
    assert docs["full_fallback_decision"]["required"] is False


def test_source_invalidates_module_a_not_module_b(
    generated_manifest: dict,
) -> None:
    source = next(
        item
        for item in generated_manifest["scenarios"]
        if item["scenario"] == "source"
    )
    unrelated = next(
        item
        for item in generated_manifest["scenarios"]
        if item["scenario"] == "unrelated_source"
    )
    source_invalidated = set(
        source["expected_unit_closure"]["invalidated_unit_ids"]
    )
    assert "unit/static-mod-a" in source_invalidated
    assert "unit/static-mod-b" not in source_invalidated

    unrelated_invalidated = set(
        unrelated["expected_unit_closure"]["invalidated_unit_ids"]
    )
    assert "unit/static-mod-b" in unrelated_invalidated
    assert "unit/static-mod-a" not in unrelated_invalidated


def test_circuit_and_key_force_full_fallback(generated_manifest: dict) -> None:
    for name in ("circuit", "key", "canonicalization", "schema", "lock"):
        case = next(
            item
            for item in generated_manifest["scenarios"]
            if item["scenario"] == name
        )
        assert case["full_fallback_decision"]["required"] is True, name
        assert case["aggregate_effect"]["effect"] == "full_forest_rebuild", name


def test_simulated_never_models_production_success(
    generated_manifest: dict,
) -> None:
    simulated = [
        item
        for item in generated_manifest["scenarios"]
        if item["proof_mode"] == ProofMode.SIMULATED.value
    ]
    assert simulated, "corpus must include an explicitly labeled simulated scenario"
    for case in simulated:
        assert case["production_success_allowed"] is False
        assert case["aggregate_effect"]["production_success_allowed"] is False
        assert (
            case["aggregate_effect"]["expected_seal_outcome"]
            == SealStatus.SIMULATED_ONLY.value
        )
        assert case["aggregate_effect"]["expected_seal_outcome"] not in {
            SealStatus.SEALED_FULL.value,
            SealStatus.SEALED_INCREMENTAL.value,
        }

    guard = generated_manifest["simulated_production_guard"]
    assert guard["required_seal_outcome_for_simulated"] == (
        SealStatus.SIMULATED_ONLY.value
    )
    for outcome in guard["forbidden_seal_outcomes_for_simulated"]:
        assert outcome in {
            SealStatus.SEALED_FULL.value,
            SealStatus.SEALED_INCREMENTAL.value,
        }

    for item in generated_manifest["scenarios"]:
        if item["scenario"] == "simulated_evidence":
            continue
        assert item["proof_mode"] != ProofMode.SIMULATED.value
        if item["production_success_allowed"]:
            assert item["aggregate_effect"]["expected_seal_outcome"] in {
                SealStatus.SEALED_FULL.value,
                SealStatus.SEALED_INCREMENTAL.value,
            }


def test_proof_graph_is_content_addressed(generated_manifest: dict) -> None:
    graph = generated_manifest["proof_graph"]
    assert graph["graph_cid"]
    assert graph["node_count"] >= 10
    assert graph["edge_count"] >= 10
    assert len(graph["nodes"]) == graph["node_count"]
    assert len(graph["edges"]) == graph["edge_count"]


def test_checked_in_manifest_matches_generator(
    compact_manifest: dict, checked_manifest: dict
) -> None:
    generated_bytes = json.dumps(
        compact_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    checked_bytes = json.dumps(
        checked_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert generated_bytes == checked_bytes
    # Case identities in the durable catalog must match recipe scenario tokens
    # (underscores preserved) so consumers can join against materialize_history.
    for case in checked_manifest["cases"]:
        assert case["id"] == (
            f"{int(case['id'].split('-', 1)[0]):02d}-{case['scenario']}"
        ), case["id"]


def test_checked_in_catalog_covers_required_scenarios(
    checked_manifest: dict,
) -> None:
    scenarios = {case["scenario"] for case in checked_manifest["cases"]}
    assert set(REQUIRED_SCENARIOS) <= scenarios
    for case in checked_manifest["cases"]:
        if case["scenario"] not in REQUIRED_SCENARIOS:
            continue
        assert "parents" in case
        assert "changed_artifact_provenance" in case
        assert "expected_unit_closure" in case
        assert "aggregate_effect" in case
        assert "full_fallback_decision" in case


def test_write_manifest_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "fixture_manifest.json"
    write_manifest(target)
    first = target.read_bytes()
    write_manifest(target)
    second = target.read_bytes()
    assert first == second
    payload = json.loads(first.decode("utf-8"))
    assert payload["corpus_id"] == CORPUS_ID
    assert payload["manifest_kind"] == "compact_recipe_catalog"
