"""Schema and safety-boundary checks for the change-propagation fixture corpus.

RPR-024: adversarial transitive-change fixtures. Expectations never grant
vector, knowledge-graph, or LLM semantic authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest


FIXTURE_ROOT = (
    Path(__file__).parents[1] / "fixtures" / "agent_supervisor" / "change_propagation"
)
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-fixture-manifest@1"
)
CORPUS_ID = "change-propagation-adversarial-v1"

ARTIFACT_ROLES = frozenset(
    {"delta", "consumers", "graph", "value_sources", "plan", "proof"}
)
EXPECTATION_SOURCES = frozenset({"reviewed_spec", "test"})

# Closed catalogue required by RPR-024 acceptance.
REQUIRED_SCENARIOS = frozenset(
    {
        "two_to_three_argument_callers",
        "unique_in_scope_value",
        "same_typed_wrong_information",
        "branch_local_value",
        "nullable_value",
        "parameter_threading",
        "config_di_factory_construction",
        "schema_serializer_generated_client",
        "new_class_method_data_structure",
        "stateful_service",
        "async_error_effect_auth_resource_lifetime_drift",
        "dependency_cycle_scc",
        "reflection_plugin_registry_ffi_frontier",
        "stale_graph_vector_proof",
        "poisoned_retrieval",
        "read_only_cross_repository",
        "partial_transaction",
        "llm_scope_escape",
        "weakened_test",
        "second_order_breaking_delta",
    }
)

# Fail-closed adversarial scenarios: must never automate writes.
FAIL_CLOSED_SCENARIOS = frozenset(
    {
        "same_typed_wrong_information",
        "branch_local_value",
        "nullable_value",
        "async_error_effect_auth_resource_lifetime_drift",
        "reflection_plugin_registry_ffi_frontier",
        "stale_graph_vector_proof",
        "poisoned_retrieval",
        "read_only_cross_repository",
        "partial_transaction",
        "llm_scope_escape",
        "weakened_test",
    }
)

# Positive analytical paths may write only after plan admission / fixed point.
ADMITTABLE_SCENARIOS = frozenset(
    {
        "two_to_three_argument_callers",
        "unique_in_scope_value",
        "parameter_threading",
        "config_di_factory_construction",
        "schema_serializer_generated_client",
        "new_class_method_data_structure",
        "stateful_service",
        "dependency_cycle_scc",
        "second_order_breaking_delta",
    }
)

REQUIRED_CALLER_KINDS = frozenset({"direct", "aliased", "wrapped", "method"})

PLAN_ADMISSION_VALUES = frozenset(
    {
        "admit_after_proof",
        "admit_scc_transaction_only",
        "require_fixed_point_iteration",
        "abstain",
        "rollback",
    }
)
AUTOMATED_WRITE_VALUES = frozenset(
    {"never", "only_after_plan_admission", "only_after_fixed_point"}
)
COMPLETION_VALUES = frozenset(
    {
        "success",
        "fail_closed",
        "rollback",
        "incomplete_until_second_order_discharged",
    }
)


def _canonical_content_id(content: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ChangePropagationFixture:
    fixture_id: str
    scenario: str
    expected: Mapping[str, Any]
    authority: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChangePropagationFixture":
        return cls(
            fixture_id=str(value["id"]),
            scenario=str(value["scenario"]),
            expected=value["expected"],
            authority=value["authority"],
            artifacts=value["artifacts"],
        )


@dataclass(frozen=True)
class ChangePropagationFixtureManifest:
    schema: str
    corpus_id: str
    fixtures: tuple[ChangePropagationFixture, ...]

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> "ChangePropagationFixtureManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            schema=str(payload["schema"]),
            corpus_id=str(payload["corpus_id"]),
            fixtures=tuple(
                ChangePropagationFixture.from_dict(case) for case in payload["cases"]
            ),
        )


@pytest.fixture(scope="module")
def manifest() -> ChangePropagationFixtureManifest:
    return ChangePropagationFixtureManifest.load()


def test_manifest_has_a_content_addressed_recipe_for_every_required_scenario(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    assert manifest.schema == MANIFEST_SCHEMA
    assert manifest.corpus_id == CORPUS_ID
    assert {fixture.scenario for fixture in manifest.fixtures} == REQUIRED_SCENARIOS
    assert len({fixture.fixture_id for fixture in manifest.fixtures}) == len(
        manifest.fixtures
    )
    assert len(manifest.fixtures) == len(REQUIRED_SCENARIOS)

    for fixture in manifest.fixtures:
        assert set(fixture.artifacts) == ARTIFACT_ROLES
        expected = fixture.expected
        assert expected["impact_disposition"]
        assert expected["value_mapping"]
        assert expected["plan_admission"] in PLAN_ADMISSION_VALUES
        assert expected["automated_write"] in AUTOMATED_WRITE_VALUES
        assert expected["fixed_point"]
        assert expected["completion"] in COMPLETION_VALUES
        assert expected["reason_codes"]
        for role, artifact in fixture.artifacts.items():
            content = artifact["content"]
            assert artifact["content_id"] == _canonical_content_id(content), (
                fixture.fixture_id,
                role,
            )
            assert isinstance(content["path"], str) and not content["path"].startswith(
                "/"
            )


def test_expectations_never_grant_vector_kg_or_llm_semantic_authority(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    forbidden_sources = {
        "vector",
        "vector_score",
        "knowledge_graph",
        "kg",
        "llm",
        "llm_semantic",
        "implementation_observation",
    }
    for fixture in manifest.fixtures:
        authority = fixture.authority
        sources = set(authority["expectation_sources"])
        assert sources <= EXPECTATION_SOURCES
        assert sources
        assert authority["implementation_observation_authoritative"] is False
        assert authority["vector_score_authoritative"] is False
        assert authority["knowledge_graph_authoritative"] is False
        assert authority["llm_semantic_authoritative"] is False
        assert authority["requires_independent_proof"] is True
        assert sources.isdisjoint(forbidden_sources)

        value_sources = fixture.artifacts["value_sources"]["content"]
        assert value_sources["semantic_authority"] is False

        # Even when retrieval scores are present as data, they are not authority.
        if "vector_score" in value_sources:
            assert authority["vector_score_authoritative"] is False
        if "llm_semantic_score" in value_sources:
            assert authority["llm_semantic_authoritative"] is False


def test_two_to_three_argument_change_covers_direct_aliased_wrapped_method_callers(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}
    case = fixtures["two_to_three_argument_callers"]
    assert set(case.expected["caller_kinds"]) == REQUIRED_CALLER_KINDS

    consumers = case.artifacts["consumers"]["content"]
    kinds = {entry["kind"] for entry in consumers["resolved"]}
    assert kinds == REQUIRED_CALLER_KINDS
    assert consumers["obligations"] == len(consumers["resolved"]) == 4
    assert consumers["one_compatible_cannot_discharge_others"] is True

    delta = case.artifacts["delta"]["content"]
    assert delta["kind"] == "parameter_add"
    assert "process(left: A, right: B)" in delta["before"]
    assert "context: C" in delta["after"]


def test_fail_closed_adversarial_cases_never_automate_writes(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}
    for scenario in FAIL_CLOSED_SCENARIOS:
        fixture = fixtures[scenario]
        assert fixture.expected["plan_admission"] in {"abstain", "rollback"}
        assert fixture.expected["automated_write"] == "never"
        assert fixture.expected["completion"] in {"fail_closed", "rollback"}


def test_admissible_cases_require_proof_before_write(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}
    for scenario in ADMITTABLE_SCENARIOS:
        fixture = fixtures[scenario]
        assert fixture.expected["plan_admission"] in {
            "admit_after_proof",
            "admit_scc_transaction_only",
            "require_fixed_point_iteration",
        }
        assert fixture.expected["automated_write"] in {
            "only_after_plan_admission",
            "only_after_fixed_point",
        }
        assert fixture.authority["requires_independent_proof"] is True


def test_unique_in_scope_value_and_same_typed_wrong_information_are_distinguished(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    unique = fixtures["unique_in_scope_value"]
    assert unique.expected["value_mapping"] == "unique_proved"
    assert unique.expected["plan_admission"] == "admit_after_proof"
    candidates = unique.artifacts["value_sources"]["content"]["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["proved"] is True
    assert candidates[0]["unique"] is True

    wrong = fixtures["same_typed_wrong_information"]
    assert wrong.expected["value_mapping"] == "wrong_refuted"
    assert wrong.expected["plan_admission"] == "abstain"
    wrong_candidates = wrong.artifacts["value_sources"]["content"]["candidates"]
    assert wrong_candidates[0]["same_type"] is True
    assert wrong_candidates[0]["refuted"] is True


def test_branch_local_and_nullable_values_fail_closed(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    branch = fixtures["branch_local_value"]
    assert branch.expected["value_mapping"] == "path_incomplete"
    assert (
        branch.artifacts["value_sources"]["content"]["candidates"][0][
            "available_on_all_paths"
        ]
        is False
    )

    nullable = fixtures["nullable_value"]
    assert nullable.expected["value_mapping"] == "nullability_mismatch"
    candidate = nullable.artifacts["value_sources"]["content"]["candidates"][0]
    assert candidate["nullability"] == "nullable"
    assert candidate["required"] == "non_null"


def test_parameter_threading_and_construction_paths_are_represented(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    threading = fixtures["parameter_threading"]
    assert threading.expected["value_mapping"] == "thread_upstream"
    assert threading.artifacts["graph"]["content"]["acyclic"] is True
    assert threading.artifacts["plan"]["content"]["transform"] == (
        "thread_parameter_through_chain"
    )

    construction = fixtures["config_di_factory_construction"]
    kinds = {
        entry["kind"]
        for entry in construction.artifacts["consumers"]["content"]["resolved"]
    }
    assert kinds == {"factory", "di", "config"}

    schema = fixtures["schema_serializer_generated_client"]
    kinds = {
        entry["kind"] for entry in schema.artifacts["consumers"]["content"]["resolved"]
    }
    assert "serializer" in kinds
    assert "generated_client" in kinds


def test_new_type_and_stateful_service_require_behavior_contracts(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    new_type = fixtures["new_class_method_data_structure"]
    assert new_type.expected["value_mapping"] == "require_behavior_contract"
    contract = new_type.artifacts["value_sources"]["content"]["behavior_contract"]
    assert contract["source"] == "reviewed_spec"
    assert "allow" in contract["methods"]

    stateful = fixtures["stateful_service"]
    assert stateful.expected["value_mapping"] == "state_transition_proved"
    assert "Active -> Suspended" in stateful.artifacts["delta"]["content"]["new_transition"]


def test_scc_partial_transaction_and_second_order_fixed_point_semantics(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    scc = fixtures["dependency_cycle_scc"]
    assert scc.expected["impact_disposition"] == "scc_grouped"
    assert scc.expected["plan_admission"] == "admit_scc_transaction_only"
    assert scc.artifacts["plan"]["content"]["partial_allowed"] is False
    assert scc.artifacts["graph"]["content"]["sccs"]

    partial = fixtures["partial_transaction"]
    assert partial.expected["plan_admission"] == "rollback"
    assert partial.expected["completion"] == "rollback"
    assert partial.artifacts["plan"]["content"]["partial_failure"] is True

    second = fixtures["second_order_breaking_delta"]
    assert second.expected["impact_disposition"] == "second_order_detected"
    assert second.expected["plan_admission"] == "require_fixed_point_iteration"
    assert second.expected["automated_write"] == "only_after_fixed_point"
    assert second.artifacts["consumers"]["content"]["second_order_consumers"] == 1
    assert second.artifacts["graph"]["content"]["post_repair_new_delta"] is True


def test_frontier_stale_poison_readonly_llm_and_weakened_test_boundaries(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    frontier = fixtures["reflection_plugin_registry_ffi_frontier"]
    assert frontier.expected["impact_disposition"] == "unknown_frontier"
    frontier_kinds = {
        entry["kind"]
        for entry in frontier.artifacts["consumers"]["content"]["frontier"]
    }
    assert frontier_kinds == {"reflection", "plugin_registry", "ffi"}
    assert frontier.artifacts["graph"]["content"]["complete"] is False

    stale = fixtures["stale_graph_vector_proof"]
    assert stale.artifacts["graph"]["content"]["stale"] is True
    assert stale.artifacts["proof"]["content"]["verdict"] == "stale_rejected"

    poison = fixtures["poisoned_retrieval"]
    assert poison.artifacts["value_sources"]["content"]["vector_score"] > 0.99
    assert poison.artifacts["value_sources"]["content"]["candidates"][0]["poisoned"] is True
    assert poison.expected["plan_admission"] == "abstain"

    readonly = fixtures["read_only_cross_repository"]
    consumer = readonly.artifacts["consumers"]["content"]["resolved"][0]
    assert consumer["read_only"] is True
    assert consumer["cross_repository"] is True

    llm = fixtures["llm_scope_escape"]
    assert llm.artifacts["plan"]["content"]["scope_escape"] is True
    assert "llm_semantic_non_authoritative" in llm.expected["reason_codes"]

    weakened = fixtures["weakened_test"]
    assert weakened.expected["value_mapping"] == "test_weakening_rejected"
    assert "weakened_test_rejected" in weakened.expected["reason_codes"]


def test_async_error_effect_auth_resource_lifetime_drift_is_multi_facet(
    manifest: ChangePropagationFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}
    drift = fixtures["async_error_effect_auth_resource_lifetime_drift"]
    facets = set(drift.artifacts["delta"]["content"]["facets"])
    assert {
        "sync_to_async",
        "error_set_change",
        "new_io_effect",
        "auth_capability_raise",
        "resource_handle_required",
        "lifetime_shortened",
    } <= facets
    assert drift.expected["plan_admission"] == "abstain"
    assert drift.expected["automated_write"] == "never"


def test_manifest_is_reproducible_from_recipe_builder() -> None:
    """Guard against hand-edits that desync content_ids from payloads."""
    build_path = FIXTURE_ROOT / "build_manifest.py"
    assert build_path.is_file()

    # Import the local builder without installing a package.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "change_propagation_build_manifest", build_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rebuilt = module.build_manifest()
    on_disk = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert rebuilt == on_disk
