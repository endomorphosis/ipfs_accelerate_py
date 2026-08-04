"""Schema and safety-boundary checks for the logic-repair fixture corpus.

LPR-004: adversarial live logic-repair fixtures. Expectations never grant
vector, knowledge-graph, comment, Tactician ranking, or LLM semantic authority.
Goals, subgoals, proof/validated-refutation/abstention, edit sets, and fixed
points are content-identified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest


FIXTURE_ROOT = (
    Path(__file__).parents[1]
    / "fixtures"
    / "agent_supervisor"
    / "tactician_hammer_logic_repair"
)
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-logic-repair-fixture-manifest@1"
)
CORPUS_ID = "tactician-hammer-logic-repair-adversarial-v1"

ARTIFACT_ROLES = frozenset(
    {
        "delta",
        "consumers",
        "goals",
        "premises",
        "subgoals",
        "plan",
        "proof",
        "edit_set",
        "fixed_point",
    }
)
EXPECTATION_SOURCES = frozenset({"reviewed_spec", "test"})

# Closed catalogue required by LPR-004 acceptance (plan §9.1–9.2).
REQUIRED_SCENARIOS = frozenset(
    {
        "unique_local_value",
        "upstream_threading",
        "deterministic_constructor",
        "multiple_callers",
        "rename_equivalence",
        "immutable_support_type",
        "stateful_support_type",
        "schema_migration",
        "async_error_migration",
        "analytical_repair",
        "model_required_path",
        "second_order_logic_gap",
        "same_typed_wrong_value",
        "vector_kg_comment_poisoning",
        "self_authored_expectation",
        "contradictory_circular_premises",
        "raw_malformed_countermodel",
        "stale_forged_proof",
        "wrong_theorem_native_statement_drift",
        "dynamic_reflection_generated_ffi_lifetime_concurrency",
        "timeout_cancellation",
        "path_prompt_escape",
        "partial_scc_rollback",
        "passing_tests_missed_caller",
        "ordinary_generic_provider_overlay",
    }
)

POSITIVE_SCENARIOS = frozenset(
    {
        "unique_local_value",
        "upstream_threading",
        "deterministic_constructor",
        "multiple_callers",
        "rename_equivalence",
        "immutable_support_type",
        "stateful_support_type",
        "schema_migration",
        "async_error_migration",
        "analytical_repair",
        "model_required_path",
        "second_order_logic_gap",
    }
)

FAIL_CLOSED_SCENARIOS = frozenset(
    {
        "same_typed_wrong_value",
        "vector_kg_comment_poisoning",
        "self_authored_expectation",
        "contradictory_circular_premises",
        "raw_malformed_countermodel",
        "stale_forged_proof",
        "wrong_theorem_native_statement_drift",
        "dynamic_reflection_generated_ffi_lifetime_concurrency",
        "timeout_cancellation",
        "path_prompt_escape",
        "partial_scc_rollback",
        "passing_tests_missed_caller",
        "ordinary_generic_provider_overlay",
    }
)

ADMITTABLE_ANALYTICAL_SCENARIOS = frozenset(
    {
        "unique_local_value",
        "upstream_threading",
        "deterministic_constructor",
        "multiple_callers",
        "rename_equivalence",
        "immutable_support_type",
        "stateful_support_type",
        "schema_migration",
        "async_error_migration",
        "analytical_repair",
    }
)

REQUIRED_CALLER_KINDS = frozenset(
    {"direct", "aliased", "wrapped", "method", "adapter"}
)

REPAIR_DISPOSITION_VALUES = frozenset(
    {"analytical", "model_required", "abstain", "rollback"}
)
PROOF_DISPOSITION_VALUES = frozenset(
    {
        "proved",
        "validated_refutation",
        "inconclusive",
        "unsupported",
        "stale",
        "abstention",
    }
)
PLAN_ADMISSION_VALUES = frozenset(
    {
        "admit_after_proof",
        "require_model",
        "require_fixed_point_iteration",
        "abstain",
        "rollback",
    }
)
AUTOMATED_WRITE_VALUES = frozenset(
    {"never", "only_after_proof", "only_after_fixed_point"}
)
COMPLETION_VALUES = frozenset(
    {
        "success",
        "fail_closed",
        "rollback",
        "incomplete_until_second_order",
        "approval_required",
    }
)


def _canonical_content_id(content: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExpectedLogicGoal:
    """Content-identified expected logic goal from a fixture recipe."""

    goal_id: str
    family: str
    positive_statement: str
    negative_counterexample_target: str
    affected_symbols: tuple[str, ...]
    expectation_authority: str
    semantic_authority: bool

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpectedLogicGoal":
        return cls(
            goal_id=str(value["goal_id"]),
            family=str(value["family"]),
            positive_statement=str(value["positive_statement"]),
            negative_counterexample_target=str(
                value["negative_counterexample_target"]
            ),
            affected_symbols=tuple(value["affected_symbols"]),
            expectation_authority=str(value["expectation_authority"]),
            semantic_authority=bool(value["semantic_authority"]),
        )


@dataclass(frozen=True)
class ExpectedRepairDisposition:
    """Content-identified expected repair disposition for a fixture case."""

    repair_disposition: str
    proof_disposition: str
    plan_admission: str
    automated_write: str
    fixed_point: str
    completion: str
    reason_codes: tuple[str, ...]
    goal_families: tuple[str, ...]
    caller_kinds: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpectedRepairDisposition":
        return cls(
            repair_disposition=str(value["repair_disposition"]),
            proof_disposition=str(value["proof_disposition"]),
            plan_admission=str(value["plan_admission"]),
            automated_write=str(value["automated_write"]),
            fixed_point=str(value["fixed_point"]),
            completion=str(value["completion"]),
            reason_codes=tuple(value["reason_codes"]),
            goal_families=tuple(value.get("goal_families") or ()),
            caller_kinds=tuple(value.get("caller_kinds") or ()),
        )


@dataclass(frozen=True)
class LogicRepairFixture:
    fixture_id: str
    scenario: str
    expected: ExpectedRepairDisposition
    authority: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, Any]]
    goals: tuple[ExpectedLogicGoal, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogicRepairFixture":
        goals_payload = value["artifacts"]["goals"]["content"]["inventory"]
        return cls(
            fixture_id=str(value["id"]),
            scenario=str(value["scenario"]),
            expected=ExpectedRepairDisposition.from_dict(value["expected"]),
            authority=value["authority"],
            artifacts=value["artifacts"],
            goals=tuple(ExpectedLogicGoal.from_dict(item) for item in goals_payload),
        )


@dataclass(frozen=True)
class LogicRepairFixtureManifest:
    schema: str
    corpus_id: str
    fixtures: tuple[LogicRepairFixture, ...]

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> "LogicRepairFixtureManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            schema=str(payload["schema"]),
            corpus_id=str(payload["corpus_id"]),
            fixtures=tuple(
                LogicRepairFixture.from_dict(case) for case in payload["cases"]
            ),
        )


@pytest.fixture(scope="module")
def manifest() -> LogicRepairFixtureManifest:
    return LogicRepairFixtureManifest.load()


def test_manifest_has_a_content_addressed_recipe_for_every_required_scenario(
    manifest: LogicRepairFixtureManifest,
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
        assert expected.repair_disposition in REPAIR_DISPOSITION_VALUES
        assert expected.proof_disposition in PROOF_DISPOSITION_VALUES
        assert expected.plan_admission in PLAN_ADMISSION_VALUES
        assert expected.automated_write in AUTOMATED_WRITE_VALUES
        assert expected.fixed_point
        assert expected.completion in COMPLETION_VALUES
        assert expected.reason_codes
        assert fixture.goals

        for role, artifact in fixture.artifacts.items():
            content = artifact["content"]
            assert artifact["content_id"] == _canonical_content_id(content), (
                fixture.fixture_id,
                role,
            )
            assert isinstance(content["path"], str) and not content["path"].startswith(
                "/"
            )


def test_expected_goals_subgoals_proof_edit_set_and_fixed_point_are_content_identified(
    manifest: LogicRepairFixtureManifest,
) -> None:
    content_id_roles = ("goals", "subgoals", "proof", "edit_set", "fixed_point")
    for fixture in manifest.fixtures:
        for role in content_id_roles:
            artifact = fixture.artifacts[role]
            assert artifact["content_id"].startswith("sha256:")
            assert len(artifact["content_id"]) == len("sha256:") + 64
            assert artifact["content_id"] == _canonical_content_id(artifact["content"])

        goals = fixture.artifacts["goals"]["content"]["inventory"]
        assert goals
        for goal in goals:
            assert goal["goal_id"]
            assert goal["family"]
            assert goal["positive_statement"]
            assert goal["negative_counterexample_target"]
            assert goal["semantic_authority"] is False

        subgoals = fixture.artifacts["subgoals"]["content"]
        assert subgoals["acyclic"] is True
        assert subgoals["dag"]
        for subgoal in subgoals["dag"]:
            assert subgoal["subgoal_id"]
            assert subgoal["parent_goal_id"]
            assert subgoal["semantic_authority"] is False

        proof = fixture.artifacts["proof"]["content"]
        assert proof["disposition"] in PROOF_DISPOSITION_VALUES or proof[
            "disposition"
        ] in {"proved", "validated_refutation", "inconclusive", "unsupported", "stale", "abstention"}
        assert "verdict" in proof

        edit_set = fixture.artifacts["edit_set"]["content"]
        assert edit_set["content_identified"] is True
        assert isinstance(edit_set["paths"], list)

        fixed_point = fixture.artifacts["fixed_point"]["content"]
        assert "disposition" in fixed_point
        assert "residual_logic_gaps" in fixed_point


def test_expectations_never_grant_vector_kg_comment_tactician_or_llm_authority(
    manifest: LogicRepairFixtureManifest,
) -> None:
    forbidden_sources = {
        "vector",
        "vector_score",
        "knowledge_graph",
        "kg",
        "comment",
        "llm",
        "llm_semantic",
        "tactician",
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
        assert authority["comment_authoritative"] is False
        assert authority["tactician_ranking_authoritative"] is False
        assert authority["llm_semantic_authoritative"] is False
        assert (
            authority["solver_verified_without_reconstruction_authoritative"] is False
        )
        assert authority["requires_independent_proof"] is True
        assert sources.isdisjoint(forbidden_sources)

        for premise in fixture.artifacts["premises"]["content"]["entries"]:
            assert premise["semantic_authority"] is False


def test_positive_analytical_paths_require_proof_before_write(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}
    for scenario in ADMITTABLE_ANALYTICAL_SCENARIOS:
        fixture = fixtures[scenario]
        assert fixture.expected.repair_disposition == "analytical"
        assert fixture.expected.proof_disposition == "proved"
        assert fixture.expected.plan_admission == "admit_after_proof"
        assert fixture.expected.automated_write == "only_after_proof"
        assert fixture.expected.completion == "success"
        assert fixture.artifacts["plan"]["content"]["analytical"] is True
        assert fixture.artifacts["proof"]["content"]["disposition"] == "proved"


def test_model_required_and_second_order_paths_are_represented(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    model = fixtures["model_required_path"]
    assert model.expected.repair_disposition == "model_required"
    assert model.expected.plan_admission == "require_model"
    assert model.expected.automated_write == "never"
    assert model.expected.completion == "approval_required"
    assert model.artifacts["plan"]["content"]["model_required"] is True
    assert model.artifacts["plan"]["content"]["llm_cannot_choose_semantics"] is True
    assert model.artifacts["edit_set"]["content"]["write_requires_approval"] is True

    second = fixtures["second_order_logic_gap"]
    assert second.expected.plan_admission == "require_fixed_point_iteration"
    assert second.expected.automated_write == "only_after_fixed_point"
    assert second.expected.fixed_point == "second_order_required"
    assert second.expected.completion == "incomplete_until_second_order"
    assert second.artifacts["consumers"]["content"]["second_order_consumers"] == 1
    assert second.artifacts["fixed_point"]["content"]["post_repair_new_delta"] is True
    assert len(second.goals) == 2


def test_unique_local_threading_constructor_and_multiple_callers(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    unique = fixtures["unique_local_value"]
    assert "unique_local_reaching_definition" in unique.expected.reason_codes
    assert unique.artifacts["plan"]["content"]["source_expression"] == "tenant_id"

    threading = fixtures["upstream_threading"]
    assert threading.artifacts["consumers"]["content"]["thread_chain"] == [
        "top",
        "mid",
        "leaf",
    ]
    assert (
        threading.artifacts["plan"]["content"]["transform"]
        == "thread_parameter_through_chain"
    )

    constructor = fixtures["deterministic_constructor"]
    kinds = {
        entry["kind"]
        for entry in constructor.artifacts["consumers"]["content"]["resolved"]
    }
    assert kinds == {"factory", "di"}
    assert (
        constructor.artifacts["plan"]["content"]["transform"]
        == "inject_constructor_arg_from_provider"
    )

    multi = fixtures["multiple_callers"]
    assert set(multi.expected.caller_kinds) == REQUIRED_CALLER_KINDS
    consumers = multi.artifacts["consumers"]["content"]
    kinds = {entry["kind"] for entry in consumers["resolved"]}
    assert kinds == REQUIRED_CALLER_KINDS
    assert consumers["obligations"] == len(consumers["resolved"]) == 5
    assert consumers["one_compatible_cannot_discharge_others"] is True
    adapter = next(entry for entry in consumers["resolved"] if entry["kind"] == "adapter")
    assert adapter["adapter_required"] is True


def test_rename_immutable_stateful_schema_and_async_migrations(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    rename = fixtures["rename_equivalence"]
    assert rename.artifacts["delta"]["content"]["kind"] == "rename_move_reexport"
    assert "behavioral_equivalence_required" in rename.expected.reason_codes

    immutable = fixtures["immutable_support_type"]
    assert immutable.artifacts["delta"]["content"]["type_kind"] == "immutable_record"
    contract = immutable.artifacts["premises"]["content"]["entries"][0][
        "behavior_contract"
    ]
    assert contract["immutable"] is True

    stateful = fixtures["stateful_support_type"]
    assert "Active -> Suspended" in stateful.artifacts["delta"]["content"][
        "new_transition"
    ]
    assert "state_transition_proved" in stateful.expected.reason_codes

    schema = fixtures["schema_migration"]
    kinds = {
        entry["kind"] for entry in schema.artifacts["consumers"]["content"]["resolved"]
    }
    assert "serializer" in kinds
    assert "generated_client" in kinds

    async_err = fixtures["async_error_migration"]
    facets = set(async_err.artifacts["delta"]["content"]["facets"])
    assert {"sync_to_async", "error_set_change"} <= facets
    assert "TimeoutError" in async_err.artifacts["delta"]["content"]["errors_after"]


def test_fail_closed_adversarial_cases_never_automate_writes(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}
    for scenario in FAIL_CLOSED_SCENARIOS:
        fixture = fixtures[scenario]
        assert fixture.expected.plan_admission in {"abstain", "rollback"}
        assert fixture.expected.automated_write == "never"
        assert fixture.expected.completion in {
            "fail_closed",
            "rollback",
        }
        assert fixture.expected.repair_disposition in {"abstain", "rollback"}


def test_same_typed_wrong_value_and_poisoning_controls(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    wrong = fixtures["same_typed_wrong_value"]
    assert wrong.expected.proof_disposition == "validated_refutation"
    cand = next(
        entry
        for entry in wrong.artifacts["premises"]["content"]["entries"]
        if entry.get("refuted")
    )
    assert cand["same_type"] is True
    assert cand["information_content"] == "request_correlation_not_session"

    poison = fixtures["vector_kg_comment_poisoning"]
    premises = poison.artifacts["premises"]["content"]["entries"]
    assert any(entry.get("vector_score", 0) > 0.99 for entry in premises)
    assert any(entry.get("poisoned") and "kg_edge" in entry for entry in premises)
    assert any(
        entry.get("poisoned") and "comment_text" in entry for entry in premises
    )
    assert poison.expected.plan_admission == "abstain"


def test_self_authored_contradictory_and_circular_premises(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    self_auth = fixtures["self_authored_expectation"]
    entry = self_auth.artifacts["premises"]["content"]["entries"][0]
    assert entry["self_referential"] is True
    assert entry["target_is_source"] is True
    assert "premise_self_referential" in self_auth.expected.reason_codes

    contra = fixtures["contradictory_circular_premises"]
    premises = contra.artifacts["premises"]["content"]
    assert premises["circular"] is True
    assert premises["contradictory"] is True
    assert "no_ex_falso_repair" in contra.expected.reason_codes
    assert "premise_corpus_inconsistent" in contra.expected.reason_codes


def test_countermodel_stale_forged_and_native_drift_controls(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    raw = fixtures["raw_malformed_countermodel"]
    cm = raw.artifacts["proof"]["content"]["raw_countermodel"]
    assert cm["malformed"] is True
    assert cm["replay_status"] == "failed"
    assert cm["authoritative"] is False
    assert "countermodel_unvalidated" in raw.expected.reason_codes

    stale = fixtures["stale_forged_proof"]
    proof = stale.artifacts["proof"]["content"]
    assert proof["stale"] is True
    assert proof["forged"] is True
    assert proof["claimed_status"] == "verified"
    assert proof["bound_tree_id"] != proof["current_tree_id"]
    assert stale.expected.proof_disposition == "stale"

    drift = fixtures["wrong_theorem_native_statement_drift"]
    binding = drift.artifacts["proof"]["content"]["native_binding"]
    assert binding["statement_equivalence"] is False
    assert binding["drift"] is True
    assert binding["native_theorem"] != binding["admitted_logic_ir"]
    assert "wrong_theorem_rejected" in drift.expected.reason_codes


def test_frontier_timeout_path_escape_scc_missed_caller_and_provider_overlay(
    manifest: LogicRepairFixtureManifest,
) -> None:
    fixtures = {fixture.scenario: fixture for fixture in manifest.fixtures}

    frontier = fixtures["dynamic_reflection_generated_ffi_lifetime_concurrency"]
    kinds = {
        entry["kind"]
        for entry in frontier.artifacts["consumers"]["content"]["frontier"]
    }
    assert {
        "dynamic_dispatch",
        "reflection",
        "generated",
        "ffi",
        "lifetime",
        "concurrency",
    } <= kinds
    assert frontier.artifacts["consumers"]["content"][
        "unknown_frontier_blocks_autonomy"
    ] is True
    assert frontier.expected.proof_disposition == "unsupported"

    timeout = fixtures["timeout_cancellation"]
    assert timeout.artifacts["proof"]["content"]["outcome"] == "timeout"
    assert timeout.artifacts["proof"]["content"]["cancelled"] is True
    assert timeout.artifacts["plan"]["content"]["resource_policy"]["cancelled"] is True

    escape = fixtures["path_prompt_escape"]
    plan = escape.artifacts["plan"]["content"]
    assert plan["scope_escape"] is True
    assert plan["prompt_escape"] is True
    assert set(plan["admitted_write_paths"]) == {"src/service.py"}
    assert len(plan["proposed_write_paths"]) > len(plan["admitted_write_paths"])

    partial = fixtures["partial_scc_rollback"]
    assert partial.expected.plan_admission == "rollback"
    assert partial.expected.completion == "rollback"
    assert partial.artifacts["plan"]["content"]["partial_failure"] is True
    assert partial.artifacts["plan"]["content"]["partial_allowed"] is False
    assert partial.artifacts["edit_set"]["content"]["rolled_back"] is True

    missed = fixtures["passing_tests_missed_caller"]
    consumers = missed.artifacts["consumers"]["content"]
    assert consumers["missed_resolved_callers"] == 1
    assert consumers["tests_passing_with_gap"] is True
    assert missed.artifacts["fixed_point"]["content"]["tests_green_insufficient"] is True
    assert "passing_tests_not_completion" in missed.expected.reason_codes

    overlay = fixtures["ordinary_generic_provider_overlay"]
    delta = overlay.artifacts["delta"]["content"]
    assert delta["before"] == "f(a, b)"
    assert delta["after"] == "f(a, b, c)"
    assert delta["explicit_lpr_request"] is False
    assert overlay.artifacts["plan"]["content"]["ordinary_provider_overlay"] is True
    assert "ordinary_provider_patch_not_lpr_request" in overlay.expected.reason_codes


def test_positive_and_adversarial_partitions_cover_full_catalogue() -> None:
    assert POSITIVE_SCENARIOS | FAIL_CLOSED_SCENARIOS == REQUIRED_SCENARIOS
    assert POSITIVE_SCENARIOS.isdisjoint(FAIL_CLOSED_SCENARIOS)
    assert ADMITTABLE_ANALYTICAL_SCENARIOS <= POSITIVE_SCENARIOS


def test_manifest_is_reproducible_from_recipe_builder() -> None:
    """Guard against hand-edits that desync content_ids from payloads."""
    build_path = FIXTURE_ROOT / "build_manifest.py"
    assert build_path.is_file()

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tactician_hammer_logic_repair_build_manifest", build_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rebuilt = module.build_manifest()
    on_disk = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert rebuilt == on_disk


def test_subgoal_dags_are_acyclic_and_parent_linked(
    manifest: LogicRepairFixtureManifest,
) -> None:
    for fixture in manifest.fixtures:
        goal_ids = {goal.goal_id for goal in fixture.goals}
        dag: Sequence[Mapping[str, Any]] = fixture.artifacts["subgoals"]["content"][
            "dag"
        ]
        subgoal_ids = {item["subgoal_id"] for item in dag}
        for item in dag:
            assert item["parent_goal_id"] in goal_ids
            for dep in item.get("depends_on") or []:
                assert dep in subgoal_ids
                assert dep != item["subgoal_id"]
