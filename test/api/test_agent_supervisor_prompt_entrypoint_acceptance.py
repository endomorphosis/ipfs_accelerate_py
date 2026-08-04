"""Frozen acceptance contract for prompt-only agent-supervisor entrypoints.

This test intentionally validates data rather than implementation.  Later
entrypoint work must execute this population without deleting cases or
weakening rollout gates to make promotion pass.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any

import pytest

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "agent_supervisor_prompt_entrypoints"
    / "manifest.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {record["id"]: record for record in records}
    assert len(indexed) == len(list(records))
    return indexed


def _gate_index(manifest: dict[str, Any], category: str) -> dict[str, dict[str, Any]]:
    return _by_id(manifest["promotion_gates"][category])


def test_manifest_is_versioned_frozen_and_change_controlled(
    manifest: dict[str, Any],
) -> None:
    assert manifest["schema_version"] == (
        "agent-supervisor-prompt-entrypoint-acceptance/v1"
    )
    assert manifest["fixture_revision"] == 1
    assert manifest["frozen"] is True
    assert manifest["change_control"]["owner_task"] == "ASE-002"
    assert "may not be removed or weakened" in manifest["change_control"]["policy"]
    assert "a new schema_version" in manifest["change_control"]["breaking_change_requires"]


def test_fixture_population_covers_every_required_target_kind(
    manifest: dict[str, Any],
) -> None:
    targets = manifest["targets"]
    expected_kinds = {
        "clean",
        "dirty",
        "nested",
        "worktree",
        "submodule",
        "ambiguous",
        "degraded",
        "adversarial",
    }
    assert set(manifest["required_target_kinds"]) == expected_kinds
    assert {target["kind"] for target in targets} == expected_kinds
    assert len({target["id"] for target in targets}) == len(targets)

    for target in targets:
        assert target["setup"]["candidate_roots"]
        assert "resolution_disposition" in target["expected"]
        assert "run_disposition" in target["expected"]
        assert target["journey_profile"] in manifest["journey_profiles"]
        assert isinstance(target["single_repository_promotion_population"], bool)


def test_transport_contract_has_prompt_only_run_and_handle_based_control(
    manifest: dict[str, Any],
) -> None:
    required_transports = {"python", "cli", "mcp", "mcp++"}
    required_operations = {"run", "status", "steer", "follow"}
    transports = _by_id(manifest["transports"])

    assert set(manifest["required_transports"]) == required_transports
    assert set(manifest["required_operations"]) == required_operations
    assert set(transports) == required_transports

    for transport in transports.values():
        assert set(transport["bindings"]) == required_operations
        assert transport["authenticated_context"]
        serialized_run_binding = json.dumps(transport["bindings"]["run"])
        assert "prompt" in serialized_run_binding.lower()
        for operation in {"status", "steer", "follow"}:
            serialized = json.dumps(transport["bindings"][operation])
            assert "run_id" in serialized.lower()

    mcp_plus_plus = transports["mcp++"]["ucan_contract"]
    assert mcp_plus_plus["required"] is True
    assert mcp_plus_plus["does_not_replace_inner_effect_authorization"] is True
    assert {
        "audience_must_match",
        "issuer_must_be_trusted",
        "resource_must_cover_target",
        "ability_must_cover_operation",
        "expiry_and_proof_chain_must_validate",
    } <= set(mcp_plus_plus)


def test_every_target_runs_the_same_operation_journey_on_every_transport(
    manifest: dict[str, Any],
) -> None:
    required_transports = set(manifest["required_transports"])
    required_operations = manifest["required_operations"]
    covered_pairs: set[tuple] = set()

    for target in manifest["targets"]:
        assert set(target["transports"]) == required_transports
        profile = manifest["journey_profiles"][target["journey_profile"]]
        operations = [step["operation"] for step in profile["steps"]]
        assert operations == required_operations
        for step in profile["steps"]:
            assert step["expected_outcome"]
            assert step["required_records"]
        covered_pairs.update(product([target["kind"]], target["transports"]))

    assert covered_pairs == set(
        product(manifest["required_target_kinds"], manifest["required_transports"])
    )


def test_dirty_nested_worktree_and_submodule_cases_freeze_safe_targeting(
    manifest: dict[str, Any],
) -> None:
    targets = {target["kind"]: target for target in manifest["targets"]}

    dirty = targets["dirty"]["expected"]
    assert dirty["mutation_location"] == "isolated_worktree"
    assert dirty["current_checkout_effect"] == "none"
    assert dirty["dirty_state_preserved_byte_for_byte"] is True

    nested = targets["nested"]
    assert nested["expected"]["selected_target"] == "outer/inner"
    assert nested["expected"]["outer_repository_effect"] == "none"

    worktree = targets["worktree"]["expected"]
    assert worktree["source_worktree_effect"] == "none"
    assert worktree["tree_identity_is_checkout_specific"] is True

    submodule = targets["submodule"]["expected"]
    assert submodule["superproject_effect"] == "none"
    assert submodule["external_git_dir_not_treated_as_repository_root"] is True


def test_ambiguous_target_is_useful_preview_without_mutation(
    manifest: dict[str, Any],
) -> None:
    ambiguous = next(
        target for target in manifest["targets"] if target["kind"] == "ambiguous"
    )
    assert ambiguous["expected"] == {
        "selected_target": None,
        "resolution_disposition": "ambiguous",
        "run_disposition": "preview_only",
        "mutation_location": None,
        "question_count": 1,
        "candidate_effect": "none",
    }
    profile = manifest["journey_profiles"][ambiguous["journey_profile"]]
    assert profile["steps"][0]["expected_outcome"] == (
        "preview_requires_target_selection"
    )
    assert profile["steps"][2]["expected_outcome"] == "rejected_no_admitted_run"


def test_degraded_case_keeps_local_progress_and_never_fakes_distribution(
    manifest: dict[str, Any],
) -> None:
    degraded = next(
        target for target in manifest["targets"] if target["kind"] == "degraded"
    )
    assert degraded["setup"]["duckdb"] == "available"
    assert degraded["setup"]["ipfs_peer"] == "unavailable"
    assert degraded["expected"]["run_disposition"] == "admitted_degraded"
    assert degraded["expected"]["local_coordination"] == "duckdb"
    assert degraded["expected"]["portable_checkpoint"] == "parquet_and_ipld"
    assert degraded["expected"]["remote_publication"] == "deferred_with_receipt"
    assert degraded["expected"]["false_distributed_durability_claim"] is False


def test_adversarial_prompt_cannot_select_target_configuration_or_authority(
    manifest: dict[str, Any],
) -> None:
    prompt_contract = manifest["prompt_contract"]
    adversarial = next(
        target for target in manifest["targets"] if target["kind"] == "adversarial"
    )

    assert prompt_contract["prompt_is_data_not_configuration"] is True
    assert {
        "repository_root",
        "executable_argv",
        "caller",
        "authority",
        "ucan",
        "merge_or_push_rights",
        "lease_backend",
    } <= set(prompt_contract["forbidden_prompt_selected_fields"])
    assert prompt_contract["durable_canaries"][0] in adversarial["setup"]["run_prompt"]
    assert adversarial["expected"]["selected_target"] == "repo"
    assert adversarial["expected"]["run_disposition"] == "denied"
    assert adversarial["expected"]["outside_allowlist_effect"] == "none"
    assert adversarial["expected"]["authority_escalation"] == "none"
    assert adversarial["expected"]["durable_canary_occurrences"] == 0


def test_duckdb_parquet_ipld_coordination_roles_are_safe_and_explicit(
    manifest: dict[str, Any],
) -> None:
    contract = manifest["coordination_storage_contract"]

    assert "authoritative_local_coordination" in contract["duckdb"]["roles"]
    assert "transactional_claim_compare_and_swap" in contract["duckdb"]["roles"]
    assert (
        "one explicitly assigned authenticated writer per shard"
        in contract["duckdb"]["write_model"]
    )

    assert contract["parquet"]["mutable_lock_or_lease_store"] is False
    assert "immutable event batches" in contract["parquet"]["roles"]
    assert contract["ipld"]["codec"] == "dag-json"
    assert contract["ipld"]["identifier"] == "CIDv1 multihash"
    assert contract["ipld"]["links_must_be_verified_before_admission"] is True
    assert contract["ipfs"]["availability_is_authority"] is False

    safety = contract["distributed_safety"]
    assert "not a consensus or locking primitive" in safety["statement"]
    assert {
        "one active writer per shard",
        "signed lease and fence records",
        "stale-writer rejection",
        "fail-closed mutation when fencing cannot be proven",
    } <= set(safety["required_properties"])


def test_provider_route_prefers_grok_with_bounded_typed_codex_fallback(
    manifest: dict[str, Any],
) -> None:
    contract = manifest["implementation_provider_contract"]
    route = contract["default_route"]
    fallback = contract["fallback"]
    cases = _by_id(contract["route_cases"])

    assert route["preferred_provider"] == "grok"
    assert route["fallback_provider"] == "codex"
    assert route["maximum_fallback_dispatches_per_task_revision"] == 1
    assert route["fallback_must_remain_within_profile_budget"] is True
    assert route["fallback_chain_may_not_expand"] is True
    assert fallback["receipt_must_commit_before_fallback_dispatch"] is True
    assert set(fallback["allowed_typed_reasons"]) == {
        "preferred_provider_unavailable",
        "preferred_provider_quota_exhausted",
        "preferred_provider_capacity_unavailable",
        "preferred_provider_pre_effect_failure",
    }
    assert {
        "reason_code",
        "observed_capability_cid",
        "task_revision",
        "budget",
        "attempt_id",
    } <= set(fallback["receipt_required_fields"])

    expected_fallbacks = {
        "grok-unavailable-falls-back-to-codex": "preferred_provider_unavailable",
        "grok-quota-falls-back-to-codex": "preferred_provider_quota_exhausted",
        "grok-capacity-falls-back-to-codex": (
            "preferred_provider_capacity_unavailable"
        ),
        "grok-pre-effect-failure-falls-back-to-codex": (
            "preferred_provider_pre_effect_failure"
        ),
    }
    for case_id, reason in expected_fallbacks.items():
        assert cases[case_id]["selected_provider"] == "codex"
        assert cases[case_id]["fallback_reason"] == reason


def test_provider_override_is_profile_only_and_review_cannot_self_attest(
    manifest: dict[str, Any],
) -> None:
    contract = manifest["implementation_provider_contract"]
    override = contract["explicit_profile_override"]
    review = contract["independent_review"]
    cases = _by_id(contract["route_cases"])

    assert override == {
        "allowed": True,
        "must_be_authenticated": True,
        "must_be_allowlisted": True,
        "must_be_recorded_in_resolution_receipt": True,
    }
    assert contract["prompt_may_select_provider"] is False
    assert cases["authenticated-profile-overrides-default"]["selected_provider"] == (
        "codex"
    )
    prompt_case = cases["prompt-provider-injection-is-ignored"]
    assert prompt_case["selected_provider"] == "grok"
    assert prompt_case["profile_override"] is None

    assert review["same_attempt_may_satisfy_review"] is False
    assert review["codex_fallback_attempt_may_self_attest"] is False
    assert {"attempt_id", "process_identity", "review_authorization"} == set(
        review["required_distinct_fields"]
    )


def test_success_gates_are_quantitative_and_cannot_hide_failures(
    manifest: dict[str, Any],
) -> None:
    gates = _gate_index(manifest, "success")

    healthy = gates["single-repository-prompt-only-healthy-rate"]
    assert healthy["operator"] == ">="
    assert healthy["threshold"] >= 0.95
    assert healthy["minimum_samples_per_transport"] >= 30

    replay = gates["deterministic-resolution-replay-rate"]
    assert replay["operator"] == "=="
    assert replay["threshold"] == 1.0
    assert replay["minimum_replays_per_fixture"] >= 10

    compatibility = gates["expert-request-compatibility-rate"]
    assert compatibility["operator"] == "=="
    assert compatibility["threshold"] == 1.0
    assert compatibility["minimum_samples"] >= 100

    fallback = gates["typed-provider-fallback-receipt-rate"]
    assert fallback["operator"] == "=="
    assert fallback["threshold"] == 1.0
    assert fallback["minimum_samples_per_fallback_reason"] >= 30


def test_latency_gates_publish_p95_budgets_for_every_transport(
    manifest: dict[str, Any],
) -> None:
    required_transports = set(manifest["required_transports"])
    gates = _gate_index(manifest, "latency")
    expected_ids = {
        "time-to-run-handle-p95",
        "time-to-first-useful-event-p95",
        "steering-acknowledgement-p95",
        "status-read-p95",
    }
    assert set(gates) == expected_ids

    for gate in gates.values():
        assert gate["operator"] == "<="
        assert gate["unit"] == "milliseconds"
        assert set(gate["threshold_by_transport_ms"]) == required_transports
        assert all(value > 0 for value in gate["threshold_by_transport_ms"].values())
        assert gate["minimum_samples_per_transport"] >= 30
        assert gate["maximum_regression_from_published_baseline"] <= 0.2

    assert max(
        gates["time-to-run-handle-p95"]["threshold_by_transport_ms"].values()
    ) <= 3000
    assert max(
        gates["time-to-first-useful-event-p95"][
            "threshold_by_transport_ms"
        ].values()
    ) <= 35000


def test_parity_is_exact_across_transports_and_storage_projections(
    manifest: dict[str, Any],
) -> None:
    gates = _gate_index(manifest, "parity")
    assert set(gates) == {
        "closed-fixture-canonical-transport-parity",
        "duckdb-parquet-ipld-projection-parity",
    }
    for gate in gates.values():
        assert gate["operator"] == "=="
        assert gate["threshold"] == 1.0
        assert gate["minimum_fixture_coverage"] == 1.0

    assert {
        "target_resolution_receipt",
        "run_id",
        "status",
        "reason_code",
        "effect_set",
        "steering_revision",
        "event_order",
        "coordination_shard",
        "fencing_generation",
    } <= set(manifest["canonical_parity_projection"])


def test_all_safety_gates_are_zero_tolerance(
    manifest: dict[str, Any],
) -> None:
    gates = _gate_index(manifest, "safety")
    assert {
        "unauthorized-effects",
        "out-of-scope-effects",
        "unexpected-effects",
        "durable-prompt-or-secret-leaks",
        "duplicate-process-trees",
        "stale-writer-commits",
        "coordination-split-brain-incidents",
        "prompt-selected-provider-routes",
        "same-attempt-independent-review-attestations",
    } == set(gates)
    for gate in gates.values():
        assert gate["operator"] == "=="
        assert gate["threshold"] == 0
        assert gate["unit"] == "count"

    assert {
        "duckdb_tables",
        "parquet_partitions",
        "ipld_blocks",
        "ipfs_pins",
        "logs",
        "errors",
        "process_argv",
        "process_environment",
    } <= set(manifest["durable_surfaces_to_inspect"])
