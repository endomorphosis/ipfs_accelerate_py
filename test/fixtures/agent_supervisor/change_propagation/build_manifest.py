#!/usr/bin/env python3
"""Build the content-addressed change-propagation adversarial fixture manifest.

Recipes stay compact; content_ids are deterministic SHA-256 digests of each
artifact payload. Re-run this script after editing RECIPES to refresh
manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "ipfs_accelerate_py/agent-supervisor/change-propagation-fixture-manifest@1"
CORPUS_ID = "change-propagation-adversarial-v1"
DESCRIPTION = (
    "Hermetic, declarative transitive-change recipes. Vector, knowledge-graph, "
    "and LLM semantic scores are data only and never expectation authority."
)

# Artifact roles bound into every fixture recipe.
ARTIFACT_ROLES = (
    "delta",
    "consumers",
    "graph",
    "value_sources",
    "plan",
    "proof",
)

# Closed scenario catalogue required by RPR-024 acceptance.
REQUIRED_SCENARIOS = (
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
)


def _canonical_content_id(content: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _artifact(content: Mapping[str, Any]) -> dict[str, Any]:
    return {"content_id": _canonical_content_id(content), "content": dict(content)}


def _base_authority(**overrides: Any) -> dict[str, Any]:
    authority = {
        "expectation_sources": ["reviewed_spec", "test"],
        "implementation_observation_authoritative": False,
        "vector_score_authoritative": False,
        "knowledge_graph_authoritative": False,
        "llm_semantic_authoritative": False,
        "requires_independent_proof": True,
    }
    authority.update(overrides)
    return authority


def _expected(
    *,
    impact_disposition: str,
    value_mapping: str,
    plan_admission: str,
    automated_write: str,
    fixed_point: str,
    completion: str,
    reason_codes: list[str],
    caller_kinds: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "impact_disposition": impact_disposition,
        "value_mapping": value_mapping,
        "plan_admission": plan_admission,
        "automated_write": automated_write,
        "fixed_point": fixed_point,
        "completion": completion,
        "reason_codes": list(reason_codes),
    }
    if caller_kinds is not None:
        payload["caller_kinds"] = list(caller_kinds)
    return payload


# ---------------------------------------------------------------------------
# Compact recipes: each maps to one content-addressed fixture case.
# ---------------------------------------------------------------------------

RECIPES: list[dict[str, Any]] = [
    {
        "id": "two-to-three-argument-callers",
        "scenario": "two_to_three_argument_callers",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="unique_proved",
            plan_admission="admit_after_proof",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=[
                "each_two_arg_caller_gets_obligation",
                "compatible_default_does_not_discharge_others",
            ],
            caller_kinds=["direct", "aliased", "wrapped", "method"],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/process_arity.json",
                "kind": "parameter_add",
                "before": "process(left: A, right: B) -> R",
                "after": "process(left: A, right: B, context: C) -> R",
                "breaking_for": ["positional_two_arg_callers"],
            },
            "consumers": {
                "path": "consumers/process_callers.json",
                "resolved": [
                    {"kind": "direct", "site": "src/client.py:run", "args": 2},
                    {"kind": "aliased", "site": "src/alias_api.py:handle", "args": 2},
                    {"kind": "wrapped", "site": "src/wrapper.py:proxy", "args": 2},
                    {"kind": "method", "site": "src/service.py:Service.run", "args": 2},
                ],
                "obligations": 4,
                "one_compatible_cannot_discharge_others": True,
            },
            "graph": {
                "path": "graph/process_graph.json",
                "edges": ["direct", "alias", "wrapper", "method_dispatch"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/context.json",
                "candidates": [{"name": "request_context", "proved": True, "unique": True}],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/process_arity.json",
                "transform": "add_argument_from_unique_source",
                "atomic": True,
                "scc_groups": [],
            },
            "proof": {
                "path": "proof/process_arity.json",
                "expectation": "reconstructed provenance for context on every caller path",
                "verdict": "required",
            },
        },
    },
    {
        "id": "unique-in-scope-value",
        "scenario": "unique_in_scope_value",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="unique_proved",
            plan_admission="admit_after_proof",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=["unique_in_scope_source_proved", "analytical_transform_available"],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/unique_arg.json",
                "kind": "parameter_add",
                "before": "emit(event: Event) -> None",
                "after": "emit(event: Event, tenant: TenantId) -> None",
            },
            "consumers": {
                "path": "consumers/emit_callers.json",
                "resolved": [{"kind": "direct", "site": "src/pipeline.py:forward", "args": 1}],
                "obligations": 1,
            },
            "graph": {
                "path": "graph/emit_graph.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/tenant.json",
                "candidates": [
                    {
                        "name": "tenant_id",
                        "type": "TenantId",
                        "available_on_all_paths": True,
                        "proved": True,
                        "unique": True,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/unique_arg.json",
                "transform": "add_argument_from_unique_source",
                "source_expression": "tenant_id",
                "atomic": True,
            },
            "proof": {
                "path": "proof/unique_arg.json",
                "expectation": "type and information-content match for tenant_id",
                "verdict": "admitted",
            },
        },
    },
    {
        "id": "same-typed-wrong-information",
        "scenario": "same_typed_wrong_information",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="wrong_refuted",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["same_type_insufficient", "information_content_mismatch"],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/wrong_info.json",
                "kind": "parameter_add",
                "before": "authorize(user: UserId) -> Token",
                "after": "authorize(user: UserId, session: SessionId) -> Token",
            },
            "consumers": {
                "path": "consumers/authorize_callers.json",
                "resolved": [{"kind": "direct", "site": "src/auth_gate.py:check", "args": 1}],
                "obligations": 1,
            },
            "graph": {
                "path": "graph/authorize_graph.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/wrong_session.json",
                "candidates": [
                    {
                        "name": "request_id",
                        "type": "SessionId",
                        "same_type": True,
                        "information_content": "request_correlation_not_session",
                        "proved": False,
                        "refuted": True,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/wrong_info.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
            },
            "proof": {
                "path": "proof/wrong_info.json",
                "expectation": "counterexample: request_id is not session authority",
                "verdict": "refuted",
            },
        },
    },
    {
        "id": "branch-local-value",
        "scenario": "branch_local_value",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="path_incomplete",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["value_not_on_all_paths", "dominated_branch_only"],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/branch_local.json",
                "kind": "parameter_add",
                "before": "ship(order: Order) -> Receipt",
                "after": "ship(order: Order, warehouse: WarehouseId) -> Receipt",
            },
            "consumers": {
                "path": "consumers/ship_callers.json",
                "resolved": [{"kind": "direct", "site": "src/fulfillment.py:dispatch", "args": 1}],
                "obligations": 1,
                "path_condition": "if order.priority == 'express' warehouse is bound",
            },
            "graph": {
                "path": "graph/ship_graph.json",
                "edges": ["direct", "branch"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/warehouse_branch.json",
                "candidates": [
                    {
                        "name": "express_warehouse",
                        "available_on_all_paths": False,
                        "dominated_branch": "priority_express",
                        "proved": False,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/branch_local.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
            },
            "proof": {
                "path": "proof/branch_local.json",
                "expectation": "counterexample path without warehouse binding",
                "verdict": "refuted",
            },
        },
    },
    {
        "id": "nullable-value",
        "scenario": "nullable_value",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="nullability_mismatch",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["nullable_source_for_non_null_input", "total_conversion_missing"],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/nullable.json",
                "kind": "parameter_add",
                "before": "render(doc: Document) -> bytes",
                "after": "render(doc: Document, locale: Locale) -> bytes",
                "required_nullability": "non_null",
            },
            "consumers": {
                "path": "consumers/render_callers.json",
                "resolved": [{"kind": "direct", "site": "src/export.py:write", "args": 1}],
                "obligations": 1,
            },
            "graph": {
                "path": "graph/render_graph.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/locale_opt.json",
                "candidates": [
                    {
                        "name": "preferred_locale",
                        "type": "Locale | None",
                        "nullability": "nullable",
                        "required": "non_null",
                        "proved": False,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/nullable.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
            },
            "proof": {
                "path": "proof/nullable.json",
                "expectation": "None is not admissible for required Locale",
                "verdict": "refuted",
            },
        },
    },
    {
        "id": "parameter-threading",
        "scenario": "parameter_threading",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="thread_upstream",
            plan_admission="admit_after_proof",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=["thread_through_acyclic_chain", "fixed_point_worklist"],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/threading.json",
                "kind": "parameter_add",
                "before": "leaf(payload: Payload) -> Result",
                "after": "leaf(payload: Payload, ctx: Context) -> Result",
            },
            "consumers": {
                "path": "consumers/thread_chain.json",
                "resolved": [
                    {"kind": "direct", "site": "src/mid.py:mid", "args": 1},
                    {"kind": "wrapped", "site": "src/top.py:top", "args": 1},
                ],
                "thread_chain": ["top", "mid", "leaf"],
                "obligations": 2,
            },
            "graph": {
                "path": "graph/thread_graph.json",
                "edges": ["top->mid", "mid->leaf"],
                "complete": True,
                "acyclic": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/thread_ctx.json",
                "candidates": [
                    {
                        "name": "request.context",
                        "available_at": "top",
                        "proved": True,
                        "unique": True,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/threading.json",
                "transform": "thread_parameter_through_chain",
                "atomic": True,
                "steps": ["add_param_mid", "add_param_top", "pass_at_top"],
            },
            "proof": {
                "path": "proof/threading.json",
                "expectation": "reconstructed acyclic threading with fixed-point discharge",
                "verdict": "admitted",
            },
        },
    },
    {
        "id": "config-di-factory-construction",
        "scenario": "config_di_factory_construction",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="construct_via_provider",
            plan_admission="admit_after_proof",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=[
                "di_container_provider_proved",
                "factory_totality_required",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/di_factory.json",
                "kind": "parameter_add",
                "before": "Worker()",
                "after": "Worker(clock: Clock)",
            },
            "consumers": {
                "path": "consumers/worker_construction.json",
                "resolved": [
                    {"kind": "factory", "site": "src/factory.py:build_worker", "args": 0},
                    {"kind": "di", "site": "src/container.py:resolve", "args": 0},
                    {"kind": "config", "site": "config/workers.yaml", "args": 0},
                ],
                "obligations": 3,
            },
            "graph": {
                "path": "graph/di_graph.json",
                "edges": ["factory", "di_registration", "config_binding"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/clock_provider.json",
                "candidates": [
                    {
                        "name": "container.provide(Clock)",
                        "construction": "di_factory",
                        "proved": True,
                        "unique": True,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/di_factory.json",
                "transform": "inject_constructor_arg_from_provider",
                "atomic": True,
            },
            "proof": {
                "path": "proof/di_factory.json",
                "expectation": "Clock provider total for construction domain",
                "verdict": "admitted",
            },
        },
    },
    {
        "id": "schema-serializer-generated-client",
        "scenario": "schema_serializer_generated_client",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="schema_total_mapping",
            plan_admission="admit_after_proof",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=[
                "schema_field_add_total",
                "serializer_and_generated_client_updated",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/schema_field.json",
                "kind": "schema_field_add",
                "before": "User { id, name }",
                "after": "User { id, name, locale }",
                "serialization": "json_v2",
            },
            "consumers": {
                "path": "consumers/schema_consumers.json",
                "resolved": [
                    {"kind": "serializer", "site": "src/serde/user.py:dump", "args": 1},
                    {"kind": "generated_client", "site": "generated/client/user.py:create", "args": 1},
                    {"kind": "test", "site": "tests/test_user_schema.py", "args": 1},
                ],
                "obligations": 3,
            },
            "graph": {
                "path": "graph/schema_graph.json",
                "edges": ["schema", "serializer", "generated_binding"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/locale_default.json",
                "candidates": [
                    {
                        "name": "DEFAULT_LOCALE",
                        "mapping": "total_for_policy",
                        "proved": True,
                        "unique": True,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/schema_field.json",
                "transform": "update_schema_serializer_generated",
                "atomic": True,
                "policy_authorized": True,
            },
            "proof": {
                "path": "proof/schema_field.json",
                "expectation": "total field mapping under current schema policy",
                "verdict": "admitted",
            },
        },
    },
    {
        "id": "new-class-method-data-structure",
        "scenario": "new_class_method_data_structure",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="require_behavior_contract",
            plan_admission="admit_after_proof",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=[
                "required_behavior_contract_before_placement",
                "unique_owner_placement_proved",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/new_type.json",
                "kind": "class_and_method_introduction",
                "required_type": "RateLimiter",
                "required_methods": ["allow", "reset"],
                "required_data": ["window", "budget"],
            },
            "consumers": {
                "path": "consumers/rate_limit_callers.json",
                "resolved": [
                    {"kind": "direct", "site": "src/api.py:handle", "needs": "RateLimiter.allow"}
                ],
                "obligations": 1,
            },
            "graph": {
                "path": "graph/new_type_graph.json",
                "edges": ["call", "ownership_layer"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/behavior_contract.json",
                "candidates": [],
                "behavior_contract": {
                    "source": "reviewed_spec",
                    "fields": ["window", "budget"],
                    "methods": ["allow", "reset"],
                    "invariants": ["budget_non_negative"],
                },
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/new_type.json",
                "transform": "introduce_type_with_proved_placement",
                "placement": "src/limits/rate_limiter.py",
                "atomic": True,
            },
            "proof": {
                "path": "proof/new_type.json",
                "expectation": "behavior and placement independently proved",
                "verdict": "admitted",
            },
        },
    },
    {
        "id": "stateful-service",
        "scenario": "stateful_service",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="state_transition_proved",
            plan_admission="admit_after_proof",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=[
                "state_machine_contract_required",
                "allowed_transitions_proved",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/stateful_service.json",
                "kind": "state_transition_change",
                "before": "Session { Idle, Active }",
                "after": "Session { Idle, Active, Suspended }",
                "new_transition": "Active -> Suspended",
            },
            "consumers": {
                "path": "consumers/session_consumers.json",
                "resolved": [
                    {"kind": "method", "site": "src/session.py:Session.suspend", "state": "Active"},
                    {"kind": "test", "site": "tests/test_session.py", "state": "Active"},
                ],
                "obligations": 2,
            },
            "graph": {
                "path": "graph/session_graph.json",
                "edges": ["state_flow", "method"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/session_state.json",
                "candidates": [
                    {
                        "name": "self.state",
                        "receiver_state": "Active",
                        "proved": True,
                        "unique": True,
                    }
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/stateful_service.json",
                "transform": "add_state_transition_and_guards",
                "atomic": True,
            },
            "proof": {
                "path": "proof/stateful_service.json",
                "expectation": "only Active may enter Suspended; Idle cannot",
                "verdict": "admitted",
            },
        },
    },
    {
        "id": "async-error-effect-auth-resource-lifetime-drift",
        "scenario": "async_error_effect_auth_resource_lifetime_drift",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="unsupported_multi_facet_drift",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "async_sync_mismatch",
                "error_contract_drift",
                "effect_capability_drift",
                "auth_resource_lifetime_drift",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/multi_facet_drift.json",
                "kind": "behavioral_facets",
                "facets": [
                    "sync_to_async",
                    "error_set_change",
                    "new_io_effect",
                    "auth_capability_raise",
                    "resource_handle_required",
                    "lifetime_shortened",
                ],
            },
            "consumers": {
                "path": "consumers/facet_callers.json",
                "resolved": [
                    {"kind": "direct", "site": "src/legacy_sync.py:call", "awaited": False},
                    {"kind": "method", "site": "src/svc.py:Svc.run", "handles_errors": ["Timeout"]},
                ],
                "obligations": 2,
            },
            "graph": {
                "path": "graph/facet_graph.json",
                "edges": ["direct", "method", "effect"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/facet_none.json",
                "candidates": [],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/facet_drift.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
            },
            "proof": {
                "path": "proof/facet_drift.json",
                "expectation": "multi-facet drift requires review; no automated mapping",
                "verdict": "unsupported",
            },
        },
    },
    {
        "id": "dependency-cycle-scc",
        "scenario": "dependency_cycle_scc",
        "expected": _expected(
            impact_disposition="scc_grouped",
            value_mapping="scc_atomic_group",
            plan_admission="admit_scc_transaction_only",
            automated_write="only_after_plan_admission",
            fixed_point="required",
            completion="success",
            reason_codes=[
                "scc_must_be_one_transaction",
                "partial_scc_forbidden",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/scc.json",
                "kind": "parameter_add",
                "symbols": ["mod_a.f", "mod_b.g"],
            },
            "consumers": {
                "path": "consumers/scc_consumers.json",
                "resolved": [
                    {"kind": "direct", "site": "src/mod_a.py:f", "args": 1},
                    {"kind": "direct", "site": "src/mod_b.py:g", "args": 1},
                ],
                "scc": ["mod_a.f", "mod_b.g"],
                "obligations": 2,
            },
            "graph": {
                "path": "graph/scc_graph.json",
                "edges": ["mod_a.f->mod_b.g", "mod_b.g->mod_a.f"],
                "sccs": [["mod_a.f", "mod_b.g"]],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/scc_ctx.json",
                "candidates": [
                    {"name": "shared_ctx", "proved": True, "unique": True, "no_forbidden_cycle": True}
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/scc.json",
                "transform": "scc_transaction_group",
                "atomic": True,
                "scc_groups": [["mod_a.f", "mod_b.g"]],
                "partial_allowed": False,
            },
            "proof": {
                "path": "proof/scc.json",
                "expectation": "both members updated or none; rollback on partial",
                "verdict": "admitted",
            },
        },
    },
    {
        "id": "reflection-plugin-registry-ffi-frontier",
        "scenario": "reflection_plugin_registry_ffi_frontier",
        "expected": _expected(
            impact_disposition="unknown_frontier",
            value_mapping="unsupported",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "reflection_frontier",
                "plugin_registry_frontier",
                "ffi_frontier",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/dynamic_frontier.json",
                "kind": "parameter_add",
                "before": "dispatch(name: str, *args)",
                "after": "dispatch(name: str, ctx: Context, *args)",
            },
            "consumers": {
                "path": "consumers/dynamic_consumers.json",
                "resolved": [],
                "frontier": [
                    {"kind": "reflection", "site": "src/reflect.py:invoke"},
                    {"kind": "plugin_registry", "site": "src/plugins.py:load"},
                    {"kind": "ffi", "site": "native/bridge.c:call"},
                ],
                "obligations": 0,
            },
            "graph": {
                "path": "graph/dynamic_graph.json",
                "edges": ["string_dispatch", "dlopen", "registry_lookup"],
                "complete": False,
                "unknown_frontier": ["reflection", "plugin", "ffi"],
            },
            "value_sources": {
                "path": "value_sources/dynamic_none.json",
                "candidates": [],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/dynamic_frontier.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
                "review_only_frontier": True,
            },
            "proof": {
                "path": "proof/dynamic_frontier.json",
                "expectation": "frontier remains explicit; no automated write",
                "verdict": "unsupported",
            },
        },
    },
    {
        "id": "stale-graph-vector-proof",
        "scenario": "stale_graph_vector_proof",
        "expected": _expected(
            impact_disposition="stale",
            value_mapping="stale_rejected",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "stale_graph_root",
                "stale_vector_index",
                "stale_proof_receipt",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/stale_roots.json",
                "kind": "parameter_add",
                "tree_id": "tree:current",
                "claimed_tree_id": "tree:stale",
            },
            "consumers": {
                "path": "consumers/stale_consumers.json",
                "resolved": [{"kind": "direct", "site": "src/app.py:main", "args": 1}],
                "root_freshness": "stale",
                "obligations": 1,
            },
            "graph": {
                "path": "graph/stale_graph.json",
                "edges": ["direct"],
                "complete": False,
                "stale": True,
                "bound_tree_id": "tree:stale",
                "unknown_frontier": ["requires_rebuild"],
            },
            "value_sources": {
                "path": "value_sources/stale_vector.json",
                "candidates": [{"name": "ctx", "vector_score": 0.97, "stale": True}],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/stale.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
            },
            "proof": {
                "path": "proof/stale.json",
                "expectation": "stale receipt cannot authorize admission",
                "verdict": "stale_rejected",
                "proof_tree_id": "tree:stale",
            },
        },
    },
    {
        "id": "poisoned-retrieval",
        "scenario": "poisoned_retrieval",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="poison_rejected",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "vector_non_authoritative",
                "knowledge_graph_non_authoritative",
                "poisoned_nomination_rejected",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/poison.json",
                "kind": "parameter_add",
                "before": "bill(account: Account) -> Invoice",
                "after": "bill(account: Account, tax: TaxCode) -> Invoice",
            },
            "consumers": {
                "path": "consumers/bill_callers.json",
                "resolved": [{"kind": "direct", "site": "src/billing.py:charge", "args": 1}],
                "obligations": 1,
            },
            "graph": {
                "path": "graph/poison_graph.json",
                "edges": ["direct", "nominated_kg_edge"],
                "complete": True,
                "unknown_frontier": [],
                "nominated_only_edges": ["kg:bill->marketing_tax"],
            },
            "value_sources": {
                "path": "value_sources/poisoned_tax.json",
                "candidates": [
                    {
                        "name": "marketing.promo_code",
                        "type": "TaxCode",
                        "vector_score": 0.999,
                        "kg_score": 0.99,
                        "poisoned": True,
                        "proved": False,
                    }
                ],
                "semantic_authority": False,
                "vector_score": 0.999,
            },
            "plan": {
                "path": "plan/poison.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
            },
            "proof": {
                "path": "proof/poison.json",
                "expectation": "high similarity cannot authorize wrong tax source",
                "verdict": "rejected",
            },
        },
    },
    {
        "id": "read-only-cross-repository",
        "scenario": "read_only_cross_repository",
        "expected": _expected(
            impact_disposition="out_of_write_authority",
            value_mapping="no_write_path",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "read_only_target",
                "cross_repository_path",
                "write_authority_missing",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/readonly.json",
                "kind": "parameter_add",
                "target_repository": "external/vendor-sdk",
            },
            "consumers": {
                "path": "consumers/readonly_consumers.json",
                "resolved": [
                    {
                        "kind": "generated_client",
                        "site": "external/vendor-sdk/client.py:call",
                        "read_only": True,
                        "cross_repository": True,
                    }
                ],
                "obligations": 1,
            },
            "graph": {
                "path": "graph/readonly_graph.json",
                "edges": ["external_import"],
                "complete": True,
                "write_authority": [],
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/readonly_none.json",
                "candidates": [],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/readonly.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
                "write_paths": [],
            },
            "proof": {
                "path": "proof/readonly.json",
                "expectation": "external read-only path never receives automated write",
                "verdict": "rejected",
            },
        },
    },
    {
        "id": "partial-transaction",
        "scenario": "partial_transaction",
        "expected": _expected(
            impact_disposition="partial",
            value_mapping="partial_group_failed",
            plan_admission="rollback",
            automated_write="never",
            fixed_point="incomplete",
            completion="rollback",
            reason_codes=[
                "partial_plan_not_completion",
                "checkpoint_rollback_required",
                "no_merge_of_partial",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/partial_tx.json",
                "kind": "parameter_add",
                "members": ["a", "b", "c"],
            },
            "consumers": {
                "path": "consumers/partial_consumers.json",
                "resolved": [
                    {"kind": "direct", "site": "src/a.py:f", "status": "applied"},
                    {"kind": "direct", "site": "src/b.py:g", "status": "failed"},
                    {"kind": "direct", "site": "src/c.py:h", "status": "pending"},
                ],
                "obligations": 3,
            },
            "graph": {
                "path": "graph/partial_graph.json",
                "edges": ["a->b", "b->c"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/partial_ctx.json",
                "candidates": [{"name": "ctx", "proved": True, "unique": True}],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/partial_tx.json",
                "transform": "atomic_multi_file",
                "atomic": True,
                "checkpoint": "cas:checkpoint-1",
                "partial_failure": True,
                "rollback_to": "cas:checkpoint-1",
            },
            "proof": {
                "path": "proof/partial_tx.json",
                "expectation": "clean compile after partial is not completion evidence",
                "verdict": "rollback",
            },
        },
    },
    {
        "id": "llm-scope-escape",
        "scenario": "llm_scope_escape",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="scope_escape_rejected",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "llm_path_expansion_rejected",
                "llm_cannot_choose_source",
                "llm_semantic_non_authoritative",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/llm_scope.json",
                "kind": "parameter_add",
                "admitted_write_paths": ["src/service.py"],
            },
            "consumers": {
                "path": "consumers/llm_consumers.json",
                "resolved": [{"kind": "direct", "site": "src/service.py:run", "args": 1}],
                "obligations": 1,
            },
            "graph": {
                "path": "graph/llm_graph.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/llm_proposal.json",
                "candidates": [
                    {
                        "name": "llm_proposed_extra_module",
                        "path": "src/unrelated.py",
                        "outside_lease": True,
                        "proved": False,
                    }
                ],
                "semantic_authority": False,
                "llm_semantic_score": 0.95,
            },
            "plan": {
                "path": "plan/llm_scope.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
                "admitted_write_paths": ["src/service.py"],
                "proposed_write_paths": ["src/service.py", "src/unrelated.py"],
                "scope_escape": True,
            },
            "proof": {
                "path": "proof/llm_scope.json",
                "expectation": "pre-provider gate rejects path expansion",
                "verdict": "rejected",
            },
        },
    },
    {
        "id": "weakened-test",
        "scenario": "weakened_test",
        "expected": _expected(
            impact_disposition="complete",
            value_mapping="test_weakening_rejected",
            plan_admission="abstain",
            automated_write="never",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "weakened_test_rejected",
                "deleted_contract_rejected",
                "suppression_rejected",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/weakened_test.json",
                "kind": "parameter_add",
                "before": "score(x: int) -> int",
                "after": "score(x: int, weight: float) -> int",
            },
            "consumers": {
                "path": "consumers/test_consumers.json",
                "resolved": [
                    {"kind": "test", "site": "tests/test_score.py:test_score", "args": 1},
                ],
                "obligations": 1,
                "proposed_test_change": "delete assertion on weight",
            },
            "graph": {
                "path": "graph/test_graph.json",
                "edges": ["test_call"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/weight.json",
                "candidates": [{"name": "DEFAULT_WEIGHT", "proved": True, "unique": True}],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/weakened_test.json",
                "transform": None,
                "atomic": False,
                "abstain": True,
                "proposed_validation": "weaken_or_skip_test",
                "accepted_validation": "dependency_complete_tests",
            },
            "proof": {
                "path": "proof/weakened_test.json",
                "expectation": "validation rejects weakened tests and suppressions",
                "verdict": "rejected",
            },
        },
    },
    {
        "id": "second-order-breaking-delta",
        "scenario": "second_order_breaking_delta",
        "expected": _expected(
            impact_disposition="second_order_detected",
            value_mapping="iterate_fixed_point",
            plan_admission="require_fixed_point_iteration",
            automated_write="only_after_fixed_point",
            fixed_point="second_order_required",
            completion="incomplete_until_second_order_discharged",
            reason_codes=[
                "repair_introduces_new_breaking_delta",
                "fixed_point_rediff_required",
                "no_false_completion",
            ],
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/second_order.json",
                "kind": "parameter_add",
                "primary": "process(left, right) -> process(left, right, context)",
                "second_order": "Context construction requires Config that callers lack",
            },
            "consumers": {
                "path": "consumers/second_order_consumers.json",
                "resolved": [
                    {"kind": "direct", "site": "src/client.py:run", "args": 2, "wave": 1},
                    {
                        "kind": "factory",
                        "site": "src/context_factory.py:build",
                        "args": 0,
                        "wave": 2,
                        "introduced_by_repair": True,
                    },
                ],
                "obligations": 2,
                "second_order_consumers": 1,
            },
            "graph": {
                "path": "graph/second_order_graph.json",
                "edges": ["direct", "constructor"],
                "complete": True,
                "unknown_frontier": [],
                "post_repair_new_delta": True,
            },
            "value_sources": {
                "path": "value_sources/second_order.json",
                "candidates": [
                    {"name": "request_context", "wave": 1, "proved": True},
                    {"name": "app_config", "wave": 2, "proved": False, "missing": True},
                ],
                "semantic_authority": False,
            },
            "plan": {
                "path": "plan/second_order.json",
                "transform": "fixed_point_iteration",
                "atomic": True,
                "waves": 2,
                "completion_blocked_until": "wave_2_discharged",
            },
            "proof": {
                "path": "proof/second_order.json",
                "expectation": "rediff detects new delta; bound exhaustion is incomplete",
                "verdict": "incomplete",
            },
        },
    },
]


def build_case(recipe: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = {
        role: _artifact(recipe["artifacts"][role]) for role in ARTIFACT_ROLES
    }
    return {
        "id": recipe["id"],
        "scenario": recipe["scenario"],
        "expected": recipe["expected"],
        "authority": recipe["authority"],
        "artifacts": artifacts,
    }


def build_manifest() -> dict[str, Any]:
    scenarios = [recipe["scenario"] for recipe in RECIPES]
    if set(scenarios) != set(REQUIRED_SCENARIOS):
        missing = set(REQUIRED_SCENARIOS) - set(scenarios)
        extra = set(scenarios) - set(REQUIRED_SCENARIOS)
        raise SystemExit(f"scenario catalogue drift: missing={missing!r} extra={extra!r}")
    if len(scenarios) != len(set(scenarios)):
        raise SystemExit("duplicate scenarios in recipes")
    return {
        "schema": SCHEMA,
        "corpus_id": CORPUS_ID,
        "description": DESCRIPTION,
        "cases": [build_case(recipe) for recipe in RECIPES],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = build_manifest()
    path = root / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path} with {len(manifest['cases'])} cases")


if __name__ == "__main__":
    main()
