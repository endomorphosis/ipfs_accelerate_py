#!/usr/bin/env python3
"""Build the content-addressed deterministic-doctor adversarial fixture manifest.

LPR-040: hermetic recipes for no-LLM diagnosis and repair. Compact declarative
payloads only — vector, KG, embedding, and model scores are data and never
expectation authority. Re-run after editing RECIPES to refresh manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-doctor-fixture-manifest@1"
)
CORPUS_ID = "deterministic-doctor-adversarial-v1"
DESCRIPTION = (
    "Hermetic deterministic-doctor recipes covering positive analytical repairs "
    "and adversarial fail-closed controls. Seeded truth may define expected "
    "outcomes; metrics, retrieval ranks, and observed implementations never "
    "grant production authority. Zero LLM or remote model-provider invocation."
)

ARTIFACT_ROLES = (
    "delta",
    "consumers",
    "graph",
    "value_sources",
    "retrieval",
    "proof",
    "plan",
    "sandbox",
    "fixed_point",
)

# Closed scenario catalogue required by LPR-040 acceptance.
REQUIRED_SCENARIOS = (
    # Positive analytical repairs
    "renamed_moved_symbol",
    "import_export_registration",
    "two_to_three_argument_callers",
    "constructor_factory_context_threading",
    "adapter_schema_serializer_manifest_artifact",
    # Adversarial fail-closed controls
    "same_type_wrong_value",
    "vector_collision",
    "kg_omission",
    "constant_embedding_fallback",
    "stale_corrupt_forged_cid_cache",
    "solver_lie_countermodel",
    "incomplete_ast_impact_scc",
    "dynamic_generated_native_ffi_public_schema_cross_root",
    "sandbox_escape",
    "crash_rollback",
    "oscillation",
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
        "expectation_sources": ["reviewed_spec", "static_evidence"],
        "implementation_observation_authoritative": False,
        "vector_score_authoritative": False,
        "knowledge_graph_authoritative": False,
        "embedding_authoritative": False,
        "llm_semantic_authoritative": False,
        "requires_independent_proof": True,
        "model_invocation_forbidden": True,
    }
    authority.update(overrides)
    return authority


def _expected(
    *,
    diagnosis: str,
    disposition: str,
    repair: str,
    fixed_point: str,
    completion: str,
    reason_codes: list[str],
    caller_kinds: list[str] | None = None,
    stages: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "diagnosis": diagnosis,
        "disposition": disposition,
        "repair": repair,
        "fixed_point": fixed_point,
        "completion": completion,
        "reason_codes": list(reason_codes),
    }
    if caller_kinds is not None:
        payload["caller_kinds"] = list(caller_kinds)
    if stages is not None:
        payload["stages"] = list(stages)
    return payload


DEFAULT_STAGES = [
    "diagnose",
    "retrieve",
    "prove",
    "transform",
    "impact",
    "transaction",
    "rollback",
    "fixed_point",
]

RECIPES: list[dict[str, Any]] = [
    # ------------------------------------------------------------------
    # Positive analytical repairs
    # ------------------------------------------------------------------
    {
        "id": "renamed-moved-symbol",
        "scenario": "renamed_moved_symbol",
        "family": "positive_analytical",
        "expected": _expected(
            diagnosis="rename_move",
            disposition="supported",
            repair="analytical",
            fixed_point="reached",
            completion="success",
            reason_codes=[
                "symbol_rename_equivalence",
                "all_resolved_callers_updated",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/rename.json",
                "kind": "rename_move",
                "before": "pkg.old_mod.process",
                "after": "pkg.new_mod.handle",
                "symbol_id": "symbol:process",
            },
            "consumers": {
                "path": "consumers/rename_callers.json",
                "resolved": [
                    {"kind": "direct", "site": "src/client.py:run"},
                    {"kind": "import", "site": "src/api.py:import"},
                ],
                "obligations": 2,
                "mandatory_callers": 2,
            },
            "graph": {
                "path": "graph/rename.json",
                "edges": ["direct", "import"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/rename.json",
                "candidates": [
                    {"name": "handle", "proved": True, "unique": True, "kind": "symbol"}
                ],
            },
            "retrieval": {
                "path": "retrieval/rename.json",
                "vector_hits": [],
                "kg_edges": ["rename_edge"],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/rename.json",
                "verdict": "reconstructed",
                "operator": "rename_symbol",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/rename.json",
                "transform": "rename_move_symbol",
                "atomic": True,
                "admitted": True,
            },
            "sandbox": {
                "path": "sandbox/rename.json",
                "enforced": True,
                "write_scope": ["pkg/old_mod.py", "pkg/new_mod.py", "src/client.py", "src/api.py"],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/rename.json",
                "iterations": 1,
                "residual_findings": 0,
                "oscillating": False,
            },
        },
    },
    {
        "id": "import-export-registration",
        "scenario": "import_export_registration",
        "family": "positive_analytical",
        "expected": _expected(
            diagnosis="import_export_registration",
            disposition="supported",
            repair="analytical",
            fixed_point="reached",
            completion="success",
            reason_codes=[
                "reexport_updated",
                "registry_entry_rewritten",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/registration.json",
                "kind": "import_export_registration",
                "before": "from pkg.legacy import Worker",
                "after": "from pkg.current import Worker",
                "registry": "PLUGIN_REGISTRY",
            },
            "consumers": {
                "path": "consumers/registration.json",
                "resolved": [
                    {"kind": "import", "site": "pkg/__init__.py"},
                    {"kind": "export", "site": "pkg/public_api.py"},
                    {"kind": "registration", "site": "pkg/plugins.py:register"},
                ],
                "obligations": 3,
                "mandatory_callers": 3,
            },
            "graph": {
                "path": "graph/registration.json",
                "edges": ["import", "export", "registration"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/registration.json",
                "candidates": [
                    {
                        "name": "Worker",
                        "proved": True,
                        "unique": True,
                        "kind": "export",
                    }
                ],
            },
            "retrieval": {
                "path": "retrieval/registration.json",
                "vector_hits": [],
                "kg_edges": ["export_edge"],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/registration.json",
                "verdict": "reconstructed",
                "operator": "rewrite_import_export_registration",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/registration.json",
                "transform": "import_export_registration",
                "atomic": True,
                "admitted": True,
            },
            "sandbox": {
                "path": "sandbox/registration.json",
                "enforced": True,
                "write_scope": [
                    "pkg/__init__.py",
                    "pkg/public_api.py",
                    "pkg/plugins.py",
                ],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/registration.json",
                "iterations": 1,
                "residual_findings": 0,
                "oscillating": False,
            },
        },
    },
    {
        "id": "two-to-three-argument-callers",
        "scenario": "two_to_three_argument_callers",
        "family": "positive_analytical",
        "expected": _expected(
            diagnosis="arity_change",
            disposition="supported",
            repair="analytical",
            fixed_point="reached",
            completion="success",
            reason_codes=[
                "each_two_arg_caller_gets_obligation",
                "compatible_default_does_not_discharge_others",
            ],
            caller_kinds=["direct", "aliased", "wrapped", "method"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/arity.json",
                "kind": "parameter_add",
                "before": "process(left: A, right: B) -> R",
                "after": "process(left: A, right: B, context: C) -> R",
            },
            "consumers": {
                "path": "consumers/arity.json",
                "resolved": [
                    {"kind": "direct", "site": "src/client.py:run", "args": 2},
                    {"kind": "aliased", "site": "src/alias_api.py:handle", "args": 2},
                    {"kind": "wrapped", "site": "src/wrapper.py:proxy", "args": 2},
                    {"kind": "method", "site": "src/service.py:Service.run", "args": 2},
                ],
                "obligations": 4,
                "mandatory_callers": 4,
                "one_compatible_cannot_discharge_others": True,
            },
            "graph": {
                "path": "graph/arity.json",
                "edges": ["direct", "alias", "wrapper", "method_dispatch"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/context.json",
                "candidates": [
                    {
                        "name": "request_context",
                        "proved": True,
                        "unique": True,
                        "kind": "local",
                    }
                ],
            },
            "retrieval": {
                "path": "retrieval/arity.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/arity.json",
                "verdict": "reconstructed",
                "operator": "add_argument_from_unique_source",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/arity.json",
                "transform": "add_argument_from_unique_source",
                "atomic": True,
                "admitted": True,
            },
            "sandbox": {
                "path": "sandbox/arity.json",
                "enforced": True,
                "write_scope": [
                    "src/client.py",
                    "src/alias_api.py",
                    "src/wrapper.py",
                    "src/service.py",
                ],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/arity.json",
                "iterations": 1,
                "residual_findings": 0,
                "oscillating": False,
            },
        },
    },
    {
        "id": "constructor-factory-context-threading",
        "scenario": "constructor_factory_context_threading",
        "family": "positive_analytical",
        "expected": _expected(
            diagnosis="constructor_threading",
            disposition="supported",
            repair="analytical",
            fixed_point="reached",
            completion="success",
            reason_codes=[
                "factory_threads_context",
                "constructor_site_updated",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/ctor.json",
                "kind": "constructor_parameter_add",
                "before": "Service(config: Config)",
                "after": "Service(config: Config, context: Context)",
            },
            "consumers": {
                "path": "consumers/ctor.json",
                "resolved": [
                    {"kind": "constructor", "site": "src/app.py:main"},
                    {"kind": "factory", "site": "src/factory.py:make_service"},
                    {"kind": "di", "site": "src/container.py:bind"},
                ],
                "obligations": 3,
                "mandatory_callers": 3,
                "thread_chain": ["container", "factory", "Service.__init__"],
            },
            "graph": {
                "path": "graph/ctor.json",
                "edges": ["constructor", "factory", "di"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/ctor.json",
                "candidates": [
                    {
                        "name": "app_context",
                        "proved": True,
                        "unique": True,
                        "kind": "factory_arg",
                    }
                ],
            },
            "retrieval": {
                "path": "retrieval/ctor.json",
                "vector_hits": [],
                "kg_edges": ["ctor_edge"],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/ctor.json",
                "verdict": "reconstructed",
                "operator": "thread_constructor_context",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/ctor.json",
                "transform": "constructor_factory_context_threading",
                "atomic": True,
                "admitted": True,
            },
            "sandbox": {
                "path": "sandbox/ctor.json",
                "enforced": True,
                "write_scope": ["src/app.py", "src/factory.py", "src/container.py"],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/ctor.json",
                "iterations": 1,
                "residual_findings": 0,
                "oscillating": False,
            },
        },
    },
    {
        "id": "adapter-schema-serializer-manifest-artifact",
        "scenario": "adapter_schema_serializer_manifest_artifact",
        "family": "positive_analytical",
        "expected": _expected(
            diagnosis="finite_schema_artifact",
            disposition="supported",
            repair="analytical",
            fixed_point="reached",
            completion="success",
            reason_codes=[
                "adapter_schema_rewritten",
                "serializer_and_manifest_aligned",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/schema.json",
                "kind": "schema_field_add",
                "before": "Message{id, body}",
                "after": "Message{id, body, tenant}",
                "artifacts": ["adapter", "schema", "serializer", "manifest"],
            },
            "consumers": {
                "path": "consumers/schema.json",
                "resolved": [
                    {"kind": "adapter", "site": "adapters/msg.py"},
                    {"kind": "schema", "site": "schemas/message.json"},
                    {"kind": "serializer", "site": "serdes/message.py"},
                    {"kind": "manifest", "site": "manifests/message.yaml"},
                    {"kind": "artifact", "site": "artifacts/message.capnp"},
                ],
                "obligations": 5,
                "mandatory_callers": 5,
            },
            "graph": {
                "path": "graph/schema.json",
                "edges": ["adapter", "schema", "serializer", "manifest", "artifact"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/schema.json",
                "candidates": [
                    {
                        "name": "tenant_id",
                        "proved": True,
                        "unique": True,
                        "kind": "schema_field",
                    }
                ],
            },
            "retrieval": {
                "path": "retrieval/schema.json",
                "vector_hits": [],
                "kg_edges": ["schema_edge"],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/schema.json",
                "verdict": "reconstructed",
                "operator": "finite_schema_artifact_repair",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/schema.json",
                "transform": "adapter_schema_serializer_manifest_artifact",
                "atomic": True,
                "admitted": True,
            },
            "sandbox": {
                "path": "sandbox/schema.json",
                "enforced": True,
                "write_scope": [
                    "adapters/msg.py",
                    "schemas/message.json",
                    "serdes/message.py",
                    "manifests/message.yaml",
                    "artifacts/message.capnp",
                ],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/schema.json",
                "iterations": 1,
                "residual_findings": 0,
                "oscillating": False,
            },
        },
    },
    # ------------------------------------------------------------------
    # Adversarial fail-closed controls
    # ------------------------------------------------------------------
    {
        "id": "same-type-wrong-value",
        "scenario": "same_type_wrong_value",
        "family": "adversarial_value",
        "expected": _expected(
            diagnosis="wrong_value",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["same_type_insufficient", "information_content_mismatch"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/wrong_value.json",
                "kind": "parameter_add",
                "before": "authorize(user: UserId) -> Token",
                "after": "authorize(user: UserId, session: SessionId) -> Token",
            },
            "consumers": {
                "path": "consumers/wrong_value.json",
                "resolved": [{"kind": "direct", "site": "src/auth.py:check", "args": 1}],
                "obligations": 1,
                "mandatory_callers": 1,
            },
            "graph": {
                "path": "graph/wrong_value.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/wrong_value.json",
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
            },
            "retrieval": {
                "path": "retrieval/wrong_value.json",
                "vector_hits": [{"id": "request_id", "score": 0.99}],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/wrong_value.json",
                "verdict": "refuted",
                "operator": None,
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/wrong_value.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/wrong_value.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/wrong_value.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "vector-collision",
        "scenario": "vector_collision",
        "family": "adversarial_retrieval",
        "expected": _expected(
            diagnosis="vector_collision",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["vector_collision_non_authoritative", "require_independent_proof"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/vector.json",
                "kind": "parameter_add",
                "before": "emit(event: Event)",
                "after": "emit(event: Event, tenant: TenantId)",
            },
            "consumers": {
                "path": "consumers/vector.json",
                "resolved": [{"kind": "direct", "site": "src/pipe.py:forward"}],
                "obligations": 1,
                "mandatory_callers": 1,
            },
            "graph": {
                "path": "graph/vector.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/vector.json",
                "candidates": [
                    {"name": "tenant_id", "proved": False, "unique": False},
                    {"name": "tenant_name", "proved": False, "unique": False},
                ],
            },
            "retrieval": {
                "path": "retrieval/vector.json",
                "vector_hits": [
                    {"id": "tenant_id", "score": 0.97, "collision": True},
                    {"id": "tenant_name", "score": 0.97, "collision": True},
                ],
                "kg_edges": [],
                "embedding_mode": "vector",
                "semantic_authority": False,
                "vector_promoted": False,
            },
            "proof": {
                "path": "proof/vector.json",
                "verdict": "ambiguous",
                "operator": None,
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/vector.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/vector.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/vector.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "kg-omission",
        "scenario": "kg_omission",
        "family": "adversarial_retrieval",
        "expected": _expected(
            diagnosis="kg_omission",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["kg_edge_missing", "kg_non_authoritative"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/kg.json",
                "kind": "parameter_add",
                "before": "ship(order: Order)",
                "after": "ship(order: Order, warehouse: WarehouseId)",
            },
            "consumers": {
                "path": "consumers/kg.json",
                "resolved": [{"kind": "direct", "site": "src/fulfill.py:dispatch"}],
                "obligations": 1,
                "mandatory_callers": 1,
            },
            "graph": {
                "path": "graph/kg.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/kg.json",
                "candidates": [],
            },
            "retrieval": {
                "path": "retrieval/kg.json",
                "vector_hits": [],
                "kg_edges": [],
                "kg_omitted_edge": "warehouse_binding",
                "embedding_mode": "exact",
                "semantic_authority": False,
                "kg_promoted": False,
            },
            "proof": {
                "path": "proof/kg.json",
                "verdict": "unsupported",
                "operator": None,
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/kg.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/kg.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/kg.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "constant-embedding-fallback",
        "scenario": "constant_embedding_fallback",
        "family": "adversarial_retrieval",
        "expected": _expected(
            diagnosis="embedding_degraded",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "constant_embedding_fallback_detected",
                "embedding_lane_disabled",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/embed.json",
                "kind": "parameter_add",
                "before": "route(msg: Msg)",
                "after": "route(msg: Msg, region: Region)",
            },
            "consumers": {
                "path": "consumers/embed.json",
                "resolved": [{"kind": "direct", "site": "src/router.py:send"}],
                "obligations": 1,
                "mandatory_callers": 1,
            },
            "graph": {
                "path": "graph/embed.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/embed.json",
                "candidates": [{"name": "region", "proved": False, "unique": False}],
            },
            "retrieval": {
                "path": "retrieval/embed.json",
                "vector_hits": [{"id": "any", "score": 0.0, "constant": True}],
                "kg_edges": [],
                "embedding_mode": "constant_fallback",
                "embedding_finite": False,
                "semantic_authority": False,
                "lane_disabled": True,
            },
            "proof": {
                "path": "proof/embed.json",
                "verdict": "unsupported",
                "operator": None,
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/embed.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/embed.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/embed.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "stale-corrupt-forged-cid-cache",
        "scenario": "stale_corrupt_forged_cid_cache",
        "family": "adversarial_cache",
        "expected": _expected(
            diagnosis="stale_forged_cache",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "stale_proof_cid_rejected",
                "forged_cache_entry_quarantined",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/stale.json",
                "kind": "parameter_add",
                "tree_id": "tree:current",
                "claimed_tree_id": "tree:stale",
                "before": "f(a)",
                "after": "f(a, b)",
            },
            "consumers": {
                "path": "consumers/stale.json",
                "resolved": [{"kind": "direct", "site": "src/x.py:call"}],
                "obligations": 1,
                "mandatory_callers": 1,
            },
            "graph": {
                "path": "graph/stale.json",
                "edges": ["direct"],
                "complete": True,
                "stale": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/stale.json",
                "candidates": [{"name": "b", "proved": True, "unique": True}],
            },
            "retrieval": {
                "path": "retrieval/stale.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/stale.json",
                "verdict": "stale",
                "operator": None,
                "cache_status": "forged",
                "claimed_cid": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "current_cid": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
            },
            "plan": {
                "path": "plan/stale.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/stale.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/stale.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "solver-lie-countermodel",
        "scenario": "solver_lie_countermodel",
        "family": "adversarial_proof",
        "expected": _expected(
            diagnosis="solver_lie",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "raw_countermodel_not_authoritative",
                "reconstruction_required",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/solver.json",
                "kind": "parameter_add",
                "before": "check(x)",
                "after": "check(x, y)",
            },
            "consumers": {
                "path": "consumers/solver.json",
                "resolved": [{"kind": "direct", "site": "src/check.py:run"}],
                "obligations": 1,
                "mandatory_callers": 1,
            },
            "graph": {
                "path": "graph/solver.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/solver.json",
                "candidates": [{"name": "y", "proved": False, "unique": False}],
            },
            "retrieval": {
                "path": "retrieval/solver.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/solver.json",
                "verdict": "raw_countermodel",
                "operator": None,
                "cache_status": "fresh",
                "solver_claimed_sat": True,
                "reconstructed": False,
                "validated_countermodel": False,
            },
            "plan": {
                "path": "plan/solver.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/solver.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/solver.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "incomplete-ast-impact-scc",
        "scenario": "incomplete_ast_impact_scc",
        "family": "adversarial_impact",
        "expected": _expected(
            diagnosis="incomplete_impact",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["incomplete_ast", "open_scc", "impact_not_closed"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/scc.json",
                "kind": "parameter_add",
                "before": "cycle_a(x)",
                "after": "cycle_a(x, y)",
            },
            "consumers": {
                "path": "consumers/scc.json",
                "resolved": [
                    {"kind": "direct", "site": "src/a.py:a"},
                    {"kind": "direct", "site": "src/b.py:b"},
                ],
                "obligations": 3,
                "mandatory_callers": 3,
                "missing_resolved": ["src/c.py:c"],
            },
            "graph": {
                "path": "graph/scc.json",
                "edges": ["a->b", "b->a"],
                "complete": False,
                "scc": ["a", "b", "c"],
                "unknown_frontier": ["c"],
            },
            "value_sources": {
                "path": "value_sources/scc.json",
                "candidates": [{"name": "y", "proved": False, "unique": False}],
            },
            "retrieval": {
                "path": "retrieval/scc.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/scc.json",
                "verdict": "unsupported",
                "operator": None,
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/scc.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
                "partial_allowed": False,
            },
            "sandbox": {
                "path": "sandbox/scc.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/scc.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "dynamic-generated-native-ffi-public-schema-cross-root",
        "scenario": "dynamic_generated_native_ffi_public_schema_cross_root",
        "family": "adversarial_frontier",
        "expected": _expected(
            diagnosis="open_frontier",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=[
                "dynamic_dispatch_frontier",
                "generated_client_frontier",
                "native_ffi_frontier",
                "public_schema_cross_root",
                "new_dependency_complex_behavior",
            ],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/frontier.json",
                "kind": "signature_change",
                "before": "invoke(cmd)",
                "after": "invoke(cmd, opts)",
                "cross_root": True,
                "new_dependency": "external.sdk.v2",
            },
            "consumers": {
                "path": "consumers/frontier.json",
                "resolved": [{"kind": "direct", "site": "src/local.py:call"}],
                "obligations": 1,
                "mandatory_callers": 1,
                "frontier": [
                    "dynamic_getattr",
                    "generated_sdk_client",
                    "native_extension",
                    "ffi_bridge",
                    "public_schema_consumer",
                    "cross_root_repo",
                ],
            },
            "graph": {
                "path": "graph/frontier.json",
                "edges": ["direct"],
                "complete": False,
                "unknown_frontier": [
                    "dynamic",
                    "generated",
                    "native",
                    "ffi",
                    "public_schema",
                    "cross_root",
                    "new_dependency",
                ],
            },
            "value_sources": {
                "path": "value_sources/frontier.json",
                "candidates": [],
                "complex_behavior_required": True,
            },
            "retrieval": {
                "path": "retrieval/frontier.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/frontier.json",
                "verdict": "unsupported",
                "operator": None,
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/frontier.json",
                "transform": None,
                "atomic": False,
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/frontier.json",
                "enforced": True,
                "write_scope": [],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/frontier.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "sandbox-escape",
        "scenario": "sandbox_escape",
        "family": "adversarial_sandbox",
        "expected": _expected(
            diagnosis="sandbox_escape",
            disposition="abstain",
            repair="none",
            fixed_point="incomplete",
            completion="fail_closed",
            reason_codes=["path_escape_rejected", "out_of_scope_write_blocked"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/escape.json",
                "kind": "parameter_add",
                "before": "write(path)",
                "after": "write(path, mode)",
            },
            "consumers": {
                "path": "consumers/escape.json",
                "resolved": [{"kind": "direct", "site": "src/io.py:write"}],
                "obligations": 1,
                "mandatory_callers": 1,
            },
            "graph": {
                "path": "graph/escape.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/escape.json",
                "candidates": [{"name": "mode", "proved": True, "unique": True}],
            },
            "retrieval": {
                "path": "retrieval/escape.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/escape.json",
                "verdict": "reconstructed",
                "operator": "add_argument_from_unique_source",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/escape.json",
                "transform": "add_argument_from_unique_source",
                "atomic": True,
                # Would be eligible analytically, but sandbox forbids write path.
                "admitted": False,
                "abstain": True,
            },
            "sandbox": {
                "path": "sandbox/escape.json",
                "enforced": True,
                "write_scope": ["src/io.py"],
                "escape_attempt": True,
                "escape_path": "../secrets/key.pem",
                "escape_blocked": True,
            },
            "fixed_point": {
                "path": "fixed_point/escape.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
            },
        },
    },
    {
        "id": "crash-rollback",
        "scenario": "crash_rollback",
        "family": "adversarial_transaction",
        "expected": _expected(
            diagnosis="crash_mid_transaction",
            disposition="rolled_back",
            repair="none",
            fixed_point="incomplete",
            completion="rollback",
            reason_codes=["crash_injected", "compensating_rollback", "no_partial_commit"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/crash.json",
                "kind": "parameter_add",
                "before": "batch(a, b)",
                "after": "batch(a, b, c)",
            },
            "consumers": {
                "path": "consumers/crash.json",
                "resolved": [
                    {"kind": "direct", "site": "src/a.py:a"},
                    {"kind": "direct", "site": "src/b.py:b"},
                ],
                "obligations": 2,
                "mandatory_callers": 2,
            },
            "graph": {
                "path": "graph/crash.json",
                "edges": ["direct"],
                "complete": True,
                "unknown_frontier": [],
            },
            "value_sources": {
                "path": "value_sources/crash.json",
                "candidates": [{"name": "c", "proved": True, "unique": True}],
            },
            "retrieval": {
                "path": "retrieval/crash.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/crash.json",
                "verdict": "reconstructed",
                "operator": "add_argument_from_unique_source",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/crash.json",
                "transform": "add_argument_from_unique_source",
                "atomic": True,
                "admitted": True,
                "partial_failure": True,
                "crash_after_first_file": True,
            },
            "sandbox": {
                "path": "sandbox/crash.json",
                "enforced": True,
                "write_scope": ["src/a.py", "src/b.py"],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/crash.json",
                "iterations": 0,
                "residual_findings": 1,
                "oscillating": False,
                "rollback_success": True,
                "partial_commit": False,
            },
        },
    },
    {
        "id": "oscillation",
        "scenario": "oscillation",
        "family": "adversarial_fixed_point",
        "expected": _expected(
            diagnosis="oscillation",
            disposition="abstain",
            repair="none",
            fixed_point="oscillating",
            completion="fail_closed",
            reason_codes=["oscillation_detected", "false_fixed_point_refused"],
            stages=DEFAULT_STAGES,
        ),
        "authority": _base_authority(),
        "artifacts": {
            "delta": {
                "path": "delta/osc.json",
                "kind": "parameter_add",
                "before": "ping(x)",
                "after": "ping(x, y)",
            },
            "consumers": {
                "path": "consumers/osc.json",
                "resolved": [
                    {"kind": "direct", "site": "src/ping.py:ping"},
                    {"kind": "direct", "site": "src/pong.py:pong"},
                ],
                "obligations": 2,
                "mandatory_callers": 2,
            },
            "graph": {
                "path": "graph/osc.json",
                "edges": ["ping->pong", "pong->ping"],
                "complete": True,
                "unknown_frontier": [],
                "post_repair_new_delta": True,
            },
            "value_sources": {
                "path": "value_sources/osc.json",
                "candidates": [{"name": "y", "proved": True, "unique": True}],
            },
            "retrieval": {
                "path": "retrieval/osc.json",
                "vector_hits": [],
                "kg_edges": [],
                "embedding_mode": "exact",
                "semantic_authority": False,
            },
            "proof": {
                "path": "proof/osc.json",
                "verdict": "reconstructed",
                "operator": "add_argument_from_unique_source",
                "cache_status": "fresh",
            },
            "plan": {
                "path": "plan/osc.json",
                "transform": "add_argument_from_unique_source",
                "atomic": True,
                "admitted": True,
            },
            "sandbox": {
                "path": "sandbox/osc.json",
                "enforced": True,
                "write_scope": ["src/ping.py", "src/pong.py"],
                "escape_attempt": False,
            },
            "fixed_point": {
                "path": "fixed_point/osc.json",
                "iterations": 3,
                "residual_findings": 1,
                "oscillating": True,
                "claimed_complete": False,
            },
        },
    },
]


def build_manifest() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    for recipe in RECIPES:
        scenario = str(recipe["scenario"])
        if scenario in seen_scenarios:
            raise ValueError(f"duplicate scenario: {scenario}")
        seen_scenarios.add(scenario)
        artifacts = {
            role: _artifact(recipe["artifacts"][role]) for role in ARTIFACT_ROLES
        }
        cases.append(
            {
                "id": recipe["id"],
                "scenario": scenario,
                "family": recipe["family"],
                "expected": recipe["expected"],
                "authority": recipe["authority"],
                "artifacts": artifacts,
            }
        )
    missing = set(REQUIRED_SCENARIOS) - seen_scenarios
    if missing:
        raise ValueError(f"missing required scenarios: {sorted(missing)}")
    extra = seen_scenarios - set(REQUIRED_SCENARIOS)
    if extra:
        raise ValueError(f"unexpected scenarios: {sorted(extra)}")
    return {
        "schema": SCHEMA,
        "corpus_id": CORPUS_ID,
        "description": DESCRIPTION,
        "artifact_roles": list(ARTIFACT_ROLES),
        "cases": cases,
    }


def main() -> None:
    manifest = build_manifest()
    out = Path(__file__).with_name("manifest.json")
    out.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out} with {len(manifest['cases'])} cases")


if __name__ == "__main__":
    main()
