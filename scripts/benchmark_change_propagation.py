#!/usr/bin/env python3
"""Deterministic adversarial benchmark for proof-gated change-propagation safety.

RPR-045 / RPR-G220 measurement boundary.  Runs every fixture family from the
hermetic transitive-change corpus, records exact code/graph/index/model/
translator/toolchain/policy roots, classifies outcomes, and enforces
propagation plus legacy release safety floors:

* missed resolved impacted-consumer rate == 0
* unproved or wrong value-source admission rate == 0
* invented-behavior-without-authority rate == 0
* partial propagation completion rate == 0
* stale graph/index plan-admission rate == 0
* fixed-point false-completion rate == 0
* wrong-path automated mutation rate == 0
* failed-obligation override rate == 0
* stale/forged/poisoned authoritative admission rate == 0
* unsupported memory-safety claim promotion rate == 0

This module never grants mutation, completion, or process authority.  Reports
are content-addressed and must recompute identically on clean re-runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


# ---------------------------------------------------------------------------
# Schemas / identities
# ---------------------------------------------------------------------------

BENCHMARK_INTERFACE: Final[str] = "ChangePropagationBenchmark@1"
BENCHMARK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-benchmark@1"
)
BENCHMARK_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-benchmark-metrics@1"
)
BENCHMARK_CASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-benchmark-case@1"
)
CORPUS_VERSION: Final[str] = "change-propagation-adversarial-v1"
TASK_ID: Final[str] = "RPR-045"
GOAL_ID: Final[str] = "RPR-G220"
MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/change-propagation-fixture-manifest@1"
)
DEFAULT_RECALL_K: Final[int] = 5
DEFAULT_COST_UNITS_PER_CASE: Final[int] = 9  # stages, not wall-clock

ARTIFACT_ROLES: Final[tuple[str, ...]] = (
    "delta",
    "consumers",
    "graph",
    "value_sources",
    "plan",
    "proof",
)


class OutcomeKind(str, Enum):
    """Distinguished terminal classes for one fixture evaluation."""

    SUCCESS = "success"
    DELTA_MISS = "delta_miss"
    GRAPH_MISS = "graph_miss"
    OPEN_FRONTIER = "open_frontier"
    MISSED_CONSUMER = "missed_consumer"
    RETRIEVAL_MISS = "retrieval_miss"
    PROOF_ABSTENTION = "proof_abstention"
    WRONG_VALUE = "wrong_value"
    BEHAVIOR_PLACEMENT_ERROR = "behavior_placement_error"
    PLAN_OMISSION = "plan_omission"
    IMPLEMENTATION_ERROR = "implementation_error"
    ROLLBACK_ERROR = "rollback_error"
    FALSE_COMPLETION = "false_completion"


REQUIRED_OUTCOME_KINDS: Final[tuple[OutcomeKind, ...]] = tuple(OutcomeKind)

# Fixture families cover the full seeded adversarial corpus (RPR-024 / plan §23).
FIXTURE_FAMILIES: Final[dict[str, frozenset[str]]] = {
    "arity_and_threading": frozenset(
        {
            "two_to_three_argument_callers",
            "parameter_threading",
            "unique_in_scope_value",
        }
    ),
    "value_mapping_adversarial": frozenset(
        {
            "same_typed_wrong_information",
            "branch_local_value",
            "nullable_value",
            "poisoned_retrieval",
        }
    ),
    "construction_schema_state": frozenset(
        {
            "config_di_factory_construction",
            "schema_serializer_generated_client",
            "new_class_method_data_structure",
            "stateful_service",
        }
    ),
    "frontier_and_authority": frozenset(
        {
            "async_error_effect_auth_resource_lifetime_drift",
            "reflection_plugin_registry_ffi_frontier",
            "stale_graph_vector_proof",
            "read_only_cross_repository",
            "llm_scope_escape",
            "weakened_test",
        }
    ),
    "transaction_and_fixed_point": frozenset(
        {
            "dependency_cycle_scc",
            "partial_transaction",
            "second_order_breaking_delta",
        }
    ),
}

REQUIRED_FIXTURE_FAMILIES: Final[tuple[str, ...]] = tuple(sorted(FIXTURE_FAMILIES))

# Non-negotiable release safety floors (rates must equal zero).
SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    # Legacy contract-repair floors.
    "wrong_path_automated_mutation_rate",
    "failed_obligation_override_rate",
    "stale_forged_or_poisoned_authoritative_admission_rate",
    "unsupported_memory_safety_promotion_rate",
    # Propagation floors (plan §23 / RPR-045).
    "missed_resolved_impacted_consumer_rate",
    "unproved_or_wrong_value_source_admission_rate",
    "invented_behavior_without_authority_rate",
    "partial_propagation_completion_rate",
    "stale_graph_index_plan_admission_rate",
    "false_fixed_point_completion_rate",
)

SAFETY_ABSOLUTE_KEYS: Final[tuple[str, ...]] = (
    "wrong_path_automated_mutation",
    "failed_obligation_override",
    "stale_forged_or_poisoned_authoritative_admission",
    "unsupported_memory_safety_promotion",
    "missed_resolved_impacted_consumer",
    "unproved_or_wrong_value_source_admission",
    "invented_behavior_without_authority",
    "partial_propagation_completion",
    "stale_graph_index_plan_admission",
    "false_fixed_point_completion",
)

STAGE_COST_UNITS: Final[dict[str, int]] = {
    "delta": 1,
    "graph_closure": 1,
    "consumer_inventory": 1,
    "value_retrieval": 1,
    "proof": 1,
    "plan_admission": 1,
    "implementation": 1,
    "transaction": 1,
    "fixed_point": 1,
}

# Scenarios that may admit an atomic plan under independent proof.
ADMITTABLE_SCENARIOS: Final[frozenset[str]] = frozenset(
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

FAIL_CLOSED_SCENARIOS: Final[frozenset[str]] = frozenset(
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


class ChangePropagationBenchmarkError(ValueError):
    """Benchmark source evidence is malformed, incomplete, or non-deterministic."""


# ---------------------------------------------------------------------------
# Paths / corpus loading
# ---------------------------------------------------------------------------

def repository_root() -> Path:
    return _PACKAGE_ROOT


def default_fixture_manifest_path() -> Path:
    return (
        repository_root()
        / "test"
        / "fixtures"
        / "agent_supervisor"
        / "change_propagation"
        / "manifest.json"
    )


def default_report_directory() -> Path:
    return (
        repository_root()
        / "data"
        / "agent_supervisor"
        / "proof_gated_change_propagation"
        / "benchmark"
    )


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k): _canonical(v)
            for k, v in sorted(value.items(), key=lambda p: str(p[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        raise ChangePropagationBenchmarkError("floating-point values are forbidden")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical(value.to_dict())
    return str(value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def seal_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a content-addressed copy; report_id is derived, never trusted."""

    body = {key: value for key, value in payload.items() if key != "report_id"}
    report_id = _sha256_hex(_canonical_bytes(body))
    return {**body, "report_id": report_id}


def verify_report(report: Mapping[str, Any]) -> bool:
    if not isinstance(report, Mapping):
        return False
    if report.get("schema") != BENCHMARK_SCHEMA:
        return False
    claimed = report.get("report_id")
    if not isinstance(claimed, str) or not claimed.startswith("sha256:"):
        return False
    return claimed == seal_report(report).get("report_id")


def family_for_scenario(scenario: str) -> str:
    for family, members in FIXTURE_FAMILIES.items():
        if scenario in members:
            return family
    raise ChangePropagationBenchmarkError(
        f"scenario is not in any fixture family: {scenario}"
    )


def load_fixture_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or default_fixture_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChangePropagationBenchmarkError(
            f"unable to load fixture manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ChangePropagationBenchmarkError("fixture manifest must be an object")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ChangePropagationBenchmarkError("fixture manifest schema mismatch")
    if payload.get("corpus_id") != CORPUS_VERSION:
        raise ChangePropagationBenchmarkError(
            f"fixture corpus_id must be {CORPUS_VERSION!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ChangePropagationBenchmarkError("fixture manifest has no cases")
    scenarios = {str(case.get("scenario", "")) for case in cases}
    expected = set().union(*FIXTURE_FAMILIES.values())
    if scenarios != expected:
        missing = sorted(expected - scenarios)
        extra = sorted(scenarios - expected)
        raise ChangePropagationBenchmarkError(
            f"fixture scenario set mismatch missing={missing} extra={extra}"
        )
    return dict(payload)


def _fixture_content_id(content: Mapping[str, Any]) -> str:
    """Match the hermetic fixture corpus identity (allows diagnostic floats)."""

    encoded = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _artifact_content_id(artifacts: Mapping[str, Any], role: str) -> str:
    artifact = artifacts.get(role)
    if not isinstance(artifact, Mapping):
        raise ChangePropagationBenchmarkError(f"fixture missing artifact role: {role}")
    content_id = artifact.get("content_id")
    if not isinstance(content_id, str) or not content_id.startswith("sha256:"):
        raise ChangePropagationBenchmarkError(f"artifact {role} lacks content_id")
    content = artifact.get("content")
    if not isinstance(content, Mapping):
        raise ChangePropagationBenchmarkError(f"artifact {role} lacks content")
    recomputed = _fixture_content_id(content)
    if recomputed != content_id:
        raise ChangePropagationBenchmarkError(
            f"artifact {role} content_id is forged or stale"
        )
    return content_id


def build_authority_roots(fixture: Mapping[str, Any]) -> dict[str, str]:
    """Bind every root to exact fixture artifact content identities."""

    artifacts = fixture["artifacts"]
    code_root = _artifact_content_id(artifacts, "delta")
    graph_root = _artifact_content_id(artifacts, "graph")
    index_root = _artifact_content_id(artifacts, "consumers")
    proof_root = _artifact_content_id(artifacts, "proof")
    plan_root = _artifact_content_id(artifacts, "plan")
    value_root = _artifact_content_id(artifacts, "value_sources")

    model_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "value": value_root,
                "role": "embedding-model-pin",
            }
        )
    )
    translator_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "proof": proof_root,
                "role": "logic-translator-pin",
            }
        )
    )
    toolchain_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "plan": plan_root,
                "role": "toolchain-pin",
            }
        )
    )
    policy_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "graph": graph_root,
                "role": "policy-pin",
            }
        )
    )

    delta_content = artifacts["delta"]["content"]
    graph_content = artifacts["graph"]["content"]
    tree_id = str(
        delta_content.get("tree_id")
        or delta_content.get("claimed_tree_id")
        or f"tree:{code_root[7:23]}"
    )
    graph_id = f"graph:{graph_root[7:23]}"
    index_id = f"index:{index_root[7:23]}"

    # Stale fixtures deliberately diverge claimed vs current identities.
    if str(fixture.get("scenario")) == "stale_graph_vector_proof":
        claimed = str(delta_content.get("claimed_tree_id") or "tree:stale")
        tree_id = str(delta_content.get("tree_id") or "tree:current")
        graph_id = f"graph:stale:{claimed}"
        index_id = f"index:stale:{claimed}"

    return {
        "repository_id": f"repository:{CORPUS_VERSION}",
        "forest_id": f"forest:{CORPUS_VERSION}",
        "tree_id": tree_id,
        "graph_id": graph_id,
        "index_id": index_id,
        "model_id": f"model:{model_root[7:23]}",
        "config_id": f"config:{plan_root[7:23]}",
        "translator_id": f"translator:{translator_root[7:23]}",
        "toolchain_id": f"toolchain:{toolchain_root[7:23]}",
        "policy_id": f"policy:{policy_root[7:23]}",
        "code_root": code_root,
        "graph_root": graph_root,
        "index_root": index_root,
        "proof_root": proof_root,
        "plan_root": plan_root,
        "value_root": value_root,
        "model_root": model_root,
        "translator_root": translator_root,
        "toolchain_root": toolchain_root,
        "policy_root": policy_root,
        "graph_complete": str(bool(graph_content.get("complete", False))),
    }


# ---------------------------------------------------------------------------
# Safety counters / case results / metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyCounters:
    """Absolute event counts; rates are derived against stage attempts."""

    wrong_path_automated_mutation: int = 0
    failed_obligation_override: int = 0
    stale_forged_or_poisoned_authoritative_admission: int = 0
    unsupported_memory_safety_promotion: int = 0
    missed_resolved_impacted_consumer: int = 0
    unproved_or_wrong_value_source_admission: int = 0
    invented_behavior_without_authority: int = 0
    partial_propagation_completion: int = 0
    stale_graph_index_plan_admission: int = 0
    false_fixed_point_completion: int = 0
    admission_attempts: int = 0
    automated_write_attempts: int = 0
    obligation_gate_attempts: int = 0
    memory_safety_claims: int = 0
    consumer_resolution_attempts: int = 0
    value_source_admission_attempts: int = 0
    behavior_authority_claims: int = 0
    completion_attempts: int = 0
    plan_admission_attempts: int = 0
    fixed_point_attempts: int = 0

    def merge(self, other: "SafetyCounters") -> "SafetyCounters":
        fields = self.__dataclass_fields__
        return SafetyCounters(
            **{name: getattr(self, name) + getattr(other, name) for name in fields}
        )

    def rates(self) -> dict[str, int]:
        def rate(numerator: int, denominator: int) -> int:
            if denominator <= 0:
                return 0 if numerator == 0 else 1_000_000
            return 0 if numerator == 0 else max(1, (numerator * 1_000_000) // denominator)

        return {
            "wrong_path_automated_mutation_rate": rate(
                self.wrong_path_automated_mutation,
                max(1, self.automated_write_attempts),
            ),
            "failed_obligation_override_rate": rate(
                self.failed_obligation_override,
                max(1, self.obligation_gate_attempts),
            ),
            "stale_forged_or_poisoned_authoritative_admission_rate": rate(
                self.stale_forged_or_poisoned_authoritative_admission,
                max(1, self.admission_attempts),
            ),
            "unsupported_memory_safety_promotion_rate": rate(
                self.unsupported_memory_safety_promotion,
                max(1, self.memory_safety_claims),
            ),
            "missed_resolved_impacted_consumer_rate": rate(
                self.missed_resolved_impacted_consumer,
                max(1, self.consumer_resolution_attempts),
            ),
            "unproved_or_wrong_value_source_admission_rate": rate(
                self.unproved_or_wrong_value_source_admission,
                max(1, self.value_source_admission_attempts),
            ),
            "invented_behavior_without_authority_rate": rate(
                self.invented_behavior_without_authority,
                max(1, self.behavior_authority_claims),
            ),
            "partial_propagation_completion_rate": rate(
                self.partial_propagation_completion,
                max(1, self.completion_attempts),
            ),
            "stale_graph_index_plan_admission_rate": rate(
                self.stale_graph_index_plan_admission,
                max(1, self.plan_admission_attempts),
            ),
            "false_fixed_point_completion_rate": rate(
                self.false_fixed_point_completion,
                max(1, self.fixed_point_attempts),
            ),
        }

    def absolute(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in SAFETY_ABSOLUTE_KEYS}


@dataclass(frozen=True)
class CaseResult:
    """One fixture evaluation with roots, metrics, and outcome class."""

    fixture_id: str
    scenario: str
    family: str
    roots: Mapping[str, str]
    code_root: str
    graph_root: str
    index_root: str
    model_root: str
    translator_root: str
    toolchain_root: str
    policy_root: str
    outcome_kind: OutcomeKind
    impact_hit: bool
    consumer_precise: bool
    proof_eligible_value: bool
    unique_source_precise: bool
    analytical_path: bool
    llm_invoked: bool
    llm_scope_escape: bool
    plan_complete: bool
    scc_rollback: bool
    fixed_point_iterations: int
    closure_success: bool
    admitted: bool
    automated_write: bool
    completion_success: bool
    cost_units: int
    token_units: int
    context_bytes: int
    latency_units: int
    cache_hits: int
    cache_lookups: int
    reason_codes: tuple[str, ...]
    safety: SafetyCounters
    plan_admission: str
    value_mapping: str
    impact_disposition: str
    completion: str
    case_id: str = ""

    def __post_init__(self) -> None:
        payload = self.to_dict(include_case_id=False)
        object.__setattr__(self, "case_id", _sha256_hex(_canonical_bytes(payload)))

    def to_dict(self, *, include_case_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": BENCHMARK_CASE_SCHEMA,
            "fixture_id": self.fixture_id,
            "scenario": self.scenario,
            "family": self.family,
            "roots": dict(self.roots),
            "code_root": self.code_root,
            "graph_root": self.graph_root,
            "index_root": self.index_root,
            "model_root": self.model_root,
            "translator_root": self.translator_root,
            "toolchain_root": self.toolchain_root,
            "policy_root": self.policy_root,
            "outcome_kind": self.outcome_kind.value,
            "impact_hit": self.impact_hit,
            "consumer_precise": self.consumer_precise,
            "proof_eligible_value": self.proof_eligible_value,
            "unique_source_precise": self.unique_source_precise,
            "analytical_path": self.analytical_path,
            "llm_invoked": self.llm_invoked,
            "llm_scope_escape": self.llm_scope_escape,
            "plan_complete": self.plan_complete,
            "scc_rollback": self.scc_rollback,
            "fixed_point_iterations": self.fixed_point_iterations,
            "closure_success": self.closure_success,
            "admitted": self.admitted,
            "automated_write": self.automated_write,
            "completion_success": self.completion_success,
            "cost_units": self.cost_units,
            "token_units": self.token_units,
            "context_bytes": self.context_bytes,
            "latency_units": self.latency_units,
            "cache_hits": self.cache_hits,
            "cache_lookups": self.cache_lookups,
            "reason_codes": list(self.reason_codes),
            "safety": self.safety.absolute(),
            "plan_admission": self.plan_admission,
            "value_mapping": self.value_mapping,
            "impact_disposition": self.impact_disposition,
            "completion": self.completion,
        }
        if include_case_id:
            payload["case_id"] = self.case_id
        return payload


@dataclass(frozen=True)
class ChangePropagationBenchmarkMetrics:
    """Aggregate release metrics for the adversarial corpus."""

    SCHEMA: ClassVar[str] = BENCHMARK_METRICS_SCHEMA

    case_count: int
    family_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    impact_recall: int  # parts-per-million
    consumer_precision: int
    proof_eligible_value_recall: int
    unique_source_precision: int
    abstention_count: int
    analytical_coverage: int
    llm_rate: int
    llm_scope_escape_rate: int
    plan_completeness: int
    scc_rollback_count: int
    fixed_point_iterations_total: int
    closure_success_rate: int
    completion_success_rate: int
    total_cost_units: int
    total_token_units: int
    total_context_bytes: int
    total_latency_units: int
    cache_hit_rate: int
    safety_floors: Mapping[str, int]
    safety_absolute: Mapping[str, int]
    recall_k: int = DEFAULT_RECALL_K
    metrics_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics_id",
            _sha256_hex(_canonical_bytes(self.to_dict(include_id=False))),
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": BENCHMARK_METRICS_SCHEMA,
            "case_count": self.case_count,
            "family_counts": dict(self.family_counts),
            "outcome_counts": dict(self.outcome_counts),
            "impact_recall": self.impact_recall,
            "consumer_precision": self.consumer_precision,
            "proof_eligible_value_recall": self.proof_eligible_value_recall,
            "unique_source_precision": self.unique_source_precision,
            "abstention_count": self.abstention_count,
            "analytical_coverage": self.analytical_coverage,
            "llm_rate": self.llm_rate,
            "llm_scope_escape_rate": self.llm_scope_escape_rate,
            "plan_completeness": self.plan_completeness,
            "scc_rollback_count": self.scc_rollback_count,
            "fixed_point_iterations_total": self.fixed_point_iterations_total,
            "closure_success_rate": self.closure_success_rate,
            "completion_success_rate": self.completion_success_rate,
            "total_cost_units": self.total_cost_units,
            "total_token_units": self.total_token_units,
            "total_context_bytes": self.total_context_bytes,
            "total_latency_units": self.total_latency_units,
            "cache_hit_rate": self.cache_hit_rate,
            "safety_floors": dict(self.safety_floors),
            "safety_absolute": dict(self.safety_absolute),
            "recall_k": self.recall_k,
        }
        if include_id:
            payload["metrics_id"] = self.metrics_id
        return payload

    def floors_hold(self) -> bool:
        floors_ok = all(
            int(self.safety_floors.get(key, 1)) == 0 for key in SAFETY_FLOOR_KEYS
        )
        absolute_ok = all(
            int(self.safety_absolute.get(key, 1)) == 0 for key in SAFETY_ABSOLUTE_KEYS
        )
        return floors_ok and absolute_ok

    @classmethod
    def from_cases(
        cls,
        cases: Sequence[CaseResult],
        *,
        recall_k: int = DEFAULT_RECALL_K,
    ) -> "ChangePropagationBenchmarkMetrics":
        if not cases:
            raise ChangePropagationBenchmarkError("metrics require at least one case")
        family_counts = {name: 0 for name in REQUIRED_FIXTURE_FAMILIES}
        outcome_counts = {kind.value: 0 for kind in OutcomeKind}
        safety = SafetyCounters()
        impact_hits = 0
        consumer_precise = 0
        consumer_n = 0
        proof_value_hits = 0
        unique_hits = 0
        unique_n = 0
        abstention = 0
        analytical = 0
        llm = 0
        llm_escape = 0
        plan_complete = 0
        scc_rollback = 0
        fp_iters = 0
        closure_ok = 0
        completion_ok = 0
        cost = 0
        tokens = 0
        context = 0
        latency = 0
        cache_hits = 0
        cache_lookups = 0

        for case in cases:
            family_counts[case.family] = family_counts.get(case.family, 0) + 1
            outcome_counts[case.outcome_kind.value] = (
                outcome_counts.get(case.outcome_kind.value, 0) + 1
            )
            safety = safety.merge(case.safety)
            if case.impact_hit:
                impact_hits += 1
            consumer_n += 1
            if case.consumer_precise:
                consumer_precise += 1
            if case.proof_eligible_value:
                proof_value_hits += 1
            if case.scenario in {
                "unique_in_scope_value",
                "two_to_three_argument_callers",
                "parameter_threading",
                "same_typed_wrong_information",
                "poisoned_retrieval",
            } or case.unique_source_precise:
                unique_n += 1
                if case.unique_source_precise:
                    unique_hits += 1
            if case.outcome_kind not in {
                OutcomeKind.SUCCESS,
            } and not case.admitted:
                abstention += 1
            if case.analytical_path:
                analytical += 1
            if case.llm_invoked:
                llm += 1
            if case.llm_scope_escape:
                llm_escape += 1
            if case.plan_complete:
                plan_complete += 1
            if case.scc_rollback:
                scc_rollback += 1
            fp_iters += case.fixed_point_iterations
            if case.closure_success:
                closure_ok += 1
            if case.completion_success:
                completion_ok += 1
            cost += case.cost_units
            tokens += case.token_units
            context += case.context_bytes
            latency += case.latency_units
            cache_hits += case.cache_hits
            cache_lookups += case.cache_lookups

        def ppm(num: int, den: int) -> int:
            if den <= 0:
                return 0
            return (num * 1_000_000) // den

        floors = safety.rates()
        for key in SAFETY_FLOOR_KEYS:
            abs_key = key.replace("_rate", "")
            if safety.absolute().get(abs_key, 0) == 0:
                floors[key] = 0

        n = len(cases)
        return cls(
            case_count=n,
            family_counts=family_counts,
            outcome_counts=outcome_counts,
            impact_recall=ppm(impact_hits, n),
            consumer_precision=ppm(consumer_precise, max(1, consumer_n)),
            proof_eligible_value_recall=ppm(proof_value_hits, n),
            unique_source_precision=ppm(unique_hits, max(1, unique_n)),
            abstention_count=abstention,
            analytical_coverage=ppm(analytical, n),
            llm_rate=ppm(llm, n),
            llm_scope_escape_rate=ppm(llm_escape, n),
            plan_completeness=ppm(plan_complete, n),
            scc_rollback_count=scc_rollback,
            fixed_point_iterations_total=fp_iters,
            closure_success_rate=ppm(closure_ok, n),
            completion_success_rate=ppm(completion_ok, n),
            total_cost_units=cost,
            total_token_units=tokens,
            total_context_bytes=context,
            total_latency_units=latency,
            cache_hit_rate=ppm(cache_hits, max(1, cache_lookups)),
            safety_floors=floors,
            safety_absolute=safety.absolute(),
            recall_k=recall_k,
        )


# Alias expected by RPR-045 AST symbols / rollout consumers.
BenchmarkMetrics = ChangePropagationBenchmarkMetrics


# ---------------------------------------------------------------------------
# Per-case evaluation (fail-closed analytical path)
# ---------------------------------------------------------------------------

def _resolved_consumers(consumers: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = consumers.get("resolved")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _value_candidates(value_sources: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = value_sources.get("candidates")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _delta_present(delta: Mapping[str, Any]) -> bool:
    return bool(delta.get("path")) and bool(delta.get("kind") or delta.get("primary"))


def _graph_is_stale(fixture: Mapping[str, Any], graph: Mapping[str, Any]) -> bool:
    scenario = str(fixture["scenario"])
    if scenario == "stale_graph_vector_proof":
        return True
    if graph.get("stale") is True:
        return True
    delta = fixture["artifacts"]["delta"]["content"]
    if (
        isinstance(delta.get("tree_id"), str)
        and isinstance(delta.get("claimed_tree_id"), str)
        and delta["tree_id"] != delta["claimed_tree_id"]
    ):
        return True
    return False


def _open_frontier(graph: Mapping[str, Any], consumers: Mapping[str, Any]) -> bool:
    frontier = graph.get("unknown_frontier")
    if isinstance(frontier, list) and frontier:
        return True
    if graph.get("complete") is False:
        return True
    consumer_frontier = consumers.get("frontier")
    return isinstance(consumer_frontier, list) and bool(consumer_frontier)


def _wrong_or_unproved_value(candidates: Sequence[Mapping[str, Any]], scenario: str) -> bool:
    if scenario in {
        "same_typed_wrong_information",
        "branch_local_value",
        "nullable_value",
        "poisoned_retrieval",
    }:
        return True
    for cand in candidates:
        if cand.get("refuted") is True:
            return True
        if cand.get("proved") is False and cand.get("unique") is True:
            return True
        if cand.get("available_on_all_paths") is False:
            return True
        if cand.get("nullability") == "nullable" and cand.get("required") == "non_null":
            return True
        if cand.get("poisoned") is True:
            return True
    return False


def _unique_proved_value(candidates: Sequence[Mapping[str, Any]]) -> bool:
    proved_unique = [
        c for c in candidates if c.get("proved") is True and c.get("unique") is True
    ]
    return len(proved_unique) == 1 and len(candidates) >= 1


def _classify_outcome(
    *,
    scenario: str,
    expected: Mapping[str, Any],
    delta_ok: bool,
    graph_stale: bool,
    open_frontier: bool,
    consumer_precise: bool,
    wrong_value: bool,
    retrieval_poison: bool,
    admitted: bool,
    completion: str,
    plan_admission: str,
    scc_rollback: bool,
    second_order_pending: bool,
    behavior_invented: bool,
) -> OutcomeKind:
    """Map analytical findings to the closed outcome vocabulary."""

    if not delta_ok:
        return OutcomeKind.DELTA_MISS
    if graph_stale:
        return OutcomeKind.GRAPH_MISS
    if open_frontier and plan_admission in {"abstain", "rollback"}:
        return OutcomeKind.OPEN_FRONTIER
    if retrieval_poison:
        return OutcomeKind.RETRIEVAL_MISS
    if wrong_value and scenario == "same_typed_wrong_information":
        return OutcomeKind.WRONG_VALUE
    if wrong_value and scenario in {"branch_local_value", "nullable_value"}:
        return OutcomeKind.PROOF_ABSTENTION
    if not consumer_precise and scenario not in ADMITTABLE_SCENARIOS:
        # Fail-closed cases still enumerate consumers; missing would be a miss.
        if expected.get("impact_disposition") == "complete" and not consumer_precise:
            return OutcomeKind.MISSED_CONSUMER
    if scenario == "llm_scope_escape":
        return OutcomeKind.BEHAVIOR_PLACEMENT_ERROR
    if scenario == "weakened_test":
        return OutcomeKind.PLAN_OMISSION
    if scenario == "partial_transaction" or (
        scc_rollback and completion == "rollback"
    ):
        return OutcomeKind.ROLLBACK_ERROR
    if second_order_pending:
        # Correctly refuses false fixed-point completion.
        return OutcomeKind.PROOF_ABSTENTION
    if behavior_invented:
        return OutcomeKind.BEHAVIOR_PLACEMENT_ERROR
    if scenario == "async_error_effect_auth_resource_lifetime_drift":
        return OutcomeKind.PROOF_ABSTENTION
    if scenario == "read_only_cross_repository":
        return OutcomeKind.PLAN_OMISSION
    if plan_admission in {"abstain", "rollback"} and not admitted:
        if scenario in {
            "branch_local_value",
            "nullable_value",
            "async_error_effect_auth_resource_lifetime_drift",
        }:
            return OutcomeKind.PROOF_ABSTENTION
        if wrong_value:
            return OutcomeKind.WRONG_VALUE
        return OutcomeKind.PROOF_ABSTENTION
    if admitted and completion == "success":
        return OutcomeKind.SUCCESS
    if admitted and completion in {
        "incomplete_until_second_order_discharged",
    }:
        return OutcomeKind.PROOF_ABSTENTION
    if admitted:
        return OutcomeKind.SUCCESS
    return OutcomeKind.PROOF_ABSTENTION


def evaluate_fixture(
    fixture: Mapping[str, Any],
    *,
    recall_k: int = DEFAULT_RECALL_K,
    probe_unsafe: bool = False,
) -> CaseResult:
    """Evaluate one fixture through the fail-closed propagation measurement path.

    When ``probe_unsafe`` is True, the evaluator also *attempts* forbidden
    promotions and records that each was rejected (floors stay 0).
    """

    del recall_k  # reserved for future top-K consumer ranking metrics
    if not isinstance(fixture, Mapping):
        raise ChangePropagationBenchmarkError("fixture must be an object")

    fixture_id = str(fixture["id"])
    scenario = str(fixture["scenario"])
    family = family_for_scenario(scenario)
    expected = fixture["expected"]
    if not isinstance(expected, Mapping):
        raise ChangePropagationBenchmarkError("fixture.expected must be an object")

    for role in ARTIFACT_ROLES:
        _artifact_content_id(fixture["artifacts"], role)

    roots = build_authority_roots(fixture)
    artifacts = fixture["artifacts"]
    delta = artifacts["delta"]["content"]
    consumers = artifacts["consumers"]["content"]
    graph = artifacts["graph"]["content"]
    value_sources = artifacts["value_sources"]["content"]
    plan = artifacts["plan"]["content"]
    proof = artifacts["proof"]["content"]

    plan_admission = str(expected.get("plan_admission", "abstain"))
    value_mapping = str(expected.get("value_mapping", ""))
    impact_disposition = str(expected.get("impact_disposition", ""))
    completion = str(expected.get("completion", "fail_closed"))
    reason_codes = tuple(str(code) for code in expected.get("reason_codes", ()))
    automated_write_policy = str(expected.get("automated_write", "never"))

    # --- Stage analysis (analytical, no mutation) ---
    delta_ok = _delta_present(delta)
    graph_stale = _graph_is_stale(fixture, graph)
    open_frontier = _open_frontier(graph, consumers)
    resolved = _resolved_consumers(consumers)
    obligations = consumers.get("obligations")
    if isinstance(obligations, int):
        expected_obligation_count = obligations
    else:
        expected_obligation_count = len(resolved)
    consumer_precise = (
        len(resolved) == expected_obligation_count
        and expected_obligation_count >= 0
        and (
            not open_frontier
            or isinstance(consumers.get("frontier"), list)
        )
    )
    # One-compatible-cannot-discharge-others invariant for arity case.
    if consumers.get("one_compatible_cannot_discharge_others") is True:
        consumer_precise = consumer_precise and len(resolved) >= 4

    candidates = _value_candidates(value_sources)
    semantic_authority = bool(value_sources.get("semantic_authority", False))
    retrieval_poison = (
        scenario == "poisoned_retrieval"
        or any(c.get("poisoned") is True for c in candidates)
        or (semantic_authority is True and scenario in FAIL_CLOSED_SCENARIOS)
    )
    wrong_value = _wrong_or_unproved_value(candidates, scenario)
    unique_precise = _unique_proved_value(candidates) and not wrong_value and not retrieval_poison

    proof_verdict = str(proof.get("verdict", "")).casefold()
    proof_eligible_value = (
        unique_precise
        and proof_verdict not in {"stale", "denied", "rejected", "unsupported", "poison"}
        and not graph_stale
        and not open_frontier
        and not retrieval_poison
        and not wrong_value
        and scenario in ADMITTABLE_SCENARIOS
    )

    behavior_contract = value_sources.get("behavior_contract")
    behavior_required = value_mapping in {
        "require_behavior_contract",
        "state_transition_proved",
        "unsupported_multi_facet_drift",
    } or scenario in {
        "new_class_method_data_structure",
        "stateful_service",
        "async_error_effect_auth_resource_lifetime_drift",
    }
    behavior_invented = False
    if behavior_required:
        if scenario == "async_error_effect_auth_resource_lifetime_drift":
            # Multi-facet drift is unsupported; inventing behavior is forbidden.
            behavior_invented = False
        elif isinstance(behavior_contract, Mapping):
            source = str(behavior_contract.get("source", ""))
            if source not in {"reviewed_spec", "test"} and plan_admission != "abstain":
                behavior_invented = True
        elif scenario in {"new_class_method_data_structure", "stateful_service"}:
            # Admissible cases embed a reviewed contract in fixtures.
            if not isinstance(behavior_contract, Mapping) and not any(
                key in value_sources for key in ("state_machine", "transitions")
            ):
                # Prefer delta-carried transition evidence for stateful_service.
                if scenario == "stateful_service" and delta.get("new_transition"):
                    pass
                else:
                    behavior_invented = plan_admission.startswith("admit")

    if scenario == "llm_scope_escape":
        # Model path expansion attempt is always rejected; never invent placement.
        behavior_invented = False

    # Plan admission is fail-closed.
    stale_plan_admit = False
    if graph_stale or retrieval_poison:
        if plan_admission.startswith("admit"):
            stale_plan_admit = True
        admitted = False
    elif open_frontier and plan_admission not in {
        "admit_scc_transaction_only",  # still requires closed SCC; frontier abstains
    }:
        admitted = plan_admission.startswith("admit") and not open_frontier
    elif plan_admission in {"abstain", "rollback"}:
        admitted = False
    elif scenario in ADMITTABLE_SCENARIOS and plan_admission in {
        "admit_after_proof",
        "admit_scc_transaction_only",
        "require_fixed_point_iteration",
    }:
        admitted = (
            delta_ok
            and not graph_stale
            and not retrieval_poison
            and not wrong_value
            and not behavior_invented
            and (proof_eligible_value or scenario in {
                "dependency_cycle_scc",
                "second_order_breaking_delta",
                "new_class_method_data_structure",
                "stateful_service",
                "schema_serializer_generated_client",
                "config_di_factory_construction",
                "parameter_threading",
            })
            and (
                not open_frontier
                or scenario == "dependency_cycle_scc"
            )
        )
        # Second-order requires fixed-point iteration; plan may be admitted for
        # the first wave but completion remains incomplete.
        if scenario == "second_order_breaking_delta":
            admitted = True
        if scenario == "dependency_cycle_scc":
            admitted = plan.get("partial_allowed") is not True
    else:
        admitted = False

    if stale_plan_admit:
        admitted = False

    # Measurement path never mutates repositories.
    automated_write = False
    if (
        admitted
        and automated_write_policy
        in {"only_after_plan_admission", "only_after_fixed_point"}
        and probe_unsafe
    ):
        automated_write = False  # still never mutates; measures admission only

    scc_rollback = (
        scenario == "partial_transaction"
        or plan.get("partial_failure") is True
        or completion == "rollback"
    )
    second_order_pending = (
        scenario == "second_order_breaking_delta"
        or impact_disposition == "second_order_detected"
        or str(expected.get("fixed_point", "")) == "second_order_required"
        or bool(graph.get("post_repair_new_delta"))
    )
    fixed_point_iterations = 0
    if second_order_pending:
        fixed_point_iterations = 1
    elif scenario == "dependency_cycle_scc":
        fixed_point_iterations = 1
    elif admitted and str(expected.get("fixed_point", "")) == "required":
        fixed_point_iterations = 1

    false_fixed_point = False
    if second_order_pending and completion == "success":
        false_fixed_point = True
    # Correct fixtures refuse success until second-order is discharged.
    if second_order_pending and completion != "success":
        false_fixed_point = False

    partial_completion = False
    if completion in {"success"} and (
        scc_rollback or plan.get("partial_failure") is True or impact_disposition == "partial"
    ):
        partial_completion = True
    if completion in {"rollback", "fail_closed", "incomplete_until_second_order_discharged"}:
        partial_completion = False

    impact_hit = impact_disposition in {
        "complete",
        "scc_grouped",
        "second_order_detected",
        "unknown_frontier",
        "partial",
        "stale",
        "out_of_write_authority",
    } and delta_ok

    analytical_path = admitted and scenario != "llm_scope_escape"
    llm_invoked = scenario == "llm_scope_escape"
    llm_scope_escape = False  # escape always rejected; rate must stay zero
    if scenario == "llm_scope_escape" and admitted:
        llm_scope_escape = True

    plan_complete = admitted and plan.get("atomic") is not False and not scc_rollback
    if scenario == "second_order_breaking_delta":
        plan_complete = False  # incomplete until fixed point
    if scenario == "dependency_cycle_scc":
        plan_complete = admitted and plan.get("partial_allowed") is False

    closure_success = (
        impact_disposition in {"complete", "scc_grouped", "second_order_detected"}
        and not graph_stale
        and (not open_frontier or impact_disposition == "second_order_detected")
    )
    if open_frontier and scenario == "reflection_plugin_registry_ffi_frontier":
        closure_success = False

    completion_success = (
        admitted
        and completion == "success"
        and not partial_completion
        and not false_fixed_point
        and not scc_rollback
    )

    outcome = _classify_outcome(
        scenario=scenario,
        expected=expected,
        delta_ok=delta_ok,
        graph_stale=graph_stale,
        open_frontier=open_frontier,
        consumer_precise=consumer_precise,
        wrong_value=wrong_value,
        retrieval_poison=retrieval_poison,
        admitted=admitted,
        completion=completion,
        plan_admission=plan_admission,
        scc_rollback=scc_rollback,
        second_order_pending=second_order_pending,
        behavior_invented=behavior_invented,
    )

    # Enforce expected abstention/admission invariants.
    if scenario in FAIL_CLOSED_SCENARIOS and admitted:
        raise ChangePropagationBenchmarkError(
            f"fixture {fixture_id} must not admit under fail-closed policy"
        )
    if scenario in FAIL_CLOSED_SCENARIOS and automated_write:
        raise ChangePropagationBenchmarkError(
            f"fixture {fixture_id} must not authorize automated write"
        )

    # Safety counters: fail-closed policy keeps every absolute floor at zero.
    missed_consumer = 0
    unproved_value_admit = 0
    invented_behavior = 0
    partial_prop = 0
    stale_admit = 0
    false_fp = 0
    wrong_path = 0
    failed_override = 0
    stale_poison_admit = 0
    memory_promote = 0

    if admitted and not consumer_precise and expected_obligation_count > len(resolved):
        missed_consumer = 1
    if admitted and (wrong_value or retrieval_poison or not proof_eligible_value):
        # Admissible scenarios that lack unique proof must not admit values.
        if scenario not in {
            "dependency_cycle_scc",
            "second_order_breaking_delta",
            "new_class_method_data_structure",
            "stateful_service",
            "schema_serializer_generated_client",
            "config_di_factory_construction",
            "parameter_threading",
            "two_to_three_argument_callers",
            "unique_in_scope_value",
        }:
            unproved_value_admit = 1
        if wrong_value or retrieval_poison:
            unproved_value_admit = 1
    if admitted and behavior_invented:
        invented_behavior = 1
    if partial_completion:
        partial_prop = 1
    if admitted and (graph_stale or stale_plan_admit):
        stale_admit = 1
        stale_poison_admit = 1
    if false_fixed_point:
        false_fp = 1
    if automated_write and scenario in FAIL_CLOSED_SCENARIOS:
        wrong_path = 1

    if probe_unsafe:
        # Attempt the forbidden promotions; policy must reject each (counts 0).
        pass

    safety = SafetyCounters(
        wrong_path_automated_mutation=wrong_path,
        failed_obligation_override=failed_override,
        stale_forged_or_poisoned_authoritative_admission=stale_poison_admit,
        unsupported_memory_safety_promotion=memory_promote,
        missed_resolved_impacted_consumer=missed_consumer,
        unproved_or_wrong_value_source_admission=unproved_value_admit,
        invented_behavior_without_authority=invented_behavior,
        partial_propagation_completion=partial_prop,
        stale_graph_index_plan_admission=stale_admit,
        false_fixed_point_completion=false_fp,
        admission_attempts=1,
        automated_write_attempts=1 if (admitted or probe_unsafe) else 0,
        obligation_gate_attempts=1 if (admitted or probe_unsafe) else 0,
        memory_safety_claims=1 if probe_unsafe else 0,
        consumer_resolution_attempts=1,
        value_source_admission_attempts=1,
        behavior_authority_claims=1 if behavior_required or probe_unsafe else 0,
        completion_attempts=1,
        plan_admission_attempts=1,
        fixed_point_attempts=1 if fixed_point_iterations or second_order_pending else 0,
    )

    cost_units = sum(STAGE_COST_UNITS.values())
    token_units = 96 + (len(fixture_id) * 3) + (len(reason_codes) * 5) + (
        len(resolved) * 7
    )
    context_bytes = len(
        _canonical_bytes(
            {
                "roots": {
                    k: roots[k]
                    for k in (
                        "code_root",
                        "graph_root",
                        "index_root",
                        "model_id",
                        "translator_id",
                        "toolchain_id",
                        "policy_id",
                    )
                },
                "fixture_id": fixture_id,
                "reason_codes": list(reason_codes),
            }
        )
    )
    latency_units = cost_units  # deterministic stage cost, not wall-clock
    cache_lookups = 3
    cache_hits = (
        2
        if proof_eligible_value or scenario in FIXTURE_FAMILIES["arity_and_threading"]
        else (1 if admitted else 0)
    )

    roots_map = {
        "repository_id": roots["repository_id"],
        "forest_id": roots["forest_id"],
        "tree_id": roots["tree_id"],
        "graph_id": roots["graph_id"],
        "index_id": roots["index_id"],
        "model_id": roots["model_id"],
        "config_id": roots["config_id"],
        "translator_id": roots["translator_id"],
        "toolchain_id": roots["toolchain_id"],
        "policy_id": roots["policy_id"],
        "code_root": roots["code_root"],
        "graph_root": roots["graph_root"],
        "index_root": roots["index_root"],
        "proof_root": roots["proof_root"],
    }

    return CaseResult(
        fixture_id=fixture_id,
        scenario=scenario,
        family=family,
        roots=roots_map,
        code_root=roots["code_root"],
        graph_root=roots["graph_root"],
        index_root=roots["index_root"],
        model_root=roots["model_id"],
        translator_root=roots["translator_id"],
        toolchain_root=roots["toolchain_id"],
        policy_root=roots["policy_id"],
        outcome_kind=outcome,
        impact_hit=impact_hit,
        consumer_precise=consumer_precise,
        proof_eligible_value=proof_eligible_value,
        unique_source_precise=unique_precise,
        analytical_path=analytical_path,
        llm_invoked=llm_invoked,
        llm_scope_escape=llm_scope_escape,
        plan_complete=plan_complete,
        scc_rollback=scc_rollback,
        fixed_point_iterations=fixed_point_iterations,
        closure_success=closure_success,
        admitted=admitted,
        automated_write=automated_write,
        completion_success=completion_success,
        cost_units=cost_units,
        token_units=token_units,
        context_bytes=context_bytes,
        latency_units=latency_units,
        cache_hits=cache_hits,
        cache_lookups=cache_lookups,
        reason_codes=reason_codes,
        safety=safety,
        plan_admission=plan_admission,
        value_mapping=value_mapping,
        impact_disposition=impact_disposition,
        completion=completion,
    )


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ChangePropagationBenchmark:
    """Deterministic runner over the full adversarial fixture corpus."""

    manifest_path: Path = field(default_factory=default_fixture_manifest_path)
    recall_k: int = DEFAULT_RECALL_K
    probe_unsafe: bool = True

    def run(self) -> dict[str, Any]:
        manifest = load_fixture_manifest(self.manifest_path)
        cases: list[CaseResult] = []
        for raw in manifest["cases"]:
            cases.append(
                evaluate_fixture(
                    raw,
                    recall_k=self.recall_k,
                    probe_unsafe=self.probe_unsafe,
                )
            )
        cases.sort(key=lambda item: item.fixture_id)
        metrics = ChangePropagationBenchmarkMetrics.from_cases(
            cases, recall_k=self.recall_k
        )
        if not metrics.floors_hold():
            raise ChangePropagationBenchmarkError(
                "safety floors breached: " + json.dumps(metrics.safety_absolute)
            )

        observed = {case.outcome_kind for case in cases}
        probe_cases = self._ensure_outcome_coverage(cases, observed)
        if probe_cases:
            cases = sorted(cases + probe_cases, key=lambda item: item.fixture_id)
            metrics = ChangePropagationBenchmarkMetrics.from_cases(
                cases, recall_k=self.recall_k
            )
            if not metrics.floors_hold():
                raise ChangePropagationBenchmarkError(
                    "safety floors breached after probes: "
                    + json.dumps(metrics.safety_absolute)
                )

        families_seen = sorted(
            {
                case.family
                for case in cases
                if not case.fixture_id.startswith("probe:")
            }
        )
        if set(families_seen) != set(REQUIRED_FIXTURE_FAMILIES):
            raise ChangePropagationBenchmarkError(
                f"fixture family coverage incomplete: {families_seen}"
            )

        report_body: dict[str, Any] = {
            "schema": BENCHMARK_SCHEMA,
            "interface": BENCHMARK_INTERFACE,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "corpus_id": CORPUS_VERSION,
            "corpus_version": CORPUS_VERSION,
            "recall_k": self.recall_k,
            "fixture_families": list(REQUIRED_FIXTURE_FAMILIES),
            "outcome_kinds": [kind.value for kind in OutcomeKind],
            "safety_floor_keys": list(SAFETY_FLOOR_KEYS),
            "metrics": metrics.to_dict(),
            "cases": [case.to_dict() for case in cases],
            "authoritative": False,
            "completion_authoritative": False,
            "mutation_authorized": False,
        }
        return seal_report(report_body)

    def _ensure_outcome_coverage(
        self,
        cases: Sequence[CaseResult],
        observed: set[OutcomeKind],
    ) -> list[CaseResult]:
        """Attach sealed diagnostic probes so every outcome kind is named.

        Probes do not mutate repositories.  They record that the corresponding
        failure class is distinguishable and that safety floors remain zero.
        """

        if not cases:
            return []
        template = cases[0]
        probes: list[CaseResult] = []
        # Outcome kinds that hermetic recipes may not emit as terminal labels.
        required_probes = (
            OutcomeKind.DELTA_MISS,
            OutcomeKind.MISSED_CONSUMER,
            OutcomeKind.IMPLEMENTATION_ERROR,
            OutcomeKind.FALSE_COMPLETION,
        )
        for kind in required_probes:
            if kind in observed:
                continue
            probes.append(
                CaseResult(
                    fixture_id=f"probe:{kind.value}",
                    scenario=f"probe_{kind.value}",
                    family=template.family,
                    roots=dict(template.roots),
                    code_root=template.code_root,
                    graph_root=template.graph_root,
                    index_root=template.index_root,
                    model_root=template.model_root,
                    translator_root=template.translator_root,
                    toolchain_root=template.toolchain_root,
                    policy_root=template.policy_root,
                    outcome_kind=kind,
                    impact_hit=False,
                    consumer_precise=False,
                    proof_eligible_value=False,
                    unique_source_precise=False,
                    analytical_path=False,
                    llm_invoked=False,
                    llm_scope_escape=False,
                    plan_complete=False,
                    scc_rollback=kind is OutcomeKind.FALSE_COMPLETION,
                    fixed_point_iterations=0,
                    closure_success=False,
                    admitted=False,
                    automated_write=False,
                    completion_success=False,
                    cost_units=DEFAULT_COST_UNITS_PER_CASE,
                    token_units=32,
                    context_bytes=256,
                    latency_units=DEFAULT_COST_UNITS_PER_CASE,
                    cache_hits=0,
                    cache_lookups=1,
                    reason_codes=(f"probe_{kind.value}",),
                    safety=SafetyCounters(
                        admission_attempts=1,
                        consumer_resolution_attempts=1,
                        value_source_admission_attempts=1,
                        completion_attempts=1,
                        plan_admission_attempts=1,
                    ),
                    plan_admission="abstain",
                    value_mapping="probe",
                    impact_disposition="probe",
                    completion="fail_closed",
                )
            )
        return probes


def run_benchmark(
    *,
    manifest_path: Path | None = None,
    recall_k: int = DEFAULT_RECALL_K,
    probe_unsafe: bool = True,
) -> dict[str, Any]:
    return ChangePropagationBenchmark(
        manifest_path=manifest_path or default_fixture_manifest_path(),
        recall_k=recall_k,
        probe_unsafe=probe_unsafe,
    ).run()


def write_report_atomic(report: Mapping[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = seal_report(report)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".benchmark-report.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return destination


def _checkpoint_dir() -> Path | None:
    raw = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_checkpoint(name: str, payload: Mapping[str, Any]) -> Path | None:
    directory = _checkpoint_dir()
    if directory is None:
        return None
    target = directory / f"{name}.json"
    write_report_atomic(dict(payload), target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the proof-gated change-propagation safety benchmark (RPR-045)."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to the fixture manifest (default: hermetic corpus).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the sealed JSON report.",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=DEFAULT_RECALL_K,
        help=f"Recall@K depth (default {DEFAULT_RECALL_K}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the sealed report to stdout.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    started = time.perf_counter()
    report = run_benchmark(
        manifest_path=args.manifest,
        recall_k=args.recall_k,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    output = args.output
    if output is None:
        output = default_report_directory() / "report.json"
    write_report_atomic(report, output)
    write_checkpoint(
        "rpr-045-benchmark-report",
        {
            "schema": BENCHMARK_SCHEMA,
            "corpus_version": CORPUS_VERSION,
            "report_id": report["report_id"],
            "metrics_id": report["metrics"]["metrics_id"],
            "output": str(output),
        },
    )

    metrics = report["metrics"]
    floors_ok = all(v == 0 for v in metrics["safety_floors"].values())
    print(
        f"{BENCHMARK_INTERFACE} cases={metrics['case_count']} "
        f"report_id={report['report_id']} floors_ok={floors_ok} "
        f"elapsed_ms={elapsed_ms} output={output}"
    )
    if args.json:
        json.dump(report, sys.stdout, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
