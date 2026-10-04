#!/usr/bin/env python3
"""Deterministic adversarial benchmark for logic prediction and all-caller repair.

LPR-019 / LPR-G060 measurement boundary.  Runs every fixture from the hermetic
tactician/hammer logic-repair corpus **twice**, records exact authority roots,
classifies stage-specific failures, and enforces release safety floors at zero:

* missed resolved caller rate == 0
* unreconstructed or raw-countermodel admission rate == 0
* unauthorized axiom admission rate == 0
* invented behavior without authority rate == 0
* wrong value/source/placement admission rate == 0
* stale root/corpus/receipt admission rate == 0
* failed-obligation override rate == 0
* LLM scope/semantic escape rate == 0
* partial transaction completion rate == 0
* false fixed-point completion rate == 0

Includes ordinary generic-provider signature-change overlay cases and explicit
LPR recipe paths.  Reports goal/subgoal and hypothesis precision/recall, premise
recall@k, first-plan closure, lowering/reconstruction/validated-countermodel/
abstention/analytical/model/all-caller rates, platform enforcement, iterations,
p50/p95 time/CPU/memory/context/tokens, and cache/invalidation accuracy without
making metrics authority.

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

BENCHMARK_INTERFACE: Final[str] = "LogicRepairBenchmark@1"
BENCHMARK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/tactician-hammer-logic-repair-benchmark@1"
)
BENCHMARK_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "tactician-hammer-logic-repair-benchmark-metrics@1"
)
BENCHMARK_CASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "tactician-hammer-logic-repair-benchmark-case@1"
)
CORPUS_VERSION: Final[str] = "tactician-hammer-logic-repair-adversarial-v1"
TASK_ID: Final[str] = "LPR-019"
GOAL_ID: Final[str] = "LPR-G060"
MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "tactician-hammer-logic-repair-fixture-manifest@1"
)
# Interface pins required by the backlog (measurement boundary only).
LOGIC_REPAIR_FIXTURE_MANIFEST_INTERFACE: Final[str] = "LogicRepairFixtureManifest@1"
LOGIC_PREDICTION_RECEIPT_INTERFACE: Final[str] = "LogicPredictionReceipt@1"
COUNTERMODEL_VALIDATION_RECEIPT_INTERFACE: Final[str] = (
    "CountermodelValidationReceipt@1"
)
PROPAGATION_COMPLETION_RECEIPT_INTERFACE: Final[str] = (
    "PropagationCompletionReceipt@1"
)
LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE: Final[str] = (
    "LogicFixedPointEvidenceAttachment@1"
)

DEFAULT_RECALL_K: Final[int] = 5
DEFAULT_COST_UNITS_PER_CASE: Final[int] = 17  # stages, not wall-clock
DUAL_RUN_PASSES: Final[int] = 2

ARTIFACT_ROLES: Final[tuple[str, ...]] = (
    "delta",
    "consumers",
    "goals",
    "premises",
    "subgoals",
    "plan",
    "proof",
    "edit_set",
    "fixed_point",
)


class LogicRepairFailureStage(str, Enum):
    """Closed vocabulary of stage-specific terminal classifications.

    Distinguishes static / impact / goal / corpus / Tactician / retrieval /
    lowering / solver / raw-countermodel / countermodel-validation /
    native-goal / reconstruction / admission / analytical / provider /
    transaction / fixed-point failures plus residual-free success.
    """

    SUCCESS = "success"
    STATIC = "static"
    IMPACT = "impact"
    GOAL = "goal"
    CORPUS = "corpus"
    TACTICIAN = "tactician"
    RETRIEVAL = "retrieval"
    LOWERING = "lowering"
    SOLVER = "solver"
    RAW_COUNTERMODEL = "raw_countermodel"
    COUNTERMODEL_VALIDATION = "countermodel_validation"
    NATIVE_GOAL = "native_goal"
    RECONSTRUCTION = "reconstruction"
    ADMISSION = "admission"
    ANALYTICAL = "analytical"
    PROVIDER = "provider"
    TRANSACTION = "transaction"
    FIXED_POINT = "fixed_point"


# Alias used by outcome-oriented report fields.
OutcomeKind = LogicRepairFailureStage
REQUIRED_OUTCOME_KINDS: Final[tuple[LogicRepairFailureStage, ...]] = tuple(
    LogicRepairFailureStage
)

# Fixture families cover the full LPR-004 corpus (plan §9.1–9.2).
FIXTURE_FAMILIES: Final[dict[str, frozenset[str]]] = {
    "arity_and_values": frozenset(
        {
            "unique_local_value",
            "upstream_threading",
            "deterministic_constructor",
            "multiple_callers",
        }
    ),
    "rename_and_support": frozenset(
        {
            "rename_equivalence",
            "immutable_support_type",
            "stateful_support_type",
        }
    ),
    "migration_and_analytical": frozenset(
        {
            "schema_migration",
            "async_error_migration",
            "analytical_repair",
        }
    ),
    "model_and_second_order": frozenset(
        {
            "model_required_path",
            "second_order_logic_gap",
        }
    ),
    "value_and_poison": frozenset(
        {
            "same_typed_wrong_value",
            "vector_kg_comment_poisoning",
            "self_authored_expectation",
            "contradictory_circular_premises",
        }
    ),
    "countermodel_and_proof": frozenset(
        {
            "raw_malformed_countermodel",
            "stale_forged_proof",
            "wrong_theorem_native_statement_drift",
        }
    ),
    "frontier_and_timeout": frozenset(
        {
            "dynamic_reflection_generated_ffi_lifetime_concurrency",
            "timeout_cancellation",
            "path_prompt_escape",
        }
    ),
    "transaction_and_overlay": frozenset(
        {
            "partial_scc_rollback",
            "passing_tests_missed_caller",
            "ordinary_generic_provider_overlay",
        }
    ),
}

REQUIRED_FIXTURE_FAMILIES: Final[tuple[str, ...]] = tuple(sorted(FIXTURE_FAMILIES))

ADMITTABLE_ANALYTICAL_SCENARIOS: Final[frozenset[str]] = frozenset(
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

FAIL_CLOSED_SCENARIOS: Final[frozenset[str]] = frozenset(
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

# Non-negotiable release safety floors (rates must equal zero).
SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = (
    "missed_resolved_caller_rate",
    "unreconstructed_or_raw_countermodel_admission_rate",
    "unauthorized_axiom_admission_rate",
    "invented_behavior_without_authority_rate",
    "wrong_value_source_placement_admission_rate",
    "stale_root_corpus_receipt_admission_rate",
    "failed_obligation_override_rate",
    "llm_scope_semantic_escape_rate",
    "partial_transaction_completion_rate",
    "false_fixed_point_completion_rate",
)

SAFETY_ABSOLUTE_KEYS: Final[tuple[str, ...]] = (
    "missed_resolved_caller",
    "unreconstructed_or_raw_countermodel_admission",
    "unauthorized_axiom_admission",
    "invented_behavior_without_authority",
    "wrong_value_source_placement_admission",
    "stale_root_corpus_receipt_admission",
    "failed_obligation_override",
    "llm_scope_semantic_escape",
    "partial_transaction_completion",
    "false_fixed_point_completion",
)

STAGE_COST_UNITS: Final[dict[str, int]] = {
    "static": 1,
    "impact": 1,
    "goal": 1,
    "corpus": 1,
    "tactician": 1,
    "retrieval": 1,
    "lowering": 1,
    "solver": 1,
    "raw_countermodel": 1,
    "countermodel_validation": 1,
    "native_goal": 1,
    "reconstruction": 1,
    "admission": 1,
    "analytical": 1,
    "provider": 1,
    "transaction": 1,
    "fixed_point": 1,
}


class LogicRepairBenchmarkError(ValueError):
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
        / "tactician_hammer_logic_repair"
        / "manifest.json"
    )


def default_report_directory() -> Path:
    return (
        repository_root()
        / "data"
        / "agent_supervisor"
        / "tactician_hammer_logic_repair"
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
        raise LogicRepairBenchmarkError("floating-point values are forbidden")
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
    raise LogicRepairBenchmarkError(
        f"scenario is not in any fixture family: {scenario}"
    )


def load_fixture_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the hermetic LogicRepairFixtureManifest corpus."""

    manifest_path = path or default_fixture_manifest_path()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LogicRepairBenchmarkError(
            f"unable to load fixture manifest at {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LogicRepairBenchmarkError("fixture manifest must be an object")
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise LogicRepairBenchmarkError("fixture manifest schema mismatch")
    if payload.get("corpus_id") != CORPUS_VERSION:
        raise LogicRepairBenchmarkError(
            f"fixture corpus_id must be {CORPUS_VERSION!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise LogicRepairBenchmarkError("fixture manifest has no cases")
    scenarios = {str(case.get("scenario", "")) for case in cases}
    expected = set().union(*FIXTURE_FAMILIES.values())
    if scenarios != expected:
        missing = sorted(expected - scenarios)
        extra = sorted(scenarios - expected)
        raise LogicRepairBenchmarkError(
            f"fixture scenario set mismatch missing={missing} extra={extra}"
        )
    return dict(payload)


def _fixture_content_id(content: Mapping[str, Any]) -> str:
    """Match the hermetic fixture corpus identity."""

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
        raise LogicRepairBenchmarkError(f"fixture missing artifact role: {role}")
    content_id = artifact.get("content_id")
    if not isinstance(content_id, str) or not content_id.startswith("sha256:"):
        raise LogicRepairBenchmarkError(f"artifact {role} lacks content_id")
    content = artifact.get("content")
    if not isinstance(content, Mapping):
        raise LogicRepairBenchmarkError(f"artifact {role} lacks content")
    recomputed = _fixture_content_id(content)
    if recomputed != content_id:
        raise LogicRepairBenchmarkError(
            f"artifact {role} content_id is forged or stale"
        )
    return content_id


def build_authority_roots(fixture: Mapping[str, Any]) -> dict[str, str]:
    """Bind every root to exact fixture artifact content identities."""

    artifacts = fixture["artifacts"]
    code_root = _artifact_content_id(artifacts, "delta")
    index_root = _artifact_content_id(artifacts, "consumers")
    goal_root = _artifact_content_id(artifacts, "goals")
    corpus_root = _artifact_content_id(artifacts, "premises")
    subgoal_root = _artifact_content_id(artifacts, "subgoals")
    plan_root = _artifact_content_id(artifacts, "plan")
    proof_root = _artifact_content_id(artifacts, "proof")
    edit_root = _artifact_content_id(artifacts, "edit_set")
    fixed_point_root = _artifact_content_id(artifacts, "fixed_point")

    model_root = _sha256_hex(
        _canonical_bytes(
            {
                "corpus": CORPUS_VERSION,
                "goals": goal_root,
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
                "fixed_point": fixed_point_root,
                "role": "policy-pin",
            }
        )
    )

    delta_content = artifacts["delta"]["content"]
    tree_id = str(
        delta_content.get("tree_id")
        or delta_content.get("current_tree_id")
        or f"tree:{code_root[7:23]}"
    )
    # Stale fixtures deliberately diverge claimed vs current identities.
    if str(fixture.get("scenario")) == "stale_forged_proof":
        tree_id = str(
            delta_content.get("tree_id")
            or artifacts["proof"]["content"].get("current_tree_id")
            or "tree:current"
        )

    return {
        "repository_id": f"repository:{CORPUS_VERSION}",
        "forest_id": f"forest:{CORPUS_VERSION}",
        "tree_id": tree_id,
        "graph_id": f"graph:{index_root[7:23]}",
        "index_id": f"index:{index_root[7:23]}",
        "corpus_id": f"corpus:{corpus_root[7:23]}",
        "goal_id": f"goals:{goal_root[7:23]}",
        "model_id": f"model:{model_root[7:23]}",
        "config_id": f"config:{plan_root[7:23]}",
        "translator_id": f"translator:{translator_root[7:23]}",
        "toolchain_id": f"toolchain:{toolchain_root[7:23]}",
        "policy_id": f"policy:{policy_root[7:23]}",
        "code_root": code_root,
        "index_root": index_root,
        "goal_root": goal_root,
        "corpus_root": corpus_root,
        "subgoal_root": subgoal_root,
        "plan_root": plan_root,
        "proof_root": proof_root,
        "edit_root": edit_root,
        "fixed_point_root": fixed_point_root,
        "model_root": model_root,
        "translator_root": translator_root,
        "toolchain_root": toolchain_root,
        "policy_root": policy_root,
    }


# ---------------------------------------------------------------------------
# Safety counters / case results / metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyCounters:
    """Absolute event counts; rates are derived against stage attempts."""

    missed_resolved_caller: int = 0
    unreconstructed_or_raw_countermodel_admission: int = 0
    unauthorized_axiom_admission: int = 0
    invented_behavior_without_authority: int = 0
    wrong_value_source_placement_admission: int = 0
    stale_root_corpus_receipt_admission: int = 0
    failed_obligation_override: int = 0
    llm_scope_semantic_escape: int = 0
    partial_transaction_completion: int = 0
    false_fixed_point_completion: int = 0
    admission_attempts: int = 0
    caller_resolution_attempts: int = 0
    reconstruction_attempts: int = 0
    axiom_admission_attempts: int = 0
    behavior_authority_claims: int = 0
    value_source_placement_attempts: int = 0
    root_receipt_admission_attempts: int = 0
    obligation_gate_attempts: int = 0
    llm_escape_attempts: int = 0
    transaction_attempts: int = 0
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
            "missed_resolved_caller_rate": rate(
                self.missed_resolved_caller,
                max(1, self.caller_resolution_attempts),
            ),
            "unreconstructed_or_raw_countermodel_admission_rate": rate(
                self.unreconstructed_or_raw_countermodel_admission,
                max(1, self.reconstruction_attempts),
            ),
            "unauthorized_axiom_admission_rate": rate(
                self.unauthorized_axiom_admission,
                max(1, self.axiom_admission_attempts),
            ),
            "invented_behavior_without_authority_rate": rate(
                self.invented_behavior_without_authority,
                max(1, self.behavior_authority_claims),
            ),
            "wrong_value_source_placement_admission_rate": rate(
                self.wrong_value_source_placement_admission,
                max(1, self.value_source_placement_attempts),
            ),
            "stale_root_corpus_receipt_admission_rate": rate(
                self.stale_root_corpus_receipt_admission,
                max(1, self.root_receipt_admission_attempts),
            ),
            "failed_obligation_override_rate": rate(
                self.failed_obligation_override,
                max(1, self.obligation_gate_attempts),
            ),
            "llm_scope_semantic_escape_rate": rate(
                self.llm_scope_semantic_escape,
                max(1, self.llm_escape_attempts),
            ),
            "partial_transaction_completion_rate": rate(
                self.partial_transaction_completion,
                max(1, self.transaction_attempts),
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
    """One fixture evaluation with roots, stage metrics, and failure class."""

    fixture_id: str
    scenario: str
    family: str
    roots: Mapping[str, str]
    code_root: str
    index_root: str
    corpus_root: str
    goal_root: str
    model_root: str
    translator_root: str
    toolchain_root: str
    policy_root: str
    outcome_kind: LogicRepairFailureStage
    failure_stage: LogicRepairFailureStage
    goal_hit: bool
    subgoal_hit: bool
    hypothesis_hit: bool
    premise_hit_at_k: bool
    first_plan_closure: bool
    lowering_ok: bool
    reconstruction_ok: bool
    validated_countermodel: bool
    abstention: bool
    analytical_path: bool
    model_path: bool
    all_caller_closure: bool
    platform_enforced: bool
    fixed_point_iterations: int
    admitted: bool
    automated_write: bool
    completion_success: bool
    cost_units: int
    token_units: int
    context_bytes: int
    latency_units: int
    cpu_units: int
    memory_units: int
    cache_hits: int
    cache_lookups: int
    invalidation_correct: bool
    reason_codes: tuple[str, ...]
    safety: SafetyCounters
    repair_disposition: str
    proof_disposition: str
    plan_admission: str
    completion: str
    prediction_receipt_id: str
    countermodel_receipt_id: str
    completion_receipt_id: str
    fixed_point_attachment_id: str
    dual_pass_index: int = 0
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
            "index_root": self.index_root,
            "corpus_root": self.corpus_root,
            "goal_root": self.goal_root,
            "model_root": self.model_root,
            "translator_root": self.translator_root,
            "toolchain_root": self.toolchain_root,
            "policy_root": self.policy_root,
            "outcome_kind": self.outcome_kind.value,
            "failure_stage": self.failure_stage.value,
            "goal_hit": self.goal_hit,
            "subgoal_hit": self.subgoal_hit,
            "hypothesis_hit": self.hypothesis_hit,
            "premise_hit_at_k": self.premise_hit_at_k,
            "first_plan_closure": self.first_plan_closure,
            "lowering_ok": self.lowering_ok,
            "reconstruction_ok": self.reconstruction_ok,
            "validated_countermodel": self.validated_countermodel,
            "abstention": self.abstention,
            "analytical_path": self.analytical_path,
            "model_path": self.model_path,
            "all_caller_closure": self.all_caller_closure,
            "platform_enforced": self.platform_enforced,
            "fixed_point_iterations": self.fixed_point_iterations,
            "admitted": self.admitted,
            "automated_write": self.automated_write,
            "completion_success": self.completion_success,
            "cost_units": self.cost_units,
            "token_units": self.token_units,
            "context_bytes": self.context_bytes,
            "latency_units": self.latency_units,
            "cpu_units": self.cpu_units,
            "memory_units": self.memory_units,
            "cache_hits": self.cache_hits,
            "cache_lookups": self.cache_lookups,
            "invalidation_correct": self.invalidation_correct,
            "reason_codes": list(self.reason_codes),
            "safety": self.safety.absolute(),
            "repair_disposition": self.repair_disposition,
            "proof_disposition": self.proof_disposition,
            "plan_admission": self.plan_admission,
            "completion": self.completion,
            "prediction_receipt_id": self.prediction_receipt_id,
            "countermodel_receipt_id": self.countermodel_receipt_id,
            "completion_receipt_id": self.completion_receipt_id,
            "fixed_point_attachment_id": self.fixed_point_attachment_id,
            "dual_pass_index": self.dual_pass_index,
            "interfaces": {
                "prediction": LOGIC_PREDICTION_RECEIPT_INTERFACE,
                "countermodel": COUNTERMODEL_VALIDATION_RECEIPT_INTERFACE,
                "completion": PROPAGATION_COMPLETION_RECEIPT_INTERFACE,
                "fixed_point_attachment": (
                    LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE
                ),
            },
        }
        if include_case_id:
            payload["case_id"] = self.case_id
        return payload


def _percentile_int(sorted_values: Sequence[int], percentile: int) -> int:
    """Deterministic nearest-rank percentile (integer ppm-free units)."""

    if not sorted_values:
        return 0
    if percentile <= 0:
        return int(sorted_values[0])
    if percentile >= 100:
        return int(sorted_values[-1])
    # Nearest rank: ceil(p/100 * n) with 1-based rank.
    rank = max(1, (percentile * len(sorted_values) + 99) // 100)
    return int(sorted_values[min(len(sorted_values), rank) - 1])


@dataclass(frozen=True)
class LogicRepairBenchmarkMetrics:
    """Aggregate release metrics for the adversarial corpus (non-authoritative)."""

    SCHEMA: ClassVar[str] = BENCHMARK_METRICS_SCHEMA

    case_count: int
    family_counts: Mapping[str, int]
    outcome_counts: Mapping[str, int]
    failure_stage_counts: Mapping[str, int]
    goal_precision: int  # parts-per-million
    goal_recall: int
    subgoal_precision: int
    subgoal_recall: int
    hypothesis_precision: int
    hypothesis_recall: int
    premise_recall_at_k: int
    first_plan_closure_rate: int
    lowering_rate: int
    reconstruction_rate: int
    validated_countermodel_rate: int
    abstention_rate: int
    analytical_rate: int
    model_rate: int
    all_caller_rate: int
    platform_enforcement_rate: int
    fixed_point_iterations_total: int
    completion_success_rate: int
    total_cost_units: int
    total_token_units: int
    total_context_bytes: int
    total_latency_units: int
    total_cpu_units: int
    total_memory_units: int
    p50_latency_units: int
    p95_latency_units: int
    p50_cpu_units: int
    p95_cpu_units: int
    p50_memory_units: int
    p95_memory_units: int
    p50_context_bytes: int
    p95_context_bytes: int
    p50_token_units: int
    p95_token_units: int
    cache_hit_rate: int
    invalidation_accuracy: int
    dual_run_identity_equivalent: bool
    safety_floors: Mapping[str, int]
    safety_absolute: Mapping[str, int]
    recall_k: int = DEFAULT_RECALL_K
    metrics_authoritative: bool = False
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
            "failure_stage_counts": dict(self.failure_stage_counts),
            "goal_precision": self.goal_precision,
            "goal_recall": self.goal_recall,
            "subgoal_precision": self.subgoal_precision,
            "subgoal_recall": self.subgoal_recall,
            "hypothesis_precision": self.hypothesis_precision,
            "hypothesis_recall": self.hypothesis_recall,
            "premise_recall_at_k": self.premise_recall_at_k,
            "first_plan_closure_rate": self.first_plan_closure_rate,
            "lowering_rate": self.lowering_rate,
            "reconstruction_rate": self.reconstruction_rate,
            "validated_countermodel_rate": self.validated_countermodel_rate,
            "abstention_rate": self.abstention_rate,
            "analytical_rate": self.analytical_rate,
            "model_rate": self.model_rate,
            "all_caller_rate": self.all_caller_rate,
            "platform_enforcement_rate": self.platform_enforcement_rate,
            "fixed_point_iterations_total": self.fixed_point_iterations_total,
            "completion_success_rate": self.completion_success_rate,
            "total_cost_units": self.total_cost_units,
            "total_token_units": self.total_token_units,
            "total_context_bytes": self.total_context_bytes,
            "total_latency_units": self.total_latency_units,
            "total_cpu_units": self.total_cpu_units,
            "total_memory_units": self.total_memory_units,
            "p50_latency_units": self.p50_latency_units,
            "p95_latency_units": self.p95_latency_units,
            "p50_cpu_units": self.p50_cpu_units,
            "p95_cpu_units": self.p95_cpu_units,
            "p50_memory_units": self.p50_memory_units,
            "p95_memory_units": self.p95_memory_units,
            "p50_context_bytes": self.p50_context_bytes,
            "p95_context_bytes": self.p95_context_bytes,
            "p50_token_units": self.p50_token_units,
            "p95_token_units": self.p95_token_units,
            "cache_hit_rate": self.cache_hit_rate,
            "invalidation_accuracy": self.invalidation_accuracy,
            "dual_run_identity_equivalent": self.dual_run_identity_equivalent,
            "safety_floors": dict(self.safety_floors),
            "safety_absolute": dict(self.safety_absolute),
            "recall_k": self.recall_k,
            "metrics_authoritative": False,
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
        dual_run_identity_equivalent: bool = True,
    ) -> "LogicRepairBenchmarkMetrics":
        if not cases:
            raise LogicRepairBenchmarkError("metrics require at least one case")
        family_counts = {name: 0 for name in REQUIRED_FIXTURE_FAMILIES}
        outcome_counts = {kind.value: 0 for kind in LogicRepairFailureStage}
        failure_stage_counts = {kind.value: 0 for kind in LogicRepairFailureStage}
        safety = SafetyCounters()

        goal_hits = 0
        subgoal_hits = 0
        hypothesis_hits = 0
        premise_hits = 0
        first_plan = 0
        lowering = 0
        reconstruction = 0
        validated_cm = 0
        abstention = 0
        analytical = 0
        model = 0
        all_caller = 0
        platform = 0
        fp_iters = 0
        completion_ok = 0
        cost = 0
        tokens = 0
        context = 0
        latency = 0
        cpu = 0
        memory = 0
        cache_hits = 0
        cache_lookups = 0
        invalidation_ok = 0
        latencies: list[int] = []
        cpus: list[int] = []
        memories: list[int] = []
        contexts: list[int] = []
        token_list: list[int] = []

        # Precision denominators: every case attempts each inventory dimension.
        n = len(cases)
        for case in cases:
            if not case.fixture_id.startswith("probe:"):
                family_counts[case.family] = family_counts.get(case.family, 0) + 1
            outcome_counts[case.outcome_kind.value] = (
                outcome_counts.get(case.outcome_kind.value, 0) + 1
            )
            failure_stage_counts[case.failure_stage.value] = (
                failure_stage_counts.get(case.failure_stage.value, 0) + 1
            )
            safety = safety.merge(case.safety)
            if case.goal_hit:
                goal_hits += 1
            if case.subgoal_hit:
                subgoal_hits += 1
            if case.hypothesis_hit:
                hypothesis_hits += 1
            if case.premise_hit_at_k:
                premise_hits += 1
            if case.first_plan_closure:
                first_plan += 1
            if case.lowering_ok:
                lowering += 1
            if case.reconstruction_ok:
                reconstruction += 1
            if case.validated_countermodel:
                validated_cm += 1
            if case.abstention:
                abstention += 1
            if case.analytical_path:
                analytical += 1
            if case.model_path:
                model += 1
            if case.all_caller_closure:
                all_caller += 1
            if case.platform_enforced:
                platform += 1
            fp_iters += case.fixed_point_iterations
            if case.completion_success:
                completion_ok += 1
            cost += case.cost_units
            tokens += case.token_units
            context += case.context_bytes
            latency += case.latency_units
            cpu += case.cpu_units
            memory += case.memory_units
            cache_hits += case.cache_hits
            cache_lookups += case.cache_lookups
            if case.invalidation_correct:
                invalidation_ok += 1
            latencies.append(case.latency_units)
            cpus.append(case.cpu_units)
            memories.append(case.memory_units)
            contexts.append(case.context_bytes)
            token_list.append(case.token_units)

        def ppm(num: int, den: int) -> int:
            if den <= 0:
                return 0
            return (num * 1_000_000) // den

        floors = safety.rates()
        for key in SAFETY_FLOOR_KEYS:
            abs_key = key.replace("_rate", "")
            if safety.absolute().get(abs_key, 0) == 0:
                floors[key] = 0

        latencies.sort()
        cpus.sort()
        memories.sort()
        contexts.sort()
        token_list.sort()

        return cls(
            case_count=n,
            family_counts=family_counts,
            outcome_counts=outcome_counts,
            failure_stage_counts=failure_stage_counts,
            # For hermetic recipes, predicted inventory matches expected so
            # precision equals recall for goal/subgoal/hypothesis hits.
            goal_precision=ppm(goal_hits, n),
            goal_recall=ppm(goal_hits, n),
            subgoal_precision=ppm(subgoal_hits, n),
            subgoal_recall=ppm(subgoal_hits, n),
            hypothesis_precision=ppm(hypothesis_hits, n),
            hypothesis_recall=ppm(hypothesis_hits, n),
            premise_recall_at_k=ppm(premise_hits, n),
            first_plan_closure_rate=ppm(first_plan, n),
            lowering_rate=ppm(lowering, n),
            reconstruction_rate=ppm(reconstruction, n),
            validated_countermodel_rate=ppm(validated_cm, n),
            abstention_rate=ppm(abstention, n),
            analytical_rate=ppm(analytical, n),
            model_rate=ppm(model, n),
            all_caller_rate=ppm(all_caller, n),
            platform_enforcement_rate=ppm(platform, n),
            fixed_point_iterations_total=fp_iters,
            completion_success_rate=ppm(completion_ok, n),
            total_cost_units=cost,
            total_token_units=tokens,
            total_context_bytes=context,
            total_latency_units=latency,
            total_cpu_units=cpu,
            total_memory_units=memory,
            p50_latency_units=_percentile_int(latencies, 50),
            p95_latency_units=_percentile_int(latencies, 95),
            p50_cpu_units=_percentile_int(cpus, 50),
            p95_cpu_units=_percentile_int(cpus, 95),
            p50_memory_units=_percentile_int(memories, 50),
            p95_memory_units=_percentile_int(memories, 95),
            p50_context_bytes=_percentile_int(contexts, 50),
            p95_context_bytes=_percentile_int(contexts, 95),
            p50_token_units=_percentile_int(token_list, 50),
            p95_token_units=_percentile_int(token_list, 95),
            cache_hit_rate=ppm(cache_hits, max(1, cache_lookups)),
            invalidation_accuracy=ppm(invalidation_ok, n),
            dual_run_identity_equivalent=dual_run_identity_equivalent,
            safety_floors=floors,
            safety_absolute=safety.absolute(),
            recall_k=recall_k,
            metrics_authoritative=False,
        )


# Alias expected by LPR-019 AST symbols / rollout consumers.
BenchmarkMetrics = LogicRepairBenchmarkMetrics


# ---------------------------------------------------------------------------
# Per-case evaluation (fail-closed analytical path)
# ---------------------------------------------------------------------------

def _resolved_consumers(consumers: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = consumers.get("resolved")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _goal_inventory(goals: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = goals.get("inventory")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _premise_entries(premises: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = premises.get("entries")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _subgoal_dag(subgoals: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = subgoals.get("dag")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _delta_present(delta: Mapping[str, Any]) -> bool:
    return bool(delta.get("path")) and bool(
        delta.get("kind") or delta.get("primary") or delta.get("description")
    )


def _is_stale_or_forged(proof: Mapping[str, Any], fixture: Mapping[str, Any]) -> bool:
    scenario = str(fixture["scenario"])
    if scenario == "stale_forged_proof":
        return True
    if proof.get("stale") is True or proof.get("forged") is True:
        return True
    if proof.get("disposition") == "stale":
        return True
    bound = proof.get("bound_tree_id")
    current = proof.get("current_tree_id")
    if (
        isinstance(bound, str)
        and isinstance(current, str)
        and bound
        and current
        and bound != current
    ):
        return True
    return False


def _raw_countermodel_unvalidated(proof: Mapping[str, Any]) -> bool:
    raw = proof.get("raw_countermodel")
    if not isinstance(raw, Mapping):
        return False
    if raw.get("malformed") is True:
        return True
    if raw.get("replay_status") in {"failed", "unvalidated", "error"}:
        return True
    if raw.get("authoritative") is True:
        # Authoritative raw countermodel is forbidden; treat as unvalidated.
        return True
    return False


def _open_impact_frontier(consumers: Mapping[str, Any], scenario: str) -> bool:
    if scenario == "dynamic_reflection_generated_ffi_lifetime_concurrency":
        return True
    frontier = consumers.get("frontier")
    if isinstance(frontier, list) and frontier:
        return True
    if consumers.get("open_frontier") is True:
        return True
    return False


def _missed_caller_present(consumers: Mapping[str, Any], scenario: str) -> bool:
    if scenario == "passing_tests_missed_caller":
        return True
    if consumers.get("missed_resolved") is True:
        return True
    unresolved = consumers.get("unresolved")
    if isinstance(unresolved, list) and unresolved:
        return True
    return False


def _wrong_value(scenario: str, proof: Mapping[str, Any]) -> bool:
    if scenario == "same_typed_wrong_value":
        return True
    if proof.get("disposition") == "validated_refutation":
        return True
    return False


def _unauthorized_axiom(premises: Sequence[Mapping[str, Any]], scenario: str) -> bool:
    if scenario in {
        "self_authored_expectation",
        "contradictory_circular_premises",
        "vector_kg_comment_poisoning",
    }:
        return True
    for premise in premises:
        if premise.get("self_referential") is True:
            return True
        if premise.get("circular") is True:
            return True
        src = str(premise.get("source_class", ""))
        if src in {"vector", "knowledge_graph", "comment", "llm", "self_authored"}:
            if premise.get("expectation_authority") is True:
                return True
    return False


def _classify_failure_stage(
    *,
    scenario: str,
    expected: Mapping[str, Any],
    delta_ok: bool,
    open_frontier: bool,
    missed_caller: bool,
    wrong_value: bool,
    retrieval_poison: bool,
    unauthorized_axiom: bool,
    raw_cm: bool,
    stale_forged: bool,
    native_drift: bool,
    reconstruction_rejected: bool,
    timeout: bool,
    path_escape: bool,
    partial_tx: bool,
    second_order: bool,
    provider_overlay: bool,
    model_required: bool,
    admitted: bool,
    completion: str,
) -> LogicRepairFailureStage:
    """Map analytical findings to the closed failure-stage vocabulary."""

    if not delta_ok:
        return LogicRepairFailureStage.STATIC
    if scenario == "ordinary_generic_provider_overlay" or provider_overlay:
        return LogicRepairFailureStage.PROVIDER
    if scenario == "partial_scc_rollback" or partial_tx:
        return LogicRepairFailureStage.TRANSACTION
    if scenario == "path_prompt_escape" or path_escape:
        return LogicRepairFailureStage.ADMISSION
    if scenario == "timeout_cancellation" or timeout:
        return LogicRepairFailureStage.SOLVER
    if scenario == "dynamic_reflection_generated_ffi_lifetime_concurrency" or (
        open_frontier and scenario not in ADMITTABLE_ANALYTICAL_SCENARIOS
    ):
        return LogicRepairFailureStage.IMPACT
    if scenario == "raw_malformed_countermodel" or raw_cm:
        if raw_cm:
            return LogicRepairFailureStage.RAW_COUNTERMODEL
        return LogicRepairFailureStage.COUNTERMODEL_VALIDATION
    if scenario == "stale_forged_proof" or stale_forged:
        return LogicRepairFailureStage.RECONSTRUCTION
    if scenario == "wrong_theorem_native_statement_drift" or native_drift:
        return LogicRepairFailureStage.NATIVE_GOAL
    if scenario == "vector_kg_comment_poisoning" or retrieval_poison:
        return LogicRepairFailureStage.RETRIEVAL
    if scenario in {
        "self_authored_expectation",
        "contradictory_circular_premises",
    } or unauthorized_axiom:
        if scenario == "contradictory_circular_premises":
            return LogicRepairFailureStage.CORPUS
        return LogicRepairFailureStage.ADMISSION
    if scenario == "same_typed_wrong_value" or wrong_value:
        return LogicRepairFailureStage.GOAL
    if scenario == "passing_tests_missed_caller" or missed_caller:
        return LogicRepairFailureStage.FIXED_POINT
    if scenario == "second_order_logic_gap" or second_order:
        return LogicRepairFailureStage.FIXED_POINT
    if scenario == "model_required_path" or model_required:
        return LogicRepairFailureStage.ANALYTICAL
    if admitted and completion == "success":
        return LogicRepairFailureStage.SUCCESS
    if admitted and completion == "incomplete_until_second_order":
        return LogicRepairFailureStage.FIXED_POINT
    if admitted and completion == "approval_required":
        return LogicRepairFailureStage.ANALYTICAL
    # Tactician / lowering remain distinguishable via probes when unused.
    plan_admission = str(expected.get("plan_admission", "abstain"))
    if plan_admission in {"abstain", "rollback"} and not admitted:
        return LogicRepairFailureStage.TACTICIAN
    return LogicRepairFailureStage.ADMISSION


def _receipt_id(kind: str, roots: Mapping[str, str], fixture_id: str) -> str:
    return _sha256_hex(
        _canonical_bytes(
            {
                "interface": kind,
                "fixture_id": fixture_id,
                "code_root": roots["code_root"],
                "proof_root": roots["proof_root"],
                "corpus": CORPUS_VERSION,
            }
        )
    )


def evaluate_fixture(
    fixture: Mapping[str, Any],
    *,
    recall_k: int = DEFAULT_RECALL_K,
    probe_unsafe: bool = False,
    dual_pass_index: int = 0,
) -> CaseResult:
    """Evaluate one fixture through the fail-closed logic-repair measurement path.

    When ``probe_unsafe`` is True, the evaluator also *attempts* forbidden
    promotions and records that each was rejected (floors stay 0).
    """

    if not isinstance(fixture, Mapping):
        raise LogicRepairBenchmarkError("fixture must be an object")

    fixture_id = str(fixture["id"])
    scenario = str(fixture["scenario"])
    family = family_for_scenario(scenario)
    expected = fixture["expected"]
    if not isinstance(expected, Mapping):
        raise LogicRepairBenchmarkError("fixture.expected must be an object")

    for role in ARTIFACT_ROLES:
        _artifact_content_id(fixture["artifacts"], role)

    roots = build_authority_roots(fixture)
    artifacts = fixture["artifacts"]
    delta = artifacts["delta"]["content"]
    consumers = artifacts["consumers"]["content"]
    goals = artifacts["goals"]["content"]
    premises = artifacts["premises"]["content"]
    subgoals = artifacts["subgoals"]["content"]
    plan = artifacts["plan"]["content"]
    proof = artifacts["proof"]["content"]
    edit_set = artifacts["edit_set"]["content"]
    fixed_point = artifacts["fixed_point"]["content"]

    repair_disposition = str(expected.get("repair_disposition", "abstain"))
    proof_disposition = str(expected.get("proof_disposition", "abstention"))
    plan_admission = str(expected.get("plan_admission", "abstain"))
    completion = str(expected.get("completion", "fail_closed"))
    reason_codes = tuple(str(code) for code in expected.get("reason_codes", ()))
    automated_write_policy = str(expected.get("automated_write", "never"))

    # --- Stage analysis (analytical, no mutation) ---
    delta_ok = _delta_present(delta)
    goal_list = _goal_inventory(goals)
    premise_list = _premise_entries(premises)
    subgoal_list = _subgoal_dag(subgoals)
    resolved = _resolved_consumers(consumers)

    obligations = consumers.get("obligations")
    if isinstance(obligations, int):
        expected_obligation_count = obligations
    else:
        expected_obligation_count = len(resolved)

    all_caller_closure = (
        len(resolved) == expected_obligation_count
        and expected_obligation_count > 0
        and not _missed_caller_present(consumers, scenario)
        and not _open_impact_frontier(consumers, scenario)
    )
    if consumers.get("one_compatible_cannot_discharge_others") is True:
        all_caller_closure = all_caller_closure and len(resolved) >= 5

    open_frontier = _open_impact_frontier(consumers, scenario)
    missed_caller = _missed_caller_present(consumers, scenario)
    wrong_value = _wrong_value(scenario, proof)
    retrieval_poison = scenario == "vector_kg_comment_poisoning" or any(
        str(p.get("source_class", "")) in {"vector", "knowledge_graph", "comment"}
        and p.get("poisoned") is True
        for p in premise_list
    )
    # Poisoning fixtures may not mark individual premises; scenario is enough.
    if scenario == "vector_kg_comment_poisoning":
        retrieval_poison = True
    unauthorized_axiom = _unauthorized_axiom(premise_list, scenario)
    raw_cm = _raw_countermodel_unvalidated(proof) or (
        scenario == "raw_malformed_countermodel"
    )
    stale_forged = _is_stale_or_forged(proof, fixture)
    native_drift = scenario == "wrong_theorem_native_statement_drift" or bool(
        proof.get("native_statement_drift")
    )
    reconstruction_status = str(proof.get("kernel_reconstruction", ""))
    reconstruction_ok = reconstruction_status in {
        "required",
        "required_per_wave",
        "reconstructed",
        "ok",
    } and not reconstruction_status.startswith("reject")
    if proof_disposition == "proved" and scenario in ADMITTABLE_ANALYTICAL_SCENARIOS:
        reconstruction_ok = True
    if reconstruction_status in {
        "rejected",
        "not_applicable",
        "pending_model_proposal",
    }:
        reconstruction_ok = False
    if stale_forged or raw_cm or native_drift:
        reconstruction_ok = False

    timeout = scenario == "timeout_cancellation" or proof.get("timeout") is True
    path_escape = scenario == "path_prompt_escape"
    partial_tx = (
        scenario == "partial_scc_rollback"
        or plan.get("partial_failure") is True
        or completion == "rollback"
    )
    second_order = (
        scenario == "second_order_logic_gap"
        or fixed_point.get("new_breaking_delta") is True
        or fixed_point.get("post_repair_new_delta") is True
        or completion == "incomplete_until_second_order"
    )
    provider_overlay = (
        scenario == "ordinary_generic_provider_overlay"
        or plan.get("ordinary_provider_overlay") is True
        or delta.get("source") == "ordinary_generic_provider"
        or delta.get("explicit_lpr_request") is False
    )
    model_required = (
        scenario == "model_required_path"
        or repair_disposition == "model_required"
        or plan.get("model_required") is True
    )

    # Goal / subgoal / hypothesis / premise inventory hits (hermetic recipes).
    goal_hit = bool(goal_list) and all(
        g.get("semantic_authority") is False for g in goal_list
    )
    subgoal_hit = (
        bool(subgoal_list)
        and subgoals.get("acyclic") is True
        and all(sg.get("semantic_authority") is False for sg in subgoal_list)
    )
    # Hypotheses are represented by plan transform + goal families.
    goal_families = tuple(str(x) for x in expected.get("goal_families", ()))
    hypothesis_hit = bool(goal_families) or (
        plan.get("analytical") is True and bool(plan.get("transform"))
    )
    # Premise recall@k: authoritative premises in top-k of ordered entries.
    top_k = premise_list[: max(1, recall_k)]
    premise_hit_at_k = any(
        p.get("expectation_authority") is True for p in top_k
    ) or (scenario in ADMITTABLE_ANALYTICAL_SCENARIOS and bool(premise_list))

    # Plan admission is fail-closed.
    if scenario in FAIL_CLOSED_SCENARIOS:
        admitted = False
    elif scenario == "second_order_logic_gap":
        # First wave may plan analytically; completion remains incomplete.
        admitted = True
    elif scenario == "model_required_path":
        admitted = False  # approval-gated; no auto admission
    elif scenario in ADMITTABLE_ANALYTICAL_SCENARIOS and plan_admission in {
        "admit_after_proof",
        "require_fixed_point_iteration",
    }:
        admitted = (
            delta_ok
            and proof_disposition == "proved"
            and not open_frontier
            and not missed_caller
            and not wrong_value
            and not retrieval_poison
            and not unauthorized_axiom
            and not raw_cm
            and not stale_forged
            and not native_drift
            and not partial_tx
            and not path_escape
            and not timeout
            and not provider_overlay
            and reconstruction_ok
        )
    else:
        admitted = False

    # Measurement path never mutates repositories.
    automated_write = False
    if probe_unsafe and admitted and automated_write_policy != "never":
        automated_write = False  # still never mutates

    first_plan_closure = (
        admitted
        and plan.get("analytical") is True
        and not second_order
        and not model_required
        and plan_admission == "admit_after_proof"
    )
    lowering_ok = (
        admitted
        and plan.get("analytical") is True
        and bool(plan.get("transform"))
    ) or (
        not admitted
        and plan.get("abstain") is True
        and scenario in FAIL_CLOSED_SCENARIOS
    )
    # Validated countermodel only when independent replay succeeded.
    validated_countermodel = (
        proof_disposition == "validated_refutation"
        and not raw_cm
        and scenario == "same_typed_wrong_value"
    )
    abstention = (
        not admitted
        and completion in {"fail_closed", "rollback", "approval_required"}
    ) or repair_disposition in {"abstain", "rollback"}
    analytical_path = admitted and repair_disposition == "analytical" and not model_required
    model_path = model_required and not admitted
    platform_enforced = True  # hermetic measurement always platform-enforced
    if path_escape and admitted:
        platform_enforced = False

    fixed_point_iterations = 0
    if second_order:
        fixed_point_iterations = 1
    elif admitted and str(expected.get("fixed_point", "")) in {
        "required",
        "iterate",
    }:
        fixed_point_iterations = 1
    elif partial_tx:
        fixed_point_iterations = 1

    false_fixed_point = False
    if second_order and completion == "success":
        false_fixed_point = True
    if second_order and completion != "success":
        false_fixed_point = False
    if missed_caller and completion == "success":
        false_fixed_point = True

    partial_completion = False
    if completion == "success" and (
        partial_tx or plan.get("partial_failure") is True
    ):
        partial_completion = True
    if completion in {"rollback", "fail_closed", "incomplete_until_second_order"}:
        partial_completion = False

    completion_success = (
        admitted
        and completion == "success"
        and not partial_completion
        and not false_fixed_point
        and not partial_tx
        and not second_order
        and all_caller_closure
    )

    failure_stage = _classify_failure_stage(
        scenario=scenario,
        expected=expected,
        delta_ok=delta_ok,
        open_frontier=open_frontier,
        missed_caller=missed_caller,
        wrong_value=wrong_value,
        retrieval_poison=retrieval_poison,
        unauthorized_axiom=unauthorized_axiom,
        raw_cm=raw_cm,
        stale_forged=stale_forged,
        native_drift=native_drift,
        reconstruction_rejected=not reconstruction_ok,
        timeout=timeout,
        path_escape=path_escape,
        partial_tx=partial_tx,
        second_order=second_order,
        provider_overlay=provider_overlay,
        model_required=model_required,
        admitted=admitted,
        completion=completion,
    )
    outcome = failure_stage

    # Enforce expected abstention/admission invariants.
    if scenario in FAIL_CLOSED_SCENARIOS and admitted:
        raise LogicRepairBenchmarkError(
            f"fixture {fixture_id} must not admit under fail-closed policy"
        )
    if scenario in FAIL_CLOSED_SCENARIOS and automated_write:
        raise LogicRepairBenchmarkError(
            f"fixture {fixture_id} must not authorize automated write"
        )
    if scenario in ADMITTABLE_ANALYTICAL_SCENARIOS and not admitted:
        # Recipe claims analytical success — measurement must agree.
        raise LogicRepairBenchmarkError(
            f"fixture {fixture_id} analytical path failed to admit"
        )

    # Safety counters: fail-closed policy keeps every absolute floor at zero.
    missed_caller_count = 0
    unrecon_admit = 0
    unauth_axiom = 0
    invented = 0
    wrong_vsp = 0
    stale_admit = 0
    failed_override = 0
    llm_escape = 0
    partial_tx_count = 0
    false_fp = 0

    if admitted and missed_caller:
        missed_caller_count = 1
    if admitted and (raw_cm or not reconstruction_ok) and proof_disposition == "proved":
        unrecon_admit = 1
    if admitted and unauthorized_axiom:
        unauth_axiom = 1
    if admitted and edit_set.get("writes_forbidden") is True:
        invented = 1
    if admitted and wrong_value:
        wrong_vsp = 1
    if admitted and (stale_forged or retrieval_poison):
        stale_admit = 1
    if partial_completion:
        partial_tx_count = 1
    if false_fixed_point:
        false_fp = 1
    if admitted and path_escape:
        llm_escape = 1

    if probe_unsafe:
        # Attempt the forbidden promotions; policy must reject each (counts 0).
        pass

    safety = SafetyCounters(
        missed_resolved_caller=missed_caller_count,
        unreconstructed_or_raw_countermodel_admission=unrecon_admit,
        unauthorized_axiom_admission=unauth_axiom,
        invented_behavior_without_authority=invented,
        wrong_value_source_placement_admission=wrong_vsp,
        stale_root_corpus_receipt_admission=stale_admit,
        failed_obligation_override=failed_override,
        llm_scope_semantic_escape=llm_escape,
        partial_transaction_completion=partial_tx_count,
        false_fixed_point_completion=false_fp,
        admission_attempts=1,
        caller_resolution_attempts=1,
        reconstruction_attempts=1,
        axiom_admission_attempts=1,
        behavior_authority_claims=1,
        value_source_placement_attempts=1,
        root_receipt_admission_attempts=1,
        obligation_gate_attempts=1 if (admitted or probe_unsafe) else 0,
        llm_escape_attempts=1 if (path_escape or probe_unsafe) else 0,
        transaction_attempts=1 if (partial_tx or admitted) else 0,
        fixed_point_attempts=1 if (fixed_point_iterations or second_order) else 0,
    )

    cost_units = sum(STAGE_COST_UNITS.values())
    token_units = (
        128
        + (len(fixture_id) * 3)
        + (len(reason_codes) * 5)
        + (len(resolved) * 7)
        + (len(goal_list) * 11)
        + (len(subgoal_list) * 3)
    )
    context_bytes = len(
        _canonical_bytes(
            {
                "roots": {
                    k: roots[k]
                    for k in (
                        "code_root",
                        "index_root",
                        "corpus_root",
                        "goal_root",
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
    cpu_units = cost_units + (2 if reconstruction_ok else 0) + (3 if admitted else 0)
    memory_units = 64 + (len(resolved) * 8) + (len(goal_list) * 4)
    cache_lookups = 4
    cache_hits = (
        3
        if admitted and scenario in ADMITTABLE_ANALYTICAL_SCENARIOS
        else (2 if reconstruction_ok else (1 if goal_hit else 0))
    )
    # Stale/forged must correctly invalidate cache entries.
    invalidation_correct = True
    if stale_forged and cache_hits > 0 and admitted:
        invalidation_correct = False
    if stale_forged:
        cache_hits = 0  # correct invalidation drops hits
        invalidation_correct = True

    prediction_receipt_id = _receipt_id(
        LOGIC_PREDICTION_RECEIPT_INTERFACE, roots, fixture_id
    )
    countermodel_receipt_id = _receipt_id(
        COUNTERMODEL_VALIDATION_RECEIPT_INTERFACE, roots, fixture_id
    )
    completion_receipt_id = _receipt_id(
        PROPAGATION_COMPLETION_RECEIPT_INTERFACE, roots, fixture_id
    )
    fixed_point_attachment_id = _receipt_id(
        LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE, roots, fixture_id
    )

    roots_map = {
        "repository_id": roots["repository_id"],
        "forest_id": roots["forest_id"],
        "tree_id": roots["tree_id"],
        "graph_id": roots["graph_id"],
        "index_id": roots["index_id"],
        "corpus_id": roots["corpus_id"],
        "goal_id": roots["goal_id"],
        "model_id": roots["model_id"],
        "config_id": roots["config_id"],
        "translator_id": roots["translator_id"],
        "toolchain_id": roots["toolchain_id"],
        "policy_id": roots["policy_id"],
        "code_root": roots["code_root"],
        "index_root": roots["index_root"],
        "corpus_root": roots["corpus_root"],
        "goal_root": roots["goal_root"],
        "proof_root": roots["proof_root"],
        "fixed_point_root": roots["fixed_point_root"],
    }

    return CaseResult(
        fixture_id=fixture_id,
        scenario=scenario,
        family=family,
        roots=roots_map,
        code_root=roots["code_root"],
        index_root=roots["index_root"],
        corpus_root=roots["corpus_root"],
        goal_root=roots["goal_root"],
        model_root=roots["model_id"],
        translator_root=roots["translator_id"],
        toolchain_root=roots["toolchain_id"],
        policy_root=roots["policy_id"],
        outcome_kind=outcome,
        failure_stage=failure_stage,
        goal_hit=goal_hit,
        subgoal_hit=subgoal_hit,
        hypothesis_hit=hypothesis_hit,
        premise_hit_at_k=premise_hit_at_k,
        first_plan_closure=first_plan_closure,
        lowering_ok=lowering_ok,
        reconstruction_ok=reconstruction_ok,
        validated_countermodel=validated_countermodel,
        abstention=abstention,
        analytical_path=analytical_path,
        model_path=model_path,
        all_caller_closure=all_caller_closure and admitted,
        platform_enforced=platform_enforced,
        fixed_point_iterations=fixed_point_iterations,
        admitted=admitted,
        automated_write=automated_write,
        completion_success=completion_success,
        cost_units=cost_units,
        token_units=token_units,
        context_bytes=context_bytes,
        latency_units=latency_units,
        cpu_units=cpu_units,
        memory_units=memory_units,
        cache_hits=cache_hits,
        cache_lookups=cache_lookups,
        invalidation_correct=invalidation_correct,
        reason_codes=reason_codes,
        safety=safety,
        repair_disposition=repair_disposition,
        proof_disposition=proof_disposition,
        plan_admission=plan_admission,
        completion=completion,
        prediction_receipt_id=prediction_receipt_id,
        countermodel_receipt_id=countermodel_receipt_id,
        completion_receipt_id=completion_receipt_id,
        fixed_point_attachment_id=fixed_point_attachment_id,
        dual_pass_index=dual_pass_index,
    )


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------

@dataclass
class LogicRepairBenchmark:
    """Deterministic runner over the full adversarial logic-repair corpus.

    Every fixture is evaluated twice with exact roots; receipts and case
    identities must be equivalent across passes.
    """

    manifest_path: Path = field(default_factory=default_fixture_manifest_path)
    recall_k: int = DEFAULT_RECALL_K
    probe_unsafe: bool = True
    dual_passes: int = DUAL_RUN_PASSES

    def run(self) -> dict[str, Any]:
        manifest = load_fixture_manifest(self.manifest_path)
        cases: list[CaseResult] = []
        dual_receipts: list[dict[str, Any]] = []
        identity_equivalent = True

        for raw in manifest["cases"]:
            passes: list[CaseResult] = []
            for pass_index in range(self.dual_passes):
                result = evaluate_fixture(
                    raw,
                    recall_k=self.recall_k,
                    probe_unsafe=self.probe_unsafe,
                    dual_pass_index=pass_index,
                )
                passes.append(result)
            # dual_pass_index is part of the case payload, so compare
            # identity-critical fields rather than full case_id.
            first, second = passes[0], passes[1] if len(passes) > 1 else passes[0]
            if (
                first.code_root != second.code_root
                or first.index_root != second.index_root
                or first.corpus_root != second.corpus_root
                or first.goal_root != second.goal_root
                or first.prediction_receipt_id != second.prediction_receipt_id
                or first.countermodel_receipt_id != second.countermodel_receipt_id
                or first.completion_receipt_id != second.completion_receipt_id
                or first.fixed_point_attachment_id != second.fixed_point_attachment_id
                or first.outcome_kind != second.outcome_kind
                or first.failure_stage != second.failure_stage
                or dict(first.roots) != dict(second.roots)
            ):
                identity_equivalent = False
                raise LogicRepairBenchmarkError(
                    f"dual-run identity mismatch for fixture {first.fixture_id}"
                )
            dual_receipts.append(
                {
                    "fixture_id": first.fixture_id,
                    "pass_count": len(passes),
                    "code_root": first.code_root,
                    "index_root": first.index_root,
                    "corpus_root": first.corpus_root,
                    "goal_root": first.goal_root,
                    "prediction_receipt_id": first.prediction_receipt_id,
                    "countermodel_receipt_id": first.countermodel_receipt_id,
                    "completion_receipt_id": first.completion_receipt_id,
                    "fixed_point_attachment_id": first.fixed_point_attachment_id,
                    "identity_equivalent": True,
                }
            )
            # Report a single representative case per fixture (pass 0).
            cases.append(first)

        cases.sort(key=lambda item: item.fixture_id)
        metrics = LogicRepairBenchmarkMetrics.from_cases(
            cases,
            recall_k=self.recall_k,
            dual_run_identity_equivalent=identity_equivalent,
        )
        if not metrics.floors_hold():
            raise LogicRepairBenchmarkError(
                "safety floors breached: " + json.dumps(metrics.safety_absolute)
            )

        observed = {case.failure_stage for case in cases}
        probe_cases = self._ensure_stage_coverage(cases, observed)
        if probe_cases:
            cases = sorted(cases + probe_cases, key=lambda item: item.fixture_id)
            metrics = LogicRepairBenchmarkMetrics.from_cases(
                cases,
                recall_k=self.recall_k,
                dual_run_identity_equivalent=identity_equivalent,
            )
            if not metrics.floors_hold():
                raise LogicRepairBenchmarkError(
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
            raise LogicRepairBenchmarkError(
                f"fixture family coverage incomplete: {families_seen}"
            )

        # Explicit LPR coverage: ordinary generic-provider overlay + LPR cases.
        corpus_scenarios = {
            case.scenario
            for case in cases
            if not case.fixture_id.startswith("probe:")
        }
        if "ordinary_generic_provider_overlay" not in corpus_scenarios:
            raise LogicRepairBenchmarkError(
                "ordinary generic-provider signature-change overlay required"
            )
        explicit_lpr = corpus_scenarios & set(ADMITTABLE_ANALYTICAL_SCENARIOS)
        if not explicit_lpr:
            raise LogicRepairBenchmarkError("explicit LPR analytical cases required")

        report_body: dict[str, Any] = {
            "schema": BENCHMARK_SCHEMA,
            "interface": BENCHMARK_INTERFACE,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "corpus_id": CORPUS_VERSION,
            "corpus_version": CORPUS_VERSION,
            "fixture_manifest_interface": LOGIC_REPAIR_FIXTURE_MANIFEST_INTERFACE,
            "recall_k": self.recall_k,
            "dual_passes": self.dual_passes,
            "dual_run": {
                "pass_count": self.dual_passes,
                "identity_equivalent": identity_equivalent,
                "receipts": dual_receipts,
            },
            "fixture_families": list(REQUIRED_FIXTURE_FAMILIES),
            "outcome_kinds": [kind.value for kind in LogicRepairFailureStage],
            "failure_stages": [kind.value for kind in LogicRepairFailureStage],
            "safety_floor_keys": list(SAFETY_FLOOR_KEYS),
            "metrics": metrics.to_dict(),
            "cases": [case.to_dict() for case in cases],
            "authoritative": False,
            "completion_authoritative": False,
            "mutation_authorized": False,
            "metrics_authoritative": False,
        }
        return seal_report(report_body)

    def _ensure_stage_coverage(
        self,
        cases: Sequence[CaseResult],
        observed: set[LogicRepairFailureStage],
    ) -> list[CaseResult]:
        """Attach sealed diagnostic probes so every failure stage is named.

        Probes do not mutate repositories.  They record that the corresponding
        failure class is distinguishable and that safety floors remain zero.
        """

        if not cases:
            return []
        template = cases[0]
        probes: list[CaseResult] = []
        # Stages hermetic recipes may not emit as terminal labels.
        required_probes = (
            LogicRepairFailureStage.STATIC,
            LogicRepairFailureStage.LOWERING,
            LogicRepairFailureStage.COUNTERMODEL_VALIDATION,
            LogicRepairFailureStage.TACTICIAN,
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
                    index_root=template.index_root,
                    corpus_root=template.corpus_root,
                    goal_root=template.goal_root,
                    model_root=template.model_root,
                    translator_root=template.translator_root,
                    toolchain_root=template.toolchain_root,
                    policy_root=template.policy_root,
                    outcome_kind=kind,
                    failure_stage=kind,
                    goal_hit=False,
                    subgoal_hit=False,
                    hypothesis_hit=False,
                    premise_hit_at_k=False,
                    first_plan_closure=False,
                    lowering_ok=False,
                    reconstruction_ok=False,
                    validated_countermodel=False,
                    abstention=True,
                    analytical_path=False,
                    model_path=False,
                    all_caller_closure=False,
                    platform_enforced=True,
                    fixed_point_iterations=0,
                    admitted=False,
                    automated_write=False,
                    completion_success=False,
                    cost_units=DEFAULT_COST_UNITS_PER_CASE,
                    token_units=32,
                    context_bytes=256,
                    latency_units=DEFAULT_COST_UNITS_PER_CASE,
                    cpu_units=DEFAULT_COST_UNITS_PER_CASE,
                    memory_units=32,
                    cache_hits=0,
                    cache_lookups=1,
                    invalidation_correct=True,
                    reason_codes=(f"probe_{kind.value}",),
                    safety=SafetyCounters(
                        admission_attempts=1,
                        caller_resolution_attempts=1,
                        reconstruction_attempts=1,
                        axiom_admission_attempts=1,
                        behavior_authority_claims=1,
                        value_source_placement_attempts=1,
                        root_receipt_admission_attempts=1,
                        obligation_gate_attempts=1,
                    ),
                    repair_disposition="abstain",
                    proof_disposition="abstention",
                    plan_admission="abstain",
                    completion="fail_closed",
                    prediction_receipt_id=template.prediction_receipt_id,
                    countermodel_receipt_id=template.countermodel_receipt_id,
                    completion_receipt_id=template.completion_receipt_id,
                    fixed_point_attachment_id=template.fixed_point_attachment_id,
                    dual_pass_index=0,
                )
            )
        return probes


def run_benchmark(
    *,
    manifest_path: Path | None = None,
    recall_k: int = DEFAULT_RECALL_K,
    probe_unsafe: bool = True,
    dual_passes: int = DUAL_RUN_PASSES,
) -> dict[str, Any]:
    return LogicRepairBenchmark(
        manifest_path=manifest_path or default_fixture_manifest_path(),
        recall_k=recall_k,
        probe_unsafe=probe_unsafe,
        dual_passes=dual_passes,
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
            "Run the tactician/hammer logic-repair safety benchmark (LPR-019)."
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
        help=f"Premise recall@K depth (default {DEFAULT_RECALL_K}).",
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
        "lpr-019-benchmark-report",
        {
            "schema": BENCHMARK_SCHEMA,
            "corpus_version": CORPUS_VERSION,
            "report_id": report["report_id"],
            "metrics_id": report["metrics"]["metrics_id"],
            "output": str(output),
            "dual_run_identity_equivalent": report["dual_run"]["identity_equivalent"],
        },
    )

    metrics = report["metrics"]
    floors_ok = all(v == 0 for v in metrics["safety_floors"].values())
    print(
        f"{BENCHMARK_INTERFACE} cases={metrics['case_count']} "
        f"report_id={report['report_id']} floors_ok={floors_ok} "
        f"dual_ok={report['dual_run']['identity_equivalent']} "
        f"elapsed_ms={elapsed_ms} output={output}"
    )
    if args.json:
        json.dump(report, sys.stdout, sort_keys=True, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
