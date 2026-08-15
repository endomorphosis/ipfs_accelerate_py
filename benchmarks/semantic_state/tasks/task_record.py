"""Benchmark task records for SemanticStateBenchmarkCorpus@1.

Task definitions are reviewed fixture authority. Checked-in candidate patches
are oracle/replay fixtures only (production_eligible=false). Oracles are not
derived from running the benchmark harness or implementation under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

BENCHMARK_TASK_SCHEMA = "ipfs_accelerate_py/semantic-state/benchmark-task@1"
BASELINE_RETRIEVAL_SCHEMA = (
    "ipfs_accelerate_py/semantic-state/baseline-retrieval-policy@1"
)
TASK_ORACLE_SCHEMA = "ipfs_accelerate_py/semantic-state/benchmark-task-oracle@1"
CANDIDATE_PATCH_SCHEMA = (
    "ipfs_accelerate_py/semantic-state/benchmark-candidate-patch@1"
)

TASK_CATEGORIES = frozenset(
    {
        "small_bug_fix",
        "test_repair",
        "api_adapter",
        "schema_migration",
        "multi_file_refactor",
        "rejection_or_escalation",
    }
)

REQUIRED_CATEGORY_COUNTS: Mapping[str, int] = {
    "small_bug_fix": 10,
    "test_repair": 6,
    "api_adapter": 6,
    "schema_migration": 6,
    "multi_file_refactor": 6,
    "rejection_or_escalation": 6,
}

MODEL_ROUTES = frozenset(
    {
        "deterministic_only",
        "small_local_model",
        "medium_model",
        "frontier_model",
        "human_review_required",
    }
)

RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})

# Candidate verification is independent of production acceptance.
CANDIDATE_VERIFICATION_OUTCOMES = frozenset(
    {
        "pass",
        "fail",
        "reject",
        "escalate",
    }
)

# Production acceptance for oracle/replay corpus rows is never "accepted".
PRODUCTION_ACCEPTANCE_OUTCOMES = frozenset(
    {
        "not_applicable",
        "rejected",
        "blocked",
    }
)


class BenchmarkCorpusError(ValueError):
    """Closed benchmark-corpus record violation."""


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BenchmarkCorpusError(f"{name} must be a nonempty trimmed string")
    if any(not char.isprintable() for char in value):
        raise BenchmarkCorpusError(f"{name} contains non-printable characters")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise BenchmarkCorpusError(f"{name} must be a boolean")
    return value


def _sorted_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    items = tuple(_text(item, f"{name}[]") for item in values)
    if len(set(items)) != len(items):
        raise BenchmarkCorpusError(f"{name} must not contain duplicates")
    ordered = tuple(sorted(items))
    if ordered != items:
        raise BenchmarkCorpusError(f"{name} must be sorted")
    return ordered


def _nonneg_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise BenchmarkCorpusError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class BaselineRetrievalPolicy:
    """Pinned offline retrieval policy shared by raw and semantic modes."""

    tokenizer_id: str
    estimator_version: str
    coverage_policy: str
    fixture_corpus_id: str
    fixture_package_path: str
    require_exact_target_source: bool
    allow_omit_required_raw: bool
    allow_network: bool
    allow_model_derived_expected_outcome: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tokenizer_id", _text(self.tokenizer_id, "tokenizer_id")
        )
        object.__setattr__(
            self,
            "estimator_version",
            _text(self.estimator_version, "estimator_version"),
        )
        object.__setattr__(
            self, "coverage_policy", _text(self.coverage_policy, "coverage_policy")
        )
        object.__setattr__(
            self,
            "fixture_corpus_id",
            _text(self.fixture_corpus_id, "fixture_corpus_id"),
        )
        object.__setattr__(
            self,
            "fixture_package_path",
            _text(self.fixture_package_path, "fixture_package_path"),
        )
        if not _bool(
            self.require_exact_target_source, "require_exact_target_source"
        ):
            raise BenchmarkCorpusError(
                "require_exact_target_source must be true for offline corpus tasks"
            )
        if _bool(self.allow_omit_required_raw, "allow_omit_required_raw"):
            raise BenchmarkCorpusError(
                "allow_omit_required_raw must be false (coverage is hard)"
            )
        if _bool(self.allow_network, "allow_network"):
            raise BenchmarkCorpusError(
                "allow_network must be false (offline corpus)"
            )
        if _bool(
            self.allow_model_derived_expected_outcome,
            "allow_model_derived_expected_outcome",
        ):
            raise BenchmarkCorpusError(
                "allow_model_derived_expected_outcome must be false"
            )
        if self.coverage_policy != "hard_coverage_no_omit_required":
            raise BenchmarkCorpusError(
                "coverage_policy must be hard_coverage_no_omit_required"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_RETRIEVAL_SCHEMA,
            "tokenizer_id": self.tokenizer_id,
            "estimator_version": self.estimator_version,
            "coverage_policy": self.coverage_policy,
            "fixture_corpus_id": self.fixture_corpus_id,
            "fixture_package_path": self.fixture_package_path,
            "require_exact_target_source": self.require_exact_target_source,
            "allow_omit_required_raw": self.allow_omit_required_raw,
            "allow_network": self.allow_network,
            "allow_model_derived_expected_outcome": (
                self.allow_model_derived_expected_outcome
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineRetrievalPolicy":
        schema = payload.get("schema")
        if schema is not None and schema != BASELINE_RETRIEVAL_SCHEMA:
            raise BenchmarkCorpusError(
                f"unsupported baseline retrieval schema {schema!r}"
            )
        return cls(
            tokenizer_id=str(payload["tokenizer_id"]),
            estimator_version=str(payload["estimator_version"]),
            coverage_policy=str(payload["coverage_policy"]),
            fixture_corpus_id=str(payload["fixture_corpus_id"]),
            fixture_package_path=str(payload["fixture_package_path"]),
            require_exact_target_source=bool(
                payload["require_exact_target_source"]
            ),
            allow_omit_required_raw=bool(payload["allow_omit_required_raw"]),
            allow_network=bool(payload["allow_network"]),
            allow_model_derived_expected_outcome=bool(
                payload["allow_model_derived_expected_outcome"]
            ),
        )


@dataclass(frozen=True)
class TaskOracle:
    """Independently authored task oracle (not harness measurement output)."""

    invalidation_symbol_ids: tuple[str, ...]
    selected_test_node_ids: tuple[str, ...]
    full_suite_test_node_ids: tuple[str, ...]
    proof_obligation_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    uncertainty: tuple[str, ...]
    expected_false_negatives: int
    fallback: str
    # Separated outcome authorities:
    candidate_verification_outcome: str
    production_acceptance: str
    oracle_authority: str

    def __post_init__(self) -> None:
        inv = _sorted_unique(
            self.invalidation_symbol_ids, "invalidation_symbol_ids"
        )
        selected = _sorted_unique(
            self.selected_test_node_ids, "selected_test_node_ids"
        )
        full = _sorted_unique(
            self.full_suite_test_node_ids, "full_suite_test_node_ids"
        )
        proofs = _sorted_unique(
            self.proof_obligation_ids, "proof_obligation_ids"
        )
        assumptions = _sorted_unique(self.assumptions, "assumptions")
        uncertainty = _sorted_unique(self.uncertainty, "uncertainty")
        false_neg = _nonneg_int(
            self.expected_false_negatives, "expected_false_negatives"
        )
        fallback = _text(self.fallback, "fallback")
        if fallback not in {"none", "full_pytest", "full_proofs", "both"}:
            raise BenchmarkCorpusError(f"unsupported fallback {fallback!r}")
        if not set(selected).issubset(set(full)):
            raise BenchmarkCorpusError(
                "selected_test_node_ids must be a subset of full_suite_test_node_ids"
            )
        cand = _text(
            self.candidate_verification_outcome, "candidate_verification_outcome"
        )
        if cand not in CANDIDATE_VERIFICATION_OUTCOMES:
            raise BenchmarkCorpusError(
                f"unsupported candidate_verification_outcome {cand!r}"
            )
        prod = _text(self.production_acceptance, "production_acceptance")
        if prod not in PRODUCTION_ACCEPTANCE_OUTCOMES:
            raise BenchmarkCorpusError(
                f"unsupported production_acceptance {prod!r}"
            )
        # Production must never silently treat replay verification as accept.
        if prod == "accepted":  # pragma: no cover - not in enum
            raise BenchmarkCorpusError(
                "production_acceptance must not be accepted for corpus fixtures"
            )
        authority = _text(self.oracle_authority, "oracle_authority")
        if authority not in {
            "reviewed_fixture_authority",
            "controlled_fixture_oracle",
        }:
            raise BenchmarkCorpusError(
                f"unsupported oracle_authority {authority!r}"
            )
        if authority == "benchmark_implementation_output":
            raise BenchmarkCorpusError(
                "oracle must not be derived from benchmark implementation output"
            )
        object.__setattr__(self, "invalidation_symbol_ids", inv)
        object.__setattr__(self, "selected_test_node_ids", selected)
        object.__setattr__(self, "full_suite_test_node_ids", full)
        object.__setattr__(self, "proof_obligation_ids", proofs)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "expected_false_negatives", false_neg)
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "candidate_verification_outcome", cand)
        object.__setattr__(self, "production_acceptance", prod)
        object.__setattr__(self, "oracle_authority", authority)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_ORACLE_SCHEMA,
            "invalidation_symbol_ids": list(self.invalidation_symbol_ids),
            "selected_test_node_ids": list(self.selected_test_node_ids),
            "full_suite_test_node_ids": list(self.full_suite_test_node_ids),
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "assumptions": list(self.assumptions),
            "uncertainty": list(self.uncertainty),
            "expected_false_negatives": self.expected_false_negatives,
            "fallback": self.fallback,
            "candidate_verification_outcome": self.candidate_verification_outcome,
            "production_acceptance": self.production_acceptance,
            "oracle_authority": self.oracle_authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskOracle":
        schema = payload.get("schema")
        if schema is not None and schema != TASK_ORACLE_SCHEMA:
            raise BenchmarkCorpusError(f"unsupported task oracle schema {schema!r}")
        return cls(
            invalidation_symbol_ids=tuple(payload["invalidation_symbol_ids"]),
            selected_test_node_ids=tuple(payload["selected_test_node_ids"]),
            full_suite_test_node_ids=tuple(payload["full_suite_test_node_ids"]),
            proof_obligation_ids=tuple(payload["proof_obligation_ids"]),
            assumptions=tuple(payload["assumptions"]),
            uncertainty=tuple(payload["uncertainty"]),
            expected_false_negatives=int(payload["expected_false_negatives"]),
            fallback=str(payload["fallback"]),
            candidate_verification_outcome=str(
                payload["candidate_verification_outcome"]
            ),
            production_acceptance=str(payload["production_acceptance"]),
            oracle_authority=str(payload["oracle_authority"]),
        )


@dataclass(frozen=True)
class CandidatePatch:
    """Oracle/replay candidate only; never model output or production-eligible."""

    candidate_id: str
    source: str
    production_eligible: bool
    base_mutation_case_id: str
    # Optional explicit path operations (path -> content replace map) as recipes.
    # Empty means "use the pinned base mutation tree as the candidate tree".
    declared_paths: tuple[str, ...]
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _text(self.candidate_id, "candidate_id")
        )
        source = _text(self.source, "source")
        if source not in {
            "oracle_replay_fixture",
            "controlled_fixture_mutation",
        }:
            raise BenchmarkCorpusError(f"unsupported candidate source {source!r}")
        if _bool(self.production_eligible, "production_eligible"):
            raise BenchmarkCorpusError(
                "checked-in candidates must set production_eligible=false"
            )
        object.__setattr__(
            self,
            "base_mutation_case_id",
            _text(self.base_mutation_case_id, "base_mutation_case_id"),
        )
        object.__setattr__(
            self,
            "declared_paths",
            _sorted_unique(self.declared_paths, "declared_paths"),
        )
        object.__setattr__(self, "notes", _text(self.notes, "notes"))
        object.__setattr__(self, "source", source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_PATCH_SCHEMA,
            "candidate_id": self.candidate_id,
            "source": self.source,
            "production_eligible": self.production_eligible,
            "base_mutation_case_id": self.base_mutation_case_id,
            "declared_paths": list(self.declared_paths),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidatePatch":
        schema = payload.get("schema")
        if schema is not None and schema != CANDIDATE_PATCH_SCHEMA:
            raise BenchmarkCorpusError(
                f"unsupported candidate patch schema {schema!r}"
            )
        return cls(
            candidate_id=str(payload["candidate_id"]),
            source=str(payload["source"]),
            production_eligible=bool(payload["production_eligible"]),
            base_mutation_case_id=str(payload["base_mutation_case_id"]),
            declared_paths=tuple(payload.get("declared_paths") or ()),
            notes=str(payload["notes"]),
        )


@dataclass(frozen=True)
class BenchmarkTask:
    """One stable offline-runnable maintenance task in the exactly-40 corpus."""

    task_id: str
    category: str
    objective: str
    target_paths: tuple[str, ...]
    base_mutation_case_id: str
    risk: str
    expected_route: str
    multi_file: bool
    frontier_or_human: bool
    baseline_retrieval: BaselineRetrievalPolicy
    oracle: TaskOracle
    candidate: CandidatePatch

    def __post_init__(self) -> None:
        task_id = _text(self.task_id, "task_id")
        if not task_id.startswith("sch-bench-"):
            raise BenchmarkCorpusError(
                f"task_id must use stable sch-bench- prefix: {task_id}"
            )
        category = _text(self.category, "category")
        if category not in TASK_CATEGORIES:
            raise BenchmarkCorpusError(f"unsupported category {category!r}")
        objective = _text(self.objective, "objective")
        targets = _sorted_unique(self.target_paths, "target_paths")
        if not targets:
            raise BenchmarkCorpusError(f"{task_id}: target_paths must be nonempty")
        mutation = _text(self.base_mutation_case_id, "base_mutation_case_id")
        risk = _text(self.risk, "risk")
        if risk not in RISK_LEVELS:
            raise BenchmarkCorpusError(f"unsupported risk {risk!r}")
        route = _text(self.expected_route, "expected_route")
        if route not in MODEL_ROUTES:
            raise BenchmarkCorpusError(f"unsupported expected_route {route!r}")
        multi_file = _bool(self.multi_file, "multi_file")
        frontier = _bool(self.frontier_or_human, "frontier_or_human")
        if not isinstance(self.baseline_retrieval, BaselineRetrievalPolicy):
            raise BenchmarkCorpusError(
                "baseline_retrieval must be a BaselineRetrievalPolicy"
            )
        if not isinstance(self.oracle, TaskOracle):
            raise BenchmarkCorpusError("oracle must be a TaskOracle")
        if not isinstance(self.candidate, CandidatePatch):
            raise BenchmarkCorpusError("candidate must be a CandidatePatch")
        if self.candidate.production_eligible:
            raise BenchmarkCorpusError(
                f"{task_id}: candidate.production_eligible must be false"
            )
        if self.candidate.base_mutation_case_id != mutation:
            raise BenchmarkCorpusError(
                f"{task_id}: candidate base_mutation_case_id must match task"
            )
        if category == "multi_file_refactor" and not multi_file:
            raise BenchmarkCorpusError(
                f"{task_id}: multi_file_refactor requires multi_file=true"
            )
        if multi_file and len(targets) < 2:
            raise BenchmarkCorpusError(
                f"{task_id}: multi_file tasks require at least two target_paths"
            )
        if category == "rejection_or_escalation":
            if self.oracle.candidate_verification_outcome not in {
                "reject",
                "escalate",
                "fail",
            }:
                raise BenchmarkCorpusError(
                    f"{task_id}: rejection/escalation tasks must not expect pass"
                )
        if frontier and route not in {
            "frontier_model",
            "human_review_required",
        }:
            raise BenchmarkCorpusError(
                f"{task_id}: frontier_or_human requires frontier/human route"
            )
        if route in {"frontier_model", "human_review_required"} and not frontier:
            raise BenchmarkCorpusError(
                f"{task_id}: frontier/human route requires frontier_or_human=true"
            )
        # Production acceptance never equals candidate verification "pass".
        if (
            self.oracle.candidate_verification_outcome == "pass"
            and self.oracle.production_acceptance
            not in {"not_applicable", "rejected", "blocked"}
        ):
            raise BenchmarkCorpusError(
                f"{task_id}: production acceptance must stay non-accepting"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "target_paths", targets)
        object.__setattr__(self, "base_mutation_case_id", mutation)
        object.__setattr__(self, "risk", risk)
        object.__setattr__(self, "expected_route", route)
        object.__setattr__(self, "multi_file", multi_file)
        object.__setattr__(self, "frontier_or_human", frontier)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_TASK_SCHEMA,
            "task_id": self.task_id,
            "category": self.category,
            "objective": self.objective,
            "target_paths": list(self.target_paths),
            "base_mutation_case_id": self.base_mutation_case_id,
            "risk": self.risk,
            "expected_route": self.expected_route,
            "multi_file": self.multi_file,
            "frontier_or_human": self.frontier_or_human,
            "baseline_retrieval": self.baseline_retrieval.to_dict(),
            "oracle": self.oracle.to_dict(),
            "candidate": self.candidate.to_dict(),
            "production_eligible": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkTask":
        schema = payload.get("schema")
        if schema is not None and schema != BENCHMARK_TASK_SCHEMA:
            raise BenchmarkCorpusError(f"unsupported task schema {schema!r}")
        production_eligible = payload.get("production_eligible", False)
        if production_eligible:
            raise BenchmarkCorpusError(
                "task production_eligible must be false for corpus fixtures"
            )
        return cls(
            task_id=str(payload["task_id"]),
            category=str(payload["category"]),
            objective=str(payload["objective"]),
            target_paths=tuple(payload["target_paths"]),
            base_mutation_case_id=str(payload["base_mutation_case_id"]),
            risk=str(payload["risk"]),
            expected_route=str(payload["expected_route"]),
            multi_file=bool(payload["multi_file"]),
            frontier_or_human=bool(payload["frontier_or_human"]),
            baseline_retrieval=BaselineRetrievalPolicy.from_dict(
                payload["baseline_retrieval"]
            ),
            oracle=TaskOracle.from_dict(payload["oracle"]),
            candidate=CandidatePatch.from_dict(payload["candidate"]),
        )
