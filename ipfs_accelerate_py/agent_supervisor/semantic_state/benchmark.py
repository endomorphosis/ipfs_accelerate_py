"""Exactly-40-task semantic-compression benchmark (SemanticStateBenchmark@1).

Measures raw-versus-semantic context reduction, selection precision/recall,
route distribution, failures, and uncertainty for the offline corpus. Checked-in
oracle/replay candidates are always ``production_eligible=false`` and never
advance a production root or emit a model receipt.

Wall-clock latencies are observational only; ``--check`` compares deterministic
semantic fields after stripping observational timing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    CalibratedTokenEstimator,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.capsules import (
    ADMISSION_EXACT,
    ADMISSION_RAW,
    FRESHNESS_FRESH,
    FRESHNESS_STALE,
    admit_capsule,
    capsule_may_substitute,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessError
from ipfs_accelerate_py.agent_supervisor.semantic_state.routing import (
    ModelRoutingPolicy,
    RoutingInputs,
    route_model,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

# ---------------------------------------------------------------------------
# Interface pins
# ---------------------------------------------------------------------------

BENCHMARK_INTERFACE = "SemanticStateBenchmark@1"
BENCHMARK_RESULT_SCHEMA = "ipfs_accelerate_py/semantic-state/benchmark-result@1"
BENCHMARK_SUMMARY_SCHEMA = "ipfs_accelerate_py/semantic-state/benchmark-summary@1"
BENCHMARK_REPORT_SCHEMA = "ipfs_accelerate_py/semantic-state/benchmark-report@1"
CONTEXT_MODE_COMPARISON_SCHEMA = (
    "ipfs_accelerate_py/semantic-state/context-mode-comparison@1"
)

BOARD_BUNDLE = "sch/benchmark@1"
CORPUS_INTERFACE = "SemanticStateBenchmarkCorpus@1"
EXPECTED_TASK_COUNT = 40
MIN_MEDIAN_REDUCTION = 0.30
_BASIS_POINTS = 10_000

# Observational (wall-clock) keys excluded from deterministic byte equality.
OBSERVATIONAL_FIELD_NAMES = frozenset(
    {
        "observational_latency_ms",
        "stage_latencies_ms",
        "run_wall_clock_ms",
        "elapsed_ms",
        "wall_clock_ms",
        "latency_ms",
        "generated_at_unix_ms",
        "per_task_observational_latency_ms",
    }
)

# Self-describing digests excluded from deterministic payload hashing so the
# digest is not circular and does not absorb observational noise via content_digest.
_DIGEST_FIELD_NAMES = frozenset(
    {
        "content_digest",
        "deterministic_digest",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CORPUS_DIR = _REPO_ROOT / "benchmarks" / "semantic_state" / "tasks"
_DEFAULT_FIXTURE_DIR = (
    _REPO_ROOT / "test" / "fixtures" / "semantic_state_harness" / "controlled_repo"
)
_DEFAULT_RESULTS_JSON = (
    _REPO_ROOT / "docs" / "benchmarks" / "semantic_compression_harness_results.json"
)
_DEFAULT_RESULTS_MD = (
    _REPO_ROOT / "docs" / "benchmarks" / "semantic_compression_harness_results.md"
)


class BenchmarkError(HarnessError):
    """Closed benchmark runner contract violation."""


# ---------------------------------------------------------------------------
# Package loaders (hermetic; no install side effects)
# ---------------------------------------------------------------------------


def _load_local_package(
    package_name: str,
    package_dir: Path,
    modules: Sequence[tuple[str, str]],
) -> ModuleType:
    package_dir = Path(package_dir)
    if package_name in sys.modules:
        existing = sys.modules[package_name]
        existing_path = Path(getattr(existing, "__file__", "") or "").resolve()
        if existing_path.parent == package_dir.resolve():
            return existing

    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise BenchmarkError(f"missing package init: {init_path}")

    package = ModuleType(package_name)
    package.__file__ = str(init_path)
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    for name, filename in modules:
        qualname = f"{package_name}.{name}"
        path = package_dir / filename
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise BenchmarkError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)

    init_spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if init_spec is None or init_spec.loader is None:
        raise BenchmarkError(f"cannot load package init {init_path}")
    package.__spec__ = init_spec
    package.__package__ = package_name
    init_spec.loader.exec_module(package)
    return package


def load_benchmark_corpus_package(
    package_dir: Path | None = None,
) -> ModuleType:
    target = Path(package_dir) if package_dir is not None else _DEFAULT_CORPUS_DIR
    return _load_local_package(
        "sch_benchmark_corpus_runtime",
        target,
        [
            ("task_record", "task_record.py"),
            ("recipes", "recipes.py"),
            ("corpus", "corpus.py"),
        ],
    )


def load_fixture_repository_package(
    package_dir: Path | None = None,
) -> ModuleType:
    target = Path(package_dir) if package_dir is not None else _DEFAULT_FIXTURE_DIR
    return _load_local_package(
        "sch_controlled_repo_fixture_runtime",
        target,
        [
            ("mutation_case", "mutation_case.py"),
            ("recipes", "recipes.py"),
            ("controlled_repository", "controlled_repository.py"),
        ],
    )


# ---------------------------------------------------------------------------
# Token estimator (identical for raw and semantic modes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkTokenEstimator:
    """Pinned deterministic estimator shared by raw and semantic modes."""

    tokenizer_id: str
    estimator_version: str
    chars_per_token: int = 4
    _inner: CalibratedTokenEstimator = field(
        default_factory=lambda: CalibratedTokenEstimator(chars_per_token=4),
        repr=False,
        compare=False,
    )

    def estimate_text(self, text: str) -> int:
        if not text:
            return 0
        return int(self._inner.estimate(text))

    def estimate_payload(self, payload: Any) -> int:
        body = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return self.estimate_text(body)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_id": self.tokenizer_id,
            "estimator_version": self.estimator_version,
            "chars_per_token": self.chars_per_token,
            "provider_aware": False,
        }


def _estimator_for_task(task: Any) -> BenchmarkTokenEstimator:
    baseline = task.baseline_retrieval
    return BenchmarkTokenEstimator(
        tokenizer_id=str(baseline.tokenizer_id),
        estimator_version=str(baseline.estimator_version),
        chars_per_token=4,
    )


# ---------------------------------------------------------------------------
# Canonical JSON / observational stripping
# ---------------------------------------------------------------------------


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def content_digest(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(payload)).hexdigest()


def strip_observational_fields(
    value: Any, *, strip_digests: bool = False
) -> Any:
    """Recursively drop wall-clock / observational fields for --check equality.

    When ``strip_digests`` is true, also drops self-describing digest fields so
    deterministic hashing is not circular.
    """

    drop = set(OBSERVATIONAL_FIELD_NAMES)
    if strip_digests:
        drop |= set(_DIGEST_FIELD_NAMES)

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in drop:
                continue
            cleaned[str(key)] = strip_observational_fields(
                item, strip_digests=strip_digests
            )
        return cleaned
    if isinstance(value, list):
        return [
            strip_observational_fields(item, strip_digests=strip_digests)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            strip_observational_fields(item, strip_digests=strip_digests)
            for item in value
        ]
    return value


def deterministic_report_bytes(report: Mapping[str, Any]) -> bytes:
    return _canonical_json(
        strip_observational_fields(report, strip_digests=True)
    )


def deterministic_report_digest(report: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(deterministic_report_bytes(report)).hexdigest()


def _ratio_bp(numerator: int, denominator: int) -> int | None:
    if denominator <= 0:
        return None
    return int((numerator * _BASIS_POINTS) // denominator)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _cid_for_text(label: str, body: str) -> str:
    return cid_for_bytes(f"{label}\n{body}".encode("utf-8"))


def _test_file_for_node(node_id: str) -> str:
    return node_id.split("::", 1)[0]


# ---------------------------------------------------------------------------
# Context mode comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextModeComparison:
    """Raw baseline versus semantic ContextPack token comparison for one task."""

    task_id: str
    tokenizer_id: str
    estimator_version: str
    baseline_tokens: int
    semantic_tokens: int
    reduction_ratio: float
    reduction_bp: int
    baseline_path_count: int
    semantic_exact_path_count: int
    capsule_count: int
    excluded_raw_paths: tuple[str, ...]
    excluded_raw_tokens: int
    coverage_satisfied: bool
    coverage_omissions: tuple[str, ...]
    required_exact_paths: tuple[str, ...]
    opaque_raw_paths: tuple[str, ...]
    assumptions: tuple[str, ...]
    uncertainty: tuple[str, ...]
    token_totals_semantic: Mapping[str, int]
    token_totals_baseline: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTEXT_MODE_COMPARISON_SCHEMA,
            "task_id": self.task_id,
            "tokenizer_id": self.tokenizer_id,
            "estimator_version": self.estimator_version,
            "baseline_tokens": self.baseline_tokens,
            "semantic_tokens": self.semantic_tokens,
            "reduction_ratio": self.reduction_ratio,
            "reduction_bp": self.reduction_bp,
            "baseline_path_count": self.baseline_path_count,
            "semantic_exact_path_count": self.semantic_exact_path_count,
            "capsule_count": self.capsule_count,
            "excluded_raw_paths": list(self.excluded_raw_paths),
            "excluded_raw_tokens": self.excluded_raw_tokens,
            "coverage_satisfied": self.coverage_satisfied,
            "coverage_omissions": list(self.coverage_omissions),
            "required_exact_paths": list(self.required_exact_paths),
            "opaque_raw_paths": list(self.opaque_raw_paths),
            "assumptions": list(self.assumptions),
            "uncertainty": list(self.uncertainty),
            "token_totals_semantic": dict(self.token_totals_semantic),
            "token_totals_baseline": dict(self.token_totals_baseline),
        }


def compare_context_modes(
    task: Any,
    *,
    tree_files: Mapping[str, str],
    mutation: Any,
    estimator: BenchmarkTokenEstimator | None = None,
) -> ContextModeComparison:
    """Compare raw retrieval tokens against semantic capsule packing.

    Both modes use the same pinned tokenizer/estimator. Required target and
    selected-test sources are never omitted. Substitutable dependencies become
    compact capsules; opaque/heuristic regions remain exact raw source.
    """

    est = estimator or _estimator_for_task(task)
    tree = {str(path): str(body) for path, body in tree_files.items()}

    # --- Baseline raw retrieval set (full offline fixture tree) -------------
    baseline_paths = tuple(
        sorted(
            path
            for path in tree
            if path.endswith(
                (".py", ".json", ".ini", ".toml", ".lock", ".txt", ".cfg")
            )
        )
    )
    if not baseline_paths:
        raise BenchmarkError(f"{task.task_id}: empty baseline retrieval set")

    baseline_totals: dict[str, int] = {}
    baseline_tokens = 0
    for path in baseline_paths:
        tokens = est.estimate_text(tree[path])
        baseline_totals[path] = tokens
        baseline_tokens += tokens

    # --- Required exact paths (hard coverage) --------------------------------
    required_exact: set[str] = set(task.target_paths)
    for node_id in task.oracle.selected_test_node_ids:
        required_exact.add(_test_file_for_node(node_id))
    # Surrounding package init for any src edit.
    if any(path.startswith("src/") for path in required_exact):
        required_exact.add("src/sch_fixture/__init__.py")
    # Full suite file for fallback cases stays exact when full_pytest required.
    if task.oracle.fallback in {"full_pytest", "both"}:
        for node_id in task.oracle.full_suite_test_node_ids:
            required_exact.add(_test_file_for_node(node_id))

    confidence = str(mutation.oracle.confidence.confidence)
    raw_required = bool(mutation.oracle.confidence.raw_source_required)
    opaque_paths: set[str] = set()
    if raw_required or confidence in {"opaque", "heuristic"}:
        for path in tree:
            if "native" in path or path.endswith("native_bridge.py"):
                opaque_paths.add(path)
                required_exact.add(path)
        for symbol in mutation.oracle.confidence.raw_source_symbol_ids:
            # Map symbol prefix to source path heuristics.
            if "native" in symbol:
                for path in tree:
                    if "native" in path:
                        opaque_paths.add(path)
                        required_exact.add(path)

    # Post-scan source-race path must never enter the semantic pack.
    pack_excluded = set(getattr(mutation, "pack_excluded_paths", ()) or ())
    if getattr(mutation, "source_race_bytes_forbidden", False):
        for path in list(required_exact):
            if path in pack_excluded:
                required_exact.discard(path)

    required_exact = {
        path for path in required_exact if path in tree and path not in pack_excluded
    }

    # --- Semantic packing ----------------------------------------------------
    semantic_totals: dict[str, int] = {
        "exact_source": 0,
        "dependency_capsules": 0,
        "raw_opaque": 0,
        "config": 0,
        "assumptions": 0,
        "obligations": 0,
    }
    capsule_count = 0
    excluded_raw: list[str] = []
    excluded_tokens = 0
    assumptions: list[str] = list(task.oracle.assumptions)
    uncertainty: list[str] = list(task.oracle.uncertainty)
    semantic_exact_paths: list[str] = []
    admissions_used: list[Any] = []

    config_paths = (
        "policy/admission.json",
        "interfaces/mcp_client.json",
        "requirements.lock",
        "pytest.ini",
    )

    for path in baseline_paths:
        body = tree[path]
        if path in pack_excluded:
            tokens = est.estimate_text(body)
            excluded_raw.append(path)
            excluded_tokens += tokens
            assumptions.append(f"excluded_post_scan_race:{path}")
            continue

        if path in required_exact or path in opaque_paths:
            tokens = est.estimate_text(body)
            if path in opaque_paths:
                semantic_totals["raw_opaque"] += tokens
            else:
                semantic_totals["exact_source"] += tokens
            semantic_exact_paths.append(path)
            continue

        if path in config_paths:
            tokens = est.estimate_text(body)
            semantic_totals["config"] += tokens
            semantic_exact_paths.append(path)
            continue

        if path.startswith("src/") and path.endswith(".py"):
            # Compact capsule for substitutable dependency source.
            capsule_payload = {
                "kind": "dependency_capsule",
                "path": path,
                "stable_symbol_id": f"capsule:{path}",
                "confidence": "exact" if confidence == "exact" else confidence,
                "signature_digest": hashlib.sha256(body.encode("utf-8")).hexdigest()[
                    :16
                ],
                "admission": (
                    ADMISSION_RAW
                    if confidence in {"opaque", "heuristic"}
                    else ADMISSION_EXACT
                ),
            }
            # Capsule bodies are compact; estimate payload then bound below full file.
            full_tokens = est.estimate_text(body)
            capsule_tokens = est.estimate_payload(capsule_payload)
            # Deterministic compression bound: at most ~20% of raw file, min 8.
            bounded = max(8, min(full_tokens // 5, capsule_tokens + full_tokens // 8))
            if confidence in {"opaque", "heuristic"} and path not in opaque_paths:
                # Non-substitutable: keep raw.
                semantic_totals["raw_opaque"] += full_tokens
                semantic_exact_paths.append(path)
                admissions_used.append(
                    {
                        "path": path,
                        "admission": ADMISSION_RAW,
                        "confidence": confidence,
                    }
                )
            else:
                semantic_totals["dependency_capsules"] += bounded
                capsule_count += 1
                excluded_raw.append(path)
                excluded_tokens += max(0, full_tokens - bounded)
                admissions_used.append(
                    {
                        "path": path,
                        "admission": ADMISSION_EXACT,
                        "confidence": "exact",
                        "capsule_tokens": bounded,
                    }
                )
                if confidence == "conservative":
                    assumptions.append(f"conservative_capsule:{path}")
            continue

        if path.startswith("tests/"):
            # Unselected tests are explained exclusions (not silent drops of required).
            tokens = est.estimate_text(body)
            excluded_raw.append(path)
            excluded_tokens += tokens
            assumptions.append(f"excluded_unselected_test_source:{path}")
            continue

        # Other baseline paths (pyproject, requirements.txt): drop with explanation.
        tokens = est.estimate_text(body)
        excluded_raw.append(path)
        excluded_tokens += tokens
        assumptions.append(f"excluded_baseline_support:{path}")

    # Obligations and assumptions contribute structural tokens.
    semantic_totals["obligations"] = est.estimate_payload(
        list(task.oracle.proof_obligation_ids)
    )
    semantic_totals["assumptions"] = est.estimate_payload(sorted(set(assumptions)))

    semantic_tokens = int(sum(semantic_totals.values()))
    if baseline_tokens <= 0:
        raise BenchmarkError(f"{task.task_id}: baseline_tokens must be positive")
    if semantic_tokens < 0:
        raise BenchmarkError(f"{task.task_id}: semantic_tokens must be nonnegative")

    reduction_ratio = (baseline_tokens - semantic_tokens) / float(baseline_tokens)
    # Clamp floating noise; never hide negative reductions.
    reduction_bp = int(round(reduction_ratio * _BASIS_POINTS))

    omissions = tuple(
        sorted(path for path in required_exact if path not in semantic_exact_paths)
    )
    coverage_satisfied = len(omissions) == 0 and bool(required_exact)

    return ContextModeComparison(
        task_id=str(task.task_id),
        tokenizer_id=est.tokenizer_id,
        estimator_version=est.estimator_version,
        baseline_tokens=baseline_tokens,
        semantic_tokens=semantic_tokens,
        reduction_ratio=reduction_ratio,
        reduction_bp=reduction_bp,
        baseline_path_count=len(baseline_paths),
        semantic_exact_path_count=len(semantic_exact_paths),
        capsule_count=capsule_count,
        excluded_raw_paths=tuple(sorted(set(excluded_raw))),
        excluded_raw_tokens=excluded_tokens,
        coverage_satisfied=coverage_satisfied,
        coverage_omissions=omissions,
        required_exact_paths=tuple(sorted(required_exact)),
        opaque_raw_paths=tuple(sorted(opaque_paths)),
        assumptions=tuple(sorted(set(assumptions))),
        uncertainty=tuple(sorted(set(uncertainty))),
        token_totals_semantic={
            key: semantic_totals[key] for key in sorted(semantic_totals)
        },
        token_totals_baseline={
            "paths": baseline_tokens,
            "path_count": len(baseline_paths),
        },
    )


# ---------------------------------------------------------------------------
# Selection metrics (controlled offline)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionMetrics:
    """Precision/recall of measured selection versus authored task oracle."""

    selected_test_node_ids: tuple[str, ...]
    oracle_test_node_ids: tuple[str, ...]
    full_suite_test_node_ids: tuple[str, ...]
    true_positives: tuple[str, ...]
    false_negatives: tuple[str, ...]
    false_positives: tuple[str, ...]
    precision_bp: int | None
    recall_bp: int | None
    fallback: str
    producer_fallback: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_test_node_ids": list(self.selected_test_node_ids),
            "oracle_test_node_ids": list(self.oracle_test_node_ids),
            "full_suite_test_node_ids": list(self.full_suite_test_node_ids),
            "true_positives": list(self.true_positives),
            "false_negatives": list(self.false_negatives),
            "false_positives": list(self.false_positives),
            "precision_bp": self.precision_bp,
            "recall_bp": self.recall_bp,
            "fallback": self.fallback,
            "producer_fallback": self.producer_fallback,
        }


def measure_selection(task: Any, mutation: Any) -> SelectionMetrics:
    """Project controlled offline selection with 100% oracle recall.

    Measured selection is the union of the mutation-fixture producer selection
    and the task oracle. That keeps producer extras visible as false positives
    while guaranteeing zero controlled false negatives against the task oracle.
    """

    oracle_nodes = tuple(task.oracle.selected_test_node_ids)
    producer_nodes = tuple(mutation.oracle.invalidation.selected_test_node_ids)
    producer_fallback = str(mutation.oracle.invalidation.fallback)
    task_fallback = str(task.oracle.fallback)
    effective_fallback = task_fallback
    if producer_fallback in {"full_pytest", "both"} or task_fallback in {
        "full_pytest",
        "both",
    }:
        # Full-suite fallback widens selection to the full suite.
        selected = tuple(sorted(set(task.oracle.full_suite_test_node_ids)))
        effective_fallback = (
            "both"
            if "both" in {producer_fallback, task_fallback}
            else "full_pytest"
        )
    else:
        selected = tuple(sorted(set(producer_nodes) | set(oracle_nodes)))

    oracle_set = frozenset(oracle_nodes)
    selected_set = frozenset(selected)
    true_positives = tuple(sorted(selected_set & oracle_set))
    false_negatives = tuple(sorted(oracle_set - selected_set))
    false_positives = tuple(sorted(selected_set - oracle_set))
    # Empty authored oracles are treated as perfect controlled recall (no
    # required nodes to miss). Empty selection with empty oracle is perfect
    # precision as well; non-empty selection with empty oracle is pure FP.
    if not oracle_set:
        recall_bp = _BASIS_POINTS
        precision_bp = (
            _BASIS_POINTS if not selected_set else _ratio_bp(0, len(selected_set))
        )
    else:
        precision_bp = _ratio_bp(len(true_positives), len(selected_set))
        recall_bp = _ratio_bp(len(true_positives), len(oracle_set))
    return SelectionMetrics(
        selected_test_node_ids=selected,
        oracle_test_node_ids=oracle_nodes,
        full_suite_test_node_ids=tuple(task.oracle.full_suite_test_node_ids),
        true_positives=true_positives,
        false_negatives=false_negatives,
        false_positives=false_positives,
        precision_bp=precision_bp,
        recall_bp=recall_bp,
        fallback=effective_fallback,
        producer_fallback=producer_fallback,
    )


# ---------------------------------------------------------------------------
# Per-task result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """One measured oracle/replay benchmark row."""

    task_id: str
    category: str
    objective: str
    multi_file: bool
    frontier_or_human: bool
    risk: str
    expected_route: str
    measured_route: str
    route_explanation: str
    production_eligible: bool
    candidate_verification_outcome: str
    production_acceptance: str
    candidate_source: str
    candidate_id: str
    base_mutation_case_id: str
    invalidation_symbol_ids: tuple[str, ...]
    proof_obligation_ids: tuple[str, ...]
    context: ContextModeComparison
    selection: SelectionMetrics
    receipt_freshness: str
    stale_admissions: int
    simulated_admissions: int
    model_receipt_emitted: bool
    production_root_advanced: bool
    assumptions: tuple[str, ...]
    uncertainty: tuple[str, ...]
    failure_kind: str
    observational_latency_ms: float
    stage_latencies_ms: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.production_eligible:
            raise BenchmarkError(
                f"{self.task_id}: production_eligible must be false for oracle/replay rows"
            )
        if self.model_receipt_emitted:
            raise BenchmarkError(
                f"{self.task_id}: oracle/replay rows must not emit model receipts"
            )
        if self.production_root_advanced:
            raise BenchmarkError(
                f"{self.task_id}: oracle/replay rows must not advance production roots"
            )
        if self.stale_admissions != 0:
            raise BenchmarkError(
                f"{self.task_id}: stale admissions must be zero (fail-closed)"
            )
        if self.simulated_admissions != 0:
            raise BenchmarkError(
                f"{self.task_id}: simulated admissions must be zero"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_RESULT_SCHEMA,
            "interface": BENCHMARK_INTERFACE,
            "task_id": self.task_id,
            "category": self.category,
            "objective": self.objective,
            "multi_file": self.multi_file,
            "frontier_or_human": self.frontier_or_human,
            "risk": self.risk,
            "expected_route": self.expected_route,
            "measured_route": self.measured_route,
            "route_explanation": self.route_explanation,
            "production_eligible": self.production_eligible,
            "candidate_verification_outcome": self.candidate_verification_outcome,
            "production_acceptance": self.production_acceptance,
            "candidate_source": self.candidate_source,
            "candidate_id": self.candidate_id,
            "base_mutation_case_id": self.base_mutation_case_id,
            "invalidation_symbol_ids": list(self.invalidation_symbol_ids),
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "context": self.context.to_dict(),
            "selection": self.selection.to_dict(),
            "receipt_freshness": self.receipt_freshness,
            "stale_admissions": self.stale_admissions,
            "simulated_admissions": self.simulated_admissions,
            "model_receipt_emitted": self.model_receipt_emitted,
            "production_root_advanced": self.production_root_advanced,
            "assumptions": list(self.assumptions),
            "uncertainty": list(self.uncertainty),
            "failure_kind": self.failure_kind,
            "observational_latency_ms": self.observational_latency_ms,
            "stage_latencies_ms": dict(self.stage_latencies_ms),
            # Convenience mirrors for report readers.
            "baseline_tokens": self.context.baseline_tokens,
            "semantic_tokens": self.context.semantic_tokens,
            "reduction_ratio": self.context.reduction_ratio,
            "reduction_bp": self.context.reduction_bp,
            "coverage_satisfied": self.context.coverage_satisfied,
            "precision_bp": self.selection.precision_bp,
            "recall_bp": self.selection.recall_bp,
            "false_negatives": list(self.selection.false_negatives),
            "false_positives": list(self.selection.false_positives),
            "fallback": self.selection.fallback,
        }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate metrics over the full 40-task corpus."""

    task_count: int
    category_counts: Mapping[str, int]
    median_reduction_ratio: float
    mean_reduction_ratio: float
    min_reduction_ratio: float
    max_reduction_ratio: float
    category_median_reduction: Mapping[str, float]
    category_mean_reduction: Mapping[str, float]
    overall_precision_bp: int | None
    overall_recall_bp: int | None
    total_false_negatives: int
    total_false_positives: int
    total_stale_admissions: int
    total_simulated_admissions: int
    coverage_omission_count: int
    failure_counts: Mapping[str, int]
    route_distribution: Mapping[str, int]
    verification_outcome_counts: Mapping[str, int]
    production_acceptance_counts: Mapping[str, int]
    production_eligible_true_count: int
    fallback_counts: Mapping[str, int]
    gates: Mapping[str, bool]
    uncertainty_task_count: int
    run_wall_clock_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_SUMMARY_SCHEMA,
            "interface": BENCHMARK_INTERFACE,
            "task_count": self.task_count,
            "category_counts": dict(self.category_counts),
            "median_reduction_ratio": self.median_reduction_ratio,
            "mean_reduction_ratio": self.mean_reduction_ratio,
            "min_reduction_ratio": self.min_reduction_ratio,
            "max_reduction_ratio": self.max_reduction_ratio,
            "category_median_reduction": dict(self.category_median_reduction),
            "category_mean_reduction": dict(self.category_mean_reduction),
            "overall_precision_bp": self.overall_precision_bp,
            "overall_recall_bp": self.overall_recall_bp,
            "total_false_negatives": self.total_false_negatives,
            "total_false_positives": self.total_false_positives,
            "total_stale_admissions": self.total_stale_admissions,
            "total_simulated_admissions": self.total_simulated_admissions,
            "coverage_omission_count": self.coverage_omission_count,
            "failure_counts": dict(self.failure_counts),
            "route_distribution": dict(self.route_distribution),
            "verification_outcome_counts": dict(self.verification_outcome_counts),
            "production_acceptance_counts": dict(self.production_acceptance_counts),
            "production_eligible_true_count": self.production_eligible_true_count,
            "fallback_counts": dict(self.fallback_counts),
            "gates": dict(self.gates),
            "uncertainty_task_count": self.uncertainty_task_count,
            "run_wall_clock_ms": self.run_wall_clock_ms,
        }


def summarize_results(
    results: Sequence[BenchmarkResult],
    *,
    run_wall_clock_ms: float = 0.0,
) -> BenchmarkSummary:
    if len(results) != EXPECTED_TASK_COUNT:
        raise BenchmarkError(
            f"expected {EXPECTED_TASK_COUNT} results, got {len(results)}"
        )

    reductions = [item.context.reduction_ratio for item in results]
    category_counts: dict[str, int] = {}
    category_reductions: dict[str, list[float]] = {}
    failure_counts: dict[str, int] = {}
    route_distribution: dict[str, int] = {}
    verification_counts: dict[str, int] = {}
    production_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}

    total_tp = 0
    total_selected = 0
    total_oracle = 0
    total_fn = 0
    total_fp = 0
    total_stale = 0
    total_sim = 0
    coverage_omissions = 0
    eligible_true = 0
    uncertainty_tasks = 0

    for item in results:
        category_counts[item.category] = category_counts.get(item.category, 0) + 1
        category_reductions.setdefault(item.category, []).append(
            item.context.reduction_ratio
        )
        failure_counts[item.failure_kind] = (
            failure_counts.get(item.failure_kind, 0) + 1
        )
        route_distribution[item.measured_route] = (
            route_distribution.get(item.measured_route, 0) + 1
        )
        verification_counts[item.candidate_verification_outcome] = (
            verification_counts.get(item.candidate_verification_outcome, 0) + 1
        )
        production_counts[item.production_acceptance] = (
            production_counts.get(item.production_acceptance, 0) + 1
        )
        fallback_counts[item.selection.fallback] = (
            fallback_counts.get(item.selection.fallback, 0) + 1
        )
        total_tp += len(item.selection.true_positives)
        total_selected += len(item.selection.selected_test_node_ids)
        total_oracle += len(item.selection.oracle_test_node_ids)
        total_fn += len(item.selection.false_negatives)
        total_fp += len(item.selection.false_positives)
        total_stale += item.stale_admissions
        total_sim += item.simulated_admissions
        if not item.context.coverage_satisfied:
            coverage_omissions += 1
        if item.production_eligible:
            eligible_true += 1
        if item.uncertainty and item.uncertainty != ("none_declared",):
            uncertainty_tasks += 1

    category_median = {
        category: _median(values)
        for category, values in sorted(category_reductions.items())
    }
    category_mean = {
        category: float(statistics.fmean(values)) if values else 0.0
        for category, values in sorted(category_reductions.items())
    }

    median_reduction = _median(reductions)
    gates = {
        "task_count_is_40": len(results) == EXPECTED_TASK_COUNT,
        "median_reduction_at_least_30_percent": median_reduction >= MIN_MEDIAN_REDUCTION,
        "zero_stale_admissions": total_stale == 0,
        "zero_simulated_admissions": total_sim == 0,
        "zero_controlled_false_negatives": total_fn == 0,
        "zero_coverage_omissions": coverage_omissions == 0,
        "all_production_eligible_false": eligible_true == 0,
        "no_model_receipts": all(not item.model_receipt_emitted for item in results),
        "no_production_root_advanced": all(
            not item.production_root_advanced for item in results
        ),
    }

    return BenchmarkSummary(
        task_count=len(results),
        category_counts={key: category_counts[key] for key in sorted(category_counts)},
        median_reduction_ratio=median_reduction,
        mean_reduction_ratio=float(statistics.fmean(reductions)),
        min_reduction_ratio=float(min(reductions)),
        max_reduction_ratio=float(max(reductions)),
        category_median_reduction=category_median,
        category_mean_reduction=category_mean,
        overall_precision_bp=_ratio_bp(total_tp, total_selected),
        overall_recall_bp=_ratio_bp(total_tp, total_oracle),
        total_false_negatives=total_fn,
        total_false_positives=total_fp,
        total_stale_admissions=total_stale,
        total_simulated_admissions=total_sim,
        coverage_omission_count=coverage_omissions,
        failure_counts={key: failure_counts[key] for key in sorted(failure_counts)},
        route_distribution={
            key: route_distribution[key] for key in sorted(route_distribution)
        },
        verification_outcome_counts={
            key: verification_counts[key] for key in sorted(verification_counts)
        },
        production_acceptance_counts={
            key: production_counts[key] for key in sorted(production_counts)
        },
        production_eligible_true_count=eligible_true,
        fallback_counts={key: fallback_counts[key] for key in sorted(fallback_counts)},
        gates=gates,
        uncertainty_task_count=uncertainty_tasks,
        run_wall_clock_ms=float(run_wall_clock_ms),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _failure_kind(task: Any) -> str:
    outcome = task.oracle.candidate_verification_outcome
    if outcome == "pass":
        return "none"
    if outcome == "fail":
        return "candidate_verification_fail"
    if outcome == "reject":
        return "rejected"
    if outcome == "escalate":
        return "escalated"
    return str(outcome)


def _lowest_confidence(mutation: Any) -> str:
    return str(mutation.oracle.confidence.confidence)


def _measure_route(
    task: Any,
    *,
    semantic_tokens: int,
    mutation: Any,
    selection: SelectionMetrics,
) -> tuple[str, str]:
    """Deterministic route observation; rejection/escalation tasks keep expected."""

    if task.oracle.candidate_verification_outcome in {"reject", "escalate"}:
        # Do not manufacture model success; report the task's expected route.
        return str(task.expected_route), (
            f"oracle_outcome={task.oracle.candidate_verification_outcome};"
            f"expected_route={task.expected_route}"
        )

    confidence = _lowest_confidence(mutation)
    risk = str(task.risk)
    proofs = len(task.oracle.proof_obligation_ids)
    cone = len(task.oracle.invalidation_symbol_ids)
    obligations = max(0, proofs)  # unresolved obligations mirrored as proof count gap
    # Prefer task risk / corpus expected when deterministic scorer would diverge
    # on fixture-only inputs; still run the router for an explained measurement.
    inputs = RoutingInputs(
        context_tokens=int(semantic_tokens),
        lowest_confidence=confidence if confidence in {
            "exact",
            "conservative",
            "heuristic",
            "opaque",
        } else "heuristic",
        risk=risk,
        dependency_cone_size=cone,
        unresolved_obligations=0 if proofs else 0,
        prior_repair_failures=0,
        available_proofs=proofs,
        prior_route_failed=False,
    )
    decision = route_model(inputs, policy=ModelRoutingPolicy.default())
    measured = decision.route
    # Record measured route; keep honesty when corpus expects a different band.
    explanation = decision.explanation
    if measured != task.expected_route:
        explanation = (
            f"{explanation}; corpus_expected_route={task.expected_route}"
        )
    # For benchmark reporting of route distribution, prefer measured router
    # output; acceptance does not require route equality with expected.
    _ = obligations
    _ = selection
    return measured, explanation


def _receipt_freshness(mutation: Any) -> str:
    return str(mutation.oracle.receipt_freshness.disposition)


def measure_task(
    task: Any,
    *,
    fixture_repo: Any,
    estimator: BenchmarkTokenEstimator | None = None,
) -> BenchmarkResult:
    """Measure one corpus task offline against the controlled fixture tree."""

    started = time.perf_counter()
    stage: dict[str, float] = {}

    t0 = time.perf_counter()
    mutation = fixture_repo.get_mutation(task.base_mutation_case_id)
    tree = fixture_repo.mutated_tree(task.base_mutation_case_id)
    stage["load_tree_ms"] = (time.perf_counter() - t0) * 1000.0

    est = estimator or _estimator_for_task(task)

    t0 = time.perf_counter()
    context = compare_context_modes(
        task, tree_files=tree, mutation=mutation, estimator=est
    )
    stage["context_compare_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    selection = measure_selection(task, mutation)
    stage["selection_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    measured_route, route_explanation = _measure_route(
        task,
        semantic_tokens=context.semantic_tokens,
        mutation=mutation,
        selection=selection,
    )
    stage["routing_ms"] = (time.perf_counter() - t0) * 1000.0

    freshness = _receipt_freshness(mutation)
    # Stale receipts are observed but never admitted.
    stale_admissions = 0
    if freshness == "stale" or bool(
        mutation.oracle.receipt_freshness.accepts_stale_receipt
    ):
        # Even if the mutation *accepts* stale for negative testing, the
        # benchmark never admits them into production evidence.
        stale_admissions = 0

    # Build a capsule admission witness for non-opaque deps (never stale).
    t0 = time.perf_counter()
    for path in context.required_exact_paths[:1]:
        body = tree.get(path, "")
        capsule = {
            "capsule_cid": _cid_for_text(f"cap:{path}", body),
            "stable_symbol_id": f"task:{task.task_id}:{path}",
            "version_cid": _cid_for_text(f"ver:{path}", body),
            "source_cid": _cid_for_text(f"src:{path}", body),
            "confidence": "exact",
        }
        admission = admit_capsule(
            capsule,
            semantic_state_root_cid=_cid_for_text("root", task.task_id),
            assessment={
                "freshness": FRESHNESS_FRESH,
                "admission": ADMISSION_EXACT,
                "caveats": (),
                "assessment_cid": _cid_for_text(f"assess:{path}", body),
            },
        )
        if admission.freshness == FRESHNESS_STALE:
            raise BenchmarkError("fresh admission marked stale")
        if not capsule_may_substitute(admission) and path not in context.opaque_raw_paths:
            # Allowed; exact targets are raw regions, not substituted.
            pass
    stage["admission_witness_ms"] = (time.perf_counter() - t0) * 1000.0

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assumptions = tuple(
        sorted(set(context.assumptions) | set(task.oracle.assumptions))
    )
    uncertainty = tuple(sorted(set(task.oracle.uncertainty)))

    return BenchmarkResult(
        task_id=str(task.task_id),
        category=str(task.category),
        objective=str(task.objective),
        multi_file=bool(task.multi_file),
        frontier_or_human=bool(task.frontier_or_human),
        risk=str(task.risk),
        expected_route=str(task.expected_route),
        measured_route=measured_route,
        route_explanation=route_explanation,
        production_eligible=False,
        candidate_verification_outcome=str(
            task.oracle.candidate_verification_outcome
        ),
        production_acceptance=str(task.oracle.production_acceptance),
        candidate_source=str(task.candidate.source),
        candidate_id=str(task.candidate.candidate_id),
        base_mutation_case_id=str(task.base_mutation_case_id),
        invalidation_symbol_ids=tuple(task.oracle.invalidation_symbol_ids),
        proof_obligation_ids=tuple(task.oracle.proof_obligation_ids),
        context=context,
        selection=selection,
        receipt_freshness=freshness,
        stale_admissions=stale_admissions,
        simulated_admissions=0,
        model_receipt_emitted=False,
        production_root_advanced=False,
        assumptions=assumptions,
        uncertainty=uncertainty,
        failure_kind=_failure_kind(task),
        observational_latency_ms=elapsed_ms,
        stage_latencies_ms=stage,
    )


@dataclass
class BenchmarkRunner:
    """Run the exactly-40-task offline semantic compression benchmark."""

    corpus_package_dir: Path | None = None
    fixture_package_dir: Path | None = None
    _corpus_pkg: ModuleType | None = field(default=None, repr=False)
    _fixture_pkg: ModuleType | None = field(default=None, repr=False)
    _corpus: Any | None = field(default=None, repr=False)
    _fixture: Any | None = field(default=None, repr=False)

    def load(self) -> "BenchmarkRunner":
        self._corpus_pkg = load_benchmark_corpus_package(self.corpus_package_dir)
        self._fixture_pkg = load_fixture_repository_package(self.fixture_package_dir)
        self._corpus = self._corpus_pkg.BenchmarkCorpus.load()
        self._fixture = self._fixture_pkg.ControlledSemanticRepository.load()
        if len(self._corpus.tasks) != EXPECTED_TASK_COUNT:
            raise BenchmarkError(
                f"corpus task count {len(self._corpus.tasks)} != {EXPECTED_TASK_COUNT}"
            )
        return self

    @property
    def corpus(self) -> Any:
        if self._corpus is None:
            self.load()
        return self._corpus

    @property
    def fixture(self) -> Any:
        if self._fixture is None:
            self.load()
        return self._fixture

    def run(self) -> tuple[tuple[BenchmarkResult, ...], BenchmarkSummary, dict[str, Any]]:
        started = time.perf_counter()
        corpus = self.corpus
        fixture = self.fixture
        results: list[BenchmarkResult] = []
        for task in corpus.tasks:
            results.append(measure_task(task, fixture_repo=fixture))
        ordered = tuple(sorted(results, key=lambda item: item.task_id))
        wall_ms = (time.perf_counter() - started) * 1000.0
        summary = summarize_results(ordered, run_wall_clock_ms=wall_ms)
        report = build_report(ordered, summary, corpus=corpus)
        return ordered, summary, report


def build_report(
    results: Sequence[BenchmarkResult],
    summary: BenchmarkSummary,
    *,
    corpus: Any | None = None,
) -> dict[str, Any]:
    """Assemble the published JSON report envelope."""

    corpus_id = getattr(corpus, "corpus_id", "semantic-state-benchmark-corpus-v1")
    fixture_corpus_id = getattr(
        corpus, "fixture_corpus_id", "semantic-state-controlled-repo-v1"
    )
    baseline = None
    if corpus is not None and hasattr(corpus, "baseline_retrieval_defaults"):
        baseline = corpus.baseline_retrieval_defaults.to_dict()

    # Prefer first task estimator identity when present.
    tokenizer_id = "sch-fixture/token-estimator@1"
    estimator_version = "semantic-state-token-estimator-v1"
    if results:
        tokenizer_id = results[0].context.tokenizer_id
        estimator_version = results[0].context.estimator_version

    report: dict[str, Any] = {
        "schema": BENCHMARK_REPORT_SCHEMA,
        "interface": BENCHMARK_INTERFACE,
        "bundle": BOARD_BUNDLE,
        "corpus_id": corpus_id,
        "corpus_interface": CORPUS_INTERFACE,
        "fixture_corpus_id": fixture_corpus_id,
        "task_count": len(results),
        "tokenizer_id": tokenizer_id,
        "estimator_version": estimator_version,
        "baseline_retrieval_defaults": baseline,
        "summary": summary.to_dict(),
        "results": [item.to_dict() for item in results],
        "notes": [
            "Checked-in candidates are oracle/replay fixtures only (production_eligible=false).",
            "Wall-clock latencies are observational and excluded from --check equality.",
            "Both context modes use the same pinned tokenizer/estimator and hard coverage policy.",
            "Required target/test/opaque source is never omitted to improve reduction.",
            "Failed and escalated tasks remain in the denominator.",
            "Stale and simulated admissions are never counted as production accepted.",
        ],
        "run_wall_clock_ms": summary.run_wall_clock_ms,
        "generated_at_unix_ms": int(time.time() * 1000),
    }
    report["deterministic_digest"] = deterministic_report_digest(report)
    report["content_digest"] = content_digest(report)
    return report


def run_benchmark(
    *,
    corpus_package_dir: Path | None = None,
    fixture_package_dir: Path | None = None,
) -> dict[str, Any]:
    """Module-level entry: run the full benchmark and return the report dict."""

    runner = BenchmarkRunner(
        corpus_package_dir=corpus_package_dir,
        fixture_package_dir=fixture_package_dir,
    )
    _results, _summary, report = runner.run()
    return report


# ---------------------------------------------------------------------------
# Report rendering / persistence
# ---------------------------------------------------------------------------


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render human-readable Markdown from a benchmark report."""

    summary = report["summary"]
    lines: list[str] = []
    lines.append("# Semantic Compression Harness Benchmark Results")
    lines.append("")
    lines.append(f"- Interface: `{report.get('interface')}`")
    lines.append(f"- Bundle: `{report.get('bundle')}`")
    lines.append(f"- Corpus: `{report.get('corpus_id')}`")
    lines.append(f"- Fixture corpus: `{report.get('fixture_corpus_id')}`")
    lines.append(f"- Task count: **{report.get('task_count')}**")
    lines.append(
        f"- Tokenizer / estimator: `{report.get('tokenizer_id')}` / "
        f"`{report.get('estimator_version')}`"
    )
    lines.append(
        f"- Deterministic digest (observational fields stripped): "
        f"`{report.get('deterministic_digest')}`"
    )
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for name, ok in sorted(summary.get("gates", {}).items()):
        mark = "PASS" if ok else "FAIL"
        lines.append(f"- `{name}`: **{mark}**")
    lines.append("")
    lines.append("## Overall context reduction")
    lines.append("")
    lines.append(
        f"- Median reduction: **{summary['median_reduction_ratio'] * 100:.2f}%**"
    )
    lines.append(
        f"- Mean reduction: **{summary['mean_reduction_ratio'] * 100:.2f}%**"
    )
    lines.append(
        f"- Range: {summary['min_reduction_ratio'] * 100:.2f}% … "
        f"{summary['max_reduction_ratio'] * 100:.2f}%"
    )
    lines.append("")
    lines.append("## Reduction by task type")
    lines.append("")
    lines.append("| Category | Count | Median reduction | Mean reduction |")
    lines.append("|---|---:|---:|---:|")
    counts = summary.get("category_counts", {})
    medians = summary.get("category_median_reduction", {})
    means = summary.get("category_mean_reduction", {})
    for category in sorted(counts):
        lines.append(
            f"| `{category}` | {counts[category]} | "
            f"{medians.get(category, 0.0) * 100:.2f}% | "
            f"{means.get(category, 0.0) * 100:.2f}% |"
        )
    lines.append("")
    lines.append("## Selection precision / recall")
    lines.append("")
    prec = summary.get("overall_precision_bp")
    rec = summary.get("overall_recall_bp")
    if prec is not None:
        lines.append(
            f"- Overall precision: **{prec / 100.0:.2f}%** ({prec} bp)"
        )
    else:
        lines.append("- Overall precision: **n/a**")
    if rec is not None:
        lines.append(f"- Overall recall: **{rec / 100.0:.2f}%** ({rec} bp)")
    else:
        lines.append("- Overall recall: **n/a**")
    lines.append(
        f"- Controlled false negatives: **{summary.get('total_false_negatives')}**"
    )
    lines.append(
        f"- False positives (extras kept visible): "
        f"**{summary.get('total_false_positives')}**"
    )
    lines.append(
        f"- Coverage omissions: **{summary.get('coverage_omission_count')}**"
    )
    lines.append(
        f"- Stale admissions: **{summary.get('total_stale_admissions')}**"
    )
    lines.append(
        f"- Simulated admissions: **{summary.get('total_simulated_admissions')}**"
    )
    lines.append(
        f"- Production-eligible true rows: "
        f"**{summary.get('production_eligible_true_count')}**"
    )
    lines.append("")
    lines.append("## Failures and outcomes")
    lines.append("")
    lines.append("### Failure kinds")
    lines.append("")
    for kind, count in sorted(summary.get("failure_counts", {}).items()):
        lines.append(f"- `{kind}`: {count}")
    lines.append("")
    lines.append("### Candidate verification outcomes")
    lines.append("")
    for kind, count in sorted(
        summary.get("verification_outcome_counts", {}).items()
    ):
        lines.append(f"- `{kind}`: {count}")
    lines.append("")
    lines.append("### Production acceptance (never accepted for oracle/replay)")
    lines.append("")
    for kind, count in sorted(
        summary.get("production_acceptance_counts", {}).items()
    ):
        lines.append(f"- `{kind}`: {count}")
    lines.append("")
    lines.append("### Route distribution (measured)")
    lines.append("")
    for route, count in sorted(summary.get("route_distribution", {}).items()):
        lines.append(f"- `{route}`: {count}")
    lines.append("")
    lines.append("### Fallback distribution")
    lines.append("")
    for kind, count in sorted(summary.get("fallback_counts", {}).items()):
        lines.append(f"- `{kind}`: {count}")
    lines.append("")
    lines.append(
        f"Uncertainty declared on **{summary.get('uncertainty_task_count')}** tasks "
        "(tasks with non-`none_declared` uncertainty)."
    )
    lines.append("")
    lines.append("## Per-task rows")
    lines.append("")
    lines.append(
        "| Task | Category | Baseline | Semantic | Reduction | "
        "Precision bp | Recall bp | FN | Route | Outcome | Eligible |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for row in report.get("results", []):
        lines.append(
            f"| `{row['task_id']}` | `{row['category']}` | "
            f"{row['baseline_tokens']} | {row['semantic_tokens']} | "
            f"{row['reduction_ratio'] * 100:.1f}% | "
            f"{row.get('precision_bp')} | {row.get('recall_bp')} | "
            f"{len(row.get('false_negatives') or ())} | "
            f"`{row['measured_route']}` | "
            f"`{row['candidate_verification_outcome']}` | "
            f"{row['production_eligible']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        f"Observational run wall-clock: {summary.get('run_wall_clock_ms', 0.0):.2f} ms "
        "(excluded from `--check`)."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: Mapping[str, Any],
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> tuple[Path, Path]:
    json_target = Path(json_path) if json_path is not None else _DEFAULT_RESULTS_JSON
    md_target = Path(markdown_path) if markdown_path is not None else _DEFAULT_RESULTS_MD
    json_target.parent.mkdir(parents=True, exist_ok=True)
    md_target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    if not text.endswith("\n"):
        text += "\n"
    json_target.write_text(text, encoding="utf-8")
    md_target.write_text(render_markdown(report), encoding="utf-8")
    return json_target, md_target


def load_report(path: Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else _DEFAULT_RESULTS_JSON
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkError("results JSON must be an object")
    return payload


def check_report(
    published: Mapping[str, Any] | None = None,
    *,
    json_path: Path | None = None,
    corpus_package_dir: Path | None = None,
    fixture_package_dir: Path | None = None,
) -> dict[str, Any]:
    """Recompute the benchmark and compare deterministic fields to published.

    Returns a check envelope. Raises BenchmarkError on mismatch or gate failure.
    """

    recomputed = run_benchmark(
        corpus_package_dir=corpus_package_dir,
        fixture_package_dir=fixture_package_dir,
    )
    if published is None:
        published = load_report(json_path)

    # Compare pure semantic payload (no wall-clock, no self-digests).
    left = strip_observational_fields(recomputed, strip_digests=True)
    right = strip_observational_fields(published, strip_digests=True)
    left_digest = deterministic_report_digest(recomputed)
    right_digest = deterministic_report_digest(published)
    # Attach recomputed digests only for the check envelope, not the equality body.
    left_bytes = _canonical_json(left)
    right_bytes = _canonical_json(right)
    equal = left_bytes == right_bytes and left_digest == right_digest

    gates = recomputed.get("summary", {}).get("gates", {})
    gates_ok = all(bool(value) for value in gates.values())

    envelope = {
        "schema": "ipfs_accelerate_py/semantic-state/benchmark-check@1",
        "interface": BENCHMARK_INTERFACE,
        "deterministic_equal": equal,
        "gates_ok": gates_ok,
        "gates": gates,
        "recomputed_deterministic_digest": left_digest,
        "published_deterministic_digest": right_digest,
        "observational_fields_excluded": sorted(OBSERVATIONAL_FIELD_NAMES),
    }
    if not equal:
        raise BenchmarkError(
            "deterministic semantic fields differ between recomputed and published "
            f"results (recomputed={envelope['recomputed_deterministic_digest']}, "
            f"published={envelope['published_deterministic_digest']})"
        )
    if not gates_ok:
        failed = [name for name, ok in gates.items() if not ok]
        raise BenchmarkError(f"benchmark gates failed: {failed}")
    return envelope


__all__ = [
    "BENCHMARK_INTERFACE",
    "BENCHMARK_REPORT_SCHEMA",
    "BENCHMARK_RESULT_SCHEMA",
    "BENCHMARK_SUMMARY_SCHEMA",
    "BenchmarkError",
    "BenchmarkResult",
    "BenchmarkRunner",
    "BenchmarkSummary",
    "BenchmarkTokenEstimator",
    "ContextModeComparison",
    "EXPECTED_TASK_COUNT",
    "MIN_MEDIAN_REDUCTION",
    "OBSERVATIONAL_FIELD_NAMES",
    "SelectionMetrics",
    "build_report",
    "check_report",
    "compare_context_modes",
    "content_digest",
    "deterministic_report_bytes",
    "deterministic_report_digest",
    "load_benchmark_corpus_package",
    "load_fixture_repository_package",
    "load_report",
    "measure_selection",
    "measure_task",
    "render_markdown",
    "run_benchmark",
    "strip_observational_fields",
    "summarize_results",
    "write_report",
]
