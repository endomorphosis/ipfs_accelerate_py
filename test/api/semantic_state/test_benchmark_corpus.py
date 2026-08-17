"""SCH-016: exactly-40-task benchmark corpus acceptance tests.

Validates SemanticStateBenchmarkCorpus@1:

* exactly 40 unique stable task IDs with required category counts;
* multi-file and frontier/human cases present;
* every task runnable offline against the pinned controlled fixture tree;
* candidate verification outcome separated from production acceptance;
* no expected outcome derived from benchmark implementation output;
* checked-in candidates are oracle/replay only (production_eligible=false).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = REPO_ROOT / "benchmarks" / "semantic_state" / "tasks"
CORPUS_JSON = REPO_ROOT / "benchmarks" / "semantic_state" / "corpus.json"
FIXTURE_DIR = (
    REPO_ROOT
    / "test"
    / "fixtures"
    / "semantic_state_harness"
    / "controlled_repo"
)

PACKAGE_NAME = "sch_benchmark_corpus_pkg"
FIXTURE_PACKAGE_NAME = "sch_controlled_repo_fixture_for_bench"

REQUIRED_CATEGORY_COUNTS = {
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

CANDIDATE_VERIFICATION_OUTCOMES = frozenset(
    {"pass", "fail", "reject", "escalate"}
)
PRODUCTION_ACCEPTANCE_OUTCOMES = frozenset(
    {"not_applicable", "rejected", "blocked"}
)

FORBIDDEN_ORACLE_AUTHORITIES = frozenset(
    {
        "benchmark_implementation_output",
        "harness_measurement",
        "model_output",
        "self_sealing",
    }
)


def _load_package(
    package_name: str, package_dir: Path, modules: list[tuple[str, str]]
) -> ModuleType:
    """Load a local package without relying on installed path layout."""

    if package_name in sys.modules:
        return sys.modules[package_name]

    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise ImportError(f"missing package init: {init_path}")

    package = ModuleType(package_name)
    package.__file__ = str(init_path)
    package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package

    def _load_submodule(name: str, filename: str) -> ModuleType:
        qualname = f"{package_name}.{name}"
        if qualname in sys.modules:
            return sys.modules[qualname]
        path = package_dir / filename
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = package_name
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        return module

    for name, filename in modules:
        _load_submodule(name, filename)

    init_spec = importlib.util.spec_from_file_location(
        package_name, init_path, submodule_search_locations=[str(package_dir)]
    )
    assert init_spec is not None and init_spec.loader is not None
    package.__spec__ = init_spec
    package.__package__ = package_name
    init_spec.loader.exec_module(package)
    return package


def _load_benchmark_package() -> ModuleType:
    return _load_package(
        PACKAGE_NAME,
        TASKS_DIR,
        [
            ("task_record", "task_record.py"),
            ("recipes", "recipes.py"),
            ("corpus", "corpus.py"),
        ],
    )


def _load_fixture_package() -> ModuleType:
    return _load_package(
        FIXTURE_PACKAGE_NAME,
        FIXTURE_DIR,
        [
            ("mutation_case", "mutation_case.py"),
            ("recipes", "recipes.py"),
            ("controlled_repository", "controlled_repository.py"),
        ],
    )


@pytest.fixture(scope="module")
def bench_pkg() -> ModuleType:
    return _load_benchmark_package()


@pytest.fixture(scope="module")
def fixture_pkg() -> ModuleType:
    return _load_fixture_package()


@pytest.fixture(scope="module")
def corpus(bench_pkg: ModuleType) -> Any:
    return bench_pkg.BenchmarkCorpus.load()


@pytest.fixture(scope="module")
def fixture_repo(fixture_pkg: ModuleType) -> Any:
    return fixture_pkg.ControlledSemanticRepository.load()


def test_package_surface(bench_pkg: ModuleType) -> None:
    assert bench_pkg.BENCHMARK_CORPUS_INTERFACE == "SemanticStateBenchmarkCorpus@1"
    assert bench_pkg.CORPUS_ID == "semantic-state-benchmark-corpus-v1"
    assert bench_pkg.EXPECTED_TASK_COUNT == 40
    assert callable(bench_pkg.BenchmarkCorpus.load)
    assert bench_pkg.BenchmarkTask is not None
    assert bench_pkg.TaskOracle is not None
    assert bench_pkg.CandidatePatch is not None
    assert bench_pkg.BaselineRetrievalPolicy is not None


def test_exactly_40_unique_stable_task_ids(corpus: Any) -> None:
    assert len(corpus.tasks) == 40
    ids = [task.task_id for task in corpus.tasks]
    assert len(set(ids)) == 40
    assert ids == sorted(ids)
    for task_id in ids:
        assert task_id.startswith("sch-bench-")
        assert task_id == task_id.strip()


def test_required_category_counts(corpus: Any) -> None:
    counts = corpus.category_counts()
    assert counts == REQUIRED_CATEGORY_COUNTS
    assert sum(counts.values()) == 40
    for category, expected in REQUIRED_CATEGORY_COUNTS.items():
        assert counts[category] == expected


def test_includes_multi_file_and_frontier_human_cases(corpus: Any) -> None:
    multi = [task for task in corpus.tasks if task.multi_file]
    frontier = [task for task in corpus.tasks if task.frontier_or_human]
    assert len(multi) >= 6
    assert any(task.category == "multi_file_refactor" for task in multi)
    assert len(frontier) >= 2
    routes = {task.expected_route for task in frontier}
    assert "frontier_model" in routes
    assert "human_review_required" in routes
    # Rejection/escalation cohort is fully present.
    rejection = [
        task
        for task in corpus.tasks
        if task.category == "rejection_or_escalation"
    ]
    assert len(rejection) == 6
    for task in rejection:
        assert task.oracle.candidate_verification_outcome in {
            "reject",
            "escalate",
            "fail",
        }


def test_every_task_runnable_offline_against_pinned_fixture(
    corpus: Any, fixture_repo: Any, fixture_pkg: ModuleType
) -> None:
    assert corpus.fixture_corpus_id == fixture_repo.corpus_id
    assert corpus.fixture_package_path == (
        "test/fixtures/semantic_state_harness/controlled_repo"
    )
    assert FIXTURE_DIR.is_dir()
    mutation_ids = {case.case_id for case in fixture_repo.mutations}

    for task in corpus.tasks:
        policy = task.baseline_retrieval
        assert policy.allow_network is False
        assert policy.require_exact_target_source is True
        assert policy.allow_omit_required_raw is False
        assert policy.allow_model_derived_expected_outcome is False
        assert policy.fixture_corpus_id == corpus.fixture_corpus_id
        assert policy.coverage_policy == "hard_coverage_no_omit_required"
        assert policy.tokenizer_id
        assert policy.estimator_version

        assert task.base_mutation_case_id in mutation_ids, task.task_id
        # Materialize mutated tree offline (in-memory) without network.
        mutated = fixture_repo.mutated_tree(task.base_mutation_case_id)
        assert mutated
        base = fixture_repo.base_tree()
        assert fixture_repo.mutated_tree_digest(
            task.base_mutation_case_id
        ) != fixture_repo.base_tree_digest()
        # Target paths exist in base and/or mutated trees (rename/delete safe).
        for path in task.target_paths:
            assert path in base or path in mutated, (task.task_id, path)

        # Candidate is pinned to the same mutation and never production-eligible.
        assert task.candidate.production_eligible is False
        assert task.candidate.base_mutation_case_id == task.base_mutation_case_id
        assert task.candidate.source in {
            "oracle_replay_fixture",
            "controlled_fixture_mutation",
        }

    # Controlled repo materialize helper remains available offline.
    assert callable(fixture_repo.materialize_base)
    assert callable(fixture_repo.materialize_mutation)


def test_candidate_verification_separated_from_production_acceptance(
    corpus: Any,
) -> None:
    for task in corpus.tasks:
        cand = task.oracle.candidate_verification_outcome
        prod = task.oracle.production_acceptance
        assert cand in CANDIDATE_VERIFICATION_OUTCOMES, task.task_id
        assert prod in PRODUCTION_ACCEPTANCE_OUTCOMES, task.task_id
        # Production acceptance must not collapse into candidate verification.
        assert cand != prod or cand in {"reject"}  # reject/reject allowed wording-wise
        # Explicit separation: production never "accepts" corpus fixtures.
        assert prod != "accepted"
        assert task.candidate.production_eligible is False
        payload = task.to_dict()
        assert payload["production_eligible"] is False
        assert "candidate_verification_outcome" in payload["oracle"]
        assert "production_acceptance" in payload["oracle"]
        assert (
            payload["oracle"]["candidate_verification_outcome"]
            != payload["oracle"]["production_acceptance"]
            or payload["oracle"]["candidate_verification_outcome"] == "reject"
        )


def test_no_expected_outcome_from_benchmark_implementation_output(
    corpus: Any,
) -> None:
    for task in corpus.tasks:
        authority = task.oracle.oracle_authority
        assert authority not in FORBIDDEN_ORACLE_AUTHORITIES
        assert authority in {
            "reviewed_fixture_authority",
            "controlled_fixture_oracle",
        }
        assert (
            task.baseline_retrieval.allow_model_derived_expected_outcome is False
        )
        # Oracle fields must be independently declared (non-null structured).
        assert isinstance(task.oracle.invalidation_symbol_ids, tuple)
        assert isinstance(task.oracle.selected_test_node_ids, tuple)
        assert isinstance(task.oracle.full_suite_test_node_ids, tuple)
        assert isinstance(task.oracle.proof_obligation_ids, tuple)
        assert isinstance(task.oracle.assumptions, tuple)
        assert isinstance(task.oracle.uncertainty, tuple)
        assert task.oracle.expected_false_negatives == 0
        assert set(task.oracle.selected_test_node_ids).issubset(
            set(task.oracle.full_suite_test_node_ids)
        )


def test_checked_in_corpus_json_matches_recipe_authority(
    corpus: Any, bench_pkg: ModuleType
) -> None:
    assert CORPUS_JSON.is_file()
    on_disk = json.loads(CORPUS_JSON.read_text(encoding="utf-8"))
    assert on_disk["interface"] == "SemanticStateBenchmarkCorpus@1"
    assert on_disk["corpus_id"] == "semantic-state-benchmark-corpus-v1"
    assert on_disk["task_count"] == 40
    assert on_disk["category_counts"] == REQUIRED_CATEGORY_COUNTS
    assert len(on_disk["task_ids"]) == 40
    assert len(on_disk["tasks"]) == 40
    assert on_disk["task_ids"] == sorted(on_disk["task_ids"])

    loaded = bench_pkg.BenchmarkCorpus.load_checked_in_json(CORPUS_JSON)
    assert loaded.manifest_digest() == corpus.manifest_digest()
    assert loaded.to_dict() == corpus.to_dict()

    # Round-trip every task record.
    for raw, task in zip(on_disk["tasks"], corpus.tasks, strict=True):
        rebuilt = bench_pkg.BenchmarkTask.from_dict(raw)
        assert rebuilt.to_dict() == task.to_dict()
        assert rebuilt.task_id == task.task_id
        assert rebuilt.candidate.production_eligible is False


def test_every_task_declares_route_risk_objective_and_targets(corpus: Any) -> None:
    for task in corpus.tasks:
        assert task.objective
        assert task.risk in {"low", "medium", "high", "critical"}
        assert task.expected_route in MODEL_ROUTES
        assert task.target_paths
        assert task.target_paths == tuple(sorted(task.target_paths))
        if task.multi_file:
            assert len(task.target_paths) >= 2
        if task.category == "multi_file_refactor":
            assert task.multi_file is True
        if task.frontier_or_human:
            assert task.expected_route in {
                "frontier_model",
                "human_review_required",
            }


def test_suite_is_fast_and_deterministic(
    corpus: Any, bench_pkg: ModuleType
) -> None:
    started = time.perf_counter()
    first = corpus.manifest_digest()
    second = bench_pkg.BenchmarkCorpus.load().manifest_digest()
    third = bench_pkg.BenchmarkCorpus.load_checked_in_json().manifest_digest()
    elapsed = time.perf_counter() - started
    assert first == second == third
    assert first.startswith("sha256:")
    assert elapsed < 2.0, f"corpus suite too slow: {elapsed:.3f}s"


def test_rejection_cohort_never_production_accepts(corpus: Any) -> None:
    for task in corpus.tasks:
        if task.category != "rejection_or_escalation":
            continue
        assert task.oracle.production_acceptance in {"rejected", "blocked"}
        assert task.oracle.candidate_verification_outcome in {
            "reject",
            "escalate",
            "fail",
        }
        assert task.candidate.production_eligible is False


def test_hard_cases_not_omitted_from_denominator(corpus: Any) -> None:
    """Failed/escalated tasks remain in the 40-task corpus (no cherry-picking)."""

    hard = [
        task
        for task in corpus.tasks
        if task.oracle.candidate_verification_outcome
        in {"reject", "escalate", "fail"}
        or task.frontier_or_human
        or task.category == "rejection_or_escalation"
    ]
    assert len(hard) >= 6
    # Opaque / race / stale / CAS / concurrent remain represented via mutations.
    mutations = {task.base_mutation_case_id for task in corpus.tasks}
    for required in {
        "stale_receipt",
        "out_of_scope_patch",
        "opaque_native",
        "post_scan_source_race",
        "failed_aba_cas",
        "concurrent_watchers_writers",
    }:
        assert required in mutations
