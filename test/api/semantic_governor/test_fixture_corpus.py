"""SCG-040: deterministic partitioned fixture corpus acceptance tests.

Validates SemanticGovernorFixtureCorpus@1 / scg/partitioned-corpus@1:

* calibration, development, and held-out partitions are non-empty, deterministic,
  and pairwise disjoint;
* required task families appear in every partition;
* the 18 adversarial scenarios are held-out only and independently declared;
* expected omissions/outcomes are scanner-derived identities, not SUT output;
* materialised trees are byte-deterministic and free of forbidden artifacts;
* optional live scanner checks confirm declared Python symbols exist in scans.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "semantic_governor"
PACKAGE_NAME = "scg_partitioned_fixture_corpus"

PARTITIONS = ("calibration", "development", "held_out")
TASK_FAMILIES = (
    "local_bug",
    "exception",
    "api_migration",
    "schema_migration",
    "state",
    "configuration",
    "fixture",
    "dynamic_import",
    "monkey_patch",
    "generated",
    "plugin",
    "refactor",
    "documentation",
    "proof",
)
ADVERSARIAL_SCENARIOS = (
    "hidden_callee_side_effect",
    "caller_exception_contract",
    "config_flag",
    "pytest_fixture",
    "serializer",
    "generated_interface",
    "stale_capsule",
    "confidence_misclassification",
    "opaque_dynamic_import",
    "behavior_only_dependency",
    "security_invariant",
    "migration_path",
    "misleading_comment",
    "prompt_injection",
    "selected_pass_full_fail",
    "test_pass_formal_fail",
    "raw_correct_compressed_wrong",
    "both_context_model_failure",
)

INTERFACE = "SemanticGovernorFixtureCorpus@1"
SCHEMA = "scg/partitioned-corpus@1"
EVIDENCE_ID = "scg/partitioned-corpus@1"
CORPUS_ID = "semantic-governor-partitioned-corpus-v1"
TASK_ID = "SCG-040"

FORBIDDEN_MARKERS = (
    "model_output",
    "completion_receipt",
    "state.db",
    "duckdb",
    "provider_response",
)


def _load_fixture_package() -> ModuleType:
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]

    init_path = FIXTURE_DIR / "__init__.py"
    if not init_path.is_file():
        raise ImportError(f"missing fixture package init: {init_path}")

    package = ModuleType(PACKAGE_NAME)
    package.__file__ = str(init_path)
    package.__path__ = [str(FIXTURE_DIR)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE_NAME] = package

    def _load_submodule(name: str, filename: str) -> ModuleType:
        qualname = f"{PACKAGE_NAME}.{name}"
        if qualname in sys.modules:
            return sys.modules[qualname]
        path = FIXTURE_DIR / filename
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = PACKAGE_NAME
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        return module

    _load_submodule("case_record", "case_record.py")
    _load_submodule("recipes", "recipes.py")
    _load_submodule("corpus", "corpus.py")

    init_spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, init_path, submodule_search_locations=[str(FIXTURE_DIR)]
    )
    assert init_spec is not None and init_spec.loader is not None
    package.__spec__ = init_spec
    package.__package__ = PACKAGE_NAME
    init_spec.loader.exec_module(package)
    assert hasattr(package, "SemanticGovernorFixtureCorpus")
    return package


@pytest.fixture(scope="module")
def fixture_pkg() -> ModuleType:
    return _load_fixture_package()


@pytest.fixture(scope="module")
def corpus(fixture_pkg: ModuleType) -> Any:
    return fixture_pkg.SemanticGovernorFixtureCorpus.load()


def test_fixture_package_surface(fixture_pkg: ModuleType) -> None:
    assert fixture_pkg.FIXTURE_CORPUS_INTERFACE == INTERFACE
    assert fixture_pkg.FIXTURE_CORPUS_SCHEMA == SCHEMA
    assert fixture_pkg.CORPUS_ID == CORPUS_ID
    assert fixture_pkg.EVIDENCE_ID == EVIDENCE_ID
    assert fixture_pkg.TASK_ID == TASK_ID
    assert callable(fixture_pkg.SemanticGovernorFixtureCorpus.load)
    assert tuple(fixture_pkg.PARTITIONS) == PARTITIONS
    assert tuple(fixture_pkg.TASK_FAMILIES) == TASK_FAMILIES
    assert tuple(fixture_pkg.ADVERSARIAL_SCENARIOS) == ADVERSARIAL_SCENARIOS


def test_corpus_loads_and_validates(corpus: Any) -> None:
    assert corpus.interface == INTERFACE
    assert corpus.schema == SCHEMA
    assert corpus.corpus_id == CORPUS_ID
    assert corpus.evidence_id == EVIDENCE_ID
    assert corpus.task_id == TASK_ID
    assert len(corpus.base_files) >= 20
    assert len(corpus.cases) >= len(TASK_FAMILIES) * len(PARTITIONS)
    corpus.validate()


def test_partitions_are_nonempty_disjoint_and_cover_all_cases(corpus: Any) -> None:
    membership = corpus.partition_membership()
    assert set(membership) == set(PARTITIONS)
    seen: set[str] = set()
    for name in PARTITIONS:
        members = membership[name]
        assert members, f"partition {name} empty"
        assert members == sorted(members)
        overlap = seen & set(members)
        assert not overlap, f"partition overlap involving {name}: {sorted(overlap)}"
        seen.update(members)
    assert seen == {case.case_id for case in corpus.cases}


def test_every_partition_covers_every_task_family(corpus: Any) -> None:
    for partition in PARTITIONS:
        families = {
            case.family for case in corpus.cases if case.partition == partition
        }
        missing = set(TASK_FAMILIES) - families
        assert not missing, f"{partition} missing families: {sorted(missing)}"


def test_adversarial_scenarios_held_out_only_and_complete(corpus: Any) -> None:
    found = {
        case.adversarial_scenario: case
        for case in corpus.cases
        if case.adversarial_scenario is not None
    }
    assert set(found) == set(ADVERSARIAL_SCENARIOS)
    for scenario, case in found.items():
        assert case.partition == "held_out", scenario
        assert case.omission is not None
        assert case.outcome is not None
        assert case.scanner_view is not None


def test_case_ids_sorted_unique_and_stable(corpus: Any) -> None:
    ids = [case.case_id for case in corpus.cases]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_every_case_has_independent_scanner_omission_and_outcome(corpus: Any) -> None:
    for case in corpus.cases:
        view = case.scanner_view.to_dict()
        omission = case.omission.to_dict()
        outcome = case.outcome.to_dict()
        assert view["schema"] == "scg/scanner-view@1"
        assert omission["schema"] == "scg/omission-oracle@1"
        assert outcome["schema"] == "scg/outcome-oracle@1"
        assert view["changed_paths"]
        assert view["changed_symbols"]
        assert view["primary_symbol"] in view["changed_symbols"]
        assert view["confidence"] in {
            "exact",
            "conservative",
            "heuristic",
            "opaque",
        }
        # Omission/outcome identities are independently declared (payload present)
        # and scanner-derived (subset of scanner universe).
        universe = (
            set(view["changed_symbols"])
            | set(view["dependency_symbols"])
            | set(view.get("context_symbols") or ())
            | set(view["opaque_symbols"])
        )
        for key in (
            "critical_omitted_symbols",
            "noncritical_omitted_symbols",
            "compressed_includes",
            "compressed_omits",
            "expansion_targets",
        ):
            assert set(omission[key]).issubset(universe), (case.case_id, key)
        assert outcome["expected_outcome"]
        assert outcome["expected_diagnosis"]
        assert set(outcome["selected_tests"]).issubset(
            set(outcome["full_suite_tests"])
        )
        assert case.production_eligible is False


def test_intentional_critical_omissions_never_auto_accept(corpus: Any) -> None:
    for case in corpus.cases:
        if case.omission.intentional_critical:
            assert case.outcome.automatic_accept_allowed is False
            assert case.omission.critical_omitted_symbols
            assert case.outcome.expected_outcome != "sufficient"


def test_scanner_changed_paths_match_tree_delta(corpus: Any, fixture_pkg: ModuleType) -> None:
    for case in corpus.cases:
        mutated = fixture_pkg.apply_operations(corpus.base_files, case.operations)
        actual = set(fixture_pkg.changed_paths(corpus.base_files, mutated))
        declared = set(case.scanner_view.changed_paths)
        assert declared.issubset(actual), (case.case_id, sorted(declared - actual))
        assert fixture_pkg.tree_digest(mutated) != corpus.base_tree_digest()


def test_manifest_is_deterministic_and_self_digesting(corpus: Any) -> None:
    first = corpus.to_manifest()
    second = corpus.to_manifest()
    assert first == second
    assert first["schema"] == SCHEMA
    assert first["interface"] == INTERFACE
    assert first["evidence_id"] == EVIDENCE_ID
    assert first["corpus_id"] == CORPUS_ID
    assert first["case_count"] == len(corpus.cases)
    assert set(first["partitions"]) == set(PARTITIONS)
    assert set(first["task_families"]) == set(TASK_FAMILIES)
    assert set(first["adversarial_scenarios"]) == set(ADVERSARIAL_SCENARIOS)
    # Recompute digest without the digest field.
    payload = dict(first)
    digest = payload.pop("corpus_digest")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
    assert digest == expected
    assert corpus.manifest_digest() == digest


def test_checked_in_manifest_matches_recipes_when_present(corpus: Any) -> None:
    path = FIXTURE_DIR / "manifest.json"
    if not path.is_file():
        pytest.skip("manifest.json not checked in yet")
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    live = corpus.to_manifest()
    assert on_disk["corpus_digest"] == live["corpus_digest"]
    assert on_disk["case_count"] == live["case_count"]
    assert on_disk["partition_membership"] == live["partition_membership"]
    assert on_disk["base_tree_digest"] == live["base_tree_digest"]


def test_materialize_base_and_case_are_byte_deterministic(
    corpus: Any, tmp_path: Path, fixture_pkg: ModuleType
) -> None:
    base_a = tmp_path / "base_a"
    base_b = tmp_path / "base_b"
    corpus.materialize_base(base_a)
    corpus.materialize_base(base_b)
    assert fixture_pkg.read_tree_bytes(base_a) == fixture_pkg.read_tree_bytes(base_b)

    case = corpus.cases[0]
    case_a = tmp_path / "case_a"
    case_b = tmp_path / "case_b"
    corpus.materialize_case(case.case_id, case_a)
    corpus.materialize_case(case.case_id, case_b)
    bytes_a = fixture_pkg.read_tree_bytes(case_a)
    bytes_b = fixture_pkg.read_tree_bytes(case_b)
    assert bytes_a == bytes_b
    # No forbidden markers in materialised trees.
    for rel, payload in bytes_a.items():
        blob = f"{rel}\n{payload.decode('utf-8', errors='replace')}".lower()
        for marker in FORBIDDEN_MARKERS:
            assert marker not in blob, (case.case_id, rel, marker)


def test_no_forbidden_artifacts_in_any_case(corpus: Any, fixture_pkg: ModuleType) -> None:
    for case in corpus.cases:
        mutated = fixture_pkg.apply_operations(corpus.base_files, case.operations)
        for path, body in mutated.items():
            blob = f"{path}\n{body}".lower()
            for marker in FORBIDDEN_MARKERS:
                assert marker not in blob, (case.case_id, path, marker)


def test_held_out_never_shares_case_ids_with_calibration_or_development(
    corpus: Any,
) -> None:
    membership = corpus.partition_membership()
    held = set(membership["held_out"])
    cal = set(membership["calibration"])
    dev = set(membership["development"])
    assert not (held & cal)
    assert not (held & dev)
    assert not (cal & dev)


def test_live_scanner_finds_declared_python_symbols(
    corpus: Any, tmp_path: Path
) -> None:
    """Scanner-derived check: declared Python symbols appear in a real scan."""

    try:
        from ipfs_datasets_py.logic.software_contracts.semantic_index.scanner import (
            scan_repository_state,
        )
    except Exception as exc:  # pragma: no cover - environment without datasets
        pytest.skip(f"scanner unavailable: {exc}")

    # Materialise base once; scan symbol universe.
    base_root = tmp_path / "scan_base"
    corpus.materialize_base(base_root)
    base_state = scan_repository_state(
        base_root, repository_id="repo:scg-fixture-base"
    )
    base_symbols = {symbol.qualified_name for symbol in base_state.symbols}
    assert base_symbols, "scanner returned no symbols for base tree"

    # Sample scannable family cases: local_bug / exception / refactor.
    sample_ids = [
        "local_bug.cal",
        "exception.dev",
        "refactor.hold",
        "api_migration.cal",
        "state.dev",
    ]
    for case_id in sample_ids:
        case = corpus.get_case(case_id)
        root = tmp_path / f"scan_{case_id.replace('.', '_')}"
        corpus.materialize_case(case_id, root)
        state = scan_repository_state(
            root, repository_id=f"repo:scg-fixture:{case_id}"
        )
        scanned = {symbol.qualified_name for symbol in state.symbols}
        # Primary changed symbols that look like Python qualified names must
        # appear in the scan of the materialised tree (or the base for
        # rename/delete edge cases — here all samples are body edits).
        for symbol in case.scanner_view.changed_symbols:
            if symbol.startswith("scg_fixture.") or symbol.startswith("tests."):
                assert symbol in scanned or symbol in base_symbols, (
                    case_id,
                    symbol,
                    sorted(scanned)[:20],
                )


def test_partition_membership_is_deterministic_across_loads(
    fixture_pkg: ModuleType,
) -> None:
    first = fixture_pkg.SemanticGovernorFixtureCorpus.load()
    second = fixture_pkg.SemanticGovernorFixtureCorpus.load()
    assert first.partition_membership() == second.partition_membership()
    assert first.base_tree_digest() == second.base_tree_digest()
    assert first.manifest_digest() == second.manifest_digest()
    assert [c.case_id for c in first.cases] == [c.case_id for c in second.cases]


def test_omission_and_outcome_oracles_are_not_empty_shells(corpus: Any) -> None:
    """Every case independently declares both inclusion and outcome fields."""

    for case in corpus.cases:
        # At least one of includes / critical / noncritical is declared.
        omission = case.omission
        assert (
            omission.compressed_includes
            or omission.critical_omitted_symbols
            or omission.noncritical_omitted_symbols
        ), case.case_id
        outcome = case.outcome
        assert outcome.reason_codes, case.case_id
        assert outcome.full_suite_tests, case.case_id
