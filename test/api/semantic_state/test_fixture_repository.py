"""SCH-014: controlled Python fixture repository acceptance tests.

Validates ControlledSemanticRepository@1:

* full mutation matrix is present, fast, and deterministic;
* every mutation declares independent changed-symbol, Merkle,
  invalidation/test/proof, receipt-freshness, and confidence/raw-source oracles;
* source-race bytes never enter a pack;
* unrelated formatting and bounded changes stay within budget;
* fixture scans read bytes only and never import or execute target modules.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "semantic_state_harness"
    / "controlled_repo"
)
PACKAGE_NAME = "sch_controlled_repo_fixture"

REQUIRED_CATEGORIES = frozenset(
    {
        "local_function_body",
        "public_signature",
        "cross_module_call",
        "dataclass_schema",
        "exception_behavior",
        "side_effect_security",
        "fixture_dependency",
        "pytest_configuration",
        "dependency_lockfile",
        "policy",
        "mcp_interface_client_adapter",
        "dynamic_import",
        "monkey_patch",
        "opaque_native",
        "unrelated_formatting",
        "deleted_symbol",
        "renamed_symbol",
        "generated_file",
        "stale_receipt",
        "failed_aba_cas",
        "interrupted_state_transition",
        "concurrent_watchers_writers",
        "post_scan_source_race",
        "out_of_scope_patch",
    }
)

ORACLE_FACETS = (
    "changed_symbol",
    "merkle",
    "invalidation",
    "receipt_freshness",
    "confidence",
)

TARGET_PACKAGE_PREFIX = "sch_fixture"


def _load_fixture_package() -> ModuleType:
    """Load controlled_repo as a standalone package without target imports."""

    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]

    init_path = FIXTURE_DIR / "__init__.py"
    if not init_path.is_file():
        raise ImportError(f"missing fixture package init: {init_path}")

    # Register package shell so relative imports resolve.
    package = ModuleType(PACKAGE_NAME)
    package.__file__ = str(init_path)
    package.__path__ = [str(FIXTURE_DIR)]  # type: ignore[attr-defined]
    sys.modules[PACKAGE_NAME] = package

    def _load_submodule(name: str, filename: str) -> ModuleType:
        qualname = f"{PACKAGE_NAME}.{name}"
        if qualname in sys.modules:
            return sys.modules[qualname]
        path = FIXTURE_DIR / filename
        # Leaf modules: omit submodule_search_locations so __spec__.parent is
        # the package name and relative imports stay warning-free.
        spec = importlib.util.spec_from_file_location(qualname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = PACKAGE_NAME
        sys.modules[qualname] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        return module

    # Dependency order: mutation_case -> recipes -> controlled_repository -> package
    _load_submodule("mutation_case", "mutation_case.py")
    _load_submodule("recipes", "recipes.py")
    controlled = _load_submodule("controlled_repository", "controlled_repository.py")
    # Execute package __init__ exports.
    init_spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME, init_path, submodule_search_locations=[str(FIXTURE_DIR)]
    )
    assert init_spec is not None and init_spec.loader is not None
    package.__spec__ = init_spec
    package.__package__ = PACKAGE_NAME
    init_spec.loader.exec_module(package)
    # Ensure primary API is present.
    assert hasattr(package, "ControlledSemanticRepository")
    assert hasattr(controlled, "ControlledSemanticRepository")
    return package


@pytest.fixture(scope="module")
def fixture_pkg() -> ModuleType:
    return _load_fixture_package()


@pytest.fixture(scope="module")
def repo(fixture_pkg: ModuleType) -> Any:
    return fixture_pkg.ControlledSemanticRepository.load()


def test_fixture_package_surface(fixture_pkg: ModuleType) -> None:
    assert fixture_pkg.CONTROLLED_REPO_INTERFACE == "ControlledSemanticRepository@1"
    assert fixture_pkg.CORPUS_ID == "semantic-state-controlled-repo-v1"
    assert callable(fixture_pkg.ControlledSemanticRepository.load)
    assert fixture_pkg.MutationCase is not None
    assert fixture_pkg.FixtureOracle is not None


def test_repository_loads_and_validates(repo: Any) -> None:
    assert repo.interface == "ControlledSemanticRepository@1"
    assert repo.corpus_id == "semantic-state-controlled-repo-v1"
    assert len(repo.base_files) >= 15
    assert len(repo.mutations) == len(REQUIRED_CATEGORIES)
    # validate() is idempotent.
    repo.validate()


def test_required_mutation_catalogue_is_complete(repo: Any) -> None:
    categories = {case.category for case in repo.mutations}
    assert categories == REQUIRED_CATEGORIES
    case_ids = [case.case_id for case in repo.mutations]
    assert case_ids == sorted(case_ids)
    assert len(set(case_ids)) == len(case_ids)


def test_every_mutation_has_independent_oracle_facets(repo: Any) -> None:
    for case in repo.mutations:
        oracle = case.oracle
        payload = oracle.to_dict()
        for facet in ORACLE_FACETS:
            assert facet in payload, (case.case_id, facet)
            assert payload[facet], (case.case_id, facet)

        # changed-symbol
        assert case.oracle.changed_symbol.symbol_ids
        assert (
            case.oracle.changed_symbol.primary_symbol_id
            in case.oracle.changed_symbol.symbol_ids
        )
        assert case.oracle.changed_symbol.change_kinds

        # Merkle
        assert (
            case.oracle.merkle.changed_node_ids
            or case.oracle.merkle.affected_path_ids
        )
        assert isinstance(case.oracle.merkle.root_changes, bool)

        # invalidation / test / proof
        inv = case.oracle.invalidation
        assert inv.fallback in {"none", "full_pytest", "full_proofs", "both"}
        assert inv.expected_false_negatives == 0
        assert set(inv.selected_test_node_ids).issubset(
            set(inv.full_suite_test_node_ids)
        )
        # proof list is always declared (may be empty for non-proof cases)
        assert isinstance(inv.proof_obligation_ids, tuple)

        # receipt freshness
        fresh = case.oracle.receipt_freshness
        assert fresh.disposition
        assert fresh.accepts_stale_receipt is False or fresh.disposition == "current"
        # Stale admission is never allowed for fail-closed harness scenarios.
        if case.harness_scenario in {
            "stale_receipt",
            "failed_aba_cas",
            "post_scan_source_race",
            "out_of_scope_patch",
        }:
            assert fresh.accepts_stale_receipt is False

        # confidence / raw-source
        conf = case.oracle.confidence
        assert conf.confidence in {"exact", "conservative", "heuristic", "opaque"}
        if conf.raw_source_required:
            assert conf.raw_source_symbol_ids
        else:
            assert conf.raw_source_symbol_ids == ()

        # Oracle/replay fixtures are never production-eligible model output.
        assert case.production_eligible is False

        # Round-trip independence: oracles reconstruct from dict alone.
        rebuilt = type(oracle).from_dict(payload)
        assert rebuilt.to_dict() == payload


def test_suite_is_fast_and_deterministic(repo: Any, fixture_pkg: ModuleType) -> None:
    started = time.perf_counter()
    first_manifest = repo.to_manifest()
    first_digest = repo.manifest_digest()
    digests = {
        case.case_id: repo.mutated_tree_digest(case.case_id)
        for case in repo.mutations
    }
    # Reload and recompute.
    again = fixture_pkg.ControlledSemanticRepository.load()
    second_manifest = again.to_manifest()
    second_digest = again.manifest_digest()
    second_digests = {
        case.case_id: again.mutated_tree_digest(case.case_id)
        for case in again.mutations
    }
    elapsed = time.perf_counter() - started

    assert first_digest == second_digest
    assert first_manifest == second_manifest
    assert digests == second_digests
    assert first_digest.startswith("sha256:")
    # Full matrix load + dual digest pass should stay well under a second on CI.
    assert elapsed < 2.0, f"fixture suite too slow: {elapsed:.3f}s"


def test_source_race_bytes_never_enter_a_pack(repo: Any, fixture_pkg: ModuleType) -> None:
    case = repo.get_mutation("post_scan_source_race")
    assert case.source_race_bytes_forbidden is True
    assert case.category == "post_scan_source_race"

    recipes = sys.modules[f"{PACKAGE_NAME}.recipes"]
    marker = recipes.SOURCE_RACE_MARKER
    race_path = recipes.SOURCE_RACE_PATH

    mutated = repo.mutated_tree(case.case_id)
    assert race_path in mutated
    assert marker in mutated[race_path].encode("utf-8")

    pack_paths = repo.declared_pack_paths(case.case_id)
    assert race_path not in pack_paths
    assert race_path in case.pack_excluded_paths

    pack_bytes = {
        path: mutated[path].encode("utf-8")
        for path in pack_paths
        if path in mutated
    }
    controlled = sys.modules[f"{PACKAGE_NAME}.controlled_repository"]
    assert controlled.pack_contains_source_race_bytes(pack_bytes) is False

    # Explicitly including race paths would expose the marker — proving exclusion.
    leak_paths = controlled.pack_candidate_paths(
        case, include_source_race_paths=True
    )
    leak_bytes = {
        path: mutated[path].encode("utf-8")
        for path in leak_paths
        if path in mutated
    }
    assert controlled.pack_contains_source_race_bytes(leak_bytes) is True


def test_unrelated_formatting_and_changes_remain_bounded(repo: Any) -> None:
    controlled = sys.modules[f"{PACKAGE_NAME}.controlled_repository"]
    formatting = repo.get_mutation("unrelated_formatting")
    assert formatting.change_is_bounded is True
    assert formatting.oracle.invalidation.invalidation_symbol_ids == ()
    assert formatting.oracle.invalidation.selected_test_node_ids == ()

    stats = controlled.bounded_change_stats(repo.base_files, formatting)
    assert stats["operation_count"] <= controlled.BOUNDED_CHANGE_MAX_OPS
    assert stats["changed_bytes"] <= controlled.BOUNDED_CHANGE_MAX_BYTES
    assert stats["changed_path_count"] == 1

    for case in repo.mutations:
        if not case.change_is_bounded:
            continue
        case_stats = controlled.bounded_change_stats(repo.base_files, case)
        assert case_stats["operation_count"] <= controlled.BOUNDED_CHANGE_MAX_OPS
        assert case_stats["changed_bytes"] <= controlled.BOUNDED_CHANGE_MAX_BYTES


def test_fixture_scan_does_not_import_or_execute_target_code(
    repo: Any, tmp_path: Path
) -> None:
    """Scan materializes trees and parses AST from bytes only."""

    before_modules = set(sys.modules)
    base_root = tmp_path / "base"
    mut_root = tmp_path / "mutated"
    repo.materialize_base(base_root, git=False)
    repo.materialize_mutation("local_function_body", mut_root, git=False)

    controlled = sys.modules[f"{PACKAGE_NAME}.controlled_repository"]
    base_bytes = controlled.read_tree_bytes(base_root)
    mut_bytes = controlled.read_tree_bytes(mut_root)

    # Scan: decode + ast.parse only. No import of sch_fixture.
    parsed = 0
    for path, payload in base_bytes.items():
        if not path.endswith(".py"):
            continue
        source = payload.decode("utf-8")
        tree = ast.parse(source, filename=path)
        assert isinstance(tree, ast.Module)
        parsed += 1
    assert parsed >= 5

    for path, payload in mut_bytes.items():
        if path.endswith(".py"):
            ast.parse(payload.decode("utf-8"), filename=path)

    after_modules = set(sys.modules)
    leaked = sorted(
        name
        for name in after_modules - before_modules
        if name == TARGET_PACKAGE_PREFIX or name.startswith(TARGET_PACKAGE_PREFIX + ".")
    )
    assert leaked == [], f"scan imported target modules: {leaked}"

    # Ensure no target package ended up on sys.path via materialization alone.
    for entry in list(sys.path):
        if entry and Path(entry).resolve() in {
            base_root.resolve(),
            (base_root / "src").resolve(),
            mut_root.resolve(),
            (mut_root / "src").resolve(),
        }:
            pytest.fail(f"materialized tree on sys.path: {entry}")


def test_materialize_git_trees_are_deterministic(repo: Any, tmp_path: Path) -> None:
    git = pytest.importorskip("subprocess").run
    probe = git(["git", "--version"], capture_output=True, check=False)
    if probe.returncode != 0:
        pytest.skip("git not available")

    first = repo.materialize_base(tmp_path / "git-a", git=True)
    second = repo.materialize_base(tmp_path / "git-b", git=True)
    assert first["tree"] == second["tree"]
    assert first["commit"] == second["commit"]
    assert len(first["tree"]) == 40

    m1 = repo.materialize_mutation(
        "cross_module_call", tmp_path / "git-m1", git=True
    )
    m2 = repo.materialize_mutation(
        "cross_module_call", tmp_path / "git-m2", git=True
    )
    assert m1["tree"] == m2["tree"]
    assert m1["tree"] != first["tree"]
    assert m1["tree_digest"] == m2["tree_digest"]
    assert m1["tree_digest"] != repo.base_tree_digest()


def test_mutation_round_trip_dicts_are_stable(repo: Any) -> None:
    for case in repo.mutations:
        payload = case.to_dict()
        rebuilt = type(case).from_dict(payload)
        assert rebuilt.to_dict() == payload
        assert rebuilt.case_id == case.case_id
        assert rebuilt.oracle.to_dict() == case.oracle.to_dict()


def test_base_tree_contains_expected_python_layout(repo: Any) -> None:
    paths = set(repo.base_files)
    expected = {
        "pyproject.toml",
        "pytest.ini",
        "requirements.lock",
        "policy/admission.json",
        "interfaces/mcp_client.json",
        "src/sch_fixture/core.py",
        "src/sch_fixture/api.py",
        "src/sch_fixture/schema.py",
        "src/sch_fixture/security.py",
        "src/sch_fixture/adapters.py",
        "src/sch_fixture/dynamic_loader.py",
        "src/sch_fixture/native_bridge.py",
        "src/sch_fixture/generated/bindings.py",
        "tests/conftest.py",
        "tests/test_core.py",
    }
    assert expected.issubset(paths)
    # No absolute or escaping paths.
    for path in paths:
        assert not path.startswith("/")
        assert ".." not in path.split("/")


def test_opaque_and_dynamic_cases_require_raw_source(repo: Any) -> None:
    for case_id in ("opaque_native", "dynamic_import", "monkey_patch"):
        case = repo.get_mutation(case_id)
        assert case.oracle.confidence.raw_source_required is True
        assert case.oracle.confidence.raw_source_symbol_ids
        assert case.oracle.confidence.confidence in {
            "heuristic",
            "opaque",
            "conservative",
        }


def test_harness_scenarios_are_labeled(repo: Any) -> None:
    expected = {
        "stale_receipt": "stale_receipt",
        "failed_aba_cas": "failed_aba_cas",
        "interrupted_state_transition": "interrupted_state_transition",
        "concurrent_watchers_writers": "concurrent_watchers_writers",
        "post_scan_source_race": "post_scan_source_race",
        "out_of_scope_patch": "out_of_scope_patch",
    }
    for case_id, scenario in expected.items():
        case = repo.get_mutation(case_id)
        assert case.harness_scenario == scenario


def test_manifest_is_compact_and_content_addressed(repo: Any) -> None:
    manifest = repo.to_manifest()
    encoded = repr(manifest)
    # Compact: must not embed full source bodies of the base tree.
    assert "def add(left: int, right: int)" not in encoded
    assert manifest["base_tree_digest"] == repo.base_tree_digest()
    assert len(manifest["mutations"]) == len(repo.mutations)
    # Digest stable under key order / reload is covered elsewhere; check shape.
    digest = repo.manifest_digest()
    assert digest == "sha256:" + hashlib.sha256(
        __import__("json")
        .dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        .encode()
    ).hexdigest()


def test_no_network_or_native_extension_artifacts(repo: Any) -> None:
    for path, body in repo.base_files.items():
        lower = path.lower()
        assert not lower.endswith((".so", ".dll", ".dylib", ".pyd"))
        assert "http://" not in body
        assert "https://" not in body
        # ctypes/cffi markers would imply native execution surfaces.
        assert "ctypes" not in body
        assert "cffi" not in body


def test_fixture_dir_is_self_contained() -> None:
    assert FIXTURE_DIR.is_dir()
    required = {
        "__init__.py",
        "controlled_repository.py",
        "mutation_case.py",
        "recipes.py",
        "README.md",
    }
    present = {path.name for path in FIXTURE_DIR.iterdir() if path.is_file()}
    assert required.issubset(present)
    # Target tree is not checked in as an importable package under fixture dir.
    assert not (FIXTURE_DIR / "src").exists()
    assert not (FIXTURE_DIR / "sch_fixture").exists()
