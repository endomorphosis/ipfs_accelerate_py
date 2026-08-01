"""Implied parent-tree validation for KITA-003 fixture corpus.

The authoritative corpus and suite live under
``ipfs_kit_py/tests/runtime_readiness/``. This module proves the nested package
is present, importable from the parent worktree, and satisfies the same
coverage and safety contracts without re-implementing the corpus.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# tests/runtime_readiness/foundations/this_file.py -> parents[3] is worktree root
ROOT = Path(__file__).resolve().parents[3]
NESTED_KIT = ROOT / "ipfs_kit_py"
NESTED_FIXTURES = (
    NESTED_KIT / "tests" / "runtime_readiness" / "fixtures"
)
NESTED_TEST = (
    NESTED_KIT
    / "tests"
    / "runtime_readiness"
    / "foundations"
    / "test_fixture_manifest.py"
)


def _ensure_nested_on_path() -> Path:
    assert NESTED_KIT.is_dir(), f"missing nested ipfs_kit_py at {NESTED_KIT}"
    kit_str = str(NESTED_KIT)
    if kit_str not in sys.path:
        sys.path.insert(0, kit_str)
    return NESTED_KIT


def test_nested_fixture_package_and_validation_test_exist() -> None:
    assert NESTED_FIXTURES.is_dir()
    assert (NESTED_FIXTURES / "recipes.py").is_file()
    assert (NESTED_FIXTURES / "schema.py").is_file()
    assert (NESTED_FIXTURES / "expand.py").is_file()
    assert (NESTED_FIXTURES / "safety.py").is_file()
    assert NESTED_TEST.is_file()
    body = NESTED_TEST.read_text(encoding="utf-8")
    assert "REQUIRED_COVERAGE_CATEGORIES" in body
    assert "CONFIRMED_BLOCKERS" in body
    assert "content_id" in body


def test_nested_fixture_corpus_covers_acceptance_criteria() -> None:
    _ensure_nested_on_path()
    fixtures_mod = importlib.import_module("tests.runtime_readiness.fixtures")
    manifest = fixtures_mod.build_manifest()
    fixtures_mod.validate_manifest(manifest)
    covered = set(manifest["coverage"]["covered_categories"])
    required = set(fixtures_mod.REQUIRED_COVERAGE_CATEGORIES)
    assert required <= covered
    assert set(fixtures_mod.CONFIRMED_BLOCKERS) == set(manifest["confirmed_blockers"])
    for ucan in fixtures_mod.REQUIRED_UCAN_VARIANTS:
        assert ucan in covered
    fixtures = fixtures_mod.expand_all_recipes()
    assert len(fixtures) >= 20
    for fixture in fixtures:
        fixtures_mod.validate_fixture(fixture)
        fixtures_mod.assert_fixture_safe(fixture)
        assert fixture["content_id"].startswith("sha256:")
        assert fixture["expected_trace"]["content_id"].startswith("sha256:")
        assert fixture["finite"] is True
        assert fixture["hermetic"] is True


def test_nested_schemas_match_declared_interfaces() -> None:
    _ensure_nested_on_path()
    fixtures_mod = importlib.import_module("tests.runtime_readiness.fixtures")
    assert fixtures_mod.RUNTIME_READINESS_FIXTURE_SCHEMA.endswith(
        "runtime-readiness/fixture@1"
    )
    assert fixtures_mod.FAULT_SCHEDULE_SCHEMA.endswith(
        "runtime-readiness/fault-schedule@1"
    )
    assert fixtures_mod.EXPECTED_STATE_TRACE_SCHEMA.endswith(
        "runtime-readiness/expected-state-trace@1"
    )
    manifest = fixtures_mod.load_manifest()
    assert "RuntimeReadinessFixture@1" in manifest["interface"]
    assert "FaultSchedule@1" in manifest["interface"]
    assert "ExpectedStateTrace@1" in manifest["interface"]
