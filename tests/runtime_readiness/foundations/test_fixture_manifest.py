"""Implied parent-tree validation for KITA-003 fixture corpus.

The authoritative corpus and suite live under
``ipfs_kit_py/tests/runtime_readiness/``. This module proves the nested package
is present, importable from the parent worktree, and satisfies the same
coverage and safety contracts without re-implementing the expander.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
NESTED_KIT = ROOT / "ipfs_kit_py"
NESTED_FIXTURES = NESTED_KIT / "tests" / "runtime_readiness" / "fixtures"
NESTED_TEST = (
    NESTED_KIT
    / "tests"
    / "runtime_readiness"
    / "foundations"
    / "test_fixture_manifest.py"
)


def _ensure_nested_on_path() -> None:
    if not NESTED_KIT.is_dir():
        raise AssertionError(f"missing nested ipfs_kit_py at {NESTED_KIT}")
    nested = str(NESTED_KIT)
    if nested not in sys.path:
        sys.path.insert(0, nested)


def test_nested_fixture_package_and_validation_test_exist() -> None:
    assert NESTED_FIXTURES.is_dir()
    assert NESTED_TEST.is_file()
    for name in ("recipes.py", "schema.py", "expand.py", "safety.py", "catalog.py"):
        assert (NESTED_FIXTURES / name).is_file()
    text = NESTED_TEST.read_text(encoding="utf-8")
    assert "REQUIRED_COVERAGE_CATEGORIES" in text
    assert "CONFIRMED_BLOCKERS" in text
    assert "content_id" in text


def test_nested_fixture_corpus_covers_acceptance_criteria() -> None:
    _ensure_nested_on_path()
    fixtures_mod = importlib.import_module("tests.runtime_readiness.fixtures")
    manifest = fixtures_mod.build_manifest()
    fixtures_mod.validate_manifest(manifest)
    covered = set(manifest["coverage"]["covered_categories"])
    assert set(fixtures_mod.REQUIRED_COVERAGE_CATEGORIES) <= covered
    assert set(manifest["confirmed_blockers"]) == set(fixtures_mod.CONFIRMED_BLOCKERS)
    for ucan in fixtures_mod.REQUIRED_UCAN_VARIANTS:
        assert ucan in covered
    fixtures = fixtures_mod.expand_all_recipes()
    assert len(fixtures) >= 16
    assert manifest["content_id"].startswith("sha256:")
    for fixture in fixtures:
        fixtures_mod.validate_fixture(fixture)
        fixtures_mod.assert_fixture_safe(fixture)
        assert fixture["expected_trace"]["finite"] is True
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
    interface = manifest["interface"]
    assert "RuntimeReadinessFixture@1" in interface
    assert "FaultSchedule@1" in interface
    assert "ExpectedStateTrace@1" in interface
