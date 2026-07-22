from __future__ import annotations

import math

import pytest

from ipfs_accelerate_py.agent_supervisor.analyzer_health import (
    ANALYZER_CANARY_FIXTURES,
    ANALYZER_SUPPORTED_FINDING_KINDS,
    ANALYZER_SUPPORTED_PARSER_PATHS,
    AnalyzerCanaryReport,
    AnalyzerCanaryResult,
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
    classify_analyzer_health,
    impossible_candidate_funnel,
    run_analyzer_canaries,
    validate_canary_registry,
)
from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    CODEBASE_SCAN_ANALYZER_VERSION,
    run_codebase_analyzer_canaries,
)


def _passing_canaries() -> AnalyzerCanaryReport:
    return AnalyzerCanaryReport(
        CODEBASE_SCAN_ANALYZER_VERSION,
        (AnalyzerCanaryResult("fixture", "line_source", (), ()),),
    )


def _inventory(**overrides: object) -> dict[str, object]:
    inventory: dict[str, object] = {
        "git_roots": 1,
        "expected_git_root_count": 1,
        "tracked_files": 10,
        "eligible_files": 8,
        "parsed_files": 8,
        "cache_hits": 0,
        "excluded_files": 2,
        "parser_failures": 0,
        "raw_candidates": 3,
        "seen_candidates": 1,
        "deduplicated_candidates": 1,
        "rejected_candidates": 0,
        "appended_tasks": 1,
        "coverage_complete": True,
    }
    inventory.update(overrides)
    return inventory


def test_canary_registry_covers_every_version_kind_and_parser_path_deterministically():
    assert validate_canary_registry() == ()
    assert set(ANALYZER_CANARY_FIXTURES) == {CODEBASE_SCAN_ANALYZER_VERSION}
    expected_pairs = {
        (parser_path, kind)
        for parser_path in ANALYZER_SUPPORTED_PARSER_PATHS[CODEBASE_SCAN_ANALYZER_VERSION]
        for kind in ANALYZER_SUPPORTED_FINDING_KINDS[CODEBASE_SCAN_ANALYZER_VERSION]
    }
    fixtures = ANALYZER_CANARY_FIXTURES[CODEBASE_SCAN_ANALYZER_VERSION]
    observed_pairs = {
        (fixture.parser_path, kind)
        for fixture in fixtures
        for kind in fixture.expected_finding_kinds
    }
    assert observed_pairs == expected_pairs

    first = run_codebase_analyzer_canaries().to_dict()
    second = run_codebase_analyzer_canaries().to_dict()
    assert first == second
    assert first["passed"] is True
    assert first["fixture_count"] == 6


def test_unknown_analyzer_version_has_a_fail_closed_missing_canary_report():
    report = run_analyzer_canaries("unknown/v9", lambda *_: ((), "line_source", ""))
    assert report.registry_present is False
    assert report.passed is False
    health = classify_analyzer_health(_inventory(), canaries=report)
    assert health.status is AnalyzerHealthStatus.UNHEALTHY
    assert health.reasons == ("missing_canaries",)


@pytest.mark.parametrize(
    ("overrides", "thresholds", "status", "reason"),
    [
        ({}, {}, AnalyzerHealthStatus.HEALTHY, ""),
        (
            {"parser_failures": 1, "parsed_files": 7},
            {"max_parser_failures": 1, "max_parser_failure_ratio": 1.0},
            AnalyzerHealthStatus.PARTIAL,
            "parser_failures_within_budget",
        ),
        (
            {"parser_failures": 2, "parsed_files": 6},
            {"max_parser_failures": 1, "max_parser_failure_ratio": 1.0},
            AnalyzerHealthStatus.UNHEALTHY,
            "parser_failure_budget_exceeded",
        ),
        (
            {"excluded_files": 8, "eligible_files": 2, "parsed_files": 2},
            {"max_excluded_file_ratio": 0.5},
            AnalyzerHealthStatus.UNHEALTHY,
            "excluded_file_budget_exceeded",
        ),
        (
            {"git_roots": 1, "expected_git_root_count": 2},
            {"min_git_root_discovery_ratio": 0.5},
            AnalyzerHealthStatus.PARTIAL,
            "incomplete_git_root_discovery",
        ),
        (
            {"git_roots": 1, "expected_git_root_count": 2},
            {"min_git_root_discovery_ratio": 1.0},
            AnalyzerHealthStatus.UNHEALTHY,
            "git_root_discovery_below_budget",
        ),
        (
            {"tracked_files": 11},
            {},
            AnalyzerHealthStatus.PARTIAL,
            "unclassified_tracked_files",
        ),
        (
            {"eligible_files": 9, "excluded_files": 1},
            {},
            AnalyzerHealthStatus.PARTIAL,
            "unparsed_eligible_files",
        ),
        (
            {"raw_candidates": 4},
            {},
            AnalyzerHealthStatus.UNHEALTHY,
            "impossible_funnel:raw_candidates_do_not_balance",
        ),
    ],
)
def test_health_classification_covers_budgets_and_impossible_evidence(
    overrides, thresholds, status, reason
):
    report = classify_analyzer_health(
        _inventory(**overrides),
        canaries=_passing_canaries(),
        thresholds=thresholds,
    )
    assert report.status is status
    if reason:
        assert reason in report.reasons
    else:
        assert report.reasons == ()
    assert report.safe_for_completion_reasoning is (status is AnalyzerHealthStatus.HEALTHY)


def test_impossible_funnel_accepts_ref201_dispositions_and_rejects_overages():
    assert impossible_candidate_funnel(_inventory()) == ()
    failures = impossible_candidate_funnel(
        _inventory(appended_tasks=4, raw_candidates=3)
    )
    assert "raw_candidates_do_not_balance" in failures
    assert "candidates_without_parsed_files" in impossible_candidate_funnel(
        _inventory(parsed_files=0, eligible_files=0, excluded_files=10)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_parser_failures": -1},
        {"min_git_roots": -1},
        {"max_parser_failure_ratio": -0.1},
        {"max_excluded_file_ratio": 1.1},
        {"min_git_root_discovery_ratio": math.nan},
    ],
)
def test_thresholds_reject_invalid_failure_budgets(kwargs):
    with pytest.raises(ValueError):
        AnalyzerHealthThresholds(**kwargs)


def test_thresholds_round_trip_and_are_embedded_in_health_report():
    thresholds = AnalyzerHealthThresholds(
        max_parser_failures=3,
        max_parser_failure_ratio=0.25,
        max_excluded_file_ratio=0.75,
        min_git_root_discovery_ratio=0.8,
        min_git_roots=2,
    )
    assert AnalyzerHealthThresholds.from_value(thresholds.to_dict()) == thresholds
    report = classify_analyzer_health(
        _inventory(git_roots=2, expected_git_root_count=2),
        canaries=_passing_canaries(),
        thresholds=thresholds,
    )
    assert report.to_dict()["thresholds"] == thresholds.to_dict()
