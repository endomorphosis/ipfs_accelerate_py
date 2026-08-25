"""Tests for SCA-231 parser-failure triage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analyzer_health import (
    AnalyzerHealthStatus,
)
from ipfs_accelerate_py.agent_supervisor.analysis.parser_failure_triage import (
    DEFAULT_REVIEWED_EXCLUSION_POLICY,
    PARSER_FAILURE_TRIAGE_EVIDENCE,
    PARSER_FAILURE_TRIAGE_INTERFACE,
    PARSER_FAILURE_TRIAGE_SCHEMA,
    REVIEWED_MAX_PARSER_FAILURES,
    REVIEWED_MAX_PARSER_FAILURE_RATIO,
    ClusterDispositionKind,
    ParserFailureTriageError,
    TriageAction,
    actionable_repair_family,
    apply_triage_to_rows,
    assess_health_after_triage,
    build_triage_from_index,
    classify_member_disposition,
    cluster_parser_failures,
    default_parser_repairs,
    detect_shebang_extension_mismatch,
    is_protected_contract_surface,
    member_from_row,
    normalize_cluster_reason,
    path_family_for,
    project_health_gate,
    run_parser_repair_fixtures,
    triage_parser_failures,
    write_parser_failure_triage_report,
)
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_health import (
    PathParseOutcome,
    build_health_report_from_coverage,
    cluster_failures,
    classify_path_dispositions,
    path_family_for_health,
)


_MODULE = "ipfs_accelerate_py.agent_supervisor.analysis.parser_failure_triage"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# test/api → test → ipfs_accelerate → external → workspace root
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

_DIAGNOSTIC_INDEX_CANDIDATES = (
    Path(
        "/home/barberb/lift_coding/data/agent_supervisor/"
        "swissknife_contract_assurance/audit/current-index-20260729/current.json"
    ),
    _WORKSPACE_ROOT
    / "data/agent_supervisor/swissknife_contract_assurance/audit/"
    "current-index-20260729/current.json",
    Path(
        "data/agent_supervisor/swissknife_contract_assurance/audit/"
        "current-index-20260729/current.json"
    ),
)

_TRIAGE_OUTPUT = (
    _WORKSPACE_ROOT
    / "data/agent_supervisor/swissknife_contract_assurance/audit/"
    "current-index-20260729/parser-failure-triage.json"
)


def _failure_row(
    path: str,
    *,
    language: str = "typescript@typescript-5.9.3",
    parser_reason: str = "typescript_parse_error:TS1002@9:79:Unterminated string literal.",
    parser_identity: str = "parser:fixture",
    content_digest: str = "",
) -> dict:
    return {
        "path": path,
        "language": language,
        "parser_status": "parse_failure",
        "parser_reason": parser_reason,
        "parser_identity": parser_identity,
        "reason_code": parser_reason,
        "disposition_kind": "parse_failure",
        "content_digest": content_digest or f"sha256:{path}",
        "tracked": True,
    }


def _success_row(path: str, *, language: str = "typescript") -> dict:
    return {
        "path": path,
        "language": language,
        "parser_status": "indexed",
        "parser_identity": "parser:fixture",
        "producer": "typescript-compiler-api",
        "tracked": True,
    }


def _diagnostic_index_path() -> Path | None:
    for candidate in _DIAGNOSTIC_INDEX_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def test_cold_import_never_starts_node() -> None:
    code = f"""
import subprocess

def forbidden(*args, **kwargs):
    raise AssertionError("cold import started a child process")

subprocess.Popen = forbidden
subprocess.run = forbidden
from {_MODULE} import (
    PARSER_FAILURE_TRIAGE_SCHEMA,
    path_family_for,
    is_protected_contract_surface,
)
assert PARSER_FAILURE_TRIAGE_SCHEMA.endswith("@1")
assert path_family_for("ipfs_accelerate_js/test/unit/test_hf_x.ts").endswith("test_hf_*")
assert is_protected_contract_surface("test/utils/mockMCPClient.js")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_PACKAGE_ROOT), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_normalize_cluster_reason_is_stable_across_column_drift() -> None:
    a = normalize_cluster_reason(
        "typescript_parse_error:TS1002@9:79:Unterminated string literal."
        "|TS1005@14:5:'{' expected."
    )
    b = normalize_cluster_reason(
        "typescript_parse_error:TS1002@99:1:Unterminated string literal."
        "|TS1005@200:40:'{' expected."
    )
    assert a == b == "typescript_parse_error:TS1002|TS1005"
    assert normalize_cluster_reason(
        "file_bytes_exceeded: source exceeds 16777216 UTF-8 bytes"
    ) == "file_bytes_exceeded"
    assert normalize_cluster_reason(
        "JSONDecodeError at line 1, column 1: Expecting value"
    ) == "json_decode_error"


def test_path_family_and_protected_surface_classification() -> None:
    assert (
        path_family_for("ipfs_accelerate_js/test/unit/test_hf_ernie.ts")
        == "ipfs_accelerate_js/test/unit/test_hf_*"
    )
    assert path_family_for_health(
        "ipfs_accelerate_js/test/unit/test_hf_ernie.ts"
    ) == path_family_for("ipfs_accelerate_js/test/unit/test_hf_ernie.ts")
    assert is_protected_contract_surface("test/utils/mockMCPClient.js")
    assert is_protected_contract_surface("src/runtime/mcp/server.ts")
    assert not is_protected_contract_surface(
        "ipfs_accelerate_js/test/unit/test_hf_ernie.ts"
    )


def test_reviewed_exclusion_cannot_hide_mcp_surface() -> None:
    member = member_from_row(
        _failure_row(
            "test/utils/mockMCPClient.js",
            language="javascript@typescript-5.9.3",
            parser_reason="typescript_parse_error:TS1005@241:5:',' expected.",
        )
    )
    disposition, action, rule = classify_member_disposition(member)
    assert member.protected_surface is True
    assert disposition is ClusterDispositionKind.GENUINE_SOURCE_DEFECT
    assert action is TriageAction.COUNT_AS_FAILURE
    assert rule == ""

    # Even a crafted policy matching the path must not apply (protected guard).
    from ipfs_accelerate_py.agent_supervisor.analysis.parser_failure_triage import (
        ReviewedExclusionRule,
    )

    evil = ReviewedExclusionRule(
        rule_id="policy:evil-exclude-mcp",
        description="must not apply to MCP",
        path_prefixes=("test/utils/",),
    )
    assert evil.matches(
        path="test/utils/mockMCPClient.js",
        language="javascript",
        reason_code="typescript_parse_error:TS1005",
        raw_reason="typescript_parse_error:TS1005",
    ) is False


def test_auto_converted_test_fixtures_are_excludable() -> None:
    member = member_from_row(
        _failure_row("ipfs_accelerate_js/test/unit/test_hf_ernie.ts")
    )
    disposition, action, rule = classify_member_disposition(member)
    assert disposition is ClusterDispositionKind.INTENTIONALLY_INVALID_FIXTURE
    assert action is TriageAction.EXCLUDE_FROM_BUDGET
    assert rule.startswith("policy:")


def test_shebang_extension_mismatch_detection() -> None:
    assert detect_shebang_extension_mismatch(
        "#!/bin/bash\nset -e\n", path="runner.js"
    )
    assert not detect_shebang_extension_mismatch(
        "export const x = 1;\n", path="runner.js"
    )


def test_every_failure_belongs_to_one_cluster() -> None:
    rows = [
        _failure_row("ipfs_accelerate_js/test/unit/test_hf_a.ts"),
        _failure_row(
            "ipfs_accelerate_js/test/unit/test_hf_b.ts",
            parser_reason=(
                "typescript_parse_error:TS1002@10:1:Unterminated string literal."
                "|TS1005@11:2:'{' expected."
            ),
        ),
        _failure_row(
            "test/utils/mockMCPClient.js",
            language="javascript@typescript-5.9.3",
            parser_reason="typescript_parse_error:TS1005@241:5:',' expected.",
        ),
        _failure_row(
            "web/legacy-archive/js/apps/strudel-broken.js",
            language="javascript@typescript-5.9.3",
            parser_reason="typescript_parse_error:TS1068@1:1:Unexpected token.",
        ),
        _success_row("src/ok.ts"),
    ]
    report = triage_parser_failures(
        rows,
        source_index_id="test-index",
        eligible_path_count=5,
    )
    assert report.complete
    assert report.failure_count == 4
    assert report.unassigned_count == 0
    assert len(report.assignments) == 4
    assert report.cluster_count == len(report.clusters) >= 1
    # Each path maps to exactly one cluster id.
    paths = [item.member.path for item in report.assignments]
    assert len(paths) == len(set(paths))
    cluster_ids = {item.cluster_id for item in report.clusters}
    for assignment in report.assignments:
        assert assignment.cluster_id in cluster_ids
    # Cluster ids are content-addressed and stable.
    again = triage_parser_failures(
        rows, source_index_id="test-index", eligible_path_count=5
    )
    assert [c.cluster_id for c in again.clusters] == [
        c.cluster_id for c in report.clusters
    ]


def test_apply_triage_never_relabels_malformed_source_as_success() -> None:
    rows = [
        _failure_row("ipfs_accelerate_js/test/unit/test_hf_a.ts"),
        _failure_row(
            "test/utils/mockMCPClient.js",
            language="javascript",
            parser_reason="typescript_parse_error:TS1005@1:1:x",
        ),
    ]
    report = triage_parser_failures(rows, eligible_path_count=2)
    rewritten = apply_triage_to_rows(rows, report.assignments)
    by_path = {row["path"]: row for row in rewritten}
    # Fixture may be excluded but never becomes indexed/success.
    fixture = by_path["ipfs_accelerate_js/test/unit/test_hf_a.ts"]
    assert fixture["parser_status"] in {"excluded", "unsupported", "parse_failure"}
    assert fixture["parser_status"] not in {"indexed", "success", "parsed"}
    # MCP surface remains a failure.
    mcp = by_path["test/utils/mockMCPClient.js"]
    assert mcp["parser_status"] == "parse_failure"


def test_thresholds_cannot_be_weakened() -> None:
    rows = [_failure_row("src/a.ts")]
    with pytest.raises(ParserFailureTriageError) as exc:
        triage_parser_failures(
            rows,
            eligible_path_count=1,
            max_parser_failures=REVIEWED_MAX_PARSER_FAILURES + 1,
        )
    assert exc.value.reason_code == "threshold_weakened"
    with pytest.raises(ParserFailureTriageError) as exc:
        triage_parser_failures(
            rows,
            eligible_path_count=1,
            max_parser_failure_ratio=REVIEWED_MAX_PARSER_FAILURE_RATIO + 0.01,
        )
    assert exc.value.reason_code == "threshold_weakened"


def test_parser_repairs_have_positive_and_negative_fixtures() -> None:
    repairs = default_parser_repairs()
    assert repairs
    for repair in repairs:
        assert repair.positive_fixtures
        assert repair.negative_fixtures
    fixture_report = run_parser_repair_fixtures(repairs)
    assert fixture_report["passed"] is True
    assert fixture_report["fixture_count"] >= 6


def test_health_gate_projection_within_budget() -> None:
    # 300 eligible, 2 residual failures after exclusions → within 10 and 1%.
    members = [
        member_from_row(
            _failure_row(f"ipfs_accelerate_js/test/unit/test_hf_{i}.ts")
        )
        for i in range(20)
    ] + [
        member_from_row(
            _failure_row(
                "test/utils/mockMCPClient.js",
                language="javascript",
                parser_reason="typescript_parse_error:TS1005@1:1:x",
            )
        ),
        member_from_row(
            _failure_row(
                "test/unit/cli/chat-command.test.js",
                language="javascript",
                parser_reason="typescript_parse_error:TS1005@199:1:'}' expected.",
            )
        ),
    ]
    clusters, assignments = cluster_parser_failures(members)
    assert clusters
    gate = project_health_gate(
        eligible_path_count=300,
        assignments=assignments,
        max_parser_failures=REVIEWED_MAX_PARSER_FAILURES,
        max_parser_failure_ratio=REVIEWED_MAX_PARSER_FAILURE_RATIO,
    )
    assert gate.residual_failure_count == 2
    assert gate.excluded_failure_count == 20
    assert gate.residual_failure_ratio <= REVIEWED_MAX_PARSER_FAILURE_RATIO
    assert gate.meets_gate is True
    assert gate.max_parser_failures == 10
    assert gate.max_parser_failure_ratio == 0.01


def test_assess_health_after_triage_uses_unchanged_thresholds() -> None:
    # Enough successes in both TS and JS so residual MCP failure stays within
    # the 1% ratio (and absolute count of 10).
    rows = (
        [_success_row(f"src/ok_{i}.ts") for i in range(200)]
        + [_success_row(f"src/js_ok_{i}.js", language="javascript") for i in range(200)]
        + [
            _failure_row("ipfs_accelerate_js/test/unit/test_hf_a.ts"),
            _failure_row("ipfs_accelerate_js/test/unit/test_hf_b.ts"),
            _failure_row(
                "test/utils/mockMCPClient.js",
                language="javascript",
                parser_reason="typescript_parse_error:TS1005@1:1:x",
            ),
        ]
    )
    report = triage_parser_failures(rows, eligible_path_count=403)
    health = assess_health_after_triage(
        rows,
        report=report,
        run_canaries=False,
        repair_authority=False,
    )
    # Residual failures within budget → not unhealthy from parser budget.
    assert health.status in {
        AnalyzerHealthStatus.HEALTHY,
        AnalyzerHealthStatus.PARTIAL,
    }
    assert "parser_failure_budget_exceeded" not in health.reasons
    # Thresholds on JS/TS remain at reviewed 10/0.01.
    for language_report in health.language_health:
        if language_report.language in {"javascript", "typescript"}:
            assert language_report.thresholds.max_parser_failures == 10
            assert language_report.thresholds.max_parser_failure_ratio == 0.01


def test_polyglot_path_family_cluster_flag() -> None:
    rows = [
        _failure_row("ipfs_accelerate_js/test/unit/test_hf_a.ts"),
        _failure_row("ipfs_accelerate_js/test/unit/test_hf_b.ts"),
        _failure_row(
            "web/legacy-archive/main.ts",
            parser_reason="typescript_parse_error:TS1005@1:1:x",
        ),
    ]
    dispositions = classify_path_dispositions(rows)
    plain = cluster_failures(dispositions, include_path_family=False)
    with_family = cluster_failures(dispositions, include_path_family=True)
    assert plain
    assert with_family
    assert any(item.path_family for item in with_family)
    assert all(not item.path_family for item in plain)


def test_write_report_is_body_free(tmp_path: Path) -> None:
    rows = [
        _failure_row("ipfs_accelerate_js/test/unit/test_hf_a.ts"),
        _failure_row(
            "test/utils/mockMCPClient.js",
            language="javascript",
            parser_reason="typescript_parse_error:TS1005@1:1:x",
        ),
    ]
    report = triage_parser_failures(rows, eligible_path_count=2)
    target = tmp_path / "triage.json"
    identity = write_parser_failure_triage_report(report, target)
    assert identity["digest"].startswith("sha256:")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == PARSER_FAILURE_TRIAGE_SCHEMA
    assert payload["interface"] == PARSER_FAILURE_TRIAGE_INTERFACE
    assert payload["evidence_id"] == PARSER_FAILURE_TRIAGE_EVIDENCE
    blob = json.dumps(payload)
    for forbidden in ("source_body", "source_code", "file_contents"):
        assert forbidden not in blob
    # Repair fixtures serialize digests only.
    for repair in payload["repairs"]:
        for fixture in repair["positive_fixtures"] + repair["negative_fixtures"]:
            assert "source" not in fixture
            assert fixture["source_sha256"].startswith("sha256:")


def test_default_policy_rules_are_reviewed() -> None:
    assert DEFAULT_REVIEWED_EXCLUSION_POLICY
    for rule in DEFAULT_REVIEWED_EXCLUSION_POLICY:
        assert rule.reviewed is True
        assert rule.rule_id.startswith("policy:")
        assert rule.description


@pytest.mark.parametrize(
    "index_required",
    [True],
)
def test_diagnostic_index_258_failures_fully_clustered(index_required: bool) -> None:
    index_path = _diagnostic_index_path()
    if index_path is None:
        if index_required:
            pytest.skip("diagnostic index bd7cd357… not available in this environment")
        return

    report = build_triage_from_index(index_path)
    assert report.failure_count == 258
    assert report.complete
    assert report.unassigned_count == 0
    assert len(report.assignments) == 258
    assert report.cluster_count == len(report.clusters) >= 1
    # Every assignment has a cluster exactly once.
    cluster_ids = {item.cluster_id for item in report.clusters}
    paths = [item.member.path for item in report.assignments]
    assert len(paths) == len(set(paths)) == 258
    for assignment in report.assignments:
        assert assignment.cluster_id in cluster_ids
        assert assignment.member.actionable_family in {
            "UNIT",
            "BROWSER",
            "ACTIVEJS",
            "PYTHON",
            "STRUCTURED",
            "LEGACY",
        }
    expected_family_counts = {
        "UNIT": 232,
        "BROWSER": 9,
        "ACTIVEJS": 4,
        "PYTHON": 3,
        "STRUCTURED": 2,
        "LEGACY": 8,
    }
    assert report.metrics["actionable_family_counts"] == expected_family_counts
    # Protected MCP surface remains budgeted.
    mcp = [
        item
        for item in report.assignments
        if "mockmcp" in item.member.path.casefold()
    ]
    assert mcp
    assert all(item.action is TriageAction.COUNT_AS_FAILURE for item in mcp)
    # Gate uses unchanged thresholds; triage is non-authoritative.
    assert report.health_gate.max_parser_failures == REVIEWED_MAX_PARSER_FAILURES
    assert (
        report.health_gate.max_parser_failure_ratio
        == REVIEWED_MAX_PARSER_FAILURE_RATIO
    )
    assert report.health_gate.meets_gate is True
    assert report.health_gate.residual_failure_count <= REVIEWED_MAX_PARSER_FAILURES
    assert report.metrics["repair_fixtures_passed"] is True
    assert report.to_dict()["non_authoritative"] is True
    assert report.to_dict()["satisfies_fresh_health_authority"] is False
    assert report.to_dict()["satisfies_repair_task"] is False
    assert report.to_dict()["authoritative_health_owner"] == "SCA-512"
    assert actionable_repair_family("test/utils/mockMCPClient.js") == "ACTIVEJS"

    # Persist the declared audit artifact when running in-repo.
    output = _TRIAGE_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    write_parser_failure_triage_report(report, output)
    assert output.is_file()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["failure_count"] == 258
    assert saved["complete"] is True
    assert saved["non_authoritative"] is True
    assert saved["completion_authoritative"] is False
    assert saved["satisfies_fresh_health_authority"] is False
    assert saved["health_gate"]["meets_gate"] is True
    assert saved["health_gate"]["thresholds_unchanged"] is True
    assert saved["health_gate"]["non_authoritative"] is True
    assert saved["metrics"]["actionable_family_counts"] == expected_family_counts
    assert len(saved["metrics"]["repair_family_manifest"]) == 6


def test_build_health_report_optional_triage(tmp_path: Path) -> None:
    rows = [
        _success_row(f"src/ok_{i}.ts") for i in range(50)
    ] + [
        _failure_row(f"ipfs_accelerate_js/test/unit/test_hf_{i}.ts")
        for i in range(15)
    ] + [
        _failure_row(
            "test/utils/mockMCPClient.js",
            language="javascript",
            parser_reason="typescript_parse_error:TS1005@1:1:x",
        ),
    ]
    coverage = {
        "schema": "test-coverage",
        "rows": rows,
    }
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(coverage), encoding="utf-8")
    # Without triage: failure budget exceeded.
    raw = build_health_report_from_coverage(
        path,
        run_canaries=False,
        repair_authority=False,
        apply_parser_failure_triage=False,
        max_disposition_samples=0,
    )
    assert raw.metrics["bounded_failure_count"] == 16
    # With triage: fixtures excluded; residual within budget.
    triaged = build_health_report_from_coverage(
        path,
        run_canaries=False,
        repair_authority=False,
        apply_parser_failure_triage=True,
        include_path_family_clusters=True,
        max_disposition_samples=0,
    )
    assert triaged.metrics["bounded_failure_count"] <= REVIEWED_MAX_PARSER_FAILURES
    assert "parser_failure_budget_exceeded" not in triaged.reasons
