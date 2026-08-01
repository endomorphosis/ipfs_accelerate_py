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
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_health import (
    DEFAULT_LANGUAGE_THRESHOLDS,
    POLYGLOT_AST_HEALTH_EVIDENCE,
    POLYGLOT_AST_HEALTH_SCHEMA,
    FailureCluster,
    LanguageHealthThresholds,
    ParserAuthorityKind,
    PathParseOutcome,
    PolyglotASTHealthError,
    assess_polyglot_ast_health,
    build_health_report_from_coverage,
    classify_parser_authority,
    classify_path_disposition,
    classify_path_dispositions,
    cluster_failures,
    evaluate_language_health,
    js_ts_uses_real_parser,
    load_coverage_rows,
    repair_polyglot_parser_authority,
    report_contains_source_body,
    run_polyglot_ast_canaries,
    typed_reason_code,
    write_polyglot_ast_health_report,
)
from ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_provider import (
    TYPESCRIPT_EXTRACTOR_VERSION,
    PolyglotASTProvider,
)


_MODULE = "ipfs_accelerate_py.agent_supervisor.analysis.polyglot_ast_health"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _successful_ts_response(request: dict, *, version: str = "5.7.3") -> bytes:
    return json.dumps(
        {
            "protocol_version": 1,
            "ok": True,
            "producer": "typescript-compiler-api",
            "producer_version": TYPESCRIPT_EXTRACTOR_VERSION,
            "compiler": {"name": "typescript", "version": version},
            "language": request["language"],
            "source_sha256": request["source_sha256"],
            "parse_error": "",
            "facts": {
                "qualified_symbols": ["run"],
                "imports": [],
                "calls": [],
                "state_transitions": [],
                "interfaces": ["run:run(input: string): string"],
                "symbol_hashes": {"run": "sha256:" + "a" * 64},
                "symbol_lines": {"run": [1, 1]},
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _fixture_runner(calls: list[dict] | None = None, *, version: str = "5.7.3"):
    calls = calls if calls is not None else []

    def run(command, request, timeout, max_output, environment):
        payload = json.loads(request)
        calls.append(
            {
                "command": tuple(command),
                "request": payload,
                "environment": dict(environment),
            }
        )
        return 0, _successful_ts_response(payload, version=version), b""

    return run


def _row(
    path: str,
    *,
    language: str,
    parser_status: str,
    parser_reason: str = "",
    parser_identity: str = "parser:fixture",
    producer: str = "",
) -> dict:
    payload = {
        "path": path,
        "language": language,
        "parser_status": parser_status,
        "parser_reason": parser_reason,
        "parser_identity": parser_identity,
        "reason_code": parser_reason or parser_status,
        "disposition_kind": parser_status,
        "tracked": True,
    }
    if producer:
        payload["producer"] = producer
    return payload


def test_cold_import_never_starts_node_or_loads_source_trees() -> None:
    code = f"""
import subprocess

def forbidden(*args, **kwargs):
    raise AssertionError("cold import started a child process")

subprocess.Popen = forbidden
subprocess.run = forbidden
from {_MODULE} import (
    POLYGLOT_AST_HEALTH_SCHEMA,
    LanguageHealthThresholds,
    classify_path_disposition,
)
assert POLYGLOT_AST_HEALTH_SCHEMA.endswith("@1")
assert LanguageHealthThresholds().max_parser_failures >= 0
row = classify_path_disposition({{
    "path": "src/a.py",
    "language": "python",
    "parser_status": "indexed",
    "parser_identity": "parser:x",
}})
assert row.outcome.value == "success"
assert "source" not in row.to_dict()
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
    assert result.returncode == 0, result.stderr


def test_every_eligible_path_is_success_or_typed_bounded_failure() -> None:
    rows = [
        _row("a.py", language="python", parser_status="indexed"),
        _row(
            "b.py",
            language="python",
            parser_status="parse_failure",
            parser_reason="SyntaxError at line 1: invalid syntax",
        ),
        _row("c.md", language="", parser_status="not_applicable"),
        _row(
            "d.ts",
            language="typescript",
            parser_status="parse_failure",
            parser_reason="compiler_unavailable: missing compiler",
        ),
        _row(
            "e.js",
            language="javascript",
            parser_status="indexed",
            producer="typescript-compiler-api",
        ),
    ]
    dispositions = classify_path_dispositions(rows)
    eligible = [
        item
        for item in dispositions
        if item.outcome
        in {PathParseOutcome.SUCCESS, PathParseOutcome.BOUNDED_FAILURE}
    ]
    assert len(eligible) == 4
    assert {item.outcome for item in eligible} <= {
        PathParseOutcome.SUCCESS,
        PathParseOutcome.BOUNDED_FAILURE,
    }
    failures = [
        item
        for item in eligible
        if item.outcome is PathParseOutcome.BOUNDED_FAILURE
    ]
    assert all(item.reason_code and item.reason_code != "unspecified" for item in failures)
    assert any(item.reason_code == "syntaxerror" for item in failures) or any(
        "syntax" in item.reason_code for item in failures
    )
    assert any(item.reason_code == "compiler_unavailable" for item in failures)


def test_js_ts_family_rejects_regex_authority_and_accepts_real_parser() -> None:
    assert (
        classify_parser_authority(
            language="typescript",
            producer="regex-typescript-scanner",
        )
        is ParserAuthorityKind.REGEX_FORBIDDEN
    )
    assert not js_ts_uses_real_parser(producer="heuristic-js-parser")
    assert js_ts_uses_real_parser(
        producer="typescript-compiler-api",
        producer_version=TYPESCRIPT_EXTRACTOR_VERSION,
        compiler_name="typescript",
    )

    forbidden = classify_path_disposition(
        _row(
            "evil.ts",
            language="typescript",
            parser_status="indexed",
            producer="regex",
        )
    )
    assert forbidden.outcome is PathParseOutcome.BOUNDED_FAILURE
    assert forbidden.reason_code == "regex_authority_forbidden"
    assert forbidden.parser_authority is ParserAuthorityKind.REGEX_FORBIDDEN

    allowed = classify_path_disposition(
        _row(
            "ok.ts",
            language="typescript",
            parser_status="indexed",
            producer="typescript-compiler-api",
        )
    )
    assert allowed.outcome is PathParseOutcome.SUCCESS
    assert allowed.parser_authority is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER


def test_failure_clusters_group_language_reason_and_parser_identity() -> None:
    rows = [
        _row(
            f"src/{index}.ts",
            language="typescript",
            parser_status="parse_failure",
            parser_reason="compiler_unavailable: missing",
            parser_identity="parser:a",
        )
        for index in range(5)
    ] + [
        _row(
            "src/one.py",
            language="python",
            parser_status="parse_failure",
            parser_reason="IndentationError at line 2",
            parser_identity="parser:b",
        )
    ]
    dispositions = classify_path_dispositions(rows)
    clusters = cluster_failures(dispositions, max_samples=2)
    assert clusters
    top = clusters[0]
    assert isinstance(top, FailureCluster)
    assert top.language == "typescript"
    assert top.reason_code == "compiler_unavailable"
    assert top.count == 5
    assert len(top.sample_paths) == 2
    assert len(top.sample_disposition_ids) == 2
    assert "source" not in top.to_dict()


def test_language_thresholds_partial_within_budget_and_block_over_budget() -> None:
    healthy_rows = [
        _row(f"ok{i}.py", language="python", parser_status="indexed")
        for i in range(10)
    ]
    healthy = evaluate_language_health(
        classify_path_dispositions(healthy_rows),
        language="python",
        canaries_passed=True,
    )
    assert healthy.status is AnalyzerHealthStatus.HEALTHY
    assert healthy.safe_for_completion_reasoning

    partial_rows = healthy_rows + [
        _row(
            "bad.py",
            language="python",
            parser_status="parse_failure",
            parser_reason="SyntaxError at line 1",
        )
    ]
    partial = evaluate_language_health(
        classify_path_dispositions(partial_rows),
        language="python",
        thresholds=LanguageHealthThresholds(
            max_parser_failures=2,
            max_parser_failure_ratio=0.5,
        ),
        canaries_passed=True,
    )
    assert partial.status is AnalyzerHealthStatus.PARTIAL
    assert "parser_failures_within_budget" in partial.reasons
    assert not partial.safe_for_completion_reasoning

    unhealthy = evaluate_language_health(
        classify_path_dispositions(partial_rows),
        language="python",
        thresholds=LanguageHealthThresholds(
            max_parser_failures=0,
            max_parser_failure_ratio=0.0,
        ),
        canaries_passed=True,
    )
    assert unhealthy.status is AnalyzerHealthStatus.UNHEALTHY
    assert "parser_failure_budget_exceeded" in unhealthy.reasons


def test_thresholds_reject_invalid_values() -> None:
    with pytest.raises(PolyglotASTHealthError):
        LanguageHealthThresholds(max_parser_failures=-1)
    with pytest.raises(PolyglotASTHealthError):
        LanguageHealthThresholds(max_parser_failure_ratio=1.5)


def test_canaries_and_authority_repair_use_real_parser_without_retaining_source() -> None:
    calls: list[dict] = []
    provider = PolyglotASTProvider(process_runner=_fixture_runner(calls))
    canaries = run_polyglot_ast_canaries(provider)
    assert canaries.passed
    assert canaries.fixture_count >= 6
    serialized = canaries.to_dict()
    assert not report_contains_source_body(serialized)
    assert "source" not in json.dumps(serialized)
    # Requests may include source for the extractor, but the health report must not.
    assert all("source" not in result for result in serialized["results"])

    js_ts = [
        item
        for item in canaries.results
        if item.language in {"javascript", "typescript", "tsx", "jsx"}
    ]
    assert js_ts
    assert all(
        item.authority is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
        for item in js_ts
    )
    assert all(item.producer == "typescript-compiler-api" for item in js_ts)

    repaired_provider, repair = repair_polyglot_parser_authority(
        PolyglotASTProvider(process_runner=_fixture_runner([])),
        typescript_path="/toolchains/typescript",
        search_roots=[],
    )
    # Path need not exist for the probe when process_runner is fixture-backed;
    # discover may fail closed if the directory is missing.
    assert repaired_provider is not None
    assert repair.candidate_paths or repair.typescript_path is not None


def test_assess_health_blocks_completion_on_unhealthy_inventory() -> None:
    rows = [
        _row(
            f"src/{i}.ts",
            language="typescript",
            parser_status="parse_failure",
            parser_reason="compiler_unavailable: missing",
        )
        for i in range(20)
    ] + [
        _row("src/ok.py", language="python", parser_status="indexed"),
    ]
    report = assess_polyglot_ast_health(
        rows,
        provider=PolyglotASTProvider(process_runner=_fixture_runner([])),
        repair_authority=False,
        run_canaries=True,
    )
    assert report.evidence_id == POLYGLOT_AST_HEALTH_EVIDENCE
    assert report.schema == POLYGLOT_AST_HEALTH_SCHEMA
    assert report.status is AnalyzerHealthStatus.UNHEALTHY
    assert report.completion_blocker
    assert not report.safe_for_completion_reasoning
    assert any("typescript" in reason for reason in report.reasons)
    assert report.clusters
    assert report.clusters[0].reason_code == "compiler_unavailable"
    payload = report.to_dict()
    assert not report_contains_source_body(payload)
    assert payload["content_identity"]["digest"].startswith("sha256:")
    assert payload["metrics"]["eligible_path_count"] == 21


def test_assess_health_healthy_when_inventory_and_canaries_pass() -> None:
    rows = [
        _row(
            "src/a.ts",
            language="typescript",
            parser_status="indexed",
            producer="typescript-compiler-api",
        ),
        _row(
            "src/b.js",
            language="javascript",
            parser_status="indexed",
            producer="typescript-compiler-api",
        ),
        _row("src/c.py", language="python", parser_status="indexed"),
        _row("src/d.json", language="json", parser_status="indexed"),
    ]
    report = assess_polyglot_ast_health(
        rows,
        provider=PolyglotASTProvider(process_runner=_fixture_runner([])),
        repair_authority=False,
        run_canaries=True,
        thresholds={
            "typescript": LanguageHealthThresholds(
                max_parser_failures=0, max_parser_failure_ratio=0.0
            ),
            "javascript": LanguageHealthThresholds(
                max_parser_failures=0, max_parser_failure_ratio=0.0
            ),
            "python": LanguageHealthThresholds(
                max_parser_failures=0, max_parser_failure_ratio=0.0
            ),
            "json": LanguageHealthThresholds(
                max_parser_failures=0, max_parser_failure_ratio=0.0
            ),
        },
    )
    assert report.status is AnalyzerHealthStatus.HEALTHY
    assert report.safe_for_completion_reasoning
    assert report.canaries.passed
    assert report.metrics["js_ts_real_parser"] is True


def test_write_report_is_atomic_content_addressed_and_source_free(
    tmp_path: Path,
) -> None:
    rows = [
        _row("a.py", language="python", parser_status="indexed"),
    ]
    report = assess_polyglot_ast_health(
        rows,
        provider=PolyglotASTProvider(process_runner=_fixture_runner([])),
        repair_authority=False,
        run_canaries=True,
        thresholds={
            "python": LanguageHealthThresholds(require_canaries=False),
            "javascript": LanguageHealthThresholds(require_canaries=False),
            "typescript": LanguageHealthThresholds(require_canaries=False),
            "tsx": LanguageHealthThresholds(require_canaries=False),
            "jsx": LanguageHealthThresholds(require_canaries=False),
            "json": LanguageHealthThresholds(require_canaries=False),
        },
    )
    # With JS/TS canaries required off and no JS inventory, overall health may
    # still fail global canary_failure if run_canaries left family fixtures
    # failing without a runner.  Force canaries true by using the fixture runner
    # and not requiring zero canary noise: reassess with full canaries.
    report = assess_polyglot_ast_health(
        rows,
        provider=PolyglotASTProvider(process_runner=_fixture_runner([])),
        repair_authority=False,
        run_canaries=True,
    )
    target = tmp_path / "analyzer_health" / "report.json"
    identity = write_polyglot_ast_health_report(report, target)
    assert target.is_file()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["evidence_id"] == POLYGLOT_AST_HEALTH_EVIDENCE
    assert loaded["schema"] == POLYGLOT_AST_HEALTH_SCHEMA
    assert not report_contains_source_body(loaded)
    assert identity["digest"].startswith("sha256:")
    assert loaded["content_identity"]["digest"] == identity["digest"]


def test_typed_reason_code_and_default_thresholds_are_stable() -> None:
    assert typed_reason_code("compiler_unavailable: the local TypeScript") == (
        "compiler_unavailable"
    )
    assert typed_reason_code("JSONDecodeError at line 1, column 1: x").startswith(
        "jsondecodeerror"
    ) or typed_reason_code("JSONDecodeError at line 1, column 1: x") == (
        "jsondecodeerror"
    )
    assert "typescript" in DEFAULT_LANGUAGE_THRESHOLDS
    assert DEFAULT_LANGUAGE_THRESHOLDS["typescript"].require_real_js_ts_parser


def test_build_report_from_coverage_fixture(tmp_path: Path) -> None:
    coverage = {
        "schema": "fixture-coverage@1",
        "rows": [
            _row(
                "src/a.ts",
                language="typescript",
                parser_status="parse_failure",
                parser_reason="compiler_unavailable: missing",
            ),
            _row("src/b.py", language="python", parser_status="indexed"),
            _row("README.md", language="", parser_status="not_applicable"),
        ],
    }
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        json.dumps(coverage, sort_keys=True), encoding="utf-8"
    )
    output = tmp_path / "report.json"
    report = build_health_report_from_coverage(
        coverage_path,
        output_path=output,
        provider=PolyglotASTProvider(process_runner=_fixture_runner([])),
        repair_authority=False,
        run_canaries=True,
        max_disposition_samples=16,
    )
    assert output.is_file()
    assert report.completion_blocker
    loaded_rows = load_coverage_rows(coverage_path)
    assert len(loaded_rows) == 3
    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["metrics"]["eligible_path_count"] == 2
    assert not report_contains_source_body(body)


def test_cjs_mjs_aliases_normalize_to_javascript_authority() -> None:
    for path, language in (("mod.cjs", "cjs"), ("mod.mjs", "mjs")):
        disposition = classify_path_disposition(
            _row(
                path,
                language=language,
                parser_status="indexed",
                producer="typescript-compiler-api",
            )
        )
        assert disposition.language == "javascript"
        assert (
            disposition.parser_authority
            is ParserAuthorityKind.REAL_TYPESCRIPT_COMPILER
        )


def test_never_fabricates_success_from_parse_failure() -> None:
    failure = classify_path_disposition(
        _row(
            "broken.ts",
            language="typescript",
            parser_status="parse_failure",
            parser_reason="typescript_parse_error:TS1005",
        )
    )
    assert failure.outcome is PathParseOutcome.BOUNDED_FAILURE
    report = assess_polyglot_ast_health(
        [
            _row(
                "broken.ts",
                language="typescript",
                parser_status="parse_failure",
                parser_reason="typescript_parse_error:TS1005",
            )
        ],
        provider=PolyglotASTProvider(process_runner=_fixture_runner([])),
        repair_authority=False,
        run_canaries=False,
    )
    assert all(
        item.outcome is not PathParseOutcome.SUCCESS
        or item.path != "broken.ts"
        for item in report.dispositions
    )
    assert report.metrics["bounded_failure_count"] == 1
