#!/usr/bin/env python3
"""Fail-closed qualification receipts for the external-agent release (EAAEF-171).

Reads a GitHub Actions workflow YAML and/or a pytest report JSON. Rejects
continue-on-error, shell success masking, skipped/xfailed required tests,
unavailable-as-passed, simulations-as-live, and stale historical counts.
Collected and passed populations must be bound and equal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-agent-release-qualification@1"
)
DEFAULT_WORKFLOW: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "external-agent-fabric.yml"
)
DOC_ROOT: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
)
QUALIFICATION_MD: Final[Path] = DOC_ROOT / "QUALIFICATION_REPORT.md"
QUALIFICATION_JSON: Final[Path] = DOC_ROOT / "qualification_report.json"
FINAL_MD: Final[Path] = DOC_ROOT / "FINAL_RECOMMENDATION.md"
FINAL_JSON: Final[Path] = DOC_ROOT / "final_recommendation.json"
REQUIRED_REPORT_KEYS: Final[tuple[str, ...]] = (
    "revisions",
    "evidence_mode",
    "epics",
    "live_eight_container_qualification",
)
UNSUPERVISED_RECOMMENDATIONS: Final[frozenset[str]] = frozenset(
    {
        "go",
        "go_for_autonomous",
        "autonomous",
        "unsupervised",
        "unsupervised_autonomy",
        "unsupervised_go",
        "production",
        "autonomous_go",
    }
)

CONTINUE_ON_ERROR = "continue-on-error"
SHELL_SUCCESS_MASKING = "shell-success-masking"
SKIPPED_REQUIRED = "skipped-required-test"
XFAIL_REQUIRED = "xfail-required-test"
UNAVAILABLE_AS_PASSED = "unavailable-as-passed"
SIMULATED_AS_LIVE = "simulated-as-live"
STALE_HISTORICAL = "stale-historical-counts"
POPULATION_MISMATCH = "collected-passed-mismatch"
MISSING_QUALIFICATION_REPORT = "missing-qualification-report"
UNSUPERVISED_AUTONOMY = "unsupervised-autonomy-claim"

SHELL_MASKING_RE = re.compile(
    r"[|][|]\s*(true|:|exit\s+0)\b",
    re.IGNORECASE,
)
SET_PLUS_E_RE = re.compile(r"\bset\s+\+e\b")
SUITE_PATH_RE = re.compile(r"(?<![\w./-])(test/[^\s'\"\\]+\.py)")
UNAVAILABLE_RE = re.compile(
    r"\b(unavailable|not available|not installed|missing provider)\b",
    re.IGNORECASE,
)

PASSED_OUTCOMES = frozenset({"passed", "pass"})
SKIPPED_OUTCOMES = frozenset({"skipped", "skip"})
FAILED_OUTCOMES = frozenset({"failed", "fail", "error"})
XFAIL_OUTCOMES = frozenset({"xfailed", "xfail", "xpassed", "xpass"})
UNAVAILABLE_OUTCOMES = frozenset({"unavailable", "not_run", "notcollected"})
LIVE_MODES = frozenset({"live", "current"})
SIMULATED_MODES = frozenset({"simulated", "simulation", "sim"})
HISTORICAL_SOURCES = frozenset(
    {"historical", "history", "archive", "archived", "previous_run", "stale"}
)


class QualificationReceiptError(ValueError):
    """Qualification receipt is not admissible."""


@dataclass(frozen=True)
class Rejection:
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class PopulationBinding:
    collected: int
    passed: int
    skipped: int
    failed: int
    xfailed: int
    xpassed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "collected": self.collected,
            "passed": self.passed,
            "skipped": self.skipped,
            "failed": self.failed,
            "xfailed": self.xfailed,
            "xpassed": self.xpassed,
        }


@dataclass(frozen=True)
class QualificationResult:
    ok: bool
    rejections: tuple[Rejection, ...]
    populations: PopulationBinding | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "ok": self.ok,
            "rejections": [item.as_dict() for item in self.rejections],
        }
        if self.populations is not None:
            payload["populations"] = self.populations.as_dict()
        return payload

    def codes(self) -> frozenset[str]:
        return frozenset(item.code for item in self.rejections)


def _nonneg_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QualificationReceiptError(f"{field} must be a non-negative integer")
    if value < 0:
        raise QualificationReceiptError(f"{field} must be a non-negative integer")
    return value


def _optional_nonneg_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _nonneg_int(value, field=field)


def _mapping(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _norm(value: object) -> str:
    return str(value or "").strip().lower()


def extract_required_suites(workflow_text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for match in SUITE_PATH_RE.finditer(workflow_text):
        path = match.group(1)
        if path not in seen:
            seen.append(path)
    return tuple(seen)


def validate_workflow_text(text: str) -> tuple[Rejection, ...]:
    rejections: list[Rejection] = []
    if "continue-on-error" in text:
        rejections.append(
            Rejection(
                CONTINUE_ON_ERROR,
                "required lanes cannot set continue-on-error",
            )
        )
    if SHELL_MASKING_RE.search(text) or SET_PLUS_E_RE.search(text):
        rejections.append(
            Rejection(
                SHELL_SUCCESS_MASKING,
                "required lanes cannot mask shell failure with || true, || :, || exit 0, or set +e",
            )
        )
    return tuple(rejections)


def validate_workflow_path(path: Path) -> QualificationResult:
    if not path.is_file():
        return QualificationResult(
            ok=False,
            rejections=(
                Rejection(CONTINUE_ON_ERROR, f"workflow is missing: {path}"),
            ),
        )
    rejections = validate_workflow_text(path.read_text(encoding="utf-8"))
    return QualificationResult(ok=not rejections, rejections=rejections)


def _summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = report.get("summary")
    return summary if isinstance(summary, Mapping) else report


def _tests(report: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = report.get("tests")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise QualificationReceiptError("tests must be a list")
    tests: list[Mapping[str, Any]] = []
    for item in raw:
        mapping = _mapping(item)
        if mapping is None:
            raise QualificationReceiptError("each test entry must be an object")
        tests.append(mapping)
    return tuple(tests)


def _test_id(test: Mapping[str, Any]) -> str:
    for key in ("nodeid", "id", "name", "node", "path"):
        value = test.get(key)
        if value:
            return str(value)
    return ""


def _outcome(test: Mapping[str, Any]) -> str:
    return _norm(test.get("outcome") or test.get("status") or test.get("result"))


def _reason_text(test: Mapping[str, Any]) -> str:
    parts = [
        test.get("reason"),
        test.get("longrepr"),
        test.get("skip_reason"),
        test.get("wasxfail"),
    ]
    keywords = test.get("keywords")
    if isinstance(keywords, Sequence) and not isinstance(keywords, (str, bytes)):
        parts.extend(str(item) for item in keywords)
    return " ".join(str(part) for part in parts if part)


def _is_required(test: Mapping[str, Any], required: Sequence[str] | None) -> bool:
    if not required:
        return True
    ident = _test_id(test)
    return any(ident == item or item in ident or ident.startswith(item) for item in required)


def _is_unavailable(test: Mapping[str, Any]) -> bool:
    if _truthy(test.get("unavailable")):
        return True
    outcome = _outcome(test)
    if outcome in UNAVAILABLE_OUTCOMES:
        return True
    return bool(UNAVAILABLE_RE.search(_reason_text(test)))


def _is_simulated(obj: Mapping[str, Any]) -> bool:
    if _truthy(obj.get("simulated")) or _truthy(obj.get("simulation")):
        return True
    mode = _norm(obj.get("mode") or obj.get("kind") or obj.get("evidence_kind"))
    return mode in SIMULATED_MODES


def _claims_live(obj: Mapping[str, Any]) -> bool:
    if obj.get("live") is False:
        return False
    if _truthy(obj.get("live")):
        return True
    mode = _norm(obj.get("mode") or obj.get("kind") or obj.get("evidence_kind"))
    if mode in SIMULATED_MODES or mode in HISTORICAL_SOURCES:
        return False
    if mode in LIVE_MODES or mode == "":
        return True
    return False


def _is_historical(obj: Mapping[str, Any]) -> bool:
    if any(_truthy(obj.get(key)) for key in ("historical", "stale", "archived")):
        return True
    if obj.get("current") is False:
        return True
    source = _norm(obj.get("source") or obj.get("counts_source") or obj.get("evidence_kind"))
    return source in HISTORICAL_SOURCES


def _declared_counts(
    report: Mapping[str, Any],
) -> tuple[int | None, int | None, int | None, int | None, int | None, int | None]:
    summary = _summary(report)
    collected = summary.get("collected", report.get("collected"))
    if collected is None:
        collected = summary.get("total", report.get("total"))
    return (
        _optional_nonneg_int(collected, field="collected"),
        _optional_nonneg_int(summary.get("passed", report.get("passed")), field="passed"),
        _optional_nonneg_int(summary.get("skipped", report.get("skipped")), field="skipped"),
        _optional_nonneg_int(
            summary.get("failed", report.get("failed", summary.get("error", report.get("error")))),
            field="failed",
        ),
        _optional_nonneg_int(summary.get("xfailed", report.get("xfailed")), field="xfailed"),
        _optional_nonneg_int(summary.get("xpassed", report.get("xpassed")), field="xpassed"),
    )


def _derived_counts(tests: Sequence[Mapping[str, Any]]) -> PopulationBinding:
    counter: Counter[str] = Counter()
    xfailed = 0
    xpassed = 0
    for test in tests:
        outcome = _outcome(test)
        if outcome in XFAIL_OUTCOMES or _truthy(test.get("wasxfail")):
            if outcome in {"xpassed", "xpass"} or (
                outcome in PASSED_OUTCOMES and _truthy(test.get("wasxfail"))
            ):
                xpassed += 1
            else:
                xfailed += 1
            continue
        if outcome in PASSED_OUTCOMES:
            counter["passed"] += 1
        elif outcome in SKIPPED_OUTCOMES or outcome in UNAVAILABLE_OUTCOMES:
            counter["skipped"] += 1
        elif outcome in FAILED_OUTCOMES:
            counter["failed"] += 1
        else:
            counter["failed"] += 1
    return PopulationBinding(
        collected=len(tests),
        passed=counter["passed"],
        skipped=counter["skipped"],
        failed=counter["failed"],
        xfailed=xfailed,
        xpassed=xpassed,
    )


def validate_pytest_report(
    report: Mapping[str, Any],
    *,
    required: Sequence[str] | None = None,
) -> QualificationResult:
    rejections: list[Rejection] = []
    try:
        tests = _tests(report)
        declared = _declared_counts(report)
    except QualificationReceiptError as exc:
        return QualificationResult(
            ok=False,
            rejections=(Rejection(POPULATION_MISMATCH, str(exc)),),
        )

    derived = _derived_counts(tests) if "tests" in report else None
    collected, passed, skipped, failed, xfailed, xpassed = declared

    if derived is not None:
        if collected is None:
            collected = derived.collected
        elif collected != derived.collected:
            rejections.append(
                Rejection(
                    POPULATION_MISMATCH,
                    f"declared collected {collected} does not bind test list {derived.collected}",
                )
            )
        if passed is None:
            passed = derived.passed
        elif passed != derived.passed:
            rejections.append(
                Rejection(
                    POPULATION_MISMATCH,
                    f"declared passed {passed} does not bind test list {derived.passed}",
                )
            )
        skipped = derived.skipped if skipped is None else skipped
        failed = derived.failed if failed is None else failed
        xfailed = derived.xfailed if xfailed is None else xfailed
        xpassed = derived.xpassed if xpassed is None else xpassed
        if skipped != derived.skipped or failed != derived.failed:
            rejections.append(
                Rejection(
                    POPULATION_MISMATCH,
                    "declared skip/fail counts do not bind the test list",
                )
            )
        if xfailed != derived.xfailed or xpassed != derived.xpassed:
            rejections.append(
                Rejection(
                    POPULATION_MISMATCH,
                    "declared xfail counts do not bind the test list",
                )
            )
    elif any(value is None for value in (collected, passed, skipped, failed, xfailed)):
        rejections.append(
            Rejection(
                POPULATION_MISMATCH,
                "report must bind collected, passed, skipped, failed, and xfailed",
            )
        )
        return QualificationResult(ok=False, rejections=tuple(rejections))

    assert collected is not None
    assert passed is not None
    skipped = 0 if skipped is None else skipped
    failed = 0 if failed is None else failed
    xfailed = 0 if xfailed is None else xfailed
    xpassed = 0 if xpassed is None else xpassed
    populations = PopulationBinding(
        collected=collected,
        passed=passed,
        skipped=skipped,
        failed=failed,
        xfailed=xfailed,
        xpassed=xpassed,
    )

    if collected <= 0 or passed != collected:
        rejections.append(
            Rejection(
                POPULATION_MISMATCH,
                f"collected {collected} is not bound to passed {passed}",
            )
        )

    if _is_historical(report) or any(_is_historical(test) for test in tests):
        rejections.append(
            Rejection(
                STALE_HISTORICAL,
                "historical or stale counts cannot satisfy current qualification",
            )
        )
    if report.get("historical_passed") is not None or report.get("historical_counts") is not None:
        rejections.append(
            Rejection(
                STALE_HISTORICAL,
                "historical count fields cannot substitute for a current population",
            )
        )

    simulated_tests = [test for test in tests if _is_simulated(test)]
    if _is_simulated(report) or simulated_tests:
        if _claims_live(report) or any(_claims_live(test) for test in simulated_tests):
            rejections.append(
                Rejection(
                    SIMULATED_AS_LIVE,
                    "simulated checks cannot be represented as live qualification",
                )
            )
        else:
            rejections.append(
                Rejection(
                    SIMULATED_AS_LIVE,
                    "simulated evidence is not live qualification",
                )
            )

    if _truthy(report.get("unavailable_as_passed")):
        rejections.append(
            Rejection(
                UNAVAILABLE_AS_PASSED,
                "unavailable checks cannot be counted as passed",
            )
        )

    required_hits: set[str] = set()
    for test in tests:
        ident = _test_id(test) or "<unnamed>"
        needed = _is_required(test, required)
        outcome = _outcome(test)
        unavailable = _is_unavailable(test)
        if needed:
            for item in required or ():
                if ident == item or item in ident or ident.startswith(item):
                    required_hits.add(item)
            if outcome in SKIPPED_OUTCOMES or outcome in UNAVAILABLE_OUTCOMES:
                rejections.append(
                    Rejection(
                        SKIPPED_REQUIRED,
                        f"required test skipped: {ident}",
                    )
                )
            if (
                outcome in XFAIL_OUTCOMES
                or _truthy(test.get("wasxfail"))
                or _truthy(test.get("xfail"))
            ):
                rejections.append(
                    Rejection(
                        XFAIL_REQUIRED,
                        f"required test xfailed: {ident}",
                    )
                )
        if unavailable and (
            outcome in PASSED_OUTCOMES
            or _truthy(test.get("counted_as_passed"))
            or needed
            and outcome not in SKIPPED_OUTCOMES
        ):
            rejections.append(
                Rejection(
                    UNAVAILABLE_AS_PASSED,
                    f"unavailable test counted as passed: {ident}",
                )
            )
        elif unavailable and outcome in SKIPPED_OUTCOMES and needed:
            rejections.append(
                Rejection(
                    UNAVAILABLE_AS_PASSED,
                    f"unavailable required test cannot pass: {ident}",
                )
            )

    if required:
        missing = [item for item in required if item not in required_hits]
        if missing:
            rejections.append(
                Rejection(
                    SKIPPED_REQUIRED,
                    "required suites were not collected: " + ", ".join(missing),
                )
            )

    unique: list[Rejection] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in rejections:
        key = (item.code, item.message)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(item)
    return QualificationResult(
        ok=not unique,
        rejections=tuple(unique),
        populations=populations,
    )


def _load_workflow(workflow: str | Path) -> str:
    if isinstance(workflow, Path):
        return workflow.read_text(encoding="utf-8")
    path = Path(workflow)
    if "\n" not in workflow and path.is_file():
        return path.read_text(encoding="utf-8")
    return workflow


def _load_report(report: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    payload = json.loads(Path(report).read_text(encoding="utf-8"))
    mapping = _mapping(payload)
    if mapping is None:
        raise QualificationReceiptError("pytest report JSON must be an object")
    return mapping


def validate_qualification_documents(
    *,
    report_md: Path | None = None,
    report_json: Path | None = None,
) -> QualificationResult:
    """EAAEF-174: QUALIFICATION_REPORT.md/json exist with required keys."""

    md_path = report_md if report_md is not None else QUALIFICATION_MD
    json_path = report_json if report_json is not None else QUALIFICATION_JSON
    rejections: list[Rejection] = []
    if not md_path.is_file():
        rejections.append(
            Rejection(
                MISSING_QUALIFICATION_REPORT,
                f"QUALIFICATION_REPORT.md is missing: {md_path}",
            )
        )
    if not json_path.is_file():
        rejections.append(
            Rejection(
                MISSING_QUALIFICATION_REPORT,
                f"qualification_report.json is missing: {json_path}",
            )
        )
        return QualificationResult(ok=False, rejections=tuple(rejections))
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return QualificationResult(
            ok=False,
            rejections=rejections
            + [
                Rejection(
                    MISSING_QUALIFICATION_REPORT,
                    f"qualification_report.json is unreadable: {exc}",
                )
            ],
        )
    mapping = _mapping(payload)
    if mapping is None:
        return QualificationResult(
            ok=False,
            rejections=rejections
            + [
                Rejection(
                    MISSING_QUALIFICATION_REPORT,
                    "qualification_report.json must be an object",
                )
            ],
        )
    missing = [key for key in REQUIRED_REPORT_KEYS if key not in mapping]
    if missing:
        rejections.append(
            Rejection(
                MISSING_QUALIFICATION_REPORT,
                "qualification_report.json missing keys: " + ", ".join(missing),
            )
        )
    if mapping.get("live_eight_container_qualification") is True:
        rejections.append(
            Rejection(
                SIMULATED_AS_LIVE,
                "live eight-container qualification did not run and cannot be claimed",
            )
        )
    mode = _norm(mapping.get("evidence_mode"))
    if mode in LIVE_MODES and not _truthy(mapping.get("live_runtime_invoked")):
        rejections.append(
            Rejection(
                SIMULATED_AS_LIVE,
                "live evidence_mode requires live_runtime_invoked",
            )
        )
    return QualificationResult(ok=not rejections, rejections=tuple(rejections))


def validate_final_recommendation(
    *,
    recommendation_json: Path | None = None,
    recommendation_md: Path | None = None,
) -> QualificationResult:
    """EAAEF-176: recommendation exists and does not claim unsupervised autonomy."""

    json_path = recommendation_json if recommendation_json is not None else FINAL_JSON
    md_path = recommendation_md if recommendation_md is not None else FINAL_MD
    rejections: list[Rejection] = []
    if not json_path.is_file():
        rejections.append(
            Rejection(
                UNSUPERVISED_AUTONOMY,
                f"final_recommendation.json is missing: {json_path}",
            )
        )
        return QualificationResult(ok=False, rejections=tuple(rejections))
    if md_path is not None and not md_path.is_file():
        rejections.append(
            Rejection(
                UNSUPERVISED_AUTONOMY,
                f"FINAL_RECOMMENDATION.md is missing: {md_path}",
            )
        )
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return QualificationResult(
            ok=False,
            rejections=(
                Rejection(
                    UNSUPERVISED_AUTONOMY,
                    f"final_recommendation.json is unreadable: {exc}",
                ),
            ),
        )
    mapping = _mapping(payload)
    if mapping is None:
        return QualificationResult(
            ok=False,
            rejections=(
                Rejection(
                    UNSUPERVISED_AUTONOMY,
                    "final_recommendation.json must be an object",
                ),
            ),
        )
    recommendation = _norm(mapping.get("recommendation"))
    if recommendation in UNSUPERVISED_RECOMMENDATIONS:
        rejections.append(
            Rejection(
                UNSUPERVISED_AUTONOMY,
                f"recommendation {recommendation!r} claims unsupervised autonomy",
            )
        )
    if _truthy(mapping.get("unsupervised_autonomy")):
        rejections.append(
            Rejection(
                UNSUPERVISED_AUTONOMY,
                "unsupervised_autonomy must not be claimed",
            )
        )
    if mapping.get("live_eight_container_qualification") is True:
        rejections.append(
            Rejection(
                SIMULATED_AS_LIVE,
                "live eight-container qualification did not run and cannot be claimed",
            )
        )
    if recommendation == "supervised_external_pilot" and not _truthy(
        mapping.get("live_eight_container_qualification")
    ):
        rejections.append(
            Rejection(
                UNSUPERVISED_AUTONOMY,
                "supervised_external_pilot requires live eight-container qualification",
            )
        )
    return QualificationResult(ok=not rejections, rejections=tuple(rejections))


def validate_release(
    workflow: str | Path | None = None,
    pytest_report: Mapping[str, Any] | str | Path | None = None,
    *,
    required: Sequence[str] | None = None,
) -> QualificationResult:
    rejections: list[Rejection] = []
    populations: PopulationBinding | None = None
    workflow_text: str | None = None
    if workflow is not None:
        try:
            workflow_text = _load_workflow(workflow)
        except OSError as exc:
            return QualificationResult(
                ok=False,
                rejections=(Rejection(CONTINUE_ON_ERROR, f"workflow unreadable: {exc}"),),
            )
        rejections.extend(validate_workflow_text(workflow_text))
    required_suites: Sequence[str] | None = required
    if required_suites is None and workflow_text is not None and pytest_report is not None:
        extracted = extract_required_suites(workflow_text)
        required_suites = extracted or None
    if pytest_report is not None:
        try:
            report = _load_report(pytest_report)
        except (OSError, json.JSONDecodeError, QualificationReceiptError) as exc:
            return QualificationResult(
                ok=False,
                rejections=(Rejection(POPULATION_MISMATCH, f"pytest report unreadable: {exc}"),),
            )
        report_result = validate_pytest_report(report, required=required_suites)
        rejections.extend(report_result.rejections)
        populations = report_result.populations
    if workflow is None and pytest_report is None:
        rejections.append(
            Rejection(POPULATION_MISMATCH, "workflow YAML and/or pytest report JSON is required")
        )
    unique: list[Rejection] = []
    seen: set[tuple[str, str]] = set()
    for item in rejections:
        key = (item.code, item.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return QualificationResult(
        ok=not unique,
        rejections=tuple(unique),
        populations=populations,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed external-agent release qualification receipts",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=None,
        help="GitHub Actions workflow YAML (defaults to external-agent-fabric.yml)",
    )
    parser.add_argument(
        "--pytest-report",
        type=Path,
        default=None,
        help="pytest report JSON with collected/passed populations",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Check QUALIFICATION_REPORT.md/json required keys (still fail-closed)",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Validate final_recommendation.json and refuse unsupervised autonomy",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    workflow = args.workflow
    document_only = bool(args.report_only or args.terminal) and args.pytest_report is None and (
        args.workflow is None
    )
    if workflow is None and not document_only and (
        args.pytest_report is None or args.terminal or args.report_only
    ):
        workflow = DEFAULT_WORKFLOW
    if workflow is None and args.pytest_report is not None:
        workflow = None
    elif workflow is None and not document_only:
        workflow = DEFAULT_WORKFLOW

    extra: list[Rejection] = []
    populations = None
    rejections: list[Rejection] = []
    if not document_only or args.pytest_report is not None or args.workflow is not None:
        result = validate_release(workflow=workflow, pytest_report=args.pytest_report)
        rejections.extend(result.rejections)
        populations = result.populations
    if args.report_only:
        extra.extend(validate_qualification_documents().rejections)
    if args.terminal:
        extra.extend(validate_final_recommendation().rejections)
    unique: list[Rejection] = []
    seen: set[tuple[str, str]] = set()
    for item in list(rejections) + extra:
        key = (item.code, item.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    result = QualificationResult(
        ok=not unique,
        rejections=tuple(unique),
        populations=populations,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


if __name__ == "__main__":
    sys.exit(main())
