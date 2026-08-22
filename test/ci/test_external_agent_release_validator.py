"""EAAEF-171: fail-closed qualification receipts reject masked or unbound evidence."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_external_agent_release import (
    CONTINUE_ON_ERROR,
    DEFAULT_WORKFLOW,
    POPULATION_MISMATCH,
    SHELL_SUCCESS_MASKING,
    SIMULATED_AS_LIVE,
    SKIPPED_REQUIRED,
    STALE_HISTORICAL,
    UNAVAILABLE_AS_PASSED,
    XFAIL_REQUIRED,
    QualificationResult,
    main,
    validate_pytest_report,
    validate_release,
    validate_workflow_text,
)

PASSING_TESTS = (
    {
        "nodeid": "test/api/test_external_agent_handoff_contracts.py::test_ok",
        "outcome": "passed",
    },
    {
        "nodeid": "test/api/test_external_agent_codex_adapter.py::test_ok",
        "outcome": "passed",
    },
)


def _passing_report(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": "live",
        "current": True,
        "collected": 2,
        "passed": 2,
        "skipped": 0,
        "failed": 0,
        "xfailed": 0,
        "xpassed": 0,
        "tests": [dict(item) for item in PASSING_TESTS],
    }
    payload.update(overrides)
    return payload


def _codes(result: QualificationResult) -> set[str]:
    return set(result.codes())


def test_rejects_continue_on_error() -> None:
    text = """
jobs:
  qualification:
    steps:
      - name: tests
        continue-on-error: true
        run: python3 -m pytest -q test/ci/test_external_agent_release_validator.py
"""
    result = validate_release(workflow=text)
    assert CONTINUE_ON_ERROR in _codes(result)
    assert result.ok is False


def test_rejects_shell_success_masking() -> None:
    text = """
jobs:
  qualification:
    steps:
      - run: |
          python3 -m pytest -q test/api/test_external_agent_handoff_contracts.py || true
"""
    result = validate_release(workflow=text)
    assert SHELL_SUCCESS_MASKING in _codes(result)
    assert CONTINUE_ON_ERROR not in _codes(result)


def test_rejects_skipped_required_tests() -> None:
    report = _passing_report(
        passed=1,
        skipped=1,
        tests=[
            dict(PASSING_TESTS[0]),
            {
                "nodeid": "test/api/test_external_agent_codex_adapter.py::test_ok",
                "outcome": "skipped",
                "reason": "not now",
            },
        ],
    )
    result = validate_pytest_report(
        report,
        required=("test/api/test_external_agent_codex_adapter.py",),
    )
    assert SKIPPED_REQUIRED in _codes(result)
    assert result.ok is False


def test_rejects_xfailed_required_tests() -> None:
    report = _passing_report(
        passed=1,
        xfailed=1,
        tests=[
            dict(PASSING_TESTS[0]),
            {
                "nodeid": "test/api/test_external_agent_codex_adapter.py::test_ok",
                "outcome": "xfailed",
                "wasxfail": True,
            },
        ],
    )
    result = validate_pytest_report(
        report,
        required=("test/api/test_external_agent_codex_adapter.py",),
    )
    assert XFAIL_REQUIRED in _codes(result)
    assert result.ok is False


def test_rejects_unavailable_as_passed() -> None:
    report = _passing_report(
        tests=[
            {
                "nodeid": "test/api/test_external_agent_handoff_contracts.py::test_ok",
                "outcome": "passed",
                "unavailable": True,
            },
            dict(PASSING_TESTS[1]),
        ]
    )
    result = validate_pytest_report(report)
    assert UNAVAILABLE_AS_PASSED in _codes(result)


def test_rejects_simulated_as_live() -> None:
    report = _passing_report(simulated=True, live=True, mode="simulated")
    result = validate_pytest_report(report)
    assert SIMULATED_AS_LIVE in _codes(result)


def test_rejects_stale_historical_counts() -> None:
    report = _passing_report(historical=True, current=False, source="historical")
    result = validate_pytest_report(report)
    assert STALE_HISTORICAL in _codes(result)


def test_binds_collected_and_passed_populations() -> None:
    mismatch = _passing_report(collected=3, passed=2)
    result = validate_pytest_report(mismatch)
    assert POPULATION_MISMATCH in _codes(result)
    assert result.ok is False
    assert result.populations is not None
    assert result.populations.collected == 3
    assert result.populations.passed == 2

    bound = validate_pytest_report(_passing_report())
    assert bound.ok is True
    assert bound.populations is not None
    assert bound.populations.collected == bound.populations.passed == 2
    assert bound.populations.skipped == bound.populations.failed == 0
    assert bound.populations.xfailed == bound.populations.xpassed == 0


def test_current_workflow_is_fail_closed() -> None:
    text = DEFAULT_WORKFLOW.read_text(encoding="utf-8")
    assert validate_workflow_text(text) == ()
    result = validate_release(workflow=DEFAULT_WORKFLOW)
    assert result.ok is True
    assert _codes(result) == set()


def test_cli_emits_bound_receipt(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        "\n".join(
            [
                "jobs:",
                "  qualification:",
                "    steps:",
                "      - run: |",
                "          set -euo pipefail",
                "          python3 -m pytest -q \\",
                "            test/api/test_external_agent_handoff_contracts.py \\",
                "            test/api/test_external_agent_codex_adapter.py",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_passing_report()), encoding="utf-8")
    assert main(["--workflow", str(workflow), "--pytest-report", str(report_path)]) == 0
    assert (
        main(
            [
                "--workflow",
                str(workflow),
                "--pytest-report",
                str(report_path),
                "--terminal",
            ]
        )
        == 0
    )
