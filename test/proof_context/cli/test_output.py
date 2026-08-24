"""PCCE-043 contract tests for stable CLI machine output and human reports."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.proof_context.cli.output import (
    EXIT_CODES,
    IDENTITY_FIELDS,
    MAX_COLLECTION_ITEMS,
    MAX_TEXT_LENGTH,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    command_result,
    exit_code_for,
    serialize,
)
from ipfs_accelerate_py.proof_context.cli.report import (
    REPORT_SCHEMA_VERSION,
    REPORT_SECTIONS,
    render_human_patch_report,
)


def _result(status: str = "succeeded", **kwargs: object) -> dict[str, object]:
    return command_result(
        status=status,
        command="seal",
        correlation_id="corr-pcce-043",
        trace_id="trace-pcce-043",
        identities={
            "repository_id": "repository:sha256:abc",
            "task_id": "PCCE-043",
            "run_id": "run:pcce-043",
            "patch_id": "patch:pcce-043",
        },
        artifact_cids=["bafyreia-output", "bafyreia-receipt"],
        details=kwargs.pop("details", {}),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("succeeded", 0),
        ("failed", 1),
        ("invalid", 2),
        ("rejected", 3),
        ("simulated", 4),
        ("unavailable", 5),
        ("stale", 6),
        ("unknown_status", 1),
    ],
)
def test_exit_code_matrix_is_closed_and_nonzero_for_non_success(
    status: str, expected: int
) -> None:
    assert exit_code_for(status) == expected
    assert exit_code_for("succeeded", provenance="simulated") == EXIT_CODES["simulated"]
    assert exit_code_for("succeeded", provenance="replayed") == EXIT_CODES["failed"]


def test_every_result_has_version_status_exit_trace_identities_and_artifact_cids() -> None:
    result = _result()
    assert set(REQUIRED_FIELDS).issubset(result)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "succeeded"
    assert result["exit_code"] == 0
    assert result["trace_id"] == "trace-pcce-043"
    assert result["correlation_id"] == "corr-pcce-043"
    assert tuple(result["identities"]) == IDENTITY_FIELDS
    assert result["artifact_cids"] == ["bafyreia-output", "bafyreia-receipt"]
    encoded = serialize(result)
    assert encoded.endswith("\n")
    assert json.loads(encoded) == result


def test_missing_identity_evidence_is_explicit_not_synthesized() -> None:
    result = command_result(
        status="unavailable",
        correlation_id="corr-unavailable",
        identities={"task_id": "PCCE-043"},
    )
    assert result["exit_code"] == 5
    assert result["identities"] == {
        "repository_id": None,
        "task_id": "PCCE-043",
        "run_id": None,
        "patch_id": None,
    }
    assert result["artifact_cids"] == []


def test_secret_redaction_and_output_bounds_apply_recursively() -> None:
    result = _result(
        details={
            "api_token": "super-secret",
            "diagnostic": "Bearer credentials-should-not-escape",
            "long": "x" * (MAX_TEXT_LENGTH + 10),
            "items": list(range(MAX_COLLECTION_ITEMS + 2)),
        },
    )
    details = result["details"]
    assert details["api_token"] == "[REDACTED]"
    assert "credentials-should-not-escape" not in details["diagnostic"]
    assert details["long"].endswith("…[truncated]")
    assert details["items"][-1] == "[truncated: additional items omitted]"
    assert "super-secret" not in serialize(result)


def test_serialize_rejects_incomplete_envelope() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        serialize({"status": "succeeded"})


@pytest.mark.parametrize("status", ["succeeded", "failed", "unavailable", "simulated"])
def test_human_report_covers_every_required_section_for_all_outcomes(status: str) -> None:
    report = render_human_patch_report(_result(status))
    assert REPORT_SCHEMA_VERSION in report
    assert f"Status: {status}" in report
    assert "Trace / correlation: trace-pcce-043 / corr-pcce-043" in report
    for title, _ in REPORT_SECTIONS:
        assert f"{title}:" in report
    assert "missing_evidence" in report


def test_human_report_labels_cost_estimate_and_baseline_honestly() -> None:
    report = render_human_patch_report(
        _result(
            details={
                "costs": {"amount": "12 tokens", "basis": "prior run", "label": "estimated"},
                "revision": "base:abc -> candidate:def",
            }
        )
    )
    assert "Costs: estimated: 12 tokens; basis: prior run" in report
    assert "Revision: base:abc -> candidate:def" in report
    assert "Task: PCCE-043" in report


def test_human_report_redacts_secrets_and_does_not_expand_logs() -> None:
    report = render_human_patch_report(
        _result(details={"routing": "token=do-not-print", "changes": "x" * (MAX_TEXT_LENGTH + 5)})
    )
    assert "do-not-print" not in report
    assert "[REDACTED]" in report
    assert "…[truncated]" in report
