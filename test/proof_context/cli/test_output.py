"""PCCE-043 semantic and end-to-end CLI output contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.proof_context.cli.output import (
    EXIT_CODES,
    IDENTITY_FIELDS,
    MAX_COLLECTION_ITEMS,
    MAX_IDENTITY_LENGTH,
    MAX_TEXT_LENGTH,
    OMITTED_UNBOUNDED,
    PROVENANCES,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    STATUSES,
    admit_result,
    command_result,
    exit_code_for,
    serialize,
)
from ipfs_accelerate_py.proof_context.cli.report import (
    REPORT_SCHEMA_VERSION,
    REPORT_SECTIONS,
    render_human_patch_report,
)
from ipfs_accelerate_py.proof_context.errors import (
    REDACTED,
    MalformedError,
    PseudoCidError,
    UnknownFieldError,
)
from ipfs_accelerate_py.proof_context.results import mint_result_cid
from jsonschema import Draft202012Validator

VALID_CID = mint_result_cid({"fixture": "pcce-043-artifact"})
REPOSITORY_STATE_CID = mint_result_cid({"fixture": "pcce-043-state"})


def _schema_path() -> Path:
    relative = Path("artifacts/proof_carrying_context_engine/cli/output_schema.json")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    raise AssertionError(f"PCCE-043 output schema not found: {relative}")


def _schema() -> dict[str, Any]:
    value = json.loads(_schema_path().read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    Draft202012Validator.check_schema(value)
    return value


def _identities(*, patch_id: str | None = "patch:pcce-043") -> dict[str, Any]:
    return {
        "repository_id": "repository:pcce-043",
        "repository_state_cid": REPOSITORY_STATE_CID,
        "task_id": "PCCE-043",
        "run_id": "run:pcce-043",
        "trace_id": "trace:pcce-043",
        "patch_id": patch_id,
        "contract_version": "0.1",
    }


def _result(
    status: str = "succeeded",
    *,
    provenance: str | None = "live",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return command_result(
        status=status,
        command="status",
        correlation_id="corr:pcce-043",
        trace_id="trace:pcce-043",
        identities=_identities(),
        artifact_cids=[VALID_CID],
        provenance=provenance,
        details=details,
        error=None if status == "succeeded" else status,
    )


def test_exit_code_matrix_is_exactly_the_frozen_taxonomy() -> None:
    assert set(EXIT_CODES) == set(STATUSES)
    assert "failed" not in EXIT_CODES
    for status in STATUSES:
        code = exit_code_for(status, provenance="live")
        assert code == EXIT_CODES[status]
        if status == "succeeded":
            assert code == 0
        else:
            assert code != 0
    assert exit_code_for("succeeded") == 0
    assert exit_code_for("succeeded", provenance="replayed") == 1
    assert exit_code_for("succeeded", provenance="simulated") == EXIT_CODES["simulated"]
    assert exit_code_for("succeeded", provenance=None) == 1
    with pytest.raises(UnknownFieldError):
        exit_code_for("unknown_status")
    with pytest.raises(UnknownFieldError):
        exit_code_for("succeeded", provenance="shadow")


def test_every_result_has_required_identity_and_artifact_fields() -> None:
    result = _result()
    assert tuple(field for field in REQUIRED_FIELDS if field not in result) == ()
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "succeeded"
    assert result["exit_code"] == 0
    assert result["trace_id"] == "trace:pcce-043"
    assert result["correlation_id"] == "corr:pcce-043"
    assert all(field in result["identities"] for field in IDENTITY_FIELDS)
    assert result["identities"]["repository_state_cid"] == REPOSITORY_STATE_CID
    assert result["artifact_cids"] == [VALID_CID]
    encoded = serialize(result)
    assert encoded.endswith("\n")
    assert json.loads(encoded) == result


def test_missing_identity_evidence_is_explicit_and_schema_valid() -> None:
    result = command_result(
        status="unavailable",
        correlation_id="corr:missing-identities",
        identities={"task_id": "PCCE-043"},
        provenance="live",
    )
    assert result["trace_id"] == "corr:missing-identities"
    assert result["correlation_id"] == "corr:missing-identities"
    assert result["identities"] == {
        "repository_id": None,
        "task_id": "PCCE-043",
        "run_id": None,
        "patch_id": None,
    }
    assert result["artifact_cids"] == []
    Draft202012Validator(_schema()).validate(result)
    with pytest.raises(MalformedError, match="correlation_id is required"):
        command_result(
            status="unavailable",
            correlation_id="",
            identities={"task_id": "PCCE-043"},
            provenance="live",
        )


def test_secret_redaction_uses_canonical_vectors_and_bounds_output() -> None:
    result = _result(
        details={
            "cookie": "cookie-test-fixture",
            "session": "session-test-fixture",
            "private_key": "private-key-test-fixture",
            "refresh_token": "refresh-test-fixture",
            "diagnostic": (
                "sk-testfixtureabcdefgh ghp_testfixtureabcdefgh xoxb-testfixtureabcdefgh"
            ),
            "source": "print('must not escape')",
            "long": "x" * (MAX_TEXT_LENGTH + 100),
            "items": list(range(MAX_COLLECTION_ITEMS + 2)),
        }
    )
    details = result["details"]
    for key in ("cookie", "session", "private_key", "refresh_token"):
        assert details[key] == REDACTED
    assert "testfixtureabcdefgh" not in details["diagnostic"]
    assert details["source"] == OMITTED_UNBOUNDED
    assert details["long"].endswith("…")
    assert len(details["long"]) <= MAX_TEXT_LENGTH
    assert details["items"][-1] == "[truncated: additional items omitted]"
    rendered = serialize(result)
    for secret in (
        "cookie-test-fixture",
        "session-test-fixture",
        "private-key-test-fixture",
        "refresh-test-fixture",
        "testfixtureabcdefgh",
        "must not escape",
    ):
        assert secret not in rendered


def test_identity_and_cid_admission_never_rewrites_claimed_identity() -> None:
    assert _result()["artifact_cids"][0] == VALID_CID
    with pytest.raises(PseudoCidError):
        command_result(
            status="succeeded",
            correlation_id="corr",
            identities=_identities(),
            artifact_cids=["sha256:deadbeef"],
        )
    with pytest.raises(PseudoCidError):
        command_result(
            status="succeeded",
            correlation_id="corr",
            identities={**_identities(), "repository_state_cid": "bafyreia-short"},
            artifact_cids=[VALID_CID],
        )
    with pytest.raises(MalformedError):
        command_result(
            status="succeeded",
            correlation_id="c" * (MAX_IDENTITY_LENGTH + 1),
            identities=_identities(),
            artifact_cids=[VALID_CID],
        )
    with pytest.raises(MalformedError):
        command_result(
            status="succeeded",
            correlation_id="sk-testfixtureabcdefgh",
            identities=_identities(),
            artifact_cids=[VALID_CID],
        )


def test_serialize_rejects_schema_drift_unknown_fields_and_exit_mismatch() -> None:
    with pytest.raises(MalformedError, match="missing required fields"):
        serialize({"status": "succeeded"})
    result = _result()
    with pytest.raises(UnknownFieldError):
        serialize({**result, "surprise": True})
    with pytest.raises(MalformedError, match="exit_code"):
        serialize({**result, "exit_code": 1})
    assert admit_result(result) == result


def test_output_schema_validates_direct_and_real_cli_envelopes() -> None:
    schema = _schema()
    validator = Draft202012Validator(schema)
    validator.validate(_result())
    validator.validate(
        command_result(
            status="verification_failed",
            correlation_id="corr",
            identities=_identities(),
            artifact_cids=[VALID_CID],
            error="verification_failed",
        )
    )
    assert schema["properties"]["status"]["enum"] == list(STATUSES)
    assert schema["properties"]["provenance"]["enum"] == [*PROVENANCES, None]
    assert schema["required"] == list(REQUIRED_FIELDS)


@pytest.mark.parametrize(
    ("status", "provenance"),
    [
        ("succeeded", "live"),
        ("verification_failed", "live"),
        ("unavailable", "live"),
        ("simulated", "simulated"),
    ],
)
def test_human_reports_cover_every_section_for_canonical_outcomes(
    status: str, provenance: str
) -> None:
    report = render_human_patch_report(_result(status, provenance=provenance))
    assert REPORT_SCHEMA_VERSION in report
    assert f"Status: {status}" in report
    assert "Trace / correlation: trace:pcce-043 / corr:pcce-043" in report
    assert f"Artifact CIDs: {VALID_CID}" in report
    for title, _ in REPORT_SECTIONS:
        assert f"{title}:" in report
    assert "Receipts: missing_evidence" in report


def test_human_report_labels_cost_estimate_baseline_and_missing_evidence() -> None:
    report = render_human_patch_report(
        _result(
            details={
                "costs": {
                    "amount": "12 tokens",
                    "basis": "prior run",
                    "label": "estimated",
                },
                "revision": "base:abc -> candidate:def",
            }
        )
    )
    assert "Costs: estimated: 12 tokens; basis: prior run" in report
    assert "Revision: base:abc -> candidate:def" in report
    assert "Receipts: missing_evidence" in report


def test_repeated_machine_and_human_rendering_is_deterministic() -> None:
    result = _result(details={"routing": {"z": 1, "a": 2}})
    assert serialize(result) == serialize(result)
    assert render_human_patch_report(result) == render_human_patch_report(result)
