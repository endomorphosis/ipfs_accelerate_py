"""Candidate JSON costs cannot be reported as retrieved source or hidden."""
import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import evidence, _collect  # noqa: F401


def upgrade(evidence):
    audit = evidence[3]
    audit["schema"] = "supervisor-worker-source-session@2"
    for row in audit["requests"]:
        row["candidate_context_page"] = row["status"] == "ok"
        row["returned_candidate_context_bytes"] = row["returned_source_bytes"]
        row["returned_source_bytes"] = 0
    return audit


def test_candidate_bytes_are_visible_and_excluded_from_source_bytes(evidence):
    upgrade(evidence)
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["counts"]["prepared_source_bytes"] == result["counts"]["delivered_source_bytes"] == 0
    assert result["counts"]["response_bytes"] == 2281
    assert result["candidate_context_counts"] == dict(prepared_pages=1, delivered_pages=1,
        prepared_bytes=40, delivered_bytes=40)
    assert result["total_token_savings_measured"] is False


@pytest.mark.parametrize("field,value", [("returned_candidate_context_bytes", True),
    ("returned_candidate_context_bytes", -1), ("returned_candidate_context_bytes", 9000),
    ("candidate_context_page", False), ("candidate_context_page", 1),
    ("returned_source_bytes", 12), ("status", "peer_refused")])
def test_invalid_candidate_accounting_refuses(evidence, field, value):
    audit = upgrade(evidence)
    row = next(row for row in audit["requests"] if row["status"] == "ok")
    row[field] = value
    evidence[5]()
    assert _collect(evidence)["status"] == "unavailable"


def test_legacy_audit_does_not_invent_candidate_usage(evidence):
    assert _collect(evidence)["candidate_context_counts"] is None
