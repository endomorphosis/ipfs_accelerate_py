"""Frozen benchmark recipe must join the actual shared owner session pins."""
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_source_view_observation as collector
from benchmarks.agent_supervisor.container_coding import terminal_source_view_policy as selection
from benchmarks.agent_supervisor.container_coding.test_terminal_source_view_observation import evidence, _mixed_pages, _collect  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from ipfs_accelerate_py.agent_supervisor.runtime.source_view_policy import SourceViewPolicy


def selected_session(evidence):
    _mixed_pages(evidence, shared=True)
    report, audit, row = evidence[2:5]
    policy = SourceViewPolicy.from_mapping(profile="python-comments@1",
        protected_lines={b"mod.py".hex(): [2]})
    report.update(arm="full", **selection.source_view_observation(policy.profile, policy.to_dict()))
    binding = report["source_view_policy_binding_sha256"]
    audit["session"]["view_policy_binding_sha256"] = binding
    audit["descriptor_sha256"] = collector._sha(client.canonical_json(audit["session"]))
    row["source_view_retrieval"].update(view_policy_binding_sha256=binding,
        descriptor_sha256=audit["descriptor_sha256"])
    evidence[5]()
    return policy


def test_selected_recipe_matches_actual_descriptor_and_retains_typed_counts(evidence):
    selected_session(evidence)
    result = _collect(evidence)
    assert result["status"] == "observed"
    assert result["source_view_policy_sha256"] == evidence[2]["source_view_policy_sha256"]
    assert result["source_view_policy_binding_sha256"] == evidence[2]["source_view_policy_binding_sha256"]
    assert result["discovery_counts"] == dict(prepared_pages=2, delivered_pages=1, prepared_rows=5, delivered_rows=2)
    assert result["counts"]["prepared_source_bytes"] > 0
    assert "source_view_policy" not in result


@pytest.mark.parametrize("change", ["recipe", "config_hash", "binding_hash", "missing_config_hash", "missing_binding_hash", "wrong_profile", "disabled", "nonfull", "missing_arm", "extra_field", "missing_recipe", "foreign_session", "legacy_downgrade"])
def test_frozen_policy_or_actual_session_tampering_cannot_report_observed(evidence, change):
    selected_session(evidence)
    report, audit, row = evidence[2:5]
    if change == "recipe": report["source_view_policy"]["protected_lines"] = {}
    elif change == "config_hash": report["source_view_policy_sha256"] = "0" * 64
    elif change == "binding_hash": report["source_view_policy_binding_sha256"] = "0" * 64
    elif change == "missing_config_hash": report.pop("source_view_policy_sha256")
    elif change == "missing_binding_hash": report.pop("source_view_policy_binding_sha256")
    elif change == "wrong_profile": report["source_view_policy"]["profile"] = "identity@1"
    elif change == "disabled": report["source_view_profile"] = "disabled"
    elif change == "nonfull": report["arm"] = "no-index"
    elif change == "missing_arm": report.pop("arm")
    elif change == "extra_field": report["source_view_policy"]["authority"] = True
    elif change == "missing_recipe": report.pop("source_view_policy")
    elif change == "foreign_session":
        audit["session"]["view_policy_binding_sha256"] = "7" * 64
        audit["descriptor_sha256"] = collector._sha(client.canonical_json(audit["session"]))
        row["source_view_retrieval"].update(view_policy_binding_sha256="7" * 64,
            descriptor_sha256=audit["descriptor_sha256"])
    else:
        audit["session"]["schema"] = client.DESCRIPTOR_SCHEMA
        for key in collector.SHARED_PIN_FIELDS: audit["session"].pop(key)
        audit["schema"] = "supervisor-worker-source-session@2"
        audit["descriptor_sha256"] = collector._sha(client.canonical_json(audit["session"]))
        row["source_view_retrieval"].update(schema="worker-source-dispatch-observation@1",
            descriptor_sha256=audit["descriptor_sha256"])
        for key in collector.SHARED_PIN_FIELDS: row["source_view_retrieval"].pop(key)
    evidence[5]()
    result = _collect(evidence)
    assert result["status"] == "unavailable" and result["counts"] is None


def test_missing_selected_session_remains_unknown_with_frozen_recipe(monkeypatch):
    policy = SourceViewPolicy()
    monkeypatch.setattr(collector, "_open_directory", lambda *a: pytest.fail("missing session reached private IO"))
    result = collector.collect(parent=Path("/absent"), scope=None, report=dict(arm="full",
        provider_invocations=[], **selection.source_view_observation(policy.profile, policy.to_dict())))
    assert result["status"] == "unavailable" and result["reason"] == "selected_session_receipt_unavailable"
    assert result["counts"] is None
