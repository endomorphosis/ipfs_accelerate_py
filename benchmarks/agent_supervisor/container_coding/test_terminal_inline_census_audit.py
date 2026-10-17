"""Actual retained-input audit with pure compiled capsules and no providers."""
import json

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import compiled  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_observation as inline


def test_real_bounded_audit_child_retains_unknown_for_unsupported_native_carrier(compiled):
    result = audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"])
    assert result["all_observed_coding_inputs_verified"] is True
    assert result["provider_calls"] == 0
    census = result["rendered_input_census"]["observations"][0]
    value = inline.validate_inline_sources(census["inline_sources"],
        model_prompt_sha256=compiled["receipts"][0]["model_prompt_sha256"],
        model_prompt_bytes=compiled["receipts"][0]["model_prompt_bytes"],
        invocation_id_sha256=census["invocation_id_sha256"])
    assert value["status"] == "unavailable"
    assert value["source_count"] is value["logical_source_bytes"] is None
    assert compiled["query"] not in json.dumps(result)


def test_optional_observer_failure_cannot_demote_exact_input_audit(compiled, monkeypatch):
    def fail(**kwargs):
        raise ValueError("PRIVATE SOURCE PATH must not escape")
    monkeypatch.setattr(inline, "observe_inline_sources", fail)
    result = audit.audit_terminal_context(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"])
    assert result["all_observed_coding_inputs_verified"] is True
    census = result["rendered_input_census"]["observations"][0]
    assert "inline_sources" not in census
    assert census["byte_partition_reconciles"] is True
    assert "PRIVATE SOURCE PATH" not in json.dumps(result)


def test_changed_prompt_receipt_does_not_acquire_source_observation(compiled):
    compiled["receipts"][0]["model_prompt_sha256"] = "0" * 64
    result = audit.audit_terminal_context(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"])
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["rendered_input_census"]["observations"] == []
