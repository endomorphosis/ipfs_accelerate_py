"""Pure authored counters preserve costs when typed lineage coverage is missing.

Trusted observer envelopes here are fixtures, not native verification receipts
or provider measurements. No Harbor/Terminal imports or execution are needed.
"""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_usage import (
    collect_provider_usage_observation,
    semantic_decision_accounting_inputs,
)
from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
    COMPLETION_BASIS,
    reconcile_router_cumulative_usage,
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _case():
    thread = "authored-native-session-00000001"
    native = {
        "schema": "router-implementation-invocation@1", "invocation_id": "invocation-authored",
        "provider": "codex_cli", "model": "fixture-model", "workspace": "/fixture/candidate",
        "phase": "coding", "purpose": "coding", "status": "provider_returned",
        "model_prompt_sha256": "1" * 64,
        "usage": {"thread_id": thread, "model_id": "fixture-model"},
        "native_rollout_usage": {
            "schema": "native-codex-cumulative-usage@1", "thread_id": thread,
            "rollout_sha256": sha256(b"authored typed-usage counter").hexdigest(),
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                      "cached_input_tokens": 50, "reasoning_output_tokens": 10},
            "usage_available": True, "observed_token_count_records": 1,
            "cache_included_in_input": True, "task_complete_observed": True,
            "usage_lifecycle_complete": True, "completion_basis": COMPLETION_BASIS,
            "lifecycle_malformed_records": 0,
        },
    }
    owner = {
        "schema": "supervisor-semantic-decision-owner-binding@1", "binding_id": "owner-binding-authored",
        "task_id": "fixture-task", "native_prompt_sha256": "a" * 64, "native_prompt_bytes": 321,
        "parent_artifact": ".runtime/query/worker-context.json", "parent_sha256": "b" * 64,
        "policy_sha256": "c" * 64, "launch_policy_artifact": "/source/.runtime/policy.json",
        "launch_policy_sha256": "f" * 64, "nomination_artifact": "/owner/origin.json",
        "nomination_sha256": "d" * 64, "provider_called": False, "completion_authority": False,
        "typed_nomination": {
            "schema": "supervisor-semantic-typed-nomination-binding@1",
            "derived_nomination_artifact": "/source/.runtime/derived.json",
            "derived_nomination_sha256": "e" * 64,
            "lineage_artifact": "/source/.runtime/lineage.json", "lineage_sha256": "2" * 64,
        },
    }
    provider = {
        "schema": "router-semantic-decision-provider@1", "call_id": "call-authored",
        "session_id": "dialogue-authored", "provider": native["provider"], "model": native["model"],
        "workspace": native["workspace"], "model_prompt_sha256": native["model_prompt_sha256"],
        "provider_called": True, "status": "provider_returned", "native_invocation": deepcopy(native),
    }
    dialogue = {
        "schema": "supervisor-semantic-decision-dialogue@1", "session_id": "dialogue-authored",
        "provider": native["provider"], "model": native["model"], "workspace": native["workspace"],
        "task_id": owner["task_id"], "parent_sha256": owner["parent_sha256"],
        "policy_sha256": owner["policy_sha256"], "status": "returned", "invocation_attempt_count": 1,
        "attempts": [{"call_id": "call-authored", "model_prompt_sha256": native["model_prompt_sha256"],
                      "provider_receipt": provider, "provider_receipt_bound": True, "status": "provider_returned"}],
        "worker_launch": {
            "original_native_prompt_sha256": owner["native_prompt_sha256"],
            "original_native_prompt_bytes": owner["native_prompt_bytes"],
            "launch_policy_artifact": owner["launch_policy_artifact"],
            "launch_policy_sha256": owner["launch_policy_sha256"],
        },
    }
    verification = {
        "schema": "supervisor-semantic-typed-nomination-verification@1", "task_id": owner["task_id"],
        "origin_nomination_sha256": owner["nomination_sha256"],
        "derived_nomination_sha256": owner["typed_nomination"]["derived_nomination_sha256"],
        "lineage_sha256": owner["typed_nomination"]["lineage_sha256"],
        "native_prompt_sha256": owner["native_prompt_sha256"], "native_prompt_bytes": owner["native_prompt_bytes"],
        "parent_artifact": owner["parent_artifact"], "parent_sha256": owner["parent_sha256"],
        "policy_sha256": owner["policy_sha256"], "launch_policy_sha256": owner["launch_policy_sha256"],
        "owner_binding_sha256": sha256(_encoded(owner)).hexdigest(),
        "source_freshness_revalidated": False, "provider_calls": 0,
        "execution_authority": False, "proof_authority": False, "completion_authority": False,
    }
    report = {
        "provider_invocations": [native], "semantic_decision_dialogues": [dialogue],
        "semantic_decision_owner_bindings": [owner], "semantic_decision_dialogue_expected": True,
        "semantic_decision_owner_binding_expected": True, "semantic_decision_typed_nomination_expected": True,
        "semantic_decision_expected_task_id": owner["task_id"],
        "semantic_decision_nomination_sha256": owner["nomination_sha256"],
        "context_input_audit": {"schema": "terminal-final-context-audit@1", "provider_calls": 0,
            "raw_prompts_exported": False, "completion_authority": False,
            "semantic_decision_inputs": {
                "schema": "terminal-semantic-decision-input-population-audit@1", "provider_calls": 0,
                "raw_prompts_exported": False, "completion_authority": False,
                "typed_nomination_verifications": [verification],
            }},
    }
    return report


def _project(report):
    before = deepcopy(report)
    rows, coverage = semantic_decision_accounting_inputs(report)
    assert report == before
    totals = reconcile_router_cumulative_usage(rows)
    assert totals["native_sessions_observed"] == 1
    assert totals["known_provider_token_subtotal"] == totals["provider_total_tokens"] == 120
    assert len(rows) == 1
    return coverage


def _gap(report, reason=None):
    coverage = _project(report)
    assert coverage["authoritative_usage_coverage_complete"] is False
    assert coverage["owner_binding_coverage_complete"] is False
    assert coverage["provider_call_count_complete"] is False
    assert coverage["all_attempts_receipted"] is False
    if reason is not None:
        assert reason in {item["reason"] for item in coverage["issues"]}
    return coverage


def _verification(report):
    return report["context_input_audit"]["semantic_decision_inputs"]["typed_nomination_verifications"][0]


def test_verified_origin_to_derived_lineage_adds_no_native_charge():
    report = _case()
    coverage = _project(report)
    assert coverage["owner_binding_coverage_complete"] is True
    assert coverage["authoritative_usage_coverage_complete"] is True
    assert coverage["typed_nomination_verified_owner_bindings_observed"] == 1
    assert coverage["typed_nomination_verifications_observed"] == 1
    assert report["semantic_decision_nomination_sha256"] != _verification(report)["derived_nomination_sha256"]


@pytest.mark.parametrize("field", ["task_id", "origin_nomination_sha256", "derived_nomination_sha256",
    "lineage_sha256", "native_prompt_sha256", "native_prompt_bytes", "parent_artifact",
    "parent_sha256", "policy_sha256", "launch_policy_sha256", "owner_binding_sha256"])
def test_foreign_verified_binding_keeps_native_subtotal_but_cannot_complete_coverage(field):
    report = _case()
    value = _verification(report)
    value[field] = (322 if field == "native_prompt_bytes" else ".runtime/foreign.json"
                    if field == "parent_artifact" else "foreign-task" if field == "task_id" else "0" * 64)
    _gap(report, "typed_nomination_verification_missing_or_ambiguous")


@pytest.mark.parametrize("field,value", [
    ("provider_calls", False), ("provider_calls", 1), ("source_freshness_revalidated", True),
    ("execution_authority", True), ("proof_authority", True), ("completion_authority", True),
    ("native_prompt_bytes", True), ("parent_artifact", "../foreign.json"),
])
def test_verification_summary_requires_exact_historical_shapes_and_no_authority(field, value):
    report = _case()
    _verification(report)[field] = value
    _gap(report, "invalid_typed_nomination_verification")


@pytest.mark.parametrize("envelope,field,value", [
    ("outer", "schema", "forged-audit@1"), ("outer", "provider_calls", False),
    ("outer", "raw_prompts_exported", True), ("outer", "completion_authority", True),
    ("inner", "schema", "forged-population@1"), ("inner", "provider_calls", False),
    ("inner", "raw_prompts_exported", True), ("inner", "completion_authority", True),
])
def test_summary_outside_exact_context_observer_envelope_is_not_trusted(envelope, field, value):
    report = _case()
    target = report["context_input_audit"]
    if envelope == "inner":
        target = target["semantic_decision_inputs"]
    target[field] = value
    _gap(report, "typed_nomination_verification_population_unavailable")


def test_printed_verification_marker_cannot_fill_missing_trusted_observation():
    report = _case()
    printed = deepcopy(_verification(report))
    del report["context_input_audit"]
    assert collect_provider_usage_observation(report, printed) is False
    _gap(report, "typed_nomination_verification_population_unavailable")


def test_changed_owner_path_cannot_reuse_summary_bound_to_previous_full_owner_receipt():
    report = _case()
    report["semantic_decision_owner_bindings"][0]["typed_nomination"]["lineage_artifact"] = "/foreign/lineage.json"
    _gap(report, "typed_nomination_verification_missing_or_ambiguous")


def test_missing_sidecar_binding_or_observer_summary_cannot_hide_complete_native_subset():
    report = _case()
    report["semantic_decision_owner_bindings"][0].pop("typed_nomination")
    _gap(report, "expected_typed_nomination_binding_missing")
    report = _case()
    report["context_input_audit"]["semantic_decision_inputs"]["typed_nomination_verifications"] = []
    _gap(report, "typed_nomination_verification_missing_or_ambiguous")


def test_duplicate_exact_summaries_and_owner_rows_are_one_observation():
    report = _case()
    report["context_input_audit"]["semantic_decision_inputs"]["typed_nomination_verifications"].append(deepcopy(_verification(report)))
    report["semantic_decision_owner_bindings"].append(deepcopy(report["semantic_decision_owner_bindings"][0]))
    coverage = _project(report)
    assert coverage["authoritative_usage_coverage_complete"] is True
    assert coverage["owner_bindings_observed"] == coverage["typed_nomination_verifications_observed"] == 1


def test_unmatched_changed_summary_is_retained_as_coverage_gap():
    report = _case()
    changed = deepcopy(_verification(report))
    changed["owner_binding_sha256"] = "0" * 64
    report["context_input_audit"]["semantic_decision_inputs"]["typed_nomination_verifications"].append(changed)
    _gap(report, "typed_nomination_verification_without_owner_binding")


def test_orphan_typed_owner_receipt_does_not_fabricate_zero_worker_calls():
    report = _case()
    report.pop("semantic_decision_dialogues")
    coverage = _gap(report, "expected_owner_bound_dialogue_missing")
    assert coverage["owner_bindings_without_dialogue_count"] == 1


@pytest.mark.parametrize("mode", [False, "enabled", 1])
def test_optout_and_nonliteral_modes_cannot_accept_typed_binding(mode):
    report = _case()
    report["semantic_decision_typed_nomination_expected"] = mode
    _gap(report, "unexpected_typed_nomination_binding" if mode is False else "invalid_expected_typed_nomination_mode")


def test_legacy_owner_without_typed_mode_or_extension_retains_existing_coverage():
    report = _case()
    report.pop("semantic_decision_typed_nomination_expected")
    report.pop("context_input_audit")
    report["semantic_decision_owner_bindings"][0].pop("typed_nomination")
    coverage = _project(report)
    assert coverage["authoritative_usage_coverage_complete"] is True
    assert "typed_nomination_expected" not in coverage
