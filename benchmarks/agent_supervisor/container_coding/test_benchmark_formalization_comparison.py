"""Bound historical formalization facts in the retained comparison artifact."""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
from benchmarks.agent_supervisor.container_coding.test_benchmark_comparison import _receipt, _baseline


@pytest.fixture
def observed():
    context = dict(schema="terminal-historical-source-formalization-observation@1",
        task_id="TASK-1", task_cid="cid:task", expected_context_bundle_sha256="a" * 64,
        dispatched_context_bundle_sha256="b" * 64, worker_evidence_sha256="c" * 64,
        chunks=2, source_files=3, contexts=[dict(kind=kind, receipt_sha256="d" * 64,
            summary_sha256="e" * 64, source_hashes_sha256="f" * 64) for kind in (
                "source384-candidate-context", "numerical-formalization-context")],
        historical_receipt_bindings_verified=True, current_sources_checked=False,
        neural_inference_replayed=False, proof_authority=False, formalization_authority=False,
        execution_authority=False, completion_authority=False, source_semantics_verified=False,
        whole_program_semantics_verified=False, source_obligations_discharged=0)
    return dict(status="verified_all_observed_model_inputs", any_native_input_verified=True,
        any_model_input_verified=True, all_observed_coding_inputs_verified=True,
        capsules=[dict(capsule_id="cid:capsule", native_prompt_sha256="1" * 64,
            native_prompt_bytes=12000, all_context_checks_passed=True,
            source_formalization_context=context, raw_candidate="MUST NOT APPEAR")],
        matches=[dict(capsule_id="cid:capsule", invocation_id="call-1", native_prompt_sha256="1" * 64,
            native_prompt_bytes=12000, model_input_verified=True,
            model_input_checks=dict(source_formalization_dispatch=True, model_prompt_sha256=True))],
        invocations=[dict(invocation_id="call-1", native_input_verified=True, model_input_verified=True)])


def project(value):
    return comparison._audit_observation({"context_input_audit": value})["source_formalization"]


def test_compact_reported_delivery_reaches_retained_comparison_without_candidate_text(observed, tmp_path):
    before = deepcopy(observed)
    receipt = _receipt(tmp_path, "formalization.json", comparison.SUPERVISOR,
        [dict(trial="formalization", supervisor={"context_input_audit": observed})], "full")
    result = comparison.collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    exported = result["rows"][1]["context_input_audit"]["source_formalization"]
    assert len(exported) == 1 and exported[0]["reported_model_input_verified"] is True
    assert exported[0]["matched_invocation_ids"] == ["call-1"]
    assert exported[0]["contexts"] == observed["capsules"][0]["source_formalization_context"]["contexts"]
    assert exported[0]["current_sources_checked"] is exported[0]["proof_authority"] is False
    assert exported[0]["source_obligations_discharged"] == 0
    assert "MUST NOT APPEAR" not in json.dumps(result)
    assert result["benchmark_advantage_claimed"] is False
    assert observed == before


@pytest.mark.parametrize("change", ["no_match", "capsule", "native_sha", "native_bytes", "dispatch", "model",
    "invocation", "duplicate_invocation", "context_failure"])
def test_receipt_or_context_presence_does_not_establish_reported_model_delivery(observed, change):
    if change == "no_match":
        observed["matches"] = []
    elif change == "capsule":
        observed["matches"][0]["capsule_id"] = "cid:foreign"
    elif change == "native_sha":
        observed["matches"][0]["native_prompt_sha256"] = "2" * 64
    elif change == "native_bytes":
        observed["matches"][0]["native_prompt_bytes"] = 7
    elif change == "dispatch":
        observed["matches"][0]["model_input_checks"]["source_formalization_dispatch"] = 1
    elif change == "model":
        observed["matches"][0]["model_input_verified"] = False
    elif change == "invocation":
        observed["invocations"][0]["model_input_verified"] = False
    elif change == "duplicate_invocation":
        observed["invocations"].append(deepcopy(observed["invocations"][0]))
    else:
        observed["capsules"][0]["all_context_checks_passed"] = False
    result = project(observed)
    assert len(result) == 1 and result[0]["reported_model_input_verified"] is False
    assert result[0]["matched_invocation_ids"] == []


@pytest.mark.parametrize("change", ["authority", "freshness", "counter_type", "hash", "extra_text", "kind",
    "duplicate_kind", "duplicate_capsule", "oversize_capsules", "oversize_matches", "missing_schema"])
def test_malformed_or_unbounded_observation_is_not_projected(observed, change):
    value = observed["capsules"][0]["source_formalization_context"]
    if change == "authority": value["proof_authority"] = True
    elif change == "freshness": value["current_sources_checked"] = True
    elif change == "counter_type": value["source_obligations_discharged"] = False
    elif change == "hash": value["worker_evidence_sha256"] = "not a digest"
    elif change == "extra_text": value["candidate_ir_json"] = "MUST NOT APPEAR"
    elif change == "kind": value["contexts"][0]["kind"] = "another-family"
    elif change == "duplicate_kind": value["contexts"][1] = dict(value["contexts"][0])
    elif change == "duplicate_capsule": observed["capsules"].append(deepcopy(observed["capsules"][0]))
    elif change == "oversize_capsules": observed["capsules"] *= 33
    elif change == "oversize_matches": observed["matches"] *= 513
    else: del value["schema"]
    assert project(observed) == []


def test_legacy_audit_keeps_formalization_unobserved():
    assert project({"status": "verified_all_observed_model_inputs"}) == []
