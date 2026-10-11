"""Pure authored metadata fixtures; no providers, model assets, or owner state."""
import copy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding.benchmark_provider_profile import resolve_provider_profile
from benchmarks.agent_supervisor.container_coding.terminal_live_census_provenance import (
    TOKEN_FIELDS, audit_planning_usage, validate_live_planning_provenance,
)


@pytest.fixture
def bound():
    selected = resolve_provider_profile()
    boundary = {"schema": "supervisor-container-worker-boundary@1", "container_id": "c" * 64,
        "image_id": "sha256:" + "d" * 64, "namespaces": {"pid": "pid:[11]", "mnt": "mnt:[22]", "net": "net:[33]"},
        "owner_uid": 1000, "worker_uid": 1001, "allowed_worktree_roots": ["/opt/ipfs-supervisor/worktrees"],
        "owner_private_paths": ["/opt/ipfs-supervisor/state", "/opt/ipfs-supervisor/home", "/opt/ipfs-supervisor/auth"],
        "provider": selected["provider"], "validation_repository_roots": ["/app"], "single_worker": True}
    raw = (json.dumps(boundary, sort_keys=True, indent=2) + "\n").encode()
    boundary_sha = hashlib.sha256(raw).hexdigest()
    native = {key: boundary[key] for key in ("schema", "container_id", "image_id", "namespaces", "owner_uid", "worker_uid")}
    native.update(purpose="coding", manifest_sha256=boundary_sha, completion_authority=False)
    deployment = {"schema": "container-worker-deployment@1", "boundary": boundary,
        "boundary_sha256": boundary_sha, "worker_worktree_root": "/opt/ipfs-supervisor/worktrees",
        "provider_calls": 0, "qualified": True,
        "qualification": {"qualified": True, "provider_calls": 0, "completion_authority": False,
            "remaining_worker_pids": [], "permission_evidence": {"schema": "container-worker-preflight@1",
                "uid": 1001, "boundary": copy.deepcopy(native), "candidate_write": True, "provider_calls": 0,
                "workspace": "/opt/ipfs-supervisor/worktrees/provider-free",
                "private_denials": ["/opt/ipfs-supervisor/state/" + name + ":" + mode
                    for name in ("preflight-key", "preflight-owner.duckdb") for mode in ("rb", "ab")]
                    + [".git:write", ".git:unlink"]}}}
    projection = {"projection_id": "canonical-prompt-goal-codex-generation-schema@1",
        "canonical_schema_sha256": "a" * 64, "canonical_schema_bytes": 800,
        "native_wire_schema_sha256": "b" * 64, "native_wire_schema_bytes": 700,
        "canonical_validation_preserved": True, "top_level_id_omitted": True,
        "bounds_annotation_omitted": True, "definitions_relocated": True, "shape_only": True}
    receipt = {"schema": "router-implementation-invocation@1", "invocation_id": "f" * 32,
        "provider": selected["provider"], "model": selected["model"], "reasoning_effort": selected["reasoning_effort"],
        "purpose": "planning", "status": "provider_returned", "router_calls": 1, "completion_authority": False,
        "prompt_sha256": "1" * 64, "native_prompt_sha256": "1" * 64, "prompt_bytes": 9000, "native_prompt_bytes": 9000,
        "model_prompt_sha256": "2" * 64, "model_prompt_bytes": 9300,
        "workspace": "/opt/ipfs-supervisor/worktrees/planner-example", "external_container_boundary": native,
        "provider_invocation_policy": {"structured_output": {"schema": "codex-native-planning-json-schema@1",
            "native_schema_projection": projection, "native_schema_requested": True, "response_schema_validated": True}},
        "usage": {"exit_code": 0, "codex_output_schema_sha256": "b" * 64, "codex_output_schema_bytes": 700,
                  "thread_id": "native-thread-123456", "total_cost_usd": 0.25},
        "native_rollout_usage": {"schema": "native-codex-cumulative-usage@1", "thread_id": "native-thread-123456",
            "rollout_sha256": "3" * 64, "observed_token_count_records": 7,
            "usage": {"input_tokens": 100, "cached_input_tokens": 20, "cache_write_input_tokens": 0,
                "output_tokens": 40, "reasoning_output_tokens": 30, "total_tokens": 140},
            "usage_available": True, "task_complete_observed": True, "cache_included_in_input": True,
            "billing_total_verified": False}}
    planning = {"schema": "terminal-indexed-planner-result@1", "qualified": True, "provider_calls": 1,
        "planning_strategy": "direct", "provider": selected["provider"], "model": selected["model"],
        "reasoning_effort": selected["reasoning_effort"], "model_request_sha256": "1" * 64,
        "execution_receipt": copy.deepcopy(receipt), "provider_receipt": {"outcome": "provider",
            "fallback": {"used": False}, "plan_root_cid": "fixture-plan",
            "parse": {"attempted": True, "status": "succeeded", "plan_root_cid": "fixture-plan"},
            "provider": {"attempted": True, "status": "succeeded", "provider_id": selected["provider"], "model_id": selected["model"]}}}
    return planning, [dict(phase="planning", **receipt)], selected, deployment


def _validate(bound):
    planning, rows, selected, deployment = bound
    return validate_live_planning_provenance(planning, rows, selected, deployment_boundary=deployment)


def test_actual_metadata_joins_and_all_six_usage_categories(bound):
    result = _validate(bound)
    assert result["actual_native_schema_observation_matched"]
    assert result["external_worker_custody_join_verified"]
    assert not result["deployment_observation_authentication_performed_here"]
    usage = result["usage"]
    assert set(usage["tokens"]) == set(TOKEN_FIELDS)
    assert usage["tokens"]["total_tokens"] == 140
    assert usage["tokens"]["input_tokens"] == 100 and usage["tokens"]["output_tokens"] == 40
    assert usage["tokens"]["reasoning_output_tokens"] == 30
    assert usage["reported_cost_usd"] == 0.25
    assert not usage["billing_total_verified"] and not usage["cumulative_records_summed"]
    assert not any(result[key] for key in ("proof_authority", "execution_authority", "completion_authority", "publication_authority"))


@pytest.mark.parametrize("mutation", ["authored", "stdin", "route", "schema-observed", "schema-requested",
    "schema-projection", "native-thread", "boundary", "deployment-string", "deployment-unqualified",
    "worker-uid", "private-denials", "remaining-worker", "namespace", "foreign-workspace", "unknown-field", "fallback"])
def test_fabricated_or_drifted_provenance_refused(bound, mutation):
    planning, rows, selected, deployment = bound
    row = rows[0]
    if mutation == "authored": planning["execution_receipt"] = None
    elif mutation == "stdin": planning["model_request_sha256"] = "9" * 64
    elif mutation == "route": row["model"] = "foreign-model"
    elif mutation == "schema-observed": row["usage"]["codex_output_schema_sha256"] = "9" * 64
    elif mutation == "schema-requested": row["provider_invocation_policy"]["structured_output"]["native_schema_requested"] = False
    elif mutation == "schema-projection": row["provider_invocation_policy"]["structured_output"]["native_schema_projection"]["projection_id"] = "caller-description"
    elif mutation == "native-thread": row["native_rollout_usage"]["thread_id"] = "foreign-thread-123456"
    elif mutation == "boundary": row["external_container_boundary"]["container_id"] = "9" * 64
    elif mutation == "deployment-string": deployment = "trusted live worker"
    elif mutation == "deployment-unqualified": deployment["qualified"] = False
    elif mutation == "worker-uid": deployment["qualification"]["permission_evidence"]["uid"] = 1000
    elif mutation == "private-denials": deployment["qualification"]["permission_evidence"]["private_denials"].pop()
    elif mutation == "remaining-worker": deployment["qualification"]["remaining_worker_pids"] = [123]
    elif mutation == "namespace": deployment["qualification"]["permission_evidence"]["boundary"]["namespaces"]["pid"] = "pid:[99]"
    elif mutation == "foreign-workspace": row["workspace"] = "/foreign/planner"
    elif mutation == "unknown-field": row["provider_response"] = "private canary"
    elif mutation == "fallback": planning["provider_receipt"]["fallback"]["used"] = True
    # Exercise each actual join, independently of the exact receipt snapshot.
    if mutation != "authored": planning["execution_receipt"] = {key: value for key, value in row.items() if key != "phase"}
    with pytest.raises(ValueError):
        validate_live_planning_provenance(planning, rows, selected, deployment_boundary=deployment)


def test_duplicate_observations_count_final_cumulative_totals_once(bound):
    planning, rows, _, _ = bound
    rows.append(copy.deepcopy(rows[0]))
    result = _validate(bound)
    assert result["usage"]["emitted_router_receipts"] == 2
    assert result["usage"]["unique_router_invocations"] == 1
    assert result["usage"]["observed_router_dispatches"] == 1
    assert result["usage"]["tokens"]["total_tokens"] == 140


def test_conflicting_duplicate_and_multiple_planning_attempts_cannot_qualify(bound):
    planning, rows, _, _ = bound
    other = copy.deepcopy(rows[0])
    other["native_rollout_usage"]["usage"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="conflicting"):
        audit_planning_usage(planning, rows + [other])
    other["invocation_id"] = "e" * 32
    with pytest.raises(ValueError, match="exactly one"):
        _validate((planning, rows + [other], bound[2], bound[3]))


def test_failed_session_usage_is_lower_bound_and_unknown_categories_stay_null(bound):
    planning, rows, _, _ = bound
    row = rows[0]
    row["status"] = "failed"
    row["failure_phase"] = "provider_invocation"
    native = row["native_rollout_usage"]
    native["task_complete_observed"] = False
    del native["usage"]["cache_write_input_tokens"]
    result = audit_planning_usage(planning, rows)
    assert result["tokens"] == dict.fromkeys(TOKEN_FIELDS)
    assert result["known_token_subtotals"]["total_tokens"] == 140
    assert result["known_token_subtotals"]["cache_write_input_tokens"] is None
    assert result["reported_cost_usd"] is None and result["known_reported_cost_subtotal_usd"] == 0.25
    assert not result["observed_complete_sessions"]


def test_authentication_failure_without_native_tokens_does_not_become_zero(bound):
    planning, rows, _, _ = bound
    row = rows[0]
    row["status"] = "failed"
    row["usage"] = {"exit_code": 1}
    row["native_rollout_usage"] = None
    result = audit_planning_usage(planning, rows)
    assert result["tokens"] == result["known_token_subtotals"] == dict.fromkeys(TOKEN_FIELDS)
    assert result["known_reported_cost_subtotal_usd"] is None
    assert result["planner_callable_attempts"] == result["unique_router_invocations"] == 1


def test_unreceipted_attempt_retains_observed_subtotal_without_claiming_complete_total(bound):
    planning, rows, _, _ = bound
    planning["provider_calls"] = 2
    result = audit_planning_usage(planning, rows)
    assert result["unreceipted_callable_attempts"] == 1
    assert result["known_token_subtotals"]["total_tokens"] == 140
    assert result["tokens"]["total_tokens"] is None
    empty = audit_planning_usage({"provider_calls": 1}, [])
    assert empty["unreceipted_callable_attempts"] == 1 and empty["tokens"]["input_tokens"] is None


@pytest.mark.parametrize("value", [-1, True, 1.5, "10"])
def test_invalid_native_counters_refused(bound, value):
    bound[1][0]["native_rollout_usage"]["usage"]["input_tokens"] = value
    with pytest.raises(ValueError, match="token categories"):
        audit_planning_usage(bound[0], bound[1])


def test_foreign_cumulative_session_reuse_refused(bound):
    planning, rows, _, _ = bound
    planning["provider_calls"] = 2
    other = copy.deepcopy(rows[0])
    other["invocation_id"] = "e" * 32
    with pytest.raises(ValueError, match="cumulative native thread"):
        audit_planning_usage(planning, rows + [other])


def test_unknown_native_usage_and_private_metadata_never_exported(bound):
    planning, rows, _, _ = bound
    planning["private_fixture_note"] = "PRIVATE_BODY_CANARY"
    rows[0]["native_rollout_usage"] = None
    planning["execution_receipt"] = {key: value for key, value in rows[0].items() if key != "phase"}
    result = _validate(bound)
    assert "PRIVATE_BODY_CANARY" not in json.dumps(result)
    assert result["usage"]["tokens"]["total_tokens"] is None


def test_available_usage_requires_native_categories(bound):
    bound[1][0]["native_rollout_usage"]["usage"] = {}
    with pytest.raises(ValueError, match="no categories"):
        audit_planning_usage(bound[0], bound[1])


def test_unknown_attempt_count_preserves_known_usage_without_complete_claim(bound):
    del bound[0]["provider_calls"]
    result = audit_planning_usage(bound[0], bound[1])
    assert result["planner_callable_attempts"] is None
    assert result["unreceipted_callable_attempts"] is None
    assert result["known_token_subtotals"]["total_tokens"] == 140
    assert result["tokens"]["total_tokens"] is None
    assert not result["all_callable_attempts_receipted"]
