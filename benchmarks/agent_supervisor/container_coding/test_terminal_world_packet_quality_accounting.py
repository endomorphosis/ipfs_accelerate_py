"""Authored comparator fixtures, never live provider or verifier evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_world_packet_quality_accounting as owner


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _pin(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _receipt(identity, total, *, thread=None, records=1, artifact=None, complete=True):
    return dict(schema="router-implementation-invocation@1", invocation_id=identity,
        provider="codex_cli", model="gpt-6.1-sol", purpose="coding", phase="coding",
        status="provider_returned", error_type=None, workspace="/private/workspace",
        model_prompt_sha256=_sha("private prompt " + identity), model_prompt_bytes=100,
        native_rollout_usage=dict(schema="native-codex-cumulative-usage@1",
            thread_id=thread or "native-session-" + identity,
            rollout_sha256=_sha(artifact or identity), observed_token_count_records=records,
            usage_available=True, task_complete_observed=complete,
            cache_included_in_input=True, billing_total_verified=False,
            usage=dict(input_tokens=total - 10, cached_input_tokens=10,
                output_tokens=10, reasoning_output_tokens=2, total_tokens=total)))


def _bundle(*, evidence="authored_fixture", baseline=120, candidate=80):
    common = {key: _sha(key) for key in owner._COMMON}
    common.update(provider="codex_cli", model="gpt-6.1-sol")
    case = dict(case_id="task-1", **{key: _sha(key) for key in owner._CASE_PINS},
        required_check_ids=["native-check-1", "native-check-2"], arms={})
    attempts, results = [], []
    for arm, total in (("baseline", baseline), ("candidate", candidate)):
        identity = arm + "-attempt-1"
        case["arms"][arm] = dict(codec_sha256=_sha("codec-" + arm),
            attempt_slots=[dict(attempt_id=identity, parent_attempt_id=None)])
        controls = {**common, **{k: case[k] for k in owner._CASE_PINS},
                    "codec_sha256": case["arms"][arm]["codec_sha256"]}
        attempts.append(dict(schema=owner.ATTEMPT_SCHEMA, attempt_id=identity,
            case_id=case["case_id"], arm=arm, parent_attempt_id=None,
            dispatch_state="dispatched", outcome="succeeded",
            output_sha256=_sha("output-" + arm),
            dispatch_evidence_sha256=_sha("dispatch-" + identity), observed_controls=controls,
            receipt=_receipt(identity, total), tool_trace=None))
        results.append(dict(schema=owner.VERIFIER_SCHEMA, case_id=case["case_id"], arm=arm,
            **{k: case[k] for k in owner._CASE_PINS}, verifier_policy_sha256=common["verifier_policy_sha256"],
            provenance="authored_fixture" if evidence == "authored_fixture" else "independently_retained_verifier",
            execution_status="completed", output_sha256=_sha("output-" + arm),
            evaluated_attempt_id=identity,
            checks=[dict(check_id=c, outcome="pass", evidence_sha256=_sha(c + arm)) for c in case["required_check_ids"]],
            reward=dict(outcome="pass", value=1, evidence_sha256=_sha("reward-" + arm))))
    declaration = dict(schema=owner.DECLARATION_SCHEMA,
        inventory_scope="pre_run_control_and_slot_inventory", experiment_id="authored-pair-1",
        evidence_kind=evidence, common_controls=common, cases=[case],
        execution_order=[a["attempt_id"] for a in attempts])
    inventory = dict(schema=owner.INVENTORY_SCHEMA, inventory_scope="post_run_observation_inventory",
        attempt_pins=[], verifier_pins=[], supplied_attempt_census_complete=True,
        unreceipted_provider_attempt=False)
    bundle = [declaration, inventory, attempts, results]
    _reseal(bundle)
    return bundle


def _reseal(bundle):
    declaration, inventory, attempts, results = bundle
    inventory["attempt_pins"] = [dict(attempt_id=a["attempt_id"], attempt_sha256=_pin(a),
        receipt_sha256=_pin(a["receipt"]) if a["receipt"] is not None else None,
        tool_trace_sha256=_pin(a["tool_trace"]) if a["tool_trace"] is not None else None) for a in attempts]
    inventory["verifier_pins"] = [dict(case_id=r["case_id"], arm=r["arm"],
        verifier_sha256=_pin(r), output_sha256=r["output_sha256"],
        evaluated_attempt_id=r["evaluated_attempt_id"]) for r in results]


def _inputs(bundle):
    return dict(zip(("expected_declaration_sha256", "expected_observation_inventory_sha256",
        "expected_attempts_sha256", "expected_verifier_results_sha256"), map(_pin, bundle)))


def _compare(bundle):
    return owner.compare_paired_quality_usage(*bundle, **_inputs(bundle))


def _add_retry(bundle, *, total=180, same_session=False, complete=True):
    declaration, _, attempts, _ = bundle
    first = attempts[0]
    identity = "baseline-attempt-2"
    retry = deepcopy(first)
    retry.update(attempt_id=identity, parent_attempt_id=first["attempt_id"], outcome="succeeded",
                 receipt=_receipt(identity, total, complete=complete))
    if same_session:
        retry["receipt"]["native_rollout_usage"].update(
            thread_id=first["receipt"]["native_rollout_usage"]["thread_id"], observed_token_count_records=2)
    declaration["cases"][0]["arms"]["baseline"]["attempt_slots"].append(
        dict(attempt_id=identity, parent_attempt_id=first["attempt_id"]))
    attempts.insert(1, retry)
    declaration["execution_order"].insert(1, identity)
    _reseal(bundle)


def test_authored_fixture_can_exercise_criteria_without_becoming_observed_evidence():
    bundle = _bundle()
    before = deepcopy(bundle)
    report = _compare(bundle)
    assert report["structural_gate"]["qualified"] is True
    assert report["quality_gate"] == dict(complete=True, candidate_nonworse=True)
    assert report["usage_gate"]["complete_native_total_tokens"] == dict(baseline=120, candidate=80)
    assert report["scientific_gate"]["criteria_met"] is True
    assert report["scientific_gate"]["observed_efficiency_qualified"] is False
    assert bundle == before
    assert all(report[key] is False for key in owner._FALSE)


def test_observed_label_is_conditional_declared_usage_never_campaign_or_bill():
    report = _compare(_bundle(evidence="observed_provider_workflow"))
    assert report["scientific_gate"]["observed_efficiency_qualified"] is True
    assert report["scientific_gate"]["comparison_scope"] == "declared_observed_workflow_usage_comparison"
    for key in ("campaign_call_population_complete", "provider_auxiliary_call_population_complete",
                "provider_internal_retries_known"):
        assert report["coverage"][key] is None
    assert report["coverage"]["control_temporal_origin_verified"] is False
    assert report["native_usage_by_arm"]["baseline"]["campaign_call_population_complete"] is None


@pytest.mark.parametrize("baseline,candidate,lower", [(120, 120, False), (80, 120, False), (120, 80, True)])
def test_usage_requires_strictly_lower_complete_population(baseline, candidate, lower):
    report = _compare(_bundle(baseline=baseline, candidate=candidate))
    assert report["usage_gate"]["strictly_lower_candidate_native_usage"] is lower
    assert report["scientific_gate"]["criteria_met"] is lower


def test_real_pair_shape_negative_control_quality_improves_but_usage_increases():
    # Authored values exercise the negative-control shape, not live case evidence.
    bundle = _bundle(baseline=100, candidate=200)
    verifier = bundle[3][0]
    verifier["reward"].update(outcome="fail", value=0)
    verifier["checks"][0]["outcome"] = "fail"
    _reseal(bundle)
    report = _compare(bundle)
    assert report["quality_gate"]["candidate_nonworse"] is True
    assert report["scientific_gate"]["criteria_met"] is False


def test_lower_usage_cannot_hide_quality_regression():
    bundle = _bundle()
    bundle[3][1]["reward"].update(outcome="fail", value=0)
    bundle[3][1]["checks"][0]["outcome"] = "fail"
    _reseal(bundle)
    report = _compare(bundle)
    assert report["quality_gate"] == dict(complete=True, candidate_nonworse=False)
    assert report["scientific_gate"]["criteria_met"] is False


@pytest.mark.parametrize("index,reason", [(0, "declaration_hash_mismatch"), (1, "inventory_hash_mismatch"),
    (2, "attempt_population_hash_mismatch"), (3, "verifier_population_hash_mismatch")])
def test_independent_whole_input_pins_cannot_be_resealed_by_caller(index, reason):
    bundle = _bundle()
    pins = _inputs(bundle)
    bundle[index] = deepcopy(bundle[index])
    if index == 0: bundle[0]["experiment_id"] = "foreign"
    elif index == 1: bundle[1]["unreceipted_provider_attempt"] = True
    elif index == 2: bundle[2][0]["outcome"] = "failed"
    else: bundle[3][0]["output_sha256"] = _sha("foreign")
    with pytest.raises(owner.QualityAccountingError, match=reason):
        owner.compare_paired_quality_usage(*bundle, **pins)


@pytest.mark.parametrize("target,reason", [("receipt", "receipt_hash_mismatch"), ("attempt", "attempt_hash_mismatch"),
    ("verifier", "verifier_hash_mismatch"), ("output", "verifier_output_mismatch")])
def test_full_population_rehash_does_not_replace_independent_per_record_pins(target, reason):
    bundle = _bundle()
    if target == "receipt":
        bundle[2][0]["receipt"]["model_prompt_bytes"] = 101
        bundle[1]["attempt_pins"][0]["attempt_sha256"] = _pin(bundle[2][0])
    elif target == "attempt": bundle[2][0]["outcome"] = "failed"
    elif target == "verifier": bundle[3][0]["reward"]["value"] = 0
    else:
        bundle[3][0]["output_sha256"] = _sha("foreign")
        bundle[1]["verifier_pins"][0]["verifier_sha256"] = _pin(bundle[3][0])
    with pytest.raises(owner.QualityAccountingError, match=reason): _compare(bundle)


@pytest.mark.parametrize("population", ["attempt", "verifier", "order", "slot"])
def test_missing_reordered_or_extra_population_is_not_selectively_omitted(population):
    bundle = _bundle()
    if population == "attempt": bundle[2].pop()
    elif population == "verifier": bundle[3].reverse()
    elif population == "order": bundle[0]["execution_order"].reverse()
    else: bundle[0]["cases"][0]["arms"]["candidate"]["attempt_slots"].clear()
    with pytest.raises(owner.QualityAccountingError): _compare(bundle)


@pytest.mark.parametrize("field", ["source_sha256", "task_sha256", "acceptance_sha256", "requirements_sha256",
    "control_sha256", "model_config_sha256", "budgets_sha256", "cache_policy_sha256", "cache_state_sha256",
    "execution_policy_sha256", "verifier_policy_sha256", "codec_sha256", "provider", "model"])
def test_each_observed_control_unknown_or_unmatched_blocks_comparison(field):
    bundle = _bundle()
    bundle[2][1]["observed_controls"][field] = None
    _reseal(bundle)
    report = _compare(bundle)
    assert "observed_" + field + "_unmatched" in report["structural_gate"]["blockers"]
    assert report["usage_gate"]["complete"] is False


def test_receipt_route_cannot_disagree_with_independent_control_pin():
    bundle = _bundle()
    bundle[2][1]["receipt"]["model"] = "foreign-model"
    _reseal(bundle)
    assert "receipt_provider_model_mismatch" in _compare(bundle)["structural_gate"]["blockers"]


@pytest.mark.parametrize("marker", [None, True])
def test_unknown_or_present_unreceipted_attempts_keep_complete_totals_null(marker):
    bundle = _bundle()
    bundle[1]["unreceipted_provider_attempt"] = marker
    report = _compare(bundle)
    assert report["usage_gate"]["complete_native_total_tokens"] == dict(baseline=None, candidate=None)
    assert report["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] == 120


@pytest.mark.parametrize("census", [None, False])
def test_incomplete_declared_attempt_census_is_not_campaign_closure(census):
    bundle = _bundle()
    bundle[1]["supplied_attempt_census_complete"] = census
    assert _compare(bundle)["usage_gate"]["complete"] is False


@pytest.mark.parametrize("state,outcome", [("dispatched", "failed"), ("unknown", "unknown")])
def test_missing_dispatched_or_unknown_receipt_withholds_totals(state, outcome):
    bundle = _bundle()
    bundle[2][1].update(dispatch_state=state, outcome=outcome, receipt=None)
    _reseal(bundle)
    report = _compare(bundle)
    assert report["usage_gate"]["complete"] is False
    assert report["native_usage_by_arm"]["candidate"]["complete_native_total_tokens"] is None


@pytest.mark.parametrize("state,outcome", [("refused", "refused"), ("not_dispatched", "not_attempted")])
def test_pinned_nondispatch_retry_retained_without_fabricating_native_zero(state, outcome):
    bundle = _bundle()
    _add_retry(bundle)
    bundle[2][1].update(dispatch_state=state, outcome=outcome, receipt=None)
    _reseal(bundle)
    report = _compare(bundle)
    assert len(report["attempts"]) == 3 and report["population"]["retry_slot_count"] == 1
    assert report["usage_gate"]["complete_native_total_tokens"]["baseline"] == 120
    assert report["attempts"][1]["observation_sha256"] is None


def test_failed_postprocessing_retains_complete_native_cost_in_population():
    bundle = _bundle()
    bundle[2][0]["outcome"] = "failed"
    bundle[2][0]["receipt"]["status"] = "failed"
    _add_retry(bundle, total=180)
    report = _compare(bundle)
    assert report["population"]["failed_dispatched_attempt_count"] == 1
    assert report["usage_gate"]["complete_native_total_tokens"]["baseline"] == 300


def test_cumulative_cli_retries_select_latest_snapshot_not_sum():
    bundle = _bundle()
    _add_retry(bundle, total=180, same_session=True)
    report = _compare(bundle)
    assert report["usage_gate"]["complete_native_total_tokens"]["baseline"] == 180
    assert report["native_usage_by_arm"]["baseline"]["native_session_count"] == 1
    assert report["native_usage_by_arm"]["baseline"]["complete_native_counts"]["fresh_input_tokens"] is None


@pytest.mark.parametrize("fault", ["regression", "equal_record_conflict", "artifact_counter_conflict"])
def test_cli_session_conflicts_remain_null(fault):
    bundle = _bundle()
    _add_retry(bundle, total=180, same_session=True)
    first = bundle[2][0]["receipt"]["native_rollout_usage"]
    last = bundle[2][1]["receipt"]["native_rollout_usage"]
    if fault == "regression": last["usage"].update(input_tokens=40, total_tokens=50)
    elif fault == "equal_record_conflict": last["observed_token_count_records"] = 1
    else: last["rollout_sha256"] = first["rollout_sha256"]
    _reseal(bundle)
    report = _compare(bundle)
    assert report["usage_gate"]["complete"] is False
    assert report["native_usage_by_arm"]["baseline"]["sessions"][0]["status"] == "conflict"


def test_cross_arm_shared_native_session_blocks_comparison():
    bundle = _bundle()
    bundle[2][1]["receipt"]["native_rollout_usage"]["thread_id"] = bundle[2][0]["receipt"]["native_rollout_usage"]["thread_id"]
    _reseal(bundle)
    assert "cross_arm_native_session_overlap" in _compare(bundle)["structural_gate"]["blockers"]


def test_same_physical_invocation_cannot_disguise_two_dispatched_attempt_slots():
    bundle = _bundle()
    _add_retry(bundle)
    bundle[2][1]["receipt"] = deepcopy(bundle[2][0]["receipt"])
    _reseal(bundle)
    report = _compare(bundle)
    assert "invocation_reused_across_attempt_slots" in report["structural_gate"]["blockers"]


@pytest.mark.parametrize("fault", ["missing_check", "unknown_check", "missing_evidence", "unknown_reward",
    "missing_output", "check_conflict", "untrusted_origin", "verifier_failed", "requirements_mismatch"])
def test_quality_needs_complete_pinned_original_requirements_and_output_population(fault):
    bundle = _bundle()
    result = bundle[3][1]
    if fault == "missing_check": result["checks"].pop()
    elif fault == "unknown_check": result["checks"][0]["outcome"] = "unknown"
    elif fault == "missing_evidence": result["checks"][0]["evidence_sha256"] = None
    elif fault == "unknown_reward": result["reward"].update(outcome="unknown", value=None)
    elif fault == "missing_output": result["output_sha256"] = None
    elif fault == "check_conflict": result["checks"][0]["outcome"] = "fail"
    elif fault == "untrusted_origin": result["provenance"] = "unknown"
    elif fault == "verifier_failed": result["execution_status"] = "failed"
    else: result["requirements_sha256"] = _sha("foreign requirements")
    _reseal(bundle)
    report = _compare(bundle)
    assert report["quality_gate"]["complete"] is False
    assert report["scientific_gate"]["criteria_met"] is False


def test_full_report_replay_rejects_rehashed_forged_gate():
    bundle = _bundle()
    report = _compare(bundle)
    report["default_adoption_authorized"] = True
    with pytest.raises(owner.QualityAccountingError, match="report_replay_mismatch"):
        owner.validate_quality_accounting_report(report, *bundle,
            expected_report_sha256=_pin(report), **_inputs(bundle))


def test_report_validation_requires_external_full_pin_then_exact_regeneration():
    bundle = _bundle()
    report = _compare(bundle)
    assert owner.quality_accounting_report_sha256(report) == _pin(report)
    assert owner.validate_quality_accounting_report(report, *bundle,
        expected_report_sha256=_pin(report), **_inputs(bundle))["valid"] is True
    with pytest.raises(owner.QualityAccountingError, match="report_hash_mismatch"):
        owner.validate_quality_accounting_report(report, *bundle,
            expected_report_sha256=_sha("foreign"), **_inputs(bundle))


def _tool(events=None, **changes):
    value = dict(schema=owner.TOOL_SCHEMA, coverage="complete_declared", declared_event_count=1,
        events=events if events is not None else [dict(event_id_sha256=_sha("tool-1"),
            kind="retrieval_response", bytes=9999, payload_sha256=_sha("private payload"))])
    value.update(changes)
    return value


def test_tool_trace_bytes_retained_separately_and_never_added_to_native_tokens():
    bundle = _bundle()
    bundle[2][1]["tool_trace"] = _tool()
    _reseal(bundle)
    report = _compare(bundle)
    assert report["attempts"][1]["tool_census"]["payload_bytes"] == 9999
    assert report["usage_gate"]["complete_native_total_tokens"]["candidate"] == 80
    assert report["tool_trace_scope"]["tool_bytes_added_to_native_tokens"] is False
    assert report["tool_trace_scope"]["provider_token_attribution"] is None


@pytest.mark.parametrize("fault", ["missing_bytes", "partial", "unknown"])
def test_tool_census_unknown_is_not_fabricated_zero(fault):
    bundle = _bundle()
    tool = _tool()
    if fault == "missing_bytes": tool["events"][0]["bytes"] = None
    else: tool["coverage"] = fault
    bundle[2][0]["tool_trace"] = tool
    _reseal(bundle)
    assert _compare(bundle)["attempts"][0]["tool_census"]["payload_bytes"] is None


@pytest.mark.parametrize("fault", ["bool_counter", "overflow", "duplicate", "count_mismatch", "hash_mismatch"])
def test_tool_trace_bound_or_pin_failures_refuse(fault):
    bundle = _bundle()
    tool = _tool()
    if fault == "bool_counter": tool["events"][0]["bytes"] = True
    elif fault == "overflow":
        tool["events"][0]["bytes"] = owner.MAX_COUNTER
        tool["events"].append(dict(tool["events"][0], event_id_sha256=_sha("tool-2"), bytes=1))
        tool["declared_event_count"] = 2
    elif fault == "duplicate": tool["events"].append(deepcopy(tool["events"][0]))
    elif fault == "count_mismatch": tool["declared_event_count"] = 2
    bundle[2][0]["tool_trace"] = tool
    _reseal(bundle)
    if fault == "hash_mismatch": bundle[1]["attempt_pins"][0]["tool_trace_sha256"] = _sha("foreign")
    with pytest.raises(owner.QualityAccountingError): _compare(bundle)


def test_unknown_raw_receipt_fields_pinned_but_never_exported():
    bundle = _bundle()
    bundle[2][0]["receipt"].update(raw_prompt="secret literal", raw_source="private source",
                                 credential="sensitive token")
    _reseal(bundle)
    report = _compare(bundle)
    rendered = json.dumps(report)
    for forbidden in ("secret literal", "private source", "sensitive token", "/private/workspace"):
        assert forbidden not in rendered
    assert report["attempts"][0]["full_receipt_sha256"] == _pin(bundle[2][0]["receipt"])


@pytest.mark.parametrize("fault", ["bool_reward", "false_pin", "extra_authority", "postrun_as_prerun",
    "duplicate_case", "duplicate_slot", "invalid_retry", "float_value"])
def test_closed_types_temporal_roles_and_population_ids_refuse(fault):
    bundle = _bundle()
    if fault == "bool_reward": bundle[3][0]["reward"]["value"] = True
    elif fault == "false_pin": bundle[0]["common_controls"]["control_sha256"] = False
    elif fault == "extra_authority": bundle[3][0]["passed"] = True
    elif fault == "postrun_as_prerun": bundle[0]["inventory_scope"] = "post_run_observation_inventory"
    elif fault == "duplicate_case": bundle[0]["cases"].append(deepcopy(bundle[0]["cases"][0]))
    elif fault == "duplicate_slot": bundle[0]["cases"][0]["arms"]["candidate"]["attempt_slots"][0]["attempt_id"] = "baseline-attempt-1"
    elif fault == "invalid_retry": bundle[0]["cases"][0]["arms"]["baseline"]["attempt_slots"][0]["parent_attempt_id"] = "candidate-attempt-1"
    else: bundle[3][0]["reward"]["value"] = 1.0
    _reseal(bundle)
    with pytest.raises(owner.QualityAccountingError): _compare(bundle)


def test_overlarge_input_refused_before_core_projection(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as core
    bundle = _bundle()
    bundle[2][0]["receipt"]["oversize"] = "x" * (owner.MAX_INPUT_BYTES + 1)
    monkeypatch.setattr(core, "observe_invocation", lambda _: pytest.fail("must refuse before projection"))
    with pytest.raises(owner.QualityAccountingError, match="json_byte_bound"):
        owner.compare_paired_quality_usage(*bundle, **_inputs(bundle))


def test_grok_final_envelopes_use_disjoint_cache_categories_without_local_proxy():
    bundle = _bundle()
    bundle[0]["common_controls"].update(provider="grok_cli", model="grok-4.7")
    for index, attempt in enumerate(bundle[2]):
        attempt["observed_controls"].update(provider="grok_cli", model="grok-4.7")
        receipt = attempt["receipt"]
        receipt.update(provider="grok_cli", model="grok-4.7", local_token_count=1)
        inputs = 30 if index == 0 else 10
        receipt["native_rollout_usage"] = dict(schema="native-grok-final-usage@1",
            usage_available=True, usage_observation_sha256=_sha("grok-" + str(index)),
            task_complete_observed=True, usage_complete_observed=True,
            totals_scope="native_final_envelope", cache_included_in_input=False,
            observed_final_usage_records=1, billing_total_verified=False,
            usage=dict(input_tokens=inputs, cached_input_tokens=20, cache_write_input_tokens=5,
                       output_tokens=7, reasoning_output_tokens=3, total_tokens=inputs + 32))
    _reseal(bundle)
    report = _compare(bundle)
    assert report["usage_gate"]["complete_native_total_tokens"] == dict(baseline=62, candidate=42)
    counts = report["native_usage_by_arm"]["candidate"]["complete_native_counts"]
    assert counts["input_tokens_including_cache"] == 35
    assert counts["fresh_input_tokens"] == 10
    assert counts["reasoning_output_tokens"] == 3


def test_cumulative_session_shared_across_cases_is_selected_once_for_whole_arm():
    bundle = _bundle()
    declaration, _, attempts, results = bundle
    case = deepcopy(declaration["cases"][0])
    case["case_id"] = "task-2"
    for arm in owner.ARMS:
        identity = arm + "-case-2"
        case["arms"][arm]["attempt_slots"] = [dict(attempt_id=identity, parent_attempt_id=None)]
        attempt = deepcopy(attempts[0 if arm == "baseline" else 1])
        attempt.update(attempt_id=identity, case_id="task-2")
        attempt["receipt"] = _receipt(identity, 200 if arm == "baseline" else 80)
        if arm == "baseline":
            attempt["receipt"]["native_rollout_usage"].update(
                thread_id=attempts[0]["receipt"]["native_rollout_usage"]["thread_id"],
                observed_token_count_records=2)
        attempts.append(attempt)
        declaration["execution_order"].append(identity)
        result = deepcopy(results[0 if arm == "baseline" else 1])
        result["case_id"] = "task-2"
        result["evaluated_attempt_id"] = identity
        results.append(result)
    declaration["cases"].append(case)
    _reseal(bundle)
    report = _compare(bundle)
    assert report["population"]["case_count"] == 2
    assert report["usage_gate"]["complete_native_total_tokens"] == dict(baseline=200, candidate=160)


def test_missing_native_usage_is_null_even_if_router_generic_usage_claims_zero():
    bundle = _bundle()
    bundle[2][1]["receipt"].update(native_rollout_usage=None,
                                 usage=dict(input_tokens=0, output_tokens=0, total_tokens=0))
    _reseal(bundle)
    report = _compare(bundle)
    assert report["usage_gate"]["complete"] is False
    assert report["native_usage_by_arm"]["candidate"]["reported_native_token_subtotal"] is None


def test_refusal_without_dispatch_evidence_cannot_hide_possible_cost():
    bundle = _bundle()
    _add_retry(bundle)
    bundle[2][1].update(dispatch_state="refused", outcome="refused", receipt=None,
                        dispatch_evidence_sha256=None)
    _reseal(bundle)
    report = _compare(bundle)
    assert "dispatch_evidence_incomplete" in report["structural_gate"]["blockers"]
    assert report["usage_gate"]["complete"] is False


def test_native_receipt_elapsed_float_is_pinned_without_becoming_usage_or_reward():
    bundle = _bundle()
    bundle[2][0]["receipt"]["seconds"] = 0.125
    _reseal(bundle)
    report = _compare(bundle)
    assert report["usage_gate"]["complete_native_total_tokens"]["baseline"] == 120
    assert "seconds" not in report["native_usage_by_arm"]["baseline"]["observations"][0]


def test_pinned_input_snapshots_cannot_be_redirected_by_later_owner_callback(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as core
    bundle = _bundle()
    original = core.observe_invocation
    def mutate_external(receipt):
        bundle[0]["cases"][0]["source_sha256"] = _sha("foreign mutation")
        bundle[3][1]["checks"][0]["outcome"] = "fail"
        return original(receipt)
    monkeypatch.setattr(core, "observe_invocation", mutate_external)
    report = _compare(bundle)
    assert report["quality_gate"]["complete"] is True
    assert report["quality_gate"]["candidate_nonworse"] is True


@pytest.mark.parametrize("fault", ["unrelated_output", "cross_arm", "unknown_attempt", "missing_attempt",
    "missing_attempt_output", "missing_verifier_output", "refused_attempt"])
def test_verifier_must_join_exact_dispatched_attempt_and_output_identity(fault):
    bundle = _bundle()
    attempt, verifier = bundle[2][1], bundle[3][1]
    if fault == "unrelated_output": verifier["output_sha256"] = _sha("unrelated output")
    elif fault == "cross_arm": verifier["evaluated_attempt_id"] = bundle[2][0]["attempt_id"]
    elif fault == "unknown_attempt": verifier["evaluated_attempt_id"] = "foreign-attempt"
    elif fault == "missing_attempt": verifier["evaluated_attempt_id"] = None
    elif fault == "missing_attempt_output": attempt["output_sha256"] = None
    elif fault == "missing_verifier_output": verifier["output_sha256"] = None
    else: attempt.update(dispatch_state="refused", outcome="refused", receipt=None, output_sha256=None)
    _reseal(bundle)
    report = _compare(bundle)
    assert report["quality_gate"]["complete"] is False
    assert report["scientific_gate"]["criteria_met"] is False


def test_valid_earlier_retry_output_can_be_reviewed_while_all_later_costs_remain():
    bundle = _bundle()
    _add_retry(bundle, total=180)
    bundle[2][1].update(outcome="failed", output_sha256=None)
    _reseal(bundle)
    report = _compare(bundle)
    assert report["quality_by_case"][0]["arms"]["baseline"]["evaluated_attempt_id"] == "baseline-attempt-1"
    assert report["quality_gate"]["complete"] is True
    assert report["usage_gate"]["complete_native_total_tokens"]["baseline"] == 300


def test_failed_postprocessing_can_leave_exact_independently_checked_output():
    bundle = _bundle()
    bundle[2][1]["outcome"] = "failed"
    bundle[2][1]["receipt"]["status"] = "failed"
    _reseal(bundle)
    report = _compare(bundle)
    assert report["quality_by_case"][0]["arms"]["candidate"]["passed"] is True
    assert report["population"]["failed_dispatched_attempt_count"] == 1


def test_verifier_attempt_join_pin_cannot_be_replaced_by_only_rehashing_verifier():
    bundle = _bundle()
    bundle[3][1]["evaluated_attempt_id"] = "baseline-attempt-1"
    bundle[1]["verifier_pins"][1]["verifier_sha256"] = _pin(bundle[3][1])
    with pytest.raises(owner.QualityAccountingError, match="verifier_attempt_pin_mismatch"):
        _compare(bundle)


def test_independent_planning_and_coding_calls_remain_roots_and_both_are_charged():
    bundle = _bundle()
    _add_retry(bundle, total=180)
    bundle[0]["cases"][0]["arms"]["baseline"]["attempt_slots"][1]["parent_attempt_id"] = None
    bundle[2][1]["parent_attempt_id"] = None
    bundle[2][0]["receipt"]["purpose"] = "planning"
    bundle[2][0]["receipt"]["phase"] = "planning"
    _reseal(bundle)
    report = _compare(bundle)
    assert report["population"]["retry_slot_count"] == 0
    assert report["population"]["dispatched_attempt_count"] == 3
    assert report["usage_gate"]["complete_native_total_tokens"]["baseline"] == 300
    assert report["native_usage_by_arm"]["baseline"]["observations"][0]["purpose"] == "planning"


def test_nonnull_retry_parent_still_must_precede_child_in_execution_order():
    bundle = _bundle()
    _add_retry(bundle)
    bundle[0]["execution_order"][0:2] = reversed(bundle[0]["execution_order"][0:2])
    bundle[2][0:2] = reversed(bundle[2][0:2])
    _reseal(bundle)
    with pytest.raises(owner.QualityAccountingError, match="retry_execution_order"):
        _compare(bundle)
