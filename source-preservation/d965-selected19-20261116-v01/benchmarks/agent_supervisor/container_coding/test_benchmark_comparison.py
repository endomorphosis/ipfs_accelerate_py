import copy
import hashlib
import json

import pytest
from benchmarks.agent_supervisor.container_coding.benchmark_comparison import (
    BASELINE,
    SUPERVISOR,
    collect,
    markdown,
)
from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison


def _receipt(tmp_path, name, schema, trials, arm=None):
    value = {
        "schema": schema,
        "task": "fix-code-vulnerability",
        "model": "pinned-model",
        "reasoning_effort": "high",
        "cli_version": "pinned-cli",
        "original_task_inputs_unchanged": True,
        "trial_count": len(trials),
        "trials": trials,
    }
    if arm:
        value["arm"] = arm
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def _baseline(tmp_path):
    return _receipt(
        tmp_path,
        "baseline.json",
        BASELINE,
        [
            {
                "trial": "baseline",
                "reward": {"reward": 1},
                "seconds": {"agent_execution": 12},
                "raw_usage": {
                    "sessions": [{"session_id": "session", "task_complete_observed": True}],
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 80,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            }
        ],
    )


def _invocation(identity="call"):
    return {
        "invocation_id": identity,
        "provider": "codex_cli",
        "phase": "planning",
        "status": "provider_returned",
        "native_rollout_usage": {
            "schema": "native-codex-cumulative-usage@1",
            "cache_included_in_input": True,
            "task_complete_observed": True,
            "usage": {
                "input_tokens": 20,
                "cached_input_tokens": 15,
                "output_tokens": 2,
                "total_tokens": 22,
            }
        },
        "raw_model_text": "MUST NOT APPEAR",
        "prompt": "MUST NOT APPEAR",
    }


def _grok_invocation(identity="grok-call", complete=None):
    return {
        "invocation_id": identity, "phase": "coding", "provider": "grok_cli", "status": "provider_returned",
        "native_rollout_usage": {
            "schema": "native-grok-final-usage@1", "usage_available": True,
            "usage_complete_observed": complete, "task_complete_observed": True,
            "totals_scope": "native_final_envelope", "cache_included_in_input": False,
            "billing_total_verified": False, "dollar_cost": None,
            "usage": {"input_tokens": 10, "cached_input_tokens": 30, "cache_write_input_tokens": 0,
                      "output_tokens": 2, "total_tokens": 42},
        },
    }


def _identified_invocation(identity="modern", *, thread="native-session-123456789", records=1,
                           inputs=100, outputs=20, complete=True, phase="coding", artifact=None):
    """Authored native receipt shape; no provider process or token savings claim."""
    artifact = artifact or identity
    return dict(schema="router-implementation-invocation@1", invocation_id=identity,
        provider="codex_cli", model="pinned-model", phase=phase, purpose=phase, status="provider_returned",
        model_prompt_sha256="a" * 64, model_prompt_bytes=123, workspace="/authored/worktree",
        native_rollout_usage=dict(schema="native-codex-cumulative-usage@1", thread_id=thread,
            usage_available=True, task_complete_observed=complete, cache_included_in_input=True,
            rollout_sha256=hashlib.sha256(artifact.encode()).hexdigest(), observed_token_count_records=records,
            billing_total_verified=False, latest_usage_snapshot_valid=True, counter_regression_observed=False,
            usage=dict(input_tokens=inputs, cached_input_tokens=inputs // 2, output_tokens=outputs,
                       reasoning_output_tokens=outputs // 2, total_tokens=inputs + outputs)))


def _interaction_trial(calls, *, marker=False, **report_fields):
    report = dict(provider_invocations=calls, unreceipted_provider_attempt=marker, **report_fields)
    return comparison._trial(dict(schema=SUPERVISOR, arm="full", task="authored-task", model="pinned-model"),
        dict(trial="authored-trial", reward={"reward": 1}, supervisor=report),
        source=dict(path="authored-receipt", sha256="b" * 64))


def _baseline_native(identities=("baseline-session-123456789",)):
    """Authored current baseline producer shape, not native provider execution."""
    usage = dict(input_tokens=100, cached_input_tokens=60, cache_write_input_tokens=None,
                 output_tokens=10, reasoning_output_tokens=5, total_tokens=110)
    sessions = [dict(session_id=identity, sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                     task_complete_observed=True, usage=dict(usage)) for index, identity in enumerate(identities)]
    total = {key: value * len(sessions) if value is not None else None for key, value in usage.items()}
    return dict(source="native_rollout_observed_cumulative_totals", sessions=sessions, usage=total,
                task_complete_observed=True, reported_cost_usd=0.125 * len(sessions))


def _baseline_trial(native, *, identity="authored-baseline", reward=1, **fields):
    return comparison._trial(dict(schema=BASELINE, task="authored-task", model="pinned-model"),
        dict(trial=identity, reward={"reward": reward}, raw_usage=native, **fields),
        source=dict(path="authored-baseline-receipt", sha256="c" * 64))


def _decision_dialogue(*natives):
    """Authored dialogue bindings; wrapper/local counts never represent charges."""
    attempts = []
    for index, native in enumerate(natives):
        call = "authored-decision-call-" + str(index)
        receipt = dict(schema="router-semantic-decision-provider@1", call_id=call,
            session_id="authored-dialogue-session", provider=native["provider"], model=native["model"],
            workspace=native["workspace"], model_prompt_sha256=native["model_prompt_sha256"],
            provider_called=True, native_invocation=native, usage={"prompt_tokens": 999999})
        attempts.append(dict(call_id=call, model_prompt_sha256=native["model_prompt_sha256"],
            provider_receipt_bound=True, provider_receipt=receipt, content_tokens=999999))
    return dict(schema="supervisor-semantic-decision-dialogue@1", session_id="authored-dialogue-session",
        provider=natives[0]["provider"], model=natives[0]["model"], workspace=natives[0]["workspace"],
        invocation_attempt_count=len(attempts), attempts=attempts,
        provider_usage_complete=True, codex_cumulative_usage={"provider_total_tokens": 1})


def _reported_collection(*, complete=True):
    """Authored local scan observation, not authenticated or exhaustive coverage."""
    counts = dict.fromkeys(comparison._COLLECTION_COUNTERS, 0)
    counts.update(entries_examined=1, log_files_seen=1, log_files_read=1,
        bytes_read=100, lines_examined=3, provider_observations=1)
    reasons = []
    if not complete:
        counts.update(entries_examined=2, log_files_seen=2, oversized_logs=1)
        reasons = ["oversized_log"]
    return dict(schema="terminal-provider-receipt-collection@1", status="observed" if complete else "incomplete",
        complete=complete, counts=counts, reason_codes=reasons, scope="retained_state_logs_only",
        attempt_population_complete=None, provider_call_count=None, campaign_call_population_complete=None,
        observation_only=True, provider_usage_authenticity_verified=False, provider_dispatch_count_verified=False,
        token_savings_measured=False, completion_authority=False, execution_authority=False)


def test_missing_selected_semantic_dialogue_blocks_outer_usage_without_rewriting_native_facts():
    native = _identified_invocation()
    native["native_rollout_usage"]["dollar_cost"] = 1
    row = _interaction_trial([native], semantic_decision_dialogue_expected=True,
        semantic_decision_coverage={"authoritative_usage_coverage_complete": True},
        provider_usage_accounting={"report_provider_total_tokens": 1})
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["reported_cost_usd"] is None and row["provider_calls"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 120
    assert row["known_reported_cost_subtotal_usd"] == 1 and row["observed_provider_invocations"] == 1
    assert row["native_session_accounting"]["complete_native_total_tokens"] == 120
    observation = row["usage_coverage_observation"]
    assert observation["complete_usage_blocked"] is True
    assert observation["semantic_decision_coverage_recomputed"] is True
    assert observation["stored_usage_summary_trusted"] is False
    assert "expected_dialogue_receipt_missing" in {issue["reason"]
        for issue in observation["semantic_decision_coverage"]["issues"]}
    assert observation["campaign_usage_complete"] is None and observation["whole_interaction_qualified"] is False


def test_semantic_accounting_owner_retains_native_receipt_found_only_inside_dialogue():
    planning = _identified_invocation("planning", phase="planning")
    nested = _identified_invocation("nested", thread="nested-native-session-123456789", inputs=70, outputs=7)
    dialogue = _decision_dialogue(nested)
    before = copy.deepcopy(dialogue)
    row = _interaction_trial([planning], semantic_decision_dialogue_expected=True,
        semantic_decision_dialogues=[dialogue], provider_usage_accounting={"report_provider_total_tokens": 1})
    assert dialogue == before
    assert row["tokens"]["total_tokens"] == row["known_token_subtotals"]["total_tokens"] == 197
    assert row["provider_calls"] == row["observed_provider_invocations"] == 2
    assert row["native_session_accounting"]["native_session_count"] == 2
    assert row["interaction_phase_coverage"]["planning_invocations"] == 1
    assert row["interaction_phase_coverage"]["coding_invocations"] == 1
    assert row["interaction_phase_coverage"]["all_phase_calls_retained"] is None
    coverage = row["usage_coverage_observation"]
    assert coverage["complete_usage_blocked"] is False
    assert coverage["semantic_decision_coverage"]["reported_provider_dispatches"] == 1
    assert coverage["semantic_decision_coverage"]["authoritative_usage_coverage_complete"] is True
    assert coverage["campaign_usage_complete"] is None
    assert "999999" not in json.dumps(row)


@pytest.mark.parametrize("key", ["semantic_decision_dialogue_expected", "semantic_decision_owner_binding_expected",
    "semantic_decision_typed_nomination_expected"])
def test_invalid_selected_semantic_expected_mode_withholds_complete_native_subset(key):
    row = _interaction_trial([_identified_invocation()], **{key: 1})
    assert row["tokens"]["total_tokens"] is None and row["provider_calls"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 120
    assert "invalid_expected_decision_mode" in {issue["reason"]
        for issue in row["usage_coverage_observation"]["semantic_decision_coverage"]["issues"]}


def test_missing_expected_semantic_owner_binding_cannot_be_repaired_by_complete_native_receipt():
    native = _identified_invocation()
    row = _interaction_trial([], semantic_decision_owner_binding_expected=True,
        semantic_decision_dialogues=[_decision_dialogue(native)])
    assert row["known_token_subtotals"]["total_tokens"] == 120
    assert row["tokens"]["total_tokens"] is None and row["provider_calls"] is None
    assert row["usage_coverage_observation"]["semantic_decision_coverage"]["owner_binding_coverage_complete"] is False


def test_unsupported_semantic_provider_keeps_native_observation_without_complete_outer_total():
    native = _grok_invocation(complete=True)
    native.update(schema="router-implementation-invocation@1", model="grok-4.7", purpose="coding",
        workspace="/authored/worktree", model_prompt_sha256="a" * 64, model_prompt_bytes=123)
    native["native_rollout_usage"].update(usage_observation_sha256="c" * 64, observed_final_usage_records=1)
    row = _interaction_trial([], semantic_decision_dialogues=[_decision_dialogue(native)])
    assert row["known_token_subtotals"]["total_tokens"] == 42
    assert row["tokens"]["total_tokens"] is None and row["provider_calls"] is None
    assert row["observed_provider_invocations"] == 1
    assert "unsupported_dialogue_provider_usage" in {issue["reason"]
        for issue in row["usage_coverage_observation"]["semantic_decision_coverage"]["issues"]}


def test_incomplete_plain_collection_with_good_receipts_blocks_only_complete_public_usage():
    native = _identified_invocation()
    native["native_rollout_usage"]["dollar_cost"] = 1
    row = _interaction_trial([native], provider_receipt_collection=_reported_collection(complete=False),
        usage_coverage_observation={"complete_usage_blocked": False})
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["reported_cost_usd"] is None and row["provider_calls"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 120
    assert row["known_reported_cost_subtotal_usd"] == 1 and row["observed_provider_invocations"] == 1
    assert row["native_session_accounting"]["complete_native_total_tokens"] == 120
    observation = row["usage_coverage_observation"]
    assert observation["provider_receipt_collection_status"] == "reported"
    assert observation["provider_receipt_collection"]["counts"]["oversized_logs"] == 1
    assert observation["complete_usage_blocked"] is True and observation["campaign_usage_complete"] is None


@pytest.mark.parametrize("explicit_marker", [True, False])
def test_clean_collection_never_repairs_unknown_attempt_marker_or_proves_campaign_coverage(explicit_marker):
    report = dict(provider_invocations=[_identified_invocation()], provider_receipt_collection=_reported_collection())
    if explicit_marker:
        report["unreceipted_provider_attempt"] = False
    row = comparison._trial(dict(schema=SUPERVISOR, arm="full", task="authored-task", model="pinned-model"),
        dict(trial="authored-trial", reward={"reward": 1}, supervisor=report),
        source=dict(path="authored-receipt", sha256="b" * 64))
    assert row["tokens"]["total_tokens"] == (120 if explicit_marker else None)
    assert row["known_token_subtotals"]["total_tokens"] == 120
    assert row["unreceipted_provider_attempt"] is (False if explicit_marker else None)
    observation = row["usage_coverage_observation"]
    assert observation["complete_usage_blocked"] is False
    assert observation["provider_receipt_collection"]["attempt_population_complete"] is None
    assert observation["campaign_usage_complete"] is None and row["whole_interaction_qualified"] is False


@pytest.mark.parametrize("gap", ["boolean_count", "authority", "complete_claim", "unknown_reason", "extra", "bound",
    "clean_unread_log", "clean_entry_overshoot", "clean_file_overshoot", "clean_line_overshoot"])
def test_invalid_reported_collection_never_qualifies_complete_usage_or_exports_raw_text(gap):
    collection = _reported_collection()
    if gap == "boolean_count":
        collection["counts"]["provider_observations"] = True
    elif gap == "authority":
        collection["provider_usage_authenticity_verified"] = True
    elif gap == "complete_claim":
        collection["attempt_population_complete"] = True
    elif gap == "unknown_reason":
        collection.update(status="incomplete", complete=False, reason_codes=["MUST NOT APPEAR"])
    elif gap == "extra":
        collection["transcript"] = "MUST NOT APPEAR"
    elif gap == "clean_unread_log":
        collection["counts"]["log_files_read"] = 0
    elif gap == "clean_entry_overshoot":
        collection["counts"]["entries_examined"] = 4097
    elif gap == "clean_file_overshoot":
        collection["counts"].update(entries_examined=129, log_files_seen=129, log_files_read=128)
    elif gap == "clean_line_overshoot":
        collection["counts"]["lines_examined"] = 131073
    else:
        collection["counts"]["entries_examined"] = 4098
    row = _interaction_trial([_identified_invocation()], provider_receipt_collection=collection)
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 120
    observation = row["usage_coverage_observation"]
    assert observation["provider_receipt_collection_status"] == "invalid"
    assert observation["provider_receipt_collection"] is None
    assert observation["complete_usage_blocked"] is True
    assert "MUST NOT APPEAR" not in json.dumps(row)


@pytest.mark.parametrize("different_artifact", [False, True])
def test_baseline_session_reuse_across_trials_is_not_additive(different_artifact):
    first = _baseline_native()
    second = copy.deepcopy(first)
    if different_artifact:
        second["sessions"][0]["sha256"] = "f" * 64
    rows = [_baseline_trial(first), _baseline_trial(second, identity="second-baseline", reward=0)]
    comparison._annotate_native_session_reuse(rows)
    assert [row["tokens"]["total_tokens"] for row in rows] == [110, 110]
    aggregate = comparison._aggregate(rows)
    assert aggregate["native_session_overlap_count"] == 1
    assert aggregate["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert aggregate["reported_cost_usd"] is None
    assert aggregate["known_token_subtotals"]["total_tokens"] == 220
    assert aggregate["observed_efficiency"]["total_tokens_per_official_pass"] is None
    failed = comparison._aggregate([row for row in rows if row["official_reward"] == 0])
    assert failed["trials"] == 1 and failed["native_session_overlap_count"] == 1
    assert failed["tokens"]["total_tokens"] is None and failed["reported_cost_usd"] is None


def test_baseline_and_supervisor_share_exact_native_identity_without_artifact_join():
    identity = "native-session-123456789"
    native = _baseline_native((identity,))
    native["sessions"][0]["sha256"] = "d" * 64
    rows = [_baseline_trial(native), _interaction_trial([_identified_invocation(artifact="different-artifact")])]
    comparison._annotate_native_session_reuse(rows)
    expected = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    assert [row["native_sessions_reused_across_comparison_trials"] for row in rows] == [[expected], [expected]]
    assert rows[0]["baseline_native_session_identity_observation"]["identity_sha256"] == [expected]
    assert [row["tokens"]["total_tokens"] for row in rows] == [110, 120]
    for row in rows:
        aggregate = comparison._aggregate([row])
        assert aggregate["native_session_overlap_count"] == 1
        assert aggregate["tokens"]["total_tokens"] is None
        assert aggregate["observed_efficiency"]["total_tokens_per_official_pass"] is None
    assert identity not in json.dumps(rows)


def test_duplicate_baseline_ids_withhold_totals_and_retain_raw_unreconciled_subtotal():
    row = _baseline_trial(_baseline_native(("duplicate-session", "duplicate-session")))
    observation = row["baseline_native_session_identity_observation"]
    assert observation["status"] == "invalid" and observation["inventory_complete"] is False
    assert observation["reason"] == "native_session_identity_duplicate"
    assert observation["observed_session_count"] == 2 and len(observation["identity_sha256"]) == 1
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["provider_calls"] is None and row["reported_cost_usd"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 220
    assert row["known_token_subtotals_basis"] == "raw_baseline_usage_unreconciled"
    assert row["known_reported_cost_subtotal_usd"] == 0.25
    assert row["reported_cost_accounting_basis"] == "raw_baseline_cost_subtotal_unreconciled"


@pytest.mark.parametrize("gap", ["missing", "boolean", "empty", "synthetic", "path", "partial", "artifact_only", "overlimit"])
def test_modern_baseline_invalid_identity_inventory_never_gains_complete_totals(gap):
    identity = "native-session-123456789"
    native = _baseline_native((identity,))
    session = native["sessions"][0]
    if gap in {"missing", "artifact_only"}:
        session.pop("session_id")
        if gap == "artifact_only":
            session.update(path="sessions/rollout-observed.jsonl", sha256="a" * 64)
    elif gap == "boolean":
        session["session_id"] = True
    elif gap == "empty":
        session["session_id"] = ""
    elif gap == "synthetic":
        session["session_id"] = "unidentified:sessions/rollout-observed.jsonl"
    elif gap == "path":
        session["session_id"] = "sessions/rollout-observed.jsonl"
    elif gap == "overlimit":
        native["sessions"] *= comparison.BASELINE_SESSION_LIMIT + 1
    else:
        extra = copy.deepcopy(session)
        extra.pop("session_id")
        native["sessions"].append(extra)
        native["usage"] = {key: value * 2 if value is not None else None for key, value in native["usage"].items()}
    row = _baseline_trial(native)
    observation = row["baseline_native_session_identity_observation"]
    assert observation["status"] == "invalid" and observation["inventory_complete"] is False
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["native_usage_complete_observed"] is False and row["reported_cost_usd"] is None
    if gap == "partial":
        expected = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        assert observation["identity_sha256"] == [expected]
        rows = [row, _interaction_trial([_identified_invocation()])]
        comparison._annotate_native_session_reuse(rows)
        assert rows[1]["native_sessions_reused_across_comparison_trials"] == [expected]
        assert comparison._aggregate([rows[1]])["tokens"]["total_tokens"] is None
    else:
        assert observation["identity_sha256"] == []
        assert comparison._native_session_keys(row) == set()


@pytest.mark.parametrize("gap", ["aggregate_usage", "session_usage", "boolean_usage", "missing_usage",
    "missing_completion", "unknown_completion", "malformed_completion"])
def test_current_baseline_usage_and_completion_bind_every_supplied_session(gap):
    native = _baseline_native()
    session = native["sessions"][0]
    if gap == "aggregate_usage":
        native["usage"]["input_tokens"] += 1
    elif gap == "session_usage":
        session["usage"]["cached_input_tokens"] = 101
    elif gap == "boolean_usage":
        session["usage"]["cached_input_tokens"] = True
    elif gap == "missing_usage":
        session.pop("usage")
    elif gap == "missing_completion":
        session.pop("task_complete_observed")
    elif gap == "malformed_completion":
        session["task_complete_observed"] = "true"
    else:
        session["task_complete_observed"] = None
    row = _baseline_trial(native)
    observation = row["baseline_native_session_identity_observation"]
    assert observation["inventory_complete"] is True
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["native_usage_complete_observed"] is not True and row["reported_cost_usd"] is None
    assert comparison._native_session_keys(row) == {("codex_cli", observation["identity_sha256"][0])}
    if gap in {"missing_completion", "unknown_completion", "malformed_completion"}:
        assert observation["usage_population_coherent"] is True
    else:
        assert observation["usage_population_coherent"] is not True


def test_historical_baseline_without_ids_stays_explicitly_unverified():
    native = dict(sessions=[{"task_complete_observed": True}], usage=dict(input_tokens=100,
        cached_input_tokens=60, output_tokens=10, total_tokens=110), reported_cost_usd=0.125)
    row = _baseline_trial(native)
    assert row["tokens"]["total_tokens"] == 110 and row["reported_cost_usd"] == 0.125
    assert row["native_token_accounting_basis"] == "legacy_baseline_identity_unverified"
    assert row["baseline_native_session_identity_observation"]["status"] == "unverified"
    assert comparison._native_session_keys(row) == set()
    assert row["whole_interaction_qualified"] is False


def test_declared_hex_native_id_is_hashed_exactly_and_supplied_annotations_are_replaced():
    identity = "a" * 64
    native = _baseline_native((identity,))
    row = _baseline_trial(native, baseline_native_session_identity_observation={"identity_sha256": ["f" * 64]},
                          native_sessions_reused_across_comparison_trials=["f" * 64])
    expected = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    assert comparison._native_session_keys(row) == {("codex_cli", expected)}
    row["native_sessions_reused_across_comparison_trials"] = [expected]
    comparison._annotate_native_session_reuse([row])
    assert row["native_sessions_reused_across_comparison_trials"] == []
    assert comparison._aggregate([row])["tokens"]["total_tokens"] == 110


def test_malformed_supplied_baseline_identity_projection_cannot_create_owner_keys():
    row = _baseline_trial(_baseline_native())
    row["baseline_native_session_identity_observation"]["observed_session_count"] = True
    assert comparison._native_session_keys(row) == set()
    row = _baseline_trial(_baseline_native())
    row["baseline_native_session_identity_observation"]["source_authenticity_verified"] = True
    assert comparison._native_session_keys(row) == set()
    row = _baseline_trial(_baseline_native())
    row["baseline_native_session_identity_observation"]["unknown"] = "must not qualify"
    assert comparison._native_session_keys(row) == set()


def test_identified_session_selects_latest_snapshot_once_and_keeps_failed_wrappers():
    early = _identified_invocation("planning", phase="planning", complete=False)
    late = _identified_invocation("coding", records=2, inputs=150, outputs=30)
    late.update(status="failed", error_type="ResponseDecodeError")
    for call in (early, late):
        call["native_rollout_usage"]["dollar_cost"] = 1
    row = _interaction_trial([early, late, late])
    assert row["tokens"] == dict(input_tokens=150, cached_input_tokens=75, output_tokens=30, total_tokens=180)
    assert row["observed_provider_invocations"] == 2
    assert row["native_session_accounting"]["native_session_count"] == 1
    assert row["native_session_accounting"]["failed_invocation_count"] == 1
    assert row["native_session_reconciliation_available"] is True
    assert row["native_token_accounting_basis"] == "native_session_reconciled"
    assert row["input_token_scope"] == "native_input_including_cache"
    assert row["interaction_phase_coverage"]["planning_observed"] is True
    assert row["interaction_phase_coverage"]["coding_observed"] is True
    assert row["whole_interaction_qualified"] is False
    assert row["reported_cost_usd"] is None
    assert row["known_reported_cost_subtotal_usd"] == 2
    assert row["reported_cost_accounting_basis"] == "raw_invocation_cost_subtotal_unreconciled"


def test_identical_session_snapshot_in_distinct_wrappers_is_not_charged_twice():
    one = _identified_invocation("one", artifact="same")
    two = _identified_invocation("two", artifact="same")
    row = _interaction_trial([one, two])
    assert row["tokens"]["total_tokens"] == 120
    assert row["native_session_accounting"]["supplied_invocation_count"] == 2
    assert row["shared_native_session_count"] == 1


@pytest.mark.parametrize("gap", ["regression", "equal_conflict", "missing_identity", "malformed_identity",
    "missing_model", "foreign_model", "foreign_workspace", "unreceipted", "unknown_marker", "failed_retry"])
def test_modern_native_gaps_withhold_complete_trial_totals_without_legacy_fallback(gap):
    one = _identified_invocation("one")
    two = _identified_invocation("two", records=2, inputs=150, outputs=30)
    marker = False
    if gap == "regression":
        two["native_rollout_usage"]["usage"].update(input_tokens=80, cached_input_tokens=40, output_tokens=20, total_tokens=100)
    elif gap == "equal_conflict":
        two["native_rollout_usage"]["observed_token_count_records"] = 1
    elif gap == "missing_identity":
        two["native_rollout_usage"].pop("thread_id")
    elif gap == "malformed_identity":
        two["native_rollout_usage"]["thread_id"] = True
    elif gap == "missing_model":
        two.pop("model")
    elif gap == "foreign_model":
        two["model"] = "foreign-model"
    elif gap == "foreign_workspace":
        two["workspace"] = "/authored/other-worktree"
    elif gap == "unreceipted":
        marker = {"phase": "coding"}
    elif gap == "unknown_marker":
        marker = None
    else:
        two.update(status="failed", error_type="TimeoutError")
        two["native_rollout_usage"].update(thread_id="failed-retry-session-123456789", task_complete_observed=False)
    row = _interaction_trial([one, two], marker=marker)
    assert row["native_token_accounting_basis"] == "native_session_reconciled"
    assert all(value is None for value in row["tokens"].values())
    assert row["native_usage_complete_observed"] is False
    assert row["observed_provider_invocations"] == 2
    assert row["reported_cost_usd"] is None
    if gap == "failed_retry":
        assert row["known_token_subtotals"]["total_tokens"] == 300
        assert row["native_session_accounting"]["failed_invocation_count"] == 1
    if gap == "unknown_marker":
        assert row["unreceipted_provider_attempt"] is None


@pytest.mark.parametrize("provider", ["foreign-provider", None])
def test_native_codex_thread_with_wrong_or_missing_provider_cannot_regain_legacy_totals(provider):
    call = _identified_invocation()
    if provider is None:
        call.pop("provider")
    else:
        call["provider"] = provider
    row = _interaction_trial([call])
    assert row["native_token_accounting_basis"] == "native_session_reconciled"
    assert all(value is None for value in row["tokens"].values())
    assert row["native_usage_complete_observed"] is False
    assert row["whole_interaction_qualified"] is False


@pytest.mark.parametrize("invocation_count", [1, 2])
def test_modern_codex_receipts_with_no_thread_ids_cannot_regain_legacy_totals(invocation_count):
    calls = [_identified_invocation("missing-identity-" + str(index)) for index in range(invocation_count)]
    for call in calls:
        call["native_rollout_usage"].pop("thread_id")
        call["native_rollout_usage"]["dollar_cost"] = 1
    row = _interaction_trial(calls)
    assert row["native_token_accounting_basis"] == "native_session_reconciled"
    assert row["tokens"] == dict.fromkeys(comparison.TOKENS)
    assert row["native_usage_complete_observed"] is False
    assert row["observed_provider_invocations"] == invocation_count
    assert row["native_session_accounting"]["native_session_count"] == 0
    assert row["reported_cost_usd"] is None
    assert row["known_reported_cost_subtotal_usd"] == invocation_count
    assert row["observed_counts_are_lower_bounds"] is False
    assert row["whole_interaction_qualified"] is False


def test_recomputed_context_ignores_forged_summary_and_keeps_local_counts_out_of_usage():
    call = _identified_invocation()
    call["local_token_count"] = 1
    row = _interaction_trial([call], context_delivery_observation={"native_invocations": {"complete_native_total_tokens": 1}})
    assert row["tokens"]["total_tokens"] == 120
    assert row["context_delivery_observation"]["native_invocations"]["complete_native_total_tokens"] == 120
    assert row["native_session_accounting"]["local_tokenizer_counts_used_as_native_usage"] is False
    assert row["native_session_accounting_source_sha256"]
    assert row["context_delivery_observation_source_sha256"]


@pytest.mark.parametrize("codex_complete", [True, False])
def test_identified_mixed_provider_input_is_normalized_without_recharging_cache(codex_complete):
    grok = _grok_invocation(complete=True)
    grok.update(schema="router-implementation-invocation@1", model="grok-4.7", purpose="coding",
                workspace="/authored/worktree", model_prompt_sha256="a" * 64, model_prompt_bytes=123)
    grok["native_rollout_usage"].update(usage_observation_sha256="c" * 64, observed_final_usage_records=1)
    row = _interaction_trial([_identified_invocation(complete=codex_complete), grok])
    expected = dict(input_tokens=140, cached_input_tokens=80, output_tokens=22, total_tokens=162)
    assert row["known_token_subtotals"] == expected
    assert row["tokens"] == (expected if codex_complete else dict.fromkeys(comparison.TOKENS))
    assert row["input_token_scope"] == "native_input_including_cache"
    assert row["cache_included_in_input"] is None
    assert row["observed_counts_are_lower_bounds"] is False
    assert row["invocations"][1]["tokens"]["input_tokens"] == 10


def test_legacy_receipts_keep_explicitly_unreconciled_descriptive_totals():
    row = _interaction_trial([_invocation()])
    assert row["tokens"]["total_tokens"] == 22
    assert row["native_token_accounting_basis"] == "legacy_invocation_sum_unreconciled"
    assert row["native_session_reconciliation_available"] is False
    assert row["native_session_accounting"] is None
    assert row["whole_interaction_qualified"] is False


def test_shared_native_session_across_retained_trials_withholds_aggregate_tokens_and_cost():
    first = _interaction_trial([_identified_invocation("first")])
    second = _interaction_trial([_identified_invocation("second", records=2, inputs=150, outputs=30)])
    totals = comparison._aggregate([first, second])
    assert totals["native_session_overlap_count"] == 1
    assert all(value is None for value in totals["tokens"].values())
    assert totals["reported_cost_usd"] is None
    assert totals["known_token_subtotals"]["total_tokens"] == 300
    assert totals["known_token_subtotals_basis"] == "per_trial_observations_unreconciled_across_shared_sessions"
    assert totals["observed_efficiency"]["total_tokens_per_official_pass"] is None
    assert "native_session_reused_across_retained_trials" in totals["observed_efficiency"]["token_ratio_unavailable_reasons"]


def test_same_native_session_cannot_be_charged_to_distinct_comparison_arms(tmp_path):
    paths = []
    for arm in ("full", "no-index"):
        trial = dict(trial="trial-" + arm, reward={"reward": 1}, supervisor=dict(
            provider_invocations=[_identified_invocation("call-" + arm)], unreceipted_provider_attempt=False))
        paths.append(_receipt(tmp_path, arm + ".json", SUPERVISOR, [trial], arm))
    result = collect(baseline=_baseline(tmp_path), supervisors=paths)
    for arm in ("full", "no-index"):
        totals = result["arm_totals"][arm]
        assert totals["native_session_overlap_count"] == 1
        assert totals["tokens"]["total_tokens"] is None
        assert totals["observed_efficiency"]["total_tokens_per_official_pass"] is None
    assert len(result["rows"][1]["native_sessions_reused_across_comparison_trials"]) == 1


def test_observed_efficiency_includes_failed_attempts_and_deduplicates_receipts(tmp_path):
    trials = []
    for index, (reward, cost) in enumerate(((1, 0.5), (0, 0.25), (1, 0.75))):
        invocation = _invocation(f"call-{index}")
        invocation["native_rollout_usage"]["dollar_cost"] = cost
        if reward == 0:
            invocation.update(status="failed", error_type="ResponseDecodeError")
        trials.append({"trial": f"trial-{index}", "reward": {"reward": reward},
            "supervisor": {"task_completed": True, "provider_invocations": [invocation, invocation]}})
    receipt = _receipt(tmp_path, "efficiency.json", SUPERVISOR, trials, "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt, receipt])
    totals = result["arm_totals"]["full"]
    efficiency = totals["observed_efficiency"]
    assert totals["trials"] == 3 and totals["observed_official_passes"] == 2
    assert totals["tokens"]["total_tokens"] == 66
    assert totals["reported_cost_usd"] == 1.5
    assert efficiency["total_tokens_per_official_pass"] == 33
    assert efficiency["reported_cost_usd_per_official_pass"] == 0.75
    assert efficiency["token_ratio_unavailable_reasons"] == []
    assert efficiency["cost_ratio_unavailable_reasons"] == []
    assert efficiency["numerator_scope"] == "all_retained_attempts_in_aggregate"
    assert efficiency["cache_included_in_input"] is True
    assert efficiency["causal_claim"] is False and efficiency["billing_total_verified"] is False
    assert result["unsuccessful_attempt_costs"]["full"]["tokens"]["total_tokens"] == 22
    assert "| Supervisor: full indexed context | 3 | 2 | 0 | 0 | 33.000 | 0.750 |" in markdown(result)
    assert "do not establish causal token savings or verified billing" in markdown(result)


@pytest.mark.parametrize("gap,token_reason,cost_reason", [
    ("unknown_reward", "unknown_official_reward", "unknown_official_reward"),
    ("zero_passes", "no_official_passes", "no_official_passes"),
    ("partial_reward", "no_official_passes", "no_official_passes"),
    ("incomplete_session", "incomplete_native_token_usage", "incomplete_reported_provider_cost"),
    ("missing_total", "incomplete_native_token_usage", None),
    ("unreceipted_attempt", "incomplete_native_token_usage", "incomplete_reported_provider_cost"),
    ("missing_cost", None, "incomplete_reported_provider_cost"),
])
def test_observed_efficiency_refuses_unknown_denominators_and_incomplete_numerators(
        tmp_path, gap, token_reason, cost_reason):
    trials = []
    for index in range(2):
        invocation = _invocation(f"call-{index}")
        invocation["native_rollout_usage"]["dollar_cost"] = 0.25
        trials.append({"trial": f"trial-{index}", "reward": {"reward": 1},
            "supervisor": {"task_completed": True, "provider_invocations": [invocation]}})
    changed = trials[1]
    native = changed["supervisor"]["provider_invocations"][0]["native_rollout_usage"]
    if gap == "unknown_reward":
        changed.pop("reward")
    elif gap in {"zero_passes", "partial_reward"}:
        for trial in trials:
            trial["reward"]["reward"] = 0.5 if gap == "partial_reward" else 0
    elif gap == "incomplete_session":
        native["task_complete_observed"] = False
    elif gap == "missing_total":
        native["usage"].pop("total_tokens")
    elif gap == "unreceipted_attempt":
        changed["supervisor"]["unreceipted_provider_attempt"] = {"phase": "coding"}
    elif gap == "missing_cost":
        native.pop("dollar_cost")
    receipt = _receipt(tmp_path, "efficiency-gap.json", SUPERVISOR, trials, "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    totals = result["arm_totals"]["full"]
    efficiency = totals["observed_efficiency"]
    assert efficiency["token_ratio_unavailable_reasons"] == ([token_reason] if token_reason else [])
    assert efficiency["cost_ratio_unavailable_reasons"] == ([cost_reason] if cost_reason else [])
    assert efficiency["total_tokens_per_official_pass"] == (None if token_reason else 22)
    assert efficiency["reported_cost_usd_per_official_pass"] == (None if cost_reason else 0.25)
    assert totals["known_token_subtotals"]["total_tokens"] == (22 if gap == "missing_total" else 44)
    assert "Total tokens / official pass" in markdown(result)


def test_setup_abort_does_not_turn_unknown_cost_into_zero_per_pass(tmp_path):
    receipt = _receipt(tmp_path, "setup-abort-efficiency.json", SUPERVISOR, [
        {"trial": "passed", "reward": {"reward": 1},
         "supervisor": {"provider_invocations": [_invocation()]}},
        {"trial": "setup-abort", "exception_type": "SetupError", "supervisor": None},
    ], "no-index")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    totals = result["arm_totals"]["no-index"]
    efficiency = totals["observed_efficiency"]
    assert totals["observed_official_passes"] == 1 and totals["unknown_reward_trials"] == 1
    assert totals["known_token_subtotals"]["total_tokens"] == 22
    assert efficiency["total_tokens_per_official_pass"] is None
    assert efficiency["token_ratio_unavailable_reasons"] == [
        "unknown_official_reward", "incomplete_native_token_usage"]
    assert efficiency["reported_cost_usd_per_official_pass"] is None


def test_observed_efficiency_preserves_mixed_provider_total_and_cache_semantics(tmp_path):
    codex, grok = _invocation(), _grok_invocation(complete=True)
    codex["native_rollout_usage"]["dollar_cost"] = 0
    grok["native_rollout_usage"]["dollar_cost"] = 0
    receipt = _receipt(tmp_path, "mixed-provider-efficiency.json", SUPERVISOR, [{
        "trial": "mixed-provider", "reward": {"reward": 1},
        "supervisor": {"provider_invocations": [codex, grok]},
    }], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    efficiency = result["arm_totals"]["full"]["observed_efficiency"]
    assert row["tokens"] == dict(input_tokens=30, cached_input_tokens=45, output_tokens=4, total_tokens=64)
    assert efficiency["total_tokens_per_official_pass"] == 64  # Native totals, without adding cache twice.
    assert efficiency["reported_cost_usd_per_official_pass"] == 0  # Explicit zero is a known cost.
    assert efficiency["cache_included_in_input"] is None
    assert [call["cache_included_in_input"] for call in row["invocations"]] == [True, False]
    assert [call["token_count_scope"] for call in row["invocations"]] == [
        "native_session_cumulative", "native_final_envelope"]


def test_native_baseline_efficiency_uses_all_failed_session_costs(tmp_path):
    baseline = _baseline(tmp_path)
    receipt = json.loads(baseline.read_text())
    failed = copy.deepcopy(receipt["trials"][0])
    failed.update(trial="failed-native", reward={"reward": 0})
    failed["raw_usage"]["sessions"][0]["session_id"] = "independent-failed-session"
    receipt["trials"].append(failed)
    receipt["trial_count"] = 2
    for trial in receipt["trials"]:
        trial["raw_usage"]["reported_cost_usd"] = 0.125
    baseline.write_text(json.dumps(receipt))
    result = collect(baseline=baseline, supervisors=[])
    efficiency = result["arm_totals"]["native-codex"]["observed_efficiency"]
    assert efficiency["total_tokens_per_official_pass"] == 220
    assert efficiency["reported_cost_usd_per_official_pass"] == 0.25
    assert efficiency["cache_included_in_input"] is True
    assert "| Native Codex harness | 2 | 1 | 0 | 0 | 220.000 | 0.250 |" in markdown(result)


@pytest.mark.parametrize("complete", [True, False, None])
def test_grok_session_completion_does_not_replace_usage_completeness(tmp_path, complete):
    invocation = _grok_invocation(complete=complete)
    receipt = _receipt(tmp_path, "grok.json", SUPERVISOR, [{
        "trial": "grok", "reward": {"reward": 1}, "supervisor": {
            "task_completed": True, "provider_invocations": [invocation, invocation]}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    call, = row["invocations"]
    assert row["official_reward"] == 1 and row["native_task_completed"] is True
    assert call["native_task_complete_observed"] is True
    assert call["native_usage_complete_observed"] is complete
    assert row["native_usage_complete_observed"] is complete
    assert row["tokens"]["total_tokens"] == (42 if complete is True else None)
    assert result["arm_totals"]["full"]["tokens"]["total_tokens"] == (42 if complete is True else None)
    assert row["known_token_subtotals"]["total_tokens"] == 42
    assert row["provider_calls"] == 1
    assert call["token_count_scope"] == "native_final_envelope"
    assert row["cache_included_in_input"] is False
    assert result["cached_tokens_are_already_in_input"] is None  # Mixed Codex/Grok semantics.
    assert row["reported_cost_usd"] is None
    efficiency = result["arm_totals"]["full"]["observed_efficiency"]
    assert efficiency["total_tokens_per_official_pass"] == (42 if complete is True else None)
    if complete is not True:
        assert "42 (observed)" in markdown(result)
        assert "≥ 42" not in markdown(result)


@pytest.mark.parametrize("mutation", ["missing_flag", "string_flag", "integer_flag", "wrong_schema",
    "wrong_scope", "wrong_provider", "missing_write", "missing_total", "inconsistent_total", "cache_semantics",
    "usage_unavailable"])
def test_grok_counter_or_completeness_contract_gaps_do_not_create_totals(tmp_path, mutation):
    invocation = _grok_invocation(complete=True)
    native = invocation["native_rollout_usage"]
    if mutation == "missing_flag": native.pop("usage_complete_observed")
    elif mutation == "string_flag": native["usage_complete_observed"] = "true"
    elif mutation == "integer_flag": native["usage_complete_observed"] = 1
    elif mutation == "wrong_schema": native["schema"] = "native-codex-cumulative-usage@1"
    elif mutation == "wrong_scope": native["totals_scope"] = "arbitrary_scope"
    elif mutation == "wrong_provider": invocation["provider"] = "codex_cli"
    elif mutation == "missing_write": native["usage"].pop("cache_write_input_tokens")
    elif mutation == "missing_total": native["usage"].pop("total_tokens")
    elif mutation == "inconsistent_total": native["usage"]["total_tokens"] += 1
    elif mutation == "cache_semantics": native["cache_included_in_input"] = True
    elif mutation == "usage_unavailable": native["usage_available"] = False
    receipt = _receipt(tmp_path, "grok-gap.json", SUPERVISOR, [{
        "trial": "grok-gap", "reward": {"reward": 1}, "supervisor": {
            "task_completed": True, "provider_invocations": [invocation]}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert all(value is None for value in row["tokens"].values())
    assert row["native_usage_complete_observed"] is None
    assert row["known_token_subtotals"]["input_tokens"] == 10
    assert all(f"≥ {value} (observed)" not in markdown(result) for value in (10, 30, 2, 42, 43))


def test_grok_incomplete_call_prevents_pair_total_and_unreceipted_attempt_remains_unknown(tmp_path):
    planning = _grok_invocation("planning", True)
    planning["phase"] = "planning"
    coding = _grok_invocation("coding", False)
    report = {"provider_invocations": [planning, coding]}
    receipt = _receipt(tmp_path, "mixed-completeness.json", SUPERVISOR, [{
        "trial": "mixed-completeness", "supervisor": report}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["native_usage_complete_observed"] is False
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 84
    coding["native_rollout_usage"]["usage_complete_observed"] = True
    report["unreceipted_provider_attempt"] = {"phase": "coding"}
    receipt = _receipt(tmp_path, "unreceipted-grok.json", SUPERVISOR, [{
        "trial": "unreceipted-grok", "supervisor": report}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["native_usage_complete_observed"] is None and row["tokens"]["total_tokens"] is None


def test_unidentified_legacy_receipt_does_not_inherit_codex_contract(tmp_path):
    invocation = _invocation()
    invocation.pop("provider")
    invocation["native_rollout_usage"].pop("schema")
    receipt = _receipt(tmp_path, "unknown-provider.json", SUPERVISOR, [{
        "trial": "unknown-provider", "supervisor": {"provider_invocations": [invocation]}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["native_usage_complete_observed"] is None
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 22


def test_stream_progress_survives_comparison_without_becoming_usage_or_completion():
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _invocations
    parser = GrokStreamParser()
    parser.feed(b'{"type":"tool_call","toolName":"read_file","args":{"path":"private"}}\n',
                elapsed_seconds=0.1)
    progress = parser.snapshot()
    row = {"invocation_id": "call", "phase": "coding", "status": "failed",
           "native_grok_progress": progress}
    observed, = _invocations({"provider_invocations": [row]})
    assert observed["native_grok_progress"] == progress
    assert observed["native_task_complete_observed"] is None
    assert all(value is None for value in observed["tokens"].values())
    assert "private" not in json.dumps(observed)
    row["native_grok_progress"] = {**progress, "raw_text": "private"}
    assert _invocations({"provider_invocations": [row]})[0]["native_grok_progress"] is None


def test_bounded_stream_usage_survives_comparison_without_replacing_final_accounting():
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from ipfs_accelerate_py.cli_runtime.grok_stream_usage import MAX_USAGE_EVENTS
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _invocations
    parser = GrokStreamParser()
    for index in range(MAX_USAGE_EVENTS + 2):
        parser.feed((json.dumps({"type": "usage", "messageId": "private-message",
            "signature": "private-signature", "stopReason": "tool_use",
            "usage": {"input_tokens": 10000 + index, "cache_read_input_tokens": 9000,
                      "output_tokens": 10, "private": "private-content"}}) + "\n").encode(),
            elapsed_seconds=index / 10)
    progress = parser.snapshot()
    invocation = _grok_invocation("streamed", True)
    invocation["native_grok_progress"] = progress
    observed, = _invocations({"provider_invocations": [invocation]})
    events = observed["native_grok_progress"]["usage_observations"]
    assert events["retained_events"] == MAX_USAGE_EVENTS and events["omitted_events"] == 2
    assert events["rows"][0]["usage"]["input_tokens"] == 10000
    assert events["rows"][-1]["event_ordinal"] == MAX_USAGE_EVENTS
    assert events["summation_permitted"] is False
    assert observed["tokens"]["total_tokens"] == 42  # Only the separate final-envelope contract.
    assert "private" not in json.dumps(observed)
    # Existing receipts lack event history; accepting them must not invent it.
    legacy = dict(progress, schema="native-grok-stream-progress@1")
    legacy.pop("usage_observations")
    invocation["native_grok_progress"] = legacy
    historical, = _invocations({"provider_invocations": [invocation]})
    assert historical["native_grok_progress"] == legacy
    assert "usage_observations" not in historical["native_grok_progress"]


def test_symbolic_coding_does_not_invent_provider_calls_or_learned_proof(tmp_path):
    receipt = _receipt(tmp_path, "symbolic.json", SUPERVISOR, [{
        "trial": "symbolic", "reward": {"reward": 1}, "supervisor": {
            "task_completed": True, "provider_invocations": [_invocation()],
            "implementation_route": "doctor_contract_candidate",
            "doctor_dispatch": {"task_cid": "task", "provider_calls": 0,
                "contract_workflow": {"proof": {"status": "proved_local_contract"}}},
            "doctor_invocations": [{"schema": "native-doctor-contract-candidate-materialization@1",
                "task_cid": "task", "status": "candidate_materialized"}],
            "initial_context": {"codebase_autoencoder": {
                "sample_count": 3, "epochs_completed": 24, "training_elapsed_seconds": 1.2,
                "metrics": {"native_kernel_calls": 24, "before_reconstruction_loss": 0.2,
                    "after_reconstruction_loss": 0.1, "holdout_evaluated": False}}},
            "context": {"new_autoencoder_training_steps": 0},
            "post_publication_proof_index": {"status": "observed", "active_receipt_ids": [],
                "invalidated_receipt_ids": ["old-proof"]}}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["provider_calls"] == 1 and row["tokens"]["total_tokens"] == 22
    assert row["doctor_observation"]["candidate_materializations_observed"] == 1
    assert row["doctor_observation"]["local_contract_proof_status"] == "proved_local_contract"
    assert row["code_learning_observation"]["new_training_steps_after_admission"] == 0
    assert row["code_learning_observation"]["formal_translation_authority"] is False
    assert row["code_learning_observation"]["holdout_evaluated"] is False
    assert row["post_publication_proof_index"]["invalidated_receipts"] == 1
    assert row["post_publication_proof_index"]["active_receipts"] == 0


def test_retains_failed_planning_and_setup_without_cache_double_count(tmp_path):
    failure = _receipt(
        tmp_path,
        "failed.json",
        SUPERVISOR,
        [
            {
                "trial": "failed-plan",
                "reward": {"reward": 0},
                "durations_seconds": {"agent_execution": 10},
                "supervisor": {
                    "task_completed": False,
                    "provider_invocations": [_invocation()],
                    "error": {"type": "ContractError", "message": "MUST NOT APPEAR"},
                },
            },
            {"trial": "setup-abort", "exception_type": "SetupError", "supervisor": None},
        ],
        "no-index",
    )
    result = collect(baseline=_baseline(tmp_path), supervisors=[failure])
    assert result["retained_trial_count"] == 3
    baseline, failed, setup = result["rows"]
    assert baseline["tokens"]["total_tokens"] == 110
    assert baseline["native_task_completed"] is None
    assert failed["tokens"]["total_tokens"] == 22
    assert failed["native_task_completed"] is False
    assert setup["tokens"]["total_tokens"] is None
    assert setup["provider_calls"] is None
    costs = result["unsuccessful_attempt_costs"]["no-index"]
    assert costs["tokens"]["total_tokens"] is None
    assert costs["known_token_subtotals"]["total_tokens"] == 22
    assert costs["unknown_token_trials"]["total_tokens"] == 1
    assert "MUST NOT APPEAR" not in json.dumps(result)
    assert "setup-abort" in markdown(result)
    assert not result["benchmark_advantage_claimed"]


def test_missing_invocation_preserves_known_cost_but_not_complete_total(tmp_path):
    receipt = _receipt(
        tmp_path,
        "partial.json",
        SUPERVISOR,
        [
            {
                "trial": "partial",
                "reward": {"reward": 0},
                "supervisor": {
                    "provider_invocations": [_invocation(), _invocation()],
                    "unreceipted_provider_attempt": {"phase": "coding"},
                },
            }
        ],
        "full",
    )
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["observed_provider_invocations"] == 1
    assert row["provider_calls"] is None
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 22


@pytest.mark.parametrize("completion", [False, None])
def test_interrupted_or_legacy_usage_is_only_an_observed_subtotal(tmp_path, completion):
    coding = _invocation("interrupted-coding")
    coding.update(phase="coding", status="failed", error_type="TimeoutExpired", timeout_seconds=45)
    coding["native_rollout_usage"]["usage"]["total_tokens"] = 244688
    coding["native_rollout_usage"]["dollar_cost"] = 0.75
    if completion is None:
        coding["native_rollout_usage"].pop("task_complete_observed")
    else:
        coding["native_rollout_usage"]["task_complete_observed"] = completion
    planning = _invocation("planning")
    planning["native_rollout_usage"]["usage"]["total_tokens"] = 21244
    planning["native_rollout_usage"]["dollar_cost"] = 0.25
    receipt = _receipt(tmp_path, "interrupted.json", SUPERVISOR, [{
        "trial": "interrupted", "reward": {"reward": 0},
        "agent_context": {"metadata": {"runtime_archive_sha256": "a" * 64,
            "private_notes": "MUST NOT APPEAR"}},
        "supervisor": {"task_completed": False, "provider_invocations": [planning, coding]},
    }], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert row["provider_calls"] == row["observed_provider_invocations"] == 2
    assert row["native_usage_complete_observed"] is completion
    assert all(value is None for value in row["tokens"].values())
    assert row["known_token_subtotals"]["total_tokens"] == 265932
    call = row["invocations"][1]
    assert call["native_task_complete_observed"] is completion
    assert call["tokens"]["total_tokens"] is None
    assert call["known_token_subtotals"]["total_tokens"] == 244688
    assert call["error_type"] == "TimeoutExpired" and call["timeout_seconds"] == 45
    assert row["reported_cost_usd"] is None
    assert row["known_reported_cost_subtotal_usd"] == 1
    costs = result["unsuccessful_attempt_costs"]["full"]
    assert costs["tokens"]["total_tokens"] is None
    assert costs["known_token_subtotals"]["total_tokens"] == 265932
    assert costs["unknown_token_trials"]["total_tokens"] == 1
    assert costs["reported_cost_usd"] is None and costs["known_reported_cost_subtotal_usd"] == 1
    assert result["arm_totals"]["full"]["tokens"]["total_tokens"] is None
    assert row["runtime_archive_sha256"] == "a" * 64
    assert len(result["comparison_source_sha256"]) == 64
    rendered = markdown(result)
    assert "≥ 265932 (observed)" in rendered and "≥ 244688 (observed)" in rendered
    assert "complete totals remain unknown" in rendered
    assert "TimeoutExpired | 45" in rendered and "a" * 64 in rendered
    assert "MUST NOT APPEAR" not in json.dumps(result)


def test_legacy_baseline_without_session_completion_keeps_only_observed_usage(tmp_path):
    baseline = _baseline(tmp_path)
    value = json.loads(baseline.read_text())
    value["trials"][0]["raw_usage"]["sessions"][0].pop("task_complete_observed")
    baseline.write_text(json.dumps(value))
    result = collect(baseline=baseline, supervisors=[])
    row = result["rows"][0]
    assert row["official_reward"] == 1
    assert row["native_usage_complete_observed"] is None
    assert row["tokens"]["total_tokens"] is None
    assert row["known_token_subtotals"]["total_tokens"] == 110
    assert result["arm_totals"]["native-codex"]["unknown_token_trials"]["total_tokens"] == 1
    assert "≥ 110 (observed)" in markdown(result)


def test_completed_native_session_can_retain_failed_postprocessing_cost(tmp_path):
    invocation = {**_invocation(), "status": "failed", "error_type": "ResponseDecodeError"}
    receipt = _receipt(tmp_path, "decode-error.json", SUPERVISOR, [{"trial": "decode-error",
        "reward": {"reward": 0}, "supervisor": {"provider_invocations": [invocation]}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert row["native_usage_complete_observed"] is True
    assert row["tokens"]["total_tokens"] == row["known_token_subtotals"]["total_tokens"] == 22
    assert row["invocations"][0]["error_type"] == "ResponseDecodeError"
    assert result["unsuccessful_attempt_costs"]["full"]["tokens"]["total_tokens"] == 22


def test_conflicting_same_invocation_is_rejected(tmp_path):
    changed = _invocation()
    changed["status"] = "failed"
    receipt = _receipt(
        tmp_path,
        "conflict.json",
        SUPERVISOR,
        [
            {
                "trial": "conflict",
                "supervisor": {"provider_invocations": [_invocation(), changed]},
            }
        ],
        "full",
    )
    with pytest.raises(ValueError, match="conflicting provider"):
        collect(baseline=_baseline(tmp_path), supervisors=[receipt])


def test_duplicate_retained_trial_counted_once_and_empty_receipt_preserved(tmp_path):
    receipt = _receipt(tmp_path, "empty.json", SUPERVISOR, [], "full")
    duplicate = _receipt(tmp_path, "one.json", SUPERVISOR, [{"trial": "one"}], "no-index")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt, duplicate, duplicate])
    assert result["retained_trial_count"] == 2
    assert len(result["receipts_without_trials"]) == 1
    assert len(result["rows"][1]["source_receipts"]) == 2
    assert result["rows"][1]["tokens"]["total_tokens"] is None


def test_context_preparation_is_not_worker_consumption_or_repair_evidence(tmp_path):
    receipt = _receipt(tmp_path, "indexed.json", SUPERVISOR, [{
        "trial": "indexed",
        "supervisor": {
            "phases": {"context_seconds": 101.0},
            "context": {"learned_embeddings": True, "native_fact_rows_replayed": 426,
                        "full_capsules": 531, "worker_capsules": 2,
                        "worker_semantic_bytes": 28388,
                        "doctor_repair": {"status": "abstained", "raw": "MUST NOT APPEAR"}},
        },
    }], "full")
    comparison = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = comparison["rows"][1]
    assert row["context_seconds"] == 101
    assert row["indexed_context_observation"]["doctor_repair_status"] == "abstained"
    assert row["indexed_context_observation"]["context_preparation_is_dispatch_evidence"] is False
    assert row["official_reward"] is None
    assert row["native_task_completed"] is None
    assert "MUST NOT APPEAR" not in json.dumps(comparison)
    assert "do not by themselves prove worker consumption" in markdown(comparison)


def test_full_lifecycle_costs_and_actual_input_observations_are_separate(tmp_path):
    invocation = {**_invocation("coding-call"), "phase": "coding",
        "native_prompt_bytes": 12000, "router_prompt_bytes": 9400, "model_prompt_bytes": 9900,
        "workspace_advisory_bytes": 500, "native_prompt_sha256": "a" * 64,
        "router_prompt_sha256": "b" * 64, "model_prompt_sha256": "c" * 64,
        "semantic_translation": {"schema": "supervisor-semantic-router-encoding@1",
            "translation_cid": "cid:translation", "native_prompt_bytes": 12000,
            "provider_prompt_bytes": 9000, "identifier_mappings": 12, "identifier_occurrences": 24,
            "freshness_checked": True, "raw_prompt": "MUST NOT APPEAR"},
        "doctor_residual_context": {"schema": "doctor-residual-router-context@1",
            "context_cid": "cid:residual", "native_capsule_id": "cid:capsule",
            "advisory_sha256": "d" * 64, "advisory_bytes": 400, "extra_provider_calls": 0,
            "source_freshness_verified": True, "candidate_only": True,
            "derived_runtime_admitted": False, "raw_advisory": "MUST NOT APPEAR"}}
    phases = {"prepare_seconds": 1, "initial_context_seconds": 12, "planning_seconds": 8,
        "context_seconds": 3, "doctor_seconds": 2, "post_publication_refresh_seconds": 4}
    receipt = _receipt(tmp_path, "full.json", SUPERVISOR, [{"trial": "full", "reward": {"reward": 0},
        "durations_seconds": {"agent_execution": 40}, "supervisor": {
            "seconds": 38, "phases": phases, "task_completed": True,
            "initial_context": {"seconds": 11, "nonoverlapping_seconds": {"vectors": 10},
                "indexed_symbols": 10, "full_capsules": 11, "world_task_count": 0,
                "provider_calls": 0, "canonical_tasks_created": False, "learned_embeddings": True},
            "planning": {"elapsed_seconds": 7, "model_request_sha256": "e" * 64,
                "initial_indexed_context": {"supplied_to_router": True, "summary_sha256": "f" * 64}},
            "context": {"learned_embeddings": True, "initial_indexes_reused": True,
                "new_embedding_calls": 0, "timings": {"nested_helper_seconds": {"cold": 9999}}},
            "implementation_route": "model_router", "doctor_dispatch": {"status": "residual",
                "task_cid": "cid:task", "provider_calls": 0, "residual_successors": 1,
                "residual_work_proposals": 0, "residual_context": {"artifact": "MUST NOT APPEAR"}},
            "provider_invocations": [invocation],
            "context_input_audit": {"status": "verified_all_observed_model_inputs",
                "any_native_input_verified": True, "any_model_input_verified": True,
                "all_observed_coding_inputs_verified": True, "coding_receipts_seen": 1,
                "usable_coding_receipts": 1, "raw_prompt": "MUST NOT APPEAR",
                "invocations": [{"invocation_id": "coding-call", "native_input_verified": True,
                    "model_input_verified": True, "status": "verified_exact_model_input"}]},
            "post_publication_context": {"status": "refreshed", "refresh_seconds": 4,
                "budget_seconds": 20, "embedding_calls": 2, "embedding_accounting": {
                    "all_refreshes_receipted": True,
                    "totals": {"local_embedding_calls": 2, "local_embedding_texts": 6,
                        "remote_embedding_calls": 0, "text_generation_calls": 0},
                    "known_subtotals": {"local_embedding_calls": 2, "local_embedding_texts": 6,
                        "remote_embedding_calls": 0, "text_generation_calls": 0}}},
        }}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    assert row["supervisor_phases_seconds"] == phases
    assert row["supervisor_seconds"] == 38 and row["durations_seconds"]["agent_execution"] == 40
    assert row["initial_indexed_context_observation"]["supplied_to_planning_router"] is True
    assert row["indexed_context_observation"]["initial_indexes_reused"] is True
    assert row["doctor_observation"]["status"] == "residual"
    assert row["doctor_observation"]["candidate_materializations_observed"] is None
    assert row["context_input_audit"]["all_observed_coding_inputs_verified"] is True
    wire = row["invocations"][0]["input_observation"]
    assert wire["native_prompt_bytes"] == 12000 and wire["router_prompt_bytes"] == 9400
    assert wire["semantic_translation"]["provider_prompt_bytes"] == 9000
    assert wire["doctor_residual_context"]["advisory_bytes"] == 400
    assert wire["measurements_are_token_savings"] is False
    refresh = row["post_publication_context"]
    assert refresh["embedding_accounting"]["totals"]["local_embedding_calls"] == 2
    assert refresh["embedding_accounting"]["totals"]["text_generation_calls"] == 0
    assert row["official_reward"] == 0 and row["native_task_completed"] is True
    assert row["tokens"]["total_tokens"] == 22  # neither byte counts nor embeddings are text tokens
    assert not result["benchmark_advantage_claimed"]
    serialized = json.dumps(result)
    assert "MUST NOT APPEAR" not in serialized and "9999" not in serialized
    rendered = markdown(result)
    assert "Cold initial context s" in rendered and "| full | 1 | 12 | 8 | 3 | 2 | 4 |" in rendered
    assert "not token savings" in rendered


def test_failed_cold_stage_is_retained_without_preparation_or_dispatch_claim(tmp_path):
    receipt = _receipt(tmp_path, "cold-failure.json", SUPERVISOR, [{"trial": "cold-failure",
        "supervisor": {"phases": {"prepare_seconds": 1, "initial_context_seconds": 55},
            "error": {"type": "TimeoutError"}}}], "full")
    comparison = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = comparison["rows"][1]
    assert row["supervisor_phases_seconds"]["initial_context_seconds"] == 55
    assert row["supervisor_phases_seconds"]["planning_seconds"] is None
    assert row["initial_indexed_context_observation"]["prepared"] is False
    assert row["initial_indexed_context_observation"]["supplied_to_planning_router"] is None
    assert row["context_input_audit"] is None and row["post_publication_context"] is None
    assert row["tokens"]["total_tokens"] is None
    assert "| cold-failure | 1 | 55 | unknown | unknown | unknown | unknown |" in markdown(comparison)


def test_declared_output_bytes_survive_comparison_without_correctness_or_token_claims(tmp_path):
    observation = {"schema": "terminal-declared-output-observation@1", "status": "observed",
        "task_cid_sha256": "a" * 64, "attempt_id_sha256": "b" * 64, "invocation_id_sha256": "c" * 64,
        "baseline_commit": "d" * 40, "private_prompt": "MUST NOT APPEAR", "task_correctness_established": True,
        "outputs": [{"path": "eigen.py", "effect": "modify", "baseline_sha256": "e" * 64,
            "candidate": {"status": "observed", "exists": True, "size_bytes": 100, "sha256": "f" * 64,
                "equals_admitted_baseline": False, "raw_source": "MUST NOT APPEAR"},
            "canonical": {"status": "observed", "exists": True, "size_bytes": 80, "sha256": "e" * 64,
                "equals_admitted_baseline": True}}]}
    receipt = _receipt(tmp_path, "output.json", SUPERVISOR, [{"trial": "timeout", "reward": {"reward": 0},
        "supervisor": {"task_completed": False, "declared_output_observation": observation}}], "no-index")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    retained = row["declared_output_observation"]
    assert retained["status"] == "observed" and retained["invocation_id_sha256"] == "c" * 64
    assert retained["outputs"][0]["candidate"]["equals_admitted_baseline"] is False
    assert retained["task_correctness_established"] is retained["patch_application_established"] is False
    assert row["official_reward"] == 0 and row["native_task_completed"] is False
    assert row["tokens"]["total_tokens"] is None and "MUST NOT APPEAR" not in json.dumps(result)


@pytest.mark.parametrize("malformed", ["missing", "extra_rows", "unsafe_path", "bad_digest", "status_list"])
def test_absent_or_malformed_byte_observation_remains_unknown(malformed):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "observed", "outputs": [
        {"path": "eigen.py", "effect": "modify", "candidate": {"status": "observed", "exists": True,
            "size_bytes": 9, "sha256": "bad"}, "canonical": {"status": "unavailable"}}]}
    if malformed == "extra_rows": raw["outputs"] *= 33
    elif malformed == "unsafe_path": raw["outputs"][0]["path"] = "../private"
    elif malformed == "status_list": raw["status"] = []
    result = _declared_output_observation({} if malformed == "missing" else {"declared_output_observation": raw})
    assert result["status"] in {"partial", "unavailable"}
    assert result["task_correctness_established"] is False
    assert "private" not in json.dumps(result) and "bad" not in json.dumps(result)


@pytest.mark.parametrize("stage", [
    "admitted_scope", "settlement_and_cleanup", "native_directory", "attempt_directory",
    "attempt_binding", "owner_event_log", "allocation_event", "allocation_root",
    "invocation_join", "declared_output_files",
])
def test_closed_output_failure_stage_survives_comparison(tmp_path, stage):
    observation = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": [], "evidence_stage": stage,
        "exception": "PRIVATE-EXCEPTION-CANARY", "completion_authority": True}
    receipt = _receipt(tmp_path, "output-stage.json", SUPERVISOR, [{"trial": "output-stage",
        "supervisor": {"task_completed": True, "declared_output_observation": observation}}], "full")
    result = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = result["rows"][1]
    retained = row["declared_output_observation"]
    assert retained["status"] == "unavailable"
    assert retained["evidence_stage"] == stage
    assert retained["outputs"] == [] and retained["completion_authority"] is False
    assert row["native_task_completed"] is True and row["official_reward"] is None
    assert row["tokens"]["total_tokens"] is None
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("value", [None, "", "PRIVATE-EXCEPTION-CANARY", "owner_event_log\nPRIVATE",
    " owner_event_log", "owner_event_log ", [], {}, True, 1])
def test_unknown_output_stage_is_omitted_without_losing_byte_observation(value):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": [], "evidence_stage": value}
    result = _declared_output_observation({"declared_output_observation": raw})
    assert result["status"] == "unavailable" and result["outputs"] == []
    assert "evidence_stage" not in result
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("change", [
    {"status": "observed"}, {"status": "partial"}, {"reason": "PRIVATE-EXCEPTION-CANARY"},
    {"schema": "unknown"}, {"outputs": None},
])
def test_stage_is_not_retained_from_inconsistent_or_malformed_observation(change):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": [], "evidence_stage": "owner_event_log", **change}
    result = _declared_output_observation({"declared_output_observation": raw})
    assert "evidence_stage" not in result
    assert "PRIVATE" not in json.dumps(result)


def test_historical_output_observation_does_not_gain_a_failure_stage():
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _declared_output_observation
    raw = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "binding_or_file_evidence_unavailable", "outputs": []}
    result = _declared_output_observation({"declared_output_observation": raw})
    assert result["status"] == "unavailable"
    assert result["reason"] == "reported_declared_byte_observation"
    assert "evidence_stage" not in result


def test_doctor_selection_is_distinct_from_matching_worker_materialization(tmp_path):
    receipt = _receipt(tmp_path, "doctor.json", SUPERVISOR, [{"trial": "doctor",
        "supervisor": {"implementation_route": "doctor_candidate",
            "doctor_dispatch": {"status": "candidate_ready", "task_cid": "cid:current", "provider_calls": 0},
            "doctor_invocations": [
                {"schema": "native-doctor-candidate-materialization@1", "task_cid": "cid:foreign",
                    "status": "candidate_materialized"},
                {"schema": "native-doctor-candidate-materialization@1", "task_cid": "cid:current",
                    "status": "candidate_materialized", "source": "MUST NOT APPEAR"}],
        }}], "full")
    comparison = collect(baseline=_baseline(tmp_path), supervisors=[receipt])
    row = comparison["rows"][1]
    assert row["doctor_observation"]["candidate_materializations_observed"] == 1
    assert row["doctor_observation"]["observed_worker_receipts"] == 1
    assert row["doctor_observation"]["selection_is_publication_or_completion_evidence"] is False
    assert row["native_task_completed"] is None and row["official_reward"] is None
    assert "MUST NOT APPEAR" not in json.dumps(comparison)


@pytest.mark.parametrize("refresh", [
    {"status": "deferred", "reason": "post_stop_refresh_budget_unavailable", "embedding_calls": None},
    {"status": "unavailable", "error_type": "_PostStopRefreshExpired", "embedding_calls": None},
    {"status": "incomplete", "embedding_calls": 0, "embedding_accounting": {
        "all_refreshes_receipted": False, "totals": {"local_embedding_calls": 0},
        "known_subtotals": {"local_embedding_calls": 1, "local_embedding_texts": 3}}},
])
def test_refresh_unknown_totals_preserve_observed_failed_costs(tmp_path, refresh):
    receipt = _receipt(tmp_path, "refresh.json", SUPERVISOR, [{"trial": "refresh",
        "supervisor": {"task_completed": True, "post_publication_context": refresh}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    observed = row["post_publication_context"]
    assert observed["embedding_calls"] is None
    assert all(value is None for value in observed["embedding_accounting"]["totals"].values())
    assert observed["embedding_accounting"]["known_subtotals"]["local_embedding_calls"] == (
        1 if refresh["status"] == "incomplete" else None)
    assert row["native_task_completed"] is True
    assert observed["task_completion_is_context_refresh_evidence"] is False


def test_reported_wire_measurements_do_not_imply_successful_input_audit(tmp_path):
    invocation = {**_invocation(), "native_prompt_bytes": 12000, "router_prompt_bytes": 9000,
        "semantic_translation": {"identifier_mappings": 10, "freshness_checked": True}}
    receipt = _receipt(tmp_path, "unaudited.json", SUPERVISOR, [{"trial": "unaudited",
        "supervisor": {"provider_invocations": [invocation], "context_input_audit": {
            "status": "verified_native_input_only", "any_native_input_verified": True,
            "any_model_input_verified": False, "all_observed_coding_inputs_verified": False}}}], "full")
    row = collect(baseline=_baseline(tmp_path), supervisors=[receipt])["rows"][1]
    assert row["invocations"][0]["input_observation"]["semantic_translation"]["identifier_mappings"] == 10
    assert row["context_input_audit"]["any_model_input_verified"] is False
    assert row["context_input_audit"]["all_observed_coding_inputs_verified"] is False


def _metadata_selection(schema="supervisor-semantic-metadata-selection@2", qualification="provider_tokenizer"):
    value = {"schema": schema, "view_receipt_sha256": "a" * 64,
        "selected_mode": "common-bindings@1", "fallback_reason": None,
        "token_proxy": "utf8-bytes-ceil-div4@1", "candidate_only": True,
        "native_complete_sha256": "b" * 64, "native_complete_bytes": 1000,
        "native_complete_proxy_tokens": 250, "candidate_complete_sha256": "c" * 64,
        "candidate_complete_bytes": 800, "candidate_complete_proxy_tokens": 200,
        "selected_complete_sha256": "c" * 64, "selected_complete_bytes": 800,
        "selected_complete_proxy_tokens": 200}
    if schema.endswith("@2"):
        identity = {"schema": "supervisor-semantic-metadata-token-counter@1",
            "provider": "codex_cli", "model": "gpt-6.1-sol", "tokenizer": "declared-tokenizer",
            "tokenizer_revision": "revision-1", "qualification": qualification,
            "measurement_scope": "complete-rendered-prompt-text@1",
            "identity_caller_supplied": True, "provider_usage_verified": False}
        value.update(selection_basis="token_counter", token_counter_status="counted",
            token_counter_identity=identity,
            token_counter_identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            native_complete_tokens=120, candidate_complete_tokens=90, selected_complete_tokens=90)
    return value


def test_absent_metadata_preserves_existing_input_observation_shape():
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    observed = _input_observation({})
    assert set(observed) == {"native_prompt_bytes", "router_prompt_bytes", "model_prompt_bytes",
        "workspace_advisory_bytes", "native_prompt_sha256", "router_prompt_sha256", "model_prompt_sha256",
        "workspace_advisory_sha256", "semantic_translation", "doctor_residual_context",
        "measurements_are_token_savings", "input_reconstruction_verified_by_receipt_presence"}
    assert "semantic_metadata_view" not in observed


@pytest.mark.parametrize("schema, qualification", [
    ("supervisor-semantic-metadata-selection@1", "provider_tokenizer"),
    ("supervisor-semantic-metadata-selection@2", "provider_tokenizer"),
    ("supervisor-semantic-metadata-selection@2", "estimate"),
])
def test_metadata_measurements_survive_comparison_without_charging_native_usage(tmp_path, schema, qualification):
    before = _invocation()
    metadata = _metadata_selection(schema, qualification)
    metadata.update(prompt="MUST NOT APPEAR", raw_source="MUST NOT APPEAR",
        counter_error="MUST NOT APPEAR", provider_usage_reported=True,
        complete_run_token_savings_claimed=True)
    if schema.endswith("@2"):
        metadata["token_counter_identity"]["source_body"] = "MUST NOT APPEAR"
    after = {**before, "semantic_metadata_view": metadata}
    reports = []
    for label, invocation in (("without", before), ("with", after)):
        path = _receipt(tmp_path, label + ".json", SUPERVISOR, [{"trial": label,
            "supervisor": {"provider_invocations": [invocation, copy.deepcopy(invocation)]}}], "full")
        reports.append(collect(baseline=_baseline(tmp_path), supervisors=[path]))
    old, new = (report["rows"][1] for report in reports)
    observed = new["invocations"][0]["input_observation"]["semantic_metadata_view"]
    assert observed["status"] == "reported"
    assert observed["selected_complete_bytes"] == 800
    assert observed["selected_complete_proxy_tokens"] == 200
    assert observed["provider_usage_reported"] is False
    assert observed["tokenizer_compatibility_verified"] is False
    assert observed["complete_run_token_savings_claimed"] is False
    if schema.endswith("@1"):
        assert observed["selection_basis"] == "byte_proxy"
        assert "selected_complete_tokens" not in observed
    else:
        assert observed["selection_basis"] == "token_counter"
        assert observed["selected_complete_tokens"] == 90
        assert observed["token_counter_identity"]["qualification"] == qualification
    for field in ("tokens", "known_token_subtotals", "provider_calls", "reported_cost_usd",
                  "native_usage_complete_observed", "cache_included_in_input"):
        assert new[field] == old[field]
    assert new["tokens"]["input_tokens"] == 20  # Includes its 15 cached tokens already.
    assert new["tokens"]["total_tokens"] == 22 and new["provider_calls"] == 1
    assert "MUST NOT APPEAR" not in json.dumps(reports[1])


@pytest.mark.parametrize("mutation", ["unknown_schema", "list_schema", "list_status", "unknown_qualification",
    "list_qualification", "unverified_scope", "identity_digest", "boolean_count", "negative_count",
    "overflow_count", "selected_count", "proxy_count", "caller_flag", "unsafe_identity"])
def test_malformed_metadata_measurements_remain_unavailable(mutation):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection()
    identity = metadata["token_counter_identity"]
    if mutation == "unknown_schema": metadata["schema"] = "unknown-MUST NOT APPEAR"
    elif mutation == "list_schema": metadata["schema"] = ["MUST NOT APPEAR"]
    elif mutation == "list_status": metadata["token_counter_status"] = ["MUST NOT APPEAR"]
    elif mutation == "unknown_qualification": identity["qualification"] = "MUST NOT APPEAR"
    elif mutation == "list_qualification": identity["qualification"] = ["MUST NOT APPEAR"]
    elif mutation == "unverified_scope": identity["measurement_scope"] = "MUST NOT APPEAR"
    elif mutation == "identity_digest": metadata["token_counter_identity_sha256"] = "MUST NOT APPEAR"
    elif mutation == "boolean_count": metadata["native_complete_tokens"] = True
    elif mutation == "negative_count": metadata["native_complete_tokens"] = -1
    elif mutation == "overflow_count": metadata["native_complete_tokens"] = 2**63
    elif mutation == "selected_count": metadata["selected_complete_tokens"] = 91
    elif mutation == "proxy_count": metadata["candidate_complete_proxy_tokens"] += 1
    elif mutation == "caller_flag": identity["identity_caller_supplied"] = 1
    elif mutation == "unsafe_identity": identity["tokenizer"] = "MUST NOT APPEAR\nsource()"
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed == {"status": "unavailable", "provider_usage_reported": False,
        "tokenizer_compatibility_verified": False, "complete_run_token_savings_claimed": False}
    assert "MUST NOT APPEAR" not in json.dumps(observed)


def test_counter_failure_retains_fallback_without_exception_text_or_invented_counts():
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection()
    metadata.update(token_counter_status="failed", selected_mode="legacy",
        fallback_reason="complete_input_token_counter_failed", native_complete_tokens=None,
        candidate_complete_tokens=None, selected_complete_tokens=None, counter_error="MUST NOT APPEAR")
    for field in ("sha256", "bytes", "proxy_tokens"):
        metadata["selected_complete_" + field] = metadata["native_complete_" + field]
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed["status"] == "reported" and observed["token_counter_status"] == "failed"
    assert observed["selected_mode"] == "legacy"
    assert observed["selected_complete_tokens"] is None
    assert "MUST NOT APPEAR" not in json.dumps(observed)


@pytest.mark.parametrize("schema", ["supervisor-semantic-metadata-selection@1", "supervisor-semantic-metadata-selection@2"])
@pytest.mark.parametrize("fallback", [False, True])
def test_instruction_metadata_mode_retains_requested_treatment_including_fallback(schema, fallback):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection(schema)
    metadata.update(requested_mode="common-bindings-instruction@1",
                    selected_mode="legacy" if fallback else "common-bindings-instruction@1")
    if fallback:
        metadata["fallback_reason"] = ("complete_input_not_smaller_under_bytes_and_proxy"
            if schema.endswith("@1") else "complete_input_not_smaller_under_bytes_and_token_counter")
        for field in ("sha256", "bytes", "proxy_tokens", "tokens"):
            key = "native_complete_" + field
            if key in metadata:
                metadata["selected_complete_" + field] = metadata[key]
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed["status"] == "reported"
    assert observed["requested_mode"] == "common-bindings-instruction@1"
    assert observed["selected_mode"] == metadata["selected_mode"]
    assert observed["selected_complete_sha256"] == metadata["selected_complete_sha256"]
    assert observed["complete_run_token_savings_claimed"] is False


@pytest.mark.parametrize("requested", [None, [], "foreign-MUST NOT APPEAR", "common-bindings@1", "legacy"])
def test_instruction_metadata_mode_refuses_missing_foreign_or_conflicting_requested_mode(requested):
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import _input_observation
    metadata = _metadata_selection()
    metadata["selected_mode"] = "common-bindings-instruction@1"
    if requested is not None:
        metadata["requested_mode"] = requested
    observed = _input_observation({"semantic_metadata_view": metadata})["semantic_metadata_view"]
    assert observed["status"] == "unavailable"
    assert "requested_mode" not in observed and "MUST NOT APPEAR" not in json.dumps(observed)
