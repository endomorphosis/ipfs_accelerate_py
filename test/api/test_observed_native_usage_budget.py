"""Between-callback native observations, with no model or CLI execution.

Authored native receipts exercise accounting/admission boundaries. The real
shared-helper test additionally runs genuine capsule production, worktree
checks and expansion, replacing only the existing provider boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.cumulative_usage_reconciliation import (
    COMPLETION_BASIS, MAX_COUNTER,
)
from ipfs_accelerate_py.agent_supervisor.runtime.observed_native_usage_budget import (
    ObservedNativeUsageBudget,
)
from ipfs_accelerate_py.agent_supervisor.runtime import observed_native_usage_budget as budget_module
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter, render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import (
    FINAL_SCHEMA, REQUEST_SCHEMA, ROUND_SCHEMA, SemanticDecisionCall,
    SemanticDecisionDialogueError, SemanticDecisionProviderFailure, SemanticDecisionReply,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_provider as shared
from ipfs_accelerate_py.cli_runtime.grok_native_usage import (
    grok_outcome_receipt, grok_usage_receipt, native_grok_observation,
)

pytest_plugins = ("test.api.test_semantic_router_integration",)

PRIVATE = "private-observed-native-budget-marker"
OPTION = "max_observed_native_tokens_before_next_dispatch"


@pytest.fixture(autouse=True)
def callback_working_directory(tmp_path, monkeypatch):
    # The actual native fixture moves into its genuine allocated worktree later.
    monkeypatch.chdir(tmp_path)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_for(workspace, *, provider="grok_cli", generation=0, session="b" * 32):
    prompt = "Authored checked decision prompt."
    rendered, _ = render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
    return SemanticDecisionCall(call_id=digest(f"call-{generation}")[:32],
        session_id=session, generation=generation, binding_sha256="c" * 64,
        provider=provider, model="pinned", prompt=prompt, timeout=30,
        max_output_tokens=512, model_prompt_sha256=digest(rendered))


def grok_observation(*, inputs=7, cached=2, cache_write=1, outputs=3,
                     complete=True, failed=False):
    payload = {"stopReason": "end_turn", "usage": {
        "input_tokens": inputs, "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": cache_write, "output_tokens": outputs,
        "total_tokens": inputs + cached + cache_write + outputs}}
    if complete is not None:
        payload["usage_is_incomplete"] = not complete
    if failed:
        payload["type"] = "error"
    return native_grok_observation(payload, exit_code=1 if failed else 0)


def provider_receipt(call, workspace, *, complete=True, failed=False,
                     inputs=100, outputs=20, native_session=None):
    rendered, _ = render_model_prompt(prompt=call.prompt, purpose="coding", workspace=workspace)
    receipt = {"schema": "router-semantic-decision-provider@1",
        "call_id": call.call_id, "session_id": call.session_id,
        "generation": call.generation, "binding_sha256": call.binding_sha256,
        "provider": call.provider, "model": call.model, "workspace": str(workspace),
        "source_repository": str(workspace), "prompt_sha256": digest(call.prompt),
        "prompt_bytes": len(call.prompt.encode("utf-8")),
        "model_prompt_sha256": call.model_prompt_sha256,
        "model_prompt_bytes": len(rendered.encode("utf-8")),
        "provider_called": True, "status": "failed" if failed else "provider_returned",
        "usage_status": "reported", "completion_authority": False}
    if call.provider == "grok_cli":
        observed = grok_observation(complete=complete, failed=failed)
        receipt.update(usage={"model_id": call.model,
            "session_id": native_session or f"native-grok-session-{call.generation}"},
            native_rollout_usage=grok_usage_receipt(observed),
            native_provider_outcome=grok_outcome_receipt(observed))
    else:
        thread = native_session or "native-thread-aaaaaaaa"
        native = {"schema": "router-implementation-invocation@1",
            "invocation_id": call.call_id, "provider": call.provider,
            "model": call.model, "workspace": str(workspace),
            "model_prompt_sha256": call.model_prompt_sha256, "purpose": "coding",
            "status": receipt["status"],
            "usage": {"thread_id": thread, "model_id": call.model},
            "native_rollout_usage": {"schema": "native-codex-cumulative-usage@1",
                "thread_id": thread, "usage_available": True,
                "usage": {"input_tokens": inputs, "output_tokens": outputs,
                    "cached_input_tokens": inputs // 2,
                    "reasoning_output_tokens": outputs // 2, "total_tokens": inputs + outputs},
                "rollout_sha256": digest(f"{thread}:{call.generation}:{inputs}:{outputs}"),
                "observed_token_count_records": call.generation + 1,
                "task_complete_observed": complete is True,
                "cache_included_in_input": True, "completion_basis": COMPLETION_BASIS,
                "usage_lifecycle_complete": complete,
                "lifecycle_malformed_records": 0, "billing_total_verified": False}}
        receipt.update(usage=deepcopy(native["usage"]), native_invocation=native,
                       native_rollout_usage=deepcopy(native["native_rollout_usage"]))
    return receipt


def wrap(target, workspace, *, provider="grok_cli", threshold=100):
    return ObservedNativeUsageBudget(target, provider=provider, model="pinned",
        workspace=workspace, max_observed_native_tokens_before_next_dispatch=threshold)


def no_authority(observation):
    assert observation["schema"] == "observed-native-usage-stop@1"
    assert observation["first_dispatch_may_overshoot"] is True
    assert observation["task_attempt_total"] is None
    assert observation["campaign_usage_complete"] is None
    for key in ("hidden_native_requests_capped", "native_cumulative_cap_enforced",
                "billing_total_verified", "completion_authority", "execution_authority"):
        assert observation[key] is False


def refused(guard, call, reason):
    with pytest.raises(SemanticDecisionProviderFailure) as captured:
        guard(call)
    receipt = captured.value.receipt
    assert receipt["schema"] == "router-semantic-decision-provider@1"
    assert receipt["provider_called"] is False
    for key in ("call_id", "session_id", "generation", "binding_sha256",
                "provider", "model", "model_prompt_sha256"):
        assert receipt[key] == getattr(call, key)
    observation = receipt["observed_native_usage_stop"]
    no_authority(observation)
    assert observation["decision"] == "deny" and observation["reason"] == reason
    assert receipt.get("native_invocation") is None
    assert receipt.get("native_rollout_usage") is None
    return receipt


@pytest.mark.parametrize("threshold", [1, 13])
def test_first_charge_can_overshoot_or_reach_threshold_but_next_callback_is_not_dispatched(tmp_path, threshold):
    calls = []
    def target(call):
        calls.append(call)
        return SemanticDecisionReply("authored response", provider_receipt(call, tmp_path))
    guard = wrap(target, tmp_path, threshold=threshold)
    first = guard(call_for(tmp_path))
    assert first.text == "authored response" and len(calls) == 1
    observation = first.receipt["observed_native_usage_stop"]
    no_authority(observation)
    assert observation["scoped_observed_total"] == 13
    assert observation["decision_basis_total"] is None
    denial = refused(guard, call_for(tmp_path, generation=1), "threshold_reached")
    assert denial["observed_native_usage_stop"]["decision_basis_total"] == 13
    assert len(calls) == 1
    snapshot = guard.observation()
    assert snapshot["provider_callback_dispatches"] == 1
    assert snapshot["denied_callbacks"] == 1


def test_known_below_threshold_allows_another_call_and_counts_distinct_equal_grok_envelopes(tmp_path):
    calls = []
    def target(call):
        calls.append(call)
        return SemanticDecisionReply("response", provider_receipt(call, tmp_path))
    guard = wrap(target, tmp_path, threshold=26)
    guard(call_for(tmp_path))
    guard(call_for(tmp_path, generation=1))
    observation = guard.observation()
    assert len(calls) == 2 and observation["scoped_observed_total"] == 26
    assert observation["usage_scope"] == "grok_native_final_envelope"
    assert observation["known_for_continuing"] is True
    refused(guard, call_for(tmp_path, generation=2), "threshold_reached")
    assert len(calls) == 2


def test_codex_latest_cumulative_session_snapshot_is_used_without_cache_or_reasoning_double_add(tmp_path):
    calls = []
    def target(call):
        calls.append(call)
        return SemanticDecisionReply("response", provider_receipt(call, tmp_path,
            inputs=100 if call.generation == 0 else 180, outputs=20))
    guard = wrap(target, tmp_path, provider="codex_cli", threshold=200)
    guard(call_for(tmp_path, provider="codex_cli"))
    guard(call_for(tmp_path, provider="codex_cli", generation=1))
    observation = guard.observation()
    assert observation["scoped_observed_total"] == 200
    assert observation["usage_scope"] == "codex_native_session_cumulative"
    assert observation["known_for_continuing"] is True
    refused(guard, call_for(tmp_path, provider="codex_cli", generation=2), "threshold_reached")
    assert len(calls) == 2


@pytest.mark.parametrize("damage", ["no_native", "incomplete", "unknown_completeness",
    "missing_counter", "counter_bool", "counter_overflow", "total_conflict", "missing_reported_session"])
def test_sparse_or_conflicting_native_evidence_denies_next_call_despite_small_generic_usage(tmp_path, damage):
    calls = []
    def target(call):
        calls.append(call)
        receipt = provider_receipt(call, tmp_path,
            complete=None if damage == "unknown_completeness" else damage != "incomplete")
        receipt["usage"].update(prompt_tokens=1, completion_tokens=1)
        if damage == "no_native":
            receipt.pop("native_rollout_usage")
        elif damage == "missing_counter":
            del receipt["native_rollout_usage"]["usage"]["cache_write_input_tokens"]
        elif damage == "counter_bool":
            receipt["native_rollout_usage"]["usage"]["input_tokens"] = True
        elif damage == "counter_overflow":
            receipt["native_rollout_usage"]["usage"]["input_tokens"] = MAX_COUNTER + 1
        elif damage == "total_conflict":
            receipt["native_rollout_usage"]["usage"]["total_tokens"] = 99
        elif damage == "missing_reported_session":
            receipt["usage"].pop("session_id")
        return SemanticDecisionReply("response", receipt)
    guard = wrap(target, tmp_path, threshold=1000)
    guard(call_for(tmp_path))
    refused(guard, call_for(tmp_path, generation=1), "prior_usage_unknown_or_conflicting")
    assert len(calls) == 1 and guard.observation()["known_for_continuing"] is False
    assert PRIVATE not in json.dumps(guard.observation(), allow_nan=False)


def test_reused_reported_grok_session_is_not_a_second_independent_charge_identity(tmp_path):
    calls = []
    def target(call):
        calls.append(call)
        return SemanticDecisionReply("response", provider_receipt(call, tmp_path,
            native_session="same-reported-session"))
    guard = wrap(target, tmp_path, threshold=1000)
    guard(call_for(tmp_path))
    guard(call_for(tmp_path, generation=1))
    assert guard.observation()["known_for_continuing"] is False
    refused(guard, call_for(tmp_path, generation=2), "prior_usage_unknown_or_conflicting")
    assert len(calls) == 2


@pytest.mark.parametrize("field", ["call_id", "session_id", "generation", "binding_sha256",
    "provider", "model", "workspace", "prompt_sha256", "prompt_bytes",
    "model_prompt_sha256", "model_prompt_bytes", "provider_called"])
def test_changed_outer_receipt_binding_cannot_authorize_a_later_callback(tmp_path, field):
    calls = []
    def target(call):
        calls.append(call)
        receipt = provider_receipt(call, tmp_path)
        receipt[field] = 99 if field in ("generation", "prompt_bytes", "model_prompt_bytes") else (
            None if field == "provider_called" else "foreign-binding")
        return SemanticDecisionReply("response", receipt)
    guard = wrap(target, tmp_path)
    guard(call_for(tmp_path))
    refused(guard, call_for(tmp_path, generation=1), "prior_usage_unknown_or_conflicting")
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["provider", "model", "workspace", "model_prompt_sha256"])
def test_codex_nested_invocation_must_join_the_exact_outer_callback(tmp_path, field):
    calls = []
    def target(call):
        calls.append(call)
        receipt = provider_receipt(call, tmp_path)
        receipt["native_invocation"][field] = "foreign-binding"
        return SemanticDecisionReply("response", receipt)
    guard = wrap(target, tmp_path, provider="codex_cli", threshold=1000)
    guard(call_for(tmp_path, provider="codex_cli"))
    refused(guard, call_for(tmp_path, provider="codex_cli", generation=1), "prior_usage_unknown_or_conflicting")
    assert len(calls) == 1


def test_charged_provider_failure_is_rethrown_by_identity_without_private_error_echo(tmp_path):
    call = call_for(tmp_path)
    original = SemanticDecisionProviderFailure(PRIVATE,
        receipt=provider_receipt(call, tmp_path, complete=False, failed=True))
    calls = []
    def target(value):
        calls.append(value)
        raise original
    guard = wrap(target, tmp_path)
    with pytest.raises(SemanticDecisionProviderFailure) as captured:
        guard(call)
    assert captured.value is original
    assert original.receipt["native_rollout_usage"]["usage"]["total_tokens"] == 13
    observation = guard.observation()
    assert observation["prior_receipt_count"] == 1
    assert observation["scoped_observed_total"] == 13
    assert observation["known_for_continuing"] is False
    refused(guard, call_for(tmp_path, generation=1), "prior_usage_unknown_or_conflicting")
    assert len(calls) == 1 and PRIVATE not in json.dumps(guard.observation())


def test_raw_failure_without_a_receipt_remains_the_original_unknown_failure(tmp_path):
    original = TimeoutError(PRIVATE)
    calls = []
    def target(call):
        calls.append(call)
        raise original
    guard = wrap(target, tmp_path)
    with pytest.raises(TimeoutError) as captured:
        guard(call_for(tmp_path))
    assert captured.value is original
    assert guard.observation()["prior_unknown_receipt_count"] >= 1
    refused(guard, call_for(tmp_path, generation=1), "prior_usage_unknown_or_conflicting")
    assert len(calls) == 1 and PRIVATE not in json.dumps(guard.observation())


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("seam", ["projection", "annotation"])
def test_fresh_accounting_or_annotation_interruption_after_success_is_not_swallowed(
        tmp_path, monkeypatch, interruption, seam):
    original = interruption(PRIVATE)
    calls = []
    def target(call):
        calls.append(call)
        return SemanticDecisionReply("response", provider_receipt(call, tmp_path))
    guard = wrap(target, tmp_path)
    if seam == "projection":
        def interrupted(*args, **kwargs):
            raise original
        monkeypatch.setattr(budget_module, "reconcile_native_usage_observations", interrupted)
    else:
        genuine_encoded = budget_module._encoded
        def interrupted(value):
            if type(value) is dict and "observed_native_usage_stop" in value:
                raise original
            return genuine_encoded(value)
        monkeypatch.setattr(budget_module, "_encoded", interrupted)
    with pytest.raises(interruption) as captured:
        guard(call_for(tmp_path))
    assert captured.value is original and len(calls) == 1


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("underlying_kind", ["provider_failure", "provider_interruption"])
def test_secondary_accounting_interruption_cannot_replace_original_provider_failure(
        tmp_path, monkeypatch, interruption, underlying_kind):
    call = call_for(tmp_path)
    if underlying_kind == "provider_failure":
        original = SemanticDecisionProviderFailure(PRIVATE,
            receipt=provider_receipt(call, tmp_path, complete=False, failed=True))
    else:
        original = KeyboardInterrupt(PRIVATE)
    secondary = interruption("private-secondary-accounting-marker")
    attempted = []
    def project(*args, **kwargs):
        attempted.append(True)
        raise secondary
    monkeypatch.setattr(budget_module, "reconcile_native_usage_observations", project)
    def target(value):
        raise original
    guard = wrap(target, tmp_path)
    with pytest.raises(type(original)) as captured:
        guard(call)
    assert captured.value is original and attempted == [True]
    assert PRIVATE not in json.dumps(guard.observation())


@pytest.mark.parametrize("change", ["same_call", "cross_session", "generation_gap", "provider", "model"])
def test_replay_or_rebinding_is_refused_before_the_target(tmp_path, change):
    calls = []
    def target(call):
        calls.append(call)
        return SemanticDecisionReply("response", provider_receipt(call, tmp_path))
    guard = wrap(target, tmp_path)
    original = call_for(tmp_path)
    guard(original)
    next_call = call_for(tmp_path, generation=1)
    if change == "same_call":
        next_call = original
    elif change == "cross_session":
        next_call = replace(next_call, session_id="d" * 32)
    elif change == "generation_gap":
        next_call = replace(next_call, generation=2)
    else:
        next_call = replace(next_call, **{change: "codex_cli" if change == "provider" else "foreign-model"})
    refused(guard, next_call, "binding_or_replay_refused")
    assert len(calls) == 1


def test_concurrent_callback_refuses_without_waiting_for_or_dispatching_second_target(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls, outcomes = [], []
    def target(call):
        calls.append(call)
        entered.set()
        assert release.wait(10), "authored callback was not released"
        return SemanticDecisionReply("response", provider_receipt(call, tmp_path))
    guard = wrap(target, tmp_path)
    def first():
        try:
            outcomes.append(guard(call_for(tmp_path)))
        except BaseException as error:
            outcomes.append(error)
    thread = threading.Thread(target=first, daemon=True)
    thread.start()
    try:
        assert entered.wait(5)
        refused(guard, call_for(tmp_path, generation=1), "concurrent_dispatch_refused")
        assert len(calls) == 1 and thread.is_alive()
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and len(outcomes) == 1
    assert isinstance(outcomes[0], SemanticDecisionReply)


def test_observation_is_a_bounded_fresh_public_copy_without_history_or_authority(tmp_path):
    receipts = []
    def target(call):
        receipt = provider_receipt(call, tmp_path)
        receipt.update(prompt=PRIVATE, response=PRIVATE, auth={"key": PRIVATE})
        receipts.append(receipt)
        return SemanticDecisionReply("response", receipt)
    guard = wrap(target, tmp_path)
    result = guard(call_for(tmp_path))
    assert "observed_native_usage_stop" not in receipts[0]
    assert result.receipt is not receipts[0]
    assert result.receipt["native_rollout_usage"] == receipts[0]["native_rollout_usage"]
    observation = guard.observation()
    no_authority(observation)
    encoded = json.dumps(observation, allow_nan=False)
    assert len(encoded.encode()) <= 4096
    for forbidden in (PRIVATE, str(tmp_path), "native-grok-session-0", "b" * 32):
        assert forbidden not in encoded
    assert "projection" not in observation and "receipts" not in observation
    observation["reason"] = PRIVATE
    assert guard.observation()["reason"] != PRIVATE
    assert guard.observation()["workspace_binding_sha256"] == digest(str(tmp_path))


def test_optional_annotation_cannot_make_an_otherwise_admissible_native_receipt_oversized(tmp_path):
    call = call_for(tmp_path)
    original = provider_receipt(call, tmp_path)
    original["padding"] = ""
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    original["padding"] = "x" * (65_408 - len(encode(original)))
    assert len(encode(original)) == 65_408
    held = deepcopy(original)
    guard = wrap(lambda value: SemanticDecisionReply("response", original), tmp_path)
    reply = guard(call)
    assert len(encode(reply.receipt)) <= 65_536
    assert "observed_native_usage_stop" not in reply.receipt
    assert reply.receipt == held and original == held
    assert reply.receipt["native_rollout_usage"] == held["native_rollout_usage"]
    observation = guard.observation()
    assert observation["scoped_observed_total"] == 13 and observation["known_for_continuing"] is True
    no_authority(observation)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, "100", MAX_COUNTER + 1])
def test_invalid_limit_refuses_before_underlying_callback_or_automatic_tokenizer(tmp_path, monkeypatch, invalid):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    effects = []
    def forbidden(*args, **kwargs):
        effects.append((args, kwargs))
        raise AssertionError("invalid threshold reached tokenizer or router setup")
    with pytest.raises(ValueError):
        wrap(forbidden, tmp_path, threshold=invalid)
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", forbidden)
    monkeypatch.setattr(shared, "SharedRouterDecisionProvider", forbidden)
    with pytest.raises(ValueError):
        shared.run_shared_router_decision_dialogue(native_prompt="plain authored request",
            repository=tmp_path, workspace=tmp_path,
            policy=SemanticDecisionPolicy(required_capsule_ids=(), optional_capsule_ids=()),
            provider="grok_cli", model="pinned", **{OPTION: invalid})
    assert effects == []


def native_route(allocated, monkeypatch, *, complete=True):
    """Keep the genuine native context/adapter/dialogue; author only provider IO."""
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    repository, workspace, prompt, source = allocated
    parent = json.loads((repository / ".runtime/semantic/worker-context.json").read_text())
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids[1:], optional_capsule_ids=ids[:1])
    calls = []
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="grok_cli", model_name="pinned", catalog_revision="authored"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    monkeypatch.setattr(llm_router, "_grok_cli_command", lambda: "/fixture/grok")
    monkeypatch.setattr(grok_cli_runner, "_resolve_trusted_grok_bin", lambda **kwargs: "/fixture/grok")
    def generate(model_prompt, **kwargs):
        calls.append((model_prompt, kwargs))
        protocol = next(json.loads(line) for line in model_prompt.splitlines()
            if line.startswith('{"binding_sha256":') and ROUND_SCHEMA in line)
        assert kwargs["provider"] == "grok_cli" and kwargs["model_name"] == "pinned"
        assert kwargs["allow_local_fallback"] is False and kwargs["allow_cross_provider_fallback"] is False
        observed = grok_observation(complete=complete)
        observed.update(exit_code=0, model_id="pinned", prompt_tokens=7, completion_tokens=3,
            cached_tokens=2, session_id=f"authored-native-session-{len(calls)}")
        set_last_cli_observation("grok_cli", observed)
        response = {key: protocol[key] for key in ("session_id", "generation", "binding_sha256")}
        if len(calls) == 1:
            response.update(schema=REQUEST_SCHEMA, capsule_ids=[ids[0]])
        else:
            assert "Supervisor verified capsule deliveries:" in model_prompt
            response.update(schema=FINAL_SCHEMA, text="Authored candidate; task acceptance is unproved.")
        return json.dumps(response)
    monkeypatch.setattr(llm_router, "generate_text", generate)
    options = {"native_prompt": prompt, "repository": repository, "workspace": workspace,
        "policy": policy, "provider": "grok_cli", "model": "pinned", "timeout": 90,
        "prompt_token_counter": PromptTokenCounter(provider="grok_cli", model="pinned",
            tokenizer_id="authored", tokenizer_revision="1", special_token_policy="authored fixture",
            count_kind="estimate", compatibility_basis="authored fixture only",
            count_tokens=lambda text: 100 if ROUND_SCHEMA in text else 1000)}
    return options, calls, ids, source


@pytest.mark.parametrize("complete,threshold,reason", [
    (True, 13, "threshold_reached"), (None, 1000, "prior_usage_unknown_or_conflicting")])
def test_actual_shared_helper_retains_native_charge_and_refuses_second_expansion_dispatch(
        allocated, monkeypatch, complete, threshold, reason):
    options, calls, ids, source = native_route(allocated, monkeypatch, complete=complete)
    with pytest.raises(SemanticDecisionDialogueError) as captured:
        shared.run_shared_router_decision_dialogue(**options, **{OPTION: threshold})
    ledger = captured.value.receipt
    assert len(calls) == 1 and len(ledger["attempts"]) == 2
    first, denied = ledger["attempts"]
    assert first["provider_receipt_bound"] is True
    assert first["provider_receipt"]["native_rollout_usage"]["usage"]["total_tokens"] == 13
    assert first["provider_receipt"]["provider_called"] is True
    assert len(ledger["expansions"]) == 1 and ledger["expansions"][0]["capsule_ids"] == [ids[0]]
    assert denied["provider_receipt_bound"] is True
    assert denied["provider_receipt"]["provider_called"] is False
    observation = denied["provider_receipt"]["observed_native_usage_stop"]
    assert observation["reason"] == reason and observation["decision"] == "deny"
    assert observation["provider_callback_dispatches"] == 1
    assert ledger["provider_dispatches_reported"] == 1
    assert ledger["provider_usage_complete"] is False
    assert ledger["whole_session_token_savings"] is None
    assert ledger["observed_native_usage_stop"] == observation
    no_authority(observation)
    assert (options["repository"] / "mod.py").read_text() == source


@pytest.mark.parametrize("explicit_none", [False, True])
def test_disabled_shared_helper_keeps_real_two_call_route_without_additive_receipt_fields(
        allocated, monkeypatch, explicit_none):
    options, calls, ids, source = native_route(allocated, monkeypatch)
    if explicit_none:
        options[OPTION] = None
    result = shared.run_shared_router_decision_dialogue(**options)
    assert len(calls) == len(result.receipt["attempts"]) == 2
    assert len(result.receipt["expansions"]) == 1
    assert "observed_native_usage_stop" not in result.receipt
    assert all("observed_native_usage_stop" not in row["provider_receipt"]
               for row in result.receipt["attempts"])
    assert result.receipt["provider_usage_complete"] is False
    assert (options["repository"] / "mod.py").read_text() == source


def test_actual_shared_helper_below_threshold_finishes_with_scoped_summary_only(allocated, monkeypatch):
    options, calls, ids, source = native_route(allocated, monkeypatch)
    result = shared.run_shared_router_decision_dialogue(**options, **{OPTION: 1000})
    assert len(calls) == len(result.receipt["attempts"]) == 2
    observation = result.receipt["observed_native_usage_stop"]
    assert observation["provider_callback_dispatches"] == 2
    assert observation["scoped_observed_total"] == 26
    assert observation["known_for_continuing"] is True
    assert observation["usage_scope"] == "grok_native_final_envelope"
    assert result.receipt["provider_usage_complete"] is False
    no_authority(observation)


def test_actual_shared_helper_failure_keeps_charged_native_receipt_and_original_provider_cause(
        allocated, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    options, calls, ids, source = native_route(allocated, monkeypatch)
    original = TimeoutError(PRIVATE)
    def fail(model_prompt, **kwargs):
        calls.append((model_prompt, kwargs))
        observed = grok_observation(complete=False, failed=True)
        observed.update(exit_code=1, model_id="pinned", session_id="failed-native-session",
            prompt_tokens=7, completion_tokens=3, cached_tokens=2)
        set_last_cli_observation("grok_cli", observed)
        raise original
    monkeypatch.setattr(llm_router, "generate_text", fail)
    with pytest.raises(SemanticDecisionDialogueError) as captured:
        shared.run_shared_router_decision_dialogue(**options, **{OPTION: 1000})
    ledger = captured.value.receipt
    assert len(calls) == len(ledger["attempts"]) == 1
    attempt = ledger["attempts"][0]
    assert attempt["status"] == "provider_failed" and attempt["provider_receipt_bound"] is True
    assert attempt["provider_receipt"]["native_rollout_usage"]["usage"]["total_tokens"] == 13
    assert attempt["provider_receipt"]["provider_called"] is True
    observation = ledger["observed_native_usage_stop"]
    assert observation["prior_receipt_count"] == 1 and observation["scoped_observed_total"] == 13
    assert observation["known_for_continuing"] is False
    no_authority(observation)
    causes, error = [], captured.value
    for _ in range(10):
        if error is None:
            break
        causes.append(error)
        error = error.__cause__
    assert any(error is original for error in causes)
    assert PRIVATE not in json.dumps(ledger)
