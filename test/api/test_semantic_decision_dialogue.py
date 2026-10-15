"""Offline owner-mediated expansion dialogue contracts, without provider calls.

Authored provider replies and counters qualify protocol/accounting boundaries.
The two native-producer cases additionally exercise current source verification;
none of these fixtures establishes provider usage or session token savings.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler,
    build_text_context_references,
    render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_decision_dialogue as dialogue
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy

pytest_plugins = (
    "test.api.test_semantic_router_translation",
    "test.api.semantic_state.test_semantic_decision_packet",
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def envelope(call, *, capsule_ids=None, text="Preserve café and trailing spaces.  \n"):
    value = {
        "schema": dialogue.FINAL_SCHEMA if capsule_ids is None else dialogue.REQUEST_SCHEMA,
        "session_id": call.session_id,
        "generation": call.generation,
        "binding_sha256": call.binding_sha256,
    }
    value.update(text=text) if capsule_ids is None else value.update(capsule_ids=list(capsule_ids))
    return value


def reply(call, value=None, **receipt_changes):
    receipt = {
        "provider": call.provider,
        "model": call.model,
        "model_prompt_sha256": call.model_prompt_sha256,
        "observed_test_marker": "retain this child receipt",
    }
    receipt.update(receipt_changes)
    return dialogue.SemanticDecisionReply(encoded(envelope(call) if value is None else value), receipt)


def counter(provider="grok_cli", model="fixture-model", count=None):
    return PromptTokenCounter(
        provider=provider,
        model=model,
        tokenizer_id="authored-counter",
        tokenizer_revision="fixture@1",
        special_token_policy="literal",
        # Authored comparison deliberately exercises the packet branch. It is
        # not a tokenizer or a claim about actual compression effectiveness.
        count_tokens=count or (
            lambda text: 100 if dialogue.ROUND_SCHEMA in text else 1000
        ),
        count_kind="estimate",
        compatibility_basis="authored_fixture_only",
    )


@pytest.fixture
def decision_case(parent, tmp_path, monkeypatch):
    """Real packet composition with an explicitly authored producer verifier."""
    repository = tmp_path / "repository"
    repository.mkdir()
    artifact = repository / ".runtime/semantic/worker-context.json"
    artifact.parent.mkdir(parents=True)
    parent = copy.deepcopy(parent)
    for index, capsule in enumerate(parent["capsules"]):
        capsule["metadata"]["dialogue_marker"] = f"CAPSULE-BODY-{index}"
    text = encoded(parent)
    artifact.write_text(text)
    (repository / "target.py").write_text(parent["raw_sources"]["target.py"])
    state = {"stale": False, "verifications": 0}

    def verified(root, *, artifact: str, text: str, task_id: str, current: bool):
        state["verifications"] += 1
        if state["stale"]:
            raise codec.SemanticTranslationError("authored source changed")
        assert root == repository and current is True
        assert artifact == ".runtime/semantic/worker-context.json"
        assert text == encoded(parent) and task_id == parent["task_id"]
        return copy.deepcopy(parent), tuple(row["stable_symbol_id"] for row in parent["capsules"])

    monkeypatch.setattr(codec, "_verified_semantic", verified)
    references = build_text_context_references(
        text,
        reference_prefix="semantic-context",
        kind="semantic-context",
        path=artifact.relative_to(repository).as_posix(),
        repository_id="repo:dialogue-fixture",
        tree_id="tree:dialogue-fixture",
        required=True,
        chunk_bytes=1201,
    )
    compiled = ContextCompiler(
        ContextBudget(max_input_tokens=32768, max_items=128, max_item_bytes=16384,
                      max_serialized_bytes=262144)
    ).compile(
        repository_id="repo:dialogue-fixture", tree_id="tree:dialogue-fixture",
        objective_id=parent["task_id"], objective_revision="sha256:task",
        policy_id="policy:fixture", policy_revision="sha256:policy", caller="supervisor:fixture",
        stage="implementation", goal={"id": parent["task_id"]},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": ["target.py"]},
        acceptance={"criteria": ["Preserve the literal acceptance condition"]}, evidence=references,
    )
    suffix = "\nLiteral instruction: preserve café and spaces.  \n"
    native_prompt = render_context_capsule(compiled.capsule) + suffix
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=(ids[1],), optional_capsule_ids=(ids[0], ids[2]))
    return SimpleNamespace(
        repository=repository, artifact=artifact, parent=parent, native_prompt=native_prompt,
        policy=policy, ids=ids, state=state, suffix=suffix,
    )


def run(case, invoke, **changes):
    arguments = {
        "native_prompt": case.native_prompt,
        "repository": case.repository,
        "workspace": case.repository,
        "policy": case.policy,
        "provider": "grok_cli",
        "model": "fixture-model",
        "invoke": invoke,
        "count_tokens": counter(),
    }
    arguments.update(changes)
    return dialogue.run_semantic_decision_dialogue(**arguments)


def retained(error, count):
    receipt = error.value.receipt
    assert len(receipt["attempts"]) == count
    assert receipt["whole_session_token_savings"] is None
    return receipt


def test_final_reply_preserves_literal_text_and_bound_frozen_call(decision_case):
    calls = []

    def invoke(call):
        calls.append(call)
        with pytest.raises(FrozenInstanceError):
            call.generation = 99
        protocol = json.loads(call.prompt.splitlines()[0])
        assert protocol["session_id"] == call.session_id
        assert protocol["generation"] == call.generation
        assert protocol["binding_sha256"] == call.binding_sha256
        assert protocol["provider"] == call.provider == "grok_cli"
        assert protocol["model"] == call.model == "fixture-model"
        assert decision_case.suffix in call.prompt
        assert "CAPSULE-BODY-1" in call.prompt
        assert "CAPSULE-BODY-0" not in call.prompt and "CAPSULE-BODY-2" not in call.prompt
        assert call.model_prompt_sha256 != hashlib.sha256(call.prompt.encode()).hexdigest()
        return reply(call)

    artifact_before = decision_case.artifact.read_bytes()
    result = run(decision_case, invoke)
    assert result.text == "Preserve café and trailing spaces.  \n"
    assert len(calls) == len(result.receipt["attempts"]) == 1
    assert result.receipt["attempts"][0]["provider_receipt"]["observed_test_marker"]
    assert result.receipt["whole_session_token_savings"] is None
    assert decision_case.state["verifications"] >= 4
    assert decision_case.artifact.read_bytes() == artifact_before
    assert "rejection_observation" not in result.receipt["attempts"][0]


def test_selected_expansion_is_delivered_without_other_optional_body(decision_case):
    calls = []

    def invoke(call):
        calls.append(call)
        if len(calls) == 1:
            return reply(call, envelope(call, capsule_ids=(decision_case.ids[0],)))
        assert "CAPSULE-BODY-0" in call.prompt and "CAPSULE-BODY-2" not in call.prompt
        assert decision_case.suffix in call.prompt
        return reply(call)

    result = run(decision_case, invoke)
    assert len(calls) == len(result.receipt["attempts"]) == 2
    assert calls[0].session_id == calls[1].session_id
    assert calls[0].call_id != calls[1].call_id and calls[0].generation < calls[1].generation
    assert len(result.receipt["expansions"]) == 1
    assert "CAPSULE-BODY-0" not in encoded(result.receipt)
    assert all("rejection_observation" not in row for row in result.receipt["attempts"])


@pytest.mark.parametrize("field,replacement", [
    ("session_id", "foreign-session"),
    ("generation", 9000),
    ("binding_sha256", "0" * 64),
    ("schema", "unregistered-dialogue@1"),
])
def test_foreign_or_unregistered_reply_is_rejected_after_receipt_retention(
    decision_case, field, replacement,
):
    def invoke(call):
        value = envelope(call)
        value[field] = replacement
        return reply(call, value)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    receipt = retained(error, 1)
    assert receipt["attempts"][0]["provider_receipt"]["observed_test_marker"]


def test_duplicate_json_keys_are_rejected_after_call(decision_case):
    def invoke(call):
        text = encoded(envelope(call))
        text = text[:-1] + ',"generation":' + str(call.generation) + "}"
        return dialogue.SemanticDecisionReply(text, reply(call).receipt)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    retained(error, 1)


def test_unknown_fields_cannot_supply_policy_or_authority(decision_case):
    def invoke(call):
        value = envelope(call)
        value.update(policy={"optional_capsule_ids": []}, completion_authority=True)
        return reply(call, value)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    retained(error, 1)


def test_plain_prose_is_not_a_final_protocol_response(decision_case):
    def invoke(call):
        return dialogue.SemanticDecisionReply("The task is complete.", reply(call).receipt)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    retained(error, 1)


def test_capsule_request_cannot_expand_required_or_foreign_bodies(decision_case):
    def invoke(call):
        return reply(call, envelope(call, capsule_ids=(decision_case.ids[1], "foreign-capsule")))

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    receipt = retained(error, 1)
    assert receipt["expansions"] == []
    # A valid response followed by an expansion refusal has no stale parser phase.
    assert "rejection_observation" not in receipt["attempts"][0]


def test_repeating_an_expansion_request_makes_no_third_call(decision_case):
    calls = []

    def invoke(call):
        calls.append(call)
        return reply(call, envelope(call, capsule_ids=(decision_case.ids[0],)))

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    assert len(calls) == 2
    assert len(retained(error, 2)["expansions"]) == 1


def test_exhausted_call_budget_does_not_start_continuation(decision_case):
    def invoke(call):
        return reply(call, envelope(call, capsule_ids=(decision_case.ids[0],)))

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke, max_calls=1)
    retained(error, 1)


def test_expansion_byte_bound_fails_without_second_call(decision_case):
    def invoke(call):
        return reply(call, envelope(call, capsule_ids=(decision_case.ids[0],)))

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke, max_expansion_bytes=32)
    assert retained(error, 1)["expansions"] == []


def test_complete_prompt_budget_is_checked_before_provider_call(decision_case):
    calls = []

    def invoke(call):
        calls.append(call)
        return reply(call)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke, max_input_tokens=1)
    assert calls == []
    retained(error, 0)


@pytest.mark.parametrize("limit,expected_calls", [(99, 0), (150, 1), (200, 2)])
def test_total_input_budget_counts_full_replay_before_continuation(decision_case, limit, expected_calls):
    calls = []

    def invoke(call):
        calls.append(call)
        return reply(call, envelope(call, capsule_ids=(decision_case.ids[0],))) if len(calls) == 1 else reply(call)

    if expected_calls == 2:
        receipt = run(decision_case, invoke, max_total_input_tokens=limit).receipt
        assert receipt["local_input_content_token_sum"] == 200
    else:
        with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
            run(decision_case, invoke, max_total_input_tokens=limit)
        receipt = retained(error, expected_calls)
        assert receipt["failure_reason"] == "total_input_budget_exceeded"
        if expected_calls:
            assert receipt["expansions"][0]["included_in_invocation_attempts"] == []
    assert len(calls) == expected_calls
    assert receipt["total_input_budget_basis"] == "estimated_content_tokens"
    assert receipt["token_budget_guaranteed"] is False


def test_cancelled_before_first_call_has_no_provider_charge(decision_case):
    calls = []

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, lambda call: calls.append(call), cancelled=lambda: True)
    assert calls == []
    retained(error, 0)


def test_cancelled_after_provider_reply_retains_attempt_and_child_receipt(decision_case):
    state = {"cancelled": False}

    def invoke(call):
        state["cancelled"] = True
        return reply(call)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke, cancelled=lambda: state["cancelled"])
    receipt = retained(error, 1)
    assert receipt["attempts"][0]["provider_receipt"]["observed_test_marker"]
    assert receipt["failure_reason"] == "cancelled"
    assert receipt["attempts"][0]["rejection_observation"] == {
        "schema": "supervisor-semantic-decision-rejection-observation@1",
        "phase": "post_return_control_check", "reason_code": "post_return_control_check_refused",
        "exception_type": "RuntimeError",
    }


def test_source_change_after_provider_reply_refuses_final_delivery(decision_case):
    def invoke(call):
        decision_case.state["stale"] = True
        return reply(call)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    receipt = retained(error, 1)
    assert receipt["attempts"][0]["provider_receipt"]["observed_test_marker"]


def test_provider_exception_retains_attempt_and_does_not_retry(decision_case):
    calls = []

    def invoke(call):
        calls.append(call)
        raise TimeoutError("authored timeout after physical invocation")

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    assert len(calls) == 1
    retained(error, 1)


def test_typed_provider_failure_keeps_partial_child_receipt(decision_case):
    partial = {"provider": "grok_cli", "model": "fixture-model", "timed_out": True,
               "usage": {"prompt_tokens": 80, "completion_tokens": 3}}

    def invoke(call):
        raise dialogue.SemanticDecisionProviderFailure("authored partial timeout", receipt=partial)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    receipt = retained(error, 1)
    assert receipt["attempts"][0]["provider_receipt"] == partial
    assert receipt["provider_usage_complete"] is False


@pytest.mark.parametrize("oversized", [False, True])
def test_failed_receipt_must_be_bound_and_bounded_before_usage_reconciliation(decision_case, oversized):
    def invoke(call):
        child = reply(call).receipt
        if oversized:
            child["untrusted_detail"] = "x" * 65537
        else:
            child["provider"] = "foreign-provider"
        raise dialogue.SemanticDecisionProviderFailure(receipt=child)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    receipt = retained(error, 1)
    assert receipt["attempts"][0]["provider_receipt_bound"] is False
    assert receipt["missing_provider_receipt_count"] == 1
    assert receipt["provider_usage_complete"] is False


def test_counter_drift_for_identical_complete_prompt_refuses_before_call(decision_case):
    seen = {}
    calls = []

    def drifting(text):
        seen[text] = seen.get(text, 0) + 1
        return (100 if dialogue.ROUND_SCHEMA in text else 1000) + seen[text]

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, lambda call: calls.append(call), count_tokens=counter(count=drifting))
    assert calls == []
    assert retained(error, 0)["failure_reason"] == "tokenizer_inconsistent"


def test_deadline_after_reply_prevents_delivery_without_erasing_attempt(decision_case, monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(dialogue.time, "monotonic", lambda: clock["now"])

    def invoke(call):
        assert 1 <= call.timeout <= 90
        clock["now"] = 90.0
        return reply(call)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    receipt = retained(error, 1)
    assert receipt["attempts"][0]["provider_receipt"]["observed_test_marker"]


def test_child_receipt_binding_mismatch_is_retained_then_rejected(decision_case):
    def invoke(call):
        return reply(call, provider="codex_cli")

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke)
    assert retained(error, 1)["attempts"][0]["provider_receipt"]["provider"] == "codex_cli"


def test_counter_cannot_rebind_provider_or_model(decision_case):
    calls = []

    with pytest.raises((ValueError, dialogue.SemanticDecisionDialogueError)):
        run(decision_case, lambda call: calls.append(call), count_tokens=counter(model="foreign-model"))
    assert calls == []


@pytest.mark.parametrize("wrapped", [False, True])
def test_codex_cumulative_snapshots_charge_one_native_session_once(decision_case, wrapped):
    calls = []
    thread = "authored-dialogue-thread-123456"

    def invoke(call):
        calls.append(call)
        ordinal = len(calls)
        inputs, outputs = (100, 20) if ordinal == 1 else (170, 30)
        child = reply(call).receipt
        child.update(
            schema="router-implementation-invocation@1", invocation_id=call.call_id,
            workspace=str(decision_case.repository), status="provider_returned",
            usage={"thread_id": thread, "model_id": call.model},
            native_rollout_usage={
                "schema": "native-codex-cumulative-usage@1", "thread_id": thread,
                "rollout_sha256": hashlib.sha256(str(ordinal).encode()).hexdigest(),
                "usage": {"input_tokens": inputs, "output_tokens": outputs,
                          "cached_input_tokens": inputs // 2, "reasoning_output_tokens": outputs // 2,
                          "total_tokens": inputs + outputs},
                "usage_available": True, "observed_token_count_records": ordinal,
                "task_complete_observed": ordinal == 2, "cache_included_in_input": True,
                "completion_basis": "native_lifecycle_after_latest_usage@1",
                "lifecycle_malformed_records": 0, "usage_lifecycle_complete": ordinal == 2,
            },
        )
        value = envelope(call, capsule_ids=(decision_case.ids[0],)) if ordinal == 1 else envelope(call)
        if wrapped:
            child = {**reply(call).receipt, "schema": "router-semantic-decision-provider@1",
                     "native_invocation": child, "provider_called": True}
        return dialogue.SemanticDecisionReply(encoded(value), child)

    result = run(decision_case, invoke, provider="codex_cli", count_tokens=counter(provider="codex_cli"))
    usage = result.receipt["codex_cumulative_usage"]
    assert len(calls) == 2 and usage["native_sessions_observed"] == 1
    assert usage["unique_invocation_ids"] == 2 and usage["provider_total_tokens"] == 200
    assert usage["cached_and_reasoning_tokens_are_subsets"] is True
    assert result.receipt["provider_usage_complete"] is True
    assert result.receipt["whole_session_token_savings"] is None
    assert all("rejection_observation" not in row for row in result.receipt["attempts"])


def test_native_fallback_requires_no_protocol_envelope(decision_case):
    calls = []
    literal = "Plain native reply preserves café and spaces.  \n"

    def invoke(call):
        calls.append(call)
        assert call.generation == 0
        assert call.prompt == decision_case.native_prompt
        return dialogue.SemanticDecisionReply(literal, reply(call).receipt)

    result = run(decision_case, invoke, count_tokens=counter(count=lambda text: 100))
    assert result.text == literal
    assert result.receipt["selected_representation"] == "native"
    assert len(calls) == len(result.receipt["attempts"]) == 1
    assert result.receipt["expansions"] == []
    assert "rejection_observation" not in result.receipt["attempts"][0]


def test_actual_native_producer_packet_expands_without_modifying_source(native):
    repository, artifact, prompt, source = native
    parent = json.loads(artifact.read_text())
    ids = tuple(row["capsule_cid"] for row in parent["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids[1:], optional_capsule_ids=(ids[0],))
    calls = []
    artifact_before = artifact.read_bytes()

    def invoke(call):
        calls.append(call)
        return reply(call, envelope(call, capsule_ids=(ids[0],))) if len(calls) == 1 else reply(call)

    result = dialogue.run_semantic_decision_dialogue(
        native_prompt=prompt, repository=repository, workspace=repository, policy=policy,
        provider="grok_cli", model="fixture-model", invoke=invoke, count_tokens=counter(),
    )
    assert len(calls) == 2 and len(result.receipt["expansions"]) == 1
    assert artifact.read_bytes() == artifact_before
    assert (repository / "mod.py").read_text() == source
    assert result.receipt["whole_session_token_savings"] is None
    assert all("rejection_observation" not in row for row in result.receipt["attempts"])


def test_actual_native_source_mutation_after_call_cannot_deliver_final(native):
    repository, artifact, prompt, source = native
    ids = tuple(row["capsule_cid"] for row in json.loads(artifact.read_text())["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids, optional_capsule_ids=())
    calls = []
    private_marker = "PRIVATE-POST-RETURN-SOURCE-AND-REPLY"

    def invoke(call):
        calls.append(call)
        (repository / "mod.py").write_text(source + "\n# " + private_marker + "\n")
        return reply(call, envelope(call, text=private_marker), provider_called=True)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        dialogue.run_semantic_decision_dialogue(
            native_prompt=prompt, repository=repository, workspace=repository, policy=policy,
            provider="grok_cli", model="fixture-model", invoke=invoke, count_tokens=counter(),
        )
    receipt = retained(error, 1)
    assert len(calls) == 1 and receipt["attempts"][0]["status"] == "rejected"
    assert receipt["attempts"][0]["provider_receipt_bound"] is True
    assert receipt["provider_dispatches_reported"] == 1 and receipt["missing_provider_receipt_count"] == 0
    assert receipt["attempts"][0]["rejection_observation"] == {
        "schema": "supervisor-semantic-decision-rejection-observation@1",
        "phase": "post_return_current_verification", "reason_code": "current_context_refused",
        "exception_type": type(error.value.__cause__).__name__,
    }
    assert receipt["failure_reason"] == type(error.value.__cause__).__name__
    assert private_marker not in encoded(receipt)


@pytest.mark.parametrize("malformed", [False, True])
def test_native_producer_response_rejection_identifies_contract_without_private_reply(native, malformed):
    repository, artifact, prompt, _ = native
    ids = tuple(row["capsule_cid"] for row in json.loads(artifact.read_text())["capsules"])
    policy = SemanticDecisionPolicy(required_capsule_ids=ids, optional_capsule_ids=())
    calls = []
    private_marker = "PRIVATE-REPLY-BINDING-AND-PARSER-DOCUMENT"
    children = []

    def invoke(call):
        calls.append(call)
        value = envelope(call, text=private_marker)
        value["binding_sha256"] = private_marker
        child = reply(call, value, provider_called=True)
        children.append(child.receipt)
        return dialogue.SemanticDecisionReply(private_marker if malformed else child.text, child.receipt)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        dialogue.run_semantic_decision_dialogue(native_prompt=prompt, repository=repository, workspace=repository,
            policy=policy, provider="grok_cli", model="fixture-model", invoke=invoke, count_tokens=counter())
    receipt = retained(error, 1)
    assert len(calls) == 1 and receipt["attempts"][0]["status"] == "rejected"
    assert receipt["attempts"][0]["provider_receipt_bound"] is True
    assert receipt["attempts"][0]["provider_receipt"] == children[0]
    assert receipt["provider_dispatches_reported"] == 1 and receipt["missing_provider_receipt_count"] == 0
    expected_type = "JSONDecodeError" if malformed else "ValueError"
    assert receipt["failure_reason"] == expected_type
    assert receipt["attempts"][0]["rejection_observation"] == {
        "schema": "supervisor-semantic-decision-rejection-observation@1",
        "phase": "response_contract_validation", "reason_code": "response_contract_refused",
        "exception_type": expected_type,
    }
    assert private_marker not in encoded(receipt)


def test_post_return_control_error_keeps_original_exception_without_its_private_message(decision_case):
    private_marker = "PRIVATE-OWNER-CONTROL-EXCEPTION-MESSAGE"
    original = ValueError(private_marker)
    called = []

    def cancelled():
        if called:
            raise original
        return False

    def invoke(call):
        called.append(call)
        return reply(call, provider_called=True)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke, cancelled=cancelled)
    receipt = retained(error, 1)
    assert len(called) == 1 and error.value.__cause__ is original
    assert receipt["failure_reason"] == "ValueError"
    assert receipt["attempts"][0]["rejection_observation"] == {
        "schema": "supervisor-semantic-decision-rejection-observation@1",
        "phase": "post_return_control_check", "reason_code": "post_return_control_check_refused",
        "exception_type": "ValueError",
    }
    assert private_marker not in encoded(receipt)


@pytest.mark.parametrize("oversized", [False, True])
def test_native_fallback_refusal_observes_native_validation_without_response_text(decision_case, oversized):
    private_marker = "PRIVATE-NATIVE-RESPONSE"
    text = private_marker * 4000 if oversized else ""
    calls = []

    def invoke(call):
        calls.append(call)
        return dialogue.SemanticDecisionReply(text, reply(call, provider_called=True).receipt)

    with pytest.raises(dialogue.SemanticDecisionDialogueError) as error:
        run(decision_case, invoke, count_tokens=counter(count=lambda _: 100))
    receipt = retained(error, 1)
    assert len(calls) == 1 and receipt["selected_representation"] == "native"
    assert receipt["failure_reason"] == "invalid_native_response"
    assert receipt["attempts"][0]["provider_receipt_bound"] is True
    assert receipt["attempts"][0]["rejection_observation"] == {
        "schema": "supervisor-semantic-decision-rejection-observation@1",
        "phase": "native_fallback_response_validation", "reason_code": "native_response_refused",
        "exception_type": "RuntimeError",
    }
    assert private_marker not in encoded(receipt)
