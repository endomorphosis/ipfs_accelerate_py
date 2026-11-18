"""Real source/capsule/socket dispatch with authored counters and provider.

The existing fixture authors only container UID admission and the model reply;
native producer, Git worktree, descriptor and registered endpoint run normally.
No paid model, remote tokenizer, provider ingestion or savings is qualified.
"""
from copy import deepcopy
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set_dispatch as subject
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set_selection as selection
from test.api.test_semantic_working_set_dispatch import (  # noqa: F401
    dispatch, evidence_service, allocated, native, provider,
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return sha256(value.encode()).hexdigest()


def token_counter(*, provider="codex_cli", model="pinned", counts=(1000, 700, 500),
                  schema_tokens=20, count_kind="tokenizer", callback=None):
    from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import coding_completion_schema
    schema = canonical(coding_completion_schema())
    seen = []
    def count(text):
        seen.append(text)
        if callback is not None:
            return callback(text)
        if text == schema:
            return schema_tokens
        name = ("working_set" if '"working_set":' in text else
            "semantic_full" if '"translated_semantic":' in text else "raw_native")
        return dict(zip(selection.REPRESENTATIONS, counts, strict=True))[name]
    counter = runner.PromptTokenCounter(provider=provider, model=model,
        tokenizer_id="authored:complete-input", tokenizer_revision="fixed-v1",
        special_token_policy="literal-only", count_kind=count_kind,
        compatibility_basis="authored_local_estimate" if count_kind == "estimate" else "caller_declaration",
        count_tokens=count)
    return counter, seen


def options(args, counter):
    return args | dict(semantic_working_set=subject.PROFILE, prompt_token_counter=counter)


@pytest.mark.parametrize("counts,winner", [
    ((100, 700, 500), "raw_native"), ((1000, 100, 500), "semantic_full"),
    ((1000, 700, 100), "working_set"), ((100, 100, 100), "raw_native"),
    ((1000, 100, 100), "semantic_full"),
])
def test_actual_selected_complete_input_and_only_deferred_preflight(dispatch, counts, winner):
    service, (_, calls), args = dispatch
    counter, seen = token_counter(counts=counts)
    _, receipt = runner.run(**options(args, counter))
    active = receipt["semantic_working_set_dispatch"]
    assert len(calls) == 1 and "prompt_selection" not in receipt
    assert active["selected_representation"] == winner
    assert active["selected_model_prompt_sha256"] == receipt["model_prompt_sha256"] == digest(calls[0][0])
    assert active["selected_router_prompt_sha256"] == receipt["router_prompt_sha256"]
    assert active["selected_model_prompt_bytes"] == len(calls[0][0].encode())
    assert subject.validate_working_set_dispatch(active) == active
    assert "semantic_working_set" not in receipt  # Historical @1 stays separate.
    assert active["preparation_only"] is True and active["provider_ingestion_verified"] is False
    assert active["total_session_token_savings"] is None
    rec = active["selection"]
    assert rec["dispatch_usable"] is False and rec["recommendation_only"] is True
    assert rec["selected_representation"] == winner and rec["eligible_candidate_measurements_complete"] is True
    assert len(set(seen)) == 3  # Complete inputs, not bare body strings.
    assert len(seen) == 9  # Three initial counts, distinct recounts, late guard.
    if winner == "raw_native":
        assert receipt["semantic_translation"] is None
        assert "semantic_prompt_components" not in receipt and "source_view_retrieval" not in receipt
        assert set(receipt["semantic_working_set_evaluated"]) == {"semantic_translation", "source_view_retrieval"}
        expected, _ = runner.render_model_prompt(prompt=service.prompt, purpose="coding", workspace=service.workspace)
        assert calls[0][0] == expected
    else:
        assert receipt["semantic_translation"] is not None and "source_view_retrieval" in receipt
        assert "semantic_working_set_evaluated" not in receipt
    assert ("semantic_working_set_preflight" in receipt) is (winner == "working_set")
    if winner == "working_set":
        assert receipt["semantic_working_set_preflight"]["provider_input_bytes"] == 0


def test_token_reversal_retains_original_semantic_candidate(dispatch):
    _, (_, calls), args = dispatch
    counter, _ = token_counter(counts=(1000, 700, 900))
    _, receipt = runner.run(**options(args, counter))
    selected = receipt["semantic_working_set_dispatch"]["selection"]
    old = selected["view_validation_receipt"]
    assert old["selected_mode"] == "capsule-references@1"
    assert old["candidate_complete_bytes"] < old["native_complete_bytes"]
    assert selected["selected_representation"] == "semantic_full"
    assert '"translated_semantic":' in calls[0][0] and '"working_set":' not in calls[0][0]
    assert selected["candidates"]["working_set"]["input_tokens"] == 900
    assert old["selection_basis"] == "complete_prompt_utf8_bytes"


@pytest.mark.parametrize("winner,counts", [
    ("raw_native", (100, 700, 500)), ("semantic_full", (1000, 100, 500)),
    ("working_set", (1000, 700, 100)),
])
def test_grok_exact_winner_uses_existing_single_router_route(dispatch, monkeypatch, winner, counts):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    _, _, args = dispatch
    calls = []
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="grok_cli", model_name="grok-4.7", catalog_revision="authored"))
    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        set_last_cli_observation("grok_cli", {"exit_code": 0})
        return "Authored provider reply; not a live model."
    monkeypatch.setattr(llm_router, "generate_text", generate)
    counter, _ = token_counter(provider="grok_cli", model="grok-4.7", counts=counts)
    _, receipt = runner.run(**options(args | dict(provider="grok_cli", model="grok-4.7"), counter))
    assert len(calls) == 1 and receipt["semantic_working_set_dispatch"]["selected_representation"] == winner
    assert receipt["model_prompt_sha256"] == digest(calls[0][0])
    assert calls[0][1]["allow_local_fallback"] is calls[0][1]["allow_cross_provider_fallback"] is False
    assert calls[0][1]["grok_max_turns"] == 128
    assert receipt["usage"] == {"exit_code": 0}


@pytest.mark.parametrize("kind", ["tokenizer", "estimate"])
@pytest.mark.parametrize("budget,accepted", [(120, True), (119, False)])
def test_native_reply_schema_is_counted_in_budget_and_contract_binding(dispatch, kind, budget, accepted):
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
    service, (state, calls), args = dispatch
    counter, seen = token_counter(counts=(1000, 700, 100), schema_tokens=20, count_kind=kind)
    state["reply"] = '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
    def observed_schema():
        literal = canonical(calls[-1][1]["codex_output_schema"])
        set_last_cli_observation("codex_cli", {**get_last_cli_observation("codex_cli"),
            "codex_output_schema_sha256": digest(literal), "codex_output_schema_bytes": len(literal.encode())})
    state["after_generate"] = observed_schema
    kwargs = options(args, counter) | dict(coding_reply_mode="ordinary-completion@1", max_input_tokens=budget)
    if not accepted:
        with pytest.raises(ValueError, match="token budget"):
            runner.run(**kwargs)
        assert not calls
        return
    _, receipt = runner.run(**kwargs)
    active = receipt["semantic_working_set_dispatch"]
    rec = active["selection"]
    expected = deepcopy(receipt["coding_reply_contract"])
    expected["response_validated"] = False
    assert active["selected_reply_contract_sha256"] == digest(canonical(expected))
    assert receipt["coding_reply_contract"]["response_validated"] is True
    assert rec["measurement_status"] == ("estimated" if kind == "estimate" else "measured")
    row = rec["candidates"]["working_set"]
    assert row["estimated_input_tokens" if kind == "estimate" else "input_tokens"] == 120
    assert row["input_tokens" if kind == "estimate" else "estimated_input_tokens"] is None
    assert rec["native_output_schema"]["estimated_content_tokens" if kind == "estimate" else "content_tokens"] == 20
    assert len(set(seen)) == 4 and len(seen) == 12
    assert rec["provider_token_usage"] is None


def test_automatic_resolver_is_used_only_when_new_profile_has_no_explicit_counter(dispatch, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer
    _, (_, calls), args = dispatch
    counter, _ = token_counter(counts=(100, 700, 500), count_kind="estimate")
    resolutions = []
    def resolve(**kwargs):
        resolutions.append(kwargs)
        return counter, {"schema": "authored-automatic-counter-provenance@1"}
    monkeypatch.setattr(automatic_prompt_tokenizer, "resolve_prompt_tokenizer", resolve)
    _, receipt = runner.run(**(args | dict(semantic_working_set=subject.PROFILE)))
    assert resolutions == [{"provider": "codex_cli", "model": "pinned"}] and len(calls) == 1
    assert receipt["semantic_working_set_dispatch"]["selection"]["measurement_status"] == "estimated"


@pytest.mark.parametrize("problem", ["fails", "nonrepeatable", "binding"])
def test_bad_counter_refuses_all_provider_calls(dispatch, problem):
    _, (_, calls), args = dispatch
    seen = {}
    def observe(text):
        if problem == "fails":
            raise RuntimeError("PRIVATE_COUNTER_BODY")
        seen[text] = seen.get(text, 0) + 1
        return seen[text]
    counter, _ = token_counter(callback=observe)
    if problem == "binding":
        counter, _ = token_counter(model="other-model")
    with pytest.raises(ValueError):
        runner.run(**options(args, counter))
    assert calls == []


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_counter_interruption_preserves_original_identity_before_provider(dispatch, kind):
    _, (_, calls), args = dispatch
    failure = kind("Authored interruption; no private reply")
    def count(_text):
        raise failure
    counter, _ = token_counter(callback=count)
    with pytest.raises(kind) as caught:
        runner.run(**options(args, counter))
    assert caught.value is failure and calls == []


@pytest.mark.parametrize("problem", ["counter", "workspace", "repository", "descriptor"])
@pytest.mark.parametrize("counts", [(100, 700, 500), (1000, 100, 500), (1000, 700, 100)])
def test_initialization_drift_prevents_generation_for_every_winner(dispatch, monkeypatch, counts, problem):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client
    service, (_, calls), args = dispatch
    counter, _ = token_counter(counts=counts)
    def initialize(*_args, **_kwargs):
        if problem == "counter":
            object.__setattr__(counter, "tokenizer_revision", "changed-after-selection")
        elif problem == "descriptor":
            original = worker_source_client.load_descriptor
            monkeypatch.setattr(worker_source_client, "load_descriptor", lambda *a, **kw:
                original(*a, **kw) | {"session_id": "changed-session"})
        else:
            root = service.workspace if problem == "workspace" else service.repository
            path = root / "mod.py"
            path.write_bytes(path.read_bytes() + b"\n# authored concurrent edit\n")
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    receipts = []
    with pytest.raises((ValueError, OSError, AssertionError)):
        runner.run(**options(args, counter), receipt_observer=receipts.append)
    assert calls == [] and receipts[-1]["router_calls"] == 0


@pytest.mark.parametrize("mutation", ["authority", "old", "selected", "model", "router", "bool_bytes", "extra", "unmeasured"])
def test_active_receipt_has_closed_shape_and_does_not_promote_recommendation(dispatch, mutation):
    _, _, args = dispatch
    counter, _ = token_counter()
    _, receipt = runner.run(**options(args, counter))
    active = deepcopy(receipt["semantic_working_set_dispatch"])
    if mutation == "authority": active["selection"]["dispatch_usable"] = True
    elif mutation == "old": active["schema"] = "supervisor-semantic-working-set-selection@1"
    elif mutation == "selected": active["selected_representation"] = "raw_native"
    elif mutation == "model": active["selected_model_prompt_sha256"] = "0" * 64
    elif mutation == "router": active["selected_router_prompt_sha256"] = "not-a-hash"
    elif mutation == "bool_bytes": active["selected_model_prompt_bytes"] = True
    elif mutation == "extra": active["execution_authority"] = True
    else: active["selection"]["measurement_status"] = "unavailable"
    with pytest.raises(ValueError):
        subject.validate_working_set_dispatch(active)
