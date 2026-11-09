"""Genuine native-view construction and authored counter adversaries; no dispatch."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set as working
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set_selection as subject
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES, PromptTokenCounter, render_model_prompt,
)
from test.api.test_semantic_metadata_view import native, native_factory  # noqa: F401
from test.api.test_semantic_working_set import descriptor


def case(native):
    repository, encoded, _ = native
    view = working.project_semantic_working_set(encoded, descriptor(encoded))
    suffix = "\nOriginal task, source obligations and acceptance remain literal."
    def rendered(text, semantic):
        return render_model_prompt(prompt=text + suffix, purpose="coding", workspace=repository,
            semantic_transport=semantic)[0]
    return dict(view=view, native_complete_prompt=rendered(encoded.native_prompt, False),
        semantic_complete_prompt=rendered(encoded.provider_prompt, True),
        working_set_complete_prompt=rendered(view.provider_prompt, True),
        provider="grok_cli", model="authored-model")


def counter(callback, **changes):
    return PromptTokenCounter(**dict(provider="grok_cli", model="authored-model",
        tokenizer_id="authored:complete-wire", tokenizer_revision="fixture-v1",
        special_token_policy="literal-only", count_tokens=callback) | changes)


def choose(native, counts=(1000, 700, 900), **changes):
    options = case(native)
    mapping = dict(zip((options[key] for key in ("native_complete_prompt", "semantic_complete_prompt",
        "working_set_complete_prompt")), counts, strict=True))
    options.update(token_counter=counter(lambda text: mapping[text]))
    options.update(changes)
    return subject.select_complete_working_set(**options)


def unknown_counts(receipt):
    for row in receipt["candidates"].values():
        assert all(row[name] is None for name in ("content_tokens", "input_tokens",
            "estimated_content_tokens", "estimated_input_tokens"))
    schema = receipt["native_output_schema"]
    assert schema["content_tokens"] is schema["estimated_content_tokens"] is None
    assert receipt["eligible_candidate_measurements_complete"] is False


def test_byte_smaller_token_worse_view_does_not_discard_semantic_alternative(native):
    options = case(native)
    old = working.select_working_set(options["view"], options["semantic_complete_prompt"],
        options["working_set_complete_prompt"])
    assert old.selected_mode == working.PROFILE
    mapping = dict(zip((options[key] for key in ("native_complete_prompt", "semantic_complete_prompt",
        "working_set_complete_prompt")), (1000, 700, 900), strict=True))
    result = subject.select_complete_working_set(**options, token_counter=counter(mapping.__getitem__))
    assert result.selected_representation == "semantic_full"
    assert result.selected_prompt == options["semantic_complete_prompt"]
    assert result.receipt["candidates"]["working_set"]["input_tokens"] == 900
    assert result.receipt["view_validation_receipt"] == old.receipt
    assert old.receipt["native_provider_input_tokens"] is None


@pytest.mark.parametrize("counts,expected", [
    ((1000, 700, 500), "working_set"), ((500, 700, 900), "raw_native"),
    ((700, 700, 700), "raw_native"), ((1000, 700, 700), "semantic_full"),
])
def test_strict_minimum_and_ties_use_fixed_nondeferral_priority(native, counts, expected):
    result = choose(native, counts)
    assert result.selected_representation == expected
    assert result.receipt["tie_break_order"] == list(subject.REPRESENTATIONS)
    assert result.receipt["working_set_endpoint_preflight_required"] is (expected == "working_set")


def test_actual_rendered_full_inputs_and_canonical_schema_are_counted_then_recounted(native):
    options, seen = case(native), []
    schema = {"type": "object", "properties": {"café": {"type": "string"}}}
    literal = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result = subject.select_complete_working_set(**options, native_output_schema=schema,
        token_counter=counter(lambda text: seen.append(text) or len(text.encode())))
    expected = [options[key] for key in ("native_complete_prompt", "semantic_complete_prompt",
        "working_set_complete_prompt")] + [literal]
    assert seen == expected + list(dict.fromkeys(expected))
    assert all("Original task, source obligations" in text for text in seen if text != literal)
    for row in result.receipt["candidates"].values():
        assert row["input_tokens"] == row["content_tokens"] + len(literal.encode())
    assert result.receipt["native_output_schema"]["canonical_json_sha256"] == sha256(literal.encode()).hexdigest()


def test_schema_cost_is_included_in_exact_budget_and_can_change_admission(native):
    options = case(native)
    schema = {"type": "object"}
    schema_text = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    mapping = {options["native_complete_prompt"]: 100, options["semantic_complete_prompt"]: 70,
        options["working_set_complete_prompt"]: 50, schema_text: 20}
    kwargs = dict(options, token_counter=counter(mapping.__getitem__), native_output_schema=schema)
    result = subject.select_complete_working_set(**kwargs, max_input_tokens=70)
    assert result.selected_representation == "working_set"
    assert result.receipt["candidates"]["working_set"]["input_tokens"] == 70
    with pytest.raises(ValueError, match="exceeds its token budget"):
        subject.select_complete_working_set(**kwargs, max_input_tokens=69)


def test_estimate_is_never_relabelled_exact_or_provider_usage(native):
    result = choose(native, token_counter=counter(lambda text: len(text.encode()),
        count_kind="estimate", compatibility_basis="utf8_byte_estimate"), max_input_tokens=200000)
    r = result.receipt
    assert r["measurement_status"] == "estimated" and r["token_count_kind"] == "estimate"
    assert r["token_budget_basis"] == "estimated_complete_input"
    assert r["tokenizer_compatibility_basis"] == "utf8_byte_estimate"
    for row in r["candidates"].values():
        assert row["input_tokens"] is row["content_tokens"] is None
        assert row["estimated_input_tokens"] == row["estimated_content_tokens"]
    assert r["native_output_schema"]["content_tokens"] is None
    assert r["native_output_schema"]["estimated_content_tokens"] == 0
    assert r["provider_token_usage"] is r["total_session_token_savings"] is None
    assert all(r[key] is False for key in subject._FALSE)


@pytest.mark.parametrize("bad", [None, "failure"])
def test_missing_or_failed_counter_falls_back_without_minimum_claim_and_cap_refuses(native, bad):
    def failure(_):
        raise RuntimeError("PRIVATE_RESPONSE_CONTENT")
    supplied = None if bad is None else counter(failure)
    result = choose(native, token_counter=supplied)
    assert result.selected_representation == "raw_native"
    assert result.receipt["measurement_status"] == ("unavailable" if bad is None else "failed")
    unknown_counts(result.receipt)
    assert "PRIVATE_RESPONSE_CONTENT" not in result.receipt_json
    with pytest.raises(ValueError, match="working consistent pinned counter"):
        choose(native, token_counter=supplied, max_input_tokens=100000)


def test_byte_excluded_raw_cannot_mask_fitting_view_or_call_counter(native):
    options, seen = case(native), []
    limit = len(options["working_set_complete_prompt"].encode())
    assert len(options["semantic_complete_prompt"].encode()) > limit
    options["native_complete_prompt"] = "N" * (limit + 1)
    result = subject.select_complete_working_set(**options, max_prompt_bytes=limit,
        token_counter=counter(lambda text: seen.append(text) or 10), max_input_tokens=10)
    assert result.selected_representation == "working_set"
    assert seen == [options["working_set_complete_prompt"]] * 2
    for name in ("raw_native", "semantic_full"):
        row = result.receipt["candidates"][name]
        assert row["byte_eligible"] is False and row["input_tokens"] is None
        assert row["exclusion_reason"] == "complete_input_byte_ceiling"


def test_unknown_counter_uses_first_byte_eligible_candidate_instead_of_byte_minimum(native):
    options = case(native)
    options["native_complete_prompt"] = "N" * (MAX_MODEL_PROMPT_BYTES + 1)
    result = subject.select_complete_working_set(**options)
    assert result.selected_representation == "semantic_full"
    assert result.receipt["selection_reason"] == "priority_fallback_no_counter"
    unknown_counts(result.receipt)


@pytest.mark.parametrize("mode", ["late", "recount", "metadata", "schema_recount", "overflow"])
def test_all_eligible_counts_are_discarded_on_any_counter_failure(native, mode):
    options, calls = case(native), []
    supplied = None
    def count(text):
        calls.append(text)
        if mode == "late" and len(calls) == 3:
            raise RuntimeError("PRIVATE_CONTEXT")
        if mode == "metadata" and len(calls) == 2:
            object.__setattr__(supplied, "compatibility_basis", "changed-fixture")
        if mode == "recount" and calls.count(text) == 2:
            return 21
        if mode == "schema_recount" and text == "{}" and calls.count(text) == 2:
            return 21
        return subject.MAX_TOKEN_COUNT if mode == "overflow" else 20
    supplied = counter(count)
    result = subject.select_complete_working_set(**options, token_counter=supplied, native_output_schema={})
    assert result.receipt["measurement_status"] == "failed"
    assert result.selected_representation == "raw_native"
    unknown_counts(result.receipt)
    assert "PRIVATE_CONTEXT" not in result.receipt_json


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "7", pytest.param(1 << 20000, id="huge_integer")])
def test_invalid_and_unbounded_counter_values_leave_no_partial_counts(native, bad):
    result = choose(native, token_counter=counter(lambda _: bad))
    assert result.receipt["measurement_status"] == "failed"
    unknown_counts(result.receipt)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_cancellation_propagates_the_original_instance(native, error):
    def cancelled(_):
        raise error
    with pytest.raises(type(error)) as raised:
        choose(native, token_counter=counter(cancelled))
    assert raised.value is error


@pytest.mark.parametrize("mutation", ["view", "descriptor", "prefix", "suffix", "duplicate", "provider", "model"])
def test_binding_and_wrapper_refusal_precedes_counter_work(native, mutation):
    options, calls = case(native), []
    if mutation == "view":
        options["view"] = replace(options["view"], provider_prompt=options["view"].provider_prompt + " ")
    elif mutation == "descriptor":
        changed = options["view"].receipt
        changed["binding"]["descriptor_sha256"] = "0" * 64
        options["view"] = replace(options["view"], receipt_json=json.dumps(changed,
            sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    elif mutation in {"prefix", "suffix"}:
        options["working_set_complete_prompt"] = ("CHANGED" + options["working_set_complete_prompt"]
            if mutation == "prefix" else options["working_set_complete_prompt"] + "CHANGED")
    elif mutation == "duplicate":
        options["semantic_complete_prompt"] += options["view"].native_prompt
    else:
        options[mutation] = "foreign-route"
    with pytest.raises(ValueError):
        subject.select_complete_working_set(**options,
            token_counter=counter(lambda text: calls.append(text) or 1))
    assert calls == []


def test_historical_view_stays_nonadmitted_and_selector_performs_no_io(native, monkeypatch):
    repository, encoded, _ = native
    options = case(native)
    (repository / "module.py").write_text("# authored source edit\n")
    historical = codec.replay_semantic_router_prompt_for_audit(prompt=encoded.native_prompt, repository=repository)
    options["view"] = working.replay_semantic_working_set_for_audit(historical, options["view"].receipt)
    def forbidden(*args, **kwargs):
        raise AssertionError("pure recommendation attempted IO")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    result = subject.select_complete_working_set(**options, token_counter=counter(lambda _: 10))
    assert result.historical_replay is True and result.receipt["historical_replay"] is True
    assert result.dispatch_usable is False and result.recommendation_only is True


def test_identical_complete_inputs_share_observed_counts_and_return_raw_priority(native):
    options, seen = case(native), []
    options["native_complete_prompt"] = options["semantic_complete_prompt"]
    result = subject.select_complete_working_set(**options,
        token_counter=counter(lambda text: seen.append(text) or 10))
    assert result.selected_representation == "raw_native"
    assert len(seen) == 5  # Three eligible occurrences and two distinct full-string recounts.


def test_validator_rejects_conflicting_counts_for_identical_full_input_hashes(native):
    options = case(native)
    options["native_complete_prompt"] = options["semantic_complete_prompt"]
    receipt = subject.select_complete_working_set(**options,
        token_counter=counter(lambda _: 10)).receipt
    receipt["candidates"]["raw_native"].update(content_tokens=11, input_tokens=11)
    with pytest.raises(ValueError, match="conflicting observations"):
        subject.validate_complete_working_set_selection_receipt(receipt)


@pytest.mark.parametrize("mutation", ["extra", "old_schema", "winner", "row_estimate", "budget",
    "schema_cost", "partial", "quality", "binding", "exclusion", "authority", "coverage", "legacy"])
def test_closed_validator_rejects_forged_accounting_and_authority(native, mutation):
    r = choose(native).receipt
    if mutation == "extra": r["unreviewed"] = 0
    elif mutation == "old_schema": r["schema"] = working.SELECTION_SCHEMA
    elif mutation == "winner": r["selected_representation"] = "working_set"
    elif mutation == "row_estimate": r["candidates"]["semantic_full"]["estimated_input_tokens"] = 700
    elif mutation == "budget": r.update(max_input_tokens=699, token_budget_basis="local_tokenizer_complete_input")
    elif mutation == "schema_cost": r["native_output_schema"]["content_tokens"] = 1
    elif mutation == "partial": r["candidates"]["raw_native"]["input_tokens"] = None
    elif mutation == "quality": r["measurement_status"] = "estimated"
    elif mutation == "binding": r["tokenizer_binding"]["model"] = "foreign-model"
    elif mutation == "exclusion": r["candidates"]["raw_native"]["byte_eligible"] = False
    elif mutation == "authority": r["dispatch_usable"] = True
    elif mutation == "coverage": r["counter_evaluation_count"] -= 1
    else: r["view_validation_receipt"]["provider_token_usage"] = 0
    with pytest.raises(ValueError):
        subject.validate_complete_working_set_selection_receipt(r)


def test_receipts_are_fresh_closed_and_contain_no_prompt_bodies_or_casual_repr(native):
    result = choose(native)
    receipt = result.receipt
    assert subject.validate_complete_working_set_selection_receipt(receipt) == receipt
    with pytest.raises(ValueError):
        working.validate_working_set_selection_receipt(receipt)
    receipt["candidates"]["semantic_full"]["content_tokens"] = 999
    assert result.receipt["candidates"]["semantic_full"]["content_tokens"] == 700
    assert result.selected_prompt not in result.receipt_json and result.selected_prompt not in repr(result)
    assert "Original task, source obligations" not in result.receipt_json


@pytest.mark.parametrize("changes", [dict(max_input_tokens=True), dict(max_prompt_bytes=0),
    dict(max_prompt_bytes=MAX_MODEL_PROMPT_BYTES + 1), dict(native_output_schema=[]),
    dict(native_output_schema={"not_finite": float("inf")})])
def test_invalid_ceilings_and_schema_refuse_before_counter_work(native, changes):
    calls = []
    with pytest.raises(ValueError):
        choose(native, token_counter=counter(lambda text: calls.append(text) or 1), **changes)
    assert calls == []


def test_no_byte_eligible_candidate_refuses_before_counter_work(native):
    calls = []
    with pytest.raises(ValueError, match="no byte-eligible"):
        choose(native, max_prompt_bytes=1, token_counter=counter(lambda text: calls.append(text) or 1))
    assert calls == []
