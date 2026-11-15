"""Historical matched-input checks using real source fixtures/authored calls.

All provider responses/counts here are authored. Byte consistency is never an
attestation of historical owners, provider ingestion, native usage, or savings.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_residual_selection as selection
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_residual_selection_replay as archive
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_invocation_replay as ordinary
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import freeze_repository_world_packet
from test.api.test_repository_context_compiler import compile_context, obligation
from test.api.test_repository_world_dispatch import (  # noqa: F401
    TASK, dispatch_case, world_session, required, scheduler, counter,
)
from test.api.test_repository_world_attempt_runner import make_journal
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_attempt_runner import run_repository_world_packet_attempt
from test.api.test_repository_world_residual_focus import request, residual


def wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def matched(dispatch_case, tmp_path):
    baseline = dispatch_case[2]
    focused = request(dict(baseline=baseline))
    pair = selection.prepare_repository_world_residual_alternatives(baseline=baseline, focused=focused)
    full_artifact = freeze_repository_world_packet(pair.full_packet)
    focused_artifact = freeze_repository_world_packet(pair.focused_packet)
    return dict(case=dispatch_case, baseline=baseline, pair=pair, full=full_artifact, focused=focused_artifact,
        tmp_path=tmp_path)


def capture(matched, *, token_counter="bytes", **changes):
    declared = counter() if token_counter == "bytes" else token_counter
    options = dict(prompt=TASK, provider="codex_cli", model="pinned", timeout=30, max_output_tokens=128,
        world_task_packet=matched["pair"].full_packet,
        world_task_residual_alternatives=matched["pair"], prompt_token_counter=declared)
    options.update(changes)
    journal, _root = make_journal(matched["tmp_path"], workspace=matched["case"][0])
    _, invocation, _diagnostic = run_repository_world_packet_attempt(journal=journal, runner_options=options)
    args = dict(full_packet_artifact=matched["full"], expected_full_packet_sha256=sha(matched["full"]),
        focused_packet_artifact=matched["focused"], expected_focused_packet_sha256=sha(matched["focused"]),
        invocation_receipt=invocation, expected_invocation_sha256=archive.invocation_receipt_sha256(invocation))
    data = archive.freeze_repository_world_residual_selection(**args)
    return invocation, args, data


@pytest.fixture
def captured(matched):
    invocation, args, data = capture(matched)
    return matched, invocation, args, data


def replay(data, **changes):
    return archive.replay_repository_world_residual_selection(data, expected_sha256=sha(data), **changes)


def test_actual_selected_complete_input_original_receipt_and_both_candidates_survive(captured):
    matched, invocation, _args, data = captured
    result = replay(data)
    observed = result.receipt()
    assert result.model_prompt == matched["case"][3]["seen"][0][0]
    assert result.recorded_invocation_receipt() == invocation
    assert observed["selected_representation"] == invocation[archive.SELECTION_FIELD]["selected_representation"]
    assert observed["both_complete_model_inputs_reconstructed"] is True
    assert observed["matched_residual_record_verified"] is observed["matched_required_references_verified"] is True
    assert observed["recorded_counts_arithmetic_checked"] is True
    assert observed["local_counter_verified"] is False and observed["local_counter_verification_status"] == "unknown"
    assert observed["trusted_counter_supplied"] is False
    assert observed["invocation_sha256"] == sha(wire(invocation))
    assert observed["derived_projection_independently_retained"] is False
    assert all(observed[key] is False for key in archive.FALSE_FIELDS)
    assert observed["total_session_token_savings"] is None
    assert not hasattr(result, "validate_current") and not hasattr(result, "require_provider_dispatch")
    assert matched["baseline"]._capsule.goal["intent"] not in wire(observed).decode()
    assert str(matched["case"][0]) not in wire(observed).decode()
    detached = result.recorded_invocation_receipt()
    detached["usage"]["prompt_tokens"] = 999999
    assert result.recorded_invocation_receipt() == invocation


@pytest.mark.parametrize("kind", ["tie", "unavailable", "failed"])
def test_full_fallback_keeps_identical_residual_and_original_optional_population(matched, kind):
    def failed(_text):
        raise RuntimeError("authored tokenizer failure")
    declared = {"tie": counter(count_tokens=lambda _text: 1), "unavailable": None,
        "failed": counter(count_tokens=failed)}[kind]
    invocation, _args, data = capture(matched, token_counter=declared)
    result = replay(data)
    assert result.receipt()["selected_representation"] == "full"
    assert matched["pair"].full_packet.prompt in result.model_prompt
    assert matched["pair"].receipt()["residual_packet_id"] in result.model_prompt
    assert invocation["router_prompt_sha256"] == sha(matched["pair"].full_packet.prompt.encode())
    assert invocation["world_task_packet"]["provider_retained_parent"] is False


def test_explicit_trusted_counter_repeats_both_complete_inputs_only(captured):
    _matched, invocation, _args, data = captured
    seen = []
    trusted = counter(count_tokens=lambda text: (seen.append(text), len(text.encode()))[1])
    result = replay(data, token_counter=trusted)
    assert len(seen) == 2
    assert {sha(text.encode()) for text in seen} == {
        item["model_prompt_sha256"] for item in invocation[archive.SELECTION_FIELD]["candidates"].values()}
    assert result.receipt()["local_counter_verified"] is True
    assert result.receipt()["provider_tokenizer_equivalence_verified"] is False
    assert result.receipt()["provider_usage_verified"] is result.receipt()["completion_authority"] is False


def test_trusted_counter_identity_mismatch_refuses_before_callback(captured):
    *_, data = captured
    declared = counter(tokenizer_revision="another-revision", count_tokens=lambda _text:
        pytest.fail("mismatched recorded counter was called"))
    with pytest.raises(archive.RepositoryWorldResidualSelectionReplayError, match="trusted_counter_binding"):
        replay(data, token_counter=declared)


def test_recorded_consistent_counts_are_not_claimed_authenticated_or_recomputed(captured):
    _matched, invocation, args, _data = captured
    forged = deepcopy(invocation)
    envelope = forged[archive.SELECTION_FIELD]
    dispatch = envelope.pop("dispatch_rechecked")
    for candidate in envelope["candidates"].values():
        candidate["content_tokens"] += 11
        candidate["input_tokens"] += 11
    chosen = envelope["candidates"][envelope["selected_representation"]]
    envelope["selected_content_tokens"] = chosen["content_tokens"]
    envelope["selected_input_tokens"] = chosen["input_tokens"]
    forged["world_task_packet"]["prompt_measurement"] = selection.repository_world_residual_selected_measurement(envelope)
    envelope["dispatch_rechecked"] = dispatch
    data = archive.freeze_repository_world_residual_selection(**(args | dict(invocation_receipt=forged,
        expected_invocation_sha256=archive.invocation_receipt_sha256(forged))))
    assert replay(data).receipt()["local_counter_verified"] is False
    with pytest.raises(archive.RepositoryWorldResidualSelectionReplayError, match="trusted_counter_reproduction"):
        replay(data, token_counter=counter())


@pytest.mark.parametrize("field", ["expected_full_packet_sha256", "expected_focused_packet_sha256", "expected_invocation_sha256"])
def test_each_original_component_pin_is_required(captured, field):
    _matched, _invocation, args, _data = captured
    with pytest.raises(ValueError):
        archive.freeze_repository_world_residual_selection(**(args | {field: "0" * 64}))


def test_original_receipt_bytes_have_exact_pin_and_no_projection_replacement(captured):
    _matched, invocation, args, data = captured
    raw = wire(invocation)
    assert archive.freeze_repository_world_residual_selection(**(args | {"invocation_receipt": raw})) == data
    changed = deepcopy(invocation)
    changed["usage"]["prompt_tokens"] = 999
    with pytest.raises(ValueError):
        archive.freeze_repository_world_residual_selection(**(args | {"invocation_receipt": changed}))
    with pytest.raises(ValueError):
        archive.replay_repository_world_residual_selection(data, expected_sha256="0" * 64)


@pytest.mark.parametrize("mutation", ["choice", "count", "candidate_hash", "reply_schema", "dispatch", "unknown", "selected_measurement", "retained_parent_true", "retained_parent_zero"])
def test_resealed_malformed_original_receipt_refuses(captured, mutation):
    _matched, invocation, args, _data = captured
    changed = deepcopy(invocation)
    envelope = changed[archive.SELECTION_FIELD]
    if mutation == "choice":
        envelope["selected_representation"] = "full" if envelope["selected_representation"] == "focused" else "focused"
    elif mutation == "count":
        envelope["candidates"]["full"]["input_tokens"] += 1
    elif mutation == "candidate_hash":
        envelope["candidates"]["full"]["model_prompt_sha256"] = "0" * 64
    elif mutation == "reply_schema":
        envelope["native_output_schema_sha256"] = "0" * 64
    elif mutation == "dispatch":
        envelope["dispatch_rechecked"] = not envelope["dispatch_rechecked"]
    elif mutation == "unknown":
        envelope["unrecognized_authority"] = False
    elif mutation == "selected_measurement":
        changed["world_task_packet"]["prompt_measurement"]["selected_input_tokens"] += 1
    elif mutation == "retained_parent_true":
        changed["world_task_packet"]["provider_retained_parent"] = True
    else:
        changed["world_task_packet"]["provider_retained_parent"] = 0
    with pytest.raises(ValueError):
        archive.freeze_repository_world_residual_selection(**(args | dict(invocation_receipt=changed,
            expected_invocation_sha256=archive.invocation_receipt_sha256(changed))))


def test_original_failure_free_baseline_is_never_a_legal_full_fallback(captured):
    matched, _invocation, args, _data = captured
    original = freeze_repository_world_packet(matched["baseline"])
    with pytest.raises(archive.RepositoryWorldResidualSelectionReplayError, match="matched_recipe_and_residual"):
        archive.freeze_repository_world_residual_selection(**(args | dict(full_packet_artifact=original,
            expected_full_packet_sha256=sha(original))))


@pytest.mark.parametrize("different", ["residual", "required", "goal"])
def test_individually_valid_foreign_focus_cannot_be_mixed_with_original_full(captured, different):
    matched, _invocation, args, _data = captured
    baseline = matched["baseline"]
    if different in {"required", "goal"}:
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import prepare_repository_world_packet
        original = json.loads(baseline._replay_material_json)
        required_refs = tuple(ref for ref in baseline._capsule.evidence if ref.required)
        if different == "required":
            required_refs = (obligation(required_refs[0].reference_id, summary="Changed independently required obligation."), *required_refs[1:])
        changes = dict(authority=original["capsule"]["authority"]["task"])
        if different == "goal":
            changes["goal"] = {"intent": "A different bounded objective."}
        compiled = compile_context(baseline._adapter, required_refs, **changes)
        recipe = dict(original["recipe"])
        for name in ("unresolved_obligations", "validation_commands", "proof_obligations",
                "acceptance_ids", "allowed_effects", "write_paths"):
            recipe[name] = tuple(recipe[name])
        if recipe["current_world"] is not None:
            from ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model import RepositoryWorldState
            recipe["current_world"] = RepositoryWorldState.from_dict(recipe["current_world"])
        baseline = prepare_repository_world_packet(adapter=baseline._adapter, compilation=compiled, **recipe)
    extra = {"residual_packet": residual(baseline, obligation_ids=("another:residual-obligation",))} if different == "residual" else {}
    focused = request(dict(baseline=baseline), **extra)
    foreign = freeze_repository_world_packet(focused.packet)
    with pytest.raises(archive.RepositoryWorldResidualSelectionReplayError):
        archive.freeze_repository_world_residual_selection(**(args | dict(focused_packet_artifact=foreign,
            expected_focused_packet_sha256=sha(foreign))))


def test_complete_reply_schema_and_instruction_are_replayed_and_counted(matched, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    original = llm_router.generate_text
    def acknowledged(prompt, **options):
        original(prompt, **options)
        schema = wire(options["codex_output_schema"])
        set_last_cli_observation("codex_cli", dict(exit_code=0, prompt_tokens=15, completion_tokens=4,
            codex_output_schema_sha256=sha(schema), codex_output_schema_bytes=len(schema)))
        return '{"schema":"supervisor-coding-completion@1","status":"candidate_ready"}'
    monkeypatch.setattr(llm_router, "generate_text", acknowledged)
    invocation, _args, data = capture(matched, coding_reply_mode="ordinary-completion@1")
    counted = []
    result = replay(data, token_counter=counter(count_tokens=lambda text:
        (counted.append(text), len(text.encode()))[1]))
    assert len(counted) == 3
    assert sha(counted[-1].encode()) == invocation["coding_reply_contract"]["native_output_schema_sha256"]
    assert result.model_prompt == matched["case"][3]["seen"][0][0]
    assert "Supervisor coding reply contract" in result.model_prompt
    assert result.receipt()["local_counter_verified"] is True


def test_replay_never_opens_source_owner_paths_or_provider(captured, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_packet as owner
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
    matched, _invocation, _args, data = captured
    def denied(*_args, **_kwargs):
        pytest.fail("historical matched replay attempted native owner, file, or provider access")
    with monkeypatch.context() as patch:
        patch.setattr(owner, "prepare_repository_world_packet", denied)
        patch.setattr(RepositorySourceSession, "__init__", denied)
        patch.setattr(llm_router, "get_llm_provider", denied)
        patch.setattr(llm_router, "generate_text", denied)
        for method in ("read_text", "read_bytes", "resolve", "stat"):
            patch.setattr(Path, method, denied)
        result = replay(data)
    assert result.model_prompt == matched["case"][3]["seen"][0][0]


def test_legacy_invocation_projection_refuses_to_discard_selection(captured):
    matched, invocation, _args, _data = captured
    selected = invocation[archive.SELECTION_FIELD]["selected_representation"]
    packet = matched[selected]
    with pytest.raises(ordinary.RepositoryWorldInvocationReplayError, match="incompatible"):
        ordinary.freeze_repository_world_invocation(packet_artifact=packet,
            expected_packet_sha256=sha(packet), invocation_receipt=invocation)


def test_future_recorded_fields_and_partial_usage_are_retained_without_authority(captured):
    _matched, invocation, args, _data = captured
    extended = deepcopy(invocation)
    extended["future_observation"] = {"values": [None, "duplicate", "duplicate"], "weight": 0.5}
    extended["usage"] = {"prompt_tokens": None, "completion_tokens": None, "partial": True}
    data = archive.freeze_repository_world_residual_selection(**(args | dict(invocation_receipt=extended,
        expected_invocation_sha256=archive.invocation_receipt_sha256(extended))))
    result = replay(data)
    assert result.recorded_invocation_receipt() == extended
    assert result.receipt()["provider_usage_verified"] is result.receipt()["billing_verified"] is False
