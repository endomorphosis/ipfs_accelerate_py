"""Both alternatives retain identical failure evidence and honest count scope."""
from dataclasses import replace
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_residual_selection as selection
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import PreparedRepositoryWorldPacket
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import (
    freeze_repository_world_packet, replay_repository_world_packet,
)
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter
from test.api.test_repository_context_compiler import native_session, required, scheduler  # noqa: F401
from test.api.test_repository_world_packet import TASK
from test.api.test_repository_world_residual_focus import baseline_case, request  # noqa: F401


FULL = "Full complete rendered input with workspace/reply guidance. " * 4
FOCUSED = "Focused complete rendered input with workspace/reply guidance."


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def pair_receipt():
    """Authored observation for pure selection; it grants no live issuance."""
    return dict(schema="repository-world-residual-alternatives@1",
        baseline_prompt_sha256="1" * 64, baseline_prompt_bytes=100,
        full_prompt_sha256="2" * 64, full_prompt_bytes=200,
        focused_prompt_sha256="3" * 64, focused_prompt_bytes=150,
        baseline_context_cid="cid:baseline", full_context_cid="cid:baseline", focused_context_cid="cid:focused",
        residual_packet_id="cid:residual", residual_context_id="cid:residual-context",
        residual_context_sha256="4" * 64, task_text_sha256="5" * 64,
        required_population_sha256="6" * 64, full_optional_population_sha256="7" * 64,
        focused_optional_population_sha256="8" * 64, required_reference_count=2,
        full_optional_count=8, focused_optional_count=1, full_baseline_optional_preserved=True,
        full_task_invariants_preserved=True, identical_residual_context=True,
        source_binding=dict(generation=1, source_tree_sha256="9" * 64,
            semantic_state_cid="cid:state", semantic_root_cid="cid:root", capsule_index_cid="cid:index"),
        provider_retained_parent=False, optional_omission_equivalence_verified=False,
        token_savings_established=False, default_adoption_authorized=False, proof_authority=False,
        execution_authority=False, completion_authority=False, provider_calls=0, provider_input_tokens=None)


def counter(callback):
    return PromptTokenCounter(provider="codex_cli", model="authored-model", tokenizer_id="local:proxy",
        tokenizer_revision="sha256:local-assets", special_token_policy="encode_without_special_tokens",
        count_tokens=callback)


def choose(token_counter=None, **changes):
    options = dict(alternatives_receipt=pair_receipt(), full_complete_prompt=FULL,
        focused_complete_prompt=FOCUSED, provider="codex_cli", model="authored-model", token_counter=token_counter)
    options.update(changes)
    return selection.select_repository_world_residual_prompt(**options)


@pytest.fixture
def issued_pair(baseline_case):
    focused = request(baseline_case)
    pair = selection.prepare_repository_world_residual_alternatives(baseline=baseline_case["baseline"], focused=focused)
    return baseline_case, focused, pair


def test_full_fallback_retains_original_optional_population_and_same_residual(issued_pair):
    case, focused, pair = issued_pair
    full, candidate = pair.full_packet, pair.focused_packet
    assert type(full) is type(candidate) is PreparedRepositoryWorldPacket
    assert candidate is focused.packet
    assert full is not case["baseline"]
    assert full._capsule is case["baseline"]._capsule
    assert full._adapter is candidate._adapter is case["adapter"]
    assert full.receipt()["residual_context"] == candidate.receipt()["residual_context"]
    assert full.prompt.count('"residual_context":') == candidate.prompt.count('"residual_context":') == 1
    assert '"residual_context":' not in case["baseline"].prompt
    observed = pair.receipt()
    assert observed["full_optional_count"] == 2 and observed["focused_optional_count"] == 1
    assert observed["full_baseline_optional_preserved"] is True
    assert observed["identical_residual_context"] is True
    assert selection.validate_repository_world_residual_alternatives(pair, workspace=case["source"], task_text=TASK) == observed
    for packet in (full, candidate):
        data = freeze_repository_world_packet(packet)
        assert replay_repository_world_packet(data, expected_sha256=sha(data)).prompt == packet.prompt


def test_fallback_does_not_recompile_population_or_consume_another_parent_slot(baseline_case, monkeypatch):
    focused = request(baseline_case)
    adapter = baseline_case["adapter"]
    parents_before = tuple(adapter._parents)
    monkeypatch.setattr(adapter, "compile", lambda **_args: pytest.fail("original query recompiled"))
    pair = selection.prepare_repository_world_residual_alternatives(baseline=baseline_case["baseline"], focused=focused)
    assert tuple(adapter._parents) == parents_before
    assert pair.full_packet._capsule is baseline_case["baseline"]._capsule


def test_pair_receipt_detaches_copies_and_slots_prevent_shadowing(issued_pair):
    _case, _focused, pair = issued_pair
    first = pair.receipt()
    first["source_binding"]["generation"] = 99
    assert pair.receipt()["source_binding"]["generation"] == 1
    for name, value in (("full_packet", pair.focused_packet), ("receipt", lambda: {})):
        with pytest.raises(AttributeError):
            object.__setattr__(pair, name, value)
    assert TASK.strip() not in json.dumps(pair.receipt())


def test_directly_created_pair_has_no_issuance(issued_pair):
    case, _focused, _pair = issued_pair
    forged = object.__new__(selection.PreparedRepositoryWorldResidualAlternatives)
    with pytest.raises(ValueError, match="original_required"):
        selection.validate_repository_world_residual_alternatives(forged, workspace=case["source"], task_text=TASK)


@pytest.mark.parametrize("name", ["_tokenizer", "_chars_per_token"])
def test_nested_compiler_estimator_replacement_refuses_before_native_fences(issued_pair, monkeypatch, name):
    case, _focused, pair = issued_pair
    monkeypatch.setattr(case["adapter"].compiler.estimator, name, object())
    monkeypatch.setattr(selection, "_current", lambda *_args, **_kwargs: pytest.fail("changed estimator reached native fence"))
    with pytest.raises(ValueError, match="estimator_changed"):
        selection.validate_repository_world_residual_alternatives(pair, workspace=case["source"], task_text=TASK)


def test_drift_during_compiler_verification_is_checked_before_native_closure(issued_pair, monkeypatch):
    case, _focused, pair = issued_pair
    compiler = case["adapter"].compiler
    original = compiler.verify_compile_result

    def verify_then_change(result):
        original(result)
        compiler.estimator._chars_per_token = object()

    monkeypatch.setattr(compiler, "verify_compile_result", verify_then_change)
    monkeypatch.setattr(selection, "_current", lambda *_args, **_kwargs: pytest.fail("verification drift reached native closure"))
    with pytest.raises(ValueError, match="estimator_changed"):
        selection.validate_repository_world_residual_alternatives(pair, workspace=case["source"], task_text=TASK)


@pytest.mark.parametrize("change", ["source", "generation", "packet", "wrapper"])
def test_original_custody_and_current_source_remain_required(issued_pair, change):
    case, focused, pair = issued_pair
    if change == "source":
        (case["source"] / "calc.py").write_text("def add(value):\n    return value + 99\n")
    elif change == "generation":
        case["session"].refresh(generation=1)
    elif change == "packet":
        object.__setattr__(pair.full_packet, "prompt", pair.full_packet.prompt + "changed")
    else:
        object.__setattr__(focused, "_receipt_json", "{}")
    with pytest.raises(ValueError):
        selection.validate_repository_world_residual_alternatives(pair, workspace=case["source"], task_text=TASK)


def test_original_failure_free_packet_is_not_a_focused_input(baseline_case):
    focused = request(baseline_case)
    forged = replace(focused, packet=baseline_case["baseline"])
    with pytest.raises(ValueError, match="lineage_required"):
        selection.prepare_repository_world_residual_alternatives(baseline=baseline_case["baseline"], focused=forged)


def test_wrong_baseline_provenance_on_focus_wrapper_is_refused(baseline_case):
    focused = request(baseline_case)
    observed = focused.receipt()
    observed["baseline_prompt_sha256"] = "0" * 64
    forged = replace(focused, _receipt_json=wire(observed).decode())
    with pytest.raises(ValueError, match="focus_baseline_changed"):
        selection.prepare_repository_world_residual_alternatives(baseline=baseline_case["baseline"], focused=forged)


def test_full_fallback_overflow_refuses_instead_of_returning_old_packet(baseline_case, monkeypatch):
    focused = request(baseline_case)

    def refuse(**kwargs):
        assert kwargs["compilation"].capsule is baseline_case["baseline"]._capsule
        assert kwargs["residual_packet"].packet_id == focused.packet.receipt()["residual_context"]["packet_id"]
        raise ValueError("world_packet_wire_overflow")

    monkeypatch.setattr(selection, "prepare_repository_world_packet", refuse)
    with pytest.raises(ValueError, match="wire_overflow"):
        selection.prepare_repository_world_residual_alternatives(baseline=baseline_case["baseline"], focused=focused)


def test_complete_strings_count_once_each_and_schema_once_with_no_measurement_recount():
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    schema_text = wire(schema).decode()
    calls = []

    def count(text):
        calls.append(text)
        return {FULL: 100, FOCUSED: 80, schema_text: 7}[text]

    label, receipt, measured = choose(counter(count), native_output_schema=schema)
    assert label == "focused" and calls == [FULL, FOCUSED, schema_text]
    assert receipt["selected_input_tokens"] == 87
    assert receipt["candidates"]["full"]["input_tokens"] == 107
    assert receipt["native_output_schema_tokens"] == 7
    assert receipt["alternatives_receipt_sha256"] == sha(wire(pair_receipt()))
    assert measured == selection.repository_world_residual_selected_measurement(receipt)
    assert calls == [FULL, FOCUSED, schema_text]
    assert measured["schema"] == "supervisor-world-task-prompt-measurement@1"
    assert measured["model_prompt_sha256"] == sha(FOCUSED.encode())
    assert measured["model_prompt_bytes"] == len(FOCUSED.encode())
    assert measured["selected_input_tokens"] == 87
    assert measured["provider_internal_history_measured"] is False
    assert measured["provider_tokenizer_equivalence_verified"] is False
    assert receipt["semantic_equivalence_verified"] is False
    assert receipt["total_session_token_savings"] is None


@pytest.mark.parametrize("focused_tokens", [100, 101])
def test_fewer_bytes_without_fewer_local_tokens_keeps_full(focused_tokens):
    label, receipt, _measurement = choose(counter(lambda text: 100 if text == FULL else focused_tokens))
    assert label == "full" and receipt["selection_reason"] == "focused_not_fewer_local_tokens"


def test_fewer_tokens_without_fewer_bytes_keeps_full():
    bigger = FULL + "focused overhead"
    label, receipt, _measurement = choose(counter(lambda text: 100 if text == FULL else 50), focused_complete_prompt=bigger)
    assert label == "full" and receipt["selection_reason"] == "focused_not_smaller_in_bytes"


def test_equal_bytes_keep_full_even_when_count_is_lower():
    other = "x" * len(FULL)
    label, _receipt, _measurement = choose(counter(lambda text: 100 if text == FULL else 80), focused_complete_prompt=other)
    assert label == "full"


def test_absent_counter_retains_full_unknown_counts_and_hard_cap_refuses():
    label, receipt, measured = choose()
    assert label == "full" and receipt["measurement_status"] == "unavailable"
    assert receipt["selected_input_tokens"] is None
    assert receipt["native_output_schema_tokens"] is None
    assert measured["selected_input_tokens"] is None
    with pytest.raises(ValueError, match="cap_requires_counter"):
        choose(max_input_tokens=1000)


@pytest.mark.parametrize("fail_at", [1, 2, 3])
def test_any_count_failure_discards_all_partial_counts_without_byte_rescue(fail_at):
    calls = []

    def count(text):
        calls.append(text)
        if len(calls) == fail_at:
            raise RuntimeError("private-tokenizer-exception-body")
        return 10

    label, receipt, measured = choose(counter(count), native_output_schema={"type": "object"})
    assert label == "full" and receipt["measurement_status"] == "failed"
    assert receipt["native_output_schema_tokens"] is None
    assert all(value["content_tokens"] is value["input_tokens"] is None for value in receipt["candidates"].values())
    assert measured["selected_input_tokens"] is None
    assert "private-tokenizer-exception-body" not in json.dumps(receipt)


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 1.5, (1 << 63), None])
def test_counts_require_bounded_positive_integers(invalid):
    label, receipt, _measured = choose(counter(lambda _text: invalid))
    assert label == "full" and receipt["measurement_status"] == "failed"
    assert receipt["selected_input_tokens"] is None


def test_content_plus_schema_overflow_is_not_a_partial_measurement():
    label, receipt, _measured = choose(counter(lambda _text: (1 << 63) - 1), native_output_schema={"type": "object"})
    assert label == "full" and receipt["measurement_status"] == "failed"
    assert receipt["native_output_schema_tokens"] is None


@pytest.mark.parametrize("field", ["provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy", "count_tokens"])
@pytest.mark.parametrize("also_raises", [False, True])
def test_counter_identity_or_callback_drift_is_refusal_even_when_count_raises(field, also_raises):
    local = None

    def count(_text):
        object.__setattr__(local, field, (lambda _text: 1) if field == "count_tokens" else "changed")
        if also_raises:
            raise RuntimeError("do not turn identity mutation into fallback")
        return 10

    local = counter(count)
    with pytest.raises(ValueError, match="counter_changed"):
        choose(local)


def test_unbound_counter_operations_ignore_instance_method_shadowing():
    local = counter(lambda text: 100 if text == FULL else 80)
    object.__setattr__(local, "count", lambda _text: 0)
    object.__setattr__(local, "binding", lambda: {"model": "forged"})
    label, receipt, _measured = choose(local)
    assert label == "focused" and receipt["tokenizer_binding"]["model"] == "authored-model"


def test_mismatching_route_refuses_before_count():
    local = counter(lambda _text: pytest.fail("foreign counter was called"))
    with pytest.raises(ValueError, match="counter_binding_invalid"):
        choose(local, model="different-model")


@pytest.mark.parametrize("cap", [True, 0, -1, 1 << 63])
def test_malformed_hard_cap_refuses(cap):
    with pytest.raises(ValueError, match="cap_invalid"):
        choose(max_input_tokens=cap)


def test_budget_applies_to_selected_complete_input_including_schema():
    local = counter(lambda text: 100 if text == FULL else 80 if text == FOCUSED else 7)
    assert choose(local, native_output_schema={}, max_input_tokens=87)[0] == "focused"
    with pytest.raises(ValueError, match="cap_exceeded"):
        choose(local, native_output_schema={}, max_input_tokens=86)
    broken = counter(lambda _text: (_ for _ in ()).throw(ValueError("failure")))
    with pytest.raises(ValueError, match="cap_requires_counter"):
        choose(broken, max_input_tokens=1000)


def test_oversized_selected_full_input_refuses():
    with pytest.raises(ValueError, match="selected_prompt_bound"):
        choose(full_complete_prompt="x" * 256001)


@pytest.mark.parametrize("change", ["extra", "hash", "boolcount", "partial", "choice", "cap", "scope", "schema", "error"])
def test_recorded_selection_validation_checks_shape_arithmetic_and_claims(change):
    _label, receipt, _measurement = choose(counter(lambda text: 100 if text == FULL else 80))
    if change == "extra":
        receipt["extra"] = "unreviewed"
    elif change == "hash":
        receipt["alternatives_receipt_sha256"] = "not-a-hash"
    elif change == "boolcount":
        receipt["candidates"]["full"]["content_tokens"] = True
    elif change == "partial":
        receipt["candidates"]["full"]["input_tokens"] = None
    elif change == "choice":
        receipt["selected_representation"] = "full"
    elif change == "cap":
        receipt["max_input_tokens"] = 1
    elif change == "scope":
        receipt["semantic_equivalence_verified"] = True
    elif change == "schema":
        receipt["native_output_schema_tokens"] = 1
    else:
        receipt["measurement_error_type"] = []
    with pytest.raises(ValueError):
        selection.validate_repository_world_residual_selection_receipt(receipt)


def test_pure_receipt_validation_cannot_create_a_live_pair():
    recorded = pair_receipt()
    copy = selection.validate_repository_world_residual_alternatives_receipt(recorded)
    copy["source_binding"]["generation"] = 99
    assert recorded["source_binding"]["generation"] == 1
    with pytest.raises(ValueError, match="exact_type_required"):
        selection.validate_repository_world_residual_alternatives(recorded, workspace=None, task_text=TASK)
