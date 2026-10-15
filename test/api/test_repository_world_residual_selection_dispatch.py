"""Complete residual alternatives reach the existing journaled provider seam.

Providers and most counters are authored. Native source/intent fences, prompt
rendering and the tiny local tokenizer fixture remain real. These tests make no
live model quality, remote tokenizer, billing or session saving claim.
"""
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_residual_selection as selection
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_residual as focus
from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_attempt_runner as attempts
from ipfs_accelerate_py.agent_supervisor.runtime import coding_reply_contract as reply
from test.api.test_repository_world_dispatch import (
    TASK, dispatch_case, world_session, required, scheduler, counter,
)
from test.api.test_repository_world_attempt_runner import make_journal
from test.api.test_repository_world_residual_focus import residual, request, native_intent_baseline
from test.api.test_repository_intent_world import intent_case
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_intent_requirement_observation import _start


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def alternative_case(dispatch_case):
    baseline = dispatch_case[2]
    focused = focus.prepare_repository_world_residual_request(baseline=baseline,
        residual_packet=residual(baseline), query="add",
        unresolved_question="Which definition implements the remaining arithmetic obligation?", limit=1)
    alternatives = selection.prepare_repository_world_residual_alternatives(baseline=baseline, focused=focused)
    return dispatch_case, alternatives


def render(case, packet, mode="legacy"):
    text, advice = runner.render_model_prompt(prompt=packet.prompt, purpose="coding",
        workspace=case[0], world_packet=True)
    text, contract = reply.apply_coding_reply_contract(model_prompt=text, mode=mode,
        purpose="coding", provider="codex_cli")
    return text, advice, contract


def counted(alternative_case, *, full=100, focused=20, mode="legacy", failure=None):
    case, alternatives = alternative_case
    full_text = render(case, alternatives.full_packet, mode)[0]
    focused_text = render(case, alternatives.focused_packet, mode)[0]
    calls = []
    def count(text):
        calls.append(text)
        if failure is not None:
            raise failure
        if text == full_text:
            return full
        if text == focused_text:
            return focused
        return 7  # Authored count for the separately rendered native reply schema.
    return counter(count_tokens=count), calls, full_text, focused_text


def options(alternative_case, **changes):
    _case, alternatives = alternative_case
    return dict(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
        max_output_tokens=128, world_task_packet=alternatives.full_packet,
        world_task_residual_alternatives=alternatives) | changes


def invoke(alternative_case, tmp_path, **changes):
    case, _alternatives = alternative_case
    journal, root = make_journal(tmp_path, workspace=case[0])
    output, receipt, diagnostic = attempts.run_repository_world_packet_attempt(
        journal=journal, runner_options=options(alternative_case, **changes))
    return output, receipt, diagnostic, root


def retained(tmp_path):
    return json.loads((tmp_path / "journal" / "receipt.json").read_bytes())


def assert_actual_input(case, alternatives, receipt, expected_label, expected_text):
    packet = alternatives.full_packet if expected_label == "full" else alternatives.focused_packet
    actual, provider_options = case[3]["seen"][0]
    assert actual == expected_text
    assert packet.prompt in actual
    assert actual.count('"residual_context":') == 1
    assert receipt["router_prompt_sha256"] == sha(packet.prompt)
    assert receipt["model_prompt_sha256"] == sha(actual)
    assert receipt["world_task_packet"]["prompt_sha256"] == sha(packet.prompt)
    assert receipt["world_task_packet"]["residual_context"] == packet.receipt()["residual_context"]
    assert receipt["world_task_packet"]["provider_retained_parent"] is False
    assert receipt["world_task_packet"]["dispatch_rechecked"] is True
    row = receipt["world_task_residual_selection"]
    assert row["selected_representation"] == expected_label
    assert row["dispatch_rechecked"] is True
    assert row["candidates"][expected_label]["model_prompt_sha256"] == sha(actual)
    assert provider_options["allocation_session_id"] == receipt["invocation_id"]
    assert "resume_session_id" not in provider_options
    assert "prompt_selection" not in receipt


@pytest.mark.parametrize("full,focused,label", [(100, 20, "focused"), (100, 100, "full"), (20, 100, "full")])
def test_complete_prompt_selection_counts_once_and_dispatches_exact_issued_packet(
        alternative_case, tmp_path, full, focused, label):
    case, alternatives = alternative_case
    token_counter, calls, full_text, focused_text = counted(alternative_case, full=full, focused=focused)
    output, receipt, diagnostic, root = invoke(alternative_case, tmp_path, prompt_token_counter=token_counter)
    assert output and diagnostic["journal"]["closed"] is True
    assert calls == [full_text, focused_text]
    expected = full_text if label == "full" else focused_text
    assert_actual_input(case, alternatives, receipt, label, expected)
    assert json.loads((root / "receipt.json").read_bytes()) == receipt
    row = receipt["world_task_residual_selection"]
    assert row["measurement_status"] == "measured"
    assert row["candidates"]["full"]["content_tokens"] == full
    assert row["candidates"]["focused"]["content_tokens"] == focused
    measured = receipt["world_task_packet"]["prompt_measurement"]
    assert measured["selected_content_tokens"] == (full if label == "full" else focused)
    assert measured["model_prompt_sha256"] == sha(expected)
    assert row["total_session_token_savings"] is None
    assert row["omission_authority"] is row["semantic_equivalence_verified"] is False


@pytest.mark.parametrize("mode", ["missing", "failed", "zero", "boolean"])
def test_unknown_counts_fall_back_to_full_packet_with_identical_residual(alternative_case, tmp_path, mode):
    case, alternatives = alternative_case
    token_counter, _calls, full_text, _focused_text = counted(alternative_case,
        full=0 if mode == "zero" else False if mode == "boolean" else 100,
        failure=ValueError("authored encoder failure") if mode == "failed" else None)
    _output, receipt, _diagnostic, _root = invoke(alternative_case, tmp_path,
        prompt_token_counter=None if mode == "missing" else token_counter)
    assert_actual_input(case, alternatives, receipt, "full", full_text)
    row = receipt["world_task_residual_selection"]
    assert row["measurement_status"] == ("unavailable" if mode == "missing" else "failed")
    for candidate in row["candidates"].values():
        assert candidate["content_tokens"] is candidate["input_tokens"] is None
    assert receipt["world_task_packet"]["prompt_measurement"]["selected_input_tokens"] is None


@pytest.mark.parametrize("missing", [False, True])
def test_hard_cap_refuses_unknown_counts_without_dispatch(alternative_case, tmp_path, missing):
    token_counter, _calls, _full, _focused = counted(alternative_case, failure=ValueError("authored count failure"))
    with pytest.raises(ValueError):
        invoke(alternative_case, tmp_path, prompt_token_counter=None if missing else token_counter, max_input_tokens=100)
    assert alternative_case[0][3]["seen"] == []
    assert json.loads((tmp_path / "journal" / "terminal.json").read_bytes())["outcome"] == "raised"
    assert not (tmp_path / "journal" / "receipt.json").exists()


@pytest.mark.parametrize("label", ["full", "focused"])
def test_both_complete_inputs_include_same_reply_contract_and_count_schema_once(
        alternative_case, tmp_path, label):
    case, alternatives = alternative_case
    full, focused = (20, 100) if label == "full" else (100, 20)
    token_counter, calls, full_text, focused_text = counted(alternative_case,
        full=full, focused=focused, mode="ordinary-completion@1")
    expected = full_text if label == "full" else focused_text
    # Preserve the native schema validator: the existing authored provider's
    # missing schema observation fails after dispatch, allowing receipt checks.
    with pytest.raises(RuntimeError, match="schema invocation metadata"):
        invoke(alternative_case, tmp_path, prompt_token_counter=token_counter,
            coding_reply_mode="ordinary-completion@1", max_input_tokens=27)
    receipt = retained(tmp_path)
    assert_actual_input(case, alternatives, receipt, label, expected)
    schema = json.dumps(reply.coding_completion_schema(), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False)
    assert calls == [full_text, focused_text, schema]
    measured = receipt["world_task_packet"]["prompt_measurement"]
    assert measured["native_output_schema_tokens"] == 7
    assert measured["selected_content_tokens"] == 20
    assert measured["selected_input_tokens"] == 27
    assert receipt["coding_reply_contract"]["model_prompt_after_sha256"] == sha(expected)
    assert receipt["coding_reply_contract"]["native_output_schema_sha256"] == sha(schema)
    assert receipt["status"] == "failed" and receipt["world_task_packet"]["provider_attempted"] is True


def test_counts_include_actual_common_native_source_and_tool_guidance(alternative_case, tmp_path):
    case, _alternatives = alternative_case
    calls = []
    def count(text):
        calls.append(text)
        return len(text.encode())
    _output, receipt, _diagnostic, _root = invoke(alternative_case, tmp_path,
        prompt_token_counter=counter(count_tokens=count), tool_output_policy="artifact-preview@2",
        repository_source_policy="repository-source-compact@1")
    assert len(calls) == 2
    assert all("Repository source tools are configured" in text and "Optional bounded output helper" in text
               and "Supervisor coding workspace contract" in text for text in calls)
    actual = case[3]["seen"][0][0]
    assert actual in calls
    for name, candidate in receipt["world_task_residual_selection"]["candidates"].items():
        assert candidate["model_prompt_sha256"] == sha(calls[0 if name == "full" else 1])
    assert receipt["repository_source_policy"]["requested_policy"] == "repository-source-compact@1"


@pytest.mark.parametrize("field", ["provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy", "count_tokens"])
@pytest.mark.parametrize("when", ["count", "provider_initialization"])
def test_counter_identity_and_callback_drift_refuse_before_provider(
        alternative_case, tmp_path, monkeypatch, field, when):
    from ipfs_accelerate_py import llm_router
    token_counter, _calls, _full, _focused = counted(alternative_case)
    def mutate():
        object.__setattr__(token_counter, field, (lambda _text: 1) if field == "count_tokens" else "changed")
    if when == "count":
        def change_during_count(text):
            mutate()
            return len(text.encode())
        object.__setattr__(token_counter, "count_tokens", change_during_count)
    else:
        def changed_provider(*_args, **_kwargs):
            mutate()
            return object()
        monkeypatch.setattr(llm_router, "get_llm_provider", changed_provider)
    with pytest.raises(ValueError, match="counter|tokenizer"):
        invoke(alternative_case, tmp_path, prompt_token_counter=token_counter)
    assert alternative_case[0][3]["seen"] == []
    if when == "provider_initialization":
        receipt = retained(tmp_path)
        assert receipt["world_task_residual_selection"]["dispatch_rechecked"] is False
        assert receipt["world_task_packet"]["provider_attempted"] is False


@pytest.mark.parametrize("field,value", [("provider", "grok_cli"), ("model", "other-model")])
def test_counter_route_mismatch_never_falls_back_or_reaches_discovery(
        alternative_case, tmp_path, monkeypatch, field, value):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda:
        pytest.fail("mismatched counter reached provider discovery"))
    with pytest.raises(ValueError, match="counter|provider/model"):
        invoke(alternative_case, tmp_path, prompt_token_counter=counter(**{field: value}))


@pytest.mark.parametrize("when", ["count", "provider_initialization"])
def test_source_drift_during_counter_or_initialization_refuses_dispatch(
        alternative_case, tmp_path, monkeypatch, when):
    from ipfs_accelerate_py import llm_router
    source = alternative_case[0][0]
    def changed():
        (source / "calc.py").write_text("def add(value):\n    return value + 99\n")
    def count(text):
        if when == "count":
            changed()
        return len(text.encode())
    if when == "provider_initialization":
        def initialize(*_args, **_kwargs):
            changed()
            return object()
        monkeypatch.setattr(llm_router, "get_llm_provider", initialize)
    with pytest.raises(ValueError):
        invoke(alternative_case, tmp_path, prompt_token_counter=counter(count_tokens=count))
    assert alternative_case[0][3]["seen"] == []
    if when == "provider_initialization":
        assert retained(tmp_path)["world_task_residual_selection"]["dispatch_rechecked"] is False


@pytest.mark.parametrize("failed_counts", [False, True])
@pytest.mark.parametrize("component", ["descriptor", "asset"])
def test_actual_local_tokenizer_assets_rechecked_after_provider_initialization_even_on_fallback(
        alternative_case, tmp_path, monkeypatch, failed_counts, component):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import OwnerPromptTokenizer
    from test.api.test_local_prompt_tokenizer import make_binding
    artifact, digest, descriptor = make_binding(tmp_path, provider="codex_cli", model="pinned")
    if failed_counts:
        def failed_count(_self, _text):
            raise ValueError("authored local encode failure")
        monkeypatch.setattr(OwnerPromptTokenizer, "count", failed_count)
    target = artifact if component == "descriptor" else artifact.parent / descriptor["asset_path"]
    def changed(*_args, **_kwargs):
        target.write_bytes(target.read_bytes() + b" ")
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", changed)
    with pytest.raises(ValueError):
        invoke(alternative_case, tmp_path, prompt_tokenizer_artifact=artifact, prompt_tokenizer_sha256=digest)
    assert alternative_case[0][3]["seen"] == []
    receipt = retained(tmp_path)
    assert receipt["world_task_residual_selection"]["dispatch_rechecked"] is False
    assert receipt["world_task_packet"]["provider_attempted"] is False
    if failed_counts:
        assert receipt["world_task_residual_selection"]["measurement_status"] == "failed"
        assert receipt["world_task_residual_selection"]["selected_representation"] == "full"


def test_failed_provider_keeps_actual_focused_packet_counts_and_full_journal(alternative_case, tmp_path):
    case, alternatives = alternative_case
    token_counter, _calls, _full, focused_text = counted(alternative_case)
    failure = TimeoutError("authored provider timeout")
    def failed(_prompt, _options):
        raise failure
    case[3]["callback"] = failed
    with pytest.raises(TimeoutError) as caught:
        invoke(alternative_case, tmp_path, prompt_token_counter=token_counter)
    assert caught.value is failure
    receipt = retained(tmp_path)
    assert_actual_input(case, alternatives, receipt, "focused", focused_text)
    assert receipt["status"] == "failed" and receipt["usage"]["prompt_tokens"] == 15
    assert json.loads((tmp_path / "journal" / "terminal.json").read_bytes())["outcome"] == "raised"


@pytest.mark.parametrize("change", ["missing_journal", "wrong_full", "mapping", "presentation", "artifact", "planning"])
def test_selection_requires_exact_full_packet_current_journal_and_supported_route(
        alternative_case, tmp_path, monkeypatch, change):
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_presentation import prepare_repository_world_packet_presentation
    case, alternatives = alternative_case
    journal, _root = make_journal(tmp_path, workspace=case[0])
    changes = {"world_task_attempt_journal": journal}
    if change == "missing_journal":
        changes["world_task_attempt_journal"] = None
    elif change == "wrong_full":
        changes["world_task_packet"] = alternatives.focused_packet
    elif change == "mapping":
        changes["world_task_residual_alternatives"] = alternatives.receipt()
    elif change == "presentation":
        changes["world_task_presentation"] = prepare_repository_world_packet_presentation(alternatives.full_packet.prompt)
    elif change == "artifact":
        changes["world_packet_artifact_dir"] = tmp_path
    else:
        changes["purpose"] = "planning"
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda:
        pytest.fail("unsupported alternative route reached provider discovery"))
    try:
        with pytest.raises(ValueError, match="alternatives"):
            runner.run(**options(alternative_case, **changes))
    finally:
        type(journal).close(journal)
    assert case[3]["seen"] == []


def test_none_preserves_existing_route_prompt_and_receipt_shape(dispatch_case):
    ordinary = dict(prompt=TASK, provider="codex_cli", model="pinned", timeout=30,
        max_output_tokens=128, world_task_packet=dispatch_case[2])
    _, before = runner.run(**ordinary)
    _, explicit = runner.run(**ordinary, world_task_residual_alternatives=None)
    assert dispatch_case[3]["seen"][0][0] == dispatch_case[3]["seen"][1][0]
    assert before.keys() == explicit.keys()
    assert "world_task_residual_selection" not in before
    assert "world_task_residual_selection" not in explicit


def test_native_intent_drift_after_selection_refuses_before_actual_provider(
        intent_case, dispatch_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    native = native_intent_baseline(intent_case)
    focused = request(native, query="answer")
    alternatives = selection.prepare_repository_world_residual_alternatives(baseline=native["baseline"], focused=focused)
    source = native["baseline"]._workspace
    monkeypatch.chdir(source)
    def changed(*_args, **_kwargs):
        _start(intent_case["case"], intent_case["task_id"])
        return object()
    monkeypatch.setattr(llm_router, "get_llm_provider", changed)
    journal, _root = make_journal(tmp_path, workspace=source)
    with pytest.raises(ValueError):
        attempts.run_repository_world_packet_attempt(journal=journal, runner_options=dict(
            prompt=native["task_text"], provider="codex_cli", model="pinned", timeout=30,
            max_output_tokens=128, world_task_packet=alternatives.full_packet,
            world_task_residual_alternatives=alternatives, prompt_token_counter=counter()))
    assert dispatch_case[3]["seen"] == []
    assert retained(tmp_path)["world_task_residual_selection"]["dispatch_rechecked"] is False
