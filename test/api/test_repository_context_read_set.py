"""Real native sessions, whole originals and adversarial local budget controls."""
from hashlib import sha256
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_context_read_set as subject
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSessionError
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES, PromptTokenCounter, render_model_prompt,
)
from test.api.test_repository_context_compiler import native_session, scheduler  # noqa: F401

OBLIGATIONS = ["  TASK: preserve arithmetic callers exactly.\n",
    "SECURITY: reject unsafe inputs; no credentials or network.\n",
    "UI_UX: retain the visible café label and its whitespace.\n"]


def counter(callback=None, **changes):
    arguments = dict(provider="grok_cli", model="grok-4.7", tokenizer_id="authored:utf8",
        tokenizer_revision="fixture-v1", special_token_policy="literal-only",
        count_tokens=callback if callback is not None else lambda text: max(1, len(text.encode())))
    arguments.update(changes)
    return PromptTokenCounter(**arguments)


def compile_set(native_session, **changes):
    source, session = native_session
    options = dict(session=session, generation=1, query="add subtract", required_paths=["notes.txt"],
        required_texts=OBLIGATIONS, task_id="authored-read-set", provider="grok_cli", model="grok-4.7",
        workspace=source, token_counter=counter(), max_input_tokens=100000)
    options.update(changes)
    return subject.compile_repository_read_set(**options)


def body(result, workspace):
    _, advisory = render_model_prompt(prompt="", purpose="coding", workspace=workspace)
    assert result.selected_prompt.startswith(advisory + subject.GUIDANCE)
    return json.loads(result.selected_prompt[len(advisory + subject.GUIDANCE):])


def test_real_native_nominations_deliver_whole_original_files_and_required_ir_text(native_session):
    source, session = native_session
    original = {path: (source / path).read_bytes() for path in ("calc.py", "notes.txt")}
    native = session.context_query(generation=1, query="add subtract")
    assert len(native["candidates"]) == 2
    result = compile_set(native_session)
    payload, receipt = body(result, source), result.receipt
    assert payload["task_id"] == "authored-read-set" and payload["required_texts"] == OBLIGATIONS
    assert receipt["selected_paths"] == ["notes.txt", "calc.py"]
    assert receipt["nominated_paths"] == ["calc.py"]  # Two native symbols, one whole file.
    assert len(receipt["files"]) == 2
    for row in payload["sources"]:
        assert row["content"].encode() == original[row["path"]]
        assert row["source_sha256"] == sha256(original[row["path"]]).hexdigest()
    assert "# Authored arithmetic fixture; never imported." in payload["sources"][1]["content"]
    status = session.status_summary()
    for key, value in receipt["session_binding"].items():
        assert status[key] == value
    for key in ("generation", "source_tree_sha256", "semantic_state_cid", "semantic_root_cid", "capsule_index_cid"):
        assert receipt["native_query_binding"][key] == native[key]
    assert receipt["selected_input_tokens"] == len(result.selected_prompt.encode())
    assert all((source / path).read_bytes() == raw for path, raw in original.items())


def test_required_path_is_not_duplicated_when_native_ranking_nominates_it(native_session):
    result = compile_set(native_session, required_paths=("calc.py",))
    assert result.receipt["selected_paths"] == ["calc.py"]
    assert len(result.receipt["files"]) == 1 and result.receipt["files"][0]["required"] is True
    assert len(body(result, native_session[0])["sources"]) == 1


def test_no_native_hit_still_returns_every_required_obligation_and_source(native_session):
    result = compile_set(native_session, query="zz_unmatched_authored_question")
    assert result.receipt["native_query_counts"]["matching_count"] == 0
    assert result.receipt["nominated_paths"] == []
    assert result.receipt["selected_paths"] == ["notes.txt"]
    assert body(result, native_session[0])["required_texts"] == OBLIGATIONS


@pytest.mark.parametrize("kind", ["token", "byte"])
def test_optional_whole_file_omitted_without_truncating_required_core(native_session, kind):
    seen = []
    def measure(prompt):
        seen.append(prompt)
        return 101 if '"path":"calc.py"' in prompt else 100
    changes = dict(token_counter=counter(measure), max_input_tokens=100)
    if kind == "byte":
        baseline = compile_set(native_session, query="no_matching_symbol")
        changes.update(max_prompt_bytes=len(baseline.selected_prompt.encode()), max_input_tokens=1000)
    result = compile_set(native_session, **changes)
    assert result.receipt["selected_paths"] == ["notes.txt"]
    optional = result.receipt["files"][1]
    assert optional["selected"] is False
    assert optional["omission_reason"] == "complete_input_" + kind + "_budget"
    assert body(result, native_session[0])["required_texts"] == OBLIGATIONS
    assert "calc.py" not in [row["path"] for row in body(result, native_session[0])["sources"]]
    if kind == "byte":
        assert all('"path":"calc.py"' not in prompt for prompt in seen)
        assert result.receipt["measurements"][1]["input_count"] is None


@pytest.mark.parametrize("kind", ["token", "byte"])
def test_required_core_overflow_refuses_instead_of_omitting_or_truncating(native_session, kind):
    calls = []
    changes = dict(token_counter=counter(lambda text: calls.append(text) or 10), max_input_tokens=9)
    if kind == "byte":
        changes.update(max_prompt_bytes=1, max_input_tokens=100)
    with pytest.raises(ValueError, match="required_" + kind + "_budget_exceeded"):
        compile_set(native_session, **changes)
    if kind == "byte":
        assert calls == []
    assert native_session[1].generation == 1


def test_estimated_counter_retains_exact_route_provenance_and_no_exact_counts(native_session):
    result = compile_set(native_session, token_counter=counter(lambda text: max(1, len(text.encode()) // 4),
        count_kind="estimate", compatibility_basis="authored_utf8_proxy"))
    r = result.receipt
    assert r["measurement_status"] == "estimated" and r["selected_input_tokens"] is None
    assert r["selected_estimated_input_tokens"] == len(result.selected_prompt.encode()) // 4
    assert r["tokenizer_binding"]["provider"] == "grok_cli" and r["tokenizer_binding"]["model"] == "grok-4.7"
    assert r["tokenizer_compatibility_basis"] == "authored_utf8_proxy"
    assert r["token_budget_basis"] == "declared_local_estimate"
    assert r["token_budget_guaranteed"] is False


@pytest.mark.parametrize("stage", ["first", "optional", "final"])
def test_source_edit_during_any_counter_stage_refuses_without_refresh(native_session, stage):
    source, session = native_session
    calls = []
    def measure(prompt):
        calls.append(prompt)
        target = 1 if stage == "first" else 3 if stage == "optional" else 5
        if len(calls) == target:
            (source / "notes.txt").write_text("Edited after native nomination.\n")
        return 10
    with pytest.raises((RepositorySourceSessionError, ValueError), match="stale_source|refresh_required|session_changed"):
        compile_set(native_session, token_counter=counter(measure))
    assert session.generation == 1 and session.status_summary()["refresh_required"] is True


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10", pytest.param(1 << 20000, id="huge_integer")])
def test_bad_counter_refuses_without_a_partial_packet_or_native_charge(native_session, bad):
    with pytest.raises(ValueError, match="counter_failed|counter_magnitude_exceeded"):
        compile_set(native_session, token_counter=counter(lambda _: bad))
    assert native_session[1].status_summary()["source_current"] is True


def test_counter_failure_never_exports_private_exception_text(native_session):
    def failed(_):
        raise RuntimeError("PRIVATE_MODEL_OR_SOURCE_BODY")
    with pytest.raises(ValueError, match="read_set_counter_failed") as raised:
        compile_set(native_session, token_counter=counter(failed))
    assert "PRIVATE_MODEL_OR_SOURCE_BODY" not in str(raised.value)
    assert raised.value.__suppress_context__ is True


@pytest.mark.parametrize("change", ["count", "binding", "kind", "basis"])
def test_counter_repeatability_and_provenance_drift_refuse(native_session, change):
    calls = []
    supplied = None
    def measure(prompt):
        calls.append(prompt)
        if len(calls) == 1 and change != "count":
            name, value = {"binding": ("tokenizer_revision", "different"),
                "kind": ("count_kind", "estimate"), "basis": ("compatibility_basis", "different")}[change]
            object.__setattr__(supplied, name, value)
        return len(calls) if change == "count" else 10
    supplied = counter(measure)
    with pytest.raises(ValueError, match="counter_not_repeatable|counter_changed"):
        compile_set(native_session, token_counter=supplied)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_cancellation_preserves_original_exception_identity(native_session, error):
    def cancelled(_):
        raise error
    with pytest.raises(type(error)) as raised:
        compile_set(native_session, token_counter=counter(cancelled))
    assert raised.value is error


def test_required_non_utf8_source_refuses_and_optional_non_utf8_is_explicit(native_session, monkeypatch):
    source, old = native_session
    # A real successor indexes the unchanged Python source and retains opaque bytes.
    (source / "opaque.bin").write_bytes(b"\xff\x00")
    old.refresh(generation=1)
    with pytest.raises(ValueError, match="required_source_not_utf8"):
        compile_set(native_session, generation=2, required_paths=["opaque.bin"])
    original_query = old.context_query
    def nominated(**kwargs):
        value = original_query(**kwargs)
        # Authored nomination extension tests presentation policy, not native ranking.
        value["candidates"] = [dict(path="opaque.bin", source_sha256=sha256(b"\xff\x00").hexdigest())]
        return value
    monkeypatch.setattr(old, "context_query", nominated)
    result = compile_set(native_session, generation=2)
    assert result.receipt["files"][1]["omission_reason"] == "non_utf8_source"
    assert result.receipt["selected_paths"] == ["notes.txt"]


@pytest.mark.parametrize("change", ["source_hash", "closing_query", "session_head"])
def test_native_nomination_hash_and_closing_identity_are_independently_joined(native_session, monkeypatch, change):
    _, session = native_session
    original_query, original_status, calls = session.context_query, session.status_summary, []
    def query(**kwargs):
        value = original_query(**kwargs)
        calls.append(value)
        if change == "source_hash": value["candidates"][0]["source_sha256"] = "0" * 64
        elif change == "closing_query" and len(calls) == 2: value["ranking"] = "different"
        return value
    monkeypatch.setattr(session, "context_query", query)
    if change == "session_head":
        status_calls = []
        def status():
            value = original_status()
            status_calls.append(value)
            if len(status_calls) > 1: value["datasets_head"] = {"generation": "foreign"}
            return value
        monkeypatch.setattr(session, "status_summary", status)
    with pytest.raises(ValueError, match="nomination_hash_conflict|source_binding_differs|session_changed"):
        compile_set(native_session)


def test_packet_and_owner_receipt_remain_separate_without_provider_or_source_execution(native_session, monkeypatch):
    from ipfs_accelerate_py import llm_router
    monkeypatch.setattr(llm_router, "generate_text", lambda **_kw: pytest.fail("no provider may be invoked"))
    result = compile_set(native_session)
    r = result.receipt
    assert r["provider_invoked"] is r["provider_delivery_verified"] is r["task_acceptance_verified"] is False
    assert r["source_coverage_complete"] is r["native_query_relevance_proved"] is False
    assert r["provider_token_usage"] is r["total_session_token_savings"] is None
    assert r["proof_authority"] is r["execution_authority"] is r["completion_authority"] is False
    assert r["session_refreshed"] is False and r["full_original_sources_only"] is True
    assert result.selected_prompt not in repr(result)
    assert OBLIGATIONS[0].strip() not in result.receipt_json
    assert "def add(value)" not in result.receipt_json and "add subtract" not in result.receipt_json
    fresh = result.receipt
    fresh["selected_paths"].append("forged.py")
    assert "forged.py" not in result.receipt["selected_paths"]


@pytest.mark.parametrize("changes", [dict(max_input_tokens=True), dict(max_prompt_bytes=0),
    dict(generation=0), dict(limit=33), dict(required_paths=["../outside.py"]),
    dict(required_paths=["bad\npath.py"]),
    dict(required_paths=["notes.txt", "notes.txt"]), dict(required_texts=[]),
    dict(required_texts=["x" * 16385]), dict(query=""), dict(token_counter=None),
    dict(token_counter=counter(model="foreign"))])
def test_invalid_inputs_refuse_before_native_query_or_counter_work(native_session, monkeypatch, changes):
    monkeypatch.setattr(native_session[1], "context_query", lambda **_kw: pytest.fail("invalid input reached native query"))
    with pytest.raises(ValueError):
        compile_set(native_session, **changes)


@pytest.mark.parametrize("field,value", [("tokenizer_id", "bad\nmetadata"),
    ("compatibility_basis", " "), ("count_kind", []), ("tokenizer_revision", "x" * 4097)])
def test_invalid_counter_metadata_refuses_before_native_work(native_session, monkeypatch, field, value):
    supplied = counter()
    object.__setattr__(supplied, field, value)
    monkeypatch.setattr(native_session[1], "context_query", lambda **_kw: pytest.fail("bad metadata reached native query"))
    with pytest.raises(ValueError):
        compile_set(native_session, token_counter=supplied)
