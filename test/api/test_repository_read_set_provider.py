"""Real source sessions and actual shared-router calls with authored backends.

The authored backend observations test dispatch custody/accounting, not native
Grok effectiveness, containment, billing, model inference or task acceptance.
"""
from hashlib import sha256
import json
import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation, set_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
from ipfs_accelerate_py.agent_supervisor.runtime import repository_read_set_provider as subject
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter
from test.api.test_repository_context_compiler import native_session, scheduler  # noqa: F401

PRIVATE = "private-read-set-response-marker"
REQUIRED = ["TASK: inspect arithmetic without changing callers.\n",
    "SECURITY: preserve validation.\n", "UI_UX: retain café labels.\n"]


def counter(callback=None):
    return PromptTokenCounter(provider="grok_cli", model="grok-4.7", tokenizer_id="authored:utf8",
        tokenizer_revision="fixture-v1", special_token_policy="literal-only",
        count_tokens=callback if callback is not None else lambda value: max(1, len(value.encode())))


def observation(*, failed=False):
    payload = {"type": "error" if failed else "end", "stopReason": "end_turn",
        "usage_is_incomplete": False, "usage": {"input_tokens": 7, "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 1, "output_tokens": 3, "total_tokens": 13}}
    return dict(native_grok_observation(payload, exit_code=1 if failed else 0),
        session_id="authored-native-session", model_id="grok-4.7", exit_code=1 if failed else 0)


@pytest.fixture
def route(native_session, monkeypatch):
    from ipfs_accelerate_py.llm_allocation import api_key_slots
    source, _ = native_session
    monkeypatch.chdir(source)
    # Keep the genuine generate_text implementation while replacing only external
    # discovery, credential and allocation persistence in these authored units.
    monkeypatch.setattr(api_key_slots, "apply_api_key_to_kwargs", lambda _provider, values, **_kw: dict(values))
    monkeypatch.setattr(llm_router, "get_allocation_session", lambda _identity: {})
    monkeypatch.setattr(llm_router, "_record_llm_allocation_observation", lambda **_kw: None)
    monkeypatch.setattr(llm_router, "_capture_cli_session", lambda *_a, **_kw: None)
    monkeypatch.setattr(llm_router, "_bind_cli_session_kwargs", lambda _provider, values: dict(values))
    calls, discoveries = [], []
    class AuthoredBackend:
        action = None
        def generate(self, prompt, *, model_name=None, **kwargs):
            calls.append(dict(prompt=prompt, model=model_name, kwargs=kwargs))
            set_last_cli_observation("grok_cli", observation())
            if self.action is not None:
                return self.action(prompt, kwargs)
            return PRIVATE
    backend = AuthoredBackend()
    def discover(provider, **kwargs):
        discoveries.append((provider, kwargs))
        return backend
    monkeypatch.setattr(llm_router, "get_llm_provider", discover)
    return backend, calls, discoveries


def inspect(native_session, **changes):
    source, session = native_session
    options = dict(session=session, generation=1, query="add subtract", required_paths=["notes.txt"],
        required_texts=REQUIRED, task_id="authored-inspection", provider="grok_cli", model="grok-4.7",
        workspace=source, token_counter=counter(), max_input_tokens=100000, timeout=5)
    options.update(changes)
    return subject.inspect_repository_read_set(**options)


def test_exact_read_set_is_sent_once_without_second_wrapper_or_default_model_retry(native_session, route):
    _, calls, discoveries = route
    result = inspect(native_session)
    assert len(calls) == len(discoveries) == 1
    call = calls[0]
    assert call["prompt"] == result.selected_prompt
    assert call["prompt"].count("Supervisor coding workspace contract:") == 1
    assert call["model"] == "grok-4.7" and discoveries[0][0] == "grok_cli"
    assert discoveries[0][1]["use_cache"] is False
    assert call["kwargs"]["grok_max_turns"] == 1 and call["kwargs"]["grok_stream_progress"] is True
    assert call["kwargs"]["grok_tools"] == "read_file"
    assert call["kwargs"]["grok_disallowed_tools"] == "read_file,search_tool,use_tool"
    assert call["kwargs"]["grok_permission_mode"] == "dontAsk"
    assert call["kwargs"]["resume_session_id"] == "" and call["kwargs"]["continue_session"] is False
    assert result.response_text == PRIVATE
    r = result.receipt
    assert r["status"] == "returned" and r["backend_dispatches"] == r["provider_callback_attempts"] == 1
    assert r["predispatch_read_set_rechecked"] is r["closing_read_set_rechecked"] is True
    assert r["model_prompt_sha256"] == sha256(result.selected_prompt.encode()).hexdigest()
    assert r["native_rollout_usage"]["usage"]["total_tokens"] == 13
    assert r["response_sha256"] == sha256(PRIVATE.encode()).hexdigest()


def test_pinning_fallback_and_retry_controls_reach_actual_generate_text(native_session, route, monkeypatch):
    calls = []
    actual = llm_router.generate_text
    def observe(prompt, **kwargs):
        calls.append(kwargs)
        return actual(prompt, **kwargs)
    monkeypatch.setattr(llm_router, "generate_text", observe)
    result = inspect(native_session)
    assert len(calls) == 1
    options = calls[0]
    assert options["allow_local_fallback"] is options["allow_cross_provider_fallback"] is False
    assert options["disable_model_retry"] is options["side_effecting"] is True
    assert options["allocation_session_id"] == options["chat_session_id"] == result.receipt["invocation_id"]
    assert len(options["allocation_session_id"]) == 32


@pytest.mark.parametrize("mutation", ["prompt", "model", "twice", "skip"])
def test_guarded_backend_refuses_middleware_changes_or_multiple_calls(native_session, route, monkeypatch, mutation):
    _, calls, _ = route
    retained = []
    def middleware(prompt, **kwargs):
        backend = kwargs["provider_instance"]
        if mutation == "skip": return "fabricated middleware reply"
        if mutation == "prompt": return backend.generate(prompt + "CHANGED", model_name="grok-4.7")
        if mutation == "model": return backend.generate(prompt, model_name=None)
        backend.generate(prompt, model_name="grok-4.7")
        return backend.generate(prompt, model_name="grok-4.7")
    monkeypatch.setattr(llm_router, "generate_text", middleware)
    with pytest.raises(ValueError):
        inspect(native_session, receipt_observer=retained.append)
    assert len(calls) == (1 if mutation == "twice" else 0)
    assert retained[-1]["status"] == "failed"
    assert retained[-1]["backend_dispatches"] == len(calls)


@pytest.mark.parametrize("stage", ["factory", "backend"])
def test_source_changes_before_or_after_provider_are_refused_with_charges_retained(native_session, route, monkeypatch, stage):
    source, _ = native_session
    backend, calls, _ = route
    retained = []
    def changed(*_args, **_kw):
        (source / "notes.txt").write_text("Changed during the authored boundary.\n")
        return PRIVATE
    if stage == "factory":
        def discover(*_a, **_kw):
            changed()
            return backend
        monkeypatch.setattr(llm_router, "get_llm_provider", discover)
    else:
        backend.action = changed
    with pytest.raises(ValueError):
        inspect(native_session, receipt_observer=retained.append)
    assert len(calls) == (1 if stage == "backend" else 0)
    receipt = retained[-1]
    assert receipt["closing_read_set_rechecked"] is False
    assert receipt["backend_dispatches"] == len(calls)
    if stage == "backend":
        assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 13
        assert receipt["status"] == "refused"


def test_external_requirements_mutation_during_recompile_blocks_actual_delegate(native_session, route, tmp_path):
    _, calls, _ = route
    external = tmp_path / "requirements"
    external.write_text("Pinned external requirement")
    expected, evaluations = external.read_bytes(), []
    def guard():
        if external.read_bytes() != expected: raise ValueError("authored_requirements_changed")
    def count(prompt):
        evaluations.append(prompt)
        if len(evaluations) == 7: external.write_text("Changed by the authored counter")
        return 10
    retained = []
    with pytest.raises(ValueError, match="authored_requirements_changed"):
        inspect(native_session, token_counter=counter(count), requirements_guard=guard, receipt_observer=retained.append)
    assert calls == [] and retained[-1]["backend_dispatches"] == 0


@pytest.mark.parametrize("secondary", ["guard", "observer", "accounting"])
def test_original_provider_failure_identity_survives_secondary_failures(native_session, route, monkeypatch, secondary):
    from ipfs_accelerate_py.cli_runtime import grok_native_usage
    backend, calls, _ = route
    original = RuntimeError("PRIVATE_PROVIDER_FAILURE")
    retained = []
    def failed(*_a): raise original
    backend.action = failed
    def guard():
        if calls and secondary == "guard": raise SystemExit(9)
    def observe(receipt):
        retained.append(receipt)
        if secondary == "observer": raise KeyboardInterrupt()
    if secondary == "accounting":
        monkeypatch.setattr(grok_native_usage, "grok_usage_receipt", lambda _obs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(RuntimeError) as raised:
        inspect(native_session, requirements_guard=guard, receipt_observer=observe)
    assert raised.value is original and len(calls) == 1
    assert retained[-1]["backend_dispatches"] == 1
    assert "PRIVATE_PROVIDER_FAILURE" not in json.dumps(retained)
    if secondary != "accounting":
        assert retained[-1]["native_rollout_usage"]["usage"]["total_tokens"] == 13


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_fresh_accounting_cancellation_is_not_swallowed_after_success(native_session, route, monkeypatch, error):
    from ipfs_accelerate_py.cli_runtime import grok_native_usage
    monkeypatch.setattr(grok_native_usage, "grok_usage_receipt", lambda _obs: (_ for _ in ()).throw(error))
    retained = []
    with pytest.raises(type(error)) as raised:
        inspect(native_session, receipt_observer=retained.append)
    assert raised.value is error and retained[-1]["status"] == "refused"
    assert retained[-1]["usage_capture_status"] == "unavailable"


def test_predelegate_clear_prevents_stale_usage_on_early_backend_error(native_session, route, monkeypatch):
    stale = observation()
    stale["native_grok_total_tokens"] = 99999
    def factory(*_a, **_kw):
        set_last_cli_observation("grok_cli", stale)
        class EarlyFailure:
            def generate(self, *_a, **_kw): raise FileNotFoundError("authored early failure")
        return EarlyFailure()
    monkeypatch.setattr(llm_router, "get_llm_provider", factory)
    retained = []
    with pytest.raises(FileNotFoundError):
        inspect(native_session, receipt_observer=retained.append)
    assert retained[-1]["native_rollout_usage"] is None
    assert retained[-1]["native_provider_outcome"]["process_outcome"] == "unknown"
    assert get_last_cli_observation("grok_cli") == {}


def test_backend_discovery_failure_cannot_publish_an_unrelated_native_charge(native_session, route, monkeypatch):
    def failed(*_a, **_kw):
        set_last_cli_observation("grok_cli", observation())
        raise RuntimeError("authored discovery failed")
    monkeypatch.setattr(llm_router, "get_llm_provider", failed)
    retained = []
    with pytest.raises(RuntimeError):
        inspect(native_session, receipt_observer=retained.append)
    assert retained[-1]["backend_dispatches"] == 0
    assert retained[-1]["native_rollout_usage"] is None


def test_fresh_observer_failure_propagates_without_a_second_provider_call(native_session, route):
    error = OSError("authored receipt publication failed")
    def observer(_receipt): raise error
    with pytest.raises(OSError) as raised:
        inspect(native_session, receipt_observer=observer)
    assert raised.value is error and len(route[1]) == 1


def test_counter_binding_change_during_backend_is_refused_but_native_usage_survives(native_session, route):
    backend, _, _ = route
    supplied = counter()
    def changed(*_a):
        object.__setattr__(supplied, "tokenizer_revision", "different-counter")
        return PRIVATE
    backend.action = changed
    retained = []
    with pytest.raises(ValueError, match="current_input_differs"):
        inspect(native_session, token_counter=supplied, receipt_observer=retained.append)
    assert retained[-1]["native_rollout_usage"]["usage"]["total_tokens"] == 13


def test_metadata_receipt_and_repr_never_export_prompt_response_or_provider_error_bodies(native_session, route):
    result = inspect(native_session)
    r = result.receipt
    assert PRIVATE not in result.receipt_json and PRIVATE not in repr(result)
    assert result.selected_prompt not in repr(result) and "def add(value)" not in result.receipt_json
    assert REQUIRED[0].strip() not in result.receipt_json
    assert r["provider_delivery_verified"] is r["task_acceptance_verified"] is r["containment_verified"] is False
    assert r["native_cumulative_cap_enforced"] is r["max_output_token_limit_guaranteed"] is False
    assert r["whole_task_usage_complete"] is r["total_session_token_savings"] is None
    assert r["billing_total_verified"] is r["completion_authority"] is r["execution_authority"] is False
    copied = result.receipt
    copied["read_set"]["selected_paths"].append("forged")
    assert "forged" not in result.receipt["read_set"]["selected_paths"]


@pytest.mark.parametrize("changes", [dict(provider="codex_cli"), dict(model="other-model"),
    dict(timeout=True), dict(timeout=1801), dict(max_output_tokens=0), dict(requirements_guard=7),
    dict(receipt_observer=7)])
def test_invalid_route_controls_refuse_before_compile_or_provider_discovery(native_session, route, monkeypatch, changes):
    monkeypatch.setattr(subject, "compile_repository_read_set", lambda **_kw: pytest.fail("invalid control reached compilation"))
    with pytest.raises(ValueError):
        inspect(native_session, **changes)
    assert route[1] == [] and route[2] == []


def test_wrong_current_directory_refuses_before_expensive_work(native_session, route, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subject, "compile_repository_read_set", lambda **_kw: pytest.fail("wrong cwd reached compilation"))
    with pytest.raises(ValueError, match="workspace_differs"):
        inspect(native_session)
    assert route[2] == []
