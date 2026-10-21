"""Deterministic native transport fixtures, never a live-provider qualification."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from ipfs_accelerate_py import llm_router as router
from ipfs_accelerate_py.cli_runtime import grok_invocation_observation as evidence
from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser

SECRET = "synthetic-private-body-and-secret"


def result(*, model="grok-4.7-build", frames=None, returncode=0):
    parser = GrokStreamParser()
    events = frames if frames is not None else [
        {"type": "text", "data": "SYNTHETIC_OK"},
        {"type": "end", "stopReason": "end_turn", "usage": {"output_tokens": 1},
         "modelUsage": {model: {"prompt": SECRET}}, "prompt": SECRET},
    ]
    parser.feed(b"".join(json.dumps(frame).encode() + b"\n" for frame in events), elapsed_seconds=.1)
    parser.eof("stdout", elapsed_seconds=.2)
    parser.eof("stderr", elapsed_seconds=.2)
    parser.process_exit(returncode, elapsed_seconds=.2)
    progress = parser.snapshot(elapsed_seconds=.2)
    payload = parser.final_payload
    return SimpleNamespace(returncode=returncode, timed_out=False, progress=progress,
        final_payload=payload, stdout=json.dumps({**(payload or {}), "text": parser.response_text}),
        stderr="", response_text=parser.response_text)


@pytest.fixture
def native(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("real process forbidden")
    monkeypatch.setattr(router.subprocess, "run", forbidden)
    monkeypatch.setattr(router.subprocess, "Popen", forbidden)
    monkeypatch.setattr(router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(router, "_grok_cli_command", lambda: "/fixture/grok")
    monkeypatch.setattr(router, "get_default_router_deps", lambda: SimpleNamespace())
    monkeypatch.setattr(router, "_response_cache_enabled", lambda: False)
    provider = router._get_grok_cli_provider()
    monkeypatch.setattr(router, "get_llm_provider", lambda name, **_kwargs: provider)
    @contextmanager
    def sampling(environment, **_kwargs):
        yield environment
    monkeypatch.setattr(router, "_grok_zero_temperature_environment", sampling)
    return provider


def invoke(**kwargs):
    return router.generate_text_with_observation("synthetic public prompt", provider="grok_cli",
        model_name="grok-4.7", grok_max_turns=1, grok_tools="", **kwargs)


def test_existing_native_process_path_binds_only_safe_final_model_keys(native, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    calls = []
    def capture(argv, **kwargs):
        calls.append(argv)
        assert Path(argv[argv.index("--prompt-file") + 1]).read_text() == "synthetic public prompt"
        return result()
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", capture)
    observed = invoke(allow_local_fallback=True, allow_cross_provider_fallback=True)
    receipt = observed.observation.to_dict()
    assert observed.text == "SYNTHETIC_OK"
    assert len(calls) == 1
    assert calls[0][calls[0].index("--output-format") + 1] == "streaming-json"
    assert receipt["native_dispatch_count"] == 1
    assert receipt["selected_provider_name"] == "grok_cli"
    assert receipt["selected_model_name"] == "grok-4.7"
    assert receipt["provider_reported_model"] == "grok-4.7-build"
    assert receipt["wire_model_observed"] is True
    assert receipt["stream_complete"] is True
    assert receipt["process_returncode"] == 0
    assert receipt["observation_only"] is True and receipt["event_origin_authenticated"] is False
    assert SECRET not in json.dumps(receipt)
    assert "synthetic public prompt" not in json.dumps(receipt)
    assert "SYNTHETIC_OK" not in repr(observed)
    assert not Path(calls[0][calls[0].index("--prompt-file") + 1]).exists()


@pytest.mark.parametrize("usage", [None, {}, [], {"grok-4.7": {}, "grok-4": {}},
    {SECRET: {}}, {"grok-4.synthetic-secret-key": {}}, {"grok-4.7 key": {}}, {"grok-" + "x" * 200: {}}, {"grok-4.7": []},
    {"grok-4.7": None}])
def test_malformed_or_multiple_model_keys_remain_unobserved(usage):
    proc = result()
    payload = {**proc.final_payload, "modelUsage": usage}
    with evidence.capture_grok_invocation() as capture:
        evidence.record_native_grok_result(payload, model="grok-4.7", returncode=0, progress=proc.progress, dispatch_token=evidence.begin_native_grok_dispatch())
    receipt = capture.observation.to_dict()
    assert receipt["native_dispatch_count"] == 1
    assert receipt["provider_reported_model"] == ""
    assert receipt["wire_model_observed"] is False
    assert SECRET not in json.dumps(receipt)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(type="usage"), lambda p: p.update(stopReason="max_turn_requests"),
    lambda p: p.update(stopReason="max_tokens"), lambda p: p.update(stop_reason="refusal"),
    lambda p: p.update(is_error=True), lambda p: p.update(error={"message": SECRET}),
])
def test_nonfinal_or_unsuccessful_payload_never_supplies_identity(mutation):
    proc = result()
    payload = dict(proc.final_payload)
    mutation(payload)
    with evidence.capture_grok_invocation() as capture:
        evidence.record_native_grok_result(payload, model="grok-4.7", returncode=0, progress=proc.progress, dispatch_token=evidence.begin_native_grok_dispatch())
    assert capture.observation.provider_reported_model == ""
    assert capture.observation.stream_complete is False


def test_usage_event_model_and_model_body_do_not_become_final_identity(native, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    frames = [{"type": "text", "data": SECRET},
        {"type": "usage", "modelUsage": {"grok-4.7": {}}},
        {"type": "text", "data": "SYNTHETIC_OK"}, {"type": "end", "stopReason": "end_turn"}]
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", lambda *_args, **_kwargs: result(frames=frames))
    observed = invoke()
    assert observed.text == "SYNTHETIC_OK"
    assert observed.observation.provider_reported_model == ""
    assert observed.observation.wire_model_observed is False


@pytest.mark.parametrize("returncode", [1, -15, True, None])
def test_nonzero_unknown_or_boolean_process_exit_never_attests_model(returncode):
    proc = result()
    with evidence.capture_grok_invocation() as capture:
        evidence.record_native_grok_result(proc.final_payload, model="grok-4.7", returncode=returncode, progress=proc.progress, dispatch_token=evidence.begin_native_grok_dispatch())
    assert capture.observation.provider_reported_model == ""
    assert capture.observation.process_returncode is None


def test_timeout_or_stream_failure_closes_scope_before_next_unobserved_call(native, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    def timeout(*_args, **_kwargs):
        raise router.subprocess.TimeoutExpired("synthetic", 1, output=SECRET)
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", timeout)
    with pytest.raises(router.subprocess.TimeoutExpired):
        invoke()
    monkeypatch.setattr(router, "generate_text", lambda *_args, **_kwargs: "cached or injected")
    observed = invoke()
    assert observed.text == "cached or injected"
    assert observed.observation.native_dispatch_count == 0
    assert observed.observation.provider_reported_model == ""
    assert observed.observation.selected_provider_name == ""


def test_cache_or_injected_provider_cannot_reuse_global_or_previous_identity(native, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture, cli_metadata
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", lambda *_args, **_kwargs: result())
    first = invoke()
    cli_metadata.set_last_cli_observation("grok_cli", {"model_id": "grok-4.7-build"})
    monkeypatch.setattr(router, "generate_text", lambda *_args, **_kwargs: "cache hit")
    second = invoke()
    assert second.observation.invocation_id != first.observation.invocation_id
    assert second.observation.native_dispatch_count == 0
    assert second.observation.selected_provider_name == ""
    assert second.observation.provider_reported_model == ""


def test_nested_and_double_dispatch_cannot_reuse_observation(native, monkeypatch):
    proc = result()
    with evidence.capture_grok_invocation() as capture:
        with pytest.raises(ValueError, match="nested"):
            invoke()
        for _ in range(2):
            evidence.record_native_grok_result(proc.final_payload, model="grok-4.7", returncode=0, progress=proc.progress, dispatch_token=evidence.begin_native_grok_dispatch())
    assert capture.observation.native_dispatch_count == 2
    assert capture.observation.provider_reported_model == ""
    with pytest.raises(ValueError, match="closed"):
        capture.record(proc.final_payload, model="grok-4.7", returncode=0, progress=proc.progress, dispatch_token=None)


def test_concurrent_invocations_do_not_share_model_or_identity(native, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    barrier = threading.Barrier(2)
    def capture(argv, **_kwargs):
        barrier.wait(timeout=5)
        return result(model=argv[argv.index("--model") + 1])
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", capture)
    def call(model):
        return router.generate_text_with_observation("synthetic", provider="grok_cli", model_name=model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(call, ["grok-4.7", "grok-4.6"]))
    assert first.observation.invocation_id != second.observation.invocation_id
    assert first.observation.provider_reported_model == "grok-4.7"
    assert second.observation.provider_reported_model == "grok-4.6"


@pytest.mark.parametrize("kwargs", [{"provider": "grok"}, {"provider": "codex_cli"},
    {"provider": "grok_cli", "grok_stream_progress": False}])
def test_observation_api_refuses_other_provider_or_nonstreaming_before_dispatch(monkeypatch, kwargs):
    monkeypatch.setattr(router, "generate_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dispatch")))
    with pytest.raises(ValueError):
        router.generate_text_with_observation("synthetic", **kwargs)


def test_caller_request_binding_is_preserved_without_reusing_native_invocation(native, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", lambda *_args, **_kwargs: result())
    request_id = "a" * 32
    first = invoke(observation_request_id=request_id)
    second = invoke(observation_request_id=request_id)
    assert first.observation.request_id == second.observation.request_id == request_id
    assert first.observation.invocation_id != second.observation.invocation_id


@pytest.mark.parametrize("request_id", ["", "secret", "A" * 32, True])
def test_invalid_request_binding_fails_before_generation(monkeypatch, request_id):
    monkeypatch.setattr(router, "generate_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dispatch")))
    with pytest.raises(ValueError, match="request id"):
        invoke(observation_request_id=request_id)


def test_caught_timeout_then_success_remains_two_dispatches_not_one(native, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    calls = []
    def capture(*_args, **_kwargs):
        calls.append("synthetic-process")
        if len(calls) == 1:
            raise router.subprocess.TimeoutExpired("synthetic", 1, output=SECRET)
        return result()
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", capture)
    def catch_and_repeat(prompt, **kwargs):
        kwargs.pop("provider", None)
        try:
            native.generate(prompt, **kwargs)
        except router.subprocess.TimeoutExpired:
            pass
        return native.generate(prompt, **kwargs)
    monkeypatch.setattr(router, "generate_text", catch_and_repeat)
    observed = invoke()
    assert len(calls) == 2 and observed.text == "SYNTHETIC_OK"
    assert observed.observation.native_dispatch_count == 2
    assert observed.observation.provider_reported_model == ""
    assert observed.observation.selected_provider_name == ""


def test_late_or_reused_finalization_token_cannot_attest_a_new_dispatch():
    proc = result()
    with evidence.capture_grok_invocation() as capture:
        old = evidence.begin_native_grok_dispatch()
        current = evidence.begin_native_grok_dispatch()
        evidence.record_native_grok_result(proc.final_payload, model="grok-4.7", returncode=0,
            progress=proc.progress, dispatch_token=old)
        evidence.record_native_grok_result(proc.final_payload, model="grok-4.7", returncode=0,
            progress=proc.progress, dispatch_token=current)
    assert capture.observation.native_dispatch_count == 2
    assert capture.observation.provider_reported_model == ""
    with evidence.capture_grok_invocation() as single:
        token = evidence.begin_native_grok_dispatch()
        for _ in range(2):
            evidence.record_native_grok_result(proc.final_payload, model="grok-4.7", returncode=0,
                progress=proc.progress, dispatch_token=token)
    assert single.observation.provider_reported_model == ""


def test_unmatched_first_finalization_permanently_invalidates_that_launch():
    proc = result()
    with evidence.capture_grok_invocation() as capture:
        token = evidence.begin_native_grok_dispatch()
        for supplied in (object(), token):
            evidence.record_native_grok_result(proc.final_payload, model="grok-4.7", returncode=0,
                progress=proc.progress, dispatch_token=supplied)
    assert capture.observation.native_dispatch_count == 1
    assert capture.observation.provider_reported_model == ""
