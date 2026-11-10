"""Authored pruning dispatch and native-receipt plumbing; no model calls."""
import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime import grok_pruning as pruning
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser, GrokStreamResult
from test.api.test_grok_stream_router import provider, end
from test.api.test_grok_container_router import routed_workspace, envelope


PINNED = "/opt/ipfs-supervisor/provider-bin/grok"


def stream_result(*, timed_out=False):
    parser = GrokStreamParser()
    parser.feed((json.dumps({"type": "text", "data": "Authored answer"}) + "\n"
        + json.dumps(end()) + "\n").encode(), elapsed_seconds=0.01)
    parser.eof("stdout", elapsed_seconds=0.02)
    parser.eof("stderr", elapsed_seconds=0.02)
    parser.process_exit(0, elapsed_seconds=0.02)
    # Match capture's public result contract without executing a provider.
    return GrokStreamResult(args=[PINNED], stdout="", stderr="", returncode=0,
        progress=parser.snapshot(), final_payload=end(), response_text="Authored answer", timed_out=timed_out)


@pytest.mark.parametrize("profile", pruning.GROK_PRUNING_PROFILES[1:])
def test_exact_profile_preflight_precedes_dispatch_and_is_retained(provider, monkeypatch, profile):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    calls = []
    receipt = authored_preflight(profile)
    monkeypatch.setenv("PRUNING_AUTHORED_ENV", "before")

    def preflight(selected, **kwargs):
        assert selected == profile and kwargs["command"] == [PINNED]
        assert kwargs["env"]["PRUNING_AUTHORED_ENV"] == "before"
        assert 0 < kwargs["timeout"] <= 3
        monkeypatch.setenv("PRUNING_AUTHORED_ENV", "after")
        calls.append("preflight")
        return receipt

    def capture(command, *, env, timeout, **kwargs):
        assert calls == ["preflight"]
        assert env["PRUNING_AUTHORED_ENV"] == "before"
        assert 0 < timeout <= 3
        assert Path(command[command.index("--prompt-file") + 1]).read_text() == "Authored instruction"
        calls.append("capture")
        return stream_result()

    monkeypatch.setattr(pruning, "run_grok_pruning_preflight", preflight)
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", capture)
    assert provider.generate("Authored instruction", timeout=3, grok_cli_cmd=[PINNED],
        grok_stream_progress=True, grok_pruning_profile=profile) == "Authored answer"
    assert calls == ["preflight", "capture"]
    assert get_last_cli_observation("grok_cli")["native_grok_pruning"] == receipt


@pytest.mark.parametrize("profile", [None, True, False, 1, {}, [], "unknown"])
def test_invalid_profile_refuses_before_prompt_file(provider, monkeypatch, profile):
    monkeypatch.setattr(llm_router.tempfile, "NamedTemporaryFile", lambda **_: pytest.fail("prompt created"))
    with pytest.raises(ValueError, match="pruning profile"):
        provider.generate("Authored", grok_cli_cmd=[PINNED], grok_stream_progress=True,
            grok_pruning_profile=profile)


@pytest.mark.parametrize("change", ["custom_binary", "extra_argument", "nonstream", "resume", "continue", "session"])
def test_explicit_profile_refuses_unqualified_routes(provider, monkeypatch, change):
    monkeypatch.setattr(pruning, "run_grok_pruning_preflight", lambda *_a, **_k: pytest.fail("inspect reached"))
    options = dict(grok_cli_cmd=[PINNED], grok_stream_progress=True,
                   grok_pruning_profile="grok-pruning-on@1")
    if change == "custom_binary": options["grok_cli_cmd"] = ["/fixture/grok"]
    if change == "extra_argument": options["grok_cli_cmd"] = [PINNED, "--config", "fixture.toml"]
    if change == "nonstream": options["grok_stream_progress"] = False
    if change == "resume": options["resume_session_id"] = "authored"
    if change == "continue": options["continue_session"] = True
    if change == "session": options["chat_session_id"] = "authored"
    with pytest.raises(ValueError, match="pruning"):
        provider.generate("Authored", **options)


def test_preflight_failure_creates_no_prompt_or_provider_process(provider, monkeypatch):
    def fail(*_a, **_k):
        raise ValueError("authored config drift")
    monkeypatch.setattr(pruning, "run_grok_pruning_preflight", fail)
    monkeypatch.setattr(llm_router.tempfile, "NamedTemporaryFile", lambda **_: pytest.fail("prompt created"))
    with pytest.raises(ValueError, match="config drift"):
        provider.generate("Authored", grok_cli_cmd=[PINNED], grok_stream_progress=True,
            grok_pruning_profile="grok-pruning-on@1")


def test_timeout_retains_preflight_but_does_not_complete_usage(provider, monkeypatch):
    from ipfs_accelerate_py.cli_runtime import grok_stream_capture
    receipt = authored_preflight("grok-pruning-off@1")
    monkeypatch.setattr(pruning, "run_grok_pruning_preflight", lambda *_a, **_k: receipt)
    def capture(*_a, **_k):
        raise subprocess.TimeoutExpired([PINNED], 3, output="", stderr="")
    monkeypatch.setattr(grok_stream_capture, "capture_grok_stream", capture)
    with pytest.raises(subprocess.TimeoutExpired):
        provider.generate("Authored", timeout=3, grok_cli_cmd=[PINNED], grok_stream_progress=True,
            grok_pruning_profile="grok-pruning-off@1")
    observation = get_last_cli_observation("grok_cli")
    assert observation["native_grok_pruning"] == receipt
    assert observation["timed_out"] is True
    assert observation.get("native_grok_task_complete") is not True
    assert observation.get("native_grok_usage_complete") is not True


def authored_preflight(profile):
    value = {key: [] for key in ("agents", "hooks", "lspServers", "marketplaces",
        "mcpServers", "plugins", "projectInstructions", "skills")}
    value.update(configSources={"layers": []}, externalCompat={}, loginPolicy={}, permissions={},
        channel="authored", cwd="/authored", grokVersion="authored", projectRoot=None, projectTrusted=False)
    effective = pruning.observe_grok_pruning_inspection(profile,
        stdout=json.dumps(value).encode(), stderr=b"", returncode=0)
    return pruning.build_grok_pruning_preflight(profile,
        file_observation=pruning._expected_file_observation(profile, 0, 0), effective_observation=effective)


@pytest.mark.parametrize("profile", pruning.GROK_PRUNING_PROFILES[1:])
def test_runner_preserves_profile_and_config_evidence_without_inventing_effectiveness(routed_workspace, monkeypatch, profile):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", lambda **_: {"authored": True})
    preflight = authored_preflight(profile)
    calls = []
    def generate(prompt, **kwargs):
        calls.append(kwargs)
        assert kwargs["grok_pruning_profile"] == profile
        set_last_cli_observation("grok_cli", {"exit_code": 0, **native_grok_observation(envelope(), exit_code=0),
            "native_grok_pruning": preflight})
        return "Authored response"
    monkeypatch.setattr(routed_workspace, "generate_text", generate)
    _, receipt = runner.run(prompt="Authored task", provider="grok_cli", model="grok-4.7",
        timeout=30, max_output_tokens=10, purpose="coding", grok_pruning_profile=profile,
        container_boundary=Path("/authored/boundary"), container_boundary_sha256="a" * 64)
    assert len(calls) == 1
    assert receipt["provider_invocation_policy"]["grok_pruning"] == preflight
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 33
    assert preflight["effective_settings_verified"] is False
    assert preflight["config_file_consumption_verified"] is False
    assert preflight["token_savings_established"] is False
    assert pruning.grok_pruning_runtime_matches({"provider_invocations": [receipt]}, profile) is True


@pytest.mark.parametrize("change", ["missing", "wrong_profile", "tampered", "claims_effectiveness"])
def test_runner_rejects_success_without_selected_preflight(routed_workspace, monkeypatch, capsys, change):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.cli_runtime.grok_native_usage import native_grok_observation
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", lambda **_: {"authored": True})
    observation = {"exit_code": 0, **native_grok_observation(envelope(), exit_code=0)}
    if change != "missing":
        preflight = authored_preflight("grok-pruning-off@1" if change == "wrong_profile" else "grok-pruning-on@1")
        if change == "tampered": preflight["PRIVATE"] = "DO-NOT-EXPORT"
        if change == "claims_effectiveness": preflight["effective_settings_verified"] = True
        observation["native_grok_pruning"] = preflight
    def generate(*_args, **_kwargs):
        set_last_cli_observation("grok_cli", observation)
        return "Authored response"
    monkeypatch.setattr(routed_workspace, "generate_text", generate)
    with pytest.raises(RuntimeError, match="pruning preflight receipt"):
        runner.run(prompt="Authored task", provider="grok_cli", model="grok-4.7",
            timeout=30, max_output_tokens=10, purpose="coding", grok_pruning_profile="grok-pruning-on@1",
            container_boundary=Path("/authored/boundary"), container_boundary_sha256="a" * 64)
    printed = capsys.readouterr().out
    receipt = json.loads(printed.splitlines()[-1])
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "provider_result_validation"
    assert receipt["grok_pruning_error"] == "missing_or_invalid_preflight"
    assert "grok_pruning" not in receipt["provider_invocation_policy"]
    assert receipt["native_rollout_usage"]["usage"]["total_tokens"] == 33
    assert "DO-NOT-EXPORT" not in printed
