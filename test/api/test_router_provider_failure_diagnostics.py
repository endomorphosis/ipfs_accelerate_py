"""Provider failures retain closed diagnoses without exporting private output."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
from ipfs_accelerate_py.llm_router import LLMRouterError
from test.api.test_supervisor_router_execution_contract import native_router


@pytest.fixture
def provider_failure_workspace(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index

    root = tmp_path / "workspace"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.chdir(root)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **_:
        SimpleNamespace(provider="codex_cli", model_name="pinned", catalog_revision="fixture"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *_, **__: object())
    return llm_router


@pytest.mark.parametrize("message,expected_code", [
    ("unauthorized", "authentication"),
    ("Codex quota exceeded (billing/plan hard limit)", "billing"),
    ("Codex usage limit reached", "rate_limit"),
    ("model not found", "not_found"),
    ("unrecognized provider diagnostic", "unknown"),
])
def test_provider_failure_projects_existing_taxonomy_and_preserves_usage(
        provider_failure_workspace, monkeypatch, capsys, message, expected_code):
    private_marker = "private-provider-diagnostic-body"
    error = LLMRouterError(message + " " + private_marker)
    observed = []

    def fail(prompt, **kwargs):
        observed.append(kwargs)
        set_last_cli_observation("codex_cli", {
            "exit_code": 1, "model_id": "pinned", "prompt_tokens": 17, "completion_tokens": 3,
        })
        raise error

    monkeypatch.setattr(provider_failure_workspace, "generate_text", fail)
    with pytest.raises(LLMRouterError) as caught:
        runner.run(prompt="private-instruction-body", provider="codex_cli", model="pinned",
            timeout=1, max_output_tokens=128, purpose="planning")
    assert caught.value is error
    captured = capsys.readouterr().out
    receipt = next(json.loads(line) for line in captured.splitlines() if line.startswith("{"))
    assert len(observed) == 1
    assert observed[0]["side_effecting"] is True
    assert observed[0]["allow_cross_provider_fallback"] is False
    assert receipt["provider_failure"] == {
        "phase": "provider_invocation", "reason_code": expected_code,
    }
    assert receipt["status"] == "failed"
    assert receipt["failure_phase"] == "provider_invocation"
    assert receipt["usage"] == {
        "exit_code": 1, "model_id": "pinned", "prompt_tokens": 17, "completion_tokens": 3,
    }
    assert receipt["completion_authority"] is False
    assert private_marker not in captured
    assert "private-instruction-body" not in captured


def test_typed_provider_timeout_keeps_partial_usage_and_private_command(
        provider_failure_workspace, monkeypatch, capsys):
    error = subprocess.TimeoutExpired(["provider", "private-command-argument"], 1)

    def fail(*_, **__):
        set_last_cli_observation("codex_cli", {"timed_out": True, "prompt_tokens": 17})
        raise error

    monkeypatch.setattr(provider_failure_workspace, "generate_text", fail)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        runner.run(prompt="instruction", provider="codex_cli", model="pinned", timeout=1,
            max_output_tokens=128, purpose="planning")
    assert caught.value is error
    captured = capsys.readouterr().out
    receipt = next(json.loads(line) for line in captured.splitlines() if line.startswith("{"))
    assert receipt["provider_failure"] == {
        "phase": "provider_invocation", "reason_code": "timeout",
    }
    assert receipt["usage"] == {"timed_out": True, "prompt_tokens": 17}
    assert "private-command-argument" not in captured


def test_classifier_failure_cannot_mask_original_provider_error(
        provider_failure_workspace, monkeypatch, capsys):
    from ipfs_accelerate_py.llm_allocation import observations
    error = LLMRouterError("private-original-error")

    def fail(*_, **__):
        raise error

    def broken_classifier(*_, **__):
        raise ValueError("private-classifier-error")

    monkeypatch.setattr(provider_failure_workspace, "generate_text", fail)
    monkeypatch.setattr(observations, "classify_provider_failure", broken_classifier)
    with pytest.raises(LLMRouterError) as caught:
        runner.run(prompt="instruction", provider="codex_cli", model="pinned", timeout=1,
            max_output_tokens=128, purpose="planning")
    assert caught.value is error
    captured = capsys.readouterr().out
    receipt = next(json.loads(line) for line in captured.splitlines() if line.startswith("{"))
    assert receipt["provider_failure"]["reason_code"] == "unknown"
    assert "private-original-error" not in captured
    assert "private-classifier-error" not in captured


def test_native_stdout_only_model_rejection_survives_shared_router(native_router, capsys):
    """A real local CLI exits before generation; only its native output is authored."""
    from ipfs_accelerate_py import llm_router

    workspace, cache, _ = native_router
    private_marker = "private-native-provider-message"
    nested = {"type": "error", "status": 400, "error": {
        "type": "invalid_request_error",
        "message": "The selected model is not supported for this account. " + private_marker,
    }}
    event = {"type": "error", "message": json.dumps(nested)}
    executable = Path(llm_router.shutil.which("codex"))
    executable.write_text("#!" + sys.executable + "\nimport sys\nsys.stdin.read()\n"
        + "print(" + repr(json.dumps(event)) + ")\nraise SystemExit(1)\n")

    with pytest.raises(LLMRouterError, match="^codex exec failed$") as caught:
        runner.run(prompt="private-task-instruction", provider="codex_cli", model="gpt-6.1-sol",
            timeout=30, max_output_tokens=128, purpose="planning")
    assert caught.value.codex_error_kind.value == "invalid_request"
    assert caught.value.codex_error_status == 400
    # Leave the old generic router retry/fallback decision inputs unchanged.
    assert not hasattr(caught.value, "status_code")
    captured = capsys.readouterr().out
    receipt = next(json.loads(line) for line in captured.splitlines()
        if line.startswith("{") and json.loads(line).get("schema") == "router-implementation-invocation@1")
    assert receipt["provider_failure"] == {
        "phase": "provider_invocation", "reason_code": "invalid_request", "http_status": 400,
    }
    assert receipt["usage"]["exit_code"] == 1
    assert "prompt_tokens" not in receipt["usage"]
    assert "completion_tokens" not in receipt["usage"]
    assert receipt["status"] == "failed"
    assert receipt["completion_authority"] is False
    assert (cache.reads, cache.writes) == (0, 0)
    assert private_marker not in captured
    assert "private-task-instruction" not in captured


@pytest.mark.parametrize("supplied_status", [True, "400", 400.0, 99, 600, None])
def test_native_error_status_requires_an_exact_bounded_integer(supplied_status):
    from ipfs_accelerate_py import llm_router
    nested = {"type": "error", "status": supplied_status,
        "error": {"type": "unknown", "message": "unclassified native error"}}
    stdout = json.dumps({"type": "error", "message": json.dumps(nested)})
    error = llm_router._codex_cli_failure("codex exec failed", stdout=stdout, stderr="")
    assert str(error) == "codex exec failed"
    assert not hasattr(error, "codex_error_status")
    assert not hasattr(error, "status_code")
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": "unknown",
    }


@pytest.mark.parametrize("diagnostic,expected", [
    ("Codex usage limit reached", "rate_limit"),
    ("insufficient_quota", "billing"),
    ("unauthorized", "authentication"),
])
def test_native_error_diagnostic_reuses_shared_provider_taxonomy(diagnostic, expected):
    from ipfs_accelerate_py import llm_router
    stdout = json.dumps({"type": "error", "message": diagnostic})
    error = llm_router._codex_cli_failure("existing exception wording", stdout=stdout, stderr="")
    assert str(error) == "existing exception wording"
    assert runner._provider_failure_diagnostic(error, provider="codex_cli") == {
        "phase": "provider_invocation", "reason_code": expected,
    }
