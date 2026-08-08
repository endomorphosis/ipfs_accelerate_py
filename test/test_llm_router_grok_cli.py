"""Tests for the first-class Grok CLI LLM router provider."""

from __future__ import annotations

import json
import subprocess

import pytest

from ipfs_accelerate_py import llm_router


class _Provider:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt: str, **kwargs: object) -> str:
        return f"{self.name}:{prompt}"


def _successful_grok_result(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    payload = {
        "text": "grok-cli-ok",
        "stopReason": "EndTurn",
        "sessionId": "session-1",
        "requestId": "request-1",
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }
    return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")


def test_grok_cli_provider_uses_bounded_headless_json_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        captured["prompt"] = open(cmd[cmd.index("--prompt-file") + 1], encoding="utf-8").read()
        return _successful_grok_result(list(cmd))

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(llm_router.subprocess, "run", fake_run)
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")
    monkeypatch.setenv("ipfs_accelerate_py_XAI_API_KEY", "alternate-test-key")
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    result = provider.generate(
        'Reply to "quoted text".',
        model_name="grok-4.5",
        timeout=12,
    )

    assert result == "grok-cli-ok"
    assert captured["prompt"] == 'Reply to "quoted text".'
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "grok"
    assert cmd[cmd.index("--model") + 1] == "grok-4.5"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--no-plan" in cmd
    assert "--no-subagents" in cmd
    assert "--disable-web-search" in cmd
    assert "--no-memory" in cmd
    assert "--verbatim" in cmd
    assert 'Reply to "quoted text".' not in cmd
    assert captured["env"]["XAI_API_KEY"] == "alternate-test-key"


def test_grok_cli_agent_command_is_noninteractive(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("implement the task", encoding="utf-8")

    cmd = llm_router.build_grok_cli_command(
        mode="agent",
        workspace=tmp_path,
        model_name="grok-4.5",
        max_turns=100_000,
        grok_bin="grok",
        prompt_file=prompt_path,
    )

    assert cmd[cmd.index("--model") + 1] == "grok-4.5"
    assert cmd[cmd.index("--max-turns") + 1] == "100000"
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
    assert cmd[cmd.index("--output-format") + 1] == "streaming-json"
    assert cmd[cmd.index("--cwd") + 1] == str(tmp_path.resolve())
    assert cmd[cmd.index("--prompt-file") + 1] == str(prompt_path)
    assert "--always-approve" in cmd
    assert "--tools" not in cmd


def test_grok_cli_provider_reports_missing_auth(monkeypatch) -> None:
    error = {
        "type": "error",
        "message": "Not signed in. To authenticate, run grok login --device-code.",
    }

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(
        llm_router.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout=json.dumps(error),
            stderr="Error: Not signed in",
        ),
    )
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    with pytest.raises(llm_router.LLMRouterError, match="grok login --device-code"):
        provider.generate("hello", disable_model_retry=True)


def test_grok_cli_provider_rejects_empty_success_payload(monkeypatch) -> None:
    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(
        llm_router.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"text": "", "stopReason": "EndTurn"}),
            stderr="",
        ),
    )
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    with pytest.raises(llm_router.LLMRouterError, match="returned no response text"):
        provider.generate("hello")


def test_grok_cli_command_template_preserves_prompt_as_one_argument(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(cmd, 0, stdout="template-ok", stderr="")

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(llm_router.subprocess, "run", fake_run)
    monkeypatch.setenv(
        "ipfs_accelerate_py_GROK_CLI_CMD",
        "grok-wrapper --model {model} --prompt {prompt}",
    )

    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    result = provider.generate('Keep "this prompt" together.', model_name="grok-4.5")

    assert result == "template-ok"
    assert captured["cmd"] == [
        "grok-wrapper",
        "--model",
        "grok-4.5",
        "--prompt",
        'Keep "this prompt" together.',
    ]
    assert captured["input"] is None


def test_grok_cli_trace_records_usage_without_prompt(monkeypatch, tmp_path) -> None:
    secret_prompt = "private prompt that must not be traced"

    monkeypatch.setattr(llm_router, "_cli_available", lambda _command: True)
    monkeypatch.setattr(
        llm_router.subprocess,
        "run",
        lambda cmd, **kwargs: _successful_grok_result(list(cmd)),
    )
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_CMD", "grok")

    trace_path = tmp_path / "grok.jsonl"
    provider = llm_router._get_grok_cli_provider()
    assert provider is not None
    assert provider.generate(secret_prompt, trace_jsonl_path=str(trace_path)) == "grok-cli-ok"

    trace_text = trace_path.read_text(encoding="utf-8")
    record = json.loads(trace_text)
    assert record["provider"] == "grok_cli"
    assert record["sessionId"] == "session-1"
    assert record["usage"]["total_tokens"] == 5
    assert secret_prompt not in trace_text


def test_grok_alias_prefers_cli_and_explicit_api_alias_uses_rest(monkeypatch) -> None:
    cli = _Provider("cli")
    api = _Provider("api")
    monkeypatch.setattr(llm_router, "_get_grok_cli_provider", lambda: cli)
    monkeypatch.setattr(llm_router, "_get_xai_provider", lambda: api)

    assert llm_router._builtin_provider_by_name("grok") is cli
    assert llm_router._builtin_provider_by_name("grok_cli") is cli
    assert llm_router._builtin_provider_by_name("xai_cli") is cli
    assert llm_router._builtin_provider_by_name("xai") is api
    assert llm_router._builtin_provider_by_name("grok_api") is api


def test_grok_alias_falls_back_to_rest_when_cli_is_missing(monkeypatch) -> None:
    api = _Provider("api")
    monkeypatch.setattr(llm_router, "_get_grok_cli_provider", lambda: None)
    monkeypatch.setattr(llm_router, "_get_xai_provider", lambda: api)

    assert llm_router._builtin_provider_by_name("grok") is api


def test_grok_cli_auto_discovery_requires_auth(monkeypatch) -> None:
    calls: list[str] = []
    grok = _Provider("grok")

    def fake_builtin(name: str):
        calls.append(name)
        return grok if name == "grok_cli" else None

    monkeypatch.delenv("ipfs_accelerate_py_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(llm_router, "_get_accelerate_provider", lambda _deps: None)
    monkeypatch.setattr(llm_router, "_get_local_hf_provider", lambda **kwargs: None)
    monkeypatch.setattr(llm_router, "_builtin_provider_by_name", fake_builtin)
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: False)

    with pytest.raises(RuntimeError, match="No LLM provider available"):
        llm_router._resolve_provider_uncached(None, deps=llm_router.get_default_router_deps())
    assert "grok_cli" not in calls

    calls.clear()
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: True)
    assert llm_router._resolve_provider_uncached(
        None,
        deps=llm_router.get_default_router_deps(),
    ) is grok
    assert "grok_cli" in calls


@pytest.mark.parametrize(
    ("kind", "reason"),
    (
        (
            llm_router.AgentCLIProviderFailureKind.GROK_QUOTA_EXHAUSTED,
            "grok_provider_insufficient_quota",
        ),
        (
            llm_router.AgentCLIProviderFailureKind.AUTHENTICATION_FAILURE,
            "grok_authentication_failure",
        ),
        (
            llm_router.AgentCLIProviderFailureKind.LAUNCH_FAILURE,
            "grok_process_did_not_launch",
        ),
    ),
)
def test_grok_codex_agent_route_allows_only_typed_pre_side_effect_failure(
    kind,
    reason,
    tmp_path,
) -> None:
    snapshot = llm_router.snapshot_agent_cli_workspace(tmp_path)
    classification = llm_router.AgentCLIFailureClassification(kind, reason)
    receipt = llm_router.serialize_agent_cli_failure_receipt(
        classification,
        returncode=19,
        activity_state=llm_router.AgentCLIActivityState.NO_ACTIVITY,
    )

    decision = llm_router.route_agent_cli_failure(
        policy=llm_router.GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY,
        primary_provider="grok",
        fallback_provider="codex",
        primary_result=llm_router.AgentCLIProviderResult(19),
        workspace_before=snapshot,
        workspace_after=snapshot,
        trusted_failure_receipt=receipt,
    )

    assert decision.should_fallback is True
    assert decision.route_record["failure_kind"] == kind.value
    assert decision.route_record["side_effects_started"] is False
    assert decision.route_record["completion_authority"] is False


@pytest.mark.parametrize(
    "kind",
    (
        llm_router.AgentCLIProviderFailureKind.GENERIC_NONZERO_EXIT,
        llm_router.AgentCLIProviderFailureKind.TASK_FAILURE,
        llm_router.AgentCLIProviderFailureKind.TRANSPORT_FAILURE,
        llm_router.AgentCLIProviderFailureKind.TIMEOUT,
        llm_router.AgentCLIProviderFailureKind.MALFORMED_OUTPUT,
    ),
)
def test_grok_codex_agent_route_keeps_terminal_failure_pinned(
    kind,
    tmp_path,
) -> None:
    snapshot = llm_router.snapshot_agent_cli_workspace(tmp_path)
    receipt = llm_router.serialize_agent_cli_failure_receipt(
        llm_router.AgentCLIFailureClassification(kind, "grok_terminal_failure"),
        returncode=19,
        activity_state=llm_router.AgentCLIActivityState.NO_ACTIVITY,
    )

    decision = llm_router.route_agent_cli_failure(
        policy=llm_router.GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY,
        primary_provider="grok",
        fallback_provider="codex",
        primary_result=llm_router.AgentCLIProviderResult(19),
        workspace_before=snapshot,
        workspace_after=snapshot,
        trusted_failure_receipt=receipt,
    )

    assert decision.should_fallback is False
    assert decision.terminal_reason == "failure_not_fallback_eligible"


@pytest.mark.parametrize(
    "activity",
    (
        llm_router.AgentCLIActivityState.STARTED,
        llm_router.AgentCLIActivityState.UNKNOWN,
    ),
)
def test_grok_codex_agent_route_rejects_failure_after_side_effect_event(
    activity,
    tmp_path,
) -> None:
    snapshot = llm_router.snapshot_agent_cli_workspace(tmp_path)
    receipt = llm_router.serialize_agent_cli_failure_receipt(
        llm_router.AgentCLIFailureClassification(
            llm_router.AgentCLIProviderFailureKind.GROK_QUOTA_EXHAUSTED,
            "grok_provider_insufficient_quota",
        ),
        returncode=19,
        activity_state=activity,
    )
    decision = llm_router.route_agent_cli_failure(
        policy=llm_router.GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY,
        primary_provider="grok",
        fallback_provider="codex",
        primary_result=llm_router.AgentCLIProviderResult(19),
        workspace_before=snapshot,
        workspace_after=snapshot,
        trusted_failure_receipt=receipt,
    )
    assert decision.should_fallback is False
    assert decision.terminal_reason == "side_effects_started"


def test_grok_codex_agent_route_rejects_mutated_workspace_and_forged_receipt(
    tmp_path,
) -> None:
    candidate = tmp_path / "candidate.py"
    candidate.write_text("before\n", encoding="utf-8")
    before = llm_router.snapshot_agent_cli_workspace(tmp_path)
    candidate.write_text("after\n", encoding="utf-8")
    after = llm_router.snapshot_agent_cli_workspace(tmp_path)
    receipt = llm_router.serialize_agent_cli_failure_receipt(
        llm_router.AgentCLIFailureClassification(
            llm_router.AgentCLIProviderFailureKind.AUTHENTICATION_FAILURE,
            "grok_authentication_failure",
        ),
        returncode=19,
        activity_state=llm_router.AgentCLIActivityState.NO_ACTIVITY,
    )
    mutated = llm_router.route_agent_cli_failure(
        policy=llm_router.GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY,
        primary_provider="grok",
        fallback_provider="codex",
        primary_result=llm_router.AgentCLIProviderResult(19),
        workspace_before=before,
        workspace_after=after,
        trusted_failure_receipt=receipt,
    )
    forged = llm_router.route_agent_cli_failure(
        policy=llm_router.GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY,
        primary_provider="grok",
        fallback_provider="codex",
        primary_result=llm_router.AgentCLIProviderResult(19),
        workspace_before=after,
        workspace_after=after,
        trusted_failure_receipt=json.dumps(
            {
                **json.loads(receipt),
                "provider_body": "authentication_failure",
            }
        ),
    )
    assert mutated.terminal_reason == "side_effects_started"
    assert forged.classification.kind is (
        llm_router.AgentCLIProviderFailureKind.MALFORMED_OUTPUT
    )
    assert forged.should_fallback is False


def test_agent_cli_workspace_snapshot_hashes_mode_and_symlink_target(tmp_path) -> None:
    target = tmp_path / "target"
    target.write_text("same bytes\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to("target")
    initial = llm_router.snapshot_agent_cli_workspace(tmp_path)
    target.chmod(0o700)
    mode_changed = llm_router.snapshot_agent_cli_workspace(tmp_path)
    link.unlink()
    link.symlink_to("other")
    link_changed = llm_router.snapshot_agent_cli_workspace(tmp_path)
    assert initial.reliable and mode_changed.reliable and link_changed.reliable
    assert len({initial.digest, mode_changed.digest, link_changed.digest}) == 3


def test_agent_cli_workspace_snapshot_hashes_gitignored_file(tmp_path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / ".gitignore").write_text("private.cache\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "tracked.txt"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    ignored = tmp_path / "private.cache"
    ignored.write_text("before\n", encoding="utf-8")

    before = llm_router.snapshot_agent_cli_workspace(tmp_path)
    ignored.write_text("after\n", encoding="utf-8")
    after = llm_router.snapshot_agent_cli_workspace(tmp_path)

    assert before.reliable and after.reliable
    assert before.digest != after.digest


def test_agent_cli_workspace_snapshot_hashes_git_control_file_and_prunes_git_dirs(
    tmp_path,
) -> None:
    control = tmp_path / ".git"
    control.write_text("gitdir: /tmp/first-control-dir\n", encoding="utf-8")
    nested_git = tmp_path / "nested" / ".git"
    nested_git.mkdir(parents=True)
    (nested_git / "private-state").write_text("ignored\n", encoding="utf-8")

    paths = llm_router._filesystem_agent_cli_snapshot_paths(tmp_path)
    before = llm_router.snapshot_agent_cli_workspace(tmp_path)
    control.write_text("gitdir: /tmp/second-control-dir\n", encoding="utf-8")
    after = llm_router.snapshot_agent_cli_workspace(tmp_path)

    assert ".git" in paths
    assert not any(path.startswith("nested/.git") for path in paths)
    assert before.reliable and after.reliable
    assert before.digest != after.digest


def test_agent_cli_classification_terminal_evidence_precedes_auth() -> None:
    timeout = llm_router.classify_grok_agent_cli_failure(
        llm_router.AgentCLIProviderResult(
            19,
            stderr="authentication failed; request timed out",
        )
    )
    transport = llm_router.classify_grok_agent_cli_failure(
        llm_router.AgentCLIProviderResult(
            19,
            stderr="authentication failed; connection reset by peer",
        )
    )
    assert timeout.kind is llm_router.AgentCLIProviderFailureKind.TIMEOUT
    assert transport.kind is (
        llm_router.AgentCLIProviderFailureKind.TRANSPORT_FAILURE
    )


def test_agent_route_readiness_negative_auth_text_outranks_zero_exit(
    monkeypatch,
) -> None:
    probes = iter(
        (
            (0, "You are not authenticated.\nDefault model: grok-4.5", None),
            (0, "Not logged in\nLogged in using ChatGPT", None),
        )
    )
    monkeypatch.setattr(
        llm_router,
        "_bounded_agent_cli_probe",
        lambda command, timeout_seconds: next(probes),
    )
    readiness = llm_router.probe_grok_codex_agent_route_readiness(
        grok_bin="/provider/grok",
        codex_bin="/provider/codex",
    )
    assert readiness.grok_ready is False
    assert readiness.codex_ready is False
    assert readiness.effective_provider == ""
    assert readiness.failure_kind is (
        llm_router.AgentCLIProviderFailureKind.AUTHENTICATION_FAILURE
    )


def test_agent_route_readiness_accepts_positive_login_status(
    monkeypatch,
) -> None:
    probes = iter(
        (
            (0, "Login successful; available model grok-4.5", None),
            (0, "Logged in using ChatGPT", None),
        )
    )
    monkeypatch.setattr(
        llm_router,
        "_bounded_agent_cli_probe",
        lambda command, timeout_seconds: next(probes),
    )

    readiness = llm_router.probe_grok_codex_agent_route_readiness(
        grok_bin="/provider/grok",
        codex_bin="/provider/codex",
    )

    assert readiness.grok_ready is True
    assert readiness.codex_ready is True
    assert readiness.effective_provider == "grok"
    assert readiness.failure_kind is None
