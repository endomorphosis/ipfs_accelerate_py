"""Pinned router dispatch preserves actual CLI status, effort and usage."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


def test_codex_router_forwards_reasoning_effort_without_shell_interpolation(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation

    monkeypatch.setattr(llm_router.shutil, "which", lambda name: "/test/codex")
    observed = []

    def execute(command, **kwargs):
        observed.append((command, kwargs))
        Path(command[command.index("--output-last-message") + 1]).write_text("done")
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed","usage":{"input_tokens":15,"output_tokens":4}}', stderr="")

    monkeypatch.setattr(llm_router.subprocess, "run", execute)
    provider = llm_router._get_codex_cli_provider()
    assert provider.generate("literal $(no command)", model_name="pinned", reasoning_effort="high") == "done"
    command, options = observed[0]
    assert command[command.index("-c") + 1] == 'model_reasoning_effort="high"'
    assert options["input"] == "literal $(no command)" and options.get("shell", False) is False
    usage = get_last_cli_observation("codex_cli")
    assert usage["reasoning_effort"] == "high" and usage["exit_code"] == 0
    with pytest.raises(ValueError, match="reasoning_effort"):
        provider.generate("prompt", reasoning_effort="high; injected")
    assert len(observed) == 1


@pytest.mark.parametrize("exit_code", [0, 1])
def test_runner_keeps_usage_and_refuses_partial_cli_failure(tmp_path, monkeypatch, capsys, exit_code):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
                        SimpleNamespace(provider="codex_cli", model_name="pinned", reasoning_effort="high", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())

    def generate(prompt, **kwargs):
        assert kwargs["allow_local_fallback"] is False and kwargs["allow_cross_provider_fallback"] is False
        assert "grok_stream_progress" not in kwargs and "grok_progress_callback" not in kwargs
        assert kwargs["reasoning_effort"] == "high" and kwargs["model_name"] == "pinned"
        set_last_cli_observation("codex_cli", {"exit_code": exit_code, "prompt_tokens": 15, "completion_tokens": 4})
        return "partial or complete output"

    monkeypatch.setattr(llm_router, "generate_text", generate)
    if exit_code:
        with pytest.raises(RuntimeError, match="observed successful exit"):
            runner.run(prompt="repair", provider="codex_cli", model="pinned", timeout=30, max_output_tokens=512)
    else:
        output, receipt = runner.run(prompt="repair", provider="codex_cli", model="pinned", timeout=30, max_output_tokens=512)
        assert output and receipt["usage"]["prompt_tokens"] == 15
    printed = capsys.readouterr().out
    assert '"prompt_tokens": 15' in printed and '"completion_tokens": 4' in printed
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "nonsecret-test-marker")
    with pytest.raises(ValueError, match="database authority"):
        runner.run(prompt="repair", provider="codex_cli", model="pinned", timeout=30, max_output_tokens=512)


def test_codex_timeout_retains_partial_thread_and_removes_temporary_output(monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    monkeypatch.setattr(llm_router.shutil, 'which', lambda name: '/test/codex')
    outputs = []
    def execute(command, **kwargs):
        outputs.append(Path(command[command.index('--output-last-message') + 1]))
        raise subprocess.TimeoutExpired(command, 1,
            output=b'{"type":"thread.started","thread_id":"exact-thread-123456"}\n', stderr=b'')
    monkeypatch.setattr(llm_router.subprocess, 'run', execute)
    with pytest.raises(subprocess.TimeoutExpired):
        llm_router._get_codex_cli_provider().generate('repair', model_name='pinned', timeout=1)
    observation = get_last_cli_observation('codex_cli')
    assert observation['thread_id'] == 'exact-thread-123456'
    assert observation['timed_out'] is True
    assert 'exit_code' not in observation and 'prompt_tokens' not in observation
    assert not outputs[0].exists()


def test_native_rollout_usage_binds_cwd_and_uses_latest_cumulative_totals(tmp_path):
    import json
    from ipfs_accelerate_py.agent_supervisor.runtime.codex_usage_receipt import recover_codex_usage
    root = tmp_path / 'codex'
    folder = root / 'sessions/2026/09/29'
    folder.mkdir(parents=True)
    workspace = tmp_path / 'work'
    workspace.mkdir()
    thread = 'exact-thread-123456'
    path = folder / f'rollout-{thread}.jsonl'
    rows = [
        {'type': 'session_meta', 'payload': {'id': thread, 'cwd': str(workspace)}},
        {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {'total_token_usage': {'input_tokens': 20, 'cached_input_tokens': 10, 'output_tokens': 2, 'total_tokens': 22}}}},
        {'type': 'event_msg', 'payload': {'type': 'token_count', 'info': {'total_token_usage': {'input_tokens': 50, 'cached_input_tokens': 30, 'output_tokens': 5, 'total_tokens': 55}}}},
    ]
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n{incomplete')
    receipt = recover_codex_usage(home=root, thread_id=thread, workspace=workspace)
    assert receipt['usage']['total_tokens'] == 55
    assert receipt['usage']['input_tokens'] == 50
    assert receipt['usage']['cached_input_tokens'] == 30
    assert receipt['observed_token_count_records'] == 2
    assert receipt['task_complete_observed'] is False and receipt['billing_total_verified'] is False
    assert recover_codex_usage(home=root, thread_id=thread, workspace=tmp_path) is None
    assert recover_codex_usage(home=root, thread_id='../' + thread, workspace=workspace) is None


@pytest.fixture
def boundary_fixture(tmp_path, monkeypatch):
    """Unit fixture for deployment metadata; no Docker authority is minted."""
    import hashlib
    import json
    import os
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary

    artifact = tmp_path / 'deployer' / 'boundary.json'
    artifact.parent.mkdir()
    worktrees = tmp_path / 'allocated'
    worktrees.mkdir(mode=0o755)
    workspace = worktrees / 'task'
    workspace.mkdir(mode=0o777)
    private_parent = tmp_path / 'owner-only'
    private_parent.mkdir(mode=0o755)
    private = private_parent / 'private'
    private.mkdir(mode=0o700)
    actual_stat = Path.stat
    actual_read_text = Path.read_text
    uid_overrides = {str(path): 0 for path in (artifact, *artifact.parents)}
    uid_overrides.update({str(path): 1000 for path in (worktrees, private_parent, private)})
    mode_overrides = {str(path): actual_stat(path).st_mode & ~0o022 for path in artifact.parents}
    capabilities = {'CapEff': 0, 'CapPrm': 0, 'CapAmb': 0, 'NoNewPrivs': 1}

    def observed_stat(path, *args, **kwargs):
        result = actual_stat(path, *args, **kwargs)
        fields = list(result)
        fields[4] = uid_overrides.get(str(path), 1000)
        fields[0] = mode_overrides.get(str(path), fields[0])
        return os.stat_result(fields)

    def observed_access(path, mode):
        try:
            bits = observed_stat(Path(path)).st_mode
        except OSError:
            return False
        # Model an unrelated UID/GID: only the other permission bits apply.
        return all(not mode & requested or bits & permitted for requested, permitted in (
            (os.R_OK, 0o004), (os.W_OK, 0o002), (os.X_OK, 0o001),
        ))

    def read_text(path, *args, **kwargs):
        if str(path) == '/proc/self/status':
            return '\n'.join(f'{key}:\t{value}' if key == 'NoNewPrivs' else f'{key}:\t{value:016x}'
                             for key, value in capabilities.items())
        return actual_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', observed_stat)
    monkeypatch.setattr(Path, 'read_text', read_text)
    monkeypatch.setattr(boundary.os, 'getuid', lambda: 1001)
    monkeypatch.setattr(boundary.os, 'geteuid', lambda: 1001)
    monkeypatch.setattr(boundary.os, 'access', observed_access)
    payload = {
        'schema': 'supervisor-container-worker-boundary@1',
        'container_id': 'a' * 64, 'image_id': 'sha256:' + 'b' * 64,
        'owner_uid': 1000, 'worker_uid': 1001,
        'namespaces': {key: os.readlink('/proc/self/ns/' + key) for key in ('pid', 'mnt', 'net')},
        'allowed_worktree_roots': [str(worktrees)], 'owner_private_paths': [str(private)],
    }

    def verify():
        raw = json.dumps(payload).encode()
        artifact.write_bytes(raw)
        artifact.chmod(0o644)
        return boundary.verify_container_worker_boundary(
            artifact=artifact, expected_sha256=hashlib.sha256(raw).hexdigest(), workspace=workspace,
        )

    return SimpleNamespace(verify=verify, payload=payload, artifact=artifact,
                           private=private, private_parent=private_parent,
                           worktrees=worktrees, workspace=workspace,
                           uid_overrides=uid_overrides, mode_overrides=mode_overrides,
                           capabilities=capabilities)


@pytest.mark.parametrize('mutation', [
    'readable_nonexecutable_private_file', 'missing_private', 'symlink_private',
    'wrong_private_owner', 'worker_writable_private_parent', 'symlink_worktree_root',
    'worker_writable_worktree_root', 'worker_writable_worktree_parent',
    'namespace_mismatch', 'same_owner_and_worker', 'effective_capability',
    'permitted_capability', 'ambient_capability',
    'privilege_escalation_allowed',
])
def test_container_boundary_refuses_authority_leaks(boundary_fixture, mutation):
    f = boundary_fixture
    if mutation == 'readable_nonexecutable_private_file':
        secret = f.private_parent / 'readable-key'
        secret.write_text('synthetic-key-only')
        secret.chmod(0o644)
        f.payload['owner_private_paths'] = [str(secret)]
    elif mutation == 'missing_private':
        f.payload['owner_private_paths'] = [str(f.private_parent / 'missing')]
    elif mutation == 'symlink_private':
        alias = f.private_parent / 'alias'
        alias.symlink_to(f.private)
        f.payload['owner_private_paths'] = [str(alias)]
    elif mutation == 'wrong_private_owner':
        f.uid_overrides[str(f.private)] = 1002
    elif mutation == 'worker_writable_private_parent':
        f.private_parent.chmod(0o777)
    elif mutation == 'symlink_worktree_root':
        alias = f.worktrees.parent / 'alias'
        alias.symlink_to(f.worktrees)
        f.payload['allowed_worktree_roots'] = [str(alias)]
    elif mutation == 'worker_writable_worktree_root':
        f.worktrees.chmod(0o777)
    elif mutation == 'worker_writable_worktree_parent':
        f.mode_overrides[str(f.worktrees.parent)] |= 0o002
    elif mutation == 'namespace_mismatch':
        f.payload['namespaces']['mnt'] = 'mnt:[foreign]'
    elif mutation == 'same_owner_and_worker':
        f.payload['owner_uid'] = 1001
    elif mutation == 'privilege_escalation_allowed':
        f.capabilities['NoNewPrivs'] = 0
    else:
        key = {'effective_capability': 'CapEff', 'permitted_capability': 'CapPrm',
               'ambient_capability': 'CapAmb'}[mutation]
        f.capabilities[key] = 1
    with pytest.raises((ValueError, OSError)):
        f.verify()


def test_container_boundary_unit_fixture_preserves_narrow_valid_scope(boundary_fixture):
    observed = boundary_fixture.verify()
    assert observed['worker_uid'] == 1001 and observed['owner_uid'] == 1000
    assert observed['completion_authority'] is False


@pytest.fixture
def grok_progress_workspace(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["grok_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **_:
        SimpleNamespace(provider="grok_cli", model_name="grok-4.7", catalog_revision="authored"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *_, **__: object())
    # Authored dispatch fixture; it mints no container or execution authority.
    monkeypatch.setattr(container_worker_boundary, "verify_container_worker_boundary", lambda **_: {"fixture": True})
    return llm_router


def _grok_progress_snapshot(usage_events=0, *, tool_event_census=False, tool_event_schema="native-grok-tool-events@1"):
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from ipfs_accelerate_py.cli_runtime.grok_stream_usage import MAX_COUNT, USAGE_FIELDS
    import json
    parser = GrokStreamParser(tool_event_census=tool_event_census, tool_event_schema=tool_event_schema)
    frame = json.dumps({"type": "usage", "messageId": "private-id", "signature": "private-signature",
                        "stopReason": "tool_use", "usage": dict.fromkeys(USAGE_FIELDS, MAX_COUNT)}).encode() + b"\n"
    for _ in range(usage_events):
        parser.feed(frame, elapsed_seconds=1)
    return parser.snapshot()


def _run_grok_progress(*, purpose="coding"):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    return runner.run(prompt="authored instruction", provider="grok_cli", model="grok-4.7",
        timeout=30, max_output_tokens=512, purpose=purpose,
        **({"container_boundary": Path("/authored/boundary"), "container_boundary_sha256": "a" * 64}
           if purpose == "coding" else {}))


@pytest.mark.parametrize("timeout", [False, True])
@pytest.mark.parametrize("usage_events", [0, 128])
@pytest.mark.parametrize("census_revision", [None, 1, 2])
def test_grok_progress_is_prefixed_advisory_and_final_metadata_survives_tls(
        grok_progress_workspace, monkeypatch, capsys, timeout, usage_events, census_revision):
    import json
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.agent_supervisor.runtime.provider_capacity_custody import execution_observation
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation, get_last_cli_observation

    snapshot = _grok_progress_snapshot(usage_events, tool_event_census=census_revision is not None,
        tool_event_schema=f"native-grok-tool-events@{census_revision or 1}")
    def generate(prompt, **kwargs):
        assert kwargs["grok_stream_progress"] is True
        assert kwargs["grok_tool_event_census"] is True
        assert kwargs["grok_tool_event_schema"] == "native-grok-tool-events@2"
        kwargs["grok_progress_callback"](snapshot)
        metadata = {"native_grok_progress": snapshot}
        metadata.update({"timed_out": True} if timeout else {"exit_code": 0})
        # Exercise the real sanitizer/TLS path: nested progress must remain a
        # validated dictionary rather than a truncated string.
        set_last_cli_observation("grok_cli", metadata)
        assert get_last_cli_observation("grok_cli")["native_grok_progress"] == snapshot
        if timeout:
            raise subprocess.TimeoutExpired(["authored-provider"], 30)
        return "authored completed response"
    monkeypatch.setattr(grok_progress_workspace, "generate_text", generate)
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_grok_progress()
    else:
        output, _ = _run_grok_progress()
        assert output == "authored completed response"
    lines = capsys.readouterr().out.splitlines()
    advisory, final_line = lines
    assert advisory.startswith(runner.GROK_PROGRESS_PREFIX)
    with pytest.raises(json.JSONDecodeError):
        json.loads(advisory)
    progress = json.loads(advisory[len(runner.GROK_PROGRESS_PREFIX):])
    final = json.loads(final_line)
    assert progress["invocation_id"] == final["invocation_id"]
    assert progress["schema"] == "router-implementation-progress@1" and progress["purpose"] == "coding"
    assert progress["observation_only"] is True and progress["provider_dispatch_observed"] is None
    assert all(progress[key] is False for key in (
        "completion_authority", "settlement_authority", "automatic_retry_admitted"))
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import grok_stream_progress_summary
    assert final["native_grok_progress"] == snapshot
    assert progress["native_grok_progress"] == grok_stream_progress_summary(snapshot)
    assert "usage_observations" not in progress["native_grok_progress"]
    assert "tool_event_observations" not in progress["native_grok_progress"]
    assert len(advisory.encode()) < 8192
    assert final["status"] == ("failed" if timeout else "provider_returned")
    assert final["completion_authority"] is False and final["native_rollout_usage"] is None
    assert not execution_observation(advisory)["task_execution_observed"]


def test_terminal_rejection_survives_actual_runner_receipt_without_promoting_usage(
        grok_progress_workspace, monkeypatch, capsys):
    import json
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import GrokStreamParser
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    parser = GrokStreamParser()
    parser.feed(b'{"type":"max_turns_reached"}\n'
                b'{"type":"end","stopReason":"max_turn_requests","sessionId":null}\n')
    snapshot = parser.snapshot()
    def generate(prompt, **kwargs):
        set_last_cli_observation("grok_cli", {"exit_code": 1, "native_grok_progress": snapshot})
        raise RuntimeError("PRIVATE provider failure")
    monkeypatch.setattr(grok_progress_workspace, "generate_text", generate)
    with pytest.raises(RuntimeError):
        _run_grok_progress()
    final = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert final["native_grok_progress"] == snapshot
    assert final["native_grok_progress"]["terminal_rejection"]["reason"] == "invalid_end_field_shape"
    assert final["status"] == "failed"
    assert final["native_rollout_usage"] is None
    assert final["completion_authority"] is False
    assert "PRIVATE" not in json.dumps(final)


def test_grok_planning_does_not_enable_or_export_coding_progress(grok_progress_workspace, monkeypatch, capsys):
    import json
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation

    def generate(prompt, **kwargs):
        assert "grok_stream_progress" not in kwargs and "grok_progress_callback" not in kwargs
        assert "grok_tool_event_census" not in kwargs
        assert "grok_tool_event_schema" not in kwargs
        set_last_cli_observation("grok_cli", {"exit_code": 0, "native_grok_progress": _grok_progress_snapshot()})
        return "authored planning response"
    monkeypatch.setattr(grok_progress_workspace, "generate_text", generate)
    _, receipt = _run_grok_progress(purpose="planning")
    assert "native_grok_progress" not in receipt
    assert json.loads(capsys.readouterr().out)["schema"] == "router-implementation-invocation@1"


@pytest.mark.parametrize("malformation", ["extra_text", "authority", "foreign_schema", "not_mapping"])
def test_invalid_grok_callback_payload_cannot_escape_or_change_success(
        grok_progress_workspace, monkeypatch, capsys, malformation):
    import json
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation

    invalid = _grok_progress_snapshot()
    if malformation == "extra_text": invalid["text"] = "PRIVATE-PROGRESS-CANARY"
    elif malformation == "authority": invalid["completion_authority"] = True
    elif malformation == "foreign_schema": invalid["schema"] = "PRIVATE-PROGRESS-CANARY"
    else: invalid = "PRIVATE-PROGRESS-CANARY"
    def generate(prompt, **kwargs):
        kwargs["grok_progress_callback"](invalid)
        set_last_cli_observation("grok_cli", {"exit_code": 0})
        return "authored completed response"
    monkeypatch.setattr(grok_progress_workspace, "generate_text", generate)
    output, receipt = _run_grok_progress()
    assert output == "authored completed response" and receipt["status"] == "provider_returned"
    printed = capsys.readouterr().out
    assert "PRIVATE" not in printed
    assert json.loads(printed)["schema"] == "router-implementation-invocation@1"


def test_progress_stdout_failure_cannot_replace_provider_outcome(grok_progress_workspace, monkeypatch, capsys):
    import builtins
    import json
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation

    printed = builtins.print
    attempts = []
    def fail_advisory(value, **kwargs):
        if isinstance(value, str) and value.startswith(runner.GROK_PROGRESS_PREFIX):
            attempts.append(True)
            raise OSError("PRIVATE-SINK-CANARY")
        printed(value, **kwargs)
    monkeypatch.setattr(runner, "print", fail_advisory, raising=False)
    def generate(prompt, **kwargs):
        kwargs["grok_progress_callback"](_grok_progress_snapshot())
        set_last_cli_observation("grok_cli", {"exit_code": 0})
        return "authored completed response"
    monkeypatch.setattr(grok_progress_workspace, "generate_text", generate)
    _, receipt = _run_grok_progress()
    assert attempts == [True] and receipt["status"] == "provider_returned"
    assert json.loads(capsys.readouterr().out)["status"] == "provider_returned"
