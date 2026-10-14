"""Native repository configuration, refusal, and owned lifecycle boundaries."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy


@pytest.fixture
def binding(tmp_path, monkeypatch):
    # Bind a fixed authored server location without waiting for SDK/server imports.
    source = tmp_path / "implementation"
    source.mkdir()
    (source / "repository_source_mcp.py").write_text("# authored fixed server fixture\n")
    monkeypatch.setattr(policy, "__file__", str(source / "repository_source_policy.py"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return policy.create_binding(workspace=workspace, requested_policy=policy.POLICIES[2])


def test_creation_has_no_files_and_closed_authority(binding):
    assert not Path(binding["invocation_dir"]).exists()
    assert binding["config"]["representation"] == "compact"
    assert all(binding[key] is False for key in policy._FALSE_FIELDS)
    guidance = policy.validate_binding(binding, workspace=Path(binding["workspace"]))
    assert "repository_read" in guidance and "repository_replace_symbol" in guidance
    assert binding["instruction_sha256"] == hashlib.sha256(guidance.encode()).hexdigest()


@pytest.mark.parametrize("policy_name,representation", [(policy.POLICIES[1], "original"),
                                                       (policy.POLICIES[2], "compact")])
def test_exact_native_table_and_cleanup(binding, policy_name, representation):
    binding = policy.create_binding(workspace=Path(binding["workspace"]), requested_policy=policy_name)
    identity = policy.create_runtime_store(binding)
    try:
        config = json.loads(Path(binding["config_path"]).read_bytes())
        assert config["representation"] == representation
        assert Path(binding["invocation_dir"]).stat().st_mode & 0o777 == 0o700
        assert Path(binding["config_path"]).stat().st_mode & 0o777 == 0o600
        argv = policy.native_config_arguments(binding)
        assert argv[::2] == ["-c", "-c"]
        settings = tomllib.loads(argv[1])["mcp_servers"]["repository_source"]
        assert settings["command"] == binding["python_executable"]
        assert settings["args"] == [binding["helper_script"], "--config", binding["config_path"],
                                    "--config-sha256", binding["config_sha256"]]
        assert settings["enabled_tools"] == list(policy.TOOLS)
        assert settings["env_vars"] == ["PYTHONPATH"]
        assert settings["startup_timeout_sec"] == settings["tool_timeout_sec"] == 60
        assert argv[3] == "mcp_optional_startup_grace_ms=0"
    finally:
        usage = policy.finish_runtime_store(binding, expected_identity=identity)
    assert usage["store_cleanup_complete"] is True
    assert not Path(binding["invocation_dir"]).exists()


@pytest.mark.parametrize("field,value", [
    ("helper_script", "/tmp/foreign.py"), ("python_executable", "/tmp/../python"),
    ("invocation_dir", "/tmp/foreign"), ("config_path", "/tmp/foreign/config.json"),
    ("state_dir", "/tmp/foreign/state"), ("audit_store", "/tmp/foreign/audit"),
    ("instruction_sha256", "0" * 64), ("config_sha256", "0" * 64),
    ("instruction_bytes", True), ("completion_authority", True),
    ("requested_policy", "foreign"), ("extra", "injection"),
])
def test_modified_bindings_are_refused(binding, field, value):
    altered = copy.deepcopy(binding)
    altered[field] = value
    with pytest.raises(ValueError):
        policy.validate_binding(altered)


@pytest.mark.parametrize("change", ["script", "config", "config_symlink"])
def test_changed_runtime_bytes_refused_before_native_command(binding, tmp_path, change):
    identity = policy.create_runtime_store(binding)
    try:
        if change == "script":
            Path(binding["helper_script"]).write_text("# changed\n")
        elif change == "config":
            Path(binding["config_path"]).write_text("{}")
        else:
            original = Path(binding["config_path"]).read_bytes()
            Path(binding["config_path"]).unlink()
            foreign = tmp_path / "foreign.json"
            foreign.write_bytes(original)
            Path(binding["config_path"]).symlink_to(foreign)
        with pytest.raises((ValueError, OSError)):
            policy.native_config_arguments(binding)
    finally:
        result = policy.finish_runtime_store(binding, expected_identity=identity)
    assert result["store_cleanup_complete"]


def test_cleanup_unlinks_symlinks_without_following_and_handles_sealed_state(binding, tmp_path):
    identity = policy.create_runtime_store(binding)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    protected = foreign / "keep"
    protected.write_text("unchanged")
    state = Path(binding["state_dir"])
    (state / "link").symlink_to(foreign, target_is_directory=True)
    package = state / "package"
    package.mkdir()
    (package / "blob").write_text("owned")
    package.chmod(0o555)
    result = policy.finish_runtime_store(binding, expected_identity=identity)
    assert result["store_cleanup_complete"]
    assert protected.read_text() == "unchanged"


def test_replaced_invocation_directory_is_not_removed(binding):
    identity = policy.create_runtime_store(binding)
    directory = Path(binding["invocation_dir"])
    moved = directory.with_name(directory.name + "-moved")
    directory.rename(moved)
    directory.mkdir(mode=0o700)
    (directory / "keep").write_text("foreign directory")
    try:
        result = policy.finish_runtime_store(binding, expected_identity=identity)
        assert not result["store_cleanup_complete"]
        assert (directory / "keep").read_text() == "foreign directory"
    finally:
        (directory / "keep").unlink()
        directory.rmdir()
        moved.rename(directory)
        assert policy.finish_runtime_store(binding, expected_identity=identity)["store_cleanup_complete"]


@pytest.mark.parametrize("kwargs", [dict(purpose="planning"), dict(provider="grok_cli"),
    dict(source_view_policy="source-view-raw@1"), dict(census_only=True)])
def test_incompatible_route_refused(kwargs):
    args = dict(purpose="coding", provider="codex_cli")
    args.update(kwargs)
    with pytest.raises(ValueError):
        policy.validate_policy(policy.POLICIES[2], **args)


@pytest.mark.parametrize("value", [None, True, {}, "unknown"])
def test_unknown_policy_refused(value):
    with pytest.raises(ValueError):
        policy.validate_policy(value, purpose="coding", provider="codex_cli")


def test_builder_refuses_unmaterialized_and_preserves_default(binding):
    from ipfs_accelerate_py import llm_router
    assert not any("repository_source" in arg for arg in llm_router.build_codex_cli_command())
    with pytest.raises((ValueError, OSError)):
        llm_router.build_codex_cli_command(codex_repository_source_binding=binding)
    identity = policy.create_runtime_store(binding)
    try:
        command = llm_router.build_codex_cli_command(codex_repository_source_binding=binding)
        assert command[-1] == "-"
        assert "mcp_optional_startup_grace_ms=0" in command
        assert sum(arg.startswith("mcp_servers.repository_source=") for arg in command) == 1
    finally:
        policy.finish_runtime_store(binding, expected_identity=identity)


def test_native_provider_forwards_config_and_observes_requested_configuration(binding, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
    from types import SimpleNamespace
    monkeypatch.chdir(binding["workspace"])
    monkeypatch.setattr(llm_router.shutil, "which", lambda name: "/test/codex")
    seen = []
    def execute(command, **kwargs):
        seen.append(command)
        Path(command[command.index("--output-last-message") + 1]).write_text("done")
        return SimpleNamespace(returncode=0, stdout='{"type":"turn.completed"}', stderr="")
    monkeypatch.setattr(llm_router, "_run_codex_repository_command", execute)
    identity = policy.create_runtime_store(binding)
    try:
        assert llm_router._get_codex_cli_provider().generate("literal task", codex_repository_source_binding=binding) == "done"
        assert "mcp_optional_startup_grace_ms=0" in seen[0]
        observation = get_last_cli_observation("codex_cli")
        assert observation["codex_repository_source_configured"] is True
        assert observation["codex_repository_source_config_sha256"] == binding["config_sha256"]
        assert "compact_reads" not in observation
    finally:
        policy.finish_runtime_store(binding, expected_identity=identity)


@pytest.mark.parametrize("timed_out", [False, True])
def test_owned_native_process_group_stops_stdio_descendants(tmp_path, timed_out):
    from ipfs_accelerate_py import llm_router
    import sys
    import time
    script = tmp_path / "native.py"
    child_pid = tmp_path / "child.pid"
    script.write_text('''
import os, pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
print('{"type":"thread.started","thread_id":"owned-test-thread"}', flush=True)
if sys.argv[2] == "timeout": time.sleep(60)
''')
    arguments = [sys.executable, str(script), str(child_pid), "timeout" if timed_out else "success"]
    if timed_out:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            llm_router._run_codex_repository_command(arguments, input="literal", text=True,
                capture_output=True, check=False, timeout=0.5)
        assert "owned-test-thread" in str(error.value.output)
    else:
        result = llm_router._run_codex_repository_command(arguments, input="literal", text=True,
            capture_output=True, check=False, timeout=5)
        assert result.returncode == 0 and "owned-test-thread" in result.stdout
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        os.kill(pid, 9)
        pytest.fail("owned native child remained live after provider returned")
