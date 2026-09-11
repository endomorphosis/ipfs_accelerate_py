"""Exercise authority forwarding through the actual managed child boundary."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as supervisor
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    launch_supervised_child, wait_for_child_exit,
)


def test_cli_config_loop_real_child_preserves_quack_bindings_and_provider_filter(tmp_path: Path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("IPFS_ACCELERATE_AGENT_STATE_", "IPFS_ACCELERATE_AGENT_QUACK_")) or key in {
            "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON", "IPFS_ACCELERATE_AGENT_TASK_SOURCE_KIND"
        }:
            monkeypatch.delenv(key)
    # This dummy credential must be inherited by the trusted daemon, while
    # remaining absent from public bindings, argv and provider environments.
    credential = "test-only-managed-owner-credential"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", credential)
    args = supervisor.parse_args([
        "--implement", "--todo-path", str(tmp_path / "control.duckdb"),
        "--state-dir", str(tmp_path / "state"),
        "--worktree-root", str(tmp_path / "worktrees"),
        "--task-source-kind", "duckdb", "--authority-mode", "quack",
        "--endpoint-secret-handle", "handle:temporary-quack-test",
        "--quack-endpoint", "quack:127.0.0.1:45123",
        "--state-store-id", "temporary-control-store",
        "--state-store-generation", "7", "--state-schema-revision", "1",
        "--state-failover-policy", "fail_closed",
    ])
    config = supervisor.supervisor_config_from_args(args, repo_root=tmp_path)
    instance = supervisor.PortalImplementationSupervisor(config)
    loop_config = instance.build_supervisor_loop_config()
    child_spec = SupervisorLoop(loop_config)._child_spec("test-forwarding")
    assert child_spec.env == loop_config.spec.launch_env
    assert child_spec.env["IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE"] == "handle:temporary-quack-test"
    assert credential not in json.dumps(dict(child_spec.env))
    assert credential not in " ".join(child_spec.command)
    module = "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    daemon_argv = list(child_spec.command[child_spec.command.index(module) + 1:])
    # Only parse in the real child. Never open a store, claim work, or call a provider.
    probe = """
import json, os, sys
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
args = daemon.parse_args(sys.argv[1:])
program = daemon.database_program_from_daemon_namespace(args)
assert os.environ.get('IPFS_ACCELERATE_AGENT_QUACK_TOKEN')
print(json.dumps(program.to_dict(), sort_keys=True))
"""
    actual_spec = replace(child_spec, command=(sys.executable, "-P", "-c", probe, *daemon_argv))
    child = launch_supervised_child(actual_spec)
    assert wait_for_child_exit(child, poll_interval_seconds=0.02) == 0
    output = child_spec.log_path.read_text()
    assert credential not in output
    assert json.loads(output.strip()) == config.database_program.to_dict()
    provider_env = instance.provider_subprocess_environment({**os.environ, **child_spec.env})
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in provider_env
    assert credential not in json.dumps(provider_env)
