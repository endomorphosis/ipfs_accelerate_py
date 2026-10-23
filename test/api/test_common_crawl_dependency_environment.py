"""Exact non-fetch policy survives both sealed command environments."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as grok
from ipfs_accelerate_py.agent_supervisor.runtime.provider_command_environment import (
    ProviderCommandEnvironmentError,
    project_provider_command_environment,
    sealed_provider_command_environment,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon
from ipfs_accelerate_py.agent_supervisor.validation.validation_runtime import (
    ValidationRuntimeError,
    build_validation_environment,
    validation_environment_for_runner,
)

FLAG = "COMMON_CRAWL_AUTO_CLONE_SOURCE"
SECRET = "COMMON_CRAWL_SEARCH_ENGINE_REPO"


def test_explicit_nonfetch_policy_is_closed_and_absence_compatible():
    source = {"PATH": "/usr/bin:/bin", FLAG: "0", SECRET: "synthetic-private-remote", "UNRELATED_SECRET": "not-approved"}
    for child in (project_provider_command_environment(source), build_validation_environment(source)):
        assert child[FLAG] == "0"
        assert SECRET not in child and "UNRELATED_SECRET" not in child
    assert FLAG not in project_provider_command_environment({"PATH": "/usr/bin:/bin"})
    assert FLAG not in build_validation_environment({})


@pytest.mark.parametrize("value", ["1", "true", "false", "", False])
@pytest.mark.parametrize("builder,error", [(project_provider_command_environment, ProviderCommandEnvironmentError), (build_validation_environment, ValidationRuntimeError)])
def test_nonfetch_policy_cannot_enable_clone_or_accept_ambiguous_values(value, builder, error):
    with pytest.raises(error, match="may only disable dependency cloning"):
        builder({"PATH": "/usr/bin:/bin", FLAG: value})


def test_actual_sealed_provider_command_receives_nonfetch_policy(tmp_path):
    code = "import json,os;print(json.dumps({'flag':os.environ.get('COMMON_CRAWL_AUTO_CLONE_SOURCE'),'secret':os.environ.get('COMMON_CRAWL_SEARCH_ENGINE_REPO')}))"
    with sealed_provider_command_environment({"PATH": "/usr/bin:/bin", FLAG: "0", SECRET: "not-approved"}) as contract:
        result = subprocess.run([contract.wrapper_path, "--", sys.executable, "-c", code],
            env={"PATH": "/usr/bin:/bin", FLAG: "1", SECRET: "not-approved"}, text=True, capture_output=True, timeout=15)
        assert contract.sealed and FLAG in contract.environment_names
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"flag": "0", "secret": None}


def test_full_ordinary_docker_builder_carries_nonfetch_policy_without_secret(tmp_path):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    home = tmp_path / "grok-home"; home.mkdir()
    prompt = tmp_path / "prompt"; prompt.write_text("inert synthetic task")
    binary = tmp_path / "inert"; binary.write_text("#!/bin/sh\nexit 0\n"); binary.chmod(0o700)
    config = tmp_path / "config"; config.mkdir()
    source = {"PATH": "/usr/bin:/bin", FLAG: "0", SECRET: "not-approved"}
    child = {"XAI_API_KEY": "synthetic-credential-never-called", FLAG: "1", SECRET: "not-approved"}
    command = grok._docker_grok_command(
        grok_command=[str(binary), "--tools", grok._SEALED_GROK_TOOLS, "--disallowed-tools", grok._SEALED_GROK_DISALLOWED_TOOLS, "--no-subagents", "--disable-web-search"],
        grok_bin=binary, workspace=workspace, prompt_path=prompt, grok_home=home,
        base_env=source, child_env=child, denied_paths=(), mask_root=tmp_path / "masks",
        docker_config=config, container_name="ipfs-accelerate-grok-123-" + "a" * 32,
        cidfile=tmp_path / "cid", docker_bin="/usr/bin/docker",
        isolation_image=grok._CODEX_TASK_TOOLCHAIN_IMAGE_ID, task_execution=True,
    )
    assert child[FLAG] == "0" and SECRET not in child
    assert FLAG in command and SECRET not in command
    wrapper = (tmp_path / "task-launchers/provider-command-env").read_text()
    bootstrap = (tmp_path / "task-launchers/task-bootstrap.py").read_text()
    assert FLAG in wrapper and FLAG in bootstrap
    assert "not-approved" not in wrapper + bootstrap + " ".join(command)
    effective, approved = grok._grok_task_container_environment(base_env=source, child_env=child, workspace=workspace, grok_home=home)
    assert effective[FLAG] == approved[FLAG] == "0"
    assert effective[grok.PROVIDER_COMMAND_ENV_DIGEST_ENV] == grok.provider_command_environment_sha256(approved)


def test_actual_authoritative_validation_child_receives_nonfetch_policy(tmp_path):
    runner = PortalImplementationDaemon._validation_command_runner
    environment = validation_environment_for_runner(build_validation_environment({FLAG: "0", SECRET: "not-approved"}), runner)
    code = "import json,os;print(json.dumps({'flag':os.environ.get('COMMON_CRAWL_AUTO_CLONE_SOURCE'),'secret':os.environ.get('COMMON_CRAWL_SEARCH_ENGINE_REPO')}))"
    command = shlex.join(["python", "-c", code])
    result = runner(spec=SimpleNamespace(command=command, raw_command=command), workspace_path=tmp_path,
                    timeout_seconds=20, environment=environment)
    assert result["returncode"] == 0, result.get("output")
    assert result["validation_python_launcher"]["sealed"] is True
    assert json.loads(result["output"].strip()) == {"flag": "0", "secret": None}
