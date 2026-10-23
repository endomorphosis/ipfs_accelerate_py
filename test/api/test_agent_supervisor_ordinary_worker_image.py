"""Ordinary successor source pins never widen protected image acceptance.

These controls assemble commands and intercept image inspection.  They never
contact Docker, execute a provider, load credentials or launch a container.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as G


@pytest.fixture
def ordinary_pin(monkeypatch):
    # Test-only source substitution; production has no image override input.
    image = "sha256:" + "c" * 64
    monkeypatch.setattr(G, "_ORDINARY_TASK_TOOLCHAIN_IMAGE_ID", image)
    monkeypatch.setattr(G, "_ordinary_task_toolchain_host_architecture", lambda: "arm64")
    return image


def _grok_inputs(tmp_path: Path, image: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "grok-home"
    home.mkdir()
    prompt = tmp_path / "prompt"
    prompt.write_text("inert command assembly fixture", encoding="utf-8")
    binary = tmp_path / "inert-grok"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o700)
    config = tmp_path / "docker-config"
    config.mkdir()
    return {
        "grok_command": [
            str(binary), "--tools", G._SEALED_GROK_TOOLS,
            "--disallowed-tools", G._SEALED_GROK_DISALLOWED_TOOLS,
            "--no-subagents", "--disable-web-search",
        ],
        "grok_bin": binary,
        "workspace": workspace,
        "prompt_path": prompt,
        "grok_home": home,
        "base_env": {"PATH": "/usr/bin:/bin", "GROK_HOME": str(home)},
        "child_env": {},
        "denied_paths": (),
        "mask_root": tmp_path / "provider-masks",
        "docker_config": config,
        "container_name": "ipfs-accelerate-grok-123-" + "a" * 32,
        "cidfile": tmp_path / "container.cid",
        "docker_bin": "/usr/bin/docker",
        "isolation_image": image,
    }


def _codex_inputs(tmp_path: Path, image: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binary = tmp_path / "codex"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o700)
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    auth.chmod(0o600)
    return {
        "codex_command": [
            str(binary), "exec", "--ignore-user-config", "--ignore-rules",
            "--ephemeral", "-s", "workspace-write", "-m", G.CODEX_QUOTA_FALLBACK_MODEL,
            "-C", str(workspace), "-c", 'model_reasoning_effort="high"', "-",
        ],
        "workspace": workspace,
        "source_auth": auth,
        "child_env": G._codex_task_container_environment(),
        "docker_config": tmp_path / "docker-config",
        "container_name": "ipfs-accelerate-codex-123-" + "a" * 32,
        "cidfile": tmp_path / "container.cid",
        "docker_bin": "/usr/bin/docker",
        "isolation_image": image,
    }


def test_pending_source_pin_refuses_before_docker(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "_ordinary_task_toolchain_host_architecture", lambda: "arm64")
    monkeypatch.setattr(G, "_ORDINARY_TASK_TOOLCHAIN_IMAGE_ID", "PENDING_QUALIFIED_ORDINARY_WORKER_IMAGE_ID")
    monkeypatch.setattr(G.subprocess, "run", lambda *a, **kw: pytest.fail("unqualified image queried Docker"))
    assert G._docker_ordinary_task_toolchain_image_id("/usr/bin/docker", docker_config=tmp_path) == ""


@pytest.mark.parametrize("observed", [
    "same", "wrong_image", "wrong_os", "wrong_architecture", "wrong_label", "extra_field", "wrong_exit",
])
def test_image_inspection_requires_exact_source_pin_platform_and_label(
    monkeypatch, tmp_path, ordinary_pin, observed,
):
    expected = [ordinary_pin, "linux", "arm64", "2026-10-16-ordinary-v1"]
    fields = list(expected)
    mutations = {
        "wrong_image": (0, "sha256:" + "d" * 64),
        "wrong_os": (1, "windows"),
        "wrong_architecture": (2, "amd64"),
        "wrong_label": (3, "2026-08-03-v2"),
    }
    if observed in mutations:
        index, value = mutations[observed]
        fields[index] = value
    if observed == "extra_field":
        fields.append("unexpected")
    calls = []

    def inspect_image(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1 if observed == "wrong_exit" else 0, stdout="|".join(fields) + "\n")

    monkeypatch.setattr(G.subprocess, "run", inspect_image)
    monkeypatch.setenv("DOCKER_HOST", "unix:///foreign.sock")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_AUTHORITY_VALIDATION_CONTAINER_IMAGE", "arbitrary:tag")
    result = G._docker_ordinary_task_toolchain_image_id("/usr/bin/docker", docker_config=tmp_path)
    assert result == (ordinary_pin if observed == "same" else "")
    assert len(calls) == 1
    command, options = calls[0]
    assert command[-1] == ordinary_pin
    assert command[1] == "--host=unix:///var/run/docker.sock"
    assert command[2:4] == ["--config", str(tmp_path)]
    assert "org.ipfs-accelerate.ordinary-worker" in command[-2]
    assert "arbitrary:tag" not in command
    assert options["timeout"] == 10


def test_protected_grok_builder_rejects_successor_before_launchers(tmp_path, ordinary_pin):
    data = _grok_inputs(tmp_path, ordinary_pin)
    with pytest.raises(ValueError, match="pinned task-toolchain"):
        G._docker_grok_command(**data, task_execution=True)
    assert not (tmp_path / "task-launchers").exists()


def test_ordinary_grok_builder_rejects_old_image_before_launchers(tmp_path, ordinary_pin):
    data = _grok_inputs(tmp_path, G._CODEX_TASK_TOOLCHAIN_IMAGE_ID)
    with pytest.raises(ValueError, match="pinned task-toolchain"):
        G._docker_ordinary_grok_command(**data, task_execution=True)
    assert not (tmp_path / "task-launchers").exists()


def test_ordinary_grok_builder_keeps_confinement(tmp_path, ordinary_pin):
    data = _grok_inputs(tmp_path, ordinary_pin)
    command = G._docker_ordinary_grok_command(**data, task_execution=True)
    assert ordinary_pin in command
    assert G._CODEX_TASK_TOOLCHAIN_IMAGE_ID not in command
    assert {"--pull=never", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--network=bridge", "--runtime=runc"} <= set(command)
    assert command[command.index("--tools") + 1] == G._DOCKER_TASK_GROK_TOOLS
    assert command[command.index(ordinary_pin) + 1:][:2] == ["-I", str(G._GROK_CONTAINER_BOOTSTRAP)]


def test_ordinary_grok_builder_cannot_reinterpret_file_only_profile(tmp_path, ordinary_pin):
    with pytest.raises(ValueError, match="task execution profile"):
        G._docker_ordinary_grok_command(**_grok_inputs(tmp_path, ordinary_pin))
    assert not (tmp_path / "task-launchers").exists()


def test_protected_codex_builder_rejects_successor_before_auth(monkeypatch, tmp_path, ordinary_pin):
    monkeypatch.setattr(G, "_validated_codex_auth_path", lambda **kw: pytest.fail("successor reached protected auth projection"))
    with pytest.raises(ValueError, match="exact pinned task-toolchain"):
        G._docker_codex_fallback_command(**_codex_inputs(tmp_path, ordinary_pin))


def test_ordinary_codex_builder_rejects_old_image_before_auth(monkeypatch, tmp_path, ordinary_pin):
    monkeypatch.setattr(G, "_validated_codex_auth_path", lambda **kw: pytest.fail("old image reached successor auth projection"))
    with pytest.raises(ValueError, match="exact pinned task-toolchain"):
        G._docker_ordinary_codex_fallback_command(**_codex_inputs(tmp_path, G.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID))


def test_ordinary_codex_builder_keeps_confinement(tmp_path, ordinary_pin):
    data = _codex_inputs(tmp_path, ordinary_pin)
    command = G._docker_ordinary_codex_fallback_command(**data)
    assert ordinary_pin in command
    assert G.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID not in command
    assert {"--pull=never", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--network=bridge", "--runtime=runc"} <= set(command)
    mounts = [command[index + 1] for index, argument in enumerate(command[:-1]) if argument == "--mount"]
    assert f"type=bind,src={data['workspace']},dst={data['workspace']}" in mounts
    assert "type=bind,src=/usr,dst=/usr,readonly" in mounts
    assert not any("docker.sock" in mount for mount in mounts)


def test_legacy_successor_and_protected_caller_remain_separate():
    legacy = inspect.getsource(G._run_legacy_codex_quota_fallback_in_docker)
    protected = inspect.getsource(G._run_codex_quota_fallback_in_docker)
    assert "_docker_ordinary_task_toolchain_image_id(" in legacy
    assert "_docker_ordinary_codex_fallback_command(" in legacy
    assert "_docker_codex_task_toolchain_image_id(" in protected
    assert "_docker_codex_fallback_command(" in protected
    assert "_docker_ordinary" not in protected
    assert G.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID == "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"
    assert G.AGENT_IMPLEMENTATION_CODEX_IMAGE_LABEL == "2026-08-03-v2"
