"""Source-qualified ordinary image selection follows the host, never ambient policy."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as G


@pytest.mark.parametrize("machine,architecture", [
    ("x86_64", "amd64"), ("amd64", "amd64"),
    ("aarch64", "arm64"), ("arm64", "arm64"),
])
def test_actual_host_architecture_selects_only_a_frozen_profile(monkeypatch, machine, architecture):
    monkeypatch.setattr(G.os, "uname", lambda: SimpleNamespace(machine=machine))
    monkeypatch.setenv("DOCKER_DEFAULT_PLATFORM", "linux/foreign")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_AUTHORITY_VALIDATION_CONTAINER_IMAGE", "foreign:tag")
    image, platform, label = G._ordinary_task_toolchain_image_profile()
    assert platform == architecture
    assert G._ordinary_task_toolchain_image_pin() == image
    if architecture == "amd64":
        assert image == "sha256:d361dafa5fd1c89981325f789ce45f918848bfa912ce35dbea486db95e0479cf"
        assert label == "2026-10-21-ordinary-amd64-v1"
    else:
        assert image == "sha256:a91a42c918514203bb6cfc7c38b320dfb68e9dfc0571bf01976c0db993e24e5e"
        assert label == "2026-10-16-ordinary-v1"


def test_unknown_host_refuses_before_any_docker_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(G.os, "uname", lambda: SimpleNamespace(machine="unqualified"))
    monkeypatch.setattr(G.subprocess, "run", lambda *_a, **_kw: pytest.fail("unknown platform probed Docker"))
    assert G._docker_ordinary_task_toolchain_image_id("/usr/bin/docker", docker_config=tmp_path) == ""


@pytest.mark.parametrize("mutation", (None, "image", "os", "architecture", "label"))
def test_amd64_inspection_requires_exact_selected_profile(monkeypatch, tmp_path, mutation):
    monkeypatch.setattr(G, "_ordinary_task_toolchain_host_architecture", lambda: "amd64")
    image, architecture, label = G._ordinary_task_toolchain_image_profile()
    fields = [image, "linux", architecture, label]
    if mutation is not None:
        fields[{"image": 0, "os": 1, "architecture": 2, "label": 3}[mutation]] = "wrong"
    calls = []

    def inspect(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="|".join(fields))

    monkeypatch.setattr(G.subprocess, "run", inspect)
    result = G._docker_ordinary_task_toolchain_image_id("/usr/bin/docker", docker_config=tmp_path)
    assert result == (image if mutation is None else "")
    assert len(calls) == 1 and calls[0][-1] == image
    assert calls[0][1] == "--host=unix:///var/run/docker.sock"


def test_amd64_build_lock_is_separate_and_preserves_the_arm64_lock():
    directory = Path(__file__).resolve().parents[2] / "containers/external-agent/ordinary-task-toolchain"
    path = directory / "amd64/stage_ordinary_worker_context.py"
    spec = importlib.util.spec_from_file_location("amd64_ordinary_staging_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stager = module.load_stager()
    lock = stager.load_lock(directory / "amd64/wheels.lock.json")
    assert len(lock.wheels) == 9
    assert lock.base_image_id == module.BASE_IMAGE_ID
    assert lock.sha256 == module.LOCK_SHA256
    assert all("aarch64" not in wheel.filename for wheel in lock.wheels)
    historical = json.loads((directory / "wheels.lock.json").read_text())
    assert historical["architecture"] == "arm64"
    assert G.AGENT_IMPLEMENTATION_CODEX_IMAGE_ID == "sha256:74c4a6ff67f397f8a10b058851d218896b2f1ee0f2cddf47741219b734de93a6"


def test_unverified_ripgrep_package_cannot_reach_archive_decompression(monkeypatch, tmp_path):
    directory = Path(__file__).resolve().parents[2] / "containers/external-agent/ordinary-task-toolchain"
    spec = importlib.util.spec_from_file_location("amd64_ordinary_ripgrep_test", directory / "amd64/stage_ordinary_worker_context.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / "unverified.deb"
    package.write_bytes(b"not an authenticated Ubuntu package")
    monkeypatch.setattr(module.subprocess, "run", lambda *_a, **_kw: pytest.fail("unverified package reached dpkg-deb"))
    stager = module.load_stager()
    with pytest.raises(stager.BuildInputError, match="frozen_ripgrep_package_mismatch"):
        module.stage_ripgrep(stager, output=tmp_path, package=package)
