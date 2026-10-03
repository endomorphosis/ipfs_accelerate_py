"""Transport controls use inert synthetic wheels; no package installation."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs


def _wheel(tmp_path, arch="aarch64", *, package="torch", version="2.13.0+cpu", tag=None,
           duplicate=False, oversized=False):
    path = tmp_path / f"torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_{arch}.whl"
    prefix = "torch-2.13.0+cpu.dist-info/"
    with zipfile.ZipFile(path, "w") as wheel:
        body = f"Name: {package}\nVersion: {version}\n".encode()
        wheel.writestr(prefix + "METADATA", body + (b"x" * 65536 if oversized else b""))
        wheel.writestr(prefix + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: "
            + (tag or f"cp312-cp312-manylinux_2_28_{arch}") + "\n")
        wheel.writestr("torch/packaging-fixture.txt", "inert test bytes, never installed")
        if duplicate:
            with pytest.warns(UserWarning, match="Duplicate name"):
                wheel.writestr(prefix + "METADATA", body)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _build(tmp_path, *, selected=True, wheel=None, pin=None):
    snapshot = tmp_path / ("1" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"packaging_fixture":true}')
    return deployment.build_runtime_archive(output=tmp_path / "bundle", model_snapshot=snapshot,
        torch_cpu_wheel=wheel if selected else None,
        torch_cpu_wheel_sha256=pin if selected else None, **_inputs(tmp_path))


@pytest.mark.parametrize("arch", ["aarch64", "x86_64"])
def test_selected_wheel_exact_bytes_and_closed_binding_roundtrip(tmp_path, arch):
    wheel, pin = _wheel(tmp_path, arch)
    result = _build(tmp_path, wheel=wheel, pin=pin)
    binding = result["torch_cpu_wheel"]
    assert binding["sha256"] == pin and binding["bytes"] == wheel.stat().st_size
    assert deployment.verify_torch_cpu_wheel_archive(tmp_path / "bundle/runtime.tar.gz", result) == binding
    with tarfile.open(tmp_path / "bundle/runtime.tar.gz") as archive:
        member = archive.getmember(binding["path"])
        assert member.isfile() and member.mode == 0o644
        assert archive.extractfile(member).read() == wheel.read_bytes()


@pytest.mark.parametrize("missing", ["path", "pin"])
def test_selection_requires_path_and_independent_pin_together(tmp_path, missing):
    wheel, pin = _wheel(tmp_path)
    with pytest.raises(ValueError, match="paired"):
        _build(tmp_path, wheel=None if missing == "path" else wheel, pin=None if missing == "pin" else pin)
    assert not (tmp_path / "bundle").exists()


def test_wheel_does_not_activate_a_new_model_runtime(tmp_path):
    wheel, pin = _wheel(tmp_path)
    with pytest.raises(ValueError, match="already selected"):
        deployment.build_runtime_archive(output=tmp_path / "bundle", torch_cpu_wheel=wheel,
            torch_cpu_wheel_sha256=pin, **_inputs(tmp_path))


@pytest.mark.parametrize("pin", ["0" * 64, "A" * 64, "not-a-pin", 123])
def test_wrong_or_malformed_independent_pin_refused(tmp_path, pin):
    wheel, _ = _wheel(tmp_path)
    with pytest.raises(ValueError):
        deployment._torch_cpu_wheel_assets(wheel, pin)


@pytest.mark.parametrize("name", ["torch-2.13.0+cu128-cp312-cp312-manylinux_2_28_aarch64.whl",
    "torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl",
    "torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl", "unselected.whl"])
def test_unsupported_version_runtime_or_wheel_filename_refused(tmp_path, name):
    wheel, pin = _wheel(tmp_path)
    other = tmp_path / name
    wheel.rename(other)
    with pytest.raises(ValueError, match="filename"):
        deployment._torch_cpu_wheel_assets(other, pin)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "oversize", "missing"])
def test_bounded_regular_independent_wheel_required(tmp_path, monkeypatch, kind):
    wheel, pin = _wheel(tmp_path)
    if kind == "symlink":
        original = tmp_path / "original"
        wheel.rename(original)
        wheel.symlink_to(original)
    elif kind == "hardlink":
        os.link(wheel, tmp_path / "another-name")
    elif kind == "oversize":
        monkeypatch.setattr(deployment, "TORCH_CPU_WHEEL_MAX_BYTES", wheel.stat().st_size - 1)
    else:
        wheel.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        deployment._torch_cpu_wheel_assets(wheel, pin)


@pytest.mark.parametrize("damage", ["package", "version", "tag", "duplicate", "oversized", "not-zip"])
def test_malformed_wheel_metadata_refused_even_with_matching_digest(tmp_path, damage):
    kwargs = dict(package="other") if damage == "package" else dict(version="2.13.0+cu128") if damage == "version" else dict(tag="cp311-cp311-manylinux_2_28_aarch64") if damage == "tag" else {damage: True} if damage in ("duplicate", "oversized") else {}
    wheel, pin = _wheel(tmp_path, **kwargs)
    if damage == "not-zip":
        wheel.write_bytes(b"not a wheel")
        pin = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        deployment._torch_cpu_wheel_assets(wheel, pin)


@pytest.mark.parametrize("damage", ["missing-binding", "missing-member", "wrong-size", "wrong-digest",
    "duplicate-member", "extra-member", "bad-path", "bad-requirement", "bool-size", "extra-key"])
def test_unbound_or_malformed_selection_refused_before_container_calls(tmp_path, damage):
    wheel, pin = _wheel(tmp_path)
    manifest = _build(tmp_path, wheel=wheel, pin=pin)
    binding = manifest["torch_cpu_wheel"]
    member = next(row for row in manifest["files"] if row["path"] == binding["path"])
    if damage == "missing-binding": del manifest["torch_cpu_wheel"]
    elif damage == "missing-member": manifest["files"].remove(member)
    elif damage == "wrong-size": member["bytes"] += 1
    elif damage == "wrong-digest": member["sha256"] = "0" * 64
    elif damage == "duplicate-member": manifest["files"].append(dict(member))
    elif damage == "extra-member": manifest["files"].append(dict(member, path=deployment.TORCH_CPU_WHEEL_PREFIX + "extra.whl"))
    elif damage == "bad-path": binding["path"] = "../outside.whl"
    elif damage == "bad-requirement": binding["requirement"] = "torch==2.12.0+cpu"
    elif damage == "bool-size": binding["bytes"] = True
    else: binding["extra"] = 1
    (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        asyncio.run(deployment.deploy_supervisor(None, archive_dir=tmp_path / "bundle", output=tmp_path / "deploy"))


@pytest.mark.parametrize("damage", ["bytes", "missing", "symlink", "duplicate"])
def test_tampered_tar_wheel_refused_with_recomputed_outer_digest(tmp_path, damage):
    wheel, pin = _wheel(tmp_path)
    manifest = _build(tmp_path, wheel=wheel, pin=pin)
    archive_path = tmp_path / "bundle/runtime.tar.gz"
    with tarfile.open(archive_path) as archive:
        members = [(member, archive.extractfile(member).read()) for member in archive]
    with tarfile.open(archive_path, "w:gz") as archive:
        for member, raw in members:
            if member.name == manifest["torch_cpu_wheel"]["path"]:
                if damage == "missing": continue
                if damage == "bytes": raw = bytes([raw[0] ^ 1]) + raw[1:]
                if damage == "symlink": member.type = tarfile.SYMTYPE; member.linkname = "elsewhere"; member.size = 0
                archive.addfile(member, io.BytesIO(raw) if member.isfile() else None)
                if damage != "duplicate": continue
            archive.addfile(member, io.BytesIO(raw))
    manifest["archive_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        asyncio.run(deployment.deploy_supervisor(None, archive_dir=tmp_path / "bundle", output=tmp_path / "deploy"))


def test_exact_extracted_probe_runs_before_third_party_installation(tmp_path, monkeypatch):
    wheel, pin = _wheel(tmp_path, platform.machine())
    _, binding = deployment._torch_cpu_wheel_assets(wheel, pin)
    target = tmp_path / "runtime" / binding["path"]
    target.parent.mkdir(parents=True)
    shutil.copyfile(wheel, target)
    monkeypatch.setattr(deployment, "ROOT", str(tmp_path / "runtime"))
    source = Path(deployment.__file__).parents[3]
    script = deployment._torch_cpu_wheel_probe(binding)
    script += "; import sys; assert not set(('torch','duckdb','numpy','pandas','requests','multiformats')) & set(sys.modules)"
    completed = subprocess.run([sys.executable, "-S", "-P", "-c", script],
        env={"PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}, text=True,
        capture_output=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == binding


@pytest.mark.parametrize("damage", ["changed", "wrong-platform", "wrong-size"])
def test_extracted_wheel_rechecked_before_install(tmp_path, monkeypatch, damage):
    wheel, pin = _wheel(tmp_path, platform.machine())
    size = wheel.stat().st_size
    if damage == "changed": wheel.write_bytes(b"changed")
    elif damage == "wrong-platform": monkeypatch.setattr(platform, "machine", lambda: "other")
    else: size += 1
    with pytest.raises(ValueError): deployment.verify_extracted_torch_cpu_wheel(wheel, pin, size)


@pytest.mark.parametrize("selected,fail_probe", [(False, False), (True, False), (True, True)])
def test_install_selection_timeout_and_no_fallback(tmp_path, selected, fail_probe):
    wheel, pin = _wheel(tmp_path)
    manifest = _build(tmp_path, selected=selected, wheel=wheel, pin=pin)
    class Environment:
        commands = []
        async def upload_file(self, *args): pass
        async def exec(self, command, **kwargs):
            self.commands.append((command, kwargs))
            probe = "verify_extracted_torch_cpu_wheel" in command
            install = "--index-url https://download.pytorch.org/whl/cpu" in command
            return SimpleNamespace(return_code=1 if install or (probe and fail_probe) else 0,
                stdout="{}", stderr="explicit controlled installation failure")
    env = Environment()
    with pytest.raises(RuntimeError, match="torch-cpu-wheel-verify" if fail_probe else "torch-cpu-install"):
        asyncio.run(deployment.deploy_supervisor(env, archive_dir=tmp_path / "bundle", output=tmp_path / "deploy"))
    calls = [(shlex.split(command), options) for command, options in env.commands if "--index-url" in command]
    if fail_probe:
        assert calls == []
    else:
        assert len(calls) == 1 and calls[0][1]["timeout_sec"] == 600
        expected = deployment.ROOT + "/" + manifest["torch_cpu_wheel"]["path"] if selected else deployment.TORCH_CPU_REQUIREMENT
        assert calls[0][0][-1] == expected and "--no-deps" not in calls[0][0]
    assert not any(" -m pip install --no-cache-dir duckdb" in command for command, _ in env.commands)
