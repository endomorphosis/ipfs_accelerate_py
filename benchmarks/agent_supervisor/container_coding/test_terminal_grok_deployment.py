"""Closed Grok native assets and per-provider worker credential isolation."""
import ast
import asyncio
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok
from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import container_worker_deployment as worker
from benchmarks.agent_supervisor.container_coding.test_terminal_deployment import _inputs


@pytest.fixture
def binary(tmp_path, monkeypatch):
    path = tmp_path / "grok"
    path.write_bytes(b"native-fixture-executable")
    path.chmod(0o755)
    monkeypatch.setattr(grok, "GROK_BYTES", path.stat().st_size)
    monkeypatch.setattr(grok, "GROK_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_selected_native_asset_is_archived_without_credentials(binary, tmp_path):
    result = deployment.build_runtime_archive(output=tmp_path / "bundle", grok_binary=binary, **_inputs(tmp_path))
    assert grok.validate_grok_binding(result, required=True) == grok.grok_binding()
    assert grok.verify_grok_archive(tmp_path / "bundle/runtime.tar.gz", result, required=True)
    assert result["credentials_in_archive"] is False
    assert result["torch_cpu_requirement"] == ""
    assert result["codex_version"] == deployment.CODEX_VERSION
    with tarfile.open(tmp_path / "bundle/runtime.tar.gz") as archive:
        assert archive.getmember(grok.GROK_PATH).mode == 0o755
        assert not any("auth.json" in member.name for member in archive)


@pytest.mark.parametrize("change", ["link", "bytes", "hash", "mode"])
def test_changed_native_binary_rejected_before_archive_creation(binary, tmp_path, change):
    if change == "link":
        link = tmp_path / "linked"
        link.symlink_to(binary)
        binary = link
    elif change == "bytes":
        binary.write_bytes(b"short")
    elif change == "hash":
        binary.write_bytes(b"x" * binary.stat().st_size)
    else:
        binary.chmod(0o644)
    with pytest.raises(ValueError):
        deployment.build_runtime_archive(output=tmp_path / "bundle", grok_binary=binary, **_inputs(tmp_path))
    assert not (tmp_path / "bundle").exists()


def test_grok_rejects_codex_cache_policy_before_archive_mutation(binary, tmp_path):
    with pytest.raises(ValueError, match="Codex setup cache"):
        deployment.build_runtime_archive(output=tmp_path / "bundle", grok_binary=binary,
            setup_cache_policy="source384-native-aarch64-dontneed@2", **_inputs(tmp_path))
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("change", ["missing", "version", "hash", "extra", "inventory", "duplicate"])
def test_manifest_requires_complete_independently_pinned_grok_inventory(binary, tmp_path, change):
    manifest = deployment.build_runtime_archive(output=tmp_path / "bundle", grok_binary=binary, **_inputs(tmp_path))
    if change == "missing":
        del manifest["grok_cli_assets"]
    elif change == "version":
        manifest["grok_cli_assets"]["version"] = "latest"
    elif change == "hash":
        manifest["grok_cli_assets"]["sha256"] = "0" * 64
    elif change == "extra":
        manifest["grok_cli_assets"]["download_url"] = "https://unbound.invalid"
    elif change == "inventory":
        manifest["files"] = [row for row in manifest["files"] if row["path"] != grok.GROK_PATH]
    else:
        manifest["files"].append(next(row for row in manifest["files"] if row["path"] == grok.GROK_PATH))
    with pytest.raises(ValueError):
        grok.validate_grok_binding(manifest, required=True)


@pytest.mark.parametrize("options", [
    {"provider": "other"}, {"provider": "grok_cli"},
    {"provider": "grok_cli", "install_codex": False},
    {"provider": "grok_cli", "install_codex": False, "auth_json": Path("/unused/auth.json"), "setup_cache_selection": {}},
])
def test_invalid_deployment_selection_refused_before_container_calls(tmp_path, options):
    with pytest.raises(ValueError):
        asyncio.run(deployment.deploy_supervisor(None, archive_dir=tmp_path / "absent",
            output=tmp_path / "deployment", **options))
    assert not (tmp_path / "deployment").exists()


@pytest.mark.parametrize("kind", ["empty", "oversize", "link", "directory"])
def test_auth_transport_accepts_only_bounded_regular_file(tmp_path, kind):
    path = tmp_path / "auth.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "link":
        target = tmp_path / "other"
        target.write_text("private")
        path.symlink_to(target)
    else:
        path.write_bytes(b"" if kind == "empty" else b"x" * 65537)
    with pytest.raises(ValueError, match="bounded regular"):
        grok.validate_auth_file(path)


def test_worker_entry_keeps_boundary_and_exact_provider_dispatch():
    tree = ast.parse(worker.WORKER_ENTRY)
    assert "selected_provider" in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "args.provider!=selected_provider" in worker.WORKER_ENTRY
    assert "'--provider',args.provider" in worker.WORKER_ENTRY
    assert "verify_container_worker_boundary" in worker.WORKER_ENTRY
    assert "libc.prctl(38,1,0,0,0)" in worker.WORKER_ENTRY
    assert "os.environ.clear()" in worker.WORKER_ENTRY
    assert "GROK_CODEX_MCPS_ENABLED" in worker.WORKER_ENTRY
    assert "--finite-proof-query-context" in worker.WORKER_ENTRY


def test_exposure_script_rechecks_native_identity_and_root_ownership():
    script = grok.grok_exposure_script()
    ast.parse(script)
    assert grok.GROK_SHA256 in script and grok.GROK_VERSION_OUTPUT in script
    assert "os.chown(target,0,0)" in script and "target.chmod(0o555)" in script
    assert "source.is_symlink()" in script and "platform.machine()!='aarch64'" in script


@pytest.mark.parametrize("output", ["grok 1.0.46 (2765805b9442)",
                                  "grok 1.0.46 (2765805b9442) [stable]"])
def test_exact_pinned_version_accepts_standalone_and_managed_metadata(output):
    assert grok.require_grok_version_output(output + "\n") == output


@pytest.mark.parametrize("output", [None, [], 1, "", "grok 1.0.46", "grok 1.0.45 (2765805b9442)",
    "grok 1.0.46 (000000000000)", "grok 1.0.46 (2765805b9442) [nightly]",
    "grok 1.0.46 (2765805b9442)\nextra output"])
def test_version_normalization_cannot_accept_another_version_build_or_channel(output):
    with pytest.raises(ValueError, match="pinned build"):
        grok.require_grok_version_output(output)


@pytest.mark.parametrize("output,exit_code", [("grok 1.0.46 (2765805b9442)", 0),
    ("grok 1.0.46 (2765805b9442) [stable]", 0),
    ("grok 1.0.46 (2765805b9442)", 1), ("grok 1.0.46 (foreign)", 0)])
def test_rendered_exposure_checks_actual_native_probe_result(binary, tmp_path, monkeypatch, capsys, output, exit_code):
    import os
    import platform
    import shutil
    import subprocess
    root = tmp_path / "runtime"
    source = root / grok.GROK_PATH
    source.parent.mkdir(parents=True)
    shutil.copyfile(binary, source)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    ownership = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid: ownership.append((str(path), uid, gid)))
    def probe(argv, **kwargs):
        assert argv == [str(root / "provider-bin/grok"), "--version"]
        assert kwargs["env"] == {"HOME": str(root / "home"), "PATH": "/usr/bin:/bin"}
        return SimpleNamespace(returncode=exit_code, stdout=output + "\n", stderr="private-diagnostic")
    monkeypatch.setattr(subprocess, "run", probe)
    if exit_code or "foreign" in output:
        with pytest.raises(SystemExit, match="native Grok version differs"):
            exec(compile(grok.grok_exposure_script(str(root)), "native-grok-exposure", "exec"), {})
        raw = capsys.readouterr().out
        result = json.loads(raw)
        assert result["schema"] == "native-grok-version-failure@1"
        assert "private-diagnostic" not in raw and "foreign" not in raw
    else:
        exec(compile(grok.grok_exposure_script(str(root)), "native-grok-exposure", "exec"), {})
        result = json.loads(capsys.readouterr().out)
        assert result["version_output_observed"] == output
        assert result["sha256"] == grok.GROK_SHA256
    assert ownership == [(str(root / "provider-bin/grok"), 0, 0)]
    assert result["provider_calls"] == 0
