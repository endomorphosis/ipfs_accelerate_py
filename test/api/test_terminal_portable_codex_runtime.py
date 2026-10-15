"""Pinned native packaging/exposure without installation or real CLI execution.

Authored ELF headers exercise byte and architecture admission only. Exposure
tests adapt only the host UID authority and mock executable checks; actual
cached CLI/container qualification is retained separately.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _elf(marker):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[16:18] = (2).to_bytes(2, "little")
    header[18:20] = (183).to_bytes(2, "little")
    return bytes(header) + marker.encode()


def _write_manifest(root, value):
    path = root / "manifest.json"
    path.write_bytes(_encoded(value))
    path.chmod(0o644)
    return _sha(path.read_bytes())


def _asset(root):
    root.mkdir(mode=0o755)
    (root / "bin").mkdir(mode=0o755)
    files = []
    for name in ("codex", "codex-code-mode-host"):
        raw = _elf("authored-header-only:" + name)
        path = root / "bin" / name
        path.write_bytes(raw)
        path.chmod(0o755)
        files.append({"name": name, "bytes": len(raw), "sha256": _sha(raw)})
    manifest = {"schema": "native-codex-portable-runtime@1", "codex_version": "0.158.0",
                "target": "aarch64-unknown-linux-musl", "files": files}
    return root, manifest, _write_manifest(root, manifest)


def _inputs(root):
    values = {}
    for name, package in (("source", "ipfs_accelerate_py"), ("datasets", "ipfs_datasets_py"), ("kit", "ipfs_kit_py")):
        path = root / name
        (path / package).mkdir(parents=True)
        (path / package / "__init__.py").write_text("VALUE = 1\n")
        values[name] = path
    bench = values["source"] / "benchmarks/agent_supervisor/container_coding"
    bench.mkdir(parents=True)
    (bench / "worker.py").write_text("VALUE = 2\n")
    extensions = root / "extensions"
    extensions.mkdir()
    for name in ("httpfs", "quack", "ducklake"):
        (extensions / (name + ".duckdb_extension")).write_bytes(b"authored extension placeholder")
    return dict(values, extension_dir=extensions)


def _bundle(tmp_path, *, opted_in=True):
    inputs = _inputs(tmp_path)
    options = {}
    if opted_in:
        root, _, digest = _asset(tmp_path / "selected-runtime")
        options = {"codex_runtime_dir": root, "codex_runtime_manifest_sha256": digest}
    path = tmp_path / "bundle"
    return path, deployment.build_runtime_archive(output=path, **inputs, **options)


def test_default_archive_and_exposure_do_not_enter_native_asset_path(tmp_path, monkeypatch):
    _, manifest = _bundle(tmp_path, opted_in=False)
    assert "codex_native_runtime" not in manifest
    assert manifest["torch_cpu_requirement"] == ""
    assert len(manifest["files"]) == 7
    assert not any(row["path"].startswith("codex-runtime/") for row in manifest["files"])
    monkeypatch.setattr(deployment, "_codex_runtime_validator", lambda: pytest.fail("default exposure entered optional validator"))
    script = deployment.native_codex_exposure_script()
    assert "matches=list((root/'home/.nvm').glob(" in script
    assert "binding_json=" not in script
    assert script.startswith("root_value='/opt/ipfs-supervisor'\nexpected_version='0.158.0'\n")


def test_optional_archive_contains_exact_pinned_inert_manifest_and_executable_members(tmp_path):
    path, manifest = _bundle(tmp_path)
    binding = manifest["codex_native_runtime"]
    assert binding["schema"] == "native-codex-portable-runtime-binding@1"
    assert binding["path"] == "codex-runtime" and binding["codex_version"] == "0.158.0"
    assert binding["target"] == "aarch64-unknown-linux-musl"
    assert manifest["torch_cpu_requirement"] == ""
    expected = {"codex-runtime/manifest.json": (binding["manifest_sha256"], 0o644)}
    expected.update({"codex-runtime/bin/" + row["name"]: (row["sha256"], 0o755) for row in binding["files"]})
    with tarfile.open(path / "runtime.tar.gz") as bundle:
        selected = [row for row in bundle.getmembers() if row.name.startswith("codex-runtime/")]
        assert {row.name for row in selected} == set(expected)
        for row in selected:
            assert row.isfile() and row.uid == row.gid == 0
            assert (_sha(bundle.extractfile(row).read()), row.mode) == expected[row.name]
    assert manifest["credentials_in_archive"] is False


@pytest.mark.parametrize("missing", ["directory", "pin"])
def test_portable_directory_and_pin_are_required_together_before_output(tmp_path, missing):
    root, _, digest = _asset(tmp_path / "selected-runtime")
    with pytest.raises(ValueError, match="required together"):
        deployment.build_runtime_archive(output=tmp_path / "never-created", **_inputs(tmp_path),
            codex_runtime_dir=None if missing == "directory" else root,
            codex_runtime_manifest_sha256=None if missing == "pin" else digest)
    assert not (tmp_path / "never-created").exists()


@pytest.mark.parametrize("alteration", [
    "wrong-pin", "manifest-tamper", "file-tamper", "wrong-version", "wrong-target", "foreign-elf",
    "non-elf", "non-executable", "symlink-root", "symlink-bin", "symlink-manifest", "symlink-file",
    "hardlink-file", "missing-file", "incomplete-manifest", "extra-file", "extra-root", "extra-manifest-key",
    "duplicate-file", "unsafe-name", "excess-members", "excess-total", "duplicate-key",
])
def test_invalid_native_assets_refuse_before_archive_output(tmp_path, alteration):
    root, manifest, digest = _asset(tmp_path / "selected-runtime")
    file = root / "bin/codex"
    if alteration == "wrong-pin":
        digest = "0" * 64
    elif alteration == "manifest-tamper":
        (root / "manifest.json").write_bytes((root / "manifest.json").read_bytes() + b" ")
    elif alteration == "file-tamper":
        file.write_bytes(file.read_bytes() + b"tamper")
    elif alteration == "wrong-version":
        manifest["codex_version"] = "0.125.0"
    elif alteration == "wrong-target":
        manifest["target"] = "aarch64-unknown-linux-gnu"
    elif alteration in {"foreign-elf", "non-elf"}:
        raw = bytearray(file.read_bytes())
        if alteration == "foreign-elf":
            raw[18:20] = (62).to_bytes(2, "little")
        else:
            raw[:4] = b"TEXT"
        file.write_bytes(raw)
        manifest["files"][0]["sha256"] = _sha(raw)
    elif alteration == "non-executable":
        file.chmod(0o644)
    elif alteration == "symlink-root":
        alias = tmp_path / "runtime-alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    elif alteration in {"symlink-bin", "symlink-manifest", "symlink-file"}:
        target = {"symlink-bin": root / "bin", "symlink-manifest": root / "manifest.json", "symlink-file": file}[alteration]
        retained = tmp_path / ("retained-" + target.name)
        target.rename(retained)
        target.symlink_to(retained, target_is_directory=alteration == "symlink-bin")
    elif alteration == "hardlink-file":
        os.link(file, tmp_path / "hardlink")
    elif alteration == "missing-file":
        (root / "bin/codex-code-mode-host").unlink()
    elif alteration == "incomplete-manifest":
        manifest["files"].pop()
    elif alteration == "extra-file":
        (root / "bin/unselected").write_bytes(b"extra")
    elif alteration == "extra-root":
        (root / "unselected").write_bytes(b"extra")
    elif alteration == "extra-manifest-key":
        manifest["download_url"] = "unused"
    elif alteration == "duplicate-file":
        manifest["files"].append(dict(manifest["files"][0]))
    elif alteration == "unsafe-name":
        manifest["files"][0]["name"] = "../codex"
    elif alteration == "excess-members":
        manifest["files"] += [{"name": "member" + str(i), "bytes": 64, "sha256": "a" * 64} for i in range(31)]
    elif alteration == "excess-total":
        for row in manifest["files"]:
            row["bytes"] = 512 * 1024 * 1024
    if alteration in {"wrong-version", "wrong-target", "foreign-elf", "non-elf", "incomplete-manifest",
                      "extra-manifest-key", "duplicate-file", "unsafe-name", "excess-members", "excess-total"}:
        digest = _write_manifest(root, manifest)
    elif alteration == "duplicate-key":
        raw = b'{"codex_version":"shadow",' + _encoded(manifest)[1:]
        (root / "manifest.json").write_bytes(raw)
        digest = _sha(raw)
    with pytest.raises((ValueError, OSError)):
        deployment.build_runtime_archive(output=tmp_path / "never-created", **_inputs(tmp_path),
            codex_runtime_dir=root, codex_runtime_manifest_sha256=digest)
    assert not (tmp_path / "never-created").exists()


def test_an_earlier_member_changed_during_later_capture_refuses_output(tmp_path, monkeypatch):
    root, _, digest = _asset(tmp_path / "selected-runtime")
    namespace = deployment._codex_runtime_validator()
    original = namespace["read_member"]

    def read(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1] == "codex-code-mode-host":
            first = root / "bin/codex"
            first.write_bytes(first.read_bytes() + b"changed after initial verification")
        return result

    namespace["read_member"] = read
    monkeypatch.setattr(deployment, "_codex_runtime_validator", lambda: namespace)
    with pytest.raises(ValueError, match="changed after capture"):
        deployment.build_runtime_archive(output=tmp_path / "never-created", **_inputs(tmp_path),
            codex_runtime_dir=root, codex_runtime_manifest_sha256=digest)
    assert not (tmp_path / "never-created").exists()


def _exposure(tmp_path, monkeypatch, *, reported_version="0.158.0", alteration=None):
    root = tmp_path / "runtime-root"
    root.mkdir(mode=0o755)
    asset, _, digest = _asset(root / "codex-runtime")
    _, binding = deployment._codex_runtime_assets(directory=asset, manifest_sha256=digest)
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="codex-cli " + reported_version + "\n" if argv[-1] == "--version" else "authored help\n")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    if alteration == "hash":
        selected = asset / "bin/codex"
        selected.write_bytes(selected.read_bytes() + b"changed")
    elif alteration == "symlink":
        selected = asset / "bin/codex"
        retained = tmp_path / "retained-codex"
        selected.rename(retained)
        selected.symlink_to(retained)
    elif alteration == "platform":
        monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    elif alteration == "root-mode":
        root.chmod(0o775)
    elif alteration == "file-mode":
        (asset / "bin/codex").chmod(0o775)
    script = deployment.native_codex_exposure_script(str(root), packaged_binding=binding)
    assert script.count("OWNER_UID = 0") == 1
    # This authored UID adaptation exercises root protection semantics locally;
    # neither the fake ELF bodies nor a provider is actually executed.
    script = script.replace("OWNER_UID = 0", "OWNER_UID = " + str(os.geteuid()), 1)
    return root, binding, calls, script


def test_exposure_checks_both_executables_and_copies_exact_protected_members(tmp_path, monkeypatch, capsys):
    root, binding, calls, script = _exposure(tmp_path, monkeypatch)
    exec(compile(script, "<authored-native-exposure>", "exec"), {})
    receipt = json.loads(capsys.readouterr().out)
    assert [argv for argv, _ in calls] == [
        [str(root / "provider-bin/codex"), "--version"],
        [str(root / "provider-bin/codex-code-mode-host"), "--help"],
    ]
    assert receipt["codex_version"] == "0.158.0" and receipt["provider_calls"] == 0
    for row in binding["files"]:
        path = root / "provider-bin" / row["name"]
        assert _sha(path.read_bytes()) == row["sha256"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o755 and path.stat().st_nlink == 1
        assert path.stat().st_ino != (root / "codex-runtime/bin" / row["name"]).stat().st_ino


@pytest.mark.parametrize("alteration", ["hash", "symlink", "platform", "root-mode", "file-mode"])
def test_packaged_exposure_refuses_invalid_assets_before_executable_checks(tmp_path, monkeypatch, alteration):
    root, _, calls, script = _exposure(tmp_path, monkeypatch, alteration=alteration)
    with pytest.raises((ValueError, OSError)):
        exec(compile(script, "<authored-native-exposure>", "exec"), {})
    assert calls == [] and not (root / "provider-bin").exists()


def test_exposure_rejects_actual_reported_cli_version_mismatch(tmp_path, monkeypatch):
    _, _, calls, script = _exposure(tmp_path, monkeypatch, reported_version="0.125.0")
    with pytest.raises(ValueError, match="executable check failed"):
        exec(compile(script, "<authored-native-exposure>", "exec"), {})
    assert len(calls) == 1 and calls[0][0][-1] == "--version"


class _Environment:
    def __init__(self):
        self.default_user = "original-owner"
        self.executions = []
        self.uploads = []

    async def upload_file(self, source, destination):
        self.uploads.append((Path(source), destination))

    async def exec(self, **kwargs):
        self.executions.append(kwargs)
        command = kwargs["command"]
        arguments = shlex.split(command)
        code = arguments[arguments.index("-c") + 1] if "-c" in arguments else command
        if "root=pathlib.Path('/app')" in code:
            stdout = json.dumps({"head": "authored-head", "status": "", "files": {}, "report_exists": False})
        elif "codex --version" in command or command.endswith("/provider-bin/codex --version"):
            stdout = "codex-cli 0.158.0\n"
        elif "AdmittedBenchmarkRuntime" in command:
            stdout = json.dumps({"python": "3.12.12", "duckdb": "1.5.5", "native_imports": True})
        else:
            stdout = ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")


@pytest.mark.parametrize("opted_in", [False, True], ids=["default-installer", "packaged-installer-bypass"])
def test_deployment_uses_installer_only_for_legacy_profile(tmp_path, monkeypatch, opted_in):
    path, manifest = _bundle(tmp_path, opted_in=opted_in)
    constructors, installations = [], []
    module = ModuleType("harbor.agents.installed.codex")

    class Codex:
        def __init__(self, **kwargs):
            constructors.append(kwargs)

        async def install(self, environment):
            installations.append(environment.default_user)

    module.Codex = Codex
    monkeypatch.setitem(sys.modules, "harbor.agents.installed.codex", module)
    environment = _Environment()
    result = asyncio.run(deployment.deploy_supervisor(environment, archive_dir=path, output=tmp_path / "deployment"))
    assert environment.default_user == "original-owner"
    assert len(constructors) == len(installations) == (0 if opted_in else 1)
    assert result["qualified"] and result["task_source_preserved"] and result["provider_calls"] == 0
    assert environment.uploads == [(path / "runtime.tar.gz", deployment.ROOT + "/runtime.tar.gz")]
    scripts = [shlex.split(row["command"])[-1] for row in environment.executions if "expected_version=" in row["command"]]
    assert len(scripts) == 1
    assert ("binding_json=" in scripts[0]) is opted_in
    if opted_in:
        assert result["codex_native_runtime"] == manifest["codex_native_runtime"]
        assert any(row["command"] == deployment.ROOT + "/provider-bin/codex --version" for row in environment.executions)
    else:
        assert "codex_native_runtime" not in result
        assert constructors[0]["version"] == "0.158.0"


@pytest.mark.parametrize("alteration", ["version", "target", "extra", "null"])
def test_invalid_packaged_binding_refuses_before_environment_or_auth(tmp_path, alteration):
    path, manifest = _bundle(tmp_path)
    value = manifest["codex_native_runtime"]
    if alteration == "version":
        value["codex_version"] = "0.125.0"
    elif alteration == "target":
        value["target"] = "x86_64-unknown-linux-musl"
    elif alteration == "extra":
        value["unsafe_fallback"] = True
    else:
        manifest["codex_native_runtime"] = None
    (path / "manifest.json").write_bytes(_encoded(manifest))
    environment = _Environment()
    with pytest.raises(ValueError):
        asyncio.run(deployment.deploy_supervisor(environment, archive_dir=path, output=tmp_path / "deployment",
                                                auth_json=tmp_path / "unread-auth.json"))
    assert environment.executions == environment.uploads == []
