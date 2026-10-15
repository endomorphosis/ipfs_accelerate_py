"""Portable runtime setup-to-worker custody, with no real CLI or Docker calls.

Exposure reuses authored ELF byte fixtures, a host UID adaptation and mocked
version/help checks. Actual native/container qualification is separate.
"""

import asyncio
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_deployment as deployment
from benchmarks.agent_supervisor.container_coding import container_worker_deployment as workers
from benchmarks.agent_supervisor.container_coding import full_supervisor_harbor_agent as adapter
from test.api.test_terminal_portable_codex_runtime import _asset, _exposure


def _binding(tmp_path):
    root, _, digest = _asset(tmp_path / "selected-runtime")
    return deployment._codex_runtime_assets(directory=root, manifest_sha256=digest)[1]


def _identity(path, *, follow_symlinks=True):
    info = path.stat() if follow_symlinks else path.lstat()
    return tuple(getattr(info, field) for field in (
        "st_dev", "st_ino", "st_nlink", "st_mode", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns",
    ))


def test_repeated_portable_exposure_rechecks_without_writing_or_replacing_members(tmp_path, monkeypatch, capsys):
    root, binding, calls, script = _exposure(tmp_path, monkeypatch)
    exec(compile(script, "<authored-fresh-exposure>", "exec"), {})
    first_receipt = json.loads(capsys.readouterr().out)
    paths = [root / "provider-bin" / row["name"] for row in binding["files"]]
    identities = {path: _identity(path) for path in paths}
    bodies = {path: path.read_bytes() for path in paths}
    original_open = os.open

    def open_readonly(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT), "repeat exposure attempted a write"
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_readonly)
    monkeypatch.setattr(os, "fchown", lambda *args, **kwargs: pytest.fail("repeat exposure changed owner"))
    monkeypatch.setattr(os, "fchmod", lambda *args, **kwargs: pytest.fail("repeat exposure changed mode"))
    exec(compile(script, "<authored-repeat-exposure>", "exec"), {})
    assert json.loads(capsys.readouterr().out) == first_receipt
    assert [_identity(path) for path in paths] == [identities[path] for path in paths]
    assert {path: path.read_bytes() for path in paths} == bodies
    assert [argv[-1] for argv, _ in calls] == ["--version", "--help", "--version", "--help"]


@pytest.mark.parametrize("alteration", ["hash", "mode", "extra", "missing", "hardlink", "symlink", "source"])
def test_repeat_exposure_refuses_drift_before_checks_and_never_repairs_it(tmp_path, monkeypatch, capsys, alteration):
    root, _, calls, script = _exposure(tmp_path, monkeypatch)
    exec(compile(script, "<authored-fresh-exposure>", "exec"), {})
    capsys.readouterr()
    calls.clear()
    selected = root / "provider-bin/codex"
    if alteration == "hash":
        raw = selected.read_bytes()
        selected.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    elif alteration == "mode":
        selected.chmod(0o775)
    elif alteration == "extra":
        (root / "provider-bin/extra").write_bytes(b"unselected")
    elif alteration == "missing":
        (root / "provider-bin/codex-code-mode-host").unlink()
    elif alteration == "hardlink":
        os.link(selected, tmp_path / "aliased-codex")
    elif alteration == "symlink":
        retained = tmp_path / "retained-codex"
        selected.rename(retained)
        selected.symlink_to(retained)
    else:
        native = root / "codex-runtime/bin/codex"
        native.write_bytes(native.read_bytes() + b"stale packaged source")
    before = {path.name: _identity(path, follow_symlinks=False) for path in (root / "provider-bin").iterdir()}
    with pytest.raises((ValueError, OSError)):
        exec(compile(script, "<authored-repeat-exposure>", "exec"), {})
    assert calls == []
    after = {path.name: _identity(path, follow_symlinks=False) for path in (root / "provider-bin").iterdir()}
    assert before == after


class _Environment:
    def __init__(self):
        self.executions = []
        self.uploads = []
        self.lookups = []
        self._platform = SimpleNamespace(_resolve_service_container=self.resolve)

    async def resolve(self, name):
        self.lookups.append(name)
        return "authored-running-container"

    async def exec(self, **kwargs):
        self.executions.append(kwargs)
        if "readlink" in kwargs["command"]:
            stdout = json.dumps({key: key + ":[17]" for key in ("pid", "mnt", "net")})
        else:
            stdout = ""
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    async def upload_file(self, source, destination):
        self.uploads.append((Path(source), destination))


@pytest.mark.parametrize("opted_in", [False, True], ids=["legacy", "portable"])
def test_worker_installer_uses_explicit_binding_and_preserves_default_script(tmp_path, monkeypatch, opted_in):
    selected = _binding(tmp_path) if opted_in else None
    docker_calls = []

    def inspect(arguments, **kwargs):
        docker_calls.append(arguments)
        return '"authored-id" "authored-image" 1234 true false'

    monkeypatch.setattr(subprocess, "check_output", inspect)
    monkeypatch.setattr(os, "readlink", lambda path: path.rsplit("/", 1)[-1] + ":[17]")
    environment = _Environment()
    result = asyncio.run(workers.install_worker_boundary(
        environment, container_name="authored-running-container", output=tmp_path / "installation",
        **({"codex_native_runtime": selected} if opted_in else {}),
    ))
    commands = [row["command"] for row in environment.executions if "expected_version=" in row["command"]]
    assert len(commands) == 1
    expected = deployment.native_codex_exposure_script(**({"packaged_binding": selected} if opted_in else {}))
    assert shlex.split(commands[0]) == ["python3", "-I", "-c", expected]
    assert docker_calls and result["qualified"] is False and result["provider_calls"] == 0
    if opted_in:
        assert result["codex_native_runtime"] == selected and result["codex_native_runtime"] is not selected
    else:
        assert "codex_native_runtime" not in result


@pytest.mark.parametrize("opted_in", [False, True], ids=["legacy", "portable"])
def test_worker_deployer_forwards_only_selected_binding_and_keeps_qualification(tmp_path, monkeypatch, opted_in):
    selected = _binding(tmp_path) if opted_in else None
    environment = _Environment()
    calls = []

    async def install(env, **kwargs):
        calls.append(("install", env, kwargs))
        return {"schema": "container-worker-deployment@1", "qualified": False,
                **({"codex_native_runtime": kwargs["codex_native_runtime"]} if opted_in else {})}

    async def qualify(env, **kwargs):
        calls.append(("qualify", env, kwargs))
        return {"qualified": True, "provider_calls": 0}

    monkeypatch.setattr(workers, "install_worker_boundary", install)
    monkeypatch.setattr(workers, "qualify_worker_boundary", qualify)
    result = asyncio.run(workers.deploy_worker_boundary(environment, output=tmp_path / "worker",
        **({"codex_native_runtime": selected} if opted_in else {})))
    assert environment.lookups == ["main"]
    expected = {"container_name": "authored-running-container", "output": tmp_path / "worker/installation"}
    if opted_in:
        expected["codex_native_runtime"] = selected
    assert calls == [("install", environment, expected),
                     ("qualify", environment, {"output": tmp_path / "worker/qualification"})]
    assert result["qualified"] and result["qualification"]["provider_calls"] == 0


@pytest.mark.parametrize("method", ["install", "deploy"])
def test_invalid_worker_binding_refuses_before_docker_environment_or_output(tmp_path, monkeypatch, method):
    selected = _binding(tmp_path)
    selected["manifest_sha256"] = "unbound"
    environment = _Environment()
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: pytest.fail("invalid binding reached Docker"))
    output = tmp_path / "never-created"
    arguments = {"output": output, "codex_native_runtime": selected}
    if method == "install":
        arguments["container_name"] = "unused"
    with pytest.raises(ValueError):
        asyncio.run(getattr(workers, method + "_worker_boundary")(environment, **arguments))
    assert not output.exists()
    assert environment.lookups == environment.executions == environment.uploads == []


def _agent(tmp_path):
    agent = adapter.FullSupervisorAgent.__new__(adapter.FullSupervisorAgent)
    agent.runtime_archive = tmp_path / "archive"
    agent.auth_json = tmp_path / "unread-auth.fixture.json"
    agent.logs_dir = tmp_path / "logs"
    return agent


@pytest.mark.parametrize("report_kind", ["none", "legacy", "portable"])
def test_actual_adapter_setup_preserves_legacy_calls_and_forwards_qualified_portable_binding(
    tmp_path, monkeypatch, report_kind,
):
    selected = _binding(tmp_path)
    report = (None if report_kind == "none" else {"schema": "terminal-native-supervisor-deployment@1",
        "qualified": True, "codex_installed": True,
        **({"codex_native_runtime": selected} if report_kind == "portable" else {})})
    agent = _agent(tmp_path)
    environment = object()
    calls = []

    async def deploy(env, **kwargs):
        calls.append(("deploy", env, kwargs))
        return report

    async def worker(env, **kwargs):
        calls.append(("worker", env, kwargs))

    monkeypatch.setattr(adapter, "deploy_supervisor", deploy)
    monkeypatch.setattr(adapter, "deploy_worker_boundary", worker)
    asyncio.run(agent.setup(environment))
    worker_options = {"output": agent.logs_dir / "worker-boundary"}
    if report_kind == "portable":
        worker_options["codex_native_runtime"] = selected
    assert calls == [("deploy", environment, {"archive_dir": agent.runtime_archive,
        "output": agent.logs_dir / "deployment", "auth_json": agent.auth_json}),
        ("worker", environment, worker_options)]
    if report_kind == "portable":
        assert calls[-1][2]["codex_native_runtime"] is not selected


@pytest.mark.parametrize("alteration", ["unqualified", "numeric-qualified", "not-installed", "foreign-report",
                                         "null-binding", "foreign-binding", "unbound-binding"])
def test_adapter_refuses_invalid_or_unqualified_explicit_binding_before_worker_side_effects(tmp_path, monkeypatch, alteration):
    selected = _binding(tmp_path)
    report = {"schema": "terminal-native-supervisor-deployment@1", "qualified": True,
              "codex_installed": True, "codex_native_runtime": selected}
    if alteration == "unqualified":
        report["qualified"] = False
    elif alteration == "numeric-qualified":
        report["qualified"] = 1
    elif alteration == "not-installed":
        report["codex_installed"] = False
    elif alteration == "foreign-report":
        report["schema"] = "foreign@1"
    elif alteration == "null-binding":
        report["codex_native_runtime"] = None
    elif alteration == "foreign-binding":
        selected["target"] = "x86_64-unknown-linux-musl"
    else:
        selected["manifest_sha256"] = "unbound"
    calls = []

    async def deploy(*args, **kwargs):
        return report

    async def worker(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(adapter, "deploy_supervisor", deploy)
    monkeypatch.setattr(adapter, "deploy_worker_boundary", worker)
    with pytest.raises(ValueError):
        asyncio.run(_agent(tmp_path).setup(object()))
    assert calls == [] and not (tmp_path / "logs/worker-boundary").exists()
