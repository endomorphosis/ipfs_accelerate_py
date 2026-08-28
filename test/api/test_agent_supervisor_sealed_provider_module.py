from __future__ import annotations

import fcntl
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import provider_fallback_runner
from ipfs_accelerate_py.agent_supervisor.sealed_provider_module import (
    LGCVF_SEALED_PROVIDER_BOOTSTRAP,
    build_sealed_provider_module_command,
    sealed_provider_capsule_descriptor,
    sealed_provider_module_command_descriptor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon,
)

_PROVIDER_MODULE = "ipfs_accelerate_py.agent_supervisor.provider_fallback_runner"
_GROK_MODULE = "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"
_REQUIRED_SEALS = (
    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL
)


@pytest.fixture
def sealed_provider_capsule() -> int:
    if (
        not sys.platform.startswith("linux")
        or not hasattr(os, "memfd_create")
        or not hasattr(os, "MFD_ALLOW_SEALING")
        or not hasattr(fcntl, "F_ADD_SEALS")
    ):
        pytest.skip("sealed memfd ZIP execution requires Linux memfd seals")

    archive = io.BytesIO()
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as bundle:
        bundle.writestr("ipfs_accelerate_py/__init__.py", "")
        bundle.writestr(
            "ipfs_accelerate_py/agent_supervisor/__init__.py",
            "",
        )
        bundle.writestr(
            "ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py",
            """\
import argparse

parser = argparse.ArgumentParser(
    prog="sealed-provider-probe",
    description="SEALED_PROVIDER_HELP",
)
parser.add_argument("--sealed-probe", action="store_true")
arguments = parser.parse_args()
if arguments.sealed_probe:
    print("SEALED_PROVIDER_EXEC:" + __file__)
""",
        )
        bundle.writestr(
            "ipfs_accelerate_py/agent_supervisor/grok_cli_runner.py",
            """\
import argparse

parser = argparse.ArgumentParser(prog="sealed-grok-probe")
parser.add_argument("--workspace", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--max-turns", required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--require-terminal-quota-frame", action="store_true")
parser.add_argument("--grok-bin", required=True)
parser.parse_args()
print("SEALED_GROK_EXEC:" + __file__)
""",
        )

    descriptor = os.memfd_create(
        "ipfs-accelerate-test-sealed-provider",
        os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING,
    )
    try:
        body = archive.getvalue()
        offset = 0
        while offset < len(body):
            written = os.write(descriptor, body[offset:])
            assert written > 0
            offset += written
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _REQUIRED_SEALS)
        assert fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _REQUIRED_SEALS == (
            _REQUIRED_SEALS
        )
        yield descriptor
    finally:
        os.close(descriptor)


def _sealed_daemon_origin(descriptor: int) -> str:
    return (
        f"/proc/self/fd/{descriptor}/ipfs_accelerate_py/agent_supervisor/"
        "todo_daemon/implementation_daemon.py"
    )


def _sealed_provider_origin(descriptor: int) -> str:
    return (
        f"/proc/self/fd/{descriptor}/ipfs_accelerate_py/agent_supervisor/"
        "provider_fallback_runner.py"
    )


def _daemon(tmp_path: Path) -> implementation_daemon.TodoImplementationDaemon:
    todo_path = tmp_path / "todo.md"
    todo_path.write_text("# Tasks\n", encoding="utf-8")
    return implementation_daemon.TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
    )


def test_sealed_provider_command_is_exact_and_runs_member_from_memfd_zip(
    sealed_provider_capsule: int,
    tmp_path: Path,
) -> None:
    descriptor = sealed_provider_capsule
    origin = _sealed_daemon_origin(descriptor)
    command = build_sealed_provider_module_command(
        _PROVIDER_MODULE,
        ["--help"],
        module_file=origin,
    )

    assert command is not None
    assert command[:6] == [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        LGCVF_SEALED_PROVIDER_BOOTSTRAP,
    ]
    assert command[6:8] == [str(descriptor), _PROVIDER_MODULE]
    assert sealed_provider_capsule_descriptor(origin) == descriptor
    assert (
        sealed_provider_capsule_descriptor(_sealed_provider_origin(descriptor))
        == descriptor
    )
    assert (
        sealed_provider_module_command_descriptor(
            command,
            module_name=_PROVIDER_MODULE,
        )
        == descriptor
    )
    assert not Path(
        f"/proc/self/fd/{descriptor}/ipfs_accelerate_py/agent_supervisor/"
        "provider_fallback_runner.py"
    ).is_file()

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        pass_fds=(descriptor,),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SEALED_PROVIDER_HELP" in completed.stdout
    assert "sealed-provider-probe" in completed.stdout


def test_sealed_provider_command_parser_rejects_shape_and_identity_tamper(
    sealed_provider_capsule: int,
) -> None:
    descriptor = sealed_provider_capsule
    command = build_sealed_provider_module_command(
        _PROVIDER_MODULE,
        ["--sealed-probe"],
        module_file=_sealed_daemon_origin(descriptor),
    )
    assert command is not None

    tampered_commands: list[list[str]] = []
    for index, replacement in (
        (0, "/bin/false"),
        (1, "-E"),
        (2, "-s"),
        (3, "-O"),
        (4, "-m"),
        (5, LGCVF_SEALED_PROVIDER_BOOTSTRAP + " "),
        (6, "2"),
        (6, "not-a-descriptor"),
        (6, "999999999"),
        (7, "ipfs_accelerate_py.agent_supervisor.lookalike_runner"),
    ):
        candidate = list(command)
        candidate[index] = replacement
        tampered_commands.append(candidate)
    tampered_commands.append(["injected-prefix", *command])

    assert all(
        sealed_provider_module_command_descriptor(candidate) == -1
        for candidate in tampered_commands
    )
    assert (
        sealed_provider_module_command_descriptor(
            command,
            module_name=_GROK_MODULE,
        )
        == -1
    )
    assert (
        sealed_provider_capsule_descriptor(
            _sealed_daemon_origin(descriptor).replace(
                "implementation_daemon.py",
                "implementation_daemon.py/extra",
            )
        )
        == -1
    )
    assert (
        sealed_provider_capsule_descriptor(
            _sealed_daemon_origin(descriptor).replace("/fd/", "/fd/-")
        )
        == -1
    )
    assert (
        build_sealed_provider_module_command(
            "ipfs_accelerate_py.agent_supervisor.lookalike_runner",
            (),
            module_file=_sealed_daemon_origin(descriptor),
        )
        is None
    )


def test_sealed_provider_bootstrap_rejects_unsealed_file_descriptor(
    tmp_path: Path,
) -> None:
    unsealed = tmp_path / "unsealed-provider.zip"
    unsealed.write_bytes(b"not a sealed provider archive")
    descriptor = os.open(unsealed, os.O_RDONLY)
    try:
        command = build_sealed_provider_module_command(
            _PROVIDER_MODULE,
            ["--help"],
            module_file=_sealed_daemon_origin(descriptor),
        )
        assert command is not None
        completed = subprocess.run(
            command,
            cwd=tmp_path,
            pass_fds=(descriptor,),
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        os.close(descriptor)

    assert completed.returncode == 78
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_sealed_capsule_fd_survives_outer_and_nested_provider_launches(
    sealed_provider_capsule: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptor = sealed_provider_capsule
    daemon_origin = Path(_sealed_daemon_origin(descriptor))
    grok_origin = daemon_origin.parents[1] / "grok_cli_runner.py"
    fallback_origin = daemon_origin.parents[1] / "provider_fallback_runner.py"
    monkeypatch.setattr(implementation_daemon, "__file__", str(daemon_origin))
    monkeypatch.setattr(
        implementation_daemon,
        "_TRUSTED_QUOTA_FALLBACK_SCRIPT",
        grok_origin,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_TRUSTED_PROVIDER_FALLBACK_SCRIPT",
        fallback_origin,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/usr/bin/true",
    )

    grok_command = implementation_daemon._grok_cli_trusted_failure_command(
        workspace_path=tmp_path,
        model="grok-4.6",
    )
    fallback_command = implementation_daemon._ordered_provider_fallback_command(
        workspace_path=tmp_path,
        primary_provider="grok",
        primary_command=grok_command,
        fallback_provider="codex",
        fallback_command=["/usr/bin/true"],
    )

    assert (
        sealed_provider_module_command_descriptor(
            grok_command,
            module_name=_GROK_MODULE,
        )
        == descriptor
    )
    assert (
        sealed_provider_module_command_descriptor(
            fallback_command,
            module_name=_PROVIDER_MODULE,
        )
        == descriptor
    )
    primary = json.loads(
        fallback_command[fallback_command.index("--primary-command-json") + 1]
    )
    assert primary == grok_command
    assert _daemon(tmp_path)._accepted_control_plane_pass_fds(fallback_command) == (
        descriptor,
    )

    launches: list[dict[str, object]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> subprocess.Popen[str]:
        launches.append(dict(kwargs))
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(
        provider_fallback_runner.subprocess,
        "Popen",
        recording_popen,
    )
    execution = provider_fallback_runner._run_provider(
        grok_command,
        workspace=tmp_path,
        prompt="harmless sealed provider probe\n",
        provider_name="grok",
    )

    captured = capsys.readouterr()
    assert execution.result.returncode == 0
    assert execution.result.launched is True
    assert "SEALED_GROK_EXEC:" in captured.out
    assert f"/proc/self/fd/{descriptor}/" in captured.out
    assert len(launches) == 1
    inherited = tuple(launches[0]["pass_fds"])
    assert inherited[0] == descriptor
    assert len(inherited) == 2
    assert len(set(inherited)) == 2
    assert launches[0]["close_fds"] is True
