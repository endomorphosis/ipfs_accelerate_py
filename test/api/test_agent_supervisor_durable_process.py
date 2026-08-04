import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.durable_process import (
    DurableProcessError,
    launch_systemd_user_service,
    main,
)


def _completed(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_systemd_user_launch_is_argv_safe_and_returns_verified_receipt(
    tmp_path,
) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["--user", "show"]:
            return _completed(
                stdout=(
                    "MainPID=4242\n"
                    "Result=success\n"
                    "ActiveState=active\n"
                    "SubState=running\n"
                )
            )
        return _completed()

    log_path = tmp_path / "logs" / "supervisor output.log"
    receipt = launch_systemd_user_service(
        ["/usr/bin/python3", "-c", "print('a;$(unsafe)')"],
        unit_name="crypto-ir-codex",
        working_directory=tmp_path,
        log_path=log_path,
        environment={"PYTHONPATH": "/tmp/a path", "SAFE": "$(literal)"},
        systemd_run_path="/usr/bin/systemd-run",
        systemctl_path="/usr/bin/systemctl",
        runner=runner,
    )

    launch_argv = calls[0][0]
    assert launch_argv[:4] == [
        "/usr/bin/systemd-run",
        "--user",
        "--quiet",
        "--unit=crypto-ir-codex.service",
    ]
    assert f"--working-directory={tmp_path}" in launch_argv
    assert (
        f"--property=StandardOutput=append:{log_path}" in launch_argv
    )
    assert "--property=SuccessExitStatus=143 SIGTERM" in launch_argv
    assert "--setenv=PYTHONPATH=/tmp/a path" in launch_argv
    assert "--setenv=SAFE=$(literal)" in launch_argv
    separator = launch_argv.index("--")
    assert launch_argv[separator + 1 :] == [
        "/usr/bin/python3",
        "-c",
        "print('a;$(unsafe)')",
    ]
    assert receipt.unit_name == "crypto-ir-codex.service"
    assert receipt.backend == "systemd-user"
    assert receipt.pid == 4242
    assert receipt.active_state == "active"
    assert receipt.sub_state == "running"
    assert Path(receipt.log_path) == log_path


@pytest.mark.parametrize(
    ("unit_name", "environment", "match"),
    [
        ("bad/unit", {}, "unit name"),
        ("valid", {"BAD-NAME": "value"}, "environment variable"),
        ("valid", {"SAFE": "bad\x00value"}, "NUL"),
    ],
)
def test_systemd_user_launch_rejects_unsafe_identifiers(
    tmp_path,
    unit_name,
    environment,
    match,
) -> None:
    with pytest.raises(DurableProcessError, match=match):
        launch_systemd_user_service(
            ["/bin/true"],
            unit_name=unit_name,
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            environment=environment,
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
        )


def test_failed_post_launch_inspection_stops_exact_created_unit(
    tmp_path,
) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["--user", "show"]:
            return _completed(
                stdout=(
                    "MainPID=0\n"
                    "Result=exit-code\n"
                    "ActiveState=failed\n"
                    "SubState=failed\n"
                )
            )
        return _completed()

    with pytest.raises(DurableProcessError, match="live main process"):
        launch_systemd_user_service(
            ["/bin/false"],
            unit_name="failed-supervisor",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=runner,
        )

    assert calls[-1] == [
        "/usr/bin/systemctl",
        "--user",
        "stop",
        "failed-supervisor.service",
    ]


def test_launch_timeout_does_not_echo_forwarded_environment(
    tmp_path,
) -> None:
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    secret = "must-not-appear-in-errors"
    with pytest.raises(DurableProcessError) as exc_info:
        launch_systemd_user_service(
            ["/bin/sleep", "10"],
            unit_name="timeout-supervisor",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            environment={"TOKEN": secret},
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=runner,
        )

    assert "timed out" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_cli_passes_only_named_environment_and_strips_separator(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    captured = {}

    def fake_launch(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            to_dict=lambda: {
                "backend": "systemd-user",
                "unit_name": "lane.service",
                "pid": 99,
            }
        )

    monkeypatch.setenv("SELECTED_ENV", "selected value")
    monkeypatch.setenv("UNSELECTED_ENV", "must not cross")
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.durable_process."
        "launch_systemd_user_service",
        fake_launch,
    )

    assert (
        main(
            [
                "--unit",
                "lane",
                "--working-directory",
                str(tmp_path),
                "--log-path",
                str(tmp_path / "lane.log"),
                "--pass-env",
                "SELECTED_ENV",
                "--",
                "/bin/echo",
                "--literal",
            ]
        )
        == 0
    )
    assert captured["command"] == ["/bin/echo", "--literal"]
    assert captured["environment"] == {"SELECTED_ENV": "selected value"}
    assert '"pid": 99' in capsys.readouterr().out
