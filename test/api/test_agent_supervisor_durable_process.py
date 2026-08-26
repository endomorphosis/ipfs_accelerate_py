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
    assert "--service-type=exec" in launch_argv
    assert not any(
        item.startswith("--property=PIDFile=") for item in launch_argv
    )
    assert "--property=GuessMainPID=no" not in launch_argv
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


def test_forking_launch_binds_systemd_to_exact_hardened_pid_file(
    tmp_path,
) -> None:
    calls = []
    pid_path = tmp_path / "state" / "configured-board-master.pid"
    pid_path.parent.mkdir()
    pid_path.write_text("4242\n", encoding="ascii")
    pid_path.chmod(0o600)

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if "--property=LoadState" in command:
            return _completed(
                stdout="LoadState=not-found\nActiveState=inactive\n"
            )
        if command[1:3] == ["--user", "show"]:
            return _completed(
                stdout=(
                    "MainPID=4242\n"
                    "Result=success\n"
                    "ActiveState=active\n"
                    "SubState=running\n"
                    "Type=forking\n"
                    f"PIDFile={pid_path}\n"
                    "GuessMainPID=no\n"
                )
            )
        return _completed()

    receipt = launch_systemd_user_service(
        [
            "/usr/bin/python3",
            "scripts/run_agent_supervisor_residual_intelligence.py",
            "--config",
            "config/agent_supervisor_residual_intelligence_scheduler.json",
            "launch-supervisor",
        ],
        unit_name="vrif-supervisor",
        working_directory=tmp_path,
        log_path=tmp_path / "logs" / "launcher.log",
        forking_pid_file=pid_path,
        systemd_run_path="/usr/bin/systemd-run",
        systemctl_path="/usr/bin/systemctl",
        runner=runner,
    )

    launch_argv = next(
        command for command, _kwargs in calls if command[0].endswith("systemd-run")
    )
    assert "--service-type=forking" in launch_argv
    assert f"--property=PIDFile={pid_path}" in launch_argv
    assert "--property=GuessMainPID=no" in launch_argv
    assert not any("QUACK_TOKEN" in item for item in launch_argv)
    assert receipt.pid == 4242
    assert sum(
        command[1:3] == ["--user", "show"]
        and "--property=MainPID" in command
        for command, _kwargs in calls
    ) == 2


def test_forking_launch_stops_exact_unit_on_pid_file_mismatch(
    tmp_path,
) -> None:
    calls = []
    pid_path = tmp_path / "configured-board-master.pid"
    pid_path.write_text("4343\n", encoding="ascii")
    pid_path.chmod(0o600)

    def runner(command, **kwargs):
        calls.append(command)
        if "--property=LoadState" in command:
            return _completed(
                stdout="LoadState=not-found\nActiveState=inactive\n"
            )
        if command[1:3] == ["--user", "show"]:
            return _completed(
                stdout=(
                    "MainPID=4242\n"
                    "ActiveState=active\n"
                    "SubState=running\n"
                    "Type=forking\n"
                    f"PIDFile={pid_path}\n"
                    "GuessMainPID=no\n"
                )
            )
        return _completed()

    with pytest.raises(DurableProcessError, match="exact PID-file"):
        launch_systemd_user_service(
            ["/bin/true"],
            unit_name="vrif-supervisor",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            forking_pid_file=pid_path,
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=runner,
        )

    assert calls[-1] == [
        "/usr/bin/systemctl",
        "--user",
        "stop",
        "vrif-supervisor.service",
    ]


def test_forking_launch_rejects_non_private_pid_file_and_stops_unit(
    tmp_path,
) -> None:
    calls = []
    pid_path = tmp_path / "configured-board-master.pid"
    pid_path.write_text("4242\n", encoding="ascii")
    pid_path.chmod(0o644)

    def runner(command, **kwargs):
        calls.append(command)
        if "--property=LoadState" in command:
            return _completed(
                stdout="LoadState=not-found\nActiveState=inactive\n"
            )
        if command[1:3] == ["--user", "show"]:
            return _completed(
                stdout=(
                    "MainPID=4242\n"
                    "ActiveState=active\n"
                    "SubState=running\n"
                    "Type=forking\n"
                    f"PIDFile={pid_path}\n"
                    "GuessMainPID=no\n"
                )
            )
        return _completed()

    with pytest.raises(DurableProcessError, match="exact PID-file"):
        launch_systemd_user_service(
            ["/bin/true"],
            unit_name="vrif-supervisor",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            forking_pid_file=pid_path,
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=runner,
        )

    assert calls[-1] == [
        "/usr/bin/systemctl",
        "--user",
        "stop",
        "vrif-supervisor.service",
    ]


def test_forking_launch_timeout_stops_exact_possibly_created_unit(
    tmp_path,
) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if "--property=LoadState" in command:
            return _completed(
                stdout="LoadState=not-found\nActiveState=inactive\n"
            )
        if command[0].endswith("systemd-run"):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return _completed()

    with pytest.raises(DurableProcessError, match="timed out"):
        launch_systemd_user_service(
            ["/bin/true"],
            unit_name="vrif-supervisor",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            forking_pid_file=tmp_path / "configured-board-master.pid",
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=runner,
        )

    assert calls[-1] == [
        "/usr/bin/systemctl",
        "--user",
        "stop",
        "vrif-supervisor.service",
    ]


def test_forking_launch_rejects_relative_pid_file_before_service_creation(
    tmp_path,
) -> None:
    calls = []

    with pytest.raises(DurableProcessError, match="must be an absolute path"):
        launch_systemd_user_service(
            ["/bin/true"],
            unit_name="vrif-supervisor",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            forking_pid_file=Path("relative-master.pid"),
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=lambda command, **kwargs: calls.append(command),
        )

    assert calls == []


def test_forking_launch_refuses_preexisting_unit_without_stopping_it(
    tmp_path,
) -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return _completed(stdout="LoadState=loaded\nActiveState=active\n")

    with pytest.raises(DurableProcessError, match="must be absent"):
        launch_systemd_user_service(
            ["/bin/true"],
            unit_name="already-running",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            forking_pid_file=tmp_path / "master.pid",
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=runner,
        )

    assert len(calls) == 1
    assert calls[0][0] == "/usr/bin/systemctl"
    assert "stop" not in calls[0]


def test_forking_launch_rejects_forwarded_quack_token_before_service_creation(
    tmp_path,
) -> None:
    calls = []
    secret = "must-never-enter-systemd-run-argv"

    with pytest.raises(DurableProcessError, match="one-use vault") as exc_info:
        launch_systemd_user_service(
            ["/bin/true"],
            unit_name="vrif-supervisor",
            working_directory=tmp_path,
            log_path=tmp_path / "service.log",
            environment={"IPFS_ACCELERATE_AGENT_QUACK_TOKEN": secret},
            forking_pid_file=tmp_path / "configured-board-master.pid",
            systemd_run_path="/usr/bin/systemd-run",
            systemctl_path="/usr/bin/systemctl",
            runner=lambda command, **kwargs: calls.append(command),
        )

    assert secret not in str(exc_info.value)
    assert calls == []


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
                "--forking-pid-file",
                str(tmp_path / "master.pid"),
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
    assert captured["forking_pid_file"] == tmp_path / "master.pid"
    assert '"pid": 99' in capsys.readouterr().out
