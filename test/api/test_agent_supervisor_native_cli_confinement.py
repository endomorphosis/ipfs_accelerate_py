"""Adversarial native provider CLI process-confinement tests."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    legacy_landed_provider_cli as native_cli,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    native_cli_subreaper,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc/self/task").is_dir(),
    reason="native CLI confinement requires Linux procfs",
)


def _process_is_live(process_id: int) -> bool:
    try:
        raw = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return False
    fields = raw[raw.rfind(")") + 2 :].split()
    return bool(fields) and fields[0] != "Z"


def test_successful_cli_cannot_leave_immediate_setsid_descendant(
    tmp_path: Path,
) -> None:
    descendant_pid_path = tmp_path / "descendant.pid"
    escaped_write_path = tmp_path / "escaped-write"
    descendant_script = (
        "import pathlib,time;"
        "time.sleep(0.4);"
        f"pathlib.Path({str(escaped_write_path)!r}).write_text('escaped');"
        "time.sleep(60)"
    )
    direct_script = (
        "import pathlib,subprocess,sys;"
        "child=subprocess.Popen("
        f"[sys.executable,'-c',{descendant_script!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL,close_fds=True,start_new_session=True);"
        f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid))"
    )

    stdout, stderr = native_cli._run_native_cli_process(
        [sys.executable, "-c", direct_script],
        cwd=tmp_path,
        timeout_seconds=5,
    )

    assert stdout == ""
    assert stderr == ""
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    time.sleep(0.6)
    assert not escaped_write_path.exists()
    assert not _process_is_live(descendant_pid)


def test_subreaper_preserves_stdout_stderr_and_exit_classification(
    tmp_path: Path,
) -> None:
    success_script = "import sys;sys.stdout.write('out');sys.stderr.write('err')"
    stdout, stderr = native_cli._run_native_cli_process(
        [sys.executable, "-c", success_script],
        cwd=tmp_path,
        timeout_seconds=5,
    )
    assert stdout == "out"
    assert stderr == "err"

    failed_script = "import sys;sys.stderr.write('private');raise SystemExit(17)"
    with pytest.raises(RuntimeError) as raised:
        native_cli._run_native_cli_process(
            [sys.executable, "-c", failed_script],
            cwd=tmp_path,
            timeout_seconds=5,
        )
    assert str(raised.value) == "legacy native provider command failed"
    assert "private" not in str(raised.value)


def test_wrapper_fails_before_launch_when_subreaper_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = False

    def forbidden_popen(*_args: object, **_kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal launched
        launched = True
        raise AssertionError("provider command must not run")

    monkeypatch.setattr(native_cli_subreaper, "_enable_child_subreaper", lambda: False)
    monkeypatch.setattr(native_cli_subreaper.subprocess, "Popen", forbidden_popen)

    result = native_cli_subreaper.main(["--", sys.executable, "-c", "pass"])

    assert result == native_cli_subreaper._CONFINEMENT_FAILURE_EXIT_CODE
    assert launched is False


def test_wrapper_passes_only_the_explicit_verified_fd(tmp_path: Path) -> None:
    read_fd, write_fd = os.pipe()
    inner = f"import os;os.write({write_fd},b'phase-frame')"
    result = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(Path(native_cli_subreaper.__file__).resolve(strict=True)),
            "--pass-fd",
            str(write_fd),
            "--",
            sys.executable,
            "-c",
            inner,
        ],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(write_fd,),
        start_new_session=True,
    )
    os.close(write_fd)
    assert result.wait(timeout=10) == 0
    assert os.read(read_fd, 64) == b"phase-frame"
    assert os.read(read_fd, 1) == b""
    os.close(read_fd)


@pytest.mark.parametrize("raw_fd", ["0", "01", "-1", "not-a-fd", "999999"])
def test_wrapper_rejects_malformed_or_unavailable_fd_before_launch(
    tmp_path: Path,
    raw_fd: str,
) -> None:
    marker = tmp_path / f"launched-{raw_fd.replace('/', '_')}"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(native_cli_subreaper.__file__).resolve(strict=True)),
            "--pass-fd",
            raw_fd,
            "--",
            sys.executable,
            "-c",
            f"import pathlib;pathlib.Path({str(marker)!r}).write_text('launched')",
        ],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=10,
    )
    assert result.returncode == native_cli_subreaper._CONFINEMENT_FAILURE_EXIT_CODE
    assert not marker.exists()


def test_explicit_fd_wrapper_reaps_detached_descendant_before_exit(
    tmp_path: Path,
) -> None:
    read_fd, write_fd = os.pipe()
    descendant_pid_path = tmp_path / "fd-descendant.pid"
    escaped_write_path = tmp_path / "fd-escaped-write"
    descendant_script = (
        "import pathlib,time;"
        "time.sleep(0.4);"
        f"pathlib.Path({str(escaped_write_path)!r}).write_text('escaped');"
        "time.sleep(60)"
    )
    direct_script = (
        "import os,pathlib,subprocess,sys;"
        "child=subprocess.Popen("
        f"[sys.executable,'-c',{descendant_script!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL,close_fds=True,start_new_session=True);"
        f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid));"
        f"os.write({write_fd},b'phase-frame')"
    )
    result = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(Path(native_cli_subreaper.__file__).resolve(strict=True)),
            "--pass-fd",
            str(write_fd),
            "--",
            sys.executable,
            "-c",
            direct_script,
        ],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        pass_fds=(write_fd,),
        start_new_session=True,
    )
    os.close(write_fd)
    assert result.wait(timeout=10) == 0
    assert os.read(read_fd, 64) == b"phase-frame"
    assert os.read(read_fd, 1) == b""
    os.close(read_fd)
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    time.sleep(0.6)
    assert not escaped_write_path.exists()
    assert not _process_is_live(descendant_pid)
