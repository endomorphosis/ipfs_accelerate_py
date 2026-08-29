"""Focused tests for non-plan-bound state-owner listener inheritance."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import (
    multi_supervisor_runner as runner,
)


def _track(
    tmp_path: Path,
    script: Path,
    *,
    name: str = "implementation",
    extra_args: tuple[str, ...] = (),
) -> runner.SupervisorTrack:
    state = tmp_path / "state" / name
    return runner.SupervisorTrack(
        name=name,
        script_path=script,
        log_path=state / "supervisor.log",
        supervisor_pid_path=state / "supervisor.pid",
        daemon_pid_path=state / "daemon.pid",
        supervisor_status_path=state / "status.json",
        extra_args=extra_args,
    )


def _bootstrap_args(descriptor: int, *, owner: str = "campaign") -> tuple[str, ...]:
    return (
        "--database-owner-session-id",
        owner,
        "--state-owner-bootstrap-fd",
        str(descriptor),
        "--state-owner-bootstrap-store-id",
        "campaign-store",
    )


def test_non_plan_bound_track_inherits_state_owner_unix_listener(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "child-result.json"
    script = tmp_path / "bootstrap-child.py"
    script.write_text(
        "import json, socket, sys\n"
        "from pathlib import Path\n"
        "fd = int(sys.argv[sys.argv.index('--state-owner-bootstrap-fd') + 1])\n"
        "result = Path(sys.argv[sys.argv.index('--result-path') + 1])\n"
        "listener = socket.socket(fileno=fd)\n"
        "payload = {\n"
        "    'family': int(listener.family),\n"
        "    'listening': listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN),\n"
        "}\n"
        "result.write_text(json.dumps(payload), encoding='utf-8')\n",
        encoding="utf-8",
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(tmp_path / "state-owner.sock"))
        listener.listen(4)
        track = _track(
            tmp_path,
            script,
            extra_args=("--result-path", str(result_path)),
        )

        process = runner.start_track(
            track,
            repo_root=tmp_path,
            common_args=_bootstrap_args(listener.fileno()),
            python_executable=sys.executable,
            output=lambda _message: None,
        )
        return_code = process.wait(timeout=10.0)

        assert return_code == 0, track.log_path.read_text(encoding="utf-8")
        assert json.loads(result_path.read_text(encoding="utf-8")) == {
            "family": int(socket.AF_UNIX),
            "listening": 1,
        }
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) == 1
    finally:
        listener.close()


def test_non_plan_bound_bootstrap_seals_lane_for_embedded_sidecar_lock(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        _open_database_writer_lock,
    )

    script = tmp_path / "bootstrap-child.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    track = _track(tmp_path, script)
    lane = track.log_path.parent
    lane.mkdir(parents=True)
    lane.chmod(0o775)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(tmp_path / "state-owner.sock"))
        listener.listen(1)
        process = runner.start_track(
            track,
            repo_root=tmp_path,
            common_args=_bootstrap_args(listener.fileno()),
            python_executable=sys.executable,
            output=lambda _message: None,
        )

        assert process.wait(timeout=10.0) == 0
        assert stat.S_IMODE(os.lstat(lane).st_mode) == 0o700
        with _open_database_writer_lock(lane / ".sidecar.writer.lock"):
            pass
    finally:
        listener.close()


def test_non_plan_bound_bootstrap_fd_rejects_malformed_and_duplicate_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = _track(tmp_path, tmp_path / "never-started.py")
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("rejected profile reached Popen"),
    )

    cases = (
        (("--state-owner-bootstrap-fd",), "missing its launch-profile value"),
        (
            ("--state-owner-bootstrap-fd", "not-an-integer"),
            "must name an integer descriptor",
        ),
        (
            ("--state-owner-bootstrap-fd", "2"),
            "must be an open descriptor >= 3",
        ),
    )
    for common_args, message in cases:
        with pytest.raises(ValueError, match=message):
            runner.start_track(
                track,
                repo_root=tmp_path,
                common_args=common_args,
                python_executable=sys.executable,
                output=lambda _message: None,
            )

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(tmp_path / "duplicate.sock"))
        listener.listen(1)
        descriptor = str(listener.fileno())
        with pytest.raises(ValueError, match="must appear exactly once"):
            runner.start_track(
                track,
                repo_root=tmp_path,
                common_args=(
                    "--state-owner-bootstrap-fd",
                    descriptor,
                    f"--state-owner-bootstrap-fd={descriptor}",
                ),
                python_executable=sys.executable,
                output=lambda _message: None,
            )
        with pytest.raises(
            ValueError,
            match="--database-owner-session-id must appear exactly once",
        ):
            runner.start_track(
                track,
                repo_root=tmp_path,
                common_args=(
                    "--state-owner-bootstrap-fd",
                    descriptor,
                ),
                python_executable=sys.executable,
                output=lambda _message: None,
            )
        with pytest.raises(
            ValueError,
            match="--database-owner-session-id must appear exactly once",
        ):
            runner.start_track(
                track,
                repo_root=tmp_path,
                common_args=(
                    "--database-owner-session-id",
                    "first",
                    "--database-owner-session-id=second",
                    "--state-owner-bootstrap-fd",
                    descriptor,
                ),
                python_executable=sys.executable,
                output=lambda _message: None,
            )
    finally:
        listener.close()


def test_non_plan_bound_bootstrap_fd_rejects_closed_non_socket_and_non_listener(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    track = _track(tmp_path, tmp_path / "never-started.py")
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("rejected descriptor reached Popen"),
    )

    closed_read, closed_write = os.pipe()
    os.close(closed_read)
    try:
        with pytest.raises(ValueError, match="is not an open descriptor"):
            runner.start_track(
                track,
                repo_root=tmp_path,
                common_args=_bootstrap_args(closed_read),
                python_executable=sys.executable,
                output=lambda _message: None,
            )
    finally:
        os.close(closed_write)

    pipe_read, pipe_write = os.pipe()
    try:
        with pytest.raises(ValueError, match="AF_UNIX listening socket"):
            runner.start_track(
                track,
                repo_root=tmp_path,
                common_args=_bootstrap_args(pipe_read),
                python_executable=sys.executable,
                output=lambda _message: None,
            )
    finally:
        os.close(pipe_read)
        os.close(pipe_write)

    inet_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    unix_non_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        inet_listener.bind(("127.0.0.1", 0))
        inet_listener.listen(1)
        unix_non_listener.bind(str(tmp_path / "not-listening.sock"))
        for descriptor in (inet_listener.fileno(), unix_non_listener.fileno()):
            with pytest.raises(ValueError, match="AF_UNIX listening socket"):
                runner.start_track(
                    track,
                    repo_root=tmp_path,
                    common_args=_bootstrap_args(descriptor),
                    python_executable=sys.executable,
                    output=lambda _message: None,
                )
    finally:
        inet_listener.close()
        unix_non_listener.close()


def test_non_plan_bound_shards_receive_distinct_stable_owner_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "implementation.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    tracks = runner.expand_implementation_track_lanes(
        f"implementation|{script}|{tmp_path / 'lanes'}|campaign",
        lanes_per_track=2,
    )
    commands: list[tuple[str, ...]] = []
    pass_fds: list[tuple[int, ...]] = []

    def capture_popen(command, **kwargs):
        commands.append(tuple(command))
        pass_fds.append(tuple(kwargs["pass_fds"]))
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(runner.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(
        runner,
        "_capture_owned_popen_birth",
        lambda _process, _profile: object(),
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    common_args: tuple[str, ...]
    try:
        listener.bind(str(tmp_path / "shared-state-owner.sock"))
        listener.listen(4)
        common_args = _bootstrap_args(listener.fileno(), owner="campaign-owner")
        for track in tracks:
            runner.start_track(
                track,
                repo_root=tmp_path,
                common_args=common_args,
                python_executable=sys.executable,
                output=lambda _message: None,
            )
    finally:
        listener.close()

    owner_sessions = [
        runner._profile_option_values(command, "--database-owner-session-id")[0]
        for command in commands
    ]
    expected = [
        (
            f"campaign-owner:shard:{index}-of-2:track:"
            f"{hashlib.sha256(track.name.encode('utf-8')).hexdigest()[:12]}"
        )
        for index, track in enumerate(tracks)
    ]
    assert owner_sessions == expected
    assert len(set(owner_sessions)) == 2
    assert pass_fds == [(int(common_args[3]),), (int(common_args[3]),)]
    assert common_args[1] == "campaign-owner"


def test_non_plan_bound_unsharded_owner_session_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "implementation.py"
    script.write_text("raise SystemExit(0)\n", encoding="utf-8")
    track = runner.expand_implementation_track_lanes(
        f"implementation|{script}|{tmp_path / 'lane'}|campaign",
        lanes_per_track=1,
    )[0]
    captured: dict[str, object] = {}

    def capture_popen(command, **kwargs):
        captured["command"] = tuple(command)
        captured["pass_fds"] = tuple(kwargs["pass_fds"])
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(runner.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(
        runner,
        "_capture_owned_popen_birth",
        lambda _process, _profile: object(),
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(tmp_path / "single-state-owner.sock"))
        listener.listen(1)
        runner.start_track(
            track,
            repo_root=tmp_path,
            common_args=_bootstrap_args(
                listener.fileno(),
                owner="single-owner",
            ),
            python_executable=sys.executable,
            output=lambda _message: None,
        )
        assert runner._profile_option_values(
            captured["command"],  # type: ignore[arg-type]
            "--database-owner-session-id",
        ) == ("single-owner",)
        assert captured["pass_fds"] == (listener.fileno(),)
    finally:
        listener.close()
