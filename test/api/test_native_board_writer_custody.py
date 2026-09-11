"""Board observations retain readiness separately from canonical writer custody."""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pytest

from test.api.test_agent_supervisor_live_board_probe import probe


@pytest.fixture
def native_board(tmp_path, monkeypatch):
    database = tmp_path / "control.duckdb"
    writer = duckdb.connect(str(database))
    writer.execute("CREATE TABLE preserved(value INTEGER)")
    writer.execute("INSERT INTO preserved VALUES (7)")
    writer.execute("CHECKPOINT")
    birth = probe.process_identity(os.getpid())
    status = tmp_path / "owner.json"
    status.write_text(
        json.dumps(
            {
                "identity": {"process_birth": birth},
                "lifecycle": "ready",
                "database_path": str(database),
            }
        )
    )
    board = {
        "board_id": "spar",
        "cwd": str(tmp_path),
        "max_lanes": 0,
        "state_root": str(tmp_path / "state"),
        "owner_status_path": str(status),
        "database_path": str(database),
        "quack_endpoint": "quack:127.0.0.1:1234",
        "status_argv": ["disposable-native-status"],
        "ensure_argv": ["never-restart"],
    }
    monkeypatch.setattr(probe, "_source_heads", lambda _: {".": "a" * 40})
    monkeypatch.setattr(probe, "_provider_busy", lambda *_: [])
    monkeypatch.setattr(
        probe,
        "_status_command",
        lambda _: (
            {
                "task_authority": {
                    "authenticated_query": True,
                    "status_counts": {"todo": 1},
                    "task_count": 1,
                }
            },
            "",
        ),
    )

    class Connected:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(probe.socket, "create_connection", lambda *a, **k: Connected())
    try:
        yield board, writer, database, birth
    finally:
        writer.close()


@pytest.mark.parametrize("board_id", ["spar", "sawm"])
def test_real_canonical_close_loss_cannot_remain_a_healthy_board(
    native_board, board_id
):
    board, writer, database, _birth = native_board
    board["board_id"] = board_id
    before = probe.observe_board(board)
    assert before["health"] == "healthy"
    # Exact pre-existing copier failure: any descriptor close on this inode
    # drops the process's POSIX locks, despite the retained DuckDB writer.
    descriptor = os.open(database, os.O_RDONLY)
    os.close(descriptor)
    writer.execute("INSERT INTO preserved VALUES (8)")
    after = probe.observe_board(board)
    assert after["details"]["owner_ready"] is True
    assert after["health"] == "degraded"
    assert "canonical_writer_lock_missing" in after["reason_codes"]
    assert after.get("recovery_action") != "ensure"
    assert after["complete"] is False


@pytest.mark.parametrize("namespace_visible", [True, False])
def test_exact_positive_lock_is_separate_from_namespace_visibility(
    native_board, monkeypatch, namespace_visible
):
    board, _writer, _database, _birth = native_board
    monkeypatch.setattr(
        probe.writer_custody,
        "observe_owner_namespaces",
        lambda _pid: {"verified": namespace_visible},
    )
    result = probe.observe_board(board)
    assert result["health"] == "healthy"
    assert result["details"]["owner_writer_custody"] == {
        "configured": True,
        "verified": True,
        "held": True,
        "namespace_verified": namespace_visible,
    }
    assert result.get("recovery_action") != "ensure"


@pytest.mark.parametrize(
    "namespace_result", [None, {"verified": False}, {"verified": True}]
)
def test_lost_lock_routes_repair_without_stop_authority(
    native_board, monkeypatch, namespace_result
):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog

    board, _writer, database, _birth = native_board
    os.close(os.open(database, os.O_RDONLY))
    monkeypatch.setattr(
        probe.writer_custody, "observe_owner_namespaces", lambda _pid: namespace_result
    )
    result = probe.observe_board(board)
    custody = result["details"]["owner_writer_custody"]
    visible = namespace_result == {"verified": True}
    assert custody["verified"] is visible
    assert custody.get("held") is (False if visible else None)
    reason = (
        "canonical_writer_lock_missing"
        if visible
        else "canonical_writer_lock_namespace_unverified"
    )
    assert reason in result["reason_codes"]
    assert result["details"]["owner_ready"] is True
    assert result.get("recovery_action") != "ensure"
    assert result["complete"] is False
    action = fleet_watchdog.select_action(
        {"health": result["health"], "observation": result, "incident_since": 0},
        {"ensure": {"argv": ["must-not-run"]}},
        now=1000,
    )
    assert action == "repair"


def _observe(database, birth, **kwargs):
    return probe.writer_custody.observe_canonical_writer_lock(
        database,
        birth,
        process_identity=kwargs.pop("process_identity", probe.process_identity),
        birth_matches=probe.birth_matches,
        namespaces=kwargs.pop("namespaces", lambda _pid: {"verified": True}),
        **kwargs,
    )


def _lock_row(database, birth, *, inode=None, pid=None, kind="WRITE", blocked=False):
    info = database.lstat()
    return (
        f"1: {'-> ' if blocked else ''}POSIX ADVISORY {kind} "
        f"{birth['pid'] if pid is None else pid} "
        f"{os.major(info.st_dev):x}:{os.minor(info.st_dev):x}:"
        f"{info.st_ino if inode is None else inode} 0 EOF\n"
    )


@pytest.mark.parametrize(
    "other", ["wal", "replica", "wrong_pid", "read", "blocked", "empty"]
)
def test_other_lock_is_never_canonical_writer_custody(native_board, other):
    _board, _writer, database, birth = native_board
    if other in {"wal", "replica"}:
        other_path = database.with_suffix(
            ".wal" if other == "wal" else ".replica.duckdb"
        )
        other_path.write_bytes(b"different file")
        row = _lock_row(other_path, birth)
    elif other == "wrong_pid":
        row = _lock_row(database, birth, pid=birth["pid"] + 1)
    elif other == "read":
        row = _lock_row(database, birth, kind="READ")
    elif other == "blocked":
        row = _lock_row(database, birth, blocked=True)
    else:
        row = ""
    assert _observe(database, birth, kernel_locks=lambda: row) == {
        "verified": True,
        "held": False,
        "namespace_verified": True,
    }


@pytest.mark.parametrize("failure", ["permission", "malformed", "oversized"])
def test_unknown_kernel_snapshot_is_not_absence(native_board, failure):
    _board, _writer, database, birth = native_board

    def snapshot():
        if failure == "permission":
            raise PermissionError("private error must not leak")
        if failure == "oversized":
            return probe.writer_custody.read_kernel_locks(maximum_bytes=1)
        return "invalid private data"

    result = _observe(database, birth, kernel_locks=snapshot)
    assert result == {
        "verified": False,
        "reason": "canonical_writer_lock_observation_unavailable",
    }


@pytest.mark.parametrize("drift", ["birth", "inode", "namespace"])
def test_final_checks_include_changes_during_namespace_observation(native_board, drift):
    _board, _writer, database, birth = native_board
    state = {"calls": 0, "changed": False}

    def namespaces(_pid):
        state["calls"] += 1
        if state["calls"] == 2:
            state["changed"] = True
            if drift == "inode":
                database.rename(database.with_suffix(".preserved.duckdb"))
                database.write_bytes(b"replacement must not receive custody")
        return {
            "verified": True,
            "identities": {"pid": str(state["calls"] if drift == "namespace" else 1)},
        }

    def identity(_pid):
        return (
            {**birth, "start_time_ticks": birth["start_time_ticks"] + 1}
            if drift == "birth" and state["changed"]
            else birth
        )

    result = _observe(
        database,
        birth,
        kernel_locks=lambda: "",
        namespaces=namespaces,
        process_identity=identity,
    )
    assert result["verified"] is False
    assert "held" not in result
    assert (
        result["reason"]
        == {
            "birth": "native_identity_changed_during_writer_lock_probe",
            "inode": "canonical_database_changed_during_writer_lock_probe",
            "namespace": "canonical_writer_lock_namespace_unverified",
        }[drift]
    )


@pytest.mark.parametrize(
    "bad_path", ["missing_status_binding", "wrong_status_binding", "symlink"]
)
def test_board_requires_exact_canonical_path_binding(native_board, bad_path):
    board, _writer, database, _birth = native_board
    status_path = Path(board["owner_status_path"])
    status = json.loads(status_path.read_text())
    if bad_path == "missing_status_binding":
        status.pop("database_path")
    elif bad_path == "wrong_status_binding":
        status["database_path"] = str(database.with_suffix(".wrong.duckdb"))
    else:
        link = database.with_suffix(".link.duckdb")
        link.symlink_to(database)
        board["database_path"] = status["database_path"] = str(link)
    status_path.write_text(json.dumps(status))
    result = probe.observe_board(board)
    assert result["health"] == "degraded"
    assert result["details"]["owner_ready"] is True
    assert result["details"]["owner_writer_custody"]["verified"] is False
    assert result.get("recovery_action") != "ensure"


def test_observation_never_opens_canonical_database(native_board, monkeypatch):
    board, writer, database, _birth = native_board
    original_open = os.open

    def guarded_open(path, *args, **kwargs):
        assert os.fspath(path) != str(database), "observation opened canonical DB"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    result = probe.observe_board(board)
    assert result["details"]["owner_writer_custody"]["held"] is True
    assert writer.execute("SELECT value FROM preserved").fetchone()[0] == 7


@pytest.mark.parametrize(
    "failure", ["pid_mismatch", "mnt_mismatch", "denied", "malformed"]
)
def test_namespace_observation_requires_exact_visible_pid_and_mount_namespaces(
    monkeypatch, failure
):
    def readlink(path):
        kind = Path(path).name
        if failure == "denied":
            raise PermissionError("namespace inaccessible")
        if failure == "malformed":
            return "arbitrary"
        owner = "/self/" not in str(path)
        different = owner and failure == kind + "_mismatch"
        return f"{kind}:[{2 if different else 1}]"

    monkeypatch.setattr(os, "readlink", readlink)
    assert probe.writer_custody.observe_owner_namespaces(42) == {"verified": False}


def test_standalone_probe_help_imports_shared_helper_without_package():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(Path(probe.__file__)), "--help"],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("custody_state", ["held", "missing", "unknown"])
def test_completed_counts_require_writer_custody_before_closeout_review(
    native_board, monkeypatch, custody_state
):
    from ipfs_accelerate_py.agent_supervisor.rescue import fleet_watchdog

    board, _writer, database, _birth = native_board
    monkeypatch.setattr(
        probe,
        "_status_command",
        lambda _: (
            {
                "task_authority": {
                    "authenticated_query": True,
                    "status_counts": {"completed": 1},
                    "task_count": 1,
                }
            },
            "",
        ),
    )
    monkeypatch.setattr(
        probe.writer_custody,
        "observe_owner_namespaces",
        lambda _pid: {"verified": custody_state != "unknown"},
    )
    if custody_state != "held":
        os.close(os.open(database, os.O_RDONLY))
    result = probe.observe_board(board)
    assert result["details"]["task_counts"] == {"completed": 1}
    assert result["details"]["owner_ready"] is True
    assert result["complete"] is False
    assert result["completion_candidate"] is (custody_state == "held")
    action = fleet_watchdog.select_action(
        {"health": result["health"], "observation": result, "incident_since": 0},
        {"ensure": {"argv": ["must-not-run"]}},
        now=1000,
    )
    assert action == ("completion_review" if custody_state == "held" else "repair")
