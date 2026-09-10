"""Exercise actual DuckLake files, crash replay, and exclusive catalog custody."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.federation import ducklake_fleet as lake
from ipfs_accelerate_py.agent_supervisor.federation.fleet_history import project_view
from test.api.test_quacklake_catalog import fleet_view, observation


@pytest.fixture
def history(tmp_path):
    with duckdb.connect() as connection:
        installed = connection.execute("SELECT installed FROM duckdb_extensions() WHERE extension_name='ducklake'").fetchone()
        if not installed or not installed[0]:
            pytest.skip("qualified installed DuckLake extension required; tests never install")
    return tmp_path / "history"


def test_real_ducklake_reopen_replay_and_unavailable_source(history):
    view = fleet_view(observation(), observation("pcpr", available=False))
    with lake.open_history(history) as connection:
        project_view(connection, view)
        before = lake.inspect_history(connection)
    assert list((history / "parquet").rglob("*.parquet"))
    with lake.open_history(history) as connection:
        project_view(connection, view)
        assert lake.inspect_history(connection) == before
        project_view(connection, fleet_view(observation("spar", available=False, stamp="2026-09-10T01:00:00+00:00")))
    with lake.open_history(history, create=False) as connection:
        assert lake.inspect_history(connection)["stored_observations"] == 3
        assert connection.execute("SELECT count(*) FROM fleet_lake.fleet_source_observations WHERE completion_authority").fetchone()[0] == 0
        payload = connection.execute("SELECT payload_json FROM fleet_lake.fleet_source_observations WHERE source_id='pcpr'").fetchone()[0]
        assert json.loads(payload)["native_receipt"] == {}
        with pytest.raises(duckdb.InvalidInputException):
            connection.execute("DELETE FROM fleet_lake.fleet_source_observations")


def test_process_crash_releases_lock_and_committed_replay_is_idempotent(history):
    ready = history.parent / "ready"
    payload = history.parent / "view.json"
    payload.write_text(json.dumps(fleet_view(observation())))
    code = """
import json, sys, time
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.federation.ducklake_fleet import open_history
from ipfs_accelerate_py.agent_supervisor.federation.fleet_history import project_view
with open_history(Path(sys.argv[1])) as connection:
    project_view(connection, json.loads(Path(sys.argv[2]).read_text()))
    Path(sys.argv[3]).write_text('committed')
    print('ready', flush=True)
    time.sleep(30)
"""
    process = subprocess.Popen([sys.executable, "-c", code, str(history), str(payload), str(ready)], stdout=subprocess.PIPE, text=True)
    try:
        import select
        assert select.select([process.stdout], [], [], 15)[0]
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(BlockingIOError):
            with lake.open_history(history):
                pytest.fail("second writer entered the catalog")
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        with lake.open_history(history) as connection:
            project_view(connection, json.loads(payload.read_text()))
            assert lake.inspect_history(connection)["stored_observations"] == 1
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdout.close()


def test_invalid_view_leaves_existing_ducklake_history_unchanged(history):
    with lake.open_history(history) as connection:
        project_view(connection, fleet_view(observation()))
        before = lake.inspect_history(connection)
        invalid = fleet_view(observation("sawm"))
        invalid["completion_authority"] = True
        with pytest.raises(ValueError):
            project_view(connection, invalid)
        assert lake.inspect_history(connection) == before


def test_no_source_file_fallback_or_catalog_creation_after_native_read_failure(history, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["export", "--history-root", str(history), "--deployment", "/missing/deployment", "--inventory", "/missing/inventory"])
    def denied(*_args):
        raise PermissionError("native grant denied")
    monkeypatch.setattr(lake, "read_native_view", denied)
    assert lake.main() == 1
    assert not history.exists()
    assert json.loads(capsys.readouterr().out)["error_type"] == "PermissionError"


def test_inspection_never_initializes_missing_history(history):
    with pytest.raises(FileNotFoundError):
        with lake.open_history(history, create=False):
            pytest.fail("inspection created a missing catalog")
    assert not history.exists()


@pytest.mark.parametrize("kind", ["relative", "public", "symlink"])
def test_directory_admission(history, kind):
    if kind == "relative":
        root = Path("relative-history")
    elif kind == "public":
        history.mkdir(mode=0o755)
        root = history
    else:
        target = history.with_name("actual")
        target.mkdir(mode=0o700)
        history.symlink_to(target, target_is_directory=True)
        root = history
    with pytest.raises(ValueError):
        with lake.open_history(root):
            pytest.fail("invalid history directory admitted")
