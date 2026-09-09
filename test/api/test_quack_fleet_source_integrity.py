"""Aggregate observations apply the actual bounded Git guard before admission."""

import importlib.util
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import live_board_probe
from ipfs_accelerate_py.agent_supervisor.runtime import quack_fleet_observer as observer


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "native source"
    root.mkdir()
    runtime = root / "runtime.py"
    runtime.write_text("qualified = True\n")
    (root / "task.py").write_text("task = True\n")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "source",
    )
    database, config = tmp_path / "control.duckdb", tmp_path / "config.json"
    database.touch()
    config.write_text("{}")
    identity = {
        "database_uuid": "uuid:test",
        "generation": 1,
        "process_birth_id": "birth:owner",
        "listen_uri": "quack:127.0.0.1:7777",
        "process_birth": {"pid": 42},
    }
    status = {"lifecycle": "ready", "identity": identity}
    calls = []

    def query(board, birth):
        calls.append(board)
        spec = importlib.util.spec_from_file_location("native_runtime", runtime)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            {
                "status_age_seconds": 0,
                "task_authority": {
                    "available": True,
                    "authenticated_query": True,
                    "status_counts": {"completed": 51},
                },
            },
            "",
            1,
        )

    adapter = SimpleNamespace(
        read_json=lambda path: status,
        birth_matches=lambda actual, expected: actual == expected,
        process_identity=lambda pid: identity["process_birth"],
        _status_with_receipt_retry=query,
        _source_integrity=live_board_probe._source_integrity,
    )
    board = {
        "id": "test-board",
        "database_path": str(database),
        "config_path": str(config),
        "owner_status_path": str(tmp_path / "owner.json"),
        "quack_endpoint": identity["listen_uri"],
        "source_integrity_paths": [{"repository": str(root), "paths": ["runtime.py"]}],
    }
    return board, adapter, runtime, calls


def refused(value):
    assert value["availability"] == "unavailable"
    assert value["reason"] == "source_integrity_not_verified"
    assert value["native_receipt"] == {}
    assert value["completion_authority"] is False
    assert value["observed_at"]


def test_dirty_native_module_is_never_imported(source):
    board, adapter, runtime, calls = source
    marker = runtime.parent / "IMPORTED"
    runtime.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    refused(observer.read_native_source(board, adapter=adapter))
    assert calls == [] and not marker.exists()


def test_clean_control_plane_allows_unrelated_task_edits(source):
    board, adapter, runtime, calls = source
    (runtime.parent / "task.py").write_text("ordinary task work\n")
    result = observer.read_native_source(board, adapter=adapter)
    assert result["availability"] == "available" and len(calls) == 1
    assert result["native_receipt"]["authority"]["status_counts"] == {"completed": 51}
    assert result["completion_authority"] is False


def test_source_changed_during_query_discards_completed_counts(source):
    board, adapter, runtime, calls = source
    original = adapter._status_with_receipt_retry

    def query(board, birth):
        native = original(board, birth)
        runtime.write_text("unqualified = True\n")
        return native

    adapter._status_with_receipt_retry = query
    refused(observer.read_native_source(board, adapter=adapter))
    assert len(calls) == 1


@pytest.mark.parametrize("flag", ["assume-unchanged", "skip-worktree"])
def test_hidden_dirty_runtime_flags_prevent_native_import(source, flag):
    board, adapter, runtime, calls = source
    git(runtime.parent, "update-index", "--" + flag, "runtime.py")
    runtime.write_text("unqualified = True\n")
    refused(observer.read_native_source(board, adapter=adapter))
    assert calls == []


@pytest.mark.parametrize(
    "answer",
    [
        None,
        {},
        {"configured": False, "valid": True},
        {"configured": True, "valid": False},
        {"configured": True, "valid": "true"},
    ],
)
def test_invalid_guard_response_fails_closed(source, answer):
    board, adapter, _, calls = source
    adapter._source_integrity = lambda board: answer
    refused(observer.read_native_source(board, adapter=adapter))
    assert calls == []


def test_missing_guard_support_fails_closed_only_for_configured_sources(source):
    board, adapter, _, calls = source
    del adapter._source_integrity
    refused(observer.read_native_source(board, adapter=adapter))
    assert calls == []
    del board["source_integrity_paths"]
    assert (
        observer.read_native_source(board, adapter=adapter)["availability"]
        == "available"
    )
    assert len(calls) == 1


def test_guard_timeout_after_query_discards_native_receipt(source):
    board, adapter, _, calls = source
    checked = []

    def guard(board):
        checked.append(board)
        if len(checked) == 2:
            raise subprocess.TimeoutExpired("git", 5)
        return {"configured": True, "valid": True}

    adapter._source_integrity = guard
    refused(observer.read_native_source(board, adapter=adapter))
    assert len(calls) == 1 and len(checked) == 2


def test_malformed_configured_scope_cannot_silently_skip_guard(source):
    board, adapter, _, calls = source
    board["source_integrity_paths"] = []
    refused(observer.read_native_source(board, adapter=adapter))
    assert calls == []


def test_post_query_guard_does_not_extend_pctdd_receipt_lifetime(source, monkeypatch):
    from datetime import datetime, timedelta, timezone

    board, adapter, _, _ = source
    board["id"] = "pctdd"
    identity = adapter.read_json(None)["identity"]
    identity.update(server_id="server:test", store_id="store:test")
    native = {
        "schema": "ipfs_accelerate_py/agent-supervisor/parallel-content-sealing-proof-carrying-tdd-operator@1",
        "task_authority": {
            "available": True,
            "authenticated_query": True,
            "direct_database_file_open": False,
            "transport": "quack_loopback_token_attach",
            "identity": dict(identity),
            "status_counts": {"completed": 51},
        },
        "state_owner": {
            "lifecycle_consistent": True,
            "authoritative_lifecycle": {
                "available": True,
                "reason": "authenticated_live_quack_query",
                "direct_database_file_open": False,
                "latest": dict(identity),
            },
        },
    }
    initial = datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc)
    clock = [initial]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    checks = []

    def guard(board):
        checks.append(board)
        if len(checks) == 2:
            clock[0] += timedelta(seconds=5)
        return {"configured": True, "valid": True}

    monkeypatch.setattr(observer, "datetime", Clock)
    adapter._source_integrity = guard
    adapter._status_with_receipt_retry = lambda board, birth: (native, "", 1)
    result = observer.read_native_source(board, adapter=adapter)
    assert result["availability"] == "available"
    assert result["observed_at"] == (initial + timedelta(seconds=5)).isoformat()
    assert (
        result["native_receipt"]["valid_until"]
        == (initial + timedelta(seconds=30)).isoformat()
    )
