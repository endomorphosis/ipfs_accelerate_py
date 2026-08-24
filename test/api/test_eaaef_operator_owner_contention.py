"""Hermetic contracts for the two EAAEF state-owner operator scripts."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    OWNER_LOCK_SUFFIX,
    OWNER_MARKER_SUFFIX,
    ExclusiveOwnerLease,
    QuackStateServerOwnershipError,
    build_server,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
START_OWNER = REPO_ROOT / "scripts/start_eaaef_quack_owner.py"
HOST_ADMISSION = REPO_ROOT / "scripts/run_eaaef_host_admission_supervisor.py"


def _load_script(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _lease(database: Path, *, server_id: str) -> ExclusiveOwnerLease:
    lease = ExclusiveOwnerLease(
        lock_path=database.with_name(f".{database.name}{OWNER_LOCK_SUFFIX}"),
        marker_path=database.with_name(f".{database.name}{OWNER_MARKER_SUFFIX}"),
    )
    lease.acquire(
        server_id=server_id,
        process_birth=current_process_birth(),
        database_path=database,
        generation=1,
    )
    return lease


@pytest.mark.parametrize("script", [START_OWNER, HOST_ADMISSION])
@pytest.mark.parametrize(
    ("argv", "expected_returncode"),
    [(["--help"], 0), (["--not-a-real-option"], 2)],
)
def test_cli_inspection_is_cold_before_repository_imports_or_writes(
    tmp_path: Path,
    script: Path,
    argv: list[str],
    expected_returncode: int,
) -> None:
    isolated = tmp_path / script.name
    shutil.copy2(script, isolated)

    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(isolated), *argv],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == expected_returncode
    assert sorted(path.name for path in tmp_path.iterdir()) == [script.name]


def test_no_argument_parser_contract_remains_accepted() -> None:
    owner = _load_script(START_OWNER, "tested_start_eaaef_quack_owner")
    host = _load_script(HOST_ADMISSION, "tested_run_eaaef_host_admission")

    assert vars(owner._parse_args([])) == {}
    assert vars(host._parse_args([])) == {}


def test_live_quack_lease_makes_offline_writer_fail_before_receipt_or_db_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _load_script(HOST_ADMISSION, "tested_host_loses_owner_race")
    database = tmp_path / "run-v-test/control.duckdb"
    status = tmp_path / "status.json"
    calls: list[str] = []
    owner = _lease(database, server_id="quack:test-winner")
    try:
        monkeypatch.setattr(host, "_active_control_db", lambda: database)
        monkeypatch.setattr(host, "STATUS_PATH", status)
        monkeypatch.setattr(
            host,
            "_collect_host_admission",
            lambda: calls.append("receipt") or {"decisions": {}},
        )
        monkeypatch.setattr(
            host,
            "_database_task_source_class",
            lambda: calls.append("database") or object,
        )

        with pytest.raises(QuackStateServerOwnershipError, match="exclusive lock"):
            host.run_once()

        assert calls == []
        assert not database.exists()
        assert not status.exists()
    finally:
        owner.release(fence_token=owner.fence_token)


class _EmptyPage:
    tasks: tuple[Any, ...] = ()


class _EmptyTaskSource:
    opens: list[Path] = []

    def __init__(self, database: Path, *, install_schema: bool) -> None:
        assert install_schema is False
        self.database = Path(database)
        type(self).opens.append(self.database)

    def __enter__(self) -> _EmptyTaskSource:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def get_task(self, _alias: str) -> None:
        return None

    def ready_tasks(self, *, limit: int) -> _EmptyPage:
        assert limit == 1000
        return _EmptyPage()

    def list_tasks(self, *, limit: int) -> _EmptyPage:
        assert limit == 1000
        return _EmptyPage()


def test_offline_writer_lease_makes_quack_owner_fail_before_db_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _load_script(HOST_ADMISSION, "tested_host_wins_owner_race")
    database = tmp_path / "run-v-test/control.duckdb"
    marker = database.with_name(f".{database.name}{OWNER_MARKER_SUFFIX}")
    receipt = tmp_path / "host-receipt.json"
    status = tmp_path / "status.json"
    receipt_started = threading.Event()
    allow_receipt_to_finish = threading.Event()
    host_results: list[dict[str, Any]] = []
    host_errors: list[BaseException] = []
    quack_calls: list[str] = []
    _EmptyTaskSource.opens = []

    def collect_receipt() -> dict[str, Any]:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["server_id"].startswith("offline:eaaef-host-admission:")
        receipt.write_text("under-exclusive-owner-lease\n", encoding="utf-8")
        receipt_started.set()
        assert allow_receipt_to_finish.wait(timeout=10)
        return {"decisions": {}}

    monkeypatch.setattr(host, "_active_control_db", lambda: database)
    monkeypatch.setattr(host, "ROOT", tmp_path)
    monkeypatch.setattr(host, "STATUS_PATH", status)
    monkeypatch.setattr(host, "_collect_host_admission", collect_receipt)
    monkeypatch.setattr(
        host, "_database_task_source_class", lambda: _EmptyTaskSource
    )

    def run_host() -> None:
        try:
            host_results.append(host.run_once())
        except BaseException as exc:  # pragma: no cover - asserted below
            host_errors.append(exc)

    thread = threading.Thread(target=run_host, daemon=True)
    thread.start()
    assert receipt_started.wait(timeout=10)

    def forbidden(stage: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            quack_calls.append(stage)
            raise AssertionError(f"Quack loser reached {stage}")

        return fail

    server = build_server(
        database_path=database,
        state_dir=tmp_path / "quack-state",
        capability_probe=forbidden("capability_probe"),
        migrate=forbidden("migration"),
        connection_factory=forbidden("database_open"),
    )
    try:
        with pytest.raises(QuackStateServerOwnershipError, match="exclusive lock"):
            server.start()
    finally:
        allow_receipt_to_finish.set()
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert host_errors == []
    assert len(host_results) == 1
    assert host_results[0]["control_db"] == "run-v-test/control.duckdb"
    assert quack_calls == []
    assert _EmptyTaskSource.opens == [database]
    assert receipt.is_file()
    assert status.is_file()
    assert not database.exists()
    assert not marker.exists()
