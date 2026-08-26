"""Hermetic contracts for the two EAAEF state-owner operator scripts."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    OWNER_LOCK_SUFFIX,
    OWNER_MARKER_SUFFIX,
    ExclusiveOwnerLease,
    QuackStateServerOwnershipError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
START_OWNER = REPO_ROOT / "scripts/start_eaaef_quack_owner.py"
HOST_ADMISSION = REPO_ROOT / "scripts/run_eaaef_host_admission_supervisor.py"
HELD_BOARD_INGEST = REPO_ROOT / "scripts/ingest_eaaef_held_board_into_duckdb.py"


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


@pytest.mark.parametrize("script", [START_OWNER, HOST_ADMISSION, HELD_BOARD_INGEST])
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
    ingest = _load_script(HELD_BOARD_INGEST, "tested_ingest_eaaef_held_board")

    assert vars(owner._parse_args([])) == {}
    assert vars(host._parse_args([])) == {}
    assert vars(ingest._parse_args([])) == {}


def test_live_quack_lease_makes_held_ingest_fail_before_receipt_or_db_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingest = _load_script(HELD_BOARD_INGEST, "tested_ingest_loses_owner_race")
    database = tmp_path / "run-v-test/control.duckdb"
    receipt = tmp_path / "held-board-catalog.json"
    calls: list[str] = []
    owner = _lease(database, server_id="quack:test-ingest-winner")
    try:
        monkeypatch.setattr(ingest, "_active_control_db", lambda: database)
        monkeypatch.setattr(ingest, "RECEIPT_PATH", receipt)
        monkeypatch.setattr(
            ingest,
            "_ingest_under_lease",
            lambda _control: calls.append("database_or_receipt") or {},
        )

        with pytest.raises(QuackStateServerOwnershipError, match="exclusive lock"):
            ingest.ingest()

        assert calls == []
        assert not database.exists()
        assert not receipt.exists()
    finally:
        owner.release(fence_token=owner.fence_token)


@pytest.mark.parametrize("scope", ("early_frontier", "full_bootstrap"))
def test_legacy_host_writer_fails_closed_before_owner_contention_or_effects(
    scope: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _load_script(
        HOST_ADMISSION,
        f"tested_disabled_host_writer_{scope}",
    )
    database = tmp_path / "run-v-test/control.duckdb"
    status = tmp_path / "status.json"
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            calls.append(name)
            raise AssertionError(f"disabled host writer reached {name}")

        return fail

    for name in (
        "_active_control_db",
        "_acquire_state_owner_lease",
        "_collect_host_admission",
        "_collect_full_host_admission",
        "_current_host_admission_identity",
        "_database_task_source_class",
        "_write_status",
    ):
        monkeypatch.setattr(host, name, forbidden(name))
    monkeypatch.setattr(host, "STATUS_PATH", status)

    with pytest.raises(
        RuntimeError,
        match="legacy mutable EAAEF host-admission scope is disabled",
    ):
        host.run_once(scope=scope)

    assert calls == []
    assert not database.exists()
    assert not status.exists()
