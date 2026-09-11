"""Preserve actual old native storage across graceful writer replacement.

Real DuckDB/native server lifecycle; the extension transport is substituted.
The old source checkout is an explicit qualification input, never a live path.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import sawm_writer_recovery as recovery

FIXTURE = Path(__file__).parent / "fixtures/sawm_owner_replacement_fixture.py"
CURRENT = Path(__file__).resolve().parents[2]


def test_old_native_unresolved_claim_survives_graceful_owner_copy_repair(tmp_path):
    old_text = os.environ.get("SAWM_DISPOSABLE_OLD_SOURCE", "")
    if not old_text:
        pytest.skip("explicit disposable old25f source checkout required")
    old = Path(old_text).resolve()
    assert str(old).startswith("/tmp/")
    assert (
        subprocess.check_output(
            ["git", "-C", str(old), "rev-parse", "HEAD"], text=True
        ).strip()
        == "25f130c664200a76db81d9f56ddebd0f4e0fb733"
    )
    assert not subprocess.check_output(
        ["git", "-C", str(old), "status", "--porcelain"], text=True
    ).strip()

    def command(source, action, *args):
        return [sys.executable, str(FIXTURE), str(source), str(tmp_path), action, *args]

    subprocess.run(command(old, "prepare"), check=True, timeout=60)
    with (tmp_path / "wrapper.log").open("wb") as log:
        wrapper = subprocess.Popen(command(old, "wrapper"), stdout=log, stderr=log)
        descriptor = os.pidfd_open(wrapper.pid)
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "wrapper-ready").exists():
                assert wrapper.poll() is None, (tmp_path / "wrapper.log").read_text()
                assert time.monotonic() < deadline
                time.sleep(0.02)
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            assert wrapper.wait(timeout=20) == 0, (tmp_path / "wrapper.log").read_text()
            assert (tmp_path / "wrapper-closed").exists()
        finally:
            if wrapper.poll() is None:
                wrapper.kill()
                wrapper.wait(timeout=5)
            os.close(descriptor)
    for source, suffix, reuse in [(old, "before", "new"), (CURRENT, "after", "reuse")]:
        with (tmp_path / ("process-" + suffix + ".log")).open("wb") as log:
            process = subprocess.Popen(
                command(source, "owner", reuse), stdout=log, stderr=log
            )
            descriptor = os.pidfd_open(process.pid)
            try:
                deadline = time.monotonic() + 60
                while not (tmp_path / ("owner-" + suffix + ".json")).exists():
                    assert process.poll() is None, (
                        tmp_path / ("process-" + suffix + ".log")
                    ).read_text()
                    assert time.monotonic() < deadline, "owner startup fixture deadline"
                    time.sleep(0.02)
                binding = recovery.process.observe_process(process.pid)
                if reuse == "new":
                    custody = recovery.canonical_writer_lost(
                        tmp_path / "control.duckdb", binding
                    )
                    assert custody["writer_missing_verified"] is True
                else:
                    with pytest.raises(
                        recovery.process.GracefulRecoveryUnverified,
                        match="canonical_writer_present",
                    ):
                        recovery.canonical_writer_lost(
                            tmp_path / "control.duckdb", binding
                        )
                native_status = json.loads(
                    (tmp_path / "owner" / "quack-state-server.status.json").read_text()
                )
                expected = {
                    "owner": {"pid": process.pid},
                    "owner_identity": native_status["identity"],
                    "custody": {
                        "canonical": recovery.observation._regular_identity(
                            tmp_path / "control.duckdb"
                        ),
                        "native_locks": [
                            recovery.observation._regular_identity(path)
                            for path in recovery.observation._locks(
                                tmp_path / "control.duckdb"
                            )
                        ],
                    },
                }
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                assert process.wait(timeout=10) == 0, (
                    tmp_path / ("process-" + suffix + ".log")
                ).read_text()
                assert json.loads(
                    (tmp_path / ("closed-" + suffix + ".json")).read_text()
                )["closed"]
                observed = recovery.observe_owner_closed(
                    tmp_path,
                    {
                        "quack_owner": {
                            "state_dir": "owner",
                            "database_path": "control.duckdb",
                        }
                    },
                    expected,
                )
                assert observed["native_stopped_identity_observed"] is True
                assert observed["callback_settlement_authority"] is False
            finally:
                if process.poll() is None:
                    # Disposable fixture teardown only; never the recovery mechanism.
                    process.kill()
                    process.wait(timeout=5)
                os.close(descriptor)
    before = json.loads((tmp_path / "owner-before.json").read_text())
    after = json.loads((tmp_path / "owner-after.json").read_text())
    assert before["protected"] == after["protected"]
    assert before["generation"] == after["generation"]
    assert before["database_uuid"] == after["database_uuid"]
    assert before["server_id"] != after["server_id"]
    assert before["writer_held_after_copy"] is False
    assert after["writer_held_after_copy"] is True
    subprocess.run(command(CURRENT, "verify"), check=True, timeout=30)
