"""Crash-recovery integration test for SqliteDurableExecutor@1 (MCPP-052).

Acceptance (todo MCPP-052 / plan gate 17 / ADR-0005):

* Start a multi-step durable task in a **real subprocess**
* Commit exactly one externally visible side effect
* Kill the subprocess (SIGKILL) — not an in-process reopen mock
* Restart against the same SQLite journal, call real ``recover()``
  (do not mock recover)
* Resume remaining work without re-applying the committed effect
* Finalize with exactly one authoritative completion receipt

Gate 17 refinement: a successful in-memory retry is **not** crash recovery.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.durable.journal import ADAPTER_ID
from ipfs_accelerate_py.mcp_server.mcplusplus.durable.sqlite_executor import (
    CRASH_RECOVERY_RECEIPT_SCHEMA,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    SqliteDurableExecutor,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

# ---------------------------------------------------------------------------
# Constants shared by parent test and subprocess workers
# ---------------------------------------------------------------------------

SIDE_EFFECT_KEY = "fx-invoice-send-once"
SIDE_EFFECT_KIND = "http_call"
STEP1_PROGRESS = "progress-step-1-after-side-effect"
STEP2_PROGRESS = "progress-step-2-after-resume"
FINALIZE_RESULT_LABEL = "result-terminal-success"
COMPLETION_IDEMPOTENCY = "finalize-after-crash-recovery"
START_IDEMPOTENCY = "crash-recovery-start-key"


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _base_req(method: str, request_id: str, **extra: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "method": method,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000),
    }
    body.update(extra)
    return body


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_observation(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSONL side-effect observation (external effect evidence)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load_observations(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _wait_for_file(path: Path, *, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {path}")


def _kill_process_tree(proc: subprocess.Popen[Any], *, timeout_s: float = 5.0) -> None:
    """Hard-kill a worker process (and its group when possible)."""

    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=timeout_s)


# ---------------------------------------------------------------------------
# Worker entrypoints (invoked as real OS processes via this module)
# ---------------------------------------------------------------------------


def _apply_side_effect_once(
    ex: SqliteDurableExecutor,
    *,
    execution_id: str,
    effect_key: str,
    effect_cid: str,
    observation_path: Path,
    phase: str,
) -> bool:
    """Apply the external side effect only when the journal has not committed it.

    Returns True when a new observation was written (effect applied now).
    """

    if ex.journal.is_side_effect_committed(execution_id, effect_key):
        return False
    _append_observation(
        observation_path,
        {
            "effect_key": effect_key,
            "effect_cid": effect_cid,
            "kind": SIDE_EFFECT_KIND,
            "phase": phase,
            "pid": os.getpid(),
            "applied_at_ms": int(time.time() * 1000),
        },
    )
    return True


def worker_phase_pre_crash(
    *,
    db_path: Path,
    observation_path: Path,
    ready_path: Path,
    hang_path: Path,
) -> int:
    """Start multi-step work, commit one side effect, then hang until killed."""

    envelope = _cid("envelope-crash-recovery-integration")
    effect_cid = _cid("effect-invoice-payload-v1")
    progress_cid = _cid(STEP1_PROGRESS)

    with SqliteDurableExecutor.open(db_path, emit_events=False) as ex:
        started = ex.start(
            _base_req(
                "start",
                "worker-start",
                envelope_cid=envelope,
                idempotency_key=START_IDEMPOTENCY,
                correlation_id="corr-crash-recovery",
            )
        )
        if not started.get("ok"):
            _write_json(
                ready_path.with_name("worker-error.json"),
                {"phase": "start", "result": started},
            )
            return 2
        execution_id = started["execution_id"]
        fence = int(started["fencing_token"])

        applied = _apply_side_effect_once(
            ex,
            execution_id=execution_id,
            effect_key=SIDE_EFFECT_KEY,
            effect_cid=effect_cid,
            observation_path=observation_path,
            phase="pre_crash",
        )
        if not applied:
            _write_json(
                ready_path.with_name("worker-error.json"),
                {"phase": "side_effect", "message": "expected first apply"},
            )
            return 3

        # Checkpoint journals the committed side effect (fence before more work).
        cp = ex.checkpoint(
            _base_req(
                "checkpoint",
                "worker-cp-1",
                execution_id=execution_id,
                fencing_token=fence,
                idempotency_key="cp-step-1-side-effect",
                progress_cid=progress_cid,
                committed_side_effects=[
                    {
                        "kind": SIDE_EFFECT_KIND,
                        "idempotency_key": SIDE_EFFECT_KEY,
                        "effect_cid": effect_cid,
                        "compensatable": True,
                    }
                ],
            )
        )
        if not cp.get("ok"):
            _write_json(
                ready_path.with_name("worker-error.json"),
                {"phase": "checkpoint", "result": cp},
            )
            return 4

        # Durable state is on disk; advertise identity for the parent kill/restart.
        _write_json(
            ready_path,
            {
                "execution_id": execution_id,
                "fencing_token": fence,
                "checkpoint_id": cp["checkpoint_id"],
                "journal_seq": cp["journal_seq"],
                "envelope_cid": envelope,
                "effect_cid": effect_cid,
                "progress_cid": progress_cid,
                "pid": os.getpid(),
                "side_effect_key": SIDE_EFFECT_KEY,
            },
        )

        # Hang until SIGKILL (real process death — not orderly close).
        hang_path.write_text("hanging\n", encoding="utf-8")
        while True:
            time.sleep(0.25)


def worker_phase_post_restart(
    *,
    db_path: Path,
    observation_path: Path,
    ready_path: Path,
    result_path: Path,
) -> int:
    """Restart worker: recover (real), resume, skip re-apply, finalize once."""

    meta = _read_json(ready_path)
    execution_id = str(meta["execution_id"])
    fence = int(meta["fencing_token"])
    checkpoint_id = str(meta["checkpoint_id"])
    effect_cid = str(meta["effect_cid"])
    envelope_cid = str(meta["envelope_cid"])

    with SqliteDurableExecutor.open(db_path, emit_events=False) as ex:
        # Real recover() — never mocked. Reconstructs from journal only.
        recovered = ex.recover(
            _base_req(
                "recover",
                "post-restart-recover",
                execution_id=execution_id,
                fencing_token=fence,
                after_kill=True,
            )
        )
        if not recovered.get("ok"):
            _write_json(result_path, {"ok": False, "phase": "recover", "result": recovered})
            return 10

        crash_receipt = recovered.get("crash_recovery_receipt") or {}
        if crash_receipt.get("schema") != CRASH_RECOVERY_RECEIPT_SCHEMA:
            _write_json(
                result_path,
                {
                    "ok": False,
                    "phase": "recover_receipt",
                    "crash_recovery_receipt": crash_receipt,
                },
            )
            return 11
        if SIDE_EFFECT_KEY not in (crash_receipt.get("side_effects_not_replayed") or []):
            _write_json(
                result_path,
                {
                    "ok": False,
                    "phase": "side_effects_not_replayed",
                    "crash_recovery_receipt": crash_receipt,
                },
            )
            return 12

        fence = int(recovered["recovered"][0]["fencing_token"])

        resumed = ex.resume(
            _base_req(
                "resume",
                "post-restart-resume",
                execution_id=execution_id,
                fencing_token=fence,
                from_checkpoint_id=checkpoint_id,
                after_recover=True,
            )
        )
        if not resumed.get("ok"):
            _write_json(result_path, {"ok": False, "phase": "resume", "result": resumed})
            return 13

        # Attempt to re-apply the same side effect after restart — must be a no-op.
        reapplied = _apply_side_effect_once(
            ex,
            execution_id=execution_id,
            effect_key=SIDE_EFFECT_KEY,
            effect_cid=effect_cid,
            observation_path=observation_path,
            phase="post_restart_must_skip",
        )
        if reapplied:
            _write_json(
                result_path,
                {
                    "ok": False,
                    "phase": "duplicate_side_effect",
                    "message": "side effect was re-applied after crash recovery",
                },
            )
            return 14

        # Continue multi-step work: second checkpoint without new external effects.
        cp2 = ex.checkpoint(
            _base_req(
                "checkpoint",
                "post-restart-cp-2",
                execution_id=execution_id,
                fencing_token=fence,
                idempotency_key="cp-step-2-resume",
                progress_cid=_cid(STEP2_PROGRESS),
                # Re-presenting the same key must not re-commit or re-apply.
                committed_side_effects=[
                    {
                        "kind": SIDE_EFFECT_KIND,
                        "idempotency_key": SIDE_EFFECT_KEY,
                        "effect_cid": effect_cid,
                        "compensatable": True,
                    }
                ],
            )
        )
        if not cp2.get("ok"):
            _write_json(result_path, {"ok": False, "phase": "checkpoint2", "result": cp2})
            return 15

        result_cid = _cid(FINALIZE_RESULT_LABEL)
        output_cid = _cid("output-terminal-1")
        finalized = ex.finalize(
            _base_req(
                "finalize",
                "post-restart-finalize",
                execution_id=execution_id,
                fencing_token=fence,
                terminal_status="succeeded",
                result_cid=result_cid,
                output_cids=[output_cid],
                idempotency_key=COMPLETION_IDEMPOTENCY,
            )
        )
        if not finalized.get("ok"):
            _write_json(result_path, {"ok": False, "phase": "finalize", "result": finalized})
            return 16

        effects = ex.journal.committed_side_effects(execution_id)
        records = ex.journal.list_records(execution_id)
        transitions = [r["transition"] for r in records]
        finalized_rows = [r for r in records if r["transition"] == "finalized"]

        _write_json(
            result_path,
            {
                "ok": True,
                "execution_id": execution_id,
                "envelope_cid": envelope_cid,
                "fencing_token": fence,
                "crash_recovery_receipt": crash_receipt,
                "finalize": {
                    "status": finalized["status"],
                    "receipt_cid": finalized["receipt_cid"],
                    "result_cid": finalized["result_cid"],
                    "terminal_status": finalized["terminal_status"],
                    "journal_seq": finalized["journal_seq"],
                },
                "committed_side_effect_keys": [
                    e["idempotency_key"] for e in effects
                ],
                "journal_transitions": transitions,
                "finalized_count": len(finalized_rows),
            },
        )
        return 0


def _cli_main(argv: List[str]) -> int:
    """Subprocess CLI: ``python this_module.py <phase> --db ...``."""

    if len(argv) < 2:
        print("usage: phase --db PATH --observation PATH ...", file=sys.stderr)
        return 64
    phase = argv[1]
    # Minimal argv parser (avoid argparse dependency quirks in -c workers).
    args: Dict[str, str] = {}
    i = 2
    while i < len(argv):
        key = argv[i]
        if not key.startswith("--") or i + 1 >= len(argv):
            print(f"bad argv near {key!r}", file=sys.stderr)
            return 64
        args[key[2:].replace("-", "_")] = argv[i + 1]
        i += 2

    db_path = Path(args["db"])
    observation_path = Path(args["observation"])
    ready_path = Path(args["ready"])

    if phase == "pre_crash":
        hang_path = Path(args["hang"])
        return worker_phase_pre_crash(
            db_path=db_path,
            observation_path=observation_path,
            ready_path=ready_path,
            hang_path=hang_path,
        )
    if phase == "post_restart":
        result_path = Path(args["result"])
        return worker_phase_post_restart(
            db_path=db_path,
            observation_path=observation_path,
            ready_path=ready_path,
            result_path=result_path,
        )
    print(f"unknown phase: {phase!r}", file=sys.stderr)
    return 64


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_crash_recovery_one_side_effect_and_one_completion_receipt(
    tmp_path: Path,
) -> None:
    """Kill a real subprocess mid-task; recover must not duplicate the effect."""

    work = tmp_path / "crash_recovery"
    work.mkdir()
    db_path = work / "durable.sqlite3"
    observation_path = work / "side_effect_observations.jsonl"
    ready_path = work / "pre_crash_ready.json"
    hang_path = work / "hanging.marker"
    result_path = work / "post_restart_result.json"
    pre_log = work / "pre_crash.stderr"
    post_log = work / "post_restart.stderr"

    module_path = Path(__file__).resolve()
    env = {
        **os.environ,
        # Ensure repo root imports resolve inside the worker process.
        "PYTHONPATH": os.pathsep.join(
            [
                str(module_path.parents[2]),
                os.environ.get("PYTHONPATH", ""),
            ]
        ).rstrip(os.pathsep),
    }

    pre_cmd = [
        sys.executable,
        str(module_path),
        "pre_crash",
        "--db",
        str(db_path),
        "--observation",
        str(observation_path),
        "--ready",
        str(ready_path),
        "--hang",
        str(hang_path),
    ]
    with pre_log.open("w", encoding="utf-8") as err_fh:
        pre = subprocess.Popen(
            pre_cmd,
            stdout=subprocess.DEVNULL,
            stderr=err_fh,
            env=env,
            start_new_session=True,
            cwd=str(module_path.parents[2]),
        )

    try:
        try:
            _wait_for_file(ready_path, timeout_s=20.0)
            _wait_for_file(hang_path, timeout_s=5.0)
        except TimeoutError:
            _kill_process_tree(pre)
            err = pre_log.read_text(encoding="utf-8") if pre_log.is_file() else ""
            pytest.fail(f"pre_crash worker never became ready.\nstderr:\n{err}")

        # Confirm the side effect was observed exactly once before the kill.
        pre_obs = _load_observations(observation_path)
        assert len(pre_obs) == 1, pre_obs
        assert pre_obs[0]["effect_key"] == SIDE_EFFECT_KEY
        assert pre_obs[0]["phase"] == "pre_crash"
        assert pre.poll() is None, "worker exited before kill"

        # Real process death (not orderly journal close in the parent).
        _kill_process_tree(pre)
        assert pre.poll() is not None
        # Negative return codes are common for SIGKILL; any exit is fine.
    finally:
        if pre.poll() is None:
            _kill_process_tree(pre)

    meta = _read_json(ready_path)
    execution_id = meta["execution_id"]

    post_cmd = [
        sys.executable,
        str(module_path),
        "post_restart",
        "--db",
        str(db_path),
        "--observation",
        str(observation_path),
        "--ready",
        str(ready_path),
        "--result",
        str(result_path),
    ]
    with post_log.open("w", encoding="utf-8") as err_fh:
        completed = subprocess.run(
            post_cmd,
            stdout=subprocess.DEVNULL,
            stderr=err_fh,
            env=env,
            cwd=str(module_path.parents[2]),
            timeout=30,
            check=False,
        )
    if completed.returncode != 0:
        err = post_log.read_text(encoding="utf-8") if post_log.is_file() else ""
        result_preview = (
            result_path.read_text(encoding="utf-8") if result_path.is_file() else ""
        )
        pytest.fail(
            f"post_restart worker failed rc={completed.returncode}\n"
            f"stderr:\n{err}\nresult:\n{result_preview}"
        )

    outcome = _read_json(result_path)
    assert outcome["ok"] is True, outcome
    assert outcome["execution_id"] == execution_id

    # --- Acceptance: exactly one side-effect observation ---
    observations = _load_observations(observation_path)
    assert len(observations) == 1, observations
    assert observations[0]["effect_key"] == SIDE_EFFECT_KEY
    assert observations[0]["phase"] == "pre_crash"
    assert outcome["committed_side_effect_keys"] == [SIDE_EFFECT_KEY]

    # --- Acceptance: exactly one authoritative completion receipt ---
    finalize = outcome["finalize"]
    assert finalize["status"] == "succeeded"
    assert finalize["terminal_status"] == "succeeded"
    assert finalize["receipt_cid"]
    assert finalize["result_cid"] == _cid(FINALIZE_RESULT_LABEL)
    assert outcome["finalized_count"] == 1

    crash_receipt = outcome["crash_recovery_receipt"]
    assert crash_receipt["schema"] == CRASH_RECOVERY_RECEIPT_SCHEMA
    assert crash_receipt["adapter_id"] == ADAPTER_ID
    assert execution_id in crash_receipt["execution_ids"]
    assert SIDE_EFFECT_KEY in crash_receipt["side_effects_not_replayed"]
    assert crash_receipt.get("receipt_cid")

    transitions = outcome["journal_transitions"]
    assert transitions[0] == "started"
    assert "checkpointed" in transitions
    assert "recovered" in transitions
    assert "resumed" in transitions
    assert transitions[-1] == "finalized"
    assert transitions.count("finalized") == 1
    assert transitions.count("recovered") == 1

    # Independent parent reopen: inspect authority matches worker outcome.
    with SqliteDurableExecutor.open(db_path, emit_events=False) as ex:
        inspected = ex.inspect(
            _base_req(
                "inspect",
                "parent-inspect",
                execution_id=execution_id,
                include_journal=True,
            )
        )
        assert inspected["ok"] is True
        assert inspected["status"] == "succeeded"
        assert inspected["receipt_cid"] == finalize["receipt_cid"]
        assert len(ex.journal.committed_side_effects(execution_id)) == 1
        assert inspected["schema"] == RESULT_SCHEMA

        # Mutators remain fail-closed after terminal finalize.
        again = ex.finalize(
            _base_req(
                "finalize",
                "parent-finalize-again",
                execution_id=execution_id,
                fencing_token=int(outcome["fencing_token"]),
                terminal_status="succeeded",
                result_cid=_cid(FINALIZE_RESULT_LABEL),
                idempotency_key="different-key-must-fail",
            )
        )
        assert again["ok"] is False
        assert again["error"]["code"] == "terminal_execution"

        # Idempotent finalize replay returns the same receipt (still one journaled finalize).
        replay = ex.finalize(
            _base_req(
                "finalize",
                "parent-finalize-idempotent",
                execution_id=execution_id,
                fencing_token=int(outcome["fencing_token"]),
                terminal_status="succeeded",
                result_cid=_cid(FINALIZE_RESULT_LABEL),
                output_cids=[_cid("output-terminal-1")],
                idempotency_key=COMPLETION_IDEMPOTENCY,
            )
        )
        assert replay["ok"] is True
        assert replay["receipt_cid"] == finalize["receipt_cid"]
        finalized_records = [
            r
            for r in ex.journal.list_records(execution_id)
            if r["transition"] == "finalized"
        ]
        assert len(finalized_records) == 1


def test_in_process_reopen_is_not_claimed_as_this_suite() -> None:
    """Document that MCPP-052 requires OS process kill, not in-memory reopen.

    Unit coverage for journal reopen lives in test_mcplusplus_durable_sqlite.py.
    This module's primary test always uses subprocess + SIGKILL.
    """

    source = Path(__file__).read_text(encoding="utf-8")
    assert "SIGKILL" in source
    assert "subprocess.Popen" in source
    assert "do not mock recover" in source.lower() or "never mocked" in source.lower()


if __name__ == "__main__":
    sys.exit(_cli_main(sys.argv))
