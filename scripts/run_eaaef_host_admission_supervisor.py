#!/usr/bin/env python3
"""Host-controlled DuckDB bootstrap supervisor.

Completes ready S and A bootstrap tasks against the embedded DuckDB control
plane without configured-board live launch, provider invocation, or Docker-socket
mounts. Held Plan-R2 tasks stay blocked. Live multi-supervisor launch remains a
separate fail-closed gate.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
    RECEIPT_DIR,
    RECEIPT_FILES,
    collect_and_write,
)

CURSOR = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "generation-cursor.json"
)
STATUS_PATH = (
    ROOT
    / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
    / "host-admission-supervisor-status.json"
)
BOARD_PATH = (
    ROOT
    / "docs/architecture/external_agent_autonomous_execution_fabric"
    / "task_board.json"
)
S_AUTO = {f"EAAEF-{number}" for number in range(180, 191)}
A_AUTO = {f"EAAEF-{number:03d}" for number in range(0, 10)}
HOST_AUTO = S_AUTO | A_AUTO | {"EAAEF-191"}
BOOTSTRAP = HOST_AUTO
ADMIT_REQUIRED_AUTO = {"EAAEF-183", "EAAEF-184"}
ADMIT_WAIT_STATUS = {
    "EAAEF-183": "waiting_rootless_engine",
    "EAAEF-184": "waiting_signed_provider_authorization",
    "EAAEF-191": "waiting_signed_admission_bundle",
}
S_PYTEST = {
    "EAAEF-180": "inventory",
    "EAAEF-181": "principals",
    "EAAEF-182": "duckdb_quack",
    "EAAEF-183": "engine_mode",
    "EAAEF-184": "provider_authorization",
    "EAAEF-185": "worker_image",
    "EAAEF-186": "container_profile",
    "EAAEF-187": "worker_network",
    "EAAEF-188": "command_fabric",
    "EAAEF-189": "native_lane",
    "EAAEF-190": "plan_r2",
    "EAAEF-191": "admission_bundle",
}
MAX_PASSES = 24


def _cid_bytes(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _active_control_db() -> Path:
    if CURSOR.is_file():
        cursor = json.loads(CURSOR.read_text(encoding="utf-8"))
        generation = str(cursor.get("active_generation") or "eaaef-run-v7")
    else:
        generation = "eaaef-run-v7"
    number = generation.rsplit("-v", 1)[-1]
    return (
        ROOT
        / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
        / f"run-v{number}"
        / "control.duckdb"
    )


def _write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _board_validations(alias: str) -> list[dict[str, object]]:
    board = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    for task in board.get("tasks") or ():
        if str(task.get("stable_task_id") or "") == alias:
            return list(task.get("execution_validation") or [])
    return []


def _run_argv(argv: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _complete_s_task(source: DatabaseTaskSource, alias: str) -> dict:
    task = source.get_task(alias)
    if task is None:
        return {"task_id": alias, "status": "missing"}
    if task.status == "completed":
        return {"task_id": alias, "status": "already_completed"}
    receipt_name = RECEIPT_FILES[alias]
    receipt_path = RECEIPT_DIR / receipt_name
    if alias in ADMIT_REQUIRED_AUTO and receipt_path.is_file():
        current = json.loads(receipt_path.read_text(encoding="utf-8"))
        if current.get("decision") != "admitted":
            return {
                "task_id": alias,
                "status": ADMIT_WAIT_STATUS.get(alias, "waiting_admission"),
                "decision": current.get("decision"),
            }
    if alias == "EAAEF-191" and receipt_path.is_file():
        current = json.loads(receipt_path.read_text(encoding="utf-8"))
        evidence = current.get("evidence") or {}
        if (
            current.get("decision") not in {"no_go", "admitted"}
            or evidence.get("independent_signature_present") is not True
            or evidence.get("launch_plan_allowed") is True
        ):
            return {
                "task_id": alias,
                "status": ADMIT_WAIT_STATUS["EAAEF-191"],
                "decision": current.get("decision"),
            }
    validation = [
        "python3",
        "-m",
        "pytest",
        "-q",
        "test/api/test_eaaef_host_admission_unblocking.py",
        "-k",
        S_PYTEST[alias],
    ]
    completed = _run_argv(validation, ROOT, 180)
    digest = _cid_bytes(
        (completed.stdout + completed.stderr + str(completed.returncode)).encode()
    )
    source.record_validation_result(
        task_cid=task.task_cid,
        outcome="passed" if completed.returncode == 0 else "failed",
        evidence_digest=digest,
        argv=validation,
    )
    if completed.returncode != 0:
        return {
            "task_id": alias,
            "status": "validation_failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-400:],
        }
    if receipt_path.is_file():
        source.record_evidence(
            task_cid=task.task_cid,
            evidence_kind="host_admission_receipt",
            digest=_cid_bytes(receipt_path.read_bytes()),
            body={"path": str(receipt_path.relative_to(ROOT))},
        )
    result = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "completed",
        {"validation": "passed", "host_controlled": True},
        evidence_digests=[digest],
    )
    return {
        "task_id": alias,
        "status": result.task.status,
        "changed": result.changed,
        "receipt_cid": result.receipt_cid,
    }


def _complete_a_task(source: DatabaseTaskSource, alias: str) -> dict:
    task = source.get_task(alias)
    if task is None:
        return {"task_id": alias, "status": "missing"}
    if task.status == "completed":
        return {"task_id": alias, "status": "already_completed"}
    commands = _board_validations(alias)
    if not commands:
        return {"task_id": alias, "status": "missing_execution_validation"}
    digests: list[str] = []
    for item in commands:
        argv = [str(part) for part in item.get("argv") or ()]
        working = str(item.get("working_directory") or ".")
        if not argv:
            return {"task_id": alias, "status": "empty_execution_validation"}
        cwd = ROOT if working in {".", ""} else ROOT / working
        completed = _run_argv(argv, cwd, 1800)
        digest = _cid_bytes(
            (completed.stdout + completed.stderr + str(completed.returncode)).encode()
        )
        digests.append(digest)
        source.record_validation_result(
            task_cid=task.task_cid,
            outcome="passed" if completed.returncode == 0 else "failed",
            evidence_digest=digest,
            argv=argv,
        )
        if completed.returncode != 0:
            return {
                "task_id": alias,
                "status": "validation_failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout[-1200:],
                "stderr": completed.stderr[-800:],
                "argv": argv,
            }
    result = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "completed",
        {"validation": "passed", "host_controlled": True, "duckdb": True},
        evidence_digests=digests,
    )
    return {
        "task_id": alias,
        "status": result.task.status,
        "changed": result.changed,
        "receipt_cid": result.receipt_cid,
    }


def _complete(source: DatabaseTaskSource, alias: str) -> dict:
    if alias in RECEIPT_FILES:
        return _complete_s_task(source, alias)
    return _complete_a_task(source, alias)


def run_once() -> dict:
    collection = collect_and_write()
    control = _active_control_db()
    completed: list[dict] = []
    ready_before: list[str] = []
    blocked_held: list[str] = []
    with DatabaseTaskSource(control, install_schema=False) as source:
        first = source.ready_tasks(limit=1000)
        ready_before = [
            item.task_alias for item in first.tasks if item.task_alias in BOOTSTRAP
        ]
        for _pass in range(MAX_PASSES):
            page = source.ready_tasks(limit=1000)
            ready = [
                item.task_alias
                for item in page.tasks
                if item.task_alias in HOST_AUTO
            ]
            held_ready = [
                item.task_alias
                for item in page.tasks
                if item.task_alias not in BOOTSTRAP
            ]
            blocked_held = held_ready
            if held_ready:
                raise RuntimeError(
                    "held Plan-R2 tasks became ready without EAAEF-009: "
                    + ",".join(held_ready)
                )
            if not ready:
                break
            progressed = False
            for alias in ready:
                result = _complete(source, alias)
                completed.append(result)
                if result.get("status") == "completed" and result.get("changed"):
                    progressed = True
            if not progressed:
                break
        after = source.ready_tasks(limit=1000)
        ready_after = [item.task_alias for item in after.tasks]
        page_all = source.list_tasks(limit=1000)
        status_counts: dict[str, int] = {}
        for item in page_all.tasks:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-host-admission-supervisor@1",
        "process_started": True,
        "configured_board_launch": False,
        "live_multi_supervisor": False,
        "provider_invoked": False,
        "control_db": str(control.relative_to(ROOT)),
        "collection": collection["decisions"],
        "completed": completed,
        "ready_before": ready_before,
        "ready_after": ready_after,
        "blocked_held": blocked_held,
        "task_count": sum(status_counts.values()),
        "status_counts": status_counts,
        "updated_at": int(time.time()),
    }
    _write_status(payload)
    return payload


def main() -> int:
    payload = run_once()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
