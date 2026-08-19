#!/usr/bin/env python3
"""Host-controlled S-epic supervisor.

Runs ready EAAEF-180..191 tasks against the rematerialized control plane
without configured-board live launch, provider invocation, or Docker-socket
mounts. Live multi-supervisor launch remains a separate fail-closed gate.
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
AUTO_TASKS = {
    "EAAEF-180",
    "EAAEF-181",
    "EAAEF-182",
    "EAAEF-183",
    "EAAEF-185",
    "EAAEF-186",
    "EAAEF-187",
    "EAAEF-188",
    "EAAEF-189",
    "EAAEF-190",
}
ADMIT_REQUIRED_AUTO = {"EAAEF-183"}
S_TASKS = {f"EAAEF-{number}" for number in range(180, 192)}


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


def _complete_auto_task(source: DatabaseTaskSource, alias: str) -> dict:
    task = source.get_task(alias)
    if task is None:
        return {"task_id": alias, "status": "missing"}
    if task.status == "completed":
        return {"task_id": alias, "status": "already_completed"}
    receipt_name = RECEIPT_FILES[alias]
    if alias in ADMIT_REQUIRED_AUTO:
        receipt_path = RECEIPT_DIR / receipt_name
        if receipt_path.is_file():
            current = json.loads(receipt_path.read_text(encoding="utf-8"))
            if current.get("decision") != "admitted":
                return {
                    "task_id": alias,
                    "status": "waiting_rootless_engine",
                    "decision": current.get("decision"),
                }
    receipt_path = RECEIPT_DIR / receipt_name
    validation = [
        "python3",
        "-m",
        "pytest",
        "-q",
        "test/api/test_eaaef_host_admission_unblocking.py",
        "-k",
        {
            "EAAEF-180": "inventory",
            "EAAEF-181": "principals",
            "EAAEF-182": "duckdb_quack",
            "EAAEF-183": "engine_mode",
            "EAAEF-185": "worker_image",
            "EAAEF-186": "container_profile",
            "EAAEF-187": "worker_network",
            "EAAEF-188": "command_fabric",
            "EAAEF-189": "native_lane",
            "EAAEF-190": "plan_r2",
        }[alias],
    ]
    completed = subprocess.run(
        validation,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
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


def run_once() -> dict:
    collection = collect_and_write()
    control = _active_control_db()
    completed: list[dict] = []
    ready: list[str] = []
    blocked_manual: list[str] = []
    with DatabaseTaskSource(control, install_schema=False) as source:
        page = source.ready_tasks()
        ready = [
            item.task_alias for item in page.tasks if item.task_alias in S_TASKS
        ]
        for alias in list(ready):
            if alias in AUTO_TASKS:
                completed.append(_complete_auto_task(source, alias))
            else:
                blocked_manual.append(alias)
        after = source.ready_tasks()
        ready_after = [item.task_alias for item in after.tasks]
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-host-admission-supervisor@1",
        "process_started": True,
        "configured_board_launch": False,
        "live_multi_supervisor": False,
        "provider_invoked": False,
        "control_db": str(control.relative_to(ROOT)),
        "collection": collection["decisions"],
        "completed": completed,
        "ready_before": ready,
        "ready_after": ready_after,
        "blocked_manual": blocked_manual,
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
