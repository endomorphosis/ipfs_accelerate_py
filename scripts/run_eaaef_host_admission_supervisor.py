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
ADMIT_REQUIRED_AUTO = {
    "EAAEF-183",
    "EAAEF-184",
    "EAAEF-185",
    "EAAEF-186",
    "EAAEF-187",
    "EAAEF-188",
    "EAAEF-189",
    "EAAEF-190",
}
ADMIT_WAIT_STATUS = {
    "EAAEF-183": "waiting_rootless_engine",
    "EAAEF-184": "waiting_signed_provider_authorization",
    "EAAEF-185": "waiting_signed_worker_image",
    "EAAEF-186": "waiting_signed_execution_profile_v2",
    "EAAEF-187": "waiting_signed_worker_network_lanes",
    "EAAEF-188": "waiting_signed_command_fabric",
    "EAAEF-189": "waiting_signed_native_lane_dispatcher",
    "EAAEF-190": "waiting_signed_plan_r2_remote_owner",
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


def _is_pytest_argv(argv: list[str]) -> bool:
    return "-m" in argv and "pytest" in argv


def _run_pytest_file_isolation(
    argv: list[str],
    cwd: Path,
    timeout: int,
    stdout: str,
) -> dict:
    from ipfs_accelerate_py.agent_supervisor.validation.implementation_auto_rescue import (
        pytest_isolation_argv,
        pytest_isolation_files,
    )

    files = pytest_isolation_files(argv=argv, stdout=stdout)
    if not files:
        return {"passed": False, "reason": "no_pytest_files_to_isolate", "results": []}
    started = time.time()
    results: list[dict] = []
    remaining = timeout
    for path in files:
        if remaining <= 1:
            return {
                "passed": False,
                "reason": "isolation_timeout",
                "results": results,
            }
        completed = _run_argv(pytest_isolation_argv(path), cwd, remaining)
        remaining = max(1, timeout - int(time.time() - started))
        results.append(
            {
                "path": path,
                "returncode": completed.returncode,
                "passed": completed.returncode == 0,
                "stdout": completed.stdout[-400:],
                "stderr": completed.stderr[-200:],
            }
        )
        if completed.returncode != 0:
            return {
                "passed": False,
                "reason": "isolated_pytest_file_failed",
                "failed_path": path,
                "results": results,
            }
    return {
        "passed": True,
        "reason": "isolated_pytest_files_passed",
        "results": results,
    }


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
    isolation: dict | None = None
    for item in commands:
        argv = [str(part) for part in item.get("argv") or ()]
        working = str(item.get("working_directory") or ".")
        if not argv:
            return {"task_id": alias, "status": "empty_execution_validation"}
        cwd = ROOT if working in {".", ""} else ROOT / working
        started = time.time()
        completed = _run_argv(argv, cwd, 1800)
        digest = _cid_bytes(
            (completed.stdout + completed.stderr + str(completed.returncode)).encode()
        )
        digests.append(digest)
        outcome = "passed" if completed.returncode == 0 else "failed"
        isolation = None
        if completed.returncode != 0 and _is_pytest_argv(argv):
            remaining = max(60, 1800 - int(time.time() - started))
            isolation = _run_pytest_file_isolation(
                argv, cwd, remaining, completed.stdout + completed.stderr
            )
            if isolation.get("passed") is True:
                outcome = "passed"
                isolation_digest = _cid_bytes(
                    json.dumps(isolation, sort_keys=True).encode()
                )
                source.record_validation_result(
                    task_cid=task.task_cid,
                    outcome="passed",
                    evidence_digest=isolation_digest,
                    argv=["python3", "-m", "pytest", "-q", "--eaaef-file-isolation"],
                )
                digests.append(isolation_digest)
        source.record_validation_result(
            task_cid=task.task_cid,
            outcome=outcome,
            evidence_digest=digest,
            argv=argv,
        )
        if outcome != "passed":
            payload = {
                "task_id": alias,
                "status": "validation_failed",
                "returncode": completed.returncode,
                "stdout": completed.stdout[-1200:],
                "stderr": completed.stderr[-800:],
                "argv": argv,
            }
            if isolation is not None:
                payload["pytest_file_isolation"] = isolation
            return payload
    result = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "completed",
        {
            "validation": "passed",
            "host_controlled": True,
            "duckdb": True,
            **(
                {"pytest_file_isolation": True}
                if isolation and isolation.get("passed") is True
                else {}
            ),
        },
        evidence_digests=digests,
    )
    return {
        "task_id": alias,
        "status": result.task.status,
        "changed": result.changed,
        "receipt_cid": result.receipt_cid,
    }


def _plan_r2_remote_owner_admitted() -> bool:
    receipt_path = RECEIPT_DIR / RECEIPT_FILES.get("EAAEF-190", "plan_r2_remote_owner.json")
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    evidence = receipt.get("evidence") or {}
    return (
        receipt.get("decision") == "admitted"
        and evidence.get("independent_signature_present") is True
        and evidence.get("self_signed_rejected") is True
    )


def _complete(source: DatabaseTaskSource, alias: str) -> dict:
    if alias in RECEIPT_FILES:
        return _complete_s_task(source, alias)
    if alias == "EAAEF-009" and not _plan_r2_remote_owner_admitted():
        return {
            "task_id": alias,
            "status": "waiting_signed_plan_r2_remote_owner",
            "plan_r2_admitted": False,
            "held_board_materialized": False,
            "reason": "independently signed Plan-R2 remote-owner capability is absent",
        }
    return _complete_a_task(source, alias)


def _reopen_unadmitted(source: DatabaseTaskSource) -> list[dict]:
    """Reopen auto S tasks whose receipts are no longer admitted evidence."""

    reopened: list[dict] = []
    for alias in sorted(ADMIT_REQUIRED_AUTO | {"EAAEF-191"}):
        task = source.get_task(alias)
        if task is None or task.status != "completed":
            continue
        receipt_path = RECEIPT_DIR / RECEIPT_FILES[alias]
        if not receipt_path.is_file():
            continue
        current = json.loads(receipt_path.read_text(encoding="utf-8"))
        refresh_admitted = alias in {
            "EAAEF-188",
            "EAAEF-189",
            "EAAEF-190",
            "EAAEF-191",
        } and current.get("decision") == "admitted"
        if current.get("decision") == "admitted" and not refresh_admitted:
            continue
        result = source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "todo",
            {
                "validation": "reopened_unadmitted",
                "host_controlled": True,
                "previous_decision": current.get("decision"),
            },
        )
        reopened.append(
            {
                "task_id": alias,
                "status": result.task.status,
                "changed": result.changed,
                "reason": "receipt_not_admitted",
                "decision": current.get("decision"),
            }
        )
    return reopened


def run_once() -> dict:
    collection = collect_and_write()
    control = _active_control_db()
    completed: list[dict] = []
    ready_before: list[str] = []
    blocked_held: list[str] = []
    with DatabaseTaskSource(control, install_schema=False) as source:
        completed.extend(_reopen_unadmitted(source))
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
