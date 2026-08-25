#!/usr/bin/env python3
"""Host-controlled DuckDB bootstrap supervisor.

Completes ready S and A bootstrap tasks against the embedded DuckDB control
plane without configured-board live launch, provider invocation, or Docker-socket
mounts. Held Plan-R2 tasks stay blocked. Live multi-supervisor launch remains a
separate fail-closed gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY_ROOT = ROOT

if TYPE_CHECKING:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
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
EARLY_FRONTIER = frozenset({"EAAEF-180", "EAAEF-181", "EAAEF-182", "EAAEF-183"})
BOOTSTRAP = HOST_AUTO
ADMIT_WAIT_STATUS = {
    "EAAEF-180": "waiting_current_blocker_inventory",
    "EAAEF-181": "waiting_current_runtime_principals",
    "EAAEF-182": "waiting_exact_duckdb_quack_155",
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
COMPLETION_DECISIONS = {
    "EAAEF-180": frozenset({"inventory"}),
    "EAAEF-181": frozenset({"bound_unadmitted"}),
    **{f"EAAEF-{number}": frozenset({"admitted"}) for number in range(182, 192)},
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a bounded execution scope; an omitted scope is early-only."""

    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--early-frontier",
        dest="scope",
        action="store_const",
        const="early_frontier",
        default=argparse.SUPPRESS,
        help="collect and complete only EAAEF-180 through EAAEF-183 (default)",
    )
    scope.add_argument(
        "--full-bootstrap",
        dest="scope",
        action="store_const",
        const="full_bootstrap",
        default=argparse.SUPPRESS,
        help="explicitly collect and advance the complete S/A bootstrap",
    )
    return parser.parse_args(argv)


def _ensure_repository_importable() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _receipt_contract() -> tuple[Path, dict[str, str]]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        RECEIPT_DIR,
        RECEIPT_FILES,
    )

    return RECEIPT_DIR, RECEIPT_FILES


def _current_host_admission_identity() -> dict[str, str]:
    """Return the exact source and board identity receipts must bind."""

    revisions: dict[str, str] = {}
    for name, revision in (("source_head", "HEAD"), ("source_tree", "HEAD^{tree}")):
        completed = _run_argv(
            ["git", "rev-parse", "--verify", revision],
            SOURCE_REPOSITORY_ROOT,
            10,
        )
        value = completed.stdout.strip()
        if completed.returncode != 0 or not value:
            raise RuntimeError(
                f"cannot resolve current EAAEF {name}: {completed.stderr.strip()}"
            )
        revisions[name] = value
    board = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    board_namespace = str(board.get("board_namespace") or "")
    board_cid = str(board.get("board_cid") or "")
    if not board_namespace or not board_cid:
        raise RuntimeError("canonical EAAEF board identity is unavailable")
    return {
        **revisions,
        "board_namespace": board_namespace,
        "board_cid": board_cid,
    }


def _verify_host_admission_task_receipt(
    *,
    task_id: str,
    receipt_dir: Path,
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    """Call the canonical verifier and normalize every failure to a no-go."""

    try:
        from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
            verify_host_admission_task_receipt,
        )

        raw = verify_host_admission_task_receipt(
            task_id=task_id,
            receipt_dir=receipt_dir,
            expected_source_head=expected_identity["source_head"],
            expected_source_tree=expected_identity["source_tree"],
            expected_board_namespace=expected_identity["board_namespace"],
            expected_board_cid=expected_identity["board_cid"],
        )
    except Exception as exc:  # A verifier failure must never admit a dependency.
        return {
            "valid": False,
            "decision": "",
            "blockers": [
                f"host receipt verification failed: {type(exc).__name__}: {exc}"
            ],
        }
    if not isinstance(raw, dict):
        return {
            "valid": False,
            "decision": "",
            "blockers": ["host receipt verifier returned a non-object result"],
        }
    raw_blockers = raw.get("blockers")
    blockers = (
        [str(item) for item in raw_blockers if str(item)]
        if isinstance(raw_blockers, (list, tuple))
        else []
    )
    return {
        "valid": raw.get("valid") is True,
        "decision": str(raw.get("decision") or ""),
        "blockers": blockers,
    }


def _host_receipt_completion_verdict(
    *,
    task_id: str,
    receipt_dir: Path,
    receipt_path: Path,
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    """Return whether one receipt may satisfy its task and dependencies."""

    verification = _verify_host_admission_task_receipt(
        task_id=task_id,
        receipt_dir=receipt_dir,
        expected_identity=expected_identity,
    )
    blockers = list(verification["blockers"])
    if not receipt_path.is_file():
        blockers.append(f"{task_id} host receipt is missing")
    decision = str(verification["decision"])
    allowed_decisions = COMPLETION_DECISIONS.get(task_id, frozenset())
    if decision not in allowed_decisions:
        blockers.append(
            f"{task_id} decision {decision or '<missing>'!r} is not "
            "completion-authorizing"
        )
    blockers = list(dict.fromkeys(blockers))
    return {
        "valid": verification["valid"],
        "decision": decision,
        "blockers": blockers,
        "completion_allowed": (
            verification["valid"] is True
            and receipt_path.is_file()
            and decision in allowed_decisions
            and not blockers
        ),
    }


def _collect_host_admission() -> dict[str, Any]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_early_frontier_and_write,
    )

    return collect_early_frontier_and_write()


def _collect_full_host_admission() -> dict[str, Any]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.validation.eaaef_host_admission import (
        collect_and_write,
    )

    return collect_and_write()


def _database_task_source_class() -> type[DatabaseTaskSource]:
    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )

    return DatabaseTaskSource


def _acquire_state_owner_lease(control: Path) -> Any:
    """Acquire the same OS lease/fence used by the live Quack owner.

    The caller must acquire this before collecting/writing host receipts or
    constructing ``DatabaseTaskSource``.  A competing live owner therefore
    fails this offline path before any DuckDB connection can be opened.
    """

    _ensure_repository_importable()
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        current_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        OWNER_LOCK_SUFFIX,
        OWNER_MARKER_SUFFIX,
        ExclusiveOwnerLease,
    )

    database = Path(control)
    lease = ExclusiveOwnerLease(
        lock_path=database.with_name(f".{database.name}{OWNER_LOCK_SUFFIX}"),
        marker_path=database.with_name(f".{database.name}{OWNER_MARKER_SUFFIX}"),
    )
    lease.acquire(
        server_id=f"offline:eaaef-host-admission:{os.getpid()}",
        process_birth=current_process_birth(),
        database_path=database,
        generation=1,
    )
    return lease


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


def _run_argv(
    argv: list[str], cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
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


def _complete_s_task(
    source: DatabaseTaskSource,
    alias: str,
    expected_identity: dict[str, str],
) -> dict:
    receipt_dir, receipt_files = _receipt_contract()
    task = source.get_task(alias)
    if task is None:
        return {"task_id": alias, "status": "missing"}
    receipt_name = receipt_files[alias]
    receipt_path = receipt_dir / receipt_name
    verdict = _host_receipt_completion_verdict(
        task_id=alias,
        receipt_dir=receipt_dir,
        receipt_path=receipt_path,
        expected_identity=expected_identity,
    )
    if verdict["completion_allowed"] is not True:
        return {
            "task_id": alias,
            "status": ADMIT_WAIT_STATUS.get(
                alias, "waiting_current_valid_host_admission_receipt"
            ),
            "decision": verdict["decision"],
            "receipt_valid": verdict["valid"],
            "blockers": verdict["blockers"],
        }
    if task.status == "completed":
        return {
            "task_id": alias,
            "status": "already_completed",
            "decision": verdict["decision"],
            "receipt_valid": True,
        }
    receipt_digest_before = _cid_bytes(receipt_path.read_bytes())
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
    final_verdict = _host_receipt_completion_verdict(
        task_id=alias,
        receipt_dir=receipt_dir,
        receipt_path=receipt_path,
        expected_identity=expected_identity,
    )
    receipt_digest_after = (
        _cid_bytes(receipt_path.read_bytes()) if receipt_path.is_file() else ""
    )
    if (
        final_verdict["completion_allowed"] is not True
        or receipt_digest_after != receipt_digest_before
    ):
        blockers = list(final_verdict["blockers"])
        if receipt_digest_after != receipt_digest_before:
            blockers.append(f"{alias} host receipt changed during validation")
        return {
            "task_id": alias,
            "status": "receipt_changed_or_revoked",
            "decision": final_verdict["decision"],
            "receipt_valid": final_verdict["valid"],
            "blockers": list(dict.fromkeys(blockers)),
        }
    source.record_evidence(
        task_cid=task.task_cid,
        evidence_kind="host_admission_receipt",
        digest=receipt_digest_after,
        body={
            "path": str(receipt_path.relative_to(ROOT)),
            "source_head": expected_identity["source_head"],
            "source_tree": expected_identity["source_tree"],
            "board_namespace": expected_identity["board_namespace"],
            "board_cid": expected_identity["board_cid"],
            "decision": final_verdict["decision"],
        },
    )
    result = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "completed",
        {"validation": "passed", "host_controlled": True},
        evidence_digests=[digest, receipt_digest_after],
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


def _plan_r2_remote_owner_admitted(
    expected_identity: dict[str, str],
) -> bool:
    receipt_dir, receipt_files = _receipt_contract()
    receipt_path = receipt_dir / receipt_files.get(
        "EAAEF-190", "plan_r2_remote_owner.json"
    )
    verdict = _host_receipt_completion_verdict(
        task_id="EAAEF-190",
        receipt_dir=receipt_dir,
        receipt_path=receipt_path,
        expected_identity=expected_identity,
    )
    return verdict["completion_allowed"] is True


def _complete(
    source: DatabaseTaskSource,
    alias: str,
    expected_identity: dict[str, str],
) -> dict:
    _receipt_dir, receipt_files = _receipt_contract()
    if alias in receipt_files:
        return _complete_s_task(source, alias, expected_identity)
    if alias == "EAAEF-009" and not _plan_r2_remote_owner_admitted(expected_identity):
        return {
            "task_id": alias,
            "status": "waiting_signed_plan_r2_remote_owner",
            "plan_r2_admitted": False,
            "held_board_materialized": False,
            "reason": "independently signed Plan-R2 remote-owner capability is absent",
        }
    return _complete_a_task(source, alias)


def _reopen_invalid_host_admission_tasks(
    source: DatabaseTaskSource,
    expected_identity: dict[str, str],
    *,
    task_ids: frozenset[str] | None = None,
) -> list[dict]:
    """Reopen completed S tasks whose receipts cannot satisfy dependencies."""

    receipt_dir, receipt_files = _receipt_contract()
    reopened: list[dict] = []
    selected = set(receipt_files) if task_ids is None else set(task_ids)
    if not selected.issubset(receipt_files):
        raise ValueError("host-admission reopen scope contains an unknown task")
    for alias in sorted(selected):
        task = source.get_task(alias)
        if task is None or task.status != "completed":
            continue
        receipt_path = receipt_dir / receipt_files[alias]
        verdict = _host_receipt_completion_verdict(
            task_id=alias,
            receipt_dir=receipt_dir,
            receipt_path=receipt_path,
            expected_identity=expected_identity,
        )
        if verdict["completion_allowed"] is True:
            continue
        result = source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "todo",
            {
                "validation": "reopened_invalid_host_admission_receipt",
                "host_controlled": True,
                "receipt_quarantined": True,
                "previous_decision": verdict["decision"],
                "receipt_valid": verdict["valid"],
                "receipt_blockers": verdict["blockers"],
            },
        )
        reopened.append(
            {
                "task_id": alias,
                "status": result.task.status,
                "changed": result.changed,
                "reason": "receipt_invalid_for_completion",
                "decision": verdict["decision"],
                "receipt_valid": verdict["valid"],
                "receipt_quarantined": True,
                "blockers": verdict["blockers"],
            }
        )
    return reopened


def run_once(*, scope: str = "early_frontier") -> dict:
    if scope not in {"early_frontier", "full_bootstrap"}:
        raise ValueError("EAAEF host-admission scope is invalid")
    early_frontier = scope == "early_frontier"
    target_tasks = EARLY_FRONTIER if early_frontier else frozenset(HOST_AUTO)
    control = _active_control_db()
    lease = _acquire_state_owner_lease(control)
    try:
        # Host-evidence materialization writes durable receipts, so it belongs
        # inside the exact same exclusive lease as the embedded DuckDB writer.
        collection = (
            _collect_host_admission()
            if early_frontier
            else _collect_full_host_admission()
        )
        expected_identity = _current_host_admission_identity()
        database_task_source = _database_task_source_class()
        completed: list[dict] = []
        ready_before: list[str] = []
        blocked_held: list[str] = []
        with database_task_source(control, install_schema=False) as source:
            completed.extend(
                _reopen_invalid_host_admission_tasks(
                    source,
                    expected_identity,
                    task_ids=target_tasks if early_frontier else None,
                )
            )
            first = source.ready_tasks(limit=1000)
            ready_before = [
                item.task_alias
                for item in first.tasks
                if item.task_alias in target_tasks
            ]
            passes = 1 if early_frontier else MAX_PASSES
            for _pass in range(passes):
                page = source.ready_tasks(limit=1000)
                ready = [
                    item.task_alias
                    for item in page.tasks
                    if item.task_alias in target_tasks
                ]
                held_ready = [
                    item.task_alias
                    for item in page.tasks
                    if item.task_alias not in target_tasks
                ]
                blocked_held = held_ready
                if held_ready and not early_frontier:
                    raise RuntimeError(
                        "held Plan-R2 tasks became ready without EAAEF-009: "
                        + ",".join(held_ready)
                    )
                if not ready:
                    break
                progressed = False
                for alias in ready:
                    if alias not in target_tasks:
                        raise RuntimeError("host-admission execution escaped its scope")
                    result = _complete(source, alias, expected_identity)
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
            "execution_scope": scope,
            "process_started": True,
            "configured_board_launch": False,
            "live_multi_supervisor": False,
            "provider_invoked": False,
            "control_db": str(control.relative_to(ROOT)),
            "expected_receipt_identity": expected_identity,
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
    finally:
        lease.release(fence_token=lease.fence_token)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_once(scope=str(getattr(args, "scope", "early_frontier")))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
