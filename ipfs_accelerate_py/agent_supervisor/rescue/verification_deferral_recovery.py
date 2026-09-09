"""Offline native recovery of a terminal checkout-verification wait.

A generic projection-wait budget is reopened only after independently reading
its failed execution phases and reconciling its historical protected snapshot.
This is a distinct recovery authority: provider quota receipts cannot qualify a
checkout wait, and neither a test pass nor this operation completes a task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..task_sources.database_task_source import DatabaseTaskSource
from ..task_sources.duckdb_state import open_duckdb_connection
from ..todo_daemon.database_portal_bridge import (
    DatabasePortalAttemptPaths, DatabasePortalExecutionBridge, database_portal_task_contract_digest,
)
from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, PortalImplementationDaemon

SCHEMA = "ipfs_accelerate_py/agent-supervisor/verification-deferral-recovery@1"
REASON = "Portal task projection is not complete"
TIMEOUT = "implementation_protected_path_verification_lock_timeout"


class VerificationDeferralRecoveryError(RuntimeError):
    pass


def digest(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(dict(value), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise VerificationDeferralRecoveryError(reason)


def require_stopped(owner_status: Path, database: Path) -> dict[str, Any]:
    owner = json.loads(owner_status.read_text())
    require(owner.get("lifecycle") == "stopped", "native owner must be stopped")
    require(Path(owner["database_path"]).resolve() == database.resolve(),
            "owner database identity differs")
    birth = owner.get("identity", {}).get("process_birth", {})
    pid = birth.get("pid")
    require(type(pid) is int and pid > 0, "owner process birth is absent")
    proc = Path("/proc") / str(pid)
    if proc.exists():
        try:
            current = int((proc / "stat").read_text().rsplit(")", 1)[1].split()[19])
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        except OSError as exc:
            raise VerificationDeferralRecoveryError("owner death cannot be proved") from exc
        require(current != birth.get("start_time_ticks") or boot != birth.get("boot_id"),
                "native owner is still alive")
    return birth


def verify_blocked(task: Any, execution: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(task.status == "blocked", "task is not blocked")
    body = dict(task.body)
    require(not DatabaseImplementationDaemon._automatic_claim_forbidden(task),
            "manual task cannot be automatically reopened")
    receipt = body.get("completion_receipt", {})
    require(receipt.get("operation") == "database_portal_typed_deferral_budget_exhausted"
            and receipt.get("attempt_consumed") is False
            and receipt.get("control_expected_revision") == task.revision - 1,
            "task has no exact exhausted non-consuming receipt")
    budget = receipt.get("retry_budget", {})
    copy = dict(budget); observation = copy.pop("observation_id", None)
    require(observation == digest(copy) and budget.get("exhausted") is True
            and budget.get("task_cid") == task.task_cid, "budget identity is invalid")
    matching = budget.get("matching_attempts")
    require(isinstance(matching, list) and 0 < len(matching) <= 64
            and not budget.get("matching_attempts_truncated")
            and all(item.get("reason") == REASON for item in matching),
            "budget contains foreign or truncated outcomes")
    rows = execution.execute("SELECT * FROM database_task_attempts WHERE task_cid = ? "
        "ORDER BY attempt_number DESC LIMIT 129", [task.task_cid]).fetchall()
    require(0 < len(rows) <= 128, "attempt population is absent or exceeds the bound")
    attempts = [dict(row) for row in rows]
    latest = attempts[0]
    require(latest["attempt_id"] == receipt.get("attempt_id")
            and latest["status"] == "failed"
            and latest["revision"] == receipt.get("execution_revision"),
            "blocked attempt is stale or running")
    for field in ("claim_id", "lease_id", "owner_session_id", "attempt_number",
                  "fencing_token", "fence_epoch"):
        require(latest[field] == receipt.get(field), "blocked claim identity differs")
    by_id = {row["attempt_id"]: row for row in attempts}
    for item in matching:
        attempt = by_id.get(item["attempt_id"], {})
        require(attempt.get("status") == "failed" and
                attempt.get("attempt_number") == item.get("attempt_number"),
                "matching execution attempt differs")
        phases = execution.execute("SELECT phase, body_json FROM attempt_phases "
            "WHERE attempt_id = ?", [item["attempt_id"]]).fetchall()
        require(not any(row["phase"] in {"provider", "effect", "validation", "complete"}
                        for row in phases), "matching attempt committed an effect")
        failed = [json.loads(row["body_json"]) for row in phases if row["phase"] == "failed"]
        require(len(failed) == 1 and failed[0].get("attempt_consumed") is False
                and failed[0].get("provider_dispatched") is False
                and failed[0].get("reason") == REASON,
                "matching wait did not prove no provider dispatch")
    return dict(receipt), attempts


def qualify_snapshot(*, repo: Path, runtime: Path, attempt_root: Path,
                     task: Any, attempts: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    for row in attempts[1:]:
        key = hashlib.sha256(row["attempt_id"].encode()).hexdigest()[:24]
        root = attempt_root / key
        if not (root / "portal-events.jsonl").is_file():
            continue
        paths = DatabasePortalAttemptPaths(root, root / "task-projection.md",
            root / "database-attempt-binding.json", root / "portal-task-state.json",
            root / "portal-strategy.json", root / "portal-events.jsonl", root / "implementation-logs")
        binding = json.loads(paths.binding.read_text())
        binding_body = dict(binding); binding_id = binding_body.pop("binding_id", None)
        require(binding_id == digest(binding_body) and binding.get("task_contract_digest") ==
                database_portal_task_contract_digest(task), "snapshot semantic binding differs")
        DatabasePortalExecutionBridge._verify_projection(paths, binding)
        require(binding.get("attempt_id") == row["attempt_id"]
                and binding.get("task_cid") == task.task_cid
                and binding.get("claim_id") == row["claim_id"], "snapshot binding differs")
        events = DatabasePortalExecutionBridge._verified_event_chain(paths)
        terminals = [e for e in events if e.get("type") == "implementation_finished"
            and e.get("task_id") == task.task_alias
            and e.get("canonical_task_cid") == task.task_cid
            and e.get("attempt_consumed") is False
            and e.get("protected_path_violation", {}).get("verification_deferred") is True
            and e.get("protected_path_violation", {}).get("reason") == TIMEOUT]
        if not terminals:
            continue
        terminal = terminals[-1]
        baseline = str(terminal.get("baseline_ref") or "")
        workspace = Path(str(terminal.get("worktree_path") or "")).resolve()
        require(workspace.is_relative_to((runtime / "worktrees").resolve())
                and workspace.is_dir() and len(baseline) == 40,
                "terminal workspace or baseline identity is invalid")
        protected = config["protected_paths"]
        require(protected and all(isinstance(p, str) and not Path(p).is_absolute()
                and ".." not in Path(p).parts for p in protected), "protected population is invalid")
        for checkout in (repo, workspace):
            validation = subprocess.run(["git", "-C", str(checkout), "diff", "--quiet",
                baseline, "--", *protected], capture_output=True, timeout=30)
            require(validation.returncode == 0, "protected content changed after reconciliation")
        marker = root / "verification-deferral-recovery.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            require(previous.get("target_attempt_id") == attempts[0]["attempt_id"],
                    "historical snapshot has already rearmed another attempt")
        active_path = root / "implementation-protected-path-active.json"
        require(not (root / "implementation-protected-path-incident.json").exists(),
                "protected mutation incident still requires qualification")
        if active_path.exists():
            active = json.loads(active_path.read_text())
            require(active.get("task_id") == task.task_alias
                    and active.get("attempt") == terminal.get("attempt")
                    and active.get("workspace_path") == terminal.get("worktree_path"),
                    "retained snapshot identity differs")
            protected = active["protected_paths"]
            require(set(protected) == set(config["protected_paths"]),
                    "configured protected population differs")
            daemon = PortalImplementationDaemon(todo_path=paths.task_projection,
                state_path=paths.state, strategy_path=paths.strategy, events_path=paths.events,
                repo_root=repo, board_namespace=config["board_namespace"],
                task_header_prefix=config["task_prefix"], implement=False,
                use_ephemeral_worktree=True, worktree_root=runtime / "worktrees",
                merge_target_branch=config["merge_target_branch"],
                implementation_protected_paths=protected)
            try:
                result = daemon._reconcile_implementation_protected_path_fence()
            finally:
                daemon.close_event_runtime()
            require(result.get("blocked") is False and result.get("reason") ==
                    "crash_reconciliation_unchanged", "snapshot did not qualify unchanged")
            events = DatabasePortalExecutionBridge._verified_event_chain(paths)
        reconciled = [e for e in events if e.get("type") ==
            "implementation_protected_path_snapshot_reconciled" and e.get("task_id") == task.task_alias
            and e.get("attempt") == terminal.get("attempt") and e.get("reason") ==
            "crash_reconciliation_unchanged" and e.get("sequence", 0) > terminal["sequence"]]
        require(reconciled and not active_path.exists(), "no durable unchanged snapshot reconciliation")
        proof = {"schema": SCHEMA, "source_attempt_id": row["attempt_id"],
            "target_attempt_id": attempts[0]["attempt_id"], "task_cid": task.task_cid,
            "terminal_event_id": terminal["event_id"],
            "reconciliation_event_id": reconciled[-1]["event_id"],
            "candidate_path": terminal.get("worktree_path", ""),
            "completion_authority": False, "provider_dispatched": False}
        proof["receipt_id"] = digest(proof)
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(proof, indent=2) + "\n")
        temporary.replace(marker)
        return proof
    raise VerificationDeferralRecoveryError("no terminal verification-deferred snapshot qualifies")


def recover(*, repo: Path, database: Path, execution: Path, attempt_root: Path,
            owner_status: Path, config_path: Path, task_alias: str, apply: bool = False) -> dict[str, Any]:
    require(all(p.is_file() for p in (database, execution, owner_status, config_path)),
            "required native input is absent")
    require_stopped(owner_status, database)
    config = json.loads(config_path.read_text())
    config["protected_paths"] = config.get("implementation_protected_paths", config.get("protected_paths", []))
    config["merge_target_branch"] = config.get("merge_target_branch") or config.get("target_branch")
    with open_duckdb_connection(execution, timeout_seconds=5) as connection:
        with DatabaseTaskSource(database, owner_id="verification-deferral-recovery", install_schema=False) as source:
            task = source.get_task(task_alias)
            require(task is not None, "canonical task is absent")
            receipt, attempts = verify_blocked(task, connection)
            proof = qualify_snapshot(repo=repo, runtime=database.parent, attempt_root=attempt_root,
                task=task, attempts=attempts, config=config)
            require_stopped(owner_status, database)
            route_fields = {k: receipt[k] for k in ("execution_route_binding",
                "execution_route_policy_id", "execution_route_origin_revision") if k in receipt}
            require(len(route_fields) in (0, 3), "partial execution route cannot be carried")
            recovery = {**route_fields, "schema": SCHEMA, "operation": "recover_verified_checkout_verification_wait",
                "reason": "unchanged_terminal_snapshot_reconciled", "source_receipt": receipt,
                "qualification": proof, "completion_authority": False}
            # A new, separately verified recovery authority owns this CAS. The
            # ordinary provider-canary supersession route cannot prove checkout
            # verification and deliberately remains unchanged.
            if not apply:
                return {"schema": SCHEMA, "task_alias": task_alias, "qualified": True,
                    "changed": False, "qualification": proof}
            result = source.intent.cas_task_status(task_cid=task.task_cid,
                expected_revision=task.revision, new_status="todo", receipt=recovery,
                expected_control_receipt=receipt)
            return {"schema": SCHEMA, "task_alias": task_alias, "status": "todo",
                "changed": result.changed, "revision": result.revision, "qualification": proof}


def recover_stopped_board(*, inventory: Path, board_id: str, apply: bool) -> dict[str, Any]:
    entries = json.loads(inventory.read_text())["boards"]
    entries = [b for b in entries if b.get("id") == board_id]
    require(len(entries) == 1, "inventory board is absent or ambiguous")
    board = entries[0]
    if any(Path(path).exists() for path in board.get("hold_paths", [])):
        return {"schema": SCHEMA, "board_id": board_id, "recovered": 0, "held": True}
    database = Path(board["database_path"])
    owner = Path(board["owner_status_path"])
    require_stopped(owner, database)
    with DatabaseTaskSource(database, owner_id="verification-deferral-discovery", install_schema=False) as source:
        tasks = source.list_tasks(status="blocked", limit=128).tasks
    executions = sorted(Path(board["state_root"]).glob("lane-*/*_database_execution.duckdb"))
    require(len(executions) <= 16, "lane execution population exceeds the bound")
    results = []
    for task in tasks:
        receipt = dict(task.body).get("completion_receipt", {})
        if receipt.get("operation") != "database_portal_typed_deferral_budget_exhausted":
            continue
        for execution in executions:
            prefix = execution.name.removesuffix("_database_execution.duckdb")
            attempt_root = execution.parent / (prefix + "_database_portal_attempts")
            try:
                result = recover(repo=Path(board["cwd"]), database=database, execution=execution,
                    attempt_root=attempt_root, owner_status=owner, config_path=Path(board["config_path"]),
                    task_alias=task.task_alias, apply=apply)
            except (VerificationDeferralRecoveryError, KeyError) as exc:
                results.append({"task_alias": task.task_alias, "lane": execution.parent.name,
                    "changed": False, "reason": str(exc)})
                continue
            results.append(result)
            break
    return {"schema": SCHEMA, "board_id": board_id, "results": results,
        "recovered": sum(r.get("changed") is True for r in results)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "database", "execution", "attempt-root", "owner-status", "config-path"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--task-alias")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--board")
    parser.add_argument("--apply", action="store_true")
    args = vars(parser.parse_args())
    inventory = args.pop("inventory")
    board = args.pop("board")
    if inventory is not None or board is not None:
        require(inventory is not None and bool(board), "inventory and board are required together")
        result = recover_stopped_board(inventory=inventory, board_id=board, apply=args["apply"])
    else:
        require(all(value is not None for key, value in args.items() if key != "apply"),
            "individual recovery requires all native paths and task alias")
        result = recover(**args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
