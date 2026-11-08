"""Repair a blocked workflow edit that deleted a required command.

The proposal gate rejects a ``run`` or ``uses`` command that was replaced by
a different command. This module puts the removed command back as its own
step, keeps every added step, and refuses bypass tokens. When the restored
tree passes the task's validation command, it records one repair task and
completes the blocked parent through the evidence gate. It does not mark a
task complete when validation fails, and it does not admit the original diff.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..task_sources.control_plane_contracts import content_identity
from ..validation.proposal_validation import (
    restore_substituted_workflow_commands,
    workflow_pairs_are_non_weakening,
)
from .obsolete_workflow_gate_rearm import (
    _git,
    _workflow_path,
    load_preserved_workflow_pairs,
    preserved_workflow_revisions,
    receipt_is_obsolete_workflow_gate_block,
)

EVENT = "workflow_command_restoration_applied"
SCHEMA = "database-workflow-command-restoration@1"
OPERATION = "workflow_command_restoration"


def _receipt(task: Any) -> Mapping[str, Any] | None:
    body = getattr(task, "body", None)
    if not isinstance(body, Mapping):
        return None
    receipt = body.get("completion_receipt")
    if not isinstance(receipt, Mapping):
        return None
    return receipt


def _events(owner: Any, receipt: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    from .database_portal_bridge import DatabasePortalExecutionBridge

    attempt = owner.get_attempt(str(receipt.get("attempt_id") or ""))
    if attempt is None:
        return None
    callback = getattr(owner, "_provider_fn", None)
    resolve = getattr(owner, "_resolve_native_portal_provider_callback", None)
    if callable(resolve):
        try:
            callback = resolve(callback, attempt)
        except Exception:
            return None
    bridge = getattr(callback, "__self__", None)
    if type(bridge) is not DatabasePortalExecutionBridge or bridge.repo_root is None:
        return None
    try:
        raw = bridge._paths(attempt).events.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if len(raw) > 8_000_000:
        return None
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(event, dict):
            events.append(event)
    return events


def _repo_root(owner: Any, receipt: Mapping[str, Any]) -> Path | None:
    from .database_portal_bridge import DatabasePortalExecutionBridge

    attempt = owner.get_attempt(str(receipt.get("attempt_id") or ""))
    if attempt is None:
        return None
    callback = getattr(owner, "_provider_fn", None)
    resolve = getattr(owner, "_resolve_native_portal_provider_callback", None)
    if callable(resolve):
        try:
            callback = resolve(callback, attempt)
        except Exception:
            return None
    bridge = getattr(callback, "__self__", None)
    if type(bridge) is not DatabasePortalExecutionBridge or bridge.repo_root is None:
        return None
    return Path(bridge.repo_root)


def _already_applied(owner: Any, settlement_id: str) -> bool:
    rows = owner._require_connection().execute(
        "SELECT body_json FROM daemon_execution_events WHERE event_type = ?",
        [EVENT],
    ).fetchall()
    for row in rows:
        try:
            body = json.loads(row[0])
        except (TypeError, ValueError, IndexError):
            return True
        if isinstance(body, dict) and body.get("settlement_id") == settlement_id:
            return True
    return False


def _source_plan(task: Any) -> Mapping[str, Any]:
    body = getattr(task, "body", None)
    if not isinstance(body, Mapping):
        return {}
    plan = body.get("source_plan_task")
    return plan if isinstance(plan, Mapping) else {}


def _validation_argv(task: Any) -> list[str] | None:
    plan = _source_plan(task)
    argv = plan.get("validation_argv")
    if (
        isinstance(argv, list)
        and len(argv) == 1
        and isinstance(argv[0], list)
        and argv[0]
        and all(isinstance(part, str) and part for part in argv[0])
    ):
        return list(argv[0])
    return None


def _write_scope(task: Any) -> tuple[str, ...]:
    plan = _source_plan(task)
    scope = plan.get("write_scope")
    if not isinstance(scope, list) or not scope:
        return ()
    cleaned = []
    for item in scope:
        if not isinstance(item, str) or not item or item.startswith("/") or ".." in item.split("/"):
            return ()
        cleaned.append(item)
    return tuple(cleaned)


def _in_scope(path: str, scope: Sequence[str]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in scope)


def _restored_files(
    repo: Path,
    pairs: Sequence[tuple[str, str | None, str | None]],
) -> dict[str, str] | None:
    restored: dict[str, str] = {}
    for path, before, after in pairs:
        if not _workflow_path(path) or before is None or after is None:
            return None
        text = restore_substituted_workflow_commands(before, after)
        if text is None or not workflow_pairs_are_non_weakening([(path, before, text)]):
            return None
        restored[path] = text
    if not restored:
        return None
    if repo is None:
        return None
    return restored


def _author(repo: Path, revision: str) -> tuple[str, str] | None:
    result = _git(repo, ["log", "-1", "--format=%an%x00%ae", revision])
    if result is None or result.returncode != 0 or "\x00" not in result.stdout:
        return None
    name, email = result.stdout.split("\x00", 1)
    name = name.strip()
    email = email.strip()
    if not name or not email or "\n" in name or "\n" in email:
        return None
    return name, email


def _run(argv: Sequence[str], *, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _audit_revision(repo: Path, revision: str, validation_argv: Sequence[str]) -> str | None:
    work = Path(tempfile.mkdtemp(prefix="workflow-command-restoration-audit-"))
    added = _git(repo, ["worktree", "add", "--detach", str(work), revision])
    if added is None or added.returncode != 0:
        _git(repo, ["worktree", "remove", "--force", str(work)])
        return None
    try:
        audit = _run(validation_argv, cwd=work)
        if audit is None or audit.returncode != 0:
            return None
        return audit.stdout
    finally:
        _git(repo, ["worktree", "remove", "--force", str(work)])


def commit_restored_tree(
    repo: Path,
    *,
    baseline: str,
    preserved: str,
    restored: Mapping[str, str],
    extra_paths: Sequence[str],
    validation_argv: Sequence[str],
    message: str,
) -> tuple[str, str] | None:
    """Commit the restored tree onto ``baseline`` and fast-forward ``repo``.

    Returns the new commit and the validation stdout. Returns None when the
    branch has moved, a path is outside the requested set, or validation fails.
    """

    head = _git(repo, ["rev-parse", "HEAD"])
    branch = _git(repo, ["branch", "--show-current"])
    if (
        head is None
        or branch is None
        or head.returncode != 0
        or branch.returncode != 0
        or head.stdout.strip() != baseline
        or not branch.stdout.strip()
    ):
        return None
    author = _author(repo, baseline)
    if author is None:
        return None
    dirty = _git(repo, ["status", "--porcelain", "--", *restored.keys(), *extra_paths])
    if dirty is None or dirty.returncode != 0 or dirty.stdout.strip():
        return None
    work = Path(tempfile.mkdtemp(prefix="workflow-command-restoration-"))
    added = _git(repo, ["worktree", "add", "--detach", str(work), baseline])
    if added is None or added.returncode != 0:
        _git(repo, ["worktree", "remove", "--force", str(work)])
        return None
    try:
        for path, text in restored.items():
            if not _workflow_path(path):
                return None
            destination = work / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        for path in extra_paths:
            if _workflow_path(path) or path.startswith("/") or ".." in path.split("/"):
                return None
            checked = _git(work, ["checkout", preserved, "--", path])
            if checked is None or checked.returncode != 0:
                return None
        staged = _git(work, ["add", "--", *restored.keys()])
        if staged is None or staged.returncode != 0:
            return None
        if extra_paths:
            forced = _git(work, ["add", "-f", "--", *extra_paths])
            if forced is None or forced.returncode != 0:
                return None
        audit = _run(validation_argv, cwd=work)
        if audit is None or audit.returncode != 0:
            return None
        name, email = author
        committed = _git(
            work,
            [
                "-c",
                f"user.name={name}",
                "-c",
                f"user.email={email}",
                "commit",
                "-m",
                message,
            ],
        )
        if committed is None or committed.returncode != 0:
            return None
        revision = _git(work, ["rev-parse", "HEAD"])
        if revision is None or revision.returncode != 0:
            return None
        new_commit = revision.stdout.strip()
        merged = _git(repo, ["merge", "--ff-only", new_commit])
        if merged is None or merged.returncode != 0:
            return None
        return new_commit, audit.stdout
    finally:
        _git(repo, ["worktree", "remove", "--force", str(work)])


def _repair_alias(parent_alias: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._:+@-" else "-"
        for character in parent_alias
    )
    return f"{cleaned[:80]}.restore-commands"


def _upsert_repair_task(
    owner: Any,
    task: Any,
    *,
    missing: Sequence[str],
    paths: Sequence[str],
    argv: Sequence[str],
) -> str | None:
    source = getattr(owner, "task_source", None)
    intent = getattr(source, "_intent", None)
    upsert = getattr(intent, "upsert_task", None)
    if not callable(upsert):
        return None
    parent_alias = str(getattr(task, "task_alias", "") or "")
    parent_cid = str(getattr(task, "task_cid", "") or "")
    goal_cid = str(getattr(task, "goal_cid", "") or "")
    if not parent_alias or not parent_cid or not goal_cid:
        return None
    alias = _repair_alias(parent_alias)
    plan = _source_plan(task)
    body = getattr(task, "body", None)
    plan_cid = ""
    if isinstance(body, Mapping) and isinstance(body.get("accepted_plan_root_cid"), str):
        plan_cid = str(body.get("accepted_plan_root_cid") or "")
    task_cid = content_identity(
        {
            "schema": SCHEMA,
            "parent_task_cid": parent_cid,
            "alias": alias,
            "missing_commands": list(missing),
        }
    )
    criterion = (
        "Restore every removed workflow run and uses command as its own step. "
        "Keep added steps. Do not add continue-on-error or || true. "
        "Missing commands: " + " | ".join(command.replace("\n", " ") for command in missing)
    )
    upsert(
        task_cid=task_cid,
        task_alias=alias,
        goal_cid=goal_cid,
        plan_cid=plan_cid,
        objective_id=str(plan.get("subgoal_id") or ""),
        ordinal=0,
        status="ready",
        priority="P1",
        body={
            "completion": "auto",
            "is_schedulable": False,
            "title": f"Restore workflow commands removed from {parent_alias}",
            "parent_task_alias": parent_alias,
            "parent_task_cid": parent_cid,
            "missing_commands": list(missing),
            "source_plan_task": {
                "id": alias,
                "title": f"Restore workflow commands removed from {parent_alias}",
                "goal_id": str(plan.get("goal_id") or ""),
                "subgoal_id": str(plan.get("subgoal_id") or ""),
                "depends_on": [],
                "repository": str(plan.get("repository") or ""),
                "outputs": list(paths),
                "write_scope": list(_write_scope(task)),
                "validation_argv": [list(argv)],
                "acceptance": criterion,
                "completion": "auto",
            },
        },
        identity={"parent_task_cid": parent_cid},
        expected_revision=0,
        dependencies=[],
        outputs=[{"path": path} for path in paths],
        acceptance=[criterion],
        validations=[list(argv)],
    )
    return task_cid


def restore_blocked_task(owner: Any, task: Any) -> dict[str, Any] | None:
    """Create a repair task and complete the parent when restoration validates."""

    if str(getattr(task, "status", "") or "").strip().lower() != "blocked":
        return None
    running = getattr(owner, "list_running_attempts", None)
    if callable(running) and list(running() or ()):
        return None
    receipt = _receipt(task)
    if receipt is None or not receipt_is_obsolete_workflow_gate_block(receipt):
        return None
    settlement_id = str(receipt.get("settlement_id") or "")
    if not settlement_id or _already_applied(owner, settlement_id):
        return None
    if int(getattr(task, "revision", 0) or 0) != int(receipt["control_expected_revision"]) + 1:
        return None
    attempt_number = receipt.get("attempt_number")
    max_attempts = getattr(owner, "max_task_attempts", None)
    if (
        type(attempt_number) is not int
        or type(max_attempts) is not int
        or max_attempts < 1
        or attempt_number < max_attempts
    ):
        return None
    argv = _validation_argv(task)
    scope = _write_scope(task)
    if argv is None or not scope:
        return None
    pairs = load_preserved_workflow_pairs(owner, task, receipt)
    if not pairs:
        return None
    events = _events(owner, receipt)
    revisions = preserved_workflow_revisions(events or [])
    repo = _repo_root(owner, receipt)
    if revisions is None or repo is None:
        return None
    baseline, preserved = revisions
    restored = _restored_files(repo, pairs)
    if restored is None:
        return None
    listed = _git(repo, ["diff", "--name-only", baseline, preserved])
    if listed is None or listed.returncode != 0:
        return None
    extra = []
    for path in listed.stdout.splitlines():
        path = path.strip()
        if not path or path in restored or not _in_scope(path, scope):
            continue
        if path.startswith("/") or ".." in path.split("/"):
            return None
        extra.append(path)
    if any(not _in_scope(path, scope) for path in list(restored) + extra):
        return None
    parent_alias = str(getattr(task, "task_alias", "") or "task")
    head = _git(repo, ["rev-parse", "HEAD"])
    head_text = head.stdout.strip() if head is not None and head.returncode == 0 else ""
    if head_text == baseline:
        committed = commit_restored_tree(
            repo,
            baseline=baseline,
            preserved=preserved,
            restored=restored,
            extra_paths=extra,
            validation_argv=argv,
            message=f"{parent_alias}: restore removed workflow commands",
        )
        if committed is None:
            return None
        new_commit, stdout = committed
    else:
        for path, text in restored.items():
            current = _git(repo, ["show", f"{head_text}:{path}"])
            if current is None or current.returncode != 0 or current.stdout != text:
                return None
        audited = _audit_revision(repo, head_text, argv)
        if audited is None:
            return None
        new_commit, stdout = head_text, audited
    digest = "sha256:" + hashlib.sha256(stdout.encode("utf-8")).hexdigest()
    repair_error = ""
    try:
        repair_cid = _upsert_repair_task(
            owner,
            task,
            missing=_missing_from_pairs(pairs),
            paths=list(restored),
            argv=argv,
        )
    except Exception as exc:
        repair_cid = None
        repair_error = f"{type(exc).__name__}: {exc}"
    source = owner.task_source
    source.record_validation_result(
        task_cid=str(task.task_cid),
        outcome="passed",
        evidence_digest=digest,
        argv=argv,
        attempt_id=str(receipt.get("attempt_id") or ""),
        body={"commit": new_commit, "operation": OPERATION},
    )
    source.record_evidence(
        task_cid=str(task.task_cid),
        evidence_kind="validation",
        digest=digest,
        body={"argv": argv, "commit": new_commit, "stdout_sha256": digest},
    )
    completion = {
        "operation": OPERATION,
        "schema": SCHEMA,
        "settlement_id": settlement_id,
        "attempt_id": str(receipt.get("attempt_id") or ""),
        "evidence_digest": digest,
        "commit": new_commit,
        "baseline": baseline,
        "preserved_commit": preserved,
        "validation_exit_code": 0,
        "completion_authority": False,
        "github_actions_executed": False,
    }
    changed = source.compare_and_set_status(
        str(task.task_cid),
        int(task.revision),
        "completed",
        completion,
        expected_control_receipt=dict(receipt),
        evidence_digests=[digest],
    )
    if getattr(changed, "changed", False) is not True:
        return None
    if repair_cid is not None:
        source.record_evidence(
            task_cid=repair_cid,
            evidence_kind="validation",
            digest=digest,
            body={"parent_task_cid": str(task.task_cid), "commit": new_commit},
        )
        source.compare_and_set_status(
            repair_cid,
            1,
            "completed",
            {
                "operation": OPERATION,
                "schema": SCHEMA,
                "parent_task_cid": str(task.task_cid),
                "evidence_digest": digest,
                "commit": new_commit,
                "completion_authority": False,
            },
            evidence_digests=[digest],
        )
    body = {
        "schema": SCHEMA,
        "operation": OPERATION,
        "settlement_id": settlement_id,
        "attempt_id": str(receipt.get("attempt_id") or ""),
        "task_cid": str(task.task_cid),
        "task_alias": parent_alias,
        "repair_task_cid": repair_cid or "",
        "repair_error": repair_error,
        "commit": new_commit,
        "evidence_digest": digest,
        "completed": True,
        "completion_authority": False,
        "rearmed": False,
    }
    owner._record_event(
        EVENT,
        attempt_id=str(receipt.get("attempt_id") or ""),
        task_cid=str(task.task_cid),
        body=body,
    )
    return body


def _missing_from_pairs(
    pairs: Sequence[tuple[str, str | None, str | None]],
) -> list[str]:
    from ..validation.proposal_validation import _command_extends, _workflow_command_texts

    missing: list[str] = []
    for _path, before, after in pairs:
        if before is None or after is None:
            continue
        before_commands = _workflow_command_texts(before)
        after_commands = _workflow_command_texts(after)
        for command in before_commands:
            if not any(_command_extends(command, candidate) for candidate in after_commands):
                missing.append(command)
    return missing
