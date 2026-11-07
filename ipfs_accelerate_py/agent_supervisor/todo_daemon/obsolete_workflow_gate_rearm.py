"""Rearm one blocked task when a workflow gate rejection is now obsolete.

The proposal gate used to call every non-additive edit of an existing
workflow a weakening. A task-owned edit that keeps every ``run`` and ``uses``
command, and adds no bypass, is not a weakening. When that corrected rule
admits the preserved commit, the blocked control row is CAS'd back to
retrying once per checker revision. The control receipt is the closed
operator rearm the coordination fence already requires. This does not mark
the task complete, reset its attempt counter, or admit a diff the current
rule still rejects.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..validation.proposal_validation import (
    workflow_gate_checker_revision,
    workflow_pairs_are_non_weakening,
)

EVENT = "obsolete_workflow_gate_rearmed"
SCHEMA = "database-obsolete-workflow-gate-rearm@1"
OPEN_OPERATION = "obsolete_workflow_gate_rearm"
CLOSED_OPERATION = "operator_control_plane_repair"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_WORKFLOW_SUFFIXES = (".yml", ".yaml")
Pair = tuple[str, str | None, str | None]
PairLoader = Callable[[Any, Any, Mapping[str, Any]], Sequence[Pair] | None]


def _receipt(task: Any) -> Mapping[str, Any] | None:
    body = getattr(task, "body", None)
    if not isinstance(body, Mapping):
        return None
    receipt = body.get("completion_receipt")
    if not isinstance(receipt, Mapping):
        return None
    return receipt


def _attempt_id_ok(value: Any) -> bool:
    text = str(value or "")
    token = text.removeprefix("attempt:")
    return text.startswith("attempt:") and bool(re.fullmatch(r"[0-9a-f]{32}", token))


def receipt_is_obsolete_workflow_gate_block(receipt: Mapping[str, Any]) -> bool:
    """True for the settled claim-failure row this recovery knows how to read.

    The weakening proof is not in this receipt. It lives in the attempt event
    log. The receipt only proves a terminal portal failure with no effect claim
    and a provider that actually ran.
    """

    attempt_number = receipt.get("attempt_number")
    return (
        receipt.get("schema")
        == "ipfs_accelerate_py/agent-supervisor/task-claim-failure-settlement@1"
        and receipt.get("operation") == "database_task_claim_failure"
        and receipt.get("failure_kind") == "terminal_portal_bridge_error"
        and receipt.get("automatic_retry_admitted") is False
        and receipt.get("provider_invocation_count") == 1
        and receipt.get("effect_claim_count") == 0
        and type(attempt_number) is int
        and attempt_number >= 1
        and _attempt_id_ok(receipt.get("attempt_id"))
    )


def events_prove_obsolete_workflow_weakening(
    events: Sequence[Mapping[str, Any]],
) -> bool:
    """True when the only gate finding is validation_weakening_forbidden."""

    rejected = [
        event for event in events if event.get("type") == "implementation_proposal_rejected"
    ]
    reviewed = [
        event for event in events if event.get("type") == "implementation_failure_reviewed"
    ]
    finished = [
        event for event in events if event.get("type") == "implementation_finished"
    ]
    if len(rejected) != 1 or len(reviewed) != 1 or len(finished) != 1:
        return False
    changed = rejected[0].get("changed_paths")
    return (
        rejected[0].get("accepted") is False
        and rejected[0].get("reason_codes") == ["validation_weakening_forbidden"]
        and reviewed[0].get("decision") == "reject"
        and reviewed[0].get("finding_codes") == ["validation_weakening_forbidden"]
        and finished[0].get("returncode") == 78
        and isinstance(changed, list)
        and any(
            isinstance(path, str) and path.startswith(".github/workflows/")
            for path in changed
        )
    )


def preserved_workflow_revisions(
    events: Sequence[Mapping[str, Any]],
) -> tuple[str, str] | None:
    """Return the preserved baseline and commit from the finished event."""

    for event in reversed(list(events)):
        if event.get("type") != "implementation_finished":
            continue
        baseline = str(event.get("baseline_ref") or "")
        commit = str(event.get("implementation_commit") or "")
        if (
            _HEX40.fullmatch(baseline)
            and _HEX40.fullmatch(commit)
            and baseline != commit
        ):
            return baseline, commit
    return None


def _workflow_path(path: str) -> bool:
    if not path or path.startswith("/") or ".." in path.split("/"):
        return False
    parts = path.split("/")
    return (
        len(parts) >= 3
        and parts[0] == ".github"
        and parts[1] == "workflows"
        and path.endswith(_WORKFLOW_SUFFIXES)
    )


def _git(repo: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_text(repo: Path, rev: str, path: str) -> str | None:
    if _HEX40.fullmatch(rev) is None or not _workflow_path(path):
        return None
    result = _git(repo, ["show", f"{rev}:{path}"])
    if result is None or result.returncode != 0:
        return None
    return result.stdout


def load_preserved_workflow_pairs(
    owner: Any,
    task: Any,
    receipt: Mapping[str, Any],
) -> list[Pair] | None:
    """Load workflow before/after text for the preserved commit."""

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
    events_path = bridge._paths(attempt).events
    try:
        raw = events_path.read_text(encoding="utf-8")
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
    if not events_prove_obsolete_workflow_weakening(events):
        return None
    revisions = preserved_workflow_revisions(events)
    if revisions is None:
        return None
    baseline, commit = revisions
    listed = _git(Path(bridge.repo_root), ["diff", "--name-only", baseline, commit])
    if listed is None or listed.returncode != 0:
        return None
    paths = [
        path
        for path in listed.stdout.splitlines()
        if _workflow_path(path.strip())
    ]
    if not paths or len(paths) > 16:
        return None
    pairs: list[Pair] = []
    for path in paths:
        before = _git_text(Path(bridge.repo_root), baseline, path)
        after = _git_text(Path(bridge.repo_root), commit, path)
        if after is None:
            return None
        pairs.append((path, before, after))
    return pairs


def _already_rearmed(owner: Any, task_cid: str, revision: str) -> bool:
    rows = owner._require_connection().execute(
        "SELECT body_json FROM daemon_execution_events "
        "WHERE event_type = ? AND task_cid = ?",
        [EVENT, task_cid],
    ).fetchall()
    for row in rows:
        try:
            body = json.loads(row[0])
        except (TypeError, ValueError, IndexError):
            return True
        if (
            isinstance(body, dict)
            and body.get("checker_revision") == revision
            and body.get("rearmed") is True
        ):
            return True
    return False


def _closed_receipt(failure: Mapping[str, Any]) -> dict[str, Any] | None:
    settlement_id = failure.get("settlement_id")
    expected_revision = failure.get("control_expected_revision")
    if (
        type(settlement_id) is not str
        or not settlement_id
        or settlement_id != settlement_id.strip()
        or len(settlement_id) > 256
        or type(expected_revision) is not int
        or expected_revision < 1
    ):
        return None
    return {
        "operation": CLOSED_OPERATION,
        "settlement_id": settlement_id,
    }


def stale_unknown_callback_yields_to_closed_rearm(
    *,
    task_status: str,
    control_receipt: Mapping[str, Any] | None,
    attempt_status: str,
    attempt_task_cid: str,
    task_cid: str,
    attempt_owner_session_id: str,
    owner_session_id: str,
    effect_claim_present: bool,
    claim_state: str,
    rearm_settlement_id: str,
) -> bool:
    """True when a stored unknown callback must not hide a closed rearm.

    The callback row stays unknown. A successor is claimable only for a
    failed attempt of this task with a released claim and no effect claim,
    after coordination has rearmed the settlement named by the closed receipt.
    """

    return (
        str(task_status or "").strip().lower() == "retrying"
        and isinstance(control_receipt, Mapping)
        and set(control_receipt) == {"operation", "settlement_id"}
        and control_receipt.get("operation") == CLOSED_OPERATION
        and isinstance(rearm_settlement_id, str)
        and rearm_settlement_id != ""
        and control_receipt.get("settlement_id") == rearm_settlement_id
        and attempt_status == "failed"
        and attempt_task_cid == task_cid
        and bool(task_cid)
        and attempt_owner_session_id == owner_session_id
        and bool(owner_session_id)
        and effect_claim_present is False
        and claim_state in {"released", "expired", "superseded"}
    )


def close_open_receipt(owner: Any, task: Any, failure: Mapping[str, Any]) -> Any:
    """Close an already-retrying workflow rearm receipt at its current revision.

    A same-status CAS does not change the receipt, and another status CAS
    would leave the coordination fence's revision check. Only the open
    workflow receipt is replaced, and only when this row is already the
    required block-then-requeue revision.
    """

    closed = _closed_receipt(failure)
    if closed is None:
        return task
    expected_revision = int(failure["control_expected_revision"]) + 2
    if (
        str(getattr(task, "status", "") or "").strip().lower() != "retrying"
        or int(getattr(task, "revision", 0) or 0) != expected_revision
    ):
        return task
    body = getattr(task, "body", None)
    current = body.get("completion_receipt") if isinstance(body, Mapping) else None
    if not isinstance(current, Mapping):
        return task
    if dict(current) == closed:
        return task
    if current.get("operation") != OPEN_OPERATION:
        return task
    source = getattr(owner, "task_source", None)
    replace = getattr(source, "replace_retrying_rearm_receipt", None)
    if not callable(replace):
        return task
    task_cid = str(getattr(task, "task_cid", "") or "")
    if not task_cid:
        return task
    if (
        replace(
            task_cid,
            expected_revision=expected_revision,
            expected_operation=OPEN_OPERATION,
            receipt=closed,
        )
        is not True
    ):
        return task
    refreshed = source.get_task(task_cid)
    return task if refreshed is None else refreshed


def rearm(
    owner: Any,
    task: Any,
    *,
    load_pairs: PairLoader | None = None,
) -> dict[str, Any] | None:
    """CAS blocked to retrying when the preserved workflow diff now passes."""

    if str(getattr(task, "status", "") or "").strip().lower() != "blocked":
        return None
    running = getattr(owner, "list_running_attempts", None)
    if callable(running) and list(running() or ()):
        return None
    receipt = _receipt(task)
    if receipt is None or not receipt_is_obsolete_workflow_gate_block(receipt):
        return None
    task_cid = str(getattr(task, "task_cid", "") or "")
    attempt_id = str(receipt.get("attempt_id") or "")
    compact = _closed_receipt(receipt)
    if not task_cid or not attempt_id or compact is None:
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
    revision = workflow_gate_checker_revision()
    if _already_rearmed(owner, task_cid, revision):
        return None
    loader = load_pairs or load_preserved_workflow_pairs
    pairs = loader(owner, task, receipt)
    if not pairs or not workflow_pairs_are_non_weakening(list(pairs)):
        return None
    preserved_commit = ""
    if isinstance(receipt.get("preserved_commit"), str):
        preserved_commit = str(receipt.get("preserved_commit") or "")
    changed = owner.task_source.rearm_blocked_task(task_cid, receipt=compact)
    if getattr(changed, "changed", False) is not True:
        return None
    coordinator = getattr(owner, "coordinator", None)
    finish = getattr(coordinator, "rearm_failed_task", None)
    observation = changed.to_dict() if callable(getattr(changed, "to_dict", None)) else None
    if callable(finish) and observation is not None:
        kwargs: dict[str, Any] = {}
        now = getattr(owner, "_now_ms", None)
        if callable(now):
            kwargs["now_ms"] = now()
        finish(
            failure_receipt=dict(receipt),
            control_task_observation=observation,
            **kwargs,
        )
    body = {
        "schema": SCHEMA,
        "operation": compact["operation"],
        "settlement_id": compact["settlement_id"],
        "attempt_id": attempt_id,
        "checker_revision": revision,
        "completion_authority": False,
        "paid_attempt_budget_reset": False,
        "task_cid": task_cid,
        "task_alias": str(getattr(task, "task_alias", "") or ""),
        "rearmed": True,
        "preserved_commit": preserved_commit,
        "workflow_paths": [path for path, _before, _after in pairs],
    }
    owner._record_event(
        EVENT,
        attempt_id=attempt_id,
        task_cid=task_cid,
        body=body,
    )
    return body
