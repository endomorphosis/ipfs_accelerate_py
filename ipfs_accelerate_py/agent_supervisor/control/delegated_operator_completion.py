"""Scheduler-delegated completion of seal-gated manual tasks.

When a program explicitly enables ``delegated_operator_completion`` in its
scheduler profile, the agent supervisor may:

1. run the task's declared validation command;
2. issue an activation-only seal under the honest
   ``delegated_supervisor`` operator profile (not a forged interactive user);
3. pin the receipt identity in the scheduler config; and
4. mark the board task ``Status: completed``.

This is opt-in automation.  It does not grant mutation, promotion, or
task-status authority beyond activation of already-produced artifacts.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manual_completion_seal import (
    DELEGATED_SUPERVISOR_OPERATOR,
    ManualCompletionSealError,
    build_manual_completion_seal,
    verify_manual_completion_seal,
    write_manual_completion_seal,
)


class DelegatedOperatorCompletionError(ValueError):
    """Raised when delegated completion cannot finish fail-closed."""


@dataclass(frozen=True)
class DelegatedOperatorCompletionPolicy:
    """Closed policy loaded from a scheduler profile."""

    enabled: bool
    allowed_task_ids: frozenset[str]
    require_validation: bool
    validation_timeout_seconds: int

    @classmethod
    def disabled(cls) -> DelegatedOperatorCompletionPolicy:
        return cls(
            enabled=False,
            allowed_task_ids=frozenset(),
            require_validation=True,
            validation_timeout_seconds=1800,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> DelegatedOperatorCompletionPolicy:
        if not value:
            return cls.disabled()
        if not isinstance(value, Mapping):
            raise DelegatedOperatorCompletionError(
                "delegated_operator_completion must be an object"
            )
        enabled = bool(value.get("enabled", False))
        raw_ids = value.get("allowed_task_ids") or value.get("task_ids") or []
        if not isinstance(raw_ids, (list, tuple)):
            raise DelegatedOperatorCompletionError(
                "delegated_operator_completion.allowed_task_ids must be a list"
            )
        allowed = frozenset(str(item).strip() for item in raw_ids if str(item).strip())
        timeout = value.get("validation_timeout_seconds", 1800)
        if type(timeout) is not int or timeout < 1 or timeout > 86_400:
            raise DelegatedOperatorCompletionError(
                "delegated_operator_completion.validation_timeout_seconds "
                "must be an integer in [1, 86400]"
            )
        return cls(
            enabled=enabled,
            allowed_task_ids=allowed,
            require_validation=bool(value.get("require_validation", True)),
            validation_timeout_seconds=timeout,
        )

    def allows(self, task_id: str) -> bool:
        return self.enabled and (
            not self.allowed_task_ids or task_id in self.allowed_task_ids
        )


def _mark_todo_completed(todo_path: Path, task_id: str) -> bool:
    text = todo_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"(## {re.escape(task_id)}\b.*?\n- Status: )pending(\n)",
        re.S,
    )
    updated, count = pattern.subn(r"\1completed\2", text, count=1)
    if count == 1:
        todo_path.write_text(updated, encoding="utf-8")
        return True
    if re.search(
        rf"## {re.escape(task_id)}\b.*?\n- Status: completed\n",
        text,
        re.S,
    ):
        return False
    raise DelegatedOperatorCompletionError(
        f"cannot mark {task_id} completed on {todo_path}"
    )


def _run_validation(
    *,
    repo_root: Path,
    command: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not command.strip():
        return {"ran": False, "returncode": 0, "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DelegatedOperatorCompletionError(
            f"validation command failed to execute: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise DelegatedOperatorCompletionError(
            "validation command failed "
            f"(rc={completed.returncode}): {completed.stderr[-2000:]}"
        )
    return {
        "ran": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "command": command,
    }


def _update_scheduler_pin(
    *,
    scheduler_path: Path,
    task_id: str,
    receipt_id: str,
    seal_config: Mapping[str, Any],
) -> None:
    payload = json.loads(scheduler_path.read_text(encoding="utf-8"))
    seals = payload.setdefault("manual_completion_seals", {})
    if task_id not in seals:
        seals[task_id] = {}
    entry = seals[task_id]
    entry["receipt_path"] = seal_config["receipt_path"]
    entry["schema"] = seal_config["schema"]
    entry["interface"] = seal_config["interface"]
    entry["policy_revision"] = seal_config["policy_revision"]
    entry["expected_receipt_id"] = receipt_id
    entry["artifact_paths"] = dict(seal_config["artifact_paths"])
    entry["grant_type"] = seal_config["grant_type"]
    entry["grant_action"] = seal_config["grant_action"]
    entry["grant_claims"] = dict(seal_config.get("grant_claims") or {})
    entry["reviewed_base_claims"] = dict(
        seal_config.get("reviewed_base_claims") or {}
    )
    scheduler_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def complete_sealed_manual_task(
    *,
    repo_root: Path,
    todo_path: Path,
    scheduler_path: Path,
    task_id: str,
    board_namespace: str,
    seal_config: Mapping[str, Any],
    validation_command: str = "",
    policy: DelegatedOperatorCompletionPolicy | None = None,
) -> dict[str, Any]:
    """Validate, seal, pin, and mark one manual seal-gated task complete."""

    policy = policy or DelegatedOperatorCompletionPolicy.disabled()
    if not policy.allows(task_id):
        raise DelegatedOperatorCompletionError(
            f"delegated completion is not enabled for {task_id}"
        )
    root = repo_root.resolve()
    required_artifacts = list(seal_config["artifact_paths"].values())
    missing = [
        path for path in required_artifacts if not (root / path).is_file()
    ]
    if missing:
        raise DelegatedOperatorCompletionError(
            f"{task_id} missing sealed artifacts: {missing}"
        )

    validation: dict[str, Any] = {"ran": False}
    if policy.require_validation:
        validation = _run_validation(
            repo_root=root,
            command=validation_command,
            timeout_seconds=policy.validation_timeout_seconds,
        )

    receipt = build_manual_completion_seal(
        repo_root=root,
        task_id=task_id,
        board_namespace=board_namespace,
        schema=str(seal_config["schema"]),
        interface=str(seal_config["interface"]),
        policy_revision=str(seal_config["policy_revision"]),
        artifact_paths=dict(seal_config["artifact_paths"]),
        grant_type=str(seal_config["grant_type"]),
        grant_action=str(seal_config["grant_action"]),
        reviewed_base_claims=dict(seal_config.get("reviewed_base_claims") or {}),
        grant_claims=dict(seal_config.get("grant_claims") or {}),
        operator=DELEGATED_SUPERVISOR_OPERATOR,
    )
    seal_path = write_manual_completion_seal(
        str(seal_config["receipt_path"]),
        receipt,
        repo_root=root,
    )
    _update_scheduler_pin(
        scheduler_path=scheduler_path,
        task_id=task_id,
        receipt_id=str(receipt["receipt_id"]),
        seal_config=seal_config,
    )
    try:
        verified = verify_manual_completion_seal(
            str(seal_config["receipt_path"]),
            repo_root=root,
            task_id=task_id,
            board_namespace=board_namespace,
            schema=str(seal_config["schema"]),
            interface=str(seal_config["interface"]),
            policy_revision=str(seal_config["policy_revision"]),
            expected_receipt_id=str(receipt["receipt_id"]),
            artifact_paths=dict(seal_config["artifact_paths"]),
            grant_type=str(seal_config["grant_type"]),
            grant_action=str(seal_config["grant_action"]),
            reviewed_base_claims=dict(
                seal_config.get("reviewed_base_claims") or {}
            ),
            grant_claims=dict(seal_config.get("grant_claims") or {}),
            allow_delegated_operator=True,
        )
    except ManualCompletionSealError as exc:
        raise DelegatedOperatorCompletionError(
            f"delegated seal verification failed: {exc}"
        ) from exc

    board_changed = _mark_todo_completed(todo_path, task_id)
    return {
        "completed": True,
        "task_id": task_id,
        "receipt_id": verified["receipt_id"],
        "seal_path": str(seal_path.relative_to(root)),
        "board_changed": board_changed,
        "operator": DELEGATED_SUPERVISOR_OPERATOR,
        "validation": validation,
    }


def complete_ready_sealed_manual_tasks(
    *,
    repo_root: Path,
    todo_path: Path,
    scheduler_path: Path,
    board_namespace: str,
    seal_configs: Mapping[str, Mapping[str, Any]],
    validation_commands: Mapping[str, str],
    completed_task_ids: Sequence[str],
    pending_task_ids: Sequence[str],
    depends_on: Mapping[str, Sequence[str]],
    policy: DelegatedOperatorCompletionPolicy,
) -> dict[str, Any]:
    """Attempt delegated completion for every currently eligible sealed task."""

    if not policy.enabled:
        return {
            "enabled": False,
            "attempted": [],
            "completed": [],
            "skipped": [],
            "errors": [],
        }

    completed = set(completed_task_ids)
    results: dict[str, Any] = {
        "enabled": True,
        "attempted": [],
        "completed": [],
        "skipped": [],
        "errors": [],
    }
    # Fixed-point: completing 060 may unlock later seal tasks in the same pass
    # only when their depends_on are already satisfied.
    for task_id in pending_task_ids:
        if not policy.allows(task_id):
            results["skipped"].append(
                {"task_id": task_id, "reason": "not_in_allowed_task_ids"}
            )
            continue
        seal = seal_configs.get(task_id)
        if seal is None:
            results["skipped"].append(
                {"task_id": task_id, "reason": "no_seal_config"}
            )
            continue
        deps = list(depends_on.get(task_id) or ())
        missing_deps = [dep for dep in deps if dep not in completed]
        if missing_deps:
            results["skipped"].append(
                {
                    "task_id": task_id,
                    "reason": "dependencies_incomplete",
                    "missing_deps": missing_deps,
                }
            )
            continue
        results["attempted"].append(task_id)
        try:
            outcome = complete_sealed_manual_task(
                repo_root=repo_root,
                todo_path=todo_path,
                scheduler_path=scheduler_path,
                task_id=task_id,
                board_namespace=board_namespace,
                seal_config=seal,
                validation_command=str(validation_commands.get(task_id) or ""),
                policy=policy,
            )
        except (
            DelegatedOperatorCompletionError,
            ManualCompletionSealError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            results["errors"].append(
                {"task_id": task_id, "error": str(exc)}
            )
            continue
        results["completed"].append(outcome)
        completed.add(task_id)
    return results


__all__ = [
    "DelegatedOperatorCompletionError",
    "DelegatedOperatorCompletionPolicy",
    "complete_ready_sealed_manual_tasks",
    "complete_sealed_manual_task",
]
