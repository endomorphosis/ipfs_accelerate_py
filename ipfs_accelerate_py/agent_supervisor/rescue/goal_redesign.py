"""Deterministic, and optionally model-assisted, redesign of blocked goals.

The sealed lean board stays at its registered rows. This planner records the
next legal step for each open goal. It does not mark a goal scientifically
complete, spend a model call, invent a missing external file, or change a
planned task to pending.

A receipt that claims completion authority is corrected back to false. A
length-truncated spent phase, a planned row this owner's compiled lineage
cannot admit, and an official release that is still missing are retained.
None of those corrections grants completion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "ipfs_accelerate_py/agent-supervisor/goal-redesign@1"
_GOALS = Path("implementation_plan/lean_refactor_arena_20261025.continuation.goals.json")
_DONE = {"completed", "complete", "done"}
_RUNNABLE = {"pending", "ready", "in_progress"}
_TRUNCATED_TASK = "JLR-024"
_TRUNCATED_RESULT = Path("data/lean_refactor_arena_20261024/execution-v2/JLR-024.result.json")
_CLOSED_TASK = "JLR-022"
_RELEASE_TASK = "JLR-035"
_RELEASE_RESULT = Path("data/lean_refactor_arena_20261024/execution-v2/JLR-035.result.json")
_PROFILE = Path("config/lean_refactor_arena_20261024_execution_v2.json")
_LINEAGE = Path("tools/lean_refactor_compiled_sources_v1.json")
_RESULT_PATHS = {
    _TRUNCATED_TASK: _TRUNCATED_RESULT,
    _RELEASE_TASK: _RELEASE_RESULT,
}
_TRUNCATED_REASON = (
    "both bounded calls returned finish_reason length and no closed proof; "
    "retain the spent phase, do not spend another call, and do not grant completion"
)
_CLOSED_REASON = (
    "this owner's compiled lineage keeps the published contracts unchanged; "
    "retain the planned row, do not flip it to pending, and do not grant completion"
)
_RELEASE_REASON = (
    "official full cohort file is unavailable; retain the release wait, "
    "do not invent the file, and do not grant completion"
)


def _deps(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _task(task_id: str, fields: Mapping[str, Any], goal_task: Mapping[str, Any]) -> dict[str, Any]:
    work = goal_task.get("work_package") if isinstance(goal_task.get("work_package"), dict) else {}
    requires = work.get("activation_requires")
    if not isinstance(requires, list):
        requires = []
    return {
        "task_id": task_id,
        "status": str(fields.get("status") or goal_task.get("initial_status") or "").strip().lower(),
        "goal_id": str(fields.get("goal_id") or goal_task.get("goal_id") or "").strip(),
        "depends": _deps(fields.get("depends_on", fields.get("depends", goal_task.get("depends_on")))),
        "action": str(work.get("action") or ""),
        "activation_requires": [str(item) for item in requires if str(item).strip()],
    }


def _closed_model_proof(content: str) -> bool:
    """True when the model body already contains a closed proof object.

    A closed body is not treated as truncation. This does not accept the proof.
    """

    text = content.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        body = text[newline + 1:] if newline >= 0 else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        text = body.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    proof = parsed.get("proof") if isinstance(parsed, dict) else None
    return isinstance(proof, str) and bool(proof.strip())


def truncated_spent_phase(document: Mapping[str, Any] | None) -> bool:
    """True only for a spent phase whose bounded calls ended on the length cap.

    The result stays unchanged. A winner, a completion claim, a promoted row,
    or a body that already parses as a proof is not this shape.
    """

    if not isinstance(document, Mapping):
        return False
    if document.get("status") != "NO_CONFIRMED_WINNER":
        return False
    if document.get("winner") not in (None, False):
        return False
    if document.get("completion_authority") is not False or document.get("proof_authority") is not False:
        return False
    if document.get("promoted") is not False or document.get("full_campaign_complete") is not False:
        return False
    limits = document.get("limits")
    attempts = document.get("attempts")
    if not isinstance(limits, Mapping) or not isinstance(attempts, list) or not attempts:
        return False
    max_calls = limits.get("max_calls")
    if type(max_calls) is not int or max_calls < 1 or len(attempts) != max_calls:
        return False
    if document.get("local_inference_calls") != max_calls:
        return False
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            return False
        if attempt.get("status") != "RETURNED" or attempt.get("http_status") != 200:
            return False
        if not str(attempt.get("candidate_error") or "").strip():
            return False
        if "sample_index" in attempt and attempt.get("sample_index") is not None:
            return False
        response = attempt.get("response")
        choices = response.get("choices") if isinstance(response, Mapping) else None
        if not isinstance(choices, list) or not choices:
            return False
        for choice in choices:
            if not isinstance(choice, Mapping) or choice.get("finish_reason") != "length":
                return False
            message = choice.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip() or _closed_model_proof(content):
                return False
    return True


def _json_object(path: Path, limit: int = 1_000_000) -> dict[str, Any] | None:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _contract_ids(document: Mapping[str, Any]) -> set[str] | None:
    contracts = document.get("contracts")
    if not isinstance(contracts, dict) or not contracts:
        return None
    if any(not isinstance(key, str) or not key for key in contracts):
        return None
    return set(contracts)


def closed_lineage_planned(repo_root: Path, task_id: str = _CLOSED_TASK) -> bool:
    """True when admitting task_id would change this owner's sealed contract set.

    The published profile and the lineage baseline must name the same
    contracts, and task_id must be absent from both. A drifted profile, a
    symlink, or a path outside the repo is not treated as closed.
    """

    if task_id != _CLOSED_TASK:
        return False
    inventory = _json_object(repo_root / _LINEAGE)
    profile = _json_object(repo_root / _PROFILE)
    if inventory is None or profile is None:
        return False
    relative = inventory.get("baseline_profile")
    if not isinstance(relative, str) or not relative or relative.startswith("/"):
        return False
    parts = Path(relative).parts
    if ".." in parts:
        return False
    baseline_path = (repo_root / relative).resolve()
    try:
        baseline_path.relative_to(repo_root.resolve())
    except ValueError:
        return False
    baseline = _json_object(baseline_path)
    if baseline is None:
        return False
    live_ids = _contract_ids(profile)
    base_ids = _contract_ids(baseline)
    if live_ids is None or base_ids is None:
        return False
    return live_ids == base_ids and task_id not in live_ids


def waiting_official_release(document: Mapping[str, Any] | None) -> bool:
    """True for a JLR-035 wait that has not called a model and claims no completion.

    The result stays unchanged. A completion claim, a model call, or a promoted
    row is not this shape, and the missing cohort file is not invented.
    """

    if not isinstance(document, Mapping):
        return False
    if document.get("status") != "WAITING_RELEASE":
        return False
    if document.get("reason") != "official_full_cohort_file_unavailable":
        return False
    if type(document.get("model_calls")) is not int or document.get("model_calls") != 0:
        return False
    if document.get("completion_authority") is not False or document.get("proof_authority") is not False:
        return False
    if document.get("full_campaign_complete") is not False or document.get("promoted") is True:
        return False
    return True


def _phase_result(repo_root: Path, task_id: str) -> dict[str, Any] | None:
    relative = _RESULT_PATHS.get(task_id)
    if relative is None:
        return None
    return _json_object(repo_root / relative)


def truncated_phase_retained(repo_root: Path) -> bool:
    """Read the sealed JLR-024 result and report whether it is a truncated spent phase."""

    return truncated_spent_phase(_phase_result(repo_root, _TRUNCATED_TASK))


def official_release_retained(repo_root: Path) -> bool:
    """Read the sealed JLR-035 result and report whether the official cohort is still absent."""

    return waiting_official_release(_phase_result(repo_root, _RELEASE_TASK))


def _work_complete_allowed(node: Mapping[str, Any]) -> bool:
    decisions = node.get("decisions")
    subgoals = node.get("subgoals")
    mutations = node.get("board_mutations")
    if not isinstance(decisions, list) or not decisions:
        return False
    if not isinstance(subgoals, list) or subgoals:
        return False
    if not isinstance(mutations, list) or mutations:
        return False
    return all(
        isinstance(item, Mapping)
        and item.get("action") == "task_status_complete"
        and item.get("completion_authority") is False
        and item.get("board_mutation") is False
        for item in decisions
    )


def correct_completion_claims(document: Any) -> tuple[Any, int]:
    """Force illegal completion claims off. Never grants completion authority.

    ``work_complete`` may stay true only when every decision is already
    ``task_status_complete``, no subgoal remains, and no board mutation is recorded.
    """

    changed = 0

    def walk(node: Any) -> Any:
        nonlocal changed
        if isinstance(node, dict):
            corrected = {key: walk(value) for key, value in node.items()}
            if "completion_authority" in corrected and corrected["completion_authority"] is not False:
                corrected["completion_authority"] = False
                changed += 1
            if "work_complete" in corrected and corrected["work_complete"] is not False and not _work_complete_allowed(corrected):
                corrected["work_complete"] = False
                changed += 1
            return corrected
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(document), changed


def enforce_completion_claims(path: Path) -> int:
    """Rewrite one supervisor receipt that claims completion it does not have.

    A missing file, a symlink, or an unreadable document is left untouched.
    Returns the number of claims cleared.
    """

    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_000_000:
            return 0
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0
    corrected, count = correct_completion_claims(document)
    if count == 0:
        return 0
    payload = json.dumps(corrected, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".partial")
    try:
        if path.is_symlink():
            return 0
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        try:
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
        except OSError:
            pass
        return 0
    return count


def _decision(
    task: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    *,
    allow_new_model_calls: bool,
    gate_reason: str,
    phase_result: Mapping[str, Any] | None = None,
    closed_lineage: bool = False,
) -> dict[str, Any]:
    missing = [dep for dep in task["depends"] if str((by_id.get(dep) or {}).get("status") or "") not in _DONE]
    base = {
        "task_id": task["task_id"],
        "goal_id": task["goal_id"],
        "status": task["status"],
        "board_mutation": False,
        "completion_authority": False,
        "provider_calls": False,
    }
    if task["status"] in _DONE:
        return {**base, "action": "task_status_complete", "reason": "board status is completed"}
    if missing:
        return {**base, "action": "wait_dependencies", "reason": "blocked by " + ", ".join(missing), "blocked_by": missing}
    if task["status"] == "planned":
        requires = task["activation_requires"] or ["reviewed execution contract"]
        if task["task_id"] == _CLOSED_TASK and closed_lineage:
            return {
                **base,
                "action": "retain_closed_lineage",
                "reason": _CLOSED_REASON,
                "activation_requires": requires,
            }
        return {
            **base,
            "action": "review_execution_contract",
            "reason": "dependency-ready planned work stays planned until a reviewed execution contract exists",
            "activation_requires": requires,
        }
    text = gate_reason.lower()
    if task["task_id"] == _RELEASE_TASK and waiting_official_release(phase_result):
        return {**base, "action": "retain_official_release", "reason": _RELEASE_REASON}
    if task["task_id"] == _TRUNCATED_TASK and truncated_spent_phase(phase_result):
        return {**base, "action": "retain_truncated_phase", "reason": _TRUNCATED_REASON}
    if "phase_already_attempted" in text:
        return {
            **base,
            "action": "register_reviewed_phase",
            "reason": "spent phase stays closed until a separately reviewed bounded phase retains all prior usage",
        }
    if "cohort" in text or ("official" in text and "unavailable" in text):
        return {
            **base,
            "action": "retry_external_release",
            "reason": "external release file is unavailable; retry the published source and do not invent the file",
        }
    if task["status"] in _RUNNABLE:
        reason = gate_reason or "runnable task remains on its registered gate"
        if not allow_new_model_calls:
            reason = reason + "; a new model call is not authorized"
        return {**base, "action": "retain_external_gate", "reason": reason}
    return {**base, "action": "retain_external_gate", "reason": "registered gate remains in force"}


def _proposals(
    raw: Any,
    tasks: Mapping[str, Mapping[str, Any]],
    *,
    allow_new_model_calls: bool,
    allow_sealed_board_mutation: bool,
) -> list[dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return []
    items = raw.get("proposals")
    if not isinstance(items, list):
        return []
    accepted: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("task_id") or "").strip()
        action = str(item.get("action") or "").strip()
        if not task_id or not action or len(accepted) == 8:
            continue
        if item.get("provider_calls") is True and not allow_new_model_calls:
            continue
        if task_id in tasks and not allow_sealed_board_mutation:
            continue
        accepted.append({
            "task_id": task_id,
            "goal_id": str(item.get("goal_id") or ""),
            "action": action,
            "rationale": str(item.get("rationale") or "")[:500],
            "admission": "proposal_only",
            "board_mutation": False,
            "completion_authority": False,
        })
    return accepted


def redesign_goals(
    tasks: Mapping[str, Mapping[str, Any]],
    *,
    goal_tasks: Mapping[str, Mapping[str, Any]] | None = None,
    allow_provider_calls: bool = False,
    allow_new_model_calls: bool = False,
    allow_sealed_board_mutation: bool = False,
    gate_reasons: Mapping[str, str] | None = None,
    phase_results: Mapping[str, Mapping[str, Any]] | None = None,
    closed_lineage: bool = False,
    llm_complete: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the next legal step for every task. The board is not written."""

    goal_tasks = goal_tasks or {}
    ids = sorted(set(tasks) | set(goal_tasks))
    normalized = {
        task_id: _task(task_id, tasks.get(task_id) or {}, goal_tasks.get(task_id) or {})
        for task_id in ids
    }
    gate_reasons = gate_reasons or {}
    phase_results = phase_results or {}
    decisions = [
        _decision(
            normalized[task_id],
            normalized,
            allow_new_model_calls=allow_new_model_calls,
            gate_reason=str(gate_reasons.get(task_id) or ""),
            phase_result=phase_results.get(task_id) if isinstance(phase_results.get(task_id), Mapping) else None,
            closed_lineage=closed_lineage,
        )
        for task_id in ids
    ]
    plan: dict[str, Any] = {
        "schema": SCHEMA,
        "mechanism": "deterministic",
        "board_mutations": [],
        "work_complete": False,
        "completion_authority": False,
        "decisions": decisions,
        "llm": {"used": False, "reason": "provider_calls_disabled", "proposals": []},
    }
    if allow_provider_calls and llm_complete is not None:
        proposals = _proposals(
            llm_complete({"schema": SCHEMA, "decisions": decisions}),
            normalized,
            allow_new_model_calls=allow_new_model_calls,
            allow_sealed_board_mutation=allow_sealed_board_mutation,
        )
        plan["mechanism"] = "deterministic+llm"
        plan["llm"] = {"used": True, "reason": "", "proposals": proposals}
    elif allow_provider_calls:
        plan["llm"]["reason"] = "completer_missing"
    return _attach_subgoals(plan)


def _attach_subgoals(plan: dict[str, Any]) -> dict[str, Any]:
    """Open decisions become proposal-only subgoals. The sealed board is not written."""

    plan["subgoals"] = [
        {
            "subgoal_id": f"{item['task_id']}:{item['action']}",
            "task_id": item["task_id"],
            "goal_id": item["goal_id"],
            "action": item["action"],
            "reason": item["reason"],
            "admission": "proposal_only",
            "board_mutation": False,
            "completion_authority": False,
        }
        for item in plan["decisions"]
        if item.get("action") != "task_status_complete"
    ]
    plan["work_complete"] = not plan["subgoals"]
    plan["completion_authority"] = False
    plan["idle_incomplete"] = bool(plan["subgoals"])
    return plan


def _load_goal_tasks(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = repo_root / _GOALS
    if not path.is_file() or path.is_symlink():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    records = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(records, list):
        return {}
    return {
        str(record.get("id")): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }


def _gate_reasons(repo_root: Path) -> dict[str, str]:
    reasons: dict[str, str] = {}
    paths = (
        Path("data/lean_refactor_arena_20261024/execution-v2/iterative-admission-gate.json"),
        Path("data/lean_refactor_arena_20261024/execution-v2/release-gate.json"),
    )
    for relative in paths:
        path = repo_root / relative
        try:
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_000_000:
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        task_id = str(document.get("task_id") or "")
        reason = str(document.get("reason") or "")
        if task_id and reason:
            reasons[task_id] = reason
    result = repo_root / "data/lean_refactor_arena_20261024/execution-v2/JLR-035.result.json"
    try:
        if result.is_file() and not result.is_symlink() and result.stat().st_size <= 1_000_000:
            document = json.loads(result.read_text(encoding="utf-8"))
            if (
                isinstance(document, dict)
                and document.get("status") == "WAITING_RELEASE"
                and document.get("reason")
            ):
                reasons["JLR-035"] = str(document["reason"])
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    return reasons


def write_goal_redesign(
    repo_root: Path,
    blocks: Mapping[str, Mapping[str, str]],
    receipt_path: Path,
) -> dict[str, Any]:
    """Write one redesign receipt. Existing bytes are kept when the plan is unchanged."""

    phase_results: dict[str, Mapping[str, Any]] = {}
    for task_id in (_TRUNCATED_TASK, _RELEASE_TASK):
        loaded = _phase_result(repo_root, task_id)
        if loaded is not None:
            phase_results[task_id] = loaded
    plan = redesign_goals(
        blocks,
        goal_tasks=_load_goal_tasks(repo_root),
        gate_reasons=_gate_reasons(repo_root),
        phase_results=phase_results,
        closed_lineage=closed_lineage_planned(repo_root),
    )
    plan, _corrections = correct_completion_claims(plan)
    payload = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    try:
        if receipt_path.is_symlink():
            return plan
        if receipt_path.is_file() and receipt_path.read_text(encoding="utf-8") == payload:
            return plan
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_path.with_name(receipt_path.name + ".partial")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, receipt_path)
    except OSError:
        return plan
    return plan
