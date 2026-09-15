#!/usr/bin/env python3
"""SPAR fleet publication gate over native closeout facts. Not an LLM.

The detached 2026-09-09 placeholder always held with
``sealed_spar_goal_and_current_source_root_acceptance_adapter_required``.
This adapter re-reads the live owner's authoritative-status snapshot and the
sealed SPAR closeout profile. ``authoritative``/``complete`` become true only
when native ``completion_authority`` is true and independent population,
claim, merge, obligation, and source-head checks still hold. Bootstrap
admission, nominated reports, and task counts cannot lift the gate.
"""

from __future__ import annotations

import collections
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "spar/native-closeout-gate@1"
BOARD_ID = "spar"
STATUS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/database-board-status@1"
PROFILE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/spar-closeout-requirements@1"
REPOS = {
    "accelerator": ".",
    "datasets": "ipfs_datasets_py",
    "kit": "ipfs_kit_py",
    "mcpplusplus": "ipfs_accelerate_py/mcplusplus",
}
TERMINALS = (
    "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json",
    "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_FINAL_REPORT.md",
    "test/api/semantic_refactoring/test_release_gate.py",
)


def _run(argv: list[str], cwd: Path, timeout: float = 45) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _git_head(root: Path) -> tuple[str, bool]:
    head = _run(["git", "rev-parse", "HEAD"], root)
    dirty = _run(["git", "status", "--porcelain", "--ignore-submodules=none"], root)
    return head.stdout.strip(), bool(dirty.stdout.strip())


def _merge_queue_paths(runtime: Path) -> list[str]:
    pending: list[str] = []
    for directory in ("pending", "processing", "failed", "quarantine"):
        pending.extend(str(path) for path in (runtime / "merge-queue" / directory).glob("*.json"))
    return pending


def evaluate(
    root: Path,
    *,
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify SPAR publication from native closeout facts. Never self-authorizes."""

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "board_id": BOARD_ID,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "authoritative": False,
        "complete": False,
        "active_claims": None,
        "pending_merges": None,
        "blocking_obligations": None,
        "source_heads": {},
        "blockers": [],
    }
    for key, rel in REPOS.items():
        repo = root if rel == "." else root / rel
        if not repo.is_dir():
            result["blockers"].append("repository_unavailable:" + key)
            continue
        try:
            head, dirty = _git_head(repo)
        except (OSError, subprocess.TimeoutExpired):
            result["blockers"].append("repository_unavailable:" + key)
            continue
        if not head:
            result["blockers"].append("repository_unavailable:" + key)
            continue
        result["source_heads"][key] = head
        if dirty:
            result["blockers"].append("source_checkout_dirty:" + key)
    if observation is None:
        try:
            status = _run(
                [
                    "python3",
                    "scripts/materialize_semantic_preserving_remodularization_program.py",
                    "authoritative-status",
                ],
                root,
                timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired):
            result["blockers"].append("native_closeout_admission_unavailable")
            return result
        if status.returncode:
            result["blockers"].append("native_closeout_admission_unavailable")
            return result
        try:
            observation = json.loads(status.stdout)
        except json.JSONDecodeError:
            result["blockers"].append("native_closeout_observation_rejected")
            return result
    if not isinstance(observation, Mapping):
        result["blockers"].append("native_closeout_observation_rejected")
        return result
    if (
        observation.get("schema") != STATUS_SCHEMA
        or observation.get("authoritative_task_observation") is not True
    ):
        result["blockers"].append("native_closeout_observation_rejected")
        return result
    result["owner_identity"] = observation.get("owner_identity")
    snapshot = observation.get("closeout_snapshot")
    if not isinstance(snapshot, Mapping):
        result["blockers"].append("native_closeout_population_truncated_or_missing")
        return result
    result["snapshot_id"] = snapshot.get("snapshot_cid")
    facts = snapshot.get("closeout_facts")
    if not isinstance(facts, Mapping):
        result["blockers"].append("native_closeout_population_truncated_or_missing")
        return result
    profile = facts.get("completion_profile")
    if not isinstance(profile, Mapping) or profile.get("schema") != PROFILE_SCHEMA:
        result["blockers"].append("native_spar_completion_profile_unavailable")
        return result
    result["native_profile_cid"] = profile.get("profile_cid")
    result["native_profile_observation_cid"] = profile.get("observation_cid")
    result["native_goal_contracts_accepted"] = profile.get("goal_contracts_accepted")
    result["native_completion_authority"] = profile.get("completion_authority") is True
    result["native_status_completion_authority"] = (
        observation.get("completion_authority") is True
    )
    datasets = profile.get("datasets_accepted_root")
    if isinstance(datasets, Mapping):
        result["current_rollout_mode"] = datasets.get("current_rollout_mode")
        result["datasets_admission_mode"] = datasets.get("admission_mode")
        result["datasets_semantic_acceptance_authority"] = (
            datasets.get("semantic_acceptance_authority") is True
        )
        if datasets.get("semantic_acceptance_authority") is not True:
            result["blockers"].append("datasets_semantic_acceptance_authority_false")
        mode = datasets.get("current_rollout_mode")
        if mode and mode != "required":
            result["blockers"].append(f"current_rollout_mode_is_not_required:{mode}")
    probes = (profile.get("datasets_accepted_root") or {}).get("source_clause_probes")
    if not isinstance(probes, Mapping):
        extra = facts.get("completion_profile") or {}
        # Adapter-side probes live on datasets extra when producer MODE_FLOORS.
        probes = extra.get("source_clause_probes") if isinstance(extra, Mapping) else {}
    if isinstance(probes, Mapping):
        result["source_clause_probes"] = {
            name: {
                "accepted": row.get("accepted") is True,
                "reason": str(row.get("reason") or "")[:256],
            }
            for name, row in probes.items()
            if isinstance(row, Mapping)
        }
        for name, row in result["source_clause_probes"].items():
            if row["accepted"] is not True:
                result["blockers"].append(f"source_clause_not_accepted:{name}")
    if profile.get("goal_contracts_accepted") is not True:
        result["blockers"].append("native_goal_contracts_not_accepted")
    if observation.get("completion_authority") is not True:
        result["blockers"].append("native_completion_authority_false")
    result["blockers"].extend(
        blocker for blocker in profile.get("blockers") or [] if isinstance(blocker, str)
    )
    if facts.get("truncated") or not facts.get("all_relations_available"):
        result["blockers"].append("native_population_truncated_or_missing")
        return result
    relations = facts.get("relations")
    if not isinstance(relations, Mapping):
        result["blockers"].append("native_population_truncated_or_missing")
        return result
    tasks = ((relations.get("tasks") or {}).get("rows") if isinstance(relations.get("tasks"), Mapping) else None) or []
    goals = ((relations.get("goals") or {}).get("rows") if isinstance(relations.get("goals"), Mapping) else None) or []
    result["task_counts"] = dict(collections.Counter(task.get("status") for task in tasks))
    result["goal_counts"] = dict(collections.Counter(goal.get("status") for goal in goals))
    result["active_claims"] = sum(
        len((relations.get(name) or {}).get("rows") or [])
        for name in ("task_claims", "leases", "resource_claims", "path_claims", "effect_claims")
        if isinstance(relations.get(name), Mapping)
    )
    result["blocking_obligations"] = sum(
        len((relations.get(name) or {}).get("rows") or [])
        for name in ("task_blocks", "proof_obligations")
        if isinstance(relations.get(name), Mapping)
    )
    runtime = root / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
    pending = _merge_queue_paths(runtime)
    result["filesystem_merge_obligations"] = pending
    merge_rows = (
        (relations.get("merge_queue_entries") or {}).get("rows")
        if isinstance(relations.get("merge_queue_entries"), Mapping)
        else []
    )
    result["pending_merges"] = len(merge_rows) + len(pending)
    if len(tasks) != 51 or any(
        task.get("status") not in ("complete", "completed") for task in tasks
    ):
        result["blockers"].append("tasks_unfinished")
    if len(goals) != 32 or any(
        goal.get("status") not in ("complete", "completed") for goal in goals
    ):
        result["blockers"].append("required_goals_unsettled")
    for key in ("active_claims", "pending_merges", "blocking_obligations"):
        if result[key]:
            result["blockers"].append(key + "_remain")
    for rel in TERMINALS:
        if not (root / rel).is_file():
            result["blockers"].append("terminal_artifact_missing:" + rel)
    result["blockers"] = sorted(set(result["blockers"]))
    native_authority = (
        observation.get("completion_authority") is True
        and profile.get("completion_authority") is True
        and profile.get("goal_contracts_accepted") is True
        and not profile.get("blockers")
    )
    if native_authority and not result["blockers"]:
        result["authoritative"] = True
        result["complete"] = True
    return result


def main() -> int:
    root = Path.cwd()
    try:
        value = evaluate(root)
    except Exception as exc:  # noqa: BLE001 - gate output must remain JSON
        value = {
            "schema": SCHEMA,
            "board_id": BOARD_ID,
            "authoritative": False,
            "complete": False,
            "error_class": type(exc).__name__,
            "blockers": ["native_closeout_gate_error"],
        }
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
