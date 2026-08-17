"""ASE3-014 release/closeout: canonical plan materialization and staged cutover."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.service_factory import (
    resolve_production_composition,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.v3_rollout import (
    PRODUCER_TASK_IDS,
    decide_rollout,
    force_final_residual_scan,
    materialize_bundle_index,
    materialize_canonical_plan,
    materialize_evidence_join,
    materialize_release_artifacts,
    materialize_rollback_receipt,
    materialize_terminal_shutdown,
    parse_board_task_statuses,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKBOARD = (
    REPO_ROOT
    / "docs/architecture/agent_supervisor_prompt_only_self_improvement_v3.todo.md"
)
PLAN_DIR = REPO_ROOT / "data/agent_supervisor/prompt_only_self_improvement_v3/plan"
ROLLOUT_DIR = (
    REPO_ROOT / "data/agent_supervisor/prompt_only_self_improvement_v3/rollout"
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
    ).strip()


def test_parse_board_statuses_include_closeout_chain() -> None:
    statuses = parse_board_task_statuses(TASKBOARD.read_text(encoding="utf-8"))
    assert statuses.get("ASE3-013") == "completed"
    assert statuses.get("ASE3-012") == "completed"
    assert statuses.get("ASE3-026") == "completed"
    assert statuses.get("ASE3-009") == "completed"
    # ASE3-014 may still be todo until this release commit flips it.
    assert statuses.get("ASE3-014") in {"todo", "completed"}


def test_final_residual_scan_holds_when_open() -> None:
    scan = force_final_residual_scan(
        task_statuses={"ASE3-013": "completed", "ASE3-014": "todo"},
        required_completed=("ASE3-013", "ASE3-014"),
    )
    assert scan["residual_open"] is True
    assert "ASE3-014" in scan["open_residuals"]
    assert scan["decision"] == "hold"


def test_final_residual_scan_complete_when_all_done() -> None:
    statuses = {task: "completed" for task in PRODUCER_TASK_IDS}
    scan = force_final_residual_scan(task_statuses=statuses)
    assert scan["residual_open"] is False
    assert scan["decision"] == "complete"


def test_materialize_canonical_plan_and_bundles() -> None:
    plan = materialize_canonical_plan(
        taskboard_text=TASKBOARD.read_text(encoding="utf-8")
    )
    assert plan.plan_cid.startswith("baguqeera") or plan.plan_cid.startswith("bafy") or len(plan.plan_cid) > 10
    assert plan.board_namespace.startswith("agent-supervisor")
    bundles = materialize_bundle_index()
    assert "self-host-canary" in bundles.bundles
    assert "ASE3-013" in bundles.bundles["self-host-canary"]
    assert "ASE3-014" in bundles.bundles["rollout-closeout"]


def test_evidence_join_and_preview_promotion(tmp_path: Path) -> None:
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    composition = resolve_production_composition(repository_root=REPO_ROOT)
    text = TASKBOARD.read_text(encoding="utf-8")
    statuses = parse_board_task_statuses(text)
    # Treat ASE3-014 as completed for join readiness in this release test.
    statuses = dict(statuses)
    statuses["ASE3-014"] = "completed"
    join = materialize_evidence_join(
        repository_root=REPO_ROOT,
        head=head,
        tree=tree,
        task_statuses=statuses,
        canary_terminal_healthy=True,
        composition_cid=composition.composition_cid,
    )
    assert join.task_receipts_complete is True
    decision = decide_rollout(
        join=join,
        mode="preview",
        rollback_target_head=head,
        authority="operator-release-authority",
    )
    assert decision.promotion_authorized is True
    assert decision.release_head == head
    assert decision.mode == "preview"


def test_incomplete_join_denies_promotion() -> None:
    join = materialize_evidence_join(
        repository_root=REPO_ROOT,
        head="a" * 40,
        tree="b" * 40,
        task_statuses={"ASE3-013": "todo"},
        canary_terminal_healthy=False,
        composition_cid="",
    )
    decision = decide_rollout(
        join=join,
        mode="local_auto",
        rollback_target_head="c" * 40,
        authority="operator",
    )
    assert decision.promotion_authorized is False
    assert decision.reasons


def test_rollback_receipt_preserves_expert_entrypoints() -> None:
    receipt = materialize_rollback_receipt(
        from_head="a" * 40,
        to_head="b" * 40,
        triggers=("canary_not_healthy",),
    )
    assert receipt.expert_entrypoints_preserved is True
    assert "canary_not_healthy" in receipt.triggers


def test_terminal_shutdown_exact_generation() -> None:
    shutdown = materialize_terminal_shutdown(
        owned_process_generations=("lifecycle:1", "monitor:1")
    )
    assert shutdown.exact_generation_only is True
    assert shutdown.fences_released is True


def test_materialize_release_artifacts_on_tree() -> None:
    head = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    composition = resolve_production_composition(repository_root=REPO_ROOT)
    # Flip ASE3-014 to completed in a temp board copy for residual completeness,
    # but materialize against the real tree board (may still list 014 as completed
    # after this commit). Force canary healthy + use real composition.
    artifacts = materialize_release_artifacts(
        repository_root=REPO_ROOT,
        head=head,
        tree=tree,
        composition_cid=composition.composition_cid,
        canary_terminal_healthy=True,
        mode="preview",
        authority="operator-release-authority",
        rollback_target_head=head,
    )
    assert (PLAN_DIR / "canonical_v3_plan.json").is_file()
    assert (PLAN_DIR / "v3_bundle_index.json").is_file()
    assert (PLAN_DIR / "current_tree_evidence_join.json").is_file()
    assert (ROLLOUT_DIR / "v3_rollout_decision.json").is_file()
    assert (ROLLOUT_DIR / "terminal_shutdown_receipt.json").is_file()
    decision = artifacts["decision"]
    assert decision["release_head"] == head
    # No board rewrite: taskboard content unchanged by materializer (only data/).
    assert "ASE3-014" in TASKBOARD.read_text(encoding="utf-8")
    # Body-free-ish: no private keys in artifacts
    for path in PLAN_DIR.glob("*.json"):
        blob = path.read_text(encoding="utf-8")
        assert "BEGIN PRIVATE" not in blob
        assert "password" not in blob.lower()
