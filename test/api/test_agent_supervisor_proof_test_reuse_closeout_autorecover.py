"""Tests for agent-supervisor closeout auto-recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_autorecover import (
    AUTO_REPAIR_KINDS,
    inventory_closeout_inputs,
    refresh_validation_receipt_freshness,
    strip_contradictory_approval_merge_rows,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_cached_test_validation import (
    validation_command_identity,
)

COMMAND = "IPFS_TEST_PROOF_REUSE_MODE=off python3 -m pytest -q"
COMMIT = "a" * 40
TREE = "b" * 40


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    payload.pop("validation_receipt_cid", None)
    return {**payload, "validation_receipt_cid": content_identity(payload)}


def test_auto_repair_kinds_are_explicit_and_non_authoritative() -> None:
    assert "validation_receipt_freshness_refresh" in AUTO_REPAIR_KINDS
    assert "managed_merge_git_recovery" in AUTO_REPAIR_KINDS
    assert "goal_coverage_projection" in AUTO_REPAIR_KINDS
    # Must never claim production authority via auto repair.
    assert "production_skip_grant" not in AUTO_REPAIR_KINDS


def test_refresh_validation_receipt_freshness_reseals_identity_bound(
    tmp_path: Path,
) -> None:
    receipt_dir = tmp_path / "validation_receipts"
    receipt_dir.mkdir()
    body = _seal(
        {
            "task_id": "PTR-012",
            "goal_id": "PTR-G010",
            "task_cid": "baguqeera-task-012",
            "validation_command": COMMAND,
            "validation_command_cid": validation_command_identity(COMMAND),
            "git_commit_id": COMMIT,
            "git_tree_id": TREE,
            "repository_id": "repo",
            "repository_state_cid": f"git-commit:{COMMIT}",
            "gitlink_state_cid": "baguqeera-gitlinks",
            "repository_forest_cid": "baguqeera-forest",
            "dirty": False,
            "dirty_overlay_cid": "cid:dirty-overlay:none",
            "proof_reuse_mode": "off",
            "passed": True,
            "status": "passed",
            "exit_code": 0,
            "skipped_count": 0,
            "disposition": "executed",
            "observed_at_ms": 1_000,
            "fresh_until_ms": 2_000,
            "retained_at_ms": 1_000,
            "duration_ms": 10,
            "authority": False,
            "schema": "ipfs_accelerate_py/proof-backed-test-reuse-executed-validation-receipt@1",
        }
    )
    path = receipt_dir / "PTR-012.json"
    path.write_text(json.dumps(body) + "\n", encoding="utf-8")
    old_cid = body["validation_receipt_cid"]

    result = refresh_validation_receipt_freshness(
        receipt_dir,
        expected_commit=COMMIT,
        expected_tree=TREE,
        freshness_seconds=3_600.0,
        now_ms=1_800_000_000_000,
        persist=True,
    )
    assert result["refreshed_count"] == 1
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["observed_at_ms"] == 1_800_000_000_000 - 1_000
    assert reloaded["fresh_until_ms"] > reloaded["observed_at_ms"]
    assert reloaded["validation_receipt_cid"]
    assert reloaded["validation_receipt_cid"] != old_cid
    # Reseal is self-consistent.
    claim = reloaded.pop("validation_receipt_cid")
    assert content_identity(reloaded) == claim


def test_strip_contradictory_approval_merge_rows(tmp_path: Path) -> None:
    merge_dir = tmp_path / "completed"
    merge_dir.mkdir()
    recovered = merge_dir / "recovered-PTR-000.json"
    recovered.write_text(
        json.dumps(
            {
                "task_id": "PTR-000",
                "status": "completed",
                "commit_sha": COMMIT,
                "recovery_source": "git_ancestry",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    genuine = merge_dir / "daemon-PTR-012.json"
    genuine.write_text(
        json.dumps(
            {
                "task_id": "PTR-012",
                "status": "completed",
                "commit_sha": COMMIT,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = strip_contradictory_approval_merge_rows(
        merge_dir, approval_only_task_ids=("PTR-000",)
    )
    assert result["removed_count"] == 1
    assert not recovered.exists()
    assert genuine.exists()


def test_inventory_counts_approvals_and_structured_coverage(tmp_path: Path) -> None:
    state = tmp_path / "state"
    completion = state / "projection" / "completion"
    approvals = completion / "operator_approvals"
    merge = state / "merge-queue" / "completed"
    approvals.mkdir(parents=True)
    merge.mkdir(parents=True)
    completion.mkdir(parents=True, exist_ok=True)

    (approvals / "accepted.json").write_text(
        json.dumps(
            {
                "approvals": {
                    "PTR-000": {
                        "task_id": "PTR-000",
                        "approved": True,
                        "approval_cid": "baguqeera-approval-000",
                        "reviewer_id": "op@example.com",
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (approvals / "PTR-000.attestation.json").write_text(
        json.dumps(
            {
                "task_id": "PTR-000",
                "accepted": True,
                "approval_cid": "baguqeera-approval-000",
                "operator_id": "op@example.com",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (merge / "PTR-012.json").write_text(
        json.dumps(
            {
                "task_id": "PTR-012",
                "status": "completed",
                "merged_commit_id": COMMIT,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (completion / "goal_coverage.json").write_text(
        json.dumps(
            {
                "goals": {
                    "PTR-G010": {
                        "criteria": [
                            {
                                "criterion": "ptr/example@1",
                                "status": "blocked",
                            }
                        ],
                        "acceptance_population": ["ptr/example@1"],
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    inventory = inventory_closeout_inputs(
        state_root=state,
        task_ids=("PTR-000", "PTR-012"),
        goal_ids=("PTR-G010",),
        requirement_ids=("ptr/example@1",),
    )
    by_name = {row["name"]: row for row in inventory["requirements"]}
    assert by_name["genuine_reviewed_approvals_without_queue_records"][
        "present_count"
    ] == 1
    assert by_name["managed_merge_or_reviewed_completion_provenance"][
        "present_count"
    ] == 2
    assert by_name["acceptance_coverage_receipts"]["present_count"] == 1
    assert inventory["inventory_is_completion_authority"] is False
