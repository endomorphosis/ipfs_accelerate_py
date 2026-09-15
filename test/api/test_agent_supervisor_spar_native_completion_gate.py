"""SPAR publication gate follows native closeout facts, not a placeholder hold."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.rescue import spar_native_completion_gate as gate


def _row(status: str) -> dict[str, str]:
    return {"status": status}


def _observation(*, native_authority: bool, pending_merges: int = 0) -> dict:
    tasks = [_row("completed") for _ in range(51)]
    goals = [_row("completed") for _ in range(32)]
    mode = "required" if native_authority else "bootstrap"
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-board-status@1",
        "authoritative_task_observation": True,
        "completion_authority": native_authority,
        "complete": native_authority,
        "owner_identity": {"generation": 1},
        "closeout_snapshot": {
            "snapshot_cid": "baguqeera-snapshot",
            "closeout_facts": {
                "truncated": False,
                "all_relations_available": True,
                "completion_profile": {
                    "schema": "ipfs_accelerate_py/agent-supervisor/spar-closeout-requirements@1",
                    "profile_cid": "baguqeera-profile",
                    "observation_cid": "baguqeera-obs",
                    "goal_contracts_accepted": True,
                    "completion_authority": native_authority,
                    "blockers": [],
                    "datasets_accepted_root": {
                        "admitted": True,
                        "admission_mode": mode,
                        "current_rollout_mode": mode,
                        "semantic_acceptance_authority": native_authority,
                    },
                },
                "relations": {
                    "tasks": {"rows": tasks},
                    "goals": {"rows": goals},
                    "task_claims": {"rows": []},
                    "leases": {"rows": []},
                    "resource_claims": {"rows": []},
                    "path_claims": {"rows": []},
                    "effect_claims": {"rows": []},
                    "task_blocks": {"rows": []},
                    "proof_obligations": {"rows": []},
                    "merge_queue_entries": {"rows": [{}] * pending_merges},
                },
            },
        },
    }


def test_bootstrap_native_authority_cannot_publish(tmp_path, monkeypatch):
    _layout(tmp_path)
    monkeypatch.setattr(gate, "_git_head", lambda _repo: ("a" * 40, False))
    monkeypatch.setattr(gate, "_merge_queue_paths", lambda _runtime: [])
    result = gate.evaluate(tmp_path, observation=_observation(native_authority=False))
    assert result["authoritative"] is False
    assert result["complete"] is False
    assert result["task_counts"] == {"completed": 51}
    assert result["goal_counts"] == {"completed": 32}
    assert result["active_claims"] == 0
    assert result["pending_merges"] == 0
    assert "native_completion_authority_false" in result["blockers"]
    assert "current_rollout_mode_is_not_required:bootstrap" in result["blockers"]
    assert "datasets_semantic_acceptance_authority_false" in result["blockers"]
    assert "sealed_spar_goal_and_current_source_root_acceptance_adapter_required" not in result["blockers"]


def test_native_authority_with_settled_population_can_publish(tmp_path, monkeypatch):
    _layout(tmp_path)
    monkeypatch.setattr(gate, "_git_head", lambda _repo: ("a" * 40, False))
    monkeypatch.setattr(gate, "_merge_queue_paths", lambda _runtime: [])
    result = gate.evaluate(tmp_path, observation=_observation(native_authority=True))
    assert result["blockers"] == []
    assert result["authoritative"] is True
    assert result["complete"] is True
    assert result["board_id"] == "spar"
    assert result["source_heads"]["accelerator"] == "a" * 40


def test_native_authority_still_holds_when_merges_remain(tmp_path, monkeypatch):
    _layout(tmp_path)
    monkeypatch.setattr(gate, "_git_head", lambda _repo: ("a" * 40, False))
    monkeypatch.setattr(gate, "_merge_queue_paths", lambda _runtime: [])
    result = gate.evaluate(
        tmp_path, observation=_observation(native_authority=True, pending_merges=1)
    )
    assert result["authoritative"] is False
    assert result["complete"] is False
    assert "pending_merges_remain" in result["blockers"]


def _layout(root: Path) -> None:
    for rel in (
        "ipfs_datasets_py",
        "ipfs_kit_py",
        "ipfs_accelerate_py/mcplusplus",
        "docs/architecture/semantic_preserving_autonomous_remodularization_inventory",
        "test/api/semantic_refactoring",
    ):
        (root / rel).mkdir(parents=True)
    (root / "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json").write_text("{}")
    (root / "docs/architecture/SEMANTIC_PRESERVING_AUTONOMOUS_REMODULARIZATION_FINAL_REPORT.md").write_text("#")
    (root / "test/api/semantic_refactoring/test_release_gate.py").write_text("def test_pass():\n    assert True\n")
