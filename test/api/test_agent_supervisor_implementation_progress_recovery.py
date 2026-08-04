"""Automatic progress recovery for stalled landed-product tasks."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_progress_recovery import (
    declared_output_presence,
    operator_landed_binding_payload,
    should_recover_stalled_task,
)


def test_declared_output_presence_reports_complete_and_missing(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    presence = declared_output_presence(
        task_id="T-1",
        outputs=["a.py", "missing.py", "dir/nested.py"],
        repo_root=tmp_path,
    )
    assert presence.present == ("a.py",)
    assert presence.missing == ("missing.py", "dir/nested.py")
    assert presence.complete is False

    (tmp_path / "missing.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "dir" / "nested.py").write_text("z = 3\n", encoding="utf-8")
    complete = declared_output_presence(
        task_id="T-1",
        outputs=["a.py", "missing.py", "dir/nested.py"],
        repo_root=tmp_path,
    )
    assert complete.complete is True
    assert complete.missing == ()


def test_should_recover_when_outputs_landed_and_repair_budget_exhausted(
    tmp_path: Path,
) -> None:
    (tmp_path / "product.py").write_text("ok\n", encoding="utf-8")
    decision = should_recover_stalled_task(
        task_id="ASE2-004",
        outputs=["product.py"],
        repo_root=tmp_path,
        attempt_count=5,
        max_repair_rounds=3,
        last_returncode=1,
        selection_idle_reason=(
            "implementation_retry_deferred:implementation_repair_round_budget_exhausted"
        ),
    )
    assert decision is not None
    assert decision.reset_attempt_budget is True
    assert decision.clear_diagnostics is True
    assert decision.treat_as_landed_outputs is True
    assert decision.reclaim_dead_lifecycle is True


def test_should_recover_skips_active_task(tmp_path: Path) -> None:
    (tmp_path / "product.py").write_text("ok\n", encoding="utf-8")
    decision = should_recover_stalled_task(
        task_id="ASE2-004",
        outputs=["product.py"],
        repo_root=tmp_path,
        attempt_count=5,
        max_repair_rounds=3,
        last_returncode=1,
        implementation_in_progress=True,
        active_task_id="ASE2-004",
    )
    assert decision is None


def test_should_not_recover_when_outputs_incomplete(tmp_path: Path) -> None:
    decision = should_recover_stalled_task(
        task_id="ASE2-002",
        outputs=["missing.py"],
        repo_root=tmp_path,
        attempt_count=5,
        max_repair_rounds=3,
        last_returncode=1,
        selection_idle_reason="implementation_repair_round_budget_exhausted",
    )
    assert decision is None


def test_operator_landed_binding_is_non_authoritative() -> None:
    payload = operator_landed_binding_payload(
        task_id="ASE2-003",
        canonical_task_cid="baguqeeraexample",
        merge_commit="abc123",
        repository_tree_id="git-tree:def456",
        present_outputs=["local_profile.py", "test_local_profile.py"],
    )
    assert payload["recovered"] is True
    assert payload["implementation_commit"] == "abc123"
    assert payload["merge_commit"] == "abc123"
    assert payload["completion_authoritative"] is False
    assert payload["proof_authoritative"] is False
    assert payload["source"] == "merge_target_declared_outputs"
    assert "local_profile.py" in payload["present_outputs"]


def test_context_insufficient_marker_triggers_recovery(tmp_path: Path) -> None:
    (tmp_path / "out.py").write_text("ok\n", encoding="utf-8")
    decision = should_recover_stalled_task(
        task_id="T",
        outputs=["out.py"],
        repo_root=tmp_path,
        attempt_count=2,
        max_repair_rounds=3,
        last_returncode=1,
        last_failure_text="production source context is unavailable or insufficient",
    )
    assert decision is not None
    assert decision.reset_attempt_budget is True
