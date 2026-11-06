"""Synthetic Native histories with real Git merges and the actual scan loop.

These are isolated test campaign records, not provider/completion evidence.
No task state, paid ordinal, retained candidate or ref is repaired by the skip.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon_native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as native


def _world(tmp_path, monkeypatch, damage=None, *, linked_target=False):
    root, worktrees, state = (tmp_path / name for name in ("repo", "worktrees", "state"))
    for path in (root, worktrees, state):
        path.mkdir(mode=0o700)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(state / "control-plane"))

    def git(*args, cwd=None):
        result = subprocess.run(["git", *args], cwd=cwd or root, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "Synthetic Native preflight fixture")
    git("config", "user.email", "synthetic@example.invalid")
    board = root / "board.md"
    board.write_text("# Synthetic fixture\n\n## TEST-30 Retained failed attempt\n\n"
        "- Status: ready\n- Completion: manual\n- Priority: P1\n- Track: ops\n"
        "- Outputs: out/task.py\n- Acceptance: Synthetic local check.\n")
    (root / "out").mkdir()
    (root / "out/task.py").write_text("value = 0\n")
    git("add", ".")
    git("commit", "-qm", "synthetic baseline")
    baseline = git("rev-parse", "HEAD")
    if linked_target:
        git("checkout", "-q", "--detach", baseline)
        linked = tmp_path / "linked-target"
        git("worktree", "add", "-q", str(linked), "main")
        root, board = linked, linked / "board.md"
    config = native.PortalSupervisorConfig(todo_path=board, state_path=state / "state.json",
        strategy_path=state / "strategy.json", events_path=state / "supervisor_events.jsonl",
        state_dir=state, repo_root=root, state_prefix="synthetic", task_prefix="TEST-",
        worktree_root=worktrees, merge_target_branch="main", worktree_reconciliation_enabled=True,
        worktree_reconciliation_preflight_enabled=True, worktree_scan_cache_enabled=False,
        worktree_reconciliation_max_merges=1)
    supervisor = native.PortalImplementationSupervisor(config)
    daemon = supervisor._build_worktree_reconciliation_daemon()
    task = daemon._load_tasks()[0]
    identity = daemon._identity_for_task(task)
    binding = {"canonical_task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key, "board_namespace": identity.board_namespace}
    original, leaf = "implementation/test-30-old-attempt-1", worktrees / "workspace_retained"
    git("worktree", "add", "-q", "-b", original, str(leaf), baseline)
    (leaf / "out/task.py").write_text("value = 1\n")
    rescue = supervisor._rescue_dirty_worktree(leaf, branch=original, head=baseline,
        target_ref="main", status_lines=supervisor._git_status_short(leaf),
        reason="stale_active_execution_auto_rescue")
    assert rescue["preserved"] is True
    branch, candidate = rescue["rescue_branch"], git("rev-parse", "HEAD", cwd=leaf)
    git("checkout", "-qb", "implementation/test-30-winning-attempt-2", baseline)
    (root / "out/task.py").write_text("value = 2\n")
    git("add", "out/task.py")
    git("commit", "-qm", "synthetic winning candidate")
    winner = git("rev-parse", "HEAD")
    git("checkout", "-q", "main")
    integration_base = git("rev-parse", "HEAD")
    git("merge", "--no-ff", "-qm", "synthetic actual two-parent integration", winner)
    integrated = git("rev-parse", "HEAD")
    board.write_text(board.read_text().replace("Status: ready", "Status: completed"))
    if damage == "unfinished":
        board.write_text(board.read_text().replace("Status: completed", "Status: ready"))
    if damage == "changed_definition":
        board.write_text(board.read_text().replace("out/task.py", "out/changed.py"))
    git("add", "board.md")
    if git("diff", "--cached", "--name-only"):
        git("commit", "-qm", "synthetic current board fixture")
    if damage == "nonintegrated_winner":
        current_board = board.read_text()
        git("checkout", "-q", "--detach", baseline)
        (root / "out/task.py").write_text("value = 2\n")
        board.write_text(current_board)
        git("add", ".")
        git("commit", "-qm", "synthetic unrelated target with same final bytes")
        git("branch", "-f", "main", git("rev-parse", "HEAD"))
        git("checkout", "-q", "main")
    start = {"task_id": task.task_id, "branch": original, "baseline_ref": baseline,
        "worktree_path": str(leaf), "attempt": 1, **binding}
    finish = {"task_id": task.task_id, "branch": "implementation/test-30-winning-attempt-2",
        "attempt": 2, **binding, "returncode": 0, "implementation_commit": winner,
        "validation_result": {"passed": True, "returncode": 0},
        "merge_result": {"attempted": True, "merged": True, "returncode": 0,
            "target_branch": "main", "implementation_commit": winner, "merge_commit": integrated,
            "integration_base_commit": integration_base, "canonical_task_cid": identity.canonical_task_cid,
            "canonical_task_key": identity.canonical_task_key,
            "completion_task_cids": {task.task_id: identity.canonical_task_cid},
            "integration_commit_proof": {"passed": True, "exact_two_parent_merge": True,
                "implementation_commit": winner, "integration_base_commit": integration_base,
                "integration_commit": integrated, "integration_ref": integrated, "target_branch": "main"}}}
    if damage == "wrong_cid":
        start["canonical_task_cid"] = "b" + "a" * 58
    if damage == "wrong_key":
        finish["canonical_task_key"] = "task/v1/" + "f" * 64
    if damage == "wrong_namespace":
        start["board_namespace"] = "foreign-namespace"
    if damage == "wrong_workspace":
        start["worktree_path"] = str(worktrees / "foreign-workspace")
    if damage == "not_later_attempt":
        finish["attempt"] = 1
    if damage == "validation_failed":
        finish["validation_result"]["passed"] = False
    if damage == "wrong_merge_cid":
        finish["merge_result"]["completion_task_cids"] = {task.task_id: "b" + "a" * 58}
    if damage == "wrong_merge_parents":
        finish["merge_result"]["integration_base_commit"] = winner
        finish["merge_result"]["integration_commit_proof"]["integration_base_commit"] = winner
    if damage == "wrong_target":
        finish["merge_result"]["target_branch"] = "foreign-target"
    events = state / "synthetic_events.jsonl"
    if damage != "no_start":
        native.append_jsonl_event(events, "implementation_started", start)
    if damage == "duplicate_start":
        native.append_jsonl_event(events, "implementation_started", start)
    if damage != "no_winner":
        native.append_jsonl_event(events, "implementation_finished", finish)
    if damage == "tampered_rescue":
        rows = config.events_path.read_text().splitlines()
        record = json.loads(rows[-1]); record["rescue_commit"] = "f" * 40
        rows[-1] = json.dumps(record); config.events_path.write_text("\n".join(rows) + "\n")
    if damage == "wrong_rescue_attestation":
        record = json.loads(config.events_path.read_text().splitlines()[-1])
        protocol = {"event_id", "sequence", "previous_event_id", "stream_id", "snapshot_id", "timestamp", "type"}
        payload = {key: value for key, value in record.items() if key not in protocol}
        payload["rescue_commit"] = "f" * 40
        config.events_path = state / "different_supervisor_events.jsonl"
        native.append_jsonl_event(config.events_path, "dirty_worktree_rescued", payload)
    if damage == "tampered_native":
        rows = events.read_text().splitlines()
        record = json.loads(rows[-1]); record["attempt"] = 99
        rows[-1] = json.dumps(record); events.write_text("\n".join(rows) + "\n")
    if damage == "partial_native":
        events.write_bytes(events.read_bytes()[:-1])
    if damage == "foreign_rescue_ref":
        git("branch", "-m", branch, "rescue/worktree/foreign-copy", cwd=leaf)
        branch = "rescue/worktree/foreign-copy"
    if damage == "dirty_candidate":
        (leaf / "out/task.py").write_text("value = 3\n")
    # A read-only negative classification must not refund/consume ordinals.
    saved = daemon_native.PortalTaskState()
    saved.implementation_attempts = {task.task_id: 2}
    saved.implementation_attempts_by_cid = {identity.canonical_task_cid: 2}
    saved.save(config.state_path)
    return SimpleNamespace(root=root, leaf=leaf, branch=branch, candidate=candidate,
        board=board, state=state, supervisor=supervisor, daemon=daemon, git=git, task=task,
        binding=binding, integrated=integrated, events=events)


def _custody(world):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import read_task_state_payload
    return {"state": read_task_state_payload(world.supervisor.config.state_path),
        "board": world.board.read_bytes(), "refs": world.git("show-ref"),
        "candidate": world.git("rev-parse", world.branch),
        "candidate_bytes": (world.leaf / "out/task.py").read_bytes(),
        "task_queue": dict(world.daemon.task_queue.__dict__)}


@pytest.mark.parametrize("linked_target", [False, True])
def test_actual_native_loop_skips_proven_obsolete_conflict_before_preflight(tmp_path, monkeypatch, linked_target):
    world = _world(tmp_path, monkeypatch, linked_target=linked_target)
    actual = world.supervisor._preflight_worktree_reconciliation_merge(world.root,
        target_ref="main", branch=world.branch)
    assert actual["mergeable"] is False and actual["reason"] == "preflight_merge_conflict"
    before = _custody(world)
    def no_preflight(*args, **kwargs):
        pytest.fail("proven_completed_obsolete_candidate_reached_merge_preflight")
    monkeypatch.setattr(world.supervisor, "_preflight_worktree_reconciliation_merge", no_preflight)
    def no_registration(*args, **kwargs):
        pytest.fail("completed_candidate_read_registered_or_saved_task_identity")
    monkeypatch.setattr(daemon_native.PortalImplementationDaemon, "_register_task_identities", no_registration)
    result = world.supervisor.reconcile_backlogged_worktrees()
    assert result["candidate_count"] == 1 and result["processed_count"] == 0
    assert result["preflight_blocked_count"] == 0 and result["reconciled_count"] == 0
    row = result["skipped"][0]
    assert row["reason"] == "reconciliation_candidate_completed_by_integrated_later_attempt"
    assert row["terminal_completion_proof"]["canonical_task_cid"] == world.binding["canonical_task_cid"]
    assert row["terminal_completion_proof"]["completion_authority"] is False
    assert _custody(world) == before
    assert world.supervisor.reconcile_backlogged_worktrees()["scan_cache_hit_count"] == 0


@pytest.mark.parametrize("damage", ["wrong_cid", "wrong_key", "wrong_namespace", "wrong_workspace",
    "unfinished", "changed_definition", "no_start", "duplicate_start", "no_winner", "not_later_attempt",
    "validation_failed", "wrong_merge_cid", "wrong_merge_parents", "wrong_target", "nonintegrated_winner", "tampered_rescue",
    "tampered_native", "partial_native", "foreign_rescue_ref", "wrong_rescue_attestation"])
def test_actual_native_loop_keeps_conflict_guard_without_exact_terminal_proof(tmp_path, monkeypatch, damage):
    world = _world(tmp_path, monkeypatch, damage)
    before = _custody(world)
    if damage == "tampered_rescue":
        from ipfs_accelerate_py.agent_supervisor.control.control_contracts import CursorReplayError
        with pytest.raises(CursorReplayError, match="non-canonical event_id"):
            world.supervisor.reconcile_backlogged_worktrees()
        assert _custody(world) == before
        return
    result = world.supervisor.reconcile_backlogged_worktrees()
    assert result["candidate_count"] == 1 and result["processed_count"] == 1
    assert result["preflight_blocked_count"] == 1 and result["reconciled_count"] == 0
    assert result["processed"][0]["merge_result"]["reason"] == "preflight_merge_conflict"
    assert not any("terminal_completion_proof" in row for row in result["skipped"])
    assert _custody(world) == before


def test_completed_skip_uses_target_ancestry_not_unrelated_current_head(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    target = world.git("rev-parse", "main")
    world.git("checkout", "-q", "-b", "unrelated-checkout", world.git("rev-list", "--max-parents=0", "HEAD"))
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.startup_terminal_candidate import completed_retained_candidate
    proof = completed_retained_candidate(world.supervisor, {"path": str(world.leaf),
        "branch": world.branch, "head": world.candidate, "target_ref": "main"})
    # Current board is no longer completed on this checkout, so no skip even
    # though the configured target retains the genuine integrated winner.
    assert proof is None
    assert world.git("rev-parse", "main") == target


def test_foreign_repository_replacing_registered_worktree_cannot_receive_completed_skip(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.startup_terminal_candidate import completed_retained_candidate
    retained = world.leaf.with_name("retained-original")
    world.leaf.rename(retained)
    # Keep the original worktree registration intact. A real independent clone
    # has the same objects, branch, HEAD and files, but different Git custody.
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(world.root), str(world.leaf)], check=True)
    world.git("checkout", "-q", "-b", world.branch, world.candidate, cwd=world.leaf)
    assert world.git("rev-parse", "HEAD", cwd=world.leaf) == world.candidate
    before = _custody(world)
    proof = completed_retained_candidate(world.supervisor, {"path": str(world.leaf),
        "branch": world.branch, "head": world.candidate, "target_ref": "main"})
    assert proof is None
    assert _custody(world) == before and retained.is_dir()
