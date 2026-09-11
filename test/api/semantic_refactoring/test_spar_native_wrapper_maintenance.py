"""Native wrappers preserve queue custody before legacy recovery effects."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from test.api.semantic_refactoring.test_launch_source_amendment_task_source import (
    _receipt_id,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as module,
)
from test.api.semantic_refactoring import (
    test_spar_owner_merge_handoff as native_fixtures,
)

base_fresh_args = native_fixtures.fresh_args
pair = native_fixtures.pair


@pytest.fixture
def fresh_args(base_fresh_args):
    repo, args, policy, amendment = base_fresh_args
    args["target_branch"] = "codex/semantic-preserving-autonomous-remodularization-v1"
    return repo, args, policy, amendment


@pytest.fixture
def native_wrapper(pair):
    repo = pair.repo
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        cwd=repo,
        check=True,
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
    ).strip()
    forest_body = {
        "source_head": head,
        "nested_repositories": [],
        "cross_repository_writes": False,
    }
    forest = {**forest_body, "source_forest_root": _receipt_id(forest_body)}
    receipt_body = {
        **pair.amendment.launch_source_forest_receipt,
        "source_head": head,
        "repository_tree": tree,
        "source_forest_root": forest["source_forest_root"],
        "source_forest": forest,
    }
    receipt_body.pop("receipt_id")
    receipt = {**receipt_body, "receipt_id": _receipt_id(receipt_body)}
    amendment = replace(
        pair.amendment,
        launch_source_head=head,
        launch_repository_tree_id=tree,
        launch_source_forest_root=forest["source_forest_root"],
        launch_source_forest_receipt_id=receipt["receipt_id"],
        launch_source_forest_receipt=receipt,
        amendment_id="",
    )
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(repo / "wrapper.sock"))
    listener.listen(1)
    state = repo / "wrapper-state"
    state.mkdir()
    try:
        config = module.PortalSupervisorConfig(
            todo_path=pair.task.config.database_path,
            state_path=state / "state.json",
            strategy_path=state / "strategy.json",
            events_path=state / "events.jsonl",
            state_dir=state,
            repo_root=repo,
            board_namespace=amendment.board_namespace,
            merge_target_branch=pair.args["target_branch"],
            merge_queue_dir=pair.args["queue_root"],
            worktree_root=repo / "worktrees",
            state_owner_bootstrap_fd=listener.fileno(),
            state_owner_bootstrap_store_id=pair.task.identity.store_id,
            database_owner_session_id=pair.session,
            database_program=DatabaseProgramConfig(
                store_id=pair.task.identity.store_id,
                store_generation=str(pair.task.identity.generation),
                schema_revision="datasets-authoritative-operational-v1",
                quack_endpoint=pair.task.identity.listen_uri,
                endpoint_secret_handle="env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
            ),
            require_launch_source_amendment=True,
            launch_source_amendment_json=amendment.to_json(),
            owner_merge_bootstrap_profile="native-owner-merge-pair@1",
        )
        # Constructor admission is real; isolate only the maintenance methods
        # from unrelated board-runtime materialization (covered by native E2E).
        wrapper = object.__new__(module.PortalImplementationSupervisor)
        wrapper.config = config
        wrapper.board_namespace = config.board_namespace
        yield wrapper
    finally:
        listener.close()


def _forbidden(*args, **kwargs):
    pytest.fail("native wrapper reached local recovery effects")


@pytest.mark.parametrize("pair", ["fake", "native"], indirect=True)
def test_retained_owner_defers_wrapper_before_scans_or_local_queue(
    pair,
    native_wrapper,
    monkeypatch,
):
    wrapper = native_wrapper
    identity = pair.queue.identity.to_dict()
    assert pair.queue.ready()["ready"] is True
    monkeypatch.setattr(wrapper, "_git_worktree_records", _forbidden)
    monkeypatch.setattr(wrapper, "_read_jsonl_events", _forbidden)
    monkeypatch.setattr(module, "PortalImplementationDaemon", _forbidden)
    for method in (
        wrapper.reconcile_backlogged_worktrees,
        wrapper.recover_already_merged_reconciliation_candidates,
    ):
        result = method()
        assert result["attempted"] is False
        assert result["deferred"] is True and result["custody_retained"] is True
        assert (
            result["completion_authority"] is False
            and result["task_authority"] is False
        )
    with pytest.raises(RuntimeError, match="forbids local reconciliation"):
        wrapper._build_worktree_reconciliation_daemon()
    assert pair.queue.ready()["ready"] is True
    assert pair.queue.identity.to_dict() == identity
    assert not wrapper.config.worktree_root.exists()


def test_native_predecessor_and_shutdown_paths_preserve_unresolved_custody(
    native_wrapper,
    monkeypatch,
):
    wrapper = native_wrapper
    metadata = {"task_id": "SPAR-unknown", "canonical_task_cid": "sha256:unknown"}
    lease = {"predecessor_attempt_lock": metadata}
    files = []
    for name, value in (
        ("retained-lease.json", lease),
        ("retained-receipt.json", metadata),
    ):
        path = wrapper.config.state_dir / name
        path.write_text(json.dumps(value))
        files.append((path, path.read_bytes(), path.stat().st_ino))
    monkeypatch.setattr(wrapper, "_build_worktree_reconciliation_daemon", _forbidden)
    monkeypatch.setattr(wrapper, "_implementation_maintenance_lock_path", _forbidden)
    assert wrapper._implementation_predecessor_claim_presence(metadata) == (
        None,
        "",
        "native_owner_merge_custody_retained",
    )
    closed = wrapper._reconcile_interrupted_implementation_after_shutdown(
        preacquired_implementation_lock=lease
    )
    assert closed["blocked"] is True and closed["reconciled"] is False
    archived = wrapper._handoff_completed_interrupted_recovery_receipt(lease, metadata)
    assert archived["blocked"] is True and archived["lease_released"] is False
    for path, raw, inode in files:
        assert path.read_bytes() == raw and path.stat().st_ino == inode


def test_ordinary_profile_keeps_original_factory_and_disabled_passes(
    tmp_path, monkeypatch
):
    config = module.PortalSupervisorConfig(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        state_dir=tmp_path,
        repo_root=tmp_path,
        worktree_reconciliation_enabled=False,
    )
    wrapper = object.__new__(module.PortalImplementationSupervisor)
    wrapper.config = config
    wrapper.board_namespace = config.board_namespace
    captured = []
    sentinel = object()

    def portal(**kwargs):
        captured.append(kwargs)
        return sentinel

    monkeypatch.setattr(module, "PortalImplementationDaemon", portal)
    assert wrapper._build_worktree_reconciliation_daemon() is sentinel
    assert captured[0]["merge_queue_dir"] == config.merge_queue_dir
    for method in (
        wrapper.reconcile_backlogged_worktrees,
        wrapper.recover_already_merged_reconciliation_candidates,
    ):
        assert method() == {
            "attempted": False,
            "reason": "worktree_reconciliation_disabled",
        }


@pytest.mark.parametrize("pair", ["native"], indirect=True)
def test_actual_native_factory_maintains_owned_queue_without_local_fallback(
    pair,
    native_wrapper,
    monkeypatch,
):
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
    from ipfs_accelerate_py.agent_supervisor.task_sources.owner_merge_bootstrap import (
        OwnerMergeBootstrapBundle,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        PortalImplementationDaemon,
        parse_args,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
        bind_database_portal_execution_from_args,
    )
    from test.api.test_agent_supervisor_owner_recovery_integration import BindingDaemon
    import os

    subprocess.run(
        ["git", "branch", pair.args["target_branch"]], cwd=pair.repo, check=True
    )
    scope = pair.args["scopes"][0]
    state = Path(scope["attempt_root"]).parent
    parsed = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--board-namespace",
            scope["board_namespace"],
            "--database-path",
            str(pair.task.config.database_path),
            "--todo-path",
            str(state / "board.md"),
            "--state-dir",
            str(state),
            "--state-prefix",
            "spar_lane_0",
            "--task-shard-index",
            "0",
            "--merge-queue-dir",
            str(pair.args["queue_root"]),
            "--merge-target-branch",
            pair.args["target_branch"],
            "--implement",
            "--once",
        ]
    )
    response = native_fixtures.issue(pair)
    bundle = OwnerMergeBootstrapBundle.from_response(
        response, request=pair.request, peer_pid=os.getpid(), peer_uid=os.geteuid()
    )
    attached = bundle.attach_merge_runtime(
        repository_root=pair.repo,
        attempt_root=Path(scope["attempt_root"]),
        board_namespace=scope["board_namespace"],
        lane_id="0",
        admitted_config_cid=scope["config_cid"],
        admitted_plan_cid=scope["plan_cid"],
    )
    before = attached.runtime.client.load_cursors()
    monkeypatch.setattr(MergeQueue, "__init__", _forbidden)
    try:
        daemon = BindingDaemon()
        bridge = bind_database_portal_execution_from_args(
            daemon,
            parsed,
            repo_root=pair.repo,
            portal_daemon_class=PortalImplementationDaemon,
            owner_merge_runtime=attached.runtime,
            admitted_owner_merge_config_cid=scope["config_cid"],
            admitted_owner_merge_plan_cid=scope["plan_cid"],
        )
        assert bridge.merge_queue is attached.runtime.queue
        assert daemon.callbacks["post_merge_recovery"]() is None
        result = daemon.callbacks["merge_train_recovery"]["pending_merge_consume_fn"]()
        assert result is None
        assert attached.runtime.client.load_cursors() == before
        assert pair.queue.ready()["ready"] is True
    finally:
        attached.close()
