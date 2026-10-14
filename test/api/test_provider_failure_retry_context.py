"""Cold retry keeps unresolved intent separate from an unaccepted source checkpoint."""
from __future__ import annotations

import json
import hashlib

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, PortalTaskState, content_identity,
)
from test.api.test_provider_failure_checkpoint_retention import (
    failed_child_case, git, run_failed_child,
)


@pytest.mark.parametrize("indexed", [False, True], ids=["no-index", "semantic-index"])
def test_cold_retry_retains_intent_and_failure_without_promoting_candidate(tmp_path, indexed):
    daemon, task, state, baseline = failed_child_case(tmp_path)
    if indexed:
        artifact = prepare_semantic_context(
            repository=daemon.repo_root, paths=["src/value.py"],
            required_raw_paths=["src/value.py"], objective=task.acceptance,
            task_id=task.task_id, output=daemon.repo_root / ".runtime/semantic/initial",
        )
        task.metadata.update({
            "Semantic context artifact": ".runtime/semantic/initial/worker-context.json",
            "Semantic context sha256": artifact["worker_payload_sha256"],
            "Semantic context refresh": "true",
        })
    checkpoint_dir = daemon._ensure_implementation_checkpoint_dir(task)
    # Worker checkpoint prose is retained as evidence, never as completion.
    (checkpoint_dir / "progress.json").write_text(json.dumps({"status": "complete"}))
    manifest = daemon._implementation_checkpoint_manifest(task)
    result = run_failed_child(daemon, task, state)
    assert result["returncode"] == 17
    assert result["validation_result"]["attempted"] is False
    assert result["validation_result"]["passed"] is False
    original_core = daemon._implementation_parent(task)[0].invariant_core
    original_budget = daemon._implementation_parent(task)[0].budget
    commit = result["failed_preservation_result"]["preserved_commit"]

    restarted = PortalImplementationDaemon(
        todo_path=daemon.todo_path, repo_root=daemon.repo_root,
        state_path=daemon.state_path, strategy_path=daemon.strategy_path,
        events_path=daemon.events_path, worktree_root=daemon.worktree_root,
    )
    persisted = PortalTaskState.load(daemon.state_path)
    assert persisted.last_implementation_commit == commit
    assert persisted.last_implementation_returncode == 17
    assert task.task_id not in persisted.completed_task_ids
    assert not persisted.implementation_in_progress
    seed = restarted._prior_attempt_seed_plan(state=persisted, attempt=2)
    assert seed["reuse_prior_attempt"] is True
    assert seed["seed_ref"] == commit
    # Selection of a retained commit is distinct from authorizing its replay.
    assert restarted._prior_seed_proposal_authority(task)["ok"] is False

    prompt = restarted._build_implementation_prompt(task, attempt=2)
    diagnostic = restarted._implementation_diagnostics[restarted._canonical_ref(task)]
    assert diagnostic.receipt_id == result["diagnostic_receipt_id"]
    assert diagnostic.failure["returncode"] == 17
    assert diagnostic.failure["validation"]["passed"] is False
    assert diagnostic.failure["validation"]["attempted"] is False
    assert set(diagnostic.unresolved_requirements) == {
        content_identity({"task_id": task.task_id, "requirement": requirement})
        for requirement in (task.acceptance, *task.validation)
    }
    assert manifest["manifest_cid"] in prompt
    assert task.acceptance in prompt
    assert task.validation[0] in prompt
    assert "src/value.py" in prompt
    assert restarted._implementation_parent(task)[0].invariant_core == original_core
    assert restarted._implementation_parent(task)[0].budget == original_budget
    # The canonical source and its index still describe the admitted baseline;
    # retained edits are not silently substituted into those formal artifacts.
    assert git(daemon.repo_root, "rev-parse", "HEAD") == baseline
    assert (daemon.repo_root / "src/value.py").read_text() == "VALUE = 1\n"
    assert git(daemon.repo_root, "show", commit + ":src/value.py") == "VALUE = 2"
    if indexed:
        chunks = sorted((row for row in json.loads(prompt)["evidence"]
                         if row["kind"] == "semantic-context"),
                        key=lambda row: row["metadata"]["chunk_index"])
        assert chunks
        assert [row["metadata"]["chunk_index"] for row in chunks] == list(range(len(chunks)))
        assert all(row["metadata"]["chunk_count"] == len(chunks)
                   and row["metadata"]["required"] is True for row in chunks)
        # Native evidence chunks may divide a source literal; verify the
        # reassembled typed payload instead of a substring across wire frames.
        semantic = json.loads("".join(row["summary"] for row in chunks))
        assert semantic["raw_sources"]["src/value.py"] == "VALUE = 1\n"
        assert semantic["manifest"]["src/value.py"]["sha256"] == hashlib.sha256(b"VALUE = 1\n").hexdigest()
        assert semantic["completion_authority"] is False
