"""SCA-229: separate implementation merge from authoritative task acceptance.

Acceptance criteria:
* A receipt with ``completion_authoritative=false`` or pending gates cannot be
  marked authoritatively completed.
* Deterministic-only tasks reject any model invocation.
* Stale post-merge validation reopens acceptance without discarding the
  implementation commit.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ACCEPTANCE_REOPENED_STALE_EVENT,
    ACCEPTANCE_STATE_AUTHORITATIVE,
    ACCEPTANCE_STATE_MERGED_PENDING,
    ACCEPTANCE_STATE_REOPENED,
    AUTHORITATIVE_COMPLETION_DENIED_EVENT,
    AUTHORITATIVE_COMPLETION_GATE_KINDS,
    AUTHORITATIVE_COMPLETION_ADMITTED_EVENT,
    DETERMINISTIC_ONLY_MODEL_REJECTED_EVENT,
    IMPLEMENTATION_RECEIPT_SCHEMA,
    IMPLEMENTATION_MERGED_PENDING_EVENT,
    MERGE_TARGET_BINDING_SCHEMA,
    AuthoritativeCompletionGate,
    DeterministicOnlyPolicy,
    ImplementationReceipt,
    PortalTask,
    TodoImplementationDaemon,
    bound_gate_evidence,
    build_implementation_receipt,
    evaluate_authoritative_completion_gate,
    promote_authoritative_completion,
    reopen_acceptance_for_stale_post_merge_validation,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.status import (
    build_merged_pending_acceptance_status,
    build_reopened_acceptance_status,
    project_authoritative_acceptance_status,
)


def _git(repo, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )


def _git_output(repo, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _daemon(tmp_path, monkeypatch: pytest.MonkeyPatch) -> TodoImplementationDaemon:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Authoritative Completion Test")
    _git(repo, "config", "user.email", "authoritative@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text(
        "# Authoritative completion tasks\n\n"
        "## SCA-229 Separate implementation merge from authoritative task acceptance\n"
        "- Status: ready\n",
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text("state/\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")

    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SCA-",
        implement=True,
        implementation_command="model-command-must-not-run",
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_completion",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        daemon,
        "_decision_runtime_mutation",
        lambda _kind, _payload, action: action(),
    )
    return daemon


def _task(
    *,
    provider_role: str = "grok-implement, codex-review",
    task_id: str = "SCA-229",
) -> PortalTask:
    return PortalTask(
        task_id=task_id,
        title="Separate implementation merge from authoritative task acceptance",
        status="ready",
        completion="manual",
        priority="P0",
        track="completion-authority",
        outputs=[
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
            "todo_daemon/implementation_daemon.py"
        ],
        validation=[
            "python3 -m pytest external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_authoritative_task_completion.py -q"
        ],
        acceptance=(
            "A task whose receipt has completion_authoritative=false or "
            "pending gates cannot be marked authoritatively completed"
        ),
        metadata={
            "Provider role": provider_role,
            "Context budget tokens": "3072",
        },
    )


def _events(daemon: TodoImplementationDaemon) -> list[dict[str, Any]]:
    if not daemon.events_path.exists():
        return []
    return [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _full_gate_evidence(**overrides: Any) -> dict[str, Any]:
    evidence = {
        "merge": {"satisfied": True, "merge_commit": "abc123"},
        "freshness": {"satisfied": True, "stale": False},
        "semantic": {"satisfied": True, "passed": True},
        "proof": {"not_applicable": True, "satisfied": True},
        "provider_review": {"satisfied": True, "review_presence": "independent"},
        "deterministic_only": {"not_applicable": True, "satisfied": True},
    }
    evidence.update(overrides)
    return evidence


def _bound_gate_evidence(
    *,
    task_id: str,
    implementation_commit: str,
    merge_commit: str,
    repository_tree_id: str,
    deterministic_only: bool = False,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "task_id": task_id,
        "implementation_commit": implementation_commit,
        "merge_commit": merge_commit,
        "repository_tree_id": repository_tree_id,
    }
    validation = {
        **binding,
        "satisfied": True,
        "passed": True,
        "stale": False,
        "validation_scope": "post_merge",
        "validation_receipt_id": "validation-receipt",
    }
    return {
        "merge": bound_gate_evidence("merge", **binding, satisfied=True),
        "freshness": bound_gate_evidence("freshness", **validation),
        "semantic": bound_gate_evidence("semantic", **validation),
        "proof": bound_gate_evidence(
            "proof",
            **binding,
            satisfied=True,
            not_applicable=True,
            applicability_decision="no_declared_proof_obligation",
        ),
        "provider_review": bound_gate_evidence(
            "provider_review",
            **binding,
            satisfied=True,
            **(
                {
                    "not_applicable": True,
                    "route_kind": "deterministic_only",
                    "model_invocation_observed": False,
                }
                if deterministic_only
                else {
                    "review_presence": "independent",
                    "provider_result_admitted": True,
                    "review_receipt_id": "review-receipt",
                }
            ),
        ),
        "deterministic_only": bound_gate_evidence(
            "deterministic_only",
            **binding,
            satisfied=True,
            not_applicable=not deterministic_only,
            policy=(
                "deterministic_only"
                if deterministic_only
                else "not_deterministic_only"
            ),
            model_invocation_observed=False,
        ),
    }


def _queued_merge_request(
    daemon: TodoImplementationDaemon,
    task: PortalTask,
    *,
    metadata: dict[str, Any] | None = None,
) -> SimpleNamespace:
    implementation_commit = _git_output(daemon.repo_root, "rev-parse", "HEAD")
    return SimpleNamespace(
        branch_name=f"implementation/{task.task_id.lower()}-adversarial",
        commit_sha=implementation_commit,
        task_id=task.task_id,
        priority=task.priority,
        attempt=1,
        target_repository_id=daemon.merge_target_repository_id,
        target_branch=daemon.resolved_merge_target_branch,
        metadata={
            "target_binding_schema": MERGE_TARGET_BINDING_SCHEMA,
            "target_repository_id": daemon.merge_target_repository_id,
            "target_branch": daemon.resolved_merge_target_branch,
            "implementation_commit": implementation_commit,
            "task": {
                field: getattr(task, field)
                for field in PortalTask.__dataclass_fields__
            },
            **dict(metadata or {}),
        },
    )


def _run_queued_merge_callback(
    daemon: TodoImplementationDaemon,
    task: PortalTask,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata: dict[str, Any] | None = None,
    final_tree_id: str = "post-merge-tree",
) -> dict[str, Any]:
    monkeypatch.setattr(
        daemon,
        "_rehydrate_merge_request_branch",
        lambda **_kwargs: {"ready": True, "rehydrated": False},
    )
    monkeypatch.setattr(
        daemon,
        "_merge_branch_to_main",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": final_tree_id,
        },
    )
    request = _queued_merge_request(daemon, task, metadata=metadata)
    return daemon._merge_train_callback(request)


def test_implementation_receipt_merge_is_not_completion_authority() -> None:
    evidence = _bound_gate_evidence(
        task_id="SCA-229",
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        repository_tree_id="tree-sha",
    )
    receipt = build_implementation_receipt(
        task_id="SCA-229",
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        repository_tree_id="tree-sha",
        merged=True,
        validation_passed=True,
        gate_evidence=evidence,
    )
    assert receipt.merged is True
    assert receipt.implementation_commit == "impl-sha"
    assert receipt.merge_commit == "merge-sha"
    assert receipt.completion_authoritative is False
    assert receipt.acceptance_state == ACCEPTANCE_STATE_MERGED_PENDING
    # Structural gates may be green, but the receipt flag stays false until
    # explicit promotion.
    assert receipt.pending_gates == ()


def test_completion_authoritative_false_blocks_authoritative_completion() -> None:
    receipt = build_implementation_receipt(
        task_id="SCA-229",
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        repository_tree_id="tree-sha",
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id="SCA-229",
            implementation_commit="impl-sha",
            merge_commit="merge-sha",
            repository_tree_id="tree-sha",
        ),
    )
    gate = evaluate_authoritative_completion_gate(receipt)
    assert gate.admitted is False
    assert gate.completion_authoritative is False
    assert "completion_authoritative_false" in gate.reason_codes
    assert gate.acceptance_state == ACCEPTANCE_STATE_MERGED_PENDING


def test_pending_gates_block_authoritative_completion() -> None:
    receipt = build_implementation_receipt(
        task_id="SCA-229",
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        merged=True,
        validation_passed=False,
        gate_evidence={
            "merge": {"satisfied": True},
            # freshness/semantic/proof/provider_review/deterministic_only absent
        },
    )
    assert set(receipt.pending_gates) >= {
        "freshness",
        "semantic",
        "proof",
        "provider_review",
        "deterministic_only",
    }
    gate = evaluate_authoritative_completion_gate(receipt)
    assert gate.admitted is False
    assert gate.pending_gates
    assert gate.completion_authoritative is False


def test_daemon_refuses_to_mark_complete_when_receipt_not_authoritative(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    marked: list[str] = []

    def _forbid_complete(*_args, **_kwargs):
        marked.append("completed")
        raise AssertionError("board must not flip to completed")

    monkeypatch.setattr(daemon, "_mark_task_or_bundle_completed_in_todo", _forbid_complete)

    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        merged=True,
        validation_passed=True,
        gate_evidence=_full_gate_evidence(),
        completion_authoritative=False,
    )
    result = daemon.mark_authoritatively_completed_if_admitted(
        task,
        receipt,
        promote=False,
    )
    assert result["authoritatively_completed"] is False
    assert result["completion_authoritative"] is False
    assert result["updated"] is False
    assert marked == []
    denied = [
        item
        for item in _events(daemon)
        if item.get("type") == AUTHORITATIVE_COMPLETION_DENIED_EVENT
    ]
    assert len(denied) == 1
    assert denied[0]["completion_authoritative"] is False


def test_daemon_refuses_pending_gates_even_if_flag_true(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda *_a, **_k: pytest.fail("pending gates cannot complete"),
    )
    receipt = ImplementationReceipt(
        task_id=task.task_id,
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        merged=True,
        validation_passed=True,
        completion_authoritative=True,
        pending_gates=("proof", "freshness"),
        gate_evidence={"merge": {"satisfied": True}},
        acceptance_state=ACCEPTANCE_STATE_MERGED_PENDING,
    )
    result = daemon.mark_authoritatively_completed_if_admitted(
        task,
        receipt,
        promote=False,
    )
    assert result["authoritatively_completed"] is False
    assert "proof" in result["pending_gates"]
    assert "freshness" in result["pending_gates"]


def test_promotion_admits_when_all_gates_satisfied(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only")
    completed: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda t, **_kwargs: completed.append(t.task_id) or {
            "updated": True,
            "task_id": t.task_id,
            "reason": "updated",
        },
    )
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        repository_tree_id="tree-sha",
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id=task.task_id,
            implementation_commit="impl-sha",
            merge_commit="merge-sha",
            repository_tree_id="tree-sha",
            deterministic_only=True,
        ),
        deterministic_only=True,
        model_invocation_observed=False,
    )
    promoted, gate = promote_authoritative_completion(receipt)
    assert gate.admitted is True
    assert promoted.completion_authoritative is True
    assert promoted.acceptance_state == ACCEPTANCE_STATE_AUTHORITATIVE
    result = daemon.mark_authoritatively_completed_if_admitted(
        task,
        receipt,
        promote=True,
    )
    assert result["authoritatively_completed"] is True
    assert result["completion_authoritative"] is True
    assert completed == [task.task_id]
    admitted = [
        item
        for item in _events(daemon)
        if item.get("type") == AUTHORITATIVE_COMPLETION_ADMITTED_EVENT
    ]
    assert len(admitted) == 1


def test_merged_pending_preserves_implementation_commit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit="keep-this-commit",
        merge_commit="merge-sha",
        merged=True,
        validation_passed=False,
        gate_evidence={"merge": {"satisfied": True}},
    )
    payload = daemon.record_merged_pending_acceptance(task, receipt)
    assert payload["implementation_commit"] == "keep-this-commit"
    assert payload["completion_authoritative"] is False
    assert payload["board_status"] == "pending"
    assert payload["acceptance_state"] == ACCEPTANCE_STATE_MERGED_PENDING
    pending_events = [
        item
        for item in _events(daemon)
        if item.get("type") == IMPLEMENTATION_MERGED_PENDING_EVENT
    ]
    assert len(pending_events) == 1
    assert pending_events[0]["implementation_commit"] == "keep-this-commit"


def test_deterministic_only_policy_rejects_model_invocation() -> None:
    task = _task(provider_role="deterministic-only", task_id="SCA-DET")
    policy = DeterministicOnlyPolicy.for_task(task)
    assert policy.deterministic_only is True
    assert policy.allows_model_invocation() is False
    rejection = policy.reject_model_invocation(provider="grok", reason="route")
    assert rejection["rejected"] is True
    assert rejection["completion_authoritative"] is False
    assert "deterministic_only_forbids_model_invocation" in rejection["reason"]


def test_daemon_rejects_model_invocation_for_deterministic_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only", task_id="SCA-DET")
    decision = daemon.reject_deterministic_only_model_invocation(
        task,
        provider="grok-implement",
        reason="implementation_command",
    )
    assert decision["rejected"] is True
    assert decision["deterministic_only"] is True
    events = [
        item
        for item in _events(daemon)
        if item.get("type") == DETERMINISTIC_ONLY_MODEL_REJECTED_EVENT
    ]
    assert len(events) == 1

    # Model-assisted provider role may still attempt models.
    assisted = _task(provider_role="grok-implement, codex-review")
    allowed = daemon.reject_deterministic_only_model_invocation(
        assisted,
        provider="grok-implement",
    )
    assert allowed["rejected"] is False


def test_deterministic_only_receipt_blocks_when_model_observed() -> None:
    receipt = build_implementation_receipt(
        task_id="SCA-DET",
        implementation_commit="impl",
        merge_commit="merge",
        merged=True,
        validation_passed=True,
        gate_evidence=_full_gate_evidence(
            provider_review={"not_applicable": True, "satisfied": True},
        ),
        deterministic_only=True,
        model_invocation_observed=True,
    )
    assert "deterministic_only" in receipt.pending_gates
    gate = evaluate_authoritative_completion_gate(
        receipt,
        require_completion_authoritative_flag=False,
    )
    assert gate.admitted is False
    assert "deterministic_only" in gate.pending_gates


def test_stale_post_merge_reopens_acceptance_preserves_commit() -> None:
    receipt = build_implementation_receipt(
        task_id="SCA-229",
        implementation_commit="durable-impl-sha",
        merge_commit="merge-sha",
        repository_tree_id="tree-sha",
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id="SCA-229",
            implementation_commit="durable-impl-sha",
            merge_commit="merge-sha",
            repository_tree_id="tree-sha",
        ),
    )
    promoted, gate = promote_authoritative_completion(receipt)
    assert gate.admitted is True
    assert promoted.completion_authoritative is True

    reopened = reopen_acceptance_for_stale_post_merge_validation(promoted)
    assert reopened.implementation_commit == "durable-impl-sha"
    assert reopened.merge_commit == "merge-sha"
    assert reopened.completion_authoritative is False
    assert reopened.validation_stale is True
    assert reopened.acceptance_state == ACCEPTANCE_STATE_REOPENED
    assert "freshness" in reopened.pending_gates
    denied = evaluate_authoritative_completion_gate(reopened)
    assert denied.admitted is False
    assert denied.completion_authoritative is False


def test_daemon_reopens_stale_post_merge_without_discarding_commit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit="keep-impl",
        merge_commit="keep-merge",
        repository_tree_id="keep-tree",
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id=task.task_id,
            implementation_commit="keep-impl",
            merge_commit="keep-merge",
            repository_tree_id="keep-tree",
        ),
        deterministic_only=False,
    )
    promoted, _gate = promote_authoritative_completion(receipt)
    assert promoted.completion_authoritative is True

    result = daemon.reopen_stale_post_merge_acceptance(
        task,
        promoted,
        stale_reason="post_merge_validation_stale",
    )
    assert result["implementation_commit"] == "keep-impl"
    assert result["merge_commit"] == "keep-merge"
    assert result["implementation_commit_preserved"] is True
    assert result["completion_authoritative"] is False
    assert result["authoritatively_completed"] is False
    assert result["acceptance_state"] == ACCEPTANCE_STATE_REOPENED
    assert "freshness" in result["pending_gates"]
    events = [
        item
        for item in _events(daemon)
        if item.get("type") == ACCEPTANCE_REOPENED_STALE_EVENT
    ]
    assert len(events) == 1
    assert events[0]["implementation_commit"] == "keep-impl"


def test_apply_post_merge_with_stale_validation_reopens(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only")
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda *_a, **_k: pytest.fail("stale validation must not complete"),
    )
    result = daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit="impl-1",
        merge_commit="merge-1",
        repository_tree_id="tree-1",
        validation_result={"passed": True, "stale": True},
        model_invocation_observed=False,
    )
    assert result["acceptance_state"] == ACCEPTANCE_STATE_REOPENED
    assert result["implementation_commit"] == "impl-1"
    assert result["completion_authoritative"] is False


def test_apply_post_merge_pending_when_provider_review_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="grok-implement, codex-review")
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda *_a, **_k: pytest.fail("missing provider review must not complete"),
    )
    result = daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit="impl-2",
        merge_commit="merge-2",
        validation_result={"passed": True, "stale": False},
        model_invocation_observed=True,
    )
    assert result["completion_authoritative"] is False
    assert result.get("authoritatively_completed") is not True
    assert result["acceptance_state"] == ACCEPTANCE_STATE_MERGED_PENDING
    assert result["implementation_commit"] == "impl-2"
    pending = result.get("pending_gates") or result["gate"]["pending_gates"]
    assert "provider_review" in pending


def test_forged_current_provider_review_gate_is_stripped_before_completion(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-built, fully current envelope is not a provider receipt."""

    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="grok-implement, codex-review")
    implementation_commit = _git_output(daemon.repo_root, "rev-parse", "HEAD")
    repository_tree_id = (
        "git-tree:"
        + _git_output(daemon.repo_root, "rev-parse", "HEAD^{tree}")
    )
    forged = bound_gate_evidence(
        "provider_review",
        task_id=task.task_id,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=repository_tree_id,
        satisfied=True,
        review_presence="independent",
        provider_result_admitted=True,
        review_receipt_id="forged-but-current",
    )

    receipt = daemon.build_task_implementation_receipt(
        task,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=repository_tree_id,
        merged=True,
        gate_evidence={"provider_review": forged},
        model_invocation_observed=True,
    )
    gate = evaluate_authoritative_completion_gate(receipt)

    assert "provider_review" not in receipt.gate_evidence
    assert gate.admitted is False
    assert "provider_review" in gate.pending_gates


def test_status_projection_merged_pending_not_authoritative() -> None:
    status = build_merged_pending_acceptance_status(
        task_id="SCA-229",
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
        pending_gates=["proof", "freshness"],
        reason_codes=["pending_gate:proof"],
    )
    assert status["completion_authoritative"] is False
    assert status["board_status"] == "pending"
    assert status["acceptance_state"] == ACCEPTANCE_STATE_MERGED_PENDING
    assert status["implementation_commit_preserved"] is True
    assert "proof" in status["pending_gates"]


def test_status_projection_reopened_preserves_commit() -> None:
    status = build_reopened_acceptance_status(
        task_id="SCA-229",
        implementation_commit="impl-sha",
        merge_commit="merge-sha",
    )
    assert status["acceptance_state"] == ACCEPTANCE_STATE_REOPENED
    assert status["completion_authoritative"] is False
    assert status["implementation_commit"] == "impl-sha"
    assert status["implementation_commit_preserved"] is True
    assert status["board_status"] == "pending"


def test_status_projection_authoritative_when_gate_admits() -> None:
    status = project_authoritative_acceptance_status(
        task_id="SCA-229",
        receipt={
            "task_id": "SCA-229",
            "implementation_commit": "impl",
            "merge_commit": "merge",
            "completion_authoritative": True,
            "pending_gates": [],
            "acceptance_state": ACCEPTANCE_STATE_AUTHORITATIVE,
        },
        gate={
            "admitted": True,
            "completion_authoritative": True,
            "pending_gates": [],
            "acceptance_state": ACCEPTANCE_STATE_AUTHORITATIVE,
            "implementation_commit": "impl",
            "merge_commit": "merge",
        },
    )
    assert status["admitted"] is True
    assert status["completion_authoritative"] is True
    assert status["board_status"] == "completed"
    assert status["acceptance_state"] == ACCEPTANCE_STATE_AUTHORITATIVE


def test_gate_kinds_cover_required_boundaries() -> None:
    assert AUTHORITATIVE_COMPLETION_GATE_KINDS == (
        "merge",
        "freshness",
        "semantic",
        "proof",
        "provider_review",
        "deterministic_only",
    )
    assert AuthoritativeCompletionGate(admitted=False).admitted is False
    assert ImplementationReceipt(task_id="x").completion_authoritative is False


def test_empty_receipt_cannot_promote() -> None:
    promoted, gate = promote_authoritative_completion({})
    assert promoted.completion_authoritative is False
    assert gate.admitted is False
    assert gate.completion_authoritative is False


def test_forged_empty_evidence_receipt_cannot_promote() -> None:
    forged = {
        "schema": IMPLEMENTATION_RECEIPT_SCHEMA,
        "task_id": "SCA-229",
        "implementation_commit": "impl",
        "merge_commit": "merge",
        "repository_tree_id": "tree",
        "merged": True,
        "validation_passed": True,
        "validation_stale": False,
        "completion_authoritative": False,
        "pending_gates": [],
        "gate_evidence": {},
        "acceptance_state": ACCEPTANCE_STATE_MERGED_PENDING,
    }
    promoted, gate = promote_authoritative_completion(forged)
    assert promoted.completion_authoritative is False
    assert gate.admitted is False
    assert gate.completion_authoritative is False


@pytest.mark.parametrize(
    "mismatch",
    (
        "task_id",
        "schema",
        "implementation_commit",
        "merge_commit",
        "repository_tree_id",
    ),
)
def test_receipt_identity_binding_mismatch_cannot_complete(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task()
    implementation_commit = _git_output(daemon.repo_root, "rev-parse", "HEAD")
    repository_tree_id = _git_output(
        daemon.repo_root,
        "rev-parse",
        "HEAD^{tree}",
    )
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        repository_tree_id=repository_tree_id,
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id=task.task_id,
            implementation_commit=implementation_commit,
            merge_commit=implementation_commit,
            repository_tree_id=repository_tree_id,
        ),
    ).to_dict()
    replacements = {
        "task_id": "SCA-FORGED",
        "schema": f"{IMPLEMENTATION_RECEIPT_SCHEMA}-forged",
        "implementation_commit": "f" * 40,
        "merge_commit": "d" * 40,
        "repository_tree_id": "e" * 40,
    }
    receipt[mismatch] = replacements[mismatch]
    marked: list[str] = []
    monkeypatch.setattr(
        daemon,
        "_mark_task_or_bundle_completed_in_todo",
        lambda selected: marked.append(selected.task_id)
        or {"updated": True, "task_id": selected.task_id},
    )

    result = daemon.mark_authoritatively_completed_if_admitted(
        task,
        receipt,
        promote=True,
    )

    assert result["authoritatively_completed"] is False
    assert result["completion_authoritative"] is False
    assert marked == []


@pytest.mark.parametrize("completion_source", ("merged_event", "shared_queue"))
def test_non_authoritative_merge_completion_source_does_not_mutate_board(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    completion_source: str,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    daemon.implement = False
    # Use the task parsed from the live board so its canonical identity is
    # exactly the one run_once() and the shared merge queue project.
    task = daemon._load_tasks()[0]
    implementation_commit = _git_output(daemon.repo_root, "rev-parse", "HEAD")
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit=implementation_commit,
        merge_commit=implementation_commit,
        merged=True,
        validation_passed=True,
        gate_evidence=_full_gate_evidence(),
        completion_authoritative=False,
    )
    pending = daemon.record_merged_pending_acceptance(task, receipt)
    assert pending["completion_authoritative"] is False
    canonical_task_cid = daemon._canonical_ref(task)
    if completion_source == "merged_event":
        daemon._record_event(
            "implementation_finished",
            {
                "task_id": task.task_id,
                "returncode": 0,
                "implementation_commit": implementation_commit,
                "merge_result": {
                    "merged": True,
                    "returncode": 0,
                    "merge_commit": implementation_commit,
                },
                "completion_authoritative": False,
                "acceptance_result": pending,
            },
        )
        monkeypatch.setattr(
            daemon,
            "_shared_merge_queue_task_cids",
            lambda _method_name: set(),
        )
    else:
        monkeypatch.setattr(
            daemon,
            "_shared_merge_queue_task_cids",
            lambda method_name: (
                {canonical_task_cid}
                if method_name == "completed_canonical_task_ids"
                else set()
            ),
        )
    monkeypatch.setattr(daemon, "_consume_one_merge_candidate", lambda: None)
    original_board = daemon.todo_path.read_text(encoding="utf-8")

    result = daemon.run_once()

    assert daemon.todo_path.read_text(encoding="utf-8") == original_board
    assert task.task_id not in result.get("completed_task_ids", [])
    assert not result.get("merged_status_repair", {}).get("updated")


def test_queued_callback_missing_validation_stays_pending(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only")
    original_board = daemon.todo_path.read_text(encoding="utf-8")

    result = _run_queued_merge_callback(
        daemon,
        task,
        monkeypatch,
        metadata={},
    )

    assert result["merged"] is False
    assert result["reason"] == "validation_evidence_missing"
    assert daemon.todo_path.read_text(encoding="utf-8") == original_board


def test_queued_callback_pre_merge_validation_unbound_to_final_tree_stays_pending(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only")
    original_board = daemon.todo_path.read_text(encoding="utf-8")

    result = _run_queued_merge_callback(
        daemon,
        task,
        monkeypatch,
        final_tree_id="post-merge-tree",
        metadata={
            "validation_proof": {
                "attempted": True,
                "passed": True,
                "stale": False,
                "repository_tree_id": "pre-merge-tree",
                "selection": {"scope": "pre_merge"},
            },
        },
    )

    acceptance = result["acceptance_result"]
    assert acceptance["completion_authoritative"] is False
    assert acceptance.get("authoritatively_completed") is not True
    assert "freshness" in acceptance["pending_gates"]
    assert daemon.todo_path.read_text(encoding="utf-8") == original_board


def test_raw_provider_admission_boolean_without_typed_receipt_stays_pending(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="grok-implement, codex-review")
    original_board = daemon.todo_path.read_text(encoding="utf-8")

    result = _run_queued_merge_callback(
        daemon,
        task,
        monkeypatch,
        metadata={
            "validation_proof": {
                "attempted": True,
                "passed": True,
                "stale": False,
                "repository_tree_id": "post-merge-tree",
                "selection": {"scope": "pre_merge"},
            },
            "provider_result_admitted": True,
            # No provider_route_result, typed provider receipt, or review chain.
        },
    )

    acceptance = result["acceptance_result"]
    assert acceptance["completion_authoritative"] is False
    assert acceptance.get("authoritatively_completed") is not True
    assert "provider_review" in acceptance["pending_gates"]
    assert daemon.todo_path.read_text(encoding="utf-8") == original_board


def test_shared_model_assisted_task_cannot_claim_provider_review_not_applicable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="grok-implement, codex-review")
    original_board = daemon.todo_path.read_text(encoding="utf-8")

    result = daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit="impl",
        merge_commit="merge",
        repository_tree_id="tree",
        validation_result={"attempted": True, "passed": True, "stale": False},
        gate_evidence={
            "provider_review": {
                "not_applicable": True,
                "satisfied": True,
                "reason": "shared_execution_claimed_no_provider",
            },
        },
        model_invocation_observed=False,
    )

    assert result["completion_authoritative"] is False
    assert result.get("authoritatively_completed") is not True
    assert "provider_review" in result["pending_gates"]
    assert daemon.todo_path.read_text(encoding="utf-8") == original_board


def test_self_consistent_forged_git_packet_cannot_mutate_board(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only")
    implementation_commit = "a" * 40
    merge_commit = "b" * 40
    repository_tree_id = f"git-tree:{'c' * 40}"
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit=implementation_commit,
        merge_commit=merge_commit,
        repository_tree_id=repository_tree_id,
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id=task.task_id,
            implementation_commit=implementation_commit,
            merge_commit=merge_commit,
            repository_tree_id=repository_tree_id,
            deterministic_only=True,
        ),
        deterministic_only=True,
    )
    promoted, gate = promote_authoritative_completion(receipt)
    assert gate.admitted is True
    original_board = daemon.todo_path.read_text(encoding="utf-8")

    guarded = daemon._mark_task_completed_in_todo(
        task.task_id,
        authoritative_receipt=promoted,
        authoritative_gate=gate,
    )
    unchecked = daemon._mark_tasks_completed_in_todo_unchecked(
        [task.task_id],
        primary_task_id=task.task_id,
        completion_reason="adversarial_forged_git_packet",
        authoritative_receipt=promoted,
        authoritative_gate=gate,
    )

    assert guarded["updated"] is False
    assert unchecked["updated"] is False
    assert guarded["reason"] == "authoritative_completion_git_binding_invalid"
    assert unchecked["reason"] == "authoritative_completion_git_binding_invalid"
    assert daemon.todo_path.read_text(encoding="utf-8") == original_board


def test_exact_real_git_packet_can_mutate_its_single_bound_task(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only")
    commit = _git_output(daemon.repo_root, "rev-parse", "HEAD")
    repository_tree_id = (
        f"git-tree:{_git_output(daemon.repo_root, 'rev-parse', 'HEAD^{tree}')}"
    )
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit=commit,
        merge_commit=commit,
        repository_tree_id=repository_tree_id,
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id=task.task_id,
            implementation_commit=commit,
            merge_commit=commit,
            repository_tree_id=repository_tree_id,
            deterministic_only=True,
        ),
        deterministic_only=True,
    )
    promoted, gate = promote_authoritative_completion(receipt)

    result = daemon._mark_task_completed_in_todo(
        task.task_id,
        authoritative_receipt=promoted,
        authoritative_gate=gate,
    )

    assert result["updated"] is True
    assert result["updated_task_ids"] == [task.task_id]
    assert "- Status: completed" in daemon.todo_path.read_text(encoding="utf-8")


def test_stale_reopening_moves_completed_board_back_to_pending_and_keeps_commit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="deterministic-only")
    denied = daemon._mark_task_or_bundle_completed_in_todo(task)
    assert denied["updated"] is False
    assert denied["reason"] == "authoritative_completion_packet_missing"
    unchecked = daemon._mark_tasks_completed_in_todo_unchecked(
        [task.task_id],
        primary_task_id=task.task_id,
        completion_reason="adversarial_direct_call",
    )
    assert unchecked["updated"] is False
    daemon.todo_path.write_text(
        daemon.todo_path.read_text(encoding="utf-8").replace(
            "- Status: ready",
            "- Status: completed",
        ),
        encoding="utf-8",
    )
    assert "- Status: completed" in daemon.todo_path.read_text(encoding="utf-8")
    receipt = build_implementation_receipt(
        task_id=task.task_id,
        implementation_commit="keep-implementation-commit",
        merge_commit="keep-merge-commit",
        repository_tree_id="validated-tree",
        merged=True,
        validation_passed=True,
        gate_evidence=_bound_gate_evidence(
            task_id=task.task_id,
            implementation_commit="keep-implementation-commit",
            merge_commit="keep-merge-commit",
            repository_tree_id="validated-tree",
            deterministic_only=True,
        ),
        deterministic_only=True,
    )
    promoted, gate = promote_authoritative_completion(receipt)
    assert gate.admitted is True
    bundle = daemon._mark_tasks_completed_in_todo_unchecked(
        [task.task_id, "SCA-SIBLING"],
        primary_task_id=task.task_id,
        completion_reason="adversarial_bundle_reuse",
        authoritative_receipt=promoted,
        authoritative_gate=gate,
    )
    assert bundle["updated"] is False
    assert bundle["reason"] == "bundle_member_authority_missing"

    result = daemon.reopen_stale_post_merge_acceptance(
        task,
        promoted,
        stale_reason="post_merge_tree_changed",
    )

    assert result["implementation_commit"] == "keep-implementation-commit"
    assert result["implementation_commit_preserved"] is True
    assert result.get("board_status") == "pending"
    board = daemon.todo_path.read_text(encoding="utf-8")
    assert "- Status: completed" not in board
    assert "- Status: todo" in board or "- Status: pending" in board


def test_live_acceptance_results_project_merged_pending_and_reopened(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _daemon(tmp_path, monkeypatch)
    task = _task(provider_role="grok-implement, codex-review")
    merged_pending = daemon.apply_post_merge_authoritative_acceptance(
        task,
        implementation_commit="pending-implementation",
        merge_commit="pending-merge",
        repository_tree_id="pending-tree",
        validation_result={"attempted": True, "passed": True, "stale": False},
        model_invocation_observed=True,
    )
    pending_status = project_authoritative_acceptance_status(
        task_id=task.task_id,
        receipt=merged_pending["receipt"],
        gate=merged_pending["gate"],
    )
    assert pending_status["acceptance_state"] == ACCEPTANCE_STATE_MERGED_PENDING
    assert pending_status["board_status"] == "pending"
    assert pending_status["completion_authoritative"] is False
    assert pending_status["implementation_commit"] == "pending-implementation"

    reopened = daemon.apply_post_merge_authoritative_acceptance(
        _task(provider_role="deterministic-only"),
        implementation_commit="reopened-implementation",
        merge_commit="reopened-merge",
        repository_tree_id="reopened-tree",
        validation_result={"attempted": True, "passed": True, "stale": True},
        model_invocation_observed=False,
    )
    reopened_status = project_authoritative_acceptance_status(
        task_id=task.task_id,
        receipt=reopened["receipt"],
        gate=reopened["gate"],
    )
    assert reopened_status["acceptance_state"] == ACCEPTANCE_STATE_REOPENED
    assert reopened_status["board_status"] == "pending"
    assert reopened_status["completion_authoritative"] is False
    assert reopened_status["implementation_commit"] == "reopened-implementation"
