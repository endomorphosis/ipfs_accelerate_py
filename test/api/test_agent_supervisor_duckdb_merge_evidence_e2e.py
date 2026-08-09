from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor.checkout_lock import (
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.code_evidence_graph import (
    POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA,
    CodeImpactIndex,
    PostMergeEvidenceReceipt,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    EvidenceAuthority,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge_train import (
    ParallelAcceptanceReceipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DUCKDB_POST_MERGE_EVIDENCE_INPUT_SCHEMA,
    DuckDBMergeIntegratedReceipt,
    PortalImplementationDaemon,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    HermeticValidationPolicy,
    ImpactValidationCheck,
    ImpactValidationKind,
    RepositoryValidationPolicy,
)
from test.api.test_agent_supervisor_task_source_e2e import _sources


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _content_receipt(record: dict[str, object], field: str) -> dict[str, object]:
    return {**record, field: content_identity(record)}


def _proof(
    *,
    obligation_id: str,
    repository_id: str,
    repository_tree_id: str,
    policy_id: str,
    observed_at: str,
) -> dict[str, object]:
    return ProofReceipt(
        obligation_id=obligation_id,
        plan_id="plan:merge-evidence-e2e",
        attempt_id=f"attempt:{obligation_id}",
        repository_id=repository_id,
        repository_tree_id=repository_tree_id,
        ast_scope_ids=("scope:implemented",),
        premise_ids=(),
        translator_id="translator:e2e",
        solver_id="solver:e2e",
        kernel_id=f"kernel:{obligation_id}",
        toolchain_id="toolchain:e2e",
        policy_id=policy_id,
        resource_budget=ResourceBudget(
            wall_time_ms=10_000,
            cpu_time_ms=8_000,
            memory_bytes=64 * 1024 * 1024,
            max_processes=2,
        ),
        verdict=ProofVerdict.PROVED,
        evidence=(
            ProofEvidence(
                kind=EvidenceKind.KERNEL_VERIFICATION,
                authority=EvidenceAuthority.KERNEL,
                verdict=EvidenceVerdict.ACCEPTED,
                artifact_id=f"artifact:{obligation_id}",
                subject_id=obligation_id,
                verifier_id=f"kernel:{obligation_id}",
                independent=True,
            ),
        ),
        started_at=observed_at,
        finished_at=observed_at,
    ).to_dict()


def _remove_raw_outputs(value: object) -> None:
    if isinstance(value, dict):
        for name in tuple(value):
            normalized = str(name).lower().replace("-", "_")
            if normalized in {"output", "stdout", "stderr", "raw_output"}:
                value.pop(name)
            else:
                _remove_raw_outputs(value[name])
    elif isinstance(value, list):
        for item in value:
            _remove_raw_outputs(item)


def _evidence_input(
    daemon: PortalImplementationDaemon,
    *,
    tree_id: str,
    workspace: Path,
    task_id: str,
    policy_id: str,
) -> dict[str, object]:
    observed = datetime.now(UTC)
    observed_at = observed.isoformat()
    repository_id = checkout_repository_id(workspace)
    check = ImpactValidationCheck(
        "declared-unit",
        ImpactValidationKind.UNIT,
        "python -m pytest -q test_declared.py",
        cacheable=False,
    )
    report = daemon.validation_scheduler.run_impact_selected(
        (check,),
        workspace_path=workspace,
        impact_index=CodeImpactIndex(
            repository_tree_id=tree_id,
            symbol_paths={},
            symbol_dependencies={},
            path_dependencies={"implemented.txt": ()},
            validation_targets={},
        ),
        changed_paths=("implemented.txt",),
        repository_policy=RepositoryValidationPolicy(
            required_kinds=(ImpactValidationKind.UNIT,),
            kind_dependencies={},
            require_acceptance_coverage=False,
            require_transitive_validation=False,
        ),
        target_tree_id=tree_id,
        runner=daemon._validation_command_runner,
        hermetic_policy=HermeticValidationPolicy(
            stability_runs=2,
            complete_selected_dag=True,
            required_techniques=(),
        ),
    )
    assert report["passed"] is True
    report = copy.deepcopy(report)
    _remove_raw_outputs(report)
    validation_receipt = report["impact_validation_receipt"]
    result = report["results"][0]

    semantic = _content_receipt(
        {
            "kind": "semantic",
            "repository_tree_id": tree_id,
            "status": "passed",
            "freshness": "current",
            "observed_at": observed_at,
            "source_validation_receipt_id": result[
                "validation_result_digest"
            ],
        },
        "validation_receipt_id",
    )
    protocol = _content_receipt(
        {
            "kind": "protocol",
            "repository_tree_id": tree_id,
            "status": "passed",
            "freshness": "current",
            "observed_at": observed_at,
            "source_validation_receipt_id": result[
                "validation_result_digest"
            ],
        },
        "validation_receipt_id",
    )
    legal = _content_receipt(
        {
            "obligation_id": "legal:merge-evidence-e2e",
            "repository_tree_id": tree_id,
            "status": "proved",
            "freshness": "current",
            "observed_at": observed_at,
        },
        "receipt_id",
    )
    theorem = _content_receipt(
        {
            "obligation_id": "theorem:merge-evidence-e2e",
            "repository_tree_id": tree_id,
            "status": "proved",
            "freshness": "current",
            "observed_at": observed_at,
        },
        "receipt_id",
    )
    proofs = [
        _proof(
            obligation_id=str(item["obligation_id"]),
            repository_id=repository_id,
            repository_tree_id=tree_id,
            policy_id=policy_id,
            observed_at=observed_at,
        )
        for item in (legal, theorem)
    ]
    source_hash = "sha256:" + hashlib.sha256(
        (workspace / "implemented.txt").read_bytes()
    ).hexdigest()
    return {
        "schema": DUCKDB_POST_MERGE_EVIDENCE_INPUT_SCHEMA,
        "policy_id": policy_id,
        "assembled_at": observed_at,
        "freshness_deadline": (observed + timedelta(hours=1)).isoformat(),
        "validation_report": report,
        "validation_receipt": validation_receipt,
        "semantic_checks": [semantic],
        "protocol_checks": [protocol],
        "legal_logic_obligations": [legal],
        "theorem_obligations": [theorem],
        "proof_receipts": proofs,
        "criterion_coverage": [
            {
                "criterion": POST_MERGE_EVIDENCE_ACCEPTANCE_CRITERIA[0],
                "repository_tree_id": tree_id,
                "implementation": ["implemented.txt"],
                "receipt_ids": [
                    validation_receipt["receipt_id"],
                    semantic["validation_receipt_id"],
                    protocol["validation_receipt_id"],
                ],
                "freshness": "current",
                "observed_at": observed_at,
            }
        ],
        "merged_tree_records": {
            "ast_records": [
                {
                    "scope_id": "scope:implemented",
                    "kind": "qualified_symbol",
                    "qualified_symbol": "fixture.implemented",
                    "repository_tree_id": tree_id,
                    "path": "implemented.txt",
                    "source_hash": source_hash,
                }
            ]
        },
    }


def _setup(
    tmp_path: Path,
    *,
    now: list[float],
    changed_submodule_paths: set[str] | None = None,
) -> dict[str, object]:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Merge Evidence E2E")
    _git(tmp_path, "config", "user.email", "merge-evidence@example.invalid")
    (tmp_path / "test_declared.py").write_text(
        "def test_declared():\n    assert True\n", encoding="utf-8"
    )
    (tmp_path / "implemented.txt").write_text("baseline\n", encoding="utf-8")
    _git(tmp_path, "add", "test_declared.py", "implemented.txt")
    _git(tmp_path, "commit", "-m", "baseline")
    baseline = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-b", "implementation/fix-001")
    (tmp_path / "implemented.txt").write_text("implemented\n", encoding="utf-8")
    _git(tmp_path, "add", "implemented.txt")
    _git(tmp_path, "commit", "-m", "FIX-001: implement")
    candidate = _git(tmp_path, "rev-parse", "HEAD")
    candidate_tree = _git(tmp_path, "rev-parse", "HEAD^{tree}")

    _markdown, database = _sources(tmp_path)
    runtime = tmp_path / "runtime"
    queue = MergeQueue(
        runtime / "merge-queue",
        max_age_seconds=1,
        clock=lambda: now[0],
    )
    daemon = PortalImplementationDaemon(
        task_source=database,
        state_path=runtime / "state.json",
        strategy_path=runtime / "strategy.json",
        events_path=runtime / "events.jsonl",
        repo_root=tmp_path,
        merge_target_branch="main",
        worktree_pool_enabled=False,
        validation_cache_dir=runtime / "validation-cache",
        merge_queue=queue,
        merge_queue_dir=runtime / "merge-queue",
    )
    task = daemon._load_tasks()[0]
    tree_id = f"git-tree:{candidate_tree}"
    policy_id = "policy:merge-evidence-e2e"
    post_merge_input = _evidence_input(
        daemon,
        tree_id=tree_id,
        workspace=tmp_path,
        task_id=task.task_id,
        policy_id=policy_id,
    )
    impact_report = post_merge_input["validation_report"]
    proposal_basis = {
        "schema": "merge-evidence-e2e-proposal@1",
        "task_id": task.task_id,
        "candidate_tree": tree_id,
        "policy_id": policy_id,
    }
    validation_result = {
        "attempted": True,
        "passed": True,
        "returncode": 0,
        "selection": {"scope": "pre_merge"},
        "results": impact_report["results"],
        "validation_dag_receipt": impact_report[
            "impact_validation_receipt"
        ],
        "proposal_gate": {
            "accepted": True,
            "proposal_id": content_identity(proposal_basis),
            "policy_id": policy_id,
            "receipt_id": content_identity(
                {**proposal_basis, "accepted": True}
            ),
        },
        "post_merge_evidence_input": post_merge_input,
    }
    request, _ = daemon._enqueue_merge_candidate(
        branch_name="implementation/fix-001",
        implementation_commit=candidate,
        baseline_ref=baseline,
        worktree_path=None,
        task=task,
        attempt=1,
        validation_result=validation_result,
        changed_submodule_paths=changed_submodule_paths,
    )
    _git(tmp_path, "checkout", "main")
    return {
        "database": database,
        "runtime": runtime,
        "queue": queue,
        "daemon": daemon,
        "task": task,
        "request": request,
        "candidate": candidate,
    }


def _restart(setup: dict[str, object], *, now: list[float], suffix: str):
    runtime = setup["runtime"]
    queue = MergeQueue(
        runtime / "merge-queue",
        max_age_seconds=1,
        clock=lambda: now[0],
    )
    daemon = PortalImplementationDaemon(
        task_source=Path(setup["database"].path),
        expected_task_source_identity=(
            setup["database"].pinned_identity
        ),
        state_path=runtime / f"{suffix}-state.json",
        strategy_path=runtime / f"{suffix}-strategy.json",
        events_path=runtime / f"{suffix}-events.jsonl",
        repo_root=Path(setup["daemon"].repo_root),
        merge_target_branch="main",
        worktree_pool_enabled=False,
        validation_cache_dir=runtime / f"{suffix}-validation-cache",
        merge_queue=queue,
        merge_queue_dir=runtime / "merge-queue",
    )
    return daemon, queue


def _assert_completed_authority(setup: dict[str, object]) -> None:
    queue = setup["queue"]
    request = queue.get(setup["request"].request_id)
    assert request is not None and request.status == "completed"
    completion = request.metadata["completion"]
    acceptance_path = (
        setup["runtime"]
        / "merge-queue/train/receipts"
        / (
            "acceptance-"
            + completion["acceptance_receipt_id"].removeprefix("sha256:")
            + ".json"
        )
    )
    acceptance = ParallelAcceptanceReceipt.from_dict(
        json.loads(acceptance_path.read_text(encoding="utf-8"))
    )
    evidence = PostMergeEvidenceReceipt.from_dict(
        acceptance.post_merge_validation["post_merge_evidence_receipt"]
    )
    assert acceptance.accepted is True
    assert evidence.accepted is True
    assert completion["post_merge_evidence_receipt_id"] == evidence.receipt_id
    task = setup["database"].get(setup["task"].task_id)
    assert task is not None and task.status == "completed"


def test_consume_one_persists_typed_merge_and_acceptance_authority(
    tmp_path: Path,
) -> None:
    now = [100.0]
    setup = _setup(tmp_path, now=now)

    result = setup["daemon"]._consume_one_merge_candidate()

    assert result is not None and result["accepted"] is True, result
    merge_payload = result["merge_result"]["merge_integrated_receipt"]
    integrated = DuckDBMergeIntegratedReceipt.from_dict(merge_payload)
    assert integrated.candidate_commit == setup["candidate"]
    queued = setup["queue"].get(setup["request"].request_id)
    validation_proof = queued.metadata["validation_proof"]
    assert set(integrated.validation_receipt_ids) == {
        validation_proof["post_merge_evidence_input"]["validation_receipt"][
            "receipt_id"
        ],
        validation_proof["validation_execution_receipt"]["receipt_id"],
    }
    receipt_path = setup["runtime"] / "merge-queue/train/receipts" / (
        "merge-integrated-"
        + integrated.receipt_id.removeprefix("sha256:")
        + ".json"
    )
    assert receipt_path.read_text(encoding="utf-8") == (
        json.dumps(
            integrated.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    _assert_completed_authority(setup)


def test_crash_after_git_merge_before_task_cas_replays_on_fresh_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    setup = _setup(tmp_path, now=now)
    daemon = setup["daemon"]

    def crash_before_task_cas(*_args, **_kwargs):
        raise KeyboardInterrupt("injected post-merge crash")

    monkeypatch.setattr(
        daemon,
        "_mark_task_completed_in_todo",
        crash_before_task_cas,
    )
    with pytest.raises(KeyboardInterrupt, match="post-merge"):
        daemon._consume_one_merge_candidate()

    task = setup["database"].get(setup["task"].task_id)
    assert task is not None and task.status != "completed"
    receipts = list(
        (setup["runtime"] / "merge-queue/train/receipts").glob(
            "merge-integrated-*.json"
        )
    )
    assert len(receipts) == 1
    DuckDBMergeIntegratedReceipt.load_file(receipts[0])

    now[0] += 2
    restart, queue = _restart(setup, now=now, suffix="pre-cas-restart")
    setup["daemon"] = restart
    setup["queue"] = queue
    replayed = restart._consume_one_merge_candidate()
    assert replayed is not None and replayed["accepted"] is True
    _assert_completed_authority(setup)


def test_evidence_failure_after_git_merge_replays_without_moving_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    setup = _setup(tmp_path, now=now)
    daemon = setup["daemon"]
    assemble = daemon._post_merge_evidence_for_integrated_claim

    def fail_evidence_once(*_args, **_kwargs):
        raise ValueError("injected evidence assembly failure")

    monkeypatch.setattr(
        daemon,
        "_post_merge_evidence_for_integrated_claim",
        fail_evidence_once,
    )
    failed = daemon._consume_one_merge_candidate()
    integrated_target = _git(tmp_path, "rev-parse", "main")

    assert failed is not None
    assert failed["status"] == "retrying"
    assert failed["integrated"] is False
    assert failed["merge_integrated"] is True
    assert failed["reason"] == "merge_integrated_evidence_pending"
    integrated = DuckDBMergeIntegratedReceipt.from_dict(
        failed["merge_integrated_receipt"]
    )
    assert integrated.merge_commit == integrated_target
    queued = setup["queue"].get(setup["request"].request_id)
    assert queued is not None and queued.status == "pending"
    task = setup["database"].get(setup["task"].task_id)
    assert task is not None and task.status != "completed"

    monkeypatch.setattr(
        daemon,
        "_post_merge_evidence_for_integrated_claim",
        assemble,
    )
    replayed = daemon._consume_one_merge_candidate()

    assert replayed is not None and replayed["accepted"] is True
    assert _git(tmp_path, "rev-parse", "main") == integrated_target
    _assert_completed_authority(setup)


def test_merge_integrated_receipt_refuses_symlink_artifact(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    receipt_dir = queue_dir / "train" / "receipts"
    receipt_dir.mkdir(parents=True)
    receipt = DuckDBMergeIntegratedReceipt(
        repository_id="repository:test",
        target_branch="main",
        request_id="request:test",
        task_id="FIX-001",
        task_cid="task-cid:test",
        task_source_identity_id="task-source:test",
        task_source_writer_id="writer:test",
        task_source_fencing_token=1,
        candidate_commit="1" * 40,
        candidate_tree="2" * 40,
        merge_commit="3" * 40,
        merge_tree="4" * 40,
        merge_parents=("5" * 40, "1" * 40),
        merge_consumer_id="consumer:test",
        lease_id="lease:test",
        fencing_token=1,
        validation_receipt_ids=("validation:test",),
        proposal_receipt_id="proposal:test",
    )
    outside = tmp_path / "outside.json"
    outside.write_text("do not replace\n", encoding="utf-8")
    target = receipt_dir / (
        "merge-integrated-"
        + receipt.receipt_id.removeprefix("sha256:")
        + ".json"
    )
    target.symlink_to(outside)

    with pytest.raises(ValueError, match="regular file"):
        PortalImplementationDaemon._persist_merge_integrated_receipt(
            queue_dir,
            receipt,
        )
    with pytest.raises(ValueError, match="regular file"):
        DuckDBMergeIntegratedReceipt.load_file(target)

    assert target.is_symlink()
    assert outside.read_text(encoding="utf-8") == "do not replace\n"


def test_typed_compound_candidate_fails_closed_before_git_merge(
    tmp_path: Path,
) -> None:
    now = [100.0]
    setup = _setup(
        tmp_path,
        now=now,
        changed_submodule_paths={"foreign-submodule"},
    )
    baseline = _git(tmp_path, "rev-parse", "main")

    result = setup["daemon"]._consume_one_merge_candidate()

    assert result is not None and result["accepted"] is False
    assert result["status"] == "quarantined"
    assert result["reason"] == "typed_compound_integration_receipt_set_missing"
    assert _git(tmp_path, "rev-parse", "main") == baseline
    assert not list(
        (setup["runtime"] / "merge-queue/train/receipts").glob(
            "merge-integrated-*.json"
        )
    )


def test_non_two_parent_integration_cannot_produce_typed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    setup = _setup(tmp_path, now=now)
    daemon = setup["daemon"]

    def fast_forward_instead_of_merge(*_args, **_kwargs):
        _git(tmp_path, "reset", "--hard", setup["candidate"])
        return {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": setup["candidate"],
            "target_commit": setup["candidate"],
        }

    monkeypatch.setattr(
        daemon,
        "_merge_branch_to_main",
        fast_forward_instead_of_merge,
    )
    result = daemon._consume_one_merge_candidate()

    assert result is not None and result["accepted"] is False
    assert result["reason"] == "merge_integrated_receipt_invalid"
    assert result["merge_result"]["merge_integrated"] is False
    assert "exact two-parent" in result["merge_result"][
        "merge_integrated_receipt_error"
    ]
    assert not list(
        (setup["runtime"] / "merge-queue/train/receipts").glob(
            "merge-integrated-*.json"
        )
    )


def test_crash_after_task_cas_before_queue_completion_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    setup = _setup(tmp_path, now=now)
    queue = setup["queue"]

    def crash_before_queue_completion(*_args, **_kwargs):
        raise KeyboardInterrupt("injected post-task-CAS crash")

    monkeypatch.setattr(queue, "complete", crash_before_queue_completion)
    with pytest.raises(KeyboardInterrupt, match="post-task-CAS"):
        setup["daemon"]._consume_one_merge_candidate()

    task = setup["database"].get(setup["task"].task_id)
    assert task is not None and task.status == "completed"
    assert list(
        (setup["runtime"] / "merge-queue/train/receipts").glob(
            "acceptance-*.json"
        )
    )

    now[0] += 2
    restart, restarted_queue = _restart(
        setup, now=now, suffix="post-cas-restart"
    )
    setup["daemon"] = restart
    setup["queue"] = restarted_queue
    replayed = restart._consume_one_merge_candidate()
    assert replayed is not None and replayed["accepted"] is True
    _assert_completed_authority(setup)
