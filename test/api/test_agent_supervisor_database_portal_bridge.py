"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_repository_id,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    WorktreeLifecycleStore,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as bridge_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    CROSS_ATTEMPT_LIFECYCLE_AUTHORITY_SCHEMA,
    CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME,
    DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2,
    DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA,
    DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA,
    DatabasePortalAttemptPaths,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    SEMANTIC_WRITER_POLICY_ENV,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    PortalTask,
    parse_args,
    parse_task_text,
    task_declared_output_paths,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)


def _attempt() -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id="attempt:001",
        claim_id="claim:001",
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        attempt_number=1,
        owner_session_id="session:bridge",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:001",
        committed_phase="claimed",
        status="running",
        started_at_ms=1,
    )


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        goal_cid="goal:inventory",
        plan_cid="plan:lgswf:1",
        revision=11,
        priority="P0",
        dependencies=("task:cid:003",),
        outputs=({"path": "inventory/result.json"},),
        validations=({"argv": ["python3", "-m", "pytest", "focused.py"]},),
        acceptance=({"criterion": "Focused validation passes"},),
        body={
            "objective": "Produce the current authority inventory",
            "completion": "auto",
            "track": "analysis",
            "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
            "write_scope": ["inventory/result.json"],
            "completion_contract": "Focused validation passes",
        },
    )


def _bridge_for_projection(tmp_path: Path) -> DatabasePortalExecutionBridge:
    return DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: object(),
    )


def _content_addressed_output(
    *,
    effect_id: str = "sha256:effect-001",
    declared_path: object = "inventory/result.json",
) -> dict[str, object]:
    return {
        "ordinal": 0,
        "path": effect_id,
        "effect": {
            "effect_id": effect_id,
            "declared_path": declared_path,
            "effect": "declared_output",
        },
    }


def test_merge_producer_records_exact_target_advanced_topology(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("branch", "-M", "main")
    git("config", "user.name", "Portal Test")
    git("config", "user.email", "portal@example.invalid")
    (repository / "seed.py").write_text("SEED = True\n", encoding="utf-8")
    git("add", "seed.py")
    git("commit", "-q", "-m", "seed")
    candidate_baseline = git("rev-parse", "HEAD")

    branch_name = "implementation/parallel-candidate"
    git("checkout", "-q", "-b", branch_name)
    (repository / "candidate.py").write_text(
        "CANDIDATE = True\n", encoding="utf-8"
    )
    git("add", "candidate.py")
    git("commit", "-q", "-m", "candidate")
    implementation_commit = git("rev-parse", "HEAD")

    git("checkout", "-q", "main")
    (repository / "concurrent.py").write_text(
        "CONCURRENT = True\n", encoding="utf-8"
    )
    git("add", "concurrent.py")
    git("commit", "-q", "-m", "concurrent integration")
    integration_base = git("rev-parse", "HEAD")

    state_dir = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=repository / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repository,
    )
    result = daemon._merge_branch_to_main(
        branch_name,
        PortalTask(
            task_id="PARALLEL-001",
            title="Integrate parallel candidate",
            status="todo",
            completion="manual",
            priority="P0",
            track="ops",
            outputs=("candidate.py",),
        ),
        1,
        baseline_ref=candidate_baseline,
    )

    assert result["merged"] is True
    assert result["candidate_baseline_ref"] == candidate_baseline
    assert result["integration_base_commit"] == integration_base
    merge_commit = str(result["merge_commit"])
    assert git("rev-list", "--parents", "-n", "1", merge_commit).split() == [
        merge_commit,
        integration_base,
        implementation_commit,
    ]

    proof = daemon._immutable_integration_commit(
        result,
        implementation_commit=implementation_commit,
        target_branch="main",
    )
    assert proof["passed"] is True
    assert proof["candidate_baseline_ref"] == candidate_baseline
    assert proof["integration_base_commit"] == integration_base
    assert proof["exact_two_parent_merge"] is True

    forged = {**result, "integration_base_commit": candidate_baseline}
    rejected = daemon._immutable_integration_commit(
        forged,
        implementation_commit=implementation_commit,
        target_branch="main",
    )
    assert rejected["passed"] is False
    assert "integration_commit_not_exact_two_parent_merge" in rejected["reasons"]


@pytest.mark.parametrize("target_advanced", (False, True))
def test_source_transition_binds_attempt_board_repository_and_exact_merge(
    tmp_path: Path,
    target_advanced: bool,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("branch", "-M", "main")
    (repository / "seed.py").write_text("SEED = True\n", encoding="utf-8")
    git("add", "seed.py")
    git(
        "-c",
        "user.name=Portal Test",
        "-c",
        "user.email=portal@example.invalid",
        "commit",
        "-q",
        "-m",
        "seed",
    )
    baseline = git("rev-parse", "HEAD")
    git("checkout", "-q", "-b", "implementation/test")
    (repository / "accepted.py").write_text("ACCEPTED = True\n", encoding="utf-8")
    git("add", "accepted.py")
    git(
        "-c",
        "user.name=Portal Test",
        "-c",
        "user.email=portal@example.invalid",
        "commit",
        "-q",
        "-m",
        "implementation",
    )
    implementation = git("rev-parse", "HEAD")
    git("checkout", "-q", "main")
    if target_advanced:
        (repository / "concurrent.py").write_text(
            "CONCURRENT = True\n", encoding="utf-8"
        )
        git("add", "concurrent.py")
        git(
            "-c",
            "user.name=Portal Test",
            "-c",
            "user.email=portal@example.invalid",
            "commit",
            "-q",
            "-m",
            "concurrent integration",
        )
    integration_base = git("rev-parse", "HEAD")
    git(
        "-c",
        "user.name=Portal Test",
        "-c",
        "user.email=portal@example.invalid",
        "merge",
        "--no-ff",
        "-q",
        "-m",
        "accept",
        implementation,
    )
    merge_commit = git("rev-parse", "HEAD")
    proof = {
        "passed": True,
        "implementation_commit": implementation,
        "integration_commit": merge_commit,
        "integration_ref": merge_commit,
        "target_branch": "main",
    }
    if target_advanced:
        proof.update(
            {
                "candidate_baseline_ref": baseline,
                "integration_base_commit": integration_base,
                "exact_two_parent_merge": True,
            }
        )
    invariant = {"passed": True, "repository_ref": merge_commit}
    identity_bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "identity-attempt",
        portal_factory=lambda _paths, _alias: object(),
        board_namespace="test-board-v1",
        task_header_prefix="## LGSWF-",
    )
    identity_projection = identity_bridge._render_projection(
        _attempt(), _record()
    )
    identity_task = parse_task_text(
        identity_projection,
        path=tmp_path / "identity-task.md",
        task_header_prefix="## LGSWF-",
    )[0]
    identity = canonical_task_identity(
        {
            "task_id": identity_task.task_id,
            "title": identity_task.title,
            "outputs": task_declared_output_paths(identity_task),
            "acceptance": identity_task.acceptance,
            "metadata": dict(identity_task.metadata),
        },
        board_namespace="test-board-v1",
        source_path=tmp_path / "identity-task.md",
    )
    canonical_task_cid = identity.canonical_task_cid
    canonical_task_key = identity.canonical_task_key
    event = {
        "type": "implementation_finished",
        "returncode": 0,
        "task_id": "LGSWF-004",
        "attempt": 1,
        "board_namespace": "test-board-v1",
        "board_completion": {
            "complete": True,
            "pending_merge": False,
            "reason": "merged_into_target",
        },
        "baseline_ref": baseline,
        "implementation_commit": implementation,
        "canonical_task_cid": canonical_task_cid,
        "canonical_task_key": canonical_task_key,
        "merge_result": {
            "merged": True,
            "returncode": 0,
            "request_id": "request:test:1",
            "baseline_ref": baseline,
            "implementation_commit": implementation,
            "merge_commit": merge_commit,
            "target_branch": "main",
            "target_repository_id": checkout_repository_id(repository),
            "canonical_task_cid": canonical_task_cid,
            "canonical_task_key": canonical_task_key,
            "integration_commit_proof": proof,
            "post_merge_declared_output_invariant": invariant,
        },
    }
    if target_advanced:
        event["merge_result"].update(
            {
                "candidate_baseline_ref": baseline,
                "integration_base_commit": integration_base,
            }
        )
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    events = attempt_root / "events.jsonl"
    events.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    paths = DatabasePortalAttemptPaths(
        root=attempt_root,
        task_projection=attempt_root / "task.md",
        binding=attempt_root / "binding.json",
        state=attempt_root / "state.json",
        strategy=attempt_root / "strategy.json",
        events=events,
        implementation_logs=attempt_root / "logs",
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: object(),
        repo_root=repository,
        board_namespace="test-board-v1",
        configured_board_admission_cid="baguqeera" + "b" * 48,
        merge_target_branch="main",
        task_header_prefix="## LGSWF-",
    )
    projection_seed = bridge._render_projection(_attempt(), _record())
    binding = bridge._binding(_attempt(), _record(), projection_seed)
    paths.binding.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.task_projection.write_text(
        projection_seed.replace("- Status: ready", "- Status: completed"),
        encoding="utf-8",
    )
    request = {
        "request_id": "request:test:1",
        "branch_name": "implementation/test",
        "task_id": "LGSWF-004",
        "priority": "P0",
        "lane_id": "lane:test",
        "enqueued_at": 1.0,
        "attempt": 1,
        "metadata": {
            "baseline_ref": baseline,
            "implementation_commit": implementation,
            "target_binding_schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            ),
            "target_repository_id": checkout_repository_id(repository),
            "target_branch": "main",
            "repo_root": str(repository),
            "completion_task_cids": {
                "LGSWF-004": canonical_task_cid,
            },
            "task": {
                "task_id": "LGSWF-004",
                "board_namespace": "test-board-v1",
                "canonical_task_cid": canonical_task_cid,
                "canonical_task_key": canonical_task_key,
                "metadata": {
                    "database attempt id": _attempt().attempt_id,
                    "database claim id": _attempt().claim_id,
                    "database task cid": _attempt().task_cid,
                },
            },
        },
        "commit_sha": implementation,
        "canonical_task_id": canonical_task_cid,
        "canonical_task_key": canonical_task_key,
        "status": "completed",
        "claimed_at": 1.0,
        "consumer_id": "consumer:test",
        "failure_count": 0,
        "failure_reason": "",
        "claim_token": "claim:test",
        "claim_generation": 1,
        "retry_not_before": 0.0,
        "dedupe_key": "d" * 64,
    }

    # A local replace ref must never alter the merge topology admitted by the
    # bridge.  The production verifier uses both --no-replace-objects and the
    # matching environment guard.
    git("replace", merge_commit, baseline)

    transition = bridge._accepted_source_transition(
        attempt=_attempt(),
        paths=paths,
        binding=binding,
        task_alias="LGSWF-004",
        task_cid="task:cid:004",
        merge_request_loader=lambda request_id: (
            request if request_id == request["request_id"] else None
        ),
    )

    assert transition is not None
    assert transition["schema"] == (
        DATABASE_PORTAL_TARGET_ADVANCED_SOURCE_TRANSITION_SCHEMA
        if target_advanced
        else DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA
    )
    if target_advanced:
        assert transition["candidate_baseline_ref"] == baseline
        assert transition["integration_base_commit"] == integration_base
        assert "baseline_ref" not in transition
        assert transition["integration_commit_proof"][
            "exact_two_parent_merge"
        ] is True
    else:
        assert transition["baseline_ref"] == baseline
        assert "candidate_baseline_ref" not in transition
    assert transition["implementation_commit"] == implementation
    assert transition["merge_commit"] == merge_commit
    assert transition["target_repository_id"] == checkout_repository_id(repository)
    assert transition["task_completion_authority"] is False
    assert transition["worker_self_approval"] is False
    assert str(transition["transition_cid"]).startswith("sha256:")

    transition_evidence = {"accepted_source_transition": transition}
    transition_receipt = {
        "schema": (
            DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
            if target_advanced
            else DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2
        ),
        "interface": bridge.INTERFACE,
        "status": "succeeded",
        "provider": "PortalImplementationDaemon",
        "accepted": True,
        "task_cid": _attempt().task_cid,
        "attempt_id": _attempt().attempt_id,
        "evidence_digest": bridge_module._sha256_bytes(
            bridge_module._canonical_json(transition_evidence)
        ),
        "portal_evidence": transition_evidence,
        "accepted_source_transition": transition,
    }
    transition_receipt["receipt_id"] = bridge_module._sha256_bytes(
        bridge_module._canonical_json(transition_receipt)
    )
    assert bridge.apply_effect(_attempt(), transition_receipt)["status"] == "applied"
    if target_advanced:
        forged_v2 = {**transition_receipt, "schema": DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2}
        forged_v2.pop("receipt_id")
        forged_v2["receipt_id"] = bridge_module._sha256_bytes(
            bridge_module._canonical_json(forged_v2)
        )
        with pytest.raises(DatabasePortalBridgeError, match="unbound source transition"):
            bridge.apply_effect(_attempt(), forged_v2)

    expected_diff = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            integration_base,
            merge_commit,
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    assert transition["changed_path_diff_sha256"] == bridge_module._sha256_bytes(
        expected_diff
    )

    if target_advanced:
        forged_event = json.loads(json.dumps(event))
        forged_event["merge_result"]["integration_base_commit"] = baseline
        events.write_text(
            json.dumps(forged_event, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DatabasePortalBridgeError, match="exact Git merge"):
            bridge._accepted_source_transition(
                attempt=_attempt(),
                paths=paths,
                binding=binding,
                task_alias="LGSWF-004",
                task_cid="task:cid:004",
                merge_request_loader=lambda request_id: (
                    request if request_id == request["request_id"] else None
                ),
            )
        events.write_text(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    for field, replacement in (
        ("canonical_task_cid", None),
        ("canonical_task_cid", "baguqeeraforeign"),
        ("canonical_task_key", None),
        ("canonical_task_key", "task:foreign"),
    ):
        inconsistent_event = json.loads(json.dumps(event))
        if replacement is None:
            inconsistent_event["merge_result"].pop(field)
        else:
            inconsistent_event["merge_result"][field] = replacement
        events.write_text(
            json.dumps(
                inconsistent_event,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DatabasePortalBridgeError, match="inconsistent"):
            bridge._accepted_source_transition(
                attempt=_attempt(),
                paths=paths,
                binding=binding,
                task_alias="LGSWF-004",
                task_cid="task:cid:004",
                merge_request_loader=lambda request_id: (
                    request if request_id == request["request_id"] else None
                ),
            )
    events.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    event["target_repository_id"] = "repository:foreign"
    events.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatabasePortalBridgeError, match="inconsistent"):
        bridge._accepted_source_transition(
            attempt=_attempt(),
            paths=paths,
            binding=binding,
            task_alias="LGSWF-004",
            task_cid="task:cid:004",
            merge_request_loader=lambda request_id: (
                request if request_id == request["request_id"] else None
            ),
        )
    event.pop("target_repository_id")
    events.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    forged_request = json.loads(json.dumps(request))
    forged_request["task_id"] = "LGSWF-OTHER"
    forged_request["metadata"]["task"]["task_id"] = "LGSWF-OTHER"
    with pytest.raises(DatabasePortalBridgeError, match="merge request is inconsistent"):
        bridge._accepted_source_transition(
            attempt=_attempt(),
            paths=paths,
            binding=binding,
            task_alias="LGSWF-004",
            task_cid="task:cid:004",
            merge_request_loader=lambda request_id: (
                forged_request if request_id == forged_request["request_id"] else None
            ),
        )

    event["board_namespace"] = "foreign-board"
    events.write_text(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatabasePortalBridgeError, match="inconsistent"):
        bridge._accepted_source_transition(
            attempt=_attempt(),
            paths=paths,
            binding=binding,
            task_alias="LGSWF-004",
            task_cid="task:cid:004",
            merge_request_loader=lambda request_id: (
                request if request_id == request["request_id"] else None
            ),
        )


def test_finished_event_projection_emits_nested_merge_repository_binding() -> None:
    payload = {
        "task_id": "LGSWF-004",
        "merge_result": {
            "target_repository_id": "repository:exact",
        },
    }

    projected = PortalImplementationDaemon._implementation_finished_event_payload(
        payload
    )

    assert "target_repository_id" not in payload
    assert projected["target_repository_id"] == "repository:exact"
    assert projected["merge_result"] == payload["merge_result"]


def test_source_transition_admits_exact_queued_reconciliation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: object(),
        repo_root=repository,
        board_namespace="test-board-v1",
        configured_board_admission_cid="baguqeera" + "b" * 48,
        merge_target_branch="main",
        task_header_prefix="## LGSWF-",
    )
    paths = bridge._paths(_attempt())
    paths.root.mkdir(parents=True)
    projection_seed = bridge._render_projection(_attempt(), _record())
    binding = bridge._binding(_attempt(), _record(), projection_seed)
    paths.binding.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.task_projection.write_text(
        projection_seed.replace("- Status: ready", "- Status: completed"),
        encoding="utf-8",
    )
    parsed_task = parse_task_text(
        projection_seed,
        path=paths.task_projection,
        task_header_prefix="## LGSWF-",
    )[0]
    identity = canonical_task_identity(
        {
            "task_id": parsed_task.task_id,
            "title": parsed_task.title,
            "outputs": task_declared_output_paths(parsed_task),
            "acceptance": parsed_task.acceptance,
            "metadata": dict(parsed_task.metadata),
        },
        board_namespace="test-board-v1",
        source_path=paths.task_projection,
    )
    baseline = "a" * 40
    implementation = "b" * 40
    merge_commit = "c" * 40
    repository_id = "repository:exact"
    request_id = "request:queued:1"
    runtime_binding = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "ignored_runtime_taskboard_binding@1"
        ),
        "passed": True,
        "authoritative": True,
        "ignored": True,
        "runtime_projection": True,
        "reason": "ignored_runtime_taskboard_bound",
        "path": str(paths.task_projection),
        "repo": str(repository),
        "relative_path": paths.task_projection.name,
    }
    snapshot = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "fsynced_taskboard_completion_snapshot@1"
        ),
        "passed": True,
        "reason": "fsynced_taskboard_completion_proven",
        "path": str(paths.task_projection),
        "expected_task_ids": ["LGSWF-004"],
        "runtime_projection": True,
        "runtime_binding": runtime_binding,
        "observed_statuses": {"LGSWF-004": "completed"},
        "observed_task_cids": {
            "LGSWF-004": identity.canonical_task_cid,
        },
        "missing_task_ids": [],
        "ambiguous_task_ids": [],
        "status_mismatches": {},
        "task_cid_mismatches": {},
    }
    queued_event = {
        "type": "implementation_finished",
        "event_id": "sha256:" + "1" * 64,
        "returncode": 0,
        "task_id": "LGSWF-004",
        "attempt": 1,
        "board_namespace": "test-board-v1",
        "board_completion": {
            "complete": False,
            "pending_merge": True,
            "reason": "merge_queued_awaiting_integration",
        },
        "baseline_ref": baseline,
        "branch": "implementation/test",
        "implementation_commit": implementation,
        "canonical_task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key,
        "target_repository_id": repository_id,
        "merge_result": {
            "attempted": False,
            "merged": False,
            "queued": True,
            "reason": "merge_queued",
            "request_id": request_id,
            "target_branch": "main",
            "target_repository_id": repository_id,
            "canonical_task_cid": identity.canonical_task_cid,
            "canonical_task_key": identity.canonical_task_key,
        },
    }
    proof = {
        "passed": True,
        "implementation_commit": implementation,
        "integration_commit": merge_commit,
        "integration_ref": merge_commit,
        "target_branch": "main",
    }
    reconciliation = {
        "type": "merge_reconciled",
        "event_id": "sha256:" + "2" * 64,
        "task_id": "LGSWF-004",
        "attempt": 1,
        "board_namespace": "test-board-v1",
        "canonical_task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key,
        "implementation_commit": implementation,
        "completion_task_cids": {
            "LGSWF-004": identity.canonical_task_cid,
        },
        "resolved": True,
        "reason": "merge_retried",
        "merge_result": {
            "attempted": True,
            "merged": True,
            "returncode": 0,
            "merge_commit": merge_commit,
            "target_branch": "main",
        },
        "integration_commit_proof": proof,
        "post_merge_declared_output_invariant": {
            "passed": True,
            "repository_ref": merge_commit,
        },
        "completion_persistence": {
            "passed": True,
            "reason": "completion_persisted",
            "expected_task_ids": ["LGSWF-004"],
            "completed_task_ids": ["LGSWF-004"],
            "missing_task_ids": [],
            "receipt_mismatches": {},
            "durable_update": True,
            "status_persisted": True,
            "runtime_taskboard_binding": runtime_binding,
            "fsynced_taskboard_snapshot": snapshot,
        },
    }
    paths.events.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in (queued_event, reconciliation)
        ),
        encoding="utf-8",
    )
    request = {
        "request_id": request_id,
        "branch_name": "implementation/test",
        "task_id": "LGSWF-004",
        "attempt": 3,
        "metadata": {
            "baseline_ref": baseline,
            "implementation_commit": implementation,
            "target_binding_schema": (
                "ipfs_accelerate_py/agent-supervisor/merge-target-binding@1"
            ),
            "target_repository_id": repository_id,
            "target_branch": "main",
            "repo_root": str(repository),
            "completion_task_cids": {
                "LGSWF-004": identity.canonical_task_cid,
            },
            "task": {
                "task_id": "LGSWF-004",
                "board_namespace": "test-board-v1",
                "canonical_task_cid": identity.canonical_task_cid,
                "canonical_task_key": identity.canonical_task_key,
                "metadata": {
                    "database attempt id": _attempt().attempt_id,
                    "database claim id": _attempt().claim_id,
                    "database task cid": _attempt().task_cid,
                },
            },
            "cancellation": {
                "at": 4.0,
                "reason": "stale_quarantined_merge",
            },
        },
        "commit_sha": implementation,
        "canonical_task_id": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key,
        "status": "cancelled",
        "dedupe_key": "d" * 64,
    }

    def fake_git(command: list[str], **_kwargs: object) -> SimpleNamespace:
        arguments = command[2:]
        if arguments[:4] == ["rev-list", "--parents", "-n", "1"]:
            stdout = f"{merge_commit} {baseline} {implementation}\n".encode()
        elif arguments == ["rev-parse", f"{implementation}^{{tree}}"]:
            stdout = ("d" * 40 + "\n").encode()
        elif arguments == ["rev-parse", f"{merge_commit}^{{tree}}"]:
            stdout = ("e" * 40 + "\n").encode()
        elif arguments[:3] == [
            "diff-tree",
            "--no-commit-id",
            "--name-status",
        ]:
            stdout = b"A\0accepted.py\0"
        else:
            raise AssertionError(arguments)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(
        bridge_module,
        "checkout_repository_id",
        lambda _path: repository_id,
    )
    monkeypatch.setattr(bridge_module.subprocess, "run", fake_git)

    transition = bridge._accepted_source_transition(
        attempt=_attempt(),
        paths=paths,
        binding=binding,
        task_alias="LGSWF-004",
        task_cid="task:cid:004",
        merge_request_loader=lambda candidate: (
            request if candidate == request_id else None
        ),
    )

    assert transition is not None
    assert (
        transition["schema"]
        == DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA
    )
    assert transition["source_event_mode"] == "queued_merge_reconciliation"
    assert transition["merge_queue_terminal_status"] == "cancelled"
    assert transition["merge_queue_attempt"] == 3
    assert transition["merge_commit"] == merge_commit
    assert transition["worker_self_approval"] is False
    evidence = {"accepted_source_transition": transition}
    provider_receipt = {
        "schema": DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V2,
        "interface": bridge.INTERFACE,
        "status": "succeeded",
        "provider": "PortalImplementationDaemon",
        "accepted": True,
        "task_cid": _attempt().task_cid,
        "attempt_id": _attempt().attempt_id,
        "evidence_digest": bridge_module._sha256_bytes(
            bridge_module._canonical_json(evidence)
        ),
        "portal_evidence": evidence,
        "accepted_source_transition": transition,
    }
    provider_receipt["receipt_id"] = bridge_module._sha256_bytes(
        bridge_module._canonical_json(provider_receipt)
    )
    assert bridge.apply_effect(_attempt(), provider_receipt)["status"] == "applied"

    for field, replacement in (
        ("canonical_task_cid", None),
        ("canonical_task_cid", "baguqeeraforeign"),
        ("canonical_task_key", None),
        ("canonical_task_key", "task:foreign"),
    ):
        inconsistent_event = json.loads(json.dumps(queued_event))
        if replacement is None:
            inconsistent_event["merge_result"].pop(field)
        else:
            inconsistent_event["merge_result"][field] = replacement
        paths.events.write_text(
            "".join(
                json.dumps(event, sort_keys=True, separators=(",", ":"))
                + "\n"
                for event in (inconsistent_event, reconciliation)
            ),
            encoding="utf-8",
        )
        with pytest.raises(DatabasePortalBridgeError, match="inconsistent"):
            bridge._accepted_source_transition(
                attempt=_attempt(),
                paths=paths,
                binding=binding,
                task_alias="LGSWF-004",
                task_cid="task:cid:004",
                merge_request_loader=lambda _candidate: request,
            )

    reconciliation["completion_persistence"]["durable_update"] = False
    paths.events.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in (queued_event, reconciliation)
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="reconciled-source completion is inconsistent",
    ):
        bridge._accepted_source_transition(
            attempt=_attempt(),
            paths=paths,
            binding=binding,
            task_alias="LGSWF-004",
            task_cid="task:cid:004",
            merge_request_loader=lambda _candidate: request,
        )


def test_projection_prefers_exact_nested_declared_output_path(
    tmp_path: Path,
) -> None:
    record = _record()
    record.outputs = (_content_addressed_output(),)

    projection = _bridge_for_projection(tmp_path)._render_projection(
        _attempt(), record
    )

    assert "- Outputs: inventory/result.json" in projection
    assert "- Outputs: sha256:effect-001" not in projection
    tasks = parse_task_text(
        projection,
        path=tmp_path / "task-projection.md",
        task_header_prefix="## LGSWF-",
    )
    assert len(tasks) == 1
    assert task_declared_output_paths(tasks[0]) == ("inventory/result.json",)


def test_projection_preserves_legacy_outer_output_path(tmp_path: Path) -> None:
    projection = _bridge_for_projection(tmp_path)._render_projection(
        _attempt(), _record()
    )

    assert "- Outputs: inventory/result.json" in projection


@pytest.mark.parametrize(
    "effect",
    (
        {
            "path": "outputs/LGSWF-006.json",
            "effect_id": "effect:LGSWF-006",
        },
        {
            "path": "outputs/LGSWF-006.json",
            "effect_id": "effect:LGSWF-006",
            "effect": "create",
        },
    ),
)
def test_projection_preserves_generic_nested_effect_id_output(
    tmp_path: Path,
    effect: dict[str, object],
) -> None:
    record = _record()
    record.outputs = (
        {
            "ordinal": 0,
            "path": "outputs/LGSWF-006.json",
            "effect": effect,
        },
    )

    projection = _bridge_for_projection(tmp_path)._render_projection(
        _attempt(), record
    )

    assert "- Outputs: outputs/LGSWF-006.json" in projection
    tasks = parse_task_text(
        projection,
        path=tmp_path / "task-projection.md",
        task_header_prefix="## LGSWF-",
    )
    assert len(tasks) == 1
    assert task_declared_output_paths(tasks[0]) == (
        "outputs/LGSWF-006.json",
    )


def test_projection_rejects_conflicting_legacy_outer_paths(tmp_path: Path) -> None:
    record = _record()
    record.outputs = (
        {"path": "inventory/result.json", "output": "inventory/other.json"},
    )

    with pytest.raises(DatabasePortalBridgeError, match="declarations conflict"):
        _bridge_for_projection(tmp_path)._render_projection(_attempt(), record)


@pytest.mark.parametrize(
    "output",
    [
        {
            **_content_addressed_output(),
            "output": "sha256:conflicting-outer-id",
        },
        {
            **_content_addressed_output(),
            "path": "sha256:conflicting-outer-id",
        },
        {
            **_content_addressed_output(),
            "effect": {
                "declared_path": "inventory/result.json",
                "effect": "declared_output",
            },
        },
        {
            **_content_addressed_output(),
            "effect": {
                "effect_id": "sha256:effect-001",
                "declared_path": "inventory/result.json",
                "effect": "declared_output",
                "authority": True,
            },
        },
        {
            **_content_addressed_output(),
            "effect": {
                "effect_id": "sha256:effect-001",
                "declared_path": "inventory/result.json",
                "effect": "write",
            },
        },
        {
            "effect_id": "sha256:effect-001",
            "declared_path": "inventory/result.json",
            "effect": "declared_output",
        },
        _content_addressed_output(declared_path=None),
        _content_addressed_output(declared_path="../inventory/result.json"),
        _content_addressed_output(declared_path="/inventory/result.json"),
        _content_addressed_output(declared_path="inventory\\result.json"),
        _content_addressed_output(declared_path="inventory//result.json"),
        _content_addressed_output(declared_path="inventory/result,other.json"),
        _content_addressed_output(declared_path="inventory/result.json\x85## SAWM-X"),
        _content_addressed_output(declared_path="inventory/result.json\u2028## SAWM-X"),
        _content_addressed_output(declared_path="inventory/result.json\u2029## SAWM-X"),
        _content_addressed_output(declared_path="none"),
        _content_addressed_output(declared_path="N/A"),
    ],
    ids=(
        "conflicting-outer-aliases",
        "effect-id-mismatch",
        "missing-effect-id",
        "open-effect-record",
        "wrong-effect-kind",
        "non-nested-declaration",
        "non-string-declared-path",
        "parent-traversal",
        "absolute-path",
        "backslash-path",
        "non-canonical-path",
        "projection-delimiter",
        "next-line-control",
        "unicode-line-separator",
        "unicode-paragraph-separator",
        "portal-none-sentinel",
        "portal-na-sentinel",
    ),
)
def test_projection_rejects_malformed_or_ambiguous_declared_output(
    tmp_path: Path,
    output: dict[str, object],
) -> None:
    record = _record()
    record.outputs = (output,)

    with pytest.raises(DatabasePortalBridgeError, match="declared-output"):
        _bridge_for_projection(tmp_path)._render_projection(_attempt(), record)


def test_datasets_authority_marker_reaches_provider_without_state_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        json.dumps(
            {
                "authority_mode": "legacy_markdown",
                "task_source_kind": "legacy-markdown",
                "failover_policy": "fail_closed",
                "explicit_legacy": True,
                "credential": "must-not-propagate",
            },
            sort_keys=True,
        ),
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "secret-token")
    portal = SimpleNamespace(
        _canonical_ref=lambda task: "task:cid:004",
        _implementation_untrusted_process_environment=(
            PortalImplementationDaemon._implementation_untrusted_process_environment
        ),
    )
    task = SimpleNamespace(task_id="LGSWF-004")

    environment = PortalImplementationDaemon._implementation_process_environment(
        portal,
        task,
        attempt=2,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert environment[SEMANTIC_TRUTH_AUTHORITY_ENV] == "ipfs_datasets_py"
    assert environment[SEMANTIC_WRITER_POLICY_ENV] == "reference_only"
    assert "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION" not in environment
    assert "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON" not in environment
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION", "schema-v1")
    ordinary_environment = (
        PortalImplementationDaemon._implementation_process_environment(
            portal,
            task,
            attempt=3,
            checkpoint_dir=tmp_path / "ordinary-checkpoint",
        )
    )
    assert SEMANTIC_TRUTH_AUTHORITY_ENV not in ordinary_environment
    assert SEMANTIC_WRITER_POLICY_ENV not in ordinary_environment


class _TaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return self.record if task_cid == "task:cid:004" else None


class _CompletingPortal:
    def __init__(self, paths: object, task_alias: str) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.closed = False

    def run_once(self) -> dict[str, object]:
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_commit": "a" * 40,
                    "last_merge_returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        self.paths.events.write_text(
            json.dumps(
                {
                    "type": "task_completed",
                    "task_id": self.task_alias,
                    "event_id": "event:complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "task_count": 1,
            "completed_count": 1,
            "active_task_id": self.task_alias,
            "implementation_result": {
                "task_id": self.task_alias,
                "returncode": 0,
                "implementation_commit": "a" * 40,
                # Raw model output must not enter the database receipt.
                "model_response": "private provider payload",
            },
            "merge_reconciliation": [
                {
                    "task_id": self.task_alias,
                    "returncode": 0,
                    "merge_commit": "b" * 40,
                    "provider_payload": "private",
                }
            ],
        }

    def close_event_runtime(self) -> None:
        self.closed = True


def _cross_attempt_recovery_fixture(
    tmp_path: Path,
    *,
    authority_allowed: bool = True,
    prior_task_revision: int = 10,
) -> tuple[
    DatabasePortalExecutionBridge,
    DatabaseTaskAttempt,
    WorktreeLifecycleStore,
    Path,
    list[object],
]:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str, cwd: Path = repository) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("branch", "-M", "main")
    git("config", "user.name", "Portal Test")
    git("config", "user.email", "portal@example.invalid")
    (repository / "seed.py").write_text("SEED = True\n", encoding="utf-8")
    git("add", "seed.py")
    git("commit", "-q", "-m", "seed")

    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    workspace = worktree_root / "prior"
    prior_branch = "implementation/lgswf-004-attempt-1"
    git("branch", prior_branch)
    git("worktree", "add", "-q", str(workspace), prior_branch)

    current_record = _record()
    prior_record = _record()
    prior_record.revision = prior_task_revision
    task_source = _TaskSource(current_record)
    attempt_root = tmp_path / "attempts"
    current_attempt = replace(_attempt(), attempt_number=2)
    prior_attempt = DatabaseTaskAttempt(
        attempt_id="attempt:prior",
        claim_id="claim:prior",
        task_cid=current_attempt.task_cid,
        task_alias=current_attempt.task_alias,
        attempt_number=1,
        owner_session_id="session:prior",
        fencing_token=6,
        fence_epoch=2,
        lease_id="lease:prior",
        committed_phase="failed",
        status="failed",
        started_at_ms=1,
    )
    projection_bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: object(),
        task_header_prefix="## LGSWF-",
    )
    prior_paths = projection_bridge._paths(prior_attempt)
    prior_seed = projection_bridge._render_projection(
        prior_attempt,
        prior_record,
    )
    prior_binding = projection_bridge._binding(
        prior_attempt,
        prior_record,
        prior_seed,
    )
    prior_paths.root.mkdir(parents=True)
    prior_paths.task_projection.write_text(prior_seed, encoding="utf-8")
    prior_paths.binding.write_text(
        json.dumps(prior_binding, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prior_identity = projection_bridge._prior_projection_identity(
        prior_paths,
        prior_binding,
    )
    prior_paths.state.write_text(
        json.dumps(
            {
                "implementation_in_progress": True,
                "active_task_id": prior_identity["task_id"],
                "active_task_cid": prior_identity["canonical_task_cid"],
                "active_task_key": prior_identity["canonical_task_key"],
                "active_attempt": 1,
                "active_worktree_path": str(workspace.resolve()),
                "active_branch": prior_branch,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    store = WorktreeLifecycleStore(
        repo_root=repository,
        proc_root=proc_root,
    )
    lifecycle = store.begin_preparing(
        task_id=current_attempt.task_alias,
        canonical_task_cid=prior_identity["canonical_task_cid"],
        attempt=1,
        lane_id=f"{prior_paths.root.resolve()}:lane-1",
        workspace_path=workspace,
        branch=prior_branch,
        merge_target="main",
        state_dir=str(prior_paths.root.resolve()),
        owner=ProcessBirthIdentity(
            pid=999_999,
            start_time_ticks=1,
            boot_id="test-boot",
            parent_pid=1,
        ),
    )
    lifecycle = store.mark_active(
        workspace,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    store.mark_settling(
        workspace,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )

    portals: list[object] = []

    class RecoveryPortal(_CompletingPortal):
        def __init__(self, paths: object, task_alias: str) -> None:
            super().__init__(paths, task_alias)
            self.worktree_lifecycle = store
            self.worktree_root = worktree_root
            self.run_count = 0
            self.checkout_lease_released = False

        @staticmethod
        def _list_process_commands() -> list[str]:
            return []

        @staticmethod
        def _docker_isolation_active_for_worktree(_path: str) -> bool:
            return False

        @staticmethod
        def _lock_owner_is_active(
            _metadata: object,
            *,
            expected_kind: str,
        ) -> bool:
            assert expected_kind == "implementation"
            return False

        @staticmethod
        def _acquire_checkout_mutation_lease(**_kwargs: object) -> tuple[
            object,
            str,
            None,
            float,
        ]:
            return object(), "acquired", None, 0.0

        def _release_checkout_mutation_lease(self, _lease: object) -> bool:
            self.checkout_lease_released = True
            return True

        def run_once(self) -> dict[str, object]:
            self.run_count += 1
            return super().run_once()

    def factory(paths: object, alias: str) -> RecoveryPortal:
        portal = RecoveryPortal(paths, alias)
        portals.append(portal)
        return portal

    def authority(
        _attempt_value: object,
        current_binding: object,
        old_binding: object,
    ) -> dict[str, object]:
        assert isinstance(current_binding, dict)
        assert isinstance(old_binding, dict)
        return {
            "schema": CROSS_ATTEMPT_LIFECYCLE_AUTHORITY_SCHEMA,
            "authorized": authority_allowed,
            "task_cid": current_binding["task_cid"],
            "task_alias": current_binding["task_alias"],
            "current_attempt_id": current_binding["attempt_id"],
            "prior_attempt_id": old_binding["attempt_id"],
            "current_attempt_number": current_attempt.attempt_number,
            "prior_attempt_number": prior_attempt.attempt_number,
            "current_binding_id": current_binding["binding_id"],
            "prior_binding_id": old_binding["binding_id"],
            "current_fencing_token": current_binding["fencing_token"],
            "prior_fencing_token": old_binding["fencing_token"],
            "current_control_binding_id": "sha256:control-current",
            "prior_control_binding_id": "sha256:control-prior",
            "current_control_task_projection_cid": (
                "sha256:projection-current"
            ),
            "prior_control_task_projection_cid": "sha256:projection-prior",
            "current_control_expected_revision": current_binding[
                "task_revision"
            ],
            "prior_control_expected_revision": old_binding["task_revision"],
            "prior_execution_status": "failed",
            "prior_claim_state": "expired",
            "prior_coordination_status": "failed",
            "legacy_current_binding": True,
            "legacy_prior_binding": True,
            "mutation_authority": False,
            "completion_authority": False,
        }

    bridge = DatabasePortalExecutionBridge(
        task_source=task_source,
        attempt_root=attempt_root,
        portal_factory=factory,
        repo_root=repository,
        board_namespace="test-board-v1",
        merge_target_branch="main",
        task_header_prefix="## LGSWF-",
        prior_attempt_authority=authority,
    )
    return bridge, current_attempt, store, workspace, portals


@pytest.mark.parametrize("prior_task_revision", [10, 11])
def test_bridge_exactly_retires_preserved_superseded_attempt_lifecycle(
    tmp_path: Path,
    prior_task_revision: int,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            prior_task_revision=prior_task_revision,
        )
    )

    provider = bridge.run_provider(attempt)

    assert provider["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None
    assert terminal.is_terminal
    assert terminal.terminal_reason == "superseded_database_attempt_preserved"
    receipt_path = (
        bridge._paths(attempt).root
        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["phase"] == "committed"
    assert receipt["worktree_deleted"] is False
    assert receipt["provider_dispatched"] is False
    assert receipt["task_completion_authority"] is False
    assert receipt["preservation"]["preservation_mode"] == (
        "clean_branch_commit"
    )
    assert "lease_id" not in json.dumps(receipt)
    assert workspace.is_dir()
    assert portals and portals[0].run_count == 1
    assert portals[0].checkout_lease_released is True


def test_bridge_preserves_lifecycle_when_database_authority_rejects(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            authority_allowed=False,
        )
    )
    before = store.workspace_path_for(workspace).read_bytes()

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_database_authority_rejected",
    ):
        bridge.run_provider(attempt)

    assert store.workspace_path_for(workspace).read_bytes() == before
    assert store.load_workspace(workspace) is not None
    assert portals and portals[0].run_count == 0
    assert portals[0].closed is True


def test_bridge_reauthorizes_database_immediately_before_lifecycle_finalize(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    original_authority = bridge.prior_attempt_authority
    assert original_authority is not None
    calls = 0

    def revoked_on_second_read(
        attempt_value: object,
        current_binding: object,
        prior_binding: object,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        authority = dict(
            original_authority(
                attempt_value,
                current_binding,
                prior_binding,
            )
        )
        if calls == 2:
            authority["authorized"] = False
        return authority

    bridge.prior_attempt_authority = revoked_on_second_read

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_database_authority_rejected",
    ):
        bridge.run_provider(attempt)

    record = store.load_workspace(workspace)
    assert record is not None and record.is_nonterminal
    receipt_path = (
        bridge._paths(attempt).root
        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["phase"] == "prepared"
    assert portals and portals[0].run_count == 0

    replacement = replace(
        record,
        lane_id=f"{Path(record.state_dir).resolve()}:replacement-lane",
    )
    store.workspace_path_for(workspace).write_text(
        json.dumps(replacement.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_receipt_authority_mismatch",
    ):
        bridge.run_provider(attempt)

    assert store.load_workspace(workspace) == replacement
    assert portals[-1].run_count == 0


def test_bridge_rejects_lifecycle_record_replacement_after_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    original = store.require_exact_dead_owner
    changed_once = False

    def replaced_record(*args: object, **kwargs: object) -> object:
        nonlocal changed_once
        record = original(*args, **kwargs)
        if not changed_once and record.is_nonterminal:
            changed_once = True
            return replace(record, updated_at=record.updated_at + 1.0)
        return record

    monkeypatch.setattr(store, "require_exact_dead_owner", replaced_record)

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_record_changed",
    ):
        bridge.run_provider(attempt)

    record = store.load_workspace(workspace)
    assert record is not None and record.is_nonterminal
    assert portals and portals[0].run_count == 0


def test_bridge_rejects_open_database_authority_mapping(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    original_authority = bridge.prior_attempt_authority
    assert original_authority is not None

    def open_authority(
        attempt_value: object,
        current_binding: object,
        prior_binding: object,
    ) -> dict[str, object]:
        value = dict(
            original_authority(
                attempt_value,
                current_binding,
                prior_binding,
            )
        )
        value["unexpected_authority"] = True
        return value

    bridge.prior_attempt_authority = open_authority
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_database_authority_rejected",
    ):
        bridge.run_provider(attempt)

    record = store.load_workspace(workspace)
    assert record is not None and record.is_nonterminal
    assert portals and portals[0].run_count == 0


def test_bridge_rejects_relevant_hashed_sibling_mismatch(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    record = store.load_workspace(workspace)
    assert record is not None
    prior_root = Path(record.state_dir)
    prior_root.rename(prior_root.with_name("f" * 24))

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_relevant_binding_invalid",
    ):
        bridge.run_provider(attempt)

    assert store.load_workspace(workspace) is not None
    assert portals and portals[0].run_count == 0


def test_bridge_rejects_prior_portal_active_tuple_mismatch(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    record = store.load_workspace(workspace)
    assert record is not None
    state_path = Path(record.state_dir) / "portal-task-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["active_task_cid"] = "baguqeeraforeign"
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_active_tuple_mismatch",
    ):
        bridge.run_provider(attempt)

    assert store.load_workspace(workspace) is not None
    assert portals and portals[0].run_count == 0


def test_bridge_fails_closed_when_workspace_process_is_active(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    process = store.proc_root / "424242"
    process.mkdir()
    (process / "cwd").symlink_to(workspace)
    (process / "cmdline").write_bytes(b"python\0worker.py\0")

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_worktree_process_active",
    ):
        bridge.run_provider(attempt)

    record = store.load_workspace(workspace)
    assert record is not None and record.is_nonterminal
    assert portals and portals[0].run_count == 0


def test_bridge_fails_closed_when_container_inventory_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )

    def unavailable(_workspace: Path) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(
            "cross_attempt_lifecycle_container_inventory_unavailable"
        )

    monkeypatch.setattr(bridge, "_strict_workspace_container_scan", unavailable)
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_container_inventory_unavailable",
    ):
        bridge.run_provider(attempt)

    record = store.load_workspace(workspace)
    assert record is not None and record.is_nonterminal
    assert portals and portals[0].run_count == 0


def test_container_root_mount_overlaps_recovery_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "worktrees" / "attempt"
    workspace.mkdir(parents=True)

    assert DatabasePortalExecutionBridge._mount_source_overlaps_workspace(
        "/",
        workspace,
    ) is True


def test_container_symlinked_parent_mount_overlaps_recovery_workspace(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "real-worktrees"
    workspace = parent / "attempt"
    workspace.mkdir(parents=True)
    alias = tmp_path / "worktrees-alias"
    alias.symlink_to(parent, target_is_directory=True)

    assert DatabasePortalExecutionBridge._mount_source_overlaps_workspace(
        str(alias),
        workspace,
    ) is True


def test_bridge_resumes_prepared_receipt_after_terminal_crash_boundary(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    first = bridge.run_provider(attempt)
    assert first["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    receipt_path = (
        bridge._paths(attempt).root
        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    committed = bridge._read_recovery_receipt(receipt_path)
    prepared_body = {
        field: committed[field]
        for field in bridge_module._RECOVERY_RECEIPT_FIELDS.difference(
            {"recovery_id", "receipt_id"}
        )
    }
    prepared_body["phase"] = "prepared"
    prepared_body["terminal_lifecycle_authority_id"] = ""
    prepared = bridge._seal_recovery_receipt(prepared_body)
    receipt_path.write_text(
        json.dumps(prepared, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    historical_workspace = workspace.with_name("historical-terminal")
    historical = replace(
        terminal,
        workspace_path=str(historical_workspace.resolve()),
        record_id="",
        updated_at=terminal.updated_at - 1.0,
    )
    store.workspace_path_for(historical_workspace).write_text(
        json.dumps(historical.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    second = bridge.run_provider(attempt)

    assert second["accepted"] is True
    resumed = bridge._read_recovery_receipt(receipt_path)
    assert resumed["phase"] == "committed"
    assert resumed["recovery_id"] == prepared["recovery_id"]
    assert store.load_workspace(workspace) == terminal
    assert len(portals) == 2
    assert portals[1].run_count == 0


def test_bridge_repairs_prepared_terminal_with_stale_task_index(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    first = bridge.run_provider(attempt)
    assert first["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    receipt_path = (
        bridge._paths(attempt).root
        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    committed = bridge._read_recovery_receipt(receipt_path)
    prepared_body = {
        field: committed[field]
        for field in bridge_module._RECOVERY_RECEIPT_FIELDS.difference(
            {"recovery_id", "receipt_id"}
        )
    }
    prepared_body["phase"] = "prepared"
    prepared_body["terminal_lifecycle_authority_id"] = ""
    prepared = bridge._seal_recovery_receipt(prepared_body)
    receipt_path.write_text(
        json.dumps(prepared, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predecessor = replace(
        terminal,
        state=terminal.state.__class__(prepared["prior_lifecycle_state"]),
        fence=prepared["prior_lifecycle_fence"],
    )
    index_path = store.task_index_path_for(
        canonical_task_cid=terminal.canonical_task_cid,
        task_id=terminal.task_id,
        attempt=terminal.attempt,
    )
    index_path.write_text(
        json.dumps(
            store._task_index_payload(predecessor),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    second = bridge.run_provider(attempt)

    assert second["accepted"] is True
    assert bridge._read_recovery_receipt(receipt_path)["phase"] == "committed"
    assert json.loads(index_path.read_text(encoding="utf-8")) == (
        store._task_index_payload(terminal)
    )
    assert len(portals) == 2
    assert portals[1].run_count == 0


def test_bridge_rejects_partial_terminal_not_bound_to_prepared_authority(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    assert bridge.run_provider(attempt)["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    receipt_path = (
        bridge._paths(attempt).root
        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    committed = bridge._read_recovery_receipt(receipt_path)
    prepared_body = {
        field: committed[field]
        for field in bridge_module._RECOVERY_RECEIPT_FIELDS.difference(
            {"recovery_id", "receipt_id"}
        )
    }
    prepared_body["phase"] = "prepared"
    prepared_body["terminal_lifecycle_authority_id"] = ""
    prepared = bridge._seal_recovery_receipt(prepared_body)
    receipt_path.write_text(
        json.dumps(prepared, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    predecessor = replace(
        terminal,
        state=terminal.state.__class__(prepared["prior_lifecycle_state"]),
        fence=prepared["prior_lifecycle_fence"],
    )
    index_path = store.task_index_path_for(
        canonical_task_cid=terminal.canonical_task_cid,
        task_id=terminal.task_id,
        attempt=terminal.attempt,
    )
    stale_index = store._task_index_payload(predecessor)
    index_path.write_text(
        json.dumps(stale_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replacement = replace(
        terminal,
        lane_id=f"{Path(terminal.state_dir).resolve()}:replacement-lane",
    )
    store.workspace_path_for(workspace).write_text(
        json.dumps(replacement.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_receipt_authority_mismatch",
    ):
        bridge.run_provider(attempt)

    assert json.loads(index_path.read_text(encoding="utf-8")) == stale_index
    assert store.load_workspace(workspace) == replacement
    assert len(portals) == 2
    assert portals[1].run_count == 0


def test_bridge_ignores_unrelated_process_count_changes_between_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    inspected = iter((3, 4))
    monkeypatch.setattr(
        bridge,
        "_strict_workspace_process_scan",
        lambda _store, _workspace: {
            "same_uid_processes_inspected": next(inspected)
        },
    )

    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    assert portals and portals[0].run_count == 1


def test_bridge_rejects_symlink_attempt_root_before_first_write(
    tmp_path: Path,
) -> None:
    target = tmp_path / "real-attempts"
    target.mkdir()
    attempt_root = tmp_path / "attempts"
    attempt_root.symlink_to(target, target_is_directory=True)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: object(),
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="attempt root is not a sealed direct child",
    ):
        bridge._ensure_attempt_projection(_attempt(), _record())

    assert tuple(target.iterdir()) == ()


def test_bridge_atomic_write_rejects_attempt_directory_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge_for_projection(tmp_path)
    attacker_directory = tmp_path / "attacker-directory"
    attacker_directory.mkdir()
    displaced_directory = tmp_path / "displaced-attempt-directory"
    original_seal = bridge._seal_attempt_directory
    swapped = False

    def seal_then_swap(
        selected_paths: DatabasePortalAttemptPaths,
        *,
        attempt_id: object,
        create: bool,
    ) -> dict[str, int]:
        nonlocal swapped
        identity = original_seal(
            selected_paths,
            attempt_id=attempt_id,
            create=create,
        )
        if not create and not swapped:
            selected_paths.root.rename(displaced_directory)
            selected_paths.root.symlink_to(
                attacker_directory,
                target_is_directory=True,
            )
            swapped = True
        return identity

    monkeypatch.setattr(bridge, "_seal_attempt_directory", seal_then_swap)

    with pytest.raises(DatabasePortalBridgeError, match="sealed directory"):
        bridge._ensure_attempt_projection(_attempt(), _record())

    assert swapped is True
    assert tuple(attacker_directory.iterdir()) == ()
    assert tuple(displaced_directory.iterdir()) == ()


def test_bridge_atomic_write_fsyncs_file_replace_and_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _bridge_for_projection(tmp_path)
    paths = bridge._paths(_attempt())
    directory_identity = bridge._seal_attempt_directory(
        paths,
        attempt_id=_attempt().attempt_id,
        create=True,
    )
    paths.state.write_bytes(b"old-state")
    operations: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        descriptor_identity = os.fstat(descriptor)
        operations.append(
            "fsync-directory"
            if stat.S_ISDIR(descriptor_identity.st_mode)
            else "fsync-file"
        )
        real_fsync(descriptor)

    def tracked_replace(
        source: str,
        target: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        assert src_dir_fd is not None
        assert src_dir_fd == dst_dir_fd
        assert Path(source).name == source
        assert target == paths.state.name
        operations.append("replace")
        real_replace(
            source,
            target,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(bridge_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(bridge_module.os, "replace", tracked_replace)

    bridge_module._atomic_write(
        paths.state,
        b"durable-state",
        sealed_directory_identity=directory_identity,
    )

    assert paths.state.read_bytes() == b"durable-state"
    assert operations == ["fsync-file", "replace", "fsync-directory"]
    assert tuple(paths.root.iterdir()) == (paths.state,)


def test_database_daemon_rejects_open_superseded_binding_records() -> None:
    current = replace(
        _attempt(),
        attempt_id="attempt:current",
        claim_id="claim:current",
        attempt_number=5,
    )
    protected: list[str] = []
    fake = SimpleNamespace(
        get_attempt=lambda _attempt_id: current,
        _protect_attempt_write=lambda attempt: protected.append(
            attempt.attempt_id
        ),
    )
    current_binding = {
        "attempt_id": current.attempt_id,
        "claim_id": current.claim_id,
        "task_cid": current.task_cid,
        "task_alias": current.task_alias,
        "fencing_token": current.fencing_token,
        "fence_epoch": current.fence_epoch,
        "lease_id": current.lease_id,
        "binding_id": "sha256:current",
    }
    prior_binding = {
        "attempt_id": "attempt:prior",
        "claim_id": "claim:prior",
        "task_cid": current.task_cid,
        "task_alias": current.task_alias,
        "fencing_token": 6,
        "fence_epoch": 2,
        "lease_id": "lease:prior",
        "binding_id": "sha256:prior",
    }

    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not a closed record",
    ):
        DatabaseImplementationDaemon.authorize_superseded_portal_attempt_binding(
            fake,
            current,
            current_binding,
            prior_binding,
        )
    assert protected == []


def test_bridge_uses_only_attempt_local_projection_and_seals_receipt(
    tmp_path: Path,
) -> None:
    canonical_board = tmp_path / "canonical-board.md"
    canonical_board.write_text(
        "# Canonical\n\n## LGSWF-004 Authority\n\n- Status: ready\n",
        encoding="utf-8",
    )
    original = canonical_board.read_bytes()
    portals: list[_CompletingPortal] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        portal = _CompletingPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )
    provider = bridge.run_provider(_attempt())
    effect = bridge.apply_effect(_attempt(), provider)
    validation = bridge.validate_effect(_attempt(), effect)

    assert provider["schema"] == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1
    assert provider["accepted"] is True
    assert provider["provider"] == "PortalImplementationDaemon"
    assert provider["completion_authority"] == "DatabaseImplementationDaemon"
    assert provider["evidence_digest"].startswith("sha256:")
    assert "private provider payload" not in json.dumps(provider)
    assert "provider_payload" not in json.dumps(provider)
    assert effect["status"] == "applied"
    assert validation["outcome"] == "passed"
    assert validation["evidence_digest"] == provider["evidence_digest"]
    assert canonical_board.read_bytes() == original
    assert portals and portals[0].closed is True
    attempt_boards = list(
        (tmp_path / "attempts").glob(
            "*/task-projection.runtime.todo.md"
        )
    )
    assert len(attempt_boards) == 1
    assert attempt_boards[0].parent == portals[0].paths.state.parent
    assert attempt_boards[0].name.endswith("runtime.todo.md")
    assert "Projection authority: false" in attempt_boards[0].read_text(encoding="utf-8")

    forged_v2 = dict(provider)
    forged_v2["schema"] = DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
    forged_v2.pop("receipt_id", None)
    forged_v2["receipt_id"] = bridge_module._sha256_bytes(
        bridge_module._canonical_json(forged_v2)
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="source-transition receipt without its transition",
    ):
        bridge.apply_effect(_attempt(), forged_v2)


def test_bridge_projection_satisfies_real_ignored_runtime_durability(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("branch", "-M", "main")
    (repository / ".gitignore").write_text(
        "attempts/\n",
        encoding="utf-8",
    )
    (repository / "README.md").write_text("seed\n", encoding="utf-8")
    git("add", ".gitignore", "README.md")
    git(
        "-c",
        "user.name=Portal Test",
        "-c",
        "user.email=portal@example.invalid",
        "commit",
        "-q",
        "-m",
        "seed",
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=repository / "attempts",
        portal_factory=lambda _paths, _alias: object(),
        repo_root=repository,
        board_namespace="test-board-v1",
        task_header_prefix="## LGSWF-",
    )
    paths = bridge._paths(_attempt())
    paths.root.mkdir(parents=True)
    paths.task_projection.write_text(
        bridge._render_projection(_attempt(), _record()),
        encoding="utf-8",
    )
    daemon = PortalImplementationDaemon(
        todo_path=paths.task_projection,
        state_path=paths.state,
        strategy_path=paths.strategy,
        events_path=paths.events,
        repo_root=repository,
        task_header_prefix="## LGSWF-",
    )
    [task] = daemon._load_tasks()
    task_cids = {
        task.task_id: daemon._identity_for_task(task).canonical_task_cid,
    }

    update = daemon._mark_reconciled_completion_in_todo(
        task,
        [task],
        task_cids,
    )
    persistence = daemon._reconciled_completion_persisted(
        update,
        task_cids,
    )

    assert update["commit_result"]["reason"] == "no_changes"
    assert persistence["passed"] is True
    assert persistence["durable_update"] is True
    assert persistence["runtime_taskboard_binding"]["ignored"] is True
    assert persistence["fsynced_taskboard_snapshot"]["passed"] is True


def test_bridge_rejects_projection_contract_tampering(tmp_path: Path) -> None:
    class TamperingPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            text = self.paths.task_projection.read_text(encoding="utf-8")
            self.paths.task_projection.write_text(
                text.replace(
                    "- Acceptance: Focused validation passes",
                    "- Acceptance: no validation required",
                ),
                encoding="utf-8",
            )
            return {"implementation_result": {"returncode": 0}}

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: TamperingPortal(paths, alias),
    )
    with pytest.raises(DatabasePortalBridgeError, match="outside its mutable status"):
        bridge.run_provider(_attempt())


def test_bridge_honors_explicit_deferral_without_reason_keyword(
    tmp_path: Path,
) -> None:
    class DeferredPortal:
        def __init__(self) -> None:
            self.closed = False

        def run_once(self) -> dict[str, object]:
            return {
                "active_task_id": "",
                "implementation_result": {
                    "task_id": "LGSWF-004",
                    "returncode": 1,
                    "reason": "validation_project_dependency_preflight_failed",
                    "deferred": True,
                    "attempt_consumed": False,
                    "provider_call_allowed": False,
                },
            }

        def close_event_runtime(self) -> None:
            self.closed = True

    portals: list[DeferredPortal] = []

    def factory(_paths: object, _alias: str) -> DeferredPortal:
        portal = DeferredPortal()
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="^validation_project_dependency_preflight_failed$",
    ):
        bridge.run_provider(_attempt())

    assert portals and portals[0].closed is True
    attempt_boards = list(
        (tmp_path / "attempts").glob(
            "*/task-projection.runtime.todo.md"
        )
    )
    assert len(attempt_boards) == 1
    projection = attempt_boards[0].read_text(encoding="utf-8")
    assert "- Status: ready" in projection
    assert "Projection authority: false" in projection


def _external_owner_recovery_result(*, owner: str) -> dict[str, object]:
    return {
        "blocked": True,
        "reason": "external_protected_checkout_recovery_required",
        "protected_checkout_recovery": {
            "required": True,
            "adopted": False,
            "blocked": True,
            "recovered": False,
            "reason": "external_protected_checkout_recovery_required",
            "protected_recovery_owner": owner,
            "lock_path": "/tmp/repository.lock",
        },
        "unchanged": True,
        "write_count": 0,
        "projection_delta": {},
        "implementation_result": None,
        "merge_reconciliation": [],
    }


def test_bridge_defers_exact_external_owner_recovery_without_settling(
    tmp_path: Path,
) -> None:
    class RecoveryBlockedPortal:
        def run_once(self) -> dict[str, object]:
            return _external_owner_recovery_result(
                owner="implementation_supervisor"
            )

        def close_event_runtime(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: RecoveryBlockedPortal(),
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="^external_protected_checkout_recovery_required$",
    ):
        bridge.run_provider(_attempt())


def test_bridge_rejects_unbound_external_owner_recovery_as_terminal(
    tmp_path: Path,
) -> None:
    class UnboundRecoveryPortal:
        def run_once(self) -> dict[str, object]:
            return _external_owner_recovery_result(owner="")

        def close_event_runtime(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: UnboundRecoveryPortal(),
    )

    with pytest.raises(DatabasePortalBridgeError) as captured:
        bridge.run_provider(_attempt())
    assert not isinstance(captured.value, DatabasePortalBridgeDeferred)
    assert str(captured.value) == (
        "external_protected_checkout_recovery_required"
    )


def test_external_owner_recovery_deferral_shape_is_closed() -> None:
    exact = _external_owner_recovery_result(
        owner="implementation_supervisor"
    )
    assert (
        DatabasePortalExecutionBridge._is_external_protected_recovery_deferral(
            exact
        )
        is True
    )

    variants: list[dict[str, object]] = []
    for field, value in (
        ("adopted", True),
        ("protected_recovery_owner", "implementation_daemon"),
        ("lock_path", ""),
    ):
        variant = json.loads(json.dumps(exact))
        variant["protected_checkout_recovery"][field] = value
        variants.append(variant)
    for field, value in (
        ("write_count", 1),
        ("projection_delta", {"changed": True}),
        ("merge_reconciliation", [{"status": "pending"}]),
    ):
        variant = json.loads(json.dumps(exact))
        variant[field] = value
        variants.append(variant)

    assert all(
        not DatabasePortalExecutionBridge._is_external_protected_recovery_deferral(
            variant
        )
        for variant in variants
    )


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_production_database_daemon_cannot_complete_with_default_noops(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:fail-closed",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:bridge",
                "tasks": [
                    {
                        "task_cid": "task:cid:004",
                        "task_id": "LGSWF-004",
                        "goal_cid": "goal:inventory",
                        "status": "ready",
                        "priority": "P0",
                        "ordinal": 4,
                        "title": "Inventory",
                    }
                ],
            }
        )
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="no provider executor",
        ):
            daemon.run_once()
        task = daemon.task_source.get_task("task:cid:004")
        assert task is not None
        assert task.status != "completed"
        assert (
            daemon.provider_invocation_recorded(
                daemon.list_running_attempts()[0].attempt_id,
                idempotency_key=f"provider:{daemon.list_running_attempts()[0].attempt_id}",
            )
            is None
        )
    finally:
        daemon.close()


def test_quack_mode_refuses_direct_duckdb_execution(tmp_path: Path) -> None:
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="loopback quack:",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            authority_mode="quack",
            task_source_kind="duckdb",
        )


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_configured_production_runner_binds_real_portal_bridge(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--todo-path",
            str(tmp_path / "canonical-board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "lgswf",
            "--worktree-root",
            ".worktrees",
            "--implement",
            "--once",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.require_real_execution is True
        assert daemon.execution_callbacks_bound is True
        assert daemon.markdown_path is None
        assert daemon.markdown_status_write_count == 0
    finally:
        daemon.close()
