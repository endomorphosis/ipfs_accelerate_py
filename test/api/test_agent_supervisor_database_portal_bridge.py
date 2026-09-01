"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import errno
import hashlib
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
    CROSS_ATTEMPT_DECLARED_OUTPUT_PRESERVATION_SCHEMA,
    CROSS_ATTEMPT_LIFECYCLE_AUTHORITY_SCHEMA,
    CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME,
    CROSS_ATTEMPT_PROTECTED_STATE_CLEARANCE_SCHEMA,
    CROSS_ATTEMPT_PROTECTED_STATE_RETIREMENT_SCHEMA,
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
    protected_marker: bool = False,
    nested_output_path: str = "",
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
    (repository / "control.todo.md").write_text(
        "# protected control\n",
        encoding="utf-8",
    )
    git("add", "seed.py", "control.todo.md")
    git("commit", "-q", "-m", "seed")

    if nested_output_path:
        nested_source = tmp_path / "nested-source"
        nested_source.mkdir()

        def nested_git(*arguments: str) -> str:
            return subprocess.run(
                ["git", *arguments],
                cwd=nested_source,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        nested_git("init", "-q")
        nested_git("branch", "-M", "main")
        nested_git("config", "user.name", "Nested Test")
        nested_git("config", "user.email", "nested@example.invalid")
        (nested_source / "README.md").write_text(
            "nested\n",
            encoding="utf-8",
        )
        nested_git("add", "README.md")
        nested_git("commit", "-q", "-m", "nested seed")
        git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(nested_source),
            "external/ipfs_datasets_py",
        )
        git("commit", "-q", "-am", "add nested datasets authority")

    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    workspace = worktree_root / "prior"
    prior_branch = "implementation/lgswf-004-attempt-1"
    git("branch", prior_branch)
    git("worktree", "add", "-q", str(workspace), prior_branch)
    if nested_output_path:
        git(
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "-q",
            cwd=workspace,
        )

    current_record = _record()
    prior_record = _record()
    if nested_output_path:
        declared = f"external/ipfs_datasets_py/{nested_output_path}"
        for selected_record in (current_record, prior_record):
            selected_record.outputs = ({"path": declared},)
            selected_record.body = {
                **selected_record.body,
                "write_scope": [declared],
            }
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
        implementation_protected_paths = ("control.todo.md",)
        repo_root = repository

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

        _implementation_protected_path_snapshot = (
            PortalImplementationDaemon._implementation_protected_path_snapshot
        )
        _implementation_protected_path_identity = staticmethod(
            PortalImplementationDaemon._implementation_protected_path_identity
        )
        _implementation_protected_snapshot_errors = staticmethod(
            PortalImplementationDaemon._implementation_protected_snapshot_errors
        )
        _implementation_protected_path_mutations = (
            PortalImplementationDaemon._implementation_protected_path_mutations
        )
        _implementation_protected_change_kind = staticmethod(
            PortalImplementationDaemon._implementation_protected_change_kind
        )
        _authorized_concurrent_protected_path_update = (
            PortalImplementationDaemon._authorized_concurrent_protected_path_update
        )
        _implementation_protected_git_head = staticmethod(
            PortalImplementationDaemon._implementation_protected_git_head
        )
        _trusted_protected_path_commit = staticmethod(
            PortalImplementationDaemon._trusted_protected_path_commit
        )

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
            "current_attempt_number": int(
                _attempt_value.attempt_number
            ),
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
    if nested_output_path:
        nested_output = workspace / "external/ipfs_datasets_py" / nested_output_path
        nested_output.parent.mkdir(parents=True, exist_ok=True)
        nested_output.write_text("VALUE = 'preserve me'\n", encoding="utf-8")
    if protected_marker:
        probe = RecoveryPortal(prior_paths, current_attempt.task_alias)
        snapshot = probe._implementation_protected_path_snapshot(workspace)
        (prior_paths.root / "implementation-protected-path-active.json").write_text(
            json.dumps(
                {
                    "schema": "implementation-protected-path-active-v1",
                    "recorded_at": "2026-09-01T00:00:00Z",
                    "task_id": current_attempt.task_alias,
                    "attempt": 1,
                    "workspace_path": str(workspace.resolve()),
                    "ephemeral_worktree": True,
                    "protected_paths": list(
                        probe.implementation_protected_paths
                    ),
                    "snapshot": snapshot,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
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


def test_bridge_attests_legacy_no_delta_rescue_before_recovery(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    original_branch = predecessor.branch.removeprefix("refs/heads/")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    original_head = git("rev-parse", "HEAD^{commit}")
    original_tree = git("rev-parse", "HEAD^{tree}")
    rescue_branch = "rescue/worktree/legacy-no-delta"
    git("checkout", "-q", "-b", rescue_branch)

    provider = bridge.run_provider(attempt)

    assert provider["accepted"] is True
    attestation_head = git("rev-parse", "HEAD^{commit}")
    assert attestation_head != original_head
    assert git("rev-parse", "HEAD^{tree}") == original_tree
    assert git("rev-parse", f"refs/heads/{original_branch}^{{commit}}") == (
        original_head
    )
    metadata = git("show", "-s", "--format=%ae%n%s%n%b", "HEAD")
    assert metadata.splitlines()[0] == "implementation-supervisor@example.invalid"
    assert f"Rescue dirty worktree {original_branch}" in metadata
    assert f"Original branch: {original_branch}" in metadata
    receipt = json.loads(
        (
            bridge._paths(attempt).root
            / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
        ).read_text(encoding="utf-8")
    )
    preservation = receipt["preservation"]
    assert preservation["head"] == attestation_head
    assert preservation["tree"] == original_tree
    assert preservation["preservation_mode"] == (
        f"legacy_no_delta_rescue_attestation:{attestation_head}"
    )
    assert receipt["provider_dispatched"] is False
    assert receipt["task_completion_authority"] is False
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    assert portals and portals[0].run_count == 1


def test_bridge_rejects_legacy_rescue_with_uncommitted_root_delta(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(tmp_path)
    )
    original_head = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "checkout", "-q", "-b", "rescue/worktree/unsafe-root-delta"],
        cwd=workspace,
        check=True,
    )
    (workspace / "seed.py").write_text("SEED = False\n", encoding="utf-8")

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_legacy_rescue_delta_unsafe",
    ):
        bridge.run_provider(attempt)

    assert subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == original_head
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None and predecessor.is_nonterminal
    assert portals and portals[0].run_count == 0


def test_bridge_preserves_receipt_then_retires_exact_dead_active_marker(
    tmp_path: Path,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    active_bytes = active_path.read_bytes()

    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    assert not active_path.exists()
    clearance_paths = tuple(
        prior_root.glob("cross-attempt-protected-state-clearance-*.json")
    )
    retired_paths = tuple(
        prior_root.glob("implementation-protected-path-retired-*.json")
    )
    marker_blobs = tuple(
        prior_root.glob("cross-attempt-protected-state-marker-*.json")
    )
    assert len(clearance_paths) == len(retired_paths) == len(marker_blobs) == 1
    assert retired_paths[0].read_bytes() == active_bytes
    assert marker_blobs[0].read_bytes() == active_bytes
    receipt = json.loads(clearance_paths[0].read_text(encoding="utf-8"))
    assert receipt["schema"] == CROSS_ATTEMPT_PROTECTED_STATE_CLEARANCE_SCHEMA
    assert receipt["protected_path_proof"]["mode"] == "exact_snapshot"
    assert receipt["clearance_phase"] == "prepared"
    assert receipt["active_marker_retired"] is False
    assert receipt["retirement_operation"] == (
        "atomic_noreplace_rename_after_receipt"
    )
    assert receipt["worktree_deleted"] is False
    assert receipt["provider_dispatched"] is False
    assert receipt["mutation_authority"] is False
    assert receipt["merge_authority"] is False
    assert receipt["task_completion_authority"] is False
    assert receipt["worker_self_approval"] is False
    assert receipt["normal_validation_required"] is True
    retirement_paths = tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )
    assert len(retirement_paths) == 1
    retirement = json.loads(
        retirement_paths[0].read_text(encoding="utf-8")
    )
    assert retirement["schema"] == CROSS_ATTEMPT_PROTECTED_STATE_RETIREMENT_SCHEMA
    assert retirement["clearance_phase"] == "retired"
    assert retirement["active_marker_retired"] is True
    assert retirement["prepared_clearance"] == receipt
    assert retirement["worker_self_approval"] is False
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    assert portals and portals[0].run_count == 1


@pytest.mark.parametrize(
    ("admit_final_source", "spoof_legacy_authority"),
    [(True, False), (False, False), (False, True)],
)
def test_bridge_protected_recovery_uses_exact_live_capsule_source_transition(
    tmp_path: Path,
    admit_final_source: bool,
    spoof_legacy_authority: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    repository = bridge.repo_root
    assert repository is not None

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    protected = repository / "control.todo.md"
    protected.write_text("# operator repair\n", encoding="utf-8")
    git("add", "control.todo.md")
    git("commit", "-q", "-m", "bounded operator repair")
    repair_head = git("rev-parse", "HEAD^{commit}")
    repair_tree = git("rev-parse", "HEAD^{tree}")
    protected.write_text("# sealed operator control\n", encoding="utf-8")
    git("add", "control.todo.md")
    git("commit", "-q", "-m", "seal operator control")
    final_head = git("rev-parse", "HEAD^{commit}")
    final_tree = git("rev-parse", "HEAD^{tree}")
    payload = protected.read_bytes()
    admission_cid = "baguqeera" + "c" * 48
    admission_payload = {
        "admission_cid": admission_cid,
        "board_namespace": "test-board-v1",
        "source_head": final_head if admit_final_source else repair_head,
        "source_tree": final_tree if admit_final_source else repair_tree,
        "control_artifacts": [
            {
                "path": "control.todo.md",
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        ],
    }
    bridge.configured_board_admission_cid = admission_cid
    bridge._configured_board_live_admission_json = json.dumps(
        admission_payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    class ParsedAdmission:
        def __init__(self, payload_value: dict[str, object]) -> None:
            self.admission_cid = str(payload_value["admission_cid"])
            self.board_namespace = str(payload_value["board_namespace"])
            self.source_head = str(payload_value["source_head"])
            self._payload = dict(payload_value)

        def as_dict(self) -> dict[str, object]:
            return json.loads(json.dumps(self._payload))

    def parse_test_admission(value: object) -> ParsedAdmission:
        assert isinstance(value, str)
        payload_value = json.loads(value)
        assert isinstance(payload_value, dict)
        return ParsedAdmission(payload_value)

    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_live_capsule as live_capsule,
    )

    monkeypatch.setattr(
        live_capsule,
        "parse_configured_board_live_capsule_admission",
        parse_test_admission,
    )

    if spoof_legacy_authority:
        original_factory = bridge.portal_factory

        def spoofing_factory(paths: object, alias: str) -> object:
            portal = original_factory(paths, alias)
            portal._authorized_concurrent_protected_path_update = (
                lambda **_kwargs: {
                    "authority": "spoofed_trusted_author",
                    "mutation_authority": False,
                    "task_completion_authority": False,
                }
            )
            return portal

        bridge.portal_factory = spoofing_factory

    if not admit_final_source:
        with pytest.raises(
            DatabasePortalBridgeDeferred,
            match="cross_attempt_lifecycle_protected_snapshot_mutated",
        ):
            bridge.run_provider(attempt)
        predecessor = store.load_workspace(workspace)
        assert predecessor is not None and predecessor.is_nonterminal
        assert portals[-1].run_count == 0
        return

    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    clearance_paths = tuple(
        Path(terminal.state_dir).glob(
            "cross-attempt-protected-state-clearance-*.json"
        )
    )
    assert len(clearance_paths) == 1
    clearance = json.loads(clearance_paths[0].read_text(encoding="utf-8"))
    trusted = clearance["protected_path_proof"]["trusted_shared_update"]
    assert trusted["authority"] == "configured_board_live_capsule"
    assert trusted["admission_cid"] == admission_cid
    assert trusted["before_head"] != trusted["after_head"] == final_head
    assert trusted["changed_protected_blobs"] == [
        {
            "path": "control.todo.md",
            "mode": "100644",
            "blob_oid": git("rev-parse", f"{final_head}:control.todo.md"),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    assert trusted["operator_or_worker_identity_inferred"] is False


def test_bridge_content_addresses_exact_declared_nested_output_before_clearance(
    tmp_path: Path,
) -> None:
    nested_path = "ipfs_datasets_py/program_execution_trace.py"
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
            nested_output_path=nested_path,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    output_path = workspace / "external/ipfs_datasets_py" / nested_path
    output_bytes = output_path.read_bytes()
    output_mode = stat.S_IMODE(output_path.stat().st_mode)

    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    assert output_path.read_bytes() == output_bytes
    preservation_paths = tuple(
        prior_root.glob("cross-attempt-declared-output-preservation-*.json")
    )
    blob_paths = tuple(
        prior_root.glob("cross-attempt-declared-output-blob-*.blob")
    )
    clearance_paths = tuple(
        prior_root.glob("cross-attempt-protected-state-clearance-*.json")
    )
    assert len(preservation_paths) == len(blob_paths) == len(clearance_paths) == 1
    assert blob_paths[0].read_bytes() == output_bytes
    preservation = json.loads(
        preservation_paths[0].read_text(encoding="utf-8")
    )
    assert preservation["schema"] == (
        CROSS_ATTEMPT_DECLARED_OUTPUT_PRESERVATION_SCHEMA
    )
    assert preservation["outputs"] == [
        {
            "blob_filename": blob_paths[0].name,
            "gitlink_commit": preservation["outputs"][0]["gitlink_commit"],
            "gitlink_path": "external/ipfs_datasets_py",
            "mode": output_mode,
            "nested_path": nested_path,
            "repository_path": f"external/ipfs_datasets_py/{nested_path}",
            "sha256": "sha256:"
            + hashlib.sha256(output_bytes).hexdigest(),
            "size": len(output_bytes),
        }
    ]
    assert preservation["worktree_deleted"] is False
    assert preservation["provider_dispatched"] is False
    assert preservation["mutation_authority"] is False
    assert preservation["merge_authority"] is False
    assert preservation["task_completion_authority"] is False
    assert preservation["normal_validation_required"] is True
    clearance = json.loads(clearance_paths[0].read_text(encoding="utf-8"))
    assert clearance["preservation"]["preservation_mode"] == (
        "content_addressed_declared_nested_outputs:"
        f"{preservation['preservation_id']}"
    )
    lifecycle_receipt = json.loads(
        (
            bridge._paths(attempt).root
            / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert lifecycle_receipt["preservation"]["preservation_mode"] == (
        clearance["preservation"]["preservation_mode"]
    )
    assert portals and portals[0].run_count == 1


def test_bridge_attests_live_shape_legacy_rescue_and_preserves_nested_output(
    tmp_path: Path,
) -> None:
    nested_path = "ipfs_datasets_py/program_execution_trace.py"
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
            nested_output_path=nested_path,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    original_branch = predecessor.branch.removeprefix("refs/heads/")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    original_head = git("rev-parse", "HEAD^{commit}")
    original_tree = git("rev-parse", "HEAD^{tree}")
    git("checkout", "-q", "-b", "rescue/worktree/live-shape-legacy")
    assert subprocess.run(
        ["git", "status", "--short", "--ignore-submodules=none"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines() == [" ? external/ipfs_datasets_py"]

    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    attestation_head = git("rev-parse", "HEAD^{commit}")
    assert git("rev-list", "--parents", "-n", "1", "HEAD") == (
        f"{attestation_head} {original_head}"
    )
    assert git("rev-parse", "HEAD^{tree}") == original_tree
    assert git("rev-parse", f"refs/heads/{original_branch}^{{commit}}") == (
        original_head
    )
    prior_root = Path(predecessor.state_dir)
    preservation_paths = tuple(
        prior_root.glob("cross-attempt-declared-output-preservation-*.json")
    )
    assert len(preservation_paths) == 1
    preservation = json.loads(preservation_paths[0].read_text(encoding="utf-8"))
    receipt = json.loads(
        (
            bridge._paths(attempt).root
            / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert receipt["preservation"]["head"] == attestation_head
    assert receipt["preservation"]["preservation_mode"] == (
        "content_addressed_declared_nested_outputs:"
        f"{preservation['preservation_id']}"
    )
    assert portals and portals[0].run_count == 1


@pytest.mark.parametrize(
    "unsafe_state",
    (
        "incident",
        "open_marker",
        "workspace_binding",
        "protected_mutation",
    ),
)
def test_bridge_never_clears_ambiguous_or_mutated_protected_state(
    tmp_path: Path,
    unsafe_state: str,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    if unsafe_state == "incident":
        (prior_root / "implementation-protected-path-incident.json").write_text(
            "incident\n",
            encoding="utf-8",
        )
    elif unsafe_state in {"open_marker", "workspace_binding"}:
        marker = json.loads(active_path.read_text(encoding="utf-8"))
        if unsafe_state == "open_marker":
            marker["unexpected"] = True
        else:
            marker["workspace_path"] = str(tmp_path / "other-workspace")
        active_path.write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        (workspace / "control.todo.md").write_text(
            "# unauthorized protected mutation\n",
            encoding="utf-8",
        )

    with pytest.raises(DatabasePortalBridgeDeferred):
        bridge.run_provider(attempt)

    assert active_path.is_file()
    assert not tuple(
        prior_root.glob("implementation-protected-path-retired-*.json")
    )
    assert not tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )
    current = store.load_workspace(workspace)
    assert current is not None and current.is_nonterminal
    assert not portals or portals[0].run_count == 0


@pytest.mark.parametrize(
    "unsafe_output",
    ("undeclared", "secret", "symlink", "tracked", "ignored"),
)
def test_bridge_never_preserves_ambiguous_or_unsafe_nested_output(
    tmp_path: Path,
    unsafe_output: str,
) -> None:
    nested_path = "ipfs_datasets_py/program_graph_builder.py"
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
            nested_output_path=nested_path,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    nested_root = workspace / "external/ipfs_datasets_py"
    output_path = nested_root / nested_path
    if unsafe_output == "undeclared":
        (nested_root / "undeclared.py").write_text(
            "UNDECLARED = True\n",
            encoding="utf-8",
        )
    elif unsafe_output == "secret":
        output_path.write_text(
            "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n",
            encoding="utf-8",
        )
    elif unsafe_output == "symlink":
        output_path.unlink()
        output_path.symlink_to("../README.md")
    elif unsafe_output == "tracked":
        (nested_root / "README.md").write_text(
            "tracked mutation\n",
            encoding="utf-8",
        )
    else:
        info_exclude = Path(
            subprocess.run(
                ["git", "rev-parse", "--git-path", "info/exclude"],
                cwd=nested_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        if not info_exclude.is_absolute():
            info_exclude = nested_root / info_exclude
        with info_exclude.open("a", encoding="utf-8") as stream:
            stream.write(f"/{nested_path}\n")

    with pytest.raises(DatabasePortalBridgeDeferred):
        bridge.run_provider(attempt)

    assert active_path.is_file()
    assert not tuple(
        prior_root.glob("cross-attempt-declared-output-preservation-*.json")
    )
    assert not tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )
    current = store.load_workspace(workspace)
    assert current is not None and current.is_nonterminal
    assert not portals or portals[0].run_count == 0


def test_bridge_reobserves_nested_bytes_before_marker_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_path = "ipfs_datasets_py/program_graph_builder.py"
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
            nested_output_path=nested_path,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    output_path = workspace / "external/ipfs_datasets_py" / nested_path
    original = bridge._preserve_declared_nested_outputs
    calls = 0

    def mutate_after_first_preservation(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        observed = original(**kwargs)
        calls += 1
        if calls == 1:
            output_path.write_text("VALUE = 'changed after receipt'\n", encoding="utf-8")
        return observed

    monkeypatch.setattr(
        bridge,
        "_preserve_declared_nested_outputs",
        mutate_after_first_preservation,
    )
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="changed_before_marker_retirement",
    ):
        bridge.run_provider(attempt)

    assert active_path.is_file()
    assert not tuple(
        prior_root.glob("implementation-protected-path-retired-*.json")
    )
    current = store.load_workspace(workspace)
    assert current is not None and current.is_nonterminal
    assert not portals or portals[0].run_count == 0


def test_bridge_recovers_exact_post_rename_pre_retirement_receipt_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    original = bridge._publish_protected_retirement_receipt
    calls = 0

    def fail_once(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DatabasePortalBridgeDeferred("simulated_post_rename_crash")
        return original(**kwargs)

    monkeypatch.setattr(
        bridge,
        "_publish_protected_retirement_receipt",
        fail_once,
    )
    with pytest.raises(DatabasePortalBridgeDeferred, match="simulated_post_rename_crash"):
        bridge.run_provider(attempt)
    assert not active_path.exists()
    assert len(
        tuple(prior_root.glob("implementation-protected-path-retired-*.json"))
    ) == 1
    assert not tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )

    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    assert len(
        tuple(prior_root.glob("cross-attempt-protected-state-retirement-*.json"))
    ) == 1
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    assert portals and portals[-1].run_count == 1


def test_bridge_resumes_terminal_cas_before_protected_marker_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    original = bridge._retire_dead_protected_active_marker

    def fail_after_terminal(**kwargs: object) -> dict[str, object]:
        if kwargs["perform_retirement"] is True:
            raise DatabasePortalBridgeDeferred(
                "simulated_terminal_before_marker_retirement_crash"
            )
        return original(**kwargs)

    monkeypatch.setattr(
        bridge,
        "_retire_dead_protected_active_marker",
        fail_after_terminal,
    )
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="simulated_terminal_before_marker_retirement_crash",
    ):
        bridge.run_provider(attempt)
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    assert active_path.is_file()
    recovery_path = (
        bridge._paths(attempt).root
        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    prepared_recovery = bridge._read_recovery_receipt(recovery_path)
    assert prepared_recovery["phase"] == "prepared"
    assert len(
        tuple(prior_root.glob("cross-attempt-protected-state-clearance-*.json"))
    ) == 1
    assert not tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )

    monkeypatch.setattr(
        bridge,
        "_retire_dead_protected_active_marker",
        original,
    )
    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    assert not active_path.exists()
    assert bridge._read_recovery_receipt(recovery_path)["phase"] == "committed"
    assert len(
        tuple(prior_root.glob("cross-attempt-protected-state-retirement-*.json"))
    ) == 1
    assert portals and portals[-1].run_count == 1


def test_bridge_distinct_successor_attempt_is_unblocked_by_terminal_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    original = bridge._retire_dead_protected_active_marker

    def fail_after_terminal(**kwargs: object) -> dict[str, object]:
        if kwargs["perform_retirement"] is True:
            raise DatabasePortalBridgeDeferred(
                "simulated_dead_recovery_attempt"
            )
        return original(**kwargs)

    monkeypatch.setattr(
        bridge,
        "_retire_dead_protected_active_marker",
        fail_after_terminal,
    )
    with pytest.raises(DatabasePortalBridgeDeferred, match="simulated_dead"):
        bridge.run_provider(attempt)
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    assert active_path.is_file()
    abandoned_recovery = bridge._read_recovery_receipt(
        bridge._paths(attempt).root / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    assert abandoned_recovery["phase"] == "prepared"
    prepared_clearance = tuple(
        prior_root.glob("cross-attempt-protected-state-clearance-*.json")
    )
    assert len(prepared_clearance) == 1
    assert json.loads(
        prepared_clearance[0].read_text(encoding="utf-8")
    )["clearance_phase"] == "prepared"

    monkeypatch.setattr(
        bridge,
        "_retire_dead_protected_active_marker",
        original,
    )
    successor = replace(
        attempt,
        attempt_id="attempt:successor-3",
        claim_id="claim:successor-3",
        attempt_number=3,
        owner_session_id="session:successor-3",
        fencing_token=8,
        fence_epoch=4,
        lease_id="lease:successor-3",
    )
    result = bridge.run_provider(successor)

    assert result["accepted"] is True
    assert store.load_workspace(workspace) == terminal
    assert not active_path.exists()
    assert len(tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )) == 1
    adoption_paths = tuple(
        prior_root.glob("cross-attempt-protected-state-adoption-*.json")
    )
    assert len(adoption_paths) == 1
    adoption = json.loads(adoption_paths[0].read_text(encoding="utf-8"))
    assert adoption["schema"] == (
        bridge_module.CROSS_ATTEMPT_PROTECTED_STATE_ADOPTION_SCHEMA
    )
    assert adoption["abandoned_binding_id"] == abandoned_recovery[
        "current_binding_id"
    ]
    assert adoption["worker_self_approval"] is False
    assert portals and portals[-1].run_count == 1

    restart = replace(
        successor,
        attempt_id="attempt:successor-4",
        claim_id="claim:successor-4",
        attempt_number=4,
        owner_session_id="session:successor-4",
        fencing_token=9,
        fence_epoch=5,
        lease_id="lease:successor-4",
    )
    restart_result = bridge.run_provider(restart)

    assert restart_result["accepted"] is True
    assert len(tuple(
        prior_root.glob("cross-attempt-protected-state-adoption-*.json")
    )) == 1
    assert portals[-1].run_count == 1


@pytest.mark.parametrize("successor_number", [3, 4])
def test_bridge_distinct_successor_completes_preterminal_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    successor_number: int,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None and predecessor.is_nonterminal
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    original_finalize = store.finalize_exact_dead_owner

    def fail_before_cas(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated_preterminal_recovery_attempt_crash")

    monkeypatch.setattr(store, "finalize_exact_dead_owner", fail_before_cas)
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_finalize_race",
    ):
        bridge.run_provider(attempt)
    still_preterminal = store.load_workspace(workspace)
    assert still_preterminal == predecessor
    assert active_path.is_file()
    recovery_path = (
        bridge._paths(attempt).root
        / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    assert bridge._read_recovery_receipt(recovery_path)["phase"] == "prepared"
    assert len(tuple(
        prior_root.glob("cross-attempt-protected-state-clearance-*.json")
    )) == 1

    monkeypatch.setattr(store, "finalize_exact_dead_owner", original_finalize)
    successor = replace(
        attempt,
        attempt_id=f"attempt:successor-{successor_number}-preterminal",
        claim_id=f"claim:successor-{successor_number}-preterminal",
        attempt_number=successor_number,
        owner_session_id=f"session:successor-{successor_number}-preterminal",
        fencing_token=5 + successor_number,
        fence_epoch=1 + successor_number,
        lease_id=f"lease:successor-{successor_number}-preterminal",
    )
    result = bridge.run_provider(successor)

    assert result["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    assert not active_path.exists()
    assert bridge._read_recovery_receipt(recovery_path)["phase"] == "committed"
    assert len(tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )) == 1
    assert len(tuple(
        prior_root.glob("cross-attempt-protected-state-adoption-*.json")
    )) == 1
    assert portals and portals[-1].run_count == 1


@pytest.mark.parametrize("race", ["receipt", "directory"])
def test_bridge_successor_adoption_revalidates_abandoned_state_under_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    original_retire = bridge._retire_dead_protected_active_marker

    def fail_after_terminal(**kwargs: object) -> dict[str, object]:
        if kwargs["perform_retirement"] is True:
            raise DatabasePortalBridgeDeferred("simulated_dead_recovery_attempt")
        return original_retire(**kwargs)

    monkeypatch.setattr(
        bridge,
        "_retire_dead_protected_active_marker",
        fail_after_terminal,
    )
    with pytest.raises(DatabasePortalBridgeDeferred, match="simulated_dead"):
        bridge.run_provider(attempt)
    monkeypatch.setattr(
        bridge,
        "_retire_dead_protected_active_marker",
        original_retire,
    )
    abandoned_paths = bridge._paths(attempt)
    recovery_path = (
        abandoned_paths.root / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    portal_class = type(portals[-1])
    original_acquire = portal_class._acquire_checkout_mutation_lease
    raced = False
    displaced = tmp_path / "displaced-abandoned-attempt"

    def race_after_candidate_scan(**kwargs: object) -> tuple[object, str, None, float]:
        nonlocal raced
        if not raced:
            if race == "receipt":
                recovery_path.write_bytes(recovery_path.read_bytes() + b" ")
            else:
                abandoned_paths.root.rename(displaced)
                abandoned_paths.root.symlink_to(
                    displaced,
                    target_is_directory=True,
                )
            raced = True
        return original_acquire(**kwargs)

    monkeypatch.setattr(
        portal_class,
        "_acquire_checkout_mutation_lease",
        staticmethod(race_after_candidate_scan),
    )
    successor = replace(
        attempt,
        attempt_id=f"attempt:successor-adoption-race-{race}",
        claim_id=f"claim:successor-adoption-race-{race}",
        attempt_number=3,
        owner_session_id=f"session:successor-adoption-race-{race}",
        fencing_token=8,
        fence_epoch=4,
        lease_id=f"lease:successor-adoption-race-{race}",
    )

    with pytest.raises(DatabasePortalBridgeError):
        bridge.run_provider(successor)

    assert raced is True
    assert (prior_root / "implementation-protected-path-active.json").is_file()
    assert portals[-1].run_count == 0


def test_bridge_successor_adoption_rejects_multiple_incomplete_transactions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    original_finalize = store.finalize_exact_dead_owner

    def fail_before_cas(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated_preterminal_recovery_attempt_crash")

    monkeypatch.setattr(store, "finalize_exact_dead_owner", fail_before_cas)
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_finalize_race",
    ):
        bridge.run_provider(attempt)
    monkeypatch.setattr(store, "finalize_exact_dead_owner", original_finalize)
    first_paths = bridge._paths(attempt)
    first_recovery = bridge._read_recovery_receipt(
        first_paths.root / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    prior_paths = bridge._direct_attempt_paths(Path(predecessor.state_dir))
    prior_binding = bridge._strict_binding(prior_paths.binding)
    abandoned = replace(
        attempt,
        attempt_id="attempt:abandoned-3",
        claim_id="claim:abandoned-3",
        attempt_number=3,
        owner_session_id="session:abandoned-3",
        fencing_token=8,
        fence_epoch=4,
        lease_id="lease:abandoned-3",
    )
    abandoned_paths, abandoned_binding = bridge._ensure_attempt_projection(
        abandoned,
        _record(),
    )
    duplicate_body = {
        field: first_recovery[field]
        for field in bridge_module._RECOVERY_RECEIPT_FIELDS.difference(
            {"recovery_id", "receipt_id"}
        )
    }
    duplicate_body.update(
        {
            "current_attempt_id": abandoned_binding["attempt_id"],
            "current_attempt_number": abandoned.attempt_number,
            "current_binding_id": abandoned_binding["binding_id"],
            "current_fencing_token": abandoned_binding["fencing_token"],
            "database_authority": bridge._validated_prior_authority(
                bridge.prior_attempt_authority(
                    abandoned,
                    abandoned_binding,
                    prior_binding,
                ),
                current_binding=abandoned_binding,
                prior_binding=prior_binding,
            ),
        }
    )
    duplicate = bridge._seal_recovery_receipt(duplicate_body)
    (abandoned_paths.root / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME).write_text(
        json.dumps(duplicate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    successor = replace(
        abandoned,
        attempt_id="attempt:successor-4-ambiguous",
        claim_id="claim:successor-4-ambiguous",
        attempt_number=4,
        owner_session_id="session:successor-4-ambiguous",
        fencing_token=9,
        fence_epoch=5,
        lease_id="lease:successor-4-ambiguous",
    )

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_abandoned_recovery_ambiguous",
    ):
        bridge.run_provider(successor)

    assert portals[-1].run_count == 0


def test_bridge_revalidates_retired_marker_before_committed_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    original = bridge._preserved_quiescent_worktree
    calls = 0

    def tamper_after_retirement(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        preservation = original(*args, **kwargs)
        calls += 1
        if calls == 3:
            retired = tuple(
                prior_root.glob("implementation-protected-path-retired-*.json")
            )
            assert len(retired) == 1
            retired[0].write_text("tampered after retirement\n", encoding="utf-8")
        return preservation

    monkeypatch.setattr(
        bridge,
        "_preserved_quiescent_worktree",
        tamper_after_retirement,
    )
    with pytest.raises(DatabasePortalBridgeDeferred):
        bridge.run_provider(attempt)

    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    recovery = bridge._read_recovery_receipt(
        bridge._paths(attempt).root / CROSS_ATTEMPT_LIFECYCLE_RECOVERY_FILENAME
    )
    assert recovery["phase"] == "prepared"
    assert not portals or portals[0].run_count == 0


def test_bridge_noreplace_retirement_never_overwrites_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_owner_mutation,
    )

    bridge, attempt, store, workspace, portals = (
        _cross_attempt_recovery_fixture(
            tmp_path,
            protected_marker=True,
        )
    )
    predecessor = store.load_workspace(workspace)
    assert predecessor is not None
    prior_root = Path(predecessor.state_dir)
    active_path = prior_root / "implementation-protected-path-active.json"
    original = quack_owner_mutation._rename_noreplace_at
    raced_payload = b"racing destination must survive\n"

    def race(directory_fd: int, source: str, target: str) -> None:
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.write(descriptor, raced_payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        original(directory_fd, source, target)

    monkeypatch.setattr(quack_owner_mutation, "_rename_noreplace_at", race)
    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="protected_marker_retirement_failed",
    ):
        bridge.run_provider(attempt)

    raced = tuple(prior_root.glob("implementation-protected-path-retired-*.json"))
    assert len(raced) == 1 and raced[0].read_bytes() == raced_payload
    assert active_path.is_file()
    assert not tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )
    current = store.load_workspace(workspace)
    assert current is not None and current.is_terminal
    assert current.terminal_reason == "superseded_database_attempt_preserved"
    assert not portals or portals[0].run_count == 0


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


def _write_fake_user_manager_process(
    proc_root: Path,
    *,
    pid: int,
    name: str,
    comm: str,
    parent_pid: int,
    process_group: int,
    session: int,
    command: bytes,
) -> Path:
    process = proc_root / str(pid)
    process.mkdir()
    uid = os.geteuid()
    (process / "status").write_text(
        (
            f"Name:\t{name}\n"
            f"Pid:\t{pid}\n"
            f"PPid:\t{parent_pid}\n"
            f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n"
        ),
        encoding="ascii",
    )
    (process / "cmdline").write_bytes(command)
    (process / "cgroup").write_text(
        (
            f"0::/user.slice/user-{uid}.slice/"
            f"user@{uid}.service/init.scope\n"
        ),
        encoding="ascii",
    )
    stat_fields = [
        "S",
        str(parent_pid),
        str(process_group),
        str(session),
        *("0" for _ in range(15)),
        "424242",
    ]
    (process / "stat").write_text(
        f"{pid} {comm} {' '.join(stat_fields)}\n",
        encoding="ascii",
    )
    (process / "cwd").symlink_to("/")
    return process


@pytest.mark.parametrize("denied_errno", [errno.EACCES, errno.EPERM])
@pytest.mark.parametrize(
    "manager_command",
    [
        b"/usr/lib/systemd/systemd\0--user\0",
        b"/lib/systemd/systemd\0--user\0",
    ],
)
def test_process_scan_admits_only_exact_unreadable_user_manager_tuple(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    denied_errno: int,
    manager_command: bytes,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    manager_pid = 424_240
    manager = _write_fake_user_manager_process(
        proc_root,
        pid=manager_pid,
        name="systemd",
        comm="(systemd)",
        parent_pid=1,
        process_group=manager_pid,
        session=manager_pid,
        command=manager_command,
    )
    pam = _write_fake_user_manager_process(
        proc_root,
        pid=manager_pid + 1,
        name="(sd-pam)",
        comm="((sd-pam))",
        parent_pid=manager_pid,
        process_group=manager_pid,
        session=manager_pid,
        command=b"(sd-pam)\0",
    )
    original_readlink = os.readlink
    unreadable = {manager / "cwd", pam / "cwd"}

    def deny_manager_cwd(path: object, *args: object, **kwargs: object) -> str:
        if Path(path) in unreadable:
            raise PermissionError(denied_errno, "procfs cwd denied", str(path))
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", deny_manager_cwd)

    result = DatabasePortalExecutionBridge._strict_workspace_process_scan(
        SimpleNamespace(proc_root=proc_root),
        workspace,
    )

    assert result == {"same_uid_processes_inspected": 2}


@pytest.mark.parametrize(
    "tamper",
    ["command", "cgroup", "parent"],
)
def test_process_scan_rejects_unreadable_user_manager_near_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    pid = 424_250
    process = _write_fake_user_manager_process(
        proc_root,
        pid=pid,
        name="systemd",
        comm="(systemd)",
        parent_pid=1,
        process_group=pid,
        session=pid,
        command=b"/usr/lib/systemd/systemd\0--user\0",
    )
    if tamper == "command":
        (process / "cmdline").write_bytes(
            b"/usr/lib/systemd/systemd\0--user\0--deserialize=9\0"
        )
    elif tamper == "cgroup":
        uid = os.geteuid()
        (process / "cgroup").write_text(
            (
                f"0::/user.slice/user-{uid}.slice/"
                f"user@{uid}.service/app.slice\n"
            ),
            encoding="ascii",
        )
    else:
        status = (process / "status").read_text(encoding="ascii")
        (process / "status").write_text(
            status.replace("PPid:\t1\n", "PPid:\t2\n"),
            encoding="ascii",
        )
    original_readlink = os.readlink

    def deny_manager_cwd(path: object, *args: object, **kwargs: object) -> str:
        if Path(path) == process / "cwd":
            raise PermissionError(errno.EACCES, "procfs cwd denied", str(path))
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", deny_manager_cwd)

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_process_inventory_unavailable",
    ):
        DatabasePortalExecutionBridge._strict_workspace_process_scan(
            SimpleNamespace(proc_root=proc_root),
            workspace,
        )


def test_process_scan_exact_manager_never_masks_later_workspace_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    manager = _write_fake_user_manager_process(
        proc_root,
        pid=424_270,
        name="systemd",
        comm="(systemd)",
        parent_pid=1,
        process_group=424_270,
        session=424_270,
        command=b"/usr/lib/systemd/systemd\0--user\0",
    )
    worker = proc_root / "424271"
    worker.mkdir()
    (worker / "cwd").symlink_to(workspace)
    (worker / "cmdline").write_bytes(b"python3\0worker.py\0")
    original_readlink = os.readlink

    def deny_manager_cwd(path: object, *args: object, **kwargs: object) -> str:
        if Path(path) == manager / "cwd":
            raise PermissionError(errno.EACCES, "procfs cwd denied", str(path))
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", deny_manager_cwd)

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_worktree_process_active",
    ):
        DatabasePortalExecutionBridge._strict_workspace_process_scan(
            SimpleNamespace(proc_root=proc_root),
            workspace,
        )


def test_process_scan_exact_manager_with_readable_workspace_cwd_is_active(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    manager = _write_fake_user_manager_process(
        proc_root,
        pid=424_280,
        name="systemd",
        comm="(systemd)",
        parent_pid=1,
        process_group=424_280,
        session=424_280,
        command=b"/usr/lib/systemd/systemd\0--user\0",
    )
    (manager / "cwd").unlink()
    (manager / "cwd").symlink_to(workspace)

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_worktree_process_active",
    ):
        DatabasePortalExecutionBridge._strict_workspace_process_scan(
            SimpleNamespace(proc_root=proc_root),
            workspace,
        )


def test_process_scan_keeps_arbitrary_unreadable_process_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    process = _write_fake_user_manager_process(
        proc_root,
        pid=424_260,
        name="python3",
        comm="(python3)",
        parent_pid=1,
        process_group=424_260,
        session=424_260,
        command=b"python3\0worker.py\0",
    )
    original_readlink = os.readlink

    def deny_process_cwd(path: object, *args: object, **kwargs: object) -> str:
        if Path(path) == process / "cwd":
            raise PermissionError(errno.EACCES, "procfs cwd denied", str(path))
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", deny_process_cwd)

    with pytest.raises(
        DatabasePortalBridgeDeferred,
        match="cross_attempt_lifecycle_process_inventory_unavailable",
    ):
        DatabasePortalExecutionBridge._strict_workspace_process_scan(
            SimpleNamespace(proc_root=proc_root),
            workspace,
        )


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
        _cross_attempt_recovery_fixture(tmp_path, protected_marker=True)
    )
    inspected = 2

    def changing_process_inventory(
        _store: object,
        _workspace: Path,
    ) -> dict[str, int]:
        nonlocal inspected
        inspected += 1
        return {"same_uid_processes_inspected": inspected}

    monkeypatch.setattr(
        bridge,
        "_strict_workspace_process_scan",
        changing_process_inventory,
    )

    result = bridge.run_provider(attempt)

    assert result["accepted"] is True
    terminal = store.load_workspace(workspace)
    assert terminal is not None and terminal.is_terminal
    prior_root = Path(terminal.state_dir)
    assert len(tuple(
        prior_root.glob("cross-attempt-protected-state-clearance-*.json")
    )) == 1
    assert len(tuple(
        prior_root.glob("cross-attempt-protected-state-retirement-*.json")
    )) == 1
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


def test_bridge_immutable_publish_rejects_parent_swap_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = tmp_path / "sealed-attempt"
    sealed.mkdir()
    identity = sealed.stat()
    displaced = tmp_path / "displaced-attempt"
    attacker = tmp_path / "attacker-attempt"
    attacker.mkdir()
    target = sealed / "receipt.json"
    real_link = bridge_module.os.link
    swapped = False

    def swap_before_link(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        if not swapped:
            sealed.rename(displaced)
            sealed.symlink_to(attacker, target_is_directory=True)
            swapped = True
        real_link(*args, **kwargs)

    monkeypatch.setattr(bridge_module.os, "link", swap_before_link)

    with pytest.raises(
        DatabasePortalBridgeError,
        match="parent identity changed",
    ):
        bridge_module._publish_immutable_file(
            target,
            b"sealed receipt\n",
            sealed_directory_identity={
                "attempt_directory_device": int(identity.st_dev),
                "attempt_directory_inode": int(identity.st_ino),
            },
        )

    assert swapped is True
    assert (displaced / target.name).read_bytes() == b"sealed receipt\n"
    assert tuple(attacker.iterdir()) == ()


def test_bridge_immutable_publish_closes_parent_fd_on_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = tmp_path / "sealed-attempt"
    sealed.mkdir()
    identity = sealed.stat()
    target = sealed / "receipt.json"
    reads = 0

    def fail_final_verification(
        _directory_descriptor: int,
        _name: str,
        *,
        noun: str,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal reads
        assert noun == "immutable database Portal artifact"
        reads += 1
        if reads == 1:
            raise FileNotFoundError
        raise DatabasePortalBridgeError("simulated immutable verification failure")

    monkeypatch.setattr(
        bridge_module,
        "_stable_regular_bytes_at",
        fail_final_verification,
    )
    descriptors_before = len(tuple(Path("/proc/self/fd").iterdir()))

    with pytest.raises(
        DatabasePortalBridgeError,
        match="simulated immutable verification failure",
    ):
        bridge_module._publish_immutable_file(
            target,
            b"sealed receipt\n",
            sealed_directory_identity={
                "attempt_directory_device": int(identity.st_dev),
                "attempt_directory_inode": int(identity.st_ino),
            },
        )

    assert reads == 2
    assert len(tuple(Path("/proc/self/fd").iterdir())) == descriptors_before


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
