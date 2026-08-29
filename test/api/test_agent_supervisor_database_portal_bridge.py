"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_repository_id,
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
    DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA_V1,
    DATABASE_PORTAL_RECONCILED_SOURCE_TRANSITION_SCHEMA,
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


def test_source_transition_binds_attempt_board_repository_and_exact_merge(
    tmp_path: Path,
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
    assert transition["schema"] == DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA
    assert transition["baseline_ref"] == baseline
    assert transition["implementation_commit"] == implementation
    assert transition["merge_commit"] == merge_commit
    assert transition["target_repository_id"] == checkout_repository_id(repository)
    assert transition["task_completion_authority"] is False
    assert transition["worker_self_approval"] is False
    assert str(transition["transition_cid"]).startswith("sha256:")

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
        "schema": DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
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
