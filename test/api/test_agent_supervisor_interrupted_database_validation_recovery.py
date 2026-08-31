from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor as portal_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)


def _task() -> PortalTask:
    return PortalTask(
        task_id="PCTDD-031",
        title="resume exact controller validation",
        status="todo",
        completion="auto",
        priority="P1",
        track="implementation",
        validation=["python -m pytest -q focused.py"],
    )


def _evidence() -> dict[str, Any]:
    return {
        "evidence_id": "sha256:" + "a" * 64,
        "reconciliation_receipt": {
            "nested_state": {
                "active_task_id": "PCTDD-031",
                "active_attempt": 1,
                "active_branch": "implementation/pctdd-031-attempt-1",
            }
        },
    }


def _daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_mode: str,
) -> tuple[PortalImplementationDaemon, list[str]]:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.state_path = tmp_path / "portal-state.json"
    PortalTaskState().save(daemon.state_path)
    task = _task()
    workspace = tmp_path / "isolated-worktree"
    workspace.mkdir()
    calls: list[str] = []
    lease = object()

    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda event_type, _payload: calls.append(f"event:{event_type}"),
    )
    monkeypatch.setattr(daemon, "_iter_merge_lifecycle_events", lambda: [])

    def acquire(**_kwargs: Any) -> tuple[object, str, None, float]:
        calls.append("lease:acquire")
        return lease, "acquired", None, 0.0

    def release(observed: object) -> bool:
        assert observed is lease
        calls.append("lease:release")
        return True

    def authority(
        _evidence_value: dict[str, Any],
        *,
        state: PortalTaskState,
    ) -> dict[str, Any]:
        assert not state.implementation_in_progress
        assert calls[-1] == "lease:acquire"
        calls.append("authority")
        return {
            "ok": True,
            "task": task,
            "workspace_path": workspace,
        }

    candidate = {
        "prepared": True,
        "task": task,
        "task_id": task.task_id,
        "candidate_branch": "rescue/worktree/pctdd-031",
        "baseline_ref": "1" * 40,
        "candidate_commit": "2" * 40,
        "candidate_authority_id": "baguqeera-candidate",
        "changed_submodule_paths": ["external/ipfs_accelerate"],
        "acceptance_inferred": False,
        "provider_dispatched": False,
        "attempt_consumed": False,
    }

    def prepare(
        _authority: dict[str, Any],
        *,
        state: PortalTaskState,
    ) -> dict[str, Any]:
        assert not state.implementation_in_progress
        calls.append("candidate:prepare")
        return dict(candidate)

    def release_claim(
        _state: PortalTaskState,
        *,
        unfinished_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert unfinished_candidate == candidate
        assert calls[-1] == "candidate:prepare"
        calls.append("claim:release")
        return {"reconciled": True, "blocked": False}

    replacement_claim = {"lease_id": "replacement-lease"}

    def build_claim(
        _task_value: PortalTask,
        _attempt: int,
        _started_at: str,
    ) -> dict[str, Any]:
        calls.append("claim:build-replacement")
        return dict(replacement_claim)

    def acquire_claim(
        _path: Path,
        metadata: dict[str, Any],
    ) -> tuple[bool, str, None]:
        assert metadata == replacement_claim
        calls.append("claim:acquire-replacement")
        return True, "acquired", None

    def validate(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["worktree_path"] == workspace
        assert kwargs["task"] is task
        assert kwargs["changed_submodule_paths"] == [
            "external/ipfs_accelerate"
        ]
        assert kwargs["preacquired_task_claim"] == replacement_claim
        calls.append("controller:validate")
        merged = validation_mode == "merged"
        queued = validation_mode == "queued"
        return {
            "returncode": 0 if merged else 1,
            "provider_dispatched": False,
            "attempt_consumed": False,
            "merge_result": {
                "merged": merged,
                "queued": queued,
                "reason": (
                    "merged" if merged else "queued" if queued else "not_attempted"
                ),
            },
            "validation_result": {
                "attempted": True,
                "passed": merged or queued,
            },
        }

    monkeypatch.setattr(daemon, "_acquire_checkout_mutation_lease", acquire)
    monkeypatch.setattr(daemon, "_release_checkout_mutation_lease", release)
    monkeypatch.setattr(
        daemon, "_interrupted_database_validation_authority", authority
    )
    monkeypatch.setattr(
        daemon, "_prepare_interrupted_database_validation_candidate", prepare
    )
    monkeypatch.setattr(
        daemon, "_reconcile_quiesced_implementation_task_claim", release_claim
    )
    monkeypatch.setattr(
        daemon, "_build_implementation_task_claim_metadata", build_claim
    )
    monkeypatch.setattr(
        daemon,
        "_identity_for_task",
        lambda _task_value: SimpleNamespace(
            canonical_task_cid="baguqeera-task"
        ),
    )
    monkeypatch.setattr(
        daemon,
        "_implementation_task_claim_path",
        lambda _task_id, canonical_task_cid: (
            tmp_path / f"{canonical_task_cid}.claim"
        ),
    )
    monkeypatch.setattr(
        daemon, "_try_acquire_implementation_task_claim", acquire_claim
    )
    monkeypatch.setattr(
        daemon, "reconcile_validated_worktree_candidate", validate
    )
    return daemon, calls


def test_interrupted_database_validation_resumes_existing_controller_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="merged")

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["provider_dispatched"] is False
    assert result["attempt_consumed"] is False
    assert calls == [
        "lease:acquire",
        "authority",
        "candidate:prepare",
        "claim:release",
        "claim:build-replacement",
        "claim:acquire-replacement",
        "lease:release",
        "controller:validate",
        "event:interrupted_database_validation_reconciled",
    ]


def test_interrupted_database_validation_failure_is_typed_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="failed")

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert result["reason"] == "interrupted_database_validation_rejected"
    assert result["controller_validation_terminal"] is True
    assert result["durable_merge_handoff"] is False
    assert result["provider_dispatched"] is False
    assert result["attempt_consumed"] is False
    assert "controller:validate" in calls


def test_interrupted_database_validation_accepts_durable_queue_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="queued")

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert result["reason"] == "interrupted_database_validation_queued"
    assert result["controller_validation_terminal"] is False
    assert result["durable_merge_handoff"] is True
    assert calls.index("claim:acquire-replacement") < calls.index(
        "lease:release"
    )


def test_interrupted_database_validation_replays_exact_queued_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, calls = _daemon(tmp_path, monkeypatch, validation_mode="queued")
    queued_validation = {
        "returncode": 1,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "merge_result": {
            "merged": False,
            "queued": True,
            "reason": "queued",
        },
        "validation_result": {"attempted": True, "passed": True},
    }
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [
            {
                "type": "interrupted_database_validation_reconciled",
                "event_id": "sha256:" + "b" * 64,
                "reason": "interrupted_database_validation_queued",
                "task_id": "PCTDD-031",
                "canonical_task_cid": "baguqeera-task",
                "attempt": 1,
                "database_evidence_id": "sha256:" + "a" * 64,
                "candidate_authority_id": "baguqeera-candidate",
                "candidate_commit": "2" * 40,
                "provider_dispatched": False,
                "attempt_consumed": False,
                "durable_merge_handoff": True,
                "controller_validation": queued_validation,
            }
        ],
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [_task()])

    result = daemon.reconcile_interrupted_database_validation_attempt(
        _evidence()
    )

    assert result["reconciled"] is True
    assert result["replayed"] is True
    assert result["durable_merge_handoff"] is True
    assert "lease:acquire" not in calls
    assert "controller:validate" not in calls


def test_database_recovery_identity_rejects_non_json_values() -> None:
    assert PortalImplementationDaemon._database_recovery_sha256(
        {"path": Path("not-json")}
    ) == ""
    assert PortalImplementationDaemon._database_recovery_sha256(
        {"number": float("nan")}
    ) == ""


def test_interrupted_validation_authority_admits_only_exact_fenced_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.state_path = tmp_path / "attempt" / "portal-state.json"
    daemon.state_path.parent.mkdir()
    daemon.worktree_root = tmp_path / "worktrees"
    workspace = daemon.worktree_root / "isolated"
    workspace.mkdir(parents=True)
    task = _task()
    identity = SimpleNamespace(
        canonical_task_key="task/v1/exact",
        canonical_task_cid="baguqeera-task",
        board_namespace="exact-board",
    )
    original_branch = "implementation/pctdd-031-attempt-1"
    baseline = "1" * 40
    current_head = "2" * 40
    state = PortalTaskState(
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_worktree_path=str(workspace),
    )
    record = SimpleNamespace(
        is_terminal=True,
        terminal_reason="controlled_restart_dead_owner",
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=1,
        branch=original_branch,
        workspace_path=str(workspace),
        record_id="lifecycle-record",
        fence=5,
    )
    daemon.worktree_lifecycle = SimpleNamespace(
        load_workspace=lambda _workspace: record
    )
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    monkeypatch.setattr(daemon, "_identity_for_task", lambda _task: identity)
    monkeypatch.setattr(daemon, "_canonical_ref", lambda _task: identity.canonical_task_cid)
    monkeypatch.setattr(daemon, "_is_git_worktree", lambda _path: True)
    monkeypatch.setattr(
        daemon,
        "_git_current_branch",
        lambda _path: "rescue/worktree/implementation-pctdd-031-attempt-1-a",
    )
    monkeypatch.setattr(
        daemon,
        "_resolved_commit_ref",
        lambda _path, ref: baseline if ref == baseline else current_head,
    )
    monkeypatch.setattr(
        daemon,
        "_git_ref_is_ancestor_in_repo",
        lambda _path, ancestor, descendant: (
            ancestor == baseline and descendant == current_head
        ),
    )

    binding = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-binding@1",
        "interface": "DatabasePortalExecutionBridge@1",
        "attempt_id": "attempt-1",
        "claim_id": "claim-1",
        "task_cid": "outer-task-cid",
        "task_alias": task.task_id,
        "goal_cid": "goal",
        "plan_cid": "plan",
        "task_revision": 3,
        "fencing_token": 7,
        "fence_epoch": 2,
        "lease_id": "outer-lease",
        "task_body_digest": "sha256:" + "3" * 64,
        "projection_seed_digest": "sha256:" + "4" * 64,
        "projection_immutable_digest": "sha256:" + "5" * 64,
        "authoritative_task_store": "duckdb",
        "projection_authority": False,
    }
    binding["binding_id"] = daemon._database_recovery_sha256(binding)
    portal = {
        "blocked": True,
        "reconciled": False,
        "reason": "task_claim_reconciliation_blocked",
        "attempt_recovery": {
            "canonical_task_cid": identity.canonical_task_cid
        },
        "protected_path_reconciliation": {
            "blocked": False,
            "reason": "crash_reconciliation_unchanged",
            "workspace_path": str(workspace),
        },
        "worktree_lifecycle_reconciliation": {
            "blocked": False,
            "reconciled": True,
            "state": "terminal",
            "workspace_path": str(workspace),
            "attempt": 1,
            "record_id": record.record_id,
            "fence": record.fence,
        },
        "task_claim_reconciliation": {
            "blocked": True,
            "reason": "canonical_task_not_terminal",
        },
    }
    nested = {
        "active": True,
        "active_phase": "validating",
        "active_task_id": task.task_id,
        "active_attempt": 1,
        "active_worktree_path": str(workspace),
        "active_branch": original_branch,
        "active_phase_detail": "; ".join(task.validation),
    }
    receipt = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-attempt-reconciliation@1",
        "stage": "blocked",
        "binding_id": binding["binding_id"],
        "attempt_root": str(daemon.state_path.parent),
        "task_alias": task.task_id,
        "task_cid": binding["task_cid"],
        "attempt_id": binding["attempt_id"],
        "claim_id": binding["claim_id"],
        "fencing_token": binding["fencing_token"],
        "fence_epoch": binding["fence_epoch"],
        "nested_state": nested,
        "portal_reconciliation": portal,
        "provider_runner_fence": {
            "applicable": True,
            "fenced": True,
            "safe_to_restart": True,
            "reason": "ordinary_provider_runner_exact_birth_fenced",
        },
    }
    receipt["receipt_id"] = daemon._database_recovery_sha256(receipt)
    evidence = {
        "schema": "ipfs_accelerate_py/agent-supervisor/database-portal-interrupted-validation-recovery@1",
        "binding": binding,
        "recovery_identity": "sha256:" + "6" * 64,
        "equivalent_receipt_ids": [receipt["receipt_id"]],
        "reconciliation_receipt": receipt,
    }
    evidence["evidence_id"] = daemon._database_recovery_sha256(evidence)
    start_event = {
        "type": "implementation_started",
        "event_id": "start-event",
        "sequence": 1,
        "task_id": task.task_id,
        "canonical_task_cid": identity.canonical_task_cid,
        "attempt": 1,
        "worktree_path": str(workspace),
        "branch": original_branch,
        "baseline_ref": baseline,
        "workspace_setup": {"base_commit": baseline},
    }
    recovery_event = {
        **portal,
        "type": "implementation_shutdown_reconciliation_blocked",
        "event_id": "recovery-event",
        "sequence": 2,
        "task_id": task.task_id,
        "attempt": 1,
    }
    harmless_outer_preflight = {
        "type": "worktree_reconciliation_validation_finished",
        "sequence": 3,
        "task_id": task.task_id,
        "attempt": 1,
        "attempt_consumed": False,
        "provider_dispatched": False,
        "returncode": 1,
        "worktree_path": str(workspace),
        "baseline_ref": baseline,
        "validation_result": {
            "attempted": False,
            "passed": False,
            "reason": "reconciliation_validation_exception",
            "error": "reconciled candidate worktree is not clean",
        },
        "merge_result": {"merged": False, "reason": "not_attempted"},
    }
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [start_event, recovery_event, harmless_outer_preflight],
    )

    authority = daemon._interrupted_database_validation_authority(
        evidence,
        state=state,
    )

    assert authority["ok"] is True
    assert authority["baseline_ref"] == baseline
    assert authority["database_evidence_id"] == evidence["evidence_id"]

    poisoned = dict(evidence)
    poisoned["recovery_identity"] = Path("not-json")
    assert daemon._interrupted_database_validation_authority(
        poisoned,
        state=state,
    )["reason"] == "database_recovery_evidence_identity_invalid"


def test_bridge_coalesces_duplicate_recovery_receipts_without_count_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = DatabasePortalExecutionBridge.__new__(
        DatabasePortalExecutionBridge
    )
    bridge.attempt_root = tmp_path / "attempts"
    attempt = SimpleNamespace(attempt_id="attempt-1")
    paths = bridge._paths(attempt)
    paths.reconciliation.mkdir(parents=True)
    binding = {"task_alias": "PCTDD-031", "binding_id": "binding-1"}
    workspace = str(tmp_path / "worktree")
    matching: dict[str, dict[str, Any]] = {}
    for index in range(132):
        receipt_id = f"sha256:{index:064x}"
        (paths.reconciliation / f"{index:064x}.json").touch()
        if index >= 130:
            matching[receipt_id] = {
                "stage": "blocked",
                "receipt_id": receipt_id,
                "blocked": True,
                "reconciled": False,
                "reason": "nested_portal_attempt_reconciliation_blocked",
                "binding_id": binding["binding_id"],
                "task_alias": binding["task_alias"],
                "terminal_provider_evidence": False,
                "provider_runner_reconciliation_authority": (
                    "ordinary_provider_runner_fence"
                ),
                "nested_state": {
                    "active": True,
                    "active_phase": "validating",
                    "active_task_id": binding["task_alias"],
                    "active_attempt": 1,
                    "active_worktree_path": workspace,
                    "active_branch": "implementation/pctdd-031-attempt-1",
                    "state_path": str(paths.state),
                    "state_digest": "sha256:" + "7" * 64,
                },
                "provider_runner_fence": {
                    "applicable": True,
                    "fenced": True,
                    "safe_to_restart": True,
                    "reason": "ordinary_provider_runner_exact_birth_fenced",
                },
                "portal_reconciliation": {
                    "blocked": True,
                    "reconciled": False,
                    "reason": "task_claim_reconciliation_blocked",
                    "protected_path_reconciliation": {
                        "blocked": False,
                        "reason": "crash_reconciliation_unchanged",
                        "task_id": binding["task_alias"],
                        "workspace_path": workspace,
                    },
                    "worktree_lifecycle_reconciliation": {
                        "blocked": False,
                        "reconciled": True,
                        "state": "terminal",
                        "task_id": binding["task_alias"],
                        "workspace_path": workspace,
                        "attempt": 1,
                        "record_id": "record-1",
                        "fence": 5,
                    },
                    "task_claim_reconciliation": {
                        "blocked": True,
                        "reconciled": False,
                        "reason": "canonical_task_not_terminal",
                        "task_id": binding["task_alias"],
                        "canonical_task_cid": "baguqeera-task",
                    },
                    "attempt_recovery": {
                        "consumed": False,
                        "attempt": 1,
                        "task_id": binding["task_alias"],
                    },
                },
            }
    prepared_id = "sha256:" + "f" * 64
    (paths.reconciliation / ("f" * 64 + ".json")).touch()

    def load(
        _attempt: Any,
        receipt_id: str,
        *,
        required_stage: str = "",
    ) -> dict[str, Any]:
        assert required_stage == ""
        if receipt_id == prepared_id:
            return {"stage": "prepared", "receipt_id": receipt_id}
        return matching.get(
            receipt_id,
            {
                "stage": "blocked",
                "receipt_id": receipt_id,
                "blocked": True,
                "reconciled": False,
                "reason": "unrelated",
            },
        )

    monkeypatch.setattr(bridge, "load_reconciliation_receipt", load)

    evidence = bridge._interrupted_validation_recovery_evidence(
        attempt,
        binding,
    )

    assert evidence is not None
    assert len(evidence["equivalent_receipt_ids"]) == 2
    assert evidence["reconciliation_receipt"]["receipt_id"] == min(
        matching
    )


def test_bridge_reconciles_more_than_128_exact_immutable_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = SimpleNamespace(
        attempt_id="attempt-1",
        claim_id="claim-1",
        task_cid="task-cid-1",
        task_alias="PCTDD-031",
        attempt_number=1,
        owner_session_id="owner-1",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease-1",
        body={},
    )
    received_evidence: list[dict[str, Any]] = []

    class ExactPortal:
        def reconcile_quiesced_active_attempt(self) -> dict[str, Any]:
            raise AssertionError("exact interrupted-validation evidence was ignored")

        def reconcile_interrupted_database_validation_attempt(
            self,
            evidence: dict[str, Any],
        ) -> dict[str, Any]:
            received_evidence.append(evidence)
            return {"reconciled": True, "blocked": False}

        def close(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: ExactPortal(),
    )
    bridge.attempt_root.mkdir()
    paths = bridge._paths(attempt)
    paths.reconciliation.mkdir(parents=True)
    paths.binding.write_text("{}\n", encoding="utf-8")
    paths.task_projection.write_text("projection\n", encoding="utf-8")
    paths.state.write_text("{}\n", encoding="utf-8")
    state_digest = "sha256:" + "7" * 64
    workspace = str(tmp_path / "isolated-worktree")
    binding = {
        "attempt_id": attempt.attempt_id,
        "task_alias": attempt.task_alias,
        "binding_id": "sha256:" + "8" * 64,
        "projection_immutable_digest": "sha256:" + "9" * 64,
    }
    nested_state = {
        "present": True,
        "state_path": str(paths.state),
        "state_digest": state_digest,
        "active": True,
        "active_task_id": attempt.task_alias,
        "active_attempt": 1,
        "active_phase": "validating",
        "active_phase_detail": "python -m pytest -q focused.py",
        "active_worktree_path": workspace,
        "active_branch": "implementation/pctdd-031-attempt-1",
    }

    def receipt_payload(index: int, *, digest: str = state_digest) -> dict[str, Any]:
        return {
            "stage": "blocked",
            "trigger": f"duplicate-{index}",
            "reconciled_at": f"2026-08-31T00:00:{index:03d}Z",
            "reconciled": False,
            "blocked": True,
            "reason": "nested_portal_attempt_reconciliation_blocked",
            "binding_id": binding["binding_id"],
            "nested_state": {
                "active": True,
                "active_phase": "validating",
                "active_task_id": attempt.task_alias,
                "active_attempt": 1,
                "active_worktree_path": workspace,
                "active_branch": "implementation/pctdd-031-attempt-1",
                "state_path": str(paths.state),
                "state_digest": digest,
            },
            "provider_runner_fence": {
                "applicable": True,
                "fenced": True,
                "safe_to_restart": True,
                "reason": "ordinary_provider_runner_exact_birth_fenced",
            },
            "provider_runner_reconciliation_authority": (
                "ordinary_provider_runner_fence"
            ),
            "portal_reconciliation": {
                "blocked": True,
                "reconciled": False,
                "reason": "task_claim_reconciliation_blocked",
                "protected_path_reconciliation": {
                    "blocked": False,
                    "reason": "crash_reconciliation_unchanged",
                    "task_id": attempt.task_alias,
                    "workspace_path": workspace,
                },
                "worktree_lifecycle_reconciliation": {
                    "blocked": False,
                    "reconciled": True,
                    "state": "terminal",
                    "task_id": attempt.task_alias,
                    "workspace_path": workspace,
                    "attempt": 1,
                    "record_id": "record-1",
                    "fence": 5,
                },
                "task_claim_reconciliation": {
                    "blocked": True,
                    "reconciled": False,
                    "reason": "canonical_task_not_terminal",
                    "task_id": attempt.task_alias,
                    "canonical_task_cid": "baguqeera-task",
                },
                "attempt_recovery": {
                    "consumed": False,
                    "attempt": 1,
                    "task_id": attempt.task_alias,
                },
            },
            "terminal_provider_evidence": False,
        }

    for index in range(132):
        bridge.persist_reconciliation_receipt(
            attempt,
            receipt_payload(index),
        )

    monkeypatch.setattr(
        bridge,
        "_record_for_attempt",
        lambda _source, _attempt: object(),
    )
    monkeypatch.setattr(
        bridge,
        "_render_projection",
        lambda _attempt, _record: "projection\n",
    )
    monkeypatch.setattr(
        bridge,
        "_binding",
        lambda _attempt, _record, _seed: dict(binding),
    )
    monkeypatch.setattr(bridge, "_read_binding", lambda _path: dict(binding))
    monkeypatch.setattr(bridge, "_verify_binding_identity", lambda _binding: None)
    monkeypatch.setattr(
        bridge,
        "_verify_projection",
        lambda _paths, _binding: "projection\n",
    )
    monkeypatch.setattr(
        bridge,
        "_projection_task_identity",
        lambda _paths, _binding, _projection: {
            "task_id": attempt.task_alias,
            "canonical_task_key": "task/v1/exact",
            "canonical_task_cid": "baguqeera-task",
            "board_namespace": "exact-board",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_strict_state_record",
        lambda _path: ({"active_phase": "validating"}, state_digest),
    )
    monkeypatch.setattr(
        bridge,
        "_verify_nested_state_identity",
        lambda _paths, _binding, _identity, **_kwargs: dict(nested_state),
    )
    bridge._binding_lookup = lambda _attempt: {**binding, "stage": "portal_entered"}
    monkeypatch.setattr(bridge, "recover_provider_result", lambda _attempt: None)
    monkeypatch.setattr(
        portal_supervisor,
        "fence_ordinary_provider_runner",
        lambda _state, *, grace_seconds: {
            "applicable": True,
            "safe_to_restart": True,
            "fenced": True,
            "reason": "ordinary_provider_runner_exact_birth_fenced",
        },
    )

    result = bridge.reconcile_quiesced_attempt(attempt)

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert len(received_evidence) == 1
    assert len(received_evidence[0]["equivalent_receipt_ids"]) == 132

    malformed = paths.reconciliation / "not-content-addressed.json"
    malformed.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        DatabasePortalBridgeError,
        match="reconciliation evidence store is not exact",
    ):
        bridge.reconcile_quiesced_attempt(attempt)
    malformed.unlink()

    bridge.persist_reconciliation_receipt(
        attempt,
        receipt_payload(999, digest="sha256:" + "a" * 64),
    )
    with pytest.raises(
        DatabasePortalBridgeError,
        match="interrupted validation evidence is ambiguous",
    ):
        bridge.reconcile_quiesced_attempt(attempt)
