from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    DISTRIBUTED_LANE_REQUIREMENT_ID,
    BundleLaneSpec,
    DistributedLaneDispatcher,
    DistributedLaneWorker,
    evaluate_distributed_lane_evidence,
    immutable_lane_input_artifact,
)
from ipfs_accelerate_py.agent_supervisor.lease_coordination import (
    ExecutionScopeConflictError,
    LeaseCoordinator,
    RemoteLaneResult,
    WorkerCapabilityReceipt,
    WorkerEnvironmentReceipt,
)
from ipfs_accelerate_py.agent_supervisor.merge_train import MergeTrain


def _bundle(name: str, *, bundle_key: str | None = None) -> dict[str, object]:
    return {
        "bundle_key": bundle_key or f"objective/distributed/{name.lower()}",
        "parallel_lane": f"lane-{name.lower()}",
        "todo_path": f"{name}.todo.md",
        "source_todo": "source.todo.md",
        "is_schedulable": True,
        "review_only": False,
        "execution_slice_task_ids": [name],
        "tasks": [
            {
                "task_id": name,
                "title": f"Implement {name}",
                "status": "todo",
                "is_schedulable": True,
                "review_only": False,
            }
        ],
    }


def _lane(
    tmp_path: Path,
    task_cid: str,
    name: str,
    *,
    bundle_key: str | None = None,
    capabilities: tuple[str, ...] = ("python",),
) -> BundleLaneSpec:
    todo = tmp_path / f"{name}.todo.md"
    todo.write_text(f"## {name}\n\n- [ ] immutable work\n", encoding="utf-8")
    payload = _bundle(name, bundle_key=bundle_key)
    payload["required_environment"] = {"python": "3.12"}
    return BundleLaneSpec(
        bundle_key=str(payload["bundle_key"]),
        parallel_lane=f"lane-{name.lower()}",
        todo_path=todo,
        state_dir=tmp_path / name / "state",
        worktree_root=tmp_path / name / "worktrees",
        state_prefix=f"agent_{name.lower()}",
        task_ids=[name],
        conflict_policy="exact paths",
        command=["python", "-m", "worker", name],
        log_path=tmp_path / name / "worker.log",
        source_todo_sha256=hashlib.sha256(todo.read_bytes()).hexdigest(),
        task_cid=task_cid,
        required_capabilities=list(capabilities),
        queue_payload=payload,
        resource_class="cpu-medium",
    )


def _worker(
    worker_id: str,
    execute,
    *,
    now: int = 1_000,
    revision: str = "worker@1",
    capabilities: tuple[str, ...] = ("python",),
) -> DistributedLaneWorker:
    capability = WorkerCapabilityReceipt(
        worker_id=worker_id,
        capabilities=capabilities,
        issued_at_ms=now - 100,
        expires_at_ms=now + 20_000,
        capability_revision=revision,
    )
    environment = WorkerEnvironmentReceipt(
        worker_id=worker_id,
        environment_id=f"env:{revision}",
        capability_receipt_id=capability.receipt_id,
        issued_at_ms=now - 100,
        expires_at_ms=now + 20_000,
        attributes={"python": "3.12"},
    )
    return DistributedLaneWorker(
        worker_id=worker_id,
        capability_receipt=capability,
        environment_receipt=environment,
        execute=execute,
    )


def _successful_result(dispatch, artifact, worker_id: str, *, commit: str = "a" * 40):
    return RemoteLaneResult(
        repository_id=dispatch.repository_id,
        worker_id=worker_id,
        task_cid=dispatch.task_cid,
        artifact_id=artifact.artifact_id,
        candidate_commit=commit,
        capability_receipt_id=dispatch.capability_receipt_cid,
        environment_receipt_id=dispatch.environment_receipt_cid,
        claim_cid=dispatch.grant.claim_cid,
        logical_epoch=dispatch.logical_epoch,
        fencing_token=dispatch.fencing_token,
        output={
            "branch_name": f"distributed/{worker_id}",
            "validation_receipt_ids": ["validation:passed"],
        },
        created_at_ms=1_001,
    )


def test_immutable_artifact_and_local_fallback_are_deterministic(
    tmp_path: Path,
) -> None:
    now = [1_000]
    bundle = _bundle("LOCAL")
    with LeaseCoordinator(tmp_path / "coordination.duckdb", clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(bundle, created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "LOCAL")
        first = immutable_lane_input_artifact(
            lane, repository_id="repository:test", created_at_ms=1_000
        )
        second = immutable_lane_input_artifact(
            lane, repository_id="repository:test", created_at_ms=1_000
        )
        assert first == second
        assert first.artifact_id == second.artifact_id
        lane.todo_path.write_text("changed after planning", encoding="utf-8")
        with pytest.raises(ValueError, match="planned immutable input"):
            immutable_lane_input_artifact(
                lane, repository_id="repository:test", created_at_ms=1_000
            )

    # Worker selection does not depend on advertisement order.
    noop = lambda *_args: {}
    workers = [_worker("worker-z", noop), _worker("worker-a", noop)]
    with LeaseCoordinator(tmp_path / "selection.duckdb", clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(_bundle("SELECT"), created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "SELECT")
        dispatcher = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=noop,
            remote_workers=reversed(workers),
            clock_ms=lambda: now[0],
        )
        assert dispatcher._eligible_workers(lane, now_ms=now[0])[0].worker_id == "worker-a"


def test_remote_publication_is_fenced_and_handed_to_merge_train(
    tmp_path: Path,
) -> None:
    now = [1_000]
    observed = []

    def execute(dispatch, artifact, _cancel):
        time.sleep(0.03)
        return _successful_result(dispatch, artifact, "worker-a")

    def merge_submit(request):
        observed.append(request)
        return {
            "accepted": True,
            "candidate_commit": request.commit_sha,
            "post_merge_evidence_passed": True,
            "post_merge_validation": {"passed": True},
            "target_commit": request.commit_sha,
        }

    with LeaseCoordinator(tmp_path / "coordination.duckdb", clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(_bundle("REMOTE"), created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "REMOTE")
        grant = coordinator.claim(lane.task_cid, "did:web:scheduler", requested_lease_ms=5_000)
        dispatcher = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=lambda *_args: pytest.fail("unexpected local fallback"),
            remote_workers=[_worker("worker-a", execute)],
            merge_submit=merge_submit,
            clock_ms=lambda: now[0],
            heartbeat_interval=0.01,
        )
        execution = dispatcher.execute(lane, grant)
        heartbeat = coordinator.latest_heartbeat(lane.task_cid)

    assert execution.execution_mode == "remote"
    assert execution.disposition == "accepted"
    assert heartbeat is not None
    assert heartbeat["provider_id"] == "worker-a"
    assert len(observed) == 1
    publication = observed[0].metadata["distributed_publication"]
    assert publication["artifact_id"] == execution.artifact_id
    assert publication["fencing_token"] == grant.fencing_token
    assert publication["digest"].startswith("sha256:")
    assert publication["digest"] == MergeTrain.distributed_publication_digest(
        publication
    )


def test_absent_or_incompatible_remote_capacity_falls_back_locally(
    tmp_path: Path,
) -> None:
    now = [1_000]
    calls: list[str] = []

    def local_execute(dispatch, artifact, _cancel):
        calls.append(dispatch.worker_id)
        return _successful_result(dispatch, artifact, "local")

    incompatible = _worker(
        "worker-no-python",
        lambda *_args: pytest.fail("incompatible worker ran"),
        capabilities=("rust",),
    )
    with LeaseCoordinator(tmp_path / "coordination.duckdb", clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(_bundle("FALLBACK"), created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "FALLBACK")
        grant = coordinator.claim(lane.task_cid, "did:web:scheduler", requested_lease_ms=5_000)
        dispatcher = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=local_execute,
            remote_workers=[incompatible],
            merge_submit=lambda request: {
                "accepted": True,
                "candidate_commit": request.commit_sha,
                "post_merge_evidence_passed": True,
            },
            clock_ms=lambda: now[0],
        )
        execution = dispatcher.execute(lane, grant)

    assert calls == ["local"]
    assert execution.execution_mode == "local"
    assert execution.fallback_reason == "remote_capacity_unavailable"


def test_partition_worker_loss_cancels_without_ambiguous_local_retry(
    tmp_path: Path,
) -> None:
    now = [1_000]
    local_calls = []

    def partitioned(*_args):
        raise ConnectionError("partition")

    with LeaseCoordinator(tmp_path / "coordination.duckdb", clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(_bundle("PARTITION"), created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "PARTITION")
        grant = coordinator.claim(lane.task_cid, "did:web:scheduler", requested_lease_ms=5_000)
        dispatcher = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=lambda *_args: local_calls.append(True),
            remote_workers=[_worker("worker-a", partitioned)],
            clock_ms=lambda: now[0],
        )
        execution = dispatcher.execute(lane, grant)
        replacement = coordinator.claim(
            lane.task_cid, "did:web:replacement", requested_lease_ms=5_000
        )

    assert execution.disposition == "worker_lost"
    assert not local_calls
    assert replacement.fencing_token == grant.fencing_token + 1


def test_lease_theft_and_capability_drift_are_quarantined(
    tmp_path: Path,
) -> None:
    now = [1_000]
    replacement_grants = []

    with LeaseCoordinator(tmp_path / "theft.duckdb", clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(_bundle("THEFT"), created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "THEFT")
        grant = coordinator.claim(lane.task_cid, "did:web:old", requested_lease_ms=5_000)

        def stolen(dispatch, artifact, _cancel):
            now[0] = 6_001
            replacement_grants.append(
                coordinator.steal(
                    lane.task_cid,
                    "did:web:new",
                    requested_lease_ms=5_000,
                )
            )
            return _successful_result(dispatch, artifact, "worker-a")

        dispatcher = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=lambda *_args: {},
            remote_workers=[_worker("worker-a", stolen)],
            clock_ms=lambda: now[0],
            heartbeat_interval=10,
        )
        stolen_execution = dispatcher.execute(lane, grant)
        assert stolen_execution.disposition == "quarantined"
        assert replacement_grants[0].fencing_token == grant.fencing_token + 1
        assert not stolen_execution.merge_result

    now[0] = 1_000
    with LeaseCoordinator(tmp_path / "drift.duckdb", clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(_bundle("DRIFT"), created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "DRIFT")
        grant = coordinator.claim(lane.task_cid, "did:web:scheduler", requested_lease_ms=5_000)
        dispatcher: DistributedLaneDispatcher

        def drifted(dispatch, artifact, _cancel):
            dispatcher.set_remote_workers(
                [
                    _worker(
                        "worker-a",
                        lambda *_args: {},
                        revision="worker@2",
                    )
                ]
            )
            return _successful_result(dispatch, artifact, "worker-a")

        dispatcher = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=lambda *_args: {},
            remote_workers=[_worker("worker-a", drifted)],
            clock_ms=lambda: now[0],
        )
        drift_execution = dispatcher.execute(lane, grant)
        assert drift_execution.disposition == "quarantined"
        assert drift_execution.publication["reason"] == "capability_drift"


@pytest.mark.parametrize(
    ("raw_result", "reason"),
    [
        ({"schema": "foreign/result@1"}, "malformed_result:ValueError"),
        (
            {
                "task_cid": "task:foreign",
                "candidate_commit": "a" * 40,
                "output": {"branch_name": "distributed/foreign"},
            },
            "foreign_task_cid",
        ),
    ],
)
def test_malformed_and_foreign_results_are_quarantined(
    tmp_path: Path,
    raw_result: dict[str, object],
    reason: str,
) -> None:
    now = [1_000]
    case = reason.replace(":", "-")
    with LeaseCoordinator(
        tmp_path / f"{case}.duckdb", clock_ms=lambda: now[0]
    ) as coordinator:
        registered = coordinator.register_bundle(
            _bundle(f"REJECT-{case}"), created_at_ms=900
        )
        lane = _lane(tmp_path, registered["task_cid"], f"REJECT-{case}")
        grant = coordinator.claim(
            lane.task_cid,
            "did:web:scheduler",
            requested_lease_ms=5_000,
        )
        execution = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=lambda *_args: pytest.fail("unexpected local fallback"),
            remote_workers=[
                _worker("worker-a", lambda *_args: dict(raw_result))
            ],
            merge_submit=lambda _request: pytest.fail(
                "quarantined result reached merge train"
            ),
            clock_ms=lambda: now[0],
        ).execute(lane, grant)

        quarantined = coordinator.list_distributed_publications(
            lane.task_cid, disposition="quarantined"
        )

    assert execution.disposition == "quarantined"
    assert execution.publication["reason"] == reason
    assert len(quarantined) == 1
    assert quarantined[0]["quarantined"] is True


def test_duplicate_restart_conflict_and_cancellation_remain_fenced(
    tmp_path: Path,
) -> None:
    now = [1_000]
    path = tmp_path / "coordination.duckdb"
    bundle = _bundle("RESTART", bundle_key="objective/shared-scope")
    captured = {}

    def execute(dispatch, artifact, _cancel):
        result = _successful_result(dispatch, artifact, "worker-a")
        captured.update(dispatch=dispatch, result=result)
        return result

    with LeaseCoordinator(path, clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(bundle, created_at_ms=900)
        lane = _lane(
            tmp_path,
            registered["task_cid"],
            "RESTART",
            bundle_key="objective/shared-scope",
        )
        grant = coordinator.claim(lane.task_cid, "did:web:scheduler", requested_lease_ms=5_000)
        conflicting = coordinator.register_bundle(
            _bundle("CONFLICT", bundle_key="objective/shared-scope"),
            created_at_ms=901,
        )
        with pytest.raises(ExecutionScopeConflictError):
            coordinator.claim(
                conflicting["task_cid"],
                "did:web:other",
                requested_lease_ms=5_000,
            )
        dispatcher = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=lambda *_args: {},
            remote_workers=[_worker("worker-a", execute)],
            merge_submit=lambda request: {
                "accepted": True,
                "candidate_commit": request.commit_sha,
                "post_merge_evidence_passed": True,
            },
            clock_ms=lambda: now[0],
        )
        execution = dispatcher.execute(lane, grant)
        assert execution.accepted

    with LeaseCoordinator(path, clock_ms=lambda: now[0]) as restarted:
        duplicate = restarted.publish_remote_result(
            captured["dispatch"],
            captured["result"],
            current_capability_receipt=_worker(
                "worker-a", lambda *_args: {}
            ).capability_receipt,
            current_environment_receipt=_worker(
                "worker-a", lambda *_args: {}
            ).environment_receipt,
            now_ms=now[0],
        )
        assert duplicate["duplicate"] is True

    cancel_path = tmp_path / "cancel.duckdb"
    ran = []
    with LeaseCoordinator(cancel_path, clock_ms=lambda: now[0]) as coordinator:
        registered = coordinator.register_bundle(_bundle("CANCEL"), created_at_ms=900)
        lane = _lane(tmp_path, registered["task_cid"], "CANCEL")
        grant = coordinator.claim(lane.task_cid, "did:web:scheduler", requested_lease_ms=5_000)
        cancellation = threading.Event()
        cancellation.set()
        execution = DistributedLaneDispatcher(
            coordinator,
            repository_id="repository:test",
            local_executor=lambda *_args: ran.append(True),
            clock_ms=lambda: now[0],
        ).execute(lane, grant, cancel_event=cancellation)
        assert execution.disposition == "cancelled"
        assert not ran


def test_distributed_evidence_requires_complete_post_merge_bindings() -> None:
    task = {"canonical_task_cid": "task:1", "task_id": "ASI-113"}
    receipt = evaluate_distributed_lane_evidence(
        repository_tree="tree:accepted",
        task_population=[task],
        effects=[{**task, "paths": ["bundle_supervisor.py"]}],
        resources=[{**task, "resource_class": "cpu-medium"}],
        ownership=[
            {
                **task,
                "claim_cid": "claim:1",
                "logical_epoch": 1,
                "fencing_token": 1,
                "input_artifact_id": "artifact:1",
                "capability_receipt_id": "capability:1",
                "environment_receipt_id": "environment:1",
            }
        ],
        validation=[
            {**task, "passed": True, "receipt_ids": ["validation:1"]}
        ],
        terminal_results=[
            {**task, "accepted": True, "candidate_commit": "abc123"}
        ],
    )
    assert receipt.proved_requirement_ids_for("tree:accepted") == (
        DISTRIBUTED_LANE_REQUIREMENT_ID,
    )
    assert not receipt.proved_requirement_ids_for("tree:foreign")
    assert not type(receipt).from_dict(receipt.to_dict()).proved_requirement_ids_for(
        "tree:accepted"
    )

    incomplete = evaluate_distributed_lane_evidence(
        repository_tree="tree:accepted",
        task_population=[task],
        effects=[],
        resources=[],
        ownership=[],
        validation=[],
        terminal_results=[],
    )
    assert incomplete.failure_codes
    assert not incomplete.proved_requirement_ids_for("tree:accepted")
