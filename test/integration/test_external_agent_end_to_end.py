"""EAAEF-145: fail-closed in-process end-to-end contract qualification.

Composes handoff → plan admit → frontier → typed Quack owner boundary →
recovery.  Until the canonical owner dispatcher is admitted, the composed run
must remain nonterminal.  A live eight-container cluster is not invoked or
claimed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.api.external_handoff import ExternalHandoffAPI
from ipfs_accelerate_py.agent_supervisor.planning.external_frontier import (
    FrontierTask,
    select_frontier,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_goal_contract import (
    ExternalGoalContract,
)
from ipfs_accelerate_py.agent_supervisor.planning.external_work_plan import ExternalWorkPlan
from ipfs_accelerate_py.agent_supervisor.planning.plan_admission import admit_plan
from ipfs_accelerate_py.agent_supervisor.runtime.external_control_recovery import recover
from ipfs_accelerate_py.agent_supervisor.runtime.external_fixed_point import terminate
from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
    ExternalQuackOwner,
    RetiredInMemoryOwnerError,
    issue_envelope,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    QuackStateServer,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_daemon_gateway import (
    QuackDaemonGatewayError,
)

WRITE_A = "ipfs_accelerate_py/agent_supervisor/handoff/adapters/codex.py"
WRITE_B = "ipfs_accelerate_py/agent_supervisor/handoff/adapters/claude.py"
SOURCE_ROOT = "sha256:" + "c" * 64
SEMANTIC_ROOT = "sha256:" + "d" * 64
BOARD_NAMESPACE = "external-agent-autonomous-execution-fabric-v1"
SHARD_ID = "eaaef-145-disposable-end-to-end-shard"
STORE_ID = "eaaef-145-control"
RECEIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/architecture/external_agent_autonomous_execution_fabric/receipts/end_to_end.json"
)
REQUIRED_HERMETIC_STAGES = (
    "handoff",
    "plan_admit",
    "frontier",
    "typed_quack_owner_boundary",
    "recovery",
    "remains_nonterminal",
)


def _goal():
    return ExternalGoalContract.compile(
        {
            "objective_id": "EAAEF-G150",
            "desired_outcomes": ("normalize export", "preserve identities"),
            "prohibited_outcomes": ("self_approve", "hidden_chain_of_thought"),
            "write_scope": (WRITE_A, WRITE_B),
            "authority_ceiling": "preview_only",
            "verification_requirements": ("focused pytest",),
            "proof_requirements": ("content identity",),
            "review_requirements": ("independent supervisor",),
            "completion_evidence": ("test receipt", "patch identity"),
            "timeout_seconds": 7200,
            "cpu_millicores": 4000,
            "ram_mib": 8192,
        }
    )


def _task(task_id: str, covers, write_scope, **overrides):
    payload = {
        "task_id": task_id,
        "covers": covers,
        "write_scope": write_scope,
        "depends_on": (),
        "timeout_seconds": 600,
        "cpu_millicores": 1000,
        "ram_mib": 1024,
    }
    payload.update(overrides)
    return payload


def _server(root: Path) -> QuackStateServer:
    return build_server(
        database_path=root / "control.duckdb",
        state_dir=root / "owner",
        port=0,
        repository_id="repository:eaaef-145-test",
        store_id=STORE_ID,
        secret_handle="handle:eaaef-145-test-owner",
        transport=FakeQuackTransport(),
    )


def _fixture_cid(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _is_evidence_cid(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _validate_current_receipt(payload: object) -> None:
    assert isinstance(payload, Mapping)
    assert payload.get("schema") == "qualification-receipt@1"
    assert payload.get("task_id") == "EAAEF-145"
    encoded = json.dumps(payload, sort_keys=True)
    assert "in_memory_ExternalQuackOwner" not in encoded
    assert payload.get("evidence_mode") == "observed_hermetic"
    assert payload.get("live_runtime_invoked") is True
    assert payload.get("live_eight_container_qualification") is False
    assert payload.get("live_quack_contacted") is False

    owner_evidence = payload.get("owner_evidence")
    assert isinstance(owner_evidence, Mapping)
    assert owner_evidence.get("interface") == "ExternalQuackOwner@1"
    assert owner_evidence.get("backing_owner_interface") == "QuackStateServer@1"
    assert str(owner_evidence.get("server_id") or "").startswith("server:")
    assert _is_evidence_cid(owner_evidence.get("lease_cid"))
    assert owner_evidence.get("production_admitted") is False
    blockers = list(owner_evidence.get("production_blockers") or ())
    assert EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER in blockers

    stages = payload.get("stages")
    assert isinstance(stages, list)
    observed_stage_names: list[str] = []
    for stage in stages:
        assert isinstance(stage, Mapping), (
            "end-to-end stages must carry observations, not assertion strings"
        )
        stage_name = str(stage.get("stage") or "")
        assert stage_name
        assert stage.get("observed") is True
        assert stage.get("outcome") == "passed"
        assert _is_evidence_cid(stage.get("evidence_cid"))
        observed_stage_names.append(stage_name)

    assert "quack_owner_apply" not in observed_stage_names
    assert observed_stage_names == list(REQUIRED_HERMETIC_STAGES)
    assert payload.get("owner_dispatch_admitted") is False
    assert payload.get("completion_claimed") is False
    assert payload.get("qualification_status") == "fail_closed_owner_facade_observed"
    assert payload.get("result") == "pass"
    assert payload.get("terminal") == "completed"
    assert list(payload.get("production_blockers") or ()) == blockers


def _truthful_hermetic_receipt_fixture() -> dict[str, object]:
    server_id = "server:observed-hermetic-owner"
    blockers = [EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER]
    return {
        "schema": "qualification-receipt@1",
        "task_id": "EAAEF-145",
        "evidence_mode": "observed_hermetic",
        "live_runtime_invoked": True,
        "live_quack_contacted": False,
        "live_eight_container_qualification": False,
        "owner_dispatch_admitted": False,
        "completion_claimed": False,
        "terminal": "completed",
        "qualification_status": "fail_closed_owner_facade_observed",
        "result": "pass",
        "production_blockers": blockers,
        "owner_evidence": {
            "interface": "ExternalQuackOwner@1",
            "backing_owner_interface": "QuackStateServer@1",
            "server_id": server_id,
            "lease_cid": _fixture_cid("eaaef-145-hermetic-owner-lease"),
            "production_admitted": False,
            "production_blockers": blockers,
        },
        "stages": [
            {
                "stage": stage,
                "observed": True,
                "outcome": "passed",
                "evidence_cid": _fixture_cid(f"eaaef-145:{stage}"),
            }
            for stage in REQUIRED_HERMETIC_STAGES
        ],
    }


def test_receipt_validator_accepts_only_observed_hermetic_fixture() -> None:
    truthful = _truthful_hermetic_receipt_fixture()
    _validate_current_receipt(truthful)

    archived_strings = json.loads(json.dumps(truthful))
    archived_strings["stages"] = [
        "handoff",
        "plan_admit",
        "frontier",
        "quack_owner_apply",
        "recovery",
        "typed_terminal",
    ]
    with pytest.raises(AssertionError):
        _validate_current_receipt(archived_strings)

    legacy = json.loads(json.dumps(truthful))
    legacy["owner"] = "in_memory_ExternalQuackOwner"
    with pytest.raises(AssertionError):
        _validate_current_receipt(legacy)

    forged_live = json.loads(json.dumps(truthful))
    forged_live["live_eight_container_qualification"] = True
    forged_live["owner_dispatch_admitted"] = True
    with pytest.raises(AssertionError):
        _validate_current_receipt(forged_live)


def test_handoff_plan_frontier_owner_recovery_remains_nonterminal(
    tmp_path: Path,
) -> None:
    api = ExternalHandoffAPI()
    started = api.handoff(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "reviewer_principal_id": "principal:reviewer",
            "session_id": "session:e2e",
            "repository_id": "repo:e2e",
            "objective_id": "objective:e2e",
            "idempotency_key": "idem:e2e-1",
        }
    )
    assert started.verdict == "admitted"
    assert started.run_status == "running"

    sequential = ExternalWorkPlan.decompose(
        _goal(),
        (
            _task("task-a", ("normalize export",), (WRITE_A,)),
            _task(
                "task-b",
                ("preserve identities",),
                (WRITE_B,),
                depends_on=("task-a",),
            ),
        ),
    )
    parallel = ExternalWorkPlan.decompose(
        _goal(),
        (
            _task("task-a", ("normalize export",), (WRITE_A,)),
            _task("task-b", ("preserve identities",), (WRITE_B,)),
        ),
    )
    admitted = admit_plan((sequential, parallel))
    assert admitted.to_dict()["verdict"] == "admitted"
    assert admitted.admitted.content_id == parallel.content_id

    frontier = select_frontier(
        (
            FrontierTask("task-a", (), (WRITE_A,), ("write-a",), 1000),
            FrontierTask("task-b", (), (WRITE_B,), ("write-b",), 1000),
        ),
        cpu_budget=4000,
    )
    assert set(frontier["task_ids"]) == {"task-a", "task-b"}

    server = _server(tmp_path)
    identity = server.start()
    try:
        owner = server.bind_external_quack_owner(
            board_namespace=BOARD_NAMESPACE,
            shard_id=SHARD_ID,
        )
        assert isinstance(owner, ExternalQuackOwner)
        lease = owner.lease()
        assert lease.owner_id == identity.server_id
        assert owner.assert_current(lease) == lease
        assert owner.production_admitted is False
        with pytest.raises(RetiredInMemoryOwnerError) as retired:
            issue_envelope(
                operation="put",
                key="e2e-run",
                value={"run_id": started.run_id, "plan_id": admitted.admitted_id},
                principal_id="principal:operator",
                idempotency_key="idem:e2e-apply",
            )
        assert retired.value.reason_code == "in_memory_owner_retired"
        with pytest.raises(
            QuackDaemonGatewayError,
            match=EXTERNAL_QUACK_OWNER_PRODUCTION_BLOCKER,
        ):
            owner.daemon_gateway()

        recovery = recover(
            current_epoch=lease.epoch,
            backup_epoch=lease.epoch,
            duplicate=False,
            ducklake_available=False,
        )
        assert recovery["accepted_stale_write"] is False
    finally:
        server.stop()

    approved = api.approve(
        {
            "principal_id": "principal:operator",
            "worker_principal_id": "principal:worker",
            "reviewer_principal_id": "principal:reviewer",
            "run_id": started.run_id,
            "authority_id": started.authority_id,
            "session_id": "session:e2e",
        }
    )
    assert approved.run_status == "approved"

    terminal = terminate(
        goals_complete=False,
        tests_current=True,
        proofs_current=True,
        invalidations_empty=True,
        merge_queue_empty=True,
        claims_empty=True,
        source_root=SOURCE_ROOT,
        semantic_root=SEMANTIC_ROOT,
    )
    assert terminal["terminal"] == "not_complete"


def test_board_declared_qualification_receipt_is_current() -> None:
    assert RECEIPT_PATH.is_file(), f"EAAEF-145 board-declared receipt is missing: {RECEIPT_PATH}"
    _validate_current_receipt(json.loads(RECEIPT_PATH.read_text(encoding="utf-8")))
