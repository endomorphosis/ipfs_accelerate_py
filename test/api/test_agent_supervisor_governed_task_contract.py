"""Focused fail-closed contracts for governed PCCE task execution."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ImplementationProviderRouter,
    ProductionContractPacket,
    ProviderRoutingError,
    RouteStatus,
    ProviderRole,
    build_production_contract_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.governed_task_contract import (
    build_governed_validation_receipt,
    build_supervisor_task_receipt,
    task_authority_partition,
    task_phase_commands,
    verify_governed_validation_receipt,
    verify_supervisor_task_receipt,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.production_reviewed_effect import (
    production_task_contract,
)


PRODUCT = "src/product.py"
RECEIPT = "artifacts/proof_carrying_context_engine/receipts/PCCE-001.json"
TASK_CID = "cidv1:task-revision:pcce-001"
REPOSITORY_ID = "git-repository:test"
COMMIT = "a" * 40
TREE = "git-tree:" + "b" * 40
EXECUTION_PLAN_CID = content_identity({"execution_plan": "PCCE-016-r6"})
FOREST = [
    {
        "commit": COMMIT,
        "present": True,
        "repository": "control",
        "repository_id": REPOSITORY_ID,
        "repository_root": ".",
        "tree_id": TREE,
        "workspace_clean": True,
    }
]


def _command(*argv: str, command_id: str = "focused") -> dict[str, object]:
    return {
        "argv": list(argv),
        "cwd": ".",
        "env": {},
        "id": command_id,
        "repository": "control",
        "repository_root": ".",
        "timeout_seconds": 120,
    }


def _command_set(*commands: dict[str, object]) -> str:
    return json.dumps(
        {
            "commands": list(commands),
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "governed-phase-command-set@1"
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _task(**changes):
    metadata = {
        "Objective": "Inventory the canonical implementations",
        "Acceptance criteria": "All inventory rows cite code and test evidence",
        "Owning repository": "endomorphosis/ipfs_accelerate_py",
        "Allowed effects": PRODUCT,
        "Prohibited effects": "task/control receipts; protected branch writes",
        "Required evidence": RECEIPT,
        "Required tests": "focused inventory contract tests",
        "Execution mode": "model-assisted-production-packet-route",
        "Executor kind": "patch_job",
        "Risk classification": "high",
        "Rollback procedure": "revert the exact candidate commit",
        "Provider effects": PRODUCT,
        "Supervisor outputs": RECEIPT,
    }
    payload = {
        "task_id": "PCCE-001",
        "title": "Discover completed subsystem implementations",
        "status": "ready",
        "completion": "supervisor",
        "priority": "P0",
        "track": "pcce",
        "depends_on": [],
        "outputs": [PRODUCT, RECEIPT],
        "validation": [f"python -m py_compile {PRODUCT}"],
        "acceptance": "Inventory is supported by code and tests",
        "metadata": metadata,
    }
    payload.update(changes)
    return PortalTask(**payload)


def _identity(task):
    return canonical_task_identity(task, board_namespace="pcce").to_dict()


def test_provider_effects_and_supervisor_outputs_exactly_partition_outputs() -> None:
    partition = task_authority_partition(_task())
    assert partition.provider_effects == (PRODUCT,)
    assert partition.supervisor_outputs == (RECEIPT,)
    assert partition.outputs == (PRODUCT, RECEIPT)

    forged = _task(
        metadata={
            **_task().metadata,
            "Provider effects": f"{PRODUCT}, {RECEIPT}",
            "Supervisor outputs": "",
        }
    )
    try:
        task_authority_partition(forged)
    except ValueError as exc:
        assert "supervisor receipt" in str(exc)
    else:  # pragma: no cover - an authority widening is never acceptable
        raise AssertionError("provider was allowed to author the task receipt")


def test_frozen_contract_and_both_provider_roles_receive_complete_contract() -> None:
    task = _task()
    contract = production_task_contract(task, _identity(task))
    assert contract["outputs"] == [PRODUCT, RECEIPT]
    assert contract["authority_partition"]["provider_effects"] == [PRODUCT]
    assert contract["intent"]["required_evidence"] == RECEIPT
    assert contract["intent"]["acceptance_criteria"] == (
        "All inventory rows cite code and test evidence"
    )
    assert contract["intent"]["owning_repository"] == (
        "endomorphosis/ipfs_accelerate_py"
    )
    assert contract["executor_kind"] == "patch_job"
    packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id="git-commit:" + COMMIT,
        write_paths=[PRODUCT],
        validation_commands=task.validation,
        acceptance_criteria=task.acceptance,
        task_contract=contract,
        independent_review_required_for_write=True,
    )
    requests = []

    def grok(request):
        requests.append(request)
        return {
            "proposal": {
                "declared_paths": [PRODUCT],
                "files": [{"path": PRODUCT, "content": "VALUE = 1\n"}],
            }
        }

    def codex(request):
        requests.append(request)
        return {"decision": "approve", "findings": []}

    result = ImplementationProviderRouter(
        grok_provider=grok,
        codex_provider=codex,
        admission_gate=lambda proposal: {
            "accepted": True,
            "reason_code": f"admitted:{proposal.role.value}",
        },
    ).route(
        packet,
        current_snapshot_id=packet.snapshot_id,
        apply=False,
    )
    assert result.provider_result_admitted is True
    implement_contract = requests[0].payload["contract_packet"]["task_contract"]
    review_contract = requests[1].payload["evidence_slice"]["task_contract"]
    assert implement_contract == contract
    assert review_contract == contract
    assert requests[0].role is ProviderRole.GROK_IMPLEMENT
    assert requests[1].role is ProviderRole.CODEX_REVIEW


def test_intent_contract_ignores_projection_state_but_readdresses_semantics() -> None:
    task = _task()
    first = production_task_contract(task, _identity(task))
    projection_update = _task(
        status="completed",
        completion="merged",
        metadata={
            **task.metadata,
            "Status": "completed",
            "Assigned worktree": "/tmp/worktree",
            "Final result CID or artifact identity": "bafy-final",
        },
    )
    assert production_task_contract(projection_update, _identity(projection_update)) == first

    semantic_update = _task(
        metadata={**task.metadata, "Required tests": "a different exact gate"}
    )
    second = production_task_contract(semantic_update, _identity(semantic_update))
    assert second != first
    assert content_identity(second) != content_identity(first)


def test_governed_phase_commands_reject_shell_git_and_nested_roots() -> None:
    valid = _task(
        metadata={
            **_task().metadata,
            "Pre-change validation": _command_set(
                _command("python", "-m", "pytest", "-q")
            ),
            "Pre-change validation policy": "require-pass",
        }
    )
    [spec] = task_phase_commands(valid, "pre_change")
    assert spec["argv"] == ["python", "-m", "pytest", "-q"]

    for argv, root, repository in (
        (("git", "reset", "--hard"), ".", "control"),
        (("python", "-m", "pytest", "&&", "git", "push"), ".", "control"),
        (
            ("python", "-m", "pytest"),
            "ipfs_accelerate_py/mcplusplus",
            "endomorphosis/Mcp-Plus-Plus",
        ),
    ):
        command = _command(*argv)
        command["repository_root"] = root
        command["repository"] = repository
        task = _task(
            metadata={
                **_task().metadata,
                "Pre-change validation": _command_set(command),
                "Pre-change validation policy": "require-pass",
            }
        )
        with pytest.raises(ValueError):
            task_phase_commands(task, "pre_change")


def test_production_packet_cannot_disable_independent_review_gate() -> None:
    task = _task()
    contract = production_task_contract(task, _identity(task))
    with pytest.raises(ProviderRoutingError):
        build_production_contract_packet(
            task_id=task.task_id,
            snapshot_id="git-commit:" + COMMIT,
            write_paths=[PRODUCT],
            task_contract=contract,
            independent_review_required_for_write=False,
        )

    packet = build_production_contract_packet(
        task_id=task.task_id,
        snapshot_id="git-commit:" + COMMIT,
        write_paths=[PRODUCT],
        task_contract=contract,
    )
    payload = dict(packet.payload)
    payload["authority"] = {
        **dict(payload["authority"]),
        "independent_review_required_for_write": False,
    }
    forged = ProductionContractPacket(
        packet_id=packet.packet_id,
        snapshot_id=packet.snapshot_id,
        task_id=packet.task_id,
        payload=payload,
    )
    calls: list[str] = []
    result = ImplementationProviderRouter(
        grok_provider=lambda _request: calls.append("grok"),
        codex_provider=lambda _request: calls.append("codex"),
    ).route(forged, current_snapshot_id=forged.snapshot_id, apply=True)
    assert result.status is RouteStatus.REJECTED
    assert calls == []


def test_record_baseline_requires_exact_failure_signature_and_rejects_forgery() -> None:
    raw = {
        "attempted": True,
        "passed": False,
        "returncode": 7,
        "validated_commit": COMMIT,
        "stale": False,
        "failed_tests": ["tests/test_known_red.py::test_red"],
        "failure_head": "assert 1 == 2",
        "forest_before": FOREST,
        "forest_after": FOREST,
        "results": [
            {
                "id": "known-red",
                "returncode": 7,
                "normalized_output_cid": content_identity(
                    {"normalized_output": "assert 1 == 2"}
                ),
                "failed_test_ids": ["tests/test_known_red.py::test_red"],
                "output": "secret raw command prose",
            }
        ],
    }
    commands = [
        _command(
            "python",
            "-m",
            "pytest",
            "tests/test_known_red.py",
            "-q",
            command_id="known-red",
        )
    ]
    observed = build_governed_validation_receipt(
        phase="pre_change",
        task_id="PCCE-016",
        task_cid="cidv1:task-revision:pcce-016",
        task_contract_cid=content_identity({"contract": "pcce-016"}),
        execution_plan_cid=EXECUTION_PLAN_CID,
        target_commit=COMMIT,
        repository_tree_id=TREE,
        repository_id=REPOSITORY_ID,
        commands=commands,
        validation_result=raw,
        policy="record-baseline",
    )
    assert observed["admitted"] is False
    assert "secret raw command prose" not in str(observed)

    expected = dict(observed["baseline_failure_signature"])
    admitted = build_governed_validation_receipt(
        phase="pre_change",
        task_id="PCCE-016",
        task_cid="cidv1:task-revision:pcce-016",
        task_contract_cid=content_identity({"contract": "pcce-016"}),
        execution_plan_cid=EXECUTION_PLAN_CID,
        target_commit=COMMIT,
        repository_tree_id=TREE,
        repository_id=REPOSITORY_ID,
        commands=commands,
        validation_result=raw,
        policy="record-baseline",
        expected_baseline_failure=expected,
    )
    assert admitted["admitted"] is True
    assert verify_governed_validation_receipt(admitted)[0] is True

    forged = dict(observed)
    forged["admitted"] = True
    unsigned = dict(forged)
    unsigned.pop("receipt_cid")
    forged["receipt_cid"] = content_identity(unsigned)
    verified, reasons = verify_governed_validation_receipt(forged)
    assert verified is False
    assert "governed_validation_derived_fields_mismatch" in reasons


def test_supervisor_receipt_cid_and_dependency_identity_fail_closed() -> None:
    task = _task()
    contract = production_task_contract(task, _identity(task))
    post = {"validation_receipt_id": "cidv1:post-change"}
    gate = {"admitted": True, "completion_authoritative": True}
    receipt = build_supervisor_task_receipt(
        task_contract=contract,
        task_contract_cid=content_identity(contract),
        completion_generation="r6",
        baseline_commit=COMMIT,
        baseline_tree_id=TREE,
        candidate_commit="c" * 40,
        candidate_tree_id="git-tree:" + "d" * 40,
        integration_commit="e" * 40,
        integration_tree_id="git-tree:" + "f" * 40,
        effect_identities=[{"path": PRODUCT, "blob_id": "git-blob:123"}],
        dependency_evidence=[],
        pre_change_receipt=None,
        post_change_receipt=post,
        acceptance_receipt=None,
        provider_evidence={"provider_execution_receipt_id": "cidv1:provider"},
        completion_gate=gate,
    )
    assert verify_supervisor_task_receipt(
        receipt,
        task_id="PCCE-001",
        task_cid=contract["canonical_task_cid"],
    )[0] is True

    stale = dict(receipt)
    stale["integration_commit"] = "0" * 40
    assert verify_supervisor_task_receipt(
        stale,
        task_id="PCCE-001",
        task_cid=contract["canonical_task_cid"],
    )[0] is False
