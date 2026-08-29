"""SPAR operator residual-admission receipts (planner/doctor/obligation/logic/repair)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.residual_authority import (
    REQUIRED_AUTHORITY_RECEIPT_KINDS,
    SPAR_BOARD_NAMESPACE,
    bind_spar_residual_authority,
    mint_authority_receipts,
    mint_spar_residual_authority_materials,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.launch_source_amendment import (
    LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA,
    LaunchSourceAmendment,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
    TaskExecutionRouteBinding,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import TaskSourceTask
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_kernel import (
    REASON_RESIDUAL_AUTHORIZED,
    evaluate_pre_implementation,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_provider_gate import (
    assert_provider_dispatch_allowed,
    evaluate_provider_gate,
)


def _cid(name: str) -> str:
    return content_identity({"fixture": name})


def _git_oid(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _spar_launch_amendment(*, head: str, tree: str) -> LaunchSourceAmendment:
    source_forest_body = {
        "source_head": head,
        "nested_repositories": [],
        "cross_repository_writes": False,
    }
    source_forest_root = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(source_forest_body)).hexdigest()
    )
    source_forest = {**source_forest_body, "source_forest_root": source_forest_root}
    forest_receipt_body = {
        "schema": LAUNCH_SOURCE_FOREST_RECEIPT_SCHEMA,
        "source_head": head,
        "repository_tree": tree,
        "source_forest_root": source_forest_root,
        "source_forest": source_forest,
    }
    forest_receipt_id = (
        "sha256:"
        + hashlib.sha256(canonical_json_bytes(forest_receipt_body)).hexdigest()
    )
    return LaunchSourceAmendment(
        board_namespace=SPAR_BOARD_NAMESPACE,
        plan_alias="SPAR-PLAN-R1",
        bootstrap_receipt_id=_cid("bootstrap-receipt"),
        bootstrap_plan_root_cid=_cid("bootstrap-plan-root"),
        bootstrap_source_head=head,
        bootstrap_repository_tree_id=tree,
        launch_source_forest_receipt_id=forest_receipt_id,
        launch_source_forest_root=source_forest_root,
        launch_source_forest_receipt={
            **forest_receipt_body,
            "receipt_id": forest_receipt_id,
        },
        launch_source_head=head,
        launch_repository_tree_id=tree,
        immutable_objectives_cid=_cid("objectives"),
        immutable_plan_cid=_cid("plan"),
        immutable_taskboard_cid=_cid("taskboard"),
        immutable_validator_cid=_cid("validator"),
        bootstrap_config_cid=_cid("bootstrap-config"),
        launch_config_cid=_cid("launch-config"),
        dependency_seal_cid=_cid("dependency-seal"),
        task_contract_set_cid=_cid("task-contract-set"),
        parent_plan_revision=2,
        amended_plan_revision=3,
    )


def test_typed_receipts_authorize_residual_kernel() -> None:
    task_cid = _cid("spar-007")
    forest_cid = _cid("spar-forest")
    receipts = mint_authority_receipts(
        task_cid=task_cid,
        repository_forest_cid=forest_cid,
    )
    assert tuple(receipts) == REQUIRED_AUTHORITY_RECEIPT_KINDS
    receipt_cids = {kind: item["content_id"] for kind, item in receipts.items()}
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
        ImplementationForestRoots,
    )

    forest = ImplementationForestRoots(
        repository_id=f"repository:{SPAR_BOARD_NAMESPACE}",
        repository_forest_cid=forest_cid,
        git_tree_id=_git_oid("tree"),
        policy_root=_cid("policy"),
    )
    result = evaluate_pre_implementation(
        {
            "task_cid": task_cid,
            "forest_roots": forest,
            "residual_packet_cid": _cid("packet"),
            "obligation_graph_cid": receipt_cids["obligation"],
            "plan_cid": receipt_cids["planner"],
            "doctor_cid": receipt_cids["doctor"],
            "authority_receipt_cids": receipt_cids,
        },
        authority_receipt_resolver=lambda cid: next(
            (item for item in receipts.values() if item["content_id"] == cid),
            None,
        ),
    )
    assert result.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    assert result.reason_code == REASON_RESIDUAL_AUTHORIZED
    assert result.authorizes_provider


def test_spar_bundle_binds_launch_forest_and_authorizes_provider(tmp_path: Path) -> None:
    head = _git_oid("head")
    tree = _git_oid("tree")
    amendment = _spar_launch_amendment(head=head, tree=tree)
    task_cid = _cid("spar-007-task")
    task = PortalTask(
        task_id="SPAR-007",
        title="Build the typed static program and refactoring graph",
        status="retrying",
        completion="auto",
        priority="P0",
        track="program-graph",
        outputs=[
            "ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/program_graph.py",
            "ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py",
        ],
        validation=[
            "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py"
        ],
        canonical_task_cid=task_cid,
        board_namespace=SPAR_BOARD_NAMESPACE,
    )
    route = TaskExecutionRouteBinding(
        policy_id=_cid("route-policy"),
        plan_root_cid=amendment.bootstrap_plan_root_cid,
        repository_tree_id=amendment.bootstrap_repository_tree_id,
        source_revision=1,
        task_cid=task_cid,
        task_alias=task.task_id,
        task_revision=1,
        task_contract_cid=_cid("task-contract"),
        execution_mode="grok-codex",
    )
    policy_root = amendment.attempt_policy_root(route.to_dict())
    materials = mint_spar_residual_authority_materials(
        task=task,
        task_cid=task_cid,
        current_git_tree_id=tree,
        execution_route_binding=route.to_dict(),
        launch_source_amendment=amendment.to_dict(),
        attempt_source_policy_root=policy_root,
        attempt=1,
    )
    assert materials["forest_roots"].repository_forest_cid == (
        amendment.launch_source_forest_root
    )
    decision = evaluate_provider_gate(
        task_cid=task_cid,
        forest_roots=materials["forest_roots"],
        residual_packet_cid=materials["residual_packet"].packet_id,
        obligation_graph_cid=materials["obligation_graph_cid"],
        plan_cid=materials["plan_cid"],
        doctor_cid=materials["doctor_cid"],
        authority_receipt_cids=materials["authority_receipt_cids"],
        authority_receipt_resolver=materials["authority_receipt_resolver"],
        allow_legacy_residual=False,
    )
    assert decision.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    assert decision.provider_authorized is True
    assert_provider_dispatch_allowed(decision)

    daemon = PortalImplementationDaemon.__new__(PortalImplementationDaemon)
    daemon.board_namespace = SPAR_BOARD_NAMESPACE
    daemon.pre_implementation_authority_materials_resolver = None
    bind_spar_residual_authority(daemon, repo_root=tmp_path, store_dir=tmp_path / "store")
    resolved = daemon.pre_implementation_authority_materials_resolver(
        task=task,
        task_cid=task_cid,
        current_git_tree_id=tree,
        execution_route_binding=route.to_dict(),
        launch_source_amendment=amendment.to_dict(),
        attempt_source_policy_root=policy_root,
        attempt=1,
    )
    persisted = tmp_path / "store" / f"{task_cid}.json"
    assert persisted.is_file()
    assert resolved["plan_cid"] == materials["plan_cid"]


def test_portal_task_uses_owned_paths_when_outputs_missing() -> None:
    source = TaskSourceTask(
        task_id="SPAR-007",
        task_cid=_cid("spar-007-owned-paths"),
        goal_id="SPAR-G021",
        goal_cid=_cid("goal"),
        title="Build the typed static program and refactoring graph",
        status="retrying",
        revision=1,
        ordinal=7,
        body={
            "owned_paths": (
                "ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/program_graph.py, "
                "ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py"
            ),
            "validation": (
                "python3 -m pytest -q "
                "ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py"
            ),
        },
        board_namespace=SPAR_BOARD_NAMESPACE,
    )
    portal = PortalImplementationDaemon._portal_task_from_source_task(source)
    assert portal.outputs == [
        "ipfs_datasets_py/ipfs_datasets_py/semantic_refactoring/program_graph.py",
        "ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py",
    ]
    assert portal.validation == [
        "python3 -m pytest -q ipfs_datasets_py/tests/unit/semantic_refactoring/test_program_graph.py"
    ]
