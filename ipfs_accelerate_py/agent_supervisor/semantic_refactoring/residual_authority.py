"""Operator-published SPAR residual-admission receipts.

The production daemon does not mint planner/doctor/obligation/logic/repair
authority by default.  SPAR is an operator-owned R&D board: this adapter
publishes exact, content-addressed residual-admission receipts bound to the
current launch forest and claimed task, then resolves them read-through.

Receipts authorize only ``residual_llm_authorized`` provider nomination.
They are not completion, merge, write, or proof authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ..planning.residual_llm_packet import ResidualLlmPacket, seal_residual_llm_packet
from ..proof.formal_verification_contracts import content_identity
from ..task_sources.launch_source_amendment import LaunchSourceAmendment
from ..todo_daemon.implementation_daemon import (
    PortalTask,
    task_declared_output_paths,
)
from ..todo_daemon.implementation_disposition import ImplementationForestRoots

SPAR_BOARD_NAMESPACE: Final[str] = (
    "semantic-preserving-autonomous-remodularization-v1"
)
AUTHORITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/authority-receipt@1"
)
RESIDUAL_AUTHORITY_BUNDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-residual-authority-bundle@1"
)
REQUIRED_AUTHORITY_RECEIPT_KINDS: Final[tuple[str, ...]] = (
    "planner",
    "doctor",
    "obligation",
    "logic",
    "repair",
)
DEFAULT_STORE_RELATIVE: Final[str] = (
    "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
    "/evidence/residual-authority"
)


class SparResidualAuthorityError(RuntimeError):
    """Fail-closed rejection for an incomplete SPAR residual bundle."""


def mint_authority_receipt(
    *,
    kind: str,
    task_cid: str,
    repository_forest_cid: str,
) -> dict[str, str]:
    """Return one typed authority receipt with a matching content identity."""

    if kind not in REQUIRED_AUTHORITY_RECEIPT_KINDS:
        raise SparResidualAuthorityError(f"unknown authority receipt kind: {kind}")
    body = {
        "schema": AUTHORITY_RECEIPT_SCHEMA,
        "receipt_kind": kind,
        "task_cid": str(task_cid or "").strip(),
        "repository_forest_cid": str(repository_forest_cid or "").strip(),
    }
    if not body["task_cid"] or not body["repository_forest_cid"]:
        raise SparResidualAuthorityError("authority receipt identity is incomplete")
    return {**body, "content_id": content_identity(body)}


def mint_authority_receipts(
    *,
    task_cid: str,
    repository_forest_cid: str,
) -> dict[str, dict[str, str]]:
    """Mint the five typed receipts required by PreImplementationKernel@1."""

    return {
        kind: mint_authority_receipt(
            kind=kind,
            task_cid=task_cid,
            repository_forest_cid=repository_forest_cid,
        )
        for kind in REQUIRED_AUTHORITY_RECEIPT_KINDS
    }


def _write_paths_for_task(task: PortalTask) -> tuple[str, ...]:
    declared = task_declared_output_paths(task)
    if declared:
        return declared
    raise SparResidualAuthorityError("SPAR residual packet has no declared write paths")


def mint_spar_residual_authority_materials(
    *,
    task: PortalTask,
    task_cid: str,
    current_git_tree_id: str,
    execution_route_binding: Mapping[str, Any],
    launch_source_amendment: Mapping[str, Any] | LaunchSourceAmendment,
    attempt_source_policy_root: str,
    attempt: int = 1,
) -> dict[str, Any]:
    """Mint one exact residual bundle bound to the current SPAR launch forest."""

    del attempt
    amendment = (
        launch_source_amendment
        if isinstance(launch_source_amendment, LaunchSourceAmendment)
        else LaunchSourceAmendment.from_dict(launch_source_amendment)
    )
    if not str(amendment.board_namespace or "").strip():
        raise SparResidualAuthorityError("residual authority requires a board namespace")
    if str(current_git_tree_id or "").strip() != amendment.launch_repository_tree_id:
        raise SparResidualAuthorityError("residual authority tree is not the launch tree")
    policy_root = amendment.attempt_policy_root(dict(execution_route_binding))
    if policy_root != str(attempt_source_policy_root or "").strip():
        raise SparResidualAuthorityError("residual authority policy root mismatch")

    forest_roots = ImplementationForestRoots(
        repository_id=f"repository:{amendment.board_namespace}",
        repository_forest_cid=amendment.launch_source_forest_root,
        git_tree_id=amendment.launch_repository_tree_id,
        policy_root=policy_root,
    )
    receipts = mint_authority_receipts(
        task_cid=task_cid,
        repository_forest_cid=forest_roots.repository_forest_cid,
    )
    receipt_cids = {kind: receipt["content_id"] for kind, receipt in receipts.items()}
    packet = seal_residual_llm_packet(
        task_id=task_cid,
        repository_id=forest_roots.repository_id,
        tree_id=forest_roots.git_tree_id,
        forest_id=forest_roots.repository_forest_cid,
        write_paths=_write_paths_for_task(task),
        obligation_ids=(receipt_cids["obligation"],),
        counterexample_capsule={"target_ids": [str(task.task_id or task_cid)]},
        validation_commands=tuple(task.validation or ()),
        authority_roots={
            "repository_forest_cid": forest_roots.repository_forest_cid,
            "policy_root": forest_roots.policy_root,
        },
    )
    receipt_by_cid = {
        receipt["content_id"]: dict(receipt) for receipt in receipts.values()
    }
    return {
        "forest_roots": forest_roots,
        "residual_packet": packet,
        "obligation_graph_cid": receipt_cids["obligation"],
        "plan_cid": receipt_cids["planner"],
        "doctor_cid": receipt_cids["doctor"],
        "authority_receipt_cids": receipt_cids,
        "authority_receipt_resolver": lambda cid, store=receipt_by_cid: store.get(cid),
        "receipts": receipts,
    }


def persist_spar_residual_authority_bundle(
    store_dir: Path,
    materials: Mapping[str, Any],
) -> Path:
    """Persist a diagnostic copy of one minted residual bundle."""

    packet = materials["residual_packet"]
    if not isinstance(packet, ResidualLlmPacket):
        raise SparResidualAuthorityError("residual packet missing from materials")
    forest = materials["forest_roots"]
    payload = {
        "schema": RESIDUAL_AUTHORITY_BUNDLE_SCHEMA,
        "task_cid": packet.task_id,
        "repository_id": packet.repository_id,
        "git_tree_id": packet.tree_id,
        "repository_forest_cid": (
            forest.repository_forest_cid
            if hasattr(forest, "repository_forest_cid")
            else packet.forest_id
        ),
        "policy_root": (
            forest.policy_root if hasattr(forest, "policy_root") else ""
        ),
        "residual_packet_cid": packet.packet_id,
        "residual_packet": packet._payload(),
        "authority_receipt_cids": dict(materials["authority_receipt_cids"]),
        "receipts": dict(materials.get("receipts") or {}),
        "plan_cid": materials["plan_cid"],
        "doctor_cid": materials["doctor_cid"],
        "obligation_graph_cid": materials["obligation_graph_cid"],
        "producer": "spar-operator-residual-admission@1",
        "completion_authority": False,
        "write_authority": False,
        "proof_authority": False,
        "nomination_only": True,
    }
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / f"{packet.task_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_spar_residual_authority_resolver(
    store_dir: Path | None = None,
):
    """Return the daemon ``pre_implementation_authority_materials_resolver``."""

    def resolve(
        *,
        task: PortalTask,
        task_cid: str,
        current_git_tree_id: str,
        execution_route_binding: Mapping[str, Any],
        launch_source_amendment: Mapping[str, Any],
        attempt_source_policy_root: str,
        attempt: int,
    ) -> dict[str, Any]:
        materials = mint_spar_residual_authority_materials(
            task=task,
            task_cid=task_cid,
            current_git_tree_id=current_git_tree_id,
            execution_route_binding=execution_route_binding,
            launch_source_amendment=launch_source_amendment,
            attempt_source_policy_root=attempt_source_policy_root,
            attempt=attempt,
        )
        if store_dir is not None:
            persist_spar_residual_authority_bundle(store_dir, materials)
        public = {
            key: materials[key]
            for key in (
                "forest_roots",
                "residual_packet",
                "obligation_graph_cid",
                "plan_cid",
                "doctor_cid",
                "authority_receipt_cids",
                "authority_receipt_resolver",
            )
        }
        return public

    return resolve


def bind_spar_residual_authority(
    daemon: Any,
    *,
    repo_root: Path,
    store_dir: Path | None = None,
) -> None:
    """Install SPAR residual-admission materials on one Portal daemon."""

    board_namespace = str(getattr(daemon, "board_namespace", "") or "")
    resolved_store = store_dir
    if resolved_store is None:
        if board_namespace:
            resolved_store = (
                Path(repo_root)
                / "data"
                / "agent_supervisor"
                / board_namespace.replace("-", "_")
                / "evidence"
                / "residual-authority"
            )
        else:
            resolved_store = Path(repo_root) / DEFAULT_STORE_RELATIVE
    daemon.pre_implementation_authority_materials_resolver = (
        build_spar_residual_authority_resolver(resolved_store)
    )


__all__ = [
    "AUTHORITY_RECEIPT_SCHEMA",
    "DEFAULT_STORE_RELATIVE",
    "REQUIRED_AUTHORITY_RECEIPT_KINDS",
    "RESIDUAL_AUTHORITY_BUNDLE_SCHEMA",
    "SPAR_BOARD_NAMESPACE",
    "SparResidualAuthorityError",
    "bind_spar_residual_authority",
    "build_spar_residual_authority_resolver",
    "mint_authority_receipt",
    "mint_authority_receipts",
    "mint_spar_residual_authority_materials",
    "persist_spar_residual_authority_bundle",
]
