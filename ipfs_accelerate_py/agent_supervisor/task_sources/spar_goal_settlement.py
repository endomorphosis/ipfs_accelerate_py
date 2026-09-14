"""Owner-only SPAR goal CAS. Missing independent roots never mutate goals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .control_plane_contracts import content_identity
from .intent_repository import IntentRepository

SCHEMA = "ipfs_accelerate_py/agent-supervisor/spar-goal-settlement@1"
RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/spar-goal-completion-receipt@1"
MISSING = "spar_native_goal_cas_settlement_adapter_required"


def _deferred(reason: str = MISSING, **extra: Any) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "admitted": False,
        "authority": "spar_native_goal_cas",
        "completion_authority": False,
        "semantic_acceptance_authority": False,
        "reason": reason,
        "changed_goal_ids": [],
    }
    result.update(extra)
    return result


def _goal_order(profile: Mapping[str, Any]) -> list[str]:
    waiting: dict[str, set[str]] = {row["goal_cid"]: set() for row in profile["goals"]}
    remaining = {row["goal_cid"] for row in profile["goals"]}
    for edge in profile["goal_edges"]:
        parent, child = edge["parent_goal_cid"], edge["child_goal_cid"]
        if parent not in remaining or child not in remaining:
            continue
        if edge["edge_kind"] == "goal_parent":
            waiting[parent].add(child)
        elif edge["edge_kind"] == "goal_dependency":
            waiting[child].add(parent)
    ordered: list[str] = []
    while remaining:
        ready = [cid for cid in remaining if not (waiting[cid] & remaining)]
        if not ready:
            raise ValueError("SPAR goal dependency graph is cyclic")
        ready.sort()
        chosen = ready[0]
        remaining.remove(chosen)
        ordered.append(chosen)
    return ordered


def observe_goal_settlement(
    *,
    native_goals: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
    accepted_root: Mapping[str, Any],
    runtime: Mapping[str, Any],
    kit: Mapping[str, Any],
) -> dict[str, Any]:
    """Read-only: goals are accepted only from native completed SPAR receipts."""
    if (
        accepted_root.get("admitted") is not True
        or runtime.get("admitted") is not True
        or kit.get("admitted") is not True
    ):
        return _deferred()
    accepted = []
    for sealed in profile["goals"]:
        goal = native_goals.get(sealed["goal_cid"])
        body = {}
        if goal is not None:
            raw = goal.get("body_json")
            if isinstance(raw, str):
                import json

                body = json.loads(raw)
            elif isinstance(raw, dict):
                body = raw
            if "body" in body and isinstance(body["body"], dict):
                receipt = (body.get("completion_receipt") or body["body"].get("completion_receipt"))
            else:
                receipt = body.get("completion_receipt")
        else:
            receipt = None
        runtime_cid = receipt.get("runtime_receipt_cid") if isinstance(receipt, Mapping) else ""
        accepted_cid = receipt.get("accepted_root_cid") if isinstance(receipt, Mapping) else ""
        bootstrap_root = (
            accepted_root.get("admitted") is True
            and accepted_root.get("admission_mode") == "bootstrap"
        )
        if (
            goal is None
            or str(goal.get("status") or "") != "completed"
            or not isinstance(receipt, Mapping)
            or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("goal_cid") != sealed["goal_cid"]
            or type(accepted_cid) is not str
            or not accepted_cid.strip()
            or type(runtime_cid) is not str
            or not runtime_cid.strip()
            or runtime.get("admitted") is not True
            or (
                accepted_cid != accepted_root.get("accepted_root_cid")
                and not bootstrap_root
            )
        ):
            return _deferred()
        accepted.append(sealed["goal_alias"])
    result = {
        "schema": SCHEMA,
        "admitted": True,
        "authority": "spar_native_goal_cas",
        "completion_authority": False,
        "semantic_acceptance_authority": False,
        "accepted_goal_ids": accepted,
        "changed_goal_ids": [],
        "idempotent_replay": True,
    }
    result["receipt_cid"] = content_identity(
        {key: value for key, value in result.items() if key != "receipt_cid"}
    )
    return result


def settle_spar_goals(
    connection: Any,
    *,
    profile: Mapping[str, Any],
    native_goals: Mapping[str, Mapping[str, Any]],
    task_evidence: Sequence[Mapping[str, Any]],
    accepted_root: Mapping[str, Any],
    runtime: Mapping[str, Any],
    kit: Mapping[str, Any],
    owner_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """All-or-none SPAR goal CAS on the exclusive owner connection."""
    observed = observe_goal_settlement(
        native_goals=native_goals,
        profile=profile,
        accepted_root=accepted_root,
        runtime=runtime,
        kit=kit,
    )
    if observed.get("admitted") is True:
        return observed
    if (
        accepted_root.get("admitted") is not True
        or runtime.get("admitted") is not True
        or kit.get("admitted") is not True
        or any(not row.get("receipt") or row.get("blockers") for row in task_evidence)
        or set(native_goals) != {row["goal_cid"] for row in profile["goals"]}
    ):
        return _deferred()
    receipts_by_task = {row["task_cid"]: row["receipt"] for row in task_evidence}
    order = _goal_order(profile)
    sealed_by_cid = {row["goal_cid"]: row for row in profile["goals"]}
    repo = IntentRepository(
        bound_connection=connection,
        install_schema=False,
        owner_id=str(owner_identity.get("server_id") or "spar-owner"),
        session_id=str(owner_identity.get("process_birth_id") or "spar-session"),
    )
    connection.execute("BEGIN TRANSACTION")
    repo._bound_transaction_depth = 1
    changed: list[str] = []
    produced: dict[str, dict[str, Any]] = {}
    try:
        for goal_cid in order:
            sealed = sealed_by_cid[goal_cid]
            goal = native_goals[goal_cid]
            if str(goal.get("status") or "") in {
                "completed",
                "complete",
                "done",
                "verified_complete",
            }:
                raise ValueError("SPAR goal is completed without an admitted SPAR receipt")
            expected = int(goal["revision"])
            child_cids = [
                edge["child_goal_cid"]
                for edge in profile["goal_edges"]
                if edge["parent_goal_cid"] == goal_cid and edge["edge_kind"] == "goal_parent"
            ]
            dep_cids = [
                edge["parent_goal_cid"]
                for edge in profile["goal_edges"]
                if edge["child_goal_cid"] == goal_cid and edge["edge_kind"] == "goal_dependency"
            ]
            task_cids = [row["task_cid"] for row in profile["tasks"] if row["goal_cid"] == goal_cid]
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "goal_cid": goal_cid,
                "goal_alias": sealed["goal_alias"],
                "expected_revision": expected,
                "new_revision": expected + 1,
                "task_receipt_cids": [
                    (receipts_by_task[cid] or {}).get("receipt_cid") for cid in task_cids
                ],
                "child_goal_receipt_cids": [produced[cid]["receipt_cid"] for cid in child_cids],
                "dependency_goal_receipt_cids": [produced[cid]["receipt_cid"] for cid in dep_cids],
                "accepted_root_cid": accepted_root["accepted_root_cid"],
                "runtime_receipt_cid": runtime["receipt_cid"],
                "kit_transition_cid": kit.get("transition_cid"),
                "source_forest_root": kit.get("source_forest_root"),
            }
            receipt["receipt_cid"] = content_identity(
                {key: value for key, value in receipt.items() if key != "receipt_cid"}
            )
            cas = repo.cas_goal_status(
                goal_cid=goal_cid,
                expected_revision=expected,
                new_status="completed",
                receipt=receipt,
            )
            produced[goal_cid] = receipt
            if cas.changed:
                changed.append(sealed["goal_alias"])
        connection.execute("COMMIT")
    except Exception as exc:  # noqa: BLE001 - roll back the whole SPAR population
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass
        return _deferred(error_class=type(exc).__name__)
    finally:
        repo._bound_transaction_depth = 0
    result = {
        "schema": SCHEMA,
        "admitted": True,
        "authority": "spar_native_goal_cas",
        "completion_authority": False,
        "semantic_acceptance_authority": False,
        "accepted_goal_ids": [sealed_by_cid[cid]["goal_alias"] for cid in order],
        "changed_goal_ids": changed,
        "idempotent_replay": not changed,
    }
    result["receipt_cid"] = content_identity(
        {key: value for key, value in result.items() if key != "receipt_cid"}
    )
    return result
