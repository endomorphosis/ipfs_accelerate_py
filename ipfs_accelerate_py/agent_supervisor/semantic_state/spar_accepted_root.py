"""Independent datasets admission for a SPAR accepted current root.

This adapter never treats nominated reports, task counts, or worker booleans as
acceptance. Missing producer code fail-closes with a typed blocker.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from ..task_sources.control_plane_contracts import content_identity

SCHEMA = "ipfs_accelerate_py/agent-supervisor/spar-accepted-root-admission@1"
PRODUCER_INTERFACE = "SparAcceptedRootProducer@1"
DATASETS_MODULE = "ipfs_datasets_py.semantic_refactoring.accepted_roots"
MISSING = "datasets_independent_accepted_root_producer_and_admission_required"
MODE_FLOORS = "required_mode_roots_safety_floors_capstone_fixed_point_acceptance_required"
REQUIRED_CLAUSES = (
    "required_mode_roots_accepted",
    "safety_floors_noncompensable_accepted",
    "self_hosted_capstone_accepted",
    "fixed_point_accepted",
)


def _deferred(reason: str, **extra: Any) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "admitted": False,
        "authority": "datasets_spar_accepted_root",
        "completion_authority": False,
        "semantic_acceptance_authority": False,
        "reason": reason,
    }
    result.update(extra)
    return result


def closed_subject(
    profile: Mapping[str, Any],
    profile_cid: str,
    *,
    source: Mapping[str, Any],
    kit: Mapping[str, Any],
    task_evidence: Sequence[Mapping[str, Any]],
    goal_requirements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind the sealed SPAR contracts that a datasets producer must re-verify."""
    forest = source.get("source_forest") if isinstance(source.get("source_forest"), Mapping) else {}
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/spar-accepted-root-subject@1",
        "profile_cid": profile_cid,
        "bootstrap_receipt_id": profile["bootstrap_receipt_id"],
        "plan_root_cid": profile["plan_root_cid"],
        "board_namespace": profile["board_namespace"],
        "source_forest_root": forest.get("source_forest_root"),
        "repository_tree_id": source.get("repository_tree_id"),
        "source_head": forest.get("source_head"),
        "kit_transition_cid": kit.get("transition_cid"),
        "kit_manifest_cid": kit.get("manifest_cid"),
        "goal_cids": [row["goal_cid"] for row in profile["goals"]],
        "goal_contract_cids": [row["contract_cid"] for row in goal_requirements],
        "task_cids": [row["task_cid"] for row in profile["tasks"]],
        "task_receipt_cids": [
            (row.get("receipt") or {}).get("receipt_cid")
            if isinstance(row.get("receipt"), Mapping)
            else None
            for row in task_evidence
        ],
        "semantic_acceptance_authority": False,
        "completion_authority": False,
    }


def _load_producer() -> Any:
    """Import the pinned datasets producer; absence is a typed non-admission."""
    try:
        from ipfs_datasets_py.semantic_refactoring import accepted_roots as producer
    except Exception as exc:  # noqa: BLE001 - missing capability is not acceptance
        raise RuntimeError(MISSING) from exc
    if (
        getattr(producer, "PRODUCER_INTERFACE", None) != PRODUCER_INTERFACE
        or not callable(getattr(producer, "admit_spar_accepted_root", None))
    ):
        raise RuntimeError(MISSING)
    return producer


def admit_accepted_root(
    profile: Mapping[str, Any],
    profile_cid: str,
    *,
    source: Mapping[str, Any],
    kit: Mapping[str, Any],
    task_evidence: Sequence[Mapping[str, Any]],
    goal_requirements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Re-admit a datasets producer result against the exact current subject."""
    if source.get("available") is not True:
        return _deferred(MISSING, source_available=False)
    if source.get("clean") is not True:
        return _deferred("current_source_observation_unclean")
    if kit.get("admitted") is not True:
        return _deferred(
            "kit_source_forest_not_admitted",
            kit_reason=kit.get("reason"),
        )
    if any(not row.get("receipt") or row.get("blockers") for row in task_evidence):
        return _deferred("task_evidence_incomplete")
    if len(task_evidence) != 51 or len(goal_requirements) != 32:
        return _deferred(
            "sealed_task_or_goal_population_mismatch",
            task_evidence_count=len(task_evidence),
            goal_requirement_count=len(goal_requirements),
        )
    subject = closed_subject(
        profile,
        profile_cid,
        source=source,
        kit=kit,
        task_evidence=task_evidence,
        goal_requirements=goal_requirements,
    )
    subject_cid = content_identity(subject)
    try:
        producer = _load_producer()
        raw = producer.admit_spar_accepted_root(copy.deepcopy(subject))
    except Exception as exc:  # noqa: BLE001 - producer absence/failure is not acceptance
        return _deferred(MISSING, error_class=type(exc).__name__)
    if not isinstance(raw, Mapping):
        return _deferred(MISSING, error_class="TypeError")
    clauses_ok = all(raw.get(name) is True for name in REQUIRED_CLAUSES)
    evidence = raw.get("evidence_cids")
    producer_ok = raw.get("producer_interface") == PRODUCER_INTERFACE
    if (
        raw.get("admitted") is not True
        or raw.get("subject_cid") != subject_cid
        or raw.get("profile_cid") != profile_cid
        or raw.get("source_forest_root") != subject["source_forest_root"]
        or not producer_ok
        or raw.get("semantic_acceptance_authority") is not True
        or not clauses_ok
        or not isinstance(evidence, list)
        or len(evidence) < 1
        or any(not isinstance(item, str) or not item for item in evidence)
    ):
        if raw.get("admitted") is True:
            reason = MODE_FLOORS
        elif producer_ok and raw.get("reason") in {MISSING, MODE_FLOORS}:
            reason = str(raw["reason"])
        elif producer_ok:
            reason = MODE_FLOORS
        else:
            reason = MISSING
        return _deferred(reason)
    return {
        "schema": SCHEMA,
        "admitted": True,
        "authority": "datasets_spar_accepted_root",
        "completion_authority": False,
        "semantic_acceptance_authority": True,
        "subject_cid": subject_cid,
        "accepted_root_cid": raw.get("accepted_root_cid") or subject_cid,
        "evidence_cids": list(evidence),
        "producer_interface": PRODUCER_INTERFACE,
    }
