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
RUNTIME_MISSING = "runtime_lane_and_merge_queue_settlement_receipt_required"
BOOTSTRAP_MODE = "bootstrap"
REQUIRED_MODE = "required"
REQUIRED_CLAUSES = (
    "required_mode_roots_accepted",
    "safety_floors_noncompensable_accepted",
    "self_hosted_capstone_accepted",
    "fixed_point_accepted",
)
CLAUSE_REPORTS = {
    "required_mode_roots_accepted": (
        "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json",
        "benchmarks/agent_supervisor/semantic_refactoring/capstone_report.json",
    ),
    "safety_floors_noncompensable_accepted": (
        "benchmarks/agent_supervisor/semantic_refactoring/benchmark_report.json",
    ),
    "self_hosted_capstone_accepted": (
        "benchmarks/agent_supervisor/semantic_refactoring/capstone_report.json",
    ),
    "fixed_point_accepted": (
        "docs/architecture/semantic_preserving_autonomous_remodularization_inventory/final_report.json",
    ),
}


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


def _task_receipt_cid(receipt: Any) -> str:
    """Bind the live completion receipt identity; never invent a CID."""
    if not isinstance(receipt, Mapping):
        return ""
    for key in ("receipt_cid", "completion_receipt_cid"):
        value = receipt.get(key)
        if type(value) is str and value.strip():
            return value
    return ""


def _producer_fail_extra(raw: Any) -> dict[str, Any]:
    """Retain producer diagnostics; never copy an admitted boolean."""
    extra: dict[str, Any] = {}
    if not isinstance(raw, Mapping):
        return extra
    error = raw.get("error")
    if type(error) is str and error:
        extra["producer_error"] = error[:512]
    producer_reason = raw.get("reason")
    if type(producer_reason) is str and producer_reason:
        extra["producer_reason"] = producer_reason[:256]
    outcomes = raw.get("clause_outcomes")
    if isinstance(outcomes, Mapping):
        closed: dict[str, Any] = {}
        for name in REQUIRED_CLAUSES:
            row = outcomes.get(name)
            if not isinstance(row, Mapping):
                continue
            closed[name] = {
                "accepted": row.get("accepted") is True,
                "reason": str(row.get("reason") or "")[:256],
            }
        if closed:
            extra["clause_outcomes"] = closed
    return extra


def sealed_current_rollout_mode() -> str:
    """Read the sealed in-code SPAR current mode. Not acceptance authority."""
    try:
        from ..semantic_refactoring.rollout import sealed_rollout_baseline
    except Exception:  # noqa: BLE001 - missing rollout contract is not acceptance
        return ""
    baseline = sealed_rollout_baseline()
    mode = baseline.get("current_mode") if isinstance(baseline, Mapping) else ""
    return mode if type(mode) is str else ""


def _sealed_current_rollout_mode() -> str:
    return sealed_current_rollout_mode()


def _runtime_settled(runtime: Any) -> bool:
    """True only for an independently admitted, settled runtime receipt."""
    if not isinstance(runtime, Mapping):
        return False
    receipt_cid = runtime.get("receipt_cid")
    return (
        runtime.get("admitted") is True
        and runtime.get("settled") is True
        and type(receipt_cid) is str
        and bool(receipt_cid.strip())
    )


def _producer_subject_verified(
    raw: Mapping[str, Any],
    *,
    subject: Mapping[str, Any],
    profile_cid: str,
    subject_cid: str,
) -> bool:
    """Producer independently closed the subject; clause booleans are not used."""
    error = raw.get("error")
    if type(error) is str and error.strip():
        return False
    if raw.get("reason") == MISSING:
        return False
    return (
        raw.get("producer_interface") == PRODUCER_INTERFACE
        and raw.get("profile_cid") == profile_cid
        and raw.get("source_forest_root") == subject["source_forest_root"]
        and raw.get("subject_cid") == subject_cid
    )


def _source_clause_probes(source: Mapping[str, Any]) -> dict[str, Any]:
    """Classify nominated reports as non-admission; never copy their booleans."""
    reports = source.get("reports") if isinstance(source.get("reports"), list) else []
    by_path = {
        str(row.get("path") or ""): row
        for row in reports
        if isinstance(row, Mapping)
    }
    forest = source.get("source_forest") if isinstance(source.get("source_forest"), Mapping) else {}
    current_root = forest.get("source_forest_root")
    current_mode = _sealed_current_rollout_mode()
    probes: dict[str, Any] = {}
    for name, paths in CLAUSE_REPORTS.items():
        blockers: list[str] = []
        digests: list[str] = []
        if name != "safety_floors_noncompensable_accepted" and current_mode != "required":
            blockers.append(
                f"current_rollout_mode_is_not_required:{current_mode or 'unobserved'}"
            )
        for path in paths:
            row = by_path.get(path)
            if not isinstance(row, Mapping) or row.get("available") is not True:
                blockers.append(f"nominated_report_unavailable:{path}")
                continue
            digest = row.get("content_digest")
            if type(digest) is str and digest:
                digests.append(digest)
            if (
                row.get("nomination_only") is True
                or row.get("can_authorize_completion") is not True
            ):
                blockers.append(f"nominated_report_cannot_authorize_clause:{path}")
            roots = row.get("authority_roots") if isinstance(row.get("authority_roots"), Mapping) else {}
            report_forest = roots.get("repository_forest_cid")
            if (
                type(current_root) is str
                and current_root
                and report_forest != current_root
            ):
                blockers.append(f"nominated_report_source_forest_mismatch:{path}")
        probes[name] = {
            "accepted": False,
            "reason": (
                blockers[0]
                if blockers
                else "nominated_report_is_not_independent_clause_evidence"
            ),
            "blockers": blockers,
            "report_digests": digests,
            "current_rollout_mode": current_mode,
            "semantic_acceptance_authority": False,
        }
    return probes


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
        "task_receipt_cids": [_task_receipt_cid(row.get("receipt")) for row in task_evidence],
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
    runtime: Mapping[str, Any] | None = None,
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
    missing_receipt_cids = sum(
        1 for row in task_evidence if not _task_receipt_cid(row.get("receipt"))
    )
    if missing_receipt_cids:
        return _deferred(
            "task_receipt_identity_incomplete",
            missing_receipt_cid_count=missing_receipt_cids,
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
        current_source = {
            "current_rollout_mode": _sealed_current_rollout_mode(),
            "source_forest_root": subject["source_forest_root"],
            "reports": source.get("reports") if isinstance(source.get("reports"), list) else [],
            "clause_records": {},
        }
        try:
            raw = producer.admit_spar_accepted_root(
                copy.deepcopy(subject), current_source=current_source
            )
        except TypeError:
            raw = producer.admit_spar_accepted_root(copy.deepcopy(subject))
    except Exception as exc:  # noqa: BLE001 - producer absence/failure is not acceptance
        return _deferred(MISSING, error_class=type(exc).__name__)
    if not isinstance(raw, Mapping):
        return _deferred(MISSING, error_class="TypeError")
    clauses_ok = all(raw.get(name) is True for name in REQUIRED_CLAUSES)
    evidence = raw.get("evidence_cids")
    producer_ok = raw.get("producer_interface") == PRODUCER_INTERFACE
    extra = _producer_fail_extra(raw)
    extra["source_clause_probes"] = _source_clause_probes(source)
    extra["current_rollout_mode"] = _sealed_current_rollout_mode()
    if (
        raw.get("admitted") is True
        and raw.get("subject_cid") == subject_cid
        and raw.get("profile_cid") == profile_cid
        and raw.get("source_forest_root") == subject["source_forest_root"]
        and producer_ok
        and raw.get("semantic_acceptance_authority") is True
        and clauses_ok
        and isinstance(evidence, list)
        and len(evidence) >= 1
        and all(isinstance(item, str) and item for item in evidence)
    ):
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
    current_mode = extra["current_rollout_mode"]
    if current_mode == BOOTSTRAP_MODE:
        subject_ok = _producer_subject_verified(
            raw,
            subject=subject,
            profile_cid=profile_cid,
            subject_cid=subject_cid,
        )
        if subject_ok and _runtime_settled(runtime):
            kit_cid = kit.get("transition_cid")
            runtime_cid = runtime["receipt_cid"] if isinstance(runtime, Mapping) else ""
            accepted_root_cid = content_identity(
                {
                    "schema": SCHEMA,
                    "admission_mode": BOOTSTRAP_MODE,
                    "current_rollout_mode": BOOTSTRAP_MODE,
                    "subject_cid": subject_cid,
                    "profile_cid": profile_cid,
                    "source_forest_root": subject["source_forest_root"],
                    "kit_transition_cid": kit_cid,
                    "task_receipt_cids": list(subject["task_receipt_cids"]),
                }
            )
            return {
                "schema": SCHEMA,
                "admitted": True,
                "authority": "datasets_spar_accepted_root",
                "completion_authority": False,
                "semantic_acceptance_authority": False,
                "admission_mode": BOOTSTRAP_MODE,
                "current_rollout_mode": BOOTSTRAP_MODE,
                "subject_cid": subject_cid,
                "accepted_root_cid": accepted_root_cid,
                "evidence_cids": [subject_cid, str(kit_cid), str(runtime_cid)],
                "producer_interface": PRODUCER_INTERFACE,
                "kit_transition_cid": kit_cid,
                "runtime_receipt_cid": runtime_cid,
                "clause_outcomes": extra.get("clause_outcomes"),
                "source_clause_probes": extra.get("source_clause_probes"),
            }
        if subject_ok:
            return _deferred(RUNTIME_MISSING, **extra)
        if producer_ok and raw.get("reason") in {MISSING, MODE_FLOORS}:
            return _deferred(str(raw["reason"]), **extra)
        return _deferred(MISSING, **extra)
    if raw.get("admitted") is True:
        reason = MODE_FLOORS
    elif producer_ok and raw.get("reason") in {MISSING, MODE_FLOORS}:
        reason = str(raw["reason"])
    elif producer_ok:
        reason = MODE_FLOORS
    else:
        reason = MISSING
    return _deferred(reason, **extra)
