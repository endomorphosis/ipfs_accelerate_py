"""WorldSnapshotBuilder@1 — independently verified authority admission.

Each required dimension is obtained from its own injected authority. Optional
DuckLake projection health is observed only through a DuckDB-recorded receipt
and never grants scheduling authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

try:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_contracts import (
        COMPONENT_OWNERS,
        COMPONENT_STATUSES,
        DATASETS_COMPONENTS,
        OPTIONAL_COMPONENTS,
        REQUIRED_COMPONENTS,
        WorldSnapshotContractError,
        parse_world_snapshot,
    )
except ImportError:  # LGSWF-011 may merge before LGSWF-010
    COMPONENT_STATUSES = frozenset(
        {"current", "stale", "unavailable", "inconsistent", "quarantined"}
    )
    REQUIRED_COMPONENTS = (
        "repository",
        "repository_tree",
        "datasets_repository",
        "datasets_semantic_state_root",
        "symbol_root",
        "capsule_index",
        "environment_bindings",
        "contract_root",
        "obligation_set",
        "accepted_plan_root",
        "plan_revision",
        "objectives",
        "goal_population",
        "subgoal_population",
        "task_population",
        "claims",
        "resource_snapshot",
        "capability_snapshot",
        "merge_root",
        "completion_root",
        "unresolved_gaps",
        "policy_root",
        "event_cursor",
    )
    OPTIONAL_COMPONENTS = ("ducklake_projection_health",)
    DATASETS_COMPONENTS = frozenset(
        {
            "datasets_repository",
            "datasets_semantic_state_root",
            "symbol_root",
            "capsule_index",
            "environment_bindings",
            "contract_root",
            "obligation_set",
        }
    )
    COMPONENT_OWNERS = {
        name: (
            "ipfs_datasets_py" if name in DATASETS_COMPONENTS else "ipfs_accelerate_py"
        )
        for name in (*REQUIRED_COMPONENTS, *OPTIONAL_COMPONENTS)
    }

    class WorldSnapshotContractError(ValueError):
        """Fallback contract error when LGSWF-010 has not merged."""

    def parse_world_snapshot(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        import hashlib
        import json

        components = dict(payload.get("components") or {})
        identity = {
            key: payload[key]
            for key in (
                "schema",
                "interface",
                "repository_id",
                "components",
                "coordination_epoch",
                "fencing_epoch",
            )
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        identity["components"] = components
        identity["snapshot_cid"] = "sha256:" + digest
        return MappingProxyType(identity)

REQUIRED_AGREEMENT = (
    "repository",
    "repository_tree",
    "accepted_plan_root",
    "task_population",
    "datasets_semantic_state_root",
    "policy_root",
)

REQUIRED_INPUTS = REQUIRED_COMPONENTS


class WorldSnapshotAdmissionError(ValueError):
    """A required authority could not admit a schedulable snapshot."""


def _status_of(record: Mapping[str, Any], name: str) -> str:
    status = record.get("status")
    if status not in COMPONENT_STATUSES:
        raise WorldSnapshotAdmissionError(f"authority {name} returned malformed status")
    return str(status)


def _agreement_key(name: str, record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        name,
        str(record.get("agreement") or record.get("cid") or ""),
        str(record.get("generation") or record.get("cid") or ""),
    )


def observe_ducklake_projection(
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Read optional DuckLake health from a DuckDB-recorded receipt only."""

    if receipt is None:
        return {
            "status": "unavailable",
            "cid": "",
            "owner": COMPONENT_OWNERS["ducklake_projection_health"],
            "authoritative": False,
            "reason": "projection-receipt-absent",
        }
    if receipt.get("tampered") is True or receipt.get("valid") is False:
        return {
            "status": "quarantined",
            "cid": str(receipt.get("receipt_cid") or ""),
            "owner": COMPONENT_OWNERS["ducklake_projection_health"],
            "authoritative": False,
            "reason": "projection-receipt-tampered",
        }
    return {
        "status": str(receipt.get("status") or "current"),
        "cid": str(receipt.get("receipt_cid") or ""),
        "owner": COMPONENT_OWNERS["ducklake_projection_health"],
        "authoritative": False,
        "reason": "duckdb-recorded-observation",
    }


def build_world_snapshot(
    authorities: Mapping[str, Mapping[str, Any]],
    *,
    repository_id: str = "ipfs_accelerate_py",
    coordination_epoch: int = 1,
    fencing_epoch: int = 1,
    ducklake_receipt: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Admit a snapshot from independently injected authority records."""

    if not isinstance(authorities, Mapping):
        raise WorldSnapshotAdmissionError("authorities must be an injected mapping")

    components: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    for name in REQUIRED_INPUTS:
        if name not in authorities:
            raise WorldSnapshotAdmissionError(f"missing injected authority: {name}")
        record = authorities[name]
        if not isinstance(record, Mapping):
            raise WorldSnapshotAdmissionError(f"authority {name} is not a mapping")
        if record.get("implicit_lookup"):
            raise WorldSnapshotAdmissionError(
                f"implicit filesystem/provider lookup forbidden for {name}"
            )
        status = _status_of(record, name)
        owner = str(record.get("owner") or COMPONENT_OWNERS[name])
        if owner != COMPONENT_OWNERS[name]:
            status = "inconsistent"
            reasons.append(f"{name}:owner-mismatch")
        cid = "" if status == "unavailable" else str(record.get("cid") or "")
        component = {"status": status, "cid": cid, "owner": COMPONENT_OWNERS[name]}
        evidence = record.get("evidence_cid")
        if evidence:
            component["evidence_cid"] = str(evidence)
        if name in DATASETS_COMPONENTS and record.get("operational"):
            raise WorldSnapshotAdmissionError(
                f"operational data offered inside datasets authority {name}"
            )
        if status != "current":
            reasons.append(f"{name}:{status}")
        components[name] = component

    observed = {
        name: _agreement_key(name, authorities[name]) for name in REQUIRED_AGREEMENT
    }
    generations = {item[2] for item in observed.values() if item[2]}
    if len({item[1] for item in observed.values() if item[1]}) > 1 and len(generations) > 1:
        # Cross-authority disagreement on the required agreement set.
        for name in REQUIRED_AGREEMENT:
            left = authorities[name]
            if any(
                _agreement_key(name, left)[1] != _agreement_key(other, authorities[other])[1]
                and _agreement_key(name, left)[1]
                and _agreement_key(other, authorities[other])[1]
                for other in REQUIRED_AGREEMENT
                if other != name
            ):
                components[name]["status"] = "inconsistent"
                reasons.append(f"{name}:agreement")

    # Explicit repository / tree / plan / population / semantic / policy keys.
    def _bind(name: str, key: str) -> str:
        return str(authorities[name].get(key) or "")

    expected_repo = _bind("repository", "repository_id") or repository_id
    checks = (
        ("repository_tree", "repository_id", expected_repo),
        ("accepted_plan_root", "repository_id", expected_repo),
        ("task_population", "plan_cid", _bind("accepted_plan_root", "cid")),
        (
            "datasets_semantic_state_root",
            "generation",
            _bind("datasets_semantic_state_root", "generation"),
        ),
        ("policy_root", "repository_id", expected_repo),
    )
    for name, key, expected in checks:
        actual = _bind(name, key)
        if expected and actual and actual != expected:
            components[name]["status"] = "inconsistent"
            reasons.append(f"{name}:disagreement:{key}")

    ducklake = observe_ducklake_projection(ducklake_receipt)
    optional = {
        "status": ducklake["status"] if ducklake["status"] in COMPONENT_STATUSES else "unavailable",
        "cid": ducklake["cid"] if ducklake["status"] != "unavailable" else "",
        "owner": COMPONENT_OWNERS["ducklake_projection_health"],
    }
    if optional["status"] != "unavailable" and not optional["cid"]:
        optional = {"status": "unavailable", "cid": "", "owner": optional["owner"]}
    components["ducklake_projection_health"] = optional

    try:
        snapshot = parse_world_snapshot(
            {
                "schema": "lgswf/supervisor-world-snapshot@1",
                "interface": "SupervisorWorldSnapshot@1",
                "repository_id": repository_id,
                "components": {
                    name: {
                        key: value
                        for key, value in record.items()
                        if key in {"status", "cid", "owner", "evidence_cid"}
                    }
                    for name, record in components.items()
                    if name in REQUIRED_COMPONENTS or name in OPTIONAL_COMPONENTS
                },
                "coordination_epoch": coordination_epoch,
                "fencing_epoch": fencing_epoch,
            }
        )
    except WorldSnapshotContractError as exc:
        raise WorldSnapshotAdmissionError(str(exc)) from exc

    required_statuses = {
        name: snapshot["components"][name]["status"] for name in REQUIRED_INPUTS
    }
    unschedulable = [
        f"{name}:{status}"
        for name, status in required_statuses.items()
        if status != "current"
    ]
    result = {
        "snapshot": snapshot,
        "schedulable": not unschedulable,
        "unschedulable_reasons": tuple(unschedulable or reasons if unschedulable else ()),
        "ducklake_projection": MappingProxyType(
            {
                "authoritative": False,
                "status": optional["status"],
                "reason": ducklake.get("reason"),
            }
        ),
        "component_status": MappingProxyType(required_statuses),
    }
    return MappingProxyType(result)


def current_authority_fixture(*, seed: str = "lgswf-011") -> dict[str, dict[str, Any]]:
    import hashlib

    def _cid(label: str) -> str:
        return "sha256:" + hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()

    plan_cid = _cid("accepted_plan_root")
    generation = _cid("datasets_semantic_state_root")
    authorities: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_COMPONENTS:
        authorities[name] = {
            "status": "current",
            "cid": _cid(name),
            "owner": COMPONENT_OWNERS[name],
            "repository_id": "ipfs_accelerate_py",
            "plan_cid": plan_cid,
            "generation": generation,
            "verified": True,
        }
    return authorities


CASF_PROJECTED_COMPONENTS = (
    "datasets_semantic_state_root",
    "task_population",
    "policy_root",
    "repository_tree",
)


def project_casf_world_inputs(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Project a SupervisorWorldSnapshot into compact CASF inputs.

    The projection is observational.  DuckLake health cannot become snapshot,
    scheduling, lease, or completion authority.
    """

    if not isinstance(result, Mapping) or "snapshot" not in result:
        raise WorldSnapshotAdmissionError("builder result is required")
    snapshot = result["snapshot"]
    if not isinstance(snapshot, Mapping) or "components" not in snapshot:
        raise WorldSnapshotAdmissionError("builder snapshot is malformed")
    components = snapshot["components"]
    ducklake = result.get("ducklake_projection") or {}
    if ducklake.get("authoritative") is True:
        raise WorldSnapshotAdmissionError("DuckLake cannot admit a world snapshot")
    projected: dict[str, Any] = {
        "schedulable": bool(result.get("schedulable")),
        "snapshot_cid": str(snapshot.get("snapshot_cid") or ""),
        "ducklake_authoritative": False,
    }
    for name in CASF_PROJECTED_COMPONENTS:
        record = components.get(name) or {}
        if not isinstance(record, Mapping):
            raise WorldSnapshotAdmissionError(f"projected component {name} is malformed")
        projected[name] = str(record.get("cid") or "")
        projected[f"{name}_status"] = str(record.get("status") or "")
    return MappingProxyType(projected)


def refuse_ducklake_world_authority(receipt: Mapping[str, Any] | None) -> None:
    """Fail closed if a DuckLake receipt tries to mint world-snapshot authority."""

    observed = observe_ducklake_projection(receipt)
    if observed.get("authoritative") is True:
        raise WorldSnapshotAdmissionError("DuckLake cannot admit a world snapshot")
    if receipt is not None and receipt.get("authoritative") is True:
        raise WorldSnapshotAdmissionError("DuckLake cannot admit a world snapshot")
